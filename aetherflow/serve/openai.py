"""OpenAI 兼容的 HTTP 推理接口，只用标准库。

**为什么是兼容子集**：`logprobs`、`n>1`、函数调用、`response_format` 都要求
模型或采样面提供框架没有的能力。与其返回一个 200、把参数悄悄忽略，不如显式
400——调用方需要知道"这条参数没有被执行"，而不是在线上才从输出里看出差别。

支持的是 ``/v1/completions``、``/v1/chat/completions``、``/v1/models`` 与
``stream=true`` 的 SSE。

**分层**：本模块只管协议与并发；分词交给 ``ByteTokenizer``、模板交给
``ChatTemplate``、调度交给 ``ServingEngine``。三者都可以注入，所以换成真实
tokenizer 或别的调度器时，这里一行都不用改。

**流式的语义**：``ServingEngine.step`` 只在请求完成时返回，所以 SSE 每轮读
一次 :meth:`ServingEngine.progress` 的快照，把新增文本推给客户端。

**stop 参与调度**：驱动线程在每个解码步之前检查在跑请求的最长可解码前缀，
命中 ``stop`` 就 :meth:`ServingEngine.cancel` 它——腾出的槽位本步就能准入
新请求，而不是生成到 ``max_tokens`` 再丢弃。已生成的 token 仍作为这条请求
的结果交给调用方，``finish_reason`` 因此是 ``"stop"``。检测粒度是 token：
跨 token 的 ``stop``（例如 "llo" 落在两个 token 里）会多生成到完整命中为止。

**认证与限流**：``api_key`` 非 ``None`` 时，``/v1/*`` 必须带
``Authorization: Bearer <key>``（``/healthz`` 免认证，否则探活也要配密钥）。
``rate_limit`` 是每个 ``rate_window`` 秒允许的请求数，超限返回 429 并带
``Retry-After``。两者都在解析请求体、入队之前判定：被拒绝的请求不占解码槽位，
也不该在队列里排队等一个注定失败的响应。

**并发模型**：一个后台线程驱动 ``step``；每个 HTTP 请求一个线程等待自己的
事件。``capacity`` 决定真正的解码并发度，HTTP 线程数不影响它。
"""

from __future__ import annotations

import hmac
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..models import ByteTokenizer, ChatTemplate, QueueFullError

_DEFAULT_MAX_TOKENS = 16
_MAX_STOP = 4


class RequestTimeout(RuntimeError):
    """请求在时限内没有完成（排队 + 生成一起算）。"""


class _BadRequest(ValueError):
    """参数不合法：映射到 400，消息直接回给调用方。"""


class _Job:
    """一条在途请求：完成事件、结果 token、是否超时/命中 stop。"""

    __slots__ = ("rid", "event", "tokens", "timed_out", "stop_hit", "stops", "error")

    def __init__(self, rid: int, stops: Sequence[str] = ()) -> None:
        self.rid = rid
        self.event = threading.Event()
        self.tokens: List[int] = []
        self.timed_out = False
        self.stop_hit = False
        # 文本停止条件由驱动线程在解码步之间检查（见 ``_stop_at_boundary``）。
        self.stops: tuple = tuple(stops)
        self.error: Optional[BaseException] = None


class OpenAIService:
    """把 ``ServingEngine`` 包装成可并发提交、可流式取值的服务。"""

    def __init__(
        self,
        engine: Any,
        *,
        tokenizer: Optional[Any] = None,
        template: Optional[Any] = None,
        model: str = "aetherflow",
        poll: float = 0.005,
        timeout: Optional[float] = None,
        api_key: Optional[str] = None,
        rate_limit: Optional[int] = None,
        rate_window: float = 60.0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.engine = engine
        self.tokenizer = tokenizer if tokenizer is not None else ByteTokenizer()
        self.template = template if template is not None else ChatTemplate()
        if not isinstance(model, str) or not model:
            raise ValueError(f"model 必须是非空字符串，得到 {model!r}")
        if not np.isfinite(poll) or poll <= 0.0:
            raise ValueError(f"poll 必须是有限正数，得到 {poll!r}")
        if timeout is not None and (not np.isfinite(timeout) or timeout <= 0.0):
            raise ValueError(f"timeout 必须是有限正数或 None，得到 {timeout!r}")
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise ValueError(f"api_key 必须是非空字符串或 None，得到 {api_key!r}")
        if rate_limit is not None:
            if isinstance(rate_limit, bool) or not isinstance(rate_limit, int):
                raise TypeError(f"rate_limit 必须是 int 或 None，得到 {rate_limit!r}")
            if rate_limit < 1:
                raise ValueError(f"rate_limit 必须 >= 1，得到 {rate_limit}")
        if not np.isfinite(rate_window) or rate_window <= 0.0:
            raise ValueError(f"rate_window 必须是有限正数，得到 {rate_window!r}")
        self.model = model
        self.poll = float(poll)
        self.timeout = timeout
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.rate_window = float(rate_window)
        self._clock = time.monotonic if clock is None else clock
        # 滑动窗口里保留的请求时间戳（只增删，不清空：窗口外的先进先出）。
        self._rate_hits: List[float] = []
        self._jobs: Dict[int, _Job] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="aetherflow-decode", daemon=True
        )
        self._thread.start()

    # -- 准入 ----------------------------------------------------------

    def admit_request(self) -> Optional[float]:
        """记一次请求配额；超限返回建议重试的秒数，准入则返回 ``None``。

        滑动窗口而不是固定窗口：固定窗口在边界上允许两倍突发（前一窗口末尾
        加后一窗口开头），而"每分钟 60 次"对调用方来说不该有这种缝隙。
        每次进入生成接口（``POST /v1/*``）都消耗一次配额，包括随后会 400 的
        请求——否则只要参数写错就能无限尝试。

        时钟可注入（默认 ``time.monotonic``），测试用假时钟逐位确定。
        """
        if self.rate_limit is None:
            return None
        now = self._clock()
        with self._lock:
            hits = self._rate_hits
            while hits and now - hits[0] >= self.rate_window:
                hits.pop(0)
            if len(hits) >= self.rate_limit:
                return self.rate_window - (now - hits[0])
            hits.append(now)
        return None

    # -- 生命周期 ------------------------------------------------------

    def close(self) -> None:
        self._closed = True
        self._thread.join(timeout=1.0)

    def __enter__(self) -> "OpenAIService":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- 提交与等待 ----------------------------------------------------

    def submit_tokens(
        self,
        prompt: Sequence[int],
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
        stops: Sequence[str] = (),
    ) -> _Job:
        """按 token 提交一条请求，立即返回句柄。

        ``stops`` 是文本停止条件（不是 token id）：驱动线程在每个解码步之前
        检查生成文本，命中就取消这条请求并把它当作已完成（见
        :meth:`_stop_at_boundary`）。空字符串会让"命中位置 0"永远成立，
        直接拒绝。
        """
        stops = tuple(stops)
        for stop in stops:
            if not isinstance(stop, str) or not stop:
                raise ValueError(f"stop 必须是非空字符串，得到 {stop!r}")
        deadline = None if self.timeout is None else self.timeout
        with self._lock:
            rid = self.engine.submit(
                [int(t) for t in prompt],
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                eos_id=self.tokenizer.eos_id,
                seed=int(seed),
                timeout=deadline,
            )
            job = _Job(rid, stops)
            # 登记与入队在同一个临界区里：``max_tokens=0`` 的请求会直接进
            # 就绪队列，驱动线程下一次 ``step`` 就能取到它；如果那时还没
            # 登记，结果就会丢。
            self._jobs[rid] = job
        return job

    def submit_text(self, text: str, **params: Any) -> _Job:
        return self.submit_tokens(self.tokenizer.encode(text), **params)

    def submit_chat(self, messages: Sequence[Mapping[str, str]], **params: Any) -> _Job:
        if not isinstance(messages, Sequence) or not messages:
            raise _BadRequest("messages 必须是非空数组")
        turns = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise _BadRequest("messages 的每一项必须是对象")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise _BadRequest("messages 的每一项必须有字符串 role 与 content")
            turns.append((role, content))
        try:
            text = self.template.render(turns)
        except ValueError as exc:
            raise _BadRequest(str(exc)) from exc
        return self.submit_text(text, **params)

    def wait(self, job: _Job, timeout: Optional[float] = None) -> List[int]:
        """等一条请求完成；超时或作业失败都抛异常。"""
        if not job.event.wait(timeout):
            self.engine.cancel(job.rid)
            with self._lock:
                self._jobs.pop(job.rid, None)
            raise RequestTimeout(
                f"请求 {job.rid} 在 {timeout}s 内未完成（排队与生成一共）"
            )
        if job.error is not None:
            raise RuntimeError(f"请求 {job.rid} 失败：{job.error}") from job.error
        if job.timed_out:
            raise RequestTimeout(
                f"请求 {job.rid} 超时，已生成 {len(job.tokens)} 个 token"
            )
        return list(job.tokens)

    # -- 文本 ----------------------------------------------------------

    def safe_decode(self, tokens: Sequence[int]) -> Optional[str]:
        """字节分词器可能把一个多字节字符切在 token 边界上；解不出就先不发。

        这不是错误：下一轮多一个 token 就能解出完整字符。返回 ``None`` 让
        调用方保持上一次已发送的文本不变。
        """
        try:
            return self.tokenizer.decode(list(tokens))
        except ValueError:
            return None

    def text_of(self, tokens: Sequence[int]) -> str:
        """请求结束时给出的文本：**最长的可解码前缀**。

        模型输出的一串 id 不保证是合法 UTF-8（它没见过"字符"这个概念）。
        这时返回一段乱码或直接断掉连接都不如"把能表示的部分表示出来"：
        从尾部逐 token 回退直到可解码。对流式来说这也正是它一直在做的事——
        唯一区别是流式还能等下一轮补齐，而结束时没有下一轮了。
        """
        tokens = list(tokens)
        for end in range(len(tokens), -1, -1):
            text = self.safe_decode(tokens[:end])
            if text is not None:
                return text
        return ""

    # -- 驱动线程 ------------------------------------------------------

    def _loop(self) -> None:
        while not self._closed:
            try:
                # 先检查 stop 再推进：命中之后这个槽位本轮就会被腾出来，不会
                # 白生成一个 token，也不会拖到 max_tokens。
                hits = self._stop_at_boundary()
                done = self.engine.step()
                expired = self.engine.take_timed_out()
            except BaseException as exc:  # noqa: BLE001 - 线程不能静默死掉
                self._fail_all(exc)
                return
            with self._lock:
                for rid, tokens in hits:
                    self._finish(rid, tokens, False, stop_hit=True)
                for rid, tokens in done:
                    self._finish(rid, tokens, False)
                for rid, tokens in expired:
                    self._finish(rid, tokens, True)
            if not hits and not done and not expired:
                time.sleep(self.poll)

    def _stop_at_boundary(self) -> List[Tuple[int, List[int]]]:
        """把已命中文本 stop 的在跑请求摘出来，返回 ``(rid, tokens)``。

        调 :meth:`ServingEngine.cancel` 是这里的重点：只在文本层截断的话，
        引擎会占着槽位生成到 ``max_tokens``，后面的请求全在队列里等一份
        没人要的输出。取消之后结果仍然要给调用方——它是正常完成，只是
        提前了，所以走 ``stop_hit`` 而不是超时通道。

        文本用 :meth:`text_of` 求"最长可解码前缀"：不完整的 UTF-8 尾巴不
        参与匹配，下一轮补齐后自然命中。
        """
        with self._lock:
            if not any(job.stops for job in self._jobs.values()):
                return []
        hits = []
        with self._lock:
            for rid, tokens in self.engine.progress():
                job = self._jobs.get(rid)
                if job is None or not job.stops:
                    continue
                text = self.text_of(tokens)
                if _truncate(text, job.stops) != text:
                    hits.append((rid, list(tokens)))
            for rid, _ in hits:
                self.engine.cancel(rid)
        return hits

    def _finish(
        self,
        rid: int,
        tokens: Sequence[int],
        timed_out: bool,
        *,
        stop_hit: bool = False,
    ) -> None:
        job = self._jobs.pop(rid, None)
        if job is None:
            return
        job.tokens = list(tokens)
        job.timed_out = timed_out
        job.stop_hit = stop_hit
        job.event.set()

    def _fail_all(self, exc: BaseException) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
            self._jobs.clear()
        for job in jobs:
            job.error = exc
            job.event.set()


def _stops_of(payload: Mapping[str, Any]) -> List[str]:
    value = payload.get("stop")
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and value and all(isinstance(s, str) for s in value):
        values = list(value)
    else:
        raise _BadRequest("stop 必须是字符串或字符串数组")
    if len(values) > _MAX_STOP:
        raise _BadRequest(f"stop 最多 {_MAX_STOP} 条")
    # 空字符串在 ``str.find`` 里永远命中位置 0：客户端会得到一条空回复，
    # 而且请求会立刻被取消。这种"看着像参数、实际让模型闭嘴"的输入直接拒绝。
    if any(not stop for stop in values):
        raise _BadRequest("stop 不能是空字符串")
    return values


def _truncate(text: str, stops: Sequence[str]) -> str:
    cut = len(text)
    for stop in stops:
        at = text.find(stop)
        if at != -1:
            cut = min(cut, at)
    return text[:cut]


def _sendable(text: str, stops: Sequence[str]) -> str:
    """流式里可以安全发出的前缀：压住"可能是 stop 开头"的尾巴。

    逐 token 推送时，``text`` 的前缀已经发给客户端了。如果当前文本的尾部
    恰好是某个 stop 的开头（例如 stop 是 "uq"、文本刚生成到 "u"），这一帧
    发出去就再也收不回来——下一帧命中 stop 只能"少发"，不能"撤回"。所以
    把这段尾巴留到下一轮：命中就永远不发，没命中就随下一帧一起发。

    只扫尾部 ``len(stop) - 1`` 个字符，代价与生成长度无关。
    """
    hold = 0
    for stop in stops:
        limit = min(len(stop) - 1, len(text))
        for size in range(1, limit + 1):
            if text[-size:] == stop[:size]:
                hold = max(hold, size)
    return text if hold == 0 else text[:-hold]


def _params(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """校验并抽取采样参数；不支持的一律显式拒绝。"""
    for key in ("logprobs", "logit_bias", "best_of", "response_format", "tools"):
        if payload.get(key) is not None:
            raise _BadRequest(f"{key} 未支持")
    if payload.get("n", 1) != 1:
        raise _BadRequest("n > 1 未支持")
    for key in ("presence_penalty", "frequency_penalty"):
        if payload.get(key, 0):
            raise _BadRequest(f"{key} 未支持（非 0）")
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
    if max_tokens is None:
        max_tokens = _DEFAULT_MAX_TOKENS
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
        raise _BadRequest("max_tokens 必须是 >= 0 的整数")
    temperature = payload.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or not np.isfinite(temperature) or temperature < 0:
        raise _BadRequest("temperature 必须是 >= 0 的有限数")
    top_p = payload.get("top_p", 1.0)
    if not isinstance(top_p, (int, float)) or not 0.0 < float(top_p) <= 1.0:
        raise _BadRequest("top_p 必须落在 (0, 1]")
    seed = payload.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise _BadRequest("seed 必须是整数")
    return {
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "stops": _stops_of(payload),
    }


def _prompt_tokens(payload: Mapping[str, Any], tokenizer: Any) -> List[int]:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        ids = tokenizer.encode(prompt)
    elif isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], str):
        ids = tokenizer.encode(prompt[0])
    elif isinstance(prompt, list) and all(
        isinstance(t, int) and not isinstance(t, bool) for t in prompt
    ):
        ids = [int(t) for t in prompt]
    else:
        raise _BadRequest("prompt 必须是字符串或整数 token 数组")
    if not ids:
        raise _BadRequest("prompt 不能为空")
    if min(ids) < 0 or max(ids) >= tokenizer.vocab_size:
        raise _BadRequest(f"prompt 的 token 必须落在 [0, {tokenizer.vocab_size})")
    return ids


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    service: OpenAIService = None  # 由 create_server 绑定

    # BaseHTTPRequestHandler 默认往 stderr 打日志；服务不应污染调用方输出。
    def log_message(self, *args: Any) -> None:
        pass

    # -- 路由 ----------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == "/healthz":
            # 探活不能要求密钥：负载均衡器与容器编排不持有业务凭证。
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
            if not self._authenticated():
                self._unauthorized()
                return
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.service.model,
                            "object": "model",
                            "created": 0,
                            "owned_by": "aetherflow",
                        }
                    ],
                },
            )
        else:
            self._send_json(404, _error(f"未知路径 {self.path}"))

    def do_POST(self) -> None:
        if self.path not in ("/v1/completions", "/v1/chat/completions"):
            self._send_json(404, _error(f"未知路径 {self.path}"))
            return
        # 认证与限流在解析请求体之前：被拒绝的请求不该占用解码槽位，也不该
        # 让服务先把 8 MiB 的 body 读进内存再告诉对方没权限。
        if not self._authenticated():
            self._unauthorized()
            return
        wait = self.service.admit_request()
        if wait is not None:
            self._rate_limited(wait)
            return
        try:
            payload = self._read_json()
            params = _params(payload)
            stops = params.pop("stops")
            if self.path.endswith("chat/completions"):
                prompt = self._chat_prompt(payload)
            else:
                prompt = _prompt_tokens(payload, self.service.tokenizer)
            job = self.service.submit_tokens(prompt, stops=stops, **params)
        except _BadRequest as exc:
            self._send_json(400, _error(str(exc)))
            return
        except QueueFullError as exc:
            self._send_json(429, _error(str(exc), kind="rate_limit_error"))
            return
        except RequestTimeout as exc:
            self._send_json(504, _error(str(exc), kind="timeout_error"))
            return
        except Exception as exc:  # noqa: BLE001 - 协议层兜底
            self._send_json(500, _error(f"{type(exc).__name__}: {exc}"))
            return

        if payload.get("stream"):
            self._stream(job, prompt, params, stops)
        else:
            self._complete(job, prompt, params, stops)

    # -- 准入 ----------------------------------------------------------

    def _authenticated(self) -> bool:
        expected = self.service.api_key
        if expected is None:
            return True
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        # 定长比较：``==`` 会在第一个不同字节处提前返回，把密钥长度与公共
        # 前缀泄漏给计时攻击者。
        return hmac.compare_digest(header[len(prefix):].strip(), expected)

    def _unauthorized(self) -> None:
        self._send_json(
            401,
            _error(
                "缺少或无效的 API key（Authorization: Bearer <key>）",
                kind="authentication_error",
            ),
            headers={"WWW-Authenticate": 'Bearer realm="aetherflow"'},
        )

    def _rate_limited(self, wait: float) -> None:
        limit = self.service.rate_limit
        window = self.service.rate_window
        self._send_json(
            429,
            _error(
                f"超出限额（每 {window:g}s {limit} 次），请 {wait:.1f}s 后重试",
                kind="rate_limit_error",
            ),
            headers={"Retry-After": str(max(1, math.ceil(wait)))},
        )

    # -- 响应 ----------------------------------------------------------

    def _chat_prompt(self, payload: Mapping[str, Any]) -> List[int]:
        messages = payload.get("messages")
        if messages is None:
            raise _BadRequest("chat/completions 需要 messages")
        turns = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise _BadRequest("messages 的每一项必须是对象")
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise _BadRequest("messages 的每一项必须有字符串 role 与 content")
            turns.append((role, content))
        try:
            text = self.service.template.render(turns)
        except ValueError as exc:
            raise _BadRequest(str(exc)) from exc
        ids = self.service.tokenizer.encode(text)
        if not ids:
            raise _BadRequest("messages 渲染后为空")
        return ids

    def _complete(self, job: _Job, prompt: List[int], params: Mapping[str, Any],
                  stops: Sequence[str]) -> None:
        try:
            tokens = self.service.wait(job, self.service.timeout)
        except RequestTimeout as exc:
            self._send_json(504, _error(str(exc), kind="timeout_error"))
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, _error(f"{type(exc).__name__}: {exc}"))
            return
        text = self.service.text_of(tokens)
        body = _completion_body(
            self.service.model, job, prompt, tokens, text,
            params["max_tokens"], stops,
        )
        self._send_json(200, body)

    def _stream(self, job: _Job, prompt: List[int], params: Mapping[str, Any],
                stops: Sequence[str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self._stream_chunks(job, prompt, params, stops)
        except (BrokenPipeError, ConnectionResetError):
            # 客户端断开是流式的常态，不是服务端错误；后台的请求也一并取消，
            # 否则它会继续占着槽位生成一份没人要的结果。
            self.service.engine.cancel(job.rid)

    def _stream_chunks(self, job: _Job, prompt: List[int],
                       params: Mapping[str, Any], stops: Sequence[str]) -> None:
        sent = ""
        while not job.event.wait(self.service.poll):
            tokens = _progress_of(self.service.engine, job.rid)
            if tokens is None:
                continue
            text = self.service.safe_decode(tokens)
            if text is None:
                # 多字节字符被切在 token 边界上：先不发，下一轮补齐再发。
                continue
            text = _truncate(text, stops)
            # 尾部可能是 stop 的开头时先压住，见 ``_sendable``。
            text = _sendable(text, stops)
            if len(text) > len(sent):
                self._sse(_chunk(self.service.model, job.rid, text[len(sent):]))
                sent = text
        if job.error is not None:
            self._sse({"error": {"message": str(job.error),
                                  "type": "server_error"}})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        if not job.timed_out:
            # 结束前用完整 token 再解一次：最后几个 token 可能没来得及进快照。
            final = _truncate(self.service.text_of(job.tokens), stops)
            if len(final) > len(sent):
                self._sse(_chunk(self.service.model, job.rid,
                                 final[len(sent):]))
        reason = (
            "length"
            if not job.stop_hit and len(job.tokens) >= params["max_tokens"]
            else "stop"
        )
        self._sse(
            {
                **_chunk_id(self.service.model, job.rid),
                "object": "text_completion",
                "choices": [{"index": 0, "text": "", "finish_reason": reason}],
                "usage": _usage(prompt, job.tokens),
            }
        )
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _sse(self, payload: Mapping[str, Any]) -> None:
        line = "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def _send_json(
        self,
        status: int,
        payload: Mapping[str, Any],
        *,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    # -- 输入 ----------------------------------------------------------

    def _read_json(self) -> Mapping[str, Any]:
        length = self.headers.get("Content-Length")
        if length is None:
            raise _BadRequest("缺少 Content-Length")
        try:
            size = int(length)
        except ValueError as exc:
            raise _BadRequest("Content-Length 不是整数") from exc
        if size < 0 or size > 8 << 20:
            raise _BadRequest("请求体过大")
        raw = self.rfile.read(size)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise _BadRequest(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise _BadRequest("请求体必须是 JSON 对象")
        return payload


def _progress_of(engine: Any, rid: int) -> Optional[List[int]]:
    for slot_rid, tokens in engine.progress():
        if slot_rid == rid:
            return tokens
    return None


def _chunk_id(model: str, rid: int) -> Dict[str, Any]:
    return {"id": f"cmpl-{rid}", "created": int(time.time()), "model": model}


def _chunk(model: str, rid: int, text: str) -> Dict[str, Any]:
    """流式增量帧：只带本次新增的文本，``finish_reason`` 留空。"""
    return {
        **_chunk_id(model, rid),
        "object": "text_completion",
        "choices": [{"index": 0, "text": text, "finish_reason": None}],
    }


def _usage(prompt: Sequence[int], tokens: Sequence[int]) -> Dict[str, int]:
    return {
        "prompt_tokens": len(prompt),
        "completion_tokens": len(tokens),
        "total_tokens": len(prompt) + len(tokens),
    }


def _completion_body(model: str, job: _Job, prompt: Sequence[int],
                     tokens: Sequence[int], text: str, max_tokens: int,
                     stops: Sequence[str]) -> Dict[str, Any]:
    cut = _truncate(text, stops)
    # 命中 stop 时 token 数通常小于 max_tokens，但也有恰好相等的情况；
    # 以 ``stop_hit`` 为准，否则客户端会把"提前停下"读成"被截断"。
    reason = "length" if not job.stop_hit and len(tokens) >= max_tokens else "stop"
    return {
        **_chunk_id(model, job.rid),
        "object": "text_completion",
        "choices": [{"index": 0, "text": cut, "finish_reason": reason,
                     "logprobs": None}],
        "usage": _usage(prompt, tokens),
    }


def _error(message: str, kind: str = "invalid_request_error") -> Dict[str, Any]:
    return {"error": {"message": message, "type": kind}}


def create_server(
    service: OpenAIService, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """建一个 HTTP 服务（不阻塞）；``port=0`` 由系统分配，取 ``server_port``。"""
    handler = type("_BoundHandler", (_Handler,), {"service": service})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(
    service: OpenAIService, host: str = "127.0.0.1", port: int = 8000
) -> None:
    """阻塞运行，直到 ``KeyboardInterrupt``。"""
    server = create_server(service, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()
