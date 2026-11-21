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
一次 :meth:`ServingEngine.progress` 的快照，把新增文本推给客户端。``stop``
只在文本层截断，**不参与调度**——命中 stop 之后引擎仍会生成到
``max_tokens``，只是不再往客户端发。把它做成"提前取消"需要调度器支持文本
级停止条件，目前没有。

**并发模型**：一个后台线程驱动 ``step``；每个 HTTP 请求一个线程等待自己的
事件。``capacity`` 决定真正的解码并发度，HTTP 线程数不影响它。
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..models import ByteTokenizer, ChatTemplate, QueueFullError

_DEFAULT_MAX_TOKENS = 16
_MAX_STOP = 4


class RequestTimeout(RuntimeError):
    """请求在时限内没有完成（排队 + 生成一起算）。"""


class _BadRequest(ValueError):
    """参数不合法：映射到 400，消息直接回给调用方。"""


class _Job:
    """一条在途请求：完成事件、结果 token、是否超时。"""

    __slots__ = ("rid", "event", "tokens", "timed_out", "error")

    def __init__(self, rid: int) -> None:
        self.rid = rid
        self.event = threading.Event()
        self.tokens: List[int] = []
        self.timed_out = False
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
        self.model = model
        self.poll = float(poll)
        self.timeout = timeout
        self._jobs: Dict[int, _Job] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._loop, name="aetherflow-decode", daemon=True
        )
        self._thread.start()

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
    ) -> _Job:
        """按 token 提交一条请求，立即返回句柄。"""
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
            job = _Job(rid)
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
                done = self.engine.step()
                expired = self.engine.take_timed_out()
            except BaseException as exc:  # noqa: BLE001 - 线程不能静默死掉
                self._fail_all(exc)
                return
            with self._lock:
                for rid, tokens in done:
                    self._finish(rid, tokens, False)
                for rid, tokens in expired:
                    self._finish(rid, tokens, True)
            if not done and not expired:
                time.sleep(self.poll)

    def _finish(self, rid: int, tokens: Sequence[int], timed_out: bool) -> None:
        job = self._jobs.pop(rid, None)
        if job is None:
            return
        job.tokens = list(tokens)
        job.timed_out = timed_out
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
        return [value]
    if isinstance(value, list) and value and all(isinstance(s, str) for s in value):
        if len(value) > _MAX_STOP:
            raise _BadRequest(f"stop 最多 {_MAX_STOP} 条")
        return list(value)
    raise _BadRequest("stop 必须是字符串或字符串数组")


def _truncate(text: str, stops: Sequence[str]) -> str:
    cut = len(text)
    for stop in stops:
        at = text.find(stop)
        if at != -1:
            cut = min(cut, at)
    return text[:cut]


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
            self._send_json(200, {"status": "ok"})
        elif self.path == "/v1/models":
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
        try:
            payload = self._read_json()
            params = _params(payload)
            stops = params.pop("stops")
            if self.path.endswith("chat/completions"):
                prompt = self._chat_prompt(payload)
            else:
                prompt = _prompt_tokens(payload, self.service.tokenizer)
            job = self.service.submit_tokens(prompt, **params)
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
        reason = "length" if len(job.tokens) >= params["max_tokens"] else "stop"
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

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
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
    reason = "length" if len(tokens) >= max_tokens else "stop"
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
