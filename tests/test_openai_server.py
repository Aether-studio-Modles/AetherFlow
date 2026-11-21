"""OpenAI 兼容层的判据：协议形状、流式增量、以及"不支持就 400"。

这一层最容易假装成功：返回 200、把不认识的参数悄悄丢掉，客户端会以为
``logprobs`` 生效了。所以每条拒绝都用一个可执行的 400 来钉住。

流式的判据不是"有 SSE 帧"，而是**增量的拼接等于非流式的完整文本**——否则
客户端看到的就是另一段话。期望值由一个独立的 ``ServingEngine``（capacity=1）
给出，证明 HTTP 层没有改变调度结果。
"""

import json
import threading
import unittest
import http.client

import numpy as np

from aetherflow import ReferenceBackend
from aetherflow.models import (
    ByteTokenizer,
    ChatTemplate,
    QueueFullError,
    ServingEngine,
)
from aetherflow.serve import OpenAIService, create_server

BACKEND = ReferenceBackend()

# 小写字母的 token id：``ByteTokenizer`` 的字节从 2 开始。
_LETTERS = [2 + code for code in range(ord("a"), ord("z") + 1)]


class _AsciiBackend:
    """把 logits 限制在小写字母上。

    随机初始化的模型会吐出任意字节，UTF-8 经常是不完整的——那会让"文本层"
    的判据退化成"能不能解码"。限制成 ASCII 之后，流式拼接与非流式文本才有
    可比的内容；"解不出来时怎么办"另有一条专门的用例。
    """

    def __init__(self, inner):
        self.inner = inner

    def execute(self, graph, feed):
        out = dict(self.inner.execute(graph, feed))
        logits = out.get("logits")
        if logits is not None:
            mask = np.full(logits.shape[-1], -1e9, dtype=logits.dtype)
            mask[_LETTERS] = 0.0
            out["logits"] = logits + mask
        return out


def _spec():
    return ByteTokenizer().spec(
        d_model=8, num_layers=1, d_state=4, d_conv=3, expand=2,
        steps=8, batch=1,
    )


class _HttpCase(unittest.TestCase):
    """起一个真 HTTP 服务；每个用例自己决定背后的引擎是什么。"""

    def start(self, service):
        self.service = service
        self.server = create_server(service, port=0)
        self.port = self.server.server_port
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop)

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()
        self.service.close()
        engine = getattr(self, "engine", None)
        if engine is not None:
            engine.close()

    def request(self, method, path, payload=None, headers=None, *, with_headers=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        all_headers = {} if body is None else {"Content-Type": "application/json"}
        all_headers.update(headers or {})
        try:
            conn.request(method, path, body=body, headers=all_headers)
            response = conn.getresponse()
            raw = response.read()
            parsed = json.loads(raw) if raw else None
            if with_headers:
                return response.status, parsed, dict(response.getheaders())
            return response.status, parsed
        finally:
            conn.close()

    def stream(self, payload):
        """读完整条 SSE，返回 ``(status, frames, done)``；frames 不含 [DONE]。"""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            conn.request(
                "POST",
                "/v1/completions",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            frames, done = [], False
            while True:
                line = response.readline()
                if not line:
                    break
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                self.assertTrue(line.startswith("data: "), line)
                data = line[len("data: "):]
                if data == "[DONE]":
                    done = True
                    break
                frames.append(json.loads(data))
            return response.status, frames, done
        finally:
            conn.close()


class TestProtocol(_HttpCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.template = ChatTemplate()
        self.spec = _spec()
        self.params = self.spec.init_params(seed=7)
        self.backend = _AsciiBackend(BACKEND)
        self.engine = ServingEngine(
            self.backend, self.params, self.spec, capacity=2
        )
        self.start(
            OpenAIService(
                self.engine,
                tokenizer=self.tok,
                template=self.template,
                model="aetherflow-test",
                poll=0.002,
            )
        )

    def _probe(self, prompt, max_tokens=4):
        """独立的调度器给出期望 token：与 HTTP 层无关的同一份语义。"""
        engine = ServingEngine(self.backend, self.params, self.spec, capacity=1)
        try:
            rid = engine.submit(
                prompt,
                max_new_tokens=max_tokens,
                temperature=0.0,
                eos_id=self.tok.eos_id,
            )
            return dict(engine.run())[rid]
        finally:
            engine.close()

    def _text(self, tokens):
        """测试侧独立算"最长可解码前缀"，不复用被测代码。"""
        tokens = list(tokens)
        for end in range(len(tokens), -1, -1):
            try:
                return self.tok.decode(tokens[:end])
            except ValueError:
                continue
        return ""

    def test_health_and_models(self):
        status, payload = self.request("GET", "/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})

        status, payload = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "list")
        self.assertEqual(payload["data"][0]["id"], "aetherflow-test")
        self.assertEqual(payload["data"][0]["object"], "model")

    def test_completion_matches_the_scheduler(self):
        prompt = self.tok.encode("hello")
        want = self._text(self._probe(prompt))
        status, payload = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "hello", "max_tokens": 4, "temperature": 0.0},
        )
        self.assertEqual(status, 200, payload)
        choice = payload["choices"][0]
        self.assertEqual(choice["text"], want)
        self.assertEqual(choice["index"], 0)
        self.assertIn(choice["finish_reason"], ("stop", "length"))
        self.assertEqual(payload["usage"]["prompt_tokens"], len(prompt))
        self.assertEqual(payload["model"], "aetherflow-test")

    def test_token_array_prompt_is_accepted(self):
        ids = self.tok.encode("hi")
        want = self._text(self._probe(ids))
        status, payload = self.request(
            "POST", "/v1/completions", {"prompt": ids, "max_tokens": 4}
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["choices"][0]["text"], want)

    def test_chat_uses_the_template(self):
        messages = [{"role": "user", "content": "hi"}]
        text = self.template.render([("user", "hi")])
        want = self._text(self._probe(self.tok.encode(text)))
        status, payload = self.request(
            "POST",
            "/v1/chat/completions",
            {"messages": messages, "max_tokens": 4},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["choices"][0]["text"], want)

    def test_stream_reassembles_to_the_non_stream_text(self):
        status, plain = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 4}
        )
        self.assertEqual(status, 200)
        status, frames, done = self.stream(
            {"prompt": "hi", "max_tokens": 4, "stream": True}
        )
        self.assertEqual(status, 200)
        self.assertTrue(done)
        self.assertGreaterEqual(len(frames), 2)
        pieces = [frame["choices"][0]["text"] for frame in frames]
        self.assertEqual("".join(pieces), plain["choices"][0]["text"])
        last = frames[-1]
        self.assertIn(last["choices"][0]["finish_reason"], ("stop", "length"))
        self.assertEqual(last["choices"][0]["text"], "")
        self.assertEqual(
            last["usage"]["completion_tokens"],
            plain["usage"]["completion_tokens"],
        )
        for frame in frames[:-1]:
            self.assertIsNone(frame["choices"][0]["finish_reason"])

    def test_stop_truncates_at_the_first_match(self):
        _, plain = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 6}
        )
        text = plain["choices"][0]["text"]
        if not text:
            self.skipTest("这条种子下没有生成文本，stop 无从截断")
        status, payload = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "hi", "max_tokens": 6, "stop": text[:1]},
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["choices"][0]["text"], "")

    def test_stop_ends_generation_early(self):
        """stop 不只是截断：命中后调度器就该腾槽，token 数必须小于上限。"""
        _, plain = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 8}
        )
        text = plain["choices"][0]["text"]
        self.assertGreaterEqual(len(text), 2, "掩码后应生成同一字母的重复串")
        status, payload = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "hi", "max_tokens": 8, "stop": text[:2]},
        )
        self.assertEqual(status, 200, payload)
        choice = payload["choices"][0]
        self.assertEqual(choice["text"], "")
        self.assertEqual(choice["finish_reason"], "stop")
        # 命中两个字符就停，不会再把 8 个 token 全生成完。
        self.assertLessEqual(payload["usage"]["completion_tokens"], 2)
        self.assertLess(payload["usage"]["completion_tokens"], 8)

    def test_stream_stop_ends_early_with_stop_reason(self):
        status, plain = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 8}
        )
        self.assertEqual(status, 200, plain)
        text = plain["choices"][0]["text"]
        self.assertGreaterEqual(len(text), 2)
        status, frames, done = self.stream(
            {"prompt": "hi", "max_tokens": 8, "stop": text[:2], "stream": True}
        )
        self.assertEqual(status, 200)
        self.assertTrue(done)
        pieces = [frame["choices"][0]["text"] for frame in frames]
        self.assertEqual("".join(pieces), "")
        last = frames[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "stop")
        self.assertLess(last["usage"]["completion_tokens"], 8)

    def test_zero_tokens_returns_an_empty_completion(self):
        status, payload = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 0}
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["choices"][0]["text"], "")
        self.assertEqual(payload["usage"]["completion_tokens"], 0)
        self.assertEqual(payload["usage"]["total_tokens"], 2)

    def test_bad_parameters_are_rejected_not_ignored(self):
        cases = [
            ({"prompt": "", "max_tokens": 2}, "prompt"),
            ({"prompt": "hi", "n": 2}, "n > 1"),
            ({"prompt": "hi", "logprobs": 1}, "logprobs"),
            ({"prompt": "hi", "logit_bias": {}}, "logit_bias"),
            ({"prompt": "hi", "best_of": 2}, "best_of"),
            (
                {"prompt": "hi", "response_format": {"type": "json_object"}},
                "response_format",
            ),
            ({"prompt": "hi", "max_tokens": -1}, "max_tokens"),
            ({"prompt": "hi", "temperature": -0.1}, "temperature"),
            ({"prompt": "hi", "top_p": 0.0}, "top_p"),
            ({"prompt": "hi", "seed": 1.5}, "seed"),
            ({"prompt": "hi", "presence_penalty": 0.5}, "presence_penalty"),
            (
                {"prompt": "hi", "frequency_penalty": 0.5},
                "frequency_penalty",
            ),
            ({"prompt": "hi", "stop": ["a", "b", "c", "d", "e"]}, "stop"),
            ({"prompt": "hi", "stop": ""}, "stop"),
            ({"prompt": "hi", "stop": ["a", ""]}, "stop"),
            ({"prompt": [999]}, "token"),
        ]
        for payload, needle in cases:
            with self.subTest(payload=payload):
                status, body = self.request(
                    "POST", "/v1/completions", payload
                )
                self.assertEqual(status, 400, body)
                self.assertEqual(body["error"]["type"], "invalid_request_error")
                self.assertIn(needle, body["error"]["message"])

    def test_bad_chat_messages_are_rejected(self):
        cases = [
            {"messages": []},
            {"messages": [{"role": "tool", "content": "x"}]},
            {"messages": [{"role": "user", "content": ""}]},
            {"messages": [{"role": "user"}]},
            {"messages": [{"role": 1, "content": "x"}]},
            {"messages": ["hi"]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                status, body = self.request(
                    "POST", "/v1/chat/completions", payload
                )
                self.assertEqual(status, 400, body)

    def test_unknown_paths_are_404(self):
        for method, path in (
            ("GET", "/v1/nope"),
            ("POST", "/v1/nope"),
            ("GET", "/v1/completions"),
        ):
            with self.subTest(method=method, path=path):
                status, body = self.request(method, path)
                self.assertEqual(status, 404, body)


class TestAuthAndRateLimit(_HttpCase):
    """认证与限流：401/429 都在解析请求体之前，且失败不占解码槽位。"""

    AUTH = {"Authorization": "Bearer sk-test"}

    def setUp(self):
        self.tok = ByteTokenizer()
        self.spec = _spec()
        self.params = self.spec.init_params(seed=7)
        self.backend = _AsciiBackend(BACKEND)
        self.engine = ServingEngine(
            self.backend, self.params, self.spec, capacity=2
        )
        self.now = 0.0
        self.start(
            OpenAIService(
                self.engine,
                tokenizer=self.tok,
                poll=0.002,
                api_key="sk-test",
                rate_limit=2,
                rate_window=10.0,
                clock=lambda: self.now,
            )
        )

    def test_wrong_or_missing_key_is_401(self):
        for headers in (
            {},
            {"Authorization": "Bearer nope"},
            {"Authorization": "sk-test"},
            {"Authorization": "Basic sk-test"},
        ):
            with self.subTest(headers=headers):
                status, body, headers_out = self.request(
                    "POST",
                    "/v1/completions",
                    {"prompt": "hi"},
                    headers=headers,
                    with_headers=True,
                )
                self.assertEqual(status, 401, body)
                self.assertEqual(body["error"]["type"], "authentication_error")
                self.assertIn("Bearer", headers_out.get("WWW-Authenticate", ""))

    def test_health_is_open_but_models_needs_the_key(self):
        self.assertEqual(self.request("GET", "/healthz")[0], 200)
        self.assertEqual(self.request("GET", "/v1/models")[0], 401)
        status, body = self.request("GET", "/v1/models", headers=self.AUTH)
        self.assertEqual(status, 200, body)

    def test_key_unlocks_completions(self):
        status, body = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "hi", "max_tokens": 2},
            headers=self.AUTH,
        )
        self.assertEqual(status, 200, body)

    def test_rate_limit_is_a_sliding_window(self):
        # 认证失败的请求不消耗配额：先打三次 401，再确认还有整整两次额度。
        for _ in range(3):
            self.assertEqual(
                self.request(
                    "POST", "/v1/completions", {"prompt": "hi"}
                )[0],
                401,
            )
        for _ in range(2):
            status, body = self.request(
                "POST",
                "/v1/completions",
                {"prompt": "hi", "max_tokens": 1},
                headers=self.AUTH,
            )
            self.assertEqual(status, 200, body)

        status, body, headers_out = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "hi", "max_tokens": 1},
            headers=self.AUTH,
            with_headers=True,
        )
        self.assertEqual(status, 429, body)
        self.assertEqual(body["error"]["type"], "rate_limit_error")
        # 第一条请求还在窗口里，建议等待时间必须落在 (0, 10] 秒。
        wait = int(headers_out["Retry-After"])
        self.assertGreaterEqual(wait, 1)
        self.assertLessEqual(wait, 10)

        # 窗口滑过去之后恢复。固定窗口会在 t=10 前后给出两倍突发，
        # 滑动窗口不会：这里把时钟拨到 10.0，最早那条恰好出窗。
        self.now = 10.0
        status, body = self.request(
            "POST",
            "/v1/completions",
            {"prompt": "hi", "max_tokens": 1},
            headers=self.AUTH,
        )
        self.assertEqual(status, 200, body)


class _RejectingEngine:
    """提交即失败：只用来钉住异常到状态码的映射。"""

    def __init__(self, exc):
        self.exc = exc

    def submit(self, *args, **kwargs):
        raise self.exc

    def step(self):
        return []

    def take_timed_out(self):
        return []

    def progress(self):
        return []

    def cancel(self, rid):
        return False


class _StuckEngine:
    """接受请求但永不完成：验证 504 而不是无限等待。"""

    def __init__(self):
        self.prompts = []

    def submit(self, prompt, **kwargs):
        self.prompts.append(list(prompt))
        return len(self.prompts) - 1

    def step(self):
        return []

    def take_timed_out(self):
        return []

    def progress(self):
        return []

    def cancel(self, rid):
        return False


class TestErrorMapping(_HttpCase):
    def test_queue_full_maps_to_429(self):
        self.start(OpenAIService(_RejectingEngine(QueueFullError("满了"))))
        status, body = self.request(
            "POST", "/v1/completions", {"prompt": "hi"}
        )
        self.assertEqual(status, 429, body)
        self.assertEqual(body["error"]["type"], "rate_limit_error")

    def test_timeout_maps_to_504(self):
        self.start(OpenAIService(_StuckEngine(), timeout=0.05, poll=0.002))
        status, body = self.request(
            "POST", "/v1/completions", {"prompt": "hi", "max_tokens": 2}
        )
        self.assertEqual(status, 504, body)
        self.assertEqual(body["error"]["type"], "timeout_error")

    def test_unexpected_failure_maps_to_500(self):
        self.start(OpenAIService(_RejectingEngine(RuntimeError("坏了"))))
        status, body = self.request(
            "POST", "/v1/completions", {"prompt": "hi"}
        )
        self.assertEqual(status, 500, body)


class TestServiceUnit(unittest.TestCase):
    """不走 HTTP 的部分：句柄等待与"半个字符先不发"。"""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.spec = _spec()
        self.params = self.spec.init_params(seed=3)
        self.engine = ServingEngine(
            BACKEND, self.params, self.spec, capacity=1
        )
        self.service = OpenAIService(
            self.engine, tokenizer=self.tok, poll=0.002
        )

    def tearDown(self):
        self.service.close()
        self.engine.close()

    def test_wait_returns_the_generated_tokens(self):
        probe = ServingEngine(BACKEND, self.params, self.spec, capacity=1)
        try:
            rid = probe.submit(
                [2, 3], max_new_tokens=3, temperature=0.0,
                eos_id=self.tok.eos_id,
            )
            want = dict(probe.run())[rid]
        finally:
            probe.close()
        job = self.service.submit_tokens([2, 3], max_tokens=3)
        self.assertEqual(self.service.wait(job, timeout=30), want)

    def test_zero_token_job_is_not_lost(self):
        """``max_tokens=0`` 直接进完成队列：登记必须早于驱动线程取走它。"""
        job = self.service.submit_tokens([2], max_tokens=0)
        self.assertEqual(self.service.wait(job, timeout=30), [])

    def test_submit_chat_rejects_empty_messages(self):
        with self.assertRaises(ValueError):
            self.service.submit_chat([])
        with self.assertRaises(ValueError):
            self.service.submit_chat([{"role": "tool", "content": "x"}])

    def test_stop_hit_ends_the_job_early(self):
        """命中 stop 后服务主动取消：job 提前完成，token 数远小于上限。

        stop 文本不能写死：模型的贪心输出取决于参数，先探测再取前两个
        token 的文本——这样断言的是"命中位置"，不是"某个特定字母"。
        """
        probe = ServingEngine(
            _AsciiBackend(BACKEND), self.params, self.spec, capacity=1
        )
        engine = ServingEngine(
            _AsciiBackend(BACKEND), self.params, self.spec, capacity=1
        )
        try:
            rid = probe.submit(
                [2, 3], max_new_tokens=8, temperature=0.0,
                eos_id=self.tok.eos_id,
            )
            full = dict(probe.run())[rid]
            self.assertGreaterEqual(len(full), 3)
            stop = self.tok.decode(full[:2])
            with OpenAIService(engine, tokenizer=self.tok, poll=0.002) as service:
                job = service.submit_tokens([2, 3], max_tokens=8, stops=(stop,))
                tokens = service.wait(job, timeout=30)
        finally:
            probe.close()
            engine.close()
        self.assertTrue(job.stop_hit)
        self.assertEqual(tokens, full[:2])
        self.assertLess(len(tokens), 8)

    def test_empty_stop_is_rejected_at_the_service_boundary(self):
        with self.assertRaises(ValueError):
            self.service.submit_tokens([2], stops=[""])
        with self.assertRaises(ValueError):
            self.service.submit_tokens([2], stops=["ok", ""])

    def test_safe_decode_holds_a_partial_utf8_character(self):
        raw = self.tok.encode("你")
        self.assertIsNone(self.service.safe_decode(raw[:2]))
        self.assertEqual(self.service.safe_decode(raw), "你")

    def test_text_of_returns_the_longest_decodable_prefix(self):
        """模型吐出的 id 不保证是合法 UTF-8：结束时给能表示的部分。"""
        tokens = self.tok.encode("ok") + self.tok.encode("你")[:2]
        with self.assertRaises(ValueError):
            self.tok.decode(tokens)
        self.assertEqual(self.service.text_of(tokens), "ok")
        self.assertEqual(self.service.text_of([]), "")


if __name__ == "__main__":
    unittest.main()
