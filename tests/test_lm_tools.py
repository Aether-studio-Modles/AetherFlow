"""工具调用的受约束输出：JSON 文法 + 执行回路。

验收分三层：

1. **文法本身**：任何合法 JSON 的每个字节都在 ``allowed()`` 里（正向闭包），
   明显非法的字节被拒（反向）；随机游走到完成后的文本必须 ``json.loads`` 成功。
2. **接进 generate**：脚本后端"每一步都想输出非法字节"，无约束时确实输出非法
   文本，有约束时被纠正成合法 JSON；走不完就报错而不是返回半截。
3. **执行回路**：受约束文本 → 解析 → 按工具签名校验 → 调用。
"""

import json
import random
import unittest

import numpy as np

from aetherflow.models.lm import LMSpec, generate
from aetherflow.models.lm_text import BYTE_OFFSET, EOS_ID, PAD_ID, ByteTokenizer
from aetherflow.models.lm_tools import (
    JsonGrammar,
    Tool,
    ToolCallGrammar,
    ToolRegistry,
    parse_tool_call,
)

_TOKENIZER = ByteTokenizer()


def _ids(text: str):
    return _TOKENIZER.encode(text)


def _feed_all(grammar, text):
    for token in _ids(text):
        grammar.feed(token)


_VALID_JSON = [
    "0",
    "-0",
    "12",
    "-12.5",
    "1e10",
    "-2.5E-3",
    "true",
    "false",
    "null",
    '""',
    '"a b"',
    '"\\n"',
    '"\\u00e9"',
    "{}",
    '{"a":1}',
    '{"a":[1,2,{"b":null}]}',
    "[]",
    "[[]]",
    "[[1],[2,3]]",
    '{"name":"add","arguments":{"a":1,"b":2}}',
]


class TestJsonGrammarAcceptsValidJson(unittest.TestCase):
    """正向闭包：合法 JSON 的每个字节都必须在当步的 allowed() 里。"""

    def test_every_byte_of_valid_json_is_allowed(self):
        for text in _VALID_JSON:
            with self.subTest(text=text):
                grammar = JsonGrammar()
                for index, token in enumerate(_ids(text)):
                    allowed = grammar.allowed()
                    self.assertIn(
                        token, allowed, f"{text!r} 的第 {index} 个字节被拒了"
                    )
                    grammar.feed(token)
                self.assertTrue(grammar.complete, f"{text!r} 没有走到完成态")
                self.assertIn(EOS_ID, grammar.allowed())
                grammar.feed(EOS_ID)
                self.assertTrue(grammar.done)

    def test_nested_depth_is_kept(self):
        text = '{"a":{"b":{"c":[1]}}}'
        grammar = JsonGrammar()
        _feed_all(grammar, text)
        self.assertTrue(grammar.complete)
        self.assertEqual(len(grammar._stack), 0)

    def test_unicode_escape_needs_four_hex_digits(self):
        grammar = JsonGrammar()
        _feed_all(grammar, '"\\u00')
        for digit in "e9":
            self.assertIn(_ids(digit)[0], grammar.allowed())
            _feed_all(grammar, digit)
        # 四位够了，回到字符串内容；再一个 hex 只是普通字符
        self.assertIn(_ids('"')[0], grammar.allowed())


class TestJsonGrammarRejectsInvalid(unittest.TestCase):
    """反向：非法字节不在集合里，喂进去直接报错。"""

    def _assert_rejected(self, prefix, bad):
        grammar = JsonGrammar()
        _feed_all(grammar, prefix)
        self.assertNotIn(_ids(bad)[0], grammar.allowed())
        with self.assertRaises(ValueError):
            grammar.feed(_ids(bad)[0])

    def test_root_states(self):
        self._assert_rejected("", "}")
        self._assert_rejected("", "]")
        self._assert_rejected("", ":")
        self._assert_rejected("", ",")
        self._assert_rejected("", " ")

    def test_object_states(self):
        self._assert_rejected("{", "1")          # key 必须是字符串
        self._assert_rejected('{"a"', "1")       # 缺冒号
        self._assert_rejected('{"a":1', ":")     # 值之后只能是 , 或 }
        self._assert_rejected('{"a":1}', ",")   # 根值已结束，只剩 EOS

    def test_array_states(self):
        self._assert_rejected("[", "}")
        self._assert_rejected("[1", "{")

    def test_string_states(self):
        self._assert_rejected('"ab', "\n")       # 未转义的控制字符
        self._assert_rejected('"ab', "\x01")
        self._assert_rejected('"ab\\', "x")      # 非法转义
        self._assert_rejected('"ab\\u12', "g")   # 非 hex

    def test_number_states(self):
        self._assert_rejected("0", "1")          # 前导零
        self._assert_rejected("1.", "e")         # 小数点后必须是数字
        self._assert_rejected("-", ".")          # 负号后必须是数字
        self._assert_rejected("1e", "e")         # 指数里必须先是数字/符号

    def test_literal_states(self):
        self._assert_rejected("t", "f")
        self._assert_rejected("nul", "n")

    def test_pad_and_premature_eos(self):
        grammar = JsonGrammar()
        with self.assertRaises(ValueError):
            grammar.feed(PAD_ID)
        with self.assertRaises(ValueError):
            grammar.feed(EOS_ID)  # 还没开始
        _feed_all(grammar, '{"a":')
        with self.assertRaises(ValueError):
            grammar.feed(EOS_ID)

    def test_feed_after_done_is_rejected(self):
        grammar = JsonGrammar()
        _feed_all(grammar, "0")
        grammar.feed(EOS_ID)
        self.assertEqual(grammar.allowed(), frozenset())
        with self.assertRaises(ValueError):
            grammar.feed(EOS_ID)


class TestJsonGrammarRandomWalk(unittest.TestCase):
    """从 allowed() 里随机取，直到完成，产物必须能被 json 解析。

    这是"文法与 json 语义一致"最直接的证据：生成器不知道 JSON 是什么，
    只按集合选字节。
    """

    def _walk(self, seed, limit=20000):
        rng = np.random.default_rng(seed)
        grammar = JsonGrammar()
        out = []
        for _ in range(limit):
            if grammar.complete:
                grammar.feed(EOS_ID)
                return bytes(out).decode("ascii")
            allowed = sorted(grammar.allowed())
            token = int(rng.choice(allowed))
            grammar.feed(token)
            out.append(token - 2)
        self.fail(f"随机游走 {limit} 步仍未完成")

    def test_walks_stay_parseable(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                text = self._walk(seed)
                json.loads(text)  # 不合法会抛 JSONDecodeError


# --------------------------------------------------------------------------
# 接进 generate
# --------------------------------------------------------------------------


class _GreedyFor:
    """每个位置都想输出同一个字节的脚本后端。"""

    def __init__(self, char, vocab_size):
        self.char = char
        self.vocab_size = vocab_size
        self.calls = 0

    def execute(self, graph, feed):
        out = {}
        for out_name, value in zip(graph.output_names, graph.outputs):
            shape = tuple(int(d) for d in value.type.shape)
            if out_name == "logits":
                buf = np.zeros(shape, dtype=np.float32)
                buf[..., _TOKENIZER.encode(self.char)[0]] = 5.0
                out["logits"] = buf
            else:
                out[out_name] = np.zeros(shape, dtype=np.float32)
        self.calls += 1
        return out


class _NestForever:
    """每一步都想再开一个数组——约束下永远不会结束。"""

    def __init__(self):
        self.calls = 0

    def execute(self, graph, feed):
        out = {}
        for out_name, value in zip(graph.output_names, graph.outputs):
            shape = tuple(int(d) for d in value.type.shape)
            if out_name == "logits":
                buf = np.zeros(shape, dtype=np.float32)
                buf[..., _TOKENIZER.encode("[")[0]] = 5.0
                out["logits"] = buf
            else:
                out[out_name] = np.zeros(shape, dtype=np.float32)
        self.calls += 1
        return out


def _spec(steps=8):
    return LMSpec(vocab_size=ByteTokenizer().vocab_size, steps=steps, batch=1)


class TestConstrainedGenerate(unittest.TestCase):
    def _decode(self, tokens):
        return _TOKENIZER.decode(tokens)

    def test_quote_lover_produces_valid_json(self):
        """一直想输出引号：无约束是 `""""`（非法），有约束是 `""`。"""
        spec = _spec()
        raw = generate(_GreedyFor('"', spec.vocab_size), {}, _ids(">"), spec, max_new_tokens=4)
        self.assertEqual(self._decode(raw), '"""' + '"')

        spec = _spec()
        grammar = JsonGrammar()
        got = generate(
            _GreedyFor('"', spec.vocab_size),
            {},
            _ids(">"),
            spec,
            max_new_tokens=8,
            constraint=grammar,
        )
        self.assertTrue(grammar.done)
        self.assertEqual(json.loads(self._decode(got)), "")

    def test_number_lover_produces_valid_json(self):
        spec = _spec()
        grammar = JsonGrammar()
        got = generate(
            _GreedyFor("0", spec.vocab_size), {}, _ids(">"), spec,
            max_new_tokens=8, constraint=grammar,
        )
        self.assertEqual(json.loads(self._decode(got)), 0)

    def test_unfinished_constraint_is_an_error(self):
        """永远不结束的约束必须报错，不能把半截 JSON 当结果返回。"""
        spec = _spec()
        with self.assertRaises(ValueError) as ctx:
            generate(
                _NestForever(), {}, _ids(">"), spec,
                max_new_tokens=6, constraint=JsonGrammar(),
            )
        self.assertIn("max_new_tokens", str(ctx.exception))

    def test_custom_stop_token_conflicts_with_constraint(self):
        spec = _spec()
        with self.assertRaises(ValueError):
            generate(
                _GreedyFor("0", spec.vocab_size), {}, _ids(">"), spec,
                max_new_tokens=4, eos_id=7, constraint=JsonGrammar(),
            )


# --------------------------------------------------------------------------
# 执行回路
# --------------------------------------------------------------------------


def _add(a, b):
    return a + b


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry([
            Tool("add", _add, {"a": int, "b": int}),
            Tool("echo", lambda text: text, {"text": str}),
        ])

    def test_end_to_end_call(self):
        self.assertEqual(self.registry.call('{"name":"add","arguments":{"a":1,"b":2}}'), 3)

    def test_names_are_sorted_and_lenable(self):
        self.assertEqual(self.registry.names, ("add", "echo"))
        self.assertEqual(len(self.registry), 2)
        self.assertIn("add", self.registry)

    def test_unknown_tool_is_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.call('{"name":"sub","arguments":{"a":1,"b":2}}')

    def test_missing_and_extra_arguments(self):
        with self.assertRaises(ValueError):
            self.registry.call('{"name":"add","arguments":{"a":1}}')
        with self.assertRaises(ValueError):
            self.registry.call('{"name":"add","arguments":{"a":1,"b":2,"c":3}}')

    def test_type_mismatch(self):
        with self.assertRaises(ValueError):
            self.registry.call('{"name":"add","arguments":{"a":1,"b":"2"}}')
        with self.assertRaises(ValueError):
            self.registry.call('{"name":"echo","arguments":{"text":3}}')

    def test_bool_is_not_int(self):
        with self.assertRaises(ValueError):
            self.registry.call('{"name":"add","arguments":{"a":true,"b":2}}')

    def test_float_accepts_int(self):
        registry = ToolRegistry([Tool("half", lambda x: x / 2, {"x": float})])
        self.assertEqual(registry.call('{"name":"half","arguments":{"x":3}}'), 1.5)

    def test_nonfinite_float_is_rejected(self):
        """审计 P1：``1e999`` 是合法 JSON，却解析成 inf；不能让它进工具。"""
        registry = ToolRegistry([Tool("half", lambda x: x / 2, {"x": float})])
        with self.assertRaisesRegex(ValueError, "应为 float"):
            registry.call('{"name":"half","arguments":{"x":1e999}}')
        with self.assertRaisesRegex(ValueError, "应为 float"):
            registry.call('{"name":"half","arguments":{"x":-1e999}}')

    def test_registry_grammar_cannot_complete_a_non_object(self):
        """审计 P1：``[]`` 是合法 JSON，但不可能被 parse_tool_call 接受。"""
        grammar = self.registry.grammar()
        allowed = grammar.allowed()
        self.assertIn(BYTE_OFFSET + ord("{"), allowed)
        self.assertNotIn(BYTE_OFFSET + ord("["), allowed)
        self.assertNotIn(BYTE_OFFSET + ord("1"), allowed)
        with self.assertRaises(ValueError):
            grammar.feed(BYTE_OFFSET + ord("["))

    def test_duplicate_registration(self):
        with self.assertRaises(ValueError):
            self.registry.register(Tool("add", _add, {"a": int, "b": int}))

    def test_overlapping_signature_is_rejected(self):
        with self.assertRaises(ValueError):
            Tool("bad", _add, {"a": int}, {"a": int})


class TestParseToolCall(unittest.TestCase):
    def test_shape(self):
        name, arguments = parse_tool_call('{"name":"add","arguments":{"a":1}}')
        self.assertEqual(name, "add")
        self.assertEqual(arguments, {"a": 1})

    def test_arguments_default_to_empty(self):
        self.assertEqual(parse_tool_call('{"name":"now"}'), ("now", {}))

    def test_invalid_payloads(self):
        for text in (
            "[]",
            "{",
            '{"arguments":{}}',
            '{"name":""}',
            '{"name":3}',
            '{"name":"a","arguments":[1]}',
            '{"name":"a","arguments":{},"extra":1}',
        ):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_tool_call(text)


class TestConstrainedOutputIsCallable(unittest.TestCase):
    """端到端：受约束解码 → 解析 → 调用。

    这里手动喂一段"模型想输出"的文本，验证文法放行、执行回路接得住。
    """

    def test_walked_json_can_drive_a_tool(self):
        text = '{"name":"add","arguments":{"a":1,"b":2}}'
        grammar = JsonGrammar()
        _feed_all(grammar, text)
        grammar.feed(EOS_ID)
        self.assertTrue(grammar.done)

        registry = ToolRegistry([Tool("add", _add, {"a": int, "b": int})])
        self.assertEqual(registry.call(text), 3)


class TestToolSchema(unittest.TestCase):
    """工具描述对齐 JSON Schema：两种写法、构造期拒绝、schema 输出可喂给模型。"""

    def test_convenience_signature_becomes_a_standard_schema(self):
        tool = Tool("add", lambda a, b, label="": None, {"a": int, "b": int}, {"label": str})
        self.assertEqual(
            tool.schema(),
            {
                "name": "add",
                "description": "",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "integer"},
                        "b": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
            },
        )

    def test_explicit_parameters_are_passed_through(self):
        parameters = {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["fast", "safe"]},
                "values": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["mode"],
            "additionalProperties": False,
        }
        tool = Tool("run", lambda mode, values=(): None, parameters=parameters)
        self.assertEqual(tool.schema()["parameters"], parameters)

    def test_mixing_the_two_spellings_is_rejected(self):
        with self.assertRaises(ValueError):
            Tool(
                "mix",
                lambda a: None,
                {"a": int},
                parameters={"type": "object", "properties": {"a": {"type": "integer"}}},
            )

    def test_required_must_reference_a_declared_property(self):
        with self.assertRaises(ValueError):
            Tool(
                "bad",
                lambda: None,
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "integer"}},
                    "required": ["b"],
                },
            )

    def test_nested_object_parameters_are_rejected_at_construction(self):
        # 受约束文法只表达标量与标量数组；在构造期拒绝，而不是等 decode 到一半。
        with self.assertRaises(ValueError):
            Tool(
                "nested",
                lambda payload: None,
                parameters={
                    "type": "object",
                    "properties": {
                        "payload": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}},
                        }
                    },
                },
            )

    def test_array_of_objects_is_rejected(self):
        with self.assertRaises(ValueError):
            Tool(
                "rows",
                lambda rows: None,
                parameters={
                    "type": "object",
                    "properties": {
                        "rows": {
                            "type": "array",
                            "items": {"type": "object", "properties": {}},
                        }
                    },
                },
            )

    def test_unmapped_python_type_is_rejected(self):
        with self.assertRaises(ValueError):
            Tool("weird", lambda items: None, {"items": list}).schema()

    def test_enum_and_array_are_checked(self):
        tool = Tool(
            "run",
            lambda mode, values: None,
            parameters={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["fast", "safe"]},
                    "values": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["mode", "values"],
            },
        )
        tool.check({"mode": "fast", "values": [1, 2.5]})
        with self.assertRaises(ValueError):
            tool.check({"mode": "slow", "values": [1]})
        with self.assertRaises(ValueError):
            tool.check({"mode": "fast", "values": [1, "x"]})
        with self.assertRaises(ValueError):
            tool.check({"mode": "fast", "values": 1})

    def test_registry_schema_is_sorted_and_complete(self):
        registry = ToolRegistry(
            [
                Tool("zeta", lambda: None),
                Tool("alpha", lambda: None),
            ]
        )
        self.assertEqual([item["name"] for item in registry.schema()], ["alpha", "zeta"])


class TestToolCallGrammar(unittest.TestCase):
    """按注册表合成的文法：生成期就把名字、键名、类型限死。"""

    @staticmethod
    def _registry():
        return ToolRegistry(
            [
                Tool("add", lambda a, b: a + b, {"a": int, "b": int}),
                Tool("echo", lambda text: text, {"text": str}),
                Tool(
                    "pick",
                    lambda items: items,
                    parameters={
                        "type": "object",
                        "properties": {
                            "items": {"type": "array", "items": {"type": "integer"}}
                        },
                        "required": ["items"],
                    },
                ),
                Tool(
                    "toggle",
                    lambda flag, label="": (flag, label),
                    {"flag": bool},
                    {"label": str},
                ),
            ]
        )

    def _walk(self, registry, text):
        grammar = registry.grammar()
        for token in _TOKENIZER.encode(text):
            self.assertIn(token, grammar.allowed(), f"{text!r} 被拒在 {chr(token - BYTE_OFFSET)!r}")
            grammar.feed(token)
        return grammar

    def test_complete_calls_round_trip(self):
        registry = self._registry()
        cases = [
            ('{"name":"add","arguments":{"a":1,"b":2}}', 3),
            ('{"name":"echo","arguments":{"text":"hi"}}', "hi"),
            ('{"name":"pick","arguments":{"items":[1,2,3]}}', [1, 2, 3]),
            ('{"name":"pick","arguments":{"items":[]}}', []),
            ('{"name":"toggle","arguments":{"flag":true}}', (True, "")),
            ('{"name":"toggle","arguments":{"flag":false,"label":"x"}}', (False, "x")),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                grammar = self._walk(registry, text)
                self.assertTrue(grammar.complete)
                grammar.feed(EOS_ID)
                self.assertTrue(grammar.done)
                self.assertEqual(registry.call(text), expected)

    def test_unknown_tool_name_cannot_be_written(self):
        registry = self._registry()
        grammar = registry.grammar()
        for token in _TOKENIZER.encode('{"name":"sub"'):
            if token not in grammar.allowed():
                return
            grammar.feed(token)
        self.fail("未注册的工具名居然被文法接受了")

    def test_extra_argument_key_cannot_be_written(self):
        registry = self._registry()
        grammar = self._walk(registry, '{"name":"add","arguments":{"a":1,')
        self.assertNotIn(_TOKENIZER.encode("c")[0], grammar.allowed())
        self.assertIn(_TOKENIZER.encode('"')[0], grammar.allowed())

    def test_integer_rejects_a_decimal_point(self):
        registry = self._registry()
        grammar = self._walk(registry, '{"name":"add","arguments":{"a":1')
        self.assertNotIn(_TOKENIZER.encode(".")[0], grammar.allowed())

    def test_required_argument_cannot_be_skipped(self):
        registry = ToolRegistry([Tool("add", lambda a, b: a + b, {"a": int, "b": int})])
        grammar = self._walk(registry, '{"name":"add","arguments":{"a":1')
        self.assertNotIn(_TOKENIZER.encode("}")[0], grammar.allowed())

    def test_optional_argument_can_be_skipped(self):
        registry = ToolRegistry(
            [Tool("toggle", lambda flag, label="": label, {"flag": bool}, {"label": str})]
        )
        grammar = registry.grammar()
        for token in _TOKENIZER.encode('{"name":"toggle","arguments":{"flag":true'):
            grammar.feed(token)
        # 必填的 flag 写完，optional 的 label 还没写：此时可以直接收尾。
        self.assertIn(_TOKENIZER.encode("}")[0], grammar.allowed())
        grammar.feed(_TOKENIZER.encode("}")[0])   # 关闭 arguments
        grammar.feed(_TOKENIZER.encode("}")[0])   # 关闭根对象
        self.assertTrue(grammar.complete)

    def test_tool_without_arguments(self):
        registry = ToolRegistry([Tool("ping", lambda: "pong")])
        self.assertEqual(registry.call('{"name":"ping","arguments":{}}'), "pong")
        grammar = self._walk(registry, '{"name":"ping","arguments":{}}')
        self.assertTrue(grammar.complete)

    def test_premature_eos_is_rejected(self):
        with self.assertRaises(ValueError):
            self._registry().grammar().feed(EOS_ID)

    def test_pad_is_rejected(self):
        with self.assertRaises(ValueError):
            self._registry().grammar().feed(PAD_ID)

    def test_empty_registry_is_rejected(self):
        with self.assertRaises(ValueError):
            ToolCallGrammar(ToolRegistry())

    def test_random_walk_stays_parseable(self):
        """从 ``allowed()`` 里随机游走到结束：生成的一定是合法调用。"""
        registry = self._registry()
        rng = random.Random(20261113)
        for _ in range(40):
            grammar = registry.grammar()
            text = bytearray()
            for _ in range(600):
                if grammar.complete:
                    break
                allowed = sorted(grammar.allowed())
                closers = [
                    token
                    for token in allowed
                    if bytes([token - BYTE_OFFSET]) in (b"}", b"]", b'"')
                ]
                # 偏向收尾：否则字符串与数组都可能一直加下去。
                token = (
                    rng.choice(closers)
                    if closers and rng.random() < 0.5
                    else rng.choice(allowed)
                )
                grammar.feed(token)
                text.append(token - BYTE_OFFSET)
            self.assertTrue(grammar.complete, text)
            grammar.feed(EOS_ID)
            registry.call(text.decode())


if __name__ == "__main__":
    unittest.main()
