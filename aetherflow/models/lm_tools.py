"""工具调用的受约束输出：字节级 JSON 文法 + 执行回路。

审计 AF-34 说得很直白："工具调用需要受约束格式输出与执行回路，不能靠多一个
算子自动获得。"这里把"受约束"落在**字节级词表**上——``ByteTokenizer`` 的每个
字节就是一个 token，于是"这一步允许哪些 token"和"这一步允许哪些字节"是同一个
集合。不需要前缀树、不需要 BPE 合并表，也不需要维护"某个 token 的字节序列"。

分三层，都不碰图：

    JsonGrammar       通用 JSON 状态机，只回答"下一个字节可以是哪些"
    ToolCallGrammar   按注册表合成：工具名枚举 + 参数键名 + 参数类型
    ToolRegistry      受约束解码出来的 JSON → 校验形状 → 调用注册的函数

为什么在主机侧做：合法的字节集合取决于**已经生成的文本**，而文本在采样之后才
存在。把它做进图里需要把"文法状态"变成张量、把转移变成算子——那是另一套数据
流图（每步都要在设备上跑转移），收益是省一次很短的同步。这里选主机侧，是因为
一次 decode 本来就是一次同步（``generate`` 每步都要回主机取 logits）。

**``JsonGrammar`` 只是格式约束，不是正确性保证**：它保证输出能 ``json.loads``，
不保证参数语义对（``{"name": "add", "arguments": {"a": "x"}}`` 是合法 JSON、非法
调用）。语义校验在 :meth:`ToolRegistry.call` 里按每个工具的 schema 做。
``ToolCallGrammar`` 往前推了一步：它从注册表合成文法，把工具名与参数键名在解码
期就限死，模型写不出未注册的工具名或多余的参数键。

**受约束解码的参数子集**：``ToolCallGrammar`` 支持标量（string / integer /
number / boolean / null）与"标量元素数组"。嵌套对象参数在构造期显式拒绝——不是
做不了，而是那需要把 schema 递归展开成动态段，收益（省掉事后校验）远小于复杂度。
拒绝时提示改用 ``JsonGrammar``（它接受任意合法 JSON，再由 ``Tool.check`` 兜底）。
"""


from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

from .lm_text import BYTE_OFFSET, EOS_ID, PAD_ID

__all__ = [
    "JsonGrammar",
    "ToolCallGrammar",
    "ToolRegistry",
    "Tool",
    "parse_tool_call",
]


# token id ↔ 字节 的双射：``BYTE_OFFSET + b``。预生成一张 256 项的表，
# 让 ``allowed()`` 只做查表，不必每步重算。
_BYTE_TOKEN: Tuple[int, ...] = tuple(BYTE_OFFSET + b for b in range(256))

_QUOTE = ord('"')
_BACKSLASH = ord("\\")


def _tokens(chars: str) -> FrozenSet[int]:
    return frozenset(_BYTE_TOKEN[ord(c)] for c in chars)


_DIGITS = _tokens("0123456789")
_HEX = _tokens("0123456789abcdefABCDEF")
_STRING_BODY = frozenset(
    _BYTE_TOKEN[b] for b in range(0x20, 0x7F) if b not in (_QUOTE, _BACKSLASH)
)
_STRING_START = _STRING_BODY | frozenset({_BYTE_TOKEN[_QUOTE], _BYTE_TOKEN[_BACKSLASH]})
_VALUE_START = _tokens('{["-0123456789tfn')
_ARRAY_NEXT = _VALUE_START | frozenset({_BYTE_TOKEN[ord("]")]})
_OBJECT_KEY_NEXT = frozenset({_BYTE_TOKEN[_QUOTE], _BYTE_TOKEN[ord("}")]})
_OBJECT_KEY_MORE = frozenset({_BYTE_TOKEN[_QUOTE]})


_NUMBER_FINAL = frozenset({"zero", "int", "frac", "exp-digits"})


def _advance_number(stage: str, ch: str, *, integer_only: bool = False) -> Optional[str]:
    """推进 JSON 数字状态；``None`` 表示这个字符不属于当前数字。

    抽成纯函数是因为有两个文法要用它：通用的 :class:`JsonGrammar` 与按 schema
    合成的 :class:`ToolCallGrammar`（后者要拒绝 ``integer`` 上的小数点与指数）。
    复制一份会立刻漂移——两处对 ``1e`` / ``1.`` 的接受面必须永远一致。

    ``integer_only=True`` 时只走 ``start → minus → zero/int``，其余转移一律
    视为"数字到此为止"，由调用方按分隔符处理。
    """
    if stage == "start":
        if ch == "-":
            return "minus"
        if ch == "0":
            return "zero"
        if ch.isdigit():
            return "int"
        return None
    if stage == "minus":
        if ch == "0":
            return "zero"
        if ch.isdigit():
            return "int"
        return None
    if stage == "zero":
        if integer_only:
            return None
        if ch == ".":
            return "dot"
        if ch in "eE":
            return "exp"
        return None
    if stage == "int":
        if ch.isdigit():
            return "int"
        if integer_only:
            return None
        if ch == ".":
            return "dot"
        if ch in "eE":
            return "exp"
        return None
    if stage == "dot":
        if ch.isdigit():
            return "frac"
        return None
    if stage == "frac":
        if ch.isdigit():
            return "frac"
        if ch in "eE":
            return "exp"
        return None
    if stage == "exp":
        if ch in "+-":
            return "exp-sign"
        if ch.isdigit():
            return "exp-digits"
        return None
    if stage == "exp-sign":
        if ch.isdigit():
            return "exp-digits"
        return None
    if stage == "exp-digits":
        if ch.isdigit():
            return "exp-digits"
        return None
    raise AssertionError(f"未知的数字阶段 {stage!r}")


@dataclass
class JsonGrammar:
    """增量式 JSON 文法：``allowed()`` 给出下一个 token 的合法集合。

    只生成**紧凑** JSON（不允许空白），因为空白对模型没有信息量，却会让
    "合法字节集合"多出一整类分支。生成结果因此总是可以 ``json.loads``。

    状态由三部分组成：

    - ``_scalars``：当前正在写的标量（字符串 / 转义 / unicode / 数字 / 字面量）；
    - ``_stack``：容器栈，每项是 ``("{", phase)`` 或 ``("[", phase)``；
    - ``complete``：根值已经写完，只剩 EOS 可选。
    """

    _stack: List[Tuple[str, str]] = field(default_factory=list, repr=False)
    _mode: str = field(default="value", repr=False)
    _number: str = field(default="start", repr=False)
    _literal: str = field(default="", repr=False)
    _unicode_left: int = field(default=4, repr=False)
    _done: bool = field(default=False, repr=False)
    # 工具调用必须是对象：``[]``、``1``、``"abc"`` 都是合法 JSON，却不可能
    # 被 ``parse_tool_call`` 接受——让它们在解码期就不可完成，比事后报错好。
    require_object: bool = False

    # -- 只读状态 ---------------------------------------------------------

    @property
    def complete(self) -> bool:
        """根值已完整；此后 ``allowed()`` 只含 EOS。

        数字的结束是**隐式**的（JSON 里没有终止符），所以"还在写数字"
        与"数字已经写完"要靠阶段判断：``12`` 写完 ``2`` 就已经完整，
        ``12.`` 不完整，``1e`` 也不完整。
        """
        if self._stack:
            return False
        if self._mode == "end":
            return True
        return self._mode == "number" and self._number in _NUMBER_FINAL

    @property
    def done(self) -> bool:
        """已经吃过 EOS。"""
        return self._done

    def reset(self) -> None:
        self._stack.clear()
        self._mode = "value"
        self._number = "start"
        self._literal = ""
        self._unicode_left = 4
        self._done = False

    # -- 允许集合 ---------------------------------------------------------

    def allowed(self) -> FrozenSet[int]:
        """下一步允许的 token id 集合。永远不含 pad。"""
        if self._done:
            return frozenset()
        if self._mode == "string":
            return _STRING_START
        if self._mode == "escape":
            return _tokens('"\\/bfnrtu')
        if self._mode == "unicode":
            return _HEX
        if self._mode == "literal":
            return frozenset({_BYTE_TOKEN[ord(self._literal[0])]})
        if self._mode == "number":
            return self._number_allowed()
        if self._mode == "end":
            return frozenset({EOS_ID})
        return self._value_allowed()

    def _value_allowed(self) -> FrozenSet[int]:
        if not self._stack:
            if self.require_object:
                return frozenset({_BYTE_TOKEN[ord("{")]})
            return _VALUE_START
        kind, phase = self._stack[-1]
        if kind == "{":
            if phase == "key":
                return _OBJECT_KEY_NEXT          # 刚打开：允许 " 或 }
            if phase == "key_more":
                return _OBJECT_KEY_MORE          # 逗号之后：只能是 "
            if phase == "colon":
                return frozenset({_BYTE_TOKEN[ord(":")]})
            if phase == "value":
                return _VALUE_START
            return self._after_value_allowed()
        if phase == "first":
            return _ARRAY_NEXT                   # 刚打开：允许值或 ]
        if phase == "value":
            return _VALUE_START                  # 逗号之后：只能是值
        return self._after_value_allowed()

    def _after_value_allowed(self) -> FrozenSet[int]:
        """一个值刚写完时允许的字节：容器里是分隔符 / 关闭，根上是 EOS。"""
        if not self._stack:
            return frozenset({EOS_ID})
        kind, _ = self._stack[-1]
        closing = ord("}" if kind == "{" else "]")
        return frozenset({_BYTE_TOKEN[ord(",")], _BYTE_TOKEN[closing]})

    def _number_allowed(self) -> FrozenSet[int]:
        stage = self._number
        if stage == "start":
            return _tokens("-0123456789")
        if stage == "minus":
            return _tokens("0123456789")
        if stage == "zero":
            return _tokens(".eE") | self._after_value_allowed()
        if stage == "int":
            return _DIGITS | _tokens(".eE") | self._after_value_allowed()
        if stage in ("dot", "exp-sign"):
            return _DIGITS
        if stage == "frac":
            return _DIGITS | _tokens("eE") | self._after_value_allowed()
        if stage == "exp":
            return _DIGITS | _tokens("+-")
        if stage == "exp-digits":
            return _DIGITS | self._after_value_allowed()
        raise AssertionError(f"未知的数字阶段 {stage!r}")

    # -- 推进 -------------------------------------------------------------

    def feed(self, token: int) -> None:
        """吃掉一个 token。越界或非法都抛 ``ValueError``。

        非法输入不是"静默忽略"：约束解码的调用方必须只喂 ``allowed()`` 里的
        token，否则说明采样那一步没接上约束。
        """
        token = int(token)
        if self._done:
            raise ValueError("JSON 已经结束，不能再喂 token")
        if token == PAD_ID:
            raise ValueError("pad 不是合法的 JSON token")
        if token == EOS_ID:
            if not self.complete:
                raise ValueError("JSON 还没写完就遇到 EOS")
            self._done = True
            return

        allowed = self.allowed()
        if token not in allowed:
            raise ValueError(
                f"token {token} 不在当前合法集合里（{len(allowed)} 个可选）"
            )

        b = token - BYTE_OFFSET
        if not 0 <= b < 256:
            raise ValueError(f"token {token} 不是字节 token")
        if self._mode == "number":
            if self._number_feed(b):
                return
            # 这个字节不属于数字：数字到此为止，再按分隔符 / 关闭符处理。
            self._mode = "value"
            self._after_value()
        elif self._mode == "string":
            if b == _QUOTE:
                self._mode = "value"
                self._after_value()
            elif b == _BACKSLASH:
                self._mode = "escape"
            return
        elif self._mode == "escape":
            self._mode = "unicode" if b == ord("u") else "string"
            if self._mode == "unicode":
                self._unicode_left = 4
            return
        elif self._mode == "unicode":
            self._unicode_left -= 1
            if self._unicode_left == 0:
                self._mode = "string"
            return
        elif self._mode == "literal":
            self._literal = self._literal[1:]
            if not self._literal:
                self._mode = "value"
                self._after_value()
            return

        self._feed_value(b)

    def _number_feed(self, b: int) -> bool:
        """推进数字状态。返回 ``False`` 表示这个字节不属于数字。"""
        stage = _advance_number(self._number, chr(b))
        if stage is None:
            return False
        self._number = stage
        return True

    def _feed_value(self, b: int) -> None:
        ch = chr(b)
        if ch in "{[":
            self._stack.append((ch, "key" if ch == "{" else "first"))
            return
        if ch == "}":
            kind, _phase = self._stack.pop()
            if kind != "{":
                raise ValueError("} 关闭的不是对象")
            self._after_value()
            return
        if ch == "]":
            kind, _phase = self._stack.pop()
            if kind != "[":
                raise ValueError("] 关闭的不是数组")
            self._after_value()
            return
        if ch == '"':
            self._mode = "string"
            return
        if ch == ":":
            self._stack[-1] = ("{", "value")
            return
        if ch == ",":
            kind, _ = self._stack[-1]
            self._stack[-1] = (kind, "key_more" if kind == "{" else "value")
            return
        if ch in "-0123456789":
            self._mode = "number"
            self._number = "start"
            self._number_feed(b)
            return
        if ch in "tfn":
            word = {"t": "rue", "f": "alse", "n": "ull"}[ch]
            self._mode = "literal"
            self._literal = word
            return
        raise ValueError(f"字节 {b} 不能作为值的开头")

    def _after_value(self) -> None:
        if not self._stack:
            self._mode = "end"
            return
        kind, phase = self._stack[-1]
        # 对象里刚写完的是 key 还是值，靠 phase 区分：key 之后要冒号，
        # 值之后要逗号或关闭。数组只有一个阶段，统一走 "comma"。
        if kind == "{" and phase in ("key", "key_more"):
            self._stack[-1] = (kind, "colon")
        else:
            self._stack[-1] = (kind, "comma")


# --------------------------------------------------------------------------
# schema：Python 类型 ↔ JSON Schema 子集
# --------------------------------------------------------------------------

# 只映射能无损落到 JSON 的标量。``list`` / ``dict`` 需要元素类型，光靠
# Python 的 ``list`` 表达不了，因此不在这个表里——要数组参数就写 schema。
_JSON_TYPE_BY_PYTHON = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
}

_SCALAR_KINDS = frozenset({"string", "integer", "number", "boolean", "null"})


def _json_type(python_type: type) -> str:
    """把 ``{参数名: Python 类型}`` 里的类型翻成 JSON Schema 的 ``type``。"""
    try:
        return _JSON_TYPE_BY_PYTHON[python_type]
    except KeyError:
        raise ValueError(
            f"无法把 {python_type!r} 映射成 JSON Schema 类型；"
            "数组 / 对象参数请直接传 parameters"
        ) from None


def _schema_kind(schema: Dict[str, Any], where: str) -> str:
    """取出参数 schema 的类型名，并在构造期拒绝不受支持的形状。

    受约束文法只表达标量与标量数组（见模块文档）。这里做的是**构造期**检查：
    如果等到第一次 decode 才发现"这个 schema 表达不了"，模型已经生成了半截
    文本，错误现场会离开 schema 定义很远。
    """
    kind = schema.get("type")
    if isinstance(kind, list):
        raise ValueError(f"{where}：不支持联合类型 {kind}")
    if kind is None:
        raise ValueError(f"{where}：缺少 type")
    if kind == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            raise ValueError(f"{where}：数组必须声明 items")
        inner = items.get("type")
        if inner not in _SCALAR_KINDS:
            raise ValueError(
                f"{where}：数组元素只支持标量，得到 {inner!r}；"
                "嵌套结构请改用 JsonGrammar"
            )
        return "array"
    if kind not in _SCALAR_KINDS:
        raise ValueError(
            f"{where}：类型 {kind!r} 不在受约束解码支持的子集里"
            "（标量 / 标量数组）；嵌套对象请改用 JsonGrammar"
        )
    return kind


def _check_against_schema(value: Any, schema: Dict[str, Any], where: str) -> None:
    """按 JSON Schema 子集递归校验一个已解析的值。

    支持 ``type`` / ``enum`` / ``required`` / ``properties`` /
    ``additionalProperties`` / ``items``。``enum`` 先于 ``type`` 检查：一个
    枚举里的字符串不该因为类型对就放过去（类型检查管不了取值）。
    """
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{where} 必须是 {schema['enum']} 之一，得到 {value!r}")
    kind = schema.get("type")
    if kind is None:
        return
    if kind == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{where} 应为对象，得到 {type(value).__name__}")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"{where} 缺少参数 {missing}")
        if schema.get("additionalProperties", True) is False:
            unknown = [key for key in value if key not in properties]
            if unknown:
                raise ValueError(f"{where} 有不认识的参数 {unknown}")
        for key, sub in properties.items():
            if key in value:
                _check_against_schema(value[key], sub, f"{where}.{key}")
        return
    if kind == "array":
        if not isinstance(value, list):
            raise ValueError(f"{where} 应为数组，得到 {type(value).__name__}")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, element in enumerate(value):
                _check_against_schema(element, items, f"{where}[{index}]")
        return
    if kind == "integer":
        # bool 是 int 的子类：声明 integer 时不能把 True 当成 1 放过去。
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{where} 应为 integer，得到 {type(value).__name__}")
        return
    if kind == "number":
        # JSON 允许 1e999：它解析成 inf 而不是报错。非有限的 float 参数进了
        # 工具就是 NaN 扩散的起点，在入口就拒绝。
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{where} 应为 number，得到 {type(value).__name__}")
        return
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{where} 应为 string，得到 {type(value).__name__}")
        return
    if kind == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{where} 应为 boolean，得到 {type(value).__name__}")
        return
    if kind == "null":
        if value is not None:
            raise ValueError(f"{where} 应为 null，得到 {type(value).__name__}")
        return
    raise ValueError(f"{where}：未知的 schema 类型 {kind!r}")


# --------------------------------------------------------------------------
# 执行回路
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    """一个可被模型调用的函数：名字 + 参数表 + 实现。

    参数有两种写法，二选一：

    - ``required`` / ``optional``：``{参数名: Python 类型}``。便捷写法，只能
      表达标量；``schema()`` 会把它翻成等价的 JSON Schema。
    - ``parameters``：完整 / 部分的 JSON Schema（``type`` / ``properties`` /
      ``required`` / ``additionalProperties`` / ``items`` / ``enum``）。数组
      与枚举只能靠它表达。

    参数校验在这里做，而不是靠"模型总会给对"——受约束文法只保证 JSON 合法
    且键名对，值的语义（枚举、范围、嵌套形状）仍需校验。
    """

    name: str
    function: Callable[..., Any]
    required: Dict[str, type] = field(default_factory=dict)
    optional: Dict[str, type] = field(default_factory=dict)
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("工具名不能为空")
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise ValueError(f"参数同时被声明为必填与可选：{sorted(overlap)}")
        if self.parameters is not None:
            if self.required or self.optional:
                raise ValueError(
                    f"工具 {self.name} 同时给了 parameters 与 required/optional；"
                    "两种写法只能二选一"
                )
            if not isinstance(self.parameters, dict):
                raise ValueError(f"工具 {self.name} 的 parameters 必须是 JSON Schema 对象")
            kind = self.parameters.get("type", "object")
            if kind != "object":
                raise ValueError(
                    f"工具 {self.name} 的 parameters 根类型必须是 object，得到 {kind!r}"
                )
            properties = self.parameters.get("properties", {})
            if not isinstance(properties, dict):
                raise ValueError(f"工具 {self.name} 的 properties 必须是对象")
            required = self.parameters.get("required", [])
            if not isinstance(required, list):
                raise ValueError(f"工具 {self.name} 的 required 必须是数组")
            unknown = [key for key in required if key not in properties]
            if unknown:
                raise ValueError(f"工具 {self.name} 的 required 里有没有声明的参数 {unknown}")
            for key, sub in properties.items():
                if not isinstance(sub, dict):
                    raise ValueError(f"工具 {self.name} 的参数 {key} 的 schema 必须是对象")
                _schema_kind(sub, f"工具 {self.name} 的参数 {key}")

    def schema(self) -> Dict[str, Any]:
        """本工具的 JSON Schema 描述，可直接喂给模型的 function calling 接口。"""
        parameters = (
            self.parameters
            if self.parameters is not None
            else {
                "type": "object",
                "properties": {
                    key: {"type": _json_type(kind)}
                    for key, kind in list(self.required.items())
                    + list(self.optional.items())
                },
                "required": list(self.required),
                "additionalProperties": False,
            }
        )
        return {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,
        }

    def check(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """校验参数：缺必填、多未知、类型不符都拒绝，返回完整 kwargs。"""
        if self.parameters is not None:
            _check_against_schema(
                arguments,
                {**self.parameters, "additionalProperties": self.parameters.get(
                    "additionalProperties", False
                )},
                f"工具 {self.name} 的参数",
            )
            return dict(arguments)
        missing = [k for k in self.required if k not in arguments]
        if missing:
            raise ValueError(f"工具 {self.name} 缺少参数 {missing}")
        unknown = [k for k in arguments if k not in self.required and k not in self.optional]
        if unknown:
            raise ValueError(f"工具 {self.name} 不认识参数 {unknown}")
        for key, expected in list(self.required.items()) + list(self.optional.items()):
            if key not in arguments:
                continue
            value = arguments[key]
            # bool 是 int 的子类：声明 int 时不能把 True 当成 1 放过去，
            # 声明 float 时整数应当接受（JSON 里没有 1.0 与 1 的区别）。
            if expected is int:
                ok = isinstance(value, int) and not isinstance(value, bool)
            elif expected is float:
                ok = (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                )
            else:
                ok = isinstance(value, expected)
            if not ok:
                raise ValueError(
                    f"工具 {self.name} 的参数 {key} 应为 {expected.__name__}，"
                    f"得到 {type(value).__name__}"
                )
        return dict(arguments)

    def __call__(self, **arguments: Any) -> Any:
        return self.function(**self.check(arguments))


def parse_tool_call(text: str) -> Tuple[str, Dict[str, Any]]:
    """把模型输出解析成 ``(工具名, 参数)``。

    形状固定为 ``{"name": str, "arguments": {...}}``——这是受约束文法与该
    函数共享的契约；多一个字段或少一个字段都拒绝，不做"猜一下"的容错。
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"不是合法 JSON：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"工具调用必须是 JSON 对象，得到 {type(payload).__name__}")
    extra = set(payload) - {"name", "arguments"}
    if extra:
        raise ValueError(f"工具调用有多余字段 {sorted(extra)}")
    name = payload.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("工具调用的 name 必须是非空字符串")
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError("工具调用的 arguments 必须是 JSON 对象")
    return name, dict(arguments)


class ToolCallGrammar:
    """按注册表合成的受约束文法：工具名、参数键名、参数类型都在解码期限死。

    与 :class:`JsonGrammar` 的分工：那个回答"下一个字节能不能组成合法 JSON"，
    这个回答"下一个字节能不能组成**某个已注册工具的一次合法调用**"。模型因此
    写不出 ``"name":"sub"``（未注册）、``"arguments":{"c":1}``（多余参数），
    也写不出 ``{"a":"x"}`` 这种类型不符的对象。

    输出形状固定为 ``{"name":"<枚举>","arguments":{...}}``，并且**强制键顺序**：
    先 ``name`` 后 ``arguments``；参数先写必填、再写可选。固定顺序让状态机不
    需要回溯，也让"省略可选参数"只有一个出口（直接写 ``}``）。

    边界的代价：optional 参数只能按声明顺序逐个省略尾部；想跳过中间的某个
    optional 必须先把它写出来。这是受约束解码换来的确定性，不是缺陷。
    """

    def __init__(self, registry: "ToolRegistry") -> None:
        tools = registry.tools
        if not tools:
            raise ValueError("注册表里没有工具，无法合成受约束文法")
        self._tools = {tool.name: tool for tool in tools}
        self._names = tuple(tool.name for tool in tools)
        # 构造期就把每个工具的 schema 检查一遍：不受支持的形状在这里报错，
        # 而不是等模型生成了半截文本才在 feed 里暴露。
        self._plans = {tool.name: self._items_for(tool) for tool in tools}
        self._restart()

    @staticmethod
    def _items_for(tool: Tool) -> Tuple[Tuple[str, Dict[str, Any], bool], ...]:
        parameters = tool.schema()["parameters"]
        properties = parameters.get("properties", {})
        required = set(parameters.get("required", []))
        ordered = [key for key in properties if key in required] + [
            key for key in properties if key not in required
        ]
        items = []
        for key in ordered:
            sub = properties[key]
            _schema_kind(sub, f"工具 {tool.name} 的参数 {key}")
            items.append((key, sub, key in required))
        return tuple(items)

    def _restart(self) -> None:
        self._pending: List[int] = []
        self._phase_after = "name"
        self._done = False
        self._phase = "name"
        self._name_prefix = ""
        self._name_candidates = self._names
        self._items: Tuple[Tuple[str, Dict[str, Any], bool], ...] = ()
        self._index = 0
        self._number = "start"
        self._integer_only = False
        self._literal = ""
        self._literal_choices: Tuple[str, ...] = ()
        self._unicode_left = 4
        self._arrays: List[Dict[str, Any]] = []
        self._returns: List[str] = []
        self._emit(b'{"name":"', "name")

    def reset(self) -> None:
        self._restart()

    @property
    def complete(self) -> bool:
        return self._phase == "end" and not self._pending

    @property
    def done(self) -> bool:
        return self._done

    # -- 允许集合 ---------------------------------------------------------

    def allowed(self) -> FrozenSet[int]:
        """下一步允许的 token id 集合。永远不含 pad。"""
        if self._done:
            return frozenset()
        if self._pending:
            return frozenset({_BYTE_TOKEN[self._pending[0]]})
        phase = self._phase
        if phase == "name":
            chars = {
                name[len(self._name_prefix)]
                for name in self._name_candidates
                if len(name) > len(self._name_prefix)
            }
            result = {_BYTE_TOKEN[ord(c)] for c in chars}
            if self._name_prefix in self._name_candidates:
                result.add(_BYTE_TOKEN[_QUOTE])
            return frozenset(result)
        if phase == "after":
            return self._after_allowed()
        if phase == "key-open":
            return frozenset({_BYTE_TOKEN[_QUOTE]})
        if phase == "object-end":
            return frozenset({_BYTE_TOKEN[ord("}")]})
        if phase == "string-open":
            return frozenset({_BYTE_TOKEN[_QUOTE]})
        if phase == "string":
            return _STRING_START
        if phase == "escape":
            return _tokens('"\\/bfnrtu')
        if phase == "unicode":
            return _HEX
        if phase == "literal":
            return frozenset({_BYTE_TOKEN[ord(self._literal[0])]})
        if phase == "bool":
            return frozenset({_BYTE_TOKEN[ord(c)] for c in self._literal_choices})
        if phase == "number":
            return self._number_allowed()
        if phase == "array-open":
            return frozenset({_BYTE_TOKEN[ord("[")]})
        if phase in ("array-start", "array-item", "array-after"):
            if phase == "array-after":
                return frozenset({_BYTE_TOKEN[ord(",")], _BYTE_TOKEN[ord("]")]})
            result = set(self._kind_start(self._arrays[-1]["schema"]))
            if phase == "array-start":
                # 只有刚打开时可以立刻收尾（空数组）；逗号之后必须给元素。
                result.add(_BYTE_TOKEN[ord("]")])
            return frozenset(result)
        if phase == "end":
            return frozenset({EOS_ID})
        raise AssertionError(f"未知阶段 {phase!r}")

    def _after_allowed(self, index: Optional[int] = None) -> FrozenSet[int]:
        """写完第 ``index`` 个参数之后允许的分隔符；默认用当前游标。"""
        if index is None:
            index = self._index
        if index < len(self._items):
            result = {_BYTE_TOKEN[ord(",")]}
            if not self._items[index][2]:
                # optional：可以就此收尾。required 不能跳。
                result.add(_BYTE_TOKEN[ord("}")])
            return frozenset(result)
        return frozenset({_BYTE_TOKEN[ord("}")]})

    @staticmethod
    def _kind_start(schema: Dict[str, Any]) -> FrozenSet[int]:
        kind = schema.get("type")
        if kind == "string":
            return frozenset({_BYTE_TOKEN[_QUOTE]})
        if kind in ("integer", "number"):
            return _tokens("-0123456789")
        if kind == "boolean":
            return _tokens("tf")
        if kind == "null":
            return _tokens("n")
        raise AssertionError(f"未知的参数类型 {kind!r}")

    def _number_allowed(self) -> FrozenSet[int]:
        stage = self._number
        if stage == "start":
            return _tokens("-0123456789")
        if stage == "minus":
            return _DIGITS
        if stage == "zero":
            extra = frozenset() if self._integer_only else _tokens(".eE")
            return extra | self._value_end_allowed()
        if stage == "int":
            extra = frozenset() if self._integer_only else _tokens(".eE")
            return _DIGITS | extra | self._value_end_allowed()
        if stage in ("dot", "exp-sign"):
            return _DIGITS
        if stage == "frac":
            return _DIGITS | _tokens("eE") | self._value_end_allowed()
        if stage == "exp":
            return _DIGITS | _tokens("+-")
        if stage == "exp-digits":
            return _DIGITS | self._value_end_allowed()
        raise AssertionError(f"未知的数字阶段 {stage!r}")

    def _value_end_allowed(self) -> FrozenSet[int]:
        if self._returns and self._returns[-1] == "array":
            return frozenset({_BYTE_TOKEN[ord(",")], _BYTE_TOKEN[ord("]")]})
        # 这个标量还没收尾（``_end_value`` 尚未调用），所以按"写完它之后"的
        # 游标算分隔符；用当前游标会把当前这个参数自己的逗号也排除掉。
        return self._after_allowed(self._index + 1)

    # -- 推进 -------------------------------------------------------------

    def feed(self, token: int) -> None:
        """吃掉一个 token。越界或非法都抛 ``ValueError``。"""
        token = int(token)
        if self._done:
            raise ValueError("工具调用已经结束，不能再喂 token")
        if token == PAD_ID:
            raise ValueError("pad 不是合法的工具调用 token")
        if token == EOS_ID:
            if not self.complete:
                raise ValueError("工具调用还没写完就遇到 EOS")
            self._done = True
            return

        allowed = self.allowed()
        if token not in allowed:
            raise ValueError(
                f"token {token} 不在当前合法集合里（{len(allowed)} 个可选）"
            )
        if self._pending:
            self._pending.pop(0)
            if not self._pending:
                self._phase = self._phase_after
            return

        b = token - BYTE_OFFSET
        if not 0 <= b < 256:
            raise ValueError(f"token {token} 不是字节 token")
        phase = self._phase
        if phase == "name":
            self._feed_name(chr(b))
        elif phase == "key-open":
            self._feed_key()
        elif phase == "object-end":
            # 模型已经写出 arguments 的闭合花括号，只剩根对象的那个。
            self._emit(b"}", "end")
        elif phase == "after":
            self._feed_after(chr(b))
        elif phase == "number":
            self._feed_number(token, b)
        elif phase == "string-open":
            self._phase = "string"
        elif phase == "string":
            if b == _QUOTE:
                self._end_value()
            elif b == _BACKSLASH:
                self._phase = "escape"
        elif phase == "escape":
            self._phase = "unicode" if b == ord("u") else "string"
            if self._phase == "unicode":
                self._unicode_left = 4
        elif phase == "unicode":
            self._unicode_left -= 1
            if self._unicode_left == 0:
                self._phase = "string"
        elif phase == "bool":
            self._literal = "rue" if chr(b) == "t" else "alse"
            self._phase = "literal"
        elif phase == "literal":
            self._literal = self._literal[1:]
            if not self._literal:
                self._end_value()
        elif phase == "array-open":
            self._phase = "array-start"
        elif phase in ("array-start", "array-item", "array-after"):
            self._feed_array(token, chr(b))
        else:
            raise AssertionError(f"未知阶段 {phase!r}")

    def _emit(self, data: bytes, phase: str) -> None:
        self._pending.extend(data)
        self._phase_after = phase
        if not self._pending:
            self._phase = phase

    def _feed_name(self, ch: str) -> None:
        if ch == '"':
            self._items = self._plans[self._name_prefix]
            self._index = 0
            # 名字的闭合引号已经由模型输出，这里接的是逗号与键名。
            self._emit(
                b',"arguments":{', "key-open" if self._items else "object-end"
            )
            return
        self._name_prefix += ch
        self._name_candidates = tuple(
            name for name in self._name_candidates if name.startswith(self._name_prefix)
        )

    def _feed_key(self) -> None:
        key, schema, _required = self._items[self._index]
        self._begin_value(schema)
        # 键的开引号由模型输出，这里接的是键名余下的字节与冒号。
        self._emit(key.encode() + b'":', self._phase)

    def _feed_after(self, ch: str) -> None:
        if ch == "}":
            # arguments 的闭合花括号已经写出，剩根对象的那个。
            self._emit(b"}", "end")
            return
        self._phase = "key-open"

    def _begin_value(self, schema: Dict[str, Any]) -> None:
        kind = schema.get("type")
        if kind == "string":
            self._phase = "string-open"
        elif kind == "integer":
            self._phase = "number"
            self._number = "start"
            self._integer_only = True
        elif kind == "number":
            self._phase = "number"
            self._number = "start"
            self._integer_only = False
        elif kind == "boolean":
            self._phase = "bool"
            self._literal_choices = ("t", "f")
        elif kind == "null":
            self._phase = "literal"
            self._literal = "null"
        elif kind == "array":
            self._phase = "array-open"
            self._arrays.append({"schema": schema.get("items", {})})
        else:
            raise AssertionError(f"未知的参数类型 {kind!r}")

    def _feed_number(self, token: int, b: int) -> None:
        stage = _advance_number(self._number, chr(b), integer_only=self._integer_only)
        if stage is not None:
            self._number = stage
            return
        # 这个字节不属于数字：数字到此为止，按分隔符 / 关闭重新分派。
        self._end_value()
        self.feed(token)

    def _feed_array(self, token: int, ch: str) -> None:
        if ch == "]":
            self._arrays.pop()
            self._end_value()
            return
        if ch == ",":
            # 逗号只是分隔符，已经消费；下一个字节是元素本身。
            self._phase = "array-item"
            return
        self._returns.append("array")
        self._begin_value(self._arrays[-1]["schema"])
        self.feed(token)

    def _end_value(self) -> None:
        if self._returns and self._returns[-1] == "array":
            self._returns.pop()
            self._phase = "array-after"
            return
        self._index += 1
        self._phase = "after"


class ToolRegistry:
    """名字 → :class:`Tool` 的注册表，``call`` 是完整的执行回路。"""

    def __init__(self, tools: Sequence[Tool] = ()) -> None:
        self._tools: Dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具 {tool.name} 已注册")
        self._tools[tool.name] = tool

    @property
    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._tools))

    @property
    def tools(self) -> Tuple[Tool, ...]:
        """按名字排序的工具元组：``schema()`` 与文法合成的稳定输入。"""
        return tuple(self._tools[name] for name in self.names)

    def schema(self) -> List[Dict[str, Any]]:
        """全部工具的 JSON Schema 列表，可直接作为 function calling 的 ``tools``。"""
        return [tool.schema() for tool in self.tools]

    def grammar(self) -> ToolCallGrammar:
        """受约束解码用的文法：工具名与参数键名都由注册表决定。

        ``JsonGrammar`` 只保证"是合法 JSON 对象"，模型仍可能写出未注册的工具名
        或多余的参数键；这里按注册表合成文法，把这些可能性在解码期排除。参数
        值类型同样受限（``integer`` 不会接受 ``1.5``），但取值约束（``enum``）
        仍由 :meth:`Tool.check` 在调用前校验——文法只认类型，不认语义。
        """
        return ToolCallGrammar(self)

    def call(self, text: str) -> Any:
        """受约束解码出的文本 → 解析 → 校验 → 调用。"""
        name, arguments = parse_tool_call(text)
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"未注册的工具 {name!r}；可用：{list(self.names)}")
        return tool(**arguments)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: object) -> bool:
        return name in self._tools
