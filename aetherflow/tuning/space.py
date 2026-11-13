"""调优参数空间：**有哪些旋钮、每个旋钮能拧到哪**。

这一层刻意不了解任何后端，也不做测量。它只回答两个问题：

1. 一个算子暴露了哪些可调参数、每个参数的候选值是什么；
2. 把这些参数展开成确定顺序的配置列表。

顺序必须确定。搜索器、配置库、测试都依赖"同一空间展开出同一串配置"——
只要顺序随哈希或字典插入顺序漂移，两次调优就会写出不同的配置库，
而"为什么这次选了这个"将无法复现。展开规则是**先声明的参数在外层**，
类似嵌套循环：``a`` 的取值最慢变，最后一个参数最快变。

候选值必须非空、可哈希且互不重复。空候选等于"这个参数没有可选值"，
把它当成合法空间会静默产出一个不可用的配置；重复候选则会让同一个配置
被测两次、并让"第几个候选胜出"变得没有意义。两者都在构造时拒绝。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, Mapping, Tuple

__all__ = ["TuningSpace", "canonical_signature"]


def _freeze_values(name: str, values) -> Tuple[Any, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"调优参数 {name!r} 的候选必须是可迭代的序列，得到 {type(values).__name__}")
    frozen = tuple(values)
    if not frozen:
        raise ValueError(f"调优参数 {name!r} 的候选为空：没有可选值的参数不应出现在空间里")
    for value in frozen:
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"调优参数 {name!r} 的候选 {value!r} 不可哈希") from exc
    if len(set(frozen)) != len(frozen):
        raise ValueError(f"调优参数 {name!r} 的候选有重复：{frozen}")
    return frozen


@dataclass(frozen=True)
class TuningSpace:
    """``参数名 -> 候选值`` 的有序集合。

    冻结的是"结构"，不是候选值本身——候选值只要求可哈希，
    可以是 ``int`` / ``str`` / ``bool`` / 元组。可变对象（列表、dict）
    不能作为候选，因为它们无法进入稳定的配置键。
    """

    params: Tuple[Tuple[str, Tuple[Any, ...]], ...]

    @classmethod
    def build(cls, params: Mapping[str, Any]) -> "TuningSpace":
        """从普通映射构造；保留声明顺序，逐个校验候选。"""
        if not params:
            raise ValueError("调优空间为空：至少要有一个可调参数")
        return cls(tuple((name, _freeze_values(name, values)) for name, values in params.items()))

    def names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.params)

    def values(self, name: str) -> Tuple[Any, ...]:
        for param, values in self.params:
            if param == name:
                return values
        raise KeyError(f"调优空间里没有参数 {name!r}，可选：{self.names()}")

    def default(self) -> Dict[str, Any]:
        """每个参数取**第一个**候选；调用方应把默认值放在首位。"""
        return {name: values[0] for name, values in self.params}

    def configs(self) -> Iterator[Dict[str, Any]]:
        """按声明顺序的笛卡尔积展开，结果确定且可复现。"""
        def walk(index: int, acc: Dict[str, Any]):
            if index == len(self.params):
                yield dict(acc)
                return
            name, values = self.params[index]
            for value in values:
                acc[name] = value
                yield from walk(index + 1, acc)
            acc.pop(name, None)

        yield from walk(0, {})

    def size(self) -> int:
        total = 1
        for _, values in self.params:
            total *= len(values)
        return total

    def to_dict(self) -> Dict[str, list]:
        return {name: list(values) for name, values in self.params}

    def __len__(self) -> int:
        return len(self.params)

    def __contains__(self, name: object) -> bool:
        return any(name == param for param, _ in self.params)


def canonical_signature(**parts: Any) -> Dict[str, Any]:
    """把"这个配置是在什么形状/属性下测出来的"规范化成可比较的字典。

    形状统一转成 list 的 list，dtype 用字符串。排序由配置库在算键时做，
    这里只保证类型干净——tuple 与 list 在 JSON 往返后会变成两种东西，
    如果签名里直接混用，``get`` 就会永远不命中。
    """
    out: Dict[str, Any] = {}
    for name, value in parts.items():
        if value is None:
            continue
        if isinstance(value, (tuple, list)) and all(
            isinstance(item, (tuple, list)) for item in value
        ):
            out[name] = [[int(d) for d in item] for item in value]
        elif isinstance(value, (tuple, list)):
            out[name] = [int(d) for d in value]
        elif hasattr(value, "value"):  # DType 之类
            out[name] = str(value.value)
        else:
            out[name] = value
    return out
