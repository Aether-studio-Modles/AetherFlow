"""跑基准、存基线、判回归。

用法：

```bash
# 采集（结果写到 JSON）
py -3.10 -m benchmarks.run --backend ref  --save benchmarks/baseline_ref.json
py -3.10 -m benchmarks.run --backend cuda --save benchmarks/baseline_cuda.json

# 回归判定：任何用例的中位数超过基线 (1 + max_slowdown) 即失败
py -3.10 -m benchmarks.run --backend cuda --check benchmarks/baseline_cuda.json
```

每个用例默认跑 3 轮取最好一轮（``--rounds``），用于抵掉整机状态造成的漂移；
``--rounds 1`` 只适合快速自测。

**只在同一台机器、同一个后端内比较**。CUDA 与参考后端的数字不可比，
不同 GPU 的数字也不可比；报告里记下后端名、设备信息与宿主机器指纹
（``machine``：CPU 型号 / 核数 / Python 版本），正是为了让这件事可核对——
``--check`` 发现指纹不同时会先打印一句"比值只作参考"。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

from aetherflow.backends.discovery import (
    BackendUnavailableError,
    get_backend,
    registered_backends,
)

from .cases import CASES, Case, build_case


def prepare_case(backend, case: Case, seed: int = 0):
    """构建 ``(graph, feed)``；带 ``split`` 的用例把权重交给后端常驻。

    参考后端没有驻留能力（它不区分设备内存），继续用完整 feed——否则
    "热身"会变成一次缺少输入的报错。有驻留能力的后端先清空上一张图的权重，
    避免上一个用例的参数残留到这一张图的合并输入里。
    """
    graph, feed = build_case(case, seed)
    if case.split is None:
        return graph, feed
    params, activations = case.split(feed)
    if not hasattr(backend, "resident_parameters"):
        return graph, feed
    if hasattr(backend, "release_parameters"):
        backend.release_parameters()
    backend.resident_parameters(graph, params)
    return graph, activations


def backend_names() -> Tuple[str, ...]:
    """CLI 的 ``--backend`` 可选值：习惯用的 ``ref`` 加上所有已注册后端。

    ``ref`` 是 ``reference`` 的短名，从列表里去掉后者避免同一个后端出现两次；
    其余名字（``cuda`` / ``hip`` / 已安装的硬件包）原样保留。
    """
    names = [name for name in registered_backends() if name != "reference"]
    return ("ref",) + tuple(names)


def open_backend(name: str, **kwargs):
    """按 CLI 名构造硬件后端，``kwargs`` 原样交给工厂。

    明确指定却装不起来时**直接失败**：这里不能走 ``get_backend`` 的自动
    回退——基准工具悄悄退回 CPU 会产出一组看起来完全正常的假性能数字。

    ``ref`` 不走这里：``run`` 用 ``ReferenceBackend``、``trend`` 用
    ``ReferenceExecutor``，两者不是同一个东西，由调用方保留原来的选择。

    装配失败转成 ``SystemExit``（与 ``_require_tunable`` 一致），这样命令行
    只看到一行原因，而不是一段 traceback。
    """
    try:
        return get_backend(name, fallback=False, **kwargs)
    except BackendUnavailableError as exc:
        raise SystemExit(f"--backend {name} 在这台机器上不可用：{exc}") from exc


def _sync(backend) -> None:
    """等设备把活干完。没有 GPU 的后端是空操作。

    cupy 后端走 ``cp.cuda``；HIP 的替身模块没有这个假命名空间（见
    ``hip/runtime.py::current_stream``），用它的模块级 ``synchronize()``。
    """
    cp = getattr(backend, "cp", None)
    if cp is None:
        return
    cuda = getattr(cp, "cuda", None)
    if cuda is not None:
        cuda.get_current_stream().synchronize()
        return
    cp.synchronize()


def warmup_backend(backend, cases: Sequence[Case], seconds: float, seed: int) -> None:
    """把设备从空闲降频状态拉起来，再开始测量。

    RTX 2080 Ti 空闲时 SM 只有 300 MHz（满频 2160）。每个用例自带的
    ``warmup`` 只跑几次小图，不足以升频——实测在刚连上的机器上直接开测，
    ``transformer_block`` 被判慢 37%，而同一份代码先压半秒之后只慢 10%。
    这种误报比不做门禁更糟：它会让人去改本来就正确的代码。

    只对持有 ``cp``（cupy）的后端生效：参考后端的"热身"只是白等时间。

    热身不按 ``prepare_case`` 驻留权重：驻留表是后端级的一份，逐个用例驻留
    会把前一个用例的输入抽走。这里只求把设备时钟拉起来；权重常驻用例的计划
    会在正式测量前被 ``prepare_case`` 清空，并按新口径在**不计时**的
    ``warmup`` 次数里重新捕获。
    """
    if seconds <= 0 or not cases or getattr(backend, "cp", None) is None:
        return
    built = [build_case(case, seed) for case in cases]
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        for graph, feed in built:
            backend.execute(graph, feed)
        _sync(backend)


def measure(
    backend, graph, feed, warmup: int, repeat: int, device_outputs: bool = False
) -> Dict[str, float]:
    """重复执行并取中位数；同步放在计时区间内，测的是"算完"的时间。

    ``device_outputs=True`` 时走 ``execute_device``：输出留在设备上，不回传
    主机。两种口径不能混为一谈——一次 4 MB 的 D2H 在 A800 上要 1.4 ms，比
    绝大多数内核本身还贵（见 DESIGN 5.251），把回传算进去测的是 PCIe 带宽。
    """
    run = backend.execute_device if device_outputs else backend.execute
    for _ in range(warmup):
        run(graph, feed)
    _sync(backend)

    samples: List[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        run(graph, feed)
        _sync(backend)
        samples.append((time.perf_counter() - t0) * 1e3)

    samples.sort()
    p90 = samples[min(len(samples) - 1, int(round(0.9 * (len(samples) - 1))))]
    return {
        "median_ms": round(statistics.median(samples), 4),
        "p90_ms": round(p90, 4),
        "min_ms": round(samples[0], 4),
    }


def _device_label(backend) -> str:
    info = getattr(backend, "device_info", None)
    if info is None:
        return "unknown"
    try:
        return str(info()).replace("\n", " ")
    except Exception:  # 统计信息拿不到不应该让基准失败
        return "unknown"


def _cpu_model() -> str:
    """宿主 CPU 型号：云 GPU 上它比显卡更容易换，而它决定 launch 开销。"""
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _machine_label() -> Dict:
    """基线必须带机器指纹，否则"整体变慢"无法归因。

    实测：同一张 RTX 2080 Ti、同一份代码，换一台宿主 CPU 更弱的实例之后，
    8 个用例**整体**慢 1.9～2.7 倍（纯 launch 开销的 ``matmul_256`` 最明显）。
    这种"全体齐平变慢"与"某个 kernel 回归"在数字上完全同形，没有指纹就只能
    靠人回忆上次是在哪台机器上采的。
    """
    return {
        "host": platform.node(),
        "cpu": _cpu_model(),
        "cores": os.cpu_count(),
        "python": platform.python_version(),
    }


def machine_note(baseline: Dict, current: Dict) -> str:
    """基线与当前是否同一台机器；不同就返回一句说明，相同或无法判断返回空串。

    只提示、不判失败：CI runner 改名、换一批同型号实例都不该让门禁变红。
    真正要防的是"跨机器比较却当成回归"。
    """
    old = baseline.get("machine")
    if not old:
        return ""
    now = current.get("machine") or {}
    old = {**old, "device": baseline.get("device")}
    now = {**now, "device": current.get("device")}
    diffs = [
        f"{key}: {old.get(key)} -> {now.get(key)}"
        for key in ("cpu", "cores", "device")
        if old.get(key) != now.get(key)
    ]
    if not diffs:
        return ""
    return "基线与当前不是同一台机器（" + "；".join(diffs) + "），比值只作参考"


def run_benchmark(
    backend,
    cases: Sequence[Case] = CASES,
    seed: int = 0,
    progress=None,
    rounds: int = 3,
    warmup_seconds: float = 0.0,
    device_outputs: bool = False,
) -> Dict:
    """跑全部用例，每个用例取 ``rounds`` 轮里最好的一轮。

    多轮不是为了"挑个好看的数字"，而是因为整轮漂移确实存在：同一台云 GPU
    连续两次全量运行，``matmul_256`` 的中位数从 0.229 跳到 0.299（+30%）。
    这种漂移不是某个 kernel 变慢，而是整机状态（邻居负载 / 时钟）变了；
    取最好一轮，让判定对准"这台机器能做到多快"，而不是"这一分钟有多吵"。

    ``warmup_seconds`` 在正式测量前持续压设备一段时间，专治"空闲降频"——
    它比"取最好一轮"更靠前：降频时整轮都是冷的，取多少轮都无济于事。
    """
    results: Dict[str, Dict] = {}
    rounds = max(1, int(rounds))
    warmup_backend(backend, cases, warmup_seconds, seed)
    for case in cases:
        graph, feed = prepare_case(backend, case, seed)
        best: Optional[Dict[str, float]] = None
        for _ in range(rounds):
            stats = measure(
                backend, graph, feed, case.warmup, case.repeat, device_outputs
            )
            if best is None or stats["median_ms"] < best["median_ms"]:
                best = stats
        results[case.name] = {**best, "note": case.note, "rounds": rounds}
        if progress is not None:
            progress(case, best)
    return {
        "backend": backend.name,
        "device": _device_label(backend),
        "machine": _machine_label(),
        "device_outputs": bool(device_outputs),
        "cases": results,
        **({"conv_impl": backend.conv_impl} if hasattr(backend, "conv_impl") else {}),
        **(
            {"graph_mode": backend.graph_mode}
            if hasattr(backend, "graph_mode")
            else {}
        ),
    }


def check_regression(
    baseline: Dict, current: Dict, max_slowdown: float = 0.25
) -> List[Dict]:
    """返回问题列表；空列表表示没有回归。

    规则刻意保守：
    - 新用例（基线里没有）不算回归，只提示；
    - 基线有、现在缺，算问题——用例被删或跑挂，同样是退化；
    - 后端名不同直接算问题，跨机/跨后端的数字没有可比性。
    """
    problems: List[Dict] = []
    if baseline.get("backend") != current.get("backend"):
        problems.append(
            {
                "kind": "backend_mismatch",
                "detail": f"基线是 {baseline.get('backend')}，当前是 {current.get('backend')}",
            }
        )
    if bool(baseline.get("device_outputs")) != bool(current.get("device_outputs")):
        # 两种口径差一倍以上（回传 vs 不回传），混比会把"换了测量方式"判成回归。
        problems.append(
            {
                "kind": "throughput_mode_mismatch",
                "detail": f"基线 device_outputs={baseline.get('device_outputs')}，"
                f"当前 {current.get('device_outputs')}；请用同一口径重新采集",
            }
        )

    base_cases = baseline.get("cases", {})
    cur_cases = current.get("cases", {})
    for name, base in base_cases.items():
        cur = cur_cases.get(name)
        if cur is None:
            problems.append({"kind": "missing", "case": name})
            continue
        limit = base["median_ms"] * (1.0 + max_slowdown)
        if cur["median_ms"] > limit:
            problems.append(
                {
                    "kind": "slower",
                    "case": name,
                    "baseline_ms": base["median_ms"],
                    "current_ms": cur["median_ms"],
                    "slowdown": round(cur["median_ms"] / base["median_ms"] - 1.0, 4),
                }
            )
    return problems


def recheck_slow(
    backend,
    baseline: Dict,
    report: Dict,
    cases: Sequence[Case] = CASES,
    max_slowdown: float = 0.25,
    rounds: int = 1,
    warmup_seconds: float = 0.0,
    device_outputs: bool = False,
):
    """对超阈值的用例重测一遍，取两次中更快的结果。

    整机级的瞬时干扰（杀毒扫描、Windows 更新、邻居容器）可以把**整轮**测量拖慢
    80%，几秒后自己又好了——实测里同一分钟内先 +84%、再正常。
    真实退化是可复现的，所以规则是"超了不算数，再测一次；还超才算回归"。
    重测同样要先热身：冷启动的重测只会比首测更慢，"取更快者"就形同虚设。
    返回 ``(问题列表, 重测的用例数)``；``report`` 会被就地更新。
    """
    problems = check_regression(baseline, report, max_slowdown)
    slow = {p["case"] for p in problems if p["kind"] == "slower"}
    if not slow:
        return problems, 0

    retry = run_benchmark(
        backend,
        cases=[c for c in cases if c.name in slow],
        rounds=rounds,
        warmup_seconds=warmup_seconds,
        device_outputs=device_outputs,
    )
    for name in slow:
        if retry["cases"][name]["median_ms"] < report["cases"][name]["median_ms"]:
            report["cases"][name] = retry["cases"][name]
    return check_regression(baseline, report, max_slowdown), len(slow)


def _print_table(report: Dict) -> None:
    print(f"backend : {report['backend']}")
    print(f"device  : {report['device']}")
    if report.get("device_outputs"):
        print("outputs : 设备侧（不计 D2H）")
    else:
        print("outputs : 主机侧（含 D2H）")
    if "conv_impl" in report:
        print(f"conv    : {report['conv_impl']}")
    if "graph_mode" in report:
        print(f"graph   : {report['graph_mode']}")
    print(f"{'case':22s} {'median':>10s} {'p90':>10s} {'min':>10s}")
    for name, stats in report["cases"].items():
        print(
            f"{name:22s} {stats['median_ms']:10.4f} "
            f"{stats['p90_ms']:10.4f} {stats['min_ms']:10.4f}"
        )


def _main(argv: Optional[Sequence[str]] = None) -> int:
    from aetherflow import ReferenceBackend

    parser = argparse.ArgumentParser(description="AetherFlow 性能基准")
    parser.add_argument("--backend", choices=backend_names(), default="ref")
    parser.add_argument("--save", help="把结果写到 JSON")
    parser.add_argument("--check", help="与该基线 JSON 比较，超阈值则退出码非 0")
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=0.25,
        help="允许的最大变慢比例（默认 0.25，即 25%%）",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="每个用例跑几轮、取最好一轮（默认 3；机器稳定时可用 1 提速）",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.5,
        help="正式测量前持续压设备的秒数，用于解除空闲降频（默认 0.5；0 关闭）",
    )
    parser.add_argument(
        "--conv-impl",
        choices=("auto", "im2col", "tiled"),
        default="auto",
        help="conv2d 的实现；auto 走启发式，im2col/tiled 用于对照实验",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("off", "auto"),
        default="off",
        help="execute 的图执行方式；auto 首次按形状捕获整图（节点数 ≥ 2 才捕获），之后只 replay",
    )
    parser.add_argument(
        "--device-outputs",
        action="store_true",
        help="输出留在设备上、不计 D2H 回传；测的是设备侧吞吐而非端到端延迟",
    )
    args = parser.parse_args(argv)

    if args.backend == "ref":
        backend = ReferenceBackend()
    else:
        # conv_impl / graph_mode 是 CUDA / HIP 共有的实现选择（HIP 继承同一批
        # 内核）；其它硬件包没有这些概念，透传过去只会撞上一个意外的 TypeError。
        kwargs = (
            {"conv_impl": args.conv_impl, "graph_mode": args.graph_mode}
            if args.backend in ("cuda", "hip")
            else {}
        )
        backend = open_backend(args.backend, **kwargs)

    if args.device_outputs and not hasattr(backend, "execute_device"):
        raise SystemExit(
            "--device-outputs 只对设备后端有效：参考后端没有设备内存，"
            "它的输出本来就在主机上"
        )

    def progress(case, stats):
        print(f"  {case.name:22s} {stats['median_ms']:10.4f} ms", flush=True)

    print(f"测量 {len(CASES)} 个用例（{backend.name}，每例 {args.rounds} 轮取最好）……")
    if args.warmup > 0:
        print(f"先热身 {args.warmup:.1f}s（解除空闲降频）……", flush=True)
    report = run_benchmark(
        backend,
        progress=progress,
        rounds=args.rounds,
        warmup_seconds=args.warmup,
        device_outputs=args.device_outputs,
    )

    problems = []
    if args.check:
        with open(args.check, encoding="utf-8") as fh:
            baseline = json.load(fh)
        note = machine_note(baseline, report)
        if note:
            print(note)
        problems, retried = recheck_slow(
            backend,
            baseline,
            report,
            max_slowdown=args.max_slowdown,
            rounds=args.rounds,
            warmup_seconds=args.warmup,
            device_outputs=args.device_outputs,
        )
        if retried:
            print(f"有 {retried} 个用例首轮超阈值，已重测一次（取更快者）")

    _print_table(report)

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"已写入 {args.save}")

    if args.check:
        if not problems:
            print(f"回归检查通过（阈值 {args.max_slowdown:.0%}）")
            return 0
        print("回归检查失败：")
        for item in problems:
            if item["kind"] == "slower":
                print(
                    f"  [slower ] {item['case']:22s} "
                    f"{item['baseline_ms']:.4f} -> {item['current_ms']:.4f} ms "
                    f"(+{item['slowdown']:.1%})"
                )
            elif item["kind"] == "missing":
                print(f"  [missing] {item['case']}")
            else:
                print(f"  [{item['kind']}] {item['detail']}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(_main())
