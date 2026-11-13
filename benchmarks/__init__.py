"""性能基准：固定图集 + 与基线对比判定回归。

与 conformance 的分工：conformance 回答"算得对不对"，这里回答"有没有变慢"。
两者都是**准入判据**，区别只是一个比数值，一个比时间。

子模块刻意不在包初始化时导入，否则 ``python -m benchmarks.run`` 会先
执行一次包初始化、再执行同一模块，产生 "found in sys.modules" 警告。
"""
