# 贡献指南

感谢参与。这个仓库很小，但**它的主张很硬**（"放置只改速度、不改语义"），
所以我们更看重**可验证**而不是**多功能**。

## 开发环境

```bash
git clone https://github.com/sky20120815/moe-residency.git && cd moe-residency
pip install -e ".[dev]"      # 运行时只依赖 numpy
python run_tests.py -v       # 53 个用例，约 0.2 秒，不需要 pytest
```

也可以 `pytest tests/`（装了 pytest 的话）。

## 三条硬规矩

1. **不要在示例里"顺手"加功能。** `engine.py` 是玩具引擎，只用于验证机制。
2. **任何新的缓存策略 / 预取器 / 压缩，都必须通过逐位不变测试**
   （`tests/test_engine.py::test_all_modes_bitexact` 是模板）。
   在你的策略里加一个 mode，然后用同样的方式跑一遍"全驻留 vs 分层"。
3. **不要引用别人项目的内部文件格式。** colibri 的 `.coli_usage` 格式未公开，
   我们只使用其**文档公开的开关**；这不是洁癖，是因为猜出来的格式会让别人的用户踩坑。

## 提交前请确认

- [ ] `python run_tests.py` 全绿
- [ ] 新策略在 `benchmarks/policy_bench.py` 的**两种负载**（stable / agentic）下都测过，
      并把数字写进 PR 描述 —— **包括它变差的情况**（我们已记录 pin 与 SLRU 在某些负载上更差）
- [ ] 如果改变了策略/预取的语义，同时更新 `docs/DESIGN.md` 的「实现纪律」
- [ ] 新增借鉴的机制时，更新 `docs/ATTRIBUTION.md`（含许可）

## 指标的正确用法（本项目最容易出错的地方）

**不要用「读盘次数」评价预取。** 预取会增加总 I/O、减少停顿。
请同时报告：**停顿 miss**（用户感知的慢）与 **预取 I/O**（带宽成本）。
参见 `docs/DESIGN.md` 第 3 节。

## 我们不做什么

见 README 的 `❌ 明确不承诺`。特别是：**本仓库不提供任何真实模型的 tok/s**。
