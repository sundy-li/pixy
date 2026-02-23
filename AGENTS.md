# AGENTS.md（pixy-rs）

适用范围：`/data/work/pixy-rs`。

## 目标

在 Rust 中复刻 `pixy-mono` 的核心能力，当前重点库：
- `crates/pixy-ai`
- `crates/pixy-agent-core`
- `crates/pixy-coding-agent`

## 文档与任务管理

- 架构与阶段计划：`docs/plan.md`
- 可执行任务看板：`docs/task.md`
- 规则：每次完成功能后，必须同步更新 `docs/task.md`（将对应 TODO 从 `[ ]` 改为 `[x]`，必要时补充“近期完成记录”）。

## 开发约定

- 优先保持行为语义与 `pixy-mono` 对齐，不追求序列化 bit-level 完全一致。
- 小步迭代：先补测试，再实现，再回归测试。
- 不移除看起来是“有意设计”的功能，除非明确确认。
- 无用户明确要求时，不做提交（commit/push）。

## 测试约定

- 代码改动后至少执行：
  - `cargo test -p <受影响crate>`
  - `cargo test`
- 未通过测试时，不宣称任务完成。

## 代码风格

- 保持类型与错误语义清晰，避免“先糊再补”。
- 公共 API 变更时，同步更新对应测试与文档。
- 输出给用户的说明尽量简洁，优先给可执行结论与下一步。
