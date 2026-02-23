# Train Smoke 报错分析与最小修复记录

## 1. 问题现象

运行：

```bash
./test_bert_base_128_comm_train_smoke.sh
```

在第 2 步（`step=001`）反向传播失败，关键日志：

- `ERROR: [rank 0] step=001 backward_failed ... python_recursion_limit: 1000`
- `loss_enc.backward()`
- `Previous line repeated 969 more times`
- `RecursionError: maximum recursion depth exceeded`

最终由 launcher 报：

- `AssertionError: process 0 has non-zero exit code 1`

## 2. 核心原因（根因）

根因是 **CrypTen 反向传播使用 Python 递归遍历计算图**，在当前训练图复杂度下触发了默认递归上限（1000），导致 `loss_enc.backward()` 在 `step=001` 爆栈。

这不是“目录迁移后路径错误”导致的问题。证据：

- 模型、tokenizer、权重均成功加载；
- 两个 rank 都能完成前向（`step=000` 可输出 loss）；
- 失败点稳定出现在 backward，且错误类型是 `RecursionError`。

日志中的 `cfg.debug` 异常是递归爆栈后的连带异常，不是首因。

## 3. 最小修改方案（已落地）

文件：`examples/text-classification/run_glue_private_train_smoke.py`

### 修改 A：提高递归上限（直接针对根因）

- 在 `main()` 启动后将 `sys.setrecursionlimit(...)` 提升到至少 `20000`
- 同时打印修改前后值，便于确认运行态

目的：避免 `loss_enc.backward()` 在深图上触发默认递归上限。

### 修改 B：统一训练模式/优化器初始化时机（减少干扰因素）

- 将 `private_model.train()` 和 `optimizer = ct.optim.SGD(...)` 移到训练循环之前
- 删除“step=0 后才切 train / 才初始化 optimizer”的逻辑

目的：避免 step0/step1 行为不一致引入额外变量，便于定位和复现实验。

### 修改 C：避免重复 `ct.init()`（降噪）

- 若子进程已初始化（launcher 已调用），则跳过 `ct.init()` 并记录 warning

目的：减少“already initialized”噪声，不改变主逻辑。

## 4. 为什么这三处是“最小必要改动”

- 只改了与报错直接相关的控制逻辑，不改模型结构、不改数据、不改损失函数；
- 优先保证“先能稳定跑过 backward”，再谈精度或训练策略；
- 保留现有诊断日志体系，便于继续排查后续问题。

## 5. 建议验证步骤

1. 先跑冒烟脚本，观察是否仍在 `step=001 backward` 处崩溃。  
2. 若通过，逐步把 `max_steps` 从 `5` 增加到 `10/20`，确认稳定性。  
3. 若仍失败，保留当前日志并重点看：
   - `python recursion limit old -> new`
   - `backward_failed` 的 `loss_snapshot` 与 `cfg_snapshot`

