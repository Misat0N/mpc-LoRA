# Train Smoke Error Analysis and Minimal Fix Record (English Version)

## 1. Symptom

Run:

```bash
./test_bert_base_128_comm_train_smoke.sh
```

Backward failed at step 2 (`step=001`). Key log lines:

- `ERROR: [rank 0] step=001 backward_failed ... python_recursion_limit: 1000`
- `loss_enc.backward()`
- `Previous line repeated 969 more times`
- `RecursionError: maximum recursion depth exceeded`

Launcher finally reported:

- `AssertionError: process 0 has non-zero exit code 1`

## 2. Core Root Cause

The root cause is that **CrypTen backward traverses the computation graph recursively in Python**. Under the current graph complexity, it hit the default recursion limit (1000), so `loss_enc.backward()` overflowed at `step=001`.

This was not caused by path issues after directory migration. Evidence:

- model, tokenizer, and checkpoints all loaded successfully;
- both ranks completed forward (`step=000` produced loss);
- failure consistently occurred in backward and the error type was `RecursionError`.

The `cfg.debug` exception in logs was a secondary exception after recursion overflow, not the first cause.

## 3. Minimal Fix Plan (Implemented)

File: `examples/text-classification/run_glue_private_train_smoke.py`

### Change A: Increase recursion limit (directly targets root cause)

- Raise `sys.setrecursionlimit(...)` to at least `20000` after `main()` starts.
- Print old/new limits for runtime verification.

Purpose: prevent `loss_enc.backward()` from hitting Python's default recursion limit on deep graphs.

### Change B: Unify train mode and optimizer initialization timing (reduce confounders)

- Move `private_model.train()` and `optimizer = ct.optim.SGD(...)` before the training loop.
- Remove logic that only switched to train / initialized optimizer after step0.

Purpose: avoid step0/step1 behavior mismatch and reduce extra variables during diagnosis.

### Change C: Avoid duplicate `ct.init()` (noise reduction)

- If subprocess is already initialized by launcher, skip `ct.init()` and log a warning.

Purpose: reduce "already initialized" noise without changing main logic.

## 4. Why These Are "Minimal Necessary Changes"

- Only changed control flow directly related to the crash.
- Did not change model architecture, data, or loss function.
- Prioritized "stable backward pass first" before accuracy or strategy tuning.
- Kept existing diagnostic logging for further troubleshooting.

## 5. Recommended Validation Steps

1. Re-run smoke script and check whether it still crashes at `step=001 backward`.
2. If it passes, gradually increase `max_steps` from `5` to `10/20` to verify stability.
3. If it still fails, keep the current log and focus on:
   - `python recursion limit old -> new`
   - `loss_snapshot` and `cfg_snapshot` in `backward_failed`.

## 6. Short Training + Evaluation After Upgrade (Step 2 + Step 3)

`run_glue_private_train_smoke.py` is extended to:

- short private training (using `train` split)
- private evaluation after training (`validation`)
- plaintext evaluation after training (`validation`)
- export trained model to `output_dir/trained_model`
- write summary to `output_dir/train_eval_summary.json`

New key arguments:

- `--per_device_train_batch_size`: training batch size
- `--max_train_steps`: short-training steps
- `--log_every_steps`: training log interval
- `--eval_max_steps`: evaluation step limit (`-1` means full set)

Recommended command (already in `test_bert_base_128_comm_train_smoke.sh`):

```bash
./test_bert_base_128_comm_train_smoke.sh
```

## 7. Representative Logs and Related Code

### 7.1 Representative Crash Log

```text
ERROR:__mp_main__:[rank 0] step=001 backward_failed cfg={'top_level_keys': [...], 'precision_bits': 16, 'validation_mode': False} loss={'loss_type': 'MPCTensor', 'loss_shape': (), 'python_recursion_limit': 1000, 'grad_fn': 'type', 'children_len': 1}
Traceback (most recent call last):
  File ".../run_glue_private_train_smoke.py", line 579, in main
    loss_enc.backward()
  File ".../crypten/cryptensor.py", line 268, in backward
    child.backward(grad_input=grad[idx], top_node=False)
  [Previous line repeated 969 more times]
RecursionError: maximum recursion depth exceeded
AssertionError: process 0 has non-zero exit code 1
```

This log proves the failure was inside recursive backward traversal, not model loading or dataset preparation.

### 7.2 Related Code Snippet: Raise Recursion Limit Early

File reference:
- `examples/text-classification/run_glue_private_train_smoke.py:264`

```python
old_recursion_limit = sys.getrecursionlimit()
target_recursion_limit = max(old_recursion_limit, 20000)
if target_recursion_limit != old_recursion_limit:
    sys.setrecursionlimit(target_recursion_limit)
logger.info("python recursion limit %s -> %s", old_recursion_limit, sys.getrecursionlimit())
```

This change directly targets the overflow shown in the crash log.

### 7.3 Related Code Snippet: Stabilize Train Setup Before Loop

File references:
- `examples/text-classification/run_glue_private_train_smoke.py:451`
- `examples/text-classification/run_glue_private_train_smoke.py:453`

```python
dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to(device)
private_model.train()
lr = 0.01
optimizer = ct.optim.SGD(private_model.parameters(), lr=lr)
```

This ensures step 0 and step 1 run under the same model mode and optimizer lifecycle.

### 7.4 Related Code Snippet: Backward Failure Capture

File reference:
- `examples/text-classification/run_glue_private_train_smoke.py:579`

```python
logger.info("[rank %s] step=%03d backward_start", rank, step)
try:
    loss_enc.backward()
except Exception:
    logger.exception(
        "[rank %s] step=%03d backward_failed cfg=%s loss=%s",
        rank,
        step,
        _cfg_snapshot(),
        _loss_snapshot(loss_enc),
    )
    raise
```

This is why the crash log includes `cfg` and `loss_snapshot`, which are the most useful fields when diagnosing deep-graph backward failures.

---

# 中文翻译版（Chinese Translation）

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

## 6. 升级后的短程训练与评估（Step 2 + Step 3）

`run_glue_private_train_smoke.py` 已扩展为：

- 短程私有训练（使用 `train` split）
- 训练后私有评估（validation）
- 训练后明文评估（validation）
- 导出训练后模型到 `output_dir/trained_model`
- 记录汇总到 `output_dir/train_eval_summary.json`

新增关键参数：

- `--per_device_train_batch_size`：训练 batch size
- `--max_train_steps`：短程训练步数
- `--log_every_steps`：训练日志间隔
- `--eval_max_steps`：评估步数上限（`-1` 为全量）

推荐命令（已写入 `test_bert_base_128_comm_train_smoke.sh`）：

```bash
./test_bert_base_128_comm_train_smoke.sh
```

## 7. 代表性日志与相关代码

### 7.1 代表性报错日志

```text
ERROR:__mp_main__:[rank 0] step=001 backward_failed cfg={'top_level_keys': [...], 'precision_bits': 16, 'validation_mode': False} loss={'loss_type': 'MPCTensor', 'loss_shape': (), 'python_recursion_limit': 1000, 'grad_fn': 'type', 'children_len': 1}
Traceback (most recent call last):
  File ".../run_glue_private_train_smoke.py", line 579, in main
    loss_enc.backward()
  File ".../crypten/cryptensor.py", line 268, in backward
    child.backward(grad_input=grad[idx], top_node=False)
  [Previous line repeated 969 more times]
RecursionError: maximum recursion depth exceeded
AssertionError: process 0 has non-zero exit code 1
```

这段日志说明失败发生在递归 backward 内部，而不是数据集加载或模型加载阶段。

### 7.2 相关代码片段：一开始就提高递归上限

文件位置：
- `examples/text-classification/run_glue_private_train_smoke.py:264`

```python
old_recursion_limit = sys.getrecursionlimit()
target_recursion_limit = max(old_recursion_limit, 20000)
if target_recursion_limit != old_recursion_limit:
    sys.setrecursionlimit(target_recursion_limit)
logger.info("python recursion limit %s -> %s", old_recursion_limit, sys.getrecursionlimit())
```

这段修改直接对应报错日志中的 `python_recursion_limit: 1000`。

### 7.3 相关代码片段：在训练循环前固定训练模式和优化器

文件位置：
- `examples/text-classification/run_glue_private_train_smoke.py:451`
- `examples/text-classification/run_glue_private_train_smoke.py:453`

```python
dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to(device)
private_model.train()
lr = 0.01
optimizer = ct.optim.SGD(private_model.parameters(), lr=lr)
```

这样做的作用是让 step0 和 step1 处在同一套训练状态下，减少额外干扰变量。

### 7.4 相关代码片段：backward 失败时的日志包装

文件位置：
- `examples/text-classification/run_glue_private_train_smoke.py:579`

```python
logger.info("[rank %s] step=%03d backward_start", rank, step)
try:
    loss_enc.backward()
except Exception:
    logger.exception(
        "[rank %s] step=%03d backward_failed cfg=%s loss=%s",
        rank,
        step,
        _cfg_snapshot(),
        _loss_snapshot(loss_enc),
    )
    raise
```

这就是为什么日志里会带出 `cfg` 和 `loss_snapshot`，也是后续排查深图 backward 问题最关键的上下文。
