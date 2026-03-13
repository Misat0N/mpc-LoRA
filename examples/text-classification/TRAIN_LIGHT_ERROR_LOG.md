# Training Error Log (Light Train Split)

## Context
- Command:
  - `./test_bert_base_128_comm_train_smoke.sh`
- Project path:
  - `examples/text-classification/`
- Mode:
  - 2-process CrypTen launcher

## Latest Failure (Recorded)
- Crash point:
  - `run_glue_private_train_smoke.py:663`
  - `private_eval_metric = private_metric.compute()`
- Error:
  - `ValueError: Evaluation module cache file doesn't exist. Please make sure that you call add or add_batch at least once before calling compute.`
- Process outcome:
  - `AssertionError: process 0 has non-zero exit code 1`

## Root Cause
- The eval loop filtered all batches before `add_batch()` (because of `--len_data 128` with dynamic padding and variable batch sequence lengths).
- Result: `private_metric` received zero samples, then `compute()` raised ValueError.

## Why Previous Logs Were Misleading
- Warnings such as ONNX shape inference warnings and CrypTen deprecation warnings were noisy but not the direct crash reason for this run.
- The direct blocker in this run is the empty metric buffer before `compute()`.

## Applied Changes

### 1) Restore runnable smoke flow
- Restored:
  - `examples/text-classification/run_glue_private_train_smoke.py`
  - `examples/text-classification/test_bert_base_128_comm_train_smoke.sh`
- Goal:
  - Keep smoke script focused on "can run end-to-end quickly" behavior.

### 2) Move formal lightweight training to a new file
- Added:
  - `examples/text-classification/run_glue_private_light_train.py`
  - `examples/text-classification/test_bert_base_128_comm_light_train.sh`

### 3) Add empty-metric guard in light train code
- In `run_glue_private_light_train.py`:
  - Added `_safe_metric_compute(...)`.
  - If `eval_steps == 0` or `plain_steps == 0`, skip `compute()` and return structured skipped result.
  - Added `skipped_by_len` counters in private/plain eval logs.

## Current Recommended Commands
- Smoke:
  - `./test_bert_base_128_comm_train_smoke.sh`
- Lightweight train + eval:
  - `./test_bert_base_128_comm_light_train.sh`

## Verification Checklist
- Smoke run should not fail at evaluate `compute()`.
- Light run should:
  - either produce metric values,
  - or explicitly log skipped metric due to zero added batches, without crashing.

---

## Follow-up Issue A (OOM During First Backward)
- Command:
  - `./test_bert_base_128_comm_light_train.sh`
- Crash point:
  - `run_glue_private_light_train.py:614`
  - `loss_enc.backward()`
- Error:
  - `torch.cuda.OutOfMemoryError: Tried to allocate 1.66 GiB ...`
- Secondary error:
  - `RuntimeError: ... Connection closed by peer ...` (rank0 OOM exit triggers rank1 comm failure)

### Root Cause
- Two MPC ranks were both using default `cuda` (effectively same GPU), causing memory contention.
- With BERT + MPC backward, memory and communication overhead are high even at batch size 1.

### Applied Fix
- Added GPU mapping argument in light train script:
  - `--gpu_ids` (example: `0,1`)
- Rank-based mapping:
  - each rank selects its own GPU by index
  - explicit `torch.cuda.set_device(...)`
- Startup script updated:
  - `test_bert_base_128_comm_light_train.sh` now passes `--gpu_ids 0,1`

---

## Follow-up Issue B (No `[train] step=... loss=...` After ~90 Minutes)
- Symptom:
  - training appears stuck in repeated logs:
    - `[DEBUG index_add_] ...`
    - `comm byte: 10.49 GB, round: 1520`
  - no visible `[train] step=... loss=...` for a long time.

### Diagnosis
- Run is spending very long inside first backward path, while third-party debug prints and communication cost prints flood stdout.
- Log flooding heavily degrades observability (and can degrade runtime due to console I/O overhead).

### Applied Fix
- Added log filtering in `run_glue_private_light_train.py`:
  - filter `index_add_` debug spam lines
  - filter repetitive `comm byte: ...` lines by default
- Added runtime toggles:
  - `--print_comm_cost` (off by default)
  - `--allow_spam_logs` (off by default)
- Adjusted default light smoke length:
  - `--max_train_steps` changed from `100` to `10` in `test_bert_base_128_comm_light_train.sh`

### Current Run Recommendation
- First validate end-to-end with:
  - `./test_bert_base_128_comm_light_train.sh`
- If still too slow in step0, reduce sequence length:
  - use `--len_data 64 --max_length 64` for sanity pass.

---

## Follow-up Issue C (Need a Guaranteed Pass-Through Configuration)
- Requirement:
  - Build an ultra-light training path that prioritizes "run through successfully" over metric quality.

### Integrated Design (Minimal Runtime Path)
- Added quick-run preset in:
  - `examples/text-classification/run_glue_private_light_train.py`
- New arguments added:
  - `--quick_run`
  - `--train_max_samples`
  - `--eval_max_samples`
  - `--train_classifier_only`
  - `--skip_private_eval`
  - `--skip_plain_eval`
- Existing stabilization arguments retained:
  - `--gpu_ids`
  - `--print_comm_cost` (default off)
  - `--allow_spam_logs` (default off)

### `--quick_run` Effective Behavior
- Forces a small and stable setup:
  - fixed short sequence (`max_length <= 32`, `len_data = max_length`)
  - one training step (`max_train_steps = 1`)
  - tiny sample cap (`train_max_samples <= 64`, `eval_max_samples <= 64`)
  - batch size 1
  - classifier-head-only training
  - skip private/plain eval by default
- Goal:
  - verify MPC training pipeline can complete backward+optimizer at least once.

### Additional Runtime Safety
- Added print filter to suppress spam outputs by default:
  - `index_add_` debug flood
  - repeated `comm byte: ...` lines
- Added synchronized decrypt logic:
  - avoid rank desync when eval is skipped but model export is needed.

### New One-Command Script
- Added:
  - `examples/text-classification/test_bert_base_comm_ultra_light.sh`
- Command:
  - `./test_bert_base_comm_ultra_light.sh`

---

## Follow-up Issue D (Need True MPC-LoRA Finetuning)
- Requirement:
  - Stop using only classifier-head smoke behavior and provide a real MPC-LoRA finetuning path with separate code and shell entrypoints.

### Core Cause
- Previous light/smoke scripts were optimized for "run-through" and debugging, not for strict LoRA adaptation workflow.
- We needed an independent script that:
  - injects LoRA modules into transformer linear layers,
  - freezes base weights,
  - trains LoRA params (optionally classifier head),
  - keeps MPC pipeline and evaluation/export behavior.

### Applied Changes
- Added new trainer:
  - `examples/text-classification/run_glue_private_mpc_lora_train.py`
- LoRA-specific implementation points:
  - Added `LoRALinear` wrapper and recursive module replacement.
  - Added trainable-parameter selector for LoRA (+optional classifier head).
  - Added LoRA args:
    - `--lora_r`
    - `--lora_alpha`
    - `--lora_dropout`
    - `--lora_target_modules`
    - `--freeze_classifier_head`
  - Added optimizer args:
    - `--learning_rate`
    - `--momentum`
  - Set train dataloader to `shuffle=True`.

### New Shell Scripts
- Standard MPC-LoRA run:
  - `examples/text-classification/test_bert_base_comm_mpc_lora.sh`
- Overnight preset:
  - `examples/text-classification/test_bert_base_comm_mpc_lora_overnight.sh`
- Nohup launcher:
  - `examples/text-classification/run_mpc_lora_overnight_nohup.sh`

### Recommended Usage
1. Quick standard run:
   - `bash test_bert_base_comm_mpc_lora.sh`
2. Overnight run:
   - `bash test_bert_base_comm_mpc_lora_overnight.sh`
3. Background overnight:
   - `bash run_mpc_lora_overnight_nohup.sh`

### Notes
- This is now a dedicated "true MPC-LoRA" path, separated from smoke and ultra-light scripts.
- Default LoRA target is `query,value` (can be changed by `LORA_TARGET_MODULES` or `--lora_target_modules`).

### Scale-Up Plan After Pass-Through
1. Increase `--max_train_steps` from `1` to `5/10`.
2. Raise `--max_length` from `32` to `64`.
3. Remove `--train_classifier_only` if full-model training is required.
4. Re-enable eval by removing skip flags after train loop is stable.

---

## Follow-up Issue E (LoRA Run Crashes at `to_pytorch()`)
- Command context:
  - `test_bert_base_comm_mpc_lora_overnight.sh`
- Crash point:
  - `run_glue_private_mpc_lora_train.py:1036`
  - `trained_model = private_model.to_pytorch()`
- Error:
  - `AttributeError: 'Identity' object has no attribute 'data'`
- Process outcome:
  - `AssertionError: process 0 has non-zero exit code 1`

### Observed Facts
- Distributed communicator startup completed normally.
- Dataset loading and tokenization completed.
- Failure happened after training/eval flow entered decrypt/export stage.
- This is not the earlier warning noise (ONNX shape inference / deprecation warnings).

### Root Cause Analysis
- Current LoRA wrapper introduces `nn.Identity()` as a submodule (`self.dropout`) inside `LoRALinear`.
- CrypTen `to_pytorch()` assumes submodules participating in its conversion path expose tensor-like `.data`.
- During conversion, it visits a module path where `Identity` is treated like a tensor-bearing module, causing:
  - `'Identity' object has no attribute 'data'`.

### Immediate Workaround (Keep Run Alive)
- Skip plaintext conversion/eval/export in overnight jobs:
  - pass `--skip_plain_eval`
  - avoid model export path that forces `to_pytorch()`
- If only private metric is needed, keep:
  - private training + private eval summary
- Practical shell toggle:
  - set `SKIP_PLAIN_EVAL=1` for overnight launch.

### Recommended Code Fix (Minimal and Safe)
1. In LoRA implementation, avoid registering `nn.Identity()` as module in the wrapped linear.
2. Replace with a functional/no-op path, e.g.:
   - keep `self.lora_dropout_p` (float) instead of `self.dropout = nn.Identity()`
   - in forward: apply dropout only when `p > 0`, otherwise use input directly.
3. Keep export guard:
   - wrap `to_pytorch()` with try/except and do not crash whole run when plain export fails;
   - still save `train_eval_summary.json` with explicit `"plain_eval_metric": {"skipped": true, "reason": "..."}`

### Validation Checklist After Fix
1. No `AttributeError: 'Identity' object has no attribute 'data'` in logs.
2. End-of-run has either:
   - saved plaintext model artifacts, or
   - explicit plain-eval/export skipped reason without process crash.
3. Final launcher exit code is 0.

### Applied Root-Fix (Implemented)
- File updated:
  - `examples/text-classification/run_glue_private_mpc_lora_train.py`
- Changes:
  1. Removed `nn.Identity()` submodule from LoRA wrapper:
     - replaced module-based no-op with functional dropout/no-op path.
  2. Replaced fragile conversion path:
     - previous: `trained_model = private_model.to_pytorch()`
     - now: decrypt + recover via `state_dict` into original PyTorch template model.
  3. Removed the hard fallback:
     - when `state_dict` recovery fails, do not fall back to `to_pytorch()`;
     - instead, record an explicit recovery error and keep summary export alive.

### Why This Works (Detailed)
1. `to_pytorch()` is a structure-dependent conversion path.
It relies on CrypTen's module traversal assumptions about submodule internals.
When custom wrapper modules include non-parameter helper modules (such as `nn.Identity`) in unexpected locations, conversion code can try to treat them like tensor-bearing leaves and access `.data`, which fails.
2. Functional dropout/no-op avoids problematic module registration.
By using a float probability (`self.lora_dropout_p`) and `F.dropout(...)` only in forward, we preserve behavior but remove `Identity` from module tree.
This makes the wrapped LoRA layer structurally closer to what CrypTen converter expects.
3. State-dict recovery is key-based tensor loading, not structure-heuristic conversion.
After decrypt, we read `private_model.state_dict()`, convert values to plain CPU tensors, and load them into the original PyTorch template model with `load_state_dict(..., strict=False)`.
This path matches parameters by key and shape, and is typically more robust for custom modules than a full graph/module conversion.
4. Failure is now explicit instead of silently switching conversion path.
If state-dict recovery cannot match parameters, code records `plain_model_recovery_failed` / `state_dict_recovery_no_match` for diagnosis instead of jumping back to a fragile converter path.

### Current Run Commands
1. Standard LoRA run (foreground):
`bash test_bert_base_comm_mpc_lora.sh`
2. Overnight preset (foreground):
`bash test_bert_base_comm_mpc_lora_overnight.sh`
3. Overnight preset (background):
`bash run_mpc_lora_overnight_nohup.sh`
4. Recommended for your 4-GPU machine (still 2-party MPC):
`GPU_IDS=0,1 bash test_bert_base_comm_mpc_lora_overnight.sh`
5. If one pair is occupied, switch pair:
`GPU_IDS=2,3 bash test_bert_base_comm_mpc_lora_overnight.sh`
6. Track latest background log:
`tail -f logs/mpc_lora_overnight_*.log`

---

## Follow-up Issue G (Plain Recovery Fails with `state_dict_recovery_no_match`)
- Symptom:
  - training completes and private metric is written,
  - but plaintext recovery still fails with:
    - `RuntimeError: state_dict_recovery_no_match`
- Typical summary output:

```json
{
  "train_steps": 1200,
  "private_eval_metric": {
    "accuracy": 0.455078125
  },
  "plain_eval_metric": {
    "skipped": true,
    "reason": "plain_model_recovery_failed",
    "error": "RuntimeError: state_dict_recovery_no_match"
  }
}
```

### Representative Log Excerpt

```text
ERROR:__mp_main__:[rank 0] state_dict recovery mapped 0 params. sample_ct_keys=[
  'bert.embeddings.word_embeddings.weight.data',
  'bert.embeddings.position_embeddings.weight.data',
  'bert.embeddings.token_type_embeddings.weight.data',
  'bert.embeddings.LayerNorm.weight.data',
  'bert.embeddings.LayerNorm.bias.data',
  'bert.encoder.layer.0.attention.self.query.base.weight.data',
  'bert.encoder.layer.0.attention.self.query.base.bias.data',
  'bert.encoder.layer.0.attention.self.query.lora_A.weight.data',
  'bert.encoder.layer.0.attention.self.query.lora_B.weight.data',
  'bert.encoder.layer.0.attention.self.key.weight.data'
]
```

### Root Cause
- CrypTen `state_dict()` keys may include wrapper suffixes like `.data`, `._tensor`, `.share`.
- The LoRA wrapper adds nested names such as `query.base.*`, `query.lora_A.*`, `query.lora_B.*`.
- If recovery logic compares raw CrypTen keys directly with plain PyTorch template keys, it can map zero parameters and fail.

### Applied Fix
- Added recursive value unwrapping:
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:254`
- Added serialized-key suffix stripping:
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:283`
- Added canonical key normalization and candidate matching:
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:295`
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:305`
- Added explicit guarded failure with sample keys:
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:365`

### Related Code Snippet

```python
def _to_torch_state_value(value, _seen=None):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if hasattr(value, "get_plain_text"):
        ...
    for attr_name in ("data", "_tensor", "share"):
        child = getattr(value, attr_name, None)
        if child is not None and child is not value:
            child_tensor = _to_torch_state_value(child, _seen=_seen)
            if child_tensor is not None:
                return child_tensor
    return None

def _strip_state_key_suffixes(key):
    suffixes = (".data", "._tensor", ".share")
    ...

if not mapped:
    sample_keys = list(ct_state.keys())[:10]
    logger.error("[rank %s] state_dict recovery mapped 0 params. sample_ct_keys=%s", rank, sample_keys)
    raise RuntimeError("state_dict_recovery_no_match")
```

This is the current root-level plaintext recovery path. If plain evaluation still fails, debugging should continue here.

---

## Appendix A (Logs + Code Anchors)

### A.1 Empty Metric Buffer Before `compute()`

Representative log:

```text
ValueError: Evaluation module cache file doesn't exist. Please make sure that you call `add` or `add_batch` at least once before calling `compute`.
AssertionError: process 0 has non-zero exit code 1
```

Relevant code:
- `examples/text-classification/run_glue_private_light_train.py:138`

```python
def _safe_metric_compute(metric, steps, rank, phase):
    if steps <= 0:
        logger.warning(
            "[rank %s] %s metric skipped: no batches were added (likely filtered by --len_data).",
            rank,
            phase,
        )
        return {"skipped": True, "reason": "no_batches", "steps": steps}
    return metric.compute()
```

### A.2 OOM During First Backward

Representative log:

```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.66 GiB ...
RuntimeError: [../third_party/gloo/gloo/transport/tcp/pair.cc:598] Connection closed by peer [127.0.1.1]:10124
```

Relevant code:
- `examples/text-classification/run_glue_private_light_train.py:648`
- `examples/text-classification/run_glue_private_light_train.py:653`

```python
local_gpu_id = gpu_id_list[rank % len(gpu_id_list)]
torch.cuda.set_device(local_gpu_id)
device = f"cuda:{local_gpu_id}"
```

This is the device-mapping fix that prevents both ranks from contending for the same default GPU.

### A.3 No Visible Train Progress Because Logs Were Flooded

Representative log:

```text
[DEBUG index_add_] dim = 1
[DEBUG index_add_] grad.shape: torch.Size([1, 128, 768])
comm byte: 10.49 GB, round: 1520
comm byte: 10.49 GB, round: 1520
comm byte: 10.49 GB, round: 1520
```

Relevant code:
- `examples/text-classification/run_glue_private_light_train.py:173`
- `examples/text-classification/run_glue_private_light_train.py:417`
- `examples/text-classification/run_glue_private_light_train.py:389`

```python
def _install_print_filter():
    ...

if not args.allow_spam_logs:
    _install_print_filter()

if args.quick_run:
    args.max_train_steps = 1
    args.eval_max_steps = 1
    args.train_classifier_only = True
```

### A.4 `to_pytorch()` / Plain Export Failure in MPC-LoRA

Representative log:

```text
AttributeError: 'Identity' object has no attribute 'data'
AssertionError: process 0 has non-zero exit code 1
```

Relevant code:
- `examples/text-classification/run_glue_private_mpc_lora_train.py:187`
- `examples/text-classification/run_glue_private_mpc_lora_train.py:1162`
- `examples/text-classification/run_glue_private_mpc_lora_train.py:1234`

```python
class LoRALinear(nn.Module):
    ...
    self.lora_dropout_p = float(dropout)
    ...
    x_lora = F.dropout(x, p=self.lora_dropout_p, training=self.training) if self.lora_dropout_p > 0 else x

need_decrypt = (not args.skip_plain_eval) or (args.output_dir is not None)
...
logger.warning("[save] skip trained model export: plaintext model unavailable")
```

The important fix here is structural: avoid `nn.Identity()` in the LoRA wrapper and make plaintext export best-effort instead of fatal.

---

# 中文翻译版（Chinese Translation）

## 训练错误日志（Light Train 拆分）

## 背景
- 命令：
  - `./test_bert_base_128_comm_train_smoke.sh`
- 项目路径：
  - `examples/text-classification/`
- 模式：
  - 2 进程 CrypTen launcher

## 最近一次失败（已记录）
- 崩溃点：
  - `run_glue_private_train_smoke.py:663`
  - `private_eval_metric = private_metric.compute()`
- 错误：
  - `ValueError: Evaluation module cache file doesn't exist. Please make sure that you call add or add_batch at least once before calling compute.`
- 进程结果：
  - `AssertionError: process 0 has non-zero exit code 1`

## 根因
- 在 eval 循环中，所有 batch 在 `add_batch()` 前都被过滤掉了（`--len_data 128` + 动态 padding + 序列长度不一致）。
- 结果：`private_metric` 没有收到任何样本，随后 `compute()` 抛出 ValueError。

## 为什么之前的日志容易误导
- ONNX shape inference 警告和 CrypTen deprecation 警告噪声很多，但不是这次崩溃的直接原因。
- 这次运行的直接阻塞点是：`compute()` 前 metric 缓冲区为空。

## 已应用修改

### 1) 恢复可运行的 smoke 流程
- 已恢复：
  - `examples/text-classification/run_glue_private_train_smoke.py`
  - `examples/text-classification/test_bert_base_128_comm_train_smoke.sh`
- 目标：
  - 保持 smoke 脚本聚焦于“快速端到端可运行”。

### 2) 将正式轻量训练迁移到新文件
- 新增：
  - `examples/text-classification/run_glue_private_light_train.py`
  - `examples/text-classification/test_bert_base_128_comm_light_train.sh`

### 3) 在 light train 中增加空 metric 保护
- 在 `run_glue_private_light_train.py` 中：
  - 新增 `_safe_metric_compute(...)`。
  - 若 `eval_steps == 0` 或 `plain_steps == 0`，跳过 `compute()` 并返回结构化的 skipped 结果。
  - 在 private/plain eval 日志中新增 `skipped_by_len` 计数。

## 当前推荐命令
- Smoke：
  - `./test_bert_base_128_comm_train_smoke.sh`
- 轻量训练 + 评估：
  - `./test_bert_base_128_comm_light_train.sh`

## 验证清单
- Smoke 运行不应在 evaluate `compute()` 处失败。
- Light 运行应当：
  - 要么产出 metric 数值，
  - 要么明确记录因零 batch 导致的 metric skipped，且不崩溃。

---

## 后续问题 A（首次 backward 期间 OOM）
- 命令：
  - `./test_bert_base_128_comm_light_train.sh`
- 崩溃点：
  - `run_glue_private_light_train.py:614`
  - `loss_enc.backward()`
- 错误：
  - `torch.cuda.OutOfMemoryError: Tried to allocate 1.66 GiB ...`
- 次级错误：
  - `RuntimeError: ... Connection closed by peer ...`（rank0 OOM 退出触发 rank1 通信失败）

### 根因
- 两个 MPC rank 都使用了默认 `cuda`（等效落在同一张 GPU），造成显存争用。
- BERT + MPC backward 的显存与通信开销很高，即使 batch size=1 也可能 OOM。

### 已应用修复
- 在 light train 脚本新增 GPU 映射参数：
  - `--gpu_ids`（示例：`0,1`）
- 基于 rank 的映射：
  - 每个 rank 按索引选择各自 GPU
  - 显式 `torch.cuda.set_device(...)`
- 启动脚本已更新：
  - `test_bert_base_128_comm_light_train.sh` 现在传递 `--gpu_ids 0,1`

---

## 后续问题 B（约 90 分钟后仍无 `[train] step=... loss=...`）
- 现象：
  - 训练看起来卡在重复日志：
    - `[DEBUG index_add_] ...`
    - `comm byte: 10.49 GB, round: 1520`
  - 长时间看不到 `[train] step=... loss=...`。

### 诊断
- 运行在第一次 backward 路径内部耗时非常长，同时第三方 debug 输出和通信代价打印大量刷屏。
- 日志洪泛严重降低可观测性（并可能因控制台 I/O 导致实际运行进一步变慢）。

### 已应用修复
- 在 `run_glue_private_light_train.py` 增加日志过滤：
  - 默认过滤 `index_add_` 调试刷屏
  - 默认过滤重复 `comm byte: ...`
- 增加运行开关：
  - `--print_comm_cost`（默认关闭）
  - `--allow_spam_logs`（默认关闭）
- 调整 light smoke 默认长度：
  - `test_bert_base_128_comm_light_train.sh` 中 `--max_train_steps` 从 `100` 改为 `10`

### 当前运行建议
- 先用下面命令做端到端验证：
  - `./test_bert_base_128_comm_light_train.sh`
- 若 step0 仍过慢，先降低序列长度：
  - `--len_data 64 --max_length 64`

---

## 后续问题 C（需要一个保证可跑通的配置）
- 需求：
  - 构建 ultra-light 训练路径，优先保证“可以跑通”，而非指标质量。

### 集成设计（最小运行路径）
- 在以下文件加入 quick-run 预设：
  - `examples/text-classification/run_glue_private_light_train.py`
- 新增参数：
  - `--quick_run`
  - `--train_max_samples`
  - `--eval_max_samples`
  - `--train_classifier_only`
  - `--skip_private_eval`
  - `--skip_plain_eval`
- 保留稳定性参数：
  - `--gpu_ids`
  - `--print_comm_cost`（默认关闭）
  - `--allow_spam_logs`（默认关闭）

### `--quick_run` 生效行为
- 强制使用小而稳定的配置：
  - 固定短序列（`max_length <= 32`，`len_data = max_length`）
  - 单步训练（`max_train_steps = 1`）
  - 小样本上限（`train_max_samples <= 64`，`eval_max_samples <= 64`）
  - batch size 为 1
  - 仅训练分类头
  - 默认跳过 private/plain eval
- 目标：
  - 验证 MPC 训练链路至少能完成一次 backward + optimizer。

### 额外运行安全措施
- 默认启用 print 过滤：
  - `index_add_` 调试刷屏
  - 重复 `comm byte: ...`
- 增加同步 decrypt 逻辑：
  - 当跳过 eval 但需要导出模型时，避免 rank 间失同步。

### 新的一键脚本
- 新增：
  - `examples/text-classification/test_bert_base_comm_ultra_light.sh`
- 命令：
  - `./test_bert_base_comm_ultra_light.sh`

---

## 后续问题 D（需要真正的 MPC-LoRA 微调）
- 需求：
  - 不再使用仅分类头 smoke 行为，提供独立的真实 MPC-LoRA 微调路径（代码与 shell 入口分离）。

### 核心原因
- 之前 light/smoke 脚本主要为“跑通”和调试优化，并非严格 LoRA 适配工作流。
- 需要独立脚本来完成：
  - 向 transformer 线性层注入 LoRA；
  - 冻结基础权重；
  - 训练 LoRA 参数（可选训练分类头）；
  - 维持 MPC 训练、评估和导出流程。

### 已应用修改
- 新增训练器：
  - `examples/text-classification/run_glue_private_mpc_lora_train.py`
- LoRA 相关实现点：
  - 增加 `LoRALinear` 包装与递归替换。
  - 增加 LoRA 可训练参数选择（可选带分类头）。
  - 增加 LoRA 参数：
    - `--lora_r`
    - `--lora_alpha`
    - `--lora_dropout`
    - `--lora_target_modules`
    - `--freeze_classifier_head`
  - 增加优化器参数：
    - `--learning_rate`
    - `--momentum`
  - 训练 dataloader 设置为 `shuffle=True`。

### 新增 shell 脚本
- 标准 MPC-LoRA 训练：
  - `examples/text-classification/test_bert_base_comm_mpc_lora.sh`
- 过夜预设：
  - `examples/text-classification/test_bert_base_comm_mpc_lora_overnight.sh`
- nohup 启动器：
  - `examples/text-classification/run_mpc_lora_overnight_nohup.sh`

### 推荐使用方式
1. 标准快速运行：
   - `bash test_bert_base_comm_mpc_lora.sh`
2. 过夜运行：
   - `bash test_bert_base_comm_mpc_lora_overnight.sh`
3. 后台过夜运行：
   - `bash run_mpc_lora_overnight_nohup.sh`

### 说明
- 现在已有独立的“真实 MPC-LoRA”路径，与 smoke/ultra-light 脚本分离。
- 默认 LoRA target 为 `query,value`（可通过 `LORA_TARGET_MODULES` 或 `--lora_target_modules` 修改）。

### 跑通后的逐步放量计划
1. 将 `--max_train_steps` 从 `1` 提升到 `5/10`。
2. 将 `--max_length` 从 `32` 提升到 `64`。
3. 若需要全参数行为，移除 `--train_classifier_only`。
4. 训练环节稳定后，再移除 skip 参数开启完整评估。

---

## 后续问题 E（LoRA 运行在 `to_pytorch()` 崩溃）
- 命令上下文：
  - `test_bert_base_comm_mpc_lora_overnight.sh`
- 崩溃点：
  - `run_glue_private_mpc_lora_train.py:1036`
  - `trained_model = private_model.to_pytorch()`
- 错误：
  - `AttributeError: 'Identity' object has no attribute 'data'`
- 进程结果：
  - `AssertionError: process 0 has non-zero exit code 1`

### 观察事实
- 分布式通信初始化完成。
- 数据加载与 tokenization 完成。
- 失败发生在训练/评估后的 decrypt/export 阶段。
- 该失败不是前述 ONNX/deprecation 类警告引起。

### 根因分析
- 现有 LoRA wrapper 在 `LoRALinear` 中引入了 `nn.Identity()` 子模块（`self.dropout`）。
- CrypTen `to_pytorch()` 的转换路径会假设相关子模块可暴露类似 tensor 的 `.data`。
- 在模块遍历中，`Identity` 被当作应有 tensor 数据的节点处理，触发：
  - `'Identity' object has no attribute 'data'`。

### 临时绕过方案（先保证任务不断）
- 在过夜任务中跳过明文转换/评估/导出：
  - 传 `--skip_plain_eval`
  - 避免触发强依赖 `to_pytorch()` 的导出路径
- 若只关心 private 指标，可保留：
  - private training + private eval summary
- shell 快速开关：
  - `SKIP_PLAIN_EVAL=1`

### 建议代码修复（最小且安全）
1. 在 LoRA 实现中避免把 `nn.Identity()` 注册为 wrapped linear 的子模块。
2. 改为函数式 dropout/no-op：
   - 使用 `self.lora_dropout_p`（float），而不是 `self.dropout = nn.Identity()`
   - 在 forward 中仅当 `p > 0` 时应用 dropout，否则直接走输入。
3. 保留导出兜底：
   - 对 `to_pytorch()` 增加 try/except，避免明文导出失败导致整任务崩溃；
   - 仍保存 `train_eval_summary.json`，并记录 `"plain_eval_metric": {"skipped": true, "reason": "..."}`。

### 修复后验证清单
1. 日志中不再出现 `AttributeError: 'Identity' object has no attribute 'data'`。
2. 任务结束时应满足以下之一：
   - 明文模型产物已保存；
   - 明确记录 plain eval/export skipped 原因且不崩溃。
3. launcher 最终退出码为 0。

### 已实施的根因修复
- 已更新文件：
  - `examples/text-classification/run_glue_private_mpc_lora_train.py`
- 修改内容：
  1. 移除 LoRA wrapper 中的 `nn.Identity()` 子模块；
     - 替换为函数式 dropout/no-op。
  2. 替换脆弱转换路径：
     - 旧：`trained_model = private_model.to_pytorch()`
     - 新：decrypt 后通过 `state_dict` 回填到原始 PyTorch 模型模板。
  3. 去掉硬回退：
     - 当 `state_dict` 恢复失败时，不再回退到 `to_pytorch()`；
     - 改为明确记录恢复失败原因，并继续写出 summary。

### 为什么这样有效（详细说明）
1. `to_pytorch()` 是结构依赖型转换路径。  
它依赖 CrypTen 对子模块内部结构的遍历假设。自定义 wrapper 若包含非参数辅助模块（如 `nn.Identity`）且位置不在预期路径上，转换代码可能把它当成张量叶子并访问 `.data`，从而失败。
2. 函数式 dropout/no-op 可避免问题模块注册。  
使用浮点概率（`self.lora_dropout_p`）并在 forward 内调用 `F.dropout(...)`，可保持行为一致，同时把 `Identity` 从模块树中移除，结构更符合 CrypTen 转换器预期。
3. `state_dict` 恢复是“按 key/shape 对齐”的张量加载，不依赖模块结构启发式。  
decrypt 后读取 `private_model.state_dict()`，将值转为 CPU 明文 tensor，再 `load_state_dict(..., strict=False)` 回填到原始 PyTorch 模型。对于自定义模块，这通常比完整图/模块转换更稳健。
4. 失败路径现在是显式可诊断的。  
如果 `state_dict` 恢复无法匹配，代码会记录 `plain_model_recovery_failed` / `state_dict_recovery_no_match`，而不是再次跳回脆弱的转换路径。

### 当前运行命令
1. 标准 LoRA（前台）：
`bash test_bert_base_comm_mpc_lora.sh`
2. 过夜预设（前台）：
`bash test_bert_base_comm_mpc_lora_overnight.sh`
3. 过夜预设（后台）：
`bash run_mpc_lora_overnight_nohup.sh`
4. 4 卡机器推荐（仍是 2-party MPC）：
`GPU_IDS=0,1 bash test_bert_base_comm_mpc_lora_overnight.sh`
5. 一对卡被占用时切换：
`GPU_IDS=2,3 bash test_bert_base_comm_mpc_lora_overnight.sh`
6. 跟踪后台日志：
`tail -f logs/mpc_lora_overnight_*.log`

---

## Follow-up Issue F (Empty Output Folder After Foreground Run)
- Symptom:
  - `output_dir` exists, but expected artifacts are missing.
  - sometimes `trained_model/` is empty.

### Root Cause
- Plain-model recovery/export could fail before summary writing.
- Previous flow created/entered save path assuming plaintext export succeeds, which can lead to confusing empty directories.

### Applied Fix
- File:
  - `examples/text-classification/run_glue_private_mpc_lora_train.py`
- Changes:
  1. Wrap plaintext model recovery in `try/except`.
  2. On recovery failure:
     - keep process alive,
     - set `plain_eval_metric` to skipped with explicit error string.
  3. Always write `train_eval_summary.json` for diagnosis.
  4. Only create/save `trained_model/` when plaintext model is actually available.
  5. Log explicit save skip reason:
     - `[save] skip trained model export: plaintext model unavailable`
  6. Strengthen state-dict recovery:
     - canonicalize CrypTen-style keys (`_modules/_parameters/_buffers`) before matching
     - match by candidate keys + shape check + dtype cast
     - remove hard fallback to `to_pytorch()` to avoid repeating `Identity.data` crash
  7. Handle serialized key/value wrappers from CrypTen:
     - strip tail suffixes like `.data` / `._tensor` / `.share` before key matching
     - recursively unwrap state values through `data/_tensor/share` and `get_plain_text()`

### How To Read Result Correctly
1. First check `train_eval_summary.json` under run `output_dir`.
2. If `plain_eval_metric.reason == plain_model_recovery_failed`, export was skipped by design.
3. Use logs to inspect root exception and decide whether to retry with updated script/environment.

---

## 后续问题 G（明文恢复失败：`state_dict_recovery_no_match`）
- 现象：
  - 训练能够正常结束，并且能写出 private metric，
  - 但明文恢复仍然失败，报：
    - `RuntimeError: state_dict_recovery_no_match`
- 典型 summary 输出：

```json
{
  "train_steps": 1200,
  "private_eval_metric": {
    "accuracy": 0.455078125
  },
  "plain_eval_metric": {
    "skipped": true,
    "reason": "plain_model_recovery_failed",
    "error": "RuntimeError: state_dict_recovery_no_match"
  }
}
```

### 代表性日志摘录

```text
ERROR:__mp_main__:[rank 0] state_dict recovery mapped 0 params. sample_ct_keys=[
  'bert.embeddings.word_embeddings.weight.data',
  'bert.embeddings.position_embeddings.weight.data',
  'bert.embeddings.token_type_embeddings.weight.data',
  'bert.embeddings.LayerNorm.weight.data',
  'bert.embeddings.LayerNorm.bias.data',
  'bert.encoder.layer.0.attention.self.query.base.weight.data',
  'bert.encoder.layer.0.attention.self.query.base.bias.data',
  'bert.encoder.layer.0.attention.self.query.lora_A.weight.data',
  'bert.encoder.layer.0.attention.self.query.lora_B.weight.data',
  'bert.encoder.layer.0.attention.self.key.weight.data'
]
```

### 根因
- CrypTen `state_dict()` 的 key 可能带有 `.data`、`._tensor`、`.share` 之类的包装后缀。
- LoRA 包装层又引入了 `query.base.*`、`query.lora_A.*`、`query.lora_B.*` 这样的嵌套名字。
- 如果恢复逻辑直接拿原始 CrypTen key 去和明文 PyTorch 模板模型的参数名对比，就可能一个都匹配不上，最终映射数为 0。

### 已应用修复
- 增加递归 value 解包：
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:254`
- 增加序列化后缀剥离：
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:283`
- 增加 canonical key 归一化与候选 key 匹配：
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:295`
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:305`
- 增加带 sample keys 的显式失败日志：
  - `examples/text-classification/run_glue_private_mpc_lora_train.py:365`

### 相关代码片段

```python
def _to_torch_state_value(value, _seen=None):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if hasattr(value, "get_plain_text"):
        ...
    for attr_name in ("data", "_tensor", "share"):
        child = getattr(value, attr_name, None)
        if child is not None and child is not value:
            child_tensor = _to_torch_state_value(child, _seen=_seen)
            if child_tensor is not None:
                return child_tensor
    return None

def _strip_state_key_suffixes(key):
    suffixes = (".data", "._tensor", ".share")
    ...

if not mapped:
    sample_keys = list(ct_state.keys())[:10]
    logger.error("[rank %s] state_dict recovery mapped 0 params. sample_ct_keys=%s", rank, sample_keys)
    raise RuntimeError("state_dict_recovery_no_match")
```

这就是当前明文恢复链路的根部逻辑。如果 plain eval 还失败，应该继续从这里追。

---

## 附录 A（日志摘录 + 代码锚点）

### A.1 `compute()` 前 metric 为空

代表性日志：

```text
ValueError: Evaluation module cache file doesn't exist. Please make sure that you call `add` or `add_batch` at least once before calling `compute`.
AssertionError: process 0 has non-zero exit code 1
```

相关代码：
- `examples/text-classification/run_glue_private_light_train.py:138`

```python
def _safe_metric_compute(metric, steps, rank, phase):
    if steps <= 0:
        logger.warning(
            "[rank %s] %s metric skipped: no batches were added (likely filtered by --len_data).",
            rank,
            phase,
        )
        return {"skipped": True, "reason": "no_batches", "steps": steps}
    return metric.compute()
```

### A.2 首次 backward 就 OOM

代表性日志：

```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.66 GiB ...
RuntimeError: [../third_party/gloo/gloo/transport/tcp/pair.cc:598] Connection closed by peer [127.0.1.1]:10124
```

相关代码：
- `examples/text-classification/run_glue_private_light_train.py:648`
- `examples/text-classification/run_glue_private_light_train.py:653`

```python
local_gpu_id = gpu_id_list[rank % len(gpu_id_list)]
torch.cuda.set_device(local_gpu_id)
device = f"cuda:{local_gpu_id}"
```

这段代码就是显卡映射修复，避免两个 rank 默认抢同一张卡。

### A.3 没有 `[train] step=...`，只有刷屏日志

代表性日志：

```text
[DEBUG index_add_] dim = 1
[DEBUG index_add_] grad.shape: torch.Size([1, 128, 768])
comm byte: 10.49 GB, round: 1520
comm byte: 10.49 GB, round: 1520
comm byte: 10.49 GB, round: 1520
```

相关代码：
- `examples/text-classification/run_glue_private_light_train.py:173`
- `examples/text-classification/run_glue_private_light_train.py:417`
- `examples/text-classification/run_glue_private_light_train.py:389`

```python
def _install_print_filter():
    ...

if not args.allow_spam_logs:
    _install_print_filter()

if args.quick_run:
    args.max_train_steps = 1
    args.eval_max_steps = 1
    args.train_classifier_only = True
```

这几段代码的作用分别是：默认抑制日志洪泛，以及在必要时强制进入最小可跑通配置。

### A.4 MPC-LoRA 中 `to_pytorch()` / 明文导出失败

代表性日志：

```text
AttributeError: 'Identity' object has no attribute 'data'
AssertionError: process 0 has non-zero exit code 1
```

相关代码：
- `examples/text-classification/run_glue_private_mpc_lora_train.py:187`
- `examples/text-classification/run_glue_private_mpc_lora_train.py:1162`
- `examples/text-classification/run_glue_private_mpc_lora_train.py:1234`

```python
class LoRALinear(nn.Module):
    ...
    self.lora_dropout_p = float(dropout)
    ...
    x_lora = F.dropout(x, p=self.lora_dropout_p, training=self.training) if self.lora_dropout_p > 0 else x

need_decrypt = (not args.skip_plain_eval) or (args.output_dir is not None)
...
logger.warning("[save] skip trained model export: plaintext model unavailable")
```

这里的关键修复不是“跳过错误”，而是两件事：
1. 不再在 LoRA 包装层里引入 `nn.Identity()` 这种会干扰转换的子模块；
2. 把明文导出改成 best-effort，而不是一失败就把整个训练任务打死。
