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
  3. Kept compatibility fallback:
     - fallback to `to_pytorch()` only if `state_dict` recovery cannot match parameters.
