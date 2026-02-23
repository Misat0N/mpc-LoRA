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

