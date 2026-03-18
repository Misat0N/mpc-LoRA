# Beaver Reuse Experiment Fix Log

## Current implementation path

The current experimental reuse path is:

1. `crypten/nn/module.py`
   - `Linear.forward()` attaches a layer tag with `use_layer_tag(...)`.
2. `crypten/gradients.py`
   - `AutogradMatMul.forward()` creates the forward beaver tag.
   - `AutogradMatMul.backward()` creates `backward_dX` and `backward_dW` tags.
3. `crypten/common/reuse_context.py`
   - carries step id, layer tag, and current beaver tag across the call chain.
4. `crypten/mpc/primitives/beaver.py`
   - `__beaver_protocol()` dispatches to `_beaver_with_reuse(...)` when
     `cfg.mpc.experimental_reuse_mask=True` and the op is in `reuse_op_types`.
5. `crypten/mpc/primitives/beaver_reuse.py`
   - owns the step-scoped cache for `A/B/C` and opened residuals.
6. `crypten/mpc/provider/tfp_provider.py` and `crypten/mpc/provider/ttp_provider.py`
   - keep the baseline provider path and increment `triple_generate_calls`.

## Issues observed before this fix

### 1. `FIX_A` semantics were not strict

Observed code path:

- `crypten/gradients.py`
  - `grad_input_tag["b_anchor"]` and `grad_input_tag["delta_anchor"]` were set for both
    `FIX_A` and `FIX_AB`.
- `crypten/mpc/primitives/beaver.py`
  - `FIX_A` still reused `B` when `b_anchor` was present.

Effect:

- `reuse_fix_a` and `reuse_fix_ab` produced the same protocol counters for:
  - `triple_generate_calls`
  - `beaver_revealed_tensors`
  - `comm_bytes`

That behavior did not match the intended meaning of `FIX_A`.

### 2. CUDA timing was asynchronous

Observed code path:

- `scripts/bench_reuse_tiny_mlp.py`
- `examples/text-classification/run_glue_private_mpc_lora_train.py`

Both used `time.perf_counter()` directly around CUDA work without `torch.cuda.synchronize()`.

Effect:

- forward / backward / step timing on CUDA could be distorted by async kernel launch.

## Runtime logs and errors observed before this fix

### Import error on another machine

```
Traceback (most recent call last):
  File "/home/yutong_wu/tianyuan_nie/REUSE/mpc-LoRA/scripts/demo_reuse_linear_step.py", line 7, in <module>
    from crypten.common.reuse_context import clear_current_reuse_step, set_current_reuse_step
ModuleNotFoundError: No module named 'crypten.common.reuse_context'
```

Likely causes:

- the repo copy did not include the new reuse files, or
- Python imported an older installed `crypten` from `site-packages`.

### Benchmark output before this fix

```
Provider: TFP
Benchmark: TinyMLP train step = encrypt inputs + forward + MSE loss + backward + SGD
Setup: repeats=3, steps=20, warmup=5, device=cuda, world_size=2

Case: tiny [B=32, in=64, hidden=64, out=16, hidden_layers=1, act=relu]
    baseline | 0.000434 | 0.051158 | 0.050861 | 0.004190 | 0.106642 | 18.00 | 1417216.00 | 18.00 | 9.00 | 1.000x
 reuse_fix_a | 0.000468 | 0.048423 | 0.049843 | 0.004505 | 0.103238 | 18.00 | 1269760.00 | 14.00 | 3.00 | 1.033x
reuse_fix_ab | 0.000419 | 0.046284 | 0.047530 | 0.004060 | 0.098293 | 18.00 | 1269760.00 | 14.00 | 3.00 | 1.085x

Case: base [B=64, in=256, hidden=256, out=64, hidden_layers=2, act=relu]
    baseline | 0.000378 | 0.087032 | 0.075235 | 0.005558 | 0.168202 | 31.00 | 23920640.00 | 28.00 | 14.00 | 1.000x
 reuse_fix_a | 0.000406 | 0.087810 | 0.075247 | 0.005678 | 0.169140 | 31.00 | 20774912.00 | 22.00 | 5.00 | 0.994x
reuse_fix_ab | 0.000454 | 0.095264 | 0.076203 | 0.006224 | 0.178145 | 31.00 | 20774912.00 | 22.00 | 5.00 | 0.944x

Case: wide [B=128, in=512, hidden=1024, out=128, hidden_layers=2, act=relu]
    baseline | 0.000656 | 0.274328 | 0.248795 | 0.008988 | 0.532767 | 31.00 | 204734464.00 | 28.00 | 14.00 | 1.000x
 reuse_fix_a | 0.000729 | 0.280169 | 0.229845 | 0.009184 | 0.519927 | 31.00 | 172228608.00 | 22.00 | 5.00 | 1.025x
reuse_fix_ab | 0.000738 | 0.274139 | 0.230076 | 0.009085 | 0.514038 | 31.00 | 172228608.00 | 22.00 | 5.00 | 1.036x
```

Interpretation of the pre-fix result:

- protocol counters were consistent with the implemented code path,
- but `reuse_fix_a` was not a strict `A`-only mode,
- and CUDA timings were not synchronized.

## Changes applied in this fix

1. `crypten/gradients.py`
   - `b_anchor` and `delta_anchor` for `grad_input_tag` now attach only in `FIX_AB`.
2. `crypten/mpc/primitives/beaver.py`
   - `FIX_A` now always uses a fresh `B`.
3. `scripts/bench_reuse_tiny_mlp.py`
   - added CUDA synchronization around timed regions,
   - added cache / anchor counters to the per-step summary and CSV / JSON export.
4. `examples/text-classification/run_glue_private_mpc_lora_train.py`
   - added CUDA synchronization to `reuse_profile`,
   - added cache / anchor counters to the profile summary.

## Additional review-driven optimization in `beaver_reuse.py`

### 1. Transpose-derived masks no longer re-share plaintext

Previous behavior:

- for a derived transpose mask, the code transposed the plaintext mask,
- then created a brand new `ArithmeticSharedTensor(...)` from that plaintext.

That meant transpose reuse still paid:

- a new `ArithmeticSharedTensor` construction,
- a new PRZS share generation path,
- and an extra materialization step.

New behavior:

- transpose-derived masks are now built from the existing source mask entry,
- the plaintext is transposed as a view,
- and the encrypted share is also transposed directly from the existing share.

Implementation points:

- `crypten/mpc/primitives/beaver_reuse.py`
  - `_apply_transform(...)` no longer forces `.contiguous()` on transpose
  - `_derive_mask_entry(...)` derives a new mask entry from the source entry
  - `_get_or_create_mask(...)` uses `_derive_mask_entry(...)` when the shape matches

### 2. `C` cache is now opportunistic instead of always-on for `FIX_AB`

Previous behavior:

- `FIX_AB` always set `cache_c=True`
- `get_or_create_C_for_op(...)` always built `c_key`
- current TinyMLP traces showed `c_cache_hit=0` and only `c_cache_miss`

That meant the code was paying cache bookkeeping cost without any observed hit.

New behavior:

- `should_cache_c(...)` enables `C` caching only when the tag pattern is more likely to
  produce a reusable `C`
- if caching is disabled, `get_or_create_C_for_op(...)` returns on the fast path without:
  - building a `c_key`
  - computing stable signatures for args / kwargs
  - storing the entry in `_c_cache`

Implementation points:

- `crypten/mpc/primitives/beaver_reuse.py`
  - `should_cache_c(...)`
  - `get_or_create_C_for_op(...)` fast path when `cache_result=False`

Expected effect:

- lower Python-side overhead for the current Linear training workload
- lower overhead in `FIX_AB` where `C` cache had no practical hit rate in the observed runs

## Follow-up fix after regression review

Observed issue after the previous transpose-path optimization:

- protocol counters remained correct,
- but runtime regressed on some CUDA cases,
- especially where transpose-derived masks / residuals were fed directly into `matmul`.

Root cause:

- the previous change removed `.contiguous()` from transpose transforms,
- so derived mask shares and anchored public residuals became strided transpose views,
- those non-contiguous tensors then entered `torch.matmul(...)`,
- which can select a worse kernel path or trigger hidden copies on GPU.

Applied fix:

1. split transform helpers by tensor kind
   - plaintext mask transform
   - shared mask transform
   - public residual transform
2. keep the "no re-sharing" benefit for derived masks
   - derived mask shares are still built from the existing source share
   - they are not rebuilt with a new `ArithmeticSharedTensor(plain, src=0)`
3. restore contiguous layout before compute
   - transpose-derived plaintext masks now use `.contiguous()`
   - transpose-derived shared masks now use `.contiguous()`
   - transpose-derived public residuals now use `.contiguous()`

Implementation points:

- `crypten/mpc/primitives/beaver_reuse.py`
  - `_apply_plain_transform(...)`
  - `_apply_shared_transform(...)`
  - `_apply_public_transform(...)`
  - `_derive_mask_entry(...)`
  - `get_opened_residual_from_anchor(...)`

Net effect intended by this follow-up fix:

- avoid the original re-sharing overhead,
- avoid non-contiguous transpose views entering `matmul`,
- preserve the protocol-level reuse behavior unchanged.

## Counter semantics cleanup

Observed issue in the previous benchmark output:

- `a_cache_hit / a_cache_miss` and `b_cache_hit / b_cache_miss` mixed together:
  - direct base-mask reuse
  - derived transpose-mask reuse
- `c_cache_miss` mixed together:
  - actual cache lookup miss
  - cache bypass fast path
  - fresh `C` generation

That made the benchmark harder to interpret. In particular:

- `a_hit=0` did not mean no reuse happened,
- `c_miss>0` did not mean the code was paying a real cache probe each time.

Applied cleanup:

### A / B mask counters

Added:

- `a_base_cache_hit`
- `a_base_cache_miss`
- `a_derived_cache_hit`
- `a_derived_generated`
- `b_base_cache_hit`
- `b_base_cache_miss`
- `b_derived_cache_hit`
- `b_derived_generated`

Meaning:

- `*_base_*` tracks direct reuse of the base mask entry for the current tagged op
- `*_derived_*` tracks transpose-derived reuse / generation from an anchor path

### C counters

Added:

- `c_cache_probe_hit`
- `c_cache_probe_miss`
- `c_cache_bypassed`
- `c_fresh_generated`

Meaning:

- `c_cache_probe_*` counts only real cache lookup events
- `c_cache_bypassed` counts the fast path where caching was deliberately skipped
- `c_fresh_generated` counts every fresh local `C = op(A, B)` construction

Implementation points:

- `crypten/mpc/primitives/beaver_reuse.py`
  - `_PERF_COUNTERS`
  - `_get_or_create_mask(...)`
  - `get_or_create_C_for_op(...)`
- `scripts/bench_reuse_tiny_mlp.py`
  - summary printout now shows:
    - mask counters
    - `C/residual` counters
- `examples/text-classification/run_glue_private_mpc_lora_train.py`
  - `reuse_profile` now records and logs the detailed counters

## Expected post-fix differences

`reuse_fix_a` should now differ from `reuse_fix_ab` in protocol counters.

For the TinyMLP benchmark:

- strict `FIX_A` should keep the same `triple_generate_calls` reduction as before,
- but `beaver_revealed_tensors` should be higher than `FIX_AB`,
- and `comm_bytes` should also move upward relative to `FIX_AB`.

## Validation commands

TinyMLP benchmark:

```bash
python scripts/bench_reuse_tiny_mlp.py \
  --world-size 2 \
  --provider TFP \
  --device cuda \
  --preset-cases tiny,base,wide \
  --steps 20 \
  --warmup 5 \
  --repeats 3 \
  --run-fix-ab \
  --verbose-comm \
  --save-json bench_reuse_tiny_mlp.json \
  --save-csv bench_reuse_tiny_mlp.csv
```

GLUE training smoke run:

```bash
python examples/text-classification/run_glue_private_mpc_lora_train.py \
  --task_name sst2 \
  --model_name_or_path distilbert-base-uncased \
  --quick_run \
  --skip_private_eval \
  --skip_plain_eval \
  --reuse_profile \
  --reuse_log_every_steps 1 \
  --experimental_reuse_mask \
  --reuse_mode FIX_A \
  --gpu_ids 0
```

## Validation status in the current workspace

### Static validation

Command:

```bash
python -m py_compile \
  crypten/gradients.py \
  crypten/mpc/primitives/beaver.py \
  scripts/bench_reuse_tiny_mlp.py \
  examples/text-classification/run_glue_private_mpc_lora_train.py
```

Status:

- completed without a Python syntax error in this workspace.

### Runtime validation attempt 1

Command:

```bash
python scripts/bench_reuse_tiny_mlp.py \
  --world-size 2 \
  --provider TFP \
  --device cpu \
  --preset-cases tiny \
  --steps 1 \
  --warmup 0 \
  --repeats 1 \
  --run-fix-ab
```

Observed error:

```
Traceback (most recent call last):
  File "D:\SJTU\newSHAFT\scripts\bench_reuse_tiny_mlp.py", line 9, in <module>
    import crypten
ModuleNotFoundError: No module named 'crypten'
```

Reason:

- the current shell session did not set `PYTHONPATH=.`, and the package was not installed in editable mode.

### Runtime validation attempt 2

Command:

```bash
set PYTHONPATH=.
python scripts/bench_reuse_tiny_mlp.py \
  --world-size 2 \
  --provider TFP \
  --device cpu \
  --preset-cases tiny \
  --steps 1 \
  --warmup 0 \
  --repeats 1 \
  --run-fix-ab
```

Observed error:

```
Traceback (most recent call last):
  File "D:\SJTU\newSHAFT\scripts\bench_reuse_tiny_mlp.py", line 9, in <module>
    import crypten
  File "D:\SJTU\newSHAFT\crypten\config\config.py", line 11, in <module>
    import yaml
ModuleNotFoundError: No module named 'yaml'
```

Reason:

- the current Python environment is missing `PyYAML`.

Immediate fix:

```bash
python -m pip install pyyaml
```
