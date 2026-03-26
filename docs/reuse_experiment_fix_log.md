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

## Shared-left fan-out reuse (`A @ B`, `A @ C`, ...)

### Goal

Add an explicit left-operand reuse path for sibling `matmul` ops that share the same
left operand `A` inside one step, for example:

- `Q = X @ Wq`
- `K = X @ Wk`
- `V = X @ Wv`

The intended reuse is:

- reuse the left Beaver mask `a_mask`
- reuse the opened left residual `epsilon = A - a_mask`
- keep `B / C / ...`, `delta`, and `C = a @ b` per-op

### Implementation path

1. `crypten/common/reuse_context.py`
   - added `set/get/clear/use_a_group(...)`
   - this is the explicit user / model annotation for shared-left fan-out reuse
2. `crypten/gradients.py`
   - `AutogradMatMul.forward()` now copies `get_current_a_group()` into the beaver tag as
     `a_group`
3. `crypten/mpc/primitives/beaver_reuse.py`
   - `_normalize_tag(...)` now carries `a_group`
   - `_base_descriptor(...)` uses a group-based key for operand `a` when `a_group` is set
   - `_residual_key(...)` uses a group-based key for `epsilon` when `a_group` is set
   - `_normalize_anchor(...)` carries `group` for operand `a`
   - `_get_or_create_mask(...)` uses the grouped `a` base key so sibling matmuls share the
     same left mask entry even when `op_uid` and `layer_id` differ
4. `scripts/bench_reuse_shared_left_fanout.py`
   - added a dedicated benchmark for `A @ B`, `A @ C`, ... shared-left fan-out reuse
   - benchmark model: shared stem + multi-head Linear fan-out
   - grouped fan-out heads run under `with use_a_group(...)`

### Expected protocol effect

For `k` sibling matmuls with the same left operand `A`:

- baseline reveal tensors: `2k`
- grouped-left reuse reveal tensors: `k + 1`

because:

- `epsilon = A - a_mask` is opened once
- each branch still opens its own `delta_i = B_i - b_i`

### Notes on scope

- this path is explicit, not automatic
- it is intended for model structures that know multiple sibling `matmul` ops share the same
  left operand
- it complements the existing forward/backward anchor reuse rather than replacing it

### Backward scope note

`a_group` is intentionally stripped from `backward_dX` tags in `crypten/gradients.py`.
That prevents unrelated `grad_output` tensors from being treated as a shared-left operand.
The group remains available on forward and `backward_dW`, where the left operand is still
logically tied to the original shared activation `A` (or its transpose anchor).

## SHARED_LEFT as the main reuse path for encrypted BERT training

### Goal

Promote the new shared-left reuse idea from a standalone fan-out benchmark into the
actual encrypted GLUE / BERT training pipeline.

The old `FIX_A` / `FIX_AB` paths remain in the lower-level runtime for compatibility,
but the training and benchmark entrypoints now treat `SHARED_LEFT` as the primary
experiment mode.

### Code path

1. `configs/default.yaml`
   - changed default `cfg.mpc.reuse_mode` from `FIX_A` to `SHARED_LEFT`
2. `crypten/mpc/primitives/beaver.py`
   - default runtime fallback for reuse mode now uses `SHARED_LEFT`
3. `crypten/gradients.py`
   - default autograd fallback for reuse mode now uses `SHARED_LEFT`
   - `backward_dW` skips the old forward/backward anchor path when `reuse_mode == SHARED_LEFT`
   - `backward_dX` strips `a_group` to avoid falsely treating `grad_output` as a shared-left operand
4. `crypten/nn/module.py`
   - `Linear.forward()` and `Gemm.forward()` honor an injected `beaver_a_group`
   - grouped modules run their matmul under `use_a_group(...)`
5. `examples/text-classification/run_glue_private_mpc_lora_train.py`
   - parser default for `--reuse_mode` changed to `SHARED_LEFT`
   - added `--shared_left_min_fanout`
   - added `--shared_left_log_groups`
   - added `_annotate_shared_left_groups_crypten_model(...)`
   - after `ct.nn.from_pytorch(...).encrypt()`, the CrypTen graph is scanned for sibling
     `Gemm` / `Linear` nodes with the same left input and compatible left transform
   - grouped nodes are annotated with:
     - `beaver_a_group`
     - `beaver_layer_tag`
   - shared-left grouping summary is logged at startup and saved into
     `train_eval_summary.json`
6. `scripts/bench_reuse_shared_left_fanout.py`
   - simplified to compare only:
     - `baseline`
     - `shared_left`

### Why graph-level auto-grouping is needed

The encrypted training script converts the HuggingFace PyTorch model into a CrypTen graph
through `ct.nn.from_pytorch(...)`.

At that point, original PyTorch module names such as `query`, `key`, `value` are no longer
reliable grouping anchors. The robust place to detect shared-left opportunities is therefore
inside the CrypTen graph:

- inspect each `ct.nn.Graph`
- collect `Gemm` / `Linear` consumers by their first input name
- only form a reuse group when fan-out >= `shared_left_min_fanout`
- also separate groups by left transform (`identity` vs `transpose`) so `A` and `A^T`
  are never mixed into one reuse group

This is the path now used for encrypted BERT / GLUE training.

### Expected effect in BERT-like models

Typical opportunities include:

- attention Q / K / V projections: same hidden state, multiple sibling projections
- sibling classifier heads or other explicit fan-out branches
- backward weight-gradient matmuls for those sibling branches

The main counters that should move under `SHARED_LEFT` are:

- `a_base_cache_hit`
- `residual_cache_hit`
- `beaver_revealed_tensors`
- communication `bytes`

The main runtime metric remains:

- `step_time_s`

### Runtime note

The grouping is explicit and step-scoped:

- only applies when `--experimental_reuse_mask --reuse_mode SHARED_LEFT` are enabled
- only affects grouped sibling matmuls that actually share one left operand
- does not require the old `FIX_A` / `FIX_AB` reasoning at the training-entry level

### Result record

The first overnight / multi-repeat CUDA result interpretation for the new
`shared_left` benchmark was written into:

- `docs/shared_left_fanout_benchmark_2026-03-27_zh.md`

That note records:

- the raw benchmark output for `tiny / base / wide`
- Chinese explanations for every printed metric
- theory-vs-observation checks for:
  - `a_base_cache_hit`
  - `residual_cache_hit`
  - `beaver_revealed_tensors`
  - `bytes`
- runtime interpretation for why protocol savings do not always translate into
  wall-clock speedup

## BERT quick-run blockers found after integrating SHARED_LEFT

### Observed symptoms

When running the encrypted GLUE / BERT quick path with:

- `--experimental_reuse_mask`
- `--reuse_mode SHARED_LEFT`

two independent blockers were found:

1. startup log showed:
   - `[shared-left] no eligible grouped Gemm/Linear fan-out was detected in the CrypTen graph`
2. training failed in:
   - `LayerNormalization.forward()`
   - `inv_sqrt()`
   - with `RuntimeError: Autograd is not supported for in-place functions.`

### Root causes

1. Shared-left auto-grouping was too narrow
   - the CrypTen graph produced by ONNX export for BERT uses `MatMul` nodes in the projection path
   - the grouping pass only considered `Gemm` / `Linear`
   - as a result, Q/K/V-style fan-out was not annotated at all

2. CrypTen approximation functions still contained in-place tensor ops
   - `inv_sqrt()` used `y -= ...` and `mul_ / div_`
   - `sqrt()` used `mul_`
   - several related approximation helpers (`log`, `reciprocal`, `_eix`, `tanh`, `_fourier_series`, `softmax`)
     also used in-place updates that are incompatible with CrypTen autograd

### Fixes applied

1. `crypten/nn/module.py`
   - `MatMul.forward()` now honors:
     - `beaver_layer_tag`
     - `beaver_a_group`
   - grouped `MatMul` nodes execute under `use_a_group(...)`

2. `examples/text-classification/run_glue_private_mpc_lora_train.py`
   - `_is_shared_left_groupable_module(...)` now includes `ct.nn.MatMul`

3. `crypten/common/functions/approximations.py`
   - replaced high-risk in-place expressions with out-of-place equivalents in:
     - `log`
     - `reciprocal`
     - `inv_sqrt`
     - `sqrt`
     - `_eix`
     - `tanh` (ODE path)
     - `_fourier_series`
     - `softmax` (ODE path)

### Expected effect after the fix

1. shared-left grouping should begin to detect real BERT fan-out `MatMul` nodes
2. the quick run should no longer fail immediately in `LayerNorm -> inv_sqrt()`
3. if later failures remain, they are more likely to be downstream modeling / approximation issues,
   not the original shared-left integration bug
