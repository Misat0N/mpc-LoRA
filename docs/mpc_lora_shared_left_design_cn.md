# MPC-LoRA + shared-left-v1.2 设计与实现说明

## 1. 项目目标

本次改动在当前分支 `shared-left-v1.2` 的真实代码基础上，补齐两条实验路径：

1. 明文 baseline：用 Hugging Face 的 `bert-base-uncased` 在 GLUE 上做普通微调，作为精度和训练流程对照。
2. MPC 微调：复用仓库现有 CrypTen 训练链路与 `shared-left-v1.2` 底层 Beaver mask / residual reuse 能力，实现面向 LoRA 的最小侵入式接入。

研究动机很直接：

- 预训练 backbone 是冻结的，大参数量但不更新。
- LoRA 是小参数分支，可训练但参数量小。
- MPC 中最昂贵的部分主要是 `secret x secret` 的矩阵乘法与由此带来的通信。
- 因此，上层设计应当显式区分“冻结主干路径”和“LoRA 增量路径”，并优先把 `shared-left` 用在同一左操作数反复参与多个 matmul 的地方，尤其是注意力层的 Q / K / V LoRA-A 投影。

## 2. 当前主流隐私/密态训练方案对比

| 方案 | 隐私强度 | 工程复杂度 | 典型瓶颈 | 适合场景 |
| --- | --- | --- | --- | --- |
| 普通明文微调 | 低 | 低 | 数据明文暴露 | 基线、快速迭代 |
| 联邦微调 | 中 | 中 | 非 IID、聚合鲁棒性、泄露梯度风险 | 多机构协作且可接受较弱威胁模型 |
| MPC 微调 | 高 | 高 | 通信与 secret-secret matmul | 高强度隐私、强密码学威胁模型 |
| HE-LoRA / Hybrid HE-LoRA | 高 | 高 | 乘法深度、近似算子、实现复杂 | 更偏推理或混合方案 |
| TEE / 可信执行环境 | 中到高 | 中 | 侧信道、硬件信任根 | 工程落地快、部署友好 |

简化判断：

- 更适合工程快速落地：TEE、普通明文、部分联邦方案。
- 更适合高强度隐私：MPC、HE、Hybrid HE。
- 在当前仓库语境下，MPC 的核心瓶颈依旧是通信，尤其是频繁的秘密矩阵乘法和中间 opening。

## 3. 我们方案的核心思想

对任意一个线性层，LoRA 写成：

$$
W' = W + AB
$$

前向计算为：

$$
y = x(W + AB) = xW + xAB = xW + (xA)B
$$

这里必须显式区分两条路径：

1. 主干路径：`xW`
2. LoRA 路径：`(xA)B`

对注意力层的 query / key / value，更具体地写成：

$$
Q = X(W_Q + A_Q B_Q) = XW_Q + (X A_Q) B_Q
$$

$$
K = X(W_K + A_K B_K) = XW_K + (X A_K) B_K
$$

$$
V = X(W_V + A_V B_V) = XW_V + (X A_V) B_V
$$

因此，`shared-left` 最自然的打点是：

- `X A_Q`
- `X A_K`
- `X A_V`

因为这三次乘法共享同一个左操作数 `X`，而右操作数不同。

## 4. 为什么 shared-left 在 LoRA 注意力层里有效

形象地说，注意力层里的同一份隐藏状态 `X` 会同时送入 Q / K / V 三个分支。若我们又在这三个分支上挂 LoRA，那么 LoRA 的第一段低秩投影都形如：

$$
X \rightarrow XA_Q,\ XA_K,\ XA_V
$$

这正好满足“同一个左矩阵，多次右乘”的模式。

本分支里 `shared-left-v1.2` 的真实语义不是“把整组 Beaver triple 任意复用”，而是：

- 通过 `beaver_a_group` / `a_group` 标记同一组左操作数消费者。
- 在同一步 `step_id` 范围内缓存左侧 mask 以及对应 residual。
- 只有当左操作数被认为是同一个、且还在同一个 step 内时，缓存才允许命中。

安全前提必须写清楚：

1. 只能针对同一个未变化的左操作数复用。
2. 变量变化后必须失效。
3. 不能把整组 triple 无条件跨不同输入、不同 step 复用。

本分支里真实的失效机制如下：

- 训练脚本每步调用 `beaver_protocol.begin_reuse_step(step_id)`。
- 步结束后调用 `beaver_protocol.end_reuse_step(step_id)`。
- `crypten/mpc/primitives/beaver_reuse.py` 内部在 `begin_step()` / `end_step()` 时清空 mask / residual cache。

## 5. 现有仓库真实代码结构梳理

### 5.1 现有 LoRA 与 CrypTen 训练入口

- `examples/text-classification/run_glue_private_mpc_lora_train.py`
  - 现有 LoRA 接入入口。
  - 真实函数名包括：
    - `LoRALinear`
    - `_inject_lora_layers(...)`
    - `_set_lora_trainable(...)`
    - `_annotate_shared_left_groups_crypten_model(...)`
  - 模型在该脚本中通过 `ct.nn.from_pytorch(model, ...).encrypt().to(device)` 转为密态模型。
- `examples/text-classification/run_glue_private_light_train.py`
  - 现有轻量私有训练参考脚本，已经接了 shared-left 分组标注逻辑。
- `examples/text-classification/run_glue_private_train_smoke.py`
  - 更轻的 smoke 训练入口。
- `examples/text-classification/run_glue_eval.py`
  - 现有明文评估脚本，只做 eval，不做训练。

### 5.2 shared-left-v1.2 的真实底层接口

- `crypten/common/reuse_context.py`
  - `set_current_reuse_step(...)`
  - `clear_current_reuse_step()`
  - `use_layer_tag(...)`
  - `use_a_group(...)`
  - `next_beaver_op_uid()`
- `crypten/gradients.py`
  - `AutogradMatMul.forward/backward`
  - 在 matmul autograd 中构造 `step_id / layer_id / op_uid / a_group` 等 Beaver tag。
- `crypten/mpc/primitives/beaver.py`
  - `_beaver_with_reuse(...)`
  - `begin_reuse_step(...)`
  - `end_reuse_step(...)`
  - `reset_reuse_stats(...)`
  - `get_reuse_stats()`
- `crypten/mpc/primitives/beaver_reuse.py`
  - `BeaverReuseCache`
  - `get_or_create_A(...)`
  - `get_or_create_B(...)`
  - `get_or_create_C_for_op(...)`
  - `cache_opened_residual(...)`
  - `get_opened_residual(...)`
  - `get_opened_residual_from_anchor(...)`
- `crypten/nn/module.py`
  - `Linear.forward`
  - `Gemm.forward`
  - `MatMul.forward`
  - 这些模块会读取 `beaver_a_group` / `beaver_layer_tag` 并进入 `use_a_group(...)` / `use_layer_tag(...)`。

### 5.3 当前 LoRA 如何插入、哪些参数被冻结、模型是否整体 encrypt

当前仓库已有 LoRA 脚本的真实行为是：

1. 在 PyTorch 模型上递归替换目标 `nn.Linear`。
2. 冻结被替换线性层的原始权重。
3. 只训练 LoRA 参数和可选的分类头。
4. 完成 PyTorch 构图后，再整体执行 `ct.nn.from_pytorch(...).encrypt()`。

因此，当前 runtime 仍然是“模型整体进入密态”的实现。也就是说：

- 代码结构上已经可以区分“冻结 backbone”与“LoRA 分支”。
- 但 runtime 层面尚未真正实现 `secret activation x public frozen weight` 的单独快路径。
- 这一点在本文档里必须如实记录，不能把“结构上分开”误写成“运行时已经 public-weight 优化完毕”。

## 6. 本次新增/整理的文件

### 6.1 `examples/text-classification/run_glue_plain_hf.py`

作用：

- 新增明文 baseline 训练脚本。
- 不是重写训练器，而是薄包装本仓库自带的 Hugging Face 官方示例 `transformers/examples/pytorch/text-classification/run_glue_no_trainer.py`。
- 默认模型名是 `bert-base-uncased`。
- 支持 GLUE，至少覆盖 `sst2`、`mrpc`、`rte`。

保留的核心参数：

- `--model_name_or_path`
- `--task_name`
- `--max_length`
- `--batch_size`
- `--learning_rate`
- `--num_train_epochs`
- `--seed`
- `--output_dir`

### 6.2 `examples/text-classification/shared_left_v12_lora.py`

作用：

- 新增显式拆分的 LoRA 线性层。
- 用 `SharedLeftSplitLoRALinear` 把逻辑写成：

$$
\text{output} = \text{backbone\_path} + \text{lora\_path}
$$

其中：

- `backbone_path = xW`
- `lora_path = (xA)B`

内部关键类：

- `SharedLeftLoRAPath`
  - 显式暴露 `xA -> (xA)B` 路径。
  - 代码注释明确说明：共享左操作数优先针对这一段。
- `SharedLeftSplitLoRALinear`
  - 冻结 `backbone_path`。
  - 只保留 `lora_path` 可训练。
- `inject_shared_left_lora_layers(...)`
  - 复用原仓库“递归替换 `nn.Linear`”的思路，不改主训练 loop。

### 6.3 `examples/text-classification/shared_left_graph_groups.py`

作用：

- 专门负责 CrypTen 图级别的 shared-left 分组。
- 相比旧版“只看 immediate left input 名字是否完全相同”的逻辑，这一版做了两层增强：
  1. 显式识别 `query/key/value` 的 `lora_A/MatMul`
  2. 对左输入做 canonicalization，允许跨 `Identity / Reshape / Flatten / Cast / Dropout` 这类别名节点回溯到同一语义左操作数

内部策略：

- `explicit_qkv_lora_a`
  - 优先把 Q/K/V 的 `XA_Q / XA_K / XA_V` 分成一组
- `generic_canonical_left`
  - 对其它可识别的 shared-left fanout 做兜底分组

### 6.4 `examples/text-classification/run_glue_private_shared_left_v12.py`

作用：

- 这是新的 MPC-LoRA + shared-left 入口。
- 不是另起一套 fake runtime，而是直接复用现有：
  - `run_glue_private_mpc_lora_train.py`
  - CrypTen 训练 loop
  - shared-left 配置与 profile 逻辑

接入方式：

1. monkey patch `LoRALinear` 为 `SharedLeftSplitLoRALinear`
2. monkey patch `_inject_lora_layers(...)`
3. 复用新的 `shared_left_graph_groups.py`，让 CrypTen 图分组优先识别 Q/K/V LoRA-A
4. 默认把 `--lora_target_modules` 设为 `query,key,value`
5. 默认打开 `--experimental_reuse_mask`
6. 默认 `--reuse_mode SHARED_LEFT`
7. 默认 `--shared_left_min_fanout 3`

这里的 `min_fanout=3` 是一个上层偏置：

- 它并不是硬编码“只允许 Q/K/V 命中”
- 它只是把自动分组策略更偏向注意力 Q / K / V 这种天然三分支 fanout
- 真正是否命中，仍由 CrypTen 图中“同左输入的 sibling Gemm / Linear / MatMul”决定

## 7. 为什么这里是“最小侵入式接入”

本次实现没有重写以下内容：

- CrypTen 的 matmul 协议
- Beaver reuse 底层缓存
- 原有多进程 launcher
- 原有数据处理 / tokenizer / 加密训练主循环

只在两层做最小改动：

1. 上层 LoRA 包装器：把数学结构写得更清楚。
2. 新入口脚本：给现有训练脚本塞入更合理的 shared-left 默认值与显式 Q/K/V LoRA 目标。

## 8. 代码层面的“主干路径 + LoRA 路径”

现在注意力目标线性层至少在代码结构上已经显式呈现为：

```python
backbone_path = self.backbone_path(x)
lora_path = self.lora_path(x)
return backbone_path + lora_path
```

而 LoRA 路径内部又进一步拆成：

```python
xa = self.lora_A(x_lora)
return self.lora_B(xa) * self.scaling
```

这对应的数学含义就是：

$$
xW + (xA)B
$$

需要特别说明的现实限制：

- 当前 CrypTen runtime 里，模型仍然是整体 `.encrypt()`。
- 所以 frozen backbone 目前还没有被真正下沉成“公开常量权重乘秘密激活”的 runtime 快路径。
- 本次实现先把结构拆清楚，并把 `shared-left` 优先瞄准 LoRA 小分支。

## 9. shared-left 何时生效、何时失效

### 9.1 生效条件

shared-left 想要命中，至少需要同时满足：

1. `--experimental_reuse_mask` 打开。
2. `reuse_mode=SHARED_LEFT`。
3. 当前操作是 matmul 路径。
4. CrypTen 图里有多个 sibling `Gemm / Linear / MatMul` 消费同一个左输入。
5. 这些模块被 `_annotate_shared_left_groups_crypten_model(...)` 标上同一个 `beaver_a_group`。

在本次修复后，条件 4 的识别方式更具体了：

- 对 LoRA 注意力层，优先匹配 Q/K/V 的 `lora_A/MatMul` 节点家族；
- 如果 immediate left input 名称不同，但能通过 alias 节点回溯到同一个 canonical left input，仍可归为同一组。

### 9.2 失效条件

当前实现里，缓存会在以下情况下失效：

1. step 切换。
2. `begin_reuse_step(step_id)` 发现 step 变了，会清空缓存。
3. `end_reuse_step(step_id)` 在步结束时清空缓存。
4. 左输入不再属于同一个 `a_group`。
5. residual / anchor 形状不匹配。

因此，本实现满足“同一个未变化左操作数、同一步内复用；变化后立即失效”的基本边界。

## 10. 实验建议

建议至少做以下对比：

1. `baseline: 明文 bert-base-uncased`
2. `private: 原始 CrypTen-LoRA`
3. `private + shared-left`
4. `只在 Q/V 挂 LoRA` 对比 `Q/K/V 全挂`

建议记录的指标：

- accuracy
- F1
- runtime
- communication bytes
- communication rounds
- memory
- Beaver reveal 次数
- number of openings

建议 ablation：

1. 是否启用 `--experimental_reuse_mask`
2. `--reuse_mode SHARED_LEFT` vs 关闭复用
3. `--lora_target_modules query,value` vs `query,key,value`
4. `--shared_left_min_fanout 2` vs `3`

## 11. 最小运行与验证步骤

### 11.1 环境安装

建议按仓库根目录 README 与文本分类目录 requirements 安装：

```bash
pip install .
pip install -r examples/text-classification/requirements.txt
pip install ./transformers
```

如果需要固定 CUDA / PyTorch 版本，可继续沿用仓库根 README 的安装方式。

### 11.2 明文 baseline 运行命令

SST-2：

```bash
python examples/text-classification/run_glue_plain_hf.py \
  --model_name_or_path bert-base-uncased \
  --task_name sst2 \
  --max_length 128 \
  --batch_size 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --seed 42 \
  --output_dir examples/text-classification/eval_plain_hf/sst2
```

MRPC / RTE 只需要把 `--task_name` 换成 `mrpc` 或 `rte`。

### 11.3 MPC-LoRA + shared-left 运行命令

一个更轻的 smoke 例子：

```bash
python examples/text-classification/run_glue_private_shared_left_v12.py \
  --task_name sst2 \
  --model_name_or_path bert-base-uncased \
  --gpu_ids 0,1 \
  --pad_to_max_length \
  --max_length 32 \
  --len_data 32 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --max_train_steps 1 \
  --eval_max_steps 1 \
  --train_max_samples 32 \
  --eval_max_samples 32 \
  --reuse_profile \
  --output_dir examples/text-classification/eval_private/sst2/shared_left_v12_smoke
```

一个正常实验起点：

```bash
python examples/text-classification/run_glue_private_shared_left_v12.py \
  --task_name sst2 \
  --model_name_or_path bert-base-uncased \
  --gpu_ids 0,1 \
  --pad_to_max_length \
  --max_length 64 \
  --len_data 64 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --max_train_steps 100 \
  --log_every_steps 5 \
  --eval_max_steps 64 \
  --lora_r 8 \
  --lora_alpha 16 \
  --reuse_profile \
  --output_dir examples/text-classification/eval_private/sst2/shared_left_v12
```

### 11.4 如何查看 shared-left 是否实际生效

建议同时看三处：

1. 启动日志
   - 会打印 `[shared-left] grouped_graphs=... groups=... grouped_modules=...`
2. 运行时 profile
   - 开 `--reuse_profile` 后，日志里会有：
     - `a_base_cache_hit`
     - `a_derived_cache_hit`
     - `residual_cache_hit`
     - `residual_anchor_hit`
3. 输出目录里的 `train_eval_summary.json`
   - 会保存：
     - `shared_left_group_summary`
     - `reuse_profile_summary`

如果这里始终没有任何 `group` 或 cache hit，那么说明当前模型图下 shared-left 还没有真正打中目标 matmul。

## 12. 当前限制与后续工作

### 12.1 当前限制

1. 当前 shared-left 主要打在 CrypTen 图中被自动识别到的 `Gemm / Linear / MatMul` 上，不是所有算子。
2. 现阶段更像“结构正确 + smoke 可跑”的研究原型，而不是全面优化完成的训练系统。
3. 当前 runtime 仍然整体 `.encrypt()` 模型，因此还没有真正实现：

$$
\text{secret activation} \times \text{public frozen weight}
$$

的独立快路径。

4. 本次实现通过显式 `xW + (xA)B` 拆分和 Q/K/V 默认 targeting，把优化焦点先集中在 LoRA 小分支上；但 shared-left 的最终命中仍由 CrypTen 图结构决定。

### 12.2 后续工作

1. 在 CrypTen runtime 层真正支持 public-weight x secret-activation 快路径。
2. 对 attention Q/K/V 之外的其它可安全识别 fanout 模式做更细粒度分组。
3. 继续把分组信息从 PyTorch 模块级显式传递到 CrypTen 图节点级，减少“只能靠图结构自动识别”的不确定性。
4. 扩展到 decoder-only / causal LM，例如 GPT 类模型。

## 13. 本次实现结论

这次改动的重点不是重新发明一套私有训练框架，而是基于当前仓库已经存在的真实调用链，把下面三件事做清楚：

1. 明文 baseline 可直接复用标准 Hugging Face 训练路径跑起来。
2. MPC-LoRA 继续复用现有 CrypTen 训练脚本。
3. 在代码结构上显式写出 `xW + (xA)B`，并把 shared-left 的默认打点偏向注意力 Q/K/V 的 LoRA-A 阶段。

这样做的价值是：

- 研究逻辑更清楚。
- 实验对照更完整。
- 后续若要把 frozen backbone 进一步常量化或公开化，代码结构已经提前为 runtime 优化留好了位置。
