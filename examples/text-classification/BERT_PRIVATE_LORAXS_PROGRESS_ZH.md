# BERT-base 低开销 MPC 密态微调：实现思路与当前进展

## 1. 项目目标

当前要实现的目标是：在 `newSHAFT` 中完成一个面向 `bert-base` 的低开销 MPC 密态微调方案，并围绕它搭建完整的对照实验。

核心要求有 6 点：

1. 从 Hugging Face 下载预训练 `bert-base` 模型并进行下游微调。
2. 主干预训练权重公开且冻结，只对输入和微调相关参数保持密态。
3. 微调位置放在注意力层，而不是扩展到整网全参数。
4. 使用已有的 `shared-left` 左复用原理降低在线阶段代价。
5. 微调方法使用 LoRA-XS，而不是标准 LoRA。
6. 搭建统一的对照组，比较不同设计点带来的通信和时间开销变化。

一句话概括：目标不是再做一个“能跑的私有 BERT 微调脚本”，而是做一个“低开销、可拆解、可做消融对比”的私有微调实验框架。

---

## 2. 原始代码基础与问题

当前实现是建立在 `newSHAFT` 已有私有训练框架之上的，核心基础入口有两条：

- `run_glue_private_mpc_lora_train.py`
  这是原始的 LoRA + CrypTen 私有微调入口。
- `run_glue_private_light_train.py`
  这是更接近全参数私有训练的轻量入口。

原始代码已经具备以下基础能力：

- 从 Hugging Face 加载 `AutoConfig`、`AutoTokenizer`、`AutoModelForSequenceClassification`
- 将 PyTorch 模型转换为 CrypTen 私有模型
- 在 2-party MPC 环境中训练与评估
- 支持 `public_non_lora_weights`，也就是“模型先整体进密态，再把一部分参数公开回明文”
- 支持 `experimental_reuse_mask` 和 `reuse_mode=SHARED_LEFT`

但原始代码还不能直接满足当前目标，主要差距在：

- 原始低秩适配是标准 LoRA，不是 LoRA-XS。
- 原始默认 LoRA 注入位置与当前实验目标不完全对齐。
- 原始代码没有按“全密态 LoRA / 主干公开 LoRA / 主干公开 LoRA-XS / 主干公开 LoRA-XS + 左复用”这条主线组织成完整对照矩阵。
- 原始代码没有直接提供“4 组实验统一汇总开销”的脚本。

---

## 3. 当前总体方案

### 3.1 总体技术路线

当前的实现思路是：

1. 复用 `newSHAFT` 原有的 CrypTen 训练循环，不重写 MPC 训练主干。
2. 先构造一个与 LoRA-XS 论文/仓库思路一致的“标准 LoRA 版本”作为中间基线。
3. 在此基础上把标准 LoRA 改造成 LoRA-XS。
4. 在 LoRA-XS 版本上再叠加 `shared-left` 左复用。
5. 用统一的超参数和统一的注入位置，构建四组消融实验。

这样做的原因是：

- 训练循环、MPC 执行、私有评估、明文恢复，这些能力原仓库已经有，不值得重写。
- 真正需要新增的是“低秩层怎么定义”和“哪些参数保持密态”。
- 通过 wrapper 方式 monkey patch 原入口，可以最大程度复用已有逻辑，并降低改动范围。

### 3.2 统一的微调位置

目前四组对照实验的低秩注入位置已经统一为注意力层的：

- `query`
- `key`
- `value`

也就是只调 BERT 自注意力里的 Q/K/V 三个线性投影层，不再把 FFN 层混进主对照矩阵里。这样做的好处是：

- 设计目标更清晰，直接对应“只调注意力层”
- 更适合 `shared-left`，因为 Q/K/V 具有天然的共享左操作数结构
- 对照组之间差异只落在“是否公开主干 / 是否 LoRA-XS / 是否左复用”，不会被目标模块差异干扰

### 3.3 密态范围设计

当前设计里最关键的一点是：不是简单地“整个模型都加密”或“整个模型都公开”，而是利用原始代码中的 `public_non_lora_weights` 机制做细粒度控制。

执行逻辑是：

1. 先把整个模型转成 CrypTen 私有模型。
2. 再把不属于微调部分的参数从密态恢复成明文张量。
3. 只保留输入和特定 trainable 参数继续处于密态。

这实际上实现的是：

- 密态激活 × 公开冻结权重
- 密态激活 × 密态可训练权重

因此“主干公开但输入仍然密态”这一点是可以成立的。

---

## 4. LoRA-XS 方案在当前代码中的落地

### 4.1 LoRA-XS 的核心思路

标准 LoRA 训练两张低秩矩阵：

- `A`
- `B`

LoRA-XS 的思路是：

1. 对预训练权重做截断 SVD。
2. 用 SVD 结果初始化低秩分解中的两端基底。
3. 将这两端基底冻结。
4. 只训练中间一个很小的 `r x r` latent mapping。

因此它的低秩路径不再是：

`B(A(x))`

而是：

`B(R(A(x)))`

其中：

- `A` 冻结
- `B` 冻结
- `R` 可训练

### 4.2 当前实现方式

当前新增文件：

- `loraxs_public_layers.py`

这里实现了：

- `LoRAXSPath`
- `LoRAXSPublicLinear`
- `inject_loraxs_layers`

当前 LoRA-XS 层的实现逻辑是：

1. 对原始线性层权重做 SVD。
2. 用 SVD 结果初始化 `lora_A` 和 `lora_B`。
3. 冻结 `lora_A` 和 `lora_B`。
4. 新增 `lora_latent` 作为唯一主要低秩可训练矩阵。
5. 前向输出为：`backbone_path(x) + lora_B(lora_latent(lora_A(x)))`

### 4.3 为什么这比标准 LoRA 更省

以 BERT-base 为例：

- `hidden_size = 768`
- `num_hidden_layers = 12`
- 每层有 `query/key/value` 三个投影
- 当前统一 `r = 8`

那么：

- 标准 LoRA 对单个 `768 -> 768` 线性层的可训练参数量约为  
  `768*8 + 8*768 = 12288`
- LoRA-XS 对单个同类线性层的可训练参数量约为  
  `8*8 = 64`

如果只看 Q/K/V：

- 总层数：`12 * 3 = 36`
- 标准 LoRA：`36 * 12288 = 442368`
- LoRA-XS：`36 * 64 = 2304`

也就是说，在不计分类头的情况下，LoRA-XS 相比标准 LoRA 将低秩可训练参数量降到了约 `1/192`。

这也是当前方案里最核心的“低开销”来源之一。

### 4.4 为什么要保留 `lora_path/lora_A` 命名

当前的 `shared-left` 图分组逻辑会显式识别：

- `query/lora_path/lora_A`
- `key/lora_path/lora_A`
- `value/lora_path/lora_A`

这类 Q/K/V 分支。

因此 LoRA-XS 封装层虽然内部结构和标准 LoRA 不同，但仍然保留了：

- `lora_path`
- `lora_A`

这套命名结构，以便现有 `shared_left_graph_groups.py` 中的显式 Q/K/V 复用逻辑继续工作。

---

## 5. 当前四组对照实验设计

当前实验矩阵已经统一成以下四组。

| 组名 | 主干权重 | 低秩方法 | 密态训练部分 | 左复用 | 入口脚本 |
| --- | --- | --- | --- | --- | --- |
| `full_private_lora` | 密态 | 标准 LoRA | 整模型密态，LoRA 训练 | 否 | `run_glue_private_full_private_lora_train.py` |
| `public_backbone_lora` | 公开冻结 | 标准 LoRA | `lora_A/lora_B + classifier` | 否 | `run_glue_private_loraxs_lora_train.py` |
| `public_backbone_loraxs` | 公开冻结 | LoRA-XS | `lora_latent + classifier` | 否 | `run_glue_private_loraxs_train.py` |
| `public_backbone_loraxs_shared_left` | 公开冻结 | LoRA-XS | `lora_latent + classifier` | 是 | `run_glue_private_loraxs_shared_left_train.py` |

这四组的共同点：

- 都从 Hugging Face 下载 `bert-base-uncased`
- 都是 BERT sequence classification
- 都只在 `query/key/value` 上做低秩注入
- 都复用同一套 CrypTen 私有训练主循环

这四组的主要差异路径是：

1. 是否将主干从密态公开出来
2. 低秩方法是 LoRA 还是 LoRA-XS
3. 是否开启 `shared-left`

这样设置后，对照关系比较清晰：

- 组 1 vs 组 2：看“主干公开冻结”本身带来的收益
- 组 2 vs 组 3：看“LoRA -> LoRA-XS”带来的收益
- 组 3 vs 组 4：看“加上 shared-left”带来的收益

---

## 6. 当前新增/修改的代码结构

### 6.1 新增入口与核心模块

| 文件 | 作用 | 当前状态 |
| --- | --- | --- |
| `loraxs_public_layers.py` | LoRA-XS 层定义与注入逻辑 | 已完成 |
| `run_glue_private_full_private_lora_train.py` | 全密态标准 LoRA 基线入口 | 已完成 |
| `run_glue_private_loraxs_lora_train.py` | 主干公开冻结 + 标准 LoRA 入口 | 已完成 |
| `run_glue_private_loraxs_train.py` | 主干公开冻结 + LoRA-XS 入口 | 已完成 |
| `run_glue_private_loraxs_shared_left_train.py` | 主干公开冻结 + LoRA-XS + shared-left 入口 | 已完成 |
| `run_bert_private_finetune_ablation.py` | 单机串行跑四组并汇总结果 | 已完成 |
| `summarize_parallel_bert_private_ablation.py` | 并行实验结果汇总 | 已完成 |
| `run_bert_private_4gpu_ablation.sh` | 四卡两波调度 driver | 已完成 |
| `run_bert_private_4gpu_ablation_nohup.sh` | 四卡两波后台启动脚本 | 已完成 |

### 6.2 已修改的原始脚本

| 文件 | 修改内容 |
| --- | --- |
| `run_glue_private_mpc_lora_train.py` | 在 summary 中补充 `final_comm_stats` 和 `total_elapsed_s`，方便比较开销 |
| `run_glue_private_light_train.py` | 同样补充 `final_comm_stats` 和 `total_elapsed_s` |

### 6.3 运行脚本

当前已经补齐单组启动脚本：

- `test_bert_base_comm_full_private_lora.sh`
- `test_bert_base_comm_loraxs_lora.sh`
- `test_bert_base_comm_loraxs.sh`
- `test_bert_base_comm_loraxs_shared_left.sh`

这些脚本的作用是方便单独跑某一组做调试或复现实验。

---

## 7. 开销统计与结果汇总

### 7.1 当前统计指标

目前代码已经统一输出以下几个对开销分析重要的字段：

- `total_elapsed_s`
- `final_comm_stats`
- `reuse_profile_summary`
- `shared_left_group_summary`
- `private_eval_metric`
- `plain_eval_metric`

这些字段最终会保存在每组实验输出目录下的：

- `train_eval_summary.json`

### 7.2 汇总脚本

当前有两套汇总方式：

- `run_bert_private_finetune_ablation.py`
  适合单机顺序跑四组
- `summarize_parallel_bert_private_ablation.py`
  适合并行/分波次跑完之后统一汇总

最终统一汇总为：

- `ablation_summary.json`
- `ablation_summary.tsv`

这意味着后续组会如果已经跑出数据，可以直接展示 TSV 表格中的：

- 总时长
- 通信统计
- 精度
- 是否公开主干
- 是否开启左复用

---

## 8. 四卡调度方案的当前处理方式

一开始的直觉方案是“4 张卡同时跑 4 组”，但这个方案后来被修正了。

原因是当前训练入口内部使用的是：

- `MultiProcessLauncher(2, ...)`

也就是说每个实验组本身就是 2-party CrypTen 任务，因此一个实验组天然要占用 2 张卡。

所以在 4 张卡环境下，正确调度方式不是“四组同开”，而是：

- 第一波：两组并行
- 第二波：两组并行

当前调度脚本就是按这个逻辑实现的：

- `GPU_PAIR_A=0,1`
- `GPU_PAIR_B=2,3`

波次安排为：

- `wave1`
  - `full_private_lora` 用 `0,1`
  - `public_backbone_lora` 用 `2,3`
- `wave2`
  - `public_backbone_loraxs` 用 `0,1`
  - `public_backbone_loraxs_shared_left` 用 `2,3`

后台运行脚本：

- `run_bert_private_4gpu_ablation_nohup.sh`

它会记录：

- `driver.log`
- 每组单独日志
- 每组 `pid`
- 最终结果输出目录

这部分已经适合直接用于“今晚后台跑，明天汇报结果”。

---

## 9. 当前进展总结

### 9.1 已完成的内容

目前已经完成的工作包括：

1. 明确了目标问题：低开销 BERT-base 私有微调。
2. 明确了四组统一对照口径，并将低秩注入位置统一为 Q/K/V。
3. 在 `newSHAFT` 中实现了 LoRA-XS 层及其注入逻辑。
4. 实现了主干公开冻结、仅保留微调部分密态的方案。
5. 实现了 LoRA-XS + shared-left 版本。
6. 补齐了开销统计字段。
7. 补齐了串行 ablation 脚本。
8. 补齐了四卡两波后台调度脚本与自动日志记录。

### 9.2 目前最重要的技术成果

可以在组会上重点强调的点有三个：

1. 不是重新写了一套 MPC 训练框架，而是在原始 `newSHAFT` 上做了最小侵入式改造。
2. 已经把“主干公开冻结 + LoRA-XS + shared-left”这条完整路线串起来了。
3. 已经把研究问题组织成清晰的四组对照，后续可以直接跑出成本和精度比较结果。

---

## 10. 当前尚未完全闭环的地方

虽然代码路径已经基本搭好，但仍有几个点需要在后续实验中进一步确认。

### 10.1 端到端跑通情况

当前更像是“框架和脚本已经搭好”，但是否已经完成完整的四组大规模正式实验，需要以实际运行结果为准。

更准确地说：

- Python 侧新增脚本已经做过静态检查
- 调度逻辑已经按“两卡一组、四卡两波”修正
- 但最终组会前是否已经拿到完整的 `ablation_summary.tsv`，取决于实际跑数情况

### 10.2 LoRA-XS 收敛性

LoRA-XS 在参数量上明显更省，但最终是否能稳定收敛、精度损失多大，仍需要实际结果来说明。

### 10.3 左复用是否真正形成有效分组

`shared-left` 的效果依赖 CrypTen 图中是否能成功识别出 Q/K/V 的共享左操作数分组。

所以正式实验中需要关注：

- `shared_left_group_summary`
- `reuse_profile_summary`

如果分组数量过少，说明左复用并没有按预期命中，需要进一步诊断。

### 10.4 分类头策略

在“主干公开冻结”的实验组中，当前设计是：

- backbone 公开冻结
- classifier 保持可训练且密态

这是为了保证下游任务仍有足够适配能力，但也意味着后续汇报时需要明确说明：

- 当前比较的是“公开主干 + 密态任务适配部分”
- 而不是“除了输入之外全部公开”

---

## 11. 下一步工作建议

如果后续继续推进，建议按下面顺序做：

1. 先正式跑通四组实验，拿到 `ablation_summary.tsv`
2. 对比四组的：
   - 总时长
   - 通信量
   - 训练稳定性
   - 私有/明文评估精度
3. 检查 `shared_left_group_summary` 是否真的形成 Q/K/V 分组
4. 如果 LoRA-XS 收敛偏弱，再调整：
   - `learning_rate`
   - `lora_r`
   - `max_train_steps`
5. 如果共享左复用收益不明显，再进一步分析图结构匹配和 fanout 条件

---

## 12. 组会可直接汇报的简版口径

可以直接用下面这段作为口头汇报主线：

“我这阶段的工作重点是把 `newSHAFT` 上的 BERT 私有微调从原始标准 LoRA 推进到一个低开销方案。具体做法是：保留原有 CrypTen 训练框架，只改低秩层和密态范围控制，把主干预训练权重公开冻结，只让输入和微调部分保持密态；然后把标准 LoRA 改成 LoRA-XS，只训练中间的 `r x r` latent mapping；最后再叠加 shared-left 左复用。现在四组对照实验的代码框架已经整理好了，包括全密态 LoRA、主干公开 LoRA、主干公开 LoRA-XS、主干公开 LoRA-XS + 左复用，并且已经补齐了四卡两波后台运行和开销汇总脚本。下一步的重点就是把四组实验完整跑出来，对比时间、通信和精度。” 

---

## 13. 相关文件索引

本轮工作最相关的文件如下：

- `examples/text-classification/loraxs_public_layers.py`
- `examples/text-classification/run_glue_private_full_private_lora_train.py`
- `examples/text-classification/run_glue_private_loraxs_lora_train.py`
- `examples/text-classification/run_glue_private_loraxs_train.py`
- `examples/text-classification/run_glue_private_loraxs_shared_left_train.py`
- `examples/text-classification/run_bert_private_finetune_ablation.py`
- `examples/text-classification/summarize_parallel_bert_private_ablation.py`
- `examples/text-classification/run_bert_private_4gpu_ablation.sh`
- `examples/text-classification/run_bert_private_4gpu_ablation_nohup.sh`
- `examples/text-classification/run_glue_private_mpc_lora_train.py`
- `examples/text-classification/run_glue_private_light_train.py`

---

## 14. BERT-base 结构总览

这一节用于在组会上快速说明：为什么原始密态 BERT-base 微调开销大，以及我们的方案具体把哪些部分“缩小了”。

### 14.1 BERT-base 模型架构

当前实验默认使用的是 Hugging Face 上的 `bert-base-uncased`。从结构上看，它是一个标准的 12 层 Transformer encoder。

核心配置如下：

- `num_hidden_layers = 12`
- `hidden_size = 768`
- `num_attention_heads = 12`
- `intermediate_size = 3072`
- `vocab_size = 30522`
- `max_position_embeddings = 512`

可以把它拆成 4 个主要部分：

1. `Embedding` 层
   包括 word embedding、position embedding、token type embedding，以及 embedding 后的 LayerNorm。
2. `12` 层 Transformer Encoder
   每层都包含：
   - 自注意力 Q/K/V 投影
   - attention output dense
   - FFN 的 intermediate dense 和 output dense
   - 两个 LayerNorm
3. `Pooler`
   把 `[CLS]` 的表示做一次线性映射和激活。
4. 任务分类头
   在 `BertForSequenceClassification` 里通常是一个 `768 -> num_labels` 的线性层。

如果只看单个 encoder layer，它的结构可以理解为：

`输入 -> Q/K/V -> attention -> attention output -> FFN(768->3072->768) -> 输出`

其中真正最“重”的参数来源，不是 Q/K/V，而是中间的 FFN 两层大矩阵。

### 14.2 BERT-base 的微调点

在当前这套实验里，微调点已经统一限定在注意力层的：

- `query`
- `key`
- `value`

也就是每层只动自注意力中的 3 个 `768 -> 768` 线性层，不再把：

- `attention.output.dense`
- `intermediate.dense`
- `output.dense`

放进主对照组的默认微调范围。

这样做的原因有 3 个：

1. 目标更清楚，直接对应“只调注意力层”。
2. Q/K/V 三个分支天然共享同一个左输入，更适合 `shared-left`。
3. 对照组差异只落在“主干是否公开、LoRA 是否变成 LoRA-XS、是否开左复用”，不会被 FFN 的额外参数干扰。

如果按 BERT-base 计算：

- 一共有 `12` 个 encoder layer
- 每层微调 `query/key/value` 共 `3` 个线性层

所以当前低秩适配实际覆盖的是：

- `12 * 3 = 36` 个线性层

### 14.3 原始密文 BERT-base 微调的参数量与主要来源

如果采用“原始全参数密态微调”的思路，那么 BERT-base 需要进入 MPC 的参数规模大约是：

- `109,482,240` 个预训练参数

如果再加上下游分类头，例如 `SST-2` 的二分类头：

- 分类头参数量约为 `768 * 2 + 2 = 1,538`

那么总参数量约为：

- `109,483,778`

这个规模解释了为什么“全参数密态 BERT-base 微调”开销会很高。

从来源拆分，大致可以分成下面几块：

| 模块 | 参数量 | 占比 | 说明 |
| --- | ---: | ---: | --- |
| Embedding 层 | `23,837,184` | 约 `21.8%` | 主要来自词表 embedding |
| 12 层 attention 子层 | `28,366,848` | 约 `25.9%` | 包含 Q/K/V、attention output 和对应 LayerNorm |
| 12 层 FFN 子层 | `56,687,616` | 约 `51.8%` | 主要来自 `768->3072` 和 `3072->768` 两个大矩阵 |
| Pooler | `590,592` | 约 `0.5%` | 占比很小 |
| 分类头 | 任务相关 | 很小 | 例如 SST-2 只有 `1,538` |

从这个表里可以看出两个关键点：

1. 原始 BERT-base 的主要参数来源是 `12` 层 encoder，尤其是 FFN 两个大矩阵。
2. 如果这些部分全部保持密态并参与训练，那么在线乘法和通信代价会非常高。

也就是说，原始密态 BERT-base 微调“贵”并不只是因为层数多，而是因为：

- 参数量本身大
- 其中大头来自 FFN 的大矩阵
- 这些矩阵如果都放进 MPC，会显著推高通信和时间成本

### 14.4 我们的方案对原始密态微调的改进

当前方案对原始密态 BERT-base 微调的改进可以概括为 4 点。

#### 改进 1：主干公开冻结，不再整网密态训练

原始全参数密态思路是：

- backbone 密态
- backbone 可训练
- 所有乘法都走密态路径

我们的方案改成：

- backbone 公开
- backbone 冻结
- 输入保持密态
- 微调参数保持密态

这样做的直接结果是：

- 大量原本昂贵的“密态激活 × 密态权重”运算，变成了“密态激活 × 公开权重”
- 整体在线代价明显下降

#### 改进 2：只在 Q/K/V 上做低秩适配，不动全参数

原始全参数微调要动的是整个 BERT-base。

当前方案只对 `query/key/value` 做适配，也就是只在 `36` 个 `768 -> 768` 线性层上加低秩更新。

这意味着：

- 不再训练 Embedding
- 不再训练 FFN 大矩阵
- 不再训练 attention output dense
- 主干结构保持不变，只加小的低秩旁路

#### 改进 3：从标准 LoRA 进一步变成 LoRA-XS

如果对 Q/K/V 使用标准 LoRA，且 `r = 8`，那么单个 `768 -> 768` 层的可训练参数量是：

- `768*8 + 8*768 = 12,288`

36 个层合计：

- `36 * 12,288 = 442,368`

如果再加上一个 SST-2 分类头：

- 总 trainable 参数约为 `443,906`

如果改成当前 LoRA-XS 方案，那么单层只训练：

- `8 * 8 = 64`

36 个层合计：

- `36 * 64 = 2,304`

如果再加分类头：

- 总 trainable 参数约为 `3,842`

所以从“可训练参数量”的角度看：

- 全参数密态微调：约 `1.09e8`
- 标准 LoRA：约 `4.44e5`
- LoRA-XS：约 `3.84e3`（以 SST-2 为例）

这就是当前方案第二层降本来源：不仅只调少量层，而且在这些层里也只训练极小的 `r x r` 核心矩阵。

#### 改进 4：利用 Q/K/V 的 shared-left 结构继续降在线代价

Q/K/V 三个分支有一个非常重要的结构特征：

- 它们共享同一个左输入 `hidden_states`

在标准执行图里，这三个分支通常会分别做自己的矩阵乘法；而在我们的方案里，`shared-left` 试图识别这些“左输入相同、右侧不同”的分支，把它们组织成可复用的组。

因此与前两点相比，`shared-left` 的作用不是减少参数量，而是：

- 减少重复的 Beaver 代价
- 降低在线阶段重复乘法带来的通信和时间成本

如果前两点解决的是“需要训练多少参数”，那么这一点解决的是“这些密态乘法怎样更便宜地执行”。

### 14.5 一句话总结

原始全参数密态 BERT-base 微调之所以昂贵，是因为：

- 需要在 MPC 中处理约 `1.09e8` 级别的参数
- 主要开销集中在 12 层 encoder，尤其是 FFN 大矩阵

我们的方案把这个问题分三步削减：

1. 主干公开冻结，只保留输入和微调部分密态
2. 只在 Q/K/V 上做低秩适配，不再训练整网
3. 将标准 LoRA 进一步改成 LoRA-XS，并结合 `shared-left` 继续降低在线代价

所以当前方案的核心不是“让 BERT 能做密态微调”，而是“让 BERT 的密态微调从不可承受的大模型全参数训练，变成一个参数量和在线代价都显著可控的实验系统”。
