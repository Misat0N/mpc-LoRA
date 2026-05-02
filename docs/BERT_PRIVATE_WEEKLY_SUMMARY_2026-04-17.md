# BERT-base 低开销密态微调周进展整理（2026-04-17）

## 1. 本周完成内容概览

本周工作主线有四条：

1. 把 `bert-base-uncased` 的 4 组密态低秩微调对照实验补齐到可运行框架。
2. 系统排查 `public backbone + LoRA-XS` 开销反而不优于 LoRA 的原因。
3. 单独拆出 `LoRA + 左复用`、`LoRA-XS + 左复用` 的开销对比，判断左复用当前到底省在什么地方。
4. 梳理本地 `.codex` 里的 skill，用于研究扫描、论文总结和组会支撑材料生成。

当前结论可以先概括为一句话：

- `public backbone` 明显有效。
- `LoRA-XS` 在线训练阶段已经比 LoRA 更省，但短步数总时长会被 setup 开销掩盖。
- `LoRA` 上的左复用已经能带来明确收益。
- `LoRA-XS` 上的左复用还没有打到最理想的位置，下一步应转向 `shared-U qkv LoRA-XS`。

---

## 2. 四组主实验结果

### 2.1 四组定义

| 组别 | 主干权重 | 低秩方式 | 是否左复用 | 主要保密对象 |
| --- | --- | --- | --- | --- |
| `full_private_lora` | 全密态 | LoRA | 否 | 整个模型 |
| `public_backbone_lora` | 公开冻结 | LoRA | 否 | 输入 + `lora_A/lora_B` + classifier |
| `public_backbone_loraxs` | 公开冻结 | LoRA-XS | 否 | 输入 + `lora_latent` + classifier |
| `public_backbone_loraxs_shared_left` | 公开冻结 | LoRA-XS | 是 | 输入 + `lora_latent` + classifier |

### 2.2 当前主结果表

说明：

- 下表主要比较训练阶段开销，核心看 `short-train finished` 和 `reuse_profile_summary`。
- `full_private_lora` 另外拿到了完整 train+eval 总时长。
- `public_backbone_lora` 早期 300 step 结果在导出阶段撞到 `decrypt()` 路径 bug，因此该组的训练统计可用，导出阶段曾失败，后续已修。

| 组别 | train steps | 训练阶段总时长(s) | 单步时间(s) | 单步通信(bytes) | triple calls | 当前精度信息 |
| --- | --- | --- | --- | --- | --- | --- |
| `full_private_lora` | 300 | 10055.29 | 33.52 | 9189985.49 | 2370 | private acc = 0.4961, plain acc = 0.4883 |
| `public_backbone_lora` | 300 | 5809.10 | 19.36 | 2879629.87 | 2127 | private acc = 0.4766，早期 plain 导出曾失败 |
| `public_backbone_loraxs` | 300 | 6130.90 | 20.44 | 2755807.79 | 2088 | 本周重点先看开销，精度未作为最终结论 |
| `public_backbone_loraxs_shared_left` | 300 | 6381.93 | 21.27 | 2640464.43 | 1766 | 本周重点先看开销，精度未作为最终结论 |

### 2.3 目前能确认的结论

1. `public backbone` 这件事是成立的。  
   从 `full_private_lora -> public_backbone_lora`，单步时间从 `33.52s` 降到 `19.36s`，训练阶段总时长从 `10055s` 降到 `5809s`。

2. `LoRA-XS` 在通信量和 triple 数上比 LoRA 更低。  
   `public_backbone_lora -> public_backbone_loraxs`，单步通信从 `2.88 MB` 降到 `2.76 MB`，triple 从 `2127` 降到 `2088`。

3. 旧版 `LoRA-XS + 左复用` 虽然进一步压低了 triple 和通信量，但没有带来更低 wall-clock。  
   这说明当前左复用应用点还没有命中 LoRA-XS 的最优结构位置。

4. 目前手里的 `LoRA-XS` 结果要分成两类看。  
   一类是修正口径后的“最新公平非复用对比”，说明 `LoRA-XS` 在线阶段已经快于 `LoRA`；另一类是“当前 shared-left 结构下的 LoRA-XS 左复用结果”，说明左复用已经压低通信和 triple，但还没有把总时长一起压下去。

---

## 3. 核心探究：LoRA 和 LoRA-XS 反常结果的纠因与 debug

本周最核心的问题不是“LoRA-XS 完全没省”，而是：

- 为什么一开始 `public backbone + LoRA-XS` 没有明显优于 LoRA；
- 为什么有些时候甚至 LoRA-XS 看起来更慢；
- 这到底是代码 bug，还是实验口径问题，还是结构设计问题。

### 3.1 已定位并修掉的问题

| 问题 | 现象 | 原因 | 当前状态 |
| --- | --- | --- | --- |
| mixed public/private 导出崩溃 | `Tensor` 没有 `get_plain_text` | `public_non_lora_weights` 后，部分参数已是明文，`decrypt()` 路径仍按密文参数处理 | 已修 |
| LoRA 的大量 `lora_B` 在 private 路径里消失 | `private_trainable_param_summary` 只有 `39/74` 命中 | 导出阶段零初始化 `lora_B` 分支被合并/折叠 | 已修，现已恢复到 `74/74` |
| “先转 CrypTen 再注入 LoRA” 路线不可用 | 注入匹配 0 个模块，或 `MatMul` 没有 `weight` | ONNX/CrypTen 转换后不再保留原始 `nn.Linear` 结构 | 已放弃该路线 |

### 3.2 纠偏后的 10 step 对比结论

为了去掉长程训练的不确定性，本周补了带 `matmul_profile` 的 10 step 对照。

| 指标 | LoRA | LoRA-XS |
| --- | --- | --- |
| `step_time_s` | 25.46 | 23.05 |
| `comm_bytes` | 86388896.0 | 82674233.6 |
| `private_calls` | 3610 | 3220 |
| `private_time_s` | 38.94 | 31.95 |
| `total_elapsed_s` | 279.16 | 277.38 |

这组结果说明：

1. 在线训练阶段，LoRA-XS 已经比 LoRA 更省。  
   单步时间下降约 `9.5%`，private matmul 调用下降约 `10.8%`，private matmul 时间下降约 `17.9%`。

2. 之所以总时长几乎拉不开，不是在线部分没省，而是 setup 吃掉了差距。  
   粗略估计：
   - LoRA setup 约 `279.16 - 10 * 25.46 = 24.5s`
   - LoRA-XS setup 约 `277.38 - 10 * 23.05 = 46.8s`
   - LoRA-XS 额外多了大约 `22s` 的前期开销

3. 因此本周已经可以把“LoRA-XS 比 LoRA 更慢”这个判断修正为：  
   `短步数总时长不明显更优`，但 `在线训练本体已经更优`。

### 3.3 最新 LoRA-XS 左复用结果应该放在哪个位置解释

你最新给出的 `LoRA-XS + 左复用` 主结果是：

| 指标 | LoRA-XS 不复用 | LoRA-XS 左复用 |
| --- | --- | --- |
| train steps | 300 | 300 |
| `step_time_s` | 20.44 | 21.27 |
| `comm_bytes` | 2755807.79 | 2640464.43 |
| `triple_generate_calls` | 2088 | 1766 |

这组结果说明的是：

1. 左复用机制已经在 `LoRA-XS` 路径里生效。  
   因为通信字节和 triple 都明显下降了。

2. 但这版 `LoRA-XS + 左复用` 还没有把“省通信/省 Beaver”转成“省 wall-clock”。  
   `step_time_s` 反而从 `20.44` 变成了 `21.27`。

3. 它和上面的公平 `10-step LoRA vs LoRA-XS` 结论并不冲突。  
   - `10-step` 非复用对比回答的是：修正口径后，`LoRA-XS` 是否已经快于 `LoRA`。答案是“是”。  
   - `300-step LoRA-XS + 左复用` 回答的是：当前这版 shared-left 是否已经把收益转成总时长下降。答案是“还没有完全做到”。

因此现在最准确的叙事应该是：

- 非复用口径下，`LoRA-XS` 已经快于 `LoRA`；
- 左复用口径下，`LoRA-XS` 已经开始省通信和 triple，但当前落点还不是最终最优结构。

### 3.4 为什么会出现这种“反常”

原因不是单一的，主要有三层：

1. 比较口径问题。  
   如果只看 `total_elapsed_s`，setup 会掩盖在线差异；如果看 `step_time_s`、`private_calls`、`private_time_s`，LoRA-XS 已经更优。

2. LoRA-XS 当前仍然有 setup 负担。  
   当前实现需要做 basis 初始化和额外结构封装，这部分在 10 step 时占比很高。

3. 旧版 LoRA-XS 结构虽然减少了一部分 `sec-sec` 乘法，但还没有把“共享左操作数”用到最理想的位置。  
   所以它省掉的 Beaver 代价，尚未完全转化成 wall-clock 收益。

---

## 4. LoRA 左复用与 LoRA-XS 左复用的开销对比

## 4.1 LoRA：左复用是有效的

下面是修正后、可直接对比的一组 `LoRA` 50 step 结果：

| 指标 | 不复用 | 左复用 | 变化 |
| --- | --- | --- | --- |
| `step_time_s` | 21.40 | 18.60 | 下降约 `13.1%` |
| `comm_bytes` | 17277779.2 | 15808871.68 | 下降约 `8.5%` |
| `triple_generate_calls` | 2127 | 1766 | 下降约 `17.0%` |
| `beaver_revealed_tensors` | 4254 | 4100 | 小幅下降 |
| `total_elapsed_s` | 1109.69 | 965.79 | 明显下降 |

解释：

- 当前左复用已经能在 `LoRA` 上真正省到东西；
- 省下来的主要是 triple 生成和部分通信字节；
- reveal 没有同比例下降，说明当前主要省在 Beaver 掩码准备阶段，而不是把 opening 次数本身大量压掉。

## 4.2 LoRA-XS：最新左复用结果表明通信下降了，但 wall-clock 还没跟上

下面是本周已有的 `LoRA-XS` 300 step 对比：

| 指标 | 不复用 | 左复用 | 变化 |
| --- | --- | --- | --- |
| `step_time_s` | 20.44 | 21.27 | 反而上升 |
| `comm_bytes` | 2755807.79 | 2640464.43 | 小幅下降 |
| `triple_generate_calls` | 2088 | 1766 | 明显下降 |

这说明：

1. 左复用机制本身在工作。  
   triple 的确降了，通信字节也降了。

2. 但它现在还没有转化成更低的单步总时长。  
   这意味着当前复用位置不够“值钱”，或者复用管理开销抵消了部分收益。

3. 现阶段的判断不是“左复用没用”，而是“LoRA-XS 上的左复用打点还不够对”。  
   所以下一步应从 `q/k/v 各自的 XA`，转向 `shared-U qkv LoRA-XS`：
   - 每层先统一算一次 `T = XU`
   - 再分别走 `T @ G_q`、`T @ G_k`、`T @ G_v`
   - 左复用直接打在三个 `T @ G_*` 分支上

4. 这组 `LoRA-XS + 左复用` 结果当前更适合在组会上被表述为“机制有效、落点待优化”。  
   不适合直接表述成“LoRA-XS 左复用已经优于 LoRA-XS 不复用”。

### 4.3 当前阶段的结论

- `LoRA + 左复用`：已经能作为有效结果汇报。
- `LoRA-XS + 左复用`：当前结果更多是“定位问题”的中间版本，还不适合作为最终设计定稿。

---

## 5. 本周代码和实验层面的关键进展

### 5.1 已补齐的核心能力

1. 4 组对照实验脚本结构已经齐全。  
   入口集中在 `newSHAFT/examples/text-classification/`。

2. `matmul_profile` 已补进训练框架。  
   现在可以区分：
   - public matmul 次数和时间
   - private matmul 次数和时间
   - 估计乘法量

3. `private_trainable_param_summary` 现在能更可靠地核对：  
   - 预期可训练参数
   - 实际进入 CrypTen private 路径的参数

4. `lora_B` 导出折叠问题已经通过 sentinel-and-restore 方案绕开。  
   这使得 LoRA 组的私有训练参数终于回到正确口径。

### 5.2 当前仍未完全结束的点

1. `LoRA-XS + 左复用` 还不是最终结构。
2. 短步数实验主要适合看开销，不适合解释最终精度。
3. 目前多组结果的重点是“成本侧结论”，不是“收敛侧结论”。

---

## 6. `.codex` 里的 skill 梳理

当前本地 `.codex` 下明确可用的核心 skill 是：

| skill | 路径 | 用途 |
| --- | --- | --- |
| `research-assistant` | `.codex/skills/research-assistant/SKILL.md` | 论文总结、论文比较、实验规划、每周 frontier scan、研究笔记整理 |

### 6.1 `research-assistant` 的定位

这个 skill 是一个研究型、默认只读的工作流，适合做：

1. 单篇论文总结
2. 多篇论文比较
3. 方法与局限提取
4. 最新论文搜索
5. 每周自动 frontier scan
6. 实验计划和 related work 草稿生成

### 6.2 其内部资源结构

| 类型 | 代表文件 | 作用 |
| --- | --- | --- |
| 配置 | `.codex/skills/research-assistant/config/weekly_frontier_scan.json` | 每周扫描主题和输出位置 |
| 脚本 | `search_latest_research.py` | 搜最近论文 |
| 脚本 | `generate_weekly_frontier_scan.py` | 生成周扫描 Markdown |
| 脚本 | `install_weekly_scan_task.ps1` | 安装 Windows 定时任务 |
| 模板 | `templates/paper_summary.md` | 单篇论文笔记模板 |
| 模板 | `templates/paper_compare.md` | 多论文对比模板 |
| 模板 | `templates/experiment_plan.md` | 实验计划模板 |
| 模板 | `templates/literature_review.md` | related work / 综述模板 |

### 6.3 对当前课题的直接帮助

这个 skill 对当前课题最有价值的地方有两个：

1. 能持续把 `LoRA / LoRA-XS / MPC / reuse` 相关论文整理成可复用的阅读卡片。
2. 能把每周的实验结果、debug 结论和下一步实验想法，沉淀成比较规范的组会材料。

---

## 7. 适合组会汇报时直接说的结论

如果只保留最核心的三句话，可以这样汇报：

1. 本周已经把 `full private LoRA / public backbone LoRA / public backbone LoRA-XS / public backbone LoRA-XS + shared-left` 四组实验框架补齐，并拿到了稳定的训练期开销画像。

2. `LoRA-XS` 的“反常慢”主要不是在线训练真的更慢，而是早期被导出 bug 和 setup 开销掩盖；在修正统计后，`LoRA-XS` 在线阶段已经比 LoRA 更省。

3. 左复用在 `LoRA` 上已经能稳定降开销；在 `LoRA-XS` 上当前还没打到最优位置，下一步要转向 `shared-U qkv LoRA-XS`，让左复用直接服务于三路 q/k/v 的共享投影。

---

## 8. 建议的下周主线

1. 把 `shared-U qkv LoRA-XS` 版本做出来，重点看 `step_time_s`、`private_calls`、`comm_bytes` 是否继续下降。
2. 长步数实验与短步数 debug 分开管理，避免再用 `total_elapsed_s` 混淆 setup 和 online 开销。
3. 精度侧实验单独按 recipe 对齐后再跑，避免把“开销问题”和“收敛问题”混在一起解释。
