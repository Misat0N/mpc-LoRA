# MPC-LoRA 训练与脚本实现说明（中文详解）

## 1. 文档目的

本文档用于解释当前仓库中 MPC-LoRA 训练脚本与配套 shell 脚本的代码逻辑、实现流程、设计取舍、日志含义以及输出物结构。

阅读目标：

1. 明确哪些文件负责“冒烟验证”，哪些文件负责“正式 MPC-LoRA 训练”。
2. 理解一次训练任务从 shell 启动到模型保存的完整调用链。
3. 理解 LoRA 是如何注入到 BERT 中的，以及它如何进入 CrypTen/MPC 训练域。
4. 理解“私有化验证”和“明文验证”的差异。
5. 理解当前实现中最容易出问题的环节，以及为什么采用当前的恢复/评估方案。

本文档基于当前仓库版本编写，关键代码文件如下：

- `examples/text-classification/run_glue_private_train_smoke.py`
- `examples/text-classification/run_glue_private_mpc_lora_train.py`
- `examples/text-classification/test_bert_base_comm_mpc_lora.sh`
- `examples/text-classification/test_bert_base_comm_mpc_lora_overnight.sh`
- `examples/text-classification/run_mpc_lora_overnight_nohup.sh`

---

## 2. 文件职责总览

### 2.1 冒烟脚本

文件：`examples/text-classification/run_glue_private_train_smoke.py:246`

职责：

- 验证 CrypTen + BERT + backward 能否跑通。
- 训练步数很短，重点是“链路是否能执行”，不是效果。
- 适合定位初始化、前向、反向、通信和基础 loss 计算问题。

它不是正式的 LoRA 微调脚本。

### 2.2 正式 MPC-LoRA 训练脚本

文件：`examples/text-classification/run_glue_private_mpc_lora_train.py:633`

职责：

- 加载 HuggingFace 的 BERT 分类模型。
- 在 PyTorch 模型中注入 LoRA。
- 只训练 LoRA 参数（可选附带训练分类头）。
- 将改造后的模型整体转换为 CrypTen 私有模型。
- 在 MPC 环境中完成训练。
- 训练后做私有化评估。
- 尝试恢复为普通 PyTorch 模型，做明文评估与模型导出。
- 保存结果 summary 与模型文件。

### 2.3 标准启动脚本

文件：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:1`

职责：

- 组织一组标准训练参数。
- 生成本次运行目录。
- 将环境变量映射为 Python 命令参数。
- 启动 `run_glue_private_mpc_lora_train.py`。

### 2.4 过夜启动脚本

文件：`examples/text-classification/test_bert_base_comm_mpc_lora_overnight.sh:1`

职责：

- 覆盖更长时长的训练默认值。
- 用于整夜跑任务。
- 最终仍调用标准启动脚本。

### 2.5 后台启动脚本

文件：`examples/text-classification/run_mpc_lora_overnight_nohup.sh`

职责：

- 将过夜训练以 `nohup` 方式放到后台运行。
- 自动生成日志文件与 pid 文件。

---

## 3. 训练任务的整体调用链

一次标准前台训练的调用链如下：

1. 执行 shell：`bash test_bert_base_comm_mpc_lora.sh`
2. shell 组装参数与输出目录：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:4`
3. shell 调用 Python：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:52`
4. Python 解析参数：`examples/text-classification/run_glue_private_mpc_lora_train.py:407`
5. Python 加载数据、模型、注入 LoRA、构建 dataloader
6. Python 初始化 CrypTen 与 GPU 映射
7. Python 将带 LoRA 的 PyTorch 模型转换为 CrypTen 私有模型
8. 在 MPC 域进行训练
9. 做私有化评估
10. decrypt 后尝试恢复为明文 PyTorch 模型
11. 做明文评估（若恢复成功）
12. 保存 summary 和训练后模型

过夜模式只是在第 1 步之前多了一层包装：

`bash test_bert_base_comm_mpc_lora_overnight.sh`

它本质上只是把：

- `MAX_TRAIN_STEPS`
- `EVAL_MAX_STEPS`
- `LEARNING_RATE`
- `RUN_TAG`

这些默认值改成更适合长跑的版本，然后继续调用标准脚本。

---

## 4. Shell 脚本层的实现逻辑

### 4.1 标准脚本如何组织实验参数

文件：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:4`

脚本首先定义了一系列环境变量默认值：

- `TASK_NAME`
- `GPU_IDS`
- `SEED`
- `MAX_LENGTH`
- `LEN_DATA`
- `MAX_TRAIN_STEPS`
- `LOG_EVERY_STEPS`
- `EVAL_MAX_STEPS`
- `LORA_R`
- `LORA_ALPHA`
- `LORA_DROPOUT`
- `LORA_TARGET_MODULES`
- `LEARNING_RATE`
- `MOMENTUM`

这些变量的作用是：

1. 让命令行保持简洁。
2. 方便用户通过环境变量覆盖默认值。
3. 让过夜脚本只改“少量关键参数”，而不需要重复整条 Python 命令。

### 4.2 输出目录管理

文件：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:27`

脚本用当前时间戳构造：

`eval_private/${TASK_NAME}/${RUN_TAG}`

这意味着每次运行都会产生一个独立目录，避免覆盖前一次结果。

### 4.3 条件参数透传

文件：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:31`

脚本把一些布尔开关收集到 `EXTRA_ARGS`：

- `--freeze_classifier_head`
- `--skip_private_eval`
- `--skip_plain_eval`
- `--print_comm_cost`
- `--allow_spam_logs`

这样 shell 本身不需要分支重写整条命令，只需要在需要时附加额外参数。

### 4.4 实际 Python 调用

文件：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:52`

这里是真正的训练入口。脚本将所有变量映射为 Python 参数，传给：

`run_glue_private_mpc_lora_train.py`

关键代码片段：

来源：`examples/text-classification/test_bert_base_comm_mpc_lora.sh:27` 与 `examples/text-classification/test_bert_base_comm_mpc_lora.sh:52`

```bash
RUN_TAG="${RUN_TAG:-mpc_lora_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="eval_private/${TASK_NAME}/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

python run_glue_private_mpc_lora_train.py \
  --model_name_or_path andeskyl/bert-base-cased-${TASK_NAME} \
  --task_name ${TASK_NAME} \
  --gpu_ids ${GPU_IDS} \
  --max_train_steps ${MAX_TRAIN_STEPS} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --learning_rate ${LEARNING_RATE} \
  --momentum ${MOMENTUM} \
  --output_dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}"
```

这段代码说明 shell 层只负责两件事：生成实验目录、把环境变量整理成 Python 参数。

---

## 5. Python 训练脚本的总体结构

主入口：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:633`

可以把这个脚本拆成 8 个阶段：

1. 参数解析与运行模式配置
2. 数据集加载与预处理
3. 模型加载与 LoRA 注入
4. CrypTen 初始化与 GPU 映射
5. 私有模型构建
6. MPC 训练循环
7. 私有化评估
8. 明文恢复、明文评估、保存输出

下面逐段说明。

---

## 6. 参数解析与运行模式

参数解析函数：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:407`

### 6.1 训练参数

常用参数：

- `--max_train_steps`：`examples/text-classification/run_glue_private_mpc_lora_train.py:479`
- `--log_every_steps`：`examples/text-classification/run_glue_private_mpc_lora_train.py:485`
- `--eval_max_steps`：`examples/text-classification/run_glue_private_mpc_lora_train.py:491`
- `--per_device_train_batch_size`：`examples/text-classification/run_glue_private_mpc_lora_train.py:467`
- `--per_device_eval_batch_size`：`examples/text-classification/run_glue_private_mpc_lora_train.py:473`

### 6.2 LoRA 参数

- `--lora_r`：`examples/text-classification/run_glue_private_mpc_lora_train.py:545`
- `--lora_alpha`：`examples/text-classification/run_glue_private_mpc_lora_train.py:551`
- `--lora_dropout`：`examples/text-classification/run_glue_private_mpc_lora_train.py:557`
- `--lora_target_modules`：`examples/text-classification/run_glue_private_mpc_lora_train.py:563`
- `--freeze_classifier_head`：`examples/text-classification/run_glue_private_mpc_lora_train.py:569`

### 6.3 评估控制参数

- `--skip_private_eval`：`examples/text-classification/run_glue_private_mpc_lora_train.py:586`
- `--skip_plain_eval`：`examples/text-classification/run_glue_private_mpc_lora_train.py:591`

### 6.4 quick_run 模式

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:637`

它会自动做以下限制：

- 将训练步数压到 1
- 将最大长度压缩到 32
- 限制样本数
- 跳过 private/plain eval
- 降低 LoRA rank

用途是：

- 快速验证链路是否能执行
- 不用于正式收敛实验

---

## 7. 数据集加载与预处理

### 7.1 加载 GLUE 数据集

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:714`

如果指定 `task_name`，就从 `nyu-mll/glue` 下载对应任务数据。

当前常用任务是：

- `sst2`

### 7.2 截断训练/评估样本数

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:729`

如果设置：

- `--train_max_samples`
- `--eval_max_samples`

脚本会在 tokenization 前直接裁切数据集，这对快跑验证很有用。

### 7.3 tokenizer 与 label 处理

模型和 tokenizer 加载：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:760`

文本预处理函数：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:858`

主要逻辑：

1. 根据任务找到句子字段
2. 用 tokenizer 编码
3. padding/truncation 到 `max_length`
4. 将 `label` 转成 `labels`

### 7.4 dataloader 构建

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:894`

训练集 dataloader：

- `shuffle=True`：`examples/text-classification/run_glue_private_mpc_lora_train.py:900`

这是正式训练与早期 smoke 版本的重要差别之一。正式训练需要随机打乱数据。

---

## 8. 模型加载与 LoRA 注入

### 8.1 先加载标准 BERT 分类模型

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:772`

这里仍然是 HuggingFace 原生的：

- `AutoModelForSequenceClassification`

也就是说，LoRA 不是外部黑箱，而是在这个 PyTorch 模型对象内部被改造出来的。

### 8.2 LoRALinear 的结构

定义位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:187`

它的内部结构是：

1. 保存原始线性层到 `self.base`
2. 冻结 `self.base` 的参数：`examples/text-classification/run_glue_private_mpc_lora_train.py:201`
3. 新建两个线性层：
   - `lora_A`：低秩降维：`examples/text-classification/run_glue_private_mpc_lora_train.py:205`
   - `lora_B`：低秩升维：`examples/text-classification/run_glue_private_mpc_lora_train.py:206`
4. 前向逻辑：
   - `base(x) + lora_B(lora_A(x)) * scaling`
   - 位置：`examples/text-classification/run_glue_private_mpc_lora_train.py:213`

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:187`

```python
class LoRALinear(nn.Module):
    def __init__(self, base_layer, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base_layer
        self.r = int(r)
        self.alpha = int(alpha)
        self.scaling = float(alpha) / float(r) if r > 0 else 0.0
        self.lora_dropout_p = float(dropout)

        for p in self.base.parameters():
            p.requires_grad = False

        if self.r > 0:
            self.lora_A = nn.Linear(self.base.in_features, self.r, bias=False)
            self.lora_B = nn.Linear(self.r, self.base.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        result = self.base(x)
        if self.r > 0:
            x_lora = F.dropout(x, p=self.lora_dropout_p, training=self.training) if self.lora_dropout_p > 0 else x
            lora_out = self.lora_B(self.lora_A(x_lora)) * self.scaling
            result = result + lora_out
        return result
```

这段代码对应 LoRA 的本质实现：冻结原线性层，只在旁路新增低秩增量。

### 8.3 为什么这样实现 LoRA

目的不是重写整个注意力模块，而是：

- 保留原模型计算路径
- 在目标线性层上增加一个低秩增量
- 只训练低秩分支的参数

这样做的优点：

1. 参数量小
2. 结构侵入性低
3. 与已有 HuggingFace 模型兼容性高

### 8.4 为什么 LoRA 注入发生在 CrypTen 转换之前

注入位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:779`

这一步发生在：

- PyTorch 明文模型阶段

然后才执行：

- `ct.nn.from_pytorch(...)`

原因：

1. CrypTen 本身没有原生 LoRA API
2. 最稳妥的方式是先构造出“已经带 LoRA 的标准 PyTorch 模型”
3. 再让 CrypTen 统一接收并转换整个模型

这也是当前实现本质上“LoRA 结构 + MPC 训练”的原因。

### 8.5 只训练 LoRA 参数

逻辑位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:243`

规则：

- `lora_A.*`
- `lora_B.*`
- 可选 `classifier.*` / `score.*`

如果启用：

- `--freeze_classifier_head`

则分类头也被冻结，只训练 LoRA 参数。

如果启用：

- `--train_classifier_only`

则强制只训练分类头，这会覆盖 LoRA 训练选择：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:793`

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:222` 与 `examples/text-classification/run_glue_private_mpc_lora_train.py:243`

```python
def _inject_lora_layers(module, target_keywords, r, alpha, dropout, prefix=""):
    replaced = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.Linear) and any(k in full_name for k in target_keywords):
            setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced.append(full_name)
        else:
            replaced.extend(
                _inject_lora_layers(
                    child,
                    target_keywords=target_keywords,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                    prefix=full_name,
                )
            )
    return replaced

def _set_lora_trainable(model, train_classifier_head=True):
    trainable = []
    for name, param in model.named_parameters():
        is_lora = ("lora_A." in name) or ("lora_B." in name)
        is_classifier = train_classifier_head and (name.startswith("classifier.") or name.startswith("score."))
        param.requires_grad = is_lora or is_classifier
        if param.requires_grad:
            trainable.append(name)
    return trainable
```

前一段决定“替换哪些层”，后一段决定“训练哪些参数”。

---

## 9. CrypTen 初始化与 GPU 映射

### 9.1 CrypTen 初始化

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:909`

逻辑：

1. 如果 launcher 已初始化过，就跳过 `ct.init()`
2. 否则显式初始化

这能减少重复初始化带来的噪声和不确定性。

### 9.2 GPU 分配

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:922`

逻辑：

1. 解析 `--gpu_ids`
2. 根据当前 `rank` 选择一张 GPU
3. 调用 `torch.cuda.set_device`

所以：

- `GPU_IDS=0,1`

表示：

- rank 0 用 GPU 0
- rank 1 用 GPU 1

这是两方 MPC 的设备映射方式。

---

## 10. 从 PyTorch 模型到私有模型

关键代码：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:949`

流程：

1. 构造 dummy 输入
2. 调用 `ct.nn.from_pytorch(model, dummy_inputs)`
3. 得到 CrypTen 模型
4. `encrypt()`
5. 放到指定 GPU 上

这一步之后，模型已经不再是普通的 PyTorch 训练，而是进入 CrypTen/MPC 训练域。

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:949`

```python
dummy = torch.zeros_like(model.dummy_inputs["input_ids"])
private_model = ct.nn.from_pytorch(model, (dummy, dummy, dummy)).encrypt().to(device)
private_model.train()
optimizer = ct.optim.SGD(private_model.parameters(), lr=lr, momentum=args.momentum)
```

这一段是明文模型进入 MPC 域的边界线。此后前向、反向和参数更新都发生在 CrypTen 私有模型上。

---

## 11. MPC 训练循环

训练循环入口：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1007`

### 11.1 每个 batch 做了什么

#### 第一步：过滤长度

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1025`

如果 `len_data > 0` 且 batch 序列长度不等于该值，则跳过。

#### 第二步：构造密文输入

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1031`

把：

- `input_ids`
- `attention_mask`
- `token_type_ids`

转成 `ct.cryptensor` 并放到对应 GPU。

#### 第三步：私有前向

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1038`

此时前向是在 CrypTen 私有模型中完成。

#### 第四步：构造损失

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1047`

当前使用：

- one-hot 标签
- MSE loss

而不是标准交叉熵。

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:1033`

```python
token_type_ids = batch.get("token_type_ids")
if token_type_ids is None:
    token_type_ids = torch.zeros_like(batch["input_ids"])

inputs_enc = ct.cryptensor(batch["input_ids"]).to(device)
attention_mask_enc = ct.cryptensor(batch["attention_mask"]).to(device)
token_type_enc = ct.cryptensor(token_type_ids).to(device)

logits_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)

num_labels = logits_enc.size(-1)
y_onehot = F.one_hot(batch["labels"], num_classes=num_labels).float()
y_enc = ct.cryptensor(y_onehot).to(device)

diff = logits_enc - y_enc
loss_enc = (diff * diff).mean()
```

这段代码体现的是“输入、标签、损失都在 MPC 域内计算”。

### 11.2 为什么使用 MSE 而不是 CE

这是工程实现上的稳定性选择。

原因：

1. 在 CrypTen/MPC 环境中，MSE 的实现路径更简单。
2. one-hot + MSE 对于“验证训练链路”和“小规模实验”更稳定。
3. 代价是它不一定是最适合 NLP 分类微调的 loss。

因此它是一个务实折中：

- 优先保证训练链路可执行
- 不代表它是最终最优训练配置

#### 第五步：反向传播

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1077`

这里是真正的私有化 backward。

#### 第六步：参数更新

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1090`

使用的是：

- `ct.optim.SGD`

其学习率和 momentum 来自命令行参数。

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:1063`

```python
optimizer.zero_grad()

try:
    loss_enc.backward()
except Exception:
    logger.exception(
        "[rank %s] train_step=%03d backward_failed cfg=%s loss=%s",
        rank,
        global_step,
        _cfg_snapshot(),
        _loss_snapshot(loss_enc),
    )
    raise

optimizer.step()
```

这就是私有训练真正发生的地方：`backward()` 触发 CrypTen 的反向图，`step()` 更新私有参数。

#### 第七步：揭示 loss 做日志

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1097`

这里通过：

- `loss_enc.get_plain_text().item()`

把 loss 解出来，只用于日志显示。

这也是你在日志中看到：

- `[train] step=... loss=...`

的来源。

---

## 12. 私有化验证（Private Eval）

入口：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1111`

### 12.1 定义

“私有化验证”指的是：

1. 模型仍然是 CrypTen 私有模型
2. 输入仍然被转换为 `ct.cryptensor`
3. 前向在 MPC 域内部完成
4. 只把最终输出解出来用于计算准确率

### 12.2 实现流程

- 切到 eval：`examples/text-classification/run_glue_private_mpc_lora_train.py:1114`
- 逐 batch 构造密文输入：`examples/text-classification/run_glue_private_mpc_lora_train.py:1127`
- `ct.no_grad()` 下前向：`examples/text-classification/run_glue_private_mpc_lora_train.py:1130`
- 把输出解成普通 tensor：`examples/text-classification/run_glue_private_mpc_lora_train.py:1133`
- 转成类别预测并计入 metric：`examples/text-classification/run_glue_private_mpc_lora_train.py:1134`

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:1116`

```python
private_eval_metric = {"skipped": True, "reason": "disabled"}
if not args.skip_private_eval:
    private_model.eval()
    private_metric = evaluate.load("glue", args.task_name) if args.task_name is not None else evaluate.load("accuracy")

    inputs_enc = ct.cryptensor(batch["input_ids"]).to(device)
    attention_mask_enc = ct.cryptensor(batch["attention_mask"]).to(device)
    token_type_enc = ct.cryptensor(token_type_ids).to(device)
    with ct.no_grad():
        outputs_enc = private_model(inputs_enc, attention_mask_enc, token_type_enc)

    outputs = outputs_enc.get_plain_text().cpu()
    predictions = outputs.argmax(dim=-1) if not is_regression else outputs.squeeze()
    private_metric.add_batch(predictions=predictions, references=batch["labels"])
```

这里最重要的区别是：模型和输入都还是私有的，只有最终输出被解出来算指标。

### 12.3 结果含义

`private_eval_metric` 表示：

- 在“私有模型 + 私有输入”的前提下得到的验证指标

它能证明：

- 私有训练和私有推理链路是通的

---

## 13. 明文恢复与明文验证（Plain Eval）

这是当前实现最复杂，也最容易出问题的一段。

### 13.1 为什么需要明文验证

目的有两个：

1. 作为对照，验证恢复后的普通 PyTorch 模型是否可用
2. 保存训练后的明文模型用于后续常规推理或分析

### 13.2 基本流程

入口：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1152`

流程：

1. `private_model.decrypt()`：`examples/text-classification/run_glue_private_mpc_lora_train.py:1159`
2. 将其搬到 CPU：`examples/text-classification/run_glue_private_mpc_lora_train.py:1162`
3. 调用 `_recover_plain_model_from_private(...)`：`examples/text-classification/run_glue_private_mpc_lora_train.py:1165`
4. 若恢复成功，使用普通 PyTorch 明文数据做评估：`examples/text-classification/run_glue_private_mpc_lora_train.py:1179`

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:1162`

```python
need_decrypt = (not args.skip_plain_eval) or (args.output_dir is not None)
trained_model = None
if need_decrypt:
    private_model.decrypt()
    private_model = private_model.to("cpu")

if rank == 0 and ((not args.skip_plain_eval) or (args.output_dir is not None)):
    try:
        trained_model = _recover_plain_model_from_private(private_model, model, rank)
        if not args.skip_plain_eval:
            trained_model = trained_model.to(device)
        trained_model.eval()
    except Exception as err:
        plain_recovery_error = f"{type(err).__name__}: {err}"
        trained_model = None
```

这一段说明明文评估依赖一个单独的“恢复”阶段，而不是训练结束后自动天然得到。

### 13.3 为什么不直接依赖 `to_pytorch()`

历史上这里曾经使用：

- `private_model.to_pytorch()`

但在自定义 LoRA 包装层下，这条路径对模块结构比较敏感，容易在遍历 `Identity` 或 CrypTen 包装模块时崩溃。

所以当前实现改成：

- 优先走 `state_dict` 恢复

也就是：

1. 从 CrypTen 私有模型中拿 `state_dict`
2. 把每个值解成 CPU tensor
3. 把 key 归一化
4. 回填到原始 PyTorch 模型模板

核心函数：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:327`

### 13.4 key/value 归一化逻辑

当前实现解决了几个具体问题：

#### 1. value 可能包在多层对象里

函数：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:254`

它会递归尝试：

- `get_plain_text()`
- `.data`
- `._tensor`
- `.share`

直到拿到普通 `torch.Tensor`

#### 2. key 可能带内部后缀

函数：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:283`

会去掉类似：

- `.data`
- `._tensor`
- `.share`

后缀。

#### 3. key 可能带 CrypTen 内部路径字段

函数：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:295`

会清理：

- `_modules`
- `_parameters`
- `_buffers`

并尝试生成候选 key：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:305`

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:254` 至 `examples/text-classification/run_glue_private_mpc_lora_train.py:339`

```python
def _to_torch_state_value(value, _seen=None):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if hasattr(value, "get_plain_text"):
        try:
            plain = value.get_plain_text()
            if torch.is_tensor(plain):
                return plain.detach().cpu()
        except Exception:
            pass
    for attr in ("data", "_tensor", "share"):
        if hasattr(value, attr):
            nested = _to_torch_state_value(getattr(value, attr), _seen)
            if nested is not None:
                return nested
    return None

def _strip_state_key_suffixes(key):
    for suffix in (".data", "._tensor", ".share"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key

def _recover_plain_model_from_private(private_model, template_model, rank):
    ct_state = private_model.state_dict()
    pt_state = template_model.state_dict()
    ...
    if mapped == 0:
        raise RuntimeError("state_dict_recovery_no_match")
```

这几段代码连起来，就是当前明文恢复链路的骨架：先解值，再清 key，再映射回模板模型。

### 13.5 恢复失败时会发生什么

如果恢复失败：

- `plain_eval_metric` 会被写成：
  - `skipped: true`
  - `reason: plain_model_recovery_failed`
  - `error: ...`

代码位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1169`

好处是：

1. 不再因为明文恢复失败把整个训练任务打死
2. 仍然会写 summary
3. 便于诊断到底是训练失败还是恢复失败

---

## 14. 结果保存逻辑

保存逻辑位置：

- `examples/text-classification/run_glue_private_mpc_lora_train.py:1221`

### 14.1 模型保存

如果 `trained_model` 存在：

- 保存到 `output_dir/trained_model`
- 保存模型权重、config、tokenizer

### 14.2 若明文模型不可用

脚本会：

- 不保存空模型目录
- 记录 warning：`[save] skip trained model export: plaintext model unavailable`

### 14.3 summary 保存

无论明文模型是否恢复成功，都会尽量写：

- `train_eval_summary.json`

字段包括：

- `train_steps`
- `private_eval_metric`
- `plain_eval_metric`
- `task_name`
- `max_train_steps`
- `eval_max_steps`

这是判断本次运行结果的第一诊断文件。

关键代码片段：

来源：`examples/text-classification/run_glue_private_mpc_lora_train.py:1226`

```python
if rank == 0 and args.output_dir is not None:
    trained_model_dir = os.path.join(args.output_dir, "trained_model")
    if trained_model is not None:
        os.makedirs(trained_model_dir, exist_ok=True)
        trained_model.save_pretrained(trained_model_dir)
        tokenizer.save_pretrained(trained_model_dir)
    else:
        logger.warning("[save] skip trained model export: plaintext model unavailable")

    summary = {
        "train_steps": global_step,
        "private_eval_metric": private_eval_metric,
        "plain_eval_metric": plain_eval_metric,
        "task_name": args.task_name,
        "max_train_steps": args.max_train_steps,
        "eval_max_steps": args.eval_max_steps,
    }
    summary_path = os.path.join(args.output_dir, "train_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
```

这段代码决定了最终产物：`trained_model/` 是否存在，以及 `train_eval_summary.json` 一定要尽量落盘。

---

## 15. 日志应该如何理解

### 15.1 训练前日志

例如：

- `DistributedCommunicator with rank 0/1`
- `World size = 2`

表示：

- MPC 双方进程已建立通信

### 15.2 数据预处理日志

- `Running tokenizer on dataset`

表示：

- 正在做文本编码

### 15.3 LoRA 注入日志

- `[lora] injected=...`

表示：

- 成功替换了多少个目标线性层

### 15.4 训练日志

- `[train] step=... loss=...`

表示：

- 当前私有训练正在正常前进

### 15.5 私有评估日志

- `[eval-private] ...`

表示：

- 私有模型评估已完成

### 15.6 明文恢复失败日志

- `recover_plain_model_failed`
- `plain_model_recovery_failed`

表示：

- 训练本身未必失败
- 失败发生在“恢复普通 PyTorch 模型”这一步

---

## 16. 当前实现的工程取舍

### 16.1 优先保证私有训练主链路

当前优先级是：

1. 模型能完成 MPC 训练
2. 模型能完成 private eval
3. 再去稳定 plain recovery / plain eval

这是因为：

- 当前真正的核心目标是“私有域中的 LoRA 微调”
- 明文恢复是导出与对照验证链路，不是训练主链路本身

### 16.2 为什么默认 target 是 `query,value`

原因：

1. 这是 LoRA 在 transformer attention 中最常见、最稳妥的注入位置
2. 参数量适中
3. 训练代价比全层注入低

### 16.3 为什么 batch size 仍然很小

因为：

- CrypTen + BERT + MPC backward 的显存和通信成本都很高
- 当前脚本更关注先跑稳，再放大规模

---

## 17. 训练脚本与冒烟脚本的本质差异

### 冒烟脚本

文件：`examples/text-classification/run_glue_private_train_smoke.py`

特点：

- 无 LoRA
- 步数很少
- 主要验证 backward 是否可跑
- 更像链路连通性测试

### MPC-LoRA 脚本

文件：`examples/text-classification/run_glue_private_mpc_lora_train.py`

特点：

- 注入 LoRA
- 控制可训练参数
- 有完整训练流程
- 有 private eval
- 尝试 plain eval 和模型导出
- 输出 summary

因此它才是你当前真正用于实验的训练脚本。

---

## 18. 建议的阅读顺序

如果你要自己继续改代码，建议按这个顺序看：

1. shell 启动脚本  
   - `examples/text-classification/test_bert_base_comm_mpc_lora.sh`
2. 参数解析  
   - `examples/text-classification/run_glue_private_mpc_lora_train.py:407`
3. LoRA 注入逻辑  
   - `examples/text-classification/run_glue_private_mpc_lora_train.py:187`
   - `examples/text-classification/run_glue_private_mpc_lora_train.py:222`
4. 训练循环  
   - `examples/text-classification/run_glue_private_mpc_lora_train.py:1007`
5. 私有评估  
   - `examples/text-classification/run_glue_private_mpc_lora_train.py:1111`
6. 明文恢复  
   - `examples/text-classification/run_glue_private_mpc_lora_train.py:327`
7. 保存逻辑  
   - `examples/text-classification/run_glue_private_mpc_lora_train.py:1221`

---

## 19. 当前输出物应该怎么看

每次训练会生成一个目录：

- `eval_private/sst2/mpc_lora_<timestamp>/`

重点看两个位置：

1. `train_eval_summary.json`
2. `trained_model/`

如果 `train_eval_summary.json` 存在但 `trained_model/` 不存在或为空，通常说明：

- 训练主流程完成了
- 但明文模型恢复或导出失败

此时优先查看：

- `private_eval_metric`
- `plain_eval_metric.reason`

而不是先怀疑训练本身失败。

---

## 20. 一句话总结

当前这套 MPC-LoRA 方案的本质是：

先在 HuggingFace 的 BERT 分类模型上注入 LoRA 低秩适配层，只保留 LoRA 参数为可训练参数；再将这个改造后的模型整体转换为 CrypTen 私有模型，在 MPC 域中完成训练与私有评估；训练结束后再尝试通过 `state_dict` 恢复普通 PyTorch 模型，以完成明文评估和模型导出。

如果后续需要，还可以继续补一份文档，专门解释：

1. `private_eval_metric` 与 `plain_eval_metric` 的对齐关系  
2. 为什么当前 accuracy 偏低  
3. 后续如何把 loss、优化器和 LoRA target 调整得更适合正式实验
