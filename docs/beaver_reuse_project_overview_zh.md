# CrypTen Beaver 掩码复用项目总览

## 1. 这份文档是写给谁的

这份文档面向两类读者：

- 第一次接触 CrypTen / MPC / Beaver 协议的人
- 想快速理解本项目“做了什么、改了哪里、为什么会变快、怎么看实验结果”的工程同学

文档目标不是只列文件名，而是把整个项目讲清楚：

- 原始系统怎么工作
- 原始模型是什么样
- Beaver triple 原本怎么参与训练
- 复用是在哪些地方插进去的
- `FIX_A` / `FIX_AB` 各自到底在复用什么
- 哪些指标说明性能提升
- 指标的含义是什么
- 目前实验结论是什么

整套内容都是围绕“纯性能实验模式”展开的，即把重点放在**减少开销**，不讨论其它主题。

---

## 2. 先用一句话讲清楚这个项目

这个项目的核心，是在 CrypTen 的 Beaver 乘法协议里加入一个**单 step 内的掩码复用机制**，让神经网络训练中多个相关的矩阵乘法可以共享同一组 Beaver mask（`A`，以及可选的 `B`），并尽可能复用已经公开过的 residual（如 `epsilon = X - A`、`delta = Y - B`），从而减少：

- triple 生成次数
- residual 的 reveal / open 数量
- 通信字节数
- 一部分本地计算开销

这个改动主要针对**Linear 层训练**里的 matmul 链路，即：

- forward 的 `X @ W^T`
- backward 的 `dX`
- backward 的 `dW`

---

## 3. 这个项目里其实有“两层架构”

初学者最容易混淆的一点是：这里不只有“模型架构”，还有“系统执行架构”。

### 3.1 系统执行架构：CrypTen 怎么跑一个私有训练

从系统角度看，整个训练流程大致是：

```text
PyTorch 模型
   ↓
ct.nn.from_pytorch(...)
   ↓
CrypTen 私有模型（参数和输入都变成加法分享张量）
   ↓
Linear / matmul 等运算最终落到 ArithmeticSharedTensor
   ↓
私有乘法调用 Beaver 协议
   ↓
通过 provider 生成 triple，或通过 reuse cache 复用 mask
   ↓
完成 forward / backward
```

也就是说：

- 你的“模型”还是神经网络
- 但真正执行时，底层乘法不是普通的 `torch.matmul`
- 而是 CrypTen 的 **ArithmeticSharedTensor + Beaver 协议**

### 3.2 任务模型架构：神经网络本身长什么样

本项目里主要用了两类模型：

#### 模型 A：TinyMLP benchmark 模型

文件：
- [bench_reuse_tiny_mlp.py](d:/SJTU/newSHAFT/scripts/bench_reuse_tiny_mlp.py)

核心定义：
- [TinyMLP](d:/SJTU/newSHAFT/scripts/bench_reuse_tiny_mlp.py:48)

结构是：

```text
输入 x
  ↓
Linear(in_features → hidden_features)
  ↓
Activation(ReLU / Tanh / Sigmoid)
  ↓
... 可重复多层 hidden Linear ...
  ↓
Linear(hidden_features → out_features)
  ↓
输出
```

这是一个非常适合做协议 benchmark 的小模型，因为：

- 结构简单
- 主要成本来自 Linear / matmul
- 很容易控制维度、层数、batch size

#### 模型 B：GLUE 文本分类模型 + LoRA

文件：
- [run_glue_private_mpc_lora_train.py](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py)

这个脚本使用 HuggingFace 的文本分类模型：

- [AutoModelForSequenceClassification.from_pretrained(...)](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py:877)

然后给它注入 LoRA：

- [LoRALinear](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py:268)
- [_inject_lora_layers](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py:303)
- [_set_lora_trainable](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py:324)

LoRA 的结构是：

```text
原始 Linear: y = W x

改成：
y = W x + B(A(x)) * scaling

其中：
A: in_features -> r
B: r -> out_features
```

对应代码：
- `self.base`
- `self.lora_A`
- `self.lora_B`
- `self.scaling`

也就是说，在真实任务里，你训练的并不是整套大模型所有参数，而是：

- LoRA 的低秩分支参数
- 以及可选的分类头参数

然后再把这个 PyTorch 模型转换成 CrypTen 私有模型：

- [ct.nn.from_pytorch(...).encrypt().to(device)](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py:1063)

---

## 4. CrypTen 原本的乘法架构是什么样

在 CrypTen 里，私有张量乘法最核心的是：

- `ArithmeticSharedTensor`
- Beaver 协议
- triple provider

### 4.1 ArithmeticSharedTensor 是什么

你可以把它理解成：

> “一个数值张量被拆成多方加法分享后的表示形式”

它不是普通的 `torch.Tensor`，而是一个私有计算张量。  
当你对它做 `mul`、`matmul` 时，底层不会直接普通乘法，而是走 Beaver 协议。

### 4.2 Beaver triple 原始流程

原始流程在：
- [beaver.py](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py)

baseline 路径：
- [_beaver_from_provider](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py:85)

一个私有矩阵乘法 `x op y` 的原始 Beaver 过程可以写成：

1. provider 生成一组三元组 `(a, b, c)`
   - 满足 `c = op(a, b)`
2. 计算 residual：
   - `epsilon = x - a`
   - `delta = y - b`
3. reveal / open：
   - 把 `epsilon` 和 `delta` 打开
4. 用公式恢复结果：

```text
z = c + epsilon * b + a * delta + epsilon * delta
```

对应代码关键位置：

- provider triple：
  - [generate_additive_triple](d:/SJTU/newSHAFT/crypten/mpc/provider/tfp_provider.py:25)
  - [generate_additive_triple](d:/SJTU/newSHAFT/crypten/mpc/provider/ttp_provider.py:31)
- baseline reveal：
  - [reveal_batch([x - a, y - b])](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py:122)
- baseline 合成结果：
  - [beaver.py:126](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py:126)

### 4.3 为什么训练里会有很多 Beaver matmul

因为一个 Linear 层训练时，至少会有 3 次 matmul：

```text
forward:
  Y = X @ W^T

backward:
  dX = dY @ W
  dW = X^T @ dY
```

所以一个 Linear 层，一个训练 step，就有：

- 1 次 forward matmul
- 2 次 backward matmul

如果网络有 `L` 个 Linear 层，那么每个 step 大概就有 `3L` 次 matmul Beaver。

这就是为什么 Linear 层训练非常适合做 Beaver 复用实验。

---

## 5. 本项目到底改了哪里

整体来说，改动只集中在几条关键链路上，没有大面积入侵其它模块。

### 5.1 配置层：增加实验开关

文件：
- [default.yaml](d:/SJTU/newSHAFT/configs/default.yaml:76)

新增配置：

```yaml
experimental_reuse_mask: False
reuse_mode: "FIX_A"
reuse_scope: "STEP"
reuse_op_types: ["matmul"]
reuse_tagging: True
```

这些开关的意思：

- `experimental_reuse_mask`
  - 是否打开复用实验
- `reuse_mode`
  - `FIX_A` 或 `FIX_AB`
- `reuse_scope`
  - 当前只支持 `STEP`，表示只在同一步内复用
- `reuse_op_types`
  - 当前只对 `matmul` 启用
- `reuse_tagging`
  - 是否使用 tag 来识别 forward / backward 对应关系

### 5.2 标签上下文：让 forward 和 backward 能对上号

文件：
- [reuse_context.py](d:/SJTU/newSHAFT/crypten/common/reuse_context.py)

这个文件提供了线程本地上下文，用来跨调用链传递：

- 当前 layer tag
- 当前 beaver tag
- 当前 step id
- 当前 op uid

核心接口：

- [use_beaver_tag](d:/SJTU/newSHAFT/crypten/common/reuse_context.py:36)
- [use_layer_tag](d:/SJTU/newSHAFT/crypten/common/reuse_context.py:61)
- [set_current_reuse_step](d:/SJTU/newSHAFT/crypten/common/reuse_context.py:74)
- [next_beaver_op_uid](d:/SJTU/newSHAFT/crypten/common/reuse_context.py:99)

这相当于给每一次 matmul 建了一个“身份证”。

### 5.3 Linear 层：给 layer 打 tag

文件：
- [module.py](d:/SJTU/newSHAFT/crypten/nn/module.py:2075)

`Linear.forward()` 里现在会：

```text
with use_layer_tag(layer_tag):
    output = x.matmul(self.weight.t())
```

意思是：

- 这次 matmul 属于哪个 Linear 层
- 后续 backward 再算 `dW` / `dX` 时，可以通过 layer tag 找回对应关系

### 5.4 AutogradMatMul：给 forward / backward 三种 matmul 打 tag

文件：
- [gradients.py](d:/SJTU/newSHAFT/crypten/gradients.py:738)

这里是整个复用设计最关键的一层。

forward 时：
- [AutogradMatMul.forward](d:/SJTU/newSHAFT/crypten/gradients.py:753)

它会构造：

```text
base_tag = {
  step_id,
  layer_id,
  op_name="matmul",
  op_uid
}
```

然后生成 `forward` tag 并包住这次 `input.matmul(other)`。

backward 时：
- [AutogradMatMul.backward](d:/SJTU/newSHAFT/crypten/gradients.py:773)

它会生成两组 tag：

- `backward_dX`
- `backward_dW`

其中：

- `dW` 的 tag 会带：
  - `a_anchor=forward.a^T`
  - `epsilon_anchor=forward.epsilon^T`
- `dX` 在 `FIX_AB` 下会额外带：
  - `b_anchor=forward.b^T`
  - `delta_anchor=forward.delta^T`

这就是 forward 和 backward 复用关联的来源。

### 5.5 Beaver 协议：baseline 与 reuse 的总开关

文件：
- [beaver.py](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py)

关键入口：
- [__beaver_protocol](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py:175)

逻辑是：

```text
如果没开 experimental_reuse_mask:
    走原始 provider Beaver
如果开了，且 op 在 reuse_op_types:
    走 _beaver_with_reuse(...)
```

也就是：

- 默认行为完全不变
- 只有实验开关打开时，才走复用逻辑

### 5.6 BeaverReuseCache：复用的真正核心

文件：
- [beaver_reuse.py](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py)

这是整个项目的心脏。

它管理：

- base masks
- derived masks
- `C` cache
- opened residual cache
- shared/plain registry

你可以把它理解成一个“单 step 的 Beaver 复用仓库”。

---

## 6. baseline 和 reuse 的流程到底差在哪

这一节是本项目最重要的概念解释。

### 6.1 baseline 流程

假设有一个 Linear forward：

```text
Y = X @ W^T
```

baseline 中这次 matmul 会：

```text
1. 生成新的 A
2. 生成新的 B
3. 生成新的 C = A @ B
4. 打开 epsilon = X - A
5. 打开 delta   = W^T - B
6. 合成结果
```

backward 的 `dX` 和 `dW` 也会各自再来一遍。

所以一个 Linear 层一整个 step，大概是：

```text
forward:  A,B,C + reveal(epsilon, delta)
backward dX: A,B,C + reveal(epsilon, delta)
backward dW: A,B,C + reveal(epsilon, delta)
```

### 6.2 FIX_A 流程

在 `FIX_A` 下：

- `A` 被固定并在相关 matmul 中复用
- `B` 不固定，仍然 fresh 生成
- backward 的 `dW` 通过 `a_anchor` 和 `epsilon_anchor` 复用 forward 的信息

一个 Linear 层的一步训练，效果相当于：

```text
forward:
  fresh A, fresh B, fresh C, reveal epsilon + delta

backward dX:
  fresh A, fresh B, fresh C, reveal epsilon + delta

backward dW:
  复用 A^T
  fresh B
  fresh C
  复用 epsilon^T
  只需再 reveal 一个新的 delta
```

也就是说：

- triple 生成减少
- `dW` 那条路径少开一个 residual

### 6.3 FIX_AB 流程

在 `FIX_AB` 下：

- `A` 固定并复用
- `B` 也固定并复用
- `dW` 复用 `A` 相关信息
- `dX` 复用 `B` 相关信息

一个 Linear 层一步训练变成：

```text
forward:
  fresh A, fresh B, fresh C, reveal epsilon + delta

backward dX:
  复用 B^T
  复用 delta^T
  只需再 reveal 一个新的 epsilon

backward dW:
  复用 A^T
  复用 epsilon^T
  只需再 reveal 一个新的 delta
```

也就是说：

- relative to baseline，每层少开 2 个 residual tensor
- relative to FIX_A，再多省掉 `dX` 上的一部分开销

---

## 7. 本项目里“复用”具体复用了什么

### 7.1 复用 `A`

来源：
- [get_or_create_A](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py:339)

意思：

- 给同一个 step、同一个层、同一组相关 matmul 分配同一个 `A`
- backward 中需要 `A^T` 时，不重新生成，而是从 forward 的 `A` 派生

### 7.2 复用 `B`

来源：
- [get_or_create_B](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py:343)

`FIX_A`：
- `B` 不复用
- 走 [create_fresh_B](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py:347)

`FIX_AB`：
- `B` 可复用
- backward 的 `dX` 可以拿 forward 的 `B^T`

### 7.3 复用 `C`

来源：
- [get_or_create_C_for_op](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py:367)

作用：

- `C = op(A, B)` 是由 `A` 和 `B` 决定的
- 如果 `(A,B)` 组合未来还会再次出现，理论上就可以缓存

但在当前 TinyMLP workload 下，`C` cache 几乎不 hit，所以后来对这部分做了更保守的策略：

- 只有更可能复用的场景才启用 cache probe
- 否则直接走 fast path fresh create

### 7.4 复用 opened residual

来源：
- [cache_opened_residual](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py:405)
- [get_opened_residual_from_anchor](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py:462)

这部分特别关键。

因为训练里最大的通信点之一，就是把：

- `epsilon = x - a`
- `delta = y - b`

打开。

如果 backward 某个 matmul 本质上需要的 residual，只是 forward residual 的转置，那么它就没必要再开一遍，而是可以直接拿：

```text
forward epsilon  → backward dW 用 epsilon^T
forward delta    → backward dX 用 delta^T
```

这就是为什么 `FIX_A` / `FIX_AB` 可以降低 `reveal_tensors` 和 `bytes`。

---

## 8. 原始模型结构 + 复用插入点，一张图看懂

### 8.1 系统级流程图

```text
PyTorch 模型
   │
   ├─ TinyMLP benchmark
   │    └─ 多层 Linear + 激活
   │
   └─ GLUE 文本分类模型
        └─ Transformer + 分类头 + 可选 LoRA
              │
              ▼
ct.nn.from_pytorch(...)
              │
              ▼
CrypTen 私有模型
              │
              ▼
Linear.forward()
  with use_layer_tag(...)
              │
              ▼
AutogradMatMul.forward()
  生成 forward tag
              │
              ▼
beaver.__beaver_protocol("matmul", ...)
   ├─ baseline: provider triple
   └─ reuse: BeaverReuseCache
              │
              ▼
forward 输出
              │
              ▼
AutogradMatMul.backward()
  生成 backward_dX / backward_dW tag
  附加 anchor 信息
              │
              ▼
beaver.__beaver_protocol("matmul", ...)
   ├─ dX 路径
   └─ dW 路径
```

### 8.2 复用插入点图

```text
forward matmul: X @ W^T
   ├─ 产生 A, B, C
   ├─ 打开 epsilon = X - A
   └─ 打开 delta   = W^T - B

backward dW: X^T @ dY
   ├─ 复用 A^T
   ├─ 复用 epsilon^T
   └─ 新开 delta

backward dX: dY @ W
   ├─ FIX_A: 不复用 B，只新开
   └─ FIX_AB:
        ├─ 复用 B^T
        ├─ 复用 delta^T
        └─ 新开 epsilon
```

---

## 9. 这套项目里有哪些指标，它们各自表示什么

这一节非常重要，因为很多看起来像“cache miss”的数字，实际含义不一样。

### 9.1 时间类指标

#### `prep_time_s`

准备阶段耗时，通常包括：

- 输入加密
- 标签加密
- `optimizer.zero_grad()`

#### `forward_time_s`

前向传播耗时。

对本项目而言，主要看：

- Linear 层的 Beaver matmul 是否变快

#### `backward_time_s`

反向传播耗时。

通常是本项目最值得关注的时间指标，因为：

- backward 里有 `dX`
- backward 里有 `dW`
- 这两条路径都可能复用 forward 的 mask / residual

#### `optim_time_s`

优化器 step 的时间。

一般不是 Beaver 复用的主要瓶颈。

#### `step_time_s`

整个训练 step 总时间。  
这是最直观的最终性能指标。

---

### 9.2 通信类指标

#### `comm_rounds`

通信轮次。

它表示：

> 发生了多少轮交互

注意：

- 本项目当前主要减少的是“每轮开的张量数”和“每轮字节数”
- 并没有显著减少 reveal 调用的轮次数

所以 `comm_rounds` 不一定会下降。

#### `comm_bytes`

总通信字节数。

这是衡量复用是否减少 reveal / open 负担的关键指标之一。  
如果 `reveal_tensors` 下降，通常 `bytes` 也会下降。

#### `comm_time_s`

通信耗时统计。

如果 communicator 开启 verbose，这个值更有参考意义。

---

### 9.3 Beaver 协议统计指标

#### `triple_generate_calls`

本 step 内 provider 被调用去生成 additive triple 的次数。

它反映：

> 有多少乘法还在走原始 provider triple 路径

如果复用生效，这个值会明显下降。

#### `beaver_reveal_calls`

Beaver 协议里调用 reveal / open 的次数。

它反映：

> 一共触发了多少次 open 流程

注意，它和 `reveal_tensors` 不一样。  
一次 reveal call 里可以打开 1 个或多个 tensor。

#### `beaver_revealed_tensors`

Beaver 协议里一共打开了多少个 tensor。

它是本项目里最关键的协议级指标之一，因为：

- baseline 每个 matmul 通常开两个：`epsilon` 和 `delta`
- `FIX_A` / `FIX_AB` 的核心收益，就是把一部分 `epsilon/delta` 复用掉

---

### 9.4 Mask 复用统计指标

这部分是后来专门清洗过语义的。

#### `a_base_cache_hit / a_base_cache_miss`

表示 `A` 的 base mask 是否直接命中。

含义：

- `base_hit`：当前 tagged op 直接拿到了已有 base `A`
- `base_miss`：当前 tagged op 需要新建一个 base `A`

#### `a_derived_cache_hit / a_derived_generated`

表示 `A` 的 derived mask（例如转置后的 `A^T`）是否命中或新生成。

含义：

- `derived_hit`：之前已经派生过这个 `A^T`
- `derived_generated`：第一次为 anchor 路径生成这个派生 mask

`B` 也完全类似：

- `b_base_cache_hit / b_base_cache_miss`
- `b_derived_cache_hit / b_derived_generated`
- `b_fresh_generated`

其中 `b_fresh_generated` 很直观：

> 这次完全 fresh 生成了一个不复用的 `B`

在 `FIX_A` 里这个值通常会比较高；在 `FIX_AB` 里应该更低。

---

### 9.5 C 相关统计指标

#### `c_cache_probe_hit / c_cache_probe_miss`

只有在真的对 `C cache` 做 lookup 时，才会计数。

含义：

- `probe_hit`：命中了已有 `C`
- `probe_miss`：查了 cache，但没命中

#### `c_cache_bypassed`

表示这次**压根没去 probe C cache**。

也就是说：

> 代码判断这次缓存大概率没意义，直接绕过 cache 逻辑

#### `c_fresh_generated`

表示本地 fresh 生成了一个新的 `C = op(A, B)`。

这是当前理解 `C` 成本最重要的统计项。

---

### 9.6 Residual 复用统计指标

#### `residual_anchor_hit / residual_anchor_miss`

表示通过 anchor 去找 forward residual 时是否命中。

例如：

- `dW` 想找 forward 的 `epsilon^T`
- `dX` 想找 forward 的 `delta^T`

如果命中，说明：

> backward 确实复用了 forward 已经打开过的 residual

这通常是最能直接说明 forward / backward 联动复用是否成功的计数。

---

## 10. 哪些指标表明性能提升

严格来说，本项目的“提升”分成两层：

### 10.1 协议层面的提升

如果你看到下面这些指标下降，就说明复用逻辑真的生效了：

- `triple_generate_calls` 下降
- `beaver_revealed_tensors` 下降
- `comm_bytes` 下降

这些指标说明：

- 需要 provider 生成的新 triple 少了
- 需要打开的新 residual 少了
- 通信数据量少了

### 10.2 端到端训练层面的提升

如果你看到下面这些指标下降，就说明这些协议节省开始变成真实加速了：

- `forward_time_s`
- `backward_time_s`
- `step_time_s`

其中最重要的是：

- `backward_time_s`
- `step_time_s`

因为本项目的收益通常主要出现在 backward。

---

## 11. 如何用理论来理解这些数字

假设一个网络有 `L` 个 Linear 层，每层每 step 有 3 个 matmul：

- 1 个 forward
- 1 个 backward `dX`
- 1 个 backward `dW`

若非 matmul 的 Beaver 操作数记为 `M`，则 baseline 理论：

```text
triple = 3L + M
reveal_tensors = 6L + 2M
```

### FIX_A 理论

`FIX_A` 中：

- matmul triple 不再走 provider
- 每层少开 `dW` 的一个 residual

所以：

```text
triple = M
reveal_tensors = 5L + 2M
```

### FIX_AB 理论

`FIX_AB` 中：

- 每层进一步少开 `dX` 的一个 residual

所以：

```text
triple = M
reveal_tensors = 4L + 2M
```

你可以把这两个公式理解成：

- `triple` 主要看“有多少 matmul 被复用路径接管了”
- `reveal_tensors` 主要看“每层少开了几个 residual”

---

## 12. 当前实验结果说明了什么

下面用最近一轮 benchmark 的结果来解释。

### 12.1 最新结果摘要

| Case | Baseline Step | FIX_A Step | FIX_AB Step | FIX_A 提升 | FIX_AB 提升 |
|---|---:|---:|---:|---:|---:|
| tiny | 0.094834 | 0.082770 | 0.083676 | 12.72% | 11.77% |
| base | 0.165892 | 0.160363 | 0.165467 | 3.33% | 0.26% |
| wide | 0.508927 | 0.499583 | 0.488790 | 1.84% | 3.96% |

协议指标：

| Case | Baseline reveal | FIX_A reveal | FIX_AB reveal | Baseline triple | FIX_A triple | FIX_AB triple |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 18 | 16 | 14 | 9 | 3 | 3 |
| base | 28 | 25 | 22 | 14 | 5 | 5 |
| wide | 28 | 25 | 22 | 14 | 5 | 5 |

### 12.2 这组结果应该怎么读

#### 结论 1：协议层逻辑是对的

因为：

- `triple` 精确降到了理论值
- `reveal_tensors` 精确降到了理论值
- `FIX_A` 与 `FIX_AB` 的统计分叉清晰

说明：

- `FIX_A` 现在确实只复用 `A`
- `FIX_AB` 额外复用了 `B`

#### 结论 2：性能收益取决于模型规模

`tiny`：

- 协议固定开销占比较高
- 所以复用带来的收益比较明显

`base`：

- `FIX_A` 还能有收益
- `FIX_AB` 在当前形状上和额外本地管理开销接近打平

`wide`：

- `FIX_AB` 收益重新变明显
- 因为更宽的 matmul 让省下来的 reveal / bytes 更容易体现为 backward 加速

#### 结论 3：当前收益主要体现在 backward

这是很符合预期的。

因为：

- backward 有两次 matmul
- forward 只有一次
- residual anchor 复用主要也是为了服务 backward

---

## 13. 这个项目迭代过程中，踩过哪些坑

为了让初学者理解工程上为什么会反复迭代，这里把关键问题单独总结。

### 坑 1：`FIX_A` 一开始并不是真的 `A-only`

早期实现里，`FIX_A` 也在 `dX` 路径上复用了 `B/delta`，导致：

- `FIX_A` 和 `FIX_AB` 的统计几乎一样

后来通过修改：

- [gradients.py](d:/SJTU/newSHAFT/crypten/gradients.py:811)
- [beaver.py](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py:135)

把语义分开后，结果才正确。

### 坑 2：GPU 计时不做 synchronize 会误导结论

早期 benchmark 用的是异步 CUDA 时间，导致：

- 看起来某些模式“变慢”或“变快”
- 但其实只是 kernel 尚未同步完成

后来在：

- [bench_reuse_tiny_mlp.py](d:/SJTU/newSHAFT/scripts/bench_reuse_tiny_mlp.py:107)
- [run_glue_private_mpc_lora_train.py](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py:216)

加入同步后，结果才可信。

### 坑 3：transpose-derived view 进入 matmul 会拖慢 GPU

为了避免重分享，曾把 transpose 改成直接返回 view。  
结果这些非连续 tensor 进入 `torch.matmul` 后，带来了明显退化。

后来把：

- plain transform
- shared transform
- public residual transform

拆开，并恢复 contiguous，问题才修住。

### 坑 4：计数器语义一开始太粗

比如：

- `a_cache_hit`
- `c_cache_miss`

看起来像“缓存命中 / 失败”，但实际上混杂了：

- base reuse
- derived reuse
- bypass
- fresh create

后来把统计拆细后，实验结果才更容易解释。

---

## 14. 目前项目的代码地图

如果你是第一次接触项目，建议按这个顺序读代码：

### 第一层：先看总入口

- [beaver.py](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver.py)

重点看：

- baseline 怎么走 provider
- experimental 怎么走 reuse cache

### 第二层：再看缓存实现

- [beaver_reuse.py](d:/SJTU/newSHAFT/crypten/mpc/primitives/beaver_reuse.py)

重点看：

- `_get_or_create_mask(...)`
- `get_or_create_C_for_op(...)`
- `get_opened_residual_from_anchor(...)`
- 各类 counters

### 第三层：看 forward/backward 怎么把 tag 串起来

- [gradients.py](d:/SJTU/newSHAFT/crypten/gradients.py:738)

重点看：

- forward tag
- `backward_dX`
- `backward_dW`
- anchor 字段

### 第四层：看 Linear 怎么注入 layer tag

- [module.py](d:/SJTU/newSHAFT/crypten/nn/module.py:2075)

### 第五层：看 benchmark 和训练脚本怎么测

- [bench_reuse_tiny_mlp.py](d:/SJTU/newSHAFT/scripts/bench_reuse_tiny_mlp.py)
- [run_glue_private_mpc_lora_train.py](d:/SJTU/newSHAFT/examples/text-classification/run_glue_private_mpc_lora_train.py)

---

## 15. 现在应该怎么运行和观察

### 15.1 跑 benchmark

例如长时间 benchmark：

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/bench_reuse_tiny_mlp.py \
  --world-size 2 \
  --provider TFP \
  --device cuda \
  --preset-cases base,wide,tall \
  --steps 200 \
  --warmup 20 \
  --repeats 8 \
  --run-fix-ab \
  --verbose-comm \
  --save-json bench_reuse_tiny_mlp_overnight.json \
  --save-csv bench_reuse_tiny_mlp_overnight.csv
```

### 15.2 跑真实训练脚本

```bash
CUDA_VISIBLE_DEVICES=2,3 python examples/text-classification/run_glue_private_mpc_lora_train.py \
  --task_name sst2 \
  --model_name_or_path distilbert-base-uncased \
  --quick_run \
  --skip_private_eval \
  --skip_plain_eval \
  --reuse_profile \
  --reuse_log_every_steps 1 \
  --experimental_reuse_mask \
  --reuse_mode FIX_AB \
  --gpu_ids 0,1
```

### 15.3 跑完优先看什么

先看大指标：

- `step_time_s`
- `backward_time_s`
- `comm_bytes`
- `beaver_revealed_tensors`
- `triple_generate_calls`

再看细指标：

- `a_base_hit/miss`
- `a_der_hit/new`
- `b_base_hit/miss`
- `b_der_hit/new`
- `b_fresh`
- `c_hit/miss`
- `c_bypass/fresh`
- `anchor_hit/miss`

建议的观察顺序是：

1. 协议是否生效
2. 通信是否下降
3. backward 是否变快
4. step 是否最终变快

---

## 16. 到目前为止，这个项目的价值是什么

如果站在工程和研究两个角度，这个项目的价值可以总结为：

### 工程价值

- 在 CrypTen 中实现了一个低侵入式的实验复用框架
- 复用逻辑集中在 `beaver.py + beaver_reuse.py`
- 默认关闭，不破坏原有行为
- 已经接通 benchmark 和真实训练脚本

### 研究价值

- 验证了 Linear 训练里 `forward + backward` 的 Beaver 掩码复用是可实现的
- 可以精确量化：
  - triple 降了多少
  - reveal 降了多少
  - bytes 降了多少
  - step time 最终降了多少
- 为后续更复杂的协议级优化打下了基础

---

## 17. 一段最简洁的总结

如果要用一句话概括整个项目，可以这么说：

> 这个项目在 CrypTen 的 Beaver matmul 协议中，加入了基于 tag 和 step-scope cache 的掩码复用机制，使同一个 Linear 层在一个训练 step 内的 forward、`dX`、`dW` 可以共享部分 Beaver mask 和已打开 residual，从而显著降低 triple 生成次数、reveal 张量数量和通信字节数，并在多个模型规模上观察到端到端训练时间下降。

