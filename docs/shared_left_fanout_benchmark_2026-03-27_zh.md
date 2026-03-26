# Shared-Left Fan-Out Benchmark 结果解读

## 1. 这份文档记录了什么

这份文档记录的是一次 `shared_left` 专项 benchmark 的结果。这个 benchmark 的目标不是测通用 `FIX_A` / `FIX_AB`，而是专门测一种新的复用思路：

- 多个 `matmul` 共享同一个左操作数
- 复用左侧 Beaver 掩码 `A`
- 复用左侧公开残差 `epsilon = X - A`

最典型的结构是：

```text
Y1 = X @ W1
Y2 = X @ W2
Y3 = X @ W3
```

也就是 `A @ B`、`A @ C`、`A @ D` 这种同左输入 fan-out 结构。


## 2. 本次运行命令

```bash
CUDA_VISIBLE_DEVICES=2,3 python scripts/bench_reuse_shared_left_fanout.py \
  --world-size 2 \
  --provider TFP \
  --device cuda \
  --gpu-ids 0,1 \
  --preset-cases tiny,base,wide \
  --steps 100 \
  --warmup 20 \
  --repeats 5 \
  --verbose-comm \
  --save-json bench_reuse_shared_left_fanout.json \
  --save-csv bench_reuse_shared_left_fanout.csv
```

说明：

- `world_size=2`：两方计算
- `provider=TFP`：使用 TFP triple provider
- `CUDA_VISIBLE_DEVICES=2,3`：物理使用 2、3 号卡
- `--gpu-ids 0,1`：在当前可见卡集合里，rank 0 用 `cuda:0`，rank 1 用 `cuda:1`
- 输出里显示的 `device=cuda:0` 是 rank 0 的设备，不代表只用了一张卡


## 3. benchmark 模型结构

本 benchmark 用的是一个“共享 stem + 多个并行 head”的小网络：

```text
input
  -> stem Linear
  -> activation
  -> head1 Linear
  -> head2 Linear
  -> head3 / head4 Linear
```

其中多个 head 的输入是同一个隐藏表示 `H`，所以天然满足 shared-left 条件：

```text
head1 = H @ W1
head2 = H @ W2
head3 = H @ W3
...
```

训练时 backward 的权重梯度也有同样结构：

```text
dW1 = H^T @ dY1
dW2 = H^T @ dY2
dW3 = H^T @ dY3
...
```

因此，shared-left 复用会命中两类位置：

1. 多个 head 的 forward
2. 多个 head 的 `dW` backward

不会命中的是：

1. stem 自己的 matmul
2. 各个 head 的 `dX`，因为左操作数变成了不同的 `grad_output_i`


## 4. 输出指标中文解释

### 4.1 主表指标

| 英文列名 | 中文名 | 含义 |
|---|---|---|
| `mode` | 模式 | `baseline` 表示原始路径；`shared_left` 表示启用共享左操作数复用 |
| `step(s)` | 单步总耗时（秒） | 一个训练 step 的平均总时间，包含加密输入、forward、loss、backward、optimizer step |
| `rounds` | 通信轮次 | 一个 step 内通信交互的总轮数 |
| `bytes` | 通信字节数 | 一个 step 内总通信字节数 |
| `reveal_tensors` | 打开的残差张量数 | 一个 step 内通过 reveal/open 打开的 residual tensor 总数 |
| `triple_gen` | triple 生成次数 | 一个 step 内 Beaver triple provider 被调用的次数 |
| `a_base_hit/miss` | 左侧基础掩码 A 的缓存命中/未命中次数 | 命中说明多个 matmul 共享了同一组左掩码 |
| `residual_hit/miss` | residual 缓存命中/未命中次数 | 这里主要关注 `epsilon` 的复用是否命中 |
| `speedup` | 相对加速比 | `baseline_step_time / 当前_step_time`，大于 1 表示加速 |

### 4.2 Delta 行指标

| 英文列名 | 中文名 | 含义 |
|---|---|---|
| `step` | step 时间变化百分比 | 相对 `baseline` 的 step 总时间变化，正值表示更快 |
| `triple` | triple 降幅 | 相对 `baseline`，triple 生成次数减少了多少 |
| `reveal_tensors` | residual 打开数量降幅 | 相对 `baseline`，被 reveal 的 residual tensor 数量减少了多少 |


## 5. 本次原始结果

### 5.1 tiny

| 模式 | 单步总耗时(s) | 通信轮次 | 通信字节数 | 打开的残差张量数 | triple 生成次数 | A 掩码命中/未命中 | residual 命中/未命中 | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.157943 | 28.00 | 1695744 | 34.00 | 17.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.000x |
| shared_left | 0.144683 | 28.00 | 1564672 | 30.00 | 5.00 | 4.00 / 8.00 | 4.00 / 20.00 | 1.092x |

相对 baseline：

- step：`+8.39%`
- triple：`-70.59%`
- reveal_tensors：`-11.76%`

### 5.2 base

| 模式 | 单步总耗时(s) | 通信轮次 | 通信字节数 | 打开的残差张量数 | triple 生成次数 | A 掩码命中/未命中 | residual 命中/未命中 | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.154159 | 28.00 | 15400960 | 34.00 | 17.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.000x |
| shared_left | 0.155997 | 28.00 | 14352384 | 30.00 | 5.00 | 4.00 / 8.00 | 4.00 / 20.00 | 0.988x |

相对 baseline：

- step：`-1.19%`
- triple：`-70.59%`
- reveal_tensors：`-11.76%`

### 5.3 wide

| 模式 | 单步总耗时(s) | 通信轮次 | 通信字节数 | 打开的残差张量数 | triple 生成次数 | A 掩码命中/未命中 | residual 命中/未命中 | 加速比 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.333537 | 33.00 | 126877696 | 42.00 | 21.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.000x |
| shared_left | 0.350661 | 33.00 | 114294784 | 36.00 | 6.00 | 6.00 / 9.00 | 6.00 / 24.00 | 0.951x |

相对 baseline：

- step：`-5.13%`
- triple：`-71.43%`
- reveal_tensors：`-14.29%`


## 6. 理论上应该发生什么

### 6.1 为什么 `triple_gen` 会大幅下降

这个 benchmark 的对比是：

- `baseline`：原始 Beaver provider 路径
- `shared_left`：开启实验复用路径，并同时启用 shared-left grouping

因此 `triple_gen` 的大幅下降，主要说明：

- 所有 `matmul` 已经不再走 provider 的 `generate_additive_triple`

它反映的是“进入了实验 matmul 复用路径”，不完全等价于“shared-left grouping 本身单独带来的收益”。

换句话说：

- `triple_gen` 是“实验路径生效”的证据
- `a_base_hit` 和 `residual_hit` 才是“共享左操作数复用真的命中”的证据

### 6.2 为什么 `reveal_tensors` 只下降一部分

shared-left 只会减少“同左输入”的那部分 `epsilon`，不会减少：

- 每个分支自己的 `delta`
- 不相关 matmul 的 reveal
- 通信轮次本身

对有 `k` 个 head 的 fan-out 结构：

1. head forward
   - baseline：`2k`
   - shared-left：`k + 1`
2. head `dW`
   - baseline：`2k`
   - shared-left：`k + 1`

总节省：

```text
2(k - 1)
```

#### 当 `k = 3`

节省：

```text
2(3 - 1) = 4
```

所以：

- `34 -> 30`

#### 当 `k = 4`

节省：

```text
2(4 - 1) = 6
```

所以：

- `42 -> 36`

这和实测完全一致。


## 7. 如何从结果判断 shared-left 真的生效了

### 7.1 `a_base_hit/miss`

这是最直观的 shared-left 命中指标。

#### tiny / base：`4 hit / 8 miss`

这说明每步共有 12 次 matmul 的左掩码请求，其中：

- stem：3 次 miss
- head `dX`：3 次 miss
- head forward：1 次 miss + 2 次 hit
- head `dW`：1 次 miss + 2 次 hit

总计：

- miss = `3 + 3 + 1 + 1 = 8`
- hit = `2 + 2 = 4`

完全对上。

#### wide：`6 hit / 9 miss`

这里有 4 个 head，因此：

- stem：3 次 miss
- head `dX`：4 次 miss
- head forward：1 次 miss + 3 次 hit
- head `dW`：1 次 miss + 3 次 hit

总计：

- miss = `3 + 4 + 1 + 1 = 9`
- hit = `3 + 3 = 6`

也完全对上。

结论：

- `a_base_hit` 的数量与 fan-out 理论值精确匹配
- 说明左侧 Beaver 掩码 `A` 的共享已经按设计发生

### 7.2 `residual_hit/miss`

这个指标主要说明 `epsilon = X - A` 的复用是否发生。

#### tiny / base：`4 hit / 20 miss`

每次 matmul 都要尝试查两个 residual：

- `epsilon`
- `delta`

总共 12 次 matmul：

```text
12 * 2 = 24 次查询
```

其中：

- 4 次命中的是共享的 `epsilon`
- 剩余 20 次未命中

这和输出一致。

#### wide：`6 hit / 24 miss`

总共 15 次 matmul：

```text
15 * 2 = 30 次查询
```

其中：

- 6 次命中共享 `epsilon`
- 24 次未命中

也和输出一致。

结论：

- `residual_hit` 的数值和理论值精确一致
- 说明 shared-left 不只是复用了 `A`，也确实复用了已经打开过的 `epsilon`


## 8. 为什么 `rounds` 没变

这是一个很重要的现象。

三组里：

- `rounds` 都没有下降

这说明当前实现减少的是：

- 每轮里打开的 residual tensor 数量
- 每轮里传输的字节数

但没有减少：

- reveal 调用次数
- 通信交互轮次

所以 shared-left 目前更准确地说是：

> 减少每轮负载，而不是减少轮数

这也是为什么协议统计明显变好，但 step 时间不一定稳定变快。


## 9. 为什么 `bytes` 下降了，但 step 时间不一定更快

这是这组结果最需要解释的地方。

### 9.1 tiny：协议节省成功转化成了时间收益

tiny 上：

- 通信字节数下降
- residual tensor 数下降
- triple 生成次数下降
- step 时间提升 `8.39%`

这说明在小规模 case 下，协议侧的节省足以覆盖 shared-left 额外引入的本地 bookkeeping 成本。

### 9.2 base：协议节省存在，但端到端几乎打平

base 上：

- `bytes` 从 `15400960` 降到 `14352384`
- `reveal_tensors` 从 `34` 降到 `30`
- `triple_gen` 从 `17` 降到 `5`

但：

- `step` 反而 `-1.19%`

这说明在这个形状上：

- 共享左掩码和 epsilon 的收益
- 与本地缓存查找、tag 处理、transform 和额外张量管理的成本

大致持平，甚至后者略高。

### 9.3 wide：协议节省更大，但 wall-clock 仍为负收益

wide 上：

- `bytes` 大幅下降
- `reveal_tensors` 从 `42` 降到 `36`
- `triple_gen` 从 `21` 降到 `6`

但：

- `step` 下降 `5.13%`

这说明在大张量 GPU 场景里，当前 shared-left 实现虽然减少了通信量，但：

- 轮次没减
- TFP triple 生成本身是本地行为，不一定是主瓶颈
- 共享路径的缓存管理和张量处理开销，仍然会影响 wall-clock

换句话说：

- 协议层面：成功
- 端到端时间：还没有在所有形状上都转化成正收益


## 10. 本次结果可以得出的结论

### 10.1 可以确认成功的部分

1. `SHARED_LEFT` 路径已真正生效  
   证据：`triple_gen` 大幅下降

2. 同左操作数的左掩码复用已真正命中  
   证据：`a_base_hit` 与理论完全一致

3. 同左操作数的 `epsilon` 复用已真正命中  
   证据：`residual_hit` 与理论完全一致

4. 协议层节省与理论严格对齐  
   证据：
   - `reveal_tensors`
   - `bytes`
   - `a_base_hit`
   - `residual_hit`

### 10.2 目前仍未解决的部分

1. `rounds` 没下降  
   说明当前实现还不是 fused primitive，只是减少了单轮负载

2. wall-clock 收益不稳定  
   - tiny：正收益明显
   - base：接近打平
   - wide：协议更省，但时间更慢

3. 当前 benchmark 反映的是“shared-left + 实验 matmul 复用路径”的整体收益  
   不是“只把 shared-left 单独拎出来”的完全隔离测量


## 11. 这组结果对后续 BERT 密态训练意味着什么

这次 benchmark 给出的最重要信息不是“所有场景都一定更快”，而是：

1. shared-left 这条思路是可实现的
2. 它在协议层面严格成立
3. 它非常适合 Q/K/V 这类同左输入 fan-out 结构
4. 它能稳定减少：
   - `A` 掩码重复生成
   - `epsilon` 重复 open
   - residual tensor 数
   - 通信字节数

因此，把它接到真实 BERT 密态训练里的意义在于：

- 先在 attention / 并行投影处稳定吃到协议节省
- 再观察这些节省能否在真实模型规模上转成 wall-clock 收益

如果后面在 BERT 上看到了：

- `shared_left_group_summary.num_groups > 0`
- `a_base_cache_hit > 0`
- `residual_cache_hit > 0`
- `beaver_revealed_tensors` 和 `comm_bytes` 下降

那就说明这条创新已经真正进入了真实模型路径。


## 12. 一句话总结

这组 benchmark 的结论是：

> 你的 shared-left 思路在协议层面已经被严格验证成功：左侧掩码 `A` 和左残差 `epsilon` 的复用都准确命中，`triple_gen`、`reveal_tensors`、`bytes` 均显著下降；但由于当前实现尚未减少通信轮次，且存在本地管理开销，端到端 step 时间收益依赖具体张量规模，在 tiny 上收益明显，在 base 和 wide 上还需要继续压实现开销。
