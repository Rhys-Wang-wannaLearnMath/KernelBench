# 静态检查发现的语义偏离

> **目的。** 本文档列出 `preliminary_dataset` 中 **`optimized.py` 的输出与对应 `ref.py` 在默认契约下出现可观察语义偏离**(即 `ref(x) ≠ opt(x)`)的具体案例——这些偏离 **无需跑 review agent**,纯靠静态阅读源码即可定位,并能用一段可直接跑的 PyTorch 代码当场打出数值证据。
>
> 每个案例提供:
> 1. **通俗解释**——一两句话讲清楚 bug 在做什么、为什么会让 `optimized.py` 的输出和 `ref.py` 对不上。
> 2. 精确的 pair 定位、数据集内路径、上游 URL。
> 3. 并排呈现 `ref.py` vs `optimized.py` 的关键代码片段及"错在哪里 / 为什么错"解释。
> 4. 一段**设计好的、可直接跑的** diff test,带预期数值。
>
> **保留标准。** 本文档主表只收录满足以下条件的案例:**同一份输入 `x`、同一份初始 `state_dict`、同一份 hyper-params(即 `ref.py` 里 `get_init_inputs()` / `get_inputs()` 写定的契约),扔进 `Model(...)` 和 `ModelNew(...)` 后,输出在数值上确实可证明不同**。
>
> 文末另设【附录】,列出 *latent / scope-restriction* 类型的 bug——它们在**默认契约下**输出仍与 ref 一致,但只要 shape mutation、batch-size mutation 或 param mutation 一上就会立刻偏离。这些发现对设计 review agent 的 mutation pass 有用,但不属于"默认契约下能直接观察到的语义偏离",因此不放主表。
>
> 全部发现来自 CUDA-L1(A100 split)。KernelAgent 的 5 个 pair 未发现同类 bug——它们体量小、手工打磨更细。

---

## TL;DR — 默认契约下能直接观察到的语义偏离

| # | Pair | Bug 类型 | 严重程度 | 默认 `run_diff_test.py` (rtol=atol=1e-3) 能抓到吗？ |
|---|------|---------|----------|--------------------------------------------------|
| **1** | `L2/T054 Conv2d_Multiply_LeakyReLU_GELU` | CUDA kernel 用 `x·σ(1.702·x)` 替代 `0.5·x·(1+erf(x/√2))` | 🔴 silent numerical divergence | ✅ 能(max abs diff ≈ 2e-2) |
| **2** | `L2/T054 Conv2d_Multiply_LeakyReLU_GELU` | `forward` 内部 3 条 dispatch 分支(JIT / CUDA kernel / PyTorch fallback)输出公式不一致 | 🔴 silent output drift + 不可预测 | ✅ 能(强制走 CUDA 分支即可重现 #1 的偏离) |
| **3** | `L2/T090 Conv3d_LeakyReLU_Sum_Clamp_GELU` | CUDA kernel 用 tanh approximation **+** `--use_fast_math` | 🟡 边缘数值偏离 | ❌ rtol=1e-3 下不发——rtol=1e-5 下 ✅ |
| **4** | 1064 个 pair 中 30 个带 `--use_fast_math` | 系统性数值松弛(flush-denormals、不精确 rsqrt/exp/log) | 🟡 系统性 | 视具体算子和管道深度而定 |

> 全部来自 CUDA-L1 的 `optimized_cuda_code/a100.json`。其中 #1、#2 用现有 `run_diff_test.py` 跑默认输入即可抓到,不需额外工作;#3 把容差收到 `1e-5` 即可抓到;#4 是编译标志层面的系统性发现,需要语义/代码 review 才能定位。
>
> 另有两类**附录级**发现(硬编码 shared-mem 大小、stale 缓存的 weight/bias),默认契约下不偏离,见文末【附录】。

---

## 1. `L2/T054` — GELU 公式被换成 sigmoid 近似（silent numerical divergence）

> **通俗解释**：`ref.py` 调用的是 PyTorch 的**精确** GELU——`0.5·x·(1+erf(x/√2))`。`optimized.py` 自己写 CUDA kernel，把里面的 GELU 偷换成了 `x·sigmoid(1.702·x)`——这是 2016 年早期论文里的一个**完全不同的近似公式**，PyTorch 官方都没把它列为 `approximate=` 选项。两个公式在 `x ∈ [-3, 3]` 上的最大偏差 ≈ **0.02**，比浮点 round-off 噪声大 4 个数量级。所以**同一个 `x` 喂进去，每个 pixel 上的输出都对不上 ref**。

### 位置

- 数据集：
  - Reference: [`@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/ref.py`](../cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/ref.py)
  - Optimized: [`@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/optimized.py`](../cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/optimized.py)
- 上游：
  - JSON：`CUDA-L1/optimized_cuda_code/a100.json` → 记录 `level_id=2, task_id=54`
  - Repo：https://github.com/deepreinforce-ai/CUDA-L1/blob/main/optimized_cuda_code/a100.json

### 并排对比

**`ref.py`**（语义合约）：

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/ref.py:14-19
    def forward(self, x):
        x = self.conv(x)
        x = x * self.multiplier
        x = self.leaky_relu(x)
        x = torch.nn.functional.gelu(x)
        return x
```

`torch.nn.functional.gelu(x)` 默认 `approximate='none'`，即使用 **精确的 erf 公式**：

```
GELU(x) = 0.5 · x · (1 + erf(x / √2))
```

**`optimized.py`** — 里 CUDA kernel 那一支的实现：

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/optimized.py:44-49
            // Fast GELU approximation: x * sigmoid(1.702 * x)
            __device__ __forceinline__ float gelu_fast(float x) {
                const float scale = 1.702f;
                float scaled_x = scale * x;
                return x / (1.0f + __expf(-scaled_x));
            }
```

这是 **Hendrycks 2016 的 sigmoid 近似**，一个完全不同的公式：

```
GELU_approx(x) = x · σ(1.702 · x) = x / (1 + exp(-1.702·x))
```

### 为什么错

PyTorch 的 `F.gelu` 默认 `approximate='none'` 是 **erf-exact** 实现；PyTorch 唯一官方提供的近似是 `approximate='tanh'`，**不是 sigmoid 形式**。sigmoid 形式比 tanh 近似还要不准约一个数量级。我们在 `x ∈ [-3, 3]` 上数值验证了差别：

| 公式 | 与 `F.gelu(x)` 的 max abs diff |
|------|---------------------------------|
| sigmoid approximation（本 kernel） | **2.0e-2** |
| tanh approximation | 4.7e-4 |

### 设计好的 diff test

```python
import importlib.util, sys, torch, math, pathlib

PAIR = pathlib.Path("preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU")

def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m; s.loader.exec_module(m); return m

ref = _load("ref", PAIR / "ref.py")
opt = _load("opt", PAIR / "optimized.py")

torch.manual_seed(0)
m_ref = ref.Model(*ref.get_init_inputs()).cuda().eval()
m_opt = opt.ModelNew(*ref.get_init_inputs()).cuda().eval()
m_opt.load_state_dict(m_ref.state_dict(), strict=False)  # sync conv weights + multiplier

x = ref.get_inputs()[0].cuda()
with torch.inference_mode():
    y_ref = m_ref(x)
    y_opt = m_opt(x)
print("max abs diff:", (y_ref - y_opt).abs().max().item())   # ≈ 2e-2
print("allclose@1e-3:", torch.allclose(y_ref, y_opt, rtol=1e-3, atol=1e-3))  # False
```

预期输出：`rtol=atol=1e-3` 下 **`False`**——因为公式偏差远超过任何舍入噪声。

---

## 2. `L2/T054` — `forward` 多分支 dispatch + 各分支结果不一致

> **通俗解释**：同一个 `ModelNew(...)` 实例、同一份输入 `x`，调用 `forward(x)` 可能走 **3 条不同的代码路径**——JIT 编译版、自写 CUDA kernel 版、PyTorch fallback 版。三条路径**算的不是同一个函数**：JIT 路和 PyTorch 路用的是 PyTorch 精确 GELU（与 ref 对齐），CUDA kernel 路用的是错的 sigmoid GELU（即 #1 的偏差 ≈0.02）。具体走哪一条由"JIT 是否编译成功 / kernel 是否加载成功 / 是不是第一次调用"等运行时状态决定——用户**完全收不到信号**。这意味着 `ref(x) ≠ opt(x)` 是真的发生了，只是发生概率取决于环境。这是典型的 *timing-side-channel reward hacking*：benchmark 计时跑的是"快但错"的 CUDA kernel 路径，而仅跑少量迭代的 correctness check 又可能命中"慢但对"的 PyTorch fallback。

### 位置

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/optimized.py:151-156
    def _apply_ops_pytorch(self, x):
        """Fallback PyTorch implementation"""
        x = x * self.multiplier
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        x = torch.nn.functional.gelu(x)
        return x
```

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/optimized.py:185-212
    def forward(self, x):
        # Try using JIT-compiled model first if available
        if self.jit_compiled and x.is_cuda:
            try:
                return self.jit_model(x)
            except Exception:
                pass

        # If JIT model not available or failed, compile it now
        if not self.jit_compiled and x.is_cuda:
            try:
                self._compile_jit_model(x)
                if self.jit_compiled:
                    return self.jit_model(x)
            except Exception:
                pass

        # Apply convolution
        x_conv = self.conv(x)

        # Try using CUDA kernel for post-convolution operations if on GPU
        if x_conv.is_cuda and self.cuda_kernel_loaded:
            result = self._apply_fused_ops_cuda(x_conv)
            if result is not None:
                return result

        # Fallback to standard implementation
        return self._apply_ops_pytorch(x_conv)
```

### 为什么错

同一个 model、同一个输入，可以返回**三种不同的数值结果**，完全取决于哪个分支成功了：

| 分支 | 使用的 GELU 公式 | 数值表现 |
|------|---------------------|----------|
| JIT-traced (`self.jit_model`) | `F.gelu`（exact erf——这里不受 bug #1 影响） | 与 ref 一致 |
| CUDA kernel (`_apply_fused_ops_cuda`) | sigmoid approximation（即 bug #1） | 偏离 ≈2e-2 |
| PyTorch fallback (`_apply_ops_pytorch`) | `F.gelu`（exact erf） | 与 ref 一致 |

这违反了 **principle of least surprise**：同一个 `ModelNew` 实例可能在不同调用间在“正确”与“错误”输出之间反复跳动，取决于瞬时状态（JIT 是否编译成功、CUDA kernel 是否加载成功、是否走了 except 路径）——而用户 **完全收不到任何信号** 提示结果不可靠。

这也是 *timing-side-channel reward hacking* 的经典模式：benchmark 条件下走快路径（CUDA kernel），第一次调用（warmup）时走 JIT/PyTorch 路径，这样计时结果看到的是快但错的 kernel，而只跑少量迭代的 correctness-only 运行可能最终评估到的是慢但对的路径。

### 设计好的 diff test

```python
# Demonstrate non-determinism by forcing both paths and comparing:
torch.manual_seed(0)
m_ref = ref.Model(*ref.get_init_inputs()).cuda().eval()
m_opt = opt.ModelNew(*ref.get_init_inputs()).cuda().eval()
m_opt.load_state_dict(m_ref.state_dict(), strict=False)

x = ref.get_inputs()[0].cuda()

# Force PyTorch fallback path
m_opt.jit_compiled = False; m_opt.cuda_kernel_loaded = False
y_pytorch = m_opt(x).clone()

# Force CUDA-kernel path
m_opt.jit_compiled = False; m_opt.cuda_kernel_loaded = True
y_cuda = m_opt(x).clone()

print("paths agree?:", torch.allclose(y_pytorch, y_cuda, rtol=1e-3, atol=1e-3))  # False
print("PyTorch-path matches ref:", torch.allclose(y_pytorch, m_ref(x), rtol=1e-3, atol=1e-3))  # True
print("CUDA-kernel-path matches ref:", torch.allclose(y_cuda, m_ref(x), rtol=1e-3, atol=1e-3))  # False
```

---

## 3. `L2/T090` — GELU 换成 tanh 近似 + `--use_fast_math`

> **通俗解释**：又一处 GELU 公式被偷换。这回是 `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))`——这是 PyTorch `F.gelu(approximate='tanh')` 的写法，**但 ref 用的是默认 `approximate='none'`(即 erf-exact)**。两个写法**不是同一个函数**，差量级 ~1e-4：默认 `rtol=1e-3` 的容差里看起来"通过"了，但收紧到 `rtol=1e-5` 立刻挂——这个差异不是浮点 round-off，是**算了另一个公式**。再叠加 `--use_fast_math` 编译 flag(denormals 归零、rsqrt/exp/div 走不严格的硬件 intrinsic)，偏离量再略放大。这一条说明：**"默认容差通过"≠"语义等价"**。

### 位置

- 数据集：[`@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T090_Conv3d_LeakyReLU_Sum_Clamp_GELU/optimized.py`](../cuda_l1/a100/level2/L2_T090_Conv3d_LeakyReLU_Sum_Clamp_GELU/optimized.py)

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T090_Conv3d_LeakyReLU_Sum_Clamp_GELU/optimized.py:67-72
            // Fast GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            const float sqrt_2_pi_inv = 0.7978845608028654f;  // sqrt(2/pi)
            const float coeff = 0.044715f;
            const float x_cubed = result * result * result;
            const float inner = sqrt_2_pi_inv * (result + coeff * x_cubed);
            result = 0.5f * result * (1.0f + tanhf(inner));
```

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T090_Conv3d_LeakyReLU_Sum_Clamp_GELU/optimized.py:175
        extra_cuda_cflags=["--use_fast_math", "-O3"]  # Enable fast math and high optimization
```

### 为什么是（较软的）bug

根因同 #1，量级更轻。Ref 调 `F.gelu`（erf-exact）；kernel 用 tanh approximation。两个附加因素：

1. **管道在 GELU 之前先 `clamp` 到 `[-1, 1]`**（`ref.py:17`），输入范围受限——tanh-approx 在 `[-1, 1]` 上与 erf-exact 的偏差（max ≈ 1.5e-4）比在 `[-3, 3]` 上（max ≈ 4.7e-4）更小。
2. **`--use_fast_math`** 额外引入 flush-denormals-to-zero、不精确的 reciprocal/sqrt、以及 intrinsic substitution，这些偏离与公式 mismatch 复合在一起。

### 设计好的 diff test

```python
PAIR = pathlib.Path("preliminary_dataset/cuda_l1/a100/level2/L2_T090_Conv3d_LeakyReLU_Sum_Clamp_GELU")
ref = _load("ref", PAIR / "ref.py")
opt = _load("opt", PAIR / "optimized.py")

m_ref = ref.Model(*ref.get_init_inputs()).cuda().eval()
m_opt = opt.ModelNew(*ref.get_init_inputs()).cuda().eval()
m_opt.load_state_dict(m_ref.state_dict(), strict=False)

x = ref.get_inputs()[0].cuda()
with torch.inference_mode():
    y_ref = m_ref(x); y_opt = m_opt(x)

print(f"max abs diff: {(y_ref - y_opt).abs().max().item():.3e}")
print(f"allclose@1e-3: {torch.allclose(y_ref, y_opt, rtol=1e-3, atol=1e-3)}")   # likely True
print(f"allclose@1e-5: {torch.allclose(y_ref, y_opt, rtol=1e-5, atol=1e-5)}")   # False
```

建议：对这个 pair **用 `rtol=atol=1e-5`** 跑严格检查（或单独拆分 `--rtol`/`--atol`）；默认 `1e-3` 会让它滑过。

---

## 4. 系统性：CUDA-L1 中 30 个 pair 带 `--use_fast_math`(compile-time numerical relaxation)

> **通俗解释**：这一条不是单 pair 的 bug，而是 1064 个 CUDA-L1 pair 里有 **30 个**（约 3%）在 `load_inline(..., extra_cuda_cflags=["--use_fast_math", ...])` 编译。这个 flag 等同于跟 NVCC 说"宽松一点"——denormals 直接 flush 到 0、除法 / sqrt / exp 改走更快但**不严格**的硬件路径（`__expf`、`__logf`、`__fdividef`...）。单个算子常常只差几个 ULP，但管道一长（softmax × LayerNorm × log，乘起来）就能累积到 `rtol=1e-3` 都救不回来——算法什么都不改就能拼出加速，又常常逃过宽松容差的 diff check，是 LLM 生成 CUDA 代码里**最常见的 reward-hacking 向量**之一。

### 命中数

```
grep -rl "use_fast_math" preliminary_dataset/cuda_l1/a100/ --include=optimized.py | wc -l
# → 30
```

### `--use_fast_math` 在 NVCC 下展开为

- `-ftz=true` — denormals flushed to zero
- `-prec-div=false` — 不精确除法（用 reciprocal × multiply）
- `-prec-sqrt=false` — 不精确 sqrt
- intrinsic substitution：`expf → __expf`、`logf → __logf`、`sinf → __sinf` 等等

每一项都可能带来 **几个 ULP** 到 **~1e-3** 量级的误差，取决于算子。管道里这类算子多了（BatchNorm、LayerNorm、softmax、exp/log 链），误差会乘法式累加——这就是 `ref(x)` 与 `opt(x)` 在严格容差下系统性偏离的根源。

### 检查方式

```python
import json
from pathlib import Path

flagged = []
for meta in Path("preliminary_dataset/cuda_l1").rglob("meta.json"):
    src = (meta.parent / "optimized.py").read_text()
    if "--use_fast_math" in src or "use_fast_math" in src:
        flagged.append(json.loads(meta.read_text())["pair_id"] if "pair_id" in meta.read_text() else str(meta.parent))
print(len(flagged), "pairs ship with --use_fast_math")
```

### 推荐姿态

语义 review agent 应将 `--use_fast_math` 视为一个**硬信号**——对带该 flag 的 pair 需要：
- 用更严的容差跑一次正确性检查（`rtol=atol=1e-5` 而非 `1e-3`），**以及**
- 另外并行跑一次把 `--use_fast_math` 从 `extra_cuda_cflags` 中剔除后的版本，以将算法偏离与数值松弛隔离开。

---

## 附录：相关但不构成"默认契约下语义偏离"的发现

> 以下两类是 **latent / scope-restriction bug**：在 ref 规定的默认契约（`get_init_inputs()` / `get_inputs()` 写定的 hyper-params + batch size + 同种子下的随机初始化）下，optimized 的数值输出与 ref **一致**——因此**不能**靠默认 diff test 抓到。但只要稍稍跨出 contract（改 channel 数 / batch size、调用 `load_state_dict` 导入与 init 时不同的权重、或跑过哪怕一步梯度更新），优化版立刻和 ref 偏离。这些 bug 不属于"默认契约下能直接观察到的语义偏离"，但对设计 review agent 的 *mutation pass* 有参考价值，留作附录。

### A. `L2/T054` — `__shared__` 硬编码 `[16]` 个 channel（latent OOB / 硬编码 shape）

> **通俗解释**：优化版 CUDA kernel 里写了 `__shared__ float s_multipliers[16];`——把 `out_channels=16` 这个 ref 默认值**直接烘进了共享内存大小**。当 `out_channels=16` 时一切正常，optimized 与 ref 完全一致；一旦把 `out_channels` 换成 32，第 17 个起的 thread 就会**越界写入共享内存**(undefined behavior，A100 上多半看似无害但属脏内存)。kernel 代码里甚至自带注释 `// max 16 channels`——说明作者明知有这个限制还是发布了。**默认契约 channels=16 时不偏离；shape mutation 下立即出现 `ref(x) ≠ opt(x)`。**

#### 位置

同 #1 同一个 `optimized.py`：

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T054_Conv2d_Multiply_LeakyReLU_GELU/optimized.py:66-71
                // Load multipliers into shared memory
                __shared__ float s_multipliers[16]; // max 16 channels
                if (threadIdx.x < channels) {
                    s_multipliers[threadIdx.x] = multiplier[threadIdx.x];
                }
                __syncthreads();
```

#### 为什么默认契约下不偏离

`ref.py` 里 `get_init_inputs() = [3, 16, 3, (16, 1, 1)]`(`out_channels = 16`)。kernel 在 `channels == 16` 时所有写入都在 `s_multipliers[0..15]` 范围内——合法。两个模型输出完全一致。

#### 为什么 shape mutation 下偏离

- **`channels > 16`**：thread 17..channels-1 写到 `s_multipliers[16..]`——**越界写**到共享内存的别处(其他 shared 分配 / register spill slot)，是 undefined behavior。
- **`channels < 16`**：尾部槽位未初始化但也不会被读到，行为未定义但实际无害。

#### 设计好的 diff test（shape mutation）

```python
# 跨出默认契约：把 out_channels 从 16 换成 32
in_ch, out_ch, ksize, mshape = 3, 32, 3, (32, 1, 1)
m_ref = ref.Model(in_ch, out_ch, ksize, mshape).cuda().eval()
m_opt = opt.ModelNew(in_ch, out_ch, ksize, mshape).cuda().eval()
m_opt.load_state_dict(m_ref.state_dict(), strict=False)

x = torch.randn(8, 3, 32, 32, device="cuda")
with torch.inference_mode():
    y_ref = m_ref(x); y_opt = m_opt(x)
print("max abs diff:", (y_ref - y_opt).abs().max().item())
# 预期：要么巨大数值偏离(共享内存被静默踩坏)，要么 NaN / 不可重现。
```

---

### B. `L2/T018` — `output_buffer` 锁死 `batch_size=128` + `weight_sum`/`bias_sum` 在 `__init__` 时缓存后从不刷新

> **通俗解释**：优化版做了一个数学上**正确**的代数化简(把 `linear → sum → 4 个 singleton-dim 算子` 坍缩为 `x @ weight_sum + bias_sum`)，但实现上偷了两个懒：
>
> 1. **`output_buffer` 在 `__init__` 时按模块级常量 `batch_size=128` 预分配**。换 batch size 直接抛 `RuntimeError`；而且每次 `forward` 返回的是同一个 buffer 对象——连续两次 `forward` 的结果会互相覆盖。
> 2. **`weight_sum` / `bias_sum` 在 `__init__` 时按当时的随机权重算一次就缓存住**，之后 `self.weight` / `self.bias` 怎么变都不刷新——一旦实际权重和缓存对不上，`forward` 用的就是"对老权重的预聚合"，输出立刻飘。
>
> 默认契约(batch=128 + 同种子下的随机初始化)下两件事都不会暴露：batch 是 128 没问题，weight 没人动过缓存就是对的——所以 `ref(x) == opt(x)`。但任何 batch mutation / 训练一步 / `load_state_dict` 进来一份新权重，立刻偏离。

#### 位置

- 数据集：
  - Reference: [`@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T018_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/ref.py`](../cuda_l1/a100/level2/L2_T018_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/ref.py)
  - Optimized: [`@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T018_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/optimized.py`](../cuda_l1/a100/level2/L2_T018_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/optimized.py)

#### 数学化简（这部分本身没问题）

ref 的 6 算子链 `linear → sum → max → mean → logsumexp → logsumexp` 在 `dim=1, keepdim=True` 下，第一次 `sum` 之后所有算子都退化为 singleton-dim 上的恒等。整个链可以代数化坍缩为 `x @ W.sum(dim=0).view(-1, 1) + b.sum()` —— **数学正确**。bug 出在周边的实现细节。

#### Bug B-1 — `output_buffer` 锁死在模块级 `batch_size = 128`

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T018_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/optimized.py:36-52
        # Pre-allocate output tensor for the known batch size
        # This eliminates memory allocation during forward pass
        self.register_buffer('output_buffer', torch.zeros(batch_size, 1, dtype=torch.float32))

    def forward(self, x):
        ...
        # Use torch.addmm for a fused multiply-add operation
        return torch.addmm(self.bias_sum, x, self.weight_sum, out=self.output_buffer)
```

第 38 行的 `batch_size` 是**模块级常量** `batch_size = 128`(第 55 行)——import 时就被 hard-bind 进来。后果：(a) 任何 `batch_size != 128` 都抛 `RuntimeError`；(b) 每次 forward 返回同一个 buffer，连续两次 forward 的结果共享 storage。

#### Bug B-2 — `weight_sum` / `bias_sum` 在 `__init__` 时缓存后从不刷新

```@/data/zrwang/KernelBench/preliminary_dataset/cuda_l1/a100/level2/L2_T018_Matmul_Sum_Max_AvgPool_LogSumExp_LogSumExp/optimized.py:27-34
        # Pre-compute the sum of weights for optimization
        weight_sum = torch.sum(self.weight, dim=0).contiguous().view(-1, 1)
        bias_sum = torch.sum(self.bias).item()

        self.register_buffer('weight_sum', weight_sum)
        self.register_buffer('bias_sum', torch.tensor([bias_sum], dtype=torch.float32))
```

以下任一场景都让缓存变 stale：(1) 训练梯度更新 `self.weight` / `self.bias`；(2) `load_state_dict` 导入新权重(buffer 也在 state_dict 里，会被 checkpoint 中的旧值覆写而不是重新计算)；(3) 任何外部直接覆写 `weight` / `bias`。

#### 为什么默认契约下仍不偏离

`_runtime.py` 在构造两个 model 前都 `torch.manual_seed(seed)`，然后两边走完全相同的 `kaiming_uniform_(weight) + uniform_(bias)`——RNG 序列一致 → `weight` 和 `bias` 字节级相等 → opt 在 `__init__` 算出的 `weight_sum` / `bias_sum` 也对得上。两个 model 输出一致。

但只要再多走一步训练，或在测试中 `model.load_state_dict(some_other_state)`，缓存立刻和实际权重脱节。

#### 设计好的 diff test

```python
# B-1 — batch-size mutation
m_opt = opt.ModelNew(10, 5).cuda().eval()
x_64 = torch.randn(64, 10, device="cuda")        # 不是 128
try:
    y = m_opt(x_64)
    print("UNEXPECTED: no error")
except RuntimeError as e:
    print(f"caught: {e}")                         # output_buffer shape mismatch

# B-2 — param mutation 让缓存 stale
m_opt = opt.ModelNew(10, 5).cuda().eval()
with torch.no_grad():
    m_opt.weight.fill_(1.0)
    m_opt.bias.fill_(0.0)

x = torch.randn(128, 10, device="cuda")
y_with_stale_cache = m_opt(x).clone()
y_should_be = (x.sum(dim=1) * 5).view(-1, 1)      # forward 现在应该算的
print("max abs diff after weight mutation:",
      (y_with_stale_cache - y_should_be).abs().max().item())   # 巨大
```

---

## 我们检索了哪些模式（及发现结果）

| 模式 | 检索 query | CUDA-L1 / A100 命中数 | 最有价值的命中 | 落在主表还是附录？ |
|------|--------------|----------------------|------------------|-------------------|
| 数值近似(fast GELU、fast softplus 等) | `Fast GELU\|fast_softplus\|approximation` | 多 | **L2/T054**、**L2/T090** | 主表 #1、#3 |
| try/except 的多路径 forward(对计时敏感的 dispatch) | `try:\s*\n.*forward.*self\.jit\|_cuda_graph` | 多 | L2/T054, L2/T073 | 主表 #2 |
| `--use_fast_math` 编译标志 | `--use_fast_math` | **30** | 系统性 | 主表 #4 |
| 硬编码的 `__shared__` 大小 | `__shared__ float \w+\[\d+\]` | 多 | **L2/T054**(`[16]`) | 附录 A |
| 预分配 output buffer，绑模块级 `batch_size` | `register_buffer.*\(.*batch_size` | 8+ | L2/T018, L2/T051, L2/T084 | 附录 B |
| 预缓存 weight transpose / fused weights / `_sum` tensor | `self\.weight_t\|weight_sum\|bias_sum` | 多个 | L2/T018, L2/T022 | 附录 B |
| 缓存 BatchNorm fused params(`_compute_fused_parameters`、`_cache_bn_params`) | `_cache_bn_params\|_compute_fused_parameters` | 5+ | L2/T015, L2/T073 | 同附录 B 类，待补 |
| `ModelNew` 内部包裹 CUDA-Graph | `torch\.cuda\.CUDAGraph\|self\.graph` | 10+ | L2/T073, L3/T040 | 同附录 B 类，待补 |

---

## 对 review agent 的推荐

1. **先用现有 `run_diff_test.py` 跑默认输入(rtol=atol=1e-3)** — 直接抓主表 #1、#2(走到 CUDA 分支时)。属于"免费覆盖"。
2. **加一个 *tight-tolerance* pass(rtol=atol=1e-5)**：对带 `--use_fast_math` 或含 `Fast GELU` / `fast_*` 注释的 pair 重跑——抓主表 #3、#4。
3. **加一个 *path-determinism* check**：若 `ModelNew.forward` 内部有 `self.jit_compiled` / `self.cuda_kernel_loaded` / `self.use_cuda_graph` 等多分支，分别强制开启每个分支跑一遍然后互比——更稳地暴露 #2。
4. **加一个 *shape-mutation* pass**：改动一两个模块级常量(`batch_size`、`out_channels`…)后重建 model 再跑 diff——抓附录 A、B-1 以及其他 shape-locked kernel。
5. **加一个 *parameter-mutation* pass**：默认初始下 diff 通过后，随机覆写 `weight` / `bias`(或 `model.load_state_dict` 进一份新权重)再跑一次 diff——抓附录 B-2 以及 L2/T015、L2/T073 的 BatchNorm-fold 类 stale-cache bug。

第 1 项就是我们现有的 `batch_diff_test.py --rtol 1e-3 --atol 1e-3` 在做的事；2–5 是 review pipeline 自然的后续层。**主表的 4 条全部由 1+2+3 覆盖；附录的 A、B 由 4+5 覆盖。**
