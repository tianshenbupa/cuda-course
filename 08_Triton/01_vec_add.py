"""
==========================================================
Triton × PyTorch 向量加法性能对比实验
----------------------------------------------------------
本脚本展示如何：

1. 使用 **Triton** 编写一个 SPMD GPU kernel (`add_kernel`) 完成向量加法。
2. 通过 `triton.testing.Benchmark` 与 **PyTorch** 的 `x + y`
   做端到端带宽 (GB/s) 基准测试。
3. 自动绘制 “向量长度 × 吞吐量” 对比曲线 (`vector-add-performance.png`)。

核心要点
▶ BLOCK_SIZE 决定每个 Triton “程序”(≈CUDA block) 处理的元素数。
▶ `tl.load / tl.store` 提供自动掩码，避免越界。
▶ Triton kernel 通过 `grid = (ceiling_div(n, BLOCK_SIZE),)` 启动，
  语义对应 CUDA 的 1D block grid。
==========================================================
"""

import time
import torch
import triton
import triton.language as tl


# --------------------------------------------------------
# Triton Kernel：向量逐元素相加 (x + y → output)
# --------------------------------------------------------
@triton.jit
def add_kernel(
        x_ptr,           # ➜ 输入向量 x 的指针
        y_ptr,           # ➜ 输入向量 y 的指针
        output_ptr,      # ➜ 输出向量 output 的指针
        n_elements,      # ➜ 向量总长度
        BLOCK_SIZE: tl.constexpr  # ➜ 每个“程序”处理的元素数 (常量形参)
):
    # program‑id 相当于 CUDA 的 blockIdx.x
    pid = tl.program_id(axis=0)

    # 计算本程序负责的元素区间起点
    block_start = pid * BLOCK_SIZE
    # 当前 BLOCK 内各线程所处理的全局索引
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 掩码：防止最后一个 BLOCK 越界
    mask = offsets < n_elements

    # 从全局内存加载 x、y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 逐元素相加
    output = x + y

    # 写回结果
    tl.store(output_ptr + offsets, output, mask=mask)


# --------------------------------------------------------
# Python 端包装函数，调用 Triton kernel
# --------------------------------------------------------
def add(x: torch.Tensor, y: torch.Tensor):
    """
    Triton 版本向量加法。保证 x、y、output 均在 GPU 上。
    返回值与 x/y 同 shape、dtype。
    """
    output = torch.empty_like(x)          # 预分配输出张量
    n_elements = output.numel()           # 元素总数

    # 1D launch grid：每个程序负责 BLOCK_SIZE 个元素
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 异步启动 kernel（未同步）
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


# --------------------------------------------------------
# 生成随机测试数据
# --------------------------------------------------------
torch.manual_seed(0)
size = 2 ** 25            # 默认最大向量长度
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')


# --------------------------------------------------------
# Benchmark 配置与运行
# triton.testing.perf_report 会自动生成表格和折线图
# --------------------------------------------------------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],                    # 横坐标：向量长度
        x_vals=[2 ** i for i in range(12, 28)],  # 2^12 (4 K) → 2^27 (134 M)
        x_log=True,                          # 对数刻度
        line_arg='provider',                 # 不同曲线：实现方式
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',                       # 纵坐标：有效带宽
        plot_name='vector-add-performance',
        args={},                             # 其他固定参数
    )
)
def benchmark(size, provider):
    """
    对给定 size 和 provider 进行一次基准测量。
    返回 (50% 分位, 20% 最快, 80% 最慢) 的 GB/s。
    """
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    # λ‑表达式用于 do_bench 循环测时
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y,
                                                     quantiles=quantiles)
    else:  # provider == 'triton'
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y),
                                                     quantiles=quantiles)

    # 带宽计算公式：读 x、读 y、写 output ⇒ 3×bytes
    gbps = lambda t_ms: 3 * x.numel() * x.element_size() / t_ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# ==== 执行基准测试并打印数据 / 绘图 ====
benchmark.run(print_data=True, show_plots=True)
