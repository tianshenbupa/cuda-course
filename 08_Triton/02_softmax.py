"""
===========================================================
Triton × PyTorch Softmax 对比示例
-----------------------------------------------------------
功能概览
1. 使用 **Triton** 编写并 JIT 编译一个按行 Softmax 的 GPU Kernel。
2. 调用 `triton_softmax` 计算 2‑D Tensor（形状：n_rows × n_cols）的
   softmax，并与 **PyTorch** 的 `torch.softmax` 结果对比。
3. 打印两者最大绝对误差与一致性判断。

核心要点
▶ 数值稳定性：先减去每行最大值再做 exp，避免溢出。  
▶ `BLOCK_SIZE` 取列数的最近 2 次幂，且不超过 1024。  
▶ Kernel 通过 `tl.arange` + mask 只加载行内有效元素。  
===========================================================
"""

import torch
import triton
import triton.language as tl

# --------------------------------------------------------
# Triton Kernel：按行 Softmax（数值稳定版）
# --------------------------------------------------------
@triton.jit
def softmax_kernel(
    output_ptr,             # ➜ 输出张量首地址
    input_ptr,              # ➜ 输入张量首地址
    input_row_stride,       # ➜ 输入张量行步长
    output_row_stride,      # ➜ 输出张量行步长
    n_cols,                 # ➜ 每行元素数
    BLOCK_SIZE: tl.constexpr,  # ➜ 每行一次加载的 BLOCK 大小（2 次幂）
):
    # 当前 program 对应的行号 (row_idx)
    row_idx = tl.program_id(axis=0)

    # 计算当前行在内存中的起始地址
    row_start_ptr    = input_ptr  + row_idx * input_row_stride
    out_row_start_ptr = output_ptr + row_idx * output_row_stride

    # -------- ① 载入一行数据到 SRAM --------
    cols = tl.arange(0, BLOCK_SIZE)                       # 0..BLOCK_SIZE-1
    mask = cols < n_cols                                  # 防止越界
    row  = tl.load(row_start_ptr + cols, mask=mask,
                   other=-float('inf'))                   # 越界填 -inf

    # -------- ② 数值稳定：减 max 后做 exp --------
    row_max   = tl.max(row, axis=0)                       # 每行最大值
    numerator = tl.exp(row - row_max)                     # e^(x - max)

    # -------- ③ 求和并归一化 --------
    denominator     = tl.sum(numerator, axis=0)           # ∑ e^(x - max)
    softmax_output  = numerator / denominator             # softmax

    # -------- ④ 写回结果 --------
    tl.store(out_row_start_ptr + cols, softmax_output, mask=mask)


# --------------------------------------------------------
# Triton 包装函数：自动确定 BLOCK_SIZE 并调用 Kernel
# --------------------------------------------------------
def triton_softmax(x: torch.Tensor):
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # BLOCK_SIZE 取列数的最近 2 次幂，最大 1024
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    # Launch grid：每个 program 处理一行
    grid = (n_rows,)

    softmax_kernel[grid](
        output, x,
        x.stride(0), output.stride(0),  # 行步长（元素数）
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


# ===================== 测试 =====================
torch.manual_seed(0)
x = torch.randn(256, 1024, device='cuda')  # 256×1024 随机输入

# PyTorch 参考结果
torch_result = torch.softmax(x, dim=1)

# Triton 结果
triton_result = triton_softmax(x)

# 误差统计
max_diff = torch.max(torch.abs(torch_result - triton_result))
print(f"Maximum difference between PyTorch and Triton results: {max_diff:.2e}")

# 一致性判断
is_close = torch.allclose(torch_result, triton_result, rtol=1e-5, atol=1e-5)
print(f"Results are close: {is_close}")
