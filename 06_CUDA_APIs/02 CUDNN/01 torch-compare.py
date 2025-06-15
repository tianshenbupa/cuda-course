"""
=========================== PyTorch 2D 卷积示例 ============================
本脚本与 CUDA/cuDNN 版本的 2D 卷积基准测试配套，主要用于数值一致性
验证。

功能概览：
- 构造 1×1×4×4 输入张量（元素 1–16）
- 构造 1×1×3×3 卷积核（元素 1–9）
- 使用 `torch.nn.functional.conv2d` 进行填充为 1 的 2D 卷积
- 打印输入、卷积核、输出张量，以及扁平化输出列表，方便与 CUDA 结果对照

作者：ChatGPT 示例
日期：2025 年
"""

import torch
import torch.nn.functional as F

# ------------------------- 参数设置 -------------------------
width = 4          # 输入张量宽度
height = 4         # 输入张量高度
kernel_size = 3    # 卷积核尺寸（3x3）
in_channels = 1    # 输入通道数
out_channels = 1   # 输出通道数
batch_size = 1     # Batch 大小

# ------------------------- 构造输入张量 -------------------------
input_values = torch.tensor(
    [
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    ],
    dtype=torch.float32,
).reshape(batch_size, in_channels, height, width)  # 形状: (1,1,4,4)

# ------------------------- 构造卷积核张量 -------------------------
kernel_values = torch.tensor(
    [
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    ],
    dtype=torch.float32,
).reshape(out_channels, in_channels, kernel_size, kernel_size)  # 形状: (1,1,3,3)

# ------------------------- 执行 2D 卷积 -------------------------
# padding=kernel_size//2 保证输出尺寸与输入一致
output = F.conv2d(input_values, kernel_values, padding=kernel_size // 2)

# ------------------------- 打印结果 -------------------------
print("Input:\n", input_values)
print("\nKernel:\n", kernel_values)
print("\nOutput:\n", output)

# 扁平化输出，便于与 CUDA 版本逐元素对比
flattened = output.flatten().tolist()
print("\nFlattened output:")
print(flattened)
print("元素数量:", len(flattened))
