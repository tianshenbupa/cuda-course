"""
示例：对比 PyTorch 纯 Python 实现与自定义 CUDA 扩展实现的多项式激活函数性能

功能概述
--------
1. 定义 `PolynomialActivation` 模块，支持 'pytorch' 与 'cuda' 两种实现：
   - pytorch: 直接采用张量算子 `x**2 + x + 1`
   - cuda: 调用自定义 CUDA 扩展 `polynomial_cuda.polynomial_activation`
2. 提供基于 `torch.autograd.Function` 的前向封装 `CUDAPolynomialActivation`
3. 通过 `benchmark` 函数对两种实现进行多次前向性能测试
4. 在 `main()` 中生成随机输入、运行测试并打印输出结果与平均耗时（毫秒）

使用说明
--------
1. 先根据项目根目录中的 `setup.py` 编译安装 CUDA 扩展：
   ```bash
   python setup.py install
   ```
2. 运行本脚本：
   ```bash
   python polynomial_activation_benchmark.py
   ```

注意事项
--------
- 请确保系统已正确安装 CUDA 且 PyTorch 支持 GPU。
- 如果需要反向传播，请在 `CUDAPolynomialActivation.backward` 中实现对应的 CUDA 后向算子。
"""

import time  # 计时用
import torch
import torch.nn as nn
import polynomial_cuda  # 假设已使用 setup.py 编译并安装


class CUDAPolynomialActivation(torch.autograd.Function):
    """封装自定义 CUDA 前向实现，可选实现反向传播"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """前向传播：调用 CUDA 扩展函数"""
        return polynomial_cuda.polynomial_activation(x)

    @staticmethod
    def backward(ctx, grad_output):
        """后向传播占位符：如需梯度请自行实现"""
        raise NotImplementedError("Backward pass not implemented")


class PolynomialActivation(nn.Module):
    """多项式激活函数，支持两种实现方式"""

    def __init__(self, implementation: str = "pytorch"):
        super().__init__()
        self.implementation = implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据 implementation 选择不同实现"""
        if self.implementation == "pytorch":
            return x ** 2 + x + 1  # 纯 PyTorch 张量运算
        elif self.implementation == "cuda":
            return CUDAPolynomialActivation.apply(x)  # 调用 CUDA 扩展
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")


def benchmark(func: nn.Module, x: torch.Tensor, name: str, num_runs: int = 1000) -> str:
    """简单基准测试：多次前向求平均耗时（毫秒）"""
    # 记录开始时间
    start_time = time.time()
    for _ in range(num_runs):
        func(x)
    # 等待所有 GPU kernel 完成，确保计时准确
    torch.cuda.synchronize()
    end_time = time.time()
    avg_ms = (end_time - start_time) / num_runs * 1000
    return f"{name}: {avg_ms:.4f} ms"


def main():
    """脚本入口：构造输入并比较两种实现的性能"""
    torch.manual_seed(0)  # 保证可重复

    # 构造 100 万元素的随机向量，放到 GPU
    x = torch.randn(1_000_000, device="cuda")

    # 初始化两种实现的激活函数模块，并移至 GPU
    pytorch_activation = PolynomialActivation(implementation="pytorch").cuda()
    cuda_activation = PolynomialActivation(implementation="cuda").cuda()

    # 调用 CUDA 实现并打印样例输出（前 10 个值）
    out = cuda_activation(x)
    print("Sample output (first 10 elements):", out[:10])

    # 运行基准测试
    pytorch_time = benchmark(pytorch_activation, x, "PyTorch built-in")
    cuda_time = benchmark(cuda_activation, x, "CUDA extension")

    # 打印性能对比
    print(pytorch_time)
    print(cuda_time)


if __name__ == "__main__":
    main()
"""
示例：对比 PyTorch 纯 Python 实现与自定义 CUDA 扩展实现的多项式激活函数性能

功能概述
--------
1. 定义 `PolynomialActivation` 模块，支持 'pytorch' 与 'cuda' 两种实现：
   - pytorch: 直接采用张量算子 `x**2 + x + 1`
   - cuda: 调用自定义 CUDA 扩展 `polynomial_cuda.polynomial_activation`
2. 提供基于 `torch.autograd.Function` 的前向封装 `CUDAPolynomialActivation`
3. 通过 `benchmark` 函数对两种实现进行多次前向性能测试
4. 在 `main()` 中生成随机输入、运行测试并打印输出结果与平均耗时（毫秒）

使用说明
--------
1. 先根据项目根目录中的 `setup.py` 编译安装 CUDA 扩展：
   ```bash
   python setup.py install
   ```
2. 运行本脚本：
   ```bash
   python polynomial_activation_benchmark.py
   ```

注意事项
--------
- 请确保系统已正确安装 CUDA 且 PyTorch 支持 GPU。
- 如果需要反向传播，请在 `CUDAPolynomialActivation.backward` 中实现对应的 CUDA 后向算子。
"""

import time  # 计时用
import torch
import torch.nn as nn
import polynomial_cuda  # 假设已使用 setup.py 编译并安装


class CUDAPolynomialActivation(torch.autograd.Function):
    """封装自定义 CUDA 前向实现，可选实现反向传播"""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """前向传播：调用 CUDA 扩展函数"""
        return polynomial_cuda.polynomial_activation(x)

    @staticmethod
    def backward(ctx, grad_output):
        """后向传播占位符：如需梯度请自行实现"""
        raise NotImplementedError("Backward pass not implemented")


class PolynomialActivation(nn.Module):
    """多项式激活函数，支持两种实现方式"""

    def __init__(self, implementation: str = "pytorch"):
        super().__init__()
        self.implementation = implementation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """根据 implementation 选择不同实现"""
        if self.implementation == "pytorch":
            return x ** 2 + x + 1  # 纯 PyTorch 张量运算
        elif self.implementation == "cuda":
            return CUDAPolynomialActivation.apply(x)  # 调用 CUDA 扩展
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")


def benchmark(func: nn.Module, x: torch.Tensor, name: str, num_runs: int = 1000) -> str:
    """简单基准测试：多次前向求平均耗时（毫秒）"""
    # 记录开始时间
    start_time = time.time()
    for _ in range(num_runs):
        func(x)
    # 等待所有 GPU kernel 完成，确保计时准确
    torch.cuda.synchronize()
    end_time = time.time()
    avg_ms = (end_time - start_time) / num_runs * 1000
    return f"{name}: {avg_ms:.4f} ms"


def main():
    """脚本入口：构造输入并比较两种实现的性能"""
    torch.manual_seed(0)  # 保证可重复

    # 构造 100 万元素的随机向量，放到 GPU
    x = torch.randn(1_000_000, device="cuda")

    # 初始化两种实现的激活函数模块，并移至 GPU
    pytorch_activation = PolynomialActivation(implementation="pytorch").cuda()
    cuda_activation = PolynomialActivation(implementation="cuda").cuda()

    # 调用 CUDA 实现并打印样例输出（前 10 个值）
    out = cuda_activation(x)
    print("Sample output (first 10 elements):", out[:10])

    # 运行基准测试
    pytorch_time = benchmark(pytorch_activation, x, "PyTorch built-in")
    cuda_time = benchmark(cuda_activation, x, "CUDA extension")

    # 打印性能对比
    print(pytorch_time)
    print(cuda_time)


if __name__ == "__main__":
    main()
