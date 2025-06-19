/*
——————————————————————————————————————————————————————————
文件说明：polynomial_activation_cuda.cu
功能概述：
    1. 实现多项式激活函数 f(x) = x^2 + x + 1 的 CUDA Kernel。
    2. 提供 C++/PyTorch 封装函数 `polynomial_activation_cuda` 以便 Python 调用。
    3. 使用 AT_DISPATCH_FLOATING_TYPES 适配 float / double 等不同精度。
编译方式：
    该文件会由 setup.py 与 torch.utils.cpp_extension 编译，最终生成可供 Python 导入的扩展模块。
使用示例（Python）：
    >>> import polynomial_cuda
    >>> y = polynomial_cuda.polynomial_activation(x)
注意事项：
    * 如需反向传播，请编写对应的 backward kernel 并在 autograd.Function 中实现。
——————————————————————————————————————————————————————————
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ===================== CUDA Kernel 实现 ===================== //
// `scalar_t` 由 AT_DISPATCH_FLOATING_TYPES 在编译期推导，
// 可对应 float、double 等浮点类型。

template <typename scalar_t>
__global__ void polynomial_activation_kernel(const scalar_t* __restrict__ x,
                                             scalar_t* __restrict__ output,
                                             size_t size)
{
    // 计算全局索引
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，防止越界访问
    if (idx < size) {
        scalar_t val = x[idx];
        output[idx] = val * val + val + static_cast<scalar_t>(1);
    }
}

// ===================== C++ 封装函数 ===================== //
// 输入：PyTorch Tensor x (GPU)
// 输出：与 x 同形状同 dtype 的 Tensor output

torch::Tensor polynomial_activation_cuda(torch::Tensor x)
{
    // 创建与输入同形状同 dtype 的输出张量
    auto output = torch::empty_like(x);

    // 为 kernel 计算网格/线程配置
    const int threads = 1024;
    const int blocks  = (x.numel() + threads - 1) / threads;

    // 根据输入 dtype 调度相应模板实例
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "polynomial_activation_cuda", ([&] {
        polynomial_activation_kernel<scalar_t><<<blocks, threads>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            x.numel());
    }));

    return output;
}

// ===================== Python 绑定 ===================== //
// 通过 pybind11 将函数导出到 Python 模块

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("polynomial_activation", &polynomial_activation_cuda,
          "多项式激活函数：y = x^2 + x + 1 (CUDA)");
}
