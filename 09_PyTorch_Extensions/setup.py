"""
setup.py 配置文件
------------------
用于将自定义 CUDA 扩展（polynomial_cuda.cu）编译为可被 Python 调用的模块。

主要功能：
1. 使用 PyTorch 提供的 BuildExtension 和 CUDAExtension。
2. 指定扩展名称为 'polynomial_cuda'，编译源码文件 'polynomial_cuda.cu'。
3. 安装后可通过 import polynomial_cuda 使用。

编译命令：
    python setup.py install
或
    python setup.py build_ext --inplace

注意：
- 需要确保 CUDA 工具链和 PyTorch 正确安装。
- 编译结果会被缓存至 ~/.cache/torch_extensions 目录中。
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 调用 setup 完成构建配置
setup(
    name='polynomial_cuda',  # 模块名，Python 中调用时使用该名
    ext_modules=[
        CUDAExtension(
            name='polynomial_cuda',  # 生成的 Python 模块名称
            sources=['polynomial_cuda.cu']  # 源文件路径（仅包含 CUDA）
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension  # 使用 PyTorch 提供的构建器
    }
)
