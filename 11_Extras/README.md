# Extras

## CUDA Compiler
![](assets/nvcc.png)

## How does CUDA handle conditional if/else logic?
- CUDA does not handle conditional if/else logic well. If you have a conditional statement in your kernel, the compiler will generate code for both branches and then use a predicated instruction to select the correct result. This can lead to a lot of wasted computation if the branches are long or if the condition is rarely met. It is generally a good idea to try to avoid conditional logic in your kernels if possible.
- If it is unavoidable, you can dig down to the PTX assembly code (`nvcc -ptx kernel.cu -o kernel`) and see how the compiler is handling it. Then you can look into the compute metrics of the instructions used and try to optimize it from there.
- Single thread going down a long nested if else statement will look more serialized and leave the other threads waiting for the next instruction while the single threads finishes. this is called **warp divergence** and is a common issue in CUDA programming when dealing with threads specifically within a warp.
- vector addition is fast because divergence isn’t possible, not a different possible way for instructions to carry out.

## Pros and Cons of Unified Memory
- Unified Memory is a feature in CUDA that allows you to allocate memory that is accessible from both the CPU (system DRAM) and the GPU. This can simplify memory management in your code, as you don't have to worry about copying data back and forth between the the RAM sticks and the GPU's memory.
- [Unified vs Explicit Memory in CUDA](https://github.com/lintenn/cudaAddVectors-explicit-vs-unified-memory)
- [Maximizing Unified Memory Performance](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)
- Prefetching is automatically taken care of by unified memory via **streams** (this is what is has lower latency in the github link above)
    - [CUDA streams - Lei Mao](https://leimao.github.io/blog/CUDA-Stream/)
    - [NVIDIA Docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)
    - Streams allow for overlapping data transfer (prefetching) with computation.
    - While one stream is executing a kernel, another stream can be transferring data for the next computation.
    - This technique is often called "double buffering" or "multi-buffering" when extended to more buffers.

![](assets/async.png)

## Memory Architectures
- DRAM/VRAM cells are the smallest unit of memory in a computer. They are made up of capacitors and transistors that store bits of data. The capacitors store the bits as electrical charges, and the transistors control the flow of electricity to read and write the data.
- ![](assets/dram-cell.png)
- SRAM (shared memory) is a type of memory that is faster and more expensive than DRAM. It is used for cache memory in CPUs and GPUs because it can be accessed more quickly than DRAM. 
- Modern NVIDIA GPUs likely use 6T (six-transistor) or 8T SRAM cells for most on-chip memory.
6T cells are compact and offer good performance, while 8T cells can provide better stability and lower power consumption at the cost of larger area.
- 6T vs 8T SRAM cells in NVIDIA GPUs across different architectures and compute capabilities isn't publicly disclosed in detail. NVIDIA, like most semiconductor companies, keeps many of these low-level design choices proprietary.
- ![](assets/sram-cell.png)
- ![](assets/8t-sram-cell.png)


## Dive deeper
- quantization -> fp32 -> fp16 -> int8
- tensor cores (wmma)
- sparsity -> [0, 0, 0, 0, -7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6]
- [CUDA by Example](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf)
- [Data-Parallel Distributed Training of Deep Learning Models](https://siboehm.com/articles/22/data-parallel-training)
- [mnist-cudnn](https://github.com/haanjack/mnist-cudnn)
- [CUDA MODE](https://github.com/cuda-mode/lectures)
- [micrograd-cuda](https://github.com/mlecauchois/micrograd-cuda)
- [micrograd](https://github.com/karpathy/micrograd)
- [GPU puzzles](https://github.com/srush/GPU-Puzzles)


以下是你提供内容的**完整中文翻译与整理注解**：

---

# 🌟 附加说明（Extras）

---

## ✅ CUDA 编译器

* `nvcc` 是 NVIDIA 提供的 CUDA 编译器，用于将 `.cu` 文件编译为可供 GPU 执行的 PTX 或二进制代码。

---

## ❓CUDA 如何处理 if/else 条件逻辑？

* CUDA **并不擅长处理条件分支逻辑**。
  如果 kernel 中包含 `if/else` 语句，编译器通常会：

  * 为 **每个分支都生成代码**
  * 并通过 **谓词化指令**（predicated instruction）来选择实际执行的路径

### ⚠️ 这样做可能导致的问题：

* **分支逻辑越复杂或满足条件的线程越少时，性能损耗越大**
* **浪费了大量未使用分支的计算资源**

### 🔍 调试建议：

* 使用 `nvcc -ptx kernel.cu -o kernel.ptx` 生成 PTX 汇编，检查实际编译结果
* 查看编译后是否出现不必要的冗余分支路径

### 🚧 Warp Divergence（线程束分歧）

* 当一个 warp（32 个线程）中有部分线程执行了不同的分支路径，**其余线程必须等待**
  → **这会导致线程束中线程串行执行，影响并行性**
* 比如一个线程深度嵌套的 if-else，其它线程就得等待它执行完

✅ 例如向量加法（vector addition）没有条件分支，因此运行效率非常高。

---

## ✅ Unified Memory（统一内存）的优劣势

### 📌 定义

* 统一内存是 CUDA 提供的一种机制，允许分配的内存在 CPU 和 GPU 间共享访问。
* 用户不再需要显式地在 host/device 之间复制数据。

### ✅ 优点

* **简化内存管理**：无需调用 `cudaMemcpy`
* **自动预取**（prefetch）：CUDA 会在合适时机使用 stream 将数据自动迁移

### ⚠️ 缺点

* 性能可预测性差，**不当使用会导致延迟**或 page fault
* 在复杂场景下，手动管理显存可能更高效

### 📚 推荐资料

* [Unified vs Explicit Memory in CUDA (GitHub 示例)](https://github.com/lintenn/cudaAddVectors-explicit-vs-unified-memory)
* [官方：最大化统一内存性能](https://developer.nvidia.com/blog/maximizing-unified-memory-performance-cuda/)

---

## 🚀 CUDA Streams 与异步数据预取

### 概念

* CUDA stream 允许多个 kernel 或数据传输 **并发执行**。
* 一般可使用「双缓冲」「多缓冲」等方式实现计算与数据传输重叠。

### 推荐阅读

* [Lei Mao 的 stream 教程](https://leimao.github.io/blog/CUDA-Stream/)
* [NVIDIA 官方文档：异步执行](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#asynchronous-concurrent-execution)

---

## 🧠 内存架构概述（Memory Architectures）

### 1. DRAM / VRAM

* 由电容与晶体管组成，每个 cell 存储 1 bit
* 成本低但访问速度慢
* GPU 全局内存使用 DRAM 技术

图示：

* ![DRAM cell](assets/dram-cell.png)

---

### 2. SRAM（共享内存 / 寄存器缓存）

* 用于 CPU/GPU 缓存（L1/L2 cache、shared memory）

* 比 DRAM 快很多，但成本高

* **6T SRAM（6 个晶体管）**：

  * 更紧凑，占用芯片面积小，速度快

* **8T SRAM**：

  * 更稳定、功耗低，但占用面积更大

图示：

* ![SRAM cell](assets/sram-cell.png)
* ![8T SRAM cell](assets/8t-sram-cell.png)

> 💡 NVIDIA 实际使用哪种 SRAM 设计未完全公开，但主要为片上高速缓存使用 6T/8T 结构。

---

## 📚 深入探索方向（Dive deeper）

| 主题                | 说明                                                                 |
| ----------------- | ------------------------------------------------------------------ |
| 量化 (Quantization) | 从 float32 → float16 → int8，牺牲精度以加速推理                               |
| Tensor Core       | GPU 矩阵乘专用硬件，支持半精度与稀疏计算                                             |
| 稀疏性 (Sparsity)    | 在模型中大量为 0 的位置可被优化跳过，如 `[0, 0, ..., 6]`                             |
| 分布式训练             | [数据并行训练分析](https://siboehm.com/articles/22/data-parallel-training) |

---

## 🔗 推荐项目与资料

* 📘 [CUDA by Example](https://edoras.sdsu.edu/~mthomas/docs/cuda/cuda_by_example.book.pdf)
* 🔍 [Explicit vs Unified Memory 示例](https://github.com/lintenn/cudaAddVectors-explicit-vs-unified-memory)
* 🔬 [mnist-cudnn（卷积加速示例）](https://github.com/haanjack/mnist-cudnn)
* 🎓 [CUDA MODE 教程集合](https://github.com/cuda-mode/lectures)
* 🧠 [micrograd-cuda（Karpathy 微型框架的 CUDA 实现）](https://github.com/mlecauchois/micrograd-cuda)
* 🧠 [micrograd（原版 Py 实现）](https://github.com/karpathy/micrograd)
* 🎮 [GPU 益智题](https://github.com/srush/GPU-Puzzles) - 用于练习 CUDA 思维模型的挑战题

---

如需我为你把这些内容转为 Markdown 学习笔记、幻灯片或整理成可打印 PDF，也可以告诉我。
