以下是该 CUDA 课程 GitHub 仓库主页的中文翻译：

---

# CUDA 课程

FreeCodeCamp 上的 CUDA 课程 GitHub 仓库

> 注意：本课程专为 Ubuntu Linux 设计。Windows 用户可以使用 Windows 子系统 (WSL) 或 Docker 容器来模拟 Ubuntu 环境。

---

## 目录

1. [深度学习生态系统](01_Deep_Learning_Ecosystem/README.md)
2. [环境设置/安装](02_Setup/README.md)
3. [C/C++ 回顾](03_C_and_C++_Review/README.md)
4. [GPU 入门简介](04_Gentle_Intro_to_GPUs/README.md)
5. [编写你的第一个 CUDA 内核](05_Writing_your_First_Kernels/README.md)
6. [CUDA API（cuBLAS、cuDNN 等）](06_CUDA_APIs/README.md)
7. [优化矩阵乘法](07_Faster_Matmul/README.md)
8. [Triton](08_Triton/README.md)
9. [PyTorch CUDA 扩展](08_PyTorch_Extensions/README.md)
10. [期末项目](09_Final_Project/README.md)
11. [补充资料](10_Extras/README.md)

---

## 课程理念

本课程旨在：

* 降低进入高性能计算（HPC）工作的门槛
* 为理解如 Karpathy 的 [llm.c](https://github.com/karpathy/llm.c) 等项目打下基础
* 将分散的 CUDA 编程资源整合成一个全面、有条理的课程

---

## 课程概述

* 专注于 GPU 内核优化以提升性能
* 涵盖 CUDA、PyTorch 和 Triton
* 注重编写高效内核的技术细节
* 针对 NVIDIA GPU 进行设计
* 最终项目是使用 CUDA 实现一个简单的 MLP 识别 MNIST

---

## 先修知识

* **必须：** Python 编程基础
* **推荐：** 反向传播中的微分和向量微积分
* **推荐：** 线性代数基础

---

## 你将学到

* 优化已有实现的方法
* 构建用于前沿研究的 CUDA 内核
* 理解 GPU 性能瓶颈，特别是内存带宽相关问题

---

## 硬件要求

* 任意 NVIDIA GTX、RTX 或数据中心级 GPU
* 没有本地硬件？可使用云端 GPU 资源

---

## CUDA/GPU 编程应用场景

* 深度学习（本课程重点）
* 图形学和光线追踪
* 流体模拟
* 视频编辑
* 加密货币挖矿
* 3D 建模
* 任何需要并行处理大规模数组的场景

---

## 学习资源

* 本 GitHub 仓库
* Stack Overflow
* NVIDIA 开发者论坛
* NVIDIA 和 PyTorch 官方文档
* 使用大型语言模型（LLM）辅助探索
* [速查表](11_Extras/assets/cheatsheet.md)

---

## 其他学习资料

* [https://github.com/CoffeeBeforeArch/cuda\_programming](https://github.com/CoffeeBeforeArch/cuda_programming)
* [https://www.youtube.com/@GPUMODE](https://www.youtube.com/@GPUMODE)
* [https://discord.com/invite/gpumode](https://discord.com/invite/gpumode)

---

## 有趣的 YouTube 视频推荐

* [GPU 是怎么工作的？探讨 GPU 架构](https://www.youtube.com/watch?v=h9Z4oGN89MU)
* [GPU 的工作原理到底是什么？](https://www.youtube.com/watch?v=58jtf24uijw&ab_channel=Graphicode)
* [面向 Python 程序员的 CUDA 入门](https://www.youtube.com/watch?v=nOxKexn3iBo&ab_channel=JeremyHoward)
* [从原子层面讲解 Transformer](https://www.youtube.com/watch?v=7lJZHbg0EQ4&ab_channel=JacobRintamaki)
* [CUDA 编程如何工作 - NVIDIA CUDA 架构师 Stephen Jones](https://www.youtube.com/watch?v=QQceTDjA4f4&ab_channel=ChristopherHollinworth)
* [使用 NVIDIA CUDA 的并行计算 - NeuralNine](https://www.youtube.com/watch?v=zSCdTOKrnII&ab_channel=NeuralNine)
* [CPU vs GPU vs TPU vs DPU vs QPU](https://www.youtube.com/watch?v=r5NQecwZs1A&ab_channel=Fireship)
* [100 秒讲清楚 CUDA 是什么](https://www.youtube.com/watch?v=pPStdjuYzSI&ab_channel=Fireship)
* [AI 如何发现更快的矩阵乘法算法](https://www.youtube.com/watch?v=fDAPJ7rvcUw&t=1s&ab_channel=QuantaMagazine)
* [最快的矩阵乘法算法](https://www.youtube.com/watch?v=sZxjuT1kUd0&ab_channel=Dr.TreforBazett)
* [从零开始：使用缓存分块进行 CUDA 矩阵乘法](https://www.youtube.com/watch?v=ga2ML1uGr5o&ab_channel=CoffeeBeforeArch)
* [从零开始：CUDA 中的矩阵乘法](https://www.youtube.com/watch?v=DpEgZe2bbU0&ab_channel=CoffeeBeforeArch)
* [GPU 编程入门](https://www.youtube.com/watch?v=G-EimI4q-TQ&ab_channel=TomNurkkala)
* [CUDA 编程](https://www.youtube.com/watch?v=xwbD6fL5qC8&ab_channel=TomNurkkala)
* [CUDA 入门（第一部分）：高级概念](https://www.youtube.com/watch?v=4APkMJdiud0&ab_channel=JoshHolloway)
* [GPU 硬件简介](https://www.youtube.com/watch?v=kUqkOAU84bA&ab_channel=TomNurkkala)

---

## 联系我

* [X（推特）](https://x.com/elliotarledge)
* [LinkedIn](https://www.linkedin.com/in/elliot-arledge-a392b7243/)
* [YouTube](https://www.youtube.com/channel/UCjlt_l6MIdxi4KoxuMjhYxg)
* [Discord](https://discord.gg/JTTcFe7Pw2)

---

如需继续了解课程内容的具体模块，我也可以帮你快速提炼每一章要点。是否需要？
