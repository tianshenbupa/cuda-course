以下内容为你提供的 **CUDA API 相关笔记** 的中文翻译与整理（保留示例代码和格式），方便阅读与后续查阅。

---

## 🚀 CUDA API（含 cuBLAS、cuDNN、cuBLASmp）

> **API** 这个词一开始可能让人困惑。简单来说，它指的是一个**库**：
>
> * 库的内部实现（源码）对用户是**不可见**的，只暴露经过高度优化并已编译好的二进制；
> * 官方文档只告诉你有哪些函数、参数该如何传，但看不到内部细节。
>   这条原则适用于本文列出的所有 CUDA 库/API。

---

## 🔒 Opaque Struct Types（不透明结构体）

* **不透明**：你无法访问或修改结构体内部成员，只能使用它的“句柄”来调用 API。
* 在 Linux 上，这些实现通常以 **`.so`（共享对象）** 文件形式分发；Windows 对应为 `.dll`。
* 例子：`cublasLtHandle_t` 表示 cuBLAS Lt 操作上下文的句柄——只能拿来传参，内部细节对外隐藏。
* 你会发现 cuFFT、cuDNN 等 CUDA 库都会以同样的“API + 句柄”模式出现。

---

## 🛠️ 快速掌握 CUDA API 的实用技巧

1. **perplexity.ai** —— 实时抓取最新资料，搜索体验友好
2. **Google** —— 传统方式，检索更广泛
3. **ChatGPT** —— 查询通用概念，避免过时信息
4. **NVIDIA 官方文档关键字搜索** —— 最权威、最详细

---

## ⚠️ 错误检查（以 cuBLAS / cuDNN 为例）

### cuBLAS 示例

```cpp
#define CUBLAS_CHECK(call)                       \
    do {                                         \
        cublasStatus_t status = (call);          \
        if (status != CUBLAS_STATUS_SUCCESS) {   \
            fprintf(stderr,                      \
                    "cuBLAS error at %s:%d: %d\\n",\
                    __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                  \
        }                                        \
    } while (0)
```

### cuDNN 示例

```cpp
#define CUDNN_CHECK(call)                                         \
    do {                                                          \
        cudnnStatus_t status = (call);                            \
        if (status != CUDNN_STATUS_SUCCESS) {                     \
            fprintf(stderr,                                       \
                    "cuDNN error at %s:%d: %s\\n",                \
                    __FILE__, __LINE__,                           \
                    cudnnGetErrorString(status));                 \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)
```

**为什么要做错误检查？**

1. 先配置好 API 的上下文（handle、stream、tensor 描述符等）；
2. 调用运算函数；
3. 立即检查返回状态：

   * 成功 → 正常继续
   * 失败 → 打印可读错误信息，避免因段错误或静默错误而难以排查

---

## 📐 矩阵乘法（Matmul）

* **cuDNN** 自带卷积、RNN 等深度学习算子，其中隐含 Matmul，但 Matmul 并非其核心接口。
* **cuBLAS** 提供最全面、最成熟且高吞吐的矩阵乘法实现（推荐首选）。
* 有了 cuBLAS 的经验后，迁移到 cuDNN、cuFFT 等其他库的“描述符‑>配置‑>调用”流程并不难。

---

## 📚 资源

* **CUDA Library Samples**（官方示例代码集合）：
  [https://github.com/NVIDIA/CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples)

---

如果你需要进一步说明如何选择特定库、如何排查不同 API 的错误，或想看更多示例（如 cuBLASmp、cuFFT 调用范例），随时告诉我！
