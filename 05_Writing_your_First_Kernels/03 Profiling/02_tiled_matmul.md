

```cpp
#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16  // 定义线程块大小（Tile 尺寸）

// 优化的矩阵乘法核函数，使用共享内存
__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int M, int N, int K) {
    // 在共享内存中为 A 和 B 分配 tile 空间
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    // 获取当前线程所在的 block 索引
    int bx = blockIdx.x, by = blockIdx.y;
    // 获取线程在 block 中的局部索引
    int tx = threadIdx.x, ty = threadIdx.y;

    // 计算当前线程对应的全局行和列索引
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;  // 累加器初始化

    // 按 tile 进行遍历计算
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // 加载 A 的 tile 到共享内存
        if (row < M && tile * TILE_SIZE + tx < K)
            sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx];
        else
            sharedA[ty][tx] = 0.0f;

        // 加载 B 的 tile 到共享内存
        if (col < N && tile * TILE_SIZE + ty < K)
            sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();  // 所有线程同步，确保共享内存数据加载完毕

        // 执行小矩阵的乘加操作
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];

        __syncthreads();  // 同步以准备加载下一 tile
    }

    // 将结果写入输出矩阵 C
    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    // 定义矩阵维度
    const int M = 1024; // A 和 C 的行数
    const int N = 1024; // B 和 C 的列数
    const int K = 1024; // A 的列数，B 的行数

    // 计算每个矩阵的字节大小
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 声明设备指针
    float *d_A, *d_B, *d_C;

    // 在 GPU 上分配内存
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // 配置线程块和网格的维度
    dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 每个 block 16x16 个线程
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE); // 确保覆盖整个矩阵

    // 启动矩阵乘法核函数
    matrixMultiplyOptimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

    // 等待 GPU 完成操作
    cudaDeviceSynchronize();

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 检查 CUDA 执行过程中是否有错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}
```

### 总结：

这个程序实现了一个 **使用共享内存优化的矩阵乘法核函数**，通过分块（Tile）和线程块并行提高计算效率。适用于大矩阵计算场景。下一步可以加入 host 数据初始化与拷贝，或者进行性能测试（如使用 `cudaEvent_t` 计时）来进一步验证优化效果。需要可以继续展开。

---
---
这段代码是一个使用 CUDA 实现 **矩阵乘法（C = A × B）** 的程序，并在 kernel 中利用了 **共享内存（shared memory）** 进行优化。下面是各部分的分析：

---

## 🔧 1. 定义常量与核函数结构

```cpp
#define TILE_SIZE 16
```

* 设置每个线程块的维度为 `16x16`，也就是一个 block 中有 256 个线程。
* 每个线程负责计算输出矩阵 `C` 中的一个元素。

---

## 🚀 2. 核函数 `matrixMultiplyOptimized`

### 输入说明：

```cpp
__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int M, int N, int K)
```

* `A` 是维度 `MxK` 的矩阵；
* `B` 是维度 `KxN` 的矩阵；
* `C` 是最终结果矩阵，维度 `MxN`；
* `M`, `N`, `K` 是矩阵的维度。

### 核心优化点：

```cpp
__shared__ float sharedA[TILE_SIZE][TILE_SIZE];
__shared__ float sharedB[TILE_SIZE][TILE_SIZE];
```

* 利用共享内存将 A 和 B 的某个子块（tile）载入，加速后续重复使用。

### 块与线程索引计算：

```cpp
int bx = blockIdx.x, by = blockIdx.y;
int tx = threadIdx.x, ty = threadIdx.y;
int row = by * TILE_SIZE + ty;
int col = bx * TILE_SIZE + tx;
```

* 计算当前线程负责计算 `C[row][col]`。

### 分 tile 进行乘加：

```cpp
for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
    ...
}
```

* 对于列数为 `K` 的中间维度，分块处理，每次处理 `TILE_SIZE` 列。
* **每轮循环：加载 A 和 B 的子块到共享内存，进行小块乘法累加。**

### 加载数据：

```cpp
sharedA[ty][tx] = A[row * K + tile * TILE_SIZE + tx]; // 加载 A
sharedB[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col]; // 加载 B
```

* 注意这里的索引是行优先存储格式（row-major）。
* 访问越界时设为 0，避免非法访问。

### 执行小 tile 的乘加：

```cpp
for (int k = 0; k < TILE_SIZE; ++k)
    sum += sharedA[ty][k] * sharedB[k][tx];
```

* 这是 tile 乘法的核心，每个线程完成一个 C 元素的乘法-加法。

### 写入结果：

```cpp
if (row < M && col < N)
    C[row * N + col] = sum;
```

* 写回最终结果矩阵。

---

## 🧠 3. 主函数 main 分析

### 矩阵维度定义：

```cpp
const int M = 1024, N = 1024, K = 1024;
```

* 使用固定的 `1024x1024` 矩阵，符合 GPU 并行场景。

### 内存分配：

```cpp
cudaMalloc(&d_A, size_A); ...
```

* 分别在设备上分配 A、B、C 的显存。

### 启动核函数：

```cpp
dim3 blockDim(TILE_SIZE, TILE_SIZE);
dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
```

* 以 tile 为单位划分整个计算网格，确保覆盖所有输出元素。

### CUDA 同步与错误检查：

```cpp
cudaDeviceSynchronize();
cudaError_t error = cudaGetLastError();
```

* 等待 kernel 执行完毕，并检查是否有错误发生。

---

## ✅ 总结要点

| 模块          | 说明                                               |
| ----------- | ------------------------------------------------ |
| **共享内存优化**  | 将频繁访问的子块缓存进 shared memory，避免重复从 global memory 读。 |
| **线程并行策略**  | 每个线程负责计算输出矩阵中的一个元素                               |
| **Tile 分块** | 使用 tile-based 方式处理中间维度（K），适用于大规模矩阵乘法             |
| **边界判断**    | 处理 `M`, `N`, `K` 不能整除 tile 大小时的边界问题              |
| **尚未包括部分**  | 代码中没有主机内存分配、数据初始化、结果拷回与验证、计时等逻辑                  |

---





