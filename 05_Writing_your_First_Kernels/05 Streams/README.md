# CUDA 流（Streams）示例

## 直观理解

可以把 **Stream** 想象成一条“河流”，流水只会沿着时间轴向前流动。例如：
1 → 把数据拷贝到 GPU（时间点 1）
2 → 在 GPU 上做计算（时间点 2）
3 → 再把数据拷回主机（时间点 3）

这就是最基本的 Stream 思路。
CUDA 允许同时拥有多条 Stream，每条 Stream 都有自己的时间线，从而让不同操作相互重叠、提高 GPU 利用率。

在训练超大规模语言模型时，如果一直在等待数据装载／传回 GPU，会非常低效。Stream 可以在 **计算** 的同时 **提前搬运** 下一批数据（俗称 *prefetching*），把数据传输延迟“隐藏”起来。

本项目演示如何使用 CUDA Streams 做并发执行、提升 GPU 吞吐，共包含两个示例：

---

## 代码片段

### 默认 Stream（0 号或空 Stream）

```cpp
// 使用空流（null stream，编号 0）启动 kernel
myKernel<<<gridSize, blockSize>>>(args);

// 等价写法（显式给出第四个参数 0）
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

回忆一下 Kernel 启动语法 `<<<Dg, Db, Ns, S>>>` 中四个参数含义：

| 参数                    | 说明                  |
| --------------------- | ------------------- |
| **Dg (dim3)**         | 网格维度与大小             |
| **Db (dim3)**         | 每个线程块维度与大小          |
| **Ns (size\_t)**      | 每块 *动态共享内存* 字节数，常省略 |
| **S (cudaStream\_t)** | 所属 Stream，可省略；默认 0  |

### 创建不同优先级的 Stream

```cpp
// 创建两个优先级不同的 Stream，让运行次序更可控
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority);   // 低优先级
cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority); // 高优先级
```

---

## 示例文件

1. **`stream_basics.cu`** – 展示异步拷贝 + Kernel 启动的基本用法
2. **`stream_advanced.cu`** – 演示优先级、回调（Callback）、跨流事件依赖等高级特性

---

## 编译命令

```bash
nvcc -o 01 01_stream_basics.cu
nvcc -o 02 02_stream_advanced.cu
```

---

## 参考资料

* NVIDIA 官方 Webinar —— *Streams and Concurrency*
  [https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)

---

## 锁页内存（Pinned Memory）

* 可以把它理解为“这块内存以后还要用，先钉在这别动”。
* 锁页内存不会被操作系统随意换页，对异步传输 **Host ↔ Device** 带宽提升显著，避免 GPU 访问到被换走的物理页导致崩溃。

```cpp
// 分配锁页内存
float* h_data;
cudaMallocHost((void**)&h_data, size);
```

---

## 事件（Events）

* **测量 Kernel 耗时**：在 Kernel 前后放置事件，精确统计执行时间。
* **跨流同步**：让一条 Stream 等待另一条 Stream 完成特定事件。
* **计算与传输重叠**：传输完成触发事件，再启动后续计算。

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(args);
cudaEventRecord(stop, stream);

cudaEventSynchronize(stop);
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

---

## 回调（Callbacks）

* 当某条 Stream 中排队的所有操作完成后，回调会在 **主机线程** 被触发。
* 典型用法：GPU 完成一批任务 → CPU 回调被唤醒 → 预处理下一批数据 → 再次提交到 GPU，形成 **流水线**。

```cpp
void CUDART_CB MyCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("GPU operation completed\\n");
    // 这里可以启动下一批 GPU 工作
}

kernel<<<grid, block, 0, stream>>>(args);
cudaStreamAddCallback(stream, MyCallback, nullptr, 0);
```

---




以下是对你提供的 CUDA Streams 学习资料的**总结与分析**：

---

## 🧠 一、总结

### 1. **CUDA Streams 的作用**

CUDA Streams 是一种 **异步任务队列**，用于让 GPU 实现 **数据拷贝和计算的并发执行**，提高设备利用率，减少空闲等待时间。

* **默认 Stream（编号0）**：串行执行，一项完成后才会进行下一项。
* **用户自定义 Streams**：多个任务流可以并发，适合大模型、视频处理等高吞吐任务。

### 2. **关键组件**

| 组件                | 作用                | 示例用法                                        |
| ----------------- | ----------------- | ------------------------------------------- |
| **Stream**        | 并发任务队列            | `cudaStreamCreate()`                        |
| **Pinned Memory** | 锁页内存，加快传输         | `cudaMallocHost()`                          |
| **Event**         | 事件记录和同步           | `cudaEventRecord()`、`cudaStreamWaitEvent()` |
| **Callback**      | 回调函数（GPU完成后CPU响应） | `cudaStreamAddCallback()`                   |

---

## 🔍 二、深入分析

### ✅ 优势

| 优势点        | 原因与效果                |
| ---------- | -------------------- |
| **提升性能**   | 异步内存拷贝 + 并行计算，隐藏传输延迟 |
| **更细粒度控制** | 支持跨流依赖、优先级调度、回调逻辑    |
| **节省时间**   | 在计算还未完成时即可加载下一批数据    |

### 💡 实战应用场景

| 场景           | 说明                           |
| ------------ | ---------------------------- |
| **训练大模型**    | 在计算当前 batch 的同时传输下一 batch 数据 |
| **图像/视频批处理** | 用多 Stream 并发多个帧的处理，提高吞吐      |
| **多任务调度**    | 高优任务绑定高优先级 Stream，实现调度控制     |

### ⚠️ 注意事项

1. **Stream 间操作不会自动同步**，必须通过 `cudaStreamSynchronize` 或 `cudaEvent` 手动控制。
2. **同一个设备指针并行访问要避免写冲突**（不同 stream 写同一个地址会产生数据竞争）。
3. **Pinned Memory 数量过多会影响系统性能**，建议按需分配。

---

## 🧩 三、与其他并发机制对比

| 技术                       | 并发类型       | 特点                  |
| ------------------------ | ---------- | ------------------- |
| **Streams**              | GPU 内部异步调度 | 主流 CUDA 并发机制        |
| **CUDA Graphs**          | 图模型批量调度    | 优化 launch 开销，适合重复计算 |
| **多线程 CPU 控制多个 Streams** | 异构并发       | 常用于 pipeline 推理系统   |

---

## 🧪 四、推荐练习方式

1. **从零实现一个异步矩阵加法 demo**（异步 copy + kernel）
2. **加入 `Event` 来控制 kernel 的执行顺序**
3. **加入两个 `Stream` 并做跨流 `WaitEvent` 控制**
4. **添加 `Callback`，让 GPU 计算结束后在 CPU 打印信息或预处理下一批数据**

---

如需进一步学习建议或完整样例项目，可继续问我。你也可以结合你自己的推荐系统 / LLM 推理任务来实际应用这些流技术。

