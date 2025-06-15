/**
 * ========================= CUDA + cuDNN 大尺寸卷积性能对比 =========================
 *
 * 本程序对比 cuDNN 隐式 GEMM 前向卷积算法与手写 naive CUDA kernel 的性能
 * 与数值一致性，输入尺寸为 batch=4、inChannels=32、H=W=224，卷积核 11×11，
 * 输出通道 64。
 *
 * 主要步骤：
 * 1. 随机生成输入与卷积核数据并拷贝到 GPU。
 * 2. 使用 cuDNN API（Tensor / Filter / Convolution 描述符）执行前向卷积。
 * 3. 实现 naiveConv2d __global__ 内核，逐元素卷积验证正确性。
 * 4. 热身 + 基准测试（20 次），记录 cuDNN 与 naive 的平均耗时 (ms)。
 * 5. 计算并打印两种实现的最大绝对误差，确认结果等价。
 *
 * 学习要点：
 * - cudnnSetTensor4dDescriptor / cudnnSetFilter4dDescriptor / cudnnSetConvolution2dDescriptor
 * - cudnnConvolutionForward 与 workspace 申请
 * - cudaEventRecord 计时
 * - blockIdx / threadIdx 三维调度实现多 Batch、多通道卷积
 *
 * 作者：ChatGPT 示例
 * 日期：2025 年
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>
#include <ctime>

// ---------------------------- 宏：错误检查 ----------------------------
#define CHECK_CUDA(call)  { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(EXIT_FAILURE); } }

// ---------------------------- Naive CUDA 卷积内核 ----------------------------
__global__ void naiveConv2d(float* input, float* kernel, float* output,
                            int width, int height,
                            int inChannels, int outChannels,
                            int kernelSize, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 输出特征图 x 坐标
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 输出特征图 y 坐标
    int outChannel = blockIdx.z % outChannels;      // 输出通道 idx
    int batchIdx   = blockIdx.z / outChannels;      // Batch idx

    if (x < width && y < height && outChannel < outChannels && batchIdx < batchSize) {
        float sum = 0.0f;
        int halfK = kernelSize / 2;
        // 逐输入通道累加
        for (int inChannel = 0; inChannel < inChannels; ++inChannel) {
            for (int ky = -halfK; ky <= halfK; ++ky) {
                for (int kx = -halfK; kx <= halfK; ++kx) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int inIdx = ((batchIdx * inChannels + inChannel) * height + iy) * width + ix;
                        int kIdx  = ((outChannel * inChannels + inChannel) * kernelSize + (ky + halfK)) * kernelSize + (kx + halfK);
                        sum += input[inIdx] * kernel[kIdx];
                    }
                }
            }
        }
        int outIdx = ((batchIdx * outChannels + outChannel) * height + y) * width + x;
        output[outIdx] = sum;
    }
}

int main() {
    // ---------------------------- 输入/卷积核尺寸 ----------------------------
    const int width       = 224;
    const int height      = 224;
    const int kernelSize  = 11;
    const int inChannels  = 32;
    const int outChannels = 64;
    const int batchSize   = 4;

    const int inputSize   = width * height * inChannels * batchSize;
    const int outputSize  = width * height * outChannels * batchSize;
    const int kernelElems = kernelSize * kernelSize * inChannels * outChannels;

    std::cout << "Image: "  << batchSize << "×" << inChannels << "×" << height << "×" << width << std::endl;
    std::cout << "Kernel: " << outChannels << "×" << inChannels << "×" << kernelSize << "×" << kernelSize << std::endl;

    // ---------------------------- 分配主机内存并随机初始化 ----------------------------
    float* h_input        = (float*)malloc(inputSize  * sizeof(float));
    float* h_kernel       = (float*)malloc(kernelElems * sizeof(float));
    float* h_out_cudnn    = (float*)malloc(outputSize * sizeof(float));
    float* h_out_naive    = (float*)malloc(outputSize * sizeof(float));

    srand(static_cast<unsigned>(time(nullptr)));
    for (int i = 0; i < inputSize;  ++i) h_input[i]  = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < kernelElems; ++i) h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;

    // ---------------------------- 分配设备内存并拷贝数据 ----------------------------
    float *d_input, *d_kernel, *d_out_cudnn, *d_out_naive;
    CHECK_CUDA(cudaMalloc(&d_input,     inputSize  * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel,    kernelElems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_cudnn, outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out_naive, outputSize * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input,  h_input,  inputSize  * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelElems * sizeof(float), cudaMemcpyHostToDevice));

    // ---------------------------- cuDNN 初始化与描述符 ----------------------------
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t  inputDesc, outputDesc;
    cudnnFilterDescriptor_t  filterDesc;
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filterDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, inChannels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batchSize, outChannels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           outChannels, inChannels, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                                kernelSize/2, kernelSize/2,  // padding
                                                1, 1,                        // stride
                                                1, 1,                        // dilation
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 选择默认算法：implicit GEMM（避免额外查询耗时）
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // workspace 大小查询 & 分配
    size_t workspaceSize = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, algo, &workspaceSize));
    void* d_workspace = nullptr;
    if (workspaceSize > 0) CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // ---------------------------- CUDA kernel 调度参数 ----------------------------
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y,
              outChannels * batchSize);

    // ---------------------------- 热身 + 基准测试 ----------------------------
    const int warmupRuns    = 5;
    const int benchmarkRuns = 20;
    float total_cudnn = 0.0f, total_naive = 0.0f;
    float alpha = 1.0f, beta = 0.0f;

    // 热身
    for (int i = 0; i < warmupRuns; ++i) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_kernel,
                                            convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, d_out_cudnn));
        naiveConv2d<<<grid, block>>>(d_input, d_kernel, d_out_naive, width, height, inChannels, outChannels, kernelSize, batchSize);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 基准测试
    for (int i = 0; i < benchmarkRuns; ++i) {
        // cuDNN
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, filterDesc, d_kernel,
                                            convDesc, algo, d_workspace, workspaceSize, &beta, outputDesc, d_out_cudnn));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_cudnn += ms;

        // Naive kernel
        CHECK_CUDA(cudaEventRecord(start));
        naiveConv2d<<<grid, block>>>(d_input, d_kernel, d_out_naive, width, height, inChannels, outChannels, kernelSize, batchSize);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_naive += ms;
    }

    std::cout << "cuDNN 平均耗时: "  << total_cudnn / benchmarkRuns << " ms" << std::endl;
    std::cout << "Naive 平均耗时: "  << total_naive / benchmarkRuns << " ms" << std::endl;

    // ---------------------------- 结果验证 ----------------------------
    CHECK_CUDA(cudaMemcpy(h_out_cudnn, d_out_cudnn, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_naive, d_out_naive, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    float maxDiff = 0.f;
    for (int i = 0; i < outputSize; ++i) {
        float diff = fabs(h_out_cudnn[i] - h_out_naive[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    std::cout << "最大绝对误差: " << maxDiff << std::endl;

    // ---------------------------- 资源释放 ----------------------------
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filterDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_out_cudnn));
    CHECK_CUDA(cudaFree(d_out_naive));
    if (workspaceSize > 0) CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_input);
    free(h_kernel);
    free(h_out_cudnn);
    free(h_out_naive);

    return 0;
}
