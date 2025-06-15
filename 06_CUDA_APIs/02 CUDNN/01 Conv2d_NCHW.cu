/**
 * ============================= CUDA + cuDNN 2D 卷积基准测试 =============================
 *
 * 本程序比较 cuDNN 提供的高性能卷积实现与手写 naive CUDA kernel 的性能和数值精度。
 * 用于理解 cuDNN 卷积接口的使用方法，并验证不同算法在小尺寸输入下的差异。
 *
 * ✅ 功能概览：
 * - 构造一个 1 个通道、4x4 尺寸、batch size=1 的输入图像
 * - 使用一个 3x3 的卷积核（单输入单输出通道）
 * - 同时通过 cuDNN 和自定义 CUDA Kernel 执行卷积运算
 * - 比较二者输出的数值差异（误差）和执行时间（ms）
 * - 输出 cuDNN 算法性能评估信息和最终卷积结果
 *
 * ✅ 技术要点：
 * - cudnnSetTensor4dDescriptor / cudnnSetFilter4dDescriptor / cudnnSetConvolution2dDescriptor
 * - cudnnGetConvolutionForwardAlgorithm_v7 获取最快算法
 * - cudaEventRecord 精确计时
 * - 使用 __global__ kernel 实现 naive 卷积
 *
 * ⚠️ 注意事项：
 * - 由于是 naive kernel，性能较差，仅适用于教学和验证
 * - 若需扩展到多 batch、多通道或非方阵图像，请注意 index 计算逻辑
 *
 * 作者：ChatGPT 示例代码（可自由修改使用）
 * 日期：2025 年
 */

#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <limits>

// CUDA 和 cuDNN 错误检查宏
#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(err)); exit(1); } }
#define CHECK_CUDNN(call) { cudnnStatus_t err = call; if (err != CUDNN_STATUS_SUCCESS) { printf("cuDNN error: %s\n", cudnnGetErrorString(err)); exit(1); } }

// 多通道二维卷积的 Naive CUDA 内核实现
__global__ void naiveConv2d(float* input, float* kernel, float* output, int width, int height, int inChannels, int outChannels, int kernelSize, int batchSize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int outChannel = blockIdx.z % outChannels;
    int batchIdx = blockIdx.z / outChannels;

    if (x < width && y < height && outChannel < outChannels && batchIdx < batchSize) {
        float sum = 0.0f;
        int halfKernel = kernelSize / 2;
        for (int inChannel = 0; inChannel < inChannels; inChannel++) {
            for (int ky = -halfKernel; ky <= halfKernel; ky++) {
                for (int kx = -halfKernel; kx <= halfKernel; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;
                    if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                        int inputIdx = ((batchIdx * inChannels + inChannel) * height + iy) * width + ix;
                        int kernelIdx = ((outChannel * inChannels + inChannel) * kernelSize + (ky + halfKernel)) * kernelSize + (kx + halfKernel);
                        sum += input[inputIdx] * kernel[kernelIdx];
                    }
                }
            }
        }
        int outputIdx = ((batchIdx * outChannels + outChannel) * height + y) * width + x;
        output[outputIdx] = sum;
    }
}

int main() {
    // 设置输入图像和卷积核参数（4x4输入，3x3卷积核）
    const int width = 4;
    const int height = 4;
    const int kernelSize = 3;
    const int inChannels = 1;
    const int outChannels = 1;
    const int batchSize = 1;
    const int inputSize = width * height * inChannels * batchSize;
    const int outputSize = width * height * outChannels * batchSize;
    const int kernelElements = kernelSize * kernelSize * inChannels * outChannels;

    // 打印基本信息
    std::cout << "Image size: " << width << "x" << height << "x" << inChannels << std::endl;
    std::cout << "Kernel size: " << kernelSize << "x" << kernelSize << "x" << inChannels << "x" << outChannels << std::endl;
    std::cout << "Batch size: " << batchSize << std::endl;

    // 分配主机内存
    float* h_input = (float*)malloc(inputSize * sizeof(float));
    float* h_kernel = (float*)malloc(kernelElements * sizeof(float));
    float* h_output_cudnn = (float*)malloc(outputSize * sizeof(float));
    float* h_output_naive = (float*)malloc(outputSize * sizeof(float));

    // 初始化输入和卷积核
    float input_values[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    float kernel_values[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
    };
    memcpy(h_input, input_values, inputSize * sizeof(float));
    memcpy(h_kernel, kernel_values, kernelElements * sizeof(float));

    // 分配设备内存
    float *d_input, *d_kernel, *d_output_cudnn, *d_output_naive;
    CHECK_CUDA(cudaMalloc(&d_input, inputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_kernel, kernelElements * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_cudnn, outputSize * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output_naive, outputSize * sizeof(float)));

    // 将数据从主机复制到设备
    CHECK_CUDA(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_kernel, h_kernel, kernelElements * sizeof(float), cudaMemcpyHostToDevice));

    // cuDNN 设置
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t kernelDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&inputDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&outputDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernelDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // 设置描述符参数
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, inChannels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize, outChannels, height, width));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernelDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels, inChannels, kernelSize, kernelSize));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, kernelSize/2, kernelSize/2, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // 选择最快的 cuDNN 卷积算法
    cudnnConvolutionFwdAlgoPerf_t perfResults[CUDNN_CONVOLUTION_FWD_ALGO_COUNT];
    int returnedAlgoCount;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn, inputDesc, kernelDesc, convDesc, outputDesc,
                                                       CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returnedAlgoCount, perfResults));
    cudnnConvolutionFwdAlgo_t algo = perfResults[0].algo;
    for (int i = 1; i < returnedAlgoCount; i++) {
        std::cout << "Algorithm: " << perfResults[i].algo << " Time: " << perfResults[i].time << std::endl;
        if (perfResults[i].status == CUDNN_STATUS_SUCCESS && perfResults[i].time < perfResults[0].time) {
            algo = perfResults[i].algo;
        }
    }
    std::cout << "Selected algorithm: " << algo << std::endl;   

    // 分配 cuDNN 卷积的临时工作区
    size_t workspaceSize;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, kernelDesc, convDesc, outputDesc, algo, &workspaceSize));
    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspaceSize));

    // 定义 naive kernel 的网格和线程块大小
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y, outChannels * batchSize);

    // 热身运行
    const int warmupRuns = 5;
    const int benchmarkRuns = 20;
    float totalTime_cudnn = 0.0f;
    float totalTime_naive = 0.0f;
    float alpha = 1.0f, beta = 0.0f;

    for (int i = 0; i < warmupRuns; i++) {
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, inChannels, outChannels, kernelSize, batchSize);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // 性能测试
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < benchmarkRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha, inputDesc, d_input, kernelDesc, d_kernel, convDesc,
                                            algo, d_workspace, workspaceSize, &beta, outputDesc, d_output_cudnn));
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float milliseconds = 0;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime_cudnn += milliseconds;

        CHECK_CUDA(cudaEventRecord(start));
        naiveConv2d<<<gridSize, blockSize>>>(d_input, d_kernel, d_output_naive, width, height, inChannels, outChannels, kernelSize, batchSize);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        totalTime_naive += milliseconds;
    }

    // 打印平均运行时间
    printf("cuDNN average time: %f ms\n", totalTime_cudnn / benchmarkRuns);
    printf("Naive kernel average time: %f ms\n", totalTime_naive / benchmarkRuns);

    // 拷贝输出结果回主机
    CHECK_CUDA(cudaMemcpy(h_output_cudnn, d_output_cudnn, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output_naive, d_output_naive, outputSize * sizeof(float), cudaMemcpyDeviceToHost));

    // 比较两种实现的最大误差
    float maxDiff = 0.0f;
    for (int i = 0; i < outputSize; i++) {
        float diff = fabs(h_output_cudnn[i] - h_output_naive[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("Max difference between cuDNN and naive kernel: %e\n", maxDiff);

    // 打印输出
    printf("\ncuDNN Output:\n");
    for (int b = 0; b < batchSize; b++) {
        for (int c = 0; c < outChannels; c++) {
            printf("Channel %d:\n", c);
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * outChannels + c) * height + h) * width + w;
                    printf("%f ", h_output_cudnn[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    printf("\nNaive Kernel Output:\n");
    for (int b = 0; b < batchSize; b++) {
        for (int c = 0; c < outChannels; c++) {
            printf("Channel %d:\n", c);
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = ((b * outChannels + c) * height + h) * width + w;
                    printf("%f ", h_output_naive[idx]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }

    // 打印扁平化输出以方便和 PyTorch 对比
    printf("\nFlattened cuDNN Output:\n");
    for (int i = 0; i < outputSize; i++) {
        printf("%f", h_output_cudnn[i]);
        if (i < outputSize - 1) printf(", ");
    }
    printf("\n");

    // 释放资源
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(inputDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(outputDesc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(kernelDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_kernel));
    CHECK_CUDA(cudaFree(d_output_cudnn));
    CHECK_CUDA(cudaFree(d_output_naive));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_input);
    free(h_kernel);
    free(h_output_cudnn);
    free(h_output_naive);

    return 0;
}
