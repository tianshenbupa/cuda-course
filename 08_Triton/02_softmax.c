#include <stdio.h>
#include <math.h>

/*
===========================================
Softmax 函数的 C 语言实现（含数值稳定性优化）

功能：
- 对一个长度为 n 的 float 数组 x[] 进行 softmax 归一化处理
- 包含数值稳定性处理（减去最大值 max），避免浮点溢出
- 示例输入：x = [1.0, 2.0, 3.0]
- 经过 softmax 后输出约为：[0.0900, 0.2447, 0.6652]

softmax 公式：
    softmax(x_i) = exp(x_i) / sum(exp(x_j))

数值稳定性技巧：
    为防止溢出，先减去 max(x)，即：
    softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
===========================================
*/

void softmax(float *x, int n) {
    float max = x[0];

    // 第一步：找到数组中的最大值（用于数值稳定）
    for (int i = 1; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }

    float sum = 0.0;

    // 第二步：将每个元素减去最大值后进行 exp 计算
    for (int i = 0; i < n; i++) {
        x[i] = exp(x[i] - max);  // 减 max 是为了防止 exp 溢出
        sum += x[i];             // 同时累加求和
    }

    // 第三步：将每个值除以总和，得到最终的概率值
    for (int i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

int main() {
    // 示例输入数组
    float x[] = {1.0, 2.0, 3.0};

    // 调用 softmax 函数
    softmax(x, 3);

    // 输出结果
    for (int i = 0; i < 3; i++) {
        printf("%f\n", x[i]);
    }

    return 0;
}
