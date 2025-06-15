import torch
import time
import math

# ========================= 常量定义 =========================
N = 1 << 19  # 元素数量（8192）
WARMUP_RUNS = 10  # 预热迭代次数
BENCHMARK_RUNS = 100  # 基准测试迭代次数
BATCH_SIZE = 256  # 批大小（当前脚本未直接用到，可根据场景调整）

# ========================= 自定义 tanh 实现 =========================

def custom_tanh(x: torch.Tensor) -> torch.Tensor:
    """使用公式 (e^{2x}-1)/(e^{2x}+1) 计算 tanh。"""
    return (torch.exp(2 * x) - 1) / (torch.exp(2 * x) + 1)

# ========================= 基准测试函数：自定义 tanh =========================

def benchmark_custom_tanh(input_tensor: torch.Tensor) -> torch.Tensor:
    # 预热，以消除首次 kernel 启动带来的时间偏差
    for _ in range(WARMUP_RUNS):
        _ = custom_tanh(input_tensor)

    torch.cuda.synchronize()  # 等待 GPU 完成所有任务

    # 正式计时
    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        _ = custom_tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) * 1000 / BENCHMARK_RUNS  # 转为毫秒
    print(f"自定义 tanh：平均耗时 {avg_time:.3f} ms")

    return custom_tanh(input_tensor)

# ========================= 基准测试函数：Torch 内置 tanh =========================

def benchmark_builtin_tanh(input_tensor: torch.Tensor) -> torch.Tensor:
    # 预热
    for _ in range(WARMUP_RUNS):
        _ = torch.tanh(input_tensor)

    torch.cuda.synchronize()

    # 正式计时
    start = time.perf_counter()
    for _ in range(BENCHMARK_RUNS):
        _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time = (end - start) * 1000 / BENCHMARK_RUNS
    print(f"Torch 内置 tanh：平均耗时 {avg_time:.3f} ms")

    return torch.tanh(input_tensor)

# ========================= 结果验证函数 =========================

def verify_outputs(custom_output: torch.Tensor, builtin_output: torch.Tensor) -> None:
    """计算并打印两种实现之间的最大绝对误差。"""
    max_diff = torch.max(torch.abs(custom_output - builtin_output)).item()
    print(f"自定义与内置结果最大差值: {max_diff:.6e}")

# ========================= 主函数 =========================

def main() -> None:
    # 选择计算设备（优先使用 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 生成范围在 [-1, 1] 的随机输入张量
    input_tensor = torch.rand((128, 32, 224, 224), device=device) * 2 - 1

    # 触发一次内置 tanh，初始化 CUDA 运行时，减少后续计时抖动
    _ = torch.tanh(input_tensor)
    torch.cuda.synchronize()

    # 基准测试自定义 tanh 实现
    custom_output = benchmark_custom_tanh(input_tensor)

    # 基准测试 Torch 内置 tanh
    builtin_output = benchmark_builtin_tanh(input_tensor)

    # 验证两种实现结果一致性
    verify_outputs(custom_output, builtin_output)

if __name__ == "__main__":
    main()
