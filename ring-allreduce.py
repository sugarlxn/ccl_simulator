"""
Ring-AllReduce — 纯 Python 模拟实现
- 不依赖 GPU / MPI，用列表模拟 k 个 rank 的内存缓冲区
- 完整展示 ReduceScatter + AllGather 两阶段
- 包含通信量统计
"""

import numpy as np
from copy import deepcopy

def ring_allreduce(tensors: list[np.ndarray], op=np.add) -> list[np.ndarray]:
    """
    Ring-AllReduce 核心实现。

    Args:
        tensors: 长度为 k 的列表，tensors[r] 是 rank r 的输入数据
        op:      规约操作，默认求和

    Returns:
        长度为 k 的列表，每个元素都是完整的规约结果
    """
    k = len(tensors)
    N = len(tensors[0])
    assert N % k == 0, f"数据长度 {N} 必须能被 rank 数 {k} 整除"
    chunk_size = N // k

    # ── 初始化：每个 rank 把自己的数据切成 k 块 ──────────────────────────
    # buffers[r][c] = rank r 持有的第 c 块数据
    buffers = [
        [tensors[r][c * chunk_size:(c + 1) * chunk_size].copy() for c in range(k)]
        for r in range(k)
    ]

    total_bytes = 0  # 通信量统计（单位：elements）

    # ════════════════════════════════════════════════════════════════════
    # Phase 1: ReduceScatter — k-1 步
    #
    # 第 step 步：
    #   rank r  发送 chunk[(r - step) mod k] 给 rank[(r+1) mod k]
    #   rank r  接收来自 rank[(r-1) mod k] 的 chunk[(r - step - 1) mod k]
    #   然后把接收到的 chunk 与本地对应 chunk 做规约
    # ════════════════════════════════════════════════════════════════════
    print("=== Phase 1: ReduceScatter ===")
    for step in range(k - 1):
        # 同时模拟所有 rank 的发送/接收（在真实系统中这些是并发的）
        sends = {}
        for r in range(k):
            send_chunk_idx = (r - step) % k
            dst = (r + 1) % k
            sends[(r, dst)] = buffers[r][send_chunk_idx].copy()
            total_bytes += chunk_size

        # 接收并规约
        for r in range(k):
            src = (r - 1) % k
            recv_chunk_idx = (r - step - 1) % k
            received = sends[(src, r)]
            buffers[r][recv_chunk_idx] = op(buffers[r][recv_chunk_idx], received)

        # 打印当前状态（仅前 4 个 rank，最多显示 3 个元素）
        _print_state(buffers, step + 1, "RS", k, chunk_size)

    print(f"\n  ReduceScatter 完成：每个 rank 持有 1 块完整规约结果")
    print(f"  通信量 = {total_bytes} elements = {k-1} 步 × {k} ranks × {chunk_size} elems/chunk\n")

    # ════════════════════════════════════════════════════════════════════
    # Phase 2: AllGather — k-1 步
    #
    # 第 step 步：
    #   rank r  发送 chunk[(r - step + 1) mod k] 给 rank[(r+1) mod k]
    #   rank r  接收来自 rank[(r-1) mod k] 的 chunk 并直接覆盖（已完整，无需规约）
    # ════════════════════════════════════════════════════════════════════
    print("=== Phase 2: AllGather ===")
    for step in range(k - 1):
        sends = {}
        for r in range(k):
            send_chunk_idx = (r - step + 1) % k
            dst = (r + 1) % k
            sends[(r, dst)] = buffers[r][send_chunk_idx].copy()
            total_bytes += chunk_size

        for r in range(k):
            src = (r - 1) % k
            recv_chunk_idx = (r - step) % k
            buffers[r][recv_chunk_idx] = sends[(src, r)]  # 直接覆盖，已是规约结果

        _print_state(buffers, step + 1, "AG", k, chunk_size)

    print(f"\n  AllGather 完成：每个 rank 持有完整规约结果")

    # 拼接每个 rank 的 k 块，得到最终结果
    results = [np.concatenate(buffers[r]) for r in range(k)]

    print(f"\n=== 通信量分析 ===")
    print(f"  ranks k       = {k}")
    print(f"   数据量 N      = {N} elements")
    print(f"  总通信量       = {total_bytes} elements")
    print(f"  理论公式       = 2×(k-1)/k×N = {2*(k-1)/k*N:.1f}")
    print(f"  利用率         = {total_bytes / (2 * N) * 100:.1f}%（相对于 Naive 2N）")

    return results


def _print_state(buffers, step, phase, k, chunk_size):
    """打印当前 ring 状态（调试可读格式）"""
    show_k = min(k, 4)
    parts = []
    for r in range(show_k):
        # 每块只显示第一个元素，标记是否已完整规约
        elems = [f"{buffers[r][c][0]:.0f}" for c in range(k)]
        parts.append(f"r{r}:[{','.join(elems)}]")
    suffix = f"...+{k-show_k}ranks" if k > show_k else ""
    print(f"  step {step:2d}: {' | '.join(parts)}{suffix}")


# ══════════════════════════════════════════════════════════════════════
# 验证：与 numpy 朴素实现对比
# ══════════════════════════════════════════════════════════════════════
def naive_allreduce(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """朴素实现：直接全局求和，用于验证正确性"""
    result = np.zeros_like(tensors[0])
    for t in tensors:
        result += t
    return [result.copy() for _ in tensors]


if __name__ == "__main__":
    np.random.seed(42)
    K = 4          # rank 数量
    N = 8          # 每个 rank 的数据量（必须能被 K 整除）

    # 模拟 K 个 GPU 各自持有的梯度
    tensors = [np.random.randint(1, 10, size=N).astype(float) for _ in range(K)]

    print("输入数据：")
    for r, t in enumerate(tensors):
        print(f"  rank {r}: {t}")
    print()

    ring_results = ring_allreduce(deepcopy(tensors))
    naive_results = naive_allreduce(tensors)

    print("\n=== 验证结果 ===")
    print(f"  期望（朴素求和）: {naive_results[0]}")
    for r, res in enumerate(ring_results):
        match = np.allclose(res, naive_results[r])
        print(f"  rank {r} 结果:    {res}  ✓" if match else f"  rank {r} 结果:    {res}  ✗ MISMATCH!")