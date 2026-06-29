"""
Tree-AllReduce — 纯 Python 模拟实现
- 不依赖 GPU / MPI，用列表模拟 k 个 rank 的内存缓冲区
- 完整展示 Reduce(归约到根) + Broadcast(从根广播) 两阶段
- 采用二项树(binomial tree)拓扑，延迟 2×⌈log2 k⌉ 步
- 包含通信量统计

与 Ring-AllReduce 的对比：
    Ring : 带宽最优，通信量 2×(k-1)/k×N，但需要 2×(k-1) 步（大 k 时延迟高）
    Tree : 延迟最优，只需 2×⌈log2 k⌉ 步，但通信量 2×(k-1)×N（带宽不最优）
    => 小数据/大规模用 Tree（延迟敏感），大数据用 Ring（带宽敏感）
"""

import numpy as np
from copy import deepcopy


def tree_allreduce(tensors: list[np.ndarray], op=np.add) -> list[np.ndarray]:
    """
    Tree-AllReduce 核心实现（二项树，根为 rank 0）。

    Args:
        tensors: 长度为 k 的列表，tensors[r] 是 rank r 的输入数据
        op:      规约操作，默认求和

    Returns:
        长度为 k 的列表，每个元素都是完整的规约结果
    """
    k = len(tensors)
    N = len(tensors[0])

    # buffers[r] = rank r 当前持有的完整数据（不切块，与 Ring 不同）
    buffers = [tensors[r].copy() for r in range(k)]

    num_steps = (k - 1).bit_length()  # ⌈log2 k⌉
    total_bytes = 0  # 通信量统计（单位：elements）

    # ════════════════════════════════════════════════════════════════════
    # Phase 1: Reduce —— 自底向上把所有数据归约到 root(rank 0)
    #
    # 第 step 步（distance d = 2^step）：
    #   对每个间隔 2d 对齐的 rank r：
    #       rank r+d  发送其完整数据给 rank r
    #       rank r    把收到的数据与本地数据规约
    #   => 每步活跃 rank 数减半，⌈log2 k⌉ 步后 rank 0 持有全局规约结果
    # ════════════════════════════════════════════════════════════════════
    print("=== Phase 1: Reduce (归约到 root) ===")
    for step in range(num_steps):
        d = 1 << step
        active = []
        for r in range(0, k, 2 * d):
            src = r + d
            if src < k:
                # rank src 发送，rank r 接收并规约
                received = buffers[src].copy()
                total_bytes += N
                buffers[r] = op(buffers[r], received)
                active.append((src, r))
        _print_state(buffers, step + 1, "RD", k, active)

    print(f"\n  Reduce 完成：rank 0 持有全局规约结果")
    print(f"  此阶段通信量 = {total_bytes} elements = (k-1)×N = {k-1}×{N}\n")

    # ════════════════════════════════════════════════════════════════════
    # Phase 2: Broadcast —— 自顶向下把 root 的结果广播给所有 rank
    #
    # 第 step 步（与 Reduce 反向，distance d = 2^step）：
    #   对每个间隔 2d 对齐的 rank r：
    #       rank r  发送其完整数据给 rank r+d（直接覆盖，已是规约结果）
    #   => 每步持有结果的 rank 数翻倍
    # ════════════════════════════════════════════════════════════════════
    print("=== Phase 2: Broadcast (从 root 广播) ===")
    for step in reversed(range(num_steps)):
        d = 1 << step
        active = []
        for r in range(0, k, 2 * d):
            dst = r + d
            if dst < k:
                # rank r 发送，rank dst 直接覆盖（已是最终结果）
                buffers[dst] = buffers[r].copy()
                total_bytes += N
                active.append((r, dst))
        _print_state(buffers, num_steps - step, "BC", k, active)

    print(f"\n  Broadcast 完成：每个 rank 持有完整规约结果")

    results = [buffers[r].copy() for r in range(k)]

    print(f"\n=== 通信量分析 ===")
    print(f"  ranks k        = {k}")
    print(f"  数据量 N       = {N} elements")
    print(f"  通信步数       = 2×⌈log2 k⌉ = {2 * num_steps} 步")
    print(f"  总通信量       = {total_bytes} elements")
    print(f"  理论公式       = 2×(k-1)×N = {2 * (k - 1) * N}")
    print(f"  对比 Ring 步数  = 2×(k-1) = {2 * (k - 1)} 步（Tree 延迟更低）")

    return results


def _print_state(buffers, step, phase, k, active):
    """打印当前 tree 状态（调试可读格式）"""
    show_k = min(k, 8)
    elems = [f"{buffers[r][0]:.0f}" for r in range(show_k)]
    suffix = f" ...+{k-show_k}ranks" if k > show_k else ""
    # active: 本步发生的 (src, dst) 通信对
    pairs = ", ".join(f"{a}→{b}" for a, b in active)
    print(f"  {phase} step {step:2d}: r0..={'[' + ','.join(elems) + ']'}{suffix}  | 通信: {pairs}")


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
    K = 4          # rank 数量（Tree 对非 2 的幂也适用）
    N = 8          # 每个 rank 的数据量（Tree 不要求能被 K 整除）

    # 模拟 K 个 GPU 各自持有的梯度
    tensors = [np.random.randint(1, 10, size=N).astype(float) for _ in range(K)]

    print("输入数据：")
    for r, t in enumerate(tensors):
        print(f"  rank {r}: {t}")
    print()

    tree_results = tree_allreduce(deepcopy(tensors))
    naive_results = naive_allreduce(tensors)

    print("\n=== 验证结果 ===")
    print(f"  期望（朴素求和）: {naive_results[0]}")
    for r, res in enumerate(tree_results):
        match = np.allclose(res, naive_results[r])
        print(f"  rank {r} 结果:    {res}  ✓" if match else f"  rank {r} 结果:    {res}  ✗ MISMATCH!")
