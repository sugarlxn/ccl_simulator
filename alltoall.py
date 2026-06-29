"""
All-to-All — 纯 Python 模拟实现
- 不依赖 GPU / MPI，用列表模拟 k 个 rank 的内存缓冲区
- 完整展示「转置式」数据交换：每个 rank 把第 c 块发给 rank c
- 包含通信量统计

语义（与 AllReduce 完全不同，不做规约）：
    输入：rank r 持有 k 块数据 in[r][0..k-1]，其中 in[r][c] 是 r 准备发给 c 的数据
    输出：rank r 持有 k 块数据 out[r][0..k-1]，其中 out[r][s] = in[s][r]
    => 相当于把 k×k 的「块矩阵」做转置，是 ReduceScatter/AllGather 的底层原语之一

与 AllReduce 的对比：
    AllReduce : 输出每个 rank 都相同（全局规约结果）
    All-to-All: 输出每个 rank 各不相同（个性化数据分发），无规约操作
    通信量    : 每个 rank 发出 (k-1) 块、收到 (k-1) 块（本地块无需通信）
"""

import numpy as np
from copy import deepcopy


def all_to_all(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """
    All-to-All 核心实现。

    Args:
        tensors: 长度为 k 的列表，tensors[r] 是 rank r 的输入数据；
                 每个 rank 的数据被均分成 k 块，第 c 块发往 rank c

    Returns:
        长度为 k 的列表，results[r] 是 rank r 收集到的数据，
        其中第 s 块来自 rank s 发给 r 的那一块
    """
    k = len(tensors)
    N = len(tensors[0])
    assert N % k == 0, f"数据长度 {N} 必须能被 rank 数 {k} 整除"
    chunk_size = N // k

    # ── 初始化：每个 rank 把自己的数据切成 k 块 ──────────────────────────
    # in_buf[r][c] = rank r 准备发给 rank c 的第 c 块
    in_buf = [
        [tensors[r][c * chunk_size:(c + 1) * chunk_size].copy() for c in range(k)]
        for r in range(k)
    ]

    # out_buf[r][s] = rank r 从 rank s 收到的块
    out_buf = [[None] * k for _ in range(k)]

    total_bytes = 0  # 通信量统计（单位：elements），仅统计跨 rank 通信

    # ════════════════════════════════════════════════════════════════════
    # 数据交换 —— 块矩阵转置
    #
    #   对所有 (r, c)：rank r 把 in_buf[r][c] 发给 rank c
    #   在接收端：rank c 的第 r 块 out_buf[c][r] = in_buf[r][c]
    #   => out[c][r] = in[r][c]，即 in 块矩阵的转置
    #
    # 实现上分 k-1 步轮转（与 Ring 同构）：第 step 步 rank r 与 rank
    # (r+step) 交换各自的目标块，对角块(本地块, step=0)直接拷贝不计通信量。
    # ════════════════════════════════════════════════════════════════════
    print("=== All-to-All: 块交换（转置）===")
    for step in range(k):
        for r in range(k):
            partner = (r + step) % k
            if step == 0:
                # 本地块无需通信：rank r 发给自己的第 r 块
                out_buf[r][r] = in_buf[r][r].copy()
            else:
                # rank r 把发给 partner 的块送出；同时收下 partner 发给自己的块
                out_buf[partner][r] = in_buf[r][partner].copy()
                total_bytes += chunk_size
        _print_state(out_buf, step, k)

    print(f"\n  All-to-All 完成：每个 rank 持有来自所有 rank 的个性化数据")

    # 拼接每个 rank 的 k 块，得到最终结果
    results = [np.concatenate(out_buf[r]) for r in range(k)]

    print(f"\n=== 通信量分析 ===")
    print(f"  ranks k        = {k}")
    print(f"  数据量 N       = {N} elements")
    print(f"  块大小         = {chunk_size} elements/chunk")
    print(f"  总通信量       = {total_bytes} elements")
    print(f"  理论公式       = (k-1)/k×N×k = (k-1)×{chunk_size}×{k} = {(k - 1) * chunk_size * k}")
    print(f"  （每个 rank 发出/收到 k-1={k-1} 块，本地块不计）")

    return results


def _print_state(out_buf, step, k):
    """打印当前 All-to-All 状态（调试可读格式）"""
    show_k = min(k, 4)
    parts = []
    for r in range(show_k):
        # 每块只显示第一个元素，未收到的块显示 '_'
        elems = [
            f"{out_buf[r][c][0]:.0f}" if out_buf[r][c] is not None else "_"
            for c in range(k)
        ]
        parts.append(f"r{r}:[{','.join(elems)}]")
    suffix = f"...+{k-show_k}ranks" if k > show_k else ""
    tag = "本地" if step == 0 else f"+{step}"
    print(f"  step {step:2d} ({tag:>4s}): {' | '.join(parts)}{suffix}")


# ══════════════════════════════════════════════════════════════════════
# 验证：与朴素转置实现对比
# ══════════════════════════════════════════════════════════════════════
def naive_all_to_all(tensors: list[np.ndarray]) -> list[np.ndarray]:
    """朴素实现：直接按定义做块矩阵转置，用于验证正确性"""
    k = len(tensors)
    N = len(tensors[0])
    chunk_size = N // k

    def chunk(r, c):
        return tensors[r][c * chunk_size:(c + 1) * chunk_size]

    # results[r] 的第 s 块 = chunk(s, r)
    return [np.concatenate([chunk(s, r) for s in range(k)]) for r in range(k)]


if __name__ == "__main__":
    np.random.seed(42)
    K = 4          # rank 数量
    N = 8          # 每个 rank 的数据量（必须能被 K 整除）

    # 为了便于肉眼验证，构造可读的数据：rank r 的第 c 块全部填充 (r*10 + c)
    chunk_size = N // K
    tensors = [
        np.concatenate([np.full(chunk_size, r * 10 + c, dtype=float) for c in range(K)])
        for r in range(K)
    ]

    print("输入数据（in[r][c] 表示 rank r 发给 rank c 的块，值 = r*10+c）：")
    for r, t in enumerate(tensors):
        print(f"  rank {r}: {t}")
    print()

    a2a_results = all_to_all(deepcopy(tensors))
    naive_results = naive_all_to_all(tensors)

    print("\n=== 验证结果 ===")
    print("（rank r 的第 s 块应等于 rank s 发给 r 的块，值 = s*10+r）")
    for r, res in enumerate(a2a_results):
        match = np.allclose(res, naive_results[r])
        print(f"  rank {r} 结果: {res}  ✓" if match else f"  rank {r} 结果: {res}  ✗ MISMATCH!")
