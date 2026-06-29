"""
DBT 构造算法 — 从 k 个 rank 编号出发，
用纯 BFS 位置公式推导每个 rank 在 Tree A / Tree B 里的
parent、left_child、right_child。
关键思想: 完全二叉树（数组存储方法）的叶子节点position 满足： 2*p+1 >= k , 即 pos >= k//2
"""

from dataclasses import dataclass, field
import math

@dataclass
class TreeNode:
    rank:        int
    parent:      int = -1          # -1 表示是 root
    children:    list[int] = field(default_factory=list)
    depth:       int = 0
    is_leaf:     bool = False


def build_binary_tree(k: int, root: int) -> dict[int, TreeNode]:
    """
    核心算法：给定 k 个 rank、指定 root，
    用 BFS 位置公式计算每个 rank 的 parent / children。

    关键公式（BFS 数组索引）：
        pos(rank x)       = (x - root + k) % k
        rank_of(pos p)    = (root + p) % k
        parent_pos(p)     = (p - 1) // 2          (p > 0)
        left_child_pos(p) = 2 * p + 1             (< k 才有效)
        right_child_pos   = 2 * p + 2             (< k 才有效)
    """
    nodes = {r: TreeNode(rank=r) for r in range(k)}

    for rank in range(k):
        pos = (rank - root + k) % k               # 该 rank 在 BFS 数组里的位置

        # ── parent ──────────────────────────────
        if pos == 0:
            nodes[rank].parent = -1               # root 无父节点
            nodes[rank].depth  = 0
        else:
            parent_pos  = (pos - 1) // 2
            parent_rank = (root + parent_pos) % k
            nodes[rank].parent = parent_rank
            nodes[rank].depth  = int(math.log2(pos + 1))   # BFS 层号

        # ── children ────────────────────────────
        for child_offset in (2 * pos + 1, 2 * pos + 2):   # left, right
            if child_offset < k:
                child_rank = (root + child_offset) % k
                nodes[rank].children.append(child_rank)

        # ── leaf 判定 ────────────────────────────
        # BFS 位置 p 是叶当且仅当 2*p+1 >= k（没有左子节点）
        nodes[rank].is_leaf = (2 * pos + 1 >= k)

    return nodes


def build_dbt(k: int, root_a: int = 0) -> tuple[dict, dict, int, int]:
    """
    构造 Double Binary Tree。

    root_B 选取规则：
        root_B = (root_A + k // 2) % k

    这保证：
        pos_B(root_A) = (root_A - root_B + k) % k
                      = (k - k//2) % k
                      = k - k//2
                      >= k//2          (即 root_A 在 Tree B 里是叶)
        对称地，root_B 在 Tree A 里的 pos = k//2，同样 >= k//2，也是叶。
    """
    root_b = (root_a + k // 2) % k

    tree_a = build_binary_tree(k, root_a)
    tree_b = build_binary_tree(k, root_b)

    return tree_a, tree_b, root_a, root_b


def print_tree(nodes: dict[int, TreeNode], root: int, name: str):
    print(f"\n── {name} (root={root}, k={len(nodes)}) ──")
    print(f"  {'rank':>4}  {'depth':>5}  {'parent':>6}  {'children':<16}  leaf?")
    print(f"  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*16}  -----")
    for r in sorted(nodes):
        n = nodes[r]
        parent_str   = str(n.parent) if n.parent != -1 else "root"
        children_str = str(n.children) if n.children else "[]"
        print(f"  {r:>4}  {n.depth:>5}  {parent_str:>6}  {children_str:<16}  {'yes' if n.is_leaf else ''}")


def verify_complementarity(tree_a, tree_b, root_a, root_b, k):
    """验证两棵树的互补性"""
    print(f"\n── 互补性验证 ──")

    # root_A 在 Tree B 里必须是叶
    assert tree_b[root_a].is_leaf, f"root_A={root_a} 在 Tree B 里不是叶！"
    print(f"  root_A (rank {root_a}) 在 Tree B 里: leaf={tree_b[root_a].is_leaf}  ✓")

    # root_B 在 Tree A 里必须是叶
    assert tree_a[root_b].is_leaf, f"root_B={root_b} 在 Tree A 里不是叶！"
    print(f"  root_B (rank {root_b}) 在 Tree A 里: leaf={tree_a[root_b].is_leaf}  ✓")

    # 每个 rank 在两棵树里至少有一个角色是 relay（非叶非根）或 root
    always_leaf = []
    for r in range(k):
        if tree_a[r].is_leaf and tree_b[r].is_leaf:
            always_leaf.append(r)
    if always_leaf:
        # 小 k 下可能存在在两棵树都是叶的节点（纯接收者）
        print(f"  注意：ranks {always_leaf} 在两棵树均为叶（纯接收，k={k} 较小时正常）")
    else:
        print(f"  所有 rank 在至少一棵树里是 relay 或 root  ✓")

    # 两棵树的步数
    steps = math.ceil(math.log2(k)) if k > 1 else 0
    print(f"  每棵树步数 = ceil(log2({k})) = {steps}")
    print(f"  两树并发：带宽利用率 ≈ 2× single tree")


if __name__ == "__main__":
    for K in [7, 8, 10]:
        print(f"\n{'='*54}")
        print(f"  k = {K}")
        tree_a, tree_b, root_a, root_b = build_dbt(K, root_a=0)
        print_tree(tree_a, root_a, "Tree A")
        print_tree(tree_b, root_b, "Tree B")
        verify_complementarity(tree_a, tree_b, root_a, root_b, K)