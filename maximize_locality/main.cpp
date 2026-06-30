#include <iostream>
#include <vector>

using namespace std;

// 使用 maximize_locality 算法构造 DBT (double binary tree)。
//
// 树的形状仍用堆 (heap) 布局表示: 下标 pos 即堆中的位置(0 为树根)，
// 子节点为 2*pos+1 / 2*pos+2，叶子满足 2*pos+1 >= k。
//
// 与 BFS/前序填充(tree[pos] = (pos+root)%k) 不同，这里按"中序遍历"
// (左子树 -> 当前节点 -> 右子树) 的顺序依次分配 rank。中序遍历的关键性质:
// 任意子树所覆盖的 rank 都是一段**连续区间**，因此物理上相邻的 rank 在树中
// 也彼此相邻(等价于对 [0, k) 做递归二分: 取中点为根，左半区间作左子树，
// 右半区间作右子树)，从而最大化通信局部性 (maximize locality)。
//
// offset 为按中序访问到的序号，最终 rank = (offset + root) % k，以支持指定 root。
//FIXME: something wrong here, with the method of maximizing locality, is not the some with NCCL topo logogy
void maximize_locality(int k, int root, vector<int>& tree, int pos, int& offset) {
    if (pos >= k) return;
    maximize_locality(k, root, tree, 2 * pos + 1, offset); // 先递归左子树
    tree[pos] = (offset + root) % k;                       // 中序访问当前节点
    ++offset;
    maximize_locality(k, root, tree, 2 * pos + 2, offset); // 再递归右子树
}

// 构造一棵以 root 为根、最大化局部性的二叉树。
void buildTree(int k, int root, vector<int>& tree) {
    tree.assign(k, -1);
    int offset = 0;
    maximize_locality(k, root, tree, 0, offset);
}

// 构造 DBT: Tree A 根为 rootA，Tree B 根为 (rootA + k/2) % k。
void buildDBT(int k, int rootA, vector<int>& treeA, vector<int>& treeB) {
    buildTree(k, rootA, treeA);
    int rootB = (rootA + k / 2) % k;
    buildTree(k, rootB, treeB);
}

// 按堆布局打印一棵树的父子关系，便于直观验证局部性。
void printTree(const string& name, int k, const vector<int>& tree) {
    cout << name << " (tree[pos] = rank): ";
    for (int rank : tree) cout << rank << " ";
    cout << "\n  root = " << tree[0] << "\n";
    for (int pos = 0; pos < k; ++pos) {
        int l = 2 * pos + 1, r = 2 * pos + 2;
        if (l >= k) continue; // 叶子节点不打印
        cout << "  rank " << tree[pos] << " -> {";
        cout << "left: " << (l < k ? tree[l] : -1);
        cout << ", right: " << (r < k ? tree[r] : -1) << "}\n";
    }
}

int main(int argc, char* argv[]) {
    int k = 8;     // 节点数量
    int rootA = 0; // Tree A 的根节点
    if (argc >= 2) k = atoi(argv[1]);
    if (argc >= 3) rootA = atoi(argv[2]);

    vector<int> treeA, treeB;
    buildDBT(k, rootA, treeA, treeB);

    printTree("Tree A", k, treeA);
    cout << "\n";
    printTree("Tree B", k, treeB);

    return 0;
}
