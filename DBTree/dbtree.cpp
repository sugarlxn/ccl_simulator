#include <vector>
#include <queue>
#include <iostream>

using namespace std;

//构造DBT double binary tree 给出rank 的数量k 以及指定root 节点
// 用 BFS(层序) 填充: tree 的下标 i 就是堆形二叉树中的位置(0 为树根)，
// 子节点为 2*i+1 / 2*i+2，叶子节点满足 2*i+1 >= k(下标即位置，关系成立)。
// 实际 rank 通过相对 root 的偏移得到: (pos + root) % k，从而以指定 root 建树。
void bfs(int k, int root, vector<int>& tree) {
    tree.assign(k, -1); // tree[pos] = rank，下标即堆位置
    queue<int> q;
    q.push(0); // 从位置0(树根)开始
    while (!q.empty()) {
        int pos = q.front(); q.pop();
        tree[pos] = (pos + root) % k; // 当前位置对应的 rank
        int l = 2 * pos + 1, r = 2 * pos + 2;
        if (l < k) q.push(l); // 左子节点
        if (r < k) q.push(r); // 右子节点
    }
}

void buildDBT(int k, int rootA, vector<int>& treeA, vector<int>& treeB) {
    bfs(k, rootA, treeA); // 构造Tree A，根节点为 rootA
    int rootB = (rootA + k/2) % k;
    bfs(k, rootB, treeB); // 构造Tree B，根节点为 rootB
}

int main(int argc, char* argv[]) {

    int k = 7; // 节点数量
    int rootA = 0; // Tree A 的根节点
    vector<int> treeA, treeB;
    buildDBT(k, rootA, treeA, treeB);

    // 输出Tree A
    cout << "Tree A: ";
    for (int node : treeA) {
        cout << node << " ";    
    }
    cout << endl;
    // 输出Tree B
    cout << "Tree B: ";
    for (int node : treeB) {
        cout << node << " ";
    }
    cout << endl;
}