#include <iostream>
#include <vector>
#include <stack>
using namespace std;

struct treeNode
{
    treeNode* parent;
    bool leftChild;
    treeNode(treeNode* p, bool l) : parent(p), leftChild(l) {}
};

struct stackNode
{
    treeNode enode;
    int value;   // 当前节点的价值
    int weight;  // 当前节点的重量
    int level;   // 当前节点对应的物品编号
    stackNode(int v, int w, int i, treeNode p) : enode(p), value(v), weight(w), level(i) {}
};

class knapSack
{
private:
    int c;                   // 背包容量
    int n;                   // 物品数量
    vector<int> w;           // 每件物品的重量
    vector<int> v;           // 每件物品的价值
    int curW;                // 当前重量
    int curV;                // 当前价值
    vector<int> bestS;       // 最优解选择
    int bestV;               // 最优解价值
    stack<stackNode> s;      // 栈用于模拟搜索过程

    double bound(int i, int curWeight, int curValue) // 剪枝用
    {
        if (curWeight > c) return 0; // 超过容量无效
        double bound = curValue;
        while (i <= n && curWeight + w[i] <= c)
        {
            curWeight += w[i];
            bound += v[i];
            i++;
        }
        if (i <= n) // 加入部分物品
            bound += (c - curWeight) * (v[i] / (double)w[i]);
        return bound;
    }

public:
    knapSack(int c, int n, vector<int> w, vector<int> v)
        : c(c), n(n), w(w), v(v), curW(0), curV(0), bestS(n + 1, 0), bestV(0) {}

    void addToStack(int value, int weight, int level, treeNode* parent, bool leftChild)
    {
        treeNode newTreeNode(parent, leftChild);
        stackNode newStackNode(value, weight, level, newTreeNode);
        s.push(newStackNode);
    }

    void solve()
    {
        treeNode* root = nullptr;
        addToStack(0, 0, 1, root, false);

        while (!s.empty()) // 栈空即相当于所有节点都被遍历
        {
            stackNode node = s.top();
            s.pop();
            int i = node.level;
            if (i > n || node.weight > c)
            {
                continue;
            }
            if (node.value > bestV)
            {
                bestV = node.value;
                curW = node.weight;
                curV = node.value;
                treeNode* currentNode = &node.enode;
                for (int j = n; j > 0; j--)
                {
                    bestS[j] = currentNode && currentNode->leftChild ? 1 : 0;
                    if (currentNode)
                    {
                        currentNode = currentNode->parent;
                    } 
                }
            }
            if (i <= n)
            {
                int leftWeight = node.weight + w[i];
                int leftValue = node.value + v[i];
                if (leftWeight <= c && bound(i + 1, leftWeight, leftValue) > bestV) // 进入左孩子节点
                {
                    addToStack(leftValue, leftWeight, i + 1, &node.enode, true);
                }
            }
            if (i <= n && bound(i + 1, node.weight, node.value) > bestV) // 进入右孩子节点
            {
                addToStack(node.value, node.weight, i + 1, &node.enode, false);
            }
        }
        cout << "Maximum value: " << bestV << endl;
        cout << "Selected items: ";
        for (int j = 1; j <= n; j++)
        {
            if (bestS[j])
                cout << j << " ";
        }
        cout << endl;
    }
};

int main()
{
    int c = 50; // 背包容量
    int n = 4;  // 物品数量
    vector<int> w = {0, 10, 20, 30, 40}; // 每件物品的重量（从1开始）
    vector<int> v = {0, 60, 100, 120, 240}; // 每件物品的价值（从1开始）

    knapSack ks(c, n, w, v);
    ks.solve();

    return 0;
}
