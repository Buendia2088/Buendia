#include <iostream>
#include <vector>
using namespace std;

class maxClique
{
private:
    int n; // 图的节点数
    int bestN; // 最大团的大小
    int curN; // 当前团的大小
    vector<int> bestS; // 存储最大团的节点
    vector<int> curS; // 存储当前团的节点
    vector<vector<int>> mat; // 图的邻接矩阵

public:
    maxClique(int nn = 0, vector<vector<int>> mm = {})
    {
        n = nn;
        bestN = curN = 0;
        mat = mm;
        bestS.resize(n + 1, 0);
        curS.resize(n + 1, 0);
        if (mm.empty()) {
            mat = vector<vector<int>>(n + 1, vector<int>(n + 1, 0));
        }
    }

    int solution()
    {
        int k = 1; // 当前节点
        while (true) 
        {
            bool flag = true;
            while (k <= n && flag) 
            {
                for (int i = 1; i < k; i++) 
                {
                    if (curS[i] == 1 && mat[k][i] == 0) // 检查是否可以加入最大团
                    { 
                        flag = false;
                        break;
                    }
                }
                if (flag) 
                {
                    curS[k] = 1; // 加入当前团
                    curN++;
                    k++;
                }
            }
            if (k > n) // 叶子节点，更新最大团
            { 
                if (curN > bestN) 
                {
                    bestN = curN;
                    bestS = curS;
                }
            } 
            else // 进入右子树
            { 
                curS[k] = 0; // 不选择当前节点
                k++; // 检查下一个节点
            }
            while (curN + n - k <= bestN) // 如果无法超过当前最大团，或者到达了叶节点
            {
                k--;
                if(curS[k] == 1) curN--;
                while (k > 0 && curS[k] == 0) // 回退到上一个左子树
                { 
                    k--;
                }                    
                if (k == 0) 
                {
                    return bestN; // 回到根节点，结束
                }
                curS[k] = 0; // 左子树处理完毕，尝试右子树
                curN--; // 回进入右子树，所以原先的左子树节点不再选择
                k++;
            }
        }
    }

    void printSolution()
    {
        cout << "Maximum Clique Size: " << bestN << endl;
        cout << "Nodes in the Maximum Clique: ";
        for (int i = 1; i <= n; i++) {
            if (bestS[i] == 1) {
                cout << i << " ";
            }
        }
        cout << endl;
    }
};

int main()
{
    vector<vector<int>> graph = {
        {0, 0, 0, 0, 0, 0}, 
        {0, 0, 1, 0, 1, 1}, 
        {0, 1, 0, 1, 0, 1}, 
        {0, 0, 1, 0, 1, 1}, 
        {0, 1, 0, 1, 0, 1}, 
        {0, 1, 1, 1, 1, 0}
    };

    maxClique mc(5, graph);
    mc.solution();
    mc.printSolution();

    return 0;
}
