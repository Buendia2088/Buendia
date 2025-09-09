#include <iostream>
#include <vector>
using namespace std;

const int M = 25, N = 1010;
bool judge[M][M];                    // 记录两列是否矛盾
int a[N][M];                          // a数组记录输入
int sum[M];                           // 存储列的和
vector<int> best_set_a, best_set_b;   // 存储A, B的最优选择
int asum, bsum, eps = 100, tc;        // 存储当前的A, B元素和和计算参数

// 判断两列是否互斥
bool isCompatible(int x, int y) 
{
    for (int i = 0; i < 1000; i++) 
    {
        if (a[i][x] + a[i][y] == 2) return false;
    }
    return true;
}

// 预处理judge[][]数组判断两列是否互斥
void initJudge() 
{
    for (int i = 0; i < 20; i++) 
    {
        for (int j = i + 1; j < 20; j++) 
        {
            if (isCompatible(i, j)) judge[i][j] = judge[j][i] = true;
        }
    }
}

// 预处理所有列的和
void sumOfCol() 
{
    for (int i = 0; i < 20; i++) 
    {
        int res = 0;
        for (int j = 0; j < 1000; j++) 
        {
            res += a[j][i];
        }
        sum[i] = res;
    }
}

// 获得选择列的元素和
int getSum(const vector<int>& cols) 
{
    int res = 0;
    for (int col : cols) 
    {
        res += sum[col];
    }
    return res;
}

// 比较两组列A和B是否更优
bool isBetter(const vector<int>& a, const vector<int>& b) 
{
    if (a.size() > b.size()) return true;
    if (a.size() < b.size()) return false;
    if (abs((int)a.size() - (int)b.size()) < eps) 
    {
        for (size_t i = 0; i < min(a.size(), b.size()); i++) 
        {
            if (a[i] < b[i]) return true;
            if (a[i] > b[i]) return false;
        }
    }
    return false;
}

// 输出结果
void get_answer(const vector<int>& cols) {
    for (int col : cols) {
        cout << col << " ";
    }
    cout << endl;
}

// 更新函数
void update(const vector<int>& set_a, const vector<int>& set_b) {
    best_set_a = set_a;
    best_set_b = set_b;
    tc = set_a.size() + set_b.size();
    eps = abs((int)set_a.size() - (int)set_b.size());
}

int main() {
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < 20; j++) {
            cin >> a[i][j];
        }
    }

    // 预处理judge[]和sum[]数组
    initJudge();
    sumOfCol();

    // 迭代选择A
    for (int i = 1; i < (1 << 20); i++) {
        vector<int> set_a, set_b;

        for (int k = 0; k < 20; k++) {
            if (i & (1 << k)) {
                set_a.push_back(k);
            }
        }

        // 通过judge判断和定义B
        for (int k = 0; k < 20; k++) {
            if (set_a.empty()) continue;
            bool compatible = true;
            for (int col : set_a) {
                if (!judge[col][k]) {
                    compatible = false;
                    break;
                }
            }
            if (compatible) {
                set_b.push_back(k);
            }
        }

        asum = set_a.size();
        bsum = set_b.size();

        if (asum > bsum && bsum > 0) {
            if (asum + bsum > tc || (asum + bsum == tc && isBetter(set_a, best_set_a))) {
                update(set_a, set_b);
            }
        }
    }

    if (!best_set_a.empty()) {
        get_answer(best_set_a);
        get_answer(best_set_b);
    } else {
        cout << endl << endl;
    }

    return 0;
}
