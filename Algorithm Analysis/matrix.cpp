#include <iostream>
#include <vector>
using namespace std;

void solution(vector<int> p, vector<vector<int>>& dp, vector<vector<int>>& s)
{
    int n = p.size() - 1;
    for(int i = 0;i < n; i++)
    {
        dp[i][i] = 0;
    }
    for(int r = 2; r <= n; r++)
    {
        for(int i = 1; i <= n - r + 1; i++)
        {
            int j = i + r - 1;
            dp[i][j] = dp[i+1][j] + p[i-1]*p[i]*p[j];
            s[i][j] = i;
            for(int k = i+1; k < j; k++)
            {
                int temp = dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j];
                if(temp < dp[i][j])
                {
                    dp[i][j] = temp;
                    s[i][j] = k;
                }
            }

        }
    }
}

void traceback(vector<vector<int>>& s, int i, int j)
{
    if(i == j)
    {
        return;
    }
    else
    {
        traceback(s, i, s[i][j]);
        traceback(s, s[i][j]+1, j);
        cout << "Multipy A" << i << "," << s[i][j] << "and A" << s[i][j]+1 << "," << j;
    }
}

int main()
{
    // 定义矩阵链的维度数组
    vector<int> p = {30, 35, 15, 5, 10, 20, 25}; // 矩阵 A1: 30x35, A2: 35x15, ..., A6: 20x25
    int n = p.size() - 1;

    // 初始化 DP 和 S 表
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
    vector<vector<int>> s(n + 1, vector<int>(n + 1, 0));

    // 计算最优矩阵连乘顺序
    solution(p, dp, s);

    // 输出最小计算成本
    cout << "Minimum number of multiplications is: " << dp[1][n] << endl;

    // 输出具体的计算顺序
    cout << "Optimal multiplication order: " << endl;
    traceback(s, 1, n);

    return 0;
}