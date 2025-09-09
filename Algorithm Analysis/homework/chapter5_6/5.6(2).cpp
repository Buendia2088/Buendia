// 对于优化前后的方法，均运行1000次，求平均时间
// 传统方法运行时间为1.881e-06
// 使用优化后的上界函数后，运行时间为1.139e-06


#include<iostream>
#include<vector>
#include <chrono>
using namespace std;

const int MAX = 114514;

class Seller
{
private:
    int n;
    vector<int> curS;
    int curC;
    vector<vector<int>> mat;
    vector<int> closest; // 记录每个节点的最小出边
    int min; // 上界函数，通过维护min实现剪枝
public:
    vector<int> bestS;
    int bestC;
    Seller(int nn = 0, vector<vector<int>> mmaatt = {})
    {
        n = nn;
        mat = mmaatt;
        curS.resize(n+1, 0);
        bestS.resize(n+1, 0);
        closest.resize(n+1, 0);
        bestC = MAX;
        curC = 0;
        min = 0;
        for(int i = 1; i <= n; i++)
        {
            curS[i] = i;
            int temp = MAX;
            for(int j = 0; j <= n; j++)
            {
                if(mat[i][j] < temp)
                {
                    temp = mat[i][j];
                }
            }
            closest[i] = temp;
        }
        
    }

    void swap(int i, int j)
    {
        int temp = curS[i];
        curS[i] = curS[j];
        curS[j] = temp;
    }

    void solution(int i)
    {
        if(i == n)
        {
            if(mat[curS[n-1]][curS[n]] < MAX && mat[curS[n]][curS[1]] < MAX 
            && (bestC == MAX || curC + mat[curS[n-1]][curS[n]] + mat[curS[n]][curS[1]] < bestC))
            {
                bestC = curC + mat[curS[n-1]][curS[n]] + mat[curS[n]][curS[1]];
                bestS = curS;
            }
        }
            
        else
        {
            for(int j = i; j <= n; j++)
            {
                if(mat[curS[i-1]][curS[j]] < MAX
                && (bestC == MAX || curC + mat[curS[i-1]][curS[j]] < bestC)) // 传统方法
                //&& (bestC == MAX || min < bestC)) // 优化后的上界函数
                {
                    swap(i, j);
                    curC += mat[curS[i-1]][curS[i]];
                    min +=  mat[curS[i-1]][curS[i]]; 
                    min -= closest[i]; // 更新min
                    solution(i+1);
                    min += closest[i];
                    min -=  mat[curS[i-1]][curS[i]]; // 回溯时，也要还原min
                    curC -= mat[curS[i-1]][curS[i]]; 
                    swap(i, j);
                }
            }
        }
    }

    
};

int main()
{
    vector<vector<int>> mat = {
        {MAX, MAX, MAX, MAX, MAX},
        {MAX, MAX, 10, 15, 20}, 
        {MAX, 10, MAX, 35, 25},
        {MAX, 15, 35, MAX, 30},
        {MAX, 20, 25, 30, MAX}
    };
    int times = 1000;
    Seller tsp(4, mat);
    auto start = chrono::high_resolution_clock::now();
    for(int i = 0; i < times; i++)
    {
        tsp.solution(2);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    cout << "Shortest Path Cost: " << tsp.bestC << endl;
    cout << "Path: ";
    for (int i = 1; i <= 4; i++) 
    {
        cout << tsp.bestS[i] << " ";
    }
    cout << tsp.bestS[1] << " (back to start)" << endl;
    cout << "Execution Time: " << elapsed.count() / 1000.0 << " seconds" << endl;
    return 0;
}
