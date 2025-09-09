#include <iostream>

#include <string>

#include <cmath>

#include <vector>

#include <algorithm>

using namespace std;

const int MAX = 10e8;

void MinCost(int L,int n,int *points)
{
    sort(points, points + n + 2); //排序切割点，方便处理
    int m[n+2][n+2]; //储存最小代价的数组
    for(int i = 0; i < n + 1; i++)
    {
        m[i][i+1] = 0; //相邻切割点之间无需切割，代价为0
    }
    for(int lengthOfIron = 2; lengthOfIron <= n + 1; lengthOfIron++) //最外层循环，用于按照切割长度顺序依次计算切割代价，便于长切割代价计算调用短切割代价计算的结果
    {
        for(int start = 0; start <= n - lengthOfIron + 1; start++) //中层循环，遍历给定切割长度时的所有切割方法
        {
            int minStep = MAX; //赋最大值
            for(int breakPoint = start + 1; breakPoint < start + lengthOfIron; breakPoint++) //内层循环，遍历给定切割方法时的所有切割点，找到最优切割点
            {
                int t = m[start][breakPoint] + m[breakPoint][start+lengthOfIron] + points[start+lengthOfIron] - points[start];
                if(t < minStep)
                {
                    minStep = t;
                    m[start][start+lengthOfIron] = minStep;
                }
            }
        }
    }
    cout << m[0][n+1]; //输出结果
}
int main() {

int L, n;

cin>>L>>n;

int *p;

p = new int[n+2];

p[0] = 0;

p[n+1] = L;

for(int i=1;i<n+1;i++){

cin>>p[i];

}

MinCost(L,n,p);

return 0;

}