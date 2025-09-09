#include <bits/stdc++.h>
using namespace std;

char a[55][55];
int n, m;

int paint()
{
    int minCount = 114514;
    int count = 0;
    for(int i = 1; i <= n - 2; i++)
    {
        for(int j = 1; j <= n - 1 - i; j++)
        {
            for(int p = 1; p <= n - i - j; p++)
            {
                for(int k = 1; k <= n; k++)
                {
                    for(int l = 1; l <= m; l++)
                    {
                        if(k <= i && a[k][l] != 'W') count++;
                        if(k > i && k <= i + j && a[k][l] != 'B') count++;
                        if(k > i + j && a[k][l] != 'R') count++;
                    }
                }
                if(count < minCount)
                {
                    minCount = count;
                }
                count = 0;
            }
        }
    }
    return minCount;
}

int main()
{
    cin >> n >> m;
    for(int i = 1; i <= n; i++)
    {
        for(int j = 1; j <= m; j++)
        {
            cin >> a[i][j];
        }
    }
    cout << paint();
}