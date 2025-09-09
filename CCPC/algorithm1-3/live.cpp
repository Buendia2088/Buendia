#include <bits/stdc++.h>
using namespace std;

int r, c, k;
char a[105][105];

int count()
{
    int count = 0;
    bool flag = true;
    for(int i = 1; i <= r; i++)
    {
        for(int j = 1; j <= c - k + 1; j++)
        {
            for(int l = j; l < j + k; l++)
            {
                if(a[i][l] == '#')
                {
                    flag = false;
                    break;
                }
            }
            if(flag) count++;
            flag = true;
        }
    }
    for(int i = 1; i <= c; i++)
    {
        for(int j = 1; j <= r - k + 1; j++)
        {
            for(int l = j; l < j + k; l++)
            {
                if(a[l][i] == '#')
                {
                    flag = false;
                    break;
                }
            }
            if(flag) count++;
            flag = true;
        }
    }
    return count;
}

int main()
{
    cin >> r >> c >> k;
    if(k == 0)
    {
        cout << 0;
        return 0;
    }
    int pointCount = 0;
    for(int i = 1; i <= r; i++)
    {
        for(int j = 1; j <= c; j++)
        {
            cin >> a[i][j];
            if(a[i][j] == '.') pointCount++;
        }
    }
    if(k == 1)
    {
        cout << pointCount;
        return 0;
    }
    cout << count();
}