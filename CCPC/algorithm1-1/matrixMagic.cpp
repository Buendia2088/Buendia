#include<iostream>
#include<vector>
using namespace std;

vector<vector<int>> swap(const vector<vector<int>>& a, int x, int y, int r, bool z)
{
    vector<vector<int>> res = a;
    for(int j = y+r; j >= y-r; j--)
    {
        for(int i = x-r; i <= x+r; i++)
        {
            if(!z) res[i][j] = a[x+y-j][i-x+y];
            else res[i][j] = a[j-y+x][x+y-i];
        }
    }
    return res;
}

int main()
{
    int n, m;
    cin >> n >> m;
    int count = 0;
    vector<vector<int>> a(n+1, vector<int>(n+1, 0));
    for(int i = 1; i <= n; i++)
    {
        for(int j = 1; j <= n; j++)
        {
            a[i][j] = ++count;
        }
    }
    for(int i = 0; i < m; i++)
    {
        int x, y, r;
        bool z;
        cin >> x >> y >> r >> z;
        a = swap(a, x, y, r, z);
    }
    for(int i = 1; i <= n; i++)
    {
        for(int j = 1; j <= n; j++)
        {
            cout << a[i][j] << " ";
        }
        cout << endl;
    }
    return 0;
}