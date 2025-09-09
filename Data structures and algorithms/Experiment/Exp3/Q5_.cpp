#include <iostream>
using namespace std;

const int MAXN = 114,INF = 514;
int g[MAXN][MAXN],dist[MAXN],n,m,res;
bool book[MAXN];

void prim()
{
    dist[1] = 0;
    book[1] = true;
    for(int i = 2 ; i <= n ;i++)dist[i] = min(dist[i],g[1][i]);
    for(int i = 2 ; i <= n ; i++)
    {
        int temp = INF;
        int t = -1;
        for(int j = 2 ; j <= n; j++)
        {
            if(!book[j]&&dist[j]<temp)
            {
                temp = dist[j];
                t = j;
            }
        }
        if(t==-1)
        {
            res = INF ; return ;
        }
        book[t] = true;
        res+=dist[t];
        for(int j = 2 ; j <= n ; j++)dist[j] = min(dist[j],g[t][j]);
    }
}

int main()
{
    cin>>n>>m;
    for(int i = 1 ; i<= n ;i++)
    {
        for(int j = 1; j <= n ;j++)
        {
            g[i][j] = INF;
        }
        dist[i] = INF;
    }
    for(int i = 1; i <= m ; i++)
    {
        int a,b,w;
        cin>>a>>b>>w;
        g[a][b] = g[b][a] = w;
    }
    prim();
    cout<<res;
    return 0;
}