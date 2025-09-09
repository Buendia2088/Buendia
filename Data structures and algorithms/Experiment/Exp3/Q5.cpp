#include <iostream>
using namespace std;

const int MAX = 114;
const int INF = 514;
int weight_graph[MAX][MAX];
int dist[MAX];
bool if_use[MAX];
int n, m, res;

void prim()
{
    dist[0] = 0;
    if_use[0] = true;
    int index = -1;
    int temp = INF;
    for(int i = 1; i < n; i++)
    {
        
    }
    
}

int main()
{
    cin >> n >> m;
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            weight_graph[i][j] = 0;
        }
        dist[i] = INF;
        if_use[i] = false;
    }
    for(int i = 0; i < n; i++)
    {
        int a, b, w;
        cin >> a >> b >> w;
        weight_graph[a][b] = weight_graph[b][a] = w;
    }
    prim();
}