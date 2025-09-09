#include <iostream>
#include <queue>
#include <vector>
using namespace std;

const int MAX = 114514;

struct heapNode
{
    int index;
    int length;

    heapNode(int i = -1, int l = 0) : index(i), length(l) {}
    bool operator<(const heapNode& other) const 
    {
        return length > other.length;  // 为了构建最小堆，将小的值排在前面
    }
};

void shortest(vector<vector<int>> a, int v, vector<int>& dist, vector<int>& p)
{
    int n = p.size() - 1;
    priority_queue<heapNode> minHeap;
    heapNode enode(v, 0);
    minHeap.push(enode);
    for(int i = 1; i < n; i++)
    {
        dist[i] = MAX;
    }
    dist[v] = 0;

    while(true)
    {
        enode = minHeap.top();  // 获取当前最小的节点
        minHeap.pop();
        for(int j = 1; j <= n; j++)
        {
            if(a[enode.index][j] < MAX && enode.length + a[enode.index][j] < dist[j])
            {
                dist[j] = enode.length + a[enode.index][j];
                p[j] = enode.index;
                heapNode newNode(j, dist[j]);
                minHeap.push(newNode);
            }
        }
        if(minHeap.empty()) break;
    }
}
void printShortestPaths(int source, const vector<int>& dist, const vector<int>& p)
{
    int n = dist.size() - 1;
    for(int i = 1; i <= n; i++)
    {
        if(dist[i] == MAX)
            cout << "从节点 " << source << " 到节点 " << i << " 不可达" << endl;
        else
        {
            cout << "从节点 " << source << " 到节点 " << i << " 的最短路径长度为 " << dist[i] << endl;
            // 打印路径
            cout << "路径为：";
            int pathNode = i;
            vector<int> path;
            while(pathNode != -1)
            {
                path.push_back(pathNode);
                pathNode = p[pathNode];
            }
            for(int j = path.size() - 1; j >= 0; j--)
                cout << path[j] << " ";
            cout << endl;
        }
    }
}

int main()
{
    int n, m;
    cout << "请输入图的节点数和边数: ";
    cin >> n >> m;

    // 创建图的邻接矩阵
    vector<vector<int>> graph(n + 1, vector<int>(n + 1, MAX));
    for(int i = 1; i <= n; i++)
        graph[i][i] = 0;  // 自己到自己距离为 0

    cout << "请输入每条边的信息 (格式: u v weight): " << endl;
    for(int i = 0; i < m; i++)
    {
        int u, v, w;
        cin >> u >> v >> w;
        graph[u][v] = w;
        graph[v][u] = w;  // 如果是无向图，取消这一行注释
    }

    int source;
    cout << "请输入源点: ";
    cin >> source;

    vector<int> dist(n + 1, MAX);  // 存储最短路径
    vector<int> p(n + 1, -1);     // 存储前驱节点

    shortest(graph, source, dist, p);  // 求最短路径

    // 打印结果
    printShortestPaths(source, dist, p);

    return 0;
}

