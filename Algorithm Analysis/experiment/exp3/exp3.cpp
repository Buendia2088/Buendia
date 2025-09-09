#include <iostream>
#include <vector>
using namespace std;

int rem[10000];
int maxDist[10000];

struct node {
public:
    int index;
    vector<pair<int, int>> children;
    node(int index_ = -1, const vector<pair<int, int>>& children_ = {})
        : index(index_), children(children_) {}
};

class dTree {
public:
    int n;
    int d; 
    vector<node> tree;
    dTree(int n_ = -1, int d_ = -1) : n(n_), d(d_) 
    {
        for (int i = 0; i < n; i++) 
        {
            vector<pair<int, int>> tempVector;
            int numOfChildren;
            cin >> numOfChildren; 
            if (numOfChildren == 0) 
            {
                tree.emplace_back(i);
                continue;
            } 
            else 
            {
                for (int j = 0; j < numOfChildren; j++) 
                {
                    pair<int, int> tempPair;
                    cin >> tempPair.first >> tempPair.second;
                    tempVector.emplace_back(tempPair);
                }
            }
            tree.emplace_back(i, tempVector);
        }
    }

    int findMaxDistance(int current) 
    {
        if(tree[current].children.empty()) 
        {
            return 0;
        }
        if(maxDist[current] != 0) return maxDist[current];
        int maxDistance = 0;
        for(auto &[child, weight] : tree[current].children) 
        {
            if(rem[child] == 1) continue;
            maxDistance = max(maxDistance, findMaxDistance(child) + weight);
        }
        maxDist[current] = maxDistance;
        return maxDistance;
    }

    void solution()
    {
        int count = 0;
        for(int i = n - 1; i >= 0; i--)
        {
            int maxDistance = findMaxDistance(i);
            if(maxDistance > d)
            {
                rem[i] = 1;
                count++;
            }
        }
        cout << count << endl;
    }
};

int main() 
{
int n, d;
cin >> n >> d;
dTree dt(n, d);
dt.solution();
return 0;
}
