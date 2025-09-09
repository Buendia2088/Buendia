#include <iostream>
using namespace std;

const int MAX = 1e6;
int a[MAX], b[MAX / 10];
int bin_search(int target, int s, int e)
{
    if(s == e)
    {
        if(target == a[s])
        {
            return s;
        }
        else
        {
            return -1;
        }
    }
    int mid = (s + e) / 2;
    if(target > a[mid])
    {
        return bin_search(target, mid + 1 ,e);
    }
    else
    {
        return bin_search(target, s, mid);
    }
}

int main()
{
    int n, m;
    cin >> n >> m;
    for(int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    for(int i = 0; i < m; i++)
    {
        int target;
        cin >> target;
        int pos = bin_search(target, 0, n);
        if(pos == -1)
        {
            cout << pos << " ";
        }
        else
        {
            cout << pos + 1 << " ";
        }
    }
    return 0;   
}