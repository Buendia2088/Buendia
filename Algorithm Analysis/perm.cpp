#include <iostream>
#include <vector>
using namespace std;

void swap(vector<int>& a, int i, int j)
{
    int temp = a[i];
    a[i] = a[j];
    a[j] = temp;
}

void perm(vector<int>& a, int n, int k)
{
    if(k == n)
    {
        for(int i = 0; i < n; i++)
        {
            cout << a[i] << " ";
        }
        cout << endl;
    }
    else
    {
        for(int i = k; i < n; i++)
        {
            swap(a, i, k);
            perm(a, n, k+1);
            swap(a, i, k);
        }
    }
}

int main()
{
    int n;
    cin >> n;
    vector<int> a;
    for(int i = 0; i < n; i++)
    {
        int temp;
        cin >> temp;
        a.push_back(temp);
    }
    perm(a, n, 0);
    return 0;
}