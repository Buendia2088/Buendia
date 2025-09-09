#include <bits/stdc++.h>
int a[6000000];
using namespace std;

int partition(int a[], int left, int right)
{
    int pivot = a[right];
    int i = left - 1;
    for(int j = left; j <= right; j++)
    {
        if(a[j] <= pivot)
        {
            i++;
            swap(a[i], a[j]);
        }      
    }
    i++;
    swap(a[i], a[right]);
    return i;
}

int Select(int a[], int left, int right, int k)
{
    if(left == right)
    {
        return a[left];
    }
    int p = partition(a, left, right);
    if(p == k)
    {
        return a[p];
    }
    else if(k < p)
    {
        return Select(a, left, p - 1, k);
    }
    else
    {
        return Select(a, p + 1, right, k);
    }
}
int main()
{
    int n, k;
    cin >> n >> k;
    for(int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    cout << Select(a, 0, n - 1, k);
    return 0;
}
