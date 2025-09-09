#include <bits/stdc++.h>
using namespace std;
int a[10005];
int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    int count = 0;
    for(int i = 0; i < n - 1; i++)
    {
        for(int j = i + 1; j < n; j++)
        {
            if(a[i] < a[j]) count++;
        }
    }
    count *= -1;
    for(int i = 1; i < n; i++)
    {
        count += i;
    }
    cout << count;
    return 0;
}