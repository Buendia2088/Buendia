#include <bits/stdc++.h>
using namespace std;
int arr[5000005];
int temp[5000005];
int n;

void combine(int s, int k, int ti, int select[])
{
    if(ti == k)
    {
        for(int i = 0; i < ti; i++)
        {
            cout << setw(5) << temp[i];
        }
        cout << endl;
        return;
    }
    for(int i = 0; i < n; i++)
    {
        if(select[arr[i]] == 1) continue;
        temp[ti] = arr[i];
        ti++;
        select[arr[i]] = 1;
        combine(i + 1, k, ti, select);
        temp[ti] = 0;
        ti--;
        select[arr[i]] = 0;
    }
}

int main()
{
    int k;
    cin >> n;
    k = n;
    for(int i = 0; i < n; i++)
    {
        arr[i] = i + 1;
    }
    int select[14];
    for(int i = 0; i < 14; i++)
    {
        select[i] = 0;
    }
    combine(0, k, 0, select);
    return 0;
}