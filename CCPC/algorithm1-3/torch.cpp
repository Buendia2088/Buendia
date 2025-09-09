#include <bits/stdc++.h>
using namespace std;

int p[10] = {6,2,5,5,4,5,6,3,7,6};
int m = 1000;

int countTorch(int k)
{
    int count = 0;
    count += p[k % 10];
    while(k /= 10)
    {
        count += p[k % 10];
    }
    return count;
}

int main()
{
    int n;
    cin >> n;
    if(n <= 4)
    {
        cout << 0;
        return 0;
    }
    n -= 4;
    int count = 0;
    for(int a = 0; a < m; a++)
    {
        for(int b = 0; b < m; b++)
        {
            if(countTorch(a) + countTorch(b) + countTorch(a + b) == n) count++;
        }
    }
    cout << count;
    return 0;
}