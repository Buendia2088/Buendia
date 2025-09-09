#include <iostream>

using namespace std;  

long long eat_candy(long long* a, int n, int x)
{
    long long ans = 0;
    if(a[0] > x) 
    {
        ans+= a[0] - x;
        a[0] = x;
    }
    for(int i = 1; i < n; i++)
    {
        if(a[i] + a[i-1] > x)
        {
            int dif = a[i] + a[i-1] - x;
            ans += dif;
            if(a[i] - dif >= 0)
            {
                a[i] -= dif;
            }
            else
            {
                a[i] = 0;
                a[i-1] -= dif - a[i];
            }
        }
    }
    return ans;
}

int main()
{
    int n, x;
    cin >> n >> x;
    long long a[n];
    for(int i = 0; i < n; i++)
    {
        int temp;
        cin >> a[i];
    }
    cout << eat_candy(a, n, x);
    return 0;
}