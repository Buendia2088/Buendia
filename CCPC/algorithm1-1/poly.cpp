#include <iostream>
using namespace std;

int main()
{
    int n;
    cin >> n;
    int* a = new int[n+1];
    for(int i = 0; i <= n; i++)
    {
        cin >> a[i];
    }
    if(n == 0)
    {
        cout << a[0];
        return 0;
    }
    while(a[0] == 0)
    {
        if(n == 0)
        {
            return 0;
        }
        for(int i = 0; i < n; i++)
        {
            a[i] = a[i+1];
        }
        n--;
    }
    if(n != 0)
    {
        if(a[0] == 1) cout << "x";
        else if(a[0] == -1) cout << "-x";
        else cout << a[0] << "x";
        if(n != 1) cout << "^" << n;
    }
    else
    {
        cout << a[0];
    }
    for(int i = 1; i <= n; i++)
    {
        if(a[i] == 0) continue;
        else if(a[i] == 1)
        {
            if(i != n && i != n-1) cout << "+x^" << n-i;
            else if(i == n) cout << "+1";
            else cout << "+x";
            continue;
        }
        else if(a[i] == -1)
        {
            if(i != n && i != n-1) cout << "-x^" << n-i;
            else if(i == n) cout << -1;
            else cout << "-x";
            continue;
        }
        else if(a[i] > 0) cout << "+" << a[i];
        else cout << a[i];
        if(i != n && i != n-1) cout << "x^" << n-i;
        else if(i == n) cout << "";
        else cout << "x";
    }
    return 0;
}