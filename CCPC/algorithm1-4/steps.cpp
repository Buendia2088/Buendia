#include <bits/stdc++.h>
using namespace std;

int f(int a)
{
    if(a == 1 || a == 0) return 1;
    else if(a < 0) return 0;
    else return (f(a - 1) + f(a - 2));
}

int main()
{
    int n;
    cin >> n;
    cout << f(n);
    return 0;
}