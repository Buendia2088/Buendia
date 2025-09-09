#include <bits/stdc++.h>
using namespace std;
string a[25];
bool compare(string a, string b) {return a+b > b+a;}

int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        cin >> a[i];
    }
    sort(a, a + n, compare);
    for(int i = 0; i < n; i++)
    {
        cout << a[i];
    }
    return 0;
}
