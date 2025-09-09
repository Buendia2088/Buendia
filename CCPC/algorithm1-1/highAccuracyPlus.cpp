#include <iostream>
#include <string>
#include <algorithm>

using namespace std;

void add(string a, string b)
{
    int i = 0;
    reverse(a.begin(), a.end());
    reverse(b.begin(), b.end());
    while(a.size() < b.size())
    {
        a += '0';
    }
    while (b.size() < a.size())
    {
        b += '0';
    }
    int store = 0;
    int ans[100] = {};
    for(int i = 0; i < a.length(); i++)
    {
        ans[i] = (a[i] - '0') + (b[i] - '0') + store;
        if(ans[i] >= 10)
        {
            store = 1;
            ans[i] -= 10;
        }
        else
        {
            store = 0;
        }
    }
    if(store == 1) cout << "1";

    for(int i = a.length() - 1; i >=0; i--)
    {
        cout << ans[i];
    }
}

int main()
{
    string a, b;
    cin >> a >> b;
    add(a, b);
    return 0;


    
}