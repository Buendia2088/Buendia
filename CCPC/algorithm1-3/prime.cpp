#include <bits/stdc++.h>
using namespace std;

bool p[1005];

int main()
{
    int a, b;
    cin >> a >> b;
    for(int i = 0; i <= b; i++)
    {
        p[i] = true;
    }   
    p[0] = p[1] = false;
    for(int i = 2; i * i <= b; i++)
    {
        if(p[i] = true)
        {
            for(int j = i * i; j <= b; j += i)
            {
                p[j] = false;
            }
        }
    }
    for(int i = a; i <= b; i++)
    {
        if(p[i]) cout << i << " ";
    }
    return 0;
}