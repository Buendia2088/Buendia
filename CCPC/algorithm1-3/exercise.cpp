#include <bits/stdc++.h>
using namespace std;

int a[25][25];

int main()
{
    int s1, s2, s3, s4;
    cin >> s1 >> s2 >> s3 >> s4;
    for(int i = 0; i < s1; i++)
    {
        cin >> a[0][i];
    }
    for(int i = 0; i < s2; i++)
    {
        cin >> a[1][i];
    }
    for(int i = 0; i < s3; i++)
    {
        cin >> a[2][i];
    }
    for(int i = 0; i < s4; i++)
    {
        cin >> a[3][i];
    }
}