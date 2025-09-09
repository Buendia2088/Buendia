#include <iostream>
#include <cstring>
#include <cstdio>
#include <algorithm>
using namespace std;
const int MAX = 10001;
char a1[MAX],b1[MAX];
int a[MAX],b[MAX];
int ans[2*MAX];
int main()
{
    cin >> a1 >> b1;
    int lenA = strlen(a1);
    int lenB = strlen(b1);
    int save = 0;
    int maxPos = 1;
    bool aIs0 = true;
    bool bIs0 = true;
    for(int i = 1; i <= lenA; i++)
    {
        a[i] = a1[lenA - i] - '0';
        if(a[i] != 0) aIs0 = false;
    }
    for(int i = 1; i <= lenB; i++)
    {
        b[i] = b1[lenB - i] - '0';
        if(b[i] != 0) bIs0 = false;
    }
    if(aIs0 == true || bIs0 == true) 
    {
        cout << 0;
        return 0;
    }
    for(int i = 1; i <= lenA; i++)
    {
        for(int j = 1; j <= lenB; j++)
        {
            ans[i+j-1] += a[i] * b[j]  + save;
        }
    }
    for(int i = 1; i < lenA + lenB; i++)
    {
        if(ans[i] > 9)
        {
            ans[i+1] += (ans[i] / 10);
            ans[i] -= ((ans[i] / 10) * 10);
        }
    }
    if(ans[lenA + lenB]) cout << ans[lenA + lenB];
    for(int i = lenA+lenB-1; i > 0; i--)
    {
        cout << ans[i];
    }
    return 0;
}