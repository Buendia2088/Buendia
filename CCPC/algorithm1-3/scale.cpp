#include<bits/stdc++.h>
using namespace std;

struct scale
{
    int a, b, c;
    /* data */
};

int main()
{
    int a, b, c;
    cin >> a >> b >> c;
    int n1, n2, n3;
    int p = 0;
    for(int k = 1; k <= 999 / c; k++)
    {
        n1 = k * a;
        n2 = k * b;
        n3 = k * c;
        if(n2 > 999) break;
        int arr[10];
        for(int i = 0; i < 10; i++) arr[i] = 0;
        int flag = 0;
        for(int i = 1; i <= 3; i++)
        {
            arr[n1%10]++;
            n1 /= 10;
        }
        for(int i = 1; i <= 3; i++)
        {
            arr[n2%10]++;
            n2 /= 10;
        }
        for(int i = 1; i <= 3; i++)
        {
            arr[n3%10]++;
            n3 /= 10;
        }
        for(int i = 1; i <= 9; i++) if(arr[i] != 1) {flag = 1; break;} 
        if(flag != 1) {cout << k * a << " " << k * b << " " << k * c << endl; p = 1;}
    }
    if(p == 0)
    {
        cout << "No!!!";
    }
}