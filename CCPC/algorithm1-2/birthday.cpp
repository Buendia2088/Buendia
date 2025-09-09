#include <bits/stdc++.h>
using namespace std;

struct stu
{
    string n;
    int y;
    int m;
    int d;
    int ind;
    /* data */
};

stu a[105];

bool compare(stu a, stu b)
{
    if(a.y != b.y) return a.y < b.y;
    else if(a.m != b.m) return a.m < b.m;
    else if(a.d != b.d) return a.d < b.d;
    else return a.ind > b.ind;
}
int main()
{
    int k;
    cin >> k;
    for(int i = 0; i < k; i++)
    {
        cin >> a[i].n >> a[i].y >> a[i].m >> a[i].d;
        a[i].ind = i;
    }
    sort(a, a + k, compare);
    for(int i = 0; i < k; i++)
    {
        cout << a[i].n << endl;
    }
    return 0;
}