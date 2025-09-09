#include <bits/stdc++.h>
using namespace std;

struct stu
{
    int id;
    int a;
    int b;
    int c;
    int sum;
    void add() {sum = a + b + c;}
};

stu arr[305];

bool compare(stu a, stu b)
{
    if(a.sum != b.sum) return a.sum > b.sum;
    else if(a.a != b.a) return a.a > b.a;
    else return a.id < b.id;
}

int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        cin >> arr[i].a >> arr[i].b >> arr[i].c;
        arr[i].id = i + 1;
        arr[i].add();
    }
    sort(arr, arr+n, compare);
    for(int i = 0; i < 5; i++)
    {
        cout << arr[i].id << " " << arr[i].a+arr[i].b+arr[i].c << endl;
    }
    return 0;
}