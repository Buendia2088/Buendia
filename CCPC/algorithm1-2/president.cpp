#include <bits/stdc++.h>
using namespace std;

struct president
{
    int id;
    string vote;
    int leng;
    void ini()
    {
        leng = vote.length();
        reverse(vote.begin(), vote.end());
        for(int i = 1; i <= 100-leng; i++)
        {
            vote += "0";
        }
        reverse(vote.begin(), vote.end());
    }
};

bool compare(president a, president b) {return a.vote > b.vote;}

president a[30];

int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        cin >> a[i].vote;
        a[i].id = i + 1;
        a[i].ini();
    }
    sort(a, a+n, compare);
    cout << a[0].id << endl;
    for(int i = 100 - a[0].leng; i <= 99; i++)
    {
        cout << a[0].vote[i];
    }
    return 0;
}