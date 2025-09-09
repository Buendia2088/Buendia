#include <bits/stdc++.h>
using namespace std;

struct volunteer
{
    int id;
    int score;
    /* data */
};

volunteer a[5005];

bool compare(volunteer a, volunteer b) 
{
    if(a.score != b.score) return a.score > b.score;
    else return a.id < b.id;
}

int main()
{
    int n, m;
    cin >> n >> m;
    for(int i = 0; i < n; i++)
    {
        cin >> a[i].id >> a[i].score;
    }
    sort(a, a + n, compare);
    int max = 0;
    int min_score = a[static_cast<int>(m*1.5)-1].score;
    for(int i = n-1; i >= 0; i--)
    {
        if(a[i].score >= min_score)
        {
            max = i;
            break;
        }
    }
    cout << min_score << " " << max + 1 << endl;
    for(int i = 0; i <= max; i++)
    {
        cout << a[i].id << " " << a[i].score << endl;
    }
    return 0;
}
