#include <iostream>
#include <algorithm>

using namespace std;

const int MAX = 1e5 + 100;
int player[MAX], group_size[MAX], q[MAX], ans = 1e9;

int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        cin >> player[i];
    }
    sort(player, player + n);
    int group_num = 0;
    for(int i = 0; i < n; i++)
    {
        int pos = lower_bound(q, q + group_num, player[i]) - q;
        while(q[pos+1] == player[i] && pos + 1 < group_num)
        {
            pos++;
        }
        if(pos >= group_num || q[pos] != player[i])
        {
            q[group_num] = player[i] + 1;
            group_size[group_num]++;
            group_num++; 
        }
        else
        {
            group_size[pos]++;
            q[pos]++;
        }
    }
    for(int i = 0; i < group_num; i++)
    {
        ans = min(ans, group_size[i]);
    }
    cout << ans;
    return 0;
}