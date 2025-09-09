#include <iostream>
#include <algorithm>
using namespace std;

long long answer;

struct minister
{
    int left;
    int right;
    minister(int l = 0, int r = 0) : left(l), right(r) {}
};

minister ministers[2000];
bool compare(minister a, minister b)
{
    return a.left * a.right < b.left * b.right;
}

int main()
{
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        cin >> ministers[i].left >> ministers[i].right;
    }
    sort(ministers, ministers + n, compare);
    long long max_coin = 0;

}