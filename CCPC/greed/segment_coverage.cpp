#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

struct segment
{
    int l, r;
    segment(int a, int b) : l(a), r(b) {}
};

bool compare(segment a, segment b)
{
    return a.r < b.r;
}

int main()
{
    vector<segment> segments;
    int n;
    cin >> n;
    for(int i = 0; i < n; i++)
    {
        int l, r;
        cin >> l >> r;
        segments.push_back(segment(l, r));
    }
    sort(segments.begin(), segments.end(), compare);
    vector<segment> result;
    result.push_back(segments[0]);
    for(int i = 1; i < n; i++)
    {
        if(segments[i].l >= result.back().r)
        {
            result.push_back(segments[i]);
        }
    }
    cout << result.size();
    return 0;
}