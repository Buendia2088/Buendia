#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int main()
{
    int n;
    cin >> n;
    vector<int> vote;
    for(int i = 0; i < n; i++)
    {
        int temp;
        cin >> temp;
        vote.push_back(temp);
    }
    sort(vote.begin(), vote.end());
    for(int i = 0; i < n; i++)
    {
        cout << vote[i] << " ";
    }
    return 0;
}