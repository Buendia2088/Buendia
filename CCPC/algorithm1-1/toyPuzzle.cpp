#include<iostream>
#include<vector>
using namespace std;

int main()
{
    vector<pair<bool, string>> a;
    int n, m;
    cin >> n >> m;
    for(int i = 0; i < n; i++)
    {
        bool tempI;
        string tempS;
        cin >> tempI >> tempS;
        a.push_back({tempI, tempS});
    }
    //pair<bool, string> curP = a[0];
    int curIndex = 0;
    for(int i = 0; i < m; i++)
    {
        bool toward;
        int steps;
        cin >> toward >> steps;
        if((a[curIndex].first == 1 && toward == 0) || (a[curIndex].first == 0 && toward == 1))
        {
            curIndex = (curIndex + steps + n) % n;

        }
        else
        {
            curIndex = (curIndex - steps + n) % n;
        }
    }
    cout << a[curIndex].second << endl;
    return 0;
}