#include <iostream>
#include <vector>
int main()
{
    using namespace std;
    int n, k;
    cin >> n >> k;
    vector<int> light(n, 0);
    for(int i = 1; i <= k; i++)
    {
        for(int j = 1; (i * j) <= n; j++)
        {
            light[((i * j) - 1)]++;
        }
    }
    for(int p = 0; p < n; p++)
    {
        if((light[p] % 2) == 1)
        {
            cout << p + 1 << " ";
        }
    }
    return 0;
}
        
