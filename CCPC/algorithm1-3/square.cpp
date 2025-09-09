#include <iostream>
using namespace std;

int main() 
{
    long long n, m;
    cin >> n >> m;
    if(n > m) swap(n, m);
    long long count_rec = 0;
    long long count_squ = 0;
    for(long long k = 1; k <= n; k++) 
    {
        count_squ += (n - k + 1) * (m - k + 1);
        for(long long l = 1; l <= m; l++)
        {
            count_rec += (n + 1 - k) * (m + 1 - l);
        }
    }
    count_rec -= count_squ;
    cout << count_squ << " " << count_rec;
    return 0;
}
