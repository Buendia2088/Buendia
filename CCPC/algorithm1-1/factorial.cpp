#include <iostream>
#include <cstring>
#include <cstdio>
using namespace std;

int res[3000];

void fact(int* a, int& lenA, int b1)
{
    int* ans = new int[3000];
    memset(ans, 0, 3000 * sizeof(int));
    int* b = new int[10];
    int count = 0;
    while (b1 > 0)
    {
        b[count++] = (b1 % 10);
        b1 /= 10;
    }
    int lenB = count;
    for (int i = 0; i < lenA; i++)
    {
        for (int j = 0; j < lenB; j++)
        {
            ans[i + j] += a[i] * b[j];
        }
    }
    for (int i = 0; i < lenA + lenB; i++)
    {
        if (ans[i] > 9)
        {
            ans[i + 1] += ans[i] / 10;
            ans[i] %= 10;
        }
    }
    lenA += lenB;
    while (lenA > 0 && ans[lenA - 1] == 0)
    {
        lenA--;
    }
    for (int i = 0; i < lenA; i++)
    {
        a[i] = ans[i];
    }
    delete[] ans;
    delete[] b;
}

int main()
{
    int t;
    cin >> t;
    while (t--)
    {
        int n, target;
        cin >> n >> target;
        res[0] = 1;
        int lenRes = 1;
        for (int i = 1; i <= n; i++)
        {
            fact(res, lenRes, i);
        }
        int count = 0;
        for (int i = 0; i < lenRes; i++)
        {
            if (res[i] == target)
            {
                count++;
            }
        }
        cout << count << endl;
    }
    return 0;
}
