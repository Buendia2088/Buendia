#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

int findMedian(int a[], int b[], int n) 
{
    int aStart = 0, aEnd = n - 1, bStart = 0, bEnd = n - 1;
    while (aStart < aEnd && bStart < bEnd) 
    {
        int aMedianIndex = (aStart + aEnd) / 2;
        int bMedianIndex = (bStart + bEnd) / 2;
        int aMedian = a[aMedianIndex];
        int bMedian = b[bMedianIndex];
        if (aMedian == bMedian)
        {
            return aMedian;
        } 
        else if (aMedian > bMedian) 
        {
            aEnd = aMedianIndex;
            bStart = bMedianIndex;
        } else 
        {
            aStart = aMedianIndex;
            bEnd = bMedianIndex;
        }
    }
    int lastArrLength = (aEnd - aStart + 1) + (bEnd - bStart + 1);
    vector<int> lastArr(lastArrLength);
    int count = 0;
    for (int i = aStart; i <= aEnd; i++) 
    {
        lastArr[count++] = a[i];
    }
    for (int i = bStart; i <= bEnd; i++) 
    {
        lastArr[count++] = b[i];
    }
    sort(lastArr.begin(), lastArr.end());
    return lastArr[(lastArrLength - 1) / 2];
}

int main() {
    int n;
    cin >> n;
    int a[n], b[n];
    for (int i = 0; i < n; i++) 
    {
        cin >> a[i];
    };
    for (int i = 0; i < n; i++) 
    {
        cin >> b[i];
    }
    cout << findMedian(a, b, n) << endl;
    return 0;
}
