#include <iostream>
using namespace std;

int HalfFind(int a[], int length, int key)
{
    int low = 0; int high = length - 1;
    while(low < high)    
    {
        int mid = (low + high) / 2;
        if(key == a[mid])
        {
            return mid;
        }
        else if(key > a[mid])
        {
            low = mid + 1;
        }
        else
        {
            high = mid - 1;
        }
    }
    return -1;
}

int main()
{ 
    int n;
    cin >> n;
    int* arr = new int[n];
    for(int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }
    int target;
    cin >> target;
    cout << HalfFind(arr, n, target);
    return 0;
    
}