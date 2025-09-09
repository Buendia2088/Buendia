#include <iostream>
using namespace std;

int findMajorityElement(int arr[], int length)
{
    int vote = 0;
    int num = arr[0];
    int index = 0;
    for(int i = 0; i < length; i++)
    {
        if(num == arr[i])
        {
            vote++;
        }
        else
        {
            vote--;
        }
        if(vote == 0)
        {
            num == arr[i];
            index = i;
            vote = 1;
        }
    }
    vote = 0;
    for(int i = 0; i < length; i++)
    {
        if(arr[i] == num)
        {
            vote++;
        }
    }
    if(vote > length / 2) return num;
    else return -1;
}

int main()
{
    int n;
    cin >> n;
    int arr[n];
    for(int i = 0; i < n; i++)
    {
        cin >> arr[i];
    }
    cout << "o(n) algorithm result" << endl;
    cout << findMajorityElement(arr, sizeof(arr) / sizeof(arr[0]));
    return 0;
}
