#include <iostream>
using namespace std;

struct workArr
{
    int workIndex;
    int workStep;
    int machIndex;
    workArr(int a = 0, int b = 0) : workIndex(a), machIndex(b), workStep(1) {}
};

int main()
{
    int m, n;
    cin >> m >> n;
    workArr* arr = new workArr[m*n];
    int* time = new int[m*n];
    int* stepCount = new int[n];
    bool* ifFinish = new bool[m*n];
    int* finishTime = new int[m];
    for(int i = 0; i < n; i++)
    {
        stepCount[i] = 0;
    }
    for(int i = 0; i < m*n; i++)
    {
        cin >> arr[i].workIndex;
        arr[i].workStep += stepCount[arr[i].workIndex-1]++;
    }
    for(int i = 1; i <= n; i++)
    {
        for(int j = 0; j < m; j++)
        {
            int temp;
            cin >> temp;
            for(int p = 0; p < m*n; p++)
            {
                if(arr[p].workIndex == i && arr[p].machIndex == 0)
                {
                    arr[p].machIndex = temp;
                    break;
                }
            }
        }
    }
    for(int i = 0; i < m*n; i++)
    {
        cout << arr[i].workIndex << " " << arr[i].workStep << " " << arr[i].machIndex;
        cout << endl;
    }
    for(int i = 0; i < m*n; i++)
    {
        cin >> time[i];
    }
    for(int i = 0; i < m; i++)
    {
        finishTime[i] = 0;
    }
    for(int i = 0; i < m*n; i++)
    {

    }
}