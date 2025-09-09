#include <iostream>
#include <cmath>
#include <iomanip>
using namespace std;

int partition(int* a[], int low, int high)
{
    int pivot = *a[low];
    while(low < high)
    {
        while(low < high && *a[high] >= pivot)
            high--;
        swap(a[low], a[high]);
        while(low < high && *a[low] <= pivot)
            low++;
        swap(a[low], a[high]);
    }
    //*a[low] = pivot;
    return low;
}

void quicksort(int* a[], int low, int high)
{
    if(low < high)
    {
        int pivotpos = partition(a, low, high);
        quicksort(a, low, pivotpos-1);
        quicksort(a, pivotpos+1, high);
    }
}
  
int main()
{
    long N;
    cin >> N;
    double output = 0.0;
    const long Size = N;
    int points[N][3] = {0};

    for(int i = 0; i < N; i++)
    {
        cin >> points[i][0] >> points[i][1] >> points[i][2];
    }

    int* ptrs[Size];
    for(int i = 0; i < Size; i++)
    {
        ptrs[i] = &points[i][2];
    }

    quicksort(ptrs, 0, N-1);
    
    int receive[Size][3];
    for(int i = 0; i < Size; i++)
    {
        receive[i][2] = *ptrs[i];
        receive[i][1] = *(ptrs[i] - 1);
        receive[i][0] = *(ptrs[i] - 2);
    }

    double res = 0.0;
    for(int i = 0; i < N - 1; i++)
    {
        for(int k = 0; k < 3; k++)
        {
            res += pow((receive[i][k] - receive[i+1][k]), 2);
        }
        output += sqrt(res);
        res = 0.0;
    }
    
    cout << fixed << setprecision(3) << output << endl;
    return 0;
}
