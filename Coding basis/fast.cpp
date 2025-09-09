#include <iostream>
using namespace std;

int partition(int a[], int low, int high)
{
    int pivot = a[low];
    while(low < high)
    {
        while(low < high && a[high] >= pivot)
            high--;
        a[low] = a[high];
        while(low < high && a[low] <= pivot)
            low++;
        a[high] = a[low];
    }
    a[low] = pivot;
    return pivot;
}

void quicksort(int a[], int low, int high)
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
    int a[7] = {3, 2, 7, 5, 4, 1, 6};
    quicksort(a, 0, 6);
    for(int i = 0; i < 7; i++)
    {
        cout << a[i] << endl;
    }
    return 0;
}
