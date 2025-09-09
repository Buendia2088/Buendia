#include <iostream>
using namespace std;

void swap(int& a, int& b)
{
    int temp = a;
    a = b;
    b = temp;
}

void BubbleSort(int* a, int n)
{
    int mark = 1;
    int flag = 0;
    while(flag == 0)
    {
        flag = 1;
        for(int i = n - 1; i >= mark; i--)
        {
            if(a[i] < a[i-1])
            {
                swap(a[i], a[i-1]);
                flag = 0;
            }
        }
        mark++;
    }
}

void SelectionSort(int a[], int n)
{
    for(int i = 0; i < n - 1; i++)
    {
        int max_index = i + 1;
        for(int j = i + 1; j < n; j++)
        {
            if(a[j] > a[max_index])
            {
                max_index = j;
            }
        }
        if(a[max_index] > a[i]);
        swap(a[max_index], a[i]);
    }
}


void print(int a[], int n)
{
    cout << a[0];
    for(int i = 1; i < n; i++)
    {
        cout << " " << a[i]; 
    }
}


int main()
{
    int n = 10;
    int a[10];
    for(int i = 0; i < 10; i++)
    {
        cin >> a[i];
    }
    BubbleSort(a, 10);
    print(a, 10);
    cout << endl;
    for(int i = 0; i < n - 1; i++)
    {
        int max_index = i + 1;
        for(int j = i + 1; j < n; j++)
        {
            if(a[j] > a[max_index])
            {
                max_index = j;
            }
        }
        if(a[max_index] > a[i])
        {
            swap(a[max_index], a[i]);
        }

    }
    print(a, 10);
}