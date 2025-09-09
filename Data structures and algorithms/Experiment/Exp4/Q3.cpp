#include <vector>
#include <iostream>
#include <algorithm>

using namespace std;

vector<int> merge_sort(vector<int>& arr);
vector<int> merge(vector<int>& left, vector<int>& right);

vector<int> merge_sort(vector<int>& arr) 
{
    if (arr.size() <= 1) 
    {
        return arr;
    }

    int mid = arr.size() / 2;
    vector<int> left_half(arr.begin(), arr.begin() + mid);
    vector<int> right_half(arr.begin() + mid, arr.end());

    left_half = merge_sort(left_half);
    right_half = merge_sort(right_half);

    return merge(left_half, right_half);
}

vector<int> merge(vector<int>& left, vector<int>& right) 
{
    vector<int> result;
    size_t left_index = 0, right_index = 0;

    while (left_index < left.size() && right_index < right.size()) 
    {
        if (left[left_index] < right[right_index]) 
        {
            result.push_back(left[left_index]);
            left_index++;
        } else 
        {
            result.push_back(right[right_index]);
            right_index++;
        }
    }

    while (left_index < left.size()) 
    {
        result.push_back(left[left_index]);
        left_index++;
    }

    while (right_index < right.size()) 
    {
        result.push_back(right[right_index]);
        right_index++;
    }

    return result;
}

void FilterDown(vector<int>& arr, int i, int EndOfHeap)
{
    int child = 2 * i + 1;
    int cur = i;
    int temp = arr[i];
    while(child <= EndOfHeap)
    {
        if(child + 1 <= EndOfHeap && arr[child] < arr[child + 1])
        {
            child++;
        }
        if(arr[cur] >= arr[child])
        {
            break;
        }
        else
        {
            arr[cur] = arr[child];
            arr[child] = temp;
            cur = child;
            child = 2 * child + 1;
        }
    }
}

vector<int> HeapSort(vector<int> a)
{
    vector<int> arr = a;
    for(int i = (arr.size() - 2) / 2; i >= 0; i--)
    {
        FilterDown(arr, i, arr.size() - 1);
    }
    for(int i = arr.size() - 1; i >= 1; i--)
    {
        int temp = arr[i];
        arr[i] = arr[0];
        arr[0] = temp;
        FilterDown(arr, 0, i - 1);
    }
    return arr;
}


int main() 
{
    int n;
    cin >> n;
    vector<int> arr;
    while(n--)
    {
        int temp; 
        cin >> temp;
        arr.push_back(temp);
    }
    vector<int> merge_sorted_arr = merge_sort(arr);
    for(int i = 0; i < merge_sorted_arr.size(); i++) 
    {
        if(i != 0)
        {
            cout << " ";
        }
        cout << merge_sorted_arr[i];
    }
    cout << endl;

    vector<int> heap_sorted_arr = HeapSort(arr);
    reverse(heap_sorted_arr.begin(), heap_sorted_arr.end());
    
    for(int i = 0; i < heap_sorted_arr.size(); i++) 
    {
        if(i != 0)
        {
            cout << " ";
        }
        cout << heap_sorted_arr[i];
    }
    return 0;
}