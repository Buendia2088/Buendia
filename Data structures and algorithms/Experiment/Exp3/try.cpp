#include <vector>
#include <iostream>

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

int main() 
{
    vector<int> arr = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    vector<int> sorted_arr = merge_sort(arr);

    for (int num : sorted_arr) 
    {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
