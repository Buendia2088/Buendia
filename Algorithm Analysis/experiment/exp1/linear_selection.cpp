#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

// 在5个元素的数组中找到中位数
int median_of_five(vector<int>& arr, int left) {
    vector<int> temp(arr.begin() + left, arr.begin() + left + 5);
    sort(temp.begin(), temp.end());
    return temp[2];  // 返回5个数中的中位数
}

// 找到中位数的中位数
int median_of_medians(vector<int>& arr, int left, int right) {
    // 如果当前范围内元素少于5个，直接找中位数
    if (right - left + 1 <= 5) {
        sort(arr.begin() + left, arr.begin() + right + 1);
        return arr[(left + right) / 2];
    }

    // 分成若干个5元素的小组，找到每组的中位数
    int num_medians = 0;
    for (int i = left; i <= right; i += 5) {
        int sub_right = min(i + 4, right);
        int median = median_of_five(arr, i);
        swap(arr[left + num_medians], arr[i + 2]);  // 把每组的中位数放到数组开头部分
        num_medians++;
    }

    // 递归地找到中位数的中位数
    return median_of_medians(arr, left, left + num_medians - 1);
}

// 使用中位数的中位数作为枢轴进行划分
int partition(vector<int>& arr, int left, int right, int pivot) 
{
    int pivot_index = find(arr.begin() + left, arr.begin() + right + 1, pivot) - arr.begin();
    swap(arr[pivot_index], arr[right]);  // 将枢轴移到最后
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (arr[j] <= pivot) {
            i++;
            swap(arr[i], arr[j]);
        }
    }

    swap(arr[i + 1], arr[right]);
    return i + 1;
}

// Median of Medians-based Quickselect algorithm
int linear_time_select(vector<int>& arr, int left, int right, int k) {
    if (left == right) {
        return arr[left];
    }

    // 使用中位数的中位数作为枢轴
    int pivot = median_of_medians(arr, left, right);

    // 对数组进行划分
    int pivot_index = partition(arr, left, right, pivot);

    // 判断第k小的元素位置
    if (pivot_index == k) {
        return arr[pivot_index];
    } else if (pivot_index > k) {
        return linear_time_select(arr, left, pivot_index - 1, k);
    } else {
        return linear_time_select(arr, pivot_index + 1, right, k);
    }
}

int main() {
    vector<int> arr = {7, 2, 1, 8, 6, 3, 5, 4};
    int k;
    
    cout << "Enter the value of k (0-based index): ";
    cin >> k;
    
    if (k >= 0 && k < arr.size()) {
        // 查找数组中第 k 小的元素
        int result = linear_time_select(arr, 0, arr.size() - 1, k);
        cout << "The " << k + 1 << "th smallest element is: " << result << endl;
    } else {
        cout << "Invalid input. Please provide a value of k between 0 and " << arr.size() - 1 << "." << endl;
    }

    return 0;
}
