#include <iostream>
#include <vector>
using namespace std;

int majorityElementRecursive(vector<int>& arr, int left, int right) 
{
    if (left == right) 
    {
        return arr[left];
    }
    int mid = left + (right - left) / 2;
    int leftMajority = majorityElementRecursive(arr, left, mid);
    int rightMajority = majorityElementRecursive(arr, mid + 1, right);
    if (leftMajority == rightMajority) 
    {
        return leftMajority;
    }
    int leftCount = 0, rightCount = 0;
    for (int i = left; i <= right; i++) //左主元素和右主元素不同，遍历当前数组确定主元素
    {
        if (arr[i] == leftMajority) 
        {
            leftCount++;
        }
        if (arr[i] == rightMajority) 
        {
            rightCount++;
        }
    }
    return leftCount > rightCount ? leftMajority : rightMajority;
}

int findMajorityElement(vector<int>& arr) //对majorityElementRecursive函数的结果进行验证
{
    int majority = majorityElementRecursive(arr, 0, arr.size() - 1);
    int count = 0;
    for (int num : arr) 
    {
        if (num == majority) 
        {
            count++;
        }
    }
    return count > arr.size() / 2 ? majority : -1;
}

int main() 
{
    int n;
    cin >> n;
    vector<int> arr;
    for(int i = 0; i < n; i++)
    {
        int temp;
        cin >> temp;
        arr.push_back(temp);
    }
    cout << "o(nlogn) algorithm result" << endl;
    cout << findMajorityElement(arr);
    return 0;
}

