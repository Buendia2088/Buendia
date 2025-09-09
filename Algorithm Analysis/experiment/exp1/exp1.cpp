#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

struct UniElement //储存数字+权值
{
    int number;
    double weight;
    UniElement(int n = -1, double w = 0) : number(n), weight(w) {}
};

bool compareByNumber(const UniElement& a, const UniElement& b) 
{
    return a.number < b.number;
}

//求中位数的中位数的函数，用来计算划分位置
int medianOfMedians(int start, int end, vector<UniElement>& elements) 
{
    if (end - start < 5) //递归结束
    {
        sort(elements.begin() + start, elements.begin() + end + 1, compareByNumber);
        return elements[(start + end) / 2].number;
    }
    int numMedians = 0;
    for (int i = start; i <= end; i += 5) //找到每五个数的中位数，并放到向量头部，方便递归处理
    {
        int groupEnd = min(i + 5, end + 1);
        sort(elements.begin() + i, elements.begin() + groupEnd, compareByNumber);
        swap(elements[start + numMedians], elements[(groupEnd + i) / 2]);
        numMedians++;
    }
    return medianOfMedians(start, start + numMedians - 1, elements);
}

//将数据分成枢轴左右两部分
int partition(int start, int end, vector<UniElement>& elements) 
{
    if (start == end) 
    {
        return start;
    }
    int median = medianOfMedians(start, end, elements);
    int medianIndex = 0;
    for (int i = start; i <= end; i++) 
    {
        if (elements[i].number == median) 
        {
            medianIndex = i;
            break;
        }
    }
    swap(elements[medianIndex], elements[end]); //将枢轴移到向量末尾，方便用循环进行划分
    int storeIndex = start; //记录当前分界线
    for (int i = start; i < end; i++) {
        if (elements[i].number < median) {
            swap(elements[i], elements[storeIndex]);
            storeIndex++;
        }
    }
    swap(elements[storeIndex], elements[end]); //枢轴回位
    return storeIndex; //此时的分界线即为枢轴位置
}

// 计算带权中位数
void WeightMedian(int length, vector<int>& numbers, vector<double>& weights, int index = 0) 
{
    index++; //index已经被我优化掉了，此处是为了防止moodle卡unused variable编译不通过
    vector<UniElement> elements;
    for (int i = 0; i < length; i++) 
    {
        elements.emplace_back(numbers[i], weights[i]);
    }
    int start = 0, end = length - 1; //start，end比length，index好用不少
    double leftWeight = 0.0;
    while (start <= end) //我采用了循环，也可以递归，区别不大
    {
        int pivotIndex = partition(start, end, elements);
        leftWeight = 0.0;
        for (int i = 0; i < pivotIndex; i++)
        {
            leftWeight += elements[i].weight;
        }
        double pivotWeight = elements[pivotIndex].weight;
        double rightWeight = 1.0 - leftWeight - pivotWeight;
        if (leftWeight <= 0.5 && rightWeight <= 0.5) //符合带权中位数要求
        {
            cout << endl << "带权中位数是：" << elements[pivotIndex].number << endl;
            return;
        } 
        else if (leftWeight > 0.5) //说明带权中位数在左侧
        {
            end = pivotIndex - 1;
        } 
        else //说明带权中位数在右侧
        { 
            start = pivotIndex + 1;
        }
    }
}

int main() 
{
    int length;
    cin >> length;

    vector<int> numbers(length);
    vector<double> weights(length);

    for (int i = 0; i < length; i++) {
        cin >> numbers[i];
    }

    for (int i = 0; i < length; i++) {
        cin >> weights[i];
    }

    WeightMedian(length, numbers, weights);
    return 0;
}