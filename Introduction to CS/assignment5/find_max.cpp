#include <iostream>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
using namespace std;

constexpr int SIZE = 10000;
constexpr int HALF_SIZE = SIZE / 2;

void generateRandomNumbers(vector<int>& numbers) 
{
    for (int i = 0; i < SIZE; i++) 
    {
        numbers.push_back(rand() % 1000);
    }
}

int findMax(const vector<int>& numbers, int start, int end) 
{
    int max = numbers[start];
    for (int i = start + 1; i < end; i++) 
    {
        if (numbers[i] > max) 
        {
            max = numbers[i];
        }
    }
    return max;
}

int main() 
{
    vector<int> numbers;
    generateRandomNumbers(numbers);

    pid_t pid = fork();

    if (pid < 0) 
    {
        cerr << "Fork failed" << endl;
        return 1;
    } 
    else if (pid == 0) // 子进程1
    {
        int max1 = findMax(numbers, 0, HALF_SIZE);
        cout << "进程1找到前5000个数中的最大值：" << max1 << std::endl;
    } 
    else // 父进程
    {
        pid_t pid2 = fork();
        if (pid2 < 0) 
        {
            cerr << "Fork failed" << std::endl;
            return 1;
        } 
        else if (pid2 == 0) // 子进程2
        {
            int max2 = findMax(numbers, HALF_SIZE, SIZE);
            cout << "进程2找到后5000个数中的最大值：" << max2 << std::endl;
        } else // 父进程等待子进程1、2结束 
        {
            wait(NULL);
            wait(NULL);
            cout << "父进程结束" << endl;
        }
    }

    return 0;
}
