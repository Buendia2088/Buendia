#include <iostream>
int main()
{
    using namespace std;
    int N;
    int rm = 0;
    int total;
    cin >> N;

    const int Max_size = 100;
    int num[Max_size] = {0};

    for(int i = 0; i < N; i++)
    {
        int a;
        cin >> a;
        bool exists = false;
        for(int j = 0; j < i; j++)
        {
            if(num[j] == a)
            {
                exists = true;
                rm++;
                break;
            }
        }
        if(exists != true)
        {
            num[i] = a;
        }
    }

    for(int i = 0; i < N - 1; i++)
    {
        for(int j = 0; j < N - 1 - i; j++)
        {
            if(num[j] > num[j + 1])
            {
                swap(num[j], num[i]);
            }
        }
    }

    total = N - rm;
    cout << total << endl;

    for(int i = 0; i < N; i++)
    {
        if(num[i] != 0)
        {
            cout << num[i] << " ";
        }
    }
    
    cout << endl;
   
    return 0;
}
     
