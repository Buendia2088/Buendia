#include <iostream>
#include <list>
using namespace std;
void training(int n)
{
    list<int> soldiers;
    for(int i = 1; i < n; i++)
    {
        soldiers.push_back(i);
    }
    int count = 1;
    int round = 1;
    while(soldiers.size() > 3)
    {
        for(list<int>::iterator it = soldiers.begin(); it != soldiers.end();)
        {
            if((round == 1 && count == 2) || (round == 2 && count == 3))
            {
                it = soldiers.erase(it);
                count = 1;
            }
            else
            {
                count++;
                it++;
            }
        }
        if(round == 1)
        {
            round = 2;
            count = 1;
        }
        else
        {
            round = 1;
            count = 1;
        }
    }
    cout << *soldiers.begin();
    for(list<int>::iterator it = ++(soldiers.begin()); it != soldiers.end(); it++)
    {
        cout << " " << *it;
    }
}

int main()
{
    int n;
    cin >> n;
    training(n);
    return 0;
}

