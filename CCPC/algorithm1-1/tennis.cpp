#include<iostream>
#include<cmath>
#include <cctype>
using namespace std;

void countScore(int max, char* input, int size)
{
    int wCount = 0;
    int lCount = 0;
    for(int i = 0; i < size; i++)
    {
        if(input[i] == 'W')
        {
            wCount++;
        }
        else 
        {
            lCount++;
        }
        if((wCount >= max || lCount >= max ) && abs(wCount - lCount) >= 2)
        {
            cout << wCount << ":" << lCount << endl;
            wCount = lCount = 0;
        }
    }
    //if(!(lCount == 0 && wCount == 0)) 
    cout << wCount << ":" << lCount << endl;
}

int main()
{
    char ch;
    char input[64000];
    int count = 0;
    while (cin.get(ch)) 
    {
        if (isspace(ch)) 
        {
            continue;
        }
        if (ch == 'E') 
        {
            break;
        }
        input[count++] = ch;
    }
    countScore(11, input, count);
    cout << endl;
    countScore(21, input, count);
    return 0;
}