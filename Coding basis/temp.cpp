#include <iostream>
using namespace std;

struct result
{
    int a;
    int b;
};

result judge(result x)
{
    x.a = 0;
    char word[255] = {0};
    cin.getline(word, 255);
    for (int i = 0; i < 255; i++)
    {
	if (((word[i] == ' ') || (x.b == 0)) && (word[i+1] == 'd') && (word[i+2] == 'o') && (word[i+3] == 'n') && (word[i+4] == 'e'))
	{
    x.a = -2;
    break;
	}
    if ((word[i] != ' ') && (word[i-1] == ' '))
    {
    x.b++;
    }
    }    
    return x;    
}

int main()
{
    result Result;
    Result.a = -1;
    Result.b = 0;
    while(Result.a != -2)
    {
        Result = judge(Result);
    }
    cout << Result.b;
    return 0;
}
    
    


