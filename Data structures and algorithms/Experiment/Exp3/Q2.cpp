#include <iostream>
#include <stack>
#include <string>
using namespace std;

int main()
{
    stack<char> sta;
    string input;
    getline(cin, input);
    for(int i = 0; i < input.length(); i++)
    {
        if(input[i] == '(')
        {
            sta.push(input[i]);
        }
        if(input[i] == ')')
        {
            if(sta.empty())
            {
                cout << "括号不匹配！";
                return 0;
            }
            else
            {
                sta.pop();
            }
        }
    }
    if(sta.empty())
    {
        cout << "括号匹配！";
    }
    else
    {
        cout << "括号不匹配！";
    }
    return 0;
}