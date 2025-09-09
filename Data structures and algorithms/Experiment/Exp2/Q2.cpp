#include <iostream>
#include <string>
using namespace std;

string find(string str)
{
    str += " ";
    string res = "";
    string temp = "";

    for(int i = 0; i < str.length(); i++)
    {
        if(str[i] != ' ')
        {
            temp += str[i];
        }
        else
        {
            if(temp.size() > res.size())
            {
                res = temp;
            }
            temp = "";
        }     
    }
    return res;
}

int main()
{
    string input;
    getline(cin, input);
    cout << find(input);
}