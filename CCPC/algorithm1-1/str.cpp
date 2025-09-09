#include <iostream>
#include <string>
using namespace std;

int main()
{
    int p1, p2, p3;
    cin >> p1 >> p2 >> p3;
    string input;
    cin >> input;
    bool* mark = new bool[input.length()];
    for(int i = 0; i < input.length(); i++)
    {
        mark[i] = false;
    }
    for(int i = 0; i < input.length(); i++)
    {
        if(input[i] == '-')
        {
            if(i == 0 || i == input.length() - 1) 
            {
                cout << "-";
                continue;
            }
            char a = input[i-1];
            char b = input[i+1];
            if(a == '-' || b == '-')
            {
                cout << "-";
                continue;
            }
            mark[i-1] = true;
            if(a + 1 == b)
            {
                cout << a << b;
                continue;
            } 
            if(a >= b)
            {
                cout << '-';
                continue;
            } 
            if(a <= '9' && b >= 'A') 
            {
                cout << '-';
                continue;
            }
            
            for(char j = a + 1; j < b; j++)
            {
                for(int p = 0; p < p2; p++)
                {
                    if(p1 == 3) cout << '*';
                    else if(p3 == 1) 
                    {
                        if(a <= '9') cout << j;
                        else if(p1 == 1) cout << static_cast<char>(tolower(j));
                        else cout << static_cast<char>(toupper(j));
                        
                    }
                    else 
                    {
                        if(a <= '9') cout << a + b - j - '0';
                        else if(p1 == 1) cout << static_cast<char>(tolower(a + b - j));
                        else cout << static_cast<char>(toupper(a + b - j));
                    }
                }
            }
        }
        else if(!mark[i]) cout << input[i];
    }
    return 0;
}