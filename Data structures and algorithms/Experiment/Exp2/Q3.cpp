#include <iostream>
#include <string>
using namespace std;

int main()
{
    string input;
    getline(cin, input);
    int n = input.length();
    int n1 = (n + 2) / 3;
    int n2 = n + 2 - 2 * n1;
    int n3 = n1;
    for(int i = 0; i < n1 - 1; i++)
    {
        cout << input[i];
        for(int j = 0; j < n2 - 2; j++)
        {
            cout << " ";
        }
        cout << input[input.length() - 1 - i];
        cout << endl;
    }
    for(int i = n1 - 1; i < n1 + n2 - 1; i++)
    {
        cout << input[i];
    }
    return 0;
}