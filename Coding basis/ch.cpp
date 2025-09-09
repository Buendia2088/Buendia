#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
using namespace std;

void decimalToBase(int decimalNumber, int base, vector<string>& vec) 
{   
    while (decimalNumber > 0) 
    {
        int remainder = decimalNumber % base;
        vec.push_back(to_string(remainder));
        decimalNumber /= base;
    }
}

int main() 
{
    unsigned long long decimalNumber;
    unsigned long long base;
    cin >> decimalNumber >> base;
    if(decimalNumber == 0)
    {
        cout << "Yes" << endl << 0;
        return 0;
    }
    vector<string> res_r;
    decimalToBase(decimalNumber, base, res_r);
    vector<string> res;
    for(int i = res_r.size() - 1; i >= 0; i--)
    {
        res.push_back(res_r[i]);
    }
    if(res == res_r)
    {
        cout << "Yes";
    }
    else
    {
        cout << "No";
    }
    cout << endl;
    
    if(res.size() > 0)
    {
        cout << res[0];
        for(int i = 1; i < res.size(); i++)
        {
            cout << " " << res[i];
        }
    }
    return 0;
}

