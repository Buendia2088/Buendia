#include <iostream>
#include <string>
using namespace std;
    
int main()
{
    int p, e, i, d;
    int count = 0;
    string res;
    string d_store;
    string num;
    string temp;
    while(true)
    {
        cin >> p >> e >> i >> d;
        if(p + e + i + d != -4)
        {
            d_store += to_string(d);
            d_store += ' ';
            int k = 1;
            count++;
            while(true)
            {
                long M = 33 * k + i;
                if(((M - p) % 23) == 0 && ((M - e) % 28 == 0))
                {
                    res += to_string(M);
                    res += " ";
                    break;
                }
                k++;
            }
        }
            else
            {
                break;
            }
    }
    const int Size = count;
    count = 0;
    int output[2][Size] = {0};
    for(int i = 0; i < d_store.length(); i++)
    {
        temp += d_store[i];
        if(d_store[i] == ' ')
        {
            output[0][count] = stoi(temp);
            temp = "";
            count++;
        }
    }
    count = 0;
    for(int i = 0; i < res.length(); i++)
    {
        temp += res[i];
        if(res[i] == ' ')
        {
            output[1][count] = stoi(temp);
            temp = "";
            count++;
        }
    }
    for(int j = 0; j < Size; j++)
    {
        cout << "Case " << j + 1 << ": the next triple peak occurs in " << output[1][j] - output[0][j] << " days." << endl;
    }
    return 0;
}
