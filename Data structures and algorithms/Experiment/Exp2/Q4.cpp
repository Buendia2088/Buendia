#include <iostream>
#include <string>
#include <vector>
using namespace std;

struct number
{
    int inte;
    int mole;
    int deno;
};

int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

number plus_n(number n1, number n2)
{
    number res;
    res.deno = n1.deno * n2.deno;
    res.mole = n1.mole * n2.deno + n2.mole * n1.deno;
    res.inte = n1.inte + n2.inte;
    if(res.mole > 0 && res.mole >= res.deno)
    {
        int n = res.mole / res.deno;
        res.inte += n;
        res.mole -= n * res.deno;
    }
    if(res.mole < 0)
    {
        if(res.mole < -res.deno)
        {
            res.inte -= 2;
            res.mole += 2 * res.deno;
        }
        else if(res.mole == -res.deno)
        {
            res.mole = 0;
            res.deno -= 1;
        }
        else
        {
            res.inte -= 1;
            res.mole += res.deno;
        }
    }
    int GCD = gcd(res.mole, res.deno);
    res.mole /= GCD;
    res.deno /= GCD;
    return res;
}


int main()
{

    vector<number> vec_n;
    int n;
    cin >> n;
    while(n--)
    {
        number num;
        string input;
        num.inte = 0;
        cin >> input;
        string temp;
        int pos = -1;
        for(int i = 0; i < input.length(); i++)
        {
            if(input[i] != '/')
                temp += input[i];
            else
            {
                num.mole = stoi(temp);
                temp = "";
                pos = i + 1;
            }
        }
        num.deno = stoi(input.substr(pos));
        vec_n.push_back(num);
    }
    number res;
    res.inte = res.mole = 0;
    res.deno = 1;
    for(int i = 0; i < vec_n.size(); i++)
    {
        res = plus_n(res, vec_n[i]);
    }
    if(res.inte == 0)
    {
        if(res.mole == 0)
        {
            cout << 0;
        }
        else
        {
            cout << res.mole << '/' << res.deno;
        }
    }
    else if(res.mole == 0)
    {
        cout << res.inte;
    }
    else
    {
        cout << res.inte << " " << res.mole << '/' << res.deno;
    }
    /*
    number n1, n2;
    cin >> n1.inte >> n1.mole >> n1.deno >> n2.inte >> n2.mole >> n2.deno;
    number res = plus_n(n1, n2);
    cout << res.inte << " " << res.mole << " " << res.deno << endl;
    */
   return 0;
}