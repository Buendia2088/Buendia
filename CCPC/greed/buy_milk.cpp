#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

struct farmer
{
    int num;
    int price;
    farmer(int n, int p) : num(n), price(p) {}
};

bool cmp(farmer a, farmer b)
{
    return a.price < b.price;
}

long long buy(int n, int m, std::vector<farmer> &farmers)
{
    long long total = 0;
    sort(farmers.begin(), farmers.end(), cmp);
    for(int i = 0; i < m; i++)
    {
        if(n > farmers[i].num)
        {
            n -= farmers[i].num;
            total += farmers[i].num * farmers[i].price;
        }
        else
        {
            total += n * farmers[i].price;
            break;
        }
    }
    return total;
}

int main()
{
    int n, m;
    cin >> n >> m;
    vector<farmer> farmers;
    for(int i = 0; i < m; i++)
    {
        int num, price;
        cin >> price >> num;
        farmers.push_back(farmer(num, price));
    }
    cout << buy(n, m, farmers);
    return 0;
}