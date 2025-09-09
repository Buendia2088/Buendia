#include <iostream>
#include <string>
#include <vector>
using namespace std;
int MAX = 10000;
int count = 0;
void move(pair<int, int>& p, int& dire, vector<vector<char>>& a)
{
    int x = p.first;
    int y = p.second;
    if(dire == 0) 
    {
        if(a[x-1][y] != '*')
        {
            x--;
        }
        else
        {
            dire = (dire + 1) % 4;
        }
    }
    else if(dire == 1) 
    {
        if(a[x][y+1] != '*')
        {
            y++;
        }
        else
        {
            dire = (dire + 1) % 4;
        }
    }
    else if(dire == 2) 
    {
        if(a[x+1][y] != '*')
        {
            x++;
        }
        else
        {
            dire = (dire + 1) % 4;
        }
    }
    else 
    {
        //cout << count << " " << "(" << x << "," << y << ")" << endl;
        if(a[x][y-1] != '*')
        {
            y--;
        }
        else
        {
            dire = (dire + 1) % 4;
        }
    }
    p.first = x;
    p.second = y;
}   

int main()
{
    vector<vector<char>> a(12, vector<char>(12, ' ')); 
    pair<int, int> farmP;
    pair<int, int> cowP;
    int farmD = 0;
    int cowD = 0;
    for (int i = 0; i <= 11; i += 11) 
    {
        for (int j = 0; j < 12; j++) 
        { 
            a[i][j] = '*';
        }
    }
    for (int j = 0; j <= 11; j += 11) 
    {
        for (int i = 0; i < 12; i++) 
        {
            a[i][j] = '*';
        }
    }
    for (int i = 1; i <= 10; i++) 
    {
        for (int j = 1; j <= 10; j++) 
        {
            cin >> a[i][j];
            if (a[i][j] == 'F') 
            {
                farmP = {i, j};
                a[i][j] = '.';
            } 
            else if (a[i][j] == 'C') 
            {
                cowP = {i, j};
                a[i][j] = '.';
            }
        }
    }
    int n = MAX;

    while(n--)
    {
        count++;
        move(farmP, farmD, a);
        move(cowP, cowD, a);
        if(farmP == cowP)
        {
            cout << count;
            return 0;
        }
    }
    cout << 0;
    return 0;
}