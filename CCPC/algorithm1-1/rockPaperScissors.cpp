#include <iostream>
#include <vector>
using namespace std;

int main()
{
    vector<vector<int>> matrix = {
        {  0, -1,  1,  1, -1 },  // 剪刀(0): 平, 输石头, 赢布, 赢蜥蜴人, 输斯波克
        {  1,  0, -1,  1, -1 },  // 石头(1): 赢剪刀, 平, 输布, 赢蜥蜴人, 输斯波克
        { -1,  1,  0, -1,  1 },  // 布(2): 输剪刀, 赢石头, 平, 输蜥蜴人, 赢斯波克
        { -1, -1,  1,  0,  1 },  // 蜥蜴人(3): 输剪刀, 输石头, 赢布, 平, 赢斯波克
        {  1,  1, -1, -1,  0 }   // 斯波克(4): 赢剪刀, 赢石头, 输布, 输蜥蜴人, 平
    };
    int n, na, nb;
    cin >> n >> na >> nb;
    vector<int> seqA(na), seqB(nb);
    for(int i = 0; i < na; i++)
    {
        cin >> seqA[i];
    }
    for(int i = 0; i < nb; i++)
    {
        cin >> seqB[i];
    }
    int indexA = 0; int indexB = 0;
    int scoreA = 0; int scoreB = 0;
    while(n--)
    {
        if(matrix[seqA[indexA]][seqB[indexB]] == -1) scoreB++;
        if(matrix[seqA[indexA]][seqB[indexB]] == 1) scoreA++;
        indexA = (indexA + 1) % na;
        indexB = (indexB + 1) % nb;
    }
    cout << scoreA << " " << scoreB;
    return 0;
}