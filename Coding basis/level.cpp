#include <iostream>
#include <string>
using namespace std;

struct stu
{
    int id;
    int score;
    int ex_score;

    int general_score()
    {
        return score*7 + ex_score*3;
    }

    int total_score()
    {
        return score + ex_score;
    }
};

void judge(stu a)
{
    if(a.general_score() >= 800 && a.total_score() >= 140)
    {
        cout << "Excellent" << endl;
    }
        else
        {
            cout << "Not excellent" << endl;
        }
}
    
int main()
{
    int N;
    cin >> N;
    const int Size = N;
    stu stu_set[Size];
    for(int i = 0; i < N; i++)
    {
        cin >> stu_set[i].id >> stu_set[i].score >> stu_set[i].ex_score;
    }
    for(int i = 0; i < N; i++)
    {   
        judge(stu_set[i]);
    }
    return 0;
}

