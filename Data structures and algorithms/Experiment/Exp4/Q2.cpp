#include <iostream>
#include <string>
#include <vector>
using namespace std;

struct student
{
    string id;
    string name;
    int grade;
    student() : id("0"), name("Zhang San"), grade(114514) {}
    student(const student& s)
    {
        id = s.id;
        name = s.name;
        grade = s.grade;
    }
};

void SWAP(student& a, student& b)
{
    student temp = a;
    a = b;
    b = temp;
}

int main()
{
    /*
    INPUT();
    SORT();
    PRINT();
    */
   vector<student> student_infomation;
    int n;
    cin >> n;
    int t = n;
    while(t--)
    {
        student temp;
        cin >> temp.id >> temp.name >> temp.grade;
        student_infomation.push_back(temp);
    }
    int flag = 0;
    int mark = n-1;
    while(flag == 0)
    {
        flag = 1;
        for(int i = 0; i < mark; i++)
        {
            if(student_infomation[i].grade < student_infomation[i+1].grade)
            {
                swap(student_infomation[i], student_infomation[i+1]);
                flag = 0;
            }
        }
        mark++;
    }
    for(int i = 0; i < student_infomation.size(); i++)
    {
        if(i != 0)
        {
            cout <<endl;
        }
        cout << student_infomation[i].id << " " << student_infomation[i].name << " " << student_infomation[i].grade;
    }
    return 0;
}