#include <iostream>
using namespace std;

class strlist
{
private:
    char* listptr;
public:
    strlist();
    ~strlist();
    void create();
    int count();
    void insert(char c);
    char* find(char c);
    void delete_list(char c);
    void print();
};

strlist::strlist()
{
    listptr = new char;
}

strlist::~strlist()
{
    delete listptr;
}

void strlist::create()
{
    cin.getline(listptr, 100);
}

int strlist::count()
{
    int res = 0;
    char* cur = listptr;
    while(*cur != '\0')
    {
        res++;
        cur++;
    }
    return res;
}

void strlist::insert(char c)
{
    char* cur = listptr;
    while(*cur != '\0')
    {
        cur++;
    }
    *cur = c;
    cur++;
    *cur = '\0';
}

void strlist::print()
{
    char* cur = listptr;
    while(*cur != '\0')
    {
        cout << *cur;
        cur++;
    }
}

char* strlist::find(char c)
{
    char* cur = listptr;
    while(*cur != '\0')
    {
        if(*cur == c)
        {
            return cur;
        }
        cur++;
    }
    return NULL;
}

void strlist::delete_list(char c)
{
    char* cur = find(c);
    if(cur == NULL)
    {
        return;
    }
    while(*(cur + 1) != '\0')
    {
        *cur = *(cur + 1);
        cur++;
    }
    *cur = '\0';
}

int main()
{
    strlist test;
    test.create();
    test.print();
    cout << endl;
    cout << test.count();
    cout << endl;
    char c;
    cin >> c;
    while(test.find(c) != NULL)
    {
        test.delete_list(c);
    }
    test.print();
    cout << endl;
    cout << test.count();
}