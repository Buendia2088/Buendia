#include <iostream>
using namespace std;

class que
{
private:
    int Maxsize;
    int Cursize;
    int* queptr;
public:
    que(int s = 100);
    ~que();
    void insert(int n);
    void print();
    void pop0_1();
};

que::que(int s) : Maxsize(s), Cursize(0)
{
    queptr = new int[Maxsize];
}

que::~que()
{
    delete[] queptr;
}

void que::insert(int n)
{
    queptr[Cursize] = n;
    Cursize++;
}

void que::print()
{
    if(Cursize == 0)
    {
        return;
    }
    else
    {
        cout << queptr[0];
    }
    for(int i = 1; i < Cursize; i++)
    {
        cout << " " << queptr[i];
    }
}

void que::pop0_1()
{
    for(int i = 0; i < Cursize - 1; i++)
    {
        queptr[i] = queptr[i+1];
    }
    Cursize--;

    for(int i = 0; i < Cursize - 1; i++)
    {
        queptr[i] = queptr[i+1];
    }
    Cursize--;
}

int main()
{
    que test;
    int n;
    cin >> n;
    while(n--)
    {
        int temp;
        cin >> temp;
        test.insert(temp);
    }
    test.pop0_1();
    test.insert(11);
    test.insert(12);
    test.print();
    return 0;
}