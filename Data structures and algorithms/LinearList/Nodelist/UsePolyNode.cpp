#include <iostream>
#include "PolyNode.h"

using namespace std;


int main()
{
    PolyNode t1(100);
    PolyNode t2(100);
    t1.create(3);
    t2.create(2);
    PolyNode res1 = t1.plus(&t2);
    PolyNode res2 = t1.minus(&t2);
    cout << "t1 is" << endl;
    t1.print();
    cout << "t2 is" << endl;
    t2.print();
    cout << "t1 plus t2 is" << endl;
    res1.print();
    cout << "t1 minus t2 is" << endl;
    res2.print();
    return 0;

}