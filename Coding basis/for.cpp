#include <iostream>
using namespace std;
int main()
{
    int a = 0;
    cout << "Enter a nonnegative integer: ";
    cin >> a;
    for (int i = 0; i <= a; i += 2)
    {
        cout << i << endl;
    }
    return 0;
}
