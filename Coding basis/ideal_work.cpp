 #include <iostream>
 #include <cmath>
 #include <sstream>
 #include <iomanip>
 #include <string>
 using namespace std;
 int main()
 {
    int a;
    int b = 0;
    int c = 0;
    stringstream ss;
    string output;
    cout << "    " << 2 << "    " << 3 << "    " << 5 << "    " << 7 << "   " << 11 << endl;
    for(int i = 13; i <= 1000; i++)
    {
        a = sqrt(i);
        for(int j = 1; j <= (a + 1); j++)
        {
            if((i % j) == 0)
            {
                b++;
            }
        }
        if(b == 1)
        {
            c++;
            if((b == 1) && (c == 5))
            {
                ss << setw(5) << i;
                ss << '\n';
                c = 0;
            }
                else if((b == 1) && (c != 5))
                {
                    ss << setw(5) << i;
                }
        }
        b = 0;
    }
    cout << ss.str() << endl;
    return 0;
}
