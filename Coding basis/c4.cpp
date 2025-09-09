
/*
// 4.1 arrayone.cpp -- small arrays of integers
#include <iostream>
using namespace std;
int main()
{
    int yams[3];
    yams[0] = 7;
    yams[1] = 8;
    yams[2] = 6;

    int yamcosts[3] = {20, 30 , 5};

    cout << yams[2] << yamcosts[0] <<endl;
    cout << 6 << 20 <<endl;
    return 0;
}

// 4.2 strings.cpp -- storing strings in an array
#include <iostream>
#include <cstring>
int main()
{
    using namespace std;
    const int Size = 15;
    char name1[Size];
    char name2[Size] = "C++owboy";
    cout << "Howdy! I'm " << name2;
    cout << "! What's your name?\n";
    cin >> name1;
    cout << "Well " << name1 << ", your name has ";
    cout << strlen(name1) << " letters and is stored\n";
    cout << "in an array of " << sizeof(name1) << "bytes.\n";
    cout << "Your initial is " << name1[0] << ".\n";
    name2[3] = '\0';
    cout << "Here are the first 3 characters of my name: ";
    cout << name2 << endl;
    return 0;
}

// 4.3&4.4 instr.cpp -- reading more than one string
#include <iostream>
using namespace std;
int main()
{
    const int Arsize = 20;
    char name[Arsize];
    char dessert[Arsize];
    cout << "Enter your name: \n";
    cin.getline(name, Arsize);
    cout << "Enter your favourite dessert: \n";
    cin.getline(dessert, Arsize);
    cout << "I have delicious " << dessert;
    cout << " for you, " << name << ".\n";
    return 0;
}

// 4.5 instr2.cpp -- reading more than one word with get() $ get()
#include <iostream>
int main()
{
    using namespace std;
    const int ArSize = 20;
    char a[ArSize];
    char b[ArSize];

    cout << "a = \n";
    cin.get(a, ArSize).get(); // 第二个get()可以读取并处理换行符
    cout << "b = \n";
    cin.get(b, ArSize).get();
    cout << "a is " << a << ", b is " << b << endl;
    return 0;
}

// 4.6 numstr.cpp -- following number input with line input
#include <iostream>
int main()
{
    using namespace std;
    int year;
    cout << "What year was your house bulit?\n";
    cin >> year;
    cin.get();
    cout << "What's its street address?\n";
    char address[80];
    cin.getline(address, 80);
    cout << "Year: " << year << endl;
    cout << "Address: " << address <<endl;
    return 0;
}
// 4.7 strtype.cpp -- using the C++ string class
#include <iostream>
#include <string>
int main()
{
    using namespace std;
    char charr1[20];
    char charr2[20] = "jaguar";
    string str1;
    string str2 = "panther";
    
    cout << "Enter a kind of feline: ";
    cin >> charr1;
    cout << "Another: ";
    cin >> charr2;
    cout << "Here are some felines:\n";
    cout << charr1 << " " << charr2 << " " << str1 << " " << str2 << endl;
    cout << "The third letter in " << charr2 << " is " << charr2[2] << endl;
    cout << "The third letter in " << str2 << " is " << str2[2] << endl;
    return 0;
}

#include <iostream>
#include <string>
int main()
{
    using namespace std;
    string s1 = "penguin";
    string s2, s3;
    s2 = s1;
    s3 = s1 + s2;
    s2 += " is cute.";
    int len = s3.size();
    cout << s1 << " " << s2 << " " << s3 << " " << len << endl;
    return 0;
}

#include <iostream>
using namespace std;
int main()
{
    int updates = 6;
    int* p_updates;
    p_updates = &updates;

    cout << "updates= " << updates << endl;
    cout << "&updates= " << &updates << endl;
    cout << "p_updates= " << p_updates << endl;
    cout << "*p_updates= " << *p_updates << endl;

    return 0;
}

#include <iostream>
using namespace std;
int main()
{
    int higgens = 5;
    int* pt = &higgens;

    cout << "Value of higgens = " << higgens;
    cout << "; Address of higgens = " << &higgens << endl;
    cout << "Value of *pt = " << *pt;
    cout << "; Value of pt = " << pt << endl;
    return 0;
}

#include <iostream>
using namespace std;
int main()
{
    int nights = 1001;
    int * pt = new int;
    *pt = 1001;

    cout << "nights value = ";
    cout << nights << ": location " << &nights << endl;
    cout << "int ";
    cout << "value = " << *pt << ": location = " << pt << endl;
    return 0;
}
*/

 
