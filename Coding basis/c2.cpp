
/*
//2.1 myfirst cpp.cpp -- displays a message
#include <iostream>
using namespace std;
int main()
{
	cout << "hello world";
	cout << endl; // 换行符
	cout << "hello C++";
	return 0;
}


//2.2 carrots.cpp -- use and displays a variable
#include <iostream>
using namespace std;
int main()
{
	int carrots; // 声明变量
	carrots = 25;
	cout << "I have";
	cout << carrots;
	cout << "carrots";
	cout << endl;
	carrots = carrots - 1;
	cout << "Crunch,crunch.Now I have" << carrots << "carrots." <<endl; // 注意空格
	return 0;
}


//2.3 getinfo.cpp -- input and output
#include <iostream>
using namespace std;
int main()
{
	int carrots;
	cout << "How many carrots do you have?" << endl;
	cin >> carrots; // 输入
	cout << "Here are two more." << endl;
	carrots = carrots + 2;
	cout << "Now you have " << carrots << " carrots." << endl;
	return 0;
}


//2.4 sqrt.cpp -- using the sprt() function
#include <iostream>
#include <cmath> // 引入<cmath>头文件，以在C++中使用数学函数和数学常量
using namespace std;
int main()
{
	double area; // double用来声明双精度浮点型变量
    cout << "Enter the floor area, in square feet, of your home: ";
    cin >> area; 
    double side;
    side = sqrt(area);
    cout << "That`s the equivalent of a square " << side
         << " feet to the side" << endl;
    return 0;
}/
    
//homework2 9.7.cpp -- exchanges variables
#include <iostream>
using namespace std;
int main()
{
    int a, b, temp;
    a = b = temp = 0;
    cout << "a=";
    cin >> a;
    cout << "b=";
    cin >> b;

    temp = a;
    a = b;
    b = temp;
    cout << "a=" << a << " b=" << b << endl;
    return 0;
}

//2.5 ourfuct.cpp -- defining your own function
#include <iostream>
void cy(int); // void说明cy()函数不返回数值，(int)说明cy函数输入整形数据
using namespace std;
int main()
{
    int a;
    cout << "Pick an integer more than 1.";
    cin >> a;
    cy(a);
    cout << "Done!" <<endl;
    return 0;
}
void cy(int n)
{
    cout << "I have " << n << " apples." << endl;
}

//2.6 convert1.cpp -- converts stone to pounds
#include <iostream>
using namespace std;
int stonetolb(int);
int main()
{
   int stone;
   cout << "Enter the weight in stone: ";
   cin >> stone;
   int pounds = stonetolb(stone);
   cout << stone << " stone = ";
   cout << pounds << " pounds" << endl;
   return 0;
}

int stonetolb(int sts) // 括号中的sts表示stonetolb函数接受一个整数参数sts
{
    return 14 * sts;
}


//2.6 convert2.cpp -- converts C to F
#include <iostream>
using namespace std;
double CF(double);
int main()
{
   double C;
   cout << "C = ";
   cin >> C;
   cout << endl;
   double F;
   F = CF(C);
   cout << "F = " << F << endl;
   return 0;
}

double CF(double a)
{
    double b;
    b = a * 9 / 5 + 32;
    return b;
}


//2.7 -- 3
#include <iostream>
#include <string> // 引入字符串相关的库函数和类，方便对字符串操作
using namespace std;

void generateFirstTwoLine(string line12)
{
    cout << line12 << endl;
}
void generateLastTwoLine(string line34)
{
    cout << line34 << endl;
}

int main()
{
    string line12, line34;

    cout << "Line1 and line2 are: ";
    cin  >> line12;

    cout << "Line3 and line4 are: ";
    cin  >> line34;

    cout << endl;

    generateFirstTwoLine(line12); // 注意此处调用函数的语法，直接打出函数即可
    generateFirstTwoLine(line12);
    generateLastTwoLine(line34);
    generateLastTwoLine(line34);

    return 0;
}


//2.7 -- 4
#include <iostream>
using namespace std;
int main()
{
    int age;
    cout << "Enter your age: ";
    cin >> age;
    int months;
    months = age * 12;
    cout << "There are " << months << " months.";
    return 0;
}
*/
//2.7 -- 7
#include <iostream>
#include <string>
using namespace std;

void print(int a, int b)
{
    cout << "Time: " << a << ":" << b << endl;
}

int main()
{
    int a, b;
    cout << "Enter the number of hours: ";
    cin >> a;
    cout << "Enter the number of minutes: ";
    cin >> b;
    print(a, b);
    return 0;
}














