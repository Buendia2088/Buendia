
/*
//2.1 myfirst cpp.cpp -- displays a message
#include <iostream>
using namespace std;
int main()
{
	cout << "hello world";
	cout << endl;
	cout << "hello C++";
	return 0;
}


//2.2 carrots.cpp -- use and displays a variable
#include <iostream>
using namespace std;
int main()
{
	int carrots;
	carrots = 25;
	cout << "I have";
	cout << carrots;
	cout << "carrots";
	cout << endl;
	carrots = carrots - 1;
	cout << "Crunch,crunch.Now I have" << carrots << "carrots." <<endl;
	return 0;
}


//2.3 getinfo.cpp -- input and output
#include <iostream>
using namespace std;
int main()
{
	int carrots;
	cout << "How many carrots do you have?" << endl;
	cin >> carrots;
	cout << "Here are two more." << endl;
	carrots = carrots + 2;
	cout << "Now you have " << carrots << " carrots." << endl;
	return 0;
}


//2.4 sqrt.cpp -- using the sprt() function
#include <iostream>
#include <cmath>
using namespace std;
int main()
{
	double area;
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

//ourfucc.cpp -- defining your own function
#include <iostream>
void cy(int);
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
*/







