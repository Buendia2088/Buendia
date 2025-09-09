#include<iostream>
#include<string>
using namespace std;
string unzip()
{
	int repeat_times;
	string answer = "";
    string temp;
    char c;
    while(cin >> c)
    {
        if(c == '?')
        {
            break;
        }
        if(c >= 'A' && c <= 'Z')
        {
            answer += c;
        }
        else if(c == '[')
        {
            cin >> repeat_times;
            temp = unzip();
            while(repeat_times--) answer += temp;
        }
        else
        {
            return answer;
        }
    }
    return answer;
}
int main()
{
	cout << unzip(); 
	return 0;
}