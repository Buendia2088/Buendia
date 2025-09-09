#include<iostream>
#include<fstream>
#include<cstring>
#include<map>

using namespace std;

string answer = "";
map<string, string> match1 = {
        {"int","int"},{"void","void"},{"if","if"},{"else","else"},
        {"return","return"},{"print","print"}
};
map<char,string> match2 = {
	{'(',"("},{')',")"},{'[',"["},{']',"]"},{'{',"{"},{'}',"}"},
        {';',";"},{',',","},{'=',"="},{'/',"DIV"},{'&',"AND"},{'|',"OR"},{'+',"+"}
};

void fail(){
	cout<<"error: invalid input"<<endl;
}

void end(string a, string b){
	string c = " " + b;
	answer += c;
}

bool case0(int &flag, string &temp,char ch){
	temp = "";
	if(ch >= '0' && ch <= '9'){
		temp += ch;
                flag = 1;
        }
        else if((ch>='a'&&ch<='z')||(ch >= 'A'&&ch<='Z')){
		flag = 2;
                temp += ch;
        }
	else{
                auto it = match2.find(ch);
		if(it != match2.end()){
			string a = " " + it->second;
			answer += a;
			flag = 0;
		}
		else if( ch == '<'||ch == '>' ){
			flag = 3;
			temp += ch;
		}
		else if( ch == ' '|| ch =='\n'|| ch == '\r' || ch == '\t'){
			flag = 0;
		}
		else{
			return false;
		}
	}
	return true;
}

bool scanner(ifstream &file){
	char ch;
	int flag = 0;
	string temp = "";
	cout<<"the original code is:"<<endl;
	while(file.get(ch)){
		cout<<ch;
		switch (flag){
			case 0:
				if(!case0(flag,temp,ch)) return false;
				break;
			case 1:
				if(ch>='0'&&ch<='9') temp += ch;
				else {
					end(temp,"i");
					if(!case0(flag,temp,ch)) return false;
				}
				break;
			case 2:
				if((ch>='a'&&ch<='z')||(ch >= 'A'&&ch<='Z')||(ch>='0'&&ch<='9')) temp += ch;
				else{
					auto it = match1.find(temp);
					if(it != match1.end()) end("_", it->second);
					else end(temp, "d");
					if(!case0(flag,temp,ch)) return false;
				}
				break;
			case 3:
				if(ch == '=') {
					temp += ch;
					end(temp, "ROP");
				}
				else{
					end(temp, "ROP");
					if(!case0(flag,temp,ch)) return false;
				}
				break;
		}
	}
	return true;
}


int main () {
	ifstream file("code.txt");
	if(!file.is_open()){
		cout<<"error: cannot open the file"<<endl;
		return 1;
	}
	if(!scanner(file)) return 1;
	cout<<endl<<"the lexical analysis result is:"<<endl;
	cout<<answer<<endl;
}
