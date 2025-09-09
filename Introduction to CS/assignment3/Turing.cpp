/**/

using namespace std;

#include<iostream>
#include<cstring>
#include<iomanip>

#define Gap 8 //Gap between Rule output
#define Exlen 0 //Extra Length of tape from two sides

class Turing_Rule
{
	friend class Turing;
	
	protected:
	string status;
	char current;
	char write;
	int move;
	string next;

	public:
	Turing_Rule():status("NULL"),current('\0'),write('\0'),move(0),next("NULL") {};
	Turing_Rule(string name, char cur, char wri, int mov, string nex):status(name),current(cur),write(wri),move(mov),next(nex) {};

	void Output();
	void ModifyRule(string name, char cur, char wri, int mov, string nex)/////Alternative of ' ' here 
	{
		if(cur=='@')
			cur = ' ';
		if(wri=='@')
			wri = ' ';
		status = name; current = cur; write = wri; move = mov; next = nex;
	};

};

void Turing_Rule::Output()
{
	cout << setw(Gap) << status << setw(Gap) << current << setw(Gap) << write << setw(Gap) << setw(Gap) << move << setw(Gap) << next << '\n'; 
}


class Turing
{
	private:
		string current_status;
		int current_position;
		char * tape;
		int commandsum;
		Turing_Rule *command;
	public:
		~Turing();
		Turing();
		void Insert(const char* input);
		Turing & operator = (Turing const & tur);
		void PrintRule();
		bool AddRule();
		void CreateRule();
		void EditRule();
		void DeleteRule();
		void PrintCurrent();
		bool Process();
		void Modify(string cs,int cp) { current_status = cs; current_position = cp; };
		void Run();
};

Turing::Turing()
{
	current_status = "NULL"; current_position = -1; tape = NULL; commandsum = 0;
	command = NULL;
}

Turing::~Turing()
{
	delete [] tape;
	delete [] command;
}

void Turing::Insert(const char* input)
{
	int len = strlen(input);
	if(tape!=NULL)
	{
		delete [] tape;
	}
	tape = new char[len+1+2*Exlen];
	int i = 0;
	while(i<len+1+2*Exlen)
	{
		tape[i] = ' ';
		i++;
	}
	strcpy(tape+Exlen,input);
}

Turing & Turing::operator=(Turing const & tur)
{
	current_status = tur.current_status;
	current_position = tur.current_position;
	commandsum = tur.commandsum;
	if(tape!=NULL)
		delete [] tape;
	if(command!=NULL)
		delete [] command;
	int len = strlen(tur.tape);
	tape = new char[len+1];
	strcpy(tape,tur.tape);
	command = new Turing_Rule[commandsum];
	int i = 0;
	while(i<commandsum)
	{
		command[i] = tur.command[i];
		i++;
	}
	return *this;
}

void Turing::PrintRule()
{
	int i = 0;
	cout << "No." << setw(Gap) << "status" << setw(Gap) << "current" << setw(Gap) << "write" << setw(Gap) << "move" << setw(Gap) << "next\n";
	while(i<commandsum)
	{
		cout << " " << i+1 << " ";
		command[i].Output();
		i++;
	}
}

bool Turing::AddRule()
{
	cout << "\nPlease input status, current, to-write, move and next\n" << "input -1 to cancel        input @ as space' ' \n";
        string s,n;
        char c,w;
        int m;
        cin >> s; 
	if(s=="-1")
	{
		cout << "RuleAdding eliminated\n\n";
		return false;
	}
	cin >> c >> w >> m >> n;
	int i = commandsum;
	int j = 0;
	commandsum++;
	Turing_Rule *extended = new Turing_Rule[commandsum];
	while(j<i)
	{
		extended[j] = command[j];
		j++;
	}
	delete [] command;
	command = extended;
	command[i].ModifyRule(s,c,w,m,n);
	cout << "\nAdded Rule: \n";
	cout << setw(Gap) << "status" << setw(Gap) << "current" << setw(Gap) << "write" << setw(Gap) << "move" << setw(Gap) << "next\n";
	command[i].Output();
	cout << "\nCurrent commandsum total: " << commandsum << "\n\n";
	return true;
}

void Turing::CreateRule()
{
	if(commandsum)
	{
		delete [] command;
		commandsum = 0;
	}
	cout << "\n\nStarting Creating Rules:\n";
	while(this->AddRule())
	{
	}
	cout << "Rules Created:\n";
	this->PrintRule();
	cout << "\n\n";
}

void Turing::EditRule()
{
	cout << "\n\nStarting Editing Rules:\n\n";
	while(1)
	{
		cout << "\nCurrent Rules:\n";
		this->PrintRule();
		cout << "Input the number of Rule you want to edit:\n";
		cout << "Input -1 to eliminate\n";
		int input;
		cin >> input;
		if(input==-1)
		{
			cout << "Edit elimated\n";
			return;
		}
		if(input>commandsum)
		{
			cout << "Wrong input--Too large\n";
			continue;
		}
		cout << "Please input status, current, to-write, move and next\n" << "input -1 to cancel     input @ as space' '\n";
        	string s,n;
        	char c,w;
        	int m;
        	cin >> s >> c >> w >> m >> n;
		command[input-1].ModifyRule(s,c,w,m,n);
		cout << "Rule edited: No." << input << "\n\n";
	}
	cout << "\n\n"; 
}

void Turing::DeleteRule()
{
	cout << "\n\nStarting Deleting Rules:\n\n";
	while(1)
	{
		cout << "\nCurrent Rules:\n";
                this->PrintRule();
                cout << "Input the number of Rule you want to delete:\n";
                cout << "Input -1 to eliminate\n";
                int input;
                cin >> input;
                if(input==-1)
                {
                        cout << "Delete elimated\n";
                        return;
                }
                if(input>commandsum)
                {
                        cout << "Wrong input--Too large\n";
                        continue;
                }
		if(input==commandsum)
		{
			commandsum--;
			cout << "Rule Deleted: No." << input << "\n\n";
			continue;
		}
		int i = input-1;
		while(i<commandsum-1)
		{
			command[i] = command[i+1];
			i++;
		}
		commandsum--;
		cout << "Rule Deleted: No. " << input << "\n\n"; 
	}
	cout << "\n\n";
}

void Turing::PrintCurrent()
{
	cout << "\n\n";
	int i = 0;
	while(i<current_position-1)
	{
                cout << "  ";
                i++;
        }
	cout << "  |" << current_status << "|\n";
	i = 0;
	while(i<current_position)
	{
		cout << "  ";
		i++;
	}
	cout << " V\n";
	i = 0;
	while(tape[i]!='\0')
	{
		cout << ' ' << tape[i];
		i++;
	}
	cout << "\n\n";
}

bool Turing::Process()
{
	int i = 0;
	while(i<commandsum)
	{
		if(current_status==command[i].status)
		{
			if((tape[current_position]==command[i].current)||command[i].current=='\\')
			{
				if(command[i].write!='\\')
					tape[current_position] = command[i].write;
				current_position += command[i].move;
				current_status = command[i].next;
				break;
			}
		}
		i++;
	}
	this->PrintCurrent();
	if(current_status == "Stop")
		return false;
	return true;
}



void Turing::Run()
{
	if(current_position == -1)
	{
		this->CreateRule();
		this->EditRule();
		this->DeleteRule();
	}
	cout << "Current Tape:\n\n" << tape << "\n\n";
	cout << "Please input initial status and initial position\n\nPosition starts from 0\n\n";
	string is; int ip;
	cin >> is >> ip;
	this->Modify(is,ip);
	int i = 0;
	while(this->Process())
	{
		cout << "Step "<< ++i << " Completed\n";
		/*if(i>=250)
			break;*/
	}
	cout << "Step " << ++i << " Completed\n";
}

int main()
{
	Turing Tur;
	char input[200];
	cout << "\nPlease input tape(Attention:Leave ENOUGH spaces in front and behind)\n";
	cin.getline(input,200);
	int len = strlen(input);
	char * tape = new char[len+1];
	strcpy(tape,input);
	Tur.Insert(tape);
	Tur.PrintCurrent();
	Tur.Run();
}
