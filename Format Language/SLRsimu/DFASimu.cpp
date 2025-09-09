#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <map>
#include <sstream>
#include <fstream>
#include <algorithm>

using namespace std;

// �������ʽ�ṹ�壬��ʾ�ķ�����
struct Production {
    string left;  // ����ʽ���󲿣����ս����
    vector<string> right;  // ����ʽ���Ҳ����������У�
};

// ����Action�ṹ�壬������ʾ�ƽ�����Լ��ת�ƵȲ���
struct Action {
    enum Type { SHIFT, REDUCE, GOTO, ACC, ERROR } type;  // ��������
    int value;  // ������״̬��Ż��Լ����ı��
    Action() : type(ERROR), value(-1) {}  // Ĭ�Ϲ��캯������ʾ��Ч����
};

vector<Production> productions;  // �洢�ķ�����
vector<string> terminals;       // �洢�ս��
vector<string> non_terminals;   // �洢���ս��
map<int, map<string, Action>> action_table;  // �洢״̬�ͷ��Ŷ�Ӧ�Ķ������ƽ�����Լ�ȣ�
map<int, map<string, int>> goto_table;       // �洢״̬�ͷ��ս����Ӧ��ת��״̬

// �ָ��ַ�����������ָ���ָ�������ַ�����ȥ���ո񲢷��طָ����ַ�������
vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        // ȥ��ÿ����Ԫ���еĿո�
        token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

// ���������ͣ�����0��ʾ���ս��������1��ʾ�Ƿ��ս��������-1��ʾδ֪����
int check_symbol_type(const string& symbol) {
    if (find(terminals.begin(), terminals.end(), symbol) != terminals.end()) {
        return 0;  // ���ս��
    }
    if (find(non_terminals.begin(), non_terminals.end(), symbol) != non_terminals.end()) {
        return 1;  // �Ƿ��ս��
    }
    return -1;  // �Ȳ����ս��Ҳ���Ƿ��ս��
}

// ���������ַ������������ַ�������Ϊ��Ӧ�� Action ����
Action parse_action(const string &s) {
    Action a;
    if (s.empty()) return a;
    if (s == "acc") {
        a.type = Action::ACC;
    } else if (s[0] == 's') {
        a.type = Action::SHIFT;
        a.value = stoi(s.substr(1));  // ��ȡ״̬���
    } else if (s[0] == 'r') {
        a.type = Action::REDUCE;
        a.value = stoi(s.substr(1));  // ��ȡ��Լ������
    } else if (isdigit(s[0])) {
        a.type = Action::GOTO;
        a.value = stoi(s);  // ��ȡ GOTO ״̬���
    }
    return a;
}

// ȥ���ַ����еĿո񣬲����ָ�������ַ���
vector<string> rm_space(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());  // ȥ���ո�
        tokens.push_back(token);  // ��ʹ�ǿո�Ҳ����Ϊ���ַ���
    }
    return tokens;
}

// �� CSV �ļ���ȡ���ݲ�ת��Ϊ��ά�ַ�������
vector<vector<string>> read_csv_to_vector(const string& filename) {
    ifstream file(filename);  // ���ļ�
    string line;
    vector<vector<string>> result;

    if (!file.is_open()) {  // �ļ���ʧ��
        cerr << "Error: Could not open file " << filename << endl;
        return result;
    }

    // ���ж�ȡ�ļ�
    while (getline(file, line)) {
        vector<string> row;
        vector<string> tokens = rm_space(line, '@');  // ʹ�� '@' ���ŷָ���
        
        // ��ÿ����Ԫ����뵱ǰ��
        for (const auto& token : tokens) {
            row.push_back(token);
        }
        
        // ����ǰ�м��뵽��ά����
        result.push_back(row);
    }

    file.close();  // �ر��ļ�
    return result;
}

// ��ȡ�ķ��ļ�������������ʽ
void read_grammar(const string &filename) {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector<string> parts = split(line, ' ');
        if (parts.empty()) continue;
        Production prod;
        prod.left = parts[0];  // ����ʽ����
        for (size_t i = 1; i < parts.size(); ++i) {
            if (parts[i] == "_") continue;  // �����շ���
            prod.right.push_back(parts[i]);  // ����ʽ���Ҳ�
        }
        productions.push_back(prod);  // ������ʽ�����б�
    }
}

// ��ȡ��������� action_table �� goto_table
void read_trans_table(const string &filename) {
    // ʹ�ö�άvector��ȡ�ļ�
    vector<vector<string>> table = read_csv_to_vector(filename);
    
    // �������ʧ�ܣ�����
    if (table.empty()) {
        cerr << "Error: Failed to read the CSV file." << endl;
        return;
    }

    // ��һ���Ƿ����У���Ϊ�ս���ͷ��ս��
    vector<string> symbols = table[0];
    size_t split_pos = find(symbols.begin(), symbols.end(), "#") - symbols.begin() + 1;

    terminals = vector<string>(symbols.begin(), symbols.begin() + split_pos);  // ��ȡ�ս��
    non_terminals = vector<string>(symbols.begin() + split_pos, symbols.end());  // ��ȡ���ս��

    // ����ÿһ��״̬����
    for (size_t row = 1; row < table.size(); ++row) {  // �ӵڶ��п�ʼ��״̬����
        vector<string> tokens = table[row];
        if (tokens.empty()) continue;

        int state = stoi(tokens[0]);  // ��ǰ״̬���
        for (int i = 1; i < tokens.size(); i++) {  // ����ÿһ�еĶ���
            if (tokens[i] != "") {
                Action a = parse_action(tokens[i]);
                if (a.type != Action::ERROR) {
                    if (i <= 23) {  // ǰ24�����ս���Ķ���
                        action_table[state][terminals[i]] = a;
                    } else {  // ��������Ƿ��ս����ת��״̬
                        goto_table[state][non_terminals[i - 24]] = stoi(tokens[i]);
                    }
                }
            }
        }
    }
}

// ��ӡÿһ����������
void print_step(int step, const stack<string> &sym_stack, const stack<int> &state_stack,
               const vector<string> &input, int pos, const string &action, int goto_state = -1) {
    cout << step << "\t| ";
    stack<string> sym_tmp = sym_stack;
    vector<string> syms;
    while (!sym_tmp.empty()) {
        syms.push_back(sym_tmp.top());
        sym_tmp.pop();
    }
    reverse(syms.begin(), syms.end());
    for (const auto &s : syms) cout << s;
    cout << "\t| ";
    for (size_t i = pos; i < input.size(); ++i) cout << input[i];
    cout << "\t| ";
    stack<int> state_tmp = state_stack;
    vector<int> states;
    while (!state_tmp.empty()) {
        states.push_back(state_tmp.top());
        state_tmp.pop();
    }
    reverse(states.begin(), states.end());
    for (int s : states) cout << s << " ";
    cout << "\t| " << action;
    if (goto_state != -1) cout << "\t| " << goto_state;
    else cout << "\t| ";
    cout << endl;
}

// ����������ִ���ƽ�����Լ�Ȳ���
bool parse(const vector<string> &input) {
    stack<string> sym_stack;
    stack<int> state_stack;
    state_stack.push(0);
    int pos = 0, step = 1;

    cout << "����\t| ����ջ\t| ����\t| ״̬ջ\t| ACTION\t| GOTO" << endl;
    cout << "---|---|---|---|---|---" << endl;

    while (true) {
        int state = state_stack.top();
        string sym = pos < input.size() ? input[pos] : "#";  // ��ȡ��ǰ����

        if (find(terminals.begin(), terminals.end(), sym) == terminals.end()) {
            cerr << "����δ֪���� " << sym << endl;
            return false;
        }

        if (action_table[state].find(sym) == action_table[state].end()) {
            cerr << sym << endl;
            cerr << "����״̬ " << state << " �޶���" << endl;
            return false;
        }

        Action a = action_table[state][sym];
        string action_str;

        // �ƽ�����
        if (a.type == Action::SHIFT) {
            sym_stack.push(sym);
            state_stack.push(a.value);
            action_str = "S" + to_string(a.value);
            pos++;
            print_step(step++, sym_stack, state_stack, input, pos, action_str);
        } 
        else if (a.type == Action::REDUCE) 
        {   // ��Լ����
            Production &prod = productions[a.value];
            int rhs_len = prod.right.size();
            for (int i = 0; i < rhs_len; ++i) {
                if (sym_stack.empty()) 
                {
                    cerr << "���󣺷���ջΪ��" << endl;
                    return false;
                }
                sym_stack.pop();
                state_stack.pop();
            }
            sym_stack.push(prod.left);
            int new_state = state_stack.top();
            int goto_state = goto_table[new_state][prod.left];
            state_stack.push(goto_state);
            action_str = "R" + to_string(a.value);
            print_step(step++, sym_stack, state_stack, input, pos, action_str, goto_state);
        } 
        else if (a.type == Action::ACC) 
        {  // ���ܲ���
            print_step(step++, sym_stack, state_stack, input, pos, "acc");
            cout << "����" << endl;
            return true;
        } 
        else 
        {
            cerr << "������Ч����" << endl;
            return false;
        }
    }
}

int main() {
    read_grammar("gram.txt");  // ��ȡ�ķ�
    read_trans_table("trans.csv");  // ��ȡ������
    // ʾ�����룺d = i #
    // vector<string> input = {"d", "=", "i", "#"};
    vector<string> input;
    ifstream file("codeinput.txt");
    if (!file.is_open()) {
        cerr << "Error opening input file" << endl;
        return 1;
    }
    string line;
    getline(file, line); 
    stringstream ss(line);
    while(ss >> line) {
        input.push_back(line);  // ������ķ��ż��뵽����������
    }
    bool result = parse(input);  // ��������
    cout << (result ? "����" : "�ܾ�") << endl;
    return 0;
}
