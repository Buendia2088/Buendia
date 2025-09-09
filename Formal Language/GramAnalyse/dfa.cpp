#include <iostream>
#include <vector>
#include <string>
#include <stack>
#include <map>
#include <sstream>
#include <fstream>
#include <algorithm>

using namespace std;

// 定义产生式结构体，表示文法规则
struct Production {
    string left;  // 产生式的左部（非终结符）
    vector<string> right;  // 产生式的右部（符号序列）
};

// 定义Action结构体，用来表示移进、规约、转移等操作
struct Action {
    enum Type { SHIFT, REDUCE, GOTO, ACC, ERROR } type;  // 动作类型
    int value;  // 关联的状态编号或规约规则的编号
    Action() : type(ERROR), value(-1) {}  // 默认构造函数，表示无效操作
};

vector<Production> productions;  // 存储文法规则
vector<string> terminals;       // 存储终结符
vector<string> non_terminals;   // 存储非终结符
map<int, map<string, Action>> action_table;  // 存储状态和符号对应的动作（移进、规约等）
map<int, map<string, int>> goto_table;       // 存储状态和非终结符对应的转移状态

// 分割字符串函数，按指定分隔符拆分字符串，去掉空格并返回分割后的字符串向量
vector<string> split(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        // 去掉每个单元格中的空格
        token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

// 检查符号类型，返回0表示是终结符，返回1表示是非终结符，返回-1表示未知符号
int check_symbol_type(const string& symbol) {
    if (find(terminals.begin(), terminals.end(), symbol) != terminals.end()) {
        return 0;  // 是终结符
    }
    if (find(non_terminals.begin(), non_terminals.end(), symbol) != non_terminals.end()) {
        return 1;  // 是非终结符
    }
    return -1;  // 既不是终结符也不是非终结符
}

// 解析动作字符串，将动作字符串解析为对应的 Action 对象
Action parse_action(const string &s) {
    Action a;
    if (s.empty()) return a;
    if (s == "acc") {
        a.type = Action::ACC;
    } else if (s[0] == 's') {
        a.type = Action::SHIFT;
        a.value = stoi(s.substr(1));  // 获取状态编号
    } else if (s[0] == 'r') {
        a.type = Action::REDUCE;
        a.value = stoi(s.substr(1));  // 获取规约规则编号
    } else if (isdigit(s[0])) {
        a.type = Action::GOTO;
        a.value = stoi(s);  // 获取 GOTO 状态编号
    }
    return a;
}

// 去除字符串中的空格，并按分隔符拆分字符串
vector<string> rm_space(const string &s, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(s);
    while (getline(tokenStream, token, delimiter)) {
        token.erase(remove_if(token.begin(), token.end(), ::isspace), token.end());  // 去掉空格
        tokens.push_back(token);  // 即使是空格也保留为空字符串
    }
    return tokens;
}

// 从 CSV 文件读取数据并转换为二维字符串向量
vector<vector<string>> read_csv_to_vector(const string& filename) {
    ifstream file(filename);  // 打开文件
    string line;
    vector<vector<string>> result;

    if (!file.is_open()) {  // 文件打开失败
        cerr << "Error: Could not open file " << filename << endl;
        return result;
    }

    // 逐行读取文件
    while (getline(file, line)) {
        vector<string> row;
        vector<string> tokens = rm_space(line, '@');  // 使用 '@' 符号分割列
        
        // 将每个单元格加入当前行
        for (const auto& token : tokens) {
            row.push_back(token);
        }
        
        // 将当前行加入到二维向量
        result.push_back(row);
    }

    file.close();  // 关闭文件
    return result;
}

// 读取文法文件并解析成生产式
void read_grammar(const string &filename) {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        vector<string> parts = split(line, ' ');
        if (parts.empty()) continue;
        Production prod;
        prod.left = parts[0];  // 产生式的左部
        for (size_t i = 1; i < parts.size(); ++i) {
            if (parts[i] == "_") continue;  // 跳过空符号
            prod.right.push_back(parts[i]);  // 产生式的右部
        }
        productions.push_back(prod);  // 将生产式加入列表
    }
}

// 读取分析表并填充 action_table 和 goto_table
void read_trans_table(const string &filename) {
    // 使用二维vector读取文件
    vector<vector<string>> table = read_csv_to_vector(filename);
    
    // 如果解析失败，返回
    if (table.empty()) {
        cerr << "Error: Failed to read the CSV file." << endl;
        return;
    }

    // 第一行是符号行，分为终结符和非终结符
    vector<string> symbols = table[0];
    size_t split_pos = find(symbols.begin(), symbols.end(), "#") - symbols.begin() + 1;

    terminals = vector<string>(symbols.begin(), symbols.begin() + split_pos);  // 提取终结符
    non_terminals = vector<string>(symbols.begin() + split_pos, symbols.end());  // 提取非终结符

    // 解析每一行状态数据
    for (size_t row = 1; row < table.size(); ++row) {  // 从第二行开始是状态数据
        vector<string> tokens = table[row];
        if (tokens.empty()) continue;

        int state = stoi(tokens[0]);  // 当前状态编号
        for (int i = 1; i < tokens.size(); i++) {  // 解析每一列的动作
            if (tokens[i] != "") {
                Action a = parse_action(tokens[i]);
                if (a.type != Action::ERROR) {
                    if (i <= 23) {  // 前24列是终结符的动作
                        action_table[state][terminals[i]] = a;
                    } else {  // 后面的列是非终结符的转移状态
                        goto_table[state][non_terminals[i - 24]] = stoi(tokens[i]);
                    }
                }
            }
        }
    }
}

// 打印每一步解析过程
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

// 解析函数，执行移进、规约等操作
bool parse(const vector<string> &input) {
    stack<string> sym_stack;
    stack<int> state_stack;
    state_stack.push(0);
    int pos = 0, step = 1;

    cout << "步骤\t| 符号栈\t| 输入\t| 状态栈\t| ACTION\t| GOTO" << endl;
    cout << "---|---|---|---|---|---" << endl;

    while (true) {
        int state = state_stack.top();
        string sym = pos < input.size() ? input[pos] : "#";  // 获取当前符号

        if (find(terminals.begin(), terminals.end(), sym) == terminals.end()) {
            cerr << "错误：未知符号 " << sym << endl;
            return false;
        }

        if (action_table[state].find(sym) == action_table[state].end()) {
            cerr << "错误：状态 " << state << " 无动作" << endl;
            return false;
        }

        Action a = action_table[state][sym];
        string action_str;

        // 移进操作
        if (a.type == Action::SHIFT) {
            sym_stack.push(sym);
            state_stack.push(a.value);
            action_str = "S" + to_string(a.value);
            pos++;
            print_step(step++, sym_stack, state_stack, input, pos, action_str);
        } else if (a.type == Action::REDUCE) {  // 规约操作
            Production &prod = productions[a.value];
            int rhs_len = prod.right.size();
            for (int i = 0; i < rhs_len; ++i) {
                if (sym_stack.empty()) {
                    cerr << "错误：符号栈为空" << endl;
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
        } else if (a.type == Action::ACC) {  // 接受操作
            print_step(step++, sym_stack, state_stack, input, pos, "acc");
            cout << "接受" << endl;
            return true;
        } else {
            cerr << "错误：无效动作" << endl;
            return false;
        }
    }
}

int main() {
    read_grammar("gram.txt");  // 读取文法
    read_trans_table("trans.csv");  // 读取分析表
    // 示例输入：d = i #
    vector<string> input = {"d", "=", "i", "#"};
    bool result = parse(input);  // 解析输入
    cout << (result ? "接受" : "拒绝") << endl;
    return 0;
}
