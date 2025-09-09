bool parse(const vector<string> &input) {
    stack<string> sym_stack;
    stack<int> state_stack;
    state_stack.push(0);
    int pos = 0, step = 1;

    cout << "步骤\t| 符号栈\t| 输入\t| 状态栈\t| ACTION\t| GOTO" << endl;
    cout << "---|---|---|---|---|---" << endl;

    while (true) {
        int state = state_stack.top();
        string sym = pos < input.size() ? input[pos] : "#";

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

        if (a.type == Action::SHIFT) {
            sym_stack.push(sym);
            state_stack.push(a.value);
            action_str = "S" + to_string(a.value);
            pos++;
            print_step(step++, sym_stack, state_stack, input, pos, action_str);
        } else if (a.type == Action::REDUCE) {
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
        } else if (a.type == Action::ACC) {
            print_step(step++, sym_stack, state_stack, input, pos, "acc");
            cout << "接受" << endl;
            return true;
        } else {
            cerr << "错误：无效动作" << endl;
            return false;
        }
    }
}


bool parse(const vector<string> &input) {
    stack<string> sym_stack;  // 符号栈
    stack<int> state_stack;   // 状态栈
    state_stack.push(0);      // 初始状态为 0
    //sym_stack.push(input[0]);
    int pos = 0;              // 输入指针
    int step = 1;             // 步骤计数

    cout << "步骤\t| 符号栈\t| 输入\t| 状态栈\t| ACTION\t| GOTO" << endl;
    cout << "---|---|---|---|---|---" << endl;

    while (true) {
        int state = state_stack.top();  // 获取当前状态
        string sym = sym_stack.empty() ? "#" : sym_stack.top();  // 获取符号栈顶的符号，如果栈为空，使用"#"表示结束符号

        // 1. 查找当前状态和符号对应的动作
        if (find(terminals.begin(), terminals.end(), sym) == terminals.end()) {
            cerr << "错误：未知符号 " << sym << endl;
            return false;
        }

        if (action_table[state].find(sym) == action_table[state].end()) {
            cerr << "错误：状态 " << state << " 无动作" << endl;
            return false;
        }

        Action a = action_table[state][sym];  // 获取动作
        string action_str;

        // 2. 执行对应的操作
        if (a.type == Action::SHIFT) {  // 移进操作
            // 将当前输入符号压入符号栈，并转移到目标状态
            sym_stack.push(input[pos]);
            state_stack.push(a.value);
            action_str = "S" + to_string(a.value);
            pos++;  // 读取下一个输入符号
            print_step(step++, sym_stack, state_stack, input, pos, action_str);

        } else if (a.type == Action::REDUCE) {  // 规约操作
            Production &prod = productions[a.value];  // 获取规约规则
            int rhs_len = prod.right.size();  // 规约的右部符号个数

            // 弹出符号栈和状态栈中的符号和状态
            for (int i = 0; i < rhs_len; ++i) {
                if (sym_stack.empty()) {
                    cerr << "错误：符号栈为空" << endl;
                    return false;
                }
                sym_stack.pop();
                state_stack.pop();
            }

            // 将规约规则的左部符号压入符号栈
            sym_stack.push(prod.left);
            int new_state = state_stack.top();  // 获取当前状态
            int goto_state = goto_table[new_state][prod.left];  // 根据左部符号找到对应的 GOTO 状态
            state_stack.push(goto_state);  // 压入新的状态

            action_str = "R" + to_string(a.value);  // 记录规约操作
            print_step(step++, sym_stack, state_stack, input, pos, action_str, goto_state);

        } else if (a.type == Action::GOTO) {  // GOTO 操作
            // 确保符号栈顶是变元（大写字母）
            if (sym_stack.top()[0] < 'A' || sym_stack.top()[0] > 'Z') {
                cerr << "错误：符号栈顶不是变元" << endl;
                return false;
            }

            // 弹出符号栈和状态栈的变元和状态
            sym_stack.pop();
            state_stack.pop();

            // 转移到新的状态
            int goto_state = a.value;  // 获取目标状态
            state_stack.push(goto_state);  // 压入目标状态

            action_str = to_string(goto_state);  // 记录 GOTO 操作
            print_step(step++, sym_stack, state_stack, input, pos, action_str);

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