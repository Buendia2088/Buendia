#include<bits/stdc++.h>
using namespace std;

struct SingleGener
{
    string left;
    vector<string> right;
};

struct SingleGenerWithDot : public SingleGener
{
    int dotIndex; // Index of the dot in the right side:
    // 0 means before the first symbol, right.size() means after the last symbol
    SingleGenerWithDot(string left, vector<string> right, int index) : SingleGener{left, right}, dotIndex(index) {}

    bool operator==(const SingleGenerWithDot& other) const
    {
        return left == other.left && right == other.right && dotIndex == other.dotIndex; // Compare the left side, right side and dot index
    }

    friend bool operator==(const SingleGener& lhs, const SingleGenerWithDot& rhs)
    {
        return lhs.left == rhs.left && lhs.right == rhs.right; // Compare the left side, right side and dot index
    }
};

struct GenerTable
{
    vector<SingleGener> table; // Hash by its index in the vector
    // Guarantee index 0 to be argumented production
    unordered_map<string, vector<string>> first;
    unordered_map<string, vector<string>> follow;
    vector<string> nonTerminal;
    vector<string> terminal;
    string startSymbol;
    string ArgumentedStartSymbol; // Augmented grammar start symbol
    void ParseTableFromFile(ifstream& file)
    {
        string line;
        getline(file, line); // Read the first line for the start symbol
        stringstream ss(line);
        ss >> startSymbol; // Read the start symbol
        // Argumented Grammer
        ArgumentedStartSymbol = startSymbol + "'"; // Augmented grammar start symbol
        table.push_back({ArgumentedStartSymbol, {startSymbol}}); // Add augmented grammar production
        nonTerminal.push_back(ArgumentedStartSymbol); // Add to non-terminal list
        while (getline(file, line))
        {
            if (line.empty()) continue; // Skip empty lines
            SingleGener gener;
            stringstream ss(line);
            ss >> gener.left; // Read the left side of the production
            if(find(nonTerminal.begin(), nonTerminal.end(), gener.left) == nonTerminal.end())
                nonTerminal.push_back(gener.left); // Add to non-terminal list if not already present

            string rightPart;
            while (ss >> rightPart) // Read the right side of the production
            {
                if(rightPart == "_") break; // Empty production
                gener.right.push_back(rightPart);
            }
            table.push_back(gener);
        }

        for(auto&& item : table)
        {
            for(auto&& right : item.right)
            {
                if(find(nonTerminal.begin(), nonTerminal.end(), right) == nonTerminal.end() && find(terminal.begin(), terminal.end(), right) == terminal.end())
                {
                    terminal.push_back(right); // Add to terminal list if not already present
                }
            }
        }
    }

    void ParseFollowFromFile(ifstream& file)
    {
        string Line;
        while(getline(file, Line))
        {
            stringstream ss(Line);
            string str;
            string temp;
            vector<string> right;
            ss >> str; // Left
            while(ss >> temp)
            {
                right.push_back(temp);
            }
            follow.insert({str, right});
        }

    }
};

struct ActionType
{
    enum Type
    {
        Uninit,
        Shift,
        Reduce,
        Accept,
    };
    Type type = Uninit; // Type of action
    int num; 
    // For Shift, num is the state number to shift to
    // For Reduce, num is the production index to reduce by
    // For Goto, num is the state number to go to
    // For Accept, num is not used
};

struct ItemDFAState
{
    vector<SingleGenerWithDot> itemSet; // Set of items in this state

    bool operator==(const ItemDFAState& other) const
    {
        return itemSet == other.itemSet; // Compare the item sets
    }

    void print(ostream& os = cout)
    {
        os<<"Set Start:"<<endl;
        for(auto&& item : itemSet)
        {
            os << item.left << " -> ";
            if(item.right.size() == 0)
            {
                os << "._"<<endl;
                continue;
            }
            for(size_t i = 0; i < item.right.size(); ++i)
            {
                if(i == item.dotIndex) os << "."; // Print the dot at the correct position
                os << item.right[i] << " ";
            }
            if(item.dotIndex == item.right.size()) os << "."; // Print the dot at the end if needed
            os << endl;
        }
        os<<"Set End."<<endl;
    }
};

struct ItemDFA
{
    GenerTable& table; // Reference to the grammar table
    int SetCount; // Number of sets in the DFA
    vector<ItemDFAState> states; // States of the DFA
    vector<vector<vector<ActionType>>> ActionTable; // Action table for the DFA, Y indexed by state number, X indexed by index of terminal symbol
    vector<vector<int>> GotoTable; // Goto table for the DFA, Y indexed by state number, X indexed by index of non-terminal symbol

    void Closure(ItemDFAState& itemSet, vector<string>& nextSymbols)
    {
        vector<SingleGenerWithDot> temp = itemSet.itemSet;
        here:
        bool hasChanged = false;
        itemSet.itemSet = temp; // Update the item set with the new items
        for(auto&& item : itemSet.itemSet)
        {
            // If the dot is at the end of the production, skip it
            if(item.dotIndex == item.right.size()) continue;
            string nextSymbol = item.right[item.dotIndex];
            if(find(nextSymbols.begin(), nextSymbols.end(), nextSymbol) == nextSymbols.end())
            {
                nextSymbols.push_back(nextSymbol); // Add the next symbol to the list
            }
            // If the next symbol is a non-terminal, find its productions
            if(find(table.nonTerminal.begin(), table.nonTerminal.end(), nextSymbol) != table.nonTerminal.end())
            {
                for(auto&& gener : table.table)
                {
                    if(gener.left == nextSymbol)
                    {
                        // Create a new item with the dot at the beginning of the production
                        SingleGenerWithDot newItem(gener.left, gener.right, 0);
                        // Check if the item is already in the set
                        if(find(temp.begin(), temp.end(), newItem) == temp.end())
                        {
                            temp.push_back(newItem);
                            hasChanged = true;
                        }
                    }
                }
            }
        }
        // If the item set has changed, repeat the closure process
        if(hasChanged)
        {
            goto here; // Repeat the closure process
        }
    }

    long int getIndexOfStates(ItemDFAState& itemSet)
    {
        for(size_t i = 0; i < states.size(); ++i)
        {
            if(states[i].itemSet == itemSet.itemSet) return i; // Return the index of the state if found
        }
        return -1; // Return -1 if not found
    }

    void updateTable(long int current_state_index, string symbol, ActionType&& action)
    {
        // Update the action table with the new action
        auto it = find(table.terminal.begin(), table.terminal.end(), symbol);
        if(it == table.terminal.end()) return; // If the symbol is not found in the terminal list, return
        auto index = distance(table.terminal.begin(), it); // Get the index of the symbol in the terminal list
        auto&& Place = ActionTable.at(current_state_index).at(index); // Get the action at the current state and symbol index
        // if(Place.type != ActionType::Uninit)
        // {
        //     // println("Conflict detected for state {} and symbol {}: now {}, previous {}", current_state_index, symbol, (int)action.type, (int)Place.type);
        //     cout<<"Conflict detected: for state "<<current_state_index
        //     <<", Symbol "<<symbol
        //     <<", Previous"<<(int)Place.type<<"_"<<Place.num
        //     <<", Now"<<(int)action.type<<"_"<<action.num
        //     <<endl;
        // }
        Place.push_back(action); // Update the action at the current state and symbol index
    }

    void dfs(ItemDFAState itemSet,const vector<string>& nextSymbols)
    {
        vector<SingleGenerWithDot> temp = itemSet.itemSet; // Copy the current item set
        long int currentStateIndex = getIndexOfStates(itemSet); // Get the index of the current state
        for(auto&& nxtChar : nextSymbols)
        {
            // cout<<nxtChar<<endl;
            ItemDFAState newItemSet; // New item set for the new state
            for(auto&& item : temp)
            {
                // If the dot is at the end of the production, skip it
                if(item.dotIndex == item.right.size()) continue;
                // If the next symbol matches the symbol after the dot, create a new item with the dot moved to the right
                if(item.right[item.dotIndex] == nxtChar)
                {
                    newItemSet.itemSet.emplace_back(item.left, item.right, item.dotIndex + 1); // Add the new item to the new item set
                }
            }
            if(!newItemSet.itemSet.empty())
            {
                vector<string> newNextSymbols;
                Closure(newItemSet, newNextSymbols); // Perform closure on the new state
                
                if(find(states.begin(), states.end(), newItemSet) == states.end())
                {
                    states.push_back(newItemSet); // Add the new state to the DFA states
                    SetCount++; // Increment the set count
                    dfs(newItemSet, newNextSymbols); // Perform DFS on the new state
                }
                long int newStateIndex = getIndexOfStates(newItemSet); // Get the index of the new state
                if(newStateIndex != -1)
                {
                    // Update the action table with the new action
                    if(find(table.terminal.begin(), table.terminal.end(), nxtChar) != table.terminal.end()) // terminal, shift
                    {
                        ActionType action;
                        action.type = ActionType::Shift; // Set the action type to shift
                        action.num = newStateIndex; // Set the state number to shift to
                        updateTable(currentStateIndex, nxtChar, move(action)); // Update the action table
                    }
                    else if(find(table.nonTerminal.begin(), table.nonTerminal.end(), nxtChar) != table.nonTerminal.end()) // non-terminal, goto
                    {
                        auto it = find(table.nonTerminal.begin(), table.nonTerminal.end(), nxtChar); // Find the index of the non-terminal symbol
                        auto index = distance(table.nonTerminal.begin(), it); // Get the index of the non-terminal symbol
                        GotoTable.at(currentStateIndex).at(index) = newStateIndex; // Update the goto table with the new state index
                    }
                }
            }
        }
        // cout<<"---"<<endl;
        // Check accept state
        for(auto&& item : temp)
        {
            if(item.dotIndex == item.right.size() && item.left == table.ArgumentedStartSymbol) // If the item is an accept state
            {
                ActionType action;
                action.type = ActionType::Accept; // Set the action type to accept
                action.num = -1; // Set to -1 for accept action
                ActionTable.at(currentStateIndex).at(table.terminal.size()).push_back(move(action)); // Update the action table for the accept state
            }
        }

        // Check reduce state
        for(auto&& item : temp)
        {
            if(item.dotIndex == item.right.size()) // If the item is a reduce state
            {
                auto it = find(table.table.begin(), table.table.end(), item); // Find the index of the item in the grammar table
                auto index = distance(table.table.begin(), it); // Get the index of the item in the grammar table
                ActionType action;
                action.type = ActionType::Reduce; // Set the action type to reduce
                action.num = index; // Set the production index to reduce by
                //updateTable(currentStateIndex, item.left, move(action)); // Update the action table for the reduce state
                auto& Form = table.follow.at(item.left);
                for(int i = 0; i <= table.terminal.size(); i++)
                {
                    auto&& c = (i == table.terminal.size() ? "#" : table.terminal.at(i));
                    if(find(Form.begin(), Form.end(), c) != Form.end())
                    {
                        auto& Modi = 
                            ActionTable.at(currentStateIndex).at(i);
                        // if(Modi.type != ActionType::Uninit)
                        // {
                        //     cout<<"Conflict detected: for state "<<currentStateIndex
                        //         <<", Symbol "<<c
                        //         <<", Previous"<<(int)Modi.type<<"_"<<Modi.num
                        //         <<", Now"<<(int)action.type<<"_"<<action.num
                        //         <<endl;
                        // }
                        Modi.push_back(action); // Update the action table for the reduce state
                    }
                }
            }
        }
    }

    void BuildDFA()
    {
        ItemDFAState startState;
        SingleGenerWithDot startItem(table.ArgumentedStartSymbol, {table.startSymbol}, 0);
        startState.itemSet.push_back(startItem); // Add the start item to the start state
        vector<string> nextSymbols; // List of next symbols to process
        Closure(startState, nextSymbols); // Perform closure on the start state
        states.push_back(startState); // Add the start state to the DFA states
        SetCount = 1; // Initialize the set count to 1
        ActionTable.resize(10000, vector<vector<ActionType>>(table.terminal.size() + 1)); // Resize the action table to accommodate the start state
        GotoTable.resize(10000, vector<int>(table.nonTerminal.size())); // Resize the goto table to accommodate the start state
        startState.print(); // Print the start state
        dfs(startState, nextSymbols); // Perform DFS to build the DFA
    }
};

int main()
{
    GenerTable generTable;
    ifstream file("gram.txt");
    if (!file.is_open())
    {
        cerr << "Error opening file" << endl;
        return 1;
    }
    generTable.ParseTableFromFile(file);
    file.close();

    ifstream file2("follow.txt");
    if(! file2.is_open())
    {
        cerr << "Error opening file" << endl;
        return 1;
    }
    generTable.ParseFollowFromFile(file2);
    file2.close();

    for(auto&& item : generTable.follow) 
    {
        cout<<item.first<<": ";
        for(auto&& item2 : item.second)
        {
            cout<<item2<<" ";
        }
        cout<<endl;
    }

    cout << generTable.nonTerminal.size() << endl;
    cout << generTable.terminal.size() << endl;
    ItemDFA itemDFA(generTable);
    itemDFA.BuildDFA(); // Build the DFA from the grammar table

    // println("DFA states count: {}", itemDFA.SetCount);
    cout << "DFA states count: " << itemDFA.SetCount << endl; // Output the number of DFA states
    
    ofstream fout("trans.csv");
    fout<<setw(10)<<"@";
    for(auto&& item : generTable.terminal)
    {
        fout<<setw(10)<<item<<"@";
    }
    fout<<setw(10)<<"#"<<"@";
    for(auto&& item : generTable.nonTerminal)
    {
        fout<<setw(10)<<item<<"@";
    }
    fout<<endl;
    int ConflictCount = 0;
    for(int i = 0; i < itemDFA.SetCount; i++)
    {
        fout<<setw(10)<<i<<"@";
        for(int j = 0; j <= generTable.terminal.size(); j++)
        {
            string op;
            auto& item = itemDFA.ActionTable.at(i).at(j);
            for(auto&& item2 : item)
            {
                if(item2.type == ActionType::Shift) op += "s" + to_string(item2.num); // Shift action
                else if(item2.type == ActionType::Reduce) op += "r" + to_string(item2.num); // Reduce action
                else if(item2.type == ActionType::Accept) op += "acc"; // Accept action
                else op = " "; // Uninitialized action
                if(&item2 != &item.back()) op += "/"; // If not the last item, add a comma
            }
            // fout<<setw(10)<<op<<"@";
            if(item.size() >= 2)
            {
                cout<<"Conflict No." << (++ConflictCount )
                    <<"detected: for state "<<i
                    <<", Symbol "<<(j == generTable.terminal.size() ? "#" : generTable.terminal.at(j))<<", "
                    <<op
                    <<endl;
            
                cout<<"Please select action for this conflict: "<<endl;
                for(int k = 0; k < item.size(); k++)
                {
                    cout<<k<<": ";
                    if(item.at(k).type == ActionType::Shift) cout<<"s" << item.at(k).num <<endl; // Shift action
                    else if(item.at(k).type == ActionType::Reduce) cout<<"r" << item.at(k).num <<endl; // Reduce action
                    else if(item.at(k).type == ActionType::Accept) cout<<"acc"<<endl; // Accept action
                    else cout<<"Uninit"<<endl; // Uninitialized action
                }
                cout<<item.size()<<": ignore" <<endl;
                int choice;
                cin>>choice; // Get user input for the action to take
                if(choice == item.size()) goto here2;
                if(choice < 0 || choice >= item.size())
                {
                    cout<<"Invalid choice, please try again."<<endl;
                    j--;
                    continue; // If invalid choice, repeat the action for the same symbol
                }
                auto& selectedAction = item.at(choice); // Get the selected action
                if(selectedAction.type == ActionType::Shift) // Shift action
                {
                    op = "s" + to_string(selectedAction.num); // Set the action to shift
                }
                else if(selectedAction.type == ActionType::Reduce) // Reduce action
                {
                    op = "r" + to_string(selectedAction.num); // Set the action to reduce
                }
                else if(selectedAction.type == ActionType::Accept) // Accept action
                {
                    op = "acc"; // Set the action to accept
                }
            }
            else if(item.size() == 1)
            {
                auto& selectedAction = item.at(0); // Get the selected action
                if(selectedAction.type == ActionType::Shift) // Shift action
                {
                    op = "s" + to_string(selectedAction.num); // Set the action to shift
                }
                else if(selectedAction.type == ActionType::Reduce) // Reduce action
                {
                    op = "r" + to_string(selectedAction.num); // Set the action to reduce
                }
                else if(selectedAction.type == ActionType::Accept) // Accept action
                {
                    op = "acc"; // Set the action to accept
                }
            }
            else op = " "; // Uninitialized action
            here2:
            fout<<setw(10)<<op<<"@"; // Output the action to the file
        }
        for(int j = 0; j < generTable.nonTerminal.size(); j++)
        {
            auto& item = itemDFA.GotoTable.at(i).at(j);
            if(item == 0) fout<<setw(10)<<"@";
            else fout<<setw(10)<<item<<"@";
        }
        fout<<endl;
    }
    
    ofstream fout2("states.txt");
    for(int i = 0; i < itemDFA.SetCount; i++)
    {
        fout2<<i<<": "<<endl;
        itemDFA.states[i].print(fout2);
    }
    fout2.close();
    return 0;
}
