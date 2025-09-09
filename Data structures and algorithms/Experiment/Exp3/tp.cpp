#include "tp.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#define EPOCH 100;
using namespace std;
graph course_graph;
vector<VexNode> QUICKLY_RES, SMOOTH_RES;
vector<vector<VexNode>> sort_res;
int minimum_average_standard = 114514;
int maximum_average_standard = -1;
int minimum_smooth_standard = 114514;
int total_terms = 114514;
int minimum_residual = 114514;
int p_value = 114514;

bool DFS(int v, vector<int>& state, graph& g) 
{
    state[v] = 1;
    AdjVexNode* neighbor = g.vex[v].adj_arc;
    while (neighbor != NULL) {
        int adj = neighbor->AdjIndex;
        if (state[adj] == 1) 
        {
            return true;
        } 
        else if (state[adj] == 0 && DFS(adj, state, g)) 
        {
            return true;
        }
        neighbor = neighbor->next_node;
    }
    state[v] = 2;
    return false;
}

bool HasCycle(graph& g) 
{
    vector<int> state(g.vex_number, 0);
    for (int i = 0; i < g.vex_number; ++i) 
    {
        if (state[i] == 0) 
        {
            if (DFS(i, state, g)) 
            {
                return true;
            }
        }
    }
    return false;
}

void INPUT()
{
    ifstream fin("data.txt");
    course_graph.course_message = new message;
    fin >> course_graph.course_message->term_num;
    fin >> course_graph.course_message->max_credit;
    fin >> course_graph.course_message->course_number;
    course_graph.vex_number = course_graph.course_message->course_number;
    if(course_graph.course_message->term_num > 12)
    {
        cout << "Too many terms: " << course_graph.course_message->term_num << endl;
        exit(0);
    }
    if(course_graph.course_message->course_number >= 100)
    {
        cout << "Too many courses: " << course_graph.course_message->course_number << endl;
        exit(0);
    }
    for(int i = 0; i < course_graph.vex_number; i++)
    {
        fin >> course_graph.vex[i].course_index >> course_graph.vex[i].course_credit;
        string input;
        string temp = "";
        getline(fin, input);
        if(input == "\r")
        {
            continue;
        }
        else
        {
            input = input.substr(1, input.length() - 2);
        }
        input += " ";
        for(int j = 0; j < input.length(); j++)
        {
            if(input[j] != ' ')
            {
                temp += input[j];
            }
            else
            {
                int temp_index = stoi(temp.substr(1, 2)) - 1;                        
                temp = "";
                AdjVexNode* p = new AdjVexNode;
                p->AdjIndex = temp_index;
                p->next_node = course_graph.vex[i].adj_arc;
                course_graph.vex[i].adj_arc = p;
            }
        }
    }
    for(int j = 0; j < course_graph.course_message->course_number; j++)
    {
        AdjVexNode* p = course_graph.vex[j].adj_arc;
        if(p == NULL)
        {
            continue;
        }
        int flag = 0;
        while(p != NULL)
        {
            int t = p->AdjIndex;
            for(int m = 0; m < course_graph.course_message->course_number; m++)
            {
                if(p->AdjIndex == stoi(course_graph.vex[m].course_index.substr(1, 2)) - 1)
                {
                    flag = 1;
                    break;
                }
            }
            if(flag == 1)
            {
                p = p->next_node;
            }
            else
            {
                break;
            }
        }
        if(flag == 0)
        {
            cout << "Invalid preparatory course(No found): " << course_graph.vex[j].course_index << "'s preparatory course: C";
            if(p->AdjIndex + 1 < 10)
            {
                cout << 0;
            }
            cout << p->AdjIndex + 1 << endl;
            cout << "Check your course information~" << endl;
            exit(0);
        }
    }
    for(int i = 0; i < course_graph.vex_number; i++)
    {
        course_graph.vex[i].in_degree = 0;
    }
    for(int i = 0; i < course_graph.vex_number; i++)
    {
        AdjVexNode* p = course_graph.vex[i].adj_arc;
        while(p != NULL)
        {
            course_graph.vex[p->AdjIndex].in_degree++;
            p = p->next_node;
        }        
    }
    if(HasCycle(course_graph))
    {
        cout << "Ring exsists, check your course infomation!" << endl;
        exit(0);
    }
}

void TopologicalSort(graph course_graph)
{
    int n = EPOCH;
    while(n--)
    {
        graph cg = course_graph;
        vector<int> indices(cg.vex_number);
        for (int i = 0; i < cg.vex_number; ++i)
        {
            indices[i] = i;
        }
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        default_random_engine rng(seed);
        shuffle(indices.begin(), indices.end(), rng);

        vector<VexNode> sort_res_temp;
        queue<AdjVexNode*> in_degree_0;
        int flag = 0;
        while(flag == 0)
        {
            flag = 1;
            for(int i = 0; i < cg.vex_number; i++)
            {
                if(cg.vex[indices[i]].in_degree == 0)
                {
                    cg.vex[indices[i]].in_degree = -114514;
                    in_degree_0.push(cg.vex[indices[i]].adj_arc);
                    sort_res_temp.push_back(cg.vex[indices[i]]);
                    flag = 0;
                }
            }
            while(!in_degree_0.empty())
            {
                AdjVexNode* p = in_degree_0.front();
                in_degree_0.pop();
                while(p != NULL)
                {
                    cg.vex[p->AdjIndex].in_degree--;
                    p = p->next_node;
                }

            }
        }
        reverse(sort_res_temp.begin(), sort_res_temp.end());
        sort_res.push_back(sort_res_temp);    
    }  
}

void JUDGE()
{
    int average_standard = 0;
    int terms_standard = 0;
    int average = 0;
    int residual = 0;
    int temp = 0;
    int total_credit = 0;
    int total_course = 0;
    for(int i = 0; i < course_graph.vex_number; i++)
    {
        total_credit += course_graph.vex[i].course_credit;
    }
    for(int i = 0; i < sort_res.size(); i++)
    {
        average_standard = 0;
        terms_standard = 0;
        temp = 0;
        average = 0;
        for(int j = 0; j < sort_res[i].size(); j++)
        {
            temp += sort_res[i][j].course_credit;
            if(temp > course_graph.course_message->max_credit)
            {
                average_standard += (course_graph.course_message->max_credit - (temp -= sort_res[i][j].course_credit));
                average += temp;
                temp = sort_res[i][j].course_credit;
                terms_standard ++;
            }
        }
        average /= (terms_standard + 1);
        temp = 0;
        residual = 0;
        for(int j = 0; j < sort_res[i].size(); j++)
        {
            temp += sort_res[i][j].course_credit;
            if(temp > course_graph.course_message->max_credit)
            {
                residual += pow(average - (temp -= sort_res[i][j].course_credit), 2);
                temp = sort_res[i][j].course_credit;
            }
        }

        if(average_standard <= minimum_average_standard)
        {
            minimum_average_standard = average_standard;
            QUICKLY_RES = sort_res[i];
            if(terms_standard < total_terms)
            {
                total_terms = terms_standard;
                QUICKLY_RES = sort_res[i];
            }
        }
    } 
    for(int p = total_credit / course_graph.course_message->term_num; p <= course_graph.course_message->max_credit; p++)
    {
        for(int i = 0; i < sort_res.size(); i++)
        {
            terms_standard = 0;
            temp = 0;
            for(int j = 0; j < sort_res[i].size(); j++)
            {
                temp += sort_res[i][j].course_credit;
                if(temp > p)
                {
                    temp = sort_res[i][j].course_credit;
                    terms_standard ++;
                }
            }
            average = total_credit / course_graph.course_message->term_num;
            temp = 0;
            residual = 0;
            total_terms = 0;
            for(int j = 0; j < sort_res[i].size(); j++)
            {
                temp += sort_res[i][j].course_credit;
                if(temp > p)
                {
                    residual += pow(average - (temp -= sort_res[i][j].course_credit), 2);
                    temp = sort_res[i][j].course_credit;
                }
            }
            residual += pow(average - temp, 2);

            if(terms_standard == course_graph.course_message->term_num - 1)
            {
                if(residual < minimum_residual)
                {
                    minimum_residual = residual;
                    SMOOTH_RES = sort_res[i];
                    p_value = p;
                }
            }
        }
    }
}

void menu()
{
    int status;
    cout << "Welcome!" << endl;
    cout << "1.Arranging evenly" << endl;
    cout << "2.Arranging quickly" << endl;
    cout << "3.Quit" << endl;
    cout << "Enter your choice: ";
    cin >> status;
    switch(status)
    {
    case 1:
    { 
        cout << "-------------------------------------------------------------" << endl << endl;;
        int temp = 0;
        int term = 0;
        int count = 0;
        cout << "           " << "term" << 1 << endl;
        cout << "course name" << "    " << "course credits" << endl;
        for(int i = 0; i < SMOOTH_RES.size(); i++)
        {
            temp += SMOOTH_RES[i].course_credit;
            if(temp > p_value)
            {
                count ++;
                cout << endl;
                cout << "           " << "term" << count + 1 << endl;
                cout << "course name" << "    " << "course credits" << endl;
                temp = SMOOTH_RES[i].course_credit;
            }
            cout << "    ";
            cout << SMOOTH_RES[i].course_index << "               ";
            cout << SMOOTH_RES[i].course_credit << endl;

        }
        cout << endl;
        cout << "-------------------------------------------------------------" << endl;      
        break;
    }
	case 2:
    { 
        cout << "-------------------------------------------------------------" << endl << endl;;
        int temp = 0;
        int term = 0;
        int count = 0;
        cout << "           " << "term" << 1 << endl;
        cout << "course name" << "    " << "course credits" << endl;
        for(int i = 0; i < QUICKLY_RES.size(); i++)
        {
            temp += QUICKLY_RES[i].course_credit;
            if(temp > 10)
            {
                count ++;
                cout << endl;
                cout << "           " << "term" << count + 1 << endl;
                cout << "course name" << "    " << "course credits" << endl;
                temp = QUICKLY_RES[i].course_credit;
            }
            cout << "    ";
            cout << QUICKLY_RES[i].course_index << "               ";
            cout << QUICKLY_RES[i].course_credit << endl;

        }
        cout << endl;
        cout << "-------------------------------------------------------------" << endl;        
        break;
    }
	case 3:
    { 
        exit(0);
        break;
    }
    default:
    { 
        cout << "Invalid number" << endl;
        break;
    }
    }
    menu();
}

int main()
{
    INPUT();
    TopologicalSort(course_graph);
    JUDGE();
    menu();
    return 0;
}
