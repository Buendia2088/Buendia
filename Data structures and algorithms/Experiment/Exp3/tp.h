#ifndef TP_H
#define TP_H
#include <iostream>
#include <string>
using namespace std;

struct AdjVexNode
{
    int AdjIndex;
    AdjVexNode* next_node;
    AdjVexNode(int i = -1) : AdjIndex(i), next_node(NULL) {}
};

struct VexNode
{
    string course_index;
    int course_credit;
    int in_degree;
    AdjVexNode* adj_arc;
};

struct message
{
    int term_num;
    int max_credit;
    int course_number;
};

struct graph
{
    VexNode vex[200];
    int vex_number;
    message* course_message;
    graph()
    {
        course_message == NULL;
    }
    graph(const graph& other)
    {
        for (int i = 0; i < other.vex_number; ++i)
        {
            vex[i] = other.vex[i];
            if (other.vex[i].adj_arc != NULL)
            {
                vex[i].adj_arc = new AdjVexNode(*other.vex[i].adj_arc);
                AdjVexNode* current = vex[i].adj_arc;
                AdjVexNode* other_current = other.vex[i].adj_arc->next_node;
                while (other_current != NULL)
                {
                    current->next_node = new AdjVexNode(*other_current);
                    current = current->next_node;
                    other_current = other_current->next_node;
                }
            }
        }
        vex_number = other.vex_number;
        if (other.course_message != NULL)
        {
            course_message = new message;
            *course_message = *other.course_message;
        }
        else
        {
            course_message = NULL;
        }
    }

    graph& operator=(const graph& other)
    {
        if (this != &other)
        {
            for (int i = 0; i < vex_number; ++i)
            {
                AdjVexNode* current = vex[i].adj_arc;
                while (current != NULL)
                {
                    AdjVexNode* temp = current;
                    current = current->next_node;
                    delete temp;
                }
            }
            delete course_message;
            for (int i = 0; i < other.vex_number; ++i)
            {
                vex[i] = other.vex[i];
                if (other.vex[i].adj_arc != NULL)
                {
                    vex[i].adj_arc = new AdjVexNode(*other.vex[i].adj_arc);
                    AdjVexNode* current = vex[i].adj_arc;
                    AdjVexNode* other_current = other.vex[i].adj_arc->next_node;
                    while (other_current != NULL)
                    {
                        current->next_node = new AdjVexNode(*other_current);
                        current = current->next_node;
                        other_current = other_current->next_node;
                    }
                }
            }
            vex_number = other.vex_number;
            if (other.course_message != NULL)
            {
                course_message = new message;
                *course_message = *other.course_message;
            }
            else
            {
                course_message = NULL;
            }
        }
        return *this;
    }
};


#endif