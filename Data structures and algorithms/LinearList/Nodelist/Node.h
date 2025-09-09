#ifndef NODE_H_
#define NODE_H_
#include "../LinearList.h"
using namespace std;

template <class Elem>
class Node:public List<Elem>
{
public:
    Elem element;
    Node* next;
    Node(const Elem& item, Node* nextval = NULL)
    {
        element = item;
        next = nextval;
    }
    Node(Node* nextval = NULL)
    {
        next = nextval;
    }
    void clear() override {}
    bool insert(const Elem&) override{}
    bool remove(Elem&) override {}
    void print() override {}
    bool IsEmpty()  override {}

};

#endif