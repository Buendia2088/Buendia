#ifndef POLYNODE_H_
#define POLYNODE_H_
#include <iostream>

class Node
{
public:
    double coef;
    int exp;
    Node* next;
    Node(double c = 0, int e = 0) : coef(c), exp(e), next(NULL) {}
};

class PolyNode : public Node
{
private:
    int MaxSize;
    Node* Nodeptr;
public:
    PolyNode(int ms = 0);
    PolyNode(PolyNode* p);
    ~PolyNode();
    void create(int n);
    PolyNode* plus(PolyNode* pn);
    PolyNode* minus(PolyNode* pn);
    void print();
    void clear();
};

#endif