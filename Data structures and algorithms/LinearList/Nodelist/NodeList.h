#ifndef NODELIST_H_
#define NODELIST_H_

#include "Node.h"

template <class Elem>
class NodeList:public Node<Elem>
{
private:
     Node<Elem>* head;
     Node<Elem>* tail;
     Node<Elem>* curr;
     void init()
     {
        curr = tail = head = new Node<Elem>;
     }
     void removeall()
     {
        while(head != NULL)
        {
            curr = head;
            head = head->next;
            delete curr;
        }
     }
public:
    NodeList();
    ~NodeList();
    void create(int n);
    bool getValue(const Elem& e);
    Node<Elem>* locate(const Elem& e);
    bool IsEmpty() override;
    bool insert(const Elem& x) override;
    bool remove(Elem& x) override;
    void clear() override;
    void print() override;

};

#endif