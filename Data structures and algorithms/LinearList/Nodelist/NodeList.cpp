#include <iostream>
#include "../LinearList.h"
#include "Node.h"
#include "NodeList.h"
using namespace std;

template <class Elem>
NodeList<Elem>::NodeList()
{
    init();
}

template <class Elem>
NodeList<Elem>::~NodeList()
{
    removeall();
}

template <class Elem>
void NodeList<Elem>::create(int n)
{
    if(!IsEmpty())
    {
        cout << "Already created~" << endl;
        return;
    }
    if(n < 1)
    {
        cout << "Illegal number~" << endl;
        return;
    }
    Node<Elem>* newNode = new Node<Elem> (NULL);
    head->next = newNode;
    curr = newNode;
    for(int i = 0; i < n-1; i++)
    {
        Node<Elem>* newnewNode = new Node<Elem> (NULL);
        curr->next = newnewNode;
        curr = newnewNode;
    }
    tail = curr;
    curr = head->next;
}

template <class Elem>
bool NodeList<Elem>::getValue(const Elem& e)
{
    if(IsEmpty())
    {
        cout << "Empty list!" << endl;
        return false;
    }
    e = curr->element;
    return true;
}

template <class Elem>
Node<Elem>* NodeList<Elem>::locate(const Elem& e)
{
    if(head->next == NULL)
    {
        cout << "Empty list!" << endl;
        return NULL;
    }    
    while(curr != NULL)
    {
        if(curr->element == e)
        {
            cout << "Found it~" << endl;
            Node<Elem>* location = curr;
            curr = head->next; 
            return location;
        }
        curr = curr->next;
    }
}

template <class Elem>
bool NodeList<Elem>::IsEmpty()
{
    if(head->next == NULL)
    {
        return true;
    }
    return false;
}

template <class Elem>
bool NodeList<Elem>::insert(const Elem& x)
{
    if(head->next == NULL)
    {
        Node<Elem>* newNode = new Node<Elem>(x);
        head->next = newNode;
        return true;
    }
    else
    {
    Node<Elem>* newNode = new Node<Elem>(x);
    newNode->next = head->next;
    head->next = newNode;
    return true;
    }

}

template <class Elem>
bool NodeList<Elem>::remove(Elem& x)
{
    if(IsEmpty())
    {
        cout << "Empty list~" << endl;
        return false;
    }  
    if(curr->next == NULL)
    {
        return true;
    } 
    Node<Elem>* temp = curr->next;
    if(temp->next == NULL)
    {
        delete temp;
        return true;
    }
    else
    {
        curr->next = temp->next;
        x = temp->element;
        delete temp;
        return true;
    }
}

template <class Elem>
void NodeList<Elem>::clear()
{
    removeall();
}

template <class Elem>
void NodeList<Elem>::print()
{
    if(IsEmpty())
    {
        cout << "Empty list~" << endl;
        return;
    }
    while(curr != NULL)
    {
        cout << curr->element << " ";
        curr = curr->next;
    }
    cout << endl;
    curr = head;
}

int main()
{
    
    NodeList<int> test_list;
    cout << "Enter number of nodes: ";
    int n;
    cin >> n;
    while(n--)
    {
        cout << "Enter value for the new node: ";
        int value;
        cin >> value;
        test_list.insert(value);
        cout << "Now the nodelist is: ";
        test_list.print();
    }
    
    return 0;
}