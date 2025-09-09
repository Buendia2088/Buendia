//#include "PolyNode.h"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

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

vector<PolyNode> vec_p;

class menu
{
public: 
    int status_num;
    menu(int s = 0);
    int display();
    void cont();
};



PolyNode::PolyNode(int ms) : MaxSize(ms)
{
    Nodeptr = new Node;
}

PolyNode::PolyNode(PolyNode* p)
{
    Nodeptr = new Node;
    Nodeptr = p->Nodeptr;
    MaxSize = p->MaxSize;
}

PolyNode::~PolyNode()
{
    clear();
}

void PolyNode::create(int n)
{
    while(n--)
    {         
            int c, e;
            cout << "Enter coef and exp: ";
            cin >> c >> e;
            Node* newNode = new Node(c, e);
            if(Nodeptr->next == NULL || e > Nodeptr->next->exp)
            {
                newNode->next = Nodeptr->next;
                Nodeptr->next = newNode;
            }
            else
            {
                Node* cur = Nodeptr->next;
                if(e == Nodeptr->next->exp)
                {
                    cur->coef += c;
                    delete newNode;
                    cout << "Successfully create a polynomial, the polynomial now is: ";
                    print();
                    continue;
                }
                while(cur->next != NULL && e < cur->next->exp)
                {
                    cur = cur->next;
                }
                if(cur->next != NULL && e == cur->next->exp)
                {
                    cur->next->coef += c;
                    delete newNode;
                }
                else
                {
                    newNode->next = cur->next;
                    cur->next = newNode;
                }
            }
            cout << "Successfully create a polynomial, the polynomial now is: ";
            print();
    }
}

void PolyNode::print()
{
    if(Nodeptr == NULL || Nodeptr->next == NULL)
    {
        cout << "EMPTY~" <<endl;
    }
    else
    {
        Node* cur = Nodeptr->next;
        while(cur != NULL)
        {
            if(cur->coef == 0)
            {
                cur = cur->next;
                continue;
            }
            if(cur == Nodeptr->next && cur->coef < 0)
            {
                cout << "-";
            }
            cout << abs(cur->coef) << "^" << cur->exp;
            if(cur->next != NULL && cur->next->coef >= 0)
            {
                cout << " + ";
            }
            else if(cur->next != NULL)
            {
                cout << " - ";
            }
            cur = cur->next;
        }
    }
    cout << endl;
}

void PolyNode::clear()
{
    if(Nodeptr == NULL)
    {
        cout << "Already cleared~" << endl;
    }
    else
    {
        Node*cur = Nodeptr->next;
        while(cur != NULL)
        {
            Node* temp = cur;
            cur = cur->next;
        }
    }
}

PolyNode* PolyNode::plus(PolyNode* pn)
{
    PolyNode* res = new PolyNode(MaxSize + pn->MaxSize);
    Node* res_pt = NULL;
    Node* res_cur = NULL;
    Node* cur1 = this->Nodeptr->next;
    Node* cur2 = pn->Nodeptr->next;
    while(cur1 != NULL || cur2 != NULL)
    {
        Node* newNode = new Node();
        if(cur1 == NULL)
        {
            newNode->coef = cur2->coef;
            newNode->exp = cur2->exp;
            cur2 = cur2->next;
        }
        else if(cur2 == NULL)
        {
            newNode->coef = cur1->coef;
            newNode->exp = cur1->exp;
            cur1 = cur1->next;
        }
        else if(cur1->exp == cur2->exp)
        {
            newNode->coef = cur1->coef + cur2->coef;
            newNode->exp = cur1->exp;
            cur1 = cur1->next;
            cur2 = cur2->next;
        }
        else if(cur1->exp > cur2->exp)
        {
            newNode->coef = cur1->coef;
            newNode->exp = cur1->exp;
            cur1 = cur1->next;
        }
        else
        {
            newNode->coef = cur2->coef;
            newNode->exp = cur2->exp;
            cur2 = cur2->next; 
        }
        if(res_pt == NULL)
        {
            res_pt = new Node;
            res_pt->next = newNode;
            res_cur = res_pt->next;
        }
        else
        {
            res_cur->next = newNode;
            res_cur = newNode;
        }
    }
    res->Nodeptr = res_pt;
    return res;
}


PolyNode* PolyNode::minus(PolyNode* pn)
{
    PolyNode* res = new PolyNode(MaxSize + pn->MaxSize);
    Node* res_pt = NULL;
    Node* res_cur = NULL;
    Node* cur1 = this->Nodeptr->next;
    Node* cur2 = pn->Nodeptr->next;
    while(cur1 != NULL || cur2 != NULL)
    {
        Node* newNode = new Node();
        if(cur1 == NULL)
        {
            newNode->coef = -cur2->coef;
            newNode->exp = cur2->exp;
            cur2 = cur2->next;
        }
        else if(cur2 == NULL)
        {
            newNode->coef = cur1->coef;
            newNode->exp = cur1->exp;
            cur1 = cur1->next;
        }
        else if(cur1->exp == cur2->exp)
        {
            newNode->coef = cur1->coef - cur2->coef;
            newNode->exp = cur1->exp;
            cur1 = cur1->next;
            cur2 = cur2->next;
        }
        else if(cur1->exp > cur2->exp)
        {
            newNode->coef = cur1->coef;
            newNode->exp = cur1->exp;
            cur1 = cur1->next;
        }
        else
        {
            newNode->coef = -cur2->coef;
            newNode->exp = cur2->exp;
            cur2 = cur2->next; 
        }
        if(res_pt == NULL)
        {
            res_pt = new Node;
            res_pt->next = newNode;
            res_cur = res_pt->next;
        }
        else
        {
            res_cur->next = newNode;
            res_cur = newNode;
        }
    }
    res->Nodeptr = res_pt;
    return res;
}

bool compare(Node* p1, Node* p2)
{
    return p1->exp > p2->exp;
}

menu::menu(int s ) : status_num(0) {}

int menu::display()
{
    cout << "Welcome" << endl;
    cout << "1. Add a polynomial" << endl;
    cout << "2. Delete a polynomial" << endl;
    cout << "3. Show all the polynomials" << endl;
    cout << "4. Add two polynomials together" << endl;
    cout << "5. Subtract one polynomial from another" << endl;
    cout << "6. Quit" << endl;
    cout << "Input a number to do the corresponding task: ";
    while(cin >> status_num)
    {
        if(status_num <= 0 && status_num > 6)
        {
            cout << "Ilegal input, try again!" << endl;
            status_num = 0;
            cout << "Input a number to do the corresponding task: ";
        }
        else
        {
            break;
        }
    }

    switch(status_num)
    {
        case 1:
        {
            PolyNode newPolyNode;
            cout << "Enter the number of terms: ";
            int num;
            cin >> num;
            newPolyNode.create(num);
            vec_p.push_back(newPolyNode);
            cont();
            break;
        }
        case 2:
        {
            if(vec_p.empty())
            {
                cout << "There isn't any polynomial." << endl;
                cont();
                break;
            }
            else
            {
                for(int i = 0; i < vec_p.size(); i++)
                {
                    cout << "Polynomial " << i + 1 << " is:" << endl;                    
                    vec_p[i].print();
                }
                cout << "Enter the number of the polynomial that you want to delete: ";
                int n = 0;
                while(cin >> n)
                {
                    if(n <= 0 || n > vec_p.size())
                    {
                        cout << "Ilegal number, try again!";
                        n = 0;
                    }
                    else
                    {
                        vec_p.erase(vec_p.begin() + n - 1);
                        cout << "Sucessfully deleted." << endl;
                        if(vec_p.size() == 0)
                        {
                            cout << "Now there is no polynomial anymore." << endl;
                            cont();
                            break;
                        }
                        else
                        {
                            cout << "Now the polynomials left are:" << endl;
                            for(int i = 0; i < vec_p.size(); i++)
                            {
                                cout << "Polynomial " << i + 1 << " is:" << endl;                    
                                vec_p[i].print();

                            }
                            cont();
                            break;
                        }
                    }
                }
            }
        }
        case 3:
        {
            if(vec_p.empty())
            {
                cout << "There isn't any polynomial." << endl;
                cont();
                break;
            }
            else
            {
                for(int i = 0; i < vec_p.size(); i++)
                {
                    cout << "Polynomial " << i + 1 << " is:" << endl;                    
                    vec_p[i].print();
                }
                cont();
                break;
            }
        }
        
        case 6:
        {
            cout << "Program terminated!" << endl;
            break;
        }
        default:
        {
            break;
        }
    }
    return 0;
}

void menu::cont()
{
    cout << "Enter C to continue: "; 
    string temp;
    cin >> temp;
    if(temp == "C")
    {
        display();
    }
    else
    {
        cout << "Program terminated!" << endl;
    }
}

int main()
{
    /*
    PolyNode t1(100);
    PolyNode t2(100);
    t1.create(3);
    t2.create(2);
    PolyNode res1 = t1.plus(&t2);
    PolyNode res2 = t1.minus(&t2);
    cout << "t1 is" << endl;
    t1.print();
    cout << "t2 is" << endl;
    t2.print();
    cout << "t1 plus t2 is" << endl;
    res1.print();
    cout << "t1 minus t2 is" << endl;
    res2.print();
    */

    menu menu1;
    menu1.display();
    return 0;

}
