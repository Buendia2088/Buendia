#include "PolyNode.h"
#include <iostream>
#include <cmath>
#include <vector>
using namespace std;

vector<PolyNode> vec_p;

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
            if(cur->coef == 0) //系数为0，不打印
            {
                cur = cur->next;
                continue;
            }
            if(cur == Nodeptr->next && cur->coef < 0) //首项系数为负，打印富豪
            {
                cout << "-";
            }
            if(abs(cur->coef == 1)) //该项系数为1或-1，省略系数
            {
                cout << "x" << "^" << cur->exp;
            }
            else
            {
                cout << abs(cur->coef) << "x" << "^" << cur->exp;
            } 
            if(cur->next != NULL && cur->next->coef >= 0) //后一项系数为正，打印加号
            {
                cout << " + ";
            }
            else if(cur->next != NULL && cur->next->coef != 0) //后一项系数为负，打印减号
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
    if (Nodeptr == NULL)
    {
        cout << "Already cleared~" << endl;
    }
    else
    {
        Node* cur = Nodeptr->next;
        while (cur != NULL)
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
        else if(cur1->exp == cur2->exp) //对于系数相加之和为0的特殊处理
        {
            if(cur1->coef + cur2->coef != 0)
            {
                newNode->coef = cur1->coef + cur2->coef;
                newNode->exp = cur1->exp;
            }
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
            if(cur1->coef - cur2->coef != 0)
            {
                newNode->coef = cur1->coef - cur2->coef;
                newNode->exp = cur1->exp;
            }
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

void PolyNode::show_at_x(int x)
{
    double result = 0;
    Node* cur = Nodeptr->next;
    while (cur != NULL)
    {
        result += cur->coef * pow(x, cur->exp);
        cur = cur->next;
    }
    cout << "The value of the polynomial at x = " << x << " is: " << result << endl;
}


bool compare(Node* p1, Node* p2)
{
    return p1->exp > p2->exp;
}

menu::menu(int s ) : status_num(s) {}

int menu::display()
{
    cout << "Welcome" << endl;
    cout << "1. Add a polynomial" << endl;
    cout << "2. Delete a polynomial" << endl;
    cout << "3. Show all the polynomials" << endl;
    cout << "4. Add two polynomials together" << endl;
    cout << "5. Subtract one polynomial from another" << endl;
    cout << "6. Caculate the value of the polynomial at x" << endl;
    cout << "7. Quit" << endl;
    cout << "Input a number to do the corresponding task: ";
    while(cin >> status_num)
    {
        if(status_num <= 0 && status_num > 7)
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
            while(cin >> num)
            {
                if(num <= 0)
                {
                    cout << "Ilegal number, try again!" << endl;
                    cout << "Enter the number of terms: ";
                }
                else
                {
                    break;
                }
            }
            newPolyNode.create(num);
            vec_p.push_back(newPolyNode);
            cont();
            break;
        }

        case 2:
        {
            if(vec_p.empty()) //向量为空
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
                    if(n <= 0 || n > vec_p.size()) //用户输入了不存在的多项式链表编号
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
            /*
            if(status_num == 0)
            {
                break;
            }
            */
            if(vec_p.empty()) //向量为空
            {
                cout << "There isn't any polynomial." << endl;
                cont();
                break;
            }
            else
            {
                for(int i = 0; i < vec_p.size(); i++) //向量非空，调用print函数打印
                {
                    cout << "Polynomial " << i + 1 << " is:" << endl;                    
                    vec_p[i].print();
                }
                cont();
                break;
            }
        }
        case 4:
        {
            if(vec_p.empty()) //向量为空
            {
                cout << "There isn't any polynomial, adding is invalid." << endl;
            }
            if(vec_p.size() == 1) //向量中只有一个多项式链表，无法相加
            {
                cout << "There is only one polynomial, adding is invalid" << endl;
            }
            if(vec_p.size() > 1)
            {
                for(int i = 0; i < vec_p.size(); i++)
                {
                    cout << "Polynomial " << i + 1 << " is:" << endl;
                    vec_p[i].print();
                }
                cout << "Please enter the numbers of the two polynomials that you want to add together, ";
                cout << "the two numbers should be seperated by a space.";
                cout << endl << "Enter here: ";
                int num1, num2;
                while(cin >> num1 >> num2)
                {
                    if(num1 < 1 || num2 < 1 || num1 > vec_p.size() || num2 > vec_p.size() ) //输入越界
                    {
                        cout << "Invalid input, try again!" << endl;
                        cout << "Enter here: ";
                        continue;
                    }
                    else //输入合法，进行相加和输出
                    {
                        PolyNode res = vec_p[num1 - 1].plus(&vec_p[num2 - 1]);
                        res.print();
                        break;
                    }
                }
            }
            cont();
            break;
        }
        case 5:
        {
            if(vec_p.empty())
            {
                cout << "There isn't any polynomial, minusing is invalid." << endl;
            }
            if(vec_p.size() == 1)
            {
                cout << "There is only one polynomial, minusing is invalid" << endl;
            }
            if(vec_p.size() > 1)
            {
                for(int i = 0; i < vec_p.size(); i++)
                {
                    cout << "Polynomial " << i + 1 << " is:" << endl;
                    vec_p[i].print();
                }
                cout << "Please enter the numbers of the two polynomials(a and b) that you want to minus(a - b), ";
                cout << "the two numbers should be seperated by a space.";
                cout << endl << "Enter here: ";
                int num1, num2;
                while(cin >> num1 >> num2)
                {
                    if(num1 < 1 || num2 < 1 || num1 > vec_p.size() || num2 > vec_p.size() )
                    {
                        cout << "Invalid input, try again!" << endl;
                        cout << "Enter here: ";
                        continue;
                    }
                    else
                    {
                        PolyNode res = vec_p[num1 - 1].minus(&vec_p[num2 - 1]);
                        res.print();
                        break;
                    }
                }
            }
            cont();
            break;
        }
        case 6:
        {
            if(vec_p.empty())
            {
                cout << "There isn't any polynomial, calculation is invalid." << endl;
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
                cout << "Please enter the numbers of the the polynomial that you want to calculate ";
                cout << endl << "Enter here: ";
                int num;
                while(cin >> num)
                {
                    if(num <= 0 || num > vec_p.size())
                    {
                        cout << "Invalid input, try again!" << endl;
                        cout << "Enter here: ";
                        continue;
                    }
                    else
                    {
                        double x;
                        cout << "Enter the value of x: ";
                        cin >> x;
                        if(x == 0)
                        {
                            Node* cur = vec_p[num - 1].Nodeptr->next;
                            while(cur->next != NULL) //由于没有负数幂，故零次幂只能在最后
                            {
                                cur = cur->next;
                            }
                            if(cur->exp == 0)
                            {
                                cout << "0^0 is ilegal!" << endl;
                                cont();
                                break;
                            }
                        }
                        vec_p[num - 1].show_at_x(x);
                        cont();
                        break;
                    }
                }
            }
        }
        case 7:
        {
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
        status_num = 0;
    }
}

