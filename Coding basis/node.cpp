#include <iostream> using namespace std;
#define nullptr NULL
struct Node {
int data;
Node* next;
Node() : data(0), next(nullptr) {}
Node(int data_) : data(data_), next(nullptr) {}
}; //这段结构体定义我们课堂上没有讲过，不过不影响答题，可以理解为这个结构体类型的两个构造函数，C++把结构体和类不太区别

class LinkedList {
private:
Node* head;
public:
LinkedList() {head = new Node;}
int AddNode(Node* node);
int DeleteNodeByData(int data);
void PrintList();
};

int LinkedList::AddNode(Node* node)
{
    Node* cur = head;
    while(cur->next != nullptr && cur->next->data < node->data)
    {
        cur = cur->next;
    }
    if(cur->next != nullptr && cur->next->data == node->data)
    {
        return 0;
    }
    node->next = cur->next;
    cur->next = node;
    return 1;
}

int LinkedList::DeleteNodeByData(int data)
{
    Node* cur = head;
    while(cur->next != nullptr && cur->next->data < data)
    {
        cur = cur->next;
    }
    if(cur->next == nullptr || cur->next->data != data)
    {
        return 0;
    }

    Node* temp = cur->next;
    cur->next = cur->next->next;
    delete temp;
    return 1;
}
}

void LinkedList::PrintList()
{
Node* cur = this->head->next;
while (cur != nullptr) {
cout << cur->data << endl;
cur = cur->next;
}
}

int main(int argc, char const *argv[])
{
LinkedList list;
int n, d;
cin >> n;
for (int i=0; i<n; i++) {
cin >> d;
Node* node = new Node(d);
list.AddNode(node);
}
list.PrintList();
cin >> n;
for (int i=0; i<n; i++) {
cin >> d;
list.DeleteNodeByData(d);
}
list.PrintList();
return 0;
}
