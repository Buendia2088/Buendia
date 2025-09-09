#include <iostream>
#include <algorithm>

using namespace std;

struct node 
{
    int data;
    node* next;
    
    node(int val) : data(val), next(nullptr) {}
};

node* list_create() 
{
    int n;
    cin >> n;
    
    if (n == 0) return nullptr;
    
    int first_data;
    cin >> first_data;
    node* head = new node(first_data);
    
    for (int i = 1; i < n; i++) 
    {
        int data;
        cin >> data;
        node* new_node = new node(data);
        
        if (data <= head->data) 
        {
            new_node->next = head;
            head = new_node;
        } 
        else 
        {
            node* cur = head;
            while (cur->next != nullptr && data > cur->next->data) 
            {
                cur = cur->next;
            }
            new_node->next = cur->next;
            cur->next = new_node;
        }
    }
    
    return head;
}

void print_list(node* head) {
    node* cur = head;
    while (cur != nullptr) {
        cout << cur->data << " ";
        cur = cur->next;
    }
    cout << endl;
}

long long move(node* head)
{
    long long steps = 0;
    while(head->next != NULL)
    {
        steps += head->data + head->next->data;
        node* new_node = new node(head->data + head->next->data);
        node* cur = head->next;
        node* temp = head;
        while (cur->next != nullptr && new_node->data > cur->next->data) 
        {
            cur = cur->next;
        }
        new_node->next = cur->next;
        cur->next = new_node;
        head = head->next->next;
        delete temp->next;
        delete temp;
    }
    return steps;
}

int main() {
    node* head = list_create();
    cout << move(head);
    return 0;
}
