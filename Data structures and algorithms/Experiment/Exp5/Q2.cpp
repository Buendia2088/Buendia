#include <iostream>
#include <cmath>
using namespace std;

struct node
{
    int data;
    node* next;
    node() : data(-1), next(NULL) {}
};

void LinearProbing(int key_arr[], int hash_table[])
{
    int* ka = new int[10];
    int* h = new int [13];
    for(int i = 0; i < 10; i++)
    {
        ka[i] = key_arr[i];
    }
    for(int i = 0; i < 13; i++)
    {
        h[i] = hash_table[i];
    }
    for(int i = 0; i < 10; i++)
    {
        int hash_num = ka[i] % 13;
        if(h[hash_num] == -1)
        {
            h[hash_num] = ka[i];
        }
        else
        {
            int cur = hash_num + 1;
            while(cur != hash_num)
            {
                if(cur == 13)
                {
                    cur = 0;
                }
                if(h[cur] == -1)
                {
                    h[cur] = ka[i];
                    break;
                }
                else
                {
                    cur++;
                }
            }
        }
    }
    cout << "Linear Probing:" << endl;
    for(int i = 0; i < 13; i++)
    {
        cout << i << ": " << h[i] << endl;
    }
    int find_times = 0;
    for(int i = 0; i < 10; i++)
    {
        int key_index = ka[i] % 13;
        if(h[key_index] == ka[i])
        {
            find_times++;
            continue;
        }
        else
        {
            find_times++;
            int cur = key_index + 1;
            while(cur != ka[i])
            {
                if(cur == 13)
                {
                    cur == 0;
                }
                find_times++;
                if(h[cur] == ka[i])
                {
                    break;
                }
                else
                {
                    cur++;
                }
            }
        }
    }
    cout << "Total finding times: " << find_times << endl;
    cout << "Average finding times: " << find_times / 10.0 << endl;
    delete[] ka;
    delete[] h;
}

int Generator(int i);

void QuadraticProbing(int key_arr[], int hash_table[])
{
    int* ka = new int[10];
    int* h = new int [13];
    for(int i = 0; i < 10; i++)
    {
        ka[i] = key_arr[i];
    }
    for(int i = 0; i < 13; i++)
    {
        h[i] = hash_table[i];
    }
    int find_times = 0;
    for(int i = 0; i < 10; i++)
    {
        find_times++;
        int key = ka[i];
        int key_index = key % 13;
        int count = 1;
        int flag = 0;
        while(h[key_index] != -1)
        {
            key_index = key % 13;
            find_times++;
            key_index += Generator(count);
            count++;
            if(key_index >= 13)
            {
                key_index -= 13;
            }
        }
        h[key_index] = key;
    }
    cout << "Quadratic Probing: " << endl;
    for(int i = 0; i < 13; i++)
    {
        cout << h[i] << endl;
    }
    cout << "Total finding times: " << find_times << endl;
    cout << "Average finding times: " << find_times / 10.0 << endl;
    delete[] ka;
    delete[] h;
}

int Generator(int i)
{
    if(i == 1)
    {
        return 1;
    }
    if(i % 2 == 0)
    {
        return  pow((i / 2) + 1, 2);
    }
    else
    {
        return -pow((i / 2) + 1, 2);
    }
}

void ChainAddress(int key_arr[], int hash_table[])
{
    int* ka = new int[10];
    node* h = new node[13];
    for(int i = 0; i < 10; i++)
    {
        ka[i] = key_arr[i];
    }
    int find_times = 0;
    for(int i = 0; i < 10; i++)
    {
        find_times++;
        int key = ka[i];
        int key_index = key % 13;
        if(h[key_index].data != -1)
        {
            node* cur = &h[key_index];
            while(cur->next != NULL)
            {
                find_times++;
                cur = cur->next;
            }
            find_times++;
            node* new_node = new node;
            new_node->data = key;
            cur->next = new_node;
        }
        else
        {
            h[key_index].data = key;
        }
    }
    cout << "Chain Address:" << endl;
    for(int i = 0; i < 13; i++)
    {
        if(h[i].data == -1)
        {
            cout << -1 << endl;
            continue;
        }
        cout << h[i].data;
        node* cur = &h[i];
        while(cur->next != NULL)
        {
            cur = cur->next;
            cout << " -> " << cur->data; 
        }
        cout << " -> NULL" << endl;
    }
    cout << "Total finding times: " << find_times << endl;
    cout << "Average finding times: " << find_times / 10.0 << endl;
    delete[] ka;
    delete[] h;
}

int main()
{
    int key_arr[] = {5, 88, 12, 56, 71, 28, 33, 43, 93, 17};
    int hash[100] = {0};
    for(int i = 0; i < 100; i++)
    {
        hash[i] = -1;
    }
    cout << "-------------------------------------" << endl;
    LinearProbing(key_arr, hash);
    cout << "-------------------------------------" << endl;
    QuadraticProbing(key_arr, hash);
    cout << "-------------------------------------" << endl;
    ChainAddress(key_arr, hash);
    cout << "-------------------------------------" << endl;    
    return 0;
}