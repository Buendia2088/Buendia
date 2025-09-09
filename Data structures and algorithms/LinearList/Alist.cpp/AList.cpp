#include <iostream>
#include "../LinearList.h"
#include "Alist.h"
using namespace std;

template <class Elem>
bool Alist<Elem>::insert(const Elem& it)
{
    if(listSize == maxSize)
    {
        cout << "FULL!" << endl;
        return false;
    }
    if(curr < 0 || curr > listSize - 1)  
    {
        cout << "WRONG POSITION!" << endl;
        return false;
    }
    for(int i = listSize; i > curr ; i--)
    {
        listArray[i] = listArray[i-1];
    }
    listArray[curr] = it;
    listSize++;
    return true;
}

template <class Elem>
bool Alist<Elem>::append(const Elem& it)
{
    if(listSize == maxSize)
    {
        cout << "FULL!" << endl;
        return false;
    }
    listArray[listSize++] = it;
    return true;
}

template <class Elem>
bool Alist<Elem>::remove(Elem& it)
{
    setCurr(it);
    if(listSize == 0)
    {
        cout << "EMPTY!" << endl;
        return false;
    }
    if(curr < 0 || curr > listSize - 1)  
    {
        cout << "WRONG POSITION!" << endl;
        return false;
    }    
    it = listArray[curr];
    for(int i = curr; i < listSize - 1; i++)
    {
        listArray[i] = listArray[i+1];
    }
    listSize--;
    return true;
}

template <class Elem>
void Alist<Elem>::print()
{
    for(int i = 0; i < listSize; i++)
    {
        cout << listArray[i] << " ";
    }
    cout << endl;
}

template <class Elem>
void Alist<Elem>::setCurr(const Elem& it) {curr = it;}

int main()
{
    Alist<int> list1(20);
    for(int i = 0; i < 10; i++)
    {
        list1.append(i*3);
    }
    cout << "Now list1 is: ";
    list1.print();
    cout << "Delete: ";
    int tar;
    cin >> tar;
    list1.remove(tar);
    cout << "List1 after removal is: ";
    list1.print();
    return 0;
}