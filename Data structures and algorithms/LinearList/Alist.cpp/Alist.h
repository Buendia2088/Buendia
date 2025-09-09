#ifndef ALIST_H_
#define ALIST_H_
#include "../LinearList.h"
using namespace std;

template <class Elem>
class Alist:public List <Elem>
{
private:
    int maxSize;
    int listSize;
    int curr;
    Elem * listArray;
public:
    Alist(int size = 0)
    {
        maxSize = size;
        listSize = 0;
        curr = 0;
        listArray = new Elem[maxSize];
    }

    ~Alist() {delete[] listArray;}

    void clear() override {listSize = curr = 0;}

    bool insert(const Elem& it) override;

    bool append(const Elem& it);

    bool remove(Elem& it) override;

    void print() override;

    void setCurr(const Elem& it);

    bool IsEmpty() {return true;}



    /*
    bool setPos(int pos);

    void Prev() {if(curr > 0) cur--;}

    void Next() {if(curr < listSize - 1) cur++;}

    */

};

#endif