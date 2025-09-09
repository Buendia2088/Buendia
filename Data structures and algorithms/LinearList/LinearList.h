#ifndef LINEAR_LIST_H_
#define LINEAR_LIST_H_

template <class Elem> class List
{
public:

    virtual void clear() = 0;
    virtual bool insert(const Elem&) = 0;
    virtual bool remove(Elem&) = 0;
    virtual void print() = 0; 
    virtual bool IsEmpty() = 0;  

    /*
    virtual bool append(const Elem&) = 0;
    virtual bool setPos(int pos) = 0; 
    virtual void next() = 0;
    virtual void prev() = 0;
    virtual void setStart() = 0;
    virtual void setEnd() = 0;
    virtual bool getValue(Elem&) const = 0;
    virtual bool IsFull() = 0;  

    virtual void clear() = 0;
    virtual bool insert(const Elem&) = 0;
    virtual bool remove(Elem&) = 0;
    virtual void print() = 0; 
    virtual bool IsEmpty() = 0;  

    virtual void clear();
    virtual bool insert(const Elem&);
    virtual bool remove(Elem&);
    virtual void print(); 
    virtual bool IsEmpty();  
    */
    
};

#endif