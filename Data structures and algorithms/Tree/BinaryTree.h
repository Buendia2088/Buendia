#ifndef BinaryTree_H_
#define BinaryTree_H_
#include <iostream>
using namespace std;

template<class T>
class BinaryTree;

template<class T>
class BinaryTreeNode
{
    friend class BinaryTree<T>;
    //friend class BinarySearchTree<T>;

private:
    T data;
    BinaryTreeNode<T>* left;
    BinaryTreeNode<T>* right;

public:
    BinaryTreeNode(const T& ele = -114514, BinaryTreeNode<T>* l = NULL, BinaryTreeNode<T>* r = NULL);
    ~BinaryTreeNode() {}
    T value() const {return data;}
    //BinaryTreeNode<T>& operator=(const BinaryTreeNode<T>& Node) {this = Node}
    BinaryTreeNode<T>* leftchild() const {return left;}
    BinaryTreeNode<T>* rightchild() const {return right;}
    void setLeftchild(BinaryTreeNode<T>* l) {left = l;}
    void setRightchild(BinaryTreeNode<T>* r) {right = r;}
    void setData(const T& v) {data = v;}
    void print() {cout << "node: " << data << ", " << "left: " << left->data << ", " << "right: " << right->data << endl;}
    bool isLeaf() const;
};

template<class T>
class BinaryTree
{
public:
    BinaryTreeNode<T>* root;
    BinaryTree(BinaryTreeNode<T>* r = NULL): root(r) {}
    ~BinaryTree() {DeleteBinaryTree(root);}
    bool IsEmpty() const;
    void visit(const T& data) {cout << data << " ";}
    BinaryTree<T>*& Root() {return root;}
    BinaryTree<T>* Parent(BinaryTreeNode<T>* current);
    BinaryTree<T>* LeftSibling(BinaryTreeNode<T>* current);
    BinaryTree<T>* RightSibling(BinaryTreeNode<T>* current);
    void CreateTree(const T& data, BinaryTreeNode<T>& leftTree, BinaryTreeNode<T>& rightTree);
    void CreateTree(BinaryTreeNode<T>*& r);
    void DeleteBinaryTree(BinaryTreeNode<T>* root);
    void PreOrder(BinaryTreeNode<T>* root);
    void InOrder(BinaryTreeNode<T>* root);
    void PostOrder(BinaryTreeNode<T>* root);
    void LevelOrder(BinaryTreeNode<T>* root);
};

#endif
