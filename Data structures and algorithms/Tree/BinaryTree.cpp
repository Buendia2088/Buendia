#include "BinaryTree.h"
#include <iostream>
#include <stack>
#include <queue>
using namespace std;


template<class T>
BinaryTreeNode<T>::BinaryTreeNode(const T& ele, BinaryTreeNode<T>* l, BinaryTreeNode<T>* r)
{
    data = ele;
    left = l;
    right = r;
}

template<class T>
bool BinaryTreeNode<T>::isLeaf() const
{
    if(left = NULL && right = NULL)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template<class T>
bool BinaryTree<T>::IsEmpty() const
{
    if(root == NULL)
    {
        return true;
    }
    else
    {
        return false;
    }
}

template<class T>
BinaryTree<T>* BinaryTree<T>::Parent(BinaryTreeNode<T>* current)
{
    // 如果当前节点为空或者根节点为空，返回空指针
    if (current == NULL || root == NULL)
        return NULL;
    
    // 如果当前节点就是根节点，没有父节点，返回空指针
    if (root == current)
        return NULL;
    
    // 从根节点开始进行深度优先搜索查找父节点
    stack<BinaryTreeNode<T>*> nodesStack;
    nodesStack.push(root);
    BinaryTreeNode<T>* parent = NULL;
    
    while (!nodesStack.empty())
    {
        BinaryTreeNode<T>* node = nodesStack.top();
        nodesStack.pop();
        
        // 检查当前节点的左右子节点是否是要查找的节点
        if (node->left == current || node->right == current)
        {
            parent = node;
            break;
        }
        
        // 如果当前节点的右子节点不为空，将其入栈
        if (node->right != NULL)
            nodesStack.push(node->right);
        
        // 如果当前节点的左子节点不为空，将其入栈
        if (node->left != NULL)
            nodesStack.push(node->left);
    }
    
    return parent;
}

template<class T>
BinaryTree<T>* BinaryTree<T>::LeftSibling(BinaryTreeNode<T>* current)
{
    return current->left;
}

template<class T>
BinaryTree<T>* BinaryTree<T>::RightSibling(BinaryTreeNode<T>* current)
{
    return current->right;
}

template<class T>
void BinaryTree<T>::CreateTree(const T& data_, BinaryTreeNode<T>& leftTree, BinaryTreeNode<T>& rightTree)
{
    data = data_;
    root->setLeftchild(leftTree.root);
    root->setRightchild(rightTree.root);
}

template<class T>
void BinaryTree<T>::CreateTree(BinaryTreeNode<T>*& r)
{
    int ch;
    cin >> ch;
    if(ch == -1) r = NULL;
    else
    {
        r = new BinaryTreeNode<T> (ch);
        CreateTree(r->left);
        CreateTree(r->right);
    }
    
}

template<class T>
void BinaryTree<T>::DeleteBinaryTree(BinaryTreeNode<T>* root)
{
    if (root == nullptr) {
        return;
    }
    DeleteBinaryTree(root->left);
    DeleteBinaryTree(root->right);
    
    // 释放当前节点
    delete root;
}

template<class T>
void BinaryTree<T>::PreOrder(BinaryTreeNode<T>* root)
{
    if(root == NULL)
    {
        return;
    }
    visit(root->value());
    PreOrder(root->leftchild());
    PreOrder(root->rightchild());
}   

template<class T>
void BinaryTree<T>::InOrder(BinaryTreeNode<T>* root)
{
    if(root == NULL)
    {
        return;
    }
    InOrder(root->leftchild());
    visit(root->value());
    InOrder(root->rightchild());
}   

template<class T>
void BinaryTree<T>::PostOrder(BinaryTreeNode<T>* root)
{
    if(root == NULL)
    {
        return;
    }
    PostOrder(root->leftchild());
    PostOrder(root->rightchild());
    visit(root->value());
}   

template<class T>
void BinaryTree<T>::LevelOrder(BinaryTreeNode<T>* root)
{
    queue<BinaryTreeNode<T>*> tQueue;
    BinaryTreeNode<T>* pointer = root;
    if(pointer) tQueue.push(pointer);
    while(!tQueue.empty())
    {
        pointer = tQueue.front();
        tQueue.pop();
        visit(pointer->value());
        if(pointer->leftchild() != NULL)
        {
            tQueue.push(pointer->leftchild());
        }
        if(pointer->rightchild() != NULL)
        {
            tQueue.push(pointer->rightchild());
        }
    }

}

int main()
{
    BinaryTree<int> tree;
    tree.CreateTree(tree.root);
    tree.LevelOrder(tree.root);

   return 0;
}