#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <unordered_map>
#include <algorithm>

using namespace std;

struct Node {
    int weight;
    char symbol;
    Node* left;
    Node* right;

    Node(int w, char s = '\0', Node* l = nullptr, Node* r = nullptr) 
        : weight(w), symbol(s), left(l), right(r) {}
};

struct Compare {
    bool operator()(Node* l, Node* r) {
        return l->weight > r->weight;
    }
};

void generateCodes(Node* root, const string& prefix, unordered_map<char, string>& huffmanCodes) {
    if (root == nullptr) return;
    if (!root->left && !root->right && root->symbol != '\0') {
        huffmanCodes[root->symbol] = prefix;
    }
    generateCodes(root->left, prefix + "0", huffmanCodes);
    generateCodes(root->right, prefix + "1", huffmanCodes);
}

int calculateWPL(Node* root, int depth) {
    if (root == nullptr) return 0;
    if (!root->left && !root->right) return root->weight * depth;
    return calculateWPL(root->left, depth + 1) + calculateWPL(root->right, depth + 1);
}

void printTree(Node* root) {
    if (root == nullptr) return;
    if (!root->left && !root->right) {
        cout << root->weight;
        return;
    }
    cout << root->weight << "(";
    if (root->left) {
        printTree(root->left);
    }
    cout << ",";
    if (root->right) {
        printTree(root->right);
    }
    cout << ")";
}

void getCodesInOrder(Node* root, unordered_map<char, string>& huffmanCodes, vector<string>& codes) {
    if (root == nullptr) return;
    if (!root->left && !root->right) {
        codes.push_back(huffmanCodes[root->symbol]);
        return;
    }
    getCodesInOrder(root->left, huffmanCodes, codes);
    getCodesInOrder(root->right, huffmanCodes, codes);
}

int main() {
    int n;
    cin >> n;

    vector<int> weights(n);
    for (int i = 0; i < n; ++i) {
        cin >> weights[i];
    }

    priority_queue<Node*, vector<Node*>, Compare> pq;
    for (int i = 0; i < n; ++i) {
        pq.push(new Node(weights[i], 'A' + i));
    }

    while (pq.size() > 1) {
        Node* left = pq.top();
        pq.pop();
        Node* right = pq.top();
        pq.pop();
        Node* parent = new Node(left->weight + right->weight, '\0', left, right);
        pq.push(parent);
    }

    Node* root = pq.top();

    int wpl = calculateWPL(root, 0);
    cout << wpl << endl;

    printTree(root);
    cout << endl;

    unordered_map<char, string> huffmanCodes;
    generateCodes(root, "", huffmanCodes);

    vector<string> codes;
    getCodesInOrder(root, huffmanCodes, codes);

    for (const auto& code : codes) {
        cout << code << " ";
    }

    return 0;
}
