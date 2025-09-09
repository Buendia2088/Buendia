#include <iostream>
#include <vector>

using namespace std;

string convertToBase(int N, int b) {
    string result = "";
    while (N > 0) {
        int digit = N % b;
        char ch = (digit < 10) ? ('0' + digit) : ('A' + digit - 10);
        result = ch + result;
        N /= b;
    }
    return result;
}

bool isPalindrome(const string& str) {
    int left = 0, right = str.length() - 1;
    while (left < right) {
        if (str[left] != str[right]) {
            return false;
        }
        left++;
        right--;
    }
    return true;
}

int main() {
    int N, b;
    cin >> N >> b;

    string baseB = convertToBase(N, b);
    bool isPalin = isPalindrome(baseB);

    if (isPalin) {
        cout << "Yes" << endl;
    } else {
        cout << "No" << endl;
    }

    for (int i = 0; i < baseB.length(); i++) {
        cout << baseB[i];
        if (i < baseB.length() - 1) {
            cout << " ";
        }
    }

    return 0;
}
