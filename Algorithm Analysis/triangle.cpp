#include <iostream>
#include <vector>
using namespace std;

// 全局变量声明
int n, half, count, sum;
vector<vector<int>> p;

// 函数声明
int solution();
void backtrace(int t);

int main() {
    // 测试用例
    cout << "请输入符号三角形的大小 n: ";
    cin >> n;

    if (n <= 0) {
        cout << "n 必须是一个正整数。" << endl;
        return 1;
    }

    // 初始化全局变量
    sum = 0;   // 符号三角形的方案总数
    count = 0; // 当前的累计和

    // 执行算法
    int result = solution();

    // 输出结果
    if (result == 0) {
        cout << "该三角形无法形成符号三角形。" << endl;
    } else {
        cout << "符号三角形的方案总数为: " << result << endl;
    }

    return 0;
}

int solution() {
    half = (n * (n + 1)) / 2; // 计算所有符号之和
    if (half % 2 == 1) {
        return 0; // 如果符号和是奇数，直接返回 0
    }
    half /= 2; // 目标为符号和的一半

    // 初始化二维数组 p
    p.resize(n + 1);
    for (int i = 0; i <= n; ++i) {
        p[i].resize(n + 1, 0);
    }

    // 开始回溯
    backtrace(1);

    return sum;
}

void backtrace(int t) {
    // 递归终止条件：如果已经构建到第 n 行
    if (t > n) {
        if (count == half) {
            sum++; // 如果当前符号和等于目标值，计数+1
        }
        return;
    }

    // 遍历当前层的所有可能值（0 或 1）
    for (int i = 0; i < 2; i++) {
        p[1][t] = i; // 设置当前层的第 t 个符号
        count += i;  // 更新累计和

        // 计算当前列以下的符号值
        for (int j = 2; j <= t; j++) {
            p[j][t - j + 1] = p[j - 1][t - j + 1] ^ p[j - 1][t - j + 2];
            count += p[j][t - j + 1];
        }

        // 剪枝：只有当前累计和小于等于目标值时，继续递归
        if (count <= half) {
            backtrace(t + 1);
        }

        // 回溯：清除当前层的状态
        for (int j = 2; j <= t; j++) {
            count -= p[j][t - j + 1];
        }
        count -= i; // 恢复初始状态
    }
}
