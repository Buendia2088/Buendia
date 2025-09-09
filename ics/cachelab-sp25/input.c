#include <stdio.h>
#include <stdlib.h>

int main() {
    char operation;    // 存储操作类型 (I 或 S)
    unsigned long long address;  // 存储地址，使用 long long 来处理大于 32 位的十六进制地址
    int length;        // 存储长度（整数）
    
    // 输入格式：操作类型 地址,长度
    while (scanf("%c %llx,%d", &operation, &address, &length) == 3) {
        // 输出分割的结果
        printf("Operation: %c\n", operation);
        printf("Address: %llx\n", address);
        printf("Length: %d\n", length);
        
        // 读取下一行
        getchar(); // 清除换行符
    }

    return 0;
}