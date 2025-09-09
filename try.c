#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int rows = 3, cols = 4;
    int **array = (int **)malloc(rows * sizeof(int *));
    if (array == NULL) {
        perror("malloc failed");
        return 1;
    }

    for (int i = 0; i < rows; i++) {
        array[i] = (int *)malloc(cols * sizeof(int));
        if (array[i] == NULL) {
            perror("malloc failed");
            return 1;
        }
    }
    
    // 示例：设置元素和打印二维数组
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            array[i][j] = i * cols + j;
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }
    
    // 释放内存
    for (int i = 0; i < rows; i++) {
        free(array[i]);
    }
    free(array);
    
    return 0;
}
