
#include <stdio.h>
#include <time.h>

void poly(const double a[], double x, long degree, double *result) {
    long i;
    double r = a[degree];
    for (i = degree - 1; i >= 0; i--) {
        r = a[i] + r * x;
    }
    *result = r;
    return;
}

int main() {
    // 多项式系数：例如 f(x) = 1 + 2x + 3x^2 + ... + (n+1)x^n
    const long degree = 1000000;
    double a[degree + 1];
    for (long i = 0; i <= degree; ++i) {
        a[i] = i + 1;
    }

    double x = 1.0001;
    double result;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    poly(a, x, degree, &result);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("Result: %f\n", result);
    printf("Time elapsed: %.9f seconds\n", elapsed);
    return 0;
}