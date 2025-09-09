#include "poly.h"
#include <stdio.h>
#include <time.h>

void poly_optim(const double *a, double x, long degree, double *result)
{
    double x2 = x * x;
    double y  = x2 * x2 * x2 * x2; // get x^8

    double u0 = 0.0, u1 = 0.0, u2 = 0.0, u3 = 0.0,
    u4 = 0.0, u5 = 0.0, u6 = 0.0, u7 = 0.0; // 8 chains

    // initialize the chains and ensure the rest of a[] is a multiple of 8
    long i = degree;
    switch ((degree + 1) % 8) {
        case 7: u6 = a[i--];
        case 6: u5 = a[i--];
        case 5: u4 = a[i--];
        case 4: u3 = a[i--];
        case 3: u2 = a[i--];
        case 2: u1 = a[i--];
        case 1: u0 = a[i--];
        default: ;
    }

    for (; i >= 7; i -= 8) {
        u7 = u7 * y + a[i];
        u6 = u6 * y + a[i - 1];
        u5 = u5 * y + a[i - 2];
        u4 = u4 * y + a[i - 3];
        u3 = u3 * y + a[i - 4];
        u2 = u2 * y + a[i - 5];
        u1 = u1 * y + a[i - 6];
        u0 = u0 * y + a[i - 7];
    }

    double r = u7;
    r = r * x + u6;
    r = r * x + u5;
    r = r * x + u4;
    r = r * x + u3;
    r = r * x + u2;
    r = r * x + u1;
    r = r * x + u0;

    *result = r;
}


void measure_time(poly_func_t poly, const double a[], double x, long degree, double *time) 
{
    struct timespec start, end;
    double result;
    int loop = 1e2; // measure multiple times and calculate the mean
    *time = 0;

    // get the cost of clock_gettime()
    double time_of_gettime = 0;
    struct timespec gettime_start, gettime_end;
    clock_gettime(CLOCK_MONOTONIC, &gettime_start);
    clock_gettime(CLOCK_MONOTONIC, &gettime_end);
    time_of_gettime += (gettime_end.tv_sec - gettime_start.tv_sec) * 1e9;
    time_of_gettime += (gettime_end.tv_nsec - gettime_start.tv_nsec);

    poly(a, x, degree, &result); // initialize cache

    for(int i = 0; i < loop; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        poly(a, x, degree, &result);
        clock_gettime(CLOCK_MONOTONIC, &end);
        *time += (end.tv_sec - start.tv_sec) * 1e9;
        *time += (end.tv_nsec - start.tv_nsec);
        *time -= time_of_gettime * 2; // get rid of the cost of clock_gettime()
    }

    *time /= loop;
}