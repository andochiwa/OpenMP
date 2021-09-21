#include <iostream>
#include <omp.h>

using namespace std;

int main() {
#pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        printf("I am thread %d\n", omp_get_thread_num());
    }

    return 0;
}

