#include <iostream>
#include <random>
#include <omp.h>

using namespace std;

void ADD(vector<int>& a, vector<int>& b, vector<int>& res, int n) {
#pragma omp parallel for default(shared)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = a[i * n + j] + b[i * n + j];
        }
    }
}

void SUB(vector<int>& a, vector<int>& b, vector<int>& res, int n) {
#pragma omp parallel for default(shared)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[i * n + j] = a[i * n + j] - b[i * n + j];
        }
    }
}

void MUL(vector<int>& a, vector<int>& b, vector<int>& res, int n) {
#pragma omp parallel for default(shared)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            int temp = 0;
            for (int k = 0; k < n; k++) {
                temp += a[i * n + k] * b[k * n + j];
            }
            res[i * n + j] = temp;
        }
    }
}

void strassen(vector<int>& A, vector<int>& B, vector<int>& C, int n) {
    // recursion end
    if (n <= 64 || n % 2 != 0) {
        MUL(A, B, C, n);
        return;
    }

    int half = n / 2;
    // initial
    vector<int> A11(half * half), A12(half * half), A21(half * half), A22(half * half),
    B11(half * half), B12(half * half), B21(half * half), B22(half * half),
    C11(half * half), C12(half * half), C21(half * half), C22(half * half),
    M1(half * half), M2(half * half), M3(half * half), M4(half * half),
    M5(half * half), M6(half * half), M7(half * half),
    resA(half * half), resB(half * half);

    // block A and B
#pragma omp parallel for default(shared)
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            A11[i * half + j] = A[i * n + j];
            A12[i * half + j] = A[i * n + (j + half)];
            A21[i * half + j] = A[(i + half) * n + j];
            A22[i * half + j] = A[(i + half) * n + (j + half)];

            B11[i * half + j] = B[i * n + j];
            B12[i * half + j] = B[i * n + (j + half)];
            B21[i * half + j] = B[(i + half) * n + j];
            B22[i * half + j] = B[(i + half) * n + (j + half)];
        }
    }

    // M1 = (A11 + A22) * (B11 + B22)
    ADD(A11, A22, resA, half);
    ADD(B11, B22, resB, half);
    strassen(resA, resB, M1, half);

    // M2 = (A21 + A22) * B11
    ADD(A21, A22, resA, half);
    strassen(resA, B11, M2, half);

    // M3 = A11 * (B12 - B22)
    SUB(B12, B22, resB, half);
    strassen(A11, resB, M3, half);

    // M4 = A22 * (B21 - B11)
    SUB(B21, B11, resB, half);
    strassen(A22, resB, M4, half);

    // M5 = (A11 + A12) * B22
    ADD(A11, A12, resA, half);
    strassen(resA, B22, M5, half);

    // M6 = (A21 - A11) * (B11 + B12)
    SUB(A21, A11, resA, half);
    ADD(B11, B12, resB, half);
    strassen(resA, resB, M6, half);

    // M7 = (A12 - A22) * (B21 + B22)
    SUB(A12, A22, resA, half);
    ADD(B21, B22, resB, half);
    strassen(resA, resB, M7, half);

    // C11 = (M1 + M4) + (M7 - M5)
    ADD(M1, M4, resA, half);
    SUB(M7, M5, resB, half);
    ADD(resA, resB, C11, half);

    // C12 = M3 + M5
    ADD(M3, M5, C12, half);

    // C21 = M2 + M4
    ADD(M2, M4, C21, half);

    // C22 = (M1 - M2) + (M3 + M6)
    SUB(M1, M2, resA, half);
    ADD(M3, M6, resB, half);
    ADD(resA, resB, C22, half);

    // merge C11, C12, C21, C22 to C
#pragma omp parallel for default(shared)
    for (int i = 0; i < half; i++) {
        for (int j = 0; j < half; j++) {
            C[i * n + j] = C11[i * half + j];
            C[i * n + (j + half)] = C12[i * half + j];
            C[(i + half) * n + j] = C21[i * half + j];
            C[(i + half) * n + (j + half)] = C22[i * half + j];
        }
    }
}

template<typename T>
double gemm(const vector<T>& a, const vector<T>& b, vector<T>& c, int size) {
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel for shared(a, b, c, size) schedule(dynamic)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            T temp = 0;
            for (int k = 0; k < size; k++) {
                temp += a[i * size + k] * b[k * size + j];
            }
            c[i * size + j] = temp;
        }
    }
    end = omp_get_wtime();
    cout << "execution time = " << end - start << endl;
    return end - start;
}

int main() {
    // initial
    int size = 3000;
    int thread_size = 1;
    cout << "please enter the threads size:";
    cin >> thread_size;
    vector<int> a(size * size), b(size * size);
    vector<int> c(size * size, 0), d(size * size, 0);
    vector<double> counts;
    uniform_int_distribution<int> u(INT_MIN / 2, INT_MAX / 2);
    default_random_engine e;
    omp_set_num_threads(thread_size);

    for (int i = 0; i < size * size; i++) {
        a[i] = u(e);
        b[i] = u(e);
    }

    // computing
    for (int freq = 0; freq < 5; freq++) {
        double start = omp_get_wtime();
        strassen(a, b, c, size);
        double time = omp_get_wtime() - start;
        cout << "execution time = " << time << endl;
        gemm(a, b, d, size);
        printf("%d\n", c == d);

        counts.push_back(time);
    }

    return 0;
}