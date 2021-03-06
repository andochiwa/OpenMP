#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <fstream>

using namespace std;

void write_to_file(int thread_size, vector<double>& counts) {
    ofstream ofs;
    string file_name = "data" + to_string(thread_size) + ".txt";
    ofs.open(file_name, ios::out);
    ofs << "thread_size: " << thread_size << ", ";
    for (auto& c : counts) {
        ofs << c << " ";
    }
    ofs << endl << "average: " << accumulate(counts.begin(), counts.end(), 0.0) / (double) counts.size();
    ofs.close();
}

template<typename T>
vector<T> transpose(const vector<T>& vec) {
    int n = sqrt(vec.size());
    vector<T> res (n * n);
#pragma omp parallel for shared(res, vec, n) schedule(dynamic)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res[j * n + i] = vec[i * n + j];
        }
    }
    return res;
}

template<typename T>
double gemm_transpose_block(const vector<vector<T>>& a, const vector<vector<T>>& b, vector<vector<T>>& c, int size) {
    double start, end;
    int block_size = 8;
    start = omp_get_wtime();
    vector<T> a2(size * size), b2(size * size);
#pragma omp parallel for shared(a2, b2, a, b, size) schedule(dynamic)
    // 2D to 1D array
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a2[i * size + j] = a[i][j];
            b2[i * size + j] = b[i][j];
        }
    }
    // transpose matrix
    vector<T> b3 = transpose(b2);
#pragma omp parallel for collapse(5) default(shared)
    // Block
    for (int ii = 0; ii < size; ii += block_size) {
        for (int jj = 0; jj < size; jj += block_size) {
            for (int kk = 0; kk < size; kk += block_size) {
                // computing
                for (int i = ii; i < ii + block_size; i++) {
                    for (int j = jj; j < jj + block_size; j++) {
                        T temp = 0;
                        for (int k = kk; k < kk + block_size; k++) {
                            temp += a2[i * size + k] * b3[j * size + k];
                        }
                        c[i][j] += temp;
                    }
                }
            }
        }
    }
    end = omp_get_wtime();
    cout << "execution time = " << end - start << endl;
    return end - start;
}

int main() {
    // initial
    int size = 2000;
    int thread_size = 1;
    cout << "please enter the threads size:";
    cin >> thread_size;
    vector<vector<double>> a(size, vector<double> (size)), b(size, vector<double> (size));
    vector<vector<double>> c(size, vector<double> (size, 0));
    vector<double> counts;
    uniform_real_distribution<double> u((double) INT_MIN / 2, (double) INT_MAX / 2);
    default_random_engine e;
    omp_set_num_threads(thread_size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = u(e);
            b[i][j] = u(e);
        }
    }

    // computing
    for (int freq = 0; freq < 5; freq++) {
        double time = gemm_transpose_block(a, b, c, size);
        counts.push_back(time);
    }

    write_to_file(thread_size, counts);

    return 0;
}