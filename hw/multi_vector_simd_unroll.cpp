#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <fstream>
#include <immintrin.h>

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

double gemm_simd_unroll(int n, vector<int>& a, vector<int>& b, vector<int>& c, int unroll) {
    double start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic) default(shared)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 8 * unroll) {
            __m256i cx[unroll]; // unroll array
            for (int x = 0; x < unroll; x++) {
                cx[x] = _mm256_setzero_si256(); // cx[x] = 0
            }
            for (int k = 0; k < n; k++) {
                auto bc_a = _mm256_set1_epi32(a[i * n + k]); // load 8 int from a
                for (int x = 0; x < unroll; x++) {
                    // vec_b = b[k][j + (x + 8)] ~ b[k][j * 8 + (x * 8)]
                    auto vec_b = _mm256_load_si256((__m256i*) &b[k * n + j + (x * 8)]);
                    // res = bc_a * vec_b
                    auto res = _mm256_mullo_epi32(bc_a, vec_b);
                    // cx[x] = res
                    cx[x] = _mm256_add_epi32(cx[x], res);
                }
            }
            for (int x = 0; x < unroll; x++) {
                // write to c[i][j + (x * 8)] from cx[x]
                _mm256_store_si256((__m256i*) &c[i * n + j + (x * 8)], cx[x]);
            }
        }
    }
    double end = omp_get_wtime();
    cout << "execution time = " << end - start << endl;
    return end - start;
}

int main() {
    // initial
    int size = 2000;
    int unroll = 4;
    int thread_size = 1;
    cout << "please enter the threads size:";
    cin >> thread_size;
    vector<int> a(size * size), b(size * size);
    vector<int> c(size * size, 0);
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
        double time = gemm_simd_unroll(size, a, b, c, unroll);
        counts.push_back(time);
    }

    write_to_file(thread_size, counts);

    return 0;
}