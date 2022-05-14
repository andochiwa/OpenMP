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

double gemm_avx_unroll_block(int n, vector<int>& a, vector<int>& b, vector<int>& c, int unroll) {
    double start = omp_get_wtime();
    int block = unroll * 16; // Requirement >= unroll * memory alignment size
#pragma omp parallel for collapse(5) default(shared)
    for (int ii = 0; ii < n; ii += block) {
        for (int jj = 0; jj < n; jj += block) {
            for (int kk = 0; kk < n; kk += block) {
                for (int i = ii; i < ii + block; i++) {
                    for (int j = jj; j < jj + block; j += 8 * unroll) {
                        __m256i cx[unroll];
                        for (int x = 0; x < unroll; x++) {
                            cx[x] = _mm256_load_si256((__m256i*) &c[i * n + j]);
                        }
                        for (int k = kk; k < kk + block; k++) {
                            auto bc_a = _mm256_set1_epi32(a[i * n + k]);
                            for (int x = 0; x < unroll; x++) {
                                auto vec_b = _mm256_load_si256((__m256i*) &b[k * n + j + (x * 8)]);
                                auto res = _mm256_mullo_epi32(bc_a, vec_b);
                                cx[x] = _mm256_add_epi32(cx[x], res);
                            }
                        }
                        for (int x = 0; x < unroll; x++) {
                            _mm256_store_si256((__m256i*) &c[i * n + j + (x * 8)], cx[x]);
                        }
                    }
                }
            }
        }
    }
    double end = omp_get_wtime();
    cout << "execution time = " << end - start << endl;
    return end - start;
}

int main() {
    // initial
    int size = 2048;
    int unroll = 4;
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
    for (int freq = 0; freq < 1; freq++) {
        double time = gemm_avx_unroll_block(size, a, b, c, unroll);
        counts.push_back(time);
    }

    write_to_file(thread_size, counts);

    return 0;
}