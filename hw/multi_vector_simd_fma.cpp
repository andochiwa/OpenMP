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

void do_block(int n, vector<double>& a, vector<double>& b, vector<double>& c, int unroll, int block,
              int ii, int jj, int kk) {
    for (int i = ii; i < min(ii + block, n); i++) {
        for (int j = jj; j < min(jj + block, n); j += 4 * unroll) {
            __m256d cx[unroll];
            for (int x = 0; x < unroll; x++) {
                cx[x] = _mm256_load_pd(&c[i * n + j]);
            }
            for (int k = kk; k < min(kk + block, n); k++) {
                auto bc_a = _mm256_set1_pd(a[i * n + k]);
                for (int x = 0; x < unroll; x++) {
                    auto vec_b = _mm256_load_pd(&b[k * n + j + (x * 4)]);
                    cx[x] = _mm256_fmadd_pd(bc_a, vec_b, cx[x]);
                }
            }
            for (int x = 0; x < unroll; x++) {
                _mm256_store_pd(&c[i * n + j + (x * 4)], cx[x]);
            }
        }
    }
}

double gemm_simd_unroll_block(int n, vector<double>& a, vector<double>& b, vector<double>& c, int unroll) {
    double start = omp_get_wtime();
    int block = unroll * 16; // Requirement >= unroll * memory alignment size
#pragma omp parallel for collapse(3) default(shared)
    for (int ii = 0; ii < n; ii += block) { // block
        for (int jj = 0; jj < n; jj += block) { // block
            for (int kk = 0; kk < n; kk += block) { // block
                do_block(n, a, b, c, unroll, block, ii, jj, kk);
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
    vector<double> a(size * size), b(size * size);
    vector<double> c(size * size, 0), d(size * size, 0);
    vector<double> counts;
    uniform_real_distribution<double> u((double) INT_MIN / 2, (double) INT_MAX / 2);
    default_random_engine e;
    omp_set_num_threads(thread_size);

    for (int i = 0; i < size * size; i++) {
        a[i] = u(e);
        b[i] = u(e);
    }

    // computing
    for (int freq = 0; freq < 5; freq++) {
        double time = gemm_simd_unroll_block(size, a, b, c, unroll);
        counts.push_back(time);
    }

    write_to_file(thread_size, counts);

    return 0;
}