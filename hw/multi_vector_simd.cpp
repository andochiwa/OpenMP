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

double gemm_simd(int n, vector<int>& a, vector<int>& b, vector<int>& c) {
    double start = omp_get_wtime();
#pragma omp parallel for schedule(dynamic) default(shared)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j += 8) {
            auto c0 = _mm256_setzero_si256(); // c0 = 0
            for (int k = 0; k < n; k++) {
                auto bc_a = _mm256_set1_epi32(a[i * n + k]); // load 8 int from a
                auto vec_b = _mm256_load_si256((__m256i*) &b[k * n + j]); // load 8 int from b
                auto res = _mm256_mullo_epi32(bc_a, vec_b); // res = a * b
                c0 = _mm256_add_epi32(c0, res); // c0 += res
            }
            _mm256_store_si256((__m256i*) &c[i * n + j], c0); // write 8 int to c from c0
        }
    }
    double end = omp_get_wtime();
    cout << "execution time = " << end - start << endl;
    return end - start;
}

int main() {
    // initial
    int size = 4000;
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
        double time = gemm_simd(size, a, b, c);
        counts.push_back(time);
    }

    write_to_file(thread_size, counts);

    return 0;
}