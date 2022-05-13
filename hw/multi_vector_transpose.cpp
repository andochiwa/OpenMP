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
double gemm_transpose(const vector<vector<T>>& a, const vector<vector<T>>& b, vector<vector<T>>& c, int size) {
    double start, end;
    start = omp_get_wtime();
    vector<T> a2(size * size), b2(size * size);
#pragma omp parallel for shared(a2, b2, a, b, size) schedule(dynamic)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a2[i * size + j] = a[i][j];
            b2[i * size + j] = b[i][j];
        }
    }
    vector<T> b3 = transpose(b2);
#pragma omp parallel for shared(a2, b3, c, size) schedule(dynamic)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            T temp = 0;
            for (int k = 0; k < size; k++) {
                temp += a2[i * size + k] * b3[j * size + k];
            }
            c[i][j] = temp;
        }
    }
    end = omp_get_wtime();
    cout << "execution time = " << end - start << endl;
    return end - start;
}

int main() {
    // initial
    int size = 4000;
    int thread_size = 1;
    cout << "please enter the threads size:";
    cin >> thread_size;
    vector<vector<int>> a(size, vector<int> (size)), b(size, vector<int> (size));
    vector<vector<int>> c(size, vector<int> (size, 0));
    vector<double> counts;
    uniform_int_distribution<int> u(INT_MIN / 2, INT_MAX / 2);
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
        double time = gemm_transpose(a, b, c, size);
        counts.push_back(time);
    }

    write_to_file(thread_size, counts);

    return 0;
}