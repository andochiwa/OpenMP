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
double gemm(const vector<vector<T>>& a, const vector<vector<T>>& b, vector<vector<T>>& c, int size) {
    double start, end;
    start = omp_get_wtime();
    vector<T> a2(size * size), b2(size * size);
#pragma omp parallel for shared(a2, b2, a, b, size)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a2[i * size + j] = a[i][j];
            b2[i * size + j] = b[i][j];
        }
    }
#pragma omp parallel for shared(a2, b2, c, size)
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            T temp = 0;
            for (int k = 0; k < size; k++) {
                temp += a2[i * size + k] * b2[k * size + j];
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
    int size = 2000;
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
        double time = gemm(a, b, c, size);
        counts.push_back(time);
    }

    write_to_file(thread_size, counts);

    return 0;
}



