#include <vector>
#include <random>
#include <omp.h>
#include <fstream>

using namespace std;

void write_to_file(int& thread_size, vector<double>& counts) {
    ofstream ofs;
    ofs.open("data.txt", ios::out);
    ofs << "thread_size: " << thread_size << ", ";
    for (auto& c : counts) {
        ofs << c << " ";
    }
    ofs << endl << "average: " << accumulate(counts.begin(), counts.end(), 0.0) / counts.size();
    ofs.close();
}

int main() {
    // initial
    int size = 2000;
    int thread_size = 16;
    vector<vector<int>> a(size, vector<int> (size)), b(size, vector<int> (size));
    vector<vector<int>> c(size, vector<int> (size, 0));
    vector<double> counts;
    uniform_int_distribution<int> u(INT_MIN, INT_MAX);
    default_random_engine e;
    omp_set_num_threads(thread_size);

    // computing
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            a[i][j] = u(e);
            b[i][j] = u(e);
        }
    }

    double start, end;

    for (int freq = 0; freq < 10; freq++) {
        start = omp_get_wtime();
        #pragma omp parallel for shared(a, b, c, size)
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                for (int k = 0; k < size; k++) {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        end = omp_get_wtime();
        printf("execution time = %f\n", end - start);
        counts.push_back(end - start);
    }

    write_to_file(thread_size, counts);

    return 0;
}



