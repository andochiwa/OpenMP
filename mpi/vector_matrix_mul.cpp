#include <iostream>
#include <random>
#include <mpi.h>
#include <array>

using namespace std;

const int dim = 10000;

void row_matrix_vector_multiply(array<int, dim * dim>& mat, array<int, dim>& vec, array<int, dim>& res) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    vector<int> localResult(dim / size, 0);
    vector<int> localMat(dim * dim / size);

    MPI_Barrier(MPI_COMM_WORLD);
    double timer = MPI_Wtime();
    // Send each row of the matrix to each process
    MPI_Scatter(mat.data(), dim * dim / size, MPI_INT, localMat.data(), dim * dim / size, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast vector
    MPI_Bcast(vec.data(), dim, MPI_INT, 0, MPI_COMM_WORLD);
    // Computing
    for (int i = 0; i < dim / size; i++) {
        for (int j = 0; j < dim; j++) {
            localResult[i] += vec[j] * localMat[i * dim + j];
        }
    }
    // Gather result
    MPI_Gather(localResult.data(), dim / size, MPI_INT, res.data(), dim / size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    timer = MPI_Wtime() - timer;
    if (rank == 0) {
        cout << "Runtime = " << timer << "\n";
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    array<int, dim * dim> mat{};
    array<int, dim> vec{};
    array<int, dim> res{};
    uniform_int_distribution<int> u(5, 100);
    default_random_engine e;
    for (int i = 0; i < dim; i++) {
        vec[i] = u(e);
        for (int j = 0; j < dim; j++) {
            mat[i * dim + j] = u(e);
        }
    }

    row_matrix_vector_multiply(mat, vec, res);
    if (rank == 0) {
        array<int, dim> rres{};
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                rres[i] += vec[j] * mat[i * dim + j];
            }
        }
        cout << "res == " << (res == rres);
    }
    MPI_Finalize();
    return 0;
}

