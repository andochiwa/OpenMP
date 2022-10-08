#include <mpi.h>
#include <iostream>
using namespace std;

const int MAX_STRING = 100;

int main() {
    string greeting;
    int comm_sz;
    int my_rank;

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank != 0) {
        greeting = "Greetings from process " + to_string(my_rank) + " of " + to_string(comm_sz);
        MPI_Send(greeting.c_str(), greeting.size(), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    } else {
        printf("Greetings from process %d of %d!\n", my_rank, comm_sz);
        for (int q = 1; q < comm_sz; ++q) {
            char buf[55];
            MPI_Recv(buf, MAX_STRING, MPI_CHAR, q, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            greeting = string(buf);
            cout << greeting << "\n";
        }
    }
    MPI_Finalize();
    return 0;

}