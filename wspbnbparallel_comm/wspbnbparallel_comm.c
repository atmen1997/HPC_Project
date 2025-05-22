#include <stdio.h>      // For printf, fopen
#include <stdlib.h>     // For standard library functions, malloc, free
#include <limits.h>     // For INT_MAX
#include <stdbool.h>    // For bool, true, false
#include <string.h>     // For strcmp, memcpy
#include <mpi.h>        // For MPI functions
#include <time.h>       // For MPI_Wtime

#define MAX_CITIES 17 // Maximum number of cities from the input
// (Recall that city 0 is fixed at the beginning so the generated tasks will have DEPTH cities, 
//  with positions 1..(DEPTH-1) chosen from {1,...,num_cities-1}.)
#define DEPTH 6
// Maximum number of tasks we can hold can be calculated with 𝑃(𝑛,d)=(𝑛-1)!/(𝑛-d-1)!, where n = num_cities and d = depth (-1 because city 0 is fixed)
#define MAX_TASKS 600000

// MPI Tags
#define TAG_TASK_REQUEST 1      // Workers send task requests to master
#define TAG_TASK_ASSIGN  2      // Master sends task assignments or termination signals to workers
#define TAG_NEW_BEST     3      // Workers send new best solutions to master
#define TAG_UPDATED_BEST 4      // Master sends updated best solution back to worker

// Structure to encapsulate a best solution (distance + path)
// For simplified communication between workers and master.
typedef struct {
    int distance;
    int path[MAX_CITIES];
} BestSolution;

// Global variables to store problem data and best result
int num_cities = 0;                         // Total number of cities (read from input)
int distances[MAX_CITIES * MAX_CITIES] = {0}; // 1D array for the distance matrix (row-major order)
int best_distance = INT_MAX;                // Global best distance found so far
int best_path[MAX_CITIES];                  // Global best path found so far
int rank, size;                           // MPI process rank and total number of processes

// Timers for Comm/Comp
double local_comm_time = 0.0;  // Time spent in MPI calls
double local_comp_time = 0.0;  // Time spent in branch-and-bound computation

// --- Function Prototypes ---
void  read_input(FILE *file);
void  master_work_assignment(void);
void  worker_execution(void);
void  branch_and_bound_from_partial(int partial_path[], int partial_depth);
void  branch_and_bound(int depth, int current_distance, int path[], bool visited[]);
void  generate_tasks(int current_depth, int task_depth, int partial_path[], bool visited[],
                       int tasks[][MAX_CITIES], int *task_count);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Record start time (for total program time).
    double start_time = MPI_Wtime();

    // All process read input file
    FILE *input_file = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            input_file = fopen(argv[i + 1], "r");
            break;
        }
    }
    read_input(input_file);
    fclose(input_file);

    // ----- Master/Worker logic -----
    if (rank == 0) {
        master_work_assignment();
    } else {
        worker_execution();
    }

    // ----- Gather max communication and computation times -----
    double max_comm_time, max_comp_time;
    MPI_Reduce(&local_comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // ----- Rank 0 prints final results -----
    if (rank == 0) {
        double end_time = MPI_Wtime();
        printf("------------------------------------------------------\n");
        printf("MPI WSP - Branch and Bound (Parallel, Dynamic Distribution)\n");
        printf("Number of Processes: %d\n", size);
        printf("Number of Cities: %d\n", num_cities);
        printf("Optimal WSP Distance: %d\n", best_distance);
        printf("Best Path: ");
        for (int i = 0; i < num_cities; i++) {
            printf("%d ", best_path[i]);
        }
        printf("\n");
        printf("Maximum communication time: %f s\n", max_comm_time);
        printf("Maximum computation time: %f s\n", max_comp_time);
        printf("Elapsed Total Time: %f seconds\n", end_time - start_time);
        printf("------------------------------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}

void read_input(FILE *file) {
    fscanf(file, "%d", &num_cities);
    if (rank == 0) {
        printf("Number of cities: %d\n", num_cities);
    }

    for (int i = 1; i < num_cities; i++) {
        for (int j = 0; j < i; j++) {
            fscanf(file, "%d", &distances[i * num_cities + j]);
            distances[j * num_cities + i] = distances[i * num_cities + j];
        }
    }

    if (rank == 0) {
        // Print the complete distance matrix.
        printf("Distance matrix:\n");
        for (int i = 0; i < num_cities; i++) {
            for (int j = 0; j < num_cities; j++) {
                printf("%d ", distances[i * num_cities + j]);
            }
            printf("\n");
        }
    }
}

// ------------------------
// Recursively generate all partial tours (tasks) of length 'task_depth'.
// The first city is assumed to be 0 (already placed in partial_path).
void generate_tasks(int current_depth, int task_depth, int partial_path[], bool visited[],
                    int tasks[][MAX_CITIES], int *task_count) {
    if (current_depth == task_depth) {
        // Store the current partial tour as one task.
        for (int i = 0; i < task_depth; i++) {
            tasks[*task_count][i] = partial_path[i];
        }
        (*task_count)++;
        return;
    }
    // Try all cities (skip 0 since it is fixed).
    for (int city = 1; city < num_cities; city++) {
        if (!visited[city]) {
            visited[city] = true;
            partial_path[current_depth] = city;
            generate_tasks(current_depth + 1, task_depth, partial_path, visited, tasks, task_count);
            visited[city] = false;
        }
    }
}

// ------------------------
// Master: generate tasks dynamically using 'generate_tasks' and assign them
// to workers. (Each task is sent as an integer array of length DEPTH+1,
// with the last element carrying the current best_distance.)
void master_work_assignment(void) {
    // Allocate the tasks array on the heap instead of the stack.
    int (*tasks)[MAX_CITIES] = malloc(MAX_TASKS * sizeof(*tasks));
    if (tasks == NULL) {
        fprintf(stderr, "Error allocating memory for tasks\n");
        exit(EXIT_FAILURE);
    }

    int task_count = 0;
    int partial_path[MAX_CITIES];
    bool visited[MAX_CITIES] = { false };
    // Fix city 0 as the starting city.
    partial_path[0] = 0;
    visited[0] = true;
    generate_tasks(1, DEPTH, partial_path, visited, tasks, &task_count);

    int current_task = 0; 
    int active_workers = size - 1;  // Master does not work.
    BestSolution received_solution;

    printf("[MASTER] Starting work assignment with %d tasks and %d active workers with task depth %d.\n",
        task_count, active_workers, DEPTH);

    while (active_workers > 0) {
        MPI_Status status;
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            if (status.MPI_TAG == TAG_TASK_REQUEST) {
                // A worker requests a task.
                int incoming_rank;
                double recv_start = MPI_Wtime();
                MPI_Recv(&incoming_rank, 1, MPI_INT, status.MPI_SOURCE,
                         TAG_TASK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double recv_end = MPI_Wtime();
                local_comm_time += (recv_end - recv_start);

                if (current_task < task_count) {
                    // Prepare task message: first DEPTH integers are the partial path,
                    // and the last integer is the current best_distance.
                    int data[DEPTH + 1];
                    for (int i = 0; i < DEPTH; i++) {
                        data[i] = tasks[current_task][i];
                    }
                    data[DEPTH] = best_distance;
                    double send_start = MPI_Wtime();
                    MPI_Request task_request;
                    MPI_Isend(data, DEPTH + 1, MPI_INT, incoming_rank, TAG_TASK_ASSIGN, MPI_COMM_WORLD, &task_request);
                    double send_end = MPI_Wtime();
                    local_comm_time += (send_end - send_start);
                    current_task++;
                } else {
                    // No more tasks: send termination signal (we set the first element to -1).
                    int stop_signal[DEPTH + 1];
                    stop_signal[0] = -1;
                    for (int i = 1; i < DEPTH + 1; i++) {
                        stop_signal[i] = -1;
                    }
                    double send_start = MPI_Wtime();
                    MPI_Request stop_request;
                    MPI_Isend(stop_signal, DEPTH + 1, MPI_INT, incoming_rank, TAG_TASK_ASSIGN, MPI_COMM_WORLD, &stop_request);
                    double send_end = MPI_Wtime();
                    local_comm_time += (send_end - send_start);
                    active_workers--;
                }
            }
            else if (status.MPI_TAG == TAG_NEW_BEST) {
                // A worker has found a new best solution.
                double recv_start = MPI_Wtime();
                MPI_Recv(&received_solution, sizeof(BestSolution), MPI_BYTE,
                         status.MPI_SOURCE, TAG_NEW_BEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                double recv_end = MPI_Wtime();
                local_comm_time += (recv_end - recv_start);

                if (received_solution.distance < best_distance) {
                    best_distance = received_solution.distance;
                    for (int i = 0; i < num_cities; i++) {
                        best_path[i] = received_solution.path[i];
                    }
                }

                double send_start = MPI_Wtime();
                MPI_Request request;
                MPI_Isend(&best_distance, 1, MPI_INT, status.MPI_SOURCE, TAG_UPDATED_BEST, MPI_COMM_WORLD, &request);
                double send_end = MPI_Wtime();
                local_comm_time += (send_end - send_start);
            }
        }
    }
    printf("[MASTER] Work assignment complete.\n");

    free(tasks);
}

// Worker: request tasks and then extend the partial tour using branch-and-bound.
// The task message is now an array of DEPTH+1 integers.
void worker_execution(void) {
    int data[DEPTH + 1];
    while (1) {
        double send_start = MPI_Wtime();
        MPI_Request task_request;
        MPI_Isend(&rank, 1, MPI_INT, 0, TAG_TASK_REQUEST, MPI_COMM_WORLD, &task_request);
        double send_end = MPI_Wtime();
        local_comm_time += (send_end - send_start);

        double recv_start = MPI_Wtime();
        MPI_Recv(data, DEPTH + 1, MPI_INT, 0, TAG_TASK_ASSIGN, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double recv_end = MPI_Wtime();
        local_comm_time += (recv_end - recv_start);

        // Termination signal: if the first element is -1, then break out.
        if (data[0] == -1) {
            break;
        }

        // Update local best_distance using the value appended to the task.
        if (data[DEPTH] < best_distance) {
            best_distance = data[DEPTH];
        }
        
        double comp_start = MPI_Wtime();
        // Continue branch-and-bound from the received partial tour.
        branch_and_bound_from_partial(data, DEPTH);
        double comp_end = MPI_Wtime();
        local_comp_time += (comp_end - comp_start);
    }
}

// Continue branch-and-bound from a given partial tour.
// 'partial_path' is an array of DEPTH cities; we compute the current cost and visited array.
void branch_and_bound_from_partial(int partial_path[], int partial_depth) {
    int current_distance = 0;
    bool visited[MAX_CITIES] = { false };
    int path[MAX_CITIES];
    for (int i = 0; i < partial_depth; i++) {
        path[i] = partial_path[i];
        if (i > 0) {
            current_distance += distances[path[i - 1] * num_cities + path[i]];
        }
        visited[path[i]] = true;
    }
    branch_and_bound(partial_depth, current_distance, path, visited);
}

void branch_and_bound(int depth, int current_distance, int path[], bool visited[]) {
    if (depth == num_cities) {
        if (current_distance < best_distance) {
            best_distance = current_distance;
            // Construct BestSolution
            BestSolution solution;
            solution.distance = best_distance;
            for (int i = 0; i < num_cities; i++) {
                solution.path[i] = path[i];
            }
            // Send to master best path found
            double send_start = MPI_Wtime();
            MPI_Send(&solution, sizeof(BestSolution), MPI_BYTE,
                     0, TAG_NEW_BEST, MPI_COMM_WORLD);
            double send_end = MPI_Wtime();
            local_comm_time += (send_end - send_start);
            // Minus the send time so computation time is not affected
            local_comp_time -= (send_end - send_start);
            // Receive updated best
            int updated_best_distance;
            double recv_start = MPI_Wtime();
            MPI_Recv(&updated_best_distance, 1, MPI_INT,
                     0, TAG_UPDATED_BEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double recv_end = MPI_Wtime();
            local_comm_time += (recv_end - recv_start);
            // Minus the receive time so computation time is not affected
            local_comp_time -= (recv_end - recv_start);
            // Update actual best
            if (updated_best_distance < best_distance) {
                best_distance = updated_best_distance;
            }
        }
        return;
    }

    int last_city = path[depth - 1];

    for (int city = 0; city < num_cities; city++) {
        if (!visited[city]) {
            int new_distance = current_distance + distances[last_city * num_cities + city];
            if (new_distance >= best_distance) {
                continue;
            }
            visited[city] = true;
            path[depth] = city;
            branch_and_bound(depth + 1, new_distance, path, visited);
            visited[city] = false;
        }
    }
}
