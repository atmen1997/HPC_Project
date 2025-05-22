#include <stdio.h>      // For printf, scanf, etc.
#include <limits.h>     // For INT_MAX
#include <stdbool.h>    // For bool, true, false
#include <mpi.h>        // For MPI functions
#include <time.h>       // For MPI_Wtime (timing)
#include <string.h>     // For strcmp

#define MAX_CITIES 19   // Maximum number of cities from the input
#define DEPTH 5         // Initial depth for path generation

// Global variables to store problem data and best result
int num_cities = 0;                         // Total number of cities (read from input)
int distances[MAX_CITIES * MAX_CITIES] = {0};       // 1D array for the distance matrix (row-major order)
int best_distance = INT_MAX;                  // Global best distance found so far
int best_path[MAX_CITIES];                    // Global best path found so far
int rank, size;                             // MPI process rank and total number of processes

// Prototypes
void read_input(FILE *file);
void generate_paths_and_distribute(int initial_depth);
void branch_and_bound(int depth, int current_distance, int path[], bool visited[]);

int main(int argc, char *argv[]) {
    // Initialize MPI environment
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
    // Read the input (number of cities and distances) from the file.
    read_input(input_file);
    fclose(input_file);

    // Start the overall computation timer.
    double comp_start_time = MPI_Wtime();
    
    // Distribute work among processes using a round-robin assignment.
    generate_paths_and_distribute(DEPTH);

    double comp_end_time = MPI_Wtime();
    double comp_time = comp_end_time - comp_start_time;
    // Use MPI_Allreduce to combine the best distances found by all processes.
    int global_best_distance;
    double comm_start_time = MPI_Wtime();
    MPI_Allreduce(&best_distance, &global_best_distance, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double comm_end_time = MPI_Wtime();
    double comm_time = comm_end_time - comm_start_time;
    // Only the process with the best result prints the final output.
    double max_comm_time, max_comp_time;
    MPI_Allreduce(&comm_time, &max_comm_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&comp_time, &max_comp_time, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (best_distance == global_best_distance) {
        double end_time = MPI_Wtime();
        // Gather the maximum comm time and maximum comp time across ranks
        printf("------------------------------------------------------\n");
        printf("MPI WSP - Branch and Bound (Parallel, Round Robin)\n");
        printf("Number of processes: %d\n", size);
        printf("Number of cities: %d\n", num_cities);
        printf("Global best distance: %d\n", global_best_distance);
        printf("Best path: ");
        for (int i = 0; i < num_cities; i++) {
            printf("%d ", best_path[i]);
        }
        printf("\n");
        printf("Maximum communication time = %f sec\n", max_comm_time);
        printf("Maximum computation time = %f sec\n", max_comp_time);
        printf("Elapsed time: %f seconds\n", end_time - start_time);
        printf("------------------------------------------------------\n");
    }
    // Finalize the MPI environment.
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

void generate_paths_and_distribute(int initial_depth) {
    int path[MAX_CITIES];
    bool visited[MAX_CITIES] = {false};

    // Fix start city as 0
    path[0] = 0;
    visited[0] = true;

    int task_id = 0;  // Task counter
    int path_count = 0;  // Path generation counter
    


    void build_branches(int depth) {  // Renamed from recursive_generate()
        if (depth == initial_depth) {
            // Assign task to an MPI process (round-robin)
            if (task_id % size == rank) {
                int initial_cost = 0;
                for (int d = 0; d < depth - 1; d++) {
                    initial_cost += distances[path[d] * num_cities + path[d + 1]];
                }
                branch_and_bound(initial_depth, initial_cost, path, visited);
            }
            task_id++;
            if (rank == 0) {
                path_count++;
            }
            return;
        }

        for (int city = 1; city < num_cities; city++) {
            if (!visited[city]) {
                path[depth] = city;
                visited[city] = true;
                build_branches(depth + 1);  
                visited[city] = false; // Backtrack
            }
        }
    }

    // Start path generation at depth 1
    build_branches(1);
    if (rank == 0) {
        printf("Total task count: %d, depth: %d , average tasks per processors: %d\n", path_count, initial_depth, path_count/size);
    }
}


void branch_and_bound(int depth, int current_distance, int path[], bool visited[]) {
    // Base case: if all cities have been visited, update the best path if the current distance is lower.
    if (depth == num_cities) {
        if (current_distance < best_distance) {
            best_distance = current_distance;
            // Copy the current path to the global best_path.
            for (int i = 0; i < num_cities; i++) {
                best_path[i] = path[i];
            }
        }
        return; // Return to explore other paths.
    }

    // Get the last city visited in the current path.
    int last_city = path[depth - 1];

    // Try to visit each city that hasn't been visited yet.
    for (int city = 0; city < num_cities; city++) {
        if (!visited[city]) {
            // Calculate the new total distance if we visit this city next.
            int new_distance = current_distance + distances[last_city * num_cities + city];
            
            // Prune the branch if the new distance is already worse than the best known distance.
            if (new_distance >= best_distance) {
                continue;
            }
            // Mark the city as visited and add it to the current path.
            visited[city] = true;
            path[depth] = city;
            
            // Recursively continue to the next depth.
            branch_and_bound(depth + 1, new_distance, path, visited);
            
            // Backtrack: unmark the city so other paths can be explored.
            visited[city] = false;
        }
    }
}
