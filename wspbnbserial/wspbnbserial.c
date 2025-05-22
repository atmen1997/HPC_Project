#include <stdio.h>    // For printf, scanf
#include <limits.h>   // For INT_MAX
#include <stdbool.h>  // For bool, true, false
#include <string.h>
#include <time.h>     // For clock(), CLOCKS_PER_SEC

#define MAX_CITIES 19 // Maximum number of cities from the input

static int num_cities = 0;               // Total number of cities (set in read_input)
static int distances[MAX_CITIES * MAX_CITIES]; // Distance matrix between cities
static int best_distance = INT_MAX;      // Best (minimum) distance found so far
static int best_path[MAX_CITIES];        // Best path found so far

// Function prototypes
void read_input(FILE *file);
void branch_and_bound(int depth, int current_distance, int path[], bool visited[]);

int main(int argc, char *argv[]) {
    // Read problem input (number of cities and distance matrix)
    FILE *input_file = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            input_file = fopen(argv[i + 1], "r");
            break;
        }
    }

    read_input(input_file);
    fclose(input_file);
    int start_city = 0;

    // Create local arrays to track the current path and visited cities.
    // These are now local to main and passed to the recursive function.
    bool visited[MAX_CITIES] = { false };
    int path[MAX_CITIES] = { 0 };

    // Start at the initial city.
    visited[start_city] = true;
    path[0] = start_city;

    // Record start time.
    clock_t start_time = clock();

    // Start recursive branch and bound from depth 1 with an initial cost of 0.
    branch_and_bound(1, 0, path, visited);

    // Record end time.
    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the results.
    printf("------------------------------------------------------\n");
    printf("WSP - Branch and Bound Results (Serial)\n");
    printf("Start city: %d\n", start_city);
    printf("Best distance: %d\n", best_distance);
    printf("Best path: ");
    for (int i = 0; i < num_cities; i++) {
        printf("%d ", best_path[i]);
    }
    printf("\n");
    printf("Time taken: %f seconds\n", time_taken);
    printf("------------------------------------------------------\n");

    return 0;
}

void read_input(FILE *file) {
    fscanf(file, "%d", &num_cities);
    printf("Number of cities: %d\n", num_cities);

    // Initialize distance matrix to 0
    for (int i = 0; i < num_cities; i++) {
        for (int j = 0; j < num_cities; j++) {
            distances[i * num_cities + j] = 0;
        }
    }

    // Read distances from the input file
    for (int i = 1; i < num_cities; i++) {
        for (int j = 0; j < i; j++) {
            fscanf(file, "%d", &distances[i * num_cities + j]);
            distances[j * num_cities + i] = distances[i * num_cities + j];
        }
    }

    printf("Distance matrix:\n");
    for (int i = 0; i < num_cities; i++) {
        for (int j = 0; j < num_cities; j++) {
            printf("%d ", distances[i * num_cities + j]);
        }
        printf("\n");
    }
}

// The branch_and_bound function now receives local arrays for the current path and visited flags.
void branch_and_bound(int depth, int current_distance, int path[], bool visited[]) {
    // If all cities have been visited and the current path is better than the best known,
    // update the best_distance and best_path.
    if ((depth == num_cities) && (current_distance < best_distance)) {
        best_distance = current_distance;
        for (int i = 0; i < num_cities; i++) {
            best_path[i] = path[i];
        }
        return;
    }

    int last_city = path[depth - 1];

    // Try visiting each city that hasn't been visited yet.
    for (int city = 0; city < num_cities; city++) {
        if (!visited[city]) {
            int new_distance = current_distance + distances[last_city * num_cities + city];

            // Prune the branch if the new distance is not promising.
            if (new_distance >= best_distance)
                continue;

            // Mark this city as visited and add it to the current path.
            visited[city] = true;
            path[depth] = city;

            // Recursively continue to the next depth.
            branch_and_bound(depth + 1, new_distance, path, visited);

            // Backtrack: unmark the city so that other paths can be explored.
            visited[city] = false;
        }
    }
}
