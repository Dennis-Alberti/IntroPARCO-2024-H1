#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <stdbool.h>

#define NUM_RUNS 5
#define NUM_RUNS_S_E 10


// Random to apply random floating numbers 
float randomFloat_Matrix(float min, float max){
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

// Check Symmetry no Optimization
void checkSymPrint(float **matrix, int N) {

    int isSym = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {  // Only check the lower triangle
            if (matrix[i][j] != matrix[j][i]) {
                isSym = 0;
            }
        }
    }
    if (isSym)
    {
        printf("Matrix is symmetric.\n");       
    }else{
        printf("Matrix is not symmetric.\n");
    }
}

// Check Symmetry no Optimization
int checkSym(float **matrix, int N) {
int isSym = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {  // Only check the lower triangle
            if (matrix[i][j] != matrix[j][i]) {
                isSym = 0;
            }
        }
    }
    return isSym;
}

// Transpose matrix
void matTranspose(float **matrix, float **transpose, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            transpose[j][i] = matrix[i][j];
        }
    }
}

// Check Symmetry using Implicit Parallelization
int checkSymImp(float **matrix, int N) {
    int isSym = 1;
    #pragma simd
    #pragma unroll(4)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {  // Only check the lower triangle
            if (matrix[i][j] != matrix[j][i]) {
                isSym = 0;
            }
        }
    }
    return isSym;
}

// Matrix Transpose using Implicit Parallelism
void matTransposeImp(float **matrix, float **transpose, int N) {
    #pragma simd
    #pragma unroll(4)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            transpose[j][i] = matrix[i][j];
        }
    }
}


// Check Symmetry using Implicit Parallelization
int checkSymOpenMP(float **matrix, int N, int threads) {
    int isSym = 1;
    // Parallelize the loop with OpenMP
    #pragma omp parallel for /* collapse(2) */ /*schedule(dynamic)*/ shared(matrix, N, threads) num_threads(threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {  // Only check the lower triangle
            if (matrix[i][j] != matrix[j][i]) {
                isSym = 0;
            }
        }
    }
    return isSym;
}


// Matrix Transposition using OpenMP
void matTransposeOpenMP(float **matrix, float **transpose, int N, int threads) {
    #pragma omp parallel for collapse(2) /*schedule(dynamic)*/ shared(matrix, transpose, N, threads) num_threads(threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            transpose[j][i] = matrix[i][j];
        }
    }
}


int main(int argc, char *argv[]) {

    float min = 1.0;
    float max = 100.0;

    // Check if the command-line argument is provided
    if (argc != 2) {
        printf("Usage: %s <matrix_dimension>\n", argv[0]);
        return 1; // Exit if the argument is missing
    }
    
    int N = atoi(argv[1]);
    if (N <= 0) {
        printf("Error: Matrix dimension must be a positive integer.\n");
        return 1; // Exit if the input is invalid
    }

    srand(time(NULL));

    // Variables for storing time
    struct timeval start_tv, end_tv;
    time_t start_time, end_time;


	// Allocate memory for the matrix dynamically
    float **matrix = (float **)malloc(N * sizeof(float *));  // Allocate mamory for the matrix rows
    float **transpose = (float **)malloc(N * sizeof(float *)); // Allocate mamory for the transpose rows
    for (int i = 0; i < N; i++) {
        matrix[i] = (float *)malloc(N * sizeof(float));    // Allocate memory for N columns in each row
        transpose[i] = (float *)malloc(N * sizeof(float));   // Allocate memory for N columns in each row
    }
	// Initialize the matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = randomFloat_Matrix(min,max);
        }
    }

    checkSymPrint(matrix,N);


    // Open CSV file for writing results
    FILE *file1 = fopen("SERIAL.csv", "w");
    if (file1 == NULL) {
        perror("Failed to open file");
        return 1;
    }

        // Open CSV file for writing results
    FILE *file2 = fopen("IMPLICIT.csv", "w");
    if (file2 == NULL) {
        perror("Failed to open file");
        return 1;
    }

        // Open CSV file for writing results
    FILE *file3 = fopen("OPEN_MP.csv", "w");
    if (file3 == NULL) {
        perror("Failed to open file");
        return 1;
    }

    // Open CSV file for writing results
    FILE *file4 = fopen("SPEED_UP EFFICENCY.csv", "w");
    if (file4 == NULL) {
        perror("Failed to open file");
        return 1;
    }



    // Write header for CSV SERIAL
    fprintf(file1, "\n SERIAL %d \n",N);
    fprintf(file1, "\n checkSym, matTranspose\n");

    // =======================
    // 1.1 Symmetry
    // =======================
    gettimeofday(&start_tv, NULL); // Start time
        int isSym = checkSym(matrix, N);
    gettimeofday(&end_tv, NULL); // End time
    // Calculate elapsed time in seconds
    long seconds_tv1 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv1 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday1 = seconds_tv1 + microseconds_tv1*1e-6;
    printf("Time taken for the checkSym: %.6f seconds and the matrix symmetry is: %d\n", elapsed_gettimeofday1, isSym);


    // =======================
    // 1.4 matTranspose
    // =======================
    gettimeofday(&start_tv, NULL); // Start time
        matTranspose(matrix, transpose, N);
    gettimeofday(&end_tv, NULL); // End time
    long seconds_tv4 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv4 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday4 = seconds_tv4 + microseconds_tv4*1e-6;
    printf("Time taken for the matTranspose: %.6f seconds\n", elapsed_gettimeofday4);

    fprintf(file1, " %.6f, %.6f\n", elapsed_gettimeofday1, elapsed_gettimeofday4); 






/*     // Write header for CSV IMPLICIT
    fprintf(file2, "\n IMPLICIT %d \n",N);
    fprintf(file2, "\n checkSymImp, matTransposeImp\n");
   
    // =======================
    // 1.2 checkSymBlockImp
    // =======================
    gettimeofday(&start_tv, NULL); // Start time
        int isSym = checkSymImp(matrix, N);
    gettimeofday(&end_tv, NULL); // End time
    // Calculate elapsed time in seconds
    long seconds_tv2 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv2 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday2 = seconds_tv2 + microseconds_tv2*1e-6;
    printf("Time taken for the checkSymImp: %.6f seconds and the matrix symmetry is: %d\n", elapsed_gettimeofday2, isSym);


    // =======================
    // 1.5 matTransposeImp
    // =======================

    gettimeofday(&start_tv, NULL); // Start time
        matTransposeImp(matrix, transpose, N);
    gettimeofday(&end_tv, NULL); // End time
    long seconds_tv5 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv5 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday5 = seconds_tv5 + microseconds_tv5*1e-6;
    printf("Time taken for the matTransposeImp: %.6f seconds\n", elapsed_gettimeofday5);

fprintf(file2, "%.6f, %.6f\n", elapsed_gettimeofday2, elapsed_gettimeofday5); */




/*     // Write header for CSV OPENMP
    fprintf(file3, "\n OPENMP %d \n",N);
    fprintf(file3, "Sym_OpenMP, threads \n");

    // =======================
    // 1.3 checkSymOpenMP
    // =======================

// Test performance with thread counts of 1, 2, 4, 8, 16, 32, and 64
for (int threads = 1; threads <= 64; threads *= 2) {
    omp_set_num_threads(threads);

double total_time_parallel = 0.0;
for (int run = 0; run < NUM_RUNS; run++) {
    gettimeofday(&start_tv, NULL); // Start time
        checkSymOpenMP(matrix, N, threads);
    gettimeofday(&end_tv, NULL); // End time
    // Calculate elapsed time in seconds
    long seconds_tv3 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv3 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday3 = seconds_tv3 + microseconds_tv3*1e-6;
    total_time_parallel += elapsed_gettimeofday3;

}
    double avg_time_parallel = total_time_parallel / NUM_RUNS;
    printf("Time taken for the checkSymOpenMP: %.6f seconds, threads: %d\n", avg_time_parallel, threads);
    fprintf(file3, "%.6f,%d \n", avg_time_parallel,threads);
    // =======================
    // 1.2 Calculate bandwidth for checkSymOpenMP
    // =======================
    // Write header for CSV

    // Calculate memory accessed
    size_t total_memory_accessed1 = 2 * N * N * sizeof(float); // Read + Write
    // Calculate bandwidth in GB/s
    double bandwidth1 = total_memory_accessed1 / (avg_time_parallel * 1e9);

    printf("bandwidth: %.4f GB/s, %d \n", bandwidth1,threads);
    fprintf(file3, "%.4f GB/s, %d \n\n", bandwidth1,threads);
}




    // =======================
    // 1.6 matTransposeOpenMP
    // =======================
    fprintf(file3, "Tran_OpenMP, threads\n");

    for (int threads = 1; threads <= 64; threads *= 2) {
    omp_set_num_threads(threads);

    double total_time_parallel = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
    gettimeofday(&start_tv, NULL); // Start time
        matTransposeOpenMP(matrix, transpose, N, threads);
    gettimeofday(&end_tv, NULL); // End time
    // Calculate elapsed time in seconds
    long seconds_tv6 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv6 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday6 = seconds_tv6 + microseconds_tv6*1e-6;
    total_time_parallel += elapsed_gettimeofday6;
    }
    double avg_time_parallel_transposta = total_time_parallel / NUM_RUNS;
    printf("Time taken for the matTransposeOpenMP: %.6f seconds, threads: %d\n", avg_time_parallel_transposta,threads);
   // Write results to CSV TranOpenMP_Time
    fprintf(file3, "%.6f,%d \n", avg_time_parallel_transposta,threads);

    // =======================
    // 1.2 Calculate bandwidth for matTransposeOpenMP
    // =======================
    // Calculate memory accessed
    size_t total_memory_accessed2 = 2 * N * N * sizeof(float); // Read + Write
    // Calculate bandwidth in GB/s
    double bandwidth2 = total_memory_accessed2 / (avg_time_parallel_transposta * 1e9);

    printf("bandwidth: %.4f GB/s, %d \n", bandwidth2,threads);
    fprintf(file3, "%.4f GB/s, %d \n\n", bandwidth2,threads);
    }  */ 






/*     // =======================
    // 1.6 avg_speedup & avg_efficiency for checkSymOpenMP
    // =======================


    // Write header for CSV
    fprintf(file4, "N: %d\n", N);
    fprintf(file4, "\n Threads, Sym_speedup, Sym_efficiency \n");


    for (int threads = 1; threads <= 64; threads *= 2) {
    omp_set_num_threads(threads);

    gettimeofday(&start_tv, NULL); // Start time
        checkSymOpenMP(matrix, N, threads);
    gettimeofday(&end_tv, NULL); // End time
    // Calculate elapsed time in seconds
    long seconds_tv3 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv3 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday3 = seconds_tv3 + microseconds_tv3*1e-6;

        double total_time_parallel = 0.0;
        for (int run = 0; run < NUM_RUNS_S_E; run++) {
            double start = omp_get_wtime();

            checkSymOpenMP(matrix, N, threads);

            double end = omp_get_wtime();
            double time_parallel = end - start;
            total_time_parallel += time_parallel;
        }
            // Compute average parallel time, speedup, and efficiency
        double avg_time_parallel = total_time_parallel / NUM_RUNS_S_E;
        double avg_speedup = elapsed_gettimeofday3 / avg_time_parallel;
        double avg_efficiency = avg_speedup / threads;

    printf("\t avg_speedup for CHECK SYMMETRY %7.2f seconds, thread n: %d\n", avg_speedup, threads);
    printf("\t avg_efficiency for CHECK SYMMETRY %9.2f%%, thread n: %d\n\n", avg_efficiency*100, threads);
    
    // Write results to CSV
    fprintf(file4, "%d,%f,%f\n", threads, avg_speedup, avg_efficiency * 100);
    }

    // =======================
    // 1.6 avg_speedup & avg_efficiency for matTransposeOpenMP
    // =======================

    // Write header for CSV
    fprintf(file4, " Threads, Tran_speedup, Tran_efficiency \n");

    for (int threads = 1; threads <= 64; threads *= 2) {
    omp_set_num_threads(threads);

    gettimeofday(&start_tv, NULL); // Start time
        matTransposeOpenMP(matrix, transpose, N, threads);
    gettimeofday(&end_tv, NULL); // End time
    // Calculate elapsed time in seconds
    long seconds_tv6 = end_tv.tv_sec - start_tv.tv_sec;
    long microseconds_tv6 = end_tv.tv_usec - start_tv.tv_usec;
    double elapsed_gettimeofday6 = seconds_tv6 + microseconds_tv6*1e-6;

        double total_time_parallel = 0.0;
        for (int run = 0; run < NUM_RUNS_S_E; run++) {
            double start = omp_get_wtime();

           matTransposeOpenMP(matrix, transpose, N, threads);

            double end = omp_get_wtime();
            double time_parallel = end - start;
            total_time_parallel += time_parallel;
        }
            // Compute average parallel time, speedup, and efficiency
        double avg_time_parallel = total_time_parallel / NUM_RUNS_S_E;
        double avg_speedup = elapsed_gettimeofday6 / avg_time_parallel;
        double avg_efficiency = avg_speedup / threads;

    printf("\t avg_speedup for TRANSPOSE %7.2f seconds, thread n: %d\n", avg_speedup, threads);
    printf("\t avg_efficiency for TRANSPOSE %9.2f%%, thread n: %d\n\n", avg_efficiency*100, threads);
    
    // Write results to CSV
    fprintf(file4, "%d,%f,%f\n", threads, avg_speedup, avg_efficiency * 100);
    } */ 



/* // Initialize the matrix
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf( "%f ",matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n\n");

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf( "%f ",transpose[i][j]);
        }
        printf("\n");
    } */

   fclose(file1);
   fclose(file2);
   fclose(file3);
   fclose(file4);

	// Free allocated memory
    for (int i = 0; i < N; i++) {
        free(matrix[i]);  // Free each row
        free(transpose[i]);
    }
    free(matrix);  // Free the row pointers
    free(transpose);

    return 0;
}
