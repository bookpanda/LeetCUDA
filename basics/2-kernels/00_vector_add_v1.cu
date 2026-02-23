#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

#define N 10000000  // Vector size = 10 million
#define BLOCK_SIZE 256

// Example:
// A = [1, 2, 3, 4, 5]
// B = [6, 7, 8, 9, 10]
// C = A + B = [7, 9, 11, 13, 15]

// CPU vector addition
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel for vector addition
__global__ void vector_add_gpu(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Initialize vector with random values
void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);

    // Initialize vectors
    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation (use CUDA events for accurate timing)
    printf("Benchmarking GPU implementation...\n");
    cudaEvent_t start_ev, stop_ev;
    CUDA_CHECK(cudaEventCreate(&start_ev));
    CUDA_CHECK(cudaEventCreate(&stop_ev));
    float gpu_total_time = 0.0f;
    for (int i = 0; i < 20; i++) {
        CUDA_CHECK(cudaEventRecord(start_ev, 0));
        vector_add_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CUDA_CHECK(cudaEventRecord(stop_ev, 0));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start_ev, stop_ev));
        gpu_total_time += elapsed;
    }
    double gpu_avg_time = (double)gpu_total_time / 20.0;
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    // Print results (CPU: seconds->ms, GPU: cudaEventElapsedTime already in ms)
    printf("CPU average time: %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time);
    printf("Speedup: %fx\n", (cpu_avg_time * 1000) / gpu_avg_time);

    // Verify results (optional)
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost));
    bool correct = true;
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double err = fabs((double)h_c_cpu[i] - (double)h_c_gpu[i]);
        if (err > 1e-5) {
            correct = false;
            max_error = fmax(max_error, err);
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");
    printf("Max error: %f\n", max_error);
    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
