#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>

/******************************************************************************************************* */
/* Helper Functions*/
/* START */

// Must use cudaDeviceSynchronize() when measuring GPU kernel operations because CUDA kernel operations are non blocking.
double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec / 1.0e6);
}

bool verify(float *gpu_result, float *cpu_result, unsigned int nRows, unsigned int nCols)
{
    const float epsilon = 1e-4; // Set a tolerance value for comparison

    for (int i = 0; i < nRows * nCols; i++)
    {

        if (std::fabs(cpu_result[i] - gpu_result[i]) > epsilon)
        {
            return false;
        }
    }
    return true;
}

/* END */
/* Helper Functions*/
/******************************************************************************************************* */

/******************************************************************************************************* */
/* Matrix Multiplication Functions*/
/* START */
void basicSgemm_h(float *a_h, float *b_h, float *c_h, unsigned int m, unsigned int k, unsigned int n)
{

    for (int outputMatrixIndex = 0; outputMatrixIndex < m * n; outputMatrixIndex++)
    {
        int row = outputMatrixIndex / n;
        int col = outputMatrixIndex % n;

        float sum = 0.0;

        for (int i = 0; i < k; i++)
            sum += a_h[row * k + i] * b_h[i * n + col];

        c_h[outputMatrixIndex] = sum;
    }

    printf("\nCPU Result\n");
    for (int i = 0; i < m * n; i++)
    {
        printf("%f ", c_h[i]);
    }
    printf("\n");
}

__global__ void matrixMulKernel_1thread1element(float *a_d, float *b_d, float *c_d, unsigned int m, unsigned int k, unsigned int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (col < n && row < m)
    {
        for (int i = 0; i < k; i++)
        {
            sum += a_d[row * k + i] * b_d[i * n + col];
        }
        c_d[row * n + col] = sum;
    }
}

__global__ void matrixMulKernel_1thread1row(float *a_d, float *b_d, float *c_d, unsigned int m, unsigned int k, unsigned int n)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global row index
    if (row < m)
    { // Check if the row index is within bounds
        for (int col = 0; col < n; col++)
        {
            float sum = 0.0;
            for (int i = 0; i < k; i++)
            {
                sum += a_d[row * k + i] * b_d[i * n + col];
            }
            c_d[row * n + col] = sum;
        }
    }
}

__global__ void matrixMulKernel_1thread1column(float *a_d, float *b_d, float *c_d, unsigned int m, unsigned int k, unsigned int n)
{

    int col = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global column index
    if (col < n)
    { // Check if the column index is within bounds
        for (int row = 0; row < m; row++)
        {
            float sum = 0.0;
            for (int i = 0; i < k; i++)
            {
                sum += a_d[col * k + i] * b_d[i * n + row];
            }
            c_d[row * n + col] = sum;
        }
    }
}
/* Matrix Multiplication Functions*/
/* END */
/******************************************************************************************************* */

void basicSgemm_d_1thread1element(float *a_h, float *b_h, float *c_h, unsigned int m, unsigned int k, unsigned int n)
{

    // (1) allocate device memory for arrays x_d, y_d, z_d
    float *a_d, *b_d, *c_d;
    cudaMalloc((void **)&a_d, sizeof(float) * m * k);
    cudaMalloc((void **)&b_d, sizeof(float) * k * n);
    cudaMalloc((void **)&c_d, sizeof(float) * m * n);

    // (2) copy matrices a_h and b_h to device memory a_d and b_d, respectively
    cudaMemcpy(a_d, a_h, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    // (3) call kernel to launch a grid of threads to perform the matrix multiplcation on GPU
    dim3 gridDim((n + 16 - 1) / 16, (m + 16 - 1) / 16);
    dim3 blockDim(16, 16);

    double start_time = myCPUTimer();
    matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(a_d, b_d, c_d, m, k, n);
    cudaDeviceSynchronize();
    double end_time = myCPUTimer();
    double elapsed_time = end_time - start_time;

    printf("\nElapsed time of 1 thread 1 output element: %f ms\n", 1000 * elapsed_time);

    // (4) Copy the result data from device memory of array c_d to host memory of array c_h
    cudaMemcpy(c_h, c_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    printf("\nGPU Result Single Element\n");
    for (int i = 0; i < m * n; i++)
    {
        printf("%f ", c_h[i]);
    }
    printf("\n");

    // (5) free device memory of a_d, b_d, and c_d
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void basicSgemm_d_1thread1row(float *a_h, float *b_h, float *c_h, unsigned int m, unsigned int k, unsigned int n)
{

    // (1) allocate device memory for arrays x_d, y_d, z_d
    float *a_d, *b_d, *c_d;
    cudaMalloc((void **)&a_d, sizeof(float) * m * k);
    cudaMalloc((void **)&b_d, sizeof(float) * k * n);
    cudaMalloc((void **)&c_d, sizeof(float) * m * n);

    // (2) copy matrices a_h and b_h to device memory a_d and b_d, respectively
    cudaMemcpy(a_d, a_h, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    // (3) call kernel to launch a grid of threads to perform the matrix multiplcation on GPU && CPU
    dim3 grid;
    dim3 block;
    if (m <= 1024)
    {
        grid.x = 1;
        block.x = m;
    }
    else
    {
        grid.x = (m + 1023) / 1024;
        block.x = 1024;
    }

    double start_time = myCPUTimer();
    matrixMulKernel_1thread1row<<<grid, block>>>(a_d, b_d, c_d, m, k, n);
    cudaDeviceSynchronize();
    double end_time = myCPUTimer();
    double elapsed_time = end_time - start_time;

    printf("\nElapsed time of 1 thread 1 output row: %f ms\n", 1000 * elapsed_time);

    // (4) Copy the result data from device memory of array c_d to host memory of array c_h
    cudaMemcpy(c_h, c_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    printf("\nGPU Result Single Row\n");
    for (int i = 0; i < m * n; i++)
    {
        printf("%f ", c_h[i]);
    }
    printf("\n");

    // (5) free device memory of a_d, b_d, and c_d
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

void basicSgemm_d_1thread1column(float *a_h, float *b_h, float *c_h, unsigned int m, unsigned int k, unsigned int n)
{

    // (1) allocate device memory for arrays x_d, y_d, z_d
    float *a_d, *b_d, *c_d;
    cudaMalloc((void **)&a_d, sizeof(float) * m * k);
    cudaMalloc((void **)&b_d, sizeof(float) * k * n);
    cudaMalloc((void **)&c_d, sizeof(float) * m * n);

    // (2) copy matrices a_h and b_h to device memory a_d and b_d, respectively
    cudaMemcpy(a_d, a_h, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float) * k * n, cudaMemcpyHostToDevice);

    // (3) call kernel to launch a grid of threads to perform the matrix multiplcation on GPU && CPU
    // dim3 gridDim(ceil(m/16.0), ceil(n/16.0), 1);
    // dim3 blockDim(16, 16, 1);

    double start_time = myCPUTimer();
    // matrixMulKernel_1thread1element<<<gridDim, blockDim>>>(a_d, b_d, c_d, m, k, n);
    cudaDeviceSynchronize();
    double end_time = myCPUTimer();
    double elapsed_time = end_time - start_time;

    printf("Elapsed time of 1 thread 1 output element: %f ms\n", 1000 * elapsed_time);

    // (4) Copy the result data from device memory of array c_d to host memory of array c_h
    cudaMemcpy(c_h, c_d, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

    // (5) free device memory of a_d, b_d, and c_d
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
}

int main(int argc, char *argv[])
{

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    srand(time(0));

    // matrix 𝐴 is of size 𝑚 × 𝑘, matrix 𝐵 is of size 𝑘 × 𝑛, and matrix 𝐶 is of size 𝑚 × 𝑛.
    float *a_h = (float *)malloc(sizeof(float) * m * k);
    for (unsigned int i = 0; i < m * k; i++)
        a_h[i] = rand() % 100 / 100.0;

    float *b_h = (float *)malloc(sizeof(float) * k * n);
    for (unsigned int i = 0; i < k * n; i++)
        b_h[i] = rand() % 100 / 100.0;

    float *c_h = (float *)calloc(m * n, sizeof(float));

    float *cpu_result = (float *)calloc(m * n, sizeof(float));

    basicSgemm_h(a_h, b_h, cpu_result, m, k, n);

    basicSgemm_d_1thread1element(a_h, b_h, c_h, m, k, n);
    if (verify(c_h, cpu_result, m, n))
        printf("Verified!\n");

    basicSgemm_d_1thread1row(a_h, b_h, c_h, m, k, n);
    if (verify(c_h, cpu_result, m, n))
        printf("Verified!\n");

    basicSgemm_d_1thread1column(a_h, b_h, c_h, m, k, n);
    if (verify(c_h, cpu_result, m, n))
        printf("Verified!\n");

    free(a_h);
    free(b_h);
    free(c_h);
    free(cpu_result);

    return 0;
}