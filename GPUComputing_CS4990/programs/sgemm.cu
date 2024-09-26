#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

/**
  *  Must use cudaDeviceSynchronize() when measuring GPU kernel operations because they are non blocking. 
 */
double myCPUTimer(){ 
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec/1.0e6);
}

__global__ void mm_kernel(float* a_d, float* b_d, float* c_d, unsigned int m, unsigned int n, unsigned int k){

    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;

    float sum = 0.0;

    if(row < m && col < n){
        for(int i = 0; i < k; i++) 
        {
            sum += a_d[row * k + i] * b_d[i * k + col];
        }
        c_d[row * k + col] = sum;
    }
}

int main(int argc, char* argv[]){

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    srand(time(0));

    /**
        allocate host memory for a_h, b_h, and c_h and intialize x_h, y_h
        matrix 𝐴 is of size 𝑚 × 𝑘, matrix 𝐵 is of size 𝑘 × 𝑛, and matrix 𝐶 is of size 𝑚 × 𝑛.
     */      
    float* a_h = (float*) malloc(sizeof(float)*m*k);
    for(unsigned int i = 0; i < m*k; i++) a_h[i] = rand()%100/100.0;

    float* b_h = (float*) malloc(sizeof(float)*k*n);
    for(unsigned int i = 0; i < k*n; i++) b_h[i] = rand()%100/100.0;

    float* c_h = (float*) calloc(m*n, sizeof(float));

    // (1) allocate device memory for arrays x_d, y_d, z_d
    float *a_d, *b_d, *c_d;
    cudaMalloc((void**) &a_d, sizeof(float)*m*k);
    cudaMalloc((void**) &b_d, sizeof(float)*k*n);
    cudaMalloc((void**) &c_d, sizeof(float)*m*n);

    // (2) copy matrices a_h and b_h to device memory a_d and b_d, respectively
    cudaMemcpy(a_d, a_h, sizeof(float)*m*k, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float)*k*n, cudaMemcpyHostToDevice);

    // (3) call kernel to launch a grid of threads to perform the matrix multiplcation on GPU && CPU
    dim3 gridDim(ceil(m/16.0), ceil(n/16.0), 1);
    dim3 blockDim(16, 16, 1);

    printf("The # of Grids: %d\n", gridDim);
    printf("The # of blocks: %d\n", blockDim);

    double gpu_elapsed_time_ms;
    double startTime = myCPUTimer();
    mm_kernel<<<gridDim, blockDim>>>(a_d, b_d, c_d, m, n, k);
    cudaDeviceSynchronize();
    double endTime = myCPUTimer();

    gpu_elapsed_time_ms = endTime - startTime;
    printf("Execution time of GPU matrix multiplication: %f\n", gpu_elapsed_time_ms);
    printf("Matrix Dimensions:\n");
    printf("\t A[%d][%d]\n", m, k);
    printf("\t B[%d][%d]\n", k, n);

    // (4) Copy the result data from device memory of array  z_d to host memory of array z_h
    cudaMemcpy(c_h, c_d, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    // (5) free device memory of a_d, b_d, and c_d 
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // free host memory of a_h, b_h, and c_h
    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}
