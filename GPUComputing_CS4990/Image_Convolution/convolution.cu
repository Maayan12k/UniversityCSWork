#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cmath>
#include <math.h>
#include <cstdio>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define FILTER_RADIUS 2
const float F_h[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25},
    {1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25, 1.0f / 25}};
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

/**
    module load opencv/4.9.0.x86_64
*/

/******************************************************************************************************* */
/* Helper Functions*/
/* START */

#define CHECK(call)                                                                  \
    {                                                                                \
        const cudaError_t cuda_ret = call;                                           \
        if (cuda_ret != cudaSuccess)                                                 \
        {                                                                            \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
            printf("code: %d, reason:%s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
            exit(-1);                                                                \
        }                                                                            \
    }

double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec / 1.0e6);
}

bool verify(cv::Mat answer1, cv::Mat answer2, unsigned int nRows, unsigned int nCols)
{
    const float relativeTolerance = 1e-2;

    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            float relativeError = ((float)answer1.at<unsigned char>(i, j) - (float)answer2.at<unsigned char>(i, j)) / 255;
            if (relativeError > relativeTolerance || relativeError < -relativeTolerance)
            {
                printf("TEST FAILED at (%d, %d) with relativeError: %f\n", i, j, relativeError);
                printf("    answer1.at<unsigned char>(%d, %d): %u\n", i, j, answer1.at<unsigned char>(i, j));
                printf("    answer2.at<unsigned char>(%d, %d): %u\n\n", i, j, answer2.at<unsigned char>(i, j));
                return false;
            }
        }
    }

    printf("TEST PASSED\n\n");
    return true;
}

/* END */
/* Helper Functions*/
/******************************************************************************************************* */

/******************************************************************************************************* */
/* Convolution Functions*/
/* START */

Mat opencv_convolution(Mat bwImage)
{
    Mat kernel1 = Mat::ones(5, 5, CV_64F);
    kernel1 = kernel1 / 25;
    Mat blurred;
    filter2D(bwImage, blurred, -1, kernel1);
    return blurred;
}

void blurImage_h(Mat &Pout_Mat_h, const cv::Mat &Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

    Pout_Mat_h = Mat::zeros(nRows, nCols, CV_8U);

    const float filterValue = 1.0f / 25.0f;

    for (int i = FILTER_RADIUS; i < nRows - FILTER_RADIUS; i++)
    {
        for (int j = FILTER_RADIUS; j < nCols - FILTER_RADIUS; j++)
        {

            float sum = 0.0f;
            for (int k = -FILTER_RADIUS; k <= FILTER_RADIUS; k++)
            {
                for (int l = -FILTER_RADIUS; l <= FILTER_RADIUS; l++)
                {
                    sum += filterValue * Pin_Mat_h.at<unsigned char>(i + k, j + l);
                }
            }
            Pout_Mat_h.at<unsigned char>(i, j) = sum;
        }
    }
}

__global__ void blurImage_Kernel(unsigned char *Pout, unsigned char *Pin, unsigned int width, unsigned int height)
{
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;

    if (outCol < width && outRow < height)
    {
        float Pvalue = 0.0f;
        int inRow = 0, inCol = 0;
        int condition = 2 * FILTER_RADIUS + 1;

        for (int fRow = 0; fRow < condition; fRow++)
        {
            for (int fCol = 0; fCol < condition; fCol++)
            {
                inRow = outRow - FILTER_RADIUS + fRow;
                inCol = outCol - FILTER_RADIUS + fCol;
                if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
                {
                    Pvalue += F[fRow][fCol] * (float)Pin[inRow * width + inCol];
                }
            }
        }

        Pout[outRow * width + outCol] = (unsigned char)min(max(Pvalue, 0.0f), 255.0f);
    }
}

void blurImage_d(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

    printf("\n\nblurImage_Kernel: \n");

    // (1) allocate device memory for arrays p_d
    unsigned char *Pin_d, *Pout_d;
    double start_time_malloc = myCPUTimer();
    cudaMalloc((void **)&Pin_d, sizeof(unsigned char) * nRows * nCols);
    cudaMalloc((void **)&Pout_d, sizeof(unsigned char) * nRows * nCols);
    double end_time_malloc = myCPUTimer();
    double elapsed_time_malloc = end_time_malloc - start_time_malloc;

    printf("\tcudaMalloc: \t\t\t\t\t\t\t\t%f s\n", elapsed_time_malloc);

    // (2) copy image matrix Pin_h to device memory Pin_d
    unsigned char *Pin_h = Pin_Mat_h.data;
    double start_time_memcpy = myCPUTimer();
    cudaMemcpy(Pin_d, Pin_h, sizeof(unsigned char) * nCols * nRows, cudaMemcpyHostToDevice);
    double end_time_memcpy = myCPUTimer();
    double elapsed_time_memcpy = end_time_memcpy - start_time_memcpy;

    printf("\tcudaMemcpy: \t\t\t\t\t\t\t\t%f s\n", elapsed_time_memcpy);

    // (3) call kernel to launch a grid of threads to perform the image convolution on GPU
    dim3 gridDim((nRows + 32 - 1) / 32, (nCols + 32 - 1) / 32);
    dim3 blockDim(32, 32);

    double start_time = myCPUTimer();
    blurImage_Kernel<<<gridDim, blockDim>>>(Pout_d, Pin_d, nRows, nCols);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    double end_time = myCPUTimer();
    double elapsed_time = end_time - start_time;

    printf("\tblurImage_Kernel<<<(%d, %d, 1), (%d, %d, 1)>>>: \t\t\t%f s\n", (nCols + 32 - 1) / 32, (nRows + 32 - 1) / 32, 32, 32, elapsed_time);

    // (4) Copy the result data from device memory of array Pout_d to host memory of array Pout_h
    Pout_Mat_h = cv::Mat::zeros(nRows, nCols, CV_8U);
    unsigned char *Pout_h = Pout_Mat_h.data;
    double start_time_memcpy2 = myCPUTimer();
    cudaMemcpy(Pout_h, Pout_d, sizeof(unsigned char) * nCols * nRows, cudaMemcpyDeviceToHost);
    double end_time_memcpy2 = myCPUTimer();
    double elapsed_time_memcpy2 = end_time_memcpy2 - start_time_memcpy2;

    // printf("\n\n\n");
    // for(int i = 0; i < nCols*nRows; i++){
    //     printf("%hhu ", Pout_h);
    // }
    // printf("\n\n\n");

    printf("\tcudaMemcpy: \t\t\t\t\t\t\t\t%f s\n\n", elapsed_time_memcpy2);

    double total_elapsed_time = elapsed_time_malloc + elapsed_time_memcpy + elapsed_time + elapsed_time_memcpy2;

    printf("Total elapsed time for convolution without tiling: %f s\n", total_elapsed_time);

    // (5) free device memory of Pin_d and Pout_d
    cudaFree(Pin_d);
    cudaFree(Pout_d);
}

/* Convolution Functions*/
/* END */
/******************************************************************************************************* */

int main(int argc, char *argv[])
{

    char *file_name = argv[1];
    printf("Blurring file: \"%s\"\n", file_name);

    // for comparison purpose, use OpenCV's 2D Filter function
    Mat Pin_Mat_h = cv::imread(file_name, IMREAD_GRAYSCALE);
    unsigned int nRows = Pin_Mat_h.rows, nCols = Pin_Mat_h.cols, nChannels = Pin_Mat_h.channels();

    printf("\n\n\n Dimension of image: %d %d \n\n\n", nRows, nCols);

    double start_time = myCPUTimer();
    Mat blurred_mat = opencv_convolution(Pin_Mat_h);
    double end_time = myCPUTimer();
    double elapsed_time = end_time - start_time;

    printf("\nOpenCV filter2D(image, blurred,  -1, kernel1): %f s\n", elapsed_time);

    // for comparison purpose, implement a CPU version
    Mat blurred_imaged;
    start_time = myCPUTimer();
    blurImage_h(blurred_imaged, Pin_Mat_h, nRows, nCols);
    end_time = myCPUTimer();
    elapsed_time = end_time - start_time;

    std::time_t t = std::time(nullptr);
    std::tm *now = std::localtime(&t);

    std::ostringstream oss;
    oss << "blurred_("
        << (now->tm_year + 1900) << '-'
        << (now->tm_mon + 1) << '-'
        << now->tm_mday << '_'
        << now->tm_sec
        << ").jpg";

    std::string filename = oss.str();
    imwrite(filename, blurred_imaged);
    printf("\nCPU Version blurImage_h(blurred_imaged, image, nRows, nCols): %f s\n", elapsed_time);

    // for comparison purpose, implement a CUDA kernel but without tiling
    Mat Pout_Mat_h(nRows, nCols, CV_8U);
    cudaMemcpyToSymbol(F, F_h, (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));
    blurImage_d(Pout_Mat_h, Pin_Mat_h, nRows, nCols);

    std::ostringstream oss_kernel;
    oss_kernel << "blurred_kernel("
               << (now->tm_year + 1900) << '-'
               << (now->tm_mon + 1) << '-'
               << now->tm_mday << '_'
               << now->tm_sec
               << ").jpg";

    std::string filename_kernel = oss_kernel.str();
    imwrite(filename_kernel, Pout_Mat_h);

    // for comparison purpose, implement a CUDA kernel but with tiling and constant memory

    return 0;
}
