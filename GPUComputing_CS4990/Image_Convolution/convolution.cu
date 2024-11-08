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
__constant__ float F_d[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

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
    // imwrite("blurred.jpg", blurred);
    return blurred;
}

void blurImage_h(cv::Mat Pout_Mat_h, cv::Mat Pin_Mat_h, unsigned int nRows, unsigned int nCols)
{

    Pout_Mat_h = Mat::zeros(nRows, nCols, CV_32F);

    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {

            if (i < FILTER_RADIUS || j < FILTER_RADIUS || (nRows - i <= FILTER_RADIUS) || (nCols - j <= FILTER_RADIUS))
                continue;
            else
            {
                int filterDimension = 2 * FILTER_RADIUS + 1;
                float sum;
                for (int k = 0; k < filterDimension; k++)
                {
                    for (int l = 0; l < filterDimension; l++)
                    {
                        sum += Pout_Mat_h.at(i + k, j + l) * Pin_Mat_h.at(i + k, j + l);
                    }
                }
            }
        }
    }
}

/* Convolution Functions*/
/* END */
/******************************************************************************************************* */

int main(int argc, char *argv[])
{

    char *file_name = argv[1];
    printf("Blurring file: \"%s\"\n", file_name);

    // for comparison purpose, use OpenCV's 2D Filter function
    Mat image = cv::imread(file_name, IMREAD_GRAYSCALE);
    unsigned int nRows = image.rows, nCols = image.cols, nChannels = image.channels();

    double start_time = myCPUTimer();
    opencv_convolution(image);
    double end_time = myCPUTimer();
    double elapsed_time = end_time - start_time;

    printf("\nOpenCV filter2D(image, blurred,  -1, kernel1): %f s\n", elapsed_time);

    // for comparison purpose, implement a CPU version
    Mat blurred_imaged;
    start_time = myCPUTimer();
    blurImage_h(blurred_imaged, image, nRows, nCols);
    end_time = myCPUTimer();
    elapsed_time = end_time - start_time;

    printf("\nCPU Version blurImage_h(blurred_imaged, image, nRows, nCols): %f s\n", elapsed_time);

    // for comparison purpose, implement a CUDA kernel but without tiling
    // cudaMemcpyToSymbol(F, F_h, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));

    // verify(answer, answer, bwImage.rows, bwImage.cols);

    // for comparison purpose, implement a CUDA kernel but with tiling and constant memory

    return 0;
}