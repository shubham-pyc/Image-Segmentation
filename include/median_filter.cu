#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "point.h"
#include <time.h>
#define TILE_SIZE 4

using namespace std;

__global__ void median_filter_kernel(const thrust::device_ptr<int> inputImageKernel,
                                     thrust::device_ptr<int> outputImagekernel,
                                     int imageWidth, int imageHeight)
{
    //Set the row and col value for each thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int sharedmem[(TILE_SIZE + 2)][(TILE_SIZE + 2)]; //initialize shared memory
    //Take some values.
    bool is_x_left = (threadIdx.x == 0), is_x_right = (threadIdx.x == TILE_SIZE - 1);
    bool is_y_top = (threadIdx.y == 0), is_y_bottom = (threadIdx.y == TILE_SIZE - 1);

    //Initialize with zero
    if (is_x_left)
        sharedmem[threadIdx.x][threadIdx.y + 1] = 0;
    else if (is_x_right)
        sharedmem[threadIdx.x + 2][threadIdx.y + 1] = 0;
    if (is_y_top)
    {
        sharedmem[threadIdx.x + 1][threadIdx.y] = 0;
        if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y] = 0;
        else if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y] = 0;
    }
    else if (is_y_bottom)
    {
        sharedmem[threadIdx.x + 1][threadIdx.y + 2] = 0;
        if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y + 2] = 0;
        else if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y + 2] = 0;
    }

    //Setup pixel values
    sharedmem[threadIdx.x + 1][threadIdx.y + 1] = inputImageKernel[row * imageWidth + col];
    //Check for boundry conditions.
    if (is_x_left && (col > 0))
        sharedmem[threadIdx.x][threadIdx.y + 1] = inputImageKernel[row * imageWidth + (col - 1)];
    else if (is_x_right && (col < imageWidth - 1))
        sharedmem[threadIdx.x + 2][threadIdx.y + 1] = inputImageKernel[row * imageWidth + (col + 1)];
    if (is_y_top && (row > 0))
    {
        sharedmem[threadIdx.x + 1][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + col];
        if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + (col - 1)];
        else if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y] = inputImageKernel[(row - 1) * imageWidth + (col + 1)];
    }
    else if (is_y_bottom && (row < imageHeight - 1))
    {
        sharedmem[threadIdx.x + 1][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + col];
        if (is_x_right)
            sharedmem[threadIdx.x + 2][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + (col + 1)];
        else if (is_x_left)
            sharedmem[threadIdx.x][threadIdx.y + 2] = inputImageKernel[(row + 1) * imageWidth + (col - 1)];
    }

    __syncthreads(); //Wait for all threads to be done.

    //Setup the filter.
    int filterVector[9] = {sharedmem[threadIdx.x][threadIdx.y], sharedmem[threadIdx.x + 1][threadIdx.y], sharedmem[threadIdx.x + 2][threadIdx.y],
                           sharedmem[threadIdx.x][threadIdx.y + 1], sharedmem[threadIdx.x + 1][threadIdx.y + 1], sharedmem[threadIdx.x + 2][threadIdx.y + 1],
                           sharedmem[threadIdx.x][threadIdx.y + 2], sharedmem[threadIdx.x + 1][threadIdx.y + 2], sharedmem[threadIdx.x + 2][threadIdx.y + 2]};

    {
        for (int i = 0; i < 9; i++)
        {
            for (int j = i + 1; j < 9; j++)
            {
                if (filterVector[i] > filterVector[j])
                {
                    //Swap Values.
                    int tmp = filterVector[i];
                    filterVector[i] = filterVector[j];
                    filterVector[j] = tmp;
                }
            }
        }
        outputImagekernel[row * imageWidth + col] = filterVector[4]; //Set the output image values.
    }
}

// bool median_filter_cuda(Bitmap *image, Bitmap *outputImage, bool sharedMemoryUse)
using DataFrame = vector<Point>;
DataFrame median_filter_cuda(const DataFrame &image, int width, int height)
{
    //Cuda error and image values.
    // int width = image->Width();
    // int height = image->Height();

    thrust::host_vector<int> h_image;
    thrust::host_vector<int> h_output;

    //converting std::vector to thrust
    for (int i = 0; i < image.size(); i++)
    {
        h_image.push_back(image[i].x);
    }
    thrust::device_vector<int> d_image = h_image;

    thrust::device_vector<int> d_output(image.size(), 0);

    //initialize images.

    //take block and grids.
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((int)ceil((float)width / (float)TILE_SIZE),
                 (int)ceil((float)height / (float)TILE_SIZE));

    //Check for shared memories and call the kernel
    median_filter_kernel<<<dimGrid, dimBlock>>>(d_image.data(), d_output.data(), width, height);

    // save output image to host.
    h_output = d_output;

    DataFrame ret_value;

    for (int i = 0; i < image.size(); i++)
    {
        Point point = {.x = h_output[i]};
        ret_value.push_back(point);
    }

    return ret_value;
    //Free the memory
    // cudaFree(deviceinputimage);
    // cudaFree(deviceOutputImage);
}