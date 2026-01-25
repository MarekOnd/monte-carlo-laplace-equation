#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/mat.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <driver_types.h>
#include <iostream>
#include <opencv2/opencv.hpp> 
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream> 
#include <random>

using namespace std; 
using namespace cv;

#define CUDA_CHECK_ERROR(call) {                                          \
    cudaError_t err = call;                                               \
    if (err != cudaSuccess) {                                             \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__; \
        std::cerr << ": " << cudaGetErrorString(err) << std::endl;        \
        exit(EXIT_FAILURE);                                               \
    }                                                                     \
}

__device__ void my_rng(int &a)
{   
   	a = (1103515245*a + 12345)%2147483649;

}

__global__ void monte_carlo_laplace_boundary_method(float *arr, bool *boundaries, int* seeds, int sizex, int sizey, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sizex*sizey) {
		if(boundaries[idx] ||
		!((int)(boundaries[idx+1]) + (int)(boundaries[idx-1]) + (int)(boundaries[idx+sizex]) + (int)(boundaries[idx-sizex]) >= 2)){
			return;
		}
		int x = idx%sizex;
		int y = idx / sizex;
		int r = seeds[idx];
		float sum = 0.;
		for(int i = 0; i < steps; i++){
			int posx = x;
			int posy = y;
			while(!boundaries[posx + sizex*posy]){
				my_rng(r);
				int dec  = (r>>12)&1;
				int move = (r>>24)&1;
				if(dec == 0){
					posx += 2*(move)-1;
				}else{
					posy += 2*(move)-1;
				}
			}
			sum += arr[posx + sizex*posy];
		}
		arr[idx] = (float)(sum/steps);
		boundaries[idx] = true;
		//arr[idx] = (float)(255.0*idx/(sizex*sizey));
    }
} 

__device__ float right_side(float x, float y){
	return sin((x+y))+1;
}

__global__ void monte_carlo_laplace(float *arr, bool *boundaries, int* seeds, int sizex, int sizey, int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sizex*sizey) {
		if(boundaries[idx]){
			return;
		}
		int x = idx%sizex;
		int y = idx / sizex;
		int r = seeds[idx];
		float sum = 0.;
		float right_side_sum = 0.;
		for(int i = 0; i < steps; i++){
			int posx = x;
			int posy = y;
			while(!boundaries[posx + sizex*posy]){
				right_side_sum += right_side(posx,posy);
				my_rng(r);
				int dec  = (r>>12)&1;
				int move = (r>>24)&1;
				if(dec == 0){
					posx += 2*(move)-1;
				}else{
					posy += 2*(move)-1;
				}
			}
			sum += arr[posx + sizex*posy];
		}
		arr[idx] = (float)(sum/steps+right_side_sum/steps/steps/4);
    }
} 



float boundary_condition(float x, float y){
	return sin(4*(x+y))+1;
}

int main(int argc, char** argv) { 
	const int sizex = 100;
	const int sizey = 100;
	const int steps = 1000;
  	float *array = new float[sizex*sizey];
  	bool *boundaries = new bool[sizex*sizey];
	int *seeds = new int[sizex*sizey];

    for(int x = 0; x < sizex; x++){
		for(int y = 0; y < sizey; y++){
			int id = x + sizex*y;
			array[id] = 0.;
			boundaries[id] = false;
			seeds[id] = rand();
			if((x == 0 || x == sizex-1) || (y == 0 || y == sizey-1)){
				boundaries[id] = true;
				array[id] = boundary_condition(1.*x/sizex, 1.*y/sizey);
			}
		}
	}

    bool *deviceBoundaries;
    cudaMalloc(&deviceBoundaries, sizex*sizey * sizeof(bool));
	cudaMemcpy(deviceBoundaries, boundaries, sizex*sizey * sizeof(bool), cudaMemcpyHostToDevice);

    float *deviceArray;
    cudaMalloc(&deviceArray, sizex*sizey * sizeof(float));
	cudaMemcpy(deviceArray, array, sizex*sizey * sizeof(float), cudaMemcpyHostToDevice);

	int *deviceSeeds;
    cudaMalloc(&deviceSeeds, sizex*sizey * sizeof(int));
	cudaMemcpy(deviceSeeds, array, sizex*sizey * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocksPerGrid = (sizex*sizey + threadsPerBlock - 1) / threadsPerBlock;
	monte_carlo_laplace<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, deviceBoundaries, deviceSeeds, sizex, sizey, steps);

	CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    cudaMemcpy(array, deviceArray, sizex*sizey * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(deviceArray);
    cudaFree(deviceBoundaries);

	Mat result = Mat(sizex, sizey, CV_32FC1, array);
	Mat image;
	normalize(result, image, 0, 255, NORM_MINMAX);
	
	image.convertTo(image, CV_8UC1);
	Mat image_color;
	applyColorMap(image, image_color, COLORMAP_VIRIDIS);
	imwrite("./result.png", image_color );

	namedWindow("graywindow");
	imshow("graywindow", image);
	namedWindow("window");
	imshow("window", image_color);
    waitKey(0);
    delete [] array;
	delete [] boundaries;
	delete [] seeds;
	return 0; 
}
