#include <driver_types.h>
#include <iostream>
#include <opencv2/opencv.hpp> 
#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream> 
#include <random>

using namespace std; 
using namespace cv;

__device__ void my_rng(int &a)
{   
   	a = (1103515245*a + 12345)%2147483649;

}

__global__ void monte_carlo_laplace(float *arr, bool *boundaries, int* seeds, int sizex, int sizey, int parallel_samples,  int steps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sizex*sizey*parallel_samples) {
        int s = idx / (sizex * sizey);
        int residual2D = idx % (sizex * sizey);
        int y = residual2D / sizex;
        int x = residual2D % sizex;
        //int id = x+sizex*y;
		if(boundaries[idx] ||!((int)(boundaries[idx+1]) + (int)(boundaries[idx-1]) + (int)(boundaries[idx+sizex]) + (int)(boundaries[idx-sizex]) >= 2)){
            return;
		}
		int r = seeds[idx];
		float sum = 0.;
		for(int i = 0; i < steps; i++){
			int posx = x;
			int posy = y;
			while(!boundaries[posx + sizex*posy + sizex*sizey*s]){
				my_rng(r);
				int dec  = (r>>12)&1;
				int move = (r>>24)&1;
				if(dec == 0){
					posx += 2*(move)-1;
				}else{
					posy += 2*(move)-1;
				}
			}
			sum += arr[posx + sizex*posy+sizex*sizey*s];
		}
		arr[idx] = (float)(sum/steps);
		boundaries[idx] = true;
    }
} 

__global__ void average_out_values(float *arrs, int sizex, int sizey, int parallel_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < sizex*sizey) {
		float sum = 0.;
		for(int s = 0; s < parallel_samples; s++){
			sum += arrs[idx + sizex*sizey*s];
		}
		for(int s = 0; s < parallel_samples; s++){
			arrs[idx + sizex*sizey*s] = (float)(1.*sum/parallel_samples);
		}
    }
} 

float boundary_condition(float x, float y){
	return 122*sin(10*(x+y))+122;
}

int main(int argc, char** argv) { 
	const int sizex = 1000;
	const int sizey = 1000;
	const int steps = 10;
    const int parallel_samples = 100;
  	float *array = new float[sizex*sizey*parallel_samples];
  	bool *boundaries = new bool[sizex*sizey*parallel_samples];
	int *seeds = new int[sizex*sizey*parallel_samples];

    for(int x = 0; x < sizex; x++){
		for(int y = 0; y < sizey; y++){
			int id = x + sizex*y;
            for(int s = 0; s < parallel_samples; s++){
				boundaries[id + sizex*sizey*s] = false;
			    array[id + sizex*sizey*s] = 0.;
                seeds[id + sizex*sizey*s] = rand();
            }
			if((x == 0 || x == sizex-1) || (y == 0 || y == sizey-1)){
                for(int s = 0; s < parallel_samples; s++){
			        array[id + sizex*sizey*s] = boundary_condition(1.*x/sizex, 1.*y/sizey);
					boundaries[id + sizex*sizey*s] = true;
                }
			}
		}
	}

    Mat ima = Mat(sizex*2, sizey, CV_32FC1, array);
    imwrite("./check.png", ima );

    bool *deviceBoundaries;
    cudaMalloc(&deviceBoundaries, sizex*sizey*parallel_samples * sizeof(bool));
	cudaMemcpy(deviceBoundaries, boundaries, sizex*sizey*parallel_samples * sizeof(bool), cudaMemcpyHostToDevice);

    float *deviceArray;
    cudaMalloc(&deviceArray, sizex*sizey*parallel_samples * sizeof(float));
	cudaMemcpy(deviceArray, array, sizex*sizey*parallel_samples * sizeof(float), cudaMemcpyHostToDevice);

	int *deviceSeeds;
    cudaMalloc(&deviceSeeds, sizex*sizey*parallel_samples * sizeof(int));
	cudaMemcpy(deviceSeeds, seeds, sizex*sizey*parallel_samples * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 16;
    int blocksPerGrid = (sizex*sizey*parallel_samples + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridSum = (sizex*sizey + threadsPerBlock - 1) / threadsPerBlock;
	for (int i = 0; i < sizex; i++) {
		monte_carlo_laplace<<<blocksPerGrid, threadsPerBlock>>>(deviceArray, deviceBoundaries, deviceSeeds, sizex, sizey, parallel_samples, steps);
		cudaDeviceSynchronize();
    	average_out_values<<<blocksPerGridSum, threadsPerBlock>>>(deviceArray, sizex, sizey, parallel_samples);
		cudaDeviceSynchronize();
		cout << 1.*i/sizex << endl;
	}

    cudaMemcpy(array, deviceArray, sizex*sizey*parallel_samples * sizeof(float), cudaMemcpyDeviceToHost);

    //Mat im = Mat(sizex*40, sizey, CV_32FC1, array);
    //imwrite("./check_end.png", im );

    cudaFree(deviceArray);
    cudaFree(deviceBoundaries);
    cudaFree(deviceSeeds);

    float* resultArray = new float[sizex*sizey];
    for (int i = 0; i < sizex*sizey; i++) {
        resultArray[i] = array[i];
    }

	Mat image = Mat(sizex, sizey, CV_32FC1, resultArray);
	imwrite("./result.png", image );
    delete [] resultArray;
    delete [] array;
	delete [] boundaries;
	delete [] seeds;
	return 0; 
}
