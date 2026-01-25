// C++ program for the above approach 
#include <cstdlib>
#include <iostream> 
#include <opencv2/opencv.hpp> 
#include <random>
using namespace std; 
using namespace cv;


float boundary_condition(float x, float y){
	return sin(10*(x+y));
}

float right_side(float x, float y){
	return 4*(x-y);
}

int main(int argc, char** argv) 
{ 
	const int sizex = 50;
	const int sizey = 50;
	const int steps = 1000;

	// Array with result values
  	float *array = new float[sizex*sizey];
	// Array which indicates if there is a boundary (always Dirichlet)
  	bool *boundaries = new bool[sizex*sizey];
	for(int x = 0; x < sizex; x++){
		for(int y = 0; y < sizey; y++){
			int id = x + sizex*y;
			array[id] = 0.;
			boundaries[id] = false;
			if((x == 0 || x == sizex-1) || (y == 0 || y == sizey-1)){
				boundaries[id] = true;
				array[id] = 125*boundary_condition(1.*x/sizex, 1.*y/sizey)+125;
			}
		}
	}

  	for(int x = 0; x < sizex; x++){
		for(int y = 0; y < sizey; y++){
		const int id = x + sizex*y;
		// Stop if it is on boundary
		if(boundaries[id]){
			continue;
		}
		// Variable to save the sum of right sides of the equation
		float right_hand_sum = 0.;
		// Repeat multiple times
		for(int i = 0; i < steps; i++){
			int posx = x;
			int posy = y;
			// Move until a boundary is reached
			while(!boundaries[posx + sizex*posy]){
				const int dec = rand()%2;
				const int move = rand()%2;
				right_hand_sum += right_side(posx,posy);
				if(dec == 0){
					posx += 2*(move)-1;
				}else{
					posy += 2*(move)-1;
				}
			}
			array[id] += array[posx + sizex*posy];
		}
		// Save the average to the array
		// speedup: doesnt divide by number of steps each time, but only here at the end
		array[id] = array[id]/steps + right_hand_sum/steps/steps/4;
  	  }
  	}
	Mat result = Mat(sizex, sizey, CV_32FC1, array);
	Mat image;
	// Float image output
	normalize(result, image, 0, 255, NORM_MINMAX);
	cv::imwrite("float_output.exr", image);
	
	// Png output with a colormap
	image.convertTo(image, CV_8UC1);
	Mat image_color;
	applyColorMap(image, image_color, COLORMAP_VIRIDIS);
	imwrite("./colored_output.png", image_color );

	/* namedWindow("graywindow");
	imshow("graywindow", image);
	namedWindow("window");
	imshow("window", image_color); */

	delete [] array;
	delete [] boundaries;
	return 0; 
}
