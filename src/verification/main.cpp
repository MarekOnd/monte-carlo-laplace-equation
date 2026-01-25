#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Error: Image not found" << std::endl;
        return -1;
    }
    cv::imshow("Original Grayscale Image", image);

    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_16S, 3); // 3x3 kernel

    cv::Mat laplacian_display;
    cv::convertScaleAbs(laplacian, laplacian_display);

    cv::imshow("Laplacian Approximation", laplacian_display);

    cv::waitKey(0);

    return 0;
}
