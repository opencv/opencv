#include <opencv2/core.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    cout << "OpenCV " << CV_VERSION << endl;
    
    // Just create a mat
    Mat gray(1080, 1920, CV_8UC1, Scalar(128));
    cout << "Created gray: " << gray.size() << " step=" << gray.step << endl;
    
    // Test bitwise_and
    Mat gray2(1080, 1920, CV_8UC1, Scalar(64));
    Mat dst;
    bitwise_and(gray, gray2, dst);
    cout << "bitwise_and OK, result at (0,0)=" << (int)dst.at<uchar>(0,0) << endl;
    
    return 0;
}
