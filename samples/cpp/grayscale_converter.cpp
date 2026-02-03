#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

int main() {
    // Read the image
    Mat image = imread("example.jpg");
    if (image.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Convert to grayscale
    Mat grayImage;
    cvtColor(image, grayImage, COLOR_BGR2GRAY);

    // Display original and grayscale images
    imshow("Original Image", image);
    imshow("Grayscale Image", grayImage);

    waitKey(0); // Wait for a key press
    return 0;
}