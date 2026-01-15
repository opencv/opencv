#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;

int main()
{
    // Use false for the 'required' parameter to prevent a C++ exception crash.
    // This matches the robust logic used in the Python tutorial.
    std::string image_path = samples::findFile("starry_night.jpg", false);

    if (image_path.empty())
    {
        std::cout << "Could not find 'starry_night.jpg'." << std::endl;
        std::cout << "Please download it from: https://github.com/opencv/opencv/blob/4.x/samples/data/starry_night.jpg" << std::endl;
        return -1;
    }

    Mat img = imread(image_path, IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window

    if(k == 's')
    {
        imwrite("starry_night.png", img);
    }

    return 0;
}
