/*
This tutorial demonstrates how to correct the skewness in a text.
The program takes as input a skewed source image and shows non skewed text.
*/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
#include <iomanip>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, "{@input | imageTextR.png | input image}");

    // Load image from the disk
    Mat image = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    if (image.empty())
    {
        cout << "Cannot load the image " + parser.get<String>("@input") << endl;
        return -1;
    }

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    Mat thresh;
    threshold(gray, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // Applying erode filter to remove random noise
    int erosion_size = 1;
    Mat element = getStructuringElement(MORPH_RECT, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
    erode(thresh, thresh, element);

    cv::Mat coords;
    findNonZero(thresh, coords);

    RotatedRect box = minAreaRect(coords);
    float angle = box.angle;

    // The cv::minAreaRect function returns values in the range [-90, 0)
    // if the angle is less than -45 we need to add 90 to it
    if (angle < -45.0f)
    {
        angle = (90.0f + angle);
    }

    // Obtaining the rotation matrix
    Point2f center((image.cols) / 2.0f, (image.rows) / 2.0f);
    Mat M = getRotationMatrix2D(center, angle, 1.0f);
    Mat rotated;

    // Rotating the image by required angle
    stringstream angle_to_str;
    angle_to_str << fixed << setprecision(2) << angle;
    warpAffine(image, rotated, M, image.size(), INTER_CUBIC, BORDER_REPLICATE);
    putText(rotated, "Angle " + angle_to_str.str() + " degrees", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
    cout << "[INFO] angle: " << angle_to_str.str() << " degrees" << endl;

    // Create subdirectory for saving results
    string output_dir = "skewness_correction";
    mkdir(output_dir.c_str(), 0777);

    // Save the rotated image to the subdirectory
    string output_path = output_dir + "/corrected_image.png";
    imwrite(output_path, rotated);

    cout << "Image saved to " << output_path << endl;

    // Comment out all display-related functions
    // imshow("Input", image);
    // imshow("Rotated", rotated);
    // waitKey(0);

    return 0;
}

