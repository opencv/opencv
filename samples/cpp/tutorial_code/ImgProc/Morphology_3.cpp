/**
 * @file Morphology_3(Extract_Lines).cpp
 * @brief Use morphology transformations for extracting horizontal and vertical lines sample code
 * @author OpenCV team
 */

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int, char** argv)
{
//! [load_image]
    // Load the image
    Mat src = imread(argv[1]);

    // Check if image is loaded fine
    if(!src.data)
        cerr << "Problem loading image!!!" << endl;

    // Show source image
    imshow("src", src);
//! [load_image]

//! [gray]
    // Transform source image to gray if it is not
    Mat gray;

    if (src.channels() == 3)
    {
        cvtColor(src, gray, CV_BGR2GRAY);
    }
    else
    {
        gray = src;
    }

    // Show gray image
    imshow("gray", gray);
//! [gray]

//! [bin]
    // Apply adaptiveThreshold at the bitwise_not of gray, notice the ~ symbol
    Mat bw;
    adaptiveThreshold(~gray, bw, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, -2);

    // Show binary image
    imshow("binary", bw);
//! [bin]

//! [init]
    // Create the images that will use to extract the horizontal and vertical lines
    Mat horizontal = bw.clone();
    Mat vertical = bw.clone();
//! [init]

//! [horiz]
    // Specify size on horizontal axis
    int horizontalsize = horizontal.cols / 30;

    // Create structure element for extracting horizontal lines through morphology operations
    Mat horizontalStructure = getStructuringElement(MORPH_RECT, Size(horizontalsize,1));

    // Apply morphology operations
    erode(horizontal, horizontal, horizontalStructure, Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, Point(-1, -1));

    // Show extracted horizontal lines
    imshow("horizontal", horizontal);
//! [horiz]

//! [vert]
    // Specify size on vertical axis
    int verticalsize = vertical.rows / 30;

    // Create structure element for extracting vertical lines through morphology operations
    Mat verticalStructure = getStructuringElement(MORPH_RECT, Size( 1,verticalsize));

    // Apply morphology operations
    erode(vertical, vertical, verticalStructure, Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, Point(-1, -1));

    // Show extracted vertical lines
    imshow("vertical", vertical);
//! [vert]

//! [smooth]
    // Inverse vertical image
    bitwise_not(vertical, vertical);
    imshow("vertical_bit", vertical);

    // Extract edges and smooth image according to the logic
    // 1. extract edges
    // 2. dilate(edges)
    // 3. src.copyTo(smooth)
    // 4. blur smooth img
    // 5. smooth.copyTo(src, edges)

    // Step 1
    Mat edges;
    adaptiveThreshold(vertical, edges, 255, CV_ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, -2);
    imshow("edges", edges);

    // Step 2
    Mat kernel = Mat::ones(2, 2, CV_8UC1);
    dilate(edges, edges, kernel);
    imshow("dilate", edges);

    // Step 3
    Mat smooth;
    vertical.copyTo(smooth);

    // Step 4
    blur(smooth, smooth, Size(2, 2));

    // Step 5
    smooth.copyTo(vertical, edges);

    // Show final result
    imshow("smooth", vertical);
//! [smooth]

    waitKey(0);
    return 0;
}