/*
 * Author: Samyak Datta (datta[dot]samyak[at]gmail.com)
 *
 * A program to demonstrate contour detection using the Active Contour
 * Model (Snake). The cvSnakeImage() method is applied to detect lip
 * contour in a lip ROI image.
 *
 */

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/legacy/legacy.hpp"

#include <iostream>
#include <cstdio>
#include <vector>

using namespace std;
using namespace cv;

static void help();
int returnLargestContourIndex(vector<vector<Point> > contours);

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        help();
        return 1;
    }

    /*
     * Load a grayscale image from the command-line and run's Otsu's thresholding
     * algorithm to convert it into a binary image. Otsu's thresholding assumes a
     * bi-modal distribution of the input image where the 2 histogram peaks correpsond
     * to foreground and background pixels. The aim of the algorithm is to generate an
     * optimal threshold that separates the 2 peaks.
     */
    char* filename = argv[1];
    Mat image = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

    Mat binary_image(image.size(), image.type());
    threshold(image, binary_image, 0, 255, THRESH_BINARY|THRESH_OTSU);

    /*
     * Find the initial set of contour points for the cvSnakeImage() method using findContours().
     * A clone for the binary image is required because findContours() modifies the input image.
     */
    Mat binary_image_clone = binary_image.clone();
    vector<vector<Point> > contours;
    findContours(binary_image_clone, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

    /*
     * The largest contour (with respect to number of contour points) is chosen to
     * eliminate the effect of noise. The main idea behind doing this is that the largest
     * contour will correspond to the lip region.
     */
    int largest_contour_idx = returnLargestContourIndex(contours);
    vector<Point> largest_contour = contours[largest_contour_idx];

    /*
     * The Mat object representing the binary iamge returned by Otsu's thresholding is converted
     * to a C-style IplImage. The vector<Point> representing the largest contour (among the several
     * contours returned by findContours()) is also converted to the C-style structure CvPoint*.
     *
     * These conversions are necessary to match the argument types of cvSnakeImage().
     */
    IplImage* img = new IplImage(binary_image);
    CvPoint* contour_points = new CvPoint[largest_contour.size()];
    for(unsigned int i = 0; i < largest_contour.size(); ++i)
        contour_points[i] = largest_contour[i];

    /*
     * Run the active contour detection by calling cvSnakeImage(). The arguments expected by the function
     * are described in detail:
     *
     * -> IplImage* src: The input image (binary) where contours are to be detected.
     * -> CVPoints* points: The initial (seed) set of contour points for the Snake algorithm to work.
     * -> int length: The number of contour points in the initial seed contour.
     * -> float* alpha, beta, gamma: Co-efficients for the energy terms of the Snake algorithm
     * -> int coeffUsage: if CV_VALUE - alpha, beta and gamma point to a single value
     *                    if CV_MATAY - points to arrays
     * -> CvTermCriteria criteria: Terminaton criteria (fixed number of ietrations or minimum accuracy)
     */
    float alpha = 0.1f, beta = 0.4f, gama = 0.5f;
    cvSnakeImage(img, contour_points, static_cast<int>(largest_contour.size()), (float*)&alpha,
            (float*)&beta, (float*)&gama, CV_VALUE, cvSize(3, 3), TermCriteria(CV_TERMCRIT_ITER, 1000, 0.1), 2);

    /*
     * Initialize a blank image for drawing the contours detected by cvSnakeImage().
     * Plot the contour points on the blank image to depict the contour.
     */
    Mat_<uchar> snake_contour(binary_image.size());
    for(int i = 0; i < snake_contour.rows; ++i)
    {
        for(int j = 0; j < snake_contour.cols; ++j)
            snake_contour.at<uchar>(i, j) = 0;
    }

    for(unsigned int i = 0; i < largest_contour.size(); ++i)
    {
        CvPoint pt = contour_points[i];
        snake_contour.at<uchar>(pt.y, pt.x) = 255;
    }

    imshow("Snake-Contours", snake_contour);
    waitKey(0);
    return 0;
}

static void help()
{
    cout << "\nThis file demonstrates Active Contour detection using OpenCV's cvSnakeImage() method.\n"
        "The program has been tested for lip-contour detection and found to give good results.\n";

    cout << "\nUSAGE: ./cpp-example-active_contours [IMAGE]\n"
        "IMAGE\n\tPath to the image taken as input.\n"
        "\nAs an example, lips.png image file has been provided. You can execute the program by:\n"
        "./cpp-example-active_contours lips.png";
}

int returnLargestContourIndex(vector<vector<Point> > contours)
{
    unsigned int max_contour_size = 0;
    int max_contour_idx = -1;
    for(unsigned int i = 0; i < contours.size(); ++i)
    {
        if(contours[i].size() > max_contour_size)
        {
            max_contour_size = static_cast<int>(contours[i].size());
            max_contour_idx = i;
        }
    }
    return max_contour_idx;
}
