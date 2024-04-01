/*******************************************************************************
 *
 * This program demonstrates various shape fitting techniques using OpenCV.
 * It reads an image, applies binary thresholding, and then detects contours.
 *
 * For each contour, it fits and draws several geometric shapes including
 * convex hulls, minimum enclosing circles, rectangles, triangles, and ellipses
 * using different fitting methods:
 *  1: OpenCV's original method fitEllipse which implements Fitzgibbon 1995 method.
 *  2: The Approximate Mean Square (AMS) method fitEllipseAMS  proposed by Taubin 1991
 *  3: The Direct least square (Direct) method fitEllipseDirect proposed by Fitzgibbon1999
 *
 * The results are displayed with unique colors
 * for each shape and fitting method for clear differentiation.
 *
 *
 *********************************************************************************/

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

string hot_keys =
    "\n\nHot keys: \n"
    "\tESC - quit the program\n"
    "\tq - quit the program\n"
    "\tc - make the circle\n"
    "\tr - make the rectangle\n"
    "\th - make the convexhull\n"
    "\tt - make the triangle\n"
    "\te - make the ellipse\n"
    "\ta - make all shapes\n";

static void help(char** argv)
{
    cout << "\nThis program demonstrates various shape fitting techniques on a set of points using functions: \n"
         << "minAreaRect(), minEnclosingTriangle(), minEnclosingCircle(), convexHull(), and ellipse().\n"
         << "Three methods are used to find the elliptical fits: \n"
         << "0: fitEllipse (OpenCV), 1: fitEllipseAMS, 2: fitEllipseDirect.\n"
         << "These can be changed using the trackbar or command line argument.\n\n"
         << "Usage: " << argv[0] << " [--image_name=<image_path> Default: ellipses.jpg] [--ellipse_method=<0/1/2> -- Default: 0 (OpenCV)]\n\n";
    cout << hot_keys << endl;
}

void processImage(int, void*);
void drawShapes(Mat &img, vector<Point> &points);
void drawConvexHull(Mat &img, vector<Point> &points);
void drawMinAreaRect(Mat &img, vector<Point> &points);
void drawFittedEllipses(Mat &img, vector<Point> &points);
void drawMinEnclosingCircle(Mat &img, vector<Point> &points);
void drawMinEnclosingTriangle(Mat &img, const vector<Point> &points);

// Shape fitting options
Mat image;
int sliderPos = 70;
string shape = "--all";
int ellipseMethod = 0;

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv,
    "{help h||}{@image|ellipses.jpg|}{ellipse_method|0|Ellipse fitting method: 0 for OpenCV, 1 for AMS, 2 for Direct.}");

    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    help(argv);
    ellipseMethod = parser.get<int>("ellipse_method");

    string filename = parser.get<string>("@image");
    image = imread(samples::findFile(filename), IMREAD_COLOR);  // Read the image from the specified path

    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    namedWindow("Shapes", WINDOW_AUTOSIZE);  // Create a window to display the results
    createTrackbar("Threshold", "Shapes", NULL, 255, processImage, &sliderPos);  // Create a threshold trackbar
    setTrackbarPos("Threshold", "Shapes", sliderPos);

    for(;;) {
        char key = (char)waitKey(0);  // Listen for a key press
        if (key == 'q' || key == 27) break;  // Exit the loop if 'q' or ESC is pressed

        switch (key) {
            case 'h': shape = "--convexhull"; break;
            case 'a': shape = "--all"; break;
            case 't': shape = "--triangle"; break;
            case 'c': shape = "--circle"; break;
            case 'e': shape = "--ellipse"; break;
            case 'r': shape = "--rectangle"; break;
            default: break;  // Do nothing for other keys
        }

        processImage(sliderPos, &sliderPos);  // Process the image with the current settings
    }

    return 0;
}

// Function to draw the minimum enclosing circle around given points
void drawMinEnclosingCircle(Mat &img, vector<Point> &points) {
    Point2f center;
    float radius = 0;
    minEnclosingCircle(points, center, radius);  // Find the enclosing circle
    // Draw the circle
    circle(img, center, cvRound(radius), Scalar(0, 0, 255), 2, LINE_AA);
}

// Function to draw the minimum enclosing triangle around given points
void drawMinEnclosingTriangle(Mat &img, const vector<Point> &points) {
    vector<Point2f> triangle;
    minEnclosingTriangle(points, triangle);  // Find the enclosing triangle

    if (triangle.size() != 3) {
        return;
    }

    // Draw the triangle
    for (int i = 0; i < 3; i++) {
        line(img, triangle[i], triangle[(i + 1) % 3], Scalar(255, 0, 0), 2, LINE_AA);
    }
}

// Function to draw the minimum area rectangle around given points
void drawMinAreaRect(Mat &img, vector<Point> &points) {
    RotatedRect box = minAreaRect(points);  // Find the minimum area rectangle
    Point2f vtx[4];
    box.points(vtx);
    // Draw the rectangle
    for (int i = 0; i < 4; i++)
        line(img, vtx[i], vtx[(i+1)%4], Scalar(0, 255, 0), 2, LINE_AA);
}

// Function to draw the convex hull of given points
void drawConvexHull(Mat &img, vector<Point> &points) {
    vector<Point> hull;
    convexHull(points, hull, false);  // Find the convex hull
    // Draw the convex hull
    polylines(img, hull, true, Scalar(255, 255, 0), 2, LINE_AA);
}

inline static bool isGoodBox(const RotatedRect& box) {
    //size.height >= size.width awalys,only if the pts are on a line or at the same point,size.width=0
    return (box.size.height <= box.size.width * 30) && (box.size.width > 0);
}

// Function to draw fitted ellipses using different methods
void drawFittedEllipses(Mat &img, vector<Point> &points) {
    switch (ellipseMethod) {
        case 0: // Standard ellipse fitting
            {
                RotatedRect fittedEllipse = fitEllipse(points);
                if (isGoodBox(fittedEllipse)) {
                    ellipse(img, fittedEllipse, Scalar(255, 0, 255), 2, LINE_AA);
                }
                putText(img, "OpenCV", Point(img.cols - 80, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2, LINE_AA);
            }
            break;
        case 1: // AMS ellipse fitting
            {
                RotatedRect fittedEllipseAMS = fitEllipseAMS(points);
                if (isGoodBox(fittedEllipseAMS)) {
                    ellipse(img, fittedEllipseAMS, Scalar(255, 0, 255), 2, LINE_AA);
                }
                putText(img, "AMS", Point(img.cols - 80, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2, LINE_AA);
            }
            break;
        case 2: // Direct ellipse fitting
            {
                RotatedRect fittedEllipseDirect = fitEllipseDirect(points);
                if (isGoodBox(fittedEllipseDirect)) {
                    ellipse(img, fittedEllipseDirect, Scalar(255, 0, 255), 2, LINE_AA);
                }
                putText(img, "Direct", Point(img.cols - 80, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2, LINE_AA);
            }
            break;
        default: // Default case falls back to OpenCV method
            {
                RotatedRect fittedEllipse = fitEllipse(points);
                if (isGoodBox(fittedEllipse)) {
                    ellipse(img, fittedEllipse, Scalar(255, 0, 255), 2, LINE_AA);
                }
                putText(img, "OpenCV (default)", Point(img.cols - 80, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2, LINE_AA);
                cout << "Warning: Invalid ellipseMethod value. Falling back to default OpenCV method." << endl;
            }
        break;
    }
}

// Function to draw shapes
void drawShapes(Mat &img, vector<Point> &points) {
    if (shape == "--circle") {
        drawMinEnclosingCircle(img, points);
    } else if (shape == "--triangle") {
        drawMinEnclosingTriangle(img, points);
    } else if (shape == "--rectangle") {
        drawMinAreaRect(img, points);
    } else if (shape == "--convexhull") {
        drawConvexHull(img, points);
    } else if (shape == "--ellipse"){
        drawFittedEllipses(img, points);
    }
    else if (shape == "--all") {
        drawMinEnclosingCircle(img, points);
        drawMinEnclosingTriangle(img, points);
        drawMinAreaRect(img, points);
        drawConvexHull(img, points);
        drawFittedEllipses(img, points);
    }
}

// Main function to process the image based on the current trackbar position
void processImage(int position, void* userData){

    int* sliderPosition = static_cast<int*>(userData);

    *sliderPosition = position;

    Mat processedImg = image.clone();  // Clone the original image for processing
    Mat gray;
    cvtColor(processedImg, gray, COLOR_BGR2GRAY);  // Convert to grayscale
    threshold(gray, gray, *sliderPosition, 255, THRESH_BINARY);  // Apply binary threshold

    Mat filteredImg;
    medianBlur(gray, filteredImg, 3);

    vector<vector<Point>> contours;
    findContours(filteredImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);  // Find contours

    if (contours.empty()) {
        return;
    }

    imshow("mask", filteredImg);  // Show the mask
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].size() < 5) {  // Check if the contour has enough points
            continue;
        }
        drawShapes(processedImg, contours[i]);
    }

    imshow("Shapes", processedImg);  // Display the processed image with fitted shapes
}
