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

const string hot_keys =
    "\n\nHot keys: \n"
    "\tESC - quit the program\n"
    "\tq - quit the program\n"
    "\tc - make the circle\n"
    "\tr - make the rectangle\n"
    "\th - make the convexhull\n"
    "\tt - make the triangle\n"
    "\te - make the ellipse\n"
    "\ta - make all shapes\n"
    "\t0 - use OpenCV's method for ellipse fitting\n"
    "\t1 - use Approximate Mean Square (AMS) method for ellipse fitting \n"
    "\t2 - use Direct least square (Direct) method for ellipse fitting\n";

static void help(char** argv)
{
    cout << "\nThis program demonstrates various shape fitting techniques on a set of points using functions: \n"
         << "minAreaRect(), minEnclosingTriangle(), minEnclosingCircle(), convexHull(), and ellipse().\n\n"
         << "Usage: " << argv[0] << " [--image_name=<image_path> Default: ellipses.jpg]\n\n";
    cout << hot_keys << endl;
}

void processImage(int, void*);
void drawShapes(Mat &img, const vector<Point> &points, int ellipseMethod, string shape);
void drawConvexHull(Mat &img, const vector<Point> &points);
void drawMinAreaRect(Mat &img, const vector<Point> &points);
void drawFittedEllipses(Mat &img, const vector<Point> &points, int ellipseMethod);
void drawMinEnclosingCircle(Mat &img, const vector<Point> &points);
void drawMinEnclosingTriangle(Mat &img, const vector<Point> &points);

// Shape fitting options
Mat image;
enum EllipseFittingMethod {
    OpenCV_Method,
    AMS_Method,
    Direct_Method
};

struct UserData {
    int sliderPos = 70;
    string shape = "--all";
    int ellipseMethod = OpenCV_Method;
};

const char* keys =
    "{help h          |            | Show help message }"
    "{@image          |ellipses.jpg| Path to input image file }";

int main(int argc, char** argv) {

    cv::CommandLineParser parser(argc, argv, keys);
    help(argv);

    if (parser.has("help"))
    {
        return 0;
    }

    UserData userData; // variable to pass all the necessary values to trackbar callback

    string filename = parser.get<string>("@image");
    image = imread(samples::findFile(filename), IMREAD_COLOR);  // Read the image from the specified path

    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    namedWindow("Shapes", WINDOW_AUTOSIZE);  // Create a window to display the results
    createTrackbar("Threshold", "Shapes", NULL, 255, processImage, &userData);  // Create a threshold trackbar
    setTrackbarPos("Threshold", "Shapes", userData.sliderPos);

    for(;;) {
        char key = (char)waitKey(0);  // Listen for a key press
        if (key == 'q' || key == 27) break;  // Exit the loop if 'q' or ESC is pressed

        switch (key) {
            case 'h': userData.shape = "--convexhull"; break;
            case 'a': userData.shape = "--all"; break;
            case 't': userData.shape = "--triangle"; break;
            case 'c': userData.shape = "--circle"; break;
            case 'e': userData.shape = "--ellipse"; break;
            case 'r': userData.shape = "--rectangle"; break;
            case '0': userData.ellipseMethod = OpenCV_Method; break;
            case '1': userData.ellipseMethod = AMS_Method; break;
            case '2': userData.ellipseMethod = Direct_Method; break;
            default: break;  // Do nothing for other keys
        }

        processImage(userData.sliderPos, &userData);  // Process the image with the current settings
    }

    return 0;
}

// Function to draw the minimum enclosing circle around given points
void drawMinEnclosingCircle(Mat &img, const vector<Point> &points) {
    Point2f center;
    float radius = 0;
    minEnclosingCircle(points, center, radius);  // Find the enclosing circle
    // Draw the circle
    circle(img, center, cvRound(radius), Scalar(0, 0, 255), 2, LINE_AA);
}

// Function to draw the minimum enclosing triangle around given points
void drawMinEnclosingTriangle(Mat &img, const vector<Point> &points) {
    vector<Point> triangle;
    minEnclosingTriangle(points, triangle);  // Find the enclosing triangle

    if (triangle.size() != 3) {
        return;
    }
    // Use polylines to draw the triangle. The 'true' argument closes the triangle.
    polylines(img, triangle, true, Scalar(255, 0, 0), 2, LINE_AA);

}

// Function to draw the minimum area rectangle around given points
void drawMinAreaRect(Mat &img, const vector<Point> &points) {
    RotatedRect box = minAreaRect(points);  // Find the minimum area rectangle
    Point2f vtx[4];
    box.points(vtx);
    // Convert Point2f to Point because polylines expects a vector of Point
    vector<Point> rectPoints;
    for (int i = 0; i < 4; i++) {
        rectPoints.push_back(Point(cvRound(vtx[i].x), cvRound(vtx[i].y)));
    }

    // Use polylines to draw the rectangle. The 'true' argument closes the loop, drawing a rectangle.
    polylines(img, rectPoints, true, Scalar(0, 255, 0), 2, LINE_AA);
}

// Function to draw the convex hull of given points
void drawConvexHull(Mat &img, const vector<Point> &points) {
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
void drawFittedEllipses(Mat &img, const vector<Point> &points, int ellipseMethod) {
    switch (ellipseMethod) {
        case OpenCV_Method: // Standard ellipse fitting
            {
                RotatedRect fittedEllipse = fitEllipse(points);
                if (isGoodBox(fittedEllipse)) {
                    ellipse(img, fittedEllipse, Scalar(255, 0, 255), 2, LINE_AA);
                }
                putText(img, "OpenCV", Point(img.cols - 80, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2, LINE_AA);
            }
            break;
        case AMS_Method: // AMS ellipse fitting
            {
                RotatedRect fittedEllipseAMS = fitEllipseAMS(points);
                if (isGoodBox(fittedEllipseAMS)) {
                    ellipse(img, fittedEllipseAMS, Scalar(255, 0, 255), 2, LINE_AA);
                }
                putText(img, "AMS", Point(img.cols - 80, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 2, LINE_AA);
            }
            break;
        case Direct_Method: // Direct ellipse fitting
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
void drawShapes(Mat &img, const vector<Point> &points, int ellipseMethod, string shape) {
    if (shape == "--circle") {
        drawMinEnclosingCircle(img, points);
    } else if (shape == "--triangle") {
        drawMinEnclosingTriangle(img, points);
    } else if (shape == "--rectangle") {
        drawMinAreaRect(img, points);
    } else if (shape == "--convexhull") {
        drawConvexHull(img, points);
    } else if (shape == "--ellipse"){
        drawFittedEllipses(img, points, ellipseMethod);
    }
    else if (shape == "--all") {
        drawMinEnclosingCircle(img, points);
        drawMinEnclosingTriangle(img, points);
        drawMinAreaRect(img, points);
        drawConvexHull(img, points);
        drawFittedEllipses(img, points, ellipseMethod);
    }
}

// Main function to process the image based on the current trackbar position
void processImage(int position, void* userData){

    UserData* data = static_cast<UserData*>(userData);

    data->sliderPos = position;

    Mat processedImg = image.clone();  // Clone the original image for processing
    Mat gray;
    cvtColor(processedImg, gray, COLOR_BGR2GRAY);  // Convert to grayscale
    threshold(gray, gray, data->sliderPos, 255, THRESH_BINARY);  // Apply binary threshold

    Mat filteredImg;
    medianBlur(gray, filteredImg, 3);

    vector<vector<Point>> contours;
    findContours(filteredImg, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);  // Find contours

    if (contours.empty()) {
        return;
    }

    imshow("Mask", filteredImg);  // Show the mask
    for (size_t i = 0; i < contours.size(); ++i) {
        if (contours[i].size() < 5) {  // Check if the contour has enough points
            continue;
        }
        drawShapes(processedImg, contours[i], data->ellipseMethod, data->shape);
    }

    imshow("Shapes", processedImg);  // Display the processed image with fitted shapes
}
