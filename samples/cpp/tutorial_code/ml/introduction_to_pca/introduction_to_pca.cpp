/**
 * @file introduction_to_pca.cpp
 * @brief This program demonstrates how to use OpenCV PCA to extract the orienation of an object
 * @author OpenCV team
 */

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace std;
using namespace cv;

// Function declarations
void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);

/**
 * @function drawAxis
 */
void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
//! [visualization1]
    double angle;
    double hypotenuse;
    angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
//    double degrees = angle * 180 / CV_PI; // convert radians to degrees (0-180 range)
//    cout << "Degrees: " << abs(degrees - 180) << endl; // angle in 0-360 degrees range

    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, LINE_AA);

    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);

    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
//! [visualization1]
}

/**
 * @function getOrientation
 */
double getOrientation(const vector<Point> &pts, Mat &img)
{
//! [pca]
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64FC1);
    for (int i = 0; i < data_pts.rows; ++i)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);

    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                      static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; ++i)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));

        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }

//! [pca]
//! [visualization]
    // Draw the principal components
    circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
    drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);

    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
//! [visualization]

    return angle;
}

/**
 * @function main
 */
int main(int argc, char** argv)
{
//! [pre-process]
    // Load image
    CommandLineParser parser(argc, argv, "{@input | ../data/pca_test1.jpg | input image}");
    parser.about( "This program demonstrates how to use OpenCV PCA to extract the orienation of an object.\n" );
    parser.printMessage();

    Mat src = imread(parser.get<String>("@input"));

    // Check if image is loaded successfully
    if(src.empty())
    {
        cout << "Problem loading image!!!" << endl;
        return EXIT_FAILURE;
    }

    imshow("src", src);

    // Convert image to grayscale
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Convert image to binary
    Mat bw;
    threshold(gray, bw, 50, 255, THRESH_BINARY | THRESH_OTSU);
//! [pre-process]

//! [contours]
    // Find all the contours in the thresholded image
    vector<vector<Point> > contours;
    findContours(bw, contours, RETR_LIST, CHAIN_APPROX_NONE);

    for (size_t i = 0; i < contours.size(); ++i)
    {
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // Ignore contours that are too small or too large
        if (area < 1e2 || 1e5 < area) continue;

        // Draw each contour only for visualisation purposes
        drawContours(src, contours, static_cast<int>(i), Scalar(0, 0, 255), 2, LINE_8);
        // Find the orientation of each shape
        getOrientation(contours[i], src);
    }
//! [contours]

    imshow("output", src);

    waitKey(0);
    return 0;
}
