#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

void drawRandomLines(Mat& image, RNG& rng, int width, int height, int lineType);
void drawRandomRectangles(Mat& image, RNG& rng, int width, int height, int lineType);
void drawRandomEllipses(Mat& image, RNG& rng, int width, int height, int lineType);
void drawRandomPolylines(Mat& image, RNG& rng, int width, int height, int lineType);
void drawRandomFilledPolygons(Mat& image, RNG& rng, int width, int height, int lineType);
void drawRandomCircles(Mat& image, RNG& rng, int width, int height, int lineType);
void drawRandomText(Mat& image, RNG& rng, int width, int height, int lineType);
void drawAll(Mat& image, RNG& rng, int width, int height, int lineType);

// Display help message to the user
static void help() {
    cout << "\nThis program demonstrates OpenCV drawing and text output functions by drawing random shapes and texts\n"
            "You can change the drawing mode by pressing keys while the program is running:\n"
            "   'l' : lines\n"
            "   'r' : rectangles\n"
            "   'e' : ellipses\n"
            "   'p' : polylines\n"
            "   'f' : filled polygons\n"
            "   'c' : circles\n"
            "   't' : text\n"
            "   'a' : all shapes\n"
            "Press 'ESC' to exit the program.\n";
}

static Scalar randomColor(RNG& rng) {
    int icolor = (unsigned)rng;
    return Scalar(icolor&255, (icolor>>8)&255, (icolor>>16)&255);
}

int main(int argc, char** argv) {
    if (argc == 2) {
        string arg = argv[1];
        if (arg == "-h" || arg == "--help") {
            help();
            return 0;
        }
    }
    help();

    // Initialize random number generator, image dimensions, and type
    RNG rng(0xFFFFFFFF);
    int width = 1000, height = 700;
    int lineType = LINE_AA;
    Mat image = Mat::zeros(height, width, CV_8UC3);

    // Draw all shapes by default
    drawAll(image, rng, width, height, lineType);
    imshow("Drawing Demo", image);
    int key = waitKey(0);

    while (key != 27) { // 27 is the ASCII code for 'ESC'
        switch (key) {
            case 'l':
                image = Mat::zeros(height, width, CV_8UC3);
                drawRandomLines(image, rng, width, height, lineType);
                break;
            case 'r':
                image = Mat::zeros(height, width, CV_8UC3);
                drawRandomRectangles(image, rng, width, height, lineType);
                break;
            case 'e':
                image = Mat::zeros(height, width, CV_8UC3);
                drawRandomEllipses(image, rng, width, height, lineType);
                break;
            case 'p':
                image = Mat::zeros(height, width, CV_8UC3);
                drawRandomPolylines(image, rng, width, height, lineType);
                break;
            case 'f':
                image = Mat::zeros(height, width, CV_8UC3);
                drawRandomFilledPolygons(image, rng, width, height, lineType);
                break;
            case 'c':
                image = Mat::zeros(height, width, CV_8UC3);
                drawRandomCircles(image, rng, width, height, lineType);
                break;
            case 't':
                image = Mat::zeros(height, width, CV_8UC3);
                drawRandomText(image, rng, width, height, lineType);
                break;
            case 'a':
                image = Mat::zeros(height, width, CV_8UC3);
                drawAll(image, rng, width, height, lineType);
                break;
        }
        imshow("Drawing Demo", image);
        key = waitKey(0);
    }
    return 0;
}

// Function implementations to draw each type of shape
// Each function follows a similar pattern
void drawRandomLines(Mat& image, RNG& rng, int width, int height, int lineType) {
    Point pt1, pt2;
    // Draw 100 random lines
    for (int i = 0; i < 100; ++i) {
        pt1.x = rng.uniform(0, width);
        pt1.y = rng.uniform(0, height);
        pt2.x = rng.uniform(0, width);
        pt2.y = rng.uniform(0, height);
        // Draw a line between the random points with a random color and thickness
        line(image, pt1, pt2, randomColor(rng), rng.uniform(1, 10), lineType);
    }
}

void drawRandomRectangles(Mat& image, RNG& rng, int width, int height, int lineType) {
    Point pt1, pt2;
    // Generate random corner points
    for (int i = 0; i < 100; ++i) {
        pt1.x = rng.uniform(0, width);
        pt1.y = rng.uniform(0, height);
        pt2.x = rng.uniform(0, width);
        pt2.y = rng.uniform(0, height);
        // Draw a rectangle with random colors and thickness (or filled if thickness is -1). MAX changes any negative number by rng.uniform to -1
        rectangle(image, pt1, pt2, randomColor(rng), MAX(rng.uniform(-1, 10), -1), lineType);
    }
}

void drawRandomEllipses(Mat& image, RNG& rng, int width, int height, int lineType) {
    Point center;
    Size axes;
    for (int i = 0; i < 50; ++i) {
        center.x = rng.uniform(0, width);
        center.y = rng.uniform(0, height);
        axes.width = rng.uniform(0, 200);
        axes.height = rng.uniform(0, 200);
        double angle = rng.uniform(0.0, 360.0);
        // Draw an ellipse with a random color and thickness
        ellipse(image, center, axes, angle, 0, 360, randomColor(rng), rng.uniform(-1, 9), lineType);
    }
}

void drawRandomPolylines(Mat& image, RNG& rng, int width, int height, int lineType) {
    for (int i = 0; i < 10; ++i) {
        Point points[1][5];
        points[0][0] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][1] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][2] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][3] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][4] = Point(rng.uniform(0, width), rng.uniform(0, height));

        const Point* ppt[1] = {points[0]};
        int npt[] = {5};

        polylines(image, ppt, npt, 1, true, randomColor(rng), rng.uniform(1, 10), lineType);
    }
}

void drawRandomFilledPolygons(Mat& image, RNG& rng, int width, int height, int lineType) {
    for (int i = 0; i < 10; ++i) {
        Point points[1][5];
        // Generate 5 random points for each polyline
        points[0][0] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][1] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][2] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][3] = Point(rng.uniform(0, width), rng.uniform(0, height));
        points[0][4] = Point(rng.uniform(0, width), rng.uniform(0, height));

        const Point* ppt[1] = {points[0]};
        int npt[] = {5};
        // Draw the polyline with a random color and thickness
        fillPoly(image, ppt, npt, 1, randomColor(rng), lineType);
    }
}

void drawRandomCircles(Mat& image, RNG& rng, int width, int height, int lineType) {
    Point center;
    for (int i = 0; i < 100; ++i) {
        // Generate a random center and radius
        center.x = rng.uniform(0, width);
        center.y = rng.uniform(0, height);
        circle(image, center, rng.uniform(0, 300), randomColor(rng), rng.uniform(-1, 9), lineType);
    }
}

void drawRandomText(Mat& image, RNG& rng, int width, int height, int lineType) {
    Point org;
    for (int i = 0; i < 50; ++i) {
        // Generate a random position for the text
        org.x = rng.uniform(0, width);
        org.y = rng.uniform(0, height);
        // Randomize font face, scale, and thickness
        int fontFace = rng.uniform(0, 3);
        double fontScale = rng.uniform(0.5, 2.0);
        int thickness = rng.uniform(1, 3);
        putText(image, "This is OpenCV drawing demo sample", org, fontFace, fontScale, randomColor(rng), thickness, lineType);
    }
}

void drawAll(Mat& image, RNG& rng, int width, int height, int lineType) {
    drawRandomLines(image, rng, width, height, lineType);
    drawRandomRectangles(image, rng, width, height, lineType);
    drawRandomEllipses(image, rng, width, height, lineType);
    drawRandomPolylines(image, rng, width, height, lineType);
    drawRandomFilledPolygons(image, rng, width, height, lineType);
    drawRandomCircles(image, rng, width, height, lineType);
    drawRandomText(image, rng, width, height, lineType);
}