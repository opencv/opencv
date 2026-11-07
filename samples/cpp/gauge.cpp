/**
  @file gauge.cpp
  @author Alessandro de Oliveira Faria (A.K.A. CABELO)
  @brief This sample application processes an image frame of an analog gauge
  and extracts its reading using functions from the OpenCV* computer vision
  library. The workflow is divided into two stages: calibration and
  measurement. Questions and suggestions email to:
  Alessandro de Oliveira Faria cabelo[at]opensuse[dot]org or OpenCV Team.
  @date Nov 26, 2025
*/

#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

struct GaugeCalibration {
    double min_angle;
    double max_angle;
    double min_value;
    double max_value;
    string units;
    int x;
    int y;
    int r;
};

static Vec3f avg_circles(const vector<Vec3f> &circles) {
    double avg_x = 0.0;
    double avg_y = 0.0;
    double avg_r = 0.0;

    int b = static_cast<int>(circles.size());
    for (int i = 0; i < b; ++i) {
        avg_x += circles[i][0];
        avg_y += circles[i][1];
        avg_r += circles[i][2];
    }

    avg_x /= b;
    avg_y /= b;
    avg_r /= b;

    return Vec3f(static_cast<float>(avg_x),
                 static_cast<float>(avg_y),
                 static_cast<float>(avg_r));
}

static double dist_2_pts(double x1, double y1, double x2, double y2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

static GaugeCalibration calibrate_gauge(string filename) {
    GaugeCalibration cal;
    Mat img = imread(filename);
    if (img.empty()) {
        cerr << "Error loading image: " << filename << endl;
        exit(1);
    }

    int height = img.rows;

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Detects circles (HoughCircles)
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 20,
                 100, 50,
                 static_cast<int>(height * 0.35),
                 static_cast<int>(height * 0.48));

    if (circles.empty()) {
        cerr << "No circles found." << endl;
        exit(1);
    }

    Vec3f avg = avg_circles(circles);
    int x = static_cast<int>(avg[0]);
    int y = static_cast<int>(avg[1]);
    int r = static_cast<int>(avg[2]);

    cal.x = x;
    cal.y = y;
    cal.r = r;

    // Draw the circle and the center.
    circle(img, Point(x, y), r, Scalar(0, 0, 255), 3, LINE_AA);
    circle(img, Point(x, y), 2, Scalar(0, 255, 0), 3, LINE_AA);

    // Generation of calibration lines.
    double separation = 10.0; // in degrees
    int interval = static_cast<int>(360.0 / separation);

    vector<Point2d> p1(interval);
    vector<Point2d> p2(interval);
    vector<Point2d> p_text(interval);

    for (int i = 0; i < interval; ++i) {
        double angle_rad = separation * i * CV_PI / 180.0;
        p1[i].x = x + 0.9 * r * std::cos(angle_rad);
        p1[i].y = y + 0.9 * r * std::sin(angle_rad);
    }

    int text_offset_x = 10;
    int text_offset_y = 5;

    for (int i = 0; i < interval; ++i) {
        double angle_rad = separation * i * CV_PI / 180.0;
        p2[i].x = x + r * std::cos(angle_rad);
        p2[i].y = y + r * std::sin(angle_rad);

        double text_angle_rad = separation * (i + 9) * CV_PI / 180.0; // i+9 = rotate 90°
        p_text[i].x = x - text_offset_x + 1.2 * r * std::cos(text_angle_rad);
        p_text[i].y = y + text_offset_y + 1.2 * r * std::sin(text_angle_rad);
    }

    // Draws lines and text.
    for (int i = 0; i < interval; ++i) {
        line(img,
             Point(static_cast<int>(p1[i].x), static_cast<int>(p1[i].y)),
             Point(static_cast<int>(p2[i].x), static_cast<int>(p2[i].y)),
             Scalar(0, 255, 0), 2);

        putText(img,
                to_string(static_cast<int>(i * separation)),
                Point(static_cast<int>(p_text[i].x), static_cast<int>(p_text[i].y)),
                FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0), 1, LINE_AA);
    }

    // Save calibration image
    imwrite("gauge-calibration.jpg", img);

    cout << "Min angle (lowest possible angle of dial) - in degrees: ";
    cin >> cal.min_angle;

    cout << "Max angle (highest possible angle) - in degrees: ";
    cin >> cal.max_angle;

    cout << "Min value: ";
    cin >> cal.min_value;

    cout << "Max value: ";
    cin >> cal.max_value;

    cout << "Enter units: ";
    cin >> cal.units;

    return cal;
}

static double get_current_value(Mat img,
                         const GaugeCalibration &cal) {
    Mat gray2;
    cvtColor(img, gray2, COLOR_BGR2GRAY);

    int thresh = 175;
    int maxValue = 255;

    Mat dst2;
    threshold(gray2, dst2, thresh, maxValue, THRESH_BINARY_INV);

    // For debugging: threshold image
    //imwrite("gauge-tempdst2.jpg", dst2);

    // Detect lines
    vector<Vec4i> lines;
    int minLineLength = 10;
    int maxLineGap = 0;
    HoughLinesP(dst2, lines, 3, CV_PI / 180, 100, minLineLength, maxLineGap);

    if (lines.empty()) {
        cerr << "No rows found." << endl;
        return 0.0;
    }

    // Filter lines by distance from center
    vector<Vec4i> final_line_list;

    double diff1LowerBound = 0.15;
    double diff1UpperBound = 0.25;
    double diff2LowerBound = 0.5;
    double diff2UpperBound = 1.0;

    int x = cal.x;
    int y = cal.y;
    int r = cal.r;

    for (size_t i = 0; i < lines.size(); ++i) {
        int x1 = lines[i][0];
        int y1 = lines[i][1];
        int x2 = lines[i][2];
        int y2 = lines[i][3];

        double diff1 = dist_2_pts(x, y, x1, y1);
        double diff2 = dist_2_pts(x, y, x2, y2);

        // ensures that diff1 is the smallest (closest to the center)
        if (diff1 > diff2) {
            std::swap(diff1, diff2);
        }

        if ((diff1 < diff1UpperBound * r) && (diff1 > diff1LowerBound * r) &&
            (diff2 < diff2UpperBound * r) && (diff2 > diff2LowerBound * r)) {
            final_line_list.push_back(lines[i]);
        }
    }

    if (final_line_list.empty()) {
        cerr << "No lines within the expected radius." << endl;
        return 0.0;
    }

    // Use the first filtered line.
    int x1 = final_line_list[0][0];
    int y1 = final_line_list[0][1];
    int x2 = final_line_list[0][2];
    int y2 = final_line_list[0][3];

    // Draw the line on the original image for debugging.
    line(img, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
    //imwrite("gauge-lines-2.jpg", img);

    // Decide which point is furthest from the center.
    double dist_pt_0 = dist_2_pts(x, y, x1, y1);
    double dist_pt_1 = dist_2_pts(x, y, x2, y2);

    double x_angle, y_angle;
    if (dist_pt_0 > dist_pt_1) {
        x_angle = x1 - x;
        y_angle = y - y1;
    } else {
        x_angle = x2 - x;
        y_angle = y - y2;
    }

    double res = 0;
    // atan(y/x) in radians
    if(x_angle != 0)
    {
        res = std::atan(y_angle / x_angle);
        res = res * 180.0 / CV_PI; // rad2deg
    }
    double final_angle = 0.0;

    if (x_angle > 0 && y_angle > 0) { // Quadrante I
        final_angle = 270.0 - res;
    }
    if (x_angle < 0 && y_angle > 0) { // Quadrante II
        final_angle = 90.0 - res;
    }
    if (x_angle < 0 && y_angle < 0) { // Quadrante III
        final_angle = 90.0 - res;
    }
    if (x_angle > 0 && y_angle < 0) { // Quadrante IV
        final_angle = 270.0 - res;
    }

    double old_min = cal.min_angle;
    double old_max = cal.max_angle;
    double new_min = cal.min_value;
    double new_max = cal.max_value;

    double old_value = final_angle;
    double old_range = (old_max - old_min);
    double new_range = (new_max - new_min);

    double new_value = (((old_value - old_min) * new_range) / old_range) + new_min;

    return new_value;
}

int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv,  "{@input   |gauge-1.jpg|Input image to reads the value using functions from the OpenCV. }");
    parser.about("Analog-gauge-reader example:\n");
    parser.printMessage();
    string filename = parser.get<String>("@input");

    Mat img = imread(filename);
    if (img.empty()) {
        cerr << "Error loading image. " << filename << endl;
        return 1;
    }

    // Calibration
    GaugeCalibration cal = calibrate_gauge(filename);

    double val = get_current_value(img, cal );
    cout << "Current reading: " << val << " " << cal.units << endl;

    return 0;
}
