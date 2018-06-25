// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef QR_CODE_HPP_
#define QR_CODE_HPP_

using namespace std;
using namespace cv;

CV_EXPORTS void DetectionQRCode(InputArray in, OutputArray points,
                                OutputArray out, String decode);

class QRDecode
{
 public:
    void init(Mat src,
              double eps_vertical_   = 0.19,
              double eps_horizontal_ = 0.09);
    void binarization();
    bool localization();
    bool transformation();
    Mat getBinBarcode() { return bin_barcode; }
    Mat getLocalizationBarcode() { return local_barcode; }
    Mat getTransformationBarcode() { return transform_barcode; }
    vector<Point> getTransformationPoints() { return transformation_points; }
    Mat getStraightBarcode() { return straight_barcode; }
 protected:
    vector<Vec3d> searchVerticalLines();
    vector<Vec3d> separateHorizontalLines(vector<Vec3d> list_lines);
    vector<Vec3d> pointClustering(vector<Vec3d> list_lines);
    void fixationPoints(vector<Point> &local_point, vector<double> &local_len);
    Point getTransformationPoint(Point left, Point center, double cos_angle_rotation,
                                 bool right_rotate = true);
    Point intersectionLines(Point a1, Point a2, Point b1, Point b2);
    vector<Point> getQuadrilateral(vector<Point> angle_list);
    double getQuadrilateralArea(Point a, Point b, Point c, Point d);
    double getCosVectors(Point a, Point b, Point c);

    Mat barcode, bin_barcode, local_barcode, transform_barcode, straight_barcode;
    vector<Point>  localization_points, transformation_points;
    vector<double> localization_length;
    double experimental_area;

    double eps_vertical, eps_horizontal;
    vector<Vec3d> result;
    vector<double> test_lines;
    uint8_t next_pixel, future_pixel;
    double length, weight, show_radius;

};

#endif  // QR_CODE_HPP_
