// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "opencv2/objdetect.hpp"
// #include "opencv2/calib3d.hpp"

#include <limits>
#include <cmath>
#include <iostream>

namespace cv
{
class QRDecode
{
 public:
    void init(Mat src, double eps_vertical_ = 0.19, double eps_horizontal_ = 0.09);
    void binarization();
    bool localization();
    bool transformation();
    Mat getBinBarcode() { return bin_barcode; }
    Mat getLocalizationBarcode() { return local_barcode; }
    Mat getTransformationBarcode() { return transform_barcode; }
    std::vector<Point> getTransformationPoints() { return transformation_points; }
    Mat getStraightBarcode() { return straight_barcode; }
 protected:
    std::vector<Vec3d> searchVerticalLines();
    std::vector<Vec3d> separateHorizontalLines(std::vector<Vec3d> list_lines);
    std::vector<Vec3d> pointClustering(std::vector<Vec3d> list_lines);
    void fixationPoints(std::vector<Point> &local_point, std::vector<double> &local_len);
    Point getTransformationPoint(Point left, Point center, double cos_angle_rotation,
                                 bool right_rotate = true);
    Point intersectionLines(Point a1, Point a2, Point b1, Point b2);
    std::vector<Point> getQuadrilateral(std::vector<Point> angle_list);
    double getQuadrilateralArea(Point a, Point b, Point c, Point d);
    double getCosVectors(Point a, Point b, Point c);

    Mat barcode, bin_barcode, local_barcode, transform_barcode, straight_barcode;
    std::vector<Point>  localization_points, transformation_points;
    std::vector<double> localization_length;
    double experimental_area;

    double eps_vertical, eps_horizontal;
    std::vector<Vec3d> result;
    std::vector<double> test_lines;
    uint8_t next_pixel, future_pixel;
    double length, weight;
};

void QRDecode::init(Mat src, double eps_vertical_, double eps_horizontal_)
{
    barcode = src;
    eps_vertical   = eps_vertical_;
    eps_horizontal = eps_horizontal_;
}

void QRDecode::binarization()
{
    Mat filter_barcode;
    GaussianBlur(barcode, filter_barcode, Size(3, 3), 0);
    threshold(filter_barcode, bin_barcode, 0, 255, THRESH_BINARY + THRESH_OTSU);
}

bool QRDecode::localization()
{
    cvtColor(bin_barcode, local_barcode, COLOR_GRAY2RGB);
    Point begin, end;

    std::vector<Vec3d> list_lines_x = searchVerticalLines();
    std::vector<Vec3d> list_lines_y = separateHorizontalLines(list_lines_x);
    std::vector<Vec3d> result_point = pointClustering(list_lines_y);
    for (int i = 0; i < 3; i++)
    {
        localization_points.push_back(
            Point(static_cast<int>(result_point[i][0]),
                  static_cast<int>(result_point[i][1] + result_point[i][2])));
        localization_length.push_back(result_point[i][2]);
    }

    fixationPoints(localization_points, localization_length);


    if (localization_points.size() != 3) { return false; }
    return true;

}

std::vector<Vec3d> QRDecode::searchVerticalLines()
{
    result.clear();
    int temp_length = 0;

    for (int x = 0; x < bin_barcode.rows; x++)
    {
        for (int y = 0; y < bin_barcode.cols; y++)
        {
            if (bin_barcode.at<uint8_t>(x, y) > 0) { continue; }

            // --------------- Search vertical lines --------------- //

            test_lines.clear();
            future_pixel = 255;

            for (int i = x; i < bin_barcode.rows - 1; i++)
            {
                next_pixel = bin_barcode.at<uint8_t>(i + 1, y);
                temp_length++;
                if (next_pixel == future_pixel)
                {
                    future_pixel = 255 - future_pixel;
                    test_lines.push_back(temp_length);
                    temp_length = 0;
                    if (test_lines.size() == 5) { break; }
                }
            }

            // --------------- Compute vertical lines --------------- //

            if (test_lines.size() == 5)
            {
                length = 0.0; weight = 0.0;

                for (size_t i = 0; i < test_lines.size(); i++) { length += test_lines[i]; }

                for (size_t i = 0; i < test_lines.size(); i++)
                {
                    if (i == 2) { weight += abs((test_lines[i] / length) - 3.0/7.0); }
                    else        { weight += abs((test_lines[i] / length) - 1.0/7.0); }
                }

                if (weight < eps_vertical)
                {
                    Vec3d line;
                    line[0] = x; line[1] = y, line[2] = length;
                    result.push_back(line);
                }
            }
        }
    }
    return result;
}

std::vector<Vec3d> QRDecode::separateHorizontalLines(std::vector<Vec3d> list_lines)
{
    result.clear();
    int temp_length = 0;
    int x, y;

    for (size_t pnt = 0; pnt < list_lines.size(); pnt++)
    {
        x = static_cast<int>(list_lines[pnt][0] + list_lines[pnt][2] / 2);
        y = static_cast<int>(list_lines[pnt][1]);

        // --------------- Search horizontal up-lines --------------- //
        test_lines.clear();
        future_pixel = 255;

        for (int j = y; j < bin_barcode.cols - 1; j++)
        {
            next_pixel = bin_barcode.at<uint8_t>(x, j + 1);
            temp_length++;
            if (next_pixel == future_pixel)
            {
                future_pixel = 255 - future_pixel;
                test_lines.push_back(temp_length);
                temp_length = 0;
                if (test_lines.size() == 3) { break; }
            }
        }

        // --------------- Search horizontal down-lines --------------- //
        future_pixel = 255;

        for (int j = y; j >= 1; j--)
        {
            next_pixel = bin_barcode.at<uint8_t>(x, j - 1);
            temp_length++;
            if (next_pixel == future_pixel)
            {
                future_pixel = 255 - future_pixel;
                test_lines.push_back(temp_length);
                temp_length = 0;
                if (test_lines.size() == 6) { break; }
            }
        }

        // --------------- Compute horizontal lines --------------- //

        if (test_lines.size() == 6)
        {
            length = 0.0; weight = 0.0;

            for (size_t i = 0; i < test_lines.size(); i++) { length += test_lines[i]; }

            for (size_t i = 0; i < test_lines.size(); i++)
            {
                if (i % 3 == 0) { weight += abs((test_lines[i] / length) - 3.0/14.0); }
                else            { weight += abs((test_lines[i] / length) - 1.0/ 7.0); }
            }
        }

        if(weight < eps_horizontal)
        {
            result.push_back(list_lines[pnt]);
        }
    }
    return result;
}

std::vector<Vec3d> QRDecode::pointClustering(std::vector<Vec3d> list_lines)
{
    std::vector<Vec3d> centers;
    std::vector<Point> clusters[3];
    double weight_clusters[3] = {0.0, 0.0, 0.0};
    Point basis[3], temp_pnt;
    double temp_norm = 0.0, temp_compute_norm, distance[3];

    basis[0] = Point(static_cast<int>(list_lines[0][1]), static_cast<int>(list_lines[0][0]));
    for (size_t i = 1; i < list_lines.size(); i++)
    {
        temp_pnt = Point(static_cast<int>(list_lines[i][1]), static_cast<int>(list_lines[i][0]));
        temp_compute_norm = norm(basis[0] - temp_pnt);
        if (temp_norm < temp_compute_norm)
        {
            basis[1] = temp_pnt;
            temp_norm = temp_compute_norm;
        }
    }

    for (size_t i = 1; i < list_lines.size(); i++)
    {
        temp_pnt = Point(static_cast<int>(list_lines[i][1]), static_cast<int>(list_lines[i][0]));
        temp_compute_norm = norm(basis[0] - temp_pnt) + norm(basis[1] - temp_pnt);
        if (temp_norm < temp_compute_norm)
        {
            basis[2] = temp_pnt;
            temp_norm = temp_compute_norm;
        }
    }

    for (size_t i = 0; i < list_lines.size(); i++)
    {
        temp_pnt = Point(static_cast<int>(list_lines[i][1]), static_cast<int>(list_lines[i][0]));
        distance[0] = norm(basis[0] - temp_pnt);
        distance[1] = norm(basis[1] - temp_pnt);
        distance[2] = norm(basis[2] - temp_pnt);
        if (distance[0] < distance[1] && distance[0] < distance[2])
        {
            clusters[0].push_back(temp_pnt);
            weight_clusters[0] += list_lines[i][2];
        }
        else if (distance[1] < distance[0] && distance[1] < distance[2])
        {
            clusters[1].push_back(temp_pnt);
            weight_clusters[1] += list_lines[i][2];
        }
        else
        {
            clusters[2].push_back(temp_pnt);
            weight_clusters[2] += list_lines[i][2];
        }
    }

    for (int i = 0; i < 3; i++)
    {
        basis[i] = Point(0, 0);
        for (size_t j = 0; j < clusters[i].size(); j++) { basis[i] += clusters[i][j]; }
        basis[i] = basis[i] / static_cast<int>(clusters[i].size());
        weight = weight_clusters[i] / (2 * clusters[i].size());
        centers.push_back(Vec3d(basis[i].x, basis[i].y, weight));
    }

    return centers;
}

void QRDecode::fixationPoints(std::vector<Point> &local_point, std::vector<double> &local_len)
{
    double cos_angles[3], norm_triangl[3];

    norm_triangl[0] = norm(local_point[1] - local_point[2]);
    norm_triangl[1] = norm(local_point[0] - local_point[2]);
    norm_triangl[2] = norm(local_point[1] - local_point[0]);

    cos_angles[0] = (pow(norm_triangl[1], 2) + pow(norm_triangl[2], 2) - pow(norm_triangl[0], 2))
                  / (2 * norm_triangl[1] * norm_triangl[2]);
    cos_angles[1] = (pow(norm_triangl[0], 2) + pow(norm_triangl[2], 2) - pow(norm_triangl[1], 2))
                  / (2 * norm_triangl[0] * norm_triangl[2]);
    cos_angles[2] = (pow(norm_triangl[0], 2) + pow(norm_triangl[1], 2) - pow(norm_triangl[2], 2))
                  / (2 * norm_triangl[0] * norm_triangl[1]);

    int i_min_cos =
      (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0 :
      (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1 : 2;

    Point temp_pnt;
    double tmp_len;
    temp_pnt = local_point[0];
    tmp_len = local_len[0];
    local_point[0] = local_point[i_min_cos];
    local_len[0] = local_len[i_min_cos];
    local_point[i_min_cos] = temp_pnt;
    local_len[i_min_cos] = tmp_len;

    Mat vector_mult(Size(3, 3), CV_32FC1);
    vector_mult.at<float>(0, 0) = 1;
    vector_mult.at<float>(1, 0) = 1;
    vector_mult.at<float>(2, 0) = 1;
    vector_mult.at<float>(0, 1) = static_cast<float>((local_point[1] - local_point[0]).x);
    vector_mult.at<float>(1, 1) = static_cast<float>((local_point[1] - local_point[0]).y);
    vector_mult.at<float>(0, 2) = static_cast<float>((local_point[2] - local_point[0]).x);
    vector_mult.at<float>(1, 2) = static_cast<float>((local_point[2] - local_point[0]).y);
    double res_vect_mult = determinant(vector_mult);
    if (res_vect_mult < 0)
    {
        temp_pnt = local_point[1];
        tmp_len = local_len[1];
        local_point[1] = local_point[2];
        local_len[1] = local_len[2];
        local_point[2] = temp_pnt;
        local_len[2] = tmp_len;
    }
}

bool QRDecode::transformation()
{
    cvtColor(bin_barcode, transform_barcode, COLOR_GRAY2RGB);
    if (localization_points.size() != 3) { return false; }

    Point red   = localization_points[0];
    Point green = localization_points[1];
    Point blue  = localization_points[2];
    Point adj_b_r_pnt,  adj_r_b_pnt, adj_g_r_pnt, adj_r_g_pnt;
    Point line_r_b_pnt, line_r_g_pnt, norm_r_b_pnt, norm_r_g_pnt;
    adj_b_r_pnt  = getTransformationPoint(blue, red, -1);
    adj_r_b_pnt  = getTransformationPoint(red, blue, -1);
    adj_g_r_pnt  = getTransformationPoint(green, red, -1);
    adj_r_g_pnt  = getTransformationPoint(red, green, -1);
    line_r_b_pnt = getTransformationPoint(red, blue,  -0.91);
    line_r_g_pnt = getTransformationPoint(red, green, -0.91);
    norm_r_b_pnt = getTransformationPoint(red, blue,  0.0, true);
    norm_r_g_pnt = getTransformationPoint(red, green, 0.0, false);

    transformation_points.push_back(intersectionLines(
        adj_r_g_pnt, line_r_g_pnt, adj_r_b_pnt, line_r_b_pnt));
    transformation_points.push_back(intersectionLines(
        adj_b_r_pnt, norm_r_g_pnt, adj_r_g_pnt, line_r_g_pnt));
    transformation_points.push_back(intersectionLines(
        norm_r_b_pnt, adj_g_r_pnt, adj_b_r_pnt, norm_r_g_pnt));
    transformation_points.push_back(intersectionLines(
        norm_r_b_pnt, adj_g_r_pnt, adj_r_b_pnt, line_r_b_pnt));

    experimental_area = getQuadrilateralArea(transformation_points[0],
                                             transformation_points[1],
                                             transformation_points[2],
                                             transformation_points[3]);
    std::vector<Point> quadrilateral = getQuadrilateral(transformation_points);
    transformation_points = quadrilateral;

    int max_length_norm = -1;
    size_t transform_size = transformation_points.size();
    for (size_t i = 0; i < transform_size; i++)
    {
        int len_norm = static_cast<int>(norm(transformation_points[i % transform_size] -
                                             transformation_points[(i + 1) % transform_size]));
        if (max_length_norm < len_norm) { max_length_norm = len_norm; }
    }

    std::vector<Point> perspective_points;
    perspective_points.push_back(Point(0, 0));
    perspective_points.push_back(Point(0, max_length_norm));
    perspective_points.push_back(Point(max_length_norm, max_length_norm));
    perspective_points.push_back(Point(max_length_norm, 0));

    // warpPerspective(bin_barcode, straight_barcode,
    //                 findHomography(transformation_points, perspective_points),
    //                 Size(max_length_norm, max_length_norm));
    return true;
}

Point QRDecode::getTransformationPoint(Point left, Point center, double cos_angle_rotation,
                                       bool right_rotate)
{
    Point temp_pnt, prev_pnt(0, 0), next_pnt, start_pnt(center);
    double temp_delta, min_delta;
    int steps = 0;

    future_pixel = 255;
    while(true)
    {
        min_delta = std::numeric_limits<double>::max();
        for (int i = -1; i < 2; i++)
        {
            for (int j = -1; j < 2; j++)
            {
                if (i == 0 && j == 0) { continue; }
                temp_pnt = Point(start_pnt.x + i, start_pnt.y + j);
                temp_delta = abs(getCosVectors(left, center, temp_pnt) - cos_angle_rotation);
                if (temp_delta < min_delta && prev_pnt != temp_pnt)
                {
                    next_pnt = temp_pnt;
                    min_delta  = temp_delta;
                }
            }
        }
        prev_pnt = start_pnt;
        start_pnt = next_pnt;
        next_pixel = bin_barcode.at<uint8_t>(start_pnt.y, start_pnt.x);
        if (next_pixel == future_pixel)
        {
            future_pixel = 255 - future_pixel;
            steps++;
            if (steps == 3) { break; }
        }
    }

    if (cos_angle_rotation == 0.0)
    {
        Mat vector_mult(Size(3, 3), CV_32FC1);
        vector_mult.at<float>(0, 0) = 1;
        vector_mult.at<float>(1, 0) = 1;
        vector_mult.at<float>(2, 0) = 1;
        vector_mult.at<float>(0, 1) = static_cast<float>((left - center).x);
        vector_mult.at<float>(1, 1) = static_cast<float>((left - center).y);
        vector_mult.at<float>(0, 2) = static_cast<float>((left - start_pnt).x);
        vector_mult.at<float>(1, 2) = static_cast<float>((left - start_pnt).y);
        double res_vect_mult = determinant(vector_mult);
        if (( right_rotate && res_vect_mult < 0) ||
            (!right_rotate && res_vect_mult > 0))
        {
            start_pnt = getTransformationPoint(start_pnt, center, -1);
        }
    }

    return start_pnt;
}

Point QRDecode::intersectionLines(Point a1, Point a2, Point b1, Point b2)
{
    Point result_square_angle(
      static_cast<int>(
        static_cast<double>
        ((a1.x * a2.y  -  a1.y * a2.x) * (b1.x - b2.x) -
         (b1.x * b2.y  -  b1.y * b2.x) * (a1.x - a2.x)) /
        ((a1.x - a2.x) * (b1.y - b2.y) -
         (a1.y - a2.y) * (b1.x - b2.x))),
      static_cast<int>(
        static_cast<double>
        ((a1.x * a2.y  -  a1.y * a2.x) * (b1.y - b2.y) -
         (b1.x * b2.y  -  b1.y * b2.x) * (a1.y - a2.y)) /
        ((a1.x - a2.x) * (b1.y - b2.y) -
         (a1.y - a2.y) * (b1.x - b2.x)))
    );
    return result_square_angle;
}

std::vector<Point> QRDecode::getQuadrilateral(std::vector<Point> angle_list)
{
    size_t angle_size = angle_list.size();
    uint8_t value, mask_value;
    Mat mask(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
    for (size_t i = 0; i < angle_size; i++)
    {
        LineIterator line_iter(bin_barcode, angle_list[ i      % angle_size],
                                            angle_list[(i + 1) % angle_size]);
        for(int j = 0; j < line_iter.count; j++, ++line_iter)
        {
            value = bin_barcode.at<uint8_t>(line_iter.pos());
            mask_value = mask.at<uint8_t>(line_iter.pos() + Point(1, 1));
            if (value == 0 && mask_value == 0)
            {
                floodFill(bin_barcode, mask, line_iter.pos(), 255);
            }
        }
    }
    std::vector<Point> locations;
    Mat mask_roi = mask(Range(1, bin_barcode.rows - 1),
                        Range(1, bin_barcode.cols - 1));

    cv::findNonZero(mask_roi, locations);

    for (size_t i = 0; i < angle_list.size(); i++)
    {
        locations.push_back(angle_list[i]);
    }

    std::vector< std::vector<Point> > hull(1), approx_hull(1);
    convexHull(Mat(locations), hull[0]);
    int hull_size = static_cast<int>(hull[0].size());

    Point min_pnt;

    std::vector<Point> min_abc;
    double min_abs_cos_abc, abs_cos_abc;
    for (int count = 0; count < 4; count++)
    {
        min_abs_cos_abc = std::numeric_limits<double>::max();
        for (int i = 0; i < hull_size; i++)
        {
            Point a = hull[0][ i      % hull_size];
            Point b = hull[0][(i + 1) % hull_size];
            Point c = hull[0][(i + 2) % hull_size];
            abs_cos_abc = abs(getCosVectors(a, b, c));

            bool flag_detect = true;
            for (size_t j = 0; j < min_abc.size(); j++)
            {
                if (min_abc[j] == b) { flag_detect = false; break; }
            }

            if (flag_detect && (abs_cos_abc < min_abs_cos_abc))
            {
                min_pnt = b;
                min_abs_cos_abc = abs_cos_abc;
            }
        }
        min_abc.push_back(min_pnt);
    }


    int min_abc_size = static_cast<int>(min_abc.size());
    std::vector<int> index_min_abc(min_abc_size);
    for (int i = 0; i < min_abc_size; i++)
    {
        for (int j = 0; j < hull_size; j++)
        {
            if (hull[0][j] == min_abc[i]) { index_min_abc[i] = j; break; }
        }
    }

    std::vector<Point> result_hull_point(angle_size);
    double min_norm, temp_norm;
    for (size_t i = 0; i < angle_size; i++)
    {
        min_norm = std::numeric_limits<double>::max();
        Point closest_pnt;
        for (int j = 0; j < min_abc_size; j++)
        {
            if (min_norm > norm(hull[0][index_min_abc[j]] - angle_list[i]))
            {
                min_norm = norm(hull[0][index_min_abc[j]] - angle_list[i]);
                closest_pnt = hull[0][index_min_abc[j]];
            }
        }
        result_hull_point[i] = closest_pnt;
    }

    int start_line[2] = {0, 0}, finish_line[2] = {0, 0}, unstable_pnt = 0;
    for (int i = 0; i < hull_size; i++)
    {
        if (result_hull_point[3] == hull[0][i]) { start_line[0] = i; }
        if (result_hull_point[2] == hull[0][i]) { finish_line[0] = start_line[1] = i; }
        if (result_hull_point[1] == hull[0][i]) { finish_line[1] = i; }
        if (result_hull_point[0] == hull[0][i]) { unstable_pnt = i; }
    }

    int index_hull, extra_index_hull, next_index_hull, extra_next_index_hull, count_points;
    Point result_side_begin[4], result_side_end[4];

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    count_points = abs(start_line[0] - finish_line[0]);
    do
    {
        if (count_points > hull_size / 2) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[0][index_hull] - angle_list[2]) >
          norm(hull[0][index_hull] - angle_list[3]) ? angle_list[3] : angle_list[2];

        Point intrsc_line_hull =
          intersectionLines(hull[0][index_hull], hull[0][next_index_hull],
                            angle_list[2], angle_list[3]);
        temp_norm = getCosVectors(hull[0][index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[0][index_hull] - hull[0][next_index_hull]) >
            norm(angle_list[2] - angle_list[3]) / 10)
        {
            min_norm = temp_norm;
            result_side_begin[0] = hull[0][index_hull];
            result_side_end[0]   = hull[0][next_index_hull];
        }


        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[0]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[0] = angle_list[2];
        result_side_end[0]   = angle_list[3];
    }

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[1];
    count_points = abs(start_line[1] - finish_line[1]);
    do
    {
        if (count_points > hull_size / 2) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[0][index_hull] - angle_list[1]) >
          norm(hull[0][index_hull] - angle_list[2]) ? angle_list[2] : angle_list[1];

        Point intrsc_line_hull =
          intersectionLines(hull[0][index_hull], hull[0][next_index_hull],
                            angle_list[1], angle_list[2]);
        temp_norm = getCosVectors(hull[0][index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[0][index_hull] - hull[0][next_index_hull]) >
            norm(angle_list[1] - angle_list[2]) / 20)
        {
            min_norm = temp_norm;
            result_side_begin[1] = hull[0][index_hull];
            result_side_end[1]   = hull[0][next_index_hull];
        }


        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[1]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[1] = angle_list[1];
        result_side_end[1]   = angle_list[2];
    }

    double test_norm[4] = { 0.0, 0.0, 0.0, 0.0 };
    int test_index[4];
    for (int i = 0; i < 4; i++)
    {
        test_index[i] = (i < 2) ? static_cast<int>(start_line[0])
                                : static_cast<int>(finish_line[1]);
        do
        {
            next_index_hull = ((i + 1) % 2 != 0) ? test_index[i] + 1 : test_index[i] - 1;
            if (next_index_hull == hull_size) { next_index_hull = 0; }
            if (next_index_hull == -1) { next_index_hull = hull_size - 1; }
            test_norm[i] += norm(hull[0][next_index_hull] - hull[0][unstable_pnt]);
            test_index[i] = next_index_hull;
        }
        while(test_index[i] != unstable_pnt);
    }

    std::vector<Point> result_angle_list(4), test_result_angle_list(4);
    double min_area = std::numeric_limits<double>::max(), test_area;
    index_hull = start_line[0];
    do
    {
        if (test_norm[0] < test_norm[1]) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        extra_index_hull = finish_line[1];
        do
        {
            if (test_norm[2] < test_norm[3]) { extra_next_index_hull = extra_index_hull + 1; }
            else { extra_next_index_hull = extra_index_hull - 1; }

            if (extra_next_index_hull == hull_size) { extra_next_index_hull = 0; }
            if (extra_next_index_hull == -1) { extra_next_index_hull = hull_size - 1; }

            test_result_angle_list[0]
                = intersectionLines(result_side_begin[0], result_side_end[0],
                                    result_side_begin[1], result_side_end[1]);
            test_result_angle_list[1]
                = intersectionLines(result_side_begin[1], result_side_end[1],
                                    hull[0][extra_index_hull], hull[0][extra_next_index_hull]);
            test_result_angle_list[2]
                = intersectionLines(hull[0][extra_index_hull], hull[0][extra_next_index_hull],
                                    hull[0][index_hull], hull[0][next_index_hull]);
            test_result_angle_list[3]
                = intersectionLines(hull[0][index_hull], hull[0][next_index_hull],
                                    result_side_begin[0], result_side_end[0]);
            test_area = getQuadrilateralArea(test_result_angle_list[0],
                                             test_result_angle_list[1],
                                             test_result_angle_list[2],
                                             test_result_angle_list[3]);
            if (min_area > test_area)
            {
                min_area = test_area;
                for (size_t i = 0; i < test_result_angle_list.size(); i++)
                {
                    result_angle_list[i] = test_result_angle_list[i];
                }
            }

            extra_index_hull = extra_next_index_hull;
        }
        while(extra_index_hull != unstable_pnt);

        index_hull = next_index_hull;
    }
    while(index_hull != unstable_pnt);

    if (norm(result_angle_list[0] - angle_list[2]) >
        norm(angle_list[2] - angle_list[1]) / 3) { result_angle_list[0] = angle_list[2]; }

    if (norm(result_angle_list[1] - angle_list[1]) >
        norm(angle_list[1] - angle_list[0]) / 3) { result_angle_list[1] = angle_list[1]; }

    if (norm(result_angle_list[2] - angle_list[0]) >
        norm(angle_list[0] - angle_list[3]) / 3) { result_angle_list[2] = angle_list[0]; }

    if (norm(result_angle_list[3] - angle_list[3]) >
        norm(angle_list[3] - angle_list[2]) / 3) { result_angle_list[3] = angle_list[3]; }



    return result_angle_list;
}

//        b __________ c
//        /           |
//       /            |
//      /      S      |
//     /              |
//   a --------------- d

double QRDecode::getQuadrilateralArea(Point a, Point b, Point c, Point d)
{
    double length_sides[4], perimeter = 0.0, result_area = 1.0;
    length_sides[0] = norm(a - b); length_sides[1] = norm(b - c);
    length_sides[2] = norm(c - d); length_sides[3] = norm(d - a);

    for (int i = 0; i < 4; i++) { perimeter += length_sides[i]; }
    perimeter /= 2;

    for (int i = 0; i < 4; i++)
    {
        result_area *= (perimeter - length_sides[i]);
    }

    result_area = sqrt(result_area);

    return result_area;
}

//      / | b
//     /  |
//    /   |
//  a/    | c

double QRDecode::getCosVectors(Point a, Point b, Point c)
{
    return ((a - b).x * (c - b).x + (a - b).y * (c - b).y) / (norm(a - b) * norm(c - b));
}

CV_EXPORTS bool detectQRCode(InputArray in, std::vector<Point> &points, double eps_x, double eps_y)
{
    CV_Assert(in.isMat());
    CV_Assert(in.getMat().type() == CV_8UC1);
    QRDecode qrdec;
    qrdec.init(in.getMat(), eps_x, eps_y);
    qrdec.binarization();
    if (!qrdec.localization()) { return false; }
    if (!qrdec.transformation()) { return false; }
    points = qrdec.getTransformationPoints();
    return true;
}

}
