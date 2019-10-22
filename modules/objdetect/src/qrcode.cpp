// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc/types_c.h"

#ifdef HAVE_QUIRC
#include "quirc.h"
#endif

#include <limits>
#include <cmath>
#include <iostream>
#include <queue>

namespace cv
{
using std::vector;

class QRDetect
{
public:
    void init(const Mat& src, double eps_vertical_ = 0.2, double eps_horizontal_ = 0.1);
    bool localization();
    bool computeTransformationPoints(size_t c);
    Mat getBinBarcode() { return bin_barcode; }
    Mat getStraightBarcode() { return straight_barcode; }
    vector<vector<Point2f>> getTransformationPoints() { return transformation_points; }
    static Point2f intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2);


protected:

    vector<Vec3d> searchHorizontalLines();
    vector<Point2f> separateVerticalLines(const vector<Vec3d> &list_lines);
    void fixationPoints(vector<Point2f> &local_point);
    vector<Point2f> getQuadrilateral(vector<Point2f> angle_list);
    bool testBypassRoute(vector<Point2f> hull, int start, int finish);
    inline double getCosVectors(Point2f a, Point2f b, Point2f c);
    bool nextSet(int *set, int n, int m);
    bool checkPoints(vector<Point2f> triangle_points);
    bool checkPointsInside(vector<Point2f> triangle_points, vector<Point2f> all_localization_points);
    Mat barcode, bin_barcode, straight_barcode, bin_barcode_fullsize, bin_barcode_temp;
    vector<vector<Point2f>> localization_points, transformation_points;
    double eps_vertical, eps_horizontal, coeff_expansion;

    Mat original;
    enum resize_direction { ZOOMING, SHRINKING, UNCHANGED } purpose;

};
void QRDetect::fixationPoints(vector<Point2f> &local_point)
{
    CV_TRACE_FUNCTION();
    double cos_angles[3], norm_triangl[3];

    norm_triangl[0] = norm(local_point[1] - local_point[2]);
    norm_triangl[1] = norm(local_point[0] - local_point[2]);
    norm_triangl[2] = norm(local_point[1] - local_point[0]);

    cos_angles[0] = (norm_triangl[1] * norm_triangl[1] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[0] * norm_triangl[0]) / (2 * norm_triangl[1] * norm_triangl[2]);
    cos_angles[1] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[1] * norm_triangl[1]) / (2 * norm_triangl[0] * norm_triangl[2]);
    cos_angles[2] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[1] * norm_triangl[1]
                  -  norm_triangl[2] * norm_triangl[2]) / (2 * norm_triangl[0] * norm_triangl[1]);

    const double angle_barrier = 0.85;
    if (fabs(cos_angles[0]) > angle_barrier || fabs(cos_angles[1]) > angle_barrier || fabs(cos_angles[2]) > angle_barrier)
    {
      local_point.clear();
      return;
    }

    size_t i_min_cos =
       (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0 :
       (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1 : 2;

    size_t index_max = 0;
    double max_area = std::numeric_limits<double>::min();
    for (size_t i = 0; i < local_point.size(); i++)
    {
        const size_t current_index = i % 3;
        const size_t left_index  = (i + 1) % 3;
        const size_t right_index = (i + 2) % 3;

        const Point2f current_point(local_point[current_index]),
            left_point(local_point[left_index]), right_point(local_point[right_index]),
            central_point(intersectionLines(current_point,
                              Point2f(static_cast<float>((local_point[left_index].x + local_point[right_index].x) * 0.5),
                                      static_cast<float>((local_point[left_index].y + local_point[right_index].y) * 0.5)),
                              Point2f(0, static_cast<float>(bin_barcode_temp.rows - 1)),
                              Point2f(static_cast<float>(bin_barcode_temp.cols - 1),
                                      static_cast<float>(bin_barcode_temp.rows - 1))));


        vector<Point2f> list_area_pnt;
        list_area_pnt.push_back(current_point);

        vector<LineIterator> list_line_iter;
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, left_point));
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, central_point));
        list_line_iter.push_back(LineIterator(bin_barcode_temp, current_point, right_point));

        for (size_t k = 0; k < list_line_iter.size(); k++)
        {
            uint8_t future_pixel = 255, count_index = 0;
            for(int j = 0; j < list_line_iter[k].count; j++, ++list_line_iter[k])
            {
                if (list_line_iter[k].pos().x >= bin_barcode_temp.cols ||
                    list_line_iter[k].pos().y >= bin_barcode_temp.rows) { break; }
                const uint8_t value = bin_barcode_temp.at<uint8_t>(list_line_iter[k].pos());
                if (value == future_pixel)
                {
                    future_pixel = 255 - future_pixel;
                    count_index++;
                    if (count_index == 3)
                    {
                        list_area_pnt.push_back(list_line_iter[k].pos());
                        break;
                    }
                }
            }
        }

        const double temp_check_area = contourArea(list_area_pnt);
        if (temp_check_area > max_area)
        {
            index_max = current_index;
            max_area = temp_check_area;
        }

    }
    if (index_max == i_min_cos) { std::swap(local_point[0], local_point[index_max]); }
    else { local_point.clear(); return; }

    const Point2f rpt = local_point[0], bpt = local_point[1], gpt = local_point[2];
    Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
    if( determinant(m) > 0 )
    {
        std::swap(local_point[1], local_point[2]);
    }
}

void QRDetect::init(const Mat& src, double eps_vertical_, double eps_horizontal_)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!src.empty());
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side < 512.0)
    {
        purpose = ZOOMING;
        coeff_expansion = 512.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR);
    }
    else if (min_side > 512.0)
    {
        purpose = SHRINKING;
        coeff_expansion = min_side / 512.0;
        const int width  = cvRound(src.size().width  / coeff_expansion);
        const int height = cvRound(src.size().height  / coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_AREA);
    }
    else
    {
        purpose = UNCHANGED;
        coeff_expansion = 1.0;
        barcode = src.clone();
    }

    eps_vertical   = eps_vertical_;
    eps_horizontal = eps_horizontal_;
    adaptiveThreshold(barcode, bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    adaptiveThreshold(src, bin_barcode_fullsize, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
}

vector<Vec3d> QRDetect::searchHorizontalLines()
{
    CV_TRACE_FUNCTION();
    vector<Vec3d> result;
    const int height_bin_barcode = bin_barcode.rows;
    const int width_bin_barcode  = bin_barcode.cols;
    const size_t test_lines_size = 5;
    double test_lines[test_lines_size];
    vector<size_t> pixels_position;

    for (int y = 0; y < height_bin_barcode; y++)
    {
        pixels_position.clear();
        const uint8_t *bin_barcode_row = bin_barcode.ptr<uint8_t>(y);

        int pos = 0;
        for (; pos < width_bin_barcode; pos++) { if (bin_barcode_row[pos] == 0) break; }

        pixels_position.push_back(pos);
        pixels_position.push_back(pos);
        pixels_position.push_back(pos);
        uint8_t future_pixel = 255;
        for (int x = pos; x < width_bin_barcode; x++)
        {
            if (bin_barcode_row[x] == future_pixel)
            {
                future_pixel = 255 - future_pixel;
                pixels_position.push_back(x);
            }
        }
        pixels_position.push_back(width_bin_barcode - 1);
        for (size_t i = 2; i < pixels_position.size() - 4; i+=2)
        {
            test_lines[0] = static_cast<double>(pixels_position[i - 1] - pixels_position[i - 2]);
            test_lines[1] = static_cast<double>(pixels_position[i    ] - pixels_position[i - 1]);
            test_lines[2] = static_cast<double>(pixels_position[i + 1] - pixels_position[i    ]);
            test_lines[3] = static_cast<double>(pixels_position[i + 2] - pixels_position[i + 1]);
            test_lines[4] = static_cast<double>(pixels_position[i + 3] - pixels_position[i + 2]);

            double length = 0.0, weight = 0.0;

            for (size_t j = 0; j < test_lines_size; j++) { length += test_lines[j]; }

            if (length == 0) { continue; }
            for (size_t j = 0; j < test_lines_size; j++)
            {
                if (j != 2) { weight += fabs((test_lines[j] / length) - 1.0/7.0); }
                else        { weight += fabs((test_lines[j] / length) - 3.0/7.0); }
            }

            if (weight < eps_vertical)
            {
                Vec3d line;
                line[0] = static_cast<double>(pixels_position[i - 2]);
                line[1] = y;
                line[2] = length;
                result.push_back(line);
            }
        }
    }
    return result;
}

vector<Point2f> QRDetect::separateVerticalLines(const vector<Vec3d> &list_lines)
{
    CV_TRACE_FUNCTION();
    vector<Vec3d> result;
    int temp_length;
    vector<Point2f> point2f_result;
    uint8_t next_pixel;
    vector<double> test_lines;
    temp_length = 0;

    for (size_t pnt = 0; pnt < list_lines.size(); pnt++)
    {
        const int x = cvRound(list_lines[pnt][0] + list_lines[pnt][2] * 0.5);
        const int y = cvRound(list_lines[pnt][1]);

        // --------------- Search vertical up-lines --------------- //

        test_lines.clear();
        uint8_t future_pixel_up = 255;

        for (int j = y; j < bin_barcode.rows - 1; j++)
        {
            next_pixel = bin_barcode.ptr<uint8_t>(j + 1)[x];
            temp_length++;
            if (next_pixel == future_pixel_up)
            {
                future_pixel_up = 255 - future_pixel_up;
                test_lines.push_back(temp_length);
                temp_length = 0;
                if (test_lines.size() == 3) { break; }
              }
          }

          // --------------- Search vertical down-lines --------------- //

          uint8_t future_pixel_down = 255;
          for (int j = y; j >= 1; j--)
          {
              next_pixel = bin_barcode.ptr<uint8_t>(j - 1)[x];
              temp_length++;
              if (next_pixel == future_pixel_down)
              {
                  future_pixel_down = 255 - future_pixel_down;
                  test_lines.push_back(temp_length);
                  temp_length = 0;
                  if (test_lines.size() == 6) { break; }
              }
          }

          // --------------- Compute vertical lines --------------- //

          if (test_lines.size() == 6)
          {
              double length = 0.0, weight = 0.0;

              for (size_t i = 0; i < test_lines.size(); i++) { length += test_lines[i]; }

              CV_Assert(length > 0);
              for (size_t i = 0; i < test_lines.size(); i++)
              {
                  if (i % 3 != 0) { weight += fabs((test_lines[i] / length) - 1.0/ 7.0); }
                  else            { weight += fabs((test_lines[i] / length) - 3.0/14.0); }
              }

              if(weight < eps_horizontal)
              {
                  result.push_back(list_lines[pnt]);
              }
          }
      }
      if (result.size() > 2)
      {
          for (size_t i = 0; i < result.size(); i++)
          {
              point2f_result.push_back(
                    Point2f(static_cast<float>(result[i][0] + result[i][2] * 0.5),
                            static_cast<float>(result[i][1])));
          }
      }
      return point2f_result;
}


bool QRDetect::checkPoints(vector<Point2f> triangle_points)
{
      int count_b = 0;
      int count_w = 0;
      Point p0(triangle_points[0].x, triangle_points[0].y);
      Point p1(triangle_points[1].x, triangle_points[1].y);
      Point p2(triangle_points[2].x, triangle_points[2].y);
      Point p3(triangle_points[3].x, triangle_points[3].y);
      Mat lineMask = Mat::zeros(bin_barcode.size(), bin_barcode.type());
      line( lineMask, p0, p1, 255, 1, 8, 0);
      line( lineMask, p1, p2, 255, 1, 8, 0);
      line( lineMask, p2, p3, 255, 1, 8, 0);
      line( lineMask, p0, p3, 255, 1, 8, 0);
      vector<vector<Point>> contours;
      vector<Vec4i> hierarchy;
      findContours(lineMask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

      if(contours.size() != 2) return false;
      Mutex mu;
      parallel_for_(Range(0, bin_barcode.rows ), [&]( const Range& range1)
      {
          for(int m = range1.start; m < range1.end; m++)
          {
              parallel_for_(Range(0, bin_barcode.cols ), [&]( const Range& range2)
              {
                  for(int l = range2.start; l < range2.end; l++)
                  {
                      double dist;
                      if((dist = pointPolygonTest(contours[0], Point2f(l, m), false)) > 0)
                      {
                          int pixel = bin_barcode.at<uchar>(m, l);
                          mu.lock();
                          if(pixel == 255)
                          {
                              count_w++;
                          }
                          if(pixel == 0)
                          {
                              count_b++;
                          }
                          mu.unlock();

                      }

                    }
               });
          }
     });
     double frac = (double)count_b / (double)count_w;
     if ((frac <= 0.835) || (frac >= 1.24)) {return false;}
     return true;
}

bool QRDetect::checkPointsInside(vector<Point2f> triangle_points, vector<Point2f> all_localization_points)
{
      Point p0(triangle_points[0].x, triangle_points[0].y);
      Point p1(triangle_points[1].x, triangle_points[1].y);
      Point p2(triangle_points[2].x, triangle_points[2].y);
      Point p3(triangle_points[3].x, triangle_points[3].y);
      Mat lineMask = Mat::zeros(bin_barcode.size(), bin_barcode.type());
      line( lineMask, p0, p1, 255, 1, 8, 0);
      line( lineMask, p1, p2, 255, 1, 8, 0);
      line( lineMask, p2, p3, 255, 1, 8, 0);
      line( lineMask, p0, p3, 255, 1, 8, 0);
      vector<vector<Point>> contours;
      vector<Vec4i> hierarchy;
      findContours(lineMask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
      int count = 0;
      if(contours.size() != 2) return false;
      for(size_t i = 0; i < all_localization_points.size(); i++)
      {
            double dist;
            if((dist = pointPolygonTest(contours[0], all_localization_points[i], false)) > 0)
            {
               count++;
            }
       }
       if (count == 3) return true;
       else return false;
}
bool QRDetect::localization()
{
    CV_TRACE_FUNCTION();

    vector<Point2f> tmp_localization_points;
    int num_qrcodes;
    size_t number_possible_purpose = 1;
    if(purpose == SHRINKING) number_possible_purpose = 2;
    Mat tmp_shrinking = bin_barcode;
    int tmp_num_points = 0;
    int num_points = -1;
    for(eps_horizontal = 0.1; eps_horizontal < 0.4; eps_horizontal += 0.1)
    {
        tmp_num_points = 0;
        num_points = -1;
        if(purpose == SHRINKING) number_possible_purpose = 2;
        else number_possible_purpose = 1;
        for(size_t k = 0; k < number_possible_purpose; k++)
        {
            if(k == 1) bin_barcode = bin_barcode_fullsize;
            vector<Vec3d> list_lines_x = searchHorizontalLines();
            if( list_lines_x.empty() )
            {
                if(k == 0)
                {
                    k = 1;
                    bin_barcode = bin_barcode_fullsize;
                    list_lines_x = searchHorizontalLines();
                    if (list_lines_x.empty()) break;
                 }
                 else break;
            }
            vector<Point2f> list_lines_y = separateVerticalLines(list_lines_x);
            if ( list_lines_y.size() < 3 )
            {
                if(k == 0)
                {
                    k = 1;
                    bin_barcode = bin_barcode_fullsize;
                    list_lines_x = searchHorizontalLines();
                    if (list_lines_x.empty()) break;
                    list_lines_y = separateVerticalLines(list_lines_x);
                    if( list_lines_y.size() < 3 ) break;
                }
                else break;
             }
             vector<int> index_list_lines_y;
             for(size_t i = 0; i < list_lines_y.size(); i++)
                 index_list_lines_y.push_back(-1);
             num_points = 0;
             for(size_t i = 0; i < list_lines_y.size() - 1; i++)
             {
                 for(size_t j = i; j < list_lines_y.size(); j++ )
                 {

                     double points_distance = sqrt((list_lines_y[i].x - list_lines_y[j].x) * (list_lines_y[i].x - list_lines_y[j].x)
                                     + ((list_lines_y[i].y - list_lines_y[j].y) * (list_lines_y[i].y - list_lines_y[j].y)));
                     if((points_distance >= 0) && (points_distance <= 10))
                     {
                         if((index_list_lines_y[i] == -1) && (index_list_lines_y[j] == -1))
                         {
                             index_list_lines_y[i] = num_points;
                             index_list_lines_y[j] = num_points;
                             num_points++;
                         }
                         else if (index_list_lines_y[i] != -1) index_list_lines_y[j] = index_list_lines_y[i];
                         else if (index_list_lines_y[j] != -1) index_list_lines_y[i] = index_list_lines_y[j];

                     }
                  }
               }
               for(size_t i = 0; i < index_list_lines_y.size(); i++)
                  if(index_list_lines_y[i] == -1)
                  {
                      index_list_lines_y[i] = num_points;
                      num_points++;
                  }
                if ((tmp_num_points < num_points) && (k == 1))
                {
                   purpose = UNCHANGED;
                   tmp_num_points = num_points;
                   bin_barcode = bin_barcode_fullsize;
                   coeff_expansion = 1.0;
                 }
                 if ((tmp_num_points < num_points) && (k == 0))
                 {
                     tmp_num_points = num_points;
                 }
        }

        if ((tmp_num_points < 3) && (tmp_num_points >= 1))
        {
            const double min_side = std::min(bin_barcode_fullsize.size().width, bin_barcode_fullsize.size().height);
            if(min_side > 512)
            {
              bin_barcode = tmp_shrinking;
              purpose = SHRINKING;
              coeff_expansion = min_side / 512.0;
            }
            if(min_side < 512)
            {
              bin_barcode = tmp_shrinking;
              purpose = ZOOMING;
              coeff_expansion = 512 / min_side;
            }
        }
        else break;
    }
    if(purpose == SHRINKING) bin_barcode = tmp_shrinking;
    num_points = tmp_num_points;
    vector<Vec3d> list_lines_x = searchHorizontalLines();
    if( list_lines_x.empty() ) { return false; }
    vector<Point2f> list_lines_y = separateVerticalLines(list_lines_x);
    if( list_lines_y.size() < 3 ) { return false; }
    if(num_points < 3) return false;
    num_qrcodes = num_points / 3;
    if ((num_points % 3) != 0) num_qrcodes++;;
    Mat labels;
    kmeans(list_lines_y, num_points, labels,
           TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
            num_points, KMEANS_PP_CENTERS, tmp_localization_points);
    vector<vector<Point2f>> triangles;
    size_t temp_num_points = 0;
    bin_barcode_temp = bin_barcode.clone();
    if (purpose == SHRINKING)
    {
       const int width  = cvRound(bin_barcode.size().width  * coeff_expansion);
       const int height = cvRound(bin_barcode.size().height * coeff_expansion);
       Size new_size(width, height);
       Mat intermediate;
       resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
       bin_barcode = intermediate.clone();
    }
    else if (purpose == ZOOMING)
    {
        const int width  = cvRound(bin_barcode.size().width  / coeff_expansion);
        const int height = cvRound(bin_barcode.size().height / coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
    }
    else bin_barcode = bin_barcode_fullsize.clone();
    Mat gray, blur_image, threshold_output;
    Mat bar = barcode;
    const int width  = cvRound(bin_barcode.size().width);
    const int height = cvRound(bin_barcode.size().height);
    Size new_size(width, height);
    Mat intermediate;
    resize(bar, bar, new_size, 0, 0, INTER_LINEAR);
    blur(bar, blur_image, Size(3, 3));
    threshold(blur_image, threshold_output, 50, 255, THRESH_BINARY);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    vector<Point2f> all_contours_points;
    for(size_t i = 0; i < contours.size(); i ++)
        for(size_t j = 0; j < contours[i].size(); j++)
            all_contours_points.push_back(contours[i][j]);

    Mat qrcode_labels;
    vector<Point2f> clustered_localization_points;

    kmeans(all_contours_points, num_qrcodes, qrcode_labels,
            TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
            num_qrcodes, KMEANS_PP_CENTERS, clustered_localization_points);

    vector<vector<Point2f>> qrcode_clusters(num_qrcodes);
    for(int i = 0; i < num_qrcodes; i++)
        for(int j = 0; j < (int)all_contours_points.size(); j++)
        {
            if (qrcode_labels.at<int>(0, j) == i)
            {
                qrcode_clusters[i].push_back(all_contours_points[j]);
            }
        }

    vector<vector<Point2f>> hull(num_qrcodes);
    for(size_t i = 0; i < qrcode_clusters.size(); i++)
        convexHull(Mat(qrcode_clusters[i]), hull[i]);

    vector<Point2f> not_resized_localization_points = tmp_localization_points;
       if (purpose == SHRINKING)
       {
           for (size_t j = 0; j < not_resized_localization_points.size(); j++)
           {
                not_resized_localization_points[j] *= coeff_expansion;
            }
       }
       else if (purpose == ZOOMING)
       {
          for (size_t j = 0; j < not_resized_localization_points.size(); j++)
          {
               not_resized_localization_points[j] /= coeff_expansion;
          }
       }
    vector<vector<Point2f>> true_points_group;
    for(size_t j = 0; j < hull.size(); j++)
    {
        Mat lineMask = Mat::zeros(bar.size(), threshold_output.type());
        for(size_t i = 0; i < hull[j].size() - 1; i++)
        {
            line( lineMask, hull[j][i], hull[j][i + 1], 255, 1, 8, 0);
        }
        line( lineMask, hull[j][0], hull[j][hull[j].size() - 1], 255, 1, 8, 0);
        vector<vector<Point>> mask_contours;
        vector<Vec4i> mask_hierarchy;
        findContours(lineMask, mask_contours, mask_hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        vector<Point2f> temp;
        true_points_group.push_back(temp);
        for(size_t i = 0; i < not_resized_localization_points.size(); i++)
        {
            double dist;
            if((dist = pointPolygonTest(mask_contours[0], not_resized_localization_points[i], false)) > 0)
            {
              true_points_group[j].push_back(tmp_localization_points[i]);
              tmp_localization_points[i].x = -1;
            }
        }
    }
    vector<Point2f> copy;
    for(size_t j = 0; j < tmp_localization_points.size(); j++)
    {
       if(tmp_localization_points[j].x != -1) copy.push_back(tmp_localization_points[j]);
    }

    tmp_localization_points = copy;

    for(int q = 0; q < num_qrcodes; q++)
    {
       for(size_t i = 0; i < true_points_group.size(); i++)
       {
          if (true_points_group[i].size() < 3)
          {
              for(size_t j = 0; j < true_points_group[i].size(); j++)
                  tmp_localization_points.push_back(true_points_group[i][j]);
              true_points_group[i].clear();
          }
       }
       vector<vector<Point2f>> temp_for_copy;
       for(size_t i = 0; i < true_points_group.size(); i++)
       {
            if(true_points_group[i].size() != 0)
              temp_for_copy.push_back(true_points_group[i]);
       }
       true_points_group = temp_for_copy;
       if(true_points_group.size() == 0)
       {
          true_points_group.push_back(tmp_localization_points);
          tmp_localization_points.clear();
        }

       if (true_points_group[0].size() < 3) break;
       for(size_t i = 0; i < true_points_group.size(); i++)
       {
         double distanse = -1;
         int iter_most_close = 0;
         for(size_t w = 0; w < true_points_group[i].size() ; w++)
         {
              if(distanse == -1)
              {
                  distanse = sqrt(true_points_group[i][w].x * true_points_group[i][w].x
                                  + true_points_group[i][w].y * true_points_group[i][w].y);
                  iter_most_close = w;
              }
              else if ((sqrt(true_points_group[i][w].x * true_points_group[i][w].x
                              + true_points_group[i][w].y * true_points_group[i][w].y)) < distanse)
              {
                  distanse = sqrt(true_points_group[i][w].x * true_points_group[i][w].x
                                  + true_points_group[i][w].y * true_points_group[i][w].y);
                  iter_most_close = w;
              }
         }

         std::swap(true_points_group[i][iter_most_close], true_points_group[i][0]);
         std::sort((true_points_group[i].begin() + 1), true_points_group[i].end(), [&](Point2f a, Point2f b) {
            return (sqrt((a.x - true_points_group[i][0].x) * (a.x - true_points_group[i][0].x)
            + (a.y - true_points_group[i][0].y) * (a.y - true_points_group[i][0].y)))
            < (sqrt((b.x - true_points_group[i][0].x) *(b.x - true_points_group[i][0].x) +
            (b.y - true_points_group[i][0].y) * (b.y - true_points_group[i][0].y)));
         });
     }

       num_points = 0;
       for(size_t i = 0; i < true_points_group.size(); i++)
       {
           num_points = num_points + true_points_group[i].size();
       }
       num_points =  num_points + tmp_localization_points.size();
       temp_num_points = num_points;

       int** ind_points = new int * [true_points_group.size()];
       for(size_t i = 0; i < true_points_group.size(); i++)
       {
         ind_points[i] = new int[true_points_group[i].size()];
       }
       for(size_t i = 0; i < true_points_group.size(); i++)
          for(size_t j = 0; j < true_points_group[i].size(); j++)  ind_points[i][j] = j;

       int set_size[true_points_group.size()];
       for(size_t i = 0; i < true_points_group.size(); i++)
           set_size[i] = true_points_group[i].size() * (true_points_group[i].size() - 1 ) * (true_points_group[i].size() - 2) / 6;
       int *** all_points = new int ** [true_points_group.size()];
       for(size_t i = 0; i < true_points_group.size(); i++)
           all_points[i] = new int* [set_size[i]];
       for(size_t i = 0; i < true_points_group.size(); i++)
           for (int j = 0; j < set_size[i]; j++)
                all_points[i][j] = new int[3];

       for(size_t i = 0; i < true_points_group.size(); i++)
           for(size_t j = 0; j < 3; j++)
               all_points[i][0][j] = ind_points[i][j];

       int iter = localization_points.size();
       for(size_t i = 0; i < true_points_group.size(); i++)
       {
           int p = 0;
           while(nextSet(ind_points[i], true_points_group[i].size(), 3))
           {
              p++;
              for(size_t j = 0; j < 3; j++)
              {
                  all_points[i][p][j] = ind_points[i][j];
              }
           }
       }
       localization_points.resize(iter + true_points_group.size());
       transformation_points.resize(iter + true_points_group.size());

       vector<vector<Point2f>> loc = true_points_group;
       int end[true_points_group.size()];
       for(size_t i = 0; i < true_points_group.size(); i++)
          end[i] = iter + set_size[i];
       for(size_t s = 0; s < true_points_group.size(); s++)
       {
           bool flag = false;

           for (int r = iter ; r < end[s]; r++)
           {
                size_t x = iter + s;
                size_t k = r - iter;
                if(flag) break;

                if ((k >= 1) && ((all_points[s][k][0] - all_points[s][k - 1][0]) == 1))
                {
                    loc[s][0].x = -2;
                    break;
                }
                bool flag_for_break = false;
                vector<Point2f> triangle;
                for(size_t l = 0; l < 3; l++)
                {
                    triangle.push_back(true_points_group[s][all_points[s][k][l]]);
                }
                if(flag) break;
                fixationPoints(triangle);
                if(triangle.size() == 3)
                {
                    localization_points[x] = triangle;

                    if (purpose == SHRINKING)
                    {
                        for (size_t j = 0; j < localization_points[x].size(); j++)
                        {
                             localization_points[x][j] *= coeff_expansion;

                         }
                    }
                    else if (purpose == ZOOMING)
                    {
                       for (size_t j = 0; j < localization_points[x].size(); j++)
                       {
                            localization_points[x][j] /= coeff_expansion;

                       }
                    }
                    for (size_t i = 0; i < localization_points[x].size(); i++)
                    {
                         for (size_t j = i + 1; j < localization_points[x].size(); j++)
                         {
                              if (norm(localization_points[x][i] - localization_points[x][j]) < 10)
                              {
                                  localization_points[x].clear();
                                  flag_for_break = true;
                                  break;
                               }
                         } if(flag_for_break) break;
                   }
                   if(flag_for_break != true)
                   {
                       if(localization_points[x].size() == 3)
                       {
                           if (computeTransformationPoints(x))
                           {
                               if(((temp_num_points / 3) == 1) || ((checkPointsInside(transformation_points[x], not_resized_localization_points)) == true))
                               {
                                   if(checkPoints(transformation_points[x]))
                                   {
                                       for(int l = 0; l < 3; l++)
                                          loc[s][all_points[s][k][l]].x = -1;

                                       flag = true;
                                       break;
                                   }
                               }
                            }
                            if(!flag)
                            {
                                 transformation_points[x].clear();
                                 localization_points[x].clear();
                             }
                        }
                    }
                }
                if(flag) break;
            }
          }
          for(size_t s = 0; s < true_points_group.size(); s++)
          {
               vector<Point2f> for_copy;
               if(loc[s].size() == 3)
               {

                 if((true_points_group.size() > 1) || ((true_points_group.size() == 1) && (tmp_localization_points.size() != 0)) )
                 {
                    for(size_t j = 0; j < true_points_group[s].size(); j++)
                    {
                        if(loc[s][j].x != -1)
                        {
                            loc[s][j].x = -1;
                            tmp_localization_points.push_back(true_points_group[s][j]);
                        }
                     }
                  }
               }
               for(size_t j = 0; j < loc[s].size(); j++)
               {
                    if((loc[s][j].x != -1) && (loc[s][j].x != -2) )
                    {
                        for_copy.push_back(true_points_group[s][j]);
                    }
                    if (loc[s][j].x == -2)
                    {
                        if((true_points_group.size() > 1))
                        {
                            tmp_localization_points.push_back(true_points_group[s][j]);
                        }

                    }
                }
                true_points_group[s] = for_copy;
             }
             temp_num_points = 0;
             for(size_t i = 0; i < true_points_group.size(); i++)
                temp_num_points = temp_num_points + true_points_group[i].size();
              temp_num_points = temp_num_points + tmp_localization_points.size();
              vector<vector<Point2f>> for_copy_loc;
              vector<vector<Point2f>> for_copy_trans;
              for(size_t j = 0; j < localization_points.size(); j++)
              {
                  if(localization_points[j].size() == 3) for_copy_loc.push_back(localization_points[j]);
                  if(transformation_points[j].size() == 4) for_copy_trans.push_back(transformation_points[j]);
              }
              localization_points = for_copy_loc;
              transformation_points = for_copy_trans;

              if((num_points - temp_num_points) == 1) q--;
              if(((num_points - temp_num_points) == 0) && (tmp_localization_points.size() == 0) && (true_points_group.size() == 1))
              break;

     }
     if ((transformation_points.size() == 0) || (localization_points.size() == 0)) return false;
    return true;
}
bool QRDetect::computeTransformationPoints(size_t c)
{
      CV_TRACE_FUNCTION();
      if (localization_points[c].size() != 3)
      {
         return false;
      }

      vector<Point> locations, non_zero_elem[3], newHull;
      vector<Point2f> new_non_zero_elem[3];
      for (size_t i = 0; i < 3 ; i++)
      {
            Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
            uint8_t next_pixel, future_pixel = 255;
            int count_test_lines = 0, index = cvRound(localization_points[c][i].x);
            for (; index < bin_barcode.cols - 1; index++)
            {

                    next_pixel = bin_barcode.ptr<uint8_t>(cvRound(localization_points[c][i].y))[index + 1];
                    if (next_pixel == future_pixel)
                    {
                        future_pixel = 255 - future_pixel;
                        count_test_lines++;

                        if (count_test_lines == 2)
                        {

                              floodFill(bin_barcode, mask,
                                        Point(index + 1, cvRound(localization_points[c][i].y)), 255,
                                        0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
                              break;
                        }
                    }

              }
              Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));
              findNonZero(mask_roi, non_zero_elem[i]);
              newHull.insert(newHull.end(), non_zero_elem[i].begin(), non_zero_elem[i].end());
        }
        convexHull(newHull, locations);
        for (size_t i = 0; i < locations.size(); i++)
        {
            for (size_t j = 0; j < 3; j++)
            {
                for (size_t k = 0; k < non_zero_elem[j].size(); k++)
                {
                    if (locations[i] == non_zero_elem[j][k])
                    {
                        new_non_zero_elem[j].push_back(locations[i]);
                    }

                }
            }
        }

        double pentagon_diag_norm = -1;
        Point2f down_left_edge_point, up_right_edge_point, up_left_edge_point;
        for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
        {
            for (size_t j = 0; j < new_non_zero_elem[2].size(); j++)
            {
                double temp_norm = norm(new_non_zero_elem[1][i] - new_non_zero_elem[2][j]);
                if (temp_norm > pentagon_diag_norm)
                {
                    down_left_edge_point = new_non_zero_elem[1][i];
                    up_right_edge_point  = new_non_zero_elem[2][j];
                    pentagon_diag_norm = temp_norm;
                }
            }
          }

          if (down_left_edge_point == Point2f(0, 0) ||
          up_right_edge_point  == Point2f(0, 0) ||
          new_non_zero_elem[0].size() == 0) { return false; }

          double max_area = -1;
          up_left_edge_point = new_non_zero_elem[0][0];

          for (size_t i = 0; i < new_non_zero_elem[0].size(); i++)
          {
              vector<Point2f> list_edge_points;
              list_edge_points.push_back(new_non_zero_elem[0][i]);
              list_edge_points.push_back(down_left_edge_point);
              list_edge_points.push_back(up_right_edge_point);

              double temp_area = fabs(contourArea(list_edge_points));
              if (max_area < temp_area)
              {
                  up_left_edge_point = new_non_zero_elem[0][i];
                  max_area = temp_area;
              }
          }

          Point2f down_max_delta_point, up_max_delta_point;
          double norm_down_max_delta = -1, norm_up_max_delta = -1;
          for (size_t i = 0; i < new_non_zero_elem[1].size(); i++)
          {
              double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[1][i]) + norm(down_left_edge_point - new_non_zero_elem[1][i]);
              if (norm_down_max_delta < temp_norm_delta)
              {
                  down_max_delta_point = new_non_zero_elem[1][i];
                  norm_down_max_delta = temp_norm_delta;
              }
          }


          for (size_t i = 0; i < new_non_zero_elem[2].size(); i++)
          {
              double temp_norm_delta = norm(up_left_edge_point - new_non_zero_elem[2][i]) + norm(up_right_edge_point - new_non_zero_elem[2][i]);
              if (norm_up_max_delta < temp_norm_delta)
              {
                  up_max_delta_point = new_non_zero_elem[2][i];
                  norm_up_max_delta = temp_norm_delta;
              }
           }
           vector<Point2f> tmp_transformation_points;
           tmp_transformation_points.push_back(down_left_edge_point);
           tmp_transformation_points.push_back(up_left_edge_point);
           tmp_transformation_points.push_back(up_right_edge_point);
           tmp_transformation_points.push_back(intersectionLines(down_left_edge_point, down_max_delta_point,
                         up_right_edge_point, up_max_delta_point));
           transformation_points[c] = tmp_transformation_points;

           vector<Point2f> quadrilateral = getQuadrilateral(transformation_points[c]);
           transformation_points[c] = quadrilateral;

    return true;
}

Point2f QRDetect::intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2)
{
    Point2f result_square_angle(
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.x - b2.x) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.x - a2.x)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x)),
                              ((a1.x * a2.y  -  a1.y * a2.x) * (b1.y - b2.y) -
                               (b1.x * b2.y  -  b1.y * b2.x) * (a1.y - a2.y)) /
                              ((a1.x - a2.x) * (b1.y - b2.y) -
                               (a1.y - a2.y) * (b1.x - b2.x))
                              );
    return result_square_angle;
}

// test function (if true then ------> else <------ )
bool QRDetect::testBypassRoute(vector<Point2f> hull, int start, int finish)
{
    CV_TRACE_FUNCTION();
    int index_hull = start, next_index_hull, hull_size = (int)hull.size();
    double test_length[2] = { 0.0, 0.0 };
    do
    {
        next_index_hull = index_hull + 1;
        if (next_index_hull == hull_size) { next_index_hull = 0; }
        test_length[0] += norm(hull[index_hull] - hull[next_index_hull]);
        index_hull = next_index_hull;
    }
    while(index_hull != finish);

    index_hull = start;
    do
    {
        next_index_hull = index_hull - 1;
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }
        test_length[1] += norm(hull[index_hull] - hull[next_index_hull]);
        index_hull = next_index_hull;
    }
    while(index_hull != finish);

    if (test_length[0] < test_length[1]) { return true; } else { return false; }
}

vector<Point2f> QRDetect::getQuadrilateral(vector<Point2f> angle_list)
{
    CV_TRACE_FUNCTION();
    size_t angle_size = angle_list.size();
    uint8_t value, mask_value;
    Mat mask = Mat::zeros(bin_barcode.rows + 2, bin_barcode.cols + 2, CV_8UC1);
    Mat fill_bin_barcode = bin_barcode.clone();
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
                floodFill(fill_bin_barcode, mask, line_iter.pos(), 255,
                          0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);
            }
        }
    }
    vector<Point> locations;
    Mat mask_roi = mask(Range(1, bin_barcode.rows - 1), Range(1, bin_barcode.cols - 1));

    findNonZero(mask_roi, locations);

    for (size_t i = 0; i < angle_list.size(); i++)
    {
        int x = cvRound(angle_list[i].x);
        int y = cvRound(angle_list[i].y);
        locations.push_back(Point(x, y));
    }

    vector<Point> integer_hull;
    convexHull(locations, integer_hull);
    int hull_size = (int)integer_hull.size();
    vector<Point2f> hull(hull_size);
    for (int i = 0; i < hull_size; i++)
    {
        float x = saturate_cast<float>(integer_hull[i].x);
        float y = saturate_cast<float>(integer_hull[i].y);
        hull[i] = Point2f(x, y);
    }

    const double experimental_area = fabs(contourArea(hull));

    vector<Point2f> result_hull_point(angle_size);
    double min_norm;
    for (size_t i = 0; i < angle_size; i++)
    {
        min_norm = std::numeric_limits<double>::max();
        Point closest_pnt;
        for (int j = 0; j < hull_size; j++)
        {
            double temp_norm = norm(hull[j] - angle_list[i]);
            if (min_norm > temp_norm)
            {
                min_norm = temp_norm;
                closest_pnt = hull[j];
            }
        }
        result_hull_point[i] = closest_pnt;
    }

    int start_line[2] = { 0, 0 }, finish_line[2] = { 0, 0 }, unstable_pnt = 0;
    for (int i = 0; i < hull_size; i++)
    {
        if (result_hull_point[2] == hull[i]) { start_line[0] = i; }
        if (result_hull_point[1] == hull[i]) { finish_line[0] = start_line[1] = i; }
        if (result_hull_point[0] == hull[i]) { finish_line[1] = i; }
        if (result_hull_point[3] == hull[i]) { unstable_pnt = i; }
    }

    int index_hull, extra_index_hull, next_index_hull, extra_next_index_hull;
    Point result_side_begin[4], result_side_end[4];

    bool bypass_orientation = testBypassRoute(hull, start_line[0], finish_line[0]);

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[1]) >
        norm(hull[index_hull] - angle_list[2]) ? angle_list[2] : angle_list[1];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                          angle_list[1], angle_list[2]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[1] - angle_list[2]) * 0.1)
        {
            min_norm = temp_norm;
            result_side_begin[0] = hull[index_hull];
            result_side_end[0]   = hull[next_index_hull];
        }


        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[0]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[0] = angle_list[1];
        result_side_end[0]   = angle_list[2];
    }

    min_norm = std::numeric_limits<double>::max();
    index_hull = start_line[1];
    bypass_orientation = testBypassRoute(hull, start_line[1], finish_line[1]);
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        Point angle_closest_pnt =  norm(hull[index_hull] - angle_list[0]) >
        norm(hull[index_hull] - angle_list[1]) ? angle_list[1] : angle_list[0];

        Point intrsc_line_hull =
        intersectionLines(hull[index_hull], hull[next_index_hull],
                          angle_list[0], angle_list[1]);
        double temp_norm = getCosVectors(hull[index_hull], intrsc_line_hull, angle_closest_pnt);
        if (min_norm > temp_norm &&
            norm(hull[index_hull] - hull[next_index_hull]) >
            norm(angle_list[0] - angle_list[1]) * 0.05)
        {
            min_norm = temp_norm;
            result_side_begin[1] = hull[index_hull];
            result_side_end[1]   = hull[next_index_hull];
        }

        index_hull = next_index_hull;
    }
    while(index_hull != finish_line[1]);

    if (min_norm == std::numeric_limits<double>::max())
    {
        result_side_begin[1] = angle_list[0];
        result_side_end[1]   = angle_list[1];
    }

    bypass_orientation = testBypassRoute(hull, start_line[0], unstable_pnt);
    const bool extra_bypass_orientation = testBypassRoute(hull, finish_line[1], unstable_pnt);

    vector<Point2f> result_angle_list(4), test_result_angle_list(4);
    double min_diff_area = std::numeric_limits<double>::max();
    index_hull = start_line[0];
    const double standart_norm = std::max(
        norm(result_side_begin[0] - result_side_end[0]),
        norm(result_side_begin[1] - result_side_end[1]));
    do
    {
        if (bypass_orientation) { next_index_hull = index_hull + 1; }
        else { next_index_hull = index_hull - 1; }

        if (next_index_hull == hull_size) { next_index_hull = 0; }
        if (next_index_hull == -1) { next_index_hull = hull_size - 1; }

        if (norm(hull[index_hull] - hull[next_index_hull]) < standart_norm * 0.1)
        { index_hull = next_index_hull; continue; }

        extra_index_hull = finish_line[1];
        do
        {
            if (extra_bypass_orientation) { extra_next_index_hull = extra_index_hull + 1; }
            else { extra_next_index_hull = extra_index_hull - 1; }

            if (extra_next_index_hull == hull_size) { extra_next_index_hull = 0; }
            if (extra_next_index_hull == -1) { extra_next_index_hull = hull_size - 1; }

            if (norm(hull[extra_index_hull] - hull[extra_next_index_hull]) < standart_norm * 0.1)
            { extra_index_hull = extra_next_index_hull; continue; }

            test_result_angle_list[0]
            = intersectionLines(result_side_begin[0], result_side_end[0],
                                result_side_begin[1], result_side_end[1]);
            test_result_angle_list[1]
            = intersectionLines(result_side_begin[1], result_side_end[1],
                                hull[extra_index_hull], hull[extra_next_index_hull]);
            test_result_angle_list[2]
            = intersectionLines(hull[extra_index_hull], hull[extra_next_index_hull],
                                hull[index_hull], hull[next_index_hull]);
            test_result_angle_list[3]
            = intersectionLines(hull[index_hull], hull[next_index_hull],
                                result_side_begin[0], result_side_end[0]);

            const double test_diff_area
                = fabs(fabs(contourArea(test_result_angle_list)) - experimental_area);
            if (min_diff_area > test_diff_area)
            {
                min_diff_area = test_diff_area;
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

    // check label points
    if (norm(result_angle_list[0] - angle_list[1]) > 2) { result_angle_list[0] = angle_list[1]; }
    if (norm(result_angle_list[1] - angle_list[0]) > 2) { result_angle_list[1] = angle_list[0]; }
    if (norm(result_angle_list[3] - angle_list[2]) > 2) { result_angle_list[3] = angle_list[2]; }

    // check calculation point
    if (norm(result_angle_list[2] - angle_list[3]) >
       (norm(result_angle_list[0] - result_angle_list[1]) +
        norm(result_angle_list[0] - result_angle_list[3])) * 0.5 )
    { result_angle_list[2] = angle_list[3]; }

    return result_angle_list;
}

//      / | b
//     /  |
//    /   |
//  a/    | c

inline double QRDetect::getCosVectors(Point2f a, Point2f b, Point2f c)
{
    return ((a - b).x * (c - b).x + (a - b).y * (c - b).y) / (norm(a - b) * norm(c - b));
}

bool QRDetect::nextSet(int *set, int n, int m)
{

  int k = m;
  for (int i = k - 1; i >= 0; --i)
    if (set[i] < n - k + i )
    {
      ++set[i];
      for (int j = i + 1; j < k; ++j)
        set[j] = set[j - 1] + 1;
      return true;
    }
  return false;
}

struct QRCodeDetector::Impl
{
public:
    Impl() { epsX = 0.2; epsY = 0.1; }
    ~Impl() {}

    double epsX, epsY;
};

QRCodeDetector::QRCodeDetector() : p(new Impl) {}
QRCodeDetector::~QRCodeDetector() {}



void QRCodeDetector::setEpsX(double epsX) { p->epsX = epsX; }
void QRCodeDetector::setEpsY(double epsY) { p->epsY = epsY; }

bool QRCodeDetector::detect(InputArray in, OutputArrayOfArrays points) const
{
  Mat inarr = in.getMat();
  CV_Assert(!inarr.empty());
  CV_Assert(inarr.depth() == CV_8U);
  if (inarr.cols <= 20 || inarr.rows <= 20)
      return false;  // image data is not enough for providing reliable results

  int incn = inarr.channels();
  if( incn == 3 || incn == 4 )
  {
      Mat gray;
      cvtColor(inarr, gray, COLOR_BGR2GRAY);
      inarr = gray;
  }
  QRDetect qrdet;
  qrdet.init(inarr, p->epsX, p->epsY);
  if (!qrdet.localization()) { return false; }
  vector<vector<Point2f>> pnts2f = qrdet.getTransformationPoints();
  vector<Mat> tempPoints;

  for(size_t i = 0; i < pnts2f.size(); i++)
  {
     Mat tempMat;
     tempPoints.push_back(tempMat);
     Mat(pnts2f[i]).convertTo(tempPoints[i], OutputArray(tempPoints[i]).fixedType() ? OutputArray(tempPoints[i]).type() : CV_32FC2);

  }
  points.createSameSize(tempPoints, CV_32FC2);
  points.assign(tempPoints);
  return true;
};


class QRDecode
{
public:
    void init(const Mat &src, const vector<vector<Point2f>> &points);
    vector<Mat>  getIntermediateBarcode() { return intermediate; }
    vector<uint8_t> getVersion() { return version; }
    vector<Mat> getStraightBarcode() { return straight; }
    vector<std::string> getDecodeInformation() { return result_info; }
    bool fullDecodingProcess();
protected:
    bool updatePerspective();
    bool versionDefinition();
    bool samplingForVersion();
    bool decodingProcess();
    Mat original;
    vector<Mat> no_border_intermediate, intermediate, straight;
    vector<vector<Point2f>> original_points;
    vector<std::string> result_info;

    vector<uint8_t> version, version_size;
    vector <float> test_perspective_size;

};



void QRDecode::init(const Mat &src, const vector<vector<Point2f>> &points)
{
      CV_TRACE_FUNCTION();
      original = src.clone();
      for(size_t i = 0; i < points.size(); i++)
      {
          original_points.push_back(points[i]);
          no_border_intermediate.push_back(Mat::zeros(src.size(), CV_8UC1));
          intermediate.push_back(Mat::zeros(src.size(), CV_8UC1));
          straight.push_back(Mat::zeros(src.size(), CV_8UC1));
          version.push_back(0);
          version_size.push_back(0);
          test_perspective_size.push_back(251);
          result_info.push_back("");
     }
}

bool QRDecode::updatePerspective()
{
    CV_TRACE_FUNCTION();
    bool flag = false;
    parallel_for_(Range(0, original_points.size()), [&] (const Range& range)
    {
        for(int i = range.start; i < range.end; i++)
        {
            const Point2f centerPt = QRDetect::intersectionLines(original_points[i][0], original_points[i][2],
                                                             original_points[i][1], original_points[i][3]);

            if (cvIsNaN(centerPt.x) || cvIsNaN(centerPt.y))
            {
                flag = true;
                break;
            }
            const Size temporary_size(cvRound(test_perspective_size[i]), cvRound(test_perspective_size[i]));

            vector<Point2f> perspective_points;
            perspective_points.push_back(Point2f(0.f, 0.f));
            perspective_points.push_back(Point2f(test_perspective_size[i], 0.f));

            perspective_points.push_back(Point2f(test_perspective_size[i], test_perspective_size[i]));
            perspective_points.push_back(Point2f(0.f, test_perspective_size[i]));

            perspective_points.push_back(Point2f(test_perspective_size[i] * 0.5f, test_perspective_size[i] * 0.5f));

            vector<Point2f> pts = original_points[i];
            pts.push_back(centerPt);

            Mat H = findHomography(pts, perspective_points);
            Mat bin_original;
            adaptiveThreshold(original, bin_original, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
            Mat temp_intermediate;
            warpPerspective(bin_original, temp_intermediate, H, temporary_size, INTER_NEAREST);
            no_border_intermediate[i] = temp_intermediate(Range(1, temp_intermediate.rows), Range(1, temp_intermediate.cols));

            const int border = cvRound(0.1 * test_perspective_size[i]);
            const int borderType = BORDER_CONSTANT;
            copyMakeBorder(no_border_intermediate[i], intermediate[i], border, border, border, border, borderType, Scalar(255));

        }
      });
    if (flag) return false;
    else return  true;
}

inline Point computeOffset(const vector<Point>& v)
{
    // compute the width/height of convex hull
    Rect areaBox = boundingRect(v);

    // compute the good offset
    // the box is consisted by 7 steps
    // to pick the middle of the stripe, it needs to be 1/14 of the size
    const int cStep = 7 * 2;
    Point offset = Point(areaBox.width, areaBox.height);
    offset /= cStep;
    return offset;
}

bool QRDecode::versionDefinition()
{
    CV_TRACE_FUNCTION();
    bool flag = false;
    parallel_for_(Range(0, original_points.size()), [&](Range range)
    {
        for(int c = range.start; c < range.end; c++)
        {
          LineIterator line_iter(intermediate[c], Point2f(0, 0), Point2f(test_perspective_size[c], test_perspective_size[c]));
          Point black_point = Point(0, 0);
          for(int j = 0; j < line_iter.count; j++, ++line_iter)
          {
              const uint8_t value = intermediate[c].at<uint8_t>(line_iter.pos());
              if (value == 0) { black_point = line_iter.pos(); break; }
          }

          Mat mask = Mat::zeros(intermediate[c].rows + 2, intermediate[c].cols + 2, CV_8UC1);
          floodFill(intermediate[c], mask, black_point, 255, 0, Scalar(), Scalar(), FLOODFILL_MASK_ONLY);

          vector<Point> locations, non_zero_elem;
          Mat mask_roi = mask(Range(1, intermediate[c].rows - 1), Range(1, intermediate[c].cols - 1));
          findNonZero(mask_roi, non_zero_elem);
          convexHull(non_zero_elem, locations);
          Point offset = computeOffset(locations);

          Point temp_remote = locations[0], remote_point;
          const Point delta_diff = offset;
          for (size_t i = 0; i < locations.size(); i++)
          {
              if (norm(black_point - temp_remote) <= norm(black_point - locations[i]))
              {
                  const uint8_t value = intermediate[c].at<uint8_t>(temp_remote - delta_diff);
                  temp_remote = locations[i];
                  if (value == 0) { remote_point = temp_remote - delta_diff; }
                  else { remote_point = temp_remote - (delta_diff / 2); }
              }
          }

            size_t transition_x = 0 , transition_y = 0;

            uint8_t future_pixel = 255;
            const uint8_t *intermediate_row = intermediate[c].ptr<uint8_t>(remote_point.y);
            for(int i = remote_point.x; i < intermediate[c].cols; i++)
            {
              if (intermediate_row[i] == future_pixel)
              {
                  future_pixel = 255 - future_pixel;
                  transition_x++;
              }
            }

            future_pixel = 255;
            for(int j = remote_point.y; j < intermediate[c].rows; j++)
            {
              const uint8_t value = intermediate[c].at<uint8_t>(Point(j, remote_point.x));
              if (value == future_pixel)
              {
                  future_pixel = 255 - future_pixel;
                  transition_y++;
                }
              }

              version[c] = saturate_cast<uint8_t>((std::min(transition_x, transition_y) - 1) * 0.25 - 1);
              if ( !(  0 < version[c] && version[c] <= 40 ) )
              {
                flag = true;
                break;
              }
              version_size[c] = 21 + (version[c] - 1) * 4;
      }
    });
    if(flag) return false;
    else return true;
}

bool QRDecode::samplingForVersion()
{
    CV_TRACE_FUNCTION();
    parallel_for_(Range(0, original_points.size()), [&](Range range)
    {
        for(int q = range.start; q < range.end; q++)
        {
          const double multiplyingFactor = (version[q] < 3)  ? 1 :
                                          (version[q] == 3) ? 1.5 :
                                         version[q] * (5 + version[q] - 4);
          const Size newFactorSize(
                        cvRound(no_border_intermediate[q].size().width  * multiplyingFactor),
                        cvRound(no_border_intermediate[q].size().height * multiplyingFactor));
          Mat postIntermediate(newFactorSize, CV_8UC1);
          resize(no_border_intermediate[q], postIntermediate, newFactorSize, 0, 0, INTER_AREA);

          const int delta_rows = cvRound((postIntermediate.rows * 1.0) / version_size[q]);
          const int delta_cols = cvRound((postIntermediate.cols * 1.0) / version_size[q]);

          vector<double> listFrequencyElem;
          for (int r = 0; r < postIntermediate.rows; r += delta_rows)
          {
              for (int c = 0; c < postIntermediate.cols; c += delta_cols)
              {
                  Mat tile = postIntermediate(
                                Range(r, min(r + delta_rows, postIntermediate.rows)),
                                Range(c, min(c + delta_cols, postIntermediate.cols)));
                  const double frequencyElem = (countNonZero(tile) * 1.0) / tile.total();
                  listFrequencyElem.push_back(frequencyElem);
                }
          }

          double dispersionEFE = std::numeric_limits<double>::max();
          double experimentalFrequencyElem = 0;
          for (double expVal = 0; expVal < 1; expVal+=0.001)
          {
              double testDispersionEFE = 0.0;
              for (size_t i = 0; i < listFrequencyElem.size(); i++)
              {
                  testDispersionEFE += (listFrequencyElem[i] - expVal) *
                                     (listFrequencyElem[i] - expVal);
              }
              testDispersionEFE /= (listFrequencyElem.size() - 1);
              if (dispersionEFE > testDispersionEFE)
              {
                  dispersionEFE = testDispersionEFE;
                  experimentalFrequencyElem = expVal;
              }
          }

          straight[q] = Mat(Size(version_size[q], version_size[q]), CV_8UC1, Scalar(0));
          for (int r = 0; r < version_size[q] * version_size[q]; r++)
          {
              int i   = r / straight[q].cols;
              int j   = r % straight[q].cols;
              straight[q].ptr<uint8_t>(i)[j] = (listFrequencyElem[r] < experimentalFrequencyElem) ? 0 : 255;
          }
        }
      });
    return true;
}

bool QRDecode::decodingProcess()
{
  bool flag = false;
    parallel_for_(Range(0, original_points.size()), [&](Range range)
    {
        for(int c = range.start; c < range.end; c++)
        {
          if (straight[c].empty())
          {
            flag = true;
            break;
           }

          quirc_code qr_code;
          memset(&qr_code, 0, sizeof(qr_code));

          qr_code.size = straight[c].size().width;
          for (int x = 0; x < qr_code.size; x++)
          {
              for (int y = 0; y < qr_code.size; y++)
              {
                  int position = y * qr_code.size + x;
                  qr_code.cell_bitmap[position >> 3]
                      |= straight[c].ptr<uint8_t>(y)[x] ? 0 : (1 << (position & 7));
              }
          }

          quirc_data qr_code_data;
          quirc_decode_error_t errorCode = quirc_decode(&qr_code, &qr_code_data);
          if (errorCode != 0)
          {
             flag = true;
             break;
          }

          for (int i = 0; i < qr_code_data.payload_len; i++)
          {
              result_info[c] += qr_code_data.payload[i];
          }

        }
  });
  if(flag) return false;
  else return true;

}

bool QRDecode::fullDecodingProcess()
{
#ifdef HAVE_QUIRC
    if (!updatePerspective())  { return false; }
    if (!versionDefinition())  { return false; }
    if (!samplingForVersion()) { return false; }
    if (!decodingProcess())    { return false; }
    return true;
#else
    std::cout << "Library QUIRC is not linked. No decoding is performed. Take it to the OpenCV repository." << std::endl;
    return false;
#endif
}


vector<cv::String> QRCodeDetector::decode(InputArray in, InputArrayOfArrays points,
                                   OutputArrayOfArrays straight_qrcode)
{
    Mat inarr = in.getMat();
    CV_Assert(!inarr.empty());
    CV_Assert(inarr.depth() == CV_8U);
    if (inarr.cols <= 20 || inarr.rows <= 20)
    {
      return vector<cv::String>(); // image data is not enough for providing reliable results
    }
    int incn = inarr.channels();
    if( incn == 3 || incn == 4 )
    {
        Mat gray;
        cvtColor(inarr, gray, COLOR_BGR2GRAY);
        inarr = gray;
    }

    vector<vector<Point2f>> src_points ;
    bool flag = true;
    for(int i = 0; i < points.size().width; i++)
    {
      vector<Point2f> tempMat = points.getMat(i);

      if(tempMat.size() != 4)
      {
          flag = false;
      }
      if ((flag == true) && (contourArea(tempMat) == 0.0)){
      flag = false;}
      if(flag == true) src_points.push_back(points.getMat(i));
      flag = true;
    }
    CV_Assert(src_points.size() > 0);
    QRDecode qrdec, qrdec2;
    qrdec.init(inarr, src_points);
    bool ok = qrdec.fullDecodingProcess();
    if ((!ok) && (std::min(inarr.size().width, inarr.size().height) > 512))
    {
        const int min_side = std::min(inarr.size().width, inarr.size().height);
        double coeff_expansion = min_side / 512;
        const int width  = cvRound(inarr.size().width  / coeff_expansion);
        const int height = cvRound(inarr.size().height / coeff_expansion);
        Size new_size(width, height);
        Mat inarr2;
        resize(inarr, inarr2, new_size, 0, 0, INTER_AREA);
        for (size_t i = 0; i < src_points.size(); i++)
        {
            for(size_t j = 0; j < 4; j++)
                src_points[i][j] /= static_cast<float>(coeff_expansion);
        }
        qrdec2.init(inarr2, src_points);
        ok = qrdec2.fullDecodingProcess();
        vector<std::string> decoded_info = qrdec2.getDecodeInformation();
        vector<Mat> straight_barcode = qrdec2.getStraightBarcode();
    }
    vector<std::string> decoded_info = qrdec.getDecodeInformation();
    vector<Mat> straight_barcode = qrdec.getStraightBarcode();
    vector<Mat> tmp_straight_qrcodes;
    for(size_t i = 0; i < straight_barcode.size(); i++)
    {
        if (ok && ((OutputArray)(straight_barcode[i])).needed())
        {

            Mat tmp_straight_qrcode;
            tmp_straight_qrcodes.push_back(tmp_straight_qrcode);
            straight_barcode[i].convertTo(((OutputArray)tmp_straight_qrcodes[i]),
                                             ((OutputArray)tmp_straight_qrcodes[i]).fixedType() ?
                                             ((OutputArray)tmp_straight_qrcodes[i]).type() : CV_32FC2);
       }
    }
    straight_qrcode.createSameSize(tmp_straight_qrcodes, CV_32FC2);
    straight_qrcode.assign(tmp_straight_qrcodes);
    vector<cv::String> result;
    for(size_t i = 0; i < decoded_info.size(); i++)
    {
       result.push_back(decoded_info[i]);
    }
    return result;

}



vector<cv::String> QRCodeDetector::detectAndDecode(InputArray in,
                                            OutputArrayOfArrays points_,
                                            OutputArrayOfArrays straight_qrcode)
{
    Mat inarr = in.getMat();
    CV_Assert(!inarr.empty());
    CV_Assert(inarr.depth() == CV_8U);

    if (inarr.cols <= 20 || inarr.rows <= 20)
    {
        vector<cv::String> str;
        return str;  // image data is not enough for providing reliable results
    }

    int incn = inarr.channels();
    if( incn == 3 || incn == 4 )
    {
        Mat gray;
        cvtColor(inarr, gray, COLOR_BGR2GRAY);
        inarr = gray;
    }

    vector<Mat> points, tempPoints;
    bool ok = detect(inarr, points);
    for(size_t i = 0; i < points.size(); i++)
    {
      if(OutputArray(points[i]).needed())
      {
        if( ok )
        {
            Mat tmp;
            tempPoints.push_back(tmp);
            Mat(points[i]).copyTo(tempPoints[i]);
        }
        else
          ((OutputArray)tempPoints[i]).release();
        }
    }
    points_.createSameSize(tempPoints, CV_32FC2);
    points_.assign(tempPoints);
    vector<cv::String> decoded_info;
    if(ok)
        decoded_info = decode(inarr, points, straight_qrcode);
    return decoded_info;
}


}
