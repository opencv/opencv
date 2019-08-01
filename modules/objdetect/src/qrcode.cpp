// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#ifdef HAVE_QUIRC
#include "quirc.h"
#endif

#include <limits>
#include <cmath>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <stdlib.h>
namespace cv
{
using std::vector;
/*bool compare_points2f(Point2f a,Point2f b)
{
    return a.y<b.y
}
*/
struct area
{
  Point2f point;
  double local_max_area;
};

float ar (Point2f a, Point2f b, Point2f c) {
	return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool intersect_1 (float a, float b, float c, float d) {
	if (a > b)  std::swap (a, b);
	if (c > d)  std::swap (c, d);
	return std::max(a,c) <= std::min(b,d);
}

bool intersect (Point2f a, Point2f b, Point2f c, Point2f d) {
	return intersect_1 (a.x, b.x, c.x, d.x)
		&& intersect_1 (a.y, b.y, c.y, d.y)
		&& ar(a,b,c) * ar(a,b,d) <= 0
		&& ar(c,d,a) * ar(c,d,b) <= 0;
}

//vector<Point2f> getList_lines_y() {return list_lines_y;}
class QRDetect
{
public:
    void init(const Mat& src, double eps_vertical_ = 0.2, double eps_horizontal_ = 0.1);
    bool localization();
    bool computeTransformationPoints();
    Mat getBinBarcode() { return bin_barcode; }
    Mat getStraightBarcode() { return straight_barcode; }

    //vector<Point2f> getTransformationPoints() { return transformation_points[0]; }
    vector<vector<Point2f>> getTransformationPoints() { return transformation_points; }
    static Point2f intersectionLines(Point2f a1, Point2f a2, Point2f b1, Point2f b2);

public:

    vector<Vec3d> searchHorizontalLines();
    vector<Point2f> separateVerticalLines(const vector<Vec3d> &list_lines);
    void fixationPoints(vector<Point2f> &local_point);
    void fixationPoints(vector<vector<Point2f>>&local_point);
    vector<Point2f> getQuadrilateral(vector<Point2f> angle_list);
    bool testBypassRoute(vector<Point2f> hull, int start, int finish);
    inline double getCosVectors(Point2f a, Point2f b, Point2f c);

    Mat barcode, bin_barcode, straight_barcode;
    vector<vector<Point2f>> localization_points, transformation_points;
    double eps_vertical, eps_horizontal, coeff_expansion;
};


void QRDetect::init(const Mat& src, double eps_vertical_, double eps_horizontal_)
{
    CV_TRACE_FUNCTION();
    CV_Assert(!src.empty());
    const double min_side = std::min(src.size().width, src.size().height);
    if (min_side < 512.0)
    {
        coeff_expansion = 512.0 / min_side;
        const int width  = cvRound(src.size().width  * coeff_expansion);
        const int height = cvRound(src.size().height  * coeff_expansion);
        Size new_size(width, height);
        resize(src, barcode, new_size, 0, 0, INTER_LINEAR);
    }
    else
    {
        coeff_expansion = 1.0;
        barcode = src;
    }

    eps_vertical   = eps_vertical_;
    eps_horizontal = eps_horizontal_;
    adaptiveThreshold(barcode, bin_barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 83, 2);
    //mycode
    //std::cout<<bin_barcode;
    //imshow("MyWindow", barcode);
   // waitKey(100);
    //endmycode
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
    int temp_length = 0;
    uint8_t next_pixel;
    vector<double> test_lines;


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

    vector<Point2f> point2f_result;
    for (size_t i = 0; i < result.size(); i++)
    {
        point2f_result.push_back(
              Point2f(static_cast<float>(result[i][0] + result[i][2] * 0.5),
                      static_cast<float>(result[i][1])));
    }
    return point2f_result;
}

/*void QRDetect::fixationPoints(vector<Point2f> &local_point)
{
    CV_TRACE_FUNCTION();
    int n=2;
    double cos_angles[2][3], norm_triangl[2][3];
    //std::cout<<local_point[0]<<" "<<local_point[1]<<" "<<local_point[2]<<std::endl;
    std::sort (local_point.begin(), local_point.begin()+local_point.size(),
	      [](Point2f& a,Point2f& b)
	      {
 		   return (sqrt(a.x*a.x+a.y*a.y))<(sqrt(b.x*b.x+b.y*b.y));
	      });
    for(int i=0;i<6;i++)std::cout<<local_point[i]<<" "; std::cout<<std::endl;
    //std::cout<<local_point[0]<<" "<<local_point[1]<<" "<<local_point[2]<<std::endl;
    /*if(local_point.size()>3)
    {
        local_point.pop_back();
        local_point.pop_back();
        local_point.pop_back();

    //for(int i=0;i<local_point.size();i++)
    //for(int c=0;c<(int)(local_point.size()/3);c++)
    for(int c=0;c<2;c++)
    {
   	 norm_triangl[c][0] = norm(local_point[3*c+1] - local_point[3*c+2]);
   	 norm_triangl[c][1] = norm(local_point[3*c] - local_point[3*c+2]);
  	 norm_triangl[c][2] = norm(local_point[3*c+1] - local_point[3*c]);

         cos_angles[c][0] = (norm_triangl[c][1] * norm_triangl[c][1] + norm_triangl[c][2] * 			        	norm_triangl[c][2] -  norm_triangl[c][0] * norm_triangl[c][0]) /  					(2 * norm_triangl[c][1] * norm_triangl[c][2]);
   	 cos_angles[c][1] = (norm_triangl[c][0] * norm_triangl[c][0] + norm_triangl[c][2] * 				norm_triangl[c][2]
              	    -  norm_triangl[c][1] * norm_triangl[c][1]) / (2 * norm_triangl[c][0] * 				norm_triangl[c][2]);
  	  cos_angles[c][2] = (norm_triangl[c][0] * norm_triangl[c][0] + norm_triangl[c][1] * 					norm_triangl[c][1]
                		  -  norm_triangl[c][2] * norm_triangl[c][2]) / (2 * 					norm_triangl[c][0] * norm_triangl[c][1]);

    	const double angle_barrier = 0.85;
   	 if (fabs(cos_angles[c][0]) > angle_barrier || fabs(cos_angles[c][1]) > angle_barrier || fabs(cos_angles[c][2]) > angle_barrier)
   	 {
      	     local_point.clear();
     	     return;

  	  }
   	 size_t i_min_cos =
       (cos_angles[c][0] < cos_angles[c][1] && cos_angles[c][0] < cos_angles[c][2]) ? (c*3) :
       (cos_angles[c][1] < cos_angles[c][0] && cos_angles[c][1] < cos_angles[c][2]) ? (c*3+1) : (c*3+2);

   	 size_t index_max = 3*c;
    	double max_area = std::numeric_limits<double>::min();
   	 for (size_t i = c*3; i < (c*3+3); i++)
   	 {
       	    const size_t current_index = (i % 3)+3*c;
     	    const size_t left_index  = ((i + 1) % 3)+3*c;
      	    const size_t right_index = ((i + 2) % 3)+3*c;

            const Point2f current_point(local_point[current_index]),
            left_point(local_point[left_index]), right_point(local_point[right_index]),
            central_point(intersectionLines(current_point,
                              Point2f(static_cast<float>((local_point[left_index].x + 					local_point[right_index].x) * 0.5),
                                      static_cast<float>((local_point[left_index].y + 					local_point[right_index].y) * 0.5)),
                              Point2f(0, static_cast<float>(bin_barcode.rows - 1)),
                              Point2f(static_cast<float>(bin_barcode.cols - 1),
                                      static_cast<float>(bin_barcode.rows - 1))));


            vector<Point2f> list_area_pnt;
            list_area_pnt.push_back(current_point);

            vector<LineIterator> list_line_iter;
            list_line_iter.push_back(LineIterator(bin_barcode, current_point, left_point));
            list_line_iter.push_back(LineIterator(bin_barcode, current_point, central_point));
            list_line_iter.push_back(LineIterator(bin_barcode, current_point, right_point));

            for (size_t k = 0; k < list_line_iter.size(); k++)
            {
                 uint8_t future_pixel = 255, count_index = 0;
                 for(int j = 0; j < list_line_iter[k].count; j++, ++list_line_iter[k])
                 {
                      if (list_line_iter[k].pos().x >= bin_barcode.cols ||
                      list_line_iter[k].pos().y >= bin_barcode.rows) { break; }
                      const uint8_t value = bin_barcode.at<uint8_t>(list_line_iter[k].pos());
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
       std::cout<<index_max<<" "<<local_point[3*c]<<" "<<local_point[3*c+1]<<" "<<local_point[3*c+2]<<std::endl;
       if (index_max == i_min_cos) { std::swap(local_point[3*c], local_point[index_max]); }
       else {local_point.clear(); return; }

       const Point2f rpt = local_point[3*c], bpt = local_point[3*c+1], gpt = local_point[3*c+2];
       Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
       if( determinant(m) > 0 )
       {
          std::swap(local_point[3*c+1], local_point[3*c+2]);
       }
    }
}
*/
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
        return ;
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
                              Point2f(0, static_cast<float>(bin_barcode.rows - 1)),
                              Point2f(static_cast<float>(bin_barcode.cols - 1),
                                      static_cast<float>(bin_barcode.rows - 1))));


        vector<Point2f> list_area_pnt;
        list_area_pnt.push_back(current_point);

        vector<LineIterator> list_line_iter;
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, left_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, central_point));
        list_line_iter.push_back(LineIterator(bin_barcode, current_point, right_point));

        for (size_t k = 0; k < list_line_iter.size(); k++)
        {
            uint8_t future_pixel = 255, count_index = 0;
            for(int j = 0; j < list_line_iter[k].count; j++, ++list_line_iter[k])
            {
                if (list_line_iter[k].pos().x >= bin_barcode.cols ||
                    list_line_iter[k].pos().y >= bin_barcode.rows) { break; }
                const uint8_t value = bin_barcode.at<uint8_t>(list_line_iter[k].pos());
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
    else { local_point.clear(); return ; }

    const Point2f rpt = local_point[0], bpt = local_point[1], gpt = local_point[2];
    Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
    if( determinant(m) > 0 )
    {
        std::swap(local_point[1], local_point[2]);
    }
    return ;
}

void QRDetect::fixationPoints(vector<vector<Point2f>>&local_point)
{
    CV_TRACE_FUNCTION();
    //int n=2;
    //double cos_angles[2][3], norm_triangl[2][3];
    //std::cout<<local_point[0]<<" "<<local_point[1]<<" "<<local_point[2]<<std::endl;
    /*std::sort (tmp_local_point.begin(), tmp_local_point.begin()+tmp_local_point.size(),
	      [](Point2f& a,Point2f& b)
	      {
 		         return (sqrt(a.x*a.x+a.y*a.y))<(sqrt(b.x*b.x+b.y*b.y));
	      });
  //  for(int i=0;i<6;i++)std::cout<<local_point[i]<<" "; std::cout<<std::endl;
    //for(size_t i=0;i<tmp_local_point.size()/3;i++)
    //    std::copy(tmp_local_point[i*3],tmp_local_point[i*3+3],local_point[i]);
    for(size_t i=0;i<6;i++)
    std::cout<<tmp_local_point[i]<<std::endl;
    for(size_t i=0;i<tmp_local_point.size()/3;i++)
    {
        vector<Point2f>tmp;
        for(size_t j=0;j<3;j++)
            {
                tmp.push_back(tmp_local_point[i*3+j]);
            }
        local_point.push_back(tmp);
    }*/
    //std::cout<<local_point[0]<<" "<<local_point[1]<<" "<<local_point[2]<<std::endl;
    /*if(local_point.size()>3)
    {
        local_point.pop_back();
        local_point.pop_back();
        local_point.pop_back();
    }*/
    //for(int i=0;i<local_point.size();i++)
    //for(int c=0;c<(int)(local_point.size()/3);c++)

    for(size_t c=0;c<local_point.size();c++)
    {

      double cos_angles[3], norm_triangl[3];
      norm_triangl[0] = norm(local_point[c][1] - local_point[c][2]);
      norm_triangl[1] = norm(local_point[c][0] - local_point[c][2]);
      norm_triangl[2] = norm(local_point[c][1] - local_point[c][0]);

      cos_angles[0] = (norm_triangl[1] * norm_triangl[1] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[0] * norm_triangl[0]) / (2 * norm_triangl[1] * norm_triangl[2]);
      cos_angles[1] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[2] * norm_triangl[2]
                  -  norm_triangl[1] * norm_triangl[1]) / (2 * norm_triangl[0] * norm_triangl[2]);
      cos_angles[2] = (norm_triangl[0] * norm_triangl[0] + norm_triangl[1] * norm_triangl[1]
                  -  norm_triangl[2] * norm_triangl[2]) / (2 * norm_triangl[0] * norm_triangl[1]);

    	const double angle_barrier = 0.85;
   	  if (fabs(cos_angles[0]) > angle_barrier || fabs(cos_angles[1]) > angle_barrier || fabs(cos_angles[2]) > angle_barrier)
   	  {

      	   local_point[c].clear();
           local_point.erase(local_point.begin()+c);
           c--;
     	     break;

  	  }
   	  size_t i_min_cos =
       (cos_angles[0] < cos_angles[1] && cos_angles[0] < cos_angles[2]) ? 0:
       (cos_angles[1] < cos_angles[0] && cos_angles[1] < cos_angles[2]) ? 1:2;

   	   size_t index_max = 0;
       double max_area = std::numeric_limits<double>::min();
   	   for (size_t i = 0; i < local_point[c].size(); i++)
   	   {
       	   const size_t current_index = i % 3;
     	     const size_t left_index  = (i + 1) % 3;
      	   const size_t right_index = (i + 2) % 3;

           const Point2f current_point(local_point[c][current_index]),
           left_point(local_point[c][left_index]), right_point(local_point[c][right_index]),
           central_point(intersectionLines(current_point,
                              Point2f(static_cast<float>((local_point[c][left_index].x + local_point[c][right_index].x) * 0.5),
                                      static_cast<float>((local_point[c][left_index].y + local_point[c][right_index].y) * 0.5)),
                              Point2f(0, static_cast<float>(bin_barcode.rows - 1)),
                              Point2f(static_cast<float>(bin_barcode.cols - 1),
                                      static_cast<float>(bin_barcode.rows - 1))));


            vector<Point2f> list_area_pnt;
            list_area_pnt.push_back(current_point);

            vector<LineIterator> list_line_iter;
            list_line_iter.push_back(LineIterator(bin_barcode, current_point, left_point));
            list_line_iter.push_back(LineIterator(bin_barcode, current_point, central_point));
            list_line_iter.push_back(LineIterator(bin_barcode, current_point, right_point));

            for (size_t k = 0; k < list_line_iter.size(); k++)
            {
                 uint8_t future_pixel = 255, count_index = 0;
                 for(int j = 0; j < list_line_iter[k].count; j++, ++list_line_iter[k])
                 {
                      if (list_line_iter[k].pos().x >= bin_barcode.cols ||
                      list_line_iter[k].pos().y >= bin_barcode.rows) { break; }
                      const uint8_t value = bin_barcode.at<uint8_t>(list_line_iter[k].pos());
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
       //std::cout<<index_max<<" "<<local_point[3*c]<<" "<<local_point[3*c+1]<<" "<<local_point[3*c+2]<<std::endl;
       if (index_max == i_min_cos) { std::swap(local_point[c][0], local_point[c][index_max]); }
       else {local_point[c].clear(); local_point.erase(local_point.begin()+c); c--;break; }


       const Point2f rpt = local_point[c][0], bpt = local_point[c][1], gpt = local_point[c][2];
       Matx22f m(rpt.x - bpt.x, rpt.y - bpt.y, gpt.x - rpt.x, gpt.y - rpt.y);
       if( determinant(m) > 0 )
       {
          std::swap(local_point[c][1], local_point[c][2]);
       }

    }


}

void swap(int *a, int i, int j)
{
  int s = a[i];
  a[i] = a[j];
  a[j] = s;
}
bool NextSet(int *a, int n)
{
  //std::cout << "Look for combination...\n";
  int j = n - 2;
  while (j != -1 && a[j] >= a[j + 1]) j--;
  if (j == -1)
    return false; // больше перестановок нет
  int k = n - 1;
  while (a[j] >= a[k]) k--;

  std::swap(a[j], a[k]);
  int l = j + 1, r = n - 1; // сортируем оставшуюся часть последовательности
  while (l<r)
    std::swap(a[l++],a[r--] );
  if((j/3)!=(k/3))
      return true;
  else return NextSet(a, n);
  //return true;
}

bool NextSet(int *a, int n, int m)
{

  int k = m;
  for (int i = k - 1; i >= 0; --i)
    if (a[i] < n - k + i )
    {
      ++a[i];
      for (int j = i + 1; j < k; ++j)
        a[j] = a[j - 1] + 1;
      return true;
    }
  return false;
}

double triangle_area(Point2f a, Point2f b, Point2f c)
{

  double s=0.5 * ( b.x-a.x)*(c.y-a.y)-(c.x-a.x)*(b.y-a.y);
  if (s<0) s=-1*s;
  return s;
}

double length(Point2f a, Point2f b)
{
  double l=sqrt((b.x-a.x)*(b.x-a.x)+(b.y-a.y)*(b.y-a.y));
  return l;
}

double perimeter(Point2f a, Point2f b, Point2f c)
{
  double p=length(a,b) + length(b,c) + length(a,c);
  return p;
}
bool QRDetect::localization()
{
    CV_TRACE_FUNCTION();
    Point2f begin, end;
    vector<Vec3d> list_lines_x = searchHorizontalLines();
    if( list_lines_x.empty() ) { return false; }
    std::cout<<"list_lines_x.size()  "<<list_lines_x.size()<<std::endl;
    vector<Point2f> list_lines_y = separateVerticalLines(list_lines_x);
    if( list_lines_y.size() < 3 ) { return false; }
    std::cout<<"list_lines_y.size()  "<<list_lines_y<<std::endl;

    for (int i = 0; i < list_lines_y.size(); i++)
    {
      //circle(barcode, list_lines_y[i], 5, Scalar(255, 255, 255));
    }

    //vector<Point2f> list_lines_y = separateVerticalLines(list_lines_x);
    int number=1;

    Mat labels;
    //std::cout<<localization_points.size()<<std::endl;
    vector<Point2f> tmp_localization_points;
    double mera, new_mera;
    vector<double> frac;
    //Mat centers;
    for(number=1; number<list_lines_y.size() ; number++)
    {
        if(number == 1)
        {
           mera = kmeans(list_lines_y,number, labels,
              TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
              number, KMEANS_PP_CENTERS, tmp_localization_points);
           number++;
        }
        else mera=new_mera;

        new_mera = kmeans(list_lines_y,number, labels,
              TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
              number, KMEANS_PP_CENTERS, tmp_localization_points);
        frac.push_back(mera/new_mera);
        //std :: cout << mera << std::endl;
std :: cout << number << ": " << mera/new_mera << std :: endl;
      /*  if(frac.size() >= 3 )
        {
            std :: cout << frac[frac.size()-3] << ' ' << frac[frac.size()-2] << ' ' << frac [frac.size()-1] <<std :: endl;
            if(( frac[frac.size()-3] < frac[frac.size()-2]) && (frac[frac.size()-2] > frac [frac.size()-1] ))
            {

              number--;
              break;
            }

          //  if (( frac[frac.size()-3] > frac[frac.size()-2]) && (frac[frac.size()-2] < frac [frac.size()-1] ))
        //    {
        //      number=1;
          //    break;
        //    }

      }*/
    //std::cout<<tmp_localization_points.size()<<std::endl;
        //std :: cout << "Mera: " << mera << std :: endl;

    }
    size_t number_points=0;
    for(size_t i=0; i<frac.size();i++)
    {
      if(frac[i]>=frac[number_points]) number_points=i;
    }


   std :: cout << "number of qr-codes points is " << number_points+2 << std :: endl;
  // if(((max_index+2)%3)!=0) return false;
   number_points+=2;
   number = number_points/3;
   if(number_points < 3) return false;
   std::cout << "number is " << number << std::endl;
   kmeans(list_lines_y, number_points, labels,
      TermCriteria( TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1),
    number_points, KMEANS_PP_CENTERS, tmp_localization_points);

   std::sort (tmp_localization_points.begin(), tmp_localization_points.begin()+tmp_localization_points.size(),
	      [](Point2f& a,Point2f& b)
	      {
 		         return (sqrt(a.x*a.x+a.y*a.y))<(sqrt(b.x*b.x+b.y*b.y));
	      });
   std::cout << tmp_localization_points.size() << std::endl;
   for(size_t k = 0; k < tmp_localization_points.size(); k++)
   {
       circle(barcode, tmp_localization_points[k], 10, Scalar(255, 255, 255));
       std::cout << tmp_localization_points[k] << " ";
  }
  std::cout << std::endl;
  //  for(int i=0;i<6;i++)std::cout<<local_point[i]<<" "; std::cout<<std::endl;
    //for(size_t i=0;i<tmp_local_point.size()/3;i++)
    //    std::copy(tmp_local_point[i*3],tmp_local_point[i*3+3],local_point[i]);
    /*for (size_t i=0;i<tmp_localization_points.size();i++)
    {
      for(size_t j=i+1;j<tmp_localization_points.size();j++)
      {
            for(size_t k=j+1;k<tmp_localization_points.size();k++)
            {
              std::cout<<i<< ' '<< j<< ' ' << k<<std::endl;
            }
      }
    }
    */
    /*
    for(size_t z=0;z<2;z++)
    {
    vector<vector<Point2f>>my_localization_points;//все возможные треугольники
    for (size_t i=0;i<tmp_localization_points.size();i++)
      for(size_t j=i+1;j<tmp_localization_points.size();j++)
            for(size_t k=j+1;k<tmp_localization_points.size();k++)
            {
                vector<Point2f>tmp_1;
                tmp_1.push_back(tmp_localization_points[i]);
                tmp_1.push_back(tmp_localization_points[j]);
                tmp_1.push_back(tmp_localization_points[k]);

                my_localization_points.push_back(tmp_1);//insert triangle
            }



     {
        while(  nextSet(mas,my_localization_points.size(),3))
        {
            for(size_t l=0;l<2;l++);
            {
            //vector<Point2f>tmp_1;
                tmp_triangle.push_back(tmp_localization_points[mas[i]]);
                nextSet(mas,my_localization_points.size(),3)
            }
        }
*/
    // my_localization_points.push_back(tmp_1);//insert triangle
    //vector<vector<area>> max_area;
    //for(size_t i=0;i<my_localization_points.size()/2;i++)
    //{

        //здесь должен быть еще один цикл для n тругольников
        //tmp_triangle.push_back(my_localization_points[i]);//первый треугольник
        //tmp_triangle.push_back(my_localization_points[(my_localization_points.size())-1-i]);//второй треугольник
      //  std :: cout << "tmp points :" << my_localization_points[i] << ' ' << my_localization_points[(my_localization_points.size())-1-i] << std::endl;
      //  tmp_triangle.push_back(my_localization_points[0]);//первый треугольник
      //  tmp_triangle.push_back(my_localization_points[1]);//первый треугольник

        bool flag;
        //все точки в tmp_localization_points
        int *mas=new int[number*3];
        int *mas_tmp=new int[number_points];
        for(size_t j=0;j<number*3;j++) mas[j]=j;
        for(size_t j=0;j<number_points;j++)
        {
          mas_tmp[j] = j;
          std::cout << mas_tmp[j] << ' ';
        }
        bool flag_for_in=true;
        bool flag_for_out =true;
        bool flag_=true;
        double sum1=10000000;
        double sum2=0;
        double sum_per1=10000000;
        double sum_per2=0;
        /*std::sort (tmp_localization_points.begin(), tmp_localization_points.begin()+tmp_localization_points.size(),
            [](Point2f& a,Point2f& b)
             {
                  return (sqrt(a.x*a.x+a.y*a.y))<(sqrt(b.x*b.x+b.y*b.y));
             });*/
        vector<vector<Point2f>> triangles;
        while( (flag_for_in==true)&&(flag_for_out==true))
        {
           vector<vector<Point2f>> tmp_triangle;
           for(size_t s=0;s<number;s++)
           {
               vector<Point2f> triangle;
               triangle.push_back(tmp_localization_points[mas[3*s]]);
               triangle.push_back(tmp_localization_points[mas[3*s+1]]);
               triangle.push_back(tmp_localization_points[mas[3*s+2]]);
               tmp_triangle.push_back(triangle);
               //std :: cout << mas[3*s] << ' ' << mas[3*s+1] << ' ' << mas[3*s+2] << ' ';

           }
           if ((number_points%3)==0)
              flag_for_in=NextSet(mas,number*3);
           else
           {
             if(flag_==false)
             {

                flag_for_in=NextSet(mas_tmp, number_points, number*3);

                for(size_t p=0; p<number*3; p++)
                {
                  mas[p] = mas_tmp[p];
                  std::cout << mas[p] << ' ';
                }
                 std::cout << std::endl;
             }
             flag_ = NextSet(mas,number*3);
            // for(size_t p=0; p<number*3; p++) std::cout << mas[p] << ' ';
            // std::cout << std::endl;
           }
           //for(size_t z=0;z<number*3;z++)
      //        std::cout << mas[z] << " ";
          // std::cout << std::endl;
           //std :: cout << std::endl;
        //for(size_t f=0;f<(tmp_localization_points.size()/3);f++)
      //      for(size_t k=f+1; k<(tmp_localization_points.size()/3) ; k++ )
           for(size_t f=0;f<(tmp_triangle.size());f++)
               for(size_t k=f+1; k<(tmp_triangle.size()) ; k++ )
                    for(size_t g=0;g<3;g++)
                    {
                       for(size_t l=g+1;l<3;l++)
                       {
                          for(size_t r=0;r<3;r++)
                          {
                              for(size_t t=r+1;t<3;t++)
                              {
                                  flag=intersect(tmp_triangle[f][g],tmp_triangle[f][l],tmp_triangle[k][r], tmp_triangle[k][t]);

                                  if(flag==true)
                                  {
                                  //  std::cout << "deleted points :" << tmp_triangle[0] << ' ' << tmp_triangle[1] << std::endl;
                                      tmp_triangle.clear();
                                    //  std :: cout <<"exit\n";
                                    break;
                                  }

                            } if (flag==true) break;

                          }if (flag==true) break;

                       }if (flag==true) break;
                    }

          if(tmp_triangle.size()==number)
          {
            //  std::cout << "не пересекаются\n";

             for(size_t q=0;q<number;q++)
             {
                 sum2+=triangle_area(tmp_triangle[q][0], tmp_triangle[q][1], tmp_triangle[q][2]);
                 sum_per2+=perimeter(tmp_triangle[q][0], tmp_triangle[q][1], tmp_triangle[q][2]);
             }
            if ((sum2 <= sum1) && (sum_per2 <= sum_per1))
            {

              sum1=sum2;
              sum_per1=sum_per2;
              triangles=tmp_triangle;



            }
            fixationPoints(tmp_triangle);
           bool flag_n = true;
           if(tmp_triangle.size()==number)
           {
            // std:: cout << "a\n";
          /*   for (size_t k = 0; k < number; k++)
             {
               for(size_t i=0; i<3; i++)
               {
                  for(size_t j=i; j<3; j++)
                  {
                    if (norm(tmp_triangle[k][i] - tmp_triangle[k][j]) < 10)
                    {
                        flag_n = false;
                        break;
                    }

                  } if (flag_n == false ) break;

                } if (flag_n == false ) break;
              }*/

            if( flag_n == true)
            {

                for(size_t i = 0; i < number; i ++)
                    localization_points.push_back(tmp_triangle[i]);
                std::cout << localization_points[0] << std::endl;
                flag_for_out = false;

            }
          }



         }
            sum2 = 0;
            sum_per2 = 0;
             //fixationPoints(tmp_triangle);
             //if(tmp_triangle.size()==number)
             //{
            //   for(size_t i=0;i<number;i++)
            //      localization_points.push_back(tmp_triangle[i]);
            //   flag_for_out=false;
              //localization_points.push_back(tmp_triangle[1]);
              //std::cout << tmp_triangle[0][0] << ' ' << tmp_triangle[1][0] << std::endl;


       }

        //localization_points.push_back(tmp_triangle);

      /*  if(current_areas.size()==2)
        {

            std::cout<<"a\n";
            max_area.push_back(current_areas);
            std::cout<<(current_areas[0]).point << ' '<< current_areas[1].point << std::endl;
        }
        */
    if ((localization_points.size()==0) && (triangles.size() == 0)) return false;
    if (localization_points.size()==0)
    {
        fixationPoints(triangles);
        for(size_t i=0;i<triangles.size();i++)
           localization_points.push_back(triangles[i]);
     }

     //for (size_t i = 0; i < localization_points[0].size(); i++)
        //circle(barcode, localization_points[1][i], 5, Scalar(255, 255, 255));
    //localization_points=tmp_triangle;
    //fixationPoints(localization_points);
    std::cout<<"size of localization points is " << localization_points.size() <<std::endl;
    //if(localization_points.size()!=number) return false;
    /*for(size_t i=0;i<localization_points.size();i++)
    {
        vector<area> current_areas=fixationPoints(localization_points[i]);//максимальная площадь из i треугольника
        if(current_areas.size()==2)
        {
            max_area.push_back(current_areas);
            std::cout<<(current_areas[0]).point << ' '<< current_areas[1].point << std::endl;
        }


    }
    */

  /*  std::sort (max_area.begin(), max_area.begin()+max_area.size(),
            [](area& a,area& b)
           {
                 return ((a[0].local_max_area)<(b[0].local_max_area));
           });
*/
    //for(size_t i=0;i<2;i++)
       //std::cout<< max_area[0].point << ' ' << max_area[1].point << std::endl;
     /*for (size_t i = 0; i < localization_points.size(); i++)
     {
        for (size_t j = i+1; j< localization_points.size(); j++)
        {
           std::cout << localization_points[i] << ' ' << localization_points[j]<<std::endl;
        }

     }
*/

    //for(size_t i=0;i<tmp_localization_points.size()/3;i++)
    //{
    //    vector<Point2f>tmp;
    //    for(size_t j=0;j<3;j++)
    //        {
    //            tmp.push_back(tmp_localization_points[i*3+j]);
    //        }
    //    localization_points.push_back(tmp);
    //}
  //  std::cout<<"size of localization points is " << localization_points.size() <<std::endl;
    //fixationPoints(localization_points);
  //  std::cout<<"size of localization points is " << localization_points.size() <<std::endl;
  /*  for(int i=0;i<localization_points.size();i++)
    {
        std::cout << localization_points[i][0]<<" " << localization_points[i][1]<<" " << localization_points[i][2]<<std::endl;
    }
*/
    //std::cout<<localization_points.size()<<std::endl;
    //if (localization_points.size() != 3) { return false; }
    //if (localization_points.size() < 6) { return false; }
    if (coeff_expansion > 1.0)
    {
        const int width  = cvRound(bin_barcode.size().width  / coeff_expansion);
        const int height = cvRound(bin_barcode.size().height / coeff_expansion);
        Size new_size(width, height);
        Mat intermediate;
        resize(bin_barcode, intermediate, new_size, 0, 0, INTER_LINEAR);
        bin_barcode = intermediate.clone();
        for (size_t i = 0; i < localization_points.size(); i++)
            for (size_t j = 0; j < localization_points[i].size(); j++)
        {
            localization_points[i][j] /= coeff_expansion;
        }
    }
    for(size_t k=0;k<localization_points.size();k++)
    {
        for (size_t i = 0; i <localization_points[k].size(); i++)
        {
            for (size_t j = i + 1; j < localization_points[k].size(); j++)
            {
                if (norm(localization_points[k][i] - localization_points[k][j]) < 10)
                {
                    localization_points.erase(localization_points.begin()+k);

                }
            }
        }
    }
    if(localization_points.size()==0) return false;
    return true;

}

bool QRDetect::computeTransformationPoints()
{
    //std::cout<<localization_points.size();
    CV_TRACE_FUNCTION();
    //if (localization_points.size() != 3) { return false; }
    for(size_t c=0;c<localization_points.size();c++)
    {
      if (localization_points[c].size() != 3) { localization_points[c].clear();
      localization_points.erase(localization_points.begin()+c); }
      vector<Point> locations, non_zero_elem[3], newHull;
      vector<Point2f> new_non_zero_elem[3];
    //std::cout<<localization_points[3]<<' '<<localization_points[4]<<' '<<localization_points[5]<<' '<<localization_points.size()<<std::endl;
        for (size_t i = 0; i <3 ; i++)
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
               transformation_points.push_back(tmp_transformation_points);
                //transformation_points[c].push_back(down_left_edge_point);
              //  transformation_points[c].push_back(up_left_edge_point);
              //  transformation_points[c].push_back(up_right_edge_point);
              //  transformation_points[c].push_back(
                //intersectionLines(down_left_edge_point, down_max_delta_point,
                          //up_right_edge_point, up_max_delta_point));
                vector<Point2f> quadrilateral = getQuadrilateral(transformation_points[c]);
                transformation_points[c] = quadrilateral;


              }

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

bool QRCodeDetector::detect(InputArray in, vector<Mat>& points, Mat &barc)
{
  std::cout<<"meow"<<std::endl;
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
  if (!qrdet.computeTransformationPoints()) { return false; }

  vector<vector<Point2f>> pnts2f = qrdet.getTransformationPoints();
  for(size_t i=0;i<pnts2f.size();i++)
  {
     Mat temp_mat;
     points.push_back(temp_mat);
     Mat(pnts2f[i]).convertTo(points[i], OutputArray(points[i]).fixedType() ? OutputArray(points[i]).type() : CV_32FC2);
  }
  barc = qrdet.barcode;
  return true;
};


bool QRCodeDetector::detect(InputArray in, OutputArray points)
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
    if (!qrdet.computeTransformationPoints()) { return false; }

    vector<vector<Point2f>> pnts2f = qrdet.getTransformationPoints();

    for(size_t i=1;i<2;i++)
        Mat(pnts2f[i]).convertTo(points, points.fixedType() ? points.type() : CV_32FC2);


    return true;
}



class QRDecode
{
public:
    //void init(const Mat &src, const vector<Point2f> &points);
    void init(const Mat &src, const vector<vector<Point2f>> &points);
  //  Mat getIntermediateBarcode() { return intermediate[0]; }
    vector<Mat> getIntermediateBarcode() { return intermediate; }
    //Mat getStraightBarcode() { return straight[0]; }
    vector<Mat> getStraightBarcode() { return straight; }
    //size_t getVersion() { return version[0]; }
    vector<uint8_t> getVersion() { return version; }
    vector<std::string> getDecodeInformation() { return result_info; }
  //  cv::String getDecodeInformation() { return ""; }
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

/*void QRDecode::init(const Mat &src, const vector<Point2f> &points)
{
    CV_TRACE_FUNCTION();
    original = src.clone();
    intermediate = vector<Mat::zeros(src.size(), CV_8UC1);
    original_points = points;
    version = 0;
    version_size = 0;
    test_perspective_size = 251;
    result_info = "";
}*/

void QRDecode::init(const Mat &src, const vector<vector<Point2f>> &points)
{
      CV_TRACE_FUNCTION();
      original = src.clone();
      std::cout<<points.size()<<std::endl;
      for(size_t i=0;i<points.size();i++)
      {
        original_points.push_back(points[i]);
        intermediate.push_back(Mat::zeros(src.size(), CV_8UC1));
        version.push_back(0);
        version_size.push_back(0);
        test_perspective_size.push_back(251);
        result_info.push_back("");
    }
}

bool QRDecode::updatePerspective()
{
    CV_TRACE_FUNCTION();
    for(size_t i=0;i<original_points.size();i++)
    {
        const Point2f centerPt = QRDetect::intersectionLines(original_points[i][0], original_points[i][2],
                                                         original_points[i][1], original_points[i][3]);
        //std::cout<<original_points[i][0]<<original_points[i][2]<<original_points[i][1]<<original_points[i][3]<<std::endl;
        if (cvIsNaN(centerPt.x) || cvIsNaN(centerPt.y))
            return false;
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
        no_border_intermediate.push_back(temp_intermediate(Range(1, temp_intermediate.rows), Range(1, temp_intermediate.cols)));

        const int border = cvRound(0.1 * test_perspective_size[i]);
        const int borderType = BORDER_CONSTANT;
        copyMakeBorder(no_border_intermediate[i], intermediate[i], border, border, border, border, borderType, Scalar(255));

    }
    return true;
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
    for(size_t c=0;c<original_points.size();c++)
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
          if ( !(  0 < version[c] && version[c] <= 40 ) ) { return false; }
          version_size[c] = 21 + (version[c] - 1) * 4;
  }
    return true;
}

bool QRDecode::samplingForVersion()
{
    CV_TRACE_FUNCTION();
    for(size_t q=0;q<original_points.size();q++)
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

      straight.push_back(Mat(Size(version_size[q], version_size[q]), CV_8UC1, Scalar(0)));
      for (int r = 0; r < version_size[q] * version_size[q]; r++)
      {
          int i   = r / straight[q].cols;
          int j   = r % straight[q].cols;
          straight[q].ptr<uint8_t>(i)[j] = (listFrequencyElem[r] < experimentalFrequencyElem) ? 0 : 255;
      }
    }
    return true;
}

bool QRDecode::decodingProcess()
{

    for(size_t c=0;c<original_points.size();c++)
    {
      if (straight[c].empty())
      {
        return false;
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
      if (errorCode != 0) { return false; }

      for (int i = 0; i < qr_code_data.payload_len; i++)
      {
          result_info[c] += qr_code_data.payload[i];
      }
    //return true;
    }
    return true;

    //return false;


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



vector<std::string> QRCodeDetector::decode(InputArray in, vector<Mat> points,
                                   vector<Mat>& straight_qrcode)
{
    Mat inarr = in.getMat();
    CV_Assert(!inarr.empty());
    CV_Assert(inarr.depth() == CV_8U);
    if (inarr.cols <= 20 || inarr.rows <= 20)
    {
        vector<std::string> v;
        return v;  // image data is not enough for providing reliable results
    }
    int incn = inarr.channels();
    if( incn == 3 || incn == 4 )
    {
        Mat gray;
        cvtColor(inarr, gray, COLOR_BGR2GRAY);
        inarr = gray;
    }

    vector<vector<Point2f>> src_points;
    for(size_t i=0;i<points.size();i++)
    {
      src_points.push_back(points[i]);
      CV_Assert(src_points[i].size()==4);
      //CV_Assert((int)((src_points[i]).size()) == 4);
      CV_CheckGT(contourArea(src_points[i]), 0.0, "Invalid QR code source points");
    }
    QRDecode qrdec;
    qrdec.init(inarr, src_points);
    bool ok = qrdec.fullDecodingProcess();
    vector<std::string > decoded_info= qrdec.getDecodeInformation();
    vector<Mat> straight_barcode=qrdec.getStraightBarcode();
    for(size_t i=0; i<straight_barcode.size(); i++)
    {
        if (ok && ((OutputArray)(straight_barcode[i])).needed())
        {

            Mat tmp;
            straight_qrcode.push_back(tmp);
            straight_barcode[i].convertTo(((OutputArray)straight_qrcode[i]),
                                             ((OutputArray)straight_qrcode[i]).fixedType() ?
                                             ((OutputArray)straight_qrcode[i]).type() : CV_32FC2);
       }
    }
    //return ok ? decoded_info : std::string();
    return decoded_info;
}




cv::String QRCodeDetector::decode(InputArray in, InputArray points,
                                   OutputArray straight_qrcode)
{
  std::cout<<"Not in my detect\n";
    /*Mat inarr = in.getMat();
    CV_Assert(!inarr.empty());
    CV_Assert(inarr.depth() == CV_8U);
    if (inarr.cols <= 20 || inarr.rows <= 20)
        return cv::String();  // image data is not enough for providing reliable results

    int incn = inarr.channels();
    if( incn == 3 || incn == 4 )
    {
        Mat gray;
        cvtColor(inarr, gray, COLOR_BGR2GRAY);
        inarr = gray;
    }

    vector<Point2f> src_points;
    points.copyTo(src_points);
    CV_Assert(src_points.size() == 4);
    CV_CheckGT(contourArea(src_points), 0.0, "Invalid QR code source points");

    QRDecode qrdec;
    qrdec.init(inarr, src_points);
    bool ok = qrdec.fullDecodingProcess();
*/
    std::string decoded_info="";
    return decoded_info;
/*
    if (ok && straight_qrcode.needed())
    {
        qrdec.getStraightBarcode().convertTo(straight_qrcode,
                                             straight_qrcode.fixedType() ?
                                             straight_qrcode.type() : CV_32FC2);
    }

    return ok ? decoded_info : std::string();
    */

}


vector<std::string> QRCodeDetector::detectAndDecode(InputArray in,
                                            vector<Mat>& points_,
                                            vector<Mat>& straight_qrcode, Mat &barcode)
{

    Mat inarr = in.getMat();
    CV_Assert(!inarr.empty());
    CV_Assert(inarr.depth() == CV_8U);
    //if (inarr.cols <= 20 || inarr.rows <= 20)
        //return <cv::String()>;  // image data is not enough for providing reliable results

    int incn = inarr.channels();
    if( incn == 3 || incn == 4 )
    {
        Mat gray;
        cvtColor(inarr, gray, COLOR_BGR2GRAY);
        inarr = gray;
    }

    vector<Mat> points;
    bool ok = detect(inarr, points, barcode);
    for(int i=0;i<points.size();i++)
    {

            if( ok )
            {
                Mat tmp;
                points_.push_back(tmp);
                Mat(points[i]).copyTo(points_[i]);
            }
            else
              ((OutputArray)points_[i]).release();

    }
    //std :: cout << "size of bounding boxes is " << points_[0] << std::endl;
    vector<std::string> decoded_info;

    if( ok )
       decoded_info = decode(inarr, points, straight_qrcode);


//endmycode
    return decoded_info;
}



cv::String QRCodeDetector::detectAndDecode(InputArray in,
                                            OutputArray points_,
                                            OutputArray straight_qrcode)
{

    Mat inarr = in.getMat();
    CV_Assert(!inarr.empty());
    CV_Assert(inarr.depth() == CV_8U);
    if (inarr.cols <= 20 || inarr.rows <= 20)
        return cv::String();  // image data is not enough for providing reliable results

    int incn = inarr.channels();
    if( incn == 3 || incn == 4 )
    {
        Mat gray;
        cvtColor(inarr, gray, COLOR_BGR2GRAY);
        inarr = gray;
    }
    vector<Point2f> points;
    bool ok = detect(inarr, points);
//Mat ok = detect(inarr, points);
//mycode
    if( points_.needed() )
    {
        if( ok )
            Mat(points).copyTo(points_);
        else
            points_.release();
    }

    std::string decoded_info;

    //if( ok )
    //  decoded_info = decode(inarr, points, straight_qrcode);
//endmycode
    return decoded_info;
 //return ok;
}

}
