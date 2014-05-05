/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                        Intel License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2000, Intel Corporation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of Intel Corporation may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"

#include <limits>
#include <utility>
#include <algorithm>

#include <math.h>

namespace cv {

inline bool is_smaller(const std::pair<int, float>& p1, const std::pair<int, float>& p2)
{
    return p1.second < p2.second;
}

static void orderContours(const std::vector<std::vector<Point> >& contours, Point2f point, std::vector<std::pair<int, float> >& order)
{
    order.clear();
    size_t i, j, n = contours.size();
    for(i = 0; i < n; i++)
    {
        size_t ni = contours[i].size();
        double min_dist = std::numeric_limits<double>::max();
        for(j = 0; j < ni; j++)
        {
            double dist = norm(Point2f((float)contours[i][j].x, (float)contours[i][j].y) - point);
            min_dist = MIN(min_dist, dist);
        }
        order.push_back(std::pair<int, float>((int)i, (float)min_dist));
    }

    std::sort(order.begin(), order.end(), is_smaller);
}

// fit second order curve to a set of 2D points
inline void fitCurve2Order(const std::vector<Point2f>& /*points*/, std::vector<float>& /*curve*/)
{
    // TBD
}

inline void findCurvesCross(const std::vector<float>& /*curve1*/, const std::vector<float>& /*curve2*/, Point2f& /*cross_point*/)
{
}

static void findLinesCrossPoint(Point2f origin1, Point2f dir1, Point2f origin2, Point2f dir2, Point2f& cross_point)
{
    float det = dir2.x*dir1.y - dir2.y*dir1.x;
    Point2f offset = origin2 - origin1;

    float alpha = (dir2.x*offset.y - dir2.y*offset.x)/det;
    cross_point = origin1 + dir1*alpha;
}

static void findCorner(const std::vector<Point2f>& contour, Point2f point, Point2f& corner)
{
    // find the nearest point
    double min_dist = std::numeric_limits<double>::max();
    int min_idx = -1;

    // find corner idx
    for(size_t i = 0; i < contour.size(); i++)
    {
        double dist = norm(contour[i] - point);
        if(dist < min_dist)
        {
            min_dist = dist;
            min_idx = (int)i;
        }
    }
    CV_Assert(min_idx >= 0);

    // temporary solution, have to make something more precise
    corner = contour[min_idx];
    return;
}

static int segment_hist_max(const Mat& hist, int& low_thresh, int& high_thresh)
{
    Mat bw;
    double total_sum = sum(hist).val[0];

    double quantile_sum = 0.0;
    //double min_quantile = 0.2;
    double low_sum = 0;
    double max_segment_length = 0;
    int max_start_x = -1;
    int max_end_x = -1;
    int start_x = 0;
    const double out_of_bells_fraction = 0.1;
    for(int x = 0; x < hist.size[0]; x++)
    {
        quantile_sum += hist.at<float>(x);
        if(quantile_sum < 0.2*total_sum) continue;

        if(quantile_sum - low_sum > out_of_bells_fraction*total_sum)
        {
            if(max_segment_length < x - start_x)
            {
                max_segment_length = x - start_x;
                max_start_x = start_x;
                max_end_x = x;
            }

            low_sum = quantile_sum;
            start_x = x;
        }
    }

    if(start_x == -1)
    {
        return 0;
    }
    else
    {
        low_thresh = cvRound(max_start_x + 0.25*(max_end_x - max_start_x));
        high_thresh = cvRound(max_start_x + 0.75*(max_end_x - max_start_x));
        return 1;
    }
}

}

bool cv::find4QuadCornerSubpix(InputArray _img, InputOutputArray _corners, Size region_size)
{
    Mat img = _img.getMat(), cornersM = _corners.getMat();
    int ncorners = cornersM.checkVector(2, CV_32F);
    CV_Assert( ncorners >= 0 );
    Point2f* corners = cornersM.ptr<Point2f>();
    const int nbins = 256;
    float ranges[] = {0, 256};
    const float* _ranges = ranges;
    Mat hist;

    Mat black_comp, white_comp;
    for(int i = 0; i < ncorners; i++)
    {
        int channels = 0;
        Rect roi(cvRound(corners[i].x - region_size.width), cvRound(corners[i].y - region_size.height),
            region_size.width*2 + 1, region_size.height*2 + 1);
        Mat img_roi = img(roi);
        calcHist(&img_roi, 1, &channels, Mat(), hist, 1, &nbins, &_ranges);

        int black_thresh = 0, white_thresh = 0;
        segment_hist_max(hist, black_thresh, white_thresh);

        threshold(img, black_comp, black_thresh, 255.0, THRESH_BINARY_INV);
        threshold(img, white_comp, white_thresh, 255.0, THRESH_BINARY);

        const int erode_count = 1;
        erode(black_comp, black_comp, Mat(), Point(-1, -1), erode_count);
        erode(white_comp, white_comp, Mat(), Point(-1, -1), erode_count);

        std::vector<std::vector<Point> > white_contours, black_contours;
        std::vector<Vec4i> white_hierarchy, black_hierarchy;
        findContours(black_comp, black_contours, black_hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
        findContours(white_comp, white_contours, white_hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

        if(black_contours.size() < 5 || white_contours.size() < 5) continue;

        // find two white and black blobs that are close to the input point
        std::vector<std::pair<int, float> > white_order, black_order;
        orderContours(black_contours, corners[i], black_order);
        orderContours(white_contours, corners[i], white_order);

        const float max_dist = 10.0f;
        if(black_order[0].second > max_dist || black_order[1].second > max_dist ||
           white_order[0].second > max_dist || white_order[1].second > max_dist)
        {
            continue; // there will be no improvement in this corner position
        }

        const std::vector<Point>* quads[4] = {&black_contours[black_order[0].first], &black_contours[black_order[1].first],
                                         &white_contours[white_order[0].first], &white_contours[white_order[1].first]};
        std::vector<Point2f> quads_approx[4];
        Point2f quad_corners[4];
        for(int k = 0; k < 4; k++)
        {
            std::vector<Point2f> temp;
            for(size_t j = 0; j < quads[k]->size(); j++) temp.push_back((*quads[k])[j]);
            approxPolyDP(Mat(temp), quads_approx[k], 0.5, true);

            findCorner(quads_approx[k], corners[i], quad_corners[k]);
            quad_corners[k] += Point2f(0.5f, 0.5f);
        }

        // cross two lines
        Point2f origin1 = quad_corners[0];
        Point2f dir1 = quad_corners[1] - quad_corners[0];
        Point2f origin2 = quad_corners[2];
        Point2f dir2 = quad_corners[3] - quad_corners[2];
        double angle = acos(dir1.dot(dir2)/(norm(dir1)*norm(dir2)));
        if(cvIsNaN(angle) || cvIsInf(angle) || angle < 0.5 || angle > CV_PI - 0.5) continue;

        findLinesCrossPoint(origin1, dir1, origin2, dir2, corners[i]);
    }

    return true;
}
