// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.


#include "precomp.hpp"

#include <queue>

namespace cv
{
struct Pix
{
    Point next_point;
    double cost;

    bool operator > (const Pix &b) const
    {
        return cost > b.cost;
    }
};

static float lcost(const Point& p, const Point& q, const Mat& gradient_magnitude, const Mat& Iy, const Mat& Ix, const Mat& zero_crossing)
{
    float fG = gradient_magnitude.at<float>(q.y, q.x);
    float dp;
    float dq;
    const float WEIGHT_LAP_ZERO_CROSS = 0.43f;
    const float WEIGHT_GRADIENT_MAGNITUDE = 0.14f;
    const float WEIGHT_GRADIENT_DIRECTION = 0.43f;
    bool isDiag = (p.x != q.x) && (p.y != q.y);

    if ((Iy.at<float>(p) * (q.x - p.x) - Ix.at<float>(p) * (q.y - p.y)) >= 0)
    {
        dp = Iy.at<float>(p) * (q.x - p.x) - Ix.at<float>(p) * (q.y - p.y);
        dq = Iy.at<float>(q) * (q.x - p.x) - Ix.at<float>(q) * (q.y - p.y);
    }
    else
    {
        dp = Iy.at<float>(p) * (p.x - q.x) + (-Ix.at<float>(p)) * (p.y - q.y);
        dq = Iy.at<float>(q) * (p.x - q.x) + (-Ix.at<float>(q)) * (p.y - q.y);
    }
    if (isDiag)
    {
        dp /= sqrtf(2);
        dq /= sqrtf(2);
    }
    else
    {
        fG /= sqrtf(2);
    }
    return  WEIGHT_LAP_ZERO_CROSS * zero_crossing.at<uchar>(q) +
            WEIGHT_GRADIENT_DIRECTION * (acosf(dp) + acosf(dq)) / static_cast<float>(CV_PI) +
            WEIGHT_GRADIENT_MAGNITUDE * fG;
}

class IntelligentScissorsImpl : public IntelligentScissors
{
public:
    void apply(InputArray img, OutputArray total_hit_map_x, OutputArray total_hit_map_y, const Point start_point)
    {
        const int EDGE_THRESHOLD_LOW = 50;
        const int EDGE_THRESHOLD_HIGH = 100;
        Mat src = img.getMat();

        total_hit_map_x.create(src.size(), CV_32SC1);
        Mat hit_map_x = total_hit_map_x.getMat();

        total_hit_map_y.create(src.size(), CV_32SC1);
        Mat hit_map_y = total_hit_map_y.getMat();

        Mat grayscale, img_canny, Ix, Iy;
        Mat zero_crossing, gradient_magnitude;

        cvtColor(src, grayscale, COLOR_BGR2GRAY);
        Canny(grayscale, img_canny, EDGE_THRESHOLD_LOW, EDGE_THRESHOLD_HIGH);
        threshold(img_canny, zero_crossing, 254, 1, THRESH_BINARY_INV);
        Sobel(grayscale, Ix, CV_32FC1, 1, 0, 1);
        Sobel(grayscale, Iy, CV_32FC1, 0, 1, 1);
        Ix.convertTo(Ix, CV_32F, 1.0 / 255);
        Iy.convertTo(Iy, CV_32F, 1.0 / 255);
        magnitude(Iy, Ix, gradient_magnitude);
        double max_val = 0.0;
        minMaxLoc(gradient_magnitude, 0, &max_val);
        if (max_val < DBL_MIN)
        {
            return;
        }
        gradient_magnitude.convertTo(gradient_magnitude, CV_32F, -1 / max_val, 1.0);

        Pix begin;
        Mat cost_map(src.size(), CV_32F, Scalar(FLT_MAX));
        Mat expand(src.size(), CV_8UC1, Scalar(0));
        Mat processed(src.size(), CV_8UC1, Scalar(0));
        Mat removed(src.size(), CV_8UC1, Scalar(0));
        std::priority_queue < Pix, std::vector<Pix>, std::greater<Pix> > L;

        cost_map.at<float>(start_point) = 0;
        processed.at<uchar>(start_point) = 1;
        begin.cost = 0;
        begin.next_point = start_point;
        L.push(begin);

        while (!L.empty())
        {
            Pix P = L.top();
            L.pop();
            Point p = P.next_point;
            processed.at<uchar>(p) = 0;
            if (removed.at<uchar>(p) == 0)
            {
                expand.at<uchar>(p) = 1;
                for (int i = -1; i <= 1; i++)
                {
                    for (int j = -1; j <= 1; j++)
                    {
                        int tx = p.x + i;
                        int ty = p.y + j;
                        if (tx < 0 || tx >= src.cols || ty < 0 || ty >= src.rows)
                            continue;
                        if (expand.at<uchar>(ty, tx) == 0)
                        {
                            Point q(tx, ty);
                            float cost = cost_map.at<float>(p) + lcost(p, q, gradient_magnitude, Iy, Ix, zero_crossing);
                            if (processed.at<uchar>(q) == 1 && cost < cost_map.at<float>(q))
                            {
                                removed.at<uchar>(q) = 1;
                            }
                            if (processed.at<uchar>(q) == 0)
                            {
                                cost_map.at<float>(q) = cost;
                                hit_map_x.at<int>(q) = p.x;
                                hit_map_y.at<int>(q) = p.y;

                                processed.at<uchar>(q) = 1;
                                Pix val;
                                val.cost = cost_map.at<float>(q);
                                val.next_point = q;
                                L.push(val);
                            }
                        }
                    }
                }
            }
        }
        hit_map_x.convertTo(total_hit_map_x, CV_32SC1);
        hit_map_y.convertTo(total_hit_map_y, CV_32SC1);
    }
};

Ptr<IntelligentScissors> cv::createIntelligentScissors()
{
    return makePtr<IntelligentScissorsImpl>();
}

}// namespace
