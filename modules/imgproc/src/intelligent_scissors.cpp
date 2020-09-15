#include "precomp.hpp"

#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <queue>


#include "opencv2/imgproc.hpp"

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


float local_cost(const Point& p, const Point& q, const Mat& gradient_magnitude, const Mat& Iy, const Mat& Ix, const Mat& zero_crossing)
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

struct Parameters
{
    Mat img, img_pre_render, img_render;
    Point end;
    std::vector<std::vector<Point> > contours;
    std::vector<Point> tmp_contour;
    Mat zero_crossing, gradient_magnitude, Ix, Iy, hit_map_x, hit_map_y;
};

void find_min_path(const Point& start, Mat img, Mat zero_crossing, Mat gradient_magnitude, Mat Ix, Mat Iy, Mat& hit_map_x, Mat& hit_map_y)
{
    Pix begin;
     Mat A(img.size(), CV_8UC1, Scalar(0));
    // Mat &img = param->img;
    Mat cost_map(img.size(), CV_32F, Scalar(FLT_MAX));
    Mat expand(img.size(), CV_8UC1, Scalar(0));
    Mat processed(img.size(), CV_8UC1, Scalar(0));
    Mat removed(img.size(), CV_8UC1, Scalar(0));
    std::priority_queue < Pix, std::vector<Pix>, std::greater<Pix> > L;

    cost_map.at<float>(start) = 0;
    processed.at<uchar>(start) = 1;
    begin.cost = 0;
    begin.next_point = start;

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
                for(int j = -1; j <= 1; j++)
                {
                    int tx = p.x + i;
                    int ty = p.y + j;
                    if (tx < 0 || tx >= img.cols || ty < 0 || ty >= img.rows)
                        continue;
                    if (expand.at<uchar>(ty, tx) == 0)
                    {
                        Point q = Point(tx, ty);
                        float cost = cost_map.at<float>(p) + local_cost(p, q, gradient_magnitude, Iy, Ix, zero_crossing);
                        if (processed.at<uchar>(q) == 1 && cost < cost_map.at<float>(q))
                        {
                            removed.at<uchar>(q) = 1;
                        }
                        if (processed.at<uchar>(q) == 0)
                        {
                            cost_map.at<float>(q) = cost;
                            hit_map_x.at<int>(q)= p.x;
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
}

}
