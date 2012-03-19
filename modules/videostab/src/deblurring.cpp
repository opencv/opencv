#include "precomp.hpp"
#include "opencv2/videostab/deblurring.hpp"
#include "opencv2/videostab/global_motion.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

float calcBlurriness(const Mat &frame)
{
    Mat Gx, Gy;
    Sobel(frame, Gx, CV_32F, 1, 0);
    Sobel(frame, Gy, CV_32F, 0, 1);
    double normGx = norm(Gx);
    double normGy = norm(Gx);
    double sumSq = normGx*normGx + normGy*normGy;
    return 1.f / (sumSq / frame.size().area() + 1e-6);
}


WeightingDeblurer::WeightingDeblurer()
{
    setSensitivity(0.1);
}


void WeightingDeblurer::deblur(int idx, Mat &frame)
{
    CV_Assert(frame.type() == CV_8UC3);

    bSum_.create(frame.size());
    gSum_.create(frame.size());
    rSum_.create(frame.size());
    wSum_.create(frame.size());

    for (int y = 0; y < frame.rows; ++y)
    {
        for (int x = 0; x < frame.cols; ++x)
        {
            Point3_<uchar> p = frame.at<Point3_<uchar> >(y,x);
            bSum_(y,x) = p.x;
            gSum_(y,x) = p.y;
            rSum_(y,x) = p.z;
            wSum_(y,x) = 1.f;
        }
    }

    for (int k = idx - radius_; k <= idx + radius_; ++k)
    {
        const Mat &neighbor = at(k, *frames_);
        float bRatio = at(idx, *blurrinessRates_) / at(k, *blurrinessRates_);
        Mat_<float> M = getMotion(idx, k, *motions_);

        if (bRatio > 1.f)
        {
            for (int y = 0; y < frame.rows; ++y)
            {
                for (int x = 0; x < frame.cols; ++x)
                {
                    int x1 = M(0,0)*x + M(0,1)*y + M(0,2);
                    int y1 = M(1,0)*x + M(1,1)*y + M(1,2);

                    if (x1 >= 0 && x1 < neighbor.cols && y1 >= 0 && y1 < neighbor.rows)
                    {
                        const Point3_<uchar> &p = frame.at<Point3_<uchar> >(y,x);
                        const Point3_<uchar> &p1 = neighbor.at<Point3_<uchar> >(y1,x1);
                        float w = bRatio * sensitivity_ /
                                (sensitivity_ + std::abs(intensity(p1) - intensity(p)));
                        bSum_(y,x) += w * p1.x;
                        gSum_(y,x) += w * p1.y;
                        rSum_(y,x) += w * p1.z;
                        wSum_(y,x) += w;
                    }
                }
            }
        }
    }

    for (int y = 0; y < frame.rows; ++y)
    {
        for (int x = 0; x < frame.cols; ++x)
        {
            float wSumInv = 1.f / wSum_(y,x);
            frame.at<Point3_<uchar> >(y,x) = Point3_<uchar>(
                    bSum_(y,x) * wSumInv, gSum_(y,x) * wSumInv, rSum_(y,x) * wSumInv);
        }
    }
}

} // namespace videostab
} // namespace cv
