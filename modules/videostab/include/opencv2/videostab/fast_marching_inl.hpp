#ifndef __OPENCV_VIDEOSTAB_FAST_MARCHING_INL_HPP__
#define __OPENCV_VIDEOSTAB_FAST_MARCHING_INL_HPP__

#include "opencv2/videostab/fast_marching.hpp"

namespace cv
{
namespace videostab
{

template <typename Inpaint>
void FastMarchingMethod::run(const cv::Mat &mask, Inpaint inpaint)
{
    using namespace std;
    using namespace cv;

    CV_Assert(mask.type() == CV_8U);

    static const int lut[4][2] = {{-1,0}, {0,-1}, {1,0}, {0,1}};

    mask.copyTo(flag_);
    flag_.create(mask.size());
    dist_.create(mask.size());
    index_.create(mask.size());
    narrowBand_.clear();
    size_ = 0;

    // init
    for (int y = 0; y < flag_.rows; ++y)
    {
        for (int x = 0; x < flag_.cols; ++x)
        {
            if (flag_(y,x) == KNOWN)
                dist_(y,x) = 0.f;
            else
            {
                int n = 0;
                int nunknown = 0;

                for (int i = 0; i < 4; ++i)
                {
                    int xn = x + lut[i][0];
                    int yn = y + lut[i][1];

                    if (xn >= 0 && xn < flag_.cols && yn >= 0 && yn < flag_.rows)
                    {
                        n++;
                        if (flag_(yn,xn) != KNOWN)
                            nunknown++;
                    }
                }

                if (n>0 && nunknown == n)
                {
                    dist_(y,x) = inf_;
                    flag_(y,x) = INSIDE;
                }
                else
                {
                    dist_(y,x) = 0.f;
                    flag_(y,x) = BAND;
                    inpaint(x, y);

                    narrowBand_.push_back(DXY(0.f,x,y));
                    index_(y,x) = size_++;
                }
            }
        }
    }

    // make heap
    for (int i = size_/2-1; i >= 0; --i)
        heapDown(i);

    // main cycle
    while (size_ > 0)
    {
        int x = narrowBand_[0].x;
        int y = narrowBand_[0].y;
        heapRemoveMin();

        flag_(y,x) = KNOWN;
        for (int n = 0; n < 4; ++n)
        {
            int xn = x + lut[n][0];
            int yn = y + lut[n][1];

            if (xn >= 0 && xn < flag_.cols && yn >= 0 && yn < flag_.rows && flag_(yn,xn) != KNOWN)
            {
                dist_(yn,xn) = min(min(solve(xn-1, yn, xn, yn-1), solve(xn+1, yn, xn, yn-1)),
                                   min(solve(xn-1, yn, xn, yn+1), solve(xn+1, yn, xn, yn+1)));

                if (flag_(yn,xn) == INSIDE)
                {
                    flag_(yn,xn) = BAND;
                    inpaint(xn, yn);
                    heapAdd(DXY(dist_(yn,xn),xn,yn));
                }
                else
                {
                    int i = index_(yn,xn);
                    if (dist_(yn,xn) < narrowBand_[i].dist)
                    {
                        narrowBand_[i].dist = dist_(yn,xn);
                        heapUp(i);
                    }
                    // works better if it's commented out
                    /*else if (dist(yn,xn) > narrowBand[i].dist)
                    {
                        narrowBand[i].dist = dist(yn,xn);
                        heapDown(i);
                    }*/
                }
            }
        }
    }
}

} // namespace videostab
} // namespace cv

#endif
