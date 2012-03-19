#ifndef __OPENCV_VIDEOSTAB_FAST_MARCHING_HPP__
#define __OPENCV_VIDEOSTAB_FAST_MARCHING_HPP__

#include <cmath>
#include <queue>
#include <algorithm>
#include "opencv2/core/core.hpp"

namespace cv
{
namespace videostab
{

// See http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf
class FastMarchingMethod
{
public:
    FastMarchingMethod() : inf_(1e6f) {}

    template <typename Inpaint>
    void run(const Mat &mask, Inpaint inpaint);

    Mat distanceMap() const { return dist_; }

private:
    enum { INSIDE = 0, BAND = 1, KNOWN = 255 };

    struct DXY
    {
        float dist;
        int x, y;

        DXY() {}
        DXY(float dist, int x, int y) : dist(dist), x(x), y(y) {}
        bool operator <(const DXY &dxy) const { return dist < dxy.dist; }
    };

    float solve(int x1, int y1, int x2, int y2) const;
    int& indexOf(const DXY &dxy) { return index_(dxy.y, dxy.x); }

    void heapUp(int idx);
    void heapDown(int idx);
    void heapAdd(const DXY &dxy);
    void heapRemoveMin();

    float inf_;

    cv::Mat_<uchar> flag_; // flag map
    cv::Mat_<float> dist_; // distance map

    cv::Mat_<int> index_; // index of point in the narrow band
    std::vector<DXY> narrowBand_; // narrow band heap
    int size_; // narrow band size
};

} // namespace videostab
} // namespace cv

#include "fast_marching_inl.hpp"

#endif
