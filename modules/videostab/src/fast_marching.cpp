#include "precomp.hpp"
#include "opencv2/videostab/fast_marching.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

float FastMarchingMethod::solve(int x1, int y1, int x2, int y2) const
{
    float sol = inf_;
    if (y1 >=0 && y1 < flag_.rows && x1 >= 0 && x1 < flag_.cols && flag_(y1,x1) == KNOWN)
    {
        float t1 = dist_(y1,x1);
        if (y2 >=0 && y2 < flag_.rows && x2 >= 0 && x2 < flag_.cols && flag_(y2,x2) == KNOWN)
        {
            float t2 = dist_(y2,x2);
            float r = sqrt(2 - sqr(t1 - t2));
            float s = (t1 + t2 - r) / 2;

            if (s >= t1 && s >= t2)
                sol = s;
            else
            {
                s += r;
                if (s >= t1 && s >= t2)
                    sol = s;
            }
        }
        else
            sol = 1 + t1;
    }
    else if (y2 >=0 && y2 < flag_.rows && x2 >= 0 && x2 < flag_.cols && flag_(y2,x2) == KNOWN)
        sol = 1 + dist_(y2,x1);
    return sol;
}


void FastMarchingMethod::heapUp(int idx)
{
    int p = (idx-1)/2;
    while (idx > 0 && narrowBand_[idx] < narrowBand_[p])
    {
        std::swap(indexOf(narrowBand_[p]), indexOf(narrowBand_[idx]));
        std::swap(narrowBand_[p], narrowBand_[idx]);
        idx = p;
        p = (idx-1)/2;
    }
}


void FastMarchingMethod::heapDown(int idx)
{
    int l, r, smallest;
    while (true)
    {
        l = 2*idx+1;
        r = 2*idx+2;
        smallest = idx;

        if (l < size_ && narrowBand_[l] < narrowBand_[smallest]) smallest = l;
        if (r < size_ && narrowBand_[r] < narrowBand_[smallest]) smallest = r;

        if (smallest == idx) break;
        else
        {
            std::swap(indexOf(narrowBand_[idx]), indexOf(narrowBand_[smallest]));
            std::swap(narrowBand_[idx], narrowBand_[smallest]);
            idx = smallest;
        }
    }
}


void FastMarchingMethod::heapAdd(const DXY &dxy)
{
    if (static_cast<int>(narrowBand_.size()) < size_ + 1)
        narrowBand_.resize(size_*2 + 1);
    narrowBand_[size_] = dxy;
    indexOf(dxy) = size_++;
    heapUp(size_-1);
}


void FastMarchingMethod::heapRemoveMin()
{
    if (size_ > 0)
    {
        size_--;
        std::swap(indexOf(narrowBand_[0]), indexOf(narrowBand_[size_]));
        std::swap(narrowBand_[0], narrowBand_[size_]);
        heapDown(0);
    }
}

} // namespace videostab
} // namespace cv
