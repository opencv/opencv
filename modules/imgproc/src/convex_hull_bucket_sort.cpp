#include "precomp.hpp"
#include "convex_hull_bucket_sort.hpp"
#include <vector>
#include <algorithm>
#include <climits>

namespace cv {
bool convex_hull_bucket_sort(const Point* data,
                             Point** out_points,
                             int& total,
                             int& ind_miny,
                             int& ind_maxy)
{
    if (total <= 0) {
        return true;
    }

    // 1) Find minX and maxX
    int minX = data[0].x;
    int maxX = data[0].x;
    for (int i = 1; i < total; ++i)
    {
        minX = std::min(minX, data[i].x);
        maxX = std::max(maxX, data[i].x);
    }
    const int rangeX = maxX - minX + 1;
    const int MAX_RANGE = 100000;
    if (rangeX <= 0 || rangeX > MAX_RANGE)
        return false; 

    // 2) Create buckets that store pointers into data
    std::vector<const Point*> min_buckets(rangeX, nullptr);
    std::vector<const Point*> max_buckets(rangeX, nullptr);

    // 3) Fill buckets
    for (int i = 0; i < total; ++i)
    {
        const int x = data[i].x;
        const int y = data[i].y;
        const int idx = x - minX;
        if (min_buckets[idx] == nullptr || y < min_buckets[idx]->y)
            min_buckets[idx] = &data[i];
        if (max_buckets[idx] == nullptr || y > max_buckets[idx]->y)
            max_buckets[idx] = &data[i];
    }

    // 4) Rebuild output pointer array in sorted X order
    int out = 0;
    ind_miny = 0;
    ind_maxy = 0;
    int cur = 0;
    for (int i = 0; i < rangeX; ++i)
    {
        if (min_buckets[i] == nullptr)
            continue;

        const Point* pmin = min_buckets[i];
        const Point* pmax = max_buckets[i];
        CV_Assert(pmax == nullptr || pmin->y <= pmax->y);
        out_points[out++] = const_cast<Point*>(pmin);
        cur= out-1;
        int y = out_points[cur]->y;
        if (out_points[ind_miny]->y > y)
            ind_miny = cur;
        if (out_points[ind_maxy]->y < y)
                    ind_maxy = cur;
        if (pmax != pmin)
            {
                out_points[out++] = const_cast<Point*>(pmax);
                cur = out-1;
                y=out_points[cur]->y;
                if (out_points[ind_maxy]->y < y)
                    ind_maxy = cur;
            }    
    }

    total = out;
    return true;
}
} // namespace cv