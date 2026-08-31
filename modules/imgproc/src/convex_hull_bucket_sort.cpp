#include "precomp.hpp"
#include "convex_hull_bucket_sort.hpp"
#include <algorithm>

namespace cv {
bool convex_hull_bucket_sort(const Point* data,
                             Point** out_points,
                             int& total,
                             int& ind_miny,
                             int& ind_maxy)
{
    const int MAX_RANGE = 100000;       // ~1.6 MB of bucket pointers (2 * 8 * MAX_RANGE)
    const int MAX_SPARSITY_FACTOR = 4;  // std::sort beats buckets on sparse ranges

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

    const int64 rangeX64 = (int64)maxX - (int64)minX + 1;
    if (rangeX64 > MAX_SPARSITY_FACTOR * (int64)total) {
        // bail out, std::sort is faster for sparse data
        return false;
    }
    if (rangeX64 > MAX_RANGE) {
        // bail out, we cannot allocate too much memory for buckets
        return false;
    }

    const int rangeX = (int)rangeX64;

    // 2) Create buckets that store pointers into data.
    //    Single allocation for both halves; min_buckets / max_buckets alias into it.
    AutoBuffer<const Point*> buckets(2 * (size_t)rangeX);
    const Point** min_buckets = buckets.data();
    const Point** max_buckets = min_buckets + rangeX;
    std::fill_n(min_buckets, 2 * (size_t)rangeX, nullptr);

    // 3) Fill buckets
    for (int i = 0; i < total; ++i)
    {
        const int x = data[i].x;
        const int y = data[i].y;
        const int idx = x - minX;
        if (min_buckets[idx] == nullptr || y < min_buckets[idx]->y) {
            min_buckets[idx] = &data[i];
        }
        if (max_buckets[idx] == nullptr || y > max_buckets[idx]->y) {
            max_buckets[idx] = &data[i];
        }
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
        CV_DbgAssert(pmax == nullptr || pmin->y <= pmax->y);
        out_points[out++] = const_cast<Point*>(pmin);
        cur = out-1;
        int y = out_points[cur]->y;
        if (out_points[ind_miny]->y > y) {
            ind_miny = cur;
        }
        if (out_points[ind_maxy]->y < y) {
            ind_maxy = cur;
        }
        if (pmax != pmin) {
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