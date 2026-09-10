#include "precomp.hpp"
#include "convex_hull_bucket_sort.hpp"
#include <algorithm>

namespace cv {
bool convex_hull_bucket_sort(const Point* data,
                             bool require_monotonic_indices,
                             Point** out_points,
                             int& total,
                             int& ind_miny,
                             int& ind_maxy)
{
    struct XBucket { const Point* lo; const Point* hi; };

    const int MAX_RANGE = 100000;       // ~1.6 MB of buckets (sizeof(XBucket) * MAX_RANGE)
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
    // having lo and hi near to each other in memory should induce better cache locality
    AutoBuffer<XBucket> buckets(rangeX);
    std::fill_n(buckets.data(), rangeX, XBucket{nullptr, nullptr});

    // 3) Fill buckets
    for (int i = 0; i < total; ++i)
    {
        const int idx = data[i].x - minX;
        const int y = data[i].y;
        XBucket& b = buckets[idx];

        if (b.lo == nullptr || y < b.lo->y) {
            b.lo = &data[i];
        }
        else if (require_monotonic_indices && y == b.lo->y && !(data[i-1] == data[i])) {
            return false; // duplicate point (not consequtive) && require_monotonic_indices -> fallback to std::sort
        }

        if (b.hi == nullptr || y > b.hi->y) {
            b.hi = &data[i];
        }
        else if (require_monotonic_indices && y == b.hi->y && !(data[i-1] == data[i])) {
            return false; // duplicate point (not consequtive) && require_monotonic_indices -> fallback to std::sort
        }
    }

    // 4) Rebuild output pointer array in sorted X order
    int out = 0;
    ind_miny = 0;
    ind_maxy = 0;
    int cur = 0;
    for (int i = 0; i < rangeX; ++i)
    {
        const Point* pmin = buckets[i].lo;
        if (pmin == nullptr)
            continue;

        const Point* pmax = buckets[i].hi;
        CV_DbgAssert(pmax != nullptr && pmin->y <= pmax->y); // when filling buckets either both pmax and pmin are set or neither.

        out_points[out++] = const_cast<Point*>(pmin);
        cur = out - 1;

        int y = out_points[cur]->y;
        if (out_points[ind_miny]->y > y) {
            ind_miny = cur;
        }
        if (out_points[ind_maxy]->y < y) {
            ind_maxy = cur;
        }

        if (pmax != pmin) {
            out_points[out++] = const_cast<Point*>(pmax);
            cur = out - 1;
            y = out_points[cur]->y;
            if (out_points[ind_maxy]->y < y)
                ind_maxy = cur;
        }
    }

    total = out;
    return true;
}
} // namespace cv
