#pragma once
#include "opencv2/core/types.hpp"
namespace cv {

// Internal helper for convexHull().
// Returns true if fast-path applied, false if caller should fallback to std::sort.
bool convex_hull_bucket_sort(const Point* data,
                             bool require_monotonic_indices,
                             Point** out_points,
                             int& total,
                             int& ind_miny,
                             int& ind_maxy);

} // namespace cv
