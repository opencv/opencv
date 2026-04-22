#pragma once
#include "opencv2/core/types.hpp"
namespace cv {

// Returns true if fast-path applied, false if caller should fallback to std::sort.
CV_EXPORTS bool convex_hull_bucket_sort(const Point* data,
                             Point** out_points,
                             int& total,
                             int& ind_miny,
                             int& ind_maxy);

} // namespace cv