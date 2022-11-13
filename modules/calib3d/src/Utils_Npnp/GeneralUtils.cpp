//
// Created by yuval on 6/14/20.
//

#include "GeneralUtils.h"

namespace NPnP {
double find_zero_bin_search(const std::function<double(double)> &func,
                            double min, double max, int depth) {
  while (true) {
    double mid = (min + max) / 2.0;
    if (depth-- == 0)
      return mid;
    if (func(max) <= 0)
      return max;
    if (func(mid) < 0)
      min = mid;
    else
      max = mid;
  }
}
} // namespace NPnP
