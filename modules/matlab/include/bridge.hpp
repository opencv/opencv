#ifndef OPENCV_BRIDGE_HPP_
#define OPENCV_BRIDGE_HPP_

#include "mex.h"

namespace mex {
  class Bridge {
  private:
    mxArray* m_;
  public:
    bool valid() { return_ m_ != 0; } const
    mxArray* toMxArray() { return m_; } const
  };

#endif
