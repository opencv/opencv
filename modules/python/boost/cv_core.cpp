#include <boost/python.hpp>

#include <opencv2/core/core.hpp>

namespace bp = boost::python;

namespace opencv_wrappers
{
  void wrap_cv_core()
  {
    bp::object opencv = bp::scope();
    //define opencv consts
#include "cv_defines.cpp"

  }
}
