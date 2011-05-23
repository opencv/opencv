#include "split.h"

#include <boost/python.hpp>

namespace bp = boost::python;

BOOST_PYTHON_MODULE(opencv)
{
  //wrap all modules
  opencv_wrappers::wrap_cv_core();
  opencv_wrappers::wrap_mat();
  opencv_wrappers::wrap_points();
  opencv_wrappers::wrap_highgui();
}



