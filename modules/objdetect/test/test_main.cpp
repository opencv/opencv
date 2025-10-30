// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

static
void initTests()
{
#ifdef HAVE_OPENCV_DNN
    cvtest::addDataSearchEnv("OPENCV_DNN_TEST_DATA_PATH");
#endif  // HAVE_OPENCV_DNN
}

CV_TEST_MAIN("cv", initTests())
