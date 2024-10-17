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
    const char* extraTestDataPath =
        getenv("OPENCV_DNN_TEST_DATA_PATH");
    if (extraTestDataPath)
        cvtest::addDataSearchPath(extraTestDataPath);

    cvtest::addDataSearchSubDirectory("");  // override "cv" prefix below to access without "../dnn" hacks
}

CV_TEST_MAIN("cv", initTests())
