#include "perf_precomp.hpp"

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

static
void initTests()
{
    cvtest::addDataSearchEnv("OPENCV_DNN_TEST_DATA_PATH");
    cvtest::addDataSearchSubDirectory("");  // override "cv" prefix below to access without "../dnn" hacks
}

CV_PERF_TEST_MAIN(video, initTests())
