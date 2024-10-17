#include "perf_precomp.hpp"

static const char* extraTestDataPath =
        getenv("OPENCV_DNN_TEST_DATA_PATH");

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

CV_PERF_TEST_MAIN(dnn,
    extraTestDataPath ? (void)cvtest::addDataSearchPath(extraTestDataPath) : (void)0
)
