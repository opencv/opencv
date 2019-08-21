#include "perf_precomp.hpp"

static const char* extraTestDataPath =
#ifdef WINRT
        NULL;
#else
        getenv("OPENCV_DNN_TEST_DATA_PATH");
#endif

CV_PERF_TEST_MAIN(dnn,
    extraTestDataPath ? (void)cvtest::addDataSearchPath(extraTestDataPath) : (void)0
)
