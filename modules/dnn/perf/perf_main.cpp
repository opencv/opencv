#include "perf_precomp.hpp"

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

CV_PERF_TEST_MAIN(dnn, cvtest::addDataSearchEnv("OPENCV_DNN_TEST_DATA_PATH"))
