#include "test_precomp.hpp"

static const char* extraTestDataPath =
#ifdef WINRT
        NULL;
#else
        getenv("OPENCV_DNN_TEST_DATA_PATH");
#endif

#if defined(HAVE_HPX)
    #include <hpx/hpx_main.hpp>
#endif

CV_TEST_MAIN("",
    extraTestDataPath ? (void)cvtest::addDataSearchPath(extraTestDataPath) : (void)0
)

namespace opencv_test
{

using namespace cv;
using namespace cv::dnn;

}
