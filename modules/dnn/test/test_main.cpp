#include "test_precomp.hpp"

static const char* extraTestDataPath =
#ifdef WINRT
        NULL;
#else
        getenv("OPENCV_DNN_TEST_DATA_PATH");
#endif

CV_TEST_MAIN("",
    extraTestDataPath ? (void)cvtest::addDataSearchPath(extraTestDataPath) : (void)0
)

namespace cvtest
{

using namespace cv;
using namespace cv::dnn;

}
