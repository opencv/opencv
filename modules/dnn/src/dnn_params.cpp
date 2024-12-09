// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "dnn_common.hpp"
#include <opencv2/core/utils/configuration.private.hpp>

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


size_t getParam_DNN_NETWORK_DUMP()
{
    static size_t DNN_NETWORK_DUMP = utils::getConfigurationParameterSizeT("OPENCV_DNN_NETWORK_DUMP", 0);
    return DNN_NETWORK_DUMP;
}

// this option is useful to run with valgrind memory errors detection
bool getParam_DNN_DISABLE_MEMORY_OPTIMIZATIONS()
{
    static bool DNN_DISABLE_MEMORY_OPTIMIZATIONS = utils::getConfigurationParameterBool("OPENCV_DNN_DISABLE_MEMORY_OPTIMIZATIONS", false);
    return DNN_DISABLE_MEMORY_OPTIMIZATIONS;
}

#ifdef HAVE_OPENCL
bool getParam_DNN_OPENCL_ALLOW_ALL_DEVICES()
{
    static bool DNN_OPENCL_ALLOW_ALL_DEVICES = utils::getConfigurationParameterBool("OPENCV_DNN_OPENCL_ALLOW_ALL_DEVICES", false);
    return DNN_OPENCL_ALLOW_ALL_DEVICES;
}
#endif

int getParam_DNN_BACKEND_DEFAULT()
{
    static int PARAM_DNN_BACKEND_DEFAULT = (int)utils::getConfigurationParameterSizeT("OPENCV_DNN_BACKEND_DEFAULT",
#ifdef OPENCV_DNN_BACKEND_DEFAULT
            (size_t)OPENCV_DNN_BACKEND_DEFAULT
#else
            (size_t)DNN_BACKEND_OPENCV
#endif
    );
    return PARAM_DNN_BACKEND_DEFAULT;
}

// Additional checks (slowdowns execution!)
bool getParam_DNN_CHECK_NAN_INF()
{
    static bool DNN_CHECK_NAN_INF = utils::getConfigurationParameterBool("OPENCV_DNN_CHECK_NAN_INF", false);
    return DNN_CHECK_NAN_INF;
}
bool getParam_DNN_CHECK_NAN_INF_DUMP()
{
    static bool DNN_CHECK_NAN_INF_DUMP = utils::getConfigurationParameterBool("OPENCV_DNN_CHECK_NAN_INF_DUMP", false);
    return DNN_CHECK_NAN_INF_DUMP;
}
bool getParam_DNN_CHECK_NAN_INF_RAISE_ERROR()
{
    static bool DNN_CHECK_NAN_INF_RAISE_ERROR = utils::getConfigurationParameterBool("OPENCV_DNN_CHECK_NAN_INF_RAISE_ERROR", false);
    return DNN_CHECK_NAN_INF_RAISE_ERROR;
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
