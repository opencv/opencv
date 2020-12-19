// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_BINDINGS_UTILS_HPP
#define OPENCV_CORE_BINDINGS_UTILS_HPP

#include <opencv2/core/async.hpp>
#include <opencv2/core/detail/async_promise.hpp>

namespace cv { namespace utils {
//! @addtogroup core_utils
//! @{

CV_EXPORTS_W String dumpInputArray(InputArray argument);

CV_EXPORTS_W String dumpInputArrayOfArrays(InputArrayOfArrays argument);

CV_EXPORTS_W String dumpInputOutputArray(InputOutputArray argument);

CV_EXPORTS_W String dumpInputOutputArrayOfArrays(InputOutputArrayOfArrays argument);

CV_WRAP static inline
String dumpBool(bool argument)
{
    return (argument) ? String("Bool: True") : String("Bool: False");
}

CV_WRAP static inline
String dumpInt(int argument)
{
    return cv::format("Int: %d", argument);
}

CV_WRAP static inline
String dumpSizeT(size_t argument)
{
    std::ostringstream oss("size_t: ", std::ios::ate);
    oss << argument;
    return oss.str();
}

CV_WRAP static inline
String dumpFloat(float argument)
{
    return cv::format("Float: %.2f", argument);
}

CV_WRAP static inline
String dumpDouble(double argument)
{
    return cv::format("Double: %.2f", argument);
}

CV_WRAP static inline
String dumpCString(const char* argument)
{
    return cv::format("String: %s", argument);
}

CV_WRAP static inline
String dumpString(const String& argument)
{
    return cv::format("String: %s", argument.c_str());
}

CV_WRAP static inline
AsyncArray testAsyncArray(InputArray argument)
{
    AsyncPromise p;
    p.setValue(argument);
    return p.getArrayResult();
}

CV_WRAP static inline
AsyncArray testAsyncException()
{
    AsyncPromise p;
    try
    {
        CV_Error(Error::StsOk, "Test: Generated async error");
    }
    catch (const cv::Exception& e)
    {
        p.setException(e);
    }
    return p.getArrayResult();
}

//! @}
}} // namespace

#endif // OPENCV_CORE_BINDINGS_UTILS_HPP
