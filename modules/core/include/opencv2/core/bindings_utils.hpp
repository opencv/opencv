// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_BINDINGS_UTILS_HPP
#define OPENCV_CORE_BINDINGS_UTILS_HPP

#include <opencv2/core/async.hpp>
#include <opencv2/core/detail/async_promise.hpp>

#include <stdexcept>

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
String testOverloadResolution(int value, const Point& point = Point(42, 24))
{
    return format("overload (int=%d, point=(x=%d, y=%d))", value, point.x,
                  point.y);
}

CV_WRAP static inline
String testOverloadResolution(const Rect& rect)
{
    return format("overload (rect=(x=%d, y=%d, w=%d, h=%d))", rect.x, rect.y,
                  rect.width, rect.height);
}

CV_WRAP static inline
String dumpRect(const Rect& argument)
{
    return format("rect: (x=%d, y=%d, w=%d, h=%d)", argument.x, argument.y,
                  argument.width, argument.height);
}

CV_WRAP static inline
String dumpTermCriteria(const TermCriteria& argument)
{
    return format("term_criteria: (type=%d, max_count=%d, epsilon=%lf",
                  argument.type, argument.maxCount, argument.epsilon);
}

CV_WRAP static inline
String dumpRotatedRect(const RotatedRect& argument)
{
    return format("rotated_rect: (c_x=%f, c_y=%f, w=%f, h=%f, a=%f)",
                  argument.center.x, argument.center.y, argument.size.width,
                  argument.size.height, argument.angle);
}

CV_WRAP static inline
String dumpRange(const Range& argument)
{
    if (argument == Range::all())
    {
        return "range: all";
    }
    else
    {
        return format("range: (s=%d, e=%d)", argument.start, argument.end);
    }
}

CV_WRAP static inline
void testRaiseGeneralException()
{
    throw std::runtime_error("exception text");
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

namespace fs {
    CV_EXPORTS_W cv::String getCacheDirectoryForDownloads();
} // namespace fs
//! @}
}} // namespaces cv /  utils

#endif // OPENCV_CORE_BINDINGS_UTILS_HPP
