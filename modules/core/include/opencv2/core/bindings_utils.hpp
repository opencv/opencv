// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_BINDINGS_UTILS_HPP
#define OPENCV_CORE_BINDINGS_UTILS_HPP

#include <opencv2/core/async.hpp>
#include <opencv2/core/detail/async_promise.hpp>
#include <opencv2/core/utils/logger.hpp>

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
RotatedRect testRotatedRect(float x, float y, float w, float h, float angle)
{
    return RotatedRect(Point2f(x, y), Size2f(w, h), angle);
}

CV_WRAP static inline
std::vector<RotatedRect> testRotatedRectVector(float x, float y, float w, float h, float angle)
{
    std::vector<RotatedRect> result;
    for (int i = 0; i < 10; i++)
        result.push_back(RotatedRect(Point2f(x + i, y + 2 * i), Size2f(w, h), angle + 10 * i));
    return result;
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
int testOverwriteNativeMethod(int argument)
{
    return argument;
}

CV_WRAP static inline
String testReservedKeywordConversion(int positional_argument, int lambda = 2, int from = 3)
{
    return format("arg=%d, lambda=%d, from=%d", positional_argument, lambda, from);
}

CV_EXPORTS_W String dumpVectorOfInt(const std::vector<int>& vec);

CV_EXPORTS_W String dumpVectorOfDouble(const std::vector<double>& vec);

CV_EXPORTS_W String dumpVectorOfRect(const std::vector<Rect>& vec);

CV_WRAP static inline
void generateVectorOfRect(size_t len, CV_OUT std::vector<Rect>& vec)
{
    vec.resize(len);
    if (len > 0)
    {
        RNG rng(12345);
        Mat tmp(static_cast<int>(len), 1, CV_32SC4);
        rng.fill(tmp, RNG::UNIFORM, 10, 20);
        tmp.copyTo(vec);
    }
}

CV_WRAP static inline
void generateVectorOfInt(size_t len, CV_OUT std::vector<int>& vec)
{
    vec.resize(len);
    if (len > 0)
    {
        RNG rng(554433);
        Mat tmp(static_cast<int>(len), 1, CV_32SC1);
        rng.fill(tmp, RNG::UNIFORM, -10, 10);
        tmp.copyTo(vec);
    }
}

CV_WRAP static inline
void generateVectorOfMat(size_t len, int rows, int cols, int dtype, CV_OUT std::vector<Mat>& vec)
{
    vec.resize(len);
    if (len > 0)
    {
        RNG rng(65431);
        for (size_t i = 0; i < len; ++i)
        {
            vec[i].create(rows, cols, dtype);
            rng.fill(vec[i], RNG::UNIFORM, 0, 10);
        }
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

namespace nested {
CV_WRAP static inline bool testEchoBooleanFunction(bool flag) {
    return flag;
}

class CV_EXPORTS_W CV_WRAP_AS(ExportClassName) OriginalClassName
{
public:
    struct CV_EXPORTS_W_SIMPLE Params
    {
        CV_PROP_RW int int_value;
        CV_PROP_RW float float_value;

        CV_WRAP explicit Params(int int_param = 123, float float_param = 3.5f)
        {
            int_value = int_param;
            float_value = float_param;
        }
    };

    explicit OriginalClassName(const OriginalClassName::Params& params = OriginalClassName::Params())
    {
        params_ = params;
    }

    CV_WRAP int getIntParam() const
    {
        return params_.int_value;
    }

    CV_WRAP float getFloatParam() const
    {
        return params_.float_value;
    }

    CV_WRAP static std::string originalName()
    {
        return "OriginalClassName";
    }

    CV_WRAP static Ptr<OriginalClassName>
    create(const OriginalClassName::Params& params = OriginalClassName::Params())
    {
        return makePtr<OriginalClassName>(params);
    }

private:
    OriginalClassName::Params params_;
};

typedef OriginalClassName::Params OriginalClassName_Params;
} // namespace nested

namespace fs {
    CV_EXPORTS_W cv::String getCacheDirectoryForDownloads();
} // namespace fs

//! @}  // core_utils
}  // namespace cv::utils

//! @cond IGNORED

CV_WRAP static inline
int setLogLevel(int level)
{
    // NB: Binding generators doesn't work with enums properly yet, so we define separate overload here
    return cv::utils::logging::setLogLevel((cv::utils::logging::LogLevel)level);
}

CV_WRAP static inline
int getLogLevel()
{
    return cv::utils::logging::getLogLevel();
}

//! @endcond IGNORED

} // namespaces cv /  utils

#endif // OPENCV_CORE_BINDINGS_UTILS_HPP
