// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRACE_HPP
#define OPENCV_TRACE_HPP

#include <opencv2/core/cvdef.h>

//! @addtogroup core_logging
// This section describes OpenCV tracing utilities.
//
//! @{

namespace cv {
namespace utils {
namespace trace {

//! Macro to trace function
#define CV_TRACE_FUNCTION()

#define CV_TRACE_FUNCTION_SKIP_NESTED()

//! Trace code scope.
//! @note Dynamic names are not supported in this macro (on stack or heap). Use string literals here only, like "initialize".
#define CV_TRACE_REGION(name_as_static_string_literal)
//! mark completed of the current opened region and create new one
//! @note Dynamic names are not supported in this macro (on stack or heap). Use string literals here only, like "step1".
#define CV_TRACE_REGION_NEXT(name_as_static_string_literal)

//! Macro to trace argument value
#define CV_TRACE_ARG(arg_id)

//! Macro to trace argument value (expanded version)
#define CV_TRACE_ARG_VALUE(arg_id, arg_name, value)

//! @cond IGNORED
#define CV_TRACE_NS cv::utils::trace

namespace details {

#ifndef __OPENCV_TRACE
# if defined __OPENCV_BUILD && !defined __OPENCV_TESTS && !defined __OPENCV_APPS
#   define __OPENCV_TRACE 1
# else
#   define __OPENCV_TRACE 0
# endif
#endif

#ifndef CV_TRACE_FILENAME
# define CV_TRACE_FILENAME __FILE__
#endif

#ifndef CV__TRACE_FUNCTION
# if defined _MSC_VER
#   define CV__TRACE_FUNCTION __FUNCSIG__
# elif defined __GNUC__
#   define CV__TRACE_FUNCTION __PRETTY_FUNCTION__
# else
#   define CV__TRACE_FUNCTION "<unknown>"
# endif
#endif

//! Thread-local instance (usually allocated on stack)
class CV_EXPORTS Region
{
public:
    struct LocationExtraData;
    struct LocationStaticStorage
    {
        LocationExtraData** ppExtra;   //< implementation specific data
        const char* name;              //< region name (function name or other custom name)
        const char* filename;          //< source code filename
        int line;                      //< source code line
        int flags;                     //< flags (implementation code path: Plain, IPP, OpenCL)
    };

    Region(const LocationStaticStorage& location);
    inline ~Region()
    {
        if (implFlags != 0)
            destroy();
        CV_DbgAssert(implFlags == 0);
        CV_DbgAssert(pImpl == NULL);
    }

    class Impl;
    Impl* pImpl; // NULL if current region is not active
    int implFlags; // see RegionFlag, 0 if region is ignored

    bool isActive() const { return pImpl != NULL; }

    void destroy();
private:
    Region(const Region&); // disabled
    Region& operator= (const Region&); // disabled
};

//! Specify region flags
enum RegionLocationFlag {
    REGION_FLAG_FUNCTION = (1 << 0),             //< region is function (=1) / nested named region (=0)
    REGION_FLAG_APP_CODE = (1 << 1),             //< region is Application code (=1) / OpenCV library code (=0)
    REGION_FLAG_SKIP_NESTED = (1 << 2),          //< avoid processing of nested regions

    REGION_FLAG_IMPL_IPP = (1 << 16),            //< region is part of IPP code path
    REGION_FLAG_IMPL_OPENCL = (2 << 16),         //< region is part of OpenCL code path
    REGION_FLAG_IMPL_OPENVX = (3 << 16),         //< region is part of OpenVX code path

    REGION_FLAG_IMPL_MASK = (15 << 16),

    REGION_FLAG_REGION_FORCE = (1 << 30),
    REGION_FLAG_REGION_NEXT = (1 << 31),         //< close previous region (see #CV_TRACE_REGION_NEXT macro)

    ENUM_REGION_FLAG_FORCE_INT = INT_MAX
};

struct CV_EXPORTS TraceArg {
public:
    struct ExtraData;
    ExtraData** ppExtra;
    const char* name;
    int flags;
};
/** @brief Add meta information to current region (function)
 * See CV_TRACE_ARG macro
 * @param arg argument information structure (global static cache)
 * @param value argument value (can by dynamic string literal in case of string, static allocation is not required)
 */
CV_EXPORTS void traceArg(const TraceArg& arg, const char* value);
//! @overload
CV_EXPORTS void traceArg(const TraceArg& arg, int value);
//! @overload
CV_EXPORTS void traceArg(const TraceArg& arg, int64 value);
//! @overload
CV_EXPORTS void traceArg(const TraceArg& arg, double value);

#define CV__TRACE_LOCATION_VARNAME(loc_id) CVAUX_CONCAT(CVAUX_CONCAT(__cv_trace_location_, loc_id), __LINE__)
#define CV__TRACE_LOCATION_EXTRA_VARNAME(loc_id) CVAUX_CONCAT(CVAUX_CONCAT(__cv_trace_location_extra_, loc_id) , __LINE__)

#define CV__TRACE_DEFINE_LOCATION_(loc_id, name, flags) \
    static CV_TRACE_NS::details::Region::LocationExtraData* CV__TRACE_LOCATION_EXTRA_VARNAME(loc_id) = 0; \
    static const CV_TRACE_NS::details::Region::LocationStaticStorage \
        CV__TRACE_LOCATION_VARNAME(loc_id) = { &(CV__TRACE_LOCATION_EXTRA_VARNAME(loc_id)), name, CV_TRACE_FILENAME, __LINE__, flags};

#define CV__TRACE_DEFINE_LOCATION_FN(name, flags) CV__TRACE_DEFINE_LOCATION_(fn, name, (flags | CV_TRACE_NS::details::REGION_FLAG_FUNCTION))


#define CV__TRACE_OPENCV_FUNCTION() \
    CV__TRACE_DEFINE_LOCATION_FN(CV__TRACE_FUNCTION, 0); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));

#define CV__TRACE_OPENCV_FUNCTION_NAME(name) \
    CV__TRACE_DEFINE_LOCATION_FN(name, 0); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));

#define CV__TRACE_APP_FUNCTION() \
    CV__TRACE_DEFINE_LOCATION_FN(CV__TRACE_FUNCTION, CV_TRACE_NS::details::REGION_FLAG_APP_CODE); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));

#define CV__TRACE_APP_FUNCTION_NAME(name) \
    CV__TRACE_DEFINE_LOCATION_FN(name, CV_TRACE_NS::details::REGION_FLAG_APP_CODE); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));


#define CV__TRACE_OPENCV_FUNCTION_SKIP_NESTED() \
    CV__TRACE_DEFINE_LOCATION_FN(CV__TRACE_FUNCTION, CV_TRACE_NS::details::REGION_FLAG_SKIP_NESTED); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));

#define CV__TRACE_OPENCV_FUNCTION_NAME_SKIP_NESTED(name) \
    CV__TRACE_DEFINE_LOCATION_FN(name, CV_TRACE_NS::details::REGION_FLAG_SKIP_NESTED); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));

#define CV__TRACE_APP_FUNCTION_SKIP_NESTED() \
    CV__TRACE_DEFINE_LOCATION_FN(CV__TRACE_FUNCTION, CV_TRACE_NS::details::REGION_FLAG_SKIP_NESTED | CV_TRACE_NS::details::REGION_FLAG_APP_CODE); \
    const CV_TRACE_NS::details::Region __region_fn(CV__TRACE_LOCATION_VARNAME(fn));


#define CV__TRACE_REGION_(name_as_static_string_literal, flags) \
    CV__TRACE_DEFINE_LOCATION_(region, name_as_static_string_literal, flags); \
    CV_TRACE_NS::details::Region CVAUX_CONCAT(__region_, __LINE__)(CV__TRACE_LOCATION_VARNAME(region));

#define CV__TRACE_REGION(name_as_static_string_literal) CV__TRACE_REGION_(name_as_static_string_literal, 0)
#define CV__TRACE_REGION_NEXT(name_as_static_string_literal) CV__TRACE_REGION_(name_as_static_string_literal, CV_TRACE_NS::details::REGION_FLAG_REGION_NEXT)

#define CV__TRACE_ARG_VARNAME(arg_id) CVAUX_CONCAT(__cv_trace_arg_ ## arg_id, __LINE__)
#define CV__TRACE_ARG_EXTRA_VARNAME(arg_id) CVAUX_CONCAT(__cv_trace_arg_extra_ ## arg_id, __LINE__)

#define CV__TRACE_DEFINE_ARG_(arg_id, name, flags) \
    static CV_TRACE_NS::details::TraceArg::ExtraData* CV__TRACE_ARG_EXTRA_VARNAME(arg_id) = 0; \
    static const CV_TRACE_NS::details::TraceArg \
        CV__TRACE_ARG_VARNAME(arg_id) = { &(CV__TRACE_ARG_EXTRA_VARNAME(arg_id)), name, flags };

#define CV__TRACE_ARG_VALUE(arg_id, arg_name, value) \
        CV__TRACE_DEFINE_ARG_(arg_id, arg_name, 0); \
        CV_TRACE_NS::details::traceArg((CV__TRACE_ARG_VARNAME(arg_id)), value);

#define CV__TRACE_ARG(arg_id) CV_TRACE_ARG_VALUE(arg_id, #arg_id, (arg_id))

} // namespace

#ifndef OPENCV_DISABLE_TRACE
#undef CV_TRACE_FUNCTION
#undef CV_TRACE_FUNCTION_SKIP_NESTED
#if __OPENCV_TRACE
#define CV_TRACE_FUNCTION CV__TRACE_OPENCV_FUNCTION
#define CV_TRACE_FUNCTION_SKIP_NESTED CV__TRACE_OPENCV_FUNCTION_SKIP_NESTED
#else
#define CV_TRACE_FUNCTION CV__TRACE_APP_FUNCTION
#define CV_TRACE_FUNCTION_SKIP_NESTED CV__TRACE_APP_FUNCTION_SKIP_NESTED
#endif

#undef CV_TRACE_REGION
#define CV_TRACE_REGION CV__TRACE_REGION

#undef CV_TRACE_REGION_NEXT
#define CV_TRACE_REGION_NEXT CV__TRACE_REGION_NEXT

#undef CV_TRACE_ARG_VALUE
#define CV_TRACE_ARG_VALUE(arg_id, arg_name, value) \
        if (__region_fn.isActive()) \
        { \
            CV__TRACE_ARG_VALUE(arg_id, arg_name, value); \
        }

#undef CV_TRACE_ARG
#define CV_TRACE_ARG CV__TRACE_ARG

#endif // OPENCV_DISABLE_TRACE

#ifdef OPENCV_TRACE_VERBOSE
#define CV_TRACE_FUNCTION_VERBOSE CV_TRACE_FUNCTION
#define CV_TRACE_REGION_VERBOSE CV_TRACE_REGION
#define CV_TRACE_REGION_NEXT_VERBOSE CV_TRACE_REGION_NEXT
#define CV_TRACE_ARG_VALUE_VERBOSE CV_TRACE_ARG_VALUE
#define CV_TRACE_ARG_VERBOSE CV_TRACE_ARG
#else
#define CV_TRACE_FUNCTION_VERBOSE(...)
#define CV_TRACE_REGION_VERBOSE(...)
#define CV_TRACE_REGION_NEXT_VERBOSE(...)
#define CV_TRACE_ARG_VALUE_VERBOSE(...)
#define CV_TRACE_ARG_VERBOSE(...)
#endif

//! @endcond

}}} // namespace

//! @}

#endif // OPENCV_TRACE_HPP
