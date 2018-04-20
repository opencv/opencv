// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_CORE_CHECK_HPP
#define OPENCV_CORE_CHECK_HPP

#include <opencv2/core/base.hpp>

namespace cv {

/** Returns string of cv::Mat depth value: CV_8U -> "CV_8U" or "<invalid depth>" */
CV_EXPORTS const char* depthToString(int depth);

/** Returns string of cv::Mat depth value: CV_8UC3 -> "CV_8UC3" or "<invalid type>" */
CV_EXPORTS const String typeToString(int type);


//! @cond IGNORED
namespace detail {

/** Returns string of cv::Mat depth value: CV_8U -> "CV_8U" or NULL */
CV_EXPORTS const char* depthToString_(int depth);

/** Returns string of cv::Mat depth value: CV_8UC3 -> "CV_8UC3" or cv::String() */
CV_EXPORTS const cv::String typeToString_(int type);

enum TestOp {
  TEST_CUSTOM = 0,
  TEST_EQ = 1,
  TEST_NE = 2,
  TEST_LE = 3,
  TEST_LT = 4,
  TEST_GE = 5,
  TEST_GT = 6,
  CV__LAST_TEST_OP
};

struct CheckContext {
    const char* func;
    const char* file;
    int line;
    enum TestOp testOp;
    const char* message;
    const char* p1_str;
    const char* p2_str;
};

#ifndef CV__CHECK_FILENAME
# define CV__CHECK_FILENAME __FILE__
#endif

#ifndef CV__CHECK_FUNCTION
# if defined _MSC_VER
#   define CV__CHECK_FUNCTION __FUNCSIG__
# elif defined __GNUC__
#   define CV__CHECK_FUNCTION __PRETTY_FUNCTION__
# else
#   define CV__CHECK_FUNCTION "<unknown>"
# endif
#endif

#define CV__CHECK_LOCATION_VARNAME(id) CVAUX_CONCAT(CVAUX_CONCAT(__cv_check_, id), __LINE__)
#define CV__DEFINE_CHECK_CONTEXT(id, message, testOp, p1_str, p2_str) \
    static const cv::detail::CheckContext CV__CHECK_LOCATION_VARNAME(id) = \
            { CV__CHECK_FUNCTION, CV__CHECK_FILENAME, __LINE__, testOp, message, p1_str, p2_str }

CV_EXPORTS void CV_NORETURN check_failed_auto(const int v1, const int v2, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_auto(const float v1, const float v2, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_auto(const double v1, const double v2, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_MatDepth(const int v1, const int v2, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_MatType(const int v1, const int v2, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_MatChannels(const int v1, const int v2, const CheckContext& ctx);

CV_EXPORTS void CV_NORETURN check_failed_auto(const int v, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_auto(const float v, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_auto(const double v, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_MatDepth(const int v, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_MatType(const int v, const CheckContext& ctx);
CV_EXPORTS void CV_NORETURN check_failed_MatChannels(const int v, const CheckContext& ctx);


#define CV__TEST_EQ(v1, v2) ((v1) == (v2))
#define CV__TEST_NE(v1, v2) ((v1) != (v2))
#define CV__TEST_LE(v1, v2) ((v1) <= (v2))
#define CV__TEST_LT(v1, v2) ((v1) < (v2))
#define CV__TEST_GE(v1, v2) ((v1) >= (v2))
#define CV__TEST_GT(v1, v2) ((v1) > (v2))

#define CV__CHECK(id, op, type, v1, v2, v1_str, v2_str, msg_str) do { \
    if(CV__TEST_##op((v1), (v2))) ; else { \
        CV__DEFINE_CHECK_CONTEXT(id, msg_str, cv::detail::TEST_ ## op, v1_str, v2_str); \
        cv::detail::check_failed_ ## type((v1), (v2), CV__CHECK_LOCATION_VARNAME(id)); \
    } \
} while (0)

#define CV__CHECK_CUSTOM_TEST(id, type, v, test_expr, v_str, test_expr_str, msg_str) do { \
    if(!!(test_expr)) ; else { \
        CV__DEFINE_CHECK_CONTEXT(id, msg_str, cv::detail::TEST_CUSTOM, v_str, test_expr_str); \
        cv::detail::check_failed_ ## type((v), CV__CHECK_LOCATION_VARNAME(id)); \
    } \
} while (0)

} // namespace
//! @endcond


/// Supported values of these types: int, float, double
#define CV_CheckEQ(v1, v2, msg)  CV__CHECK(_, EQ, auto, v1, v2, #v1, #v2, msg)
#define CV_CheckNE(v1, v2, msg)  CV__CHECK(_, NE, auto, v1, v2, #v1, #v2, msg)
#define CV_CheckLE(v1, v2, msg)  CV__CHECK(_, LE, auto, v1, v2, #v1, #v2, msg)
#define CV_CheckLT(v1, v2, msg)  CV__CHECK(_, LT, auto, v1, v2, #v1, #v2, msg)
#define CV_CheckGE(v1, v2, msg)  CV__CHECK(_, GE, auto, v1, v2, #v1, #v2, msg)
#define CV_CheckGT(v1, v2, msg)  CV__CHECK(_, GT, auto, v1, v2, #v1, #v2, msg)

/// Check with additional "decoding" of type values in error message
#define CV_CheckTypeEQ(t1, t2, msg)  CV__CHECK(_, EQ, MatType, t1, t2, #t1, #t2, msg)
/// Check with additional "decoding" of depth values in error message
#define CV_CheckDepthEQ(d1, d2, msg)  CV__CHECK(_, EQ, MatDepth, d1, d2, #d1, #d2, msg)

#define CV_CheckChannelsEQ(c1, c2, msg)  CV__CHECK(_, EQ, MatChannels, c1, c2, #c1, #c2, msg)


/// Example: type == CV_8UC1 || type == CV_8UC3
#define CV_CheckType(t, test_expr, msg)  CV__CHECK_CUSTOM_TEST(_, MatType, t, (test_expr), #t, #test_expr, msg)

/// Example: depth == CV_32F || depth == CV_64F
#define CV_CheckDepth(t, test_expr, msg)  CV__CHECK_CUSTOM_TEST(_, MatDepth, t, (test_expr), #t, #test_expr, msg)

/// Some complex conditions: CV_Check(src2, src2.empty() || (src2.type() == src1.type() && src2.size() == src1.size()), "src2 should have same size/type as src1")
// TODO define pretty-printers: #define CV_Check(v, test_expr, msg)  CV__CHECK_CUSTOM_TEST(_, auto, v, (test_expr), #v, #test_expr, msg)

} // namespace

#endif // OPENCV_CORE_CHECK_HPP
