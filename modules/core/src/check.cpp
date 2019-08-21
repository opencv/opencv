// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "opencv2/core/check.hpp"

namespace cv {

const char* depthToString(int depth)
{
    const char* s = detail::depthToString_(depth);
    return s ? s : "<invalid depth>";
}

const cv::String typeToString(int type)
{
    cv::String s = detail::typeToString_(type);
    if (s.empty())
    {
        static cv::String invalidType("<invalid type>");
        return invalidType;
    }
    return s;
}


namespace detail {

static const char* getTestOpPhraseStr(unsigned testOp)
{
    static const char* _names[] = { "{custom check}", "equal to", "not equal to", "less than or equal to", "less than", "greater than or equal to", "greater than" };
    CV_DbgAssert(testOp < CV__LAST_TEST_OP);
    return testOp < CV__LAST_TEST_OP ? _names[testOp] : "???";
}
static const char* getTestOpMath(unsigned testOp)
{
    static const char* _names[] = { "???", "==", "!=", "<=", "<", ">=", ">" };
    CV_DbgAssert(testOp < CV__LAST_TEST_OP);
    return testOp < CV__LAST_TEST_OP ? _names[testOp] : "???";
}

const char* depthToString_(int depth)
{
    static const char* depthNames[] = { "CV_8U", "CV_8S", "CV_16U", "CV_16S", "CV_32S", "CV_32F", "CV_64F", "CV_USRTYPE1" };
    return (depth <= CV_USRTYPE1 && depth >= 0) ? depthNames[depth] : NULL;
}

const cv::String typeToString_(int type)
{
    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);
    if (depth >= 0 && depth <= CV_USRTYPE1)
        return cv::format("%sC%d", depthToString_(depth), cn);
    return cv::String();
}

template<typename T> static CV_NORETURN
void check_failed_auto_(const T& v1, const T& v2, const CheckContext& ctx)
{
    std::stringstream ss;
    ss  << ctx.message << " (expected: '" << ctx.p1_str << " " << getTestOpMath(ctx.testOp) << " " << ctx.p2_str << "'), where" << std::endl
        << "    '" << ctx.p1_str << "' is " << v1 << std::endl;
    if (ctx.testOp != TEST_CUSTOM && ctx.testOp < CV__LAST_TEST_OP)
    {
        ss << "must be " << getTestOpPhraseStr(ctx.testOp) << std::endl;
    }
    ss  << "    '" << ctx.p2_str << "' is " << v2;
    cv::errorNoReturn(cv::Error::StsError, ss.str(), ctx.func, ctx.file, ctx.line);
}
void check_failed_MatDepth(const int v1, const int v2, const CheckContext& ctx)
{
    std::stringstream ss;
    ss  << ctx.message << " (expected: '" << ctx.p1_str << " " << getTestOpMath(ctx.testOp) << " " << ctx.p2_str << "'), where" << std::endl
        << "    '" << ctx.p1_str << "' is " << v1 << " (" << depthToString(v1) << ")" << std::endl;
    if (ctx.testOp != TEST_CUSTOM && ctx.testOp < CV__LAST_TEST_OP)
    {
        ss << "must be " << getTestOpPhraseStr(ctx.testOp) << std::endl;
    }
    ss  << "    '" << ctx.p2_str << "' is " << v2 << " (" << depthToString(v2) << ")";
    cv::errorNoReturn(cv::Error::StsError, ss.str(), ctx.func, ctx.file, ctx.line);
}
void check_failed_MatType(const int v1, const int v2, const CheckContext& ctx)
{
    std::stringstream ss;
    ss  << ctx.message << " (expected: '" << ctx.p1_str << " " << getTestOpMath(ctx.testOp) << " " << ctx.p2_str << "'), where" << std::endl
        << "    '" << ctx.p1_str << "' is " << v1 << " (" << typeToString(v1) << ")" << std::endl;
    if (ctx.testOp != TEST_CUSTOM && ctx.testOp < CV__LAST_TEST_OP)
    {
        ss << "must be " << getTestOpPhraseStr(ctx.testOp) << std::endl;
    }
    ss  << "    '" << ctx.p2_str << "' is " << v2 << " (" << typeToString(v2) << ")";
    cv::errorNoReturn(cv::Error::StsError, ss.str(), ctx.func, ctx.file, ctx.line);
}
void check_failed_MatChannels(const int v1, const int v2, const CheckContext& ctx)
{
    check_failed_auto_<int>(v1, v2, ctx);
}
void check_failed_auto(const int v1, const int v2, const CheckContext& ctx)
{
    check_failed_auto_<int>(v1, v2, ctx);
}
void check_failed_auto(const size_t v1, const size_t v2, const CheckContext& ctx)
{
    check_failed_auto_<size_t>(v1, v2, ctx);
}
void check_failed_auto(const float v1, const float v2, const CheckContext& ctx)
{
    check_failed_auto_<float>(v1, v2, ctx);
}
void check_failed_auto(const double v1, const double v2, const CheckContext& ctx)
{
    check_failed_auto_<double>(v1, v2, ctx);
}


template<typename T> static CV_NORETURN
void check_failed_auto_(const T& v, const CheckContext& ctx)
{
    std::stringstream ss;
    ss  << ctx.message << ":" << std::endl
        << "    '" << ctx.p2_str << "'" << std::endl
        << "where" << std::endl
        << "    '" << ctx.p1_str << "' is " << v;
    cv::errorNoReturn(cv::Error::StsError, ss.str(), ctx.func, ctx.file, ctx.line);
}
void check_failed_MatDepth(const int v, const CheckContext& ctx)
{
    std::stringstream ss;
    ss  << ctx.message << ":" << std::endl
        << "    '" << ctx.p2_str << "'" << std::endl
        << "where" << std::endl
        << "    '" << ctx.p1_str << "' is " << v << " (" << depthToString(v) << ")";
    cv::errorNoReturn(cv::Error::StsError, ss.str(), ctx.func, ctx.file, ctx.line);
}
void check_failed_MatType(const int v, const CheckContext& ctx)
{
    std::stringstream ss;
    ss  << ctx.message << ":" << std::endl
        << "    '" << ctx.p2_str << "'" << std::endl
        << "where" << std::endl
        << "    '" << ctx.p1_str << "' is " << v << " (" << typeToString(v) << ")";
    cv::errorNoReturn(cv::Error::StsError, ss.str(), ctx.func, ctx.file, ctx.line);
}
void check_failed_MatChannels(const int v, const CheckContext& ctx)
{
    check_failed_auto_<int>(v, ctx);
}
void check_failed_auto(const int v, const CheckContext& ctx)
{
    check_failed_auto_<int>(v, ctx);
}
void check_failed_auto(const size_t v, const CheckContext& ctx)
{
    check_failed_auto_<size_t>(v, ctx);
}
void check_failed_auto(const float v, const CheckContext& ctx)
{
    check_failed_auto_<float>(v, ctx);
}
void check_failed_auto(const double v, const CheckContext& ctx)
{
    check_failed_auto_<double>(v, ctx);
}


}} // namespace
