// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation

#ifndef OPENCV_GAPI_TESTS_COMMON_HPP
#define OPENCV_GAPI_TESTS_COMMON_HPP

#include <iostream>
#include <tuple>
#include <type_traits>

#include <opencv2/ts.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/util/util.hpp>

#include "gapi_tests_helpers.hpp"
#include <opencv2/gapi/render/render.hpp>

namespace
{
    inline std::ostream& operator<<(std::ostream& o, const cv::GCompileArg& arg)
    {
        return o << (arg.tag.empty() ? "empty" : arg.tag);
    }

    inline std::ostream& operator<<(std::ostream& o, const cv::gapi::wip::draw::Prim& p)
    {
        using namespace cv::gapi::wip::draw;
        switch (p.index())
        {
            case Prim::index_of<Rect>():
                o << "cv::gapi::draw::Rect";
                break;
            case Prim::index_of<Text>():
                o << "cv::gapi::draw::Text";
                break;
            case Prim::index_of<Circle>():
                o << "cv::gapi::draw::Circle";
                break;
            case Prim::index_of<Line>():
                o << "cv::gapi::draw::Line";
                break;
            case Prim::index_of<Mosaic>():
                o << "cv::gapi::draw::Mosaic";
                break;
            case Prim::index_of<Image>():
                o << "cv::gapi::draw::Image";
                break;
            case Prim::index_of<Poly>():
                o << "cv::gapi::draw::Poly";
                break;
            default: o << "Unrecognized primitive";
        }

        return o;
    }
}

namespace opencv_test
{

class TestFunctional
{
public:
    cv::Mat in_mat1;
    cv::Mat in_mat2;
    cv::Mat out_mat_gapi;
    cv::Mat out_mat_ocv;

    cv::Scalar sc;

    cv::Scalar initScalarRandU(unsigned upper)
    {
        auto& rng = cv::theRNG();
        double s1 = rng(upper);  // FIXIT: RNG result is 'int', not double
        double s2 = rng(upper);
        double s3 = rng(upper);
        double s4 = rng(upper);
        return cv::Scalar(s1, s2, s3, s4);
    }

    void initOutMats(cv::Size sz_in, int dtype)
    {
        if (dtype != -1)
        {
            out_mat_gapi = cv::Mat(sz_in, dtype);
            out_mat_ocv = cv::Mat(sz_in, dtype);
        }
    }

    void initMatsRandU(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1 = cv::Mat(sz_in, type);
        in_mat2 = cv::Mat(sz_in, type);

        sc = initScalarRandU(100);

        // Details: https://github.com/opencv/opencv/pull/16083
        //if (CV_MAT_DEPTH(type) < CV_32F)
        if (1)
        {
            cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
            cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
        }
        else
        {
            const int fscale = 256;  // avoid bits near ULP, generate stable test input
            Mat in_mat32s(in_mat1.size(), CV_MAKE_TYPE(CV_32S, CV_MAT_CN(type)));
            cv::randu(in_mat32s, cv::Scalar::all(0), cv::Scalar::all(255 * fscale));
            in_mat32s.convertTo(in_mat1, type, 1.0f / fscale, 0);

            cv::randu(in_mat32s, cv::Scalar::all(0), cv::Scalar::all(255 * fscale));
            in_mat32s.convertTo(in_mat2, type, 1.0f / fscale, 0);
        }

        if (createOutputMatrices)
        {
            initOutMats(sz_in, dtype);
        }
    }

    void initMatrixRandU(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1 = cv::Mat(sz_in, type);

        sc = initScalarRandU(100);
        if (CV_MAT_DEPTH(type) < CV_32F)
        {
            cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
        }
        else
        {
            const int fscale = 256;  // avoid bits near ULP, generate stable test input
            Mat in_mat32s(in_mat1.size(), CV_MAKE_TYPE(CV_32S, CV_MAT_CN(type)));
            cv::randu(in_mat32s, cv::Scalar::all(0), cv::Scalar::all(255 * fscale));
            in_mat32s.convertTo(in_mat1, type, 1.0f / fscale, 0);
        }

        if (createOutputMatrices)
        {
            initOutMats(sz_in, dtype);
        }
    }

    void initMatrixRandN(int type, cv::Size sz_in, int dtype, bool createOutputMatrices = true)
    {
        in_mat1 = cv::Mat(sz_in, type);
        cv::randn(in_mat1, cv::Scalar::all(127), cv::Scalar::all(40.f));

        if (createOutputMatrices)
        {
            initOutMats(sz_in, dtype);
        }
    }

    // empty function intended to show that nothing is to be initialized via TestFunctional methods
    void initNothing(int, cv::Size, int, bool = true) {}
};

template<class T>
class TestParams: public TestFunctional, public TestWithParam<T>{};

template<class T>
class TestPerfParams: public TestFunctional, public perf::TestBaseWithParam<T>{};

using compare_f = std::function<bool(const cv::Mat &a, const cv::Mat &b)>;

using compare_scalar_f = std::function<bool(const cv::Scalar &a, const cv::Scalar &b)>;

// FIXME: re-use MatType. current problem: "special values" interpreted incorrectly (-1 is printed
//        as 16FC512)
struct MatType2
{
public:
    MatType2(int val = 0) : _value(val) {}
    operator int() const { return _value; }
    friend std::ostream& operator<<(std::ostream& os, const MatType2& t)
    {
        switch (t)
        {
            case -1: return os << "SAME_TYPE";
            default: PrintTo(MatType(t), &os); return os;
        }
    }
private:
    int _value;
};

// Universal parameter wrapper for common (pre-defined) and specific (user-defined) parameters
template<typename ...SpecificParams>
struct Params
{
    using gcomp_args_function_t = cv::GCompileArgs(*)();
    using common_params_t = std::tuple<MatType2, cv::Size, MatType2, gcomp_args_function_t>;
    using specific_params_t = std::tuple<SpecificParams...>;
    using params_t = std::tuple<MatType2, cv::Size, MatType2, gcomp_args_function_t, SpecificParams...>;
    static constexpr const size_t common_params_size = std::tuple_size<common_params_t>::value;
    static constexpr const size_t specific_params_size = std::tuple_size<specific_params_t>::value;

    template<size_t I>
    static const typename std::tuple_element<I, common_params_t>::type&
    getCommon(const params_t& t)
    {
        static_assert(I < common_params_size, "Index out of range");
        return std::get<I>(t);
    }

    template<size_t I>
    static const typename std::tuple_element<I, specific_params_t>::type&
    getSpecific(const params_t& t)
    {
        static_assert(specific_params_size > 0,
            "Impossible to call this function: no specific parameters specified");
        static_assert(I < specific_params_size, "Index out of range");
        return std::get<common_params_size + I>(t);
    }
};

// Base class for test fixtures
template<typename ...SpecificParams>
struct TestWithParamBase : TestFunctional,
    TestWithParam<typename Params<SpecificParams...>::params_t>
{
    using AllParams = Params<SpecificParams...>;

    MatType2 type = getCommonParam<0>();
    cv::Size sz = getCommonParam<1>();
    MatType2 dtype = getCommonParam<2>();

    // Get common (pre-defined) parameter value by index
    template<size_t I>
    inline auto getCommonParam() const
        -> decltype(AllParams::template getCommon<I>(this->GetParam()))
    {
        return AllParams::template getCommon<I>(this->GetParam());
    }

    // Get specific (user-defined) parameter value by index
    template<size_t I>
    inline auto getSpecificParam() const
        -> decltype(AllParams::template getSpecific<I>(this->GetParam()))
    {
        return AllParams::template getSpecific<I>(this->GetParam());
    }

    // Return G-API compile arguments specified for test fixture
    inline cv::GCompileArgs getCompileArgs() const
    {
        return getCommonParam<3>()();
    }
};

/**
 * @private
 * @brief Create G-API test fixture with TestWithParamBase base class
 * @param Fixture   test fixture name
 * @param InitF     callable that will initialize default available members (from TestFunctional)
 * @param API       base class API. Specifies types of user-defined parameters. If there are no such
 *                  parameters, empty angle brackets ("<>") must be specified.
 * @param Number    number of user-defined parameters (corresponds to the number of types in API).
 *                  if there are no such parameters, 0 must be specified.
 * @param ...       list of names of user-defined parameters. if there are no parameters, the list
 *                  must be empty.
 */
#define GAPI_TEST_FIXTURE(Fixture, InitF, API, Number, ...) \
    struct Fixture : public TestWithParamBase API { \
        static_assert(Number == AllParams::specific_params_size, \
            "Number of user-defined parameters doesn't match size of __VA_ARGS__"); \
        __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__)) \
        Fixture() { InitF(type, sz, dtype); } \
    };

// Wrapper for test fixture API. Use to specify multiple types.
// Example: FIXTURE_API(int, bool) expands to <int, bool>
#define FIXTURE_API(...) <__VA_ARGS__>

template<typename T1, typename T2>
struct CompareF
{
    using callable_t = std::function<bool(const T1& a, const T2& b)>;
    CompareF(callable_t&& cmp, std::string&& cmp_name) :
        _comparator(std::move(cmp)), _name(std::move(cmp_name)) {}
    bool operator()(const T1& a, const T2& b) const
    {
        return _comparator(a, b);
    }
    friend std::ostream& operator<<(std::ostream& os, const CompareF<T1, T2>& obj)
    {
        return os << obj._name;
    }
private:
    callable_t _comparator;
    std::string _name;
};

using CompareMats = CompareF<cv::Mat, cv::Mat>;
using CompareScalars = CompareF<cv::Scalar, cv::Scalar>;

template<typename T>
struct Wrappable
{
    compare_f to_compare_f()
    {
        T t = *static_cast<T*const>(this);
        return [t](const cv::Mat &a, const cv::Mat &b)
        {
            return t(a, b);
        };
    }

    CompareMats to_compare_obj()
    {
        T t = *static_cast<T*const>(this);
        std::stringstream ss;
        ss << t;
        return CompareMats(to_compare_f(), ss.str());
    }
};

template<typename T>
struct WrappableScalar
{
    compare_scalar_f to_compare_f()
    {
        T t = *static_cast<T*const>(this);
        return [t](const cv::Scalar &a, const cv::Scalar &b)
        {
            return t(a, b);
        };
    }

    CompareScalars to_compare_obj()
    {
        T t = *static_cast<T*const>(this);
        std::stringstream ss;
        ss << t;
        return CompareScalars(to_compare_f(), ss.str());
    }
};


class AbsExact : public Wrappable<AbsExact>
{
public:
    AbsExact() {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (cv::norm(in1, in2, NORM_INF) != 0)
        {
            std::cout << "AbsExact error: G-API output and reference output matrixes are not bitexact equal."  << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const AbsExact&)
    {
        return os << "AbsExact()";
    }
};

class AbsTolerance : public Wrappable<AbsTolerance>
{
public:
    AbsTolerance(double tol) : _tol(tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        if (cv::norm(in1, in2, NORM_INF) > _tol)
        {
            std::cout << "AbsTolerance error: Number of different pixels in " << std::endl;
            std::cout << "G-API output and reference output matrixes exceeds " << _tol << " pixels threshold." << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const AbsTolerance& obj)
    {
        return os << "AbsTolerance(" << std::to_string(obj._tol) << ")";
    }
private:
    double _tol;
};

class Tolerance_FloatRel_IntAbs : public Wrappable<Tolerance_FloatRel_IntAbs>
{
public:
    Tolerance_FloatRel_IntAbs(double tol, double tol8u) : _tol(tol), _tol8u(tol8u) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        int depth = CV_MAT_DEPTH(in1.type());
        {
            double err = depth >= CV_32F ? cv::norm(in1, in2, NORM_L1 | NORM_RELATIVE)
                                                     : cv::norm(in1, in2, NORM_INF);
            double tolerance = depth >= CV_32F ? _tol : _tol8u;
            if (err > tolerance)
            {
                std::cout << "Tolerance_FloatRel_IntAbs error: err=" << err
                          << "  tolerance=" << tolerance
                          << "  depth=" << cv::typeToString(depth) << std::endl;
                return false;
            }
            else
            {
                return true;
            }
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const Tolerance_FloatRel_IntAbs& obj)
    {
        return os << "Tolerance_FloatRel_IntAbs(" << obj._tol << ", " << obj._tol8u << ")";
    }
private:
    double _tol;
    double _tol8u;
};


class AbsSimilarPoints : public Wrappable<AbsSimilarPoints>
{
public:
    AbsSimilarPoints(double tol, double percent) : _tol(tol), _percent(percent) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        Mat diff;
        cv::absdiff(in1, in2, diff);
        Mat err_mask = diff > _tol;
        int err_points = (cv::countNonZero)(err_mask.reshape(1));
        double max_err_points = _percent * std::max((size_t)1000, in1.total());
        if (err_points > max_err_points)
        {
            std::cout << "AbsSimilarPoints error: err_points=" << err_points
                      << "  max_err_points=" << max_err_points << " (total=" << in1.total() << ")"
                      << "  diff_tolerance=" << _tol << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const AbsSimilarPoints& obj)
    {
        return os << "AbsSimilarPoints(" << obj._tol << ", " << obj._percent << ")";
    }
private:
    double _tol;
    double _percent;
};


class ToleranceFilter : public Wrappable<ToleranceFilter>
{
public:
    ToleranceFilter(double tol, double tol8u, double inf_tol = 2.0) : _tol(tol), _tol8u(tol8u), _inf_tol(inf_tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        int depth = CV_MAT_DEPTH(in1.type());
        {
            double err_Inf = cv::norm(in1, in2, NORM_INF);
            if (err_Inf > _inf_tol)
            {
                std::cout << "ToleranceFilter error: err_Inf=" << err_Inf << "  tolerance=" << _inf_tol << std::endl;
                return false;
            }
            double err = cv::norm(in1, in2, NORM_L2 | NORM_RELATIVE);
            double tolerance = depth >= CV_32F ? _tol : _tol8u;
            if (err > tolerance)
            {
                std::cout << "ToleranceFilter error: err=" << err << "  tolerance=" << tolerance
                          << "  depth=" << cv::depthToString(depth)
                          << std::endl;
                return false;
            }
        }
        return true;
    }
    friend std::ostream& operator<<(std::ostream& os, const ToleranceFilter& obj)
    {
        return os << "ToleranceFilter(" << obj._tol << ", " << obj._tol8u << ", "
                  << obj._inf_tol << ")";
    }
private:
    double _tol;
    double _tol8u;
    double _inf_tol;
};

class ToleranceColor : public Wrappable<ToleranceColor>
{
public:
    ToleranceColor(double tol, double inf_tol = 2.0) : _tol(tol), _inf_tol(inf_tol) {}
    bool operator() (const cv::Mat& in1, const cv::Mat& in2) const
    {
        {
            double err_Inf = cv::norm(in1, in2, NORM_INF);
            if (err_Inf > _inf_tol)
            {
                std::cout << "ToleranceColor error: err_Inf=" << err_Inf << "  tolerance=" << _inf_tol << std::endl;;
                return false;
            }
            double err = cv::norm(in1, in2, NORM_L1 | NORM_RELATIVE);
            if (err > _tol)
            {
                std::cout << "ToleranceColor error: err=" << err << "  tolerance=" << _tol << std::endl;;
                return false;
            }
        }
        return true;
    }
    friend std::ostream& operator<<(std::ostream& os, const ToleranceColor& obj)
    {
        return os << "ToleranceColor(" << obj._tol << ", " << obj._inf_tol << ")";
    }
private:
    double _tol;
    double _inf_tol;
};

class AbsToleranceScalar : public WrappableScalar<AbsToleranceScalar>
{
public:
    AbsToleranceScalar(double tol) : _tol(tol) {}
    bool operator() (const cv::Scalar& in1, const cv::Scalar& in2) const
    {
        double abs_err = std::abs(in1[0] - in2[0]) / std::max(1.0, std::abs(in2[0]));
        if (abs_err > _tol)
        {
            std::cout << "AbsToleranceScalar error: abs_err=" << abs_err << "  tolerance=" << _tol << " in1[0]" << in1[0] << " in2[0]" << in2[0] << std::endl;;
            return false;
        }
        else
        {
            return true;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const AbsToleranceScalar& obj)
    {
        return os << "AbsToleranceScalar(" << std::to_string(obj._tol) << ")";
    }
private:
    double _tol;
};
} // namespace opencv_test

namespace
{
inline std::ostream& operator<<(std::ostream& os, const opencv_test::compare_f&)
{
    return os << "compare_f";
}

inline std::ostream& operator<<(std::ostream& os, const opencv_test::compare_scalar_f&)
{
    return os << "compare_scalar_f";
}
}  // anonymous namespace

// Note: namespace must match the namespace of the type of the printed object
namespace cv
{
inline std::ostream& operator<<(std::ostream& os, CmpTypes op)
{
#define CASE(v) case CmpTypes::v: os << #v; break
    switch (op)
    {
        CASE(CMP_EQ);
        CASE(CMP_GT);
        CASE(CMP_GE);
        CASE(CMP_LT);
        CASE(CMP_LE);
        CASE(CMP_NE);
        default: GAPI_Assert(false && "unknown CmpTypes value");
    }
#undef CASE
    return os;
}

inline std::ostream& operator<<(std::ostream& os, NormTypes op)
{
#define CASE(v) case NormTypes::v: os << #v; break
    switch (op)
    {
        CASE(NORM_INF);
        CASE(NORM_L1);
        CASE(NORM_L2);
        CASE(NORM_L2SQR);
        CASE(NORM_HAMMING);
        CASE(NORM_HAMMING2);
        CASE(NORM_RELATIVE);
        CASE(NORM_MINMAX);
        default: GAPI_Assert(false && "unknown NormTypes value");
    }
#undef CASE
    return os;
}
}  // namespace cv

#endif //OPENCV_GAPI_TESTS_COMMON_HPP
