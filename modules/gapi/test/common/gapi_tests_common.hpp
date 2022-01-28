// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation

#ifndef OPENCV_GAPI_TESTS_COMMON_HPP
#define OPENCV_GAPI_TESTS_COMMON_HPP

#include <iostream>
#include <tuple>
#include <type_traits>
#include <time.h>

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

    template <typename T> inline void initPointRandU(cv::RNG &rng, cv::Point_<T>& pt)
    {
        GAPI_Assert(std::is_integral<T>::value);
        pt = cv::Point_<T>(static_cast<T>(static_cast<char>(rng(CHAR_MAX + 1U))),
                           static_cast<T>(static_cast<char>(rng(CHAR_MAX + 1U))));
    }

    template <typename T> inline void initPointRandU(cv::RNG &rng, cv::Point3_<T>& pt)
    {
        GAPI_Assert(std::is_integral<T>::value);
        pt = cv::Point3_<T>(static_cast<T>(static_cast<char>(rng(CHAR_MAX + 1U))),
                            static_cast<T>(static_cast<char>(rng(CHAR_MAX + 1U))),
                            static_cast<T>(static_cast<char>(rng(CHAR_MAX + 1U))));
    }

    template <typename F> inline void initFloatPointRandU(cv::RNG &rng, cv::Point_<F> &pt)
    {
        GAPI_Assert(std::is_floating_point<F>::value);
        static const int fscale = 256;  // avoid bits near ULP, generate stable test input
        pt = cv::Point_<F>(rng.uniform(0, 255 * fscale) / static_cast<F>(fscale),
                           rng.uniform(0, 255 * fscale) / static_cast<F>(fscale));
    }

    template<> inline void initPointRandU(cv::RNG &rng, cv::Point2f &pt)
    { initFloatPointRandU(rng, pt); }

    template<> inline void initPointRandU(cv::RNG &rng, cv::Point2d &pt)
    { initFloatPointRandU(rng, pt); }

    template <typename F> inline void initFloatPointRandU(cv::RNG &rng, cv::Point3_<F> &pt)
    {
        GAPI_Assert(std::is_floating_point<F>::value);
        static const int fscale = 256;  // avoid bits near ULP, generate stable test input
        pt = cv::Point3_<F>(rng.uniform(0, 255 * fscale) / static_cast<F>(fscale),
                            rng.uniform(0, 255 * fscale) / static_cast<F>(fscale),
                            rng.uniform(0, 255 * fscale) / static_cast<F>(fscale));
    }

    template<> inline void initPointRandU(cv::RNG &rng, cv::Point3f &pt)
    { initFloatPointRandU(rng, pt); }

    template<> inline void initPointRandU(cv::RNG &rng, cv::Point3d &pt)
    { initFloatPointRandU(rng, pt); }
} // namespace

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

    // integral Scalar initialization
    cv::Scalar initScalarRandU(unsigned upper)
    {
        cv::RNG rng(time(nullptr));
        double s1 = rng(upper);
        double s2 = rng(upper);
        double s3 = rng(upper);
        double s4 = rng(upper);
        return cv::Scalar(s1, s2, s3, s4);
    }

    // floating-point Scalar initialization (cv::core)
    cv::Scalar initScalarRandU()
    {
        cv::RNG rng(time(nullptr));
        double s1 = exp(rng.uniform(-1, 6) * 3.0 * CV_LOG2) * (rng.uniform(0, 2) ? 1. : -1.);
        double s2 = exp(rng.uniform(-1, 6) * 3.0 * CV_LOG2) * (rng.uniform(0, 2) ? 1. : -1.);
        double s3 = exp(rng.uniform(-1, 6) * 3.0 * CV_LOG2) * (rng.uniform(0, 2) ? 1. : -1.);
        double s4 = exp(rng.uniform(-1, 6) * 3.0 * CV_LOG2) * (rng.uniform(0, 2) ? 1. : -1.);
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

        int sdepth = CV_MAT_DEPTH(type);
        int ddepth = (dtype >= 0) ? CV_MAT_DEPTH(dtype)
                                  : sdepth;             // dtype == -1 <=> dtype == SAME_TYPE

        if ((sdepth >= CV_32F) || (ddepth >= CV_32F))
        {
            sc = initScalarRandU(); // initializing by floating-points
        }
        else
        {
            switch (sdepth)
            {
            case CV_8U:
                sc = initScalarRandU(UCHAR_MAX + 1U);
                break;
            case CV_16U:
                sc = initScalarRandU(USHRT_MAX + 1U);
                break;
            case CV_16S:
                sc = initScalarRandU(SHRT_MAX + 1U);
                break;
            default:
                sc = initScalarRandU(SCHAR_MAX + 1U);
                break;
            }
        }

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

        int sdepth = CV_MAT_DEPTH(type);
        int ddepth = (dtype >= 0) ? CV_MAT_DEPTH(dtype)
                                  : sdepth;             // dtype == -1 <=> dtype == SAME_TYPE

        if ((sdepth >= CV_32F) || (ddepth >= CV_32F))
        {
            sc = initScalarRandU();
        }
        else
        {
            switch (sdepth)
            {
            case CV_8U:
                sc = initScalarRandU(UCHAR_MAX + 1U);
                break;
            case CV_16U:
                sc = initScalarRandU(USHRT_MAX + 1U);
                break;
            case CV_16S:
                sc = initScalarRandU(SHRT_MAX + 1U);
                break;
            default:
                sc = initScalarRandU(SCHAR_MAX + 1U);
                break;
            }
        }

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

    void initMatFromImage(int type, const std::string& fileName)
    {

        int channels = (type >> CV_CN_SHIFT) + 1;
        GAPI_Assert(channels == 1 || channels == 3 || channels == 4);
        const int readFlags = (channels == 1) ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;
        cv::Mat mat = cv::imread(findDataFile(fileName), readFlags);
        if (channels == 4)
        {
            cv::cvtColor(mat, in_mat1, cv::COLOR_BGR2BGRA);
        }
        else
        {
            in_mat1 = mat;
        }

        int depth = CV_MAT_DEPTH(type);
        if (in_mat1.depth() != depth)
        {
            in_mat1.convertTo(in_mat1, depth);
        }
    }

    void initMatsFromImages(int channels, const std::string& pattern, int imgNum)
    {
        GAPI_Assert(channels == 1 || channels == 3 || channels == 4);
        const int flags = (channels == 1) ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR;

        cv::Mat m1 = cv::imread(findDataFile(cv::format(pattern.c_str(), imgNum)), flags);
        cv::Mat m2 = cv::imread(findDataFile(cv::format(pattern.c_str(), imgNum + 1)), flags);
        if (channels == 4)
        {
            cvtColor(m1, in_mat1, cv::COLOR_BGR2BGRA);
            cvtColor(m2, in_mat2, cv::COLOR_BGR2BGRA);
        }
        else
        {
            std::tie(in_mat1, in_mat2) = std::make_tuple(m1, m2);
        }
    }

    template <typename T>
    inline void initPointRandU(cv::RNG& rng, T& pt) const
    { ::initPointRandU(rng, pt); }

// Disable unreachable code warning for MSVS 2015
#if defined _MSC_VER && _MSC_VER < 1910 /*MSVS 2017*/
#pragma warning(push)
#pragma warning(disable: 4702)
#endif
    // initialize std::vector<cv::Point_<T>>/std::vector<cv::Point3_<T>>
    template <typename T, template <typename> class Pt>
    void initPointsVectorRandU(const int sz_in, std::vector<Pt<T>> &vec_) const
    {
        cv::RNG& rng = theRNG();

        vec_.clear();
        vec_.reserve(sz_in);

        for (int i = 0; i < sz_in; i++)
        {
            Pt<T> pt;
            initPointRandU(rng, pt);
            vec_.emplace_back(pt);
        }
    }
#if defined _MSC_VER && _MSC_VER < 1910 /*MSVS 2017*/
#pragma warning(pop)
#endif

    template<typename Pt>
    inline void initMatByPointsVectorRandU(const cv::Size &sz_in)
    {
            std::vector<Pt> in_vector;
            initPointsVectorRandU(sz_in.width, in_vector);
            in_mat1 = cv::Mat(in_vector, true);
    }

    // initialize Mat by a vector of Points
    template<template <typename> class Pt>
    inline void initMatByPointsVectorRandU(int type, cv::Size sz_in, int)
    {
        int depth = CV_MAT_DEPTH(type);
        switch (depth)
        {
        case CV_8U:
            initMatByPointsVectorRandU<Pt<uchar>>(sz_in);
            break;
        case CV_8S:
            initMatByPointsVectorRandU<Pt<char>>(sz_in);
            break;
        case CV_16U:
            initMatByPointsVectorRandU<Pt<ushort>>(sz_in);
            break;
        case CV_16S:
            initMatByPointsVectorRandU<Pt<short>>(sz_in);
            break;
        case CV_32S:
            initMatByPointsVectorRandU<Pt<int>>(sz_in);
            break;
        case CV_32F:
            initMatByPointsVectorRandU<Pt<float>>(sz_in);
            break;
        case CV_64F:
            initMatByPointsVectorRandU<Pt<double>>(sz_in);
            break;
        case CV_16F:
            initMatByPointsVectorRandU<Pt<cv::float16_t>>(sz_in);
            break;
        default:
            GAPI_Assert(false && "Unsupported depth");
            break;
        }
    }

    // empty function intended to show that nothing is to be initialized via TestFunctional methods
    void initNothing(int, cv::Size, int, bool = true) {}
};

template<class T>
class TestPerfParams: public TestFunctional, public perf::TestBaseWithParam<T>{};

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
template<typename CommonParams, typename SpecificParams>
struct ParamsBase;

template<typename... CommonParams, typename... SpecificParams>
struct ParamsBase<std::tuple<CommonParams...>, std::tuple<SpecificParams...>>
{
    using common_params_t   = std::tuple<CommonParams...>;
    using specific_params_t = std::tuple<SpecificParams...>;
    using params_t          = std::tuple<CommonParams..., SpecificParams...>;
    static constexpr const size_t common_params_size   = std::tuple_size<common_params_t>::value;
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

template<typename... SpecificParams>
struct Params : public ParamsBase<std::tuple<MatType2,cv::Size,MatType2,cv::GCompileArgs(*)()>,
                                  std::tuple<SpecificParams...>>
{
    static constexpr const size_t compile_args_num = 3;
};

template<typename ...SpecificParams>
struct ParamsSpecific : public ParamsBase<std::tuple<cv::GCompileArgs(*)()>,
                                          std::tuple<SpecificParams...>>
{
    static constexpr const size_t compile_args_num = 0;
};

// Base class for test fixtures
template<typename AllParams>
struct TestWithParamsBase : TestFunctional, TestWithParam<typename AllParams::params_t>
{
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
        return getCommonParam<AllParams::compile_args_num>()();
    }
};

template<typename... SpecificParams>
struct TestWithParams : public TestWithParamsBase<Params<SpecificParams...>>
{
    using AllParams = Params<SpecificParams...>;

    MatType2 type  = this->template getCommonParam<0>();
    cv::Size sz    = this->template getCommonParam<1>();
    MatType2 dtype = this->template getCommonParam<2>();
};

template<typename... SpecificParams>
struct TestWithParamsSpecific : public TestWithParamsBase<ParamsSpecific<SpecificParams...>>
{
    using AllParams = ParamsSpecific<SpecificParams...>;
};


/**
 * @private
 * @brief Create G-API test fixture with TestWithParams base class
 * @param Fixture   test fixture name
 * @param InitF     callable that will initialize default available members (from TestFunctional)
 * @param API       base class API. Specifies types of user-defined parameters. If there are no such
 *                  parameters, empty angle brackets ("<>") must be specified.
 * @param Number    number of user-defined parameters (corresponds to the number of types in API).
 *                  if there are no such parameters, 0 must be specified.
 * @param ...       list of names of user-defined parameters. if there are no parameters, the list
 *                  must be empty.
 */
 //TODO: Consider to remove `Number` and use `std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value`
#define GAPI_TEST_FIXTURE(Fixture, InitF, API, Number, ...) \
    struct Fixture : public TestWithParams API { \
        static_assert(Number == AllParams::specific_params_size, \
            "Number of user-defined parameters doesn't match size of __VA_ARGS__"); \
        __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__)) \
        Fixture() { InitF(type, sz, dtype); } \
    };

/**
 * @private
 * @brief Create G-API test fixture with TestWithParams base class and additional base class.
 * @param Fixture   test fixture name.
   @param ExtBase   additional base class.
 * @param InitF     callable that will initialize default available members (from TestFunctional)
 * @param API       base class API. Specifies types of user-defined parameters. If there are no such
 *                  parameters, empty angle brackets ("<>") must be specified.
 * @param Number    number of user-defined parameters (corresponds to the number of types in API).
 *                  if there are no such parameters, 0 must be specified.
 * @param ...       list of names of user-defined parameters. if there are no parameters, the list
 *                  must be empty.
 */
#define GAPI_TEST_EXT_BASE_FIXTURE(Fixture, ExtBase, InitF, API, Number, ...) \
    struct Fixture : public TestWithParams API, public ExtBase { \
        static_assert(Number == AllParams::specific_params_size, \
            "Number of user-defined parameters doesn't match size of __VA_ARGS__"); \
        __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__)) \
        Fixture() { InitF(type, sz, dtype); } \
    };

/**
 * @private
 * @brief Create G-API test fixture with TestWithParamsSpecific base class
 *        This fixture has reduced number of common parameters and no initialization;
 *        it should be used if you don't need common parameters of GAPI_TEST_FIXTURE.
 * @param Fixture   test fixture name
 * @param API       base class API. Specifies types of user-defined parameters. If there are no such
 *                  parameters, empty angle brackets ("<>") must be specified.
 * @param Number    number of user-defined parameters (corresponds to the number of types in API).
 *                  if there are no such parameters, 0 must be specified.
 * @param ...       list of names of user-defined parameters. if there are no parameters, the list
 *                  must be empty.
 */
#define GAPI_TEST_FIXTURE_SPEC_PARAMS(Fixture, API, Number, ...) \
    struct Fixture : public TestWithParamsSpecific API { \
        static_assert(Number == AllParams::specific_params_size, \
            "Number of user-defined parameters doesn't match size of __VA_ARGS__"); \
        __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__)) \
    };

// Wrapper for test fixture API. Use to specify multiple types.
// Example: FIXTURE_API(int, bool) expands to <int, bool>
#define FIXTURE_API(...) <__VA_ARGS__>


using compare_f = std::function<bool(const cv::Mat &a, const cv::Mat &b)>;
using compare_scalar_f = std::function<bool(const cv::Scalar &a, const cv::Scalar &b)>;
using compare_rect_f = std::function<bool(const cv::Rect &a, const cv::Rect &b)>;

template<typename Elem>
using compare_vector_f = std::function<bool(const std::vector<Elem> &a,
                                            const std::vector<Elem> &b)>;

template<typename Elem, int cn>
using compare_vec_f = std::function<bool(const cv::Vec<Elem, cn> &a, const cv::Vec<Elem, cn> &b)>;

template<typename T1, typename T2>
struct CompareF
{
    CompareF() = default;

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
using CompareRects = CompareF<cv::Rect, cv::Rect>;

template<typename Elem>
using CompareVectors = CompareF<std::vector<Elem>, std::vector<Elem>>;

template<typename Elem, int cn>
using CompareVecs = CompareF<cv::Vec<Elem, cn>, cv::Vec<Elem, cn>>;

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

template<typename T>
struct WrappableRect
{
    compare_rect_f to_compare_f()
    {
        T t = *static_cast<T*const>(this);
        return [t](const cv::Rect &a, const cv::Rect &b)
        {
            return t(a, b);
        };
    }

    CompareRects to_compare_obj()
    {
        T t = *static_cast<T*const>(this);
        std::stringstream ss;
        ss << t;
        return CompareRects(to_compare_f(), ss.str());
    }
};

template<typename T, typename Elem>
struct WrappableVector
{
    compare_vector_f<Elem> to_compare_f()
    {
        T t = *static_cast<T* const>(this);
        return [t](const std::vector<Elem>& a,
                   const std::vector<Elem>& b)
        {
            return t(a, b);
        };
    }

    CompareVectors<Elem> to_compare_obj()
    {
        T t = *static_cast<T* const>(this);
        std::stringstream ss;
        ss << t;
        return CompareVectors<Elem>(to_compare_f(), ss.str());
    }
};

template<typename T, typename Elem, int cn>
struct WrappableVec
{
    compare_vec_f<Elem, cn> to_compare_f()
    {
        T t = *static_cast<T* const>(this);
        return [t](const cv::Vec<Elem, cn> &a, const cv::Vec<Elem, cn> &b)
        {
            return t(a, b);
        };
    }

    CompareVecs<Elem, cn> to_compare_obj()
    {
        T t = *static_cast<T* const>(this);
        std::stringstream ss;
        ss << t;
        return CompareVecs<Elem, cn>(to_compare_f(), ss.str());
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
                std::cout << "ToleranceColor error: err_Inf=" << err_Inf
                          << "  tolerance=" << _inf_tol << std::endl;
                return false;
            }
            double err = cv::norm(in1, in2, NORM_L1 | NORM_RELATIVE);
            if (err > _tol)
            {
                std::cout << "ToleranceColor error: err=" << err
                          << "  tolerance=" << _tol << std::endl;
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
            std::cout << "AbsToleranceScalar error: abs_err=" << abs_err << "  tolerance=" << _tol
                      << " in1[0]" << in1[0] << " in2[0]" << in2[0] << std::endl;
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

class IoUToleranceRect : public WrappableRect<IoUToleranceRect>
{
public:
    IoUToleranceRect(double tol) : _tol(tol) {}
    bool operator() (const cv::Rect& in1, const cv::Rect& in2) const
    {
        // determine the (x, y)-coordinates of the intersection rectangle
        int xA = max(in1.x, in2.x);
        int yA = max(in1.y, in2.y);
        int xB = min(in1.br().x, in2.br().x);
        int yB = min(in1.br().y, in2.br().y);
        // compute the area of intersection rectangle
        int interArea = max(0, xB - xA) * max(0, yB - yA);
        // compute the area of union rectangle
        int unionArea = in1.area() + in2.area() - interArea;

        double iou = interArea / unionArea;
        double err = 1 - iou;
        if (err > _tol)
        {
            std::cout << "IoUToleranceRect error: err=" << err << "  tolerance=" << _tol
                      << " in1.x="      << in1.x      << " in2.x="      << in2.x
                      << " in1.y="      << in1.y      << " in2.y="      << in2.y
                      << " in1.width="  << in1.width  << " in2.width="  << in2.width
                      << " in1.height=" << in1.height << " in2.height=" << in2.height << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const IoUToleranceRect& obj)
    {
        return os << "IoUToleranceRect(" << std::to_string(obj._tol) << ")";
    }
private:
    double _tol;
};

template<typename Elem>
class AbsExactVector : public WrappableVector<AbsExactVector<Elem>, Elem>
{
public:
    AbsExactVector() {}
    bool operator() (const std::vector<Elem>& in1,
                     const std::vector<Elem>& in2) const
    {
        if (cv::norm(in1, in2, NORM_INF, cv::noArray()) != 0)
        {
            std::cout << "AbsExact error: G-API output and reference output vectors are not"
                         " bitexact equal." << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const AbsExactVector<Elem>&)
    {
        return os << "AbsExactVector()";
    }
};

template<typename Elem, int cn>
class RelDiffToleranceVec : public WrappableVec<RelDiffToleranceVec<Elem, cn>, Elem, cn>
{
public:
    RelDiffToleranceVec(double tol) : _tol(tol) {}
    bool operator() (const cv::Vec<Elem, cn> &in1, const cv::Vec<Elem, cn> &in2) const
    {
        double abs_err  = cv::norm(in1, in2, cv::NORM_L1);
        double in2_norm = cv::norm(in2, cv::NORM_L1);
        // Checks to avoid dividing by zero
        double err = abs_err ? abs_err / (in2_norm ? in2_norm : cv::norm(in1, cv::NORM_L1))
                             : abs_err;
        if (err > _tol)
        {
            std::cout << "RelDiffToleranceVec error: err=" << err << "  tolerance=" << _tol;
            for (int i = 0; i < cn; i++)
            {
                std::cout << " in1[" << i << "]=" << in1[i] << " in2[" << i << "]=" << in2[i];
            }
            std::cout << std::endl;
            return false;
        }
        else
        {
            return true;
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const RelDiffToleranceVec<Elem, cn>& obj)
    {
        return os << "RelDiffToleranceVec(" << std::to_string(obj._tol) << ")";
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

inline std::ostream& operator<<(std::ostream& os, const opencv_test::compare_rect_f&)
{
    return os << "compare_rect_f";
}

template<typename Elem>
inline std::ostream& operator<<(std::ostream& os, const opencv_test::compare_vector_f<Elem>&)
{
    return os << "compare_vector_f";
}

template<typename Elem, int cn>
inline std::ostream& operator<<(std::ostream& os, const opencv_test::compare_vec_f<Elem, cn>&)
{
    return os << "compare_vec_f";
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

inline std::ostream& operator<<(std::ostream& os, RetrievalModes op)
{
#define CASE(v) case RetrievalModes::v: os << #v; break
    switch (op)
    {
        CASE(RETR_EXTERNAL);
        CASE(RETR_LIST);
        CASE(RETR_CCOMP);
        CASE(RETR_TREE);
        CASE(RETR_FLOODFILL);
        default: GAPI_Assert(false && "unknown RetrievalModes value");
    }
#undef CASE
    return os;
}

inline std::ostream& operator<<(std::ostream& os, ContourApproximationModes op)
{
#define CASE(v) case ContourApproximationModes::v: os << #v; break
    switch (op)
    {
        CASE(CHAIN_APPROX_NONE);
        CASE(CHAIN_APPROX_SIMPLE);
        CASE(CHAIN_APPROX_TC89_L1);
        CASE(CHAIN_APPROX_TC89_KCOS);
        default: GAPI_Assert(false && "unknown ContourApproximationModes value");
    }
#undef CASE
    return os;
}

inline std::ostream& operator<<(std::ostream& os, MorphTypes op)
{
#define CASE(v) case MorphTypes::v: os << #v; break
    switch (op)
    {
        CASE(MORPH_ERODE);
        CASE(MORPH_DILATE);
        CASE(MORPH_OPEN);
        CASE(MORPH_CLOSE);
        CASE(MORPH_GRADIENT);
        CASE(MORPH_TOPHAT);
        CASE(MORPH_BLACKHAT);
        CASE(MORPH_HITMISS);
        default: GAPI_Assert(false && "unknown MorphTypes value");
    }
#undef CASE
    return os;
}

inline std::ostream& operator<<(std::ostream& os, DistanceTypes op)
{
#define CASE(v) case DistanceTypes::v: os << #v; break
    switch (op)
    {
        CASE(DIST_USER);
        CASE(DIST_L1);
        CASE(DIST_L2);
        CASE(DIST_C);
        CASE(DIST_L12);
        CASE(DIST_FAIR);
        CASE(DIST_WELSCH);
        CASE(DIST_HUBER);
        default: GAPI_Assert(false && "unknown DistanceTypes value");
    }
#undef CASE
    return os;
}

inline std::ostream& operator<<(std::ostream& os, KmeansFlags op)
{
    int op_(op);
    switch (op_)
    {
    case KmeansFlags::KMEANS_RANDOM_CENTERS:
        os << "KMEANS_RANDOM_CENTERS";
        break;
    case KmeansFlags::KMEANS_PP_CENTERS:
        os << "KMEANS_PP_CENTERS";
        break;
    case KmeansFlags::KMEANS_RANDOM_CENTERS | KmeansFlags::KMEANS_USE_INITIAL_LABELS:
        os << "KMEANS_RANDOM_CENTERS | KMEANS_USE_INITIAL_LABELS";
        break;
    case KmeansFlags::KMEANS_PP_CENTERS | KmeansFlags::KMEANS_USE_INITIAL_LABELS:
        os << "KMEANS_PP_CENTERS | KMEANS_USE_INITIAL_LABELS";
        break;
    default: GAPI_Assert(false && "unknown KmeansFlags value");
    }
    return os;
}
}  // namespace cv

#endif //OPENCV_GAPI_TESTS_COMMON_HPP
