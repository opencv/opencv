#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wmissing-declarations"
#  if defined __clang__ || defined __APPLE__
#    pragma GCC diagnostic ignored "-Wmissing-prototypes"
#    pragma GCC diagnostic ignored "-Wextra"
#  endif
#endif

#ifndef __OPENCV_TEST_PRECOMP_HPP__
#define __OPENCV_TEST_PRECOMP_HPP__

#include <iostream>
#include "opencv2/ts.hpp"
#include "opencv2/core/core_c.h"

#include "opencv2/core/private.hpp"

#define MWIDTH 256
#define MHEIGHT 256

#define MIN_VALUE 171
#define MAX_VALUE 357

#define RNG_SEED 123456

template <typename T>
struct TSTestWithParam : public ::testing::TestWithParam<T>
{
    cv::RNG rng;

    TSTestWithParam()
    {
        rng = cv::RNG(RNG_SEED);
    }

    int randomInt(int minVal, int maxVal)
    {
        return rng.uniform(minVal, maxVal);
    }

    double randomDouble(double minVal, double maxVal)
    {
        return rng.uniform(minVal, maxVal);
    }

    double randomDoubleLog(double minVal, double maxVal)
    {
        double logMin = log((double)minVal + 1);
        double logMax = log((double)maxVal + 1);
        double pow = rng.uniform(logMin, logMax);
        double v = exp(pow) - 1;
        CV_Assert(v >= minVal && (v < maxVal || (v == minVal && v == maxVal)));
        return v;
    }

    cv::Size randomSize(int minVal, int maxVal)
    {
#if 1
        return cv::Size((int)randomDoubleLog(minVal, maxVal), (int)randomDoubleLog(minVal, maxVal));
#else
        return cv::Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
#endif
    }

    cv::Size randomSize(int minValX, int maxValX, int minValY, int maxValY)
    {
#if 1
        return cv::Size(randomDoubleLog(minValX, maxValX), randomDoubleLog(minValY, maxValY));
#else
        return cv::Size(randomInt(minVal, maxVal), randomInt(minVal, maxVal));
#endif
    }

    cv::Scalar randomScalar(double minVal, double maxVal)
    {
        return cv::Scalar(randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal), randomDouble(minVal, maxVal));
    }

    cv::Mat randomMat(cv::Size size, int type, double minVal, double maxVal, bool useRoi = false)
    {
        cv::RNG dataRng(rng.next());
        return cvtest::randomMat(dataRng, size, type, minVal, maxVal, useRoi);
    }

};

#define PARAM_TEST_CASE(name, ...) struct name : public TSTestWithParam< std::tr1::tuple< __VA_ARGS__ > >

#define GET_PARAM(k) std::tr1::get< k >(GetParam())

#define UMAT_TEST_CHANNELS testing::Values(1, 2, 3, 4)

#define UMAT_TEST_SIZES testing::Values(cv::Size(1,1), cv::Size(1,128), cv::Size(128,1), cv::Size(128, 128), cv::Size(640,480), cv::Size(751,373), cv::Size(1200, 1200))

#define UMAT_TEST_DEPTH testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F)

# define CORE_TEST_P(test_case_name, test_name) \
    class GTEST_TEST_CLASS_NAME_(test_case_name, test_name) : \
        public test_case_name { \
    public: \
        GTEST_TEST_CLASS_NAME_(test_case_name, test_name)() { } \
        virtual void TestBody(); \
        void CoreTestBody(); \
    private: \
        static int AddToRegistry() \
        { \
            ::testing::UnitTest::GetInstance()->parameterized_test_registry(). \
              GetTestCasePatternHolder<test_case_name>(\
                  #test_case_name, __FILE__, __LINE__)->AddTestPattern(\
                      #test_case_name, \
                      #test_name, \
                      new ::testing::internal::TestMetaFactory< \
                          GTEST_TEST_CLASS_NAME_(test_case_name, test_name)>()); \
            return 0; \
        } \
    \
        static int gtest_registering_dummy_; \
        GTEST_DISALLOW_COPY_AND_ASSIGN_(\
            GTEST_TEST_CLASS_NAME_(test_case_name, test_name)); \
    }; \
    \
    int GTEST_TEST_CLASS_NAME_(test_case_name, \
                             test_name)::gtest_registering_dummy_ = \
      GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::AddToRegistry(); \
    \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::TestBody() \
    { \
        try \
        { \
            CoreTestBody(); \
        } \
        catch (...) \
        { \
                std::cout << "Something wrong in CoreTestBody running" << std::endl; \
                throw; \
        } \
    } \
    \
    void GTEST_TEST_CLASS_NAME_(test_case_name, test_name)::CoreTestBody()

#endif
