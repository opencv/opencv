// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_CORE_TESTS_COMMON_HPP
#define OPENCV_GAPI_CORE_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/core.hpp"

#include <opencv2/core.hpp>

namespace opencv_test
{
namespace
{
template <typename Elem, typename CmpF>
inline bool compareKMeansOutputs(const std::vector<Elem>& outGAPI,
                                 const std::vector<Elem>& outOCV,
                                 const CmpF& = AbsExact().to_compare_obj())
{
    return AbsExactVector<Elem>().to_compare_f()(outGAPI, outOCV);
}

inline bool compareKMeansOutputs(const cv::Mat& outGAPI,
                                 const cv::Mat& outOCV,
                                 const CompareMats& cmpF)
{
    return cmpF(outGAPI, outOCV);
}
}

// Overload with initializing the labels
template<typename Labels, typename In>
cv::GComputation kmeansTestGAPI(const In& in, const Labels& bestLabels, const int K,
                                const cv::KmeansFlags flags, cv::GCompileArgs&& args,
                                double& compact_gapi, Labels& labels_gapi, In& centers_gapi)
{
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0);
    const int attempts = 1;

    cv::detail::g_type_of_t<In> gIn, centers;
    cv::GOpaque<double> compactness;
    cv::detail::g_type_of_t<Labels> inLabels, outLabels;
    std::tie(compactness, outLabels, centers) =
        cv::gapi::kmeans(gIn, K, inLabels, criteria, attempts, flags);
    cv::GComputation c(cv::GIn(gIn, inLabels), cv::GOut(compactness, outLabels, centers));
    c.apply(cv::gin(in, bestLabels), cv::gout(compact_gapi, labels_gapi, centers_gapi),
            std::move(args));
    return c;
}

// Overload for vector<Point> tests w/o initializing the labels
template<typename Pt>
cv::GComputation kmeansTestGAPI(const std::vector<Pt>& in, const int K,
                                const cv::KmeansFlags flags, cv::GCompileArgs&& args,
                                double& compact_gapi, std::vector<int>& labels_gapi,
                                std::vector<Pt>& centers_gapi)
{
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0);
    const int attempts = 1;

    cv::GArray<Pt> gIn, centers;
    cv::GOpaque<double> compactness;
    cv::GArray<int> inLabels(std::vector<int>{}), outLabels;
    std::tie(compactness, outLabels, centers) =
        cv::gapi::kmeans(gIn, K, inLabels, criteria, attempts, flags);
    cv::GComputation c(cv::GIn(gIn), cv::GOut(compactness, outLabels, centers));
    c.apply(cv::gin(in), cv::gout(compact_gapi, labels_gapi, centers_gapi), std::move(args));
    return c;
}

// Overload for Mat tests w/o initializing the labels
static cv::GComputation kmeansTestGAPI(const cv::Mat& in, const int K,
                                       const cv::KmeansFlags flags, cv::GCompileArgs&& args,
                                       double& compact_gapi, cv::Mat& labels_gapi,
                                       cv::Mat& centers_gapi)
{
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0);
    const int attempts = 1;

    cv::GMat gIn, centers, labels;
    cv::GOpaque<double> compactness;
    std::tie(compactness, labels, centers) = cv::gapi::kmeans(gIn, K, criteria, attempts, flags);
    cv::GComputation c(cv::GIn(gIn), cv::GOut(compactness, labels, centers));
    c.apply(cv::gin(in), cv::gout(compact_gapi, labels_gapi, centers_gapi), std::move(args));
    return c;
}

template<typename Pt>
void kmeansTestValidate(const cv::Size& sz, const MatType2&, const int K,
                        const double compact_gapi, const std::vector<int>& labels_gapi,
                        const std::vector<Pt>& centers_gapi)
{
    const int amount = sz.height;
    // Validation
    EXPECT_GE(compact_gapi, 0.);
    EXPECT_EQ(labels_gapi.size(), static_cast<size_t>(amount));
    EXPECT_EQ(centers_gapi.size(), static_cast<size_t>(K));
}

static void kmeansTestValidate(const cv::Size& sz, const MatType2& type, const int K,
                               const double compact_gapi, const cv::Mat& labels_gapi,
                               const cv::Mat& centers_gapi)
{
    const int chan   = (type >> CV_CN_SHIFT) + 1;
    const int amount = sz.height != 1 ? sz.height : sz.width;
    const int dim    = sz.height != 1 ? sz.width * chan : chan;
    // Validation
    EXPECT_GE(compact_gapi, 0.);
    EXPECT_FALSE(labels_gapi.empty());
    EXPECT_FALSE(centers_gapi.empty());
    EXPECT_EQ(labels_gapi.rows, amount);
    EXPECT_EQ(labels_gapi.cols, 1);
    EXPECT_EQ(centers_gapi.rows, K);
    EXPECT_EQ(centers_gapi.cols, dim);
}

template<typename Labels, typename In>
void kmeansTestOpenCVCompare(const In& in, const Labels& bestLabels, const int K,
                             const cv::KmeansFlags flags, const double compact_gapi,
                             const Labels& labels_gapi, const In& centers_gapi,
                             const CompareMats& cmpF = AbsExact().to_compare_obj())
{
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 30, 0);
    const int attempts = 1;
    Labels labels_ocv;
    In centers_ocv;
    { // step to generalize cv::Mat & std::vector cases of bestLabels' types
        cv::Mat bestLabelsMat(bestLabels);
        bestLabelsMat.copyTo(labels_ocv);
    }
    // OpenCV code /////////////////////////////////////////////////////////////
    double compact_ocv = cv::kmeans(in, K, labels_ocv, criteria, attempts, flags, centers_ocv);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(compact_gapi == compact_ocv);
    EXPECT_TRUE(compareKMeansOutputs(labels_gapi, labels_ocv, cmpF));
    EXPECT_TRUE(compareKMeansOutputs(centers_gapi, centers_ocv, cmpF));
}

// If an input type is cv::Mat, labels' type is also cv::Mat;
// in other cases, their type has to be std::vector<int>
template<typename In>
using KMeansLabelType = typename std::conditional<std::is_same<In, cv::Mat>::value,
                                                  cv::Mat,
                                                  std::vector<int>
                                                 >::type;
template<typename In, typename Labels = KMeansLabelType<In> >
void kmeansTestBody(const In& in, const cv::Size& sz, const MatType2& type, const int K,
                    const cv::KmeansFlags flags, cv::GCompileArgs&& args,
                    const CompareMats& cmpF = AbsExact().to_compare_obj())
{
    double compact_gapi = -1.;
    Labels labels_gapi;
    In centers_gapi;
    if (flags & cv::KMEANS_USE_INITIAL_LABELS)
    {
        Labels bestLabels;
        { // step to generalize cv::Mat & std::vector cases of bestLabels' types
            const int amount = (sz.height != 1 || sz.width == -1) ? sz.height : sz.width;
            cv::Mat bestLabelsMat(cv::Size{1, amount}, CV_32SC1);
            cv::randu(bestLabelsMat, 0, K);
            bestLabelsMat.copyTo(bestLabels);
        }
        kmeansTestGAPI(in, bestLabels, K, flags, std::move(args), compact_gapi, labels_gapi,
                       centers_gapi);
        kmeansTestOpenCVCompare(in, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi, cmpF);
    }
    else
    {
        kmeansTestGAPI(in, K, flags, std::move(args), compact_gapi, labels_gapi, centers_gapi);
        kmeansTestValidate(sz, type, K, compact_gapi, labels_gapi, centers_gapi);
    }
}
} // namespace opencv_test

#endif // OPENCV_GAPI_CORE_TESTS_COMMON_HPP
