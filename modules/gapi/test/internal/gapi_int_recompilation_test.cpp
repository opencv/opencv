// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "api/gcomputation_priv.hpp"

#include <backends/fluid/gfluidcore.hpp>
#include <backends/fluid/gfluidimgproc.hpp>

namespace opencv_test
{

TEST(GComputationCompile, NoRecompileWithSameMeta)
{
    cv::GMat in;
    cv::GComputation cc(in, in+in);

    cv::Mat in_mat1 = cv::Mat::eye  (32, 32, CV_8UC1);
    cv::Mat in_mat2 = cv::Mat::zeros(32, 32, CV_8UC1);
    cv::Mat out_mat;

    cc.apply(in_mat1, out_mat);
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, NoRecompileWithWrongMeta)
{
    cv::GMat in;
    cv::GComputation cc(in, in+in);

    cv::Mat in_mat1 = cv::Mat::eye  (32, 32, CV_8UC1);
    cv::Mat in_mat2 = cv::Mat::zeros(32, 32, CV_8UC1);
    cv::Mat out_mat;

    cc.apply(in_mat1, out_mat);
    auto comp1 = cc.priv().m_lastCompiled;

    EXPECT_THROW(cc.apply(cv::gin(cv::Scalar(128)), cv::gout(out_mat)), std::logic_error);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, RecompileWithDifferentMeta)
{
    cv::GMat in;
    cv::GComputation cc(in, in+in);

    cv::Mat in_mat1 = cv::Mat::eye  (32, 32, CV_8UC1);
    cv::Mat in_mat2 = cv::Mat::zeros(64, 64, CV_32F);
    cv::Mat out_mat;

    cc.apply(in_mat1, out_mat);
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are different
    EXPECT_NE(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, ReshapeWithDifferentDims)
{
    cv::GMat in;
    cv::GComputation cc(in, in+in);

    cv::Mat in_mat1 = cv::Mat::eye  (32, 32, CV_8UC1);
    cv::Mat in_mat2 = cv::Mat::zeros(64, 64, CV_8UC1);
    cv::Mat out_mat;

    cc.apply(in_mat1, out_mat, cv::compile_args(cv::gapi::core::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, ReshapeResizeDownScale)
{
    cv::Size szOut(4, 4);
    cv::GMat in;
    cv::GComputation cc(in, cv::gapi::resize(in, szOut));

    cv::Mat in_mat1( 8,  8, CV_8UC3);
    cv::Mat in_mat2(16, 16, CV_8UC3);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat out_mat1, out_mat2;

    cc.apply(in_mat1, out_mat1, cv::compile_args(cv::gapi::core::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat2);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());

    cv::Mat cv_out_mat1, cv_out_mat2;
    cv::resize(in_mat1, cv_out_mat1, szOut);
    cv::resize(in_mat2, cv_out_mat2, szOut);

    EXPECT_EQ(0, cv::countNonZero(out_mat1 != cv_out_mat1));
    EXPECT_EQ(0, cv::countNonZero(out_mat2 != cv_out_mat2));
}

TEST(GComputationCompile, ReshapeSwitchToUpscaleFromDownscale)
{
    cv::Size szOut(4, 4);
    cv::GMat in;
    cv::GComputation cc(in, cv::gapi::resize(in, szOut));

    cv::Mat in_mat1(8, 8, CV_8UC3);
    cv::Mat in_mat2(2, 2, CV_8UC3);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat out_mat1, out_mat2;

    cc.apply(in_mat1, out_mat1, cv::compile_args(cv::gapi::core::fluid::kernels()));

    // Currently such switches are unsupported
    EXPECT_ANY_THROW(cc.apply(in_mat2, out_mat2));
}

TEST(GComputationCompile, ReshapeBlur)
{
    cv::Size kernelSize{3, 3};
    cv::GMat in;
    cv::GComputation cc(in, cv::gapi::blur(in, kernelSize));

    cv::Mat in_mat1( 8,  8, CV_8UC1);
    cv::Mat in_mat2(16, 16, CV_8UC1);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat out_mat1, out_mat2;

    cc.apply(in_mat1, out_mat1, cv::compile_args(cv::gapi::imgproc::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat2);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());

    cv::Mat cv_out_mat1, cv_out_mat2;
    cv::blur(in_mat1, cv_out_mat1, kernelSize);
    cv::blur(in_mat2, cv_out_mat2, kernelSize);

    EXPECT_EQ(0, cv::countNonZero(out_mat1 != cv_out_mat1));
    EXPECT_EQ(0, cv::countNonZero(out_mat2 != cv_out_mat2));
}

} // opencv_test
