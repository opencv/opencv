// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "api/gcomputation_priv.hpp"

#include "opencv2/gapi/fluid/gfluidkernel.hpp"
#include "opencv2/gapi/fluid/core.hpp"
#include "opencv2/gapi/fluid/imgproc.hpp"

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

TEST(GComputationCompile, FluidReshapeWithDifferentDims)
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

TEST(GComputationCompile, FluidReshapeResizeDownScale)
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

TEST(GComputationCompile, FluidReshapeSwitchToUpscaleFromDownscale)
{
    cv::Size szOut(4, 4);
    cv::GMat in;
    cv::GComputation cc(in, cv::gapi::resize(in, szOut));

    cv::Mat in_mat1( 8,  8, CV_8UC3);
    cv::Mat in_mat2( 2,  2, CV_8UC3);
    cv::Mat in_mat3(16, 16, CV_8UC3);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat3, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::Mat out_mat1, out_mat2, out_mat3;

    cc.apply(in_mat1, out_mat1, cv::compile_args(cv::gapi::core::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat2);
    auto comp2 = cc.priv().m_lastCompiled;

    cc.apply(in_mat3, out_mat3);
    auto comp3 = cc.priv().m_lastCompiled;

    EXPECT_EQ(&comp1.priv(), &comp2.priv());
    EXPECT_EQ(&comp1.priv(), &comp3.priv());

    cv::Mat cv_out_mat1, cv_out_mat2, cv_out_mat3;
    cv::resize(in_mat1, cv_out_mat1, szOut);
    cv::resize(in_mat2, cv_out_mat2, szOut);
    cv::resize(in_mat3, cv_out_mat3, szOut);

    EXPECT_EQ(0, cv::countNonZero(out_mat1 != cv_out_mat1));
    EXPECT_EQ(0, cv::countNonZero(out_mat2 != cv_out_mat2));
    EXPECT_EQ(0, cv::countNonZero(out_mat3 != cv_out_mat3));
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

TEST(GComputationCompile, ReshapeRois)
{
    cv::Size kernelSize{3, 3};
    cv::Size szOut(8, 8);
    cv::GMat in;
    auto blurred = cv::gapi::blur(in, kernelSize);
    cv::GComputation cc(in, cv::gapi::resize(blurred, szOut));

    cv::Mat first_in_mat(8, 8, CV_8UC3);
    cv::randn(first_in_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));
    cv::Mat first_out_mat;
    auto fluidKernels = cv::gapi::combine(gapi::imgproc::fluid::kernels(),
                                          gapi::core::fluid::kernels(),
                                          cv::unite_policy::REPLACE);
    cc.apply(first_in_mat, first_out_mat, cv::compile_args(fluidKernels));
    auto first_comp = cc.priv().m_lastCompiled;

    constexpr int niter = 4;
    for (int i = 0; i < niter; i++)
    {
        int width  = 4 + 2*i;
        int height = width;
        cv::Mat in_mat(width, height, CV_8UC3);
        cv::randn(in_mat, cv::Scalar::all(127), cv::Scalar::all(40.f));
        cv::Mat out_mat = cv::Mat::zeros(szOut, CV_8UC3);

        int x = 0;
        int y = szOut.height * i / niter;
        int roiW = szOut.width;
        int roiH = szOut.height / niter;
        cv::Rect roi{x, y, roiW, roiH};

        cc.apply(in_mat, out_mat, cv::compile_args(cv::GFluidOutputRois{{to_own(roi)}}));
        auto comp = cc.priv().m_lastCompiled;

        EXPECT_EQ(&first_comp.priv(), &comp.priv());

        cv::Mat blur_mat, cv_out_mat;
        cv::blur(in_mat, blur_mat, kernelSize);
        cv::resize(blur_mat, cv_out_mat, szOut);

        EXPECT_EQ(0, cv::countNonZero(out_mat(roi) != cv_out_mat(roi)));
    }
}

} // opencv_test
