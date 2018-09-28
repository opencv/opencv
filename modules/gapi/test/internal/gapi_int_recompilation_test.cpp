// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "api/gcomputation_priv.hpp"

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

} // opencv_test
