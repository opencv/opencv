/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "test_precomp.hpp"

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)
#include <thread>
#endif

namespace opencv_test { namespace {

typedef tuple<Size> OFParams;
typedef TestWithParam<OFParams> DenseOpticalFlow_DIS;
typedef TestWithParam<OFParams> DenseOpticalFlow_VariationalRefinement;

TEST_P(DenseOpticalFlow_DIS, MultithreadReproducibility)
{
    double MAX_DIF = 0.01;
    double MAX_MEAN_DIF = 0.001;
    int loopsCount = 2;
    RNG rng(0);

    OFParams params = GetParam();
    Size size = get<0>(params);

    int nThreads = cv::getNumThreads();
    if (nThreads == 1)
        throw SkipTestException("Single thread environment");
    for (int iter = 0; iter <= loopsCount; iter++)
    {
        Mat frame1(size, CV_8U);
        randu(frame1, 0, 255);
        Mat frame2(size, CV_8U);
        randu(frame2, 0, 255);

        Ptr<DISOpticalFlow> algo = DISOpticalFlow::create();
        int psz = rng.uniform(4, 16);
        int pstr = rng.uniform(1, psz - 1);
        int grad_iter = rng.uniform(1, 64);
        int var_iter = rng.uniform(0, 10);
        bool use_mean_normalization = !!rng.uniform(0, 2);
        bool use_spatial_propagation = !!rng.uniform(0, 2);
        algo->setFinestScale(0);
        algo->setPatchSize(psz);
        algo->setPatchStride(pstr);
        algo->setGradientDescentIterations(grad_iter);
        algo->setVariationalRefinementIterations(var_iter);
        algo->setUseMeanNormalization(use_mean_normalization);
        algo->setUseSpatialPropagation(use_spatial_propagation);

        cv::setNumThreads(nThreads);
        Mat resMultiThread;
        algo->calc(frame1, frame2, resMultiThread);

        cv::setNumThreads(1);
        Mat resSingleThread;
        algo->calc(frame1, frame2, resSingleThread);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_L1), MAX_MEAN_DIF * frame1.total());

        // resulting flow should be within the frame bounds:
        double min_val, max_val;
        minMaxLoc(resMultiThread, &min_val, &max_val);
        EXPECT_LE(abs(min_val), sqrt( static_cast<double>(size.height * size.height + size.width * size.width)) );
        EXPECT_LE(abs(max_val), sqrt( static_cast<double>(size.height * size.height + size.width * size.width)) );
    }
}

INSTANTIATE_TEST_CASE_P(FullSet, DenseOpticalFlow_DIS, Values(szODD, szQVGA));

TEST_P(DenseOpticalFlow_VariationalRefinement, MultithreadReproducibility)
{
    double MAX_DIF = 0.01;
    double MAX_MEAN_DIF = 0.001;
    float input_flow_rad = 5.0;
    int loopsCount = 2;
    RNG rng(0);

    OFParams params = GetParam();
    Size size = get<0>(params);

    int nThreads = cv::getNumThreads();
    if (nThreads == 1)
        throw SkipTestException("Single thread environment");
    for (int iter = 0; iter <= loopsCount; iter++)
    {
        Mat frame1(size, CV_8U);
        randu(frame1, 0, 255);
        Mat frame2(size, CV_8U);
        randu(frame2, 0, 255);
        Mat flow(size, CV_32FC2);
        randu(flow, -input_flow_rad, input_flow_rad);

        Ptr<VariationalRefinement> var = VariationalRefinement::create();
        var->setAlpha(rng.uniform(1.0f, 100.0f));
        var->setGamma(rng.uniform(0.1f, 10.0f));
        var->setDelta(rng.uniform(0.1f, 10.0f));
        var->setSorIterations(rng.uniform(1, 20));
        var->setFixedPointIterations(rng.uniform(1, 20));
        var->setOmega(rng.uniform(1.01f, 1.99f));

        cv::setNumThreads(nThreads);
        Mat resMultiThread;
        flow.copyTo(resMultiThread);
        var->calc(frame1, frame2, resMultiThread);

        cv::setNumThreads(1);
        Mat resSingleThread;
        flow.copyTo(resSingleThread);
        var->calc(frame1, frame2, resSingleThread);

        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_INF), MAX_DIF);
        EXPECT_LE(cv::norm(resSingleThread, resMultiThread, NORM_L1), MAX_MEAN_DIF * frame1.total());

        // resulting flow should be within the frame bounds:
        double min_val, max_val;
        minMaxLoc(resMultiThread, &min_val, &max_val);
        EXPECT_LE(abs(min_val), sqrt( static_cast<double>(size.height * size.height + size.width * size.width)) );
        EXPECT_LE(abs(max_val), sqrt( static_cast<double>(size.height * size.height + size.width * size.width)) );
    }
}

INSTANTIATE_TEST_CASE_P(FullSet, DenseOpticalFlow_VariationalRefinement, Values(szODD, szQVGA));

#if !defined(OPENCV_DISABLE_THREAD_SUPPORT)
// One instance with expansion reuse enabled, driven by several threads on independent sequences.
TEST(DenseOpticalFlow_Farneback, ReuseConcurrentCalls)
{
    const int threads = 4, pairs = 6;
    std::vector<std::vector<Mat> > frames(threads), actual(threads);
    for( int t = 0; t < threads; t++ )
    {
        actual[t].resize(pairs);
        for( int i = 0; i <= pairs; i++ )
        {
            Mat frame(48, 64, CV_8U);
            theRNG().fill(frame, RNG::UNIFORM, 0, 256);
            GaussianBlur(frame, frame, Size(9, 9), 3, 3);
            frames[t].push_back(frame);
        }
    }

    Ptr<FarnebackOpticalFlow> algo =
        FarnebackOpticalFlow::create(3, 0.5, false, 9, 3, 5, 1.1, OPTFLOW_FARNEBACK_REUSE_EXPANSION);
    std::vector<std::thread> workers;
    for( int t = 0; t < threads; t++ )
        workers.push_back(std::thread([&, t]() {
            for( int i = 0; i < pairs; i++ )
                algo->calc(frames[t][i], frames[t][i + 1], actual[t][i]);
        }));
    for( size_t t = 0; t < workers.size(); t++ )
        workers[t].join();

    Ptr<FarnebackOpticalFlow> fresh = FarnebackOpticalFlow::create(3, 0.5, false, 9, 3, 5, 1.1, 0);
    for( int t = 0; t < threads; t++ )
        for( int i = 0; i < pairs; i++ )
        {
            Mat expected;
            fresh->calc(frames[t][i], frames[t][i + 1], expected);
            ASSERT_EQ(expected.size(), actual[t][i].size()) << "thread " << t << ", pair " << i;
            EXPECT_EQ(0, cvtest::norm(expected, actual[t][i], NORM_INF))
                << "thread " << t << ", pair " << i;
        }
}
#endif

}} // namespace
