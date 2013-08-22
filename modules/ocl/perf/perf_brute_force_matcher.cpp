/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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
#include "perf_precomp.hpp"

using namespace perf;

#define OCL_BFMATCHER_TYPICAL_MAT_SIZES ::testing::Values(cv::Size(128, 500), cv::Size(128, 1000), cv::Size(128, 2000))

//////////////////// BruteForceMatch /////////////////

typedef TestBaseWithParam<Size> BruteForceMatcherFixture;

PERF_TEST_P(BruteForceMatcherFixture, match,
            OCL_BFMATCHER_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    vector<DMatch> matches;
    Mat query(srcSize, CV_32F), train(srcSize, CV_32F);
    declare.in(query, train).time(srcSize.height == 2000 ? 8 : 4 );
    randu(query, 0.0f, 1.0f);
    randu(train, 0.0f, 1.0f);

    if (impl == "plain")
    {
        BFMatcher matcher(NORM_L2);
        TEST_CYCLE() matcher.match(query, train, matches);

        SANITY_CHECK_MATCHES(matches);
    }
    else if (impl == "ocl")
    {
        // Init GPU matcher
        ocl::BruteForceMatcher_OCL_base oclMatcher(ocl::BruteForceMatcher_OCL_base::L2Dist);
        ocl::oclMat oclQuery(query), oclTrain(train);

        TEST_CYCLE() oclMatcher.match(oclQuery, oclTrain, matches);

        SANITY_CHECK_MATCHES(matches);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

//PERF_TEST_P(BruteForceMatcherFixture, matchSingle,
//            OCL_BFMATCHER_TYPICAL_MAT_SIZES)
//{
//    const Size srcSize = GetParam();
//    const string impl = getSelectedImpl();

//    Mat query(srcSize, CV_32F), train(srcSize, CV_32F);
//    Mat trainIdx, distance;

//    randu(query, 0.0f, 1.0f);
//    randu(train, 0.0f, 1.0f);

//    if (impl == "plain")
//        CV_TEST_FAIL_NO_IMPL();
//    else if (impl == "ocl")
//    {
//        ocl::oclMat oclQuery(query), oclTrain(train), oclTrainIdx, oclDistance;

//        TEST_CYCLE() oclMatcher->matchSingle(oclQuery, oclTrain, oclTrainIdx, oclDistance);

//        oclTrainIdx.download(trainIdx);
//        oclDistance.download(distance);

//        SANITY_CHECK(trainIdx);
//        SANITY_CHECK(distance);
//    }
//#ifdef HAVE_OPENCV_GPU
//    else if (impl == "gpu")
//        CV_TEST_FAIL_NO_IMPL();
//#endif
//    else
//        CV_TEST_FAIL_NO_IMPL();
//}

PERF_TEST_P(BruteForceMatcherFixture, knnMatch,
            OCL_BFMATCHER_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    vector<vector<DMatch> > matches(2);
    Mat query(srcSize, CV_32F), train(srcSize, CV_32F);
    randu(query, 0.0f, 1.0f);
    randu(train, 0.0f, 1.0f);

    declare.in(query, train);
    if (srcSize.height == 2000)
        declare.time(8);

    if (impl == "plain")
    {
        BFMatcher matcher (NORM_L2);
        TEST_CYCLE() matcher.knnMatch(query, train, matches, 2);

        std::vector<DMatch> & matches0 = matches[0], & matches1 = matches[1];
        SANITY_CHECK_MATCHES(matches0);
        SANITY_CHECK_MATCHES(matches1);
    }
    else if (impl == "ocl")
    {
        ocl::BruteForceMatcher_OCL_base oclMatcher(ocl::BruteForceMatcher_OCL_base::L2Dist);
        ocl::oclMat oclQuery(query), oclTrain(train);

        TEST_CYCLE() oclMatcher.knnMatch(oclQuery, oclTrain, matches, 2);

        std::vector<DMatch> & matches0 = matches[0], & matches1 = matches[1];
        SANITY_CHECK_MATCHES(matches0);
        SANITY_CHECK_MATCHES(matches1);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

//PERF_TEST_P(BruteForceMatcherFixture, knnMatchSingle,
//            OCL_BFMATCHER_TYPICAL_MAT_SIZES)
//{
//    const Size srcSize = GetParam();
//    const string impl = getSelectedImpl();

//    Mat query(srcSize, CV_32F), train(srcSize, CV_32F);
//    Mat trainIdx, distance, allDist;

//    randu(query, 0.0f, 1.0f);
//    randu(train, 0.0f, 1.0f);

//    if (impl == "plain")
//        CV_TEST_FAIL_NO_IMPL();
//    else if (impl == "ocl")
//    {
//        ocl::oclMat oclQuery(query), oclTrain(train), oclTrainIdx, oclDistance, oclAllDist;

//        TEST_CYCLE() oclMatcher->knnMatchSingle(oclQuery, oclTrain, oclTrainIdx, oclDistance, oclAllDist, 2);

//        oclTrainIdx.download(trainIdx);
//        oclDistance.download(distance);
//        oclAllDist.download(allDist);

//        SANITY_CHECK(trainIdx);
//        SANITY_CHECK(distance);
//        SANITY_CHECK(allDist);
//    }
//#ifdef HAVE_OPENCV_GPU
//    else if (impl == "gpu")
//        CV_TEST_FAIL_NO_IMPL();
//#endif
//    else
//        CV_TEST_FAIL_NO_IMPL();
//}

PERF_TEST_P(BruteForceMatcherFixture, DISABLED_radiusMatch,
            OCL_BFMATCHER_TYPICAL_MAT_SIZES)
{
    const Size srcSize = GetParam();
    const string impl = getSelectedImpl();

    const float max_distance = 2.0f;
    vector<vector<DMatch> > matches(2);
    Mat query(srcSize, CV_32F), train(srcSize, CV_32F);
    declare.in(query, train);
    Mat trainIdx, distance, allDist;

    randu(query, 0.0f, 1.0f);
    randu(train, 0.0f, 1.0f);

    if (impl == "plain")
    {
        BFMatcher matcher (NORM_L2);
        TEST_CYCLE() matcher.radiusMatch(query, matches, max_distance);

        std::vector<DMatch> & matches0 = matches[0], & matches1 = matches[1];
        SANITY_CHECK_MATCHES(matches0);
        SANITY_CHECK_MATCHES(matches1);
    }
    else if (impl == "ocl")
    {
        ocl::oclMat oclQuery(query), oclTrain(train);
        ocl::BruteForceMatcher_OCL_base oclMatcher(ocl::BruteForceMatcher_OCL_base::L2Dist);

        TEST_CYCLE() oclMatcher.radiusMatch(oclQuery, oclTrain, matches, max_distance);

        std::vector<DMatch> & matches0 = matches[0], & matches1 = matches[1];
        SANITY_CHECK_MATCHES(matches0);
        SANITY_CHECK_MATCHES(matches1);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

//PERF_TEST_P(BruteForceMatcherFixture, radiusMatchSingle,
//            OCL_BFMATCHER_TYPICAL_MAT_SIZES)
//{
//    const Size srcSize = GetParam();
//    const string impl = getSelectedImpl();

//    const float max_distance = 2.0f;
//    Mat query(srcSize, CV_32F), train(srcSize, CV_32F);
//    Mat trainIdx, distance, nMatches;

//    randu(query, 0.0f, 1.0f);
//    randu(train, 0.0f, 1.0f);

//    if (impl == "plain")
//        CV_TEST_FAIL_NO_IMPL();
//    else if (impl == "ocl")
//    {
//        ocl::oclMat oclQuery(query), oclTrain(train), oclTrainIdx, oclDistance, oclNMatches;

//        TEST_CYCLE() oclMatcher->radiusMatchSingle(oclQuery, oclTrain, oclTrainIdx, oclDistance, oclNMatches, max_distance);

//        oclTrainIdx.download(trainIdx);
//        oclDistance.download(distance);
//        oclNMatches.download(nMatches);

//        SANITY_CHECK(trainIdx);
//        SANITY_CHECK(distance);
//        SANITY_CHECK(nMatches);
//    }
//#ifdef HAVE_OPENCV_GPU
//    else if (impl == "gpu")
//        CV_TEST_FAIL_NO_IMPL();
//#endif
//    else
//        CV_TEST_FAIL_NO_IMPL();
//}

#undef OCL_BFMATCHER_TYPICAL_MAT_SIZES
