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

//////////////////// BruteForceMatch /////////////////
PERFTEST(BruteForceMatcher)
{
    Mat trainIdx_cpu;
    Mat distance_cpu;
    Mat allDist_cpu;
    Mat nMatches_cpu;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        // Init CPU matcher
        int desc_len = 64;

        BFMatcher matcher(NORM_L2);

        Mat query;
        gen(query, size, desc_len, CV_32F, 0, 1);

        Mat train;
        gen(train, size, desc_len, CV_32F, 0, 1);
        // Output
        vector< vector<DMatch> > matches(2);
        vector< vector<DMatch> > d_matches(2);
        // Init GPU matcher
        ocl::BruteForceMatcher_OCL_base d_matcher(ocl::BruteForceMatcher_OCL_base::L2Dist);

        ocl::oclMat d_query(query);
        ocl::oclMat d_train(train);

        ocl::oclMat d_trainIdx, d_distance, d_allDist, d_nMatches;

        SUBTEST << size << "; match";

        matcher.match(query, train, matches[0]);

        CPU_ON;
        matcher.match(query, train, matches[0]);
        CPU_OFF;

        WARMUP_ON;
        d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
        WARMUP_OFF;

        GPU_ON;
        d_matcher.matchSingle(d_query, d_train, d_trainIdx, d_distance);
        GPU_OFF;

        GPU_FULL_ON;
        d_query.upload(query);
        d_train.upload(train);
        d_matcher.match(d_query, d_train, d_matches[0]);
        GPU_FULL_OFF;

        int diff = abs((int)d_matches[0].size() - (int)matches[0].size());
        if(diff == 0)
            TestSystem::instance().setAccurate(1, 0);
        else
            TestSystem::instance().setAccurate(0, diff);

        SUBTEST << size << "; knnMatch";

        matcher.knnMatch(query, train, matches, 2);

        CPU_ON;
        matcher.knnMatch(query, train, matches, 2);
        CPU_OFF;

        WARMUP_ON;
        d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);
        WARMUP_OFF;

        GPU_ON;
        d_matcher.knnMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_allDist, 2);
        GPU_OFF;

        GPU_FULL_ON;
        d_query.upload(query);
        d_train.upload(train);
        d_matcher.knnMatch(d_query, d_train, d_matches, 2);
        GPU_FULL_OFF;

        diff = abs((int)d_matches[0].size() - (int)matches[0].size());
        if(diff == 0)
            TestSystem::instance().setAccurate(1, 0);
        else
            TestSystem::instance().setAccurate(0, diff);

        SUBTEST << size << "; radiusMatch";

        float max_distance = 2.0f;

        matcher.radiusMatch(query, train, matches, max_distance);

        CPU_ON;
        matcher.radiusMatch(query, train, matches, max_distance);
        CPU_OFF;

        d_trainIdx.release();

        WARMUP_ON;
        d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);
        WARMUP_OFF;

        GPU_ON;
        d_matcher.radiusMatchSingle(d_query, d_train, d_trainIdx, d_distance, d_nMatches, max_distance);
        GPU_OFF;

        GPU_FULL_ON;
        d_query.upload(query);
        d_train.upload(train);
        d_matcher.radiusMatch(d_query, d_train, d_matches, max_distance);
        GPU_FULL_OFF;

        diff = abs((int)d_matches[0].size() - (int)matches[0].size());
        if(diff == 0)
            TestSystem::instance().setAccurate(1, 0);
        else
            TestSystem::instance().setAccurate(0, diff);
    }
}