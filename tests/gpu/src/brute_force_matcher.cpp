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

#include "gputest.hpp"
#include <algorithm>

using namespace cv;
using namespace cv::gpu;
using namespace std;

class CV_GpuBruteForceMatcherTest : public CvTest 
{
public:
    CV_GpuBruteForceMatcherTest() : CvTest( "GPU-BruteForceMatcher", "BruteForceMatcher" ) {}

protected:
    void run(int) 
    {
        try 
        {
            BruteForceMatcher< L2<float> > matcherCPU;
            BruteForceMatcher_GPU< L2<float> > matcherGPU;
            
            vector<DMatch> matchesCPU, matchesGPU;
            vector< vector<DMatch> > knnMatchesCPU, knnMatchesGPU;
            vector< vector<DMatch> > radiusMatchesCPU, radiusMatchesGPU;

            RNG rng(*ts->get_rng());

            const int desc_len = rng.uniform(40, 300);

            Mat queryCPU(rng.uniform(100, 300), desc_len, CV_32F);            
            rng.fill(queryCPU, cv::RNG::UNIFORM, cv::Scalar::all(0.0), cv::Scalar::all(10.0));
            GpuMat queryGPU(queryCPU);

            const int nTrains = rng.uniform(1, 5);

            vector<Mat> trainsCPU(nTrains);
            vector<GpuMat> trainsGPU(nTrains);

            vector<Mat> masksCPU(nTrains);
            vector<GpuMat> masksGPU(nTrains);

            for (int i = 0; i < nTrains; ++i)
            {
                Mat train(rng.uniform(100, 300), desc_len, CV_32F);
                rng.fill(train, cv::RNG::UNIFORM, cv::Scalar::all(0.0), cv::Scalar::all(10.0));

                trainsCPU[i] = train;
                trainsGPU[i].upload(train);

                bool with_mask = rng.uniform(0, 10) < 5;
                if (with_mask)
                {
                    Mat mask(queryCPU.rows, train.rows, CV_8U);
                    rng.fill(mask, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(200));

                    masksCPU[i] = mask;
                    masksGPU[i].upload(mask);
                }
            }

            matcherCPU.add(trainsCPU);
            matcherGPU.add(trainsGPU);

            matcherCPU.match(queryCPU, matchesCPU, masksCPU);
            matcherGPU.match(queryGPU, matchesGPU, masksGPU);

            if (!compareMatches(matchesCPU, matchesGPU))
            {
                ts->printf(CvTS::LOG, "Match FAIL");
                ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
                return;
            }

            const int knn = rng.uniform(3, 10);

            matcherCPU.knnMatch(queryCPU, knnMatchesCPU, knn, masksCPU, true);
            matcherGPU.knnMatch(queryGPU, knnMatchesGPU, knn, masksGPU, true);

            if (!compareMatches(knnMatchesCPU, knnMatchesGPU))
            {
                ts->printf(CvTS::LOG, "KNN Match FAIL");
                ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
                return;
            }

            const float maxDistance = rng.uniform(25.0f, 65.0f);
            
            matcherCPU.radiusMatch(queryCPU, radiusMatchesCPU, maxDistance, masksCPU, true);
            matcherGPU.radiusMatch(queryGPU, radiusMatchesGPU, maxDistance, masksGPU, true);

            if (!compareMatches(radiusMatchesCPU, radiusMatchesGPU))
            {
                ts->printf(CvTS::LOG, "Radius Match FAIL");
                ts->set_failed_test_info(CvTS::FAIL_MISMATCH);
                return;
            }
        }
        catch (const cv::Exception& e) 
        {
            if (!check_and_treat_gpu_exception(e, ts))
                throw;
            return;
        }

        ts->set_failed_test_info(CvTS::OK);
    }

private:
    static void convertMatches(const vector< vector<DMatch> >& knnMatches, vector<DMatch>& matches)
    {
        matches.clear();
        for (size_t i = 0; i < knnMatches.size(); ++i)
            copy(knnMatches[i].begin(), knnMatches[i].end(), back_inserter(matches));
    }

    struct DMatchEqual : public binary_function<DMatch, DMatch, bool>
    {
        bool operator()(const DMatch& m1, const DMatch& m2) const
        {
            return m1.imgIdx == m2.imgIdx && m1.queryIdx == m2.queryIdx && m1.trainIdx == m2.trainIdx;
        }
    };
    
    static bool compareMatches(const vector<DMatch>& matches1, const vector<DMatch>& matches2)
    {
        if (matches1.size() != matches2.size())
            return false;

        return equal(matches1.begin(), matches1.end(), matches2.begin(), DMatchEqual());
    }

    static bool compareMatches(const vector< vector<DMatch> >& matches1, const vector< vector<DMatch> >& matches2)
    {
        vector<DMatch> m1, m2;
        convertMatches(matches1, m1);
        convertMatches(matches2, m2);
        return compareMatches(m1, m2);
    }
} brute_force_matcher_test;