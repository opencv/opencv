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
// Copyright (C) 2010-2012, Multicoreware inc., all rights reserved.
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

#include "precomp.hpp"
#ifdef HAVE_OPENCL
namespace
{

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // BruteForceMatcher

    CV_ENUM(DistType, cv::ocl::BruteForceMatcher_OCL_base::L1Dist, cv::ocl::BruteForceMatcher_OCL_base::L2Dist, cv::ocl::BruteForceMatcher_OCL_base::HammingDist)
    IMPLEMENT_PARAM_CLASS(DescriptorSize, int)

    PARAM_TEST_CASE(BruteForceMatcher/*, NormCode*/, DistType, DescriptorSize)
    {
        //std::vector<cv::ocl::Info> oclinfo;
        cv::ocl::BruteForceMatcher_OCL_base::DistType distType;
        int normCode;
        int dim;

        int queryDescCount;
        int countFactor;

        cv::Mat query, train;

        virtual void SetUp()
        {
            //normCode = GET_PARAM(0);
            distType = (cv::ocl::BruteForceMatcher_OCL_base::DistType)(int)GET_PARAM(0);
            dim = GET_PARAM(1);

            //int devnums = getDevice(oclinfo, OPENCV_DEFAULT_OPENCL_DEVICE);
            //CV_Assert(devnums > 0);

            queryDescCount = 300; // must be even number because we split train data in some cases in two
            countFactor = 4; // do not change it

            cv::RNG &rng = cvtest::TS::ptr()->get_rng();

            cv::Mat queryBuf, trainBuf;

            // Generate query descriptors randomly.
            // Descriptor vector elements are integer values.
            queryBuf.create(queryDescCount, dim, CV_32SC1);
            rng.fill(queryBuf, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(3));
            queryBuf.convertTo(queryBuf, CV_32FC1);

            // Generate train decriptors as follows:
            // copy each query descriptor to train set countFactor times
            // and perturb some one element of the copied descriptors in
            // in ascending order. General boundaries of the perturbation
            // are (0.f, 1.f).
            trainBuf.create(queryDescCount * countFactor, dim, CV_32FC1);
            float step = 1.f / countFactor;
            for (int qIdx = 0; qIdx < queryDescCount; qIdx++)
            {
                cv::Mat queryDescriptor = queryBuf.row(qIdx);
                for (int c = 0; c < countFactor; c++)
                {
                    int tIdx = qIdx * countFactor + c;
                    cv::Mat trainDescriptor = trainBuf.row(tIdx);
                    queryDescriptor.copyTo(trainDescriptor);
                    int elem = rng(dim);
                    float diff = rng.uniform(step * c, step * (c + 1));
                    trainDescriptor.at<float>(0, elem) += diff;
                }
            }

            queryBuf.convertTo(query, CV_32F);
            trainBuf.convertTo(train, CV_32F);
        }
    };

    TEST_P(BruteForceMatcher, DISABLED_Match_Single)
    {
        cv::ocl::BruteForceMatcher_OCL_base matcher(distType);

        std::vector<cv::DMatch> matches;
        matcher.match(cv::ocl::oclMat(query),  cv::ocl::oclMat(train),  matches);

        ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

        int badCount = 0;
        for (size_t i = 0; i < matches.size(); i++)
        {
            cv::DMatch match = matches[i];
            if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor) || (match.imgIdx != 0))
                badCount++;
        }

        ASSERT_EQ(0, badCount);
    }

    TEST_P(BruteForceMatcher, DISABLED_KnnMatch_2_Single)
    {
        const int knn = 2;

        cv::ocl::BruteForceMatcher_OCL_base matcher(distType);

        std::vector< std::vector<cv::DMatch> > matches;
        matcher.knnMatch(cv::ocl::oclMat(query), cv::ocl::oclMat(train), matches, knn);

        ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

        int badCount = 0;
        for (size_t i = 0; i < matches.size(); i++)
        {
            if ((int)matches[i].size() != knn)
                badCount++;
            else
            {
                int localBadCount = 0;
                for (int k = 0; k < knn; k++)
                {
                    cv::DMatch match = matches[i][k];
                    if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor + k) || (match.imgIdx != 0))
                        localBadCount++;
                }
                badCount += localBadCount > 0 ? 1 : 0;
            }
        }

        ASSERT_EQ(0, badCount);
    }

    TEST_P(BruteForceMatcher, RadiusMatch_Single)
    {
        float radius;
        if(distType == cv::ocl::BruteForceMatcher_OCL_base::L2Dist)
            radius = 1.f / countFactor / countFactor;
        else
            radius = 1.f / countFactor;

        cv::ocl::BruteForceMatcher_OCL_base matcher(distType);

        // assume support atomic.
        //if (!supportFeature(devInfo, cv::gpu::GLOBAL_ATOMICS))
        //{
        //    try
        //    {
        //        std::vector< std::vector<cv::DMatch> > matches;
        //        matcher.radiusMatch(loadMat(query), loadMat(train), matches, radius);
        //    }
        //    catch (const cv::Exception& e)
        //    {
        //        ASSERT_EQ(CV_StsNotImplemented, e.code);
        //    }
        //}
        //else
        {
            std::vector< std::vector<cv::DMatch> > matches;
            matcher.radiusMatch(cv::ocl::oclMat(query), cv::ocl::oclMat(train), matches, radius);

            ASSERT_EQ(static_cast<size_t>(queryDescCount), matches.size());

            int badCount = 0;
            for (size_t i = 0; i < matches.size(); i++)
            {
                if ((int)matches[i].size() != 1)
                {
                    badCount++;
                }
                else
                {
                    cv::DMatch match = matches[i][0];
                    if ((match.queryIdx != (int)i) || (match.trainIdx != (int)i * countFactor) || (match.imgIdx != 0))
                        badCount++;
                }
            }

            ASSERT_EQ(0, badCount);
        }
    }

    INSTANTIATE_TEST_CASE_P(GPU_Features2D, BruteForceMatcher, testing::Combine(
                                //ALL_DEVICES,
                                testing::Values(DistType(cv::ocl::BruteForceMatcher_OCL_base::L1Dist), DistType(cv::ocl::BruteForceMatcher_OCL_base::L2Dist)),
                                testing::Values(DescriptorSize(57), DescriptorSize(64), DescriptorSize(83), DescriptorSize(128), DescriptorSize(179), DescriptorSize(256), DescriptorSize(304))));

} // namespace
#endif
