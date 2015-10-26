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
//    Erping Pang,   pang_er_ping@163.com
//    Xiaopeng Fu,   fuxiaopeng2222@163.com
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#ifdef HAVE_OPENCL

using namespace cvtest;
using namespace testing;
using namespace std;
using namespace cv;

#define OCL_KMEANS_USE_INITIAL_LABELS 1
#define OCL_KMEANS_PP_CENTERS         2

PARAM_TEST_CASE(Kmeans, int, int, int)
{
    int type;
    int K;
    int flags;
    Mat src ;
    ocl::oclMat d_src, d_dists;

    Mat labels, centers;
    ocl::oclMat d_labels, d_centers;
    virtual void SetUp()
    {
        K = GET_PARAM(0);
        type = GET_PARAM(1);
        flags = GET_PARAM(2);

        // MWIDTH=256, MHEIGHT=256. defined in utility.hpp
        Size size = Size(MWIDTH, MHEIGHT);
        src.create(size, type);
        int row_idx = 0;
        const int max_neighbour = MHEIGHT / K - 1;
        CV_Assert(K <= MWIDTH);
        for(int i = 0; i < K; i++ )
        {
            Mat center_row_header = src.row(row_idx);
            center_row_header.setTo(0);
            int nchannel = center_row_header.channels();
            for(int j = 0; j < nchannel; j++)
                center_row_header.at<float>(0, i*nchannel+j) = 50000.0;

            for(int j = 0; (j < max_neighbour) ||
                           (i == K-1 && j < max_neighbour + MHEIGHT%K); j ++)
            {
                Mat cur_row_header = src.row(row_idx + 1 + j);
                center_row_header.copyTo(cur_row_header);
                Mat tmpmat = randomMat(cur_row_header.size(), cur_row_header.type(), -200, 200, false);
                cur_row_header += tmpmat;
            }
            row_idx += 1 + max_neighbour;
        }
    }
};
OCL_TEST_P(Kmeans, Mat){
    if(flags & KMEANS_USE_INITIAL_LABELS)
    {
        // inital a given labels
        labels.create(src.rows, 1, CV_32S);
        int *label = labels.ptr<int>();
        for(int i = 0; i < src.rows; i++)
            label[i] = rng.uniform(0, K);
        d_labels.upload(labels);
    }
    d_src.upload(src);

    for(int j = 0; j < LOOP_TIMES; j++)
    {
        kmeans(src, K, labels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 0),
            1, flags, centers);
        ocl::kmeans(d_src, K, d_labels,
            TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 0),
            1, flags, d_centers);
        Mat dd_labels(d_labels);
        Mat dd_centers(d_centers);
        if(flags & KMEANS_USE_INITIAL_LABELS)
        {
            EXPECT_MAT_NEAR(labels, dd_labels, 0);
            EXPECT_MAT_NEAR(centers, dd_centers, 1e-3);
        }
        else
        {
            int row_idx = 0;
            for(int i = 0; i < K; i++)
            {
                // verify lables with ground truth resutls
                int label = labels.at<int>(row_idx);
                int header_label = dd_labels.at<int>(row_idx);
                for(int j2 = 0; (j2 < MHEIGHT/K)||(i == K-1 && j2 < MHEIGHT/K+MHEIGHT%K); j2++)
                {
                    ASSERT_NEAR(labels.at<int>(row_idx+j2), label, 0);
                    ASSERT_NEAR(dd_labels.at<int>(row_idx+j2), header_label, 0);
                }

                // verify centers
                float *center = centers.ptr<float>(label);
                float *header_center = dd_centers.ptr<float>(header_label);
                for(int t = 0; t < centers.cols; t++)
                    ASSERT_NEAR(center[t], header_center[t], 1e-3);

                row_idx += MHEIGHT/K;
            }
        }
    }
}

INSTANTIATE_TEST_CASE_P(OCL_ML, Kmeans, Combine(
    Values(3, 5, 8),
    Values(CV_32FC1, CV_32FC2, CV_32FC4),
    Values(OCL_KMEANS_USE_INITIAL_LABELS/*, OCL_KMEANS_PP_CENTERS*/)));


/////////////////////////////// DistanceToCenters //////////////////////////////////////////

CV_ENUM(DistType, NORM_L1, NORM_L2SQR)

PARAM_TEST_CASE(distanceToCenters, DistType, bool)
{
    int distType;
    bool useRoi;

    Mat src, centers, src_roi, centers_roi;
    ocl::oclMat ocl_src, ocl_centers, ocl_src_roi, ocl_centers_roi;

    virtual void SetUp()
    {
        distType = GET_PARAM(0);
        useRoi = GET_PARAM(1);
    }

    void random_roi()
    {
        Size roiSizeSrc = randomSize(1, MAX_VALUE);
        Size roiSizeCenters = randomSize(1, MAX_VALUE);
        roiSizeSrc.width = roiSizeCenters.width;

        Border srcBorder = randomBorder(0, useRoi ? MAX_VALUE : 0);
        randomSubMat(src, src_roi, roiSizeSrc, srcBorder, CV_32FC1, -MAX_VALUE, MAX_VALUE);

        Border centersBorder = randomBorder(0, useRoi ? 500 : 0);
        randomSubMat(centers, centers_roi, roiSizeCenters, centersBorder, CV_32FC1, -MAX_VALUE, MAX_VALUE);

        for (int i = 0; i < centers.rows; i++)
            centers.at<float>(i, randomInt(0, centers.cols)) = (float)randomDouble(SHRT_MAX, INT_MAX);

        generateOclMat(ocl_src, ocl_src_roi, src, roiSizeSrc, srcBorder);
        generateOclMat(ocl_centers, ocl_centers_roi, centers, roiSizeCenters, centersBorder);
    }
};

OCL_TEST_P(distanceToCenters, Accuracy)
{
    for (int j = 0; j < LOOP_TIMES; j++)
    {
        random_roi();

        Mat labels, dists;
        ocl::distanceToCenters(ocl_src_roi, ocl_centers_roi, dists, labels, distType);

        EXPECT_EQ(dists.size(), labels.size());

        Mat batch_dists;
        cv::batchDistance(src_roi, centers_roi, batch_dists, CV_32FC1, noArray(), distType);

        std::vector<float> gold_dists_v;
        gold_dists_v.reserve(batch_dists.rows);

        for (int i = 0; i < batch_dists.rows; i++)
        {
            Mat r = batch_dists.row(i);
            double mVal;
            Point mLoc;
            minMaxLoc(r, &mVal, NULL, &mLoc, NULL);

            int ocl_label = labels.at<int>(i, 0);
            EXPECT_EQ(mLoc.x, ocl_label);

            gold_dists_v.push_back(static_cast<float>(mVal));
        }

        double relative_error = cv::norm(Mat(gold_dists_v), dists, NORM_INF | NORM_RELATIVE);
        ASSERT_LE(relative_error, 1e-5);
    }
}

INSTANTIATE_TEST_CASE_P (OCL_ML, distanceToCenters, Combine(DistType::all(), Bool()));

#endif
