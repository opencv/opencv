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
//     and/or other oclMaterials provided with the distribution.
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
    cv::Mat src ;
    ocl::oclMat d_src, d_dists;

    Mat labels, centers;
    ocl::oclMat d_labels, d_centers;
    cv::RNG rng ;
    virtual void SetUp(){
        K = GET_PARAM(0);
        type = GET_PARAM(1);
        flags = GET_PARAM(2);
        rng = TS::ptr()->get_rng();

        // MWIDTH=256, MHEIGHT=256. defined in utility.hpp
        cv::Size size = cv::Size(MWIDTH, MHEIGHT);
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
                Mat tmpmat = randomMat(rng, cur_row_header.size(), cur_row_header.type(), -200, 200, false);
                cur_row_header += tmpmat;
            }
            row_idx += 1 + max_neighbour;
        }
    }
};
TEST_P(Kmeans, Mat){

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
            TermCriteria( TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0),
            1, flags, centers);

        ocl::kmeans(d_src, K, d_labels,
            TermCriteria( TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 0),
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
                for(int j = 0; (j < MHEIGHT/K)||(i == K-1 && j < MHEIGHT/K+MHEIGHT%K); j++)
                {
                    ASSERT_NEAR(labels.at<int>(row_idx+j), label, 0);
                    ASSERT_NEAR(dd_labels.at<int>(row_idx+j), header_label, 0);
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

#endif
