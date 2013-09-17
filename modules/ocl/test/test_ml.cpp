///////////////////////////////////////////////////////////////////////////////////////
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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma,        jin@multicorewareinc.com
//    Xiaopeng Fu,   fuxiaopeng2222@163.com
//    Erping Pang,   pang_er_ping@163.com
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
using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;
///////K-NEAREST NEIGHBOR//////////////////////////
static void genTrainData(Mat& trainData, int trainDataRow, int trainDataCol,
                         Mat& trainLabel = Mat().setTo(Scalar::all(0)), int nClasses = 0)
{
    cv::RNG &rng = TS::ptr()->get_rng();
    cv::Size size(trainDataCol, trainDataRow);
    trainData = randomMat(rng, size, CV_32FC1, 1.0, 1000.0, false);
    if(nClasses != 0)
    {
        cv::Size size1(trainDataRow, 1);
        trainLabel = randomMat(rng, size1, CV_8UC1, 0, nClasses - 1, false);
        trainLabel.convertTo(trainLabel, CV_32FC1);
    }
}

PARAM_TEST_CASE(KNN, int, Size, int, bool)
{
    int k;
    int trainDataCol;
    int testDataRow;
    int nClass;
    bool regression;
    virtual void SetUp()
    {
        k = GET_PARAM(0);
        nClass = GET_PARAM(2);
        trainDataCol = GET_PARAM(1).width;
        testDataRow = GET_PARAM(1).height;
        regression = GET_PARAM(3);
    }
};

TEST_P(KNN, Accuracy)
{
    Mat trainData, trainLabels;
    const int trainDataRow = 500;
    genTrainData(trainData, trainDataRow, trainDataCol, trainLabels, nClass);

    Mat testData, testLabels;
    genTrainData(testData, testDataRow, trainDataCol);

    KNearestNeighbour knn_ocl;
    CvKNearest knn_cpu;
    Mat best_label_cpu;
    oclMat best_label_ocl;

    /*ocl k-Nearest_Neighbor start*/
    oclMat trainData_ocl;
    trainData_ocl.upload(trainData);
    Mat simpleIdx;
    knn_ocl.train(trainData, trainLabels, simpleIdx, regression);

    oclMat testdata;
    testdata.upload(testData);
    knn_ocl.find_nearest(testdata, k, best_label_ocl);
    /*ocl k-Nearest_Neighbor end*/

    /*cpu k-Nearest_Neighbor start*/
    knn_cpu.train(trainData, trainLabels, simpleIdx, regression);
    knn_cpu.find_nearest(testData, k, &best_label_cpu);
    /*cpu k-Nearest_Neighbor end*/
    if(regression)
    {
        EXPECT_MAT_SIMILAR(Mat(best_label_ocl), best_label_cpu, 1e-5);
    }
    else
    {
        EXPECT_MAT_NEAR(Mat(best_label_ocl), best_label_cpu, 0.0);
    }
}
INSTANTIATE_TEST_CASE_P(OCL_ML, KNN, Combine(Values(6, 5), Values(Size(200, 400), Size(300, 600)),
    Values(4, 3), Values(false, true)));
#endif // HAVE_OPENCL