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
//    Jin Ma, jin@multicorewareinc.com
//    Xiaopeng Fu, fuxiaopeng2222@163.com
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
using namespace std;
using namespace cv::ocl;
using namespace cv;
using std::tr1::tuple;
using std::tr1::get;
////////////////////////////////// K-NEAREST NEIGHBOR ////////////////////////////////////
static void genData(Mat& trainData, Size size, Mat& trainLabel = Mat().setTo(Scalar::all(0)), int nClasses = 0)
{
    trainData.create(size, CV_32FC1);
    randu(trainData, 1.0, 100.0);

    if(nClasses != 0)
    {
        trainLabel.create(size.height, 1, CV_8UC1);
        randu(trainLabel, 0, nClasses - 1);
        trainLabel.convertTo(trainLabel, CV_32FC1);
    }
}

typedef tuple<int> KNNParamType;
typedef TestBaseWithParam<KNNParamType> KNNFixture;

PERF_TEST_P(KNNFixture, KNN,
            testing::Values(1000, 2000, 4000))
{
    KNNParamType params = GetParam();
    const int rows = get<0>(params);
    int columns = 100;
    int k = rows/250;

    Mat trainData, trainLabels;
    Size size(columns, rows);
    genData(trainData, size, trainLabels, 3);

    Mat testData;
    genData(testData, size);
    Mat best_label;

    if(RUN_PLAIN_IMPL)
    {
        TEST_CYCLE()
        {
            CvKNearest knn_cpu;
            knn_cpu.train(trainData, trainLabels);
            knn_cpu.find_nearest(testData, k, &best_label);
        }
    }else if(RUN_OCL_IMPL)
    {
        cv::ocl::oclMat best_label_ocl;
        cv::ocl::oclMat testdata;
        testdata.upload(testData);

        OCL_TEST_CYCLE()
        {
            cv::ocl::KNearestNeighbour knn_ocl;
            knn_ocl.train(trainData, trainLabels);
            knn_ocl.find_nearest(testdata, k, best_label_ocl);
        }
        best_label_ocl.download(best_label);
    }else
        OCL_PERF_ELSE
    SANITY_CHECK(best_label);
}