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

using namespace cv;
using namespace cv::ocl;
using namespace cvtest;
using namespace testing;

///////K-NEAREST NEIGHBOR//////////////////////////

static void genTrainData(cv::RNG& rng, Mat& trainData, int trainDataRow, int trainDataCol,
                         Mat& trainLabel = Mat().setTo(Scalar::all(0)), int nClasses = 0)
{
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

OCL_TEST_P(KNN, Accuracy)
{
    Mat trainData, trainLabels;
    const int trainDataRow = 500;
    genTrainData(rng, trainData, trainDataRow, trainDataCol, trainLabels, nClass);

    Mat testData, testLabels;
    genTrainData(rng, testData, testDataRow, trainDataCol);

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

#ifdef HAVE_CLAMDBLAS // TODO does not work non-blas version of SVM

////////////////////////////////SVM/////////////////////////////////////////////////

PARAM_TEST_CASE(SVM_OCL, int, int, int)
{
    cv::Size size;
    int kernel_type;
    int svm_type;
    Mat src, labels, samples, labels_predict;
    int K;

    virtual void SetUp()
    {

        kernel_type = GET_PARAM(0);
        svm_type = GET_PARAM(1);
        K = GET_PARAM(2);
        cv::Size size = cv::Size(MWIDTH, MHEIGHT);
        src.create(size, CV_32FC1);
        labels.create(1, size.height, CV_32SC1);
        int row_idx = 0;
        const int max_number = size.height / K - 1;
        CV_Assert(K <= size.height);
        for(int i = 0; i < K; i++ )
        {
            Mat center_row_header = src.row(row_idx);
            center_row_header.setTo(0);
            int nchannel = center_row_header.channels();
            for(int j = 0; j < nchannel; j++)
            {
                center_row_header.at<float>(0, i * nchannel + j) = 500.0;
            }
            labels.at<int>(0, row_idx) = i;
            for(int j = 0; (j < max_number) ||
                    (i == K - 1 && j < max_number + size.height % K); j ++)
            {
                Mat cur_row_header = src.row(row_idx + 1 + j);
                center_row_header.copyTo(cur_row_header);
                Mat tmpmat = randomMat(cur_row_header.size(), cur_row_header.type(), 1, 100, false);
                cur_row_header += tmpmat;
                labels.at<int>(0, row_idx + 1 + j) = i;
            }
            row_idx += 1 + max_number;
        }
        labels.convertTo(labels, CV_32FC1);
        cv::Size test_size = cv::Size(MWIDTH, 100);
        samples.create(test_size, CV_32FC1);
        labels_predict.create(1, test_size.height, CV_32SC1);
        const int max_number_test = test_size.height / K - 1;
        row_idx = 0;
        for(int i = 0; i < K; i++ )
        {
            Mat center_row_header = samples.row(row_idx);
            center_row_header.setTo(0);
            int nchannel = center_row_header.channels();
            for(int j = 0; j < nchannel; j++)
            {
                center_row_header.at<float>(0, i * nchannel + j) = 500.0;
            }
            labels_predict.at<int>(0, row_idx) = i;
            for(int j = 0; (j < max_number_test) ||
                    (i == K - 1 && j < max_number_test + test_size.height % K); j ++)
            {
                Mat cur_row_header = samples.row(row_idx + 1 + j);
                center_row_header.copyTo(cur_row_header);
                Mat tmpmat = randomMat(cur_row_header.size(), cur_row_header.type(), 1, 100, false);
                cur_row_header += tmpmat;
                labels_predict.at<int>(0, row_idx + 1 + j) = i;
            }
            row_idx += 1 + max_number_test;
        }
        labels_predict.convertTo(labels_predict, CV_32FC1);
    }
};

OCL_TEST_P(SVM_OCL, Accuracy)
{
    CvSVMParams params;
    params.degree = 0.4;
    params.gamma = 1;
    params.coef0 = 1;
    params.C = 1;
    params.nu = 0.5;
    params.p = 1;
    params.svm_type = svm_type;
    params.kernel_type = kernel_type;

    params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.001);

    CvSVM SVM;
    SVM.train(src, labels, Mat(), Mat(), params);

    cv::ocl::CvSVM_OCL SVM_OCL;
    SVM_OCL.train(src, labels, Mat(), Mat(), params);

    int c = SVM.get_support_vector_count();
    int c1 = SVM_OCL.get_support_vector_count();

    Mat sv(c, MHEIGHT, CV_32FC1);
    Mat sv_ocl(c1, MHEIGHT, CV_32FC1);
    for(int i = 0; i < c; i++)
    {
        const float* v = SVM.get_support_vector(i);

        for(int j = 0; j < MHEIGHT; j++)
        {
            sv.at<float>(i, j) = v[j];
        }
    }
    for(int i = 0; i < c1; i++)
    {
        const float* v_ocl = SVM_OCL.get_support_vector(i);

        for(int j = 0; j < MHEIGHT; j++)
        {
            sv_ocl.at<float>(i, j) = v_ocl[j];
        }
    }
    cv::BFMatcher matcher(cv::NORM_L2);
    std::vector<cv::DMatch> matches;
    matcher.match(sv, sv_ocl, matches);
    int count = 0;

    for(std::vector<cv::DMatch>::iterator itr = matches.begin(); itr != matches.end(); itr++)
    {
        if((*itr).distance < 0.1)
        {
            count ++;
        }
    }
    if(c != 0)
    {
        float matchedRatio = (float)count / c;
        EXPECT_GT(matchedRatio, 0.95);
    }
    if(c != 0)
    {
        CvMat *result = cvCreateMat(1, samples.rows, CV_32FC1);
        CvMat test_samples = samples;

        CvMat *result_ocl = cvCreateMat(1, samples.rows, CV_32FC1);

        SVM.predict(&test_samples, result);

        SVM_OCL.predict(&test_samples, result_ocl);

        int true_resp = 0, true_resp_ocl = 0;
        for (int i = 0; i < samples.rows; i++)
        {
            if (result->data.fl[i] == labels_predict.at<float>(0, i))
            {
                true_resp++;
            }
        }
        float matchedRatio = (float)true_resp / samples.rows;

        for (int i = 0; i < samples.rows; i++)
        {
            if (result_ocl->data.fl[i] == labels_predict.at<float>(0, i))
            {
                true_resp_ocl++;
            }
        }
        float matchedRatio_ocl = (float)true_resp_ocl / samples.rows;

        if(matchedRatio != 0 && true_resp_ocl < true_resp)
        {
            EXPECT_NEAR(matchedRatio_ocl, matchedRatio, 0.03);
        }
    }
}

// TODO FIXIT: CvSVM::EPS_SVR case is crashed inside CPU implementation
// Anonymous enums are not supported well so cast them to 'int'

INSTANTIATE_TEST_CASE_P(OCL_ML, SVM_OCL, testing::Combine(
                            Values((int)CvSVM::LINEAR, (int)CvSVM::POLY, (int)CvSVM::RBF, (int)CvSVM::SIGMOID),
                            Values((int)CvSVM::C_SVC, (int)CvSVM::NU_SVC, (int)CvSVM::ONE_CLASS, (int)CvSVM::NU_SVR),
                            Values(2, 3, 4)
                        ));

#endif // HAVE_CLAMDBLAS

#endif // HAVE_OPENCL
