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
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jin Ma, jin@multicorewareinc.com
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

#include "precomp.hpp"
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

KNearestNeighbour::KNearestNeighbour()
{
    clear();
}

KNearestNeighbour::~KNearestNeighbour()
{
    clear();
    samples_ocl.release();
}

void KNearestNeighbour::clear()
{
    CvKNearest::clear();
}

bool KNearestNeighbour::train(const Mat& trainData, Mat& labels, Mat& sampleIdx,
                              bool isRegression, int _max_k, bool updateBase)
{
    max_k = _max_k;
    bool cv_knn_train = CvKNearest::train(trainData, labels, sampleIdx, isRegression, max_k, updateBase);

    CvVectors* s = CvKNearest::samples;

    cv::Mat samples_mat(s->count, CvKNearest::var_count + 1, s->type);

    float* s1 = (float*)(s + 1);
    for(int i = 0; i < s->count; i++)
    {
        float* t1 = s->data.fl[i];
        for(int j = 0; j < CvKNearest::var_count; j++)
        {
            Point pos(j, i);
            samples_mat.at<float>(pos) = t1[j];
        }

        Point pos_label(CvKNearest::var_count, i);
        samples_mat.at<float>(pos_label) = s1[i];
    }

    samples_ocl = samples_mat;
    return cv_knn_train;
}

void KNearestNeighbour::find_nearest(const oclMat& samples, int k, oclMat& lables)
{
    CV_Assert(!samples_ocl.empty());
    lables.create(samples.rows, 1, CV_32FC1);

    CV_Assert(samples.cols == CvKNearest::var_count);
    CV_Assert(samples.type() == CV_32FC1);
    CV_Assert(k >= 1 && k <= max_k);

    int k1 = KNearest::get_sample_count();
    k1 = MIN( k1, k );

    String kernel_name = "knn_find_nearest";
    cl_ulong local_memory_size = (cl_ulong)Context::getContext()->getDeviceInfo().localMemorySize;
    int nThreads = local_memory_size / (2 * k * 4);
    if(nThreads >= 256)
        nThreads = 256;

    int smem_size = nThreads * k * 4 * 2;
    size_t local_thread[] = {1, nThreads, 1};
    size_t global_thread[] = {1, samples.rows, 1};

    char build_option[50];
    if(!Context::getContext()->supportsFeature(FEATURE_CL_DOUBLE))
    {
        sprintf(build_option, " ");
    }else
        sprintf(build_option, "-D DOUBLE_SUPPORT");

    std::vector< std::pair<size_t, const void*> > args;

    int samples_ocl_step = samples_ocl.step/samples_ocl.elemSize();
    int samples_step = samples.step/samples.elemSize();
    int lables_step = lables.step/lables.elemSize();

    int _regression = 0;
    if(CvKNearest::regression)
        _regression = 1;

    args.push_back(make_pair(sizeof(cl_mem), (void*)&samples.data));
    args.push_back(make_pair(sizeof(cl_int), (void*)&samples.rows));
    args.push_back(make_pair(sizeof(cl_int), (void*)&samples.cols));
    args.push_back(make_pair(sizeof(cl_int), (void*)&samples_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&k));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&samples_ocl.data));
    args.push_back(make_pair(sizeof(cl_int), (void*)&samples_ocl.rows));
    args.push_back(make_pair(sizeof(cl_int), (void*)&samples_ocl_step));
    args.push_back(make_pair(sizeof(cl_mem), (void*)&lables.data));
    args.push_back(make_pair(sizeof(cl_int), (void*)&lables_step));
    args.push_back(make_pair(sizeof(cl_int), (void*)&_regression));
    args.push_back(make_pair(sizeof(cl_int), (void*)&k1));
    args.push_back(make_pair(sizeof(cl_int), (void*)&samples_ocl.cols));
    args.push_back(make_pair(sizeof(cl_int), (void*)&nThreads));
    args.push_back(make_pair(smem_size, (void*)NULL));
    openCLExecuteKernel(Context::getContext(), &knearest, kernel_name, global_thread, local_thread, args, -1, -1, build_option);
}