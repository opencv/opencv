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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
// Authors:
//  * Peter Andreas Entschev, peter@entschev.com
//
//M*/

#include "precomp.hpp"
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

cv::ocl::FAST_OCL::FAST_OCL(int _threshold, bool _nonmaxSupression, double _keypointsRatio) :
    nonmaxSupression(_nonmaxSupression), threshold(_threshold), keypointsRatio(_keypointsRatio), count_(0)
{
}

void cv::ocl::FAST_OCL::operator ()(const oclMat& image, const oclMat& mask, std::vector<KeyPoint>& keypoints)
{
    if (image.empty())
        return;

    (*this)(image, mask, d_keypoints_);
    downloadKeypoints(d_keypoints_, keypoints);
}

void cv::ocl::FAST_OCL::downloadKeypoints(const oclMat& d_keypoints, std::vector<KeyPoint>& keypoints)
{
    if (d_keypoints.empty())
        return;

    Mat h_keypoints(d_keypoints);
    convertKeypoints(h_keypoints, keypoints);
}

void cv::ocl::FAST_OCL::convertKeypoints(const Mat& h_keypoints, std::vector<KeyPoint>& keypoints)
{
    if (h_keypoints.empty())
        return;

    CV_Assert(h_keypoints.rows == ROWS_COUNT && h_keypoints.elemSize() == 4);

    int npoints = h_keypoints.cols;

    keypoints.resize(npoints);

    const float* loc_x = h_keypoints.ptr<float>(X_ROW);
    const float* loc_y = h_keypoints.ptr<float>(Y_ROW);
    const float* response_row = h_keypoints.ptr<float>(RESPONSE_ROW);

    for (int i = 0; i < npoints; ++i)
    {
        KeyPoint kp(loc_x[i], loc_y[i], static_cast<float>(FEATURE_SIZE), -1, response_row[i]);
        keypoints[i] = kp;
    }
}

void cv::ocl::FAST_OCL::operator ()(const oclMat& img, const oclMat& mask, oclMat& keypoints)
{
    calcKeyPointsLocation(img, mask);
    keypoints.cols = getKeyPoints(keypoints);
}

int cv::ocl::FAST_OCL::calcKeyPointsLocation(const oclMat& img, const oclMat& mask)
{
    CV_Assert(img.type() == CV_8UC1);
    CV_Assert(mask.empty() || (mask.type() == CV_8UC1 && mask.size() == img.size()));

    int maxKeypoints = static_cast<int>(keypointsRatio * img.size().area());

    ensureSizeIsEnough(ROWS_COUNT, maxKeypoints, CV_32SC1, kpLoc_);
    kpLoc_.setTo(Scalar::all(0));

    if (nonmaxSupression)
    {
        ensureSizeIsEnough(img.size(), CV_32SC1, score_);
        score_.setTo(Scalar::all(0));
    }

    count_ = calcKeypointsOCL(img, mask, maxKeypoints);
    count_ = std::min(count_, maxKeypoints);

    return count_;
}

int cv::ocl::FAST_OCL::calcKeypointsOCL(const oclMat& img, const oclMat& mask, int maxKeypoints)
{
    size_t localThreads[3] = {16, 16, 1};
    size_t globalThreads[3] = {divUp(img.cols - 6, localThreads[0]) * localThreads[0],
                               divUp(img.rows - 6, localThreads[1]) * localThreads[1],
                               1};

    Context *clCxt = Context::getContext();
    String kernelName = (mask.empty()) ? "calcKeypoints" : "calcKeypointsWithMask";
    std::vector< std::pair<size_t, const void *> > args;

    int counter = 0;
    int err = CL_SUCCESS;
    cl_mem counterCL = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(),
                                    CL_MEM_COPY_HOST_PTR, sizeof(int),
                                    &counter, &err);

    int kpLocStep = kpLoc_.step / kpLoc_.elemSize();
    int scoreStep = score_.step / score_.elemSize();
    int nms = (nonmaxSupression) ? 1 : 0;

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&img.data));
    if (!mask.empty()) args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mask.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&kpLoc_.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&score_.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&counterCL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&nms));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&maxKeypoints));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&threshold));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img.rows));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&img.cols));
    if (!mask.empty()) args.push_back( std::make_pair( sizeof(cl_int), (void *)&mask.step));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&kpLocStep));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&scoreStep));

    openCLExecuteKernel(clCxt, &featdetect_fast, kernelName, globalThreads, localThreads, args, -1, -1);

    openCLSafeCall(clEnqueueReadBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(),
                                       counterCL, CL_TRUE, 0, sizeof(int), &counter, 0, NULL, NULL));
    openCLSafeCall(clReleaseMemObject(counterCL));

    return counter;
}

int cv::ocl::FAST_OCL::nonmaxSupressionOCL(oclMat& keypoints)
{
    size_t localThreads[3] = {256, 1, 1};
    size_t globalThreads[3] = {count_, 1, 1};

    Context *clCxt = Context::getContext();
    String kernelName = "nonmaxSupression";
    std::vector< std::pair<size_t, const void *> > args;

    int counter = 0;
    int err = CL_SUCCESS;
    cl_mem counterCL = clCreateBuffer(*(cl_context*)clCxt->getOpenCLContextPtr(),
                                    CL_MEM_COPY_HOST_PTR, sizeof(int),
                                    &counter, &err);

    int kpLocStep = kpLoc_.step / kpLoc_.elemSize();
    int sStep = score_.step / score_.elemSize();
    int kStep = keypoints.step / keypoints.elemSize();

    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&kpLoc_.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&score_.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&keypoints.data));
    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&counterCL));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&count_));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&kpLocStep));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&sStep));
    args.push_back( std::make_pair( sizeof(cl_int), (void *)&kStep));

    openCLExecuteKernel(clCxt, &featdetect_fast, kernelName, globalThreads, localThreads, args, -1, -1);

    openCLSafeCall(clEnqueueReadBuffer(*(cl_command_queue*)clCxt->getOpenCLCommandQueuePtr(),
                                       counterCL, CL_TRUE, 0, sizeof(int), &counter, 0, NULL, NULL));
    openCLSafeCall(clReleaseMemObject(counterCL));

    return counter;
}

int cv::ocl::FAST_OCL::getKeyPoints(oclMat& keypoints)
{
    if (count_ == 0)
        return 0;

    if (nonmaxSupression)
    {
        ensureSizeIsEnough(ROWS_COUNT, count_, CV_32FC1, keypoints);
        return nonmaxSupressionOCL(keypoints);
    }

    kpLoc_.convertTo(keypoints, CV_32FC1);
    Mat k = keypoints;

    return count_;
}

void cv::ocl::FAST_OCL::release()
{
    kpLoc_.release();
    score_.release();

    d_keypoints_.release();
}
