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
//    Peng Xiao, pengxiao@multicorewareinc.com
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

#include "precomp.hpp"
#include "opencl_kernels.hpp"

using namespace cv;
using namespace cv::ocl;

cv::ocl::CannyBuf::CannyBuf(const oclMat &dx_, const oclMat &dy_) : dx(dx_), dy(dy_), counter(1, 1, CV_32SC1)
{
    CV_Assert(dx_.type() == CV_32SC1 && dy_.type() == CV_32SC1 && dx_.size() == dy_.size());

    create(dx_.size(), -1);
}

void cv::ocl::CannyBuf::create(const Size &image_size, int apperture_size)
{
    ensureSizeIsEnough(image_size, CV_32SC1, dx);
    ensureSizeIsEnough(image_size, CV_32SC1, dy);

    if(apperture_size == 3)
    {
        ensureSizeIsEnough(image_size, CV_32SC1, dx_buf);
        ensureSizeIsEnough(image_size, CV_32SC1, dy_buf);
    }
    else if(apperture_size > 0)
    {
        Mat kx, ky;
        if (!filterDX)
        {
            filterDX = createDerivFilter_GPU(CV_8U, CV_32S, 1, 0, apperture_size, BORDER_REPLICATE);
        }
        if (!filterDY)
        {
            filterDY = createDerivFilter_GPU(CV_8U, CV_32S, 0, 1, apperture_size, BORDER_REPLICATE);
        }
    }
    ensureSizeIsEnough(2 * (image_size.height + 2), image_size.width + 2, CV_32FC1, edgeBuf);

    ensureSizeIsEnough(1, image_size.area(), CV_16UC2, trackBuf1);
    ensureSizeIsEnough(1, image_size.area(), CV_16UC2, trackBuf2);
}

void cv::ocl::CannyBuf::release()
{
    dx.release();
    dy.release();
    dx_buf.release();
    dy_buf.release();
    edgeBuf.release();
    trackBuf1.release();
    trackBuf2.release();
}

namespace cv
{
    namespace ocl
    {
        namespace canny
        {
            void calcSobelRowPass_gpu(const oclMat &src, oclMat &dx_buf, oclMat &dy_buf, int rows, int cols);

            void calcMagnitude_gpu(const oclMat &dx_buf, const oclMat &dy_buf, oclMat &dx, oclMat &dy, oclMat &mag, int rows, int cols, bool L2Grad);
            void calcMagnitude_gpu(const oclMat &dx, const oclMat &dy, oclMat &mag, int rows, int cols, bool L2Grad);

            void calcMap_gpu(oclMat &dx, oclMat &dy, oclMat &mag, oclMat &map, int rows, int cols, float low_thresh, float high_thresh);

            void edgesHysteresisLocal_gpu(oclMat &map, oclMat &st1, oclMat& counter, int rows, int cols);

            void edgesHysteresisGlobal_gpu(oclMat &map, oclMat &st1, oclMat &st2, oclMat& counter, int rows, int cols);

            void getEdges_gpu(oclMat &map, oclMat &dst, int rows, int cols);
        }
    }
}// cv::ocl

namespace
{
    void CannyCaller(CannyBuf &buf, oclMat &dst, float low_thresh, float high_thresh)
    {
        using namespace ::cv::ocl::canny;
        oclMat magBuf = buf.edgeBuf(Rect(0, 0, buf.edgeBuf.cols, buf.edgeBuf.rows / 2));
        oclMat mapBuf = buf.edgeBuf(Rect(0, buf.edgeBuf.rows / 2, buf.edgeBuf.cols, buf.edgeBuf.rows / 2));

        calcMap_gpu(buf.dx, buf.dy, magBuf, mapBuf, dst.rows, dst.cols, low_thresh, high_thresh);

        edgesHysteresisLocal_gpu(mapBuf, buf.trackBuf1, buf.counter, dst.rows, dst.cols);

        edgesHysteresisGlobal_gpu(mapBuf, buf.trackBuf1, buf.trackBuf2, buf.counter, dst.rows, dst.cols);

        getEdges_gpu(mapBuf, dst, dst.rows, dst.cols);
    }
}

void cv::ocl::Canny(const oclMat &src, oclMat &dst, double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
    CannyBuf buf(src.size(), apperture_size);
    Canny(src, buf, dst, low_thresh, high_thresh, apperture_size, L2gradient);
}

void cv::ocl::Canny(const oclMat &src, CannyBuf &buf, oclMat &dst, double low_thresh, double high_thresh, int apperture_size, bool L2gradient)
{
    using namespace ::cv::ocl::canny;

    CV_Assert(src.type() == CV_8UC1);

    if( low_thresh > high_thresh )
        std::swap( low_thresh, high_thresh );

    dst.create(src.size(), CV_8U);
    dst.setTo(Scalar::all(0));

    buf.create(src.size(), apperture_size);
    buf.edgeBuf.setTo(Scalar::all(0));

    oclMat magBuf = buf.edgeBuf(Rect(0, 0, buf.edgeBuf.cols, buf.edgeBuf.rows / 2));

    if (apperture_size == 3)
    {
        calcSobelRowPass_gpu(src, buf.dx_buf, buf.dy_buf, src.rows, src.cols);

        calcMagnitude_gpu(buf.dx_buf, buf.dy_buf, buf.dx, buf.dy, magBuf, src.rows, src.cols, L2gradient);
    }
    else
    {
        buf.filterDX->apply(src, buf.dx);
        buf.filterDY->apply(src, buf.dy);

        calcMagnitude_gpu(buf.dx, buf.dy, magBuf, src.rows, src.cols, L2gradient);
    }
    CannyCaller(buf, dst, static_cast<float>(low_thresh), static_cast<float>(high_thresh));
}
void cv::ocl::Canny(const oclMat &dx, const oclMat &dy, oclMat &dst, double low_thresh, double high_thresh, bool L2gradient)
{
    CannyBuf buf(dx, dy);
    Canny(dx, dy, buf, dst, low_thresh, high_thresh, L2gradient);
}

void cv::ocl::Canny(const oclMat &dx, const oclMat &dy, CannyBuf &buf, oclMat &dst, double low_thresh, double high_thresh, bool L2gradient)
{
    using namespace ::cv::ocl::canny;

    CV_Assert(dx.type() == CV_32SC1 && dy.type() == CV_32SC1 && dx.size() == dy.size());

    if( low_thresh > high_thresh )
        std::swap( low_thresh, high_thresh);

    dst.create(dx.size(), CV_8U);
    dst.setTo(Scalar::all(0));

    buf.dx = dx;
    buf.dy = dy;
    buf.create(dx.size(), -1);
    buf.edgeBuf.setTo(Scalar::all(0));

    oclMat magBuf = buf.edgeBuf(Rect(0, 0, buf.edgeBuf.cols, buf.edgeBuf.rows / 2));

    calcMagnitude_gpu(buf.dx, buf.dy, magBuf, dx.rows, dx.cols, L2gradient);

    CannyCaller(buf, dst, static_cast<float>(low_thresh), static_cast<float>(high_thresh));
}

void canny::calcSobelRowPass_gpu(const oclMat &src, oclMat &dx_buf, oclMat &dy_buf, int rows, int cols)
{
    Context *clCxt = src.clCxt;
    string kernelName = "calcSobelRowPass";
    vector< pair<size_t, const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&src.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dx_buf.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dy_buf.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx_buf.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx_buf.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy_buf.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy_buf.offset));

    size_t globalThreads[3] = {(size_t)cols, (size_t)rows, 1};
    size_t localThreads[3]  = {16, 16, 1};
    openCLExecuteKernel(clCxt, &imgproc_canny, kernelName, globalThreads, localThreads, args, -1, -1);
}

void canny::calcMagnitude_gpu(const oclMat &dx_buf, const oclMat &dy_buf, oclMat &dx, oclMat &dy, oclMat &mag, int rows, int cols, bool L2Grad)
{
    Context *clCxt = dx_buf.clCxt;
    string kernelName = "calcMagnitude_buf";
    vector< pair<size_t, const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&dx_buf.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dy_buf.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dx.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dy.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&mag.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx_buf.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx_buf.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy_buf.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy_buf.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&mag.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&mag.offset));

    size_t globalThreads[3] = {(size_t)cols, (size_t)rows, 1};
    size_t localThreads[3]  = {16, 16, 1};

    const char * build_options = L2Grad ? "-D L2GRAD":"";
    openCLExecuteKernel(clCxt, &imgproc_canny, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}
void canny::calcMagnitude_gpu(const oclMat &dx, const oclMat &dy, oclMat &mag, int rows, int cols, bool L2Grad)
{
    Context *clCxt = dx.clCxt;
    string kernelName = "calcMagnitude";
    vector< pair<size_t, const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&dx.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dy.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&mag.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&mag.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&mag.offset));

    size_t globalThreads[3] = {(size_t)cols, (size_t)rows, 1};
    size_t localThreads[3]  = {16, 16, 1};

    const char * build_options = L2Grad ? "-D L2GRAD":"";
    openCLExecuteKernel(clCxt, &imgproc_canny, kernelName, globalThreads, localThreads, args, -1, -1, build_options);
}

void canny::calcMap_gpu(oclMat &dx, oclMat &dy, oclMat &mag, oclMat &map, int rows, int cols, float low_thresh, float high_thresh)
{
    Context *clCxt = dx.clCxt;

    vector< pair<size_t, const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&dx.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dy.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&mag.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&map.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( make_pair( sizeof(cl_float), (void *)&low_thresh));
    args.push_back( make_pair( sizeof(cl_float), (void *)&high_thresh));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dx.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dy.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&mag.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&mag.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&map.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&map.offset));


    size_t globalThreads[3] = {(size_t)cols, (size_t)rows, 1};
    string kernelName = "calcMap";
    size_t localThreads[3]  = {16, 16, 1};

    openCLExecuteKernel(clCxt, &imgproc_canny, kernelName, globalThreads, localThreads, args, -1, -1);
}

void canny::edgesHysteresisLocal_gpu(oclMat &map, oclMat &st1, oclMat& counter, int rows, int cols)
{
    Context *clCxt = map.clCxt;
    vector< pair<size_t, const void *> > args;

    Mat counterMat(counter.rows, counter.cols, counter.type());
    counterMat.at<int>(0, 0) = 0;
    counter.upload(counterMat);

    args.push_back( make_pair( sizeof(cl_mem), (void *)&map.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&st1.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&counter.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
    cl_int stepBytes = map.step;
    args.push_back( make_pair( sizeof(cl_int), (void *)&stepBytes));
    cl_int offsetBytes = map.offset;
    args.push_back( make_pair( sizeof(cl_int), (void *)&offsetBytes));

    size_t globalThreads[3] = {(size_t)cols, (size_t)rows, 1};
    size_t localThreads[3]  = {16, 16, 1};

    openCLExecuteKernel(clCxt, &imgproc_canny, "edgesHysteresisLocal", globalThreads, localThreads, args, -1, -1);
}

void canny::edgesHysteresisGlobal_gpu(oclMat &map, oclMat &st1, oclMat &st2, oclMat& counter, int rows, int cols)
{
    Context *clCxt = map.clCxt;
    vector< pair<size_t, const void *> > args;
    size_t localThreads[3]  = {128, 1, 1};

    while(1 > 0)
    {
        Mat counterMat; counter.download(counterMat);
        int count = counterMat.at<int>(0, 0);
        CV_Assert(count >= 0);
        if (count == 0)
            break;

        counterMat.at<int>(0, 0) = 0;
        counter.upload(counterMat);

        args.clear();
        size_t globalThreads[3] = {(size_t)std::min((unsigned)count, 65535u) * 128, divUp(count, 65535), 1};
        args.push_back( make_pair( sizeof(cl_mem), (void *)&map.data));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&st1.data));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&st2.data));
        args.push_back( make_pair( sizeof(cl_mem), (void *)&counter.data));
        args.push_back( make_pair( sizeof(cl_int), (void *)&rows));
        args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
        args.push_back( make_pair( sizeof(cl_int), (void *)&count));
        args.push_back( make_pair( sizeof(cl_int), (void *)&map.step));
        args.push_back( make_pair( sizeof(cl_int), (void *)&map.offset));

        openCLExecuteKernel(clCxt, &imgproc_canny, "edgesHysteresisGlobal", globalThreads, localThreads, args, -1, -1);
        std::swap(st1, st2);
    }
}

void canny::getEdges_gpu(oclMat &map, oclMat &dst, int rows, int cols)
{
    Context *clCxt = map.clCxt;
    string kernelName = "getEdges";
    vector< pair<size_t, const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&map.data));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data));
    args.push_back( make_pair( sizeof(cl_int), (void *)&rows));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cols));
    args.push_back( make_pair( sizeof(cl_int), (void *)&map.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&map.offset));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.step));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.offset));

    size_t globalThreads[3] = {(size_t)cols, (size_t)rows, 1};
    size_t localThreads[3]  = {16, 16, 1};

    openCLExecuteKernel(clCxt, &imgproc_canny, kernelName, globalThreads, localThreads, args, -1, -1);
}
