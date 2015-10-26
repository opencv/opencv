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
//      Dachuan Zhao, dachuan@multicorewareinc.com
//      Yao Wang, bitwangyaoyao@gmail.com
//      Nathan, liujun@multicorewareinc.com
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

struct dim3
{
    unsigned int x, y, z;
};

static void calcPatchSize(cv::Size winSize, int cn, dim3 &block, dim3 &patch, bool isDeviceArch11)
{
    winSize.width *= cn;

    if (winSize.width > 32 && winSize.width > 2 * winSize.height)
    {
        block.x = isDeviceArch11 ? 16 : 32;
        block.y = 8;
    }
    else
    {
        block.x = 16;
        block.y = isDeviceArch11 ? 8 : 16;
    }

    patch.x = (winSize.width  + block.x - 1) / block.x;
    patch.y = (winSize.height + block.y - 1) / block.y;

    block.z = patch.z = 1;
}

static void lkSparse_run(oclMat &I, oclMat &J,
                         const oclMat &prevPts, oclMat &nextPts, oclMat &status, oclMat& err, bool /*GET_MIN_EIGENVALS*/, int ptcount,
                         int level, dim3 patch, Size winSize, int iters)
{
    Context  *clCxt = I.clCxt;
    string kernelName = "lkSparse";
    size_t localThreads[3]  = { 8, 8, 1 };
    size_t globalThreads[3] = { 8 * (size_t)ptcount, 8, 1};
    int cn = I.oclchannels();
    char calcErr = level==0?1:0;

    vector<pair<size_t , const void *> > args;

    cl_mem ITex = bindTexture(I);
    cl_mem JTex = bindTexture(J);

    args.push_back( make_pair( sizeof(cl_mem), (void *)&ITex ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&JTex ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&prevPts.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&prevPts.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&nextPts.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&nextPts.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&status.data ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&err.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&level ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.cols ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.x ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.y ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cn ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));

    bool is_cpu = isCpuDevice();
    if (is_cpu)
    {
        openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth(), (char*)" -D CPU");
    }
    else
    {
        stringstream idxStr;
        idxStr << kernelName << "_C" << I.oclchannels() << "_D" << I.depth();
        cl_kernel kernel = openCLGetKernelFromSource(clCxt, &pyrlk, idxStr.str());
        int wave_size = (int)queryWaveFrontSize(kernel);
        openCLSafeCall(clReleaseKernel(kernel));

        static char opt[32] = {0};
        sprintf(opt, "-D WAVE_SIZE=%d", wave_size);

        openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads,
                            args, I.oclchannels(), I.depth(), opt);
    }
    releaseTexture(ITex);
    releaseTexture(JTex);
}

void cv::ocl::PyrLKOpticalFlow::sparse(const oclMat &prevImg, const oclMat &nextImg, const oclMat &prevPts, oclMat &nextPts, oclMat &status, oclMat *err)
{
    if (prevPts.empty())
    {
        nextPts.release();
        status.release();
        if (err) err->release();
        return;
    }

    derivLambda = std::min(std::max(derivLambda, 0.0), 1.0);

    iters = std::min(std::max(iters, 0), 100);

    const int cn = prevImg.oclchannels();

    dim3 block, patch;
    calcPatchSize(winSize, cn, block, patch, isDeviceArch11_);

    CV_Assert(derivLambda >= 0);
    CV_Assert(maxLevel >= 0 && winSize.width > 2 && winSize.height > 2);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    CV_Assert(patch.x > 0 && patch.x < 6 && patch.y > 0 && patch.y < 6);
    CV_Assert(prevPts.rows == 1 && prevPts.type() == CV_32FC2);

    if (useInitialFlow)
        CV_Assert(nextPts.size() == prevPts.size() && nextPts.type() == CV_32FC2);
    else
        ensureSizeIsEnough(1, prevPts.cols, prevPts.type(), nextPts);

    oclMat temp1 = (useInitialFlow ? nextPts : prevPts).reshape(1);
    oclMat temp2 = nextPts.reshape(1);
    multiply(1.0f/(1<<maxLevel)/2.0f, temp1, temp2);

    ensureSizeIsEnough(1, prevPts.cols, CV_8UC1, status);
    status.setTo(Scalar::all(1));

    bool errMat = false;
    if (!err)
    {
        err = new oclMat(1, prevPts.cols, CV_32FC1);
        errMat = true;
    }
    else
        ensureSizeIsEnough(1, prevPts.cols, CV_32FC1, *err);

    // build the image pyramids.
    prevPyr_.resize(maxLevel + 1);
    nextPyr_.resize(maxLevel + 1);

    if (cn == 1 || cn == 4)
    {
        prevImg.convertTo(prevPyr_[0], CV_32F);
        nextImg.convertTo(nextPyr_[0], CV_32F);
    }

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown(nextPyr_[level - 1], nextPyr_[level]);
    }

    // dI/dx ~ Ix, dI/dy ~ Iy
    for (int level = maxLevel; level >= 0; level--)
    {
        lkSparse_run(prevPyr_[level], nextPyr_[level],
                     prevPts, nextPts, status, *err, getMinEigenVals, prevPts.cols,
                     level, patch, winSize, iters);
    }

    if(errMat)
        delete err;
}

static void lkDense_run(oclMat &I, oclMat &J, oclMat &u, oclMat &v,
                        oclMat &prevU, oclMat &prevV, oclMat *err, Size winSize, int iters)
{
    Context  *clCxt = I.clCxt;

    string kernelName = "lkDense";

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { (size_t)I.cols, (size_t)I.rows, 1};

    cl_char calcErr = err ? 1 : 0;

    cl_mem ITex;
    cl_mem JTex;

    ITex = bindTexture(I);
    JTex = bindTexture(J);

    vector<pair<size_t , const void *> > args;

    args.push_back( make_pair( sizeof(cl_mem), (void *)&ITex ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&JTex ));

    args.push_back( make_pair( sizeof(cl_mem), (void *)&u.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&u.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&v.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&v.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&prevU.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&prevU.step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&prevV.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&prevV.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&I.cols ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));

    openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth());

    releaseTexture(ITex);
    releaseTexture(JTex);
}

void cv::ocl::PyrLKOpticalFlow::dense(const oclMat &prevImg, const oclMat &nextImg, oclMat &u, oclMat &v, oclMat *err)
{
    CV_Assert(prevImg.type() == CV_8UC1);
    CV_Assert(prevImg.size() == nextImg.size() && prevImg.type() == nextImg.type());
    CV_Assert(maxLevel >= 0);
    CV_Assert(winSize.width > 2 && winSize.height > 2);

    if (err)
        err->create(prevImg.size(), CV_32FC1);

    prevPyr_.resize(maxLevel + 1);
    nextPyr_.resize(maxLevel + 1);

    prevPyr_[0] = prevImg;
    nextImg.convertTo(nextPyr_[0], CV_32F);

    for (int level = 1; level <= maxLevel; ++level)
    {
        pyrDown(prevPyr_[level - 1], prevPyr_[level]);
        pyrDown(nextPyr_[level - 1], nextPyr_[level]);
    }

    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[0]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, uPyr_[1]);
    ensureSizeIsEnough(prevImg.size(), CV_32FC1, vPyr_[1]);
    uPyr_[1].setTo(Scalar::all(0));
    vPyr_[1].setTo(Scalar::all(0));

    Size winSize2i(winSize.width, winSize.height);

    int idx = 0;

    for (int level = maxLevel; level >= 0; level--)
    {
        int idx2 = (idx + 1) & 1;

        lkDense_run(prevPyr_[level], nextPyr_[level], uPyr_[idx], vPyr_[idx], uPyr_[idx2], vPyr_[idx2],
                    level == 0 ? err : 0, winSize2i, iters);

        if (level > 0)
            idx = idx2;
    }

    uPyr_[idx].copyTo(u);
    vPyr_[idx].copyTo(v);
}
