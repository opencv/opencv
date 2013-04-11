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
//		Dachuan Zhao, dachuan@multicorewareinc.com
//		Yao Wang, bitwangyaoyao@gmail.com
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


#include "precomp.hpp"

using namespace std;
using namespace cv;
using namespace cv::ocl;

namespace cv
{
namespace ocl
{
///////////////////////////OpenCL kernel strings///////////////////////////
extern const char *pyrlk;
extern const char *pyrlk_no_image;
extern const char *arithm_mul;
}
}

struct dim3
{
    unsigned int x, y, z;
};

struct float2
{
    float x, y;
};

struct int2
{
    int x, y;
};

namespace
{
void calcPatchSize(cv::Size winSize, int cn, dim3 &block, dim3 &patch, bool isDeviceArch11)
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
}

static void multiply_cus(const oclMat &src1, oclMat &dst, float scalar)
{
    if(!src1.clCxt->supportsFeature(Context::CL_DOUBLE) && src1.type() == CV_64F)
    {
        CV_Error(CV_GpuNotSupported, "Selected device don't support double\r\n");
        return;
    }

    CV_Assert(src1.cols == dst.cols &&
              src1.rows == dst.rows);

    CV_Assert(src1.type() == dst.type());
    CV_Assert(src1.depth() != CV_8S);

    Context  *clCxt = src1.clCxt;

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { src1.cols,
                                src1.rows,
                                1
                              };

    int dst_step1 = dst.cols * dst.elemSize();
    vector<pair<size_t , const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.offset ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1.cols ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst_step1 ));
    args.push_back( make_pair( sizeof(float), (float *)&scalar ));

    openCLExecuteKernel(clCxt, &arithm_mul, "arithm_muls", globalThreads, localThreads, args, -1, src1.depth());
}

static void lkSparse_run(oclMat &I, oclMat &J,
                         const oclMat &prevPts, oclMat &nextPts, oclMat &status, oclMat& err, bool /*GET_MIN_EIGENVALS*/, int ptcount,
                         int level, dim3 patch, Size winSize, int iters)
{
    Context  *clCxt = I.clCxt;
    int elemCntPerRow = I.step / I.elemSize();
    string kernelName = "lkSparse";
    bool isImageSupported = support_image2d();
    size_t localThreads[3]  = { 8, isImageSupported ? 8 : 32, 1 };
    size_t globalThreads[3] = { 8 * ptcount, isImageSupported ? 8 : 32, 1};
    int cn = I.oclchannels();
    char calcErr;
    if (level == 0)
    {
        calcErr = 1;
    }
    else
    {
        calcErr = 0;
    }

    vector<pair<size_t , const void *> > args;

    cl_mem ITex = isImageSupported ? bindTexture(I) : (cl_mem)I.data;
    cl_mem JTex = isImageSupported ? bindTexture(J) : (cl_mem)J.data;

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
    if (!isImageSupported)
        args.push_back( make_pair( sizeof(cl_int), (void *)&elemCntPerRow ) );
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.x ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&patch.y ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&cn ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));

    bool is_cpu;
    queryDeviceInfo(IS_CPU_DEVICE, &is_cpu);
    if (is_cpu)
    {
        openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth(), (char*)" -D CPU");
        releaseTexture(ITex);
        releaseTexture(JTex);
    }
    else
    {
        if(isImageSupported)
        {
            openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth());
            releaseTexture(ITex);
            releaseTexture(JTex);
        }
        else
        {
            openCLExecuteKernel(clCxt, &pyrlk_no_image, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth());
        }
    }
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
    multiply_cus(temp1, temp2, 1.0f / (1 << maxLevel) / 2.0f);
    //::multiply(temp1, 1.0f / (1 << maxLevel) / 2.0f, temp2);

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
    bool isImageSupported = support_image2d();
    int elemCntPerRow = I.step / I.elemSize();

    string kernelName = "lkDense";

    size_t localThreads[3]  = { 16, 16, 1 };
    size_t globalThreads[3] = { I.cols, I.rows, 1};

    bool calcErr;
    if (err)
    {
        calcErr = true;
    }
    else
    {
        calcErr = false;
    }

    cl_mem ITex;
    cl_mem JTex;

    if (isImageSupported)
    {
        ITex = bindTexture(I);
        JTex = bindTexture(J);
    }
    else
    {
        ITex = (cl_mem)I.data;
        JTex = (cl_mem)J.data;
    }

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
    //args.push_back( make_pair( sizeof(cl_mem), (void *)&(*err).data ));
    //args.push_back( make_pair( sizeof(cl_int), (void *)&(*err).step ));
    if (!isImageSupported)
    {
        args.push_back( make_pair( sizeof(cl_int), (void *)&elemCntPerRow ) );
    }
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.width ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&winSize.height ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&iters ));
    args.push_back( make_pair( sizeof(cl_char), (void *)&calcErr ));

    if (isImageSupported)
    {
        openCLExecuteKernel(clCxt, &pyrlk, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth());

        releaseTexture(ITex);
        releaseTexture(JTex);
    }
    else
    {
        openCLExecuteKernel(clCxt, &pyrlk_no_image, kernelName, globalThreads, localThreads, args, I.oclchannels(), I.depth());
    }
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
