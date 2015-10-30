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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Xiaopeng Fu, xiaopeng@multicorewareinc.com
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

namespace cv
{
namespace ocl
{
namespace stereoBM
{
/////////////////////////////////////////////////////////////////////////
//////////////////////////prefilter_xsbel////////////////////////////////
////////////////////////////////////////////////////////////////////////
static void prefilter_xsobel(const oclMat &input, oclMat &output, int prefilterCap)
{
    string kernelName = "prefilter_xsobel";

    size_t blockSize = 1;
    size_t globalThreads[3] = { (size_t)input.cols, (size_t)input.rows, 1 };
    size_t localThreads[3]  = { blockSize, blockSize, 1 };

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&input.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&output.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&input.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&input.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&prefilterCap));

    openCLExecuteKernel(Context::getContext(), &stereobm, kernelName,
        globalThreads, localThreads, args, -1, -1);
}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////common////////////////////////////////////
////////////////////////////////////////////////////////////////////////
#define N_DISPARITIES 8
#define ROWSperTHREAD 21
#define BLOCK_W 128

////////////////////////////////////////////////////////////////////////////
///////////////////////////////stereoBM_GPU////////////////////////////////
////////////////////////////////////////////////////////////////////////////
static void stereo_bm(const oclMat &left, const oclMat &right,  oclMat &disp,
               int maxdisp, int winSize,  oclMat &minSSD_buf)
{
    int winsz2 = winSize >> 1;

    string kernelName = "stereoKernel";

    disp.setTo(Scalar_<unsigned char>::all(0));
    minSSD_buf.setTo(Scalar_<unsigned int>::all(0xFFFFFFFF));

    size_t minssd_step = minSSD_buf.step / minSSD_buf.elemSize();
    size_t local_mem_size = (N_DISPARITIES * (BLOCK_W + 2 * winsz2)) *
                            sizeof(cl_uint);
    //size_t blockSize = 1;
    size_t localThreads[]  = { BLOCK_W, 1, 1 };
    size_t globalThreads[] = { (size_t)left.cols - maxdisp - 2 * winsz2,
                               divUp(left.rows - 2 * winsz2, ROWSperTHREAD),
                               1 };

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&left.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&right.data));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&minSSD_buf.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&minssd_step));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&disp.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&disp.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&left.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&left.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&left.step));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&maxdisp));
    args.push_back(std::make_pair(local_mem_size, (void *)NULL));

    char opt [128];
    sprintf(opt, "-D radius=%d", winsz2);
    openCLExecuteKernel(Context::getContext(), &stereobm, kernelName,
        globalThreads, localThreads, args, -1, -1, opt);
}
////////////////////////////////////////////////////////////////////////////
///////////////////////////////postfilter_textureness///////////////////////
////////////////////////////////////////////////////////////////////////////
static void postfilter_textureness(oclMat &left, int winSize,
                            float avergeTexThreshold, oclMat &disparity)
{
    string kernelName = "textureness_kernel";

    size_t blockSize = 1;
    size_t localThreads[]  = { BLOCK_W, blockSize ,1};
    size_t globalThreads[] = { (size_t)left.cols,
                               divUp(left.rows, 2 * ROWSperTHREAD),
                               1 };

    size_t local_mem_size = (localThreads[0] + localThreads[0] + (winSize / 2) * 2) * sizeof(float);

    std::vector< std::pair<size_t, const void *> > args;
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&disparity.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&disparity.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&disparity.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&disparity.step));
    args.push_back(std::make_pair(sizeof(cl_mem), (void *)&left.data));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&left.rows));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&left.cols));
    args.push_back(std::make_pair(sizeof(cl_int), (void *)&winSize));
    args.push_back(std::make_pair(sizeof(cl_float), (void *)&avergeTexThreshold));
    args.push_back(std::make_pair(local_mem_size, (void*)NULL));
    openCLExecuteKernel(Context::getContext(), &stereobm, kernelName,
        globalThreads, localThreads, args, -1, -1);
}
//////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////operator/////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
static void operator_(oclMat &minSSD, oclMat &leBuf, oclMat &riBuf, int preset, int ndisp,
               int winSize, float avergeTexThreshold, const oclMat &left,
               const oclMat &right, oclMat &disparity)

{
    CV_DbgAssert(left.rows == right.rows && left.cols == right.cols);
    CV_DbgAssert(left.type() == CV_8UC1);
    CV_DbgAssert(right.type() == CV_8UC1);

    disparity.create(left.size(), CV_8UC1);
    minSSD.create(left.size(), CV_32SC1);

    oclMat le_for_bm =  left;
    oclMat ri_for_bm = right;

    if (preset == cv::ocl::StereoBM_OCL::PREFILTER_XSOBEL)
    {
        leBuf.create( left.size(),  left.type());
        riBuf.create(right.size(), right.type());

        prefilter_xsobel( left, leBuf, 31);
        prefilter_xsobel(right, riBuf, 31);

        le_for_bm = leBuf;
        ri_for_bm = riBuf;
    }

    stereo_bm(le_for_bm, ri_for_bm, disparity, ndisp, winSize, minSSD);

    if (avergeTexThreshold)
    {
        postfilter_textureness(le_for_bm, winSize, avergeTexThreshold, disparity);
    }
}
}
}
}
const float defaultAvgTexThreshold = 3;

cv::ocl::StereoBM_OCL::StereoBM_OCL()
    : preset(BASIC_PRESET), ndisp(DEFAULT_NDISP), winSize(DEFAULT_WINSZ),
      avergeTexThreshold(defaultAvgTexThreshold)  {}

cv::ocl::StereoBM_OCL::StereoBM_OCL(int preset_, int ndisparities_, int winSize_)
    : preset(preset_), ndisp(ndisparities_), winSize(winSize_),
      avergeTexThreshold(defaultAvgTexThreshold)
{
    const int max_supported_ndisp = 1 << (sizeof(unsigned char) * 8);
    CV_Assert(0 < ndisp && ndisp <= max_supported_ndisp);
    CV_Assert(ndisp % 8 == 0);
    CV_Assert(winSize % 2 == 1);
}

bool cv::ocl::StereoBM_OCL::checkIfGpuCallReasonable()
{
    return true;
}

void cv::ocl::StereoBM_OCL::operator() ( const oclMat &left, const oclMat &right,
        oclMat &disparity)
{
    cv::ocl::stereoBM::operator_(minSSD, leBuf, riBuf, preset, ndisp, winSize, avergeTexThreshold, left, right, disparity);
}
