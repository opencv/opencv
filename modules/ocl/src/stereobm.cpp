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
#include <vector>

using namespace cv;
using namespace cv::ocl;

namespace cv
{
namespace ocl
{

///////////////////////////OpenCL kernel strings///////////////////////////
extern const char *stereobm;

}
}
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
    Context *clCxt = input.clCxt;

    std::string kernelName = "prefilter_xsobel";
    cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobm, kernelName);

    size_t blockSize = 1;
    size_t globalThreads[3] = { input.cols, input.rows, 1 };
    size_t localThreads[3]  = { blockSize, blockSize, 1 };

    openCLVerifyKernel(clCxt, kernel,  localThreads);
    openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input.data));
    openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&output.data));
    openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&input.rows));
    openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&input.cols));
    openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_int), (void *)&prefilterCap));

    openCLSafeCall(clEnqueueNDRangeKernel((cl_command_queue)clCxt->oclCommandQueue(), kernel, 3, NULL,
                                          globalThreads, localThreads, 0, NULL, NULL));

    clFinish((cl_command_queue)clCxt->oclCommandQueue());
    openCLSafeCall(clReleaseKernel(kernel));

}
//////////////////////////////////////////////////////////////////////////
//////////////////////////////common////////////////////////////////////
////////////////////////////////////////////////////////////////////////
#define N_DISPARITIES 8
#define ROWSperTHREAD 21
#define BLOCK_W 128
static inline int divUp(int total, int grain)
{
    return (total + grain - 1) / grain;
}
////////////////////////////////////////////////////////////////////////////
///////////////////////////////stereoBM_GPU////////////////////////////////
////////////////////////////////////////////////////////////////////////////
static void stereo_bm(const oclMat &left, const oclMat &right,  oclMat &disp,
               int maxdisp, int winSize,  oclMat &minSSD_buf)
{
    int winsz2 = winSize >> 1;

    //if(winsz2 == 0 || winsz2 >= calles_num)
    //cv::ocl:error("Unsupported window size", __FILE__, __LINE__, __FUNCTION__);

    Context *clCxt = left.clCxt;

    std::string kernelName = "stereoKernel";
    cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobm, kernelName);

    disp.setTo(Scalar_<unsigned char>::all(0));
    minSSD_buf.setTo(Scalar_<unsigned int>::all(0xFFFFFFFF));

    size_t minssd_step = minSSD_buf.step / minSSD_buf.elemSize();
    size_t local_mem_size = (BLOCK_W + N_DISPARITIES * (BLOCK_W + 2 * winsz2)) *
                            sizeof(cl_uint);
    //size_t blockSize = 1;
    size_t localThreads[]  = { BLOCK_W, 1,1};
    size_t globalThreads[] = { divUp(left.cols - maxdisp - 2 * winsz2, BLOCK_W) *BLOCK_W,
                               divUp(left.rows - 2 * winsz2, ROWSperTHREAD),
                               1
                             };

    openCLVerifyKernel(clCxt, kernel, localThreads);
    openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&left.data));
    openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&right.data));
    openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&minSSD_buf.data));
    openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&minssd_step));
    openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&disp.data));
    openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&disp.step));
    openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&left.cols));
    openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&left.rows));
    openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_int), (void *)&left.step));
    openCLSafeCall(clSetKernelArg(kernel, 9, sizeof(cl_int), (void *)&maxdisp));
    openCLSafeCall(clSetKernelArg(kernel, 10, sizeof(cl_int), (void *)&winsz2));
    openCLSafeCall(clSetKernelArg(kernel, 11, local_mem_size, (void *)NULL));

    openCLSafeCall(clEnqueueNDRangeKernel((cl_command_queue)clCxt->oclCommandQueue(), kernel, 2, NULL,
                                          globalThreads, localThreads, 0, NULL, NULL));


    clFinish((cl_command_queue)clCxt->oclCommandQueue());
    openCLSafeCall(clReleaseKernel(kernel));
}
////////////////////////////////////////////////////////////////////////////
///////////////////////////////postfilter_textureness///////////////////////
////////////////////////////////////////////////////////////////////////////
static void postfilter_textureness(oclMat &left, int winSize,
                            float avergeTexThreshold, oclMat &disparity)
{
    Context *clCxt = left.clCxt;

    std::string kernelName = "textureness_kernel";
    cl_kernel kernel = openCLGetKernelFromSource(clCxt, &stereobm, kernelName);

    size_t blockSize = 1;
    size_t localThreads[]  = { BLOCK_W, blockSize ,1};
    size_t globalThreads[] = { divUp(left.cols, BLOCK_W) *BLOCK_W,
                               divUp(left.rows, 2 * ROWSperTHREAD),
                               1
                             };

    size_t local_mem_size = (localThreads[0] + localThreads[0] + (winSize / 2) * 2) * sizeof(float);

    openCLVerifyKernel(clCxt, kernel,  localThreads);
    openCLSafeCall(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&disparity.data));
    openCLSafeCall(clSetKernelArg(kernel, 1, sizeof(cl_int), (void *)&disparity.rows));
    openCLSafeCall(clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&disparity.cols));
    openCLSafeCall(clSetKernelArg(kernel, 3, sizeof(cl_int), (void *)&disparity.step));
    openCLSafeCall(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&left.data));
    openCLSafeCall(clSetKernelArg(kernel, 5, sizeof(cl_int), (void *)&left.rows));
    openCLSafeCall(clSetKernelArg(kernel, 6, sizeof(cl_int), (void *)&left.cols));
    openCLSafeCall(clSetKernelArg(kernel, 7, sizeof(cl_int), (void *)&winSize));
    openCLSafeCall(clSetKernelArg(kernel, 8, sizeof(cl_float), (void *)&avergeTexThreshold));
    openCLSafeCall(clSetKernelArg(kernel, 9, local_mem_size, NULL));
    openCLSafeCall(clEnqueueNDRangeKernel((cl_command_queue)clCxt->oclCommandQueue(), kernel, 2, NULL,
                                          globalThreads, localThreads, 0, NULL, NULL));

    clFinish((cl_command_queue)clCxt->oclCommandQueue());
    openCLSafeCall(clReleaseKernel(kernel));
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
