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

#include <iomanip>
#include "precomp.hpp"

namespace cv { namespace ocl {

// used for clAmdBlas library to avoid redundant setup/teardown
void clBlasSetup();
void clBlasTeardown();

}} /* namespace cv { namespace ocl */


#if !defined HAVE_CLAMDBLAS
void cv::ocl::gemm(const oclMat&, const oclMat&, double,
                   const oclMat&, double, oclMat&, int)
{
    CV_Error(Error::StsNotImplemented, "OpenCL BLAS is not implemented");
}

void cv::ocl::clBlasSetup()
{
    CV_Error(CV_StsNotImplemented, "OpenCL BLAS is not implemented");
}

void cv::ocl::clBlasTeardown()
{
    //intentionally do nothing
}

#else
#include "clAmdBlas.h"
using namespace cv;

static bool clBlasInitialized = false;
static Mutex cs;

void cv::ocl::clBlasSetup()
{
    if(!clBlasInitialized)
    {
        AutoLock al(cs);
        if(!clBlasInitialized)
        {
            openCLSafeCall(clAmdBlasSetup());
            clBlasInitialized = true;
        }
    }
}

void cv::ocl::clBlasTeardown()
{
    AutoLock al(cs);
    if(clBlasInitialized)
    {
        clAmdBlasTeardown();
        clBlasInitialized = false;
    }
}

void cv::ocl::gemm(const oclMat &src1, const oclMat &src2, double alpha,
                   const oclMat &src3, double beta, oclMat &dst, int flags)
{
    CV_Assert(src1.cols == src2.rows &&
              (src3.empty() || (src1.rows == src3.rows && src2.cols == src3.cols)));
    CV_Assert(!(cv::GEMM_3_T & flags)); // cv::GEMM_3_T is not supported
    if(!src3.empty())
    {
        src3.copyTo(dst);
    }
    else
    {
        dst.create(src1.rows, src2.cols, src1.type());
        dst.setTo(Scalar::all(0));
    }

    clBlasSetup();

    const clAmdBlasTranspose transA = (cv::GEMM_1_T & flags) ? clAmdBlasTrans : clAmdBlasNoTrans;
    const clAmdBlasTranspose transB = (cv::GEMM_2_T & flags) ? clAmdBlasTrans : clAmdBlasNoTrans;
    const clAmdBlasOrder     order  = clAmdBlasRowMajor;

    const int M = src1.rows;
    const int N = src2.cols;
    const int K = src1.cols;
    int lda     = src1.step;
    int ldb     = src2.step;
    int ldc     = dst.step;
    int offa    = src1.offset;
    int offb    = src2.offset;
    int offc    = dst.offset;

    cl_command_queue clq = (cl_command_queue)src1.clCxt->oclCommandQueue();
    switch(src1.type())
    {
    case CV_32FC1:
        lda  /= sizeof(float);
        ldb  /= sizeof(float);
        ldc  /= sizeof(float);
        offa /= sizeof(float);
        offb /= sizeof(float);
        offc /= sizeof(float);

        openCLSafeCall
        (
            clAmdBlasSgemmEx(order, transA, transB, M, N, K,
                             alpha, (const cl_mem)src1.data, offa, lda, (const cl_mem)src2.data, offb, ldb,
                             beta, (cl_mem)dst.data, offc, ldc, 1, &clq, 0, NULL, NULL)
        );
        break;
    case CV_64FC1:
        lda  /= sizeof(double);
        ldb  /= sizeof(double);
        ldc  /= sizeof(double);
        offa /= sizeof(double);
        offb /= sizeof(double);
        offc /= sizeof(double);
        openCLSafeCall
        (
            clAmdBlasDgemmEx(order, transA, transB, M, N, K,
                             alpha, (const cl_mem)src1.data, offa, lda, (const cl_mem)src2.data, offb, ldb,
                             beta, (cl_mem)dst.data, offc, ldc, 1, &clq, 0, NULL, NULL)
        );
        break;
    case CV_32FC2:
    {
        lda  /= (2*sizeof(float));
        ldb  /= (2*sizeof(float));
        ldc  /= (2*sizeof(float));
        offa /= (2*sizeof(float));
        offb /= (2*sizeof(float));
        offc /= (2*sizeof(float));
        cl_float2 alpha_2 = {{alpha, 0}};
        cl_float2 beta_2  = {{beta, 0}};
        openCLSafeCall
        (
            clAmdBlasCgemmEx(order, transA, transB, M, N, K,
                             alpha_2, (const cl_mem)src1.data, offa, lda, (const cl_mem)src2.data, offb, ldb,
                             beta_2, (cl_mem)dst.data, offc, ldc, 1, &clq, 0, NULL, NULL)
        );
    }
    break;
    case CV_64FC2:
    {
        lda  /= (2*sizeof(double));
        ldb  /= (2*sizeof(double));
        ldc  /= (2*sizeof(double));
        offa /= (2*sizeof(double));
        offb /= (2*sizeof(double));
        offc /= (2*sizeof(double));
        cl_double2 alpha_2 = {{alpha, 0}};
        cl_double2 beta_2  = {{beta, 0}};
        openCLSafeCall
        (
            clAmdBlasZgemmEx(order, transA, transB, M, N, K,
                             alpha_2, (const cl_mem)src1.data, offa, lda, (const cl_mem)src2.data, offb, ldb,
                             beta_2, (cl_mem)dst.data, offc, ldc, 1, &clq, 0, NULL, NULL)
        );
    }
    break;
    }
}
#endif
