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
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
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
#include "perf_precomp.hpp"

///////////// gemm ////////////////////////
PERFTEST(gemm)
{
    Mat src1, src2, src3, dst, ocl_dst;
    ocl::oclMat d_src1, d_src2, d_src3, d_dst;

    for (int size = Min_Size; size <= Max_Size; size *= Multiple)
    {
        SUBTEST << size << 'x' << size;

        gen(src1, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src2, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));
        gen(src3, size, size, CV_32FC1, Scalar::all(-10), Scalar::all(10));

        gemm(src1, src2, 1.0, src3, 1.0, dst);

        CPU_ON;
        gemm(src1, src2, 1.0, src3, 1.0, dst);
        CPU_OFF;

        d_src1.upload(src1);
        d_src2.upload(src2);
        d_src3.upload(src3);

        WARMUP_ON;
        ocl::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        WARMUP_OFF;

        GPU_ON;
        ocl::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        GPU_OFF;

        GPU_FULL_ON;
        d_src1.upload(src1);
        d_src2.upload(src2);
        d_src3.upload(src3);
        ocl::gemm(d_src1, d_src2, 1.0, d_src3, 1.0, d_dst);
        d_dst.download(ocl_dst);
        GPU_FULL_OFF;

        TestSystem::instance().ExpectedMatNear(ocl_dst, dst, src1.cols * src1.rows * 1e-4);
    }
}