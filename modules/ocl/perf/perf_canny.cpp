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

///////////// Canny ////////////////////////
PERFTEST(Canny)
{
    Mat img = imread(abspath("aloeL.jpg"), IMREAD_GRAYSCALE);

    if (img.empty())
    {
        throw runtime_error("can't open aloeL.jpg");
    }

    SUBTEST << img.cols << 'x' << img.rows << "; aloeL.jpg" << "; edges" << "; CV_8UC1";

    Mat edges(img.size(), CV_8UC1), ocl_edges;

    CPU_ON;
    Canny(img, edges, 50.0, 100.0);
    CPU_OFF;

    ocl::oclMat d_img(img);
    ocl::oclMat d_edges;
    ocl::CannyBuf d_buf;

    WARMUP_ON;
    ocl::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    WARMUP_OFF;

    GPU_ON;
    ocl::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    GPU_OFF;

    GPU_FULL_ON;
    d_img.upload(img);
    ocl::Canny(d_img, d_buf, d_edges, 50.0, 100.0);
    d_edges.download(ocl_edges);
    GPU_FULL_OFF;

    TestSystem::instance().ExceptedMatSimilar(edges, ocl_edges, 2e-2);
}