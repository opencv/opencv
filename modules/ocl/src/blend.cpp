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
//    Nathan, liujun@multicorewareinc.com
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
#include <iomanip>

using namespace cv;
using namespace cv::ocl;

namespace cv
{
    namespace ocl
    {
        ////////////////////////////////////OpenCL kernel strings//////////////////////////
        extern const char *blend_linear;
    }
}

void cv::ocl::blendLinear(const oclMat &img1, const oclMat &img2, const oclMat &weights1, const oclMat &weights2,
                          oclMat &result)
{
    cv::ocl::Context *ctx = img1.clCxt;
    CV_Assert(ctx == img2.clCxt && ctx == weights1.clCxt && ctx == weights2.clCxt);
    int channels = img1.oclchannels();
    int depth = img1.depth();
    int rows = img1.rows;
    int cols = img1.cols;
    int istep = img1.step1();
    int wstep = weights1.step1();
    size_t globalSize[] = {cols * channels / 4, rows, 1};
    size_t localSize[] = {256, 1, 1};

    std::vector< std::pair<size_t, const void *> > args;
    result.create(img1.size(), CV_MAKE_TYPE(depth,img1.channels()));
    if(globalSize[0] != 0)
    {
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&result.data ));
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&img1.data ));
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&img2.data ));
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&weights1.data ));
        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&weights2.data ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&rows ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&istep ));
        args.push_back( std::make_pair( sizeof(cl_int), (void *)&wstep ));
        String kernelName = "BlendLinear";

        openCLExecuteKernel(ctx, &blend_linear, kernelName, globalSize, localSize, args, channels, depth);
    }
}
