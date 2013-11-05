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

void cv::ocl::blendLinear(const oclMat &src1, const oclMat &src2, const oclMat &weights1, const oclMat &weights2,
                          oclMat &dst)
{
    CV_Assert(src1.depth() <= CV_32F);
    CV_Assert(src1.size() == src2.size() && src1.type() == src2.type());
    CV_Assert(weights1.size() == weights2.size() && weights1.size() == src1.size() &&
              weights1.type() == CV_32FC1 && weights2.type() == CV_32FC1);

    dst.create(src1.size(), src1.type());

    size_t globalSize[] = { dst.cols, dst.rows, 1};
    size_t localSize[] = { 16, 16, 1 };

    int depth = dst.depth(), ocn = dst.oclchannels();
    int src1_step = src1.step / src1.elemSize(), src1_offset = src1.offset / src1.elemSize();
    int src2_step = src2.step / src2.elemSize(), src2_offset = src2.offset / src2.elemSize();
    int weight1_step = weights1.step / weights1.elemSize(), weight1_offset = weights1.offset / weights1.elemSize();
    int weight2_step = weights2.step / weights2.elemSize(), weight2_offset = weights2.offset / weights2.elemSize();
    int dst_step = dst.step / dst.elemSize(), dst_offset = dst.offset / dst.elemSize();

    const char * const channelMap[] = { "", "", "2", "4", "4" };
    const char * const typeMap[] = { "uchar", "char", "ushort", "short", "int", "float", "double" };
    std::string buildOptions = format("-D T=%s%s -D convertToT=convert_%s%s%s -D FT=float%s -D convertToFT=convert_float%s",
                                      typeMap[depth], channelMap[ocn], typeMap[depth], channelMap[ocn],
                                      depth >= CV_32S ? "" : "_sat_rte", channelMap[ocn], channelMap[ocn]);

    vector< pair<size_t, const void *> > args;
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src1.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1_offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src1_step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&src2.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src2_offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&src2_step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&weights1.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&weight1_offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&weight1_step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&weights2.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&weight2_offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&weight2_step ));
    args.push_back( make_pair( sizeof(cl_mem), (void *)&dst.data ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst_offset ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst_step ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.rows ));
    args.push_back( make_pair( sizeof(cl_int), (void *)&dst.cols ));

    openCLExecuteKernel(src1.clCxt, &blend_linear, "blendLinear", globalSize, localSize, args,
                        -1, -1, buildOptions.c_str());
}
