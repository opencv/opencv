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
//		Zhang Chunpeng chunpeng@multicorewareinc.com
//		Yao Wang, yao@multicorewareinc.com
//
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

/* Haar features calculation */
//#define EMU

#include "precomp.hpp"

using namespace cv;
using namespace cv::ocl;

namespace cv
{
    namespace ocl
    {
        extern const char *pyr_up;
        void pyrUp(const cv::ocl::oclMat &src, cv::ocl::oclMat &dst)
        {
            dst.create(src.rows * 2, src.cols * 2, src.type());

            Context *clCxt = src.clCxt;

            const String kernelName = "pyrUp";

            std::vector< std::pair<size_t, const void *> > args;
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
            args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst.data));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.rows));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.rows));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.cols));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.cols));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.offset));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.offset));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.step));
            args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst.step));

            size_t globalThreads[3] = {dst.cols, dst.rows, 1};
            size_t localThreads[3]  = {16, 16, 1};


            openCLExecuteKernel(clCxt, &pyr_up, kernelName, globalThreads, localThreads, args, src.oclchannels(), src.depth());
        }
    }
}
