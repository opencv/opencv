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

#include "mcwutil.hpp"

#if defined (HAVE_OPENCL)

using namespace std;



namespace cv
{
    namespace ocl
    {

        inline int divUp(int total, int grain)
        {
            return (total + grain - 1) / grain;
        }

        // provide additional methods for the user to interact with the command queue after a task is fired
        static void openCLExecuteKernel_2(Context *clCxt , const char **source, string kernelName, size_t globalThreads[3],
                                   size_t localThreads[3],  vector< pair<size_t, const void *> > &args, int channels,
                                   int depth, char *build_options, FLUSH_MODE finish_mode)
        {
            //construct kernel name
            //The rule is functionName_Cn_Dn, C represent Channels, D Represent DataType Depth, n represent an integer number
            //for exmaple split_C2_D2, represent the split kernel with channels =2 and dataType Depth = 2(Data type is char)
            stringstream idxStr;
            if(channels != -1)
                idxStr << "_C" << channels;
            if(depth != -1)
                idxStr << "_D" << depth;
            kernelName += idxStr.str();

            cl_kernel kernel;
            kernel = openCLGetKernelFromSource(clCxt, source, kernelName, build_options);

            if ( localThreads != NULL)
            {
                globalThreads[0] = divUp(globalThreads[0], localThreads[0]) * localThreads[0];
                globalThreads[1] = divUp(globalThreads[1], localThreads[1]) * localThreads[1];
                globalThreads[2] = divUp(globalThreads[2], localThreads[2]) * localThreads[2];

                //size_t blockSize = localThreads[0] * localThreads[1] * localThreads[2];
                cv::ocl::openCLVerifyKernel(clCxt, kernel,  localThreads);
            }
            for(size_t i = 0; i < args.size(); i ++)
                openCLSafeCall(clSetKernelArg(kernel, i, args[i].first, args[i].second));

            openCLSafeCall(clEnqueueNDRangeKernel(clCxt->impl->clCmdQueue, kernel, 3, NULL, globalThreads,
                                                  localThreads, 0, NULL, NULL));

            switch(finish_mode)
            {
            case CLFINISH:
                clFinish(clCxt->impl->clCmdQueue);
            case CLFLUSH:
                clFlush(clCxt->impl->clCmdQueue);
                break;
            case DISABLE:
            default:
                break;
            }
            openCLSafeCall(clReleaseKernel(kernel));
        }

        void openCLExecuteKernel2(Context *clCxt , const char **source, string kernelName,
                                  size_t globalThreads[3], size_t localThreads[3],
                                  vector< pair<size_t, const void *> > &args, int channels, int depth, FLUSH_MODE finish_mode)
        {
            openCLExecuteKernel2(clCxt, source, kernelName, globalThreads, localThreads, args,
                                 channels, depth, NULL, finish_mode);
        }
        void openCLExecuteKernel2(Context *clCxt , const char **source, string kernelName,
                                  size_t globalThreads[3], size_t localThreads[3],
                                  vector< pair<size_t, const void *> > &args, int channels, int depth, char *build_options, FLUSH_MODE finish_mode)

        {
            openCLExecuteKernel_2(clCxt, source, kernelName, globalThreads, localThreads, args, channels, depth,
                                  build_options, finish_mode);
        }
    }//namespace ocl

}//namespace cv
#endif