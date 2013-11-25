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
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Jia Haipeng, jiahaipeng95@gmail.com
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
        namespace split_merge
        {
            static void merge_vector_run(const oclMat *mat_src, size_t n, oclMat &mat_dst)
            {
                if(!mat_dst.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && mat_dst.type() == CV_64F)
                {
                    CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
                    return;
                }

                Context  *clCxt = mat_dst.clCxt;
                int channels = mat_dst.oclchannels();
                int depth = mat_dst.depth();

                String kernelName = "merge_vector";

                int vector_lengths[4][7] = {{0, 0, 0, 0, 0, 0, 0},
                    {2, 2, 1, 1, 1, 1, 1},
                    {4, 4, 2, 2 , 1, 1, 1},
                    {1, 1, 1, 1, 1, 1, 1}
                };

                size_t vector_length = vector_lengths[channels - 1][depth];
                int offset_cols = (mat_dst.offset / mat_dst.elemSize()) & (vector_length - 1);
                int cols = divUp(mat_dst.cols + offset_cols, vector_length);

                size_t localThreads[3]  = { 64, 4, 1 };
                size_t globalThreads[3] = { cols, mat_dst.rows, 1 };

                int dst_step1 = mat_dst.cols * mat_dst.elemSize();
                std::vector<std::pair<size_t , const void *> > args;
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_dst.data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst.step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst.offset));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src[0].data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[0].step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[0].offset));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src[1].data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[1].step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[1].offset));

                if(channels == 4)
                {
                    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src[2].data));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[2].step));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[2].offset));

                    if(n == 3)
                    {
                        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src[2].data));
                        args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[2].step));
                        args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[2].offset));
                    }
                    else if( n == 4)
                    {
                        args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src[3].data));
                        args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[3].step));
                        args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src[3].offset));
                    }
                }

                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst.rows));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step1));

                openCLExecuteKernel(clCxt, &merge_mat, kernelName, globalThreads, localThreads, args, channels, depth);
            }
            static void merge(const oclMat *mat_src, size_t n, oclMat &mat_dst)
            {
                CV_Assert(mat_src);
                CV_Assert(n > 0);

                int depth = mat_src[0].depth();
                Size size = mat_src[0].size();

                int total_channels = 0;

                for(size_t i = 0; i < n; ++i)
                {
                    CV_Assert(depth == mat_src[i].depth());
                    CV_Assert(size == mat_src[i].size());

                    total_channels += mat_src[i].oclchannels();
                }

                CV_Assert(total_channels <= 4);

                if(total_channels == 1)
                {
                    mat_src[0].copyTo(mat_dst);
                    return;
                }

                mat_dst.create(size, CV_MAKETYPE(depth, total_channels));
                merge_vector_run(mat_src, n, mat_dst);
            }
            static void split_vector_run(const oclMat &src, oclMat *dst)
            {

                if(!src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && src.type() == CV_64F)
                {
                    CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
                    return;
                }

                Context  *clCtx = src.clCxt;
                int channels = src.channels();
                int depth = src.depth();
                depth = (depth == CV_8S) ? CV_8U : depth;
                depth = (depth == CV_16S) ? CV_16U : depth;

                String kernelName = "split_vector";

                size_t VEC_SIZE = 4;

                std::vector<std::pair<size_t , const void *> > args;
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&src.data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&src.step));
                int srcOffsetXBytes = src.offset % src.step;
                int srcOffsetY = src.offset / src.step;
                cl_int2 srcOffset = {{srcOffsetXBytes, srcOffsetY}};
                args.push_back( std::make_pair( sizeof(cl_int2), (void *)&srcOffset));

                bool dst0Aligned = false, dst1Aligned = false, dst2Aligned = false, dst3Aligned = false;
                int alignSize = dst[0].elemSize1() * VEC_SIZE;
                int alignMask = alignSize - 1;

                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst[0].data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst[0].step));
                int dst0OffsetXBytes = dst[0].offset % dst[0].step;
                int dst0OffsetY = dst[0].offset / dst[0].step;
                cl_int2 dst0Offset = {{dst0OffsetXBytes, dst0OffsetY}};
                args.push_back( std::make_pair( sizeof(cl_int2), (void *)&dst0Offset));
                if ((dst0OffsetXBytes & alignMask) == 0)
                    dst0Aligned = true;

                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst[1].data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst[1].step));
                int dst1OffsetXBytes = dst[1].offset % dst[1].step;
                int dst1OffsetY = dst[1].offset / dst[1].step;
                cl_int2 dst1Offset = {{dst1OffsetXBytes, dst1OffsetY}};
                args.push_back( std::make_pair( sizeof(cl_int2), (void *)&dst1Offset));
                if ((dst1OffsetXBytes & alignMask) == 0)
                    dst1Aligned = true;

                // DON'T MOVE VARIABLES INTO 'IF' BODY
                int dst2OffsetXBytes, dst2OffsetY;
                cl_int2 dst2Offset;
                int dst3OffsetXBytes, dst3OffsetY;
                cl_int2 dst3Offset;
                if (channels >= 3)
                {
                    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst[2].data));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst[2].step));
                    dst2OffsetXBytes = dst[2].offset % dst[2].step;
                    dst2OffsetY = dst[2].offset / dst[2].step;
                    dst2Offset.s[0] = dst2OffsetXBytes; dst2Offset.s[1] = dst2OffsetY;
                    args.push_back( std::make_pair( sizeof(cl_int2), (void *)&dst2Offset));
                    if ((dst2OffsetXBytes & alignMask) == 0)
                        dst2Aligned = true;
                }

                if (channels >= 4)
                {
                    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&dst[3].data));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst[3].step));
                    dst3OffsetXBytes = dst[3].offset % dst[3].step;
                    dst3OffsetY = dst[3].offset / dst[3].step;
                    dst3Offset.s[0] = dst3OffsetXBytes; dst3Offset.s[1] = dst3OffsetY;
                    args.push_back( std::make_pair( sizeof(cl_int2), (void *)&dst3Offset));
                    if ((dst3OffsetXBytes & alignMask) == 0)
                        dst3Aligned = true;
                }

                cl_int2 size = {{ src.cols, src.rows }};
                args.push_back( std::make_pair( sizeof(cl_int2), (void *)&size));

                String build_options =
                        cv::format("-D VEC_SIZE=%d -D DATA_DEPTH=%d -D DATA_CHAN=%d",
                                   (int)VEC_SIZE, depth, channels);

                if (dst0Aligned)
                    build_options += " -D DST0_ALIGNED";
                if (dst1Aligned)
                    build_options += " -D DST1_ALIGNED";
                if (dst2Aligned)
                    build_options += " -D DST2_ALIGNED";
                if (dst3Aligned)
                    build_options += " -D DST3_ALIGNED";

                const DeviceInfo& devInfo = clCtx->getDeviceInfo();

                // TODO Workaround for issues. Need to investigate a problem.
                if (channels == 2
                        && devInfo.deviceType == CVCL_DEVICE_TYPE_CPU
                        && devInfo.platform->platformVendor.find("Intel") != std::string::npos
                        && (devInfo.deviceVersion.find("Build 56860") != std::string::npos
                            || devInfo.deviceVersion.find("Build 76921") != std::string::npos
                            || devInfo.deviceVersion.find("Build 78712") != std::string::npos))
                    build_options += " -D BYPASS_VSTORE=true";

                size_t globalThreads[3] = { divUp(src.cols, VEC_SIZE), src.rows, 1 };
                openCLExecuteKernel(clCtx, &split_mat, kernelName, globalThreads, NULL, args, -1, -1, build_options.c_str());
            }
            static void split(const oclMat &mat_src, oclMat *mat_dst)
            {
                CV_Assert(mat_dst);

                int depth = mat_src.depth();
                int num_channels = mat_src.channels();
                Size size = mat_src.size();

                if (num_channels == 1)
                {
                    mat_src.copyTo(mat_dst[0]);
                    return;
                }

                for (int i = 0; i < mat_src.oclchannels(); i++)
                    mat_dst[i].create(size, CV_MAKETYPE(depth, 1));

                split_vector_run(mat_src, mat_dst);
            }
        }
    }
}

void cv::ocl::merge(const oclMat *src, size_t n, oclMat &dst)
{
    split_merge::merge(src, n, dst);
}
void cv::ocl::merge(const std::vector<oclMat> &src, oclMat &dst)
{
    split_merge::merge(&src[0], src.size(), dst);
}

void cv::ocl::split(const oclMat &src, oclMat *dst)
{
    split_merge::split(src, dst);
}
void cv::ocl::split(const oclMat &src, std::vector<oclMat> &dst)
{
    dst.resize(src.oclchannels()); // TODO Why oclchannels?
    if(src.oclchannels() > 0)
        split_merge::split(src, &dst[0]);
}
