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
            static void split_vector_run(const oclMat &mat_src, oclMat *mat_dst)
            {

                if(!mat_src.clCxt->supportsFeature(FEATURE_CL_DOUBLE) && mat_src.type() == CV_64F)
                {
                    CV_Error(Error::OpenCLDoubleNotSupported, "Selected device doesn't support double");
                    return;
                }

                Context  *clCxt = mat_src.clCxt;
                int channels = mat_src.oclchannels();
                int depth = mat_src.depth();

                String kernelName = "split_vector";

                int vector_lengths[4][7] = {{0, 0, 0, 0, 0, 0, 0},
                    {4, 4, 2, 2, 1, 1, 1},
                    {4, 4, 2, 2 , 1, 1, 1},
                    {4, 4, 2, 2, 1, 1, 1}
                };

                size_t vector_length = vector_lengths[channels - 1][mat_dst[0].depth()];

                int max_offset_cols = 0;
                for(int i = 0; i < channels; i++)
                {
                    int offset_cols = (mat_dst[i].offset / mat_dst[i].elemSize()) & (vector_length - 1);
                    if(max_offset_cols < offset_cols)
                        max_offset_cols = offset_cols;
                }

                int cols =  vector_length == 1 ? divUp(mat_src.cols, vector_length)
                            : divUp(mat_src.cols + max_offset_cols, vector_length);

                size_t localThreads[3]  = { 64, 4, 1 };
                size_t globalThreads[3] = { cols, mat_src.rows, 1 };

                int dst_step1 = mat_dst[0].cols * mat_dst[0].elemSize();
                std::vector<std::pair<size_t , const void *> > args;
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_src.data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src.step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src.offset));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_dst[0].data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[0].step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[0].offset));
                args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_dst[1].data));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[1].step));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[1].offset));
                if(channels >= 3)
                {

                    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_dst[2].data));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[2].step));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[2].offset));
                }
                if(channels >= 4)
                {
                    args.push_back( std::make_pair( sizeof(cl_mem), (void *)&mat_dst[3].data));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[3].step));
                    args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_dst[3].offset));
                }

                args.push_back( std::make_pair( sizeof(cl_int), (void *)&mat_src.rows));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&cols));
                args.push_back( std::make_pair( sizeof(cl_int), (void *)&dst_step1));

                openCLExecuteKernel(clCxt, &split_mat, kernelName, globalThreads, localThreads, args, channels, depth);
            }
            static void split(const oclMat &mat_src, oclMat *mat_dst)
            {
                CV_Assert(mat_dst);

                int depth = mat_src.depth();
                int num_channels = mat_src.oclchannels();
                Size size = mat_src.size();

                if(num_channels == 1)
                {
                    mat_src.copyTo(mat_dst[0]);
                    return;
                }

                int i;
                for(i = 0; i < num_channels; i++)
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
    dst.resize(src.oclchannels());
    if(src.oclchannels() > 0)
        split_merge::split(src, &dst[0]);
}
