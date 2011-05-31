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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or bpied warranties, including, but not limited to, the bpied
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

using namespace std;

#if !defined (HAVE_CUDA)

void cv::gpu::merge(const GpuMat* /*src*/, size_t /*count*/, GpuMat& /*dst*/, Stream& /*stream*/) { throw_nogpu(); }
void cv::gpu::merge(const vector<GpuMat>& /*src*/, GpuMat& /*dst*/, Stream& /*stream*/) { throw_nogpu(); }
void cv::gpu::split(const GpuMat& /*src*/, GpuMat* /*dst*/, Stream& /*stream*/) { throw_nogpu(); }
void cv::gpu::split(const GpuMat& /*src*/, vector<GpuMat>& /*dst*/, Stream& /*stream*/) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace split_merge 
{    
    extern "C" void merge_caller(const DevMem2D* src, DevMem2D& dst, 
                                 int total_channels, int elem_size, 
                                 const cudaStream_t& stream);

    extern "C" void split_caller(const DevMem2D& src, DevMem2D* dst, 
                                 int num_channels, int elem_size1, 
                                 const cudaStream_t& stream);

    void merge(const GpuMat* src, size_t n, GpuMat& dst, const cudaStream_t& stream) 
    {
        CV_Assert(src);
        CV_Assert(n > 0);
       
        bool double_ok = TargetArchs::builtWith(NATIVE_DOUBLE) && 
                         DeviceInfo().supports(NATIVE_DOUBLE);
        CV_Assert(src[0].depth() != CV_64F || double_ok);

        int depth = src[0].depth();
        Size size = src[0].size();

        bool single_channel_only = true;
        int total_channels = 0;

        for (size_t i = 0; i < n; ++i)
        {
            CV_Assert(src[i].size() == size);
            CV_Assert(src[i].depth() == depth);
            single_channel_only = single_channel_only && src[i].channels() == 1;
            total_channels += src[i].channels();
        }

        CV_Assert(single_channel_only);
        CV_Assert(total_channels <= 4);

        if (total_channels == 1)  
            src[0].copyTo(dst);
        else 
        {
            dst.create(size, CV_MAKETYPE(depth, total_channels));

            DevMem2D src_as_devmem[4];
            for(size_t i = 0; i < n; ++i)
                src_as_devmem[i] = src[i];

            DevMem2D dst_as_devmem(dst);
            split_merge::merge_caller(src_as_devmem, dst_as_devmem,
                                      total_channels, CV_ELEM_SIZE(depth),
                                      stream);
        }   
    }


    void split(const GpuMat& src, GpuMat* dst, const cudaStream_t& stream) 
    {
        CV_Assert(dst);

        bool double_ok = TargetArchs::builtWith(NATIVE_DOUBLE) && 
                         DeviceInfo().supports(NATIVE_DOUBLE);
        CV_Assert(src.depth() != CV_64F || double_ok);

        int depth = src.depth();
        int num_channels = src.channels();
        Size size = src.size();

        if (num_channels == 1)
        {
            src.copyTo(dst[0]);
            return;
        }

        for (int i = 0; i < num_channels; ++i)
            dst[i].create(src.size(), depth);

        CV_Assert(num_channels <= 4);

        DevMem2D dst_as_devmem[4];
        for (int i = 0; i < num_channels; ++i)
            dst_as_devmem[i] = dst[i];

        DevMem2D src_as_devmem(src);
        split_merge::split_caller(src_as_devmem, dst_as_devmem,
                                  num_channels, src.elemSize1(), 
                                  stream);
    }


}}}


void cv::gpu::merge(const GpuMat* src, size_t n, GpuMat& dst, Stream& stream) 
{ 
    split_merge::merge(src, n, dst, StreamAccessor::getStream(stream));
}


void cv::gpu::merge(const vector<GpuMat>& src, GpuMat& dst, Stream& stream) 
{
    split_merge::merge(&src[0], src.size(), dst, StreamAccessor::getStream(stream));
}


void cv::gpu::split(const GpuMat& src, GpuMat* dst, Stream& stream) 
{
    split_merge::split(src, dst, StreamAccessor::getStream(stream));
}


void cv::gpu::split(const GpuMat& src, vector<GpuMat>& dst, Stream& stream) 
{
    dst.resize(src.channels());
    if(src.channels() > 0)
        split_merge::split(src, &dst[0], StreamAccessor::getStream(stream));
}

#endif /* !defined (HAVE_CUDA) */
