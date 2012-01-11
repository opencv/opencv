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

#include "internal_shared.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/filters.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace imgproc 
    {    
        template <typename Ptr2D, typename T> __global__ void resize(const Ptr2D src, float fx, float fy, DevMem2D_<T> dst)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                const float xcoo = x / fx;
                const float ycoo = y / fy;

                dst.ptr(y)[x] = saturate_cast<T>(src(ycoo, xcoo));
            }
        }
        template <typename Ptr2D, typename T> __global__ void resizeNN(const Ptr2D src, float fx, float fy, DevMem2D_<T> dst)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                const float xcoo = x / fx;
                const float ycoo = y / fy;

                dst.ptr(y)[x] = src(__float2int_rd(ycoo), __float2int_rd(xcoo));
            }
        }

        template <template <typename> class Filter, typename T> struct ResizeDispatcherStream
        {
            static void call(DevMem2D_<T> src, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
            {            
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);
                Filter< BorderReader< PtrStep<T>, BrdReplicate<T> > > filter_src(brdSrc);

                resize<<<grid, block, 0, stream>>>(filter_src, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
            }
        };
        template <typename T> struct ResizeDispatcherStream<PointFilter, T>
        {
            static void call(DevMem2D_<T> src, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
            {            
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);

                resizeNN<<<grid, block, 0, stream>>>(brdSrc, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
            }
        };

        template <template <typename> class Filter, typename T> struct ResizeDispatcherNonStream
        {
            static void call(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst)
            {            
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);
                Filter< BorderReader< PtrStep<T>, BrdReplicate<T> > > filter_src(brdSrc);

                resize<<<grid, block>>>(filter_src, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };
        template <typename T> struct ResizeDispatcherNonStream<PointFilter, T>
        {
            static void call(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst)
            {            
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);

                resizeNN<<<grid, block>>>(brdSrc, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        #define OPENCV_GPU_IMPLEMENT_RESIZE_TEX(type) \
            texture< type , cudaTextureType2D> tex_resize_ ## type (0, cudaFilterModePoint, cudaAddressModeClamp); \
            struct tex_resize_ ## type ## _reader \
            { \
                typedef type elem_type; \
                typedef int index_type; \
                int xoff, yoff; \
                tex_resize_ ## type ## _reader (int xoff_, int yoff_) : xoff(xoff_), yoff(yoff_) {} \
                __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const \
                { \
                    return tex2D(tex_resize_ ## type , x + xoff, y + yoff); \
                } \
            }; \
            template <template <typename> class Filter> struct ResizeDispatcherNonStream<Filter, type> \
            { \
                static void call(DevMem2D_< type > src, DevMem2D_< type > srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_< type > dst) \
                { \
                    dim3 block(32, 8); \
                    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                    bindTexture(&tex_resize_ ## type , srcWhole); \
                    tex_resize_ ## type ##_reader texSrc(xoff, yoff); \
                    BrdReplicate< type > brd(src.rows, src.cols); \
                    BorderReader< tex_resize_ ## type ##_reader , BrdReplicate< type > > brdSrc(texSrc, brd); \
                    Filter< BorderReader< tex_resize_ ## type ##_reader , BrdReplicate< type > > > filter_src(brdSrc); \
                    resize<<<grid, block>>>(filter_src, fx, fy, dst); \
                    cudaSafeCall( cudaGetLastError() ); \
                    cudaSafeCall( cudaDeviceSynchronize() ); \
                } \
            }; \
            template <> struct ResizeDispatcherNonStream<PointFilter, type> \
            { \
                static void call(DevMem2D_< type > src, DevMem2D_< type > srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_< type > dst) \
                { \
                    dim3 block(32, 8); \
                    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                    bindTexture(&tex_resize_ ## type , srcWhole); \
                    tex_resize_ ## type ##_reader texSrc(xoff, yoff); \
                    BrdReplicate< type > brd(src.rows, src.cols); \
                    BorderReader< tex_resize_ ## type ##_reader , BrdReplicate< type > > brdSrc(texSrc, brd); \
                    resizeNN<<<grid, block>>>(brdSrc, fx, fy, dst); \
                    cudaSafeCall( cudaGetLastError() ); \
                    cudaSafeCall( cudaDeviceSynchronize() ); \
                } \
            };
            
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(uchar)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(uchar2)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(uchar4)

        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(schar)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(char2)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(char4)

        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(ushort)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(ushort2)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(ushort4)

        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(short)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(short2)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(short4)

        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(int)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(int2)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(int4)

        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(float)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(float2)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(float4)

        #undef OPENCV_GPU_IMPLEMENT_RESIZE_TEX

        template <template <typename> class Filter, typename T> struct ResizeDispatcher
        { 
            static void call(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
            {
                if (stream == 0)
                    ResizeDispatcherNonStream<Filter, T>::call(src, srcWhole, xoff, yoff, fx, fy, dst);
                else
                    ResizeDispatcherStream<Filter, T>::call(src, fx, fy, dst, stream);
            }
        };

        template <typename T> void resize_gpu(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, 
            DevMem2Db dst, int interpolation, cudaStream_t stream)
        {
            typedef void (*caller_t)(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream);

            static const caller_t callers[3] = 
            {
                ResizeDispatcher<PointFilter, T>::call, ResizeDispatcher<LinearFilter, T>::call, ResizeDispatcher<CubicFilter, T>::call
            };

            callers[interpolation](static_cast< DevMem2D_<T> >(src), static_cast< DevMem2D_<T> >(srcWhole), xoff, yoff, fx, fy, 
                static_cast< DevMem2D_<T> >(dst), stream);
        }

        template void resize_gpu<uchar >(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<uchar2>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<uchar3>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<uchar4>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);

        //template void resize_gpu<schar>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<char2>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<char3>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<char4>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);

        template void resize_gpu<ushort >(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<ushort2>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<ushort3>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<ushort4>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);

        template void resize_gpu<short >(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<short2>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<short3>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<short4>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);

        //template void resize_gpu<int >(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<int2>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<int3>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<int4>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);

        template void resize_gpu<float >(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<float2>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<float3>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<float4>(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy, DevMem2Db dst, int interpolation, cudaStream_t stream);
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device
