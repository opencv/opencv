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
#include <cfloat>
#include <opencv2/gpu/device/scan.hpp>

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename Ptr2D, typename T> __global__ void resize(const Ptr2D src, float fx, float fy, PtrStepSz<T> dst)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                const float xcoo = x * fx;
                const float ycoo = y * fy;

                dst(y, x) = saturate_cast<T>(src(ycoo, xcoo));
            }
        }

        template <typename Ptr2D, typename T> __global__ void resize_area(const Ptr2D src, float fx, float fy, PtrStepSz<T> dst)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                dst(y, x) = saturate_cast<T>(src(y, x));
            }
        }

        template <template <typename> class Filter, typename T> struct ResizeDispatcherStream
        {
            static void call(PtrStepSz<T> src, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);
                Filter< BorderReader< PtrStep<T>, BrdReplicate<T> > > filteredSrc(brdSrc, fx, fy);

                resize<<<grid, block, 0, stream>>>(filteredSrc, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
            }
        };

        template <typename T> struct ResizeDispatcherStream<AreaFilter, T>
        {
            static void call(PtrStepSz<T> src, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdConstant<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdConstant<T> > brdSrc(src, brd);
                AreaFilter< BorderReader< PtrStep<T>, BrdConstant<T> > > filteredSrc(brdSrc, fx, fy);
                resize_area<<<grid, block, 0, stream>>>(filteredSrc, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        template <typename T> struct ResizeDispatcherStream<IntegerAreaFilter, T>
        {
            static void call(PtrStepSz<T> src, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
                BrdConstant<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdConstant<T> > brdSrc(src, brd);
                IntegerAreaFilter< BorderReader< PtrStep<T>, BrdConstant<T> > > filteredSrc(brdSrc, fx, fy);
                resize_area<<<grid, block, 0, stream>>>(filteredSrc, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );
            }
        };

        template <template <typename> class Filter, typename T> struct ResizeDispatcherNonStream
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz<T> dst)
            {
                (void)srcWhole;
                (void)xoff;
                (void)yoff;

                dim3 block(32, 8);
                dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);
                Filter< BorderReader< PtrStep<T>, BrdReplicate<T> > > filteredSrc(brdSrc);

                resize<<<grid, block>>>(filteredSrc, fx, fy, dst);
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
                const int xoff; \
                const int yoff; \
                __host__ tex_resize_ ## type ## _reader(int xoff_, int yoff_) : xoff(xoff_), yoff(yoff_) {} \
                __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const \
                { \
                    return tex2D(tex_resize_ ## type, x + xoff, y + yoff); \
                } \
            }; \
            template <template <typename> class Filter> struct ResizeDispatcherNonStream<Filter, type > \
            { \
                static void call(PtrStepSz< type > src, PtrStepSz< type > srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz< type > dst) \
                { \
                    dim3 block(32, 8); \
                    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                    bindTexture(&tex_resize_ ## type, srcWhole); \
                    tex_resize_ ## type ## _reader texSrc(xoff, yoff); \
                    if (srcWhole.cols == src.cols && srcWhole.rows == src.rows) \
                    { \
                        Filter<tex_resize_ ## type ## _reader> filteredSrc(texSrc); \
                        resize<<<grid, block>>>(filteredSrc, fx, fy, dst); \
                    } \
                    else \
                    { \
                        BrdReplicate< type > brd(src.rows, src.cols); \
                        BorderReader<tex_resize_ ## type ## _reader, BrdReplicate< type > > brdSrc(texSrc, brd); \
                        Filter< BorderReader<tex_resize_ ## type ## _reader, BrdReplicate< type > > > filteredSrc(brdSrc); \
                        resize<<<grid, block>>>(filteredSrc, fx, fy, dst); \
                    } \
                    cudaSafeCall( cudaGetLastError() ); \
                    cudaSafeCall( cudaDeviceSynchronize() ); \
                } \
            };

        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(uchar)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(uchar4)

        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(schar)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(char4)

        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(ushort)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(ushort4)

        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(short)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(short4)

        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(int)
        //OPENCV_GPU_IMPLEMENT_RESIZE_TEX(int4)

        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(float)
        OPENCV_GPU_IMPLEMENT_RESIZE_TEX(float4)

        #undef OPENCV_GPU_IMPLEMENT_RESIZE_TEX

        template <template <typename> class Filter, typename T> struct ResizeDispatcher
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                if (stream == 0)
                    ResizeDispatcherNonStream<Filter, T>::call(src, srcWhole, xoff, yoff, fx, fy, dst);
                else
                    ResizeDispatcherStream<Filter, T>::call(src, fx, fy, dst, stream);
            }
        };

        template <typename T> struct ResizeDispatcher<AreaFilter, T>
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                (void)srcWhole;
                (void)xoff;
                (void)yoff;
                int iscale_x = (int)round(fx);
                int iscale_y = (int)round(fy);

                if( std::abs(fx - iscale_x) < FLT_MIN && std::abs(fy - iscale_y) < FLT_MIN)
                    ResizeDispatcherStream<IntegerAreaFilter, T>::call(src, fx, fy, dst, stream);
                else
                    ResizeDispatcherStream<AreaFilter, T>::call(src, fx, fy, dst, stream);
            }
        };

        template <typename T> void resize_gpu(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy,
            PtrStepSzb dst, int interpolation, cudaStream_t stream)
        {
            typedef void (*caller_t)(PtrStepSz<T> src, PtrStepSz<T> srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream);

            static const caller_t callers[4] =
            {
                ResizeDispatcher<PointFilter, T>::call,
                ResizeDispatcher<LinearFilter, T>::call,
                ResizeDispatcher<CubicFilter, T>::call,
                ResizeDispatcher<AreaFilter, T>::call
            };
            // chenge to linear if area interpolation upscaling
            if (interpolation == 3 && (fx <= 1.f || fy <= 1.f))
                interpolation = 1;

            callers[interpolation](static_cast< PtrStepSz<T> >(src), static_cast< PtrStepSz<T> >(srcWhole), xoff, yoff, fx, fy,
                static_cast< PtrStepSz<T> >(dst), stream);
        }

        template void resize_gpu<uchar >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<uchar2>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<uchar3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<uchar4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);

        //template void resize_gpu<schar>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<char2>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<char3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<char4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);

        template void resize_gpu<ushort >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<ushort2>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<ushort3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<ushort4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);

        template void resize_gpu<short >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<short2>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<short3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<short4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);

        //template void resize_gpu<int >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<int2>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<int3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<int4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);

        template void resize_gpu<float >(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        //template void resize_gpu<float2>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<float3>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);
        template void resize_gpu<float4>(PtrStepSzb src, PtrStepSzb srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSzb dst, int interpolation, cudaStream_t stream);

        template<typename T> struct scan_traits{};

        template<> struct scan_traits<uchar>
        {
            typedef float scan_line_type;
        };

    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device
