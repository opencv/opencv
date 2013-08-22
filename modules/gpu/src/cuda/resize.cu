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

#if !defined CUDA_DISABLER

#include <cfloat>
#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/border_interpolate.hpp"
#include "opencv2/gpu/device/vec_traits.hpp"
#include "opencv2/gpu/device/vec_math.hpp"
#include "opencv2/gpu/device/saturate_cast.hpp"
#include "opencv2/gpu/device/filters.hpp"

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T> __global__ void resize_nearest(const PtrStep<T> src, const float fx, const float fy, PtrStepSz<T> dst)
        {
            const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
            const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

            if (dst_x < dst.cols && dst_y < dst.rows)
            {
                const float src_x = dst_x * fx;
                const float src_y = dst_y * fy;

                dst(dst_y, dst_x) = src(__float2int_rz(src_y), __float2int_rz(src_x));
            }
        }

        template <typename T> __global__ void resize_linear(const PtrStepSz<T> src, const float fx, const float fy, PtrStepSz<T> dst)
        {
            typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type;

            const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
            const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

            if (dst_x < dst.cols && dst_y < dst.rows)
            {
                const float src_x = dst_x * fx;
                const float src_y = dst_y * fy;

                work_type out = VecTraits<work_type>::all(0);

                const int x1 = __float2int_rd(src_x);
                const int y1 = __float2int_rd(src_y);
                const int x2 = x1 + 1;
                const int y2 = y1 + 1;
                const int x2_read = ::min(x2, src.cols - 1);
                const int y2_read = ::min(y2, src.rows - 1);

                T src_reg = src(y1, x1);
                out = out + src_reg * ((x2 - src_x) * (y2 - src_y));

                src_reg = src(y1, x2_read);
                out = out + src_reg * ((src_x - x1) * (y2 - src_y));

                src_reg = src(y2_read, x1);
                out = out + src_reg * ((x2 - src_x) * (src_y - y1));

                src_reg = src(y2_read, x2_read);
                out = out + src_reg * ((src_x - x1) * (src_y - y1));

                dst(dst_y, dst_x) = saturate_cast<T>(out);
            }
        }

        template <class Ptr2D, typename T> __global__ void resize(const Ptr2D src, const float fx, const float fy, PtrStepSz<T> dst)
        {
            const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
            const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

            if (dst_x < dst.cols && dst_y < dst.rows)
            {
                const float src_x = dst_x * fx;
                const float src_y = dst_y * fy;

                dst(dst_y, dst_x) = src(src_y, src_x);
            }
        }

        template <template <typename> class Filter, typename T> struct ResizeDispatcherStream
        {
            static void call(PtrStepSz<T> src, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                const dim3 block(32, 8);
                const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);
                Filter< BorderReader< PtrStep<T>, BrdReplicate<T> > > filteredSrc(brdSrc);

                resize<<<grid, block, 0, stream>>>(filteredSrc, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
            }
        };
        template <typename T> struct ResizeDispatcherStream<PointFilter, T>
        {
            static void call(PtrStepSz<T> src, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                const dim3 block(32, 8);
                const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                resize_nearest<<<grid, block, 0, stream>>>(src, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
            }
        };
        template <typename T> struct ResizeDispatcherStream<LinearFilter, T>
        {
            static void call(PtrStepSz<T> src, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                const dim3 block(32, 8);
                const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                resize_linear<<<grid, block, 0, stream>>>(src, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );
            }
        };

        template <template <typename> class Filter, typename T> struct ResizeDispatcherNonStream
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T>, int, int, float fx, float fy, PtrStepSz<T> dst)
            {
                const dim3 block(32, 8);
                const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                BrdReplicate<T> brd(src.rows, src.cols);
                BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);
                Filter< BorderReader< PtrStep<T>, BrdReplicate<T> > > filteredSrc(brdSrc);

                resize<<<grid, block>>>(filteredSrc, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };
        template <typename T> struct ResizeDispatcherNonStream<PointFilter, T>
        {
        static void call(PtrStepSz<T> src, PtrStepSz<T>, int, int, float fx, float fy, PtrStepSz<T> dst)
            {
                const dim3 block(32, 8);
                const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                resize_nearest<<<grid, block>>>(src, fx, fy, dst);
                cudaSafeCall( cudaGetLastError() );

                cudaSafeCall( cudaDeviceSynchronize() );
            }
        };
        template <typename T> struct ResizeDispatcherNonStream<LinearFilter, T>
        {
        static void call(PtrStepSz<T> src, PtrStepSz<T>, int, int, float fx, float fy, PtrStepSz<T> dst)
            {
                const dim3 block(32, 8);
                const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                resize_linear<<<grid, block>>>(src, fx, fy, dst);
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
                int xoff; \
                int yoff; \
                __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const \
                { \
                    return tex2D(tex_resize_ ## type, x + xoff, y + yoff); \
                } \
            }; \
            template <template <typename> class Filter> struct ResizeDispatcherNonStream<Filter, type > \
            { \
                static void call(PtrStepSz< type > src, PtrStepSz< type > srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz< type > dst) \
                { \
                    const dim3 block(32, 8); \
                    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                    bindTexture(&tex_resize_ ## type, srcWhole); \
                    tex_resize_ ## type ## _reader texSrc; \
                    texSrc.xoff = xoff; \
                    texSrc.yoff = yoff; \
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
            }; \
            template <> struct ResizeDispatcherNonStream<PointFilter, type > \
            { \
                static void call(PtrStepSz< type > src, PtrStepSz< type > srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz< type > dst) \
                { \
                    const dim3 block(32, 8); \
                    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                    bindTexture(&tex_resize_ ## type, srcWhole); \
                    tex_resize_ ## type ## _reader texSrc; \
                    texSrc.xoff = xoff; \
                    texSrc.yoff = yoff; \
                    if (srcWhole.cols == src.cols && srcWhole.rows == src.rows) \
                    { \
                        PointFilter<tex_resize_ ## type ## _reader> filteredSrc(texSrc); \
                        resize<<<grid, block>>>(filteredSrc, fx, fy, dst); \
                    } \
                    else \
                    { \
                        BrdReplicate< type > brd(src.rows, src.cols); \
                        BorderReader<tex_resize_ ## type ## _reader, BrdReplicate< type > > brdSrc(texSrc, brd); \
                        PointFilter< BorderReader<tex_resize_ ## type ## _reader, BrdReplicate< type > > > filteredSrc(brdSrc); \
                        resize<<<grid, block>>>(filteredSrc, fx, fy, dst); \
                    } \
                    cudaSafeCall( cudaGetLastError() ); \
                    cudaSafeCall( cudaDeviceSynchronize() ); \
                } \
            }; \
            template <> struct ResizeDispatcherNonStream<LinearFilter, type > \
            { \
                static void call(PtrStepSz< type > src, PtrStepSz< type > srcWhole, int xoff, int yoff, float fx, float fy, PtrStepSz< type > dst) \
                { \
                    const dim3 block(32, 8); \
                    const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
                    bindTexture(&tex_resize_ ## type, srcWhole); \
                    tex_resize_ ## type ## _reader texSrc; \
                    texSrc.xoff = xoff; \
                    texSrc.yoff = yoff; \
                    if (srcWhole.cols == src.cols && srcWhole.rows == src.rows) \
                    { \
                        LinearFilter<tex_resize_ ## type ## _reader> filteredSrc(texSrc); \
                        resize<<<grid, block>>>(filteredSrc, fx, fy, dst); \
                    } \
                    else \
                    { \
                        BrdReplicate< type > brd(src.rows, src.cols); \
                        BorderReader<tex_resize_ ## type ## _reader, BrdReplicate< type > > brdSrc(texSrc, brd); \
                        LinearFilter< BorderReader<tex_resize_ ## type ## _reader, BrdReplicate< type > > > filteredSrc(brdSrc); \
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

        template <typename Ptr2D, typename T> __global__ void resize_area(const Ptr2D src, PtrStepSz<T> dst)
        {
            const int x = blockDim.x * blockIdx.x + threadIdx.x;
            const int y = blockDim.y * blockIdx.y + threadIdx.y;

            if (x < dst.cols && y < dst.rows)
            {
                dst(y, x) = src(y, x);
            }
        }

        template <typename T> struct ResizeAreaDispatcher
        {
            static void call(PtrStepSz<T> src, PtrStepSz<T>, int, int, float fx, float fy, PtrStepSz<T> dst, cudaStream_t stream)
            {
                const int iscale_x = (int) round(fx);
                const int iscale_y = (int) round(fy);

                const dim3 block(32, 8);
                const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

                if (std::abs(fx - iscale_x) < FLT_MIN && std::abs(fy - iscale_y) < FLT_MIN)
                {
                    BrdConstant<T> brd(src.rows, src.cols);
                    BorderReader< PtrStep<T>, BrdConstant<T> > brdSrc(src, brd);
                    IntegerAreaFilter< BorderReader< PtrStep<T>, BrdConstant<T> > > filteredSrc(brdSrc, fx, fy);
                    resize_area<<<grid, block, 0, stream>>>(filteredSrc, dst);
                }
                else
                {
                    BrdConstant<T> brd(src.rows, src.cols);
                    BorderReader< PtrStep<T>, BrdConstant<T> > brdSrc(src, brd);
                    AreaFilter< BorderReader< PtrStep<T>, BrdConstant<T> > > filteredSrc(brdSrc, fx, fy);
                    resize_area<<<grid, block, 0, stream>>>(filteredSrc, dst);
                }

                cudaSafeCall( cudaGetLastError() );
                if (stream == 0)
                    cudaSafeCall( cudaDeviceSynchronize() );
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
                ResizeAreaDispatcher<T>::call
            };

            // change to linear if area interpolation upscaling
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
    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device

#endif /* CUDA_DISABLER */
