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
    // kernels

    template <typename T> __global__ void resize_nearest(const PtrStep<T> src, PtrStepSz<T> dst, const float fy, const float fx)
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

    template <typename T> __global__ void resize_linear(const PtrStepSz<T> src, PtrStepSz<T> dst, const float fy, const float fx)
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

    template <class Ptr2D, typename T> __global__ void resize(const Ptr2D src, PtrStepSz<T> dst, const float fy, const float fx)
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

    template <typename Ptr2D, typename T> __global__ void resize_area(const Ptr2D src, PtrStepSz<T> dst)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x < dst.cols && y < dst.rows)
        {
            dst(y, x) = src(y, x);
        }
    }

    // textures

    template <typename T> struct TextureAccessor;

    #define OPENCV_GPU_IMPLEMENT_RESIZE_TEX(type) \
        texture<type, cudaTextureType2D, cudaReadModeElementType> tex_resize_##type (0, cudaFilterModePoint, cudaAddressModeClamp); \
        template <> struct TextureAccessor<type> \
        { \
            typedef type elem_type; \
            typedef int index_type; \
            int xoff; \
            int yoff; \
            __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const \
            { \
                return tex2D(tex_resize_##type, x + xoff, y + yoff); \
            } \
            __host__ static void bind(const PtrStepSz<type>& mat) \
            { \
                bindTexture(&tex_resize_##type, mat); \
            } \
        };

    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(uchar)
    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(uchar4)

    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(ushort)
    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(ushort4)

    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(short)
    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(short4)

    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(float)
    OPENCV_GPU_IMPLEMENT_RESIZE_TEX(float4)

    #undef OPENCV_GPU_IMPLEMENT_RESIZE_TEX

    template <typename T>
    TextureAccessor<T> texAccessor(const PtrStepSz<T>& mat, int yoff, int xoff)
    {
        TextureAccessor<T>::bind(mat);

        TextureAccessor<T> t;
        t.xoff = xoff;
        t.yoff = yoff;

        return t;
    }

    // callers for nearest interpolation

    template <typename T>
    void call_resize_nearest_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        resize_nearest<<<grid, block, 0, stream>>>(src, dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_nearest_tex(const PtrStepSz<T>& /*src*/, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        resize<<<grid, block>>>(texAccessor(srcWhole, yoff, xoff), dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // callers for linear interpolation

    template <typename T>
    void call_resize_linear_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        resize_linear<<<grid, block>>>(src, dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_linear_tex(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        if (srcWhole.data == src.data)
        {
            TextureAccessor<T> texSrc = texAccessor(src, 0, 0);
            LinearFilter< TextureAccessor<T> > filteredSrc(texSrc);

            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }
        else
        {
            TextureAccessor<T> texSrc = texAccessor(srcWhole, yoff, xoff);

            BrdReplicate<T> brd(src.rows, src.cols);
            BorderReader<TextureAccessor<T>, BrdReplicate<T> > brdSrc(texSrc, brd);
            LinearFilter< BorderReader<TextureAccessor<T>, BrdReplicate<T> > > filteredSrc(brdSrc);

            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }

        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // callers for cubic interpolation

    template <typename T>
    void call_resize_cubic_glob(const PtrStepSz<T>& src, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        BrdReplicate<T> brd(src.rows, src.cols);
        BorderReader< PtrStep<T>, BrdReplicate<T> > brdSrc(src, brd);
        CubicFilter< BorderReader< PtrStep<T>, BrdReplicate<T> > > filteredSrc(brdSrc);

        resize<<<grid, block, 0, stream>>>(filteredSrc, dst, fy, fx);
        cudaSafeCall( cudaGetLastError() );

        if (stream == 0)
            cudaSafeCall( cudaDeviceSynchronize() );
    }

    template <typename T>
    void call_resize_cubic_tex(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx)
    {
        const dim3 block(32, 8);
        const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        if (srcWhole.data == src.data)
        {
            TextureAccessor<T> texSrc = texAccessor(src, 0, 0);
            CubicFilter< TextureAccessor<T> > filteredSrc(texSrc);

            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }
        else
        {
            TextureAccessor<T> texSrc = texAccessor(srcWhole, yoff, xoff);

            BrdReplicate<T> brd(src.rows, src.cols);
            BorderReader<TextureAccessor<T>, BrdReplicate<T> > brdSrc(texSrc, brd);
            CubicFilter< BorderReader<TextureAccessor<T>, BrdReplicate<T> > > filteredSrc(brdSrc);

            resize<<<grid, block>>>(filteredSrc, dst, fy, fx);
        }

        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }

    // ResizeNearestDispatcher

    template <typename T> struct ResizeNearestDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& /*srcWhole*/, int /*yoff*/, int /*xoff*/, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_nearest_glob(src, dst, fy, fx, stream);
        }
    };

    template <typename T> struct SelectImplForNearest
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            if (stream)
                call_resize_nearest_glob(src, dst, fy, fx, stream);
            else
            {
                if (fx > 1 || fy > 1)
                    call_resize_nearest_glob(src, dst, fy, fx, 0);
                else
                    call_resize_nearest_tex(src, srcWhole, yoff, xoff, dst, fy, fx);
            }
        }
    };

    template <> struct ResizeNearestDispatcher<uchar> : SelectImplForNearest<uchar> {};
    template <> struct ResizeNearestDispatcher<uchar4> : SelectImplForNearest<uchar4> {};

    template <> struct ResizeNearestDispatcher<ushort> : SelectImplForNearest<ushort> {};
    template <> struct ResizeNearestDispatcher<ushort4> : SelectImplForNearest<ushort4> {};

    template <> struct ResizeNearestDispatcher<short> : SelectImplForNearest<short> {};
    template <> struct ResizeNearestDispatcher<short4> : SelectImplForNearest<short4> {};

    template <> struct ResizeNearestDispatcher<float> : SelectImplForNearest<float> {};
    template <> struct ResizeNearestDispatcher<float4> : SelectImplForNearest<float4> {};

    // ResizeLinearDispatcher

    template <typename T> struct ResizeLinearDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_linear_glob(src, dst, fy, fx, stream);
        }
    };

    template <typename T> struct SelectImplForLinear
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            if (stream)
                call_resize_linear_glob(src, dst, fy, fx, stream);
            else
            {
                if (fx > 1 || fy > 1)
                    call_resize_linear_glob(src, dst, fy, fx, 0);
                else
                    call_resize_linear_tex(src, srcWhole, yoff, xoff, dst, fy, fx);
            }
        }
    };

    template <> struct ResizeLinearDispatcher<uchar> : SelectImplForLinear<uchar> {};
    template <> struct ResizeLinearDispatcher<uchar4> : SelectImplForLinear<uchar4> {};

    template <> struct ResizeLinearDispatcher<ushort> : SelectImplForLinear<ushort> {};
    template <> struct ResizeLinearDispatcher<ushort4> : SelectImplForLinear<ushort4> {};

    template <> struct ResizeLinearDispatcher<short> : SelectImplForLinear<short> {};
    template <> struct ResizeLinearDispatcher<short4> : SelectImplForLinear<short4> {};

    template <> struct ResizeLinearDispatcher<float> : SelectImplForLinear<float> {};
    template <> struct ResizeLinearDispatcher<float4> : SelectImplForLinear<float4> {};

    // ResizeCubicDispatcher

    template <typename T> struct ResizeCubicDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            call_resize_cubic_glob(src, dst, fy, fx, stream);
        }
    };

    template <typename T> struct SelectImplForCubic
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
        {
            if (stream)
                call_resize_cubic_glob(src, dst, fy, fx, stream);
            else
                call_resize_cubic_tex(src, srcWhole, yoff, xoff, dst, fy, fx);
        }
    };

    template <> struct ResizeCubicDispatcher<uchar> : SelectImplForCubic<uchar> {};
    template <> struct ResizeCubicDispatcher<uchar4> : SelectImplForCubic<uchar4> {};

    template <> struct ResizeCubicDispatcher<ushort> : SelectImplForCubic<ushort> {};
    template <> struct ResizeCubicDispatcher<ushort4> : SelectImplForCubic<ushort4> {};

    template <> struct ResizeCubicDispatcher<short> : SelectImplForCubic<short> {};
    template <> struct ResizeCubicDispatcher<short4> : SelectImplForCubic<short4> {};

    template <> struct ResizeCubicDispatcher<float> : SelectImplForCubic<float> {};
    template <> struct ResizeCubicDispatcher<float4> : SelectImplForCubic<float4> {};

    // ResizeAreaDispatcher

    template <typename T> struct ResizeAreaDispatcher
    {
        static void call(const PtrStepSz<T>& src, const PtrStepSz<T>&, int, int, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream)
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

    // resize

    template <typename T> void resize(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream)
    {
        typedef void (*func_t)(const PtrStepSz<T>& src, const PtrStepSz<T>& srcWhole, int yoff, int xoff, const PtrStepSz<T>& dst, float fy, float fx, cudaStream_t stream);
        static const func_t funcs[4] =
        {
            ResizeNearestDispatcher<T>::call,
            ResizeLinearDispatcher<T>::call,
            ResizeCubicDispatcher<T>::call,
            ResizeAreaDispatcher<T>::call
        };

        // change to linear if area interpolation upscaling
        if (interpolation == 3 && (fx <= 1.f || fy <= 1.f))
            interpolation = 1;

        funcs[interpolation](static_cast< PtrStepSz<T> >(src), static_cast< PtrStepSz<T> >(srcWhole), yoff, xoff, static_cast< PtrStepSz<T> >(dst), fy, fx, stream);
    }

    template void resize<uchar >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<uchar3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<uchar4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);

    template void resize<ushort >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<ushort3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<ushort4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);

    template void resize<short >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<short3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<short4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);

    template void resize<float >(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<float3>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
    template void resize<float4>(const PtrStepSzb& src, const PtrStepSzb& srcWhole, int yoff, int xoff, const PtrStepSzb& dst, float fy, float fx, int interpolation, cudaStream_t stream);
}}}

#endif /* CUDA_DISABLER */
