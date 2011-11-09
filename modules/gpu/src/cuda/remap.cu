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

BEGIN_OPENCV_DEVICE_NAMESPACE

namespace imgproc {
    
template <typename Ptr2D, typename T> __global__ void remap(const Ptr2D src, const PtrStepf mapx, const PtrStepf mapy, DevMem2D_<T> dst)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < dst.cols && y < dst.rows)
    {
        const float xcoo = mapx.ptr(y)[x];
        const float ycoo = mapy.ptr(y)[x];

        dst.ptr(y)[x] = saturate_cast<T>(src(ycoo, xcoo));
    }
}

template <template <typename> class Filter, template <typename> class B, typename T> struct RemapDispatcherStream
{
    static void call(const DevMem2D_<T>& src, const DevMem2Df& mapx, const DevMem2Df& mapy, const DevMem2D_<T>& dst, 
        const float* borderValue, cudaStream_t stream, int)
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type; 
        
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
        BorderReader< PtrStep<T>, B<work_type> > brdSrc(src, brd);
        Filter< BorderReader< PtrStep<T>, B<work_type> > > filter_src(brdSrc);

        remap<<<grid, block, 0, stream>>>(filter_src, mapx, mapy, dst);
        cudaSafeCall( cudaGetLastError() );
    }
};

template <template <typename> class Filter, template <typename> class B, typename T> struct RemapDispatcherNonStream
{
    static void call(const DevMem2D_<T>& src, const DevMem2Df& mapx, const DevMem2Df& mapy, const DevMem2D_<T>& dst, const float* borderValue, int)
    {
        typedef typename TypeVec<float, VecTraits<T>::cn>::vec_type work_type; 
        
        dim3 block(32, 8);
        dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

        B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue));
        BorderReader< PtrStep<T>, B<work_type> > brdSrc(src, brd);
        Filter< BorderReader< PtrStep<T>, B<work_type> > > filter_src(brdSrc);

        remap<<<grid, block>>>(filter_src, mapx, mapy, dst);
        cudaSafeCall( cudaGetLastError() );

        cudaSafeCall( cudaDeviceSynchronize() );
    }
};

#define OPENCV_GPU_IMPLEMENT_REMAP_TEX(type) \
    texture< type , cudaTextureType2D> tex_remap_ ## type (0, cudaFilterModePoint, cudaAddressModeClamp); \
    struct tex_remap_ ## type ## _reader \
    { \
        typedef type elem_type; \
        typedef int index_type; \
        __device__ __forceinline__ elem_type operator ()(index_type y, index_type x) const \
        { \
            return tex2D(tex_remap_ ## type , x, y); \
        } \
    }; \
    template <template <typename> class Filter, template <typename> class B> struct RemapDispatcherNonStream<Filter, B, type> \
    { \
        static void call(const DevMem2D_< type >& src, const DevMem2Df& mapx, const DevMem2Df& mapy, const DevMem2D_< type >& dst, const float* borderValue, int cc) \
        { \
            typedef typename TypeVec<float, VecTraits< type >::cn>::vec_type work_type; \
            dim3 block(32, cc >= 20 ? 8 : 4); \
            dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
            bindTexture(&tex_remap_ ## type , src); \
            tex_remap_ ## type ##_reader texSrc; \
            B<work_type> brd(src.rows, src.cols, VecTraits<work_type>::make(borderValue)); \
            BorderReader< tex_remap_ ## type ##_reader, B<work_type> > brdSrc(texSrc, brd); \
            Filter< BorderReader< tex_remap_ ## type ##_reader, B<work_type> > > filter_src(brdSrc); \
            remap<<<grid, block>>>(filter_src, mapx, mapy, dst); \
            cudaSafeCall( cudaGetLastError() ); \
            cudaSafeCall( cudaDeviceSynchronize() ); \
        } \
    }; \
    template <template <typename> class Filter> struct RemapDispatcherNonStream<Filter, BrdReplicate, type> \
    { \
        static void call(const DevMem2D_< type >& src, const DevMem2Df& mapx, const DevMem2Df& mapy, const DevMem2D_< type >& dst, const float*, int) \
        { \
            dim3 block(32, 8); \
            dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y)); \
            bindTexture(&tex_remap_ ## type , src); \
            tex_remap_ ## type ##_reader texSrc; \
            Filter< tex_remap_ ## type ##_reader > filter_src(texSrc); \
            remap<<<grid, block>>>(filter_src, mapx, mapy, dst); \
            cudaSafeCall( cudaGetLastError() ); \
            cudaSafeCall( cudaDeviceSynchronize() ); \
        } \
    };
    
OPENCV_GPU_IMPLEMENT_REMAP_TEX(uchar)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(uchar2)
OPENCV_GPU_IMPLEMENT_REMAP_TEX(uchar4)

//OPENCV_GPU_IMPLEMENT_REMAP_TEX(schar)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(char2)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(char4)

OPENCV_GPU_IMPLEMENT_REMAP_TEX(ushort)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(ushort2)
OPENCV_GPU_IMPLEMENT_REMAP_TEX(ushort4)

OPENCV_GPU_IMPLEMENT_REMAP_TEX(short)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(short2)
OPENCV_GPU_IMPLEMENT_REMAP_TEX(short4)

//OPENCV_GPU_IMPLEMENT_REMAP_TEX(int)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(int2)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(int4)

OPENCV_GPU_IMPLEMENT_REMAP_TEX(float)
//OPENCV_GPU_IMPLEMENT_REMAP_TEX(float2)
OPENCV_GPU_IMPLEMENT_REMAP_TEX(float4)

#undef OPENCV_GPU_IMPLEMENT_REMAP_TEX

template <template <typename> class Filter, template <typename> class B, typename T> struct RemapDispatcher
{ 
    static void call(const DevMem2D_<T>& src, const DevMem2Df& mapx, const DevMem2Df& mapy, const DevMem2D_<T>& dst, 
        const float* borderValue, cudaStream_t stream, int cc)
    {
        if (stream == 0)
            RemapDispatcherNonStream<Filter, B, T>::call(src, mapx, mapy, dst, borderValue, cc);
        else
            RemapDispatcherStream<Filter, B, T>::call(src, mapx, mapy, dst, borderValue, stream, cc);
    }
};

template <typename T> void remap_gpu(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, 
    int borderMode, const float* borderValue, cudaStream_t stream, int cc)
{
    typedef void (*caller_t)(const DevMem2D_<T>& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2D_<T>& dst, 
        const float* borderValue, cudaStream_t stream, int cc);

    static const caller_t callers[3][5] = 
    {
        { 
            RemapDispatcher<PointFilter, BrdReflect101, T>::call, 
            RemapDispatcher<PointFilter, BrdReplicate, T>::call, 
            RemapDispatcher<PointFilter, BrdConstant, T>::call, 
            RemapDispatcher<PointFilter, BrdReflect, T>::call, 
            RemapDispatcher<PointFilter, BrdWrap, T>::call 
        },
        { 
            RemapDispatcher<LinearFilter, BrdReflect101, T>::call, 
            RemapDispatcher<LinearFilter, BrdReplicate, T>::call, 
            RemapDispatcher<LinearFilter, BrdConstant, T>::call, 
            RemapDispatcher<LinearFilter, BrdReflect, T>::call, 
            RemapDispatcher<LinearFilter, BrdWrap, T>::call 
        },
        { 
            RemapDispatcher<CubicFilter, BrdReflect101, T>::call, 
            RemapDispatcher<CubicFilter, BrdReplicate, T>::call, 
            RemapDispatcher<CubicFilter, BrdConstant, T>::call, 
            RemapDispatcher<CubicFilter, BrdReflect, T>::call, 
            RemapDispatcher<CubicFilter, BrdWrap, T>::call 
        }
    };

    callers[interpolation][borderMode](static_cast< DevMem2D_<T> >(src), xmap, ymap, static_cast< DevMem2D_<T> >(dst), borderValue, stream, cc);
}

template void remap_gpu<uchar >(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<uchar2>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<uchar3>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<uchar4>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);

//template void remap_gpu<schar>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<char2>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<char3>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<char4>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);

template void remap_gpu<ushort >(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<ushort2>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<ushort3>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<ushort4>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);

template void remap_gpu<short >(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<short2>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<short3>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<short4>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);

//template void remap_gpu<int >(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<int2>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<int3>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<int4>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);

template void remap_gpu<float >(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
//template void remap_gpu<float2>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<float3>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);
template void remap_gpu<float4>(const DevMem2Db& src, const DevMem2Df& xmap, const DevMem2Df& ymap, const DevMem2Db& dst, int interpolation, int borderMode, const float* borderValue, cudaStream_t stream, int cc);

} // namespace imgproc

END_OPENCV_DEVICE_NAMESPACE
