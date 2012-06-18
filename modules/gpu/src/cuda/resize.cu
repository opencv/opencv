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
# include <cfloat>

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
                const float xcoo = x * fx;
                const float ycoo = y * fy;

                dst(y, x) = saturate_cast<T>(src(ycoo, xcoo));
            }
        }

        template <typename Ptr2D, typename T> __global__ void resize_area(const Ptr2D src, float fx, float fy, DevMem2D_<T> dst)
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
            static void call(DevMem2D_<T> src, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
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
            static void call(DevMem2D_<T> src, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
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
            static void call(DevMem2D_<T> src, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
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
            static void call(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst)
            {
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
                static void call(DevMem2D_< type > src, DevMem2D_< type > srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_< type > dst) \
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
            static void call(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
            {
                if (stream == 0)
                    ResizeDispatcherNonStream<Filter, T>::call(src, srcWhole, xoff, yoff, fx, fy, dst);
                else
                    ResizeDispatcherStream<Filter, T>::call(src, fx, fy, dst, stream);
            }
        };

        template <typename T> struct ResizeDispatcher<AreaFilter, T>
        {
            static void call(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream)
            {
                int iscale_x = round(fx);
                int iscale_y = round(fy);

                if( std::abs(fx - iscale_x) < FLT_MIN && std::abs(fy - iscale_y) < FLT_MIN)
                    ResizeDispatcherStream<IntegerAreaFilter, T>::call(src, fx, fy, dst, stream);
                else
                    ResizeDispatcherStream<AreaFilter, T>::call(src, fx, fy, dst, stream);
            }
        };

        template <typename T> void resize_gpu(DevMem2Db src, DevMem2Db srcWhole, int xoff, int yoff, float fx, float fy,
            DevMem2Db dst, int interpolation, cudaStream_t stream)
        {
            typedef void (*caller_t)(DevMem2D_<T> src, DevMem2D_<T> srcWhole, int xoff, int yoff, float fx, float fy, DevMem2D_<T> dst, cudaStream_t stream);

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

        template<typename T> struct scan_traits{};

        template<> struct scan_traits<uchar>
        {
            typedef float scan_line_type;
        };

//        template <typename T>
//        __global__ void resize_area_scan(const DevMem2D_<T> src, DevMem2D_<T> dst, int fx, int fy,  DevMem2D_<T> buffer)
//        {
//            typedef typename scan_traits<T>::scan_line_type W;
//            extern __shared__ W line[];

//            const int x = threadIdx.x;
//            const int y = blockIdx.x;

//            if (y >= src.rows) return;

//            int offset = 1;

//            line[2 * x + 0] = src(y, 2 * x + 0);
//            line[2 * x + 1] = src(y, 2 * x + 1);

//            __syncthreads();//???
//            // reduction
//            for (int d = blockDim.x; d > 0; d >>= 1)
//            {
//                __syncthreads();
//                if (x < d)
//                {
//                    int ai = 2 * x * offset -1 + 1 * offset;
//                    int bi = 2 * x * offset -1 + 2 * offset;
//                    line[bi] += line[ai];
//                }

//                offset *= 2;
//            }

//            __syncthreads();
//            // convolution
//            if (x == 0) { line[(blockDim.x << 1) - 1] = 0; printf("offset: %d!!!!!!!!!!!!!\n", fx);}

//            for (int d = 1; d < (blockDim.x << 1); d *= 2)
//            {
//                offset >>= 1;

//                __syncthreads();
//                if (x < d)
//                {
//                    int ai = offset * 2 * x + 1 * offset - 1;
//                    int bi = offset * 2 * x + 2 * offset - 1;

//                    W t = line[ai];
//                    line[ai] = line[bi];
//                    line[bi] += t;
//                }
//            }
//            __syncthreads();

//            // calculate sum
//            int start = 0;
//            int out_idx = 0;
//            int end = start + fx;
//            while (start < (blockDim.x << 1) && end < (blockDim.x << 1))
//            {
//                buffer(y, out_idx) = saturate_cast<T>((line[end] - line[start]) / fx);
//                start = end;
//                end = start + fx;
//                out_idx++;
//            }

//        }

        template <typename T>
        __device__ void scan_y(DevMem2D_<typename scan_traits<T>::scan_line_type> buffer,int fx, int fy,  DevMem2D_<T> dst,
                               typename scan_traits<T>::scan_line_type* line, int g_base)
        {
            typedef typename scan_traits<T>::scan_line_type W;

            const int y = threadIdx.x;
            const int x = blockIdx.x;

            float scale = 1.f / (fx * fy);

            if (x >= buffer.cols) return;

            int offset = 1;
            line[2 * y + 0] = buffer((g_base * fy) + 2 * y + 1, x);

            if (y != (blockDim.x -1) )
                line[2 * y + 1] = buffer((g_base * fy) + 2 * y + 2, x);
            else
                line[2 * y + 1] = 0;

            __syncthreads();

            // reduction
            for (int d = blockDim.x; d > 0; d >>= 1)
            {
                __syncthreads();
                if (y < d)
                {
                    int ai = 2 * y * offset -1 + 1 * offset;
                    int bi = 2 * y * offset -1 + 2 * offset;
                    line[bi] += line[ai];
                }

                offset *= 2;
            }

            __syncthreads();
            // convolution
            if (y == 0) line[(blockDim.x << 1) - 1] = (W)buffer(0, x);

            for (int d = 1; d < (blockDim.x << 1); d *= 2)
            {
                offset >>= 1;

                __syncthreads();
                if (y < d)
                {
                    int ai = offset * 2 * y + 1 * offset - 1;
                    int bi = offset * 2 * y + 2 * offset - 1;


                    W t = line[ai];
                    line[ai] = line[bi];
                    line[bi] += t;
                }
            }
            __syncthreads();

            if (y < dst.rows)
            {
                W start = (y == 0)? (W)0:line[y * fy -1];
                W end = line[y * fy + fy - 1];
                dst(g_base +  y ,x) = saturate_cast<T>((end - start) * scale);
            }
        }

        template <typename T>
        __device__ void scan_x(const DevMem2D_<T> src, int fx, int fy, DevMem2D_<typename scan_traits<T>::scan_line_type> buffer,
                               typename scan_traits<T>::scan_line_type* line, int g_base)
        {
            typedef typename scan_traits<T>::scan_line_type W;

            const int x = threadIdx.x;
            const int y = blockIdx.x;

            float scale = 1.f / (fx * fy);

            if (y >= src.rows) return;

            int offset = 1;

            line[2 * x + 0] = (W)src(y, (g_base * fx) + 2 * x + 1);

            if (x != (blockDim.x -1) )
                line[2 * x + 1] = (W)src(y, (g_base * fx) + 2 * x + 2);
            else
                line[2 * x + 1] = 0;

            __syncthreads();

            // reduction
            for (int d = blockDim.x; d > 0; d >>= 1)
            {
                __syncthreads();
                if (x < d)
                {
                    int ai = 2 * x * offset -1 + 1 * offset;
                    int bi = 2 * x * offset -1 + 2 * offset;
                    line[bi] += line[ai];
                }

                offset *= 2;
            }

            __syncthreads();
            // convolution
            if (x == 0) line[(blockDim.x << 1) - 1] = (W)src(y, 0);

            for (int d = 1; d < (blockDim.x << 1); d *= 2)
            {
                offset >>= 1;

                __syncthreads();
                if (x < d)
                {
                    int ai = offset * 2 * x + 1 * offset - 1;
                    int bi = offset * 2 * x + 2 * offset - 1;

                    W t = line[ai];
                    line[ai] = line[bi];
                    line[bi] += t;
                }
            }
            __syncthreads();

            if (x < buffer.cols)
            {
                W start = (x == 0)? (W)0:line[x * fx -1];
                W end = line[x * fx + fx - 1];
                buffer(y, g_base +  x) =(end - start);
            }
        }

        enum ScanKind { exclusive,  inclusive } ;

        template <ScanKind Kind , class T>
        __device__ __forceinline__ T scan_warp ( volatile T *ptr , const unsigned int idx = threadIdx.x )
        {
            const unsigned int lane = idx & 31;

            if ( lane >=  1) ptr [idx ] = ptr [idx -  1] + ptr [idx];
            if ( lane >=  2) ptr [idx ] = ptr [idx -  2] + ptr [idx];
            if ( lane >=  4) ptr [idx ] = ptr [idx -  4] + ptr [idx];
            if ( lane >=  8) ptr [idx ] = ptr [idx -  8] + ptr [idx];
            if ( lane >= 16) ptr [idx ] = ptr [idx - 16] + ptr [idx];

            if( Kind == inclusive )
                return ptr [idx ];
            else
                return (lane > 0) ? ptr [idx - 1] : 0;
        }

        template <ScanKind Kind , class T>
        __device__ __forceinline__ T scan_block( volatile T *ptr)
        {
            const unsigned int idx = threadIdx.x;
            const unsigned int lane = idx & 31;
            const unsigned int warp = idx >> 5;

            T val = scan_warp <Kind>( ptr , idx );
            __syncthreads ();

            if( lane == 31 )
                ptr [ warp ] = ptr [idx ];

            __syncthreads ();

            if( warp == 0 )
                scan_warp<inclusive>( ptr , idx );

            __syncthreads ();

            if ( warp > 0)
                val = ptr [warp -1] + val;

            __syncthreads ();

            ptr[idx] = val;

            __syncthreads ();

            return val ;
        }

        template<typename T, typename W>
        __global__ void resise_scan_fast_x(const DevMem2D_<T> src, DevMem2D_<W> dst, int fx, int fy, int thred_lines, int stride)
        {
            extern __shared__ W sbuf[];

            const unsigned int tid = threadIdx. x;

            // load line-block on shared memory
            int y = blockIdx.x / thred_lines;
            int input_stride = (blockIdx.x % thred_lines) * stride;
            int x = input_stride  + tid;

            // store global data in shared memory
            if (x  < src.cols && y < src.rows)
                sbuf[tid] = src(y, x);
            else
                sbuf[tid] = 0;
            __syncthreads();

            scan_block<inclusive, W>(sbuf);

            float scale =  __fdividef(1.f, fx);
            int out_stride = input_stride / fx;
            int count = blockDim.x / fx;

            if (tid < count)
            {
                int start_idx = (tid == 0)? 0 : tid * fx - 1;
                int end_idx = tid * fx + fx - 1;

                W start = (tid == 0)? (W)0:sbuf[start_idx];
                W end = sbuf[end_idx];

                if (blockIdx.x == 0)
                    printf("%d~~~~~~~~ start_idx %d, end_idx %d, start %f, end %f\n",
                           tid, start_idx, end_idx, start, end);

                dst(y, out_stride  +  tid) = (end - start);
            }
        }

        template<typename T, typename W>
        __global__ void resise_scan_fast_y(const DevMem2D_<W> src, DevMem2D_<T> dst, int fx, int fy, int thred_lines, int stride)
        {
            extern __shared__ W sbuf[];

            const unsigned int tid = threadIdx. x;

            // load line-block on shared memory
            int x = blockIdx.x / thred_lines;

            int global_stride = (blockIdx.x % thred_lines) * stride;
            int y = global_stride + tid;

            // store global data in shared memory
            if (x  < src.cols && y < src.rows)
                sbuf[tid] = src(y, x);
            else
                sbuf[tid] = 0;

            __syncthreads();
            scan_block<inclusive, W>(sbuf);

            float scale =  __fdividef(1.f, fx * fy);
            int out_stride = global_stride / fx;
            int count = blockDim.x / fx;

            if (tid < count)
            {
                int start_idx = (tid == 0)? 0 : tid * fx - 1;
                int end_idx = tid * fx + fx - 1;

                W start = (tid == 0)? (W)0:sbuf[start_idx];
                W end = sbuf[end_idx];

                if (blockIdx.x == 0)
                    printf("!!!!!!!!%d~~~~~~~~ start_idx %d, end_idx %d, start %f, end %f\n",
                           tid, start_idx, end_idx, start, end);

                dst(out_stride  +  tid, x) = saturate_cast<T>((end - start) * scale);
            }
        }

        template <typename T>
        void resize_area_gpu(const DevMem2Db src, DevMem2Db dst,float fx, float fy,
                             int interpolation, DevMem2Df buffer, cudaStream_t stream)
        {
            (void)interpolation;

            int iscale_x = round(fx);
            int iscale_y = round(fy);

            int warps = 4;
            const int threads = 32 * warps;
            int input_stride = threads / iscale_x;

            int thred_lines = divUp(src.cols, input_stride * iscale_x);
            int blocks = src.rows * thred_lines;

            printf("device code executed for X coordinate with:\nsize %d warps %d, threads %d, thred_lines %d, blocks %d input strude %d\n",
                   src.cols, warps, threads, thred_lines, blocks, input_stride * iscale_x);

            typedef typename scan_traits<T>::scan_line_type smem_type;

            resise_scan_fast_x<T, smem_type><<<blocks, threads, warps * 32 * sizeof(smem_type)>>>
                    (src, buffer, iscale_x, iscale_y, thred_lines, input_stride * iscale_x);

            input_stride = threads / iscale_y;
            thred_lines = divUp(src.rows, input_stride * iscale_y);
            blocks = dst.cols * thred_lines;

            printf("device code executed for Y coordinate with:\nsize %d warps %d, threads %d, thred_lines %d, blocks %d\n",
                   dst.rows, warps, threads, thred_lines, blocks);

            resise_scan_fast_y<T, smem_type><<<blocks, threads, warps * 32 * sizeof(smem_type)>>>
                    (buffer, dst, iscale_x, iscale_y, thred_lines, input_stride * iscale_y);

            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall( cudaDeviceSynchronize() );
        }

        template void resize_area_gpu<uchar>(DevMem2Db src, DevMem2Db dst, float fx, float fy, int interpolation, DevMem2Df buffer, cudaStream_t stream);

    } // namespace imgproc
}}} // namespace cv { namespace gpu { namespace device
