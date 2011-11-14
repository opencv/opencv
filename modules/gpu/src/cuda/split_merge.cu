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

#include "internal_shared.hpp"

namespace cv { namespace gpu { namespace device 
{
    namespace split_merge 
    {
        template <typename T, size_t elem_size = sizeof(T)>
        struct TypeTraits 
        {
            typedef T type;
            typedef T type2;
            typedef T type3;
            typedef T type4;
        };

        template <typename T>
        struct TypeTraits<T, 1>
        {
            typedef char type;
            typedef char2 type2;
            typedef char3 type3;
            typedef char4 type4;
        };

        template <typename T>
        struct TypeTraits<T, 2>
        {
            typedef short type;
            typedef short2 type2;
            typedef short3 type3;
            typedef short4 type4;
        };

        template <typename T>
        struct TypeTraits<T, 4> 
        {
            typedef int type;
            typedef int2 type2;
            typedef int3 type3;
            typedef int4 type4;
        };

        template <typename T>
        struct TypeTraits<T, 8> 
        {
            typedef double type;
            typedef double2 type2;
            //typedef double3 type3;
            //typedef double4 type3;
        };

        typedef void (*MergeFunction)(const DevMem2Db* src, DevMem2Db& dst, const cudaStream_t& stream);
        typedef void (*SplitFunction)(const DevMem2Db& src, DevMem2Db* dst, const cudaStream_t& stream);

        //------------------------------------------------------------
        // Merge    

        template <typename T>
        __global__ void mergeC2_(const uchar* src0, size_t src0_step, 
                                 const uchar* src1, size_t src1_step, 
                                 int rows, int cols, uchar* dst, size_t dst_step)
        {
            typedef typename TypeTraits<T>::type2 dst_type;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const T* src0_y = (const T*)(src0 + y * src0_step);
            const T* src1_y = (const T*)(src1 + y * src1_step);
            dst_type* dst_y = (dst_type*)(dst + y * dst_step);

            if (x < cols && y < rows) 
            {                        
                dst_type dst_elem;
                dst_elem.x = src0_y[x];
                dst_elem.y = src1_y[x];
                dst_y[x] = dst_elem;
            }
        }


        template <typename T>
        __global__ void mergeC3_(const uchar* src0, size_t src0_step, 
                                 const uchar* src1, size_t src1_step, 
                                 const uchar* src2, size_t src2_step, 
                                 int rows, int cols, uchar* dst, size_t dst_step)
        {
            typedef typename TypeTraits<T>::type3 dst_type;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const T* src0_y = (const T*)(src0 + y * src0_step);
            const T* src1_y = (const T*)(src1 + y * src1_step);
            const T* src2_y = (const T*)(src2 + y * src2_step);
            dst_type* dst_y = (dst_type*)(dst + y * dst_step);

            if (x < cols && y < rows) 
            {                        
                dst_type dst_elem;
                dst_elem.x = src0_y[x];
                dst_elem.y = src1_y[x];
                dst_elem.z = src2_y[x];
                dst_y[x] = dst_elem;
            }
        }


        template <>
        __global__ void mergeC3_<double>(const uchar* src0, size_t src0_step, 
                                 const uchar* src1, size_t src1_step, 
                                 const uchar* src2, size_t src2_step, 
                                 int rows, int cols, uchar* dst, size_t dst_step)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const double* src0_y = (const double*)(src0 + y * src0_step);
            const double* src1_y = (const double*)(src1 + y * src1_step);
            const double* src2_y = (const double*)(src2 + y * src2_step);
            double* dst_y = (double*)(dst + y * dst_step);

            if (x < cols && y < rows) 
            {                        
                dst_y[3 * x] = src0_y[x];
                dst_y[3 * x + 1] = src1_y[x];
                dst_y[3 * x + 2] = src2_y[x];
            }
        }


        template <typename T>
        __global__ void mergeC4_(const uchar* src0, size_t src0_step, 
                                 const uchar* src1, size_t src1_step, 
                                 const uchar* src2, size_t src2_step, 
                                 const uchar* src3, size_t src3_step, 
                                 int rows, int cols, uchar* dst, size_t dst_step)
        {
            typedef typename TypeTraits<T>::type4 dst_type;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const T* src0_y = (const T*)(src0 + y * src0_step);
            const T* src1_y = (const T*)(src1 + y * src1_step);
            const T* src2_y = (const T*)(src2 + y * src2_step);
            const T* src3_y = (const T*)(src3 + y * src3_step);
            dst_type* dst_y = (dst_type*)(dst + y * dst_step);

            if (x < cols && y < rows) 
            {                        
                dst_type dst_elem;
                dst_elem.x = src0_y[x];
                dst_elem.y = src1_y[x];
                dst_elem.z = src2_y[x];
                dst_elem.w = src3_y[x];
                dst_y[x] = dst_elem;
            }
        }


        template <>
        __global__ void mergeC4_<double>(const uchar* src0, size_t src0_step, 
                                 const uchar* src1, size_t src1_step, 
                                 const uchar* src2, size_t src2_step, 
                                 const uchar* src3, size_t src3_step, 
                                 int rows, int cols, uchar* dst, size_t dst_step)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const double* src0_y = (const double*)(src0 + y * src0_step);
            const double* src1_y = (const double*)(src1 + y * src1_step);
            const double* src2_y = (const double*)(src2 + y * src2_step);
            const double* src3_y = (const double*)(src3 + y * src3_step);
            double2* dst_y = (double2*)(dst + y * dst_step);

            if (x < cols && y < rows) 
            {                        
                dst_y[2 * x] = make_double2(src0_y[x], src1_y[x]);
                dst_y[2 * x + 1] = make_double2(src2_y[x], src3_y[x]);
            }
        }


        template <typename T>
        static void mergeC2_(const DevMem2Db* src, DevMem2Db& dst, const cudaStream_t& stream)
        {
            dim3 blockDim(32, 8);
            dim3 gridDim(divUp(dst.cols, blockDim.x), divUp(dst.rows, blockDim.y));
            mergeC2_<T><<<gridDim, blockDim, 0, stream>>>(
                    src[0].data, src[0].step,
                    src[1].data, src[1].step,
                    dst.rows, dst.cols, dst.data, dst.step);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        template <typename T>
        static void mergeC3_(const DevMem2Db* src, DevMem2Db& dst, const cudaStream_t& stream)
        {
            dim3 blockDim(32, 8);
            dim3 gridDim(divUp(dst.cols, blockDim.x), divUp(dst.rows, blockDim.y));
            mergeC3_<T><<<gridDim, blockDim, 0, stream>>>(
                    src[0].data, src[0].step,
                    src[1].data, src[1].step,
                    src[2].data, src[2].step,
                    dst.rows, dst.cols, dst.data, dst.step);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        template <typename T>
        static void mergeC4_(const DevMem2Db* src, DevMem2Db& dst, const cudaStream_t& stream)
        {
            dim3 blockDim(32, 8);
            dim3 gridDim(divUp(dst.cols, blockDim.x), divUp(dst.rows, blockDim.y));
            mergeC4_<T><<<gridDim, blockDim, 0, stream>>>(
                    src[0].data, src[0].step,
                    src[1].data, src[1].step,
                    src[2].data, src[2].step,
                    src[3].data, src[3].step,
                    dst.rows, dst.cols, dst.data, dst.step);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        void merge_caller(const DevMem2Db* src, DevMem2Db& dst,
                                     int total_channels, size_t elem_size,
                                     const cudaStream_t& stream)
        {
            static MergeFunction merge_func_tbl[] =
            {
                mergeC2_<char>, mergeC2_<short>, mergeC2_<int>, 0, mergeC2_<double>,
                mergeC3_<char>, mergeC3_<short>, mergeC3_<int>, 0, mergeC3_<double>,
                mergeC4_<char>, mergeC4_<short>, mergeC4_<int>, 0, mergeC4_<double>,
            };

            size_t merge_func_id = (total_channels - 2) * 5 + (elem_size >> 1);
            MergeFunction merge_func = merge_func_tbl[merge_func_id];

            if (merge_func == 0)
                cv::gpu::error("Unsupported channel count or data type", __FILE__, __LINE__);

            merge_func(src, dst, stream);
        }



        //------------------------------------------------------------
        // Split


        template <typename T>
        __global__ void splitC2_(const uchar* src, size_t src_step, 
                                int rows, int cols,
                                uchar* dst0, size_t dst0_step,
                                uchar* dst1, size_t dst1_step)
        {
            typedef typename TypeTraits<T>::type2 src_type;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const src_type* src_y = (const src_type*)(src + y * src_step);
            T* dst0_y = (T*)(dst0 + y * dst0_step);
            T* dst1_y = (T*)(dst1 + y * dst1_step);

            if (x < cols && y < rows) 
            {
                src_type src_elem = src_y[x];
                dst0_y[x] = src_elem.x;
                dst1_y[x] = src_elem.y;
            }
        }


        template <typename T>
        __global__ void splitC3_(const uchar* src, size_t src_step, 
                                int rows, int cols,
                                uchar* dst0, size_t dst0_step,
                                uchar* dst1, size_t dst1_step,
                                uchar* dst2, size_t dst2_step)
        {
            typedef typename TypeTraits<T>::type3 src_type;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const src_type* src_y = (const src_type*)(src + y * src_step);
            T* dst0_y = (T*)(dst0 + y * dst0_step);
            T* dst1_y = (T*)(dst1 + y * dst1_step);
            T* dst2_y = (T*)(dst2 + y * dst2_step);

            if (x < cols && y < rows) 
            {
                src_type src_elem = src_y[x];
                dst0_y[x] = src_elem.x;
                dst1_y[x] = src_elem.y;
                dst2_y[x] = src_elem.z;
            }
        }


        template <>
        __global__ void splitC3_<double>(
                const uchar* src, size_t src_step, int rows, int cols,
                uchar* dst0, size_t dst0_step,
                uchar* dst1, size_t dst1_step,
                uchar* dst2, size_t dst2_step)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const double* src_y = (const double*)(src + y * src_step);
            double* dst0_y = (double*)(dst0 + y * dst0_step);
            double* dst1_y = (double*)(dst1 + y * dst1_step);
            double* dst2_y = (double*)(dst2 + y * dst2_step);

            if (x < cols && y < rows) 
            {
                dst0_y[x] = src_y[3 * x];
                dst1_y[x] = src_y[3 * x + 1];
                dst2_y[x] = src_y[3 * x + 2];
            }
        }


        template <typename T>
        __global__ void splitC4_(const uchar* src, size_t src_step, int rows, int cols,
                                uchar* dst0, size_t dst0_step,
                                uchar* dst1, size_t dst1_step,
                                uchar* dst2, size_t dst2_step,
                                uchar* dst3, size_t dst3_step)
        {
            typedef typename TypeTraits<T>::type4 src_type;

            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const src_type* src_y = (const src_type*)(src + y * src_step);
            T* dst0_y = (T*)(dst0 + y * dst0_step);
            T* dst1_y = (T*)(dst1 + y * dst1_step);
            T* dst2_y = (T*)(dst2 + y * dst2_step);
            T* dst3_y = (T*)(dst3 + y * dst3_step);

            if (x < cols && y < rows) 
            {
                src_type src_elem = src_y[x];
                dst0_y[x] = src_elem.x;
                dst1_y[x] = src_elem.y;
                dst2_y[x] = src_elem.z;
                dst3_y[x] = src_elem.w;
            }
        }


        template <>
        __global__ void splitC4_<double>(
                const uchar* src, size_t src_step, int rows, int cols,
                uchar* dst0, size_t dst0_step,
                uchar* dst1, size_t dst1_step,
                uchar* dst2, size_t dst2_step,
                uchar* dst3, size_t dst3_step)
        {
            const int x = blockIdx.x * blockDim.x + threadIdx.x;
            const int y = blockIdx.y * blockDim.y + threadIdx.y;

            const double2* src_y = (const double2*)(src + y * src_step);
            double* dst0_y = (double*)(dst0 + y * dst0_step);
            double* dst1_y = (double*)(dst1 + y * dst1_step);
            double* dst2_y = (double*)(dst2 + y * dst2_step);
            double* dst3_y = (double*)(dst3 + y * dst3_step);

            if (x < cols && y < rows) 
            {
                double2 src_elem1 = src_y[2 * x];
                double2 src_elem2 = src_y[2 * x + 1];
                dst0_y[x] = src_elem1.x;
                dst1_y[x] = src_elem1.y;
                dst2_y[x] = src_elem2.x;
                dst3_y[x] = src_elem2.y;
            }
        }

        template <typename T>
        static void splitC2_(const DevMem2Db& src, DevMem2Db* dst, const cudaStream_t& stream)
        {
            dim3 blockDim(32, 8);
            dim3 gridDim(divUp(src.cols, blockDim.x), divUp(src.rows, blockDim.y));
            splitC2_<T><<<gridDim, blockDim, 0, stream>>>(
                    src.data, src.step, src.rows, src.cols,
                    dst[0].data, dst[0].step,
                    dst[1].data, dst[1].step);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        template <typename T>
        static void splitC3_(const DevMem2Db& src, DevMem2Db* dst, const cudaStream_t& stream)
        {
            dim3 blockDim(32, 8);
            dim3 gridDim(divUp(src.cols, blockDim.x), divUp(src.rows, blockDim.y));
            splitC3_<T><<<gridDim, blockDim, 0, stream>>>(
                    src.data, src.step, src.rows, src.cols,
                    dst[0].data, dst[0].step,
                    dst[1].data, dst[1].step,
                    dst[2].data, dst[2].step);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        template <typename T>
        static void splitC4_(const DevMem2Db& src, DevMem2Db* dst, const cudaStream_t& stream)
        {
            dim3 blockDim(32, 8);
            dim3 gridDim(divUp(src.cols, blockDim.x), divUp(src.rows, blockDim.y));
            splitC4_<T><<<gridDim, blockDim, 0, stream>>>(
                     src.data, src.step, src.rows, src.cols,
                     dst[0].data, dst[0].step,
                     dst[1].data, dst[1].step,
                     dst[2].data, dst[2].step,
                     dst[3].data, dst[3].step);
            cudaSafeCall( cudaGetLastError() );

            if (stream == 0)
                cudaSafeCall(cudaDeviceSynchronize());
        }


        void split_caller(const DevMem2Db& src, DevMem2Db* dst, int num_channels, size_t elem_size1, const cudaStream_t& stream)
        {
            static SplitFunction split_func_tbl[] =
            {
                splitC2_<char>, splitC2_<short>, splitC2_<int>, 0, splitC2_<double>,
                splitC3_<char>, splitC3_<short>, splitC3_<int>, 0, splitC3_<double>,
                splitC4_<char>, splitC4_<short>, splitC4_<int>, 0, splitC4_<double>,
            };

            size_t split_func_id = (num_channels - 2) * 5 + (elem_size1 >> 1);
            SplitFunction split_func = split_func_tbl[split_func_id];

            if (split_func == 0)
                cv::gpu::error("Unsupported channel count or data type", __FILE__, __LINE__);

            split_func(src, dst, stream);
        }
    } // namespace split_merge
}}} // namespace cv { namespace gpu { namespace device
