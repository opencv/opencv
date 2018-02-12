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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Copyright (c) 2016-2017 Fabian David Tschopp, all rights reserved.
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

#include "../../precomp.hpp"
#include "common.hpp"
#include "math_functions.hpp"
#include <vector>
#include "opencl_kernels_dnn.hpp"

namespace cv
{
namespace dnn
{
namespace ocl4dnn
{

#ifdef HAVE_OPENCL
// Create and copy buffer to image for GEMM's matrix A and B.
// Will return image to caller if the input image is NULL. Otherwise,
// will use the image directly. It's caller's responsibility to
// release the created image.
template<typename Dtype>
ocl::Image2D ocl4dnnGEMMCopyBufferToImage(UMat buffer, int offset,
                                          bool is_matrix_a, bool transpose,
                                          bool padding, int padded_height,
                                          int padded_width, int height,
                                          int width, int ld)
{
    ocl::Image2D image;

    if (!is_matrix_a && transpose)
    {
        if (ld == width)
        {
            image = ocl::Image2D(buffer);
        } else {
            // For matrix B with transpose, we need to handle them differently.
            // As we can't use the sub group block read to get a row easily,
            // we have to use CL_FLOAT type with read_imagef to get the row.
            UMat mat(height, width, CV_32FC1);
            image = ocl::Image2D(mat);

            ocl::Kernel oclk_gemm_copy("gemm_buffer_copy_image_transpose_float", ocl::dnn::gemm_image_oclsrc);

            size_t global_copy[2];
            global_copy[0] = width;
            global_copy[1] = height;
            oclk_gemm_copy.set(0, ocl::KernelArg::PtrReadOnly(buffer));
            oclk_gemm_copy.set(1, image);
            oclk_gemm_copy.set(2, offset);
            oclk_gemm_copy.set(3, width);
            oclk_gemm_copy.set(4, height);
            oclk_gemm_copy.set(5, ld);
            oclk_gemm_copy.run(2, global_copy, NULL, false);
        }
    } else {
        if (!padding)
        {
            // copy without padding.
            image = ocl::Image2D(buffer);
        } else {
            UMat mat(padded_height, padded_width, CV_8UC4);
            image = ocl::Image2D(mat);

            ocl::Kernel oclk_gemm_copy("gemm_buffer_copy_image_no_transpose_float",
                                       ocl::dnn::gemm_image_oclsrc);

            size_t global_copy[2];
            global_copy[0] = padded_width;
            global_copy[1] = padded_height;

            oclk_gemm_copy.set(0, ocl::KernelArg::PtrReadOnly(buffer));
            oclk_gemm_copy.set(1, image);
            oclk_gemm_copy.set(2, offset);
            oclk_gemm_copy.set(3, width);
            oclk_gemm_copy.set(4, height);
            oclk_gemm_copy.set(5, ld);

            oclk_gemm_copy.run(2, global_copy, NULL, false);
        }
    }

    return image;
}

template
ocl::Image2D ocl4dnnGEMMCopyBufferToImage<float>(UMat buffer, int offset,
                                                 bool is_matrix_a, bool transpose,
                                                 bool padding, int padded_height,
                                                 int padded_width, int height,
                                                 int width,  int ld);

enum gemm_type_t
{
    GEMM_TYPE_NONE = 0,
    GEMM_TYPE_FAST_IMAGE_32_1,
    GEMM_TYPE_FAST_IMAGE_32_2,
    GEMM_TYPE_FAST_IMAGE_B_IMAGE,
    GEMM_TYPE_MAX
};

template<typename Dtype>
static bool ocl4dnnFastImageGEMM(const CBLAS_TRANSPOSE TransA,
                                 const CBLAS_TRANSPOSE TransB, const int32_t M,
                                 const int32_t N, const int32_t K, const Dtype alpha,
                                 const UMat A, const int32_t offA, const UMat B,
                                 const int32_t offB, const Dtype beta, UMat C,
                                 const int32_t offC, bool is_image_a, bool is_image_b,
                                 enum gemm_type_t gemm_type,
                                 const size_t max_image_size)
{
    CHECK_EQ(gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 || gemm_type == GEMM_TYPE_FAST_IMAGE_32_2 ||
             gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE, true) << "Invalid fast image gemm type." << std::endl;

    if (is_image_a)
    {
        CHECK_EQ(offA, 0) << "Invalid input image offset." << std::endl;
        return false;
    }

    if (is_image_b)
    {
        CHECK_EQ(offB, 0) << "Invalid input image offset." << std::endl;
        return false;
    }

    int widthA = (TransA == CblasNoTrans) ? K : M;
    int heightA = (TransA == CblasNoTrans) ? M : K;
    int widthB = (TransB == CblasNoTrans) ? N : K;
    int heightB = (TransB == CblasNoTrans) ? K : N;

    int ldA = widthA;
    int ldB = widthB;
    int ldC = N;

    int A_start_x = 0, A_start_y = 0, B_start_x = 0;
    int B_start_y = 0, C_start_x = 0, C_start_y = 0;
    int blocksize = 1024;
    if (gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE)
        blocksize = max_image_size;
    int blockA_width = blocksize;
    int blockA_height = blocksize;
    int blockB_width = blocksize;
    int blockB_height = blocksize;
    int blockC_width = blocksize;
    int blockC_height = blocksize;

    int use_buffer_indicator = 8;
    // To fix the edge problem caused by the sub group block read.
    // we have to pad the image if it's not multiple of tile.
    // just padding one line is enough as the sub group block read
    // will clamp to edge according to the spec.

    ocl::Image2D ImA;
    ocl::Image2D ImB;

    std::string kernel_name("gemm_");
    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 || gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE)
        kernel_name += "32_1_";
    else
        kernel_name += "32_2_";

    if (TransA == CblasNoTrans)
        kernel_name += "N";
    else
        kernel_name += "T";

    if (TransB == CblasNoTrans)
    {
        kernel_name += "N_";
    } else {
        kernel_name += "T_";
        if (is_image_b || (K % use_buffer_indicator != 0))
        {
            kernel_name += "SCALAR_";
        } else {
            kernel_name += "BUFFER_";
        }
    }

    if (alpha == 1)
        kernel_name += "1_";
    else
        kernel_name += "0_";

    if (beta == 0)
        kernel_name += "0";
    else
        kernel_name += "1";

    kernel_name += "_float";

    ocl::Kernel oclk_gemm_float(kernel_name.c_str(), ocl::dnn::gemm_image_oclsrc);
    if (oclk_gemm_float.empty())
        return false;

    while (C_start_y < M)
    {
        blockC_width = std::min(static_cast<int>(N) - C_start_x, blocksize);
        blockC_height = std::min(static_cast<int>(M) - C_start_y, blocksize);

        int isFirstColBlock = 1;
        for (int k = 0; k < K; k += blocksize)
        {
            blockA_width = std::min(widthA - A_start_x, blocksize);
            blockA_height = std::min(heightA - A_start_y, blocksize);
            blockB_width = std::min(widthB - B_start_x, blocksize);
            blockB_height = std::min(heightB - B_start_y, blocksize);
            int block_Ksize = std::min(static_cast<int>(K) - k, blocksize);

            int padded_k = block_Ksize + ((block_Ksize & 7) ? (8 - (block_Ksize & 7)) : 0);
            int imageA_w = (TransA == CblasNoTrans) ? padded_k : blockA_width;
            int imageA_h = (TransA == CblasNoTrans) ? blockA_height : padded_k;
            int imageB_w = (TransB == CblasNoTrans) ? blockB_width : padded_k;
            int imageB_h = (TransB == CblasNoTrans) ? padded_k : blockB_height;

            int blockA_offset = offA + A_start_y * ldA + A_start_x;
            int blockB_offset = offB + B_start_y * ldB + B_start_x;
            int blockC_offset = offC + C_start_y * ldC + C_start_x;
            if (TransB == CblasNoTrans)
            {
                bool padding_A = false;
                bool padding_B = false;

                if (!is_image_a && !is_image_b)
                {
                    if (M * K < N * K)
                        padding_B = true;
                    else
                        padding_A = true;
                }

                if (!is_image_a)
                {
                    ImA = ocl4dnnGEMMCopyBufferToImage<Dtype>(A, blockA_offset,
                                                              true, TransA != CblasNoTrans,
                                                              padding_A, imageA_h, imageA_w,
                                                              blockA_height, blockA_width, ldA);
                }
                if (!is_image_b)
                {
                    ImB = ocl4dnnGEMMCopyBufferToImage<Dtype>(B, blockB_offset,
                                                              false, false,
                                                              padding_B, imageB_h, imageB_w,
                                                              blockB_height, blockB_width, ldB);
                }
            } else {
                // We will use normal read_imagef to read image B when B has transpose.
                // thus we don't need to pad image A at all.
                if (!is_image_a)
                {
                    bool padding;
                    padding = !is_image_b;
                    ImA = ocl4dnnGEMMCopyBufferToImage<Dtype>(A, blockA_offset,
                                                              true, TransA != CblasNoTrans,
                                                              padding, imageA_h, imageA_w,
                                                              blockA_height, blockA_width, ldA);
                }

                if (!is_image_b && (K % use_buffer_indicator != 0))
                {
                    ImB = ocl4dnnGEMMCopyBufferToImage<Dtype>(B, blockB_offset,
                                                              false, true, false, imageB_h, imageB_w,
                                                              blockB_height, blockB_width, ldB);
                }
            }

            size_t global[2];
            if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 || gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE)
            {
                global[0] = (size_t)( blockC_width + 7 ) & ~7;
            } else {
                global[0] = (size_t)( (blockC_width / 2 ) + 7 ) ^ ~7;
            }
            global[1] = (size_t)(blockC_height + 31) / 32;

            size_t local[2];
            local[0] = 8;
            local[1] = 1;

            cl_uint arg_idx = 0;
            if (is_image_a)
                oclk_gemm_float.set(arg_idx++, ocl::KernelArg::PtrReadOnly(A));
            else
                oclk_gemm_float.set(arg_idx++, ImA);

            if (TransB == CblasNoTrans || is_image_b || (K % use_buffer_indicator != 0))
            {
                if (is_image_b)
                    oclk_gemm_float.set(arg_idx++, ocl::KernelArg::PtrReadOnly(B));
                else
                    oclk_gemm_float.set(arg_idx++, ImB);
            } else {
                oclk_gemm_float.set(arg_idx++, ocl::KernelArg::PtrReadOnly(B));
                oclk_gemm_float.set(arg_idx++, blockB_offset);
                oclk_gemm_float.set(arg_idx++, ldB);
            }
            oclk_gemm_float.set(arg_idx++, ocl::KernelArg::PtrWriteOnly(C));
            oclk_gemm_float.set(arg_idx++, blockC_offset);
            oclk_gemm_float.set(arg_idx++, blockC_height);
            oclk_gemm_float.set(arg_idx++, blockC_width);
            oclk_gemm_float.set(arg_idx++, ldC);
            oclk_gemm_float.set(arg_idx++, alpha);
            oclk_gemm_float.set(arg_idx++, beta);
            oclk_gemm_float.set(arg_idx++, padded_k);
            if (TransB != CblasNoTrans)
                oclk_gemm_float.set(arg_idx++, block_Ksize);
            oclk_gemm_float.set(arg_idx++, isFirstColBlock);

            if (!oclk_gemm_float.run(2, global, local, false))
                return false;

            if (TransA == CblasNoTrans)
                A_start_x += blockA_width;
            else
                A_start_y += blockA_height;

            if (TransB == CblasNoTrans)
                B_start_y += blockB_height;
            else
                B_start_x += blockB_width;

            isFirstColBlock = 0;
        }

        C_start_x += blockC_width;
        if (TransA == CblasNoTrans)
            A_start_x = 0;
        else
            A_start_y = 0;
        if (TransB == CblasNoTrans)
        {
            B_start_x += blockB_width;
            B_start_y = 0;
        } else {
            B_start_y += blockB_height;
            B_start_x = 0;
        }
        if (C_start_x >= N)
        {
            C_start_x = 0;
            B_start_x = 0;
            B_start_y = 0;
            C_start_y += blockC_height;
            if (TransA == CblasNoTrans)
                A_start_y += blockA_height;
            else
                A_start_x += blockA_width;
        }
    }

    return true;
}

template<typename Dtype>
bool ocl4dnnGEMMCommon(const CBLAS_TRANSPOSE TransB,
                       const int32_t M, const int32_t N, const int32_t K,
                       const UMat A, const UMat B,
                       const UMat B_image, UMat C,
                       const size_t max_image_size)
{
    gemm_type_t gemm_type = GEMM_TYPE_FAST_IMAGE_32_1;

    if (gemm_type == GEMM_TYPE_FAST_IMAGE_32_1 ||
        gemm_type == GEMM_TYPE_FAST_IMAGE_32_2)
    {
        return ocl4dnnFastImageGEMM<Dtype>(CblasNoTrans, TransB, M, N, K,
                                           (Dtype)1., A, 0, B, 0, (Dtype)0., C,
                                           0, false, false, gemm_type, max_image_size);
    }
    else if (gemm_type == GEMM_TYPE_FAST_IMAGE_B_IMAGE)
    {
        return ocl4dnnFastImageGEMM<Dtype>(CblasNoTrans, TransB, M, N, K,
                                           (Dtype)1., A, 0, B_image, 0, (Dtype)0., C,
                                           0, false, true,
                                           GEMM_TYPE_FAST_IMAGE_B_IMAGE,
                                           max_image_size);
    }
    return false;
}

template bool ocl4dnnGEMMCommon<float>(const CBLAS_TRANSPOSE TransB,
                                       const int32_t M, const int32_t N, const int32_t K,
                                       const UMat A, const UMat B,
                                       const UMat B_image, UMat C,
                                       const size_t max_image_size);

template<typename Dtype>
bool ocl4dnnGEMV(const CBLAS_TRANSPOSE TransA,
                 const int32_t M, const int32_t N, const Dtype alpha,
                 const UMat A, const int32_t offA, const UMat x,
                 const int32_t offx, const Dtype beta, UMat y,
                 const int32_t offy)
{
    return false;
}

template<>
bool ocl4dnnGEMV<float>(const CBLAS_TRANSPOSE TransA,
                 const int32_t M, const int32_t N, const float alpha,
                 const UMat A, const int32_t offA, const UMat x,
                 const int32_t offx, const float beta, UMat y,
                 const int32_t offy)
{
    bool ret = false;

    if (TransA == CblasNoTrans)
    {
        ocl::Kernel k(CL_KERNEL_SELECT("matvec_mul4"), cv::ocl::dnn::matvec_mul_oclsrc);
        if (k.empty())
            return false;

        uint row_size = M;
        uint col_size = N;

        if (row_size >= 4)
        {
            size_t localsize[] = { 128 };
            size_t globalsize[] = { row_size / 4 * localsize[0] };

            uint argId = 0;
            k.set(argId++, ocl::KernelArg::PtrReadOnly(A));
            k.set(argId++, offA);
            k.set(argId++, cl_uint(col_size));
            k.set(argId++, cl_uint(col_size%4));
            k.set(argId++, ocl::KernelArg::PtrReadOnly(x));
            k.set(argId++, offx);
            k.set(argId++, alpha);
            k.set(argId++, beta);
            k.set(argId++, ocl::KernelArg::PtrWriteOnly(y));
            k.set(argId++, offy);
            k.set(argId++, NULL, localsize[0] * sizeof(cl_float4));

            ret = k.run(1, globalsize, localsize, false);
        }

        if ((row_size % 4) != 0 && ret)
        {
            ocl::Kernel k_1(CL_KERNEL_SELECT("matvec_mul1"), cv::ocl::dnn::matvec_mul_oclsrc);
            size_t localsize[] = { 128 };
            size_t globalsize[] = { row_size % 4 * localsize[0] };
            uint row_offset = row_size - (row_size % 4);

            uint argId = 0;
            k_1.set(argId++, ocl::KernelArg::PtrReadOnly(A));
            k_1.set(argId++, offA);
            k_1.set(argId++, cl_uint(col_size));
            k_1.set(argId++, cl_uint(row_offset));
            k_1.set(argId++, cl_uint(col_size%4));
            k_1.set(argId++, ocl::KernelArg::PtrReadOnly(x));
            k_1.set(argId++, offx);
            k_1.set(argId++, alpha);
            k_1.set(argId++, beta);
            k_1.set(argId++, ocl::KernelArg::PtrWriteOnly(y));
            k_1.set(argId++, offy);
            k_1.set(argId++, NULL, localsize[0] * sizeof(cl_float));

            ret = k_1.run(1, globalsize, localsize, false);
        }
    }
    return ret;
}

template<typename Dtype>
bool ocl4dnnAXPY(const int32_t N, const Dtype alpha,
                 const UMat X, const int32_t offX, UMat Y,
                 const int32_t offY)
{
    ocl::Kernel oclk_axpy(CL_KERNEL_SELECT("axpy"), cv::ocl::dnn::math_oclsrc);
    if (oclk_axpy.empty())
        return false;

    size_t global[] = { 128 * 128 };
    size_t local[] = { 128 };

    cl_uint argIdx = 0;
    oclk_axpy.set(argIdx++, N);
    oclk_axpy.set(argIdx++, alpha);
    oclk_axpy.set(argIdx++, ocl::KernelArg::PtrReadOnly(X));
    oclk_axpy.set(argIdx++, offX);
    oclk_axpy.set(argIdx++, ocl::KernelArg::PtrWriteOnly(Y));
    oclk_axpy.set(argIdx++, offY);

    return oclk_axpy.run(1, global, local, false);
}

template bool ocl4dnnAXPY<float>(const int32_t N, const float alpha,
                                 const UMat X, const int32_t offX,
                                 UMat Y, const int32_t offY);

#endif  // HAVE_OPENCL

} // namespace ocl4dnn
} // namespace dnn
} // namespce cv
