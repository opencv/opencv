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

#include <cstring>
#include "opencv2/gpu/device/common.hpp"
#include "opencv2/gpu/device/transform.hpp"
#include "opencv2/gpu/device/functional.hpp"

using namespace cv::gpu;
using namespace cv::gpu::device;

namespace
{
    texture<uchar, cudaTextureType1D, cudaReadModeElementType> texLutTable;

    struct LutC1 : public unary_function<uchar, uchar>
    {
        typedef uchar value_type;
        typedef uchar index_type;

        cudaTextureObject_t texLutTableObj;

        __device__ __forceinline__ uchar operator ()(uchar x) const
        {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 300)
            // Use the texture reference
            return tex1Dfetch(texLutTable, x);
        #else
            // Use the texture object
            return tex1Dfetch<uchar>(texLutTableObj, x);
        #endif
        }
    };
    struct LutC3 : public unary_function<uchar3, uchar3>
    {
        typedef uchar3 value_type;
        typedef uchar3 index_type;

        cudaTextureObject_t texLutTableObj;

        __device__ __forceinline__ uchar3 operator ()(const uchar3& x) const
        {
        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 300)
            // Use the texture reference
            return make_uchar3(tex1Dfetch(texLutTable, x.x * 3), tex1Dfetch(texLutTable, x.y * 3 + 1), tex1Dfetch(texLutTable, x.z * 3 + 2));
        #else
            // Use the texture object
            return make_uchar3(tex1Dfetch<uchar>(texLutTableObj, x.x * 3), tex1Dfetch<uchar>(texLutTableObj, x.y * 3 + 1), tex1Dfetch<uchar>(texLutTableObj, x.z * 3 + 2));
        #endif
        }
    };
}

namespace arithm
{
    void lut(PtrStepSzb src, uchar* lut, int lut_cn, PtrStepSzb dst, bool cc30, cudaStream_t stream)
    {
        cudaTextureObject_t texLutTableObj;

        if (cc30)
        {
            // Use the texture object
            cudaResourceDesc texRes;
            std::memset(&texRes, 0, sizeof(texRes));
            texRes.resType = cudaResourceTypeLinear;
            texRes.res.linear.devPtr = lut;
            texRes.res.linear.desc = cudaCreateChannelDesc<uchar>();
            texRes.res.linear.sizeInBytes = 256 * lut_cn * sizeof(uchar);

            cudaTextureDesc texDescr;
            std::memset(&texDescr, 0, sizeof(texDescr));

            cudaSafeCall( cudaCreateTextureObject(&texLutTableObj, &texRes, &texDescr, 0) );
        }
        else
        {
            // Use the texture reference
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
            cudaSafeCall( cudaBindTexture(0, &texLutTable, lut, &desc) );
        }

        if (lut_cn == 1)
        {
            LutC1 op;
            op.texLutTableObj = texLutTableObj;

            transform((PtrStepSz<uchar>) src, (PtrStepSz<uchar>) dst, op, WithOutMask(), stream);
        }
        else if (lut_cn == 3)
        {
            LutC3 op;
            op.texLutTableObj = texLutTableObj;

            transform((PtrStepSz<uchar3>) src, (PtrStepSz<uchar3>) dst, op, WithOutMask(), stream);
        }

        if (cc30)
        {
            // Use the texture object
            cudaSafeCall( cudaDestroyTextureObject(texLutTableObj) );
        }
        else
        {
            // Use the texture reference
            cudaSafeCall( cudaUnbindTexture(texLutTable) );
        }
    }
}

#endif
