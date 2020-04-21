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

#include "opencv2/opencv_modules.hpp"

#ifndef HAVE_OPENCV_CUDEV

#error "opencv_cudev is required"

#else

#include "../lut.hpp"

#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudev.hpp"
#include "opencv2/core/private.cuda.hpp"

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

namespace cv { namespace cuda {

    texture<uchar, cudaTextureType1D, cudaReadModeElementType> texLutTable;

    LookUpTableImpl::LookUpTableImpl(InputArray _lut)
    {
        if (_lut.kind() == _InputArray::CUDA_GPU_MAT)
        {
            d_lut = _lut.getGpuMat();
        }
        else
        {
            Mat h_lut = _lut.getMat();
            d_lut.upload(Mat(1, 256, h_lut.type(), h_lut.data));
        }

        CV_Assert( d_lut.depth() == CV_8U );
        CV_Assert( d_lut.rows == 1 && d_lut.cols == 256 );

        cc30 = deviceSupports(FEATURE_SET_COMPUTE_30);

        if (cc30)
        {
            // Use the texture object
            cudaResourceDesc texRes;
            std::memset(&texRes, 0, sizeof(texRes));
            texRes.resType = cudaResourceTypeLinear;
            texRes.res.linear.devPtr = d_lut.data;
            texRes.res.linear.desc = cudaCreateChannelDesc<uchar>();
            texRes.res.linear.sizeInBytes = 256 * d_lut.channels() * sizeof(uchar);

            cudaTextureDesc texDescr;
            std::memset(&texDescr, 0, sizeof(texDescr));

            CV_CUDEV_SAFE_CALL( cudaCreateTextureObject(&texLutTableObj, &texRes, &texDescr, 0) );
        }
        else
        {
            // Use the texture reference
            cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar>();
            CV_CUDEV_SAFE_CALL( cudaBindTexture(0, &texLutTable, d_lut.data, &desc) );
        }
    }

    LookUpTableImpl::~LookUpTableImpl()
    {
        if (cc30)
        {
            // Use the texture object
            cudaDestroyTextureObject(texLutTableObj);
        }
        else
        {
            // Use the texture reference
            cudaUnbindTexture(texLutTable);
        }
    }

    struct LutTablePtrC1
    {
        typedef uchar value_type;
        typedef uchar index_type;

        cudaTextureObject_t texLutTableObj;

        __device__ __forceinline__ uchar operator ()(uchar, uchar x) const
        {
        #if CV_CUDEV_ARCH < 300
            // Use the texture reference
            return tex1Dfetch(texLutTable, x);
        #else
            // Use the texture object
            return tex1Dfetch<uchar>(texLutTableObj, x);
        #endif
        }
    };
    struct LutTablePtrC3
    {
        typedef uchar3 value_type;
        typedef uchar3 index_type;

        cudaTextureObject_t texLutTableObj;

        __device__ __forceinline__ uchar3 operator ()(const uchar3&, const uchar3& x) const
        {
        #if CV_CUDEV_ARCH < 300
            // Use the texture reference
            return make_uchar3(tex1Dfetch(texLutTable, x.x * 3), tex1Dfetch(texLutTable, x.y * 3 + 1), tex1Dfetch(texLutTable, x.z * 3 + 2));
        #else
            // Use the texture object
            return make_uchar3(tex1Dfetch<uchar>(texLutTableObj, x.x * 3), tex1Dfetch<uchar>(texLutTableObj, x.y * 3 + 1), tex1Dfetch<uchar>(texLutTableObj, x.z * 3 + 2));
        #endif
        }
    };

    void LookUpTableImpl::transform(InputArray _src, OutputArray _dst, Stream& stream)
    {
        GpuMat src = getInputMat(_src, stream);

        const int cn = src.channels();
        const int lut_cn = d_lut.channels();

        CV_Assert( src.type() == CV_8UC1 || src.type() == CV_8UC3 );
        CV_Assert( lut_cn == 1 || lut_cn == cn );

        GpuMat dst = getOutputMat(_dst, src.size(), src.type(), stream);

        if (lut_cn == 1)
        {
            GpuMat_<uchar> src1(src.reshape(1));
            GpuMat_<uchar> dst1(dst.reshape(1));

            LutTablePtrC1 tbl;
            tbl.texLutTableObj = texLutTableObj;

            dst1.assign(lut_(src1, tbl), stream);
        }
        else if (lut_cn == 3)
        {
            GpuMat_<uchar3>& src3 = (GpuMat_<uchar3>&) src;
            GpuMat_<uchar3>& dst3 = (GpuMat_<uchar3>&) dst;

            LutTablePtrC3 tbl;
            tbl.texLutTableObj = texLutTableObj;

            dst3.assign(lut_(src3, tbl), stream);
        }

        syncOutput(dst, _dst, stream);
    }

} }

#endif
