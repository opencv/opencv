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

#include "precomp.hpp"

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::gpu::pyrDown(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::pyrUp(const GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::ImagePyramid::build(const GpuMat&, int, Stream&) { throw_nogpu(); }
void cv::gpu::ImagePyramid::getLayer(GpuMat&, Size, Stream&) const { throw_nogpu(); }

#else // HAVE_CUDA

//////////////////////////////////////////////////////////////////////////////
// pyrDown

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T> void pyrDown_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

void cv::gpu::pyrDown(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    using namespace cv::gpu::device::imgproc;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {pyrDown_gpu<uchar>      , 0 /*pyrDown_gpu<uchar2>*/ , pyrDown_gpu<uchar3>      , pyrDown_gpu<uchar4>      },
        {0 /*pyrDown_gpu<schar>*/, 0 /*pyrDown_gpu<schar2>*/ , 0 /*pyrDown_gpu<schar3>*/, 0 /*pyrDown_gpu<schar4>*/},
        {pyrDown_gpu<ushort>     , 0 /*pyrDown_gpu<ushort2>*/, pyrDown_gpu<ushort3>     , pyrDown_gpu<ushort4>     },
        {pyrDown_gpu<short>      , 0 /*pyrDown_gpu<short2>*/ , pyrDown_gpu<short3>      , pyrDown_gpu<short4>      },
        {0 /*pyrDown_gpu<int>*/  , 0 /*pyrDown_gpu<int2>*/   , 0 /*pyrDown_gpu<int3>*/  , 0 /*pyrDown_gpu<int4>*/  },
        {pyrDown_gpu<float>      , 0 /*pyrDown_gpu<float2>*/ , pyrDown_gpu<float3>      , pyrDown_gpu<float4>      }
    };

    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert(func != 0);

    dst.create((src.rows + 1) / 2, (src.cols + 1) / 2, src.type());

    func(src, dst, StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// pyrUp

namespace cv { namespace gpu { namespace device
{
    namespace imgproc
    {
        template <typename T> void pyrUp_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

void cv::gpu::pyrUp(const GpuMat& src, GpuMat& dst, Stream& stream)
{
    using namespace cv::gpu::device::imgproc;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {pyrUp_gpu<uchar>      , 0 /*pyrUp_gpu<uchar2>*/ , pyrUp_gpu<uchar3>      , pyrUp_gpu<uchar4>      },
        {0 /*pyrUp_gpu<schar>*/, 0 /*pyrUp_gpu<schar2>*/ , 0 /*pyrUp_gpu<schar3>*/, 0 /*pyrUp_gpu<schar4>*/},
        {pyrUp_gpu<ushort>     , 0 /*pyrUp_gpu<ushort2>*/, pyrUp_gpu<ushort3>     , pyrUp_gpu<ushort4>     },
        {pyrUp_gpu<short>      , 0 /*pyrUp_gpu<short2>*/ , pyrUp_gpu<short3>      , pyrUp_gpu<short4>      },
        {0 /*pyrUp_gpu<int>*/  , 0 /*pyrUp_gpu<int2>*/   , 0 /*pyrUp_gpu<int3>*/  , 0 /*pyrUp_gpu<int4>*/  },
        {pyrUp_gpu<float>      , 0 /*pyrUp_gpu<float2>*/ , pyrUp_gpu<float3>      , pyrUp_gpu<float4>      }
    };

    CV_Assert(src.depth() <= CV_32F && src.channels() <= 4);

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert(func != 0);

    dst.create(src.rows * 2, src.cols * 2, src.type());

    func(src, dst, StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// ImagePyramid

namespace cv { namespace gpu { namespace device
{
    namespace pyramid
    {
        template <typename T> void kernelDownsampleX2_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
        template <typename T> void kernelInterpolateFrom1_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

void cv::gpu::ImagePyramid::build(const GpuMat& img, int numLayers, Stream& stream)
{
    using namespace cv::gpu::device::pyramid;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {kernelDownsampleX2_gpu<uchar1>       , 0 /*kernelDownsampleX2_gpu<uchar2>*/ , kernelDownsampleX2_gpu<uchar3>      , kernelDownsampleX2_gpu<uchar4>      },
        {0 /*kernelDownsampleX2_gpu<char1>*/  , 0 /*kernelDownsampleX2_gpu<char2>*/  , 0 /*kernelDownsampleX2_gpu<char3>*/ , 0 /*kernelDownsampleX2_gpu<char4>*/ },
        {kernelDownsampleX2_gpu<ushort1>      , 0 /*kernelDownsampleX2_gpu<ushort2>*/, kernelDownsampleX2_gpu<ushort3>     , kernelDownsampleX2_gpu<ushort4>     },
        {0 /*kernelDownsampleX2_gpu<short1>*/ , 0 /*kernelDownsampleX2_gpu<short2>*/ , 0 /*kernelDownsampleX2_gpu<short3>*/, 0 /*kernelDownsampleX2_gpu<short4>*/},
        {0 /*kernelDownsampleX2_gpu<int1>*/   , 0 /*kernelDownsampleX2_gpu<int2>*/   , 0 /*kernelDownsampleX2_gpu<int3>*/  , 0 /*kernelDownsampleX2_gpu<int4>*/  },
        {kernelDownsampleX2_gpu<float1>       , 0 /*kernelDownsampleX2_gpu<float2>*/ , kernelDownsampleX2_gpu<float3>      , kernelDownsampleX2_gpu<float4>      }
    };

    CV_Assert(img.depth() <= CV_32F && img.channels() <= 4);

    const func_t func = funcs[img.depth()][img.channels() - 1];
    CV_Assert(func != 0);

    layer0_ = img;
    Size szLastLayer = img.size();
    nLayers_ = 1;

    if (numLayers <= 0)
        numLayers = 255; //it will cut-off when any of the dimensions goes 1

    pyramid_.resize(numLayers);

    for (int i = 0; i < numLayers - 1; ++i)
    {
        Size szCurLayer(szLastLayer.width / 2, szLastLayer.height / 2);

        if (szCurLayer.width == 0 || szCurLayer.height == 0)
            break;

        ensureSizeIsEnough(szCurLayer, img.type(), pyramid_[i]);
        nLayers_++;

        const GpuMat& prevLayer = i == 0 ? layer0_ : pyramid_[i - 1];

        func(prevLayer, pyramid_[i], StreamAccessor::getStream(stream));

        szLastLayer = szCurLayer;
    }
}

void cv::gpu::ImagePyramid::getLayer(GpuMat& outImg, Size outRoi, Stream& stream) const
{
    using namespace cv::gpu::device::pyramid;

    typedef void (*func_t)(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);

    static const func_t funcs[6][4] =
    {
        {kernelInterpolateFrom1_gpu<uchar1>      , 0 /*kernelInterpolateFrom1_gpu<uchar2>*/ , kernelInterpolateFrom1_gpu<uchar3>      , kernelInterpolateFrom1_gpu<uchar4>      },
        {0 /*kernelInterpolateFrom1_gpu<char1>*/ , 0 /*kernelInterpolateFrom1_gpu<char2>*/  , 0 /*kernelInterpolateFrom1_gpu<char3>*/ , 0 /*kernelInterpolateFrom1_gpu<char4>*/ },
        {kernelInterpolateFrom1_gpu<ushort1>     , 0 /*kernelInterpolateFrom1_gpu<ushort2>*/, kernelInterpolateFrom1_gpu<ushort3>     , kernelInterpolateFrom1_gpu<ushort4>     },
        {0 /*kernelInterpolateFrom1_gpu<short1>*/, 0 /*kernelInterpolateFrom1_gpu<short2>*/ , 0 /*kernelInterpolateFrom1_gpu<short3>*/, 0 /*kernelInterpolateFrom1_gpu<short4>*/},
        {0 /*kernelInterpolateFrom1_gpu<int1>*/  , 0 /*kernelInterpolateFrom1_gpu<int2>*/   , 0 /*kernelInterpolateFrom1_gpu<int3>*/  , 0 /*kernelInterpolateFrom1_gpu<int4>*/  },
        {kernelInterpolateFrom1_gpu<float1>      , 0 /*kernelInterpolateFrom1_gpu<float2>*/ , kernelInterpolateFrom1_gpu<float3>      , kernelInterpolateFrom1_gpu<float4>      }
    };

    CV_Assert(outRoi.width <= layer0_.cols && outRoi.height <= layer0_.rows && outRoi.width > 0 && outRoi.height > 0);

    ensureSizeIsEnough(outRoi, layer0_.type(), outImg);

    const func_t func = funcs[outImg.depth()][outImg.channels() - 1];
    CV_Assert(func != 0);

    if (outRoi.width == layer0_.cols && outRoi.height == layer0_.rows)
    {
        if (stream)
            stream.enqueueCopy(layer0_, outImg);
        else
            layer0_.copyTo(outImg);
    }

    float lastScale = 1.0f;
    float curScale;
    GpuMat lastLayer = layer0_;
    GpuMat curLayer;

    for (int i = 0; i < nLayers_ - 1; ++i)
    {
        curScale = lastScale * 0.5f;
        curLayer = pyramid_[i];

        if (outRoi.width == curLayer.cols && outRoi.height == curLayer.rows)
        {
            if (stream)
                stream.enqueueCopy(curLayer, outImg);
            else
                curLayer.copyTo(outImg);
        }

        if (outRoi.width >= curLayer.cols && outRoi.height >= curLayer.rows)
            break;

        lastScale = curScale;
        lastLayer = curLayer;
    }

    func(lastLayer, outImg, StreamAccessor::getStream(stream));
}

#endif // HAVE_CUDA
