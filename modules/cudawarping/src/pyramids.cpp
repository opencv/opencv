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

using namespace cv;
using namespace cv::cuda;

#if !defined HAVE_CUDA || defined(CUDA_DISABLER)

void cv::cuda::pyrDown(InputArray, OutputArray, Stream&) { throw_no_cuda(); }
void cv::cuda::pyrUp(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

Ptr<ImagePyramid> cv::cuda::createImagePyramid(InputArray, int, Stream&) { throw_no_cuda(); return Ptr<ImagePyramid>(); }

#else // HAVE_CUDA

//////////////////////////////////////////////////////////////////////////////
// pyrDown

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template <typename T> void pyrDown_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

void cv::cuda::pyrDown(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace cv::cuda::device::imgproc;

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

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.depth() <= CV_32F && src.channels() <= 4 );

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert( func != 0 );

    _dst.create((src.rows + 1) / 2, (src.cols + 1) / 2, src.type());
    GpuMat dst = _dst.getGpuMat();

    func(src, dst, StreamAccessor::getStream(stream));
}


//////////////////////////////////////////////////////////////////////////////
// pyrUp

namespace cv { namespace cuda { namespace device
{
    namespace imgproc
    {
        template <typename T> void pyrUp_gpu(PtrStepSzb src, PtrStepSzb dst, cudaStream_t stream);
    }
}}}

void cv::cuda::pyrUp(InputArray _src, OutputArray _dst, Stream& stream)
{
    using namespace cv::cuda::device::imgproc;

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

    GpuMat src = _src.getGpuMat();

    CV_Assert( src.depth() <= CV_32F && src.channels() <= 4 );

    const func_t func = funcs[src.depth()][src.channels() - 1];
    CV_Assert( func != 0 );

    _dst.create(src.rows * 2, src.cols * 2, src.type());
    GpuMat dst = _dst.getGpuMat();

    func(src, dst, StreamAccessor::getStream(stream));
}

//////////////////////////////////////////////////////////////////////////////
// ImagePyramid

#ifdef HAVE_OPENCV_CUDALEGACY

namespace
{
    class ImagePyramidImpl : public ImagePyramid
    {
    public:
        ImagePyramidImpl(InputArray img, int nLayers, Stream& stream);

        void getLayer(OutputArray outImg, Size outRoi, Stream& stream = Stream::Null()) const;

    private:
        GpuMat layer0_;
        std::vector<GpuMat> pyramid_;
        int nLayers_;
    };

    ImagePyramidImpl::ImagePyramidImpl(InputArray _img, int numLayers, Stream& stream)
    {
        GpuMat img = _img.getGpuMat();

        CV_Assert( img.depth() <= CV_32F && img.channels() <= 4 );

        img.copyTo(layer0_, stream);

        Size szLastLayer = img.size();
        nLayers_ = 1;

        if (numLayers <= 0)
            numLayers = 255; // it will cut-off when any of the dimensions goes 1

        pyramid_.resize(numLayers);

        for (int i = 0; i < numLayers - 1; ++i)
        {
            Size szCurLayer(szLastLayer.width / 2, szLastLayer.height / 2);

            if (szCurLayer.width == 0 || szCurLayer.height == 0)
                break;

            ensureSizeIsEnough(szCurLayer, img.type(), pyramid_[i]);
            nLayers_++;

            const GpuMat& prevLayer = i == 0 ? layer0_ : pyramid_[i - 1];

            cv::cuda::device::pyramid::downsampleX2(prevLayer, pyramid_[i], img.depth(), img.channels(), StreamAccessor::getStream(stream));

            szLastLayer = szCurLayer;
        }
    }

    void ImagePyramidImpl::getLayer(OutputArray _outImg, Size outRoi, Stream& stream) const
    {
        CV_Assert( outRoi.width <= layer0_.cols && outRoi.height <= layer0_.rows && outRoi.width > 0 && outRoi.height > 0 );

        ensureSizeIsEnough(outRoi, layer0_.type(), _outImg);
        GpuMat outImg = _outImg.getGpuMat();

        if (outRoi.width == layer0_.cols && outRoi.height == layer0_.rows)
        {
            layer0_.copyTo(outImg, stream);
            return;
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
                curLayer.copyTo(outImg, stream);
            }

            if (outRoi.width >= curLayer.cols && outRoi.height >= curLayer.rows)
                break;

            lastScale = curScale;
            lastLayer = curLayer;
        }

        cv::cuda::device::pyramid::interpolateFrom1(lastLayer, outImg, outImg.depth(), outImg.channels(), StreamAccessor::getStream(stream));
    }
}

#endif

Ptr<ImagePyramid> cv::cuda::createImagePyramid(InputArray img, int nLayers, Stream& stream)
{
#ifndef HAVE_OPENCV_CUDALEGACY
    (void) img;
    (void) nLayers;
    (void) stream;
    throw_no_cuda();
    return Ptr<ImagePyramid>();
#else
    return Ptr<ImagePyramid>(new ImagePyramidImpl(img, nLayers, stream));
#endif
}

#endif // HAVE_CUDA
