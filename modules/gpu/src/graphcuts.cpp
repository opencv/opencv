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

#include "precomp.hpp"

#if !defined (HAVE_CUDA)

void cv::gpu::graphcut(GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }
void cv::gpu::graphcut(GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, GpuMat&, Stream&) { throw_nogpu(); }

void cv::gpu::labelComponents(const GpuMat&, GpuMat&, GpuMat&, const cv::Scalar&, const cv::Scalar&, Stream&) { throw_nogpu(); }

#else /* !defined (HAVE_CUDA) */

namespace cv { namespace gpu { namespace device
{
    namespace ccl
    {
        void labelComponents(const DevMem2D& edges, DevMem2Di comps, cudaStream_t stream);
        void computeEdges(const DevMem2D& image, DevMem2D edges, const int lo, const int hi, cudaStream_t stream);
    }
}}}

void cv::gpu::labelComponents(const GpuMat& image, GpuMat& mask, GpuMat& components, const cv::Scalar& lo, const cv::Scalar& hi, Stream& s)
{
    CV_Assert(!image.empty());

    int type = image.type();
    CV_Assert(type == CV_8UC1);

    if (image.size() != mask.size() || mask.type() != CV_8UC1)
        mask.create(image.size(), CV_8UC1);

    if (image.size() != components.size() || components.type() != CV_32SC1)
        components.create(image.size(), CV_32SC1);

    cudaStream_t stream = StreamAccessor::getStream(s);

    device::ccl::computeEdges(image, mask, lo[0], hi[0], stream);
    device::ccl::labelComponents(mask, components, stream);
}


namespace
{
    typedef NppStatus (*init_func_t)(NppiSize oSize, NppiGraphcutState** ppState, Npp8u* pDeviceMem);

    class NppiGraphcutStateHandler
    {
    public:
        NppiGraphcutStateHandler(NppiSize sznpp, Npp8u* pDeviceMem, const init_func_t func)
        {
            nppSafeCall( func(sznpp, &pState, pDeviceMem) );
        }

        ~NppiGraphcutStateHandler()
        {
            nppSafeCall( nppiGraphcutFree(pState) );
        }

        operator NppiGraphcutState*()
        {
            return pState;
        }

    private:
        NppiGraphcutState* pState;
    };
}

void cv::gpu::graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& bottom, GpuMat& labels, GpuMat& buf, Stream& s)
{
#if (CUDA_VERSION < 5000)
    CV_Assert(terminals.type() == CV_32S);
#else
    CV_Assert(terminals.type() == CV_32S || terminals.type() == CV_32F);
#endif

    Size src_size = terminals.size();

    CV_Assert(leftTransp.size() == Size(src_size.height, src_size.width));
    CV_Assert(leftTransp.type() == terminals.type());

    CV_Assert(rightTransp.size() == Size(src_size.height, src_size.width));
    CV_Assert(rightTransp.type() == terminals.type());

    CV_Assert(top.size() == src_size);
    CV_Assert(top.type() == terminals.type());

    CV_Assert(bottom.size() == src_size);
    CV_Assert(bottom.type() == terminals.type());

    labels.create(src_size, CV_8U);

    NppiSize sznpp;
    sznpp.width = src_size.width;
    sznpp.height = src_size.height;

    int bufsz;
    nppSafeCall( nppiGraphcutGetSize(sznpp, &bufsz) );

    ensureSizeIsEnough(1, bufsz, CV_8U, buf);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    NppiGraphcutStateHandler state(sznpp, buf.ptr<Npp8u>(), nppiGraphcutInitAlloc);

#if (CUDA_VERSION < 5000)
    nppSafeCall( nppiGraphcut_32s8u(terminals.ptr<Npp32s>(), leftTransp.ptr<Npp32s>(), rightTransp.ptr<Npp32s>(), top.ptr<Npp32s>(), bottom.ptr<Npp32s>(),
        static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), state) );
#else
    if (terminals.type() == CV_32S)
    {
        nppSafeCall( nppiGraphcut_32s8u(terminals.ptr<Npp32s>(), leftTransp.ptr<Npp32s>(), rightTransp.ptr<Npp32s>(), top.ptr<Npp32s>(), bottom.ptr<Npp32s>(),
            static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), state) );
    }
    else
    {
        nppSafeCall( nppiGraphcut_32f8u(terminals.ptr<Npp32f>(), leftTransp.ptr<Npp32f>(), rightTransp.ptr<Npp32f>(), top.ptr<Npp32f>(), bottom.ptr<Npp32f>(),
            static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), state) );
    }
#endif

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

void cv::gpu::graphcut(GpuMat& terminals, GpuMat& leftTransp, GpuMat& rightTransp, GpuMat& top, GpuMat& topLeft, GpuMat& topRight,
              GpuMat& bottom, GpuMat& bottomLeft, GpuMat& bottomRight, GpuMat& labels, GpuMat& buf, Stream& s)
{
#if (CUDA_VERSION < 5000)
    CV_Assert(terminals.type() == CV_32S);
#else
    CV_Assert(terminals.type() == CV_32S || terminals.type() == CV_32F);
#endif

    Size src_size = terminals.size();

    CV_Assert(leftTransp.size() == Size(src_size.height, src_size.width));
    CV_Assert(leftTransp.type() == terminals.type());

    CV_Assert(rightTransp.size() == Size(src_size.height, src_size.width));
    CV_Assert(rightTransp.type() == terminals.type());

    CV_Assert(top.size() == src_size);
    CV_Assert(top.type() == terminals.type());

    CV_Assert(topLeft.size() == src_size);
    CV_Assert(topLeft.type() == terminals.type());

    CV_Assert(topRight.size() == src_size);
    CV_Assert(topRight.type() == terminals.type());

    CV_Assert(bottom.size() == src_size);
    CV_Assert(bottom.type() == terminals.type());

    CV_Assert(bottomLeft.size() == src_size);
    CV_Assert(bottomLeft.type() == terminals.type());

    CV_Assert(bottomRight.size() == src_size);
    CV_Assert(bottomRight.type() == terminals.type());

    labels.create(src_size, CV_8U);

    NppiSize sznpp;
    sznpp.width = src_size.width;
    sznpp.height = src_size.height;

    int bufsz;
    nppSafeCall( nppiGraphcut8GetSize(sznpp, &bufsz) );

    ensureSizeIsEnough(1, bufsz, CV_8U, buf);

    cudaStream_t stream = StreamAccessor::getStream(s);

    NppStreamHandler h(stream);

    NppiGraphcutStateHandler state(sznpp, buf.ptr<Npp8u>(), nppiGraphcut8InitAlloc);

#if (CUDA_VERSION < 5000)
    nppSafeCall( nppiGraphcut8_32s8u(terminals.ptr<Npp32s>(), leftTransp.ptr<Npp32s>(), rightTransp.ptr<Npp32s>(),
        top.ptr<Npp32s>(), topLeft.ptr<Npp32s>(), topRight.ptr<Npp32s>(),
        bottom.ptr<Npp32s>(), bottomLeft.ptr<Npp32s>(), bottomRight.ptr<Npp32s>(),
        static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), state) );
#else
    if (terminals.type() == CV_32S)
    {
        nppSafeCall( nppiGraphcut8_32s8u(terminals.ptr<Npp32s>(), leftTransp.ptr<Npp32s>(), rightTransp.ptr<Npp32s>(),
            top.ptr<Npp32s>(), topLeft.ptr<Npp32s>(), topRight.ptr<Npp32s>(),
            bottom.ptr<Npp32s>(), bottomLeft.ptr<Npp32s>(), bottomRight.ptr<Npp32s>(),
            static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), state) );
    }
    else
    {
        nppSafeCall( nppiGraphcut8_32f8u(terminals.ptr<Npp32f>(), leftTransp.ptr<Npp32f>(), rightTransp.ptr<Npp32f>(),
            top.ptr<Npp32f>(), topLeft.ptr<Npp32f>(), topRight.ptr<Npp32f>(),
            bottom.ptr<Npp32f>(), bottomLeft.ptr<Npp32f>(), bottomRight.ptr<Npp32f>(),
            static_cast<int>(terminals.step), static_cast<int>(leftTransp.step), sznpp, labels.ptr<Npp8u>(), static_cast<int>(labels.step), state) );
    }
#endif

    if (stream == 0)
        cudaSafeCall( cudaDeviceSynchronize() );
}

#endif /* !defined (HAVE_CUDA) */
