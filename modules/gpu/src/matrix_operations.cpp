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
using namespace cv::gpu;

#if !defined (HAVE_CUDA)

void cv::gpu::registerPageLocked(Mat&) { throw_nogpu(); }
void cv::gpu::unregisterPageLocked(Mat&) { throw_nogpu(); }
void cv::gpu::CudaMem::create(int /*_rows*/, int /*_cols*/, int /*_type*/, int /*type_alloc*/) { throw_nogpu(); }
bool cv::gpu::CudaMem::canMapHostMemory() { throw_nogpu(); return false; }
void cv::gpu::CudaMem::release() { throw_nogpu(); }
GpuMat cv::gpu::CudaMem::createGpuMatHeader () const { throw_nogpu(); return GpuMat(); }

#else /* !defined (HAVE_CUDA) */

void registerPageLocked(Mat& m)
{
    cudaSafeCall( cudaHostRegister(m.ptr(), m.step * m.rows, cudaHostRegisterPortable) );
}

void unregisterPageLocked(Mat& m)
{
    cudaSafeCall( cudaHostUnregister(m.ptr()) );
}

bool cv::gpu::CudaMem::canMapHostMemory()
{
    cudaDeviceProp prop;
    cudaSafeCall( cudaGetDeviceProperties(&prop, getDevice()) );
    return (prop.canMapHostMemory != 0) ? true : false;
}

namespace
{
    size_t alignUpStep(size_t what, size_t alignment)
    {
        size_t alignMask = alignment-1;
        size_t inverseAlignMask = ~alignMask;
        size_t res = (what + alignMask) & inverseAlignMask;
        return res;
    }
}

void cv::gpu::CudaMem::create(int _rows, int _cols, int _type, int _alloc_type)
{
    if (_alloc_type == ALLOC_ZEROCOPY && !canMapHostMemory())
            cv::gpu::error("ZeroCopy is not supported by current device", __FILE__, __LINE__);

    _type &= TYPE_MASK;
    if( rows == _rows && cols == _cols && type() == _type && data )
        return;
    if( data )
        release();
    CV_DbgAssert( _rows >= 0 && _cols >= 0 );
    if( _rows > 0 && _cols > 0 )
    {
        flags = Mat::MAGIC_VAL + Mat::CONTINUOUS_FLAG + _type;
        rows = _rows;
        cols = _cols;
        step = elemSize()*cols;
        if (_alloc_type == ALLOC_ZEROCOPY)
        {
            cudaDeviceProp prop;
            cudaSafeCall( cudaGetDeviceProperties(&prop, getDevice()) );
            step = alignUpStep(step, prop.textureAlignment);
        }
        int64 _nettosize = (int64)step*rows;
        size_t nettosize = (size_t)_nettosize;
        if( _nettosize != (int64)nettosize )
            CV_Error(CV_StsNoMem, "Too big buffer is allocated");
        size_t datasize = alignSize(nettosize, (int)sizeof(*refcount));

        //datastart = data = (uchar*)fastMalloc(datasize + sizeof(*refcount));
        alloc_type = _alloc_type;
        void *ptr;

        switch (alloc_type)
        {
            case ALLOC_PAGE_LOCKED:    cudaSafeCall( cudaHostAlloc( &ptr, datasize, cudaHostAllocDefault) ); break;
            case ALLOC_ZEROCOPY:       cudaSafeCall( cudaHostAlloc( &ptr, datasize, cudaHostAllocMapped) );  break;
            case ALLOC_WRITE_COMBINED: cudaSafeCall( cudaHostAlloc( &ptr, datasize, cudaHostAllocWriteCombined) ); break;
            default: cv::gpu::error("Invalid alloc type", __FILE__, __LINE__);
        }

        datastart = data =  (uchar*)ptr;
        dataend = data + nettosize;

        refcount = (int*)cv::fastMalloc(sizeof(*refcount));
        *refcount = 1;
    }
}

GpuMat cv::gpu::CudaMem::createGpuMatHeader () const
{
    GpuMat res;
    if (alloc_type == ALLOC_ZEROCOPY)
    {
        void *pdev;
        cudaSafeCall( cudaHostGetDevicePointer( &pdev, data, 0 ) );
        res = GpuMat(rows, cols, type(), pdev, step);
    }
    else
        cv::gpu::error("Zero-copy is not supported or memory was allocated without zero-copy flag", __FILE__, __LINE__);

    return res;
}

void cv::gpu::CudaMem::release()
{
    if( refcount && CV_XADD(refcount, -1) == 1 )
    {
        cudaSafeCall( cudaFreeHost(datastart ) );
        fastFree(refcount);
    }
    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}

#endif /* !defined (HAVE_CUDA) */
