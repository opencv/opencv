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

////////////////////////////////////////////////////////////////////////
//////////////////////////////// GpuMat ////////////////////////////////
////////////////////////////////////////////////////////////////////////


#if !defined (HAVE_CUDA)

namespace cv
{
    namespace gpu
    {
        void GpuMat::upload(const Mat& /*m*/) { throw_nogpu(); }
        void GpuMat::download(cv::Mat& /*m*/) const { throw_nogpu(); }
        void GpuMat::copyTo( GpuMat& /*m*/ ) const { throw_nogpu(); }
        void GpuMat::copyTo( GpuMat& /*m*/, const GpuMat&/* mask */) const { throw_nogpu(); }
        void GpuMat::convertTo( GpuMat& /*m*/, int /*rtype*/, double /*alpha*/, double /*beta*/ ) const { throw_nogpu(); }
        GpuMat& GpuMat::operator = (const Scalar& /*s*/) { throw_nogpu(); return *this; }
        GpuMat& GpuMat::setTo(const Scalar& /*s*/, const GpuMat& /*mask*/) { throw_nogpu(); return *this; }
        GpuMat GpuMat::reshape(int /*new_cn*/, int /*new_rows*/) const { throw_nogpu(); return GpuMat(); }
        void GpuMat::create(int /*_rows*/, int /*_cols*/, int /*_type*/) { throw_nogpu(); }
        void GpuMat::release() { throw_nogpu(); }

        void MatPL::create(int /*_rows*/, int /*_cols*/, int /*_type*/) { throw_nogpu(); }
        void MatPL::release() { throw_nogpu(); }
    }

}


#else /* !defined (HAVE_CUDA) */


void cv::gpu::GpuMat::upload(const Mat& m)
{
    CV_DbgAssert(!m.empty());
    create(m.size(), m.type());
    cudaSafeCall( cudaMemcpy2D(data, step, m.data, m.step, cols * elemSize(), rows, cudaMemcpyHostToDevice) );
}

void cv::gpu::GpuMat::download(cv::Mat& m) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    cudaSafeCall( cudaMemcpy2D(m.data, m.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToHost) );
}

void cv::gpu::GpuMat::copyTo( GpuMat& m ) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    cudaSafeCall( cudaMemcpy2D(m.data, m.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice) );
    cudaSafeCall( cudaThreadSynchronize() );
}

void cv::gpu::GpuMat::copyTo( GpuMat& /*m*/, const GpuMat&/* mask */) const
{    
    CV_Assert(!"Not implemented");
}

void cv::gpu::GpuMat::convertTo( GpuMat& /*m*/, int /*rtype*/, double /*alpha*/, double /*beta*/ ) const
{
    CV_Assert(!"Not implemented");
}

GpuMat& cv::gpu::GpuMat::operator = (const Scalar& /*s*/)
{
    CV_Assert(!"Not implemented"); 
    return *this;
}

GpuMat& cv::gpu::GpuMat::setTo(const Scalar& /*s*/, const GpuMat& /*mask*/)
{
    CV_Assert(!"Not implemented");    
    return *this;
}


GpuMat cv::gpu::GpuMat::reshape(int new_cn, int new_rows) const
{
    GpuMat hdr = *this;

    int cn = channels();
    if( new_cn == 0 )
        new_cn = cn;

    int total_width = cols * cn;

    if( (new_cn > total_width || total_width % new_cn != 0) && new_rows == 0 )
        new_rows = rows * total_width / new_cn;

    if( new_rows != 0 && new_rows != rows )
    {
        int total_size = total_width * rows;
        if( !isContinuous() )
            CV_Error( CV_BadStep, "The matrix is not continuous, thus its number of rows can not be changed" );

        if( (unsigned)new_rows > (unsigned)total_size )
            CV_Error( CV_StsOutOfRange, "Bad new number of rows" );

        total_width = total_size / new_rows;

        if( total_width * new_rows != total_size )
            CV_Error( CV_StsBadArg, "The total number of matrix elements is not divisible by the new number of rows" );

        hdr.rows = new_rows;
        hdr.step = total_width * elemSize1();
    }

    int new_width = total_width / new_cn;

    if( new_width * new_cn != total_width )
        CV_Error( CV_BadNumChannels, "The total width is not divisible by the new number of channels" );

    hdr.cols = new_width;
    hdr.flags = (hdr.flags & ~CV_MAT_CN_MASK) | ((new_cn-1) << CV_CN_SHIFT);
    return hdr;
}

void cv::gpu::GpuMat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    if( rows == _rows && cols == _cols && type() == _type && data )
        return;
    if( data )
        release();
    CV_DbgAssert( _rows >= 0 && _cols >= 0 );
    if( _rows > 0 && _cols > 0 )
    {
        flags = Mat::MAGIC_VAL + _type;
        rows = _rows;
        cols = _cols;

        size_t esz = elemSize();                

        void *dev_ptr;
        cudaSafeCall( cudaMallocPitch(&dev_ptr, &step, esz * cols, rows) );

        if (esz * cols == step)
            flags |= Mat::CONTINUOUS_FLAG;

        int64 _nettosize = (int64)step*rows;
        size_t nettosize = (size_t)_nettosize;

        datastart = data = (uchar*)dev_ptr;
        dataend = data + nettosize;            

        refcount = (int*)fastMalloc(sizeof(*refcount));
        *refcount = 1;
    }
}

void cv::gpu::GpuMat::release()
{
    if( refcount && CV_XADD(refcount, -1) == 1 )
    {
        fastFree(refcount);
        cudaSafeCall( cudaFree(datastart) );        
    }
    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}


///////////////////////////////////////////////////////////////////////
//////////////////////////////// MatPL ////////////////////////////////
///////////////////////////////////////////////////////////////////////

void cv::gpu::MatPL::create(int _rows, int _cols, int _type)
{
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
        int64 _nettosize = (int64)step*rows;
        size_t nettosize = (size_t)_nettosize;
        if( _nettosize != (int64)nettosize )
            CV_Error(CV_StsNoMem, "Too big buffer is allocated");
        size_t datasize = alignSize(nettosize, (int)sizeof(*refcount));

        //datastart = data = (uchar*)fastMalloc(datasize + sizeof(*refcount));        
        void *ptr;
        cudaSafeCall( cudaHostAlloc( &ptr, datasize, cudaHostAllocDefault) );

        datastart = data =  (uchar*)ptr;        
        dataend = data + nettosize;       

        refcount = (int*)cv::fastMalloc(sizeof(*refcount));
        *refcount = 1;
    }
}

void cv::gpu::MatPL::release()
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