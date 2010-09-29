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

        void CudaMem::create(int /*_rows*/, int /*_cols*/, int /*_type*/, int /*type_alloc*/) { throw_nogpu(); }
        bool CudaMem::can_device_map_to_host() { throw_nogpu(); return false; }
        void CudaMem::release() { throw_nogpu(); }
        GpuMat CudaMem::createGpuMatHeader () const { throw_nogpu(); return GpuMat(); }
    }

}

#else /* !defined (HAVE_CUDA) */

void cv::gpu::GpuMat::upload(const Mat& m)
{
    CV_DbgAssert(!m.empty());
    create(m.size(), m.type());
    cudaSafeCall( cudaMemcpy2D(data, step, m.data, m.step, cols * elemSize(), rows, cudaMemcpyHostToDevice) );
}

void cv::gpu::GpuMat::upload(const CudaMem& m, Stream& stream)
{
    CV_DbgAssert(!m.empty());
    stream.enqueueUpload(m, *this);
}

void cv::gpu::GpuMat::download(cv::Mat& m) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    cudaSafeCall( cudaMemcpy2D(m.data, m.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToHost) );
}

void cv::gpu::GpuMat::download(CudaMem& m, Stream& stream) const
{
    CV_DbgAssert(!m.empty());
    stream.enqueueDownload(*this, m);
}

void cv::gpu::GpuMat::copyTo( GpuMat& m ) const
{
    CV_DbgAssert(!this->empty());
    m.create(size(), type());
    cudaSafeCall( cudaMemcpy2D(m.data, m.step, data, step, cols * elemSize(), rows, cudaMemcpyDeviceToDevice) );
    cudaSafeCall( cudaThreadSynchronize() );
}

void cv::gpu::GpuMat::copyTo( GpuMat& mat, const GpuMat& mask ) const
{
    if (mask.empty())
    {
        copyTo(mat);
    }
    else
    {
        mat.create(size(), type());
        cv::gpu::matrix_operations::copy_to_with_mask(*this, mat, depth(), mask, channels());
    }
}

void cv::gpu::GpuMat::convertTo( GpuMat& dst, int rtype, double alpha, double beta ) const
{
    bool noScale = fabs(alpha-1) < std::numeric_limits<double>::epsilon() && fabs(beta) < std::numeric_limits<double>::epsilon();

    if( rtype < 0 )
        rtype = type();
    else
        rtype = CV_MAKETYPE(CV_MAT_DEPTH(rtype), channels());
    
    int stype = type();
    int sdepth = depth(), ddepth = CV_MAT_DEPTH(rtype);
    if( sdepth == ddepth && noScale )
    {
        copyTo(dst);
        return;
    }

    GpuMat temp;
    const GpuMat* psrc = this;
    if( sdepth != ddepth && psrc == &dst )
        psrc = &(temp = *this);

    dst.create( size(), rtype );

    if (!noScale)
        matrix_operations::convert_to(*psrc, sdepth, dst, ddepth, psrc->channels(), alpha, beta);
    else
    {
        NppiSize sz;
        sz.width = cols;
        sz.height = rows;

        if (stype == CV_8UC1 && ddepth == CV_16U)
            nppSafeCall( nppiConvert_8u16u_C1R(psrc->ptr<Npp8u>(), psrc->step, dst.ptr<Npp16u>(), dst.step, sz) );
        else if (stype == CV_16UC1 && ddepth == CV_8U)
            nppSafeCall( nppiConvert_16u8u_C1R(psrc->ptr<Npp16u>(), psrc->step, dst.ptr<Npp8u>(), dst.step, sz) );
        else if (stype == CV_8UC4 && ddepth == CV_16U)
            nppSafeCall( nppiConvert_8u16u_C4R(psrc->ptr<Npp8u>(), psrc->step, dst.ptr<Npp16u>(), dst.step, sz) );
        else if (stype == CV_16UC4 && ddepth == CV_8U)
            nppSafeCall( nppiConvert_16u8u_C4R(psrc->ptr<Npp16u>(), psrc->step, dst.ptr<Npp8u>(), dst.step, sz) );
        else if (stype == CV_8UC1 && ddepth == CV_16S)
            nppSafeCall( nppiConvert_8u16s_C1R(psrc->ptr<Npp8u>(), psrc->step, dst.ptr<Npp16s>(), dst.step, sz) );
        else if (stype == CV_16SC1 && ddepth == CV_8U)
            nppSafeCall( nppiConvert_16s8u_C1R(psrc->ptr<Npp16s>(), psrc->step, dst.ptr<Npp8u>(), dst.step, sz) );
        else if (stype == CV_8UC4 && ddepth == CV_16S)
            nppSafeCall( nppiConvert_8u16s_C4R(psrc->ptr<Npp8u>(), psrc->step, dst.ptr<Npp16s>(), dst.step, sz) );
        else if (stype == CV_16SC4 && ddepth == CV_8U)
            nppSafeCall( nppiConvert_16s8u_C4R(psrc->ptr<Npp16s>(), psrc->step, dst.ptr<Npp8u>(), dst.step, sz) );
        else if (stype == CV_16SC1 && ddepth == CV_32F)
            nppSafeCall( nppiConvert_16s32f_C1R(psrc->ptr<Npp16s>(), psrc->step, dst.ptr<Npp32f>(), dst.step, sz) );
        else if (stype == CV_32FC1 && ddepth == CV_16S)
            nppSafeCall( nppiConvert_32f16s_C1R(psrc->ptr<Npp32f>(), psrc->step, dst.ptr<Npp16s>(), dst.step, sz, NPP_RND_NEAR) );
        else if (stype == CV_8UC1 && ddepth == CV_32F)
            nppSafeCall( nppiConvert_8u32f_C1R(psrc->ptr<Npp8u>(), psrc->step, dst.ptr<Npp32f>(), dst.step, sz) );
        else if (stype == CV_32FC1 && ddepth == CV_8U)
            nppSafeCall( nppiConvert_32f8u_C1R(psrc->ptr<Npp32f>(), psrc->step, dst.ptr<Npp8u>(), dst.step, sz, NPP_RND_NEAR) );
        else if (stype == CV_16UC1 && ddepth == CV_32F)
            nppSafeCall( nppiConvert_16u32f_C1R(psrc->ptr<Npp16u>(), psrc->step, dst.ptr<Npp32f>(), dst.step, sz) );
        else if (stype == CV_32FC1 && ddepth == CV_16U)
            nppSafeCall( nppiConvert_32f16u_C1R(psrc->ptr<Npp32f>(), psrc->step, dst.ptr<Npp16u>(), dst.step, sz, NPP_RND_NEAR) );
        else if (stype == CV_16UC1 && ddepth == CV_32S)
            nppSafeCall( nppiConvert_16u32s_C1R(psrc->ptr<Npp16u>(), psrc->step, dst.ptr<Npp32s>(), dst.step, sz) );
        else if (stype == CV_16SC1 && ddepth == CV_32S)
            nppSafeCall( nppiConvert_16s32s_C1R(psrc->ptr<Npp16s>(), psrc->step, dst.ptr<Npp32s>(), dst.step, sz) );
        else
            matrix_operations::convert_to(*psrc, sdepth, dst, ddepth, psrc->channels(), 1.0, 0.0);
    }
}

GpuMat& GpuMat::operator = (const Scalar& s)
{
    setTo(s);
    return *this;
}

GpuMat& GpuMat::setTo(const Scalar& s, const GpuMat& mask)
{
    CV_Assert(mask.type() == CV_8UC1);

    CV_DbgAssert(!this->empty());
    
    NppiSize sz;
    sz.width  = cols;
    sz.height = rows;

    if (mask.empty())
    {
        switch (type())
        {
        case CV_8UC1:
            {
                Npp8u nVal = (Npp8u)s[0];
                nppSafeCall( nppiSet_8u_C1R(nVal, ptr<Npp8u>(), step, sz) );
                break;
            }
        case CV_8UC4:
            {
                Scalar_<Npp8u> nVal = s;
                nppSafeCall( nppiSet_8u_C4R(nVal.val, ptr<Npp8u>(), step, sz) );
                break;
            }
        case CV_16UC1:
            {
                Npp16u nVal = (Npp16u)s[0];
                nppSafeCall( nppiSet_16u_C1R(nVal, ptr<Npp16u>(), step, sz) );
                break;
            }
        /*case CV_16UC2:
            {
                Scalar_<Npp16u> nVal = s;
                nppSafeCall( nppiSet_16u_C2R(nVal.val, ptr<Npp16u>(), step, sz) );
                break;
            }*/
        case CV_16UC4:
            {
                Scalar_<Npp16u> nVal = s;
                nppSafeCall( nppiSet_16u_C4R(nVal.val, ptr<Npp16u>(), step, sz) );
                break;
            }
        case CV_16SC1:
            {
                Npp16s nVal = (Npp16s)s[0];
                nppSafeCall( nppiSet_16s_C1R(nVal, ptr<Npp16s>(), step, sz) );
                break;
            }
        /*case CV_16SC2:
            {
                Scalar_<Npp16s> nVal = s;
                nppSafeCall( nppiSet_16s_C2R(nVal.val, ptr<Npp16s>(), step, sz) );
                break;
            }*/
        case CV_16SC4:
            {
                Scalar_<Npp16s> nVal = s;
                nppSafeCall( nppiSet_16s_C4R(nVal.val, ptr<Npp16s>(), step, sz) );
                break;
            }
        case CV_32SC1:
            {
                Npp32s nVal = (Npp32s)s[0];
                nppSafeCall( nppiSet_32s_C1R(nVal, ptr<Npp32s>(), step, sz) );
                break;
            }
        case CV_32SC4:
            {
                Scalar_<Npp32s> nVal = s;
                nppSafeCall( nppiSet_32s_C4R(nVal.val, ptr<Npp32s>(), step, sz) );
                break;
            }
        case CV_32FC1:
            {
                Npp32f nVal = (Npp32f)s[0];
                nppSafeCall( nppiSet_32f_C1R(nVal, ptr<Npp32f>(), step, sz) );
                break;
            }
        case CV_32FC4:
            {
                Scalar_<Npp32f> nVal = s;
                nppSafeCall( nppiSet_32f_C4R(nVal.val, ptr<Npp32f>(), step, sz) );
                break;
            }
        default:
            matrix_operations::set_to_without_mask( *this, depth(), s.val, channels());
        }        
    }
    else
    {
        switch (type())
        {
        case CV_8UC1:
            {
                Npp8u nVal = (Npp8u)s[0];
                nppSafeCall( nppiSet_8u_C1MR(nVal, ptr<Npp8u>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_8UC4:
            {
                Scalar_<Npp8u> nVal = s;
                nppSafeCall( nppiSet_8u_C4MR(nVal.val, ptr<Npp8u>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_16UC1:
            {
                Npp16u nVal = (Npp16u)s[0];
                nppSafeCall( nppiSet_16u_C1MR(nVal, ptr<Npp16u>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_16UC4:
            {
                Scalar_<Npp16u> nVal = s;
                nppSafeCall( nppiSet_16u_C4MR(nVal.val, ptr<Npp16u>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_16SC1:
            {
                Npp16s nVal = (Npp16s)s[0];
                nppSafeCall( nppiSet_16s_C1MR(nVal, ptr<Npp16s>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_16SC4:
            {
                Scalar_<Npp16s> nVal = s;
                nppSafeCall( nppiSet_16s_C4MR(nVal.val, ptr<Npp16s>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_32SC1:
            {
                Npp32s nVal = (Npp32s)s[0];
                nppSafeCall( nppiSet_32s_C1MR(nVal, ptr<Npp32s>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_32SC4:
            {
                Scalar_<Npp32s> nVal = s;
                nppSafeCall( nppiSet_32s_C4MR(nVal.val, ptr<Npp32s>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_32FC1:
            {
                Npp32f nVal = (Npp32f)s[0];
                nppSafeCall( nppiSet_32f_C1MR(nVal, ptr<Npp32f>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        case CV_32FC4:
            {
                Scalar_<Npp32f> nVal = s;
                nppSafeCall( nppiSet_32f_C4MR(nVal.val, ptr<Npp32f>(), step, sz, mask.ptr<Npp8u>(), mask.step) );
                break;
            }
        default:
            matrix_operations::set_to_with_mask( *this, depth(), s.val, mask, channels());
        }
    }

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
//////////////////////////////// CudaMem //////////////////////////////
///////////////////////////////////////////////////////////////////////

bool cv::gpu::CudaMem::can_device_map_to_host()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return (prop.canMapHostMemory != 0) ? true : false;
}

void cv::gpu::CudaMem::create(int _rows, int _cols, int _type, int _alloc_type)
{   
    if (_alloc_type == ALLOC_ZEROCOPY && !can_device_map_to_host())
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
