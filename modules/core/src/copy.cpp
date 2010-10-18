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

/* ////////////////////////////////////////////////////////////////////
//
//  Mat basic operations: Copy, Set
//
// */

#include "precomp.hpp"

namespace cv
{

template<typename T> static void
copyMask_(const Mat& srcmat, Mat& dstmat, const Mat& maskmat)
{
    const uchar* mask = maskmat.data;
    size_t sstep = srcmat.step;
    size_t dstep = dstmat.step;
    size_t mstep = maskmat.step;
    Size size = getContinuousSize(srcmat, dstmat, maskmat);

    for( int y = 0; y < size.height; y++, mask += mstep )
    {
        const T* src = (const T*)(srcmat.data + sstep*y);
        T* dst = (T*)(dstmat.data + dstep*y);
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
        {
            if( mask[x] )
                dst[x] = src[x];
            if( mask[x+1] )
                dst[x+1] = src[x+1];
            if( mask[x+2] )
                dst[x+2] = src[x+2];
            if( mask[x+3] )
                dst[x+3] = src[x+3];
        }
        for( ; x < size.width; x++ )
            if( mask[x] )
                dst[x] = src[x];
    }
}

template<typename T> static void
setMask_(const void* _scalar, Mat& dstmat, const Mat& maskmat)
{
    T scalar = *(T*)_scalar;
    const uchar* mask = maskmat.data;
    size_t dstep = dstmat.step;
    size_t mstep = maskmat.step;
    Size size = dstmat.size();

    if( dstmat.isContinuous() && maskmat.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( int y = 0; y < size.height; y++, mask += mstep )
    {
        T* dst = (T*)(dstmat.data + dstep*y);
        int x = 0;
        for( ; x <= size.width - 4; x += 4 )
        {
            if( mask[x] )
                dst[x] = scalar;
            if( mask[x+1] )
                dst[x+1] = scalar;
            if( mask[x+2] )
                dst[x+2] = scalar;
            if( mask[x+3] )
                dst[x+3] = scalar;
        }
        for( ; x < size.width; x++ )
            if( mask[x] )
                dst[x] = scalar;
    }
}

typedef void (*SetMaskFunc)(const void* scalar, Mat& dst, const Mat& mask);

CopyMaskFunc g_copyMaskFuncTab[] =
{
    0,
    copyMask_<uchar>, // 1
    copyMask_<ushort>, // 2
    copyMask_<Vec<uchar,3> >, // 3
    copyMask_<int>, // 4
    0,
    copyMask_<Vec<ushort,3> >, // 6
    0,
    copyMask_<int64>, // 8
    0, 0, 0,
    copyMask_<Vec<int,3> >, // 12
    0, 0, 0,
    copyMask_<Vec<int64,2> >, // 16
    0, 0, 0, 0, 0, 0, 0,
    copyMask_<Vec<int64,3> >, // 24
    0, 0, 0, 0, 0, 0, 0,
    copyMask_<Vec<int64,4> > // 32
};

static SetMaskFunc setMaskFuncTab[] =
{
    0,
    setMask_<uchar>, // 1
    setMask_<ushort>, // 2
    setMask_<Vec<uchar,3> >, // 3
    setMask_<int>, // 4
    0,
    setMask_<Vec<ushort,3> >, // 6
    0,
    setMask_<int64>, // 8
    0, 0, 0,
    setMask_<Vec<int,3> >, // 12
    0, 0, 0,
    setMask_<Vec<int64,2> >, // 16
    0, 0, 0, 0, 0, 0, 0,
    setMask_<Vec<int64,3> >, // 24
    0, 0, 0, 0, 0, 0, 0,
    setMask_<Vec<int64,4> > // 32
};


/* dst = src */
void Mat::copyTo( Mat& dst ) const
{
    if( data == dst.data && data != 0 )
        return;
    
    if( dims > 2 )
    {
        dst.create( dims, size, type() );
        if( total() != 0 )
        {
            const Mat* arrays[] = { this, &dst, 0 };
            Mat planes[2];
            NAryMatIterator it(arrays, planes);
            CV_DbgAssert(it.planes[0].isContinuous() &&
                         it.planes[1].isContinuous());
            size_t planeSize = it.planes[0].elemSize()*it.planes[0].rows*it.planes[0].cols;
        
            for( int i = 0; i < it.nplanes; i++, ++it )
                memcpy(it.planes[1].data, it.planes[0].data, planeSize);
        }
        return;
    }
    
    dst.create( rows, cols, type() );
    Size sz = size();
    
    if( rows > 0 && cols > 0 )
    {
        const uchar* sptr = data;
        uchar* dptr = dst.data;

        size_t width = sz.width*elemSize();
        if( isContinuous() && dst.isContinuous() )
        {
            width *= sz.height;
            sz.height = 1;
        }

        for( ; sz.height--; sptr += step, dptr += dst.step )
            memcpy( dptr, sptr, width );
    }
}

void Mat::copyTo( Mat& dst, const Mat& mask ) const
{
    if( !mask.data )
    {
        copyTo(dst);
        return;
    }
    
    if( dims > 2 )
    {
        dst.create( dims, size, type() );
        const Mat* arrays[] = { this, &dst, &mask, 0 };
        Mat planes[3];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            it.planes[0].copyTo(it.planes[1], it.planes[2]);
        return;
    }

    uchar* data0 = dst.data;
    dst.create( size(), type() );
    if( dst.data != data0 ) // do not leave dst uninitialized
        dst = Scalar(0);
    getCopyMaskFunc((int)elemSize())(*this, dst, mask);
}

Mat& Mat::operator = (const Scalar& s)
{
    if( dims > 2 )
    {
        const Mat* arrays[] = { this, 0 };
        Mat planes[1];
        NAryMatIterator it(arrays, planes);
        
        for( int i = 0; i < it.nplanes; i++, ++it )
            it.planes[0] = s;
        return *this;
    }
    
    Size sz = size();
    uchar* dst = data;

    sz.width *= (int)elemSize();
    if( isContinuous() )
    {
        sz.width *= sz.height;
        sz.height = 1;
    }
    
    if( s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 )
    {
        for( ; sz.height--; dst += step )
            memset( dst, 0, sz.width );
    }
    else
    {
        int t = type(), esz1 = (int)elemSize1();
        double scalar[12];
        scalarToRawData(s, scalar, t, 12);
        int copy_len = 12*esz1;
        uchar* dst_limit = dst + sz.width;
        
        if( sz.height-- )
        {
            while( dst + copy_len <= dst_limit )
            {
                memcpy( dst, scalar, copy_len );
                dst += copy_len;
            }
            memcpy( dst, scalar, dst_limit - dst );
        }

        if( sz.height > 0 )
        {
            dst = dst_limit - sz.width + step;
            for( ; sz.height--; dst += step )
                memcpy( dst, data, sz.width );
        }
    }
    return *this;
}

Mat& Mat::setTo(const Scalar& s, const Mat& mask)
{
    if( !mask.data )
        *this = s;
    else
    {
        CV_Assert( channels() <= 4 );
        SetMaskFunc func = setMaskFuncTab[elemSize()];
        CV_Assert( func != 0 );
        double buf[4];
        scalarToRawData(s, buf, type(), 0);
        
        if( dims > 2 )
        {
            const Mat* arrays[] = { this, &mask, 0 };
            Mat planes[2];
            NAryMatIterator it(arrays, planes);
            
            for( int i = 0; i < it.nplanes; i++, ++it )
                func(buf, it.planes[0], it.planes[1]);
        }
        else
            func(buf, *this, mask);
    }
    return *this;
}


template<typename T> static void
flipHoriz_( const Mat& srcmat, Mat& dstmat, bool flipv )
{
    uchar* dst0 = dstmat.data;
    size_t srcstep = srcmat.step;
    int dststep = (int)dstmat.step;
    Size size = srcmat.size();

    if( flipv )
    {
        dst0 += (size.height - 1)*dststep;
        dststep = -dststep;
    }

    for( int y = 0; y < size.height; y++ )
    {
        const T* src = (const T*)(srcmat.data + srcstep*y);
        T* dst = (T*)(dst0 + dststep*y);

        for( int i = 0; i < (size.width + 1)/2; i++ )
        {
            T t0 = src[i], t1 = src[size.width - i - 1];
            dst[i] = t1; dst[size.width - i - 1] = t0;
        }
    }
}

typedef void (*FlipHorizFunc)( const Mat& src, Mat& dst, bool flipv );

static void
flipVert( const Mat& srcmat, Mat& dstmat )
{
    const uchar* src = srcmat.data;
    uchar* dst = dstmat.data;
    size_t srcstep = srcmat.step, dststep = dstmat.step;
    Size size = srcmat.size();
    const uchar* src1 = src + (size.height - 1)*srcstep;
    uchar* dst1 = dst + (size.height - 1)*dststep;
    size.width *= (int)srcmat.elemSize();

    for( int y = 0; y < (size.height + 1)/2; y++, src += srcstep, src1 -= srcstep,
                                              dst += dststep, dst1 -= dststep )
    {
        int i = 0;
        if( ((size_t)(src)|(size_t)(dst)|(size_t)src1|(size_t)dst1) % sizeof(int) == 0 )
        {
            for( ; i <= size.width - 16; i += 16 )
            {
                int t0 = ((int*)(src + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;

                t0 = ((int*)(src + i))[1];
                t1 = ((int*)(src1 + i))[1];

                ((int*)(dst + i))[1] = t1;
                ((int*)(dst1 + i))[1] = t0;

                t0 = ((int*)(src + i))[2];
                t1 = ((int*)(src1 + i))[2];

                ((int*)(dst + i))[2] = t1;
                ((int*)(dst1 + i))[2] = t0;

                t0 = ((int*)(src + i))[3];
                t1 = ((int*)(src1 + i))[3];

                ((int*)(dst + i))[3] = t1;
                ((int*)(dst1 + i))[3] = t0;
            }

            for( ; i <= size.width - 4; i += 4 )
            {
                int t0 = ((int*)(src + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;
            }
        }

        for( ; i < size.width; i++ )
        {
            uchar t0 = src[i];
            uchar t1 = src1[i];

            dst[i] = t1;
            dst1[i] = t0;
        }
    }
}

void flip( const Mat& src, Mat& dst, int flip_mode )
{
    static FlipHorizFunc tab[] =
    {
        0,
        flipHoriz_<uchar>, // 1
        flipHoriz_<ushort>, // 2
        flipHoriz_<Vec<uchar,3> >, // 3
        flipHoriz_<int>, // 4
        0,
        flipHoriz_<Vec<ushort,3> >, // 6
        0,
        flipHoriz_<int64>, // 8
        0, 0, 0,
        flipHoriz_<Vec<int,3> >, // 12
        0, 0, 0,
        flipHoriz_<Vec<int64,2> >, // 16
        0, 0, 0, 0, 0, 0, 0,
        flipHoriz_<Vec<int64,3> >, // 24
        0, 0, 0, 0, 0, 0, 0,
        flipHoriz_<Vec<int64,4> > // 32
    };
    
    CV_Assert( src.dims <= 2 );
    dst.create( src.size(), src.type() );

    if( flip_mode == 0 )
        flipVert( src, dst );
    else
    {
        int esz = (int)src.elemSize();
        CV_Assert( esz <= 32 );
        FlipHorizFunc func = tab[esz];
        CV_Assert( func != 0 );

        if( flip_mode > 0 )
            func( src, dst, false );
        else if( src.data != dst.data )
            func( src, dst, true );
        else
        {
            func( dst, dst, false );
            flipVert( dst, dst );
        }
    }
}


void repeat(const Mat& src, int ny, int nx, Mat& dst)
{
    CV_Assert( src.dims <= 2 );
    
    dst.create(src.rows*ny, src.cols*nx, src.type());
    Size ssize = src.size(), dsize = dst.size();
    int esz = (int)src.elemSize();
    int x, y;
    ssize.width *= esz; dsize.width *= esz;

    for( y = 0; y < ssize.height; y++ )
    {
        for( x = 0; x < dsize.width; x += ssize.width )
            memcpy( dst.data + y*dst.step + x, src.data + y*src.step, ssize.width );
    }

    for( ; y < dsize.height; y++ )
        memcpy( dst.data + y*dst.step, dst.data + (y - ssize.height)*dst.step, dsize.width );
}

}

/* dst = src */
CV_IMPL void
cvCopy( const void* srcarr, void* dstarr, const void* maskarr )
{
    if( CV_IS_SPARSE_MAT(srcarr) && CV_IS_SPARSE_MAT(dstarr))
    {
        CV_Assert( maskarr == 0 );
        CvSparseMat* src1 = (CvSparseMat*)srcarr;
        CvSparseMat* dst1 = (CvSparseMat*)dstarr;
        CvSparseMatIterator iterator;
        CvSparseNode* node;

        dst1->dims = src1->dims;
        memcpy( dst1->size, src1->size, src1->dims*sizeof(src1->size[0]));
        dst1->valoffset = src1->valoffset;
        dst1->idxoffset = src1->idxoffset;
        cvClearSet( dst1->heap );

        if( src1->heap->active_count >= dst1->hashsize*CV_SPARSE_HASH_RATIO )
        {
            cvFree( &dst1->hashtable );
            dst1->hashsize = src1->hashsize;
            dst1->hashtable =
                (void**)cvAlloc( dst1->hashsize*sizeof(dst1->hashtable[0]));
        }

        memset( dst1->hashtable, 0, dst1->hashsize*sizeof(dst1->hashtable[0]));

        for( node = cvInitSparseMatIterator( src1, &iterator );
             node != 0; node = cvGetNextSparseNode( &iterator ))
        {
            CvSparseNode* node_copy = (CvSparseNode*)cvSetNew( dst1->heap );
            int tabidx = node->hashval & (dst1->hashsize - 1);
            CV_MEMCPY_AUTO( node_copy, node, dst1->heap->elem_size );
            node_copy->next = (CvSparseNode*)dst1->hashtable[tabidx];
            dst1->hashtable[tabidx] = node_copy;
        }
        return;
    }
    cv::Mat src = cv::cvarrToMat(srcarr, false, true, 1), dst = cv::cvarrToMat(dstarr, false, true, 1);
    CV_Assert( src.depth() == dst.depth() && src.size() == dst.size() );
    
    int coi1 = 0, coi2 = 0;
    if( CV_IS_IMAGE(srcarr) )
        coi1 = cvGetImageCOI((const IplImage*)srcarr);
    if( CV_IS_IMAGE(dstarr) )
        coi2 = cvGetImageCOI((const IplImage*)dstarr);
    
    if( coi1 || coi2 )
    {
        CV_Assert( (coi1 != 0 || src.channels() == 1) &&
            (coi2 != 0 || dst.channels() == 1) );
        
        int pair[] = { std::max(coi1-1, 0), std::max(coi2-1, 0) };
        cv::mixChannels( &src, 1, &dst, 1, pair, 1 );
        return;
    }
    else
        CV_Assert( src.channels() == dst.channels() );
    
    if( !maskarr )
        src.copyTo(dst);
    else
        src.copyTo(dst, cv::cvarrToMat(maskarr));
}

CV_IMPL void
cvSet( void* arr, CvScalar value, const void* maskarr )
{
    cv::Mat m = cv::cvarrToMat(arr);
    if( !maskarr )
        m = value;
    else
        m.setTo(value, cv::cvarrToMat(maskarr));
}

CV_IMPL void
cvSetZero( CvArr* arr )
{
    if( CV_IS_SPARSE_MAT(arr) )
    {
        CvSparseMat* mat1 = (CvSparseMat*)arr;
        cvClearSet( mat1->heap );
        if( mat1->hashtable )
            memset( mat1->hashtable, 0, mat1->hashsize*sizeof(mat1->hashtable[0]));
        return;
    }
    cv::Mat m = cv::cvarrToMat(arr);
    m = cv::Scalar(0);
}

CV_IMPL void
cvFlip( const CvArr* srcarr, CvArr* dstarr, int flip_mode )
{
    cv::Mat src = cv::cvarrToMat(srcarr);
    cv::Mat dst;
    
    if (!dstarr)
      dst = src;
    else
      dst = cv::cvarrToMat(dstarr);
    
    CV_Assert( src.type() == dst.type() && src.size() == dst.size() );
    cv::flip( src, dst, flip_mode );
}

CV_IMPL void
cvRepeat( const CvArr* srcarr, CvArr* dstarr )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.type() == dst.type() &&
        dst.rows % src.rows == 0 && dst.cols % src.cols == 0 );
    cv::repeat(src, dst.rows/src.rows, dst.cols/src.cols, dst);
}

/* End of file. */
