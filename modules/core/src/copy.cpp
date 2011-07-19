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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
copyMask_(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep, Size size)
{
    for( ; size.height--; mask += mstep, _src += sstep, _dst += dstep )
    {
        const T* src = (const T*)_src;
        T* dst = (T*)_dst;
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

static void
copyMaskGeneric(const uchar* _src, size_t sstep, const uchar* mask, size_t mstep, uchar* _dst, size_t dstep, Size size, void* _esz)
{
    size_t k, esz = *(size_t*)_esz;
    for( ; size.height--; mask += mstep, _src += sstep, _dst += dstep )
    {
        const uchar* src = _src;
        uchar* dst = _dst;
        int x = 0;
        for( ; x < size.width; x++, src += esz, dst += esz )
        {
            if( !mask[x] )
                continue;
            for( k = 0; k < esz; k++ )
                dst[k] = src[k];
        }
    }
}
    
    
#define DEF_COPY_MASK(suffix, type) \
static void copyMask##suffix(const uchar* src, size_t sstep, const uchar* mask, size_t mstep, \
                             uchar* dst, size_t dstep, Size size, void*) \
{ \
    copyMask_<type>(src, sstep, mask, mstep, dst, dstep, size); \
}
    
    
DEF_COPY_MASK(8u, uchar);
DEF_COPY_MASK(16u, ushort);
DEF_COPY_MASK(8uC3, Vec3b);
DEF_COPY_MASK(32s, int);
DEF_COPY_MASK(16uC3, Vec3s);
DEF_COPY_MASK(32sC2, Vec2i);
DEF_COPY_MASK(32sC3, Vec3i);
DEF_COPY_MASK(32sC4, Vec4i);
DEF_COPY_MASK(32sC6, Vec6i);
DEF_COPY_MASK(32sC8, Vec8i);
    
BinaryFunc copyMaskTab[] =
{
    0,
    copyMask8u,
    copyMask16u,
    copyMask8uC3,
    copyMask32s,
    0,
    copyMask16uC3,
    0,
    copyMask32sC2,
    0, 0, 0,
    copyMask32sC3,
    0, 0, 0,
    copyMask32sC4,
    0, 0, 0, 0, 0, 0, 0,
    copyMask32sC6,
    0, 0, 0, 0, 0, 0, 0,
    copyMask32sC8
};
    
BinaryFunc getCopyMaskFunc(size_t esz)
{
    return esz <= 32 && copyMaskTab[esz] ? copyMaskTab[esz] : copyMaskGeneric;
}

/* dst = src */
void Mat::copyTo( OutputArray _dst ) const
{
    int dtype = _dst.type();
    if( _dst.fixedType() && dtype != type() )
    {
        convertTo( _dst, dtype );
        return;
    }
    
    if( empty() )
    {
        _dst.release();
        return;
    }
    
    if( dims <= 2 )
    {
        _dst.create( rows, cols, type() );
        Mat dst = _dst.getMat();
        if( data == dst.data )
            return;
        
        if( rows > 0 && cols > 0 )
        {
            const uchar* sptr = data;
            uchar* dptr = dst.data;
            
            // to handle the copying 1xn matrix => nx1 std vector.
            Size sz = size() == dst.size() ?
                getContinuousSize(*this, dst) :
                getContinuousSize(*this);
            size_t len = sz.width*elemSize();
            
            for( ; sz.height--; sptr += step, dptr += dst.step )
                memcpy( dptr, sptr, len );
        }
        return;
    }
    
    _dst.create( dims, size, type() );
    Mat dst = _dst.getMat();
    if( data == dst.data )
        return;
    
    if( total() != 0 )
    {
        const Mat* arrays[] = { this, &dst };
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs, 2);
        size_t size = it.size*elemSize();
    
        for( size_t i = 0; i < it.nplanes; i++, ++it )
            memcpy(ptrs[1], ptrs[0], size);
    }
}

void Mat::copyTo( OutputArray _dst, InputArray _mask ) const
{
    Mat mask = _mask.getMat();
    if( !mask.data )
    {
        copyTo(_dst);
        return;
    }
    
    CV_Assert( mask.type() == CV_8U );
    
    size_t esz = elemSize();
    BinaryFunc copymask = getCopyMaskFunc(esz);
    
    uchar* data0 = _dst.getMat().data;
    _dst.create( dims, size, type() );
    Mat dst = _dst.getMat();
    
    if( dst.data != data0 ) // do not leave dst uninitialized
        dst = Scalar(0);
    
    if( dims <= 2 )
    {
        Size sz = getContinuousSize(*this, dst, mask);
        copymask(data, step, mask.data, mask.step, dst.data, dst.step, sz, &esz);
        return;
    }
    
    const Mat* arrays[] = { this, &dst, &mask, 0 };
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    Size sz((int)it.size, 1);
    
    for( size_t i = 0; i < it.nplanes; i++, ++it )
        copymask(ptrs[0], 0, ptrs[2], 0, ptrs[1], 0, sz, &esz);
}

Mat& Mat::operator = (const Scalar& s)
{
    const Mat* arrays[] = { this };
    uchar* ptr;
    NAryMatIterator it(arrays, &ptr, 1);
    size_t size = it.size*elemSize();
    
    if( s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0 )
    {
        for( size_t i = 0; i < it.nplanes; i++, ++it )
            memset( ptr, 0, size );
    }
    else
    {
        if( it.nplanes > 0 )
        {
            double scalar[12];
            scalarToRawData(s, scalar, type(), 12);
            size_t blockSize = 12*elemSize1();
            
            for( size_t j = 0; j < size; j += blockSize )
            {
                size_t sz = MIN(blockSize, size - j);
                memcpy( ptr + j, scalar, sz );
            }
        }
        
        for( size_t i = 1; i < it.nplanes; i++ )
        {
            ++it;
            memcpy( ptr, data, size );
        }
    }
    return *this;
}

    
Mat& Mat::setTo(InputArray _value, InputArray _mask)
{
    if( !data )
        return *this;
    
    Mat value = _value.getMat(), mask = _mask.getMat();
    
    CV_Assert( checkScalar(value, type(), _value.kind(), _InputArray::MAT ));
    CV_Assert( mask.empty() || mask.type() == CV_8U );
    
    size_t esz = elemSize();
    BinaryFunc copymask = getCopyMaskFunc(esz);
    
    const Mat* arrays[] = { this, !mask.empty() ? &mask : 0, 0 };
    uchar* ptrs[2]={0,0};
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, blockSize0 = std::min(total, (int)((BLOCK_SIZE + esz-1)/esz));
    AutoBuffer<uchar> _scbuf(blockSize0*esz + 32);
    uchar* scbuf = alignPtr((uchar*)_scbuf, (int)sizeof(double));
    convertAndUnrollScalar( value, type(), scbuf, blockSize0 );
    
    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( int j = 0; j < total; j += blockSize0 )
        {
            Size sz(std::min(blockSize0, total - j), 1);
            size_t blockSize = sz.width*esz;
            if( ptrs[1] )
            {
                copymask(scbuf, 0, ptrs[1], 0, ptrs[0], 0, sz, &esz);
                ptrs[1] += sz.width;
            }
            else
                memcpy(ptrs[0], scbuf, blockSize);
            ptrs[0] += blockSize;
        }
    }
    return *this;
}


static void
flipHoriz( const uchar* src, size_t sstep, uchar* dst, size_t dstep, Size size, size_t esz )
{
    int i, j, limit = (int)(((size.width + 1)/2)*esz);
    AutoBuffer<int> _tab(size.width*esz);
    int* tab = _tab;
    
    for( i = 0; i < size.width; i++ )
        for( size_t k = 0; k < esz; k++ )
            tab[i*esz + k] = (int)((size.width - i - 1)*esz + k);

    for( ; size.height--; src += sstep, dst += dstep )
    {
        for( i = 0; i < limit; i++ )
        {
            j = tab[i];
            uchar t0 = src[i], t1 = src[j];
            dst[i] = t1; dst[j] = t0;
        }
    }
}

static void
flipVert( const uchar* src0, size_t sstep, uchar* dst0, size_t dstep, Size size, size_t esz )
{
    const uchar* src1 = src0 + (size.height - 1)*sstep;
    uchar* dst1 = dst0 + (size.height - 1)*dstep;
    size.width *= (int)esz;

    for( int y = 0; y < (size.height + 1)/2; y++, src0 += sstep, src1 -= sstep,
                                                  dst0 += dstep, dst1 -= dstep )
    {
        int i = 0;
        if( ((size_t)src0|(size_t)dst0|(size_t)src1|(size_t)dst1) % sizeof(int) == 0 )
        {
            for( ; i <= size.width - 16; i += 16 )
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;

                t0 = ((int*)(src0 + i))[1];
                t1 = ((int*)(src1 + i))[1];

                ((int*)(dst0 + i))[1] = t1;
                ((int*)(dst1 + i))[1] = t0;

                t0 = ((int*)(src0 + i))[2];
                t1 = ((int*)(src1 + i))[2];

                ((int*)(dst0 + i))[2] = t1;
                ((int*)(dst1 + i))[2] = t0;

                t0 = ((int*)(src0 + i))[3];
                t1 = ((int*)(src1 + i))[3];

                ((int*)(dst0 + i))[3] = t1;
                ((int*)(dst1 + i))[3] = t0;
            }

            for( ; i <= size.width - 4; i += 4 )
            {
                int t0 = ((int*)(src0 + i))[0];
                int t1 = ((int*)(src1 + i))[0];

                ((int*)(dst0 + i))[0] = t1;
                ((int*)(dst1 + i))[0] = t0;
            }
        }

        for( ; i < size.width; i++ )
        {
            uchar t0 = src0[i];
            uchar t1 = src1[i];

            dst0[i] = t1;
            dst1[i] = t0;
        }
    }
}

void flip( InputArray _src, OutputArray _dst, int flip_mode )
{
    Mat src = _src.getMat();
    
    CV_Assert( src.dims <= 2 );
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();
    size_t esz = src.elemSize();

    if( flip_mode <= 0 )
        flipVert( src.data, src.step, dst.data, dst.step, src.size(), esz );
    else
        flipHoriz( src.data, src.step, dst.data, dst.step, src.size(), esz );
    
    if( flip_mode < 0 )
        flipHoriz( dst.data, dst.step, dst.data, dst.step, dst.size(), esz );
}


void repeat(InputArray _src, int ny, int nx, OutputArray _dst)
{
    Mat src = _src.getMat();
    CV_Assert( src.dims <= 2 );
    
    _dst.create(src.rows*ny, src.cols*nx, src.type());
    Mat dst = _dst.getMat();
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

Mat repeat(const Mat& src, int ny, int nx)
{
    if( nx == 1 && ny == 1 )
        return src;
    Mat dst;
    repeat(src, ny, nx, dst);
    return dst;
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
            memcpy( node_copy, node, dst1->heap->elem_size );
            node_copy->next = (CvSparseNode*)dst1->hashtable[tabidx];
            dst1->hashtable[tabidx] = node_copy;
        }
        return;
    }
    cv::Mat src = cv::cvarrToMat(srcarr, false, true, 1), dst = cv::cvarrToMat(dstarr, false, true, 1);
    CV_Assert( src.depth() == dst.depth() && src.size == dst.size );
    
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
        m.setTo(cv::Scalar(value), cv::cvarrToMat(maskarr));
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
