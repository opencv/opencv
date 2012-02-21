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

/****************************************************************************************\
                                    Base Image Filter
\****************************************************************************************/

/*
 Various border types, image boundaries are denoted with '|'
 
 * BORDER_REPLICATE:     aaaaaa|abcdefgh|hhhhhhh
 * BORDER_REFLECT:       fedcba|abcdefgh|hgfedcb
 * BORDER_REFLECT_101:   gfedcb|abcdefgh|gfedcba
 * BORDER_WRAP:          cdefgh|abcdefgh|abcdefg        
 * BORDER_CONSTANT:      iiiiii|abcdefgh|iiiiiii  with some specified 'i'
 */
int cv::borderInterpolate( int p, int len, int borderType )
{
    if( (unsigned)p < (unsigned)len )
        ;
    else if( borderType == BORDER_REPLICATE )
        p = p < 0 ? 0 : len - 1;
    else if( borderType == BORDER_REFLECT || borderType == BORDER_REFLECT_101 )
    {
        int delta = borderType == BORDER_REFLECT_101;
        if( len == 1 )
            return 0;
        do
        {
            if( p < 0 )
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        }
        while( (unsigned)p >= (unsigned)len );
    }
    else if( borderType == BORDER_WRAP )
    {
        if( p < 0 )
            p -= ((p-len+1)/len)*len;
        if( p >= len )
            p %= len;
    }
    else if( borderType == BORDER_CONSTANT )
        p = -1;
    else
        CV_Error( CV_StsBadArg, "Unknown/unsupported border type" );
    return p;
}


namespace cv
{

BaseRowFilter::BaseRowFilter() { ksize = anchor = -1; }
BaseRowFilter::~BaseRowFilter() {}

BaseColumnFilter::BaseColumnFilter() { ksize = anchor = -1; }
BaseColumnFilter::~BaseColumnFilter() {}
void BaseColumnFilter::reset() {}

BaseFilter::BaseFilter() { ksize = Size(-1,-1); anchor = Point(-1,-1); }
BaseFilter::~BaseFilter() {}
void BaseFilter::reset() {}

FilterEngine::FilterEngine()
{
    srcType = dstType = bufType = -1;
    rowBorderType = columnBorderType = BORDER_REPLICATE;
    bufStep = startY = startY0 = endY = rowCount = dstY = 0;
    maxWidth = 0;

    wholeSize = Size(-1,-1);
}
    

FilterEngine::FilterEngine( const Ptr<BaseFilter>& _filter2D,
                            const Ptr<BaseRowFilter>& _rowFilter,
                            const Ptr<BaseColumnFilter>& _columnFilter,
                            int _srcType, int _dstType, int _bufType,
                            int _rowBorderType, int _columnBorderType,
                            const Scalar& _borderValue )
{
    init(_filter2D, _rowFilter, _columnFilter, _srcType, _dstType, _bufType,
         _rowBorderType, _columnBorderType, _borderValue);
}
    
FilterEngine::~FilterEngine()
{
}


void FilterEngine::init( const Ptr<BaseFilter>& _filter2D,
                         const Ptr<BaseRowFilter>& _rowFilter,
                         const Ptr<BaseColumnFilter>& _columnFilter,
                         int _srcType, int _dstType, int _bufType,
                         int _rowBorderType, int _columnBorderType,
                         const Scalar& _borderValue )
{
    _srcType = CV_MAT_TYPE(_srcType);
    _bufType = CV_MAT_TYPE(_bufType);
    _dstType = CV_MAT_TYPE(_dstType);
        
    srcType = _srcType;
    int srcElemSize = (int)getElemSize(srcType);
    dstType = _dstType;
    bufType = _bufType;
    
    filter2D = _filter2D;
    rowFilter = _rowFilter;
    columnFilter = _columnFilter;

    if( _columnBorderType < 0 )
        _columnBorderType = _rowBorderType;
    
    rowBorderType = _rowBorderType;
    columnBorderType = _columnBorderType;
    
    CV_Assert( columnBorderType != BORDER_WRAP );
    
    if( isSeparable() )
    {
        CV_Assert( !rowFilter.empty() && !columnFilter.empty() );
        ksize = Size(rowFilter->ksize, columnFilter->ksize);
        anchor = Point(rowFilter->anchor, columnFilter->anchor);
    }
    else
    {
        CV_Assert( bufType == srcType );
        ksize = filter2D->ksize;
        anchor = filter2D->anchor;
    }

    CV_Assert( 0 <= anchor.x && anchor.x < ksize.width &&
               0 <= anchor.y && anchor.y < ksize.height );

    borderElemSize = srcElemSize/(CV_MAT_DEPTH(srcType) >= CV_32S ? sizeof(int) : 1);    
    int borderLength = std::max(ksize.width - 1, 1);
    borderTab.resize(borderLength*borderElemSize);

    maxWidth = bufStep = 0;
    constBorderRow.clear();

    if( rowBorderType == BORDER_CONSTANT || columnBorderType == BORDER_CONSTANT )
    {
        constBorderValue.resize(srcElemSize*borderLength);
        scalarToRawData(_borderValue, &constBorderValue[0], srcType,
                        borderLength*CV_MAT_CN(srcType));
    }

    wholeSize = Size(-1,-1);
}

static const int VEC_ALIGN = CV_MALLOC_ALIGN;

int FilterEngine::start(Size _wholeSize, Rect _roi, int _maxBufRows)
{
    int i, j;
    
    wholeSize = _wholeSize;
    roi = _roi;
    CV_Assert( roi.x >= 0 && roi.y >= 0 && roi.width >= 0 && roi.height >= 0 &&
        roi.x + roi.width <= wholeSize.width &&
        roi.y + roi.height <= wholeSize.height );

    int esz = (int)getElemSize(srcType);
    int bufElemSize = (int)getElemSize(bufType);
    const uchar* constVal = !constBorderValue.empty() ? &constBorderValue[0] : 0;

    if( _maxBufRows < 0 )
        _maxBufRows = ksize.height + 3;
    _maxBufRows = std::max(_maxBufRows, std::max(anchor.y, ksize.height-anchor.y-1)*2+1);

    if( maxWidth < roi.width || _maxBufRows != (int)rows.size() )
    {
        rows.resize(_maxBufRows);
        maxWidth = std::max(maxWidth, roi.width);
        int cn = CV_MAT_CN(srcType);
        srcRow.resize(esz*(maxWidth + ksize.width - 1));
        if( columnBorderType == BORDER_CONSTANT )
        {
            constBorderRow.resize(getElemSize(bufType)*(maxWidth + ksize.width - 1 + VEC_ALIGN));
            uchar *dst = alignPtr(&constBorderRow[0], VEC_ALIGN), *tdst;
            int n = (int)constBorderValue.size(), N;
            N = (maxWidth + ksize.width - 1)*esz;
            tdst = isSeparable() ? &srcRow[0] : dst;
            
            for( i = 0; i < N; i += n )
            {
                n = std::min( n, N - i );
                for(j = 0; j < n; j++)
                    tdst[i+j] = constVal[j];
            }

            if( isSeparable() )
                (*rowFilter)(&srcRow[0], dst, maxWidth, cn);
        }
        
        int maxBufStep = bufElemSize*(int)alignSize(maxWidth +
            (!isSeparable() ? ksize.width - 1 : 0),VEC_ALIGN);
        ringBuf.resize(maxBufStep*rows.size()+VEC_ALIGN);
    }

    // adjust bufstep so that the used part of the ring buffer stays compact in memory
    bufStep = bufElemSize*(int)alignSize(roi.width + (!isSeparable() ? ksize.width - 1 : 0),16);

    dx1 = std::max(anchor.x - roi.x, 0);
    dx2 = std::max(ksize.width - anchor.x - 1 + roi.x + roi.width - wholeSize.width, 0);

    // recompute border tables
    if( dx1 > 0 || dx2 > 0 )
    {
        if( rowBorderType == BORDER_CONSTANT )
        {
            int nr = isSeparable() ? 1 : (int)rows.size();
            for( i = 0; i < nr; i++ )
            {
                uchar* dst = isSeparable() ? &srcRow[0] : alignPtr(&ringBuf[0],VEC_ALIGN) + bufStep*i;
                memcpy( dst, constVal, dx1*esz );
                memcpy( dst + (roi.width + ksize.width - 1 - dx2)*esz, constVal, dx2*esz );
            }
        }
        else
        {
            int xofs1 = std::min(roi.x, anchor.x) - roi.x;
            
            int btab_esz = borderElemSize, wholeWidth = wholeSize.width;
            int* btab = (int*)&borderTab[0];
            
            for( i = 0; i < dx1; i++ )
            {
                int p0 = (borderInterpolate(i-dx1, wholeWidth, rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[i*btab_esz + j] = p0 + j;
            }

            for( i = 0; i < dx2; i++ )
            {
                int p0 = (borderInterpolate(wholeWidth + i, wholeWidth, rowBorderType) + xofs1)*btab_esz;
                for( j = 0; j < btab_esz; j++ )
                    btab[(i + dx1)*btab_esz + j] = p0 + j;
            }
        }
    }

    rowCount = dstY = 0;
    startY = startY0 = std::max(roi.y - anchor.y, 0);
    endY = std::min(roi.y + roi.height + ksize.height - anchor.y - 1, wholeSize.height);
    if( !columnFilter.empty() )
        columnFilter->reset();
    if( !filter2D.empty() )
        filter2D->reset();

    return startY;
}


int FilterEngine::start(const Mat& src, const Rect& _srcRoi,
                        bool isolated, int maxBufRows)
{
    Rect srcRoi = _srcRoi;
    
    if( srcRoi == Rect(0,0,-1,-1) )
        srcRoi = Rect(0,0,src.cols,src.rows);
    
    CV_Assert( srcRoi.x >= 0 && srcRoi.y >= 0 &&
        srcRoi.width >= 0 && srcRoi.height >= 0 &&
        srcRoi.x + srcRoi.width <= src.cols &&
        srcRoi.y + srcRoi.height <= src.rows );

    Point ofs;
    Size wholeSize(src.cols, src.rows);
    if( !isolated )
        src.locateROI( wholeSize, ofs );
    start( wholeSize, srcRoi + ofs, maxBufRows );

    return startY - ofs.y;
}


int FilterEngine::remainingInputRows() const
{
    return endY - startY - rowCount;
}

int FilterEngine::remainingOutputRows() const
{
    return roi.height - dstY;
}

int FilterEngine::proceed( const uchar* src, int srcstep, int count,
                           uchar* dst, int dststep )
{
    CV_Assert( wholeSize.width > 0 && wholeSize.height > 0 );
    
    const int *btab = &borderTab[0];
    int esz = (int)getElemSize(srcType), btab_esz = borderElemSize;
    uchar** brows = &rows[0];
    int bufRows = (int)rows.size();
    int cn = CV_MAT_CN(bufType);
    int width = roi.width, kwidth = ksize.width;
    int kheight = ksize.height, ay = anchor.y;
    int _dx1 = dx1, _dx2 = dx2;
    int width1 = roi.width + kwidth - 1;
    int xofs1 = std::min(roi.x, anchor.x);
    bool isSep = isSeparable();
    bool makeBorder = (_dx1 > 0 || _dx2 > 0) && rowBorderType != BORDER_CONSTANT;
    int dy = 0, i = 0;

    src -= xofs1*esz;
    count = std::min(count, remainingInputRows());

    CV_Assert( src && dst && count > 0 );

    for(;; dst += dststep*i, dy += i)
    {
        int dcount = bufRows - ay - startY - rowCount + roi.y;
        dcount = dcount > 0 ? dcount : bufRows - kheight + 1;
        dcount = std::min(dcount, count);
        count -= dcount;
        for( ; dcount-- > 0; src += srcstep )
        {
            int bi = (startY - startY0 + rowCount) % bufRows;
            uchar* brow = alignPtr(&ringBuf[0], VEC_ALIGN) + bi*bufStep;
            uchar* row = isSep ? &srcRow[0] : brow;
            
            if( ++rowCount > bufRows )
            {
                --rowCount;
                ++startY;
            }

            memcpy( row + _dx1*esz, src, (width1 - _dx2 - _dx1)*esz );

            if( makeBorder )
            {
                if( btab_esz*(int)sizeof(int) == esz )
                {
                    const int* isrc = (const int*)src;
                    int* irow = (int*)row;

                    for( i = 0; i < _dx1*btab_esz; i++ )
                        irow[i] = isrc[btab[i]];
                    for( i = 0; i < _dx2*btab_esz; i++ )
                        irow[i + (width1 - _dx2)*btab_esz] = isrc[btab[i+_dx1*btab_esz]];
                }
                else
                {
                    for( i = 0; i < _dx1*esz; i++ )
                        row[i] = src[btab[i]];
                    for( i = 0; i < _dx2*esz; i++ )
                        row[i + (width1 - _dx2)*esz] = src[btab[i+_dx1*esz]];
                }
            }
            
            if( isSep )
                (*rowFilter)(row, brow, width, CV_MAT_CN(srcType));
        }

        int max_i = std::min(bufRows, roi.height - (dstY + dy) + (kheight - 1));
        for( i = 0; i < max_i; i++ )
        {
            int srcY = borderInterpolate(dstY + dy + i + roi.y - ay,
                            wholeSize.height, columnBorderType);
            if( srcY < 0 ) // can happen only with constant border type
                brows[i] = alignPtr(&constBorderRow[0], VEC_ALIGN);
            else
            {
                CV_Assert( srcY >= startY );
                if( srcY >= startY + rowCount )
                    break;
                int bi = (srcY - startY0) % bufRows;
                brows[i] = alignPtr(&ringBuf[0], VEC_ALIGN) + bi*bufStep;
            }
        }
        if( i < kheight )
            break;
        i -= kheight - 1;
        if( isSeparable() )
            (*columnFilter)((const uchar**)brows, dst, dststep, i, roi.width*cn);
        else
            (*filter2D)((const uchar**)brows, dst, dststep, i, roi.width, cn);
    }

    dstY += dy;
    CV_Assert( dstY <= roi.height );
    return dy;
}


void FilterEngine::apply(const Mat& src, Mat& dst,
    const Rect& _srcRoi, Point dstOfs, bool isolated)
{
    CV_Assert( src.type() == srcType && dst.type() == dstType );
    
    Rect srcRoi = _srcRoi;
    if( srcRoi == Rect(0,0,-1,-1) )
        srcRoi = Rect(0,0,src.cols,src.rows);
    
    if( srcRoi.area() == 0 )
        return;

    CV_Assert( dstOfs.x >= 0 && dstOfs.y >= 0 &&
        dstOfs.x + srcRoi.width <= dst.cols &&
        dstOfs.y + srcRoi.height <= dst.rows );

    int y = start(src, srcRoi, isolated);
    proceed( src.data + y*src.step, (int)src.step, endY - startY,
             dst.data + dstOfs.y*dst.step + dstOfs.x*dst.elemSize(), (int)dst.step );
}

}

/****************************************************************************************\
*                                 Separable linear filter                                *
\****************************************************************************************/

int cv::getKernelType(InputArray filter_kernel, Point anchor)
{
    Mat _kernel = filter_kernel.getMat();
    CV_Assert( _kernel.channels() == 1 );
    int i, sz = _kernel.rows*_kernel.cols;

    Mat kernel;
    _kernel.convertTo(kernel, CV_64F);

    const double* coeffs = (double*)kernel.data;
    double sum = 0;
    int type = KERNEL_SMOOTH + KERNEL_INTEGER;
    if( (_kernel.rows == 1 || _kernel.cols == 1) &&
        anchor.x*2 + 1 == _kernel.cols &&
        anchor.y*2 + 1 == _kernel.rows )
        type |= (KERNEL_SYMMETRICAL + KERNEL_ASYMMETRICAL);

    for( i = 0; i < sz; i++ )
    {
        double a = coeffs[i], b = coeffs[sz - i - 1];
        if( a != b )
            type &= ~KERNEL_SYMMETRICAL;
        if( a != -b )
            type &= ~KERNEL_ASYMMETRICAL;
        if( a < 0 )
            type &= ~KERNEL_SMOOTH;
        if( a != saturate_cast<int>(a) )
            type &= ~KERNEL_INTEGER;
        sum += a;
    }

    if( fabs(sum - 1) > FLT_EPSILON*(fabs(sum) + 1) )
        type &= ~KERNEL_SMOOTH;
    return type;
}


namespace cv
{

struct RowNoVec
{
    RowNoVec() {}
    RowNoVec(const Mat&) {}
    int operator()(const uchar*, uchar*, int, int) const { return 0; }
};

struct ColumnNoVec
{
    ColumnNoVec() {}
    ColumnNoVec(const Mat&, int, int, double) {}
    int operator()(const uchar**, uchar*, int) const { return 0; }
};

struct SymmRowSmallNoVec
{
    SymmRowSmallNoVec() {}
    SymmRowSmallNoVec(const Mat&, int) {}
    int operator()(const uchar*, uchar*, int, int) const { return 0; }
};

struct SymmColumnSmallNoVec
{
    SymmColumnSmallNoVec() {}
    SymmColumnSmallNoVec(const Mat&, int, int, double) {}
    int operator()(const uchar**, uchar*, int) const { return 0; }
};

struct FilterNoVec
{
    FilterNoVec() {}
    FilterNoVec(const Mat&, int, double) {}
    int operator()(const uchar**, uchar*, int) const { return 0; }
};


#if CV_SSE2

///////////////////////////////////// 8u-16s & 8u-8u //////////////////////////////////

struct RowVec_8u32s
{
    RowVec_8u32s() { smallValues = false; }
    RowVec_8u32s( const Mat& _kernel )
    {
        kernel = _kernel;
        smallValues = true;
        int k, ksize = kernel.rows + kernel.cols - 1;
        for( k = 0; k < ksize; k++ )
        {
            int v = ((const int*)kernel.data)[k];
            if( v < SHRT_MIN || v > SHRT_MAX )
            {
                smallValues = false;
                break;
            }
        }
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;
        
        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        const int* _kx = (const int*)kernel.data;
        width *= cn;

        if( smallValues )
        {
            for( ; i <= width - 16; i += 16 )
            {
                const uchar* src = _src + i;
                __m128i f, z = _mm_setzero_si128(), s0 = z, s1 = z, s2 = z, s3 = z;
                __m128i x0, x1, x2, x3;

                for( k = 0; k < _ksize; k++, src += cn )
                {
                    f = _mm_cvtsi32_si128(_kx[k]);
                    f = _mm_shuffle_epi32(f, 0);
                    f = _mm_packs_epi32(f, f);

                    x0 = _mm_loadu_si128((const __m128i*)src);
                    x2 = _mm_unpackhi_epi8(x0, z);
                    x0 = _mm_unpacklo_epi8(x0, z);
                    x1 = _mm_mulhi_epi16(x0, f);
                    x3 = _mm_mulhi_epi16(x2, f);
                    x0 = _mm_mullo_epi16(x0, f);
                    x2 = _mm_mullo_epi16(x2, f);

                    s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
                    s1 = _mm_add_epi32(s1, _mm_unpackhi_epi16(x0, x1));
                    s2 = _mm_add_epi32(s2, _mm_unpacklo_epi16(x2, x3));
                    s3 = _mm_add_epi32(s3, _mm_unpackhi_epi16(x2, x3));
                }
                
                _mm_store_si128((__m128i*)(dst + i), s0);
                _mm_store_si128((__m128i*)(dst + i + 4), s1);
                _mm_store_si128((__m128i*)(dst + i + 8), s2);
                _mm_store_si128((__m128i*)(dst + i + 12), s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                const uchar* src = _src + i;
                __m128i f, z = _mm_setzero_si128(), s0 = z, x0, x1;

                for( k = 0; k < _ksize; k++, src += cn )
                {
                    f = _mm_cvtsi32_si128(_kx[k]);
                    f = _mm_shuffle_epi32(f, 0);
                    f = _mm_packs_epi32(f, f);

                    x0 = _mm_cvtsi32_si128(*(const int*)src);
                    x0 = _mm_unpacklo_epi8(x0, z);
                    x1 = _mm_mulhi_epi16(x0, f);
                    x0 = _mm_mullo_epi16(x0, f);
                    s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
                }
                _mm_store_si128((__m128i*)(dst + i), s0);
            }
        }
        return i;
    }

    Mat kernel;
    bool smallValues;
};


struct SymmRowSmallVec_8u32s
{
    SymmRowSmallVec_8u32s() { smallValues = false; }
    SymmRowSmallVec_8u32s( const Mat& _kernel, int _symmetryType )
    {
        kernel = _kernel;
        symmetryType = _symmetryType;
        smallValues = true;
        int k, ksize = kernel.rows + kernel.cols - 1;
        for( k = 0; k < ksize; k++ )
        {
            int v = ((const int*)kernel.data)[k];
            if( v < SHRT_MIN || v > SHRT_MAX )
            {
                smallValues = false;
                break;
            }
        }
    }

    int operator()(const uchar* src, uchar* _dst, int width, int cn) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;
        
        int i = 0, j, k, _ksize = kernel.rows + kernel.cols - 1;
        int* dst = (int*)_dst;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int* kx = (const int*)kernel.data + _ksize/2;
        if( !smallValues )
            return 0;

        src += (_ksize/2)*cn;
        width *= cn;

        __m128i z = _mm_setzero_si128();
        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn));
                        y0 = _mm_unpackhi_epi8(x0, z);
                        x0 = _mm_unpacklo_epi8(x0, z);
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);
                        y2 = _mm_unpackhi_epi8(x2, z);
                        x2 = _mm_unpacklo_epi8(x2, z);
                        x0 = _mm_add_epi16(x0, _mm_add_epi16(_mm_add_epi16(x1, x1), x2));
                        y0 = _mm_add_epi16(y0, _mm_add_epi16(_mm_add_epi16(y1, y1), y2));
                        _mm_store_si128((__m128i*)(dst + i), _mm_unpacklo_epi16(x0, z));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_unpackhi_epi16(x0, z));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_unpacklo_epi16(y0, z));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_unpackhi_epi16(y0, z));
                    }
                else if( kx[0] == -2 && kx[1] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn));
                        y0 = _mm_unpackhi_epi8(x0, z);
                        x0 = _mm_unpacklo_epi8(x0, z);
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);
                        y2 = _mm_unpackhi_epi8(x2, z);
                        x2 = _mm_unpacklo_epi8(x2, z);
                        x0 = _mm_add_epi16(x0, _mm_sub_epi16(x2, _mm_add_epi16(x1, x1)));
                        y0 = _mm_add_epi16(y0, _mm_sub_epi16(y2, _mm_add_epi16(y1, y1)));
                        _mm_store_si128((__m128i*)(dst + i), _mm_srai_epi32(_mm_unpacklo_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_srai_epi32(_mm_unpacklo_epi16(y0, y0),16));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_srai_epi32(_mm_unpackhi_epi16(y0, y0),16));
                    }
                else
                {
                    __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                            k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0);
                    k0 = _mm_packs_epi32(k0, k0);
                    k1 = _mm_packs_epi32(k1, k1);

                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, t0, t1, z0, z1, z2, z3;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn));
                        y0 = _mm_add_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x2, z));
                        x0 = _mm_add_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x2, z));
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);

                        t1 = _mm_mulhi_epi16(x1, k0);
                        t0 = _mm_mullo_epi16(x1, k0);
                        x2 = _mm_mulhi_epi16(x0, k1);
                        x0 = _mm_mullo_epi16(x0, k1);
                        z0 = _mm_unpacklo_epi16(t0, t1);
                        z1 = _mm_unpackhi_epi16(t0, t1);
                        z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(x0, x2));
                        z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(x0, x2));

                        t1 = _mm_mulhi_epi16(y1, k0);
                        t0 = _mm_mullo_epi16(y1, k0);
                        y1 = _mm_mulhi_epi16(y0, k1);
                        y0 = _mm_mullo_epi16(y0, k1);
                        z2 = _mm_unpacklo_epi16(t0, t1);
                        z3 = _mm_unpackhi_epi16(t0, t1);
                        z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                        z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));
                        _mm_store_si128((__m128i*)(dst + i), z0);
                        _mm_store_si128((__m128i*)(dst + i + 4), z1);
                        _mm_store_si128((__m128i*)(dst + i + 8), z2);
                        _mm_store_si128((__m128i*)(dst + i + 12), z3);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn*2));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn*2));
                        y0 = _mm_unpackhi_epi8(x0, z);
                        x0 = _mm_unpacklo_epi8(x0, z);
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);
                        y2 = _mm_unpackhi_epi8(x2, z);
                        x2 = _mm_unpacklo_epi8(x2, z);
                        x0 = _mm_add_epi16(x0, _mm_sub_epi16(x2, _mm_add_epi16(x1, x1)));
                        y0 = _mm_add_epi16(y0, _mm_sub_epi16(y2, _mm_add_epi16(y1, y1)));
                        _mm_store_si128((__m128i*)(dst + i), _mm_srai_epi32(_mm_unpacklo_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_srai_epi32(_mm_unpacklo_epi16(y0, y0),16));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_srai_epi32(_mm_unpackhi_epi16(y0, y0),16));
                    }
                else
                {
                    __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                            k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0),
                            k2 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[2]), 0);
                    k0 = _mm_packs_epi32(k0, k0);
                    k1 = _mm_packs_epi32(k1, k1);
                    k2 = _mm_packs_epi32(k2, k2);

                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, x2, y0, y1, t0, t1, z0, z1, z2, z3;
                        x0 = _mm_loadu_si128((__m128i*)(src - cn));
                        x1 = _mm_loadu_si128((__m128i*)src);
                        x2 = _mm_loadu_si128((__m128i*)(src + cn));
                        y0 = _mm_add_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x2, z));
                        x0 = _mm_add_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x2, z));
                        y1 = _mm_unpackhi_epi8(x1, z);
                        x1 = _mm_unpacklo_epi8(x1, z);

                        t1 = _mm_mulhi_epi16(x1, k0);
                        t0 = _mm_mullo_epi16(x1, k0);
                        x2 = _mm_mulhi_epi16(x0, k1);
                        x0 = _mm_mullo_epi16(x0, k1);
                        z0 = _mm_unpacklo_epi16(t0, t1);
                        z1 = _mm_unpackhi_epi16(t0, t1);
                        z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(x0, x2));
                        z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(x0, x2));

                        t1 = _mm_mulhi_epi16(y1, k0);
                        t0 = _mm_mullo_epi16(y1, k0);
                        y1 = _mm_mulhi_epi16(y0, k1);
                        y0 = _mm_mullo_epi16(y0, k1);
                        z2 = _mm_unpacklo_epi16(t0, t1);
                        z3 = _mm_unpackhi_epi16(t0, t1);
                        z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                        z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));

                        x0 = _mm_loadu_si128((__m128i*)(src - cn*2));
                        x1 = _mm_loadu_si128((__m128i*)(src + cn*2));
                        y1 = _mm_add_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x1, z));
                        y0 = _mm_add_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x1, z));

                        t1 = _mm_mulhi_epi16(y0, k2);
                        t0 = _mm_mullo_epi16(y0, k2);
                        y0 = _mm_mullo_epi16(y1, k2);
                        y1 = _mm_mulhi_epi16(y1, k2);
                        z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(t0, t1));
                        z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(t0, t1));
                        z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                        z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));

                        _mm_store_si128((__m128i*)(dst + i), z0);
                        _mm_store_si128((__m128i*)(dst + i + 4), z1);
                        _mm_store_si128((__m128i*)(dst + i + 8), z2);
                        _mm_store_si128((__m128i*)(dst + i + 12), z3);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, y0;
                        x0 = _mm_loadu_si128((__m128i*)(src + cn));
                        x1 = _mm_loadu_si128((__m128i*)(src - cn));
                        y0 = _mm_sub_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x1, z));
                        x0 = _mm_sub_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x1, z));
                        _mm_store_si128((__m128i*)(dst + i), _mm_srai_epi32(_mm_unpacklo_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 4), _mm_srai_epi32(_mm_unpackhi_epi16(x0, x0),16));
                        _mm_store_si128((__m128i*)(dst + i + 8), _mm_srai_epi32(_mm_unpacklo_epi16(y0, y0),16));
                        _mm_store_si128((__m128i*)(dst + i + 12), _mm_srai_epi32(_mm_unpackhi_epi16(y0, y0),16));
                    }
                else
                {
                    __m128i k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0);
                    k1 = _mm_packs_epi32(k1, k1);

                    for( ; i <= width - 16; i += 16, src += 16 )
                    {
                        __m128i x0, x1, y0, y1, z0, z1, z2, z3;
                        x0 = _mm_loadu_si128((__m128i*)(src + cn));
                        x1 = _mm_loadu_si128((__m128i*)(src - cn));
                        y0 = _mm_sub_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x1, z));
                        x0 = _mm_sub_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x1, z));

                        x1 = _mm_mulhi_epi16(x0, k1);
                        x0 = _mm_mullo_epi16(x0, k1);
                        z0 = _mm_unpacklo_epi16(x0, x1);
                        z1 = _mm_unpackhi_epi16(x0, x1);

                        y1 = _mm_mulhi_epi16(y0, k1);
                        y0 = _mm_mullo_epi16(y0, k1);
                        z2 = _mm_unpacklo_epi16(y0, y1);
                        z3 = _mm_unpackhi_epi16(y0, y1);
                        _mm_store_si128((__m128i*)(dst + i), z0);
                        _mm_store_si128((__m128i*)(dst + i + 4), z1);
                        _mm_store_si128((__m128i*)(dst + i + 8), z2);
                        _mm_store_si128((__m128i*)(dst + i + 12), z3);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                __m128i k0 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[0]), 0),
                        k1 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[1]), 0),
                        k2 = _mm_shuffle_epi32(_mm_cvtsi32_si128(kx[2]), 0);
                k0 = _mm_packs_epi32(k0, k0);
                k1 = _mm_packs_epi32(k1, k1);
                k2 = _mm_packs_epi32(k2, k2);

                for( ; i <= width - 16; i += 16, src += 16 )
                {
                    __m128i x0, x1, x2, y0, y1, t0, t1, z0, z1, z2, z3;
                    x0 = _mm_loadu_si128((__m128i*)(src + cn));
                    x2 = _mm_loadu_si128((__m128i*)(src - cn));
                    y0 = _mm_sub_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x2, z));
                    x0 = _mm_sub_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x2, z));

                    x2 = _mm_mulhi_epi16(x0, k1);
                    x0 = _mm_mullo_epi16(x0, k1);
                    z0 = _mm_unpacklo_epi16(x0, x2);
                    z1 = _mm_unpackhi_epi16(x0, x2);
                    y1 = _mm_mulhi_epi16(y0, k1);
                    y0 = _mm_mullo_epi16(y0, k1);
                    z2 = _mm_unpacklo_epi16(y0, y1);
                    z3 = _mm_unpackhi_epi16(y0, y1);

                    x0 = _mm_loadu_si128((__m128i*)(src + cn*2));
                    x1 = _mm_loadu_si128((__m128i*)(src - cn*2));
                    y1 = _mm_sub_epi16(_mm_unpackhi_epi8(x0, z), _mm_unpackhi_epi8(x1, z));
                    y0 = _mm_sub_epi16(_mm_unpacklo_epi8(x0, z), _mm_unpacklo_epi8(x1, z));

                    t1 = _mm_mulhi_epi16(y0, k2);
                    t0 = _mm_mullo_epi16(y0, k2);
                    y0 = _mm_mullo_epi16(y1, k2);
                    y1 = _mm_mulhi_epi16(y1, k2);
                    z0 = _mm_add_epi32(z0, _mm_unpacklo_epi16(t0, t1));
                    z1 = _mm_add_epi32(z1, _mm_unpackhi_epi16(t0, t1));
                    z2 = _mm_add_epi32(z2, _mm_unpacklo_epi16(y0, y1));
                    z3 = _mm_add_epi32(z3, _mm_unpackhi_epi16(y0, y1));

                    _mm_store_si128((__m128i*)(dst + i), z0);
                    _mm_store_si128((__m128i*)(dst + i + 4), z1);
                    _mm_store_si128((__m128i*)(dst + i + 8), z2);
                    _mm_store_si128((__m128i*)(dst + i + 12), z3);
                }
            }
        }

        src -= (_ksize/2)*cn;
        kx -= _ksize/2;
        for( ; i <= width - 4; i += 4, src += 4 )
        {
            __m128i f, s0 = z, x0, x1;

            for( k = j = 0; k < _ksize; k++, j += cn )
            {
                f = _mm_cvtsi32_si128(kx[k]);
                f = _mm_shuffle_epi32(f, 0);
                f = _mm_packs_epi32(f, f);

                x0 = _mm_cvtsi32_si128(*(const int*)(src + j));
                x0 = _mm_unpacklo_epi8(x0, z);
                x1 = _mm_mulhi_epi16(x0, f);
                x0 = _mm_mullo_epi16(x0, f);
                s0 = _mm_add_epi32(s0, _mm_unpacklo_epi16(x0, x1));
            }
            _mm_store_si128((__m128i*)(dst + i), s0);
        }

        return i;
    }

    Mat kernel;
    int symmetryType;
    bool smallValues;
};


struct SymmColumnVec_32s8u
{
    SymmColumnVec_32s8u() { symmetryType=0; }
    SymmColumnVec_32s8u(const Mat& _kernel, int _symmetryType, int _bits, double _delta)
    {
        symmetryType = _symmetryType;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;
        
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = (const float*)kernel.data + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const __m128i *S, *S2;
        __m128 d4 = _mm_set1_ps(delta);

        if( symmetrical )
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 s0, s1, s2, s3;
                __m128i x0, x1;
                S = (const __m128i*)(src[0] + i);
                s0 = _mm_cvtepi32_ps(_mm_load_si128(S));
                s1 = _mm_cvtepi32_ps(_mm_load_si128(S+1));
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);
                s1 = _mm_add_ps(_mm_mul_ps(s1, f), d4);
                s2 = _mm_cvtepi32_ps(_mm_load_si128(S+2));
                s3 = _mm_cvtepi32_ps(_mm_load_si128(S+3));
                s2 = _mm_add_ps(_mm_mul_ps(s2, f), d4);
                s3 = _mm_add_ps(_mm_mul_ps(s3, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    x1 = _mm_add_epi32(_mm_load_si128(S+1), _mm_load_si128(S2+1));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                    x0 = _mm_add_epi32(_mm_load_si128(S+2), _mm_load_si128(S2+2));
                    x1 = _mm_add_epi32(_mm_load_si128(S+3), _mm_load_si128(S2+3));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                }

                x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
                x0 = _mm_packus_epi16(x0, x1);
                _mm_storeu_si128((__m128i*)(dst + i), x0);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128i x0;
                __m128 s0 = _mm_cvtepi32_ps(_mm_load_si128((const __m128i*)(src[0] + i)));
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                }

                x0 = _mm_cvtps_epi32(s0);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
            }
        }
        else
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f, s0 = d4, s1 = d4, s2 = d4, s3 = d4;
                __m128i x0, x1;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    x1 = _mm_sub_epi32(_mm_load_si128(S+1), _mm_load_si128(S2+1));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                    x0 = _mm_sub_epi32(_mm_load_si128(S+2), _mm_load_si128(S2+2));
                    x1 = _mm_sub_epi32(_mm_load_si128(S+3), _mm_load_si128(S2+3));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(_mm_cvtepi32_ps(x1), f));
                }

                x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
                x0 = _mm_packus_epi16(x0, x1);
                _mm_storeu_si128((__m128i*)(dst + i), x0);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f, s0 = d4;
                __m128i x0;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = (const __m128i*)(src[k] + i);
                    S2 = (const __m128i*)(src[-k] + i);
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_epi32(_mm_load_si128(S), _mm_load_si128(S2));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0), f));
                }

                x0 = _mm_cvtps_epi32(s0);
                x0 = _mm_packs_epi32(x0, x0);
                x0 = _mm_packus_epi16(x0, x0);
                *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


struct SymmColumnSmallVec_32s16s
{
    SymmColumnSmallVec_32s16s() { symmetryType=0; }
    SymmColumnSmallVec_32s16s(const Mat& _kernel, int _symmetryType, int _bits, double _delta)
    {
        symmetryType = _symmetryType;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;
        
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = (const float*)kernel.data + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const int** src = (const int**)_src;
        const int *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        short* dst = (short*)_dst;
        __m128 df4 = _mm_set1_ps(delta);
        __m128i d4 = _mm_cvtps_epi32(df4);

        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_si128((__m128i*)(S0 + i));
                    s1 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S1 + i));
                    s3 = _mm_load_si128((__m128i*)(S1 + i + 4));
                    s4 = _mm_load_si128((__m128i*)(S2 + i));
                    s5 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s0 = _mm_add_epi32(s0, _mm_add_epi32(s4, _mm_add_epi32(s2, s2)));
                    s1 = _mm_add_epi32(s1, _mm_add_epi32(s5, _mm_add_epi32(s3, s3)));
                    s0 = _mm_add_epi32(s0, d4);
                    s1 = _mm_add_epi32(s1, d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_si128((__m128i*)(S0 + i));
                    s1 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S1 + i));
                    s3 = _mm_load_si128((__m128i*)(S1 + i + 4));
                    s4 = _mm_load_si128((__m128i*)(S2 + i));
                    s5 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s0 = _mm_add_epi32(s0, _mm_sub_epi32(s4, _mm_add_epi32(s2, s2)));
                    s1 = _mm_add_epi32(s1, _mm_sub_epi32(s5, _mm_add_epi32(s3, s3)));
                    s0 = _mm_add_epi32(s0, d4);
                    s1 = _mm_add_epi32(s1, d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else
            {
                __m128 k0 = _mm_set1_ps(ky[0]), k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1;
                    s0 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(S1 + i)));
                    s1 = _mm_cvtepi32_ps(_mm_load_si128((__m128i*)(S1 + i + 4)));
                    s0 = _mm_add_ps(_mm_mul_ps(s0, k0), df4);
                    s1 = _mm_add_ps(_mm_mul_ps(s1, k0), df4);
                    __m128i x0, x1;
                    x0 = _mm_add_epi32(_mm_load_si128((__m128i*)(S0 + i)),
                                       _mm_load_si128((__m128i*)(S2 + i)));
                    x1 = _mm_add_epi32(_mm_load_si128((__m128i*)(S0 + i + 4)),
                                       _mm_load_si128((__m128i*)(S2 + i + 4)));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0),k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1),k1));
                    x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                    _mm_storeu_si128((__m128i*)(dst + i), x0);
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128i s0, s1, s2, s3;
                    s0 = _mm_load_si128((__m128i*)(S2 + i));
                    s1 = _mm_load_si128((__m128i*)(S2 + i + 4));
                    s2 = _mm_load_si128((__m128i*)(S0 + i));
                    s3 = _mm_load_si128((__m128i*)(S0 + i + 4));
                    s0 = _mm_add_epi32(_mm_sub_epi32(s0, s2), d4);
                    s1 = _mm_add_epi32(_mm_sub_epi32(s1, s3), d4);
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0, s1));
                }
            }
            else
            {
                __m128 k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0 = df4, s1 = df4;
                    __m128i x0, x1;
                    x0 = _mm_sub_epi32(_mm_load_si128((__m128i*)(S0 + i)),
                                       _mm_load_si128((__m128i*)(S2 + i)));
                    x1 = _mm_sub_epi32(_mm_load_si128((__m128i*)(S0 + i + 4)),
                                       _mm_load_si128((__m128i*)(S2 + i + 4)));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(_mm_cvtepi32_ps(x0),k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(_mm_cvtepi32_ps(x1),k1));
                    x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
                    _mm_storeu_si128((__m128i*)(dst + i), x0);
                }
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};

    
/////////////////////////////////////// 16s //////////////////////////////////
    
struct RowVec_16s32f
{
    RowVec_16s32f() {}
    RowVec_16s32f( const Mat& _kernel )
    {
        kernel = _kernel;
        sse2_supported = checkHardwareSupport(CV_CPU_SSE2);
    }
    
    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        if( !sse2_supported )
            return 0;
        
        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* _kx = (const float*)kernel.data;
        width *= cn;
        
        for( ; i <= width - 8; i += 8 )
        {
            const short* src = (const short*)_src + i;
            __m128 f, s0 = _mm_setzero_ps(), s1 = s0, x0, x1;
            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = _mm_load_ss(_kx+k);
                f = _mm_shuffle_ps(f, f, 0);
                
                __m128i x0i = _mm_loadu_si128((const __m128i*)src);
                __m128i x1i = _mm_srai_epi32(_mm_unpackhi_epi16(x0i, x0i), 16);
                x0i = _mm_srai_epi32(_mm_unpacklo_epi16(x0i, x0i), 16);
                x0 = _mm_cvtepi32_ps(x0i);
                x1 = _mm_cvtepi32_ps(x1i);
                s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
            }
            _mm_store_ps(dst + i, s0);
            _mm_store_ps(dst + i + 4, s1);
        }
        return i;
    }
    
    Mat kernel;
    bool sse2_supported;
};
    
    
struct SymmColumnVec_32f16s
{
    SymmColumnVec_32f16s() { symmetryType=0; }
    SymmColumnVec_32f16s(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
        sse2_supported = checkHardwareSupport(CV_CPU_SSE2);
    }
    
    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        if( !sse2_supported )
            return 0;
        
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = (const float*)kernel.data + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S, *S2;
        short* dst = (short*)_dst;
        __m128 d4 = _mm_set1_ps(delta);
        
        if( symmetrical )
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 s0, s1, s2, s3;
                __m128 x0, x1;
                S = src[0] + i;
                s0 = _mm_load_ps(S);
                s1 = _mm_load_ps(S+4);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);
                s1 = _mm_add_ps(_mm_mul_ps(s1, f), d4);
                s2 = _mm_load_ps(S+8);
                s3 = _mm_load_ps(S+12);
                s2 = _mm_add_ps(_mm_mul_ps(s2, f), d4);
                s3 = _mm_add_ps(_mm_mul_ps(s3, f), d4);
                
                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_add_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_add_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_add_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }
                
                __m128i s0i = _mm_cvtps_epi32(s0);
                __m128i s1i = _mm_cvtps_epi32(s1);
                __m128i s2i = _mm_cvtps_epi32(s2);
                __m128i s3i = _mm_cvtps_epi32(s3);
                
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0i, s1i));
                _mm_storeu_si128((__m128i*)(dst + i + 8), _mm_packs_epi32(s2i, s3i));
            }
            
            for( ; i <= width - 4; i += 4 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 x0, s0 = _mm_load_ps(src[0] + i);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);
                
                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    x0 = _mm_add_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }
                
                __m128i s0i = _mm_cvtps_epi32(s0);
                _mm_storel_epi64((__m128i*)(dst + i), _mm_packs_epi32(s0i, s0i));
            }
        }
        else
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f, s0 = d4, s1 = d4, s2 = d4, s3 = d4;
                __m128 x0, x1;
                S = src[0] + i;
                
                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_sub_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_sub_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_sub_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }
                
                __m128i s0i = _mm_cvtps_epi32(s0);
                __m128i s1i = _mm_cvtps_epi32(s1);
                __m128i s2i = _mm_cvtps_epi32(s2);
                __m128i s3i = _mm_cvtps_epi32(s3);
                
                _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(s0i, s1i));
                _mm_storeu_si128((__m128i*)(dst + i + 8), _mm_packs_epi32(s2i, s3i));
            }
            
            for( ; i <= width - 4; i += 4 )
            {
                __m128 f, x0, s0 = d4;
                
                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }
                
                __m128i s0i = _mm_cvtps_epi32(s0);
                _mm_storel_epi64((__m128i*)(dst + i), _mm_packs_epi32(s0i, s0i));
            }
        }
        
        return i;
    }
    
    int symmetryType;
    float delta;
    Mat kernel;
    bool sse2_supported;
};    
    

/////////////////////////////////////// 32f //////////////////////////////////

struct RowVec_32f
{
    RowVec_32f() {}
    RowVec_32f( const Mat& _kernel )
    {
        kernel = _kernel;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;
        
        int i = 0, k, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* _kx = (const float*)kernel.data;
        width *= cn;

        for( ; i <= width - 8; i += 8 )
        {
            const float* src = (const float*)_src + i;
            __m128 f, s0 = _mm_setzero_ps(), s1 = s0, x0, x1;
            for( k = 0; k < _ksize; k++, src += cn )
            {
                f = _mm_load_ss(_kx+k);
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_loadu_ps(src);
                x1 = _mm_loadu_ps(src + 4);
                s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
            }
            _mm_store_ps(dst + i, s0);
            _mm_store_ps(dst + i + 4, s1);
        }
        return i;
    }

    Mat kernel;
};


struct SymmRowSmallVec_32f
{
    SymmRowSmallVec_32f() {}
    SymmRowSmallVec_32f( const Mat& _kernel, int _symmetryType )
    {
        kernel = _kernel;
        symmetryType = _symmetryType;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;
        
        int i = 0, _ksize = kernel.rows + kernel.cols - 1;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize/2)*cn;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float* kx = (const float*)kernel.data + _ksize/2;
        width *= cn;

        if( symmetrical )
        {
            if( _ksize == 1 )
                return 0;
            if( _ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);
                        x0 = _mm_add_ps(x0, _mm_add_ps(_mm_add_ps(x1, x1), x2));
                        y0 = _mm_add_ps(y0, _mm_add_ps(_mm_add_ps(y1, y1), y2));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else if( kx[0] == -2 && kx[1] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);
                        x0 = _mm_add_ps(x0, _mm_sub_ps(x2, _mm_add_ps(x1, x1)));
                        y0 = _mm_add_ps(y0, _mm_sub_ps(y2, _mm_add_ps(y1, y1)));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else
                {
                    __m128 k0 = _mm_set1_ps(kx[0]), k1 = _mm_set1_ps(kx[1]);
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);

                        x0 = _mm_mul_ps(_mm_add_ps(x0, x2), k1);
                        y0 = _mm_mul_ps(_mm_add_ps(y0, y2), k1);
                        x0 = _mm_add_ps(x0, _mm_mul_ps(x1, k0));
                        y0 = _mm_add_ps(y0, _mm_mul_ps(y1, k0));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                if( kx[0] == -2 && kx[1] == 0 && kx[2] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn*2);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn*2);
                        y0 = _mm_loadu_ps(src - cn*2 + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn*2 + 4);
                        x0 = _mm_add_ps(x0, _mm_sub_ps(x2, _mm_add_ps(x1, x1)));
                        y0 = _mm_add_ps(y0, _mm_sub_ps(y2, _mm_add_ps(y1, y1)));
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else
                {
                    __m128 k0 = _mm_set1_ps(kx[0]), k1 = _mm_set1_ps(kx[1]), k2 = _mm_set1_ps(kx[2]);
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x1, x2, y0, y1, y2;
                        x0 = _mm_loadu_ps(src - cn);
                        x1 = _mm_loadu_ps(src);
                        x2 = _mm_loadu_ps(src + cn);
                        y0 = _mm_loadu_ps(src - cn + 4);
                        y1 = _mm_loadu_ps(src + 4);
                        y2 = _mm_loadu_ps(src + cn + 4);

                        x0 = _mm_mul_ps(_mm_add_ps(x0, x2), k1);
                        y0 = _mm_mul_ps(_mm_add_ps(y0, y2), k1);
                        x0 = _mm_add_ps(x0, _mm_mul_ps(x1, k0));
                        y0 = _mm_add_ps(y0, _mm_mul_ps(y1, k0));
                        
                        x2 = _mm_add_ps(_mm_loadu_ps(src + cn*2), _mm_loadu_ps(src - cn*2));
                        y2 = _mm_add_ps(_mm_loadu_ps(src + cn*2 + 4), _mm_loadu_ps(src - cn*2 + 4));
                        x0 = _mm_add_ps(x0, _mm_mul_ps(x2, k2));
                        y0 = _mm_add_ps(y0, _mm_mul_ps(y2, k2));
                        
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                }
            }
        }
        else
        {
            if( _ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x2, y0, y2;
                        x0 = _mm_loadu_ps(src + cn);
                        x2 = _mm_loadu_ps(src - cn);
                        y0 = _mm_loadu_ps(src + cn + 4);
                        y2 = _mm_loadu_ps(src - cn + 4);
                        x0 = _mm_sub_ps(x0, x2);
                        y0 = _mm_sub_ps(y0, y2);
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                else
                {
                    __m128 k1 = _mm_set1_ps(kx[1]);
                    for( ; i <= width - 8; i += 8, src += 8 )
                    {
                        __m128 x0, x2, y0, y2;
                        x0 = _mm_loadu_ps(src + cn);
                        x2 = _mm_loadu_ps(src - cn);
                        y0 = _mm_loadu_ps(src + cn + 4);
                        y2 = _mm_loadu_ps(src - cn + 4);

                        x0 = _mm_mul_ps(_mm_sub_ps(x0, x2), k1);
                        y0 = _mm_mul_ps(_mm_sub_ps(y0, y2), k1);
                        _mm_store_ps(dst + i, x0);
                        _mm_store_ps(dst + i + 4, y0);
                    }
                }
            }
            else if( _ksize == 5 )
            {
                __m128 k1 = _mm_set1_ps(kx[1]), k2 = _mm_set1_ps(kx[2]);
                for( ; i <= width - 8; i += 8, src += 8 )
                {
                    __m128 x0, x2, y0, y2;
                    x0 = _mm_loadu_ps(src + cn);
                    x2 = _mm_loadu_ps(src - cn);
                    y0 = _mm_loadu_ps(src + cn + 4);
                    y2 = _mm_loadu_ps(src - cn + 4);

                    x0 = _mm_mul_ps(_mm_sub_ps(x0, x2), k1);
                    y0 = _mm_mul_ps(_mm_sub_ps(y0, y2), k1);
                    
                    x2 = _mm_sub_ps(_mm_loadu_ps(src + cn*2), _mm_loadu_ps(src - cn*2));
                    y2 = _mm_sub_ps(_mm_loadu_ps(src + cn*2 + 4), _mm_loadu_ps(src - cn*2 + 4));
                    x0 = _mm_add_ps(x0, _mm_mul_ps(x2, k2));
                    y0 = _mm_add_ps(y0, _mm_mul_ps(y2, k2));
                    
                    _mm_store_ps(dst + i, x0);
                    _mm_store_ps(dst + i + 4, y0);
                }
            }
        }

        return i;
    }

    Mat kernel;
    int symmetryType;
};


struct SymmColumnVec_32f
{
    SymmColumnVec_32f() { symmetryType=0; }
    SymmColumnVec_32f(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;
        
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = (const float*)kernel.data + ksize2;
        int i = 0, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S, *S2;
        float* dst = (float*)_dst;
        __m128 d4 = _mm_set1_ps(delta);

        if( symmetrical )
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 s0, s1, s2, s3;
                __m128 x0, x1;
                S = src[0] + i;
                s0 = _mm_load_ps(S);
                s1 = _mm_load_ps(S+4);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);
                s1 = _mm_add_ps(_mm_mul_ps(s1, f), d4);
                s2 = _mm_load_ps(S+8);
                s3 = _mm_load_ps(S+12);
                s2 = _mm_add_ps(_mm_mul_ps(s2, f), d4);
                s3 = _mm_add_ps(_mm_mul_ps(s3, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_add_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_add_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_add_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_add_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }

                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
                _mm_storeu_ps(dst + i + 8, s2);
                _mm_storeu_ps(dst + i + 12, s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f = _mm_load_ss(ky);
                f = _mm_shuffle_ps(f, f, 0);
                __m128 x0, s0 = _mm_load_ps(src[0] + i);
                s0 = _mm_add_ps(_mm_mul_ps(s0, f), d4);

                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    x0 = _mm_add_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }

                _mm_storeu_ps(dst + i, s0);
            }
        }
        else
        {
            for( ; i <= width - 16; i += 16 )
            {
                __m128 f, s0 = d4, s1 = d4, s2 = d4, s3 = d4;
                __m128 x0, x1;
                S = src[0] + i;

                for( k = 1; k <= ksize2; k++ )
                {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_ps(_mm_load_ps(S), _mm_load_ps(S2));
                    x1 = _mm_sub_ps(_mm_load_ps(S+4), _mm_load_ps(S2+4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1, f));
                    x0 = _mm_sub_ps(_mm_load_ps(S+8), _mm_load_ps(S2+8));
                    x1 = _mm_sub_ps(_mm_load_ps(S+12), _mm_load_ps(S2+12));
                    s2 = _mm_add_ps(s2, _mm_mul_ps(x0, f));
                    s3 = _mm_add_ps(s3, _mm_mul_ps(x1, f));
                }

                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
                _mm_storeu_ps(dst + i + 8, s2);
                _mm_storeu_ps(dst + i + 12, s3);
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 f, x0, s0 = d4;

                for( k = 1; k <= ksize2; k++ )
                {
                    f = _mm_load_ss(ky+k);
                    f = _mm_shuffle_ps(f, f, 0);
                    x0 = _mm_sub_ps(_mm_load_ps(src[k]+i), _mm_load_ps(src[-k] + i));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0, f));
                }

                _mm_storeu_ps(dst + i, s0);
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


struct SymmColumnSmallVec_32f
{
    SymmColumnSmallVec_32f() { symmetryType=0; }
    SymmColumnSmallVec_32f(const Mat& _kernel, int _symmetryType, int, double _delta)
    {
        symmetryType = _symmetryType;
        kernel = _kernel;
        delta = (float)_delta;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;
        
        int ksize2 = (kernel.rows + kernel.cols - 1)/2;
        const float* ky = (const float*)kernel.data + ksize2;
        int i = 0;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        const float** src = (const float**)_src;
        const float *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        float* dst = (float*)_dst;
        __m128 d4 = _mm_set1_ps(delta);

        if( symmetrical )
        {
            if( ky[0] == 2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_ps(S0 + i);
                    s1 = _mm_load_ps(S0 + i + 4);
                    s2 = _mm_load_ps(S1 + i);
                    s3 = _mm_load_ps(S1 + i + 4);
                    s4 = _mm_load_ps(S2 + i);
                    s5 = _mm_load_ps(S2 + i + 4);
                    s0 = _mm_add_ps(s0, _mm_add_ps(s4, _mm_add_ps(s2, s2)));
                    s1 = _mm_add_ps(s1, _mm_add_ps(s5, _mm_add_ps(s3, s3)));
                    s0 = _mm_add_ps(s0, d4);
                    s1 = _mm_add_ps(s1, d4);
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
            else if( ky[0] == -2 && ky[1] == 1 )
            {
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, s2, s3, s4, s5;
                    s0 = _mm_load_ps(S0 + i);
                    s1 = _mm_load_ps(S0 + i + 4);
                    s2 = _mm_load_ps(S1 + i);
                    s3 = _mm_load_ps(S1 + i + 4);
                    s4 = _mm_load_ps(S2 + i);
                    s5 = _mm_load_ps(S2 + i + 4);
                    s0 = _mm_add_ps(s0, _mm_sub_ps(s4, _mm_add_ps(s2, s2)));
                    s1 = _mm_add_ps(s1, _mm_sub_ps(s5, _mm_add_ps(s3, s3)));
                    s0 = _mm_add_ps(s0, d4);
                    s1 = _mm_add_ps(s1, d4);
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
            else
            {
                __m128 k0 = _mm_set1_ps(ky[0]), k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, x0, x1;
                    s0 = _mm_load_ps(S1 + i);
                    s1 = _mm_load_ps(S1 + i + 4);
                    s0 = _mm_add_ps(_mm_mul_ps(s0, k0), d4);
                    s1 = _mm_add_ps(_mm_mul_ps(s1, k0), d4);
                    x0 = _mm_add_ps(_mm_load_ps(S0 + i), _mm_load_ps(S2 + i));
                    x1 = _mm_add_ps(_mm_load_ps(S0 + i + 4), _mm_load_ps(S2 + i + 4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0,k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1,k1));
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
        }
        else
        {
            if( fabs(ky[1]) == 1 && ky[1] == -ky[-1] )
            {
                if( ky[1] < 0 )
                    std::swap(S0, S2);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0, s1, s2, s3;
                    s0 = _mm_load_ps(S2 + i);
                    s1 = _mm_load_ps(S2 + i + 4);
                    s2 = _mm_load_ps(S0 + i);
                    s3 = _mm_load_ps(S0 + i + 4);
                    s0 = _mm_add_ps(_mm_sub_ps(s0, s2), d4);
                    s1 = _mm_add_ps(_mm_sub_ps(s1, s3), d4);
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
            else
            {
                __m128 k1 = _mm_set1_ps(ky[1]);
                for( ; i <= width - 8; i += 8 )
                {
                    __m128 s0 = d4, s1 = d4, x0, x1;
                    x0 = _mm_sub_ps(_mm_load_ps(S2 + i), _mm_load_ps(S0 + i));
                    x1 = _mm_sub_ps(_mm_load_ps(S2 + i + 4), _mm_load_ps(S0 + i + 4));
                    s0 = _mm_add_ps(s0, _mm_mul_ps(x0,k1));
                    s1 = _mm_add_ps(s1, _mm_mul_ps(x1,k1));
                    _mm_storeu_ps(dst + i, s0);
                    _mm_storeu_ps(dst + i + 4, s1);
                }
            }
        }

        return i;
    }

    int symmetryType;
    float delta;
    Mat kernel;
};


/////////////////////////////// non-separable filters ///////////////////////////////

///////////////////////////////// 8u<->8u, 8u<->16s /////////////////////////////////

struct FilterVec_8u
{
    FilterVec_8u() {}
    FilterVec_8u(const Mat& _kernel, int _bits, double _delta)
    {
        Mat kernel;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        vector<Point> coords;
        preprocess2DKernel(kernel, coords, coeffs);
        _nz = (int)coords.size();
    }

    int operator()(const uchar** src, uchar* dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;
        
        const float* kf = (const float*)&coeffs[0];
        int i = 0, k, nz = _nz;
        __m128 d4 = _mm_set1_ps(delta);

        for( ; i <= width - 16; i += 16 )
        {
            __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;
            __m128i x0, x1, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0, t1;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_loadu_si128((const __m128i*)(src[k] + i));
                x1 = _mm_unpackhi_epi8(x0, z);
                x0 = _mm_unpacklo_epi8(x0, z);

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x1, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x1, z));
                s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
                s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
            x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
            x0 = _mm_packus_epi16(x0, x1);
            _mm_storeu_si128((__m128i*)(dst + i), x0);
        }

        for( ; i <= width - 4; i += 4 )
        {
            __m128 s0 = d4;
            __m128i x0, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_cvtsi32_si128(*(const int*)(src[k] + i));
                x0 = _mm_unpacklo_epi8(x0, z);
                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), z);
            x0 = _mm_packus_epi16(x0, x0);
            *(int*)(dst + i) = _mm_cvtsi128_si32(x0);
        }

        return i;
    }

    int _nz;
    vector<uchar> coeffs;
    float delta;
};


struct FilterVec_8u16s
{
    FilterVec_8u16s() {}
    FilterVec_8u16s(const Mat& _kernel, int _bits, double _delta)
    {
        Mat kernel;
        _kernel.convertTo(kernel, CV_32F, 1./(1 << _bits), 0);
        delta = (float)(_delta/(1 << _bits));
        vector<Point> coords;
        preprocess2DKernel(kernel, coords, coeffs);
        _nz = (int)coords.size();
    }

    int operator()(const uchar** src, uchar* _dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;
        
        const float* kf = (const float*)&coeffs[0];
        short* dst = (short*)_dst;
        int i = 0, k, nz = _nz;
        __m128 d4 = _mm_set1_ps(delta);

        for( ; i <= width - 16; i += 16 )
        {
            __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;
            __m128i x0, x1, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0, t1;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_loadu_si128((const __m128i*)(src[k] + i));
                x1 = _mm_unpackhi_epi8(x0, z);
                x0 = _mm_unpacklo_epi8(x0, z);

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));

                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x1, z));
                t1 = _mm_cvtepi32_ps(_mm_unpackhi_epi16(x1, z));
                s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
                s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), _mm_cvtps_epi32(s1));
            x1 = _mm_packs_epi32(_mm_cvtps_epi32(s2), _mm_cvtps_epi32(s3));
            _mm_storeu_si128((__m128i*)(dst + i), x0);
            _mm_storeu_si128((__m128i*)(dst + i + 8), x1);
        }

        for( ; i <= width - 4; i += 4 )
        {
            __m128 s0 = d4;
            __m128i x0, z = _mm_setzero_si128();

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0;
                f = _mm_shuffle_ps(f, f, 0);

                x0 = _mm_cvtsi32_si128(*(const int*)(src[k] + i));
                x0 = _mm_unpacklo_epi8(x0, z);
                t0 = _mm_cvtepi32_ps(_mm_unpacklo_epi16(x0, z));
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            }

            x0 = _mm_packs_epi32(_mm_cvtps_epi32(s0), z);
            _mm_storel_epi64((__m128i*)(dst + i), x0);
        }

        return i;
    }

    int _nz;
    vector<uchar> coeffs;
    float delta;
};


struct FilterVec_32f
{
    FilterVec_32f() {}
    FilterVec_32f(const Mat& _kernel, int, double _delta)
    {
        delta = (float)_delta;
        vector<Point> coords;
        preprocess2DKernel(_kernel, coords, coeffs);
        _nz = (int)coords.size();
    }

    int operator()(const uchar** _src, uchar* _dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;
        
        const float* kf = (const float*)&coeffs[0];
        const float** src = (const float**)_src;
        float* dst = (float*)_dst;
        int i = 0, k, nz = _nz;
        __m128 d4 = _mm_set1_ps(delta);

        for( ; i <= width - 16; i += 16 )
        {
            __m128 s0 = d4, s1 = d4, s2 = d4, s3 = d4;

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0, t1;
                f = _mm_shuffle_ps(f, f, 0);
                const float* S = src[k] + i;

                t0 = _mm_loadu_ps(S);
                t1 = _mm_loadu_ps(S + 4);
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
                s1 = _mm_add_ps(s1, _mm_mul_ps(t1, f));

                t0 = _mm_loadu_ps(S + 8);
                t1 = _mm_loadu_ps(S + 12);
                s2 = _mm_add_ps(s2, _mm_mul_ps(t0, f));
                s3 = _mm_add_ps(s3, _mm_mul_ps(t1, f));
            }

            _mm_storeu_ps(dst + i, s0);
            _mm_storeu_ps(dst + i + 4, s1);
            _mm_storeu_ps(dst + i + 8, s2);
            _mm_storeu_ps(dst + i + 12, s3);
        }

        for( ; i <= width - 4; i += 4 )
        {
            __m128 s0 = d4;

            for( k = 0; k < nz; k++ )
            {
                __m128 f = _mm_load_ss(kf+k), t0;
                f = _mm_shuffle_ps(f, f, 0);
                t0 = _mm_loadu_ps(src[k] + i);
                s0 = _mm_add_ps(s0, _mm_mul_ps(t0, f));
            }
            _mm_storeu_ps(dst + i, s0);
        }

        return i;
    }

    int _nz;
    vector<uchar> coeffs;
    float delta;
};


#else

typedef RowNoVec RowVec_8u32s;
typedef RowNoVec RowVec_16s32f;
typedef RowNoVec RowVec_32f;
typedef SymmRowSmallNoVec SymmRowSmallVec_8u32s;
typedef SymmRowSmallNoVec SymmRowSmallVec_32f;
typedef ColumnNoVec SymmColumnVec_32s8u;
typedef ColumnNoVec SymmColumnVec_32f16s;
typedef ColumnNoVec SymmColumnVec_32f;
typedef SymmColumnSmallNoVec SymmColumnSmallVec_32s16s;
typedef SymmColumnSmallNoVec SymmColumnSmallVec_32f;
typedef FilterNoVec FilterVec_8u;
typedef FilterNoVec FilterVec_8u16s;
typedef FilterNoVec FilterVec_32f;

#endif


template<typename ST, typename DT, class VecOp> struct RowFilter : public BaseRowFilter
{
    RowFilter( const Mat& _kernel, int _anchor, const VecOp& _vecOp=VecOp() )
    {
        if( _kernel.isContinuous() )
            kernel = _kernel;
        else
            _kernel.copyTo(kernel);
        anchor = _anchor;
        ksize = kernel.rows + kernel.cols - 1;
        CV_Assert( kernel.type() == DataType<DT>::type &&
                   (kernel.rows == 1 || kernel.cols == 1));
        vecOp = _vecOp;
    }
    
    void operator()(const uchar* src, uchar* dst, int width, int cn)
    {
        int _ksize = ksize;
        const DT* kx = (const DT*)kernel.data;
        const ST* S;
        DT* D = (DT*)dst;
        int i, k;

        i = vecOp(src, dst, width, cn);
        width *= cn;
        #if CV_ENABLE_UNROLLED
        for( ; i <= width - 4; i += 4 )
        {
            S = (const ST*)src + i;
            DT f = kx[0];
            DT s0 = f*S[0], s1 = f*S[1], s2 = f*S[2], s3 = f*S[3];

            for( k = 1; k < _ksize; k++ )
            {
                S += cn;
                f = kx[k];
                s0 += f*S[0]; s1 += f*S[1];
                s2 += f*S[2]; s3 += f*S[3];
            }
            
            D[i] = s0; D[i+1] = s1;
            D[i+2] = s2; D[i+3] = s3;
        }
        #endif
        for( ; i < width; i++ )
        {
            S = (const ST*)src + i;
            DT s0 = kx[0]*S[0];
            for( k = 1; k < _ksize; k++ )
            {
                S += cn;
                s0 += kx[k]*S[0];
            }
            D[i] = s0;
        }
    }

    Mat kernel;
    VecOp vecOp;
};


template<typename ST, typename DT, class VecOp> struct SymmRowSmallFilter :
    public RowFilter<ST, DT, VecOp>
{
    SymmRowSmallFilter( const Mat& _kernel, int _anchor, int _symmetryType,
                        const VecOp& _vecOp = VecOp())
        : RowFilter<ST, DT, VecOp>( _kernel, _anchor, _vecOp )
    {
        symmetryType = _symmetryType;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 && this->ksize <= 5 );
    }
    
    void operator()(const uchar* src, uchar* dst, int width, int cn)
    {
        int ksize2 = this->ksize/2, ksize2n = ksize2*cn;
        const DT* kx = (const DT*)this->kernel.data + ksize2;
        bool symmetrical = (this->symmetryType & KERNEL_SYMMETRICAL) != 0;
        DT* D = (DT*)dst;
        int i = this->vecOp(src, dst, width, cn), j, k;
        const ST* S = (const ST*)src + i + ksize2n;
        width *= cn;

        if( symmetrical )
        {
            if( this->ksize == 1 && kx[0] == 1 )
            {
                for( ; i <= width - 2; i += 2 )
                {
                    DT s0 = S[i], s1 = S[i+1];
                    D[i] = s0; D[i+1] = s1;
                }
                S += i;
            }
            else if( this->ksize == 3 )
            {
                if( kx[0] == 2 && kx[1] == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[-cn] + S[0]*2 + S[cn], s1 = S[1-cn] + S[1]*2 + S[1+cn];
                        D[i] = s0; D[i+1] = s1;
                    }
                else if( kx[0] == -2 && kx[1] == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[-cn] - S[0]*2 + S[cn], s1 = S[1-cn] - S[1]*2 + S[1+cn];
                        D[i] = s0; D[i+1] = s1;
                    }
                else
                {
                    DT k0 = kx[0], k1 = kx[1];
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[0]*k0 + (S[-cn] + S[cn])*k1, s1 = S[1]*k0 + (S[1-cn] + S[1+cn])*k1;
                        D[i] = s0; D[i+1] = s1;
                    }
                }
            }
            else if( this->ksize == 5 )
            {
                DT k0 = kx[0], k1 = kx[1], k2 = kx[2];
                if( k0 == -2 && k1 == 0 && k2 == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = -2*S[0] + S[-cn*2] + S[cn*2];
                        DT s1 = -2*S[1] + S[1-cn*2] + S[1+cn*2];
                        D[i] = s0; D[i+1] = s1;
                    }
                else
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[0]*k0 + (S[-cn] + S[cn])*k1 + (S[-cn*2] + S[cn*2])*k2;
                        DT s1 = S[1]*k0 + (S[1-cn] + S[1+cn])*k1 + (S[1-cn*2] + S[1+cn*2])*k2;
                        D[i] = s0; D[i+1] = s1;
                    }
            }

            for( ; i < width; i++, S++ )
            {
                DT s0 = kx[0]*S[0];
                for( k = 1, j = cn; k <= ksize2; k++, j += cn )
                    s0 += kx[k]*(S[j] + S[-j]);
                D[i] = s0;
            }
        }
        else
        {
            if( this->ksize == 3 )
            {
                if( kx[0] == 0 && kx[1] == 1 )
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = S[cn] - S[-cn], s1 = S[1+cn] - S[1-cn];
                        D[i] = s0; D[i+1] = s1;
                    }
                else
                {
                    DT k1 = kx[1];
                    for( ; i <= width - 2; i += 2, S += 2 )
                    {
                        DT s0 = (S[cn] - S[-cn])*k1, s1 = (S[1+cn] - S[1-cn])*k1;
                        D[i] = s0; D[i+1] = s1;
                    }
                }
            }
            else if( this->ksize == 5 )
            {
                DT k1 = kx[1], k2 = kx[2];
                for( ; i <= width - 2; i += 2, S += 2 )
                {
                    DT s0 = (S[cn] - S[-cn])*k1 + (S[cn*2] - S[-cn*2])*k2;
                    DT s1 = (S[1+cn] - S[1-cn])*k1 + (S[1+cn*2] - S[1-cn*2])*k2;
                    D[i] = s0; D[i+1] = s1;
                }
            }

            for( ; i < width; i++, S++ )
            {
                DT s0 = kx[0]*S[0];
                for( k = 1, j = cn; k <= ksize2; k++, j += cn )
                    s0 += kx[k]*(S[j] - S[-j]);
                D[i] = s0;
            }
        }
    }

    int symmetryType;
};


template<class CastOp, class VecOp> struct ColumnFilter : public BaseColumnFilter
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;
    
    ColumnFilter( const Mat& _kernel, int _anchor,
        double _delta, const CastOp& _castOp=CastOp(),
        const VecOp& _vecOp=VecOp() )
    {
        if( _kernel.isContinuous() )
            kernel = _kernel;
        else
            _kernel.copyTo(kernel);
        anchor = _anchor;
        ksize = kernel.rows + kernel.cols - 1;
        delta = saturate_cast<ST>(_delta);
        castOp0 = _castOp;
        vecOp = _vecOp;
        CV_Assert( kernel.type() == DataType<ST>::type &&
                   (kernel.rows == 1 || kernel.cols == 1));
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width)
    {
        const ST* ky = (const ST*)kernel.data;
        ST _delta = delta;
        int _ksize = ksize;
        int i, k;
        CastOp castOp = castOp0;

        for( ; count--; dst += dststep, src++ )
        {
            DT* D = (DT*)dst;
            i = vecOp(src, dst, width);
			#if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                ST f = ky[0];
                const ST* S = (const ST*)src[0] + i;
                ST s0 = f*S[0] + _delta, s1 = f*S[1] + _delta,
                    s2 = f*S[2] + _delta, s3 = f*S[3] + _delta;

                for( k = 1; k < _ksize; k++ )
                {
                    S = (const ST*)src[k] + i; f = ky[k];
                    s0 += f*S[0]; s1 += f*S[1];
                    s2 += f*S[2]; s3 += f*S[3];
                }

                D[i] = castOp(s0); D[i+1] = castOp(s1);
                D[i+2] = castOp(s2); D[i+3] = castOp(s3);
            }
            #endif
            for( ; i < width; i++ )
            {
                ST s0 = ky[0]*((const ST*)src[0])[i] + _delta;
                for( k = 1; k < _ksize; k++ )
                    s0 += ky[k]*((const ST*)src[k])[i];
                D[i] = castOp(s0);
            }
        }
    }

    Mat kernel;
    CastOp castOp0;
    VecOp vecOp;
    ST delta;
};


template<class CastOp, class VecOp> struct SymmColumnFilter : public ColumnFilter<CastOp, VecOp>
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    SymmColumnFilter( const Mat& _kernel, int _anchor,
        double _delta, int _symmetryType,
        const CastOp& _castOp=CastOp(),
        const VecOp& _vecOp=VecOp())
        : ColumnFilter<CastOp, VecOp>( _kernel, _anchor, _delta, _castOp, _vecOp )
    {
        symmetryType = _symmetryType;
        CV_Assert( (symmetryType & (KERNEL_SYMMETRICAL | KERNEL_ASYMMETRICAL)) != 0 );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width)
    {
        int ksize2 = this->ksize/2;
        const ST* ky = (const ST*)this->kernel.data + ksize2;
        int i, k;
        bool symmetrical = (symmetryType & KERNEL_SYMMETRICAL) != 0;
        ST _delta = this->delta;
        CastOp castOp = this->castOp0;
        src += ksize2;

        if( symmetrical )
        {
            for( ; count--; dst += dststep, src++ )
            {
                DT* D = (DT*)dst;
                i = (this->vecOp)(src, dst, width);
                #if CV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST f = ky[0];
                    const ST* S = (const ST*)src[0] + i, *S2;
                    ST s0 = f*S[0] + _delta, s1 = f*S[1] + _delta,
                        s2 = f*S[2] + _delta, s3 = f*S[3] + _delta;

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = (const ST*)src[k] + i;
                        S2 = (const ST*)src[-k] + i;
                        f = ky[k];
                        s0 += f*(S[0] + S2[0]);
                        s1 += f*(S[1] + S2[1]);
                        s2 += f*(S[2] + S2[2]);
                        s3 += f*(S[3] + S2[3]);
                    }

                    D[i] = castOp(s0); D[i+1] = castOp(s1);
                    D[i+2] = castOp(s2); D[i+3] = castOp(s3);
                }
                #endif
                for( ; i < width; i++ )
                {
                    ST s0 = ky[0]*((const ST*)src[0])[i] + _delta;
                    for( k = 1; k <= ksize2; k++ )
                        s0 += ky[k]*(((const ST*)src[k])[i] + ((const ST*)src[-k])[i]);
                    D[i] = castOp(s0);
                }
            }
        }
        else
        {
            for( ; count--; dst += dststep, src++ )
            {
                DT* D = (DT*)dst;
                i = this->vecOp(src, dst, width);
                #if CV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST f = ky[0];
                    const ST *S, *S2;
                    ST s0 = _delta, s1 = _delta, s2 = _delta, s3 = _delta;

                    for( k = 1; k <= ksize2; k++ )
                    {
                        S = (const ST*)src[k] + i;
                        S2 = (const ST*)src[-k] + i;
                        f = ky[k];
                        s0 += f*(S[0] - S2[0]);
                        s1 += f*(S[1] - S2[1]);
                        s2 += f*(S[2] - S2[2]);
                        s3 += f*(S[3] - S2[3]);
                    }

                    D[i] = castOp(s0); D[i+1] = castOp(s1);
                    D[i+2] = castOp(s2); D[i+3] = castOp(s3);
                }
                #endif
                for( ; i < width; i++ )
                {
                    ST s0 = _delta;
                    for( k = 1; k <= ksize2; k++ )
                        s0 += ky[k]*(((const ST*)src[k])[i] - ((const ST*)src[-k])[i]);
                    D[i] = castOp(s0);
                }
            }
        }
    }

    int symmetryType;
};


template<class CastOp, class VecOp>
struct SymmColumnSmallFilter : public SymmColumnFilter<CastOp, VecOp>
{
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;
    
    SymmColumnSmallFilter( const Mat& _kernel, int _anchor,
                           double _delta, int _symmetryType,
                           const CastOp& _castOp=CastOp(),
                           const VecOp& _vecOp=VecOp())
        : SymmColumnFilter<CastOp, VecOp>( _kernel, _anchor, _delta, _symmetryType, _castOp, _vecOp )
    {
        CV_Assert( this->ksize == 3 );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width)
    {
        int ksize2 = this->ksize/2;
        const ST* ky = (const ST*)this->kernel.data + ksize2;
        int i;
        bool symmetrical = (this->symmetryType & KERNEL_SYMMETRICAL) != 0;
        bool is_1_2_1 = ky[0] == 1 && ky[1] == 2;
        bool is_1_m2_1 = ky[0] == 1 && ky[1] == -2;
        bool is_m1_0_1 = ky[1] == 1 || ky[1] == -1;
        ST f0 = ky[0], f1 = ky[1];
        ST _delta = this->delta;
        CastOp castOp = this->castOp0;
        src += ksize2;

        for( ; count--; dst += dststep, src++ )
        {
            DT* D = (DT*)dst;
            i = (this->vecOp)(src, dst, width);
            const ST* S0 = (const ST*)src[-1];
            const ST* S1 = (const ST*)src[0];
            const ST* S2 = (const ST*)src[1];

            if( symmetrical )
            {
                if( is_1_2_1 )
                {
					#if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = S0[i] + S1[i]*2 + S2[i] + _delta;
                        ST s1 = S0[i+1] + S1[i+1]*2 + S2[i+1] + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = S0[i+2] + S1[i+2]*2 + S2[i+2] + _delta;
                        s1 = S0[i+3] + S1[i+3]*2 + S2[i+3] + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #else
		            for( ; i < width; i ++ )
                    {
                        ST s0 = S0[i] + S1[i]*2 + S2[i] + _delta;
                        D[i] = castOp(s0);
                    }
                    #endif
                }
                else if( is_1_m2_1 )
                {
					#if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = S0[i] - S1[i]*2 + S2[i] + _delta;
                        ST s1 = S0[i+1] - S1[i+1]*2 + S2[i+1] + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = S0[i+2] - S1[i+2]*2 + S2[i+2] + _delta;
                        s1 = S0[i+3] - S1[i+3]*2 + S2[i+3] + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #else
		            for( ; i < width; i ++ )
                    {
                        ST s0 = S0[i] - S1[i]*2 + S2[i] + _delta;
                        D[i] = castOp(s0);
                    }
                    #endif
                }
                else
                {
                   #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = (S0[i] + S2[i])*f1 + S1[i]*f0 + _delta;
                        ST s1 = (S0[i+1] + S2[i+1])*f1 + S1[i+1]*f0 + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = (S0[i+2] + S2[i+2])*f1 + S1[i+2]*f0 + _delta;
                        s1 = (S0[i+3] + S2[i+3])*f1 + S1[i+3]*f0 + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #else
                    for( ; i < width; i ++ )
                    {
                        ST s0 = (S0[i] + S2[i])*f1 + S1[i]*f0 + _delta;
                        D[i] = castOp(s0);
                    }
                    #endif
                }
                for( ; i < width; i++ )
                    D[i] = castOp((S0[i] + S2[i])*f1 + S1[i]*f0 + _delta);
            }
            else
            {
                if( is_m1_0_1 )
                {
                    if( f1 < 0 )
                        std::swap(S0, S2);
                   #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = S2[i] - S0[i] + _delta;
                        ST s1 = S2[i+1] - S0[i+1] + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = S2[i+2] - S0[i+2] + _delta;
                        s1 = S2[i+3] - S0[i+3] + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #else
	                for( ; i < width; i ++ )
                    {
                        ST s0 = S2[i] - S0[i] + _delta;
                        D[i] = castOp(s0);
                    }
                    #endif
                    if( f1 < 0 )
                        std::swap(S0, S2);
                }
                else
                {
                   #if CV_ENABLE_UNROLLED
                    for( ; i <= width - 4; i += 4 )
                    {
                        ST s0 = (S2[i] - S0[i])*f1 + _delta;
                        ST s1 = (S2[i+1] - S0[i+1])*f1 + _delta;
                        D[i] = castOp(s0);
                        D[i+1] = castOp(s1);

                        s0 = (S2[i+2] - S0[i+2])*f1 + _delta;
                        s1 = (S2[i+3] - S0[i+3])*f1 + _delta;
                        D[i+2] = castOp(s0);
                        D[i+3] = castOp(s1);
                    }
                    #endif
                }

                for( ; i < width; i++ )
                    D[i] = castOp((S2[i] - S0[i])*f1 + _delta);
            }
        }
    }
};

template<typename ST, typename DT> struct Cast
{
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template<typename ST, typename DT, int bits> struct FixedPtCast
{
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits-1) };

    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
};

template<typename ST, typename DT> struct FixedPtCastEx
{
    typedef ST type1;
    typedef DT rtype;

    FixedPtCastEx() : SHIFT(0), DELTA(0) {}
    FixedPtCastEx(int bits) : SHIFT(bits), DELTA(bits ? 1 << (bits-1) : 0) {}
    DT operator()(ST val) const { return saturate_cast<DT>((val + DELTA)>>SHIFT); }
    int SHIFT, DELTA;
};

}
    
cv::Ptr<cv::BaseRowFilter> cv::getLinearRowFilter( int srcType, int bufType,
                                                   InputArray _kernel, int anchor,
                                                   int symmetryType )
{
    Mat kernel = _kernel.getMat();
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(bufType);
    int cn = CV_MAT_CN(srcType);
    CV_Assert( cn == CV_MAT_CN(bufType) &&
        ddepth >= std::max(sdepth, CV_32S) &&
        kernel.type() == ddepth );
    int ksize = kernel.rows + kernel.cols - 1;

    if( (symmetryType & (KERNEL_SYMMETRICAL|KERNEL_ASYMMETRICAL)) != 0 && ksize <= 5 )
    {
        if( sdepth == CV_8U && ddepth == CV_32S )
            return Ptr<BaseRowFilter>(new SymmRowSmallFilter<uchar, int, SymmRowSmallVec_8u32s>
                (kernel, anchor, symmetryType, SymmRowSmallVec_8u32s(kernel, symmetryType)));
        if( sdepth == CV_32F && ddepth == CV_32F )
            return Ptr<BaseRowFilter>(new SymmRowSmallFilter<float, float, SymmRowSmallVec_32f>
                (kernel, anchor, symmetryType, SymmRowSmallVec_32f(kernel, symmetryType)));
    }
        
    if( sdepth == CV_8U && ddepth == CV_32S )
        return Ptr<BaseRowFilter>(new RowFilter<uchar, int, RowVec_8u32s>
            (kernel, anchor, RowVec_8u32s(kernel)));
    if( sdepth == CV_8U && ddepth == CV_32F )
        return Ptr<BaseRowFilter>(new RowFilter<uchar, float, RowNoVec>(kernel, anchor));
    if( sdepth == CV_8U && ddepth == CV_64F )
        return Ptr<BaseRowFilter>(new RowFilter<uchar, double, RowNoVec>(kernel, anchor));
    if( sdepth == CV_16U && ddepth == CV_32F )
        return Ptr<BaseRowFilter>(new RowFilter<ushort, float, RowNoVec>(kernel, anchor));
    if( sdepth == CV_16U && ddepth == CV_64F )
        return Ptr<BaseRowFilter>(new RowFilter<ushort, double, RowNoVec>(kernel, anchor));
    if( sdepth == CV_16S && ddepth == CV_32F )
        return Ptr<BaseRowFilter>(new RowFilter<short, float, RowVec_16s32f>
                                  (kernel, anchor, RowVec_16s32f(kernel)));
    if( sdepth == CV_16S && ddepth == CV_64F )
        return Ptr<BaseRowFilter>(new RowFilter<short, double, RowNoVec>(kernel, anchor));
    if( sdepth == CV_32F && ddepth == CV_32F )
        return Ptr<BaseRowFilter>(new RowFilter<float, float, RowVec_32f>
            (kernel, anchor, RowVec_32f(kernel)));
    if( sdepth == CV_64F && ddepth == CV_64F )
        return Ptr<BaseRowFilter>(new RowFilter<double, double, RowNoVec>(kernel, anchor));

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of source format (=%d), and buffer format (=%d)",
        srcType, bufType));

    return Ptr<BaseRowFilter>(0);
}


cv::Ptr<cv::BaseColumnFilter> cv::getLinearColumnFilter( int bufType, int dstType,
                                             InputArray _kernel, int anchor,
                                             int symmetryType, double delta, 
                                             int bits )
{
    Mat kernel = _kernel.getMat();
    int sdepth = CV_MAT_DEPTH(bufType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(dstType);
    CV_Assert( cn == CV_MAT_CN(bufType) &&
        sdepth >= std::max(ddepth, CV_32S) &&
        kernel.type() == sdepth );

    if( !(symmetryType & (KERNEL_SYMMETRICAL|KERNEL_ASYMMETRICAL)) )
    {
        if( ddepth == CV_8U && sdepth == CV_32S )
            return Ptr<BaseColumnFilter>(new ColumnFilter<FixedPtCastEx<int, uchar>, ColumnNoVec>
            (kernel, anchor, delta, FixedPtCastEx<int, uchar>(bits)));
        if( ddepth == CV_8U && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<float, uchar>, ColumnNoVec>(kernel, anchor, delta));
        if( ddepth == CV_8U && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<double, uchar>, ColumnNoVec>(kernel, anchor, delta));
        if( ddepth == CV_16U && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<float, ushort>, ColumnNoVec>(kernel, anchor, delta));
        if( ddepth == CV_16U && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<double, ushort>, ColumnNoVec>(kernel, anchor, delta));
        if( ddepth == CV_16S && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<float, short>, ColumnNoVec>(kernel, anchor, delta));
        if( ddepth == CV_16S && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<double, short>, ColumnNoVec>(kernel, anchor, delta));
        if( ddepth == CV_32F && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<float, float>, ColumnNoVec>(kernel, anchor, delta));
        if( ddepth == CV_64F && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new ColumnFilter<Cast<double, double>, ColumnNoVec>(kernel, anchor, delta));
    }
    else
    {
        int ksize = kernel.rows + kernel.cols - 1;
        if( ksize == 3 )
        {
            if( ddepth == CV_8U && sdepth == CV_32S )
                return Ptr<BaseColumnFilter>(new SymmColumnSmallFilter<
                    FixedPtCastEx<int, uchar>, SymmColumnVec_32s8u>
                    (kernel, anchor, delta, symmetryType, FixedPtCastEx<int, uchar>(bits),
                    SymmColumnVec_32s8u(kernel, symmetryType, bits, delta)));
            if( ddepth == CV_16S && sdepth == CV_32S && bits == 0 )
                return Ptr<BaseColumnFilter>(new SymmColumnSmallFilter<Cast<int, short>,
                    SymmColumnSmallVec_32s16s>(kernel, anchor, delta, symmetryType,
                        Cast<int, short>(), SymmColumnSmallVec_32s16s(kernel, symmetryType, bits, delta)));
            if( ddepth == CV_32F && sdepth == CV_32F )
                return Ptr<BaseColumnFilter>(new SymmColumnSmallFilter<
                    Cast<float, float>,SymmColumnSmallVec_32f>
                    (kernel, anchor, delta, symmetryType, Cast<float, float>(),
                    SymmColumnSmallVec_32f(kernel, symmetryType, 0, delta)));
        }
        if( ddepth == CV_8U && sdepth == CV_32S )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<FixedPtCastEx<int, uchar>, SymmColumnVec_32s8u>
                (kernel, anchor, delta, symmetryType, FixedPtCastEx<int, uchar>(bits),
                SymmColumnVec_32s8u(kernel, symmetryType, bits, delta)));
        if( ddepth == CV_8U && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<float, uchar>, ColumnNoVec>
                (kernel, anchor, delta, symmetryType));
        if( ddepth == CV_8U && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<double, uchar>, ColumnNoVec>
                (kernel, anchor, delta, symmetryType));
        if( ddepth == CV_16U && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<float, ushort>, ColumnNoVec>
                (kernel, anchor, delta, symmetryType));
        if( ddepth == CV_16U && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<double, ushort>, ColumnNoVec>
                (kernel, anchor, delta, symmetryType));
        if( ddepth == CV_16S && sdepth == CV_32S )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<int, short>, ColumnNoVec>
                (kernel, anchor, delta, symmetryType));
        if( ddepth == CV_16S && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<float, short>, SymmColumnVec_32f16s>
                 (kernel, anchor, delta, symmetryType, Cast<float, short>(),
                  SymmColumnVec_32f16s(kernel, symmetryType, 0, delta)));
        if( ddepth == CV_16S && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<double, short>, ColumnNoVec>
                (kernel, anchor, delta, symmetryType));
        if( ddepth == CV_32F && sdepth == CV_32F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<float, float>, SymmColumnVec_32f>
                (kernel, anchor, delta, symmetryType, Cast<float, float>(),
                SymmColumnVec_32f(kernel, symmetryType, 0, delta)));
        if( ddepth == CV_64F && sdepth == CV_64F )
            return Ptr<BaseColumnFilter>(new SymmColumnFilter<Cast<double, double>, ColumnNoVec>
                (kernel, anchor, delta, symmetryType));
    }

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of buffer format (=%d), and destination format (=%d)",
        bufType, dstType));

    return Ptr<BaseColumnFilter>(0);
}


cv::Ptr<cv::FilterEngine> cv::createSeparableLinearFilter(
    int _srcType, int _dstType,
    InputArray __rowKernel, InputArray __columnKernel,
    Point _anchor, double _delta,
    int _rowBorderType, int _columnBorderType,
    const Scalar& _borderValue )
{
    Mat _rowKernel = __rowKernel.getMat(), _columnKernel = __columnKernel.getMat();
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);
    int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
    int cn = CV_MAT_CN(_srcType);
    CV_Assert( cn == CV_MAT_CN(_dstType) );
    int rsize = _rowKernel.rows + _rowKernel.cols - 1;
    int csize = _columnKernel.rows + _columnKernel.cols - 1;
    if( _anchor.x < 0 )
        _anchor.x = rsize/2;
    if( _anchor.y < 0 )
        _anchor.y = csize/2;
    int rtype = getKernelType(_rowKernel,
        _rowKernel.rows == 1 ? Point(_anchor.x, 0) : Point(0, _anchor.x));
    int ctype = getKernelType(_columnKernel,
        _columnKernel.rows == 1 ? Point(_anchor.y, 0) : Point(0, _anchor.y));
    Mat rowKernel, columnKernel;

    int bdepth = std::max(CV_32F,std::max(sdepth, ddepth));
    int bits = 0;

    if( sdepth == CV_8U &&
        ((rtype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
          ctype == KERNEL_SMOOTH+KERNEL_SYMMETRICAL &&
          ddepth == CV_8U) ||
         ((rtype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
          (ctype & (KERNEL_SYMMETRICAL+KERNEL_ASYMMETRICAL)) &&
          (rtype & ctype & KERNEL_INTEGER) &&
          ddepth == CV_16S)) )
    {
        bdepth = CV_32S;
        bits = ddepth == CV_8U ? 8 : 0;
        _rowKernel.convertTo( rowKernel, CV_32S, 1 << bits );
        _columnKernel.convertTo( columnKernel, CV_32S, 1 << bits );
        bits *= 2;
        _delta *= (1 << bits);
    }
    else
    {
        if( _rowKernel.type() != bdepth )
            _rowKernel.convertTo( rowKernel, bdepth );
        else
            rowKernel = _rowKernel;
        if( _columnKernel.type() != bdepth )
            _columnKernel.convertTo( columnKernel, bdepth );
        else
            columnKernel = _columnKernel;
    }

    int _bufType = CV_MAKETYPE(bdepth, cn);
    Ptr<BaseRowFilter> _rowFilter = getLinearRowFilter(
        _srcType, _bufType, rowKernel, _anchor.x, rtype);
    Ptr<BaseColumnFilter> _columnFilter = getLinearColumnFilter(
        _bufType, _dstType, columnKernel, _anchor.y, ctype, _delta, bits );

    return Ptr<FilterEngine>( new FilterEngine(Ptr<BaseFilter>(0), _rowFilter, _columnFilter,
        _srcType, _dstType, _bufType, _rowBorderType, _columnBorderType, _borderValue ));
}


/****************************************************************************************\
*                               Non-separable linear filter                              *
\****************************************************************************************/

namespace cv
{

void preprocess2DKernel( const Mat& kernel, vector<Point>& coords, vector<uchar>& coeffs )
{
    int i, j, k, nz = countNonZero(kernel), ktype = kernel.type();
    if(nz == 0)
        nz = 1;
    CV_Assert( ktype == CV_8U || ktype == CV_32S || ktype == CV_32F || ktype == CV_64F );
    coords.resize(nz);
    coeffs.resize(nz*getElemSize(ktype));
    uchar* _coeffs = &coeffs[0];

    for( i = k = 0; i < kernel.rows; i++ )
    {
        const uchar* krow = kernel.data + kernel.step*i;
        for( j = 0; j < kernel.cols; j++ )
        {
            if( ktype == CV_8U )
            {
                uchar val = krow[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                _coeffs[k++] = val;
            }
            else if( ktype == CV_32S )
            {
                int val = ((const int*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((int*)_coeffs)[k++] = val;
            }
            else if( ktype == CV_32F )
            {
                float val = ((const float*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((float*)_coeffs)[k++] = val;
            }
            else
            {
                double val = ((const double*)krow)[j];
                if( val == 0 )
                    continue;
                coords[k] = Point(j,i);
                ((double*)_coeffs)[k++] = val;
            }
        }
    }
}


template<typename ST, class CastOp, class VecOp> struct Filter2D : public BaseFilter
{
    typedef typename CastOp::type1 KT;
    typedef typename CastOp::rtype DT;
    
    Filter2D( const Mat& _kernel, Point _anchor,
        double _delta, const CastOp& _castOp=CastOp(),
        const VecOp& _vecOp=VecOp() )
    {
        anchor = _anchor;
        ksize = _kernel.size();
        delta = saturate_cast<KT>(_delta);
        castOp0 = _castOp;
        vecOp = _vecOp;
        CV_Assert( _kernel.type() == DataType<KT>::type );
        preprocess2DKernel( _kernel, coords, coeffs );
        ptrs.resize( coords.size() );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int cn)
    {
        KT _delta = delta;
        const Point* pt = &coords[0];
        const KT* kf = (const KT*)&coeffs[0];
        const ST** kp = (const ST**)&ptrs[0];
        int i, k, nz = (int)coords.size();
        CastOp castOp = castOp0;

        width *= cn;
        for( ; count > 0; count--, dst += dststep, src++ )
        {
            DT* D = (DT*)dst;

            for( k = 0; k < nz; k++ )
                kp[k] = (const ST*)src[pt[k].y] + pt[k].x*cn;

            i = vecOp((const uchar**)kp, dst, width);
            #if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                KT s0 = _delta, s1 = _delta, s2 = _delta, s3 = _delta;

                for( k = 0; k < nz; k++ )
                {
                    const ST* sptr = kp[k] + i;
                    KT f = kf[k];
                    s0 += f*sptr[0];
                    s1 += f*sptr[1];
                    s2 += f*sptr[2];
                    s3 += f*sptr[3];
                }

                D[i] = castOp(s0); D[i+1] = castOp(s1);
                D[i+2] = castOp(s2); D[i+3] = castOp(s3);
            }
            #endif
            for( ; i < width; i++ )
            {
                KT s0 = _delta;
                for( k = 0; k < nz; k++ )
                    s0 += kf[k]*kp[k][i];
                D[i] = castOp(s0);
            }
        }
    }

    vector<Point> coords;
    vector<uchar> coeffs;
    vector<uchar*> ptrs;
    KT delta;
    CastOp castOp0;
    VecOp vecOp;
};

}

cv::Ptr<cv::BaseFilter> cv::getLinearFilter(int srcType, int dstType,
                                InputArray filter_kernel, Point anchor,
                                double delta, int bits)
{
    Mat _kernel = filter_kernel.getMat();
    int sdepth = CV_MAT_DEPTH(srcType), ddepth = CV_MAT_DEPTH(dstType);
    int cn = CV_MAT_CN(srcType), kdepth = _kernel.depth();
    CV_Assert( cn == CV_MAT_CN(dstType) && ddepth >= sdepth );

    anchor = normalizeAnchor(anchor, _kernel.size());

    /*if( sdepth == CV_8U && ddepth == CV_8U && kdepth == CV_32S )
        return Ptr<BaseFilter>(new Filter2D<uchar, FixedPtCastEx<int, uchar>, FilterVec_8u>
            (_kernel, anchor, delta, FixedPtCastEx<int, uchar>(bits),
            FilterVec_8u(_kernel, bits, delta)));
    if( sdepth == CV_8U && ddepth == CV_16S && kdepth == CV_32S )
        return Ptr<BaseFilter>(new Filter2D<uchar, FixedPtCastEx<int, short>, FilterVec_8u16s>
            (_kernel, anchor, delta, FixedPtCastEx<int, short>(bits),
            FilterVec_8u16s(_kernel, bits, delta)));*/

    kdepth = sdepth == CV_64F || ddepth == CV_64F ? CV_64F : CV_32F;
    Mat kernel;
    if( _kernel.type() == kdepth )
        kernel = _kernel;
    else
        _kernel.convertTo(kernel, kdepth, _kernel.type() == CV_32S ? 1./(1 << bits) : 1.);
    
    if( sdepth == CV_8U && ddepth == CV_8U )
        return Ptr<BaseFilter>(new Filter2D<uchar, Cast<float, uchar>, FilterVec_8u>
            (kernel, anchor, delta, Cast<float, uchar>(), FilterVec_8u(kernel, 0, delta)));
    if( sdepth == CV_8U && ddepth == CV_16U )
        return Ptr<BaseFilter>(new Filter2D<uchar,
            Cast<float, ushort>, FilterNoVec>(kernel, anchor, delta));
    if( sdepth == CV_8U && ddepth == CV_16S )
        return Ptr<BaseFilter>(new Filter2D<uchar, Cast<float, short>, FilterVec_8u16s>
            (kernel, anchor, delta, Cast<float, short>(), FilterVec_8u16s(kernel, 0, delta)));
    if( sdepth == CV_8U && ddepth == CV_32F )
        return Ptr<BaseFilter>(new Filter2D<uchar,
            Cast<float, float>, FilterNoVec>(kernel, anchor, delta));
    if( sdepth == CV_8U && ddepth == CV_64F )
        return Ptr<BaseFilter>(new Filter2D<uchar,
            Cast<double, double>, FilterNoVec>(kernel, anchor, delta));

    if( sdepth == CV_16U && ddepth == CV_16U )
        return Ptr<BaseFilter>(new Filter2D<ushort,
            Cast<float, ushort>, FilterNoVec>(kernel, anchor, delta));
    if( sdepth == CV_16U && ddepth == CV_32F )
        return Ptr<BaseFilter>(new Filter2D<ushort,
            Cast<float, float>, FilterNoVec>(kernel, anchor, delta));
    if( sdepth == CV_16U && ddepth == CV_64F )
        return Ptr<BaseFilter>(new Filter2D<ushort,
            Cast<double, double>, FilterNoVec>(kernel, anchor, delta));

    if( sdepth == CV_16S && ddepth == CV_16S )
        return Ptr<BaseFilter>(new Filter2D<short,
            Cast<float, short>, FilterNoVec>(kernel, anchor, delta));
    if( sdepth == CV_16S && ddepth == CV_32F )
        return Ptr<BaseFilter>(new Filter2D<short,
            Cast<float, float>, FilterNoVec>(kernel, anchor, delta));
    if( sdepth == CV_16S && ddepth == CV_64F )
        return Ptr<BaseFilter>(new Filter2D<short,
            Cast<double, double>, FilterNoVec>(kernel, anchor, delta));

    if( sdepth == CV_32F && ddepth == CV_32F )
        return Ptr<BaseFilter>(new Filter2D<float, Cast<float, float>, FilterVec_32f>
            (kernel, anchor, delta, Cast<float, float>(), FilterVec_32f(kernel, 0, delta)));
    if( sdepth == CV_64F && ddepth == CV_64F )
        return Ptr<BaseFilter>(new Filter2D<double,
            Cast<double, double>, FilterNoVec>(kernel, anchor, delta));

    CV_Error_( CV_StsNotImplemented,
        ("Unsupported combination of source format (=%d), and destination format (=%d)",
        srcType, dstType));

    return Ptr<BaseFilter>(0);
}


cv::Ptr<cv::FilterEngine> cv::createLinearFilter( int _srcType, int _dstType,
                                              InputArray filter_kernel,
                                              Point _anchor, double _delta,
                                              int _rowBorderType, int _columnBorderType,
                                              const Scalar& _borderValue )
{
    Mat _kernel = filter_kernel.getMat();
    _srcType = CV_MAT_TYPE(_srcType);
    _dstType = CV_MAT_TYPE(_dstType);    
    int cn = CV_MAT_CN(_srcType);
    CV_Assert( cn == CV_MAT_CN(_dstType) );

    Mat kernel = _kernel;
    int bits = 0;

    /*int sdepth = CV_MAT_DEPTH(_srcType), ddepth = CV_MAT_DEPTH(_dstType);
    int ktype = _kernel.depth() == CV_32S ? KERNEL_INTEGER : getKernelType(_kernel, _anchor); 
    if( sdepth == CV_8U && (ddepth == CV_8U || ddepth == CV_16S) &&
        _kernel.rows*_kernel.cols <= (1 << 10) )
    {
        bits = (ktype & KERNEL_INTEGER) ? 0 : 11;
        _kernel.convertTo(kernel, CV_32S, 1 << bits);
    }*/
    
    Ptr<BaseFilter> _filter2D = getLinearFilter(_srcType, _dstType,
        kernel, _anchor, _delta, bits);

    return Ptr<FilterEngine>(new FilterEngine(_filter2D, Ptr<BaseRowFilter>(0),
        Ptr<BaseColumnFilter>(0), _srcType, _dstType, _srcType,
        _rowBorderType, _columnBorderType, _borderValue ));
}


void cv::filter2D( InputArray _src, OutputArray _dst, int ddepth,
                   InputArray _kernel, Point anchor,
                   double delta, int borderType )
{
    Mat src = _src.getMat(), kernel = _kernel.getMat();
    
    if( ddepth < 0 )
        ddepth = src.depth();

#if CV_SSE2
    int dft_filter_size = ((src.depth() == CV_8U && (ddepth == CV_8U || ddepth == CV_16S)) ||
        (src.depth() == CV_32F && ddepth == CV_32F)) && checkHardwareSupport(CV_CPU_SSE3)? 130 : 50;
#else
    int dft_filter_size = 50;
#endif

    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();
    anchor = normalizeAnchor(anchor, kernel.size());

#ifdef HAVE_TEGRA_OPTIMIZATION
    if( tegra::filter2D(src, dst, kernel, anchor, delta, borderType) )
        return;
#endif

    if( kernel.cols*kernel.rows >= dft_filter_size )
    {
        Mat temp;
        if( src.data != dst.data )
            temp = dst;
        else
            temp.create(dst.size(), dst.type());
        crossCorr( src, kernel, temp, src.size(),
                   CV_MAKETYPE(ddepth, src.channels()),
                   anchor, delta, borderType );
        if( temp.data != dst.data )
            temp.copyTo(dst);
        return;
    }

    Ptr<FilterEngine> f = createLinearFilter(src.type(), dst.type(), kernel,
                                             anchor, delta, borderType & ~BORDER_ISOLATED );
    f->apply(src, dst, Rect(0,0,-1,-1), Point(), (borderType & BORDER_ISOLATED) != 0 );
}


void cv::sepFilter2D( InputArray _src, OutputArray _dst, int ddepth,
                      InputArray _kernelX, InputArray _kernelY, Point anchor,
                      double delta, int borderType )
{
    Mat src = _src.getMat(), kernelX = _kernelX.getMat(), kernelY = _kernelY.getMat();
    
    if( ddepth < 0 )
        ddepth = src.depth();

    _dst.create( src.size(), CV_MAKETYPE(ddepth, src.channels()) );
    Mat dst = _dst.getMat();

    Ptr<FilterEngine> f = createSeparableLinearFilter(src.type(),
        dst.type(), kernelX, kernelY, anchor, delta, borderType & ~BORDER_ISOLATED );
    f->apply(src, dst, Rect(0,0,-1,-1), Point(), (borderType & BORDER_ISOLATED) != 0 );
}


CV_IMPL void
cvFilter2D( const CvArr* srcarr, CvArr* dstarr, const CvMat* _kernel, CvPoint anchor )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    cv::Mat kernel = cv::cvarrToMat(_kernel);

    CV_Assert( src.size() == dst.size() && src.channels() == dst.channels() );

    cv::filter2D( src, dst, dst.depth(), kernel, anchor, 0, cv::BORDER_REPLICATE );
}

/* End of file. */
