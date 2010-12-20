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

#ifndef __OPENCV_GPU_MATRIX_OPERATIONS_HPP__
#define __OPENCV_GPU_MATRIX_OPERATIONS_HPP__

namespace cv
{

namespace gpu
{

////////////////////////////////////////////////////////////////////////
//////////////////////////////// GpuMat ////////////////////////////////
////////////////////////////////////////////////////////////////////////

inline GpuMat::GpuMat() : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0) {}

inline GpuMat::GpuMat(int _rows, int _cols, int _type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if( _rows > 0 && _cols > 0 )
        create( _rows, _cols, _type );
}

inline GpuMat::GpuMat(Size _size, int _type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if( _size.height > 0 && _size.width > 0 )
        create( _size.height, _size.width, _type );
}

inline GpuMat::GpuMat(int _rows, int _cols, int _type, const Scalar& _s)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if(_rows > 0 && _cols > 0)
    {
        create(_rows, _cols, _type);
        *this = _s;
    }
}

inline GpuMat::GpuMat(Size _size, int _type, const Scalar& _s)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if( _size.height > 0 && _size.width > 0 )
    {
        create( _size.height, _size.width, _type );
        *this = _s;
    }
}

inline GpuMat::GpuMat(const GpuMat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend)
{
    if( refcount )
        CV_XADD(refcount, 1);
}

inline GpuMat::GpuMat(int _rows, int _cols, int _type, void* _data, size_t _step)
    : flags(Mat::MAGIC_VAL + (_type & TYPE_MASK)), rows(_rows), cols(_cols), step(_step), data((uchar*)_data), refcount(0),
    datastart((uchar*)_data), dataend((uchar*)_data)
{
    size_t minstep = cols*elemSize();
    if( step == Mat::AUTO_STEP )
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if( rows == 1 ) step = minstep;
        CV_DbgAssert( step >= minstep );
        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step*(rows-1) + minstep;
}

inline GpuMat::GpuMat(Size _size, int _type, void* _data, size_t _step)
    : flags(Mat::MAGIC_VAL + (_type & TYPE_MASK)), rows(_size.height), cols(_size.width),
    step(_step), data((uchar*)_data), refcount(0),
    datastart((uchar*)_data), dataend((uchar*)_data)
{
    size_t minstep = cols*elemSize();
    if( step == Mat::AUTO_STEP )
    {
        step = minstep;
        flags |= Mat::CONTINUOUS_FLAG;
    }
    else
    {
        if( rows == 1 ) step = minstep;
        CV_DbgAssert( step >= minstep );
        flags |= step == minstep ? Mat::CONTINUOUS_FLAG : 0;
    }
    dataend += step*(rows-1) + minstep;
}


inline GpuMat::GpuMat(const GpuMat& m, const Range& rowRange, const Range& colRange)
{
    flags = m.flags;
    step = m.step; refcount = m.refcount;
    data = m.data; datastart = m.datastart; dataend = m.dataend;

    if( rowRange == Range::all() )
        rows = m.rows;
    else
    {
        CV_Assert( 0 <= rowRange.start && rowRange.start <= rowRange.end && rowRange.end <= m.rows );
        rows = rowRange.size();
        data += step*rowRange.start;
    }

    if( colRange == Range::all() )
        cols = m.cols;
    else
    {
        CV_Assert( 0 <= colRange.start && colRange.start <= colRange.end && colRange.end <= m.cols );
        cols = colRange.size();
        data += colRange.start*elemSize();
        flags &= cols < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    }

    if( rows == 1 )
        flags |= Mat::CONTINUOUS_FLAG;

    if( refcount )
        CV_XADD(refcount, 1);
    if( rows <= 0 || cols <= 0 )
        rows = cols = 0;
}

inline GpuMat::GpuMat(const GpuMat& m, const Rect& roi)
    : flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~Mat::CONTINUOUS_FLAG : -1;
    data += roi.x*elemSize();
    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols &&
        0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );
    if( refcount )
        CV_XADD(refcount, 1);
    if( rows <= 0 || cols <= 0 )
        rows = cols = 0;
}

inline GpuMat::GpuMat(const Mat& m)
: flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0) { upload(m); }

inline GpuMat::~GpuMat() { release(); }

inline GpuMat& GpuMat::operator = (const GpuMat& m)
{
    if( this != &m )
    {
        if( m.refcount )
            CV_XADD(m.refcount, 1);
        release();
        flags = m.flags;
        rows = m.rows; cols = m.cols;
        step = m.step; data = m.data;
        datastart = m.datastart; dataend = m.dataend;
        refcount = m.refcount;
    }
    return *this;
}

inline GpuMat& GpuMat::operator = (const Mat& m) { upload(m); return *this; }

template <class T> inline GpuMat::operator DevMem2D_<T>() const { return DevMem2D_<T>(rows, cols, (T*)data, step); }
template <class T> inline GpuMat::operator PtrStep_<T>() const { return PtrStep_<T>(static_cast< DevMem2D_<T> >(*this)); }

//CPP: void GpuMat::upload(const Mat& m);

 inline GpuMat::operator Mat() const
 {
     Mat m;
     download(m);
     return m;
 }

//CPP void GpuMat::download(cv::Mat& m) const;

inline GpuMat GpuMat::row(int y) const { return GpuMat(*this, Range(y, y+1), Range::all()); }
inline GpuMat GpuMat::col(int x) const { return GpuMat(*this, Range::all(), Range(x, x+1)); }
inline GpuMat GpuMat::rowRange(int startrow, int endrow) const { return GpuMat(*this, Range(startrow, endrow), Range::all()); }
inline GpuMat GpuMat::rowRange(const Range& r) const { return GpuMat(*this, r, Range::all()); }
inline GpuMat GpuMat::colRange(int startcol, int endcol) const { return GpuMat(*this, Range::all(), Range(startcol, endcol)); }
inline GpuMat GpuMat::colRange(const Range& r) const { return GpuMat(*this, Range::all(), r); }

inline GpuMat GpuMat::clone() const
{
    GpuMat m;
    copyTo(m);
    return m;
}

//CPP void GpuMat::copyTo( GpuMat& m ) const;
//CPP void GpuMat::copyTo( GpuMat& m, const GpuMat& mask  ) const;
//CPP void GpuMat::convertTo( GpuMat& m, int rtype, double alpha=1, double beta=0 ) const;

inline void GpuMat::assignTo( GpuMat& m, int type ) const
{
    if( type < 0 )
        m = *this;
    else
        convertTo(m, type);
}

//CPP GpuMat& GpuMat::operator = (const Scalar& s);
//CPP GpuMat& GpuMat::setTo(const Scalar& s, const GpuMat& mask=GpuMat());
//CPP GpuMat GpuMat::reshape(int _cn, int _rows=0) const;
inline void GpuMat::create(Size _size, int _type) { create(_size.height, _size.width, _type); }
//CPP void GpuMat::create(int _rows, int _cols, int _type);
//CPP void GpuMat::release();

inline void GpuMat::swap(GpuMat& b)
{
    std::swap( flags, b.flags );
    std::swap( rows, b.rows ); std::swap( cols, b.cols );
    std::swap( step, b.step ); std::swap( data, b.data );
    std::swap( datastart, b.datastart );
    std::swap( dataend, b.dataend );
    std::swap( refcount, b.refcount );
}

inline void GpuMat::locateROI( Size& wholeSize, Point& ofs ) const
{
    size_t esz = elemSize(), minstep;
    ptrdiff_t delta1 = data - datastart, delta2 = dataend - datastart;
    CV_DbgAssert( step > 0 );
    if( delta1 == 0 )
        ofs.x = ofs.y = 0;
    else
    {
        ofs.y = (int)(delta1/step);
        ofs.x = (int)((delta1 - step*ofs.y)/esz);
        CV_DbgAssert( data == datastart + ofs.y*step + ofs.x*esz );
    }
    minstep = (ofs.x + cols)*esz;
    wholeSize.height = (int)((delta2 - minstep)/step + 1);
    wholeSize.height = std::max(wholeSize.height, ofs.y + rows);
    wholeSize.width = (int)((delta2 - step*(wholeSize.height-1))/esz);
    wholeSize.width = std::max(wholeSize.width, ofs.x + cols);
}

inline GpuMat& GpuMat::adjustROI( int dtop, int dbottom, int dleft, int dright )
{
    Size wholeSize; Point ofs;
    size_t esz = elemSize();
    locateROI( wholeSize, ofs );
    int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
    int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
    data += (row1 - ofs.y)*step + (col1 - ofs.x)*esz;
    rows = row2 - row1; cols = col2 - col1;
    if( esz*cols == step || rows == 1 )
        flags |= Mat::CONTINUOUS_FLAG;
    else
        flags &= ~Mat::CONTINUOUS_FLAG;
    return *this;
}

inline GpuMat GpuMat::operator()( Range rowRange, Range colRange ) const { return GpuMat(*this, rowRange, colRange); }
inline GpuMat GpuMat::operator()( const Rect& roi ) const { return GpuMat(*this, roi); }

inline bool GpuMat::isContinuous() const { return (flags & Mat::CONTINUOUS_FLAG) != 0; }
inline size_t GpuMat::elemSize() const { return CV_ELEM_SIZE(flags); }
inline size_t GpuMat::elemSize1() const { return CV_ELEM_SIZE1(flags); }
inline int GpuMat::type() const { return CV_MAT_TYPE(flags); }
inline int GpuMat::depth() const { return CV_MAT_DEPTH(flags); }
inline int GpuMat::channels() const { return CV_MAT_CN(flags); }
inline size_t GpuMat::step1() const { return step/elemSize1(); }
inline Size GpuMat::size() const { return Size(cols, rows); }
inline bool GpuMat::empty() const { return data == 0; }

inline uchar* GpuMat::ptr(int y)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step*y;
}

inline const uchar* GpuMat::ptr(int y) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step*y;
}

template<typename _Tp> inline _Tp* GpuMat::ptr(int y)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return (_Tp*)(data + step*y);
}

template<typename _Tp> inline const _Tp* GpuMat::ptr(int y) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return (const _Tp*)(data + step*y);
}

inline GpuMat GpuMat::t() const
{
    GpuMat tmp;
    transpose(*this, tmp);
    return tmp;
}

static inline void swap( GpuMat& a, GpuMat& b ) { a.swap(b); }


///////////////////////////////////////////////////////////////////////
//////////////////////////////// CudaMem ////////////////////////////////
///////////////////////////////////////////////////////////////////////

inline CudaMem::CudaMem()  : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0) {}
inline CudaMem::CudaMem(int _rows, int _cols, int _type, int _alloc_type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0)
{
    if( _rows > 0 && _cols > 0 )
        create( _rows, _cols, _type, _alloc_type);
}

inline CudaMem::CudaMem(Size _size, int _type, int _alloc_type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0)
{
    if( _size.height > 0 && _size.width > 0 )
        create( _size.height, _size.width, _type, _alloc_type);
}

inline CudaMem::CudaMem(const CudaMem& m) : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data), refcount(m.refcount), datastart(m.datastart), dataend(m.dataend), alloc_type(m.alloc_type)
{
    if( refcount )
        CV_XADD(refcount, 1);
}

inline CudaMem::CudaMem(const Mat& m, int _alloc_type) : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0), alloc_type(0)
{
    if( m.rows > 0 && m.cols > 0 )
        create( m.size(), m.type(), _alloc_type);

    Mat tmp = createMatHeader();
    m.copyTo(tmp);
}

inline CudaMem::~CudaMem()
{
    release();
}

inline CudaMem& CudaMem::operator = (const CudaMem& m)
{
    if( this != &m )
    {
        if( m.refcount )
            CV_XADD(m.refcount, 1);
        release();
        flags = m.flags;
        rows = m.rows; cols = m.cols;
        step = m.step; data = m.data;
        datastart = m.datastart;
        dataend = m.dataend;
        refcount = m.refcount;
        alloc_type = m.alloc_type;
    }
    return *this;
}

inline CudaMem CudaMem::clone() const
{
    CudaMem m(size(), type(), alloc_type);
    Mat to = m;
    Mat from = *this;
    from.copyTo(to);
    return m;
}

inline void CudaMem::create(Size _size, int _type, int _alloc_type) { create(_size.height, _size.width, _type, _alloc_type); }


//CCP void CudaMem::create(int _rows, int _cols, int _type, int _alloc_type);
//CPP void CudaMem::release();

inline Mat CudaMem::createMatHeader() const { return Mat(size(), type(), data); }
inline CudaMem::operator Mat() const { return createMatHeader(); }

inline CudaMem::operator GpuMat() const { return createGpuMatHeader(); }
//CPP GpuMat CudaMem::createGpuMatHeader() const;

inline bool CudaMem::isContinuous() const { return (flags & Mat::CONTINUOUS_FLAG) != 0; }
inline size_t CudaMem::elemSize() const { return CV_ELEM_SIZE(flags); }
inline size_t CudaMem::elemSize1() const { return CV_ELEM_SIZE1(flags); }
inline int CudaMem::type() const { return CV_MAT_TYPE(flags); }
inline int CudaMem::depth() const { return CV_MAT_DEPTH(flags); }
inline int CudaMem::channels() const { return CV_MAT_CN(flags); }
inline size_t CudaMem::step1() const { return step/elemSize1(); }
inline Size CudaMem::size() const { return Size(cols, rows); }
inline bool CudaMem::empty() const { return data == 0; }

//////////////////////////////////////////////////////////////////////////////
// Arithmetical operations

inline GpuMat operator ~ (const GpuMat& src)
{
    GpuMat dst;
    bitwise_not(src, dst);
    return dst;
}


inline GpuMat operator | (const GpuMat& src1, const GpuMat& src2)
{
    GpuMat dst;
    bitwise_or(src1, src2, dst);
    return dst;
}


inline GpuMat operator & (const GpuMat& src1, const GpuMat& src2)
{
    GpuMat dst;
    bitwise_and(src1, src2, dst);
    return dst;
}


inline GpuMat operator ^ (const GpuMat& src1, const GpuMat& src2)
{
    GpuMat dst;
    bitwise_xor(src1, src2, dst);
    return dst;
}


} /* end of namespace gpu */

} /* end of namespace cv */

#endif /* __OPENCV_GPU_MATRIX_OPERATIONS_HPP__ */
