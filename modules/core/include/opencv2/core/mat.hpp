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

#ifndef __OPENCV_CORE_MATRIX_OPERATIONS_HPP__
#define __OPENCV_CORE_MATRIX_OPERATIONS_HPP__

#ifndef SKIP_INCLUDES
#include <limits.h>
#include <string.h>
#endif // SKIP_INCLUDES

#ifdef __cplusplus

namespace cv
{

//////////////////////////////// Mat ////////////////////////////////

inline Mat::Mat()
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0) {}

inline Mat::Mat(int _rows, int _cols, int _type)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0), datastart(0), dataend(0)
{
    if( _rows > 0 && _cols > 0 )
        create( _rows, _cols, _type );
}

inline Mat::Mat(int _rows, int _cols, int _type, const Scalar& _s)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    if(_rows > 0 && _cols > 0)
    {
        create(_rows, _cols, _type);
        *this = _s;
    }
}

inline Mat::Mat(Size _size, int _type)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    if( _size.height > 0 && _size.width > 0 )
        create( _size.height, _size.width, _type );
}
    
inline Mat::Mat(Size _size, int _type, const Scalar& _s)
    : flags(0), rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    if( _size.height > 0 && _size.width > 0 )
    {
        create( _size.height, _size.width, _type );
        *this = _s;
    }
}    

inline Mat::Mat(const Mat& m)
    : flags(m.flags), rows(m.rows), cols(m.cols), step(m.step), data(m.data),
    refcount(m.refcount), datastart(m.datastart), dataend(m.dataend)
{
    if( refcount )
        CV_XADD(refcount, 1);
}

inline Mat::Mat(int _rows, int _cols, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), rows(_rows), cols(_cols),
    step(_step), data((uchar*)_data), refcount(0),
    datastart((uchar*)_data), dataend((uchar*)_data)
{
    size_t minstep = cols*elemSize();
    if( step == AUTO_STEP )
    {
        step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        if( rows == 1 ) step = minstep;
        CV_DbgAssert( step >= minstep );
        flags |= step == minstep ? CONTINUOUS_FLAG : 0;
    }
    dataend += step*(rows-1) + minstep;
}

inline Mat::Mat(Size _size, int _type, void* _data, size_t _step)
    : flags(MAGIC_VAL + (_type & TYPE_MASK)), rows(_size.height), cols(_size.width),
    step(_step), data((uchar*)_data), refcount(0),
    datastart((uchar*)_data), dataend((uchar*)_data)
{
    size_t minstep = cols*elemSize();
    if( step == AUTO_STEP )
    {
        step = minstep;
        flags |= CONTINUOUS_FLAG;
    }
    else
    {
        if( rows == 1 ) step = minstep;
        CV_DbgAssert( step >= minstep );
        flags |= step == minstep ? CONTINUOUS_FLAG : 0;
    }
    dataend += step*(rows-1) + minstep;
}

inline Mat::Mat(const Mat& m, const Range& rowRange, const Range& colRange)
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
        flags &= cols < m.cols ? ~CONTINUOUS_FLAG : -1;
    }

    if( rows == 1 )
        flags |= CONTINUOUS_FLAG;

    if( refcount )
        CV_XADD(refcount, 1);
    if( rows <= 0 || cols <= 0 )
        rows = cols = 0;
}

inline Mat::Mat(const Mat& m, const Rect& roi)
    : flags(m.flags), rows(roi.height), cols(roi.width),
    step(m.step), data(m.data + roi.y*step), refcount(m.refcount),
    datastart(m.datastart), dataend(m.dataend)
{
    flags &= roi.width < m.cols ? ~CONTINUOUS_FLAG : -1;
    data += roi.x*elemSize();
    CV_Assert( 0 <= roi.x && 0 <= roi.width && roi.x + roi.width <= m.cols &&
        0 <= roi.y && 0 <= roi.height && roi.y + roi.height <= m.rows );
    if( refcount )
        CV_XADD(refcount, 1);
    if( rows <= 0 || cols <= 0 )
        rows = cols = 0;
}

inline Mat::Mat(const CvMat* m, bool copyData)
    : flags(MAGIC_VAL + (m->type & (CV_MAT_TYPE_MASK|CV_MAT_CONT_FLAG))),
    rows(m->rows), cols(m->cols), step(m->step), data(m->data.ptr), refcount(0),
    datastart(m->data.ptr), dataend(m->data.ptr)
{
    if( step == 0 )
        step = cols*elemSize();
    size_t minstep = cols*elemSize();
    dataend += step*(rows-1) + minstep;
    if( copyData )
    {
        data = datastart = dataend = 0;
        Mat(m->rows, m->cols, m->type, m->data.ptr, m->step).copyTo(*this);
    }
}

CV_EXPORTS Mat cvarrToMat(const CvArr* arr, bool copyData=false,
                             bool allowND=true, int coiMode=0);

template<typename _Tp> inline Mat::Mat(const vector<_Tp>& vec, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG),
    rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    if(vec.empty())
        return;
    if( !copyData )
    {
        rows = (int)vec.size();
        cols = 1;
        step = sizeof(_Tp);
        data = datastart = (uchar*)&vec[0];
        dataend = datastart + rows*step;
    }
    else
        Mat((int)vec.size(), 1, DataType<_Tp>::type, (uchar*)&vec[0]).copyTo(*this);
}
    
    
template<typename _Tp, int n> inline Mat::Mat(const Vec<_Tp, n>& vec, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG),
    rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    if( !copyData )
    {
        rows = n;
        cols = 1;
        step = sizeof(_Tp);
        data = datastart = (uchar*)vec.val;
        dataend = datastart + rows*step;
    }
    else
        Mat(n, 1, DataType<_Tp>::type, vec.val).copyTo(*this);
}


template<typename _Tp, int m, int n> inline Mat::Mat(const Matx<_Tp,m,n>& M, bool copyData)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG),
    rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    if( !copyData )
    {
        rows = m;
        cols = n;
        step = n*sizeof(_Tp);
        data = datastart = (uchar*)M.val;
        dataend = datastart + rows*step;
    }
    else
        Mat(m, n, DataType<_Tp>::type, (uchar*)M.val).copyTo(*this);    
}

    
template<typename _Tp> inline Mat::Mat(const Point_<_Tp>& pt)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG),
    rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    create(2, 1, DataType<_Tp>::type);
    ((_Tp*)data)[0] = pt.x;
    ((_Tp*)data)[1] = pt.y;
}
    

template<typename _Tp> inline Mat::Mat(const Point3_<_Tp>& pt)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG),
    rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    create(3, 1, DataType<_Tp>::type);
    ((_Tp*)data)[0] = pt.x;
    ((_Tp*)data)[1] = pt.y;
    ((_Tp*)data)[2] = pt.z;
}

template<typename _Tp> inline Mat::Mat(const MatCommaInitializer_<_Tp>& commaInitializer)
    : flags(MAGIC_VAL | DataType<_Tp>::type | CV_MAT_CONT_FLAG),
    rows(0), cols(0), step(0), data(0), refcount(0),
    datastart(0), dataend(0)
{
    *this = *commaInitializer;
}
    
inline Mat::~Mat()
{
    release();
}

inline Mat& Mat::operator = (const Mat& m)
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
    
inline Mat Mat::row(int y) const { return Mat(*this, Range(y, y+1), Range::all()); }
inline Mat Mat::col(int x) const { return Mat(*this, Range::all(), Range(x, x+1)); }
inline Mat Mat::rowRange(int startrow, int endrow) const
    { return Mat(*this, Range(startrow, endrow), Range::all()); }
inline Mat Mat::rowRange(const Range& r) const
    { return Mat(*this, r, Range::all()); }
inline Mat Mat::colRange(int startcol, int endcol) const
    { return Mat(*this, Range::all(), Range(startcol, endcol)); }
inline Mat Mat::colRange(const Range& r) const
    { return Mat(*this, Range::all(), r); }

inline Mat Mat::diag(int d) const
{
    Mat m = *this;
    size_t esz = elemSize();
    int len;

    if( d >= 0 )
    {
        len = std::min(cols - d, rows);
        m.data += esz*d;
    }
    else
    {
        len = std::min(rows + d, cols);
        m.data -= step*d;
    }
    CV_DbgAssert( len > 0 );
    m.rows = len;
    m.cols = 1;
    m.step += esz;
    if( m.rows > 1 )
        m.flags &= ~CONTINUOUS_FLAG;
    else
        m.flags |= CONTINUOUS_FLAG;
    return m;
}

inline Mat Mat::diag(const Mat& d)
{
    Mat m(d.rows, d.rows, d.type(), Scalar(0)), md = m.diag();
    d.copyTo(md);
    return m;
}

inline Mat Mat::clone() const
{
    Mat m;
    copyTo(m);
    return m;
}

inline void Mat::assignTo( Mat& m, int type ) const
{
    if( type < 0 )
        m = *this;
    else
        convertTo(m, type);
}

inline void Mat::create(int _rows, int _cols, int _type)
{
    _type &= TYPE_MASK;
    if( rows == _rows && cols == _cols && type() == _type && data )
        return;
    if( data )
        release();
    CV_DbgAssert( _rows >= 0 && _cols >= 0 );
    if( _rows > 0 && _cols > 0 )
    {
        flags = MAGIC_VAL + CONTINUOUS_FLAG + _type;
        rows = _rows;
        cols = _cols;
        step = elemSize()*cols;
        int64 _nettosize = (int64)step*rows;
        size_t nettosize = (size_t)_nettosize;
        if( _nettosize != (int64)nettosize )
            CV_Error(CV_StsNoMem, "Too big buffer is allocated");
        size_t datasize = alignSize(nettosize, (int)sizeof(*refcount));
        datastart = data = (uchar*)fastMalloc(datasize + sizeof(*refcount));
        dataend = data + nettosize;
        refcount = (int*)(data + datasize);
        *refcount = 1;
    }
}

inline void Mat::create(Size _size, int _type)
{
    create(_size.height, _size.width, _type);
}

inline void Mat::addref()
{ if( refcount ) CV_XADD(refcount, 1); }

inline void Mat::release()
{
    if( refcount && CV_XADD(refcount, -1) == 1 )
        fastFree(datastart);
    data = datastart = dataend = 0;
    step = rows = cols = 0;
    refcount = 0;
}

inline void Mat::locateROI( Size& wholeSize, Point& ofs ) const
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

inline Mat& Mat::adjustROI( int dtop, int dbottom, int dleft, int dright )
{
    Size wholeSize; Point ofs;
    size_t esz = elemSize();
    locateROI( wholeSize, ofs );
    int row1 = std::max(ofs.y - dtop, 0), row2 = std::min(ofs.y + rows + dbottom, wholeSize.height);
    int col1 = std::max(ofs.x - dleft, 0), col2 = std::min(ofs.x + cols + dright, wholeSize.width);
    data += (row1 - ofs.y)*step + (col1 - ofs.x)*esz;
    rows = row2 - row1; cols = col2 - col1;
    if( esz*cols == step || rows == 1 )
        flags |= CONTINUOUS_FLAG;
    else
        flags &= ~CONTINUOUS_FLAG;
    return *this;
}

inline Mat Mat::operator()( Range rowRange, Range colRange ) const
{
    return Mat(*this, rowRange, colRange);
}

inline Mat Mat::operator()( const Rect& roi ) const
{ return Mat(*this, roi); }

inline Mat::operator CvMat() const
{
    CvMat m = cvMat(rows, cols, type(), data);
    m.step = (int)step;
    m.type = (m.type & ~CONTINUOUS_FLAG) | (flags & CONTINUOUS_FLAG);
    return m;
}

inline bool Mat::isContinuous() const { return (flags & CONTINUOUS_FLAG) != 0; }
inline size_t Mat::elemSize() const { return CV_ELEM_SIZE(flags); }
inline size_t Mat::elemSize1() const { return CV_ELEM_SIZE1(flags); }
inline int Mat::type() const { return CV_MAT_TYPE(flags); }
inline int Mat::depth() const { return CV_MAT_DEPTH(flags); }
inline int Mat::channels() const { return CV_MAT_CN(flags); }
inline size_t Mat::step1() const { return step/elemSize1(); }
inline Size Mat::size() const { return Size(cols, rows); }
inline bool Mat::empty() const { return data == 0; }

inline uchar* Mat::ptr(int y)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step*y;
}

inline const uchar* Mat::ptr(int y) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return data + step*y;
}

template<typename _Tp> inline _Tp* Mat::ptr(int y)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return (_Tp*)(data + step*y);
}

template<typename _Tp> inline const _Tp* Mat::ptr(int y) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows );
    return (const _Tp*)(data + step*y);
}

template<typename _Tp> inline _Tp& Mat::at(int y, int x)
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows &&
        (unsigned)(x*DataType<_Tp>::channels) < (unsigned)(cols*channels()) &&
        CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((_Tp*)(data + step*y))[x];
}

template<typename _Tp> inline const _Tp& Mat::at(int y, int x) const
{
    CV_DbgAssert( (unsigned)y < (unsigned)rows &&
        (unsigned)(x*DataType<_Tp>::channels) < (unsigned)(cols*channels()) &&
        CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((const _Tp*)(data + step*y))[x];
}

template<typename _Tp> inline _Tp& Mat::at(int i)
{
    CV_DbgAssert( CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1() );
    
    if( cols == 1 )
    {
        CV_DbgAssert( (unsigned)i < (unsigned)rows );
        return *(_Tp*)(data + step*i);
    }
    
    CV_DbgAssert( rows == 1 && (unsigned)(i*DataType<_Tp>::channels) < (unsigned)(cols*channels()) );
    return ((_Tp*)data)[i];
}

template<typename _Tp> inline const _Tp& Mat::at(int i) const
{
    CV_DbgAssert( CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1() );
    
    if( cols == 1 )
    {
        CV_DbgAssert( (unsigned)i < (unsigned)rows );
        return *(_Tp*)(data + step*i);
    }
    
    CV_DbgAssert( rows == 1 && (unsigned)(i*DataType<_Tp>::channels) < (unsigned)(cols*channels()) );
    return ((_Tp*)data)[i];
}
    
template<typename _Tp> inline _Tp& Mat::at(Point pt)
{
    CV_DbgAssert( (unsigned)pt.y < (unsigned)rows &&
        (unsigned)(pt.x*DataType<_Tp>::channels) < (unsigned)(cols*channels()) &&
        CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((_Tp*)(data + step*pt.y))[pt.x];
}

template<typename _Tp> inline const _Tp& Mat::at(Point pt) const
{
    CV_DbgAssert( (unsigned)pt.y < (unsigned)rows &&
        (unsigned)(pt.x*DataType<_Tp>::channels) < (unsigned)(cols*channels()) &&
        CV_ELEM_SIZE1(DataType<_Tp>::depth) == elemSize1());
    return ((const _Tp*)(data + step*pt.y))[pt.x];
}
    
template<typename _Tp> inline MatConstIterator_<_Tp> Mat::begin() const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatConstIterator_<_Tp>((const Mat_<_Tp>*)this);
}

template<typename _Tp> inline MatConstIterator_<_Tp> Mat::end() const
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatConstIterator_<_Tp> it((const Mat_<_Tp>*)this);
    it.ptr = it.sliceEnd = (_Tp*)(data + step*(rows-1)) + cols;
    return it;
}

template<typename _Tp> inline MatIterator_<_Tp> Mat::begin()
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    return MatIterator_<_Tp>((Mat_<_Tp>*)this);
}

template<typename _Tp> inline MatIterator_<_Tp> Mat::end()
{
    CV_DbgAssert( elemSize() == sizeof(_Tp) );
    MatIterator_<_Tp> it((Mat_<_Tp>*)this);
    it.ptr = it.sliceEnd = (_Tp*)(data + step*(rows-1)) + cols;
    return it;
}
        
static inline void swap( Mat& a, Mat& b )
{
    std::swap( a.flags, b.flags );
    std::swap( a.rows, b.rows ); std::swap( a.cols, b.cols );
    std::swap( a.step, b.step ); std::swap( a.data, b.data );
    std::swap( a.datastart, b.datastart );
    std::swap( a.dataend, b.dataend );
    std::swap( a.refcount, b.refcount );
}
    
template<typename _Tp> inline Mat::operator vector<_Tp>() const
{
    CV_Assert( (rows == 1 || cols == 1) && channels() == DataType<_Tp>::channels );
    
    int n = rows + cols - 1;
    if( isContinuous() && type() == DataType<_Tp>::type )
        return vector<_Tp>((_Tp*)data,(_Tp*)data + n);
    vector<_Tp> v(n); Mat tmp(rows, cols, DataType<_Tp>::type, &v[0]);
    convertTo(tmp, tmp.type());
    return v;
}

template<typename _Tp, int n> inline Mat::operator Vec<_Tp, n>() const
{
    CV_Assert( (rows == 1 || cols == 1) && rows + cols - 1 == n &&
               channels() == DataType<_Tp>::channels );
    
    if( isContinuous() && type() == DataType<_Tp>::type )
        return Vec<_Tp, n>((_Tp*)data);
    Vec<_Tp, n> v; Mat tmp(rows, cols, DataType<_Tp>::type, v.val);
    convertTo(tmp, tmp.type());
    return v;
}
    
template<typename _Tp, int m, int n> inline Mat::operator Matx<_Tp, m, n>() const
{
    CV_Assert( rows == m && cols == n &&
               channels() == DataType<_Tp>::channels );
    
    if( isContinuous() && type() == DataType<_Tp>::type )
        return Matx<_Tp, m, n>((_Tp*)data);
    Matx<_Tp, m, n> mtx; Mat tmp(rows, cols, DataType<_Tp>::type, mtx.val);
    convertTo(tmp, tmp.type());
    return mtx;
}    
    
inline SVD::SVD() {}
inline SVD::SVD( const Mat& m, int flags ) { operator ()(m, flags); }
inline void SVD::solveZ( const Mat& m, Mat& dst )
{
    SVD svd(m);
    svd.vt.row(svd.vt.rows-1).reshape(1,svd.vt.cols).copyTo(dst);
}

template<typename _Tp, int m, int n, int nm> inline void
    SVD::compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt )
{
    assert( nm == MIN(m, n));
    Mat _a(a, false), _u(u, false), _w(w, false), _vt(vt, false);
    SVD::compute(_a, _w, _u, _vt);
    CV_Assert(_w.data == (uchar*)&w.val[0] && _u.data == (uchar*)&u.val[0] && _vt.data == (uchar*)&vt.val[0]);
}
    
template<typename _Tp, int m, int n, int nm> inline void
SVD::compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w )
{
    assert( nm == MIN(m, n));
    Mat _a(a, false), _w(w, false);
    SVD::compute(_a, _w);
    CV_Assert(_w.data == (uchar*)&w.val[0]);
}
    
template<typename _Tp, int m, int n, int nm, int nb> inline void
SVD::backSubst( const Matx<_Tp, nm, 1>& w, const Matx<_Tp, m, nm>& u,
                const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs,
                Matx<_Tp, n, nb>& dst )
{
    assert( nm == MIN(m, n));
    Mat _u(u, false), _w(w, false), _vt(vt, false), _rhs(_rhs, false), _dst(dst, false);
    SVD::backSubst(_w, _u, _vt, _rhs, _dst);
    CV_Assert(_dst.data == (uchar*)&dst.val[0]);
}
    
///////////////////////////////// Mat_<_Tp> ////////////////////////////////////

template<typename _Tp> inline Mat_<_Tp>::Mat_() :
    Mat() { flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<_Tp>::type; }
    
template<typename _Tp> inline Mat_<_Tp>::Mat_(int _rows, int _cols) :
    Mat(_rows, _cols, DataType<_Tp>::type) {}

template<typename _Tp> inline Mat_<_Tp>::Mat_(int _rows, int _cols, const _Tp& value) :
    Mat(_rows, _cols, DataType<_Tp>::type) { *this = value; }

template<typename _Tp> inline Mat_<_Tp>::Mat_(Size _size) :
    Mat(_size.height, _size.width, DataType<_Tp>::type) {}
    
template<typename _Tp> inline Mat_<_Tp>::Mat_(Size _size, const _Tp& value) :
    Mat(_size.height, _size.width, DataType<_Tp>::type) { *this = value; }
    
template<typename _Tp> inline Mat_<_Tp>::Mat_(const Mat& m) : Mat()
{ flags = (flags & ~CV_MAT_TYPE_MASK) | DataType<_Tp>::type; *this = m; }

template<typename _Tp> inline Mat_<_Tp>::Mat_(const Mat_& m) : Mat(m) {}

template<typename _Tp> inline Mat_<_Tp>::Mat_(int _rows, int _cols, _Tp* _data, size_t _step)
    : Mat(_rows, _cols, DataType<_Tp>::type, _data, _step) {}

template<typename _Tp> inline Mat_<_Tp>::Mat_(const Mat_& m, const Range& rowRange, const Range& colRange)
    : Mat(m, rowRange, colRange) {}

template<typename _Tp> inline Mat_<_Tp>::Mat_(const Mat_& m, const Rect& roi)
    : Mat(m, roi) {}

template<typename _Tp> template<int n> inline
    Mat_<_Tp>::Mat_(const Vec<_Tp, n>& vec, bool copyData)
    : Mat(vec, copyData)
{
}

template<typename _Tp> template<int m, int n> inline
    Mat_<_Tp>::Mat_(const Matx<_Tp,m,n>& M, bool copyData)
    : Mat(M, copyData)
{
}    
    
template<typename _Tp> inline Mat_<_Tp>::Mat_(const Point_<_Tp>& pt)
    : Mat(2, 1, DataType<_Tp>::type)
{
    ((_Tp*)data)[0] = pt.x;
    ((_Tp*)data)[1] = pt.y;
}


template<typename _Tp> inline Mat_<_Tp>::Mat_(const Point3_<_Tp>& pt)
    : Mat(3, 1, DataType<_Tp>::type)
{
    ((_Tp*)data)[0] = pt.x;
    ((_Tp*)data)[1] = pt.y;
    ((_Tp*)data)[2] = pt.z;
}
    

template<typename _Tp> inline Mat_<_Tp>::Mat_(const MatCommaInitializer_<_Tp>& commaInitializer)
: Mat(commaInitializer)
{
}
    
    
template<typename _Tp> inline Mat_<_Tp>::Mat_(const vector<_Tp>& vec, bool copyData)
    : Mat(vec, copyData)
{}

template<typename _Tp> inline Mat_<_Tp>& Mat_<_Tp>::operator = (const Mat& m)
{
    if( DataType<_Tp>::type == m.type() )
    {
        Mat::operator = (m);
        return *this;
    }
    if( DataType<_Tp>::depth == m.depth() )
    {
        return (*this = m.reshape(DataType<_Tp>::channels));
    }
    CV_DbgAssert(DataType<_Tp>::channels == m.channels());
    m.convertTo(*this, type());
    return *this;
}

template<typename _Tp> inline Mat_<_Tp>& Mat_<_Tp>::operator = (const Mat_& m)
{
    Mat::operator=(m);
    return *this;
}

template<typename _Tp> inline Mat_<_Tp>& Mat_<_Tp>::operator = (const _Tp& s)
{
    typedef typename DataType<_Tp>::vec_type VT;
    Mat::operator=(Scalar((const VT&)s));
    return *this;
}
    

template<typename _Tp> inline void Mat_<_Tp>::create(int _rows, int _cols)
{
    Mat::create(_rows, _cols, DataType<_Tp>::type);
}

template<typename _Tp> inline void Mat_<_Tp>::create(Size _size)
{
    Mat::create(_size, DataType<_Tp>::type);
}

template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::cross(const Mat_& m) const
{ return Mat_<_Tp>(Mat::cross(m)); }

template<typename _Tp> template<typename T2> inline Mat_<_Tp>::operator Mat_<T2>() const
{ return Mat_<T2>(*this); }

template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::row(int y) const
{ return Mat_(*this, Range(y, y+1), Range::all()); }
template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::col(int x) const
{ return Mat_(*this, Range::all(), Range(x, x+1)); }
template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::diag(int d) const
{ return Mat_(Mat::diag(d)); }
template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::clone() const
{ return Mat_(Mat::clone()); }

template<typename _Tp> inline size_t Mat_<_Tp>::elemSize() const
{
    CV_DbgAssert( Mat::elemSize() == sizeof(_Tp) );
    return sizeof(_Tp);
}

template<typename _Tp> inline size_t Mat_<_Tp>::elemSize1() const
{
    CV_DbgAssert( Mat::elemSize1() == sizeof(_Tp)/DataType<_Tp>::channels );
    return sizeof(_Tp)/DataType<_Tp>::channels;
}
template<typename _Tp> inline int Mat_<_Tp>::type() const
{
    CV_DbgAssert( Mat::type() == DataType<_Tp>::type );
    return DataType<_Tp>::type;
}
template<typename _Tp> inline int Mat_<_Tp>::depth() const
{
    CV_DbgAssert( Mat::depth() == DataType<_Tp>::depth );
    return DataType<_Tp>::depth;
}
template<typename _Tp> inline int Mat_<_Tp>::channels() const
{
    CV_DbgAssert( Mat::channels() == DataType<_Tp>::channels );
    return DataType<_Tp>::channels;
}
template<typename _Tp> inline size_t Mat_<_Tp>::stepT() const { return step/elemSize(); }
template<typename _Tp> inline size_t Mat_<_Tp>::step1() const { return step/elemSize1(); }

template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::reshape(int _rows) const
{ return Mat_<_Tp>(Mat::reshape(0,_rows)); }

template<typename _Tp> inline Mat_<_Tp>& Mat_<_Tp>::adjustROI( int dtop, int dbottom, int dleft, int dright )
{ return (Mat_<_Tp>&)(Mat::adjustROI(dtop, dbottom, dleft, dright));  }

template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::operator()( const Range& rowRange, const Range& colRange ) const
{ return Mat_<_Tp>(*this, rowRange, colRange); }

template<typename _Tp> inline Mat_<_Tp> Mat_<_Tp>::operator()( const Rect& roi ) const
{ return Mat_<_Tp>(*this, roi); }

template<typename _Tp> inline _Tp* Mat_<_Tp>::operator [](int y)
{ return (_Tp*)ptr(y); }
template<typename _Tp> inline const _Tp* Mat_<_Tp>::operator [](int y) const
{ return (const _Tp*)ptr(y); }

template<typename _Tp> inline _Tp& Mat_<_Tp>::operator ()(int row, int col)
{
    CV_DbgAssert( (unsigned)row < (unsigned)rows &&
                 (unsigned)col < (unsigned)cols &&
                 type() == DataType<_Tp>::type );
    return ((_Tp*)(data + step*row))[col];
}

template<typename _Tp> inline const _Tp& Mat_<_Tp>::operator ()(int row, int col) const
{
    CV_DbgAssert( (unsigned)row < (unsigned)rows &&
                 (unsigned)col < (unsigned)cols &&
                 type() == DataType<_Tp>::type );
    return ((const _Tp*)(data + step*row))[col];
}

template<typename _Tp> inline _Tp& Mat_<_Tp>::operator ()(int i)
{
    return at<_Tp>(i);
}
    
template<typename _Tp> inline const _Tp& Mat_<_Tp>::operator ()(int i) const
{
    return at<_Tp>(i);
}
    
template<typename _Tp> inline _Tp& Mat_<_Tp>::operator ()(Point pt)
{
    CV_DbgAssert( (unsigned)pt.y < (unsigned)rows &&
                 (unsigned)pt.x < (unsigned)cols &&
                 type() == DataType<_Tp>::type );
    return ((_Tp*)(data + step*pt.y))[pt.x];
}

template<typename _Tp> inline const _Tp& Mat_<_Tp>::operator ()(Point pt) const
{
    CV_DbgAssert( (unsigned)pt.y < (unsigned)rows &&
                  (unsigned)pt.x < (unsigned)cols &&
                 type() == DataType<_Tp>::type );
    return ((const _Tp*)(data + step*pt.y))[pt.x];
}

template<typename _Tp> inline Mat_<_Tp>::operator vector<_Tp>() const
{
    return this->Mat::operator vector<_Tp>();
}

template<typename _Tp> template<int n> inline Mat_<_Tp>::operator Vec<_Tp, n>() const
{
    return this->Mat::operator Vec<_Tp, n>();
}

template<typename _Tp> template<int m, int n> inline Mat_<_Tp>::operator Matx<_Tp, m, n>() const
{
    return this->Mat::operator Matx<_Tp, m, n>();
}    

    
template<typename T1, typename T2, typename Op> inline void
process( const Mat_<T1>& m1, Mat_<T2>& m2, Op op )
{
    int y, x, rows = m1.rows, cols = m1.cols;
    int c1 = m1.channels(), c2 = m2.channels();

    CV_DbgAssert( m1.size() == m2.size() );

    for( y = 0; y < rows; y++ )
    {
        const T1* src = m1[y];
        T2* dst = m2[y];

        for( x = 0; x < cols; x++ )
            dst[x] = op(src[x]);
    }
}

template<typename T1, typename T2, typename T3, typename Op> inline void
process( const Mat_<T1>& m1, const Mat_<T2>& m2, Mat_<T3>& m3, Op op )
{
    int y, x, rows = m1.rows, cols = m1.cols;

    CV_DbgAssert( m1.size() == m2.size() );

    for( y = 0; y < rows; y++ )
    {
        const T1* src1 = m1[y];
        const T2* src2 = m2[y];
        T3* dst = m3[y];

        for( x = 0; x < cols; x++ )
            dst[x] = op( src1[x], src2[x] );
    }
}


class CV_EXPORTS MatOp
{    
public:
    MatOp() {};
    virtual ~MatOp() {};
    
    virtual bool elementWise(const MatExpr& expr) const;
    virtual void assign(const MatExpr& expr, Mat& m, int type=-1) const = 0;
    virtual void roi(const MatExpr& expr, const Range& rowRange,
                     const Range& colRange, MatExpr& res) const;
    virtual void diag(const MatExpr& expr, int d, MatExpr& res) const;
    virtual void augAssignAdd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignSubtract(const MatExpr& expr, Mat& m) const;
    virtual void augAssignMultiply(const MatExpr& expr, Mat& m) const;
    virtual void augAssignDivide(const MatExpr& expr, Mat& m) const;
    virtual void augAssignAnd(const MatExpr& expr, Mat& m) const;
    virtual void augAssignOr(const MatExpr& expr, Mat& m) const;
    virtual void augAssignXor(const MatExpr& expr, Mat& m) const;
    
    virtual void add(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void add(const MatExpr& expr1, const Scalar& s, MatExpr& res) const;
    
    virtual void subtract(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void subtract(const Scalar& s, const MatExpr& expr, MatExpr& res) const;
    
    virtual void multiply(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void multiply(const MatExpr& expr1, double s, MatExpr& res) const;
    
    virtual void divide(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res, double scale=1) const;
    virtual void divide(double s, const MatExpr& expr, MatExpr& res) const;
        
    virtual void abs(const MatExpr& expr, MatExpr& res) const;
    
    virtual void transpose(const MatExpr& expr, MatExpr& res) const;
    virtual void matmul(const MatExpr& expr1, const MatExpr& expr2, MatExpr& res) const;
    virtual void invert(const MatExpr& expr, int method, MatExpr& res) const;
};

    
class CV_EXPORTS MatExpr
{
public:
    MatExpr() : op(0), flags(0), a(Mat()), b(Mat()), c(Mat()), alpha(0), beta(0), s(Scalar()) {}
    MatExpr(const MatOp* _op, int _flags, const Mat& _a=Mat(), const Mat& _b=Mat(),
            const Mat& _c=Mat(), double _alpha=1, double _beta=1, const Scalar& _s=Scalar())
        : op(_op), flags(_flags), a(_a), b(_b), c(_c), alpha(_alpha), beta(_beta), s(_s) {}
    explicit MatExpr(const Mat& m);
    operator Mat() const
    {
        Mat m;
        op->assign(*this, m);
        return m;
    }
    
    template<typename _Tp> operator Mat_<_Tp>() const
    {
        Mat_<_Tp> m;
        op->assign(*this, m, DataType<_Tp>::type);
        return m;
    }
    
    MatExpr row(int y) const;
    MatExpr col(int x) const;
    MatExpr diag(int d=0) const;
    MatExpr operator()( const Range& rowRange, const Range& colRange ) const;
    MatExpr operator()( const Rect& roi ) const;
    
    Mat cross(const Mat& m) const;
    double dot(const Mat& m) const;
    
    MatExpr t() const;
    MatExpr inv(int method = DECOMP_LU) const;
    MatExpr mul(const MatExpr& e, double scale=1) const;
    MatExpr mul(const Mat& m, double scale=1) const;
    
    const MatOp* op;
    int flags;
    
    Mat a, b, c;
    double alpha, beta;
    Scalar s;
};
    

CV_EXPORTS MatExpr operator + (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator + (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator + (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator + (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator + (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator - (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator - (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const Mat& a);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator - (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e, const Scalar& s);
CV_EXPORTS MatExpr operator - (const Scalar& s, const MatExpr& e);
CV_EXPORTS MatExpr operator - (const MatExpr& e1, const MatExpr& e2);

CV_EXPORTS MatExpr operator - (const Mat& m);
CV_EXPORTS MatExpr operator - (const MatExpr& e);

CV_EXPORTS MatExpr operator * (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator * (const Mat& a, double s);
CV_EXPORTS MatExpr operator * (double s, const Mat& a);
CV_EXPORTS MatExpr operator * (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator * (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator * (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator * (const MatExpr& e1, const MatExpr& e2);
    
CV_EXPORTS MatExpr operator / (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator / (const Mat& a, double s);
CV_EXPORTS MatExpr operator / (double s, const Mat& a);
CV_EXPORTS MatExpr operator / (const MatExpr& e, const Mat& m);
CV_EXPORTS MatExpr operator / (const Mat& m, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e, double s);
CV_EXPORTS MatExpr operator / (double s, const MatExpr& e);
CV_EXPORTS MatExpr operator / (const MatExpr& e1, const MatExpr& e2);    

CV_EXPORTS MatExpr operator < (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator < (const Mat& a, double s);
CV_EXPORTS MatExpr operator < (double s, const Mat& a);

CV_EXPORTS MatExpr operator <= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator <= (const Mat& a, double s);
CV_EXPORTS MatExpr operator <= (double s, const Mat& a);

CV_EXPORTS MatExpr operator == (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator == (const Mat& a, double s);
CV_EXPORTS MatExpr operator == (double s, const Mat& a);

CV_EXPORTS MatExpr operator != (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator != (const Mat& a, double s);
CV_EXPORTS MatExpr operator != (double s, const Mat& a);

CV_EXPORTS MatExpr operator >= (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator >= (const Mat& a, double s);
CV_EXPORTS MatExpr operator >= (double s, const Mat& a);

CV_EXPORTS MatExpr operator > (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator > (const Mat& a, double s);
CV_EXPORTS MatExpr operator > (double s, const Mat& a);    
    
CV_EXPORTS MatExpr min(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr min(const Mat& a, double s);
CV_EXPORTS MatExpr min(double s, const Mat& a);

CV_EXPORTS MatExpr max(const Mat& a, const Mat& b);
CV_EXPORTS MatExpr max(const Mat& a, double s);
CV_EXPORTS MatExpr max(double s, const Mat& a);

template<typename _Tp> static inline MatExpr min(const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    return cv::min((const Mat&)a, (const Mat&)b);
}

template<typename _Tp> static inline MatExpr min(const Mat_<_Tp>& a, double s)
{
    return cv::min((const Mat&)a, s);
}

template<typename _Tp> static inline MatExpr min(double s, const Mat_<_Tp>& a)
{
    return cv::min((const Mat&)a, s);
}    

template<typename _Tp> static inline MatExpr max(const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    return cv::max((const Mat&)a, (const Mat&)b);
}

template<typename _Tp> static inline MatExpr max(const Mat_<_Tp>& a, double s)
{
    return cv::max((const Mat&)a, s);
}

template<typename _Tp> static inline MatExpr max(double s, const Mat_<_Tp>& a)
{
    return cv::max((const Mat&)a, s);
}        
    
CV_EXPORTS MatExpr operator & (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator & (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator & (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator | (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator | (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator | (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator ^ (const Mat& a, const Mat& b);
CV_EXPORTS MatExpr operator ^ (const Mat& a, const Scalar& s);
CV_EXPORTS MatExpr operator ^ (const Scalar& s, const Mat& a);

CV_EXPORTS MatExpr operator ~(const Mat& m);
    
CV_EXPORTS MatExpr abs(const Mat& m);
CV_EXPORTS MatExpr abs(const MatExpr& e);
    
template<typename _Tp> static inline MatExpr abs(const Mat_<_Tp>& m)
{
    return cv::abs((const Mat&)m);
}

////////////////////////////// Augmenting algebraic operations //////////////////////////////////
    
inline Mat& Mat::operator = (const MatExpr& e)
{
    e.op->assign(e, *this);
    return *this;
}    

template<typename _Tp> Mat_<_Tp>& Mat_<_Tp>::operator = (const MatExpr& e)
{
    e.op->assign(e, *this, DataType<_Tp>::type);
    return *this;
}

static inline Mat& operator += (const Mat& a, const Mat& b)
{
    add(a, b, (Mat&)a);
    return (Mat&)a;
}

static inline Mat& operator += (const Mat& a, const Scalar& s)
{
    add(a, s, (Mat&)a);
    return (Mat&)a;
}    

template<typename _Tp> static inline
Mat_<_Tp>& operator += (const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    add(a, b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator += (const Mat_<_Tp>& a, const Scalar& s)
{
    add(a, s, (Mat&)a);
    return (Mat_<_Tp>&)a;
}    

static inline Mat& operator += (const Mat& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, (Mat&)a); 
    return (Mat&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator += (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignAdd(b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}
    
static inline Mat& operator -= (const Mat& a, const Mat& b)
{
    subtract(a, b, (Mat&)a);
    return (Mat&)a;
}

static inline Mat& operator -= (const Mat& a, const Scalar& s)
{
    subtract(a, s, (Mat&)a);
    return (Mat&)a;
}    

template<typename _Tp> static inline
Mat_<_Tp>& operator -= (const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    subtract(a, b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator -= (const Mat_<_Tp>& a, const Scalar& s)
{
    subtract(a, s, (Mat&)a);
    return (Mat_<_Tp>&)a;
}    

static inline Mat& operator -= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, (Mat&)a); 
    return (Mat&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator -= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignSubtract(b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}    

static inline Mat& operator *= (const Mat& a, const Mat& b)
{
    gemm(a, b, 1, Mat(), 0, (Mat&)a, 0);
    return (Mat&)a;
}

static inline Mat& operator *= (const Mat& a, double s)
{
    a.convertTo((Mat&)a, -1, s);
    return (Mat&)a;
}    

template<typename _Tp> static inline
Mat_<_Tp>& operator *= (const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    gemm(a, b, 1, Mat(), 0, (Mat&)a, 0);
    return (Mat_<_Tp>&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator *= (const Mat_<_Tp>& a, double s)
{
    a.convertTo((Mat&)a, -1, s);
    return (Mat_<_Tp>&)a;
}    

static inline Mat& operator *= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, (Mat&)a); 
    return (Mat&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator *= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignMultiply(b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}    
    
static inline Mat& operator /= (const Mat& a, const Mat& b)
{
    divide(a, b, (Mat&)a);
    return (Mat&)a;
}

static inline Mat& operator /= (const Mat& a, double s)
{
    a.convertTo((Mat&)a, -1, 1./s);
    return (Mat&)a;
}    

template<typename _Tp> static inline
Mat_<_Tp>& operator /= (const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    divide(a, b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator /= (const Mat_<_Tp>& a, double s)
{
    a.convertTo((Mat&)a, -1, 1./s);
    return (Mat_<_Tp>&)a;
}    

static inline Mat& operator /= (const Mat& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, (Mat&)a); 
    return (Mat&)a;
}

template<typename _Tp> static inline
Mat_<_Tp>& operator /= (const Mat_<_Tp>& a, const MatExpr& b)
{
    b.op->augAssignDivide(b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}

////////////////////////////// Logical operations ///////////////////////////////

static inline Mat& operator &= (const Mat& a, const Mat& b)
{
    bitwise_and(a, b, (Mat&)a);
    return (Mat&)a;
}

static inline Mat& operator &= (const Mat& a, const Scalar& s)
{
    bitwise_and(a, s, (Mat&)a);
    return (Mat&)a;
}    

template<typename _Tp> static inline Mat_<_Tp>&
operator &= (const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    bitwise_and(a, b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}    

template<typename _Tp> static inline Mat_<_Tp>&
operator &= (const Mat_<_Tp>& a, const Scalar& s)
{
    bitwise_and(a, s, (Mat&)a);
    return (Mat_<_Tp>&)a;
}        
    
static inline Mat& operator |= (const Mat& a, const Mat& b)
{
    bitwise_or(a, b, (Mat&)a);
    return (Mat&)a;
}

static inline Mat& operator |= (const Mat& a, const Scalar& s)
{
    bitwise_or(a, s, (Mat&)a);
    return (Mat&)a;
}    

template<typename _Tp> static inline Mat_<_Tp>&
operator |= (const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    bitwise_or(a, b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}    

template<typename _Tp> static inline Mat_<_Tp>&
operator |= (const Mat_<_Tp>& a, const Scalar& s)
{
    bitwise_or(a, s, (Mat&)a);
    return (Mat_<_Tp>&)a;
}        
    
static inline Mat& operator ^= (const Mat& a, const Mat& b)
{
    bitwise_xor(a, b, (Mat&)a);
    return (Mat&)a;
}

static inline Mat& operator ^= (const Mat& a, const Scalar& s)
{
    bitwise_xor(a, s, (Mat&)a);
    return (Mat&)a;
}    

template<typename _Tp> static inline Mat_<_Tp>&
operator ^= (const Mat_<_Tp>& a, const Mat_<_Tp>& b)
{
    bitwise_xor(a, b, (Mat&)a);
    return (Mat_<_Tp>&)a;
}    

template<typename _Tp> static inline Mat_<_Tp>&
operator ^= (const Mat_<_Tp>& a, const Scalar& s)
{
    bitwise_xor(a, s, (Mat&)a);
    return (Mat_<_Tp>&)a;
}        

/////////////////////////////// Miscellaneous operations //////////////////////////////

static inline void merge(const vector<Mat>& mv, Mat& dst)
{ merge(&mv[0], mv.size(), dst); }
    
template<typename _Tp> void merge(const Mat_<_Tp>* mvbegin, size_t count, Mat& dst)
{ merge( (const Mat*)mvbegin, count, dst ); }

static inline void split(const Mat& m, vector<Mat>& mv)
{
    mv.resize(m.channels());
    if(m.channels() > 0)
        split(m, &mv[0]);
}    
    
template<typename _Tp> void split(const Mat& src, vector<Mat_<_Tp> >& mv)
{ split(src, (vector<Mat>&)mv ); }

static inline void mixChannels(const vector<Mat>& src, vector<Mat>& dst,
                               const int* fromTo, int npairs)
{
    mixChannels(&src[0], (int)src.size(), &dst[0], (int)dst.size(), fromTo, npairs);
}
 
//////////////////////////////////////////////////////////////
    
template<typename _Tp> inline MatExpr Mat_<_Tp>::zeros(int rows, int cols)
{
    return Mat::zeros(rows, cols, DataType<_Tp>::type);
}
    
template<typename _Tp> inline MatExpr Mat_<_Tp>::zeros(Size sz)
{
    return Mat::zeros(sz, DataType<_Tp>::type);
}    
    
template<typename _Tp> inline MatExpr Mat_<_Tp>::ones(int rows, int cols)
{
    return Mat::ones(rows, cols, DataType<_Tp>::type);
}

template<typename _Tp> inline MatExpr Mat_<_Tp>::ones(Size sz)
{
    return Mat::ones(sz, DataType<_Tp>::type);
}    
    
template<typename _Tp> inline MatExpr Mat_<_Tp>::eye(int rows, int cols)
{
    return Mat::eye(rows, cols, DataType<_Tp>::type);
}

template<typename _Tp> inline MatExpr Mat_<_Tp>::eye(Size sz)
{
    return Mat::eye(sz, DataType<_Tp>::type);
}    
    
//////////// Iterators & Comma initializers //////////////////

template<typename _Tp> inline MatConstIterator_<_Tp>::MatConstIterator_()
    : m(0), ptr(0), sliceEnd(0) {}

template<typename _Tp> inline MatConstIterator_<_Tp>::MatConstIterator_(const Mat_<_Tp>* _m) : m(_m)
{
    if( !_m )
        ptr = sliceEnd = 0;
    else
    {
        ptr = (_Tp*)_m->data;
        sliceEnd = ptr + (_m->isContinuous() ? _m->rows*_m->cols : _m->cols);
    }
}

template<typename _Tp> inline MatConstIterator_<_Tp>::
    MatConstIterator_(const Mat_<_Tp>* _m, int _row, int _col) : m(_m)
{
    if( !_m )
        ptr = sliceEnd = 0;
    else
    {
        CV_DbgAssert( (unsigned)_row < _m->rows && (unsigned)_col < _m->cols );
        ptr = (_Tp*)(_m->data + _m->step*_row);
        sliceEnd = _m->isContinuous() ? (_Tp*)_m->data + _m->rows*_m->cols : ptr + _m->cols;
        ptr += _col;
    }
}

template<typename _Tp> inline MatConstIterator_<_Tp>::
    MatConstIterator_(const Mat_<_Tp>* _m, Point _pt) : m(_m)
{
    if( !_m )
        ptr = sliceEnd = 0;
    else
    {
        CV_DbgAssert( (unsigned)_pt.y < (unsigned)_m->rows && (unsigned)_pt.x < (unsigned)_m->cols );
        ptr = (_Tp*)(_m->data + _m->step*_pt.y);
        sliceEnd = _m->isContinuous() ? (_Tp*)_m->data + _m->rows*_m->cols : ptr + _m->cols;
        ptr += _pt.x;
    }
}

template<typename _Tp> inline MatConstIterator_<_Tp>::
    MatConstIterator_(const MatConstIterator_& it)
    : m(it.m), ptr(it.ptr), sliceEnd(it.sliceEnd) {}

template<typename _Tp> inline MatConstIterator_<_Tp>&
    MatConstIterator_<_Tp>::operator = (const MatConstIterator_& it )
{
    m = it.m; ptr = it.ptr; sliceEnd = it.sliceEnd;
    return *this;
}

template<typename _Tp> inline _Tp MatConstIterator_<_Tp>::operator *() const { return *ptr; }

template<typename _Tp> inline MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator += (int ofs)
{
    if( !m || ofs == 0 )
        return *this;
    ptr += ofs;
    if( m->isContinuous() )
    {
        if( ptr > sliceEnd )
            ptr = sliceEnd;
        else if( ptr < (_Tp*)m->data )
            ptr = (_Tp*)m->data;
    }
    else if( ptr >= sliceEnd || ptr < sliceEnd - m->cols )
    {
        ptr -= ofs;
        Point pt = pos();
        int cols = m->cols;
        ofs += pt.y*cols + pt.x;
        if( ofs >= cols*m->rows )
        {
            ptr = sliceEnd = (_Tp*)(m->data + m->step*(m->rows-1)) + cols; 
            return *this; 
        }
        else if( ofs < 0 )
            ofs = 0;
        pt.y = ofs/cols;
        pt.x = ofs - pt.y*cols;
        ptr = (_Tp*)(m->data + m->step*pt.y);
        sliceEnd = ptr + cols;
        ptr += pt.x;
    }
    return *this;
}

template<typename _Tp> inline MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator -= (int ofs)
{ return (*this += -ofs); }

template<typename _Tp> inline MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator --()
{ return (*this += -1); }

template<typename _Tp> inline MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator --(int)
{
    MatConstIterator_ b = *this;
    *this += -1;
    return b;
}

template<typename _Tp> inline MatConstIterator_<_Tp>& MatConstIterator_<_Tp>::operator ++()
{
    if( m && ++ptr >= sliceEnd )
    {
        --ptr;
        *this += 1;
    }
    return *this;
}

template<typename _Tp> inline MatConstIterator_<_Tp> MatConstIterator_<_Tp>::operator ++(int)
{
    MatConstIterator_ b = *this;
    if( m && ++ptr >= sliceEnd )
    {
        --ptr;
        *this += 1;
    }
    return b;
}

template<typename _Tp> inline Point MatConstIterator_<_Tp>::pos() const
{
    if( !m )
        return Point();
    if( m->isContinuous() )
    {
        ptrdiff_t ofs = ptr - (_Tp*)m->data;
        int y = (int)(ofs / m->cols), x = (int)(ofs - (ptrdiff_t)y*m->cols);
        return Point(x, y);
    }
    else
    {
        ptrdiff_t ofs = (uchar*)ptr - m->data;
        int y = (int)(ofs / m->step), x = (int)((ofs - y*m->step)/sizeof(_Tp));
        return Point(x, y);
    }
}

template<typename _Tp> inline MatIterator_<_Tp>::MatIterator_() : MatConstIterator_<_Tp>() {}

template<typename _Tp> inline MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m)
    : MatConstIterator_<_Tp>(_m) {}

template<typename _Tp> inline MatIterator_<_Tp>::MatIterator_(Mat_<_Tp>* _m, int _row, int _col)
    : MatConstIterator_<_Tp>(_m, _row, _col) {}

template<typename _Tp> inline MatIterator_<_Tp>::MatIterator_(const Mat_<_Tp>* _m, Point _pt)
    : MatConstIterator_<_Tp>(_m, _pt) {}

template<typename _Tp> inline MatIterator_<_Tp>::MatIterator_(const MatIterator_& it)
    : MatConstIterator_<_Tp>(it) {}

template<typename _Tp> inline MatIterator_<_Tp>& MatIterator_<_Tp>::operator = (const MatIterator_<_Tp>& it )
{
    this->m = it.m; this->ptr = it.ptr; this->sliceEnd = it.sliceEnd;
    return *this;
}

template<typename _Tp> inline _Tp& MatIterator_<_Tp>::operator *() const { return *(this->ptr); }

template<typename _Tp> inline MatIterator_<_Tp>& MatIterator_<_Tp>::operator += (int ofs)
{
    MatConstIterator_<_Tp>::operator += (ofs);
    return *this;
}

template<typename _Tp> inline MatIterator_<_Tp>& MatIterator_<_Tp>::operator -= (int ofs)
{
    MatConstIterator_<_Tp>::operator += (-ofs);
    return *this;
}

template<typename _Tp> inline MatIterator_<_Tp>& MatIterator_<_Tp>::operator --()
{
    MatConstIterator_<_Tp>::operator += (-1);
    return *this;
}

template<typename _Tp> inline MatIterator_<_Tp> MatIterator_<_Tp>::operator --(int)
{
    MatIterator_ b = *this;
    MatConstIterator_<_Tp>::operator += (-1);
    return b;
}

template<typename _Tp> inline MatIterator_<_Tp>& MatIterator_<_Tp>::operator ++()
{
    if( this->m && ++this->ptr >= this->sliceEnd )
    {
        --this->ptr;
        MatConstIterator_<_Tp>::operator += (1);
    }
    return *this;
}

template<typename _Tp> inline MatIterator_<_Tp> MatIterator_<_Tp>::operator ++(int)
{
    MatIterator_ b = *this;
    if( this->m && ++this->ptr >= this->sliceEnd )
    {
        --this->ptr;
        MatConstIterator_<_Tp>::operator += (1);
    }
    return b;
}

template<typename _Tp> static inline bool
operator == (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{ return a.m == b.m && a.ptr == b.ptr; }

template<typename _Tp> static inline bool
operator != (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{ return !(a == b); }

template<typename _Tp> static inline bool
operator < (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{ return a.ptr < b.ptr; }

template<typename _Tp> static inline bool
operator > (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{ return a.ptr > b.ptr; }

template<typename _Tp> static inline bool
operator <= (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{ return a.ptr <= b.ptr; }

template<typename _Tp> static inline bool
operator >= (const MatConstIterator_<_Tp>& a, const MatConstIterator_<_Tp>& b)
{ return a.ptr >= b.ptr; }

template<typename _Tp> static inline int
operator - (const MatConstIterator_<_Tp>& b, const MatConstIterator_<_Tp>& a)
{
    if( a.m != b.m )
        return INT_MAX;
    if( a.sliceEnd == b.sliceEnd )
        return b.ptr - a.ptr;
    {
        Point ap = a.pos(), bp = b.pos();
        if( bp.y > ap.y )
            return (bp.y - ap.y - 1)*a.m->cols + (a.m->cols - ap.x) + bp.x;
        if( bp.y < ap.y )
            return -((ap.y - bp.y - 1)*a.m->cols + (a.m->cols - bp.x) + ap.x);
        return bp.x - ap.x;
    }
}

template<typename _Tp> static inline MatConstIterator_<_Tp>
operator + (const MatConstIterator_<_Tp>& a, int ofs)
{ MatConstIterator_<_Tp> b = a; return b += ofs; }

template<typename _Tp> static inline MatConstIterator_<_Tp>
operator + (int ofs, const MatConstIterator_<_Tp>& a)
{ MatConstIterator_<_Tp> b = a; return b += ofs; }

template<typename _Tp> static inline MatConstIterator_<_Tp>
operator - (const MatConstIterator_<_Tp>& a, int ofs)
{ MatConstIterator_<_Tp> b = a; return b += -ofs; }

template<typename _Tp> inline _Tp MatConstIterator_<_Tp>::operator [](int i) const
{ return *(*this + i); }

template<typename _Tp> static inline MatIterator_<_Tp>
operator + (const MatIterator_<_Tp>& a, int ofs)
{ MatIterator_<_Tp> b = a; return b += ofs; }

template<typename _Tp> static inline MatIterator_<_Tp>
operator + (int ofs, const MatIterator_<_Tp>& a)
{ MatIterator_<_Tp> b = a; return b += ofs; }

template<typename _Tp> static inline MatIterator_<_Tp>
operator - (const MatIterator_<_Tp>& a, int ofs)
{ MatIterator_<_Tp> b = a; return b += -ofs; }

template<typename _Tp> inline _Tp& MatIterator_<_Tp>::operator [](int i) const
{ return *(*this + i); }

template<typename _Tp> inline MatConstIterator_<_Tp> Mat_<_Tp>::begin() const
{ return Mat::begin<_Tp>(); }

template<typename _Tp> inline MatConstIterator_<_Tp> Mat_<_Tp>::end() const
{ return Mat::end<_Tp>(); }

template<typename _Tp> inline MatIterator_<_Tp> Mat_<_Tp>::begin()
{ return Mat::begin<_Tp>(); }

template<typename _Tp> inline MatIterator_<_Tp> Mat_<_Tp>::end()
{ return Mat::end<_Tp>(); }

template<typename _Tp> inline MatCommaInitializer_<_Tp>::MatCommaInitializer_(Mat_<_Tp>* _m) : it(_m) {}

template<typename _Tp> template<typename T2> inline MatCommaInitializer_<_Tp>&
MatCommaInitializer_<_Tp>::operator , (T2 v)
{
    CV_DbgAssert( this->it < this->it.m->end() );
    *this->it = _Tp(v); ++this->it;
    return *this;
}

template<typename _Tp> inline Mat_<_Tp> MatCommaInitializer_<_Tp>::operator *() const
{
    CV_DbgAssert( this->it == this->it.m->end() );
    return *this->it.m;
}

template<typename _Tp> inline MatCommaInitializer_<_Tp>::operator Mat_<_Tp>() const
{
    CV_DbgAssert( this->it == this->it.m->end() );
    return *this->it.m;
}    
    
template<typename _Tp> inline void
MatCommaInitializer_<_Tp>::assignTo(Mat& m, int type) const
{
    Mat_<_Tp>(*this).assignTo(m, type);
}

template<typename _Tp, typename T2> static inline MatCommaInitializer_<_Tp>
operator << (const Mat_<_Tp>& m, T2 val)
{
    MatCommaInitializer_<_Tp> commaInitializer((Mat_<_Tp>*)&m);
    return (commaInitializer, val);
}

//////////////////////////////// MatND ////////////////////////////////

inline MatND::MatND()
 : flags(MAGIC_VAL), dims(0), refcount(0), data(0), datastart(0), dataend(0)
{
}

inline MatND::MatND(int _dims, const int* _sizes, int _type)
 : flags(MAGIC_VAL), dims(0), refcount(0), data(0), datastart(0), dataend(0)
{
    create(_dims, _sizes, _type);
}

inline MatND::MatND(int _dims, const int* _sizes, int _type, const Scalar& _s)
 : flags(MAGIC_VAL), dims(0), refcount(0), data(0), datastart(0), dataend(0)
{
    create(_dims, _sizes, _type);
    *this = _s;
}

inline MatND::MatND(const MatND& m)
 : flags(m.flags), dims(m.dims), refcount(m.refcount),
 data(m.data), datastart(m.datastart), dataend(m.dataend)
{
    int i, d = dims;
    for( i = 0; i < d; i++ )
    {
        size[i] = m.size[i];
        step[i] = m.step[i];
    }
    if( refcount )
        CV_XADD(refcount, 1);
}

inline MatND::MatND(const Mat& m)
 : flags(m.flags), dims(2), refcount(m.refcount),
   data(m.data), datastart(m.datastart), dataend(m.dataend)
{
    size[0] = m.rows; size[1] = m.cols;
    step[0] = m.step; step[1] = m.elemSize();
    if( refcount )
        CV_XADD(refcount, 1);
}

static inline MatND cvarrToMatND(const CvArr* arr, bool copyData=false, int coiMode=0)
{
    if( CV_IS_MAT(arr) || CV_IS_IMAGE(arr))
        return MatND(cvarrToMat(arr, copyData, true, coiMode));
    else if( CV_IS_MATND(arr) )
        return MatND((const CvMatND*)arr, copyData);
    return MatND();
}

inline MatND::MatND(const CvMatND* m, bool copyData)
  : flags(MAGIC_VAL|(m->type & (CV_MAT_TYPE_MASK|CV_MAT_CONT_FLAG))),
  dims(m->dims), refcount(0), data(m->data.ptr)
{
    int i, d = dims;
    for( i = 0; i < d; i++ )
    {
        size[i] = m->dim[i].size;
        step[i] = m->dim[i].step;
    }
    datastart = data;
    dataend = datastart + size[0]*step[0];
    if( copyData )
    {
        MatND temp(*this);
        temp.copyTo(*this);
    }
}

inline MatND::~MatND() { release(); }

inline MatND& MatND::operator = (const MatND& m)
{
    if( this != &m )
    {
        if( m.refcount )
            CV_XADD(m.refcount, 1);
        release();
        flags = m.flags;
        dims = m.dims;
        data = m.data;
        datastart = m.datastart;
        dataend = m.dataend;
        refcount = m.refcount;
        int i, d = dims;
        for( i = 0; i < d; i++ )
        {
            size[i] = m.size[i];
            step[i] = m.step[i];
        }
    }
    return *this;
}

inline MatND MatND::clone() const
{
    MatND temp;
    this->copyTo(temp);
    return temp;
}

inline MatND MatND::operator()(const Range* ranges) const
{
    return MatND(*this, ranges);
}

inline void MatND::assignTo( MatND& m, int type ) const
{
    if( type < 0 )
        m = *this;
    else
        convertTo(m, type);
}

inline void MatND::addref()
{
    if( refcount ) CV_XADD(refcount, 1);
}

inline void MatND::release()
{
    if( refcount && CV_XADD(refcount, -1) == 1 )
        fastFree(datastart);
    dims = 0;
    data = datastart = dataend = 0;
    refcount = 0;
}

inline bool MatND::isContinuous() const { return (flags & CONTINUOUS_FLAG) != 0; }
inline size_t MatND::elemSize() const { return getElemSize(flags); }
inline size_t MatND::elemSize1() const { return CV_ELEM_SIZE1(flags); }
inline int MatND::type() const { return CV_MAT_TYPE(flags); }
inline int MatND::depth() const { return CV_MAT_DEPTH(flags); }
inline int MatND::channels() const { return CV_MAT_CN(flags); }

inline size_t MatND::step1(int i) const
{ CV_DbgAssert((unsigned)i < (unsigned)dims); return step[i]/elemSize1(); }

inline uchar* MatND::ptr(int i0)
{
    CV_DbgAssert( dims == 1 && data &&
        (unsigned)i0 < (unsigned)size[0] );
    return data + i0*step[0];
}

inline const uchar* MatND::ptr(int i0) const
{
    CV_DbgAssert( dims == 1 && data &&
        (unsigned)i0 < (unsigned)size[0] );
    return data + i0*step[0];
}

inline uchar* MatND::ptr(int i0, int i1)
{
    CV_DbgAssert( dims == 2 && data &&
        (unsigned)i0 < (unsigned)size[0] &&
        (unsigned)i1 < (unsigned)size[1] );
    return data + i0*step[0] + i1*step[1];
}

inline const uchar* MatND::ptr(int i0, int i1) const
{
    CV_DbgAssert( dims == 2 && data &&
        (unsigned)i0 < (unsigned)size[0] &&
        (unsigned)i1 < (unsigned)size[1] );
    return data + i0*step[0] + i1*step[1];
}

inline uchar* MatND::ptr(int i0, int i1, int i2)
{
    CV_DbgAssert( dims == 3 && data &&
        (unsigned)i0 < (unsigned)size[0] &&
        (unsigned)i1 < (unsigned)size[1] &&
        (unsigned)i2 < (unsigned)size[2] );
    return data + i0*step[0] + i1*step[1] + i2*step[2];
}

inline const uchar* MatND::ptr(int i0, int i1, int i2) const
{
    CV_DbgAssert( dims == 3 && data &&
        (unsigned)i0 < (unsigned)size[0] &&
        (unsigned)i1 < (unsigned)size[1] &&
        (unsigned)i2 < (unsigned)size[2] );
    return data + i0*step[0] + i1*step[1] + i2*step[2];
}

inline uchar* MatND::ptr(const int* idx)
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( data );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size[i] );
        p += idx[i]*step[i];
    }
    return p;
}

inline const uchar* MatND::ptr(const int* idx) const
{
    int i, d = dims;
    uchar* p = data;
    CV_DbgAssert( data );
    for( i = 0; i < d; i++ )
    {
        CV_DbgAssert( (unsigned)idx[i] < (unsigned)size[i] );
        p += idx[i]*step[i];
    }
    return p;
}

template<typename _Tp> inline _Tp& MatND::at(int i0)
{ return *(_Tp*)ptr(i0); }
template<typename _Tp> inline const _Tp& MatND::at(int i0) const
{ return *(const _Tp*)ptr(i0); }
template<typename _Tp> inline _Tp& MatND::at(int i0, int i1)
{ return *(_Tp*)ptr(i0, i1); }
template<typename _Tp> inline const _Tp& MatND::at(int i0, int i1) const
{ return *(const _Tp*)ptr(i0, i1); }
template<typename _Tp> inline _Tp& MatND::at(int i0, int i1, int i2)
{ return *(_Tp*)ptr(i0, i1, i2); }
template<typename _Tp> inline const _Tp& MatND::at(int i0, int i1, int i2) const
{ return *(const _Tp*)ptr(i0, i1, i2); }
template<typename _Tp> inline _Tp& MatND::at(const int* idx)
{ return *(_Tp*)ptr(idx); }
template<typename _Tp> inline const _Tp& MatND::at(const int* idx) const
{ return *(const _Tp*)ptr(idx); }

inline NAryMatNDIterator::NAryMatNDIterator()
{
}

inline void subtract(const MatND& a, const Scalar& s, MatND& c, const MatND& mask=MatND())
{
    add(a, -s, c, mask);
}


template<typename _Tp> inline MatND_<_Tp>::MatND_()
{
    flags = MAGIC_VAL | DataType<_Tp>::type;
}

template<typename _Tp> inline MatND_<_Tp>::MatND_(int _dims, const int* _sizes)
: MatND(_dims, _sizes, DataType<_Tp>::type)
{
}

template<typename _Tp> inline MatND_<_Tp>::MatND_(int _dims, const int* _sizes, const _Tp& _s)
: MatND(_dims, _sizes, DataType<_Tp>::type, Scalar(_s))
{
}

template<typename _Tp> inline MatND_<_Tp>::MatND_(const MatND& m)
{
    if( m.type() == DataType<_Tp>::type )
        *this = (const MatND_<_Tp>&)m;
    else
        m.convertTo(this, DataType<_Tp>::type);
}

template<typename _Tp> inline MatND_<_Tp>::MatND_(const MatND_<_Tp>& m) : MatND(m)
{
}

template<typename _Tp> inline MatND_<_Tp>::MatND_(const MatND_<_Tp>& m, const Range* ranges)
: MatND(m, ranges)
{
}

template<typename _Tp> inline MatND_<_Tp>::MatND_(const CvMatND* m, bool copyData)
{
    *this = MatND(m, copyData || CV_MAT_TYPE(m->type) != DataType<_Tp>::type);
}

template<typename _Tp> inline MatND_<_Tp>& MatND_<_Tp>::operator = (const MatND& m)
{
    if( DataType<_Tp>::type == m.type() )
    {
        Mat::operator = (m);
        return *this;
    }
    if( DataType<_Tp>::depth == m.depth() )
    {
        return (*this = m.reshape(DataType<_Tp>::channels));
    }
    CV_DbgAssert(DataType<_Tp>::channels == m.channels());
    m.convertTo(*this, DataType<_Tp>::type);
    return *this;
}

template<typename _Tp> inline MatND_<_Tp>& MatND_<_Tp>::operator = (const MatND_<_Tp>& m)
{
    return ((MatND&)*this = m);
}

template<typename _Tp> inline MatND_<_Tp>& MatND_<_Tp>::operator = (const _Tp& s)
{
    return (MatND&)*this = Scalar(s);
}

template<typename _Tp> inline void MatND_<_Tp>::create(int _dims, const int* _sizes)
{
    MatND::create(_dims, _sizes, DataType<_Tp>::type);
}

template<typename _Tp> template<typename _Tp2> inline MatND_<_Tp>::operator MatND_<_Tp2>() const
{
    return MatND_<_Tp2>((const MatND&)*this);
}

template<typename _Tp> inline MatND_<_Tp> MatND_<_Tp>::clone() const
{
    MatND_<_Tp> temp;
    this->copyTo(temp);
    return temp;
}

template<typename _Tp> inline MatND_<_Tp>
MatND_<_Tp>::operator()(const Range* ranges) const
{ return MatND_<_Tp>(*this, ranges); }

template<typename _Tp> inline size_t MatND_<_Tp>::elemSize() const
{ return CV_ELEM_SIZE(DataType<_Tp>::type); }

template<typename _Tp> inline size_t MatND_<_Tp>::elemSize1() const
{ return CV_ELEM_SIZE1(DataType<_Tp>::type); }

template<typename _Tp> inline int MatND_<_Tp>::type() const
{ return DataType<_Tp>::type; }

template<typename _Tp> inline int MatND_<_Tp>::depth() const
{ return DataType<_Tp>::depth; }

template<typename _Tp> inline int MatND_<_Tp>::channels() const
{ return DataType<_Tp>::channels; }

template<typename _Tp> inline size_t MatND_<_Tp>::stepT(int i) const
{
    CV_DbgAssert( (unsigned)i < (unsigned)dims );
    return step[i]/elemSize();
}

template<typename _Tp> inline size_t MatND_<_Tp>::step1(int i) const
{
    CV_DbgAssert( (unsigned)i < (unsigned)dims );
    return step[i]/elemSize1();
}

template<typename _Tp> inline _Tp& MatND_<_Tp>::operator ()(const int* idx)
{
    uchar* ptr = data;
    int i, d = dims;
    for( i = 0; i < d; i++ )
    {
        int ii = idx[i];
        CV_DbgAssert( (unsigned)ii < (unsigned)size[i] );
        ptr += ii*step[i];
    }
    return *(_Tp*)ptr;
}

template<typename _Tp> inline const _Tp& MatND_<_Tp>::operator ()(const int* idx) const
{
    const uchar* ptr = data;
    int i, d = dims;
    for( i = 0; i < d; i++ )
    {
        int ii = idx[i];
        CV_DbgAssert( (unsigned)ii < (unsigned)size[i] );
        ptr += ii*step[i];
    }
    return *(const _Tp*)ptr;
}

template<typename _Tp> inline _Tp& MatND_<_Tp>::operator ()(int i0)
{
    CV_DbgAssert( dims == 1 &&
                 (unsigned)i0 < (unsigned)size[0] );
    
    return *(_Tp*)(data + i0*step[0]);
}

template<typename _Tp> inline const _Tp& MatND_<_Tp>::operator ()(int i0) const
{
    CV_DbgAssert( dims == 1 &&
                 (unsigned)i0 < (unsigned)size[0] );
    
    return *(const _Tp*)(data + i0*step[0]);
}
    
    
template<typename _Tp> inline _Tp& MatND_<_Tp>::operator ()(int i0, int i1)
{
    CV_DbgAssert( dims == 2 &&
                 (unsigned)i0 < (unsigned)size[0] &&
                 (unsigned)i1 < (unsigned)size[1] );
    
    return *(_Tp*)(data + i0*step[0] + i1*step[1]);
}

template<typename _Tp> inline const _Tp& MatND_<_Tp>::operator ()(int i0, int i1) const
{
    CV_DbgAssert( dims == 2 &&
                 (unsigned)i0 < (unsigned)size[0] &&
                 (unsigned)i1 < (unsigned)size[1] );
    
    return *(const _Tp*)(data + i0*step[0] + i1*step[1]);
}
    
    
template<typename _Tp> inline _Tp& MatND_<_Tp>::operator ()(int i0, int i1, int i2)
{
    CV_DbgAssert( dims == 3 &&
        (unsigned)i0 < (unsigned)size[0] &&
        (unsigned)i1 < (unsigned)size[1] &&
        (unsigned)i2 < (unsigned)size[2] );

    return *(_Tp*)(data + i0*step[0] + i1*step[1] + i2*step[2]);
}

template<typename _Tp> inline const _Tp& MatND_<_Tp>::operator ()(int i0, int i1, int i2) const
{
    CV_DbgAssert( dims == 3 &&
        (unsigned)i0 < (unsigned)size[0] &&
        (unsigned)i1 < (unsigned)size[1] &&
        (unsigned)i2 < (unsigned)size[2] );

    return *(const _Tp*)(data + i0*step[0] + i1*step[1] + i2*step[2]);
}

    
static inline void merge(const vector<MatND>& mv, MatND& dst)
{
    merge(&mv[0], mv.size(), dst);
}
    
static inline void split(const MatND& m, vector<MatND>& mv)
{
    mv.resize(m.channels());
    if(m.channels() > 0)
        split(m, &mv[0]);
}

static inline void mixChannels(const vector<MatND>& src, vector<MatND>& dst,
                               const int* fromTo, int npairs)
{
    mixChannels(&src[0], (int)src.size(), &dst[0], (int)dst.size(), fromTo, npairs);
}   

//////////////////////////////// SparseMat ////////////////////////////////

inline SparseMat::SparseMat()
: flags(MAGIC_VAL), hdr(0)
{
}

inline SparseMat::SparseMat(int _dims, const int* _sizes, int _type)
: flags(MAGIC_VAL), hdr(0)
{
    create(_dims, _sizes, _type);
}

inline SparseMat::SparseMat(const SparseMat& m)
: flags(m.flags), hdr(m.hdr)
{
    addref();
}

inline SparseMat::~SparseMat()
{
    release();
}

inline SparseMat& SparseMat::operator = (const SparseMat& m)
{
    if( this != &m )
    {
        if( m.hdr )
            CV_XADD(&m.hdr->refcount, 1);
        release();
        flags = m.flags;
        hdr = m.hdr;
    }
    return *this;
}

inline SparseMat& SparseMat::operator = (const Mat& m)
{ return (*this = SparseMat(m)); }

inline SparseMat& SparseMat::operator = (const MatND& m)
{ return (*this = SparseMat(m)); }

inline SparseMat SparseMat::clone() const
{
    SparseMat temp;
    this->copyTo(temp);
    return temp;
}


inline void SparseMat::assignTo( SparseMat& m, int type ) const
{
    if( type < 0 )
        m = *this;
    else
        convertTo(m, type);
}

inline void SparseMat::addref()
{ if( hdr ) CV_XADD(&hdr->refcount, 1); }

inline void SparseMat::release()
{
    if( hdr && CV_XADD(&hdr->refcount, -1) == 1 )
        delete hdr;
    hdr = 0;
}

inline size_t SparseMat::elemSize() const
{ return CV_ELEM_SIZE(flags); }

inline size_t SparseMat::elemSize1() const
{ return CV_ELEM_SIZE1(flags); }

inline int SparseMat::type() const
{ return CV_MAT_TYPE(flags); }

inline int SparseMat::depth() const
{ return CV_MAT_DEPTH(flags); }

inline int SparseMat::channels() const
{ return CV_MAT_CN(flags); }

inline const int* SparseMat::size() const
{
    return hdr ? hdr->size : 0;
}

inline int SparseMat::size(int i) const
{
    if( hdr )
    {
        CV_DbgAssert((unsigned)i < (unsigned)hdr->dims);
        return hdr->size[i];
    }
    return 0;
}

inline int SparseMat::dims() const
{
    return hdr ? hdr->dims : 0;
}

inline size_t SparseMat::nzcount() const
{
    return hdr ? hdr->nodeCount : 0;
}

inline size_t SparseMat::hash(int i0) const
{
    return (size_t)i0;
}

inline size_t SparseMat::hash(int i0, int i1) const
{
    return (size_t)(unsigned)i0*HASH_SCALE + (unsigned)i1;
}

inline size_t SparseMat::hash(int i0, int i1, int i2) const
{
    return ((size_t)(unsigned)i0*HASH_SCALE + (unsigned)i1)*HASH_SCALE + (unsigned)i2;
}

inline size_t SparseMat::hash(const int* idx) const
{
    size_t h = (unsigned)idx[0];
    if( !hdr )
        return 0;
    int i, d = hdr->dims;
    for( i = 1; i < d; i++ )
        h = h*HASH_SCALE + (unsigned)idx[i];
    return h;
}

template<typename _Tp> inline _Tp& SparseMat::ref(int i0, size_t* hashval)
{ return *(_Tp*)((SparseMat*)this)->ptr(i0, true, hashval); }
    
template<typename _Tp> inline _Tp& SparseMat::ref(int i0, int i1, size_t* hashval)
{ return *(_Tp*)((SparseMat*)this)->ptr(i0, i1, true, hashval); }

template<typename _Tp> inline _Tp& SparseMat::ref(int i0, int i1, int i2, size_t* hashval)
{ return *(_Tp*)((SparseMat*)this)->ptr(i0, i1, i2, true, hashval); }

template<typename _Tp> inline _Tp& SparseMat::ref(const int* idx, size_t* hashval)
{ return *(_Tp*)((SparseMat*)this)->ptr(idx, true, hashval); }

template<typename _Tp> inline _Tp SparseMat::value(int i0, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(i0, false, hashval);
    return p ? *p : _Tp();
}    
    
template<typename _Tp> inline _Tp SparseMat::value(int i0, int i1, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(i0, i1, false, hashval);
    return p ? *p : _Tp();
}

template<typename _Tp> inline _Tp SparseMat::value(int i0, int i1, int i2, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(i0, i1, i2, false, hashval);
    return p ? *p : _Tp();
}

template<typename _Tp> inline _Tp SparseMat::value(const int* idx, size_t* hashval) const
{
    const _Tp* p = (const _Tp*)((SparseMat*)this)->ptr(idx, false, hashval);
    return p ? *p : _Tp();
}

template<typename _Tp> inline const _Tp* SparseMat::find(int i0, size_t* hashval) const
{ return (const _Tp*)((SparseMat*)this)->ptr(i0, false, hashval); }
    
template<typename _Tp> inline const _Tp* SparseMat::find(int i0, int i1, size_t* hashval) const
{ return (const _Tp*)((SparseMat*)this)->ptr(i0, i1, false, hashval); }

template<typename _Tp> inline const _Tp* SparseMat::find(int i0, int i1, int i2, size_t* hashval) const
{ return (const _Tp*)((SparseMat*)this)->ptr(i0, i1, i2, false, hashval); }

template<typename _Tp> inline const _Tp* SparseMat::find(const int* idx, size_t* hashval) const
{ return (const _Tp*)((SparseMat*)this)->ptr(idx, false, hashval); }

template<typename _Tp> inline _Tp& SparseMat::value(Node* n)
{ return *(_Tp*)((uchar*)n + hdr->valueOffset); }

template<typename _Tp> inline const _Tp& SparseMat::value(const Node* n) const
{ return *(const _Tp*)((const uchar*)n + hdr->valueOffset); }

inline SparseMat::Node* SparseMat::node(size_t nidx)
{ return (Node*)&hdr->pool[nidx]; }

inline const SparseMat::Node* SparseMat::node(size_t nidx) const
{ return (const Node*)&hdr->pool[nidx]; }

inline SparseMatIterator SparseMat::begin()
{ return SparseMatIterator(this); }

inline SparseMatConstIterator SparseMat::begin() const
{ return SparseMatConstIterator(this); }

inline SparseMatIterator SparseMat::end()
{ SparseMatIterator it(this); it.seekEnd(); return it; }
    
inline SparseMatConstIterator SparseMat::end() const
{ SparseMatConstIterator it(this); it.seekEnd(); return it; }
    
template<typename _Tp> inline SparseMatIterator_<_Tp> SparseMat::begin()
{ return SparseMatIterator_<_Tp>(this); }
    
template<typename _Tp> inline SparseMatConstIterator_<_Tp> SparseMat::begin() const
{ return SparseMatConstIterator_<_Tp>(this); }
    
template<typename _Tp> inline SparseMatIterator_<_Tp> SparseMat::end()
{ SparseMatIterator_<_Tp> it(this); it.seekEnd(); return it; }

template<typename _Tp> inline SparseMatConstIterator_<_Tp> SparseMat::end() const
{ SparseMatConstIterator_<_Tp> it(this); it.seekEnd(); return it; }
    
    
inline SparseMatConstIterator::SparseMatConstIterator()
: m(0), hashidx(0), ptr(0)
{
}

inline SparseMatConstIterator::SparseMatConstIterator(const SparseMatConstIterator& it)
: m(it.m), hashidx(it.hashidx), ptr(it.ptr)
{
}

static inline bool operator == (const SparseMatConstIterator& it1, const SparseMatConstIterator& it2)
{ return it1.m == it2.m && it1.hashidx == it2.hashidx && it1.ptr == it2.ptr; }

static inline bool operator != (const SparseMatConstIterator& it1, const SparseMatConstIterator& it2)
{ return !(it1 == it2); }


inline SparseMatConstIterator& SparseMatConstIterator::operator = (const SparseMatConstIterator& it)
{
    if( this != &it )
    {
        m = it.m;
        hashidx = it.hashidx;
        ptr = it.ptr;
    }
    return *this;
}

template<typename _Tp> inline const _Tp& SparseMatConstIterator::value() const
{ return *(_Tp*)ptr; }

inline const SparseMat::Node* SparseMatConstIterator::node() const
{
    return ptr && m && m->hdr ?
        (const SparseMat::Node*)(ptr - m->hdr->valueOffset) : 0;
}

inline SparseMatConstIterator SparseMatConstIterator::operator ++(int)
{
    SparseMatConstIterator it = *this;
    ++*this;
    return it;
}

    
inline void SparseMatConstIterator::seekEnd()
{
    if( m && m->hdr )
    {
        hashidx = m->hdr->hashtab.size();
        ptr = 0;
    }
}
    
inline SparseMatIterator::SparseMatIterator()
{}

inline SparseMatIterator::SparseMatIterator(SparseMat* _m)
: SparseMatConstIterator(_m)
{}

inline SparseMatIterator::SparseMatIterator(const SparseMatIterator& it)
: SparseMatConstIterator(it)
{
}

inline SparseMatIterator& SparseMatIterator::operator = (const SparseMatIterator& it)
{
    (SparseMatConstIterator&)*this = it;
    return *this;
}

template<typename _Tp> inline _Tp& SparseMatIterator::value() const
{ return *(_Tp*)ptr; }

inline SparseMat::Node* SparseMatIterator::node() const
{
    return (SparseMat::Node*)SparseMatConstIterator::node();
}

inline SparseMatIterator& SparseMatIterator::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

inline SparseMatIterator SparseMatIterator::operator ++(int)
{
    SparseMatIterator it = *this;
    ++*this;
    return it;
}


template<typename _Tp> inline SparseMat_<_Tp>::SparseMat_()
{ flags = MAGIC_VAL | DataType<_Tp>::type; }

template<typename _Tp> inline SparseMat_<_Tp>::SparseMat_(int _dims, const int* _sizes)
: SparseMat(_dims, _sizes, DataType<_Tp>::type)
{}

template<typename _Tp> inline SparseMat_<_Tp>::SparseMat_(const SparseMat& m)
{
    if( m.type() == DataType<_Tp>::type )
        *this = (const SparseMat_<_Tp>&)m;
    else
        m.convertTo(this, DataType<_Tp>::type);
}

template<typename _Tp> inline SparseMat_<_Tp>::SparseMat_(const SparseMat_<_Tp>& m)
{
    this->flags = m.flags;
    this->hdr = m.hdr;
    if( this->hdr )
        CV_XADD(&this->hdr->refcount, 1);
}

template<typename _Tp> inline SparseMat_<_Tp>::SparseMat_(const Mat& m)
{
    SparseMat sm(m);
    *this = sm;
}

template<typename _Tp> inline SparseMat_<_Tp>::SparseMat_(const MatND& m)
{
    SparseMat sm(m);
    *this = sm;
}

template<typename _Tp> inline SparseMat_<_Tp>::SparseMat_(const CvSparseMat* m)
{
    SparseMat sm(m);
    *this = sm;
}

template<typename _Tp> inline SparseMat_<_Tp>&
SparseMat_<_Tp>::operator = (const SparseMat_<_Tp>& m)
{
    if( this != &m )
    {
        if( m.hdr ) CV_XADD(&m.hdr->refcount, 1);
        release();
        flags = m.flags;
        hdr = m.hdr;
    }
    return *this;
}

template<typename _Tp> inline SparseMat_<_Tp>&
SparseMat_<_Tp>::operator = (const SparseMat& m)
{
    if( m.type() == DataType<_Tp>::type )
        return (*this = (const SparseMat_<_Tp>&)m);
    m.convertTo(*this, DataType<_Tp>::type);
    return *this;
}

template<typename _Tp> inline SparseMat_<_Tp>&
SparseMat_<_Tp>::operator = (const Mat& m)
{ return (*this = SparseMat(m)); }

template<typename _Tp> inline SparseMat_<_Tp>&
SparseMat_<_Tp>::operator = (const MatND& m)
{ return (*this = SparseMat(m)); }

template<typename _Tp> inline SparseMat_<_Tp>
SparseMat_<_Tp>::clone() const
{
    SparseMat_<_Tp> m;
    this->copyTo(m);
    return m;
}

template<typename _Tp> inline void
SparseMat_<_Tp>::create(int _dims, const int* _sizes)
{
    SparseMat::create(_dims, _sizes, DataType<_Tp>::type);
}

template<typename _Tp> inline
SparseMat_<_Tp>::operator CvSparseMat*() const
{
    return SparseMat::operator CvSparseMat*();
}

template<typename _Tp> inline int SparseMat_<_Tp>::type() const
{ return DataType<_Tp>::type; }

template<typename _Tp> inline int SparseMat_<_Tp>::depth() const
{ return DataType<_Tp>::depth; }

template<typename _Tp> inline int SparseMat_<_Tp>::channels() const
{ return DataType<_Tp>::channels; }

template<typename _Tp> inline _Tp&
SparseMat_<_Tp>::ref(int i0, size_t* hashval)
{ return SparseMat::ref<_Tp>(i0, hashval); }

template<typename _Tp> inline _Tp
SparseMat_<_Tp>::operator()(int i0, size_t* hashval) const
{ return SparseMat::value<_Tp>(i0, hashval); }    
    
template<typename _Tp> inline _Tp&
SparseMat_<_Tp>::ref(int i0, int i1, size_t* hashval)
{ return SparseMat::ref<_Tp>(i0, i1, hashval); }

template<typename _Tp> inline _Tp
SparseMat_<_Tp>::operator()(int i0, int i1, size_t* hashval) const
{ return SparseMat::value<_Tp>(i0, i1, hashval); }

template<typename _Tp> inline _Tp&
SparseMat_<_Tp>::ref(int i0, int i1, int i2, size_t* hashval)
{ return SparseMat::ref<_Tp>(i0, i1, i2, hashval); }

template<typename _Tp> inline _Tp
SparseMat_<_Tp>::operator()(int i0, int i1, int i2, size_t* hashval) const
{ return SparseMat::value<_Tp>(i0, i1, i2, hashval); }

template<typename _Tp> inline _Tp&
SparseMat_<_Tp>::ref(const int* idx, size_t* hashval)
{ return SparseMat::ref<_Tp>(idx, hashval); }

template<typename _Tp> inline _Tp
SparseMat_<_Tp>::operator()(const int* idx, size_t* hashval) const
{ return SparseMat::value<_Tp>(idx, hashval); }

template<typename _Tp> inline SparseMatIterator_<_Tp> SparseMat_<_Tp>::begin()
{ return SparseMatIterator_<_Tp>(this); }

template<typename _Tp> inline SparseMatConstIterator_<_Tp> SparseMat_<_Tp>::begin() const
{ return SparseMatConstIterator_<_Tp>(this); }

template<typename _Tp> inline SparseMatIterator_<_Tp> SparseMat_<_Tp>::end()
{ SparseMatIterator_<_Tp> it(this); it.seekEnd(); return it; }
    
template<typename _Tp> inline SparseMatConstIterator_<_Tp> SparseMat_<_Tp>::end() const
{ SparseMatConstIterator_<_Tp> it(this); it.seekEnd(); return it; }

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_()
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMat_<_Tp>* _m)
: SparseMatConstIterator(_m)
{}

template<typename _Tp> inline
SparseMatConstIterator_<_Tp>::SparseMatConstIterator_(const SparseMatConstIterator_<_Tp>& it)
: SparseMatConstIterator(it)
{}

template<typename _Tp> inline SparseMatConstIterator_<_Tp>&
SparseMatConstIterator_<_Tp>::operator = (const SparseMatConstIterator_<_Tp>& it)
{ return ((SparseMatConstIterator&)*this = it); }

template<typename _Tp> inline const _Tp&
SparseMatConstIterator_<_Tp>::operator *() const
{ return *(const _Tp*)this->ptr; }

template<typename _Tp> inline SparseMatConstIterator_<_Tp>&
SparseMatConstIterator_<_Tp>::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline SparseMatConstIterator_<_Tp>
SparseMatConstIterator_<_Tp>::operator ++(int)
{
    SparseMatConstIterator it = *this;
    SparseMatConstIterator::operator ++();
    return it;
}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_()
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(SparseMat_<_Tp>* _m)
: SparseMatConstIterator_<_Tp>(_m)
{}

template<typename _Tp> inline
SparseMatIterator_<_Tp>::SparseMatIterator_(const SparseMatIterator_<_Tp>& it)
: SparseMatConstIterator_<_Tp>(it)
{}

template<typename _Tp> inline SparseMatIterator_<_Tp>&
SparseMatIterator_<_Tp>::operator = (const SparseMatIterator_<_Tp>& it)
{ return ((SparseMatIterator&)*this = it); }

template<typename _Tp> inline _Tp&
SparseMatIterator_<_Tp>::operator *() const
{ return *(_Tp*)this->ptr; }

template<typename _Tp> inline SparseMatIterator_<_Tp>&
SparseMatIterator_<_Tp>::operator ++()
{
    SparseMatConstIterator::operator ++();
    return *this;
}

template<typename _Tp> inline SparseMatIterator_<_Tp>
SparseMatIterator_<_Tp>::operator ++(int)
{
    SparseMatIterator it = *this;
    SparseMatConstIterator::operator ++();
    return it;
}
    
}

#endif
#endif
