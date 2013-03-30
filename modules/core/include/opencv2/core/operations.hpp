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

#ifndef __OPENCV_CORE_OPERATIONS_HPP__
#define __OPENCV_CORE_OPERATIONS_HPP__

#ifndef __cplusplus
#  error operations.hpp header must be compiled as C++
#endif

#ifndef SKIP_INCLUDES
  #include <string.h>
  #include <limits.h>
#endif // SKIP_INCLUDES


#include <limits>

#ifdef _MSC_VER
# pragma warning(push)
# pragma warning(disable:4127) //conditional expression is constant
#endif

namespace cv
{

inline int fast_abs(uchar v) { return v; }
inline int fast_abs(schar v) { return std::abs((int)v); }
inline int fast_abs(ushort v) { return v; }
inline int fast_abs(short v) { return std::abs((int)v); }
inline int fast_abs(int v) { return std::abs(v); }
inline float fast_abs(float v) { return std::abs(v); }
inline double fast_abs(double v) { return std::abs(v); }

namespace internal
{

template<typename _Tp, int m> struct CV_EXPORTS Matx_FastInvOp
{
    bool operator()(const Matx<_Tp, m, m>& a, Matx<_Tp, m, m>& b, int method) const
    {
        Matx<_Tp, m, m> temp = a;

        // assume that b is all 0's on input => make it a unity matrix
        for( int i = 0; i < m; i++ )
            b(i, i) = (_Tp)1;

        if( method == DECOMP_CHOLESKY )
            return Cholesky(temp.val, m*sizeof(_Tp), m, b.val, m*sizeof(_Tp), m);

        return LU(temp.val, m*sizeof(_Tp), m, b.val, m*sizeof(_Tp), m) != 0;
    }
};

template<typename _Tp> struct CV_EXPORTS Matx_FastInvOp<_Tp, 2>
{
    bool operator()(const Matx<_Tp, 2, 2>& a, Matx<_Tp, 2, 2>& b, int) const
    {
        _Tp d = determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        b(1,1) = a(0,0)*d;
        b(0,0) = a(1,1)*d;
        b(0,1) = -a(0,1)*d;
        b(1,0) = -a(1,0)*d;
        return true;
    }
};

template<typename _Tp> struct CV_EXPORTS Matx_FastInvOp<_Tp, 3>
{
    bool operator()(const Matx<_Tp, 3, 3>& a, Matx<_Tp, 3, 3>& b, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        b(0,0) = (a(1,1) * a(2,2) - a(1,2) * a(2,1)) * d;
        b(0,1) = (a(0,2) * a(2,1) - a(0,1) * a(2,2)) * d;
        b(0,2) = (a(0,1) * a(1,2) - a(0,2) * a(1,1)) * d;

        b(1,0) = (a(1,2) * a(2,0) - a(1,0) * a(2,2)) * d;
        b(1,1) = (a(0,0) * a(2,2) - a(0,2) * a(2,0)) * d;
        b(1,2) = (a(0,2) * a(1,0) - a(0,0) * a(1,2)) * d;

        b(2,0) = (a(1,0) * a(2,1) - a(1,1) * a(2,0)) * d;
        b(2,1) = (a(0,1) * a(2,0) - a(0,0) * a(2,1)) * d;
        b(2,2) = (a(0,0) * a(1,1) - a(0,1) * a(1,0)) * d;
        return true;
    }
};


template<typename _Tp, int m, int n> struct CV_EXPORTS Matx_FastSolveOp
{
    bool operator()(const Matx<_Tp, m, m>& a, const Matx<_Tp, m, n>& b,
                    Matx<_Tp, m, n>& x, int method) const
    {
        Matx<_Tp, m, m> temp = a;
        x = b;
        if( method == DECOMP_CHOLESKY )
            return Cholesky(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n);

        return LU(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n) != 0;
    }
};

template<typename _Tp> struct CV_EXPORTS Matx_FastSolveOp<_Tp, 2, 1>
{
    bool operator()(const Matx<_Tp, 2, 2>& a, const Matx<_Tp, 2, 1>& b,
                    Matx<_Tp, 2, 1>& x, int) const
    {
        _Tp d = determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        x(0) = (b(0)*a(1,1) - b(1)*a(0,1))*d;
        x(1) = (b(1)*a(0,0) - b(0)*a(1,0))*d;
        return true;
    }
};

template<typename _Tp> struct CV_EXPORTS Matx_FastSolveOp<_Tp, 3, 1>
{
    bool operator()(const Matx<_Tp, 3, 3>& a, const Matx<_Tp, 3, 1>& b,
                    Matx<_Tp, 3, 1>& x, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if( d == 0 )
            return false;
        d = 1/d;
        x(0) = d*(b(0)*(a(1,1)*a(2,2) - a(1,2)*a(2,1)) -
                a(0,1)*(b(1)*a(2,2) - a(1,2)*b(2)) +
                a(0,2)*(b(1)*a(2,1) - a(1,1)*b(2)));

        x(1) = d*(a(0,0)*(b(1)*a(2,2) - a(1,2)*b(2)) -
                b(0)*(a(1,0)*a(2,2) - a(1,2)*a(2,0)) +
                a(0,2)*(a(1,0)*b(2) - b(1)*a(2,0)));

        x(2) = d*(a(0,0)*(a(1,1)*b(2) - b(1)*a(2,1)) -
                a(0,1)*(a(1,0)*b(2) - b(1)*a(2,0)) +
                b(0)*(a(1,0)*a(2,1) - a(1,1)*a(2,0)));
        return true;
    }
};

} // internal

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::randu(_Tp a, _Tp b)
{
    Matx<_Tp,m,n> M;
    cv::randu(M, Scalar(a), Scalar(b));
    return M;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp,m,n> Matx<_Tp,m,n>::randn(_Tp a, _Tp b)
{
    Matx<_Tp,m,n> M;
    cv::randn(M, Scalar(a), Scalar(b));
    return M;
}

template<typename _Tp, int m, int n> inline
Matx<_Tp, n, m> Matx<_Tp, m, n>::inv(int method) const
{
    Matx<_Tp, n, m> b;
    bool ok;
    if( method == DECOMP_LU || method == DECOMP_CHOLESKY )
        ok = internal::Matx_FastInvOp<_Tp, m>()(*this, b, method);
    else
    {
        Mat A(*this, false), B(b, false);
        ok = (invert(A, B, method) != 0);
    }
    return ok ? b : Matx<_Tp, n, m>::zeros();
}

template<typename _Tp, int m, int n> template<int l> inline
Matx<_Tp, n, l> Matx<_Tp, m, n>::solve(const Matx<_Tp, m, l>& rhs, int method) const
{
    Matx<_Tp, n, l> x;
    bool ok;
    if( method == DECOMP_LU || method == DECOMP_CHOLESKY )
        ok = internal::Matx_FastSolveOp<_Tp, m, l>()(*this, rhs, x, method);
    else
    {
        Mat A(*this, false), B(rhs, false), X(x, false);
        ok = cv::solve(A, B, X, method);
    }

    return ok ? x : Matx<_Tp, n, l>::zeros();
}



////////////////////////////// Augmenting algebraic & logical operations //////////////////////////////////

#define CV_MAT_AUG_OPERATOR1(op, cvop, A, B) \
    static inline A& operator op (A& a, const B& b) { cvop; return a; }

#define CV_MAT_AUG_OPERATOR(op, cvop, A, B)   \
    CV_MAT_AUG_OPERATOR1(op, cvop, A, B)      \
    CV_MAT_AUG_OPERATOR1(op, cvop, const A, B)

#define CV_MAT_AUG_OPERATOR_T(op, cvop, A, B)                   \
    template<typename _Tp> CV_MAT_AUG_OPERATOR1(op, cvop, A, B) \
    template<typename _Tp> CV_MAT_AUG_OPERATOR1(op, cvop, const A, B)

CV_MAT_AUG_OPERATOR  (+=, cv::add(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (+=, cv::add(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (-=, cv::subtract(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (-=, cv::subtract(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat, Mat)
CV_MAT_AUG_OPERATOR_T(*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR  (*=, a.convertTo(a, -1, b), Mat, double)
CV_MAT_AUG_OPERATOR_T(*=, a.convertTo(a, -1, b), Mat_<_Tp>, double)

CV_MAT_AUG_OPERATOR  (/=, cv::divide(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR_T(/=, cv::divide(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(/=, cv::divide(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR  (/=, a.convertTo((Mat&)a, -1, 1./b), Mat, double)
CV_MAT_AUG_OPERATOR_T(/=, a.convertTo((Mat&)a, -1, 1./b), Mat_<_Tp>, double)

CV_MAT_AUG_OPERATOR  (&=, cv::bitwise_and(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (&=, cv::bitwise_and(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (|=, cv::bitwise_or(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (|=, cv::bitwise_or(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (^=, cv::bitwise_xor(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (^=, cv::bitwise_xor(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Mat_<_Tp>)

#undef CV_MAT_AUG_OPERATOR_T
#undef CV_MAT_AUG_OPERATOR
#undef CV_MAT_AUG_OPERATOR1



///////////////////////////////////////////// SVD //////////////////////////////////////////////////////

inline SVD::SVD() {}
inline SVD::SVD( InputArray m, int flags ) { operator ()(m, flags); }
inline void SVD::solveZ( InputArray m, OutputArray _dst )
{
    Mat mtx = m.getMat();
    SVD svd(mtx, (mtx.rows >= mtx.cols ? 0 : SVD::FULL_UV));
    _dst.create(svd.vt.cols, 1, svd.vt.type());
    Mat dst = _dst.getMat();
    svd.vt.row(svd.vt.rows-1).reshape(1,svd.vt.cols).copyTo(dst);
}

template<typename _Tp, int m, int n, int nm> inline void
    SVD::compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w, Matx<_Tp, m, nm>& u, Matx<_Tp, n, nm>& vt )
{
    CV_StaticAssert( nm == MIN(m, n), "Invalid size of output vector.");
    Mat _a(a, false), _u(u, false), _w(w, false), _vt(vt, false);
    SVD::compute(_a, _w, _u, _vt);
    CV_Assert(_w.data == (uchar*)&w.val[0] && _u.data == (uchar*)&u.val[0] && _vt.data == (uchar*)&vt.val[0]);
}

template<typename _Tp, int m, int n, int nm> inline void
SVD::compute( const Matx<_Tp, m, n>& a, Matx<_Tp, nm, 1>& w )
{
    CV_StaticAssert( nm == MIN(m, n), "Invalid size of output vector.");
    Mat _a(a, false), _w(w, false);
    SVD::compute(_a, _w);
    CV_Assert(_w.data == (uchar*)&w.val[0]);
}

template<typename _Tp, int m, int n, int nm, int nb> inline void
SVD::backSubst( const Matx<_Tp, nm, 1>& w, const Matx<_Tp, m, nm>& u,
                const Matx<_Tp, n, nm>& vt, const Matx<_Tp, m, nb>& rhs,
                Matx<_Tp, n, nb>& dst )
{
    CV_StaticAssert( nm == MIN(m, n), "Invalid size of output vector.");
    Mat _u(u, false), _w(w, false), _vt(vt, false), _rhs(rhs, false), _dst(dst, false);
    SVD::backSubst(_w, _u, _vt, _rhs, _dst);
    CV_Assert(_dst.data == (uchar*)&dst.val[0]);
}

//////////////////////////////// Vector ////////////////////////////////

// template vector class. It is similar to STL's vector,
// with a few important differences:
//   1) it can be created on top of user-allocated data w/o copying it
//   2) vector b = a means copying the header,
//      not the underlying data (use clone() to make a deep copy)
template <typename _Tp> class CV_EXPORTS Vector
{
public:
    typedef _Tp value_type;
    typedef _Tp* iterator;
    typedef const _Tp* const_iterator;
    typedef _Tp& reference;
    typedef const _Tp& const_reference;

    struct CV_EXPORTS Hdr
    {
        Hdr() : data(0), datastart(0), refcount(0), size(0), capacity(0) {};
        _Tp* data;
        _Tp* datastart;
        int* refcount;
        size_t size;
        size_t capacity;
    };

    Vector() {}
    Vector(size_t _size)  { resize(_size); }
    Vector(size_t _size, const _Tp& val)
    {
        resize(_size);
        for(size_t i = 0; i < _size; i++)
            hdr.data[i] = val;
    }
    Vector(_Tp* _data, size_t _size, bool _copyData=false)
    { set(_data, _size, _copyData); }

    template<int n> Vector(const Vec<_Tp, n>& vec)
    { set((_Tp*)&vec.val[0], n, true); }

    Vector(const std::vector<_Tp>& vec, bool _copyData=false)
    { set(!vec.empty() ? (_Tp*)&vec[0] : 0, vec.size(), _copyData); }

    Vector(const Vector& d) { *this = d; }

    Vector(const Vector& d, const Range& r_)
    {
        Range r = r_ == Range::all() ? Range(0, d.size()) : r_;
        /*if( r == Range::all() )
            r = Range(0, d.size());*/
        if( r.size() > 0 && r.start >= 0 && r.end <= d.size() )
        {
            if( d.hdr.refcount )
                CV_XADD(d.hdr.refcount, 1);
            hdr.refcount = d.hdr.refcount;
            hdr.datastart = d.hdr.datastart;
            hdr.data = d.hdr.data + r.start;
            hdr.capacity = hdr.size = r.size();
        }
    }

    Vector<_Tp>& operator = (const Vector& d)
    {
        if( this != &d )
        {
            if( d.hdr.refcount )
                CV_XADD(d.hdr.refcount, 1);
            release();
            hdr = d.hdr;
        }
        return *this;
    }

    ~Vector()  { release(); }

    Vector<_Tp> clone() const
    { return hdr.data ? Vector<_Tp>(hdr.data, hdr.size, true) : Vector<_Tp>(); }

    void copyTo(Vector<_Tp>& vec) const
    {
        size_t i, sz = size();
        vec.resize(sz);
        const _Tp* src = hdr.data;
        _Tp* dst = vec.hdr.data;
        for( i = 0; i < sz; i++ )
            dst[i] = src[i];
    }

    void copyTo(std::vector<_Tp>& vec) const
    {
        size_t i, sz = size();
        vec.resize(sz);
        const _Tp* src = hdr.data;
        _Tp* dst = sz ? &vec[0] : 0;
        for( i = 0; i < sz; i++ )
            dst[i] = src[i];
    }

    // operator CvMat() const
    // { return cvMat((int)size(), 1, type(), (void*)hdr.data); }

    _Tp& operator [] (size_t i) { CV_DbgAssert( i < size() ); return hdr.data[i]; }
    const _Tp& operator [] (size_t i) const { CV_DbgAssert( i < size() ); return hdr.data[i]; }
    Vector operator() (const Range& r) const { return Vector(*this, r); }
    _Tp& back() { CV_DbgAssert(!empty()); return hdr.data[hdr.size-1]; }
    const _Tp& back() const { CV_DbgAssert(!empty()); return hdr.data[hdr.size-1]; }
    _Tp& front() { CV_DbgAssert(!empty()); return hdr.data[0]; }
    const _Tp& front() const { CV_DbgAssert(!empty()); return hdr.data[0]; }

    _Tp* begin() { return hdr.data; }
    _Tp* end() { return hdr.data + hdr.size; }
    const _Tp* begin() const { return hdr.data; }
    const _Tp* end() const { return hdr.data + hdr.size; }

    void addref() { if( hdr.refcount ) CV_XADD(hdr.refcount, 1); }
    void release()
    {
        if( hdr.refcount && CV_XADD(hdr.refcount, -1) == 1 )
        {
            delete[] hdr.datastart;
            delete hdr.refcount;
        }
        hdr = Hdr();
    }

    void set(_Tp* _data, size_t _size, bool _copyData=false)
    {
        if( !_copyData )
        {
            release();
            hdr.data = hdr.datastart = _data;
            hdr.size = hdr.capacity = _size;
            hdr.refcount = 0;
        }
        else
        {
            reserve(_size);
            for( size_t i = 0; i < _size; i++ )
                hdr.data[i] = _data[i];
            hdr.size = _size;
        }
    }

    void reserve(size_t newCapacity)
    {
        _Tp* newData;
        int* newRefcount;
        size_t i, oldSize = hdr.size;
        if( (!hdr.refcount || *hdr.refcount == 1) && hdr.capacity >= newCapacity )
            return;
        newCapacity = std::max(newCapacity, oldSize);
        newData = new _Tp[newCapacity];
        newRefcount = new int(1);
        for( i = 0; i < oldSize; i++ )
            newData[i] = hdr.data[i];
        release();
        hdr.data = hdr.datastart = newData;
        hdr.capacity = newCapacity;
        hdr.size = oldSize;
        hdr.refcount = newRefcount;
    }

    void resize(size_t newSize)
    {
        size_t i;
        newSize = std::max(newSize, (size_t)0);
        if( (!hdr.refcount || *hdr.refcount == 1) && hdr.size == newSize )
            return;
        if( newSize > hdr.capacity )
            reserve(std::max(newSize, std::max((size_t)4, hdr.capacity*2)));
        for( i = hdr.size; i < newSize; i++ )
            hdr.data[i] = _Tp();
        hdr.size = newSize;
    }

    Vector<_Tp>& push_back(const _Tp& elem)
    {
        if( hdr.size == hdr.capacity )
            reserve( std::max((size_t)4, hdr.capacity*2) );
        hdr.data[hdr.size++] = elem;
        return *this;
    }

    Vector<_Tp>& pop_back()
    {
        if( hdr.size > 0 )
            --hdr.size;
        return *this;
    }

    size_t size() const { return hdr.size; }
    size_t capacity() const { return hdr.capacity; }
    bool empty() const { return hdr.size == 0; }
    void clear() { resize(0); }
    int type() const { return DataType<_Tp>::type; }

protected:
    Hdr hdr;
};


template<typename _Tp> inline typename DataType<_Tp>::work_type
dot(const Vector<_Tp>& v1, const Vector<_Tp>& v2)
{
    typedef typename DataType<_Tp>::work_type _Tw;
    size_t i = 0, n = v1.size();
    assert(v1.size() == v2.size());

    _Tw s = 0;
    const _Tp *ptr1 = &v1[0], *ptr2 = &v2[0];
    for( ; i < n; i++ )
        s += (_Tw)ptr1[i]*ptr2[i];

    return s;
}

// Multiply-with-Carry RNG
inline RNG::RNG() { state = 0xffffffff; }
inline RNG::RNG(uint64 _state) { state = _state ? _state : 0xffffffff; }
inline unsigned RNG::next()
{
    state = (uint64)(unsigned)state* /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
    return (unsigned)state;
}

inline RNG::operator uchar() { return (uchar)next(); }
inline RNG::operator schar() { return (schar)next(); }
inline RNG::operator ushort() { return (ushort)next(); }
inline RNG::operator short() { return (short)next(); }
inline RNG::operator unsigned() { return next(); }
inline unsigned RNG::operator ()(unsigned N) {return (unsigned)uniform(0,N);}
inline unsigned RNG::operator ()() {return next();}
inline RNG::operator int() { return (int)next(); }
// * (2^32-1)^-1
inline RNG::operator float() { return next()*2.3283064365386962890625e-10f; }
inline RNG::operator double()
{
    unsigned t = next();
    return (((uint64)t << 32) | next())*5.4210108624275221700372640043497e-20;
}
inline int RNG::uniform(int a, int b) { return a == b ? a : (int)(next()%(b - a) + a); }
inline float RNG::uniform(float a, float b) { return ((float)*this)*(b - a) + a; }
inline double RNG::uniform(double a, double b) { return ((double)*this)*(b - a) + a; }

inline uchar* LineIterator::operator *() { return ptr; }
inline LineIterator& LineIterator::operator ++()
{
    int mask = err < 0 ? -1 : 0;
    err += minusDelta + (plusDelta & mask);
    ptr += minusStep + (plusStep & mask);
    return *this;
}
inline LineIterator LineIterator::operator ++(int)
{
    LineIterator it = *this;
    ++(*this);
    return it;
}
inline Point LineIterator::pos() const
{
    Point p;
    p.y = (int)((ptr - ptr0)/step);
    p.x = (int)(((ptr - ptr0) - p.y*step)/elemSize);
    return p;
}

//! returns the next unifomly-distributed random number of the specified type
template<typename _Tp> static inline _Tp randu()
{
  return (_Tp)theRNG();
}


//////////////////////////////////////// XML & YAML I/O ////////////////////////////////////

CV_EXPORTS_W void write( FileStorage& fs, const String& name, int value );
CV_EXPORTS_W void write( FileStorage& fs, const String& name, float value );
CV_EXPORTS_W void write( FileStorage& fs, const String& name, double value );
CV_EXPORTS_W void write( FileStorage& fs, const String& name, const String& value );

template<typename _Tp> inline void write(FileStorage& fs, const _Tp& value)
{ write(fs, String(), value); }

CV_EXPORTS void writeScalar( FileStorage& fs, int value );
CV_EXPORTS void writeScalar( FileStorage& fs, float value );
CV_EXPORTS void writeScalar( FileStorage& fs, double value );
CV_EXPORTS void writeScalar( FileStorage& fs, const String& value );

template<> inline void write( FileStorage& fs, const int& value )
{
    writeScalar(fs, value);
}

template<> inline void write( FileStorage& fs, const float& value )
{
    writeScalar(fs, value);
}

template<> inline void write( FileStorage& fs, const double& value )
{
    writeScalar(fs, value);
}

template<> inline void write( FileStorage& fs, const String& value )
{
    writeScalar(fs, value);
}

template<typename _Tp> inline void write(FileStorage& fs, const Point_<_Tp>& pt )
{
    write(fs, pt.x);
    write(fs, pt.y);
}

template<typename _Tp> inline void write(FileStorage& fs, const Point3_<_Tp>& pt )
{
    write(fs, pt.x);
    write(fs, pt.y);
    write(fs, pt.z);
}

template<typename _Tp> inline void write(FileStorage& fs, const Size_<_Tp>& sz )
{
    write(fs, sz.width);
    write(fs, sz.height);
}

template<typename _Tp> inline void write(FileStorage& fs, const Complex<_Tp>& c )
{
    write(fs, c.re);
    write(fs, c.im);
}

template<typename _Tp> inline void write(FileStorage& fs, const Rect_<_Tp>& r )
{
    write(fs, r.x);
    write(fs, r.y);
    write(fs, r.width);
    write(fs, r.height);
}

template<typename _Tp, int cn> inline void write(FileStorage& fs, const Vec<_Tp, cn>& v )
{
    for(int i = 0; i < cn; i++)
        write(fs, v.val[i]);
}

template<typename _Tp> inline void write(FileStorage& fs, const Scalar_<_Tp>& s )
{
    write(fs, s.val[0]);
    write(fs, s.val[1]);
    write(fs, s.val[2]);
    write(fs, s.val[3]);
}

inline void write(FileStorage& fs, const Range& r )
{
    write(fs, r.start);
    write(fs, r.end);
}

class CV_EXPORTS WriteStructContext
{
public:
    WriteStructContext(FileStorage& _fs, const String& name,
        int flags, const String& typeName=String());
    ~WriteStructContext();
    FileStorage* fs;
};

template<typename _Tp> inline void write(FileStorage& fs, const String& name, const Point_<_Tp>& pt )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt.x);
    write(fs, pt.y);
}

template<typename _Tp> inline void write(FileStorage& fs, const String& name, const Point3_<_Tp>& pt )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, pt.x);
    write(fs, pt.y);
    write(fs, pt.z);
}

template<typename _Tp> inline void write(FileStorage& fs, const String& name, const Size_<_Tp>& sz )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, sz.width);
    write(fs, sz.height);
}

template<typename _Tp> inline void write(FileStorage& fs, const String& name, const Complex<_Tp>& c )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, c.re);
    write(fs, c.im);
}

template<typename _Tp> inline void write(FileStorage& fs, const String& name, const Rect_<_Tp>& r )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r.x);
    write(fs, r.y);
    write(fs, r.width);
    write(fs, r.height);
}

template<typename _Tp, int cn> inline void write(FileStorage& fs, const String& name, const Vec<_Tp, cn>& v )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    for(int i = 0; i < cn; i++)
        write(fs, v.val[i]);
}

template<typename _Tp> inline void write(FileStorage& fs, const String& name, const Scalar_<_Tp>& s )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, s.val[0]);
    write(fs, s.val[1]);
    write(fs, s.val[2]);
    write(fs, s.val[3]);
}

inline void write(FileStorage& fs, const String& name, const Range& r )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+FileNode::FLOW);
    write(fs, r.start);
    write(fs, r.end);
}

template<typename _Tp, int numflag> class CV_EXPORTS VecWriterProxy
{
public:
    VecWriterProxy( FileStorage* _fs ) : fs(_fs) {}
    void operator()(const std::vector<_Tp>& vec) const
    {
        size_t i, count = vec.size();
        for( i = 0; i < count; i++ )
            write( *fs, vec[i] );
    }
    FileStorage* fs;
};

template<typename _Tp> class CV_EXPORTS VecWriterProxy<_Tp,1>
{
public:
    VecWriterProxy( FileStorage* _fs ) : fs(_fs) {}
    void operator()(const std::vector<_Tp>& vec) const
    {
        int _fmt = DataType<_Tp>::fmt;
        char fmt[] = { (char)((_fmt>>8)+'1'), (char)_fmt, '\0' };
        fs->writeRaw( String(fmt), !vec.empty() ? (uchar*)&vec[0] : 0, vec.size()*sizeof(_Tp) );
    }
    FileStorage* fs;
};

template<typename _Tp> static inline void write( FileStorage& fs, const std::vector<_Tp>& vec )
{
    VecWriterProxy<_Tp, DataType<_Tp>::fmt != 0> w(&fs);
    w(vec);
}

template<typename _Tp> static inline void write( FileStorage& fs, const String& name,
                                                const std::vector<_Tp>& vec )
{
    WriteStructContext ws(fs, name, FileNode::SEQ+(DataType<_Tp>::fmt != 0 ? FileNode::FLOW : 0));
    write(fs, vec);
}

CV_EXPORTS_W void write( FileStorage& fs, const String& name, const Mat& value );
CV_EXPORTS void write( FileStorage& fs, const String& name, const SparseMat& value );

template<typename _Tp> static inline FileStorage& operator << (FileStorage& fs, const _Tp& value)
{
    if( !fs.isOpened() )
        return fs;
    if( fs.state == FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP )
        CV_Error( Error::StsError, "No element name has been given" );
    write( fs, fs.elname, value );
    if( fs.state & FileStorage::INSIDE_MAP )
        fs.state = FileStorage::NAME_EXPECTED + FileStorage::INSIDE_MAP;
    return fs;
}

CV_EXPORTS FileStorage& operator << (FileStorage& fs, const String& str);

static inline FileStorage& operator << (FileStorage& fs, const char* str)
{ return (fs << String(str)); }

static inline FileStorage& operator << (FileStorage& fs, char* value)
{ return (fs << String(value)); }

inline FileNode::FileNode() : fs(0), node(0) {}
inline FileNode::FileNode(const CvFileStorage* _fs, const CvFileNode* _node)
    : fs(_fs), node(_node) {}

inline FileNode::FileNode(const FileNode& _node) : fs(_node.fs), node(_node.node) {}

inline bool FileNode::empty() const { return node == 0; }
inline bool FileNode::isNone() const { return type() == NONE; }
inline bool FileNode::isSeq() const { return type() == SEQ; }
inline bool FileNode::isMap() const { return type() == MAP; }
inline bool FileNode::isInt() const { return type() == INT; }
inline bool FileNode::isReal() const { return type() == REAL; }
inline bool FileNode::isString() const { return type() == STR; }

inline CvFileNode* FileNode::operator *() { return (CvFileNode*)node; }
inline const CvFileNode* FileNode::operator* () const { return node; }

CV_EXPORTS void read(const FileNode& node, int& value, int default_value);
CV_EXPORTS void read(const FileNode& node, float& value, float default_value);
CV_EXPORTS void read(const FileNode& node, double& value, double default_value);
CV_EXPORTS void read(const FileNode& node, String& value, const String& default_value);
CV_EXPORTS void read(const FileNode& node, SparseMat& mat, const SparseMat& default_mat=SparseMat() );
CV_EXPORTS_W void read(const FileNode& node, Mat& mat, const Mat& default_mat=Mat() );

CV_EXPORTS void read(const FileNode& node, std::vector<KeyPoint>& keypoints);
CV_EXPORTS void write(FileStorage& fs, const String& objname, const std::vector<KeyPoint>& keypoints);

static inline void read(const FileNode& node, bool& value, bool default_value)
{
    int temp; read(node, temp, (int)default_value);
    value = temp != 0;
}

static inline void read(const FileNode& node, uchar& value, uchar default_value)
{
    int temp; read(node, temp, (int)default_value);
    value = saturate_cast<uchar>(temp);
}

static inline void read(const FileNode& node, schar& value, schar default_value)
{
    int temp; read(node, temp, (int)default_value);
    value = saturate_cast<schar>(temp);
}

static inline void read(const FileNode& node, ushort& value, ushort default_value)
{
    int temp; read(node, temp, (int)default_value);
    value = saturate_cast<ushort>(temp);
}

static inline void read(const FileNode& node, short& value, short default_value)
{
    int temp; read(node, temp, (int)default_value);
    value = saturate_cast<short>(temp);
}

inline FileNode::operator int() const
{
    int value;
    read(*this, value, 0);
    return value;
}
inline FileNode::operator float() const
{
    float value;
    read(*this, value, 0.f);
    return value;
}
inline FileNode::operator double() const
{
    double value;
    read(*this, value, 0.);
    return value;
}
inline FileNode::operator String() const
{
    String value;
    read(*this, value, value);
    return value;
}

inline String::String(const FileNode& fn): cstr_(0), len_(0)
{
    read(fn, *this, *this);
}

inline void FileNode::readRaw( const String& fmt, uchar* vec, size_t len ) const
{
    begin().readRaw( fmt, vec, len );
}

template<typename _Tp, int numflag> class CV_EXPORTS VecReaderProxy
{
public:
    VecReaderProxy( FileNodeIterator* _it ) : it(_it) {}
    void operator()(std::vector<_Tp>& vec, size_t count) const
    {
        count = std::min(count, it->remaining);
        vec.resize(count);
        for( size_t i = 0; i < count; i++, ++(*it) )
            read(**it, vec[i], _Tp());
    }
    FileNodeIterator* it;
};

template<typename _Tp> class CV_EXPORTS VecReaderProxy<_Tp,1>
{
public:
    VecReaderProxy( FileNodeIterator* _it ) : it(_it) {}
    void operator()(std::vector<_Tp>& vec, size_t count) const
    {
        size_t remaining = it->remaining, cn = DataType<_Tp>::channels;
        int _fmt = DataType<_Tp>::fmt;
        char fmt[] = { (char)((_fmt>>8)+'1'), (char)_fmt, '\0' };
        size_t remaining1 = remaining/cn;
        count = count < remaining1 ? count : remaining1;
        vec.resize(count);
        it->readRaw( String(fmt), !vec.empty() ? (uchar*)&vec[0] : 0, count*sizeof(_Tp) );
    }
    FileNodeIterator* it;
};

template<typename _Tp> static inline void
read( FileNodeIterator& it, std::vector<_Tp>& vec, size_t maxCount=(size_t)INT_MAX )
{
    VecReaderProxy<_Tp, DataType<_Tp>::fmt != 0> r(&it);
    r(vec, maxCount);
}

template<typename _Tp> static inline void
read( const FileNode& node, std::vector<_Tp>& vec, const std::vector<_Tp>& default_value=std::vector<_Tp>() )
{
    if(!node.node)
        vec = default_value;
    else
    {
        FileNodeIterator it = node.begin();
        read( it, vec );
    }
}

inline FileNodeIterator FileNode::begin() const
{
    return FileNodeIterator(fs, node);
}

inline FileNodeIterator FileNode::end() const
{
    return FileNodeIterator(fs, node, size());
}

inline FileNode FileNodeIterator::operator *() const
{ return FileNode(fs, (const CvFileNode*)reader.ptr); }

inline FileNode FileNodeIterator::operator ->() const
{ return FileNode(fs, (const CvFileNode*)reader.ptr); }

template<typename _Tp> static inline FileNodeIterator& operator >> (FileNodeIterator& it, _Tp& value)
{ read( *it, value, _Tp()); return ++it; }

template<typename _Tp> static inline
FileNodeIterator& operator >> (FileNodeIterator& it, std::vector<_Tp>& vec)
{
    VecReaderProxy<_Tp, DataType<_Tp>::fmt != 0> r(&it);
    r(vec, (size_t)INT_MAX);
    return it;
}

template<typename _Tp> static inline void operator >> (const FileNode& n, _Tp& value)
{ read( n, value, _Tp()); }

template<typename _Tp> static inline void operator >> (const FileNode& n, std::vector<_Tp>& vec)
{ FileNodeIterator it = n.begin(); it >> vec; }

static inline bool operator == (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it1.fs == it2.fs && it1.container == it2.container &&
        it1.reader.ptr == it2.reader.ptr && it1.remaining == it2.remaining;
}

static inline bool operator != (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return !(it1 == it2);
}

static inline ptrdiff_t operator - (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it2.remaining - it1.remaining;
}

static inline bool operator < (const FileNodeIterator& it1, const FileNodeIterator& it2)
{
    return it1.remaining > it2.remaining;
}

inline FileNode FileStorage::getFirstTopLevelNode() const
{
    FileNode r = root();
    FileNodeIterator it = r.begin();
    return it != r.end() ? *it : FileNode();
}

//////////////////////////////////////// Various algorithms ////////////////////////////////////

template<typename _Tp> static inline _Tp gcd(_Tp a, _Tp b)
{
    if( a < b )
        std::swap(a, b);
    while( b > 0 )
    {
        _Tp r = a % b;
        a = b;
        b = r;
    }
    return a;
}

// This function splits the input sequence or set into one or more equivalence classes and
// returns the vector of labels - 0-based class indexes for each element.
// predicate(a,b) returns true if the two sequence elements certainly belong to the same class.
//
// The algorithm is described in "Introduction to Algorithms"
// by Cormen, Leiserson and Rivest, the chapter "Data structures for disjoint sets"
template<typename _Tp, class _EqPredicate> int
partition( const std::vector<_Tp>& _vec, std::vector<int>& labels,
           _EqPredicate predicate=_EqPredicate())
{
    int i, j, N = (int)_vec.size();
    const _Tp* vec = &_vec[0];

    const int PARENT=0;
    const int RANK=1;

    std::vector<int> _nodes(N*2);
    int (*nodes)[2] = (int(*)[2])&_nodes[0];

    // The first O(N) pass: create N single-vertex trees
    for(i = 0; i < N; i++)
    {
        nodes[i][PARENT]=-1;
        nodes[i][RANK] = 0;
    }

    // The main O(N^2) pass: merge connected components
    for( i = 0; i < N; i++ )
    {
        int root = i;

        // find root
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];

        for( j = 0; j < N; j++ )
        {
            if( i == j || !predicate(vec[i], vec[j]))
                continue;
            int root2 = j;

            while( nodes[root2][PARENT] >= 0 )
                root2 = nodes[root2][PARENT];

            if( root2 != root )
            {
                // unite both trees
                int rank = nodes[root][RANK], rank2 = nodes[root2][RANK];
                if( rank > rank2 )
                    nodes[root2][PARENT] = root;
                else
                {
                    nodes[root][PARENT] = root2;
                    nodes[root2][RANK] += rank == rank2;
                    root = root2;
                }
                CV_Assert( nodes[root][PARENT] < 0 );

                int k = j, parent;

                // compress the path from node2 to root
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }

                // compress the path from node to root
                k = i;
                while( (parent = nodes[k][PARENT]) >= 0 )
                {
                    nodes[k][PARENT] = root;
                    k = parent;
                }
            }
        }
    }

    // Final O(N) pass: enumerate classes
    labels.resize(N);
    int nclasses = 0;

    for( i = 0; i < N; i++ )
    {
        int root = i;
        while( nodes[root][PARENT] >= 0 )
            root = nodes[root][PARENT];
        // re-use the rank as the class label
        if( nodes[root][RANK] >= 0 )
            nodes[root][RANK] = ~nclasses++;
        labels[i] = ~nodes[root][RANK];
    }

    return nclasses;
}


//////////////////////////////////////////////////////////////////////////////

class CV_EXPORTS Formatter
{
public:
    virtual ~Formatter() {}
    virtual void write(std::ostream& out, const Mat& m, const int* params=0, int nparams=0) const = 0;
    virtual void write(std::ostream& out, const void* data, int nelems, int type,
                       const int* params=0, int nparams=0) const = 0;
    static const Formatter* get(const char* fmt="");
    static const Formatter* setDefault(const Formatter* fmt);
};


struct CV_EXPORTS Formatted
{
    Formatted(const Mat& m, const Formatter* fmt,
              const std::vector<int>& params);
    Formatted(const Mat& m, const Formatter* fmt,
              const int* params=0);
    Mat mtx;
    const Formatter* fmt;
    std::vector<int> params;
};

static inline Formatted format(const Mat& mtx, const char* fmt,
                               const std::vector<int>& params=std::vector<int>())
{
    return Formatted(mtx, Formatter::get(fmt), params);
}

template<typename _Tp> static inline Formatted format(const std::vector<Point_<_Tp> >& vec,
                                                      const char* fmt, const std::vector<int>& params=std::vector<int>())
{
    return Formatted(Mat(vec), Formatter::get(fmt), params);
}

template<typename _Tp> static inline Formatted format(const std::vector<Point3_<_Tp> >& vec,
                                                      const char* fmt, const std::vector<int>& params=std::vector<int>())
{
    return Formatted(Mat(vec), Formatter::get(fmt), params);
}

/** \brief prints Mat to the output stream in Matlab notation
 * use like
 @verbatim
 Mat my_mat = Mat::eye(3,3,CV_32F);
 std::cout << my_mat;
 @endverbatim
 */
static inline std::ostream& operator << (std::ostream& out, const Mat& mtx)
{
    Formatter::get()->write(out, mtx);
    return out;
}

/** \brief prints Mat to the output stream allows in the specified notation (see format)
 * use like
 @verbatim
 Mat my_mat = Mat::eye(3,3,CV_32F);
 std::cout << my_mat;
 @endverbatim
 */
static inline std::ostream& operator << (std::ostream& out, const Formatted& fmtd)
{
    fmtd.fmt->write(out, fmtd.mtx);
    return out;
}


template<typename _Tp> static inline std::ostream& operator << (std::ostream& out,
                                                                const std::vector<Point_<_Tp> >& vec)
{
    Formatter::get()->write(out, Mat(vec));
    return out;
}


template<typename _Tp> static inline std::ostream& operator << (std::ostream& out,
                                                                const std::vector<Point3_<_Tp> >& vec)
{
    Formatter::get()->write(out, Mat(vec));
    return out;
}


/** Writes a Matx to an output stream.
 */
template<typename _Tp, int m, int n> inline std::ostream& operator<<(std::ostream& out, const Matx<_Tp, m, n>& matx)
{
    out << cv::Mat(matx);
    return out;
}

/** Writes a point to an output stream in Matlab notation
 */
template<typename _Tp> inline std::ostream& operator<<(std::ostream& out, const Point_<_Tp>& p)
{
    out << "[" << p.x << ", " << p.y << "]";
    return out;
}

/** Writes a point to an output stream in Matlab notation
 */
template<typename _Tp> inline std::ostream& operator<<(std::ostream& out, const Point3_<_Tp>& p)
{
    out << "[" << p.x << ", " << p.y << ", " << p.z << "]";
    return out;
}

/** Writes a Vec to an output stream. Format example : [10, 20, 30]
 */
template<typename _Tp, int n> inline std::ostream& operator<<(std::ostream& out, const Vec<_Tp, n>& vec)
{
    out << "[";

    if(Vec<_Tp, n>::depth < CV_32F)
    {
        for (int i = 0; i < n - 1; ++i) {
            out << (int)vec[i] << ", ";
        }
        out << (int)vec[n-1] << "]";
    }
    else
    {
        for (int i = 0; i < n - 1; ++i) {
            out << vec[i] << ", ";
        }
        out << vec[n-1] << "]";
    }

    return out;
}

/** Writes a Size_ to an output stream. Format example : [640 x 480]
 */
template<typename _Tp> inline std::ostream& operator<<(std::ostream& out, const Size_<_Tp>& size)
{
    out << "[" << size.width << " x " << size.height << "]";
    return out;
}

/** Writes a Rect_ to an output stream. Format example : [640 x 480 from (10, 20)]
 */
template<typename _Tp> inline std::ostream& operator<<(std::ostream& out, const Rect_<_Tp>& rect)
{
    out << "[" << rect.width << " x " << rect.height << " from (" << rect.x << ", " << rect.y << ")]";
    return out;
}


template<typename _Tp> inline Ptr<_Tp> Algorithm::create(const String& name)
{
    return _create(name).ptr<_Tp>();
}

template<typename _Tp>
inline void Algorithm::set(const char* _name, const Ptr<_Tp>& value)
{
    Ptr<Algorithm> algo_ptr = value. template ptr<cv::Algorithm>();
    if (algo_ptr.empty()) {
        CV_Error( Error::StsUnsupportedFormat, "unknown/unsupported Ptr type of the second parameter of the method Algorithm::set");
    }
    info()->set(this, _name, ParamType<Algorithm>::type, &algo_ptr);
}

template<typename _Tp>
inline void Algorithm::set(const String& _name, const Ptr<_Tp>& value)
{
    this->set<_Tp>(_name.c_str(), value);
}

template<typename _Tp>
inline void Algorithm::setAlgorithm(const char* _name, const Ptr<_Tp>& value)
{
    Ptr<Algorithm> algo_ptr = value. template ptr<cv::Algorithm>();
    if (algo_ptr.empty()) {
        CV_Error( Error::StsUnsupportedFormat, "unknown/unsupported Ptr type of the second parameter of the method Algorithm::set");
    }
    info()->set(this, _name, ParamType<Algorithm>::type, &algo_ptr);
}

template<typename _Tp>
inline void Algorithm::setAlgorithm(const String& _name, const Ptr<_Tp>& value)
{
    this->set<_Tp>(_name.c_str(), value);
}

template<typename _Tp> inline typename ParamType<_Tp>::member_type Algorithm::get(const String& _name) const
{
    typename ParamType<_Tp>::member_type value;
    info()->get(this, _name.c_str(), ParamType<_Tp>::type, &value);
    return value;
}

template<typename _Tp> inline typename ParamType<_Tp>::member_type Algorithm::get(const char* _name) const
{
    typename ParamType<_Tp>::member_type value;
    info()->get(this, _name, ParamType<_Tp>::type, &value);
    return value;
}

template<typename _Tp, typename _Base> inline void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                  Ptr<_Tp>& value, bool readOnly, Ptr<_Tp> (Algorithm::*getter)(), void (Algorithm::*setter)(const Ptr<_Tp>&),
                  const String& help)
{
    //TODO: static assert: _Tp inherits from _Base
    addParam_(algo, parameter, ParamType<_Base>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

template<typename _Tp> inline void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter,
                  Ptr<_Tp>& value, bool readOnly, Ptr<_Tp> (Algorithm::*getter)(), void (Algorithm::*setter)(const Ptr<_Tp>&),
                  const String& help)
{
    //TODO: static assert: _Tp inherits from Algorithm
    addParam_(algo, parameter, ParamType<Algorithm>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

}

#ifdef _MSC_VER
# pragma warning(pop)
#endif

#endif
