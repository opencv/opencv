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

#include <cstdio>

namespace cv
{

////////////////////////////// Matx methods depending on core API /////////////////////////////

namespace internal
{

template<typename _Tp, int m> struct Matx_FastInvOp
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

template<typename _Tp> struct Matx_FastInvOp<_Tp, 2>
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

template<typename _Tp> struct Matx_FastInvOp<_Tp, 3>
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


template<typename _Tp, int m, int n> struct Matx_FastSolveOp
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

template<typename _Tp> struct Matx_FastSolveOp<_Tp, 2, 1>
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

template<typename _Tp> struct Matx_FastSolveOp<_Tp, 3, 1>
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



////////////////////////// Augmenting algebraic & logical operations //////////////////////////

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



///////////////////////////////////////////// SVD /////////////////////////////////////////////

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



/////////////////////////////////// Multiply-with-Carry RNG ///////////////////////////////////

inline RNG::RNG()              { state = 0xffffffff; }
inline RNG::RNG(uint64 _state) { state = _state ? _state : 0xffffffff; }

inline RNG::operator uchar()    { return (uchar)next(); }
inline RNG::operator schar()    { return (schar)next(); }
inline RNG::operator ushort()   { return (ushort)next(); }
inline RNG::operator short()    { return (short)next(); }
inline RNG::operator int()      { return (int)next(); }
inline RNG::operator unsigned() { return next(); }
inline RNG::operator float()    { return next()*2.3283064365386962890625e-10f; }
inline RNG::operator double()   { unsigned t = next(); return (((uint64)t << 32) | next()) * 5.4210108624275221700372640043497e-20; }

inline unsigned RNG::operator ()(unsigned N) { return (unsigned)uniform(0,N); }
inline unsigned RNG::operator ()()           { return next(); }

inline int    RNG::uniform(int a, int b)       { return a == b ? a : (int)(next() % (b - a) + a); }
inline float  RNG::uniform(float a, float b)   { return ((float)*this)*(b - a) + a; }
inline double RNG::uniform(double a, double b) { return ((double)*this)*(b - a) + a; }

inline unsigned RNG::next()
{
    state = (uint64)(unsigned)state* /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
    return (unsigned)state;
}



///////////////////////////////////////// LineIterator ////////////////////////////////////////

inline
uchar* LineIterator::operator *()
{
    return ptr;
}

inline
LineIterator& LineIterator::operator ++()
{
    int mask = err < 0 ? -1 : 0;
    err += minusDelta + (plusDelta & mask);
    ptr += minusStep + (plusStep & mask);
    return *this;
}

inline
LineIterator LineIterator::operator ++(int)
{
    LineIterator it = *this;
    ++(*this);
    return it;
}

inline
Point LineIterator::pos() const
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



///////////////////////////////// Formatted output of cv::Mat /////////////////////////////////

static inline
Ptr<Formatted> format(InputArray mtx, int fmt)
{
    return Formatter::get(fmt)->format(mtx.getMat());
}

static inline
int print(Ptr<Formatted> fmtd, FILE* stream = stdout)
{
    int written = 0;
    fmtd->reset();
    for(const char* str = fmtd->next(); str; str = fmtd->next())
        written += fputs(str, stream);

    return written;
}

static inline
int print(const Mat& mtx, FILE* stream = stdout)
{
    return print(Formatter::get()->format(mtx), stream);
}

template<typename _Tp> static inline
int print(const std::vector<Point_<_Tp> >& vec, FILE* stream = stdout)
{
    return print(Formatter::get()->format(Mat(vec)), stream);
}

template<typename _Tp> static inline
int print(const std::vector<Point3_<_Tp> >& vec, FILE* stream = stdout)
{
    return print(Formatter::get()->format(Mat(vec)), stream);
}

template<typename _Tp, int m, int n> static inline
int print(const Matx<_Tp, m, n>& matx, FILE* stream = stdout)
{
    return print(Formatter::get()->format(matx), stream);
}



////////////////////////////////////////// Algorithm //////////////////////////////////////////

template<typename _Tp> inline
Ptr<_Tp> Algorithm::create(const String& name)
{
    return _create(name).dynamicCast<_Tp>();
}

template<typename _Tp> inline
void Algorithm::set(const char* _name, const Ptr<_Tp>& value)
{
    Ptr<Algorithm> algo_ptr = value. template dynamicCast<cv::Algorithm>();
    if (!algo_ptr) {
        CV_Error( Error::StsUnsupportedFormat, "unknown/unsupported Ptr type of the second parameter of the method Algorithm::set");
    }
    info()->set(this, _name, ParamType<Algorithm>::type, &algo_ptr);
}

template<typename _Tp> inline
void Algorithm::set(const String& _name, const Ptr<_Tp>& value)
{
    this->set<_Tp>(_name.c_str(), value);
}

template<typename _Tp> inline
void Algorithm::setAlgorithm(const char* _name, const Ptr<_Tp>& value)
{
    Ptr<Algorithm> algo_ptr = value. template ptr<cv::Algorithm>();
    if (!algo_ptr) {
        CV_Error( Error::StsUnsupportedFormat, "unknown/unsupported Ptr type of the second parameter of the method Algorithm::set");
    }
    info()->set(this, _name, ParamType<Algorithm>::type, &algo_ptr);
}

template<typename _Tp> inline
void Algorithm::setAlgorithm(const String& _name, const Ptr<_Tp>& value)
{
    this->set<_Tp>(_name.c_str(), value);
}

template<typename _Tp> inline
typename ParamType<_Tp>::member_type Algorithm::get(const String& _name) const
{
    typename ParamType<_Tp>::member_type value;
    info()->get(this, _name.c_str(), ParamType<_Tp>::type, &value);
    return value;
}

template<typename _Tp> inline
typename ParamType<_Tp>::member_type Algorithm::get(const char* _name) const
{
    typename ParamType<_Tp>::member_type value;
    info()->get(this, _name, ParamType<_Tp>::type, &value);
    return value;
}

template<typename _Tp, typename _Base> inline
void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter, Ptr<_Tp>& value, bool readOnly,
                             Ptr<_Tp> (Algorithm::*getter)(), void (Algorithm::*setter)(const Ptr<_Tp>&),
                             const String& help)
{
    //TODO: static assert: _Tp inherits from _Base
    addParam_(algo, parameter, ParamType<_Base>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}

template<typename _Tp> inline
void AlgorithmInfo::addParam(Algorithm& algo, const char* parameter, Ptr<_Tp>& value, bool readOnly,
                             Ptr<_Tp> (Algorithm::*getter)(), void (Algorithm::*setter)(const Ptr<_Tp>&),
                             const String& help)
{
    //TODO: static assert: _Tp inherits from Algorithm
    addParam_(algo, parameter, ParamType<Algorithm>::type, &value, readOnly,
              (Algorithm::Getter)getter, (Algorithm::Setter)setter, help);
}



} // cv

#endif
