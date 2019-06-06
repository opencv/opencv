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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
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

#ifndef OPENCV_CORE_OPERATIONS_HPP
#define OPENCV_CORE_OPERATIONS_HPP

#ifndef __cplusplus
#  error operations.hpp header must be compiled as C++
#endif

#include <cstdio>

//! @cond IGNORED

namespace cv
{

////////////////////////////// Matx methods depending on core API /////////////////////////////

namespace internal
{

template<typename _Tp, int m, int n> struct Matx_FastInvOp
{
    bool operator()(const Matx<_Tp, m, n>& a, Matx<_Tp, n, m>& b, int method) const
    {
        return invert(a, b, method) != 0;
    }
};

template<typename _Tp, int m> struct Matx_FastInvOp<_Tp, m, m>
{
    bool operator()(const Matx<_Tp, m, m>& a, Matx<_Tp, m, m>& b, int method) const
    {
        if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
        {
            Matx<_Tp, m, m> temp = a;

            // assume that b is all 0's on input => make it a unity matrix
            for (int i = 0; i < m; i++)
                b(i, i) = (_Tp)1;

            if (method == DECOMP_CHOLESKY)
                return Cholesky(temp.val, m*sizeof(_Tp), m, b.val, m*sizeof(_Tp), m);

            return LU(temp.val, m*sizeof(_Tp), m, b.val, m*sizeof(_Tp), m) != 0;
        }
        else
        {
            return invert(a, b, method) != 0;
        }
    }
};

template<typename _Tp> struct Matx_FastInvOp<_Tp, 2, 2>
{
    bool operator()(const Matx<_Tp, 2, 2>& a, Matx<_Tp, 2, 2>& b, int /*method*/) const
    {
        _Tp d = (_Tp)determinant(a);
        if (d == 0)
            return false;
        d = 1/d;
        b(1,1) = a(0,0)*d;
        b(0,0) = a(1,1)*d;
        b(0,1) = -a(0,1)*d;
        b(1,0) = -a(1,0)*d;
        return true;
    }
};

template<typename _Tp> struct Matx_FastInvOp<_Tp, 3, 3>
{
    bool operator()(const Matx<_Tp, 3, 3>& a, Matx<_Tp, 3, 3>& b, int /*method*/) const
    {
        _Tp d = (_Tp)determinant(a);
        if (d == 0)
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


template<typename _Tp, int m, int l, int n> struct Matx_FastSolveOp
{
    bool operator()(const Matx<_Tp, m, l>& a, const Matx<_Tp, m, n>& b,
                    Matx<_Tp, l, n>& x, int method) const
    {
        return cv::solve(a, b, x, method);
    }
};

template<typename _Tp, int m, int n> struct Matx_FastSolveOp<_Tp, m, m, n>
{
    bool operator()(const Matx<_Tp, m, m>& a, const Matx<_Tp, m, n>& b,
                    Matx<_Tp, m, n>& x, int method) const
    {
        if (method == DECOMP_LU || method == DECOMP_CHOLESKY)
        {
            Matx<_Tp, m, m> temp = a;
            x = b;
            if( method == DECOMP_CHOLESKY )
                return Cholesky(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n);

            return LU(temp.val, m*sizeof(_Tp), m, x.val, n*sizeof(_Tp), n) != 0;
        }
        else
        {
            return cv::solve(a, b, x, method);
        }
    }
};

template<typename _Tp> struct Matx_FastSolveOp<_Tp, 2, 2, 1>
{
    bool operator()(const Matx<_Tp, 2, 2>& a, const Matx<_Tp, 2, 1>& b,
                    Matx<_Tp, 2, 1>& x, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if (d == 0)
            return false;
        d = 1/d;
        x(0) = (b(0)*a(1,1) - b(1)*a(0,1))*d;
        x(1) = (b(1)*a(0,0) - b(0)*a(1,0))*d;
        return true;
    }
};

template<typename _Tp> struct Matx_FastSolveOp<_Tp, 3, 3, 1>
{
    bool operator()(const Matx<_Tp, 3, 3>& a, const Matx<_Tp, 3, 1>& b,
                    Matx<_Tp, 3, 1>& x, int) const
    {
        _Tp d = (_Tp)determinant(a);
        if (d == 0)
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
Matx<_Tp, n, m> Matx<_Tp, m, n>::inv(int method, bool *p_is_ok /*= NULL*/) const
{
    Matx<_Tp, n, m> b;
    bool ok = cv::internal::Matx_FastInvOp<_Tp, m, n>()(*this, b, method);
    if (p_is_ok) *p_is_ok = ok;
    return ok ? b : Matx<_Tp, n, m>::zeros();
}

template<typename _Tp, int m, int n> template<int l> inline
Matx<_Tp, n, l> Matx<_Tp, m, n>::solve(const Matx<_Tp, m, l>& rhs, int method) const
{
    Matx<_Tp, n, l> x;
    bool ok = cv::internal::Matx_FastSolveOp<_Tp, m, n, l>()(*this, rhs, x, method);
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

#define CV_MAT_AUG_OPERATOR_TN(op, cvop, A)                                \
    template<typename _Tp, int m, int n> static inline A& operator op (A& a, const Matx<_Tp,m,n>& b) { cvop; return a; } \
    template<typename _Tp, int m, int n> static inline const A& operator op (const A& a, const Matx<_Tp,m,n>& b) { cvop; return a; }

CV_MAT_AUG_OPERATOR  (+=, cv::add(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (+=, cv::add(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(+=, cv::add(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR_TN(+=, cv::add(a,Mat(b),a), Mat)
CV_MAT_AUG_OPERATOR_TN(+=, cv::add(a,Mat(b),a), Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (-=, cv::subtract(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (-=, cv::subtract(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(-=, cv::subtract(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR_TN(-=, cv::subtract(a,Mat(b),a), Mat)
CV_MAT_AUG_OPERATOR_TN(-=, cv::subtract(a,Mat(b),a), Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat, Mat)
CV_MAT_AUG_OPERATOR_T(*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(*=, cv::gemm(a, b, 1, Mat(), 0, a, 0), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR  (*=, a.convertTo(a, -1, b), Mat, double)
CV_MAT_AUG_OPERATOR_T(*=, a.convertTo(a, -1, b), Mat_<_Tp>, double)
CV_MAT_AUG_OPERATOR_TN(*=, cv::gemm(a, Mat(b), 1, Mat(), 0, a, 0), Mat)
CV_MAT_AUG_OPERATOR_TN(*=, cv::gemm(a, Mat(b), 1, Mat(), 0, a, 0), Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (/=, cv::divide(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR_T(/=, cv::divide(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(/=, cv::divide(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR  (/=, a.convertTo((Mat&)a, -1, 1./b), Mat, double)
CV_MAT_AUG_OPERATOR_T(/=, a.convertTo((Mat&)a, -1, 1./b), Mat_<_Tp>, double)
CV_MAT_AUG_OPERATOR_TN(/=, cv::divide(a, Mat(b), a), Mat)
CV_MAT_AUG_OPERATOR_TN(/=, cv::divide(a, Mat(b), a), Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (&=, cv::bitwise_and(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (&=, cv::bitwise_and(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(&=, cv::bitwise_and(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR_TN(&=, cv::bitwise_and(a, Mat(b), a), Mat)
CV_MAT_AUG_OPERATOR_TN(&=, cv::bitwise_and(a, Mat(b), a), Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (|=, cv::bitwise_or(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (|=, cv::bitwise_or(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(|=, cv::bitwise_or(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR_TN(|=, cv::bitwise_or(a, Mat(b), a), Mat)
CV_MAT_AUG_OPERATOR_TN(|=, cv::bitwise_or(a, Mat(b), a), Mat_<_Tp>)

CV_MAT_AUG_OPERATOR  (^=, cv::bitwise_xor(a,b,a), Mat, Mat)
CV_MAT_AUG_OPERATOR  (^=, cv::bitwise_xor(a,b,a), Mat, Scalar)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Mat)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Scalar)
CV_MAT_AUG_OPERATOR_T(^=, cv::bitwise_xor(a,b,a), Mat_<_Tp>, Mat_<_Tp>)
CV_MAT_AUG_OPERATOR_TN(^=, cv::bitwise_xor(a, Mat(b), a), Mat)
CV_MAT_AUG_OPERATOR_TN(^=, cv::bitwise_xor(a, Mat(b), a), Mat_<_Tp>)

#undef CV_MAT_AUG_OPERATOR_TN
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

inline bool RNG::operator ==(const RNG& other) const { return state == other.state; }

inline unsigned RNG::next()
{
    state = (uint64)(unsigned)state* /*CV_RNG_COEFF*/ 4164903690U + (unsigned)(state >> 32);
    return (unsigned)state;
}

//! returns the next uniformly-distributed random number of the specified type
template<typename _Tp> static inline _Tp randu()
{
  return (_Tp)theRNG();
}

///////////////////////////////// Formatted string generation /////////////////////////////////

/** @brief Returns a text string formatted using the printf-like expression.

The function acts like sprintf but forms and returns an STL string. It can be used to form an error
message in the Exception constructor.
@param fmt printf-compatible formatting specifiers.
 */
CV_EXPORTS String format( const char* fmt, ... );

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

static inline
int print(const UMat& mtx, FILE* stream = stdout)
{
    return print(Formatter::get()->format(mtx.getMat(ACCESS_READ)), stream);
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
    return print(Formatter::get()->format(cv::Mat(matx)), stream);
}

//! @endcond

/****************************************************************************************\
*                                  Auxiliary algorithms                                  *
\****************************************************************************************/

/** @brief Splits an element set into equivalency classes.

The generic function partition implements an \f$O(N^2)\f$ algorithm for splitting a set of \f$N\f$ elements
into one or more equivalency classes, as described in
<http://en.wikipedia.org/wiki/Disjoint-set_data_structure> . The function returns the number of
equivalency classes.
@param _vec Set of elements stored as a vector.
@param labels Output vector of labels. It contains as many elements as vec. Each label labels[i] is
a 0-based cluster index of `vec[i]`.
@param predicate Equivalence predicate (pointer to a boolean function of two arguments or an
instance of the class that has the method bool operator()(const _Tp& a, const _Tp& b) ). The
predicate returns true when the elements are certainly in the same class, and returns false if they
may or may not be in the same class.
@ingroup core_cluster
*/
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

} // cv

#endif
