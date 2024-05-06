/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#ifndef OPENCV_CORE_MATX_HPP
#define OPENCV_CORE_MATX_HPP

#ifndef __cplusplus
#  error matx.hpp header must be compiled as C++
#endif

#include "opencv2/core/cvdef.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/saturate.hpp"

#include <initializer_list>

namespace cv
{

//! @addtogroup core_basic
//! @{

//! @cond IGNORED
// FIXIT Remove this (especially CV_EXPORTS modifier)
struct CV_EXPORTS Matx_AddOp { Matx_AddOp() {} Matx_AddOp(const Matx_AddOp&) {} };
struct CV_EXPORTS Matx_SubOp { Matx_SubOp() {} Matx_SubOp(const Matx_SubOp&) {} };
struct CV_EXPORTS Matx_ScaleOp { Matx_ScaleOp() {} Matx_ScaleOp(const Matx_ScaleOp&) {} };
struct CV_EXPORTS Matx_MulOp { Matx_MulOp() {} Matx_MulOp(const Matx_MulOp&) {} };
struct CV_EXPORTS Matx_DivOp { Matx_DivOp() {} Matx_DivOp(const Matx_DivOp&) {} };
struct CV_EXPORTS Matx_MatMulOp { Matx_MatMulOp() {} Matx_MatMulOp(const Matx_MatMulOp&) {} };
struct CV_EXPORTS Matx_TOp { Matx_TOp() {} Matx_TOp(const Matx_TOp&) {} };
//! @endcond

////////////////////////////// Small Matrix ///////////////////////////

/** @brief Template class for small matrices whose type and size are known at compilation time

If you need a more flexible type, use Mat . The elements of the matrix M are accessible using the
M(i,j) notation. Most of the common matrix operations (see also @ref MatrixExpressions ) are
available. To do an operation on Matx that is not implemented, you can easily convert the matrix to
Mat and backwards:
@code{.cpp}
    Matx33f m(1, 2, 3,
              4, 5, 6,
              7, 8, 9);
    cout << sum(Mat(m*m.t())) << endl;
@endcode
Except of the plain constructor which takes a list of elements, Matx can be initialized from a C-array:
@code{.cpp}
    float values[] = { 1, 2, 3};
    Matx31f m(values);
@endcode
In case if C++11 features are available, std::initializer_list can be also used to initialize Matx:
@code{.cpp}
    Matx31f m = { 1, 2, 3};
@endcode
 */
template<typename _Tp, int m, int n> class Matx
{
public:
    enum {
           rows     = m,
           cols     = n,
           channels = rows*cols,
#ifdef OPENCV_TRAITS_ENABLE_DEPRECATED
           depth    = traits::Type<_Tp>::value,
           type     = CV_MAKETYPE(depth, channels),
#endif
           shortdim = (m < n ? m : n)
         };

    typedef _Tp                           value_type;
    typedef Matx<_Tp, m, n>               mat_type;
    typedef Matx<_Tp, shortdim, 1> diag_type;

    //! default constructor
    Matx();

    explicit Matx(_Tp v0); //!< 1x1 matrix
    Matx(_Tp v0, _Tp v1); //!< 1x2 or 2x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2); //!< 1x3 or 3x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 1x4, 2x2 or 4x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 1x5 or 5x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 1x6, 2x3, 3x2 or 6x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 1x7 or 7x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 1x8, 2x4, 4x2 or 8x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 1x9, 3x3 or 9x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 1x10, 2x5 or 5x2 or 10x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11); //!< 1x12, 2x6, 3x4, 4x3, 6x2 or 12x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11,
         _Tp v12, _Tp v13); //!< 1x14, 2x7, 7x2 or 14x1 matrix
    Matx(_Tp v0, _Tp v1, _Tp v2, _Tp v3,
         _Tp v4, _Tp v5, _Tp v6, _Tp v7,
         _Tp v8, _Tp v9, _Tp v10, _Tp v11,
         _Tp v12, _Tp v13, _Tp v14, _Tp v15); //!< 1x16, 4x4 or 16x1 matrix
    explicit Matx(const _Tp* vals); //!< initialize from a plain array

    Matx(std::initializer_list<_Tp>); //!< initialize from an initializer list

    CV_NODISCARD_STD static Matx all(_Tp alpha);
    CV_NODISCARD_STD static Matx zeros();
    CV_NODISCARD_STD static Matx ones();
    CV_NODISCARD_STD static Matx eye();
    CV_NODISCARD_STD static Matx diag(const diag_type& d);
    /** @brief Generates uniformly distributed random numbers
    @param a Range boundary.
    @param b The other range boundary (boundaries don't have to be ordered, the lower boundary is inclusive,
    the upper one is exclusive).
     */
    CV_NODISCARD_STD static Matx randu(_Tp a, _Tp b);
    /** @brief Generates normally distributed random numbers
    @param a Mean value.
    @param b Standard deviation.
     */
    CV_NODISCARD_STD static Matx randn(_Tp a, _Tp b);

    //! dot product computed with the default precision
    _Tp dot(const Matx<_Tp, m, n>& v) const;

    //! dot product computed in double-precision arithmetics
    double ddot(const Matx<_Tp, m, n>& v) const;

    //! conversion to another data type
    template<typename T2> operator Matx<T2, m, n>() const;

    //! change the matrix shape
    template<int m1, int n1> Matx<_Tp, m1, n1> reshape() const;

    //! extract part of the matrix
    template<int m1, int n1> Matx<_Tp, m1, n1> get_minor(int base_row, int base_col) const;

    //! extract the matrix row
    Matx<_Tp, 1, n> row(int i) const;

    //! extract the matrix column
    Matx<_Tp, m, 1> col(int i) const;

    //! extract the matrix diagonal
    diag_type diag() const;

    //! transpose the matrix
    Matx<_Tp, n, m> t() const;

    //! invert the matrix
    Matx<_Tp, n, m> inv(int method=DECOMP_LU, bool *p_is_ok = NULL) const;

    //! solve linear system
    template<int l> Matx<_Tp, n, l> solve(const Matx<_Tp, m, l>& rhs, int flags=DECOMP_LU) const;
    Vec<_Tp, n> solve(const Vec<_Tp, m>& rhs, int method) const;

    //! multiply two matrices element-wise
    Matx<_Tp, m, n> mul(const Matx<_Tp, m, n>& a) const;

    //! divide two matrices element-wise
    Matx<_Tp, m, n> div(const Matx<_Tp, m, n>& a) const;

    //! element access
    const _Tp& operator ()(int row, int col) const;
    _Tp& operator ()(int row, int col);

    //! 1D element access
    const _Tp& operator ()(int i) const;
    _Tp& operator ()(int i);

    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_AddOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_SubOp);
    template<typename _T2> Matx(const Matx<_Tp, m, n>& a, _T2 alpha, Matx_ScaleOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_MulOp);
    Matx(const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b, Matx_DivOp);
    template<int l> Matx(const Matx<_Tp, m, l>& a, const Matx<_Tp, l, n>& b, Matx_MatMulOp);
    Matx(const Matx<_Tp, n, m>& a, Matx_TOp);

    _Tp val[m*n]; ///< matrix elements
};

typedef Matx<float, 1, 2> Matx12f;
typedef Matx<double, 1, 2> Matx12d;
typedef Matx<float, 1, 3> Matx13f;
typedef Matx<double, 1, 3> Matx13d;
typedef Matx<float, 1, 4> Matx14f;
typedef Matx<double, 1, 4> Matx14d;
typedef Matx<float, 1, 6> Matx16f;
typedef Matx<double, 1, 6> Matx16d;

typedef Matx<float, 2, 1> Matx21f;
typedef Matx<double, 2, 1> Matx21d;
typedef Matx<float, 3, 1> Matx31f;
typedef Matx<double, 3, 1> Matx31d;
typedef Matx<float, 4, 1> Matx41f;
typedef Matx<double, 4, 1> Matx41d;
typedef Matx<float, 6, 1> Matx61f;
typedef Matx<double, 6, 1> Matx61d;

typedef Matx<float, 2, 2> Matx22f;
typedef Matx<double, 2, 2> Matx22d;
typedef Matx<float, 2, 3> Matx23f;
typedef Matx<double, 2, 3> Matx23d;
typedef Matx<float, 3, 2> Matx32f;
typedef Matx<double, 3, 2> Matx32d;

typedef Matx<float, 3, 3> Matx33f;
typedef Matx<double, 3, 3> Matx33d;

typedef Matx<float, 3, 4> Matx34f;
typedef Matx<double, 3, 4> Matx34d;
typedef Matx<float, 4, 3> Matx43f;
typedef Matx<double, 4, 3> Matx43d;

typedef Matx<float, 4, 4> Matx44f;
typedef Matx<double, 4, 4> Matx44d;
typedef Matx<float, 6, 6> Matx66f;
typedef Matx<double, 6, 6> Matx66d;

template<typename _Tp, int m> static inline
double determinant(const Matx<_Tp, m, m>& a);

template<typename _Tp, int m, int n> static inline
double trace(const Matx<_Tp, m, n>& a);

template<typename _Tp, int m, int n> static inline
double norm(const Matx<_Tp, m, n>& M);

template<typename _Tp, int m, int n> static inline
double norm(const Matx<_Tp, m, n>& M, int normType);

template<typename _Tp1, typename _Tp2, int m, int n> static inline
Matx<_Tp1, m, n>& operator += (Matx<_Tp1, m, n>& a, const Matx<_Tp2, m, n>& b);

template<typename _Tp1, typename _Tp2, int m, int n> static inline
Matx<_Tp1, m, n>& operator -= (Matx<_Tp1, m, n>& a, const Matx<_Tp2, m, n>& b);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator + (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator - (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator *= (Matx<_Tp, m, n>& a, int alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator *= (Matx<_Tp, m, n>& a, float alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator *= (Matx<_Tp, m, n>& a, double alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, n>& a, int alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, n>& a, float alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, n>& a, double alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (int alpha, const Matx<_Tp, m, n>& a);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (float alpha, const Matx<_Tp, m, n>& a);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator * (double alpha, const Matx<_Tp, m, n>& a);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator /= (Matx<_Tp, m, n>& a, float alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n>& operator /= (Matx<_Tp, m, n>& a, double alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator / (const Matx<_Tp, m, n>& a, float alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator / (const Matx<_Tp, m, n>& a, double alpha);

template<typename _Tp, int m, int n> static inline
Matx<_Tp, m, n> operator - (const Matx<_Tp, m, n>& a);

template<typename _Tp, int m, int n, int l> static inline
Matx<_Tp, m, n> operator * (const Matx<_Tp, m, l>& a, const Matx<_Tp, l, n>& b);

template<typename _Tp, int m, int n> static inline
Vec<_Tp, m> operator * (const Matx<_Tp, m, n>& a, const Vec<_Tp, n>& b);

template<typename _Tp, int m, int n> static inline
bool operator == (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b);

template<typename _Tp, int m, int n> static inline
bool operator != (const Matx<_Tp, m, n>& a, const Matx<_Tp, m, n>& b);


/////////////////////// Vec (used as element of multi-channel images /////////////////////

/** @brief Template class for short numerical vectors, a partial case of Matx

This template class represents short numerical vectors (of 1, 2, 3, 4 ... elements) on which you
can perform basic arithmetical operations, access individual elements using [] operator etc. The
vectors are allocated on stack, as opposite to std::valarray, std::vector, cv::Mat etc., which
elements are dynamically allocated in the heap.

The template takes 2 parameters:
@tparam _Tp element type
@tparam cn the number of elements

In addition to the universal notation like Vec<float, 3>, you can use shorter aliases
for the most popular specialized variants of Vec, e.g. Vec3f ~ Vec<float, 3>.

It is possible to convert Vec\<T,2\> to/from Point_, Vec\<T,3\> to/from Point3_ , and Vec\<T,4\>
to CvScalar or Scalar_. Use operator[] to access the elements of Vec.

All the expected vector operations are also implemented:
-   v1 = v2 + v3
-   v1 = v2 - v3
-   v1 = v2 \* scale
-   v1 = scale \* v2
-   v1 = -v2
-   v1 += v2 and other augmenting operations
-   v1 == v2, v1 != v2
-   norm(v1) (euclidean norm)
The Vec class is commonly used to describe pixel types of multi-channel arrays. See Mat for details.
*/
template<typename _Tp, int cn> class Vec : public Matx<_Tp, cn, 1>
{
public:
    typedef _Tp value_type;
    enum {
           channels = cn,
#ifdef OPENCV_TRAITS_ENABLE_DEPRECATED
           depth    = Matx<_Tp, cn, 1>::depth,
           type     = CV_MAKETYPE(depth, channels),
#endif
           _dummy_enum_finalizer = 0
         };

    //! default constructor
    Vec();

    Vec(_Tp v0); //!< 1-element vector constructor
    Vec(_Tp v0, _Tp v1); //!< 2-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2); //!< 3-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3); //!< 4-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4); //!< 5-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5); //!< 6-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6); //!< 7-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7); //!< 8-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8); //!< 9-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9); //!< 10-element vector constructor
    Vec(_Tp v0, _Tp v1, _Tp v2, _Tp v3, _Tp v4, _Tp v5, _Tp v6, _Tp v7, _Tp v8, _Tp v9, _Tp v10, _Tp v11, _Tp v12, _Tp v13); //!< 14-element vector constructor
    explicit Vec(const _Tp* values);

    Vec(std::initializer_list<_Tp>);

    Vec(const Vec<_Tp, cn>& v);

    static Vec all(_Tp alpha);
    static Vec ones();
    static Vec randn(_Tp a, _Tp b);
    static Vec randu(_Tp a, _Tp b);
    static Vec zeros();
    static Vec diag(_Tp alpha) = delete;
    static Vec eye() = delete;

    //! per-element multiplication
    Vec mul(const Vec<_Tp, cn>& v) const;

    //! conjugation (makes sense for complex numbers and quaternions)
    Vec conj() const;

    /*!
      cross product of the two 3D vectors.

      For other dimensionalities the exception is raised
    */
    Vec cross(const Vec& v) const;
    //! conversion to another data type
    template<typename T2> operator Vec<T2, cn>() const;

    /*! element access */
    const _Tp& operator [](int i) const;
    _Tp& operator[](int i);
    const _Tp& operator ()(int i) const;
    _Tp& operator ()(int i);

    Vec<_Tp, cn>& operator=(const Vec<_Tp, cn>& rhs) = default;

    Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_AddOp);
    Vec(const Matx<_Tp, cn, 1>& a, const Matx<_Tp, cn, 1>& b, Matx_SubOp);
    template<typename _T2> Vec(const Matx<_Tp, cn, 1>& a, _T2 alpha, Matx_ScaleOp);
};

/** @name Shorter aliases for the most popular specializations of Vec<T,n>
  @{
*/
typedef Vec<uchar, 2> Vec2b;
typedef Vec<uchar, 3> Vec3b;
typedef Vec<uchar, 4> Vec4b;

typedef Vec<short, 2> Vec2s;
typedef Vec<short, 3> Vec3s;
typedef Vec<short, 4> Vec4s;

typedef Vec<ushort, 2> Vec2w;
typedef Vec<ushort, 3> Vec3w;
typedef Vec<ushort, 4> Vec4w;

typedef Vec<int, 2> Vec2i;
typedef Vec<int, 3> Vec3i;
typedef Vec<int, 4> Vec4i;
typedef Vec<int, 6> Vec6i;
typedef Vec<int, 8> Vec8i;

typedef Vec<float, 2> Vec2f;
typedef Vec<float, 3> Vec3f;
typedef Vec<float, 4> Vec4f;
typedef Vec<float, 6> Vec6f;

typedef Vec<double, 2> Vec2d;
typedef Vec<double, 3> Vec3d;
typedef Vec<double, 4> Vec4d;
typedef Vec<double, 6> Vec6d;
/** @} */

template<typename _Tp, int cn> inline
Vec<_Tp, cn> normalize(const Vec<_Tp, cn>& v);

template<typename _Tp1, typename _Tp2, int cn> static inline
Vec<_Tp1, cn>& operator += (Vec<_Tp1, cn>& a, const Vec<_Tp2, cn>& b);

template<typename _Tp1, typename _Tp2, int cn> static inline
Vec<_Tp1, cn>& operator -= (Vec<_Tp1, cn>& a, const Vec<_Tp2, cn>& b);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator + (const Vec<_Tp, cn>& a, const Vec<_Tp, cn>& b);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator - (const Vec<_Tp, cn>& a, const Vec<_Tp, cn>& b);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator *= (Vec<_Tp, cn>& a, int alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator *= (Vec<_Tp, cn>& a, float alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator *= (Vec<_Tp, cn>& a, double alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator /= (Vec<_Tp, cn>& a, int alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator /= (Vec<_Tp, cn>& a, float alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn>& operator /= (Vec<_Tp, cn>& a, double alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (const Vec<_Tp, cn>& a, int alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (int alpha, const Vec<_Tp, cn>& a);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (const Vec<_Tp, cn>& a, float alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (float alpha, const Vec<_Tp, cn>& a);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (const Vec<_Tp, cn>& a, double alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator * (double alpha, const Vec<_Tp, cn>& a);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator / (const Vec<_Tp, cn>& a, int alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator / (const Vec<_Tp, cn>& a, float alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator / (const Vec<_Tp, cn>& a, double alpha);

template<typename _Tp, int cn> static inline
Vec<_Tp, cn> operator - (const Vec<_Tp, cn>& a);

template<typename _Tp> inline
Vec<_Tp, 4> operator * (const Vec<_Tp, 4>& v1, const Vec<_Tp, 4>& v2);

template<typename _Tp> inline
Vec<_Tp, 4>& operator *= (Vec<_Tp, 4>& v1, const Vec<_Tp, 4>& v2);

//! @} core_basic

} // cv

#include "opencv2/core/matx.inl.hpp"

#endif // OPENCV_CORE_MATX_HPP
