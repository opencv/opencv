//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// 2x2, 3x3, and 4x4 transformation matrix templates
//

#ifndef INCLUDED_IMATHMATRIX_H
#define INCLUDED_IMATHMATRIX_H

#include "ImathExport.h"
#include "ImathNamespace.h"

#include "ImathFun.h"
#include "ImathPlatform.h"
#include "ImathShear.h"
#include "ImathVec.h"

#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string.h>

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// suppress exception specification warnings
#    pragma warning(disable : 4290)
#endif

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

/// Enum used to indicate uninitialized construction of Matrix22,
/// Matrix33, Matrix44
enum IMATH_EXPORT_ENUM Uninitialized
{
    UNINITIALIZED
};

///
/// 2x2 transformation matrix
///

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Matrix22
{
  public:

    /// @{
    /// @name Direct access to elements
    
    /// Matrix elements
    T x[2][2];

    /// @}
    
    /// Row access
    IMATH_HOSTDEVICE T* operator[] (int i) IMATH_NOEXCEPT;

    /// Row access
    IMATH_HOSTDEVICE const T* operator[] (int i) const IMATH_NOEXCEPT;

    /// @{
    ///	@name Constructors and Assignment

    /// Uninitialized
    IMATH_HOSTDEVICE Matrix22 (Uninitialized) IMATH_NOEXCEPT {}

    /// Default constructor: initialize to identity
    ///
    ///     1 0
    ///     0 1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22() IMATH_NOEXCEPT;

    /// Initialize to scalar constant:
    ///
    ///     a a
    ///     a a
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22 (T a) IMATH_NOEXCEPT;

    /// Construct from 2x2 array:
    ///
    ///     a[0][0] a[0][1]
    ///     a[1][0] a[1][1]
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22 (const T a[2][2]) IMATH_NOEXCEPT;
    /// Construct from given scalar values:
    ///
    ///     a b
    ///     c d
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22 (T a, T b, T c, T d) IMATH_NOEXCEPT;

    /// Copy constructor
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22 (const Matrix22& v) IMATH_NOEXCEPT;

    /// Construct from Matrix22 of another base type
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 explicit Matrix22 (const Matrix22<S>& v) IMATH_NOEXCEPT;

    /// Assignment
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator= (const Matrix22& v) IMATH_NOEXCEPT;

    /// Assignment from scalar
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator= (T a) IMATH_NOEXCEPT;

    /// Destructor
    ~Matrix22() IMATH_NOEXCEPT = default;

    /// @}

#if IMATH_FOREIGN_VECTOR_INTEROP
    /// @{
    /// @name Interoperability with other matrix types
    ///
    /// Construction and assignment are allowed from other classes that
    /// appear to be equivalent matrix types, provided that they support
    /// double-subscript (i.e., `m[j][i]`) giving the same type as the
    /// elements of this matrix, and their total size appears to be the
    /// right number of matrix elements.
    ///
    /// This functionality is disabled for gcc 4.x, which seems to have a
    /// compiler bug that results in spurious errors. It can also be
    /// disabled by defining IMATH_FOREIGN_VECTOR_INTEROP to be 0 prior to
    /// including any Imath header files.
    ///
    template<typename M, IMATH_ENABLE_IF(has_double_subscript<M,T,2,2>::value)>
    IMATH_HOSTDEVICE explicit Matrix22 (const M& m)
        : Matrix22(T(m[0][0]), T(m[0][1]), T(m[1][0]), T(m[1][1]))
    { }

    template<typename M, IMATH_ENABLE_IF(has_double_subscript<M,T,2,2>::value)>
    IMATH_HOSTDEVICE const Matrix22& operator= (const M& m)
    {
        *this = Matrix22(T(m[0][0]), T(m[0][1]), T(m[1][0]), T(m[1][1]));
        return *this;
    }
    /// @}
#endif

    /// @{
    /// @name Compatibility with Sb

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE T* getValue() IMATH_NOEXCEPT;

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE const T* getValue() const IMATH_NOEXCEPT;

    /// Return the value in `v`
    template <class S> IMATH_HOSTDEVICE void getValue (Matrix22<S>& v) const IMATH_NOEXCEPT;

    /// Set the value
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22& setValue (const Matrix22<S>& v) IMATH_NOEXCEPT;

    /// Set the value
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22& setTheMatrix (const Matrix22<S>& v) IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Arithmetic and Comparison
    
    /// Equality
    IMATH_HOSTDEVICE constexpr bool operator== (const Matrix22& v) const IMATH_NOEXCEPT;

    /// Inequality
    IMATH_HOSTDEVICE constexpr bool operator!= (const Matrix22& v) const IMATH_NOEXCEPT;

    /// Compare two matrices and test if they are "approximately equal":
    /// @return True if the coefficients of this and `m` are the same
    /// with an absolute error of no more than e, i.e., for all i, j:
    ///
    ///     abs (this[i][j] - m[i][j]) <= e
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithAbsError (const Matrix22<T>& v, T e) const IMATH_NOEXCEPT;

    /// Compare two matrices and test if they are "approximately equal":
    /// @return True if the coefficients of this and m are the same with
    /// a relative error of no more than e, i.e., for all i, j:
    ///
    ///     abs (this[i] - v[i][j]) <= e * abs (this[i][j])
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithRelError (const Matrix22<T>& v, T e) const IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator+= (const Matrix22& v) IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator+= (T a) IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE constexpr Matrix22 operator+ (const Matrix22& v) const IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator-= (const Matrix22& v) IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator-= (T a) IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE constexpr Matrix22 operator- (const Matrix22& v) const IMATH_NOEXCEPT;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE constexpr Matrix22 operator-() const IMATH_NOEXCEPT;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& negate() IMATH_NOEXCEPT;

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator*= (T a) IMATH_NOEXCEPT;

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Matrix22 operator* (T a) const IMATH_NOEXCEPT;

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator/= (T a) IMATH_NOEXCEPT;

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Matrix22 operator/ (T a) const IMATH_NOEXCEPT;

    /// Matrix-matrix multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& operator*= (const Matrix22& v) IMATH_NOEXCEPT;

    /// Matrix-matrix multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22 operator* (const Matrix22& v) const IMATH_NOEXCEPT;

    /// Vector * matrix multiplication
    /// @param[in] src Input vector
    /// @param[out] dst transformed vector
    template <class S> IMATH_HOSTDEVICE void multDirMatrix (const Vec2<S>& src, Vec2<S>& dst) const IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Maniplation

    /// Set to the identity
    IMATH_HOSTDEVICE void makeIdentity() IMATH_NOEXCEPT;

    /// Transpose
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& transpose() IMATH_NOEXCEPT;

    /// Return the transpose
    IMATH_HOSTDEVICE constexpr Matrix22 transposed() const IMATH_NOEXCEPT;

    /// Invert in place
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    /// @return const reference to this
    IMATH_CONSTEXPR14 const Matrix22& invert (bool singExc);

    /// Invert in place
    /// @return const reference to this
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& invert() IMATH_NOEXCEPT;

    /// Return the inverse, leaving this unmodified.
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    IMATH_CONSTEXPR14 Matrix22<T> inverse (bool singExc) const;

    /// Return the inverse, leaving this unmodified.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix22<T> inverse() const IMATH_NOEXCEPT;

    /// Determinant
    IMATH_HOSTDEVICE constexpr T determinant() const IMATH_NOEXCEPT;

    /// Trace
    IMATH_HOSTDEVICE constexpr T trace() const IMATH_NOEXCEPT;

    /// Set matrix to rotation by r (in radians)
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE const Matrix22& setRotation (S r) IMATH_NOEXCEPT;

    /// Rotate the given matrix by r (in radians)
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& rotate (S r) IMATH_NOEXCEPT;

    /// Set matrix to scale by given uniform factor
    /// @return const referenced to this
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& setScale (T s) IMATH_NOEXCEPT;

    /// Set matrix to scale by given vector
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& setScale (const Vec2<S>& s) IMATH_NOEXCEPT;

    // Scale the matrix by s
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix22& scale (const Vec2<S>& s) IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Numeric Limits
    
    /// Largest possible negative value
    IMATH_HOSTDEVICE constexpr static T baseTypeLowest() IMATH_NOEXCEPT { return std::numeric_limits<T>::lowest(); }

    /// Largest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeMax() IMATH_NOEXCEPT { return std::numeric_limits<T>::max(); }

    /// Smallest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeSmallest() IMATH_NOEXCEPT { return std::numeric_limits<T>::min(); }

    /// Smallest possible e for which 1+e != 1
    IMATH_HOSTDEVICE constexpr static T baseTypeEpsilon() IMATH_NOEXCEPT { return std::numeric_limits<T>::epsilon(); }

    /// @}
    
    /// Return the number of the row and column dimensions, i.e. 2.
    IMATH_HOSTDEVICE constexpr static unsigned int dimensions() IMATH_NOEXCEPT { return 2; }

    /// The base type: In templates that accept a parameter `V`, you
    /// can refer to `T` as `V::BaseType`
    typedef T BaseType;

    /// The base vector type
    typedef Vec2<T> BaseVecType;
};

///
/// 3x3 transformation matrix
///

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Matrix33
{
  public:

    /// @{
    /// @name Direct access to elements
    
    /// Matrix elements
    T x[3][3];

    /// @}
    
    /// Row access
    IMATH_HOSTDEVICE T* operator[] (int i) IMATH_NOEXCEPT;

    /// Row access
    IMATH_HOSTDEVICE const T* operator[] (int i) const IMATH_NOEXCEPT;

    /// @{
    ///	@name Constructors and Assignment

    /// Uninitialized
    IMATH_HOSTDEVICE Matrix33 (Uninitialized) IMATH_NOEXCEPT {}

    /// Default constructor: initialize to identity
    ///     1 0 0
    ///     0 1 0
    ///     0 0 1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33() IMATH_NOEXCEPT;

    /// Initialize to scalar constant
    ///     a a a
    ///     a a a
    ///     a a a
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33 (T a) IMATH_NOEXCEPT;

    /// Construct from 3x3 array 
    ///     a[0][0] a[0][1] a[0][2]
    ///     a[1][0] a[1][1] a[1][2]
    ///     a[2][0] a[2][1] a[2][2]
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33 (const T a[3][3]) IMATH_NOEXCEPT;
    /// Construct from given scalar values
    ///     a b c
    ///     d e f
    ///     g h i
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33 (T a, T b, T c, T d, T e, T f, T g, T h, T i) IMATH_NOEXCEPT;

    /// Copy constructor
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33 (const Matrix33& v) IMATH_NOEXCEPT;

    /// Construct from Matrix33 of another base type
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 explicit Matrix33 (const Matrix33<S>& v) IMATH_NOEXCEPT;

    /// Assignment operator
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator= (const Matrix33& v) IMATH_NOEXCEPT;

    /// Assignment from scalar
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator= (T a) IMATH_NOEXCEPT;

    /// Destructor
    ~Matrix33() IMATH_NOEXCEPT = default;

    /// @}

#if IMATH_FOREIGN_VECTOR_INTEROP
    /// @{
    /// @name Interoperability with other matrix types
    ///
    /// Construction and assignment are allowed from other classes that
    /// appear to be equivalent matrix types, provided that they support
    /// double-subscript (i.e., `m[j][i]`) giving the same type as the
    /// elements of this matrix, and their total size appears to be the
    /// right number of matrix elements.
    ///
    /// This functionality is disabled for gcc 4.x, which seems to have a
    /// compiler bug that results in spurious errors. It can also be
    /// disabled by defining IMATH_FOREIGN_VECTOR_INTEROP to be 0 prior to
    /// including any Imath header files.
    ///
    template<typename M, IMATH_ENABLE_IF(has_double_subscript<M,T,3,3>::value)>
    IMATH_HOSTDEVICE explicit Matrix33 (const M& m)
        : Matrix33(T(m[0][0]), T(m[0][1]), T(m[0][2]),
                   T(m[1][0]), T(m[1][1]), T(m[1][2]),
                   T(m[2][0]), T(m[2][1]), T(m[2][2]))
    { }

    /// Interoperability assignment from another type that behaves as if it
    /// were an equivalent matrix.
    template<typename M, IMATH_ENABLE_IF(has_double_subscript<M,T,3,3>::value)>
    IMATH_HOSTDEVICE const Matrix33& operator= (const M& m)
    {
        *this = Matrix33(T(m[0][0]), T(m[0][1]), T(m[0][2]),
                         T(m[1][0]), T(m[1][1]), T(m[1][2]),
                         T(m[2][0]), T(m[2][1]), T(m[2][2]));
        return *this;
    }
    /// @}
#endif

    /// @{
    /// @name Compatibility with Sb

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE T* getValue() IMATH_NOEXCEPT;

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE const T* getValue() const IMATH_NOEXCEPT;

    /// Return the value in `v`
    template <class S> IMATH_HOSTDEVICE void getValue (Matrix33<S>& v) const IMATH_NOEXCEPT;

    /// Set the value
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33& setValue (const Matrix33<S>& v) IMATH_NOEXCEPT;

    /// Set the value
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33& setTheMatrix (const Matrix33<S>& v) IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Arithmetic and Comparison
    
    /// Equality
    IMATH_HOSTDEVICE constexpr bool operator== (const Matrix33& v) const IMATH_NOEXCEPT;

    /// Inequality
    IMATH_HOSTDEVICE constexpr bool operator!= (const Matrix33& v) const IMATH_NOEXCEPT;

    /// Compare two matrices and test if they are "approximately equal":
    /// @return True if the coefficients of this and `m` are the same
    /// with an absolute error of no more than e, i.e., for all i, j:
    ///
    ///     abs (this[i][j] - m[i][j]) <= e
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithAbsError (const Matrix33<T>& v, T e) const IMATH_NOEXCEPT;

    /// Compare two matrices and test if they are "approximately equal":
    /// @return True if the coefficients of this and m are the same with
    /// a relative error of no more than e, i.e., for all i, j:
    ///
    ///     abs (this[i] - v[i][j]) <= e * abs (this[i][j])
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithRelError (const Matrix33<T>& v, T e) const IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator+= (const Matrix33& v) IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator+= (T a) IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE constexpr Matrix33 operator+ (const Matrix33& v) const IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator-= (const Matrix33& v) IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator-= (T a) IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE constexpr Matrix33 operator- (const Matrix33& v) const IMATH_NOEXCEPT;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE constexpr Matrix33 operator-() const IMATH_NOEXCEPT;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& negate() IMATH_NOEXCEPT;

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator*= (T a) IMATH_NOEXCEPT;

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Matrix33 operator* (T a) const IMATH_NOEXCEPT;

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator/= (T a) IMATH_NOEXCEPT;

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Matrix33 operator/ (T a) const IMATH_NOEXCEPT;

    /// Matrix-matrix multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& operator*= (const Matrix33& v) IMATH_NOEXCEPT;

    /// Matrix-matrix multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33 operator* (const Matrix33& v) const IMATH_NOEXCEPT;

    /// Vector-matrix multiplication: a homogeneous transformation
    /// by computing Vec3 (src.x, src.y, 1) * m and dividing by the
    /// result's third element.
    /// @param[in] src The input vector
    /// @param[out] dst The output vector
    template <class S> IMATH_HOSTDEVICE void multVecMatrix (const Vec2<S>& src, Vec2<S>& dst) const IMATH_NOEXCEPT;

    /// Vector-matrix multiplication: multiply `src` by the upper left 2x2
    /// submatrix, ignoring the rest of matrix.
    /// @param[in] src The input vector
    /// @param[out] dst The output vector
    template <class S> IMATH_HOSTDEVICE void multDirMatrix (const Vec2<S>& src, Vec2<S>& dst) const IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Maniplation

    /// Set to the identity matrix
    IMATH_HOSTDEVICE void makeIdentity() IMATH_NOEXCEPT;

    /// Transpose
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& transpose() IMATH_NOEXCEPT;

    /// Return the transpose
    IMATH_HOSTDEVICE constexpr Matrix33 transposed() const IMATH_NOEXCEPT;

    /// Invert in place using the determinant.
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    /// @return const reference to this
    IMATH_CONSTEXPR14 const Matrix33& invert (bool singExc);

    /// Invert in place using the determinant.
    /// @return const reference to this
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& invert() IMATH_NOEXCEPT;

    /// Return the inverse using the determinant, leaving this unmodified.
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    IMATH_CONSTEXPR14 Matrix33<T> inverse (bool singExc) const;

    /// Return the inverse using the determinant, leaving this unmodified.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix33<T> inverse() const IMATH_NOEXCEPT;

    /// Invert in place using the Gauss-Jordan method. Significantly slower
    /// but more accurate than invert().
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    /// @return const reference to this
    const Matrix33& gjInvert (bool singExc);
    
    /// Invert in place using the Gauss-Jordan method. Significantly slower
    /// but more accurate than invert().
    /// @return const reference to this
    IMATH_HOSTDEVICE const Matrix33& gjInvert() IMATH_NOEXCEPT;

    /// Return the inverse using the Gauss-Jordan method, leaving this
    /// unmodified. Significantly slower but more accurate than inverse().
    Matrix33<T> gjInverse (bool singExc) const;

    /// Return the inverse using the Gauss-Jordan method. Significantly slower,
    /// leaving this unmodified. Slower but more accurate than inverse().
    IMATH_HOSTDEVICE Matrix33<T> gjInverse() const IMATH_NOEXCEPT;

    /// Calculate the matrix minor of the (r,c) element
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T minorOf (const int r, const int c) const IMATH_NOEXCEPT;

    /// Build a minor using the specified rows and columns
    IMATH_HOSTDEVICE
    constexpr T fastMinor (const int r0, const int r1, const int c0, const int c1) const IMATH_NOEXCEPT;

    /// Determinant
    IMATH_HOSTDEVICE constexpr T determinant() const IMATH_NOEXCEPT;

    /// Trace
    IMATH_HOSTDEVICE constexpr T trace() const IMATH_NOEXCEPT;

    /// Set matrix to rotation by r (in radians)
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE const Matrix33& setRotation (S r) IMATH_NOEXCEPT;

    // Rotate the given matrix by r (in radians)
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& rotate (S r) IMATH_NOEXCEPT;

    /// Set matrix to scale by given uniform factor
    /// @return const referenced to this
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& setScale (T s) IMATH_NOEXCEPT;

    /// Set matrix to scale by given vector
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& setScale (const Vec2<S>& s) IMATH_NOEXCEPT;

    /// Scale the matrix by s
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& scale (const Vec2<S>& s) IMATH_NOEXCEPT;

    /// Set matrix to translation by given vector
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& setTranslation (const Vec2<S>& t) IMATH_NOEXCEPT;

    /// Return the translation component
    IMATH_HOSTDEVICE constexpr Vec2<T> translation() const IMATH_NOEXCEPT;

    /// Translate the matrix by t
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& translate (const Vec2<S>& t) IMATH_NOEXCEPT;

    /// Set matrix to shear x for each y coord. by given factor xy
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& setShear (const S& h) IMATH_NOEXCEPT;

    /// Set matrix to shear x for each y coord. by given factor h.x
    /// and to shear y for each x coord. by given factor h.y
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& setShear (const Vec2<S>& h) IMATH_NOEXCEPT;

    /// Shear the matrix in x for each y coord. by given factor xy
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& shear (const S& xy) IMATH_NOEXCEPT;

    /// Shear the matrix in x for each y coord. by given factor xy
    /// and shear y for each x coord. by given factor yx
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix33& shear (const Vec2<S>& h) IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Numeric Limits
    
    /// Largest possible negative value
    IMATH_HOSTDEVICE constexpr static T baseTypeLowest() IMATH_NOEXCEPT { return std::numeric_limits<T>::lowest(); }

    /// Largest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeMax() IMATH_NOEXCEPT { return std::numeric_limits<T>::max(); }

    /// Smallest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeSmallest() IMATH_NOEXCEPT { return std::numeric_limits<T>::min(); }

    /// Smallest possible e for which 1+e != 1
    IMATH_HOSTDEVICE constexpr static T baseTypeEpsilon() IMATH_NOEXCEPT { return std::numeric_limits<T>::epsilon(); }

    /// @}
    
    /// Return the number of the row and column dimensions, i.e. 3.
    IMATH_HOSTDEVICE constexpr static unsigned int dimensions() IMATH_NOEXCEPT { return 3; }

    /// The base type: In templates that accept a parameter `V` (could be a Color4), you can refer to `T` as `V::BaseType`
    typedef T BaseType;

    /// The base vector type
    typedef Vec3<T> BaseVecType;
};

///
/// 4x4 transformation matrix
///

template <class T> class IMATH_EXPORT_TEMPLATE_TYPE Matrix44
{
  public:

    /// @{
    /// @name Direct access to elements
    
    /// Matrix elements
    T x[4][4];

    /// @}
    
    /// Row access
    IMATH_HOSTDEVICE T* operator[] (int i) IMATH_NOEXCEPT;

    /// Row access
    IMATH_HOSTDEVICE const T* operator[] (int i) const IMATH_NOEXCEPT;

    /// @{
    ///	@name Constructors and Assignment

    /// Uninitialized
    IMATH_HOSTDEVICE constexpr Matrix44 (Uninitialized) IMATH_NOEXCEPT {}

    /// Default constructor: initialize to identity
    ///     1 0 0 0
    ///     0 1 0 0
    ///     0 0 1 0
    ///     0 0 0 1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44() IMATH_NOEXCEPT;

    /// Initialize to scalar constant
    ///     a a a a
    ///     a a a a
    ///     a a a a
    ///     a a a a
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44 (T a) IMATH_NOEXCEPT;

    /// Construct from 4x4 array 
    ///     a[0][0] a[0][1] a[0][2] a[0][3]
    ///     a[1][0] a[1][1] a[1][2] a[1][3]
    ///     a[2][0] a[2][1] a[2][2] a[2][3]
    ///     a[3][0] a[3][1] a[3][2] a[3][3]
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44 (const T a[4][4]) IMATH_NOEXCEPT;
    /// Construct from given scalar values
    ///     a b c d
    ///     e f g h
    ///     i j k l
    ///     m n o p
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14
    Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l, T m, T n, T o, T p) IMATH_NOEXCEPT;


    /// Construct from a 3x3 rotation matrix and a translation vector
    ///     r r r 0
    ///     r r r 0
    ///     r r r 0
    ///     t t t 1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44 (Matrix33<T> r, Vec3<T> t) IMATH_NOEXCEPT;

    /// Copy constructor
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44 (const Matrix44& v) IMATH_NOEXCEPT;

    /// Construct from Matrix44 of another base type
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 explicit Matrix44 (const Matrix44<S>& v) IMATH_NOEXCEPT;

    /// Assignment operator
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator= (const Matrix44& v) IMATH_NOEXCEPT;

    /// Assignment from scalar
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator= (T a) IMATH_NOEXCEPT;

    /// Destructor
    ~Matrix44() IMATH_NOEXCEPT = default;

    /// @}

#if IMATH_FOREIGN_VECTOR_INTEROP
    /// @{
    /// @name Interoperability with other matrix types
    ///
    /// Construction and assignment are allowed from other classes that
    /// appear to be equivalent matrix types, provided that they support
    /// double-subscript (i.e., `m[j][i]`) giving the same type as the
    /// elements of this matrix, and their total size appears to be the
    /// right number of matrix elements.
    ///
    /// This functionality is disabled for gcc 4.x, which seems to have a
    /// compiler bug that results in spurious errors. It can also be
    /// disabled by defining IMATH_FOREIGN_VECTOR_INTEROP to be 0 prior to
    /// including any Imath header files.
    ///
    template<typename M, IMATH_ENABLE_IF(has_double_subscript<M,T,4,4>::value)>
    IMATH_HOSTDEVICE explicit Matrix44 (const M& m)
        : Matrix44(T(m[0][0]), T(m[0][1]), T(m[0][2]), T(m[0][3]),
                   T(m[1][0]), T(m[1][1]), T(m[1][2]), T(m[1][3]),
                   T(m[2][0]), T(m[2][1]), T(m[2][2]), T(m[2][3]),
                   T(m[3][0]), T(m[3][1]), T(m[3][2]), T(m[3][3]))
    { }

    /// Interoperability assignment from another type that behaves as if it
    /// were an equivalent matrix.
    template<typename M, IMATH_ENABLE_IF(has_double_subscript<M,T,4,4>::value)>
    IMATH_HOSTDEVICE const Matrix44& operator= (const M& m)
    {
        *this = Matrix44(T(m[0][0]), T(m[0][1]), T(m[0][2]), T(m[0][3]),
                         T(m[1][0]), T(m[1][1]), T(m[1][2]), T(m[1][3]),
                         T(m[2][0]), T(m[2][1]), T(m[2][2]), T(m[2][3]),
                         T(m[3][0]), T(m[3][1]), T(m[3][2]), T(m[3][3]));
        return *this;
    }
    /// @}
#endif

    /// @{
    /// @name Compatibility with Sb

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE T* getValue() IMATH_NOEXCEPT;

    /// Return a raw pointer to the array of values
    IMATH_HOSTDEVICE const T* getValue() const IMATH_NOEXCEPT;

    /// Return the value in `v`
    template <class S> IMATH_HOSTDEVICE void getValue (Matrix44<S>& v) const IMATH_NOEXCEPT;

    /// Set the value
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44& setValue (const Matrix44<S>& v) IMATH_NOEXCEPT;

    /// Set the value
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44& setTheMatrix (const Matrix44<S>& v) IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Arithmetic and Comparison
    
    /// Equality
    IMATH_HOSTDEVICE constexpr bool operator== (const Matrix44& v) const IMATH_NOEXCEPT;

    /// Inequality
    IMATH_HOSTDEVICE constexpr bool operator!= (const Matrix44& v) const IMATH_NOEXCEPT;

    /// Compare two matrices and test if they are "approximately equal":
    /// @return True if the coefficients of this and `m` are the same
    /// with an absolute error of no more than e, i.e., for all i, j:
    ///
    ///     abs (this[i][j] - m[i][j]) <= e
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithAbsError (const Matrix44<T>& v, T e) const IMATH_NOEXCEPT;

    /// Compare two matrices and test if they are "approximately equal":
    /// @return True if the coefficients of this and m are the same with
    /// a relative error of no more than e, i.e., for all i, j:
    ///
    ///     abs (this[i] - v[i][j]) <= e * abs (this[i][j])
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 bool equalWithRelError (const Matrix44<T>& v, T e) const IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator+= (const Matrix44& v) IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator+= (T a) IMATH_NOEXCEPT;

    /// Component-wise addition
    IMATH_HOSTDEVICE constexpr Matrix44 operator+ (const Matrix44& v) const IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator-= (const Matrix44& v) IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator-= (T a) IMATH_NOEXCEPT;

    /// Component-wise subtraction
    IMATH_HOSTDEVICE constexpr Matrix44 operator- (const Matrix44& v) const IMATH_NOEXCEPT;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE constexpr Matrix44 operator-() const IMATH_NOEXCEPT;

    /// Component-wise multiplication by -1
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& negate() IMATH_NOEXCEPT;

    /// Component-wise multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator*= (T a) IMATH_NOEXCEPT;

    /// Component-wise multiplication
    IMATH_HOSTDEVICE constexpr Matrix44 operator* (T a) const IMATH_NOEXCEPT;

    /// Component-wise division
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator/= (T a) IMATH_NOEXCEPT;

    /// Component-wise division
    IMATH_HOSTDEVICE constexpr Matrix44 operator/ (T a) const IMATH_NOEXCEPT;

    /// Matrix-matrix multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& operator*= (const Matrix44& v) IMATH_NOEXCEPT;

    /// Matrix-matrix multiplication
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44 operator* (const Matrix44& v) const IMATH_NOEXCEPT;

    /// Matrix-matrix multiplication: compute c = a * b
    IMATH_HOSTDEVICE
    static void multiply (const Matrix44& a,     // assumes that
                          const Matrix44& b,     // &a != &c and
                          Matrix44& c) IMATH_NOEXCEPT; // &b != &c.

    /// Matrix-matrix multiplication returning a result.
    IMATH_HOSTDEVICE
    static IMATH_CONSTEXPR14 Matrix44 multiply (const Matrix44& a, const Matrix44& b) IMATH_NOEXCEPT;

    /// Vector-matrix multiplication: a homogeneous transformation
    /// by computing Vec3 (src.x, src.y, src.z, 1) * m and dividing by the
    /// result's third element.
    /// @param[in] src The input vector
    /// @param[out] dst The output vector
    template <class S> IMATH_HOSTDEVICE void multVecMatrix (const Vec3<S>& src, Vec3<S>& dst) const IMATH_NOEXCEPT;

    /// Vector-matrix multiplication: multiply `src` by the upper left 2x2
    /// submatrix, ignoring the rest of matrix.
    /// @param[in] src The input vector
    /// @param[out] dst The output vector
    template <class S> IMATH_HOSTDEVICE void multDirMatrix (const Vec3<S>& src, Vec3<S>& dst) const IMATH_NOEXCEPT;

    /// @}

    /// @{
    /// @name Maniplation

    /// Set to the identity matrix
    IMATH_HOSTDEVICE void makeIdentity() IMATH_NOEXCEPT;

    /// Transpose
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& transpose() IMATH_NOEXCEPT;

    /// Return the transpose
    IMATH_HOSTDEVICE constexpr Matrix44 transposed() const IMATH_NOEXCEPT;

    /// Invert in place using the determinant.
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    /// @return const reference to this
    IMATH_CONSTEXPR14 const Matrix44& invert (bool singExc);

    /// Invert in place using the determinant.
    /// @return const reference to this
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& invert() IMATH_NOEXCEPT;

    /// Return the inverse using the determinant, leaving this unmodified.
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    IMATH_CONSTEXPR14 Matrix44<T> inverse (bool singExc) const;

    /// Return the inverse using the determinant, leaving this unmodified.
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 Matrix44<T> inverse() const IMATH_NOEXCEPT;

    /// Invert in place using the Gauss-Jordan method. Significantly slower
    /// but more accurate than invert().
    /// @param singExc If true, throw an exception if the matrix cannot be inverted.
    /// @return const reference to this
    IMATH_CONSTEXPR14 const Matrix44& gjInvert (bool singExc);

    /// Invert in place using the Gauss-Jordan method. Significantly slower
    /// but more accurate than invert().
    /// @return const reference to this
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& gjInvert() IMATH_NOEXCEPT;

    /// Return the inverse using the Gauss-Jordan method, leaving this
    /// unmodified. Significantly slower but more accurate than inverse().
    Matrix44<T> gjInverse (bool singExc) const;

    /// Return the inverse using the Gauss-Jordan method, leaving this
    /// unmodified Significantly slower but more accurate than inverse().
    IMATH_HOSTDEVICE Matrix44<T> gjInverse() const IMATH_NOEXCEPT;

    /// Calculate the matrix minor of the (r,c) element
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T minorOf (const int r, const int c) const IMATH_NOEXCEPT;

    /// Build a minor using the specified rows and columns
    IMATH_HOSTDEVICE
    constexpr T fastMinor (const int r0,
                           const int r1,
                           const int r2,
                           const int c0,
                           const int c1,
                           const int c2) const IMATH_NOEXCEPT;

    /// Determinant
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 T determinant() const IMATH_NOEXCEPT;

    /// Trace
    IMATH_HOSTDEVICE constexpr T trace() const IMATH_NOEXCEPT;

    /// Set matrix to rotation by XYZ euler angles (in radians)
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE const Matrix44& setEulerAngles (const Vec3<S>& r) IMATH_NOEXCEPT;

    /// Set matrix to rotation around given axis by given angle (in radians)
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& setAxisAngle (const Vec3<S>& ax, S ang) IMATH_NOEXCEPT;

    /// Rotate the matrix by XYZ euler angles in r (in radians)
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE const Matrix44& rotate (const Vec3<S>& r) IMATH_NOEXCEPT;

    /// Set matrix to scale by given uniform factor
    /// @return const referenced to this
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& setScale (T s) IMATH_NOEXCEPT;

    /// Set matrix to scale by given vector
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& setScale (const Vec3<S>& s) IMATH_NOEXCEPT;

    /// Scale the matrix by s
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& scale (const Vec3<S>& s) IMATH_NOEXCEPT;

    /// Set matrix to translation by given vector
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& setTranslation (const Vec3<S>& t) IMATH_NOEXCEPT;

    /// Return translation component
    IMATH_HOSTDEVICE constexpr const Vec3<T> translation() const IMATH_NOEXCEPT;

    /// Translate the matrix by t
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& translate (const Vec3<S>& t) IMATH_NOEXCEPT;

    /// Set matrix to shear by given vector h.  The resulting matrix
    /// - will shear x for each y coord. by a factor of h[0] ;
    /// - will shear x for each z coord. by a factor of h[1] ;
    /// - will shear y for each z coord. by a factor of h[2] .
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& setShear (const Vec3<S>& h) IMATH_NOEXCEPT;

    /// Set matrix to shear by given factors.  The resulting matrix
    /// - will shear x for each y coord. by a factor of h.xy ;
    /// - will shear x for each z coord. by a factor of h.xz ;
    /// - will shear y for each z coord. by a factor of h.yz ;
    /// - will shear y for each x coord. by a factor of h.yx ;
    /// - will shear z for each x coord. by a factor of h.zx ;
    /// - will shear z for each y coord. by a factor of h.zy .
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& setShear (const Shear6<S>& h) IMATH_NOEXCEPT;

    /// Shear the matrix by given vector.  The composed matrix
    /// will be `shear` * `this`, where the shear matrix ...
    /// - will shear x for each y coord. by a factor of h[0] ;
    /// - will shear x for each z coord. by a factor of h[1] ;
    /// - will shear y for each z coord. by a factor of h[2] .
    /// @return const referenced to this
    template <class S> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& shear (const Vec3<S>& h) IMATH_NOEXCEPT;

    /// Shear the matrix by the given factors.  The composed matrix
    /// will be `shear` * `this`, where the shear matrix ...
    /// - will shear x for each y coord. by a factor of h.xy ;
    /// - will shear x for each z coord. by a factor of h.xz ;
    /// - will shear y for each z coord. by a factor of h.yz ;
    /// - will shear y for each x coord. by a factor of h.yx ;
    /// - will shear z for each x coord. by a factor of h.zx ;
    /// - will shear z for each y coord. by a factor of h.zy .
    /// @return const referenced to this
    template <class S>
    IMATH_HOSTDEVICE IMATH_CONSTEXPR14 const Matrix44& shear (const Shear6<S>& h) IMATH_NOEXCEPT;

    /// @}
    
    /// @{
    /// @name Numeric Limits
    
    /// Largest possible negative value
    IMATH_HOSTDEVICE constexpr static T baseTypeLowest() IMATH_NOEXCEPT { return std::numeric_limits<T>::lowest(); }

    /// Largest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeMax() IMATH_NOEXCEPT { return std::numeric_limits<T>::max(); }

    /// Smallest possible positive value
    IMATH_HOSTDEVICE constexpr static T baseTypeSmallest() IMATH_NOEXCEPT { return std::numeric_limits<T>::min(); }

    /// Smallest possible e for which 1+e != 1
    IMATH_HOSTDEVICE constexpr static T baseTypeEpsilon() IMATH_NOEXCEPT { return std::numeric_limits<T>::epsilon(); }

    /// @}
    
    /// Return the number of the row and column dimensions, i.e. 4
    IMATH_HOSTDEVICE constexpr static unsigned int dimensions() IMATH_NOEXCEPT { return 4; }

    /// The base type: In templates that accept a parameter `V` (could be a Color4), you can refer to `T` as `V::BaseType`
    typedef T BaseType;

    /// The base vector type
    typedef Vec4<T> BaseVecType;
};

/// Stream output, as:
///     (m00 m01
///      m10 m11)
template <class T> std::ostream& operator<< (std::ostream& s, const Matrix22<T>& m);

/// Stream output, as:
///     (m00 m01 m02
///      m10 m11 m12
///      m20 m21 m22)
template <class T> std::ostream& operator<< (std::ostream& s, const Matrix33<T>& m);

/// Stream output, as:
///
///     (m00 m01 m02 m03
///      m10 m11 m12 m13
///      m20 m21 m22 m23
///      m30 m31 m32 m33)
template <class T> std::ostream& operator<< (std::ostream& s, const Matrix44<T>& m);

//---------------------------------------------
// Vector-times-matrix multiplication operators
//---------------------------------------------

/// Vector-matrix multiplication: v *= m
template <class S, class T>
IMATH_HOSTDEVICE inline const Vec2<S>& operator*= (Vec2<S>& v, const Matrix22<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: r = v * m
template <class S, class T>
IMATH_HOSTDEVICE inline Vec2<S> operator* (const Vec2<S>& v, const Matrix22<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: v *= m
template <class S, class T>
IMATH_HOSTDEVICE inline const Vec2<S>& operator*= (Vec2<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: r = v * m
template <class S, class T>
IMATH_HOSTDEVICE inline Vec2<S> operator* (const Vec2<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: v *= m
template <class S, class T>
IMATH_HOSTDEVICE inline const Vec3<S>& operator*= (Vec3<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: r = v * m
template <class S, class T>
IMATH_HOSTDEVICE inline Vec3<S> operator* (const Vec3<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: v *= m
template <class S, class T>
IMATH_HOSTDEVICE inline const Vec3<S>& operator*= (Vec3<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: r = v * m
template <class S, class T>
IMATH_HOSTDEVICE inline Vec3<S> operator* (const Vec3<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: v *= m
template <class S, class T>
IMATH_HOSTDEVICE inline const Vec4<S>& operator*= (Vec4<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT;

/// Vector-matrix multiplication: r = v * m
template <class S, class T>
IMATH_HOSTDEVICE inline Vec4<S> operator* (const Vec4<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT;

//-------------------------
// Typedefs for convenience
//-------------------------

/// 2x2 matrix of float
typedef Matrix22<float> M22f;

/// 2x2 matrix of double
typedef Matrix22<double> M22d;

/// 3x3 matrix of float
typedef Matrix33<float> M33f;

/// 3x3 matrix of double
typedef Matrix33<double> M33d;

/// 4x4 matrix of float
typedef Matrix44<float> M44f;

/// 4x4 matrix of double
typedef Matrix44<double> M44d;

//---------------------------
// Implementation of Matrix22
//---------------------------

template <class T>
IMATH_HOSTDEVICE inline T*
Matrix22<T>::operator[] (int i) IMATH_NOEXCEPT
{
    return x[i];
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Matrix22<T>::operator[] (int i) const IMATH_NOEXCEPT
{
    return x[i];
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>::Matrix22() IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>::Matrix22 (T a) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = a;
    x[1][0] = a;
    x[1][1] = a;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>::Matrix22 (
    const T a[2][2]) IMATH_NOEXCEPT
{
    // Function calls and aliasing issues can inhibit vectorization versus
    // straight assignment of data members, so instead of this:
    //     memcpy (x, a, sizeof (x));
    // we do this:
    x[0][0] = a[0][0];
    x[0][1] = a[0][1];
    x[1][0] = a[1][0];
    x[1][1] = a[1][1];
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>::Matrix22 (
    T a, T b, T c, T d) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = b;
    x[1][0] = c;
    x[1][1] = d;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>::Matrix22 (const Matrix22& v) IMATH_NOEXCEPT
{
    // Function calls and aliasing issues can inhibit vectorization versus
    // straight assignment of data members, so we don't do this:
    //     memcpy (x, v.x, sizeof (x));
    // we do this:
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>::Matrix22 (const Matrix22<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = T (v.x[0][0]);
    x[0][1] = T (v.x[0][1]);
    x[1][0] = T (v.x[1][0]);
    x[1][1] = T (v.x[1][1]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator= (const Matrix22& v) IMATH_NOEXCEPT
{
    // Function calls and aliasing issues can inhibit vectorization versus
    // straight assignment of data members, so we don't do this:
    //     memcpy (x, v.x, sizeof (x));
    // we do this:
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator= (T a) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = a;
    x[1][0] = a;
    x[1][1] = a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE inline T*
Matrix22<T>::getValue () IMATH_NOEXCEPT
{
    return (T*) &x[0][0];
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Matrix22<T>::getValue() const IMATH_NOEXCEPT
{
    return (const T*) &x[0][0];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix22<T>::getValue (Matrix22<S>& v) const IMATH_NOEXCEPT
{
    v.x[0][0] = x[0][0];
    v.x[0][1] = x[0][1];
    v.x[1][0] = x[1][0];
    v.x[1][1] = x[1][1];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>&
Matrix22<T>::setValue (const Matrix22<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>&
Matrix22<T>::setTheMatrix (const Matrix22<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    return *this;
}

template <class T>
IMATH_HOSTDEVICE inline void
Matrix22<T>::makeIdentity() IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Matrix22<T>::operator== (const Matrix22& v) const IMATH_NOEXCEPT
{
    return x[0][0] == v.x[0][0] && x[0][1] == v.x[0][1] && x[1][0] == v.x[1][0] &&
           x[1][1] == v.x[1][1];
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Matrix22<T>::operator!= (const Matrix22& v) const IMATH_NOEXCEPT
{
    return x[0][0] != v.x[0][0] || x[0][1] != v.x[0][1] || x[1][0] != v.x[1][0] ||
           x[1][1] != v.x[1][1];
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Matrix22<T>::equalWithAbsError (const Matrix22<T>& m, T e) const IMATH_NOEXCEPT
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this).x[i][j], m.x[i][j], e))
                return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Matrix22<T>::equalWithRelError (const Matrix22<T>& m, T e) const IMATH_NOEXCEPT
{
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this).x[i][j], m.x[i][j], e))
                return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator+= (const Matrix22<T>& v) IMATH_NOEXCEPT
{
    x[0][0] += v.x[0][0];
    x[0][1] += v.x[0][1];
    x[1][0] += v.x[1][0];
    x[1][1] += v.x[1][1];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator+= (T a) IMATH_NOEXCEPT
{
    x[0][0] += a;
    x[0][1] += a;
    x[1][0] += a;
    x[1][1] += a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix22<T>
Matrix22<T>::operator+ (const Matrix22<T>& v) const IMATH_NOEXCEPT
{
    return Matrix22 (x[0][0] + v.x[0][0],
                     x[0][1] + v.x[0][1],
                     x[1][0] + v.x[1][0],
                     x[1][1] + v.x[1][1]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator-= (const Matrix22<T>& v) IMATH_NOEXCEPT
{
    x[0][0] -= v.x[0][0];
    x[0][1] -= v.x[0][1];
    x[1][0] -= v.x[1][0];
    x[1][1] -= v.x[1][1];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator-= (T a) IMATH_NOEXCEPT
{
    x[0][0] -= a;
    x[0][1] -= a;
    x[1][0] -= a;
    x[1][1] -= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix22<T>
Matrix22<T>::operator- (const Matrix22<T>& v) const IMATH_NOEXCEPT
{
    return Matrix22 (x[0][0] - v.x[0][0],
                     x[0][1] - v.x[0][1],
                     x[1][0] - v.x[1][0],
                     x[1][1] - v.x[1][1]);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix22<T>
Matrix22<T>::operator-() const IMATH_NOEXCEPT
{
    return Matrix22 (-x[0][0], -x[0][1], -x[1][0], -x[1][1]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::negate() IMATH_NOEXCEPT
{
    x[0][0] = -x[0][0];
    x[0][1] = -x[0][1];
    x[1][0] = -x[1][0];
    x[1][1] = -x[1][1];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator*= (T a) IMATH_NOEXCEPT
{
    x[0][0] *= a;
    x[0][1] *= a;
    x[1][0] *= a;
    x[1][1] *= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix22<T>
Matrix22<T>::operator* (T a) const IMATH_NOEXCEPT
{
    return Matrix22 (x[0][0] * a, x[0][1] * a, x[1][0] * a, x[1][1] * a);
}

/// Matrix-scalar multiplication
template <class T>
IMATH_HOSTDEVICE inline Matrix22<T>
operator* (T a, const Matrix22<T>& v) IMATH_NOEXCEPT
{
    return v * a;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator*= (const Matrix22<T>& v) IMATH_NOEXCEPT
{
    Matrix22 tmp (T (0));

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                tmp.x[i][j] += x[i][k] * v.x[k][j];

    *this = tmp;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>
Matrix22<T>::operator* (const Matrix22<T>& v) const IMATH_NOEXCEPT
{
    Matrix22 tmp (T (0));

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                tmp.x[i][j] += x[i][k] * v.x[k][j];

    return tmp;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix22<T>::multDirMatrix (const Vec2<S>& src, Vec2<S>& dst) const IMATH_NOEXCEPT
{
    S a, b;

    a = src.x * x[0][0] + src.y * x[1][0];
    b = src.x * x[0][1] + src.y * x[1][1];

    dst.x = a;
    dst.y = b;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::operator/= (T a) IMATH_NOEXCEPT
{
    x[0][0] /= a;
    x[0][1] /= a;
    x[1][0] /= a;
    x[1][1] /= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix22<T>
Matrix22<T>::operator/ (T a) const IMATH_NOEXCEPT
{
    return Matrix22 (x[0][0] / a, x[0][1] / a, x[1][0] / a, x[1][1] / a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::transpose() IMATH_NOEXCEPT
{
    Matrix22 tmp (x[0][0], x[1][0], x[0][1], x[1][1]);
    *this = tmp;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix22<T>
Matrix22<T>::transposed() const IMATH_NOEXCEPT
{
    return Matrix22 (x[0][0], x[1][0], x[0][1], x[1][1]);
}

template <class T>
IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::invert (bool singExc)
{
    *this = inverse (singExc);
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::invert() IMATH_NOEXCEPT
{
    *this = inverse();
    return *this;
}

template <class T>
IMATH_CONSTEXPR14 inline Matrix22<T>
Matrix22<T>::inverse (bool singExc) const
{
    Matrix22 s (x[1][1], -x[0][1], -x[1][0], x[0][0]);

    T r = x[0][0] * x[1][1] - x[1][0] * x[0][1];

    if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
    {
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                s[i][j] /= r;
            }
        }
    }
    else
    {
        T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                if (mr > IMATH_INTERNAL_NAMESPACE::abs (s[i][j]))
                {
                    s[i][j] /= r;
                }
                else
                {
                    if (singExc)
                        throw std::invalid_argument ("Cannot invert "
                                                     "singular matrix.");
                    return Matrix22();
                }
            }
        }
    }
    return s;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix22<T>
Matrix22<T>::inverse() const IMATH_NOEXCEPT
{
    Matrix22 s (x[1][1], -x[0][1], -x[1][0], x[0][0]);

    T r = x[0][0] * x[1][1] - x[1][0] * x[0][1];

    if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
    {
        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                s[i][j] /= r;
            }
        }
    }
    else
    {
        T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

        for (int i = 0; i < 2; ++i)
        {
            for (int j = 0; j < 2; ++j)
            {
                if (mr > IMATH_INTERNAL_NAMESPACE::abs (s[i][j]))
                {
                    s[i][j] /= r;
                }
                else
                {
                    return Matrix22();
                }
            }
        }
    }
    return s;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Matrix22<T>::determinant() const IMATH_NOEXCEPT
{
    return x[0][0] * x[1][1] - x[1][0] * x[0][1];
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Matrix22<T>::trace () const IMATH_NOEXCEPT
{
    return x[0][0] + x[1][1];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline const Matrix22<T>&
Matrix22<T>::setRotation (S r) IMATH_NOEXCEPT
{
    S cos_r, sin_r;

    cos_r = cos ((T) r);
    sin_r = sin ((T) r);

    x[0][0] = cos_r;
    x[0][1] = sin_r;

    x[1][0] = -sin_r;
    x[1][1] = cos_r;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::rotate (S r) IMATH_NOEXCEPT
{
    *this *= Matrix22<T>().setRotation (r);
    return *this;
}

template <class T>
IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::setScale (T s) IMATH_NOEXCEPT
{
    //
    // Set the matrix to:
    //  | s 0 |
    //  | 0 s |
    //

    x[0][0] = s;
    x[0][1] = static_cast<T> (0);
    x[1][0] = static_cast<T> (0);
    x[1][1] = s;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::setScale (const Vec2<S>& s) IMATH_NOEXCEPT
{
    //
    // Set the matrix to:
    //  | s.x  0  |
    //  |  0  s.y |
    //

    x[0][0] = s.x;
    x[0][1] = static_cast<T> (0);
    x[1][0] = static_cast<T> (0);
    x[1][1] = s.y;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix22<T>&
Matrix22<T>::scale (const Vec2<S>& s) IMATH_NOEXCEPT
{
    x[0][0] *= s.x;
    x[0][1] *= s.x;

    x[1][0] *= s.y;
    x[1][1] *= s.y;

    return *this;
}

//---------------------------
// Implementation of Matrix33
//---------------------------

template <class T>
IMATH_HOSTDEVICE inline T*
Matrix33<T>::operator[] (int i) IMATH_NOEXCEPT
{
    return x[i];
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Matrix33<T>::operator[] (int i) const IMATH_NOEXCEPT
{
    return x[i];
}

template <class T>
IMATH_HOSTDEVICE inline IMATH_CONSTEXPR14
Matrix33<T>::Matrix33() IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>::Matrix33 (T a) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = a;
    x[0][2] = a;
    x[1][0] = a;
    x[1][1] = a;
    x[1][2] = a;
    x[2][0] = a;
    x[2][1] = a;
    x[2][2] = a;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>::Matrix33 (
    const T a[3][3]) IMATH_NOEXCEPT
{
    // Function calls and aliasing issues can inhibit vectorization versus
    // straight assignment of data members, so instead of this:
    //     memcpy (x, a, sizeof (x));
    // we do this:
    x[0][0] = a[0][0];
    x[0][1] = a[0][1];
    x[0][2] = a[0][2];
    x[1][0] = a[1][0];
    x[1][1] = a[1][1];
    x[1][2] = a[1][2];
    x[2][0] = a[2][0];
    x[2][1] = a[2][1];
    x[2][2] = a[2][2];
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>::Matrix33 (T a, T b, T c, T d, T e, T f, T g, T h, T i) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = b;
    x[0][2] = c;
    x[1][0] = d;
    x[1][1] = e;
    x[1][2] = f;
    x[2][0] = g;
    x[2][1] = h;
    x[2][2] = i;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>::Matrix33 (const Matrix33& v) IMATH_NOEXCEPT
{
    // Function calls and aliasing issues can inhibit vectorization versus
    // straight assignment of data members, so instead of this:
    //     memcpy (x, v.x, sizeof (x));
    // we do this:
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>::Matrix33 (const Matrix33<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = T (v.x[0][0]);
    x[0][1] = T (v.x[0][1]);
    x[0][2] = T (v.x[0][2]);
    x[1][0] = T (v.x[1][0]);
    x[1][1] = T (v.x[1][1]);
    x[1][2] = T (v.x[1][2]);
    x[2][0] = T (v.x[2][0]);
    x[2][1] = T (v.x[2][1]);
    x[2][2] = T (v.x[2][2]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator= (const Matrix33& v) IMATH_NOEXCEPT
{
    // Function calls and aliasing issues can inhibit vectorization versus
    // straight assignment of data members, so instead of this:
    //     memcpy (x, v.x, sizeof (x));
    // we do this:
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator= (T a) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = a;
    x[0][2] = a;
    x[1][0] = a;
    x[1][1] = a;
    x[1][2] = a;
    x[2][0] = a;
    x[2][1] = a;
    x[2][2] = a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE inline T*
Matrix33<T>::getValue () IMATH_NOEXCEPT
{
    return (T*) &x[0][0];
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Matrix33<T>::getValue() const IMATH_NOEXCEPT
{
    return (const T*) &x[0][0];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix33<T>::getValue (Matrix33<S>& v) const IMATH_NOEXCEPT
{
    v.x[0][0] = x[0][0];
    v.x[0][1] = x[0][1];
    v.x[0][2] = x[0][2];
    v.x[1][0] = x[1][0];
    v.x[1][1] = x[1][1];
    v.x[1][2] = x[1][2];
    v.x[2][0] = x[2][0];
    v.x[2][1] = x[2][1];
    v.x[2][2] = x[2][2];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>&
Matrix33<T>::setValue (const Matrix33<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>&
Matrix33<T>::setTheMatrix (const Matrix33<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    return *this;
}

template <class T>
IMATH_HOSTDEVICE inline void
Matrix33<T>::makeIdentity() IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Matrix33<T>::operator== (const Matrix33& v) const IMATH_NOEXCEPT
{
    return x[0][0] == v.x[0][0] && x[0][1] == v.x[0][1] && x[0][2] == v.x[0][2] &&
           x[1][0] == v.x[1][0] && x[1][1] == v.x[1][1] && x[1][2] == v.x[1][2] &&
           x[2][0] == v.x[2][0] && x[2][1] == v.x[2][1] && x[2][2] == v.x[2][2];
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Matrix33<T>::operator!= (const Matrix33& v) const IMATH_NOEXCEPT
{
    return x[0][0] != v.x[0][0] || x[0][1] != v.x[0][1] || x[0][2] != v.x[0][2] ||
           x[1][0] != v.x[1][0] || x[1][1] != v.x[1][1] || x[1][2] != v.x[1][2] ||
           x[2][0] != v.x[2][0] || x[2][1] != v.x[2][1] || x[2][2] != v.x[2][2];
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Matrix33<T>::equalWithAbsError (const Matrix33<T>& m, T e) const IMATH_NOEXCEPT
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Matrix33<T>::equalWithRelError (const Matrix33<T>& m, T e) const IMATH_NOEXCEPT
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator+= (const Matrix33<T>& v) IMATH_NOEXCEPT
{
    x[0][0] += v.x[0][0];
    x[0][1] += v.x[0][1];
    x[0][2] += v.x[0][2];
    x[1][0] += v.x[1][0];
    x[1][1] += v.x[1][1];
    x[1][2] += v.x[1][2];
    x[2][0] += v.x[2][0];
    x[2][1] += v.x[2][1];
    x[2][2] += v.x[2][2];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator+= (T a) IMATH_NOEXCEPT
{
    x[0][0] += a;
    x[0][1] += a;
    x[0][2] += a;
    x[1][0] += a;
    x[1][1] += a;
    x[1][2] += a;
    x[2][0] += a;
    x[2][1] += a;
    x[2][2] += a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix33<T>
Matrix33<T>::operator+ (const Matrix33<T>& v) const IMATH_NOEXCEPT
{
    return Matrix33 (x[0][0] + v.x[0][0],
                     x[0][1] + v.x[0][1],
                     x[0][2] + v.x[0][2],
                     x[1][0] + v.x[1][0],
                     x[1][1] + v.x[1][1],
                     x[1][2] + v.x[1][2],
                     x[2][0] + v.x[2][0],
                     x[2][1] + v.x[2][1],
                     x[2][2] + v.x[2][2]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator-= (const Matrix33<T>& v) IMATH_NOEXCEPT
{
    x[0][0] -= v.x[0][0];
    x[0][1] -= v.x[0][1];
    x[0][2] -= v.x[0][2];
    x[1][0] -= v.x[1][0];
    x[1][1] -= v.x[1][1];
    x[1][2] -= v.x[1][2];
    x[2][0] -= v.x[2][0];
    x[2][1] -= v.x[2][1];
    x[2][2] -= v.x[2][2];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator-= (T a) IMATH_NOEXCEPT
{
    x[0][0] -= a;
    x[0][1] -= a;
    x[0][2] -= a;
    x[1][0] -= a;
    x[1][1] -= a;
    x[1][2] -= a;
    x[2][0] -= a;
    x[2][1] -= a;
    x[2][2] -= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix33<T>
Matrix33<T>::operator- (const Matrix33<T>& v) const IMATH_NOEXCEPT
{
    return Matrix33 (x[0][0] - v.x[0][0],
                     x[0][1] - v.x[0][1],
                     x[0][2] - v.x[0][2],
                     x[1][0] - v.x[1][0],
                     x[1][1] - v.x[1][1],
                     x[1][2] - v.x[1][2],
                     x[2][0] - v.x[2][0],
                     x[2][1] - v.x[2][1],
                     x[2][2] - v.x[2][2]);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix33<T>
Matrix33<T>::operator-() const IMATH_NOEXCEPT
{
    return Matrix33 (-x[0][0],
                     -x[0][1],
                     -x[0][2],
                     -x[1][0],
                     -x[1][1],
                     -x[1][2],
                     -x[2][0],
                     -x[2][1],
                     -x[2][2]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::negate() IMATH_NOEXCEPT
{
    x[0][0] = -x[0][0];
    x[0][1] = -x[0][1];
    x[0][2] = -x[0][2];
    x[1][0] = -x[1][0];
    x[1][1] = -x[1][1];
    x[1][2] = -x[1][2];
    x[2][0] = -x[2][0];
    x[2][1] = -x[2][1];
    x[2][2] = -x[2][2];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator*= (T a) IMATH_NOEXCEPT
{
    x[0][0] *= a;
    x[0][1] *= a;
    x[0][2] *= a;
    x[1][0] *= a;
    x[1][1] *= a;
    x[1][2] *= a;
    x[2][0] *= a;
    x[2][1] *= a;
    x[2][2] *= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix33<T>
Matrix33<T>::operator* (T a) const IMATH_NOEXCEPT
{
    return Matrix33 (x[0][0] * a,
                     x[0][1] * a,
                     x[0][2] * a,
                     x[1][0] * a,
                     x[1][1] * a,
                     x[1][2] * a,
                     x[2][0] * a,
                     x[2][1] * a,
                     x[2][2] * a);
}

/// Matrix-scalar multiplication
template <class T>
IMATH_HOSTDEVICE inline Matrix33<T> constexpr
operator* (T a, const Matrix33<T>& v) IMATH_NOEXCEPT
{
    return v * a;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator*= (const Matrix33<T>& v) IMATH_NOEXCEPT
{
    // Avoid initializing with 0 values before immediately overwriting them,
    // and unroll all loops for the best autovectorization.
    Matrix33 tmp(IMATH_INTERNAL_NAMESPACE::UNINITIALIZED);

    tmp.x[0][0] = x[0][0] * v.x[0][0] + x[0][1] * v.x[1][0] + x[0][2] * v.x[2][0];
    tmp.x[0][1] = x[0][0] * v.x[0][1] + x[0][1] * v.x[1][1] + x[0][2] * v.x[2][1];
    tmp.x[0][2] = x[0][0] * v.x[0][2] + x[0][1] * v.x[1][2] + x[0][2] * v.x[2][2];

    tmp.x[1][0] = x[1][0] * v.x[0][0] + x[1][1] * v.x[1][0] + x[1][2] * v.x[2][0];
    tmp.x[1][1] = x[1][0] * v.x[0][1] + x[1][1] * v.x[1][1] + x[1][2] * v.x[2][1];
    tmp.x[1][2] = x[1][0] * v.x[0][2] + x[1][1] * v.x[1][2] + x[1][2] * v.x[2][2];

    tmp.x[2][0] = x[2][0] * v.x[0][0] + x[2][1] * v.x[1][0] + x[2][2] * v.x[2][0];
    tmp.x[2][1] = x[2][0] * v.x[0][1] + x[2][1] * v.x[1][1] + x[2][2] * v.x[2][1];
    tmp.x[2][2] = x[2][0] * v.x[0][2] + x[2][1] * v.x[1][2] + x[2][2] * v.x[2][2];

    *this = tmp;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>
Matrix33<T>::operator* (const Matrix33<T>& v) const IMATH_NOEXCEPT
{
    // Avoid initializing with 0 values before immediately overwriting them,
    // and unroll all loops for the best autovectorization.
    Matrix33 tmp(IMATH_INTERNAL_NAMESPACE::UNINITIALIZED);

    tmp.x[0][0] = x[0][0] * v.x[0][0] + x[0][1] * v.x[1][0] + x[0][2] * v.x[2][0];
    tmp.x[0][1] = x[0][0] * v.x[0][1] + x[0][1] * v.x[1][1] + x[0][2] * v.x[2][1];
    tmp.x[0][2] = x[0][0] * v.x[0][2] + x[0][1] * v.x[1][2] + x[0][2] * v.x[2][2];

    tmp.x[1][0] = x[1][0] * v.x[0][0] + x[1][1] * v.x[1][0] + x[1][2] * v.x[2][0];
    tmp.x[1][1] = x[1][0] * v.x[0][1] + x[1][1] * v.x[1][1] + x[1][2] * v.x[2][1];
    tmp.x[1][2] = x[1][0] * v.x[0][2] + x[1][1] * v.x[1][2] + x[1][2] * v.x[2][2];

    tmp.x[2][0] = x[2][0] * v.x[0][0] + x[2][1] * v.x[1][0] + x[2][2] * v.x[2][0];
    tmp.x[2][1] = x[2][0] * v.x[0][1] + x[2][1] * v.x[1][1] + x[2][2] * v.x[2][1];
    tmp.x[2][2] = x[2][0] * v.x[0][2] + x[2][1] * v.x[1][2] + x[2][2] * v.x[2][2];

    return tmp;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix33<T>::multVecMatrix (const Vec2<S>& src, Vec2<S>& dst) const IMATH_NOEXCEPT
{
    S a, b, w;

    a = src.x * x[0][0] + src.y * x[1][0] + x[2][0];
    b = src.x * x[0][1] + src.y * x[1][1] + x[2][1];
    w = src.x * x[0][2] + src.y * x[1][2] + x[2][2];

    dst.x = a / w;
    dst.y = b / w;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix33<T>::multDirMatrix (const Vec2<S>& src, Vec2<S>& dst) const IMATH_NOEXCEPT
{
    S a, b;

    a = src.x * x[0][0] + src.y * x[1][0];
    b = src.x * x[0][1] + src.y * x[1][1];

    dst.x = a;
    dst.y = b;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::operator/= (T a) IMATH_NOEXCEPT
{
    x[0][0] /= a;
    x[0][1] /= a;
    x[0][2] /= a;
    x[1][0] /= a;
    x[1][1] /= a;
    x[1][2] /= a;
    x[2][0] /= a;
    x[2][1] /= a;
    x[2][2] /= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix33<T>
Matrix33<T>::operator/ (T a) const IMATH_NOEXCEPT
{
    return Matrix33 (x[0][0] / a,
                     x[0][1] / a,
                     x[0][2] / a,
                     x[1][0] / a,
                     x[1][1] / a,
                     x[1][2] / a,
                     x[2][0] / a,
                     x[2][1] / a,
                     x[2][2] / a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::transpose() IMATH_NOEXCEPT
{
    Matrix33 tmp (x[0][0], x[1][0], x[2][0], x[0][1], x[1][1], x[2][1], x[0][2], x[1][2], x[2][2]);
    *this = tmp;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix33<T>
Matrix33<T>::transposed() const IMATH_NOEXCEPT
{
    return Matrix33 (x[0][0],
                     x[1][0],
                     x[2][0],
                     x[0][1],
                     x[1][1],
                     x[2][1],
                     x[0][2],
                     x[1][2],
                     x[2][2]);
}

template <class T>
const inline Matrix33<T>&
Matrix33<T>::gjInvert (bool singExc)
{
    *this = gjInverse (singExc);
    return *this;
}

template <class T>
IMATH_HOSTDEVICE const inline Matrix33<T>&
Matrix33<T>::gjInvert() IMATH_NOEXCEPT
{
    *this = gjInverse();
    return *this;
}

template <class T>
inline Matrix33<T>
Matrix33<T>::gjInverse (bool singExc) const
{
    int i, j, k;
    Matrix33 s;
    Matrix33 t (*this);

    // Forward elimination

    for (i = 0; i < 2; i++)
    {
        int pivot = i;

        T pivotsize = t.x[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (j = i + 1; j < 3; j++)
        {
            T tmp = t.x[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot     = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            if (singExc)
                throw std::invalid_argument ("Cannot invert singular matrix.");

            return Matrix33();
        }

        if (pivot != i)
        {
            for (j = 0; j < 3; j++)
            {
                T tmp;

                tmp           = t.x[i][j];
                t.x[i][j]     = t.x[pivot][j];
                t.x[pivot][j] = tmp;

                tmp           = s.x[i][j];
                s.x[i][j]     = s.x[pivot][j];
                s.x[pivot][j] = tmp;
            }
        }

        for (j = i + 1; j < 3; j++)
        {
            T f = t.x[j][i] / t.x[i][i];

            for (k = 0; k < 3; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    // Backward substitution

    for (i = 2; i >= 0; --i)
    {
        T f;

        if ((f = t[i][i]) == 0)
        {
            if (singExc)
                throw std::invalid_argument ("Cannot invert singular matrix.");

            return Matrix33();
        }

        for (j = 0; j < 3; j++)
        {
            t.x[i][j] /= f;
            s.x[i][j] /= f;
        }

        for (j = 0; j < i; j++)
        {
            f = t.x[j][i];

            for (k = 0; k < 3; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    return s;
}

template <class T>
IMATH_HOSTDEVICE inline Matrix33<T>
Matrix33<T>::gjInverse() const IMATH_NOEXCEPT
{
    int i, j, k;
    Matrix33 s;
    Matrix33 t (*this);

    // Forward elimination

    for (i = 0; i < 2; i++)
    {
        int pivot = i;

        T pivotsize = t.x[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (j = i + 1; j < 3; j++)
        {
            T tmp = t.x[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot     = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            return Matrix33();
        }

        if (pivot != i)
        {
            for (j = 0; j < 3; j++)
            {
                T tmp;

                tmp           = t.x[i][j];
                t.x[i][j]     = t.x[pivot][j];
                t.x[pivot][j] = tmp;

                tmp           = s.x[i][j];
                s.x[i][j]     = s.x[pivot][j];
                s.x[pivot][j] = tmp;
            }
        }

        for (j = i + 1; j < 3; j++)
        {
            T f = t.x[j][i] / t.x[i][i];

            for (k = 0; k < 3; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    // Backward substitution

    for (i = 2; i >= 0; --i)
    {
        T f;

        if ((f = t.x[i][i]) == 0)
        {
            return Matrix33();
        }

        for (j = 0; j < 3; j++)
        {
            t.x[i][j] /= f;
            s.x[i][j] /= f;
        }

        for (j = 0; j < i; j++)
        {
            f = t.x[j][i];

            for (k = 0; k < 3; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    return s;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::invert (bool singExc)
{
    *this = inverse (singExc);
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::invert() IMATH_NOEXCEPT
{
    *this = inverse();
    return *this;
}

template <class T>
IMATH_CONSTEXPR14 inline Matrix33<T>
Matrix33<T>::inverse (bool singExc) const
{
    if (x[0][2] != 0 || x[1][2] != 0 || x[2][2] != 1)
    {
        Matrix33 s (x[1][1] * x[2][2] - x[2][1] * x[1][2],
                    x[2][1] * x[0][2] - x[0][1] * x[2][2],
                    x[0][1] * x[1][2] - x[1][1] * x[0][2],

                    x[2][0] * x[1][2] - x[1][0] * x[2][2],
                    x[0][0] * x[2][2] - x[2][0] * x[0][2],
                    x[1][0] * x[0][2] - x[0][0] * x[1][2],

                    x[1][0] * x[2][1] - x[2][0] * x[1][1],
                    x[2][0] * x[0][1] - x[0][0] * x[2][1],
                    x[0][0] * x[1][1] - x[1][0] * x[0][1]);

        T r = x[0][0] * s[0][0] + x[0][1] * s[1][0] + x[0][2] * s[2][0];

        if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    s.x[i][j] /= r;
                }
            }
        }
        else
        {
            T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    if (mr > IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
                    {
                        s.x[i][j] /= r;
                    }
                    else
                    {
                        if (singExc)
                            throw std::invalid_argument ("Cannot invert "
                                                         "singular matrix.");
                        return Matrix33();
                    }
                }
            }
        }

        return s;
    }
    else
    {
        Matrix33 s (x[1][1],
                    -x[0][1],
                    0,

                    -x[1][0],
                    x[0][0],
                    0,

                    0,
                    0,
                    1);

        T r = x[0][0] * x[1][1] - x[1][0] * x[0][1];

        if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
        {
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    s.x[i][j] /= r;
                }
            }
        }
        else
        {
            T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    if (mr > IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
                    {
                        s.x[i][j] /= r;
                    }
                    else
                    {
                        if (singExc)
                            throw std::invalid_argument ("Cannot invert "
                                                         "singular matrix.");
                        return Matrix33();
                    }
                }
            }
        }

        s.x[2][0] = -x[2][0] * s.x[0][0] - x[2][1] * s.x[1][0];
        s.x[2][1] = -x[2][0] * s.x[0][1] - x[2][1] * s.x[1][1];

        return s;
    }
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix33<T>
Matrix33<T>::inverse () const IMATH_NOEXCEPT
{
    if (x[0][2] != 0 || x[1][2] != 0 || x[2][2] != 1)
    {
        Matrix33 s (x[1][1] * x[2][2] - x[2][1] * x[1][2],
                    x[2][1] * x[0][2] - x[0][1] * x[2][2],
                    x[0][1] * x[1][2] - x[1][1] * x[0][2],

                    x[2][0] * x[1][2] - x[1][0] * x[2][2],
                    x[0][0] * x[2][2] - x[2][0] * x[0][2],
                    x[1][0] * x[0][2] - x[0][0] * x[1][2],

                    x[1][0] * x[2][1] - x[2][0] * x[1][1],
                    x[2][0] * x[0][1] - x[0][0] * x[2][1],
                    x[0][0] * x[1][1] - x[1][0] * x[0][1]);

        T r = x[0][0] * s.x[0][0] + x[0][1] * s.x[1][0] + x[0][2] * s.x[2][0];

        if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
        {
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    s.x[i][j] /= r;
                }
            }
        }
        else
        {
            T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    if (mr > IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
                    {
                        s.x[i][j] /= r;
                    }
                    else
                    {
                        return Matrix33();
                    }
                }
            }
        }

        return s;
    }
    else
    {
        Matrix33 s (x[1][1],
                    -x[0][1],
                    0,

                    -x[1][0],
                    x[0][0],
                    0,

                    0,
                    0,
                    1);

        T r = x[0][0] * x[1][1] - x[1][0] * x[0][1];

        if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
        {
            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    s.x[i][j] /= r;
                }
            }
        }
        else
        {
            T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

            for (int i = 0; i < 2; ++i)
            {
                for (int j = 0; j < 2; ++j)
                {
                    if (mr > IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
                    {
                        s.x[i][j] /= r;
                    }
                    else
                    {
                        return Matrix33();
                    }
                }
            }
        }

        s.x[2][0] = -x[2][0] * s.x[0][0] - x[2][1] * s.x[1][0];
        s.x[2][1] = -x[2][0] * s.x[0][1] - x[2][1] * s.x[1][1];

        return s;
    }
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T
Matrix33<T>::minorOf (const int r, const int c) const IMATH_NOEXCEPT
{
    int r0 = 0 + (r < 1 ? 1 : 0);
    int r1 = 1 + (r < 2 ? 1 : 0);
    int c0 = 0 + (c < 1 ? 1 : 0);
    int c1 = 1 + (c < 2 ? 1 : 0);

    return x[r0][c0] * x[r1][c1] - x[r1][c0] * x[r0][c1];
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Matrix33<T>::fastMinor (const int r0, const int r1, const int c0, const int c1) const IMATH_NOEXCEPT
{
    return x[r0][c0] * x[r1][c1] - x[r0][c1] * x[r1][c0];
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Matrix33<T>::determinant() const IMATH_NOEXCEPT
{
    return x[0][0] * (x[1][1] * x[2][2] - x[1][2] * x[2][1]) +
           x[0][1] * (x[1][2] * x[2][0] - x[1][0] * x[2][2]) +
           x[0][2] * (x[1][0] * x[2][1] - x[1][1] * x[2][0]);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Matrix33<T>::trace () const IMATH_NOEXCEPT
{
    return x[0][0] + x[1][1] + x[2][2];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline const Matrix33<T>&
Matrix33<T>::setRotation (S r) IMATH_NOEXCEPT
{
    S cos_r, sin_r;

    cos_r = cos ((T) r);
    sin_r = sin ((T) r);

    x[0][0] = cos_r;
    x[0][1] = sin_r;
    x[0][2] = 0;

    x[1][0] = -sin_r;
    x[1][1] = cos_r;
    x[1][2] = 0;

    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::rotate (S r) IMATH_NOEXCEPT
{
    *this *= Matrix33<T>().setRotation (r);
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::setScale (T s) IMATH_NOEXCEPT
{
    //
    // Set the matrix to a 2D homogeneous transform scale:
    //  | s 0 0 |
    //  | 0 s 0 |
    //  | 0 0 1 |
    //

    x[0][0] = s;
    x[0][1] = 0;
    x[0][2] = 0;
    x[1][0] = 0;
    x[1][1] = s;
    x[1][2] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::setScale (const Vec2<S>& s) IMATH_NOEXCEPT
{
    //
    // Set the matrix to a 2D homogeneous transform scale:
    //  | s.x  0   0 |
    //  |  0  s.y  0 |
    //  |  0   0   1 |
    //

    x[0][0] = s.x;
    x[0][1] = 0;
    x[0][2] = 0;
    x[1][0] = 0;
    x[1][1] = s.y;
    x[1][2] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::scale (const Vec2<S>& s) IMATH_NOEXCEPT
{
    x[0][0] *= s.x;
    x[0][1] *= s.x;
    x[0][2] *= s.x;

    x[1][0] *= s.y;
    x[1][1] *= s.y;
    x[1][2] *= s.y;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::setTranslation (const Vec2<S>& t) IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;

    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;

    x[2][0] = t.x;
    x[2][1] = t.y;
    x[2][2] = 1;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Vec2<T>
Matrix33<T>::translation() const IMATH_NOEXCEPT
{
    return Vec2<T> (x[2][0], x[2][1]);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::translate (const Vec2<S>& t) IMATH_NOEXCEPT
{
    x[2][0] += t.x * x[0][0] + t.y * x[1][0];
    x[2][1] += t.x * x[0][1] + t.y * x[1][1];
    x[2][2] += t.x * x[0][2] + t.y * x[1][2];

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::setShear (const S& xy) IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;

    x[1][0] = xy;
    x[1][1] = 1;
    x[1][2] = 0;

    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::setShear (const Vec2<S>& h) IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = h.y;
    x[0][2] = 0;

    x[1][0] = h.x;
    x[1][1] = 1;
    x[1][2] = 0;

    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::shear (const S& xy) IMATH_NOEXCEPT
{
    //
    // In this case, we don't need a temp. copy of the matrix
    // because we never use a value on the RHS after we've
    // changed it on the LHS.
    //

    x[1][0] += xy * x[0][0];
    x[1][1] += xy * x[0][1];
    x[1][2] += xy * x[0][2];

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix33<T>&
Matrix33<T>::shear (const Vec2<S>& h) IMATH_NOEXCEPT
{
    Matrix33<T> P (*this);

    x[0][0] = P.x[0][0] + h.y * P.x[1][0];
    x[0][1] = P.x[0][1] + h.y * P.x[1][1];
    x[0][2] = P.x[0][2] + h.y * P.x[1][2];

    x[1][0] = P.x[1][0] + h.x * P.x[0][0];
    x[1][1] = P.x[1][1] + h.x * P.x[0][1];
    x[1][2] = P.x[1][2] + h.x * P.x[0][2];

    return *this;
}

//---------------------------
// Implementation of Matrix44
//---------------------------

template <class T>
IMATH_HOSTDEVICE inline T*
Matrix44<T>::operator[] (int i) IMATH_NOEXCEPT
{
    return x[i];
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Matrix44<T>::operator[] (int i) const IMATH_NOEXCEPT
{
    return x[i];
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>::Matrix44() IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>::Matrix44 (T a) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = a;
    x[0][2] = a;
    x[0][3] = a;
    x[1][0] = a;
    x[1][1] = a;
    x[1][2] = a;
    x[1][3] = a;
    x[2][0] = a;
    x[2][1] = a;
    x[2][2] = a;
    x[2][3] = a;
    x[3][0] = a;
    x[3][1] = a;
    x[3][2] = a;
    x[3][3] = a;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>::Matrix44 (
    const T a[4][4]) IMATH_NOEXCEPT
{
    x[0][0] = a[0][0];
    x[0][1] = a[0][1];
    x[0][2] = a[0][2];
    x[0][3] = a[0][3];
    x[1][0] = a[1][0];
    x[1][1] = a[1][1];
    x[1][2] = a[1][2];
    x[1][3] = a[1][3];
    x[2][0] = a[2][0];
    x[2][1] = a[2][1];
    x[2][2] = a[2][2];
    x[2][3] = a[2][3];
    x[3][0] = a[3][0];
    x[3][1] = a[3][1];
    x[3][2] = a[3][2];
    x[3][3] = a[3][3];
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<
    T>::Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l, T m, T n, T o, T p) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = b;
    x[0][2] = c;
    x[0][3] = d;
    x[1][0] = e;
    x[1][1] = f;
    x[1][2] = g;
    x[1][3] = h;
    x[2][0] = i;
    x[2][1] = j;
    x[2][2] = k;
    x[2][3] = l;
    x[3][0] = m;
    x[3][1] = n;
    x[3][2] = o;
    x[3][3] = p;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>::Matrix44 (Matrix33<T> r, Vec3<T> t) IMATH_NOEXCEPT
{
    x[0][0] = r.x[0][0];
    x[0][1] = r.x[0][1];
    x[0][2] = r.x[0][2];
    x[0][3] = 0;
    x[1][0] = r.x[1][0];
    x[1][1] = r.x[1][1];
    x[1][2] = r.x[1][2];
    x[1][3] = 0;
    x[2][0] = r.x[2][0];
    x[2][1] = r.x[2][1];
    x[2][2] = r.x[2][2];
    x[2][3] = 0;
    x[3][0] = t.x;
    x[3][1] = t.y;
    x[3][2] = t.z;
    x[3][3] = 1;
}

template <class T> IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>::Matrix44 (const Matrix44& v) IMATH_NOEXCEPT
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[0][3] = v.x[0][3];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[1][3] = v.x[1][3];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    x[2][3] = v.x[2][3];
    x[3][0] = v.x[3][0];
    x[3][1] = v.x[3][1];
    x[3][2] = v.x[3][2];
    x[3][3] = v.x[3][3];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>::Matrix44 (const Matrix44<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = T (v.x[0][0]);
    x[0][1] = T (v.x[0][1]);
    x[0][2] = T (v.x[0][2]);
    x[0][3] = T (v.x[0][3]);
    x[1][0] = T (v.x[1][0]);
    x[1][1] = T (v.x[1][1]);
    x[1][2] = T (v.x[1][2]);
    x[1][3] = T (v.x[1][3]);
    x[2][0] = T (v.x[2][0]);
    x[2][1] = T (v.x[2][1]);
    x[2][2] = T (v.x[2][2]);
    x[2][3] = T (v.x[2][3]);
    x[3][0] = T (v.x[3][0]);
    x[3][1] = T (v.x[3][1]);
    x[3][2] = T (v.x[3][2]);
    x[3][3] = T (v.x[3][3]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator= (const Matrix44& v) IMATH_NOEXCEPT
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[0][3] = v.x[0][3];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[1][3] = v.x[1][3];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    x[2][3] = v.x[2][3];
    x[3][0] = v.x[3][0];
    x[3][1] = v.x[3][1];
    x[3][2] = v.x[3][2];
    x[3][3] = v.x[3][3];
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator= (T a) IMATH_NOEXCEPT
{
    x[0][0] = a;
    x[0][1] = a;
    x[0][2] = a;
    x[0][3] = a;
    x[1][0] = a;
    x[1][1] = a;
    x[1][2] = a;
    x[1][3] = a;
    x[2][0] = a;
    x[2][1] = a;
    x[2][2] = a;
    x[2][3] = a;
    x[3][0] = a;
    x[3][1] = a;
    x[3][2] = a;
    x[3][3] = a;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE inline T*
Matrix44<T>::getValue () IMATH_NOEXCEPT
{
    return (T*) &x[0][0];
}

template <class T>
IMATH_HOSTDEVICE inline const T*
Matrix44<T>::getValue() const IMATH_NOEXCEPT
{
    return (const T*) &x[0][0];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix44<T>::getValue (Matrix44<S>& v) const IMATH_NOEXCEPT
{
    v.x[0][0] = x[0][0];
    v.x[0][1] = x[0][1];
    v.x[0][2] = x[0][2];
    v.x[0][3] = x[0][3];
    v.x[1][0] = x[1][0];
    v.x[1][1] = x[1][1];
    v.x[1][2] = x[1][2];
    v.x[1][3] = x[1][3];
    v.x[2][0] = x[2][0];
    v.x[2][1] = x[2][1];
    v.x[2][2] = x[2][2];
    v.x[2][3] = x[2][3];
    v.x[3][0] = x[3][0];
    v.x[3][1] = x[3][1];
    v.x[3][2] = x[3][2];
    v.x[3][3] = x[3][3];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>&
Matrix44<T>::setValue (const Matrix44<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = T(v.x[0][0]);
    x[0][1] = T(v.x[0][1]);
    x[0][2] = T(v.x[0][2]);
    x[0][3] = T(v.x[0][3]);
    x[1][0] = T(v.x[1][0]);
    x[1][1] = T(v.x[1][1]);
    x[1][2] = T(v.x[1][2]);
    x[1][3] = T(v.x[1][3]);
    x[2][0] = T(v.x[2][0]);
    x[2][1] = T(v.x[2][1]);
    x[2][2] = T(v.x[2][2]);
    x[2][3] = T(v.x[2][3]);
    x[3][0] = T(v.x[3][0]);
    x[3][1] = T(v.x[3][1]);
    x[3][2] = T(v.x[3][2]);
    x[3][3] = T(v.x[3][3]);
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>&
Matrix44<T>::setTheMatrix (const Matrix44<S>& v) IMATH_NOEXCEPT
{
    x[0][0] = v.x[0][0];
    x[0][1] = v.x[0][1];
    x[0][2] = v.x[0][2];
    x[0][3] = v.x[0][3];
    x[1][0] = v.x[1][0];
    x[1][1] = v.x[1][1];
    x[1][2] = v.x[1][2];
    x[1][3] = v.x[1][3];
    x[2][0] = v.x[2][0];
    x[2][1] = v.x[2][1];
    x[2][2] = v.x[2][2];
    x[2][3] = v.x[2][3];
    x[3][0] = v.x[3][0];
    x[3][1] = v.x[3][1];
    x[3][2] = v.x[3][2];
    x[3][3] = v.x[3][3];
    return *this;
}

template <class T>
IMATH_HOSTDEVICE inline void
Matrix44<T>::makeIdentity() IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Matrix44<T>::operator== (const Matrix44& v) const IMATH_NOEXCEPT
{
    return x[0][0] == v.x[0][0] && x[0][1] == v.x[0][1] && x[0][2] == v.x[0][2] &&
           x[0][3] == v.x[0][3] && x[1][0] == v.x[1][0] && x[1][1] == v.x[1][1] &&
           x[1][2] == v.x[1][2] && x[1][3] == v.x[1][3] && x[2][0] == v.x[2][0] &&
           x[2][1] == v.x[2][1] && x[2][2] == v.x[2][2] && x[2][3] == v.x[2][3] &&
           x[3][0] == v.x[3][0] && x[3][1] == v.x[3][1] && x[3][2] == v.x[3][2] &&
           x[3][3] == v.x[3][3];
}

template <class T>
IMATH_HOSTDEVICE constexpr inline bool
Matrix44<T>::operator!= (const Matrix44& v) const IMATH_NOEXCEPT
{
    return x[0][0] != v.x[0][0] || x[0][1] != v.x[0][1] || x[0][2] != v.x[0][2] ||
           x[0][3] != v.x[0][3] || x[1][0] != v.x[1][0] || x[1][1] != v.x[1][1] ||
           x[1][2] != v.x[1][2] || x[1][3] != v.x[1][3] || x[2][0] != v.x[2][0] ||
           x[2][1] != v.x[2][1] || x[2][2] != v.x[2][2] || x[2][3] != v.x[2][3] ||
           x[3][0] != v.x[3][0] || x[3][1] != v.x[3][1] || x[3][2] != v.x[3][2] ||
           x[3][3] != v.x[3][3];
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Matrix44<T>::equalWithAbsError (const Matrix44<T>& m, T e) const IMATH_NOEXCEPT
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this).x[i][j], m.x[i][j], e))
                return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline bool
Matrix44<T>::equalWithRelError (const Matrix44<T>& m, T e) const IMATH_NOEXCEPT
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this).x[i][j], m.x[i][j], e))
                return false;

    return true;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator+= (const Matrix44<T>& v) IMATH_NOEXCEPT
{
    x[0][0] += v.x[0][0];
    x[0][1] += v.x[0][1];
    x[0][2] += v.x[0][2];
    x[0][3] += v.x[0][3];
    x[1][0] += v.x[1][0];
    x[1][1] += v.x[1][1];
    x[1][2] += v.x[1][2];
    x[1][3] += v.x[1][3];
    x[2][0] += v.x[2][0];
    x[2][1] += v.x[2][1];
    x[2][2] += v.x[2][2];
    x[2][3] += v.x[2][3];
    x[3][0] += v.x[3][0];
    x[3][1] += v.x[3][1];
    x[3][2] += v.x[3][2];
    x[3][3] += v.x[3][3];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator+= (T a) IMATH_NOEXCEPT
{
    x[0][0] += a;
    x[0][1] += a;
    x[0][2] += a;
    x[0][3] += a;
    x[1][0] += a;
    x[1][1] += a;
    x[1][2] += a;
    x[1][3] += a;
    x[2][0] += a;
    x[2][1] += a;
    x[2][2] += a;
    x[2][3] += a;
    x[3][0] += a;
    x[3][1] += a;
    x[3][2] += a;
    x[3][3] += a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix44<T>
Matrix44<T>::operator+ (const Matrix44<T>& v) const IMATH_NOEXCEPT
{
    return Matrix44 (x[0][0] + v.x[0][0],
                     x[0][1] + v.x[0][1],
                     x[0][2] + v.x[0][2],
                     x[0][3] + v.x[0][3],
                     x[1][0] + v.x[1][0],
                     x[1][1] + v.x[1][1],
                     x[1][2] + v.x[1][2],
                     x[1][3] + v.x[1][3],
                     x[2][0] + v.x[2][0],
                     x[2][1] + v.x[2][1],
                     x[2][2] + v.x[2][2],
                     x[2][3] + v.x[2][3],
                     x[3][0] + v.x[3][0],
                     x[3][1] + v.x[3][1],
                     x[3][2] + v.x[3][2],
                     x[3][3] + v.x[3][3]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator-= (const Matrix44<T>& v) IMATH_NOEXCEPT
{
    x[0][0] -= v.x[0][0];
    x[0][1] -= v.x[0][1];
    x[0][2] -= v.x[0][2];
    x[0][3] -= v.x[0][3];
    x[1][0] -= v.x[1][0];
    x[1][1] -= v.x[1][1];
    x[1][2] -= v.x[1][2];
    x[1][3] -= v.x[1][3];
    x[2][0] -= v.x[2][0];
    x[2][1] -= v.x[2][1];
    x[2][2] -= v.x[2][2];
    x[2][3] -= v.x[2][3];
    x[3][0] -= v.x[3][0];
    x[3][1] -= v.x[3][1];
    x[3][2] -= v.x[3][2];
    x[3][3] -= v.x[3][3];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator-= (T a) IMATH_NOEXCEPT
{
    x[0][0] -= a;
    x[0][1] -= a;
    x[0][2] -= a;
    x[0][3] -= a;
    x[1][0] -= a;
    x[1][1] -= a;
    x[1][2] -= a;
    x[1][3] -= a;
    x[2][0] -= a;
    x[2][1] -= a;
    x[2][2] -= a;
    x[2][3] -= a;
    x[3][0] -= a;
    x[3][1] -= a;
    x[3][2] -= a;
    x[3][3] -= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix44<T>
Matrix44<T>::operator- (const Matrix44<T>& v) const IMATH_NOEXCEPT
{
    return Matrix44 (x[0][0] - v.x[0][0],
                     x[0][1] - v.x[0][1],
                     x[0][2] - v.x[0][2],
                     x[0][3] - v.x[0][3],
                     x[1][0] - v.x[1][0],
                     x[1][1] - v.x[1][1],
                     x[1][2] - v.x[1][2],
                     x[1][3] - v.x[1][3],
                     x[2][0] - v.x[2][0],
                     x[2][1] - v.x[2][1],
                     x[2][2] - v.x[2][2],
                     x[2][3] - v.x[2][3],
                     x[3][0] - v.x[3][0],
                     x[3][1] - v.x[3][1],
                     x[3][2] - v.x[3][2],
                     x[3][3] - v.x[3][3]);
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix44<T>
Matrix44<T>::operator-() const IMATH_NOEXCEPT
{
    return Matrix44 (-x[0][0],
                     -x[0][1],
                     -x[0][2],
                     -x[0][3],
                     -x[1][0],
                     -x[1][1],
                     -x[1][2],
                     -x[1][3],
                     -x[2][0],
                     -x[2][1],
                     -x[2][2],
                     -x[2][3],
                     -x[3][0],
                     -x[3][1],
                     -x[3][2],
                     -x[3][3]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::negate() IMATH_NOEXCEPT
{
    x[0][0] = -x[0][0];
    x[0][1] = -x[0][1];
    x[0][2] = -x[0][2];
    x[0][3] = -x[0][3];
    x[1][0] = -x[1][0];
    x[1][1] = -x[1][1];
    x[1][2] = -x[1][2];
    x[1][3] = -x[1][3];
    x[2][0] = -x[2][0];
    x[2][1] = -x[2][1];
    x[2][2] = -x[2][2];
    x[2][3] = -x[2][3];
    x[3][0] = -x[3][0];
    x[3][1] = -x[3][1];
    x[3][2] = -x[3][2];
    x[3][3] = -x[3][3];

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator*= (T a) IMATH_NOEXCEPT
{
    x[0][0] *= a;
    x[0][1] *= a;
    x[0][2] *= a;
    x[0][3] *= a;
    x[1][0] *= a;
    x[1][1] *= a;
    x[1][2] *= a;
    x[1][3] *= a;
    x[2][0] *= a;
    x[2][1] *= a;
    x[2][2] *= a;
    x[2][3] *= a;
    x[3][0] *= a;
    x[3][1] *= a;
    x[3][2] *= a;
    x[3][3] *= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix44<T>
Matrix44<T>::operator* (T a) const IMATH_NOEXCEPT
{
    return Matrix44 (x[0][0] * a,
                     x[0][1] * a,
                     x[0][2] * a,
                     x[0][3] * a,
                     x[1][0] * a,
                     x[1][1] * a,
                     x[1][2] * a,
                     x[1][3] * a,
                     x[2][0] * a,
                     x[2][1] * a,
                     x[2][2] * a,
                     x[2][3] * a,
                     x[3][0] * a,
                     x[3][1] * a,
                     x[3][2] * a,
                     x[3][3] * a);
}

/// Matrix-scalar multiplication
template <class T>
IMATH_HOSTDEVICE inline Matrix44<T>
operator* (T a, const Matrix44<T>& v) IMATH_NOEXCEPT
{
    return v * a;
}


template <class T>
IMATH_HOSTDEVICE inline IMATH_CONSTEXPR14 Matrix44<T>
Matrix44<T>::multiply (const Matrix44 &a, const Matrix44 &b) IMATH_NOEXCEPT
{
    const auto a00 = a.x[0][0];
    const auto a01 = a.x[0][1];
    const auto a02 = a.x[0][2];
    const auto a03 = a.x[0][3];

    const auto c00  = a00 * b.x[0][0] + a01 * b.x[1][0] + a02 * b.x[2][0] + a03 * b.x[3][0];
    const auto c01  = a00 * b.x[0][1] + a01 * b.x[1][1] + a02 * b.x[2][1] + a03 * b.x[3][1];
    const auto c02  = a00 * b.x[0][2] + a01 * b.x[1][2] + a02 * b.x[2][2] + a03 * b.x[3][2];
    const auto c03  = a00 * b.x[0][3] + a01 * b.x[1][3] + a02 * b.x[2][3] + a03 * b.x[3][3];

    const auto a10 = a.x[1][0];
    const auto a11 = a.x[1][1];
    const auto a12 = a.x[1][2];
    const auto a13 = a.x[1][3];

    const auto c10  = a10 * b.x[0][0] + a11 * b.x[1][0] + a12 * b.x[2][0] + a13 * b.x[3][0];
    const auto c11  = a10 * b.x[0][1] + a11 * b.x[1][1] + a12 * b.x[2][1] + a13 * b.x[3][1];
    const auto c12  = a10 * b.x[0][2] + a11 * b.x[1][2] + a12 * b.x[2][2] + a13 * b.x[3][2];
    const auto c13  = a10 * b.x[0][3] + a11 * b.x[1][3] + a12 * b.x[2][3] + a13 * b.x[3][3];

    const auto a20 = a.x[2][0];
    const auto a21 = a.x[2][1];
    const auto a22 = a.x[2][2];
    const auto a23 = a.x[2][3];

    const auto c20 = a20 * b.x[0][0] + a21 * b.x[1][0] + a22 * b.x[2][0] + a23 * b.x[3][0];
    const auto c21 = a20 * b.x[0][1] + a21 * b.x[1][1] + a22 * b.x[2][1] + a23 * b.x[3][1];
    const auto c22 = a20 * b.x[0][2] + a21 * b.x[1][2] + a22 * b.x[2][2] + a23 * b.x[3][2];
    const auto c23 = a20 * b.x[0][3] + a21 * b.x[1][3] + a22 * b.x[2][3] + a23 * b.x[3][3];

    const auto a30 = a.x[3][0];
    const auto a31 = a.x[3][1];
    const auto a32 = a.x[3][2];
    const auto a33 = a.x[3][3];

    const auto c30 = a30 * b.x[0][0] + a31 * b.x[1][0] + a32 * b.x[2][0] + a33 * b.x[3][0];
    const auto c31 = a30 * b.x[0][1] + a31 * b.x[1][1] + a32 * b.x[2][1] + a33 * b.x[3][1];
    const auto c32 = a30 * b.x[0][2] + a31 * b.x[1][2] + a32 * b.x[2][2] + a33 * b.x[3][2];
    const auto c33 = a30 * b.x[0][3] + a31 * b.x[1][3] + a32 * b.x[2][3] + a33 * b.x[3][3];
    return Matrix44(c00, c01, c02, c03,
                    c10, c11, c12, c13,
                    c20, c21, c22, c23,
                    c30, c31, c32, c33);
}


template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator*= (const Matrix44<T>& v) IMATH_NOEXCEPT
{
    *this = multiply(*this, v);
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>
Matrix44<T>::operator* (const Matrix44<T>& v) const IMATH_NOEXCEPT
{
    return multiply(*this, v);
}

template <class T>
IMATH_HOSTDEVICE inline void
Matrix44<T>::multiply (const Matrix44<T>& a, const Matrix44<T>& b, Matrix44<T>& c) IMATH_NOEXCEPT
{
    c = multiply(a, b);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix44<T>::multVecMatrix (const Vec3<S>& src, Vec3<S>& dst) const IMATH_NOEXCEPT
{
    S a, b, c, w;

    a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0] + x[3][0];
    b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1] + x[3][1];
    c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2] + x[3][2];
    w = src.x * x[0][3] + src.y * x[1][3] + src.z * x[2][3] + x[3][3];

    dst.x = a / w;
    dst.y = b / w;
    dst.z = c / w;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline void
Matrix44<T>::multDirMatrix (const Vec3<S>& src, Vec3<S>& dst) const IMATH_NOEXCEPT
{
    S a, b, c;

    a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0];
    b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1];
    c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2];

    dst.x = a;
    dst.y = b;
    dst.z = c;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::operator/= (T a) IMATH_NOEXCEPT
{
    x[0][0] /= a;
    x[0][1] /= a;
    x[0][2] /= a;
    x[0][3] /= a;
    x[1][0] /= a;
    x[1][1] /= a;
    x[1][2] /= a;
    x[1][3] /= a;
    x[2][0] /= a;
    x[2][1] /= a;
    x[2][2] /= a;
    x[2][3] /= a;
    x[3][0] /= a;
    x[3][1] /= a;
    x[3][2] /= a;
    x[3][3] /= a;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix44<T>
Matrix44<T>::operator/ (T a) const IMATH_NOEXCEPT
{
    return Matrix44 (x[0][0] / a,
                     x[0][1] / a,
                     x[0][2] / a,
                     x[0][3] / a,
                     x[1][0] / a,
                     x[1][1] / a,
                     x[1][2] / a,
                     x[1][3] / a,
                     x[2][0] / a,
                     x[2][1] / a,
                     x[2][2] / a,
                     x[2][3] / a,
                     x[3][0] / a,
                     x[3][1] / a,
                     x[3][2] / a,
                     x[3][3] / a);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::transpose() IMATH_NOEXCEPT
{
    Matrix44 tmp (x[0][0],
                  x[1][0],
                  x[2][0],
                  x[3][0],
                  x[0][1],
                  x[1][1],
                  x[2][1],
                  x[3][1],
                  x[0][2],
                  x[1][2],
                  x[2][2],
                  x[3][2],
                  x[0][3],
                  x[1][3],
                  x[2][3],
                  x[3][3]);
    *this = tmp;
    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline Matrix44<T>
Matrix44<T>::transposed() const IMATH_NOEXCEPT
{
    return Matrix44 (x[0][0],
                     x[1][0],
                     x[2][0],
                     x[3][0],
                     x[0][1],
                     x[1][1],
                     x[2][1],
                     x[3][1],
                     x[0][2],
                     x[1][2],
                     x[2][2],
                     x[3][2],
                     x[0][3],
                     x[1][3],
                     x[2][3],
                     x[3][3]);
}

template <class T>
IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::gjInvert (bool singExc)
{
    *this = gjInverse (singExc);
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::gjInvert() IMATH_NOEXCEPT
{
    *this = gjInverse();
    return *this;
}

template <class T>
inline Matrix44<T>
Matrix44<T>::gjInverse (bool singExc) const
{
    int i, j, k;
    Matrix44 s;
    Matrix44 t (*this);

    // Forward elimination

    for (i = 0; i < 3; i++)
    {
        int pivot = i;

        T pivotsize = t.x[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (j = i + 1; j < 4; j++)
        {
            T tmp = t.x[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot     = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            if (singExc)
                throw std::invalid_argument ("Cannot invert singular matrix.");

            return Matrix44();
        }

        if (pivot != i)
        {
            for (j = 0; j < 4; j++)
            {
                T tmp;

                tmp           = t.x[i][j];
                t.x[i][j]     = t.x[pivot][j];
                t.x[pivot][j] = tmp;

                tmp           = s.x[i][j];
                s.x[i][j]     = s.x[pivot][j];
                s.x[pivot][j] = tmp;
            }
        }

        for (j = i + 1; j < 4; j++)
        {
            T f = t.x[j][i] / t.x[i][i];

            for (k = 0; k < 4; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    // Backward substitution

    for (i = 3; i >= 0; --i)
    {
        T f;

        if ((f = t.x[i][i]) == 0)
        {
            if (singExc)
                throw std::invalid_argument ("Cannot invert singular matrix.");

            return Matrix44();
        }

        for (j = 0; j < 4; j++)
        {
            t.x[i][j] /= f;
            s.x[i][j] /= f;
        }

        for (j = 0; j < i; j++)
        {
            f = t.x[j][i];

            for (k = 0; k < 4; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    return s;
}

template <class T>
IMATH_HOSTDEVICE inline Matrix44<T>
Matrix44<T>::gjInverse() const IMATH_NOEXCEPT
{
    int i, j, k;
    Matrix44 s;
    Matrix44 t (*this);

    // Forward elimination

    for (i = 0; i < 3; i++)
    {
        int pivot = i;

        T pivotsize = t.x[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (j = i + 1; j < 4; j++)
        {
            T tmp = t.x[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot     = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            return Matrix44();
        }

        if (pivot != i)
        {
            for (j = 0; j < 4; j++)
            {
                T tmp;

                tmp           = t.x[i][j];
                t.x[i][j]     = t.x[pivot][j];
                t.x[pivot][j] = tmp;

                tmp           = s.x[i][j];
                s.x[i][j]     = s.x[pivot][j];
                s.x[pivot][j] = tmp;
            }
        }

        for (j = i + 1; j < 4; j++)
        {
            T f = t.x[j][i] / t.x[i][i];

            for (k = 0; k < 4; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    // Backward substitution

    for (i = 3; i >= 0; --i)
    {
        T f;

        if ((f = t.x[i][i]) == 0)
        {
            return Matrix44();
        }

        for (j = 0; j < 4; j++)
        {
            t.x[i][j] /= f;
            s.x[i][j] /= f;
        }

        for (j = 0; j < i; j++)
        {
            f = t.x[j][i];

            for (k = 0; k < 4; k++)
            {
                t.x[j][k] -= f * t.x[i][k];
                s.x[j][k] -= f * s.x[i][k];
            }
        }
    }

    return s;
}

template <class T>
IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::invert (bool singExc)
{
    *this = inverse (singExc);
    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::invert() IMATH_NOEXCEPT
{
    *this = inverse();
    return *this;
}

template <class T>
IMATH_CONSTEXPR14 inline Matrix44<T>
Matrix44<T>::inverse (bool singExc) const
{
    if (x[0][3] != 0 || x[1][3] != 0 || x[2][3] != 0 || x[3][3] != 1)
        return gjInverse (singExc);

    Matrix44 s (x[1][1] * x[2][2] - x[2][1] * x[1][2],
                x[2][1] * x[0][2] - x[0][1] * x[2][2],
                x[0][1] * x[1][2] - x[1][1] * x[0][2],
                0,

                x[2][0] * x[1][2] - x[1][0] * x[2][2],
                x[0][0] * x[2][2] - x[2][0] * x[0][2],
                x[1][0] * x[0][2] - x[0][0] * x[1][2],
                0,

                x[1][0] * x[2][1] - x[2][0] * x[1][1],
                x[2][0] * x[0][1] - x[0][0] * x[2][1],
                x[0][0] * x[1][1] - x[1][0] * x[0][1],
                0,

                0,
                0,
                0,
                1);

    T r = x[0][0] * s.x[0][0] + x[0][1] * s.x[1][0] + x[0][2] * s.x[2][0];

    if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                s.x[i][j] /= r;
            }
        }
    }
    else
    {
        T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (mr > IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
                {
                    s.x[i][j] /= r;
                }
                else
                {
                    if (singExc)
                        throw std::invalid_argument ("Cannot invert singular matrix.");

                    return Matrix44();
                }
            }
        }
    }

    s.x[3][0] = -x[3][0] * s.x[0][0] - x[3][1] * s.x[1][0] - x[3][2] * s.x[2][0];
    s.x[3][1] = -x[3][0] * s.x[0][1] - x[3][1] * s.x[1][1] - x[3][2] * s.x[2][1];
    s.x[3][2] = -x[3][0] * s.x[0][2] - x[3][1] * s.x[1][2] - x[3][2] * s.x[2][2];

    return s;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline Matrix44<T>
Matrix44<T>::inverse() const IMATH_NOEXCEPT
{
    if (x[0][3] != 0 || x[1][3] != 0 || x[2][3] != 0 || x[3][3] != 1)
        return gjInverse();

    Matrix44 s (x[1][1] * x[2][2] - x[2][1] * x[1][2],
                x[2][1] * x[0][2] - x[0][1] * x[2][2],
                x[0][1] * x[1][2] - x[1][1] * x[0][2],
                0,

                x[2][0] * x[1][2] - x[1][0] * x[2][2],
                x[0][0] * x[2][2] - x[2][0] * x[0][2],
                x[1][0] * x[0][2] - x[0][0] * x[1][2],
                0,

                x[1][0] * x[2][1] - x[2][0] * x[1][1],
                x[2][0] * x[0][1] - x[0][0] * x[2][1],
                x[0][0] * x[1][1] - x[1][0] * x[0][1],
                0,

                0,
                0,
                0,
                1);

    T r = x[0][0] * s.x[0][0] + x[0][1] * s.x[1][0] + x[0][2] * s.x[2][0];

    if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                s.x[i][j] /= r;
            }
        }
    }
    else
    {
        T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / std::numeric_limits<T>::min();

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (mr > IMATH_INTERNAL_NAMESPACE::abs (s.x[i][j]))
                {
                    s.x[i][j] /= r;
                }
                else
                {
                    return Matrix44();
                }
            }
        }
    }

    s.x[3][0] = -x[3][0] * s.x[0][0] - x[3][1] * s.x[1][0] - x[3][2] * s.x[2][0];
    s.x[3][1] = -x[3][0] * s.x[0][1] - x[3][1] * s.x[1][1] - x[3][2] * s.x[2][1];
    s.x[3][2] = -x[3][0] * s.x[0][2] - x[3][1] * s.x[1][2] - x[3][2] * s.x[2][2];

    return s;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Matrix44<T>::fastMinor (const int r0,
                        const int r1,
                        const int r2,
                        const int c0,
                        const int c1,
                        const int c2) const IMATH_NOEXCEPT
{
    return x[r0][c0] * (x[r1][c1] * x[r2][c2] - x[r1][c2] * x[r2][c1]) +
           x[r0][c1] * (x[r1][c2] * x[r2][c0] - x[r1][c0] * x[r2][c2]) +
           x[r0][c2] * (x[r1][c0] * x[r2][c1] - x[r1][c1] * x[r2][c0]);
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T
Matrix44<T>::minorOf (const int r, const int c) const IMATH_NOEXCEPT
{
    int r0 = 0 + (r < 1 ? 1 : 0);
    int r1 = 1 + (r < 2 ? 1 : 0);
    int r2 = 2 + (r < 3 ? 1 : 0);
    int c0 = 0 + (c < 1 ? 1 : 0);
    int c1 = 1 + (c < 2 ? 1 : 0);
    int c2 = 2 + (c < 3 ? 1 : 0);

    Matrix33<T> working (x[r0][c0],
                         x[r1][c0],
                         x[r2][c0],
                         x[r0][c1],
                         x[r1][c1],
                         x[r2][c1],
                         x[r0][c2],
                         x[r1][c2],
                         x[r2][c2]);

    return working.determinant();
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline T
Matrix44<T>::determinant() const IMATH_NOEXCEPT
{
    T sum = (T) 0;

    if (x[0][3] != 0.)
        sum -= x[0][3] * fastMinor (1, 2, 3, 0, 1, 2);
    if (x[1][3] != 0.)
        sum += x[1][3] * fastMinor (0, 2, 3, 0, 1, 2);
    if (x[2][3] != 0.)
        sum -= x[2][3] * fastMinor (0, 1, 3, 0, 1, 2);
    if (x[3][3] != 0.)
        sum += x[3][3] * fastMinor (0, 1, 2, 0, 1, 2);

    return sum;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline T
Matrix44<T>::trace () const IMATH_NOEXCEPT
{
    return x[0][0] + x[1][1] + x[2][2] + x[3][3];
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline const Matrix44<T>&
Matrix44<T>::setEulerAngles (const Vec3<S>& r) IMATH_NOEXCEPT
{
    S cos_rz, sin_rz, cos_ry, sin_ry, cos_rx, sin_rx;

    cos_rz = cos ((T) r.z);
    cos_ry = cos ((T) r.y);
    cos_rx = cos ((T) r.x);

    sin_rz = sin ((T) r.z);
    sin_ry = sin ((T) r.y);
    sin_rx = sin ((T) r.x);

    x[0][0] = cos_rz * cos_ry;
    x[0][1] = sin_rz * cos_ry;
    x[0][2] = -sin_ry;
    x[0][3] = 0;

    x[1][0] = -sin_rz * cos_rx + cos_rz * sin_ry * sin_rx;
    x[1][1] = cos_rz * cos_rx + sin_rz * sin_ry * sin_rx;
    x[1][2] = cos_ry * sin_rx;
    x[1][3] = 0;

    x[2][0] = sin_rz * sin_rx + cos_rz * sin_ry * cos_rx;
    x[2][1] = -cos_rz * sin_rx + sin_rz * sin_ry * cos_rx;
    x[2][2] = cos_ry * cos_rx;
    x[2][3] = 0;

    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::setAxisAngle (const Vec3<S>& axis, S angle) IMATH_NOEXCEPT
{
    Vec3<S> unit (axis.normalized());
    S sine   = std::sin (angle);
    S cosine = std::cos (angle);

    x[0][0] = unit.x * unit.x * (1 - cosine) + cosine;
    x[0][1] = unit.x * unit.y * (1 - cosine) + unit.z * sine;
    x[0][2] = unit.x * unit.z * (1 - cosine) - unit.y * sine;
    x[0][3] = 0;

    x[1][0] = unit.x * unit.y * (1 - cosine) - unit.z * sine;
    x[1][1] = unit.y * unit.y * (1 - cosine) + cosine;
    x[1][2] = unit.y * unit.z * (1 - cosine) + unit.x * sine;
    x[1][3] = 0;

    x[2][0] = unit.x * unit.z * (1 - cosine) + unit.y * sine;
    x[2][1] = unit.y * unit.z * (1 - cosine) - unit.x * sine;
    x[2][2] = unit.z * unit.z * (1 - cosine) + cosine;
    x[2][3] = 0;

    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE inline const Matrix44<T>&
Matrix44<T>::rotate (const Vec3<S>& r) IMATH_NOEXCEPT
{
    S cos_rz, sin_rz, cos_ry, sin_ry, cos_rx, sin_rx;
    S m00, m01, m02;
    S m10, m11, m12;
    S m20, m21, m22;

    cos_rz = cos ((S) r.z);
    cos_ry = cos ((S) r.y);
    cos_rx = cos ((S) r.x);

    sin_rz = sin ((S) r.z);
    sin_ry = sin ((S) r.y);
    sin_rx = sin ((S) r.x);

    m00 = cos_rz * cos_ry;
    m01 = sin_rz * cos_ry;
    m02 = -sin_ry;
    m10 = -sin_rz * cos_rx + cos_rz * sin_ry * sin_rx;
    m11 = cos_rz * cos_rx + sin_rz * sin_ry * sin_rx;
    m12 = cos_ry * sin_rx;
    m20 = -sin_rz * -sin_rx + cos_rz * sin_ry * cos_rx;
    m21 = cos_rz * -sin_rx + sin_rz * sin_ry * cos_rx;
    m22 = cos_ry * cos_rx;

    Matrix44<T> P (*this);

    x[0][0] = P.x[0][0] * m00 + P.x[1][0] * m01 + P.x[2][0] * m02;
    x[0][1] = P.x[0][1] * m00 + P.x[1][1] * m01 + P.x[2][1] * m02;
    x[0][2] = P.x[0][2] * m00 + P.x[1][2] * m01 + P.x[2][2] * m02;
    x[0][3] = P.x[0][3] * m00 + P.x[1][3] * m01 + P.x[2][3] * m02;

    x[1][0] = P.x[0][0] * m10 + P.x[1][0] * m11 + P.x[2][0] * m12;
    x[1][1] = P.x[0][1] * m10 + P.x[1][1] * m11 + P.x[2][1] * m12;
    x[1][2] = P.x[0][2] * m10 + P.x[1][2] * m11 + P.x[2][2] * m12;
    x[1][3] = P.x[0][3] * m10 + P.x[1][3] * m11 + P.x[2][3] * m12;

    x[2][0] = P.x[0][0] * m20 + P.x[1][0] * m21 + P.x[2][0] * m22;
    x[2][1] = P.x[0][1] * m20 + P.x[1][1] * m21 + P.x[2][1] * m22;
    x[2][2] = P.x[0][2] * m20 + P.x[1][2] * m21 + P.x[2][2] * m22;
    x[2][3] = P.x[0][3] * m20 + P.x[1][3] * m21 + P.x[2][3] * m22;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::setScale (T s) IMATH_NOEXCEPT
{
    //
    // Set the matrix to a 3D homogeneous transform scale:
    //  | s 0 0 0 |
    //  | 0 s 0 0 |
    //  | 0 0 s 0 |
    //  | 0 0 0 1 |
    //

    x[0][0] = s;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = s;
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = s;
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::setScale (const Vec3<S>& s) IMATH_NOEXCEPT
{
    //
    // Set the matrix to a 3D homogeneous transform scale:
    //  | s.x  0   0   0 |
    //  |  0  s.y  0   0 |
    //  |  0   0  s.z  0 |
    //  |  0   0   0   1 |
    //

    x[0][0] = s.x;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;
    x[1][0] = 0;
    x[1][1] = s.y;
    x[1][2] = 0;
    x[1][3] = 0;
    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = s.z;
    x[2][3] = 0;
    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;
    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::scale (const Vec3<S>& s) IMATH_NOEXCEPT
{
    x[0][0] *= s.x;
    x[0][1] *= s.x;
    x[0][2] *= s.x;
    x[0][3] *= s.x;

    x[1][0] *= s.y;
    x[1][1] *= s.y;
    x[1][2] *= s.y;
    x[1][3] *= s.y;

    x[2][0] *= s.z;
    x[2][1] *= s.z;
    x[2][2] *= s.z;
    x[2][3] *= s.z;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::setTranslation (const Vec3<S>& t) IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;

    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;

    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;
    x[2][3] = 0;

    x[3][0] = t.x;
    x[3][1] = t.y;
    x[3][2] = t.z;
    x[3][3] = 1;

    return *this;
}

template <class T>
IMATH_HOSTDEVICE constexpr inline const Vec3<T>
Matrix44<T>::translation() const IMATH_NOEXCEPT
{
    return Vec3<T> (x[3][0], x[3][1], x[3][2]);
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::translate (const Vec3<S>& t) IMATH_NOEXCEPT
{
    x[3][0] += t.x * x[0][0] + t.y * x[1][0] + t.z * x[2][0];
    x[3][1] += t.x * x[0][1] + t.y * x[1][1] + t.z * x[2][1];
    x[3][2] += t.x * x[0][2] + t.y * x[1][2] + t.z * x[2][2];
    x[3][3] += t.x * x[0][3] + t.y * x[1][3] + t.z * x[2][3];

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::setShear (const Vec3<S>& h) IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;

    x[1][0] = h.x;
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;

    x[2][0] = h.y;
    x[2][1] = h.z;
    x[2][2] = 1;
    x[2][3] = 0;

    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::setShear (const Shear6<S>& h) IMATH_NOEXCEPT
{
    x[0][0] = 1;
    x[0][1] = h.yx;
    x[0][2] = h.zx;
    x[0][3] = 0;

    x[1][0] = h.xy;
    x[1][1] = 1;
    x[1][2] = h.zy;
    x[1][3] = 0;

    x[2][0] = h.xz;
    x[2][1] = h.yz;
    x[2][2] = 1;
    x[2][3] = 0;

    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::shear (const Vec3<S>& h) IMATH_NOEXCEPT
{
    //
    // In this case, we don't need a temp. copy of the matrix
    // because we never use a value on the RHS after we've
    // changed it on the LHS.
    //

    for (int i = 0; i < 4; i++)
    {
        x[2][i] += h.y * x[0][i] + h.z * x[1][i];
        x[1][i] += h.x * x[0][i];
    }

    return *this;
}

template <class T>
template <class S>
IMATH_HOSTDEVICE IMATH_CONSTEXPR14 inline const Matrix44<T>&
Matrix44<T>::shear (const Shear6<S>& h) IMATH_NOEXCEPT
{
    Matrix44<T> P (*this);

    for (int i = 0; i < 4; i++)
    {
        x[0][i] = P.x[0][i] + h.yx * P.x[1][i] + h.zx * P.x[2][i];
        x[1][i] = h.xy * P.x[0][i] + P.x[1][i] + h.zy * P.x[2][i];
        x[2][i] = h.xz * P.x[0][i] + h.yz * P.x[1][i] + P.x[2][i];
    }

    return *this;
}

//--------------------------------
// Implementation of stream output
//--------------------------------

template <class T>
std::ostream&
operator<< (std::ostream& s, const Matrix22<T>& m)
{
    std::ios_base::fmtflags oldFlags = s.flags();
    int width;

    if (s.flags() & std::ios_base::fixed)
    {
        s.setf (std::ios_base::showpoint);
        width = static_cast<int> (s.precision()) + 5;
    }
    else
    {
        s.setf (std::ios_base::scientific);
        s.setf (std::ios_base::showpoint);
        width = static_cast<int> (s.precision()) + 8;
    }

    s << "(" << std::setw (width) << m[0][0] << " " << std::setw (width) << m[0][1] << "\n"
      <<

        " " << std::setw (width) << m[1][0] << " " << std::setw (width) << m[1][1] << ")\n";

    s.flags (oldFlags);
    return s;
}

template <class T>
std::ostream&
operator<< (std::ostream& s, const Matrix33<T>& m)
{
    std::ios_base::fmtflags oldFlags = s.flags();
    int width;

    if (s.flags() & std::ios_base::fixed)
    {
        s.setf (std::ios_base::showpoint);
        width = static_cast<int> (s.precision()) + 5;
    }
    else
    {
        s.setf (std::ios_base::scientific);
        s.setf (std::ios_base::showpoint);
        width = static_cast<int> (s.precision()) + 8;
    }

    s << "(" << std::setw (width) << m[0][0] << " " << std::setw (width) << m[0][1] << " "
      << std::setw (width) << m[0][2] << "\n"
      <<

        " " << std::setw (width) << m[1][0] << " " << std::setw (width) << m[1][1] << " "
      << std::setw (width) << m[1][2] << "\n"
      <<

        " " << std::setw (width) << m[2][0] << " " << std::setw (width) << m[2][1] << " "
      << std::setw (width) << m[2][2] << ")\n";

    s.flags (oldFlags);
    return s;
}

template <class T>
std::ostream&
operator<< (std::ostream& s, const Matrix44<T>& m)
{
    std::ios_base::fmtflags oldFlags = s.flags();
    int width;

    if (s.flags() & std::ios_base::fixed)
    {
        s.setf (std::ios_base::showpoint);
        width = static_cast<int> (s.precision()) + 5;
    }
    else
    {
        s.setf (std::ios_base::scientific);
        s.setf (std::ios_base::showpoint);
        width = static_cast<int> (s.precision()) + 8;
    }

    s << "(" << std::setw (width) << m[0][0] << " " << std::setw (width) << m[0][1] << " "
      << std::setw (width) << m[0][2] << " " << std::setw (width) << m[0][3] << "\n"
      <<

        " " << std::setw (width) << m[1][0] << " " << std::setw (width) << m[1][1] << " "
      << std::setw (width) << m[1][2] << " " << std::setw (width) << m[1][3] << "\n"
      <<

        " " << std::setw (width) << m[2][0] << " " << std::setw (width) << m[2][1] << " "
      << std::setw (width) << m[2][2] << " " << std::setw (width) << m[2][3] << "\n"
      <<

        " " << std::setw (width) << m[3][0] << " " << std::setw (width) << m[3][1] << " "
      << std::setw (width) << m[3][2] << " " << std::setw (width) << m[3][3] << ")\n";

    s.flags (oldFlags);
    return s;
}

//---------------------------------------------------------------
// Implementation of vector-times-matrix multiplication operators
//---------------------------------------------------------------

template <class S, class T>
IMATH_HOSTDEVICE inline const Vec2<S>&
operator*= (Vec2<S>& v, const Matrix22<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1]);

    v.x = x;
    v.y = y;

    return v;
}

template <class S, class T>
IMATH_HOSTDEVICE inline Vec2<S>
operator* (const Vec2<S>& v, const Matrix22<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1]);

    return Vec2<S> (x, y);
}

template <class S, class T>
IMATH_HOSTDEVICE inline const Vec2<S>&
operator*= (Vec2<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + m.x[2][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + m.x[2][1]);
    S w = S (v.x * m.x[0][2] + v.y * m.x[1][2] + m.x[2][2]);

    v.x = x / w;
    v.y = y / w;

    return v;
}

template <class S, class T>
IMATH_HOSTDEVICE inline Vec2<S>
operator* (const Vec2<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + m.x[2][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + m.x[2][1]);
    S w = S (v.x * m.x[0][2] + v.y * m.x[1][2] + m.x[2][2]);

    return Vec2<S> (x / w, y / w);
}

template <class S, class T>
IMATH_HOSTDEVICE inline const Vec3<S>&
operator*= (Vec3<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1]);
    S z = S (v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2]);

    v.x = x;
    v.y = y;
    v.z = z;

    return v;
}

template <class S, class T>
IMATH_HOSTDEVICE inline Vec3<S>
operator* (const Vec3<S>& v, const Matrix33<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1]);
    S z = S (v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2]);

    return Vec3<S> (x, y, z);
}

template <class S, class T>
IMATH_HOSTDEVICE inline const Vec3<S>&
operator*= (Vec3<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + m.x[3][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + m.x[3][1]);
    S z = S (v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + m.x[3][2]);
    S w = S (v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + m.x[3][3]);

    v.x = x / w;
    v.y = y / w;
    v.z = z / w;

    return v;
}

template <class S, class T>
IMATH_HOSTDEVICE inline Vec3<S>
IMATH_HOSTDEVICE operator* (const Vec3<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + m.x[3][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + m.x[3][1]);
    S z = S (v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + m.x[3][2]);
    S w = S (v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + m.x[3][3]);

    return Vec3<S> (x / w, y / w, z / w);
}

template <class S, class T>
IMATH_HOSTDEVICE inline const Vec4<S>&
IMATH_HOSTDEVICE operator*= (Vec4<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + v.w * m.x[3][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + v.w * m.x[3][1]);
    S z = S (v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + v.w * m.x[3][2]);
    S w = S (v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + v.w * m.x[3][3]);

    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;

    return v;
}

template <class S, class T>
IMATH_HOSTDEVICE inline Vec4<S>
IMATH_HOSTDEVICE operator* (const Vec4<S>& v, const Matrix44<T>& m) IMATH_NOEXCEPT
{
    S x = S (v.x * m.x[0][0] + v.y * m.x[1][0] + v.z * m.x[2][0] + v.w * m.x[3][0]);
    S y = S (v.x * m.x[0][1] + v.y * m.x[1][1] + v.z * m.x[2][1] + v.w * m.x[3][1]);
    S z = S (v.x * m.x[0][2] + v.y * m.x[1][2] + v.z * m.x[2][2] + v.w * m.x[3][2]);
    S w = S (v.x * m.x[0][3] + v.y * m.x[1][3] + v.z * m.x[2][3] + v.w * m.x[3][3]);

    return Vec4<S> (x, y, z, w);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHMATRIX_H
