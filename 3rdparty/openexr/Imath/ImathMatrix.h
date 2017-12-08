///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////



#ifndef INCLUDED_IMATHMATRIX_H
#define INCLUDED_IMATHMATRIX_H

//----------------------------------------------------------------
//
//      2D (3x3) and 3D (4x4) transformation matrix templates.
//
//----------------------------------------------------------------

#include "ImathPlatform.h"
#include "ImathFun.h"
#include "ImathExc.h"
#include "ImathVec.h"
#include "ImathShear.h"
#include "ImathNamespace.h"

#include <cstring>
#include <iostream>
#include <iomanip>
#include <string.h>

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// suppress exception specification warnings
#pragma warning(disable:4290)
#endif


IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

enum Uninitialized {UNINITIALIZED};


template <class T> class Matrix33
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T           x[3][3];

    T *         operator [] (int i);
    const T *   operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    Matrix33 (Uninitialized) {}

    Matrix33 ();
                                // 1 0 0
                                // 0 1 0
                                // 0 0 1

    Matrix33 (T a);
                                // a a a
                                // a a a
                                // a a a

    Matrix33 (const T a[3][3]);
                                // a[0][0] a[0][1] a[0][2]
                                // a[1][0] a[1][1] a[1][2]
                                // a[2][0] a[2][1] a[2][2]

    Matrix33 (T a, T b, T c, T d, T e, T f, T g, T h, T i);

                                // a b c
                                // d e f
                                // g h i


    //--------------------------------
    // Copy constructor and assignment
    //--------------------------------

    Matrix33 (const Matrix33 &v);
    template <class S> explicit Matrix33 (const Matrix33<S> &v);

    const Matrix33 &    operator = (const Matrix33 &v);
    const Matrix33 &    operator = (T a);


    //----------------------
    // Compatibility with Sb
    //----------------------
    
    T *                 getValue ();
    const T *           getValue () const;

    template <class S>
    void                getValue (Matrix33<S> &v) const;
    template <class S>
    Matrix33 &          setValue (const Matrix33<S> &v);

    template <class S>
    Matrix33 &          setTheMatrix (const Matrix33<S> &v);


    //---------
    // Identity
    //---------

    void                makeIdentity();


    //---------
    // Equality
    //---------

    bool                operator == (const Matrix33 &v) const;
    bool                operator != (const Matrix33 &v) const;

    //-----------------------------------------------------------------------
    // Compare two matrices and test if they are "approximately equal":
    //
    // equalWithAbsError (m, e)
    //
    //      Returns true if the coefficients of this and m are the same with
    //      an absolute error of no more than e, i.e., for all i, j
    //
    //      abs (this[i][j] - m[i][j]) <= e
    //
    // equalWithRelError (m, e)
    //
    //      Returns true if the coefficients of this and m are the same with
    //      a relative error of no more than e, i.e., for all i, j
    //
    //      abs (this[i] - v[i][j]) <= e * abs (this[i][j])
    //-----------------------------------------------------------------------

    bool                equalWithAbsError (const Matrix33<T> &v, T e) const;
    bool                equalWithRelError (const Matrix33<T> &v, T e) const;


    //------------------------
    // Component-wise addition
    //------------------------

    const Matrix33 &    operator += (const Matrix33 &v);
    const Matrix33 &    operator += (T a);
    Matrix33            operator + (const Matrix33 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Matrix33 &    operator -= (const Matrix33 &v);
    const Matrix33 &    operator -= (T a);
    Matrix33            operator - (const Matrix33 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Matrix33            operator - () const;
    const Matrix33 &    negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Matrix33 &    operator *= (T a);
    Matrix33            operator * (T a) const;


    //-----------------------------------
    // Matrix-times-matrix multiplication
    //-----------------------------------

    const Matrix33 &    operator *= (const Matrix33 &v);
    Matrix33            operator * (const Matrix33 &v) const;


    //-----------------------------------------------------------------
    // Vector-times-matrix multiplication; see also the "operator *"
    // functions defined below.
    //
    // m.multVecMatrix(src,dst) implements a homogeneous transformation
    // by computing Vec3 (src.x, src.y, 1) * m and dividing by the
    // result's third element.
    //
    // m.multDirMatrix(src,dst) multiplies src by the upper left 2x2
    // submatrix, ignoring the rest of matrix m.
    //-----------------------------------------------------------------

    template <class S>
    void                multVecMatrix(const Vec2<S> &src, Vec2<S> &dst) const;

    template <class S>
    void                multDirMatrix(const Vec2<S> &src, Vec2<S> &dst) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Matrix33 &    operator /= (T a);
    Matrix33            operator / (T a) const;


    //------------------
    // Transposed matrix
    //------------------

    const Matrix33 &    transpose ();
    Matrix33            transposed () const;


    //------------------------------------------------------------
    // Inverse matrix: If singExc is false, inverting a singular
    // matrix produces an identity matrix.  If singExc is true,
    // inverting a singular matrix throws a SingMatrixExc.
    //
    // inverse() and invert() invert matrices using determinants;
    // gjInverse() and gjInvert() use the Gauss-Jordan method.
    //
    // inverse() and invert() are significantly faster than
    // gjInverse() and gjInvert(), but the results may be slightly
    // less accurate.
    // 
    //------------------------------------------------------------

    const Matrix33 &    invert (bool singExc = false)
                        throw (IEX_NAMESPACE::MathExc);

    Matrix33<T>         inverse (bool singExc = false) const
                        throw (IEX_NAMESPACE::MathExc);

    const Matrix33 &    gjInvert (bool singExc = false)
                        throw (IEX_NAMESPACE::MathExc);

    Matrix33<T>         gjInverse (bool singExc = false) const
                        throw (IEX_NAMESPACE::MathExc);


    //------------------------------------------------
    // Calculate the matrix minor of the (r,c) element
    //------------------------------------------------

    T                   minorOf (const int r, const int c) const;

    //---------------------------------------------------
    // Build a minor using the specified rows and columns
    //---------------------------------------------------

    T                   fastMinor (const int r0, const int r1, 
                                   const int c0, const int c1) const;

    //------------
    // Determinant
    //------------

    T                   determinant() const;

    //-----------------------------------------
    // Set matrix to rotation by r (in radians)
    //-----------------------------------------

    template <class S>
    const Matrix33 &    setRotation (S r);


    //-----------------------------
    // Rotate the given matrix by r
    //-----------------------------

    template <class S>
    const Matrix33 &    rotate (S r);


    //--------------------------------------------
    // Set matrix to scale by given uniform factor
    //--------------------------------------------

    const Matrix33 &    setScale (T s);


    //------------------------------------
    // Set matrix to scale by given vector
    //------------------------------------

    template <class S>
    const Matrix33 &    setScale (const Vec2<S> &s);


    //----------------------
    // Scale the matrix by s
    //----------------------

    template <class S>
    const Matrix33 &    scale (const Vec2<S> &s);


    //------------------------------------------
    // Set matrix to translation by given vector
    //------------------------------------------

    template <class S>
    const Matrix33 &    setTranslation (const Vec2<S> &t);


    //-----------------------------
    // Return translation component
    //-----------------------------

    Vec2<T>             translation () const;


    //--------------------------
    // Translate the matrix by t
    //--------------------------

    template <class S>
    const Matrix33 &    translate (const Vec2<S> &t);


    //-----------------------------------------------------------
    // Set matrix to shear x for each y coord. by given factor xy
    //-----------------------------------------------------------

    template <class S>
    const Matrix33 &    setShear (const S &h);


    //-------------------------------------------------------------
    // Set matrix to shear x for each y coord. by given factor h[0]
    // and to shear y for each x coord. by given factor h[1]
    //-------------------------------------------------------------

    template <class S>
    const Matrix33 &    setShear (const Vec2<S> &h);


    //-----------------------------------------------------------
    // Shear the matrix in x for each y coord. by given factor xy
    //-----------------------------------------------------------

    template <class S>
    const Matrix33 &    shear (const S &xy);


    //-----------------------------------------------------------
    // Shear the matrix in x for each y coord. by given factor xy
    // and shear y for each x coord. by given factor yx
    //-----------------------------------------------------------

    template <class S>
    const Matrix33 &    shear (const Vec2<S> &h);


    //--------------------------------------------------------
    // Number of the row and column dimensions, since
    // Matrix33 is a square matrix.
    //--------------------------------------------------------

    static unsigned int	dimensions() {return 3;}


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T            baseTypeMin()           {return limits<T>::min();}
    static T            baseTypeMax()           {return limits<T>::max();}
    static T            baseTypeSmallest()      {return limits<T>::smallest();}
    static T            baseTypeEpsilon()       {return limits<T>::epsilon();}

    typedef T		BaseType;
    typedef Vec3<T>	BaseVecType;

  private:

    template <typename R, typename S>
    struct isSameType
    {
        enum {value = 0};
    };

    template <typename R>
    struct isSameType<R, R>
    {
        enum {value = 1};
    };
};


template <class T> class Matrix44
{
  public:

    //-------------------
    // Access to elements
    //-------------------

    T           x[4][4];

    T *         operator [] (int i);
    const T *   operator [] (int i) const;


    //-------------
    // Constructors
    //-------------

    Matrix44 (Uninitialized) {}

    Matrix44 ();
                                // 1 0 0 0
                                // 0 1 0 0
                                // 0 0 1 0
                                // 0 0 0 1

    Matrix44 (T a);
                                // a a a a
                                // a a a a
                                // a a a a
                                // a a a a

    Matrix44 (const T a[4][4]) ;
                                // a[0][0] a[0][1] a[0][2] a[0][3]
                                // a[1][0] a[1][1] a[1][2] a[1][3]
                                // a[2][0] a[2][1] a[2][2] a[2][3]
                                // a[3][0] a[3][1] a[3][2] a[3][3]

    Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h,
              T i, T j, T k, T l, T m, T n, T o, T p);

                                // a b c d
                                // e f g h
                                // i j k l
                                // m n o p

    Matrix44 (Matrix33<T> r, Vec3<T> t);
                                // r r r 0
                                // r r r 0
                                // r r r 0
                                // t t t 1


    //--------------------------------
    // Copy constructor and assignment
    //--------------------------------

    Matrix44 (const Matrix44 &v);
    template <class S> explicit Matrix44 (const Matrix44<S> &v);

    const Matrix44 &    operator = (const Matrix44 &v);
    const Matrix44 &    operator = (T a);


    //----------------------
    // Compatibility with Sb
    //----------------------
    
    T *                 getValue ();
    const T *           getValue () const;

    template <class S>
    void                getValue (Matrix44<S> &v) const;
    template <class S>
    Matrix44 &          setValue (const Matrix44<S> &v);

    template <class S>
    Matrix44 &          setTheMatrix (const Matrix44<S> &v);

    //---------
    // Identity
    //---------

    void                makeIdentity();


    //---------
    // Equality
    //---------

    bool                operator == (const Matrix44 &v) const;
    bool                operator != (const Matrix44 &v) const;

    //-----------------------------------------------------------------------
    // Compare two matrices and test if they are "approximately equal":
    //
    // equalWithAbsError (m, e)
    //
    //      Returns true if the coefficients of this and m are the same with
    //      an absolute error of no more than e, i.e., for all i, j
    //
    //      abs (this[i][j] - m[i][j]) <= e
    //
    // equalWithRelError (m, e)
    //
    //      Returns true if the coefficients of this and m are the same with
    //      a relative error of no more than e, i.e., for all i, j
    //
    //      abs (this[i] - v[i][j]) <= e * abs (this[i][j])
    //-----------------------------------------------------------------------

    bool                equalWithAbsError (const Matrix44<T> &v, T e) const;
    bool                equalWithRelError (const Matrix44<T> &v, T e) const;


    //------------------------
    // Component-wise addition
    //------------------------

    const Matrix44 &    operator += (const Matrix44 &v);
    const Matrix44 &    operator += (T a);
    Matrix44            operator + (const Matrix44 &v) const;


    //---------------------------
    // Component-wise subtraction
    //---------------------------

    const Matrix44 &    operator -= (const Matrix44 &v);
    const Matrix44 &    operator -= (T a);
    Matrix44            operator - (const Matrix44 &v) const;


    //------------------------------------
    // Component-wise multiplication by -1
    //------------------------------------

    Matrix44            operator - () const;
    const Matrix44 &    negate ();


    //------------------------------
    // Component-wise multiplication
    //------------------------------

    const Matrix44 &    operator *= (T a);
    Matrix44            operator * (T a) const;


    //-----------------------------------
    // Matrix-times-matrix multiplication
    //-----------------------------------

    const Matrix44 &    operator *= (const Matrix44 &v);
    Matrix44            operator * (const Matrix44 &v) const;

    static void         multiply (const Matrix44 &a,    // assumes that
                                  const Matrix44 &b,    // &a != &c and
                                  Matrix44 &c);         // &b != &c.


    //-----------------------------------------------------------------
    // Vector-times-matrix multiplication; see also the "operator *"
    // functions defined below.
    //
    // m.multVecMatrix(src,dst) implements a homogeneous transformation
    // by computing Vec4 (src.x, src.y, src.z, 1) * m and dividing by
    // the result's third element.
    //
    // m.multDirMatrix(src,dst) multiplies src by the upper left 3x3
    // submatrix, ignoring the rest of matrix m.
    //-----------------------------------------------------------------

    template <class S>
    void                multVecMatrix(const Vec3<S> &src, Vec3<S> &dst) const;

    template <class S>
    void                multDirMatrix(const Vec3<S> &src, Vec3<S> &dst) const;


    //------------------------
    // Component-wise division
    //------------------------

    const Matrix44 &    operator /= (T a);
    Matrix44            operator / (T a) const;


    //------------------
    // Transposed matrix
    //------------------

    const Matrix44 &    transpose ();
    Matrix44            transposed () const;


    //------------------------------------------------------------
    // Inverse matrix: If singExc is false, inverting a singular
    // matrix produces an identity matrix.  If singExc is true,
    // inverting a singular matrix throws a SingMatrixExc.
    //
    // inverse() and invert() invert matrices using determinants;
    // gjInverse() and gjInvert() use the Gauss-Jordan method.
    //
    // inverse() and invert() are significantly faster than
    // gjInverse() and gjInvert(), but the results may be slightly
    // less accurate.
    // 
    //------------------------------------------------------------

    const Matrix44 &    invert (bool singExc = false)
                        throw (IEX_NAMESPACE::MathExc);

    Matrix44<T>         inverse (bool singExc = false) const
                        throw (IEX_NAMESPACE::MathExc);

    const Matrix44 &    gjInvert (bool singExc = false)
                        throw (IEX_NAMESPACE::MathExc);

    Matrix44<T>         gjInverse (bool singExc = false) const
                        throw (IEX_NAMESPACE::MathExc);


    //------------------------------------------------
    // Calculate the matrix minor of the (r,c) element
    //------------------------------------------------

    T                   minorOf (const int r, const int c) const;

    //---------------------------------------------------
    // Build a minor using the specified rows and columns
    //---------------------------------------------------

    T                   fastMinor (const int r0, const int r1, const int r2,
                                   const int c0, const int c1, const int c2) const;

    //------------
    // Determinant
    //------------

    T                   determinant() const;

    //--------------------------------------------------------
    // Set matrix to rotation by XYZ euler angles (in radians)
    //--------------------------------------------------------

    template <class S>
    const Matrix44 &    setEulerAngles (const Vec3<S>& r);


    //--------------------------------------------------------
    // Set matrix to rotation around given axis by given angle
    //--------------------------------------------------------

    template <class S>
    const Matrix44 &    setAxisAngle (const Vec3<S>& ax, S ang);


    //-------------------------------------------
    // Rotate the matrix by XYZ euler angles in r
    //-------------------------------------------

    template <class S>
    const Matrix44 &    rotate (const Vec3<S> &r);


    //--------------------------------------------
    // Set matrix to scale by given uniform factor
    //--------------------------------------------

    const Matrix44 &    setScale (T s);


    //------------------------------------
    // Set matrix to scale by given vector
    //------------------------------------

    template <class S>
    const Matrix44 &    setScale (const Vec3<S> &s);


    //----------------------
    // Scale the matrix by s
    //----------------------

    template <class S>
    const Matrix44 &    scale (const Vec3<S> &s);


    //------------------------------------------
    // Set matrix to translation by given vector
    //------------------------------------------

    template <class S>
    const Matrix44 &    setTranslation (const Vec3<S> &t);


    //-----------------------------
    // Return translation component
    //-----------------------------

    const Vec3<T>       translation () const;


    //--------------------------
    // Translate the matrix by t
    //--------------------------

    template <class S>
    const Matrix44 &    translate (const Vec3<S> &t);


    //-------------------------------------------------------------
    // Set matrix to shear by given vector h.  The resulting matrix
    //    will shear x for each y coord. by a factor of h[0] ;
    //    will shear x for each z coord. by a factor of h[1] ;
    //    will shear y for each z coord. by a factor of h[2] .
    //-------------------------------------------------------------

    template <class S>
    const Matrix44 &    setShear (const Vec3<S> &h);


    //------------------------------------------------------------
    // Set matrix to shear by given factors.  The resulting matrix
    //    will shear x for each y coord. by a factor of h.xy ;
    //    will shear x for each z coord. by a factor of h.xz ;
    //    will shear y for each z coord. by a factor of h.yz ; 
    //    will shear y for each x coord. by a factor of h.yx ;
    //    will shear z for each x coord. by a factor of h.zx ;
    //    will shear z for each y coord. by a factor of h.zy .
    //------------------------------------------------------------

    template <class S>
    const Matrix44 &    setShear (const Shear6<S> &h);


    //--------------------------------------------------------
    // Shear the matrix by given vector.  The composed matrix 
    // will be <shear> * <this>, where the shear matrix ...
    //    will shear x for each y coord. by a factor of h[0] ;
    //    will shear x for each z coord. by a factor of h[1] ;
    //    will shear y for each z coord. by a factor of h[2] .
    //--------------------------------------------------------

    template <class S>
    const Matrix44 &    shear (const Vec3<S> &h);

    //--------------------------------------------------------
    // Number of the row and column dimensions, since
    // Matrix44 is a square matrix.
    //--------------------------------------------------------

    static unsigned int	dimensions() {return 4;}


    //------------------------------------------------------------
    // Shear the matrix by the given factors.  The composed matrix 
    // will be <shear> * <this>, where the shear matrix ...
    //    will shear x for each y coord. by a factor of h.xy ;
    //    will shear x for each z coord. by a factor of h.xz ;
    //    will shear y for each z coord. by a factor of h.yz ;
    //    will shear y for each x coord. by a factor of h.yx ;
    //    will shear z for each x coord. by a factor of h.zx ;
    //    will shear z for each y coord. by a factor of h.zy .
    //------------------------------------------------------------

    template <class S>
    const Matrix44 &    shear (const Shear6<S> &h);


    //-------------------------------------------------
    // Limitations of type T (see also class limits<T>)
    //-------------------------------------------------

    static T            baseTypeMin()           {return limits<T>::min();}
    static T            baseTypeMax()           {return limits<T>::max();}
    static T            baseTypeSmallest()      {return limits<T>::smallest();}
    static T            baseTypeEpsilon()       {return limits<T>::epsilon();}

    typedef T		BaseType;
    typedef Vec4<T>	BaseVecType;

  private:

    template <typename R, typename S>
    struct isSameType
    {
        enum {value = 0};
    };

    template <typename R>
    struct isSameType<R, R>
    {
        enum {value = 1};
    };
};


//--------------
// Stream output
//--------------

template <class T>
std::ostream &  operator << (std::ostream & s, const Matrix33<T> &m); 

template <class T>
std::ostream &  operator << (std::ostream & s, const Matrix44<T> &m); 


//---------------------------------------------
// Vector-times-matrix multiplication operators
//---------------------------------------------

template <class S, class T>
const Vec2<S> &            operator *= (Vec2<S> &v, const Matrix33<T> &m);

template <class S, class T>
Vec2<S>                    operator * (const Vec2<S> &v, const Matrix33<T> &m);

template <class S, class T>
const Vec3<S> &            operator *= (Vec3<S> &v, const Matrix33<T> &m);

template <class S, class T>
Vec3<S>                    operator * (const Vec3<S> &v, const Matrix33<T> &m);

template <class S, class T>
const Vec3<S> &            operator *= (Vec3<S> &v, const Matrix44<T> &m);

template <class S, class T>
Vec3<S>                    operator * (const Vec3<S> &v, const Matrix44<T> &m);

template <class S, class T>
const Vec4<S> &            operator *= (Vec4<S> &v, const Matrix44<T> &m);

template <class S, class T>
Vec4<S>                    operator * (const Vec4<S> &v, const Matrix44<T> &m);

//-------------------------
// Typedefs for convenience
//-------------------------

typedef Matrix33 <float>  M33f;
typedef Matrix33 <double> M33d;
typedef Matrix44 <float>  M44f;
typedef Matrix44 <double> M44d;


//---------------------------
// Implementation of Matrix33
//---------------------------

template <class T>
inline T *
Matrix33<T>::operator [] (int i)
{
    return x[i];
}

template <class T>
inline const T *
Matrix33<T>::operator [] (int i) const
{
    return x[i];
}

template <class T>
inline
Matrix33<T>::Matrix33 ()
{
    memset (x, 0, sizeof (x));
    x[0][0] = 1;
    x[1][1] = 1;
    x[2][2] = 1;
}

template <class T>
inline
Matrix33<T>::Matrix33 (T a)
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
inline
Matrix33<T>::Matrix33 (const T a[3][3]) 
{
    memcpy (x, a, sizeof (x));
}

template <class T>
inline
Matrix33<T>::Matrix33 (T a, T b, T c, T d, T e, T f, T g, T h, T i)
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
inline
Matrix33<T>::Matrix33 (const Matrix33 &v)
{
    memcpy (x, v.x, sizeof (x));
}

template <class T>
template <class S>
inline
Matrix33<T>::Matrix33 (const Matrix33<S> &v)
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
inline const Matrix33<T> &
Matrix33<T>::operator = (const Matrix33 &v)
{
    memcpy (x, v.x, sizeof (x));
    return *this;
}

template <class T>
inline const Matrix33<T> &
Matrix33<T>::operator = (T a)
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
inline T *
Matrix33<T>::getValue ()
{
    return (T *) &x[0][0];
}

template <class T>
inline const T *
Matrix33<T>::getValue () const
{
    return (const T *) &x[0][0];
}

template <class T>
template <class S>
inline void
Matrix33<T>::getValue (Matrix33<S> &v) const
{
    if (isSameType<S,T>::value)
    {
        memcpy (v.x, x, sizeof (x));
    }
    else
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
}

template <class T>
template <class S>
inline Matrix33<T> &
Matrix33<T>::setValue (const Matrix33<S> &v)
{
    if (isSameType<S,T>::value)
    {
        memcpy (x, v.x, sizeof (x));
    }
    else
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
    }

    return *this;
}

template <class T>
template <class S>
inline Matrix33<T> &
Matrix33<T>::setTheMatrix (const Matrix33<S> &v)
{
    if (isSameType<S,T>::value)
    {
        memcpy (x, v.x, sizeof (x));
    }
    else
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
    }

    return *this;
}

template <class T>
inline void
Matrix33<T>::makeIdentity()
{
    memset (x, 0, sizeof (x));
    x[0][0] = 1;
    x[1][1] = 1;
    x[2][2] = 1;
}

template <class T>
bool
Matrix33<T>::operator == (const Matrix33 &v) const
{
    return x[0][0] == v.x[0][0] &&
           x[0][1] == v.x[0][1] &&
           x[0][2] == v.x[0][2] &&
           x[1][0] == v.x[1][0] &&
           x[1][1] == v.x[1][1] &&
           x[1][2] == v.x[1][2] &&
           x[2][0] == v.x[2][0] &&
           x[2][1] == v.x[2][1] &&
           x[2][2] == v.x[2][2];
}

template <class T>
bool
Matrix33<T>::operator != (const Matrix33 &v) const
{
    return x[0][0] != v.x[0][0] ||
           x[0][1] != v.x[0][1] ||
           x[0][2] != v.x[0][2] ||
           x[1][0] != v.x[1][0] ||
           x[1][1] != v.x[1][1] ||
           x[1][2] != v.x[1][2] ||
           x[2][0] != v.x[2][0] ||
           x[2][1] != v.x[2][1] ||
           x[2][2] != v.x[2][2];
}

template <class T>
bool
Matrix33<T>::equalWithAbsError (const Matrix33<T> &m, T e) const
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T>
bool
Matrix33<T>::equalWithRelError (const Matrix33<T> &m, T e) const
{
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T>
const Matrix33<T> &
Matrix33<T>::operator += (const Matrix33<T> &v)
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
const Matrix33<T> &
Matrix33<T>::operator += (T a)
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
Matrix33<T>
Matrix33<T>::operator + (const Matrix33<T> &v) const
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
const Matrix33<T> &
Matrix33<T>::operator -= (const Matrix33<T> &v)
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
const Matrix33<T> &
Matrix33<T>::operator -= (T a)
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
Matrix33<T>
Matrix33<T>::operator - (const Matrix33<T> &v) const
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
Matrix33<T>
Matrix33<T>::operator - () const
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
const Matrix33<T> &
Matrix33<T>::negate ()
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
const Matrix33<T> &
Matrix33<T>::operator *= (T a)
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
Matrix33<T>
Matrix33<T>::operator * (T a) const
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

template <class T>
inline Matrix33<T>
operator * (T a, const Matrix33<T> &v)
{
    return v * a;
}

template <class T>
const Matrix33<T> &
Matrix33<T>::operator *= (const Matrix33<T> &v)
{
    Matrix33 tmp (T (0));

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                tmp.x[i][j] += x[i][k] * v.x[k][j];

    *this = tmp;
    return *this;
}

template <class T>
Matrix33<T>
Matrix33<T>::operator * (const Matrix33<T> &v) const
{
    Matrix33 tmp (T (0));

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
                tmp.x[i][j] += x[i][k] * v.x[k][j];

    return tmp;
}

template <class T>
template <class S>
void
Matrix33<T>::multVecMatrix(const Vec2<S> &src, Vec2<S> &dst) const
{
    S a, b, w;

    a = src[0] * x[0][0] + src[1] * x[1][0] + x[2][0];
    b = src[0] * x[0][1] + src[1] * x[1][1] + x[2][1];
    w = src[0] * x[0][2] + src[1] * x[1][2] + x[2][2];

    dst.x = a / w;
    dst.y = b / w;
}

template <class T>
template <class S>
void
Matrix33<T>::multDirMatrix(const Vec2<S> &src, Vec2<S> &dst) const
{
    S a, b;

    a = src[0] * x[0][0] + src[1] * x[1][0];
    b = src[0] * x[0][1] + src[1] * x[1][1];

    dst.x = a;
    dst.y = b;
}

template <class T>
const Matrix33<T> &
Matrix33<T>::operator /= (T a)
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
Matrix33<T>
Matrix33<T>::operator / (T a) const
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
const Matrix33<T> &
Matrix33<T>::transpose ()
{
    Matrix33 tmp (x[0][0],
                  x[1][0],
                  x[2][0],
                  x[0][1],
                  x[1][1],
                  x[2][1],
                  x[0][2],
                  x[1][2],
                  x[2][2]);
    *this = tmp;
    return *this;
}

template <class T>
Matrix33<T>
Matrix33<T>::transposed () const
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
const Matrix33<T> &
Matrix33<T>::gjInvert (bool singExc) throw (IEX_NAMESPACE::MathExc)
{
    *this = gjInverse (singExc);
    return *this;
}

template <class T>
Matrix33<T>
Matrix33<T>::gjInverse (bool singExc) const throw (IEX_NAMESPACE::MathExc)
{
    int i, j, k;
    Matrix33 s;
    Matrix33 t (*this);

    // Forward elimination

    for (i = 0; i < 2 ; i++)
    {
        int pivot = i;

        T pivotsize = t[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (j = i + 1; j < 3; j++)
        {
            T tmp = t[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            if (singExc)
                throw ::IMATH_INTERNAL_NAMESPACE::SingMatrixExc ("Cannot invert singular matrix.");

            return Matrix33();
        }

        if (pivot != i)
        {
            for (j = 0; j < 3; j++)
            {
                T tmp;

                tmp = t[i][j];
                t[i][j] = t[pivot][j];
                t[pivot][j] = tmp;

                tmp = s[i][j];
                s[i][j] = s[pivot][j];
                s[pivot][j] = tmp;
            }
        }

        for (j = i + 1; j < 3; j++)
        {
            T f = t[j][i] / t[i][i];

            for (k = 0; k < 3; k++)
            {
                t[j][k] -= f * t[i][k];
                s[j][k] -= f * s[i][k];
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
                throw ::IMATH_INTERNAL_NAMESPACE::SingMatrixExc ("Cannot invert singular matrix.");

            return Matrix33();
        }

        for (j = 0; j < 3; j++)
        {
            t[i][j] /= f;
            s[i][j] /= f;
        }

        for (j = 0; j < i; j++)
        {
            f = t[j][i];

            for (k = 0; k < 3; k++)
            {
                t[j][k] -= f * t[i][k];
                s[j][k] -= f * s[i][k];
            }
        }
    }

    return s;
}

template <class T>
const Matrix33<T> &
Matrix33<T>::invert (bool singExc) throw (IEX_NAMESPACE::MathExc)
{
    *this = inverse (singExc);
    return *this;
}

template <class T>
Matrix33<T>
Matrix33<T>::inverse (bool singExc) const throw (IEX_NAMESPACE::MathExc)
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
                    s[i][j] /= r;
                }
            }
        }
        else
        {
            T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / limits<T>::smallest();

            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    if (mr > IMATH_INTERNAL_NAMESPACE::abs (s[i][j]))
                    {
                        s[i][j] /= r;
                    }
                    else
                    {
                        if (singExc)
                            throw SingMatrixExc ("Cannot invert "
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
        Matrix33 s ( x[1][1],
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
                    s[i][j] /= r;
                }
            }
        }
        else
        {
            T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / limits<T>::smallest();

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
                            throw SingMatrixExc ("Cannot invert "
                                                 "singular matrix.");
                        return Matrix33();
                    }
                }
            }
        }

        s[2][0] = -x[2][0] * s[0][0] - x[2][1] * s[1][0];
        s[2][1] = -x[2][0] * s[0][1] - x[2][1] * s[1][1];

        return s;
    }
}

template <class T>
inline T
Matrix33<T>::minorOf (const int r, const int c) const
{
    int r0 = 0 + (r < 1 ? 1 : 0);
    int r1 = 1 + (r < 2 ? 1 : 0);
    int c0 = 0 + (c < 1 ? 1 : 0);
    int c1 = 1 + (c < 2 ? 1 : 0);

    return x[r0][c0]*x[r1][c1] - x[r1][c0]*x[r0][c1];
}

template <class T>
inline T
Matrix33<T>::fastMinor( const int r0, const int r1,
                        const int c0, const int c1) const
{
    return x[r0][c0]*x[r1][c1] - x[r0][c1]*x[r1][c0];
}

template <class T>
inline T
Matrix33<T>::determinant () const
{
    return x[0][0]*(x[1][1]*x[2][2] - x[1][2]*x[2][1]) +
           x[0][1]*(x[1][2]*x[2][0] - x[1][0]*x[2][2]) +
           x[0][2]*(x[1][0]*x[2][1] - x[1][1]*x[2][0]);
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::setRotation (S r)
{
    S cos_r, sin_r;

    cos_r = Math<T>::cos (r);
    sin_r = Math<T>::sin (r);

    x[0][0] =  cos_r;
    x[0][1] =  sin_r;
    x[0][2] =  0;

    x[1][0] =  -sin_r;
    x[1][1] =  cos_r;
    x[1][2] =  0;

    x[2][0] =  0;
    x[2][1] =  0;
    x[2][2] =  1;

    return *this;
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::rotate (S r)
{
    *this *= Matrix33<T>().setRotation (r);
    return *this;
}

template <class T>
const Matrix33<T> &
Matrix33<T>::setScale (T s)
{
    memset (x, 0, sizeof (x));
    x[0][0] = s;
    x[1][1] = s;
    x[2][2] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::setScale (const Vec2<S> &s)
{
    memset (x, 0, sizeof (x));
    x[0][0] = s[0];
    x[1][1] = s[1];
    x[2][2] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::scale (const Vec2<S> &s)
{
    x[0][0] *= s[0];
    x[0][1] *= s[0];
    x[0][2] *= s[0];

    x[1][0] *= s[1];
    x[1][1] *= s[1];
    x[1][2] *= s[1];

    return *this;
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::setTranslation (const Vec2<S> &t)
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;

    x[1][0] = 0;
    x[1][1] = 1;
    x[1][2] = 0;

    x[2][0] = t[0];
    x[2][1] = t[1];
    x[2][2] = 1;

    return *this;
}

template <class T>
inline Vec2<T> 
Matrix33<T>::translation () const
{
    return Vec2<T> (x[2][0], x[2][1]);
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::translate (const Vec2<S> &t)
{
    x[2][0] += t[0] * x[0][0] + t[1] * x[1][0];
    x[2][1] += t[0] * x[0][1] + t[1] * x[1][1];
    x[2][2] += t[0] * x[0][2] + t[1] * x[1][2];

    return *this;
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::setShear (const S &xy)
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
const Matrix33<T> &
Matrix33<T>::setShear (const Vec2<S> &h)
{
    x[0][0] = 1;
    x[0][1] = h[1];
    x[0][2] = 0;

    x[1][0] = h[0];
    x[1][1] = 1;
    x[1][2] = 0;

    x[2][0] = 0;
    x[2][1] = 0;
    x[2][2] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix33<T> &
Matrix33<T>::shear (const S &xy)
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
const Matrix33<T> &
Matrix33<T>::shear (const Vec2<S> &h)
{
    Matrix33<T> P (*this);
    
    x[0][0] = P[0][0] + h[1] * P[1][0];
    x[0][1] = P[0][1] + h[1] * P[1][1];
    x[0][2] = P[0][2] + h[1] * P[1][2];
    
    x[1][0] = P[1][0] + h[0] * P[0][0];
    x[1][1] = P[1][1] + h[0] * P[0][1];
    x[1][2] = P[1][2] + h[0] * P[0][2];

    return *this;
}


//---------------------------
// Implementation of Matrix44
//---------------------------

template <class T>
inline T *
Matrix44<T>::operator [] (int i)
{
    return x[i];
}

template <class T>
inline const T *
Matrix44<T>::operator [] (int i) const
{
    return x[i];
}

template <class T>
inline
Matrix44<T>::Matrix44 ()
{
    memset (x, 0, sizeof (x));
    x[0][0] = 1;
    x[1][1] = 1;
    x[2][2] = 1;
    x[3][3] = 1;
}

template <class T>
inline
Matrix44<T>::Matrix44 (T a)
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
inline
Matrix44<T>::Matrix44 (const T a[4][4]) 
{
    memcpy (x, a, sizeof (x));
}

template <class T>
inline
Matrix44<T>::Matrix44 (T a, T b, T c, T d, T e, T f, T g, T h,
                       T i, T j, T k, T l, T m, T n, T o, T p)
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


template <class T>
inline
Matrix44<T>::Matrix44 (Matrix33<T> r, Vec3<T> t)
{
    x[0][0] = r[0][0];
    x[0][1] = r[0][1];
    x[0][2] = r[0][2];
    x[0][3] = 0;
    x[1][0] = r[1][0];
    x[1][1] = r[1][1];
    x[1][2] = r[1][2];
    x[1][3] = 0;
    x[2][0] = r[2][0];
    x[2][1] = r[2][1];
    x[2][2] = r[2][2];
    x[2][3] = 0;
    x[3][0] = t[0];
    x[3][1] = t[1];
    x[3][2] = t[2];
    x[3][3] = 1;
}

template <class T>
inline
Matrix44<T>::Matrix44 (const Matrix44 &v)
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
inline
Matrix44<T>::Matrix44 (const Matrix44<S> &v)
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
inline const Matrix44<T> &
Matrix44<T>::operator = (const Matrix44 &v)
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
inline const Matrix44<T> &
Matrix44<T>::operator = (T a)
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
inline T *
Matrix44<T>::getValue ()
{
    return (T *) &x[0][0];
}

template <class T>
inline const T *
Matrix44<T>::getValue () const
{
    return (const T *) &x[0][0];
}

template <class T>
template <class S>
inline void
Matrix44<T>::getValue (Matrix44<S> &v) const
{
    if (isSameType<S,T>::value)
    {
        memcpy (v.x, x, sizeof (x));
    }
    else
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
}

template <class T>
template <class S>
inline Matrix44<T> &
Matrix44<T>::setValue (const Matrix44<S> &v)
{
    if (isSameType<S,T>::value)
    {
        memcpy (x, v.x, sizeof (x));
    }
    else
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

    return *this;
}

template <class T>
template <class S>
inline Matrix44<T> &
Matrix44<T>::setTheMatrix (const Matrix44<S> &v)
{
    if (isSameType<S,T>::value)
    {
        memcpy (x, v.x, sizeof (x));
    }
    else
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

    return *this;
}

template <class T>
inline void
Matrix44<T>::makeIdentity()
{
    memset (x, 0, sizeof (x));
    x[0][0] = 1;
    x[1][1] = 1;
    x[2][2] = 1;
    x[3][3] = 1;
}

template <class T>
bool
Matrix44<T>::operator == (const Matrix44 &v) const
{
    return x[0][0] == v.x[0][0] &&
           x[0][1] == v.x[0][1] &&
           x[0][2] == v.x[0][2] &&
           x[0][3] == v.x[0][3] &&
           x[1][0] == v.x[1][0] &&
           x[1][1] == v.x[1][1] &&
           x[1][2] == v.x[1][2] &&
           x[1][3] == v.x[1][3] &&
           x[2][0] == v.x[2][0] &&
           x[2][1] == v.x[2][1] &&
           x[2][2] == v.x[2][2] &&
           x[2][3] == v.x[2][3] &&
           x[3][0] == v.x[3][0] &&
           x[3][1] == v.x[3][1] &&
           x[3][2] == v.x[3][2] &&
           x[3][3] == v.x[3][3];
}

template <class T>
bool
Matrix44<T>::operator != (const Matrix44 &v) const
{
    return x[0][0] != v.x[0][0] ||
           x[0][1] != v.x[0][1] ||
           x[0][2] != v.x[0][2] ||
           x[0][3] != v.x[0][3] ||
           x[1][0] != v.x[1][0] ||
           x[1][1] != v.x[1][1] ||
           x[1][2] != v.x[1][2] ||
           x[1][3] != v.x[1][3] ||
           x[2][0] != v.x[2][0] ||
           x[2][1] != v.x[2][1] ||
           x[2][2] != v.x[2][2] ||
           x[2][3] != v.x[2][3] ||
           x[3][0] != v.x[3][0] ||
           x[3][1] != v.x[3][1] ||
           x[3][2] != v.x[3][2] ||
           x[3][3] != v.x[3][3];
}

template <class T>
bool
Matrix44<T>::equalWithAbsError (const Matrix44<T> &m, T e) const
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithAbsError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T>
bool
Matrix44<T>::equalWithRelError (const Matrix44<T> &m, T e) const
{
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            if (!IMATH_INTERNAL_NAMESPACE::equalWithRelError ((*this)[i][j], m[i][j], e))
                return false;

    return true;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator += (const Matrix44<T> &v)
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
const Matrix44<T> &
Matrix44<T>::operator += (T a)
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
Matrix44<T>
Matrix44<T>::operator + (const Matrix44<T> &v) const
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
const Matrix44<T> &
Matrix44<T>::operator -= (const Matrix44<T> &v)
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
const Matrix44<T> &
Matrix44<T>::operator -= (T a)
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
Matrix44<T>
Matrix44<T>::operator - (const Matrix44<T> &v) const
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
Matrix44<T>
Matrix44<T>::operator - () const
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
const Matrix44<T> &
Matrix44<T>::negate ()
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
const Matrix44<T> &
Matrix44<T>::operator *= (T a)
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
Matrix44<T>
Matrix44<T>::operator * (T a) const
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

template <class T>
inline Matrix44<T>
operator * (T a, const Matrix44<T> &v)
{
    return v * a;
}

template <class T>
inline const Matrix44<T> &
Matrix44<T>::operator *= (const Matrix44<T> &v)
{
    Matrix44 tmp (T (0));

    multiply (*this, v, tmp);
    *this = tmp;
    return *this;
}

template <class T>
inline Matrix44<T>
Matrix44<T>::operator * (const Matrix44<T> &v) const
{
    Matrix44 tmp (T (0));

    multiply (*this, v, tmp);
    return tmp;
}

template <class T>
void
Matrix44<T>::multiply (const Matrix44<T> &a,
                       const Matrix44<T> &b,
                       Matrix44<T> &c)
{
    register const T * IMATH_RESTRICT ap = &a.x[0][0];
    register const T * IMATH_RESTRICT bp = &b.x[0][0];
    register       T * IMATH_RESTRICT cp = &c.x[0][0];

    register T a0, a1, a2, a3;

    a0 = ap[0];
    a1 = ap[1];
    a2 = ap[2];
    a3 = ap[3];

    cp[0]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[1]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[2]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[3]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[4];
    a1 = ap[5];
    a2 = ap[6];
    a3 = ap[7];

    cp[4]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[5]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[6]  = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[7]  = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[8];
    a1 = ap[9];
    a2 = ap[10];
    a3 = ap[11];

    cp[8]  = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[9]  = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[10] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[11] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];

    a0 = ap[12];
    a1 = ap[13];
    a2 = ap[14];
    a3 = ap[15];

    cp[12] = a0 * bp[0]  + a1 * bp[4]  + a2 * bp[8]  + a3 * bp[12];
    cp[13] = a0 * bp[1]  + a1 * bp[5]  + a2 * bp[9]  + a3 * bp[13];
    cp[14] = a0 * bp[2]  + a1 * bp[6]  + a2 * bp[10] + a3 * bp[14];
    cp[15] = a0 * bp[3]  + a1 * bp[7]  + a2 * bp[11] + a3 * bp[15];
}

template <class T> template <class S>
void
Matrix44<T>::multVecMatrix(const Vec3<S> &src, Vec3<S> &dst) const
{
    S a, b, c, w;

    a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0] + x[3][0];
    b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1] + x[3][1];
    c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2] + x[3][2];
    w = src[0] * x[0][3] + src[1] * x[1][3] + src[2] * x[2][3] + x[3][3];

    dst.x = a / w;
    dst.y = b / w;
    dst.z = c / w;
}

template <class T> template <class S>
void
Matrix44<T>::multDirMatrix(const Vec3<S> &src, Vec3<S> &dst) const
{
    S a, b, c;

    a = src[0] * x[0][0] + src[1] * x[1][0] + src[2] * x[2][0];
    b = src[0] * x[0][1] + src[1] * x[1][1] + src[2] * x[2][1];
    c = src[0] * x[0][2] + src[1] * x[1][2] + src[2] * x[2][2];

    dst.x = a;
    dst.y = b;
    dst.z = c;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::operator /= (T a)
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
Matrix44<T>
Matrix44<T>::operator / (T a) const
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
const Matrix44<T> &
Matrix44<T>::transpose ()
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
Matrix44<T>
Matrix44<T>::transposed () const
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
const Matrix44<T> &
Matrix44<T>::gjInvert (bool singExc) throw (IEX_NAMESPACE::MathExc)
{
    *this = gjInverse (singExc);
    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::gjInverse (bool singExc) const throw (IEX_NAMESPACE::MathExc)
{
    int i, j, k;
    Matrix44 s;
    Matrix44 t (*this);

    // Forward elimination

    for (i = 0; i < 3 ; i++)
    {
        int pivot = i;

        T pivotsize = t[i][i];

        if (pivotsize < 0)
            pivotsize = -pivotsize;

        for (j = i + 1; j < 4; j++)
        {
            T tmp = t[j][i];

            if (tmp < 0)
                tmp = -tmp;

            if (tmp > pivotsize)
            {
                pivot = j;
                pivotsize = tmp;
            }
        }

        if (pivotsize == 0)
        {
            if (singExc)
                throw ::IMATH_INTERNAL_NAMESPACE::SingMatrixExc ("Cannot invert singular matrix.");

            return Matrix44();
        }

        if (pivot != i)
        {
            for (j = 0; j < 4; j++)
            {
                T tmp;

                tmp = t[i][j];
                t[i][j] = t[pivot][j];
                t[pivot][j] = tmp;

                tmp = s[i][j];
                s[i][j] = s[pivot][j];
                s[pivot][j] = tmp;
            }
        }

        for (j = i + 1; j < 4; j++)
        {
            T f = t[j][i] / t[i][i];

            for (k = 0; k < 4; k++)
            {
                t[j][k] -= f * t[i][k];
                s[j][k] -= f * s[i][k];
            }
        }
    }

    // Backward substitution

    for (i = 3; i >= 0; --i)
    {
        T f;

        if ((f = t[i][i]) == 0)
        {
            if (singExc)
                throw ::IMATH_INTERNAL_NAMESPACE::SingMatrixExc ("Cannot invert singular matrix.");

            return Matrix44();
        }

        for (j = 0; j < 4; j++)
        {
            t[i][j] /= f;
            s[i][j] /= f;
        }

        for (j = 0; j < i; j++)
        {
            f = t[j][i];

            for (k = 0; k < 4; k++)
            {
                t[j][k] -= f * t[i][k];
                s[j][k] -= f * s[i][k];
            }
        }
    }

    return s;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::invert (bool singExc) throw (IEX_NAMESPACE::MathExc)
{
    *this = inverse (singExc);
    return *this;
}

template <class T>
Matrix44<T>
Matrix44<T>::inverse (bool singExc) const throw (IEX_NAMESPACE::MathExc)
{
    if (x[0][3] != 0 || x[1][3] != 0 || x[2][3] != 0 || x[3][3] != 1)
        return gjInverse(singExc);

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

    T r = x[0][0] * s[0][0] + x[0][1] * s[1][0] + x[0][2] * s[2][0];

    if (IMATH_INTERNAL_NAMESPACE::abs (r) >= 1)
    {
        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                s[i][j] /= r;
            }
        }
    }
    else
    {
        T mr = IMATH_INTERNAL_NAMESPACE::abs (r) / limits<T>::smallest();

        for (int i = 0; i < 3; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (mr > IMATH_INTERNAL_NAMESPACE::abs (s[i][j]))
                {
                    s[i][j] /= r;
                }
                else
                {
                    if (singExc)
                        throw SingMatrixExc ("Cannot invert singular matrix.");

                    return Matrix44();
                }
            }
        }
    }

    s[3][0] = -x[3][0] * s[0][0] - x[3][1] * s[1][0] - x[3][2] * s[2][0];
    s[3][1] = -x[3][0] * s[0][1] - x[3][1] * s[1][1] - x[3][2] * s[2][1];
    s[3][2] = -x[3][0] * s[0][2] - x[3][1] * s[1][2] - x[3][2] * s[2][2];

    return s;
}

template <class T>
inline T
Matrix44<T>::fastMinor( const int r0, const int r1, const int r2,
                        const int c0, const int c1, const int c2) const
{
    return x[r0][c0] * (x[r1][c1]*x[r2][c2] - x[r1][c2]*x[r2][c1])
         + x[r0][c1] * (x[r1][c2]*x[r2][c0] - x[r1][c0]*x[r2][c2])
         + x[r0][c2] * (x[r1][c0]*x[r2][c1] - x[r1][c1]*x[r2][c0]);
}

template <class T>
inline T
Matrix44<T>::minorOf (const int r, const int c) const
{
    int r0 = 0 + (r < 1 ? 1 : 0);
    int r1 = 1 + (r < 2 ? 1 : 0);
    int r2 = 2 + (r < 3 ? 1 : 0);
    int c0 = 0 + (c < 1 ? 1 : 0);
    int c1 = 1 + (c < 2 ? 1 : 0);
    int c2 = 2 + (c < 3 ? 1 : 0);

    Matrix33<T> working (x[r0][c0],x[r1][c0],x[r2][c0],
                         x[r0][c1],x[r1][c1],x[r2][c1],
                         x[r0][c2],x[r1][c2],x[r2][c2]);

    return working.determinant();
}

template <class T>
inline T
Matrix44<T>::determinant () const
{
    T sum = (T)0;

    if (x[0][3] != 0.) sum -= x[0][3] * fastMinor(1,2,3,0,1,2);
    if (x[1][3] != 0.) sum += x[1][3] * fastMinor(0,2,3,0,1,2);
    if (x[2][3] != 0.) sum -= x[2][3] * fastMinor(0,1,3,0,1,2);
    if (x[3][3] != 0.) sum += x[3][3] * fastMinor(0,1,2,0,1,2);

    return sum;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setEulerAngles (const Vec3<S>& r)
{
    S cos_rz, sin_rz, cos_ry, sin_ry, cos_rx, sin_rx;
    
    cos_rz = Math<T>::cos (r[2]);
    cos_ry = Math<T>::cos (r[1]);
    cos_rx = Math<T>::cos (r[0]);
    
    sin_rz = Math<T>::sin (r[2]);
    sin_ry = Math<T>::sin (r[1]);
    sin_rx = Math<T>::sin (r[0]);
    
    x[0][0] =  cos_rz * cos_ry;
    x[0][1] =  sin_rz * cos_ry;
    x[0][2] = -sin_ry;
    x[0][3] =  0;
    
    x[1][0] = -sin_rz * cos_rx + cos_rz * sin_ry * sin_rx;
    x[1][1] =  cos_rz * cos_rx + sin_rz * sin_ry * sin_rx;
    x[1][2] =  cos_ry * sin_rx;
    x[1][3] =  0;
    
    x[2][0] =  sin_rz * sin_rx + cos_rz * sin_ry * cos_rx;
    x[2][1] = -cos_rz * sin_rx + sin_rz * sin_ry * cos_rx;
    x[2][2] =  cos_ry * cos_rx;
    x[2][3] =  0;

    x[3][0] =  0;
    x[3][1] =  0;
    x[3][2] =  0;
    x[3][3] =  1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setAxisAngle (const Vec3<S>& axis, S angle)
{
    Vec3<S> unit (axis.normalized());
    S sine   = Math<T>::sin (angle);
    S cosine = Math<T>::cos (angle);

    x[0][0] = unit[0] * unit[0] * (1 - cosine) + cosine;
    x[0][1] = unit[0] * unit[1] * (1 - cosine) + unit[2] * sine;
    x[0][2] = unit[0] * unit[2] * (1 - cosine) - unit[1] * sine;
    x[0][3] = 0;

    x[1][0] = unit[0] * unit[1] * (1 - cosine) - unit[2] * sine;
    x[1][1] = unit[1] * unit[1] * (1 - cosine) + cosine;
    x[1][2] = unit[1] * unit[2] * (1 - cosine) + unit[0] * sine;
    x[1][3] = 0;

    x[2][0] = unit[0] * unit[2] * (1 - cosine) + unit[1] * sine;
    x[2][1] = unit[1] * unit[2] * (1 - cosine) - unit[0] * sine;
    x[2][2] = unit[2] * unit[2] * (1 - cosine) + cosine;
    x[2][3] = 0;

    x[3][0] = 0;
    x[3][1] = 0;
    x[3][2] = 0;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::rotate (const Vec3<S> &r)
{
    S cos_rz, sin_rz, cos_ry, sin_ry, cos_rx, sin_rx;
    S m00, m01, m02;
    S m10, m11, m12;
    S m20, m21, m22;

    cos_rz = Math<S>::cos (r[2]);
    cos_ry = Math<S>::cos (r[1]);
    cos_rx = Math<S>::cos (r[0]);
    
    sin_rz = Math<S>::sin (r[2]);
    sin_ry = Math<S>::sin (r[1]);
    sin_rx = Math<S>::sin (r[0]);

    m00 =  cos_rz *  cos_ry;
    m01 =  sin_rz *  cos_ry;
    m02 = -sin_ry;
    m10 = -sin_rz *  cos_rx + cos_rz * sin_ry * sin_rx;
    m11 =  cos_rz *  cos_rx + sin_rz * sin_ry * sin_rx;
    m12 =  cos_ry *  sin_rx;
    m20 = -sin_rz * -sin_rx + cos_rz * sin_ry * cos_rx;
    m21 =  cos_rz * -sin_rx + sin_rz * sin_ry * cos_rx;
    m22 =  cos_ry *  cos_rx;

    Matrix44<T> P (*this);

    x[0][0] = P[0][0] * m00 + P[1][0] * m01 + P[2][0] * m02;
    x[0][1] = P[0][1] * m00 + P[1][1] * m01 + P[2][1] * m02;
    x[0][2] = P[0][2] * m00 + P[1][2] * m01 + P[2][2] * m02;
    x[0][3] = P[0][3] * m00 + P[1][3] * m01 + P[2][3] * m02;

    x[1][0] = P[0][0] * m10 + P[1][0] * m11 + P[2][0] * m12;
    x[1][1] = P[0][1] * m10 + P[1][1] * m11 + P[2][1] * m12;
    x[1][2] = P[0][2] * m10 + P[1][2] * m11 + P[2][2] * m12;
    x[1][3] = P[0][3] * m10 + P[1][3] * m11 + P[2][3] * m12;

    x[2][0] = P[0][0] * m20 + P[1][0] * m21 + P[2][0] * m22;
    x[2][1] = P[0][1] * m20 + P[1][1] * m21 + P[2][1] * m22;
    x[2][2] = P[0][2] * m20 + P[1][2] * m21 + P[2][2] * m22;
    x[2][3] = P[0][3] * m20 + P[1][3] * m21 + P[2][3] * m22;

    return *this;
}

template <class T>
const Matrix44<T> &
Matrix44<T>::setScale (T s)
{
    memset (x, 0, sizeof (x));
    x[0][0] = s;
    x[1][1] = s;
    x[2][2] = s;
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setScale (const Vec3<S> &s)
{
    memset (x, 0, sizeof (x));
    x[0][0] = s[0];
    x[1][1] = s[1];
    x[2][2] = s[2];
    x[3][3] = 1;

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::scale (const Vec3<S> &s)
{
    x[0][0] *= s[0];
    x[0][1] *= s[0];
    x[0][2] *= s[0];
    x[0][3] *= s[0];

    x[1][0] *= s[1];
    x[1][1] *= s[1];
    x[1][2] *= s[1];
    x[1][3] *= s[1];

    x[2][0] *= s[2];
    x[2][1] *= s[2];
    x[2][2] *= s[2];
    x[2][3] *= s[2];

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setTranslation (const Vec3<S> &t)
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

    x[3][0] = t[0];
    x[3][1] = t[1];
    x[3][2] = t[2];
    x[3][3] = 1;

    return *this;
}

template <class T>
inline const Vec3<T>
Matrix44<T>::translation () const
{
    return Vec3<T> (x[3][0], x[3][1], x[3][2]);
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::translate (const Vec3<S> &t)
{
    x[3][0] += t[0] * x[0][0] + t[1] * x[1][0] + t[2] * x[2][0];
    x[3][1] += t[0] * x[0][1] + t[1] * x[1][1] + t[2] * x[2][1];
    x[3][2] += t[0] * x[0][2] + t[1] * x[1][2] + t[2] * x[2][2];
    x[3][3] += t[0] * x[0][3] + t[1] * x[1][3] + t[2] * x[2][3];

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::setShear (const Vec3<S> &h)
{
    x[0][0] = 1;
    x[0][1] = 0;
    x[0][2] = 0;
    x[0][3] = 0;

    x[1][0] = h[0];
    x[1][1] = 1;
    x[1][2] = 0;
    x[1][3] = 0;

    x[2][0] = h[1];
    x[2][1] = h[2];
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
const Matrix44<T> &
Matrix44<T>::setShear (const Shear6<S> &h)
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
const Matrix44<T> &
Matrix44<T>::shear (const Vec3<S> &h)
{
    //
    // In this case, we don't need a temp. copy of the matrix 
    // because we never use a value on the RHS after we've 
    // changed it on the LHS.
    // 

    for (int i=0; i < 4; i++)
    {
        x[2][i] += h[1] * x[0][i] + h[2] * x[1][i];
        x[1][i] += h[0] * x[0][i];
    }

    return *this;
}

template <class T>
template <class S>
const Matrix44<T> &
Matrix44<T>::shear (const Shear6<S> &h)
{
    Matrix44<T> P (*this);

    for (int i=0; i < 4; i++)
    {
        x[0][i] = P[0][i] + h.yx * P[1][i] + h.zx * P[2][i];
        x[1][i] = h.xy * P[0][i] + P[1][i] + h.zy * P[2][i];
        x[2][i] = h.xz * P[0][i] + h.yz * P[1][i] + P[2][i];
    }

    return *this;
}


//--------------------------------
// Implementation of stream output
//--------------------------------

template <class T>
std::ostream &
operator << (std::ostream &s, const Matrix33<T> &m)
{
    std::ios_base::fmtflags oldFlags = s.flags();
    int width;

    if (s.flags() & std::ios_base::fixed)
    {
        s.setf (std::ios_base::showpoint);
        width = s.precision() + 5;
    }
    else
    {
        s.setf (std::ios_base::scientific);
        s.setf (std::ios_base::showpoint);
        width = s.precision() + 8;
    }

    s << "(" << std::setw (width) << m[0][0] <<
         " " << std::setw (width) << m[0][1] <<
         " " << std::setw (width) << m[0][2] << "\n" <<

         " " << std::setw (width) << m[1][0] <<
         " " << std::setw (width) << m[1][1] <<
         " " << std::setw (width) << m[1][2] << "\n" <<

         " " << std::setw (width) << m[2][0] <<
         " " << std::setw (width) << m[2][1] <<
         " " << std::setw (width) << m[2][2] << ")\n";

    s.flags (oldFlags);
    return s;
}

template <class T>
std::ostream &
operator << (std::ostream &s, const Matrix44<T> &m)
{
    std::ios_base::fmtflags oldFlags = s.flags();
    int width;

    if (s.flags() & std::ios_base::fixed)
    {
        s.setf (std::ios_base::showpoint);
        width = s.precision() + 5;
    }
    else
    {
        s.setf (std::ios_base::scientific);
        s.setf (std::ios_base::showpoint);
        width = s.precision() + 8;
    }

    s << "(" << std::setw (width) << m[0][0] <<
         " " << std::setw (width) << m[0][1] <<
         " " << std::setw (width) << m[0][2] <<
         " " << std::setw (width) << m[0][3] << "\n" <<

         " " << std::setw (width) << m[1][0] <<
         " " << std::setw (width) << m[1][1] <<
         " " << std::setw (width) << m[1][2] <<
         " " << std::setw (width) << m[1][3] << "\n" <<

         " " << std::setw (width) << m[2][0] <<
         " " << std::setw (width) << m[2][1] <<
         " " << std::setw (width) << m[2][2] <<
         " " << std::setw (width) << m[2][3] << "\n" <<

         " " << std::setw (width) << m[3][0] <<
         " " << std::setw (width) << m[3][1] <<
         " " << std::setw (width) << m[3][2] <<
         " " << std::setw (width) << m[3][3] << ")\n";

    s.flags (oldFlags);
    return s;
}


//---------------------------------------------------------------
// Implementation of vector-times-matrix multiplication operators
//---------------------------------------------------------------

template <class S, class T>
inline const Vec2<S> &
operator *= (Vec2<S> &v, const Matrix33<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + m[2][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + m[2][1]);
    S w = S(v.x * m[0][2] + v.y * m[1][2] + m[2][2]);

    v.x = x / w;
    v.y = y / w;

    return v;
}

template <class S, class T>
inline Vec2<S>
operator * (const Vec2<S> &v, const Matrix33<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + m[2][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + m[2][1]);
    S w = S(v.x * m[0][2] + v.y * m[1][2] + m[2][2]);

    return Vec2<S> (x / w, y / w);
}


template <class S, class T>
inline const Vec3<S> &
operator *= (Vec3<S> &v, const Matrix33<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2]);

    v.x = x;
    v.y = y;
    v.z = z;

    return v;
}

template <class S, class T>
inline Vec3<S>
operator * (const Vec3<S> &v, const Matrix33<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2]);

    return Vec3<S> (x, y, z);
}


template <class S, class T>
inline const Vec3<S> &
operator *= (Vec3<S> &v, const Matrix44<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0] + m[3][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1] + m[3][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2] + m[3][2]);
    S w = S(v.x * m[0][3] + v.y * m[1][3] + v.z * m[2][3] + m[3][3]);

    v.x = x / w;
    v.y = y / w;
    v.z = z / w;

    return v;
}

template <class S, class T>
inline Vec3<S>
operator * (const Vec3<S> &v, const Matrix44<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0] + m[3][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1] + m[3][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2] + m[3][2]);
    S w = S(v.x * m[0][3] + v.y * m[1][3] + v.z * m[2][3] + m[3][3]);

    return Vec3<S> (x / w, y / w, z / w);
}


template <class S, class T>
inline const Vec4<S> &
operator *= (Vec4<S> &v, const Matrix44<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0] + v.w * m[3][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1] + v.w * m[3][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2] + v.w * m[3][2]);
    S w = S(v.x * m[0][3] + v.y * m[1][3] + v.z * m[2][3] + v.w * m[3][3]);

    v.x = x;
    v.y = y;
    v.z = z;
    v.w = w;

    return v;
}

template <class S, class T>
inline Vec4<S>
operator * (const Vec4<S> &v, const Matrix44<T> &m)
{
    S x = S(v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0] + v.w * m[3][0]);
    S y = S(v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1] + v.w * m[3][1]);
    S z = S(v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2] + v.w * m[3][2]);
    S w = S(v.x * m[0][3] + v.y * m[1][3] + v.z * m[2][3] + v.w * m[3][3]);

    return Vec4<S> (x, y, z, w);
}

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IMATHMATRIX_H
