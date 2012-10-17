///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IMATHFRAME_H
#define INCLUDED_IMATHFRAME_H

namespace Imath {

template<class T> class Vec3;
template<class T> class Matrix44;

//
//  These methods compute a set of reference frames, defined by their
//  transformation matrix, along a curve. It is designed so that the
//  array of points and the array of matrices used to fetch these routines
//  don't need to be ordered as the curve.
//
//  A typical usage would be :
//
//      m[0] = Imath::firstFrame( p[0], p[1], p[2] );
//      for( int i = 1; i < n - 1; i++ )
//      {
//          m[i] = Imath::nextFrame( m[i-1], p[i-1], p[i], t[i-1], t[i] );
//      }
//      m[n-1] = Imath::lastFrame( m[n-2], p[n-2], p[n-1] );
//
//  See Graphics Gems I for the underlying algorithm.
//

template<class T> Matrix44<T> firstFrame( const Vec3<T>&,    // First point
                                          const Vec3<T>&,    // Second point
                                          const Vec3<T>& );  // Third point

template<class T> Matrix44<T> nextFrame( const Matrix44<T>&, // Previous matrix
                                         const Vec3<T>&,     // Previous point
                                         const Vec3<T>&,     // Current point
                                         Vec3<T>&,           // Previous tangent
                                         Vec3<T>& );         // Current tangent

template<class T> Matrix44<T> lastFrame( const Matrix44<T>&, // Previous matrix
                                         const Vec3<T>&,     // Previous point
                                         const Vec3<T>& );   // Last point

//
//  firstFrame - Compute the first reference frame along a curve.
//
//  This function returns the transformation matrix to the reference frame
//  defined by the three points 'pi', 'pj' and 'pk'. Note that if the two
//  vectors <pi,pj> and <pi,pk> are colinears, an arbitrary twist value will
//  be choosen.
//
//  Throw 'NullVecExc' if 'pi' and 'pj' are equals.
//

template<class T> Matrix44<T> firstFrame
(
    const Vec3<T>& pi,             // First point
    const Vec3<T>& pj,             // Second point
    const Vec3<T>& pk )            // Third point
{
    Vec3<T> t = pj - pi; t.normalizeExc();

    Vec3<T> n = t.cross( pk - pi ); n.normalize();
    if( n.length() == 0.0f )
    {
        int i = fabs( t[0] ) < fabs( t[1] ) ? 0 : 1;
        if( fabs( t[2] ) < fabs( t[i] )) i = 2;

        Vec3<T> v( 0.0, 0.0, 0.0 ); v[i] = 1.0;
        n = t.cross( v ); n.normalize();
    }

    Vec3<T> b = t.cross( n );

    Matrix44<T> M;

    M[0][0] =  t[0]; M[0][1] =  t[1]; M[0][2] =  t[2]; M[0][3] = 0.0,
    M[1][0] =  n[0]; M[1][1] =  n[1]; M[1][2] =  n[2]; M[1][3] = 0.0,
    M[2][0] =  b[0]; M[2][1] =  b[1]; M[2][2] =  b[2]; M[2][3] = 0.0,
    M[3][0] = pi[0]; M[3][1] = pi[1]; M[3][2] = pi[2]; M[3][3] = 1.0;

    return M;
}

//
//  nextFrame - Compute the next reference frame along a curve.
//
//  This function returns the transformation matrix to the next reference
//  frame defined by the previously computed transformation matrix and the
//  new point and tangent vector along the curve.
//

template<class T> Matrix44<T> nextFrame
(
    const Matrix44<T>&  Mi,             // Previous matrix
    const Vec3<T>&      pi,             // Previous point
    const Vec3<T>&      pj,             // Current point
    Vec3<T>&            ti,             // Previous tangent vector
    Vec3<T>&            tj )            // Current tangent vector
{
    Vec3<T> a(0.0, 0.0, 0.0);		// Rotation axis.
    T r = 0.0;				// Rotation angle.

    if( ti.length() != 0.0 && tj.length() != 0.0 )
    {
        ti.normalize(); tj.normalize();
        T dot = ti.dot( tj );

        //
        //  This is *really* necessary :
        //

        if( dot > 1.0 ) dot = 1.0;
        else if( dot < -1.0 ) dot = -1.0;

        r = acosf( dot );
        a = ti.cross( tj );
    }

    if( a.length() != 0.0 && r != 0.0 )
    {
        Matrix44<T> R; R.setAxisAngle( a, r );
        Matrix44<T> Tj; Tj.translate(  pj );
        Matrix44<T> Ti; Ti.translate( -pi );

        return Mi * Ti * R * Tj;
    }
    else
    {
        Matrix44<T> Tr; Tr.translate( pj - pi );

        return Mi * Tr;
    }
}

//
//  lastFrame - Compute the last reference frame along a curve.
//
//  This function returns the transformation matrix to the last reference
//  frame defined by the previously computed transformation matrix and the
//  last point along the curve.
//

template<class T> Matrix44<T> lastFrame
(
    const Matrix44<T>&  Mi,             // Previous matrix
    const Vec3<T>&      pi,             // Previous point
    const Vec3<T>&      pj )            // Last point
{
    Matrix44<T> Tr; Tr.translate( pj - pi );

    return Mi * Tr;
}

} // namespace Imath

#endif
