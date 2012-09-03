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





//----------------------------------------------------------------------------
//
//	Implementation of non-template items declared in ImathMatrixAlgo.h
//
//----------------------------------------------------------------------------

#include "ImathMatrixAlgo.h"
#include <cmath>

#if defined(OPENEXR_DLL)
    #define EXPORT_CONST __declspec(dllexport)
#else
    #define EXPORT_CONST const
#endif

namespace Imath {

EXPORT_CONST M33f identity33f ( 1, 0, 0,
				0, 1, 0,
				0, 0, 1);

EXPORT_CONST M33d identity33d ( 1, 0, 0,
				0, 1, 0,
				0, 0, 1);

EXPORT_CONST M44f identity44f ( 1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1);

EXPORT_CONST M44d identity44d ( 1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1);

namespace
{

class KahanSum
{
public:
    KahanSum() : _total(0), _correction(0) {}

    void
    operator+= (const double val)
    {
        const double y = val - _correction;
        const double t = _total + y;
        _correction = (t - _total) - y;
        _total = t;
    }

    double get() const
    {
        return _total;
    }

private:
    double _total;
    double _correction;
};

}

template <typename T>
M44d
procrustesRotationAndTranslation (const Vec3<T>* A, const Vec3<T>* B, const T* weights, const size_t numPoints, const bool doScale)
{
    if (numPoints == 0)
        return M44d();

    // Always do the accumulation in double precision:
    V3d Acenter (0.0);
    V3d Bcenter (0.0);
    double weightsSum = 0.0;

    if (weights == 0)
    {
        for (int i = 0; i < numPoints; ++i)
        {
            Acenter += (V3d) A[i];
            Bcenter += (V3d) B[i];
        }
        weightsSum = (double) numPoints;
    }
    else
    {
        for (int i = 0; i < numPoints; ++i)
        {
            const double w = weights[i];
            weightsSum += w;

            Acenter += w * (V3d) A[i];
            Bcenter += w * (V3d) B[i];
        }
    }

    if (weightsSum == 0)
        return M44d();

    Acenter /= weightsSum;
    Bcenter /= weightsSum;

    //
    // Find Q such that |Q*A - B|  (actually A-Acenter and B-Bcenter, weighted)
    // is minimized in the least squares sense.
    // From Golub/Van Loan, p.601
    //
    // A,B are 3xn
    // Let C = B A^T   (where A is 3xn and B^T is nx3, so C is 3x3)
    // Compute the SVD: C = U D V^T  (U,V rotations, D diagonal).
    // Throw away the D part, and return Q = U V^T
    M33d C (0.0);
    if (weights == 0)
    {
        for (int i = 0; i < numPoints; ++i)
            C += outerProduct ((V3d) B[i] - Bcenter, (V3d) A[i] - Acenter);
    }
    else
    {
        for (int i = 0; i < numPoints; ++i)
        {
            const double w = weights[i];
            C += outerProduct (w * ((V3d) B[i] - Bcenter), (V3d) A[i] - Acenter);
        }
    }

    M33d U, V;
    V3d S;
    jacobiSVD (C, U, S, V, Imath::limits<double>::epsilon(), true);

    // We want Q.transposed() here since we are going to be using it in the
    // Imath style (multiplying vectors on the right, v' = v*A^T):
    const M33d Qt = V * U.transposed();

    double s = 1.0;
    if (doScale && numPoints > 1)
    {
        // Finding a uniform scale: let us assume the Q is completely fixed
        // at this point (solving for both simultaneously seems much harder).  
        // We are trying to compute (again, per Golub and van Loan)
        //    min || s*A*Q - B ||_F
        // Notice that we've jammed a uniform scale in front of the Q.  
        // Now, the Frobenius norm (the least squares norm over matrices)
        // has the neat property that it is equivalent to minimizing the trace
        // of M^T*M (see your friendly neighborhood linear algebra text for a
        // derivation).  Thus, we can expand this out as
        //   min tr (s*A*Q - B)^T*(s*A*Q - B)
        // = min tr(Q^T*A^T*s*s*A*Q) + tr(B^T*B) - 2*tr(Q^T*A^T*s*B)  by linearity of the trace
        // = min s^2 tr(A^T*A) + tr(B^T*B) - 2*s*tr(Q^T*A^T*B)        using the fact that the trace is invariant
        //                                                            under similarity transforms Q*M*Q^T
        // If we differentiate w.r.t. s and set this to 0, we get
        // 0 = 2*s*tr(A^T*A) - 2*tr(Q^T*A^T*B)
        // so
        // 2*s*tr(A^T*A) = 2*s*tr(Q^T*A^T*B)
        // s = tr(Q^T*A^T*B) / tr(A^T*A)

        KahanSum traceATA;
        if (weights == 0)
        {
            for (int i = 0; i < numPoints; ++i)
                traceATA += ((V3d) A[i] - Acenter).length2();
        }
        else
        {
            for (int i = 0; i < numPoints; ++i)
                traceATA += ((double) weights[i]) * ((V3d) A[i] - Acenter).length2();
        }

        KahanSum traceBATQ;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                traceBATQ += Qt[j][i] * C[i][j];

        s = traceBATQ.get() / traceATA.get();
    }

    // Q is the rotation part of what we want to return.
    // The entire transform is:
    //    (translate origin to Bcenter) * Q * (translate Acenter to origin)
    //                last                                first
    // The effect of this on a point is:
    //    (translate origin to Bcenter) * Q * (translate Acenter to origin) * point
    //  = (translate origin to Bcenter) * Q * (-Acenter + point)
    //  = (translate origin to Bcenter) * (-Q*Acenter + Q*point)
    //  = (translate origin to Bcenter) * (translate Q*Acenter to origin) * Q*point
    //  = (translate Q*Acenter to Bcenter) * Q*point
    // So what we want to return is:
    //    (translate Q*Acenter to Bcenter) * Q
    //
    // In block form, this is:
    //   [ 1 0 0  | ] [       0 ] [ 1 0 0  |  ]   [ 1 0 0  | ] [           |   ]   [                 ]
    //   [ 0 1 0 tb ] [  s*Q  0 ] [ 0 1 0 -ta ] = [ 0 1 0 tb ] [  s*Q  -s*Q*ta ] = [   Q   tb-s*Q*ta ]
    //   [ 0 0 1  | ] [       0 ] [ 0 0 1  |  ]   [ 0 0 1  | ] [           |   ]   [                 ]
    //   [ 0 0 0  1 ] [ 0 0 0 1 ] [ 0 0 0  1  ]   [ 0 0 0  1 ] [ 0 0 0     1   ]   [ 0 0 0    1      ]
    // (ofc the whole thing is transposed for Imath).  
    const V3d translate = Bcenter - s*Acenter*Qt;

    return M44d (s*Qt.x[0][0], s*Qt.x[0][1], s*Qt.x[0][2], T(0),
                 s*Qt.x[1][0], s*Qt.x[1][1], s*Qt.x[1][2], T(0),
                 s*Qt.x[2][0], s*Qt.x[2][1], s*Qt.x[2][2], T(0),
                 translate.x, translate.y, translate.z, T(1));
} // procrustesRotationAndTranslation

template <typename T>
M44d
procrustesRotationAndTranslation (const Vec3<T>* A, const Vec3<T>* B, const size_t numPoints, const bool doScale)
{
    return procrustesRotationAndTranslation (A, B, (const T*) 0, numPoints, doScale);
} // procrustesRotationAndTranslation


template M44d procrustesRotationAndTranslation (const V3d* from, const V3d* to, const size_t numPoints, const bool doScale);
template M44d procrustesRotationAndTranslation (const V3f* from, const V3f* to, const size_t numPoints, const bool doScale);
template M44d procrustesRotationAndTranslation (const V3d* from, const V3d* to, const double* weights, const size_t numPoints, const bool doScale);
template M44d procrustesRotationAndTranslation (const V3f* from, const V3f* to, const float* weights, const size_t numPoints, const bool doScale);


namespace
{

// Applies the 2x2 Jacobi rotation
//   [  c s 0 ]    [ 1  0 0 ]    [  c 0 s ]
//   [ -s c 0 ] or [ 0  c s ] or [  0 1 0 ]
//   [  0 0 1 ]    [ 0 -s c ]    [ -s 0 c ]
// from the right; that is, computes
//   J * A
// for the Jacobi rotation J and the matrix A.  This is efficient because we
// only need to touch exactly the 2 columns that are affected, so we never
// need to explicitly construct the J matrix.  
template <typename T, int j, int k>
void
jacobiRotateRight (Imath::Matrix33<T>& A,
                   const T c,
                   const T s)
{
    for (int i = 0; i < 3; ++i)
    {
        const T tau1 = A[i][j];
        const T tau2 = A[i][k];
        A[i][j] = c * tau1 - s * tau2;
        A[i][k] = s * tau1 + c * tau2;
    }
}

template <typename T>
void
jacobiRotateRight (Imath::Matrix44<T>& A,
                   const int j,
                   const int k,
                   const T c,
                   const T s)
{
    for (int i = 0; i < 4; ++i)
    {
        const T tau1 = A[i][j];
        const T tau2 = A[i][k];
        A[i][j] = c * tau1 - s * tau2;
        A[i][k] = s * tau1 + c * tau2;
    }
}

// This routine solves the 2x2 SVD:
//     [  c1   s1 ] [ w   x ] [  c2  s2 ]   [ d1    0 ]
//     [          ] [       ] [         ] = [         ]
//     [ -s1   c1 ] [ y   z ] [ -s2  c2 ]   [  0   d2 ]
// where
//      [ w   x ]
//  A = [       ]
//      [ y   z ]
// is the subset of A consisting of the [j,k] entries, A([j k], [j k]) in
// Matlab parlance.  The method is the 'USVD' algorithm described in the
// following paper:
//    'Computation of the Singular Value Decomposition using Mesh-Connected Processors'
//    by Richard P. Brent, Franklin T. Luk, and Charles Van Loan
// It breaks the computation into two steps: the first symmetrizes the matrix,
// and the second diagonalizes the symmetric matrix.  
template <typename T, int j, int k, int l>
bool
twoSidedJacobiRotation (Imath::Matrix33<T>& A,
                        Imath::Matrix33<T>& U,
                        Imath::Matrix33<T>& V,
                        const T tol)
{
    // Load everything into local variables to make things easier on the
    // optimizer:
    const T w = A[j][j];
    const T x = A[j][k];
    const T y = A[k][j];
    const T z = A[k][k];

    // We will keep track of whether we're actually performing any rotations,
    // since if the matrix is already diagonal we'll end up with the identity
    // as our Jacobi rotation and we can short-circuit.
    bool changed = false;

    // The first step is to symmetrize the 2x2 matrix,
    //   [ c  s ]^T [ w x ] = [ p q ]
    //   [ -s c ]   [ y z ]   [ q r ]
    T mu_1 = w + z;
    T mu_2 = x - y;

    T c, s;
    if (std::abs(mu_2) <= tol*std::abs(mu_1))  // Already symmetric (to tolerance)
    {                                          // Note that the <= is important here
        c = T(1);                              // because we want to bypass the computation
        s = T(0);                              // of rho if mu_1 = mu_2 = 0.

        const T p = w;
        const T r = z;
        mu_1 = r - p;
        mu_2 = x + y;
    }
    else
    {
        const T rho = mu_1 / mu_2;
        s = T(1) / std::sqrt (T(1) + rho*rho);  // TODO is there a native inverse square root function?
        if (rho < 0)
            s = -s;
        c = s * rho;

        mu_1 = s * (x + y) + c * (z - w);   // = r - p
        mu_2 = T(2) * (c * x - s * z);      // = 2*q

        changed = true;
    }

    // The second stage diagonalizes,
    //   [ c2   s2 ]^T [ p q ] [ c2  s2 ]  = [ d1   0 ]
    //   [ -s2  c2 ]   [ q r ] [ -s2 c2 ]    [  0  d2 ]
    T c_2, s_2;
    if (std::abs(mu_2) <= tol*std::abs(mu_1))
    {
       c_2 = T(1);
       s_2 = T(0);
    }
    else
    {
        const T rho_2 = mu_1 / mu_2;
        T t_2 = T(1) / (std::abs(rho_2) + std::sqrt(1 + rho_2*rho_2));
        if (rho_2 < 0)
            t_2 = -t_2;
        c_2 = T(1) / std::sqrt (T(1) + t_2*t_2);
        s_2 = c_2 * t_2;

        changed = true;
    }

    const T c_1 = c_2 * c - s_2 * s;
    const T s_1 = s_2 * c + c_2 * s;

    if (!changed)
    {
        // We've decided that the off-diagonal entries are already small
        // enough, so we'll set them to zero.  This actually appears to result
        // in smaller errors than leaving them be, possibly because it prevents
        // us from trying to do extra rotations later that we don't need.
        A[k][j] = 0;
        A[j][k] = 0;
        return false;
    }

    const T d_1 = c_1*(w*c_2 - x*s_2) - s_1*(y*c_2 - z*s_2);
    const T d_2 = s_1*(w*s_2 + x*c_2) + c_1*(y*s_2 + z*c_2);

    // For the entries we just zeroed out, we'll just set them to 0, since
    // they should be 0 up to machine precision.  
    A[j][j] = d_1;
    A[k][k] = d_2;
    A[k][j] = 0;
    A[j][k] = 0;

    // Rotate the entries that _weren't_ involved in the 2x2 SVD:
    {
        // Rotate on the left by
        //    [  c1 s1 0 ]^T      [  c1 0 s1 ]^T      [ 1   0  0 ]^T
        //    [ -s1 c1 0 ]    or  [   0 1  0 ]    or  [ 0  c1 s1 ]
        //    [   0  0 1 ]        [ -s1 0 c1 ]        [ 0 -s1 c1 ]
        // This has the effect of adding the (weighted) ith and jth _rows_ to
        // each other.  
        const T tau1 = A[j][l];
        const T tau2 = A[k][l];
        A[j][l] = c_1 * tau1 - s_1 * tau2;
        A[k][l] = s_1 * tau1 + c_1 * tau2;
    }

    {
        // Rotate on the right by
        //    [  c2 s2 0 ]      [  c2 0 s2 ]      [ 1   0  0 ]
        //    [ -s2 c2 0 ]  or  [   0 1  0 ]  or  [ 0  c2 s2 ]
        //    [   0  0 1 ]      [ -s2 0 c2 ]      [ 0 -s2 c2 ]
        // This has the effect of adding the (weighted) ith and jth _columns_ to
        // each other.  
        const T tau1 = A[l][j];
        const T tau2 = A[l][k];
        A[l][j] = c_2 * tau1 - s_2 * tau2;
        A[l][k] = s_2 * tau1 + c_2 * tau2;
    }

    // Now apply the rotations to U and V:
    // Remember that we have 
    //    R1^T * A * R2 = D
    // This is in the 2x2 case, but after doing a bunch of these
    // we will get something like this for the 3x3 case:
    //   ... R1b^T * R1a^T * A * R2a * R2b * ... = D
    //   -----------------       ---------------
    //        = U^T                    = V
    // So,
    //   U = R1a * R1b * ...
    //   V = R2a * R2b * ...
    jacobiRotateRight<T, j, k> (U, c_1, s_1);
    jacobiRotateRight<T, j, k> (V, c_2, s_2);

    return true;
}

template <typename T>
bool
twoSidedJacobiRotation (Imath::Matrix44<T>& A,
                        int j,
                        int k,
                        Imath::Matrix44<T>& U,
                        Imath::Matrix44<T>& V,
                        const T tol)
{
    // Load everything into local variables to make things easier on the
    // optimizer:
    const T w = A[j][j];
    const T x = A[j][k];
    const T y = A[k][j];
    const T z = A[k][k];

    // We will keep track of whether we're actually performing any rotations,
    // since if the matrix is already diagonal we'll end up with the identity
    // as our Jacobi rotation and we can short-circuit.
    bool changed = false;

    // The first step is to symmetrize the 2x2 matrix,
    //   [ c  s ]^T [ w x ] = [ p q ]
    //   [ -s c ]   [ y z ]   [ q r ]
    T mu_1 = w + z;
    T mu_2 = x - y;

    T c, s;
    if (std::abs(mu_2) <= tol*std::abs(mu_1))  // Already symmetric (to tolerance)
    {                                          // Note that the <= is important here
        c = T(1);                              // because we want to bypass the computation
        s = T(0);                              // of rho if mu_1 = mu_2 = 0.

        const T p = w;
        const T r = z;
        mu_1 = r - p;
        mu_2 = x + y;
    }
    else
    {
        const T rho = mu_1 / mu_2;
        s = T(1) / std::sqrt (T(1) + rho*rho);  // TODO is there a native inverse square root function?
        if (rho < 0)
            s = -s;
        c = s * rho;

        mu_1 = s * (x + y) + c * (z - w);   // = r - p
        mu_2 = T(2) * (c * x - s * z);      // = 2*q

        changed = true;
    }

    // The second stage diagonalizes,
    //   [ c2   s2 ]^T [ p q ] [ c2  s2 ]  = [ d1   0 ]
    //   [ -s2  c2 ]   [ q r ] [ -s2 c2 ]    [  0  d2 ]
    T c_2, s_2;
    if (std::abs(mu_2) <= tol*std::abs(mu_1))
    {
       c_2 = T(1);
       s_2 = T(0);
    }
    else
    {
        const T rho_2 = mu_1 / mu_2;
        T t_2 = T(1) / (std::abs(rho_2) + std::sqrt(1 + rho_2*rho_2));
        if (rho_2 < 0)
            t_2 = -t_2;
        c_2 = T(1) / std::sqrt (T(1) + t_2*t_2);
        s_2 = c_2 * t_2;

        changed = true;
    }

    const T c_1 = c_2 * c - s_2 * s;
    const T s_1 = s_2 * c + c_2 * s;

    if (!changed)
    {
        // We've decided that the off-diagonal entries are already small
        // enough, so we'll set them to zero.  This actually appears to result
        // in smaller errors than leaving them be, possibly because it prevents
        // us from trying to do extra rotations later that we don't need.
        A[k][j] = 0;
        A[j][k] = 0;
        return false;
    }

    const T d_1 = c_1*(w*c_2 - x*s_2) - s_1*(y*c_2 - z*s_2);
    const T d_2 = s_1*(w*s_2 + x*c_2) + c_1*(y*s_2 + z*c_2);

    // For the entries we just zeroed out, we'll just set them to 0, since
    // they should be 0 up to machine precision.  
    A[j][j] = d_1;
    A[k][k] = d_2;
    A[k][j] = 0;
    A[j][k] = 0;

    // Rotate the entries that _weren't_ involved in the 2x2 SVD:
    for (int l = 0; l < 4; ++l)
    {
        if (l == j || l == k)
            continue;

        // Rotate on the left by
        //    [ 1               ]
        //    [   .             ]
        //    [     c2   s2     ]  j
        //    [        1        ]
        //    [    -s2   c2     ]  k
        //    [             .   ]
        //    [               1 ]
        //          j    k
        //
        // This has the effect of adding the (weighted) ith and jth _rows_ to
        // each other.  
        const T tau1 = A[j][l];
        const T tau2 = A[k][l];
        A[j][l] = c_1 * tau1 - s_1 * tau2;
        A[k][l] = s_1 * tau1 + c_1 * tau2;
    }

    for (int l = 0; l < 4; ++l)
    {
        // We set the A[j/k][j/k] entries already
        if (l == j || l == k)
            continue;

        // Rotate on the right by
        //    [ 1               ]
        //    [   .             ]
        //    [     c2   s2     ]  j
        //    [        1        ]
        //    [    -s2   c2     ]  k
        //    [             .   ]
        //    [               1 ]
        //          j    k
        //
        // This has the effect of adding the (weighted) ith and jth _columns_ to
        // each other.  
        const T tau1 = A[l][j];
        const T tau2 = A[l][k];
        A[l][j] = c_2 * tau1 - s_2 * tau2;
        A[l][k] = s_2 * tau1 + c_2 * tau2;
    }

    // Now apply the rotations to U and V:
    // Remember that we have 
    //    R1^T * A * R2 = D
    // This is in the 2x2 case, but after doing a bunch of these
    // we will get something like this for the 3x3 case:
    //   ... R1b^T * R1a^T * A * R2a * R2b * ... = D
    //   -----------------       ---------------
    //        = U^T                    = V
    // So,
    //   U = R1a * R1b * ...
    //   V = R2a * R2b * ...
    jacobiRotateRight (U, j, k, c_1, s_1);
    jacobiRotateRight (V, j, k, c_2, s_2);

    return true;
}

template <typename T>
void
swapColumns (Imath::Matrix33<T>& A, int j, int k)
{
    for (int i = 0; i < 3; ++i)
        std::swap (A[i][j], A[i][k]);
}

template <typename T>
T
maxOffDiag (const Imath::Matrix33<T>& A)
{
    T result = 0;
    result = std::max (result, std::abs (A[0][1]));
    result = std::max (result, std::abs (A[0][2]));
    result = std::max (result, std::abs (A[1][0]));
    result = std::max (result, std::abs (A[1][2]));
    result = std::max (result, std::abs (A[2][0]));
    result = std::max (result, std::abs (A[2][1]));
    return result;
}

template <typename T>
T
maxOffDiag (const Imath::Matrix44<T>& A)
{
    T result = 0;
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (i != j)
                result = std::max (result, std::abs (A[i][j]));
        }
    }

    return result;
}

template <typename T>
void
twoSidedJacobiSVD (Imath::Matrix33<T> A,
                   Imath::Matrix33<T>& U,
                   Imath::Vec3<T>& S,
                   Imath::Matrix33<T>& V,
                   const T tol,
                   const bool forcePositiveDeterminant)
{
    // The two-sided Jacobi SVD works by repeatedly zeroing out
    // off-diagonal entries of the matrix, 2 at a time.  Basically,
    // we can take our 3x3 matrix,
    //    [* * *]
    //    [* * *]
    //    [* * *]
    // and use a pair of orthogonal transforms to zero out, say, the
    // pair of entries (0, 1) and (1, 0):
    //  [ c1 s1  ] [* * *] [ c2 s2  ]   [*   *]
    //  [-s1 c1  ] [* * *] [-s2 c2  ] = [  * *]
    //  [       1] [* * *] [       1]   [* * *]
    // When we go to zero out the next pair of entries (say, (0, 2) and (2, 0))
    // then we don't expect those entries to stay 0:
    //  [ c1 s1  ] [*   *] [ c2 s2  ]   [* *  ]
    //  [-s1 c1  ] [  * *] [-s2 c2  ] = [* * *]
    //  [       1] [* * *] [       1]   [  * *]
    // However, if we keep doing this, we'll find that the off-diagonal entries
    // converge to 0 fairly quickly (convergence should be roughly cubic).  The 
    // result is a diagonal A matrix and a bunch of orthogonal transforms:
    //               [* * *]                [*    ]
    //  L1 L2 ... Ln [* * *] Rn ... R2 R1 = [  *  ]
    //               [* * *]                [    *]
    //  ------------ ------- ------------   -------
    //      U^T         A         V            S
    // This turns out to be highly accurate because (1) orthogonal transforms
    // are extremely stable to compute and apply (this is why QR factorization
    // works so well, FWIW) and because (2) by applying everything to the original
    // matrix A instead of computing (A^T * A) we avoid any precision loss that
    // would result from that.  
    U.makeIdentity();
    V.makeIdentity();

    const int maxIter = 20;  // In case we get really unlucky, prevents infinite loops
    const T absTol = tol * maxOffDiag (A);  // Tolerance is in terms of the maximum
    if (absTol != 0)                        // _off-diagonal_ entry.
    {
        int numIter = 0;
        do
        {
            ++numIter;
            bool changed = twoSidedJacobiRotation<T, 0, 1, 2> (A, U, V, tol);
            changed = twoSidedJacobiRotation<T, 0, 2, 1> (A, U, V, tol) || changed;
            changed = twoSidedJacobiRotation<T, 1, 2, 0> (A, U, V, tol) || changed;
            if (!changed)
                break;
        } while (maxOffDiag(A) > absTol && numIter < maxIter);
    }

    // The off-diagonal entries are (effectively) 0, so whatever's left on the
    // diagonal are the singular values:
    S.x = A[0][0];
    S.y = A[1][1];
    S.z = A[2][2];

    // Nothing thus far has guaranteed that the singular values are positive,
    // so let's go back through and flip them if not (since by contract we are
    // supposed to return all positive SVs):
    for (int i = 0; i < 3; ++i)
    {
        if (S[i] < 0)
        {
            // If we flip S[i], we need to flip the corresponding column of U
            // (we could also pick V if we wanted; it doesn't really matter):
            S[i] = -S[i];
            for (int j = 0; j < 3; ++j)
                U[j][i] = -U[j][i];
        }
    }

    // Order the singular values from largest to smallest; this requires
    // exactly two passes through the data using bubble sort:
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < (2 - i); ++j)
        {
            // No absolute values necessary since we already ensured that
            // they're positive:
            if (S[j] < S[j+1])
            {
                // If we swap singular values we also have to swap
                // corresponding columns in U and V:
                std::swap (S[j], S[j+1]);
                swapColumns (U, j, j+1);
                swapColumns (V, j, j+1);
            }
        }
    }

    if (forcePositiveDeterminant)
    {
        // We want to guarantee that the returned matrices always have positive
        // determinant.  We can do this by adding the appropriate number of
        // matrices of the form:
        //       [ 1       ]
        //  L =  [    1    ]
        //       [      -1 ]
        // Note that L' = L and L*L = Identity.  Thus we can add:
        //   U*L*L*S*V = (U*L)*(L*S)*V
        // if U has a negative determinant, and
        //   U*S*L*L*V = U*(S*L)*(L*V)
        // if V has a neg. determinant.
        if (U.determinant() < 0)
        {
            for (int i = 0; i < 3; ++i)
                U[i][2] = -U[i][2];
            S.z = -S.z;
        }
   
        if (V.determinant() < 0)
        {
            for (int i = 0; i < 3; ++i)
                V[i][2] = -V[i][2];
            S.z = -S.z;
        }
    }
}

template <typename T>
void
twoSidedJacobiSVD (Imath::Matrix44<T> A,
                   Imath::Matrix44<T>& U,
                   Imath::Vec4<T>& S,
                   Imath::Matrix44<T>& V,
                   const T tol,
                   const bool forcePositiveDeterminant)
{
    // Please see the Matrix33 version for a detailed description of the algorithm.
    U.makeIdentity();
    V.makeIdentity();

    const int maxIter = 20;  // In case we get really unlucky, prevents infinite loops
    const T absTol = tol * maxOffDiag (A);  // Tolerance is in terms of the maximum
    if (absTol != 0)                        // _off-diagonal_ entry.
    {
        int numIter = 0;
        do
        {
            ++numIter;
            bool changed = twoSidedJacobiRotation (A, 0, 1, U, V, tol);
            changed = twoSidedJacobiRotation (A, 0, 2, U, V, tol) || changed;
            changed = twoSidedJacobiRotation (A, 0, 3, U, V, tol) || changed;
            changed = twoSidedJacobiRotation (A, 1, 2, U, V, tol) || changed;
            changed = twoSidedJacobiRotation (A, 1, 3, U, V, tol) || changed;
            changed = twoSidedJacobiRotation (A, 2, 3, U, V, tol) || changed;
            if (!changed)
                break;
        } while (maxOffDiag(A) > absTol && numIter < maxIter);
    }

    // The off-diagonal entries are (effectively) 0, so whatever's left on the
    // diagonal are the singular values:
    S[0] = A[0][0];
    S[1] = A[1][1];
    S[2] = A[2][2];
    S[3] = A[3][3];

    // Nothing thus far has guaranteed that the singular values are positive,
    // so let's go back through and flip them if not (since by contract we are
    // supposed to return all positive SVs):
    for (int i = 0; i < 4; ++i)
    {
        if (S[i] < 0)
        {
            // If we flip S[i], we need to flip the corresponding column of U
            // (we could also pick V if we wanted; it doesn't really matter):
            S[i] = -S[i];
            for (int j = 0; j < 4; ++j)
                U[j][i] = -U[j][i];
        }
    }

    // Order the singular values from largest to smallest using insertion sort:
    for (int i = 1; i < 4; ++i)
    {
        const Imath::Vec4<T> uCol (U[0][i], U[1][i], U[2][i], U[3][i]);
        const Imath::Vec4<T> vCol (V[0][i], V[1][i], V[2][i], V[3][i]);
        const T sVal = S[i];

        int j = i - 1;
        while (std::abs (S[j]) < std::abs (sVal))
        {
            for (int k = 0; k < 4; ++k)
                U[k][j+1] = U[k][j];
            for (int k = 0; k < 4; ++k)
                V[k][j+1] = V[k][j];
            S[j+1] = S[j];

            --j;
            if (j < 0)
                break;
        }

        for (int k = 0; k < 4; ++k)
            U[k][j+1] = uCol[k];
        for (int k = 0; k < 4; ++k)
            V[k][j+1] = vCol[k];
        S[j+1] = sVal;
    }

    if (forcePositiveDeterminant)
    {
        // We want to guarantee that the returned matrices always have positive
        // determinant.  We can do this by adding the appropriate number of
        // matrices of the form:
        //       [ 1          ]
        //  L =  [    1       ]
        //       [       1    ]
        //       [         -1 ]
        // Note that L' = L and L*L = Identity.  Thus we can add:
        //   U*L*L*S*V = (U*L)*(L*S)*V
        // if U has a negative determinant, and
        //   U*S*L*L*V = U*(S*L)*(L*V)
        // if V has a neg. determinant.
        if (U.determinant() < 0)
        {
            for (int i = 0; i < 4; ++i)
                U[i][3] = -U[i][3];
            S[3] = -S[3];
        }
   
        if (V.determinant() < 0)
        {
            for (int i = 0; i < 4; ++i)
                V[i][3] = -V[i][3];
            S[3] = -S[3];
        }
    }
}

}

template <typename T>
void
jacobiSVD (const Imath::Matrix33<T>& A,
           Imath::Matrix33<T>& U,
           Imath::Vec3<T>& S,
           Imath::Matrix33<T>& V,
           const T tol,
           const bool forcePositiveDeterminant)
{
    twoSidedJacobiSVD (A, U, S, V, tol, forcePositiveDeterminant);
}

template <typename T>
void
jacobiSVD (const Imath::Matrix44<T>& A,
           Imath::Matrix44<T>& U,
           Imath::Vec4<T>& S,
           Imath::Matrix44<T>& V,
           const T tol,
           const bool forcePositiveDeterminant)
{
    twoSidedJacobiSVD (A, U, S, V, tol, forcePositiveDeterminant);
}

template void jacobiSVD (const Imath::Matrix33<float>& A,
                         Imath::Matrix33<float>& U,
                         Imath::Vec3<float>& S,
                         Imath::Matrix33<float>& V,
                         const float tol,
                         const bool forcePositiveDeterminant);
template void jacobiSVD (const Imath::Matrix33<double>& A,
                         Imath::Matrix33<double>& U,
                         Imath::Vec3<double>& S,
                         Imath::Matrix33<double>& V,
                         const double tol,
                         const bool forcePositiveDeterminant);
template void jacobiSVD (const Imath::Matrix44<float>& A,
                         Imath::Matrix44<float>& U,
                         Imath::Vec4<float>& S,
                         Imath::Matrix44<float>& V,
                         const float tol,
                         const bool forcePositiveDeterminant);
template void jacobiSVD (const Imath::Matrix44<double>& A,
                         Imath::Matrix44<double>& U,
                         Imath::Vec4<double>& S,
                         Imath::Matrix44<double>& V,
                         const double tol,
                         const bool forcePositiveDeterminant);

namespace
{

template <int j, int k, typename TM>
inline 
void
jacobiRotateRight (TM& A,
                   const typename TM::BaseType s,
                   const typename TM::BaseType tau)
{
    typedef typename TM::BaseType T;

    for (unsigned int i = 0; i < TM::dimensions(); ++i)
    {
        const T nu1 = A[i][j];
        const T nu2 = A[i][k];
        A[i][j] -= s * (nu2 + tau * nu1);
        A[i][k] += s * (nu1 - tau * nu2);
   }
}

template <int j, int k, int l, typename T>
bool
jacobiRotation (Matrix33<T>& A,
                Matrix33<T>& V,
                Vec3<T>& Z,
                const T tol)
{
    // Load everything into local variables to make things easier on the
    // optimizer:
    const T x = A[j][j];
    const T y = A[j][k];
    const T z = A[k][k];

    // The first stage diagonalizes,
    //   [ c  s ]^T [ x y ] [ c -s ]  = [ d1   0 ]
    //   [ -s c ]   [ y z ] [ s  c ]    [  0  d2 ]
    const T mu1 = z - x;
    const T mu2 = 2 * y;

    if (std::abs(mu2) <= tol*std::abs(mu1))
    {
        // We've decided that the off-diagonal entries are already small
        // enough, so we'll set them to zero.  This actually appears to result
        // in smaller errors than leaving them be, possibly because it prevents
        // us from trying to do extra rotations later that we don't need.
        A[j][k] = 0;
        return false;
    }
    const T rho = mu1 / mu2;
    const T t = (rho < 0 ? T(-1) : T(1)) / (std::abs(rho) + std::sqrt(1 + rho*rho));
    const T c = T(1) / std::sqrt (T(1) + t*t);
    const T s = t * c;
    const T tau = s / (T(1) + c);
    const T h = t * y;

    // Update diagonal elements.
    Z[j] -= h;
    Z[k] += h;
    A[j][j] -= h;
    A[k][k] += h;

    // For the entries we just zeroed out, we'll just set them to 0, since
    // they should be 0 up to machine precision.  
    A[j][k] = 0;

    // We only update upper triagnular elements of A, since
    // A is supposed to be symmetric.
    T& offd1 = l < j ? A[l][j] : A[j][l];
    T& offd2 = l < k ? A[l][k] : A[k][l];
    const T nu1 = offd1;
    const T nu2 = offd2;
    offd1 = nu1 - s * (nu2 + tau * nu1);
    offd2 = nu2 + s * (nu1 - tau * nu2); 

    // Apply rotation to V
    jacobiRotateRight<j, k> (V, s, tau);

    return true;
}

template <int j, int k, int l1, int l2, typename T>
bool
jacobiRotation (Matrix44<T>& A,
                Matrix44<T>& V,
                Vec4<T>& Z,
                const T tol)
{
    const T x = A[j][j];
    const T y = A[j][k];
    const T z = A[k][k];

    const T mu1 = z - x;
    const T mu2 = T(2) * y;

    // Let's see if rho^(-1) = mu2 / mu1 is less than tol
    // This test also checks if rho^2 will overflow 
    // when tol^(-1) < sqrt(limits<T>::max()).
    if (std::abs(mu2) <= tol*std::abs(mu1))
    {
        A[j][k] = 0;
        return true;
    }

    const T rho = mu1 / mu2;
    const T t = (rho < 0 ? T(-1) : T(1)) / (std::abs(rho) + std::sqrt(1 + rho*rho));
    const T c = T(1) / std::sqrt (T(1) + t*t);
    const T s = c * t;
    const T tau = s / (T(1) + c);
    const T h = t * y;

    Z[j] -= h;
    Z[k] += h;
    A[j][j] -= h;
    A[k][k] += h;
    A[j][k] = 0;

    {
        T& offd1 = l1 < j ? A[l1][j] : A[j][l1];
        T& offd2 = l1 < k ? A[l1][k] : A[k][l1];
        const T nu1 = offd1;
        const T nu2 = offd2;
        offd1 -= s * (nu2 + tau * nu1);
        offd2 += s * (nu1 - tau * nu2); 
    }

    {
        T& offd1 = l2 < j ? A[l2][j] : A[j][l2];
        T& offd2 = l2 < k ? A[l2][k] : A[k][l2];
        const T nu1 = offd1;
        const T nu2 = offd2;
        offd1 -= s * (nu2 + tau * nu1);
        offd2 += s * (nu1 - tau * nu2); 
    }

    jacobiRotateRight<j, k> (V, s, tau);

    return true;
}

template <typename TM>
inline
typename TM::BaseType
maxOffDiagSymm (const TM& A)
{
    typedef typename TM::BaseType T;
    T result = 0;
    for (unsigned int i = 0; i < TM::dimensions(); ++i)
        for (unsigned int j = i+1; j < TM::dimensions(); ++j)
            result = std::max (result, std::abs (A[i][j]));

   return result;
}

} // namespace

template <typename T>
void
jacobiEigenSolver (Matrix33<T>& A,
                   Vec3<T>& S,
                   Matrix33<T>& V,
                   const T tol)
{
    V.makeIdentity();
    for(int i = 0; i < 3; ++i) {
        S[i] = A[i][i];
    }

    const int maxIter = 20;  // In case we get really unlucky, prevents infinite loops
    const T absTol = tol * maxOffDiagSymm (A);  // Tolerance is in terms of the maximum
    if (absTol != 0)                        // _off-diagonal_ entry.
    {
        int numIter = 0;
        do
        {
            // Z is for accumulating small changes (h) to diagonal entries
            // of A for one sweep. Adding h's directly to A might cause
            // a cancellation effect when h is relatively very small to 
            // the corresponding diagonal entry of A and 
            // this will increase numerical errors
            Vec3<T> Z(0, 0, 0);
            ++numIter;
            bool changed = jacobiRotation<0, 1, 2> (A, V, Z, tol);
            changed = jacobiRotation<0, 2, 1> (A, V, Z, tol) || changed;
            changed = jacobiRotation<1, 2, 0> (A, V, Z, tol) || changed;
            // One sweep passed. Add accumulated changes (Z) to singular values (S)
            // Update diagonal elements of A for better accuracy as well.
            for(int i = 0; i < 3; ++i) {
                A[i][i] = S[i] += Z[i];
            }
            if (!changed)
                break;
        } while (maxOffDiagSymm(A) > absTol && numIter < maxIter);
    }
}

template <typename T>
void
jacobiEigenSolver (Matrix44<T>& A,
                   Vec4<T>& S,
                   Matrix44<T>& V,
                   const T tol)
{
    V.makeIdentity();

    for(int i = 0; i < 4; ++i) {
        S[i] = A[i][i];
    }

    const int maxIter = 20;  // In case we get really unlucky, prevents infinite loops
    const T absTol = tol * maxOffDiagSymm (A);  // Tolerance is in terms of the maximum
    if (absTol != 0)                        // _off-diagonal_ entry.
    {
        int numIter = 0;
        do
        {
            ++numIter;
            Vec4<T> Z(0, 0, 0, 0);
            bool changed = jacobiRotation<0, 1, 2, 3> (A, V, Z, tol);
            changed = jacobiRotation<0, 2, 1, 3> (A, V, Z, tol) || changed;
            changed = jacobiRotation<0, 3, 1, 2> (A, V, Z, tol) || changed;
            changed = jacobiRotation<1, 2, 0, 3> (A, V, Z, tol) || changed;
            changed = jacobiRotation<1, 3, 0, 2> (A, V, Z, tol) || changed;
            changed = jacobiRotation<2, 3, 0, 1> (A, V, Z, tol) || changed;
            for(int i = 0; i < 4; ++i) {
                A[i][i] = S[i] += Z[i];
            }
           if (!changed)
                break;
        } while (maxOffDiagSymm(A) > absTol && numIter < maxIter);
    }
}

template <typename TM, typename TV>
void
maxEigenVector (TM& A, TV& V)
{
    TV S;
    TM MV;
    jacobiEigenSolver(A, S, MV);

    int maxIdx(0);
    for(unsigned int i = 1; i < TV::dimensions(); ++i)
    {
        if(std::abs(S[i]) > std::abs(S[maxIdx]))
            maxIdx = i;
    }

    for(unsigned int i = 0; i < TV::dimensions(); ++i)
        V[i] = MV[i][maxIdx];
}

template <typename TM, typename TV>
void
minEigenVector (TM& A, TV& V)
{
    TV S;
    TM MV;
    jacobiEigenSolver(A, S, MV);

    int minIdx(0);
    for(unsigned int i = 1; i < TV::dimensions(); ++i)
    {
        if(std::abs(S[i]) < std::abs(S[minIdx]))
            minIdx = i;
    }

   for(unsigned int i = 0; i < TV::dimensions(); ++i)
        V[i] = MV[i][minIdx];
}

template void jacobiEigenSolver (Matrix33<float>& A,
                                 Vec3<float>& S,
                                 Matrix33<float>& V,
                                 const float tol);
template void jacobiEigenSolver (Matrix33<double>& A,
                                 Vec3<double>& S,
                                 Matrix33<double>& V,
                                 const double tol);
template void jacobiEigenSolver (Matrix44<float>& A,
                                 Vec4<float>& S,
                                 Matrix44<float>& V,
                                 const float tol);
template void jacobiEigenSolver (Matrix44<double>& A,
                                 Vec4<double>& S,
                                 Matrix44<double>& V,
                                 const double tol);

template void maxEigenVector (Matrix33<float>& A,
                              Vec3<float>& S);
template void maxEigenVector (Matrix44<float>& A,
                              Vec4<float>& S);
template void maxEigenVector (Matrix33<double>& A,
                              Vec3<double>& S);
template void maxEigenVector (Matrix44<double>& A,
                              Vec4<double>& S);

template void minEigenVector (Matrix33<float>& A,
                              Vec3<float>& S);
template void minEigenVector (Matrix44<float>& A,
                              Vec4<float>& S);
template void minEigenVector (Matrix33<double>& A,
                              Vec3<double>& S);
template void minEigenVector (Matrix44<double>& A,
                              Vec4<double>& S);

} // namespace Imath
