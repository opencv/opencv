/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DNRM2
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      DOUBLE PRECISION FUNCTION DNRM2(N,X,INCX)
//
//      .. Scalar Arguments ..
//      INTEGER INCX,N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION X(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DNRM2 returns the euclidean norm of a vector via the function
//> name, so that
//>
//>    DNRM2 := sqrt( x'*x )
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>         number of elements in input vector(s)
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>         storage spacing between elements of DX
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date November 2017
//
//> \ingroup double_blas_level1
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  -- This version written on 25-October-1982.
//>     Modified on 14-October-1993 to inline the call to DLASSQ.
//>     Sven Hammarling, Nag Ltd.
//> \endverbatim
//>
// =====================================================================
double dnrm2_(int *n, double *x, int *incx)
{
    // System generated locals
    int i__1, i__2;
    double ret_val, d__1;

    // Local variables
    int ix;
    double ssq, norm, scale, absxi;

    //
    // -- Reference BLAS level1 routine (version 3.8.0) --
    // -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    // Parameter adjustments
    --x;

    // Function Body
    if (*n < 1 || *incx < 1) {
	norm = 0.;
    } else if (*n == 1) {
	norm = abs(x[1]);
    } else {
	scale = 0.;
	ssq = 1.;
	//       The following loop is equivalent to this call to the LAPACK
	//       auxiliary routine:
	//       CALL DLASSQ( N, X, INCX, SCALE, SSQ )
	//
	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    if (x[ix] != 0.) {
		absxi = (d__1 = x[ix], abs(d__1));
		if (scale < absxi) {
		    // Computing 2nd power
		    d__1 = scale / absxi;
		    ssq = ssq * (d__1 * d__1) + 1.;
		    scale = absxi;
		} else {
		    // Computing 2nd power
		    d__1 = absxi / scale;
		    ssq += d__1 * d__1;
		}
	    }
// L10:
	}
	norm = scale * sqrt(ssq);
    }
    ret_val = norm;
    return ret_val;
    //
    //    End of DNRM2.
    //
} // dnrm2_

