/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLASSQ updates a sum of squares represented in scaled form.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASSQ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlassq.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlassq.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlassq.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASSQ( N, X, INCX, SCALE, SUMSQ )
//
//      .. Scalar Arguments ..
//      INTEGER            INCX, N
//      DOUBLE PRECISION   SCALE, SUMSQ
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   X( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASSQ  returns the values  scl  and  smsq  such that
//>
//>    ( scl**2 )*smsq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq,
//>
//> where  x( i ) = X( 1 + ( i - 1 )*INCX ). The value of  sumsq  is
//> assumed to be non-negative and  scl  returns the value
//>
//>    scl = max( scale, abs( x( i ) ) ).
//>
//> scale and sumsq must be supplied in SCALE and SUMSQ and
//> scl and smsq are overwritten on SCALE and SUMSQ respectively.
//>
//> The routine makes only one pass through the vector x.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of elements to be used from the vector X.
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension (1+(N-1)*INCX)
//>          The vector for which a scaled sum of squares is computed.
//>             x( i )  = X( 1 + ( i - 1 )*INCX ), 1 <= i <= n.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>          The increment between successive values of the vector X.
//>          INCX > 0.
//> \endverbatim
//>
//> \param[in,out] SCALE
//> \verbatim
//>          SCALE is DOUBLE PRECISION
//>          On entry, the value  scale  in the equation above.
//>          On exit, SCALE is overwritten with  scl , the scaling factor
//>          for the sum of squares.
//> \endverbatim
//>
//> \param[in,out] SUMSQ
//> \verbatim
//>          SUMSQ is DOUBLE PRECISION
//>          On entry, the value  sumsq  in the equation above.
//>          On exit, SUMSQ is overwritten with  smsq , the basic sum of
//>          squares from which  scl  has been factored out.
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
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
// =====================================================================
/* Subroutine */ int dlassq_(int *n, double *x, int *incx, double *scale,
	double *sumsq)
{
    // System generated locals
    int i__1, i__2;
    double d__1;

    // Local variables
    int ix;
    double absxi;
    extern int disnan_(double *);

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --x;

    // Function Body
    if (*n > 0) {
	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    absxi = (d__1 = x[ix], abs(d__1));
	    if (absxi > 0. || disnan_(&absxi)) {
		if (*scale < absxi) {
		    // Computing 2nd power
		    d__1 = *scale / absxi;
		    *sumsq = *sumsq * (d__1 * d__1) + 1;
		    *scale = absxi;
		} else {
		    // Computing 2nd power
		    d__1 = absxi / *scale;
		    *sumsq += d__1 * d__1;
		}
	    }
// L10:
	}
    }
    return 0;
    //
    //    End of DLASSQ
    //
} // dlassq_

