/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLARFG generates an elementary reflector (Householder matrix).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARFG + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarfg.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarfg.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarfg.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARFG( N, ALPHA, X, INCX, TAU )
//
//      .. Scalar Arguments ..
//      INTEGER            INCX, N
//      DOUBLE PRECISION   ALPHA, TAU
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
//> DLARFG generates a real elementary reflector H of order n, such
//> that
//>
//>       H * ( alpha ) = ( beta ),   H**T * H = I.
//>           (   x   )   (   0  )
//>
//> where alpha and beta are scalars, and x is an (n-1)-element real
//> vector. H is represented in the form
//>
//>       H = I - tau * ( 1 ) * ( 1 v**T ) ,
//>                     ( v )
//>
//> where tau is a real scalar and v is a real (n-1)-element
//> vector.
//>
//> If the elements of x are all zero, then tau = 0 and H is taken to be
//> the unit matrix.
//>
//> Otherwise  1 <= tau <= 2.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the elementary reflector.
//> \endverbatim
//>
//> \param[in,out] ALPHA
//> \verbatim
//>          ALPHA is DOUBLE PRECISION
//>          On entry, the value alpha.
//>          On exit, it is overwritten with the value beta.
//> \endverbatim
//>
//> \param[in,out] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension
//>                         (1+(N-2)*abs(INCX))
//>          On entry, the vector x.
//>          On exit, it is overwritten with the vector v.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>          The increment between elements of X. INCX > 0.
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>          The value tau.
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
//> \ingroup doubleOTHERauxiliary
//
// =====================================================================
/* Subroutine */ int dlarfg_(int *n, double *alpha, double *x, int *incx,
	double *tau)
{
    // System generated locals
    int i__1;
    double d__1;

    // Local variables
    int j, knt;
    double beta;
    extern double dnrm2_(int *, double *, int *);
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    double xnorm;
    extern double dlapy2_(double *, double *), dlamch_(char *);
    double safmin, rsafmn;

    //
    // -- LAPACK auxiliary routine (version 3.8.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
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
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --x;

    // Function Body
    if (*n <= 1) {
	*tau = 0.;
	return 0;
    }
    i__1 = *n - 1;
    xnorm = dnrm2_(&i__1, &x[1], incx);
    if (xnorm == 0.) {
	//
	//       H  =  I
	//
	*tau = 0.;
    } else {
	//
	//       general case
	//
	d__1 = dlapy2_(alpha, &xnorm);
	beta = -d_sign(&d__1, alpha);
	safmin = dlamch_("S") / dlamch_("E");
	knt = 0;
	if (abs(beta) < safmin) {
	    //
	    //          XNORM, BETA may be inaccurate; scale X and recompute them
	    //
	    rsafmn = 1. / safmin;
L10:
	    ++knt;
	    i__1 = *n - 1;
	    dscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    *alpha *= rsafmn;
	    if (abs(beta) < safmin && knt < 20) {
		goto L10;
	    }
	    //
	    //          New BETA is at most 1, at least SAFMIN
	    //
	    i__1 = *n - 1;
	    xnorm = dnrm2_(&i__1, &x[1], incx);
	    d__1 = dlapy2_(alpha, &xnorm);
	    beta = -d_sign(&d__1, alpha);
	}
	*tau = (beta - *alpha) / beta;
	i__1 = *n - 1;
	d__1 = 1. / (*alpha - beta);
	dscal_(&i__1, &d__1, &x[1], incx);
	//
	//       If ALPHA is subnormal, it may lose relative accuracy
	//
	i__1 = knt;
	for (j = 1; j <= i__1; ++j) {
	    beta *= safmin;
// L20:
	}
	*alpha = beta;
    }
    return 0;
    //
    //    End of DLARFG
    //
} // dlarfg_

