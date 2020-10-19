/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLAPY2 returns sqrt(x2+y2).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAPY2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlapy2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlapy2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlapy2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      DOUBLE PRECISION FUNCTION DLAPY2( X, Y )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   X, Y
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary
//> overflow.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] X
//> \verbatim
//>          X is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] Y
//> \verbatim
//>          Y is DOUBLE PRECISION
//>          X and Y specify the values x and y.
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
//> \date June 2017
//
//> \ingroup OTHERauxiliary
//
// =====================================================================
double dlapy2_(double *x, double *y)
{
    // System generated locals
    double ret_val, d__1;

    // Local variables
    int x_is_nan__, y_is_nan__;
    double w, z__, xabs, yabs;
    extern int disnan_(double *);

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
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
    //    .. Executable Statements ..
    //
    x_is_nan__ = disnan_(x);
    y_is_nan__ = disnan_(y);
    if (x_is_nan__) {
	ret_val = *x;
    }
    if (y_is_nan__) {
	ret_val = *y;
    }
    if (! (x_is_nan__ || y_is_nan__)) {
	xabs = abs(*x);
	yabs = abs(*y);
	w = max(xabs,yabs);
	z__ = min(xabs,yabs);
	if (z__ == 0.) {
	    ret_val = w;
	} else {
	    // Computing 2nd power
	    d__1 = z__ / w;
	    ret_val = w * sqrt(d__1 * d__1 + 1.);
	}
    }
    return ret_val;
    //
    //    End of DLAPY2
    //
} // dlapy2_

