/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DCOMBSSQ adds two scaled sum of squares quantities.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//
// Definition:
// ===========
//
//      SUBROUTINE DCOMBSSQ( V1, V2 )
//
//      .. Array Arguments ..
//      DOUBLE PRECISION   V1( 2 ), V2( 2 )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DCOMBSSQ adds two scaled sum of squares quantities, V1 := V1 + V2.
//> That is,
//>
//>    V1_scale**2 * V1_sumsq := V1_scale**2 * V1_sumsq
//>                            + V2_scale**2 * V2_sumsq
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in,out] V1
//> \verbatim
//>          V1 is DOUBLE PRECISION array, dimension (2).
//>          The first scaled sum.
//>          V1(1) = V1_scale, V1(2) = V1_sumsq.
//> \endverbatim
//>
//> \param[in] V2
//> \verbatim
//>          V2 is DOUBLE PRECISION array, dimension (2).
//>          The second scaled sum.
//>          V2(1) = V2_scale, V2(2) = V2_sumsq.
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
//> \date November 2018
//
//> \ingroup OTHERauxiliary
//
// =====================================================================
/* Subroutine */ int dcombssq_(double *v1, double *v2)
{
    // System generated locals
    double d__1;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2018
    //
    //    .. Array Arguments ..
    //    ..
    //
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --v2;
    --v1;

    // Function Body
    if (v1[1] >= v2[1]) {
	if (v1[1] != 0.) {
	    // Computing 2nd power
	    d__1 = v2[1] / v1[1];
	    v1[2] += d__1 * d__1 * v2[2];
	}
    } else {
	// Computing 2nd power
	d__1 = v1[1] / v2[1];
	v1[2] = v2[2] + d__1 * d__1 * v1[2];
	v1[1] = v2[1];
    }
    return 0;
    //
    //    End of DCOMBSSQ
    //
} // dcombssq_

