/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b SSCAL
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE SSCAL(N,SA,SX,INCX)
//
//      .. Scalar Arguments ..
//      REAL SA
//      INTEGER INCX,N
//      ..
//      .. Array Arguments ..
//      REAL SX(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    SSCAL scales a vector by a constant.
//>    uses unrolled loops for increment equal to 1.
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
//> \param[in] SA
//> \verbatim
//>          SA is REAL
//>           On entry, SA specifies the scalar alpha.
//> \endverbatim
//>
//> \param[in,out] SX
//> \verbatim
//>          SX is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>         storage spacing between elements of SX
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
//> \ingroup single_blas_level1
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>     jack dongarra, linpack, 3/11/78.
//>     modified 3/93 to return if incx .le. 0.
//>     modified 12/3/93, array(1) declarations changed to array(*)
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int sscal_(int *n, float *sa, float *sx, int *incx)
{
    // System generated locals
    int i__1, i__2;

    // Local variables
    int i__, m, mp1, nincx;

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
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    // Parameter adjustments
    --sx;

    // Function Body
    if (*n <= 0 || *incx <= 0) {
	return 0;
    }
    if (*incx == 1) {
	//
	//       code for increment equal to 1
	//
	//
	//       clean-up loop
	//
	m = *n % 5;
	if (m != 0) {
	    i__1 = m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		sx[i__] = *sa * sx[i__];
	    }
	    if (*n < 5) {
		return 0;
	    }
	}
	mp1 = m + 1;
	i__1 = *n;
	for (i__ = mp1; i__ <= i__1; i__ += 5) {
	    sx[i__] = *sa * sx[i__];
	    sx[i__ + 1] = *sa * sx[i__ + 1];
	    sx[i__ + 2] = *sa * sx[i__ + 2];
	    sx[i__ + 3] = *sa * sx[i__ + 3];
	    sx[i__ + 4] = *sa * sx[i__ + 4];
	}
    } else {
	//
	//       code for increment not equal to 1
	//
	nincx = *n * *incx;
	i__1 = nincx;
	i__2 = *incx;
	for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	    sx[i__] = *sa * sx[i__];
	}
    }
    return 0;
} // sscal_

