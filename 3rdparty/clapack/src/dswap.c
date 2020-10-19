/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DSWAP
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
//
//      .. Scalar Arguments ..
//      INTEGER INCX,INCY,N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION DX(*),DY(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DSWAP interchanges two vectors.
//>    uses unrolled loops for increments equal to 1.
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
//> \param[in,out] DX
//> \verbatim
//>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>         storage spacing between elements of DX
//> \endverbatim
//>
//> \param[in,out] DY
//> \verbatim
//>          DY is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
//> \endverbatim
//>
//> \param[in] INCY
//> \verbatim
//>          INCY is INTEGER
//>         storage spacing between elements of DY
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
//>     jack dongarra, linpack, 3/11/78.
//>     modified 12/3/93, array(1) declarations changed to array(*)
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dswap_(int *n, double *dx, int *incx, double *dy, int *
	incy)
{
    // System generated locals
    int i__1;

    // Local variables
    int i__, m, ix, iy, mp1;
    double dtemp;

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
    --dy;
    --dx;

    // Function Body
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	//
	//      code for both increments equal to 1
	//
	//
	//      clean-up loop
	//
	m = *n % 3;
	if (m != 0) {
	    i__1 = m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		dtemp = dx[i__];
		dx[i__] = dy[i__];
		dy[i__] = dtemp;
	    }
	    if (*n < 3) {
		return 0;
	    }
	}
	mp1 = m + 1;
	i__1 = *n;
	for (i__ = mp1; i__ <= i__1; i__ += 3) {
	    dtemp = dx[i__];
	    dx[i__] = dy[i__];
	    dy[i__] = dtemp;
	    dtemp = dx[i__ + 1];
	    dx[i__ + 1] = dy[i__ + 1];
	    dy[i__ + 1] = dtemp;
	    dtemp = dx[i__ + 2];
	    dx[i__ + 2] = dy[i__ + 2];
	    dy[i__ + 2] = dtemp;
	}
    } else {
	//
	//      code for unequal increments or equal increments not equal
	//        to 1
	//
	ix = 1;
	iy = 1;
	if (*incx < 0) {
	    ix = (-(*n) + 1) * *incx + 1;
	}
	if (*incy < 0) {
	    iy = (-(*n) + 1) * *incy + 1;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dtemp = dx[ix];
	    dx[ix] = dy[iy];
	    dy[iy] = dtemp;
	    ix += *incx;
	    iy += *incy;
	}
    }
    return 0;
} // dswap_

