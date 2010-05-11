#include "clapack.h"

doublereal sasum_(integer *n, real *sx, integer *incx)
{
    /* System generated locals */
    integer i__1, i__2;
    real ret_val, r__1, r__2, r__3, r__4, r__5, r__6;

    /* Local variables */
    integer i__, m, mp1, nincx;
    real stemp;

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*     takes the sum of the absolute values. */
/*     uses unrolled loops for increment equal to one. */
/*     jack dongarra, linpack, 3/11/78. */
/*     modified 3/93 to return if incx .le. 0. */
/*     modified 12/3/93, array(1) declarations changed to array(*) */


/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
    /* Parameter adjustments */
    --sx;

    /* Function Body */
    ret_val = 0.f;
    stemp = 0.f;
    if (*n <= 0 || *incx <= 0) {
	return ret_val;
    }
    if (*incx == 1) {
	goto L20;
    }

/*        code for increment not equal to 1 */

    nincx = *n * *incx;
    i__1 = nincx;
    i__2 = *incx;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	stemp += (r__1 = sx[i__], dabs(r__1));
/* L10: */
    }
    ret_val = stemp;
    return ret_val;

/*        code for increment equal to 1 */


/*        clean-up loop */

L20:
    m = *n % 6;
    if (m == 0) {
	goto L40;
    }
    i__2 = m;
    for (i__ = 1; i__ <= i__2; ++i__) {
	stemp += (r__1 = sx[i__], dabs(r__1));
/* L30: */
    }
    if (*n < 6) {
	goto L60;
    }
L40:
    mp1 = m + 1;
    i__2 = *n;
    for (i__ = mp1; i__ <= i__2; i__ += 6) {
	stemp = stemp + (r__1 = sx[i__], dabs(r__1)) + (r__2 = sx[i__ + 1], 
		dabs(r__2)) + (r__3 = sx[i__ + 2], dabs(r__3)) + (r__4 = sx[
		i__ + 3], dabs(r__4)) + (r__5 = sx[i__ + 4], dabs(r__5)) + (
		r__6 = sx[i__ + 5], dabs(r__6));
/* L50: */
    }
L60:
    ret_val = stemp;
    return ret_val;
} /* sasum_ */
