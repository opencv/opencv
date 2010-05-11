#include "clapack.h"

/* Subroutine */ int dlabad_(doublereal *small, doublereal *large)
{
    /* Builtin functions */
    double d_lg10(doublereal *), sqrt(doublereal);


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLABAD takes as input the values computed by DLAMCH for underflow and */
/*  overflow, and returns the square root of each of these values if the */
/*  log of LARGE is sufficiently large.  This subroutine is intended to */
/*  identify machines with a large exponent range, such as the Crays, and */
/*  redefine the underflow and overflow limits to be the square roots of */
/*  the values computed by DLAMCH.  This subroutine is needed because */
/*  DLAMCH does not compensate for poor arithmetic in the upper half of */
/*  the exponent range, as is found on a Cray. */

/*  Arguments */
/*  ========= */

/*  SMALL   (input/output) DOUBLE PRECISION */
/*          On entry, the underflow threshold as computed by DLAMCH. */
/*          On exit, if LOG10(LARGE) is sufficiently large, the square */
/*          root of SMALL, otherwise unchanged. */

/*  LARGE   (input/output) DOUBLE PRECISION */
/*          On entry, the overflow threshold as computed by DLAMCH. */
/*          On exit, if LOG10(LARGE) is sufficiently large, the square */
/*          root of LARGE, otherwise unchanged. */

/*  ===================================================================== */

/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     If it looks like we're on a Cray, take the square root of */
/*     SMALL and LARGE to avoid overflow and underflow problems. */

    if (d_lg10(large) > 2e3) {
	*small = sqrt(*small);
	*large = sqrt(*large);
    }

    return 0;

/*     End of DLABAD */

} /* dlabad_ */
