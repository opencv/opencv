#include "clapack.h"

/* Subroutine */ int slabad_(real *small, real *large)
{
    /* Builtin functions */
    double r_lg10(real *), sqrt(doublereal);


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLABAD takes as input the values computed by SLAMCH for underflow and */
/*  overflow, and returns the square root of each of these values if the */
/*  log of LARGE is sufficiently large.  This subroutine is intended to */
/*  identify machines with a large exponent range, such as the Crays, and */
/*  redefine the underflow and overflow limits to be the square roots of */
/*  the values computed by SLAMCH.  This subroutine is needed because */
/*  SLAMCH does not compensate for poor arithmetic in the upper half of */
/*  the exponent range, as is found on a Cray. */

/*  Arguments */
/*  ========= */

/*  SMALL   (input/output) REAL */
/*          On entry, the underflow threshold as computed by SLAMCH. */
/*          On exit, if LOG10(LARGE) is sufficiently large, the square */
/*          root of SMALL, otherwise unchanged. */

/*  LARGE   (input/output) REAL */
/*          On entry, the overflow threshold as computed by SLAMCH. */
/*          On exit, if LOG10(LARGE) is sufficiently large, the square */
/*          root of LARGE, otherwise unchanged. */

/*  ===================================================================== */

/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     If it looks like we're on a Cray, take the square root of */
/*     SMALL and LARGE to avoid overflow and underflow problems. */

    if (r_lg10(large) > 2e3f) {
	*small = sqrt(*small);
	*large = sqrt(*large);
    }

    return 0;

/*     End of SLABAD */

} /* slabad_ */
