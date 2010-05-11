#include "clapack.h"

/* Subroutine */ int dlarrk_(integer *n, integer *iw, doublereal *gl, 
	doublereal *gu, doublereal *d__, doublereal *e2, doublereal *pivmin, 
	doublereal *reltol, doublereal *w, doublereal *werr, integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer i__, it;
    doublereal mid, eps, tmp1, tmp2, left, atoli, right;
    integer itmax;
    doublereal rtoli, tnorm;
    extern doublereal dlamch_(char *);
    integer negcnt;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLARRK computes one eigenvalue of a symmetric tridiagonal */
/*  matrix T to suitable accuracy. This is an auxiliary code to be */
/*  called from DSTEMR. */

/*  To avoid overflow, the matrix must be scaled so that its */
/*  largest element is no greater than overflow**(1/2) * */
/*  underflow**(1/4) in absolute value, and for greatest */
/*  accuracy, it should not be much smaller than that. */

/*  See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal */
/*  Matrix", Report CS41, Computer Science Dept., Stanford */
/*  University, July 21, 1966. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the tridiagonal matrix T.  N >= 0. */

/*  IW      (input) INTEGER */
/*          The index of the eigenvalues to be returned. */

/*  GL      (input) DOUBLE PRECISION */
/*  GU      (input) DOUBLE PRECISION */
/*          An upper and a lower bound on the eigenvalue. */

/*  D       (input) DOUBLE PRECISION array, dimension (N) */
/*          The n diagonal elements of the tridiagonal matrix T. */

/*  E2      (input) DOUBLE PRECISION array, dimension (N-1) */
/*          The (n-1) squared off-diagonal elements of the tridiagonal matrix T. */

/*  PIVMIN  (input) DOUBLE PRECISION */
/*          The minimum pivot allowed in the Sturm sequence for T. */

/*  RELTOL  (input) DOUBLE PRECISION */
/*          The minimum relative width of an interval.  When an interval */
/*          is narrower than RELTOL times the larger (in */
/*          magnitude) endpoint, then it is considered to be */
/*          sufficiently small, i.e., converged.  Note: this should */
/*          always be at least radix*machine epsilon. */

/*  W       (output) DOUBLE PRECISION */

/*  WERR    (output) DOUBLE PRECISION */
/*          The error bound on the corresponding eigenvalue approximation */
/*          in W. */

/*  INFO    (output) INTEGER */
/*          = 0:       Eigenvalue converged */
/*          = -1:      Eigenvalue did NOT converge */

/*  Internal Parameters */
/*  =================== */

/*  FUDGE   DOUBLE PRECISION, default = 2 */
/*          A "fudge factor" to widen the Gershgorin intervals. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Get machine constants */
    /* Parameter adjustments */
    --e2;
    --d__;

    /* Function Body */
    eps = dlamch_("P");
/* Computing MAX */
    d__1 = abs(*gl), d__2 = abs(*gu);
    tnorm = max(d__1,d__2);
    rtoli = *reltol;
    atoli = *pivmin * 4.;
    itmax = (integer) ((log(tnorm + *pivmin) - log(*pivmin)) / log(2.)) + 2;
    *info = -1;
    left = *gl - tnorm * 2. * eps * *n - *pivmin * 4.;
    right = *gu + tnorm * 2. * eps * *n + *pivmin * 4.;
    it = 0;
L10:

/*     Check if interval converged or maximum number of iterations reached */

    tmp1 = (d__1 = right - left, abs(d__1));
/* Computing MAX */
    d__1 = abs(right), d__2 = abs(left);
    tmp2 = max(d__1,d__2);
/* Computing MAX */
    d__1 = max(atoli,*pivmin), d__2 = rtoli * tmp2;
    if (tmp1 < max(d__1,d__2)) {
	*info = 0;
	goto L30;
    }
    if (it > itmax) {
	goto L30;
    }

/*     Count number of negative pivots for mid-point */

    ++it;
    mid = (left + right) * .5;
    negcnt = 0;
    tmp1 = d__[1] - mid;
    if (abs(tmp1) < *pivmin) {
	tmp1 = -(*pivmin);
    }
    if (tmp1 <= 0.) {
	++negcnt;
    }

    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	tmp1 = d__[i__] - e2[i__ - 1] / tmp1 - mid;
	if (abs(tmp1) < *pivmin) {
	    tmp1 = -(*pivmin);
	}
	if (tmp1 <= 0.) {
	    ++negcnt;
	}
/* L20: */
    }
    if (negcnt >= *iw) {
	right = mid;
    } else {
	left = mid;
    }
    goto L10;
L30:

/*     Converged or maximum number of iterations reached */

    *w = (left + right) * .5;
    *werr = (d__1 = right - left, abs(d__1)) * .5;
    return 0;

/*     End of DLARRK */

} /* dlarrk_ */
