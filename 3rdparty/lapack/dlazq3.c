#include "clapack.h"

/* Subroutine */ int dlazq3_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, doublereal *dmin__, doublereal *sigma, doublereal *desig, 
	 doublereal *qmax, integer *nfail, integer *iter, integer *ndiv, 
	logical *ieee, integer *ttype, doublereal *dmin1, doublereal *dmin2, 
	doublereal *dn, doublereal *dn1, doublereal *dn2, doublereal *tau)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    doublereal g, s, t;
    integer j4, nn;
    doublereal eps, tol;
    integer n0in, ipn4;
    doublereal tol2, temp;
    extern /* Subroutine */ int dlasq5_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, doublereal *, doublereal *, logical *), dlasq6_(
	    integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *), dlazq4_(integer *, integer *, doublereal *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *, 
	     doublereal *);
    extern doublereal dlamch_(char *);
    doublereal safmin;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAZQ3 checks for deflation, computes a shift (TAU) and calls dqds. */
/*  In case of failure it changes shifts, and tries again until output */
/*  is positive. */

/*  Arguments */
/*  ========= */

/*  I0     (input) INTEGER */
/*         First index. */

/*  N0     (input) INTEGER */
/*         Last index. */

/*  Z      (input) DOUBLE PRECISION array, dimension ( 4*N ) */
/*         Z holds the qd array. */

/*  PP     (input) INTEGER */
/*         PP=0 for ping, PP=1 for pong. */

/*  DMIN   (output) DOUBLE PRECISION */
/*         Minimum value of d. */

/*  SIGMA  (output) DOUBLE PRECISION */
/*         Sum of shifts used in current segment. */

/*  DESIG  (input/output) DOUBLE PRECISION */
/*         Lower order part of SIGMA */

/*  QMAX   (input) DOUBLE PRECISION */
/*         Maximum value of q. */

/*  NFAIL  (output) INTEGER */
/*         Number of times shift was too big. */

/*  ITER   (output) INTEGER */
/*         Number of iterations. */

/*  NDIV   (output) INTEGER */
/*         Number of divisions. */

/*  IEEE   (input) LOGICAL */
/*         Flag for IEEE or non IEEE arithmetic (passed to DLASQ5). */

/*  TTYPE  (input/output) INTEGER */
/*         Shift type.  TTYPE is passed as an argument in order to save */
/*         its value between calls to DLAZQ3 */

/*  DMIN1  (input/output) REAL */
/*  DMIN2  (input/output) REAL */
/*  DN     (input/output) REAL */
/*  DN1    (input/output) REAL */
/*  DN2    (input/output) REAL */
/*  TAU    (input/output) REAL */
/*         These are passed as arguments in order to save their values */
/*         between calls to DLAZQ3 */

/*  This is a thread safe version of DLASQ3, which passes TTYPE, DMIN1, */
/*  DMIN2, DN, DN1. DN2 and TAU through the argument list in place of */
/*  declaring them in a SAVE statment. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Function .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --z__;

    /* Function Body */
    n0in = *n0;
    eps = dlamch_("Precision");
    safmin = dlamch_("Safe minimum");
    tol = eps * 100.;
/* Computing 2nd power */
    d__1 = tol;
    tol2 = d__1 * d__1;
    g = 0.;

/*     Check for deflation. */

L10:

    if (*n0 < *i0) {
	return 0;
    }
    if (*n0 == *i0) {
	goto L20;
    }
    nn = (*n0 << 2) + *pp;
    if (*n0 == *i0 + 1) {
	goto L40;
    }

/*     Check whether E(N0-1) is negligible, 1 eigenvalue. */

    if (z__[nn - 5] > tol2 * (*sigma + z__[nn - 3]) && z__[nn - (*pp << 1) - 
	    4] > tol2 * z__[nn - 7]) {
	goto L30;
    }

L20:

    z__[(*n0 << 2) - 3] = z__[(*n0 << 2) + *pp - 3] + *sigma;
    --(*n0);
    goto L10;

/*     Check  whether E(N0-2) is negligible, 2 eigenvalues. */

L30:

    if (z__[nn - 9] > tol2 * *sigma && z__[nn - (*pp << 1) - 8] > tol2 * z__[
	    nn - 11]) {
	goto L50;
    }

L40:

    if (z__[nn - 3] > z__[nn - 7]) {
	s = z__[nn - 3];
	z__[nn - 3] = z__[nn - 7];
	z__[nn - 7] = s;
    }
    if (z__[nn - 5] > z__[nn - 3] * tol2) {
	t = (z__[nn - 7] - z__[nn - 3] + z__[nn - 5]) * .5;
	s = z__[nn - 3] * (z__[nn - 5] / t);
	if (s <= t) {
	    s = z__[nn - 3] * (z__[nn - 5] / (t * (sqrt(s / t + 1.) + 1.)));
	} else {
	    s = z__[nn - 3] * (z__[nn - 5] / (t + sqrt(t) * sqrt(t + s)));
	}
	t = z__[nn - 7] + (s + z__[nn - 5]);
	z__[nn - 3] *= z__[nn - 7] / t;
	z__[nn - 7] = t;
    }
    z__[(*n0 << 2) - 7] = z__[nn - 7] + *sigma;
    z__[(*n0 << 2) - 3] = z__[nn - 3] + *sigma;
    *n0 += -2;
    goto L10;

L50:

/*     Reverse the qd-array, if warranted. */

    if (*dmin__ <= 0. || *n0 < n0in) {
	if (z__[(*i0 << 2) + *pp - 3] * 1.5 < z__[(*n0 << 2) + *pp - 3]) {
	    ipn4 = *i0 + *n0 << 2;
	    i__1 = *i0 + *n0 - 1 << 1;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		temp = z__[j4 - 3];
		z__[j4 - 3] = z__[ipn4 - j4 - 3];
		z__[ipn4 - j4 - 3] = temp;
		temp = z__[j4 - 2];
		z__[j4 - 2] = z__[ipn4 - j4 - 2];
		z__[ipn4 - j4 - 2] = temp;
		temp = z__[j4 - 1];
		z__[j4 - 1] = z__[ipn4 - j4 - 5];
		z__[ipn4 - j4 - 5] = temp;
		temp = z__[j4];
		z__[j4] = z__[ipn4 - j4 - 4];
		z__[ipn4 - j4 - 4] = temp;
/* L60: */
	    }
	    if (*n0 - *i0 <= 4) {
		z__[(*n0 << 2) + *pp - 1] = z__[(*i0 << 2) + *pp - 1];
		z__[(*n0 << 2) - *pp] = z__[(*i0 << 2) - *pp];
	    }
/* Computing MIN */
	    d__1 = *dmin2, d__2 = z__[(*n0 << 2) + *pp - 1];
	    *dmin2 = min(d__1,d__2);
/* Computing MIN */
	    d__1 = z__[(*n0 << 2) + *pp - 1], d__2 = z__[(*i0 << 2) + *pp - 1]
		    , d__1 = min(d__1,d__2), d__2 = z__[(*i0 << 2) + *pp + 3];
	    z__[(*n0 << 2) + *pp - 1] = min(d__1,d__2);
/* Computing MIN */
	    d__1 = z__[(*n0 << 2) - *pp], d__2 = z__[(*i0 << 2) - *pp], d__1 =
		     min(d__1,d__2), d__2 = z__[(*i0 << 2) - *pp + 4];
	    z__[(*n0 << 2) - *pp] = min(d__1,d__2);
/* Computing MAX */
	    d__1 = *qmax, d__2 = z__[(*i0 << 2) + *pp - 3], d__1 = max(d__1,
		    d__2), d__2 = z__[(*i0 << 2) + *pp + 1];
	    *qmax = max(d__1,d__2);
	    *dmin__ = -0.;
	}
    }

/* Computing MIN */
    d__1 = z__[(*n0 << 2) + *pp - 1], d__2 = z__[(*n0 << 2) + *pp - 9], d__1 =
	     min(d__1,d__2), d__2 = *dmin2 + z__[(*n0 << 2) - *pp];
    if (*dmin__ < 0. || safmin * *qmax < min(d__1,d__2)) {

/*        Choose a shift. */

	dlazq4_(i0, n0, &z__[1], pp, &n0in, dmin__, dmin1, dmin2, dn, dn1, 
		dn2, tau, ttype, &g);

/*        Call dqds until DMIN > 0. */

L80:

	dlasq5_(i0, n0, &z__[1], pp, tau, dmin__, dmin1, dmin2, dn, dn1, dn2, 
		ieee);

	*ndiv += *n0 - *i0 + 2;
	++(*iter);

/*        Check status. */

	if (*dmin__ >= 0. && *dmin1 > 0.) {

/*           Success. */

	    goto L100;

	} else if (*dmin__ < 0. && *dmin1 > 0. && z__[(*n0 - 1 << 2) - *pp] < 
		tol * (*sigma + *dn1) && abs(*dn) < tol * *sigma) {

/*           Convergence hidden by negative DN. */

	    z__[(*n0 - 1 << 2) - *pp + 2] = 0.;
	    *dmin__ = 0.;
	    goto L100;
	} else if (*dmin__ < 0.) {

/*           TAU too big. Select new TAU and try again. */

	    ++(*nfail);
	    if (*ttype < -22) {

/*              Failed twice. Play it safe. */

		*tau = 0.;
	    } else if (*dmin1 > 0.) {

/*              Late failure. Gives excellent shift. */

		*tau = (*tau + *dmin__) * (1. - eps * 2.);
		*ttype += -11;
	    } else {

/*              Early failure. Divide by 4. */

		*tau *= .25;
		*ttype += -12;
	    }
	    goto L80;
	} else if (*dmin__ != *dmin__) {

/*           NaN. */

	    *tau = 0.;
	    goto L80;
	} else {

/*           Possible underflow. Play it safe. */

	    goto L90;
	}
    }

/*     Risk of underflow. */

L90:
    dlasq6_(i0, n0, &z__[1], pp, dmin__, dmin1, dmin2, dn, dn1, dn2);
    *ndiv += *n0 - *i0 + 2;
    ++(*iter);
    *tau = 0.;

L100:
    if (*tau < *sigma) {
	*desig += *tau;
	t = *sigma + *desig;
	*desig -= t - *sigma;
    } else {
	t = *sigma + *tau;
	*desig = *sigma - (t - *tau) + *desig;
    }
    *sigma = t;

    return 0;

/*     End of DLAZQ3 */

} /* dlazq3_ */
