#include "clapack.h"

integer dlaneg_(integer *n, doublereal *d__, doublereal *lld, doublereal *
	sigma, doublereal *pivmin, integer *r__)
{
    /* System generated locals */
    integer ret_val, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer j;
    doublereal p, t;
    integer bj;
    doublereal tmp;
    integer neg1, neg2;
    doublereal bsav, gamma, dplus;
    extern logical disnan_(doublereal *);
    integer negcnt;
    logical sawnan;
    doublereal dminus;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLANEG computes the Sturm count, the number of negative pivots */
/*  encountered while factoring tridiagonal T - sigma I = L D L^T. */
/*  This implementation works directly on the factors without forming */
/*  the tridiagonal matrix T.  The Sturm count is also the number of */
/*  eigenvalues of T less than sigma. */

/*  This routine is called from DLARRB. */

/*  The current routine does not use the PIVMIN parameter but rather */
/*  requires IEEE-754 propagation of Infinities and NaNs.  This */
/*  routine also has no input range restrictions but does require */
/*  default exception handling such that x/0 produces Inf when x is */
/*  non-zero, and Inf/Inf produces NaN.  For more information, see: */

/*    Marques, Riedy, and Voemel, "Benefits of IEEE-754 Features in */
/*    Modern Symmetric Tridiagonal Eigensolvers," SIAM Journal on */
/*    Scientific Computing, v28, n5, 2006.  DOI 10.1137/050641624 */
/*    (Tech report version in LAWN 172 with the same title.) */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix. */

/*  D       (input) DOUBLE PRECISION array, dimension (N) */
/*          The N diagonal elements of the diagonal matrix D. */

/*  LLD     (input) DOUBLE PRECISION array, dimension (N-1) */
/*          The (N-1) elements L(i)*L(i)*D(i). */

/*  SIGMA   (input) DOUBLE PRECISION */
/*          Shift amount in T - sigma I = L D L^T. */

/*  PIVMIN  (input) DOUBLE PRECISION */
/*          The minimum pivot in the Sturm sequence.  May be used */
/*          when zero pivots are encountered on non-IEEE-754 */
/*          architectures. */

/*  R       (input) INTEGER */
/*          The twist index for the twisted factorization that is used */
/*          for the negcount. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Christof Voemel, University of California, Berkeley, USA */
/*     Jason Riedy, University of California, Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     Some architectures propagate Infinities and NaNs very slowly, so */
/*     the code computes counts in BLKLEN chunks.  Then a NaN can */
/*     propagate at most BLKLEN columns before being detected.  This is */
/*     not a general tuning parameter; it needs only to be just large */
/*     enough that the overhead is tiny in common cases. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */
    /* Parameter adjustments */
    --lld;
    --d__;

    /* Function Body */
    negcnt = 0;
/*     I) upper part: L D L^T - SIGMA I = L+ D+ L+^T */
    t = -(*sigma);
    i__1 = *r__ - 1;
    for (bj = 1; bj <= i__1; bj += 128) {
	neg1 = 0;
	bsav = t;
/* Computing MIN */
	i__3 = bj + 127, i__4 = *r__ - 1;
	i__2 = min(i__3,i__4);
	for (j = bj; j <= i__2; ++j) {
	    dplus = d__[j] + t;
	    if (dplus < 0.) {
		++neg1;
	    }
	    tmp = t / dplus;
	    t = tmp * lld[j] - *sigma;
/* L21: */
	}
	sawnan = disnan_(&t);
/*     Run a slower version of the above loop if a NaN is detected. */
/*     A NaN should occur only with a zero pivot after an infinite */
/*     pivot.  In that case, substituting 1 for T/DPLUS is the */
/*     correct limit. */
	if (sawnan) {
	    neg1 = 0;
	    t = bsav;
/* Computing MIN */
	    i__3 = bj + 127, i__4 = *r__ - 1;
	    i__2 = min(i__3,i__4);
	    for (j = bj; j <= i__2; ++j) {
		dplus = d__[j] + t;
		if (dplus < 0.) {
		    ++neg1;
		}
		tmp = t / dplus;
		if (disnan_(&tmp)) {
		    tmp = 1.;
		}
		t = tmp * lld[j] - *sigma;
/* L22: */
	    }
	}
	negcnt += neg1;
/* L210: */
    }

/*     II) lower part: L D L^T - SIGMA I = U- D- U-^T */
    p = d__[*n] - *sigma;
    i__1 = *r__;
    for (bj = *n - 1; bj >= i__1; bj += -128) {
	neg2 = 0;
	bsav = p;
/* Computing MAX */
	i__3 = bj - 127;
	i__2 = max(i__3,*r__);
	for (j = bj; j >= i__2; --j) {
	    dminus = lld[j] + p;
	    if (dminus < 0.) {
		++neg2;
	    }
	    tmp = p / dminus;
	    p = tmp * d__[j] - *sigma;
/* L23: */
	}
	sawnan = disnan_(&p);
/*     As above, run a slower version that substitutes 1 for Inf/Inf. */

	if (sawnan) {
	    neg2 = 0;
	    p = bsav;
/* Computing MAX */
	    i__3 = bj - 127;
	    i__2 = max(i__3,*r__);
	    for (j = bj; j >= i__2; --j) {
		dminus = lld[j] + p;
		if (dminus < 0.) {
		    ++neg2;
		}
		tmp = p / dminus;
		if (disnan_(&tmp)) {
		    tmp = 1.;
		}
		p = tmp * d__[j] - *sigma;
/* L24: */
	    }
	}
	negcnt += neg2;
/* L230: */
    }

/*     III) Twist index */
/*       T was shifted by SIGMA initially. */
    gamma = t + *sigma + p;
    if (gamma < 0.) {
	++negcnt;
    }
    ret_val = negcnt;
    return ret_val;
} /* dlaneg_ */
