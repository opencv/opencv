#include "clapack.h"

/* Subroutine */ int slarrj_(integer *n, real *d__, real *e2, integer *ifirst, 
	 integer *ilast, real *rtol, integer *offset, real *w, real *werr, 
	real *work, integer *iwork, real *pivmin, real *spdiam, integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer i__, j, k, p;
    real s;
    integer i1, i2, ii;
    real fac, mid;
    integer cnt;
    real tmp, left;
    integer iter, nint, prev, next, savi1;
    real right, width, dplus;
    integer olnint, maxitr;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Given the initial eigenvalue approximations of T, SLARRJ */
/*  does  bisection to refine the eigenvalues of T, */
/*  W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial */
/*  guesses for these eigenvalues are input in W, the corresponding estimate */
/*  of the error in these guesses in WERR. During bisection, intervals */
/*  [left, right] are maintained by storing their mid-points and */
/*  semi-widths in the arrays W and WERR respectively. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix. */

/*  D       (input) REAL             array, dimension (N) */
/*          The N diagonal elements of T. */

/*  E2      (input) REAL             array, dimension (N-1) */
/*          The Squares of the (N-1) subdiagonal elements of T. */

/*  IFIRST  (input) INTEGER */
/*          The index of the first eigenvalue to be computed. */

/*  ILAST   (input) INTEGER */
/*          The index of the last eigenvalue to be computed. */

/*  RTOL   (input) REAL */
/*          Tolerance for the convergence of the bisection intervals. */
/*          An interval [LEFT,RIGHT] has converged if */
/*          RIGHT-LEFT.LT.RTOL*MAX(|LEFT|,|RIGHT|). */

/*  OFFSET  (input) INTEGER */
/*          Offset for the arrays W and WERR, i.e., the IFIRST-OFFSET */
/*          through ILAST-OFFSET elements of these arrays are to be used. */

/*  W       (input/output) REAL             array, dimension (N) */
/*          On input, W( IFIRST-OFFSET ) through W( ILAST-OFFSET ) are */
/*          estimates of the eigenvalues of L D L^T indexed IFIRST through */
/*          ILAST. */
/*          On output, these estimates are refined. */

/*  WERR    (input/output) REAL             array, dimension (N) */
/*          On input, WERR( IFIRST-OFFSET ) through WERR( ILAST-OFFSET ) are */
/*          the errors in the estimates of the corresponding elements in W. */
/*          On output, these errors are refined. */

/*  WORK    (workspace) REAL             array, dimension (2*N) */
/*          Workspace. */

/*  IWORK   (workspace) INTEGER array, dimension (2*N) */
/*          Workspace. */

/*  PIVMIN  (input) DOUBLE PRECISION */
/*          The minimum pivot in the Sturm sequence for T. */

/*  SPDIAM  (input) DOUBLE PRECISION */
/*          The spectral diameter of T. */

/*  INFO    (output) INTEGER */
/*          Error flag. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Beresford Parlett, University of California, Berkeley, USA */
/*     Jim Demmel, University of California, Berkeley, USA */
/*     Inderjit Dhillon, University of Texas, Austin, USA */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Christof Voemel, University of California, Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */

/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --iwork;
    --work;
    --werr;
    --w;
    --e2;
    --d__;

    /* Function Body */
    *info = 0;

    maxitr = (integer) ((log(*spdiam + *pivmin) - log(*pivmin)) / log(2.f)) + 
	    2;

/*     Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ]. */
/*     The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while */
/*     Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 ) */
/*     for an unconverged interval is set to the index of the next unconverged */
/*     interval, and is -1 or 0 for a converged interval. Thus a linked */
/*     list of unconverged intervals is set up. */

    i1 = *ifirst;
    i2 = *ilast;
/*     The number of unconverged intervals */
    nint = 0;
/*     The last unconverged interval found */
    prev = 0;
    i__1 = i2;
    for (i__ = i1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	left = w[ii] - werr[ii];
	mid = w[ii];
	right = w[ii] + werr[ii];
	width = right - mid;
/* Computing MAX */
	r__1 = dabs(left), r__2 = dabs(right);
	tmp = dmax(r__1,r__2);
/*        The following test prevents the test of converged intervals */
	if (width < *rtol * tmp) {
/*           This interval has already converged and does not need refinement. */
/*           (Note that the gaps might change through refining the */
/*            eigenvalues, however, they can only get bigger.) */
/*           Remove it from the list. */
	    iwork[k - 1] = -1;
/*           Make sure that I1 always points to the first unconverged interval */
	    if (i__ == i1 && i__ < i2) {
		i1 = i__ + 1;
	    }
	    if (prev >= i1 && i__ <= i2) {
		iwork[(prev << 1) - 1] = i__ + 1;
	    }
	} else {
/*           unconverged interval found */
	    prev = i__;
/*           Make sure that [LEFT,RIGHT] contains the desired eigenvalue */

/*           Do while( CNT(LEFT).GT.I-1 ) */

	    fac = 1.f;
L20:
	    cnt = 0;
	    s = left;
	    dplus = d__[1] - s;
	    if (dplus < 0.f) {
		++cnt;
	    }
	    i__2 = *n;
	    for (j = 2; j <= i__2; ++j) {
		dplus = d__[j] - s - e2[j - 1] / dplus;
		if (dplus < 0.f) {
		    ++cnt;
		}
/* L30: */
	    }
	    if (cnt > i__ - 1) {
		left -= werr[ii] * fac;
		fac *= 2.f;
		goto L20;
	    }

/*           Do while( CNT(RIGHT).LT.I ) */

	    fac = 1.f;
L50:
	    cnt = 0;
	    s = right;
	    dplus = d__[1] - s;
	    if (dplus < 0.f) {
		++cnt;
	    }
	    i__2 = *n;
	    for (j = 2; j <= i__2; ++j) {
		dplus = d__[j] - s - e2[j - 1] / dplus;
		if (dplus < 0.f) {
		    ++cnt;
		}
/* L60: */
	    }
	    if (cnt < i__) {
		right += werr[ii] * fac;
		fac *= 2.f;
		goto L50;
	    }
	    ++nint;
	    iwork[k - 1] = i__ + 1;
	    iwork[k] = cnt;
	}
	work[k - 1] = left;
	work[k] = right;
/* L75: */
    }
    savi1 = i1;

/*     Do while( NINT.GT.0 ), i.e. there are still unconverged intervals */
/*     and while (ITER.LT.MAXITR) */

    iter = 0;
L80:
    prev = i1 - 1;
    i__ = i1;
    olnint = nint;
    i__1 = olnint;
    for (p = 1; p <= i__1; ++p) {
	k = i__ << 1;
	ii = i__ - *offset;
	next = iwork[k - 1];
	left = work[k - 1];
	right = work[k];
	mid = (left + right) * .5f;
/*        semiwidth of interval */
	width = right - mid;
/* Computing MAX */
	r__1 = dabs(left), r__2 = dabs(right);
	tmp = dmax(r__1,r__2);
	if (width < *rtol * tmp || iter == maxitr) {
/*           reduce number of unconverged intervals */
	    --nint;
/*           Mark interval as converged. */
	    iwork[k - 1] = 0;
	    if (i1 == i__) {
		i1 = next;
	    } else {
/*              Prev holds the last unconverged interval previously examined */
		if (prev >= i1) {
		    iwork[(prev << 1) - 1] = next;
		}
	    }
	    i__ = next;
	    goto L100;
	}
	prev = i__;

/*        Perform one bisection step */

	cnt = 0;
	s = mid;
	dplus = d__[1] - s;
	if (dplus < 0.f) {
	    ++cnt;
	}
	i__2 = *n;
	for (j = 2; j <= i__2; ++j) {
	    dplus = d__[j] - s - e2[j - 1] / dplus;
	    if (dplus < 0.f) {
		++cnt;
	    }
/* L90: */
	}
	if (cnt <= i__ - 1) {
	    work[k - 1] = mid;
	} else {
	    work[k] = mid;
	}
	i__ = next;
L100:
	;
    }
    ++iter;
/*     do another loop if there are still unconverged intervals */
/*     However, in the last iteration, all intervals are accepted */
/*     since this is the best we can do. */
    if (nint > 0 && iter <= maxitr) {
	goto L80;
    }


/*     At this point, all the intervals have converged */
    i__1 = *ilast;
    for (i__ = savi1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
/*        All intervals marked by '0' have been refined. */
	if (iwork[k - 1] == 0) {
	    w[ii] = (work[k - 1] + work[k]) * .5f;
	    werr[ii] = work[k] - w[ii];
	}
/* L110: */
    }

    return 0;

/*     End of SLARRJ */

} /* slarrj_ */
