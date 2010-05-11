#include "clapack.h"

/* Subroutine */ int slarrb_(integer *n, real *d__, real *lld, integer *
	ifirst, integer *ilast, real *rtol1, real *rtol2, integer *offset, 
	real *w, real *wgap, real *werr, real *work, integer *iwork, real *
	pivmin, real *spdiam, integer *twist, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer i__, k, r__, i1, ii, ip;
    real gap, mid, tmp, back, lgap, rgap, left;
    integer iter, nint, prev, next;
    real cvrgd, right, width;
    extern integer slaneg_(integer *, real *, real *, real *, real *, integer 
	    *);
    integer negcnt;
    real mnwdth;
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

/*  Given the relatively robust representation(RRR) L D L^T, SLARRB */
/*  does "limited" bisection to refine the eigenvalues of L D L^T, */
/*  W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial */
/*  guesses for these eigenvalues are input in W, the corresponding estimate */
/*  of the error in these guesses and their gaps are input in WERR */
/*  and WGAP, respectively. During bisection, intervals */
/*  [left, right] are maintained by storing their mid-points and */
/*  semi-widths in the arrays W and WERR respectively. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix. */

/*  D       (input) REAL             array, dimension (N) */
/*          The N diagonal elements of the diagonal matrix D. */

/*  LLD     (input) REAL             array, dimension (N-1) */
/*          The (N-1) elements L(i)*L(i)*D(i). */

/*  IFIRST  (input) INTEGER */
/*          The index of the first eigenvalue to be computed. */

/*  ILAST   (input) INTEGER */
/*          The index of the last eigenvalue to be computed. */

/*  RTOL1   (input) REAL */
/*  RTOL2   (input) REAL */
/*          Tolerance for the convergence of the bisection intervals. */
/*          An interval [LEFT,RIGHT] has converged if */
/*          RIGHT-LEFT.LT.MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) ) */
/*          where GAP is the (estimated) distance to the nearest */
/*          eigenvalue. */

/*  OFFSET  (input) INTEGER */
/*          Offset for the arrays W, WGAP and WERR, i.e., the IFIRST-OFFSET */
/*          through ILAST-OFFSET elements of these arrays are to be used. */

/*  W       (input/output) REAL             array, dimension (N) */
/*          On input, W( IFIRST-OFFSET ) through W( ILAST-OFFSET ) are */
/*          estimates of the eigenvalues of L D L^T indexed IFIRST throug */
/*          ILAST. */
/*          On output, these estimates are refined. */

/*  WGAP    (input/output) REAL             array, dimension (N-1) */
/*          On input, the (estimated) gaps between consecutive */
/*          eigenvalues of L D L^T, i.e., WGAP(I-OFFSET) is the gap between */
/*          eigenvalues I and I+1. Note that if IFIRST.EQ.ILAST */
/*          then WGAP(IFIRST-OFFSET) must be set to ZERO. */
/*          On output, these gaps are refined. */

/*  WERR    (input/output) REAL             array, dimension (N) */
/*          On input, WERR( IFIRST-OFFSET ) through WERR( ILAST-OFFSET ) are */
/*          the errors in the estimates of the corresponding elements in W. */
/*          On output, these errors are refined. */

/*  WORK    (workspace) REAL             array, dimension (2*N) */
/*          Workspace. */

/*  IWORK   (workspace) INTEGER array, dimension (2*N) */
/*          Workspace. */

/*  PIVMIN  (input) DOUBLE PRECISION */
/*          The minimum pivot in the Sturm sequence. */

/*  SPDIAM  (input) DOUBLE PRECISION */
/*          The spectral diameter of the matrix. */

/*  TWIST   (input) INTEGER */
/*          The twist index for the twisted factorization that is used */
/*          for the negcount. */
/*          TWIST = N: Compute negcount from L D L^T - LAMBDA I = L+ D+ L+^T */
/*          TWIST = 1: Compute negcount from L D L^T - LAMBDA I = U- D- U-^T */
/*          TWIST = R: Compute negcount from L D L^T - LAMBDA I = N(r) D(r) N(r) */

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
/*     .. External Functions .. */

/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --iwork;
    --work;
    --werr;
    --wgap;
    --w;
    --lld;
    --d__;

    /* Function Body */
    *info = 0;

    maxitr = (integer) ((log(*spdiam + *pivmin) - log(*pivmin)) / log(2.f)) + 
	    2;
    mnwdth = *pivmin * 2.f;

    r__ = *twist;
    if (r__ < 1 || r__ > *n) {
	r__ = *n;
    }

/*     Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ]. */
/*     The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while */
/*     Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 ) */
/*     for an unconverged interval is set to the index of the next unconverged */
/*     interval, and is -1 or 0 for a converged interval. Thus a linked */
/*     list of unconverged intervals is set up. */

    i1 = *ifirst;
/*     The number of unconverged intervals */
    nint = 0;
/*     The last unconverged interval found */
    prev = 0;
    rgap = wgap[i1 - *offset];
    i__1 = *ilast;
    for (i__ = i1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	left = w[ii] - werr[ii];
	right = w[ii] + werr[ii];
	lgap = rgap;
	rgap = wgap[ii];
	gap = dmin(lgap,rgap);
/*        Make sure that [LEFT,RIGHT] contains the desired eigenvalue */
/*        Compute negcount from dstqds facto L+D+L+^T = L D L^T - LEFT */

/*        Do while( NEGCNT(LEFT).GT.I-1 ) */

	back = werr[ii];
L20:
	negcnt = slaneg_(n, &d__[1], &lld[1], &left, pivmin, &r__);
	if (negcnt > i__ - 1) {
	    left -= back;
	    back *= 2.f;
	    goto L20;
	}

/*        Do while( NEGCNT(RIGHT).LT.I ) */
/*        Compute negcount from dstqds facto L+D+L+^T = L D L^T - RIGHT */

	back = werr[ii];
L50:
	negcnt = slaneg_(n, &d__[1], &lld[1], &right, pivmin, &r__);
	if (negcnt < i__) {
	    right += back;
	    back *= 2.f;
	    goto L50;
	}
	width = (r__1 = left - right, dabs(r__1)) * .5f;
/* Computing MAX */
	r__1 = dabs(left), r__2 = dabs(right);
	tmp = dmax(r__1,r__2);
/* Computing MAX */
	r__1 = *rtol1 * gap, r__2 = *rtol2 * tmp;
	cvrgd = dmax(r__1,r__2);
	if (width <= cvrgd || width <= mnwdth) {
/*           This interval has already converged and does not need refinement. */
/*           (Note that the gaps might change through refining the */
/*            eigenvalues, however, they can only get bigger.) */
/*           Remove it from the list. */
	    iwork[k - 1] = -1;
/*           Make sure that I1 always points to the first unconverged interval */
	    if (i__ == i1 && i__ < *ilast) {
		i1 = i__ + 1;
	    }
	    if (prev >= i1 && i__ <= *ilast) {
		iwork[(prev << 1) - 1] = i__ + 1;
	    }
	} else {
/*           unconverged interval found */
	    prev = i__;
	    ++nint;
	    iwork[k - 1] = i__ + 1;
	    iwork[k] = negcnt;
	}
	work[k - 1] = left;
	work[k] = right;
/* L75: */
    }

/*     Do while( NINT.GT.0 ), i.e. there are still unconverged intervals */
/*     and while (ITER.LT.MAXITR) */

    iter = 0;
L80:
    prev = i1 - 1;
    i__ = i1;
    olnint = nint;
    i__1 = olnint;
    for (ip = 1; ip <= i__1; ++ip) {
	k = i__ << 1;
	ii = i__ - *offset;
	rgap = wgap[ii];
	lgap = rgap;
	if (ii > 1) {
	    lgap = wgap[ii - 1];
	}
	gap = dmin(lgap,rgap);
	next = iwork[k - 1];
	left = work[k - 1];
	right = work[k];
	mid = (left + right) * .5f;
/*        semiwidth of interval */
	width = right - mid;
/* Computing MAX */
	r__1 = dabs(left), r__2 = dabs(right);
	tmp = dmax(r__1,r__2);
/* Computing MAX */
	r__1 = *rtol1 * gap, r__2 = *rtol2 * tmp;
	cvrgd = dmax(r__1,r__2);
	if (width <= cvrgd || width <= mnwdth || iter == maxitr) {
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

	negcnt = slaneg_(n, &d__[1], &lld[1], &mid, pivmin, &r__);
	if (negcnt <= i__ - 1) {
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
    for (i__ = *ifirst; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
/*        All intervals marked by '0' have been refined. */
	if (iwork[k - 1] == 0) {
	    w[ii] = (work[k - 1] + work[k]) * .5f;
	    werr[ii] = work[k] - w[ii];
	}
/* L110: */
    }

    i__1 = *ilast;
    for (i__ = *ifirst + 1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
/* Computing MAX */
	r__1 = 0.f, r__2 = w[ii] - werr[ii] - w[ii - 1] - werr[ii - 1];
	wgap[ii - 1] = dmax(r__1,r__2);
/* L111: */
    }
    return 0;

/*     End of SLARRB */

} /* slarrb_ */
