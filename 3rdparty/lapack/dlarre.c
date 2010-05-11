#include "clapack.h"

/* Table of constant values */

static integer c__1 = 1;
static integer c__2 = 2;

/* Subroutine */ int dlarre_(char *range, integer *n, doublereal *vl, 
	doublereal *vu, integer *il, integer *iu, doublereal *d__, doublereal 
	*e, doublereal *e2, doublereal *rtol1, doublereal *rtol2, doublereal *
	spltol, integer *nsplit, integer *isplit, integer *m, doublereal *w, 
	doublereal *werr, doublereal *wgap, integer *iblock, integer *indexw, 
	doublereal *gers, doublereal *pivmin, doublereal *work, integer *
	iwork, integer *info)
{
    /* System generated locals */
    integer i__1, i__2;
    doublereal d__1, d__2, d__3;

    /* Builtin functions */
    double sqrt(doublereal), log(doublereal);

    /* Local variables */
    integer i__, j;
    doublereal s1, s2;
    integer mb;
    doublereal gl;
    integer in, mm;
    doublereal gu;
    integer cnt;
    doublereal eps, tau, tmp, rtl;
    integer cnt1, cnt2;
    doublereal tmp1, eabs;
    integer iend, jblk;
    doublereal eold;
    integer indl;
    doublereal dmax__, emax;
    integer wend, idum, indu;
    doublereal rtol;
    integer iseed[4];
    doublereal avgap, sigma;
    extern logical lsame_(char *, char *);
    integer iinfo;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *);
    logical norep;
    extern /* Subroutine */ int dlasq2_(integer *, doublereal *, integer *);
    extern doublereal dlamch_(char *);
    integer ibegin;
    logical forceb;
    integer irange;
    doublereal sgndef;
    extern /* Subroutine */ int dlarra_(integer *, doublereal *, doublereal *, 
	     doublereal *, doublereal *, doublereal *, integer *, integer *, 
	    integer *), dlarrb_(integer *, doublereal *, doublereal *, 
	    integer *, integer *, doublereal *, doublereal *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *, 
	     doublereal *, doublereal *, integer *, integer *), dlarrc_(char *
, integer *, doublereal *, doublereal *, doublereal *, doublereal 
	    *, doublereal *, integer *, integer *, integer *, integer *);
    integer wbegin;
    extern /* Subroutine */ int dlarrd_(char *, char *, integer *, doublereal 
	    *, doublereal *, integer *, integer *, doublereal *, doublereal *, 
	     doublereal *, doublereal *, doublereal *, doublereal *, integer *
, integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, integer *, integer *, doublereal *, integer *, 
	    integer *);
    doublereal safmin, spdiam;
    extern /* Subroutine */ int dlarrk_(integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, integer *);
    logical usedqd;
    doublereal clwdth, isleft;
    extern /* Subroutine */ int dlarnv_(integer *, integer *, integer *, 
	    doublereal *);
    doublereal isrght, bsrtol, dpivot;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  To find the desired eigenvalues of a given real symmetric */
/*  tridiagonal matrix T, DLARRE sets any "small" off-diagonal */
/*  elements to zero, and for each unreduced block T_i, it finds */
/*  (a) a suitable shift at one end of the block's spectrum, */
/*  (b) the base representation, T_i - sigma_i I = L_i D_i L_i^T, and */
/*  (c) eigenvalues of each L_i D_i L_i^T. */
/*  The representations and eigenvalues found are then used by */
/*  DSTEMR to compute the eigenvectors of T. */
/*  The accuracy varies depending on whether bisection is used to */
/*  find a few eigenvalues or the dqds algorithm (subroutine DLASQ2) to */
/*  conpute all and then discard any unwanted one. */
/*  As an added benefit, DLARRE also outputs the n */
/*  Gerschgorin intervals for the matrices L_i D_i L_i^T. */

/*  Arguments */
/*  ========= */

/*  RANGE   (input) CHARACTER */
/*          = 'A': ("All")   all eigenvalues will be found. */
/*          = 'V': ("Value") all eigenvalues in the half-open interval */
/*                           (VL, VU] will be found. */
/*          = 'I': ("Index") the IL-th through IU-th eigenvalues (of the */
/*                           entire matrix) will be found. */

/*  N       (input) INTEGER */
/*          The order of the matrix. N > 0. */

/*  VL      (input/output) DOUBLE PRECISION */
/*  VU      (input/output) DOUBLE PRECISION */
/*          If RANGE='V', the lower and upper bounds for the eigenvalues. */
/*          Eigenvalues less than or equal to VL, or greater than VU, */
/*          will not be returned.  VL < VU. */
/*          If RANGE='I' or ='A', DLARRE computes bounds on the desired */
/*          part of the spectrum. */

/*  IL      (input) INTEGER */
/*  IU      (input) INTEGER */
/*          If RANGE='I', the indices (in ascending order) of the */
/*          smallest and largest eigenvalues to be returned. */
/*          1 <= IL <= IU <= N. */

/*  D       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the N diagonal elements of the tridiagonal */
/*          matrix T. */
/*          On exit, the N diagonal elements of the diagonal */
/*          matrices D_i. */

/*  E       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the first (N-1) entries contain the subdiagonal */
/*          elements of the tridiagonal matrix T; E(N) need not be set. */
/*          On exit, E contains the subdiagonal elements of the unit */
/*          bidiagonal matrices L_i. The entries E( ISPLIT( I ) ), */
/*          1 <= I <= NSPLIT, contain the base points sigma_i on output. */

/*  E2      (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the first (N-1) entries contain the SQUARES of the */
/*          subdiagonal elements of the tridiagonal matrix T; */
/*          E2(N) need not be set. */
/*          On exit, the entries E2( ISPLIT( I ) ), */
/*          1 <= I <= NSPLIT, have been set to zero */

/*  RTOL1   (input) DOUBLE PRECISION */
/*  RTOL2   (input) DOUBLE PRECISION */
/*           Parameters for bisection. */
/*           An interval [LEFT,RIGHT] has converged if */
/*           RIGHT-LEFT.LT.MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) ) */

/*  SPLTOL (input) DOUBLE PRECISION */
/*          The threshold for splitting. */

/*  NSPLIT  (output) INTEGER */
/*          The number of blocks T splits into. 1 <= NSPLIT <= N. */

/*  ISPLIT  (output) INTEGER array, dimension (N) */
/*          The splitting points, at which T breaks up into blocks. */
/*          The first block consists of rows/columns 1 to ISPLIT(1), */
/*          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2), */
/*          etc., and the NSPLIT-th consists of rows/columns */
/*          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N. */

/*  M       (output) INTEGER */
/*          The total number of eigenvalues (of all L_i D_i L_i^T) */
/*          found. */

/*  W       (output) DOUBLE PRECISION array, dimension (N) */
/*          The first M elements contain the eigenvalues. The */
/*          eigenvalues of each of the blocks, L_i D_i L_i^T, are */
/*          sorted in ascending order ( DLARRE may use the */
/*          remaining N-M elements as workspace). */

/*  WERR    (output) DOUBLE PRECISION array, dimension (N) */
/*          The error bound on the corresponding eigenvalue in W. */

/*  WGAP    (output) DOUBLE PRECISION array, dimension (N) */
/*          The separation from the right neighbor eigenvalue in W. */
/*          The gap is only with respect to the eigenvalues of the same block */
/*          as each block has its own representation tree. */
/*          Exception: at the right end of a block we store the left gap */

/*  IBLOCK  (output) INTEGER array, dimension (N) */
/*          The indices of the blocks (submatrices) associated with the */
/*          corresponding eigenvalues in W; IBLOCK(i)=1 if eigenvalue */
/*          W(i) belongs to the first block from the top, =2 if W(i) */
/*          belongs to the second block, etc. */

/*  INDEXW  (output) INTEGER array, dimension (N) */
/*          The indices of the eigenvalues within each block (submatrix); */
/*          for example, INDEXW(i)= 10 and IBLOCK(i)=2 imply that the */
/*          i-th eigenvalue W(i) is the 10-th eigenvalue in block 2 */

/*  GERS    (output) DOUBLE PRECISION array, dimension (2*N) */
/*          The N Gerschgorin intervals (the i-th Gerschgorin interval */
/*          is (GERS(2*i-1), GERS(2*i)). */

/*  PIVMIN  (output) DOUBLE PRECISION */
/*          The minimum pivot in the Sturm sequence for T. */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (6*N) */
/*          Workspace. */

/*  IWORK   (workspace) INTEGER array, dimension (5*N) */
/*          Workspace. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          > 0:  A problem occured in DLARRE. */
/*          < 0:  One of the called subroutines signaled an internal problem. */
/*                Needs inspection of the corresponding parameter IINFO */
/*                for further information. */

/*          =-1:  Problem in DLARRD. */
/*          = 2:  No base representation could be found in MAXTRY iterations. */
/*                Increasing MAXTRY and recompilation might be a remedy. */
/*          =-3:  Problem in DLARRB when computing the refined root */
/*                representation for DLASQ2. */
/*          =-4:  Problem in DLARRB when preforming bisection on the */
/*                desired part of the spectrum. */
/*          =-5:  Problem in DLASQ2. */
/*          =-6:  Problem in DLASQ2. */

/*  Further Details */
/*  The base representations are required to suffer very little */
/*  element growth and consequently define all their eigenvalues to */
/*  high relative accuracy. */
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
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --iwork;
    --work;
    --gers;
    --indexw;
    --iblock;
    --wgap;
    --werr;
    --w;
    --isplit;
    --e2;
    --e;
    --d__;

    /* Function Body */
    *info = 0;

/*     Decode RANGE */

    if (lsame_(range, "A")) {
	irange = 1;
    } else if (lsame_(range, "V")) {
	irange = 3;
    } else if (lsame_(range, "I")) {
	irange = 2;
    }
    *m = 0;
/*     Get machine constants */
    safmin = dlamch_("S");
    eps = dlamch_("P");
/*     Set parameters */
    rtl = sqrt(eps);
    bsrtol = sqrt(eps);
/*     Treat case of 1x1 matrix for quick return */
    if (*n == 1) {
	if (irange == 1 || irange == 3 && d__[1] > *vl && d__[1] <= *vu || 
		irange == 2 && *il == 1 && *iu == 1) {
	    *m = 1;
	    w[1] = d__[1];
/*           The computation error of the eigenvalue is zero */
	    werr[1] = 0.;
	    wgap[1] = 0.;
	    iblock[1] = 1;
	    indexw[1] = 1;
	    gers[1] = d__[1];
	    gers[2] = d__[1];
	}
/*        store the shift for the initial RRR, which is zero in this case */
	e[1] = 0.;
	return 0;
    }
/*     General case: tridiagonal matrix of order > 1 */

/*     Init WERR, WGAP. Compute Gerschgorin intervals and spectral diameter. */
/*     Compute maximum off-diagonal entry and pivmin. */
    gl = d__[1];
    gu = d__[1];
    eold = 0.;
    emax = 0.;
    e[*n] = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	werr[i__] = 0.;
	wgap[i__] = 0.;
	eabs = (d__1 = e[i__], abs(d__1));
	if (eabs >= emax) {
	    emax = eabs;
	}
	tmp1 = eabs + eold;
	gers[(i__ << 1) - 1] = d__[i__] - tmp1;
/* Computing MIN */
	d__1 = gl, d__2 = gers[(i__ << 1) - 1];
	gl = min(d__1,d__2);
	gers[i__ * 2] = d__[i__] + tmp1;
/* Computing MAX */
	d__1 = gu, d__2 = gers[i__ * 2];
	gu = max(d__1,d__2);
	eold = eabs;
/* L5: */
    }
/*     The minimum pivot allowed in the Sturm sequence for T */
/* Computing MAX */
/* Computing 2nd power */
    d__3 = emax;
    d__1 = 1., d__2 = d__3 * d__3;
    *pivmin = safmin * max(d__1,d__2);
/*     Compute spectral diameter. The Gerschgorin bounds give an */
/*     estimate that is wrong by at most a factor of SQRT(2) */
    spdiam = gu - gl;
/*     Compute splitting points */
    dlarra_(n, &d__[1], &e[1], &e2[1], spltol, &spdiam, nsplit, &isplit[1], &
	    iinfo);
/*     Can force use of bisection instead of faster DQDS. */
/*     Option left in the code for future multisection work. */
    forceb = FALSE_;
    if (irange == 1 && ! forceb) {
/*        Set interval [VL,VU] that contains all eigenvalues */
	*vl = gl;
	*vu = gu;
    } else {
/*        We call DLARRD to find crude approximations to the eigenvalues */
/*        in the desired range. In case IRANGE = INDRNG, we also obtain the */
/*        interval (VL,VU] that contains all the wanted eigenvalues. */
/*        An interval [LEFT,RIGHT] has converged if */
/*        RIGHT-LEFT.LT.RTOL*MAX(ABS(LEFT),ABS(RIGHT)) */
/*        DLARRD needs a WORK of size 4*N, IWORK of size 3*N */
	dlarrd_(range, "B", n, vl, vu, il, iu, &gers[1], &bsrtol, &d__[1], &e[
		1], &e2[1], pivmin, nsplit, &isplit[1], &mm, &w[1], &werr[1], 
		vl, vu, &iblock[1], &indexw[1], &work[1], &iwork[1], &iinfo);
	if (iinfo != 0) {
	    *info = -1;
	    return 0;
	}
/*        Make sure that the entries M+1 to N in W, WERR, IBLOCK, INDEXW are 0 */
	i__1 = *n;
	for (i__ = mm + 1; i__ <= i__1; ++i__) {
	    w[i__] = 0.;
	    werr[i__] = 0.;
	    iblock[i__] = 0;
	    indexw[i__] = 0;
/* L14: */
	}
    }
/* ** */
/*     Loop over unreduced blocks */
    ibegin = 1;
    wbegin = 1;
    i__1 = *nsplit;
    for (jblk = 1; jblk <= i__1; ++jblk) {
	iend = isplit[jblk];
	in = iend - ibegin + 1;
/*        1 X 1 block */
	if (in == 1) {
	    if (irange == 1 || irange == 3 && d__[ibegin] > *vl && d__[ibegin]
		     <= *vu || irange == 2 && iblock[wbegin] == jblk) {
		++(*m);
		w[*m] = d__[ibegin];
		werr[*m] = 0.;
/*              The gap for a single block doesn't matter for the later */
/*              algorithm and is assigned an arbitrary large value */
		wgap[*m] = 0.;
		iblock[*m] = jblk;
		indexw[*m] = 1;
		++wbegin;
	    }
/*           E( IEND ) holds the shift for the initial RRR */
	    e[iend] = 0.;
	    ibegin = iend + 1;
	    goto L170;
	}

/*        Blocks of size larger than 1x1 */

/*        E( IEND ) will hold the shift for the initial RRR, for now set it =0 */
	e[iend] = 0.;

/*        Find local outer bounds GL,GU for the block */
	gl = d__[ibegin];
	gu = d__[ibegin];
	i__2 = iend;
	for (i__ = ibegin; i__ <= i__2; ++i__) {
/* Computing MIN */
	    d__1 = gers[(i__ << 1) - 1];
	    gl = min(d__1,gl);
/* Computing MAX */
	    d__1 = gers[i__ * 2];
	    gu = max(d__1,gu);
/* L15: */
	}
	spdiam = gu - gl;
	if (! (irange == 1 && ! forceb)) {
/*           Count the number of eigenvalues in the current block. */
	    mb = 0;
	    i__2 = mm;
	    for (i__ = wbegin; i__ <= i__2; ++i__) {
		if (iblock[i__] == jblk) {
		    ++mb;
		} else {
		    goto L21;
		}
/* L20: */
	    }
L21:
	    if (mb == 0) {
/*              No eigenvalue in the current block lies in the desired range */
/*              E( IEND ) holds the shift for the initial RRR */
		e[iend] = 0.;
		ibegin = iend + 1;
		goto L170;
	    } else {
/*              Decide whether dqds or bisection is more efficient */
		usedqd = (doublereal) mb > in * .5 && ! forceb;
		wend = wbegin + mb - 1;
/*              Calculate gaps for the current block */
/*              In later stages, when representations for individual */
/*              eigenvalues are different, we use SIGMA = E( IEND ). */
		sigma = 0.;
		i__2 = wend - 1;
		for (i__ = wbegin; i__ <= i__2; ++i__) {
/* Computing MAX */
		    d__1 = 0., d__2 = w[i__ + 1] - werr[i__ + 1] - (w[i__] + 
			    werr[i__]);
		    wgap[i__] = max(d__1,d__2);
/* L30: */
		}
/* Computing MAX */
		d__1 = 0., d__2 = *vu - sigma - (w[wend] + werr[wend]);
		wgap[wend] = max(d__1,d__2);
/*              Find local index of the first and last desired evalue. */
		indl = indexw[wbegin];
		indu = indexw[wend];
	    }
	}
	if (irange == 1 && ! forceb || usedqd) {
/*           Case of DQDS */
/*           Find approximations to the extremal eigenvalues of the block */
	    dlarrk_(&in, &c__1, &gl, &gu, &d__[ibegin], &e2[ibegin], pivmin, &
		    rtl, &tmp, &tmp1, &iinfo);
	    if (iinfo != 0) {
		*info = -1;
		return 0;
	    }
/* Computing MAX */
	    d__2 = gl, d__3 = tmp - tmp1 - eps * 100. * (d__1 = tmp - tmp1, 
		    abs(d__1));
	    isleft = max(d__2,d__3);
	    dlarrk_(&in, &in, &gl, &gu, &d__[ibegin], &e2[ibegin], pivmin, &
		    rtl, &tmp, &tmp1, &iinfo);
	    if (iinfo != 0) {
		*info = -1;
		return 0;
	    }
/* Computing MIN */
	    d__2 = gu, d__3 = tmp + tmp1 + eps * 100. * (d__1 = tmp + tmp1, 
		    abs(d__1));
	    isrght = min(d__2,d__3);
/*           Improve the estimate of the spectral diameter */
	    spdiam = isrght - isleft;
	} else {
/*           Case of bisection */
/*           Find approximations to the wanted extremal eigenvalues */
/* Computing MAX */
	    d__2 = gl, d__3 = w[wbegin] - werr[wbegin] - eps * 100. * (d__1 = 
		    w[wbegin] - werr[wbegin], abs(d__1));
	    isleft = max(d__2,d__3);
/* Computing MIN */
	    d__2 = gu, d__3 = w[wend] + werr[wend] + eps * 100. * (d__1 = w[
		    wend] + werr[wend], abs(d__1));
	    isrght = min(d__2,d__3);
	}
/*        Decide whether the base representation for the current block */
/*        L_JBLK D_JBLK L_JBLK^T = T_JBLK - sigma_JBLK I */
/*        should be on the left or the right end of the current block. */
/*        The strategy is to shift to the end which is "more populated" */
/*        Furthermore, decide whether to use DQDS for the computation of */
/*        the eigenvalue approximations at the end of DLARRE or bisection. */
/*        dqds is chosen if all eigenvalues are desired or the number of */
/*        eigenvalues to be computed is large compared to the blocksize. */
	if (irange == 1 && ! forceb) {
/*           If all the eigenvalues have to be computed, we use dqd */
	    usedqd = TRUE_;
/*           INDL is the local index of the first eigenvalue to compute */
	    indl = 1;
	    indu = in;
/*           MB =  number of eigenvalues to compute */
	    mb = in;
	    wend = wbegin + mb - 1;
/*           Define 1/4 and 3/4 points of the spectrum */
	    s1 = isleft + spdiam * .25;
	    s2 = isrght - spdiam * .25;
	} else {
/*           DLARRD has computed IBLOCK and INDEXW for each eigenvalue */
/*           approximation. */
/*           choose sigma */
	    if (usedqd) {
		s1 = isleft + spdiam * .25;
		s2 = isrght - spdiam * .25;
	    } else {
		tmp = min(isrght,*vu) - max(isleft,*vl);
		s1 = max(isleft,*vl) + tmp * .25;
		s2 = min(isrght,*vu) - tmp * .25;
	    }
	}
/*        Compute the negcount at the 1/4 and 3/4 points */
	if (mb > 1) {
	    dlarrc_("T", &in, &s1, &s2, &d__[ibegin], &e[ibegin], pivmin, &
		    cnt, &cnt1, &cnt2, &iinfo);
	}
	if (mb == 1) {
	    sigma = gl;
	    sgndef = 1.;
	} else if (cnt1 - indl >= indu - cnt2) {
	    if (irange == 1 && ! forceb) {
		sigma = max(isleft,gl);
	    } else if (usedqd) {
/*              use Gerschgorin bound as shift to get pos def matrix */
/*              for dqds */
		sigma = isleft;
	    } else {
/*              use approximation of the first desired eigenvalue of the */
/*              block as shift */
		sigma = max(isleft,*vl);
	    }
	    sgndef = 1.;
	} else {
	    if (irange == 1 && ! forceb) {
		sigma = min(isrght,gu);
	    } else if (usedqd) {
/*              use Gerschgorin bound as shift to get neg def matrix */
/*              for dqds */
		sigma = isrght;
	    } else {
/*              use approximation of the first desired eigenvalue of the */
/*              block as shift */
		sigma = min(isrght,*vu);
	    }
	    sgndef = -1.;
	}
/*        An initial SIGMA has been chosen that will be used for computing */
/*        T - SIGMA I = L D L^T */
/*        Define the increment TAU of the shift in case the initial shift */
/*        needs to be refined to obtain a factorization with not too much */
/*        element growth. */
	if (usedqd) {
/*           The initial SIGMA was to the outer end of the spectrum */
/*           the matrix is definite and we need not retreat. */
	    tau = spdiam * eps * *n + *pivmin * 2.;
	} else {
	    if (mb > 1) {
		clwdth = w[wend] + werr[wend] - w[wbegin] - werr[wbegin];
		avgap = (d__1 = clwdth / (doublereal) (wend - wbegin), abs(
			d__1));
		if (sgndef == 1.) {
/* Computing MAX */
		    d__1 = wgap[wbegin];
		    tau = max(d__1,avgap) * .5;
/* Computing MAX */
		    d__1 = tau, d__2 = werr[wbegin];
		    tau = max(d__1,d__2);
		} else {
/* Computing MAX */
		    d__1 = wgap[wend - 1];
		    tau = max(d__1,avgap) * .5;
/* Computing MAX */
		    d__1 = tau, d__2 = werr[wend];
		    tau = max(d__1,d__2);
		}
	    } else {
		tau = werr[wbegin];
	    }
	}

	for (idum = 1; idum <= 6; ++idum) {
/*           Compute L D L^T factorization of tridiagonal matrix T - sigma I. */
/*           Store D in WORK(1:IN), L in WORK(IN+1:2*IN), and reciprocals of */
/*           pivots in WORK(2*IN+1:3*IN) */
	    dpivot = d__[ibegin] - sigma;
	    work[1] = dpivot;
	    dmax__ = abs(work[1]);
	    j = ibegin;
	    i__2 = in - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[(in << 1) + i__] = 1. / work[i__];
		tmp = e[j] * work[(in << 1) + i__];
		work[in + i__] = tmp;
		dpivot = d__[j + 1] - sigma - tmp * e[j];
		work[i__ + 1] = dpivot;
/* Computing MAX */
		d__1 = dmax__, d__2 = abs(dpivot);
		dmax__ = max(d__1,d__2);
		++j;
/* L70: */
	    }
/*           check for element growth */
	    if (dmax__ > spdiam * 64.) {
		norep = TRUE_;
	    } else {
		norep = FALSE_;
	    }
	    if (usedqd && ! norep) {
/*              Ensure the definiteness of the representation */
/*              All entries of D (of L D L^T) must have the same sign */
		i__2 = in;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    tmp = sgndef * work[i__];
		    if (tmp < 0.) {
			norep = TRUE_;
		    }
/* L71: */
		}
	    }
	    if (norep) {
/*              Note that in the case of IRANGE=ALLRNG, we use the Gerschgorin */
/*              shift which makes the matrix definite. So we should end up */
/*              here really only in the case of IRANGE = VALRNG or INDRNG. */
		if (idum == 5) {
		    if (sgndef == 1.) {
/*                    The fudged Gerschgorin shift should succeed */
			sigma = gl - spdiam * 2. * eps * *n - *pivmin * 4.;
		    } else {
			sigma = gu + spdiam * 2. * eps * *n + *pivmin * 4.;
		    }
		} else {
		    sigma -= sgndef * tau;
		    tau *= 2.;
		}
	    } else {
/*              an initial RRR is found */
		goto L83;
	    }
/* L80: */
	}
/*        if the program reaches this point, no base representation could be */
/*        found in MAXTRY iterations. */
	*info = 2;
	return 0;
L83:
/*        At this point, we have found an initial base representation */
/*        T - SIGMA I = L D L^T with not too much element growth. */
/*        Store the shift. */
	e[iend] = sigma;
/*        Store D and L. */
	dcopy_(&in, &work[1], &c__1, &d__[ibegin], &c__1);
	i__2 = in - 1;
	dcopy_(&i__2, &work[in + 1], &c__1, &e[ibegin], &c__1);
	if (mb > 1) {

/*           Perturb each entry of the base representation by a small */
/*           (but random) relative amount to overcome difficulties with */
/*           glued matrices. */

	    for (i__ = 1; i__ <= 4; ++i__) {
		iseed[i__ - 1] = 1;
/* L122: */
	    }
	    i__2 = (in << 1) - 1;
	    dlarnv_(&c__2, iseed, &i__2, &work[1]);
	    i__2 = in - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		d__[ibegin + i__ - 1] *= eps * 8. * work[i__] + 1.;
		e[ibegin + i__ - 1] *= eps * 8. * work[in + i__] + 1.;
/* L125: */
	    }
	    d__[iend] *= eps * 4. * work[in] + 1.;

	}

/*        Don't update the Gerschgorin intervals because keeping track */
/*        of the updates would be too much work in DLARRV. */
/*        We update W instead and use it to locate the proper Gerschgorin */
/*        intervals. */
/*        Compute the required eigenvalues of L D L' by bisection or dqds */
	if (! usedqd) {
/*           If DLARRD has been used, shift the eigenvalue approximations */
/*           according to their representation. This is necessary for */
/*           a uniform DLARRV since dqds computes eigenvalues of the */
/*           shifted representation. In DLARRV, W will always hold the */
/*           UNshifted eigenvalue approximation. */
	    i__2 = wend;
	    for (j = wbegin; j <= i__2; ++j) {
		w[j] -= sigma;
		werr[j] += (d__1 = w[j], abs(d__1)) * eps;
/* L134: */
	    }
/*           call DLARRB to reduce eigenvalue error of the approximations */
/*           from DLARRD */
	    i__2 = iend - 1;
	    for (i__ = ibegin; i__ <= i__2; ++i__) {
/* Computing 2nd power */
		d__1 = e[i__];
		work[i__] = d__[i__] * (d__1 * d__1);
/* L135: */
	    }
/*           use bisection to find EV from INDL to INDU */
	    i__2 = indl - 1;
	    dlarrb_(&in, &d__[ibegin], &work[ibegin], &indl, &indu, rtol1, 
		    rtol2, &i__2, &w[wbegin], &wgap[wbegin], &werr[wbegin], &
		    work[(*n << 1) + 1], &iwork[1], pivmin, &spdiam, &in, &
		    iinfo);
	    if (iinfo != 0) {
		*info = -4;
		return 0;
	    }
/*           DLARRB computes all gaps correctly except for the last one */
/*           Record distance to VU/GU */
/* Computing MAX */
	    d__1 = 0., d__2 = *vu - sigma - (w[wend] + werr[wend]);
	    wgap[wend] = max(d__1,d__2);
	    i__2 = indu;
	    for (i__ = indl; i__ <= i__2; ++i__) {
		++(*m);
		iblock[*m] = jblk;
		indexw[*m] = i__;
/* L138: */
	    }
	} else {
/*           Call dqds to get all eigs (and then possibly delete unwanted */
/*           eigenvalues). */
/*           Note that dqds finds the eigenvalues of the L D L^T representation */
/*           of T to high relative accuracy. High relative accuracy */
/*           might be lost when the shift of the RRR is subtracted to obtain */
/*           the eigenvalues of T. However, T is not guaranteed to define its */
/*           eigenvalues to high relative accuracy anyway. */
/*           Set RTOL to the order of the tolerance used in DLASQ2 */
/*           This is an ESTIMATED error, the worst case bound is 4*N*EPS */
/*           which is usually too large and requires unnecessary work to be */
/*           done by bisection when computing the eigenvectors */
	    rtol = log((doublereal) in) * 4. * eps;
	    j = ibegin;
	    i__2 = in - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[(i__ << 1) - 1] = (d__1 = d__[j], abs(d__1));
		work[i__ * 2] = e[j] * e[j] * work[(i__ << 1) - 1];
		++j;
/* L140: */
	    }
	    work[(in << 1) - 1] = (d__1 = d__[iend], abs(d__1));
	    work[in * 2] = 0.;
	    dlasq2_(&in, &work[1], &iinfo);
	    if (iinfo != 0) {
/*              If IINFO = -5 then an index is part of a tight cluster */
/*              and should be changed. The index is in IWORK(1) and the */
/*              gap is in WORK(N+1) */
		*info = -5;
		return 0;
	    } else {
/*              Test that all eigenvalues are positive as expected */
		i__2 = in;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    if (work[i__] < 0.) {
			*info = -6;
			return 0;
		    }
/* L149: */
		}
	    }
	    if (sgndef > 0.) {
		i__2 = indu;
		for (i__ = indl; i__ <= i__2; ++i__) {
		    ++(*m);
		    w[*m] = work[in - i__ + 1];
		    iblock[*m] = jblk;
		    indexw[*m] = i__;
/* L150: */
		}
	    } else {
		i__2 = indu;
		for (i__ = indl; i__ <= i__2; ++i__) {
		    ++(*m);
		    w[*m] = -work[i__];
		    iblock[*m] = jblk;
		    indexw[*m] = i__;
/* L160: */
		}
	    }
	    i__2 = *m;
	    for (i__ = *m - mb + 1; i__ <= i__2; ++i__) {
/*              the value of RTOL below should be the tolerance in DLASQ2 */
		werr[i__] = rtol * (d__1 = w[i__], abs(d__1));
/* L165: */
	    }
	    i__2 = *m - 1;
	    for (i__ = *m - mb + 1; i__ <= i__2; ++i__) {
/*              compute the right gap between the intervals */
/* Computing MAX */
		d__1 = 0., d__2 = w[i__ + 1] - werr[i__ + 1] - (w[i__] + werr[
			i__]);
		wgap[i__] = max(d__1,d__2);
/* L166: */
	    }
/* Computing MAX */
	    d__1 = 0., d__2 = *vu - sigma - (w[*m] + werr[*m]);
	    wgap[*m] = max(d__1,d__2);
	}
/*        proceed with next block */
	ibegin = iend + 1;
	wbegin = wend + 1;
L170:
	;
    }

    return 0;

/*     end of DLARRE */

} /* dlarre_ */
