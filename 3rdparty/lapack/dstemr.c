#include "clapack.h"

/* Table of constant values */

static integer c__1 = 1;
static doublereal c_b18 = .001;

/* Subroutine */ int dstemr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, integer *m, doublereal *w, doublereal *z__, integer *ldz, 
	 integer *nzc, integer *isuppz, logical *tryrac, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info)
{
    /* System generated locals */
    integer z_dim1, z_offset, i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j;
    doublereal r1, r2;
    integer jj;
    doublereal cs;
    integer in;
    doublereal sn, wl, wu;
    integer iil, iiu;
    doublereal eps, tmp;
    integer indd, iend, jblk, wend;
    doublereal rmin, rmax;
    integer itmp;
    doublereal tnrm;
    extern /* Subroutine */ int dlae2_(doublereal *, doublereal *, doublereal 
	    *, doublereal *, doublereal *);
    integer inde2, itmp2;
    doublereal rtol1, rtol2;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    doublereal scale;
    integer indgp;
    extern logical lsame_(char *, char *);
    integer iinfo, iindw, ilast;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *), dswap_(integer *, doublereal *, integer 
	    *, doublereal *, integer *);
    integer lwmin;
    logical wantz;
    extern /* Subroutine */ int dlaev2_(doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *);
    extern doublereal dlamch_(char *);
    logical alleig;
    integer ibegin;
    logical indeig;
    integer iindbl;
    logical valeig;
    extern /* Subroutine */ int dlarrc_(char *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *, 
	     integer *, integer *, integer *), dlarre_(char *, 
	    integer *, doublereal *, doublereal *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *, integer *);
    integer wbegin;
    doublereal safmin;
    extern /* Subroutine */ int dlarrj_(integer *, doublereal *, doublereal *, 
	     integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *, 
	     integer *), xerbla_(char *, integer *);
    doublereal bignum;
    integer inderr, iindwk, indgrs, offset;
    extern doublereal dlanst_(char *, integer *, doublereal *, doublereal *);
    extern /* Subroutine */ int dlarrr_(integer *, doublereal *, doublereal *, 
	     integer *), dlarrv_(integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, integer *, integer *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    integer *, integer *), dlasrt_(char *, integer *, doublereal *, 
	    integer *);
    doublereal thresh;
    integer iinspl, ifirst, indwrk, liwmin, nzcmin;
    doublereal pivmin;
    integer nsplit;
    doublereal smlnum;
    logical lquery, zquery;


/*  -- LAPACK computational routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DSTEMR computes selected eigenvalues and, optionally, eigenvectors */
/*  of a real symmetric tridiagonal matrix T. Any such unreduced matrix has */
/*  a well defined set of pairwise different real eigenvalues, the corresponding */
/*  real eigenvectors are pairwise orthogonal. */

/*  The spectrum may be computed either completely or partially by specifying */
/*  either an interval (VL,VU] or a range of indices IL:IU for the desired */
/*  eigenvalues. */

/*  Depending on the number of desired eigenvalues, these are computed either */
/*  by bisection or the dqds algorithm. Numerically orthogonal eigenvectors are */
/*  computed by the use of various suitable L D L^T factorizations near clusters */
/*  of close eigenvalues (referred to as RRRs, Relatively Robust */
/*  Representations). An informal sketch of the algorithm follows. */

/*  For each unreduced block (submatrix) of T, */
/*     (a) Compute T - sigma I  = L D L^T, so that L and D */
/*         define all the wanted eigenvalues to high relative accuracy. */
/*         This means that small relative changes in the entries of D and L */
/*         cause only small relative changes in the eigenvalues and */
/*         eigenvectors. The standard (unfactored) representation of the */
/*         tridiagonal matrix T does not have this property in general. */
/*     (b) Compute the eigenvalues to suitable accuracy. */
/*         If the eigenvectors are desired, the algorithm attains full */
/*         accuracy of the computed eigenvalues only right before */
/*         the corresponding vectors have to be computed, see steps c) and d). */
/*     (c) For each cluster of close eigenvalues, select a new */
/*         shift close to the cluster, find a new factorization, and refine */
/*         the shifted eigenvalues to suitable accuracy. */
/*     (d) For each eigenvalue with a large enough relative separation compute */
/*         the corresponding eigenvector by forming a rank revealing twisted */
/*         factorization. Go back to (c) for any clusters that remain. */

/*  For more details, see: */
/*  - Inderjit S. Dhillon and Beresford N. Parlett: "Multiple representations */
/*    to compute orthogonal eigenvectors of symmetric tridiagonal matrices," */
/*    Linear Algebra and its Applications, 387(1), pp. 1-28, August 2004. */
/*  - Inderjit Dhillon and Beresford Parlett: "Orthogonal Eigenvectors and */
/*    Relative Gaps," SIAM Journal on Matrix Analysis and Applications, Vol. 25, */
/*    2004.  Also LAPACK Working Note 154. */
/*  - Inderjit Dhillon: "A new O(n^2) algorithm for the symmetric */
/*    tridiagonal eigenvalue/eigenvector problem", */
/*    Computer Science Division Technical Report No. UCB/CSD-97-971, */
/*    UC Berkeley, May 1997. */

/*  Notes: */
/*  1.DSTEMR works only on machines which follow IEEE-754 */
/*  floating-point standard in their handling of infinities and NaNs. */
/*  This permits the use of efficient inner loops avoiding a check for */
/*  zero divisors. */

/*  Arguments */
/*  ========= */

/*  JOBZ    (input) CHARACTER*1 */
/*          = 'N':  Compute eigenvalues only; */
/*          = 'V':  Compute eigenvalues and eigenvectors. */

/*  RANGE   (input) CHARACTER*1 */
/*          = 'A': all eigenvalues will be found. */
/*          = 'V': all eigenvalues in the half-open interval (VL,VU] */
/*                 will be found. */
/*          = 'I': the IL-th through IU-th eigenvalues will be found. */

/*  N       (input) INTEGER */
/*          The order of the matrix.  N >= 0. */

/*  D       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the N diagonal elements of the tridiagonal matrix */
/*          T. On exit, D is overwritten. */

/*  E       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the (N-1) subdiagonal elements of the tridiagonal */
/*          matrix T in elements 1 to N-1 of E. E(N) need not be set on */
/*          input, but is used internally as workspace. */
/*          On exit, E is overwritten. */

/*  VL      (input) DOUBLE PRECISION */
/*  VU      (input) DOUBLE PRECISION */
/*          If RANGE='V', the lower and upper bounds of the interval to */
/*          be searched for eigenvalues. VL < VU. */
/*          Not referenced if RANGE = 'A' or 'I'. */

/*  IL      (input) INTEGER */
/*  IU      (input) INTEGER */
/*          If RANGE='I', the indices (in ascending order) of the */
/*          smallest and largest eigenvalues to be returned. */
/*          1 <= IL <= IU <= N, if N > 0. */
/*          Not referenced if RANGE = 'A' or 'V'. */

/*  M       (output) INTEGER */
/*          The total number of eigenvalues found.  0 <= M <= N. */
/*          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1. */

/*  W       (output) DOUBLE PRECISION array, dimension (N) */
/*          The first M elements contain the selected eigenvalues in */
/*          ascending order. */

/*  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M) ) */
/*          If JOBZ = 'V', and if INFO = 0, then the first M columns of Z */
/*          contain the orthonormal eigenvectors of the matrix T */
/*          corresponding to the selected eigenvalues, with the i-th */
/*          column of Z holding the eigenvector associated with W(i). */
/*          If JOBZ = 'N', then Z is not referenced. */
/*          Note: the user must ensure that at least max(1,M) columns are */
/*          supplied in the array Z; if RANGE = 'V', the exact value of M */
/*          is not known in advance and can be computed with a workspace */
/*          query by setting NZC = -1, see below. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDZ >= 1, and if */
/*          JOBZ = 'V', then LDZ >= max(1,N). */

/*  NZC     (input) INTEGER */
/*          The number of eigenvectors to be held in the array Z. */
/*          If RANGE = 'A', then NZC >= max(1,N). */
/*          If RANGE = 'V', then NZC >= the number of eigenvalues in (VL,VU]. */
/*          If RANGE = 'I', then NZC >= IU-IL+1. */
/*          If NZC = -1, then a workspace query is assumed; the */
/*          routine calculates the number of columns of the array Z that */
/*          are needed to hold the eigenvectors. */
/*          This value is returned as the first entry of the Z array, and */
/*          no error message related to NZC is issued by XERBLA. */

/*  ISUPPZ  (output) INTEGER ARRAY, dimension ( 2*max(1,M) ) */
/*          The support of the eigenvectors in Z, i.e., the indices */
/*          indicating the nonzero elements in Z. The i-th computed eigenvector */
/*          is nonzero only in elements ISUPPZ( 2*i-1 ) through */
/*          ISUPPZ( 2*i ). This is relevant in the case when the matrix */
/*          is split. ISUPPZ is only accessed when JOBZ is 'V' and N > 0. */

/*  TRYRAC  (input/output) LOGICAL */
/*          If TRYRAC.EQ..TRUE., indicates that the code should check whether */
/*          the tridiagonal matrix defines its eigenvalues to high relative */
/*          accuracy.  If so, the code uses relative-accuracy preserving */
/*          algorithms that might be (a bit) slower depending on the matrix. */
/*          If the matrix does not define its eigenvalues to high relative */
/*          accuracy, the code can uses possibly faster algorithms. */
/*          If TRYRAC.EQ..FALSE., the code is not required to guarantee */
/*          relatively accurate eigenvalues and can use the fastest possible */
/*          techniques. */
/*          On exit, a .TRUE. TRYRAC will be set to .FALSE. if the matrix */
/*          does not define its eigenvalues to high relative accuracy. */

/*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal */
/*          (and minimal) LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >= max(1,18*N) */
/*          if JOBZ = 'V', and LWORK >= max(1,12*N) if JOBZ = 'N'. */
/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  IWORK   (workspace/output) INTEGER array, dimension (LIWORK) */
/*          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK. */

/*  LIWORK  (input) INTEGER */
/*          The dimension of the array IWORK.  LIWORK >= max(1,10*N) */
/*          if the eigenvectors are desired, and LIWORK >= max(1,8*N) */
/*          if only the eigenvalues are to be computed. */
/*          If LIWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal size of the IWORK array, */
/*          returns this value as the first entry of the IWORK array, and */
/*          no error message related to LIWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          On exit, INFO */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = 1X, internal error in DLARRE, */
/*                if INFO = 2X, internal error in DLARRV. */
/*                Here, the digit X = ABS( IINFO ) < 10, where IINFO is */
/*                the nonzero error code returned by DLARRE or */
/*                DLARRV, respectively. */


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
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --e;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --isuppz;
    --work;
    --iwork;

    /* Function Body */
    wantz = lsame_(jobz, "V");
    alleig = lsame_(range, "A");
    valeig = lsame_(range, "V");
    indeig = lsame_(range, "I");

    lquery = *lwork == -1 || *liwork == -1;
    zquery = *nzc == -1;
    *tryrac = *info != 0;
/*     DSTEMR needs WORK of size 6*N, IWORK of size 3*N. */
/*     In addition, DLARRE needs WORK of size 6*N, IWORK of size 5*N. */
/*     Furthermore, DLARRV needs WORK of size 12*N, IWORK of size 7*N. */
    if (wantz) {
	lwmin = *n * 18;
	liwmin = *n * 10;
    } else {
/*        need less workspace if only the eigenvalues are wanted */
	lwmin = *n * 12;
	liwmin = *n << 3;
    }
    wl = 0.;
    wu = 0.;
    iil = 0;
    iiu = 0;
    if (valeig) {
/*        We do not reference VL, VU in the cases RANGE = 'I','A' */
/*        The interval (WL, WU] contains all the wanted eigenvalues. */
/*        It is either given by the user or computed in DLARRE. */
	wl = *vl;
	wu = *vu;
    } else if (indeig) {
/*        We do not reference IL, IU in the cases RANGE = 'V','A' */
	iil = *il;
	iiu = *iu;
    }

    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (alleig || valeig || indeig)) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (valeig && *n > 0 && wu <= wl) {
	*info = -7;
    } else if (indeig && (iil < 1 || iil > *n)) {
	*info = -8;
    } else if (indeig && (iiu < iil || iiu > *n)) {
	*info = -9;
    } else if (*ldz < 1 || wantz && *ldz < *n) {
	*info = -13;
    } else if (*lwork < lwmin && ! lquery) {
	*info = -17;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -19;
    }

/*     Get machine constants. */

    safmin = dlamch_("Safe minimum");
    eps = dlamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = sqrt(smlnum);
/* Computing MIN */
    d__1 = sqrt(bignum), d__2 = 1. / sqrt(sqrt(safmin));
    rmax = min(d__1,d__2);

    if (*info == 0) {
	work[1] = (doublereal) lwmin;
	iwork[1] = liwmin;

	if (wantz && alleig) {
	    nzcmin = *n;
	} else if (wantz && valeig) {
	    dlarrc_("T", n, vl, vu, &d__[1], &e[1], &safmin, &nzcmin, &itmp, &
		    itmp2, info);
	} else if (wantz && indeig) {
	    nzcmin = iiu - iil + 1;
	} else {
/*           WANTZ .EQ. FALSE. */
	    nzcmin = 0;
	}
	if (zquery && *info == 0) {
	    z__[z_dim1 + 1] = (doublereal) nzcmin;
	} else if (*nzc < nzcmin && ! zquery) {
	    *info = -14;
	}
    }
    if (*info != 0) {

	i__1 = -(*info);
	xerbla_("DSTEMR", &i__1);

	return 0;
    } else if (lquery || zquery) {
	return 0;
    }

/*     Handle N = 0, 1, and 2 cases immediately */

    *m = 0;
    if (*n == 0) {
	return 0;
    }

    if (*n == 1) {
	if (alleig || indeig) {
	    *m = 1;
	    w[1] = d__[1];
	} else {
	    if (wl < d__[1] && wu >= d__[1]) {
		*m = 1;
		w[1] = d__[1];
	    }
	}
	if (wantz && ! zquery) {
	    z__[z_dim1 + 1] = 1.;
	    isuppz[1] = 1;
	    isuppz[2] = 1;
	}
	return 0;
    }

    if (*n == 2) {
	if (! wantz) {
	    dlae2_(&d__[1], &e[1], &d__[2], &r1, &r2);
	} else if (wantz && ! zquery) {
	    dlaev2_(&d__[1], &e[1], &d__[2], &r1, &r2, &cs, &sn);
	}
	if (alleig || valeig && r2 > wl && r2 <= wu || indeig && iil == 1) {
	    ++(*m);
	    w[*m] = r2;
	    if (wantz && ! zquery) {
		z__[*m * z_dim1 + 1] = -sn;
		z__[*m * z_dim1 + 2] = cs;
/*              Note: At most one of SN and CS can be zero. */
		if (sn != 0.) {
		    if (cs != 0.) {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[(*m << 1) - 1] = 2;
		    } else {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[(*m << 1) - 1] = 1;
		    }
		} else {
		    isuppz[(*m << 1) - 1] = 2;
		    isuppz[*m * 2] = 2;
		}
	    }
	}
	if (alleig || valeig && r1 > wl && r1 <= wu || indeig && iiu == 2) {
	    ++(*m);
	    w[*m] = r1;
	    if (wantz && ! zquery) {
		z__[*m * z_dim1 + 1] = cs;
		z__[*m * z_dim1 + 2] = sn;
/*              Note: At most one of SN and CS can be zero. */
		if (sn != 0.) {
		    if (cs != 0.) {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[(*m << 1) - 1] = 2;
		    } else {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[(*m << 1) - 1] = 1;
		    }
		} else {
		    isuppz[(*m << 1) - 1] = 2;
		    isuppz[*m * 2] = 2;
		}
	    }
	}
	return 0;
    }
/*     Continue with general N */
    indgrs = 1;
    inderr = (*n << 1) + 1;
    indgp = *n * 3 + 1;
    indd = (*n << 2) + 1;
    inde2 = *n * 5 + 1;
    indwrk = *n * 6 + 1;

    iinspl = 1;
    iindbl = *n + 1;
    iindw = (*n << 1) + 1;
    iindwk = *n * 3 + 1;

/*     Scale matrix to allowable range, if necessary. */
/*     The allowable range is related to the PIVMIN parameter; see the */
/*     comments in DLARRD.  The preference for scaling small values */
/*     up is heuristic; we expect users' matrices not to be close to the */
/*     RMAX threshold. */

    scale = 1.;
    tnrm = dlanst_("M", n, &d__[1], &e[1]);
    if (tnrm > 0. && tnrm < rmin) {
	scale = rmin / tnrm;
    } else if (tnrm > rmax) {
	scale = rmax / tnrm;
    }
    if (scale != 1.) {
	dscal_(n, &scale, &d__[1], &c__1);
	i__1 = *n - 1;
	dscal_(&i__1, &scale, &e[1], &c__1);
	tnrm *= scale;
	if (valeig) {
/*           If eigenvalues in interval have to be found, */
/*           scale (WL, WU] accordingly */
	    wl *= scale;
	    wu *= scale;
	}
    }

/*     Compute the desired eigenvalues of the tridiagonal after splitting */
/*     into smaller subblocks if the corresponding off-diagonal elements */
/*     are small */
/*     THRESH is the splitting parameter for DLARRE */
/*     A negative THRESH forces the old splitting criterion based on the */
/*     size of the off-diagonal. A positive THRESH switches to splitting */
/*     which preserves relative accuracy. */

    if (*tryrac) {
/*        Test whether the matrix warrants the more expensive relative approach. */
	dlarrr_(n, &d__[1], &e[1], &iinfo);
    } else {
/*        The user does not care about relative accurately eigenvalues */
	iinfo = -1;
    }
/*     Set the splitting criterion */
    if (iinfo == 0) {
	thresh = eps;
    } else {
	thresh = -eps;
/*        relative accuracy is desired but T does not guarantee it */
	*tryrac = FALSE_;
    }

    if (*tryrac) {
/*        Copy original diagonal, needed to guarantee relative accuracy */
	dcopy_(n, &d__[1], &c__1, &work[indd], &c__1);
    }
/*     Store the squares of the offdiagonal values of T */
    i__1 = *n - 1;
    for (j = 1; j <= i__1; ++j) {
/* Computing 2nd power */
	d__1 = e[j];
	work[inde2 + j - 1] = d__1 * d__1;
/* L5: */
    }
/*     Set the tolerance parameters for bisection */
    if (! wantz) {
/*        DLARRE computes the eigenvalues to full precision. */
	rtol1 = eps * 4.;
	rtol2 = eps * 4.;
    } else {
/*        DLARRE computes the eigenvalues to less than full precision. */
/*        DLARRV will refine the eigenvalue approximations, and we can */
/*        need less accurate initial bisection in DLARRE. */
/*        Note: these settings do only affect the subset case and DLARRE */
	rtol1 = sqrt(eps);
/* Computing MAX */
	d__1 = sqrt(eps) * .005, d__2 = eps * 4.;
	rtol2 = max(d__1,d__2);
    }
    dlarre_(range, n, &wl, &wu, &iil, &iiu, &d__[1], &e[1], &work[inde2], &
	    rtol1, &rtol2, &thresh, &nsplit, &iwork[iinspl], m, &w[1], &work[
	    inderr], &work[indgp], &iwork[iindbl], &iwork[iindw], &work[
	    indgrs], &pivmin, &work[indwrk], &iwork[iindwk], &iinfo);
    if (iinfo != 0) {
	*info = abs(iinfo) + 10;
	return 0;
    }
/*     Note that if RANGE .NE. 'V', DLARRE computes bounds on the desired */
/*     part of the spectrum. All desired eigenvalues are contained in */
/*     (WL,WU] */
    if (wantz) {

/*        Compute the desired eigenvectors corresponding to the computed */
/*        eigenvalues */

	dlarrv_(n, &wl, &wu, &d__[1], &e[1], &pivmin, &iwork[iinspl], m, &
		c__1, m, &c_b18, &rtol1, &rtol2, &w[1], &work[inderr], &work[
		indgp], &iwork[iindbl], &iwork[iindw], &work[indgrs], &z__[
		z_offset], ldz, &isuppz[1], &work[indwrk], &iwork[iindwk], &
		iinfo);
	if (iinfo != 0) {
	    *info = abs(iinfo) + 20;
	    return 0;
	}
    } else {
/*        DLARRE computes eigenvalues of the (shifted) root representation */
/*        DLARRV returns the eigenvalues of the unshifted matrix. */
/*        However, if the eigenvectors are not desired by the user, we need */
/*        to apply the corresponding shifts from DLARRE to obtain the */
/*        eigenvalues of the original matrix. */
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    itmp = iwork[iindbl + j - 1];
	    w[j] += e[iwork[iinspl + itmp - 1]];
/* L20: */
	}
    }

    if (*tryrac) {
/*        Refine computed eigenvalues so that they are relatively accurate */
/*        with respect to the original matrix T. */
	ibegin = 1;
	wbegin = 1;
	i__1 = iwork[iindbl + *m - 1];
	for (jblk = 1; jblk <= i__1; ++jblk) {
	    iend = iwork[iinspl + jblk - 1];
	    in = iend - ibegin + 1;
	    wend = wbegin - 1;
/*           check if any eigenvalues have to be refined in this block */
L36:
	    if (wend < *m) {
		if (iwork[iindbl + wend] == jblk) {
		    ++wend;
		    goto L36;
		}
	    }
	    if (wend < wbegin) {
		ibegin = iend + 1;
		goto L39;
	    }
	    offset = iwork[iindw + wbegin - 1] - 1;
	    ifirst = iwork[iindw + wbegin - 1];
	    ilast = iwork[iindw + wend - 1];
	    rtol2 = eps * 4.;
	    dlarrj_(&in, &work[indd + ibegin - 1], &work[inde2 + ibegin - 1], 
		    &ifirst, &ilast, &rtol2, &offset, &w[wbegin], &work[
		    inderr + wbegin - 1], &work[indwrk], &iwork[iindwk], &
		    pivmin, &tnrm, &iinfo);
	    ibegin = iend + 1;
	    wbegin = wend + 1;
L39:
	    ;
	}
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

    if (scale != 1.) {
	d__1 = 1. / scale;
	dscal_(m, &d__1, &w[1], &c__1);
    }

/*     If eigenvalues are not in increasing order, then sort them, */
/*     possibly along with eigenvectors. */

    if (nsplit > 1) {
	if (! wantz) {
	    dlasrt_("I", m, &w[1], &iinfo);
	    if (iinfo != 0) {
		*info = 3;
		return 0;
	    }
	} else {
	    i__1 = *m - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__ = 0;
		tmp = w[j];
		i__2 = *m;
		for (jj = j + 1; jj <= i__2; ++jj) {
		    if (w[jj] < tmp) {
			i__ = jj;
			tmp = w[jj];
		    }
/* L50: */
		}
		if (i__ != 0) {
		    w[i__] = w[j];
		    w[j] = tmp;
		    if (wantz) {
			dswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[j * 
				z_dim1 + 1], &c__1);
			itmp = isuppz[(i__ << 1) - 1];
			isuppz[(i__ << 1) - 1] = isuppz[(j << 1) - 1];
			isuppz[(j << 1) - 1] = itmp;
			itmp = isuppz[i__ * 2];
			isuppz[i__ * 2] = isuppz[j * 2];
			isuppz[j * 2] = itmp;
		    }
		}
/* L60: */
	    }
	}
    }


    work[1] = (doublereal) lwmin;
    iwork[1] = liwmin;
    return 0;

/*     End of DSTEMR */

} /* dstemr_ */
