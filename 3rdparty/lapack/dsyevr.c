/* dsyevr.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "clapack.h"


/* Table of constant values */

static integer c__10 = 10;
static integer c__1 = 1;
static integer c__2 = 2;
static integer c__3 = 3;
static integer c__4 = 4;
static integer c_n1 = -1;

/* Subroutine */ int dsyevr_(char *jobz, char *range, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *vl, doublereal *vu, integer *
	il, integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, z_dim1, z_offset, i__1, i__2;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j, nb, jj;
    doublereal eps, vll, vuu, tmp1;
    integer indd, inde;
    doublereal anrm;
    integer imax;
    doublereal rmin, rmax;
    integer inddd, indee;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    doublereal sigma;
    extern logical lsame_(char *, char *);
    integer iinfo;
    char order[1];
    integer indwk;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *), dswap_(integer *, doublereal *, integer 
	    *, doublereal *, integer *);
    integer lwmin;
    logical lower, wantz;
    extern doublereal dlamch_(char *);
    logical alleig, indeig;
    integer iscale, ieeeok, indibl, indifl;
    logical valeig;
    doublereal safmin;
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal abstll, bignum;
    integer indtau, indisp;
    extern /* Subroutine */ int dstein_(integer *, doublereal *, doublereal *, 
	     integer *, doublereal *, integer *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, integer *, integer *), 
	    dsterf_(integer *, doublereal *, doublereal *, integer *);
    integer indiwo, indwkn;
    extern doublereal dlansy_(char *, char *, integer *, doublereal *, 
	    integer *, doublereal *);
    extern /* Subroutine */ int dstebz_(char *, char *, integer *, doublereal 
	    *, doublereal *, integer *, integer *, doublereal *, doublereal *, 
	     doublereal *, integer *, integer *, doublereal *, integer *, 
	    integer *, doublereal *, integer *, integer *), 
	    dstemr_(char *, char *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, integer *, integer *, 
	    logical *, doublereal *, integer *, integer *, integer *, integer 
	    *);
    integer liwmin;
    logical tryrac;
    extern /* Subroutine */ int dormtr_(char *, char *, char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *, integer *);
    integer llwrkn, llwork, nsplit;
    doublereal smlnum;
    extern /* Subroutine */ int dsytrd_(char *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *, 
	     integer *, integer *);
    integer lwkopt;
    logical lquery;


/*  -- LAPACK driver routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DSYEVR computes selected eigenvalues and, optionally, eigenvectors */
/*  of a real symmetric matrix A.  Eigenvalues and eigenvectors can be */
/*  selected by specifying either a range of values or a range of */
/*  indices for the desired eigenvalues. */

/*  DSYEVR first reduces the matrix A to tridiagonal form T with a call */
/*  to DSYTRD.  Then, whenever possible, DSYEVR calls DSTEMR to compute */
/*  the eigenspectrum using Relatively Robust Representations.  DSTEMR */
/*  computes eigenvalues by the dqds algorithm, while orthogonal */
/*  eigenvectors are computed from various "good" L D L^T representations */
/*  (also known as Relatively Robust Representations). Gram-Schmidt */
/*  orthogonalization is avoided as far as possible. More specifically, */
/*  the various steps of the algorithm are as follows. */

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

/*  The desired accuracy of the output can be specified by the input */
/*  parameter ABSTOL. */

/*  For more details, see DSTEMR's documentation and: */
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


/*  Note 1 : DSYEVR calls DSTEMR when the full spectrum is requested */
/*  on machines which conform to the ieee-754 floating point standard. */
/*  DSYEVR calls DSTEBZ and SSTEIN on non-ieee machines and */
/*  when partial spectrum requests are made. */

/*  Normal execution of DSTEMR may create NaNs and infinities and */
/*  hence may abort due to a floating point exception in environments */
/*  which do not handle NaNs and infinities in the ieee standard default */
/*  manner. */

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
/* ********* For RANGE = 'V' or 'I' and IU - IL < N - 1, DSTEBZ and */
/* ********* DSTEIN are called */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  Upper triangle of A is stored; */
/*          = 'L':  Lower triangle of A is stored. */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA, N) */
/*          On entry, the symmetric matrix A.  If UPLO = 'U', the */
/*          leading N-by-N upper triangular part of A contains the */
/*          upper triangular part of the matrix A.  If UPLO = 'L', */
/*          the leading N-by-N lower triangular part of A contains */
/*          the lower triangular part of the matrix A. */
/*          On exit, the lower triangle (if UPLO='L') or the upper */
/*          triangle (if UPLO='U') of A, including the diagonal, is */
/*          destroyed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  VL      (input) DOUBLE PRECISION */
/*  VU      (input) DOUBLE PRECISION */
/*          If RANGE='V', the lower and upper bounds of the interval to */
/*          be searched for eigenvalues. VL < VU. */
/*          Not referenced if RANGE = 'A' or 'I'. */

/*  IL      (input) INTEGER */
/*  IU      (input) INTEGER */
/*          If RANGE='I', the indices (in ascending order) of the */
/*          smallest and largest eigenvalues to be returned. */
/*          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0. */
/*          Not referenced if RANGE = 'A' or 'V'. */

/*  ABSTOL  (input) DOUBLE PRECISION */
/*          The absolute error tolerance for the eigenvalues. */
/*          An approximate eigenvalue is accepted as converged */
/*          when it is determined to lie in an interval [a,b] */
/*          of width less than or equal to */

/*                  ABSTOL + EPS *   max( |a|,|b| ) , */

/*          where EPS is the machine precision.  If ABSTOL is less than */
/*          or equal to zero, then  EPS*|T|  will be used in its place, */
/*          where |T| is the 1-norm of the tridiagonal matrix obtained */
/*          by reducing A to tridiagonal form. */

/*          See "Computing Small Singular Values of Bidiagonal Matrices */
/*          with Guaranteed High Relative Accuracy," by Demmel and */
/*          Kahan, LAPACK Working Note #3. */

/*          If high relative accuracy is important, set ABSTOL to */
/*          DLAMCH( 'Safe minimum' ).  Doing so will guarantee that */
/*          eigenvalues are computed to high relative accuracy when */
/*          possible in future releases.  The current code does not */
/*          make any guarantees about high relative accuracy, but */
/*          future releases will. See J. Barlow and J. Demmel, */
/*          "Computing Accurate Eigensystems of Scaled Diagonally */
/*          Dominant Matrices", LAPACK Working Note #7, for a discussion */
/*          of which matrices define their eigenvalues to high relative */
/*          accuracy. */

/*  M       (output) INTEGER */
/*          The total number of eigenvalues found.  0 <= M <= N. */
/*          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1. */

/*  W       (output) DOUBLE PRECISION array, dimension (N) */
/*          The first M elements contain the selected eigenvalues in */
/*          ascending order. */

/*  Z       (output) DOUBLE PRECISION array, dimension (LDZ, max(1,M)) */
/*          If JOBZ = 'V', then if INFO = 0, the first M columns of Z */
/*          contain the orthonormal eigenvectors of the matrix A */
/*          corresponding to the selected eigenvalues, with the i-th */
/*          column of Z holding the eigenvector associated with W(i). */
/*          If JOBZ = 'N', then Z is not referenced. */
/*          Note: the user must ensure that at least max(1,M) columns are */
/*          supplied in the array Z; if RANGE = 'V', the exact value of M */
/*          is not known in advance and an upper bound must be used. */
/*          Supplying N columns is always safe. */

/*  LDZ     (input) INTEGER */
/*          The leading dimension of the array Z.  LDZ >= 1, and if */
/*          JOBZ = 'V', LDZ >= max(1,N). */

/*  ISUPPZ  (output) INTEGER array, dimension ( 2*max(1,M) ) */
/*          The support of the eigenvectors in Z, i.e., the indices */
/*          indicating the nonzero elements in Z. The i-th eigenvector */
/*          is nonzero only in elements ISUPPZ( 2*i-1 ) through */
/*          ISUPPZ( 2*i ). */
/* ********* Implemented only for RANGE = 'A' or 'I' and IU - IL = N - 1 */

/*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK.  LWORK >= max(1,26*N). */
/*          For optimal efficiency, LWORK >= (NB+6)*N, */
/*          where NB is the max of the blocksize for DSYTRD and DORMTR */
/*          returned by ILAENV. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  IWORK   (workspace/output) INTEGER array, dimension (MAX(1,LIWORK)) */
/*          On exit, if INFO = 0, IWORK(1) returns the optimal LWORK. */

/*  LIWORK  (input) INTEGER */
/*          The dimension of the array IWORK.  LIWORK >= max(1,10*N). */

/*          If LIWORK = -1, then a workspace query is assumed; the */
/*          routine only calculates the optimal size of the IWORK array, */
/*          returns this value as the first entry of the IWORK array, and */
/*          no error message related to LIWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  Internal error */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Inderjit Dhillon, IBM Almaden, USA */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Ken Stanley, Computer Science Division, University of */
/*       California at Berkeley, USA */
/*     Jason Riedy, Computer Science Division, University of */
/*       California at Berkeley, USA */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --isuppz;
    --work;
    --iwork;

    /* Function Body */
    ieeeok = ilaenv_(&c__10, "DSYEVR", "N", &c__1, &c__2, &c__3, &c__4);

    lower = lsame_(uplo, "L");
    wantz = lsame_(jobz, "V");
    alleig = lsame_(range, "A");
    valeig = lsame_(range, "V");
    indeig = lsame_(range, "I");

    lquery = *lwork == -1 || *liwork == -1;

/* Computing MAX */
    i__1 = 1, i__2 = *n * 26;
    lwmin = max(i__1,i__2);
/* Computing MAX */
    i__1 = 1, i__2 = *n * 10;
    liwmin = max(i__1,i__2);

    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (alleig || valeig || indeig)) {
	*info = -2;
    } else if (! (lower || lsame_(uplo, "U"))) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*lda < max(1,*n)) {
	*info = -6;
    } else {
	if (valeig) {
	    if (*n > 0 && *vu <= *vl) {
		*info = -8;
	    }
	} else if (indeig) {
	    if (*il < 1 || *il > max(1,*n)) {
		*info = -9;
	    } else if (*iu < min(*n,*il) || *iu > *n) {
		*info = -10;
	    }
	}
    }
    if (*info == 0) {
	if (*ldz < 1 || wantz && *ldz < *n) {
	    *info = -15;
	} else if (*lwork < lwmin && ! lquery) {
	    *info = -18;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -20;
	}
    }

    if (*info == 0) {
	nb = ilaenv_(&c__1, "DSYTRD", uplo, n, &c_n1, &c_n1, &c_n1);
/* Computing MAX */
	i__1 = nb, i__2 = ilaenv_(&c__1, "DORMTR", uplo, n, &c_n1, &c_n1, &
		c_n1);
	nb = max(i__1,i__2);
/* Computing MAX */
	i__1 = (nb + 1) * *n;
	lwkopt = max(i__1,lwmin);
	work[1] = (doublereal) lwkopt;
	iwork[1] = liwmin;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYEVR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    *m = 0;
    if (*n == 0) {
	work[1] = 1.;
	return 0;
    }

    if (*n == 1) {
	work[1] = 7.;
	if (alleig || indeig) {
	    *m = 1;
	    w[1] = a[a_dim1 + 1];
	} else {
	    if (*vl < a[a_dim1 + 1] && *vu >= a[a_dim1 + 1]) {
		*m = 1;
		w[1] = a[a_dim1 + 1];
	    }
	}
	if (wantz) {
	    z__[z_dim1 + 1] = 1.;
	}
	return 0;
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

/*     Scale matrix to allowable range, if necessary. */

    iscale = 0;
    abstll = *abstol;
    vll = *vl;
    vuu = *vu;
    anrm = dlansy_("M", uplo, n, &a[a_offset], lda, &work[1]);
    if (anrm > 0. && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	if (lower) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j + 1;
		dscal_(&i__2, &sigma, &a[j + j * a_dim1], &c__1);
/* L10: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		dscal_(&j, &sigma, &a[j * a_dim1 + 1], &c__1);
/* L20: */
	    }
	}
	if (*abstol > 0.) {
	    abstll = *abstol * sigma;
	}
	if (valeig) {
	    vll = *vl * sigma;
	    vuu = *vu * sigma;
	}
    }
/*     Initialize indices into workspaces.  Note: The IWORK indices are */
/*     used only if DSTERF or DSTEMR fail. */
/*     WORK(INDTAU:INDTAU+N-1) stores the scalar factors of the */
/*     elementary reflectors used in DSYTRD. */
    indtau = 1;
/*     WORK(INDD:INDD+N-1) stores the tridiagonal's diagonal entries. */
    indd = indtau + *n;
/*     WORK(INDE:INDE+N-1) stores the off-diagonal entries of the */
/*     tridiagonal matrix from DSYTRD. */
    inde = indd + *n;
/*     WORK(INDDD:INDDD+N-1) is a copy of the diagonal entries over */
/*     -written by DSTEMR (the DSTERF path copies the diagonal to W). */
    inddd = inde + *n;
/*     WORK(INDEE:INDEE+N-1) is a copy of the off-diagonal entries over */
/*     -written while computing the eigenvalues in DSTERF and DSTEMR. */
    indee = inddd + *n;
/*     INDWK is the starting offset of the left-over workspace, and */
/*     LLWORK is the remaining workspace size. */
    indwk = indee + *n;
    llwork = *lwork - indwk + 1;
/*     IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and */
/*     stores the block indices of each of the M<=N eigenvalues. */
    indibl = 1;
/*     IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and */
/*     stores the starting and finishing indices of each block. */
    indisp = indibl + *n;
/*     IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors */
/*     that corresponding to eigenvectors that fail to converge in */
/*     DSTEIN.  This information is discarded; if any fail, the driver */
/*     returns INFO > 0. */
    indifl = indisp + *n;
/*     INDIWO is the offset of the remaining integer workspace. */
    indiwo = indisp + *n;

/*     Call DSYTRD to reduce symmetric matrix to tridiagonal form. */

    dsytrd_(uplo, n, &a[a_offset], lda, &work[indd], &work[inde], &work[
	    indtau], &work[indwk], &llwork, &iinfo);

/*     If all eigenvalues are desired */
/*     then call DSTERF or DSTEMR and DORMTR. */

    if ((alleig || indeig && *il == 1 && *iu == *n) && ieeeok == 1) {
	if (! wantz) {
	    dcopy_(n, &work[indd], &c__1, &w[1], &c__1);
	    i__1 = *n - 1;
	    dcopy_(&i__1, &work[inde], &c__1, &work[indee], &c__1);
	    dsterf_(n, &w[1], &work[indee], info);
	} else {
	    i__1 = *n - 1;
	    dcopy_(&i__1, &work[inde], &c__1, &work[indee], &c__1);
	    dcopy_(n, &work[indd], &c__1, &work[inddd], &c__1);

	    if (*abstol <= *n * 2. * eps) {
		tryrac = TRUE_;
	    } else {
		tryrac = FALSE_;
	    }
	    dstemr_(jobz, "A", n, &work[inddd], &work[indee], vl, vu, il, iu, 
		    m, &w[1], &z__[z_offset], ldz, n, &isuppz[1], &tryrac, &
		    work[indwk], lwork, &iwork[1], liwork, info);



/*        Apply orthogonal matrix used in reduction to tridiagonal */
/*        form to eigenvectors returned by DSTEIN. */

	    if (wantz && *info == 0) {
		indwkn = inde;
		llwrkn = *lwork - indwkn + 1;
		dormtr_("L", uplo, "N", n, m, &a[a_offset], lda, &work[indtau]
, &z__[z_offset], ldz, &work[indwkn], &llwrkn, &iinfo);
	    }
	}


	if (*info == 0) {
/*           Everything worked.  Skip DSTEBZ/DSTEIN.  IWORK(:) are */
/*           undefined. */
	    *m = *n;
	    goto L30;
	}
	*info = 0;
    }

/*     Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN. */
/*     Also call DSTEBZ and DSTEIN if DSTEMR fails. */

    if (wantz) {
	*(unsigned char *)order = 'B';
    } else {
	*(unsigned char *)order = 'E';
    }
    dstebz_(range, order, n, &vll, &vuu, il, iu, &abstll, &work[indd], &work[
	    inde], m, &nsplit, &w[1], &iwork[indibl], &iwork[indisp], &work[
	    indwk], &iwork[indiwo], info);

    if (wantz) {
	dstein_(n, &work[indd], &work[inde], m, &w[1], &iwork[indibl], &iwork[
		indisp], &z__[z_offset], ldz, &work[indwk], &iwork[indiwo], &
		iwork[indifl], info);

/*        Apply orthogonal matrix used in reduction to tridiagonal */
/*        form to eigenvectors returned by DSTEIN. */

	indwkn = inde;
	llwrkn = *lwork - indwkn + 1;
	dormtr_("L", uplo, "N", n, m, &a[a_offset], lda, &work[indtau], &z__[
		z_offset], ldz, &work[indwkn], &llwrkn, &iinfo);
    }

/*     If matrix was scaled, then rescale eigenvalues appropriately. */

/*  Jump here if DSTEMR/DSTEIN succeeded. */
L30:
    if (iscale == 1) {
	if (*info == 0) {
	    imax = *m;
	} else {
	    imax = *info - 1;
	}
	d__1 = 1. / sigma;
	dscal_(&imax, &d__1, &w[1], &c__1);
    }

/*     If eigenvalues are not in order, then sort them, along with */
/*     eigenvectors.  Note: We do not sort the IFAIL portion of IWORK. */
/*     It may not be initialized (if DSTEMR/DSTEIN succeeded), and we do */
/*     not return this detailed information to the user. */

    if (wantz) {
	i__1 = *m - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__ = 0;
	    tmp1 = w[j];
	    i__2 = *m;
	    for (jj = j + 1; jj <= i__2; ++jj) {
		if (w[jj] < tmp1) {
		    i__ = jj;
		    tmp1 = w[jj];
		}
/* L40: */
	    }

	    if (i__ != 0) {
		w[i__] = w[j];
		w[j] = tmp1;
		dswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[j * z_dim1 + 1], 
			 &c__1);
	    }
/* L50: */
	}
    }

/*     Set WORK(1) to optimal workspace size. */

    work[1] = (doublereal) lwkopt;
    iwork[1] = liwmin;

    return 0;

/*     End of DSYEVR */

} /* dsyevr_ */
