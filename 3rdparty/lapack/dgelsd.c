#include "clapack.h"

/* Table of constant values */

static integer c__6 = 6;
static integer c_n1 = -1;
static integer c__9 = 9;
static integer c__0 = 0;
static integer c__1 = 1;
static doublereal c_b82 = 0.;

/* Subroutine */ int dgelsd_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	s, doublereal *rcond, integer *rank, doublereal *work, integer *lwork, 
	 integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3, i__4;

    /* Builtin functions */
    double log(doublereal);

    /* Local variables */
    integer ie, il, mm;
    doublereal eps, anrm, bnrm;
    integer itau, nlvl, iascl, ibscl;
    doublereal sfmin;
    integer minmn, maxmn, itaup, itauq, mnthr, nwork;
    extern /* Subroutine */ int dlabad_(doublereal *, doublereal *), dgebrd_(
	    integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *, 
	     integer *);
    extern doublereal dlamch_(char *), dlange_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *);
    extern /* Subroutine */ int dgelqf_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *, integer *), 
	    dlalsd_(char *, integer *, integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, integer *, integer *), dlascl_(char *, 
	    integer *, integer *, doublereal *, doublereal *, integer *, 
	    integer *, doublereal *, integer *, integer *), dgeqrf_(
	    integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, integer *, integer *), dlacpy_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, integer *), dlaset_(char *, integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, integer *), xerbla_(char *, 
	    integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    doublereal bignum;
    extern /* Subroutine */ int dormbr_(char *, char *, char *, integer *, 
	    integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, integer *);
    integer wlalsd;
    extern /* Subroutine */ int dormlq_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *, integer *);
    integer ldwork;
    extern /* Subroutine */ int dormqr_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *, integer *);
    integer minwrk, maxwrk;
    doublereal smlnum;
    logical lquery;
    integer smlsiz;


/*  -- LAPACK driver routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGELSD computes the minimum-norm solution to a real linear least */
/*  squares problem: */
/*      minimize 2-norm(| b - A*x |) */
/*  using the singular value decomposition (SVD) of A. A is an M-by-N */
/*  matrix which may be rank-deficient. */

/*  Several right hand side vectors b and solution vectors x can be */
/*  handled in a single call; they are stored as the columns of the */
/*  M-by-NRHS right hand side matrix B and the N-by-NRHS solution */
/*  matrix X. */

/*  The problem is solved in three steps: */
/*  (1) Reduce the coefficient matrix A to bidiagonal form with */
/*      Householder transformations, reducing the original problem */
/*      into a "bidiagonal least squares problem" (BLS) */
/*  (2) Solve the BLS using a divide and conquer approach. */
/*  (3) Apply back all the Householder tranformations to solve */
/*      the original least squares problem. */

/*  The effective rank of A is determined by treating as zero those */
/*  singular values which are less than RCOND times the largest singular */
/*  value. */

/*  The divide and conquer algorithm makes very mild assumptions about */
/*  floating point arithmetic. It will work on machines with a guard */
/*  digit in add/subtract, or on those binary machines without guard */
/*  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or */
/*  Cray-2. It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of A. M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of A. N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrices B and X. NRHS >= 0. */

/*  A       (input) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix A. */
/*          On exit, A has been destroyed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS) */
/*          On entry, the M-by-NRHS right hand side matrix B. */
/*          On exit, B is overwritten by the N-by-NRHS solution */
/*          matrix X.  If m >= n and RANK = n, the residual */
/*          sum-of-squares for the solution in the i-th column is given */
/*          by the sum of squares of elements n+1:m in that column. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B. LDB >= max(1,max(M,N)). */

/*  S       (output) DOUBLE PRECISION array, dimension (min(M,N)) */
/*          The singular values of A in decreasing order. */
/*          The condition number of A in the 2-norm = S(1)/S(min(m,n)). */

/*  RCOND   (input) DOUBLE PRECISION */
/*          RCOND is used to determine the effective rank of A. */
/*          Singular values S(i) <= RCOND*S(1) are treated as zero. */
/*          If RCOND < 0, machine precision is used instead. */

/*  RANK    (output) INTEGER */
/*          The effective rank of A, i.e., the number of singular values */
/*          which are greater than RCOND*S(1). */

/*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK must be at least 1. */
/*          The exact minimum amount of workspace needed depends on M, */
/*          N and NRHS. As long as LWORK is at least */
/*              12*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)**2, */
/*          if M is greater than or equal to N or */
/*              12*M + 2*M*SMLSIZ + 8*M*NLVL + M*NRHS + (SMLSIZ+1)**2, */
/*          if M is less than N, the code will execute correctly. */
/*          SMLSIZ is returned by ILAENV and is equal to the maximum */
/*          size of the subproblems at the bottom of the computation */
/*          tree (usually about 25), and */
/*             NLVL = MAX( 0, INT( LOG_2( MIN( M,N )/(SMLSIZ+1) ) ) + 1 ) */
/*          For good performance, LWORK should generally be larger. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  IWORK   (workspace) INTEGER array, dimension (MAX(1,LIWORK)) */
/*          LIWORK >= 3 * MINMN * NLVL + 11 * MINMN, */
/*          where MINMN = MIN( M,N ). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  the algorithm for computing the SVD failed to converge; */
/*                if INFO = i, i off-diagonal elements of an intermediate */
/*                bidiagonal form did not converge to zero. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Ren-Cang Li, Computer Science Division, University of */
/*       California at Berkeley, USA */
/*     Osni Marques, LBNL/NERSC, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --s;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(*m,*n);
    maxmn = max(*m,*n);
    mnthr = ilaenv_(&c__6, "DGELSD", " ", m, n, nrhs, &c_n1);
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*ldb < max(1,maxmn)) {
	*info = -7;
    }

    smlsiz = ilaenv_(&c__9, "DGELSD", " ", &c__0, &c__0, &c__0, &c__0);

/*     Compute workspace. */
/*     (Note: Comments in the code beginning "Workspace:" describe the */
/*     minimal amount of workspace needed at that point in the code, */
/*     as well as the preferred amount for good performance. */
/*     NB refers to the optimal block size for the immediately */
/*     following subroutine, as returned by ILAENV.) */

    minwrk = 1;
    minmn = max(1,minmn);
/* Computing MAX */
    i__1 = (integer) (log((doublereal) minmn / (doublereal) (smlsiz + 1)) / 
	    log(2.)) + 1;
    nlvl = max(i__1,0);

    if (*info == 0) {
	maxwrk = 0;
	mm = *m;
	if (*m >= *n && *m >= mnthr) {

/*           Path 1a - overdetermined, with many more rows than columns. */

	    mm = *n;
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + *n * ilaenv_(&c__1, "DGEQRF", " ", m, 
		    n, &c_n1, &c_n1);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n + *nrhs * ilaenv_(&c__1, "DORMQR", "LT", 
		    m, nrhs, n, &c_n1);
	    maxwrk = max(i__1,i__2);
	}
	if (*m >= *n) {

/*           Path 1 - overdetermined or exactly determined. */

/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n * 3 + (mm + *n) * ilaenv_(&c__1, "DGEBRD"
, " ", &mm, n, &c_n1, &c_n1);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n * 3 + *nrhs * ilaenv_(&c__1, "DORMBR", 
		    "QLT", &mm, nrhs, n, &c_n1);
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n * 3 + (*n - 1) * ilaenv_(&c__1, "DORMBR", 
		     "PLN", n, nrhs, n, &c_n1);
	    maxwrk = max(i__1,i__2);
/* Computing 2nd power */
	    i__1 = smlsiz + 1;
	    wlalsd = *n * 9 + (*n << 1) * smlsiz + (*n << 3) * nlvl + *n * *
		    nrhs + i__1 * i__1;
/* Computing MAX */
	    i__1 = maxwrk, i__2 = *n * 3 + wlalsd;
	    maxwrk = max(i__1,i__2);
/* Computing MAX */
	    i__1 = *n * 3 + mm, i__2 = *n * 3 + *nrhs, i__1 = max(i__1,i__2), 
		    i__2 = *n * 3 + wlalsd;
	    minwrk = max(i__1,i__2);
	}
	if (*n > *m) {
/* Computing 2nd power */
	    i__1 = smlsiz + 1;
	    wlalsd = *m * 9 + (*m << 1) * smlsiz + (*m << 3) * nlvl + *m * *
		    nrhs + i__1 * i__1;
	    if (*n >= mnthr) {

/*              Path 2a - underdetermined, with many more columns */
/*              than rows. */

		maxwrk = *m + *m * ilaenv_(&c__1, "DGELQF", " ", m, n, &c_n1, 
			&c_n1);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + (*m << 2) + (*m << 1) * 
			ilaenv_(&c__1, "DGEBRD", " ", m, m, &c_n1, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + (*m << 2) + *nrhs * ilaenv_(&
			c__1, "DORMBR", "QLT", m, nrhs, m, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + (*m << 2) + (*m - 1) * 
			ilaenv_(&c__1, "DORMBR", "PLN", m, nrhs, m, &c_n1);
		maxwrk = max(i__1,i__2);
		if (*nrhs > 1) {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + *m + *m * *nrhs;
		    maxwrk = max(i__1,i__2);
		} else {
/* Computing MAX */
		    i__1 = maxwrk, i__2 = *m * *m + (*m << 1);
		    maxwrk = max(i__1,i__2);
		}
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m + *nrhs * ilaenv_(&c__1, "DORMLQ", 
			"LT", n, nrhs, m, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * *m + (*m << 2) + wlalsd;
		maxwrk = max(i__1,i__2);
	    } else {

/*              Path 2 - remaining underdetermined cases. */

		maxwrk = *m * 3 + (*n + *m) * ilaenv_(&c__1, "DGEBRD", " ", m, 
			 n, &c_n1, &c_n1);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * 3 + *nrhs * ilaenv_(&c__1, "DORMBR"
, "QLT", m, nrhs, n, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR", 
			"PLN", n, nrhs, m, &c_n1);
		maxwrk = max(i__1,i__2);
/* Computing MAX */
		i__1 = maxwrk, i__2 = *m * 3 + wlalsd;
		maxwrk = max(i__1,i__2);
	    }
/* Computing MAX */
	    i__1 = *m * 3 + *nrhs, i__2 = *m * 3 + *m, i__1 = max(i__1,i__2), 
		    i__2 = *m * 3 + wlalsd;
	    minwrk = max(i__1,i__2);
	}
	minwrk = min(minwrk,maxwrk);
	work[1] = (doublereal) maxwrk;
	if (*lwork < minwrk && ! lquery) {
	    *info = -12;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGELSD", &i__1);
	return 0;
    } else if (lquery) {
	goto L10;
    }

/*     Quick return if possible. */

    if (*m == 0 || *n == 0) {
	*rank = 0;
	return 0;
    }

/*     Get machine parameters. */

    eps = dlamch_("P");
    sfmin = dlamch_("S");
    smlnum = sfmin / eps;
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);

/*     Scale A if max entry outside range [SMLNUM,BIGNUM]. */

    anrm = dlange_("M", m, n, &a[a_offset], lda, &work[1]);
    iascl = 0;
    if (anrm > 0. && anrm < smlnum) {

/*        Scale matrix norm up to SMLNUM. */

	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, 
		info);
	iascl = 1;
    } else if (anrm > bignum) {

/*        Scale matrix norm down to BIGNUM. */

	dlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, 
		info);
	iascl = 2;
    } else if (anrm == 0.) {

/*        Matrix all zero. Return zero solution. */

	i__1 = max(*m,*n);
	dlaset_("F", &i__1, nrhs, &c_b82, &c_b82, &b[b_offset], ldb);
	dlaset_("F", &minmn, &c__1, &c_b82, &c_b82, &s[1], &c__1);
	*rank = 0;
	goto L10;
    }

/*     Scale B if max entry outside range [SMLNUM,BIGNUM]. */

    bnrm = dlange_("M", m, nrhs, &b[b_offset], ldb, &work[1]);
    ibscl = 0;
    if (bnrm > 0. && bnrm < smlnum) {

/*        Scale matrix norm up to SMLNUM. */

	dlascl_("G", &c__0, &c__0, &bnrm, &smlnum, m, nrhs, &b[b_offset], ldb, 
		 info);
	ibscl = 1;
    } else if (bnrm > bignum) {

/*        Scale matrix norm down to BIGNUM. */

	dlascl_("G", &c__0, &c__0, &bnrm, &bignum, m, nrhs, &b[b_offset], ldb, 
		 info);
	ibscl = 2;
    }

/*     If M < N make sure certain entries of B are zero. */

    if (*m < *n) {
	i__1 = *n - *m;
	dlaset_("F", &i__1, nrhs, &c_b82, &c_b82, &b[*m + 1 + b_dim1], ldb);
    }

/*     Overdetermined case. */

    if (*m >= *n) {

/*        Path 1 - overdetermined or exactly determined. */

	mm = *m;
	if (*m >= mnthr) {

/*           Path 1a - overdetermined, with many more rows than columns. */

	    mm = *n;
	    itau = 1;
	    nwork = itau + *n;

/*           Compute A=Q*R. */
/*           (Workspace: need 2*N, prefer N+N*NB) */

	    i__1 = *lwork - nwork + 1;
	    dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &i__1, 
		     info);

/*           Multiply B by transpose(Q). */
/*           (Workspace: need N+NRHS, prefer N+NRHS*NB) */

	    i__1 = *lwork - nwork + 1;
	    dormqr_("L", "T", m, nrhs, n, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[nwork], &i__1, info);

/*           Zero out below R. */

	    if (*n > 1) {
		i__1 = *n - 1;
		i__2 = *n - 1;
		dlaset_("L", &i__1, &i__2, &c_b82, &c_b82, &a[a_dim1 + 2], 
			lda);
	    }
	}

	ie = 1;
	itauq = ie + *n;
	itaup = itauq + *n;
	nwork = itaup + *n;

/*        Bidiagonalize R in A. */
/*        (Workspace: need 3*N+MM, prefer 3*N+(MM+N)*NB) */

	i__1 = *lwork - nwork + 1;
	dgebrd_(&mm, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		work[itaup], &work[nwork], &i__1, info);

/*        Multiply B by transpose of left bidiagonalizing vectors of R. */
/*        (Workspace: need 3*N+NRHS, prefer 3*N+NRHS*NB) */

	i__1 = *lwork - nwork + 1;
	dormbr_("Q", "L", "T", &mm, nrhs, n, &a[a_offset], lda, &work[itauq], 
		&b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Solve the bidiagonal least squares problem. */

	dlalsd_("U", &smlsiz, n, nrhs, &s[1], &work[ie], &b[b_offset], ldb, 
		rcond, rank, &work[nwork], &iwork[1], info);
	if (*info != 0) {
	    goto L10;
	}

/*        Multiply B by right bidiagonalizing vectors of R. */

	i__1 = *lwork - nwork + 1;
	dormbr_("P", "L", "N", n, nrhs, n, &a[a_offset], lda, &work[itaup], &
		b[b_offset], ldb, &work[nwork], &i__1, info);

    } else /* if(complicated condition) */ {
/* Computing MAX */
	i__1 = *m, i__2 = (*m << 1) - 4, i__1 = max(i__1,i__2), i__1 = max(
		i__1,*nrhs), i__2 = *n - *m * 3, i__1 = max(i__1,i__2);
	if (*n >= mnthr && *lwork >= (*m << 2) + *m * *m + max(i__1,wlalsd)) {

/*        Path 2a - underdetermined, with many more columns than rows */
/*        and sufficient workspace for an efficient algorithm. */

	    ldwork = *m;
/* Computing MAX */
/* Computing MAX */
	    i__3 = *m, i__4 = (*m << 1) - 4, i__3 = max(i__3,i__4), i__3 = 
		    max(i__3,*nrhs), i__4 = *n - *m * 3;
	    i__1 = (*m << 2) + *m * *lda + max(i__3,i__4), i__2 = *m * *lda + 
		    *m + *m * *nrhs, i__1 = max(i__1,i__2), i__2 = (*m << 2) 
		    + *m * *lda + wlalsd;
	    if (*lwork >= max(i__1,i__2)) {
		ldwork = *lda;
	    }
	    itau = 1;
	    nwork = *m + 1;

/*        Compute A=L*Q. */
/*        (Workspace: need 2*M, prefer M+M*NB) */

	    i__1 = *lwork - nwork + 1;
	    dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &i__1, 
		     info);
	    il = nwork;

/*        Copy L to WORK(IL), zeroing out above its diagonal. */

	    dlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwork);
	    i__1 = *m - 1;
	    i__2 = *m - 1;
	    dlaset_("U", &i__1, &i__2, &c_b82, &c_b82, &work[il + ldwork], &
		    ldwork);
	    ie = il + ldwork * *m;
	    itauq = ie + *m;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*        Bidiagonalize L in WORK(IL). */
/*        (Workspace: need M*M+5*M, prefer M*M+4*M+2*M*NB) */

	    i__1 = *lwork - nwork + 1;
	    dgebrd_(m, m, &work[il], &ldwork, &s[1], &work[ie], &work[itauq], 
		    &work[itaup], &work[nwork], &i__1, info);

/*        Multiply B by transpose of left bidiagonalizing vectors of L. */
/*        (Workspace: need M*M+4*M+NRHS, prefer M*M+4*M+NRHS*NB) */

	    i__1 = *lwork - nwork + 1;
	    dormbr_("Q", "L", "T", m, nrhs, m, &work[il], &ldwork, &work[
		    itauq], &b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Solve the bidiagonal least squares problem. */

	    dlalsd_("U", &smlsiz, m, nrhs, &s[1], &work[ie], &b[b_offset], 
		    ldb, rcond, rank, &work[nwork], &iwork[1], info);
	    if (*info != 0) {
		goto L10;
	    }

/*        Multiply B by right bidiagonalizing vectors of L. */

	    i__1 = *lwork - nwork + 1;
	    dormbr_("P", "L", "N", m, nrhs, m, &work[il], &ldwork, &work[
		    itaup], &b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Zero out below first M rows of B. */

	    i__1 = *n - *m;
	    dlaset_("F", &i__1, nrhs, &c_b82, &c_b82, &b[*m + 1 + b_dim1], 
		    ldb);
	    nwork = itau + *m;

/*        Multiply transpose(Q) by B. */
/*        (Workspace: need M+NRHS, prefer M+NRHS*NB) */

	    i__1 = *lwork - nwork + 1;
	    dormlq_("L", "T", n, nrhs, m, &a[a_offset], lda, &work[itau], &b[
		    b_offset], ldb, &work[nwork], &i__1, info);

	} else {

/*        Path 2 - remaining underdetermined cases. */

	    ie = 1;
	    itauq = ie + *m;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*        Bidiagonalize A. */
/*        (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB) */

	    i__1 = *lwork - nwork + 1;
	    dgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[nwork], &i__1, info);

/*        Multiply B by transpose of left bidiagonalizing vectors. */
/*        (Workspace: need 3*M+NRHS, prefer 3*M+NRHS*NB) */

	    i__1 = *lwork - nwork + 1;
	    dormbr_("Q", "L", "T", m, nrhs, n, &a[a_offset], lda, &work[itauq]
, &b[b_offset], ldb, &work[nwork], &i__1, info);

/*        Solve the bidiagonal least squares problem. */

	    dlalsd_("L", &smlsiz, m, nrhs, &s[1], &work[ie], &b[b_offset], 
		    ldb, rcond, rank, &work[nwork], &iwork[1], info);
	    if (*info != 0) {
		goto L10;
	    }

/*        Multiply B by right bidiagonalizing vectors of A. */

	    i__1 = *lwork - nwork + 1;
	    dormbr_("P", "L", "N", n, nrhs, m, &a[a_offset], lda, &work[itaup]
, &b[b_offset], ldb, &work[nwork], &i__1, info);

	}
    }

/*     Undo scaling. */

    if (iascl == 1) {
	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, n, nrhs, &b[b_offset], ldb, 
		 info);
	dlascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    } else if (iascl == 2) {
	dlascl_("G", &c__0, &c__0, &anrm, &bignum, n, nrhs, &b[b_offset], ldb, 
		 info);
	dlascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		minmn, info);
    }
    if (ibscl == 1) {
	dlascl_("G", &c__0, &c__0, &smlnum, &bnrm, n, nrhs, &b[b_offset], ldb, 
		 info);
    } else if (ibscl == 2) {
	dlascl_("G", &c__0, &c__0, &bignum, &bnrm, n, nrhs, &b[b_offset], ldb, 
		 info);
    }

L10:
    work[1] = (doublereal) maxwrk;
    return 0;

/*     End of DGELSD */

} /* dgelsd_ */
