#include "clapack.h"

/* Table of constant values */

static integer c__1 = 1;
static integer c_n1 = -1;
static integer c__0 = 0;
static doublereal c_b227 = 0.;
static doublereal c_b248 = 1.;

/* Subroutine */ int dgesdd_(char *jobz, integer *m, integer *n, doublereal *
	a, integer *lda, doublereal *s, doublereal *u, integer *ldu, 
	doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, 
	integer *iwork, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1, 
	    i__2, i__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, ie, il, ir, iu, blk;
    doublereal dum[1], eps;
    integer ivt, iscl;
    doublereal anrm;
    integer idum[1], ierr, itau;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *);
    extern logical lsame_(char *, char *);
    integer chunk, minmn, wrkbl, itaup, itauq, mnthr;
    logical wntqa;
    integer nwork;
    logical wntqn, wntqo, wntqs;
    extern /* Subroutine */ int dbdsdc_(char *, char *, integer *, doublereal 
	    *, doublereal *, doublereal *, integer *, doublereal *, integer *, 
	     doublereal *, integer *, doublereal *, integer *, integer *), dgebrd_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, integer *, integer *);
    extern doublereal dlamch_(char *), dlange_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *);
    integer bdspac;
    extern /* Subroutine */ int dgelqf_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *, integer *), 
	    dlascl_(char *, integer *, integer *, doublereal *, doublereal *, 
	    integer *, integer *, doublereal *, integer *, integer *),
	     dgeqrf_(integer *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *, integer *), dlacpy_(char *, 
	     integer *, integer *, doublereal *, integer *, doublereal *, 
	    integer *), dlaset_(char *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, integer *), 
	    xerbla_(char *, integer *), dorgbr_(char *, integer *, 
	    integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, integer *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    doublereal bignum;
    extern /* Subroutine */ int dormbr_(char *, char *, char *, integer *, 
	    integer *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, integer *), dorglq_(integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, integer *, 
	    integer *), dorgqr_(integer *, integer *, integer *, doublereal *, 
	     integer *, doublereal *, doublereal *, integer *, integer *);
    integer ldwrkl, ldwrkr, minwrk, ldwrku, maxwrk, ldwkvt;
    doublereal smlnum;
    logical wntqas, lquery;


/*  -- LAPACK driver routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGESDD computes the singular value decomposition (SVD) of a real */
/*  M-by-N matrix A, optionally computing the left and right singular */
/*  vectors.  If singular vectors are desired, it uses a */
/*  divide-and-conquer algorithm. */

/*  The SVD is written */

/*       A = U * SIGMA * transpose(V) */

/*  where SIGMA is an M-by-N matrix which is zero except for its */
/*  min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and */
/*  V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA */
/*  are the singular values of A; they are real and non-negative, and */
/*  are returned in descending order.  The first min(m,n) columns of */
/*  U and V are the left and right singular vectors of A. */

/*  Note that the routine returns VT = V**T, not V. */

/*  The divide and conquer algorithm makes very mild assumptions about */
/*  floating point arithmetic. It will work on machines with a guard */
/*  digit in add/subtract, or on those binary machines without guard */
/*  digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or */
/*  Cray-2. It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

/*  Arguments */
/*  ========= */

/*  JOBZ    (input) CHARACTER*1 */
/*          Specifies options for computing all or part of the matrix U: */
/*          = 'A':  all M columns of U and all N rows of V**T are */
/*                  returned in the arrays U and VT; */
/*          = 'S':  the first min(M,N) columns of U and the first */
/*                  min(M,N) rows of V**T are returned in the arrays U */
/*                  and VT; */
/*          = 'O':  If M >= N, the first N columns of U are overwritten */
/*                  on the array A and all rows of V**T are returned in */
/*                  the array VT; */
/*                  otherwise, all columns of U are returned in the */
/*                  array U and the first M rows of V**T are overwritten */
/*                  in the array A; */
/*          = 'N':  no columns of U or rows of V**T are computed. */

/*  M       (input) INTEGER */
/*          The number of rows of the input matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the input matrix A.  N >= 0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the M-by-N matrix A. */
/*          On exit, */
/*          if JOBZ = 'O',  A is overwritten with the first N columns */
/*                          of U (the left singular vectors, stored */
/*                          columnwise) if M >= N; */
/*                          A is overwritten with the first M rows */
/*                          of V**T (the right singular vectors, stored */
/*                          rowwise) otherwise. */
/*          if JOBZ .ne. 'O', the contents of A are destroyed. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  S       (output) DOUBLE PRECISION array, dimension (min(M,N)) */
/*          The singular values of A, sorted so that S(i) >= S(i+1). */

/*  U       (output) DOUBLE PRECISION array, dimension (LDU,UCOL) */
/*          UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N; */
/*          UCOL = min(M,N) if JOBZ = 'S'. */
/*          If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M */
/*          orthogonal matrix U; */
/*          if JOBZ = 'S', U contains the first min(M,N) columns of U */
/*          (the left singular vectors, stored columnwise); */
/*          if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced. */

/*  LDU     (input) INTEGER */
/*          The leading dimension of the array U.  LDU >= 1; if */
/*          JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M. */

/*  VT      (output) DOUBLE PRECISION array, dimension (LDVT,N) */
/*          If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the */
/*          N-by-N orthogonal matrix V**T; */
/*          if JOBZ = 'S', VT contains the first min(M,N) rows of */
/*          V**T (the right singular vectors, stored rowwise); */
/*          if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced. */

/*  LDVT    (input) INTEGER */
/*          The leading dimension of the array VT.  LDVT >= 1; if */
/*          JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N; */
/*          if JOBZ = 'S', LDVT >= min(M,N). */

/*  WORK    (workspace/output) DOUBLE PRECISION array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK; */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >= 1. */
/*          If JOBZ = 'N', */
/*            LWORK >= 3*min(M,N) + max(max(M,N),7*min(M,N)). */
/*          If JOBZ = 'O', */
/*            LWORK >= 3*min(M,N)*min(M,N) + */
/*                     max(max(M,N),5*min(M,N)*min(M,N)+4*min(M,N)). */
/*          If JOBZ = 'S' or 'A' */
/*            LWORK >= 3*min(M,N)*min(M,N) + */
/*                     max(max(M,N),4*min(M,N)*min(M,N)+4*min(M,N)). */
/*          For good performance, LWORK should generally be larger. */
/*          If LWORK = -1 but other input arguments are legal, WORK(1) */
/*          returns the optimal LWORK. */

/*  IWORK   (workspace) INTEGER array, dimension (8*min(M,N)) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  DBDSDC did not converge, updating process failed. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Huan Ren, Computer Science Division, University of */
/*     California at Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Local Arrays .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --s;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;
    minmn = min(*m,*n);
    wntqa = lsame_(jobz, "A");
    wntqs = lsame_(jobz, "S");
    wntqas = wntqa || wntqs;
    wntqo = lsame_(jobz, "O");
    wntqn = lsame_(jobz, "N");
    lquery = *lwork == -1;

    if (! (wntqa || wntqs || wntqo || wntqn)) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*ldu < 1 || wntqas && *ldu < *m || wntqo && *m < *n && *ldu < *
	    m) {
	*info = -8;
    } else if (*ldvt < 1 || wntqa && *ldvt < *n || wntqs && *ldvt < minmn || 
	    wntqo && *m >= *n && *ldvt < *n) {
	*info = -10;
    }

/*     Compute workspace */
/*      (Note: Comments in the code beginning "Workspace:" describe the */
/*       minimal amount of workspace needed at that point in the code, */
/*       as well as the preferred amount for good performance. */
/*       NB refers to the optimal block size for the immediately */
/*       following subroutine, as returned by ILAENV.) */

    if (*info == 0) {
	minwrk = 1;
	maxwrk = 1;
	if (*m >= *n && minmn > 0) {

/*           Compute space needed for DBDSDC */

	    mnthr = (integer) (minmn * 11. / 6.);
	    if (wntqn) {
		bdspac = *n * 7;
	    } else {
		bdspac = *n * 3 * *n + (*n << 2);
	    }
	    if (*m >= mnthr) {
		if (wntqn) {

/*                 Path 1 (M much larger than N, JOBZ='N') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "DGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", n, n, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n;
		    maxwrk = max(i__1,i__2);
		    minwrk = bdspac + *n;
		} else if (wntqo) {

/*                 Path 2 (M much larger than N, JOBZ='O') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "DGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "DORGQR", 
			    " ", m, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", n, n, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "QLN", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "PRT", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + (*n << 1) * *n;
		    minwrk = bdspac + (*n << 1) * *n + *n * 3;
		} else if (wntqs) {

/*                 Path 3 (M much larger than N, JOBZ='S') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "DGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *n * ilaenv_(&c__1, "DORGQR", 
			    " ", m, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", n, n, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "QLN", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "PRT", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *n * *n;
		    minwrk = bdspac + *n * *n + *n * 3;
		} else if (wntqa) {

/*                 Path 4 (M much larger than N, JOBZ='A') */

		    wrkbl = *n + *n * ilaenv_(&c__1, "DGEQRF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n + *m * ilaenv_(&c__1, "DORGQR", 
			    " ", m, m, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + (*n << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", n, n, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "QLN", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "PRT", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *n * *n;
		    minwrk = bdspac + *n * *n + *n * 3;
		}
	    } else {

/*              Path 5 (M at least N, but not much larger) */

		wrkbl = *n * 3 + (*m + *n) * ilaenv_(&c__1, "DGEBRD", " ", m, 
			n, &c_n1, &c_n1);
		if (wntqn) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		} else if (wntqo) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "QLN", m, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "PRT", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *n;
/* Computing MAX */
		    i__1 = *m, i__2 = *n * *n + bdspac;
		    minwrk = *n * 3 + max(i__1,i__2);
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "QLN", m, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "PRT", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *n * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "QLN", m, m, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *n * 3 + *n * ilaenv_(&c__1, "DORMBR"
, "PRT", n, n, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = maxwrk, i__2 = bdspac + *n * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		}
	    }
	} else if (minmn > 0) {

/*           Compute space needed for DBDSDC */

	    mnthr = (integer) (minmn * 11. / 6.);
	    if (wntqn) {
		bdspac = *m * 7;
	    } else {
		bdspac = *m * 3 * *m + (*m << 2);
	    }
	    if (*n >= mnthr) {
		if (wntqn) {

/*                 Path 1t (N much larger than M, JOBZ='N') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "DGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", m, m, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m;
		    maxwrk = max(i__1,i__2);
		    minwrk = bdspac + *m;
		} else if (wntqo) {

/*                 Path 2t (N much larger than M, JOBZ='O') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "DGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "DORGLQ", 
			    " ", m, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", m, m, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "QLN", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "PRT", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + (*m << 1) * *m;
		    minwrk = bdspac + (*m << 1) * *m + *m * 3;
		} else if (wntqs) {

/*                 Path 3t (N much larger than M, JOBZ='S') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "DGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *m * ilaenv_(&c__1, "DORGLQ", 
			    " ", m, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", m, m, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "QLN", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "PRT", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *m;
		    minwrk = bdspac + *m * *m + *m * 3;
		} else if (wntqa) {

/*                 Path 4t (N much larger than M, JOBZ='A') */

		    wrkbl = *m + *m * ilaenv_(&c__1, "DGELQF", " ", m, n, &
			    c_n1, &c_n1);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m + *n * ilaenv_(&c__1, "DORGLQ", 
			    " ", n, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + (*m << 1) * ilaenv_(&c__1, 
			    "DGEBRD", " ", m, m, &c_n1, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "QLN", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "PRT", m, m, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *m;
		    minwrk = bdspac + *m * *m + *m * 3;
		}
	    } else {

/*              Path 5t (N greater than M, but not much larger) */

		wrkbl = *m * 3 + (*m + *n) * ilaenv_(&c__1, "DGEBRD", " ", m, 
			n, &c_n1, &c_n1);
		if (wntqn) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		} else if (wntqo) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "QLN", m, m, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "PRT", m, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *n;
/* Computing MAX */
		    i__1 = *n, i__2 = *m * *m + bdspac;
		    minwrk = *m * 3 + max(i__1,i__2);
		} else if (wntqs) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "QLN", m, m, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "PRT", m, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		} else if (wntqa) {
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "QLN", m, m, n, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = *m * 3 + *m * ilaenv_(&c__1, "DORMBR"
, "PRT", n, n, m, &c_n1);
		    wrkbl = max(i__1,i__2);
/* Computing MAX */
		    i__1 = wrkbl, i__2 = bdspac + *m * 3;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		}
	    }
	}
	maxwrk = max(maxwrk,minwrk);
	work[1] = (doublereal) maxwrk;

	if (*lwork < minwrk && ! lquery) {
	    *info = -12;
	}
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGESDD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }

/*     Get machine constants */

    eps = dlamch_("P");
    smlnum = sqrt(dlamch_("S")) / eps;
    bignum = 1. / smlnum;

/*     Scale A if max element outside range [SMLNUM,BIGNUM] */

    anrm = dlange_("M", m, n, &a[a_offset], lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
	iscl = 1;
	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, &
		ierr);
    } else if (anrm > bignum) {
	iscl = 1;
	dlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, &
		ierr);
    }

    if (*m >= *n) {

/*        A has at least as many rows as columns. If A has sufficiently */
/*        more rows than columns, first reduce using the QR */
/*        decomposition (if sufficient workspace available) */

	if (*m >= mnthr) {

	    if (wntqn) {

/*              Path 1 (M much larger than N, JOBZ='N') */
/*              No singular vectors to be computed */

		itau = 1;
		nwork = itau + *n;

/*              Compute A=Q*R */
/*              (Workspace: need 2*N, prefer N+N*NB) */

		i__1 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Zero out below R */

		i__1 = *n - 1;
		i__2 = *n - 1;
		dlaset_("L", &i__1, &i__2, &c_b227, &c_b227, &a[a_dim1 + 2], 
			lda);
		ie = 1;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in A */
/*              (Workspace: need 4*N, prefer 3*N+2*N*NB) */

		i__1 = *lwork - nwork + 1;
		dgebrd_(n, n, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nwork = ie + *n;

/*              Perform bidiagonal SVD, computing singular values only */
/*              (Workspace: need N+BDSPAC) */

		dbdsdc_("U", "N", n, &s[1], &work[ie], dum, &c__1, dum, &c__1, 
			 dum, idum, &work[nwork], &iwork[1], info);

	    } else if (wntqo) {

/*              Path 2 (M much larger than N, JOBZ = 'O') */
/*              N left singular vectors to be overwritten on A and */
/*              N right singular vectors to be computed in VT */

		ir = 1;

/*              WORK(IR) is LDWRKR by N */

		if (*lwork >= *lda * *n + *n * *n + *n * 3 + bdspac) {
		    ldwrkr = *lda;
		} else {
		    ldwrkr = (*lwork - *n * *n - *n * 3 - bdspac) / *n;
		}
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;

/*              Compute A=Q*R */
/*              (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		i__1 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy R to WORK(IR), zeroing out below it */

		dlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__1 = *n - 1;
		i__2 = *n - 1;
		dlaset_("L", &i__1, &i__2, &c_b227, &c_b227, &work[ir + 1], &
			ldwrkr);

/*              Generate Q in A */
/*              (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		i__1 = *lwork - nwork + 1;
		dorgqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__1, &ierr);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in VT, copying result to WORK(IR) */
/*              (Workspace: need N*N+4*N, prefer N*N+3*N+2*N*NB) */

		i__1 = *lwork - nwork + 1;
		dgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*              WORK(IU) is N by N */

		iu = nwork;
		nwork = iu + *n * *n;

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in WORK(IU) and computing right */
/*              singular vectors of bidiagonal matrix in VT */
/*              (Workspace: need N+N*N+BDSPAC) */

		dbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], n, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Overwrite WORK(IU) by left singular vectors of R */
/*              and VT by right singular vectors of R */
/*              (Workspace: need 2*N*N+3*N, prefer 2*N*N+2*N+N*NB) */

		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &work[iu], n, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);

/*              Multiply Q in A by left singular vectors of R in */
/*              WORK(IU), storing result in WORK(IR) and copying to A */
/*              (Workspace: need 2*N*N, prefer N*N+M*N) */

		i__1 = *m;
		i__2 = ldwrkr;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += 
			i__2) {
/* Computing MIN */
		    i__3 = *m - i__ + 1;
		    chunk = min(i__3,ldwrkr);
		    dgemm_("N", "N", &chunk, n, n, &c_b248, &a[i__ + a_dim1], 
			    lda, &work[iu], n, &c_b227, &work[ir], &ldwrkr);
		    dlacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ + 
			    a_dim1], lda);
/* L10: */
		}

	    } else if (wntqs) {

/*              Path 3 (M much larger than N, JOBZ='S') */
/*              N left singular vectors to be computed in U and */
/*              N right singular vectors to be computed in VT */

		ir = 1;

/*              WORK(IR) is N by N */

		ldwrkr = *n;
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;

/*              Compute A=Q*R */
/*              (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		i__2 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy R to WORK(IR), zeroing out below it */

		dlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__2 = *n - 1;
		i__1 = *n - 1;
		dlaset_("L", &i__2, &i__1, &c_b227, &c_b227, &work[ir + 1], &
			ldwrkr);

/*              Generate Q in A */
/*              (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		i__2 = *lwork - nwork + 1;
		dorgqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__2, &ierr);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in WORK(IR) */
/*              (Workspace: need N*N+4*N, prefer N*N+3*N+2*N*NB) */

		i__2 = *lwork - nwork + 1;
		dgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagoal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in VT */
/*              (Workspace: need N+BDSPAC) */

		dbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Overwrite U by left singular vectors of R and VT */
/*              by right singular vectors of R */
/*              (Workspace: need N*N+3*N, prefer N*N+2*N+N*NB) */

		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*              Multiply Q in A by left singular vectors of R in */
/*              WORK(IR), storing result in U */
/*              (Workspace: need N*N) */

		dlacpy_("F", n, n, &u[u_offset], ldu, &work[ir], &ldwrkr);
		dgemm_("N", "N", m, n, n, &c_b248, &a[a_offset], lda, &work[
			ir], &ldwrkr, &c_b227, &u[u_offset], ldu);

	    } else if (wntqa) {

/*              Path 4 (M much larger than N, JOBZ='A') */
/*              M left singular vectors to be computed in U and */
/*              N right singular vectors to be computed in VT */

		iu = 1;

/*              WORK(IU) is N by N */

		ldwrku = *n;
		itau = iu + ldwrku * *n;
		nwork = itau + *n;

/*              Compute A=Q*R, copying result to U */
/*              (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		i__2 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		dlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);

/*              Generate Q in U */
/*              (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */
		i__2 = *lwork - nwork + 1;
		dorgqr_(m, m, n, &u[u_offset], ldu, &work[itau], &work[nwork], 
			 &i__2, &ierr);

/*              Produce R in A, zeroing out other entries */

		i__2 = *n - 1;
		i__1 = *n - 1;
		dlaset_("L", &i__2, &i__1, &c_b227, &c_b227, &a[a_dim1 + 2], 
			lda);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;

/*              Bidiagonalize R in A */
/*              (Workspace: need N*N+4*N, prefer N*N+3*N+2*N*NB) */

		i__2 = *lwork - nwork + 1;
		dgebrd_(n, n, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in WORK(IU) and computing right */
/*              singular vectors of bidiagonal matrix in VT */
/*              (Workspace: need N+N*N+BDSPAC) */

		dbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], n, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Overwrite WORK(IU) by left singular vectors of R and VT */
/*              by right singular vectors of R */
/*              (Workspace: need N*N+3*N, prefer N*N+2*N+N*NB) */

		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", n, n, n, &a[a_offset], lda, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			ierr);
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*              Multiply Q in U by left singular vectors of R in */
/*              WORK(IU), storing result in A */
/*              (Workspace: need N*N) */

		dgemm_("N", "N", m, n, n, &c_b248, &u[u_offset], ldu, &work[
			iu], &ldwrku, &c_b227, &a[a_offset], lda);

/*              Copy left singular vectors of A from A to U */

		dlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);

	    }

	} else {

/*           M .LT. MNTHR */

/*           Path 5 (M at least N, but not much larger) */
/*           Reduce to bidiagonal form without QR decomposition */

	    ie = 1;
	    itauq = ie + *n;
	    itaup = itauq + *n;
	    nwork = itaup + *n;

/*           Bidiagonalize A */
/*           (Workspace: need 3*N+M, prefer 3*N+(M+N)*NB) */

	    i__2 = *lwork - nwork + 1;
	    dgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*              Perform bidiagonal SVD, only computing singular values */
/*              (Workspace: need N+BDSPAC) */

		dbdsdc_("U", "N", n, &s[1], &work[ie], dum, &c__1, dum, &c__1, 
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		iu = nwork;
		if (*lwork >= *m * *n + *n * 3 + bdspac) {

/*                 WORK( IU ) is M by N */

		    ldwrku = *m;
		    nwork = iu + ldwrku * *n;
		    dlaset_("F", m, n, &c_b227, &c_b227, &work[iu], &ldwrku);
		} else {

/*                 WORK( IU ) is N by N */

		    ldwrku = *n;
		    nwork = iu + ldwrku * *n;

/*                 WORK(IR) is LDWRKR by N */

		    ir = nwork;
		    ldwrkr = (*lwork - *n * *n - *n * 3) / *n;
		}
		nwork = iu + ldwrku * *n;

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in WORK(IU) and computing right */
/*              singular vectors of bidiagonal matrix in VT */
/*              (Workspace: need N+N*N+BDSPAC) */

		dbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], &ldwrku, &
			vt[vt_offset], ldvt, dum, idum, &work[nwork], &iwork[
			1], info);

/*              Overwrite VT by right singular vectors of A */
/*              (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

		if (*lwork >= *m * *n + *n * 3 + bdspac) {

/*                 Overwrite WORK(IU) by left singular vectors of A */
/*                 (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		    i__2 = *lwork - nwork + 1;
		    dormbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			    itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			    ierr);

/*                 Copy left singular vectors of A from WORK(IU) to A */

		    dlacpy_("F", m, n, &work[iu], &ldwrku, &a[a_offset], lda);
		} else {

/*                 Generate Q in A */
/*                 (Workspace: need N*N+2*N, prefer N*N+N+N*NB) */

		    i__2 = *lwork - nwork + 1;
		    dorgbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &
			    work[nwork], &i__2, &ierr);

/*                 Multiply Q in A by left singular vectors of */
/*                 bidiagonal matrix in WORK(IU), storing result in */
/*                 WORK(IR) and copying to A */
/*                 (Workspace: need 2*N*N, prefer N*N+M*N) */

		    i__2 = *m;
		    i__1 = ldwrkr;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
/* Computing MIN */
			i__3 = *m - i__ + 1;
			chunk = min(i__3,ldwrkr);
			dgemm_("N", "N", &chunk, n, n, &c_b248, &a[i__ + 
				a_dim1], lda, &work[iu], &ldwrku, &c_b227, &
				work[ir], &ldwrkr);
			dlacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ + 
				a_dim1], lda);
/* L20: */
		    }
		}

	    } else if (wntqs) {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in VT */
/*              (Workspace: need N+BDSPAC) */

		dlaset_("F", m, n, &c_b227, &c_b227, &u[u_offset], ldu);
		dbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Overwrite U by left singular vectors of A and VT */
/*              by right singular vectors of A */
/*              (Workspace: need 3*N, prefer 2*N+N*NB) */

		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else if (wntqa) {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in VT */
/*              (Workspace: need N+BDSPAC) */

		dlaset_("F", m, m, &c_b227, &c_b227, &u[u_offset], ldu);
		dbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Set the right corner of U to identity matrix */

		if (*m > *n) {
		    i__1 = *m - *n;
		    i__2 = *m - *n;
		    dlaset_("F", &i__1, &i__2, &c_b227, &c_b248, &u[*n + 1 + (
			    *n + 1) * u_dim1], ldu);
		}

/*              Overwrite U by left singular vectors of A and VT */
/*              by right singular vectors of A */
/*              (Workspace: need N*N+2*N+M, prefer N*N+2*N+M*NB) */

		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }

	}

    } else {

/*        A has more columns than rows. If A has sufficiently more */
/*        columns than rows, first reduce using the LQ decomposition (if */
/*        sufficient workspace available) */

	if (*n >= mnthr) {

	    if (wntqn) {

/*              Path 1t (N much larger than M, JOBZ='N') */
/*              No singular vectors to be computed */

		itau = 1;
		nwork = itau + *m;

/*              Compute A=L*Q */
/*              (Workspace: need 2*M, prefer M+M*NB) */

		i__1 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Zero out above L */

		i__1 = *m - 1;
		i__2 = *m - 1;
		dlaset_("U", &i__1, &i__2, &c_b227, &c_b227, &a[(a_dim1 << 1) 
			+ 1], lda);
		ie = 1;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in A */
/*              (Workspace: need 4*M, prefer 3*M+2*M*NB) */

		i__1 = *lwork - nwork + 1;
		dgebrd_(m, m, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nwork = ie + *m;

/*              Perform bidiagonal SVD, computing singular values only */
/*              (Workspace: need M+BDSPAC) */

		dbdsdc_("U", "N", m, &s[1], &work[ie], dum, &c__1, dum, &c__1, 
			 dum, idum, &work[nwork], &iwork[1], info);

	    } else if (wntqo) {

/*              Path 2t (N much larger than M, JOBZ='O') */
/*              M right singular vectors to be overwritten on A and */
/*              M left singular vectors to be computed in U */

		ivt = 1;

/*              IVT is M by M */

		il = ivt + *m * *m;
		if (*lwork >= *m * *n + *m * *m + *m * 3 + bdspac) {

/*                 WORK(IL) is M by N */

		    ldwrkl = *m;
		    chunk = *n;
		} else {
		    ldwrkl = *m;
		    chunk = (*lwork - *m * *m) / *m;
		}
		itau = il + ldwrkl * *m;
		nwork = itau + *m;

/*              Compute A=L*Q */
/*              (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		i__1 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);

/*              Copy L to WORK(IL), zeroing about above it */

		dlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__1 = *m - 1;
		i__2 = *m - 1;
		dlaset_("U", &i__1, &i__2, &c_b227, &c_b227, &work[il + 
			ldwrkl], &ldwrkl);

/*              Generate Q in A */
/*              (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		i__1 = *lwork - nwork + 1;
		dorglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__1, &ierr);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in WORK(IL) */
/*              (Workspace: need M*M+4*M, prefer M*M+3*M+2*M*NB) */

		i__1 = *lwork - nwork + 1;
		dgebrd_(m, m, &work[il], &ldwrkl, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U, and computing right singular */
/*              vectors of bidiagonal matrix in WORK(IVT) */
/*              (Workspace: need M+M*M+BDSPAC) */

		dbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], m, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Overwrite U by left singular vectors of L and WORK(IVT) */
/*              by right singular vectors of L */
/*              (Workspace: need 2*M*M+3*M, prefer 2*M*M+2*M+M*NB) */

		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &work[ivt], m, &work[nwork], &i__1, &ierr);

/*              Multiply right singular vectors of L in WORK(IVT) by Q */
/*              in A, storing result in WORK(IL) and copying to A */
/*              (Workspace: need 2*M*M, prefer M*M+M*N) */

		i__1 = *n;
		i__2 = chunk;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += 
			i__2) {
/* Computing MIN */
		    i__3 = *n - i__ + 1;
		    blk = min(i__3,chunk);
		    dgemm_("N", "N", m, &blk, m, &c_b248, &work[ivt], m, &a[
			    i__ * a_dim1 + 1], lda, &c_b227, &work[il], &
			    ldwrkl);
		    dlacpy_("F", m, &blk, &work[il], &ldwrkl, &a[i__ * a_dim1 
			    + 1], lda);
/* L30: */
		}

	    } else if (wntqs) {

/*              Path 3t (N much larger than M, JOBZ='S') */
/*              M right singular vectors to be computed in VT and */
/*              M left singular vectors to be computed in U */

		il = 1;

/*              WORK(IL) is M by M */

		ldwrkl = *m;
		itau = il + ldwrkl * *m;
		nwork = itau + *m;

/*              Compute A=L*Q */
/*              (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		i__2 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);

/*              Copy L to WORK(IL), zeroing out above it */

		dlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__2 = *m - 1;
		i__1 = *m - 1;
		dlaset_("U", &i__2, &i__1, &c_b227, &c_b227, &work[il + 
			ldwrkl], &ldwrkl);

/*              Generate Q in A */
/*              (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		i__2 = *lwork - nwork + 1;
		dorglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork], 
			 &i__2, &ierr);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in WORK(IU), copying result to U */
/*              (Workspace: need M*M+4*M, prefer M*M+3*M+2*M*NB) */

		i__2 = *lwork - nwork + 1;
		dgebrd_(m, m, &work[il], &ldwrkl, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in VT */
/*              (Workspace: need M+BDSPAC) */

		dbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Overwrite U by left singular vectors of L and VT */
/*              by right singular vectors of L */
/*              (Workspace: need M*M+3*M, prefer M*M+2*M+M*NB) */

		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);

/*              Multiply right singular vectors of L in WORK(IL) by */
/*              Q in A, storing result in VT */
/*              (Workspace: need M*M) */

		dlacpy_("F", m, m, &vt[vt_offset], ldvt, &work[il], &ldwrkl);
		dgemm_("N", "N", m, n, m, &c_b248, &work[il], &ldwrkl, &a[
			a_offset], lda, &c_b227, &vt[vt_offset], ldvt);

	    } else if (wntqa) {

/*              Path 4t (N much larger than M, JOBZ='A') */
/*              N right singular vectors to be computed in VT and */
/*              M left singular vectors to be computed in U */

		ivt = 1;

/*              WORK(IVT) is M by M */

		ldwkvt = *m;
		itau = ivt + ldwkvt * *m;
		nwork = itau + *m;

/*              Compute A=L*Q, copying result to VT */
/*              (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		i__2 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		dlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

/*              Generate Q in VT */
/*              (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		i__2 = *lwork - nwork + 1;
		dorglq_(n, n, m, &vt[vt_offset], ldvt, &work[itau], &work[
			nwork], &i__2, &ierr);

/*              Produce L in A, zeroing out other entries */

		i__2 = *m - 1;
		i__1 = *m - 1;
		dlaset_("U", &i__2, &i__1, &c_b227, &c_b227, &a[(a_dim1 << 1) 
			+ 1], lda);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;

/*              Bidiagonalize L in A */
/*              (Workspace: need M*M+4*M, prefer M*M+3*M+2*M*NB) */

		i__2 = *lwork - nwork + 1;
		dgebrd_(m, m, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in WORK(IVT) */
/*              (Workspace: need M+M*M+BDSPAC) */

		dbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1]
, info);

/*              Overwrite U by left singular vectors of L and WORK(IVT) */
/*              by right singular vectors of L */
/*              (Workspace: need M*M+3*M, prefer M*M+2*M+M*NB) */

		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, m, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, m, m, &a[a_offset], lda, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &
			ierr);

/*              Multiply right singular vectors of L in WORK(IVT) by */
/*              Q in VT, storing result in A */
/*              (Workspace: need M*M) */

		dgemm_("N", "N", m, n, m, &c_b248, &work[ivt], &ldwkvt, &vt[
			vt_offset], ldvt, &c_b227, &a[a_offset], lda);

/*              Copy right singular vectors of A from A to VT */

		dlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);

	    }

	} else {

/*           N .LT. MNTHR */

/*           Path 5t (N greater than M, but not much larger) */
/*           Reduce to bidiagonal form without LQ decomposition */

	    ie = 1;
	    itauq = ie + *m;
	    itaup = itauq + *m;
	    nwork = itaup + *m;

/*           Bidiagonalize A */
/*           (Workspace: need 3*M+N, prefer 3*M+(M+N)*NB) */

	    i__2 = *lwork - nwork + 1;
	    dgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {

/*              Perform bidiagonal SVD, only computing singular values */
/*              (Workspace: need M+BDSPAC) */

		dbdsdc_("L", "N", m, &s[1], &work[ie], dum, &c__1, dum, &c__1, 
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		ldwkvt = *m;
		ivt = nwork;
		if (*lwork >= *m * *n + *m * 3 + bdspac) {

/*                 WORK( IVT ) is M by N */

		    dlaset_("F", m, n, &c_b227, &c_b227, &work[ivt], &ldwkvt);
		    nwork = ivt + ldwkvt * *n;
		} else {

/*                 WORK( IVT ) is M by M */

		    nwork = ivt + ldwkvt * *m;
		    il = nwork;

/*                 WORK(IL) is M by CHUNK */

		    chunk = (*lwork - *m * *m - *m * 3) / *m;
		}

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in WORK(IVT) */
/*              (Workspace: need M*M+BDSPAC) */

		dbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1]
, info);

/*              Overwrite U by left singular vectors of A */
/*              (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr);

		if (*lwork >= *m * *n + *m * 3 + bdspac) {

/*                 Overwrite WORK(IVT) by left singular vectors of A */
/*                 (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		    i__2 = *lwork - nwork + 1;
		    dormbr_("P", "R", "T", m, n, m, &a[a_offset], lda, &work[
			    itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, 
			    &ierr);

/*                 Copy right singular vectors of A from WORK(IVT) to A */

		    dlacpy_("F", m, n, &work[ivt], &ldwkvt, &a[a_offset], lda);
		} else {

/*                 Generate P**T in A */
/*                 (Workspace: need M*M+2*M, prefer M*M+M+M*NB) */

		    i__2 = *lwork - nwork + 1;
		    dorgbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &
			    work[nwork], &i__2, &ierr);

/*                 Multiply Q in A by right singular vectors of */
/*                 bidiagonal matrix in WORK(IVT), storing result in */
/*                 WORK(IL) and copying to A */
/*                 (Workspace: need 2*M*M, prefer M*M+M*N) */

		    i__2 = *n;
		    i__1 = chunk;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
/* Computing MIN */
			i__3 = *n - i__ + 1;
			blk = min(i__3,chunk);
			dgemm_("N", "N", m, &blk, m, &c_b248, &work[ivt], &
				ldwkvt, &a[i__ * a_dim1 + 1], lda, &c_b227, &
				work[il], m);
			dlacpy_("F", m, &blk, &work[il], m, &a[i__ * a_dim1 + 
				1], lda);
/* L40: */
		    }
		}
	    } else if (wntqs) {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in VT */
/*              (Workspace: need M+BDSPAC) */

		dlaset_("F", m, n, &c_b227, &c_b227, &vt[vt_offset], ldvt);
		dbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Overwrite U by left singular vectors of A and VT */
/*              by right singular vectors of A */
/*              (Workspace: need 3*M, prefer 2*M+M*NB) */

		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else if (wntqa) {

/*              Perform bidiagonal SVD, computing left singular vectors */
/*              of bidiagonal matrix in U and computing right singular */
/*              vectors of bidiagonal matrix in VT */
/*              (Workspace: need M+BDSPAC) */

		dlaset_("F", n, n, &c_b227, &c_b227, &vt[vt_offset], ldvt);
		dbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1], 
			info);

/*              Set the right corner of VT to identity matrix */

		if (*n > *m) {
		    i__1 = *n - *m;
		    i__2 = *n - *m;
		    dlaset_("F", &i__1, &i__2, &c_b227, &c_b248, &vt[*m + 1 + 
			    (*m + 1) * vt_dim1], ldvt);
		}

/*              Overwrite U by left singular vectors of A and VT */
/*              by right singular vectors of A */
/*              (Workspace: need 2*M+N, prefer 2*M+N*NB) */

		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }

	}

    }

/*     Undo scaling if necessary */

    if (iscl == 1) {
	if (anrm > bignum) {
	    dlascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
	if (anrm < smlnum) {
	    dlascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
    }

/*     Return optimal workspace in WORK(1) */

    work[1] = (doublereal) maxwrk;

    return 0;

/*     End of DGESDD */

} /* dgesdd_ */
