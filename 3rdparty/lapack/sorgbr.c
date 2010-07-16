/* sorgbr.f -- translated by f2c (version 20061008).
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

static integer c__1 = 1;
static integer c_n1 = -1;

/* Subroutine */ int sorgbr_(char *vect, integer *m, integer *n, integer *k, 
	real *a, integer *lda, real *tau, real *work, integer *lwork, integer 
	*info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, j, nb, mn;
    extern logical lsame_(char *, char *);
    integer iinfo;
    logical wantq;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int sorglq_(integer *, integer *, integer *, real 
	    *, integer *, real *, real *, integer *, integer *), sorgqr_(
	    integer *, integer *, integer *, real *, integer *, real *, real *
, integer *, integer *);
    integer lwkopt;
    logical lquery;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SORGBR generates one of the real orthogonal matrices Q or P**T */
/*  determined by SGEBRD when reducing a real matrix A to bidiagonal */
/*  form: A = Q * B * P**T.  Q and P**T are defined as products of */
/*  elementary reflectors H(i) or G(i) respectively. */

/*  If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q */
/*  is of order M: */
/*  if m >= k, Q = H(1) H(2) . . . H(k) and SORGBR returns the first n */
/*  columns of Q, where m >= n >= k; */
/*  if m < k, Q = H(1) H(2) . . . H(m-1) and SORGBR returns Q as an */
/*  M-by-M matrix. */

/*  If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**T */
/*  is of order N: */
/*  if k < n, P**T = G(k) . . . G(2) G(1) and SORGBR returns the first m */
/*  rows of P**T, where n >= m >= k; */
/*  if k >= n, P**T = G(n-1) . . . G(2) G(1) and SORGBR returns P**T as */
/*  an N-by-N matrix. */

/*  Arguments */
/*  ========= */

/*  VECT    (input) CHARACTER*1 */
/*          Specifies whether the matrix Q or the matrix P**T is */
/*          required, as defined in the transformation applied by SGEBRD: */
/*          = 'Q':  generate Q; */
/*          = 'P':  generate P**T. */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix Q or P**T to be returned. */
/*          M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix Q or P**T to be returned. */
/*          N >= 0. */
/*          If VECT = 'Q', M >= N >= min(M,K); */
/*          if VECT = 'P', N >= M >= min(N,K). */

/*  K       (input) INTEGER */
/*          If VECT = 'Q', the number of columns in the original M-by-K */
/*          matrix reduced by SGEBRD. */
/*          If VECT = 'P', the number of rows in the original K-by-N */
/*          matrix reduced by SGEBRD. */
/*          K >= 0. */

/*  A       (input/output) REAL array, dimension (LDA,N) */
/*          On entry, the vectors which define the elementary reflectors, */
/*          as returned by SGEBRD. */
/*          On exit, the M-by-N matrix Q or P**T. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A. LDA >= max(1,M). */

/*  TAU     (input) REAL array, dimension */
/*                                (min(M,K)) if VECT = 'Q' */
/*                                (min(N,K)) if VECT = 'P' */
/*          TAU(i) must contain the scalar factor of the elementary */
/*          reflector H(i) or G(i), which determines Q or P**T, as */
/*          returned by SGEBRD in its array argument TAUQ or TAUP. */

/*  WORK    (workspace/output) REAL array, dimension (MAX(1,LWORK)) */
/*          On exit, if INFO = 0, WORK(1) returns the optimal LWORK. */

/*  LWORK   (input) INTEGER */
/*          The dimension of the array WORK. LWORK >= max(1,min(M,N)). */
/*          For optimum performance LWORK >= min(M,N)*NB, where NB */
/*          is the optimal blocksize. */

/*          If LWORK = -1, then a workspace query is assumed; the routine */
/*          only calculates the optimal size of the WORK array, returns */
/*          this value as the first entry of the WORK array, and no error */
/*          message related to LWORK is issued by XERBLA. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  ===================================================================== */

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

/*     Test the input arguments */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    /* Function Body */
    *info = 0;
    wantq = lsame_(vect, "Q");
    mn = min(*m,*n);
    lquery = *lwork == -1;
    if (! wantq && ! lsame_(vect, "P")) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0 || wantq && (*n > *m || *n < min(*m,*k)) || ! wantq && (
	    *m > *n || *m < min(*n,*k))) {
	*info = -3;
    } else if (*k < 0) {
	*info = -4;
    } else if (*lda < max(1,*m)) {
	*info = -6;
    } else if (*lwork < max(1,mn) && ! lquery) {
	*info = -9;
    }

    if (*info == 0) {
	if (wantq) {
	    nb = ilaenv_(&c__1, "SORGQR", " ", m, n, k, &c_n1);
	} else {
	    nb = ilaenv_(&c__1, "SORGLQ", " ", m, n, k, &c_n1);
	}
	lwkopt = max(1,mn) * nb;
	work[1] = (real) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORGBR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	work[1] = 1.f;
	return 0;
    }

    if (wantq) {

/*        Form Q, determined by a call to SGEBRD to reduce an m-by-k */
/*        matrix */

	if (*m >= *k) {

/*           If m >= k, assume m >= n >= k */

	    sorgqr_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);

	} else {

/*           If m < k, assume m = n */

/*           Shift the vectors which define the elementary reflectors one */
/*           column to the right, and set the first row and column of Q */
/*           to those of the unit matrix */

	    for (j = *m; j >= 2; --j) {
		a[j * a_dim1 + 1] = 0.f;
		i__1 = *m;
		for (i__ = j + 1; i__ <= i__1; ++i__) {
		    a[i__ + j * a_dim1] = a[i__ + (j - 1) * a_dim1];
/* L10: */
		}
/* L20: */
	    }
	    a[a_dim1 + 1] = 1.f;
	    i__1 = *m;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		a[i__ + a_dim1] = 0.f;
/* L30: */
	    }
	    if (*m > 1) {

/*              Form Q(2:m,2:m) */

		i__1 = *m - 1;
		i__2 = *m - 1;
		i__3 = *m - 1;
		sorgqr_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    } else {

/*        Form P', determined by a call to SGEBRD to reduce a k-by-n */
/*        matrix */

	if (*k < *n) {

/*           If k < n, assume k <= m <= n */

	    sorglq_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);

	} else {

/*           If k >= n, assume m = n */

/*           Shift the vectors which define the elementary reflectors one */
/*           row downward, and set the first row and column of P' to */
/*           those of the unit matrix */

	    a[a_dim1 + 1] = 1.f;
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		a[i__ + a_dim1] = 0.f;
/* L40: */
	    }
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		for (i__ = j - 1; i__ >= 2; --i__) {
		    a[i__ + j * a_dim1] = a[i__ - 1 + j * a_dim1];
/* L50: */
		}
		a[j * a_dim1 + 1] = 0.f;
/* L60: */
	    }
	    if (*n > 1) {

/*              Form P'(2:n,2:n) */

		i__1 = *n - 1;
		i__2 = *n - 1;
		i__3 = *n - 1;
		sorglq_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    }
    work[1] = (real) lwkopt;
    return 0;

/*     End of SORGBR */

} /* sorgbr_ */
