/* dgelq2.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int dgelq2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, k;
    doublereal aii;
    extern /* Subroutine */ int dlarf_(char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, integer *, 
	    doublereal *), dlarfp_(integer *, doublereal *, 
	    doublereal *, integer *, doublereal *), xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGELQ2 computes an LQ factorization of a real m by n matrix A: */
/*  A = L * Q. */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  M >= 0. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  N >= 0. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the m by n matrix A. */
/*          On exit, the elements on and below the diagonal of the array */
/*          contain the m by min(m,n) lower trapezoidal matrix L (L is */
/*          lower triangular if m <= n); the elements above the diagonal, */
/*          with the array TAU, represent the orthogonal matrix Q as a */
/*          product of elementary reflectors (see Further Details). */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  TAU     (output) DOUBLE PRECISION array, dimension (min(M,N)) */
/*          The scalar factors of the elementary reflectors (see Further */
/*          Details). */

/*  WORK    (workspace) DOUBLE PRECISION array, dimension (M) */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -i, the i-th argument had an illegal value */

/*  Further Details */
/*  =============== */

/*  The matrix Q is represented as a product of elementary reflectors */

/*     Q = H(k) . . . H(2) H(1), where k = min(m,n). */

/*  Each H(i) has the form */

/*     H(i) = I - tau * v * v' */

/*  where tau is a real scalar, and v is a real vector with */
/*  v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n), */
/*  and tau in TAU(i). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
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
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGELQ2", &i__1);
	return 0;
    }

    k = min(*m,*n);

    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Generate elementary reflector H(i) to annihilate A(i,i+1:n) */

	i__2 = *n - i__ + 1;
/* Computing MIN */
	i__3 = i__ + 1;
	dlarfp_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3, *n)* a_dim1]
, lda, &tau[i__]);
	if (i__ < *m) {

/*           Apply H(i) to A(i+1:m,i:n) from the right */

	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.;
	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
	    dlarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[
		    i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
/* L10: */
    }
    return 0;

/*     End of DGELQ2 */

} /* dgelq2_ */
