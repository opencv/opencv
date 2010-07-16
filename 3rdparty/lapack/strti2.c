/* strti2.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int strti2_(char *uplo, char *diag, integer *n, real *a, 
	integer *lda, integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    integer j;
    real ajj;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sscal_(integer *, real *, real *, integer *);
    logical upper;
    extern /* Subroutine */ int strmv_(char *, char *, char *, integer *, 
	    real *, integer *, real *, integer *), 
	    xerbla_(char *, integer *);
    logical nounit;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  STRTI2 computes the inverse of a real upper or lower triangular */
/*  matrix. */

/*  This is the Level 2 BLAS version of the algorithm. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the matrix A is upper or lower triangular. */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  DIAG    (input) CHARACTER*1 */
/*          Specifies whether or not the matrix A is unit triangular. */
/*          = 'N':  Non-unit triangular */
/*          = 'U':  Unit triangular */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  A       (input/output) REAL array, dimension (LDA,N) */
/*          On entry, the triangular matrix A.  If UPLO = 'U', the */
/*          leading n by n upper triangular part of the array A contains */
/*          the upper triangular matrix, and the strictly lower */
/*          triangular part of A is not referenced.  If UPLO = 'L', the */
/*          leading n by n lower triangular part of the array A contains */
/*          the lower triangular matrix, and the strictly upper */
/*          triangular part of A is not referenced.  If DIAG = 'U', the */
/*          diagonal elements of A are also not referenced and are */
/*          assumed to be 1. */

/*          On exit, the (triangular) inverse of the original matrix, in */
/*          the same storage format. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,N). */

/*  INFO    (output) INTEGER */
/*          = 0: successful exit */
/*          < 0: if INFO = -k, the k-th argument had an illegal value */

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

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    nounit = lsame_(diag, "N");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("STRTI2", &i__1);
	return 0;
    }

    if (upper) {

/*        Compute inverse of upper triangular matrix. */

	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (nounit) {
		a[j + j * a_dim1] = 1.f / a[j + j * a_dim1];
		ajj = -a[j + j * a_dim1];
	    } else {
		ajj = -1.f;
	    }

/*           Compute elements 1:j-1 of j-th column. */

	    i__2 = j - 1;
	    strmv_("Upper", "No transpose", diag, &i__2, &a[a_offset], lda, &
		    a[j * a_dim1 + 1], &c__1);
	    i__2 = j - 1;
	    sscal_(&i__2, &ajj, &a[j * a_dim1 + 1], &c__1);
/* L10: */
	}
    } else {

/*        Compute inverse of lower triangular matrix. */

	for (j = *n; j >= 1; --j) {
	    if (nounit) {
		a[j + j * a_dim1] = 1.f / a[j + j * a_dim1];
		ajj = -a[j + j * a_dim1];
	    } else {
		ajj = -1.f;
	    }
	    if (j < *n) {

/*              Compute elements j+1:n of j-th column. */

		i__1 = *n - j;
		strmv_("Lower", "No transpose", diag, &i__1, &a[j + 1 + (j + 
			1) * a_dim1], lda, &a[j + 1 + j * a_dim1], &c__1);
		i__1 = *n - j;
		sscal_(&i__1, &ajj, &a[j + 1 + j * a_dim1], &c__1);
	    }
/* L20: */
	}
    }

    return 0;

/*     End of STRTI2 */

} /* strti2_ */
