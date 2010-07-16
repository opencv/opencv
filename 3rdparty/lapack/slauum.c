/* slauum.f -- translated by f2c (version 20061008).
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
static real c_b15 = 1.f;

/* Subroutine */ int slauum_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2, i__3, i__4;

    /* Local variables */
    integer i__, ib, nb;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, integer *, integer *, 
	    integer *, real *, real *, integer *, real *, integer *, real *, 
	    real *, integer *);
    logical upper;
    extern /* Subroutine */ int strmm_(char *, char *, char *, char *, 
	    integer *, integer *, real *, real *, integer *, real *, integer *
), ssyrk_(char *, char *, integer 
	    *, integer *, real *, real *, integer *, real *, real *, integer *
), slauu2_(char *, integer *, real *, integer *, 
	    integer *), xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAUUM computes the product U * U' or L' * L, where the triangular */
/*  factor U or L is stored in the upper or lower triangular part of */
/*  the array A. */

/*  If UPLO = 'U' or 'u' then the upper triangle of the result is stored, */
/*  overwriting the factor U in A. */
/*  If UPLO = 'L' or 'l' then the lower triangle of the result is stored, */
/*  overwriting the factor L in A. */

/*  This is the blocked form of the algorithm, calling Level 3 BLAS. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the triangular factor stored in the array A */
/*          is upper or lower triangular: */
/*          = 'U':  Upper triangular */
/*          = 'L':  Lower triangular */

/*  N       (input) INTEGER */
/*          The order of the triangular factor U or L.  N >= 0. */

/*  A       (input/output) REAL array, dimension (LDA,N) */
/*          On entry, the triangular factor U or L. */
/*          On exit, if UPLO = 'U', the upper triangle of A is */
/*          overwritten with the upper triangle of the product U * U'; */
/*          if UPLO = 'L', the lower triangle of A is overwritten with */
/*          the lower triangle of the product L' * L. */

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
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAUUM", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Determine the block size for this environment. */

    nb = ilaenv_(&c__1, "SLAUUM", uplo, n, &c_n1, &c_n1, &c_n1);

    if (nb <= 1 || nb >= *n) {

/*        Use unblocked code */

	slauu2_(uplo, n, &a[a_offset], lda, info);
    } else {

/*        Use blocked code */

	if (upper) {

/*           Compute the product U * U'. */

	    i__1 = *n;
	    i__2 = nb;
	    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = min(i__3,i__4);
		i__3 = i__ - 1;
		strmm_("Right", "Upper", "Transpose", "Non-unit", &i__3, &ib, 
			&c_b15, &a[i__ + i__ * a_dim1], lda, &a[i__ * a_dim1 
			+ 1], lda)
			;
		slauu2_("Upper", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    sgemm_("No transpose", "Transpose", &i__3, &ib, &i__4, &
			    c_b15, &a[(i__ + ib) * a_dim1 + 1], lda, &a[i__ + 
			    (i__ + ib) * a_dim1], lda, &c_b15, &a[i__ * 
			    a_dim1 + 1], lda);
		    i__3 = *n - i__ - ib + 1;
		    ssyrk_("Upper", "No transpose", &ib, &i__3, &c_b15, &a[
			    i__ + (i__ + ib) * a_dim1], lda, &c_b15, &a[i__ + 
			    i__ * a_dim1], lda);
		}
/* L10: */
	    }
	} else {

/*           Compute the product L' * L. */

	    i__2 = *n;
	    i__1 = nb;
	    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
/* Computing MIN */
		i__3 = nb, i__4 = *n - i__ + 1;
		ib = min(i__3,i__4);
		i__3 = i__ - 1;
		strmm_("Left", "Lower", "Transpose", "Non-unit", &ib, &i__3, &
			c_b15, &a[i__ + i__ * a_dim1], lda, &a[i__ + a_dim1], 
			lda);
		slauu2_("Lower", &ib, &a[i__ + i__ * a_dim1], lda, info);
		if (i__ + ib <= *n) {
		    i__3 = i__ - 1;
		    i__4 = *n - i__ - ib + 1;
		    sgemm_("Transpose", "No transpose", &ib, &i__3, &i__4, &
			    c_b15, &a[i__ + ib + i__ * a_dim1], lda, &a[i__ + 
			    ib + a_dim1], lda, &c_b15, &a[i__ + a_dim1], lda);
		    i__3 = *n - i__ - ib + 1;
		    ssyrk_("Lower", "Transpose", &ib, &i__3, &c_b15, &a[i__ + 
			    ib + i__ * a_dim1], lda, &c_b15, &a[i__ + i__ * 
			    a_dim1], lda);
		}
/* L20: */
	    }
	}
    }

    return 0;

/*     End of SLAUUM */

} /* slauum_ */
