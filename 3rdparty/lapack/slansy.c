#include "clapack.h"

/* Table of constant values */

static integer c__1 = 1;

doublereal slansy_(char *norm, char *uplo, integer *n, real *a, integer *lda, 
	real *work)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    real ret_val, r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__, j;
    real sum, absa, scale;
    extern logical lsame_(char *, char *);
    real value;
    extern /* Subroutine */ int slassq_(integer *, real *, integer *, real *, 
	    real *);


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLANSY  returns the value of the one norm,  or the Frobenius norm, or */
/*  the  infinity norm,  or the  element of  largest absolute value  of a */
/*  real symmetric matrix A. */

/*  Description */
/*  =========== */

/*  SLANSY returns the value */

/*     SLANSY = ( max(abs(A(i,j))), NORM = 'M' or 'm' */
/*              ( */
/*              ( norm1(A),         NORM = '1', 'O' or 'o' */
/*              ( */
/*              ( normI(A),         NORM = 'I' or 'i' */
/*              ( */
/*              ( normF(A),         NORM = 'F', 'f', 'E' or 'e' */

/*  where  norm1  denotes the  one norm of a matrix (maximum column sum), */
/*  normI  denotes the  infinity norm  of a matrix  (maximum row sum) and */
/*  normF  denotes the  Frobenius norm of a matrix (square root of sum of */
/*  squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm. */

/*  Arguments */
/*  ========= */

/*  NORM    (input) CHARACTER*1 */
/*          Specifies the value to be returned in SLANSY as described */
/*          above. */

/*  UPLO    (input) CHARACTER*1 */
/*          Specifies whether the upper or lower triangular part of the */
/*          symmetric matrix A is to be referenced. */
/*          = 'U':  Upper triangular part of A is referenced */
/*          = 'L':  Lower triangular part of A is referenced */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0.  When N = 0, SLANSY is */
/*          set to zero. */

/*  A       (input) REAL array, dimension (LDA,N) */
/*          The symmetric matrix A.  If UPLO = 'U', the leading n by n */
/*          upper triangular part of A contains the upper triangular part */
/*          of the matrix A, and the strictly lower triangular part of A */
/*          is not referenced.  If UPLO = 'L', the leading n by n lower */
/*          triangular part of A contains the lower triangular part of */
/*          the matrix A, and the strictly upper triangular part of A is */
/*          not referenced. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(N,1). */

/*  WORK    (workspace) REAL array, dimension (MAX(1,LWORK)), */
/*          where LWORK >= N when NORM = 'I' or '1' or 'O'; otherwise, */
/*          WORK is not referenced. */

/* ===================================================================== */

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

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --work;

    /* Function Body */
    if (*n == 0) {
	value = 0.f;
    } else if (lsame_(norm, "M")) {

/*        Find max(abs(A(i,j))). */

	value = 0.f;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
/* Computing MAX */
		    r__2 = value, r__3 = (r__1 = a[i__ + j * a_dim1], dabs(
			    r__1));
		    value = dmax(r__2,r__3);
/* L10: */
		}
/* L20: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = j; i__ <= i__2; ++i__) {
/* Computing MAX */
		    r__2 = value, r__3 = (r__1 = a[i__ + j * a_dim1], dabs(
			    r__1));
		    value = dmax(r__2,r__3);
/* L30: */
		}
/* L40: */
	    }
	}
    } else if (lsame_(norm, "I") || lsame_(norm, "O") || *(unsigned char *)norm == '1') {

/*        Find normI(A) ( = norm1(A), since A is symmetric). */

	value = 0.f;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = 0.f;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    absa = (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		    sum += absa;
		    work[i__] += absa;
/* L50: */
		}
		work[j] = sum + (r__1 = a[j + j * a_dim1], dabs(r__1));
/* L60: */
	    }
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
/* Computing MAX */
		r__1 = value, r__2 = work[i__];
		value = dmax(r__1,r__2);
/* L70: */
	    }
	} else {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] = 0.f;
/* L80: */
	    }
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = work[j] + (r__1 = a[j + j * a_dim1], dabs(r__1));
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    absa = (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		    sum += absa;
		    work[i__] += absa;
/* L90: */
		}
		value = dmax(value,sum);
/* L100: */
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {

/*        Find normF(A). */

	scale = 0.f;
	sum = 1.f;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		i__2 = j - 1;
		slassq_(&i__2, &a[j * a_dim1 + 1], &c__1, &scale, &sum);
/* L110: */
	    }
	} else {
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j;
		slassq_(&i__2, &a[j + 1 + j * a_dim1], &c__1, &scale, &sum);
/* L120: */
	    }
	}
	sum *= 2;
	i__1 = *lda + 1;
	slassq_(n, &a[a_offset], &i__1, &scale, &sum);
	value = scale * sqrt(sum);
    }

    ret_val = value;
    return ret_val;

/*     End of SLANSY */

} /* slansy_ */
