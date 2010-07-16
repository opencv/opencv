/* ssyrk.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int ssyrk_(char *uplo, char *trans, integer *n, integer *k, 
	real *alpha, real *a, integer *lda, real *beta, real *c__, integer *
	ldc)
{
    /* System generated locals */
    integer a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;

    /* Local variables */
    integer i__, j, l, info;
    real temp;
    extern logical lsame_(char *, char *);
    integer nrowa;
    logical upper;
    extern /* Subroutine */ int xerbla_(char *, integer *);

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SSYRK  performs one of the symmetric rank k operations */

/*     C := alpha*A*A' + beta*C, */

/*  or */

/*     C := alpha*A'*A + beta*C, */

/*  where  alpha and beta  are scalars, C is an  n by n  symmetric matrix */
/*  and  A  is an  n by k  matrix in the first case and a  k by n  matrix */
/*  in the second case. */

/*  Arguments */
/*  ========== */

/*  UPLO   - CHARACTER*1. */
/*           On  entry,   UPLO  specifies  whether  the  upper  or  lower */
/*           triangular  part  of the  array  C  is to be  referenced  as */
/*           follows: */

/*              UPLO = 'U' or 'u'   Only the  upper triangular part of  C */
/*                                  is to be referenced. */

/*              UPLO = 'L' or 'l'   Only the  lower triangular part of  C */
/*                                  is to be referenced. */

/*           Unchanged on exit. */

/*  TRANS  - CHARACTER*1. */
/*           On entry,  TRANS  specifies the operation to be performed as */
/*           follows: */

/*              TRANS = 'N' or 'n'   C := alpha*A*A' + beta*C. */

/*              TRANS = 'T' or 't'   C := alpha*A'*A + beta*C. */

/*              TRANS = 'C' or 'c'   C := alpha*A'*A + beta*C. */

/*           Unchanged on exit. */

/*  N      - INTEGER. */
/*           On entry,  N specifies the order of the matrix C.  N must be */
/*           at least zero. */
/*           Unchanged on exit. */

/*  K      - INTEGER. */
/*           On entry with  TRANS = 'N' or 'n',  K  specifies  the number */
/*           of  columns   of  the   matrix   A,   and  on   entry   with */
/*           TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number */
/*           of rows of the matrix  A.  K must be at least zero. */
/*           Unchanged on exit. */

/*  ALPHA  - REAL            . */
/*           On entry, ALPHA specifies the scalar alpha. */
/*           Unchanged on exit. */

/*  A      - REAL             array of DIMENSION ( LDA, ka ), where ka is */
/*           k  when  TRANS = 'N' or 'n',  and is  n  otherwise. */
/*           Before entry with  TRANS = 'N' or 'n',  the  leading  n by k */
/*           part of the array  A  must contain the matrix  A,  otherwise */
/*           the leading  k by n  part of the array  A  must contain  the */
/*           matrix A. */
/*           Unchanged on exit. */

/*  LDA    - INTEGER. */
/*           On entry, LDA specifies the first dimension of A as declared */
/*           in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n' */
/*           then  LDA must be at least  max( 1, n ), otherwise  LDA must */
/*           be at least  max( 1, k ). */
/*           Unchanged on exit. */

/*  BETA   - REAL            . */
/*           On entry, BETA specifies the scalar beta. */
/*           Unchanged on exit. */

/*  C      - REAL             array of DIMENSION ( LDC, n ). */
/*           Before entry  with  UPLO = 'U' or 'u',  the leading  n by n */
/*           upper triangular part of the array C must contain the upper */
/*           triangular part  of the  symmetric matrix  and the strictly */
/*           lower triangular part of C is not referenced.  On exit, the */
/*           upper triangular part of the array  C is overwritten by the */
/*           upper triangular part of the updated matrix. */
/*           Before entry  with  UPLO = 'L' or 'l',  the leading  n by n */
/*           lower triangular part of the array C must contain the lower */
/*           triangular part  of the  symmetric matrix  and the strictly */
/*           upper triangular part of C is not referenced.  On exit, the */
/*           lower triangular part of the array  C is overwritten by the */
/*           lower triangular part of the updated matrix. */

/*  LDC    - INTEGER. */
/*           On entry, LDC specifies the first dimension of C as declared */
/*           in  the  calling  (sub)  program.   LDC  must  be  at  least */
/*           max( 1, n ). */
/*           Unchanged on exit. */


/*  Level 3 Blas routine. */

/*  -- Written on 8-February-1989. */
/*     Jack Dongarra, Argonne National Laboratory. */
/*     Iain Duff, AERE Harwell. */
/*     Jeremy Du Croz, Numerical Algorithms Group Ltd. */
/*     Sven Hammarling, Numerical Algorithms Group Ltd. */


/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Parameters .. */
/*     .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    /* Function Body */
    if (lsame_(trans, "N")) {
	nrowa = *n;
    } else {
	nrowa = *k;
    }
    upper = lsame_(uplo, "U");

    info = 0;
    if (! upper && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, 
	    "T") && ! lsame_(trans, "C")) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*k < 0) {
	info = 4;
    } else if (*lda < max(1,nrowa)) {
	info = 7;
    } else if (*ldc < max(1,*n)) {
	info = 10;
    }
    if (info != 0) {
	xerbla_("SSYRK ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {
	return 0;
    }

/*     And when  alpha.eq.zero. */

    if (*alpha == 0.f) {
	if (upper) {
	    if (*beta == 0.f) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
/* L10: */
		    }
/* L20: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L30: */
		    }
/* L40: */
		}
	    }
	} else {
	    if (*beta == 0.f) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
/* L50: */
		    }
/* L60: */
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L70: */
		    }
/* L80: */
		}
	    }
	}
	return 0;
    }

/*     Start the operations. */

    if (lsame_(trans, "N")) {

/*        Form  C := alpha*A*A' + beta*C. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.f) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
/* L90: */
		    }
		} else if (*beta != 1.f) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L100: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.f) {
			temp = *alpha * a[j + l * a_dim1];
			i__3 = j;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l * 
				    a_dim1];
/* L110: */
			}
		    }
/* L120: */
		}
/* L130: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.f) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
/* L140: */
		    }
		} else if (*beta != 1.f) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
/* L150: */
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.f) {
			temp = *alpha * a[j + l * a_dim1];
			i__3 = *n;
			for (i__ = j; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l * 
				    a_dim1];
/* L160: */
			}
		    }
/* L170: */
		}
/* L180: */
	    }
	}
    } else {

/*        Form  C := alpha*A'*A + beta*C. */

	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.f;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L190: */
		    }
		    if (*beta == 0.f) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L200: */
		}
/* L210: */
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = j; i__ <= i__2; ++i__) {
		    temp = 0.f;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
/* L220: */
		    }
		    if (*beta == 0.f) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
/* L230: */
		}
/* L240: */
	    }
	}
    }

    return 0;

/*     End of SSYRK . */

} /* ssyrk_ */
