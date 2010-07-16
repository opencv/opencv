/* slagtf.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int slagtf_(integer *n, real *a, real *lambda, real *b, real 
	*c__, real *tol, real *d__, integer *in, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    integer k;
    real tl, eps, piv1, piv2, temp, mult, scale1, scale2;
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAGTF factorizes the matrix (T - lambda*I), where T is an n by n */
/*  tridiagonal matrix and lambda is a scalar, as */

/*     T - lambda*I = PLU, */

/*  where P is a permutation matrix, L is a unit lower tridiagonal matrix */
/*  with at most one non-zero sub-diagonal elements per column and U is */
/*  an upper triangular matrix with at most two non-zero super-diagonal */
/*  elements per column. */

/*  The factorization is obtained by Gaussian elimination with partial */
/*  pivoting and implicit row scaling. */

/*  The parameter LAMBDA is included in the routine so that SLAGTF may */
/*  be used, in conjunction with SLAGTS, to obtain eigenvectors of T by */
/*  inverse iteration. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix T. */

/*  A       (input/output) REAL array, dimension (N) */
/*          On entry, A must contain the diagonal elements of T. */

/*          On exit, A is overwritten by the n diagonal elements of the */
/*          upper triangular matrix U of the factorization of T. */

/*  LAMBDA  (input) REAL */
/*          On entry, the scalar lambda. */

/*  B       (input/output) REAL array, dimension (N-1) */
/*          On entry, B must contain the (n-1) super-diagonal elements of */
/*          T. */

/*          On exit, B is overwritten by the (n-1) super-diagonal */
/*          elements of the matrix U of the factorization of T. */

/*  C       (input/output) REAL array, dimension (N-1) */
/*          On entry, C must contain the (n-1) sub-diagonal elements of */
/*          T. */

/*          On exit, C is overwritten by the (n-1) sub-diagonal elements */
/*          of the matrix L of the factorization of T. */

/*  TOL     (input) REAL */
/*          On entry, a relative tolerance used to indicate whether or */
/*          not the matrix (T - lambda*I) is nearly singular. TOL should */
/*          normally be chose as approximately the largest relative error */
/*          in the elements of T. For example, if the elements of T are */
/*          correct to about 4 significant figures, then TOL should be */
/*          set to about 5*10**(-4). If TOL is supplied as less than eps, */
/*          where eps is the relative machine precision, then the value */
/*          eps is used in place of TOL. */

/*  D       (output) REAL array, dimension (N-2) */
/*          On exit, D is overwritten by the (n-2) second super-diagonal */
/*          elements of the matrix U of the factorization of T. */

/*  IN      (output) INTEGER array, dimension (N) */
/*          On exit, IN contains details of the permutation matrix P. If */
/*          an interchange occurred at the kth step of the elimination, */
/*          then IN(k) = 1, otherwise IN(k) = 0. The element IN(n) */
/*          returns the smallest positive integer j such that */

/*             abs( u(j,j) ).le. norm( (T - lambda*I)(j) )*TOL, */

/*          where norm( A(j) ) denotes the sum of the absolute values of */
/*          the jth row of the matrix A. If no such j exists then IN(n) */
/*          is returned as zero. If IN(n) is returned as positive, then a */
/*          diagonal element of U is small, indicating that */
/*          (T - lambda*I) is singular or nearly singular, */

/*  INFO    (output) INTEGER */
/*          = 0   : successful exit */
/*          .lt. 0: if INFO = -k, the kth argument had an illegal value */

/* ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --in;
    --d__;
    --c__;
    --b;
    --a;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("SLAGTF", &i__1);
	return 0;
    }

    if (*n == 0) {
	return 0;
    }

    a[1] -= *lambda;
    in[*n] = 0;
    if (*n == 1) {
	if (a[1] == 0.f) {
	    in[1] = 1;
	}
	return 0;
    }

    eps = slamch_("Epsilon");

    tl = dmax(*tol,eps);
    scale1 = dabs(a[1]) + dabs(b[1]);
    i__1 = *n - 1;
    for (k = 1; k <= i__1; ++k) {
	a[k + 1] -= *lambda;
	scale2 = (r__1 = c__[k], dabs(r__1)) + (r__2 = a[k + 1], dabs(r__2));
	if (k < *n - 1) {
	    scale2 += (r__1 = b[k + 1], dabs(r__1));
	}
	if (a[k] == 0.f) {
	    piv1 = 0.f;
	} else {
	    piv1 = (r__1 = a[k], dabs(r__1)) / scale1;
	}
	if (c__[k] == 0.f) {
	    in[k] = 0;
	    piv2 = 0.f;
	    scale1 = scale2;
	    if (k < *n - 1) {
		d__[k] = 0.f;
	    }
	} else {
	    piv2 = (r__1 = c__[k], dabs(r__1)) / scale2;
	    if (piv2 <= piv1) {
		in[k] = 0;
		scale1 = scale2;
		c__[k] /= a[k];
		a[k + 1] -= c__[k] * b[k];
		if (k < *n - 1) {
		    d__[k] = 0.f;
		}
	    } else {
		in[k] = 1;
		mult = a[k] / c__[k];
		a[k] = c__[k];
		temp = a[k + 1];
		a[k + 1] = b[k] - mult * temp;
		if (k < *n - 1) {
		    d__[k] = b[k + 1];
		    b[k + 1] = -mult * d__[k];
		}
		b[k] = temp;
		c__[k] = mult;
	    }
	}
	if (dmax(piv1,piv2) <= tl && in[*n] == 0) {
	    in[*n] = k;
	}
/* L10: */
    }
    if ((r__1 = a[*n], dabs(r__1)) <= scale1 * tl && in[*n] == 0) {
	in[*n] = *n;
    }

    return 0;

/*     End of SLAGTF */

} /* slagtf_ */
