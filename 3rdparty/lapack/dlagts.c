#include "clapack.h"

/* Subroutine */ int dlagts_(integer *job, integer *n, doublereal *a, 
	doublereal *b, doublereal *c__, doublereal *d__, integer *in, 
	doublereal *y, doublereal *tol, integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2, d__3, d__4, d__5;

    /* Builtin functions */
    double d_sign(doublereal *, doublereal *);

    /* Local variables */
    integer k;
    doublereal ak, eps, temp, pert, absak, sfmin;
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, integer *);
    doublereal bignum;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAGTS may be used to solve one of the systems of equations */

/*     (T - lambda*I)*x = y   or   (T - lambda*I)'*x = y, */

/*  where T is an n by n tridiagonal matrix, for x, following the */
/*  factorization of (T - lambda*I) as */

/*     (T - lambda*I) = P*L*U , */

/*  by routine DLAGTF. The choice of equation to be solved is */
/*  controlled by the argument JOB, and in each case there is an option */
/*  to perturb zero or very small diagonal elements of U, this option */
/*  being intended for use in applications such as inverse iteration. */

/*  Arguments */
/*  ========= */

/*  JOB     (input) INTEGER */
/*          Specifies the job to be performed by DLAGTS as follows: */
/*          =  1: The equations  (T - lambda*I)x = y  are to be solved, */
/*                but diagonal elements of U are not to be perturbed. */
/*          = -1: The equations  (T - lambda*I)x = y  are to be solved */
/*                and, if overflow would otherwise occur, the diagonal */
/*                elements of U are to be perturbed. See argument TOL */
/*                below. */
/*          =  2: The equations  (T - lambda*I)'x = y  are to be solved, */
/*                but diagonal elements of U are not to be perturbed. */
/*          = -2: The equations  (T - lambda*I)'x = y  are to be solved */
/*                and, if overflow would otherwise occur, the diagonal */
/*                elements of U are to be perturbed. See argument TOL */
/*                below. */

/*  N       (input) INTEGER */
/*          The order of the matrix T. */

/*  A       (input) DOUBLE PRECISION array, dimension (N) */
/*          On entry, A must contain the diagonal elements of U as */
/*          returned from DLAGTF. */

/*  B       (input) DOUBLE PRECISION array, dimension (N-1) */
/*          On entry, B must contain the first super-diagonal elements of */
/*          U as returned from DLAGTF. */

/*  C       (input) DOUBLE PRECISION array, dimension (N-1) */
/*          On entry, C must contain the sub-diagonal elements of L as */
/*          returned from DLAGTF. */

/*  D       (input) DOUBLE PRECISION array, dimension (N-2) */
/*          On entry, D must contain the second super-diagonal elements */
/*          of U as returned from DLAGTF. */

/*  IN      (input) INTEGER array, dimension (N) */
/*          On entry, IN must contain details of the matrix P as returned */
/*          from DLAGTF. */

/*  Y       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the right hand side vector y. */
/*          On exit, Y is overwritten by the solution vector x. */

/*  TOL     (input/output) DOUBLE PRECISION */
/*          On entry, with  JOB .lt. 0, TOL should be the minimum */
/*          perturbation to be made to very small diagonal elements of U. */
/*          TOL should normally be chosen as about eps*norm(U), where eps */
/*          is the relative machine precision, but if TOL is supplied as */
/*          non-positive, then it is reset to eps*max( abs( u(i,j) ) ). */
/*          If  JOB .gt. 0  then TOL is not referenced. */

/*          On exit, TOL is changed as described above, only if TOL is */
/*          non-positive on entry. Otherwise TOL is unchanged. */

/*  INFO    (output) INTEGER */
/*          = 0   : successful exit */
/*          .lt. 0: if INFO = -i, the i-th argument had an illegal value */
/*          .gt. 0: overflow would occur when computing the INFO(th) */
/*                  element of the solution vector x. This can only occur */
/*                  when JOB is supplied as positive and either means */
/*                  that a diagonal element of U is very small, or that */
/*                  the elements of the right-hand side vector y are very */
/*                  large. */

/*  ===================================================================== */

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
    --y;
    --in;
    --d__;
    --c__;
    --b;
    --a;

    /* Function Body */
    *info = 0;
    if (abs(*job) > 2 || *job == 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLAGTS", &i__1);
	return 0;
    }

    if (*n == 0) {
	return 0;
    }

    eps = dlamch_("Epsilon");
    sfmin = dlamch_("Safe minimum");
    bignum = 1. / sfmin;

    if (*job < 0) {
	if (*tol <= 0.) {
	    *tol = abs(a[1]);
	    if (*n > 1) {
/* Computing MAX */
		d__1 = *tol, d__2 = abs(a[2]), d__1 = max(d__1,d__2), d__2 = 
			abs(b[1]);
		*tol = max(d__1,d__2);
	    }
	    i__1 = *n;
	    for (k = 3; k <= i__1; ++k) {
/* Computing MAX */
		d__4 = *tol, d__5 = (d__1 = a[k], abs(d__1)), d__4 = max(d__4,
			d__5), d__5 = (d__2 = b[k - 1], abs(d__2)), d__4 = 
			max(d__4,d__5), d__5 = (d__3 = d__[k - 2], abs(d__3));
		*tol = max(d__4,d__5);
/* L10: */
	    }
	    *tol *= eps;
	    if (*tol == 0.) {
		*tol = eps;
	    }
	}
    }

    if (abs(*job) == 1) {
	i__1 = *n;
	for (k = 2; k <= i__1; ++k) {
	    if (in[k - 1] == 0) {
		y[k] -= c__[k - 1] * y[k - 1];
	    } else {
		temp = y[k - 1];
		y[k - 1] = y[k];
		y[k] = temp - c__[k - 1] * y[k];
	    }
/* L20: */
	}
	if (*job == 1) {
	    for (k = *n; k >= 1; --k) {
		if (k <= *n - 2) {
		    temp = y[k] - b[k] * y[k + 1] - d__[k] * y[k + 2];
		} else if (k == *n - 1) {
		    temp = y[k] - b[k] * y[k + 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    *info = k;
			    return 0;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			*info = k;
			return 0;
		    }
		}
		y[k] = temp / ak;
/* L30: */
	    }
	} else {
	    for (k = *n; k >= 1; --k) {
		if (k <= *n - 2) {
		    temp = y[k] - b[k] * y[k + 1] - d__[k] * y[k + 2];
		} else if (k == *n - 1) {
		    temp = y[k] - b[k] * y[k + 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		pert = d_sign(tol, &ak);
L40:
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    ak += pert;
			    pert *= 2;
			    goto L40;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			ak += pert;
			pert *= 2;
			goto L40;
		    }
		}
		y[k] = temp / ak;
/* L50: */
	    }
	}
    } else {

/*        Come to here if  JOB = 2 or -2 */

	if (*job == 2) {
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		if (k >= 3) {
		    temp = y[k] - b[k - 1] * y[k - 1] - d__[k - 2] * y[k - 2];
		} else if (k == 2) {
		    temp = y[k] - b[k - 1] * y[k - 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    *info = k;
			    return 0;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			*info = k;
			return 0;
		    }
		}
		y[k] = temp / ak;
/* L60: */
	    }
	} else {
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		if (k >= 3) {
		    temp = y[k] - b[k - 1] * y[k - 1] - d__[k - 2] * y[k - 2];
		} else if (k == 2) {
		    temp = y[k] - b[k - 1] * y[k - 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		pert = d_sign(tol, &ak);
L70:
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    ak += pert;
			    pert *= 2;
			    goto L70;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			ak += pert;
			pert *= 2;
			goto L70;
		    }
		}
		y[k] = temp / ak;
/* L80: */
	    }
	}

	for (k = *n; k >= 2; --k) {
	    if (in[k - 1] == 0) {
		y[k - 1] -= c__[k - 1] * y[k];
	    } else {
		temp = y[k - 1];
		y[k - 1] = y[k];
		y[k] = temp - c__[k - 1] * y[k];
	    }
/* L90: */
	}
    }

/*     End of DLAGTS */

    return 0;
} /* dlagts_ */
