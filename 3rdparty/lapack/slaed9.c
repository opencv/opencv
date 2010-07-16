/* slaed9.f -- translated by f2c (version 20061008).
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

/* Subroutine */ int slaed9_(integer *k, integer *kstart, integer *kstop, 
	integer *n, real *d__, real *q, integer *ldq, real *rho, real *dlamda, 
	 real *w, real *s, integer *lds, integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, s_dim1, s_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    integer i__, j;
    real temp;
    extern doublereal snrm2_(integer *, real *, integer *);
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *), slaed4_(integer *, integer *, real *, real *, real *, 
	    real *, real *, integer *);
    extern doublereal slamc3_(real *, real *);
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

/*  SLAED9 finds the roots of the secular equation, as defined by the */
/*  values in D, Z, and RHO, between KSTART and KSTOP.  It makes the */
/*  appropriate calls to SLAED4 and then stores the new matrix of */
/*  eigenvectors for use in calculating the next level of Z vectors. */

/*  Arguments */
/*  ========= */

/*  K       (input) INTEGER */
/*          The number of terms in the rational function to be solved by */
/*          SLAED4.  K >= 0. */

/*  KSTART  (input) INTEGER */
/*  KSTOP   (input) INTEGER */
/*          The updated eigenvalues Lambda(I), KSTART <= I <= KSTOP */
/*          are to be computed.  1 <= KSTART <= KSTOP <= K. */

/*  N       (input) INTEGER */
/*          The number of rows and columns in the Q matrix. */
/*          N >= K (delation may result in N > K). */

/*  D       (output) REAL array, dimension (N) */
/*          D(I) contains the updated eigenvalues */
/*          for KSTART <= I <= KSTOP. */

/*  Q       (workspace) REAL array, dimension (LDQ,N) */

/*  LDQ     (input) INTEGER */
/*          The leading dimension of the array Q.  LDQ >= max( 1, N ). */

/*  RHO     (input) REAL */
/*          The value of the parameter in the rank one update equation. */
/*          RHO >= 0 required. */

/*  DLAMDA  (input) REAL array, dimension (K) */
/*          The first K elements of this array contain the old roots */
/*          of the deflated updating problem.  These are the poles */
/*          of the secular equation. */

/*  W       (input) REAL array, dimension (K) */
/*          The first K elements of this array contain the components */
/*          of the deflation-adjusted updating vector. */

/*  S       (output) REAL array, dimension (LDS, K) */
/*          Will contain the eigenvectors of the repaired matrix which */
/*          will be stored for subsequent Z vector calculation and */
/*          multiplied by the previously accumulated eigenvectors */
/*          to update the system. */

/*  LDS     (input) INTEGER */
/*          The leading dimension of S.  LDS >= max( 1, K ). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  if INFO = 1, an eigenvalue did not converge */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Jeff Rutter, Computer Science Division, University of California */
/*     at Berkeley, USA */

/*  ===================================================================== */

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
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --dlamda;
    --w;
    s_dim1 = *lds;
    s_offset = 1 + s_dim1;
    s -= s_offset;

    /* Function Body */
    *info = 0;

    if (*k < 0) {
	*info = -1;
    } else if (*kstart < 1 || *kstart > max(1,*k)) {
	*info = -2;
    } else if (max(1,*kstop) < *kstart || *kstop > max(1,*k)) {
	*info = -3;
    } else if (*n < *k) {
	*info = -4;
    } else if (*ldq < max(1,*k)) {
	*info = -7;
    } else if (*lds < max(1,*k)) {
	*info = -12;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLAED9", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*k == 0) {
	return 0;
    }

/*     Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can */
/*     be computed with high relative accuracy (barring over/underflow). */
/*     This is a problem on machines without a guard digit in */
/*     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2). */
/*     The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I), */
/*     which on any of these machines zeros out the bottommost */
/*     bit of DLAMDA(I) if it is 1; this makes the subsequent */
/*     subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation */
/*     occurs. On binary machines with a guard digit (almost all */
/*     machines) it does not change DLAMDA(I) at all. On hexadecimal */
/*     and decimal machines with a guard digit, it slightly */
/*     changes the bottommost bits of DLAMDA(I). It does not account */
/*     for hexadecimal or decimal machines without guard digits */
/*     (we know of none). We use a subroutine call to compute */
/*     2*DLAMBDA(I) to prevent optimizing compilers from eliminating */
/*     this code. */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dlamda[i__] = slamc3_(&dlamda[i__], &dlamda[i__]) - dlamda[i__];
/* L10: */
    }

    i__1 = *kstop;
    for (j = *kstart; j <= i__1; ++j) {
	slaed4_(k, &j, &dlamda[1], &w[1], &q[j * q_dim1 + 1], rho, &d__[j], 
		info);

/*        If the zero finder fails, the computation is terminated. */

	if (*info != 0) {
	    goto L120;
	}
/* L20: */
    }

    if (*k == 1 || *k == 2) {
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    i__2 = *k;
	    for (j = 1; j <= i__2; ++j) {
		s[j + i__ * s_dim1] = q[j + i__ * q_dim1];
/* L30: */
	    }
/* L40: */
	}
	goto L120;
    }

/*     Compute updated W. */

    scopy_(k, &w[1], &c__1, &s[s_offset], &c__1);

/*     Initialize W(I) = Q(I,I) */

    i__1 = *ldq + 1;
    scopy_(k, &q[q_offset], &i__1, &w[1], &c__1);
    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w[i__] *= q[i__ + j * q_dim1] / (dlamda[i__] - dlamda[j]);
/* L50: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    w[i__] *= q[i__ + j * q_dim1] / (dlamda[i__] - dlamda[j]);
/* L60: */
	}
/* L70: */
    }
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__1 = sqrt(-w[i__]);
	w[i__] = r_sign(&r__1, &s[i__ + s_dim1]);
/* L80: */
    }

/*     Compute eigenvectors of the modified rank-1 modification. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *k;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    q[i__ + j * q_dim1] = w[i__] / q[i__ + j * q_dim1];
/* L90: */
	}
	temp = snrm2_(k, &q[j * q_dim1 + 1], &c__1);
	i__2 = *k;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    s[i__ + j * s_dim1] = q[i__ + j * q_dim1] / temp;
/* L100: */
	}
/* L110: */
    }

L120:
    return 0;

/*     End of SLAED9 */

} /* slaed9_ */
