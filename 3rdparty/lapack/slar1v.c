/* slar1v.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int slar1v_(integer *n, integer *b1, integer *bn, real *
	lambda, real *d__, real *l, real *ld, real *lld, real *pivmin, real *
	gaptol, real *z__, logical *wantnc, integer *negcnt, real *ztz, real *
	mingma, integer *r__, integer *isuppz, real *nrminv, real *resid, 
	real *rqcorr, real *work)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2, r__3;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__;
    real s;
    integer r1, r2;
    real eps, tmp;
    integer neg1, neg2, indp, inds;
    real dplus;
    extern doublereal slamch_(char *);
    integer indlpl, indumn;
    extern logical sisnan_(real *);
    real dminus;
    logical sawnan1, sawnan2;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAR1V computes the (scaled) r-th column of the inverse of */
/*  the sumbmatrix in rows B1 through BN of the tridiagonal matrix */
/*  L D L^T - sigma I. When sigma is close to an eigenvalue, the */
/*  computed vector is an accurate eigenvector. Usually, r corresponds */
/*  to the index where the eigenvector is largest in magnitude. */
/*  The following steps accomplish this computation : */
/*  (a) Stationary qd transform,  L D L^T - sigma I = L(+) D(+) L(+)^T, */
/*  (b) Progressive qd transform, L D L^T - sigma I = U(-) D(-) U(-)^T, */
/*  (c) Computation of the diagonal elements of the inverse of */
/*      L D L^T - sigma I by combining the above transforms, and choosing */
/*      r as the index where the diagonal of the inverse is (one of the) */
/*      largest in magnitude. */
/*  (d) Computation of the (scaled) r-th column of the inverse using the */
/*      twisted factorization obtained by combining the top part of the */
/*      the stationary and the bottom part of the progressive transform. */

/*  Arguments */
/*  ========= */

/*  N        (input) INTEGER */
/*           The order of the matrix L D L^T. */

/*  B1       (input) INTEGER */
/*           First index of the submatrix of L D L^T. */

/*  BN       (input) INTEGER */
/*           Last index of the submatrix of L D L^T. */

/*  LAMBDA    (input) REAL */
/*           The shift. In order to compute an accurate eigenvector, */
/*           LAMBDA should be a good approximation to an eigenvalue */
/*           of L D L^T. */

/*  L        (input) REAL             array, dimension (N-1) */
/*           The (n-1) subdiagonal elements of the unit bidiagonal matrix */
/*           L, in elements 1 to N-1. */

/*  D        (input) REAL             array, dimension (N) */
/*           The n diagonal elements of the diagonal matrix D. */

/*  LD       (input) REAL             array, dimension (N-1) */
/*           The n-1 elements L(i)*D(i). */

/*  LLD      (input) REAL             array, dimension (N-1) */
/*           The n-1 elements L(i)*L(i)*D(i). */

/*  PIVMIN   (input) REAL */
/*           The minimum pivot in the Sturm sequence. */

/*  GAPTOL   (input) REAL */
/*           Tolerance that indicates when eigenvector entries are negligible */
/*           w.r.t. their contribution to the residual. */

/*  Z        (input/output) REAL             array, dimension (N) */
/*           On input, all entries of Z must be set to 0. */
/*           On output, Z contains the (scaled) r-th column of the */
/*           inverse. The scaling is such that Z(R) equals 1. */

/*  WANTNC   (input) LOGICAL */
/*           Specifies whether NEGCNT has to be computed. */

/*  NEGCNT   (output) INTEGER */
/*           If WANTNC is .TRUE. then NEGCNT = the number of pivots < pivmin */
/*           in the  matrix factorization L D L^T, and NEGCNT = -1 otherwise. */

/*  ZTZ      (output) REAL */
/*           The square of the 2-norm of Z. */

/*  MINGMA   (output) REAL */
/*           The reciprocal of the largest (in magnitude) diagonal */
/*           element of the inverse of L D L^T - sigma I. */

/*  R        (input/output) INTEGER */
/*           The twist index for the twisted factorization used to */
/*           compute Z. */
/*           On input, 0 <= R <= N. If R is input as 0, R is set to */
/*           the index where (L D L^T - sigma I)^{-1} is largest */
/*           in magnitude. If 1 <= R <= N, R is unchanged. */
/*           On output, R contains the twist index used to compute Z. */
/*           Ideally, R designates the position of the maximum entry in the */
/*           eigenvector. */

/*  ISUPPZ   (output) INTEGER array, dimension (2) */
/*           The support of the vector in Z, i.e., the vector Z is */
/*           nonzero only in elements ISUPPZ(1) through ISUPPZ( 2 ). */

/*  NRMINV   (output) REAL */
/*           NRMINV = 1/SQRT( ZTZ ) */

/*  RESID    (output) REAL */
/*           The residual of the FP vector. */
/*           RESID = ABS( MINGMA )/SQRT( ZTZ ) */

/*  RQCORR   (output) REAL */
/*           The Rayleigh Quotient correction to LAMBDA. */
/*           RQCORR = MINGMA*TMP */

/*  WORK     (workspace) REAL             array, dimension (4*N) */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Beresford Parlett, University of California, Berkeley, USA */
/*     Jim Demmel, University of California, Berkeley, USA */
/*     Inderjit Dhillon, University of Texas, Austin, USA */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Christof Voemel, University of California, Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --work;
    --isuppz;
    --z__;
    --lld;
    --ld;
    --l;
    --d__;

    /* Function Body */
    eps = slamch_("Precision");
    if (*r__ == 0) {
	r1 = *b1;
	r2 = *bn;
    } else {
	r1 = *r__;
	r2 = *r__;
    }
/*     Storage for LPLUS */
    indlpl = 0;
/*     Storage for UMINUS */
    indumn = *n;
    inds = (*n << 1) + 1;
    indp = *n * 3 + 1;
    if (*b1 == 1) {
	work[inds] = 0.f;
    } else {
	work[inds + *b1 - 1] = lld[*b1 - 1];
    }

/*     Compute the stationary transform (using the differential form) */
/*     until the index R2. */

    sawnan1 = FALSE_;
    neg1 = 0;
    s = work[inds + *b1 - 1] - *lambda;
    i__1 = r1 - 1;
    for (i__ = *b1; i__ <= i__1; ++i__) {
	dplus = d__[i__] + s;
	work[indlpl + i__] = ld[i__] / dplus;
	if (dplus < 0.f) {
	    ++neg1;
	}
	work[inds + i__] = s * work[indlpl + i__] * l[i__];
	s = work[inds + i__] - *lambda;
/* L50: */
    }
    sawnan1 = sisnan_(&s);
    if (sawnan1) {
	goto L60;
    }
    i__1 = r2 - 1;
    for (i__ = r1; i__ <= i__1; ++i__) {
	dplus = d__[i__] + s;
	work[indlpl + i__] = ld[i__] / dplus;
	work[inds + i__] = s * work[indlpl + i__] * l[i__];
	s = work[inds + i__] - *lambda;
/* L51: */
    }
    sawnan1 = sisnan_(&s);

L60:
    if (sawnan1) {
/*        Runs a slower version of the above loop if a NaN is detected */
	neg1 = 0;
	s = work[inds + *b1 - 1] - *lambda;
	i__1 = r1 - 1;
	for (i__ = *b1; i__ <= i__1; ++i__) {
	    dplus = d__[i__] + s;
	    if (dabs(dplus) < *pivmin) {
		dplus = -(*pivmin);
	    }
	    work[indlpl + i__] = ld[i__] / dplus;
	    if (dplus < 0.f) {
		++neg1;
	    }
	    work[inds + i__] = s * work[indlpl + i__] * l[i__];
	    if (work[indlpl + i__] == 0.f) {
		work[inds + i__] = lld[i__];
	    }
	    s = work[inds + i__] - *lambda;
/* L70: */
	}
	i__1 = r2 - 1;
	for (i__ = r1; i__ <= i__1; ++i__) {
	    dplus = d__[i__] + s;
	    if (dabs(dplus) < *pivmin) {
		dplus = -(*pivmin);
	    }
	    work[indlpl + i__] = ld[i__] / dplus;
	    work[inds + i__] = s * work[indlpl + i__] * l[i__];
	    if (work[indlpl + i__] == 0.f) {
		work[inds + i__] = lld[i__];
	    }
	    s = work[inds + i__] - *lambda;
/* L71: */
	}
    }

/*     Compute the progressive transform (using the differential form) */
/*     until the index R1 */

    sawnan2 = FALSE_;
    neg2 = 0;
    work[indp + *bn - 1] = d__[*bn] - *lambda;
    i__1 = r1;
    for (i__ = *bn - 1; i__ >= i__1; --i__) {
	dminus = lld[i__] + work[indp + i__];
	tmp = d__[i__] / dminus;
	if (dminus < 0.f) {
	    ++neg2;
	}
	work[indumn + i__] = l[i__] * tmp;
	work[indp + i__ - 1] = work[indp + i__] * tmp - *lambda;
/* L80: */
    }
    tmp = work[indp + r1 - 1];
    sawnan2 = sisnan_(&tmp);
    if (sawnan2) {
/*        Runs a slower version of the above loop if a NaN is detected */
	neg2 = 0;
	i__1 = r1;
	for (i__ = *bn - 1; i__ >= i__1; --i__) {
	    dminus = lld[i__] + work[indp + i__];
	    if (dabs(dminus) < *pivmin) {
		dminus = -(*pivmin);
	    }
	    tmp = d__[i__] / dminus;
	    if (dminus < 0.f) {
		++neg2;
	    }
	    work[indumn + i__] = l[i__] * tmp;
	    work[indp + i__ - 1] = work[indp + i__] * tmp - *lambda;
	    if (tmp == 0.f) {
		work[indp + i__ - 1] = d__[i__] - *lambda;
	    }
/* L100: */
	}
    }

/*     Find the index (from R1 to R2) of the largest (in magnitude) */
/*     diagonal element of the inverse */

    *mingma = work[inds + r1 - 1] + work[indp + r1 - 1];
    if (*mingma < 0.f) {
	++neg1;
    }
    if (*wantnc) {
	*negcnt = neg1 + neg2;
    } else {
	*negcnt = -1;
    }
    if (dabs(*mingma) == 0.f) {
	*mingma = eps * work[inds + r1 - 1];
    }
    *r__ = r1;
    i__1 = r2 - 1;
    for (i__ = r1; i__ <= i__1; ++i__) {
	tmp = work[inds + i__] + work[indp + i__];
	if (tmp == 0.f) {
	    tmp = eps * work[inds + i__];
	}
	if (dabs(tmp) <= dabs(*mingma)) {
	    *mingma = tmp;
	    *r__ = i__ + 1;
	}
/* L110: */
    }

/*     Compute the FP vector: solve N^T v = e_r */

    isuppz[1] = *b1;
    isuppz[2] = *bn;
    z__[*r__] = 1.f;
    *ztz = 1.f;

/*     Compute the FP vector upwards from R */

    if (! sawnan1 && ! sawnan2) {
	i__1 = *b1;
	for (i__ = *r__ - 1; i__ >= i__1; --i__) {
	    z__[i__] = -(work[indlpl + i__] * z__[i__ + 1]);
	    if (((r__1 = z__[i__], dabs(r__1)) + (r__2 = z__[i__ + 1], dabs(
		    r__2))) * (r__3 = ld[i__], dabs(r__3)) < *gaptol) {
		z__[i__] = 0.f;
		isuppz[1] = i__ + 1;
		goto L220;
	    }
	    *ztz += z__[i__] * z__[i__];
/* L210: */
	}
L220:
	;
    } else {
/*        Run slower loop if NaN occurred. */
	i__1 = *b1;
	for (i__ = *r__ - 1; i__ >= i__1; --i__) {
	    if (z__[i__ + 1] == 0.f) {
		z__[i__] = -(ld[i__ + 1] / ld[i__]) * z__[i__ + 2];
	    } else {
		z__[i__] = -(work[indlpl + i__] * z__[i__ + 1]);
	    }
	    if (((r__1 = z__[i__], dabs(r__1)) + (r__2 = z__[i__ + 1], dabs(
		    r__2))) * (r__3 = ld[i__], dabs(r__3)) < *gaptol) {
		z__[i__] = 0.f;
		isuppz[1] = i__ + 1;
		goto L240;
	    }
	    *ztz += z__[i__] * z__[i__];
/* L230: */
	}
L240:
	;
    }
/*     Compute the FP vector downwards from R in blocks of size BLKSIZ */
    if (! sawnan1 && ! sawnan2) {
	i__1 = *bn - 1;
	for (i__ = *r__; i__ <= i__1; ++i__) {
	    z__[i__ + 1] = -(work[indumn + i__] * z__[i__]);
	    if (((r__1 = z__[i__], dabs(r__1)) + (r__2 = z__[i__ + 1], dabs(
		    r__2))) * (r__3 = ld[i__], dabs(r__3)) < *gaptol) {
		z__[i__ + 1] = 0.f;
		isuppz[2] = i__;
		goto L260;
	    }
	    *ztz += z__[i__ + 1] * z__[i__ + 1];
/* L250: */
	}
L260:
	;
    } else {
/*        Run slower loop if NaN occurred. */
	i__1 = *bn - 1;
	for (i__ = *r__; i__ <= i__1; ++i__) {
	    if (z__[i__] == 0.f) {
		z__[i__ + 1] = -(ld[i__ - 1] / ld[i__]) * z__[i__ - 1];
	    } else {
		z__[i__ + 1] = -(work[indumn + i__] * z__[i__]);
	    }
	    if (((r__1 = z__[i__], dabs(r__1)) + (r__2 = z__[i__ + 1], dabs(
		    r__2))) * (r__3 = ld[i__], dabs(r__3)) < *gaptol) {
		z__[i__ + 1] = 0.f;
		isuppz[2] = i__;
		goto L280;
	    }
	    *ztz += z__[i__ + 1] * z__[i__ + 1];
/* L270: */
	}
L280:
	;
    }

/*     Compute quantities for convergence test */

    tmp = 1.f / *ztz;
    *nrminv = sqrt(tmp);
    *resid = dabs(*mingma) * *nrminv;
    *rqcorr = *mingma * tmp;


    return 0;

/*     End of SLAR1V */

} /* slar1v_ */
