#include "clapack.h"

/* Table of constant values */

static integer c__1 = 1;
static integer c__0 = 0;
static real c_b8 = 1.f;

/* Subroutine */ int slasd8_(integer *icompq, integer *k, real *d__, real *
	z__, real *vf, real *vl, real *difl, real *difr, integer *lddifr, 
	real *dsigma, real *work, integer *info)
{
    /* System generated locals */
    integer difr_dim1, difr_offset, i__1, i__2;
    real r__1, r__2;

    /* Builtin functions */
    double sqrt(doublereal), r_sign(real *, real *);

    /* Local variables */
    integer i__, j;
    real dj, rho;
    integer iwk1, iwk2, iwk3;
    real temp;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    integer iwk2i, iwk3i;
    extern doublereal snrm2_(integer *, real *, integer *);
    real diflj, difrj, dsigj;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *);
    extern doublereal slamc3_(real *, real *);
    extern /* Subroutine */ int slasd4_(integer *, integer *, real *, real *, 
	    real *, real *, real *, real *, integer *), xerbla_(char *, 
	    integer *);
    real dsigjp;
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *, 
	    real *, integer *, integer *, real *, integer *, integer *), slaset_(char *, integer *, integer *, real *, real *, 
	    real *, integer *);


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLASD8 finds the square roots of the roots of the secular equation, */
/*  as defined by the values in DSIGMA and Z. It makes the appropriate */
/*  calls to SLASD4, and stores, for each  element in D, the distance */
/*  to its two nearest poles (elements in DSIGMA). It also updates */
/*  the arrays VF and VL, the first and last components of all the */
/*  right singular vectors of the original bidiagonal matrix. */

/*  SLASD8 is called from SLASD6. */

/*  Arguments */
/*  ========= */

/*  ICOMPQ  (input) INTEGER */
/*          Specifies whether singular vectors are to be computed in */
/*          factored form in the calling routine: */
/*          = 0: Compute singular values only. */
/*          = 1: Compute singular vectors in factored form as well. */

/*  K       (input) INTEGER */
/*          The number of terms in the rational function to be solved */
/*          by SLASD4.  K >= 1. */

/*  D       (output) REAL array, dimension ( K ) */
/*          On output, D contains the updated singular values. */

/*  Z       (input) REAL array, dimension ( K ) */
/*          The first K elements of this array contain the components */
/*          of the deflation-adjusted updating row vector. */

/*  VF      (input/output) REAL array, dimension ( K ) */
/*          On entry, VF contains  information passed through DBEDE8. */
/*          On exit, VF contains the first K components of the first */
/*          components of all right singular vectors of the bidiagonal */
/*          matrix. */

/*  VL      (input/output) REAL array, dimension ( K ) */
/*          On entry, VL contains  information passed through DBEDE8. */
/*          On exit, VL contains the first K components of the last */
/*          components of all right singular vectors of the bidiagonal */
/*          matrix. */

/*  DIFL    (output) REAL array, dimension ( K ) */
/*          On exit, DIFL(I) = D(I) - DSIGMA(I). */

/*  DIFR    (output) REAL array, */
/*                   dimension ( LDDIFR, 2 ) if ICOMPQ = 1 and */
/*                   dimension ( K ) if ICOMPQ = 0. */
/*          On exit, DIFR(I,1) = D(I) - DSIGMA(I+1), DIFR(K,1) is not */
/*          defined and will not be referenced. */

/*          If ICOMPQ = 1, DIFR(1:K,2) is an array containing the */
/*          normalizing factors for the right singular vector matrix. */

/*  LDDIFR  (input) INTEGER */
/*          The leading dimension of DIFR, must be at least K. */

/*  DSIGMA  (input) REAL array, dimension ( K ) */
/*          The first K elements of this array contain the old roots */
/*          of the deflated updating problem.  These are the poles */
/*          of the secular equation. */

/*  WORK    (workspace) REAL array, dimension at least 3 * K */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  if INFO = 1, an singular value did not converge */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Huan Ren, Computer Science Division, University of */
/*     California at Berkeley, USA */

/*  ===================================================================== */

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

/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --z__;
    --vf;
    --vl;
    --difl;
    difr_dim1 = *lddifr;
    difr_offset = 1 + difr_dim1;
    difr -= difr_offset;
    --dsigma;
    --work;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*k < 1) {
	*info = -2;
    } else if (*lddifr < *k) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD8", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*k == 1) {
	d__[1] = dabs(z__[1]);
	difl[1] = d__[1];
	if (*icompq == 1) {
	    difl[2] = 1.f;
	    difr[(difr_dim1 << 1) + 1] = 1.f;
	}
	return 0;
    }

/*     Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can */
/*     be computed with high relative accuracy (barring over/underflow). */
/*     This is a problem on machines without a guard digit in */
/*     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2). */
/*     The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I), */
/*     which on any of these machines zeros out the bottommost */
/*     bit of DSIGMA(I) if it is 1; this makes the subsequent */
/*     subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation */
/*     occurs. On binary machines with a guard digit (almost all */
/*     machines) it does not change DSIGMA(I) at all. On hexadecimal */
/*     and decimal machines with a guard digit, it slightly */
/*     changes the bottommost bits of DSIGMA(I). It does not account */
/*     for hexadecimal or decimal machines without guard digits */
/*     (we know of none). We use a subroutine call to compute */
/*     2*DSIGMA(I) to prevent optimizing compilers from eliminating */
/*     this code. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dsigma[i__] = slamc3_(&dsigma[i__], &dsigma[i__]) - dsigma[i__];
/* L10: */
    }

/*     Book keeping. */

    iwk1 = 1;
    iwk2 = iwk1 + *k;
    iwk3 = iwk2 + *k;
    iwk2i = iwk2 - 1;
    iwk3i = iwk3 - 1;

/*     Normalize Z. */

    rho = snrm2_(k, &z__[1], &c__1);
    slascl_("G", &c__0, &c__0, &rho, &c_b8, k, &c__1, &z__[1], k, info);
    rho *= rho;

/*     Initialize WORK(IWK3). */

    slaset_("A", k, &c__1, &c_b8, &c_b8, &work[iwk3], k);

/*     Compute the updated singular values, the arrays DIFL, DIFR, */
/*     and the updated Z. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	slasd4_(k, &j, &dsigma[1], &z__[1], &work[iwk1], &rho, &d__[j], &work[
		iwk2], info);

/*        If the root finder fails, the computation is terminated. */

	if (*info != 0) {
	    return 0;
	}
	work[iwk3i + j] = work[iwk3i + j] * work[j] * work[iwk2i + j];
	difl[j] = -work[j];
	difr[j + difr_dim1] = -work[j + 1];
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[iwk3i + i__] = work[iwk3i + i__] * work[i__] * work[iwk2i + 
		    i__] / (dsigma[i__] - dsigma[j]) / (dsigma[i__] + dsigma[
		    j]);
/* L20: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[iwk3i + i__] = work[iwk3i + i__] * work[i__] * work[iwk2i + 
		    i__] / (dsigma[i__] - dsigma[j]) / (dsigma[i__] + dsigma[
		    j]);
/* L30: */
	}
/* L40: */
    }

/*     Compute updated Z. */

    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	r__2 = sqrt((r__1 = work[iwk3i + i__], dabs(r__1)));
	z__[i__] = r_sign(&r__2, &z__[i__]);
/* L50: */
    }

/*     Update VF and VL. */

    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	diflj = difl[j];
	dj = d__[j];
	dsigj = -dsigma[j];
	if (j < *k) {
	    difrj = -difr[j + difr_dim1];
	    dsigjp = -dsigma[j + 1];
	}
	work[j] = -z__[j] / diflj / (dsigma[j] + dj);
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    work[i__] = z__[i__] / (slamc3_(&dsigma[i__], &dsigj) - diflj) / (
		    dsigma[i__] + dj);
/* L60: */
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[i__] = z__[i__] / (slamc3_(&dsigma[i__], &dsigjp) + difrj) / 
		    (dsigma[i__] + dj);
/* L70: */
	}
	temp = snrm2_(k, &work[1], &c__1);
	work[iwk2i + j] = sdot_(k, &work[1], &c__1, &vf[1], &c__1) / temp;
	work[iwk3i + j] = sdot_(k, &work[1], &c__1, &vl[1], &c__1) / temp;
	if (*icompq == 1) {
	    difr[j + (difr_dim1 << 1)] = temp;
	}
/* L80: */
    }

    scopy_(k, &work[iwk2], &c__1, &vf[1], &c__1);
    scopy_(k, &work[iwk3], &c__1, &vl[1], &c__1);

    return 0;

/*     End of SLASD8 */

} /* slasd8_ */
