/* sbdsdc.f -- translated by f2c (version 20061008).
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

static integer c__9 = 9;
static integer c__0 = 0;
static real c_b15 = 1.f;
static integer c__1 = 1;
static real c_b29 = 0.f;

/* Subroutine */ int sbdsdc_(char *uplo, char *compq, integer *n, real *d__, 
	real *e, real *u, integer *ldu, real *vt, integer *ldvt, real *q, 
	integer *iq, real *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double r_sign(real *, real *), log(doublereal);

    /* Local variables */
    integer i__, j, k;
    real p, r__;
    integer z__, ic, ii, kk;
    real cs;
    integer is, iu;
    real sn;
    integer nm1;
    real eps;
    integer ivt, difl, difr, ierr, perm, mlvl, sqre;
    extern logical lsame_(char *, char *);
    integer poles;
    extern /* Subroutine */ int slasr_(char *, char *, char *, integer *, 
	    integer *, real *, real *, real *, integer *);
    integer iuplo, nsize, start;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *), sswap_(integer *, real *, integer *, real *, integer *
), slasd0_(integer *, integer *, real *, real *, real *, integer *
, real *, integer *, integer *, integer *, real *, integer *);
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int slasda_(integer *, integer *, integer *, 
	    integer *, real *, real *, real *, integer *, real *, integer *, 
	    real *, real *, real *, real *, integer *, integer *, integer *, 
	    integer *, real *, real *, real *, real *, integer *, integer *), 
	    xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *);
    extern /* Subroutine */ int slascl_(char *, integer *, integer *, real *, 
	    real *, integer *, integer *, real *, integer *, integer *);
    integer givcol;
    extern /* Subroutine */ int slasdq_(char *, integer *, integer *, integer 
	    *, integer *, integer *, real *, real *, real *, integer *, real *
, integer *, real *, integer *, real *, integer *);
    integer icompq;
    extern /* Subroutine */ int slaset_(char *, integer *, integer *, real *, 
	    real *, real *, integer *), slartg_(real *, real *, real *
, real *, real *);
    real orgnrm;
    integer givnum;
    extern doublereal slanst_(char *, integer *, real *, real *);
    integer givptr, qstart, smlsiz, wstart, smlszp;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SBDSDC computes the singular value decomposition (SVD) of a real */
/*  N-by-N (upper or lower) bidiagonal matrix B:  B = U * S * VT, */
/*  using a divide and conquer method, where S is a diagonal matrix */
/*  with non-negative diagonal elements (the singular values of B), and */
/*  U and VT are orthogonal matrices of left and right singular vectors, */
/*  respectively. SBDSDC can be used to compute all singular values, */
/*  and optionally, singular vectors or singular vectors in compact form. */

/*  This code makes very mild assumptions about floating point */
/*  arithmetic. It will work on machines with a guard digit in */
/*  add/subtract, or on those binary machines without guard digits */
/*  which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2. */
/*  It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none.  See SLASD3 for details. */

/*  The code currently calls SLASDQ if singular values only are desired. */
/*  However, it can be slightly modified to compute singular values */
/*  using the divide and conquer method. */

/*  Arguments */
/*  ========= */

/*  UPLO    (input) CHARACTER*1 */
/*          = 'U':  B is upper bidiagonal. */
/*          = 'L':  B is lower bidiagonal. */

/*  COMPQ   (input) CHARACTER*1 */
/*          Specifies whether singular vectors are to be computed */
/*          as follows: */
/*          = 'N':  Compute singular values only; */
/*          = 'P':  Compute singular values and compute singular */
/*                  vectors in compact form; */
/*          = 'I':  Compute singular values and singular vectors. */

/*  N       (input) INTEGER */
/*          The order of the matrix B.  N >= 0. */

/*  D       (input/output) REAL array, dimension (N) */
/*          On entry, the n diagonal elements of the bidiagonal matrix B. */
/*          On exit, if INFO=0, the singular values of B. */

/*  E       (input/output) REAL array, dimension (N-1) */
/*          On entry, the elements of E contain the offdiagonal */
/*          elements of the bidiagonal matrix whose SVD is desired. */
/*          On exit, E has been destroyed. */

/*  U       (output) REAL array, dimension (LDU,N) */
/*          If  COMPQ = 'I', then: */
/*             On exit, if INFO = 0, U contains the left singular vectors */
/*             of the bidiagonal matrix. */
/*          For other values of COMPQ, U is not referenced. */

/*  LDU     (input) INTEGER */
/*          The leading dimension of the array U.  LDU >= 1. */
/*          If singular vectors are desired, then LDU >= max( 1, N ). */

/*  VT      (output) REAL array, dimension (LDVT,N) */
/*          If  COMPQ = 'I', then: */
/*             On exit, if INFO = 0, VT' contains the right singular */
/*             vectors of the bidiagonal matrix. */
/*          For other values of COMPQ, VT is not referenced. */

/*  LDVT    (input) INTEGER */
/*          The leading dimension of the array VT.  LDVT >= 1. */
/*          If singular vectors are desired, then LDVT >= max( 1, N ). */

/*  Q       (output) REAL array, dimension (LDQ) */
/*          If  COMPQ = 'P', then: */
/*             On exit, if INFO = 0, Q and IQ contain the left */
/*             and right singular vectors in a compact form, */
/*             requiring O(N log N) space instead of 2*N**2. */
/*             In particular, Q contains all the REAL data in */
/*             LDQ >= N*(11 + 2*SMLSIZ + 8*INT(LOG_2(N/(SMLSIZ+1)))) */
/*             words of memory, where SMLSIZ is returned by ILAENV and */
/*             is equal to the maximum size of the subproblems at the */
/*             bottom of the computation tree (usually about 25). */
/*          For other values of COMPQ, Q is not referenced. */

/*  IQ      (output) INTEGER array, dimension (LDIQ) */
/*          If  COMPQ = 'P', then: */
/*             On exit, if INFO = 0, Q and IQ contain the left */
/*             and right singular vectors in a compact form, */
/*             requiring O(N log N) space instead of 2*N**2. */
/*             In particular, IQ contains all INTEGER data in */
/*             LDIQ >= N*(3 + 3*INT(LOG_2(N/(SMLSIZ+1)))) */
/*             words of memory, where SMLSIZ is returned by ILAENV and */
/*             is equal to the maximum size of the subproblems at the */
/*             bottom of the computation tree (usually about 25). */
/*          For other values of COMPQ, IQ is not referenced. */

/*  WORK    (workspace) REAL array, dimension (MAX(1,LWORK)) */
/*          If COMPQ = 'N' then LWORK >= (4 * N). */
/*          If COMPQ = 'P' then LWORK >= (6 * N). */
/*          If COMPQ = 'I' then LWORK >= (3 * N**2 + 4 * N). */

/*  IWORK   (workspace) INTEGER array, dimension (8*N) */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*          > 0:  The algorithm failed to compute an singular value. */
/*                The update process of divide and conquer failed. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Huan Ren, Computer Science Division, University of */
/*     California at Berkeley, USA */
/*  ===================================================================== */
/*  Changed dimension statement in comment describing E from (N) to */
/*  (N-1).  Sven, 17 Feb 05. */
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
    --d__;
    --e;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --q;
    --iq;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    iuplo = 0;
    if (lsame_(uplo, "U")) {
	iuplo = 1;
    }
    if (lsame_(uplo, "L")) {
	iuplo = 2;
    }
    if (lsame_(compq, "N")) {
	icompq = 0;
    } else if (lsame_(compq, "P")) {
	icompq = 1;
    } else if (lsame_(compq, "I")) {
	icompq = 2;
    } else {
	icompq = -1;
    }
    if (iuplo == 0) {
	*info = -1;
    } else if (icompq < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ldu < 1 || icompq == 2 && *ldu < *n) {
	*info = -7;
    } else if (*ldvt < 1 || icompq == 2 && *ldvt < *n) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SBDSDC", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }
    smlsiz = ilaenv_(&c__9, "SBDSDC", " ", &c__0, &c__0, &c__0, &c__0);
    if (*n == 1) {
	if (icompq == 1) {
	    q[1] = r_sign(&c_b15, &d__[1]);
	    q[smlsiz * *n + 1] = 1.f;
	} else if (icompq == 2) {
	    u[u_dim1 + 1] = r_sign(&c_b15, &d__[1]);
	    vt[vt_dim1 + 1] = 1.f;
	}
	d__[1] = dabs(d__[1]);
	return 0;
    }
    nm1 = *n - 1;

/*     If matrix lower bidiagonal, rotate to be upper bidiagonal */
/*     by applying Givens rotations on the left */

    wstart = 1;
    qstart = 3;
    if (icompq == 1) {
	scopy_(n, &d__[1], &c__1, &q[1], &c__1);
	i__1 = *n - 1;
	scopy_(&i__1, &e[1], &c__1, &q[*n + 1], &c__1);
    }
    if (iuplo == 2) {
	qstart = 5;
	wstart = (*n << 1) - 1;
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    slartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (icompq == 1) {
		q[i__ + (*n << 1)] = cs;
		q[i__ + *n * 3] = sn;
	    } else if (icompq == 2) {
		work[i__] = cs;
		work[nm1 + i__] = -sn;
	    }
/* L10: */
	}
    }

/*     If ICOMPQ = 0, use SLASDQ to compute the singular values. */

    if (icompq == 0) {
	slasdq_("U", &c__0, n, &c__0, &c__0, &c__0, &d__[1], &e[1], &vt[
		vt_offset], ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[
		wstart], info);
	goto L40;
    }

/*     If N is smaller than the minimum divide size SMLSIZ, then solve */
/*     the problem with another solver. */

    if (*n <= smlsiz) {
	if (icompq == 2) {
	    slaset_("A", n, n, &c_b29, &c_b15, &u[u_offset], ldu);
	    slaset_("A", n, n, &c_b29, &c_b15, &vt[vt_offset], ldvt);
	    slasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &vt[vt_offset]
, ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[
		    wstart], info);
	} else if (icompq == 1) {
	    iu = 1;
	    ivt = iu + *n;
	    slaset_("A", n, n, &c_b29, &c_b15, &q[iu + (qstart - 1) * *n], n);
	    slaset_("A", n, n, &c_b29, &c_b15, &q[ivt + (qstart - 1) * *n], n);
	    slasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &q[ivt + (
		    qstart - 1) * *n], n, &q[iu + (qstart - 1) * *n], n, &q[
		    iu + (qstart - 1) * *n], n, &work[wstart], info);
	}
	goto L40;
    }

    if (icompq == 2) {
	slaset_("A", n, n, &c_b29, &c_b15, &u[u_offset], ldu);
	slaset_("A", n, n, &c_b29, &c_b15, &vt[vt_offset], ldvt);
    }

/*     Scale. */

    orgnrm = slanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.f) {
	return 0;
    }
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, n, &c__1, &d__[1], n, &ierr);
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b15, &nm1, &c__1, &e[1], &nm1, &
	    ierr);

    eps = slamch_("Epsilon");

    mlvl = (integer) (log((real) (*n) / (real) (smlsiz + 1)) / log(2.f)) + 1;
    smlszp = smlsiz + 1;

    if (icompq == 1) {
	iu = 1;
	ivt = smlsiz + 1;
	difl = ivt + smlszp;
	difr = difl + mlvl;
	z__ = difr + (mlvl << 1);
	ic = z__ + mlvl;
	is = ic + 1;
	poles = is + 1;
	givnum = poles + (mlvl << 1);

	k = 1;
	givptr = 2;
	perm = 3;
	givcol = perm + mlvl;
    }

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = d__[i__], dabs(r__1)) < eps) {
	    d__[i__] = r_sign(&eps, &d__[i__]);
	}
/* L20: */
    }

    start = 1;
    sqre = 0;

    i__1 = nm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = e[i__], dabs(r__1)) < eps || i__ == nm1) {

/*        Subproblem found. First determine its size and then */
/*        apply divide and conquer on it. */

	    if (i__ < nm1) {

/*        A subproblem with E(I) small for I < NM1. */

		nsize = i__ - start + 1;
	    } else if ((r__1 = e[i__], dabs(r__1)) >= eps) {

/*        A subproblem with E(NM1) not too small but I = NM1. */

		nsize = *n - start + 1;
	    } else {

/*        A subproblem with E(NM1) small. This implies an */
/*        1-by-1 subproblem at D(N). Solve this 1-by-1 problem */
/*        first. */

		nsize = i__ - start + 1;
		if (icompq == 2) {
		    u[*n + *n * u_dim1] = r_sign(&c_b15, &d__[*n]);
		    vt[*n + *n * vt_dim1] = 1.f;
		} else if (icompq == 1) {
		    q[*n + (qstart - 1) * *n] = r_sign(&c_b15, &d__[*n]);
		    q[*n + (smlsiz + qstart - 1) * *n] = 1.f;
		}
		d__[*n] = (r__1 = d__[*n], dabs(r__1));
	    }
	    if (icompq == 2) {
		slasd0_(&nsize, &sqre, &d__[start], &e[start], &u[start + 
			start * u_dim1], ldu, &vt[start + start * vt_dim1], 
			ldvt, &smlsiz, &iwork[1], &work[wstart], info);
	    } else {
		slasda_(&icompq, &smlsiz, &nsize, &sqre, &d__[start], &e[
			start], &q[start + (iu + qstart - 2) * *n], n, &q[
			start + (ivt + qstart - 2) * *n], &iq[start + k * *n], 
			 &q[start + (difl + qstart - 2) * *n], &q[start + (
			difr + qstart - 2) * *n], &q[start + (z__ + qstart - 
			2) * *n], &q[start + (poles + qstart - 2) * *n], &iq[
			start + givptr * *n], &iq[start + givcol * *n], n, &
			iq[start + perm * *n], &q[start + (givnum + qstart - 
			2) * *n], &q[start + (ic + qstart - 2) * *n], &q[
			start + (is + qstart - 2) * *n], &work[wstart], &
			iwork[1], info);
		if (*info != 0) {
		    return 0;
		}
	    }
	    start = i__ + 1;
	}
/* L30: */
    }

/*     Unscale */

    slascl_("G", &c__0, &c__0, &c_b15, &orgnrm, n, &c__1, &d__[1], n, &ierr);
L40:

/*     Use Selection Sort to minimize swaps of singular vectors */

    i__1 = *n;
    for (ii = 2; ii <= i__1; ++ii) {
	i__ = ii - 1;
	kk = i__;
	p = d__[i__];
	i__2 = *n;
	for (j = ii; j <= i__2; ++j) {
	    if (d__[j] > p) {
		kk = j;
		p = d__[j];
	    }
/* L50: */
	}
	if (kk != i__) {
	    d__[kk] = d__[i__];
	    d__[i__] = p;
	    if (icompq == 1) {
		iq[i__] = kk;
	    } else if (icompq == 2) {
		sswap_(n, &u[i__ * u_dim1 + 1], &c__1, &u[kk * u_dim1 + 1], &
			c__1);
		sswap_(n, &vt[i__ + vt_dim1], ldvt, &vt[kk + vt_dim1], ldvt);
	    }
	} else if (icompq == 1) {
	    iq[i__] = i__;
	}
/* L60: */
    }

/*     If ICOMPQ = 1, use IQ(N,1) as the indicator for UPLO */

    if (icompq == 1) {
	if (iuplo == 1) {
	    iq[*n] = 1;
	} else {
	    iq[*n] = 0;
	}
    }

/*     If B is lower bidiagonal, update U by those Givens rotations */
/*     which rotated B to be upper bidiagonal */

    if (iuplo == 2 && icompq == 2) {
	slasr_("L", "V", "B", n, n, &work[1], &work[*n], &u[u_offset], ldu);
    }

    return 0;

/*     End of SBDSDC */

} /* sbdsdc_ */
