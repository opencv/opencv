/* dlals0.f -- translated by f2c (version 20061008).
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

static doublereal c_b5 = -1.;
static integer c__1 = 1;
static doublereal c_b11 = 1.;
static doublereal c_b13 = 0.;
static integer c__0 = 0;

/* Subroutine */ int dlals0_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *nrhs, doublereal *b, integer *ldb, doublereal 
	*bx, integer *ldbx, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, doublereal *givnum, integer *ldgnum, doublereal *
	poles, doublereal *difl, doublereal *difr, doublereal *z__, integer *
	k, doublereal *c__, doublereal *s, doublereal *work, integer *info)
{
    /* System generated locals */
    integer givcol_dim1, givcol_offset, b_dim1, b_offset, bx_dim1, bx_offset, 
	    difr_dim1, difr_offset, givnum_dim1, givnum_offset, poles_dim1, 
	    poles_offset, i__1, i__2;
    doublereal d__1;

    /* Local variables */
    integer i__, j, m, n;
    doublereal dj;
    integer nlp1;
    doublereal temp;
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    doublereal diflj, difrj, dsigj;
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *), dcopy_(integer *, 
	    doublereal *, integer *, doublereal *, integer *);
    extern doublereal dlamc3_(doublereal *, doublereal *);
    extern /* Subroutine */ int dlascl_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    integer *, integer *), dlacpy_(char *, integer *, integer 
	    *, doublereal *, integer *, doublereal *, integer *), 
	    xerbla_(char *, integer *);
    doublereal dsigjp;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLALS0 applies back the multiplying factors of either the left or the */
/*  right singular vector matrix of a diagonal matrix appended by a row */
/*  to the right hand side matrix B in solving the least squares problem */
/*  using the divide-and-conquer SVD approach. */

/*  For the left singular vector matrix, three types of orthogonal */
/*  matrices are involved: */

/*  (1L) Givens rotations: the number of such rotations is GIVPTR; the */
/*       pairs of columns/rows they were applied to are stored in GIVCOL; */
/*       and the C- and S-values of these rotations are stored in GIVNUM. */

/*  (2L) Permutation. The (NL+1)-st row of B is to be moved to the first */
/*       row, and for J=2:N, PERM(J)-th row of B is to be moved to the */
/*       J-th row. */

/*  (3L) The left singular vector matrix of the remaining matrix. */

/*  For the right singular vector matrix, four types of orthogonal */
/*  matrices are involved: */

/*  (1R) The right singular vector matrix of the remaining matrix. */

/*  (2R) If SQRE = 1, one extra Givens rotation to generate the right */
/*       null space. */

/*  (3R) The inverse transformation of (2L). */

/*  (4R) The inverse transformation of (1L). */

/*  Arguments */
/*  ========= */

/*  ICOMPQ (input) INTEGER */
/*         Specifies whether singular vectors are to be computed in */
/*         factored form: */
/*         = 0: Left singular vector matrix. */
/*         = 1: Right singular vector matrix. */

/*  NL     (input) INTEGER */
/*         The row dimension of the upper block. NL >= 1. */

/*  NR     (input) INTEGER */
/*         The row dimension of the lower block. NR >= 1. */

/*  SQRE   (input) INTEGER */
/*         = 0: the lower block is an NR-by-NR square matrix. */
/*         = 1: the lower block is an NR-by-(NR+1) rectangular matrix. */

/*         The bidiagonal matrix has row dimension N = NL + NR + 1, */
/*         and column dimension M = N + SQRE. */

/*  NRHS   (input) INTEGER */
/*         The number of columns of B and BX. NRHS must be at least 1. */

/*  B      (input/output) DOUBLE PRECISION array, dimension ( LDB, NRHS ) */
/*         On input, B contains the right hand sides of the least */
/*         squares problem in rows 1 through M. On output, B contains */
/*         the solution X in rows 1 through N. */

/*  LDB    (input) INTEGER */
/*         The leading dimension of B. LDB must be at least */
/*         max(1,MAX( M, N ) ). */

/*  BX     (workspace) DOUBLE PRECISION array, dimension ( LDBX, NRHS ) */

/*  LDBX   (input) INTEGER */
/*         The leading dimension of BX. */

/*  PERM   (input) INTEGER array, dimension ( N ) */
/*         The permutations (from deflation and sorting) applied */
/*         to the two blocks. */

/*  GIVPTR (input) INTEGER */
/*         The number of Givens rotations which took place in this */
/*         subproblem. */

/*  GIVCOL (input) INTEGER array, dimension ( LDGCOL, 2 ) */
/*         Each pair of numbers indicates a pair of rows/columns */
/*         involved in a Givens rotation. */

/*  LDGCOL (input) INTEGER */
/*         The leading dimension of GIVCOL, must be at least N. */

/*  GIVNUM (input) DOUBLE PRECISION array, dimension ( LDGNUM, 2 ) */
/*         Each number indicates the C or S value used in the */
/*         corresponding Givens rotation. */

/*  LDGNUM (input) INTEGER */
/*         The leading dimension of arrays DIFR, POLES and */
/*         GIVNUM, must be at least K. */

/*  POLES  (input) DOUBLE PRECISION array, dimension ( LDGNUM, 2 ) */
/*         On entry, POLES(1:K, 1) contains the new singular */
/*         values obtained from solving the secular equation, and */
/*         POLES(1:K, 2) is an array containing the poles in the secular */
/*         equation. */

/*  DIFL   (input) DOUBLE PRECISION array, dimension ( K ). */
/*         On entry, DIFL(I) is the distance between I-th updated */
/*         (undeflated) singular value and the I-th (undeflated) old */
/*         singular value. */

/*  DIFR   (input) DOUBLE PRECISION array, dimension ( LDGNUM, 2 ). */
/*         On entry, DIFR(I, 1) contains the distances between I-th */
/*         updated (undeflated) singular value and the I+1-th */
/*         (undeflated) old singular value. And DIFR(I, 2) is the */
/*         normalizing factor for the I-th right singular vector. */

/*  Z      (input) DOUBLE PRECISION array, dimension ( K ) */
/*         Contain the components of the deflation-adjusted updating row */
/*         vector. */

/*  K      (input) INTEGER */
/*         Contains the dimension of the non-deflated matrix, */
/*         This is the order of the related secular equation. 1 <= K <=N. */

/*  C      (input) DOUBLE PRECISION */
/*         C contains garbage if SQRE =0 and the C-value of a Givens */
/*         rotation related to the right null space if SQRE = 1. */

/*  S      (input) DOUBLE PRECISION */
/*         S contains garbage if SQRE =0 and the S-value of a Givens */
/*         rotation related to the right null space if SQRE = 1. */

/*  WORK   (workspace) DOUBLE PRECISION array, dimension ( K ) */

/*  INFO   (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Ren-Cang Li, Computer Science Division, University of */
/*       California at Berkeley, USA */
/*     Osni Marques, LBNL/NERSC, USA */

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
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    bx_dim1 = *ldbx;
    bx_offset = 1 + bx_dim1;
    bx -= bx_offset;
    --perm;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    difr_dim1 = *ldgnum;
    difr_offset = 1 + difr_dim1;
    difr -= difr_offset;
    poles_dim1 = *ldgnum;
    poles_offset = 1 + poles_dim1;
    poles -= poles_offset;
    givnum_dim1 = *ldgnum;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;
    --difl;
    --z__;
    --work;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*nl < 1) {
	*info = -2;
    } else if (*nr < 1) {
	*info = -3;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -4;
    }

    n = *nl + *nr + 1;

    if (*nrhs < 1) {
	*info = -5;
    } else if (*ldb < n) {
	*info = -7;
    } else if (*ldbx < n) {
	*info = -9;
    } else if (*givptr < 0) {
	*info = -11;
    } else if (*ldgcol < n) {
	*info = -13;
    } else if (*ldgnum < n) {
	*info = -15;
    } else if (*k < 1) {
	*info = -20;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLALS0", &i__1);
	return 0;
    }

    m = n + *sqre;
    nlp1 = *nl + 1;

    if (*icompq == 0) {

/*        Apply back orthogonal transformations from the left. */

/*        Step (1L): apply back the Givens rotations performed. */

	i__1 = *givptr;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    drot_(nrhs, &b[givcol[i__ + (givcol_dim1 << 1)] + b_dim1], ldb, &
		    b[givcol[i__ + givcol_dim1] + b_dim1], ldb, &givnum[i__ + 
		    (givnum_dim1 << 1)], &givnum[i__ + givnum_dim1]);
/* L10: */
	}

/*        Step (2L): permute rows of B. */

	dcopy_(nrhs, &b[nlp1 + b_dim1], ldb, &bx[bx_dim1 + 1], ldbx);
	i__1 = n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    dcopy_(nrhs, &b[perm[i__] + b_dim1], ldb, &bx[i__ + bx_dim1], 
		    ldbx);
/* L20: */
	}

/*        Step (3L): apply the inverse of the left singular vector */
/*        matrix to BX. */

	if (*k == 1) {
	    dcopy_(nrhs, &bx[bx_offset], ldbx, &b[b_offset], ldb);
	    if (z__[1] < 0.) {
		dscal_(nrhs, &c_b5, &b[b_offset], ldb);
	    }
	} else {
	    i__1 = *k;
	    for (j = 1; j <= i__1; ++j) {
		diflj = difl[j];
		dj = poles[j + poles_dim1];
		dsigj = -poles[j + (poles_dim1 << 1)];
		if (j < *k) {
		    difrj = -difr[j + difr_dim1];
		    dsigjp = -poles[j + 1 + (poles_dim1 << 1)];
		}
		if (z__[j] == 0. || poles[j + (poles_dim1 << 1)] == 0.) {
		    work[j] = 0.;
		} else {
		    work[j] = -poles[j + (poles_dim1 << 1)] * z__[j] / diflj /
			     (poles[j + (poles_dim1 << 1)] + dj);
		}
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    if (z__[i__] == 0. || poles[i__ + (poles_dim1 << 1)] == 
			    0.) {
			work[i__] = 0.;
		    } else {
			work[i__] = poles[i__ + (poles_dim1 << 1)] * z__[i__] 
				/ (dlamc3_(&poles[i__ + (poles_dim1 << 1)], &
				dsigj) - diflj) / (poles[i__ + (poles_dim1 << 
				1)] + dj);
		    }
/* L30: */
		}
		i__2 = *k;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    if (z__[i__] == 0. || poles[i__ + (poles_dim1 << 1)] == 
			    0.) {
			work[i__] = 0.;
		    } else {
			work[i__] = poles[i__ + (poles_dim1 << 1)] * z__[i__] 
				/ (dlamc3_(&poles[i__ + (poles_dim1 << 1)], &
				dsigjp) + difrj) / (poles[i__ + (poles_dim1 <<
				 1)] + dj);
		    }
/* L40: */
		}
		work[1] = -1.;
		temp = dnrm2_(k, &work[1], &c__1);
		dgemv_("T", k, nrhs, &c_b11, &bx[bx_offset], ldbx, &work[1], &
			c__1, &c_b13, &b[j + b_dim1], ldb);
		dlascl_("G", &c__0, &c__0, &temp, &c_b11, &c__1, nrhs, &b[j + 
			b_dim1], ldb, info);
/* L50: */
	    }
	}

/*        Move the deflated rows of BX to B also. */

	if (*k < max(m,n)) {
	    i__1 = n - *k;
	    dlacpy_("A", &i__1, nrhs, &bx[*k + 1 + bx_dim1], ldbx, &b[*k + 1 
		    + b_dim1], ldb);
	}
    } else {

/*        Apply back the right orthogonal transformations. */

/*        Step (1R): apply back the new right singular vector matrix */
/*        to B. */

	if (*k == 1) {
	    dcopy_(nrhs, &b[b_offset], ldb, &bx[bx_offset], ldbx);
	} else {
	    i__1 = *k;
	    for (j = 1; j <= i__1; ++j) {
		dsigj = poles[j + (poles_dim1 << 1)];
		if (z__[j] == 0.) {
		    work[j] = 0.;
		} else {
		    work[j] = -z__[j] / difl[j] / (dsigj + poles[j + 
			    poles_dim1]) / difr[j + (difr_dim1 << 1)];
		}
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    if (z__[j] == 0.) {
			work[i__] = 0.;
		    } else {
			d__1 = -poles[i__ + 1 + (poles_dim1 << 1)];
			work[i__] = z__[j] / (dlamc3_(&dsigj, &d__1) - difr[
				i__ + difr_dim1]) / (dsigj + poles[i__ + 
				poles_dim1]) / difr[i__ + (difr_dim1 << 1)];
		    }
/* L60: */
		}
		i__2 = *k;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    if (z__[j] == 0.) {
			work[i__] = 0.;
		    } else {
			d__1 = -poles[i__ + (poles_dim1 << 1)];
			work[i__] = z__[j] / (dlamc3_(&dsigj, &d__1) - difl[
				i__]) / (dsigj + poles[i__ + poles_dim1]) / 
				difr[i__ + (difr_dim1 << 1)];
		    }
/* L70: */
		}
		dgemv_("T", k, nrhs, &c_b11, &b[b_offset], ldb, &work[1], &
			c__1, &c_b13, &bx[j + bx_dim1], ldbx);
/* L80: */
	    }
	}

/*        Step (2R): if SQRE = 1, apply back the rotation that is */
/*        related to the right null space of the subproblem. */

	if (*sqre == 1) {
	    dcopy_(nrhs, &b[m + b_dim1], ldb, &bx[m + bx_dim1], ldbx);
	    drot_(nrhs, &bx[bx_dim1 + 1], ldbx, &bx[m + bx_dim1], ldbx, c__, 
		    s);
	}
	if (*k < max(m,n)) {
	    i__1 = n - *k;
	    dlacpy_("A", &i__1, nrhs, &b[*k + 1 + b_dim1], ldb, &bx[*k + 1 + 
		    bx_dim1], ldbx);
	}

/*        Step (3R): permute rows of B. */

	dcopy_(nrhs, &bx[bx_dim1 + 1], ldbx, &b[nlp1 + b_dim1], ldb);
	if (*sqre == 1) {
	    dcopy_(nrhs, &bx[m + bx_dim1], ldbx, &b[m + b_dim1], ldb);
	}
	i__1 = n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    dcopy_(nrhs, &bx[i__ + bx_dim1], ldbx, &b[perm[i__] + b_dim1], 
		    ldb);
/* L90: */
	}

/*        Step (4R): apply back the Givens rotations performed. */

	for (i__ = *givptr; i__ >= 1; --i__) {
	    d__1 = -givnum[i__ + givnum_dim1];
	    drot_(nrhs, &b[givcol[i__ + (givcol_dim1 << 1)] + b_dim1], ldb, &
		    b[givcol[i__ + givcol_dim1] + b_dim1], ldb, &givnum[i__ + 
		    (givnum_dim1 << 1)], &d__1);
/* L100: */
	}
    }

    return 0;

/*     End of DLALS0 */

} /* dlals0_ */
