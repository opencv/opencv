#include "clapack.h"

/* Table of constant values */

static integer c__1 = 1;
static doublereal c_b30 = 0.;

/* Subroutine */ int dlasd2_(integer *nl, integer *nr, integer *sqre, integer 
	*k, doublereal *d__, doublereal *z__, doublereal *alpha, doublereal *
	beta, doublereal *u, integer *ldu, doublereal *vt, integer *ldvt, 
	doublereal *dsigma, doublereal *u2, integer *ldu2, doublereal *vt2, 
	integer *ldvt2, integer *idxp, integer *idx, integer *idxc, integer *
	idxq, integer *coltyp, integer *info)
{
    /* System generated locals */
    integer u_dim1, u_offset, u2_dim1, u2_offset, vt_dim1, vt_offset, 
	    vt2_dim1, vt2_offset, i__1;
    doublereal d__1, d__2;

    /* Local variables */
    doublereal c__;
    integer i__, j, m, n;
    doublereal s;
    integer k2;
    doublereal z1;
    integer ct, jp;
    doublereal eps, tau, tol;
    integer psm[4], nlp1, nlp2, idxi, idxj;
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    integer ctot[4], idxjp;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *);
    integer jprev;
    extern doublereal dlapy2_(doublereal *, doublereal *), dlamch_(char *);
    extern /* Subroutine */ int dlamrg_(integer *, integer *, doublereal *, 
	    integer *, integer *, integer *), dlacpy_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, integer *), dlaset_(char *, integer *, integer *, doublereal *, 
	    doublereal *, doublereal *, integer *), xerbla_(char *, 
	    integer *);
    doublereal hlftol;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLASD2 merges the two sets of singular values together into a single */
/*  sorted set.  Then it tries to deflate the size of the problem. */
/*  There are two ways in which deflation can occur:  when two or more */
/*  singular values are close together or if there is a tiny entry in the */
/*  Z vector.  For each such occurrence the order of the related secular */
/*  equation problem is reduced by one. */

/*  DLASD2 is called from DLASD1. */

/*  Arguments */
/*  ========= */

/*  NL     (input) INTEGER */
/*         The row dimension of the upper block.  NL >= 1. */

/*  NR     (input) INTEGER */
/*         The row dimension of the lower block.  NR >= 1. */

/*  SQRE   (input) INTEGER */
/*         = 0: the lower block is an NR-by-NR square matrix. */
/*         = 1: the lower block is an NR-by-(NR+1) rectangular matrix. */

/*         The bidiagonal matrix has N = NL + NR + 1 rows and */
/*         M = N + SQRE >= N columns. */

/*  K      (output) INTEGER */
/*         Contains the dimension of the non-deflated matrix, */
/*         This is the order of the related secular equation. 1 <= K <=N. */

/*  D      (input/output) DOUBLE PRECISION array, dimension(N) */
/*         On entry D contains the singular values of the two submatrices */
/*         to be combined.  On exit D contains the trailing (N-K) updated */
/*         singular values (those which were deflated) sorted into */
/*         increasing order. */

/*  Z      (output) DOUBLE PRECISION array, dimension(N) */
/*         On exit Z contains the updating row vector in the secular */
/*         equation. */

/*  ALPHA  (input) DOUBLE PRECISION */
/*         Contains the diagonal element associated with the added row. */

/*  BETA   (input) DOUBLE PRECISION */
/*         Contains the off-diagonal element associated with the added */
/*         row. */

/*  U      (input/output) DOUBLE PRECISION array, dimension(LDU,N) */
/*         On entry U contains the left singular vectors of two */
/*         submatrices in the two square blocks with corners at (1,1), */
/*         (NL, NL), and (NL+2, NL+2), (N,N). */
/*         On exit U contains the trailing (N-K) updated left singular */
/*         vectors (those which were deflated) in its last N-K columns. */

/*  LDU    (input) INTEGER */
/*         The leading dimension of the array U.  LDU >= N. */

/*  VT     (input/output) DOUBLE PRECISION array, dimension(LDVT,M) */
/*         On entry VT' contains the right singular vectors of two */
/*         submatrices in the two square blocks with corners at (1,1), */
/*         (NL+1, NL+1), and (NL+2, NL+2), (M,M). */
/*         On exit VT' contains the trailing (N-K) updated right singular */
/*         vectors (those which were deflated) in its last N-K columns. */
/*         In case SQRE =1, the last row of VT spans the right null */
/*         space. */

/*  LDVT   (input) INTEGER */
/*         The leading dimension of the array VT.  LDVT >= M. */

/*  DSIGMA (output) DOUBLE PRECISION array, dimension (N) */
/*         Contains a copy of the diagonal elements (K-1 singular values */
/*         and one zero) in the secular equation. */

/*  U2     (output) DOUBLE PRECISION array, dimension(LDU2,N) */
/*         Contains a copy of the first K-1 left singular vectors which */
/*         will be used by DLASD3 in a matrix multiply (DGEMM) to solve */
/*         for the new left singular vectors. U2 is arranged into four */
/*         blocks. The first block contains a column with 1 at NL+1 and */
/*         zero everywhere else; the second block contains non-zero */
/*         entries only at and above NL; the third contains non-zero */
/*         entries only below NL+1; and the fourth is dense. */

/*  LDU2   (input) INTEGER */
/*         The leading dimension of the array U2.  LDU2 >= N. */

/*  VT2    (output) DOUBLE PRECISION array, dimension(LDVT2,N) */
/*         VT2' contains a copy of the first K right singular vectors */
/*         which will be used by DLASD3 in a matrix multiply (DGEMM) to */
/*         solve for the new right singular vectors. VT2 is arranged into */
/*         three blocks. The first block contains a row that corresponds */
/*         to the special 0 diagonal element in SIGMA; the second block */
/*         contains non-zeros only at and before NL +1; the third block */
/*         contains non-zeros only at and after  NL +2. */

/*  LDVT2  (input) INTEGER */
/*         The leading dimension of the array VT2.  LDVT2 >= M. */

/*  IDXP   (workspace) INTEGER array dimension(N) */
/*         This will contain the permutation used to place deflated */
/*         values of D at the end of the array. On output IDXP(2:K) */
/*         points to the nondeflated D-values and IDXP(K+1:N) */
/*         points to the deflated singular values. */

/*  IDX    (workspace) INTEGER array dimension(N) */
/*         This will contain the permutation used to sort the contents of */
/*         D into ascending order. */

/*  IDXC   (output) INTEGER array dimension(N) */
/*         This will contain the permutation used to arrange the columns */
/*         of the deflated U matrix into three groups:  the first group */
/*         contains non-zero entries only at and above NL, the second */
/*         contains non-zero entries only below NL+2, and the third is */
/*         dense. */

/*  IDXQ   (input/output) INTEGER array dimension(N) */
/*         This contains the permutation which separately sorts the two */
/*         sub-problems in D into ascending order.  Note that entries in */
/*         the first hlaf of this permutation must first be moved one */
/*         position backward; and entries in the second half */
/*         must first have NL+1 added to their values. */

/*  COLTYP (workspace/output) INTEGER array dimension(N) */
/*         As workspace, this will contain a label which will indicate */
/*         which of the following types a column in the U2 matrix or a */
/*         row in the VT2 matrix is: */
/*         1 : non-zero in the upper half only */
/*         2 : non-zero in the lower half only */
/*         3 : dense */
/*         4 : deflated */

/*         On exit, it is an array of dimension 4, with COLTYP(I) being */
/*         the dimension of the I-th type columns. */

/*  INFO   (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ming Gu and Huan Ren, Computer Science Division, University of */
/*     California at Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Arrays .. */
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
    --z__;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --dsigma;
    u2_dim1 = *ldu2;
    u2_offset = 1 + u2_dim1;
    u2 -= u2_offset;
    vt2_dim1 = *ldvt2;
    vt2_offset = 1 + vt2_dim1;
    vt2 -= vt2_offset;
    --idxp;
    --idx;
    --idxc;
    --idxq;
    --coltyp;

    /* Function Body */
    *info = 0;

    if (*nl < 1) {
	*info = -1;
    } else if (*nr < 1) {
	*info = -2;
    } else if (*sqre != 1 && *sqre != 0) {
	*info = -3;
    }

    n = *nl + *nr + 1;
    m = n + *sqre;

    if (*ldu < n) {
	*info = -10;
    } else if (*ldvt < m) {
	*info = -12;
    } else if (*ldu2 < n) {
	*info = -15;
    } else if (*ldvt2 < m) {
	*info = -17;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASD2", &i__1);
	return 0;
    }

    nlp1 = *nl + 1;
    nlp2 = *nl + 2;

/*     Generate the first part of the vector Z; and move the singular */
/*     values in the first part of D one position backward. */

    z1 = *alpha * vt[nlp1 + nlp1 * vt_dim1];
    z__[1] = z1;
    for (i__ = *nl; i__ >= 1; --i__) {
	z__[i__ + 1] = *alpha * vt[i__ + nlp1 * vt_dim1];
	d__[i__ + 1] = d__[i__];
	idxq[i__ + 1] = idxq[i__] + 1;
/* L10: */
    }

/*     Generate the second part of the vector Z. */

    i__1 = m;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	z__[i__] = *beta * vt[i__ + nlp2 * vt_dim1];
/* L20: */
    }

/*     Initialize some reference arrays. */

    i__1 = nlp1;
    for (i__ = 2; i__ <= i__1; ++i__) {
	coltyp[i__] = 1;
/* L30: */
    }
    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	coltyp[i__] = 2;
/* L40: */
    }

/*     Sort the singular values into increasing order */

    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	idxq[i__] += nlp1;
/* L50: */
    }

/*     DSIGMA, IDXC, IDXC, and the first column of U2 */
/*     are used as storage space. */

    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	dsigma[i__] = d__[idxq[i__]];
	u2[i__ + u2_dim1] = z__[idxq[i__]];
	idxc[i__] = coltyp[idxq[i__]];
/* L60: */
    }

    dlamrg_(nl, nr, &dsigma[2], &c__1, &c__1, &idx[2]);

    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	idxi = idx[i__] + 1;
	d__[i__] = dsigma[idxi];
	z__[i__] = u2[idxi + u2_dim1];
	coltyp[i__] = idxc[idxi];
/* L70: */
    }

/*     Calculate the allowable deflation tolerance */

    eps = dlamch_("Epsilon");
/* Computing MAX */
    d__1 = abs(*alpha), d__2 = abs(*beta);
    tol = max(d__1,d__2);
/* Computing MAX */
    d__2 = (d__1 = d__[n], abs(d__1));
    tol = eps * 8. * max(d__2,tol);

/*     There are 2 kinds of deflation -- first a value in the z-vector */
/*     is small, second two (or more) singular values are very close */
/*     together (their difference is small). */

/*     If the value in the z-vector is small, we simply permute the */
/*     array so that the corresponding singular value is moved to the */
/*     end. */

/*     If two values in the D-vector are close, we perform a two-sided */
/*     rotation designed to make one of the corresponding z-vector */
/*     entries zero, and then permute the array so that the deflated */
/*     singular value is moved to the end. */

/*     If there are multiple singular values then the problem deflates. */
/*     Here the number of equal singular values are found.  As each equal */
/*     singular value is found, an elementary reflector is computed to */
/*     rotate the corresponding singular subspace so that the */
/*     corresponding components of Z are zero in this new basis. */

    *k = 1;
    k2 = n + 1;
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	if ((d__1 = z__[j], abs(d__1)) <= tol) {

/*           Deflate due to small z component. */

	    --k2;
	    idxp[k2] = j;
	    coltyp[j] = 4;
	    if (j == n) {
		goto L120;
	    }
	} else {
	    jprev = j;
	    goto L90;
	}
/* L80: */
    }
L90:
    j = jprev;
L100:
    ++j;
    if (j > n) {
	goto L110;
    }
    if ((d__1 = z__[j], abs(d__1)) <= tol) {

/*        Deflate due to small z component. */

	--k2;
	idxp[k2] = j;
	coltyp[j] = 4;
    } else {

/*        Check if singular values are close enough to allow deflation. */

	if ((d__1 = d__[j] - d__[jprev], abs(d__1)) <= tol) {

/*           Deflation is possible. */

	    s = z__[jprev];
	    c__ = z__[j];

/*           Find sqrt(a**2+b**2) without overflow or */
/*           destructive underflow. */

	    tau = dlapy2_(&c__, &s);
	    c__ /= tau;
	    s = -s / tau;
	    z__[j] = tau;
	    z__[jprev] = 0.;

/*           Apply back the Givens rotation to the left and right */
/*           singular vector matrices. */

	    idxjp = idxq[idx[jprev] + 1];
	    idxj = idxq[idx[j] + 1];
	    if (idxjp <= nlp1) {
		--idxjp;
	    }
	    if (idxj <= nlp1) {
		--idxj;
	    }
	    drot_(&n, &u[idxjp * u_dim1 + 1], &c__1, &u[idxj * u_dim1 + 1], &
		    c__1, &c__, &s);
	    drot_(&m, &vt[idxjp + vt_dim1], ldvt, &vt[idxj + vt_dim1], ldvt, &
		    c__, &s);
	    if (coltyp[j] != coltyp[jprev]) {
		coltyp[j] = 3;
	    }
	    coltyp[jprev] = 4;
	    --k2;
	    idxp[k2] = jprev;
	    jprev = j;
	} else {
	    ++(*k);
	    u2[*k + u2_dim1] = z__[jprev];
	    dsigma[*k] = d__[jprev];
	    idxp[*k] = jprev;
	    jprev = j;
	}
    }
    goto L100;
L110:

/*     Record the last singular value. */

    ++(*k);
    u2[*k + u2_dim1] = z__[jprev];
    dsigma[*k] = d__[jprev];
    idxp[*k] = jprev;

L120:

/*     Count up the total number of the various types of columns, then */
/*     form a permutation which positions the four column types into */
/*     four groups of uniform structure (although one or more of these */
/*     groups may be empty). */

    for (j = 1; j <= 4; ++j) {
	ctot[j - 1] = 0;
/* L130: */
    }
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	ct = coltyp[j];
	++ctot[ct - 1];
/* L140: */
    }

/*     PSM(*) = Position in SubMatrix (of types 1 through 4) */

    psm[0] = 2;
    psm[1] = ctot[0] + 2;
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];

/*     Fill out the IDXC array so that the permutation which it induces */
/*     will place all type-1 columns first, all type-2 columns next, */
/*     then all type-3's, and finally all type-4's, starting from the */
/*     second column. This applies similarly to the rows of VT. */

    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	jp = idxp[j];
	ct = coltyp[jp];
	idxc[psm[ct - 1]] = j;
	++psm[ct - 1];
/* L150: */
    }

/*     Sort the singular values and corresponding singular vectors into */
/*     DSIGMA, U2, and VT2 respectively.  The singular values/vectors */
/*     which were not deflated go into the first K slots of DSIGMA, U2, */
/*     and VT2 respectively, while those which were deflated go into the */
/*     last N - K slots, except that the first column/row will be treated */
/*     separately. */

    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	jp = idxp[j];
	dsigma[j] = d__[jp];
	idxj = idxq[idx[idxp[idxc[j]]] + 1];
	if (idxj <= nlp1) {
	    --idxj;
	}
	dcopy_(&n, &u[idxj * u_dim1 + 1], &c__1, &u2[j * u2_dim1 + 1], &c__1);
	dcopy_(&m, &vt[idxj + vt_dim1], ldvt, &vt2[j + vt2_dim1], ldvt2);
/* L160: */
    }

/*     Determine DSIGMA(1), DSIGMA(2) and Z(1) */

    dsigma[1] = 0.;
    hlftol = tol / 2.;
    if (abs(dsigma[2]) <= hlftol) {
	dsigma[2] = hlftol;
    }
    if (m > n) {
	z__[1] = dlapy2_(&z1, &z__[m]);
	if (z__[1] <= tol) {
	    c__ = 1.;
	    s = 0.;
	    z__[1] = tol;
	} else {
	    c__ = z1 / z__[1];
	    s = z__[m] / z__[1];
	}
    } else {
	if (abs(z1) <= tol) {
	    z__[1] = tol;
	} else {
	    z__[1] = z1;
	}
    }

/*     Move the rest of the updating row to Z. */

    i__1 = *k - 1;
    dcopy_(&i__1, &u2[u2_dim1 + 2], &c__1, &z__[2], &c__1);

/*     Determine the first column of U2, the first row of VT2 and the */
/*     last row of VT. */

    dlaset_("A", &n, &c__1, &c_b30, &c_b30, &u2[u2_offset], ldu2);
    u2[nlp1 + u2_dim1] = 1.;
    if (m > n) {
	i__1 = nlp1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    vt[m + i__ * vt_dim1] = -s * vt[nlp1 + i__ * vt_dim1];
	    vt2[i__ * vt2_dim1 + 1] = c__ * vt[nlp1 + i__ * vt_dim1];
/* L170: */
	}
	i__1 = m;
	for (i__ = nlp2; i__ <= i__1; ++i__) {
	    vt2[i__ * vt2_dim1 + 1] = s * vt[m + i__ * vt_dim1];
	    vt[m + i__ * vt_dim1] = c__ * vt[m + i__ * vt_dim1];
/* L180: */
	}
    } else {
	dcopy_(&m, &vt[nlp1 + vt_dim1], ldvt, &vt2[vt2_dim1 + 1], ldvt2);
    }
    if (m > n) {
	dcopy_(&m, &vt[m + vt_dim1], ldvt, &vt2[m + vt2_dim1], ldvt2);
    }

/*     The deflated singular values and their corresponding vectors go */
/*     into the back of D, U, and V respectively. */

    if (n > *k) {
	i__1 = n - *k;
	dcopy_(&i__1, &dsigma[*k + 1], &c__1, &d__[*k + 1], &c__1);
	i__1 = n - *k;
	dlacpy_("A", &n, &i__1, &u2[(*k + 1) * u2_dim1 + 1], ldu2, &u[(*k + 1)
		 * u_dim1 + 1], ldu);
	i__1 = n - *k;
	dlacpy_("A", &i__1, &m, &vt2[*k + 1 + vt2_dim1], ldvt2, &vt[*k + 1 + 
		vt_dim1], ldvt);
    }

/*     Copy CTOT into COLTYP for referencing in DLASD3. */

    for (j = 1; j <= 4; ++j) {
	coltyp[j] = ctot[j - 1];
/* L190: */
    }

    return 0;

/*     End of DLASD2 */

} /* dlasd2_ */
