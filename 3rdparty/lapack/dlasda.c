#include "clapack.h"

/* Table of constant values */

static integer c__0 = 0;
static doublereal c_b11 = 0.;
static doublereal c_b12 = 1.;
static integer c__1 = 1;
static integer c__2 = 2;

/* Subroutine */ int dlasda_(integer *icompq, integer *smlsiz, integer *n, 
	integer *sqre, doublereal *d__, doublereal *e, doublereal *u, integer 
	*ldu, doublereal *vt, integer *k, doublereal *difl, doublereal *difr, 
	doublereal *z__, doublereal *poles, integer *givptr, integer *givcol, 
	integer *ldgcol, integer *perm, doublereal *givnum, doublereal *c__, 
	doublereal *s, doublereal *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer givcol_dim1, givcol_offset, perm_dim1, perm_offset, difl_dim1, 
	    difl_offset, difr_dim1, difr_offset, givnum_dim1, givnum_offset, 
	    poles_dim1, poles_offset, u_dim1, u_offset, vt_dim1, vt_offset, 
	    z_dim1, z_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    integer i__, j, m, i1, ic, lf, nd, ll, nl, vf, nr, vl, im1, ncc, nlf, nrf,
	     vfi, iwk, vli, lvl, nru, ndb1, nlp1, lvl2, nrp1;
    doublereal beta;
    integer idxq, nlvl;
    doublereal alpha;
    integer inode, ndiml, ndimr, idxqi, itemp;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *);
    integer sqrei;
    extern /* Subroutine */ int dlasd6_(integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, integer *, integer *, integer *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *, doublereal *, doublereal *, 
	     doublereal *, integer *, integer *);
    integer nwork1, nwork2;
    extern /* Subroutine */ int dlasdq_(char *, integer *, integer *, integer 
	    *, integer *, integer *, doublereal *, doublereal *, doublereal *, 
	     integer *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, integer *), dlasdt_(integer *, integer *, 
	    integer *, integer *, integer *, integer *, integer *), dlaset_(
	    char *, integer *, integer *, doublereal *, doublereal *, 
	    doublereal *, integer *), xerbla_(char *, integer *);
    integer smlszp;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Using a divide and conquer approach, DLASDA computes the singular */
/*  value decomposition (SVD) of a real upper bidiagonal N-by-M matrix */
/*  B with diagonal D and offdiagonal E, where M = N + SQRE. The */
/*  algorithm computes the singular values in the SVD B = U * S * VT. */
/*  The orthogonal matrices U and VT are optionally computed in */
/*  compact form. */

/*  A related subroutine, DLASD0, computes the singular values and */
/*  the singular vectors in explicit form. */

/*  Arguments */
/*  ========= */

/*  ICOMPQ (input) INTEGER */
/*         Specifies whether singular vectors are to be computed */
/*         in compact form, as follows */
/*         = 0: Compute singular values only. */
/*         = 1: Compute singular vectors of upper bidiagonal */
/*              matrix in compact form. */

/*  SMLSIZ (input) INTEGER */
/*         The maximum size of the subproblems at the bottom of the */
/*         computation tree. */

/*  N      (input) INTEGER */
/*         The row dimension of the upper bidiagonal matrix. This is */
/*         also the dimension of the main diagonal array D. */

/*  SQRE   (input) INTEGER */
/*         Specifies the column dimension of the bidiagonal matrix. */
/*         = 0: The bidiagonal matrix has column dimension M = N; */
/*         = 1: The bidiagonal matrix has column dimension M = N + 1. */

/*  D      (input/output) DOUBLE PRECISION array, dimension ( N ) */
/*         On entry D contains the main diagonal of the bidiagonal */
/*         matrix. On exit D, if INFO = 0, contains its singular values. */

/*  E      (input) DOUBLE PRECISION array, dimension ( M-1 ) */
/*         Contains the subdiagonal entries of the bidiagonal matrix. */
/*         On exit, E has been destroyed. */

/*  U      (output) DOUBLE PRECISION array, */
/*         dimension ( LDU, SMLSIZ ) if ICOMPQ = 1, and not referenced */
/*         if ICOMPQ = 0. If ICOMPQ = 1, on exit, U contains the left */
/*         singular vector matrices of all subproblems at the bottom */
/*         level. */

/*  LDU    (input) INTEGER, LDU = > N. */
/*         The leading dimension of arrays U, VT, DIFL, DIFR, POLES, */
/*         GIVNUM, and Z. */

/*  VT     (output) DOUBLE PRECISION array, */
/*         dimension ( LDU, SMLSIZ+1 ) if ICOMPQ = 1, and not referenced */
/*         if ICOMPQ = 0. If ICOMPQ = 1, on exit, VT' contains the right */
/*         singular vector matrices of all subproblems at the bottom */
/*         level. */

/*  K      (output) INTEGER array, */
/*         dimension ( N ) if ICOMPQ = 1 and dimension 1 if ICOMPQ = 0. */
/*         If ICOMPQ = 1, on exit, K(I) is the dimension of the I-th */
/*         secular equation on the computation tree. */

/*  DIFL   (output) DOUBLE PRECISION array, dimension ( LDU, NLVL ), */
/*         where NLVL = floor(log_2 (N/SMLSIZ))). */

/*  DIFR   (output) DOUBLE PRECISION array, */
/*                  dimension ( LDU, 2 * NLVL ) if ICOMPQ = 1 and */
/*                  dimension ( N ) if ICOMPQ = 0. */
/*         If ICOMPQ = 1, on exit, DIFL(1:N, I) and DIFR(1:N, 2 * I - 1) */
/*         record distances between singular values on the I-th */
/*         level and singular values on the (I -1)-th level, and */
/*         DIFR(1:N, 2 * I ) contains the normalizing factors for */
/*         the right singular vector matrix. See DLASD8 for details. */

/*  Z      (output) DOUBLE PRECISION array, */
/*                  dimension ( LDU, NLVL ) if ICOMPQ = 1 and */
/*                  dimension ( N ) if ICOMPQ = 0. */
/*         The first K elements of Z(1, I) contain the components of */
/*         the deflation-adjusted updating row vector for subproblems */
/*         on the I-th level. */

/*  POLES  (output) DOUBLE PRECISION array, */
/*         dimension ( LDU, 2 * NLVL ) if ICOMPQ = 1, and not referenced */
/*         if ICOMPQ = 0. If ICOMPQ = 1, on exit, POLES(1, 2*I - 1) and */
/*         POLES(1, 2*I) contain  the new and old singular values */
/*         involved in the secular equations on the I-th level. */

/*  GIVPTR (output) INTEGER array, */
/*         dimension ( N ) if ICOMPQ = 1, and not referenced if */
/*         ICOMPQ = 0. If ICOMPQ = 1, on exit, GIVPTR( I ) records */
/*         the number of Givens rotations performed on the I-th */
/*         problem on the computation tree. */

/*  GIVCOL (output) INTEGER array, */
/*         dimension ( LDGCOL, 2 * NLVL ) if ICOMPQ = 1, and not */
/*         referenced if ICOMPQ = 0. If ICOMPQ = 1, on exit, for each I, */
/*         GIVCOL(1, 2 *I - 1) and GIVCOL(1, 2 *I) record the locations */
/*         of Givens rotations performed on the I-th level on the */
/*         computation tree. */

/*  LDGCOL (input) INTEGER, LDGCOL = > N. */
/*         The leading dimension of arrays GIVCOL and PERM. */

/*  PERM   (output) INTEGER array, */
/*         dimension ( LDGCOL, NLVL ) if ICOMPQ = 1, and not referenced */
/*         if ICOMPQ = 0. If ICOMPQ = 1, on exit, PERM(1, I) records */
/*         permutations done on the I-th level of the computation tree. */

/*  GIVNUM (output) DOUBLE PRECISION array, */
/*         dimension ( LDU,  2 * NLVL ) if ICOMPQ = 1, and not */
/*         referenced if ICOMPQ = 0. If ICOMPQ = 1, on exit, for each I, */
/*         GIVNUM(1, 2 *I - 1) and GIVNUM(1, 2 *I) record the C- and S- */
/*         values of Givens rotations performed on the I-th level on */
/*         the computation tree. */

/*  C      (output) DOUBLE PRECISION array, */
/*         dimension ( N ) if ICOMPQ = 1, and dimension 1 if ICOMPQ = 0. */
/*         If ICOMPQ = 1 and the I-th subproblem is not square, on exit, */
/*         C( I ) contains the C-value of a Givens rotation related to */
/*         the right null space of the I-th subproblem. */

/*  S      (output) DOUBLE PRECISION array, dimension ( N ) if */
/*         ICOMPQ = 1, and dimension 1 if ICOMPQ = 0. If ICOMPQ = 1 */
/*         and the I-th subproblem is not square, on exit, S( I ) */
/*         contains the S-value of a Givens rotation related to */
/*         the right null space of the I-th subproblem. */

/*  WORK   (workspace) DOUBLE PRECISION array, dimension */
/*         (6 * N + (SMLSIZ + 1)*(SMLSIZ + 1)). */

/*  IWORK  (workspace) INTEGER array. */
/*         Dimension must be at least (7 * N). */

/*  INFO   (output) INTEGER */
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
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --d__;
    --e;
    givnum_dim1 = *ldu;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;
    poles_dim1 = *ldu;
    poles_offset = 1 + poles_dim1;
    poles -= poles_offset;
    z_dim1 = *ldu;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    difr_dim1 = *ldu;
    difr_offset = 1 + difr_dim1;
    difr -= difr_offset;
    difl_dim1 = *ldu;
    difl_offset = 1 + difl_dim1;
    difl -= difl_offset;
    vt_dim1 = *ldu;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    --k;
    --givptr;
    perm_dim1 = *ldgcol;
    perm_offset = 1 + perm_dim1;
    perm -= perm_offset;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    --c__;
    --s;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*smlsiz < 3) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -4;
    } else if (*ldu < *n + *sqre) {
	*info = -8;
    } else if (*ldgcol < *n) {
	*info = -17;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASDA", &i__1);
	return 0;
    }

    m = *n + *sqre;

/*     If the input matrix is too small, call DLASDQ to find the SVD. */

    if (*n <= *smlsiz) {
	if (*icompq == 0) {
	    dlasdq_("U", sqre, n, &c__0, &c__0, &c__0, &d__[1], &e[1], &vt[
		    vt_offset], ldu, &u[u_offset], ldu, &u[u_offset], ldu, &
		    work[1], info);
	} else {
	    dlasdq_("U", sqre, n, &m, n, &c__0, &d__[1], &e[1], &vt[vt_offset]
, ldu, &u[u_offset], ldu, &u[u_offset], ldu, &work[1], 
		    info);
	}
	return 0;
    }

/*     Book-keeping and  set up the computation tree. */

    inode = 1;
    ndiml = inode + *n;
    ndimr = ndiml + *n;
    idxq = ndimr + *n;
    iwk = idxq + *n;

    ncc = 0;
    nru = 0;

    smlszp = *smlsiz + 1;
    vf = 1;
    vl = vf + m;
    nwork1 = vl + m;
    nwork2 = nwork1 + smlszp * smlszp;

    dlasdt_(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr], 
	    smlsiz);

/*     for the nodes on bottom level of the tree, solve */
/*     their subproblems by DLASDQ. */

    ndb1 = (nd + 1) / 2;
    i__1 = nd;
    for (i__ = ndb1; i__ <= i__1; ++i__) {

/*        IC : center row of each node */
/*        NL : number of rows of left  subproblem */
/*        NR : number of rows of right subproblem */
/*        NLF: starting row of the left   subproblem */
/*        NRF: starting row of the right  subproblem */

	i1 = i__ - 1;
	ic = iwork[inode + i1];
	nl = iwork[ndiml + i1];
	nlp1 = nl + 1;
	nr = iwork[ndimr + i1];
	nlf = ic - nl;
	nrf = ic + 1;
	idxqi = idxq + nlf - 2;
	vfi = vf + nlf - 1;
	vli = vl + nlf - 1;
	sqrei = 1;
	if (*icompq == 0) {
	    dlaset_("A", &nlp1, &nlp1, &c_b11, &c_b12, &work[nwork1], &smlszp);
	    dlasdq_("U", &sqrei, &nl, &nlp1, &nru, &ncc, &d__[nlf], &e[nlf], &
		    work[nwork1], &smlszp, &work[nwork2], &nl, &work[nwork2], 
		    &nl, &work[nwork2], info);
	    itemp = nwork1 + nl * smlszp;
	    dcopy_(&nlp1, &work[nwork1], &c__1, &work[vfi], &c__1);
	    dcopy_(&nlp1, &work[itemp], &c__1, &work[vli], &c__1);
	} else {
	    dlaset_("A", &nl, &nl, &c_b11, &c_b12, &u[nlf + u_dim1], ldu);
	    dlaset_("A", &nlp1, &nlp1, &c_b11, &c_b12, &vt[nlf + vt_dim1], 
		    ldu);
	    dlasdq_("U", &sqrei, &nl, &nlp1, &nl, &ncc, &d__[nlf], &e[nlf], &
		    vt[nlf + vt_dim1], ldu, &u[nlf + u_dim1], ldu, &u[nlf + 
		    u_dim1], ldu, &work[nwork1], info);
	    dcopy_(&nlp1, &vt[nlf + vt_dim1], &c__1, &work[vfi], &c__1);
	    dcopy_(&nlp1, &vt[nlf + nlp1 * vt_dim1], &c__1, &work[vli], &c__1)
		    ;
	}
	if (*info != 0) {
	    return 0;
	}
	i__2 = nl;
	for (j = 1; j <= i__2; ++j) {
	    iwork[idxqi + j] = j;
/* L10: */
	}
	if (i__ == nd && *sqre == 0) {
	    sqrei = 0;
	} else {
	    sqrei = 1;
	}
	idxqi += nlp1;
	vfi += nlp1;
	vli += nlp1;
	nrp1 = nr + sqrei;
	if (*icompq == 0) {
	    dlaset_("A", &nrp1, &nrp1, &c_b11, &c_b12, &work[nwork1], &smlszp);
	    dlasdq_("U", &sqrei, &nr, &nrp1, &nru, &ncc, &d__[nrf], &e[nrf], &
		    work[nwork1], &smlszp, &work[nwork2], &nr, &work[nwork2], 
		    &nr, &work[nwork2], info);
	    itemp = nwork1 + (nrp1 - 1) * smlszp;
	    dcopy_(&nrp1, &work[nwork1], &c__1, &work[vfi], &c__1);
	    dcopy_(&nrp1, &work[itemp], &c__1, &work[vli], &c__1);
	} else {
	    dlaset_("A", &nr, &nr, &c_b11, &c_b12, &u[nrf + u_dim1], ldu);
	    dlaset_("A", &nrp1, &nrp1, &c_b11, &c_b12, &vt[nrf + vt_dim1], 
		    ldu);
	    dlasdq_("U", &sqrei, &nr, &nrp1, &nr, &ncc, &d__[nrf], &e[nrf], &
		    vt[nrf + vt_dim1], ldu, &u[nrf + u_dim1], ldu, &u[nrf + 
		    u_dim1], ldu, &work[nwork1], info);
	    dcopy_(&nrp1, &vt[nrf + vt_dim1], &c__1, &work[vfi], &c__1);
	    dcopy_(&nrp1, &vt[nrf + nrp1 * vt_dim1], &c__1, &work[vli], &c__1)
		    ;
	}
	if (*info != 0) {
	    return 0;
	}
	i__2 = nr;
	for (j = 1; j <= i__2; ++j) {
	    iwork[idxqi + j] = j;
/* L20: */
	}
/* L30: */
    }

/*     Now conquer each subproblem bottom-up. */

    j = pow_ii(&c__2, &nlvl);
    for (lvl = nlvl; lvl >= 1; --lvl) {
	lvl2 = (lvl << 1) - 1;

/*        Find the first node LF and last node LL on */
/*        the current level LVL. */

	if (lvl == 1) {
	    lf = 1;
	    ll = 1;
	} else {
	    i__1 = lvl - 1;
	    lf = pow_ii(&c__2, &i__1);
	    ll = (lf << 1) - 1;
	}
	i__1 = ll;
	for (i__ = lf; i__ <= i__1; ++i__) {
	    im1 = i__ - 1;
	    ic = iwork[inode + im1];
	    nl = iwork[ndiml + im1];
	    nr = iwork[ndimr + im1];
	    nlf = ic - nl;
	    nrf = ic + 1;
	    if (i__ == ll) {
		sqrei = *sqre;
	    } else {
		sqrei = 1;
	    }
	    vfi = vf + nlf - 1;
	    vli = vl + nlf - 1;
	    idxqi = idxq + nlf - 1;
	    alpha = d__[ic];
	    beta = e[ic];
	    if (*icompq == 0) {
		dlasd6_(icompq, &nl, &nr, &sqrei, &d__[nlf], &work[vfi], &
			work[vli], &alpha, &beta, &iwork[idxqi], &perm[
			perm_offset], &givptr[1], &givcol[givcol_offset], 
			ldgcol, &givnum[givnum_offset], ldu, &poles[
			poles_offset], &difl[difl_offset], &difr[difr_offset], 
			 &z__[z_offset], &k[1], &c__[1], &s[1], &work[nwork1], 
			 &iwork[iwk], info);
	    } else {
		--j;
		dlasd6_(icompq, &nl, &nr, &sqrei, &d__[nlf], &work[vfi], &
			work[vli], &alpha, &beta, &iwork[idxqi], &perm[nlf + 
			lvl * perm_dim1], &givptr[j], &givcol[nlf + lvl2 * 
			givcol_dim1], ldgcol, &givnum[nlf + lvl2 * 
			givnum_dim1], ldu, &poles[nlf + lvl2 * poles_dim1], &
			difl[nlf + lvl * difl_dim1], &difr[nlf + lvl2 * 
			difr_dim1], &z__[nlf + lvl * z_dim1], &k[j], &c__[j], 
			&s[j], &work[nwork1], &iwork[iwk], info);
	    }
	    if (*info != 0) {
		return 0;
	    }
/* L40: */
	}
/* L50: */
    }

    return 0;

/*     End of DLASDA */

} /* dlasda_ */
