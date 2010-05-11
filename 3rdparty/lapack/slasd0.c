#include "clapack.h"

/* Table of constant values */

static integer c__0 = 0;
static integer c__2 = 2;

/* Subroutine */ int slasd0_(integer *n, integer *sqre, real *d__, real *e, 
	real *u, integer *ldu, real *vt, integer *ldvt, integer *smlsiz, 
	integer *iwork, real *work, integer *info)
{
    /* System generated locals */
    integer u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;

    /* Builtin functions */
    integer pow_ii(integer *, integer *);

    /* Local variables */
    integer i__, j, m, i1, ic, lf, nd, ll, nl, nr, im1, ncc, nlf, nrf, iwk, 
	    lvl, ndb1, nlp1, nrp1;
    real beta;
    integer idxq, nlvl;
    real alpha;
    integer inode, ndiml, idxqc, ndimr, itemp, sqrei;
    extern /* Subroutine */ int slasd1_(integer *, integer *, integer *, real 
	    *, real *, real *, real *, integer *, real *, integer *, integer *
, integer *, real *, integer *), xerbla_(char *, integer *), slasdq_(char *, integer *, integer *, integer *, integer 
	    *, integer *, real *, real *, real *, integer *, real *, integer *
, real *, integer *, real *, integer *), slasdt_(integer *
, integer *, integer *, integer *, integer *, integer *, integer *
);


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Using a divide and conquer approach, SLASD0 computes the singular */
/*  value decomposition (SVD) of a real upper bidiagonal N-by-M */
/*  matrix B with diagonal D and offdiagonal E, where M = N + SQRE. */
/*  The algorithm computes orthogonal matrices U and VT such that */
/*  B = U * S * VT. The singular values S are overwritten on D. */

/*  A related subroutine, SLASDA, computes only the singular values, */
/*  and optionally, the singular vectors in compact form. */

/*  Arguments */
/*  ========= */

/*  N      (input) INTEGER */
/*         On entry, the row dimension of the upper bidiagonal matrix. */
/*         This is also the dimension of the main diagonal array D. */

/*  SQRE   (input) INTEGER */
/*         Specifies the column dimension of the bidiagonal matrix. */
/*         = 0: The bidiagonal matrix has column dimension M = N; */
/*         = 1: The bidiagonal matrix has column dimension M = N+1; */

/*  D      (input/output) REAL array, dimension (N) */
/*         On entry D contains the main diagonal of the bidiagonal */
/*         matrix. */
/*         On exit D, if INFO = 0, contains its singular values. */

/*  E      (input) REAL array, dimension (M-1) */
/*         Contains the subdiagonal entries of the bidiagonal matrix. */
/*         On exit, E has been destroyed. */

/*  U      (output) REAL array, dimension at least (LDQ, N) */
/*         On exit, U contains the left singular vectors. */

/*  LDU    (input) INTEGER */
/*         On entry, leading dimension of U. */

/*  VT     (output) REAL array, dimension at least (LDVT, M) */
/*         On exit, VT' contains the right singular vectors. */

/*  LDVT   (input) INTEGER */
/*         On entry, leading dimension of VT. */

/*  SMLSIZ (input) INTEGER */
/*         On entry, maximum size of the subproblems at the */
/*         bottom of the computation tree. */

/*  IWORK  (workspace) INTEGER array, dimension (8*N) */

/*  WORK   (workspace) REAL array, dimension (3*M**2+2*M) */

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

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
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
    --iwork;
    --work;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -1;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -2;
    }

    m = *n + *sqre;

    if (*ldu < *n) {
	*info = -6;
    } else if (*ldvt < m) {
	*info = -8;
    } else if (*smlsiz < 3) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASD0", &i__1);
	return 0;
    }

/*     If the input matrix is too small, call SLASDQ to find the SVD. */

    if (*n <= *smlsiz) {
	slasdq_("U", sqre, n, &m, n, &c__0, &d__[1], &e[1], &vt[vt_offset], 
		ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[1], info);
	return 0;
    }

/*     Set up the computation tree. */

    inode = 1;
    ndiml = inode + *n;
    ndimr = ndiml + *n;
    idxq = ndimr + *n;
    iwk = idxq + *n;
    slasdt_(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr], 
	    smlsiz);

/*     For the nodes on bottom level of the tree, solve */
/*     their subproblems by SLASDQ. */

    ndb1 = (nd + 1) / 2;
    ncc = 0;
    i__1 = nd;
    for (i__ = ndb1; i__ <= i__1; ++i__) {

/*     IC : center row of each node */
/*     NL : number of rows of left  subproblem */
/*     NR : number of rows of right subproblem */
/*     NLF: starting row of the left   subproblem */
/*     NRF: starting row of the right  subproblem */

	i1 = i__ - 1;
	ic = iwork[inode + i1];
	nl = iwork[ndiml + i1];
	nlp1 = nl + 1;
	nr = iwork[ndimr + i1];
	nrp1 = nr + 1;
	nlf = ic - nl;
	nrf = ic + 1;
	sqrei = 1;
	slasdq_("U", &sqrei, &nl, &nlp1, &nl, &ncc, &d__[nlf], &e[nlf], &vt[
		nlf + nlf * vt_dim1], ldvt, &u[nlf + nlf * u_dim1], ldu, &u[
		nlf + nlf * u_dim1], ldu, &work[1], info);
	if (*info != 0) {
	    return 0;
	}
	itemp = idxq + nlf - 2;
	i__2 = nl;
	for (j = 1; j <= i__2; ++j) {
	    iwork[itemp + j] = j;
/* L10: */
	}
	if (i__ == nd) {
	    sqrei = *sqre;
	} else {
	    sqrei = 1;
	}
	nrp1 = nr + sqrei;
	slasdq_("U", &sqrei, &nr, &nrp1, &nr, &ncc, &d__[nrf], &e[nrf], &vt[
		nrf + nrf * vt_dim1], ldvt, &u[nrf + nrf * u_dim1], ldu, &u[
		nrf + nrf * u_dim1], ldu, &work[1], info);
	if (*info != 0) {
	    return 0;
	}
	itemp = idxq + ic;
	i__2 = nr;
	for (j = 1; j <= i__2; ++j) {
	    iwork[itemp + j - 1] = j;
/* L20: */
	}
/* L30: */
    }

/*     Now conquer each subproblem bottom-up. */

    for (lvl = nlvl; lvl >= 1; --lvl) {

/*        Find the first node LF and last node LL on the */
/*        current level LVL. */

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
	    if (*sqre == 0 && i__ == ll) {
		sqrei = *sqre;
	    } else {
		sqrei = 1;
	    }
	    idxqc = idxq + nlf - 1;
	    alpha = d__[ic];
	    beta = e[ic];
	    slasd1_(&nl, &nr, &sqrei, &d__[nlf], &alpha, &beta, &u[nlf + nlf *
		     u_dim1], ldu, &vt[nlf + nlf * vt_dim1], ldvt, &iwork[
		    idxqc], &iwork[iwk], &work[1], info);
	    if (*info != 0) {
		return 0;
	    }
/* L40: */
	}
/* L50: */
    }

    return 0;

/*     End of SLASD0 */

} /* slasd0_ */
