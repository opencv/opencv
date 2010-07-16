/* dlaed2.f -- translated by f2c (version 20061008).
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

static doublereal c_b3 = -1.;
static integer c__1 = 1;

/* Subroutine */ int dlaed2_(integer *k, integer *n, integer *n1, doublereal *
	d__, doublereal *q, integer *ldq, integer *indxq, doublereal *rho, 
	doublereal *z__, doublereal *dlamda, doublereal *w, doublereal *q2, 
	integer *indx, integer *indxc, integer *indxp, integer *coltyp, 
	integer *info)
{
    /* System generated locals */
    integer q_dim1, q_offset, i__1, i__2;
    doublereal d__1, d__2, d__3, d__4;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    doublereal c__;
    integer i__, j;
    doublereal s, t;
    integer k2, n2, ct, nj, pj, js, iq1, iq2, n1p1;
    doublereal eps, tau, tol;
    integer psm[4], imax, jmax;
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    integer ctot[4];
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *), dcopy_(integer *, doublereal *, integer *, doublereal 
	    *, integer *);
    extern doublereal dlapy2_(doublereal *, doublereal *), dlamch_(char *);
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dlamrg_(integer *, integer *, doublereal *, 
	    integer *, integer *, integer *), dlacpy_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, integer *), xerbla_(char *, integer *);


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLAED2 merges the two sets of eigenvalues together into a single */
/*  sorted set.  Then it tries to deflate the size of the problem. */
/*  There are two ways in which deflation can occur:  when two or more */
/*  eigenvalues are close together or if there is a tiny entry in the */
/*  Z vector.  For each such occurrence the order of the related secular */
/*  equation problem is reduced by one. */

/*  Arguments */
/*  ========= */

/*  K      (output) INTEGER */
/*         The number of non-deflated eigenvalues, and the order of the */
/*         related secular equation. 0 <= K <=N. */

/*  N      (input) INTEGER */
/*         The dimension of the symmetric tridiagonal matrix.  N >= 0. */

/*  N1     (input) INTEGER */
/*         The location of the last eigenvalue in the leading sub-matrix. */
/*         min(1,N) <= N1 <= N/2. */

/*  D      (input/output) DOUBLE PRECISION array, dimension (N) */
/*         On entry, D contains the eigenvalues of the two submatrices to */
/*         be combined. */
/*         On exit, D contains the trailing (N-K) updated eigenvalues */
/*         (those which were deflated) sorted into increasing order. */

/*  Q      (input/output) DOUBLE PRECISION array, dimension (LDQ, N) */
/*         On entry, Q contains the eigenvectors of two submatrices in */
/*         the two square blocks with corners at (1,1), (N1,N1) */
/*         and (N1+1, N1+1), (N,N). */
/*         On exit, Q contains the trailing (N-K) updated eigenvectors */
/*         (those which were deflated) in its last N-K columns. */

/*  LDQ    (input) INTEGER */
/*         The leading dimension of the array Q.  LDQ >= max(1,N). */

/*  INDXQ  (input/output) INTEGER array, dimension (N) */
/*         The permutation which separately sorts the two sub-problems */
/*         in D into ascending order.  Note that elements in the second */
/*         half of this permutation must first have N1 added to their */
/*         values. Destroyed on exit. */

/*  RHO    (input/output) DOUBLE PRECISION */
/*         On entry, the off-diagonal element associated with the rank-1 */
/*         cut which originally split the two submatrices which are now */
/*         being recombined. */
/*         On exit, RHO has been modified to the value required by */
/*         DLAED3. */

/*  Z      (input) DOUBLE PRECISION array, dimension (N) */
/*         On entry, Z contains the updating vector (the last */
/*         row of the first sub-eigenvector matrix and the first row of */
/*         the second sub-eigenvector matrix). */
/*         On exit, the contents of Z have been destroyed by the updating */
/*         process. */

/*  DLAMDA (output) DOUBLE PRECISION array, dimension (N) */
/*         A copy of the first K eigenvalues which will be used by */
/*         DLAED3 to form the secular equation. */

/*  W      (output) DOUBLE PRECISION array, dimension (N) */
/*         The first k values of the final deflation-altered z-vector */
/*         which will be passed to DLAED3. */

/*  Q2     (output) DOUBLE PRECISION array, dimension (N1**2+(N-N1)**2) */
/*         A copy of the first K eigenvectors which will be used by */
/*         DLAED3 in a matrix multiply (DGEMM) to solve for the new */
/*         eigenvectors. */

/*  INDX   (workspace) INTEGER array, dimension (N) */
/*         The permutation used to sort the contents of DLAMDA into */
/*         ascending order. */

/*  INDXC  (output) INTEGER array, dimension (N) */
/*         The permutation used to arrange the columns of the deflated */
/*         Q matrix into three groups:  the first group contains non-zero */
/*         elements only at and above N1, the second contains */
/*         non-zero elements only below N1, and the third is dense. */

/*  INDXP  (workspace) INTEGER array, dimension (N) */
/*         The permutation used to place deflated values of D at the end */
/*         of the array.  INDXP(1:K) points to the nondeflated D-values */
/*         and INDXP(K+1:N) points to the deflated eigenvalues. */

/*  COLTYP (workspace/output) INTEGER array, dimension (N) */
/*         During execution, a label which will indicate which of the */
/*         following types a column in the Q2 matrix is: */
/*         1 : non-zero in the upper half only; */
/*         2 : dense; */
/*         3 : non-zero in the lower half only; */
/*         4 : deflated. */
/*         On exit, COLTYP(i) is the number of columns of type i, */
/*         for i=1 to 4 only. */

/*  INFO   (output) INTEGER */
/*          = 0:  successful exit. */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Jeff Rutter, Computer Science Division, University of California */
/*     at Berkeley, USA */
/*  Modified by Francoise Tisseur, University of Tennessee. */

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
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --indxq;
    --z__;
    --dlamda;
    --w;
    --q2;
    --indx;
    --indxc;
    --indxp;
    --coltyp;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -2;
    } else if (*ldq < max(1,*n)) {
	*info = -6;
    } else /* if(complicated condition) */ {
/* Computing MIN */
	i__1 = 1, i__2 = *n / 2;
	if (min(i__1,i__2) > *n1 || *n / 2 < *n1) {
	    *info = -3;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLAED2", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

    n2 = *n - *n1;
    n1p1 = *n1 + 1;

    if (*rho < 0.) {
	dscal_(&n2, &c_b3, &z__[n1p1], &c__1);
    }

/*     Normalize z so that norm(z) = 1.  Since z is the concatenation of */
/*     two normalized vectors, norm2(z) = sqrt(2). */

    t = 1. / sqrt(2.);
    dscal_(n, &t, &z__[1], &c__1);

/*     RHO = ABS( norm(z)**2 * RHO ) */

    *rho = (d__1 = *rho * 2., abs(d__1));

/*     Sort the eigenvalues into increasing order */

    i__1 = *n;
    for (i__ = n1p1; i__ <= i__1; ++i__) {
	indxq[i__] += *n1;
/* L10: */
    }

/*     re-integrate the deflated parts from the last pass */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dlamda[i__] = d__[indxq[i__]];
/* L20: */
    }
    dlamrg_(n1, &n2, &dlamda[1], &c__1, &c__1, &indxc[1]);
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	indx[i__] = indxq[indxc[i__]];
/* L30: */
    }

/*     Calculate the allowable deflation tolerance */

    imax = idamax_(n, &z__[1], &c__1);
    jmax = idamax_(n, &d__[1], &c__1);
    eps = dlamch_("Epsilon");
/* Computing MAX */
    d__3 = (d__1 = d__[jmax], abs(d__1)), d__4 = (d__2 = z__[imax], abs(d__2))
	    ;
    tol = eps * 8. * max(d__3,d__4);

/*     If the rank-1 modifier is small enough, no more needs to be done */
/*     except to reorganize Q so that its columns correspond with the */
/*     elements in D. */

    if (*rho * (d__1 = z__[imax], abs(d__1)) <= tol) {
	*k = 0;
	iq2 = 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__ = indx[j];
	    dcopy_(n, &q[i__ * q_dim1 + 1], &c__1, &q2[iq2], &c__1);
	    dlamda[j] = d__[i__];
	    iq2 += *n;
/* L40: */
	}
	dlacpy_("A", n, n, &q2[1], n, &q[q_offset], ldq);
	dcopy_(n, &dlamda[1], &c__1, &d__[1], &c__1);
	goto L190;
    }

/*     If there are multiple eigenvalues then the problem deflates.  Here */
/*     the number of equal eigenvalues are found.  As each equal */
/*     eigenvalue is found, an elementary reflector is computed to rotate */
/*     the corresponding eigensubspace so that the corresponding */
/*     components of Z are zero in this new basis. */

    i__1 = *n1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	coltyp[i__] = 1;
/* L50: */
    }
    i__1 = *n;
    for (i__ = n1p1; i__ <= i__1; ++i__) {
	coltyp[i__] = 3;
/* L60: */
    }


    *k = 0;
    k2 = *n + 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	nj = indx[j];
	if (*rho * (d__1 = z__[nj], abs(d__1)) <= tol) {

/*           Deflate due to small z component. */

	    --k2;
	    coltyp[nj] = 4;
	    indxp[k2] = nj;
	    if (j == *n) {
		goto L100;
	    }
	} else {
	    pj = nj;
	    goto L80;
	}
/* L70: */
    }
L80:
    ++j;
    nj = indx[j];
    if (j > *n) {
	goto L100;
    }
    if (*rho * (d__1 = z__[nj], abs(d__1)) <= tol) {

/*        Deflate due to small z component. */

	--k2;
	coltyp[nj] = 4;
	indxp[k2] = nj;
    } else {

/*        Check if eigenvalues are close enough to allow deflation. */

	s = z__[pj];
	c__ = z__[nj];

/*        Find sqrt(a**2+b**2) without overflow or */
/*        destructive underflow. */

	tau = dlapy2_(&c__, &s);
	t = d__[nj] - d__[pj];
	c__ /= tau;
	s = -s / tau;
	if ((d__1 = t * c__ * s, abs(d__1)) <= tol) {

/*           Deflation is possible. */

	    z__[nj] = tau;
	    z__[pj] = 0.;
	    if (coltyp[nj] != coltyp[pj]) {
		coltyp[nj] = 2;
	    }
	    coltyp[pj] = 4;
	    drot_(n, &q[pj * q_dim1 + 1], &c__1, &q[nj * q_dim1 + 1], &c__1, &
		    c__, &s);
/* Computing 2nd power */
	    d__1 = c__;
/* Computing 2nd power */
	    d__2 = s;
	    t = d__[pj] * (d__1 * d__1) + d__[nj] * (d__2 * d__2);
/* Computing 2nd power */
	    d__1 = s;
/* Computing 2nd power */
	    d__2 = c__;
	    d__[nj] = d__[pj] * (d__1 * d__1) + d__[nj] * (d__2 * d__2);
	    d__[pj] = t;
	    --k2;
	    i__ = 1;
L90:
	    if (k2 + i__ <= *n) {
		if (d__[pj] < d__[indxp[k2 + i__]]) {
		    indxp[k2 + i__ - 1] = indxp[k2 + i__];
		    indxp[k2 + i__] = pj;
		    ++i__;
		    goto L90;
		} else {
		    indxp[k2 + i__ - 1] = pj;
		}
	    } else {
		indxp[k2 + i__ - 1] = pj;
	    }
	    pj = nj;
	} else {
	    ++(*k);
	    dlamda[*k] = d__[pj];
	    w[*k] = z__[pj];
	    indxp[*k] = pj;
	    pj = nj;
	}
    }
    goto L80;
L100:

/*     Record the last eigenvalue. */

    ++(*k);
    dlamda[*k] = d__[pj];
    w[*k] = z__[pj];
    indxp[*k] = pj;

/*     Count up the total number of the various types of columns, then */
/*     form a permutation which positions the four column types into */
/*     four uniform groups (although one or more of these groups may be */
/*     empty). */

    for (j = 1; j <= 4; ++j) {
	ctot[j - 1] = 0;
/* L110: */
    }
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	ct = coltyp[j];
	++ctot[ct - 1];
/* L120: */
    }

/*     PSM(*) = Position in SubMatrix (of types 1 through 4) */

    psm[0] = 1;
    psm[1] = ctot[0] + 1;
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    *k = *n - ctot[3];

/*     Fill out the INDXC array so that the permutation which it induces */
/*     will place all type-1 columns first, all type-2 columns next, */
/*     then all type-3's, and finally all type-4's. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	js = indxp[j];
	ct = coltyp[js];
	indx[psm[ct - 1]] = js;
	indxc[psm[ct - 1]] = j;
	++psm[ct - 1];
/* L130: */
    }

/*     Sort the eigenvalues and corresponding eigenvectors into DLAMDA */
/*     and Q2 respectively.  The eigenvalues/vectors which were not */
/*     deflated go into the first K slots of DLAMDA and Q2 respectively, */
/*     while those which were deflated go into the last N - K slots. */

    i__ = 1;
    iq1 = 1;
    iq2 = (ctot[0] + ctot[1]) * *n1 + 1;
    i__1 = ctot[0];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	dcopy_(n1, &q[js * q_dim1 + 1], &c__1, &q2[iq1], &c__1);
	z__[i__] = d__[js];
	++i__;
	iq1 += *n1;
/* L140: */
    }

    i__1 = ctot[1];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	dcopy_(n1, &q[js * q_dim1 + 1], &c__1, &q2[iq1], &c__1);
	dcopy_(&n2, &q[*n1 + 1 + js * q_dim1], &c__1, &q2[iq2], &c__1);
	z__[i__] = d__[js];
	++i__;
	iq1 += *n1;
	iq2 += n2;
/* L150: */
    }

    i__1 = ctot[2];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	dcopy_(&n2, &q[*n1 + 1 + js * q_dim1], &c__1, &q2[iq2], &c__1);
	z__[i__] = d__[js];
	++i__;
	iq2 += n2;
/* L160: */
    }

    iq1 = iq2;
    i__1 = ctot[3];
    for (j = 1; j <= i__1; ++j) {
	js = indx[i__];
	dcopy_(n, &q[js * q_dim1 + 1], &c__1, &q2[iq2], &c__1);
	iq2 += *n;
	z__[i__] = d__[js];
	++i__;
/* L170: */
    }

/*     The deflated eigenvalues and their corresponding vectors go back */
/*     into the last N - K slots of D and Q respectively. */

    dlacpy_("A", n, &ctot[3], &q2[iq1], n, &q[(*k + 1) * q_dim1 + 1], ldq);
    i__1 = *n - *k;
    dcopy_(&i__1, &z__[*k + 1], &c__1, &d__[*k + 1], &c__1);

/*     Copy CTOT into COLTYP for referencing in DLAED3. */

    for (j = 1; j <= 4; ++j) {
	coltyp[j] = ctot[j - 1];
/* L180: */
    }

L190:
    return 0;

/*     End of DLAED2 */

} /* dlaed2_ */
