#include "clapack.h"

/* Table of constant values */

static integer c__1 = 1;
static real c_b6 = 0.f;
static integer c__0 = 0;
static real c_b11 = 1.f;

/* Subroutine */ int slalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, real *d__, real *e, real *b, integer *ldb, real *rcond, 
	integer *rank, real *work, integer *iwork, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2;
    real r__1;

    /* Builtin functions */
    double log(doublereal), r_sign(real *, real *);

    /* Local variables */
    integer c__, i__, j, k;
    real r__;
    integer s, u, z__;
    real cs;
    integer bx;
    real sn;
    integer st, vt, nm1, st1;
    real eps;
    integer iwk;
    real tol;
    integer difl, difr;
    real rcnd;
    integer perm, nsub, nlvl, sqre, bxst;
    extern /* Subroutine */ int srot_(integer *, real *, integer *, real *, 
	    integer *, real *, real *), sgemm_(char *, char *, integer *, 
	    integer *, integer *, real *, real *, integer *, real *, integer *
, real *, real *, integer *);
    integer poles, sizei, nsize;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *);
    integer nwork, icmpq1, icmpq2;
    extern doublereal slamch_(char *);
    extern /* Subroutine */ int slasda_(integer *, integer *, integer *, 
	    integer *, real *, real *, real *, integer *, real *, integer *, 
	    real *, real *, real *, real *, integer *, integer *, integer *, 
	    integer *, real *, real *, real *, real *, integer *, integer *), 
	    xerbla_(char *, integer *), slalsa_(integer *, integer *, 
	    integer *, integer *, real *, integer *, real *, integer *, real *
, integer *, real *, integer *, real *, real *, real *, real *, 
	    integer *, integer *, integer *, integer *, real *, real *, real *
, real *, integer *, integer *), slascl_(char *, integer *, 
	    integer *, real *, real *, integer *, integer *, real *, integer *
, integer *);
    integer givcol;
    extern integer isamax_(integer *, real *, integer *);
    extern /* Subroutine */ int slasdq_(char *, integer *, integer *, integer 
	    *, integer *, integer *, real *, real *, real *, integer *, real *
, integer *, real *, integer *, real *, integer *), 
	    slacpy_(char *, integer *, integer *, real *, integer *, real *, 
	    integer *), slartg_(real *, real *, real *, real *, real *
), slaset_(char *, integer *, integer *, real *, real *, real *, 
	    integer *);
    real orgnrm;
    integer givnum;
    extern doublereal slanst_(char *, integer *, real *, real *);
    extern /* Subroutine */ int slasrt_(char *, integer *, real *, integer *);
    integer givptr, smlszp;


/*  -- LAPACK routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLALSD uses the singular value decomposition of A to solve the least */
/*  squares problem of finding X to minimize the Euclidean norm of each */
/*  column of A*X-B, where A is N-by-N upper bidiagonal, and X and B */
/*  are N-by-NRHS. The solution X overwrites B. */

/*  The singular values of A smaller than RCOND times the largest */
/*  singular value are treated as zero in solving the least squares */
/*  problem; in this case a minimum norm solution is returned. */
/*  The actual singular values are returned in D in ascending order. */

/*  This code makes very mild assumptions about floating point */
/*  arithmetic. It will work on machines with a guard digit in */
/*  add/subtract, or on those binary machines without guard digits */
/*  which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2. */
/*  It could conceivably fail on hexadecimal or decimal machines */
/*  without guard digits, but we know of none. */

/*  Arguments */
/*  ========= */

/*  UPLO   (input) CHARACTER*1 */
/*         = 'U': D and E define an upper bidiagonal matrix. */
/*         = 'L': D and E define a  lower bidiagonal matrix. */

/*  SMLSIZ (input) INTEGER */
/*         The maximum size of the subproblems at the bottom of the */
/*         computation tree. */

/*  N      (input) INTEGER */
/*         The dimension of the  bidiagonal matrix.  N >= 0. */

/*  NRHS   (input) INTEGER */
/*         The number of columns of B. NRHS must be at least 1. */

/*  D      (input/output) REAL array, dimension (N) */
/*         On entry D contains the main diagonal of the bidiagonal */
/*         matrix. On exit, if INFO = 0, D contains its singular values. */

/*  E      (input/output) REAL array, dimension (N-1) */
/*         Contains the super-diagonal entries of the bidiagonal matrix. */
/*         On exit, E has been destroyed. */

/*  B      (input/output) REAL array, dimension (LDB,NRHS) */
/*         On input, B contains the right hand sides of the least */
/*         squares problem. On output, B contains the solution X. */

/*  LDB    (input) INTEGER */
/*         The leading dimension of B in the calling subprogram. */
/*         LDB must be at least max(1,N). */

/*  RCOND  (input) REAL */
/*         The singular values of A less than or equal to RCOND times */
/*         the largest singular value are treated as zero in solving */
/*         the least squares problem. If RCOND is negative, */
/*         machine precision is used instead. */
/*         For example, if diag(S)*X=B were the least squares problem, */
/*         where diag(S) is a diagonal matrix of singular values, the */
/*         solution would be X(i) = B(i) / S(i) if S(i) is greater than */
/*         RCOND*max(S), and X(i) = 0 if S(i) is less than or equal to */
/*         RCOND*max(S). */

/*  RANK   (output) INTEGER */
/*         The number of singular values of A greater than RCOND times */
/*         the largest singular value. */

/*  WORK   (workspace) REAL array, dimension at least */
/*         (9*N + 2*N*SMLSIZ + 8*N*NLVL + N*NRHS + (SMLSIZ+1)**2), */
/*         where NLVL = max(0, INT(log_2 (N/(SMLSIZ+1))) + 1). */

/*  IWORK  (workspace) INTEGER array, dimension at least */
/*         (3*N*NLVL + 11*N) */

/*  INFO   (output) INTEGER */
/*         = 0:  successful exit. */
/*         < 0:  if INFO = -i, the i-th argument had an illegal value. */
/*         > 0:  The algorithm failed to compute an singular value while */
/*               working on the submatrix lying in rows and columns */
/*               INFO/(N+1) through MOD(INFO,N+1). */

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
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;
    --iwork;

    /* Function Body */
    *info = 0;

    if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 1) {
	*info = -4;
    } else if (*ldb < 1 || *ldb < *n) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLALSD", &i__1);
	return 0;
    }

    eps = slamch_("Epsilon");

/*     Set up the tolerance. */

    if (*rcond <= 0.f || *rcond >= 1.f) {
	rcnd = eps;
    } else {
	rcnd = *rcond;
    }

    *rank = 0;

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	if (d__[1] == 0.f) {
	    slaset_("A", &c__1, nrhs, &c_b6, &c_b6, &b[b_offset], ldb);
	} else {
	    *rank = 1;
	    slascl_("G", &c__0, &c__0, &d__[1], &c_b11, &c__1, nrhs, &b[
		    b_offset], ldb, info);
	    d__[1] = dabs(d__[1]);
	}
	return 0;
    }

/*     Rotate the matrix if it is lower bidiagonal. */

    if (*(unsigned char *)uplo == 'L') {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    slartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (*nrhs == 1) {
		srot_(&c__1, &b[i__ + b_dim1], &c__1, &b[i__ + 1 + b_dim1], &
			c__1, &cs, &sn);
	    } else {
		work[(i__ << 1) - 1] = cs;
		work[i__ * 2] = sn;
	    }
/* L10: */
	}
	if (*nrhs > 1) {
	    i__1 = *nrhs;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		i__2 = *n - 1;
		for (j = 1; j <= i__2; ++j) {
		    cs = work[(j << 1) - 1];
		    sn = work[j * 2];
		    srot_(&c__1, &b[j + i__ * b_dim1], &c__1, &b[j + 1 + i__ *
			     b_dim1], &c__1, &cs, &sn);
/* L20: */
		}
/* L30: */
	    }
	}
    }

/*     Scale. */

    nm1 = *n - 1;
    orgnrm = slanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.f) {
	slaset_("A", n, nrhs, &c_b6, &c_b6, &b[b_offset], ldb);
	return 0;
    }

    slascl_("G", &c__0, &c__0, &orgnrm, &c_b11, n, &c__1, &d__[1], n, info);
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b11, &nm1, &c__1, &e[1], &nm1, 
	    info);

/*     If N is smaller than the minimum divide size SMLSIZ, then solve */
/*     the problem with another solver. */

    if (*n <= *smlsiz) {
	nwork = *n * *n + 1;
	slaset_("A", n, n, &c_b6, &c_b11, &work[1], n);
	slasdq_("U", &c__0, n, n, &c__0, nrhs, &d__[1], &e[1], &work[1], n, &
		work[1], n, &b[b_offset], ldb, &work[nwork], info);
	if (*info != 0) {
	    return 0;
	}
	tol = rcnd * (r__1 = d__[isamax_(n, &d__[1], &c__1)], dabs(r__1));
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (d__[i__] <= tol) {
		slaset_("A", &c__1, nrhs, &c_b6, &c_b6, &b[i__ + b_dim1], ldb);
	    } else {
		slascl_("G", &c__0, &c__0, &d__[i__], &c_b11, &c__1, nrhs, &b[
			i__ + b_dim1], ldb, info);
		++(*rank);
	    }
/* L40: */
	}
	sgemm_("T", "N", n, nrhs, n, &c_b11, &work[1], n, &b[b_offset], ldb, &
		c_b6, &work[nwork], n);
	slacpy_("A", n, nrhs, &work[nwork], n, &b[b_offset], ldb);

/*        Unscale. */

	slascl_("G", &c__0, &c__0, &c_b11, &orgnrm, n, &c__1, &d__[1], n, 
		info);
	slasrt_("D", n, &d__[1], info);
	slascl_("G", &c__0, &c__0, &orgnrm, &c_b11, n, nrhs, &b[b_offset], 
		ldb, info);

	return 0;
    }

/*     Book-keeping and setting up some constants. */

    nlvl = (integer) (log((real) (*n) / (real) (*smlsiz + 1)) / log(2.f)) + 1;

    smlszp = *smlsiz + 1;

    u = 1;
    vt = *smlsiz * *n + 1;
    difl = vt + smlszp * *n;
    difr = difl + nlvl * *n;
    z__ = difr + (nlvl * *n << 1);
    c__ = z__ + nlvl * *n;
    s = c__ + *n;
    poles = s + *n;
    givnum = poles + (nlvl << 1) * *n;
    bx = givnum + (nlvl << 1) * *n;
    nwork = bx + *n * *nrhs;

    sizei = *n + 1;
    k = sizei + *n;
    givptr = k + *n;
    perm = givptr + *n;
    givcol = perm + nlvl * *n;
    iwk = givcol + (nlvl * *n << 1);

    st = 1;
    sqre = 0;
    icmpq1 = 1;
    icmpq2 = 0;
    nsub = 0;

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = d__[i__], dabs(r__1)) < eps) {
	    d__[i__] = r_sign(&eps, &d__[i__]);
	}
/* L50: */
    }

    i__1 = nm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((r__1 = e[i__], dabs(r__1)) < eps || i__ == nm1) {
	    ++nsub;
	    iwork[nsub] = st;

/*           Subproblem found. First determine its size and then */
/*           apply divide and conquer on it. */

	    if (i__ < nm1) {

/*              A subproblem with E(I) small for I < NM1. */

		nsize = i__ - st + 1;
		iwork[sizei + nsub - 1] = nsize;
	    } else if ((r__1 = e[i__], dabs(r__1)) >= eps) {

/*              A subproblem with E(NM1) not too small but I = NM1. */

		nsize = *n - st + 1;
		iwork[sizei + nsub - 1] = nsize;
	    } else {

/*              A subproblem with E(NM1) small. This implies an */
/*              1-by-1 subproblem at D(N), which is not solved */
/*              explicitly. */

		nsize = i__ - st + 1;
		iwork[sizei + nsub - 1] = nsize;
		++nsub;
		iwork[nsub] = *n;
		iwork[sizei + nsub - 1] = 1;
		scopy_(nrhs, &b[*n + b_dim1], ldb, &work[bx + nm1], n);
	    }
	    st1 = st - 1;
	    if (nsize == 1) {

/*              This is a 1-by-1 subproblem and is not solved */
/*              explicitly. */

		scopy_(nrhs, &b[st + b_dim1], ldb, &work[bx + st1], n);
	    } else if (nsize <= *smlsiz) {

/*              This is a small subproblem and is solved by SLASDQ. */

		slaset_("A", &nsize, &nsize, &c_b6, &c_b11, &work[vt + st1], 
			n);
		slasdq_("U", &c__0, &nsize, &nsize, &c__0, nrhs, &d__[st], &e[
			st], &work[vt + st1], n, &work[nwork], n, &b[st + 
			b_dim1], ldb, &work[nwork], info);
		if (*info != 0) {
		    return 0;
		}
		slacpy_("A", &nsize, nrhs, &b[st + b_dim1], ldb, &work[bx + 
			st1], n);
	    } else {

/*              A large problem. Solve it using divide and conquer. */

		slasda_(&icmpq1, smlsiz, &nsize, &sqre, &d__[st], &e[st], &
			work[u + st1], n, &work[vt + st1], &iwork[k + st1], &
			work[difl + st1], &work[difr + st1], &work[z__ + st1], 
			 &work[poles + st1], &iwork[givptr + st1], &iwork[
			givcol + st1], n, &iwork[perm + st1], &work[givnum + 
			st1], &work[c__ + st1], &work[s + st1], &work[nwork], 
			&iwork[iwk], info);
		if (*info != 0) {
		    return 0;
		}
		bxst = bx + st1;
		slalsa_(&icmpq2, smlsiz, &nsize, nrhs, &b[st + b_dim1], ldb, &
			work[bxst], n, &work[u + st1], n, &work[vt + st1], &
			iwork[k + st1], &work[difl + st1], &work[difr + st1], 
			&work[z__ + st1], &work[poles + st1], &iwork[givptr + 
			st1], &iwork[givcol + st1], n, &iwork[perm + st1], &
			work[givnum + st1], &work[c__ + st1], &work[s + st1], 
			&work[nwork], &iwork[iwk], info);
		if (*info != 0) {
		    return 0;
		}
	    }
	    st = i__ + 1;
	}
/* L60: */
    }

/*     Apply the singular values and treat the tiny ones as zero. */

    tol = rcnd * (r__1 = d__[isamax_(n, &d__[1], &c__1)], dabs(r__1));

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Some of the elements in D can be negative because 1-by-1 */
/*        subproblems were not solved explicitly. */

	if ((r__1 = d__[i__], dabs(r__1)) <= tol) {
	    slaset_("A", &c__1, nrhs, &c_b6, &c_b6, &work[bx + i__ - 1], n);
	} else {
	    ++(*rank);
	    slascl_("G", &c__0, &c__0, &d__[i__], &c_b11, &c__1, nrhs, &work[
		    bx + i__ - 1], n, info);
	}
	d__[i__] = (r__1 = d__[i__], dabs(r__1));
/* L70: */
    }

/*     Now apply back the right singular vectors. */

    icmpq2 = 1;
    i__1 = nsub;
    for (i__ = 1; i__ <= i__1; ++i__) {
	st = iwork[i__];
	st1 = st - 1;
	nsize = iwork[sizei + i__ - 1];
	bxst = bx + st1;
	if (nsize == 1) {
	    scopy_(nrhs, &work[bxst], n, &b[st + b_dim1], ldb);
	} else if (nsize <= *smlsiz) {
	    sgemm_("T", "N", &nsize, nrhs, &nsize, &c_b11, &work[vt + st1], n, 
		     &work[bxst], n, &c_b6, &b[st + b_dim1], ldb);
	} else {
	    slalsa_(&icmpq2, smlsiz, &nsize, nrhs, &work[bxst], n, &b[st + 
		    b_dim1], ldb, &work[u + st1], n, &work[vt + st1], &iwork[
		    k + st1], &work[difl + st1], &work[difr + st1], &work[z__ 
		    + st1], &work[poles + st1], &iwork[givptr + st1], &iwork[
		    givcol + st1], n, &iwork[perm + st1], &work[givnum + st1], 
		     &work[c__ + st1], &work[s + st1], &work[nwork], &iwork[
		    iwk], info);
	    if (*info != 0) {
		return 0;
	    }
	}
/* L80: */
    }

/*     Unscale and sort the singular values. */

    slascl_("G", &c__0, &c__0, &c_b11, &orgnrm, n, &c__1, &d__[1], n, info);
    slasrt_("D", n, &d__[1], info);
    slascl_("G", &c__0, &c__0, &orgnrm, &c_b11, n, nrhs, &b[b_offset], ldb, 
	    info);

    return 0;

/*     End of SLALSD */

} /* slalsd_ */
