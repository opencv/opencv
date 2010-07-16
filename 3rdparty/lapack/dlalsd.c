/* dlalsd.f -- translated by f2c (version 20061008).
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
static doublereal c_b6 = 0.;
static integer c__0 = 0;
static doublereal c_b11 = 1.;

/* Subroutine */ int dlalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, doublereal *d__, doublereal *e, doublereal *b, integer *ldb, 
	doublereal *rcond, integer *rank, doublereal *work, integer *iwork, 
	integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2;
    doublereal d__1;

    /* Builtin functions */
    double log(doublereal), d_sign(doublereal *, doublereal *);

    /* Local variables */
    integer c__, i__, j, k;
    doublereal r__;
    integer s, u, z__;
    doublereal cs;
    integer bx;
    doublereal sn;
    integer st, vt, nm1, st1;
    doublereal eps;
    integer iwk;
    doublereal tol;
    integer difl, difr;
    doublereal rcnd;
    integer perm, nsub;
    extern /* Subroutine */ int drot_(integer *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *);
    integer nlvl, sqre, bxst;
    extern /* Subroutine */ int dgemm_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *),
	     dcopy_(integer *, doublereal *, integer *, doublereal *, integer 
	    *);
    integer poles, sizei, nsize, nwork, icmpq1, icmpq2;
    extern doublereal dlamch_(char *);
    extern /* Subroutine */ int dlasda_(integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, doublereal *, 
	     doublereal *, integer *, integer *, integer *, integer *, 
	    doublereal *, doublereal *, doublereal *, doublereal *, integer *, 
	     integer *), dlalsa_(integer *, integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, integer *, integer *, integer *, 
	    integer *, doublereal *, doublereal *, doublereal *, doublereal *, 
	     integer *, integer *), dlascl_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, integer *, doublereal *, 
	    integer *, integer *);
    extern integer idamax_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int dlasdq_(char *, integer *, integer *, integer 
	    *, integer *, integer *, doublereal *, doublereal *, doublereal *, 
	     integer *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, integer *), dlacpy_(char *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, integer *), dlartg_(doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *), dlaset_(char *, integer *, integer *, 
	     doublereal *, doublereal *, doublereal *, integer *), 
	    xerbla_(char *, integer *);
    integer givcol;
    extern doublereal dlanst_(char *, integer *, doublereal *, doublereal *);
    extern /* Subroutine */ int dlasrt_(char *, integer *, doublereal *, 
	    integer *);
    doublereal orgnrm;
    integer givnum, givptr, smlszp;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLALSD uses the singular value decomposition of A to solve the least */
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

/*  D      (input/output) DOUBLE PRECISION array, dimension (N) */
/*         On entry D contains the main diagonal of the bidiagonal */
/*         matrix. On exit, if INFO = 0, D contains its singular values. */

/*  E      (input/output) DOUBLE PRECISION array, dimension (N-1) */
/*         Contains the super-diagonal entries of the bidiagonal matrix. */
/*         On exit, E has been destroyed. */

/*  B      (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS) */
/*         On input, B contains the right hand sides of the least */
/*         squares problem. On output, B contains the solution X. */

/*  LDB    (input) INTEGER */
/*         The leading dimension of B in the calling subprogram. */
/*         LDB must be at least max(1,N). */

/*  RCOND  (input) DOUBLE PRECISION */
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

/*  WORK   (workspace) DOUBLE PRECISION array, dimension at least */
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
	xerbla_("DLALSD", &i__1);
	return 0;
    }

    eps = dlamch_("Epsilon");

/*     Set up the tolerance. */

    if (*rcond <= 0. || *rcond >= 1.) {
	rcnd = eps;
    } else {
	rcnd = *rcond;
    }

    *rank = 0;

/*     Quick return if possible. */

    if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	if (d__[1] == 0.) {
	    dlaset_("A", &c__1, nrhs, &c_b6, &c_b6, &b[b_offset], ldb);
	} else {
	    *rank = 1;
	    dlascl_("G", &c__0, &c__0, &d__[1], &c_b11, &c__1, nrhs, &b[
		    b_offset], ldb, info);
	    d__[1] = abs(d__[1]);
	}
	return 0;
    }

/*     Rotate the matrix if it is lower bidiagonal. */

    if (*(unsigned char *)uplo == 'L') {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (*nrhs == 1) {
		drot_(&c__1, &b[i__ + b_dim1], &c__1, &b[i__ + 1 + b_dim1], &
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
		    drot_(&c__1, &b[j + i__ * b_dim1], &c__1, &b[j + 1 + i__ *
			     b_dim1], &c__1, &cs, &sn);
/* L20: */
		}
/* L30: */
	    }
	}
    }

/*     Scale. */

    nm1 = *n - 1;
    orgnrm = dlanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.) {
	dlaset_("A", n, nrhs, &c_b6, &c_b6, &b[b_offset], ldb);
	return 0;
    }

    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b11, n, &c__1, &d__[1], n, info);
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b11, &nm1, &c__1, &e[1], &nm1, 
	    info);

/*     If N is smaller than the minimum divide size SMLSIZ, then solve */
/*     the problem with another solver. */

    if (*n <= *smlsiz) {
	nwork = *n * *n + 1;
	dlaset_("A", n, n, &c_b6, &c_b11, &work[1], n);
	dlasdq_("U", &c__0, n, n, &c__0, nrhs, &d__[1], &e[1], &work[1], n, &
		work[1], n, &b[b_offset], ldb, &work[nwork], info);
	if (*info != 0) {
	    return 0;
	}
	tol = rcnd * (d__1 = d__[idamax_(n, &d__[1], &c__1)], abs(d__1));
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (d__[i__] <= tol) {
		dlaset_("A", &c__1, nrhs, &c_b6, &c_b6, &b[i__ + b_dim1], ldb);
	    } else {
		dlascl_("G", &c__0, &c__0, &d__[i__], &c_b11, &c__1, nrhs, &b[
			i__ + b_dim1], ldb, info);
		++(*rank);
	    }
/* L40: */
	}
	dgemm_("T", "N", n, nrhs, n, &c_b11, &work[1], n, &b[b_offset], ldb, &
		c_b6, &work[nwork], n);
	dlacpy_("A", n, nrhs, &work[nwork], n, &b[b_offset], ldb);

/*        Unscale. */

	dlascl_("G", &c__0, &c__0, &c_b11, &orgnrm, n, &c__1, &d__[1], n, 
		info);
	dlasrt_("D", n, &d__[1], info);
	dlascl_("G", &c__0, &c__0, &orgnrm, &c_b11, n, nrhs, &b[b_offset], 
		ldb, info);

	return 0;
    }

/*     Book-keeping and setting up some constants. */

    nlvl = (integer) (log((doublereal) (*n) / (doublereal) (*smlsiz + 1)) / 
	    log(2.)) + 1;

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
	if ((d__1 = d__[i__], abs(d__1)) < eps) {
	    d__[i__] = d_sign(&eps, &d__[i__]);
	}
/* L50: */
    }

    i__1 = nm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((d__1 = e[i__], abs(d__1)) < eps || i__ == nm1) {
	    ++nsub;
	    iwork[nsub] = st;

/*           Subproblem found. First determine its size and then */
/*           apply divide and conquer on it. */

	    if (i__ < nm1) {

/*              A subproblem with E(I) small for I < NM1. */

		nsize = i__ - st + 1;
		iwork[sizei + nsub - 1] = nsize;
	    } else if ((d__1 = e[i__], abs(d__1)) >= eps) {

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
		dcopy_(nrhs, &b[*n + b_dim1], ldb, &work[bx + nm1], n);
	    }
	    st1 = st - 1;
	    if (nsize == 1) {

/*              This is a 1-by-1 subproblem and is not solved */
/*              explicitly. */

		dcopy_(nrhs, &b[st + b_dim1], ldb, &work[bx + st1], n);
	    } else if (nsize <= *smlsiz) {

/*              This is a small subproblem and is solved by DLASDQ. */

		dlaset_("A", &nsize, &nsize, &c_b6, &c_b11, &work[vt + st1], 
			n);
		dlasdq_("U", &c__0, &nsize, &nsize, &c__0, nrhs, &d__[st], &e[
			st], &work[vt + st1], n, &work[nwork], n, &b[st + 
			b_dim1], ldb, &work[nwork], info);
		if (*info != 0) {
		    return 0;
		}
		dlacpy_("A", &nsize, nrhs, &b[st + b_dim1], ldb, &work[bx + 
			st1], n);
	    } else {

/*              A large problem. Solve it using divide and conquer. */

		dlasda_(&icmpq1, smlsiz, &nsize, &sqre, &d__[st], &e[st], &
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
		dlalsa_(&icmpq2, smlsiz, &nsize, nrhs, &b[st + b_dim1], ldb, &
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

    tol = rcnd * (d__1 = d__[idamax_(n, &d__[1], &c__1)], abs(d__1));

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Some of the elements in D can be negative because 1-by-1 */
/*        subproblems were not solved explicitly. */

	if ((d__1 = d__[i__], abs(d__1)) <= tol) {
	    dlaset_("A", &c__1, nrhs, &c_b6, &c_b6, &work[bx + i__ - 1], n);
	} else {
	    ++(*rank);
	    dlascl_("G", &c__0, &c__0, &d__[i__], &c_b11, &c__1, nrhs, &work[
		    bx + i__ - 1], n, info);
	}
	d__[i__] = (d__1 = d__[i__], abs(d__1));
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
	    dcopy_(nrhs, &work[bxst], n, &b[st + b_dim1], ldb);
	} else if (nsize <= *smlsiz) {
	    dgemm_("T", "N", &nsize, nrhs, &nsize, &c_b11, &work[vt + st1], n, 
		     &work[bxst], n, &c_b6, &b[st + b_dim1], ldb);
	} else {
	    dlalsa_(&icmpq2, smlsiz, &nsize, nrhs, &work[bxst], n, &b[st + 
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

    dlascl_("G", &c__0, &c__0, &c_b11, &orgnrm, n, &c__1, &d__[1], n, info);
    dlasrt_("D", n, &d__[1], info);
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b11, n, nrhs, &b[b_offset], ldb, 
	    info);

    return 0;

/*     End of DLALSD */

} /* dlalsd_ */
