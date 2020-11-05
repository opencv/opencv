/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DBDSDC
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DBDSDC + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dbdsdc.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dbdsdc.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dbdsdc.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DBDSDC( UPLO, COMPQ, N, D, E, U, LDU, VT, LDVT, Q, IQ,
//                         WORK, IWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          COMPQ, UPLO
//      INTEGER            INFO, LDU, LDVT, N
//      ..
//      .. Array Arguments ..
//      INTEGER            IQ( * ), IWORK( * )
//      DOUBLE PRECISION   D( * ), E( * ), Q( * ), U( LDU, * ),
//     $                   VT( LDVT, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DBDSDC computes the singular value decomposition (SVD) of a real
//> N-by-N (upper or lower) bidiagonal matrix B:  B = U * S * VT,
//> using a divide and conquer method, where S is a diagonal matrix
//> with non-negative diagonal elements (the singular values of B), and
//> U and VT are orthogonal matrices of left and right singular vectors,
//> respectively. DBDSDC can be used to compute all singular values,
//> and optionally, singular vectors or singular vectors in compact form.
//>
//> This code makes very mild assumptions about floating point
//> arithmetic. It will work on machines with a guard digit in
//> add/subtract, or on those binary machines without guard digits
//> which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
//> It could conceivably fail on hexadecimal or decimal machines
//> without guard digits, but we know of none.  See DLASD3 for details.
//>
//> The code currently calls DLASDQ if singular values only are desired.
//> However, it can be slightly modified to compute singular values
//> using the divide and conquer method.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U':  B is upper bidiagonal.
//>          = 'L':  B is lower bidiagonal.
//> \endverbatim
//>
//> \param[in] COMPQ
//> \verbatim
//>          COMPQ is CHARACTER*1
//>          Specifies whether singular vectors are to be computed
//>          as follows:
//>          = 'N':  Compute singular values only;
//>          = 'P':  Compute singular values and compute singular
//>                  vectors in compact form;
//>          = 'I':  Compute singular values and singular vectors.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix B.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the n diagonal elements of the bidiagonal matrix B.
//>          On exit, if INFO=0, the singular values of B.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          On entry, the elements of E contain the offdiagonal
//>          elements of the bidiagonal matrix whose SVD is desired.
//>          On exit, E has been destroyed.
//> \endverbatim
//>
//> \param[out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension (LDU,N)
//>          If  COMPQ = 'I', then:
//>             On exit, if INFO = 0, U contains the left singular vectors
//>             of the bidiagonal matrix.
//>          For other values of COMPQ, U is not referenced.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>          The leading dimension of the array U.  LDU >= 1.
//>          If singular vectors are desired, then LDU >= max( 1, N ).
//> \endverbatim
//>
//> \param[out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension (LDVT,N)
//>          If  COMPQ = 'I', then:
//>             On exit, if INFO = 0, VT**T contains the right singular
//>             vectors of the bidiagonal matrix.
//>          For other values of COMPQ, VT is not referenced.
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>          The leading dimension of the array VT.  LDVT >= 1.
//>          If singular vectors are desired, then LDVT >= max( 1, N ).
//> \endverbatim
//>
//> \param[out] Q
//> \verbatim
//>          Q is DOUBLE PRECISION array, dimension (LDQ)
//>          If  COMPQ = 'P', then:
//>             On exit, if INFO = 0, Q and IQ contain the left
//>             and right singular vectors in a compact form,
//>             requiring O(N log N) space instead of 2*N**2.
//>             In particular, Q contains all the DOUBLE PRECISION data in
//>             LDQ >= N*(11 + 2*SMLSIZ + 8*INT(LOG_2(N/(SMLSIZ+1))))
//>             words of memory, where SMLSIZ is returned by ILAENV and
//>             is equal to the maximum size of the subproblems at the
//>             bottom of the computation tree (usually about 25).
//>          For other values of COMPQ, Q is not referenced.
//> \endverbatim
//>
//> \param[out] IQ
//> \verbatim
//>          IQ is INTEGER array, dimension (LDIQ)
//>          If  COMPQ = 'P', then:
//>             On exit, if INFO = 0, Q and IQ contain the left
//>             and right singular vectors in a compact form,
//>             requiring O(N log N) space instead of 2*N**2.
//>             In particular, IQ contains all INTEGER data in
//>             LDIQ >= N*(3 + 3*INT(LOG_2(N/(SMLSIZ+1))))
//>             words of memory, where SMLSIZ is returned by ILAENV and
//>             is equal to the maximum size of the subproblems at the
//>             bottom of the computation tree (usually about 25).
//>          For other values of COMPQ, IQ is not referenced.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          If COMPQ = 'N' then LWORK >= (4 * N).
//>          If COMPQ = 'P' then LWORK >= (6 * N).
//>          If COMPQ = 'I' then LWORK >= (3 * N**2 + 4 * N).
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (8*N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  The algorithm failed to compute a singular value.
//>                The update process of divide and conquer failed.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup auxOTHERcomputational
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dbdsdc_(char *uplo, char *compq, int *n, double *d__,
	double *e, double *u, int *ldu, double *vt, int *ldvt, double *q, int
	*iq, double *work, int *iwork, int *info)
{
    // Table of constant values
    int c__9 = 9;
    int c__0 = 0;
    double c_b15 = 1.;
    int c__1 = 1;
    double c_b29 = 0.;

    // System generated locals
    int u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;
    double d__1;

    // Local variables
    int i__, j, k;
    double p, r__;
    int z__, ic, ii, kk;
    double cs;
    int is, iu;
    double sn;
    int nm1;
    double eps;
    int ivt, difl, difr, ierr, perm, mlvl, sqre;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dlasr_(char *, char *, char *, int *, int *,
	    double *, double *, double *, int *), dcopy_(int *, double *, int
	    *, double *, int *), dswap_(int *, double *, int *, double *, int
	    *);
    int poles, iuplo, nsize, start;
    extern /* Subroutine */ int dlasd0_(int *, int *, double *, double *,
	    double *, int *, double *, int *, int *, int *, double *, int *);
    extern double dlamch_(char *);
    extern /* Subroutine */ int dlasda_(int *, int *, int *, int *, double *,
	    double *, double *, int *, double *, int *, double *, double *,
	    double *, double *, int *, int *, int *, int *, double *, double *
	    , double *, double *, int *, int *), dlascl_(char *, int *, int *,
	     double *, double *, int *, int *, double *, int *, int *),
	    dlasdq_(char *, int *, int *, int *, int *, int *, double *,
	    double *, double *, int *, double *, int *, double *, int *,
	    double *, int *), dlaset_(char *, int *, int *, double *, double *
	    , double *, int *), dlartg_(double *, double *, double *, double *
	    , double *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    int givcol;
    extern double dlanst_(char *, int *, double *, double *);
    int icompq;
    double orgnrm;
    int givnum, givptr, qstart, smlsiz, wstart, smlszp;

    //
    // -- LAPACK computational routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    // Changed dimension statement in comment describing E from (N) to
    // (N-1).  Sven, 17 Feb 05.
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
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

    // Function Body
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
	xerbla_("DBDSDC", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	return 0;
    }
    smlsiz = ilaenv_(&c__9, "DBDSDC", " ", &c__0, &c__0, &c__0, &c__0);
    if (*n == 1) {
	if (icompq == 1) {
	    q[1] = d_sign(&c_b15, &d__[1]);
	    q[smlsiz * *n + 1] = 1.;
	} else if (icompq == 2) {
	    u[u_dim1 + 1] = d_sign(&c_b15, &d__[1]);
	    vt[vt_dim1 + 1] = 1.;
	}
	d__[1] = abs(d__[1]);
	return 0;
    }
    nm1 = *n - 1;
    //
    //    If matrix lower bidiagonal, rotate to be upper bidiagonal
    //    by applying Givens rotations on the left
    //
    wstart = 1;
    qstart = 3;
    if (icompq == 1) {
	dcopy_(n, &d__[1], &c__1, &q[1], &c__1);
	i__1 = *n - 1;
	dcopy_(&i__1, &e[1], &c__1, &q[*n + 1], &c__1);
    }
    if (iuplo == 2) {
	qstart = 5;
	if (icompq == 2) {
	    wstart = (*n << 1) - 1;
	}
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
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
// L10:
	}
    }
    //
    //    If ICOMPQ = 0, use DLASDQ to compute the singular values.
    //
    if (icompq == 0) {
	//       Ignore WSTART, instead using WORK( 1 ), since the two vectors
	//       for CS and -SN above are added only if ICOMPQ == 2,
	//       and adding them exceeds documented WORK size of 4*n.
	dlasdq_("U", &c__0, n, &c__0, &c__0, &c__0, &d__[1], &e[1], &vt[
		vt_offset], ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[
		1], info);
	goto L40;
    }
    //
    //    If N is smaller than the minimum divide size SMLSIZ, then solve
    //    the problem with another solver.
    //
    if (*n <= smlsiz) {
	if (icompq == 2) {
	    dlaset_("A", n, n, &c_b29, &c_b15, &u[u_offset], ldu);
	    dlaset_("A", n, n, &c_b29, &c_b15, &vt[vt_offset], ldvt);
	    dlasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &vt[vt_offset]
		    , ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[
		    wstart], info);
	} else if (icompq == 1) {
	    iu = 1;
	    ivt = iu + *n;
	    dlaset_("A", n, n, &c_b29, &c_b15, &q[iu + (qstart - 1) * *n], n);
	    dlaset_("A", n, n, &c_b29, &c_b15, &q[ivt + (qstart - 1) * *n], n)
		    ;
	    dlasdq_("U", &c__0, n, n, n, &c__0, &d__[1], &e[1], &q[ivt + (
		    qstart - 1) * *n], n, &q[iu + (qstart - 1) * *n], n, &q[
		    iu + (qstart - 1) * *n], n, &work[wstart], info);
	}
	goto L40;
    }
    if (icompq == 2) {
	dlaset_("A", n, n, &c_b29, &c_b15, &u[u_offset], ldu);
	dlaset_("A", n, n, &c_b29, &c_b15, &vt[vt_offset], ldvt);
    }
    //
    //    Scale.
    //
    orgnrm = dlanst_("M", n, &d__[1], &e[1]);
    if (orgnrm == 0.) {
	return 0;
    }
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b15, n, &c__1, &d__[1], n, &ierr);
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b15, &nm1, &c__1, &e[1], &nm1, &
	    ierr);
    eps = dlamch_("Epsilon") * .9;
    mlvl = (int) (log((double) (*n) / (double) (smlsiz + 1)) / log(2.)) + 1;
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
	if ((d__1 = d__[i__], abs(d__1)) < eps) {
	    d__[i__] = d_sign(&eps, &d__[i__]);
	}
// L20:
    }
    start = 1;
    sqre = 0;
    i__1 = nm1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((d__1 = e[i__], abs(d__1)) < eps || i__ == nm1) {
	    //
	    //          Subproblem found. First determine its size and then
	    //          apply divide and conquer on it.
	    //
	    if (i__ < nm1) {
		//
		//             A subproblem with E(I) small for I < NM1.
		//
		nsize = i__ - start + 1;
	    } else if ((d__1 = e[i__], abs(d__1)) >= eps) {
		//
		//             A subproblem with E(NM1) not too small but I = NM1.
		//
		nsize = *n - start + 1;
	    } else {
		//
		//             A subproblem with E(NM1) small. This implies an
		//             1-by-1 subproblem at D(N). Solve this 1-by-1 problem
		//             first.
		//
		nsize = i__ - start + 1;
		if (icompq == 2) {
		    u[*n + *n * u_dim1] = d_sign(&c_b15, &d__[*n]);
		    vt[*n + *n * vt_dim1] = 1.;
		} else if (icompq == 1) {
		    q[*n + (qstart - 1) * *n] = d_sign(&c_b15, &d__[*n]);
		    q[*n + (smlsiz + qstart - 1) * *n] = 1.;
		}
		d__[*n] = (d__1 = d__[*n], abs(d__1));
	    }
	    if (icompq == 2) {
		dlasd0_(&nsize, &sqre, &d__[start], &e[start], &u[start +
			start * u_dim1], ldu, &vt[start + start * vt_dim1],
			ldvt, &smlsiz, &iwork[1], &work[wstart], info);
	    } else {
		dlasda_(&icompq, &smlsiz, &nsize, &sqre, &d__[start], &e[
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
	    }
	    if (*info != 0) {
		return 0;
	    }
	    start = i__ + 1;
	}
// L30:
    }
    //
    //    Unscale
    //
    dlascl_("G", &c__0, &c__0, &c_b15, &orgnrm, n, &c__1, &d__[1], n, &ierr);
L40:
    //
    //    Use Selection Sort to minimize swaps of singular vectors
    //
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
// L50:
	}
	if (kk != i__) {
	    d__[kk] = d__[i__];
	    d__[i__] = p;
	    if (icompq == 1) {
		iq[i__] = kk;
	    } else if (icompq == 2) {
		dswap_(n, &u[i__ * u_dim1 + 1], &c__1, &u[kk * u_dim1 + 1], &
			c__1);
		dswap_(n, &vt[i__ + vt_dim1], ldvt, &vt[kk + vt_dim1], ldvt);
	    }
	} else if (icompq == 1) {
	    iq[i__] = i__;
	}
// L60:
    }
    //
    //    If ICOMPQ = 1, use IQ(N,1) as the indicator for UPLO
    //
    if (icompq == 1) {
	if (iuplo == 1) {
	    iq[*n] = 1;
	} else {
	    iq[*n] = 0;
	}
    }
    //
    //    If B is lower bidiagonal, update U by those Givens rotations
    //    which rotated B to be upper bidiagonal
    //
    if (iuplo == 2 && icompq == 2) {
	dlasr_("L", "V", "B", n, n, &work[1], &work[*n], &u[u_offset], ldu);
    }
    return 0;
    //
    //    End of DBDSDC
    //
} // dbdsdc_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DBDSQR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DBDSQR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dbdsqr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dbdsqr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dbdsqr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DBDSQR( UPLO, N, NCVT, NRU, NCC, D, E, VT, LDVT, U,
//                         LDU, C, LDC, WORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDC, LDU, LDVT, N, NCC, NCVT, NRU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   C( LDC, * ), D( * ), E( * ), U( LDU, * ),
//     $                   VT( LDVT, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DBDSQR computes the singular values and, optionally, the right and/or
//> left singular vectors from the singular value decomposition (SVD) of
//> a real N-by-N (upper or lower) bidiagonal matrix B using the implicit
//> zero-shift QR algorithm.  The SVD of B has the form
//>
//>    B = Q * S * P**T
//>
//> where S is the diagonal matrix of singular values, Q is an orthogonal
//> matrix of left singular vectors, and P is an orthogonal matrix of
//> right singular vectors.  If left singular vectors are requested, this
//> subroutine actually returns U*Q instead of Q, and, if right singular
//> vectors are requested, this subroutine returns P**T*VT instead of
//> P**T, for given real input matrices U and VT.  When U and VT are the
//> orthogonal matrices that reduce a general matrix A to bidiagonal
//> form:  A = U*B*VT, as computed by DGEBRD, then
//>
//>    A = (U*Q) * S * (P**T*VT)
//>
//> is the SVD of A.  Optionally, the subroutine may also compute Q**T*C
//> for a given real input matrix C.
//>
//> See "Computing  Small Singular Values of Bidiagonal Matrices With
//> Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
//> LAPACK Working Note #3 (or SIAM J. Sci. Statist. Comput. vol. 11,
//> no. 5, pp. 873-912, Sept 1990) and
//> "Accurate singular values and differential qd algorithms," by
//> B. Parlett and V. Fernando, Technical Report CPAM-554, Mathematics
//> Department, University of California at Berkeley, July 1992
//> for a detailed description of the algorithm.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U':  B is upper bidiagonal;
//>          = 'L':  B is lower bidiagonal.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix B.  N >= 0.
//> \endverbatim
//>
//> \param[in] NCVT
//> \verbatim
//>          NCVT is INTEGER
//>          The number of columns of the matrix VT. NCVT >= 0.
//> \endverbatim
//>
//> \param[in] NRU
//> \verbatim
//>          NRU is INTEGER
//>          The number of rows of the matrix U. NRU >= 0.
//> \endverbatim
//>
//> \param[in] NCC
//> \verbatim
//>          NCC is INTEGER
//>          The number of columns of the matrix C. NCC >= 0.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the n diagonal elements of the bidiagonal matrix B.
//>          On exit, if INFO=0, the singular values of B in decreasing
//>          order.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          On entry, the N-1 offdiagonal elements of the bidiagonal
//>          matrix B.
//>          On exit, if INFO = 0, E is destroyed; if INFO > 0, D and E
//>          will contain the diagonal and superdiagonal elements of a
//>          bidiagonal matrix orthogonally equivalent to the one given
//>          as input.
//> \endverbatim
//>
//> \param[in,out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension (LDVT, NCVT)
//>          On entry, an N-by-NCVT matrix VT.
//>          On exit, VT is overwritten by P**T * VT.
//>          Not referenced if NCVT = 0.
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>          The leading dimension of the array VT.
//>          LDVT >= max(1,N) if NCVT > 0; LDVT >= 1 if NCVT = 0.
//> \endverbatim
//>
//> \param[in,out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension (LDU, N)
//>          On entry, an NRU-by-N matrix U.
//>          On exit, U is overwritten by U * Q.
//>          Not referenced if NRU = 0.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>          The leading dimension of the array U.  LDU >= max(1,NRU).
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC, NCC)
//>          On entry, an N-by-NCC matrix C.
//>          On exit, C is overwritten by Q**T * C.
//>          Not referenced if NCC = 0.
//> \endverbatim
//>
//> \param[in] LDC
//> \verbatim
//>          LDC is INTEGER
//>          The leading dimension of the array C.
//>          LDC >= max(1,N) if NCC > 0; LDC >=1 if NCC = 0.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (4*(N-1))
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  If INFO = -i, the i-th argument had an illegal value
//>          > 0:
//>             if NCVT = NRU = NCC = 0,
//>                = 1, a split was marked by a positive value in E
//>                = 2, current block of Z not diagonalized after 30*N
//>                     iterations (in inner while loop)
//>                = 3, termination criterion of outer while loop not met
//>                     (program created more than N unreduced blocks)
//>             else NCVT = NRU = NCC = 0,
//>                   the algorithm did not converge; D and E contain the
//>                   elements of a bidiagonal matrix which is orthogonally
//>                   similar to the input matrix B;  if INFO = i, i
//>                   elements of E have not converged to zero.
//> \endverbatim
//
//> \par Internal Parameters:
// =========================
//>
//> \verbatim
//>  TOLMUL  DOUBLE PRECISION, default = max(10,min(100,EPS**(-1/8)))
//>          TOLMUL controls the convergence criterion of the QR loop.
//>          If it is positive, TOLMUL*EPS is the desired relative
//>             precision in the computed singular values.
//>          If it is negative, abs(TOLMUL*EPS*sigma_max) is the
//>             desired absolute accuracy in the computed singular
//>             values (corresponds to relative accuracy
//>             abs(TOLMUL*EPS) in the largest singular value.
//>          abs(TOLMUL) should be between 1 and 1/EPS, and preferably
//>             between 10 (for fast convergence) and .1/EPS
//>             (for there to be some accuracy in the results).
//>          Default is to lose at either one eighth or 2 of the
//>             available decimal digits in each computed singular value
//>             (whichever is smaller).
//>
//>  MAXITR  INTEGER, default = 6
//>          MAXITR controls the maximum number of passes of the
//>          algorithm through its inner loop. The algorithms stops
//>          (and so fails to converge) if the number of passes
//>          through the inner loop exceeds MAXITR*N**2.
//>
//> \endverbatim
//
//> \par Note:
// ===========
//>
//> \verbatim
//>  Bug report from Cezary Dendek.
//>  On March 23rd 2017, the INTEGER variable MAXIT = MAXITR*N**2 is
//>  removed since it can overflow pretty easily (for N larger or equal
//>  than 18,919). We instead use MAXITDIVN = MAXITR*N.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dbdsqr_(char *uplo, int *n, int *ncvt, int *nru, int *
	ncc, double *d__, double *e, double *vt, int *ldvt, double *u, int *
	ldu, double *c__, int *ldc, double *work, int *info)
{
    // Table of constant values
    double c_b15 = -.125;
    int c__1 = 1;
    double c_b49 = 1.;
    double c_b72 = -1.;

    // System generated locals
    int c_dim1, c_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;
    double d__1, d__2, d__3, d__4;

    // Local variables
    int iterdivn;
    double f, g, h__;
    int i__, j, m;
    double r__;
    int maxitdivn;
    double cs;
    int ll;
    double sn, mu;
    int nm1, nm12, nm13, lll;
    double eps, sll, tol, abse;
    int idir;
    double abss;
    int oldm;
    double cosl;
    int isub, iter;
    double unfl, sinl, cosr, smin, smax, sinr;
    extern /* Subroutine */ int drot_(int *, double *, int *, double *, int *,
	     double *, double *), dlas2_(double *, double *, double *, double
	    *, double *), dscal_(int *, double *, double *, int *);
    extern int lsame_(char *, char *);
    double oldcs;
    extern /* Subroutine */ int dlasr_(char *, char *, char *, int *, int *,
	    double *, double *, double *, int *);
    int oldll;
    double shift, sigmn, oldsn;
    extern /* Subroutine */ int dswap_(int *, double *, int *, double *, int *
	    );
    double sminl, sigmx;
    int lower;
    extern /* Subroutine */ int dlasq1_(int *, double *, double *, double *,
	    int *), dlasv2_(double *, double *, double *, double *, double *,
	    double *, double *, double *, double *);
    extern double dlamch_(char *);
    extern /* Subroutine */ int dlartg_(double *, double *, double *, double *
	    , double *), xerbla_(char *, int *);
    double sminoa, thresh;
    int rotate;
    double tolmul;

    //
    // -- LAPACK computational routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;
    --e;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    *info = 0;
    lower = lsame_(uplo, "L");
    if (! lsame_(uplo, "U") && ! lower) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ncvt < 0) {
	*info = -3;
    } else if (*nru < 0) {
	*info = -4;
    } else if (*ncc < 0) {
	*info = -5;
    } else if (*ncvt == 0 && *ldvt < 1 || *ncvt > 0 && *ldvt < max(1,*n)) {
	*info = -9;
    } else if (*ldu < max(1,*nru)) {
	*info = -11;
    } else if (*ncc == 0 && *ldc < 1 || *ncc > 0 && *ldc < max(1,*n)) {
	*info = -13;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DBDSQR", &i__1);
	return 0;
    }
    if (*n == 0) {
	return 0;
    }
    if (*n == 1) {
	goto L160;
    }
    //
    //    ROTATE is true if any singular vectors desired, false otherwise
    //
    rotate = *ncvt > 0 || *nru > 0 || *ncc > 0;
    //
    //    If no singular vectors desired, use qd algorithm
    //
    if (! rotate) {
	dlasq1_(n, &d__[1], &e[1], &work[1], info);
	//
	//    If INFO equals 2, dqds didn't finish, try to finish
	//
	if (*info != 2) {
	    return 0;
	}
	*info = 0;
    }
    nm1 = *n - 1;
    nm12 = nm1 + nm1;
    nm13 = nm12 + nm1;
    idir = 0;
    //
    //    Get machine constants
    //
    eps = dlamch_("Epsilon");
    unfl = dlamch_("Safe minimum");
    //
    //    If matrix lower bidiagonal, rotate to be upper bidiagonal
    //    by applying Givens rotations on the left
    //
    if (lower) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    work[i__] = cs;
	    work[nm1 + i__] = sn;
// L10:
	}
	//
	//       Update singular vectors if desired
	//
	if (*nru > 0) {
	    dlasr_("R", "V", "F", nru, n, &work[1], &work[*n], &u[u_offset],
		    ldu);
	}
	if (*ncc > 0) {
	    dlasr_("L", "V", "F", n, ncc, &work[1], &work[*n], &c__[c_offset],
		     ldc);
	}
    }
    //
    //    Compute singular values to relative accuracy TOL
    //    (By setting TOL to be negative, algorithm will compute
    //    singular values to absolute accuracy ABS(TOL)*norm(input matrix))
    //
    // Computing MAX
    // Computing MIN
    d__3 = 100., d__4 = pow_dd(&eps, &c_b15);
    d__1 = 10., d__2 = min(d__3,d__4);
    tolmul = max(d__1,d__2);
    tol = tolmul * eps;
    //
    //    Compute approximate maximum, minimum singular values
    //
    smax = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	// Computing MAX
	d__2 = smax, d__3 = (d__1 = d__[i__], abs(d__1));
	smax = max(d__2,d__3);
// L20:
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	// Computing MAX
	d__2 = smax, d__3 = (d__1 = e[i__], abs(d__1));
	smax = max(d__2,d__3);
// L30:
    }
    sminl = 0.;
    if (tol >= 0.) {
	//
	//       Relative accuracy desired
	//
	sminoa = abs(d__[1]);
	if (sminoa == 0.) {
	    goto L50;
	}
	mu = sminoa;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    mu = (d__2 = d__[i__], abs(d__2)) * (mu / (mu + (d__1 = e[i__ - 1]
		    , abs(d__1))));
	    sminoa = min(sminoa,mu);
	    if (sminoa == 0.) {
		goto L50;
	    }
// L40:
	}
L50:
	sminoa /= sqrt((double) (*n));
	// Computing MAX
	d__1 = tol * sminoa, d__2 = *n * (*n * unfl) * 6;
	thresh = max(d__1,d__2);
    } else {
	//
	//       Absolute accuracy desired
	//
	// Computing MAX
	d__1 = abs(tol) * smax, d__2 = *n * (*n * unfl) * 6;
	thresh = max(d__1,d__2);
    }
    //
    //    Prepare for main iteration loop for the singular values
    //    (MAXIT is the maximum number of passes through the inner
    //    loop permitted before nonconvergence signalled.)
    //
    maxitdivn = *n * 6;
    iterdivn = 0;
    iter = -1;
    oldll = -1;
    oldm = -1;
    //
    //    M points to last element of unconverged part of matrix
    //
    m = *n;
    //
    //    Begin main iteration loop
    //
L60:
    //
    //    Check for convergence or exceeding iteration count
    //
    if (m <= 1) {
	goto L160;
    }
    if (iter >= *n) {
	iter -= *n;
	++iterdivn;
	if (iterdivn >= maxitdivn) {
	    goto L200;
	}
    }
    //
    //    Find diagonal block of matrix to work on
    //
    if (tol < 0. && (d__1 = d__[m], abs(d__1)) <= thresh) {
	d__[m] = 0.;
    }
    smax = (d__1 = d__[m], abs(d__1));
    smin = smax;
    i__1 = m - 1;
    for (lll = 1; lll <= i__1; ++lll) {
	ll = m - lll;
	abss = (d__1 = d__[ll], abs(d__1));
	abse = (d__1 = e[ll], abs(d__1));
	if (tol < 0. && abss <= thresh) {
	    d__[ll] = 0.;
	}
	if (abse <= thresh) {
	    goto L80;
	}
	smin = min(smin,abss);
	// Computing MAX
	d__1 = max(smax,abss);
	smax = max(d__1,abse);
// L70:
    }
    ll = 0;
    goto L90;
L80:
    e[ll] = 0.;
    //
    //    Matrix splits since E(LL) = 0
    //
    if (ll == m - 1) {
	//
	//       Convergence of bottom singular value, return to top of loop
	//
	--m;
	goto L60;
    }
L90:
    ++ll;
    //
    //    E(LL) through E(M-1) are nonzero, E(LL-1) is zero
    //
    if (ll == m - 1) {
	//
	//       2 by 2 block, handle separately
	//
	dlasv2_(&d__[m - 1], &e[m - 1], &d__[m], &sigmn, &sigmx, &sinr, &cosr,
		 &sinl, &cosl);
	d__[m - 1] = sigmx;
	e[m - 1] = 0.;
	d__[m] = sigmn;
	//
	//       Compute singular vectors, if desired
	//
	if (*ncvt > 0) {
	    drot_(ncvt, &vt[m - 1 + vt_dim1], ldvt, &vt[m + vt_dim1], ldvt, &
		    cosr, &sinr);
	}
	if (*nru > 0) {
	    drot_(nru, &u[(m - 1) * u_dim1 + 1], &c__1, &u[m * u_dim1 + 1], &
		    c__1, &cosl, &sinl);
	}
	if (*ncc > 0) {
	    drot_(ncc, &c__[m - 1 + c_dim1], ldc, &c__[m + c_dim1], ldc, &
		    cosl, &sinl);
	}
	m += -2;
	goto L60;
    }
    //
    //    If working on new submatrix, choose shift direction
    //    (from larger end diagonal element towards smaller)
    //
    if (ll > oldm || m < oldll) {
	if ((d__1 = d__[ll], abs(d__1)) >= (d__2 = d__[m], abs(d__2))) {
	    //
	    //          Chase bulge from top (big end) to bottom (small end)
	    //
	    idir = 1;
	} else {
	    //
	    //          Chase bulge from bottom (big end) to top (small end)
	    //
	    idir = 2;
	}
    }
    //
    //    Apply convergence tests
    //
    if (idir == 1) {
	//
	//       Run convergence test in forward direction
	//       First apply standard test to bottom of matrix
	//
	if ((d__2 = e[m - 1], abs(d__2)) <= abs(tol) * (d__1 = d__[m], abs(
		d__1)) || tol < 0. && (d__3 = e[m - 1], abs(d__3)) <= thresh)
		{
	    e[m - 1] = 0.;
	    goto L60;
	}
	if (tol >= 0.) {
	    //
	    //          If relative accuracy desired,
	    //          apply convergence criterion forward
	    //
	    mu = (d__1 = d__[ll], abs(d__1));
	    sminl = mu;
	    i__1 = m - 1;
	    for (lll = ll; lll <= i__1; ++lll) {
		if ((d__1 = e[lll], abs(d__1)) <= tol * mu) {
		    e[lll] = 0.;
		    goto L60;
		}
		mu = (d__2 = d__[lll + 1], abs(d__2)) * (mu / (mu + (d__1 = e[
			lll], abs(d__1))));
		sminl = min(sminl,mu);
// L100:
	    }
	}
    } else {
	//
	//       Run convergence test in backward direction
	//       First apply standard test to top of matrix
	//
	if ((d__2 = e[ll], abs(d__2)) <= abs(tol) * (d__1 = d__[ll], abs(d__1)
		) || tol < 0. && (d__3 = e[ll], abs(d__3)) <= thresh) {
	    e[ll] = 0.;
	    goto L60;
	}
	if (tol >= 0.) {
	    //
	    //          If relative accuracy desired,
	    //          apply convergence criterion backward
	    //
	    mu = (d__1 = d__[m], abs(d__1));
	    sminl = mu;
	    i__1 = ll;
	    for (lll = m - 1; lll >= i__1; --lll) {
		if ((d__1 = e[lll], abs(d__1)) <= tol * mu) {
		    e[lll] = 0.;
		    goto L60;
		}
		mu = (d__2 = d__[lll], abs(d__2)) * (mu / (mu + (d__1 = e[lll]
			, abs(d__1))));
		sminl = min(sminl,mu);
// L110:
	    }
	}
    }
    oldll = ll;
    oldm = m;
    //
    //    Compute shift.  First, test if shifting would ruin relative
    //    accuracy, and if so set the shift to zero.
    //
    // Computing MAX
    d__1 = eps, d__2 = tol * .01;
    if (tol >= 0. && *n * tol * (sminl / smax) <= max(d__1,d__2)) {
	//
	//       Use a zero shift to avoid loss of relative accuracy
	//
	shift = 0.;
    } else {
	//
	//       Compute the shift from 2-by-2 block at end of matrix
	//
	if (idir == 1) {
	    sll = (d__1 = d__[ll], abs(d__1));
	    dlas2_(&d__[m - 1], &e[m - 1], &d__[m], &shift, &r__);
	} else {
	    sll = (d__1 = d__[m], abs(d__1));
	    dlas2_(&d__[ll], &e[ll], &d__[ll + 1], &shift, &r__);
	}
	//
	//       Test if shift negligible, and if so set to zero
	//
	if (sll > 0.) {
	    // Computing 2nd power
	    d__1 = shift / sll;
	    if (d__1 * d__1 < eps) {
		shift = 0.;
	    }
	}
    }
    //
    //    Increment iteration count
    //
    iter = iter + m - ll;
    //
    //    If SHIFT = 0, do simplified QR iteration
    //
    if (shift == 0.) {
	if (idir == 1) {
	    //
	    //          Chase bulge from top to bottom
	    //          Save cosines and sines for later singular vector updates
	    //
	    cs = 1.;
	    oldcs = 1.;
	    i__1 = m - 1;
	    for (i__ = ll; i__ <= i__1; ++i__) {
		d__1 = d__[i__] * cs;
		dlartg_(&d__1, &e[i__], &cs, &sn, &r__);
		if (i__ > ll) {
		    e[i__ - 1] = oldsn * r__;
		}
		d__1 = oldcs * r__;
		d__2 = d__[i__ + 1] * sn;
		dlartg_(&d__1, &d__2, &oldcs, &oldsn, &d__[i__]);
		work[i__ - ll + 1] = cs;
		work[i__ - ll + 1 + nm1] = sn;
		work[i__ - ll + 1 + nm12] = oldcs;
		work[i__ - ll + 1 + nm13] = oldsn;
// L120:
	    }
	    h__ = d__[m] * cs;
	    d__[m] = h__ * oldcs;
	    e[m - 1] = h__ * oldsn;
	    //
	    //          Update singular vectors
	    //
	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "F", &i__1, ncvt, &work[1], &work[*n], &vt[
			ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		dlasr_("R", "V", "F", nru, &i__1, &work[nm12 + 1], &work[nm13
			+ 1], &u[ll * u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "F", &i__1, ncc, &work[nm12 + 1], &work[nm13
			+ 1], &c__[ll + c_dim1], ldc);
	    }
	    //
	    //          Test convergence
	    //
	    if ((d__1 = e[m - 1], abs(d__1)) <= thresh) {
		e[m - 1] = 0.;
	    }
	} else {
	    //
	    //          Chase bulge from bottom to top
	    //          Save cosines and sines for later singular vector updates
	    //
	    cs = 1.;
	    oldcs = 1.;
	    i__1 = ll + 1;
	    for (i__ = m; i__ >= i__1; --i__) {
		d__1 = d__[i__] * cs;
		dlartg_(&d__1, &e[i__ - 1], &cs, &sn, &r__);
		if (i__ < m) {
		    e[i__] = oldsn * r__;
		}
		d__1 = oldcs * r__;
		d__2 = d__[i__ - 1] * sn;
		dlartg_(&d__1, &d__2, &oldcs, &oldsn, &d__[i__]);
		work[i__ - ll] = cs;
		work[i__ - ll + nm1] = -sn;
		work[i__ - ll + nm12] = oldcs;
		work[i__ - ll + nm13] = -oldsn;
// L130:
	    }
	    h__ = d__[ll] * cs;
	    d__[ll] = h__ * oldcs;
	    e[ll] = h__ * oldsn;
	    //
	    //          Update singular vectors
	    //
	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "B", &i__1, ncvt, &work[nm12 + 1], &work[
			nm13 + 1], &vt[ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		dlasr_("R", "V", "B", nru, &i__1, &work[1], &work[*n], &u[ll *
			 u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "B", &i__1, ncc, &work[1], &work[*n], &c__[
			ll + c_dim1], ldc);
	    }
	    //
	    //          Test convergence
	    //
	    if ((d__1 = e[ll], abs(d__1)) <= thresh) {
		e[ll] = 0.;
	    }
	}
    } else {
	//
	//       Use nonzero shift
	//
	if (idir == 1) {
	    //
	    //          Chase bulge from top to bottom
	    //          Save cosines and sines for later singular vector updates
	    //
	    f = ((d__1 = d__[ll], abs(d__1)) - shift) * (d_sign(&c_b49, &d__[
		    ll]) + shift / d__[ll]);
	    g = e[ll];
	    i__1 = m - 1;
	    for (i__ = ll; i__ <= i__1; ++i__) {
		dlartg_(&f, &g, &cosr, &sinr, &r__);
		if (i__ > ll) {
		    e[i__ - 1] = r__;
		}
		f = cosr * d__[i__] + sinr * e[i__];
		e[i__] = cosr * e[i__] - sinr * d__[i__];
		g = sinr * d__[i__ + 1];
		d__[i__ + 1] = cosr * d__[i__ + 1];
		dlartg_(&f, &g, &cosl, &sinl, &r__);
		d__[i__] = r__;
		f = cosl * e[i__] + sinl * d__[i__ + 1];
		d__[i__ + 1] = cosl * d__[i__ + 1] - sinl * e[i__];
		if (i__ < m - 1) {
		    g = sinl * e[i__ + 1];
		    e[i__ + 1] = cosl * e[i__ + 1];
		}
		work[i__ - ll + 1] = cosr;
		work[i__ - ll + 1 + nm1] = sinr;
		work[i__ - ll + 1 + nm12] = cosl;
		work[i__ - ll + 1 + nm13] = sinl;
// L140:
	    }
	    e[m - 1] = f;
	    //
	    //          Update singular vectors
	    //
	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "F", &i__1, ncvt, &work[1], &work[*n], &vt[
			ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		dlasr_("R", "V", "F", nru, &i__1, &work[nm12 + 1], &work[nm13
			+ 1], &u[ll * u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "F", &i__1, ncc, &work[nm12 + 1], &work[nm13
			+ 1], &c__[ll + c_dim1], ldc);
	    }
	    //
	    //          Test convergence
	    //
	    if ((d__1 = e[m - 1], abs(d__1)) <= thresh) {
		e[m - 1] = 0.;
	    }
	} else {
	    //
	    //          Chase bulge from bottom to top
	    //          Save cosines and sines for later singular vector updates
	    //
	    f = ((d__1 = d__[m], abs(d__1)) - shift) * (d_sign(&c_b49, &d__[m]
		    ) + shift / d__[m]);
	    g = e[m - 1];
	    i__1 = ll + 1;
	    for (i__ = m; i__ >= i__1; --i__) {
		dlartg_(&f, &g, &cosr, &sinr, &r__);
		if (i__ < m) {
		    e[i__] = r__;
		}
		f = cosr * d__[i__] + sinr * e[i__ - 1];
		e[i__ - 1] = cosr * e[i__ - 1] - sinr * d__[i__];
		g = sinr * d__[i__ - 1];
		d__[i__ - 1] = cosr * d__[i__ - 1];
		dlartg_(&f, &g, &cosl, &sinl, &r__);
		d__[i__] = r__;
		f = cosl * e[i__ - 1] + sinl * d__[i__ - 1];
		d__[i__ - 1] = cosl * d__[i__ - 1] - sinl * e[i__ - 1];
		if (i__ > ll + 1) {
		    g = sinl * e[i__ - 2];
		    e[i__ - 2] = cosl * e[i__ - 2];
		}
		work[i__ - ll] = cosr;
		work[i__ - ll + nm1] = -sinr;
		work[i__ - ll + nm12] = cosl;
		work[i__ - ll + nm13] = -sinl;
// L150:
	    }
	    e[ll] = f;
	    //
	    //          Test convergence
	    //
	    if ((d__1 = e[ll], abs(d__1)) <= thresh) {
		e[ll] = 0.;
	    }
	    //
	    //          Update singular vectors if desired
	    //
	    if (*ncvt > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "B", &i__1, ncvt, &work[nm12 + 1], &work[
			nm13 + 1], &vt[ll + vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		i__1 = m - ll + 1;
		dlasr_("R", "V", "B", nru, &i__1, &work[1], &work[*n], &u[ll *
			 u_dim1 + 1], ldu);
	    }
	    if (*ncc > 0) {
		i__1 = m - ll + 1;
		dlasr_("L", "V", "B", &i__1, ncc, &work[1], &work[*n], &c__[
			ll + c_dim1], ldc);
	    }
	}
    }
    //
    //    QR iteration finished, go back and check convergence
    //
    goto L60;
    //
    //    All singular values converged, so make them positive
    //
L160:
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (d__[i__] < 0.) {
	    d__[i__] = -d__[i__];
	    //
	    //          Change sign of singular vectors, if desired
	    //
	    if (*ncvt > 0) {
		dscal_(ncvt, &c_b72, &vt[i__ + vt_dim1], ldvt);
	    }
	}
// L170:
    }
    //
    //    Sort the singular values into decreasing order (insertion sort on
    //    singular values, but only one transposition per singular vector)
    //
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	//
	//       Scan for smallest D(I)
	//
	isub = 1;
	smin = d__[1];
	i__2 = *n + 1 - i__;
	for (j = 2; j <= i__2; ++j) {
	    if (d__[j] <= smin) {
		isub = j;
		smin = d__[j];
	    }
// L180:
	}
	if (isub != *n + 1 - i__) {
	    //
	    //          Swap singular values and vectors
	    //
	    d__[isub] = d__[*n + 1 - i__];
	    d__[*n + 1 - i__] = smin;
	    if (*ncvt > 0) {
		dswap_(ncvt, &vt[isub + vt_dim1], ldvt, &vt[*n + 1 - i__ +
			vt_dim1], ldvt);
	    }
	    if (*nru > 0) {
		dswap_(nru, &u[isub * u_dim1 + 1], &c__1, &u[(*n + 1 - i__) *
			u_dim1 + 1], &c__1);
	    }
	    if (*ncc > 0) {
		dswap_(ncc, &c__[isub + c_dim1], ldc, &c__[*n + 1 - i__ +
			c_dim1], ldc);
	    }
	}
// L190:
    }
    goto L220;
    //
    //    Maximum number of iterations exceeded, failure to converge
    //
L200:
    *info = 0;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (e[i__] != 0.) {
	    ++(*info);
	}
// L210:
    }
L220:
    return 0;
    //
    //    End of DBDSQR
    //
} // dbdsqr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEBD2 reduces a general matrix to bidiagonal form using an unblocked algorithm.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEBD2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgebd2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgebd2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgebd2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEBD2( M, N, A, LDA, D, E, TAUQ, TAUP, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), D( * ), E( * ), TAUP( * ),
//     $                   TAUQ( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEBD2 reduces a real general m by n matrix A to upper or lower
//> bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//>
//> If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows in the matrix A.  M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns in the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the m by n general matrix to be reduced.
//>          On exit,
//>          if m >= n, the diagonal and the first superdiagonal are
//>            overwritten with the upper bidiagonal matrix B; the
//>            elements below the diagonal, with the array TAUQ, represent
//>            the orthogonal matrix Q as a product of elementary
//>            reflectors, and the elements above the first superdiagonal,
//>            with the array TAUP, represent the orthogonal matrix P as
//>            a product of elementary reflectors;
//>          if m < n, the diagonal and the first subdiagonal are
//>            overwritten with the lower bidiagonal matrix B; the
//>            elements below the first subdiagonal, with the array TAUQ,
//>            represent the orthogonal matrix Q as a product of
//>            elementary reflectors, and the elements above the diagonal,
//>            with the array TAUP, represent the orthogonal matrix P as
//>            a product of elementary reflectors.
//>          See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (min(M,N))
//>          The diagonal elements of the bidiagonal matrix B:
//>          D(i) = A(i,i).
//> \endverbatim
//>
//> \param[out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (min(M,N)-1)
//>          The off-diagonal elements of the bidiagonal matrix B:
//>          if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
//>          if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.
//> \endverbatim
//>
//> \param[out] TAUQ
//> \verbatim
//>          TAUQ is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors which
//>          represent the orthogonal matrix Q. See Further Details.
//> \endverbatim
//>
//> \param[out] TAUP
//> \verbatim
//>          TAUP is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors which
//>          represent the orthogonal matrix P. See Further Details.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (max(M,N))
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit.
//>          < 0: if INFO = -i, the i-th argument had an illegal value.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrices Q and P are represented as products of elementary
//>  reflectors:
//>
//>  If m >= n,
//>
//>     Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
//>
//>  Each H(i) and G(i) has the form:
//>
//>     H(i) = I - tauq * v * v**T  and G(i) = I - taup * u * u**T
//>
//>  where tauq and taup are real scalars, and v and u are real vectors;
//>  v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);
//>  u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);
//>  tauq is stored in TAUQ(i) and taup in TAUP(i).
//>
//>  If m < n,
//>
//>     Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)
//>
//>  Each H(i) and G(i) has the form:
//>
//>     H(i) = I - tauq * v * v**T  and G(i) = I - taup * u * u**T
//>
//>  where tauq and taup are real scalars, and v and u are real vectors;
//>  v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in A(i+2:m,i);
//>  u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in A(i,i+1:n);
//>  tauq is stored in TAUQ(i) and taup in TAUP(i).
//>
//>  The contents of A on exit are illustrated by the following examples:
//>
//>  m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):
//>
//>    (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
//>    (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
//>    (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
//>    (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
//>    (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
//>    (  v1  v2  v3  v4  v5 )
//>
//>  where d and e denote diagonal and off-diagonal elements of B, vi
//>  denotes an element of the vector defining H(i), and ui an element of
//>  the vector defining G(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgebd2_(int *m, int *n, double *a, int *lda, double *d__,
	 double *e, double *tauq, double *taup, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *), dlarfg_(int *, double *,
	    double *, int *, double *), xerbla_(char *, int *);

    //
    // -- LAPACK computational routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    --work;

    // Function Body
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info < 0) {
	i__1 = -(*info);
	xerbla_("DGEBD2", &i__1);
	return 0;
    }
    if (*m >= *n) {
	//
	//       Reduce to upper bidiagonal form
	//
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    //
	    //          Generate elementary reflector H(i) to annihilate A(i+1:m,i)
	    //
	    i__2 = *m - i__ + 1;
	    // Computing MIN
	    i__3 = i__ + 1;
	    dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ *
		    a_dim1], &c__1, &tauq[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.;
	    //
	    //          Apply H(i) to A(i:m,i+1:n) from the left
	    //
	    if (i__ < *n) {
		i__2 = *m - i__ + 1;
		i__3 = *n - i__;
		dlarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &
			tauq[i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]
			);
	    }
	    a[i__ + i__ * a_dim1] = d__[i__];
	    if (i__ < *n) {
		//
		//             Generate elementary reflector G(i) to annihilate
		//             A(i,i+2:n)
		//
		i__2 = *n - i__;
		// Computing MIN
		i__3 = i__ + 2;
		dlarfg_(&i__2, &a[i__ + (i__ + 1) * a_dim1], &a[i__ + min(
			i__3,*n) * a_dim1], lda, &taup[i__]);
		e[i__] = a[i__ + (i__ + 1) * a_dim1];
		a[i__ + (i__ + 1) * a_dim1] = 1.;
		//
		//             Apply G(i) to A(i+1:m,i+1:n) from the right
		//
		i__2 = *m - i__;
		i__3 = *n - i__;
		dlarf_("Right", &i__2, &i__3, &a[i__ + (i__ + 1) * a_dim1],
			lda, &taup[i__], &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &work[1]);
		a[i__ + (i__ + 1) * a_dim1] = e[i__];
	    } else {
		taup[i__] = 0.;
	    }
// L10:
	}
    } else {
	//
	//       Reduce to lower bidiagonal form
	//
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    //
	    //          Generate elementary reflector G(i) to annihilate A(i,i+1:n)
	    //
	    i__2 = *n - i__ + 1;
	    // Computing MIN
	    i__3 = i__ + 1;
	    dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3,*n) *
		    a_dim1], lda, &taup[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.;
	    //
	    //          Apply G(i) to A(i+1:m,i:n) from the right
	    //
	    if (i__ < *m) {
		i__2 = *m - i__;
		i__3 = *n - i__ + 1;
		dlarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &
			taup[i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    }
	    a[i__ + i__ * a_dim1] = d__[i__];
	    if (i__ < *m) {
		//
		//             Generate elementary reflector H(i) to annihilate
		//             A(i+2:m,i)
		//
		i__2 = *m - i__;
		// Computing MIN
		i__3 = i__ + 2;
		dlarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*m) +
			i__ * a_dim1], &c__1, &tauq[i__]);
		e[i__] = a[i__ + 1 + i__ * a_dim1];
		a[i__ + 1 + i__ * a_dim1] = 1.;
		//
		//             Apply H(i) to A(i+1:m,i+1:n) from the left
		//
		i__2 = *m - i__;
		i__3 = *n - i__;
		dlarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &
			c__1, &tauq[i__], &a[i__ + 1 + (i__ + 1) * a_dim1],
			lda, &work[1]);
		a[i__ + 1 + i__ * a_dim1] = e[i__];
	    } else {
		tauq[i__] = 0.;
	    }
// L20:
	}
    }
    return 0;
    //
    //    End of DGEBD2
    //
} // dgebd2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEBRD
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEBRD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgebrd.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgebrd.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgebrd.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEBRD( M, N, A, LDA, D, E, TAUQ, TAUP, WORK, LWORK,
//                         INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), D( * ), E( * ), TAUP( * ),
//     $                   TAUQ( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEBRD reduces a general real M-by-N matrix A to upper or lower
//> bidiagonal form B by an orthogonal transformation: Q**T * A * P = B.
//>
//> If m >= n, B is upper bidiagonal; if m < n, B is lower bidiagonal.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows in the matrix A.  M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns in the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the M-by-N general matrix to be reduced.
//>          On exit,
//>          if m >= n, the diagonal and the first superdiagonal are
//>            overwritten with the upper bidiagonal matrix B; the
//>            elements below the diagonal, with the array TAUQ, represent
//>            the orthogonal matrix Q as a product of elementary
//>            reflectors, and the elements above the first superdiagonal,
//>            with the array TAUP, represent the orthogonal matrix P as
//>            a product of elementary reflectors;
//>          if m < n, the diagonal and the first subdiagonal are
//>            overwritten with the lower bidiagonal matrix B; the
//>            elements below the first subdiagonal, with the array TAUQ,
//>            represent the orthogonal matrix Q as a product of
//>            elementary reflectors, and the elements above the diagonal,
//>            with the array TAUP, represent the orthogonal matrix P as
//>            a product of elementary reflectors.
//>          See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (min(M,N))
//>          The diagonal elements of the bidiagonal matrix B:
//>          D(i) = A(i,i).
//> \endverbatim
//>
//> \param[out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (min(M,N)-1)
//>          The off-diagonal elements of the bidiagonal matrix B:
//>          if m >= n, E(i) = A(i,i+1) for i = 1,2,...,n-1;
//>          if m < n, E(i) = A(i+1,i) for i = 1,2,...,m-1.
//> \endverbatim
//>
//> \param[out] TAUQ
//> \verbatim
//>          TAUQ is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors which
//>          represent the orthogonal matrix Q. See Further Details.
//> \endverbatim
//>
//> \param[out] TAUP
//> \verbatim
//>          TAUP is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors which
//>          represent the orthogonal matrix P. See Further Details.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The length of the array WORK.  LWORK >= max(1,M,N).
//>          For optimum performance LWORK >= (M+N)*NB, where NB
//>          is the optimal blocksize.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date November 2017
//
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrices Q and P are represented as products of elementary
//>  reflectors:
//>
//>  If m >= n,
//>
//>     Q = H(1) H(2) . . . H(n)  and  P = G(1) G(2) . . . G(n-1)
//>
//>  Each H(i) and G(i) has the form:
//>
//>     H(i) = I - tauq * v * v**T  and G(i) = I - taup * u * u**T
//>
//>  where tauq and taup are real scalars, and v and u are real vectors;
//>  v(1:i-1) = 0, v(i) = 1, and v(i+1:m) is stored on exit in A(i+1:m,i);
//>  u(1:i) = 0, u(i+1) = 1, and u(i+2:n) is stored on exit in A(i,i+2:n);
//>  tauq is stored in TAUQ(i) and taup in TAUP(i).
//>
//>  If m < n,
//>
//>     Q = H(1) H(2) . . . H(m-1)  and  P = G(1) G(2) . . . G(m)
//>
//>  Each H(i) and G(i) has the form:
//>
//>     H(i) = I - tauq * v * v**T  and G(i) = I - taup * u * u**T
//>
//>  where tauq and taup are real scalars, and v and u are real vectors;
//>  v(1:i) = 0, v(i+1) = 1, and v(i+2:m) is stored on exit in A(i+2:m,i);
//>  u(1:i-1) = 0, u(i) = 1, and u(i+1:n) is stored on exit in A(i,i+1:n);
//>  tauq is stored in TAUQ(i) and taup in TAUP(i).
//>
//>  The contents of A on exit are illustrated by the following examples:
//>
//>  m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):
//>
//>    (  d   e   u1  u1  u1 )           (  d   u1  u1  u1  u1  u1 )
//>    (  v1  d   e   u2  u2 )           (  e   d   u2  u2  u2  u2 )
//>    (  v1  v2  d   e   u3 )           (  v1  e   d   u3  u3  u3 )
//>    (  v1  v2  v3  d   e  )           (  v1  v2  e   d   u4  u4 )
//>    (  v1  v2  v3  v4  d  )           (  v1  v2  v3  e   d   u5 )
//>    (  v1  v2  v3  v4  v5 )
//>
//>  where d and e denote diagonal and off-diagonal elements of B, vi
//>  denotes an element of the vector defining H(i), and ui an element of
//>  the vector defining G(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgebrd_(int *m, int *n, double *a, int *lda, double *d__,
	 double *e, double *tauq, double *taup, double *work, int *lwork, int
	*info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;
    double c_b21 = -1.;
    double c_b22 = 1.;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    // Local variables
    int i__, j, nb, nx, ws;
    extern /* Subroutine */ int dgemm_(char *, char *, int *, int *, int *,
	    double *, double *, int *, double *, int *, double *, double *,
	    int *);
    int nbmin, iinfo, minmn;
    extern /* Subroutine */ int dgebd2_(int *, int *, double *, int *, double
	    *, double *, double *, double *, double *, int *), dlabrd_(int *,
	    int *, int *, double *, int *, double *, double *, double *,
	    double *, double *, int *, double *, int *), xerbla_(char *, int *
	    );
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int ldwrkx, ldwrky, lwkopt;
    int lquery;

    //
    // -- LAPACK computational routine (version 3.8.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    --work;

    // Function Body
    *info = 0;
    // Computing MAX
    i__1 = 1, i__2 = ilaenv_(&c__1, "DGEBRD", " ", m, n, &c_n1, &c_n1);
    nb = max(i__1,i__2);
    lwkopt = (*m + *n) * nb;
    work[1] = (double) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else /* if(complicated condition) */ {
	// Computing MAX
	i__1 = max(1,*m);
	if (*lwork < max(i__1,*n) && ! lquery) {
	    *info = -10;
	}
    }
    if (*info < 0) {
	i__1 = -(*info);
	xerbla_("DGEBRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    minmn = min(*m,*n);
    if (minmn == 0) {
	work[1] = 1.;
	return 0;
    }
    ws = max(*m,*n);
    ldwrkx = *m;
    ldwrky = *n;
    if (nb > 1 && nb < minmn) {
	//
	//       Set the crossover point NX.
	//
	// Computing MAX
	i__1 = nb, i__2 = ilaenv_(&c__3, "DGEBRD", " ", m, n, &c_n1, &c_n1);
	nx = max(i__1,i__2);
	//
	//       Determine when to switch from blocked to unblocked code.
	//
	if (nx < minmn) {
	    ws = (*m + *n) * nb;
	    if (*lwork < ws) {
		//
		//             Not enough work space for the optimal NB, consider using
		//             a smaller block size.
		//
		nbmin = ilaenv_(&c__2, "DGEBRD", " ", m, n, &c_n1, &c_n1);
		if (*lwork >= (*m + *n) * nbmin) {
		    nb = *lwork / (*m + *n);
		} else {
		    nb = 1;
		    nx = minmn;
		}
	    }
	}
    } else {
	nx = minmn;
    }
    i__1 = minmn - nx;
    i__2 = nb;
    for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	//
	//       Reduce rows and columns i:i+nb-1 to bidiagonal form and return
	//       the matrices X and Y which are needed to update the unreduced
	//       part of the matrix
	//
	i__3 = *m - i__ + 1;
	i__4 = *n - i__ + 1;
	dlabrd_(&i__3, &i__4, &nb, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[
		i__], &tauq[i__], &taup[i__], &work[1], &ldwrkx, &work[ldwrkx
		* nb + 1], &ldwrky);
	//
	//       Update the trailing submatrix A(i+nb:m,i+nb:n), using an update
	//       of the form  A := A - V*Y**T - X*U**T
	//
	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	dgemm_("No transpose", "Transpose", &i__3, &i__4, &nb, &c_b21, &a[i__
		+ nb + i__ * a_dim1], lda, &work[ldwrkx * nb + nb + 1], &
		ldwrky, &c_b22, &a[i__ + nb + (i__ + nb) * a_dim1], lda);
	i__3 = *m - i__ - nb + 1;
	i__4 = *n - i__ - nb + 1;
	dgemm_("No transpose", "No transpose", &i__3, &i__4, &nb, &c_b21, &
		work[nb + 1], &ldwrkx, &a[i__ + (i__ + nb) * a_dim1], lda, &
		c_b22, &a[i__ + nb + (i__ + nb) * a_dim1], lda);
	//
	//       Copy diagonal and off-diagonal elements of B back into A
	//
	if (*m >= *n) {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j + j * a_dim1] = d__[j];
		a[j + (j + 1) * a_dim1] = e[j];
// L10:
	    }
	} else {
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j + j * a_dim1] = d__[j];
		a[j + 1 + j * a_dim1] = e[j];
// L20:
	    }
	}
// L30:
    }
    //
    //    Use unblocked code to reduce the remainder of the matrix
    //
    i__2 = *m - i__ + 1;
    i__1 = *n - i__ + 1;
    dgebd2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__], &
	    tauq[i__], &taup[i__], &work[1], &iinfo);
    work[1] = (double) ws;
    return 0;
    //
    //    End of DGEBRD
    //
} // dgebrd_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGELQ2 computes the LQ factorization of a general rectangular matrix using an unblocked algorithm.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGELQ2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgelq2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgelq2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgelq2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGELQ2( M, N, A, LDA, TAU, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGELQ2 computes an LQ factorization of a real m-by-n matrix A:
//>
//>    A = ( L 0 ) *  Q
//>
//> where:
//>
//>    Q is a n-by-n orthogonal matrix;
//>    L is an lower-triangular m-by-m matrix;
//>    0 is a m-by-(n-m) zero matrix, if m < n.
//>
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.  M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the m by n matrix A.
//>          On exit, the elements on and below the diagonal of the array
//>          contain the m by min(m,n) lower trapezoidal matrix L (L is
//>          lower triangular if m <= n); the elements above the diagonal,
//>          with the array TAU, represent the orthogonal matrix Q as a
//>          product of elementary reflectors (see Further Details).
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (M)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit
//>          < 0: if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date November 2019
//
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrix Q is represented as a product of elementary reflectors
//>
//>     Q = H(k) . . . H(2) H(1), where k = min(m,n).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),
//>  and tau in TAU(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgelq2_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *info)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, k;
    double aii;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *), dlarfg_(int *, double *,
	    double *, int *, double *), xerbla_(char *, int *);

    //
    // -- LAPACK computational routine (version 3.9.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2019
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGELQ2", &i__1);
	return 0;
    }
    k = min(*m,*n);
    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	//
	//       Generate elementary reflector H(i) to annihilate A(i,i+1:n)
	//
	i__2 = *n - i__ + 1;
	// Computing MIN
	i__3 = i__ + 1;
	dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3,*n) * a_dim1]
		, lda, &tau[i__]);
	if (i__ < *m) {
	    //
	    //          Apply H(i) to A(i+1:m,i:n) from the right
	    //
	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.;
	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
	    dlarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[
		    i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
// L10:
    }
    return 0;
    //
    //    End of DGELQ2
    //
} // dgelq2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGELQF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGELQF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgelqf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgelqf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgelqf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGELQF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGELQF computes an LQ factorization of a real M-by-N matrix A:
//>
//>    A = ( L 0 ) *  Q
//>
//> where:
//>
//>    Q is a N-by-N orthogonal matrix;
//>    L is an lower-triangular M-by-M matrix;
//>    0 is a M-by-(N-M) zero matrix, if M < N.
//>
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.  M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the M-by-N matrix A.
//>          On exit, the elements on and below the diagonal of the array
//>          contain the m-by-min(m,n) lower trapezoidal matrix L (L is
//>          lower triangular if m <= n); the elements above the diagonal,
//>          with the array TAU, represent the orthogonal matrix Q as a
//>          product of elementary reflectors (see Further Details).
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK.  LWORK >= max(1,M).
//>          For optimum performance LWORK >= M*NB, where NB is the
//>          optimal blocksize.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date November 2019
//
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrix Q is represented as a product of elementary reflectors
//>
//>     Q = H(k) . . . H(2) H(1), where k = min(m,n).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i-1) = 0 and v(i) = 1; v(i+1:n) is stored on exit in A(i,i+1:n),
//>  and tau in TAU(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgelqf_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    // Local variables
    int i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int dgelq2_(int *, int *, double *, int *, double
	    *, double *, int *), dlarfb_(char *, char *, char *, char *, int *
	    , int *, int *, double *, int *, double *, int *, double *, int *,
	     double *, int *), dlarft_(char *, char *, int *, int *, double *,
	     int *, double *, double *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int ldwork, lwkopt;
    int lquery;

    //
    // -- LAPACK computational routine (version 3.9.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2019
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    nb = ilaenv_(&c__1, "DGELQF", " ", m, n, &c_n1, &c_n1);
    lwkopt = *m * nb;
    work[1] = (double) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGELQF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    k = min(*m,*n);
    if (k == 0) {
	work[1] = 1.;
	return 0;
    }
    nbmin = 2;
    nx = 0;
    iws = *m;
    if (nb > 1 && nb < k) {
	//
	//       Determine when to cross over from blocked to unblocked code.
	//
	// Computing MAX
	i__1 = 0, i__2 = ilaenv_(&c__3, "DGELQF", " ", m, n, &c_n1, &c_n1);
	nx = max(i__1,i__2);
	if (nx < k) {
	    //
	    //          Determine if workspace is large enough for blocked code.
	    //
	    ldwork = *m;
	    iws = ldwork * nb;
	    if (*lwork < iws) {
		//
		//             Not enough workspace to use optimal NB:  reduce NB and
		//             determine the minimum value of NB.
		//
		nb = *lwork / ldwork;
		// Computing MAX
		i__1 = 2, i__2 = ilaenv_(&c__2, "DGELQF", " ", m, n, &c_n1, &
			c_n1);
		nbmin = max(i__1,i__2);
	    }
	}
    }
    if (nb >= nbmin && nb < k && nx < k) {
	//
	//       Use blocked code initially
	//
	i__1 = k - nx;
	i__2 = nb;
	for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	    // Computing MIN
	    i__3 = k - i__ + 1;
	    ib = min(i__3,nb);
	    //
	    //          Compute the LQ factorization of the current block
	    //          A(i:i+ib-1,i:n)
	    //
	    i__3 = *n - i__ + 1;
	    dgelq2_(&ib, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *m) {
		//
		//             Form the triangular factor of the block reflector
		//             H = H(i) H(i+1) . . . H(i+ib-1)
		//
		i__3 = *n - i__ + 1;
		dlarft_("Forward", "Rowwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);
		//
		//             Apply H to A(i+ib:m,i:n) from the right
		//
		i__3 = *m - i__ - ib + 1;
		i__4 = *n - i__ + 1;
		dlarfb_("Right", "No transpose", "Forward", "Rowwise", &i__3,
			&i__4, &ib, &a[i__ + i__ * a_dim1], lda, &work[1], &
			ldwork, &a[i__ + ib + i__ * a_dim1], lda, &work[ib +
			1], &ldwork);
	    }
// L10:
	}
    } else {
	i__ = 1;
    }
    //
    //    Use unblocked code to factor the last or only block.
    //
    if (i__ <= k) {
	i__2 = *m - i__ + 1;
	i__1 = *n - i__ + 1;
	dgelq2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }
    work[1] = (double) iws;
    return 0;
    //
    //    End of DGELQF
    //
} // dgelqf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEQR2 computes the QR factorization of a general rectangular matrix using an unblocked algorithm.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEQR2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgeqr2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgeqr2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgeqr2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEQR2( M, N, A, LDA, TAU, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEQR2 computes a QR factorization of a real m-by-n matrix A:
//>
//>    A = Q * ( R ),
//>            ( 0 )
//>
//> where:
//>
//>    Q is a m-by-m orthogonal matrix;
//>    R is an upper-triangular n-by-n matrix;
//>    0 is a (m-n)-by-n zero matrix, if m > n.
//>
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.  M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the m by n matrix A.
//>          On exit, the elements on and above the diagonal of the array
//>          contain the min(m,n) by n upper trapezoidal matrix R (R is
//>          upper triangular if m >= n); the elements below the diagonal,
//>          with the array TAU, represent the orthogonal matrix Q as a
//>          product of elementary reflectors (see Further Details).
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit
//>          < 0: if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date November 2019
//
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrix Q is represented as a product of elementary reflectors
//>
//>     Q = H(1) H(2) . . . H(k), where k = min(m,n).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
//>  and tau in TAU(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgeqr2_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, k;
    double aii;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *), dlarfg_(int *, double *,
	    double *, int *, double *), xerbla_(char *, int *);

    //
    // -- LAPACK computational routine (version 3.9.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2019
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEQR2", &i__1);
	return 0;
    }
    k = min(*m,*n);
    i__1 = k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	//
	//       Generate elementary reflector H(i) to annihilate A(i+1:m,i)
	//
	i__2 = *m - i__ + 1;
	// Computing MIN
	i__3 = i__ + 1;
	dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ * a_dim1]
		, &c__1, &tau[i__]);
	if (i__ < *n) {
	    //
	    //          Apply H(i) to A(i:m,i+1:n) from the left
	    //
	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.;
	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    dlarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
// L10:
    }
    return 0;
    //
    //    End of DGEQR2
    //
} // dgeqr2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEQRF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEQRF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgeqrf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgeqrf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgeqrf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEQRF computes a QR factorization of a real M-by-N matrix A:
//>
//>    A = Q * ( R ),
//>            ( 0 )
//>
//> where:
//>
//>    Q is a M-by-M orthogonal matrix;
//>    R is an upper-triangular N-by-N matrix;
//>    0 is a (M-N)-by-N zero matrix, if M > N.
//>
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.  M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the M-by-N matrix A.
//>          On exit, the elements on and above the diagonal of the array
//>          contain the min(M,N)-by-N upper trapezoidal matrix R (R is
//>          upper triangular if m >= n); the elements below the diagonal,
//>          with the array TAU, represent the orthogonal matrix Q as a
//>          product of min(m,n) elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK.  LWORK >= max(1,N).
//>          For optimum performance LWORK >= N*NB, where NB is
//>          the optimal blocksize.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date November 2019
//
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrix Q is represented as a product of elementary reflectors
//>
//>     Q = H(1) H(2) . . . H(k), where k = min(m,n).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
//>  and tau in TAU(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgeqrf_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    // Local variables
    int i__, k, ib, nb, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int dgeqr2_(int *, int *, double *, int *, double
	    *, double *, int *), dlarfb_(char *, char *, char *, char *, int *
	    , int *, int *, double *, int *, double *, int *, double *, int *,
	     double *, int *), dlarft_(char *, char *, int *, int *, double *,
	     int *, double *, double *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int ldwork, lwkopt;
    int lquery;

    //
    // -- LAPACK computational routine (version 3.9.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2019
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    nb = ilaenv_(&c__1, "DGEQRF", " ", m, n, &c_n1, &c_n1);
    lwkopt = *n * nb;
    work[1] = (double) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*m)) {
	*info = -4;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEQRF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    k = min(*m,*n);
    if (k == 0) {
	work[1] = 1.;
	return 0;
    }
    nbmin = 2;
    nx = 0;
    iws = *n;
    if (nb > 1 && nb < k) {
	//
	//       Determine when to cross over from blocked to unblocked code.
	//
	// Computing MAX
	i__1 = 0, i__2 = ilaenv_(&c__3, "DGEQRF", " ", m, n, &c_n1, &c_n1);
	nx = max(i__1,i__2);
	if (nx < k) {
	    //
	    //          Determine if workspace is large enough for blocked code.
	    //
	    ldwork = *n;
	    iws = ldwork * nb;
	    if (*lwork < iws) {
		//
		//             Not enough workspace to use optimal NB:  reduce NB and
		//             determine the minimum value of NB.
		//
		nb = *lwork / ldwork;
		// Computing MAX
		i__1 = 2, i__2 = ilaenv_(&c__2, "DGEQRF", " ", m, n, &c_n1, &
			c_n1);
		nbmin = max(i__1,i__2);
	    }
	}
    }
    if (nb >= nbmin && nb < k && nx < k) {
	//
	//       Use blocked code initially
	//
	i__1 = k - nx;
	i__2 = nb;
	for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	    // Computing MIN
	    i__3 = k - i__ + 1;
	    ib = min(i__3,nb);
	    //
	    //          Compute the QR factorization of the current block
	    //          A(i:m,i:i+ib-1)
	    //
	    i__3 = *m - i__ + 1;
	    dgeqr2_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *n) {
		//
		//             Form the triangular factor of the block reflector
		//             H = H(i) H(i+1) . . . H(i+ib-1)
		//
		i__3 = *m - i__ + 1;
		dlarft_("Forward", "Columnwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);
		//
		//             Apply H**T to A(i:m,i+ib:n) from the left
		//
		i__3 = *m - i__ + 1;
		i__4 = *n - i__ - ib + 1;
		dlarfb_("Left", "Transpose", "Forward", "Columnwise", &i__3, &
			i__4, &ib, &a[i__ + i__ * a_dim1], lda, &work[1], &
			ldwork, &a[i__ + (i__ + ib) * a_dim1], lda, &work[ib
			+ 1], &ldwork);
	    }
// L10:
	}
    } else {
	i__ = 1;
    }
    //
    //    Use unblocked code to factor the last or only block.
    //
    if (i__ <= k) {
	i__2 = *m - i__ + 1;
	i__1 = *n - i__ + 1;
	dgeqr2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }
    work[1] = (double) iws;
    return 0;
    //
    //    End of DGEQRF
    //
} // dgeqrf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGESDD
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGESDD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgesdd.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgesdd.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgesdd.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT,
//                         WORK, LWORK, IWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          JOBZ
//      INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      INTEGER            IWORK( * )
//      DOUBLE PRECISION   A( LDA, * ), S( * ), U( LDU, * ),
//     $                   VT( LDVT, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGESDD computes the singular value decomposition (SVD) of a real
//> M-by-N matrix A, optionally computing the left and right singular
//> vectors.  If singular vectors are desired, it uses a
//> divide-and-conquer algorithm.
//>
//> The SVD is written
//>
//>      A = U * SIGMA * transpose(V)
//>
//> where SIGMA is an M-by-N matrix which is zero except for its
//> min(m,n) diagonal elements, U is an M-by-M orthogonal matrix, and
//> V is an N-by-N orthogonal matrix.  The diagonal elements of SIGMA
//> are the singular values of A; they are real and non-negative, and
//> are returned in descending order.  The first min(m,n) columns of
//> U and V are the left and right singular vectors of A.
//>
//> Note that the routine returns VT = V**T, not V.
//>
//> The divide and conquer algorithm makes very mild assumptions about
//> floating point arithmetic. It will work on machines with a guard
//> digit in add/subtract, or on those binary machines without guard
//> digits which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or
//> Cray-2. It could conceivably fail on hexadecimal or decimal machines
//> without guard digits, but we know of none.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOBZ
//> \verbatim
//>          JOBZ is CHARACTER*1
//>          Specifies options for computing all or part of the matrix U:
//>          = 'A':  all M columns of U and all N rows of V**T are
//>                  returned in the arrays U and VT;
//>          = 'S':  the first min(M,N) columns of U and the first
//>                  min(M,N) rows of V**T are returned in the arrays U
//>                  and VT;
//>          = 'O':  If M >= N, the first N columns of U are overwritten
//>                  on the array A and all rows of V**T are returned in
//>                  the array VT;
//>                  otherwise, all columns of U are returned in the
//>                  array U and the first M rows of V**T are overwritten
//>                  in the array A;
//>          = 'N':  no columns of U or rows of V**T are computed.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the input matrix A.  M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the input matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the M-by-N matrix A.
//>          On exit,
//>          if JOBZ = 'O',  A is overwritten with the first N columns
//>                          of U (the left singular vectors, stored
//>                          columnwise) if M >= N;
//>                          A is overwritten with the first M rows
//>                          of V**T (the right singular vectors, stored
//>                          rowwise) otherwise.
//>          if JOBZ .ne. 'O', the contents of A are destroyed.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] S
//> \verbatim
//>          S is DOUBLE PRECISION array, dimension (min(M,N))
//>          The singular values of A, sorted so that S(i) >= S(i+1).
//> \endverbatim
//>
//> \param[out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension (LDU,UCOL)
//>          UCOL = M if JOBZ = 'A' or JOBZ = 'O' and M < N;
//>          UCOL = min(M,N) if JOBZ = 'S'.
//>          If JOBZ = 'A' or JOBZ = 'O' and M < N, U contains the M-by-M
//>          orthogonal matrix U;
//>          if JOBZ = 'S', U contains the first min(M,N) columns of U
//>          (the left singular vectors, stored columnwise);
//>          if JOBZ = 'O' and M >= N, or JOBZ = 'N', U is not referenced.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>          The leading dimension of the array U.  LDU >= 1; if
//>          JOBZ = 'S' or 'A' or JOBZ = 'O' and M < N, LDU >= M.
//> \endverbatim
//>
//> \param[out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension (LDVT,N)
//>          If JOBZ = 'A' or JOBZ = 'O' and M >= N, VT contains the
//>          N-by-N orthogonal matrix V**T;
//>          if JOBZ = 'S', VT contains the first min(M,N) rows of
//>          V**T (the right singular vectors, stored rowwise);
//>          if JOBZ = 'O' and M < N, or JOBZ = 'N', VT is not referenced.
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>          The leading dimension of the array VT.  LDVT >= 1;
//>          if JOBZ = 'A' or JOBZ = 'O' and M >= N, LDVT >= N;
//>          if JOBZ = 'S', LDVT >= min(M,N).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK;
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK. LWORK >= 1.
//>          If LWORK = -1, a workspace query is assumed.  The optimal
//>          size for the WORK array is calculated and stored in WORK(1),
//>          and no other work except argument checking is performed.
//>
//>          Let mx = max(M,N) and mn = min(M,N).
//>          If JOBZ = 'N', LWORK >= 3*mn + max( mx, 7*mn ).
//>          If JOBZ = 'O', LWORK >= 3*mn + max( mx, 5*mn*mn + 4*mn ).
//>          If JOBZ = 'S', LWORK >= 4*mn*mn + 7*mn.
//>          If JOBZ = 'A', LWORK >= 4*mn*mn + 6*mn + mx.
//>          These are not tight minimums in all cases; see comments inside code.
//>          For good performance, LWORK should generally be larger;
//>          a query is recommended.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (8*min(M,N))
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  DBDSDC did not converge, updating process failed.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup doubleGEsing
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dgesdd_(char *jobz, int *m, int *n, double *a, int *lda,
	double *s, double *u, int *ldu, double *vt, int *ldvt, double *work,
	int *lwork, int *iwork, int *info)
{
    // Table of constant values
    int c_n1 = -1;
    int c__0 = 0;
    double c_b63 = 0.;
    int c__1 = 1;
    double c_b84 = 1.;

    // System generated locals
    int a_dim1, a_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2,
	    i__3;

    // Local variables
    int lwork_dorglq_mn__, lwork_dorglq_nn__, lwork_dorgqr_mm__,
	    lwork_dorgqr_mn__, i__, ie, lwork_dorgbr_p_mm__, il,
	    lwork_dorgbr_q_nn__, ir, iu, blk;
    double dum[1], eps;
    int ivt, iscl;
    double anrm;
    int idum[1], ierr, itau, lwork_dormbr_qln_mm__, lwork_dormbr_qln_mn__,
	    lwork_dormbr_qln_nn__, lwork_dormbr_prt_mm__,
	    lwork_dormbr_prt_mn__, lwork_dormbr_prt_nn__;
    extern /* Subroutine */ int dgemm_(char *, char *, int *, int *, int *,
	    double *, double *, int *, double *, int *, double *, double *,
	    int *);
    extern int lsame_(char *, char *);
    int chunk, minmn, wrkbl, itaup, itauq, mnthr;
    int wntqa;
    int nwork;
    int wntqn, wntqo, wntqs;
    extern /* Subroutine */ int dbdsdc_(char *, char *, int *, double *,
	    double *, double *, int *, double *, int *, double *, int *,
	    double *, int *, int *), dgebrd_(int *, int *, double *, int *,
	    double *, double *, double *, double *, double *, int *, int *);
    extern double dlamch_(char *), dlange_(char *, int *, int *, double *,
	    int *, double *);
    int bdspac;
    extern /* Subroutine */ int dgelqf_(int *, int *, double *, int *, double
	    *, double *, int *, int *), dlascl_(char *, int *, int *, double *
	    , double *, int *, int *, double *, int *, int *), dgeqrf_(int *,
	    int *, double *, int *, double *, double *, int *, int *),
	    dlacpy_(char *, int *, int *, double *, int *, double *, int *),
	    dlaset_(char *, int *, int *, double *, double *, double *, int *)
	    , xerbla_(char *, int *), dorgbr_(char *, int *, int *, int *,
	    double *, int *, double *, double *, int *, int *);
    double bignum;
    extern /* Subroutine */ int dormbr_(char *, char *, char *, int *, int *,
	    int *, double *, int *, double *, double *, int *, double *, int *
	    , int *), dorglq_(int *, int *, int *, double *, int *, double *,
	    double *, int *, int *), dorgqr_(int *, int *, int *, double *,
	    int *, double *, double *, int *, int *);
    int ldwrkl, ldwrkr, minwrk, ldwrku, maxwrk, ldwkvt;
    double smlnum;
    int wntqas, lquery;
    int lwork_dgebrd_mm__, lwork_dgebrd_mn__, lwork_dgebrd_nn__,
	    lwork_dgelqf_mn__, lwork_dgeqrf_mn__;

    //
    // -- LAPACK driver routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --s;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --work;
    --iwork;

    // Function Body
    *info = 0;
    minmn = min(*m,*n);
    wntqa = lsame_(jobz, "A");
    wntqs = lsame_(jobz, "S");
    wntqas = wntqa || wntqs;
    wntqo = lsame_(jobz, "O");
    wntqn = lsame_(jobz, "N");
    lquery = *lwork == -1;
    if (! (wntqa || wntqs || wntqo || wntqn)) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*ldu < 1 || wntqas && *ldu < *m || wntqo && *m < *n && *ldu < *
	    m) {
	*info = -8;
    } else if (*ldvt < 1 || wntqa && *ldvt < *n || wntqs && *ldvt < minmn ||
	    wntqo && *m >= *n && *ldvt < *n) {
	*info = -10;
    }
    //
    //    Compute workspace
    //      Note: Comments in the code beginning "Workspace:" describe the
    //      minimal amount of workspace allocated at that point in the code,
    //      as well as the preferred amount for good performance.
    //      NB refers to the optimal block size for the immediately
    //      following subroutine, as returned by ILAENV.
    //
    if (*info == 0) {
	minwrk = 1;
	maxwrk = 1;
	bdspac = 0;
	mnthr = (int) (minmn * 11. / 6.);
	if (*m >= *n && minmn > 0) {
	    //
	    //          Compute space needed for DBDSDC
	    //
	    if (wntqn) {
		//             dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
		//             keep 7*N for backwards compatibility.
		bdspac = *n * 7;
	    } else {
		bdspac = *n * 3 * *n + (*n << 2);
	    }
	    //
	    //          Compute space preferred for each routine
	    dgebrd_(m, n, dum, m, dum, dum, dum, dum, dum, &c_n1, &ierr);
	    lwork_dgebrd_mn__ = (int) dum[0];
	    dgebrd_(n, n, dum, n, dum, dum, dum, dum, dum, &c_n1, &ierr);
	    lwork_dgebrd_nn__ = (int) dum[0];
	    dgeqrf_(m, n, dum, m, dum, dum, &c_n1, &ierr);
	    lwork_dgeqrf_mn__ = (int) dum[0];
	    dorgbr_("Q", n, n, n, dum, n, dum, dum, &c_n1, &ierr);
	    lwork_dorgbr_q_nn__ = (int) dum[0];
	    dorgqr_(m, m, n, dum, m, dum, dum, &c_n1, &ierr);
	    lwork_dorgqr_mm__ = (int) dum[0];
	    dorgqr_(m, n, n, dum, m, dum, dum, &c_n1, &ierr);
	    lwork_dorgqr_mn__ = (int) dum[0];
	    dormbr_("P", "R", "T", n, n, n, dum, n, dum, dum, n, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_prt_nn__ = (int) dum[0];
	    dormbr_("Q", "L", "N", n, n, n, dum, n, dum, dum, n, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_qln_nn__ = (int) dum[0];
	    dormbr_("Q", "L", "N", m, n, n, dum, m, dum, dum, m, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_qln_mn__ = (int) dum[0];
	    dormbr_("Q", "L", "N", m, m, n, dum, m, dum, dum, m, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_qln_mm__ = (int) dum[0];
	    if (*m >= mnthr) {
		if (wntqn) {
		    //
		    //                Path 1 (M >> N, JOBZ='N')
		    //
		    wrkbl = *n + lwork_dgeqrf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dgebrd_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = bdspac + *n;
		    maxwrk = max(i__1,i__2);
		    minwrk = bdspac + *n;
		} else if (wntqo) {
		    //
		    //                Path 2 (M >> N, JOBZ='O')
		    //
		    wrkbl = *n + lwork_dgeqrf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n + lwork_dorgqr_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dgebrd_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_qln_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_prt_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + (*n << 1) * *n;
		    minwrk = bdspac + (*n << 1) * *n + *n * 3;
		} else if (wntqs) {
		    //
		    //                Path 3 (M >> N, JOBZ='S')
		    //
		    wrkbl = *n + lwork_dgeqrf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n + lwork_dorgqr_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dgebrd_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_qln_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_prt_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *n * *n;
		    minwrk = bdspac + *n * *n + *n * 3;
		} else if (wntqa) {
		    //
		    //                Path 4 (M >> N, JOBZ='A')
		    //
		    wrkbl = *n + lwork_dgeqrf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n + lwork_dorgqr_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dgebrd_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_qln_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_prt_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *n * *n;
		    // Computing MAX
		    i__1 = *n * 3 + bdspac, i__2 = *n + *m;
		    minwrk = *n * *n + max(i__1,i__2);
		}
	    } else {
		//
		//             Path 5 (M >= N, but not much larger)
		//
		wrkbl = *n * 3 + lwork_dgebrd_mn__;
		if (wntqn) {
		    //                Path 5n (M >= N, jobz='N')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + bdspac;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		} else if (wntqo) {
		    //                Path 5o (M >= N, jobz='O')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_prt_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_qln_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *n;
		    // Computing MAX
		    i__1 = *m, i__2 = *n * *n + bdspac;
		    minwrk = *n * 3 + max(i__1,i__2);
		} else if (wntqs) {
		    //                Path 5s (M >= N, jobz='S')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_qln_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_prt_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + bdspac;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		} else if (wntqa) {
		    //                Path 5a (M >= N, jobz='A')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_qln_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + lwork_dormbr_prt_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *n * 3 + bdspac;
		    maxwrk = max(i__1,i__2);
		    minwrk = *n * 3 + max(*m,bdspac);
		}
	    }
	} else if (minmn > 0) {
	    //
	    //          Compute space needed for DBDSDC
	    //
	    if (wntqn) {
		//             dbdsdc needs only 4*N (or 6*N for uplo=L for LAPACK <= 3.6)
		//             keep 7*N for backwards compatibility.
		bdspac = *m * 7;
	    } else {
		bdspac = *m * 3 * *m + (*m << 2);
	    }
	    //
	    //          Compute space preferred for each routine
	    dgebrd_(m, n, dum, m, dum, dum, dum, dum, dum, &c_n1, &ierr);
	    lwork_dgebrd_mn__ = (int) dum[0];
	    dgebrd_(m, m, &a[a_offset], m, &s[1], dum, dum, dum, dum, &c_n1, &
		    ierr);
	    lwork_dgebrd_mm__ = (int) dum[0];
	    dgelqf_(m, n, &a[a_offset], m, dum, dum, &c_n1, &ierr);
	    lwork_dgelqf_mn__ = (int) dum[0];
	    dorglq_(n, n, m, dum, n, dum, dum, &c_n1, &ierr);
	    lwork_dorglq_nn__ = (int) dum[0];
	    dorglq_(m, n, m, &a[a_offset], m, dum, dum, &c_n1, &ierr);
	    lwork_dorglq_mn__ = (int) dum[0];
	    dorgbr_("P", m, m, m, &a[a_offset], n, dum, dum, &c_n1, &ierr);
	    lwork_dorgbr_p_mm__ = (int) dum[0];
	    dormbr_("P", "R", "T", m, m, m, dum, m, dum, dum, m, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_prt_mm__ = (int) dum[0];
	    dormbr_("P", "R", "T", m, n, m, dum, m, dum, dum, m, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_prt_mn__ = (int) dum[0];
	    dormbr_("P", "R", "T", n, n, m, dum, n, dum, dum, n, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_prt_nn__ = (int) dum[0];
	    dormbr_("Q", "L", "N", m, m, m, dum, m, dum, dum, m, dum, &c_n1, &
		    ierr);
	    lwork_dormbr_qln_mm__ = (int) dum[0];
	    if (*n >= mnthr) {
		if (wntqn) {
		    //
		    //                Path 1t (N >> M, JOBZ='N')
		    //
		    wrkbl = *m + lwork_dgelqf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dgebrd_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = bdspac + *m;
		    maxwrk = max(i__1,i__2);
		    minwrk = bdspac + *m;
		} else if (wntqo) {
		    //
		    //                Path 2t (N >> M, JOBZ='O')
		    //
		    wrkbl = *m + lwork_dgelqf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m + lwork_dorglq_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dgebrd_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_qln_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_prt_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + (*m << 1) * *m;
		    minwrk = bdspac + (*m << 1) * *m + *m * 3;
		} else if (wntqs) {
		    //
		    //                Path 3t (N >> M, JOBZ='S')
		    //
		    wrkbl = *m + lwork_dgelqf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m + lwork_dorglq_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dgebrd_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_qln_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_prt_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *m;
		    minwrk = bdspac + *m * *m + *m * 3;
		} else if (wntqa) {
		    //
		    //                Path 4t (N >> M, JOBZ='A')
		    //
		    wrkbl = *m + lwork_dgelqf_mn__;
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m + lwork_dorglq_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dgebrd_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_qln_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_prt_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *m;
		    // Computing MAX
		    i__1 = *m * 3 + bdspac, i__2 = *m + *n;
		    minwrk = *m * *m + max(i__1,i__2);
		}
	    } else {
		//
		//             Path 5t (N > M, but not much larger)
		//
		wrkbl = *m * 3 + lwork_dgebrd_mn__;
		if (wntqn) {
		    //                Path 5tn (N > M, jobz='N')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + bdspac;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		} else if (wntqo) {
		    //                Path 5to (N > M, jobz='O')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_qln_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_prt_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + bdspac;
		    wrkbl = max(i__1,i__2);
		    maxwrk = wrkbl + *m * *n;
		    // Computing MAX
		    i__1 = *n, i__2 = *m * *m + bdspac;
		    minwrk = *m * 3 + max(i__1,i__2);
		} else if (wntqs) {
		    //                Path 5ts (N > M, jobz='S')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_qln_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_prt_mn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + bdspac;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		} else if (wntqa) {
		    //                Path 5ta (N > M, jobz='A')
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_qln_mm__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + lwork_dormbr_prt_nn__;
		    wrkbl = max(i__1,i__2);
		    // Computing MAX
		    i__1 = wrkbl, i__2 = *m * 3 + bdspac;
		    maxwrk = max(i__1,i__2);
		    minwrk = *m * 3 + max(*n,bdspac);
		}
	    }
	}
	maxwrk = max(maxwrk,minwrk);
	work[1] = (double) maxwrk;
	if (*lwork < minwrk && ! lquery) {
	    *info = -12;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGESDD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0) {
	return 0;
    }
    //
    //    Get machine constants
    //
    eps = dlamch_("P");
    smlnum = sqrt(dlamch_("S")) / eps;
    bignum = 1. / smlnum;
    //
    //    Scale A if max element outside range [SMLNUM,BIGNUM]
    //
    anrm = dlange_("M", m, n, &a[a_offset], lda, dum);
    iscl = 0;
    if (anrm > 0. && anrm < smlnum) {
	iscl = 1;
	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda, &
		ierr);
    } else if (anrm > bignum) {
	iscl = 1;
	dlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda, &
		ierr);
    }
    if (*m >= *n) {
	//
	//       A has at least as many rows as columns. If A has sufficiently
	//       more rows than columns, first reduce using the QR
	//       decomposition (if sufficient workspace available)
	//
	if (*m >= mnthr) {
	    if (wntqn) {
		//
		//             Path 1 (M >> N, JOBZ='N')
		//             No singular vectors to be computed
		//
		itau = 1;
		nwork = itau + *n;
		//
		//             Compute A=Q*R
		//             Workspace: need   N [tau] + N    [work]
		//             Workspace: prefer N [tau] + N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);
		//
		//             Zero out below R
		//
		i__1 = *n - 1;
		i__2 = *n - 1;
		dlaset_("L", &i__1, &i__2, &c_b63, &c_b63, &a[a_dim1 + 2],
			lda);
		ie = 1;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;
		//
		//             Bidiagonalize R in A
		//             Workspace: need   3*N [e, tauq, taup] + N      [work]
		//             Workspace: prefer 3*N [e, tauq, taup] + 2*N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgebrd_(n, n, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nwork = ie + *n;
		//
		//             Perform bidiagonal SVD, computing singular values only
		//             Workspace: need   N [e] + BDSPAC
		//
		dbdsdc_("U", "N", n, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		//
		//             Path 2 (M >> N, JOBZ = 'O')
		//             N left singular vectors to be overwritten on A and
		//             N right singular vectors to be computed in VT
		//
		ir = 1;
		//
		//             WORK(IR) is LDWRKR by N
		//
		if (*lwork >= *lda * *n + *n * *n + *n * 3 + bdspac) {
		    ldwrkr = *lda;
		} else {
		    ldwrkr = (*lwork - *n * *n - *n * 3 - bdspac) / *n;
		}
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;
		//
		//             Compute A=Q*R
		//             Workspace: need   N*N [R] + N [tau] + N    [work]
		//             Workspace: prefer N*N [R] + N [tau] + N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);
		//
		//             Copy R to WORK(IR), zeroing out below it
		//
		dlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__1 = *n - 1;
		i__2 = *n - 1;
		dlaset_("L", &i__1, &i__2, &c_b63, &c_b63, &work[ir + 1], &
			ldwrkr);
		//
		//             Generate Q in A
		//             Workspace: need   N*N [R] + N [tau] + N    [work]
		//             Workspace: prefer N*N [R] + N [tau] + N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dorgqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__1, &ierr);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;
		//
		//             Bidiagonalize R in WORK(IR)
		//             Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
		//             Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		//
		//             WORK(IU) is N by N
		//
		iu = nwork;
		nwork = iu + *n * *n;
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in WORK(IU) and computing right
		//             singular vectors of bidiagonal matrix in VT
		//             Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + BDSPAC
		//
		dbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], n, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Overwrite WORK(IU) by left singular vectors of R
		//             and VT by right singular vectors of R
		//             Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N    [work]
		//             Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &work[iu], n, &work[nwork], &i__1, &ierr);
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
		//
		//             Multiply Q in A by left singular vectors of R in
		//             WORK(IU), storing result in WORK(IR) and copying to A
		//             Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N*N [U]
		//             Workspace: prefer M*N [R] + 3*N [e, tauq, taup] + N*N [U]
		//
		i__1 = *m;
		i__2 = ldwrkr;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			i__2) {
		    // Computing MIN
		    i__3 = *m - i__ + 1;
		    chunk = min(i__3,ldwrkr);
		    dgemm_("N", "N", &chunk, n, n, &c_b84, &a[i__ + a_dim1],
			    lda, &work[iu], n, &c_b63, &work[ir], &ldwrkr);
		    dlacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ +
			    a_dim1], lda);
// L10:
		}
	    } else if (wntqs) {
		//
		//             Path 3 (M >> N, JOBZ='S')
		//             N left singular vectors to be computed in U and
		//             N right singular vectors to be computed in VT
		//
		ir = 1;
		//
		//             WORK(IR) is N by N
		//
		ldwrkr = *n;
		itau = ir + ldwrkr * *n;
		nwork = itau + *n;
		//
		//             Compute A=Q*R
		//             Workspace: need   N*N [R] + N [tau] + N    [work]
		//             Workspace: prefer N*N [R] + N [tau] + N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		//
		//             Copy R to WORK(IR), zeroing out below it
		//
		dlacpy_("U", n, n, &a[a_offset], lda, &work[ir], &ldwrkr);
		i__2 = *n - 1;
		i__1 = *n - 1;
		dlaset_("L", &i__2, &i__1, &c_b63, &c_b63, &work[ir + 1], &
			ldwrkr);
		//
		//             Generate Q in A
		//             Workspace: need   N*N [R] + N [tau] + N    [work]
		//             Workspace: prefer N*N [R] + N [tau] + N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dorgqr_(m, n, n, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__2, &ierr);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;
		//
		//             Bidiagonalize R in WORK(IR)
		//             Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N      [work]
		//             Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + 2*N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgebrd_(n, n, &work[ir], &ldwrkr, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagoal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in VT
		//             Workspace: need   N*N [R] + 3*N [e, tauq, taup] + BDSPAC
		//
		dbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Overwrite U by left singular vectors of R and VT
		//             by right singular vectors of R
		//             Workspace: need   N*N [R] + 3*N [e, tauq, taup] + N    [work]
		//             Workspace: prefer N*N [R] + 3*N [e, tauq, taup] + N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", n, n, n, &work[ir], &ldwrkr, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr)
			;
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &work[ir], &ldwrkr, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
		//
		//             Multiply Q in A by left singular vectors of R in
		//             WORK(IR), storing result in U
		//             Workspace: need   N*N [R]
		//
		dlacpy_("F", n, n, &u[u_offset], ldu, &work[ir], &ldwrkr);
		dgemm_("N", "N", m, n, n, &c_b84, &a[a_offset], lda, &work[ir]
			, &ldwrkr, &c_b63, &u[u_offset], ldu);
	    } else if (wntqa) {
		//
		//             Path 4 (M >> N, JOBZ='A')
		//             M left singular vectors to be computed in U and
		//             N right singular vectors to be computed in VT
		//
		iu = 1;
		//
		//             WORK(IU) is N by N
		//
		ldwrku = *n;
		itau = iu + ldwrku * *n;
		nwork = itau + *n;
		//
		//             Compute A=Q*R, copying result to U
		//             Workspace: need   N*N [U] + N [tau] + N    [work]
		//             Workspace: prefer N*N [U] + N [tau] + N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgeqrf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		dlacpy_("L", m, n, &a[a_offset], lda, &u[u_offset], ldu);
		//
		//             Generate Q in U
		//             Workspace: need   N*N [U] + N [tau] + M    [work]
		//             Workspace: prefer N*N [U] + N [tau] + M*NB [work]
		i__2 = *lwork - nwork + 1;
		dorgqr_(m, m, n, &u[u_offset], ldu, &work[itau], &work[nwork],
			 &i__2, &ierr);
		//
		//             Produce R in A, zeroing out other entries
		//
		i__2 = *n - 1;
		i__1 = *n - 1;
		dlaset_("L", &i__2, &i__1, &c_b63, &c_b63, &a[a_dim1 + 2],
			lda);
		ie = itau;
		itauq = ie + *n;
		itaup = itauq + *n;
		nwork = itaup + *n;
		//
		//             Bidiagonalize R in A
		//             Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N      [work]
		//             Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + 2*N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgebrd_(n, n, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in WORK(IU) and computing right
		//             singular vectors of bidiagonal matrix in VT
		//             Workspace: need   N*N [U] + 3*N [e, tauq, taup] + BDSPAC
		//
		dbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], n, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Overwrite WORK(IU) by left singular vectors of R and VT
		//             by right singular vectors of R
		//             Workspace: need   N*N [U] + 3*N [e, tauq, taup] + N    [work]
		//             Workspace: prefer N*N [U] + 3*N [e, tauq, taup] + N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", n, n, n, &a[a_offset], lda, &work[
			itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			ierr);
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
		//
		//             Multiply Q in U by left singular vectors of R in
		//             WORK(IU), storing result in A
		//             Workspace: need   N*N [U]
		//
		dgemm_("N", "N", m, n, n, &c_b84, &u[u_offset], ldu, &work[iu]
			, &ldwrku, &c_b63, &a[a_offset], lda);
		//
		//             Copy left singular vectors of A from A to U
		//
		dlacpy_("F", m, n, &a[a_offset], lda, &u[u_offset], ldu);
	    }
	} else {
	    //
	    //          M .LT. MNTHR
	    //
	    //          Path 5 (M >= N, but not much larger)
	    //          Reduce to bidiagonal form without QR decomposition
	    //
	    ie = 1;
	    itauq = ie + *n;
	    itaup = itauq + *n;
	    nwork = itaup + *n;
	    //
	    //          Bidiagonalize A
	    //          Workspace: need   3*N [e, tauq, taup] + M        [work]
	    //          Workspace: prefer 3*N [e, tauq, taup] + (M+N)*NB [work]
	    //
	    i__2 = *lwork - nwork + 1;
	    dgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {
		//
		//             Path 5n (M >= N, JOBZ='N')
		//             Perform bidiagonal SVD, only computing singular values
		//             Workspace: need   3*N [e, tauq, taup] + BDSPAC
		//
		dbdsdc_("U", "N", n, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		//             Path 5o (M >= N, JOBZ='O')
		iu = nwork;
		if (*lwork >= *m * *n + *n * 3 + bdspac) {
		    //
		    //                WORK( IU ) is M by N
		    //
		    ldwrku = *m;
		    nwork = iu + ldwrku * *n;
		    dlaset_("F", m, n, &c_b63, &c_b63, &work[iu], &ldwrku);
		    //                IR is unused; silence compile warnings
		    ir = -1;
		} else {
		    //
		    //                WORK( IU ) is N by N
		    //
		    ldwrku = *n;
		    nwork = iu + ldwrku * *n;
		    //
		    //                WORK(IR) is LDWRKR by N
		    //
		    ir = nwork;
		    ldwrkr = (*lwork - *n * *n - *n * 3) / *n;
		}
		nwork = iu + ldwrku * *n;
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in WORK(IU) and computing right
		//             singular vectors of bidiagonal matrix in VT
		//             Workspace: need   3*N [e, tauq, taup] + N*N [U] + BDSPAC
		//
		dbdsdc_("U", "I", n, &s[1], &work[ie], &work[iu], &ldwrku, &
			vt[vt_offset], ldvt, dum, idum, &work[nwork], &iwork[
			1], info);
		//
		//             Overwrite VT by right singular vectors of A
		//             Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
		//             Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
		if (*lwork >= *m * *n + *n * 3 + bdspac) {
		    //
		    //                Path 5o-fast
		    //                Overwrite WORK(IU) by left singular vectors of A
		    //                Workspace: need   3*N [e, tauq, taup] + M*N [U] + N    [work]
		    //                Workspace: prefer 3*N [e, tauq, taup] + M*N [U] + N*NB [work]
		    //
		    i__2 = *lwork - nwork + 1;
		    dormbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			    itauq], &work[iu], &ldwrku, &work[nwork], &i__2, &
			    ierr);
		    //
		    //                Copy left singular vectors of A from WORK(IU) to A
		    //
		    dlacpy_("F", m, n, &work[iu], &ldwrku, &a[a_offset], lda);
		} else {
		    //
		    //                Path 5o-slow
		    //                Generate Q in A
		    //                Workspace: need   3*N [e, tauq, taup] + N*N [U] + N    [work]
		    //                Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + N*NB [work]
		    //
		    i__2 = *lwork - nwork + 1;
		    dorgbr_("Q", m, n, n, &a[a_offset], lda, &work[itauq], &
			    work[nwork], &i__2, &ierr);
		    //
		    //                Multiply Q in A by left singular vectors of
		    //                bidiagonal matrix in WORK(IU), storing result in
		    //                WORK(IR) and copying to A
		    //                Workspace: need   3*N [e, tauq, taup] + N*N [U] + NB*N [R]
		    //                Workspace: prefer 3*N [e, tauq, taup] + N*N [U] + M*N  [R]
		    //
		    i__2 = *m;
		    i__1 = ldwrkr;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
			// Computing MIN
			i__3 = *m - i__ + 1;
			chunk = min(i__3,ldwrkr);
			dgemm_("N", "N", &chunk, n, n, &c_b84, &a[i__ +
				a_dim1], lda, &work[iu], &ldwrku, &c_b63, &
				work[ir], &ldwrkr);
			dlacpy_("F", &chunk, n, &work[ir], &ldwrkr, &a[i__ +
				a_dim1], lda);
// L20:
		    }
		}
	    } else if (wntqs) {
		//
		//             Path 5s (M >= N, JOBZ='S')
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in VT
		//             Workspace: need   3*N [e, tauq, taup] + BDSPAC
		//
		dlaset_("F", m, n, &c_b63, &c_b63, &u[u_offset], ldu);
		dbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Overwrite U by left singular vectors of A and VT
		//             by right singular vectors of A
		//             Workspace: need   3*N [e, tauq, taup] + N    [work]
		//             Workspace: prefer 3*N [e, tauq, taup] + N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, n, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr)
			;
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, n, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else if (wntqa) {
		//
		//             Path 5a (M >= N, JOBZ='A')
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in VT
		//             Workspace: need   3*N [e, tauq, taup] + BDSPAC
		//
		dlaset_("F", m, m, &c_b63, &c_b63, &u[u_offset], ldu);
		dbdsdc_("U", "I", n, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Set the right corner of U to identity matrix
		//
		if (*m > *n) {
		    i__1 = *m - *n;
		    i__2 = *m - *n;
		    dlaset_("F", &i__1, &i__2, &c_b63, &c_b84, &u[*n + 1 + (*
			    n + 1) * u_dim1], ldu);
		}
		//
		//             Overwrite U by left singular vectors of A and VT
		//             by right singular vectors of A
		//             Workspace: need   3*N [e, tauq, taup] + M    [work]
		//             Workspace: prefer 3*N [e, tauq, taup] + M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr)
			;
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }
	}
    } else {
	//
	//       A has more columns than rows. If A has sufficiently more
	//       columns than rows, first reduce using the LQ decomposition (if
	//       sufficient workspace available)
	//
	if (*n >= mnthr) {
	    if (wntqn) {
		//
		//             Path 1t (N >> M, JOBZ='N')
		//             No singular vectors to be computed
		//
		itau = 1;
		nwork = itau + *m;
		//
		//             Compute A=L*Q
		//             Workspace: need   M [tau] + M [work]
		//             Workspace: prefer M [tau] + M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);
		//
		//             Zero out above L
		//
		i__1 = *m - 1;
		i__2 = *m - 1;
		dlaset_("U", &i__1, &i__2, &c_b63, &c_b63, &a[(a_dim1 << 1) +
			1], lda);
		ie = 1;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;
		//
		//             Bidiagonalize L in A
		//             Workspace: need   3*M [e, tauq, taup] + M      [work]
		//             Workspace: prefer 3*M [e, tauq, taup] + 2*M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgebrd_(m, m, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		nwork = ie + *m;
		//
		//             Perform bidiagonal SVD, computing singular values only
		//             Workspace: need   M [e] + BDSPAC
		//
		dbdsdc_("U", "N", m, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		//
		//             Path 2t (N >> M, JOBZ='O')
		//             M right singular vectors to be overwritten on A and
		//             M left singular vectors to be computed in U
		//
		ivt = 1;
		//
		//             WORK(IVT) is M by M
		//             WORK(IL)  is M by M; it is later resized to M by chunk for gemm
		//
		il = ivt + *m * *m;
		if (*lwork >= *m * *n + *m * *m + *m * 3 + bdspac) {
		    ldwrkl = *m;
		    chunk = *n;
		} else {
		    ldwrkl = *m;
		    chunk = (*lwork - *m * *m) / *m;
		}
		itau = il + ldwrkl * *m;
		nwork = itau + *m;
		//
		//             Compute A=L*Q
		//             Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
		//             Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__1, &ierr);
		//
		//             Copy L to WORK(IL), zeroing about above it
		//
		dlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__1 = *m - 1;
		i__2 = *m - 1;
		dlaset_("U", &i__1, &i__2, &c_b63, &c_b63, &work[il + ldwrkl],
			 &ldwrkl);
		//
		//             Generate Q in A
		//             Workspace: need   M*M [VT] + M*M [L] + M [tau] + M    [work]
		//             Workspace: prefer M*M [VT] + M*M [L] + M [tau] + M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dorglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__1, &ierr);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;
		//
		//             Bidiagonalize L in WORK(IL)
		//             Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M      [work]
		//             Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dgebrd_(m, m, &work[il], &ldwrkl, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__1, &ierr);
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U, and computing right singular
		//             vectors of bidiagonal matrix in WORK(IVT)
		//             Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + BDSPAC
		//
		dbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], m, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Overwrite U by left singular vectors of L and WORK(IVT)
		//             by right singular vectors of L
		//             Workspace: need   M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M    [work]
		//             Workspace: prefer M*M [VT] + M*M [L] + 3*M [e, tauq, taup] + M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr)
			;
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &work[ivt], m, &work[nwork], &i__1, &ierr);
		//
		//             Multiply right singular vectors of L in WORK(IVT) by Q
		//             in A, storing result in WORK(IL) and copying to A
		//             Workspace: need   M*M [VT] + M*M [L]
		//             Workspace: prefer M*M [VT] + M*N [L]
		//             At this point, L is resized as M by chunk.
		//
		i__1 = *n;
		i__2 = chunk;
		for (i__ = 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ +=
			i__2) {
		    // Computing MIN
		    i__3 = *n - i__ + 1;
		    blk = min(i__3,chunk);
		    dgemm_("N", "N", m, &blk, m, &c_b84, &work[ivt], m, &a[
			    i__ * a_dim1 + 1], lda, &c_b63, &work[il], &
			    ldwrkl);
		    dlacpy_("F", m, &blk, &work[il], &ldwrkl, &a[i__ * a_dim1
			    + 1], lda);
// L30:
		}
	    } else if (wntqs) {
		//
		//             Path 3t (N >> M, JOBZ='S')
		//             M right singular vectors to be computed in VT and
		//             M left singular vectors to be computed in U
		//
		il = 1;
		//
		//             WORK(IL) is M by M
		//
		ldwrkl = *m;
		itau = il + ldwrkl * *m;
		nwork = itau + *m;
		//
		//             Compute A=L*Q
		//             Workspace: need   M*M [L] + M [tau] + M    [work]
		//             Workspace: prefer M*M [L] + M [tau] + M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		//
		//             Copy L to WORK(IL), zeroing out above it
		//
		dlacpy_("L", m, m, &a[a_offset], lda, &work[il], &ldwrkl);
		i__2 = *m - 1;
		i__1 = *m - 1;
		dlaset_("U", &i__2, &i__1, &c_b63, &c_b63, &work[il + ldwrkl],
			 &ldwrkl);
		//
		//             Generate Q in A
		//             Workspace: need   M*M [L] + M [tau] + M    [work]
		//             Workspace: prefer M*M [L] + M [tau] + M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dorglq_(m, n, m, &a[a_offset], lda, &work[itau], &work[nwork],
			 &i__2, &ierr);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;
		//
		//             Bidiagonalize L in WORK(IU).
		//             Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M      [work]
		//             Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + 2*M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgebrd_(m, m, &work[il], &ldwrkl, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in VT
		//             Workspace: need   M*M [L] + 3*M [e, tauq, taup] + BDSPAC
		//
		dbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Overwrite U by left singular vectors of L and VT
		//             by right singular vectors of L
		//             Workspace: need   M*M [L] + 3*M [e, tauq, taup] + M    [work]
		//             Workspace: prefer M*M [L] + 3*M [e, tauq, taup] + M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, m, &work[il], &ldwrkl, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr)
			;
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, m, m, &work[il], &ldwrkl, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__2, &
			ierr);
		//
		//             Multiply right singular vectors of L in WORK(IL) by
		//             Q in A, storing result in VT
		//             Workspace: need   M*M [L]
		//
		dlacpy_("F", m, m, &vt[vt_offset], ldvt, &work[il], &ldwrkl);
		dgemm_("N", "N", m, n, m, &c_b84, &work[il], &ldwrkl, &a[
			a_offset], lda, &c_b63, &vt[vt_offset], ldvt);
	    } else if (wntqa) {
		//
		//             Path 4t (N >> M, JOBZ='A')
		//             N right singular vectors to be computed in VT and
		//             M left singular vectors to be computed in U
		//
		ivt = 1;
		//
		//             WORK(IVT) is M by M
		//
		ldwkvt = *m;
		itau = ivt + ldwkvt * *m;
		nwork = itau + *m;
		//
		//             Compute A=L*Q, copying result to VT
		//             Workspace: need   M*M [VT] + M [tau] + M    [work]
		//             Workspace: prefer M*M [VT] + M [tau] + M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgelqf_(m, n, &a[a_offset], lda, &work[itau], &work[nwork], &
			i__2, &ierr);
		dlacpy_("U", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
		//
		//             Generate Q in VT
		//             Workspace: need   M*M [VT] + M [tau] + N    [work]
		//             Workspace: prefer M*M [VT] + M [tau] + N*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dorglq_(n, n, m, &vt[vt_offset], ldvt, &work[itau], &work[
			nwork], &i__2, &ierr);
		//
		//             Produce L in A, zeroing out other entries
		//
		i__2 = *m - 1;
		i__1 = *m - 1;
		dlaset_("U", &i__2, &i__1, &c_b63, &c_b63, &a[(a_dim1 << 1) +
			1], lda);
		ie = itau;
		itauq = ie + *m;
		itaup = itauq + *m;
		nwork = itaup + *m;
		//
		//             Bidiagonalize L in A
		//             Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + M      [work]
		//             Workspace: prefer M*M [VT] + 3*M [e, tauq, taup] + 2*M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dgebrd_(m, m, &a[a_offset], lda, &s[1], &work[ie], &work[
			itauq], &work[itaup], &work[nwork], &i__2, &ierr);
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in WORK(IVT)
		//             Workspace: need   M*M [VT] + 3*M [e, tauq, taup] + BDSPAC
		//
		dbdsdc_("U", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1]
			, info);
		//
		//             Overwrite U by left singular vectors of L and WORK(IVT)
		//             by right singular vectors of L
		//             Workspace: need   M*M [VT] + 3*M [e, tauq, taup]+ M    [work]
		//             Workspace: prefer M*M [VT] + 3*M [e, tauq, taup]+ M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, m, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr)
			;
		i__2 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, m, m, &a[a_offset], lda, &work[
			itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2, &
			ierr);
		//
		//             Multiply right singular vectors of L in WORK(IVT) by
		//             Q in VT, storing result in A
		//             Workspace: need   M*M [VT]
		//
		dgemm_("N", "N", m, n, m, &c_b84, &work[ivt], &ldwkvt, &vt[
			vt_offset], ldvt, &c_b63, &a[a_offset], lda);
		//
		//             Copy right singular vectors of A from A to VT
		//
		dlacpy_("F", m, n, &a[a_offset], lda, &vt[vt_offset], ldvt);
	    }
	} else {
	    //
	    //          N .LT. MNTHR
	    //
	    //          Path 5t (N > M, but not much larger)
	    //          Reduce to bidiagonal form without LQ decomposition
	    //
	    ie = 1;
	    itauq = ie + *m;
	    itaup = itauq + *m;
	    nwork = itaup + *m;
	    //
	    //          Bidiagonalize A
	    //          Workspace: need   3*M [e, tauq, taup] + N        [work]
	    //          Workspace: prefer 3*M [e, tauq, taup] + (M+N)*NB [work]
	    //
	    i__2 = *lwork - nwork + 1;
	    dgebrd_(m, n, &a[a_offset], lda, &s[1], &work[ie], &work[itauq], &
		    work[itaup], &work[nwork], &i__2, &ierr);
	    if (wntqn) {
		//
		//             Path 5tn (N > M, JOBZ='N')
		//             Perform bidiagonal SVD, only computing singular values
		//             Workspace: need   3*M [e, tauq, taup] + BDSPAC
		//
		dbdsdc_("L", "N", m, &s[1], &work[ie], dum, &c__1, dum, &c__1,
			 dum, idum, &work[nwork], &iwork[1], info);
	    } else if (wntqo) {
		//             Path 5to (N > M, JOBZ='O')
		ldwkvt = *m;
		ivt = nwork;
		if (*lwork >= *m * *n + *m * 3 + bdspac) {
		    //
		    //                WORK( IVT ) is M by N
		    //
		    dlaset_("F", m, n, &c_b63, &c_b63, &work[ivt], &ldwkvt);
		    nwork = ivt + ldwkvt * *n;
		    //                IL is unused; silence compile warnings
		    il = -1;
		} else {
		    //
		    //                WORK( IVT ) is M by M
		    //
		    nwork = ivt + ldwkvt * *m;
		    il = nwork;
		    //
		    //                WORK(IL) is M by CHUNK
		    //
		    chunk = (*lwork - *m * *m - *m * 3) / *m;
		}
		//
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in WORK(IVT)
		//             Workspace: need   3*M [e, tauq, taup] + M*M [VT] + BDSPAC
		//
		dbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &
			work[ivt], &ldwkvt, dum, idum, &work[nwork], &iwork[1]
			, info);
		//
		//             Overwrite U by left singular vectors of A
		//             Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
		//             Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
		//
		i__2 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__2, &ierr)
			;
		if (*lwork >= *m * *n + *m * 3 + bdspac) {
		    //
		    //                Path 5to-fast
		    //                Overwrite WORK(IVT) by left singular vectors of A
		    //                Workspace: need   3*M [e, tauq, taup] + M*N [VT] + M    [work]
		    //                Workspace: prefer 3*M [e, tauq, taup] + M*N [VT] + M*NB [work]
		    //
		    i__2 = *lwork - nwork + 1;
		    dormbr_("P", "R", "T", m, n, m, &a[a_offset], lda, &work[
			    itaup], &work[ivt], &ldwkvt, &work[nwork], &i__2,
			    &ierr);
		    //
		    //                Copy right singular vectors of A from WORK(IVT) to A
		    //
		    dlacpy_("F", m, n, &work[ivt], &ldwkvt, &a[a_offset], lda)
			    ;
		} else {
		    //
		    //                Path 5to-slow
		    //                Generate P**T in A
		    //                Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M    [work]
		    //                Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*NB [work]
		    //
		    i__2 = *lwork - nwork + 1;
		    dorgbr_("P", m, n, m, &a[a_offset], lda, &work[itaup], &
			    work[nwork], &i__2, &ierr);
		    //
		    //                Multiply Q in A by right singular vectors of
		    //                bidiagonal matrix in WORK(IVT), storing result in
		    //                WORK(IL) and copying to A
		    //                Workspace: need   3*M [e, tauq, taup] + M*M [VT] + M*NB [L]
		    //                Workspace: prefer 3*M [e, tauq, taup] + M*M [VT] + M*N  [L]
		    //
		    i__2 = *n;
		    i__1 = chunk;
		    for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ +=
			     i__1) {
			// Computing MIN
			i__3 = *n - i__ + 1;
			blk = min(i__3,chunk);
			dgemm_("N", "N", m, &blk, m, &c_b84, &work[ivt], &
				ldwkvt, &a[i__ * a_dim1 + 1], lda, &c_b63, &
				work[il], m);
			dlacpy_("F", m, &blk, &work[il], m, &a[i__ * a_dim1 +
				1], lda);
// L40:
		    }
		}
	    } else if (wntqs) {
		//
		//             Path 5ts (N > M, JOBZ='S')
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in VT
		//             Workspace: need   3*M [e, tauq, taup] + BDSPAC
		//
		dlaset_("F", m, n, &c_b63, &c_b63, &vt[vt_offset], ldvt);
		dbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Overwrite U by left singular vectors of A and VT
		//             by right singular vectors of A
		//             Workspace: need   3*M [e, tauq, taup] + M    [work]
		//             Workspace: prefer 3*M [e, tauq, taup] + M*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr)
			;
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", m, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    } else if (wntqa) {
		//
		//             Path 5ta (N > M, JOBZ='A')
		//             Perform bidiagonal SVD, computing left singular vectors
		//             of bidiagonal matrix in U and computing right singular
		//             vectors of bidiagonal matrix in VT
		//             Workspace: need   3*M [e, tauq, taup] + BDSPAC
		//
		dlaset_("F", n, n, &c_b63, &c_b63, &vt[vt_offset], ldvt);
		dbdsdc_("L", "I", m, &s[1], &work[ie], &u[u_offset], ldu, &vt[
			vt_offset], ldvt, dum, idum, &work[nwork], &iwork[1],
			info);
		//
		//             Set the right corner of VT to identity matrix
		//
		if (*n > *m) {
		    i__1 = *n - *m;
		    i__2 = *n - *m;
		    dlaset_("F", &i__1, &i__2, &c_b63, &c_b84, &vt[*m + 1 + (*
			    m + 1) * vt_dim1], ldvt);
		}
		//
		//             Overwrite U by left singular vectors of A and VT
		//             by right singular vectors of A
		//             Workspace: need   3*M [e, tauq, taup] + N    [work]
		//             Workspace: prefer 3*M [e, tauq, taup] + N*NB [work]
		//
		i__1 = *lwork - nwork + 1;
		dormbr_("Q", "L", "N", m, m, n, &a[a_offset], lda, &work[
			itauq], &u[u_offset], ldu, &work[nwork], &i__1, &ierr)
			;
		i__1 = *lwork - nwork + 1;
		dormbr_("P", "R", "T", n, n, m, &a[a_offset], lda, &work[
			itaup], &vt[vt_offset], ldvt, &work[nwork], &i__1, &
			ierr);
	    }
	}
    }
    //
    //    Undo scaling if necessary
    //
    if (iscl == 1) {
	if (anrm > bignum) {
	    dlascl_("G", &c__0, &c__0, &bignum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
	if (anrm < smlnum) {
	    dlascl_("G", &c__0, &c__0, &smlnum, &anrm, &minmn, &c__1, &s[1], &
		    minmn, &ierr);
	}
    }
    //
    //    Return optimal workspace in WORK(1)
    //
    work[1] = (double) maxwrk;
    return 0;
    //
    //    End of DGESDD
    //
} // dgesdd_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLABRD reduces the first nb rows and columns of a general matrix to a bidiagonal form.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLABRD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlabrd.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlabrd.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlabrd.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLABRD( M, N, NB, A, LDA, D, E, TAUQ, TAUP, X, LDX, Y,
//                         LDY )
//
//      .. Scalar Arguments ..
//      INTEGER            LDA, LDX, LDY, M, N, NB
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), D( * ), E( * ), TAUP( * ),
//     $                   TAUQ( * ), X( LDX, * ), Y( LDY, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLABRD reduces the first NB rows and columns of a real general
//> m by n matrix A to upper or lower bidiagonal form by an orthogonal
//> transformation Q**T * A * P, and returns the matrices X and Y which
//> are needed to apply the transformation to the unreduced part of A.
//>
//> If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
//> bidiagonal form.
//>
//> This is an auxiliary routine called by DGEBRD
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows in the matrix A.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns in the matrix A.
//> \endverbatim
//>
//> \param[in] NB
//> \verbatim
//>          NB is INTEGER
//>          The number of leading rows and columns of A to be reduced.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the m by n general matrix to be reduced.
//>          On exit, the first NB rows and columns of the matrix are
//>          overwritten; the rest of the array is unchanged.
//>          If m >= n, elements on and below the diagonal in the first NB
//>            columns, with the array TAUQ, represent the orthogonal
//>            matrix Q as a product of elementary reflectors; and
//>            elements above the diagonal in the first NB rows, with the
//>            array TAUP, represent the orthogonal matrix P as a product
//>            of elementary reflectors.
//>          If m < n, elements below the diagonal in the first NB
//>            columns, with the array TAUQ, represent the orthogonal
//>            matrix Q as a product of elementary reflectors, and
//>            elements on and above the diagonal in the first NB rows,
//>            with the array TAUP, represent the orthogonal matrix P as
//>            a product of elementary reflectors.
//>          See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (NB)
//>          The diagonal elements of the first NB rows and columns of
//>          the reduced matrix.  D(i) = A(i,i).
//> \endverbatim
//>
//> \param[out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (NB)
//>          The off-diagonal elements of the first NB rows and columns of
//>          the reduced matrix.
//> \endverbatim
//>
//> \param[out] TAUQ
//> \verbatim
//>          TAUQ is DOUBLE PRECISION array, dimension (NB)
//>          The scalar factors of the elementary reflectors which
//>          represent the orthogonal matrix Q. See Further Details.
//> \endverbatim
//>
//> \param[out] TAUP
//> \verbatim
//>          TAUP is DOUBLE PRECISION array, dimension (NB)
//>          The scalar factors of the elementary reflectors which
//>          represent the orthogonal matrix P. See Further Details.
//> \endverbatim
//>
//> \param[out] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension (LDX,NB)
//>          The m-by-nb matrix X required to update the unreduced part
//>          of A.
//> \endverbatim
//>
//> \param[in] LDX
//> \verbatim
//>          LDX is INTEGER
//>          The leading dimension of the array X. LDX >= max(1,M).
//> \endverbatim
//>
//> \param[out] Y
//> \verbatim
//>          Y is DOUBLE PRECISION array, dimension (LDY,NB)
//>          The n-by-nb matrix Y required to update the unreduced part
//>          of A.
//> \endverbatim
//>
//> \param[in] LDY
//> \verbatim
//>          LDY is INTEGER
//>          The leading dimension of the array Y. LDY >= max(1,N).
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup doubleOTHERauxiliary
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrices Q and P are represented as products of elementary
//>  reflectors:
//>
//>     Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb)
//>
//>  Each H(i) and G(i) has the form:
//>
//>     H(i) = I - tauq * v * v**T  and G(i) = I - taup * u * u**T
//>
//>  where tauq and taup are real scalars, and v and u are real vectors.
//>
//>  If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in
//>  A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in
//>  A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).
//>
//>  If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in
//>  A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in
//>  A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).
//>
//>  The elements of the vectors v and u together form the m-by-nb matrix
//>  V and the nb-by-n matrix U**T which are needed, with X and Y, to apply
//>  the transformation to the unreduced part of the matrix, using a block
//>  update of the form:  A := A - V*Y**T - X*U**T.
//>
//>  The contents of A on exit are illustrated by the following examples
//>  with nb = 2:
//>
//>  m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):
//>
//>    (  1   1   u1  u1  u1 )           (  1   u1  u1  u1  u1  u1 )
//>    (  v1  1   1   u2  u2 )           (  1   1   u2  u2  u2  u2 )
//>    (  v1  v2  a   a   a  )           (  v1  1   a   a   a   a  )
//>    (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
//>    (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
//>    (  v1  v2  a   a   a  )
//>
//>  where a denotes an element of the original matrix which is unchanged,
//>  vi denotes an element of the vector defining H(i), and ui an element
//>  of the vector defining G(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlabrd_(int *m, int *n, int *nb, double *a, int *lda,
	double *d__, double *e, double *tauq, double *taup, double *x, int *
	ldx, double *y, int *ldy)
{
    // Table of constant values
    double c_b4 = -1.;
    double c_b5 = 1.;
    int c__1 = 1;
    double c_b16 = 0.;

    // System generated locals
    int a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;

    // Local variables
    int i__;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *),
	    dgemv_(char *, int *, int *, double *, double *, int *, double *,
	    int *, double *, double *, int *), dlarfg_(int *, double *,
	    double *, int *, double *);

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Quick return if possible
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    // Function Body
    if (*m <= 0 || *n <= 0) {
	return 0;
    }
    if (*m >= *n) {
	//
	//       Reduce to upper bidiagonal form
	//
	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    //
	    //          Update A(i:m,i)
	    //
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b4, &a[i__ + a_dim1], lda,
		     &y[i__ + y_dim1], ldy, &c_b5, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b4, &x[i__ + x_dim1], ldx,
		     &a[i__ * a_dim1 + 1], &c__1, &c_b5, &a[i__ + i__ *
		    a_dim1], &c__1);
	    //
	    //          Generate reflection Q(i) to annihilate A(i+1:m,i)
	    //
	    i__2 = *m - i__ + 1;
	    // Computing MIN
	    i__3 = i__ + 1;
	    dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ *
		    a_dim1], &c__1, &tauq[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    if (i__ < *n) {
		a[i__ + i__ * a_dim1] = 1.;
		//
		//             Compute Y(i+1:n,i)
		//
		i__2 = *m - i__ + 1;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + (i__ + 1) *
			a_dim1], lda, &a[i__ + i__ * a_dim1], &c__1, &c_b16, &
			y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + a_dim1],
			lda, &a[i__ + i__ * a_dim1], &c__1, &c_b16, &y[i__ *
			y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b5, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &x[i__ + x_dim1],
			ldx, &a[i__ + i__ * a_dim1], &c__1, &c_b16, &y[i__ *
			y_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b4, &a[(i__ + 1) *
			a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &c_b5,
			&y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		dscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);
		//
		//             Update A(i,i+1:n)
		//
		i__2 = *n - i__;
		dgemv_("No transpose", &i__2, &i__, &c_b4, &y[i__ + 1 +
			y_dim1], ldy, &a[i__ + a_dim1], lda, &c_b5, &a[i__ + (
			i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b4, &a[(i__ + 1) *
			a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b5, &a[
			i__ + (i__ + 1) * a_dim1], lda);
		//
		//             Generate reflection P(i) to annihilate A(i,i+2:n)
		//
		i__2 = *n - i__;
		// Computing MIN
		i__3 = i__ + 2;
		dlarfg_(&i__2, &a[i__ + (i__ + 1) * a_dim1], &a[i__ + min(
			i__3,*n) * a_dim1], lda, &taup[i__]);
		e[i__] = a[i__ + (i__ + 1) * a_dim1];
		a[i__ + (i__ + 1) * a_dim1] = 1.;
		//
		//             Compute X(i+1:m,i)
		//
		i__2 = *m - i__;
		i__3 = *n - i__;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + (i__
			+ 1) * a_dim1], lda, &a[i__ + (i__ + 1) * a_dim1],
			lda, &c_b16, &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		dgemv_("Transpose", &i__2, &i__, &c_b5, &y[i__ + 1 + y_dim1],
			ldy, &a[i__ + (i__ + 1) * a_dim1], lda, &c_b16, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		dgemv_("No transpose", &i__2, &i__, &c_b4, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[(i__ + 1) *
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b16, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		dscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
	    }
// L10:
	}
    } else {
	//
	//       Reduce to lower bidiagonal form
	//
	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    //
	    //          Update A(i,i:n)
	    //
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b4, &y[i__ + y_dim1], ldy,
		     &a[i__ + a_dim1], lda, &c_b5, &a[i__ + i__ * a_dim1],
		    lda);
	    i__2 = i__ - 1;
	    i__3 = *n - i__ + 1;
	    dgemv_("Transpose", &i__2, &i__3, &c_b4, &a[i__ * a_dim1 + 1],
		    lda, &x[i__ + x_dim1], ldx, &c_b5, &a[i__ + i__ * a_dim1],
		     lda);
	    //
	    //          Generate reflection P(i) to annihilate A(i,i+1:n)
	    //
	    i__2 = *n - i__ + 1;
	    // Computing MIN
	    i__3 = i__ + 1;
	    dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3,*n) *
		    a_dim1], lda, &taup[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    if (i__ < *m) {
		a[i__ + i__ * a_dim1] = 1.;
		//
		//             Compute X(i+1:m,i)
		//
		i__2 = *m - i__;
		i__3 = *n - i__ + 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + i__ *
			 a_dim1], lda, &a[i__ + i__ * a_dim1], lda, &c_b16, &
			x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &y[i__ + y_dim1],
			ldy, &a[i__ + i__ * a_dim1], lda, &c_b16, &x[i__ *
			x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &a[i__ + 1 +
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__ + 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ * a_dim1 +
			1], lda, &a[i__ + i__ * a_dim1], lda, &c_b16, &x[i__ *
			 x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &x[i__ + 1 +
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		dscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
		//
		//             Update A(i+1:m,i)
		//
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &a[i__ + 1 +
			a_dim1], lda, &y[i__ + y_dim1], ldy, &c_b5, &a[i__ +
			1 + i__ * a_dim1], &c__1);
		i__2 = *m - i__;
		dgemv_("No transpose", &i__2, &i__, &c_b4, &x[i__ + 1 +
			x_dim1], ldx, &a[i__ * a_dim1 + 1], &c__1, &c_b5, &a[
			i__ + 1 + i__ * a_dim1], &c__1);
		//
		//             Generate reflection Q(i) to annihilate A(i+2:m,i)
		//
		i__2 = *m - i__;
		// Computing MIN
		i__3 = i__ + 2;
		dlarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*m) +
			i__ * a_dim1], &c__1, &tauq[i__]);
		e[i__] = a[i__ + 1 + i__ * a_dim1];
		a[i__ + 1 + i__ * a_dim1] = 1.;
		//
		//             Compute Y(i+1:n,i)
		//
		i__2 = *m - i__;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + (i__ +
			1) * a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1,
			&c_b16, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + a_dim1],
			 lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, &y[
			i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &y[i__ + 1 +
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b5, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		dgemv_("Transpose", &i__2, &i__, &c_b5, &x[i__ + 1 + x_dim1],
			ldx, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, &y[
			i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		dgemv_("Transpose", &i__, &i__2, &c_b4, &a[(i__ + 1) * a_dim1
			+ 1], lda, &y[i__ * y_dim1 + 1], &c__1, &c_b5, &y[i__
			+ 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		dscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);
	    }
// L20:
	}
    }
    return 0;
    //
    //    End of DLABRD
    //
} // dlabrd_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAED6 used by sstedc. Computes one Newton step in solution of the secular equation.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAED6 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaed6.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaed6.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaed6.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAED6( KNITER, ORGATI, RHO, D, Z, FINIT, TAU, INFO )
//
//      .. Scalar Arguments ..
//      LOGICAL            ORGATI
//      INTEGER            INFO, KNITER
//      DOUBLE PRECISION   FINIT, RHO, TAU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( 3 ), Z( 3 )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAED6 computes the positive or negative root (closest to the origin)
//> of
//>                  z(1)        z(2)        z(3)
//> f(x) =   rho + --------- + ---------- + ---------
//>                 d(1)-x      d(2)-x      d(3)-x
//>
//> It is assumed that
//>
//>       if ORGATI = .true. the root is between d(2) and d(3);
//>       otherwise it is between d(1) and d(2)
//>
//> This routine will be called by DLAED4 when necessary. In most cases,
//> the root sought is the smallest in magnitude, though it might not be
//> in some extremely rare situations.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] KNITER
//> \verbatim
//>          KNITER is INTEGER
//>               Refer to DLAED4 for its significance.
//> \endverbatim
//>
//> \param[in] ORGATI
//> \verbatim
//>          ORGATI is LOGICAL
//>               If ORGATI is true, the needed root is between d(2) and
//>               d(3); otherwise it is between d(1) and d(2).  See
//>               DLAED4 for further details.
//> \endverbatim
//>
//> \param[in] RHO
//> \verbatim
//>          RHO is DOUBLE PRECISION
//>               Refer to the equation f(x) above.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (3)
//>               D satisfies d(1) < d(2) < d(3).
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (3)
//>               Each of the elements in z must be positive.
//> \endverbatim
//>
//> \param[in] FINIT
//> \verbatim
//>          FINIT is DOUBLE PRECISION
//>               The value of f at 0. It is more accurate than the one
//>               evaluated inside this routine (if someone wants to do
//>               so).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>               The root of the equation f(x).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>               = 0: successful exit
//>               > 0: if INFO = 1, failure to converge
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup auxOTHERcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  10/02/03: This version has a few statements commented out for thread
//>  safety (machine parameters are computed on each entry). SJH.
//>
//>  05/10/06: Modified from a new version of Ren-Cang Li, use
//>     Gragg-Thornton-Warner cubic convergent scheme for better stability.
//> \endverbatim
//
//> \par Contributors:
// ==================
//>
//>     Ren-Cang Li, Computer Science Division, University of California
//>     at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlaed6_(int *kniter, int *orgati, double *rho, double *
	d__, double *z__, double *finit, double *tau, int *info)
{
    // System generated locals
    int i__1;
    double d__1, d__2, d__3, d__4;

    // Local variables
    double a, b, c__, f;
    int i__;
    double fc, df, ddf, lbd, eta, ubd, eps, base;
    int iter;
    double temp, temp1, temp2, temp3, temp4;
    int scale;
    int niter;
    double small1, small2, sminv1, sminv2;
    extern double dlamch_(char *);
    double dscale[3], sclfac, zscale[3], erretm, sclinv;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --z__;
    --d__;

    // Function Body
    *info = 0;
    if (*orgati) {
	lbd = d__[2];
	ubd = d__[3];
    } else {
	lbd = d__[1];
	ubd = d__[2];
    }
    if (*finit < 0.) {
	lbd = 0.;
    } else {
	ubd = 0.;
    }
    niter = 1;
    *tau = 0.;
    if (*kniter == 2) {
	if (*orgati) {
	    temp = (d__[3] - d__[2]) / 2.;
	    c__ = *rho + z__[1] / (d__[1] - d__[2] - temp);
	    a = c__ * (d__[2] + d__[3]) + z__[2] + z__[3];
	    b = c__ * d__[2] * d__[3] + z__[2] * d__[3] + z__[3] * d__[2];
	} else {
	    temp = (d__[1] - d__[2]) / 2.;
	    c__ = *rho + z__[3] / (d__[3] - d__[2] - temp);
	    a = c__ * (d__[1] + d__[2]) + z__[1] + z__[2];
	    b = c__ * d__[1] * d__[2] + z__[1] * d__[2] + z__[2] * d__[1];
	}
	// Computing MAX
	d__1 = abs(a), d__2 = abs(b), d__1 = max(d__1,d__2), d__2 = abs(c__);
	temp = max(d__1,d__2);
	a /= temp;
	b /= temp;
	c__ /= temp;
	if (c__ == 0.) {
	    *tau = b / a;
	} else if (a <= 0.) {
	    *tau = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (
		    c__ * 2.);
	} else {
	    *tau = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__, abs(d__1))
		    ));
	}
	if (*tau < lbd || *tau > ubd) {
	    *tau = (lbd + ubd) / 2.;
	}
	if (d__[1] == *tau || d__[2] == *tau || d__[3] == *tau) {
	    *tau = 0.;
	} else {
	    temp = *finit + *tau * z__[1] / (d__[1] * (d__[1] - *tau)) + *tau
		    * z__[2] / (d__[2] * (d__[2] - *tau)) + *tau * z__[3] / (
		    d__[3] * (d__[3] - *tau));
	    if (temp <= 0.) {
		lbd = *tau;
	    } else {
		ubd = *tau;
	    }
	    if (abs(*finit) <= abs(temp)) {
		*tau = 0.;
	    }
	}
    }
    //
    //    get machine parameters for possible scaling to avoid overflow
    //
    //    modified by Sven: parameters SMALL1, SMINV1, SMALL2,
    //    SMINV2, EPS are not SAVEd anymore between one call to the
    //    others but recomputed at each call
    //
    eps = dlamch_("Epsilon");
    base = dlamch_("Base");
    i__1 = (int) (log(dlamch_("SafMin")) / log(base) / 3.);
    small1 = pow_di(&base, &i__1);
    sminv1 = 1. / small1;
    small2 = small1 * small1;
    sminv2 = sminv1 * sminv1;
    //
    //    Determine if scaling of inputs necessary to avoid overflow
    //    when computing 1/TEMP**3
    //
    if (*orgati) {
	// Computing MIN
	d__3 = (d__1 = d__[2] - *tau, abs(d__1)), d__4 = (d__2 = d__[3] - *
		tau, abs(d__2));
	temp = min(d__3,d__4);
    } else {
	// Computing MIN
	d__3 = (d__1 = d__[1] - *tau, abs(d__1)), d__4 = (d__2 = d__[2] - *
		tau, abs(d__2));
	temp = min(d__3,d__4);
    }
    scale = FALSE_;
    if (temp <= small1) {
	scale = TRUE_;
	if (temp <= small2) {
	    //
	    //       Scale up by power of radix nearest 1/SAFMIN**(2/3)
	    //
	    sclfac = sminv2;
	    sclinv = small2;
	} else {
	    //
	    //       Scale up by power of radix nearest 1/SAFMIN**(1/3)
	    //
	    sclfac = sminv1;
	    sclinv = small1;
	}
	//
	//       Scaling up safe because D, Z, TAU scaled elsewhere to be O(1)
	//
	for (i__ = 1; i__ <= 3; ++i__) {
	    dscale[i__ - 1] = d__[i__] * sclfac;
	    zscale[i__ - 1] = z__[i__] * sclfac;
// L10:
	}
	*tau *= sclfac;
	lbd *= sclfac;
	ubd *= sclfac;
    } else {
	//
	//       Copy D and Z to DSCALE and ZSCALE
	//
	for (i__ = 1; i__ <= 3; ++i__) {
	    dscale[i__ - 1] = d__[i__];
	    zscale[i__ - 1] = z__[i__];
// L20:
	}
    }
    fc = 0.;
    df = 0.;
    ddf = 0.;
    for (i__ = 1; i__ <= 3; ++i__) {
	temp = 1. / (dscale[i__ - 1] - *tau);
	temp1 = zscale[i__ - 1] * temp;
	temp2 = temp1 * temp;
	temp3 = temp2 * temp;
	fc += temp1 / dscale[i__ - 1];
	df += temp2;
	ddf += temp3;
// L30:
    }
    f = *finit + *tau * fc;
    if (abs(f) <= 0.) {
	goto L60;
    }
    if (f <= 0.) {
	lbd = *tau;
    } else {
	ubd = *tau;
    }
    //
    //       Iteration begins -- Use Gragg-Thornton-Warner cubic convergent
    //                           scheme
    //
    //    It is not hard to see that
    //
    //          1) Iterations will go up monotonically
    //             if FINIT < 0;
    //
    //          2) Iterations will go down monotonically
    //             if FINIT > 0.
    //
    iter = niter + 1;
    for (niter = iter; niter <= 40; ++niter) {
	if (*orgati) {
	    temp1 = dscale[1] - *tau;
	    temp2 = dscale[2] - *tau;
	} else {
	    temp1 = dscale[0] - *tau;
	    temp2 = dscale[1] - *tau;
	}
	a = (temp1 + temp2) * f - temp1 * temp2 * df;
	b = temp1 * temp2 * f;
	c__ = f - (temp1 + temp2) * df + temp1 * temp2 * ddf;
	// Computing MAX
	d__1 = abs(a), d__2 = abs(b), d__1 = max(d__1,d__2), d__2 = abs(c__);
	temp = max(d__1,d__2);
	a /= temp;
	b /= temp;
	c__ /= temp;
	if (c__ == 0.) {
	    eta = b / a;
	} else if (a <= 0.) {
	    eta = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (c__
		    * 2.);
	} else {
	    eta = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))
		    );
	}
	if (f * eta >= 0.) {
	    eta = -f / df;
	}
	*tau += eta;
	if (*tau < lbd || *tau > ubd) {
	    *tau = (lbd + ubd) / 2.;
	}
	fc = 0.;
	erretm = 0.;
	df = 0.;
	ddf = 0.;
	for (i__ = 1; i__ <= 3; ++i__) {
	    if (dscale[i__ - 1] - *tau != 0.) {
		temp = 1. / (dscale[i__ - 1] - *tau);
		temp1 = zscale[i__ - 1] * temp;
		temp2 = temp1 * temp;
		temp3 = temp2 * temp;
		temp4 = temp1 / dscale[i__ - 1];
		fc += temp4;
		erretm += abs(temp4);
		df += temp2;
		ddf += temp3;
	    } else {
		goto L60;
	    }
// L40:
	}
	f = *finit + *tau * fc;
	erretm = (abs(*finit) + abs(*tau) * erretm) * 8. + abs(*tau) * df;
	if (abs(f) <= eps * 4. * erretm || ubd - lbd <= eps * 4. * abs(*tau))
		{
	    goto L60;
	}
	if (f <= 0.) {
	    lbd = *tau;
	} else {
	    ubd = *tau;
	}
// L50:
    }
    *info = 1;
L60:
    //
    //    Undo scaling
    //
    if (scale) {
	*tau *= sclinv;
    }
    return 0;
    //
    //    End of DLAED6
    //
} // dlaed6_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAMRG creates a permutation list to merge the entries of two independently sorted sets into a single set sorted in ascending order.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAMRG + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlamrg.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlamrg.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlamrg.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAMRG( N1, N2, A, DTRD1, DTRD2, INDEX )
//
//      .. Scalar Arguments ..
//      INTEGER            DTRD1, DTRD2, N1, N2
//      ..
//      .. Array Arguments ..
//      INTEGER            INDEX( * )
//      DOUBLE PRECISION   A( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAMRG will create a permutation list which will merge the elements
//> of A (which is composed of two independently sorted sets) into a
//> single set which is sorted in ascending order.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N1
//> \verbatim
//>          N1 is INTEGER
//> \endverbatim
//>
//> \param[in] N2
//> \verbatim
//>          N2 is INTEGER
//>         These arguments contain the respective lengths of the two
//>         sorted lists to be merged.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (N1+N2)
//>         The first N1 elements of A contain a list of numbers which
//>         are sorted in either ascending or descending order.  Likewise
//>         for the final N2 elements.
//> \endverbatim
//>
//> \param[in] DTRD1
//> \verbatim
//>          DTRD1 is INTEGER
//> \endverbatim
//>
//> \param[in] DTRD2
//> \verbatim
//>          DTRD2 is INTEGER
//>         These are the strides to be taken through the array A.
//>         Allowable strides are 1 and -1.  They indicate whether a
//>         subset of A is sorted in ascending (DTRDx = 1) or descending
//>         (DTRDx = -1) order.
//> \endverbatim
//>
//> \param[out] INDEX
//> \verbatim
//>          INDEX is INTEGER array, dimension (N1+N2)
//>         On exit this array will contain a permutation such that
//>         if B( I ) = A( INDEX( I ) ) for I=1,N1+N2, then B will be
//>         sorted in ascending order.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlamrg_(int *n1, int *n2, double *a, int *dtrd1, int *
	dtrd2, int *index)
{
    // System generated locals
    int i__1;

    // Local variables
    int i__, ind1, ind2, n1sv, n2sv;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --index;
    --a;

    // Function Body
    n1sv = *n1;
    n2sv = *n2;
    if (*dtrd1 > 0) {
	ind1 = 1;
    } else {
	ind1 = *n1;
    }
    if (*dtrd2 > 0) {
	ind2 = *n1 + 1;
    } else {
	ind2 = *n1 + *n2;
    }
    i__ = 1;
    //    while ( (N1SV > 0) & (N2SV > 0) )
L10:
    if (n1sv > 0 && n2sv > 0) {
	if (a[ind1] <= a[ind2]) {
	    index[i__] = ind1;
	    ++i__;
	    ind1 += *dtrd1;
	    --n1sv;
	} else {
	    index[i__] = ind2;
	    ++i__;
	    ind2 += *dtrd2;
	    --n2sv;
	}
	goto L10;
    }
    //    end while
    if (n1sv == 0) {
	i__1 = n2sv;
	for (n1sv = 1; n1sv <= i__1; ++n1sv) {
	    index[i__] = ind2;
	    ++i__;
	    ind2 += *dtrd2;
// L20:
	}
    } else {
	//    N2SV .EQ. 0
	i__1 = n1sv;
	for (n2sv = 1; n2sv <= i__1; ++n2sv) {
	    index[i__] = ind1;
	    ++i__;
	    ind1 += *dtrd1;
// L30:
	}
    }
    return 0;
    //
    //    End of DLAMRG
    //
} // dlamrg_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLANST returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the element of largest absolute value of a real symmetric tridiagonal matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLANST + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlanst.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlanst.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlanst.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      DOUBLE PRECISION FUNCTION DLANST( NORM, N, D, E )
//
//      .. Scalar Arguments ..
//      CHARACTER          NORM
//      INTEGER            N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), E( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLANST  returns the value of the one norm,  or the Frobenius norm, or
//> the  infinity norm,  or the  element of  largest absolute value  of a
//> real symmetric tridiagonal matrix A.
//> \endverbatim
//>
//> \return DLANST
//> \verbatim
//>
//>    DLANST = ( max(abs(A(i,j))), NORM = 'M' or 'm'
//>             (
//>             ( norm1(A),         NORM = '1', 'O' or 'o'
//>             (
//>             ( normI(A),         NORM = 'I' or 'i'
//>             (
//>             ( normF(A),         NORM = 'F', 'f', 'E' or 'e'
//>
//> where  norm1  denotes the  one norm of a matrix (maximum column sum),
//> normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
//> normF  denotes the  Frobenius norm of a matrix (square root of sum of
//> squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] NORM
//> \verbatim
//>          NORM is CHARACTER*1
//>          Specifies the value to be returned in DLANST as described
//>          above.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.  N >= 0.  When N = 0, DLANST is
//>          set to zero.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The diagonal elements of A.
//> \endverbatim
//>
//> \param[in] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          The (n-1) sub-diagonal or super-diagonal elements of A.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
// =====================================================================
double dlanst_(char *norm, int *n, double *d__, double *e)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int i__1;
    double ret_val, d__1, d__2, d__3;

    // Local variables
    int i__;
    double sum, scale;
    extern int lsame_(char *, char *);
    double anorm;
    extern int disnan_(double *);
    extern /* Subroutine */ int dlassq_(int *, double *, int *, double *,
	    double *);

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --e;
    --d__;

    // Function Body
    if (*n <= 0) {
	anorm = 0.;
    } else if (lsame_(norm, "M")) {
	//
	//       Find max(abs(A(i,j))).
	//
	anorm = (d__1 = d__[*n], abs(d__1));
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    sum = (d__1 = d__[i__], abs(d__1));
	    if (anorm < sum || disnan_(&sum)) {
		anorm = sum;
	    }
	    sum = (d__1 = e[i__], abs(d__1));
	    if (anorm < sum || disnan_(&sum)) {
		anorm = sum;
	    }
// L10:
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)norm == '1' || lsame_(
	    norm, "I")) {
	//
	//       Find norm1(A).
	//
	if (*n == 1) {
	    anorm = abs(d__[1]);
	} else {
	    anorm = abs(d__[1]) + abs(e[1]);
	    sum = (d__1 = e[*n - 1], abs(d__1)) + (d__2 = d__[*n], abs(d__2));
	    if (anorm < sum || disnan_(&sum)) {
		anorm = sum;
	    }
	    i__1 = *n - 1;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		sum = (d__1 = d__[i__], abs(d__1)) + (d__2 = e[i__], abs(d__2)
			) + (d__3 = e[i__ - 1], abs(d__3));
		if (anorm < sum || disnan_(&sum)) {
		    anorm = sum;
		}
// L20:
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {
	//
	//       Find normF(A).
	//
	scale = 0.;
	sum = 1.;
	if (*n > 1) {
	    i__1 = *n - 1;
	    dlassq_(&i__1, &e[1], &c__1, &scale, &sum);
	    sum *= 2;
	}
	dlassq_(n, &d__[1], &c__1, &scale, &sum);
	anorm = scale * sqrt(sum);
    }
    ret_val = anorm;
    return ret_val;
    //
    //    End of DLANST
    //
} // dlanst_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAS2 computes singular values of a 2-by-2 triangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAS2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlas2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlas2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlas2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAS2( F, G, H, SSMIN, SSMAX )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   F, G, H, SSMAX, SSMIN
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAS2  computes the singular values of the 2-by-2 matrix
//>    [  F   G  ]
//>    [  0   H  ].
//> On return, SSMIN is the smaller singular value and SSMAX is the
//> larger singular value.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] F
//> \verbatim
//>          F is DOUBLE PRECISION
//>          The (1,1) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] G
//> \verbatim
//>          G is DOUBLE PRECISION
//>          The (1,2) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] H
//> \verbatim
//>          H is DOUBLE PRECISION
//>          The (2,2) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[out] SSMIN
//> \verbatim
//>          SSMIN is DOUBLE PRECISION
//>          The smaller singular value.
//> \endverbatim
//>
//> \param[out] SSMAX
//> \verbatim
//>          SSMAX is DOUBLE PRECISION
//>          The larger singular value.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Barring over/underflow, all output quantities are correct to within
//>  a few units in the last place (ulps), even in the absence of a guard
//>  digit in addition/subtraction.
//>
//>  In IEEE arithmetic, the code works correctly if one matrix element is
//>  infinite.
//>
//>  Overflow will not occur unless the largest singular value itself
//>  overflows, or is within a few ulps of overflow. (On machines with
//>  partial overflow, like the Cray, overflow may occur if the largest
//>  singular value is within a factor of 2 of overflow.)
//>
//>  Underflow is harmless if underflow is gradual. Otherwise, results
//>  may correspond to a matrix modified by perturbations of size near
//>  the underflow threshold.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlas2_(double *f, double *g, double *h__, double *ssmin,
	double *ssmax)
{
    // System generated locals
    double d__1, d__2;

    // Local variables
    double c__, fa, ga, ha, as, at, au, fhmn, fhmx;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // ====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    fa = abs(*f);
    ga = abs(*g);
    ha = abs(*h__);
    fhmn = min(fa,ha);
    fhmx = max(fa,ha);
    if (fhmn == 0.) {
	*ssmin = 0.;
	if (fhmx == 0.) {
	    *ssmax = ga;
	} else {
	    // Computing 2nd power
	    d__1 = min(fhmx,ga) / max(fhmx,ga);
	    *ssmax = max(fhmx,ga) * sqrt(d__1 * d__1 + 1.);
	}
    } else {
	if (ga < fhmx) {
	    as = fhmn / fhmx + 1.;
	    at = (fhmx - fhmn) / fhmx;
	    // Computing 2nd power
	    d__1 = ga / fhmx;
	    au = d__1 * d__1;
	    c__ = 2. / (sqrt(as * as + au) + sqrt(at * at + au));
	    *ssmin = fhmn * c__;
	    *ssmax = fhmx / c__;
	} else {
	    au = fhmx / ga;
	    if (au == 0.) {
		//
		//             Avoid possible harmful underflow if exponent range
		//             asymmetric (true SSMIN may not underflow even if
		//             AU underflows)
		//
		*ssmin = fhmn * fhmx / ga;
		*ssmax = ga;
	    } else {
		as = fhmn / fhmx + 1.;
		at = (fhmx - fhmn) / fhmx;
		// Computing 2nd power
		d__1 = as * au;
		// Computing 2nd power
		d__2 = at * au;
		c__ = 1. / (sqrt(d__1 * d__1 + 1.) + sqrt(d__2 * d__2 + 1.));
		*ssmin = fhmn * c__ * au;
		*ssmin += *ssmin;
		*ssmax = ga / (c__ + c__);
	    }
	}
    }
    return 0;
    //
    //    End of DLAS2
    //
} // dlas2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD0 computes the singular values of a real upper bidiagonal n-by-m matrix B with diagonal d and off-diagonal e. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD0 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd0.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd0.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd0.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD0( N, SQRE, D, E, U, LDU, VT, LDVT, SMLSIZ, IWORK,
//                         WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDU, LDVT, N, SMLSIZ, SQRE
//      ..
//      .. Array Arguments ..
//      INTEGER            IWORK( * )
//      DOUBLE PRECISION   D( * ), E( * ), U( LDU, * ), VT( LDVT, * ),
//     $                   WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Using a divide and conquer approach, DLASD0 computes the singular
//> value decomposition (SVD) of a real upper bidiagonal N-by-M
//> matrix B with diagonal D and offdiagonal E, where M = N + SQRE.
//> The algorithm computes orthogonal matrices U and VT such that
//> B = U * S * VT. The singular values S are overwritten on D.
//>
//> A related subroutine, DLASDA, computes only the singular values,
//> and optionally, the singular vectors in compact form.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>         On entry, the row dimension of the upper bidiagonal matrix.
//>         This is also the dimension of the main diagonal array D.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>         Specifies the column dimension of the bidiagonal matrix.
//>         = 0: The bidiagonal matrix has column dimension M = N;
//>         = 1: The bidiagonal matrix has column dimension M = N+1;
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>         On entry D contains the main diagonal of the bidiagonal
//>         matrix.
//>         On exit D, if INFO = 0, contains its singular values.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (M-1)
//>         Contains the subdiagonal entries of the bidiagonal matrix.
//>         On exit, E has been destroyed.
//> \endverbatim
//>
//> \param[out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension (LDU, N)
//>         On exit, U contains the left singular vectors.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>         On entry, leading dimension of U.
//> \endverbatim
//>
//> \param[out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension (LDVT, M)
//>         On exit, VT**T contains the right singular vectors.
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>         On entry, leading dimension of VT.
//> \endverbatim
//>
//> \param[in] SMLSIZ
//> \verbatim
//>          SMLSIZ is INTEGER
//>         On entry, maximum size of the subproblems at the
//>         bottom of the computation tree.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (8*N)
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (3*M**2+2*M)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  if INFO = 1, a singular value did not converge
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd0_(int *n, int *sqre, double *d__, double *e,
	double *u, int *ldu, double *vt, int *ldvt, int *smlsiz, int *iwork,
	double *work, int *info)
{
    // Table of constant values
    int c__0 = 0;
    int c__2 = 2;

    // System generated locals
    int u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;

    // Local variables
    int i__, j, m, i1, ic, lf, nd, ll, nl, nr, im1, ncc, nlf, nrf, iwk, lvl,
	    ndb1, nlp1, nrp1;
    double beta;
    int idxq, nlvl;
    double alpha;
    int inode, ndiml, idxqc, ndimr, itemp, sqrei;
    extern /* Subroutine */ int dlasd1_(int *, int *, int *, double *, double
	    *, double *, double *, int *, double *, int *, int *, int *,
	    double *, int *), dlasdq_(char *, int *, int *, int *, int *, int
	    *, double *, double *, double *, int *, double *, int *, double *,
	     int *, double *, int *), dlasdt_(int *, int *, int *, int *, int
	    *, int *, int *), xerbla_(char *, int *);

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
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

    // Function Body
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
	xerbla_("DLASD0", &i__1);
	return 0;
    }
    //
    //    If the input matrix is too small, call DLASDQ to find the SVD.
    //
    if (*n <= *smlsiz) {
	dlasdq_("U", sqre, n, &m, n, &c__0, &d__[1], &e[1], &vt[vt_offset],
		ldvt, &u[u_offset], ldu, &u[u_offset], ldu, &work[1], info);
	return 0;
    }
    //
    //    Set up the computation tree.
    //
    inode = 1;
    ndiml = inode + *n;
    ndimr = ndiml + *n;
    idxq = ndimr + *n;
    iwk = idxq + *n;
    dlasdt_(n, &nlvl, &nd, &iwork[inode], &iwork[ndiml], &iwork[ndimr],
	    smlsiz);
    //
    //    For the nodes on bottom level of the tree, solve
    //    their subproblems by DLASDQ.
    //
    ndb1 = (nd + 1) / 2;
    ncc = 0;
    i__1 = nd;
    for (i__ = ndb1; i__ <= i__1; ++i__) {
	//
	//    IC : center row of each node
	//    NL : number of rows of left  subproblem
	//    NR : number of rows of right subproblem
	//    NLF: starting row of the left   subproblem
	//    NRF: starting row of the right  subproblem
	//
	i1 = i__ - 1;
	ic = iwork[inode + i1];
	nl = iwork[ndiml + i1];
	nlp1 = nl + 1;
	nr = iwork[ndimr + i1];
	nrp1 = nr + 1;
	nlf = ic - nl;
	nrf = ic + 1;
	sqrei = 1;
	dlasdq_("U", &sqrei, &nl, &nlp1, &nl, &ncc, &d__[nlf], &e[nlf], &vt[
		nlf + nlf * vt_dim1], ldvt, &u[nlf + nlf * u_dim1], ldu, &u[
		nlf + nlf * u_dim1], ldu, &work[1], info);
	if (*info != 0) {
	    return 0;
	}
	itemp = idxq + nlf - 2;
	i__2 = nl;
	for (j = 1; j <= i__2; ++j) {
	    iwork[itemp + j] = j;
// L10:
	}
	if (i__ == nd) {
	    sqrei = *sqre;
	} else {
	    sqrei = 1;
	}
	nrp1 = nr + sqrei;
	dlasdq_("U", &sqrei, &nr, &nrp1, &nr, &ncc, &d__[nrf], &e[nrf], &vt[
		nrf + nrf * vt_dim1], ldvt, &u[nrf + nrf * u_dim1], ldu, &u[
		nrf + nrf * u_dim1], ldu, &work[1], info);
	if (*info != 0) {
	    return 0;
	}
	itemp = idxq + ic;
	i__2 = nr;
	for (j = 1; j <= i__2; ++j) {
	    iwork[itemp + j - 1] = j;
// L20:
	}
// L30:
    }
    //
    //    Now conquer each subproblem bottom-up.
    //
    for (lvl = nlvl; lvl >= 1; --lvl) {
	//
	//       Find the first node LF and last node LL on the
	//       current level LVL.
	//
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
	    dlasd1_(&nl, &nr, &sqrei, &d__[nlf], &alpha, &beta, &u[nlf + nlf *
		     u_dim1], ldu, &vt[nlf + nlf * vt_dim1], ldvt, &iwork[
		    idxqc], &iwork[iwk], &work[1], info);
	    //
	    //       Report the possible convergence failure.
	    //
	    if (*info != 0) {
		return 0;
	    }
// L40:
	}
// L50:
    }
    return 0;
    //
    //    End of DLASD0
    //
} // dlasd0_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD1 computes the SVD of an upper bidiagonal matrix B of the specified size. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD1 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd1.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd1.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd1.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD1( NL, NR, SQRE, D, ALPHA, BETA, U, LDU, VT, LDVT,
//                         IDXQ, IWORK, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDU, LDVT, NL, NR, SQRE
//      DOUBLE PRECISION   ALPHA, BETA
//      ..
//      .. Array Arguments ..
//      INTEGER            IDXQ( * ), IWORK( * )
//      DOUBLE PRECISION   D( * ), U( LDU, * ), VT( LDVT, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASD1 computes the SVD of an upper bidiagonal N-by-M matrix B,
//> where N = NL + NR + 1 and M = N + SQRE. DLASD1 is called from DLASD0.
//>
//> A related subroutine DLASD7 handles the case in which the singular
//> values (and the singular vectors in factored form) are desired.
//>
//> DLASD1 computes the SVD as follows:
//>
//>               ( D1(in)    0    0       0 )
//>   B = U(in) * (   Z1**T   a   Z2**T    b ) * VT(in)
//>               (   0       0   D2(in)   0 )
//>
//>     = U(out) * ( D(out) 0) * VT(out)
//>
//> where Z**T = (Z1**T a Z2**T b) = u**T VT**T, and u is a vector of dimension M
//> with ALPHA and BETA in the NL+1 and NL+2 th entries and zeros
//> elsewhere; and the entry b is empty if SQRE = 0.
//>
//> The left singular vectors of the original matrix are stored in U, and
//> the transpose of the right singular vectors are stored in VT, and the
//> singular values are in D.  The algorithm consists of three stages:
//>
//>    The first stage consists of deflating the size of the problem
//>    when there are multiple singular values or when there are zeros in
//>    the Z vector.  For each such occurrence the dimension of the
//>    secular equation problem is reduced by one.  This stage is
//>    performed by the routine DLASD2.
//>
//>    The second stage consists of calculating the updated
//>    singular values. This is done by finding the square roots of the
//>    roots of the secular equation via the routine DLASD4 (as called
//>    by DLASD3). This routine also calculates the singular vectors of
//>    the current problem.
//>
//>    The final stage consists of computing the updated singular vectors
//>    directly using the updated singular values.  The singular vectors
//>    for the current problem are multiplied with the singular vectors
//>    from the overall problem.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] NL
//> \verbatim
//>          NL is INTEGER
//>         The row dimension of the upper block.  NL >= 1.
//> \endverbatim
//>
//> \param[in] NR
//> \verbatim
//>          NR is INTEGER
//>         The row dimension of the lower block.  NR >= 1.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>         = 0: the lower block is an NR-by-NR square matrix.
//>         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
//>
//>         The bidiagonal matrix has row dimension N = NL + NR + 1,
//>         and column dimension M = N + SQRE.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array,
//>                        dimension (N = NL+NR+1).
//>         On entry D(1:NL,1:NL) contains the singular values of the
//>         upper block; and D(NL+2:N) contains the singular values of
//>         the lower block. On exit D(1:N) contains the singular values
//>         of the modified matrix.
//> \endverbatim
//>
//> \param[in,out] ALPHA
//> \verbatim
//>          ALPHA is DOUBLE PRECISION
//>         Contains the diagonal element associated with the added row.
//> \endverbatim
//>
//> \param[in,out] BETA
//> \verbatim
//>          BETA is DOUBLE PRECISION
//>         Contains the off-diagonal element associated with the added
//>         row.
//> \endverbatim
//>
//> \param[in,out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension(LDU,N)
//>         On entry U(1:NL, 1:NL) contains the left singular vectors of
//>         the upper block; U(NL+2:N, NL+2:N) contains the left singular
//>         vectors of the lower block. On exit U contains the left
//>         singular vectors of the bidiagonal matrix.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>         The leading dimension of the array U.  LDU >= max( 1, N ).
//> \endverbatim
//>
//> \param[in,out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension(LDVT,M)
//>         where M = N + SQRE.
//>         On entry VT(1:NL+1, 1:NL+1)**T contains the right singular
//>         vectors of the upper block; VT(NL+2:M, NL+2:M)**T contains
//>         the right singular vectors of the lower block. On exit
//>         VT**T contains the right singular vectors of the
//>         bidiagonal matrix.
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>         The leading dimension of the array VT.  LDVT >= max( 1, M ).
//> \endverbatim
//>
//> \param[in,out] IDXQ
//> \verbatim
//>          IDXQ is INTEGER array, dimension(N)
//>         This contains the permutation which will reintegrate the
//>         subproblem just solved back into sorted order, i.e.
//>         D( IDXQ( I = 1, N ) ) will be in ascending order.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension( 4 * N )
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension( 3*M**2 + 2*M )
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  if INFO = 1, a singular value did not converge
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd1_(int *nl, int *nr, int *sqre, double *d__, double
	*alpha, double *beta, double *u, int *ldu, double *vt, int *ldvt, int
	*idxq, int *iwork, double *work, int *info)
{
    // Table of constant values
    int c__0 = 0;
    double c_b7 = 1.;
    int c__1 = 1;
    int c_n1 = -1;

    // System generated locals
    int u_dim1, u_offset, vt_dim1, vt_offset, i__1;
    double d__1, d__2;

    // Local variables
    int i__, k, m, n, n1, n2, iq, iz, iu2, ldq, idx, ldu2, ivt2, idxc, idxp,
	    ldvt2;
    extern /* Subroutine */ int dlasd2_(int *, int *, int *, int *, double *,
	    double *, double *, double *, double *, int *, double *, int *,
	    double *, double *, int *, double *, int *, int *, int *, int *,
	    int *, int *, int *), dlasd3_(int *, int *, int *, int *, double *
	    , double *, int *, double *, double *, int *, double *, int *,
	    double *, int *, double *, int *, int *, int *, double *, int *),
	    dlascl_(char *, int *, int *, double *, double *, int *, int *,
	    double *, int *, int *), dlamrg_(int *, int *, double *, int *,
	    int *, int *);
    int isigma;
    extern /* Subroutine */ int xerbla_(char *, int *);
    double orgnrm;
    int coltyp;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    --idxq;
    --iwork;
    --work;

    // Function Body
    *info = 0;
    if (*nl < 1) {
	*info = -1;
    } else if (*nr < 1) {
	*info = -2;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -3;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASD1", &i__1);
	return 0;
    }
    n = *nl + *nr + 1;
    m = n + *sqre;
    //
    //    The following values are for bookkeeping purposes only.  They are
    //    integer pointers which indicate the portion of the workspace
    //    used by a particular array in DLASD2 and DLASD3.
    //
    ldu2 = n;
    ldvt2 = m;
    iz = 1;
    isigma = iz + m;
    iu2 = isigma + n;
    ivt2 = iu2 + ldu2 * n;
    iq = ivt2 + ldvt2 * m;
    idx = 1;
    idxc = idx + n;
    coltyp = idxc + n;
    idxp = coltyp + n;
    //
    //    Scale.
    //
    // Computing MAX
    d__1 = abs(*alpha), d__2 = abs(*beta);
    orgnrm = max(d__1,d__2);
    d__[*nl + 1] = 0.;
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((d__1 = d__[i__], abs(d__1)) > orgnrm) {
	    orgnrm = (d__1 = d__[i__], abs(d__1));
	}
// L10:
    }
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b7, &n, &c__1, &d__[1], &n, info);
    *alpha /= orgnrm;
    *beta /= orgnrm;
    //
    //    Deflate singular values.
    //
    dlasd2_(nl, nr, sqre, &k, &d__[1], &work[iz], alpha, beta, &u[u_offset],
	    ldu, &vt[vt_offset], ldvt, &work[isigma], &work[iu2], &ldu2, &
	    work[ivt2], &ldvt2, &iwork[idxp], &iwork[idx], &iwork[idxc], &
	    idxq[1], &iwork[coltyp], info);
    //
    //    Solve Secular Equation and update singular vectors.
    //
    ldq = k;
    dlasd3_(nl, nr, sqre, &k, &d__[1], &work[iq], &ldq, &work[isigma], &u[
	    u_offset], ldu, &work[iu2], &ldu2, &vt[vt_offset], ldvt, &work[
	    ivt2], &ldvt2, &iwork[idxc], &iwork[coltyp], &work[iz], info);
    //
    //    Report the convergence failure.
    //
    if (*info != 0) {
	return 0;
    }
    //
    //    Unscale.
    //
    dlascl_("G", &c__0, &c__0, &c_b7, &orgnrm, &n, &c__1, &d__[1], &n, info);
    //
    //    Prepare the IDXQ sorting permutation.
    //
    n1 = k;
    n2 = n - k;
    dlamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &idxq[1]);
    return 0;
    //
    //    End of DLASD1
    //
} // dlasd1_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD2 merges the two sets of singular values together into a single sorted set. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD2( NL, NR, SQRE, K, D, Z, ALPHA, BETA, U, LDU, VT,
//                         LDVT, DSIGMA, U2, LDU2, VT2, LDVT2, IDXP, IDX,
//                         IDXC, IDXQ, COLTYP, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, K, LDU, LDU2, LDVT, LDVT2, NL, NR, SQRE
//      DOUBLE PRECISION   ALPHA, BETA
//      ..
//      .. Array Arguments ..
//      INTEGER            COLTYP( * ), IDX( * ), IDXC( * ), IDXP( * ),
//     $                   IDXQ( * )
//      DOUBLE PRECISION   D( * ), DSIGMA( * ), U( LDU, * ),
//     $                   U2( LDU2, * ), VT( LDVT, * ), VT2( LDVT2, * ),
//     $                   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASD2 merges the two sets of singular values together into a single
//> sorted set.  Then it tries to deflate the size of the problem.
//> There are two ways in which deflation can occur:  when two or more
//> singular values are close together or if there is a tiny entry in the
//> Z vector.  For each such occurrence the order of the related secular
//> equation problem is reduced by one.
//>
//> DLASD2 is called from DLASD1.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] NL
//> \verbatim
//>          NL is INTEGER
//>         The row dimension of the upper block.  NL >= 1.
//> \endverbatim
//>
//> \param[in] NR
//> \verbatim
//>          NR is INTEGER
//>         The row dimension of the lower block.  NR >= 1.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>         = 0: the lower block is an NR-by-NR square matrix.
//>         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
//>
//>         The bidiagonal matrix has N = NL + NR + 1 rows and
//>         M = N + SQRE >= N columns.
//> \endverbatim
//>
//> \param[out] K
//> \verbatim
//>          K is INTEGER
//>         Contains the dimension of the non-deflated matrix,
//>         This is the order of the related secular equation. 1 <= K <=N.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension(N)
//>         On entry D contains the singular values of the two submatrices
//>         to be combined.  On exit D contains the trailing (N-K) updated
//>         singular values (those which were deflated) sorted into
//>         increasing order.
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension(N)
//>         On exit Z contains the updating row vector in the secular
//>         equation.
//> \endverbatim
//>
//> \param[in] ALPHA
//> \verbatim
//>          ALPHA is DOUBLE PRECISION
//>         Contains the diagonal element associated with the added row.
//> \endverbatim
//>
//> \param[in] BETA
//> \verbatim
//>          BETA is DOUBLE PRECISION
//>         Contains the off-diagonal element associated with the added
//>         row.
//> \endverbatim
//>
//> \param[in,out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension(LDU,N)
//>         On entry U contains the left singular vectors of two
//>         submatrices in the two square blocks with corners at (1,1),
//>         (NL, NL), and (NL+2, NL+2), (N,N).
//>         On exit U contains the trailing (N-K) updated left singular
//>         vectors (those which were deflated) in its last N-K columns.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>         The leading dimension of the array U.  LDU >= N.
//> \endverbatim
//>
//> \param[in,out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension(LDVT,M)
//>         On entry VT**T contains the right singular vectors of two
//>         submatrices in the two square blocks with corners at (1,1),
//>         (NL+1, NL+1), and (NL+2, NL+2), (M,M).
//>         On exit VT**T contains the trailing (N-K) updated right singular
//>         vectors (those which were deflated) in its last N-K columns.
//>         In case SQRE =1, the last row of VT spans the right null
//>         space.
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>         The leading dimension of the array VT.  LDVT >= M.
//> \endverbatim
//>
//> \param[out] DSIGMA
//> \verbatim
//>          DSIGMA is DOUBLE PRECISION array, dimension (N)
//>         Contains a copy of the diagonal elements (K-1 singular values
//>         and one zero) in the secular equation.
//> \endverbatim
//>
//> \param[out] U2
//> \verbatim
//>          U2 is DOUBLE PRECISION array, dimension(LDU2,N)
//>         Contains a copy of the first K-1 left singular vectors which
//>         will be used by DLASD3 in a matrix multiply (DGEMM) to solve
//>         for the new left singular vectors. U2 is arranged into four
//>         blocks. The first block contains a column with 1 at NL+1 and
//>         zero everywhere else; the second block contains non-zero
//>         entries only at and above NL; the third contains non-zero
//>         entries only below NL+1; and the fourth is dense.
//> \endverbatim
//>
//> \param[in] LDU2
//> \verbatim
//>          LDU2 is INTEGER
//>         The leading dimension of the array U2.  LDU2 >= N.
//> \endverbatim
//>
//> \param[out] VT2
//> \verbatim
//>          VT2 is DOUBLE PRECISION array, dimension(LDVT2,N)
//>         VT2**T contains a copy of the first K right singular vectors
//>         which will be used by DLASD3 in a matrix multiply (DGEMM) to
//>         solve for the new right singular vectors. VT2 is arranged into
//>         three blocks. The first block contains a row that corresponds
//>         to the special 0 diagonal element in SIGMA; the second block
//>         contains non-zeros only at and before NL +1; the third block
//>         contains non-zeros only at and after  NL +2.
//> \endverbatim
//>
//> \param[in] LDVT2
//> \verbatim
//>          LDVT2 is INTEGER
//>         The leading dimension of the array VT2.  LDVT2 >= M.
//> \endverbatim
//>
//> \param[out] IDXP
//> \verbatim
//>          IDXP is INTEGER array, dimension(N)
//>         This will contain the permutation used to place deflated
//>         values of D at the end of the array. On output IDXP(2:K)
//>         points to the nondeflated D-values and IDXP(K+1:N)
//>         points to the deflated singular values.
//> \endverbatim
//>
//> \param[out] IDX
//> \verbatim
//>          IDX is INTEGER array, dimension(N)
//>         This will contain the permutation used to sort the contents of
//>         D into ascending order.
//> \endverbatim
//>
//> \param[out] IDXC
//> \verbatim
//>          IDXC is INTEGER array, dimension(N)
//>         This will contain the permutation used to arrange the columns
//>         of the deflated U matrix into three groups:  the first group
//>         contains non-zero entries only at and above NL, the second
//>         contains non-zero entries only below NL+2, and the third is
//>         dense.
//> \endverbatim
//>
//> \param[in,out] IDXQ
//> \verbatim
//>          IDXQ is INTEGER array, dimension(N)
//>         This contains the permutation which separately sorts the two
//>         sub-problems in D into ascending order.  Note that entries in
//>         the first hlaf of this permutation must first be moved one
//>         position backward; and entries in the second half
//>         must first have NL+1 added to their values.
//> \endverbatim
//>
//> \param[out] COLTYP
//> \verbatim
//>          COLTYP is INTEGER array, dimension(N)
//>         As workspace, this will contain a label which will indicate
//>         which of the following types a column in the U2 matrix or a
//>         row in the VT2 matrix is:
//>         1 : non-zero in the upper half only
//>         2 : non-zero in the lower half only
//>         3 : dense
//>         4 : deflated
//>
//>         On exit, it is an array of dimension 4, with COLTYP(I) being
//>         the dimension of the I-th type columns.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd2_(int *nl, int *nr, int *sqre, int *k, double *d__,
	 double *z__, double *alpha, double *beta, double *u, int *ldu,
	double *vt, int *ldvt, double *dsigma, double *u2, int *ldu2, double *
	vt2, int *ldvt2, int *idxp, int *idx, int *idxc, int *idxq, int *
	coltyp, int *info)
{
    // Table of constant values
    int c__1 = 1;
    double c_b30 = 0.;

    // System generated locals
    int u_dim1, u_offset, u2_dim1, u2_offset, vt_dim1, vt_offset, vt2_dim1,
	    vt2_offset, i__1;
    double d__1, d__2;

    // Local variables
    double c__;
    int i__, j, m, n;
    double s;
    int k2;
    double z1;
    int ct, jp;
    double eps, tau, tol;
    int psm[4], nlp1, nlp2, idxi, idxj;
    extern /* Subroutine */ int drot_(int *, double *, int *, double *, int *,
	     double *, double *);
    int ctot[4], idxjp;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int jprev;
    extern double dlapy2_(double *, double *), dlamch_(char *);
    extern /* Subroutine */ int dlamrg_(int *, int *, double *, int *, int *,
	    int *), dlacpy_(char *, int *, int *, double *, int *, double *,
	    int *), dlaset_(char *, int *, int *, double *, double *, double *
	    , int *), xerbla_(char *, int *);
    double hlftol;

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
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

    // Function Body
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
    //
    //    Generate the first part of the vector Z; and move the singular
    //    values in the first part of D one position backward.
    //
    z1 = *alpha * vt[nlp1 + nlp1 * vt_dim1];
    z__[1] = z1;
    for (i__ = *nl; i__ >= 1; --i__) {
	z__[i__ + 1] = *alpha * vt[i__ + nlp1 * vt_dim1];
	d__[i__ + 1] = d__[i__];
	idxq[i__ + 1] = idxq[i__] + 1;
// L10:
    }
    //
    //    Generate the second part of the vector Z.
    //
    i__1 = m;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	z__[i__] = *beta * vt[i__ + nlp2 * vt_dim1];
// L20:
    }
    //
    //    Initialize some reference arrays.
    //
    i__1 = nlp1;
    for (i__ = 2; i__ <= i__1; ++i__) {
	coltyp[i__] = 1;
// L30:
    }
    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	coltyp[i__] = 2;
// L40:
    }
    //
    //    Sort the singular values into increasing order
    //
    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	idxq[i__] += nlp1;
// L50:
    }
    //
    //    DSIGMA, IDXC, IDXC, and the first column of U2
    //    are used as storage space.
    //
    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	dsigma[i__] = d__[idxq[i__]];
	u2[i__ + u2_dim1] = z__[idxq[i__]];
	idxc[i__] = coltyp[idxq[i__]];
// L60:
    }
    dlamrg_(nl, nr, &dsigma[2], &c__1, &c__1, &idx[2]);
    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	idxi = idx[i__] + 1;
	d__[i__] = dsigma[idxi];
	z__[i__] = u2[idxi + u2_dim1];
	coltyp[i__] = idxc[idxi];
// L70:
    }
    //
    //    Calculate the allowable deflation tolerance
    //
    eps = dlamch_("Epsilon");
    // Computing MAX
    d__1 = abs(*alpha), d__2 = abs(*beta);
    tol = max(d__1,d__2);
    // Computing MAX
    d__2 = (d__1 = d__[n], abs(d__1));
    tol = eps * 8. * max(d__2,tol);
    //
    //    There are 2 kinds of deflation -- first a value in the z-vector
    //    is small, second two (or more) singular values are very close
    //    together (their difference is small).
    //
    //    If the value in the z-vector is small, we simply permute the
    //    array so that the corresponding singular value is moved to the
    //    end.
    //
    //    If two values in the D-vector are close, we perform a two-sided
    //    rotation designed to make one of the corresponding z-vector
    //    entries zero, and then permute the array so that the deflated
    //    singular value is moved to the end.
    //
    //    If there are multiple singular values then the problem deflates.
    //    Here the number of equal singular values are found.  As each equal
    //    singular value is found, an elementary reflector is computed to
    //    rotate the corresponding singular subspace so that the
    //    corresponding components of Z are zero in this new basis.
    //
    *k = 1;
    k2 = n + 1;
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	if ((d__1 = z__[j], abs(d__1)) <= tol) {
	    //
	    //          Deflate due to small z component.
	    //
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
// L80:
    }
L90:
    j = jprev;
L100:
    ++j;
    if (j > n) {
	goto L110;
    }
    if ((d__1 = z__[j], abs(d__1)) <= tol) {
	//
	//       Deflate due to small z component.
	//
	--k2;
	idxp[k2] = j;
	coltyp[j] = 4;
    } else {
	//
	//       Check if singular values are close enough to allow deflation.
	//
	if ((d__1 = d__[j] - d__[jprev], abs(d__1)) <= tol) {
	    //
	    //          Deflation is possible.
	    //
	    s = z__[jprev];
	    c__ = z__[j];
	    //
	    //          Find sqrt(a**2+b**2) without overflow or
	    //          destructive underflow.
	    //
	    tau = dlapy2_(&c__, &s);
	    c__ /= tau;
	    s = -s / tau;
	    z__[j] = tau;
	    z__[jprev] = 0.;
	    //
	    //          Apply back the Givens rotation to the left and right
	    //          singular vector matrices.
	    //
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
    //
    //    Record the last singular value.
    //
    ++(*k);
    u2[*k + u2_dim1] = z__[jprev];
    dsigma[*k] = d__[jprev];
    idxp[*k] = jprev;
L120:
    //
    //    Count up the total number of the various types of columns, then
    //    form a permutation which positions the four column types into
    //    four groups of uniform structure (although one or more of these
    //    groups may be empty).
    //
    for (j = 1; j <= 4; ++j) {
	ctot[j - 1] = 0;
// L130:
    }
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	ct = coltyp[j];
	++ctot[ct - 1];
// L140:
    }
    //
    //    PSM(*) = Position in SubMatrix (of types 1 through 4)
    //
    psm[0] = 2;
    psm[1] = ctot[0] + 2;
    psm[2] = psm[1] + ctot[1];
    psm[3] = psm[2] + ctot[2];
    //
    //    Fill out the IDXC array so that the permutation which it induces
    //    will place all type-1 columns first, all type-2 columns next,
    //    then all type-3's, and finally all type-4's, starting from the
    //    second column. This applies similarly to the rows of VT.
    //
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	jp = idxp[j];
	ct = coltyp[jp];
	idxc[psm[ct - 1]] = j;
	++psm[ct - 1];
// L150:
    }
    //
    //    Sort the singular values and corresponding singular vectors into
    //    DSIGMA, U2, and VT2 respectively.  The singular values/vectors
    //    which were not deflated go into the first K slots of DSIGMA, U2,
    //    and VT2 respectively, while those which were deflated go into the
    //    last N - K slots, except that the first column/row will be treated
    //    separately.
    //
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
// L160:
    }
    //
    //    Determine DSIGMA(1), DSIGMA(2) and Z(1)
    //
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
    //
    //    Move the rest of the updating row to Z.
    //
    i__1 = *k - 1;
    dcopy_(&i__1, &u2[u2_dim1 + 2], &c__1, &z__[2], &c__1);
    //
    //    Determine the first column of U2, the first row of VT2 and the
    //    last row of VT.
    //
    dlaset_("A", &n, &c__1, &c_b30, &c_b30, &u2[u2_offset], ldu2);
    u2[nlp1 + u2_dim1] = 1.;
    if (m > n) {
	i__1 = nlp1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    vt[m + i__ * vt_dim1] = -s * vt[nlp1 + i__ * vt_dim1];
	    vt2[i__ * vt2_dim1 + 1] = c__ * vt[nlp1 + i__ * vt_dim1];
// L170:
	}
	i__1 = m;
	for (i__ = nlp2; i__ <= i__1; ++i__) {
	    vt2[i__ * vt2_dim1 + 1] = s * vt[m + i__ * vt_dim1];
	    vt[m + i__ * vt_dim1] = c__ * vt[m + i__ * vt_dim1];
// L180:
	}
    } else {
	dcopy_(&m, &vt[nlp1 + vt_dim1], ldvt, &vt2[vt2_dim1 + 1], ldvt2);
    }
    if (m > n) {
	dcopy_(&m, &vt[m + vt_dim1], ldvt, &vt2[m + vt2_dim1], ldvt2);
    }
    //
    //    The deflated singular values and their corresponding vectors go
    //    into the back of D, U, and V respectively.
    //
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
    //
    //    Copy CTOT into COLTYP for referencing in DLASD3.
    //
    for (j = 1; j <= 4; ++j) {
	coltyp[j] = ctot[j - 1];
// L190:
    }
    return 0;
    //
    //    End of DLASD2
    //
} // dlasd2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD3 finds all square roots of the roots of the secular equation, as defined by the values in D and Z, and then updates the singular vectors by matrix multiplication. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD3 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd3.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd3.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd3.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD3( NL, NR, SQRE, K, D, Q, LDQ, DSIGMA, U, LDU, U2,
//                         LDU2, VT, LDVT, VT2, LDVT2, IDXC, CTOT, Z,
//                         INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, K, LDQ, LDU, LDU2, LDVT, LDVT2, NL, NR,
//     $                   SQRE
//      ..
//      .. Array Arguments ..
//      INTEGER            CTOT( * ), IDXC( * )
//      DOUBLE PRECISION   D( * ), DSIGMA( * ), Q( LDQ, * ), U( LDU, * ),
//     $                   U2( LDU2, * ), VT( LDVT, * ), VT2( LDVT2, * ),
//     $                   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASD3 finds all the square roots of the roots of the secular
//> equation, as defined by the values in D and Z.  It makes the
//> appropriate calls to DLASD4 and then updates the singular
//> vectors by matrix multiplication.
//>
//> This code makes very mild assumptions about floating point
//> arithmetic. It will work on machines with a guard digit in
//> add/subtract, or on those binary machines without guard digits
//> which subtract like the Cray XMP, Cray YMP, Cray C 90, or Cray 2.
//> It could conceivably fail on hexadecimal or decimal machines
//> without guard digits, but we know of none.
//>
//> DLASD3 is called from DLASD1.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] NL
//> \verbatim
//>          NL is INTEGER
//>         The row dimension of the upper block.  NL >= 1.
//> \endverbatim
//>
//> \param[in] NR
//> \verbatim
//>          NR is INTEGER
//>         The row dimension of the lower block.  NR >= 1.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>         = 0: the lower block is an NR-by-NR square matrix.
//>         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
//>
//>         The bidiagonal matrix has N = NL + NR + 1 rows and
//>         M = N + SQRE >= N columns.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>         The size of the secular equation, 1 =< K = < N.
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension(K)
//>         On exit the square roots of the roots of the secular equation,
//>         in ascending order.
//> \endverbatim
//>
//> \param[out] Q
//> \verbatim
//>          Q is DOUBLE PRECISION array, dimension (LDQ,K)
//> \endverbatim
//>
//> \param[in] LDQ
//> \verbatim
//>          LDQ is INTEGER
//>         The leading dimension of the array Q.  LDQ >= K.
//> \endverbatim
//>
//> \param[in,out] DSIGMA
//> \verbatim
//>          DSIGMA is DOUBLE PRECISION array, dimension(K)
//>         The first K elements of this array contain the old roots
//>         of the deflated updating problem.  These are the poles
//>         of the secular equation.
//> \endverbatim
//>
//> \param[out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension (LDU, N)
//>         The last N - K columns of this matrix contain the deflated
//>         left singular vectors.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>         The leading dimension of the array U.  LDU >= N.
//> \endverbatim
//>
//> \param[in] U2
//> \verbatim
//>          U2 is DOUBLE PRECISION array, dimension (LDU2, N)
//>         The first K columns of this matrix contain the non-deflated
//>         left singular vectors for the split problem.
//> \endverbatim
//>
//> \param[in] LDU2
//> \verbatim
//>          LDU2 is INTEGER
//>         The leading dimension of the array U2.  LDU2 >= N.
//> \endverbatim
//>
//> \param[out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension (LDVT, M)
//>         The last M - K columns of VT**T contain the deflated
//>         right singular vectors.
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>         The leading dimension of the array VT.  LDVT >= N.
//> \endverbatim
//>
//> \param[in,out] VT2
//> \verbatim
//>          VT2 is DOUBLE PRECISION array, dimension (LDVT2, N)
//>         The first K columns of VT2**T contain the non-deflated
//>         right singular vectors for the split problem.
//> \endverbatim
//>
//> \param[in] LDVT2
//> \verbatim
//>          LDVT2 is INTEGER
//>         The leading dimension of the array VT2.  LDVT2 >= N.
//> \endverbatim
//>
//> \param[in] IDXC
//> \verbatim
//>          IDXC is INTEGER array, dimension ( N )
//>         The permutation used to arrange the columns of U (and rows of
//>         VT) into three groups:  the first group contains non-zero
//>         entries only at and above (or before) NL +1; the second
//>         contains non-zero entries only at and below (or after) NL+2;
//>         and the third is dense. The first column of U and the row of
//>         VT are treated separately, however.
//>
//>         The rows of the singular vectors found by DLASD4
//>         must be likewise permuted before the matrix multiplies can
//>         take place.
//> \endverbatim
//>
//> \param[in] CTOT
//> \verbatim
//>          CTOT is INTEGER array, dimension ( 4 )
//>         A count of the total number of the various types of columns
//>         in U (or rows in VT), as described in IDXC. The fourth column
//>         type is any column which has been deflated.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (K)
//>         The first K elements of this array contain the components
//>         of the deflation-adjusted updating row vector.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>         = 0:  successful exit.
//>         < 0:  if INFO = -i, the i-th argument had an illegal value.
//>         > 0:  if INFO = 1, a singular value did not converge
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd3_(int *nl, int *nr, int *sqre, int *k, double *d__,
	 double *q, int *ldq, double *dsigma, double *u, int *ldu, double *u2,
	 int *ldu2, double *vt, int *ldvt, double *vt2, int *ldvt2, int *idxc,
	 int *ctot, double *z__, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__0 = 0;
    double c_b13 = 1.;
    double c_b26 = 0.;

    // System generated locals
    int q_dim1, q_offset, u_dim1, u_offset, u2_dim1, u2_offset, vt_dim1,
	    vt_offset, vt2_dim1, vt2_offset, i__1, i__2;
    double d__1, d__2;

    // Local variables
    int i__, j, m, n, jc;
    double rho;
    int nlp1, nlp2, nrp1;
    double temp;
    extern double dnrm2_(int *, double *, int *);
    extern /* Subroutine */ int dgemm_(char *, char *, int *, int *, int *,
	    double *, double *, int *, double *, int *, double *, double *,
	    int *);
    int ctemp;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int ktemp;
    extern double dlamc3_(double *, double *);
    extern /* Subroutine */ int dlasd4_(int *, int *, double *, double *,
	    double *, double *, double *, double *, int *), dlascl_(char *,
	    int *, int *, double *, double *, int *, int *, double *, int *,
	    int *), dlacpy_(char *, int *, int *, double *, int *, double *,
	    int *), xerbla_(char *, int *);

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --dsigma;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    u2_dim1 = *ldu2;
    u2_offset = 1 + u2_dim1;
    u2 -= u2_offset;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    vt2_dim1 = *ldvt2;
    vt2_offset = 1 + vt2_dim1;
    vt2 -= vt2_offset;
    --idxc;
    --ctot;
    --z__;

    // Function Body
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
    nlp1 = *nl + 1;
    nlp2 = *nl + 2;
    if (*k < 1 || *k > n) {
	*info = -4;
    } else if (*ldq < *k) {
	*info = -7;
    } else if (*ldu < n) {
	*info = -10;
    } else if (*ldu2 < n) {
	*info = -12;
    } else if (*ldvt < m) {
	*info = -14;
    } else if (*ldvt2 < m) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASD3", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*k == 1) {
	d__[1] = abs(z__[1]);
	dcopy_(&m, &vt2[vt2_dim1 + 1], ldvt2, &vt[vt_dim1 + 1], ldvt);
	if (z__[1] > 0.) {
	    dcopy_(&n, &u2[u2_dim1 + 1], &c__1, &u[u_dim1 + 1], &c__1);
	} else {
	    i__1 = n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		u[i__ + u_dim1] = -u2[i__ + u2_dim1];
// L10:
	    }
	}
	return 0;
    }
    //
    //    Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can
    //    be computed with high relative accuracy (barring over/underflow).
    //    This is a problem on machines without a guard digit in
    //    add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
    //    The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I),
    //    which on any of these machines zeros out the bottommost
    //    bit of DSIGMA(I) if it is 1; this makes the subsequent
    //    subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation
    //    occurs. On binary machines with a guard digit (almost all
    //    machines) it does not change DSIGMA(I) at all. On hexadecimal
    //    and decimal machines with a guard digit, it slightly
    //    changes the bottommost bits of DSIGMA(I). It does not account
    //    for hexadecimal or decimal machines without guard digits
    //    (we know of none). We use a subroutine call to compute
    //    2*DSIGMA(I) to prevent optimizing compilers from eliminating
    //    this code.
    //
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dsigma[i__] = dlamc3_(&dsigma[i__], &dsigma[i__]) - dsigma[i__];
// L20:
    }
    //
    //    Keep a copy of Z.
    //
    dcopy_(k, &z__[1], &c__1, &q[q_offset], &c__1);
    //
    //    Normalize Z.
    //
    rho = dnrm2_(k, &z__[1], &c__1);
    dlascl_("G", &c__0, &c__0, &rho, &c_b13, k, &c__1, &z__[1], k, info);
    rho *= rho;
    //
    //    Find the new singular values.
    //
    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	dlasd4_(k, &j, &dsigma[1], &z__[1], &u[j * u_dim1 + 1], &rho, &d__[j],
		 &vt[j * vt_dim1 + 1], info);
	//
	//       If the zero finder fails, report the convergence failure.
	//
	if (*info != 0) {
	    return 0;
	}
// L30:
    }
    //
    //    Compute updated Z.
    //
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	z__[i__] = u[i__ + *k * u_dim1] * vt[i__ + *k * vt_dim1];
	i__2 = i__ - 1;
	for (j = 1; j <= i__2; ++j) {
	    z__[i__] *= u[i__ + j * u_dim1] * vt[i__ + j * vt_dim1] / (dsigma[
		    i__] - dsigma[j]) / (dsigma[i__] + dsigma[j]);
// L40:
	}
	i__2 = *k - 1;
	for (j = i__; j <= i__2; ++j) {
	    z__[i__] *= u[i__ + j * u_dim1] * vt[i__ + j * vt_dim1] / (dsigma[
		    i__] - dsigma[j + 1]) / (dsigma[i__] + dsigma[j + 1]);
// L50:
	}
	d__2 = sqrt((d__1 = z__[i__], abs(d__1)));
	z__[i__] = d_sign(&d__2, &q[i__ + q_dim1]);
// L60:
    }
    //
    //    Compute left singular vectors of the modified diagonal matrix,
    //    and store related information for the right singular vectors.
    //
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	vt[i__ * vt_dim1 + 1] = z__[1] / u[i__ * u_dim1 + 1] / vt[i__ *
		vt_dim1 + 1];
	u[i__ * u_dim1 + 1] = -1.;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    vt[j + i__ * vt_dim1] = z__[j] / u[j + i__ * u_dim1] / vt[j + i__
		    * vt_dim1];
	    u[j + i__ * u_dim1] = dsigma[j] * vt[j + i__ * vt_dim1];
// L70:
	}
	temp = dnrm2_(k, &u[i__ * u_dim1 + 1], &c__1);
	q[i__ * q_dim1 + 1] = u[i__ * u_dim1 + 1] / temp;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    jc = idxc[j];
	    q[j + i__ * q_dim1] = u[jc + i__ * u_dim1] / temp;
// L80:
	}
// L90:
    }
    //
    //    Update the left singular vector matrix.
    //
    if (*k == 2) {
	dgemm_("N", "N", &n, k, k, &c_b13, &u2[u2_offset], ldu2, &q[q_offset],
		 ldq, &c_b26, &u[u_offset], ldu);
	goto L100;
    }
    if (ctot[1] > 0) {
	dgemm_("N", "N", nl, k, &ctot[1], &c_b13, &u2[(u2_dim1 << 1) + 1],
		ldu2, &q[q_dim1 + 2], ldq, &c_b26, &u[u_dim1 + 1], ldu);
	if (ctot[3] > 0) {
	    ktemp = ctot[1] + 2 + ctot[2];
	    dgemm_("N", "N", nl, k, &ctot[3], &c_b13, &u2[ktemp * u2_dim1 + 1]
		    , ldu2, &q[ktemp + q_dim1], ldq, &c_b13, &u[u_dim1 + 1],
		    ldu);
	}
    } else if (ctot[3] > 0) {
	ktemp = ctot[1] + 2 + ctot[2];
	dgemm_("N", "N", nl, k, &ctot[3], &c_b13, &u2[ktemp * u2_dim1 + 1],
		ldu2, &q[ktemp + q_dim1], ldq, &c_b26, &u[u_dim1 + 1], ldu);
    } else {
	dlacpy_("F", nl, k, &u2[u2_offset], ldu2, &u[u_offset], ldu);
    }
    dcopy_(k, &q[q_dim1 + 1], ldq, &u[nlp1 + u_dim1], ldu);
    ktemp = ctot[1] + 2;
    ctemp = ctot[2] + ctot[3];
    dgemm_("N", "N", nr, k, &ctemp, &c_b13, &u2[nlp2 + ktemp * u2_dim1], ldu2,
	     &q[ktemp + q_dim1], ldq, &c_b26, &u[nlp2 + u_dim1], ldu);
    //
    //    Generate the right singular vectors.
    //
L100:
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	temp = dnrm2_(k, &vt[i__ * vt_dim1 + 1], &c__1);
	q[i__ + q_dim1] = vt[i__ * vt_dim1 + 1] / temp;
	i__2 = *k;
	for (j = 2; j <= i__2; ++j) {
	    jc = idxc[j];
	    q[i__ + j * q_dim1] = vt[jc + i__ * vt_dim1] / temp;
// L110:
	}
// L120:
    }
    //
    //    Update the right singular vector matrix.
    //
    if (*k == 2) {
	dgemm_("N", "N", k, &m, k, &c_b13, &q[q_offset], ldq, &vt2[vt2_offset]
		, ldvt2, &c_b26, &vt[vt_offset], ldvt);
	return 0;
    }
    ktemp = ctot[1] + 1;
    dgemm_("N", "N", k, &nlp1, &ktemp, &c_b13, &q[q_dim1 + 1], ldq, &vt2[
	    vt2_dim1 + 1], ldvt2, &c_b26, &vt[vt_dim1 + 1], ldvt);
    ktemp = ctot[1] + 2 + ctot[2];
    if (ktemp <= *ldvt2) {
	dgemm_("N", "N", k, &nlp1, &ctot[3], &c_b13, &q[ktemp * q_dim1 + 1],
		ldq, &vt2[ktemp + vt2_dim1], ldvt2, &c_b13, &vt[vt_dim1 + 1],
		ldvt);
    }
    ktemp = ctot[1] + 1;
    nrp1 = *nr + *sqre;
    if (ktemp > 1) {
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    q[i__ + ktemp * q_dim1] = q[i__ + q_dim1];
// L130:
	}
	i__1 = m;
	for (i__ = nlp2; i__ <= i__1; ++i__) {
	    vt2[ktemp + i__ * vt2_dim1] = vt2[i__ * vt2_dim1 + 1];
// L140:
	}
    }
    ctemp = ctot[2] + 1 + ctot[3];
    dgemm_("N", "N", k, &nrp1, &ctemp, &c_b13, &q[ktemp * q_dim1 + 1], ldq, &
	    vt2[ktemp + nlp2 * vt2_dim1], ldvt2, &c_b26, &vt[nlp2 * vt_dim1 +
	    1], ldvt);
    return 0;
    //
    //    End of DLASD3
    //
} // dlasd3_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD4 computes the square root of the i-th updated eigenvalue of a positive symmetric rank-one modification to a positive diagonal matrix. Used by dbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD4 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd4.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd4.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd4.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD4( N, I, D, Z, DELTA, RHO, SIGMA, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            I, INFO, N
//      DOUBLE PRECISION   RHO, SIGMA
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), DELTA( * ), WORK( * ), Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> This subroutine computes the square root of the I-th updated
//> eigenvalue of a positive symmetric rank-one modification to
//> a positive diagonal matrix whose entries are given as the squares
//> of the corresponding entries in the array d, and that
//>
//>        0 <= D(i) < D(j)  for  i < j
//>
//> and that RHO > 0. This is arranged by the calling routine, and is
//> no loss in generality.  The rank-one modified system is thus
//>
//>        diag( D ) * diag( D ) +  RHO * Z * Z_transpose.
//>
//> where we assume the Euclidean norm of Z is 1.
//>
//> The method consists of approximating the rational functions in the
//> secular equation by simpler interpolating rational functions.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>         The length of all arrays.
//> \endverbatim
//>
//> \param[in] I
//> \verbatim
//>          I is INTEGER
//>         The index of the eigenvalue to be computed.  1 <= I <= N.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension ( N )
//>         The original eigenvalues.  It is assumed that they are in
//>         order, 0 <= D(I) < D(J)  for I < J.
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( N )
//>         The components of the updating vector.
//> \endverbatim
//>
//> \param[out] DELTA
//> \verbatim
//>          DELTA is DOUBLE PRECISION array, dimension ( N )
//>         If N .ne. 1, DELTA contains (D(j) - sigma_I) in its  j-th
//>         component.  If N = 1, then DELTA(1) = 1.  The vector DELTA
//>         contains the information necessary to construct the
//>         (singular) eigenvectors.
//> \endverbatim
//>
//> \param[in] RHO
//> \verbatim
//>          RHO is DOUBLE PRECISION
//>         The scalar in the symmetric updating formula.
//> \endverbatim
//>
//> \param[out] SIGMA
//> \verbatim
//>          SIGMA is DOUBLE PRECISION
//>         The computed sigma_I, the I-th updated eigenvalue.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension ( N )
//>         If N .ne. 1, WORK contains (D(j) + sigma_I) in its  j-th
//>         component.  If N = 1, then WORK( 1 ) = 1.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>         = 0:  successful exit
//>         > 0:  if INFO = 1, the updating process failed.
//> \endverbatim
//
//> \par Internal Parameters:
// =========================
//>
//> \verbatim
//>  Logical variable ORGATI (origin-at-i?) is used for distinguishing
//>  whether D(i) or D(i+1) is treated as the origin.
//>
//>            ORGATI = .true.    origin at i
//>            ORGATI = .false.   origin at i+1
//>
//>  Logical variable SWTCH3 (switch-for-3-poles?) is for noting
//>  if we are working with THREE poles!
//>
//>  MAXIT is the maximum number of iterations allowed for each
//>  eigenvalue.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ren-Cang Li, Computer Science Division, University of California
//>     at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd4_(int *n, int *i__, double *d__, double *z__,
	double *delta, double *rho, double *sigma, double *work, int *info)
{
    // System generated locals
    int i__1;
    double d__1;

    // Local variables
    double a, b, c__;
    int j;
    double w, dd[3];
    int ii;
    double dw, zz[3];
    int ip1;
    double sq2, eta, phi, eps, tau, psi;
    int iim1, iip1;
    double tau2, dphi, sglb, dpsi, sgub;
    int iter;
    double temp, prew, temp1, temp2, dtiim, delsq, dtiip;
    int niter;
    double dtisq;
    int swtch;
    double dtnsq;
    extern /* Subroutine */ int dlaed6_(int *, int *, double *, double *,
	    double *, double *, double *, int *), dlasd5_(int *, double *,
	    double *, double *, double *, double *, double *);
    double delsq2, dtnsq1;
    int swtch3;
    extern double dlamch_(char *);
    int orgati;
    double erretm, dtipsq, rhoinv;
    int geomavg;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Since this routine is called in an inner loop, we do no argument
    //    checking.
    //
    //    Quick return for N=1 and 2.
    //
    // Parameter adjustments
    --work;
    --delta;
    --z__;
    --d__;

    // Function Body
    *info = 0;
    if (*n == 1) {
	//
	//       Presumably, I=1 upon entry
	//
	*sigma = sqrt(d__[1] * d__[1] + *rho * z__[1] * z__[1]);
	delta[1] = 1.;
	work[1] = 1.;
	return 0;
    }
    if (*n == 2) {
	dlasd5_(i__, &d__[1], &z__[1], &delta[1], rho, sigma, &work[1]);
	return 0;
    }
    //
    //    Compute machine epsilon
    //
    eps = dlamch_("Epsilon");
    rhoinv = 1. / *rho;
    tau2 = 0.;
    //
    //    The case I = N
    //
    if (*i__ == *n) {
	//
	//       Initialize some basic variables
	//
	ii = *n - 1;
	niter = 1;
	//
	//       Calculate initial guess
	//
	temp = *rho / 2.;
	//
	//       If ||Z||_2 is not one, then TEMP should be set to
	//       RHO * ||Z||_2^2 / TWO
	//
	temp1 = temp / (d__[*n] + sqrt(d__[*n] * d__[*n] + temp));
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] = d__[j] + d__[*n] + temp1;
	    delta[j] = d__[j] - d__[*n] - temp1;
// L10:
	}
	psi = 0.;
	i__1 = *n - 2;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / (delta[j] * work[j]);
// L20:
	}
	c__ = rhoinv + psi;
	w = c__ + z__[ii] * z__[ii] / (delta[ii] * work[ii]) + z__[*n] * z__[*
		n] / (delta[*n] * work[*n]);
	if (w <= 0.) {
	    temp1 = sqrt(d__[*n] * d__[*n] + *rho);
	    temp = z__[*n - 1] * z__[*n - 1] / ((d__[*n - 1] + temp1) * (d__[*
		    n] - d__[*n - 1] + *rho / (d__[*n] + temp1))) + z__[*n] *
		    z__[*n] / *rho;
	    //
	    //          The following TAU2 is to approximate
	    //          SIGMA_n^2 - D( N )*D( N )
	    //
	    if (c__ <= temp) {
		tau = *rho;
	    } else {
		delsq = (d__[*n] - d__[*n - 1]) * (d__[*n] + d__[*n - 1]);
		a = -c__ * delsq + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*
			n];
		b = z__[*n] * z__[*n] * delsq;
		if (a < 0.) {
		    tau2 = b * 2. / (sqrt(a * a + b * 4. * c__) - a);
		} else {
		    tau2 = (a + sqrt(a * a + b * 4. * c__)) / (c__ * 2.);
		}
		tau = tau2 / (d__[*n] + sqrt(d__[*n] * d__[*n] + tau2));
	    }
	    //
	    //          It can be proved that
	    //              D(N)^2+RHO/2 <= SIGMA_n^2 < D(N)^2+TAU2 <= D(N)^2+RHO
	    //
	} else {
	    delsq = (d__[*n] - d__[*n - 1]) * (d__[*n] + d__[*n - 1]);
	    a = -c__ * delsq + z__[*n - 1] * z__[*n - 1] + z__[*n] * z__[*n];
	    b = z__[*n] * z__[*n] * delsq;
	    //
	    //          The following TAU2 is to approximate
	    //          SIGMA_n^2 - D( N )*D( N )
	    //
	    if (a < 0.) {
		tau2 = b * 2. / (sqrt(a * a + b * 4. * c__) - a);
	    } else {
		tau2 = (a + sqrt(a * a + b * 4. * c__)) / (c__ * 2.);
	    }
	    tau = tau2 / (d__[*n] + sqrt(d__[*n] * d__[*n] + tau2));
	    //
	    //          It can be proved that
	    //          D(N)^2 < D(N)^2+TAU2 < SIGMA(N)^2 < D(N)^2+RHO/2
	    //
	}
	//
	//       The following TAU is to approximate SIGMA_n - D( N )
	//
	//        TAU = TAU2 / ( D( N )+SQRT( D( N )*D( N )+TAU2 ) )
	//
	*sigma = d__[*n] + tau;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] = d__[j] - d__[*n] - tau;
	    work[j] = d__[j] + d__[*n] + tau;
// L30:
	}
	//
	//       Evaluate PSI and the derivative DPSI
	//
	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (delta[j] * work[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
// L40:
	}
	erretm = abs(erretm);
	//
	//       Evaluate PHI and the derivative DPHI
	//
	temp = z__[*n] / (delta[*n] * work[*n]);
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8. + erretm - phi + rhoinv;
	//   $          + ABS( TAU2 )*( DPSI+DPHI )
	//
	w = rhoinv + phi + psi;
	//
	//       Test for convergence
	//
	if (abs(w) <= eps * erretm) {
	    goto L240;
	}
	//
	//       Calculate the new step
	//
	++niter;
	dtnsq1 = work[*n - 1] * delta[*n - 1];
	dtnsq = work[*n] * delta[*n];
	c__ = w - dtnsq1 * dpsi - dtnsq * dphi;
	a = (dtnsq + dtnsq1) * w - dtnsq * dtnsq1 * (dpsi + dphi);
	b = dtnsq * dtnsq1 * w;
	if (c__ < 0.) {
	    c__ = abs(c__);
	}
	if (c__ == 0.) {
	    eta = *rho - *sigma * *sigma;
	} else if (a >= 0.) {
	    eta = (a + sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (c__
		    * 2.);
	} else {
	    eta = b * 2. / (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))
		    );
	}
	//
	//       Note, eta should be positive if w is negative, and
	//       eta should be negative otherwise. However,
	//       if for some reason caused by roundoff, eta*w > 0,
	//       we simply use one Newton step instead. This way
	//       will guarantee eta*w < 0.
	//
	if (w * eta > 0.) {
	    eta = -w / (dpsi + dphi);
	}
	temp = eta - dtnsq;
	if (temp > *rho) {
	    eta = *rho + dtnsq;
	}
	eta /= *sigma + sqrt(eta + *sigma * *sigma);
	tau += eta;
	*sigma += eta;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    delta[j] -= eta;
	    work[j] += eta;
// L50:
	}
	//
	//       Evaluate PSI and the derivative DPSI
	//
	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = ii;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
// L60:
	}
	erretm = abs(erretm);
	//
	//       Evaluate PHI and the derivative DPHI
	//
	tau2 = work[*n] * delta[*n];
	temp = z__[*n] / tau2;
	phi = z__[*n] * temp;
	dphi = temp * temp;
	erretm = (-phi - psi) * 8. + erretm - phi + rhoinv;
	//   $          + ABS( TAU2 )*( DPSI+DPHI )
	//
	w = rhoinv + phi + psi;
	//
	//       Main loop to update the values of the array   DELTA
	//
	iter = niter + 1;
	for (niter = iter; niter <= 400; ++niter) {
	    //
	    //          Test for convergence
	    //
	    if (abs(w) <= eps * erretm) {
		goto L240;
	    }
	    //
	    //          Calculate the new step
	    //
	    dtnsq1 = work[*n - 1] * delta[*n - 1];
	    dtnsq = work[*n] * delta[*n];
	    c__ = w - dtnsq1 * dpsi - dtnsq * dphi;
	    a = (dtnsq + dtnsq1) * w - dtnsq1 * dtnsq * (dpsi + dphi);
	    b = dtnsq1 * dtnsq * w;
	    if (a >= 0.) {
		eta = (a + sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (
			c__ * 2.);
	    } else {
		eta = b * 2. / (a - sqrt((d__1 = a * a - b * 4. * c__, abs(
			d__1))));
	    }
	    //
	    //          Note, eta should be positive if w is negative, and
	    //          eta should be negative otherwise. However,
	    //          if for some reason caused by roundoff, eta*w > 0,
	    //          we simply use one Newton step instead. This way
	    //          will guarantee eta*w < 0.
	    //
	    if (w * eta > 0.) {
		eta = -w / (dpsi + dphi);
	    }
	    temp = eta - dtnsq;
	    if (temp <= 0.) {
		eta /= 2.;
	    }
	    eta /= *sigma + sqrt(eta + *sigma * *sigma);
	    tau += eta;
	    *sigma += eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		delta[j] -= eta;
		work[j] += eta;
// L70:
	    }
	    //
	    //          Evaluate PSI and the derivative DPSI
	    //
	    dpsi = 0.;
	    psi = 0.;
	    erretm = 0.;
	    i__1 = ii;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / (work[j] * delta[j]);
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
// L80:
	    }
	    erretm = abs(erretm);
	    //
	    //          Evaluate PHI and the derivative DPHI
	    //
	    tau2 = work[*n] * delta[*n];
	    temp = z__[*n] / tau2;
	    phi = z__[*n] * temp;
	    dphi = temp * temp;
	    erretm = (-phi - psi) * 8. + erretm - phi + rhoinv;
	    //   $             + ABS( TAU2 )*( DPSI+DPHI )
	    //
	    w = rhoinv + phi + psi;
// L90:
	}
	//
	//       Return with INFO = 1, NITER = MAXIT and not converged
	//
	*info = 1;
	goto L240;
	//
	//       End for the case I = N
	//
    } else {
	//
	//       The case for I < N
	//
	niter = 1;
	ip1 = *i__ + 1;
	//
	//       Calculate initial guess
	//
	delsq = (d__[ip1] - d__[*i__]) * (d__[ip1] + d__[*i__]);
	delsq2 = delsq / 2.;
	sq2 = sqrt((d__[*i__] * d__[*i__] + d__[ip1] * d__[ip1]) / 2.);
	temp = delsq2 / (d__[*i__] + sq2);
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] = d__[j] + d__[*i__] + temp;
	    delta[j] = d__[j] - d__[*i__] - temp;
// L100:
	}
	psi = 0.;
	i__1 = *i__ - 1;
	for (j = 1; j <= i__1; ++j) {
	    psi += z__[j] * z__[j] / (work[j] * delta[j]);
// L110:
	}
	phi = 0.;
	i__1 = *i__ + 2;
	for (j = *n; j >= i__1; --j) {
	    phi += z__[j] * z__[j] / (work[j] * delta[j]);
// L120:
	}
	c__ = rhoinv + psi + phi;
	w = c__ + z__[*i__] * z__[*i__] / (work[*i__] * delta[*i__]) + z__[
		ip1] * z__[ip1] / (work[ip1] * delta[ip1]);
	geomavg = FALSE_;
	if (w > 0.) {
	    //
	    //          d(i)^2 < the ith sigma^2 < (d(i)^2+d(i+1)^2)/2
	    //
	    //          We choose d(i) as origin.
	    //
	    orgati = TRUE_;
	    ii = *i__;
	    sglb = 0.;
	    sgub = delsq2 / (d__[*i__] + sq2);
	    a = c__ * delsq + z__[*i__] * z__[*i__] + z__[ip1] * z__[ip1];
	    b = z__[*i__] * z__[*i__] * delsq;
	    if (a > 0.) {
		tau2 = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__, abs(
			d__1))));
	    } else {
		tau2 = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) /
			(c__ * 2.);
	    }
	    //
	    //          TAU2 now is an estimation of SIGMA^2 - D( I )^2. The
	    //          following, however, is the corresponding estimation of
	    //          SIGMA - D( I ).
	    //
	    tau = tau2 / (d__[*i__] + sqrt(d__[*i__] * d__[*i__] + tau2));
	    temp = sqrt(eps);
	    if (d__[*i__] <= temp * d__[ip1] && (d__1 = z__[*i__], abs(d__1))
		    <= temp && d__[*i__] > 0.) {
		// Computing MIN
		d__1 = d__[*i__] * 10.;
		tau = min(d__1,sgub);
		geomavg = TRUE_;
	    }
	} else {
	    //
	    //          (d(i)^2+d(i+1)^2)/2 <= the ith sigma^2 < d(i+1)^2/2
	    //
	    //          We choose d(i+1) as origin.
	    //
	    orgati = FALSE_;
	    ii = ip1;
	    sglb = -delsq2 / (d__[ii] + sq2);
	    sgub = 0.;
	    a = c__ * delsq - z__[*i__] * z__[*i__] - z__[ip1] * z__[ip1];
	    b = z__[ip1] * z__[ip1] * delsq;
	    if (a < 0.) {
		tau2 = b * 2. / (a - sqrt((d__1 = a * a + b * 4. * c__, abs(
			d__1))));
	    } else {
		tau2 = -(a + sqrt((d__1 = a * a + b * 4. * c__, abs(d__1)))) /
			 (c__ * 2.);
	    }
	    //
	    //          TAU2 now is an estimation of SIGMA^2 - D( IP1 )^2. The
	    //          following, however, is the corresponding estimation of
	    //          SIGMA - D( IP1 ).
	    //
	    tau = tau2 / (d__[ip1] + sqrt((d__1 = d__[ip1] * d__[ip1] + tau2,
		    abs(d__1))));
	}
	*sigma = d__[ii] + tau;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] = d__[j] + d__[ii] + tau;
	    delta[j] = d__[j] - d__[ii] - tau;
// L130:
	}
	iim1 = ii - 1;
	iip1 = ii + 1;
	//
	//       Evaluate PSI and the derivative DPSI
	//
	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
// L150:
	}
	erretm = abs(erretm);
	//
	//       Evaluate PHI and the derivative DPHI
	//
	dphi = 0.;
	phi = 0.;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
// L160:
	}
	w = rhoinv + phi + psi;
	//
	//       W is the value of the secular function with
	//       its ii-th element removed.
	//
	swtch3 = FALSE_;
	if (orgati) {
	    if (w < 0.) {
		swtch3 = TRUE_;
	    }
	} else {
	    if (w > 0.) {
		swtch3 = TRUE_;
	    }
	}
	if (ii == 1 || ii == *n) {
	    swtch3 = FALSE_;
	}
	temp = z__[ii] / (work[ii] * delta[ii]);
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w += temp;
	erretm = (phi - psi) * 8. + erretm + rhoinv * 2. + abs(temp) * 3.;
	//   $          + ABS( TAU2 )*DW
	//
	//       Test for convergence
	//
	if (abs(w) <= eps * erretm) {
	    goto L240;
	}
	if (w <= 0.) {
	    sglb = max(sglb,tau);
	} else {
	    sgub = min(sgub,tau);
	}
	//
	//       Calculate the new step
	//
	++niter;
	if (! swtch3) {
	    dtipsq = work[ip1] * delta[ip1];
	    dtisq = work[*i__] * delta[*i__];
	    if (orgati) {
		// Computing 2nd power
		d__1 = z__[*i__] / dtisq;
		c__ = w - dtipsq * dw + delsq * (d__1 * d__1);
	    } else {
		// Computing 2nd power
		d__1 = z__[ip1] / dtipsq;
		c__ = w - dtisq * dw - delsq * (d__1 * d__1);
	    }
	    a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
	    b = dtipsq * dtisq * w;
	    if (c__ == 0.) {
		if (a == 0.) {
		    if (orgati) {
			a = z__[*i__] * z__[*i__] + dtipsq * dtipsq * (dpsi +
				dphi);
		    } else {
			a = z__[ip1] * z__[ip1] + dtisq * dtisq * (dpsi +
				dphi);
		    }
		}
		eta = b / a;
	    } else if (a <= 0.) {
		eta = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1)))) / (
			c__ * 2.);
	    } else {
		eta = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__, abs(
			d__1))));
	    }
	} else {
	    //
	    //          Interpolation using THREE most relevant poles
	    //
	    dtiim = work[iim1] * delta[iim1];
	    dtiip = work[iip1] * delta[iip1];
	    temp = rhoinv + psi + phi;
	    if (orgati) {
		temp1 = z__[iim1] / dtiim;
		temp1 *= temp1;
		c__ = temp - dtiip * (dpsi + dphi) - (d__[iim1] - d__[iip1]) *
			 (d__[iim1] + d__[iip1]) * temp1;
		zz[0] = z__[iim1] * z__[iim1];
		if (dpsi < temp1) {
		    zz[2] = dtiip * dtiip * dphi;
		} else {
		    zz[2] = dtiip * dtiip * (dpsi - temp1 + dphi);
		}
	    } else {
		temp1 = z__[iip1] / dtiip;
		temp1 *= temp1;
		c__ = temp - dtiim * (dpsi + dphi) - (d__[iip1] - d__[iim1]) *
			 (d__[iim1] + d__[iip1]) * temp1;
		if (dphi < temp1) {
		    zz[0] = dtiim * dtiim * dpsi;
		} else {
		    zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
		}
		zz[2] = z__[iip1] * z__[iip1];
	    }
	    zz[1] = z__[ii] * z__[ii];
	    dd[0] = dtiim;
	    dd[1] = delta[ii] * work[ii];
	    dd[2] = dtiip;
	    dlaed6_(&niter, &orgati, &c__, dd, zz, &w, &eta, info);
	    if (*info != 0) {
		//
		//             If INFO is not 0, i.e., DLAED6 failed, switch back
		//             to 2 pole interpolation.
		//
		swtch3 = FALSE_;
		*info = 0;
		dtipsq = work[ip1] * delta[ip1];
		dtisq = work[*i__] * delta[*i__];
		if (orgati) {
		    // Computing 2nd power
		    d__1 = z__[*i__] / dtisq;
		    c__ = w - dtipsq * dw + delsq * (d__1 * d__1);
		} else {
		    // Computing 2nd power
		    d__1 = z__[ip1] / dtipsq;
		    c__ = w - dtisq * dw - delsq * (d__1 * d__1);
		}
		a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
		b = dtipsq * dtisq * w;
		if (c__ == 0.) {
		    if (a == 0.) {
			if (orgati) {
			    a = z__[*i__] * z__[*i__] + dtipsq * dtipsq * (
				    dpsi + dphi);
			} else {
			    a = z__[ip1] * z__[ip1] + dtisq * dtisq * (dpsi +
				    dphi);
			}
		    }
		    eta = b / a;
		} else if (a <= 0.) {
		    eta = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1))))
			     / (c__ * 2.);
		} else {
		    eta = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__,
			    abs(d__1))));
		}
	    }
	}
	//
	//       Note, eta should be positive if w is negative, and
	//       eta should be negative otherwise. However,
	//       if for some reason caused by roundoff, eta*w > 0,
	//       we simply use one Newton step instead. This way
	//       will guarantee eta*w < 0.
	//
	if (w * eta >= 0.) {
	    eta = -w / dw;
	}
	eta /= *sigma + sqrt(*sigma * *sigma + eta);
	temp = tau + eta;
	if (temp > sgub || temp < sglb) {
	    if (w < 0.) {
		eta = (sgub - tau) / 2.;
	    } else {
		eta = (sglb - tau) / 2.;
	    }
	    if (geomavg) {
		if (w < 0.) {
		    if (tau > 0.) {
			eta = sqrt(sgub * tau) - tau;
		    }
		} else {
		    if (sglb > 0.) {
			eta = sqrt(sglb * tau) - tau;
		    }
		}
	    }
	}
	prew = w;
	tau += eta;
	*sigma += eta;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    work[j] += eta;
	    delta[j] -= eta;
// L170:
	}
	//
	//       Evaluate PSI and the derivative DPSI
	//
	dpsi = 0.;
	psi = 0.;
	erretm = 0.;
	i__1 = iim1;
	for (j = 1; j <= i__1; ++j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    psi += z__[j] * temp;
	    dpsi += temp * temp;
	    erretm += psi;
// L180:
	}
	erretm = abs(erretm);
	//
	//       Evaluate PHI and the derivative DPHI
	//
	dphi = 0.;
	phi = 0.;
	i__1 = iip1;
	for (j = *n; j >= i__1; --j) {
	    temp = z__[j] / (work[j] * delta[j]);
	    phi += z__[j] * temp;
	    dphi += temp * temp;
	    erretm += phi;
// L190:
	}
	tau2 = work[ii] * delta[ii];
	temp = z__[ii] / tau2;
	dw = dpsi + dphi + temp * temp;
	temp = z__[ii] * temp;
	w = rhoinv + phi + psi + temp;
	erretm = (phi - psi) * 8. + erretm + rhoinv * 2. + abs(temp) * 3.;
	//   $          + ABS( TAU2 )*DW
	//
	swtch = FALSE_;
	if (orgati) {
	    if (-w > abs(prew) / 10.) {
		swtch = TRUE_;
	    }
	} else {
	    if (w > abs(prew) / 10.) {
		swtch = TRUE_;
	    }
	}
	//
	//       Main loop to update the values of the array   DELTA and WORK
	//
	iter = niter + 1;
	for (niter = iter; niter <= 400; ++niter) {
	    //
	    //          Test for convergence
	    //
	    if (abs(w) <= eps * erretm) {
		//    $          .OR. (SGUB-SGLB).LE.EIGHT*ABS(SGUB+SGLB) ) THEN
		goto L240;
	    }
	    if (w <= 0.) {
		sglb = max(sglb,tau);
	    } else {
		sgub = min(sgub,tau);
	    }
	    //
	    //          Calculate the new step
	    //
	    if (! swtch3) {
		dtipsq = work[ip1] * delta[ip1];
		dtisq = work[*i__] * delta[*i__];
		if (! swtch) {
		    if (orgati) {
			// Computing 2nd power
			d__1 = z__[*i__] / dtisq;
			c__ = w - dtipsq * dw + delsq * (d__1 * d__1);
		    } else {
			// Computing 2nd power
			d__1 = z__[ip1] / dtipsq;
			c__ = w - dtisq * dw - delsq * (d__1 * d__1);
		    }
		} else {
		    temp = z__[ii] / (work[ii] * delta[ii]);
		    if (orgati) {
			dpsi += temp * temp;
		    } else {
			dphi += temp * temp;
		    }
		    c__ = w - dtisq * dpsi - dtipsq * dphi;
		}
		a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
		b = dtipsq * dtisq * w;
		if (c__ == 0.) {
		    if (a == 0.) {
			if (! swtch) {
			    if (orgati) {
				a = z__[*i__] * z__[*i__] + dtipsq * dtipsq *
					(dpsi + dphi);
			    } else {
				a = z__[ip1] * z__[ip1] + dtisq * dtisq * (
					dpsi + dphi);
			    }
			} else {
			    a = dtisq * dtisq * dpsi + dtipsq * dtipsq * dphi;
			}
		    }
		    eta = b / a;
		} else if (a <= 0.) {
		    eta = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(d__1))))
			     / (c__ * 2.);
		} else {
		    eta = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__,
			    abs(d__1))));
		}
	    } else {
		//
		//             Interpolation using THREE most relevant poles
		//
		dtiim = work[iim1] * delta[iim1];
		dtiip = work[iip1] * delta[iip1];
		temp = rhoinv + psi + phi;
		if (swtch) {
		    c__ = temp - dtiim * dpsi - dtiip * dphi;
		    zz[0] = dtiim * dtiim * dpsi;
		    zz[2] = dtiip * dtiip * dphi;
		} else {
		    if (orgati) {
			temp1 = z__[iim1] / dtiim;
			temp1 *= temp1;
			temp2 = (d__[iim1] - d__[iip1]) * (d__[iim1] + d__[
				iip1]) * temp1;
			c__ = temp - dtiip * (dpsi + dphi) - temp2;
			zz[0] = z__[iim1] * z__[iim1];
			if (dpsi < temp1) {
			    zz[2] = dtiip * dtiip * dphi;
			} else {
			    zz[2] = dtiip * dtiip * (dpsi - temp1 + dphi);
			}
		    } else {
			temp1 = z__[iip1] / dtiip;
			temp1 *= temp1;
			temp2 = (d__[iip1] - d__[iim1]) * (d__[iim1] + d__[
				iip1]) * temp1;
			c__ = temp - dtiim * (dpsi + dphi) - temp2;
			if (dphi < temp1) {
			    zz[0] = dtiim * dtiim * dpsi;
			} else {
			    zz[0] = dtiim * dtiim * (dpsi + (dphi - temp1));
			}
			zz[2] = z__[iip1] * z__[iip1];
		    }
		}
		dd[0] = dtiim;
		dd[1] = delta[ii] * work[ii];
		dd[2] = dtiip;
		dlaed6_(&niter, &orgati, &c__, dd, zz, &w, &eta, info);
		if (*info != 0) {
		    //
		    //                If INFO is not 0, i.e., DLAED6 failed, switch
		    //                back to two pole interpolation
		    //
		    swtch3 = FALSE_;
		    *info = 0;
		    dtipsq = work[ip1] * delta[ip1];
		    dtisq = work[*i__] * delta[*i__];
		    if (! swtch) {
			if (orgati) {
			    // Computing 2nd power
			    d__1 = z__[*i__] / dtisq;
			    c__ = w - dtipsq * dw + delsq * (d__1 * d__1);
			} else {
			    // Computing 2nd power
			    d__1 = z__[ip1] / dtipsq;
			    c__ = w - dtisq * dw - delsq * (d__1 * d__1);
			}
		    } else {
			temp = z__[ii] / (work[ii] * delta[ii]);
			if (orgati) {
			    dpsi += temp * temp;
			} else {
			    dphi += temp * temp;
			}
			c__ = w - dtisq * dpsi - dtipsq * dphi;
		    }
		    a = (dtipsq + dtisq) * w - dtipsq * dtisq * dw;
		    b = dtipsq * dtisq * w;
		    if (c__ == 0.) {
			if (a == 0.) {
			    if (! swtch) {
				if (orgati) {
				    a = z__[*i__] * z__[*i__] + dtipsq *
					    dtipsq * (dpsi + dphi);
				} else {
				    a = z__[ip1] * z__[ip1] + dtisq * dtisq *
					    (dpsi + dphi);
				}
			    } else {
				a = dtisq * dtisq * dpsi + dtipsq * dtipsq *
					dphi;
			    }
			}
			eta = b / a;
		    } else if (a <= 0.) {
			eta = (a - sqrt((d__1 = a * a - b * 4. * c__, abs(
				d__1)))) / (c__ * 2.);
		    } else {
			eta = b * 2. / (a + sqrt((d__1 = a * a - b * 4. * c__,
				 abs(d__1))));
		    }
		}
	    }
	    //
	    //          Note, eta should be positive if w is negative, and
	    //          eta should be negative otherwise. However,
	    //          if for some reason caused by roundoff, eta*w > 0,
	    //          we simply use one Newton step instead. This way
	    //          will guarantee eta*w < 0.
	    //
	    if (w * eta >= 0.) {
		eta = -w / dw;
	    }
	    eta /= *sigma + sqrt(*sigma * *sigma + eta);
	    temp = tau + eta;
	    if (temp > sgub || temp < sglb) {
		if (w < 0.) {
		    eta = (sgub - tau) / 2.;
		} else {
		    eta = (sglb - tau) / 2.;
		}
		if (geomavg) {
		    if (w < 0.) {
			if (tau > 0.) {
			    eta = sqrt(sgub * tau) - tau;
			}
		    } else {
			if (sglb > 0.) {
			    eta = sqrt(sglb * tau) - tau;
			}
		    }
		}
	    }
	    prew = w;
	    tau += eta;
	    *sigma += eta;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		work[j] += eta;
		delta[j] -= eta;
// L200:
	    }
	    //
	    //          Evaluate PSI and the derivative DPSI
	    //
	    dpsi = 0.;
	    psi = 0.;
	    erretm = 0.;
	    i__1 = iim1;
	    for (j = 1; j <= i__1; ++j) {
		temp = z__[j] / (work[j] * delta[j]);
		psi += z__[j] * temp;
		dpsi += temp * temp;
		erretm += psi;
// L210:
	    }
	    erretm = abs(erretm);
	    //
	    //          Evaluate PHI and the derivative DPHI
	    //
	    dphi = 0.;
	    phi = 0.;
	    i__1 = iip1;
	    for (j = *n; j >= i__1; --j) {
		temp = z__[j] / (work[j] * delta[j]);
		phi += z__[j] * temp;
		dphi += temp * temp;
		erretm += phi;
// L220:
	    }
	    tau2 = work[ii] * delta[ii];
	    temp = z__[ii] / tau2;
	    dw = dpsi + dphi + temp * temp;
	    temp = z__[ii] * temp;
	    w = rhoinv + phi + psi + temp;
	    erretm = (phi - psi) * 8. + erretm + rhoinv * 2. + abs(temp) * 3.;
	    //   $             + ABS( TAU2 )*DW
	    //
	    if (w * prew > 0. && abs(w) > abs(prew) / 10.) {
		swtch = ! swtch;
	    }
// L230:
	}
	//
	//       Return with INFO = 1, NITER = MAXIT and not converged
	//
	*info = 1;
    }
L240:
    return 0;
    //
    //    End of DLASD4
    //
} // dlasd4_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD5 computes the square root of the i-th eigenvalue of a positive symmetric rank-one modification of a 2-by-2 diagonal matrix. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD5 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd5.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd5.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd5.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD5( I, D, Z, DELTA, RHO, DSIGMA, WORK )
//
//      .. Scalar Arguments ..
//      INTEGER            I
//      DOUBLE PRECISION   DSIGMA, RHO
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( 2 ), DELTA( 2 ), WORK( 2 ), Z( 2 )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> This subroutine computes the square root of the I-th eigenvalue
//> of a positive symmetric rank-one modification of a 2-by-2 diagonal
//> matrix
//>
//>            diag( D ) * diag( D ) +  RHO * Z * transpose(Z) .
//>
//> The diagonal entries in the array D are assumed to satisfy
//>
//>            0 <= D(i) < D(j)  for  i < j .
//>
//> We also assume RHO > 0 and that the Euclidean norm of the vector
//> Z is one.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I
//> \verbatim
//>          I is INTEGER
//>         The index of the eigenvalue to be computed.  I = 1 or I = 2.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension ( 2 )
//>         The original eigenvalues.  We assume 0 <= D(1) < D(2).
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 2 )
//>         The components of the updating vector.
//> \endverbatim
//>
//> \param[out] DELTA
//> \verbatim
//>          DELTA is DOUBLE PRECISION array, dimension ( 2 )
//>         Contains (D(j) - sigma_I) in its  j-th component.
//>         The vector DELTA contains the information necessary
//>         to construct the eigenvectors.
//> \endverbatim
//>
//> \param[in] RHO
//> \verbatim
//>          RHO is DOUBLE PRECISION
//>         The scalar in the symmetric updating formula.
//> \endverbatim
//>
//> \param[out] DSIGMA
//> \verbatim
//>          DSIGMA is DOUBLE PRECISION
//>         The computed sigma_I, the I-th updated eigenvalue.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension ( 2 )
//>         WORK contains (D(j) + sigma_I) in its  j-th component.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ren-Cang Li, Computer Science Division, University of California
//>     at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd5_(int *i__, double *d__, double *z__, double *
	delta, double *rho, double *dsigma, double *work)
{
    // System generated locals
    double d__1;

    // Local variables
    double b, c__, w, del, tau, delsq;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --work;
    --delta;
    --z__;
    --d__;

    // Function Body
    del = d__[2] - d__[1];
    delsq = del * (d__[2] + d__[1]);
    if (*i__ == 1) {
	w = *rho * 4. * (z__[2] * z__[2] / (d__[1] + d__[2] * 3.) - z__[1] *
		z__[1] / (d__[1] * 3. + d__[2])) / del + 1.;
	if (w > 0.) {
	    b = delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[1] * z__[1] * delsq;
	    //
	    //          B > ZERO, always
	    //
	    //          The following TAU is DSIGMA * DSIGMA - D( 1 ) * D( 1 )
	    //
	    tau = c__ * 2. / (b + sqrt((d__1 = b * b - c__ * 4., abs(d__1))));
	    //
	    //          The following TAU is DSIGMA - D( 1 )
	    //
	    tau /= d__[1] + sqrt(d__[1] * d__[1] + tau);
	    *dsigma = d__[1] + tau;
	    delta[1] = -tau;
	    delta[2] = del - tau;
	    work[1] = d__[1] * 2. + tau;
	    work[2] = d__[1] + tau + d__[2];
	    //          DELTA( 1 ) = -Z( 1 ) / TAU
	    //          DELTA( 2 ) = Z( 2 ) / ( DEL-TAU )
	} else {
	    b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[2] * z__[2] * delsq;
	    //
	    //          The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 )
	    //
	    if (b > 0.) {
		tau = c__ * -2. / (b + sqrt(b * b + c__ * 4.));
	    } else {
		tau = (b - sqrt(b * b + c__ * 4.)) / 2.;
	    }
	    //
	    //          The following TAU is DSIGMA - D( 2 )
	    //
	    tau /= d__[2] + sqrt((d__1 = d__[2] * d__[2] + tau, abs(d__1)));
	    *dsigma = d__[2] + tau;
	    delta[1] = -(del + tau);
	    delta[2] = -tau;
	    work[1] = d__[1] + tau + d__[2];
	    work[2] = d__[2] * 2. + tau;
	    //          DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
	    //          DELTA( 2 ) = -Z( 2 ) / TAU
	}
	//       TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
	//       DELTA( 1 ) = DELTA( 1 ) / TEMP
	//       DELTA( 2 ) = DELTA( 2 ) / TEMP
    } else {
	//
	//       Now I=2
	//
	b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	c__ = *rho * z__[2] * z__[2] * delsq;
	//
	//       The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 )
	//
	if (b > 0.) {
	    tau = (b + sqrt(b * b + c__ * 4.)) / 2.;
	} else {
	    tau = c__ * 2. / (-b + sqrt(b * b + c__ * 4.));
	}
	//
	//       The following TAU is DSIGMA - D( 2 )
	//
	tau /= d__[2] + sqrt(d__[2] * d__[2] + tau);
	*dsigma = d__[2] + tau;
	delta[1] = -(del + tau);
	delta[2] = -tau;
	work[1] = d__[1] + tau + d__[2];
	work[2] = d__[2] * 2. + tau;
	//       DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU )
	//       DELTA( 2 ) = -Z( 2 ) / TAU
	//       TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) )
	//       DELTA( 1 ) = DELTA( 1 ) / TEMP
	//       DELTA( 2 ) = DELTA( 2 ) / TEMP
    }
    return 0;
    //
    //    End of DLASD5
    //
} // dlasd5_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD6 computes the SVD of an updated upper bidiagonal matrix obtained by merging two smaller ones by appending a row. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD6 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd6.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd6.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd6.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD6( ICOMPQ, NL, NR, SQRE, D, VF, VL, ALPHA, BETA,
//                         IDXQ, PERM, GIVPTR, GIVCOL, LDGCOL, GIVNUM,
//                         LDGNUM, POLES, DIFL, DIFR, Z, K, C, S, WORK,
//                         IWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            GIVPTR, ICOMPQ, INFO, K, LDGCOL, LDGNUM, NL,
//     $                   NR, SQRE
//      DOUBLE PRECISION   ALPHA, BETA, C, S
//      ..
//      .. Array Arguments ..
//      INTEGER            GIVCOL( LDGCOL, * ), IDXQ( * ), IWORK( * ),
//     $                   PERM( * )
//      DOUBLE PRECISION   D( * ), DIFL( * ), DIFR( * ),
//     $                   GIVNUM( LDGNUM, * ), POLES( LDGNUM, * ),
//     $                   VF( * ), VL( * ), WORK( * ), Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASD6 computes the SVD of an updated upper bidiagonal matrix B
//> obtained by merging two smaller ones by appending a row. This
//> routine is used only for the problem which requires all singular
//> values and optionally singular vector matrices in factored form.
//> B is an N-by-M matrix with N = NL + NR + 1 and M = N + SQRE.
//> A related subroutine, DLASD1, handles the case in which all singular
//> values and singular vectors of the bidiagonal matrix are desired.
//>
//> DLASD6 computes the SVD as follows:
//>
//>               ( D1(in)    0    0       0 )
//>   B = U(in) * (   Z1**T   a   Z2**T    b ) * VT(in)
//>               (   0       0   D2(in)   0 )
//>
//>     = U(out) * ( D(out) 0) * VT(out)
//>
//> where Z**T = (Z1**T a Z2**T b) = u**T VT**T, and u is a vector of dimension M
//> with ALPHA and BETA in the NL+1 and NL+2 th entries and zeros
//> elsewhere; and the entry b is empty if SQRE = 0.
//>
//> The singular values of B can be computed using D1, D2, the first
//> components of all the right singular vectors of the lower block, and
//> the last components of all the right singular vectors of the upper
//> block. These components are stored and updated in VF and VL,
//> respectively, in DLASD6. Hence U and VT are not explicitly
//> referenced.
//>
//> The singular values are stored in D. The algorithm consists of two
//> stages:
//>
//>       The first stage consists of deflating the size of the problem
//>       when there are multiple singular values or if there is a zero
//>       in the Z vector. For each such occurrence the dimension of the
//>       secular equation problem is reduced by one. This stage is
//>       performed by the routine DLASD7.
//>
//>       The second stage consists of calculating the updated
//>       singular values. This is done by finding the roots of the
//>       secular equation via the routine DLASD4 (as called by DLASD8).
//>       This routine also updates VF and VL and computes the distances
//>       between the updated singular values and the old singular
//>       values.
//>
//> DLASD6 is called from DLASDA.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ICOMPQ
//> \verbatim
//>          ICOMPQ is INTEGER
//>         Specifies whether singular vectors are to be computed in
//>         factored form:
//>         = 0: Compute singular values only.
//>         = 1: Compute singular vectors in factored form as well.
//> \endverbatim
//>
//> \param[in] NL
//> \verbatim
//>          NL is INTEGER
//>         The row dimension of the upper block.  NL >= 1.
//> \endverbatim
//>
//> \param[in] NR
//> \verbatim
//>          NR is INTEGER
//>         The row dimension of the lower block.  NR >= 1.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>         = 0: the lower block is an NR-by-NR square matrix.
//>         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
//>
//>         The bidiagonal matrix has row dimension N = NL + NR + 1,
//>         and column dimension M = N + SQRE.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension ( NL+NR+1 ).
//>         On entry D(1:NL,1:NL) contains the singular values of the
//>         upper block, and D(NL+2:N) contains the singular values
//>         of the lower block. On exit D(1:N) contains the singular
//>         values of the modified matrix.
//> \endverbatim
//>
//> \param[in,out] VF
//> \verbatim
//>          VF is DOUBLE PRECISION array, dimension ( M )
//>         On entry, VF(1:NL+1) contains the first components of all
//>         right singular vectors of the upper block; and VF(NL+2:M)
//>         contains the first components of all right singular vectors
//>         of the lower block. On exit, VF contains the first components
//>         of all right singular vectors of the bidiagonal matrix.
//> \endverbatim
//>
//> \param[in,out] VL
//> \verbatim
//>          VL is DOUBLE PRECISION array, dimension ( M )
//>         On entry, VL(1:NL+1) contains the  last components of all
//>         right singular vectors of the upper block; and VL(NL+2:M)
//>         contains the last components of all right singular vectors of
//>         the lower block. On exit, VL contains the last components of
//>         all right singular vectors of the bidiagonal matrix.
//> \endverbatim
//>
//> \param[in,out] ALPHA
//> \verbatim
//>          ALPHA is DOUBLE PRECISION
//>         Contains the diagonal element associated with the added row.
//> \endverbatim
//>
//> \param[in,out] BETA
//> \verbatim
//>          BETA is DOUBLE PRECISION
//>         Contains the off-diagonal element associated with the added
//>         row.
//> \endverbatim
//>
//> \param[in,out] IDXQ
//> \verbatim
//>          IDXQ is INTEGER array, dimension ( N )
//>         This contains the permutation which will reintegrate the
//>         subproblem just solved back into sorted order, i.e.
//>         D( IDXQ( I = 1, N ) ) will be in ascending order.
//> \endverbatim
//>
//> \param[out] PERM
//> \verbatim
//>          PERM is INTEGER array, dimension ( N )
//>         The permutations (from deflation and sorting) to be applied
//>         to each block. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[out] GIVPTR
//> \verbatim
//>          GIVPTR is INTEGER
//>         The number of Givens rotations which took place in this
//>         subproblem. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[out] GIVCOL
//> \verbatim
//>          GIVCOL is INTEGER array, dimension ( LDGCOL, 2 )
//>         Each pair of numbers indicates a pair of columns to take place
//>         in a Givens rotation. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[in] LDGCOL
//> \verbatim
//>          LDGCOL is INTEGER
//>         leading dimension of GIVCOL, must be at least N.
//> \endverbatim
//>
//> \param[out] GIVNUM
//> \verbatim
//>          GIVNUM is DOUBLE PRECISION array, dimension ( LDGNUM, 2 )
//>         Each number indicates the C or S value to be used in the
//>         corresponding Givens rotation. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[in] LDGNUM
//> \verbatim
//>          LDGNUM is INTEGER
//>         The leading dimension of GIVNUM and POLES, must be at least N.
//> \endverbatim
//>
//> \param[out] POLES
//> \verbatim
//>          POLES is DOUBLE PRECISION array, dimension ( LDGNUM, 2 )
//>         On exit, POLES(1,*) is an array containing the new singular
//>         values obtained from solving the secular equation, and
//>         POLES(2,*) is an array containing the poles in the secular
//>         equation. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[out] DIFL
//> \verbatim
//>          DIFL is DOUBLE PRECISION array, dimension ( N )
//>         On exit, DIFL(I) is the distance between I-th updated
//>         (undeflated) singular value and the I-th (undeflated) old
//>         singular value.
//> \endverbatim
//>
//> \param[out] DIFR
//> \verbatim
//>          DIFR is DOUBLE PRECISION array,
//>                   dimension ( LDDIFR, 2 ) if ICOMPQ = 1 and
//>                   dimension ( K ) if ICOMPQ = 0.
//>          On exit, DIFR(I,1) = D(I) - DSIGMA(I+1), DIFR(K,1) is not
//>          defined and will not be referenced.
//>
//>          If ICOMPQ = 1, DIFR(1:K,2) is an array containing the
//>          normalizing factors for the right singular vector matrix.
//>
//>         See DLASD8 for details on DIFL and DIFR.
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( M )
//>         The first elements of this array contain the components
//>         of the deflation-adjusted updating row vector.
//> \endverbatim
//>
//> \param[out] K
//> \verbatim
//>          K is INTEGER
//>         Contains the dimension of the non-deflated matrix,
//>         This is the order of the related secular equation. 1 <= K <=N.
//> \endverbatim
//>
//> \param[out] C
//> \verbatim
//>          C is DOUBLE PRECISION
//>         C contains garbage if SQRE =0 and the C-value of a Givens
//>         rotation related to the right null space if SQRE = 1.
//> \endverbatim
//>
//> \param[out] S
//> \verbatim
//>          S is DOUBLE PRECISION
//>         S contains garbage if SQRE =0 and the S-value of a Givens
//>         rotation related to the right null space if SQRE = 1.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension ( 4 * M )
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension ( 3 * N )
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  if INFO = 1, a singular value did not converge
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd6_(int *icompq, int *nl, int *nr, int *sqre, double
	*d__, double *vf, double *vl, double *alpha, double *beta, int *idxq,
	int *perm, int *givptr, int *givcol, int *ldgcol, double *givnum, int
	*ldgnum, double *poles, double *difl, double *difr, double *z__, int *
	k, double *c__, double *s, double *work, int *iwork, int *info)
{
    // Table of constant values
    int c__0 = 0;
    double c_b7 = 1.;
    int c__1 = 1;
    int c_n1 = -1;

    // System generated locals
    int givcol_dim1, givcol_offset, givnum_dim1, givnum_offset, poles_dim1,
	    poles_offset, i__1;
    double d__1, d__2;

    // Local variables
    int i__, m, n, n1, n2, iw, idx, idxc, idxp, ivfw, ivlw;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    ), dlasd7_(int *, int *, int *, int *, int *, double *, double *,
	    double *, double *, double *, double *, double *, double *,
	    double *, double *, int *, int *, int *, int *, int *, int *, int
	    *, double *, int *, double *, double *, int *), dlasd8_(int *,
	    int *, double *, double *, double *, double *, double *, double *,
	     int *, double *, double *, int *), dlascl_(char *, int *, int *,
	    double *, double *, int *, int *, double *, int *, int *),
	    dlamrg_(int *, int *, double *, int *, int *, int *);
    int isigma;
    extern /* Subroutine */ int xerbla_(char *, int *);
    double orgnrm;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;
    --vf;
    --vl;
    --idxq;
    --perm;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    poles_dim1 = *ldgnum;
    poles_offset = 1 + poles_dim1;
    poles -= poles_offset;
    givnum_dim1 = *ldgnum;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;
    --difl;
    --difr;
    --z__;
    --work;
    --iwork;

    // Function Body
    *info = 0;
    n = *nl + *nr + 1;
    m = n + *sqre;
    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*nl < 1) {
	*info = -2;
    } else if (*nr < 1) {
	*info = -3;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -4;
    } else if (*ldgcol < n) {
	*info = -14;
    } else if (*ldgnum < n) {
	*info = -16;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASD6", &i__1);
	return 0;
    }
    //
    //    The following values are for bookkeeping purposes only.  They are
    //    integer pointers which indicate the portion of the workspace
    //    used by a particular array in DLASD7 and DLASD8.
    //
    isigma = 1;
    iw = isigma + n;
    ivfw = iw + m;
    ivlw = ivfw + m;
    idx = 1;
    idxc = idx + n;
    idxp = idxc + n;
    //
    //    Scale.
    //
    // Computing MAX
    d__1 = abs(*alpha), d__2 = abs(*beta);
    orgnrm = max(d__1,d__2);
    d__[*nl + 1] = 0.;
    i__1 = n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if ((d__1 = d__[i__], abs(d__1)) > orgnrm) {
	    orgnrm = (d__1 = d__[i__], abs(d__1));
	}
// L10:
    }
    dlascl_("G", &c__0, &c__0, &orgnrm, &c_b7, &n, &c__1, &d__[1], &n, info);
    *alpha /= orgnrm;
    *beta /= orgnrm;
    //
    //    Sort and Deflate singular values.
    //
    dlasd7_(icompq, nl, nr, sqre, k, &d__[1], &z__[1], &work[iw], &vf[1], &
	    work[ivfw], &vl[1], &work[ivlw], alpha, beta, &work[isigma], &
	    iwork[idx], &iwork[idxp], &idxq[1], &perm[1], givptr, &givcol[
	    givcol_offset], ldgcol, &givnum[givnum_offset], ldgnum, c__, s,
	    info);
    //
    //    Solve Secular Equation, compute DIFL, DIFR, and update VF, VL.
    //
    dlasd8_(icompq, k, &d__[1], &z__[1], &vf[1], &vl[1], &difl[1], &difr[1],
	    ldgnum, &work[isigma], &work[iw], info);
    //
    //    Report the possible convergence failure.
    //
    if (*info != 0) {
	return 0;
    }
    //
    //    Save the poles if ICOMPQ = 1.
    //
    if (*icompq == 1) {
	dcopy_(k, &d__[1], &c__1, &poles[poles_dim1 + 1], &c__1);
	dcopy_(k, &work[isigma], &c__1, &poles[(poles_dim1 << 1) + 1], &c__1);
    }
    //
    //    Unscale.
    //
    dlascl_("G", &c__0, &c__0, &c_b7, &orgnrm, &n, &c__1, &d__[1], &n, info);
    //
    //    Prepare the IDXQ sorting permutation.
    //
    n1 = *k;
    n2 = n - *k;
    dlamrg_(&n1, &n2, &d__[1], &c__1, &c_n1, &idxq[1]);
    return 0;
    //
    //    End of DLASD6
    //
} // dlasd6_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD7 merges the two sets of singular values together into a single sorted set. Then it tries to deflate the size of the problem. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD7 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd7.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd7.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd7.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD7( ICOMPQ, NL, NR, SQRE, K, D, Z, ZW, VF, VFW, VL,
//                         VLW, ALPHA, BETA, DSIGMA, IDX, IDXP, IDXQ,
//                         PERM, GIVPTR, GIVCOL, LDGCOL, GIVNUM, LDGNUM,
//                         C, S, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            GIVPTR, ICOMPQ, INFO, K, LDGCOL, LDGNUM, NL,
//     $                   NR, SQRE
//      DOUBLE PRECISION   ALPHA, BETA, C, S
//      ..
//      .. Array Arguments ..
//      INTEGER            GIVCOL( LDGCOL, * ), IDX( * ), IDXP( * ),
//     $                   IDXQ( * ), PERM( * )
//      DOUBLE PRECISION   D( * ), DSIGMA( * ), GIVNUM( LDGNUM, * ),
//     $                   VF( * ), VFW( * ), VL( * ), VLW( * ), Z( * ),
//     $                   ZW( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASD7 merges the two sets of singular values together into a single
//> sorted set. Then it tries to deflate the size of the problem. There
//> are two ways in which deflation can occur:  when two or more singular
//> values are close together or if there is a tiny entry in the Z
//> vector. For each such occurrence the order of the related
//> secular equation problem is reduced by one.
//>
//> DLASD7 is called from DLASD6.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ICOMPQ
//> \verbatim
//>          ICOMPQ is INTEGER
//>          Specifies whether singular vectors are to be computed
//>          in compact form, as follows:
//>          = 0: Compute singular values only.
//>          = 1: Compute singular vectors of upper
//>               bidiagonal matrix in compact form.
//> \endverbatim
//>
//> \param[in] NL
//> \verbatim
//>          NL is INTEGER
//>         The row dimension of the upper block. NL >= 1.
//> \endverbatim
//>
//> \param[in] NR
//> \verbatim
//>          NR is INTEGER
//>         The row dimension of the lower block. NR >= 1.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>         = 0: the lower block is an NR-by-NR square matrix.
//>         = 1: the lower block is an NR-by-(NR+1) rectangular matrix.
//>
//>         The bidiagonal matrix has
//>         N = NL + NR + 1 rows and
//>         M = N + SQRE >= N columns.
//> \endverbatim
//>
//> \param[out] K
//> \verbatim
//>          K is INTEGER
//>         Contains the dimension of the non-deflated matrix, this is
//>         the order of the related secular equation. 1 <= K <=N.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension ( N )
//>         On entry D contains the singular values of the two submatrices
//>         to be combined. On exit D contains the trailing (N-K) updated
//>         singular values (those which were deflated) sorted into
//>         increasing order.
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( M )
//>         On exit Z contains the updating row vector in the secular
//>         equation.
//> \endverbatim
//>
//> \param[out] ZW
//> \verbatim
//>          ZW is DOUBLE PRECISION array, dimension ( M )
//>         Workspace for Z.
//> \endverbatim
//>
//> \param[in,out] VF
//> \verbatim
//>          VF is DOUBLE PRECISION array, dimension ( M )
//>         On entry, VF(1:NL+1) contains the first components of all
//>         right singular vectors of the upper block; and VF(NL+2:M)
//>         contains the first components of all right singular vectors
//>         of the lower block. On exit, VF contains the first components
//>         of all right singular vectors of the bidiagonal matrix.
//> \endverbatim
//>
//> \param[out] VFW
//> \verbatim
//>          VFW is DOUBLE PRECISION array, dimension ( M )
//>         Workspace for VF.
//> \endverbatim
//>
//> \param[in,out] VL
//> \verbatim
//>          VL is DOUBLE PRECISION array, dimension ( M )
//>         On entry, VL(1:NL+1) contains the  last components of all
//>         right singular vectors of the upper block; and VL(NL+2:M)
//>         contains the last components of all right singular vectors
//>         of the lower block. On exit, VL contains the last components
//>         of all right singular vectors of the bidiagonal matrix.
//> \endverbatim
//>
//> \param[out] VLW
//> \verbatim
//>          VLW is DOUBLE PRECISION array, dimension ( M )
//>         Workspace for VL.
//> \endverbatim
//>
//> \param[in] ALPHA
//> \verbatim
//>          ALPHA is DOUBLE PRECISION
//>         Contains the diagonal element associated with the added row.
//> \endverbatim
//>
//> \param[in] BETA
//> \verbatim
//>          BETA is DOUBLE PRECISION
//>         Contains the off-diagonal element associated with the added
//>         row.
//> \endverbatim
//>
//> \param[out] DSIGMA
//> \verbatim
//>          DSIGMA is DOUBLE PRECISION array, dimension ( N )
//>         Contains a copy of the diagonal elements (K-1 singular values
//>         and one zero) in the secular equation.
//> \endverbatim
//>
//> \param[out] IDX
//> \verbatim
//>          IDX is INTEGER array, dimension ( N )
//>         This will contain the permutation used to sort the contents of
//>         D into ascending order.
//> \endverbatim
//>
//> \param[out] IDXP
//> \verbatim
//>          IDXP is INTEGER array, dimension ( N )
//>         This will contain the permutation used to place deflated
//>         values of D at the end of the array. On output IDXP(2:K)
//>         points to the nondeflated D-values and IDXP(K+1:N)
//>         points to the deflated singular values.
//> \endverbatim
//>
//> \param[in] IDXQ
//> \verbatim
//>          IDXQ is INTEGER array, dimension ( N )
//>         This contains the permutation which separately sorts the two
//>         sub-problems in D into ascending order.  Note that entries in
//>         the first half of this permutation must first be moved one
//>         position backward; and entries in the second half
//>         must first have NL+1 added to their values.
//> \endverbatim
//>
//> \param[out] PERM
//> \verbatim
//>          PERM is INTEGER array, dimension ( N )
//>         The permutations (from deflation and sorting) to be applied
//>         to each singular block. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[out] GIVPTR
//> \verbatim
//>          GIVPTR is INTEGER
//>         The number of Givens rotations which took place in this
//>         subproblem. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[out] GIVCOL
//> \verbatim
//>          GIVCOL is INTEGER array, dimension ( LDGCOL, 2 )
//>         Each pair of numbers indicates a pair of columns to take place
//>         in a Givens rotation. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[in] LDGCOL
//> \verbatim
//>          LDGCOL is INTEGER
//>         The leading dimension of GIVCOL, must be at least N.
//> \endverbatim
//>
//> \param[out] GIVNUM
//> \verbatim
//>          GIVNUM is DOUBLE PRECISION array, dimension ( LDGNUM, 2 )
//>         Each number indicates the C or S value to be used in the
//>         corresponding Givens rotation. Not referenced if ICOMPQ = 0.
//> \endverbatim
//>
//> \param[in] LDGNUM
//> \verbatim
//>          LDGNUM is INTEGER
//>         The leading dimension of GIVNUM, must be at least N.
//> \endverbatim
//>
//> \param[out] C
//> \verbatim
//>          C is DOUBLE PRECISION
//>         C contains garbage if SQRE =0 and the C-value of a Givens
//>         rotation related to the right null space if SQRE = 1.
//> \endverbatim
//>
//> \param[out] S
//> \verbatim
//>          S is DOUBLE PRECISION
//>         S contains garbage if SQRE =0 and the S-value of a Givens
//>         rotation related to the right null space if SQRE = 1.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>         = 0:  successful exit.
//>         < 0:  if INFO = -i, the i-th argument had an illegal value.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd7_(int *icompq, int *nl, int *nr, int *sqre, int *k,
	 double *d__, double *z__, double *zw, double *vf, double *vfw,
	double *vl, double *vlw, double *alpha, double *beta, double *dsigma,
	int *idx, int *idxp, int *idxq, int *perm, int *givptr, int *givcol,
	int *ldgcol, double *givnum, int *ldgnum, double *c__, double *s, int
	*info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int givcol_dim1, givcol_offset, givnum_dim1, givnum_offset, i__1;
    double d__1, d__2;

    // Local variables
    int i__, j, m, n, k2;
    double z1;
    int jp;
    double eps, tau, tol;
    int nlp1, nlp2, idxi, idxj;
    extern /* Subroutine */ int drot_(int *, double *, int *, double *, int *,
	     double *, double *);
    int idxjp;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int jprev;
    extern double dlapy2_(double *, double *), dlamch_(char *);
    extern /* Subroutine */ int dlamrg_(int *, int *, double *, int *, int *,
	    int *), xerbla_(char *, int *);
    double hlftol;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;
    --z__;
    --zw;
    --vf;
    --vfw;
    --vl;
    --vlw;
    --dsigma;
    --idx;
    --idxp;
    --idxq;
    --perm;
    givcol_dim1 = *ldgcol;
    givcol_offset = 1 + givcol_dim1;
    givcol -= givcol_offset;
    givnum_dim1 = *ldgnum;
    givnum_offset = 1 + givnum_dim1;
    givnum -= givnum_offset;

    // Function Body
    *info = 0;
    n = *nl + *nr + 1;
    m = n + *sqre;
    if (*icompq < 0 || *icompq > 1) {
	*info = -1;
    } else if (*nl < 1) {
	*info = -2;
    } else if (*nr < 1) {
	*info = -3;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -4;
    } else if (*ldgcol < n) {
	*info = -22;
    } else if (*ldgnum < n) {
	*info = -24;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASD7", &i__1);
	return 0;
    }
    nlp1 = *nl + 1;
    nlp2 = *nl + 2;
    if (*icompq == 1) {
	*givptr = 0;
    }
    //
    //    Generate the first part of the vector Z and move the singular
    //    values in the first part of D one position backward.
    //
    z1 = *alpha * vl[nlp1];
    vl[nlp1] = 0.;
    tau = vf[nlp1];
    for (i__ = *nl; i__ >= 1; --i__) {
	z__[i__ + 1] = *alpha * vl[i__];
	vl[i__] = 0.;
	vf[i__ + 1] = vf[i__];
	d__[i__ + 1] = d__[i__];
	idxq[i__ + 1] = idxq[i__] + 1;
// L10:
    }
    vf[1] = tau;
    //
    //    Generate the second part of the vector Z.
    //
    i__1 = m;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	z__[i__] = *beta * vf[i__];
	vf[i__] = 0.;
// L20:
    }
    //
    //    Sort the singular values into increasing order
    //
    i__1 = n;
    for (i__ = nlp2; i__ <= i__1; ++i__) {
	idxq[i__] += nlp1;
// L30:
    }
    //
    //    DSIGMA, IDXC, IDXC, and ZW are used as storage space.
    //
    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	dsigma[i__] = d__[idxq[i__]];
	zw[i__] = z__[idxq[i__]];
	vfw[i__] = vf[idxq[i__]];
	vlw[i__] = vl[idxq[i__]];
// L40:
    }
    dlamrg_(nl, nr, &dsigma[2], &c__1, &c__1, &idx[2]);
    i__1 = n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	idxi = idx[i__] + 1;
	d__[i__] = dsigma[idxi];
	z__[i__] = zw[idxi];
	vf[i__] = vfw[idxi];
	vl[i__] = vlw[idxi];
// L50:
    }
    //
    //    Calculate the allowable deflation tolerance
    //
    eps = dlamch_("Epsilon");
    // Computing MAX
    d__1 = abs(*alpha), d__2 = abs(*beta);
    tol = max(d__1,d__2);
    // Computing MAX
    d__2 = (d__1 = d__[n], abs(d__1));
    tol = eps * 64. * max(d__2,tol);
    //
    //    There are 2 kinds of deflation -- first a value in the z-vector
    //    is small, second two (or more) singular values are very close
    //    together (their difference is small).
    //
    //    If the value in the z-vector is small, we simply permute the
    //    array so that the corresponding singular value is moved to the
    //    end.
    //
    //    If two values in the D-vector are close, we perform a two-sided
    //    rotation designed to make one of the corresponding z-vector
    //    entries zero, and then permute the array so that the deflated
    //    singular value is moved to the end.
    //
    //    If there are multiple singular values then the problem deflates.
    //    Here the number of equal singular values are found.  As each equal
    //    singular value is found, an elementary reflector is computed to
    //    rotate the corresponding singular subspace so that the
    //    corresponding components of Z are zero in this new basis.
    //
    *k = 1;
    k2 = n + 1;
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	if ((d__1 = z__[j], abs(d__1)) <= tol) {
	    //
	    //          Deflate due to small z component.
	    //
	    --k2;
	    idxp[k2] = j;
	    if (j == n) {
		goto L100;
	    }
	} else {
	    jprev = j;
	    goto L70;
	}
// L60:
    }
L70:
    j = jprev;
L80:
    ++j;
    if (j > n) {
	goto L90;
    }
    if ((d__1 = z__[j], abs(d__1)) <= tol) {
	//
	//       Deflate due to small z component.
	//
	--k2;
	idxp[k2] = j;
    } else {
	//
	//       Check if singular values are close enough to allow deflation.
	//
	if ((d__1 = d__[j] - d__[jprev], abs(d__1)) <= tol) {
	    //
	    //          Deflation is possible.
	    //
	    *s = z__[jprev];
	    *c__ = z__[j];
	    //
	    //          Find sqrt(a**2+b**2) without overflow or
	    //          destructive underflow.
	    //
	    tau = dlapy2_(c__, s);
	    z__[j] = tau;
	    z__[jprev] = 0.;
	    *c__ /= tau;
	    *s = -(*s) / tau;
	    //
	    //          Record the appropriate Givens rotation
	    //
	    if (*icompq == 1) {
		++(*givptr);
		idxjp = idxq[idx[jprev] + 1];
		idxj = idxq[idx[j] + 1];
		if (idxjp <= nlp1) {
		    --idxjp;
		}
		if (idxj <= nlp1) {
		    --idxj;
		}
		givcol[*givptr + (givcol_dim1 << 1)] = idxjp;
		givcol[*givptr + givcol_dim1] = idxj;
		givnum[*givptr + (givnum_dim1 << 1)] = *c__;
		givnum[*givptr + givnum_dim1] = *s;
	    }
	    drot_(&c__1, &vf[jprev], &c__1, &vf[j], &c__1, c__, s);
	    drot_(&c__1, &vl[jprev], &c__1, &vl[j], &c__1, c__, s);
	    --k2;
	    idxp[k2] = jprev;
	    jprev = j;
	} else {
	    ++(*k);
	    zw[*k] = z__[jprev];
	    dsigma[*k] = d__[jprev];
	    idxp[*k] = jprev;
	    jprev = j;
	}
    }
    goto L80;
L90:
    //
    //    Record the last singular value.
    //
    ++(*k);
    zw[*k] = z__[jprev];
    dsigma[*k] = d__[jprev];
    idxp[*k] = jprev;
L100:
    //
    //    Sort the singular values into DSIGMA. The singular values which
    //    were not deflated go into the first K slots of DSIGMA, except
    //    that DSIGMA(1) is treated separately.
    //
    i__1 = n;
    for (j = 2; j <= i__1; ++j) {
	jp = idxp[j];
	dsigma[j] = d__[jp];
	vfw[j] = vf[jp];
	vlw[j] = vl[jp];
// L110:
    }
    if (*icompq == 1) {
	i__1 = n;
	for (j = 2; j <= i__1; ++j) {
	    jp = idxp[j];
	    perm[j] = idxq[idx[jp] + 1];
	    if (perm[j] <= nlp1) {
		--perm[j];
	    }
// L120:
	}
    }
    //
    //    The deflated singular values go back into the last N - K slots of
    //    D.
    //
    i__1 = n - *k;
    dcopy_(&i__1, &dsigma[*k + 1], &c__1, &d__[*k + 1], &c__1);
    //
    //    Determine DSIGMA(1), DSIGMA(2), Z(1), VF(1), VL(1), VF(M), and
    //    VL(M).
    //
    dsigma[1] = 0.;
    hlftol = tol / 2.;
    if (abs(dsigma[2]) <= hlftol) {
	dsigma[2] = hlftol;
    }
    if (m > n) {
	z__[1] = dlapy2_(&z1, &z__[m]);
	if (z__[1] <= tol) {
	    *c__ = 1.;
	    *s = 0.;
	    z__[1] = tol;
	} else {
	    *c__ = z1 / z__[1];
	    *s = -z__[m] / z__[1];
	}
	drot_(&c__1, &vf[m], &c__1, &vf[1], &c__1, c__, s);
	drot_(&c__1, &vl[m], &c__1, &vl[1], &c__1, c__, s);
    } else {
	if (abs(z1) <= tol) {
	    z__[1] = tol;
	} else {
	    z__[1] = z1;
	}
    }
    //
    //    Restore Z, VF, and VL.
    //
    i__1 = *k - 1;
    dcopy_(&i__1, &zw[2], &c__1, &z__[2], &c__1);
    i__1 = n - 1;
    dcopy_(&i__1, &vfw[2], &c__1, &vf[2], &c__1);
    i__1 = n - 1;
    dcopy_(&i__1, &vlw[2], &c__1, &vl[2], &c__1);
    return 0;
    //
    //    End of DLASD7
    //
} // dlasd7_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASD8 finds the square roots of the roots of the secular equation, and stores, for each element in D, the distance to its two nearest poles. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASD8 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasd8.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasd8.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasd8.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASD8( ICOMPQ, K, D, Z, VF, VL, DIFL, DIFR, LDDIFR,
//                         DSIGMA, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            ICOMPQ, INFO, K, LDDIFR
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), DIFL( * ), DIFR( LDDIFR, * ),
//     $                   DSIGMA( * ), VF( * ), VL( * ), WORK( * ),
//     $                   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASD8 finds the square roots of the roots of the secular equation,
//> as defined by the values in DSIGMA and Z. It makes the appropriate
//> calls to DLASD4, and stores, for each  element in D, the distance
//> to its two nearest poles (elements in DSIGMA). It also updates
//> the arrays VF and VL, the first and last components of all the
//> right singular vectors of the original bidiagonal matrix.
//>
//> DLASD8 is called from DLASD6.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ICOMPQ
//> \verbatim
//>          ICOMPQ is INTEGER
//>          Specifies whether singular vectors are to be computed in
//>          factored form in the calling routine:
//>          = 0: Compute singular values only.
//>          = 1: Compute singular vectors in factored form as well.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of terms in the rational function to be solved
//>          by DLASD4.  K >= 1.
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension ( K )
//>          On output, D contains the updated singular values.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( K )
//>          On entry, the first K elements of this array contain the
//>          components of the deflation-adjusted updating row vector.
//>          On exit, Z is updated.
//> \endverbatim
//>
//> \param[in,out] VF
//> \verbatim
//>          VF is DOUBLE PRECISION array, dimension ( K )
//>          On entry, VF contains  information passed through DBEDE8.
//>          On exit, VF contains the first K components of the first
//>          components of all right singular vectors of the bidiagonal
//>          matrix.
//> \endverbatim
//>
//> \param[in,out] VL
//> \verbatim
//>          VL is DOUBLE PRECISION array, dimension ( K )
//>          On entry, VL contains  information passed through DBEDE8.
//>          On exit, VL contains the first K components of the last
//>          components of all right singular vectors of the bidiagonal
//>          matrix.
//> \endverbatim
//>
//> \param[out] DIFL
//> \verbatim
//>          DIFL is DOUBLE PRECISION array, dimension ( K )
//>          On exit, DIFL(I) = D(I) - DSIGMA(I).
//> \endverbatim
//>
//> \param[out] DIFR
//> \verbatim
//>          DIFR is DOUBLE PRECISION array,
//>                   dimension ( LDDIFR, 2 ) if ICOMPQ = 1 and
//>                   dimension ( K ) if ICOMPQ = 0.
//>          On exit, DIFR(I,1) = D(I) - DSIGMA(I+1), DIFR(K,1) is not
//>          defined and will not be referenced.
//>
//>          If ICOMPQ = 1, DIFR(1:K,2) is an array containing the
//>          normalizing factors for the right singular vector matrix.
//> \endverbatim
//>
//> \param[in] LDDIFR
//> \verbatim
//>          LDDIFR is INTEGER
//>          The leading dimension of DIFR, must be at least K.
//> \endverbatim
//>
//> \param[in,out] DSIGMA
//> \verbatim
//>          DSIGMA is DOUBLE PRECISION array, dimension ( K )
//>          On entry, the first K elements of this array contain the old
//>          roots of the deflated updating problem.  These are the poles
//>          of the secular equation.
//>          On exit, the elements of DSIGMA may be very slightly altered
//>          in value.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (3*K)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  if INFO = 1, a singular value did not converge
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasd8_(int *icompq, int *k, double *d__, double *z__,
	double *vf, double *vl, double *difl, double *difr, int *lddifr,
	double *dsigma, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__0 = 0;
    double c_b8 = 1.;

    // System generated locals
    int difr_dim1, difr_offset, i__1, i__2;
    double d__1, d__2;

    // Local variables
    int i__, j;
    double dj, rho;
    int iwk1, iwk2, iwk3;
    extern double ddot_(int *, double *, int *, double *, int *);
    double temp;
    extern double dnrm2_(int *, double *, int *);
    int iwk2i, iwk3i;
    double diflj, difrj, dsigj;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    extern double dlamc3_(double *, double *);
    extern /* Subroutine */ int dlasd4_(int *, int *, double *, double *,
	    double *, double *, double *, double *, int *), dlascl_(char *,
	    int *, int *, double *, double *, int *, int *, double *, int *,
	    int *), dlaset_(char *, int *, int *, double *, double *, double *
	    , int *), xerbla_(char *, int *);
    double dsigjp;

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
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

    // Function Body
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
	xerbla_("DLASD8", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*k == 1) {
	d__[1] = abs(z__[1]);
	difl[1] = d__[1];
	if (*icompq == 1) {
	    difl[2] = 1.;
	    difr[(difr_dim1 << 1) + 1] = 1.;
	}
	return 0;
    }
    //
    //    Modify values DSIGMA(i) to make sure all DSIGMA(i)-DSIGMA(j) can
    //    be computed with high relative accuracy (barring over/underflow).
    //    This is a problem on machines without a guard digit in
    //    add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
    //    The following code replaces DSIGMA(I) by 2*DSIGMA(I)-DSIGMA(I),
    //    which on any of these machines zeros out the bottommost
    //    bit of DSIGMA(I) if it is 1; this makes the subsequent
    //    subtractions DSIGMA(I)-DSIGMA(J) unproblematic when cancellation
    //    occurs. On binary machines with a guard digit (almost all
    //    machines) it does not change DSIGMA(I) at all. On hexadecimal
    //    and decimal machines with a guard digit, it slightly
    //    changes the bottommost bits of DSIGMA(I). It does not account
    //    for hexadecimal or decimal machines without guard digits
    //    (we know of none). We use a subroutine call to compute
    //    2*DLAMBDA(I) to prevent optimizing compilers from eliminating
    //    this code.
    //
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	dsigma[i__] = dlamc3_(&dsigma[i__], &dsigma[i__]) - dsigma[i__];
// L10:
    }
    //
    //    Book keeping.
    //
    iwk1 = 1;
    iwk2 = iwk1 + *k;
    iwk3 = iwk2 + *k;
    iwk2i = iwk2 - 1;
    iwk3i = iwk3 - 1;
    //
    //    Normalize Z.
    //
    rho = dnrm2_(k, &z__[1], &c__1);
    dlascl_("G", &c__0, &c__0, &rho, &c_b8, k, &c__1, &z__[1], k, info);
    rho *= rho;
    //
    //    Initialize WORK(IWK3).
    //
    dlaset_("A", k, &c__1, &c_b8, &c_b8, &work[iwk3], k);
    //
    //    Compute the updated singular values, the arrays DIFL, DIFR,
    //    and the updated Z.
    //
    i__1 = *k;
    for (j = 1; j <= i__1; ++j) {
	dlasd4_(k, &j, &dsigma[1], &z__[1], &work[iwk1], &rho, &d__[j], &work[
		iwk2], info);
	//
	//       If the root finder fails, report the convergence failure.
	//
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
// L20:
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[iwk3i + i__] = work[iwk3i + i__] * work[i__] * work[iwk2i +
		    i__] / (dsigma[i__] - dsigma[j]) / (dsigma[i__] + dsigma[
		    j]);
// L30:
	}
// L40:
    }
    //
    //    Compute updated Z.
    //
    i__1 = *k;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__2 = sqrt((d__1 = work[iwk3i + i__], abs(d__1)));
	z__[i__] = d_sign(&d__2, &z__[i__]);
// L50:
    }
    //
    //    Update VF and VL.
    //
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
	    work[i__] = z__[i__] / (dlamc3_(&dsigma[i__], &dsigj) - diflj) / (
		    dsigma[i__] + dj);
// L60:
	}
	i__2 = *k;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    work[i__] = z__[i__] / (dlamc3_(&dsigma[i__], &dsigjp) + difrj) /
		    (dsigma[i__] + dj);
// L70:
	}
	temp = dnrm2_(k, &work[1], &c__1);
	work[iwk2i + j] = ddot_(k, &work[1], &c__1, &vf[1], &c__1) / temp;
	work[iwk3i + j] = ddot_(k, &work[1], &c__1, &vl[1], &c__1) / temp;
	if (*icompq == 1) {
	    difr[j + (difr_dim1 << 1)] = temp;
	}
// L80:
    }
    dcopy_(k, &work[iwk2], &c__1, &vf[1], &c__1);
    dcopy_(k, &work[iwk3], &c__1, &vl[1], &c__1);
    return 0;
    //
    //    End of DLASD8
    //
} // dlasd8_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASDA computes the singular value decomposition (SVD) of a real upper bidiagonal matrix with diagonal d and off-diagonal e. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASDA + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasda.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasda.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasda.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASDA( ICOMPQ, SMLSIZ, N, SQRE, D, E, U, LDU, VT, K,
//                         DIFL, DIFR, Z, POLES, GIVPTR, GIVCOL, LDGCOL,
//                         PERM, GIVNUM, C, S, WORK, IWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            ICOMPQ, INFO, LDGCOL, LDU, N, SMLSIZ, SQRE
//      ..
//      .. Array Arguments ..
//      INTEGER            GIVCOL( LDGCOL, * ), GIVPTR( * ), IWORK( * ),
//     $                   K( * ), PERM( LDGCOL, * )
//      DOUBLE PRECISION   C( * ), D( * ), DIFL( LDU, * ), DIFR( LDU, * ),
//     $                   E( * ), GIVNUM( LDU, * ), POLES( LDU, * ),
//     $                   S( * ), U( LDU, * ), VT( LDU, * ), WORK( * ),
//     $                   Z( LDU, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Using a divide and conquer approach, DLASDA computes the singular
//> value decomposition (SVD) of a real upper bidiagonal N-by-M matrix
//> B with diagonal D and offdiagonal E, where M = N + SQRE. The
//> algorithm computes the singular values in the SVD B = U * S * VT.
//> The orthogonal matrices U and VT are optionally computed in
//> compact form.
//>
//> A related subroutine, DLASD0, computes the singular values and
//> the singular vectors in explicit form.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ICOMPQ
//> \verbatim
//>          ICOMPQ is INTEGER
//>         Specifies whether singular vectors are to be computed
//>         in compact form, as follows
//>         = 0: Compute singular values only.
//>         = 1: Compute singular vectors of upper bidiagonal
//>              matrix in compact form.
//> \endverbatim
//>
//> \param[in] SMLSIZ
//> \verbatim
//>          SMLSIZ is INTEGER
//>         The maximum size of the subproblems at the bottom of the
//>         computation tree.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>         The row dimension of the upper bidiagonal matrix. This is
//>         also the dimension of the main diagonal array D.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>         Specifies the column dimension of the bidiagonal matrix.
//>         = 0: The bidiagonal matrix has column dimension M = N;
//>         = 1: The bidiagonal matrix has column dimension M = N + 1.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension ( N )
//>         On entry D contains the main diagonal of the bidiagonal
//>         matrix. On exit D, if INFO = 0, contains its singular values.
//> \endverbatim
//>
//> \param[in] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension ( M-1 )
//>         Contains the subdiagonal entries of the bidiagonal matrix.
//>         On exit, E has been destroyed.
//> \endverbatim
//>
//> \param[out] U
//> \verbatim
//>          U is DOUBLE PRECISION array,
//>         dimension ( LDU, SMLSIZ ) if ICOMPQ = 1, and not referenced
//>         if ICOMPQ = 0. If ICOMPQ = 1, on exit, U contains the left
//>         singular vector matrices of all subproblems at the bottom
//>         level.
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER, LDU = > N.
//>         The leading dimension of arrays U, VT, DIFL, DIFR, POLES,
//>         GIVNUM, and Z.
//> \endverbatim
//>
//> \param[out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array,
//>         dimension ( LDU, SMLSIZ+1 ) if ICOMPQ = 1, and not referenced
//>         if ICOMPQ = 0. If ICOMPQ = 1, on exit, VT**T contains the right
//>         singular vector matrices of all subproblems at the bottom
//>         level.
//> \endverbatim
//>
//> \param[out] K
//> \verbatim
//>          K is INTEGER array,
//>         dimension ( N ) if ICOMPQ = 1 and dimension 1 if ICOMPQ = 0.
//>         If ICOMPQ = 1, on exit, K(I) is the dimension of the I-th
//>         secular equation on the computation tree.
//> \endverbatim
//>
//> \param[out] DIFL
//> \verbatim
//>          DIFL is DOUBLE PRECISION array, dimension ( LDU, NLVL ),
//>         where NLVL = floor(log_2 (N/SMLSIZ))).
//> \endverbatim
//>
//> \param[out] DIFR
//> \verbatim
//>          DIFR is DOUBLE PRECISION array,
//>                  dimension ( LDU, 2 * NLVL ) if ICOMPQ = 1 and
//>                  dimension ( N ) if ICOMPQ = 0.
//>         If ICOMPQ = 1, on exit, DIFL(1:N, I) and DIFR(1:N, 2 * I - 1)
//>         record distances between singular values on the I-th
//>         level and singular values on the (I -1)-th level, and
//>         DIFR(1:N, 2 * I ) contains the normalizing factors for
//>         the right singular vector matrix. See DLASD8 for details.
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array,
//>                  dimension ( LDU, NLVL ) if ICOMPQ = 1 and
//>                  dimension ( N ) if ICOMPQ = 0.
//>         The first K elements of Z(1, I) contain the components of
//>         the deflation-adjusted updating row vector for subproblems
//>         on the I-th level.
//> \endverbatim
//>
//> \param[out] POLES
//> \verbatim
//>          POLES is DOUBLE PRECISION array,
//>         dimension ( LDU, 2 * NLVL ) if ICOMPQ = 1, and not referenced
//>         if ICOMPQ = 0. If ICOMPQ = 1, on exit, POLES(1, 2*I - 1) and
//>         POLES(1, 2*I) contain  the new and old singular values
//>         involved in the secular equations on the I-th level.
//> \endverbatim
//>
//> \param[out] GIVPTR
//> \verbatim
//>          GIVPTR is INTEGER array,
//>         dimension ( N ) if ICOMPQ = 1, and not referenced if
//>         ICOMPQ = 0. If ICOMPQ = 1, on exit, GIVPTR( I ) records
//>         the number of Givens rotations performed on the I-th
//>         problem on the computation tree.
//> \endverbatim
//>
//> \param[out] GIVCOL
//> \verbatim
//>          GIVCOL is INTEGER array,
//>         dimension ( LDGCOL, 2 * NLVL ) if ICOMPQ = 1, and not
//>         referenced if ICOMPQ = 0. If ICOMPQ = 1, on exit, for each I,
//>         GIVCOL(1, 2 *I - 1) and GIVCOL(1, 2 *I) record the locations
//>         of Givens rotations performed on the I-th level on the
//>         computation tree.
//> \endverbatim
//>
//> \param[in] LDGCOL
//> \verbatim
//>          LDGCOL is INTEGER, LDGCOL = > N.
//>         The leading dimension of arrays GIVCOL and PERM.
//> \endverbatim
//>
//> \param[out] PERM
//> \verbatim
//>          PERM is INTEGER array,
//>         dimension ( LDGCOL, NLVL ) if ICOMPQ = 1, and not referenced
//>         if ICOMPQ = 0. If ICOMPQ = 1, on exit, PERM(1, I) records
//>         permutations done on the I-th level of the computation tree.
//> \endverbatim
//>
//> \param[out] GIVNUM
//> \verbatim
//>          GIVNUM is DOUBLE PRECISION array,
//>         dimension ( LDU,  2 * NLVL ) if ICOMPQ = 1, and not
//>         referenced if ICOMPQ = 0. If ICOMPQ = 1, on exit, for each I,
//>         GIVNUM(1, 2 *I - 1) and GIVNUM(1, 2 *I) record the C- and S-
//>         values of Givens rotations performed on the I-th level on
//>         the computation tree.
//> \endverbatim
//>
//> \param[out] C
//> \verbatim
//>          C is DOUBLE PRECISION array,
//>         dimension ( N ) if ICOMPQ = 1, and dimension 1 if ICOMPQ = 0.
//>         If ICOMPQ = 1 and the I-th subproblem is not square, on exit,
//>         C( I ) contains the C-value of a Givens rotation related to
//>         the right null space of the I-th subproblem.
//> \endverbatim
//>
//> \param[out] S
//> \verbatim
//>          S is DOUBLE PRECISION array, dimension ( N ) if
//>         ICOMPQ = 1, and dimension 1 if ICOMPQ = 0. If ICOMPQ = 1
//>         and the I-th subproblem is not square, on exit, S( I )
//>         contains the S-value of a Givens rotation related to
//>         the right null space of the I-th subproblem.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension
//>         (6 * N + (SMLSIZ + 1)*(SMLSIZ + 1)).
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (7*N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  if INFO = 1, a singular value did not converge
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasda_(int *icompq, int *smlsiz, int *n, int *sqre,
	double *d__, double *e, double *u, int *ldu, double *vt, int *k,
	double *difl, double *difr, double *z__, double *poles, int *givptr,
	int *givcol, int *ldgcol, int *perm, double *givnum, double *c__,
	double *s, double *work, int *iwork, int *info)
{
    // Table of constant values
    int c__0 = 0;
    double c_b11 = 0.;
    double c_b12 = 1.;
    int c__1 = 1;
    int c__2 = 2;

    // System generated locals
    int givcol_dim1, givcol_offset, perm_dim1, perm_offset, difl_dim1,
	    difl_offset, difr_dim1, difr_offset, givnum_dim1, givnum_offset,
	    poles_dim1, poles_offset, u_dim1, u_offset, vt_dim1, vt_offset,
	    z_dim1, z_offset, i__1, i__2;

    // Local variables
    int i__, j, m, i1, ic, lf, nd, ll, nl, vf, nr, vl, im1, ncc, nlf, nrf,
	    vfi, iwk, vli, lvl, nru, ndb1, nlp1, lvl2, nrp1;
    double beta;
    int idxq, nlvl;
    double alpha;
    int inode, ndiml, ndimr, idxqi, itemp;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int sqrei;
    extern /* Subroutine */ int dlasd6_(int *, int *, int *, int *, double *,
	    double *, double *, double *, double *, int *, int *, int *, int *
	    , int *, double *, int *, double *, double *, double *, double *,
	    int *, double *, double *, double *, int *, int *);
    int nwork1, nwork2;
    extern /* Subroutine */ int dlasdq_(char *, int *, int *, int *, int *,
	    int *, double *, double *, double *, int *, double *, int *,
	    double *, int *, double *, int *), dlasdt_(int *, int *, int *,
	    int *, int *, int *, int *), dlaset_(char *, int *, int *, double
	    *, double *, double *, int *), xerbla_(char *, int *);
    int smlszp;

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
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

    // Function Body
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
    //
    //    If the input matrix is too small, call DLASDQ to find the SVD.
    //
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
    //
    //    Book-keeping and  set up the computation tree.
    //
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
    //
    //    for the nodes on bottom level of the tree, solve
    //    their subproblems by DLASDQ.
    //
    ndb1 = (nd + 1) / 2;
    i__1 = nd;
    for (i__ = ndb1; i__ <= i__1; ++i__) {
	//
	//       IC : center row of each node
	//       NL : number of rows of left  subproblem
	//       NR : number of rows of right subproblem
	//       NLF: starting row of the left   subproblem
	//       NRF: starting row of the right  subproblem
	//
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
	    dlaset_("A", &nlp1, &nlp1, &c_b11, &c_b12, &work[nwork1], &smlszp)
		    ;
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
// L10:
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
	    dlaset_("A", &nrp1, &nrp1, &c_b11, &c_b12, &work[nwork1], &smlszp)
		    ;
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
// L20:
	}
// L30:
    }
    //
    //    Now conquer each subproblem bottom-up.
    //
    j = pow_ii(&c__2, &nlvl);
    for (lvl = nlvl; lvl >= 1; --lvl) {
	lvl2 = (lvl << 1) - 1;
	//
	//       Find the first node LF and last node LL on
	//       the current level LVL.
	//
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
// L40:
	}
// L50:
    }
    return 0;
    //
    //    End of DLASDA
    //
} // dlasda_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASDQ computes the SVD of a real bidiagonal matrix with diagonal d and off-diagonal e. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASDQ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasdq.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasdq.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasdq.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASDQ( UPLO, SQRE, N, NCVT, NRU, NCC, D, E, VT, LDVT,
//                         U, LDU, C, LDC, WORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDC, LDU, LDVT, N, NCC, NCVT, NRU, SQRE
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   C( LDC, * ), D( * ), E( * ), U( LDU, * ),
//     $                   VT( LDVT, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASDQ computes the singular value decomposition (SVD) of a real
//> (upper or lower) bidiagonal matrix with diagonal D and offdiagonal
//> E, accumulating the transformations if desired. Letting B denote
//> the input bidiagonal matrix, the algorithm computes orthogonal
//> matrices Q and P such that B = Q * S * P**T (P**T denotes the transpose
//> of P). The singular values S are overwritten on D.
//>
//> The input matrix U  is changed to U  * Q  if desired.
//> The input matrix VT is changed to P**T * VT if desired.
//> The input matrix C  is changed to Q**T * C  if desired.
//>
//> See "Computing  Small Singular Values of Bidiagonal Matrices With
//> Guaranteed High Relative Accuracy," by J. Demmel and W. Kahan,
//> LAPACK Working Note #3, for a detailed description of the algorithm.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>        On entry, UPLO specifies whether the input bidiagonal matrix
//>        is upper or lower bidiagonal, and whether it is square are
//>        not.
//>           UPLO = 'U' or 'u'   B is upper bidiagonal.
//>           UPLO = 'L' or 'l'   B is lower bidiagonal.
//> \endverbatim
//>
//> \param[in] SQRE
//> \verbatim
//>          SQRE is INTEGER
//>        = 0: then the input matrix is N-by-N.
//>        = 1: then the input matrix is N-by-(N+1) if UPLU = 'U' and
//>             (N+1)-by-N if UPLU = 'L'.
//>
//>        The bidiagonal matrix has
//>        N = NL + NR + 1 rows and
//>        M = N + SQRE >= N columns.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>        On entry, N specifies the number of rows and columns
//>        in the matrix. N must be at least 0.
//> \endverbatim
//>
//> \param[in] NCVT
//> \verbatim
//>          NCVT is INTEGER
//>        On entry, NCVT specifies the number of columns of
//>        the matrix VT. NCVT must be at least 0.
//> \endverbatim
//>
//> \param[in] NRU
//> \verbatim
//>          NRU is INTEGER
//>        On entry, NRU specifies the number of rows of
//>        the matrix U. NRU must be at least 0.
//> \endverbatim
//>
//> \param[in] NCC
//> \verbatim
//>          NCC is INTEGER
//>        On entry, NCC specifies the number of columns of
//>        the matrix C. NCC must be at least 0.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>        On entry, D contains the diagonal entries of the
//>        bidiagonal matrix whose SVD is desired. On normal exit,
//>        D contains the singular values in ascending order.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array.
//>        dimension is (N-1) if SQRE = 0 and N if SQRE = 1.
//>        On entry, the entries of E contain the offdiagonal entries
//>        of the bidiagonal matrix whose SVD is desired. On normal
//>        exit, E will contain 0. If the algorithm does not converge,
//>        D and E will contain the diagonal and superdiagonal entries
//>        of a bidiagonal matrix orthogonally equivalent to the one
//>        given as input.
//> \endverbatim
//>
//> \param[in,out] VT
//> \verbatim
//>          VT is DOUBLE PRECISION array, dimension (LDVT, NCVT)
//>        On entry, contains a matrix which on exit has been
//>        premultiplied by P**T, dimension N-by-NCVT if SQRE = 0
//>        and (N+1)-by-NCVT if SQRE = 1 (not referenced if NCVT=0).
//> \endverbatim
//>
//> \param[in] LDVT
//> \verbatim
//>          LDVT is INTEGER
//>        On entry, LDVT specifies the leading dimension of VT as
//>        declared in the calling (sub) program. LDVT must be at
//>        least 1. If NCVT is nonzero LDVT must also be at least N.
//> \endverbatim
//>
//> \param[in,out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension (LDU, N)
//>        On entry, contains a  matrix which on exit has been
//>        postmultiplied by Q, dimension NRU-by-N if SQRE = 0
//>        and NRU-by-(N+1) if SQRE = 1 (not referenced if NRU=0).
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>        On entry, LDU  specifies the leading dimension of U as
//>        declared in the calling (sub) program. LDU must be at
//>        least max( 1, NRU ) .
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC, NCC)
//>        On entry, contains an N-by-NCC matrix which on exit
//>        has been premultiplied by Q**T  dimension N-by-NCC if SQRE = 0
//>        and (N+1)-by-NCC if SQRE = 1 (not referenced if NCC=0).
//> \endverbatim
//>
//> \param[in] LDC
//> \verbatim
//>          LDC is INTEGER
//>        On entry, LDC  specifies the leading dimension of C as
//>        declared in the calling (sub) program. LDC must be at
//>        least 1. If NCC is nonzero, LDC must also be at least N.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (4*N)
//>        Workspace. Only referenced if one of NCVT, NRU, or NCC is
//>        nonzero, and if N is at least 2.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>        On exit, a value of 0 indicates a successful exit.
//>        If INFO < 0, argument number -INFO is illegal.
//>        If INFO > 0, the algorithm did not converge, and INFO
//>        specifies how many superdiagonals did not converge.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasdq_(char *uplo, int *sqre, int *n, int *ncvt, int *
	nru, int *ncc, double *d__, double *e, double *vt, int *ldvt, double *
	u, int *ldu, double *c__, int *ldc, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int c_dim1, c_offset, u_dim1, u_offset, vt_dim1, vt_offset, i__1, i__2;

    // Local variables
    int i__, j;
    double r__, cs, sn;
    int np1, isub;
    double smin;
    int sqre1;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dlasr_(char *, char *, char *, int *, int *,
	    double *, double *, double *, int *), dswap_(int *, double *, int
	    *, double *, int *);
    int iuplo;
    extern /* Subroutine */ int dlartg_(double *, double *, double *, double *
	    , double *), xerbla_(char *, int *), dbdsqr_(char *, int *, int *,
	     int *, int *, double *, double *, double *, int *, double *, int
	    *, double *, int *, double *, int *);
    int rotate;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;
    --e;
    vt_dim1 = *ldvt;
    vt_offset = 1 + vt_dim1;
    vt -= vt_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    *info = 0;
    iuplo = 0;
    if (lsame_(uplo, "U")) {
	iuplo = 1;
    }
    if (lsame_(uplo, "L")) {
	iuplo = 2;
    }
    if (iuplo == 0) {
	*info = -1;
    } else if (*sqre < 0 || *sqre > 1) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ncvt < 0) {
	*info = -4;
    } else if (*nru < 0) {
	*info = -5;
    } else if (*ncc < 0) {
	*info = -6;
    } else if (*ncvt == 0 && *ldvt < 1 || *ncvt > 0 && *ldvt < max(1,*n)) {
	*info = -10;
    } else if (*ldu < max(1,*nru)) {
	*info = -12;
    } else if (*ncc == 0 && *ldc < 1 || *ncc > 0 && *ldc < max(1,*n)) {
	*info = -14;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASDQ", &i__1);
	return 0;
    }
    if (*n == 0) {
	return 0;
    }
    //
    //    ROTATE is true if any singular vectors desired, false otherwise
    //
    rotate = *ncvt > 0 || *nru > 0 || *ncc > 0;
    np1 = *n + 1;
    sqre1 = *sqre;
    //
    //    If matrix non-square upper bidiagonal, rotate to be lower
    //    bidiagonal.  The rotations are on the right.
    //
    if (iuplo == 1 && sqre1 == 1) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (rotate) {
		work[i__] = cs;
		work[*n + i__] = sn;
	    }
// L10:
	}
	dlartg_(&d__[*n], &e[*n], &cs, &sn, &r__);
	d__[*n] = r__;
	e[*n] = 0.;
	if (rotate) {
	    work[*n] = cs;
	    work[*n + *n] = sn;
	}
	iuplo = 2;
	sqre1 = 0;
	//
	//       Update singular vectors if desired.
	//
	if (*ncvt > 0) {
	    dlasr_("L", "V", "F", &np1, ncvt, &work[1], &work[np1], &vt[
		    vt_offset], ldvt);
	}
    }
    //
    //    If matrix lower bidiagonal, rotate to be upper bidiagonal
    //    by applying Givens rotations on the left.
    //
    if (iuplo == 2) {
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dlartg_(&d__[i__], &e[i__], &cs, &sn, &r__);
	    d__[i__] = r__;
	    e[i__] = sn * d__[i__ + 1];
	    d__[i__ + 1] = cs * d__[i__ + 1];
	    if (rotate) {
		work[i__] = cs;
		work[*n + i__] = sn;
	    }
// L20:
	}
	//
	//       If matrix (N+1)-by-N lower bidiagonal, one additional
	//       rotation is needed.
	//
	if (sqre1 == 1) {
	    dlartg_(&d__[*n], &e[*n], &cs, &sn, &r__);
	    d__[*n] = r__;
	    if (rotate) {
		work[*n] = cs;
		work[*n + *n] = sn;
	    }
	}
	//
	//       Update singular vectors if desired.
	//
	if (*nru > 0) {
	    if (sqre1 == 0) {
		dlasr_("R", "V", "F", nru, n, &work[1], &work[np1], &u[
			u_offset], ldu);
	    } else {
		dlasr_("R", "V", "F", nru, &np1, &work[1], &work[np1], &u[
			u_offset], ldu);
	    }
	}
	if (*ncc > 0) {
	    if (sqre1 == 0) {
		dlasr_("L", "V", "F", n, ncc, &work[1], &work[np1], &c__[
			c_offset], ldc);
	    } else {
		dlasr_("L", "V", "F", &np1, ncc, &work[1], &work[np1], &c__[
			c_offset], ldc);
	    }
	}
    }
    //
    //    Call DBDSQR to compute the SVD of the reduced real
    //    N-by-N upper bidiagonal matrix.
    //
    dbdsqr_("U", n, ncvt, nru, ncc, &d__[1], &e[1], &vt[vt_offset], ldvt, &u[
	    u_offset], ldu, &c__[c_offset], ldc, &work[1], info);
    //
    //    Sort the singular values into ascending order (insertion sort on
    //    singular values, but only one transposition per singular vector)
    //
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	//
	//       Scan for smallest D(I).
	//
	isub = i__;
	smin = d__[i__];
	i__2 = *n;
	for (j = i__ + 1; j <= i__2; ++j) {
	    if (d__[j] < smin) {
		isub = j;
		smin = d__[j];
	    }
// L30:
	}
	if (isub != i__) {
	    //
	    //          Swap singular values and vectors.
	    //
	    d__[isub] = d__[i__];
	    d__[i__] = smin;
	    if (*ncvt > 0) {
		dswap_(ncvt, &vt[isub + vt_dim1], ldvt, &vt[i__ + vt_dim1],
			ldvt);
	    }
	    if (*nru > 0) {
		dswap_(nru, &u[isub * u_dim1 + 1], &c__1, &u[i__ * u_dim1 + 1]
			, &c__1);
	    }
	    if (*ncc > 0) {
		dswap_(ncc, &c__[isub + c_dim1], ldc, &c__[i__ + c_dim1], ldc)
			;
	    }
	}
// L40:
    }
    return 0;
    //
    //    End of DLASDQ
    //
} // dlasdq_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASDT creates a tree of subproblems for bidiagonal divide and conquer. Used by sbdsdc.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASDT + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasdt.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasdt.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasdt.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASDT( N, LVL, ND, INODE, NDIML, NDIMR, MSUB )
//
//      .. Scalar Arguments ..
//      INTEGER            LVL, MSUB, N, ND
//      ..
//      .. Array Arguments ..
//      INTEGER            INODE( * ), NDIML( * ), NDIMR( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASDT creates a tree of subproblems for bidiagonal divide and
//> conquer.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          On entry, the number of diagonal elements of the
//>          bidiagonal matrix.
//> \endverbatim
//>
//> \param[out] LVL
//> \verbatim
//>          LVL is INTEGER
//>          On exit, the number of levels on the computation tree.
//> \endverbatim
//>
//> \param[out] ND
//> \verbatim
//>          ND is INTEGER
//>          On exit, the number of nodes on the tree.
//> \endverbatim
//>
//> \param[out] INODE
//> \verbatim
//>          INODE is INTEGER array, dimension ( N )
//>          On exit, centers of subproblems.
//> \endverbatim
//>
//> \param[out] NDIML
//> \verbatim
//>          NDIML is INTEGER array, dimension ( N )
//>          On exit, row dimensions of left children.
//> \endverbatim
//>
//> \param[out] NDIMR
//> \verbatim
//>          NDIMR is INTEGER array, dimension ( N )
//>          On exit, row dimensions of right children.
//> \endverbatim
//>
//> \param[in] MSUB
//> \verbatim
//>          MSUB is INTEGER
//>          On entry, the maximum row dimension each subproblem at the
//>          bottom of the tree can be of.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>     Ming Gu and Huan Ren, Computer Science Division, University of
//>     California at Berkeley, USA
//>
// =====================================================================
/* Subroutine */ int dlasdt_(int *n, int *lvl, int *nd, int *inode, int *
	ndiml, int *ndimr, int *msub)
{
    // System generated locals
    int i__1, i__2;

    // Local variables
    int i__, il, ir, maxn;
    double temp;
    int nlvl, llst, ncrnt;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Find the number of levels on the tree.
    //
    // Parameter adjustments
    --ndimr;
    --ndiml;
    --inode;

    // Function Body
    maxn = max(1,*n);
    temp = log((double) maxn / (double) (*msub + 1)) / log(2.);
    *lvl = (int) temp + 1;
    i__ = *n / 2;
    inode[1] = i__ + 1;
    ndiml[1] = i__;
    ndimr[1] = *n - i__ - 1;
    il = 0;
    ir = 1;
    llst = 1;
    i__1 = *lvl - 1;
    for (nlvl = 1; nlvl <= i__1; ++nlvl) {
	//
	//       Constructing the tree at (NLVL+1)-st level. The number of
	//       nodes created on this level is LLST * 2.
	//
	i__2 = llst - 1;
	for (i__ = 0; i__ <= i__2; ++i__) {
	    il += 2;
	    ir += 2;
	    ncrnt = llst + i__;
	    ndiml[il] = ndiml[ncrnt] / 2;
	    ndimr[il] = ndiml[ncrnt] - ndiml[il] - 1;
	    inode[il] = inode[ncrnt] - ndimr[il] - 1;
	    ndiml[ir] = ndimr[ncrnt] / 2;
	    ndimr[ir] = ndimr[ncrnt] - ndiml[ir] - 1;
	    inode[ir] = inode[ncrnt] + ndiml[ir] + 1;
// L10:
	}
	llst <<= 1;
// L20:
    }
    *nd = (llst << 1) - 1;
    return 0;
    //
    //    End of DLASDT
    //
} // dlasdt_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ1 computes the singular values of a real square bidiagonal matrix. Used by sbdsqr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ1 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq1.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq1.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq1.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ1( N, D, E, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), E( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ1 computes the singular values of a real N-by-N bidiagonal
//> matrix with diagonal D and off-diagonal E. The singular values
//> are computed to high relative accuracy, in the absence of
//> denormalization, underflow and overflow. The algorithm was first
//> presented in
//>
//> "Accurate singular values and differential qd algorithms" by K. V.
//> Fernando and B. N. Parlett, Numer. Math., Vol-67, No. 2, pp. 191-230,
//> 1994,
//>
//> and the present implementation is described in "An implementation of
//> the dqds Algorithm (Positive Case)", LAPACK Working Note.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>        The number of rows and columns in the matrix. N >= 0.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>        On entry, D contains the diagonal elements of the
//>        bidiagonal matrix whose SVD is desired. On normal exit,
//>        D contains the singular values in decreasing order.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N)
//>        On entry, elements E(1:N-1) contain the off-diagonal elements
//>        of the bidiagonal matrix whose SVD is desired.
//>        On exit, E is overwritten.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (4*N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>        = 0: successful exit
//>        < 0: if INFO = -i, the i-th argument had an illegal value
//>        > 0: the algorithm failed
//>             = 1, a split was marked by a positive value in E
//>             = 2, current block of Z not diagonalized after 100*N
//>                  iterations (in inner while loop)  On exit D and E
//>                  represent a matrix with the same singular values
//>                  which the calling subroutine could use to finish the
//>                  computation, or even feed back into DLASQ1
//>             = 3, termination criterion of outer while loop not met
//>                  (program created more than N unreduced blocks)
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasq1_(int *n, double *d__, double *e, double *work,
	int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__2 = 2;
    int c__0 = 0;

    // System generated locals
    int i__1, i__2;
    double d__1, d__2, d__3;

    // Local variables
    int i__;
    double eps;
    extern /* Subroutine */ int dlas2_(double *, double *, double *, double *,
	     double *);
    double scale;
    int iinfo;
    double sigmn;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    double sigmx;
    extern /* Subroutine */ int dlasq2_(int *, double *, int *);
    extern double dlamch_(char *);
    extern /* Subroutine */ int dlascl_(char *, int *, int *, double *,
	    double *, int *, int *, double *, int *, int *);
    double safmin;
    extern /* Subroutine */ int xerbla_(char *, int *), dlasrt_(char *, int *,
	     double *, int *);

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --work;
    --e;
    --d__;

    // Function Body
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("DLASQ1", &i__1);
	return 0;
    } else if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	d__[1] = abs(d__[1]);
	return 0;
    } else if (*n == 2) {
	dlas2_(&d__[1], &e[1], &d__[2], &sigmn, &sigmx);
	d__[1] = sigmx;
	d__[2] = sigmn;
	return 0;
    }
    //
    //    Estimate the largest singular value.
    //
    sigmx = 0.;
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	d__[i__] = (d__1 = d__[i__], abs(d__1));
	// Computing MAX
	d__2 = sigmx, d__3 = (d__1 = e[i__], abs(d__1));
	sigmx = max(d__2,d__3);
// L10:
    }
    d__[*n] = (d__1 = d__[*n], abs(d__1));
    //
    //    Early return if SIGMX is zero (matrix is already diagonal).
    //
    if (sigmx == 0.) {
	dlasrt_("D", n, &d__[1], &iinfo);
	return 0;
    }
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	// Computing MAX
	d__1 = sigmx, d__2 = d__[i__];
	sigmx = max(d__1,d__2);
// L20:
    }
    //
    //    Copy D and E into WORK (in the Z format) and scale (squaring the
    //    input data makes scaling by a power of the radix pointless).
    //
    eps = dlamch_("Precision");
    safmin = dlamch_("Safe minimum");
    scale = sqrt(eps / safmin);
    dcopy_(n, &d__[1], &c__1, &work[1], &c__2);
    i__1 = *n - 1;
    dcopy_(&i__1, &e[1], &c__1, &work[2], &c__2);
    i__1 = (*n << 1) - 1;
    i__2 = (*n << 1) - 1;
    dlascl_("G", &c__0, &c__0, &sigmx, &scale, &i__1, &c__1, &work[1], &i__2,
	    &iinfo);
    //
    //    Compute the q's and e's.
    //
    i__1 = (*n << 1) - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	// Computing 2nd power
	d__1 = work[i__];
	work[i__] = d__1 * d__1;
// L30:
    }
    work[*n * 2] = 0.;
    dlasq2_(n, &work[1], info);
    if (*info == 0) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__[i__] = sqrt(work[i__]);
// L40:
	}
	dlascl_("G", &c__0, &c__0, &scale, &sigmx, n, &c__1, &d__[1], n, &
		iinfo);
    } else if (*info == 2) {
	//
	//    Maximum number of iterations exceeded.  Move data from WORK
	//    into D and E so the calling subroutine can try to finish
	//
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    d__[i__] = sqrt(work[(i__ << 1) - 1]);
	    e[i__] = sqrt(work[i__ * 2]);
	}
	dlascl_("G", &c__0, &c__0, &scale, &sigmx, n, &c__1, &d__[1], n, &
		iinfo);
	dlascl_("G", &c__0, &c__0, &scale, &sigmx, n, &c__1, &e[1], n, &iinfo)
		;
    }
    return 0;
    //
    //    End of DLASQ1
    //
} // dlasq1_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ2 computes all the eigenvalues of the symmetric positive definite tridiagonal matrix associated with the qd Array Z to high relative accuracy. Used by sbdsqr and sstegr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ2( N, Z, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ2 computes all the eigenvalues of the symmetric positive
//> definite tridiagonal matrix associated with the qd array Z to high
//> relative accuracy are computed to high relative accuracy, in the
//> absence of denormalization, underflow and overflow.
//>
//> To see the relation of Z to the tridiagonal matrix, let L be a
//> unit lower bidiagonal matrix with subdiagonals Z(2,4,6,,..) and
//> let U be an upper bidiagonal matrix with 1's above and diagonal
//> Z(1,3,5,,..). The tridiagonal is L*U or, if you prefer, the
//> symmetric tridiagonal to which it is similar.
//>
//> Note : DLASQ2 defines a logical variable, IEEE, which is true
//> on machines which follow ieee-754 floating-point standard in their
//> handling of infinities and NaNs, and false otherwise. This variable
//> is passed to DLASQ3.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>        The number of rows and columns in the matrix. N >= 0.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N )
//>        On entry Z holds the qd array. On exit, entries 1 to N hold
//>        the eigenvalues in decreasing order, Z( 2*N+1 ) holds the
//>        trace, and Z( 2*N+2 ) holds the sum of the eigenvalues. If
//>        N > 2, then Z( 2*N+3 ) holds the iteration count, Z( 2*N+4 )
//>        holds NDIVS/NIN^2, and Z( 2*N+5 ) holds the percentage of
//>        shifts that failed.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>        = 0: successful exit
//>        < 0: if the i-th argument is a scalar and had an illegal
//>             value, then INFO = -i, if the i-th argument is an
//>             array and the j-entry had an illegal value, then
//>             INFO = -(i*100+j)
//>        > 0: the algorithm failed
//>              = 1, a split was marked by a positive value in E
//>              = 2, current block of Z not diagonalized after 100*N
//>                   iterations (in inner while loop).  On exit Z holds
//>                   a qd array with the same eigenvalues as the given Z.
//>              = 3, termination criterion of outer while loop not met
//>                   (program created more than N unreduced blocks)
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup auxOTHERcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Local Variables: I0:N0 defines a current unreduced segment of Z.
//>  The shifts are accumulated in SIGMA. Iteration count is in ITER.
//>  Ping-pong is controlled by PP (alternates between 0 and 1).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlasq2_(int *n, double *z__, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__2 = 2;
    int c__10 = 10;
    int c__3 = 3;
    int c__4 = 4;
    int c__11 = 11;

    // System generated locals
    int i__1, i__2, i__3;
    double d__1, d__2;

    // Local variables
    double d__, e, g;
    int k;
    double s, t;
    int i0, i1, i4, n0, n1;
    double dn;
    int pp;
    double dn1, dn2, dee, eps, tau, tol;
    int ipn4;
    double tol2;
    int ieee;
    int nbig;
    double dmin__, emin, emax;
    int kmin, ndiv, iter;
    double qmin, temp, qmax, zmax;
    int splt;
    double dmin1, dmin2;
    int nfail;
    double desig, trace, sigma;
    int iinfo;
    double tempe, tempq;
    int ttype;
    extern /* Subroutine */ int dlasq3_(int *, int *, double *, int *, double
	    *, double *, double *, double *, int *, int *, int *, int *, int *
	    , double *, double *, double *, double *, double *, double *,
	    double *);
    extern double dlamch_(char *);
    double deemin;
    int iwhila, iwhilb;
    double oldemn, safmin;
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dlasrt_(char *, int *, double *, int *);

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments.
    //    (in case DLASQ2 is not called by DLASQ1)
    //
    // Parameter adjustments
    --z__;

    // Function Body
    *info = 0;
    eps = dlamch_("Precision");
    safmin = dlamch_("Safe minimum");
    tol = eps * 100.;
    // Computing 2nd power
    d__1 = tol;
    tol2 = d__1 * d__1;
    if (*n < 0) {
	*info = -1;
	xerbla_("DLASQ2", &c__1);
	return 0;
    } else if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	//
	//       1-by-1 case.
	//
	if (z__[1] < 0.) {
	    *info = -201;
	    xerbla_("DLASQ2", &c__2);
	}
	return 0;
    } else if (*n == 2) {
	//
	//       2-by-2 case.
	//
	if (z__[2] < 0. || z__[3] < 0.) {
	    *info = -2;
	    xerbla_("DLASQ2", &c__2);
	    return 0;
	} else if (z__[3] > z__[1]) {
	    d__ = z__[3];
	    z__[3] = z__[1];
	    z__[1] = d__;
	}
	z__[5] = z__[1] + z__[2] + z__[3];
	if (z__[2] > z__[3] * tol2) {
	    t = (z__[1] - z__[3] + z__[2]) * .5;
	    s = z__[3] * (z__[2] / t);
	    if (s <= t) {
		s = z__[3] * (z__[2] / (t * (sqrt(s / t + 1.) + 1.)));
	    } else {
		s = z__[3] * (z__[2] / (t + sqrt(t) * sqrt(t + s)));
	    }
	    t = z__[1] + (s + z__[2]);
	    z__[3] *= z__[1] / t;
	    z__[1] = t;
	}
	z__[2] = z__[3];
	z__[6] = z__[2] + z__[1];
	return 0;
    }
    //
    //    Check for negative data and compute sums of q's and e's.
    //
    z__[*n * 2] = 0.;
    emin = z__[2];
    qmax = 0.;
    zmax = 0.;
    d__ = 0.;
    e = 0.;
    i__1 = *n - 1 << 1;
    for (k = 1; k <= i__1; k += 2) {
	if (z__[k] < 0.) {
	    *info = -(k + 200);
	    xerbla_("DLASQ2", &c__2);
	    return 0;
	} else if (z__[k + 1] < 0.) {
	    *info = -(k + 201);
	    xerbla_("DLASQ2", &c__2);
	    return 0;
	}
	d__ += z__[k];
	e += z__[k + 1];
	// Computing MAX
	d__1 = qmax, d__2 = z__[k];
	qmax = max(d__1,d__2);
	// Computing MIN
	d__1 = emin, d__2 = z__[k + 1];
	emin = min(d__1,d__2);
	// Computing MAX
	d__1 = max(qmax,zmax), d__2 = z__[k + 1];
	zmax = max(d__1,d__2);
// L10:
    }
    if (z__[(*n << 1) - 1] < 0.) {
	*info = -((*n << 1) + 199);
	xerbla_("DLASQ2", &c__2);
	return 0;
    }
    d__ += z__[(*n << 1) - 1];
    // Computing MAX
    d__1 = qmax, d__2 = z__[(*n << 1) - 1];
    qmax = max(d__1,d__2);
    zmax = max(qmax,zmax);
    //
    //    Check for diagonality.
    //
    if (e == 0.) {
	i__1 = *n;
	for (k = 2; k <= i__1; ++k) {
	    z__[k] = z__[(k << 1) - 1];
// L20:
	}
	dlasrt_("D", n, &z__[1], &iinfo);
	z__[(*n << 1) - 1] = d__;
	return 0;
    }
    trace = d__ + e;
    //
    //    Check for zero data.
    //
    if (trace == 0.) {
	z__[(*n << 1) - 1] = 0.;
	return 0;
    }
    //
    //    Check whether the machine is IEEE conformable.
    //
    ieee = ilaenv_(&c__10, "DLASQ2", "N", &c__1, &c__2, &c__3, &c__4) == 1 &&
	    ilaenv_(&c__11, "DLASQ2", "N", &c__1, &c__2, &c__3, &c__4) == 1;
    //
    //    Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
    //
    for (k = *n << 1; k >= 2; k += -2) {
	z__[k * 2] = 0.;
	z__[(k << 1) - 1] = z__[k];
	z__[(k << 1) - 2] = 0.;
	z__[(k << 1) - 3] = z__[k - 1];
// L30:
    }
    i0 = 1;
    n0 = *n;
    //
    //    Reverse the qd-array, if warranted.
    //
    if (z__[(i0 << 2) - 3] * 1.5 < z__[(n0 << 2) - 3]) {
	ipn4 = i0 + n0 << 2;
	i__1 = i0 + n0 - 1 << 1;
	for (i4 = i0 << 2; i4 <= i__1; i4 += 4) {
	    temp = z__[i4 - 3];
	    z__[i4 - 3] = z__[ipn4 - i4 - 3];
	    z__[ipn4 - i4 - 3] = temp;
	    temp = z__[i4 - 1];
	    z__[i4 - 1] = z__[ipn4 - i4 - 5];
	    z__[ipn4 - i4 - 5] = temp;
// L40:
	}
    }
    //
    //    Initial split checking via dqd and Li's test.
    //
    pp = 0;
    for (k = 1; k <= 2; ++k) {
	d__ = z__[(n0 << 2) + pp - 3];
	i__1 = (i0 << 2) + pp;
	for (i4 = (n0 - 1 << 2) + pp; i4 >= i__1; i4 += -4) {
	    if (z__[i4 - 1] <= tol2 * d__) {
		z__[i4 - 1] = -0.;
		d__ = z__[i4 - 3];
	    } else {
		d__ = z__[i4 - 3] * (d__ / (d__ + z__[i4 - 1]));
	    }
// L50:
	}
	//
	//       dqd maps Z to ZZ plus Li's test.
	//
	emin = z__[(i0 << 2) + pp + 1];
	d__ = z__[(i0 << 2) + pp - 3];
	i__1 = (n0 - 1 << 2) + pp;
	for (i4 = (i0 << 2) + pp; i4 <= i__1; i4 += 4) {
	    z__[i4 - (pp << 1) - 2] = d__ + z__[i4 - 1];
	    if (z__[i4 - 1] <= tol2 * d__) {
		z__[i4 - 1] = -0.;
		z__[i4 - (pp << 1) - 2] = d__;
		z__[i4 - (pp << 1)] = 0.;
		d__ = z__[i4 + 1];
	    } else if (safmin * z__[i4 + 1] < z__[i4 - (pp << 1) - 2] &&
		    safmin * z__[i4 - (pp << 1) - 2] < z__[i4 + 1]) {
		temp = z__[i4 + 1] / z__[i4 - (pp << 1) - 2];
		z__[i4 - (pp << 1)] = z__[i4 - 1] * temp;
		d__ *= temp;
	    } else {
		z__[i4 - (pp << 1)] = z__[i4 + 1] * (z__[i4 - 1] / z__[i4 - (
			pp << 1) - 2]);
		d__ = z__[i4 + 1] * (d__ / z__[i4 - (pp << 1) - 2]);
	    }
	    // Computing MIN
	    d__1 = emin, d__2 = z__[i4 - (pp << 1)];
	    emin = min(d__1,d__2);
// L60:
	}
	z__[(n0 << 2) - pp - 2] = d__;
	//
	//       Now find qmax.
	//
	qmax = z__[(i0 << 2) - pp - 2];
	i__1 = (n0 << 2) - pp - 2;
	for (i4 = (i0 << 2) - pp + 2; i4 <= i__1; i4 += 4) {
	    // Computing MAX
	    d__1 = qmax, d__2 = z__[i4];
	    qmax = max(d__1,d__2);
// L70:
	}
	//
	//       Prepare for the next iteration on K.
	//
	pp = 1 - pp;
// L80:
    }
    //
    //    Initialise variables to pass to DLASQ3.
    //
    ttype = 0;
    dmin1 = 0.;
    dmin2 = 0.;
    dn = 0.;
    dn1 = 0.;
    dn2 = 0.;
    g = 0.;
    tau = 0.;
    iter = 2;
    nfail = 0;
    ndiv = n0 - i0 << 1;
    i__1 = *n + 1;
    for (iwhila = 1; iwhila <= i__1; ++iwhila) {
	if (n0 < 1) {
	    goto L170;
	}
	//
	//       While array unfinished do
	//
	//       E(N0) holds the value of SIGMA when submatrix in I0:N0
	//       splits from the rest of the array, but is negated.
	//
	desig = 0.;
	if (n0 == *n) {
	    sigma = 0.;
	} else {
	    sigma = -z__[(n0 << 2) - 1];
	}
	if (sigma < 0.) {
	    *info = 1;
	    return 0;
	}
	//
	//       Find last unreduced submatrix's top index I0, find QMAX and
	//       EMIN. Find Gershgorin-type bound if Q's much greater than E's.
	//
	emax = 0.;
	if (n0 > i0) {
	    emin = (d__1 = z__[(n0 << 2) - 5], abs(d__1));
	} else {
	    emin = 0.;
	}
	qmin = z__[(n0 << 2) - 3];
	qmax = qmin;
	for (i4 = n0 << 2; i4 >= 8; i4 += -4) {
	    if (z__[i4 - 5] <= 0.) {
		goto L100;
	    }
	    if (qmin >= emax * 4.) {
		// Computing MIN
		d__1 = qmin, d__2 = z__[i4 - 3];
		qmin = min(d__1,d__2);
		// Computing MAX
		d__1 = emax, d__2 = z__[i4 - 5];
		emax = max(d__1,d__2);
	    }
	    // Computing MAX
	    d__1 = qmax, d__2 = z__[i4 - 7] + z__[i4 - 5];
	    qmax = max(d__1,d__2);
	    // Computing MIN
	    d__1 = emin, d__2 = z__[i4 - 5];
	    emin = min(d__1,d__2);
// L90:
	}
	i4 = 4;
L100:
	i0 = i4 / 4;
	pp = 0;
	if (n0 - i0 > 1) {
	    dee = z__[(i0 << 2) - 3];
	    deemin = dee;
	    kmin = i0;
	    i__2 = (n0 << 2) - 3;
	    for (i4 = (i0 << 2) + 1; i4 <= i__2; i4 += 4) {
		dee = z__[i4] * (dee / (dee + z__[i4 - 2]));
		if (dee <= deemin) {
		    deemin = dee;
		    kmin = (i4 + 3) / 4;
		}
// L110:
	    }
	    if (kmin - i0 << 1 < n0 - kmin && deemin <= z__[(n0 << 2) - 3] *
		    .5) {
		ipn4 = i0 + n0 << 2;
		pp = 2;
		i__2 = i0 + n0 - 1 << 1;
		for (i4 = i0 << 2; i4 <= i__2; i4 += 4) {
		    temp = z__[i4 - 3];
		    z__[i4 - 3] = z__[ipn4 - i4 - 3];
		    z__[ipn4 - i4 - 3] = temp;
		    temp = z__[i4 - 2];
		    z__[i4 - 2] = z__[ipn4 - i4 - 2];
		    z__[ipn4 - i4 - 2] = temp;
		    temp = z__[i4 - 1];
		    z__[i4 - 1] = z__[ipn4 - i4 - 5];
		    z__[ipn4 - i4 - 5] = temp;
		    temp = z__[i4];
		    z__[i4] = z__[ipn4 - i4 - 4];
		    z__[ipn4 - i4 - 4] = temp;
// L120:
		}
	    }
	}
	//
	//       Put -(initial shift) into DMIN.
	//
	// Computing MAX
	d__1 = 0., d__2 = qmin - sqrt(qmin) * 2. * sqrt(emax);
	dmin__ = -max(d__1,d__2);
	//
	//       Now I0:N0 is unreduced.
	//       PP = 0 for ping, PP = 1 for pong.
	//       PP = 2 indicates that flipping was applied to the Z array and
	//              and that the tests for deflation upon entry in DLASQ3
	//              should not be performed.
	//
	nbig = (n0 - i0 + 1) * 100;
	i__2 = nbig;
	for (iwhilb = 1; iwhilb <= i__2; ++iwhilb) {
	    if (i0 > n0) {
		goto L150;
	    }
	    //
	    //          While submatrix unfinished take a good dqds step.
	    //
	    dlasq3_(&i0, &n0, &z__[1], &pp, &dmin__, &sigma, &desig, &qmax, &
		    nfail, &iter, &ndiv, &ieee, &ttype, &dmin1, &dmin2, &dn, &
		    dn1, &dn2, &g, &tau);
	    pp = 1 - pp;
	    //
	    //          When EMIN is very small check for splits.
	    //
	    if (pp == 0 && n0 - i0 >= 3) {
		if (z__[n0 * 4] <= tol2 * qmax || z__[(n0 << 2) - 1] <= tol2 *
			 sigma) {
		    splt = i0 - 1;
		    qmax = z__[(i0 << 2) - 3];
		    emin = z__[(i0 << 2) - 1];
		    oldemn = z__[i0 * 4];
		    i__3 = n0 - 3 << 2;
		    for (i4 = i0 << 2; i4 <= i__3; i4 += 4) {
			if (z__[i4] <= tol2 * z__[i4 - 3] || z__[i4 - 1] <=
				tol2 * sigma) {
			    z__[i4 - 1] = -sigma;
			    splt = i4 / 4;
			    qmax = 0.;
			    emin = z__[i4 + 3];
			    oldemn = z__[i4 + 4];
			} else {
			    // Computing MAX
			    d__1 = qmax, d__2 = z__[i4 + 1];
			    qmax = max(d__1,d__2);
			    // Computing MIN
			    d__1 = emin, d__2 = z__[i4 - 1];
			    emin = min(d__1,d__2);
			    // Computing MIN
			    d__1 = oldemn, d__2 = z__[i4];
			    oldemn = min(d__1,d__2);
			}
// L130:
		    }
		    z__[(n0 << 2) - 1] = emin;
		    z__[n0 * 4] = oldemn;
		    i0 = splt + 1;
		}
	    }
// L140:
	}
	*info = 2;
	//
	//       Maximum number of iterations exceeded, restore the shift
	//       SIGMA and place the new d's and e's in a qd array.
	//       This might need to be done for several blocks
	//
	i1 = i0;
	n1 = n0;
L145:
	tempq = z__[(i0 << 2) - 3];
	z__[(i0 << 2) - 3] += sigma;
	i__2 = n0;
	for (k = i0 + 1; k <= i__2; ++k) {
	    tempe = z__[(k << 2) - 5];
	    z__[(k << 2) - 5] *= tempq / z__[(k << 2) - 7];
	    tempq = z__[(k << 2) - 3];
	    z__[(k << 2) - 3] = z__[(k << 2) - 3] + sigma + tempe - z__[(k <<
		    2) - 5];
	}
	//
	//       Prepare to do this on the previous block if there is one
	//
	if (i1 > 1) {
	    n1 = i1 - 1;
	    while(i1 >= 2 && z__[(i1 << 2) - 5] >= 0.) {
		--i1;
	    }
	    sigma = -z__[(n1 << 2) - 1];
	    goto L145;
	}
	i__2 = *n;
	for (k = 1; k <= i__2; ++k) {
	    z__[(k << 1) - 1] = z__[(k << 2) - 3];
	    //
	    //       Only the block 1..N0 is unfinished.  The rest of the e's
	    //       must be essentially zero, although sometimes other data
	    //       has been stored in them.
	    //
	    if (k < n0) {
		z__[k * 2] = z__[(k << 2) - 1];
	    } else {
		z__[k * 2] = 0.;
	    }
	}
	return 0;
	//
	//       end IWHILB
	//
L150:
// L160:
	;
    }
    *info = 3;
    return 0;
    //
    //    end IWHILA
    //
L170:
    //
    //    Move q's to the front.
    //
    i__1 = *n;
    for (k = 2; k <= i__1; ++k) {
	z__[k] = z__[(k << 2) - 3];
// L180:
    }
    //
    //    Sort and compute sum of eigenvalues.
    //
    dlasrt_("D", n, &z__[1], &iinfo);
    e = 0.;
    for (k = *n; k >= 1; --k) {
	e += z__[k];
// L190:
    }
    //
    //    Store trace, sum(eigenvalues) and information on performance.
    //
    z__[(*n << 1) + 1] = trace;
    z__[(*n << 1) + 2] = e;
    z__[(*n << 1) + 3] = (double) iter;
    // Computing 2nd power
    i__1 = *n;
    z__[(*n << 1) + 4] = (double) ndiv / (double) (i__1 * i__1);
    z__[(*n << 1) + 5] = nfail * 100. / (double) iter;
    return 0;
    //
    //    End of DLASQ2
    //
} // dlasq2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ3 checks for deflation, computes a shift and calls dqds. Used by sbdsqr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ3 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq3.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq3.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq3.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ3( I0, N0, Z, PP, DMIN, SIGMA, DESIG, QMAX, NFAIL,
//                         ITER, NDIV, IEEE, TTYPE, DMIN1, DMIN2, DN, DN1,
//                         DN2, G, TAU )
//
//      .. Scalar Arguments ..
//      LOGICAL            IEEE
//      INTEGER            I0, ITER, N0, NDIV, NFAIL, PP
//      DOUBLE PRECISION   DESIG, DMIN, DMIN1, DMIN2, DN, DN1, DN2, G,
//     $                   QMAX, SIGMA, TAU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ3 checks for deflation, computes a shift (TAU) and calls dqds.
//> In case of failure it changes shifts, and tries again until output
//> is positive.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>         First index.
//> \endverbatim
//>
//> \param[in,out] N0
//> \verbatim
//>          N0 is INTEGER
//>         Last index.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N0 )
//>         Z holds the qd array.
//> \endverbatim
//>
//> \param[in,out] PP
//> \verbatim
//>          PP is INTEGER
//>         PP=0 for ping, PP=1 for pong.
//>         PP=2 indicates that flipping was applied to the Z array
//>         and that the initial tests for deflation should not be
//>         performed.
//> \endverbatim
//>
//> \param[out] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>         Minimum value of d.
//> \endverbatim
//>
//> \param[out] SIGMA
//> \verbatim
//>          SIGMA is DOUBLE PRECISION
//>         Sum of shifts used in current segment.
//> \endverbatim
//>
//> \param[in,out] DESIG
//> \verbatim
//>          DESIG is DOUBLE PRECISION
//>         Lower order part of SIGMA
//> \endverbatim
//>
//> \param[in] QMAX
//> \verbatim
//>          QMAX is DOUBLE PRECISION
//>         Maximum value of q.
//> \endverbatim
//>
//> \param[in,out] NFAIL
//> \verbatim
//>          NFAIL is INTEGER
//>         Increment NFAIL by 1 each time the shift was too big.
//> \endverbatim
//>
//> \param[in,out] ITER
//> \verbatim
//>          ITER is INTEGER
//>         Increment ITER by 1 for each iteration.
//> \endverbatim
//>
//> \param[in,out] NDIV
//> \verbatim
//>          NDIV is INTEGER
//>         Increment NDIV by 1 for each division.
//> \endverbatim
//>
//> \param[in] IEEE
//> \verbatim
//>          IEEE is LOGICAL
//>         Flag for IEEE or non IEEE arithmetic (passed to DLASQ5).
//> \endverbatim
//>
//> \param[in,out] TTYPE
//> \verbatim
//>          TTYPE is INTEGER
//>         Shift type.
//> \endverbatim
//>
//> \param[in,out] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DN1
//> \verbatim
//>          DN1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DN2
//> \verbatim
//>          DN2 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] G
//> \verbatim
//>          G is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>
//>         These are passed as arguments in order to save their values
//>         between calls to DLASQ3.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasq3_(int *i0, int *n0, double *z__, int *pp, double *
	dmin__, double *sigma, double *desig, double *qmax, int *nfail, int *
	iter, int *ndiv, int *ieee, int *ttype, double *dmin1, double *dmin2,
	double *dn, double *dn1, double *dn2, double *g, double *tau)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Local variables
    double s, t;
    int j4, nn;
    double eps, tol;
    int n0in, ipn4;
    double tol2, temp;
    extern /* Subroutine */ int dlasq4_(int *, int *, double *, int *, int *,
	    double *, double *, double *, double *, double *, double *,
	    double *, int *, double *), dlasq5_(int *, int *, double *, int *,
	     double *, double *, double *, double *, double *, double *,
	    double *, double *, int *, double *), dlasq6_(int *, int *,
	    double *, int *, double *, double *, double *, double *, double *,
	     double *);
    extern double dlamch_(char *);
    extern int disnan_(double *);

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. External Function ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --z__;

    // Function Body
    n0in = *n0;
    eps = dlamch_("Precision");
    tol = eps * 100.;
    // Computing 2nd power
    d__1 = tol;
    tol2 = d__1 * d__1;
    //
    //    Check for deflation.
    //
L10:
    if (*n0 < *i0) {
	return 0;
    }
    if (*n0 == *i0) {
	goto L20;
    }
    nn = (*n0 << 2) + *pp;
    if (*n0 == *i0 + 1) {
	goto L40;
    }
    //
    //    Check whether E(N0-1) is negligible, 1 eigenvalue.
    //
    if (z__[nn - 5] > tol2 * (*sigma + z__[nn - 3]) && z__[nn - (*pp << 1) -
	    4] > tol2 * z__[nn - 7]) {
	goto L30;
    }
L20:
    z__[(*n0 << 2) - 3] = z__[(*n0 << 2) + *pp - 3] + *sigma;
    --(*n0);
    goto L10;
    //
    //    Check  whether E(N0-2) is negligible, 2 eigenvalues.
    //
L30:
    if (z__[nn - 9] > tol2 * *sigma && z__[nn - (*pp << 1) - 8] > tol2 * z__[
	    nn - 11]) {
	goto L50;
    }
L40:
    if (z__[nn - 3] > z__[nn - 7]) {
	s = z__[nn - 3];
	z__[nn - 3] = z__[nn - 7];
	z__[nn - 7] = s;
    }
    t = (z__[nn - 7] - z__[nn - 3] + z__[nn - 5]) * .5;
    if (z__[nn - 5] > z__[nn - 3] * tol2 && t != 0.) {
	s = z__[nn - 3] * (z__[nn - 5] / t);
	if (s <= t) {
	    s = z__[nn - 3] * (z__[nn - 5] / (t * (sqrt(s / t + 1.) + 1.)));
	} else {
	    s = z__[nn - 3] * (z__[nn - 5] / (t + sqrt(t) * sqrt(t + s)));
	}
	t = z__[nn - 7] + (s + z__[nn - 5]);
	z__[nn - 3] *= z__[nn - 7] / t;
	z__[nn - 7] = t;
    }
    z__[(*n0 << 2) - 7] = z__[nn - 7] + *sigma;
    z__[(*n0 << 2) - 3] = z__[nn - 3] + *sigma;
    *n0 += -2;
    goto L10;
L50:
    if (*pp == 2) {
	*pp = 0;
    }
    //
    //    Reverse the qd-array, if warranted.
    //
    if (*dmin__ <= 0. || *n0 < n0in) {
	if (z__[(*i0 << 2) + *pp - 3] * 1.5 < z__[(*n0 << 2) + *pp - 3]) {
	    ipn4 = *i0 + *n0 << 2;
	    i__1 = *i0 + *n0 - 1 << 1;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		temp = z__[j4 - 3];
		z__[j4 - 3] = z__[ipn4 - j4 - 3];
		z__[ipn4 - j4 - 3] = temp;
		temp = z__[j4 - 2];
		z__[j4 - 2] = z__[ipn4 - j4 - 2];
		z__[ipn4 - j4 - 2] = temp;
		temp = z__[j4 - 1];
		z__[j4 - 1] = z__[ipn4 - j4 - 5];
		z__[ipn4 - j4 - 5] = temp;
		temp = z__[j4];
		z__[j4] = z__[ipn4 - j4 - 4];
		z__[ipn4 - j4 - 4] = temp;
// L60:
	    }
	    if (*n0 - *i0 <= 4) {
		z__[(*n0 << 2) + *pp - 1] = z__[(*i0 << 2) + *pp - 1];
		z__[(*n0 << 2) - *pp] = z__[(*i0 << 2) - *pp];
	    }
	    // Computing MIN
	    d__1 = *dmin2, d__2 = z__[(*n0 << 2) + *pp - 1];
	    *dmin2 = min(d__1,d__2);
	    // Computing MIN
	    d__1 = z__[(*n0 << 2) + *pp - 1], d__2 = z__[(*i0 << 2) + *pp - 1]
		    , d__1 = min(d__1,d__2), d__2 = z__[(*i0 << 2) + *pp + 3];
	    z__[(*n0 << 2) + *pp - 1] = min(d__1,d__2);
	    // Computing MIN
	    d__1 = z__[(*n0 << 2) - *pp], d__2 = z__[(*i0 << 2) - *pp], d__1 =
		     min(d__1,d__2), d__2 = z__[(*i0 << 2) - *pp + 4];
	    z__[(*n0 << 2) - *pp] = min(d__1,d__2);
	    // Computing MAX
	    d__1 = *qmax, d__2 = z__[(*i0 << 2) + *pp - 3], d__1 = max(d__1,
		    d__2), d__2 = z__[(*i0 << 2) + *pp + 1];
	    *qmax = max(d__1,d__2);
	    *dmin__ = -0.;
	}
    }
    //
    //    Choose a shift.
    //
    dlasq4_(i0, n0, &z__[1], pp, &n0in, dmin__, dmin1, dmin2, dn, dn1, dn2,
	    tau, ttype, g);
    //
    //    Call dqds until DMIN > 0.
    //
L70:
    dlasq5_(i0, n0, &z__[1], pp, tau, sigma, dmin__, dmin1, dmin2, dn, dn1,
	    dn2, ieee, &eps);
    *ndiv += *n0 - *i0 + 2;
    ++(*iter);
    //
    //    Check status.
    //
    if (*dmin__ >= 0. && *dmin1 >= 0.) {
	//
	//       Success.
	//
	goto L90;
    } else if (*dmin__ < 0. && *dmin1 > 0. && z__[(*n0 - 1 << 2) - *pp] < tol
	    * (*sigma + *dn1) && abs(*dn) < tol * *sigma) {
	//
	//       Convergence hidden by negative DN.
	//
	z__[(*n0 - 1 << 2) - *pp + 2] = 0.;
	*dmin__ = 0.;
	goto L90;
    } else if (*dmin__ < 0.) {
	//
	//       TAU too big. Select new TAU and try again.
	//
	++(*nfail);
	if (*ttype < -22) {
	    //
	    //          Failed twice. Play it safe.
	    //
	    *tau = 0.;
	} else if (*dmin1 > 0.) {
	    //
	    //          Late failure. Gives excellent shift.
	    //
	    *tau = (*tau + *dmin__) * (1. - eps * 2.);
	    *ttype += -11;
	} else {
	    //
	    //          Early failure. Divide by 4.
	    //
	    *tau *= .25;
	    *ttype += -12;
	}
	goto L70;
    } else if (disnan_(dmin__)) {
	//
	//       NaN.
	//
	if (*tau == 0.) {
	    goto L80;
	} else {
	    *tau = 0.;
	    goto L70;
	}
    } else {
	//
	//       Possible underflow. Play it safe.
	//
	goto L80;
    }
    //
    //    Risk of underflow.
    //
L80:
    dlasq6_(i0, n0, &z__[1], pp, dmin__, dmin1, dmin2, dn, dn1, dn2);
    *ndiv += *n0 - *i0 + 2;
    ++(*iter);
    *tau = 0.;
L90:
    if (*tau < *sigma) {
	*desig += *tau;
	t = *sigma + *desig;
	*desig -= t - *sigma;
    } else {
	t = *sigma + *tau;
	*desig = *sigma - (t - *tau) + *desig;
    }
    *sigma = t;
    return 0;
    //
    //    End of DLASQ3
    //
} // dlasq3_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ4 computes an approximation to the smallest eigenvalue using values of d from the previous transform. Used by sbdsqr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ4 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq4.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq4.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq4.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ4( I0, N0, Z, PP, N0IN, DMIN, DMIN1, DMIN2, DN,
//                         DN1, DN2, TAU, TTYPE, G )
//
//      .. Scalar Arguments ..
//      INTEGER            I0, N0, N0IN, PP, TTYPE
//      DOUBLE PRECISION   DMIN, DMIN1, DMIN2, DN, DN1, DN2, G, TAU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ4 computes an approximation TAU to the smallest eigenvalue
//> using values of d from the previous transform.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>        First index.
//> \endverbatim
//>
//> \param[in] N0
//> \verbatim
//>          N0 is INTEGER
//>        Last index.
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N0 )
//>        Z holds the qd array.
//> \endverbatim
//>
//> \param[in] PP
//> \verbatim
//>          PP is INTEGER
//>        PP=0 for ping, PP=1 for pong.
//> \endverbatim
//>
//> \param[in] N0IN
//> \verbatim
//>          N0IN is INTEGER
//>        The value of N0 at start of EIGTEST.
//> \endverbatim
//>
//> \param[in] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>        Minimum value of d.
//> \endverbatim
//>
//> \param[in] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ).
//> \endverbatim
//>
//> \param[in] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
//> \endverbatim
//>
//> \param[in] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//>        d(N)
//> \endverbatim
//>
//> \param[in] DN1
//> \verbatim
//>          DN1 is DOUBLE PRECISION
//>        d(N-1)
//> \endverbatim
//>
//> \param[in] DN2
//> \verbatim
//>          DN2 is DOUBLE PRECISION
//>        d(N-2)
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>        This is the shift.
//> \endverbatim
//>
//> \param[out] TTYPE
//> \verbatim
//>          TTYPE is INTEGER
//>        Shift type.
//> \endverbatim
//>
//> \param[in,out] G
//> \verbatim
//>          G is DOUBLE PRECISION
//>        G is passed as an argument in order to save its value between
//>        calls to DLASQ4.
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup auxOTHERcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  CNST1 = 9/16
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlasq4_(int *i0, int *n0, double *z__, int *pp, int *
	n0in, double *dmin__, double *dmin1, double *dmin2, double *dn,
	double *dn1, double *dn2, double *tau, int *ttype, double *g)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Local variables
    double s, a2, b1, b2;
    int i4, nn, np;
    double gam, gap1, gap2;

    //
    // -- LAPACK computational routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    A negative DMIN forces the shift to take that absolute value
    //    TTYPE records the type of shift.
    //
    // Parameter adjustments
    --z__;

    // Function Body
    if (*dmin__ <= 0.) {
	*tau = -(*dmin__);
	*ttype = -1;
	return 0;
    }
    nn = (*n0 << 2) + *pp;
    if (*n0in == *n0) {
	//
	//       No eigenvalues deflated.
	//
	if (*dmin__ == *dn || *dmin__ == *dn1) {
	    b1 = sqrt(z__[nn - 3]) * sqrt(z__[nn - 5]);
	    b2 = sqrt(z__[nn - 7]) * sqrt(z__[nn - 9]);
	    a2 = z__[nn - 7] + z__[nn - 5];
	    //
	    //          Cases 2 and 3.
	    //
	    if (*dmin__ == *dn && *dmin1 == *dn1) {
		gap2 = *dmin2 - a2 - *dmin2 * .25;
		if (gap2 > 0. && gap2 > b2) {
		    gap1 = a2 - *dn - b2 / gap2 * b2;
		} else {
		    gap1 = a2 - *dn - (b1 + b2);
		}
		if (gap1 > 0. && gap1 > b1) {
		    // Computing MAX
		    d__1 = *dn - b1 / gap1 * b1, d__2 = *dmin__ * .5;
		    s = max(d__1,d__2);
		    *ttype = -2;
		} else {
		    s = 0.;
		    if (*dn > b1) {
			s = *dn - b1;
		    }
		    if (a2 > b1 + b2) {
			// Computing MIN
			d__1 = s, d__2 = a2 - (b1 + b2);
			s = min(d__1,d__2);
		    }
		    // Computing MAX
		    d__1 = s, d__2 = *dmin__ * .333;
		    s = max(d__1,d__2);
		    *ttype = -3;
		}
	    } else {
		//
		//             Case 4.
		//
		*ttype = -4;
		s = *dmin__ * .25;
		if (*dmin__ == *dn) {
		    gam = *dn;
		    a2 = 0.;
		    if (z__[nn - 5] > z__[nn - 7]) {
			return 0;
		    }
		    b2 = z__[nn - 5] / z__[nn - 7];
		    np = nn - 9;
		} else {
		    np = nn - (*pp << 1);
		    gam = *dn1;
		    if (z__[np - 4] > z__[np - 2]) {
			return 0;
		    }
		    a2 = z__[np - 4] / z__[np - 2];
		    if (z__[nn - 9] > z__[nn - 11]) {
			return 0;
		    }
		    b2 = z__[nn - 9] / z__[nn - 11];
		    np = nn - 13;
		}
		//
		//             Approximate contribution to norm squared from I < NN-1.
		//
		a2 += b2;
		i__1 = (*i0 << 2) - 1 + *pp;
		for (i4 = np; i4 >= i__1; i4 += -4) {
		    if (b2 == 0.) {
			goto L20;
		    }
		    b1 = b2;
		    if (z__[i4] > z__[i4 - 2]) {
			return 0;
		    }
		    b2 *= z__[i4] / z__[i4 - 2];
		    a2 += b2;
		    if (max(b2,b1) * 100. < a2 || .563 < a2) {
			goto L20;
		    }
// L10:
		}
L20:
		a2 *= 1.05;
		//
		//             Rayleigh quotient residual bound.
		//
		if (a2 < .563) {
		    s = gam * (1. - sqrt(a2)) / (a2 + 1.);
		}
	    }
	} else if (*dmin__ == *dn2) {
	    //
	    //          Case 5.
	    //
	    *ttype = -5;
	    s = *dmin__ * .25;
	    //
	    //          Compute contribution to norm squared from I > NN-2.
	    //
	    np = nn - (*pp << 1);
	    b1 = z__[np - 2];
	    b2 = z__[np - 6];
	    gam = *dn2;
	    if (z__[np - 8] > b2 || z__[np - 4] > b1) {
		return 0;
	    }
	    a2 = z__[np - 8] / b2 * (z__[np - 4] / b1 + 1.);
	    //
	    //          Approximate contribution to norm squared from I < NN-2.
	    //
	    if (*n0 - *i0 > 2) {
		b2 = z__[nn - 13] / z__[nn - 15];
		a2 += b2;
		i__1 = (*i0 << 2) - 1 + *pp;
		for (i4 = nn - 17; i4 >= i__1; i4 += -4) {
		    if (b2 == 0.) {
			goto L40;
		    }
		    b1 = b2;
		    if (z__[i4] > z__[i4 - 2]) {
			return 0;
		    }
		    b2 *= z__[i4] / z__[i4 - 2];
		    a2 += b2;
		    if (max(b2,b1) * 100. < a2 || .563 < a2) {
			goto L40;
		    }
// L30:
		}
L40:
		a2 *= 1.05;
	    }
	    if (a2 < .563) {
		s = gam * (1. - sqrt(a2)) / (a2 + 1.);
	    }
	} else {
	    //
	    //          Case 6, no information to guide us.
	    //
	    if (*ttype == -6) {
		*g += (1. - *g) * .333;
	    } else if (*ttype == -18) {
		*g = .083250000000000005;
	    } else {
		*g = .25;
	    }
	    s = *g * *dmin__;
	    *ttype = -6;
	}
    } else if (*n0in == *n0 + 1) {
	//
	//       One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
	//
	if (*dmin1 == *dn1 && *dmin2 == *dn2) {
	    //
	    //          Cases 7 and 8.
	    //
	    *ttype = -7;
	    s = *dmin1 * .333;
	    if (z__[nn - 5] > z__[nn - 7]) {
		return 0;
	    }
	    b1 = z__[nn - 5] / z__[nn - 7];
	    b2 = b1;
	    if (b2 == 0.) {
		goto L60;
	    }
	    i__1 = (*i0 << 2) - 1 + *pp;
	    for (i4 = (*n0 << 2) - 9 + *pp; i4 >= i__1; i4 += -4) {
		a2 = b1;
		if (z__[i4] > z__[i4 - 2]) {
		    return 0;
		}
		b1 *= z__[i4] / z__[i4 - 2];
		b2 += b1;
		if (max(b1,a2) * 100. < b2) {
		    goto L60;
		}
// L50:
	    }
L60:
	    b2 = sqrt(b2 * 1.05);
	    // Computing 2nd power
	    d__1 = b2;
	    a2 = *dmin1 / (d__1 * d__1 + 1.);
	    gap2 = *dmin2 * .5 - a2;
	    if (gap2 > 0. && gap2 > b2 * a2) {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - a2 * 1.01 * (b2 / gap2) * b2);
		s = max(d__1,d__2);
	    } else {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - b2 * 1.01);
		s = max(d__1,d__2);
		*ttype = -8;
	    }
	} else {
	    //
	    //          Case 9.
	    //
	    s = *dmin1 * .25;
	    if (*dmin1 == *dn1) {
		s = *dmin1 * .5;
	    }
	    *ttype = -9;
	}
    } else if (*n0in == *n0 + 2) {
	//
	//       Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
	//
	//       Cases 10 and 11.
	//
	if (*dmin2 == *dn2 && z__[nn - 5] * 2. < z__[nn - 7]) {
	    *ttype = -10;
	    s = *dmin2 * .333;
	    if (z__[nn - 5] > z__[nn - 7]) {
		return 0;
	    }
	    b1 = z__[nn - 5] / z__[nn - 7];
	    b2 = b1;
	    if (b2 == 0.) {
		goto L80;
	    }
	    i__1 = (*i0 << 2) - 1 + *pp;
	    for (i4 = (*n0 << 2) - 9 + *pp; i4 >= i__1; i4 += -4) {
		if (z__[i4] > z__[i4 - 2]) {
		    return 0;
		}
		b1 *= z__[i4] / z__[i4 - 2];
		b2 += b1;
		if (b1 * 100. < b2) {
		    goto L80;
		}
// L70:
	    }
L80:
	    b2 = sqrt(b2 * 1.05);
	    // Computing 2nd power
	    d__1 = b2;
	    a2 = *dmin2 / (d__1 * d__1 + 1.);
	    gap2 = z__[nn - 7] + z__[nn - 9] - sqrt(z__[nn - 11]) * sqrt(z__[
		    nn - 9]) - a2;
	    if (gap2 > 0. && gap2 > b2 * a2) {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - a2 * 1.01 * (b2 / gap2) * b2);
		s = max(d__1,d__2);
	    } else {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - b2 * 1.01);
		s = max(d__1,d__2);
	    }
	} else {
	    s = *dmin2 * .25;
	    *ttype = -11;
	}
    } else if (*n0in > *n0 + 2) {
	//
	//       Case 12, more than two eigenvalues deflated. No information.
	//
	s = 0.;
	*ttype = -12;
    }
    *tau = s;
    return 0;
    //
    //    End of DLASQ4
    //
} // dlasq4_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ5 computes one dqds transform in ping-pong form. Used by sbdsqr and sstegr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ5 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq5.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq5.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq5.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ5( I0, N0, Z, PP, TAU, SIGMA, DMIN, DMIN1, DMIN2, DN,
//                         DNM1, DNM2, IEEE, EPS )
//
//      .. Scalar Arguments ..
//      LOGICAL            IEEE
//      INTEGER            I0, N0, PP
//      DOUBLE PRECISION   DMIN, DMIN1, DMIN2, DN, DNM1, DNM2, TAU, SIGMA, EPS
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ5 computes one dqds transform in ping-pong form, one
//> version for IEEE machines another for non IEEE machines.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>        First index.
//> \endverbatim
//>
//> \param[in] N0
//> \verbatim
//>          N0 is INTEGER
//>        Last index.
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N )
//>        Z holds the qd array. EMIN is stored in Z(4*N0) to avoid
//>        an extra argument.
//> \endverbatim
//>
//> \param[in] PP
//> \verbatim
//>          PP is INTEGER
//>        PP=0 for ping, PP=1 for pong.
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>        This is the shift.
//> \endverbatim
//>
//> \param[in] SIGMA
//> \verbatim
//>          SIGMA is DOUBLE PRECISION
//>        This is the accumulated shift up to this step.
//> \endverbatim
//>
//> \param[out] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>        Minimum value of d.
//> \endverbatim
//>
//> \param[out] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ).
//> \endverbatim
//>
//> \param[out] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
//> \endverbatim
//>
//> \param[out] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//>        d(N0), the last value of d.
//> \endverbatim
//>
//> \param[out] DNM1
//> \verbatim
//>          DNM1 is DOUBLE PRECISION
//>        d(N0-1).
//> \endverbatim
//>
//> \param[out] DNM2
//> \verbatim
//>          DNM2 is DOUBLE PRECISION
//>        d(N0-2).
//> \endverbatim
//>
//> \param[in] IEEE
//> \verbatim
//>          IEEE is LOGICAL
//>        Flag for IEEE or non IEEE arithmetic.
//> \endverbatim
//>
//> \param[in] EPS
//> \verbatim
//>          EPS is DOUBLE PRECISION
//>        This is the value of epsilon used.
//> \endverbatim
//>
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2017
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasq5_(int *i0, int *n0, double *z__, int *pp, double *
	tau, double *sigma, double *dmin__, double *dmin1, double *dmin2,
	double *dn, double *dnm1, double *dnm2, int *ieee, double *eps)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Local variables
    double d__;
    int j4, j4p2;
    double emin, temp, dthresh;

    //
    // -- LAPACK computational routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameter ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --z__;

    // Function Body
    if (*n0 - *i0 - 1 <= 0) {
	return 0;
    }
    dthresh = *eps * (*sigma + *tau);
    if (*tau < dthresh * .5) {
	*tau = 0.;
    }
    if (*tau != 0.) {
	j4 = (*i0 << 2) + *pp - 3;
	emin = z__[j4 + 4];
	d__ = z__[j4] - *tau;
	*dmin__ = d__;
	*dmin1 = -z__[j4];
	if (*ieee) {
	    //
	    //       Code for IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    temp = z__[j4 + 1] / z__[j4 - 2];
		    d__ = d__ * temp - *tau;
		    *dmin__ = min(*dmin__,d__);
		    z__[j4] = z__[j4 - 1] * temp;
		    // Computing MIN
		    d__1 = z__[j4];
		    emin = min(d__1,emin);
// L10:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    temp = z__[j4 + 2] / z__[j4 - 3];
		    d__ = d__ * temp - *tau;
		    *dmin__ = min(*dmin__,d__);
		    z__[j4 - 1] = z__[j4] * temp;
		    // Computing MIN
		    d__1 = z__[j4 - 1];
		    emin = min(d__1,emin);
// L20:
		}
	    }
	    //
	    //       Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dn);
	} else {
	    //
	    //       Code for non IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
			d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4];
		    emin = min(d__1,d__2);
// L30:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
			d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4 - 1];
		    emin = min(d__1,d__2);
// L40:
		}
	    }
	    //
	    //       Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    if (*dnm2 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    if (*dnm1 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dn);
	}
    } else {
	//    This is the version that sets d's to zero if they are small enough
	j4 = (*i0 << 2) + *pp - 3;
	emin = z__[j4 + 4];
	d__ = z__[j4] - *tau;
	*dmin__ = d__;
	*dmin1 = -z__[j4];
	if (*ieee) {
	    //
	    //    Code for IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    temp = z__[j4 + 1] / z__[j4 - 2];
		    d__ = d__ * temp - *tau;
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    z__[j4] = z__[j4 - 1] * temp;
		    // Computing MIN
		    d__1 = z__[j4];
		    emin = min(d__1,emin);
// L50:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    temp = z__[j4 + 2] / z__[j4 - 3];
		    d__ = d__ * temp - *tau;
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    z__[j4 - 1] = z__[j4] * temp;
		    // Computing MIN
		    d__1 = z__[j4 - 1];
		    emin = min(d__1,emin);
// L60:
		}
	    }
	    //
	    //    Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dn);
	} else {
	    //
	    //    Code for non IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
			d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		    }
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4];
		    emin = min(d__1,d__2);
// L70:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
			d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		    }
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4 - 1];
		    emin = min(d__1,d__2);
// L80:
		}
	    }
	    //
	    //    Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    if (*dnm2 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    if (*dnm1 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dn);
	}
    }
    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return 0;
    //
    //    End of DLASQ5
    //
} // dlasq5_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ6 computes one dqd transform in ping-pong form. Used by sbdsqr and sstegr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ6 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq6.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq6.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq6.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ6( I0, N0, Z, PP, DMIN, DMIN1, DMIN2, DN,
//                         DNM1, DNM2 )
//
//      .. Scalar Arguments ..
//      INTEGER            I0, N0, PP
//      DOUBLE PRECISION   DMIN, DMIN1, DMIN2, DN, DNM1, DNM2
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ6 computes one dqd (shift equal to zero) transform in
//> ping-pong form, with protection against underflow and overflow.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>        First index.
//> \endverbatim
//>
//> \param[in] N0
//> \verbatim
//>          N0 is INTEGER
//>        Last index.
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N )
//>        Z holds the qd array. EMIN is stored in Z(4*N0) to avoid
//>        an extra argument.
//> \endverbatim
//>
//> \param[in] PP
//> \verbatim
//>          PP is INTEGER
//>        PP=0 for ping, PP=1 for pong.
//> \endverbatim
//>
//> \param[out] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>        Minimum value of d.
//> \endverbatim
//>
//> \param[out] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ).
//> \endverbatim
//>
//> \param[out] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
//> \endverbatim
//>
//> \param[out] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//>        d(N0), the last value of d.
//> \endverbatim
//>
//> \param[out] DNM1
//> \verbatim
//>          DNM1 is DOUBLE PRECISION
//>        d(N0-1).
//> \endverbatim
//>
//> \param[out] DNM2
//> \verbatim
//>          DNM2 is DOUBLE PRECISION
//>        d(N0-2).
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasq6_(int *i0, int *n0, double *z__, int *pp, double *
	dmin__, double *dmin1, double *dmin2, double *dn, double *dnm1,
	double *dnm2)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Local variables
    double d__;
    int j4, j4p2;
    double emin, temp;
    extern double dlamch_(char *);
    double safmin;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameter ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Function ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --z__;

    // Function Body
    if (*n0 - *i0 - 1 <= 0) {
	return 0;
    }
    safmin = dlamch_("Safe minimum");
    j4 = (*i0 << 2) + *pp - 3;
    emin = z__[j4 + 4];
    d__ = z__[j4];
    *dmin__ = d__;
    if (*pp == 0) {
	i__1 = *n0 - 3 << 2;
	for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
	    z__[j4 - 2] = d__ + z__[j4 - 1];
	    if (z__[j4 - 2] == 0.) {
		z__[j4] = 0.;
		d__ = z__[j4 + 1];
		*dmin__ = d__;
		emin = 0.;
	    } else if (safmin * z__[j4 + 1] < z__[j4 - 2] && safmin * z__[j4
		    - 2] < z__[j4 + 1]) {
		temp = z__[j4 + 1] / z__[j4 - 2];
		z__[j4] = z__[j4 - 1] * temp;
		d__ *= temp;
	    } else {
		z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
		d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]);
	    }
	    *dmin__ = min(*dmin__,d__);
	    // Computing MIN
	    d__1 = emin, d__2 = z__[j4];
	    emin = min(d__1,d__2);
// L10:
	}
    } else {
	i__1 = *n0 - 3 << 2;
	for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
	    z__[j4 - 3] = d__ + z__[j4];
	    if (z__[j4 - 3] == 0.) {
		z__[j4 - 1] = 0.;
		d__ = z__[j4 + 2];
		*dmin__ = d__;
		emin = 0.;
	    } else if (safmin * z__[j4 + 2] < z__[j4 - 3] && safmin * z__[j4
		    - 3] < z__[j4 + 2]) {
		temp = z__[j4 + 2] / z__[j4 - 3];
		z__[j4 - 1] = z__[j4] * temp;
		d__ *= temp;
	    } else {
		z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
		d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]);
	    }
	    *dmin__ = min(*dmin__,d__);
	    // Computing MIN
	    d__1 = emin, d__2 = z__[j4 - 1];
	    emin = min(d__1,d__2);
// L20:
	}
    }
    //
    //    Unroll last two steps.
    //
    *dnm2 = d__;
    *dmin2 = *dmin__;
    j4 = (*n0 - 2 << 2) - *pp;
    j4p2 = j4 + (*pp << 1) - 1;
    z__[j4 - 2] = *dnm2 + z__[j4p2];
    if (z__[j4 - 2] == 0.) {
	z__[j4] = 0.;
	*dnm1 = z__[j4p2 + 2];
	*dmin__ = *dnm1;
	emin = 0.;
    } else if (safmin * z__[j4p2 + 2] < z__[j4 - 2] && safmin * z__[j4 - 2] <
	    z__[j4p2 + 2]) {
	temp = z__[j4p2 + 2] / z__[j4 - 2];
	z__[j4] = z__[j4p2] * temp;
	*dnm1 = *dnm2 * temp;
    } else {
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]);
    }
    *dmin__ = min(*dmin__,*dnm1);
    *dmin1 = *dmin__;
    j4 += 4;
    j4p2 = j4 + (*pp << 1) - 1;
    z__[j4 - 2] = *dnm1 + z__[j4p2];
    if (z__[j4 - 2] == 0.) {
	z__[j4] = 0.;
	*dn = z__[j4p2 + 2];
	*dmin__ = *dn;
	emin = 0.;
    } else if (safmin * z__[j4p2 + 2] < z__[j4 - 2] && safmin * z__[j4 - 2] <
	    z__[j4p2 + 2]) {
	temp = z__[j4p2 + 2] / z__[j4 - 2];
	z__[j4] = z__[j4p2] * temp;
	*dn = *dnm1 * temp;
    } else {
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]);
    }
    *dmin__ = min(*dmin__,*dn);
    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return 0;
    //
    //    End of DLASQ6
    //
} // dlasq6_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASR applies a sequence of plane rotations to a general rectangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASR( SIDE, PIVOT, DIRECT, M, N, C, S, A, LDA )
//
//      .. Scalar Arguments ..
//      CHARACTER          DIRECT, PIVOT, SIDE
//      INTEGER            LDA, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), C( * ), S( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASR applies a sequence of plane rotations to a real matrix A,
//> from either the left or the right.
//>
//> When SIDE = 'L', the transformation takes the form
//>
//>    A := P*A
//>
//> and when SIDE = 'R', the transformation takes the form
//>
//>    A := A*P**T
//>
//> where P is an orthogonal matrix consisting of a sequence of z plane
//> rotations, with z = M when SIDE = 'L' and z = N when SIDE = 'R',
//> and P**T is the transpose of P.
//>
//> When DIRECT = 'F' (Forward sequence), then
//>
//>    P = P(z-1) * ... * P(2) * P(1)
//>
//> and when DIRECT = 'B' (Backward sequence), then
//>
//>    P = P(1) * P(2) * ... * P(z-1)
//>
//> where P(k) is a plane rotation matrix defined by the 2-by-2 rotation
//>
//>    R(k) = (  c(k)  s(k) )
//>         = ( -s(k)  c(k) ).
//>
//> When PIVOT = 'V' (Variable pivot), the rotation is performed
//> for the plane (k,k+1), i.e., P(k) has the form
//>
//>    P(k) = (  1                                            )
//>           (       ...                                     )
//>           (              1                                )
//>           (                   c(k)  s(k)                  )
//>           (                  -s(k)  c(k)                  )
//>           (                                1              )
//>           (                                     ...       )
//>           (                                            1  )
//>
//> where R(k) appears as a rank-2 modification to the identity matrix in
//> rows and columns k and k+1.
//>
//> When PIVOT = 'T' (Top pivot), the rotation is performed for the
//> plane (1,k+1), so P(k) has the form
//>
//>    P(k) = (  c(k)                    s(k)                 )
//>           (         1                                     )
//>           (              ...                              )
//>           (                     1                         )
//>           ( -s(k)                    c(k)                 )
//>           (                                 1             )
//>           (                                      ...      )
//>           (                                             1 )
//>
//> where R(k) appears in rows and columns 1 and k+1.
//>
//> Similarly, when PIVOT = 'B' (Bottom pivot), the rotation is
//> performed for the plane (k,z), giving P(k) the form
//>
//>    P(k) = ( 1                                             )
//>           (      ...                                      )
//>           (             1                                 )
//>           (                  c(k)                    s(k) )
//>           (                         1                     )
//>           (                              ...              )
//>           (                                     1         )
//>           (                 -s(k)                    c(k) )
//>
//> where R(k) appears in rows and columns k and z.  The rotations are
//> performed without ever forming P(k) explicitly.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          Specifies whether the plane rotation matrix P is applied to
//>          A on the left or the right.
//>          = 'L':  Left, compute A := P*A
//>          = 'R':  Right, compute A:= A*P**T
//> \endverbatim
//>
//> \param[in] PIVOT
//> \verbatim
//>          PIVOT is CHARACTER*1
//>          Specifies the plane for which P(k) is a plane rotation
//>          matrix.
//>          = 'V':  Variable pivot, the plane (k,k+1)
//>          = 'T':  Top pivot, the plane (1,k+1)
//>          = 'B':  Bottom pivot, the plane (k,z)
//> \endverbatim
//>
//> \param[in] DIRECT
//> \verbatim
//>          DIRECT is CHARACTER*1
//>          Specifies whether P is a forward or backward sequence of
//>          plane rotations.
//>          = 'F':  Forward, P = P(z-1)*...*P(2)*P(1)
//>          = 'B':  Backward, P = P(1)*P(2)*...*P(z-1)
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.  If m <= 1, an immediate
//>          return is effected.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.  If n <= 1, an
//>          immediate return is effected.
//> \endverbatim
//>
//> \param[in] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension
//>                  (M-1) if SIDE = 'L'
//>                  (N-1) if SIDE = 'R'
//>          The cosines c(k) of the plane rotations.
//> \endverbatim
//>
//> \param[in] S
//> \verbatim
//>          S is DOUBLE PRECISION array, dimension
//>                  (M-1) if SIDE = 'L'
//>                  (N-1) if SIDE = 'R'
//>          The sines s(k) of the plane rotations.  The 2-by-2 plane
//>          rotation part of the matrix P(k), R(k), has the form
//>          R(k) = (  c(k)  s(k) )
//>                 ( -s(k)  c(k) ).
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          The M-by-N matrix A.  On exit, A is overwritten by P*A if
//>          SIDE = 'L' or by A*P**T if SIDE = 'R'.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
// =====================================================================
/* Subroutine */ int dlasr_(char *side, char *pivot, char *direct, int *m,
	int *n, double *c__, double *s, double *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, info;
    double temp;
    extern int lsame_(char *, char *);
    double ctemp, stemp;
    extern /* Subroutine */ int xerbla_(char *, int *);

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters
    //
    // Parameter adjustments
    --c__;
    --s;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    info = 0;
    if (! (lsame_(side, "L") || lsame_(side, "R"))) {
	info = 1;
    } else if (! (lsame_(pivot, "V") || lsame_(pivot, "T") || lsame_(pivot,
	    "B"))) {
	info = 2;
    } else if (! (lsame_(direct, "F") || lsame_(direct, "B"))) {
	info = 3;
    } else if (*m < 0) {
	info = 4;
    } else if (*n < 0) {
	info = 5;
    } else if (*lda < max(1,*m)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("DLASR ", &info);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0) {
	return 0;
    }
    if (lsame_(side, "L")) {
	//
	//       Form  P * A
	//
	if (lsame_(pivot, "V")) {
	    if (lsame_(direct, "F")) {
		i__1 = *m - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + 1 + i__ * a_dim1];
			    a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp *
				    a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j
				    + i__ * a_dim1];
// L10:
			}
		    }
// L20:
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + 1 + i__ * a_dim1];
			    a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp *
				    a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j
				    + i__ * a_dim1];
// L30:
			}
		    }
// L40:
		}
	    }
	} else if (lsame_(pivot, "T")) {
	    if (lsame_(direct, "F")) {
		i__1 = *m;
		for (j = 2; j <= i__1; ++j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1. || stemp != 0.) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = ctemp * temp - stemp * a[
				    i__ * a_dim1 + 1];
			    a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[
				    i__ * a_dim1 + 1];
// L50:
			}
		    }
// L60:
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1. || stemp != 0.) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = ctemp * temp - stemp * a[
				    i__ * a_dim1 + 1];
			    a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[
				    i__ * a_dim1 + 1];
// L70:
			}
		    }
// L80:
		}
	    }
	} else if (lsame_(pivot, "B")) {
	    if (lsame_(direct, "F")) {
		i__1 = *m - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1]
				     + ctemp * temp;
			    a[*m + i__ * a_dim1] = ctemp * a[*m + i__ *
				    a_dim1] - stemp * temp;
// L90:
			}
		    }
// L100:
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1]
				     + ctemp * temp;
			    a[*m + i__ * a_dim1] = ctemp * a[*m + i__ *
				    a_dim1] - stemp * temp;
// L110:
			}
		    }
// L120:
		}
	    }
	}
    } else if (lsame_(side, "R")) {
	//
	//       Form A * P**T
	//
	if (lsame_(pivot, "V")) {
	    if (lsame_(direct, "F")) {
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + (j + 1) * a_dim1];
			    a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp *
				     a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * temp + ctemp * a[
				    i__ + j * a_dim1];
// L130:
			}
		    }
// L140:
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + (j + 1) * a_dim1];
			    a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp *
				     a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * temp + ctemp * a[
				    i__ + j * a_dim1];
// L150:
			}
		    }
// L160:
		}
	    }
	} else if (lsame_(pivot, "T")) {
	    if (lsame_(direct, "F")) {
		i__1 = *n;
		for (j = 2; j <= i__1; ++j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1. || stemp != 0.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = ctemp * temp - stemp * a[
				    i__ + a_dim1];
			    a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ +
				    a_dim1];
// L170:
			}
		    }
// L180:
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1. || stemp != 0.) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = ctemp * temp - stemp * a[
				    i__ + a_dim1];
			    a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ +
				    a_dim1];
// L190:
			}
		    }
// L200:
		}
	    }
	} else if (lsame_(pivot, "B")) {
	    if (lsame_(direct, "F")) {
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1]
				     + ctemp * temp;
			    a[i__ + *n * a_dim1] = ctemp * a[i__ + *n *
				    a_dim1] - stemp * temp;
// L210:
			}
		    }
// L220:
		}
	    } else if (lsame_(direct, "B")) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1. || stemp != 0.) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1]
				     + ctemp * temp;
			    a[i__ + *n * a_dim1] = ctemp * a[i__ + *n *
				    a_dim1] - stemp * temp;
// L230:
			}
		    }
// L240:
		}
	    }
	}
    }
    return 0;
    //
    //    End of DLASR
    //
} // dlasr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASRT sorts numbers in increasing or decreasing order.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASRT + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasrt.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasrt.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasrt.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASRT( ID, N, D, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          ID
//      INTEGER            INFO, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Sort the numbers in D in increasing order (if ID = 'I') or
//> in decreasing order (if ID = 'D' ).
//>
//> Use Quick Sort, reverting to Insertion sort on arrays of
//> size <= 20. Dimension of STACK limits N to about 2**32.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ID
//> \verbatim
//>          ID is CHARACTER*1
//>          = 'I': sort D in increasing order;
//>          = 'D': sort D in decreasing order.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The length of the array D.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the array to be sorted.
//>          On exit, D has been sorted into increasing order
//>          (D(1) <= ... <= D(N) ) or into decreasing order
//>          (D(1) >= ... >= D(N) ), depending on ID.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date June 2016
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasrt_(char *id, int *n, double *d__, int *info)
{
    // System generated locals
    int i__1, i__2;

    // Local variables
    int i__, j;
    double d1, d2, d3;
    int dir;
    double tmp;
    int endd;
    extern int lsame_(char *, char *);
    int stack[64]	/* was [2][32] */;
    double dmnmx;
    int start;
    extern /* Subroutine */ int xerbla_(char *, int *);
    int stkpnt;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;

    // Function Body
    *info = 0;
    dir = -1;
    if (lsame_(id, "D")) {
	dir = 0;
    } else if (lsame_(id, "I")) {
	dir = 1;
    }
    if (dir == -1) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASRT", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n <= 1) {
	return 0;
    }
    stkpnt = 1;
    stack[0] = 1;
    stack[1] = *n;
L10:
    start = stack[(stkpnt << 1) - 2];
    endd = stack[(stkpnt << 1) - 1];
    --stkpnt;
    if (endd - start <= 20 && endd - start > 0) {
	//
	//       Do Insertion sort on D( START:ENDD )
	//
	if (dir == 0) {
	    //
	    //          Sort into decreasing order
	    //
	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] > d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L30;
		    }
// L20:
		}
L30:
		;
	    }
	} else {
	    //
	    //          Sort into increasing order
	    //
	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] < d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L50;
		    }
// L40:
		}
L50:
		;
	    }
	}
    } else if (endd - start > 20) {
	//
	//       Partition D( START:ENDD ) and stack parts, largest one first
	//
	//       Choose partition entry as median of 3
	//
	d1 = d__[start];
	d2 = d__[endd];
	i__ = (start + endd) / 2;
	d3 = d__[i__];
	if (d1 < d2) {
	    if (d3 < d1) {
		dmnmx = d1;
	    } else if (d3 < d2) {
		dmnmx = d3;
	    } else {
		dmnmx = d2;
	    }
	} else {
	    if (d3 < d2) {
		dmnmx = d2;
	    } else if (d3 < d1) {
		dmnmx = d3;
	    } else {
		dmnmx = d1;
	    }
	}
	if (dir == 0) {
	    //
	    //          Sort into decreasing order
	    //
	    i__ = start - 1;
	    j = endd + 1;
L60:
L70:
	    --j;
	    if (d__[j] < dmnmx) {
		goto L70;
	    }
L80:
	    ++i__;
	    if (d__[i__] > dmnmx) {
		goto L80;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L60;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
	    } else {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
	    }
	} else {
	    //
	    //          Sort into increasing order
	    //
	    i__ = start - 1;
	    j = endd + 1;
L90:
L100:
	    --j;
	    if (d__[j] > dmnmx) {
		goto L100;
	    }
L110:
	    ++i__;
	    if (d__[i__] < dmnmx) {
		goto L110;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L90;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
	    } else {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
	    }
	}
    }
    if (stkpnt > 0) {
	goto L10;
    }
    return 0;
    //
    //    End of DLASRT
    //
} // dlasrt_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASV2 computes the singular value decomposition of a 2-by-2 triangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASV2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasv2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasv2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasv2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASV2( F, G, H, SSMIN, SSMAX, SNR, CSR, SNL, CSL )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   CSL, CSR, F, G, H, SNL, SNR, SSMAX, SSMIN
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASV2 computes the singular value decomposition of a 2-by-2
//> triangular matrix
//>    [  F   G  ]
//>    [  0   H  ].
//> On return, abs(SSMAX) is the larger singular value, abs(SSMIN) is the
//> smaller singular value, and (CSL,SNL) and (CSR,SNR) are the left and
//> right singular vectors for abs(SSMAX), giving the decomposition
//>
//>    [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
//>    [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ].
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] F
//> \verbatim
//>          F is DOUBLE PRECISION
//>          The (1,1) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] G
//> \verbatim
//>          G is DOUBLE PRECISION
//>          The (1,2) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] H
//> \verbatim
//>          H is DOUBLE PRECISION
//>          The (2,2) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[out] SSMIN
//> \verbatim
//>          SSMIN is DOUBLE PRECISION
//>          abs(SSMIN) is the smaller singular value.
//> \endverbatim
//>
//> \param[out] SSMAX
//> \verbatim
//>          SSMAX is DOUBLE PRECISION
//>          abs(SSMAX) is the larger singular value.
//> \endverbatim
//>
//> \param[out] SNL
//> \verbatim
//>          SNL is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] CSL
//> \verbatim
//>          CSL is DOUBLE PRECISION
//>          The vector (CSL, SNL) is a unit left singular vector for the
//>          singular value abs(SSMAX).
//> \endverbatim
//>
//> \param[out] SNR
//> \verbatim
//>          SNR is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] CSR
//> \verbatim
//>          CSR is DOUBLE PRECISION
//>          The vector (CSR, SNR) is a unit right singular vector for the
//>          singular value abs(SSMAX).
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup OTHERauxiliary
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Any input parameter may be aliased with any output parameter.
//>
//>  Barring over/underflow and assuming a guard digit in subtraction, all
//>  output quantities are correct to within a few units in the last
//>  place (ulps).
//>
//>  In IEEE arithmetic, the code works correctly if one matrix element is
//>  infinite.
//>
//>  Overflow will not occur unless the largest singular value itself
//>  overflows or is within a few ulps of overflow. (On machines with
//>  partial overflow, like the Cray, overflow may occur if the largest
//>  singular value is within a factor of 2 of overflow.)
//>
//>  Underflow is harmless if underflow is gradual. Otherwise, results
//>  may correspond to a matrix modified by perturbations of size near
//>  the underflow threshold.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlasv2_(double *f, double *g, double *h__, double *ssmin,
	 double *ssmax, double *snr, double *csr, double *snl, double *csl)
{
    // Table of constant values
    double c_b3 = 2.;
    double c_b4 = 1.;

    // System generated locals
    double d__1;

    // Local variables
    double a, d__, l, m, r__, s, t, fa, ga, ha, ft, gt, ht, mm, tt, clt, crt,
	    slt, srt;
    int pmax;
    double temp;
    int swap;
    double tsign;
    extern double dlamch_(char *);
    int gasmal;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    ft = *f;
    fa = abs(ft);
    ht = *h__;
    ha = abs(*h__);
    //
    //    PMAX points to the maximum absolute element of matrix
    //      PMAX = 1 if F largest in absolute values
    //      PMAX = 2 if G largest in absolute values
    //      PMAX = 3 if H largest in absolute values
    //
    pmax = 1;
    swap = ha > fa;
    if (swap) {
	pmax = 3;
	temp = ft;
	ft = ht;
	ht = temp;
	temp = fa;
	fa = ha;
	ha = temp;
	//
	//       Now FA .ge. HA
	//
    }
    gt = *g;
    ga = abs(gt);
    if (ga == 0.) {
	//
	//       Diagonal matrix
	//
	*ssmin = ha;
	*ssmax = fa;
	clt = 1.;
	crt = 1.;
	slt = 0.;
	srt = 0.;
    } else {
	gasmal = TRUE_;
	if (ga > fa) {
	    pmax = 2;
	    if (fa / ga < dlamch_("EPS")) {
		//
		//             Case of very large GA
		//
		gasmal = FALSE_;
		*ssmax = ga;
		if (ha > 1.) {
		    *ssmin = fa / (ga / ha);
		} else {
		    *ssmin = fa / ga * ha;
		}
		clt = 1.;
		slt = ht / gt;
		srt = 1.;
		crt = ft / gt;
	    }
	}
	if (gasmal) {
	    //
	    //          Normal case
	    //
	    d__ = fa - ha;
	    if (d__ == fa) {
		//
		//             Copes with infinite F or H
		//
		l = 1.;
	    } else {
		l = d__ / fa;
	    }
	    //
	    //          Note that 0 .le. L .le. 1
	    //
	    m = gt / ft;
	    //
	    //          Note that abs(M) .le. 1/macheps
	    //
	    t = 2. - l;
	    //
	    //          Note that T .ge. 1
	    //
	    mm = m * m;
	    tt = t * t;
	    s = sqrt(tt + mm);
	    //
	    //          Note that 1 .le. S .le. 1 + 1/macheps
	    //
	    if (l == 0.) {
		r__ = abs(m);
	    } else {
		r__ = sqrt(l * l + mm);
	    }
	    //
	    //          Note that 0 .le. R .le. 1 + 1/macheps
	    //
	    a = (s + r__) * .5;
	    //
	    //          Note that 1 .le. A .le. 1 + abs(M)
	    //
	    *ssmin = ha / a;
	    *ssmax = fa * a;
	    if (mm == 0.) {
		//
		//             Note that M is very tiny
		//
		if (l == 0.) {
		    t = d_sign(&c_b3, &ft) * d_sign(&c_b4, &gt);
		} else {
		    t = gt / d_sign(&d__, &ft) + m / t;
		}
	    } else {
		t = (m / (s + t) + m / (r__ + l)) * (a + 1.);
	    }
	    l = sqrt(t * t + 4.);
	    crt = 2. / l;
	    srt = t / l;
	    clt = (crt + srt * m) / a;
	    slt = ht / ft * srt / a;
	}
    }
    if (swap) {
	*csl = srt;
	*snl = crt;
	*csr = slt;
	*snr = clt;
    } else {
	*csl = clt;
	*snl = slt;
	*csr = crt;
	*snr = srt;
    }
    //
    //    Correct signs of SSMAX and SSMIN
    //
    if (pmax == 1) {
	tsign = d_sign(&c_b4, csr) * d_sign(&c_b4, csl) * d_sign(&c_b4, f);
    }
    if (pmax == 2) {
	tsign = d_sign(&c_b4, snr) * d_sign(&c_b4, csl) * d_sign(&c_b4, g);
    }
    if (pmax == 3) {
	tsign = d_sign(&c_b4, snr) * d_sign(&c_b4, snl) * d_sign(&c_b4, h__);
    }
    *ssmax = d_sign(ssmax, &tsign);
    d__1 = tsign * d_sign(&c_b4, f) * d_sign(&c_b4, h__);
    *ssmin = d_sign(ssmin, &d__1);
    return 0;
    //
    //    End of DLASV2
    //
} // dlasv2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORGBR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORGBR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorgbr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorgbr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorgbr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORGBR( VECT, M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          VECT
//      INTEGER            INFO, K, LDA, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DORGBR generates one of the real orthogonal matrices Q or P**T
//> determined by DGEBRD when reducing a real matrix A to bidiagonal
//> form: A = Q * B * P**T.  Q and P**T are defined as products of
//> elementary reflectors H(i) or G(i) respectively.
//>
//> If VECT = 'Q', A is assumed to have been an M-by-K matrix, and Q
//> is of order M:
//> if m >= k, Q = H(1) H(2) . . . H(k) and DORGBR returns the first n
//> columns of Q, where m >= n >= k;
//> if m < k, Q = H(1) H(2) . . . H(m-1) and DORGBR returns Q as an
//> M-by-M matrix.
//>
//> If VECT = 'P', A is assumed to have been a K-by-N matrix, and P**T
//> is of order N:
//> if k < n, P**T = G(k) . . . G(2) G(1) and DORGBR returns the first m
//> rows of P**T, where n >= m >= k;
//> if k >= n, P**T = G(n-1) . . . G(2) G(1) and DORGBR returns P**T as
//> an N-by-N matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] VECT
//> \verbatim
//>          VECT is CHARACTER*1
//>          Specifies whether the matrix Q or the matrix P**T is
//>          required, as defined in the transformation applied by DGEBRD:
//>          = 'Q':  generate Q;
//>          = 'P':  generate P**T.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix Q or P**T to be returned.
//>          M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix Q or P**T to be returned.
//>          N >= 0.
//>          If VECT = 'Q', M >= N >= min(M,K);
//>          if VECT = 'P', N >= M >= min(N,K).
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          If VECT = 'Q', the number of columns in the original M-by-K
//>          matrix reduced by DGEBRD.
//>          If VECT = 'P', the number of rows in the original K-by-N
//>          matrix reduced by DGEBRD.
//>          K >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the vectors which define the elementary reflectors,
//>          as returned by DGEBRD.
//>          On exit, the M-by-N matrix Q or P**T.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A. LDA >= max(1,M).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension
//>                                (min(M,K)) if VECT = 'Q'
//>                                (min(N,K)) if VECT = 'P'
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i) or G(i), which determines Q or P**T, as
//>          returned by DGEBRD in its array argument TAUQ or TAUP.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK. LWORK >= max(1,min(M,N)).
//>          For optimum performance LWORK >= min(M,N)*NB, where NB
//>          is the optimal blocksize.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date April 2012
//
//> \ingroup doubleGBcomputational
//
// =====================================================================
/* Subroutine */ int dorgbr_(char *vect, int *m, int *n, int *k, double *a,
	int *lda, double *tau, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c_n1 = -1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, j, mn;
    extern int lsame_(char *, char *);
    int iinfo;
    int wantq;
    extern /* Subroutine */ int xerbla_(char *, int *), dorglq_(int *, int *,
	    int *, double *, int *, double *, double *, int *, int *),
	    dorgqr_(int *, int *, int *, double *, int *, double *, double *,
	    int *, int *);
    int lwkopt;
    int lquery;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    April 2012
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    wantq = lsame_(vect, "Q");
    mn = min(*m,*n);
    lquery = *lwork == -1;
    if (! wantq && ! lsame_(vect, "P")) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0 || wantq && (*n > *m || *n < min(*m,*k)) || ! wantq && (
	    *m > *n || *m < min(*n,*k))) {
	*info = -3;
    } else if (*k < 0) {
	*info = -4;
    } else if (*lda < max(1,*m)) {
	*info = -6;
    } else if (*lwork < max(1,mn) && ! lquery) {
	*info = -9;
    }
    if (*info == 0) {
	work[1] = 1.;
	if (wantq) {
	    if (*m >= *k) {
		dorgqr_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], &c_n1,
			&iinfo);
	    } else {
		if (*m > 1) {
		    i__1 = *m - 1;
		    i__2 = *m - 1;
		    i__3 = *m - 1;
		    dorgqr_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &
			    tau[1], &work[1], &c_n1, &iinfo);
		}
	    }
	} else {
	    if (*k < *n) {
		dorglq_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], &c_n1,
			&iinfo);
	    } else {
		if (*n > 1) {
		    i__1 = *n - 1;
		    i__2 = *n - 1;
		    i__3 = *n - 1;
		    dorglq_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &
			    tau[1], &work[1], &c_n1, &iinfo);
		}
	    }
	}
	lwkopt = (int) work[1];
	lwkopt = max(lwkopt,mn);
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORGBR", &i__1);
	return 0;
    } else if (lquery) {
	work[1] = (double) lwkopt;
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0) {
	work[1] = 1.;
	return 0;
    }
    if (wantq) {
	//
	//       Form Q, determined by a call to DGEBRD to reduce an m-by-k
	//       matrix
	//
	if (*m >= *k) {
	    //
	    //          If m >= k, assume m >= n >= k
	    //
	    dorgqr_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);
	} else {
	    //
	    //          If m < k, assume m = n
	    //
	    //          Shift the vectors which define the elementary reflectors one
	    //          column to the right, and set the first row and column of Q
	    //          to those of the unit matrix
	    //
	    for (j = *m; j >= 2; --j) {
		a[j * a_dim1 + 1] = 0.;
		i__1 = *m;
		for (i__ = j + 1; i__ <= i__1; ++i__) {
		    a[i__ + j * a_dim1] = a[i__ + (j - 1) * a_dim1];
// L10:
		}
// L20:
	    }
	    a[a_dim1 + 1] = 1.;
	    i__1 = *m;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		a[i__ + a_dim1] = 0.;
// L30:
	    }
	    if (*m > 1) {
		//
		//             Form Q(2:m,2:m)
		//
		i__1 = *m - 1;
		i__2 = *m - 1;
		i__3 = *m - 1;
		dorgqr_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    } else {
	//
	//       Form P**T, determined by a call to DGEBRD to reduce a k-by-n
	//       matrix
	//
	if (*k < *n) {
	    //
	    //          If k < n, assume k <= m <= n
	    //
	    dorglq_(m, n, k, &a[a_offset], lda, &tau[1], &work[1], lwork, &
		    iinfo);
	} else {
	    //
	    //          If k >= n, assume m = n
	    //
	    //          Shift the vectors which define the elementary reflectors one
	    //          row downward, and set the first row and column of P**T to
	    //          those of the unit matrix
	    //
	    a[a_dim1 + 1] = 1.;
	    i__1 = *n;
	    for (i__ = 2; i__ <= i__1; ++i__) {
		a[i__ + a_dim1] = 0.;
// L40:
	    }
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		for (i__ = j - 1; i__ >= 2; --i__) {
		    a[i__ + j * a_dim1] = a[i__ - 1 + j * a_dim1];
// L50:
		}
		a[j * a_dim1 + 1] = 0.;
// L60:
	    }
	    if (*n > 1) {
		//
		//             Form P**T(2:n,2:n)
		//
		i__1 = *n - 1;
		i__2 = *n - 1;
		i__3 = *n - 1;
		dorglq_(&i__1, &i__2, &i__3, &a[(a_dim1 << 1) + 2], lda, &tau[
			1], &work[1], lwork, &iinfo);
	    }
	}
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORGBR
    //
} // dorgbr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORGL2
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORGL2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorgl2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorgl2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorgl2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORGL2( M, N, K, A, LDA, TAU, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, K, LDA, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DORGL2 generates an m by n real matrix Q with orthonormal rows,
//> which is defined as the first m rows of a product of k elementary
//> reflectors of order n
//>
//>       Q  =  H(k) . . . H(2) H(1)
//>
//> as returned by DGELQF.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix Q. M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix Q. N >= M.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines the
//>          matrix Q. M >= K >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the i-th row must contain the vector which defines
//>          the elementary reflector H(i), for i = 1,2,...,k, as returned
//>          by DGELQF in the first k rows of its array argument A.
//>          On exit, the m-by-n matrix Q.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The first dimension of the array A. LDA >= max(1,M).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGELQF.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (M)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit
//>          < 0: if INFO = -i, the i-th argument has an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup doubleOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dorgl2_(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *info)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;
    double d__1;

    // Local variables
    int i__, j, l;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *),
	    dlarf_(char *, int *, int *, double *, int *, double *, double *,
	    int *, double *), xerbla_(char *, int *);

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*k < 0 || *k > *m) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORGL2", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m <= 0) {
	return 0;
    }
    if (*k < *m) {
	//
	//       Initialise rows k+1:m to rows of the unit matrix
	//
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (l = *k + 1; l <= i__2; ++l) {
		a[l + j * a_dim1] = 0.;
// L10:
	    }
	    if (j > *k && j <= *m) {
		a[j + j * a_dim1] = 1.;
	    }
// L20:
	}
    }
    for (i__ = *k; i__ >= 1; --i__) {
	//
	//       Apply H(i) to A(i:m,i:n) from the right
	//
	if (i__ < *n) {
	    if (i__ < *m) {
		a[i__ + i__ * a_dim1] = 1.;
		i__1 = *m - i__;
		i__2 = *n - i__ + 1;
		dlarf_("Right", &i__1, &i__2, &a[i__ + i__ * a_dim1], lda, &
			tau[i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    }
	    i__1 = *n - i__;
	    d__1 = -tau[i__];
	    dscal_(&i__1, &d__1, &a[i__ + (i__ + 1) * a_dim1], lda);
	}
	a[i__ + i__ * a_dim1] = 1. - tau[i__];
	//
	//       Set A(i,1:i-1) to zero
	//
	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    a[i__ + l * a_dim1] = 0.;
// L30:
	}
// L40:
    }
    return 0;
    //
    //    End of DORGL2
    //
} // dorgl2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORGLQ
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORGLQ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorglq.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorglq.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorglq.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORGLQ( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, K, LDA, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DORGLQ generates an M-by-N real matrix Q with orthonormal rows,
//> which is defined as the first M rows of a product of K elementary
//> reflectors of order N
//>
//>       Q  =  H(k) . . . H(2) H(1)
//>
//> as returned by DGELQF.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix Q. M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix Q. N >= M.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines the
//>          matrix Q. M >= K >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the i-th row must contain the vector which defines
//>          the elementary reflector H(i), for i = 1,2,...,k, as returned
//>          by DGELQF in the first k rows of its array argument A.
//>          On exit, the M-by-N matrix Q.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The first dimension of the array A. LDA >= max(1,M).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGELQF.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK. LWORK >= max(1,M).
//>          For optimum performance LWORK >= M*NB, where NB is
//>          the optimal blocksize.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument has an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup doubleOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dorglq_(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, j, l, ib, nb, ki, kk, nx, iws, nbmin, iinfo;
    extern /* Subroutine */ int dorgl2_(int *, int *, int *, double *, int *,
	    double *, double *, int *), dlarfb_(char *, char *, char *, char *
	    , int *, int *, int *, double *, int *, double *, int *, double *,
	     int *, double *, int *), dlarft_(char *, char *, int *, int *,
	    double *, int *, double *, double *, int *), xerbla_(char *, int *
	    );
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int ldwork, lwkopt;
    int lquery;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    nb = ilaenv_(&c__1, "DORGLQ", " ", m, n, k, &c_n1);
    lwkopt = max(1,*m) * nb;
    work[1] = (double) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < *m) {
	*info = -2;
    } else if (*k < 0 || *k > *m) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*lwork < max(1,*m) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORGLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m <= 0) {
	work[1] = 1.;
	return 0;
    }
    nbmin = 2;
    nx = 0;
    iws = *m;
    if (nb > 1 && nb < *k) {
	//
	//       Determine when to cross over from blocked to unblocked code.
	//
	// Computing MAX
	i__1 = 0, i__2 = ilaenv_(&c__3, "DORGLQ", " ", m, n, k, &c_n1);
	nx = max(i__1,i__2);
	if (nx < *k) {
	    //
	    //          Determine if workspace is large enough for blocked code.
	    //
	    ldwork = *m;
	    iws = ldwork * nb;
	    if (*lwork < iws) {
		//
		//             Not enough workspace to use optimal NB:  reduce NB and
		//             determine the minimum value of NB.
		//
		nb = *lwork / ldwork;
		// Computing MAX
		i__1 = 2, i__2 = ilaenv_(&c__2, "DORGLQ", " ", m, n, k, &c_n1)
			;
		nbmin = max(i__1,i__2);
	    }
	}
    }
    if (nb >= nbmin && nb < *k && nx < *k) {
	//
	//       Use blocked code after the last block.
	//       The first kk rows are handled by the block method.
	//
	ki = (*k - nx - 1) / nb * nb;
	// Computing MIN
	i__1 = *k, i__2 = ki + nb;
	kk = min(i__1,i__2);
	//
	//       Set A(kk+1:m,1:kk) to zero.
	//
	i__1 = kk;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = kk + 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = 0.;
// L10:
	    }
// L20:
	}
    } else {
	kk = 0;
    }
    //
    //    Use unblocked code for the last or only block.
    //
    if (kk < *m) {
	i__1 = *m - kk;
	i__2 = *n - kk;
	i__3 = *k - kk;
	dorgl2_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
		tau[kk + 1], &work[1], &iinfo);
    }
    if (kk > 0) {
	//
	//       Use blocked code
	//
	i__1 = -nb;
	for (i__ = ki + 1; i__1 < 0 ? i__ >= 1 : i__ <= 1; i__ += i__1) {
	    // Computing MIN
	    i__2 = nb, i__3 = *k - i__ + 1;
	    ib = min(i__2,i__3);
	    if (i__ + ib <= *m) {
		//
		//             Form the triangular factor of the block reflector
		//             H = H(i) H(i+1) . . . H(i+ib-1)
		//
		i__2 = *n - i__ + 1;
		dlarft_("Forward", "Rowwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);
		//
		//             Apply H**T to A(i+ib:m,i:n) from the right
		//
		i__2 = *m - i__ - ib + 1;
		i__3 = *n - i__ + 1;
		dlarfb_("Right", "Transpose", "Forward", "Rowwise", &i__2, &
			i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[1], &
			ldwork, &a[i__ + ib + i__ * a_dim1], lda, &work[ib +
			1], &ldwork);
	    }
	    //
	    //          Apply H**T to columns i:n of current block
	    //
	    i__2 = *n - i__ + 1;
	    dorgl2_(&ib, &i__2, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);
	    //
	    //          Set columns 1:i-1 of current block to zero
	    //
	    i__2 = i__ - 1;
	    for (j = 1; j <= i__2; ++j) {
		i__3 = i__ + ib - 1;
		for (l = i__; l <= i__3; ++l) {
		    a[l + j * a_dim1] = 0.;
// L30:
		}
// L40:
	    }
// L50:
	}
    }
    work[1] = (double) iws;
    return 0;
    //
    //    End of DORGLQ
    //
} // dorglq_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORMBR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORMBR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dormbr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dormbr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dormbr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORMBR( VECT, SIDE, TRANS, M, N, K, A, LDA, TAU, C,
//                         LDC, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS, VECT
//      INTEGER            INFO, K, LDA, LDC, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), C( LDC, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> If VECT = 'Q', DORMBR overwrites the general real M-by-N matrix C
//> with
//>                 SIDE = 'L'     SIDE = 'R'
//> TRANS = 'N':      Q * C          C * Q
//> TRANS = 'T':      Q**T * C       C * Q**T
//>
//> If VECT = 'P', DORMBR overwrites the general real M-by-N matrix C
//> with
//>                 SIDE = 'L'     SIDE = 'R'
//> TRANS = 'N':      P * C          C * P
//> TRANS = 'T':      P**T * C       C * P**T
//>
//> Here Q and P**T are the orthogonal matrices determined by DGEBRD when
//> reducing a real matrix A to bidiagonal form: A = Q * B * P**T. Q and
//> P**T are defined as products of elementary reflectors H(i) and G(i)
//> respectively.
//>
//> Let nq = m if SIDE = 'L' and nq = n if SIDE = 'R'. Thus nq is the
//> order of the orthogonal matrix Q or P**T that is applied.
//>
//> If VECT = 'Q', A is assumed to have been an NQ-by-K matrix:
//> if nq >= k, Q = H(1) H(2) . . . H(k);
//> if nq < k, Q = H(1) H(2) . . . H(nq-1).
//>
//> If VECT = 'P', A is assumed to have been a K-by-NQ matrix:
//> if k < nq, P = G(1) G(2) . . . G(k);
//> if k >= nq, P = G(1) G(2) . . . G(nq-1).
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] VECT
//> \verbatim
//>          VECT is CHARACTER*1
//>          = 'Q': apply Q or Q**T;
//>          = 'P': apply P or P**T.
//> \endverbatim
//>
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'L': apply Q, Q**T, P or P**T from the Left;
//>          = 'R': apply Q, Q**T, P or P**T from the Right.
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          = 'N':  No transpose, apply Q  or P;
//>          = 'T':  Transpose, apply Q**T or P**T.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix C. M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix C. N >= 0.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          If VECT = 'Q', the number of columns in the original
//>          matrix reduced by DGEBRD.
//>          If VECT = 'P', the number of rows in the original
//>          matrix reduced by DGEBRD.
//>          K >= 0.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension
//>                                (LDA,min(nq,K)) if VECT = 'Q'
//>                                (LDA,nq)        if VECT = 'P'
//>          The vectors which define the elementary reflectors H(i) and
//>          G(i), whose products determine the matrices Q and P, as
//>          returned by DGEBRD.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//>          If VECT = 'Q', LDA >= max(1,nq);
//>          if VECT = 'P', LDA >= max(1,min(nq,K)).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (min(nq,K))
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i) or G(i) which determines Q or P, as returned
//>          by DGEBRD in the array argument TAUQ or TAUP.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC,N)
//>          On entry, the M-by-N matrix C.
//>          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q
//>          or P*C or P**T*C or C*P or C*P**T.
//> \endverbatim
//>
//> \param[in] LDC
//> \verbatim
//>          LDC is INTEGER
//>          The leading dimension of the array C. LDC >= max(1,M).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK.
//>          If SIDE = 'L', LWORK >= max(1,N);
//>          if SIDE = 'R', LWORK >= max(1,M).
//>          For optimum performance LWORK >= N*NB if SIDE = 'L', and
//>          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
//>          blocksize.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup doubleOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dormbr_(char *vect, char *side, char *trans, int *m, int
	*n, int *k, double *a, int *lda, double *tau, double *c__, int *ldc,
	double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__2 = 2;

    // System generated locals
    address a__1[2];
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3[2];
    char ch__1[2+1]={'\0'};

    // Local variables
    int i1, i2, nb, mi, ni, nq, nw;
    int left;
    extern int lsame_(char *, char *);
    int iinfo;
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dormlq_(char *, char *, int *, int *, int *,
	    double *, int *, double *, double *, int *, double *, int *, int *
	    );
    int notran;
    extern /* Subroutine */ int dormqr_(char *, char *, int *, int *, int *,
	    double *, int *, double *, double *, int *, double *, int *, int *
	    );
    int applyq;
    char transt[1+1]={'\0'};
    int lwkopt;
    int lquery;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    *info = 0;
    applyq = lsame_(vect, "Q");
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;
    //
    //    NQ is the order of Q or P and NW is the minimum dimension of WORK
    //
    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! applyq && ! lsame_(vect, "P")) {
	*info = -1;
    } else if (! left && ! lsame_(side, "R")) {
	*info = -2;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*k < 0) {
	*info = -6;
    } else /* if(complicated condition) */ {
	// Computing MAX
	i__1 = 1, i__2 = min(nq,*k);
	if (applyq && *lda < max(1,nq) || ! applyq && *lda < max(i__1,i__2)) {
	    *info = -8;
	} else if (*ldc < max(1,*m)) {
	    *info = -11;
	} else if (*lwork < max(1,nw) && ! lquery) {
	    *info = -13;
	}
    }
    if (*info == 0) {
	if (applyq) {
	    if (left) {
		// Writing concatenation
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2);
		i__1 = *m - 1;
		i__2 = *m - 1;
		nb = ilaenv_(&c__1, "DORMQR", ch__1, &i__1, n, &i__2, &c_n1);
	    } else {
		// Writing concatenation
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "DORMQR", ch__1, m, &i__1, &i__2, &c_n1);
	    }
	} else {
	    if (left) {
		// Writing concatenation
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2);
		i__1 = *m - 1;
		i__2 = *m - 1;
		nb = ilaenv_(&c__1, "DORMLQ", ch__1, &i__1, n, &i__2, &c_n1);
	    } else {
		// Writing concatenation
		i__3[0] = 1, a__1[0] = side;
		i__3[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__3, &c__2);
		i__1 = *n - 1;
		i__2 = *n - 1;
		nb = ilaenv_(&c__1, "DORMLQ", ch__1, m, &i__1, &i__2, &c_n1);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORMBR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    work[1] = 1.;
    if (*m == 0 || *n == 0) {
	return 0;
    }
    if (applyq) {
	//
	//       Apply Q
	//
	if (nq >= *k) {
	    //
	    //          Q was determined by a call to DGEBRD with nq >= k
	    //
	    dormqr_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {
	    //
	    //          Q was determined by a call to DGEBRD with nq < k
	    //
	    if (left) {
		mi = *m - 1;
		ni = *n;
		i1 = 2;
		i2 = 1;
	    } else {
		mi = *m;
		ni = *n - 1;
		i1 = 1;
		i2 = 2;
	    }
	    i__1 = nq - 1;
	    dormqr_(side, trans, &mi, &ni, &i__1, &a[a_dim1 + 2], lda, &tau[1]
		    , &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
	}
    } else {
	//
	//       Apply P
	//
	if (notran) {
	    *(unsigned char *)transt = 'T';
	} else {
	    *(unsigned char *)transt = 'N';
	}
	if (nq > *k) {
	    //
	    //          P was determined by a call to DGEBRD with nq > k
	    //
	    dormlq_(side, transt, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		    c_offset], ldc, &work[1], lwork, &iinfo);
	} else if (nq > 1) {
	    //
	    //          P was determined by a call to DGEBRD with nq <= k
	    //
	    if (left) {
		mi = *m - 1;
		ni = *n;
		i1 = 2;
		i2 = 1;
	    } else {
		mi = *m;
		ni = *n - 1;
		i1 = 1;
		i2 = 2;
	    }
	    i__1 = nq - 1;
	    dormlq_(side, transt, &mi, &ni, &i__1, &a[(a_dim1 << 1) + 1], lda,
		     &tau[1], &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &
		    iinfo);
	}
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORMBR
    //
} // dormbr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORML2 multiplies a general matrix by the orthogonal matrix from a LQ factorization determined by sgelqf (unblocked algorithm).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORML2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorml2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorml2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorml2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORML2( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), C( LDC, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DORML2 overwrites the general real m by n matrix C with
//>
//>       Q * C  if SIDE = 'L' and TRANS = 'N', or
//>
//>       Q**T* C  if SIDE = 'L' and TRANS = 'T', or
//>
//>       C * Q  if SIDE = 'R' and TRANS = 'N', or
//>
//>       C * Q**T if SIDE = 'R' and TRANS = 'T',
//>
//> where Q is a real orthogonal matrix defined as the product of k
//> elementary reflectors
//>
//>       Q = H(k) . . . H(2) H(1)
//>
//> as returned by DGELQF. Q is of order m if SIDE = 'L' and of order n
//> if SIDE = 'R'.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'L': apply Q or Q**T from the Left
//>          = 'R': apply Q or Q**T from the Right
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          = 'N': apply Q  (No transpose)
//>          = 'T': apply Q**T (Transpose)
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix C. M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix C. N >= 0.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines
//>          the matrix Q.
//>          If SIDE = 'L', M >= K >= 0;
//>          if SIDE = 'R', N >= K >= 0.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension
//>                               (LDA,M) if SIDE = 'L',
//>                               (LDA,N) if SIDE = 'R'
//>          The i-th row must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          DGELQF in the first k rows of its array argument A.
//>          A is modified by the routine but restored on exit.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A. LDA >= max(1,K).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGELQF.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC,N)
//>          On entry, the m by n matrix C.
//>          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
//> \endverbatim
//>
//> \param[in] LDC
//> \verbatim
//>          LDC is INTEGER
//>          The leading dimension of the array C. LDC >= max(1,M).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension
//>                                   (N) if SIDE = 'L',
//>                                   (M) if SIDE = 'R'
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit
//>          < 0: if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup doubleOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dorml2_(char *side, char *trans, int *m, int *n, int *k,
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *info)
{
    // System generated locals
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;

    // Local variables
    int i__, i1, i2, i3, ic, jc, mi, ni, nq;
    double aii;
    int left;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    int notran;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    //
    //    NQ is the order of Q
    //
    if (left) {
	nq = *m;
    } else {
	nq = *n;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,*k)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORML2", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || *k == 0) {
	return 0;
    }
    if (left && notran || ! left && ! notran) {
	i1 = 1;
	i2 = *k;
	i3 = 1;
    } else {
	i1 = *k;
	i2 = 1;
	i3 = -1;
    }
    if (left) {
	ni = *n;
	jc = 1;
    } else {
	mi = *m;
	ic = 1;
    }
    i__1 = i2;
    i__2 = i3;
    for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	if (left) {
	    //
	    //          H(i) is applied to C(i:m,1:n)
	    //
	    mi = *m - i__ + 1;
	    ic = i__;
	} else {
	    //
	    //          H(i) is applied to C(1:m,i:n)
	    //
	    ni = *n - i__ + 1;
	    jc = i__;
	}
	//
	//       Apply H(i)
	//
	aii = a[i__ + i__ * a_dim1];
	a[i__ + i__ * a_dim1] = 1.;
	dlarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], lda, &tau[i__], &c__[
		ic + jc * c_dim1], ldc, &work[1]);
	a[i__ + i__ * a_dim1] = aii;
// L10:
    }
    return 0;
    //
    //    End of DORML2
    //
} // dorml2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORMLQ
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORMLQ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dormlq.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dormlq.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dormlq.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORMLQ( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), C( LDC, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DORMLQ overwrites the general real M-by-N matrix C with
//>
//>                 SIDE = 'L'     SIDE = 'R'
//> TRANS = 'N':      Q * C          C * Q
//> TRANS = 'T':      Q**T * C       C * Q**T
//>
//> where Q is a real orthogonal matrix defined as the product of k
//> elementary reflectors
//>
//>       Q = H(k) . . . H(2) H(1)
//>
//> as returned by DGELQF. Q is of order M if SIDE = 'L' and of order N
//> if SIDE = 'R'.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'L': apply Q or Q**T from the Left;
//>          = 'R': apply Q or Q**T from the Right.
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          = 'N':  No transpose, apply Q;
//>          = 'T':  Transpose, apply Q**T.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix C. M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix C. N >= 0.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines
//>          the matrix Q.
//>          If SIDE = 'L', M >= K >= 0;
//>          if SIDE = 'R', N >= K >= 0.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension
//>                               (LDA,M) if SIDE = 'L',
//>                               (LDA,N) if SIDE = 'R'
//>          The i-th row must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          DGELQF in the first k rows of its array argument A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A. LDA >= max(1,K).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGELQF.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC,N)
//>          On entry, the M-by-N matrix C.
//>          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
//> \endverbatim
//>
//> \param[in] LDC
//> \verbatim
//>          LDC is INTEGER
//>          The leading dimension of the array C. LDC >= max(1,M).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK.
//>          If SIDE = 'L', LWORK >= max(1,N);
//>          if SIDE = 'R', LWORK >= max(1,M).
//>          For good performance, LWORK should generally be larger.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//> \endverbatim
//
// Authors:
// ========
//
//> \author Univ. of Tennessee
//> \author Univ. of California Berkeley
//> \author Univ. of Colorado Denver
//> \author NAG Ltd.
//
//> \date December 2016
//
//> \ingroup doubleOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dormlq_(char *side, char *trans, int *m, int *n, int *k,
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__2 = 2;
    int c__65 = 65;

    // System generated locals
    address a__1[2];
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3[2], i__4, i__5;
    char ch__1[2+1]={'\0'};

    // Local variables
    int i__, i1, i2, i3, ib, ic, jc, nb, mi, ni, nq, nw, iwt;
    int left;
    extern int lsame_(char *, char *);
    int nbmin, iinfo;
    extern /* Subroutine */ int dorml2_(char *, char *, int *, int *, int *,
	    double *, int *, double *, double *, int *, double *, int *),
	    dlarfb_(char *, char *, char *, char *, int *, int *, int *,
	    double *, int *, double *, int *, double *, int *, double *, int *
	    ), dlarft_(char *, char *, int *, int *, double *, int *, double *
	    , double *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int notran;
    int ldwork;
    char transt[1+1]={'\0'};
    int lwkopt;
    int lquery;

    //
    // -- LAPACK computational routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    *info = 0;
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;
    //
    //    NQ is the order of Q and NW is the minimum dimension of WORK
    //
    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,*k)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }
    if (*info == 0) {
	//
	//       Compute the workspace requirements
	//
	// Computing MIN
	// Writing concatenation
	i__3[0] = 1, a__1[0] = side;
	i__3[1] = 1, a__1[1] = trans;
	s_cat(ch__1, a__1, i__3, &c__2);
	i__1 = 64, i__2 = ilaenv_(&c__1, "DORMLQ", ch__1, m, n, k, &c_n1);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb + 4160;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORMLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || *k == 0) {
	work[1] = 1.;
	return 0;
    }
    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < *k) {
	if (*lwork < nw * nb + 4160) {
	    nb = (*lwork - 4160) / ldwork;
	    // Computing MAX
	    // Writing concatenation
	    i__3[0] = 1, a__1[0] = side;
	    i__3[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__3, &c__2);
	    i__1 = 2, i__2 = ilaenv_(&c__2, "DORMLQ", ch__1, m, n, k, &c_n1);
	    nbmin = max(i__1,i__2);
	}
    }
    if (nb < nbmin || nb >= *k) {
	//
	//       Use unblocked code
	//
	dorml2_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {
	//
	//       Use blocked code
	//
	iwt = nw * nb + 1;
	if (left && notran || ! left && ! notran) {
	    i1 = 1;
	    i2 = *k;
	    i3 = nb;
	} else {
	    i1 = (*k - 1) / nb * nb + 1;
	    i2 = 1;
	    i3 = -nb;
	}
	if (left) {
	    ni = *n;
	    jc = 1;
	} else {
	    mi = *m;
	    ic = 1;
	}
	if (notran) {
	    *(unsigned char *)transt = 'T';
	} else {
	    *(unsigned char *)transt = 'N';
	}
	i__1 = i2;
	i__2 = i3;
	for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	    // Computing MIN
	    i__4 = nb, i__5 = *k - i__ + 1;
	    ib = min(i__4,i__5);
	    //
	    //          Form the triangular factor of the block reflector
	    //          H = H(i) H(i+1) . . . H(i+ib-1)
	    //
	    i__4 = nq - i__ + 1;
	    dlarft_("Forward", "Rowwise", &i__4, &ib, &a[i__ + i__ * a_dim1],
		    lda, &tau[i__], &work[iwt], &c__65);
	    if (left) {
		//
		//             H or H**T is applied to C(i:m,1:n)
		//
		mi = *m - i__ + 1;
		ic = i__;
	    } else {
		//
		//             H or H**T is applied to C(1:m,i:n)
		//
		ni = *n - i__ + 1;
		jc = i__;
	    }
	    //
	    //          Apply H or H**T
	    //
	    dlarfb_(side, transt, "Forward", "Rowwise", &mi, &ni, &ib, &a[i__
		    + i__ * a_dim1], lda, &work[iwt], &c__65, &c__[ic + jc *
		    c_dim1], ldc, &work[1], &ldwork);
// L10:
	}
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORMLQ
    //
} // dormlq_

