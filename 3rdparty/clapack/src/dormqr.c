/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DORM2R multiplies a general matrix by the orthogonal matrix from a QR factorization determined by sgeqrf (unblocked algorithm).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORM2R + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorm2r.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorm2r.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorm2r.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORM2R( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
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
//> DORM2R overwrites the general real m by n matrix C with
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
//>       Q = H(1) H(2) . . . H(k)
//>
//> as returned by DGEQRF. Q is of order m if SIDE = 'L' and of order n
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
//>          A is DOUBLE PRECISION array, dimension (LDA,K)
//>          The i-th column must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          DGEQRF in the first k columns of its array argument A.
//>          A is modified by the routine but restored on exit.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//>          If SIDE = 'L', LDA >= max(1,M);
//>          if SIDE = 'R', LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGEQRF.
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
/* Subroutine */ int dorm2r_(char *side, char *trans, int *m, int *n, int *k,
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *info)
{
    // Table of constant values
    int c__1 = 1;

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
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORM2R", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || *k == 0) {
	return 0;
    }
    if (left && ! notran || ! left && notran) {
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
	dlarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], &c__1, &tau[i__], &c__[
		ic + jc * c_dim1], ldc, &work[1]);
	a[i__ + i__ * a_dim1] = aii;
// L10:
    }
    return 0;
    //
    //    End of DORM2R
    //
} // dorm2r_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORMQR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORMQR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dormqr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dormqr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dormqr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORMQR( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
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
//> DORMQR overwrites the general real M-by-N matrix C with
//>
//>                 SIDE = 'L'     SIDE = 'R'
//> TRANS = 'N':      Q * C          C * Q
//> TRANS = 'T':      Q**T * C       C * Q**T
//>
//> where Q is a real orthogonal matrix defined as the product of k
//> elementary reflectors
//>
//>       Q = H(1) H(2) . . . H(k)
//>
//> as returned by DGEQRF. Q is of order M if SIDE = 'L' and of order N
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
//>          A is DOUBLE PRECISION array, dimension (LDA,K)
//>          The i-th column must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          DGEQRF in the first k columns of its array argument A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//>          If SIDE = 'L', LDA >= max(1,M);
//>          if SIDE = 'R', LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGEQRF.
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
/* Subroutine */ int dormqr_(char *side, char *trans, int *m, int *n, int *k,
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
    extern /* Subroutine */ int dorm2r_(char *, char *, int *, int *, int *,
	    double *, int *, double *, double *, int *, double *, int *),
	    dlarfb_(char *, char *, char *, char *, int *, int *, int *,
	    double *, int *, double *, int *, double *, int *, double *, int *
	    ), dlarft_(char *, char *, int *, int *, double *, int *, double *
	    , double *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int notran;
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
    } else if (*lda < max(1,nq)) {
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "DORMQR", ch__1, m, n, k, &c_n1);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb + 4160;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORMQR", &i__1);
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "DORMQR", ch__1, m, n, k, &c_n1);
	    nbmin = max(i__1,i__2);
	}
    }
    if (nb < nbmin || nb >= *k) {
	//
	//       Use unblocked code
	//
	dorm2r_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {
	//
	//       Use blocked code
	//
	iwt = nw * nb + 1;
	if (left && ! notran || ! left && notran) {
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
	    dlarft_("Forward", "Columnwise", &i__4, &ib, &a[i__ + i__ *
		    a_dim1], lda, &tau[i__], &work[iwt], &c__65);
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
	    dlarfb_(side, trans, "Forward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ + i__ * a_dim1], lda, &work[iwt], &c__65, &c__[ic +
		    jc * c_dim1], ldc, &work[1], &ldwork);
// L10:
	}
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORMQR
    //
} // dormqr_

