/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

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

