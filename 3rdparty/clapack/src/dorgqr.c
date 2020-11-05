/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DORG2R generates all or part of the orthogonal matrix Q from a QR factorization determined by sgeqrf (unblocked algorithm).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORG2R + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorg2r.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorg2r.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorg2r.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORG2R( M, N, K, A, LDA, TAU, WORK, INFO )
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
//> DORG2R generates an m by n real matrix Q with orthonormal columns,
//> which is defined as the first n columns of a product of k elementary
//> reflectors of order m
//>
//>       Q  =  H(1) H(2) . . . H(k)
//>
//> as returned by DGEQRF.
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
//>          The number of columns of the matrix Q. M >= N >= 0.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines the
//>          matrix Q. N >= K >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the i-th column must contain the vector which
//>          defines the elementary reflector H(i), for i = 1,2,...,k, as
//>          returned by DGEQRF in the first k columns of its array
//>          argument A.
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
//>          reflector H(i), as returned by DGEQRF.
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
/* Subroutine */ int dorg2r_(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;

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
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORG2R", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    //
    //    Initialise columns k+1:n to columns of the unit matrix
    //
    i__1 = *n;
    for (j = *k + 1; j <= i__1; ++j) {
	i__2 = *m;
	for (l = 1; l <= i__2; ++l) {
	    a[l + j * a_dim1] = 0.;
// L10:
	}
	a[j + j * a_dim1] = 1.;
// L20:
    }
    for (i__ = *k; i__ >= 1; --i__) {
	//
	//       Apply H(i) to A(i:m,i:n) from the left
	//
	if (i__ < *n) {
	    a[i__ + i__ * a_dim1] = 1.;
	    i__1 = *m - i__ + 1;
	    i__2 = *n - i__;
	    dlarf_("Left", &i__1, &i__2, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	}
	if (i__ < *m) {
	    i__1 = *m - i__;
	    d__1 = -tau[i__];
	    dscal_(&i__1, &d__1, &a[i__ + 1 + i__ * a_dim1], &c__1);
	}
	a[i__ + i__ * a_dim1] = 1. - tau[i__];
	//
	//       Set A(1:i-1,i) to zero
	//
	i__1 = i__ - 1;
	for (l = 1; l <= i__1; ++l) {
	    a[l + i__ * a_dim1] = 0.;
// L30:
	}
// L40:
    }
    return 0;
    //
    //    End of DORG2R
    //
} // dorg2r_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORGQR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORGQR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorgqr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorgqr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorgqr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORGQR( M, N, K, A, LDA, TAU, WORK, LWORK, INFO )
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
//> DORGQR generates an M-by-N real matrix Q with orthonormal columns,
//> which is defined as the first N columns of a product of K elementary
//> reflectors of order M
//>
//>       Q  =  H(1) H(2) . . . H(k)
//>
//> as returned by DGEQRF.
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
//>          The number of columns of the matrix Q. M >= N >= 0.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines the
//>          matrix Q. N >= K >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the i-th column must contain the vector which
//>          defines the elementary reflector H(i), for i = 1,2,...,k, as
//>          returned by DGEQRF in the first k columns of its array
//>          argument A.
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
//>          reflector H(i), as returned by DGEQRF.
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
//>          The dimension of the array WORK. LWORK >= max(1,N).
//>          For optimum performance LWORK >= N*NB, where NB is the
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
/* Subroutine */ int dorgqr_(int *m, int *n, int *k, double *a, int *lda,
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
    extern /* Subroutine */ int dorg2r_(int *, int *, int *, double *, int *,
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
    nb = ilaenv_(&c__1, "DORGQR", " ", m, n, k, &c_n1);
    lwkopt = max(1,*n) * nb;
    work[1] = (double) lwkopt;
    lquery = *lwork == -1;
    if (*m < 0) {
	*info = -1;
    } else if (*n < 0 || *n > *m) {
	*info = -2;
    } else if (*k < 0 || *k > *n) {
	*info = -3;
    } else if (*lda < max(1,*m)) {
	*info = -5;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORGQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	work[1] = 1.;
	return 0;
    }
    nbmin = 2;
    nx = 0;
    iws = *n;
    if (nb > 1 && nb < *k) {
	//
	//       Determine when to cross over from blocked to unblocked code.
	//
	// Computing MAX
	i__1 = 0, i__2 = ilaenv_(&c__3, "DORGQR", " ", m, n, k, &c_n1);
	nx = max(i__1,i__2);
	if (nx < *k) {
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "DORGQR", " ", m, n, k, &c_n1)
			;
		nbmin = max(i__1,i__2);
	    }
	}
    }
    if (nb >= nbmin && nb < *k && nx < *k) {
	//
	//       Use blocked code after the last block.
	//       The first kk columns are handled by the block method.
	//
	ki = (*k - nx - 1) / nb * nb;
	// Computing MIN
	i__1 = *k, i__2 = ki + nb;
	kk = min(i__1,i__2);
	//
	//       Set A(1:kk,kk+1:n) to zero.
	//
	i__1 = *n;
	for (j = kk + 1; j <= i__1; ++j) {
	    i__2 = kk;
	    for (i__ = 1; i__ <= i__2; ++i__) {
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
    if (kk < *n) {
	i__1 = *m - kk;
	i__2 = *n - kk;
	i__3 = *k - kk;
	dorg2r_(&i__1, &i__2, &i__3, &a[kk + 1 + (kk + 1) * a_dim1], lda, &
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
	    if (i__ + ib <= *n) {
		//
		//             Form the triangular factor of the block reflector
		//             H = H(i) H(i+1) . . . H(i+ib-1)
		//
		i__2 = *m - i__ + 1;
		dlarft_("Forward", "Columnwise", &i__2, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);
		//
		//             Apply H to A(i:m,i+ib:n) from the left
		//
		i__2 = *m - i__ + 1;
		i__3 = *n - i__ - ib + 1;
		dlarfb_("Left", "No transpose", "Forward", "Columnwise", &
			i__2, &i__3, &ib, &a[i__ + i__ * a_dim1], lda, &work[
			1], &ldwork, &a[i__ + (i__ + ib) * a_dim1], lda, &
			work[ib + 1], &ldwork);
	    }
	    //
	    //          Apply H to rows i:m of current block
	    //
	    i__2 = *m - i__ + 1;
	    dorg2r_(&i__2, &ib, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &
		    work[1], &iinfo);
	    //
	    //          Set rows 1:i-1 of current block to zero
	    //
	    i__2 = i__ + ib - 1;
	    for (j = i__; j <= i__2; ++j) {
		i__3 = i__ - 1;
		for (l = 1; l <= i__3; ++l) {
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
    //    End of DORGQR
    //
} // dorgqr_

