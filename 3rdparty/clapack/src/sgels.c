/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b ILASLC scans a matrix for its last non-zero column.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download ILASLC + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ilaslc.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ilaslc.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ilaslc.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER FUNCTION ILASLC( M, N, A, LDA )
//
//      .. Scalar Arguments ..
//      INTEGER            M, N, LDA
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> ILASLC scans A for its last non-zero column.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          The m by n matrix A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A. LDA >= max(1,M).
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
//> \ingroup realOTHERauxiliary
//
// =====================================================================
int ilaslc_(int *m, int *n, float *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, ret_val, i__1;

    // Local variables
    int i__;

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
    //    .. Executable Statements ..
    //
    //    Quick test for the common case where one corner is non-zero.
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    if (*n == 0) {
	ret_val = *n;
    } else if (a[*n * a_dim1 + 1] != 0.f || a[*m + *n * a_dim1] != 0.f) {
	ret_val = *n;
    } else {
	//    Now scan each column from the end, returning with the first non-zero.
	for (ret_val = *n; ret_val >= 1; --ret_val) {
	    i__1 = *m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		if (a[i__ + ret_val * a_dim1] != 0.f) {
		    return ret_val;
		}
	    }
	}
    }
    return ret_val;
} // ilaslc_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b ILASLR scans a matrix for its last non-zero row.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download ILASLR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/ilaslr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/ilaslr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/ilaslr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER FUNCTION ILASLR( M, N, A, LDA )
//
//      .. Scalar Arguments ..
//      INTEGER            M, N, LDA
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> ILASLR scans A for its last non-zero row.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          The m by n matrix A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A. LDA >= max(1,M).
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
//> \ingroup realOTHERauxiliary
//
// =====================================================================
int ilaslr_(int *m, int *n, float *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, ret_val, i__1;

    // Local variables
    int i__, j;

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
    //    .. Executable Statements ..
    //
    //    Quick test for the common case where one corner is non-zero.
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    if (*m == 0) {
	ret_val = *m;
    } else if (a[*m + a_dim1] != 0.f || a[*m + *n * a_dim1] != 0.f) {
	ret_val = *m;
    } else {
	//    Scan up each column tracking the last zero row seen.
	ret_val = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__ = *m;
	    while(a[max(i__,1) + j * a_dim1] == 0.f && i__ >= 1) {
		--i__;
	    }
	    ret_val = max(ret_val,i__);
	}
    }
    return ret_val;
} // ilaslr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SCOMBSSQ adds two scaled sum of squares quantities
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//
// Definition:
// ===========
//
//      SUBROUTINE SCOMBSSQ( V1, V2 )
//
//      .. Array Arguments ..
//      REAL               V1( 2 ), V2( 2 )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SCOMBSSQ adds two scaled sum of squares quantities, V1 := V1 + V2.
//> That is,
//>
//>    V1_scale**2 * V1_sumsq := V1_scale**2 * V1_sumsq
//>                            + V2_scale**2 * V2_sumsq
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in,out] V1
//> \verbatim
//>          V1 is REAL array, dimension (2).
//>          The first scaled sum.
//>          V1(1) = V1_scale, V1(2) = V1_sumsq.
//> \endverbatim
//>
//> \param[in] V2
//> \verbatim
//>          V2 is REAL array, dimension (2).
//>          The second scaled sum.
//>          V2(1) = V2_scale, V2(2) = V2_sumsq.
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
//> \date November 2018
//
//> \ingroup OTHERauxiliary
//
// =====================================================================
/* Subroutine */ int scombssq_(float *v1, float *v2)
{
    // System generated locals
    float r__1;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    November 2018
    //
    //    .. Array Arguments ..
    //    ..
    //
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --v2;
    --v1;

    // Function Body
    if (v1[1] >= v2[1]) {
	if (v1[1] != 0.f) {
	    // Computing 2nd power
	    r__1 = v2[1] / v1[1];
	    v1[2] += r__1 * r__1 * v2[2];
	}
    } else {
	// Computing 2nd power
	r__1 = v1[1] / v2[1];
	v1[2] = v2[2] + r__1 * r__1 * v1[2];
	v1[1] = v2[1];
    }
    return 0;
    //
    //    End of SCOMBSSQ
    //
} // scombssq_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SCOPY
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE SCOPY(N,SX,INCX,SY,INCY)
//
//      .. Scalar Arguments ..
//      INTEGER INCX,INCY,N
//      ..
//      .. Array Arguments ..
//      REAL SX(*),SY(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    SCOPY copies a vector, x, to a vector, y.
//>    uses unrolled loops for increments equal to 1.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>         number of elements in input vector(s)
//> \endverbatim
//>
//> \param[in] SX
//> \verbatim
//>          SX is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>         storage spacing between elements of SX
//> \endverbatim
//>
//> \param[out] SY
//> \verbatim
//>          SY is REAL array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
//> \endverbatim
//>
//> \param[in] INCY
//> \verbatim
//>          INCY is INTEGER
//>         storage spacing between elements of SY
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
//> \ingroup single_blas_level1
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>     jack dongarra, linpack, 3/11/78.
//>     modified 12/3/93, array(1) declarations changed to array(*)
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int scopy_(int *n, float *sx, int *incx, float *sy, int *
	incy)
{
    // System generated locals
    int i__1;

    // Local variables
    int i__, m, ix, iy, mp1;

    //
    // -- Reference BLAS level1 routine (version 3.8.0) --
    // -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
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
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    // Parameter adjustments
    --sy;
    --sx;

    // Function Body
    if (*n <= 0) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	//
	//       code for both increments equal to 1
	//
	//
	//       clean-up loop
	//
	m = *n % 7;
	if (m != 0) {
	    i__1 = m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		sy[i__] = sx[i__];
	    }
	    if (*n < 7) {
		return 0;
	    }
	}
	mp1 = m + 1;
	i__1 = *n;
	for (i__ = mp1; i__ <= i__1; i__ += 7) {
	    sy[i__] = sx[i__];
	    sy[i__ + 1] = sx[i__ + 1];
	    sy[i__ + 2] = sx[i__ + 2];
	    sy[i__ + 3] = sx[i__ + 3];
	    sy[i__ + 4] = sx[i__ + 4];
	    sy[i__ + 5] = sx[i__ + 5];
	    sy[i__ + 6] = sx[i__ + 6];
	}
    } else {
	//
	//       code for unequal increments or equal increments
	//         not equal to 1
	//
	ix = 1;
	iy = 1;
	if (*incx < 0) {
	    ix = (-(*n) + 1) * *incx + 1;
	}
	if (*incy < 0) {
	    iy = (-(*n) + 1) * *incy + 1;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    sy[iy] = sx[ix];
	    ix += *incx;
	    iy += *incy;
	}
    }
    return 0;
} // scopy_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGELQ2 computes the LQ factorization of a general rectangular matrix using an unblocked algorithm.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGELQ2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgelq2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgelq2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgelq2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGELQ2( M, N, A, LDA, TAU, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGELQ2 computes an LQ factorization of a real m-by-n matrix A:
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
//>          A is REAL array, dimension (LDA,N)
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
//>          TAU is REAL array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is REAL array, dimension (M)
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
//> \ingroup realGEcomputational
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
/* Subroutine */ int sgelq2_(int *m, int *n, float *a, int *lda, float *tau,
	float *work, int *info)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, k;
    float aii;
    extern /* Subroutine */ int slarf_(char *, int *, int *, float *, int *,
	    float *, float *, int *, float *), xerbla_(char *, int *),
	    slarfg_(int *, float *, float *, int *, float *);

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
	xerbla_("SGELQ2", &i__1);
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
	slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3,*n) * a_dim1]
		, lda, &tau[i__]);
	if (i__ < *m) {
	    //
	    //          Apply H(i) to A(i+1:m,i:n) from the right
	    //
	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.f;
	    i__2 = *m - i__;
	    i__3 = *n - i__ + 1;
	    slarf_("Right", &i__2, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[
		    i__], &a[i__ + 1 + i__ * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
// L10:
    }
    return 0;
    //
    //    End of SGELQ2
    //
} // sgelq2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGELQF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGELQF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgelqf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgelqf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgelqf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGELQF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGELQF computes an LQ factorization of a real M-by-N matrix A:
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
//>          A is REAL array, dimension (LDA,N)
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
//>          TAU is REAL array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is REAL array, dimension (MAX(1,LWORK))
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
//> \ingroup realGEcomputational
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
/* Subroutine */ int sgelqf_(int *m, int *n, float *a, int *lda, float *tau,
	float *work, int *lwork, int *info)
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
    extern /* Subroutine */ int sgelq2_(int *, int *, float *, int *, float *,
	     float *, int *), slarfb_(char *, char *, char *, char *, int *,
	    int *, int *, float *, int *, float *, int *, float *, int *,
	    float *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int slarft_(char *, char *, int *, int *, float *,
	     int *, float *, float *, int *);
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
    nb = ilaenv_(&c__1, "SGELQF", " ", m, n, &c_n1, &c_n1);
    lwkopt = *m * nb;
    work[1] = (float) lwkopt;
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
	xerbla_("SGELQF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    k = min(*m,*n);
    if (k == 0) {
	work[1] = 1.f;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "SGELQF", " ", m, n, &c_n1, &c_n1);
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "SGELQF", " ", m, n, &c_n1, &
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
	    sgelq2_(&ib, &i__3, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *m) {
		//
		//             Form the triangular factor of the block reflector
		//             H = H(i) H(i+1) . . . H(i+ib-1)
		//
		i__3 = *n - i__ + 1;
		slarft_("Forward", "Rowwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);
		//
		//             Apply H to A(i+ib:m,i:n) from the right
		//
		i__3 = *m - i__ - ib + 1;
		i__4 = *n - i__ + 1;
		slarfb_("Right", "No transpose", "Forward", "Rowwise", &i__3,
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
	sgelq2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }
    work[1] = (float) iws;
    return 0;
    //
    //    End of SGELQF
    //
} // sgelqf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief <b> SGELS solves overdetermined or underdetermined systems for GE matrices</b>
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGELS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgels.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgels.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgels.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGELS( TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK,
//                        INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          TRANS
//      INTEGER            INFO, LDA, LDB, LWORK, M, N, NRHS
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), B( LDB, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGELS solves overdetermined or underdetermined real linear systems
//> involving an M-by-N matrix A, or its transpose, using a QR or LQ
//> factorization of A.  It is assumed that A has full rank.
//>
//> The following options are provided:
//>
//> 1. If TRANS = 'N' and m >= n:  find the least squares solution of
//>    an overdetermined system, i.e., solve the least squares problem
//>                 minimize || B - A*X ||.
//>
//> 2. If TRANS = 'N' and m < n:  find the minimum norm solution of
//>    an underdetermined system A * X = B.
//>
//> 3. If TRANS = 'T' and m >= n:  find the minimum norm solution of
//>    an underdetermined system A**T * X = B.
//>
//> 4. If TRANS = 'T' and m < n:  find the least squares solution of
//>    an overdetermined system, i.e., solve the least squares problem
//>                 minimize || B - A**T * X ||.
//>
//> Several right hand side vectors b and solution vectors x can be
//> handled in a single call; they are stored as the columns of the
//> M-by-NRHS right hand side matrix B and the N-by-NRHS solution
//> matrix X.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          = 'N': the linear system involves A;
//>          = 'T': the linear system involves A**T.
//> \endverbatim
//>
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
//> \param[in] NRHS
//> \verbatim
//>          NRHS is INTEGER
//>          The number of right hand sides, i.e., the number of
//>          columns of the matrices B and X. NRHS >=0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          On entry, the M-by-N matrix A.
//>          On exit,
//>            if M >= N, A is overwritten by details of its QR
//>                       factorization as returned by SGEQRF;
//>            if M <  N, A is overwritten by details of its LQ
//>                       factorization as returned by SGELQF.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[in,out] B
//> \verbatim
//>          B is REAL array, dimension (LDB,NRHS)
//>          On entry, the matrix B of right hand side vectors, stored
//>          columnwise; B is M-by-NRHS if TRANS = 'N', or N-by-NRHS
//>          if TRANS = 'T'.
//>          On exit, if INFO = 0, B is overwritten by the solution
//>          vectors, stored columnwise:
//>          if TRANS = 'N' and m >= n, rows 1 to n of B contain the least
//>          squares solution vectors; the residual sum of squares for the
//>          solution in each column is given by the sum of squares of
//>          elements N+1 to M in that column;
//>          if TRANS = 'N' and m < n, rows 1 to N of B contain the
//>          minimum norm solution vectors;
//>          if TRANS = 'T' and m >= n, rows 1 to M of B contain the
//>          minimum norm solution vectors;
//>          if TRANS = 'T' and m < n, rows 1 to M of B contain the
//>          least squares solution vectors; the residual sum of squares
//>          for the solution in each column is given by the sum of
//>          squares of elements M+1 to N in that column.
//> \endverbatim
//>
//> \param[in] LDB
//> \verbatim
//>          LDB is INTEGER
//>          The leading dimension of the array B. LDB >= MAX(1,M,N).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is REAL array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK.
//>          LWORK >= max( 1, MN + max( MN, NRHS ) ).
//>          For optimal performance,
//>          LWORK >= max( 1, MN + max( MN, NRHS )*NB ).
//>          where MN = min(M,N) and NB is the optimum block size.
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
//>          > 0:  if INFO =  i, the i-th diagonal element of the
//>                triangular factor of A is zero, so that A does not have
//>                full rank; the least squares solution could not be
//>                computed.
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
//> \ingroup realGEsolve
//
// =====================================================================
/* Subroutine */ int sgels_(char *trans, int *m, int *n, int *nrhs, float *a,
	int *lda, float *b, int *ldb, float *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    float c_b33 = 0.f;
    int c__0 = 0;

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    // Local variables
    int i__, j, nb, mn;
    float anrm, bnrm;
    int brow;
    int tpsd;
    int iascl, ibscl;
    extern int lsame_(char *, char *);
    int wsize;
    float rwork[1];
    extern /* Subroutine */ int slabad_(float *, float *);
    extern double slamch_(char *), slange_(char *, int *, int *, float *, int
	    *, float *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int scllen;
    float bignum;
    extern /* Subroutine */ int sgelqf_(int *, int *, float *, int *, float *,
	     float *, int *, int *), slascl_(char *, int *, int *, float *,
	    float *, int *, int *, float *, int *, int *), sgeqrf_(int *, int
	    *, float *, int *, float *, float *, int *, int *), slaset_(char *
	    , int *, int *, float *, float *, float *, int *);
    float smlnum;
    extern /* Subroutine */ int sormlq_(char *, char *, int *, int *, int *,
	    float *, int *, float *, float *, int *, float *, int *, int *);
    int lquery;
    extern /* Subroutine */ int sormqr_(char *, char *, int *, int *, int *,
	    float *, int *, float *, float *, int *, float *, int *, int *),
	    strtrs_(char *, char *, char *, int *, int *, float *, int *,
	    float *, int *, int *);

    //
    // -- LAPACK driver routine (version 3.7.0) --
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
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments.
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    --work;

    // Function Body
    *info = 0;
    mn = min(*m,*n);
    lquery = *lwork == -1;
    if (! (lsame_(trans, "N") || lsame_(trans, "T"))) {
	*info = -1;
    } else if (*m < 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*nrhs < 0) {
	*info = -4;
    } else if (*lda < max(1,*m)) {
	*info = -6;
    } else /* if(complicated condition) */ {
	// Computing MAX
	i__1 = max(1,*m);
	if (*ldb < max(i__1,*n)) {
	    *info = -8;
	} else /* if(complicated condition) */ {
	    // Computing MAX
	    i__1 = 1, i__2 = mn + max(mn,*nrhs);
	    if (*lwork < max(i__1,i__2) && ! lquery) {
		*info = -10;
	    }
	}
    }
    //
    //    Figure out optimal block size
    //
    if (*info == 0 || *info == -10) {
	tpsd = TRUE_;
	if (lsame_(trans, "N")) {
	    tpsd = FALSE_;
	}
	if (*m >= *n) {
	    nb = ilaenv_(&c__1, "SGEQRF", " ", m, n, &c_n1, &c_n1);
	    if (tpsd) {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "SORMQR", "LN", m, nrhs, n, &
			c_n1);
		nb = max(i__1,i__2);
	    } else {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "SORMQR", "LT", m, nrhs, n, &
			c_n1);
		nb = max(i__1,i__2);
	    }
	} else {
	    nb = ilaenv_(&c__1, "SGELQF", " ", m, n, &c_n1, &c_n1);
	    if (tpsd) {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "SORMLQ", "LT", n, nrhs, m, &
			c_n1);
		nb = max(i__1,i__2);
	    } else {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "SORMLQ", "LN", n, nrhs, m, &
			c_n1);
		nb = max(i__1,i__2);
	    }
	}
	//
	// Computing MAX
	i__1 = 1, i__2 = mn + max(mn,*nrhs) * nb;
	wsize = max(i__1,i__2);
	work[1] = (float) wsize;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGELS ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    // Computing MIN
    i__1 = min(*m,*n);
    if (min(i__1,*nrhs) == 0) {
	i__1 = max(*m,*n);
	slaset_("Full", &i__1, nrhs, &c_b33, &c_b33, &b[b_offset], ldb);
	return 0;
    }
    //
    //    Get machine parameters
    //
    smlnum = slamch_("S") / slamch_("P");
    bignum = 1.f / smlnum;
    slabad_(&smlnum, &bignum);
    //
    //    Scale A, B if max element outside range [SMLNUM,BIGNUM]
    //
    anrm = slange_("M", m, n, &a[a_offset], lda, rwork);
    iascl = 0;
    if (anrm > 0.f && anrm < smlnum) {
	//
	//       Scale matrix norm up to SMLNUM
	//
	slascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda,
		info);
	iascl = 1;
    } else if (anrm > bignum) {
	//
	//       Scale matrix norm down to BIGNUM
	//
	slascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda,
		info);
	iascl = 2;
    } else if (anrm == 0.f) {
	//
	//       Matrix all zero. Return zero solution.
	//
	i__1 = max(*m,*n);
	slaset_("F", &i__1, nrhs, &c_b33, &c_b33, &b[b_offset], ldb);
	goto L50;
    }
    brow = *m;
    if (tpsd) {
	brow = *n;
    }
    bnrm = slange_("M", &brow, nrhs, &b[b_offset], ldb, rwork);
    ibscl = 0;
    if (bnrm > 0.f && bnrm < smlnum) {
	//
	//       Scale matrix norm up to SMLNUM
	//
	slascl_("G", &c__0, &c__0, &bnrm, &smlnum, &brow, nrhs, &b[b_offset],
		ldb, info);
	ibscl = 1;
    } else if (bnrm > bignum) {
	//
	//       Scale matrix norm down to BIGNUM
	//
	slascl_("G", &c__0, &c__0, &bnrm, &bignum, &brow, nrhs, &b[b_offset],
		ldb, info);
	ibscl = 2;
    }
    if (*m >= *n) {
	//
	//       compute QR factorization of A
	//
	i__1 = *lwork - mn;
	sgeqrf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
		;
	//
	//       workspace at least N, optimally N*NB
	//
	if (! tpsd) {
	    //
	    //          Least-Squares Problem min || A * X - B ||
	    //
	    //          B(1:M,1:NRHS) := Q**T * B(1:M,1:NRHS)
	    //
	    i__1 = *lwork - mn;
	    sormqr_("Left", "Transpose", m, nrhs, n, &a[a_offset], lda, &work[
		    1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);
	    //
	    //          workspace at least NRHS, optimally NRHS*NB
	    //
	    //          B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
	    //
	    strtrs_("Upper", "No transpose", "Non-unit", n, nrhs, &a[a_offset]
		    , lda, &b[b_offset], ldb, info);
	    if (*info > 0) {
		return 0;
	    }
	    scllen = *n;
	} else {
	    //
	    //          Underdetermined system of equations A**T * X = B
	    //
	    //          B(1:N,1:NRHS) := inv(R**T) * B(1:N,1:NRHS)
	    //
	    strtrs_("Upper", "Transpose", "Non-unit", n, nrhs, &a[a_offset],
		    lda, &b[b_offset], ldb, info);
	    if (*info > 0) {
		return 0;
	    }
	    //
	    //          B(N+1:M,1:NRHS) = ZERO
	    //
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = *n + 1; i__ <= i__2; ++i__) {
		    b[i__ + j * b_dim1] = 0.f;
// L10:
		}
// L20:
	    }
	    //
	    //          B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS)
	    //
	    i__1 = *lwork - mn;
	    sormqr_("Left", "No transpose", m, nrhs, n, &a[a_offset], lda, &
		    work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);
	    //
	    //          workspace at least NRHS, optimally NRHS*NB
	    //
	    scllen = *m;
	}
    } else {
	//
	//       Compute LQ factorization of A
	//
	i__1 = *lwork - mn;
	sgelqf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
		;
	//
	//       workspace at least M, optimally M*NB.
	//
	if (! tpsd) {
	    //
	    //          underdetermined system of equations A * X = B
	    //
	    //          B(1:M,1:NRHS) := inv(L) * B(1:M,1:NRHS)
	    //
	    strtrs_("Lower", "No transpose", "Non-unit", m, nrhs, &a[a_offset]
		    , lda, &b[b_offset], ldb, info);
	    if (*info > 0) {
		return 0;
	    }
	    //
	    //          B(M+1:N,1:NRHS) = 0
	    //
	    i__1 = *nrhs;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = *m + 1; i__ <= i__2; ++i__) {
		    b[i__ + j * b_dim1] = 0.f;
// L30:
		}
// L40:
	    }
	    //
	    //          B(1:N,1:NRHS) := Q(1:N,:)**T * B(1:M,1:NRHS)
	    //
	    i__1 = *lwork - mn;
	    sormlq_("Left", "Transpose", n, nrhs, m, &a[a_offset], lda, &work[
		    1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);
	    //
	    //          workspace at least NRHS, optimally NRHS*NB
	    //
	    scllen = *n;
	} else {
	    //
	    //          overdetermined system min || A**T * X - B ||
	    //
	    //          B(1:N,1:NRHS) := Q * B(1:N,1:NRHS)
	    //
	    i__1 = *lwork - mn;
	    sormlq_("Left", "No transpose", n, nrhs, m, &a[a_offset], lda, &
		    work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);
	    //
	    //          workspace at least NRHS, optimally NRHS*NB
	    //
	    //          B(1:M,1:NRHS) := inv(L**T) * B(1:M,1:NRHS)
	    //
	    strtrs_("Lower", "Transpose", "Non-unit", m, nrhs, &a[a_offset],
		    lda, &b[b_offset], ldb, info);
	    if (*info > 0) {
		return 0;
	    }
	    scllen = *m;
	}
    }
    //
    //    Undo scaling
    //
    if (iascl == 1) {
	slascl_("G", &c__0, &c__0, &anrm, &smlnum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (iascl == 2) {
	slascl_("G", &c__0, &c__0, &anrm, &bignum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }
    if (ibscl == 1) {
	slascl_("G", &c__0, &c__0, &smlnum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (ibscl == 2) {
	slascl_("G", &c__0, &c__0, &bignum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }
L50:
    work[1] = (float) wsize;
    return 0;
    //
    //    End of SGELS
    //
} // sgels_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGEMV
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE SGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
//
//      .. Scalar Arguments ..
//      REAL ALPHA,BETA
//      INTEGER INCX,INCY,LDA,M,N
//      CHARACTER TRANS
//      ..
//      .. Array Arguments ..
//      REAL A(LDA,*),X(*),Y(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGEMV  performs one of the matrix-vector operations
//>
//>    y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
//>
//> where alpha and beta are scalars, x and y are vectors and A is an
//> m by n matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>           On entry, TRANS specifies the operation to be performed as
//>           follows:
//>
//>              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
//>
//>              TRANS = 'T' or 't'   y := alpha*A**T*x + beta*y.
//>
//>              TRANS = 'C' or 'c'   y := alpha*A**T*x + beta*y.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>           On entry, M specifies the number of rows of the matrix A.
//>           M must be at least zero.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry, N specifies the number of columns of the matrix A.
//>           N must be at least zero.
//> \endverbatim
//>
//> \param[in] ALPHA
//> \verbatim
//>          ALPHA is REAL
//>           On entry, ALPHA specifies the scalar alpha.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is REAL array, dimension ( LDA, N )
//>           Before entry, the leading m by n part of the array A must
//>           contain the matrix of coefficients.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in the calling (sub) program. LDA must be at least
//>           max( 1, m ).
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is REAL array, dimension at least
//>           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
//>           and at least
//>           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
//>           Before entry, the incremented array X must contain the
//>           vector x.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>           On entry, INCX specifies the increment for the elements of
//>           X. INCX must not be zero.
//> \endverbatim
//>
//> \param[in] BETA
//> \verbatim
//>          BETA is REAL
//>           On entry, BETA specifies the scalar beta. When BETA is
//>           supplied as zero then Y need not be set on input.
//> \endverbatim
//>
//> \param[in,out] Y
//> \verbatim
//>          Y is REAL array, dimension at least
//>           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
//>           and at least
//>           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
//>           Before entry with BETA non-zero, the incremented array Y
//>           must contain the vector y. On exit, Y is overwritten by the
//>           updated vector y.
//> \endverbatim
//>
//> \param[in] INCY
//> \verbatim
//>          INCY is INTEGER
//>           On entry, INCY specifies the increment for the elements of
//>           Y. INCY must not be zero.
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
//> \ingroup single_blas_level2
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Level 2 Blas routine.
//>  The vector and matrix arguments are not referenced when N = 0, or M = 0
//>
//>  -- Written on 22-October-1986.
//>     Jack Dongarra, Argonne National Lab.
//>     Jeremy Du Croz, Nag Central Office.
//>     Sven Hammarling, Nag Central Office.
//>     Richard Hanson, Sandia National Labs.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int sgemv_(char *trans, int *m, int *n, float *alpha, float *
	a, int *lda, float *x, int *incx, float *beta, float *y, int *incy)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, ix, iy, jx, jy, kx, ky, info;
    float temp;
    int lenx, leny;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, int *);

    //
    // -- Reference BLAS level2 routine (version 3.7.0) --
    // -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
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
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;
    --y;

    // Function Body
    info = 0;
    if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans, "C"))
	    {
	info = 1;
    } else if (*m < 0) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*lda < max(1,*m)) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    } else if (*incy == 0) {
	info = 11;
    }
    if (info != 0) {
	xerbla_("SGEMV ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*m == 0 || *n == 0 || *alpha == 0.f && *beta == 1.f) {
	return 0;
    }
    //
    //    Set  LENX  and  LENY, the lengths of the vectors x and y, and set
    //    up the start points in  X  and  Y.
    //
    if (lsame_(trans, "N")) {
	lenx = *n;
	leny = *m;
    } else {
	lenx = *m;
	leny = *n;
    }
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (lenx - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (leny - 1) * *incy;
    }
    //
    //    Start the operations. In this version the elements of A are
    //    accessed sequentially with one pass through A.
    //
    //    First form  y := beta*y.
    //
    if (*beta != 1.f) {
	if (*incy == 1) {
	    if (*beta == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.f;
// L10:
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
// L20:
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.f) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.f;
		    iy += *incy;
// L30:
		}
	    } else {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = *beta * y[iy];
		    iy += *incy;
// L40:
		}
	    }
	}
    }
    if (*alpha == 0.f) {
	return 0;
    }
    if (lsame_(trans, "N")) {
	//
	//       Form  y := alpha*A*x + y.
	//
	jx = kx;
	if (*incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = *alpha * x[jx];
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    y[i__] += temp * a[i__ + j * a_dim1];
// L50:
		}
		jx += *incx;
// L60:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = *alpha * x[jx];
		iy = ky;
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    y[iy] += temp * a[i__ + j * a_dim1];
		    iy += *incy;
// L70:
		}
		jx += *incx;
// L80:
	    }
	}
    } else {
	//
	//       Form  y := alpha*A**T*x + y.
	//
	jy = ky;
	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.f;
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp += a[i__ + j * a_dim1] * x[i__];
// L90:
		}
		y[jy] += *alpha * temp;
		jy += *incy;
// L100:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp = 0.f;
		ix = kx;
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp += a[i__ + j * a_dim1] * x[ix];
		    ix += *incx;
// L110:
		}
		y[jy] += *alpha * temp;
		jy += *incy;
// L120:
	    }
	}
    }
    return 0;
    //
    //    End of SGEMV .
    //
} // sgemv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGEQR2 computes the QR factorization of a general rectangular matrix using an unblocked algorithm.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGEQR2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgeqr2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgeqr2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgeqr2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGEQR2( M, N, A, LDA, TAU, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGEQR2 computes a QR factorization of a real m-by-n matrix A:
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
//>          A is REAL array, dimension (LDA,N)
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
//>          TAU is REAL array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is REAL array, dimension (N)
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
//> \ingroup realGEcomputational
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
/* Subroutine */ int sgeqr2_(int *m, int *n, float *a, int *lda, float *tau,
	float *work, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, k;
    float aii;
    extern /* Subroutine */ int slarf_(char *, int *, int *, float *, int *,
	    float *, float *, int *, float *), xerbla_(char *, int *),
	    slarfg_(int *, float *, float *, int *, float *);

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
	xerbla_("SGEQR2", &i__1);
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
	slarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3,*m) + i__ * a_dim1]
		, &c__1, &tau[i__]);
	if (i__ < *n) {
	    //
	    //          Apply H(i) to A(i:m,i+1:n) from the left
	    //
	    aii = a[i__ + i__ * a_dim1];
	    a[i__ + i__ * a_dim1] = 1.f;
	    i__2 = *m - i__ + 1;
	    i__3 = *n - i__;
	    slarf_("Left", &i__2, &i__3, &a[i__ + i__ * a_dim1], &c__1, &tau[
		    i__], &a[i__ + (i__ + 1) * a_dim1], lda, &work[1]);
	    a[i__ + i__ * a_dim1] = aii;
	}
// L10:
    }
    return 0;
    //
    //    End of SGEQR2
    //
} // sgeqr2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGEQRF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGEQRF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgeqrf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgeqrf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgeqrf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGEQRF( M, N, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGEQRF computes a QR factorization of a real M-by-N matrix A:
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
//>          A is REAL array, dimension (LDA,N)
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
//>          TAU is REAL array, dimension (min(M,N))
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is REAL array, dimension (MAX(1,LWORK))
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
//> \ingroup realGEcomputational
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
/* Subroutine */ int sgeqrf_(int *m, int *n, float *a, int *lda, float *tau,
	float *work, int *lwork, int *info)
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
    extern /* Subroutine */ int sgeqr2_(int *, int *, float *, int *, float *,
	     float *, int *), slarfb_(char *, char *, char *, char *, int *,
	    int *, int *, float *, int *, float *, int *, float *, int *,
	    float *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int slarft_(char *, char *, int *, int *, float *,
	     int *, float *, float *, int *);
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
    nb = ilaenv_(&c__1, "SGEQRF", " ", m, n, &c_n1, &c_n1);
    lwkopt = *n * nb;
    work[1] = (float) lwkopt;
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
	xerbla_("SGEQRF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    k = min(*m,*n);
    if (k == 0) {
	work[1] = 1.f;
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
	i__1 = 0, i__2 = ilaenv_(&c__3, "SGEQRF", " ", m, n, &c_n1, &c_n1);
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
		i__1 = 2, i__2 = ilaenv_(&c__2, "SGEQRF", " ", m, n, &c_n1, &
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
	    sgeqr2_(&i__3, &ib, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[
		    1], &iinfo);
	    if (i__ + ib <= *n) {
		//
		//             Form the triangular factor of the block reflector
		//             H = H(i) H(i+1) . . . H(i+ib-1)
		//
		i__3 = *m - i__ + 1;
		slarft_("Forward", "Columnwise", &i__3, &ib, &a[i__ + i__ *
			a_dim1], lda, &tau[i__], &work[1], &ldwork);
		//
		//             Apply H**T to A(i:m,i+ib:n) from the left
		//
		i__3 = *m - i__ + 1;
		i__4 = *n - i__ - ib + 1;
		slarfb_("Left", "Transpose", "Forward", "Columnwise", &i__3, &
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
	sgeqr2_(&i__2, &i__1, &a[i__ + i__ * a_dim1], lda, &tau[i__], &work[1]
		, &iinfo);
    }
    work[1] = (float) iws;
    return 0;
    //
    //    End of SGEQRF
    //
} // sgeqrf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGER
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE SGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA)
//
//      .. Scalar Arguments ..
//      REAL ALPHA
//      INTEGER INCX,INCY,LDA,M,N
//      ..
//      .. Array Arguments ..
//      REAL A(LDA,*),X(*),Y(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGER   performs the rank 1 operation
//>
//>    A := alpha*x*y**T + A,
//>
//> where alpha is a scalar, x is an m element vector, y is an n element
//> vector and A is an m by n matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>           On entry, M specifies the number of rows of the matrix A.
//>           M must be at least zero.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry, N specifies the number of columns of the matrix A.
//>           N must be at least zero.
//> \endverbatim
//>
//> \param[in] ALPHA
//> \verbatim
//>          ALPHA is REAL
//>           On entry, ALPHA specifies the scalar alpha.
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is REAL array, dimension at least
//>           ( 1 + ( m - 1 )*abs( INCX ) ).
//>           Before entry, the incremented array X must contain the m
//>           element vector x.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>           On entry, INCX specifies the increment for the elements of
//>           X. INCX must not be zero.
//> \endverbatim
//>
//> \param[in] Y
//> \verbatim
//>          Y is REAL array, dimension at least
//>           ( 1 + ( n - 1 )*abs( INCY ) ).
//>           Before entry, the incremented array Y must contain the n
//>           element vector y.
//> \endverbatim
//>
//> \param[in] INCY
//> \verbatim
//>          INCY is INTEGER
//>           On entry, INCY specifies the increment for the elements of
//>           Y. INCY must not be zero.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is REAL array, dimension ( LDA, N )
//>           Before entry, the leading m by n part of the array A must
//>           contain the matrix of coefficients. On exit, A is
//>           overwritten by the updated matrix.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in the calling (sub) program. LDA must be at least
//>           max( 1, m ).
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
//> \ingroup single_blas_level2
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Level 2 Blas routine.
//>
//>  -- Written on 22-October-1986.
//>     Jack Dongarra, Argonne National Lab.
//>     Jeremy Du Croz, Nag Central Office.
//>     Sven Hammarling, Nag Central Office.
//>     Richard Hanson, Sandia National Labs.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int sger_(int *m, int *n, float *alpha, float *x, int *incx,
	float *y, int *incy, float *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, ix, jy, kx, info;
    float temp;
    extern /* Subroutine */ int xerbla_(char *, int *);

    //
    // -- Reference BLAS level2 routine (version 3.7.0) --
    // -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
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
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    info = 0;
    if (*m < 0) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    } else if (*lda < max(1,*m)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("SGER  ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*m == 0 || *n == 0 || *alpha == 0.f) {
	return 0;
    }
    //
    //    Start the operations. In this version the elements of A are
    //    accessed sequentially with one pass through A.
    //
    if (*incy > 0) {
	jy = 1;
    } else {
	jy = 1 - (*n - 1) * *incy;
    }
    if (*incx == 1) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (y[jy] != 0.f) {
		temp = *alpha * y[jy];
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    a[i__ + j * a_dim1] += x[i__] * temp;
// L10:
		}
	    }
	    jy += *incy;
// L20:
	}
    } else {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*m - 1) * *incx;
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    if (y[jy] != 0.f) {
		temp = *alpha * y[jy];
		ix = kx;
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    a[i__ + j * a_dim1] += x[ix] * temp;
		    ix += *incx;
// L30:
		}
	    }
	    jy += *incy;
// L40:
	}
    }
    return 0;
    //
    //    End of SGER  .
    //
} // sger_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLABAD
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLABAD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slabad.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slabad.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slabad.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLABAD( SMALL, LARGE )
//
//      .. Scalar Arguments ..
//      REAL               LARGE, SMALL
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLABAD takes as input the values computed by SLAMCH for underflow and
//> overflow, and returns the square root of each of these values if the
//> log of LARGE is sufficiently large.  This subroutine is intended to
//> identify machines with a large exponent range, such as the Crays, and
//> redefine the underflow and overflow limits to be the square roots of
//> the values computed by SLAMCH.  This subroutine is needed because
//> SLAMCH does not compensate for poor arithmetic in the upper half of
//> the exponent range, as is found on a Cray.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in,out] SMALL
//> \verbatim
//>          SMALL is REAL
//>          On entry, the underflow threshold as computed by SLAMCH.
//>          On exit, if LOG10(LARGE) is sufficiently large, the square
//>          root of SMALL, otherwise unchanged.
//> \endverbatim
//>
//> \param[in,out] LARGE
//> \verbatim
//>          LARGE is REAL
//>          On entry, the overflow threshold as computed by SLAMCH.
//>          On exit, if LOG10(LARGE) is sufficiently large, the square
//>          root of LARGE, otherwise unchanged.
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
/* Subroutine */ int slabad_(float *small, float *large)
{
    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    If it looks like we're on a Cray, take the square root of
    //    SMALL and LARGE to avoid overflow and underflow problems.
    //
    if (r_lg10(large) > 2e3f) {
	*small = sqrt(*small);
	*large = sqrt(*large);
    }
    return 0;
    //
    //    End of SLABAD
    //
} // slabad_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLANGE returns the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any element of a general rectangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLANGE + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slange.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slange.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slange.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      REAL             FUNCTION SLANGE( NORM, M, N, A, LDA, WORK )
//
//      .. Scalar Arguments ..
//      CHARACTER          NORM
//      INTEGER            LDA, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLANGE  returns the value of the one norm,  or the Frobenius norm, or
//> the  infinity norm,  or the  element of  largest absolute value  of a
//> real matrix A.
//> \endverbatim
//>
//> \return SLANGE
//> \verbatim
//>
//>    SLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
//>          Specifies the value to be returned in SLANGE as described
//>          above.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.  M >= 0.  When M = 0,
//>          SLANGE is set to zero.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.  N >= 0.  When N = 0,
//>          SLANGE is set to zero.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          The m by n matrix A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(M,1).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is REAL array, dimension (MAX(1,LWORK)),
//>          where LWORK >= M when NORM = 'I'; otherwise, WORK is not
//>          referenced.
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
//> \ingroup realGEauxiliary
//
// =====================================================================
double slange_(char *norm, int *m, int *n, float *a, int *lda, float *work)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2;
    float ret_val, r__1;

    // Local variables
    extern /* Subroutine */ int scombssq_(float *, float *);
    int i__, j;
    float sum, ssq[2], temp;
    extern int lsame_(char *, char *);
    float value;
    extern int sisnan_(float *);
    float colssq[2];
    extern /* Subroutine */ int slassq_(int *, float *, int *, float *, float
	    *);

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
    //=====================================================================
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
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --work;

    // Function Body
    if (min(*m,*n) == 0) {
	value = 0.f;
    } else if (lsame_(norm, "M")) {
	//
	//       Find max(abs(A(i,j))).
	//
	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		temp = (r__1 = a[i__ + j * a_dim1], dabs(r__1));
		if (value < temp || sisnan_(&temp)) {
		    value = temp;
		}
// L10:
	    }
// L20:
	}
    } else if (lsame_(norm, "O") || *(unsigned char *)norm == '1') {
	//
	//       Find norm1(A).
	//
	value = 0.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = 0.f;
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		sum += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
// L30:
	    }
	    if (value < sum || sisnan_(&sum)) {
		value = sum;
	    }
// L40:
	}
    } else if (lsame_(norm, "I")) {
	//
	//       Find normI(A).
	//
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    work[i__] = 0.f;
// L50:
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[i__] += (r__1 = a[i__ + j * a_dim1], dabs(r__1));
// L60:
	    }
// L70:
	}
	value = 0.f;
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    temp = work[i__];
	    if (value < temp || sisnan_(&temp)) {
		value = temp;
	    }
// L80:
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {
	//
	//       Find normF(A).
	//       SSQ(1) is scale
	//       SSQ(2) is sum-of-squares
	//       For better accuracy, sum each column separately.
	//
	ssq[0] = 0.f;
	ssq[1] = 1.f;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    colssq[0] = 0.f;
	    colssq[1] = 1.f;
	    slassq_(m, &a[j * a_dim1 + 1], &c__1, colssq, &colssq[1]);
	    scombssq_(ssq, colssq);
// L90:
	}
	value = ssq[0] * sqrt(ssq[1]);
    }
    ret_val = value;
    return ret_val;
    //
    //    End of SLANGE
    //
} // slange_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLAPY2 returns sqrt(x2+y2).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLAPY2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slapy2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slapy2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slapy2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      REAL             FUNCTION SLAPY2( X, Y )
//
//      .. Scalar Arguments ..
//      REAL               X, Y
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary
//> overflow.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] X
//> \verbatim
//>          X is REAL
//> \endverbatim
//>
//> \param[in] Y
//> \verbatim
//>          Y is REAL
//>          X and Y specify the values x and y.
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
// =====================================================================
double slapy2_(float *x, float *y)
{
    // System generated locals
    float ret_val, r__1;

    // Local variables
    int x_is_nan__, y_is_nan__;
    float w, z__, xabs, yabs;
    extern int sisnan_(float *);

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    ..
    //    .. Executable Statements ..
    //
    x_is_nan__ = sisnan_(x);
    y_is_nan__ = sisnan_(y);
    if (x_is_nan__) {
	ret_val = *x;
    }
    if (y_is_nan__) {
	ret_val = *y;
    }
    if (! (x_is_nan__ || y_is_nan__)) {
	xabs = dabs(*x);
	yabs = dabs(*y);
	w = dmax(xabs,yabs);
	z__ = dmin(xabs,yabs);
	if (z__ == 0.f) {
	    ret_val = w;
	} else {
	    // Computing 2nd power
	    r__1 = z__ / w;
	    ret_val = w * sqrt(r__1 * r__1 + 1.f);
	}
    }
    return ret_val;
    //
    //    End of SLAPY2
    //
} // slapy2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLARF applies an elementary reflector to a general rectangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLARF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slarf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slarf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slarf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLARF( SIDE, M, N, V, INCV, TAU, C, LDC, WORK )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE
//      INTEGER            INCV, LDC, M, N
//      REAL               TAU
//      ..
//      .. Array Arguments ..
//      REAL               C( LDC, * ), V( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLARF applies a real elementary reflector H to a real m by n matrix
//> C, from either the left or the right. H is represented in the form
//>
//>       H = I - tau * v * v**T
//>
//> where tau is a real scalar and v is a real vector.
//>
//> If tau = 0, then H is taken to be the unit matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'L': form  H * C
//>          = 'R': form  C * H
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix C.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix C.
//> \endverbatim
//>
//> \param[in] V
//> \verbatim
//>          V is REAL array, dimension
//>                     (1 + (M-1)*abs(INCV)) if SIDE = 'L'
//>                  or (1 + (N-1)*abs(INCV)) if SIDE = 'R'
//>          The vector v in the representation of H. V is not used if
//>          TAU = 0.
//> \endverbatim
//>
//> \param[in] INCV
//> \verbatim
//>          INCV is INTEGER
//>          The increment between elements of v. INCV <> 0.
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is REAL
//>          The value tau in the representation of H.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension (LDC,N)
//>          On entry, the m by n matrix C.
//>          On exit, C is overwritten by the matrix H * C if SIDE = 'L',
//>          or C * H if SIDE = 'R'.
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
//>          WORK is REAL array, dimension
//>                         (N) if SIDE = 'L'
//>                      or (M) if SIDE = 'R'
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
//> \ingroup realOTHERauxiliary
//
// =====================================================================
/* Subroutine */ int slarf_(char *side, int *m, int *n, float *v, int *incv,
	float *tau, float *c__, int *ldc, float *work)
{
    // Table of constant values
    float c_b4 = 1.f;
    float c_b5 = 0.f;
    int c__1 = 1;

    // System generated locals
    int c_dim1, c_offset;
    float r__1;

    // Local variables
    int i__;
    int applyleft;
    extern /* Subroutine */ int sger_(int *, int *, float *, float *, int *,
	    float *, int *, float *, int *);
    extern int lsame_(char *, char *);
    int lastc;
    extern /* Subroutine */ int sgemv_(char *, int *, int *, float *, float *,
	     int *, float *, int *, float *, float *, int *);
    int lastv;
    extern int ilaslc_(int *, int *, float *, int *), ilaslr_(int *, int *,
	    float *, int *);

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
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    applyleft = lsame_(side, "L");
    lastv = 0;
    lastc = 0;
    if (*tau != 0.f) {
	//    Set up variables for scanning V.  LASTV begins pointing to the end
	//    of V.
	if (applyleft) {
	    lastv = *m;
	} else {
	    lastv = *n;
	}
	if (*incv > 0) {
	    i__ = (lastv - 1) * *incv + 1;
	} else {
	    i__ = 1;
	}
	//    Look for the last non-zero row in V.
	while(lastv > 0 && v[i__] == 0.f) {
	    --lastv;
	    i__ -= *incv;
	}
	if (applyleft) {
	    //    Scan for the last non-zero column in C(1:lastv,:).
	    lastc = ilaslc_(&lastv, n, &c__[c_offset], ldc);
	} else {
	    //    Scan for the last non-zero row in C(:,1:lastv).
	    lastc = ilaslr_(m, &lastv, &c__[c_offset], ldc);
	}
    }
    //    Note that lastc.eq.0 renders the BLAS operations null; no special
    //    case is needed at this level.
    if (applyleft) {
	//
	//       Form  H * C
	//
	if (lastv > 0) {
	    //
	    //          w(1:lastc,1) := C(1:lastv,1:lastc)**T * v(1:lastv,1)
	    //
	    sgemv_("Transpose", &lastv, &lastc, &c_b4, &c__[c_offset], ldc, &
		    v[1], incv, &c_b5, &work[1], &c__1);
	    //
	    //          C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**T
	    //
	    r__1 = -(*tau);
	    sger_(&lastv, &lastc, &r__1, &v[1], incv, &work[1], &c__1, &c__[
		    c_offset], ldc);
	}
    } else {
	//
	//       Form  C * H
	//
	if (lastv > 0) {
	    //
	    //          w(1:lastc,1) := C(1:lastc,1:lastv) * v(1:lastv,1)
	    //
	    sgemv_("No transpose", &lastc, &lastv, &c_b4, &c__[c_offset], ldc,
		     &v[1], incv, &c_b5, &work[1], &c__1);
	    //
	    //          C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**T
	    //
	    r__1 = -(*tau);
	    sger_(&lastc, &lastv, &r__1, &work[1], &c__1, &v[1], incv, &c__[
		    c_offset], ldc);
	}
    }
    return 0;
    //
    //    End of SLARF
    //
} // slarf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLARFB applies a block reflector or its transpose to a general rectangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLARFB + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slarfb.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slarfb.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slarfb.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLARFB( SIDE, TRANS, DIRECT, STOREV, M, N, K, V, LDV,
//                         T, LDT, C, LDC, WORK, LDWORK )
//
//      .. Scalar Arguments ..
//      CHARACTER          DIRECT, SIDE, STOREV, TRANS
//      INTEGER            K, LDC, LDT, LDV, LDWORK, M, N
//      ..
//      .. Array Arguments ..
//      REAL               C( LDC, * ), T( LDT, * ), V( LDV, * ),
//     $                   WORK( LDWORK, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLARFB applies a real block reflector H or its transpose H**T to a
//> real m by n matrix C, from either the left or the right.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'L': apply H or H**T from the Left
//>          = 'R': apply H or H**T from the Right
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          = 'N': apply H (No transpose)
//>          = 'T': apply H**T (Transpose)
//> \endverbatim
//>
//> \param[in] DIRECT
//> \verbatim
//>          DIRECT is CHARACTER*1
//>          Indicates how H is formed from a product of elementary
//>          reflectors
//>          = 'F': H = H(1) H(2) . . . H(k) (Forward)
//>          = 'B': H = H(k) . . . H(2) H(1) (Backward)
//> \endverbatim
//>
//> \param[in] STOREV
//> \verbatim
//>          STOREV is CHARACTER*1
//>          Indicates how the vectors which define the elementary
//>          reflectors are stored:
//>          = 'C': Columnwise
//>          = 'R': Rowwise
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix C.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix C.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The order of the matrix T (= the number of elementary
//>          reflectors whose product defines the block reflector).
//>          If SIDE = 'L', M >= K >= 0;
//>          if SIDE = 'R', N >= K >= 0.
//> \endverbatim
//>
//> \param[in] V
//> \verbatim
//>          V is REAL array, dimension
//>                                (LDV,K) if STOREV = 'C'
//>                                (LDV,M) if STOREV = 'R' and SIDE = 'L'
//>                                (LDV,N) if STOREV = 'R' and SIDE = 'R'
//>          The matrix V. See Further Details.
//> \endverbatim
//>
//> \param[in] LDV
//> \verbatim
//>          LDV is INTEGER
//>          The leading dimension of the array V.
//>          If STOREV = 'C' and SIDE = 'L', LDV >= max(1,M);
//>          if STOREV = 'C' and SIDE = 'R', LDV >= max(1,N);
//>          if STOREV = 'R', LDV >= K.
//> \endverbatim
//>
//> \param[in] T
//> \verbatim
//>          T is REAL array, dimension (LDT,K)
//>          The triangular k by k matrix T in the representation of the
//>          block reflector.
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of the array T. LDT >= K.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension (LDC,N)
//>          On entry, the m by n matrix C.
//>          On exit, C is overwritten by H*C or H**T*C or C*H or C*H**T.
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
//>          WORK is REAL array, dimension (LDWORK,K)
//> \endverbatim
//>
//> \param[in] LDWORK
//> \verbatim
//>          LDWORK is INTEGER
//>          The leading dimension of the array WORK.
//>          If SIDE = 'L', LDWORK >= max(1,N);
//>          if SIDE = 'R', LDWORK >= max(1,M).
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
//> \date June 2013
//
//> \ingroup realOTHERauxiliary
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The shape of the matrix V and the storage of the vectors which define
//>  the H(i) is best illustrated by the following example with n = 5 and
//>  k = 3. The elements equal to 1 are not stored; the corresponding
//>  array elements are modified but restored on exit. The rest of the
//>  array is not used.
//>
//>  DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':
//>
//>               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
//>                   ( v1  1    )                     (     1 v2 v2 v2 )
//>                   ( v1 v2  1 )                     (        1 v3 v3 )
//>                   ( v1 v2 v3 )
//>                   ( v1 v2 v3 )
//>
//>  DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':
//>
//>               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
//>                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
//>                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
//>                   (     1 v3 )
//>                   (        1 )
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int slarfb_(char *side, char *trans, char *direct, char *
	storev, int *m, int *n, int *k, float *v, int *ldv, float *t, int *
	ldt, float *c__, int *ldc, float *work, int *ldwork)
{
    // Table of constant values
    int c__1 = 1;
    float c_b14 = 1.f;
    float c_b25 = -1.f;

    // System generated locals
    int c_dim1, c_offset, t_dim1, t_offset, v_dim1, v_offset, work_dim1,
	    work_offset, i__1, i__2;

    // Local variables
    int i__, j;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, int *, int *, int *,
	    float *, float *, int *, float *, int *, float *, float *, int *),
	     scopy_(int *, float *, int *, float *, int *), strmm_(char *,
	    char *, char *, char *, int *, int *, float *, float *, int *,
	    float *, int *);
    char transt[1+1]={'\0'};

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2013
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
    //    .. Executable Statements ..
    //
    //    Quick return if possible
    //
    // Parameter adjustments
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    work_dim1 = *ldwork;
    work_offset = 1 + work_dim1;
    work -= work_offset;

    // Function Body
    if (*m <= 0 || *n <= 0) {
	return 0;
    }
    if (lsame_(trans, "N")) {
	*(unsigned char *)transt = 'T';
    } else {
	*(unsigned char *)transt = 'N';
    }
    if (lsame_(storev, "C")) {
	if (lsame_(direct, "F")) {
	    //
	    //          Let  V =  ( V1 )    (first K rows)
	    //                    ( V2 )
	    //          where  V1  is unit lower triangular.
	    //
	    if (lsame_(side, "L")) {
		//
		//             Form  H * C  or  H**T * C  where  C = ( C1 )
		//                                                   ( C2 )
		//
		//             W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
		//
		//             W := C1**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
// L10:
		}
		//
		//             W := W * V1
		//
		strmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b14,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {
		    //
		    //                W := W + C2**T * V2
		    //
		    i__1 = *m - *k;
		    sgemm_("Transpose", "No transpose", n, k, &i__1, &c_b14, &
			    c__[*k + 1 + c_dim1], ldc, &v[*k + 1 + v_dim1],
			    ldv, &c_b14, &work[work_offset], ldwork);
		}
		//
		//             W := W * T**T  or  W * T
		//
		strmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - V * W**T
		//
		if (*m > *k) {
		    //
		    //                C2 := C2 - V2 * W**T
		    //
		    i__1 = *m - *k;
		    sgemm_("No transpose", "Transpose", &i__1, n, k, &c_b25, &
			    v[*k + 1 + v_dim1], ldv, &work[work_offset],
			    ldwork, &c_b14, &c__[*k + 1 + c_dim1], ldc);
		}
		//
		//             W := W * V1**T
		//
		strmm_("Right", "Lower", "Transpose", "Unit", n, k, &c_b14, &
			v[v_offset], ldv, &work[work_offset], ldwork);
		//
		//             C1 := C1 - W**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[j + i__ * c_dim1] -= work[i__ + j * work_dim1];
// L20:
		    }
// L30:
		}
	    } else if (lsame_(side, "R")) {
		//
		//             Form  C * H  or  C * H**T  where  C = ( C1  C2 )
		//
		//             W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
		//
		//             W := C1
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
// L40:
		}
		//
		//             W := W * V1
		//
		strmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b14,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {
		    //
		    //                W := W + C2 * V2
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b14, &c__[(*k + 1) * c_dim1 + 1], ldc, &v[*k +
			    1 + v_dim1], ldv, &c_b14, &work[work_offset],
			    ldwork);
		}
		//
		//             W := W * T  or  W * T**T
		//
		strmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - W * V**T
		//
		if (*n > *k) {
		    //
		    //                C2 := C2 - W * V2**T
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, &i__1, k, &c_b25, &
			    work[work_offset], ldwork, &v[*k + 1 + v_dim1],
			    ldv, &c_b14, &c__[(*k + 1) * c_dim1 + 1], ldc);
		}
		//
		//             W := W * V1**T
		//
		strmm_("Right", "Lower", "Transpose", "Unit", m, k, &c_b14, &
			v[v_offset], ldv, &work[work_offset], ldwork);
		//
		//             C1 := C1 - W
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] -= work[i__ + j * work_dim1];
// L50:
		    }
// L60:
		}
	    }
	} else {
	    //
	    //          Let  V =  ( V1 )
	    //                    ( V2 )    (last K rows)
	    //          where  V2  is unit upper triangular.
	    //
	    if (lsame_(side, "L")) {
		//
		//             Form  H * C  or  H**T * C  where  C = ( C1 )
		//                                                   ( C2 )
		//
		//             W := C**T * V  =  (C1**T * V1 + C2**T * V2)  (stored in WORK)
		//
		//             W := C2**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
// L70:
		}
		//
		//             W := W * V2
		//
		strmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b14,
			 &v[*m - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*m > *k) {
		    //
		    //                W := W + C1**T * V1
		    //
		    i__1 = *m - *k;
		    sgemm_("Transpose", "No transpose", n, k, &i__1, &c_b14, &
			    c__[c_offset], ldc, &v[v_offset], ldv, &c_b14, &
			    work[work_offset], ldwork);
		}
		//
		//             W := W * T**T  or  W * T
		//
		strmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - V * W**T
		//
		if (*m > *k) {
		    //
		    //                C1 := C1 - V1 * W**T
		    //
		    i__1 = *m - *k;
		    sgemm_("No transpose", "Transpose", &i__1, n, k, &c_b25, &
			    v[v_offset], ldv, &work[work_offset], ldwork, &
			    c_b14, &c__[c_offset], ldc);
		}
		//
		//             W := W * V2**T
		//
		strmm_("Right", "Upper", "Transpose", "Unit", n, k, &c_b14, &
			v[*m - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		//
		//             C2 := C2 - W**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[*m - *k + j + i__ * c_dim1] -= work[i__ + j *
				work_dim1];
// L80:
		    }
// L90:
		}
	    } else if (lsame_(side, "R")) {
		//
		//             Form  C * H  or  C * H'  where  C = ( C1  C2 )
		//
		//             W := C * V  =  (C1*V1 + C2*V2)  (stored in WORK)
		//
		//             W := C2
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
// L100:
		}
		//
		//             W := W * V2
		//
		strmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b14,
			 &v[*n - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		if (*n > *k) {
		    //
		    //                W := W + C1 * V1
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, k, &i__1, &
			    c_b14, &c__[c_offset], ldc, &v[v_offset], ldv, &
			    c_b14, &work[work_offset], ldwork);
		}
		//
		//             W := W * T  or  W * T**T
		//
		strmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - W * V**T
		//
		if (*n > *k) {
		    //
		    //                C1 := C1 - W * V1**T
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, &i__1, k, &c_b25, &
			    work[work_offset], ldwork, &v[v_offset], ldv, &
			    c_b14, &c__[c_offset], ldc);
		}
		//
		//             W := W * V2**T
		//
		strmm_("Right", "Upper", "Transpose", "Unit", m, k, &c_b14, &
			v[*n - *k + 1 + v_dim1], ldv, &work[work_offset],
			ldwork);
		//
		//             C2 := C2 - W
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + (*n - *k + j) * c_dim1] -= work[i__ + j *
				work_dim1];
// L110:
		    }
// L120:
		}
	    }
	}
    } else if (lsame_(storev, "R")) {
	if (lsame_(direct, "F")) {
	    //
	    //          Let  V =  ( V1  V2 )    (V1: first K columns)
	    //          where  V1  is unit upper triangular.
	    //
	    if (lsame_(side, "L")) {
		//
		//             Form  H * C  or  H**T * C  where  C = ( C1 )
		//                                                   ( C2 )
		//
		//             W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
		//
		//             W := C1**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[j + c_dim1], ldc, &work[j * work_dim1 + 1],
			     &c__1);
// L130:
		}
		//
		//             W := W * V1**T
		//
		strmm_("Right", "Upper", "Transpose", "Unit", n, k, &c_b14, &
			v[v_offset], ldv, &work[work_offset], ldwork);
		if (*m > *k) {
		    //
		    //                W := W + C2**T * V2**T
		    //
		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", n, k, &i__1, &c_b14, &
			    c__[*k + 1 + c_dim1], ldc, &v[(*k + 1) * v_dim1 +
			    1], ldv, &c_b14, &work[work_offset], ldwork);
		}
		//
		//             W := W * T**T  or  W * T
		//
		strmm_("Right", "Upper", transt, "Non-unit", n, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - V**T * W**T
		//
		if (*m > *k) {
		    //
		    //                C2 := C2 - V2**T * W**T
		    //
		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", &i__1, n, k, &c_b25, &v[(
			    *k + 1) * v_dim1 + 1], ldv, &work[work_offset],
			    ldwork, &c_b14, &c__[*k + 1 + c_dim1], ldc);
		}
		//
		//             W := W * V1
		//
		strmm_("Right", "Upper", "No transpose", "Unit", n, k, &c_b14,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		//
		//             C1 := C1 - W**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[j + i__ * c_dim1] -= work[i__ + j * work_dim1];
// L140:
		    }
// L150:
		}
	    } else if (lsame_(side, "R")) {
		//
		//             Form  C * H  or  C * H**T  where  C = ( C1  C2 )
		//
		//             W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
		//
		//             W := C1
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[j * c_dim1 + 1], &c__1, &work[j *
			    work_dim1 + 1], &c__1);
// L160:
		}
		//
		//             W := W * V1**T
		//
		strmm_("Right", "Upper", "Transpose", "Unit", m, k, &c_b14, &
			v[v_offset], ldv, &work[work_offset], ldwork);
		if (*n > *k) {
		    //
		    //                W := W + C2 * V2**T
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, k, &i__1, &c_b14, &
			    c__[(*k + 1) * c_dim1 + 1], ldc, &v[(*k + 1) *
			    v_dim1 + 1], ldv, &c_b14, &work[work_offset],
			    ldwork);
		}
		//
		//             W := W * T  or  W * T**T
		//
		strmm_("Right", "Upper", trans, "Non-unit", m, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - W * V
		//
		if (*n > *k) {
		    //
		    //                C2 := C2 - W * V2
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, &i__1, k, &
			    c_b25, &work[work_offset], ldwork, &v[(*k + 1) *
			    v_dim1 + 1], ldv, &c_b14, &c__[(*k + 1) * c_dim1
			    + 1], ldc);
		}
		//
		//             W := W * V1
		//
		strmm_("Right", "Upper", "No transpose", "Unit", m, k, &c_b14,
			 &v[v_offset], ldv, &work[work_offset], ldwork);
		//
		//             C1 := C1 - W
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] -= work[i__ + j * work_dim1];
// L170:
		    }
// L180:
		}
	    }
	} else {
	    //
	    //          Let  V =  ( V1  V2 )    (V2: last K columns)
	    //          where  V2  is unit lower triangular.
	    //
	    if (lsame_(side, "L")) {
		//
		//             Form  H * C  or  H**T * C  where  C = ( C1 )
		//                                                   ( C2 )
		//
		//             W := C**T * V**T  =  (C1**T * V1**T + C2**T * V2**T) (stored in WORK)
		//
		//             W := C2**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(n, &c__[*m - *k + j + c_dim1], ldc, &work[j *
			    work_dim1 + 1], &c__1);
// L190:
		}
		//
		//             W := W * V2**T
		//
		strmm_("Right", "Lower", "Transpose", "Unit", n, k, &c_b14, &
			v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[work_offset]
			, ldwork);
		if (*m > *k) {
		    //
		    //                W := W + C1**T * V1**T
		    //
		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", n, k, &i__1, &c_b14, &
			    c__[c_offset], ldc, &v[v_offset], ldv, &c_b14, &
			    work[work_offset], ldwork);
		}
		//
		//             W := W * T**T  or  W * T
		//
		strmm_("Right", "Lower", transt, "Non-unit", n, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - V**T * W**T
		//
		if (*m > *k) {
		    //
		    //                C1 := C1 - V1**T * W**T
		    //
		    i__1 = *m - *k;
		    sgemm_("Transpose", "Transpose", &i__1, n, k, &c_b25, &v[
			    v_offset], ldv, &work[work_offset], ldwork, &
			    c_b14, &c__[c_offset], ldc);
		}
		//
		//             W := W * V2
		//
		strmm_("Right", "Lower", "No transpose", "Unit", n, k, &c_b14,
			 &v[(*m - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		//
		//             C2 := C2 - W**T
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[*m - *k + j + i__ * c_dim1] -= work[i__ + j *
				work_dim1];
// L200:
		    }
// L210:
		}
	    } else if (lsame_(side, "R")) {
		//
		//             Form  C * H  or  C * H**T  where  C = ( C1  C2 )
		//
		//             W := C * V**T  =  (C1*V1**T + C2*V2**T)  (stored in WORK)
		//
		//             W := C2
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    scopy_(m, &c__[(*n - *k + j) * c_dim1 + 1], &c__1, &work[
			    j * work_dim1 + 1], &c__1);
// L220:
		}
		//
		//             W := W * V2**T
		//
		strmm_("Right", "Lower", "Transpose", "Unit", m, k, &c_b14, &
			v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[work_offset]
			, ldwork);
		if (*n > *k) {
		    //
		    //                W := W + C1 * V1**T
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "Transpose", m, k, &i__1, &c_b14, &
			    c__[c_offset], ldc, &v[v_offset], ldv, &c_b14, &
			    work[work_offset], ldwork);
		}
		//
		//             W := W * T  or  W * T**T
		//
		strmm_("Right", "Lower", trans, "Non-unit", m, k, &c_b14, &t[
			t_offset], ldt, &work[work_offset], ldwork);
		//
		//             C := C - W * V
		//
		if (*n > *k) {
		    //
		    //                C1 := C1 - W * V1
		    //
		    i__1 = *n - *k;
		    sgemm_("No transpose", "No transpose", m, &i__1, k, &
			    c_b25, &work[work_offset], ldwork, &v[v_offset],
			    ldv, &c_b14, &c__[c_offset], ldc);
		}
		//
		//             W := W * V2
		//
		strmm_("Right", "Lower", "No transpose", "Unit", m, k, &c_b14,
			 &v[(*n - *k + 1) * v_dim1 + 1], ldv, &work[
			work_offset], ldwork);
		//
		//             C1 := C1 - W
		//
		i__1 = *k;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + (*n - *k + j) * c_dim1] -= work[i__ + j *
				work_dim1];
// L230:
		    }
// L240:
		}
	    }
	}
    }
    return 0;
    //
    //    End of SLARFB
    //
} // slarfb_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLARFG generates an elementary reflector (Householder matrix).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLARFG + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slarfg.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slarfg.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slarfg.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLARFG( N, ALPHA, X, INCX, TAU )
//
//      .. Scalar Arguments ..
//      INTEGER            INCX, N
//      REAL               ALPHA, TAU
//      ..
//      .. Array Arguments ..
//      REAL               X( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLARFG generates a real elementary reflector H of order n, such
//> that
//>
//>       H * ( alpha ) = ( beta ),   H**T * H = I.
//>           (   x   )   (   0  )
//>
//> where alpha and beta are scalars, and x is an (n-1)-element real
//> vector. H is represented in the form
//>
//>       H = I - tau * ( 1 ) * ( 1 v**T ) ,
//>                     ( v )
//>
//> where tau is a real scalar and v is a real (n-1)-element
//> vector.
//>
//> If the elements of x are all zero, then tau = 0 and H is taken to be
//> the unit matrix.
//>
//> Otherwise  1 <= tau <= 2.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the elementary reflector.
//> \endverbatim
//>
//> \param[in,out] ALPHA
//> \verbatim
//>          ALPHA is REAL
//>          On entry, the value alpha.
//>          On exit, it is overwritten with the value beta.
//> \endverbatim
//>
//> \param[in,out] X
//> \verbatim
//>          X is REAL array, dimension
//>                         (1+(N-2)*abs(INCX))
//>          On entry, the vector x.
//>          On exit, it is overwritten with the vector v.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>          The increment between elements of X. INCX > 0.
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is REAL
//>          The value tau.
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
//> \ingroup realOTHERauxiliary
//
// =====================================================================
/* Subroutine */ int slarfg_(int *n, float *alpha, float *x, int *incx, float
	*tau)
{
    // System generated locals
    int i__1;
    float r__1;

    // Local variables
    int j, knt;
    float beta;
    extern double snrm2_(int *, float *, int *);
    extern /* Subroutine */ int sscal_(int *, float *, float *, int *);
    float xnorm;
    extern double slapy2_(float *, float *), slamch_(char *);
    float safmin, rsafmn;

    //
    // -- LAPACK auxiliary routine (version 3.8.0) --
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
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --x;

    // Function Body
    if (*n <= 1) {
	*tau = 0.f;
	return 0;
    }
    i__1 = *n - 1;
    xnorm = snrm2_(&i__1, &x[1], incx);
    if (xnorm == 0.f) {
	//
	//       H  =  I
	//
	*tau = 0.f;
    } else {
	//
	//       general case
	//
	r__1 = slapy2_(alpha, &xnorm);
	beta = -r_sign(&r__1, alpha);
	safmin = slamch_("S") / slamch_("E");
	knt = 0;
	if (dabs(beta) < safmin) {
	    //
	    //          XNORM, BETA may be inaccurate; scale X and recompute them
	    //
	    rsafmn = 1.f / safmin;
L10:
	    ++knt;
	    i__1 = *n - 1;
	    sscal_(&i__1, &rsafmn, &x[1], incx);
	    beta *= rsafmn;
	    *alpha *= rsafmn;
	    if (dabs(beta) < safmin && knt < 20) {
		goto L10;
	    }
	    //
	    //          New BETA is at most 1, at least SAFMIN
	    //
	    i__1 = *n - 1;
	    xnorm = snrm2_(&i__1, &x[1], incx);
	    r__1 = slapy2_(alpha, &xnorm);
	    beta = -r_sign(&r__1, alpha);
	}
	*tau = (beta - *alpha) / beta;
	i__1 = *n - 1;
	r__1 = 1.f / (*alpha - beta);
	sscal_(&i__1, &r__1, &x[1], incx);
	//
	//       If ALPHA is subnormal, it may lose relative accuracy
	//
	i__1 = knt;
	for (j = 1; j <= i__1; ++j) {
	    beta *= safmin;
// L20:
	}
	*alpha = beta;
    }
    return 0;
    //
    //    End of SLARFG
    //
} // slarfg_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLARFT forms the triangular factor T of a block reflector H = I - vtvH
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLARFT + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slarft.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slarft.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slarft.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLARFT( DIRECT, STOREV, N, K, V, LDV, TAU, T, LDT )
//
//      .. Scalar Arguments ..
//      CHARACTER          DIRECT, STOREV
//      INTEGER            K, LDT, LDV, N
//      ..
//      .. Array Arguments ..
//      REAL               T( LDT, * ), TAU( * ), V( LDV, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLARFT forms the triangular factor T of a real block reflector H
//> of order n, which is defined as a product of k elementary reflectors.
//>
//> If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
//>
//> If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
//>
//> If STOREV = 'C', the vector which defines the elementary reflector
//> H(i) is stored in the i-th column of the array V, and
//>
//>    H  =  I - V * T * V**T
//>
//> If STOREV = 'R', the vector which defines the elementary reflector
//> H(i) is stored in the i-th row of the array V, and
//>
//>    H  =  I - V**T * T * V
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] DIRECT
//> \verbatim
//>          DIRECT is CHARACTER*1
//>          Specifies the order in which the elementary reflectors are
//>          multiplied to form the block reflector:
//>          = 'F': H = H(1) H(2) . . . H(k) (Forward)
//>          = 'B': H = H(k) . . . H(2) H(1) (Backward)
//> \endverbatim
//>
//> \param[in] STOREV
//> \verbatim
//>          STOREV is CHARACTER*1
//>          Specifies how the vectors which define the elementary
//>          reflectors are stored (see also Further Details):
//>          = 'C': columnwise
//>          = 'R': rowwise
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the block reflector H. N >= 0.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The order of the triangular factor T (= the number of
//>          elementary reflectors). K >= 1.
//> \endverbatim
//>
//> \param[in] V
//> \verbatim
//>          V is REAL array, dimension
//>                               (LDV,K) if STOREV = 'C'
//>                               (LDV,N) if STOREV = 'R'
//>          The matrix V. See further details.
//> \endverbatim
//>
//> \param[in] LDV
//> \verbatim
//>          LDV is INTEGER
//>          The leading dimension of the array V.
//>          If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K.
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is REAL array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i).
//> \endverbatim
//>
//> \param[out] T
//> \verbatim
//>          T is REAL array, dimension (LDT,K)
//>          The k by k triangular factor T of the block reflector.
//>          If DIRECT = 'F', T is upper triangular; if DIRECT = 'B', T is
//>          lower triangular. The rest of the array is not used.
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of the array T. LDT >= K.
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
//> \ingroup realOTHERauxiliary
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The shape of the matrix V and the storage of the vectors which define
//>  the H(i) is best illustrated by the following example with n = 5 and
//>  k = 3. The elements equal to 1 are not stored.
//>
//>  DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':
//>
//>               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
//>                   ( v1  1    )                     (     1 v2 v2 v2 )
//>                   ( v1 v2  1 )                     (        1 v3 v3 )
//>                   ( v1 v2 v3 )
//>                   ( v1 v2 v3 )
//>
//>  DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':
//>
//>               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
//>                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
//>                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
//>                   (     1 v3 )
//>                   (        1 )
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int slarft_(char *direct, char *storev, int *n, int *k,
	float *v, int *ldv, float *tau, float *t, int *ldt)
{
    // Table of constant values
    int c__1 = 1;
    float c_b7 = 1.f;

    // System generated locals
    int t_dim1, t_offset, v_dim1, v_offset, i__1, i__2, i__3;
    float r__1;

    // Local variables
    int i__, j, prevlastv;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int sgemv_(char *, int *, int *, float *, float *,
	     int *, float *, int *, float *, float *, int *);
    int lastv;
    extern /* Subroutine */ int strmv_(char *, char *, char *, int *, float *,
	     int *, float *, int *);

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
    //    .. External Subroutines ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Quick return if possible
    //
    // Parameter adjustments
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --tau;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;

    // Function Body
    if (*n == 0) {
	return 0;
    }
    if (lsame_(direct, "F")) {
	prevlastv = *n;
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    prevlastv = max(i__,prevlastv);
	    if (tau[i__] == 0.f) {
		//
		//             H(i)  =  I
		//
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    t[j + i__ * t_dim1] = 0.f;
		}
	    } else {
		//
		//             general case
		//
		if (lsame_(storev, "C")) {
		    //                Skip any trailing zeros.
		    i__2 = i__ + 1;
		    for (lastv = *n; lastv >= i__2; --lastv) {
			if (v[lastv + i__ * v_dim1] != 0.f) {
			    break;
			}
		    }
		    i__2 = i__ - 1;
		    for (j = 1; j <= i__2; ++j) {
			t[j + i__ * t_dim1] = -tau[i__] * v[i__ + j * v_dim1];
		    }
		    j = min(lastv,prevlastv);
		    //
		    //                T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)**T * V(i:j,i)
		    //
		    i__2 = j - i__;
		    i__3 = i__ - 1;
		    r__1 = -tau[i__];
		    sgemv_("Transpose", &i__2, &i__3, &r__1, &v[i__ + 1 +
			    v_dim1], ldv, &v[i__ + 1 + i__ * v_dim1], &c__1, &
			    c_b7, &t[i__ * t_dim1 + 1], &c__1);
		} else {
		    //                Skip any trailing zeros.
		    i__2 = i__ + 1;
		    for (lastv = *n; lastv >= i__2; --lastv) {
			if (v[i__ + lastv * v_dim1] != 0.f) {
			    break;
			}
		    }
		    i__2 = i__ - 1;
		    for (j = 1; j <= i__2; ++j) {
			t[j + i__ * t_dim1] = -tau[i__] * v[j + i__ * v_dim1];
		    }
		    j = min(lastv,prevlastv);
		    //
		    //                T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)**T
		    //
		    i__2 = i__ - 1;
		    i__3 = j - i__;
		    r__1 = -tau[i__];
		    sgemv_("No transpose", &i__2, &i__3, &r__1, &v[(i__ + 1) *
			     v_dim1 + 1], ldv, &v[i__ + (i__ + 1) * v_dim1],
			    ldv, &c_b7, &t[i__ * t_dim1 + 1], &c__1);
		}
		//
		//             T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
		//
		i__2 = i__ - 1;
		strmv_("Upper", "No transpose", "Non-unit", &i__2, &t[
			t_offset], ldt, &t[i__ * t_dim1 + 1], &c__1);
		t[i__ + i__ * t_dim1] = tau[i__];
		if (i__ > 1) {
		    prevlastv = max(prevlastv,lastv);
		} else {
		    prevlastv = lastv;
		}
	    }
	}
    } else {
	prevlastv = 1;
	for (i__ = *k; i__ >= 1; --i__) {
	    if (tau[i__] == 0.f) {
		//
		//             H(i)  =  I
		//
		i__1 = *k;
		for (j = i__; j <= i__1; ++j) {
		    t[j + i__ * t_dim1] = 0.f;
		}
	    } else {
		//
		//             general case
		//
		if (i__ < *k) {
		    if (lsame_(storev, "C")) {
			//                   Skip any leading zeros.
			i__1 = i__ - 1;
			for (lastv = 1; lastv <= i__1; ++lastv) {
			    if (v[lastv + i__ * v_dim1] != 0.f) {
				break;
			    }
			}
			i__1 = *k;
			for (j = i__ + 1; j <= i__1; ++j) {
			    t[j + i__ * t_dim1] = -tau[i__] * v[*n - *k + i__
				    + j * v_dim1];
			}
			j = max(lastv,prevlastv);
			//
			//                   T(i+1:k,i) = -tau(i) * V(j:n-k+i,i+1:k)**T * V(j:n-k+i,i)
			//
			i__1 = *n - *k + i__ - j;
			i__2 = *k - i__;
			r__1 = -tau[i__];
			sgemv_("Transpose", &i__1, &i__2, &r__1, &v[j + (i__
				+ 1) * v_dim1], ldv, &v[j + i__ * v_dim1], &
				c__1, &c_b7, &t[i__ + 1 + i__ * t_dim1], &
				c__1);
		    } else {
			//                   Skip any leading zeros.
			i__1 = i__ - 1;
			for (lastv = 1; lastv <= i__1; ++lastv) {
			    if (v[i__ + lastv * v_dim1] != 0.f) {
				break;
			    }
			}
			i__1 = *k;
			for (j = i__ + 1; j <= i__1; ++j) {
			    t[j + i__ * t_dim1] = -tau[i__] * v[j + (*n - *k
				    + i__) * v_dim1];
			}
			j = max(lastv,prevlastv);
			//
			//                   T(i+1:k,i) = -tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)**T
			//
			i__1 = *k - i__;
			i__2 = *n - *k + i__ - j;
			r__1 = -tau[i__];
			sgemv_("No transpose", &i__1, &i__2, &r__1, &v[i__ +
				1 + j * v_dim1], ldv, &v[i__ + j * v_dim1],
				ldv, &c_b7, &t[i__ + 1 + i__ * t_dim1], &c__1)
				;
		    }
		    //
		    //                T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i)
		    //
		    i__1 = *k - i__;
		    strmv_("Lower", "No transpose", "Non-unit", &i__1, &t[i__
			    + 1 + (i__ + 1) * t_dim1], ldt, &t[i__ + 1 + i__ *
			     t_dim1], &c__1);
		    if (i__ > 1) {
			prevlastv = min(prevlastv,lastv);
		    } else {
			prevlastv = lastv;
		    }
		}
		t[i__ + i__ * t_dim1] = tau[i__];
	    }
	}
    }
    return 0;
    //
    //    End of SLARFT
    //
} // slarft_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLASCL multiplies a general rectangular matrix by a real scalar defined as cto/cfrom.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLASCL + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slascl.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slascl.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slascl.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLASCL( TYPE, KL, KU, CFROM, CTO, M, N, A, LDA, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          TYPE
//      INTEGER            INFO, KL, KU, LDA, M, N
//      REAL               CFROM, CTO
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLASCL multiplies the M by N real matrix A by the real scalar
//> CTO/CFROM.  This is done without over/underflow as long as the final
//> result CTO*A(I,J)/CFROM does not over/underflow. TYPE specifies that
//> A may be full, upper triangular, lower triangular, upper Hessenberg,
//> or banded.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] TYPE
//> \verbatim
//>          TYPE is CHARACTER*1
//>          TYPE indices the storage type of the input matrix.
//>          = 'G':  A is a full matrix.
//>          = 'L':  A is a lower triangular matrix.
//>          = 'U':  A is an upper triangular matrix.
//>          = 'H':  A is an upper Hessenberg matrix.
//>          = 'B':  A is a symmetric band matrix with lower bandwidth KL
//>                  and upper bandwidth KU and with the only the lower
//>                  half stored.
//>          = 'Q':  A is a symmetric band matrix with lower bandwidth KL
//>                  and upper bandwidth KU and with the only the upper
//>                  half stored.
//>          = 'Z':  A is a band matrix with lower bandwidth KL and upper
//>                  bandwidth KU. See SGBTRF for storage details.
//> \endverbatim
//>
//> \param[in] KL
//> \verbatim
//>          KL is INTEGER
//>          The lower bandwidth of A.  Referenced only if TYPE = 'B',
//>          'Q' or 'Z'.
//> \endverbatim
//>
//> \param[in] KU
//> \verbatim
//>          KU is INTEGER
//>          The upper bandwidth of A.  Referenced only if TYPE = 'B',
//>          'Q' or 'Z'.
//> \endverbatim
//>
//> \param[in] CFROM
//> \verbatim
//>          CFROM is REAL
//> \endverbatim
//>
//> \param[in] CTO
//> \verbatim
//>          CTO is REAL
//>
//>          The matrix A is multiplied by CTO/CFROM. A(I,J) is computed
//>          without over/underflow if the final result CTO*A(I,J)/CFROM
//>          can be represented without over/underflow.  CFROM must be
//>          nonzero.
//> \endverbatim
//>
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
//>          A is REAL array, dimension (LDA,N)
//>          The matrix to be multiplied by CTO/CFROM.  See TYPE for the
//>          storage type.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//>          If TYPE = 'G', 'L', 'U', 'H', LDA >= max(1,M);
//>             TYPE = 'B', LDA >= KL+1;
//>             TYPE = 'Q', LDA >= KU+1;
//>             TYPE = 'Z', LDA >= 2*KL+KU+1.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          0  - successful exit
//>          <0 - if INFO = -i, the i-th argument had an illegal value.
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
// =====================================================================
/* Subroutine */ int slascl_(char *type__, int *kl, int *ku, float *cfrom,
	float *cto, int *m, int *n, float *a, int *lda, int *info)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    // Local variables
    int i__, j, k1, k2, k3, k4;
    float mul, cto1;
    int done;
    float ctoc;
    extern int lsame_(char *, char *);
    int itype;
    float cfrom1;
    extern double slamch_(char *);
    float cfromc;
    extern /* Subroutine */ int xerbla_(char *, int *);
    float bignum;
    extern int sisnan_(float *);
    float smlnum;

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
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    *info = 0;
    if (lsame_(type__, "G")) {
	itype = 0;
    } else if (lsame_(type__, "L")) {
	itype = 1;
    } else if (lsame_(type__, "U")) {
	itype = 2;
    } else if (lsame_(type__, "H")) {
	itype = 3;
    } else if (lsame_(type__, "B")) {
	itype = 4;
    } else if (lsame_(type__, "Q")) {
	itype = 5;
    } else if (lsame_(type__, "Z")) {
	itype = 6;
    } else {
	itype = -1;
    }
    if (itype == -1) {
	*info = -1;
    } else if (*cfrom == 0.f || sisnan_(cfrom)) {
	*info = -4;
    } else if (sisnan_(cto)) {
	*info = -5;
    } else if (*m < 0) {
	*info = -6;
    } else if (*n < 0 || itype == 4 && *n != *m || itype == 5 && *n != *m) {
	*info = -7;
    } else if (itype <= 3 && *lda < max(1,*m)) {
	*info = -9;
    } else if (itype >= 4) {
	// Computing MAX
	i__1 = *m - 1;
	if (*kl < 0 || *kl > max(i__1,0)) {
	    *info = -2;
	} else /* if(complicated condition) */ {
	    // Computing MAX
	    i__1 = *n - 1;
	    if (*ku < 0 || *ku > max(i__1,0) || (itype == 4 || itype == 5) &&
		    *kl != *ku) {
		*info = -3;
	    } else if (itype == 4 && *lda < *kl + 1 || itype == 5 && *lda < *
		    ku + 1 || itype == 6 && *lda < (*kl << 1) + *ku + 1) {
		*info = -9;
	    }
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SLASCL", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0 || *m == 0) {
	return 0;
    }
    //
    //    Get machine parameters
    //
    smlnum = slamch_("S");
    bignum = 1.f / smlnum;
    cfromc = *cfrom;
    ctoc = *cto;
L10:
    cfrom1 = cfromc * smlnum;
    if (cfrom1 == cfromc) {
	//       CFROMC is an inf.  Multiply by a correctly signed zero for
	//       finite CTOC, or a NaN if CTOC is infinite.
	mul = ctoc / cfromc;
	done = TRUE_;
	cto1 = ctoc;
    } else {
	cto1 = ctoc / bignum;
	if (cto1 == ctoc) {
	    //          CTOC is either 0 or an inf.  In both cases, CTOC itself
	    //          serves as the correct multiplication factor.
	    mul = ctoc;
	    done = TRUE_;
	    cfromc = 1.f;
	} else if (dabs(cfrom1) > dabs(ctoc) && ctoc != 0.f) {
	    mul = smlnum;
	    done = FALSE_;
	    cfromc = cfrom1;
	} else if (dabs(cto1) > dabs(cfromc)) {
	    mul = bignum;
	    done = FALSE_;
	    ctoc = cto1;
	} else {
	    mul = ctoc / cfromc;
	    done = TRUE_;
	}
    }
    if (itype == 0) {
	//
	//       Full matrix
	//
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
// L20:
	    }
// L30:
	}
    } else if (itype == 1) {
	//
	//       Lower triangular matrix
	//
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
// L40:
	    }
// L50:
	}
    } else if (itype == 2) {
	//
	//       Upper triangular matrix
	//
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = min(j,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
// L60:
	    }
// L70:
	}
    } else if (itype == 3) {
	//
	//       Upper Hessenberg matrix
	//
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    // Computing MIN
	    i__3 = j + 1;
	    i__2 = min(i__3,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
// L80:
	    }
// L90:
	}
    } else if (itype == 4) {
	//
	//       Lower half of a symmetric band matrix
	//
	k3 = *kl + 1;
	k4 = *n + 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    // Computing MIN
	    i__3 = k3, i__4 = k4 - j;
	    i__2 = min(i__3,i__4);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
// L100:
	    }
// L110:
	}
    } else if (itype == 5) {
	//
	//       Upper half of a symmetric band matrix
	//
	k1 = *ku + 2;
	k3 = *ku + 1;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    // Computing MAX
	    i__2 = k1 - j;
	    i__3 = k3;
	    for (i__ = max(i__2,1); i__ <= i__3; ++i__) {
		a[i__ + j * a_dim1] *= mul;
// L120:
	    }
// L130:
	}
    } else if (itype == 6) {
	//
	//       Band matrix
	//
	k1 = *kl + *ku + 2;
	k2 = *kl + 1;
	k3 = (*kl << 1) + *ku + 1;
	k4 = *kl + *ku + 1 + *m;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    // Computing MAX
	    i__3 = k1 - j;
	    // Computing MIN
	    i__4 = k3, i__5 = k4 - j;
	    i__2 = min(i__4,i__5);
	    for (i__ = max(i__3,k2); i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] *= mul;
// L140:
	    }
// L150:
	}
    }
    if (! done) {
	goto L10;
    }
    return 0;
    //
    //    End of SLASCL
    //
} // slascl_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLASET initializes the off-diagonal elements and the diagonal elements of a matrix to given values.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLASET + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slaset.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slaset.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slaset.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLASET( UPLO, M, N, ALPHA, BETA, A, LDA )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            LDA, M, N
//      REAL               ALPHA, BETA
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLASET initializes an m-by-n matrix A to BETA on the diagonal and
//> ALPHA on the offdiagonals.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          Specifies the part of the matrix A to be set.
//>          = 'U':      Upper triangular part is set; the strictly lower
//>                      triangular part of A is not changed.
//>          = 'L':      Lower triangular part is set; the strictly upper
//>                      triangular part of A is not changed.
//>          Otherwise:  All of the matrix A is set.
//> \endverbatim
//>
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
//> \param[in] ALPHA
//> \verbatim
//>          ALPHA is REAL
//>          The constant to which the offdiagonal elements are to be set.
//> \endverbatim
//>
//> \param[in] BETA
//> \verbatim
//>          BETA is REAL
//>          The constant to which the diagonal elements are to be set.
//> \endverbatim
//>
//> \param[out] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          On exit, the leading m-by-n submatrix of A is set as follows:
//>
//>          if UPLO = 'U', A(i,j) = ALPHA, 1<=i<=j-1, 1<=j<=n,
//>          if UPLO = 'L', A(i,j) = ALPHA, j+1<=i<=m, 1<=j<=n,
//>          otherwise,     A(i,j) = ALPHA, 1<=i<=m, 1<=j<=n, i.ne.j,
//>
//>          and, for all UPLO, A(i,i) = BETA, 1<=i<=min(m,n).
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
/* Subroutine */ int slaset_(char *uplo, int *m, int *n, float *alpha, float *
	beta, float *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, j;
    extern int lsame_(char *, char *);

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
    //=====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    if (lsame_(uplo, "U")) {
	//
	//       Set the strictly upper triangular or trapezoidal part of the
	//       array to ALPHA.
	//
	i__1 = *n;
	for (j = 2; j <= i__1; ++j) {
	    // Computing MIN
	    i__3 = j - 1;
	    i__2 = min(i__3,*m);
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = *alpha;
// L10:
	    }
// L20:
	}
    } else if (lsame_(uplo, "L")) {
	//
	//       Set the strictly lower triangular or trapezoidal part of the
	//       array to ALPHA.
	//
	i__1 = min(*m,*n);
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = j + 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = *alpha;
// L30:
	    }
// L40:
	}
    } else {
	//
	//       Set the leading m-by-n submatrix to ALPHA.
	//
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		a[i__ + j * a_dim1] = *alpha;
// L50:
	    }
// L60:
	}
    }
    //
    //    Set the first min(M,N) diagonal elements to BETA.
    //
    i__1 = min(*m,*n);
    for (i__ = 1; i__ <= i__1; ++i__) {
	a[i__ + i__ * a_dim1] = *beta;
// L70:
    }
    return 0;
    //
    //    End of SLASET
    //
} // slaset_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLASSQ updates a sum of squares represented in scaled form.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLASSQ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slassq.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slassq.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slassq.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLASSQ( N, X, INCX, SCALE, SUMSQ )
//
//      .. Scalar Arguments ..
//      INTEGER            INCX, N
//      REAL               SCALE, SUMSQ
//      ..
//      .. Array Arguments ..
//      REAL               X( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLASSQ  returns the values  scl  and  smsq  such that
//>
//>    ( scl**2 )*smsq = x( 1 )**2 +...+ x( n )**2 + ( scale**2 )*sumsq,
//>
//> where  x( i ) = X( 1 + ( i - 1 )*INCX ). The value of  sumsq  is
//> assumed to be non-negative and  scl  returns the value
//>
//>    scl = max( scale, abs( x( i ) ) ).
//>
//> scale and sumsq must be supplied in SCALE and SUMSQ and
//> scl and smsq are overwritten on SCALE and SUMSQ respectively.
//>
//> The routine makes only one pass through the vector x.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of elements to be used from the vector X.
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is REAL array, dimension (1+(N-1)*INCX)
//>          The vector for which a scaled sum of squares is computed.
//>             x( i )  = X( 1 + ( i - 1 )*INCX ), 1 <= i <= n.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>          The increment between successive values of the vector X.
//>          INCX > 0.
//> \endverbatim
//>
//> \param[in,out] SCALE
//> \verbatim
//>          SCALE is REAL
//>          On entry, the value  scale  in the equation above.
//>          On exit, SCALE is overwritten with  scl , the scaling factor
//>          for the sum of squares.
//> \endverbatim
//>
//> \param[in,out] SUMSQ
//> \verbatim
//>          SUMSQ is REAL
//>          On entry, the value  sumsq  in the equation above.
//>          On exit, SUMSQ is overwritten with  smsq , the basic sum of
//>          squares from which  scl  has been factored out.
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
/* Subroutine */ int slassq_(int *n, float *x, int *incx, float *scale, float
	*sumsq)
{
    // System generated locals
    int i__1, i__2;
    float r__1;

    // Local variables
    int ix;
    float absxi;
    extern int sisnan_(float *);

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
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --x;

    // Function Body
    if (*n > 0) {
	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    absxi = (r__1 = x[ix], dabs(r__1));
	    if (absxi > 0.f || sisnan_(&absxi)) {
		if (*scale < absxi) {
		    // Computing 2nd power
		    r__1 = *scale / absxi;
		    *sumsq = *sumsq * (r__1 * r__1) + 1;
		    *scale = absxi;
		} else {
		    // Computing 2nd power
		    r__1 = absxi / *scale;
		    *sumsq += r__1 * r__1;
		}
	    }
// L10:
	}
    }
    return 0;
    //
    //    End of SLASSQ
    //
} // slassq_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SNRM2
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      REAL FUNCTION SNRM2(N,X,INCX)
//
//      .. Scalar Arguments ..
//      INTEGER INCX,N
//      ..
//      .. Array Arguments ..
//      REAL X(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SNRM2 returns the euclidean norm of a vector via the function
//> name, so that
//>
//>    SNRM2 := sqrt( x'*x ).
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>         number of elements in input vector(s)
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is REAL array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>         storage spacing between elements of SX
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
//> \ingroup single_blas_level1
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  -- This version written on 25-October-1982.
//>     Modified on 14-October-1993 to inline the call to SLASSQ.
//>     Sven Hammarling, Nag Ltd.
//> \endverbatim
//>
// =====================================================================
double snrm2_(int *n, float *x, int *incx)
{
    // System generated locals
    int i__1, i__2;
    float ret_val, r__1;

    // Local variables
    int ix;
    float ssq, norm, scale, absxi;

    //
    // -- Reference BLAS level1 routine (version 3.8.0) --
    // -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
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
    //    .. Intrinsic Functions ..
    //    ..
    // Parameter adjustments
    --x;

    // Function Body
    if (*n < 1 || *incx < 1) {
	norm = 0.f;
    } else if (*n == 1) {
	norm = dabs(x[1]);
    } else {
	scale = 0.f;
	ssq = 1.f;
	//       The following loop is equivalent to this call to the LAPACK
	//       auxiliary routine:
	//       CALL SLASSQ( N, X, INCX, SCALE, SSQ )
	//
	i__1 = (*n - 1) * *incx + 1;
	i__2 = *incx;
	for (ix = 1; i__2 < 0 ? ix >= i__1 : ix <= i__1; ix += i__2) {
	    if (x[ix] != 0.f) {
		absxi = (r__1 = x[ix], dabs(r__1));
		if (scale < absxi) {
		    // Computing 2nd power
		    r__1 = scale / absxi;
		    ssq = ssq * (r__1 * r__1) + 1.f;
		    scale = absxi;
		} else {
		    // Computing 2nd power
		    r__1 = absxi / scale;
		    ssq += r__1 * r__1;
		}
	    }
// L10:
	}
	norm = scale * sqrt(ssq);
    }
    ret_val = norm;
    return ret_val;
    //
    //    End of SNRM2.
    //
} // snrm2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SORM2R multiplies a general matrix by the orthogonal matrix from a QR factorization determined by sgeqrf (unblocked algorithm).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SORM2R + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sorm2r.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sorm2r.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sorm2r.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SORM2R( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), C( LDC, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SORM2R overwrites the general real m by n matrix C with
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
//> as returned by SGEQRF. Q is of order m if SIDE = 'L' and of order n
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
//>          A is REAL array, dimension (LDA,K)
//>          The i-th column must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          SGEQRF in the first k columns of its array argument A.
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
//>          TAU is REAL array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by SGEQRF.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension (LDC,N)
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
//>          WORK is REAL array, dimension
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
//> \ingroup realOTHERcomputational
//
// =====================================================================
/* Subroutine */ int sorm2r_(char *side, char *trans, int *m, int *n, int *k,
	float *a, int *lda, float *tau, float *c__, int *ldc, float *work,
	int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;

    // Local variables
    int i__, i1, i2, i3, ic, jc, mi, ni, nq;
    float aii;
    int left;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int slarf_(char *, int *, int *, float *, int *,
	    float *, float *, int *, float *), xerbla_(char *, int *);
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
	xerbla_("SORM2R", &i__1);
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
	a[i__ + i__ * a_dim1] = 1.f;
	slarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], &c__1, &tau[i__], &c__[
		ic + jc * c_dim1], ldc, &work[1]);
	a[i__ + i__ * a_dim1] = aii;
// L10:
    }
    return 0;
    //
    //    End of SORM2R
    //
} // sorm2r_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SORML2 multiplies a general matrix by the orthogonal matrix from a LQ factorization determined by sgelqf (unblocked algorithm).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SORML2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sorml2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sorml2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sorml2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SORML2( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), C( LDC, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SORML2 overwrites the general real m by n matrix C with
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
//> as returned by SGELQF. Q is of order m if SIDE = 'L' and of order n
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
//>          A is REAL array, dimension
//>                               (LDA,M) if SIDE = 'L',
//>                               (LDA,N) if SIDE = 'R'
//>          The i-th row must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          SGELQF in the first k rows of its array argument A.
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
//>          TAU is REAL array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by SGELQF.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension (LDC,N)
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
//>          WORK is REAL array, dimension
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
//> \ingroup realOTHERcomputational
//
// =====================================================================
/* Subroutine */ int sorml2_(char *side, char *trans, int *m, int *n, int *k,
	float *a, int *lda, float *tau, float *c__, int *ldc, float *work,
	int *info)
{
    // System generated locals
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;

    // Local variables
    int i__, i1, i2, i3, ic, jc, mi, ni, nq;
    float aii;
    int left;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int slarf_(char *, int *, int *, float *, int *,
	    float *, float *, int *, float *), xerbla_(char *, int *);
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
	xerbla_("SORML2", &i__1);
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
	a[i__ + i__ * a_dim1] = 1.f;
	slarf_(side, &mi, &ni, &a[i__ + i__ * a_dim1], lda, &tau[i__], &c__[
		ic + jc * c_dim1], ldc, &work[1]);
	a[i__ + i__ * a_dim1] = aii;
// L10:
    }
    return 0;
    //
    //    End of SORML2
    //
} // sorml2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SORMLQ
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SORMLQ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sormlq.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sormlq.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sormlq.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SORMLQ( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), C( LDC, * ), TAU( * ),
//     $                   WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SORMLQ overwrites the general real M-by-N matrix C with
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
//> as returned by SGELQF. Q is of order M if SIDE = 'L' and of order N
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
//>          A is REAL array, dimension
//>                               (LDA,M) if SIDE = 'L',
//>                               (LDA,N) if SIDE = 'R'
//>          The i-th row must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          SGELQF in the first k rows of its array argument A.
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
//>          TAU is REAL array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by SGELQF.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension (LDC,N)
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
//>          WORK is REAL array, dimension (MAX(1,LWORK))
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
//> \ingroup realOTHERcomputational
//
// =====================================================================
/* Subroutine */ int sormlq_(char *side, char *trans, int *m, int *n, int *k,
	float *a, int *lda, float *tau, float *c__, int *ldc, float *work,
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
    extern /* Subroutine */ int sorml2_(char *, char *, int *, int *, int *,
	    float *, int *, float *, float *, int *, float *, int *), slarfb_(
	    char *, char *, char *, char *, int *, int *, int *, float *, int
	    *, float *, int *, float *, int *, float *, int *), xerbla_(char *
	    , int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int slarft_(char *, char *, int *, int *, float *,
	     int *, float *, float *, int *);
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "SORMLQ", ch__1, m, n, k, &c_n1);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb + 4160;
	work[1] = (float) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORMLQ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || *k == 0) {
	work[1] = 1.f;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "SORMLQ", ch__1, m, n, k, &c_n1);
	    nbmin = max(i__1,i__2);
	}
    }
    if (nb < nbmin || nb >= *k) {
	//
	//       Use unblocked code
	//
	sorml2_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
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
	    slarft_("Forward", "Rowwise", &i__4, &ib, &a[i__ + i__ * a_dim1],
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
	    slarfb_(side, transt, "Forward", "Rowwise", &mi, &ni, &ib, &a[i__
		    + i__ * a_dim1], lda, &work[iwt], &c__65, &c__[ic + jc *
		    c_dim1], ldc, &work[1], &ldwork);
// L10:
	}
    }
    work[1] = (float) lwkopt;
    return 0;
    //
    //    End of SORMLQ
    //
} // sormlq_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SORMQR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SORMQR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sormqr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sormqr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sormqr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SORMQR( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), C( LDC, * ), TAU( * ),
//     $                   WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SORMQR overwrites the general real M-by-N matrix C with
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
//> as returned by SGEQRF. Q is of order M if SIDE = 'L' and of order N
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
//>          A is REAL array, dimension (LDA,K)
//>          The i-th column must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          SGEQRF in the first k columns of its array argument A.
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
//>          TAU is REAL array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by SGEQRF.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension (LDC,N)
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
//>          WORK is REAL array, dimension (MAX(1,LWORK))
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
//> \ingroup realOTHERcomputational
//
// =====================================================================
/* Subroutine */ int sormqr_(char *side, char *trans, int *m, int *n, int *k,
	float *a, int *lda, float *tau, float *c__, int *ldc, float *work,
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
    extern /* Subroutine */ int sorm2r_(char *, char *, int *, int *, int *,
	    float *, int *, float *, float *, int *, float *, int *), slarfb_(
	    char *, char *, char *, char *, int *, int *, int *, float *, int
	    *, float *, int *, float *, int *, float *, int *), xerbla_(char *
	    , int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int slarft_(char *, char *, int *, int *, float *,
	     int *, float *, float *, int *);
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
	i__1 = 64, i__2 = ilaenv_(&c__1, "SORMQR", ch__1, m, n, k, &c_n1);
	nb = min(i__1,i__2);
	lwkopt = max(1,nw) * nb + 4160;
	work[1] = (float) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SORMQR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || *k == 0) {
	work[1] = 1.f;
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
	    i__1 = 2, i__2 = ilaenv_(&c__2, "SORMQR", ch__1, m, n, k, &c_n1);
	    nbmin = max(i__1,i__2);
	}
    }
    if (nb < nbmin || nb >= *k) {
	//
	//       Use unblocked code
	//
	sorm2r_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
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
	    slarft_("Forward", "Columnwise", &i__4, &ib, &a[i__ + i__ *
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
	    slarfb_(side, trans, "Forward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ + i__ * a_dim1], lda, &work[iwt], &c__65, &c__[ic +
		    jc * c_dim1], ldc, &work[1], &ldwork);
// L10:
	}
    }
    work[1] = (float) lwkopt;
    return 0;
    //
    //    End of SORMQR
    //
} // sormqr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b STRMM
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE STRMM(SIDE,UPLO,TRANSA,DIAG,M,N,ALPHA,A,LDA,B,LDB)
//
//      .. Scalar Arguments ..
//      REAL ALPHA
//      INTEGER LDA,LDB,M,N
//      CHARACTER DIAG,SIDE,TRANSA,UPLO
//      ..
//      .. Array Arguments ..
//      REAL A(LDA,*),B(LDB,*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> STRMM  performs one of the matrix-matrix operations
//>
//>    B := alpha*op( A )*B,   or   B := alpha*B*op( A ),
//>
//> where  alpha  is a scalar,  B  is an m by n matrix,  A  is a unit, or
//> non-unit,  upper or lower triangular matrix  and  op( A )  is one  of
//>
//>    op( A ) = A   or   op( A ) = A**T.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>           On entry,  SIDE specifies whether  op( A ) multiplies B from
//>           the left or right as follows:
//>
//>              SIDE = 'L' or 'l'   B := alpha*op( A )*B.
//>
//>              SIDE = 'R' or 'r'   B := alpha*B*op( A ).
//> \endverbatim
//>
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>           On entry, UPLO specifies whether the matrix A is an upper or
//>           lower triangular matrix as follows:
//>
//>              UPLO = 'U' or 'u'   A is an upper triangular matrix.
//>
//>              UPLO = 'L' or 'l'   A is a lower triangular matrix.
//> \endverbatim
//>
//> \param[in] TRANSA
//> \verbatim
//>          TRANSA is CHARACTER*1
//>           On entry, TRANSA specifies the form of op( A ) to be used in
//>           the matrix multiplication as follows:
//>
//>              TRANSA = 'N' or 'n'   op( A ) = A.
//>
//>              TRANSA = 'T' or 't'   op( A ) = A**T.
//>
//>              TRANSA = 'C' or 'c'   op( A ) = A**T.
//> \endverbatim
//>
//> \param[in] DIAG
//> \verbatim
//>          DIAG is CHARACTER*1
//>           On entry, DIAG specifies whether or not A is unit triangular
//>           as follows:
//>
//>              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
//>
//>              DIAG = 'N' or 'n'   A is not assumed to be unit
//>                                  triangular.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>           On entry, M specifies the number of rows of B. M must be at
//>           least zero.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry, N specifies the number of columns of B.  N must be
//>           at least zero.
//> \endverbatim
//>
//> \param[in] ALPHA
//> \verbatim
//>          ALPHA is REAL
//>           On entry,  ALPHA specifies the scalar  alpha. When  alpha is
//>           zero then  A is not referenced and  B need not be set before
//>           entry.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is REAL array, dimension ( LDA, k ), where k is m
//>           when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.
//>           Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k
//>           upper triangular part of the array  A must contain the upper
//>           triangular matrix  and the strictly lower triangular part of
//>           A is not referenced.
//>           Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k
//>           lower triangular part of the array  A must contain the lower
//>           triangular matrix  and the strictly upper triangular part of
//>           A is not referenced.
//>           Note that when  DIAG = 'U' or 'u',  the diagonal elements of
//>           A  are not referenced either,  but are assumed to be  unity.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in the calling (sub) program.  When  SIDE = 'L' or 'l'  then
//>           LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'
//>           then LDA must be at least max( 1, n ).
//> \endverbatim
//>
//> \param[in,out] B
//> \verbatim
//>          B is REAL array, dimension ( LDB, N )
//>           Before entry,  the leading  m by n part of the array  B must
//>           contain the matrix  B,  and  on exit  is overwritten  by the
//>           transformed matrix.
//> \endverbatim
//>
//> \param[in] LDB
//> \verbatim
//>          LDB is INTEGER
//>           On entry, LDB specifies the first dimension of B as declared
//>           in  the  calling  (sub)  program.   LDB  must  be  at  least
//>           max( 1, m ).
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
//> \ingroup single_blas_level3
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Level 3 Blas routine.
//>
//>  -- Written on 8-February-1989.
//>     Jack Dongarra, Argonne National Laboratory.
//>     Iain Duff, AERE Harwell.
//>     Jeremy Du Croz, Numerical Algorithms Group Ltd.
//>     Sven Hammarling, Numerical Algorithms Group Ltd.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int strmm_(char *side, char *uplo, char *transa, char *diag,
	int *m, int *n, float *alpha, float *a, int *lda, float *b, int *ldb)
{
    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    // Local variables
    int i__, j, k, info;
    float temp;
    int lside;
    extern int lsame_(char *, char *);
    int nrowa;
    int upper;
    extern /* Subroutine */ int xerbla_(char *, int *);
    int nounit;

    //
    // -- Reference BLAS level3 routine (version 3.7.0) --
    // -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
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
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Parameters ..
    //    ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    // Function Body
    lside = lsame_(side, "L");
    if (lside) {
	nrowa = *m;
    } else {
	nrowa = *n;
    }
    nounit = lsame_(diag, "N");
    upper = lsame_(uplo, "U");
    info = 0;
    if (! lside && ! lsame_(side, "R")) {
	info = 1;
    } else if (! upper && ! lsame_(uplo, "L")) {
	info = 2;
    } else if (! lsame_(transa, "N") && ! lsame_(transa, "T") && ! lsame_(
	    transa, "C")) {
	info = 3;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
	info = 4;
    } else if (*m < 0) {
	info = 5;
    } else if (*n < 0) {
	info = 6;
    } else if (*lda < max(1,nrowa)) {
	info = 9;
    } else if (*ldb < max(1,*m)) {
	info = 11;
    }
    if (info != 0) {
	xerbla_("STRMM ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*m == 0 || *n == 0) {
	return 0;
    }
    //
    //    And when  alpha.eq.zero.
    //
    if (*alpha == 0.f) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = 0.f;
// L10:
	    }
// L20:
	}
	return 0;
    }
    //
    //    Start the operations.
    //
    if (lside) {
	if (lsame_(transa, "N")) {
	    //
	    //          Form  B := alpha*A*B.
	    //
	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (k = 1; k <= i__2; ++k) {
			if (b[k + j * b_dim1] != 0.f) {
			    temp = *alpha * b[k + j * b_dim1];
			    i__3 = k - 1;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * a[i__ + k *
					a_dim1];
// L30:
			    }
			    if (nounit) {
				temp *= a[k + k * a_dim1];
			    }
			    b[k + j * b_dim1] = temp;
			}
// L40:
		    }
// L50:
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    for (k = *m; k >= 1; --k) {
			if (b[k + j * b_dim1] != 0.f) {
			    temp = *alpha * b[k + j * b_dim1];
			    b[k + j * b_dim1] = temp;
			    if (nounit) {
				b[k + j * b_dim1] *= a[k + k * a_dim1];
			    }
			    i__2 = *m;
			    for (i__ = k + 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * a[i__ + k *
					a_dim1];
// L60:
			    }
			}
// L70:
		    }
// L80:
		}
	    }
	} else {
	    //
	    //          Form  B := alpha*A**T*B.
	    //
	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    for (i__ = *m; i__ >= 1; --i__) {
			temp = b[i__ + j * b_dim1];
			if (nounit) {
			    temp *= a[i__ + i__ * a_dim1];
			}
			i__2 = i__ - 1;
			for (k = 1; k <= i__2; ++k) {
			    temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
// L90:
			}
			b[i__ + j * b_dim1] = *alpha * temp;
// L100:
		    }
// L110:
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp = b[i__ + j * b_dim1];
			if (nounit) {
			    temp *= a[i__ + i__ * a_dim1];
			}
			i__3 = *m;
			for (k = i__ + 1; k <= i__3; ++k) {
			    temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
// L120:
			}
			b[i__ + j * b_dim1] = *alpha * temp;
// L130:
		    }
// L140:
		}
	    }
	}
    } else {
	if (lsame_(transa, "N")) {
	    //
	    //          Form  B := alpha*B*A.
	    //
	    if (upper) {
		for (j = *n; j >= 1; --j) {
		    temp = *alpha;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__1 = *m;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
// L150:
		    }
		    i__1 = j - 1;
		    for (k = 1; k <= i__1; ++k) {
			if (a[k + j * a_dim1] != 0.f) {
			    temp = *alpha * a[k + j * a_dim1];
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
// L160:
			    }
			}
// L170:
		    }
// L180:
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = *alpha;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
// L190:
		    }
		    i__2 = *n;
		    for (k = j + 1; k <= i__2; ++k) {
			if (a[k + j * a_dim1] != 0.f) {
			    temp = *alpha * a[k + j * a_dim1];
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
// L200:
			    }
			}
// L210:
		    }
// L220:
		}
	    }
	} else {
	    //
	    //          Form  B := alpha*B*A**T.
	    //
	    if (upper) {
		i__1 = *n;
		for (k = 1; k <= i__1; ++k) {
		    i__2 = k - 1;
		    for (j = 1; j <= i__2; ++j) {
			if (a[j + k * a_dim1] != 0.f) {
			    temp = *alpha * a[j + k * a_dim1];
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
// L230:
			    }
			}
// L240:
		    }
		    temp = *alpha;
		    if (nounit) {
			temp *= a[k + k * a_dim1];
		    }
		    if (temp != 1.f) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
// L250:
			}
		    }
// L260:
		}
	    } else {
		for (k = *n; k >= 1; --k) {
		    i__1 = *n;
		    for (j = k + 1; j <= i__1; ++j) {
			if (a[j + k * a_dim1] != 0.f) {
			    temp = *alpha * a[j + k * a_dim1];
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k *
					b_dim1];
// L270:
			    }
			}
// L280:
		    }
		    temp = *alpha;
		    if (nounit) {
			temp *= a[k + k * a_dim1];
		    }
		    if (temp != 1.f) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
// L290:
			}
		    }
// L300:
		}
	    }
	}
    }
    return 0;
    //
    //    End of STRMM .
    //
} // strmm_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b STRMV
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE STRMV(UPLO,TRANS,DIAG,N,A,LDA,X,INCX)
//
//      .. Scalar Arguments ..
//      INTEGER INCX,LDA,N
//      CHARACTER DIAG,TRANS,UPLO
//      ..
//      .. Array Arguments ..
//      REAL A(LDA,*),X(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> STRMV  performs one of the matrix-vector operations
//>
//>    x := A*x,   or   x := A**T*x,
//>
//> where x is an n element vector and  A is an n by n unit, or non-unit,
//> upper or lower triangular matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>           On entry, UPLO specifies whether the matrix is an upper or
//>           lower triangular matrix as follows:
//>
//>              UPLO = 'U' or 'u'   A is an upper triangular matrix.
//>
//>              UPLO = 'L' or 'l'   A is a lower triangular matrix.
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>           On entry, TRANS specifies the operation to be performed as
//>           follows:
//>
//>              TRANS = 'N' or 'n'   x := A*x.
//>
//>              TRANS = 'T' or 't'   x := A**T*x.
//>
//>              TRANS = 'C' or 'c'   x := A**T*x.
//> \endverbatim
//>
//> \param[in] DIAG
//> \verbatim
//>          DIAG is CHARACTER*1
//>           On entry, DIAG specifies whether or not A is unit
//>           triangular as follows:
//>
//>              DIAG = 'U' or 'u'   A is assumed to be unit triangular.
//>
//>              DIAG = 'N' or 'n'   A is not assumed to be unit
//>                                  triangular.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry, N specifies the order of the matrix A.
//>           N must be at least zero.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is REAL array, dimension ( LDA, N )
//>           Before entry with  UPLO = 'U' or 'u', the leading n by n
//>           upper triangular part of the array A must contain the upper
//>           triangular matrix and the strictly lower triangular part of
//>           A is not referenced.
//>           Before entry with UPLO = 'L' or 'l', the leading n by n
//>           lower triangular part of the array A must contain the lower
//>           triangular matrix and the strictly upper triangular part of
//>           A is not referenced.
//>           Note that when  DIAG = 'U' or 'u', the diagonal elements of
//>           A are not referenced either, but are assumed to be unity.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in the calling (sub) program. LDA must be at least
//>           max( 1, n ).
//> \endverbatim
//>
//> \param[in,out] X
//> \verbatim
//>          X is REAL array, dimension at least
//>           ( 1 + ( n - 1 )*abs( INCX ) ).
//>           Before entry, the incremented array X must contain the n
//>           element vector x. On exit, X is overwritten with the
//>           transformed vector x.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>           On entry, INCX specifies the increment for the elements of
//>           X. INCX must not be zero.
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
//> \ingroup single_blas_level2
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Level 2 Blas routine.
//>  The vector and matrix arguments are not referenced when N = 0, or M = 0
//>
//>  -- Written on 22-October-1986.
//>     Jack Dongarra, Argonne National Lab.
//>     Jeremy Du Croz, Nag Central Office.
//>     Sven Hammarling, Nag Central Office.
//>     Richard Hanson, Sandia National Labs.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int strmv_(char *uplo, char *trans, char *diag, int *n,
	float *a, int *lda, float *x, int *incx)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, ix, jx, kx, info;
    float temp;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    int nounit;

    //
    // -- Reference BLAS level2 routine (version 3.7.0) --
    // -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
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
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --x;

    // Function Body
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans,
	     "C")) {
	info = 2;
    } else if (! lsame_(diag, "U") && ! lsame_(diag, "N")) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*lda < max(1,*n)) {
	info = 6;
    } else if (*incx == 0) {
	info = 8;
    }
    if (info != 0) {
	xerbla_("STRMV ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*n == 0) {
	return 0;
    }
    nounit = lsame_(diag, "N");
    //
    //    Set up the start point in X if the increment is not unity. This
    //    will be  ( N - 1 )*INCX  too small for descending loops.
    //
    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }
    //
    //    Start the operations. In this version the elements of A are
    //    accessed sequentially with one pass through A.
    //
    if (lsame_(trans, "N")) {
	//
	//       Form  x := A*x.
	//
	if (lsame_(uplo, "U")) {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[j] != 0.f) {
			temp = x[j];
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    x[i__] += temp * a[i__ + j * a_dim1];
// L10:
			}
			if (nounit) {
			    x[j] *= a[j + j * a_dim1];
			}
		    }
// L20:
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    if (x[jx] != 0.f) {
			temp = x[jx];
			ix = kx;
			i__2 = j - 1;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    x[ix] += temp * a[i__ + j * a_dim1];
			    ix += *incx;
// L30:
			}
			if (nounit) {
			    x[jx] *= a[j + j * a_dim1];
			}
		    }
		    jx += *incx;
// L40:
		}
	    }
	} else {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    if (x[j] != 0.f) {
			temp = x[j];
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    x[i__] += temp * a[i__ + j * a_dim1];
// L50:
			}
			if (nounit) {
			    x[j] *= a[j + j * a_dim1];
			}
		    }
// L60:
		}
	    } else {
		kx += (*n - 1) * *incx;
		jx = kx;
		for (j = *n; j >= 1; --j) {
		    if (x[jx] != 0.f) {
			temp = x[jx];
			ix = kx;
			i__1 = j + 1;
			for (i__ = *n; i__ >= i__1; --i__) {
			    x[ix] += temp * a[i__ + j * a_dim1];
			    ix -= *incx;
// L70:
			}
			if (nounit) {
			    x[jx] *= a[j + j * a_dim1];
			}
		    }
		    jx -= *incx;
// L80:
		}
	    }
	}
    } else {
	//
	//       Form  x := A**T*x.
	//
	if (lsame_(uplo, "U")) {
	    if (*incx == 1) {
		for (j = *n; j >= 1; --j) {
		    temp = x[j];
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    for (i__ = j - 1; i__ >= 1; --i__) {
			temp += a[i__ + j * a_dim1] * x[i__];
// L90:
		    }
		    x[j] = temp;
// L100:
		}
	    } else {
		jx = kx + (*n - 1) * *incx;
		for (j = *n; j >= 1; --j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    for (i__ = j - 1; i__ >= 1; --i__) {
			ix -= *incx;
			temp += a[i__ + j * a_dim1] * x[ix];
// L110:
		    }
		    x[jx] = temp;
		    jx -= *incx;
// L120:
		}
	    }
	} else {
	    if (*incx == 1) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[j];
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			temp += a[i__ + j * a_dim1] * x[i__];
// L130:
		    }
		    x[j] = temp;
// L140:
		}
	    } else {
		jx = kx;
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = x[jx];
		    ix = jx;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__2 = *n;
		    for (i__ = j + 1; i__ <= i__2; ++i__) {
			ix += *incx;
			temp += a[i__ + j * a_dim1] * x[ix];
// L150:
		    }
		    x[jx] = temp;
		    jx += *incx;
// L160:
		}
	    }
	}
    }
    return 0;
    //
    //    End of STRMV .
    //
} // strmv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b STRTRS
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download STRTRS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/strtrs.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/strtrs.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/strtrs.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE STRTRS( UPLO, TRANS, DIAG, N, NRHS, A, LDA, B, LDB,
//                         INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          DIAG, TRANS, UPLO
//      INTEGER            INFO, LDA, LDB, N, NRHS
//      ..
//      .. Array Arguments ..
//      REAL               A( LDA, * ), B( LDB, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> STRTRS solves a triangular system of the form
//>
//>    A * X = B  or  A**T * X = B,
//>
//> where A is a triangular matrix of order N, and B is an N-by-NRHS
//> matrix.  A check is made to verify that A is nonsingular.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U':  A is upper triangular;
//>          = 'L':  A is lower triangular.
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          Specifies the form of the system of equations:
//>          = 'N':  A * X = B  (No transpose)
//>          = 'T':  A**T * X = B  (Transpose)
//>          = 'C':  A**H * X = B  (Conjugate transpose = Transpose)
//> \endverbatim
//>
//> \param[in] DIAG
//> \verbatim
//>          DIAG is CHARACTER*1
//>          = 'N':  A is non-unit triangular;
//>          = 'U':  A is unit triangular.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in] NRHS
//> \verbatim
//>          NRHS is INTEGER
//>          The number of right hand sides, i.e., the number of columns
//>          of the matrix B.  NRHS >= 0.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          The triangular matrix A.  If UPLO = 'U', the leading N-by-N
//>          upper triangular part of the array A contains the upper
//>          triangular matrix, and the strictly lower triangular part of
//>          A is not referenced.  If UPLO = 'L', the leading N-by-N lower
//>          triangular part of the array A contains the lower triangular
//>          matrix, and the strictly upper triangular part of A is not
//>          referenced.  If DIAG = 'U', the diagonal elements of A are
//>          also not referenced and are assumed to be 1.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in,out] B
//> \verbatim
//>          B is REAL array, dimension (LDB,NRHS)
//>          On entry, the right hand side matrix B.
//>          On exit, if INFO = 0, the solution matrix X.
//> \endverbatim
//>
//> \param[in] LDB
//> \verbatim
//>          LDB is INTEGER
//>          The leading dimension of the array B.  LDB >= max(1,N).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0: if INFO = -i, the i-th argument had an illegal value
//>          > 0: if INFO = i, the i-th diagonal element of A is zero,
//>               indicating that the matrix is singular and the solutions
//>               X have not been computed.
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
//> \ingroup realOTHERcomputational
//
// =====================================================================
/* Subroutine */ int strtrs_(char *uplo, char *trans, char *diag, int *n, int
	*nrhs, float *a, int *lda, float *b, int *ldb, int *info)
{
    // Table of constant values
    float c_b12 = 1.f;

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *, int *,
	    int *, float *, float *, int *, float *, int *), xerbla_(char *,
	    int *);
    int nounit;

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
    //    Test the input parameters.
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    // Function Body
    *info = 0;
    nounit = lsame_(diag, "N");
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans,
	     "C")) {
	*info = -2;
    } else if (! nounit && ! lsame_(diag, "U")) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*nrhs < 0) {
	*info = -5;
    } else if (*lda < max(1,*n)) {
	*info = -7;
    } else if (*ldb < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("STRTRS", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	return 0;
    }
    //
    //    Check for singularity.
    //
    if (nounit) {
	i__1 = *n;
	for (*info = 1; *info <= i__1; ++(*info)) {
	    if (a[*info + *info * a_dim1] == 0.f) {
		return 0;
	    }
// L10:
	}
    }
    *info = 0;
    //
    //    Solve A * x = b  or  A**T * x = b.
    //
    strsm_("Left", uplo, trans, diag, n, nrhs, &c_b12, &a[a_offset], lda, &b[
	    b_offset], ldb);
    return 0;
    //
    //    End of STRTRS
    //
} // strtrs_

