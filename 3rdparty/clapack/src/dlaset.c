/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLASET initializes the off-diagonal elements and the diagonal elements of a matrix to given values.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASET + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaset.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaset.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaset.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASET( UPLO, M, N, ALPHA, BETA, A, LDA )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            LDA, M, N
//      DOUBLE PRECISION   ALPHA, BETA
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASET initializes an m-by-n matrix A to BETA on the diagonal and
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
//>          ALPHA is DOUBLE PRECISION
//>          The constant to which the offdiagonal elements are to be set.
//> \endverbatim
//>
//> \param[in] BETA
//> \verbatim
//>          BETA is DOUBLE PRECISION
//>          The constant to which the diagonal elements are to be set.
//> \endverbatim
//>
//> \param[out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
/* Subroutine */ int dlaset_(char *uplo, int *m, int *n, double *alpha,
	double *beta, double *a, int *lda)
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
    //    End of DLASET
    //
} // dlaset_

