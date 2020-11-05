/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DCOMBSSQ adds two scaled sum of squares quantities.
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
//      SUBROUTINE DCOMBSSQ( V1, V2 )
//
//      .. Array Arguments ..
//      DOUBLE PRECISION   V1( 2 ), V2( 2 )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DCOMBSSQ adds two scaled sum of squares quantities, V1 := V1 + V2.
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
//>          V1 is DOUBLE PRECISION array, dimension (2).
//>          The first scaled sum.
//>          V1(1) = V1_scale, V1(2) = V1_sumsq.
//> \endverbatim
//>
//> \param[in] V2
//> \verbatim
//>          V2 is DOUBLE PRECISION array, dimension (2).
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
/* Subroutine */ int dcombssq_(double *v1, double *v2)
{
    // System generated locals
    double d__1;

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
	if (v1[1] != 0.) {
	    // Computing 2nd power
	    d__1 = v2[1] / v1[1];
	    v1[2] += d__1 * d__1 * v2[2];
	}
    } else {
	// Computing 2nd power
	d__1 = v1[1] / v2[1];
	v1[2] = v2[2] + d__1 * d__1 * v1[2];
	v1[1] = v2[1];
    }
    return 0;
    //
    //    End of DCOMBSSQ
    //
} // dcombssq_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLANGE returns the value of the 1-norm, Frobenius norm, infinity-norm, or the largest absolute value of any element of a general rectangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLANGE + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlange.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlange.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlange.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      DOUBLE PRECISION FUNCTION DLANGE( NORM, M, N, A, LDA, WORK )
//
//      .. Scalar Arguments ..
//      CHARACTER          NORM
//      INTEGER            LDA, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLANGE  returns the value of the one norm,  or the Frobenius norm, or
//> the  infinity norm,  or the  element of  largest absolute value  of a
//> real matrix A.
//> \endverbatim
//>
//> \return DLANGE
//> \verbatim
//>
//>    DLANGE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
//>          Specifies the value to be returned in DLANGE as described
//>          above.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix A.  M >= 0.  When M = 0,
//>          DLANGE is set to zero.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.  N >= 0.  When N = 0,
//>          DLANGE is set to zero.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK)),
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
//> \ingroup doubleGEauxiliary
//
// =====================================================================
double dlange_(char *norm, int *m, int *n, double *a, int *lda, double *work)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2;
    double ret_val, d__1;

    // Local variables
    extern /* Subroutine */ int dcombssq_(double *, double *);
    int i__, j;
    double sum, ssq[2], temp;
    extern int lsame_(char *, char *);
    double value;
    extern int disnan_(double *);
    extern /* Subroutine */ int dlassq_(int *, double *, int *, double *,
	    double *);
    double colssq[2];

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
	value = 0.;
    } else if (lsame_(norm, "M")) {
	//
	//       Find max(abs(A(i,j))).
	//
	value = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		temp = (d__1 = a[i__ + j * a_dim1], abs(d__1));
		if (value < temp || disnan_(&temp)) {
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
	value = 0.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = 0.;
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		sum += (d__1 = a[i__ + j * a_dim1], abs(d__1));
// L30:
	    }
	    if (value < sum || disnan_(&sum)) {
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
	    work[i__] = 0.;
// L50:
	}
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[i__] += (d__1 = a[i__ + j * a_dim1], abs(d__1));
// L60:
	    }
// L70:
	}
	value = 0.;
	i__1 = *m;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    temp = work[i__];
	    if (value < temp || disnan_(&temp)) {
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
	ssq[0] = 0.;
	ssq[1] = 1.;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    colssq[0] = 0.;
	    colssq[1] = 1.;
	    dlassq_(m, &a[j * a_dim1 + 1], &c__1, colssq, &colssq[1]);
	    dcombssq_(ssq, colssq);
// L90:
	}
	value = ssq[0] * sqrt(ssq[1]);
    }
    ret_val = value;
    return ret_val;
    //
    //    End of DLANGE
    //
} // dlange_

