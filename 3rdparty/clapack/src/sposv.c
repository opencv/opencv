/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief <b> SPOSV computes the solution to system of linear equations A * X = B for PO matrices</b>
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SPOSV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sposv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sposv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sposv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SPOSV( UPLO, N, NRHS, A, LDA, B, LDB, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
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
//> SPOSV computes the solution to a real system of linear equations
//>    A * X = B,
//> where A is an N-by-N symmetric positive definite matrix and X and B
//> are N-by-NRHS matrices.
//>
//> The Cholesky decomposition is used to factor A as
//>    A = U**T* U,  if UPLO = 'U', or
//>    A = L * L**T,  if UPLO = 'L',
//> where U is an upper triangular matrix and L is a lower triangular
//> matrix.  The factored form of A is then used to solve the system of
//> equations A * X = B.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U':  Upper triangle of A is stored;
//>          = 'L':  Lower triangle of A is stored.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of linear equations, i.e., the order of the
//>          matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in] NRHS
//> \verbatim
//>          NRHS is INTEGER
//>          The number of right hand sides, i.e., the number of columns
//>          of the matrix B.  NRHS >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
//>          N-by-N upper triangular part of A contains the upper
//>          triangular part of the matrix A, and the strictly lower
//>          triangular part of A is not referenced.  If UPLO = 'L', the
//>          leading N-by-N lower triangular part of A contains the lower
//>          triangular part of the matrix A, and the strictly upper
//>          triangular part of A is not referenced.
//>
//>          On exit, if INFO = 0, the factor U or L from the Cholesky
//>          factorization A = U**T*U or A = L*L**T.
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
//>          On entry, the N-by-NRHS right hand side matrix B.
//>          On exit, if INFO = 0, the N-by-NRHS solution matrix X.
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
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  if INFO = i, the leading minor of order i of A is not
//>                positive definite, so the factorization could not be
//>                completed, and the solution has not been computed.
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
//> \ingroup realPOsolve
//
// =====================================================================
/* Subroutine */ int sposv_(char *uplo, int *n, int *nrhs, float *a, int *lda,
	 float *b, int *ldb, int *info)
{
    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, int *), spotrf_(char *, int *,
	     float *, int *, int *), spotrs_(char *, int *, int *, float *,
	    int *, float *, int *, int *);

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
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOSV ", &i__1);
	return 0;
    }
    //
    //    Compute the Cholesky factorization A = U**T*U or A = L*L**T.
    //
    spotrf_(uplo, n, &a[a_offset], lda, info);
    if (*info == 0) {
	//
	//       Solve the system A*X = B, overwriting B with X.
	//
	spotrs_(uplo, n, nrhs, &a[a_offset], lda, &b[b_offset], ldb, info);
    }
    return 0;
    //
    //    End of SPOSV
    //
} // sposv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SPOTRF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SPOTRF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/spotrf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/spotrf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/spotrf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SPOTRF( UPLO, N, A, LDA, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDA, N
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
//> SPOTRF computes the Cholesky factorization of a real symmetric
//> positive definite matrix A.
//>
//> The factorization has the form
//>    A = U**T * U,  if UPLO = 'U', or
//>    A = L  * L**T,  if UPLO = 'L',
//> where U is an upper triangular matrix and L is lower triangular.
//>
//> This is the block version of the algorithm, calling Level 3 BLAS.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U':  Upper triangle of A is stored;
//>          = 'L':  Lower triangle of A is stored.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
//>          N-by-N upper triangular part of A contains the upper
//>          triangular part of the matrix A, and the strictly lower
//>          triangular part of A is not referenced.  If UPLO = 'L', the
//>          leading N-by-N lower triangular part of A contains the lower
//>          triangular part of the matrix A, and the strictly upper
//>          triangular part of A is not referenced.
//>
//>          On exit, if INFO = 0, the factor U or L from the Cholesky
//>          factorization A = U**T*U or A = L*L**T.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  if INFO = i, the leading minor of order i is not
//>                positive definite, and the factorization could not be
//>                completed.
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
//> \ingroup realPOcomputational
//
// =====================================================================
/* Subroutine */ int spotrf_(char *uplo, int *n, float *a, int *lda, int *
	info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    float c_b13 = -1.f;
    float c_b14 = 1.f;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    // Local variables
    int j, jb, nb;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int sgemm_(char *, char *, int *, int *, int *,
	    float *, float *, int *, float *, int *, float *, float *, int *);
    int upper;
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *, int *,
	    int *, float *, float *, int *, float *, int *), ssyrk_(char *,
	    char *, int *, int *, float *, float *, int *, float *, float *,
	    int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int spotrf2_(char *, int *, float *, int *, int *)
	    ;

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

    // Function Body
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTRF", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	return 0;
    }
    //
    //    Determine the block size for this environment.
    //
    nb = ilaenv_(&c__1, "SPOTRF", uplo, n, &c_n1, &c_n1, &c_n1);
    if (nb <= 1 || nb >= *n) {
	//
	//       Use unblocked code.
	//
	spotrf2_(uplo, n, &a[a_offset], lda, info);
    } else {
	//
	//       Use blocked code.
	//
	if (upper) {
	    //
	    //          Compute the Cholesky factorization A = U**T*U.
	    //
	    i__1 = *n;
	    i__2 = nb;
	    for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
		//
		//             Update and factorize the current diagonal block and test
		//             for non-positive-definiteness.
		//
		// Computing MIN
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
		ssyrk_("Upper", "Transpose", &jb, &i__3, &c_b13, &a[j *
			a_dim1 + 1], lda, &c_b14, &a[j + j * a_dim1], lda);
		spotrf2_("Upper", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {
		    //
		    //                Compute the current block row.
		    //
		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    sgemm_("Transpose", "No transpose", &jb, &i__3, &i__4, &
			    c_b13, &a[j * a_dim1 + 1], lda, &a[(j + jb) *
			    a_dim1 + 1], lda, &c_b14, &a[j + (j + jb) *
			    a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    strsm_("Left", "Upper", "Transpose", "Non-unit", &jb, &
			    i__3, &c_b14, &a[j + j * a_dim1], lda, &a[j + (j
			    + jb) * a_dim1], lda);
		}
// L10:
	    }
	} else {
	    //
	    //          Compute the Cholesky factorization A = L*L**T.
	    //
	    i__2 = *n;
	    i__1 = nb;
	    for (j = 1; i__1 < 0 ? j >= i__2 : j <= i__2; j += i__1) {
		//
		//             Update and factorize the current diagonal block and test
		//             for non-positive-definiteness.
		//
		// Computing MIN
		i__3 = nb, i__4 = *n - j + 1;
		jb = min(i__3,i__4);
		i__3 = j - 1;
		ssyrk_("Lower", "No transpose", &jb, &i__3, &c_b13, &a[j +
			a_dim1], lda, &c_b14, &a[j + j * a_dim1], lda);
		spotrf2_("Lower", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {
		    //
		    //                Compute the current block column.
		    //
		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    sgemm_("No transpose", "Transpose", &i__3, &jb, &i__4, &
			    c_b13, &a[j + jb + a_dim1], lda, &a[j + a_dim1],
			    lda, &c_b14, &a[j + jb + j * a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    strsm_("Right", "Lower", "Transpose", "Non-unit", &i__3, &
			    jb, &c_b14, &a[j + j * a_dim1], lda, &a[j + jb +
			    j * a_dim1], lda);
		}
// L20:
	    }
	}
    }
    goto L40;
L30:
    *info = *info + j - 1;
L40:
    return 0;
    //
    //    End of SPOTRF
    //
} // spotrf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SPOTRF2
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      RECURSIVE SUBROUTINE SPOTRF2( UPLO, N, A, LDA, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDA, N
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
//> SPOTRF2 computes the Cholesky factorization of a real symmetric
//> positive definite matrix A using the recursive algorithm.
//>
//> The factorization has the form
//>    A = U**T * U,  if UPLO = 'U', or
//>    A = L  * L**T,  if UPLO = 'L',
//> where U is an upper triangular matrix and L is lower triangular.
//>
//> This is the recursive version of the algorithm. It divides
//> the matrix into four submatrices:
//>
//>        [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
//>    A = [ -----|----- ]  with n1 = n/2
//>        [  A21 | A22  ]       n2 = n-n1
//>
//> The subroutine calls itself to factor A11. Update and scale A21
//> or A12, update A22 then call itself to factor A22.
//>
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U':  Upper triangle of A is stored;
//>          = 'L':  Lower triangle of A is stored.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
//>          N-by-N upper triangular part of A contains the upper
//>          triangular part of the matrix A, and the strictly lower
//>          triangular part of A is not referenced.  If UPLO = 'L', the
//>          leading N-by-N lower triangular part of A contains the lower
//>          triangular part of the matrix A, and the strictly upper
//>          triangular part of A is not referenced.
//>
//>          On exit, if INFO = 0, the factor U or L from the Cholesky
//>          factorization A = U**T*U or A = L*L**T.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  if INFO = i, the leading minor of order i is not
//>                positive definite, and the factorization could not be
//>                completed.
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
//> \ingroup realPOcomputational
//
// =====================================================================
/* Subroutine */ int spotrf2_(char *uplo, int *n, float *a, int *lda, int *
	info)
{
    // Table of constant values
    float c_b9 = 1.f;
    float c_b11 = -1.f;

    // System generated locals
    int a_dim1, a_offset, i__1;

    // Local variables
    int n1, n2;
    extern int lsame_(char *, char *);
    int iinfo;
    int upper;
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *, int *,
	    int *, float *, float *, int *, float *, int *), ssyrk_(char *,
	    char *, int *, int *, float *, float *, int *, float *, float *,
	    int *), xerbla_(char *, int *);
    extern int sisnan_(float *);

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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTRF2", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	return 0;
    }
    //
    //    N=1 case
    //
    if (*n == 1) {
	//
	//       Test for non-positive-definiteness
	//
	if (a[a_dim1 + 1] <= 0.f || sisnan_(&a[a_dim1 + 1])) {
	    *info = 1;
	    return 0;
	}
	//
	//       Factor
	//
	a[a_dim1 + 1] = sqrt(a[a_dim1 + 1]);
	//
	//    Use recursive code
	//
    } else {
	n1 = *n / 2;
	n2 = *n - n1;
	//
	//       Factor A11
	//
	spotrf2_(uplo, &n1, &a[a_dim1 + 1], lda, &iinfo);
	if (iinfo != 0) {
	    *info = iinfo;
	    return 0;
	}
	//
	//       Compute the Cholesky factorization A = U**T*U
	//
	if (upper) {
	    //
	    //          Update and scale A12
	    //
	    strsm_("L", "U", "T", "N", &n1, &n2, &c_b9, &a[a_dim1 + 1], lda, &
		    a[(n1 + 1) * a_dim1 + 1], lda);
	    //
	    //          Update and factor A22
	    //
	    ssyrk_(uplo, "T", &n2, &n1, &c_b11, &a[(n1 + 1) * a_dim1 + 1],
		    lda, &c_b9, &a[n1 + 1 + (n1 + 1) * a_dim1], lda);
	    spotrf2_(uplo, &n2, &a[n1 + 1 + (n1 + 1) * a_dim1], lda, &iinfo);
	    if (iinfo != 0) {
		*info = iinfo + n1;
		return 0;
	    }
	    //
	    //       Compute the Cholesky factorization A = L*L**T
	    //
	} else {
	    //
	    //          Update and scale A21
	    //
	    strsm_("R", "L", "T", "N", &n2, &n1, &c_b9, &a[a_dim1 + 1], lda, &
		    a[n1 + 1 + a_dim1], lda);
	    //
	    //          Update and factor A22
	    //
	    ssyrk_(uplo, "N", &n2, &n1, &c_b11, &a[n1 + 1 + a_dim1], lda, &
		    c_b9, &a[n1 + 1 + (n1 + 1) * a_dim1], lda);
	    spotrf2_(uplo, &n2, &a[n1 + 1 + (n1 + 1) * a_dim1], lda, &iinfo);
	    if (iinfo != 0) {
		*info = iinfo + n1;
		return 0;
	    }
	}
    }
    return 0;
    //
    //    End of SPOTRF2
    //
} // spotrf2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SPOTRS
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SPOTRS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/spotrs.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/spotrs.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/spotrs.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
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
//> SPOTRS solves a system of linear equations A*X = B with a symmetric
//> positive definite matrix A using the Cholesky factorization
//> A = U**T*U or A = L*L**T computed by SPOTRF.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U':  Upper triangle of A is stored;
//>          = 'L':  Lower triangle of A is stored.
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
//>          The triangular factor U or L from the Cholesky factorization
//>          A = U**T*U or A = L*L**T, as computed by SPOTRF.
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
//>          On exit, the solution matrix X.
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
//> \ingroup realPOcomputational
//
// =====================================================================
/* Subroutine */ int spotrs_(char *uplo, int *n, int *nrhs, float *a, int *
	lda, float *b, int *ldb, int *info)
{
    // Table of constant values
    float c_b9 = 1.f;

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern int lsame_(char *, char *);
    int upper;
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *, int *,
	    int *, float *, float *, int *, float *, int *), xerbla_(char *,
	    int *);

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
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPOTRS", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0 || *nrhs == 0) {
	return 0;
    }
    if (upper) {
	//
	//       Solve A*X = B where A = U**T *U.
	//
	//       Solve U**T *X = B, overwriting B with X.
	//
	strsm_("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &c_b9, &a[
		a_offset], lda, &b[b_offset], ldb);
	//
	//       Solve U*X = B, overwriting B with X.
	//
	strsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b9, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {
	//
	//       Solve A*X = B where A = L*L**T.
	//
	//       Solve L*X = B, overwriting B with X.
	//
	strsm_("Left", "Lower", "No transpose", "Non-unit", n, nrhs, &c_b9, &
		a[a_offset], lda, &b[b_offset], ldb);
	//
	//       Solve L**T *X = B, overwriting B with X.
	//
	strsm_("Left", "Lower", "Transpose", "Non-unit", n, nrhs, &c_b9, &a[
		a_offset], lda, &b[b_offset], ldb);
    }
    return 0;
    //
    //    End of SPOTRS
    //
} // spotrs_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SSYRK
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE SSYRK(UPLO,TRANS,N,K,ALPHA,A,LDA,BETA,C,LDC)
//
//      .. Scalar Arguments ..
//      REAL ALPHA,BETA
//      INTEGER K,LDA,LDC,N
//      CHARACTER TRANS,UPLO
//      ..
//      .. Array Arguments ..
//      REAL A(LDA,*),C(LDC,*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SSYRK  performs one of the symmetric rank k operations
//>
//>    C := alpha*A*A**T + beta*C,
//>
//> or
//>
//>    C := alpha*A**T*A + beta*C,
//>
//> where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
//> and  A  is an  n by k  matrix in the first case and a  k by n  matrix
//> in the second case.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>           On  entry,   UPLO  specifies  whether  the  upper  or  lower
//>           triangular  part  of the  array  C  is to be  referenced  as
//>           follows:
//>
//>              UPLO = 'U' or 'u'   Only the  upper triangular part of  C
//>                                  is to be referenced.
//>
//>              UPLO = 'L' or 'l'   Only the  lower triangular part of  C
//>                                  is to be referenced.
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>           On entry,  TRANS  specifies the operation to be performed as
//>           follows:
//>
//>              TRANS = 'N' or 'n'   C := alpha*A*A**T + beta*C.
//>
//>              TRANS = 'T' or 't'   C := alpha*A**T*A + beta*C.
//>
//>              TRANS = 'C' or 'c'   C := alpha*A**T*A + beta*C.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry,  N specifies the order of the matrix C.  N must be
//>           at least zero.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>           On entry with  TRANS = 'N' or 'n',  K  specifies  the number
//>           of  columns   of  the   matrix   A,   and  on   entry   with
//>           TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
//>           of rows of the matrix  A.  K must be at least zero.
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
//>          A is REAL array, dimension ( LDA, ka ), where ka is
//>           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
//>           Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
//>           part of the array  A  must contain the matrix  A,  otherwise
//>           the leading  k by n  part of the array  A  must contain  the
//>           matrix A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
//>           then  LDA must be at least  max( 1, n ), otherwise  LDA must
//>           be at least  max( 1, k ).
//> \endverbatim
//>
//> \param[in] BETA
//> \verbatim
//>          BETA is REAL
//>           On entry, BETA specifies the scalar beta.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension ( LDC, N )
//>           Before entry  with  UPLO = 'U' or 'u',  the leading  n by n
//>           upper triangular part of the array C must contain the upper
//>           triangular part  of the  symmetric matrix  and the strictly
//>           lower triangular part of C is not referenced.  On exit, the
//>           upper triangular part of the array  C is overwritten by the
//>           upper triangular part of the updated matrix.
//>           Before entry  with  UPLO = 'L' or 'l',  the leading  n by n
//>           lower triangular part of the array C must contain the lower
//>           triangular part  of the  symmetric matrix  and the strictly
//>           upper triangular part of C is not referenced.  On exit, the
//>           lower triangular part of the array  C is overwritten by the
//>           lower triangular part of the updated matrix.
//> \endverbatim
//>
//> \param[in] LDC
//> \verbatim
//>          LDC is INTEGER
//>           On entry, LDC specifies the first dimension of C as declared
//>           in  the  calling  (sub)  program.   LDC  must  be  at  least
//>           max( 1, n ).
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
/* Subroutine */ int ssyrk_(char *uplo, char *trans, int *n, int *k, float *
	alpha, float *a, int *lda, float *beta, float *c__, int *ldc)
{
    // System generated locals
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;

    // Local variables
    int i__, j, l, info;
    float temp;
    extern int lsame_(char *, char *);
    int nrowa;
    int upper;
    extern /* Subroutine */ int xerbla_(char *, int *);

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
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    // Function Body
    if (lsame_(trans, "N")) {
	nrowa = *n;
    } else {
	nrowa = *k;
    }
    upper = lsame_(uplo, "U");
    info = 0;
    if (! upper && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T") && ! lsame_(trans,
	     "C")) {
	info = 2;
    } else if (*n < 0) {
	info = 3;
    } else if (*k < 0) {
	info = 4;
    } else if (*lda < max(1,nrowa)) {
	info = 7;
    } else if (*ldc < max(1,*n)) {
	info = 10;
    }
    if (info != 0) {
	xerbla_("SSYRK ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {
	return 0;
    }
    //
    //    And when  alpha.eq.zero.
    //
    if (*alpha == 0.f) {
	if (upper) {
	    if (*beta == 0.f) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
// L10:
		    }
// L20:
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L30:
		    }
// L40:
		}
	    }
	} else {
	    if (*beta == 0.f) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
// L50:
		    }
// L60:
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L70:
		    }
// L80:
		}
	    }
	}
	return 0;
    }
    //
    //    Start the operations.
    //
    if (lsame_(trans, "N")) {
	//
	//       Form  C := alpha*A*A**T + beta*C.
	//
	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.f) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
// L90:
		    }
		} else if (*beta != 1.f) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L100:
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.f) {
			temp = *alpha * a[j + l * a_dim1];
			i__3 = j;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l *
				    a_dim1];
// L110:
			}
		    }
// L120:
		}
// L130:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.f) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
// L140:
		    }
		} else if (*beta != 1.f) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L150:
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.f) {
			temp = *alpha * a[j + l * a_dim1];
			i__3 = *n;
			for (i__ = j; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] += temp * a[i__ + l *
				    a_dim1];
// L160:
			}
		    }
// L170:
		}
// L180:
	    }
	}
    } else {
	//
	//       Form  C := alpha*A**T*A + beta*C.
	//
	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.f;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
// L190:
		    }
		    if (*beta == 0.f) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
// L200:
		}
// L210:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = j; i__ <= i__2; ++i__) {
		    temp = 0.f;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
// L220:
		    }
		    if (*beta == 0.f) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
// L230:
		}
// L240:
	    }
	}
    }
    return 0;
    //
    //    End of SSYRK .
    //
} // ssyrk_

