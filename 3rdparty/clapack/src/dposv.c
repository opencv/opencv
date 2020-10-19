/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief <b> DPOSV computes the solution to system of linear equations A * X = B for PO matrices</b>
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DPOSV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dposv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dposv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dposv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DPOSV( UPLO, N, NRHS, A, LDA, B, LDB, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDA, LDB, N, NRHS
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DPOSV computes the solution to a real system of linear equations
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
//>          B is DOUBLE PRECISION array, dimension (LDB,NRHS)
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
//> \ingroup doublePOsolve
//
// =====================================================================
/* Subroutine */ int dposv_(char *uplo, int *n, int *nrhs, double *a, int *
	lda, double *b, int *ldb, int *info)
{
    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, int *), dpotrf_(char *, int *,
	     double *, int *, int *), dpotrs_(char *, int *, int *, double *, 
	    int *, double *, int *, int *);

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
	xerbla_("DPOSV ", &i__1);
	return 0;
    }
    //
    //    Compute the Cholesky factorization A = U**T*U or A = L*L**T.
    //
    dpotrf_(uplo, n, &a[a_offset], lda, info);
    if (*info == 0) {
	//
	//       Solve the system A*X = B, overwriting B with X.
	//
	dpotrs_(uplo, n, nrhs, &a[a_offset], lda, &b[b_offset], ldb, info);
    }
    return 0;
    //
    //    End of DPOSV
    //
} // dposv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DPOTRF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DPOTRF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dpotrf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dpotrf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dpotrf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DPOTRF( UPLO, N, A, LDA, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDA, N
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
//> DPOTRF computes the Cholesky factorization of a real symmetric
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
//> \ingroup doublePOcomputational
//
// =====================================================================
/* Subroutine */ int dpotrf_(char *uplo, int *n, double *a, int *lda, int *
	info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    double c_b13 = -1.;
    double c_b14 = 1.;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    // Local variables
    int j, jb, nb;
    extern /* Subroutine */ int dgemm_(char *, char *, int *, int *, int *, 
	    double *, double *, int *, double *, int *, double *, double *, 
	    int *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dtrsm_(char *, char *, char *, char *, int *, 
	    int *, double *, double *, int *, double *, int *);
    int upper;
    extern /* Subroutine */ int dsyrk_(char *, char *, int *, int *, double *,
	     double *, int *, double *, double *, int *), xerbla_(char *, int 
	    *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dpotrf2_(char *, int *, double *, int *, int *
	    );

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
	xerbla_("DPOTRF", &i__1);
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
    nb = ilaenv_(&c__1, "DPOTRF", uplo, n, &c_n1, &c_n1, &c_n1);
    if (nb <= 1 || nb >= *n) {
	//
	//       Use unblocked code.
	//
	dpotrf2_(uplo, n, &a[a_offset], lda, info);
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
		dsyrk_("Upper", "Transpose", &jb, &i__3, &c_b13, &a[j * 
			a_dim1 + 1], lda, &c_b14, &a[j + j * a_dim1], lda);
		dpotrf2_("Upper", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {
		    //
		    //                Compute the current block row.
		    //
		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    dgemm_("Transpose", "No transpose", &jb, &i__3, &i__4, &
			    c_b13, &a[j * a_dim1 + 1], lda, &a[(j + jb) * 
			    a_dim1 + 1], lda, &c_b14, &a[j + (j + jb) * 
			    a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    dtrsm_("Left", "Upper", "Transpose", "Non-unit", &jb, &
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
		dsyrk_("Lower", "No transpose", &jb, &i__3, &c_b13, &a[j + 
			a_dim1], lda, &c_b14, &a[j + j * a_dim1], lda);
		dpotrf2_("Lower", &jb, &a[j + j * a_dim1], lda, info);
		if (*info != 0) {
		    goto L30;
		}
		if (j + jb <= *n) {
		    //
		    //                Compute the current block column.
		    //
		    i__3 = *n - j - jb + 1;
		    i__4 = j - 1;
		    dgemm_("No transpose", "Transpose", &i__3, &jb, &i__4, &
			    c_b13, &a[j + jb + a_dim1], lda, &a[j + a_dim1], 
			    lda, &c_b14, &a[j + jb + j * a_dim1], lda);
		    i__3 = *n - j - jb + 1;
		    dtrsm_("Right", "Lower", "Transpose", "Non-unit", &i__3, &
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
    //    End of DPOTRF
    //
} // dpotrf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DPOTRF2
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      RECURSIVE SUBROUTINE DPOTRF2( UPLO, N, A, LDA, INFO )
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
//> DPOTRF2 computes the Cholesky factorization of a real symmetric
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
//> or A12, update A22 then calls itself to factor A22.
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
//> \ingroup doublePOcomputational
//
// =====================================================================
/* Subroutine */ int dpotrf2_(char *uplo, int *n, double *a, int *lda, int *
	info)
{
    // Table of constant values
    double c_b9 = 1.;
    double c_b11 = -1.;

    // System generated locals
    int a_dim1, a_offset, i__1;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int n1, n2;
    extern int lsame_(char *, char *);
    int iinfo;
    extern /* Subroutine */ int dtrsm_(char *, char *, char *, char *, int *, 
	    int *, double *, double *, int *, double *, int *);
    int upper;
    extern /* Subroutine */ int dsyrk_(char *, char *, int *, int *, double *,
	     double *, int *, double *, double *, int *);
    extern int disnan_(double *);
    extern /* Subroutine */ int xerbla_(char *, int *);

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
	xerbla_("DPOTRF2", &i__1);
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
	if (a[a_dim1 + 1] <= 0. || disnan_(&a[a_dim1 + 1])) {
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
	dpotrf2_(uplo, &n1, &a[a_dim1 + 1], lda, &iinfo);
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
	    dtrsm_("L", "U", "T", "N", &n1, &n2, &c_b9, &a[a_dim1 + 1], lda, &
		    a[(n1 + 1) * a_dim1 + 1], lda);
	    //
	    //          Update and factor A22
	    //
	    dsyrk_(uplo, "T", &n2, &n1, &c_b11, &a[(n1 + 1) * a_dim1 + 1], 
		    lda, &c_b9, &a[n1 + 1 + (n1 + 1) * a_dim1], lda);
	    dpotrf2_(uplo, &n2, &a[n1 + 1 + (n1 + 1) * a_dim1], lda, &iinfo);
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
	    dtrsm_("R", "L", "T", "N", &n2, &n1, &c_b9, &a[a_dim1 + 1], lda, &
		    a[n1 + 1 + a_dim1], lda);
	    //
	    //          Update and factor A22
	    //
	    dsyrk_(uplo, "N", &n2, &n1, &c_b11, &a[n1 + 1 + a_dim1], lda, &
		    c_b9, &a[n1 + 1 + (n1 + 1) * a_dim1], lda);
	    dpotrf2_(uplo, &n2, &a[n1 + 1 + (n1 + 1) * a_dim1], lda, &iinfo);
	    if (iinfo != 0) {
		*info = iinfo + n1;
		return 0;
	    }
	}
    }
    return 0;
    //
    //    End of DPOTRF2
    //
} // dpotrf2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DPOTRS
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DPOTRS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dpotrs.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dpotrs.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dpotrs.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DPOTRS( UPLO, N, NRHS, A, LDA, B, LDB, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDA, LDB, N, NRHS
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), B( LDB, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DPOTRS solves a system of linear equations A*X = B with a symmetric
//> positive definite matrix A using the Cholesky factorization
//> A = U**T*U or A = L*L**T computed by DPOTRF.
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          The triangular factor U or L from the Cholesky factorization
//>          A = U**T*U or A = L*L**T, as computed by DPOTRF.
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
//>          B is DOUBLE PRECISION array, dimension (LDB,NRHS)
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
//> \ingroup doublePOcomputational
//
// =====================================================================
/* Subroutine */ int dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *
	lda, double *b, int *ldb, int *info)
{
    // Table of constant values
    double c_b9 = 1.;

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dtrsm_(char *, char *, char *, char *, int *, 
	    int *, double *, double *, int *, double *, int *);
    int upper;
    extern /* Subroutine */ int xerbla_(char *, int *);

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
	xerbla_("DPOTRS", &i__1);
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
	dtrsm_("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &c_b9, &a[
		a_offset], lda, &b[b_offset], ldb);
	//
	//       Solve U*X = B, overwriting B with X.
	//
	dtrsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b9, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {
	//
	//       Solve A*X = B where A = L*L**T.
	//
	//       Solve L*X = B, overwriting B with X.
	//
	dtrsm_("Left", "Lower", "No transpose", "Non-unit", n, nrhs, &c_b9, &
		a[a_offset], lda, &b[b_offset], ldb);
	//
	//       Solve L**T *X = B, overwriting B with X.
	//
	dtrsm_("Left", "Lower", "Transpose", "Non-unit", n, nrhs, &c_b9, &a[
		a_offset], lda, &b[b_offset], ldb);
    }
    return 0;
    //
    //    End of DPOTRS
    //
} // dpotrs_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSYRK
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DSYRK(UPLO,TRANS,N,K,ALPHA,A,LDA,BETA,C,LDC)
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION ALPHA,BETA
//      INTEGER K,LDA,LDC,N
//      CHARACTER TRANS,UPLO
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION A(LDA,*),C(LDC,*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSYRK  performs one of the symmetric rank k operations
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
//>          ALPHA is DOUBLE PRECISION.
//>           On entry, ALPHA specifies the scalar alpha.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension ( LDA, ka ), where ka is
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
//>          BETA is DOUBLE PRECISION.
//>           On entry, BETA specifies the scalar beta.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension ( LDC, N )
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
//> \ingroup double_blas_level3
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
/* Subroutine */ int dsyrk_(char *uplo, char *trans, int *n, int *k, double *
	alpha, double *a, int *lda, double *beta, double *c__, int *ldc)
{
    // System generated locals
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3;

    // Local variables
    int i__, j, l, info;
    double temp;
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
	xerbla_("DSYRK ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*n == 0 || (*alpha == 0. || *k == 0) && *beta == 1.) {
	return 0;
    }
    //
    //    And when  alpha.eq.zero.
    //
    if (*alpha == 0.) {
	if (upper) {
	    if (*beta == 0.) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
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
	    if (*beta == 0.) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
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
		if (*beta == 0.) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
// L90:
		    }
		} else if (*beta != 1.) {
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L100:
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.) {
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
		if (*beta == 0.) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.;
// L140:
		    }
		} else if (*beta != 1.) {
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L150:
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    if (a[j + l * a_dim1] != 0.) {
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
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
// L190:
		    }
		    if (*beta == 0.) {
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
		    temp = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * a[l + j * a_dim1];
// L220:
		    }
		    if (*beta == 0.) {
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
    //    End of DSYRK .
    //
} // dsyrk_

