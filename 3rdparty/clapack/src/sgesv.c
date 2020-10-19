/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b ISAMAX
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      INTEGER FUNCTION ISAMAX(N,SX,INCX)
//
//      .. Scalar Arguments ..
//      INTEGER INCX,N
//      ..
//      .. Array Arguments ..
//      REAL SX(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    ISAMAX finds the index of the first element having maximum absolute value.
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
//> \ingroup aux_blas
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>     jack dongarra, linpack, 3/11/78.
//>     modified 3/93 to return if incx .le. 0.
//>     modified 12/3/93, array(1) declarations changed to array(*)
//> \endverbatim
//>
// =====================================================================
int isamax_(int *n, float *sx, int *incx)
{
    // System generated locals
    int ret_val, i__1;
    float r__1;

    // Local variables
    int i__, ix;
    float smax;

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
    --sx;

    // Function Body
    ret_val = 0;
    if (*n < 1 || *incx <= 0) {
	return ret_val;
    }
    ret_val = 1;
    if (*n == 1) {
	return ret_val;
    }
    if (*incx == 1) {
	//
	//       code for increment equal to 1
	//
	smax = dabs(sx[1]);
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if ((r__1 = sx[i__], dabs(r__1)) > smax) {
		ret_val = i__;
		smax = (r__1 = sx[i__], dabs(r__1));
	    }
	}
    } else {
	//
	//       code for increment not equal to 1
	//
	ix = 1;
	smax = dabs(sx[1]);
	ix += *incx;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if ((r__1 = sx[ix], dabs(r__1)) > smax) {
		ret_val = i__;
		smax = (r__1 = sx[ix], dabs(r__1));
	    }
	    ix += *incx;
	}
    }
    return ret_val;
} // isamax_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief <b> SGESV computes the solution to system of linear equations A * X = B for GE matrices</b> (simple driver)
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGESV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgesv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgesv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgesv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGESV( N, NRHS, A, LDA, IPIV, B, LDB, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, LDB, N, NRHS
//      ..
//      .. Array Arguments ..
//      INTEGER            IPIV( * )
//      REAL               A( LDA, * ), B( LDB, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGESV computes the solution to a real system of linear equations
//>    A * X = B,
//> where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
//>
//> The LU decomposition with partial pivoting and row interchanges is
//> used to factor A as
//>    A = P * L * U,
//> where P is a permutation matrix, L is unit lower triangular, and U is
//> upper triangular.  The factored form of A is then used to solve the
//> system of equations A * X = B.
//> \endverbatim
//
// Arguments:
// ==========
//
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
//>          On entry, the N-by-N coefficient matrix A.
//>          On exit, the factors L and U from the factorization
//>          A = P*L*U; the unit diagonal elements of L are not stored.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] IPIV
//> \verbatim
//>          IPIV is INTEGER array, dimension (N)
//>          The pivot indices that define the permutation matrix P;
//>          row i of the matrix was interchanged with row IPIV(i).
//> \endverbatim
//>
//> \param[in,out] B
//> \verbatim
//>          B is REAL array, dimension (LDB,NRHS)
//>          On entry, the N-by-NRHS matrix of right hand side matrix B.
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
//>          > 0:  if INFO = i, U(i,i) is exactly zero.  The factorization
//>                has been completed, but the factor U is exactly
//>                singular, so the solution could not be computed.
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
/* Subroutine */ int sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv,
	float *b, int *ldb, int *info)
{
    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern /* Subroutine */ int xerbla_(char *, int *), sgetrf_(int *, int *,
	    float *, int *, int *, int *), sgetrs_(char *, int *, int *,
	    float *, int *, int *, float *, int *, int *);

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
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    // Function Body
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*nrhs < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*ldb < max(1,*n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGESV ", &i__1);
	return 0;
    }
    //
    //    Compute the LU factorization of A.
    //
    sgetrf_(n, n, &a[a_offset], lda, &ipiv[1], info);
    if (*info == 0) {
	//
	//       Solve the system A*X = B, overwriting B with X.
	//
	sgetrs_("No transpose", n, nrhs, &a[a_offset], lda, &ipiv[1], &b[
		b_offset], ldb, info);
    }
    return 0;
    //
    //    End of SGESV
    //
} // sgesv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGETRF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGETRF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgetrf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgetrf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgetrf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGETRF( M, N, A, LDA, IPIV, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, M, N
//      ..
//      .. Array Arguments ..
//      INTEGER            IPIV( * )
//      REAL               A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGETRF computes an LU factorization of a general M-by-N matrix A
//> using partial pivoting with row interchanges.
//>
//> The factorization has the form
//>    A = P * L * U
//> where P is a permutation matrix, L is lower triangular with unit
//> diagonal elements (lower trapezoidal if m > n), and U is upper
//> triangular (upper trapezoidal if m < n).
//>
//> This is the right-looking Level 3 BLAS version of the algorithm.
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
//>          On entry, the M-by-N matrix to be factored.
//>          On exit, the factors L and U from the factorization
//>          A = P*L*U; the unit diagonal elements of L are not stored.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] IPIV
//> \verbatim
//>          IPIV is INTEGER array, dimension (min(M,N))
//>          The pivot indices; for 1 <= i <= min(M,N), row i of the
//>          matrix was interchanged with row IPIV(i).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
//>                has been completed, but the factor U is exactly
//>                singular, and division by zero will occur if it is used
//>                to solve a system of equations.
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
//> \ingroup realGEcomputational
//
// =====================================================================
/* Subroutine */ int sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv,
	int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    float c_b16 = 1.f;
    float c_b19 = -1.f;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    // Local variables
    int i__, j, jb, nb, iinfo;
    extern /* Subroutine */ int sgemm_(char *, char *, int *, int *, int *,
	    float *, float *, int *, float *, int *, float *, float *, int *),
	     strsm_(char *, char *, char *, char *, int *, int *, float *,
	    float *, int *, float *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int slaswp_(int *, float *, int *, int *, int *,
	    int *, int *), sgetrf2_(int *, int *, float *, int *, int *, int *
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
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;

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
	xerbla_("SGETRF", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0) {
	return 0;
    }
    //
    //    Determine the block size for this environment.
    //
    nb = ilaenv_(&c__1, "SGETRF", " ", m, n, &c_n1, &c_n1);
    if (nb <= 1 || nb >= min(*m,*n)) {
	//
	//       Use unblocked code.
	//
	sgetrf2_(m, n, &a[a_offset], lda, &ipiv[1], info);
    } else {
	//
	//       Use blocked code.
	//
	i__1 = min(*m,*n);
	i__2 = nb;
	for (j = 1; i__2 < 0 ? j >= i__1 : j <= i__1; j += i__2) {
	    // Computing MIN
	    i__3 = min(*m,*n) - j + 1;
	    jb = min(i__3,nb);
	    //
	    //          Factor diagonal and subdiagonal blocks and test for exact
	    //          singularity.
	    //
	    i__3 = *m - j + 1;
	    sgetrf2_(&i__3, &jb, &a[j + j * a_dim1], lda, &ipiv[j], &iinfo);
	    //
	    //          Adjust INFO and the pivot indices.
	    //
	    if (*info == 0 && iinfo > 0) {
		*info = iinfo + j - 1;
	    }
	    // Computing MIN
	    i__4 = *m, i__5 = j + jb - 1;
	    i__3 = min(i__4,i__5);
	    for (i__ = j; i__ <= i__3; ++i__) {
		ipiv[i__] = j - 1 + ipiv[i__];
// L10:
	    }
	    //
	    //          Apply interchanges to columns 1:J-1.
	    //
	    i__3 = j - 1;
	    i__4 = j + jb - 1;
	    slaswp_(&i__3, &a[a_offset], lda, &j, &i__4, &ipiv[1], &c__1);
	    if (j + jb <= *n) {
		//
		//             Apply interchanges to columns J+JB:N.
		//
		i__3 = *n - j - jb + 1;
		i__4 = j + jb - 1;
		slaswp_(&i__3, &a[(j + jb) * a_dim1 + 1], lda, &j, &i__4, &
			ipiv[1], &c__1);
		//
		//             Compute block row of U.
		//
		i__3 = *n - j - jb + 1;
		strsm_("Left", "Lower", "No transpose", "Unit", &jb, &i__3, &
			c_b16, &a[j + j * a_dim1], lda, &a[j + (j + jb) *
			a_dim1], lda);
		if (j + jb <= *m) {
		    //
		    //                Update trailing submatrix.
		    //
		    i__3 = *m - j - jb + 1;
		    i__4 = *n - j - jb + 1;
		    sgemm_("No transpose", "No transpose", &i__3, &i__4, &jb,
			    &c_b19, &a[j + jb + j * a_dim1], lda, &a[j + (j +
			    jb) * a_dim1], lda, &c_b16, &a[j + jb + (j + jb) *
			     a_dim1], lda);
		}
	    }
// L20:
	}
    }
    return 0;
    //
    //    End of SGETRF
    //
} // sgetrf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGETRF2
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      RECURSIVE SUBROUTINE SGETRF2( M, N, A, LDA, IPIV, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDA, M, N
//      ..
//      .. Array Arguments ..
//      INTEGER            IPIV( * )
//      REAL               A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGETRF2 computes an LU factorization of a general M-by-N matrix A
//> using partial pivoting with row interchanges.
//>
//> The factorization has the form
//>    A = P * L * U
//> where P is a permutation matrix, L is lower triangular with unit
//> diagonal elements (lower trapezoidal if m > n), and U is upper
//> triangular (upper trapezoidal if m < n).
//>
//> This is the recursive version of the algorithm. It divides
//> the matrix into four submatrices:
//>
//>        [  A11 | A12  ]  where A11 is n1 by n1 and A22 is n2 by n2
//>    A = [ -----|----- ]  with n1 = min(m,n)/2
//>        [  A21 | A22  ]       n2 = n-n1
//>
//>                                       [ A11 ]
//> The subroutine calls itself to factor [ --- ],
//>                                       [ A12 ]
//>                 [ A12 ]
//> do the swaps on [ --- ], solve A12, update A22,
//>                 [ A22 ]
//>
//> then calls itself to factor A22 and do the swaps on A21.
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
//>          On entry, the M-by-N matrix to be factored.
//>          On exit, the factors L and U from the factorization
//>          A = P*L*U; the unit diagonal elements of L are not stored.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,M).
//> \endverbatim
//>
//> \param[out] IPIV
//> \verbatim
//>          IPIV is INTEGER array, dimension (min(M,N))
//>          The pivot indices; for 1 <= i <= min(M,N), row i of the
//>          matrix was interchanged with row IPIV(i).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
//>                has been completed, but the factor U is exactly
//>                singular, and division by zero will occur if it is used
//>                to solve a system of equations.
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
//> \ingroup realGEcomputational
//
// =====================================================================
/* Subroutine */ int sgetrf2_(int *m, int *n, float *a, int *lda, int *ipiv,
	int *info)
{
    // Table of constant values
    int c__1 = 1;
    float c_b13 = 1.f;
    float c_b16 = -1.f;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2;
    float r__1;

    // Local variables
    int i__, n1, n2;
    float temp;
    int iinfo;
    extern /* Subroutine */ int sscal_(int *, float *, float *, int *),
	    sgemm_(char *, char *, int *, int *, int *, float *, float *, int
	    *, float *, int *, float *, float *, int *);
    float sfmin;
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *, int *,
	    int *, float *, float *, int *, float *, int *);
    extern double slamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int isamax_(int *, float *, int *);
    extern /* Subroutine */ int slaswp_(int *, float *, int *, int *, int *,
	    int *, int *);

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
    --ipiv;

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
	xerbla_("SGETRF2", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0) {
	return 0;
    }
    if (*m == 1) {
	//
	//       Use unblocked code for one row case
	//       Just need to handle IPIV and INFO
	//
	ipiv[1] = 1;
	if (a[a_dim1 + 1] == 0.f) {
	    *info = 1;
	}
    } else if (*n == 1) {
	//
	//       Use unblocked code for one column case
	//
	//
	//       Compute machine safe minimum
	//
	sfmin = slamch_("S");
	//
	//       Find pivot and test for singularity
	//
	i__ = isamax_(m, &a[a_dim1 + 1], &c__1);
	ipiv[1] = i__;
	if (a[i__ + a_dim1] != 0.f) {
	    //
	    //          Apply the interchange
	    //
	    if (i__ != 1) {
		temp = a[a_dim1 + 1];
		a[a_dim1 + 1] = a[i__ + a_dim1];
		a[i__ + a_dim1] = temp;
	    }
	    //
	    //          Compute elements 2:M of the column
	    //
	    if ((r__1 = a[a_dim1 + 1], dabs(r__1)) >= sfmin) {
		i__1 = *m - 1;
		r__1 = 1.f / a[a_dim1 + 1];
		sscal_(&i__1, &r__1, &a[a_dim1 + 2], &c__1);
	    } else {
		i__1 = *m - 1;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    a[i__ + 1 + a_dim1] /= a[a_dim1 + 1];
// L10:
		}
	    }
	} else {
	    *info = 1;
	}
    } else {
	//
	//       Use recursive code
	//
	n1 = min(*m,*n) / 2;
	n2 = *n - n1;
	//
	//              [ A11 ]
	//       Factor [ --- ]
	//              [ A21 ]
	//
	sgetrf2_(m, &n1, &a[a_offset], lda, &ipiv[1], &iinfo);
	if (*info == 0 && iinfo > 0) {
	    *info = iinfo;
	}
	//
	//                             [ A12 ]
	//       Apply interchanges to [ --- ]
	//                             [ A22 ]
	//
	slaswp_(&n2, &a[(n1 + 1) * a_dim1 + 1], lda, &c__1, &n1, &ipiv[1], &
		c__1);
	//
	//       Solve A12
	//
	strsm_("L", "L", "N", "U", &n1, &n2, &c_b13, &a[a_offset], lda, &a[(
		n1 + 1) * a_dim1 + 1], lda);
	//
	//       Update A22
	//
	i__1 = *m - n1;
	sgemm_("N", "N", &i__1, &n2, &n1, &c_b16, &a[n1 + 1 + a_dim1], lda, &
		a[(n1 + 1) * a_dim1 + 1], lda, &c_b13, &a[n1 + 1 + (n1 + 1) *
		a_dim1], lda);
	//
	//       Factor A22
	//
	i__1 = *m - n1;
	sgetrf2_(&i__1, &n2, &a[n1 + 1 + (n1 + 1) * a_dim1], lda, &ipiv[n1 +
		1], &iinfo);
	//
	//       Adjust INFO and the pivot indices
	//
	if (*info == 0 && iinfo > 0) {
	    *info = iinfo + n1;
	}
	i__1 = min(*m,*n);
	for (i__ = n1 + 1; i__ <= i__1; ++i__) {
	    ipiv[i__] += n1;
// L20:
	}
	//
	//       Apply interchanges to A21
	//
	i__1 = n1 + 1;
	i__2 = min(*m,*n);
	slaswp_(&n1, &a[a_dim1 + 1], lda, &i__1, &i__2, &ipiv[1], &c__1);
    }
    return 0;
    //
    //    End of SGETRF2
    //
} // sgetrf2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SGETRS
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SGETRS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sgetrs.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sgetrs.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sgetrs.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SGETRS( TRANS, N, NRHS, A, LDA, IPIV, B, LDB, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          TRANS
//      INTEGER            INFO, LDA, LDB, N, NRHS
//      ..
//      .. Array Arguments ..
//      INTEGER            IPIV( * )
//      REAL               A( LDA, * ), B( LDB, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGETRS solves a system of linear equations
//>    A * X = B  or  A**T * X = B
//> with a general N-by-N matrix A using the LU factorization computed
//> by SGETRF.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          Specifies the form of the system of equations:
//>          = 'N':  A * X = B  (No transpose)
//>          = 'T':  A**T* X = B  (Transpose)
//>          = 'C':  A**T* X = B  (Conjugate transpose = Transpose)
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
//>          The factors L and U from the factorization A = P*L*U
//>          as computed by SGETRF.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in] IPIV
//> \verbatim
//>          IPIV is INTEGER array, dimension (N)
//>          The pivot indices from SGETRF; for 1<=i<=N, row i of the
//>          matrix was interchanged with row IPIV(i).
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
//> \ingroup realGEcomputational
//
// =====================================================================
/* Subroutine */ int sgetrs_(char *trans, int *n, int *nrhs, float *a, int *
	lda, int *ipiv, float *b, int *ldb, int *info)
{
    // Table of constant values
    int c__1 = 1;
    float c_b12 = 1.f;
    int c_n1 = -1;

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int strsm_(char *, char *, char *, char *, int *,
	    int *, float *, float *, int *, float *, int *), xerbla_(char *,
	    int *);
    int notran;
    extern /* Subroutine */ int slaswp_(int *, float *, int *, int *, int *,
	    int *, int *);

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
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    // Function Body
    *info = 0;
    notran = lsame_(trans, "N");
    if (! notran && ! lsame_(trans, "T") && ! lsame_(trans, "C")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldb < max(1,*n)) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SGETRS", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0 || *nrhs == 0) {
	return 0;
    }
    if (notran) {
	//
	//       Solve A * X = B.
	//
	//       Apply row interchanges to the right hand sides.
	//
	slaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c__1);
	//
	//       Solve L*X = B, overwriting B with X.
	//
	strsm_("Left", "Lower", "No transpose", "Unit", n, nrhs, &c_b12, &a[
		a_offset], lda, &b[b_offset], ldb);
	//
	//       Solve U*X = B, overwriting B with X.
	//
	strsm_("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &c_b12, &
		a[a_offset], lda, &b[b_offset], ldb);
    } else {
	//
	//       Solve A**T * X = B.
	//
	//       Solve U**T *X = B, overwriting B with X.
	//
	strsm_("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &c_b12, &a[
		a_offset], lda, &b[b_offset], ldb);
	//
	//       Solve L**T *X = B, overwriting B with X.
	//
	strsm_("Left", "Lower", "Transpose", "Unit", n, nrhs, &c_b12, &a[
		a_offset], lda, &b[b_offset], ldb);
	//
	//       Apply row interchanges to the solution vectors.
	//
	slaswp_(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c_n1);
    }
    return 0;
    //
    //    End of SGETRS
    //
} // sgetrs_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLASWP performs a series of row interchanges on a general rectangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLASWP + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slaswp.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slaswp.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slaswp.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE SLASWP( N, A, LDA, K1, K2, IPIV, INCX )
//
//      .. Scalar Arguments ..
//      INTEGER            INCX, K1, K2, LDA, N
//      ..
//      .. Array Arguments ..
//      INTEGER            IPIV( * )
//      REAL               A( LDA, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SLASWP performs a series of row interchanges on the matrix A.
//> One row interchange is initiated for each of rows K1 through K2 of A.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix A.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is REAL array, dimension (LDA,N)
//>          On entry, the matrix of column dimension N to which the row
//>          interchanges will be applied.
//>          On exit, the permuted matrix.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//> \endverbatim
//>
//> \param[in] K1
//> \verbatim
//>          K1 is INTEGER
//>          The first element of IPIV for which a row interchange will
//>          be done.
//> \endverbatim
//>
//> \param[in] K2
//> \verbatim
//>          K2 is INTEGER
//>          (K2-K1+1) is the number of elements of IPIV for which a row
//>          interchange will be done.
//> \endverbatim
//>
//> \param[in] IPIV
//> \verbatim
//>          IPIV is INTEGER array, dimension (K1+(K2-K1)*abs(INCX))
//>          The vector of pivot indices. Only the elements in positions
//>          K1 through K1+(K2-K1)*abs(INCX) of IPIV are accessed.
//>          IPIV(K1+(K-K1)*abs(INCX)) = L implies rows K and L are to be
//>          interchanged.
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>          The increment between successive values of IPIV. If INCX
//>          is negative, the pivots are applied in reverse order.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Modified by
//>   R. C. Whaley, Computer Science Dept., Univ. of Tenn., Knoxville, USA
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int slaswp_(int *n, float *a, int *lda, int *k1, int *k2,
	int *ipiv, int *incx)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    // Local variables
    int i__, j, k, i1, i2, n32, ip, ix, ix0, inc;
    float temp;

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
    //=====================================================================
    //
    //    .. Local Scalars ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Interchange row I with row IPIV(K1+(I-K1)*abs(INCX)) for each of rows
    //    K1 through K2.
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;

    // Function Body
    if (*incx > 0) {
	ix0 = *k1;
	i1 = *k1;
	i2 = *k2;
	inc = 1;
    } else if (*incx < 0) {
	ix0 = *k1 + (*k1 - *k2) * *incx;
	i1 = *k2;
	i2 = *k1;
	inc = -1;
    } else {
	return 0;
    }
    n32 = *n / 32 << 5;
    if (n32 != 0) {
	i__1 = n32;
	for (j = 1; j <= i__1; j += 32) {
	    ix = ix0;
	    i__2 = i2;
	    i__3 = inc;
	    for (i__ = i1; i__3 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__3)
		    {
		ip = ipiv[ix];
		if (ip != i__) {
		    i__4 = j + 31;
		    for (k = j; k <= i__4; ++k) {
			temp = a[i__ + k * a_dim1];
			a[i__ + k * a_dim1] = a[ip + k * a_dim1];
			a[ip + k * a_dim1] = temp;
// L10:
		    }
		}
		ix += *incx;
// L20:
	    }
// L30:
	}
    }
    if (n32 != *n) {
	++n32;
	ix = ix0;
	i__1 = i2;
	i__3 = inc;
	for (i__ = i1; i__3 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__3) {
	    ip = ipiv[ix];
	    if (ip != i__) {
		i__2 = *n;
		for (k = n32; k <= i__2; ++k) {
		    temp = a[i__ + k * a_dim1];
		    a[i__ + k * a_dim1] = a[ip + k * a_dim1];
		    a[ip + k * a_dim1] = temp;
// L40:
		}
	    }
	    ix += *incx;
// L50:
	}
    }
    return 0;
    //
    //    End of SLASWP
    //
} // slaswp_

