/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief <b> DGELS solves overdetermined or underdetermined systems for GE matrices</b>
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGELS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgels.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgels.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgels.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGELS( TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK,
//                        INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          TRANS
//      INTEGER            INFO, LDA, LDB, LWORK, M, N, NRHS
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGELS solves overdetermined or underdetermined real linear systems
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the M-by-N matrix A.
//>          On exit,
//>            if M >= N, A is overwritten by details of its QR
//>                       factorization as returned by DGEQRF;
//>            if M <  N, A is overwritten by details of its LQ
//>                       factorization as returned by DGELQF.
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
//>          B is DOUBLE PRECISION array, dimension (LDB,NRHS)
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
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
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
//> \ingroup doubleGEsolve
//
// =====================================================================
/* Subroutine */ int dgels_(char *trans, int *m, int *n, int *nrhs, double *a,
	 int *lda, double *b, int *ldb, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    double c_b33 = 0.;
    int c__0 = 0;

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1, i__2;

    // Local variables
    int i__, j, nb, mn;
    double anrm, bnrm;
    int brow;
    int tpsd;
    int iascl, ibscl;
    extern int lsame_(char *, char *);
    int wsize;
    double rwork[1];
    extern /* Subroutine */ int dlabad_(double *, double *);
    extern double dlamch_(char *), dlange_(char *, int *, int *, double *,
	    int *, double *);
    extern /* Subroutine */ int dgelqf_(int *, int *, double *, int *, double
	    *, double *, int *, int *), dlascl_(char *, int *, int *, double *
	    , double *, int *, int *, double *, int *, int *), dgeqrf_(int *,
	    int *, double *, int *, double *, double *, int *, int *),
	    dlaset_(char *, int *, int *, double *, double *, double *, int *)
	    , xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int scllen;
    double bignum;
    extern /* Subroutine */ int dormlq_(char *, char *, int *, int *, int *,
	    double *, int *, double *, double *, int *, double *, int *, int *
	    ), dormqr_(char *, char *, int *, int *, int *, double *, int *,
	    double *, double *, int *, double *, int *, int *);
    double smlnum;
    int lquery;
    extern /* Subroutine */ int dtrtrs_(char *, char *, char *, int *, int *,
	    double *, int *, double *, int *, int *);

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
	    nb = ilaenv_(&c__1, "DGEQRF", " ", m, n, &c_n1, &c_n1);
	    if (tpsd) {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMQR", "LN", m, nrhs, n, &
			c_n1);
		nb = max(i__1,i__2);
	    } else {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMQR", "LT", m, nrhs, n, &
			c_n1);
		nb = max(i__1,i__2);
	    }
	} else {
	    nb = ilaenv_(&c__1, "DGELQF", " ", m, n, &c_n1, &c_n1);
	    if (tpsd) {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMLQ", "LT", n, nrhs, m, &
			c_n1);
		nb = max(i__1,i__2);
	    } else {
		// Computing MAX
		i__1 = nb, i__2 = ilaenv_(&c__1, "DORMLQ", "LN", n, nrhs, m, &
			c_n1);
		nb = max(i__1,i__2);
	    }
	}
	//
	// Computing MAX
	i__1 = 1, i__2 = mn + max(mn,*nrhs) * nb;
	wsize = max(i__1,i__2);
	work[1] = (double) wsize;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGELS ", &i__1);
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
	dlaset_("Full", &i__1, nrhs, &c_b33, &c_b33, &b[b_offset], ldb);
	return 0;
    }
    //
    //    Get machine parameters
    //
    smlnum = dlamch_("S") / dlamch_("P");
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);
    //
    //    Scale A, B if max element outside range [SMLNUM,BIGNUM]
    //
    anrm = dlange_("M", m, n, &a[a_offset], lda, rwork);
    iascl = 0;
    if (anrm > 0. && anrm < smlnum) {
	//
	//       Scale matrix norm up to SMLNUM
	//
	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, m, n, &a[a_offset], lda,
		info);
	iascl = 1;
    } else if (anrm > bignum) {
	//
	//       Scale matrix norm down to BIGNUM
	//
	dlascl_("G", &c__0, &c__0, &anrm, &bignum, m, n, &a[a_offset], lda,
		info);
	iascl = 2;
    } else if (anrm == 0.) {
	//
	//       Matrix all zero. Return zero solution.
	//
	i__1 = max(*m,*n);
	dlaset_("F", &i__1, nrhs, &c_b33, &c_b33, &b[b_offset], ldb);
	goto L50;
    }
    brow = *m;
    if (tpsd) {
	brow = *n;
    }
    bnrm = dlange_("M", &brow, nrhs, &b[b_offset], ldb, rwork);
    ibscl = 0;
    if (bnrm > 0. && bnrm < smlnum) {
	//
	//       Scale matrix norm up to SMLNUM
	//
	dlascl_("G", &c__0, &c__0, &bnrm, &smlnum, &brow, nrhs, &b[b_offset],
		ldb, info);
	ibscl = 1;
    } else if (bnrm > bignum) {
	//
	//       Scale matrix norm down to BIGNUM
	//
	dlascl_("G", &c__0, &c__0, &bnrm, &bignum, &brow, nrhs, &b[b_offset],
		ldb, info);
	ibscl = 2;
    }
    if (*m >= *n) {
	//
	//       compute QR factorization of A
	//
	i__1 = *lwork - mn;
	dgeqrf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
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
	    dormqr_("Left", "Transpose", m, nrhs, n, &a[a_offset], lda, &work[
		    1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);
	    //
	    //          workspace at least NRHS, optimally NRHS*NB
	    //
	    //          B(1:N,1:NRHS) := inv(R) * B(1:N,1:NRHS)
	    //
	    dtrtrs_("Upper", "No transpose", "Non-unit", n, nrhs, &a[a_offset]
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
	    dtrtrs_("Upper", "Transpose", "Non-unit", n, nrhs, &a[a_offset],
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
		    b[i__ + j * b_dim1] = 0.;
// L10:
		}
// L20:
	    }
	    //
	    //          B(1:M,1:NRHS) := Q(1:N,:) * B(1:N,1:NRHS)
	    //
	    i__1 = *lwork - mn;
	    dormqr_("Left", "No transpose", m, nrhs, n, &a[a_offset], lda, &
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
	dgelqf_(m, n, &a[a_offset], lda, &work[1], &work[mn + 1], &i__1, info)
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
	    dtrtrs_("Lower", "No transpose", "Non-unit", m, nrhs, &a[a_offset]
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
		    b[i__ + j * b_dim1] = 0.;
// L30:
		}
// L40:
	    }
	    //
	    //          B(1:N,1:NRHS) := Q(1:N,:)**T * B(1:M,1:NRHS)
	    //
	    i__1 = *lwork - mn;
	    dormlq_("Left", "Transpose", n, nrhs, m, &a[a_offset], lda, &work[
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
	    dormlq_("Left", "No transpose", n, nrhs, m, &a[a_offset], lda, &
		    work[1], &b[b_offset], ldb, &work[mn + 1], &i__1, info);
	    //
	    //          workspace at least NRHS, optimally NRHS*NB
	    //
	    //          B(1:M,1:NRHS) := inv(L**T) * B(1:M,1:NRHS)
	    //
	    dtrtrs_("Lower", "Transpose", "Non-unit", m, nrhs, &a[a_offset],
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
	dlascl_("G", &c__0, &c__0, &anrm, &smlnum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (iascl == 2) {
	dlascl_("G", &c__0, &c__0, &anrm, &bignum, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }
    if (ibscl == 1) {
	dlascl_("G", &c__0, &c__0, &smlnum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    } else if (ibscl == 2) {
	dlascl_("G", &c__0, &c__0, &bignum, &bnrm, &scllen, nrhs, &b[b_offset]
		, ldb, info);
    }
L50:
    work[1] = (double) wsize;
    return 0;
    //
    //    End of DGELS
    //
} // dgels_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DTRTRS
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DTRTRS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dtrtrs.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dtrtrs.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dtrtrs.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DTRTRS( UPLO, TRANS, DIAG, N, NRHS, A, LDA, B, LDB,
//                         INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          DIAG, TRANS, UPLO
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
//> DTRTRS solves a triangular system of the form
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
//>          B is DOUBLE PRECISION array, dimension (LDB,NRHS)
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
//> \ingroup doubleOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dtrtrs_(char *uplo, char *trans, char *diag, int *n, int
	*nrhs, double *a, int *lda, double *b, int *ldb, int *info)
{
    // Table of constant values
    double c_b12 = 1.;

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, i__1;

    // Local variables
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dtrsm_(char *, char *, char *, char *, int *,
	    int *, double *, double *, int *, double *, int *), xerbla_(char *
	    , int *);
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
	xerbla_("DTRTRS", &i__1);
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
	    if (a[*info + *info * a_dim1] == 0.) {
		return 0;
	    }
// L10:
	}
    }
    *info = 0;
    //
    //    Solve A * x = b  or  A**T * x = b.
    //
    dtrsm_("Left", uplo, trans, diag, n, nrhs, &c_b12, &a[a_offset], lda, &b[
	    b_offset], ldb);
    return 0;
    //
    //    End of DTRTRS
    //
} // dtrtrs_

