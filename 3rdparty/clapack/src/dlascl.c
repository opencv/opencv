/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLASCL multiplies a general rectangular matrix by a real scalar defined as cto/cfrom.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASCL + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlascl.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlascl.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlascl.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASCL( TYPE, KL, KU, CFROM, CTO, M, N, A, LDA, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          TYPE
//      INTEGER            INFO, KL, KU, LDA, M, N
//      DOUBLE PRECISION   CFROM, CTO
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
//> DLASCL multiplies the M by N real matrix A by the real scalar
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
//>                  bandwidth KU. See DGBTRF for storage details.
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
//>          CFROM is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] CTO
//> \verbatim
//>          CTO is DOUBLE PRECISION
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
/* Subroutine */ int dlascl_(char *type__, int *kl, int *ku, double *cfrom,
	double *cto, int *m, int *n, double *a, int *lda, int *info)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4, i__5;

    // Local variables
    int i__, j, k1, k2, k3, k4;
    double mul, cto1;
    int done;
    double ctoc;
    extern int lsame_(char *, char *);
    int itype;
    double cfrom1;
    extern double dlamch_(char *);
    double cfromc;
    extern int disnan_(double *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    double bignum, smlnum;

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
    } else if (*cfrom == 0. || disnan_(cfrom)) {
	*info = -4;
    } else if (disnan_(cto)) {
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
	xerbla_("DLASCL", &i__1);
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
    smlnum = dlamch_("S");
    bignum = 1. / smlnum;
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
	    cfromc = 1.;
	} else if (abs(cfrom1) > abs(ctoc) && ctoc != 0.) {
	    mul = smlnum;
	    done = FALSE_;
	    cfromc = cfrom1;
	} else if (abs(cto1) > abs(cfromc)) {
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
    //    End of DLASCL
    //
} // dlascl_

