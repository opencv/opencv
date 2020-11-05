/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b SGEMM
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE SGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
//
//      .. Scalar Arguments ..
//      REAL ALPHA,BETA
//      INTEGER K,LDA,LDB,LDC,M,N
//      CHARACTER TRANSA,TRANSB
//      ..
//      .. Array Arguments ..
//      REAL A(LDA,*),B(LDB,*),C(LDC,*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SGEMM  performs one of the matrix-matrix operations
//>
//>    C := alpha*op( A )*op( B ) + beta*C,
//>
//> where  op( X ) is one of
//>
//>    op( X ) = X   or   op( X ) = X**T,
//>
//> alpha and beta are scalars, and A, B and C are matrices, with op( A )
//> an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] TRANSA
//> \verbatim
//>          TRANSA is CHARACTER*1
//>           On entry, TRANSA specifies the form of op( A ) to be used in
//>           the matrix multiplication as follows:
//>
//>              TRANSA = 'N' or 'n',  op( A ) = A.
//>
//>              TRANSA = 'T' or 't',  op( A ) = A**T.
//>
//>              TRANSA = 'C' or 'c',  op( A ) = A**T.
//> \endverbatim
//>
//> \param[in] TRANSB
//> \verbatim
//>          TRANSB is CHARACTER*1
//>           On entry, TRANSB specifies the form of op( B ) to be used in
//>           the matrix multiplication as follows:
//>
//>              TRANSB = 'N' or 'n',  op( B ) = B.
//>
//>              TRANSB = 'T' or 't',  op( B ) = B**T.
//>
//>              TRANSB = 'C' or 'c',  op( B ) = B**T.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>           On entry,  M  specifies  the number  of rows  of the  matrix
//>           op( A )  and of the  matrix  C.  M  must  be at least  zero.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry,  N  specifies the number  of columns of the matrix
//>           op( B ) and the number of columns of the matrix C. N must be
//>           at least zero.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>           On entry,  K  specifies  the number of columns of the matrix
//>           op( A ) and the number of rows of the matrix op( B ). K must
//>           be at least  zero.
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
//>           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
//>           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
//>           part of the array  A  must contain the matrix  A,  otherwise
//>           the leading  k by m  part of the array  A  must contain  the
//>           matrix A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
//>           LDA must be at least  max( 1, m ), otherwise  LDA must be at
//>           least  max( 1, k ).
//> \endverbatim
//>
//> \param[in] B
//> \verbatim
//>          B is REAL array, dimension ( LDB, kb ), where kb is
//>           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
//>           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
//>           part of the array  B  must contain the matrix  B,  otherwise
//>           the leading  n by k  part of the array  B  must contain  the
//>           matrix B.
//> \endverbatim
//>
//> \param[in] LDB
//> \verbatim
//>          LDB is INTEGER
//>           On entry, LDB specifies the first dimension of B as declared
//>           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
//>           LDB must be at least  max( 1, k ), otherwise  LDB must be at
//>           least  max( 1, n ).
//> \endverbatim
//>
//> \param[in] BETA
//> \verbatim
//>          BETA is REAL
//>           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
//>           supplied as zero then C need not be set on input.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is REAL array, dimension ( LDC, N )
//>           Before entry, the leading  m by n  part of the array  C must
//>           contain the matrix  C,  except when  beta  is zero, in which
//>           case C need not be set on entry.
//>           On exit, the array  C  is overwritten by the  m by n  matrix
//>           ( alpha*op( A )*op( B ) + beta*C ).
//> \endverbatim
//>
//> \param[in] LDC
//> \verbatim
//>          LDC is INTEGER
//>           On entry, LDC specifies the first dimension of C as declared
//>           in  the  calling  (sub)  program.   LDC  must  be  at  least
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
/* Subroutine */ int sgemm_(char *transa, char *transb, int *m, int *n, int *
	k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta,
	float *c__, int *ldc)
{
    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2,
	    i__3;

    // Local variables
    int i__, j, l, info;
    int nota, notb;
    float temp;
    int ncola;
    extern int lsame_(char *, char *);
    int nrowa, nrowb;
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
    //    Set  NOTA  and  NOTB  as  true if  A  and  B  respectively are not
    //    transposed and set  NROWA, NCOLA and  NROWB  as the number of rows
    //    and  columns of  A  and the  number of  rows  of  B  respectively.
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;

    // Function Body
    nota = lsame_(transa, "N");
    notb = lsame_(transb, "N");
    if (nota) {
	nrowa = *m;
	ncola = *k;
    } else {
	nrowa = *k;
	ncola = *m;
    }
    if (notb) {
	nrowb = *k;
    } else {
	nrowb = *n;
    }
    //
    //    Test the input parameters.
    //
    info = 0;
    if (! nota && ! lsame_(transa, "C") && ! lsame_(transa, "T")) {
	info = 1;
    } else if (! notb && ! lsame_(transb, "C") && ! lsame_(transb, "T")) {
	info = 2;
    } else if (*m < 0) {
	info = 3;
    } else if (*n < 0) {
	info = 4;
    } else if (*k < 0) {
	info = 5;
    } else if (*lda < max(1,nrowa)) {
	info = 8;
    } else if (*ldb < max(1,nrowb)) {
	info = 10;
    } else if (*ldc < max(1,*m)) {
	info = 13;
    }
    if (info != 0) {
	xerbla_("SGEMM ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*m == 0 || *n == 0 || (*alpha == 0.f || *k == 0) && *beta == 1.f) {
	return 0;
    }
    //
    //    And if  alpha.eq.zero.
    //
    if (*alpha == 0.f) {
	if (*beta == 0.f) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = 0.f;
// L10:
		}
// L20:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L30:
		}
// L40:
	    }
	}
	return 0;
    }
    //
    //    Start the operations.
    //
    if (notb) {
	if (nota) {
	    //
	    //          Form  C := alpha*A*B + beta*C.
	    //
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.f) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
// L50:
		    }
		} else if (*beta != 1.f) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L60:
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    temp = *alpha * b[l + j * b_dim1];
		    i__3 = *m;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
// L70:
		    }
// L80:
		}
// L90:
	    }
	} else {
	    //
	    //          Form  C := alpha*A**T*B + beta*C
	    //
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.f;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * b[l + j * b_dim1];
// L100:
		    }
		    if (*beta == 0.f) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
// L110:
		}
// L120:
	    }
	}
    } else {
	if (nota) {
	    //
	    //          Form  C := alpha*A*B**T + beta*C
	    //
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (*beta == 0.f) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = 0.f;
// L130:
		    }
		} else if (*beta != 1.f) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1];
// L140:
		    }
		}
		i__2 = *k;
		for (l = 1; l <= i__2; ++l) {
		    temp = *alpha * b[j + l * b_dim1];
		    i__3 = *m;
		    for (i__ = 1; i__ <= i__3; ++i__) {
			c__[i__ + j * c_dim1] += temp * a[i__ + l * a_dim1];
// L150:
		    }
// L160:
		}
// L170:
	    }
	} else {
	    //
	    //          Form  C := alpha*A**T*B**T + beta*C
	    //
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *m;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp = 0.f;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp += a[l + i__ * a_dim1] * b[j + l * b_dim1];
// L180:
		    }
		    if (*beta == 0.f) {
			c__[i__ + j * c_dim1] = *alpha * temp;
		    } else {
			c__[i__ + j * c_dim1] = *alpha * temp + *beta * c__[
				i__ + j * c_dim1];
		    }
// L190:
		}
// L200:
	    }
	}
    }
    return 0;
    //
    //    End of SGEMM .
    //
} // sgemm_

