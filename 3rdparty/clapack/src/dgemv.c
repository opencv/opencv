/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DGEMV
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION ALPHA,BETA
//      INTEGER INCX,INCY,LDA,M,N
//      CHARACTER TRANS
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEMV  performs one of the matrix-vector operations
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
//>          ALPHA is DOUBLE PRECISION.
//>           On entry, ALPHA specifies the scalar alpha.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension ( LDA, N )
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
//>          X is DOUBLE PRECISION array, dimension at least
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
//>          BETA is DOUBLE PRECISION.
//>           On entry, BETA specifies the scalar beta. When BETA is
//>           supplied as zero then Y need not be set on input.
//> \endverbatim
//>
//> \param[in,out] Y
//> \verbatim
//>          Y is DOUBLE PRECISION array, dimension at least
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
//> \ingroup double_blas_level2
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
/* Subroutine */ int dgemv_(char *trans, int *m, int *n, double *alpha,
	double *a, int *lda, double *x, int *incx, double *beta, double *y,
	int *incy)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, ix, iy, jx, jy, kx, ky, info;
    double temp;
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
	xerbla_("DGEMV ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*m == 0 || *n == 0 || *alpha == 0. && *beta == 1.) {
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
    if (*beta != 1.) {
	if (*incy == 1) {
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.;
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
	    if (*beta == 0.) {
		i__1 = leny;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.;
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
    if (*alpha == 0.) {
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
		temp = 0.;
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
		temp = 0.;
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
    //    End of DGEMV .
    //
} // dgemv_

