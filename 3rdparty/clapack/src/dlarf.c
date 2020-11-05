/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DGER
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DGER(M,N,ALPHA,X,INCX,Y,INCY,A,LDA)
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION ALPHA
//      INTEGER INCX,INCY,LDA,M,N
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
//> DGER   performs the rank 1 operation
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
//>          ALPHA is DOUBLE PRECISION.
//>           On entry, ALPHA specifies the scalar alpha.
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension at least
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
//>          Y is DOUBLE PRECISION array, dimension at least
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
//>          A is DOUBLE PRECISION array, dimension ( LDA, N )
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
//> \ingroup double_blas_level2
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
/* Subroutine */ int dger_(int *m, int *n, double *alpha, double *x, int *
	incx, double *y, int *incy, double *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, ix, jy, kx, info;
    double temp;
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
	xerbla_("DGER  ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*m == 0 || *n == 0 || *alpha == 0.) {
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
	    if (y[jy] != 0.) {
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
	    if (y[jy] != 0.) {
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
    //    End of DGER  .
    //
} // dger_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARF applies an elementary reflector to a general rectangular matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARF( SIDE, M, N, V, INCV, TAU, C, LDC, WORK )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE
//      INTEGER            INCV, LDC, M, N
//      DOUBLE PRECISION   TAU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   C( LDC, * ), V( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLARF applies a real elementary reflector H to a real m by n matrix
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
//>          V is DOUBLE PRECISION array, dimension
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
//>          TAU is DOUBLE PRECISION
//>          The value tau in the representation of H.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC,N)
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
//>          WORK is DOUBLE PRECISION array, dimension
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
//> \ingroup doubleOTHERauxiliary
//
// =====================================================================
/* Subroutine */ int dlarf_(char *side, int *m, int *n, double *v, int *incv,
	double *tau, double *c__, int *ldc, double *work)
{
    // Table of constant values
    double c_b4 = 1.;
    double c_b5 = 0.;
    int c__1 = 1;

    // System generated locals
    int c_dim1, c_offset;
    double d__1;

    // Local variables
    int i__;
    int applyleft;
    extern /* Subroutine */ int dger_(int *, int *, double *, double *, int *,
	     double *, int *, double *, int *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, int *, int *, double *, double
	    *, int *, double *, int *, double *, double *, int *);
    int lastc, lastv;
    extern int iladlc_(int *, int *, double *, int *), iladlr_(int *, int *,
	    double *, int *);

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
    if (*tau != 0.) {
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
	while(lastv > 0 && v[i__] == 0.) {
	    --lastv;
	    i__ -= *incv;
	}
	if (applyleft) {
	    //    Scan for the last non-zero column in C(1:lastv,:).
	    lastc = iladlc_(&lastv, n, &c__[c_offset], ldc);
	} else {
	    //    Scan for the last non-zero row in C(:,1:lastv).
	    lastc = iladlr_(m, &lastv, &c__[c_offset], ldc);
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
	    dgemv_("Transpose", &lastv, &lastc, &c_b4, &c__[c_offset], ldc, &
		    v[1], incv, &c_b5, &work[1], &c__1);
	    //
	    //          C(1:lastv,1:lastc) := C(...) - v(1:lastv,1) * w(1:lastc,1)**T
	    //
	    d__1 = -(*tau);
	    dger_(&lastv, &lastc, &d__1, &v[1], incv, &work[1], &c__1, &c__[
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
	    dgemv_("No transpose", &lastc, &lastv, &c_b4, &c__[c_offset], ldc,
		     &v[1], incv, &c_b5, &work[1], &c__1);
	    //
	    //          C(1:lastc,1:lastv) := C(...) - w(1:lastc,1) * v(1:lastv,1)**T
	    //
	    d__1 = -(*tau);
	    dger_(&lastc, &lastv, &d__1, &work[1], &c__1, &v[1], incv, &c__[
		    c_offset], ldc);
	}
    }
    return 0;
    //
    //    End of DLARF
    //
} // dlarf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b ILADLC scans a matrix for its last non-zero column.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download ILADLC + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/iladlc.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/iladlc.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/iladlc.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER FUNCTION ILADLC( M, N, A, LDA )
//
//      .. Scalar Arguments ..
//      INTEGER            M, N, LDA
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
//> ILADLC scans A for its last non-zero column.
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
//> \ingroup OTHERauxiliary
//
// =====================================================================
int iladlc_(int *m, int *n, double *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, ret_val, i__1;

    // Local variables
    int i__;

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
    if (*n == 0) {
	ret_val = *n;
    } else if (a[*n * a_dim1 + 1] != 0. || a[*m + *n * a_dim1] != 0.) {
	ret_val = *n;
    } else {
	//    Now scan each column from the end, returning with the first non-zero.
	for (ret_val = *n; ret_val >= 1; --ret_val) {
	    i__1 = *m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		if (a[i__ + ret_val * a_dim1] != 0.) {
		    return ret_val;
		}
	    }
	}
    }
    return ret_val;
} // iladlc_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b ILADLR scans a matrix for its last non-zero row.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download ILADLR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/iladlr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/iladlr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/iladlr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER FUNCTION ILADLR( M, N, A, LDA )
//
//      .. Scalar Arguments ..
//      INTEGER            M, N, LDA
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
//> ILADLR scans A for its last non-zero row.
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
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
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
//> \ingroup OTHERauxiliary
//
// =====================================================================
int iladlr_(int *m, int *n, double *a, int *lda)
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
    } else if (a[*m + a_dim1] != 0. || a[*m + *n * a_dim1] != 0.) {
	ret_val = *m;
    } else {
	//    Scan up each column tracking the last zero row seen.
	ret_val = 0;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__ = *m;
	    while(a[max(i__,1) + j * a_dim1] == 0. && i__ >= 1) {
		--i__;
	    }
	    ret_val = max(ret_val,i__);
	}
    }
    return ret_val;
} // iladlr_

