/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DAXPY
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION DA
//      INTEGER INCX,INCY,N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION DX(*),DY(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DAXPY constant times a vector plus a vector.
//>    uses unrolled loops for increments equal to one.
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
//> \param[in] DA
//> \verbatim
//>          DA is DOUBLE PRECISION
//>           On entry, DA specifies the scalar alpha.
//> \endverbatim
//>
//> \param[in] DX
//> \verbatim
//>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>         storage spacing between elements of DX
//> \endverbatim
//>
//> \param[in,out] DY
//> \verbatim
//>          DY is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
//> \endverbatim
//>
//> \param[in] INCY
//> \verbatim
//>          INCY is INTEGER
//>         storage spacing between elements of DY
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
//> \ingroup double_blas_level1
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>     jack dongarra, linpack, 3/11/78.
//>     modified 12/3/93, array(1) declarations changed to array(*)
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int daxpy_(int *n, double *da, double *dx, int *incx, double
	*dy, int *incy)
{
    // System generated locals
    int i__1;

    // Local variables
    int i__, m, ix, iy, mp1;

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
    --dy;
    --dx;

    // Function Body
    if (*n <= 0) {
	return 0;
    }
    if (*da == 0.) {
	return 0;
    }
    if (*incx == 1 && *incy == 1) {
	//
	//       code for both increments equal to 1
	//
	//
	//       clean-up loop
	//
	m = *n % 4;
	if (m != 0) {
	    i__1 = m;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		dy[i__] += *da * dx[i__];
	    }
	}
	if (*n < 4) {
	    return 0;
	}
	mp1 = m + 1;
	i__1 = *n;
	for (i__ = mp1; i__ <= i__1; i__ += 4) {
	    dy[i__] += *da * dx[i__];
	    dy[i__ + 1] += *da * dx[i__ + 1];
	    dy[i__ + 2] += *da * dx[i__ + 2];
	    dy[i__ + 3] += *da * dx[i__ + 3];
	}
    } else {
	//
	//       code for unequal increments or equal increments
	//         not equal to 1
	//
	ix = 1;
	iy = 1;
	if (*incx < 0) {
	    ix = (-(*n) + 1) * *incx + 1;
	}
	if (*incy < 0) {
	    iy = (-(*n) + 1) * *incy + 1;
	}
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    dy[iy] += *da * dx[ix];
	    ix += *incx;
	    iy += *incy;
	}
    }
    return 0;
} // daxpy_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEBAK
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEBAK + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgebak.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgebak.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgebak.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEBAK( JOB, SIDE, N, ILO, IHI, SCALE, M, V, LDV,
//                         INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          JOB, SIDE
//      INTEGER            IHI, ILO, INFO, LDV, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   SCALE( * ), V( LDV, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEBAK forms the right or left eigenvectors of a real general matrix
//> by backward transformation on the computed eigenvectors of the
//> balanced matrix output by DGEBAL.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOB
//> \verbatim
//>          JOB is CHARACTER*1
//>          Specifies the type of backward transformation required:
//>          = 'N': do nothing, return immediately;
//>          = 'P': do backward transformation for permutation only;
//>          = 'S': do backward transformation for scaling only;
//>          = 'B': do backward transformations for both permutation and
//>                 scaling.
//>          JOB must be the same as the argument JOB supplied to DGEBAL.
//> \endverbatim
//>
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'R':  V contains right eigenvectors;
//>          = 'L':  V contains left eigenvectors.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of rows of the matrix V.  N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>          The integers ILO and IHI determined by DGEBAL.
//>          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
//> \endverbatim
//>
//> \param[in] SCALE
//> \verbatim
//>          SCALE is DOUBLE PRECISION array, dimension (N)
//>          Details of the permutation and scaling factors, as returned
//>          by DGEBAL.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of columns of the matrix V.  M >= 0.
//> \endverbatim
//>
//> \param[in,out] V
//> \verbatim
//>          V is DOUBLE PRECISION array, dimension (LDV,M)
//>          On entry, the matrix of right or left eigenvectors to be
//>          transformed, as returned by DHSEIN or DTREVC.
//>          On exit, V is overwritten by the transformed eigenvectors.
//> \endverbatim
//>
//> \param[in] LDV
//> \verbatim
//>          LDV is INTEGER
//>          The leading dimension of the array V. LDV >= max(1,N).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
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
//> \ingroup doubleGEcomputational
//
// =====================================================================
/* Subroutine */ int dgebak_(char *job, char *side, int *n, int *ilo, int *
	ihi, double *scale, int *m, double *v, int *ldv, int *info)
{
    // System generated locals
    int v_dim1, v_offset, i__1;

    // Local variables
    int i__, k;
    double s;
    int ii;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dswap_(int *, double *, int *, double *, int *
	    );
    int leftv;
    extern /* Subroutine */ int xerbla_(char *, int *);
    int rightv;

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
    //    Decode and Test the input parameters
    //
    // Parameter adjustments
    --scale;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;

    // Function Body
    rightv = lsame_(side, "R");
    leftv = lsame_(side, "L");
    *info = 0;
    if (! lsame_(job, "N") && ! lsame_(job, "P") && ! lsame_(job, "S") && !
	    lsame_(job, "B")) {
	*info = -1;
    } else if (! rightv && ! leftv) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -4;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -5;
    } else if (*m < 0) {
	*info = -7;
    } else if (*ldv < max(1,*n)) {
	*info = -9;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEBAK", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	return 0;
    }
    if (*m == 0) {
	return 0;
    }
    if (lsame_(job, "N")) {
	return 0;
    }
    if (*ilo == *ihi) {
	goto L30;
    }
    //
    //    Backward balance
    //
    if (lsame_(job, "S") || lsame_(job, "B")) {
	if (rightv) {
	    i__1 = *ihi;
	    for (i__ = *ilo; i__ <= i__1; ++i__) {
		s = scale[i__];
		dscal_(m, &s, &v[i__ + v_dim1], ldv);
// L10:
	    }
	}
	if (leftv) {
	    i__1 = *ihi;
	    for (i__ = *ilo; i__ <= i__1; ++i__) {
		s = 1. / scale[i__];
		dscal_(m, &s, &v[i__ + v_dim1], ldv);
// L20:
	    }
	}
    }
    //
    //    Backward permutation
    //
    //    For  I = ILO-1 step -1 until 1,
    //             IHI+1 step 1 until N do --
    //
L30:
    if (lsame_(job, "P") || lsame_(job, "B")) {
	if (rightv) {
	    i__1 = *n;
	    for (ii = 1; ii <= i__1; ++ii) {
		i__ = ii;
		if (i__ >= *ilo && i__ <= *ihi) {
		    goto L40;
		}
		if (i__ < *ilo) {
		    i__ = *ilo - ii;
		}
		k = (int) scale[i__];
		if (k == i__) {
		    goto L40;
		}
		dswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
L40:
		;
	    }
	}
	if (leftv) {
	    i__1 = *n;
	    for (ii = 1; ii <= i__1; ++ii) {
		i__ = ii;
		if (i__ >= *ilo && i__ <= *ihi) {
		    goto L50;
		}
		if (i__ < *ilo) {
		    i__ = *ilo - ii;
		}
		k = (int) scale[i__];
		if (k == i__) {
		    goto L50;
		}
		dswap_(m, &v[i__ + v_dim1], ldv, &v[k + v_dim1], ldv);
L50:
		;
	    }
	}
    }
    return 0;
    //
    //    End of DGEBAK
    //
} // dgebak_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEBAL
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEBAL + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgebal.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgebal.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgebal.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEBAL( JOB, N, A, LDA, ILO, IHI, SCALE, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          JOB
//      INTEGER            IHI, ILO, INFO, LDA, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), SCALE( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEBAL balances a general real matrix A.  This involves, first,
//> permuting A by a similarity transformation to isolate eigenvalues
//> in the first 1 to ILO-1 and last IHI+1 to N elements on the
//> diagonal; and second, applying a diagonal similarity transformation
//> to rows and columns ILO to IHI to make the rows and columns as
//> close in norm as possible.  Both steps are optional.
//>
//> Balancing may reduce the 1-norm of the matrix, and improve the
//> accuracy of the computed eigenvalues and/or eigenvectors.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOB
//> \verbatim
//>          JOB is CHARACTER*1
//>          Specifies the operations to be performed on A:
//>          = 'N':  none:  simply set ILO = 1, IHI = N, SCALE(I) = 1.0
//>                  for i = 1,...,N;
//>          = 'P':  permute only;
//>          = 'S':  scale only;
//>          = 'B':  both permute and scale.
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
//>          On entry, the input matrix A.
//>          On exit,  A is overwritten by the balanced matrix.
//>          If JOB = 'N', A is not referenced.
//>          See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//> \param[out] IHI
//> \verbatim
//>          IHI is INTEGER
//>          ILO and IHI are set to integers such that on exit
//>          A(i,j) = 0 if i > j and j = 1,...,ILO-1 or I = IHI+1,...,N.
//>          If JOB = 'N' or 'S', ILO = 1 and IHI = N.
//> \endverbatim
//>
//> \param[out] SCALE
//> \verbatim
//>          SCALE is DOUBLE PRECISION array, dimension (N)
//>          Details of the permutations and scaling factors applied to
//>          A.  If P(j) is the index of the row and column interchanged
//>          with row and column j and D(j) is the scaling factor
//>          applied to row and column j, then
//>          SCALE(j) = P(j)    for j = 1,...,ILO-1
//>                   = D(j)    for j = ILO,...,IHI
//>                   = P(j)    for j = IHI+1,...,N.
//>          The order in which the interchanges are made is N to IHI+1,
//>          then 1 to ILO-1.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
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
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The permutations consist of row and column interchanges which put
//>  the matrix in the form
//>
//>             ( T1   X   Y  )
//>     P A P = (  0   B   Z  )
//>             (  0   0   T2 )
//>
//>  where T1 and T2 are upper triangular matrices whose eigenvalues lie
//>  along the diagonal.  The column indices ILO and IHI mark the starting
//>  and ending columns of the submatrix B. Balancing consists of applying
//>  a diagonal similarity transformation inv(D) * B * D to make the
//>  1-norms of each row of B and its corresponding column nearly equal.
//>  The output matrix is
//>
//>     ( T1     X*D          Y    )
//>     (  0  inv(D)*B*D  inv(D)*Z ).
//>     (  0      0           T2   )
//>
//>  Information about the permutations P and the diagonal matrix D is
//>  returned in the vector SCALE.
//>
//>  This subroutine is based on the EISPACK routine BALANC.
//>
//>  Modified by Tzu-Yi Chen, Computer Science Division, University of
//>    California at Berkeley, USA
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgebal_(char *job, int *n, double *a, int *lda, int *ilo,
	 int *ihi, double *scale, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2;
    double d__1, d__2;

    // Local variables
    double c__, f, g;
    int i__, j, k, l, m;
    double r__, s, ca, ra;
    int ica, ira, iexc;
    extern double dnrm2_(int *, double *, int *);
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dswap_(int *, double *, int *, double *, int *
	    );
    double sfmin1, sfmin2, sfmax1, sfmax2;
    extern double dlamch_(char *);
    extern int idamax_(int *, double *, int *);
    extern int disnan_(double *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    int noconv;

    //
    // -- LAPACK computational routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
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
    //    Test the input parameters
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --scale;

    // Function Body
    *info = 0;
    if (! lsame_(job, "N") && ! lsame_(job, "P") && ! lsame_(job, "S") && !
	    lsame_(job, "B")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEBAL", &i__1);
	return 0;
    }
    k = 1;
    l = *n;
    if (*n == 0) {
	goto L210;
    }
    if (lsame_(job, "N")) {
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    scale[i__] = 1.;
// L10:
	}
	goto L210;
    }
    if (lsame_(job, "S")) {
	goto L120;
    }
    //
    //    Permutation to isolate eigenvalues if possible
    //
    goto L50;
    //
    //    Row and column exchange.
    //
L20:
    scale[m] = (double) j;
    if (j == m) {
	goto L30;
    }
    dswap_(&l, &a[j * a_dim1 + 1], &c__1, &a[m * a_dim1 + 1], &c__1);
    i__1 = *n - k + 1;
    dswap_(&i__1, &a[j + k * a_dim1], lda, &a[m + k * a_dim1], lda);
L30:
    switch (iexc) {
	case 1:  goto L40;
	case 2:  goto L80;
    }
    //
    //    Search for rows isolating an eigenvalue and push them down.
    //
L40:
    if (l == 1) {
	goto L210;
    }
    --l;
L50:
    for (j = l; j >= 1; --j) {
	i__1 = l;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (i__ == j) {
		goto L60;
	    }
	    if (a[j + i__ * a_dim1] != 0.) {
		goto L70;
	    }
L60:
	    ;
	}
	m = l;
	iexc = 1;
	goto L20;
L70:
	;
    }
    goto L90;
    //
    //    Search for columns isolating an eigenvalue and push them left.
    //
L80:
    ++k;
L90:
    i__1 = l;
    for (j = k; j <= i__1; ++j) {
	i__2 = l;
	for (i__ = k; i__ <= i__2; ++i__) {
	    if (i__ == j) {
		goto L100;
	    }
	    if (a[i__ + j * a_dim1] != 0.) {
		goto L110;
	    }
L100:
	    ;
	}
	m = k;
	iexc = 2;
	goto L20;
L110:
	;
    }
L120:
    i__1 = l;
    for (i__ = k; i__ <= i__1; ++i__) {
	scale[i__] = 1.;
// L130:
    }
    if (lsame_(job, "P")) {
	goto L210;
    }
    //
    //    Balance the submatrix in rows K to L.
    //
    //    Iterative loop for norm reduction
    //
    sfmin1 = dlamch_("S") / dlamch_("P");
    sfmax1 = 1. / sfmin1;
    sfmin2 = sfmin1 * 2.;
    sfmax2 = 1. / sfmin2;
L140:
    noconv = FALSE_;
    i__1 = l;
    for (i__ = k; i__ <= i__1; ++i__) {
	i__2 = l - k + 1;
	c__ = dnrm2_(&i__2, &a[k + i__ * a_dim1], &c__1);
	i__2 = l - k + 1;
	r__ = dnrm2_(&i__2, &a[i__ + k * a_dim1], lda);
	ica = idamax_(&l, &a[i__ * a_dim1 + 1], &c__1);
	ca = (d__1 = a[ica + i__ * a_dim1], abs(d__1));
	i__2 = *n - k + 1;
	ira = idamax_(&i__2, &a[i__ + k * a_dim1], lda);
	ra = (d__1 = a[i__ + (ira + k - 1) * a_dim1], abs(d__1));
	//
	//       Guard against zero C or R due to underflow.
	//
	if (c__ == 0. || r__ == 0.) {
	    goto L200;
	}
	g = r__ / 2.;
	f = 1.;
	s = c__ + r__;
L160:
	// Computing MAX
	d__1 = max(f,c__);
	// Computing MIN
	d__2 = min(r__,g);
	if (c__ >= g || max(d__1,ca) >= sfmax2 || min(d__2,ra) <= sfmin2) {
	    goto L170;
	}
	d__1 = c__ + f + ca + r__ + g + ra;
	if (disnan_(&d__1)) {
	    //
	    //          Exit if NaN to avoid infinite loop
	    //
	    *info = -3;
	    i__2 = -(*info);
	    xerbla_("DGEBAL", &i__2);
	    return 0;
	}
	f *= 2.;
	c__ *= 2.;
	ca *= 2.;
	r__ /= 2.;
	g /= 2.;
	ra /= 2.;
	goto L160;
L170:
	g = c__ / 2.;
L180:
	// Computing MIN
	d__1 = min(f,c__), d__1 = min(d__1,g);
	if (g < r__ || max(r__,ra) >= sfmax2 || min(d__1,ca) <= sfmin2) {
	    goto L190;
	}
	f /= 2.;
	c__ /= 2.;
	g /= 2.;
	ca /= 2.;
	r__ *= 2.;
	ra *= 2.;
	goto L180;
	//
	//       Now balance.
	//
L190:
	if (c__ + r__ >= s * .95) {
	    goto L200;
	}
	if (f < 1. && scale[i__] < 1.) {
	    if (f * scale[i__] <= sfmin1) {
		goto L200;
	    }
	}
	if (f > 1. && scale[i__] > 1.) {
	    if (scale[i__] >= sfmax1 / f) {
		goto L200;
	    }
	}
	g = 1. / f;
	scale[i__] *= f;
	noconv = TRUE_;
	i__2 = *n - k + 1;
	dscal_(&i__2, &g, &a[i__ + k * a_dim1], lda);
	dscal_(&l, &f, &a[i__ * a_dim1 + 1], &c__1);
L200:
	;
    }
    if (noconv) {
	goto L140;
    }
L210:
    *ilo = k;
    *ihi = l;
    return 0;
    //
    //    End of DGEBAL
    //
} // dgebal_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief <b> DGEEV computes the eigenvalues and, optionally, the left and/or right eigenvectors for GE matrices</b>
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEEV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgeev.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgeev.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgeev.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEEV( JOBVL, JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR,
//                        LDVR, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          JOBVL, JOBVR
//      INTEGER            INFO, LDA, LDVL, LDVR, LWORK, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), VL( LDVL, * ), VR( LDVR, * ),
//     $                   WI( * ), WORK( * ), WR( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEEV computes for an N-by-N real nonsymmetric matrix A, the
//> eigenvalues and, optionally, the left and/or right eigenvectors.
//>
//> The right eigenvector v(j) of A satisfies
//>                  A * v(j) = lambda(j) * v(j)
//> where lambda(j) is its eigenvalue.
//> The left eigenvector u(j) of A satisfies
//>               u(j)**H * A = lambda(j) * u(j)**H
//> where u(j)**H denotes the conjugate-transpose of u(j).
//>
//> The computed eigenvectors are normalized to have Euclidean norm
//> equal to 1 and largest component real.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOBVL
//> \verbatim
//>          JOBVL is CHARACTER*1
//>          = 'N': left eigenvectors of A are not computed;
//>          = 'V': left eigenvectors of A are computed.
//> \endverbatim
//>
//> \param[in] JOBVR
//> \verbatim
//>          JOBVR is CHARACTER*1
//>          = 'N': right eigenvectors of A are not computed;
//>          = 'V': right eigenvectors of A are computed.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A. N >= 0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the N-by-N matrix A.
//>          On exit, A has been overwritten.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] WR
//> \verbatim
//>          WR is DOUBLE PRECISION array, dimension (N)
//> \endverbatim
//>
//> \param[out] WI
//> \verbatim
//>          WI is DOUBLE PRECISION array, dimension (N)
//>          WR and WI contain the real and imaginary parts,
//>          respectively, of the computed eigenvalues.  Complex
//>          conjugate pairs of eigenvalues appear consecutively
//>          with the eigenvalue having the positive imaginary part
//>          first.
//> \endverbatim
//>
//> \param[out] VL
//> \verbatim
//>          VL is DOUBLE PRECISION array, dimension (LDVL,N)
//>          If JOBVL = 'V', the left eigenvectors u(j) are stored one
//>          after another in the columns of VL, in the same order
//>          as their eigenvalues.
//>          If JOBVL = 'N', VL is not referenced.
//>          If the j-th eigenvalue is real, then u(j) = VL(:,j),
//>          the j-th column of VL.
//>          If the j-th and (j+1)-st eigenvalues form a complex
//>          conjugate pair, then u(j) = VL(:,j) + i*VL(:,j+1) and
//>          u(j+1) = VL(:,j) - i*VL(:,j+1).
//> \endverbatim
//>
//> \param[in] LDVL
//> \verbatim
//>          LDVL is INTEGER
//>          The leading dimension of the array VL.  LDVL >= 1; if
//>          JOBVL = 'V', LDVL >= N.
//> \endverbatim
//>
//> \param[out] VR
//> \verbatim
//>          VR is DOUBLE PRECISION array, dimension (LDVR,N)
//>          If JOBVR = 'V', the right eigenvectors v(j) are stored one
//>          after another in the columns of VR, in the same order
//>          as their eigenvalues.
//>          If JOBVR = 'N', VR is not referenced.
//>          If the j-th eigenvalue is real, then v(j) = VR(:,j),
//>          the j-th column of VR.
//>          If the j-th and (j+1)-st eigenvalues form a complex
//>          conjugate pair, then v(j) = VR(:,j) + i*VR(:,j+1) and
//>          v(j+1) = VR(:,j) - i*VR(:,j+1).
//> \endverbatim
//>
//> \param[in] LDVR
//> \verbatim
//>          LDVR is INTEGER
//>          The leading dimension of the array VR.  LDVR >= 1; if
//>          JOBVR = 'V', LDVR >= N.
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
//>          The dimension of the array WORK.  LWORK >= max(1,3*N), and
//>          if JOBVL = 'V' or JOBVR = 'V', LWORK >= 4*N.  For good
//>          performance, LWORK must generally be larger.
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
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
//>          > 0:  if INFO = i, the QR algorithm failed to compute all the
//>                eigenvalues, and no eigenvectors have been computed;
//>                elements i+1:N of WR and WI contain eigenvalues which
//>                have converged.
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
// @precisions fortran d -> s
//
//> \ingroup doubleGEeigen
//
// =====================================================================
/* Subroutine */ int dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *
	lda, double *wr, double *wi, double *vl, int *ldvl, double *vr, int *
	ldvr, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__0 = 0;
    int c_n1 = -1;

    // System generated locals
    int a_dim1, a_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1, i__2,
	    i__3;
    double d__1, d__2;

    // Local variables
    int i__, k;
    double r__, cs, sn;
    int ihi;
    double scl;
    int ilo;
    double dum[1], eps;
    int lwork_trevc__, ibal;
    char side[1+1]={'\0'};
    double anrm;
    int ierr, itau;
    extern /* Subroutine */ int drot_(int *, double *, int *, double *, int *,
	     double *, double *);
    int iwrk, nout;
    extern double dnrm2_(int *, double *, int *);
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    extern int lsame_(char *, char *);
    extern double dlapy2_(double *, double *);
    extern /* Subroutine */ int dlabad_(double *, double *), dgebak_(char *,
	    char *, int *, int *, int *, double *, int *, double *, int *,
	    int *), dgebal_(char *, int *, double *, int *, int *, int *,
	    double *, int *);
    int scalea;
    extern double dlamch_(char *);
    double cscale;
    extern double dlange_(char *, int *, int *, double *, int *, double *);
    extern /* Subroutine */ int dgehrd_(int *, int *, int *, double *, int *,
	    double *, double *, int *, int *), dlascl_(char *, int *, int *,
	    double *, double *, int *, int *, double *, int *, int *);
    extern int idamax_(int *, double *, int *);
    extern /* Subroutine */ int dlacpy_(char *, int *, int *, double *, int *,
	     double *, int *), dlartg_(double *, double *, double *, double *,
	     double *), xerbla_(char *, int *);
    int select[1];
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    double bignum;
    extern /* Subroutine */ int dorghr_(int *, int *, int *, double *, int *,
	    double *, double *, int *, int *), dhseqr_(char *, char *, int *,
	    int *, int *, double *, int *, double *, double *, double *, int *
	    , double *, int *, int *);
    int minwrk, maxwrk;
    int wantvl;
    double smlnum;
    int hswork;
    int lquery, wantvr;
    extern /* Subroutine */ int dtrevc3_(char *, char *, int *, int *, double
	    *, int *, double *, int *, double *, int *, int *, int *, double *
	    , int *, int *);

    //
    // -- LAPACK driver routine (version 3.7.0) --
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
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --wr;
    --wi;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;

    // Function Body
    *info = 0;
    lquery = *lwork == -1;
    wantvl = lsame_(jobvl, "V");
    wantvr = lsame_(jobvr, "V");
    if (! wantvl && ! lsame_(jobvl, "N")) {
	*info = -1;
    } else if (! wantvr && ! lsame_(jobvr, "N")) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*ldvl < 1 || wantvl && *ldvl < *n) {
	*info = -9;
    } else if (*ldvr < 1 || wantvr && *ldvr < *n) {
	*info = -11;
    }
    //
    //    Compute workspace
    //     (Note: Comments in the code beginning "Workspace:" describe the
    //      minimal amount of workspace needed at that point in the code,
    //      as well as the preferred amount for good performance.
    //      NB refers to the optimal block size for the immediately
    //      following subroutine, as returned by ILAENV.
    //      HSWORK refers to the workspace preferred by DHSEQR, as
    //      calculated below. HSWORK is computed assuming ILO=1 and IHI=N,
    //      the worst case.)
    //
    if (*info == 0) {
	if (*n == 0) {
	    minwrk = 1;
	    maxwrk = 1;
	} else {
	    maxwrk = (*n << 1) + *n * ilaenv_(&c__1, "DGEHRD", " ", n, &c__1,
		    n, &c__0);
	    if (wantvl) {
		minwrk = *n << 2;
		// Computing MAX
		i__1 = maxwrk, i__2 = (*n << 1) + (*n - 1) * ilaenv_(&c__1,
			"DORGHR", " ", n, &c__1, n, &c_n1);
		maxwrk = max(i__1,i__2);
		dhseqr_("S", "V", n, &c__1, n, &a[a_offset], lda, &wr[1], &wi[
			1], &vl[vl_offset], ldvl, &work[1], &c_n1, info);
		hswork = (int) work[1];
		// Computing MAX
		i__1 = maxwrk, i__2 = *n + 1, i__1 = max(i__1,i__2), i__2 = *
			n + hswork;
		maxwrk = max(i__1,i__2);
		dtrevc3_("L", "B", select, n, &a[a_offset], lda, &vl[
			vl_offset], ldvl, &vr[vr_offset], ldvr, n, &nout, &
			work[1], &c_n1, &ierr);
		lwork_trevc__ = (int) work[1];
		// Computing MAX
		i__1 = maxwrk, i__2 = *n + lwork_trevc__;
		maxwrk = max(i__1,i__2);
		// Computing MAX
		i__1 = maxwrk, i__2 = *n << 2;
		maxwrk = max(i__1,i__2);
	    } else if (wantvr) {
		minwrk = *n << 2;
		// Computing MAX
		i__1 = maxwrk, i__2 = (*n << 1) + (*n - 1) * ilaenv_(&c__1,
			"DORGHR", " ", n, &c__1, n, &c_n1);
		maxwrk = max(i__1,i__2);
		dhseqr_("S", "V", n, &c__1, n, &a[a_offset], lda, &wr[1], &wi[
			1], &vr[vr_offset], ldvr, &work[1], &c_n1, info);
		hswork = (int) work[1];
		// Computing MAX
		i__1 = maxwrk, i__2 = *n + 1, i__1 = max(i__1,i__2), i__2 = *
			n + hswork;
		maxwrk = max(i__1,i__2);
		dtrevc3_("R", "B", select, n, &a[a_offset], lda, &vl[
			vl_offset], ldvl, &vr[vr_offset], ldvr, n, &nout, &
			work[1], &c_n1, &ierr);
		lwork_trevc__ = (int) work[1];
		// Computing MAX
		i__1 = maxwrk, i__2 = *n + lwork_trevc__;
		maxwrk = max(i__1,i__2);
		// Computing MAX
		i__1 = maxwrk, i__2 = *n << 2;
		maxwrk = max(i__1,i__2);
	    } else {
		minwrk = *n * 3;
		dhseqr_("E", "N", n, &c__1, n, &a[a_offset], lda, &wr[1], &wi[
			1], &vr[vr_offset], ldvr, &work[1], &c_n1, info);
		hswork = (int) work[1];
		// Computing MAX
		i__1 = maxwrk, i__2 = *n + 1, i__1 = max(i__1,i__2), i__2 = *
			n + hswork;
		maxwrk = max(i__1,i__2);
	    }
	    maxwrk = max(maxwrk,minwrk);
	}
	work[1] = (double) maxwrk;
	if (*lwork < minwrk && ! lquery) {
	    *info = -13;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEEV ", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	return 0;
    }
    //
    //    Get machine constants
    //
    eps = dlamch_("P");
    smlnum = dlamch_("S");
    bignum = 1. / smlnum;
    dlabad_(&smlnum, &bignum);
    smlnum = sqrt(smlnum) / eps;
    bignum = 1. / smlnum;
    //
    //    Scale A if max element outside range [SMLNUM,BIGNUM]
    //
    anrm = dlange_("M", n, n, &a[a_offset], lda, dum);
    scalea = FALSE_;
    if (anrm > 0. && anrm < smlnum) {
	scalea = TRUE_;
	cscale = smlnum;
    } else if (anrm > bignum) {
	scalea = TRUE_;
	cscale = bignum;
    }
    if (scalea) {
	dlascl_("G", &c__0, &c__0, &anrm, &cscale, n, n, &a[a_offset], lda, &
		ierr);
    }
    //
    //    Balance the matrix
    //    (Workspace: need N)
    //
    ibal = 1;
    dgebal_("B", n, &a[a_offset], lda, &ilo, &ihi, &work[ibal], &ierr);
    //
    //    Reduce to upper Hessenberg form
    //    (Workspace: need 3*N, prefer 2*N+N*NB)
    //
    itau = ibal + *n;
    iwrk = itau + *n;
    i__1 = *lwork - iwrk + 1;
    dgehrd_(n, &ilo, &ihi, &a[a_offset], lda, &work[itau], &work[iwrk], &i__1,
	     &ierr);
    if (wantvl) {
	//
	//       Want left eigenvectors
	//       Copy Householder vectors to VL
	//
	*(unsigned char *)side = 'L';
	dlacpy_("L", n, n, &a[a_offset], lda, &vl[vl_offset], ldvl);
	//
	//       Generate orthogonal matrix in VL
	//       (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
	//
	i__1 = *lwork - iwrk + 1;
	dorghr_(n, &ilo, &ihi, &vl[vl_offset], ldvl, &work[itau], &work[iwrk],
		 &i__1, &ierr);
	//
	//       Perform QR iteration, accumulating Schur vectors in VL
	//       (Workspace: need N+1, prefer N+HSWORK (see comments) )
	//
	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	dhseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &wr[1], &wi[1], &
		vl[vl_offset], ldvl, &work[iwrk], &i__1, info);
	if (wantvr) {
	    //
	    //          Want left and right eigenvectors
	    //          Copy Schur vectors to VR
	    //
	    *(unsigned char *)side = 'B';
	    dlacpy_("F", n, n, &vl[vl_offset], ldvl, &vr[vr_offset], ldvr);
	}
    } else if (wantvr) {
	//
	//       Want right eigenvectors
	//       Copy Householder vectors to VR
	//
	*(unsigned char *)side = 'R';
	dlacpy_("L", n, n, &a[a_offset], lda, &vr[vr_offset], ldvr);
	//
	//       Generate orthogonal matrix in VR
	//       (Workspace: need 3*N-1, prefer 2*N+(N-1)*NB)
	//
	i__1 = *lwork - iwrk + 1;
	dorghr_(n, &ilo, &ihi, &vr[vr_offset], ldvr, &work[itau], &work[iwrk],
		 &i__1, &ierr);
	//
	//       Perform QR iteration, accumulating Schur vectors in VR
	//       (Workspace: need N+1, prefer N+HSWORK (see comments) )
	//
	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	dhseqr_("S", "V", n, &ilo, &ihi, &a[a_offset], lda, &wr[1], &wi[1], &
		vr[vr_offset], ldvr, &work[iwrk], &i__1, info);
    } else {
	//
	//       Compute eigenvalues only
	//       (Workspace: need N+1, prefer N+HSWORK (see comments) )
	//
	iwrk = itau;
	i__1 = *lwork - iwrk + 1;
	dhseqr_("E", "N", n, &ilo, &ihi, &a[a_offset], lda, &wr[1], &wi[1], &
		vr[vr_offset], ldvr, &work[iwrk], &i__1, info);
    }
    //
    //    If INFO .NE. 0 from DHSEQR, then quit
    //
    if (*info != 0) {
	goto L50;
    }
    if (wantvl || wantvr) {
	//
	//       Compute left and/or right eigenvectors
	//       (Workspace: need 4*N, prefer N + N + 2*N*NB)
	//
	i__1 = *lwork - iwrk + 1;
	dtrevc3_(side, "B", select, n, &a[a_offset], lda, &vl[vl_offset],
		ldvl, &vr[vr_offset], ldvr, n, &nout, &work[iwrk], &i__1, &
		ierr);
    }
    if (wantvl) {
	//
	//       Undo balancing of left eigenvectors
	//       (Workspace: need N)
	//
	dgebak_("B", "L", n, &ilo, &ihi, &work[ibal], n, &vl[vl_offset], ldvl,
		 &ierr);
	//
	//       Normalize left eigenvectors and make largest component real
	//
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (wi[i__] == 0.) {
		scl = 1. / dnrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
		dscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
	    } else if (wi[i__] > 0.) {
		d__1 = dnrm2_(n, &vl[i__ * vl_dim1 + 1], &c__1);
		d__2 = dnrm2_(n, &vl[(i__ + 1) * vl_dim1 + 1], &c__1);
		scl = 1. / dlapy2_(&d__1, &d__2);
		dscal_(n, &scl, &vl[i__ * vl_dim1 + 1], &c__1);
		dscal_(n, &scl, &vl[(i__ + 1) * vl_dim1 + 1], &c__1);
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
		    // Computing 2nd power
		    d__1 = vl[k + i__ * vl_dim1];
		    // Computing 2nd power
		    d__2 = vl[k + (i__ + 1) * vl_dim1];
		    work[iwrk + k - 1] = d__1 * d__1 + d__2 * d__2;
// L10:
		}
		k = idamax_(n, &work[iwrk], &c__1);
		dlartg_(&vl[k + i__ * vl_dim1], &vl[k + (i__ + 1) * vl_dim1],
			&cs, &sn, &r__);
		drot_(n, &vl[i__ * vl_dim1 + 1], &c__1, &vl[(i__ + 1) *
			vl_dim1 + 1], &c__1, &cs, &sn);
		vl[k + (i__ + 1) * vl_dim1] = 0.;
	    }
// L20:
	}
    }
    if (wantvr) {
	//
	//       Undo balancing of right eigenvectors
	//       (Workspace: need N)
	//
	dgebak_("B", "R", n, &ilo, &ihi, &work[ibal], n, &vr[vr_offset], ldvr,
		 &ierr);
	//
	//       Normalize right eigenvectors and make largest component real
	//
	i__1 = *n;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    if (wi[i__] == 0.) {
		scl = 1. / dnrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
		dscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
	    } else if (wi[i__] > 0.) {
		d__1 = dnrm2_(n, &vr[i__ * vr_dim1 + 1], &c__1);
		d__2 = dnrm2_(n, &vr[(i__ + 1) * vr_dim1 + 1], &c__1);
		scl = 1. / dlapy2_(&d__1, &d__2);
		dscal_(n, &scl, &vr[i__ * vr_dim1 + 1], &c__1);
		dscal_(n, &scl, &vr[(i__ + 1) * vr_dim1 + 1], &c__1);
		i__2 = *n;
		for (k = 1; k <= i__2; ++k) {
		    // Computing 2nd power
		    d__1 = vr[k + i__ * vr_dim1];
		    // Computing 2nd power
		    d__2 = vr[k + (i__ + 1) * vr_dim1];
		    work[iwrk + k - 1] = d__1 * d__1 + d__2 * d__2;
// L30:
		}
		k = idamax_(n, &work[iwrk], &c__1);
		dlartg_(&vr[k + i__ * vr_dim1], &vr[k + (i__ + 1) * vr_dim1],
			&cs, &sn, &r__);
		drot_(n, &vr[i__ * vr_dim1 + 1], &c__1, &vr[(i__ + 1) *
			vr_dim1 + 1], &c__1, &cs, &sn);
		vr[k + (i__ + 1) * vr_dim1] = 0.;
	    }
// L40:
	}
    }
    //
    //    Undo scaling if necessary
    //
L50:
    if (scalea) {
	i__1 = *n - *info;
	// Computing MAX
	i__3 = *n - *info;
	i__2 = max(i__3,1);
	dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wr[*info +
		1], &i__2, &ierr);
	i__1 = *n - *info;
	// Computing MAX
	i__3 = *n - *info;
	i__2 = max(i__3,1);
	dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wi[*info +
		1], &i__2, &ierr);
	if (*info > 0) {
	    i__1 = ilo - 1;
	    dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wr[1],
		    n, &ierr);
	    i__1 = ilo - 1;
	    dlascl_("G", &c__0, &c__0, &cscale, &anrm, &i__1, &c__1, &wi[1],
		    n, &ierr);
	}
    }
    work[1] = (double) maxwrk;
    return 0;
    //
    //    End of DGEEV
    //
} // dgeev_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEHD2 reduces a general square matrix to upper Hessenberg form using an unblocked algorithm.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEHD2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgehd2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgehd2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgehd2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEHD2( N, ILO, IHI, A, LDA, TAU, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, ILO, INFO, LDA, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEHD2 reduces a real general matrix A to upper Hessenberg form H by
//> an orthogonal similarity transformation:  Q**T * A * Q = H .
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>
//>          It is assumed that A is already upper triangular in rows
//>          and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
//>          set by a previous call to DGEBAL; otherwise they should be
//>          set to 1 and N respectively. See Further Details.
//>          1 <= ILO <= IHI <= max(1,N).
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the n by n general matrix to be reduced.
//>          On exit, the upper triangle and the first subdiagonal of A
//>          are overwritten with the upper Hessenberg matrix H, and the
//>          elements below the first subdiagonal, with the array TAU,
//>          represent the orthogonal matrix Q as a product of elementary
//>          reflectors. See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (N-1)
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit.
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
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
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrix Q is represented as a product of (ihi-ilo) elementary
//>  reflectors
//>
//>     Q = H(ilo) H(ilo+1) . . . H(ihi-1).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
//>  exit in A(i+2:ihi,i), and tau in TAU(i).
//>
//>  The contents of A are illustrated by the following example, with
//>  n = 7, ilo = 2 and ihi = 6:
//>
//>  on entry,                        on exit,
//>
//>  ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
//>  (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
//>  (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
//>  (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
//>  (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
//>  (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
//>  (                         a )    (                          a )
//>
//>  where a denotes an element of the original matrix A, h denotes a
//>  modified element of the upper Hessenberg matrix H, and vi denotes an
//>  element of the vector defining H(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgehd2_(int *n, int *ilo, int *ihi, double *a, int *lda,
	double *tau, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__;
    double aii;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *), dlarfg_(int *, double *,
	    double *, int *, double *), xerbla_(char *, int *);

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
    --tau;
    --work;

    // Function Body
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -2;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEHD2", &i__1);
	return 0;
    }
    i__1 = *ihi - 1;
    for (i__ = *ilo; i__ <= i__1; ++i__) {
	//
	//       Compute elementary reflector H(i) to annihilate A(i+2:ihi,i)
	//
	i__2 = *ihi - i__;
	// Computing MIN
	i__3 = i__ + 2;
	dlarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*n) + i__ *
		a_dim1], &c__1, &tau[i__]);
	aii = a[i__ + 1 + i__ * a_dim1];
	a[i__ + 1 + i__ * a_dim1] = 1.;
	//
	//       Apply H(i) to A(1:ihi,i+1:ihi) from the right
	//
	i__2 = *ihi - i__;
	dlarf_("Right", ihi, &i__2, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
		i__], &a[(i__ + 1) * a_dim1 + 1], lda, &work[1]);
	//
	//       Apply H(i) to A(i+1:ihi,i+1:n) from the left
	//
	i__2 = *ihi - i__;
	i__3 = *n - i__;
	dlarf_("Left", &i__2, &i__3, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
		i__], &a[i__ + 1 + (i__ + 1) * a_dim1], lda, &work[1]);
	a[i__ + 1 + i__ * a_dim1] = aii;
// L10:
    }
    return 0;
    //
    //    End of DGEHD2
    //
} // dgehd2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DGEHRD
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DGEHRD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dgehrd.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dgehrd.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dgehrd.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DGEHRD( N, ILO, IHI, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, ILO, INFO, LDA, LWORK, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION  A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DGEHRD reduces a real general matrix A to upper Hessenberg form H by
//> an orthogonal similarity transformation:  Q**T * A * Q = H .
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.  N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>
//>          It is assumed that A is already upper triangular in rows
//>          and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
//>          set by a previous call to DGEBAL; otherwise they should be
//>          set to 1 and N respectively. See Further Details.
//>          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the N-by-N general matrix to be reduced.
//>          On exit, the upper triangle and the first subdiagonal of A
//>          are overwritten with the upper Hessenberg matrix H, and the
//>          elements below the first subdiagonal, with the array TAU,
//>          represent the orthogonal matrix Q as a product of elementary
//>          reflectors. See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (N-1)
//>          The scalar factors of the elementary reflectors (see Further
//>          Details). Elements 1:ILO-1 and IHI:N-1 of TAU are set to
//>          zero.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (LWORK)
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The length of the array WORK.  LWORK >= max(1,N).
//>          For good performance, LWORK should generally be larger.
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
//>          < 0:  if INFO = -i, the i-th argument had an illegal value.
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
//> \ingroup doubleGEcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrix Q is represented as a product of (ihi-ilo) elementary
//>  reflectors
//>
//>     Q = H(ilo) H(ilo+1) . . . H(ihi-1).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i) = 0, v(i+1) = 1 and v(ihi+1:n) = 0; v(i+2:ihi) is stored on
//>  exit in A(i+2:ihi,i), and tau in TAU(i).
//>
//>  The contents of A are illustrated by the following example, with
//>  n = 7, ilo = 2 and ihi = 6:
//>
//>  on entry,                        on exit,
//>
//>  ( a   a   a   a   a   a   a )    (  a   a   h   h   h   h   a )
//>  (     a   a   a   a   a   a )    (      a   h   h   h   h   a )
//>  (     a   a   a   a   a   a )    (      h   h   h   h   h   h )
//>  (     a   a   a   a   a   a )    (      v2  h   h   h   h   h )
//>  (     a   a   a   a   a   a )    (      v2  v3  h   h   h   h )
//>  (     a   a   a   a   a   a )    (      v2  v3  v4  h   h   h )
//>  (                         a )    (                          a )
//>
//>  where a denotes an element of the original matrix A, h denotes a
//>  modified element of the upper Hessenberg matrix H, and vi denotes an
//>  element of the vector defining H(i).
//>
//>  This file is a slight modification of LAPACK-3.0's DGEHRD
//>  subroutine incorporating improvements proposed by Quintana-Orti and
//>  Van de Geijn (2006). (See DLAHR2.)
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dgehrd_(int *n, int *ilo, int *ihi, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;
    int c__65 = 65;
    double c_b25 = -1.;
    double c_b26 = 1.;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3, i__4;

    // Local variables
    int i__, j, ib;
    double ei;
    int nb, nh, nx, iwt;
    extern /* Subroutine */ int dgemm_(char *, char *, int *, int *, int *,
	    double *, double *, int *, double *, int *, double *, double *,
	    int *);
    int nbmin, iinfo;
    extern /* Subroutine */ int dtrmm_(char *, char *, char *, char *, int *,
	    int *, double *, double *, int *, double *, int *), daxpy_(int *,
	    double *, double *, int *, double *, int *), dgehd2_(int *, int *,
	     int *, double *, int *, double *, double *, int *), dlahr2_(int *
	    , int *, int *, double *, int *, double *, double *, int *,
	    double *, int *), dlarfb_(char *, char *, char *, char *, int *,
	    int *, int *, double *, int *, double *, int *, double *, int *,
	    double *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int ldwork, lwkopt;
    int lquery;

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Test the input parameters
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -2;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -8;
    }
    if (*info == 0) {
	//
	//       Compute the workspace requirements
	//
	// Computing MIN
	i__1 = 64, i__2 = ilaenv_(&c__1, "DGEHRD", " ", n, ilo, ihi, &c_n1);
	nb = min(i__1,i__2);
	lwkopt = *n * nb + 4160;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGEHRD", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Set elements 1:ILO-1 and IHI:N-1 of TAU to zero
    //
    i__1 = *ilo - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	tau[i__] = 0.;
// L10:
    }
    i__1 = *n - 1;
    for (i__ = max(1,*ihi); i__ <= i__1; ++i__) {
	tau[i__] = 0.;
// L20:
    }
    //
    //    Quick return if possible
    //
    nh = *ihi - *ilo + 1;
    if (nh <= 1) {
	work[1] = 1.;
	return 0;
    }
    //
    //    Determine the block size
    //
    // Computing MIN
    i__1 = 64, i__2 = ilaenv_(&c__1, "DGEHRD", " ", n, ilo, ihi, &c_n1);
    nb = min(i__1,i__2);
    nbmin = 2;
    if (nb > 1 && nb < nh) {
	//
	//       Determine when to cross over from blocked to unblocked code
	//       (last block is always handled by unblocked code)
	//
	// Computing MAX
	i__1 = nb, i__2 = ilaenv_(&c__3, "DGEHRD", " ", n, ilo, ihi, &c_n1);
	nx = max(i__1,i__2);
	if (nx < nh) {
	    //
	    //          Determine if workspace is large enough for blocked code
	    //
	    if (*lwork < *n * nb + 4160) {
		//
		//             Not enough workspace to use optimal NB:  determine the
		//             minimum value of NB, and reduce NB or force use of
		//             unblocked code
		//
		// Computing MAX
		i__1 = 2, i__2 = ilaenv_(&c__2, "DGEHRD", " ", n, ilo, ihi, &
			c_n1);
		nbmin = max(i__1,i__2);
		if (*lwork >= *n * nbmin + 4160) {
		    nb = (*lwork - 4160) / *n;
		} else {
		    nb = 1;
		}
	    }
	}
    }
    ldwork = *n;
    if (nb < nbmin || nb >= nh) {
	//
	//       Use unblocked code below
	//
	i__ = *ilo;
    } else {
	//
	//       Use blocked code
	//
	iwt = *n * nb + 1;
	i__1 = *ihi - 1 - nx;
	i__2 = nb;
	for (i__ = *ilo; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	    // Computing MIN
	    i__3 = nb, i__4 = *ihi - i__;
	    ib = min(i__3,i__4);
	    //
	    //          Reduce columns i:i+ib-1 to Hessenberg form, returning the
	    //          matrices V and T of the block reflector H = I - V*T*V**T
	    //          which performs the reduction, and also the matrix Y = A*V*T
	    //
	    dlahr2_(ihi, &i__, &ib, &a[i__ * a_dim1 + 1], lda, &tau[i__], &
		    work[iwt], &c__65, &work[1], &ldwork);
	    //
	    //          Apply the block reflector H to A(1:ihi,i+ib:ihi) from the
	    //          right, computing  A := A - Y * V**T. V(i+ib,ib-1) must be set
	    //          to 1
	    //
	    ei = a[i__ + ib + (i__ + ib - 1) * a_dim1];
	    a[i__ + ib + (i__ + ib - 1) * a_dim1] = 1.;
	    i__3 = *ihi - i__ - ib + 1;
	    dgemm_("No transpose", "Transpose", ihi, &i__3, &ib, &c_b25, &
		    work[1], &ldwork, &a[i__ + ib + i__ * a_dim1], lda, &
		    c_b26, &a[(i__ + ib) * a_dim1 + 1], lda);
	    a[i__ + ib + (i__ + ib - 1) * a_dim1] = ei;
	    //
	    //          Apply the block reflector H to A(1:i,i+1:i+ib-1) from the
	    //          right
	    //
	    i__3 = ib - 1;
	    dtrmm_("Right", "Lower", "Transpose", "Unit", &i__, &i__3, &c_b26,
		     &a[i__ + 1 + i__ * a_dim1], lda, &work[1], &ldwork);
	    i__3 = ib - 2;
	    for (j = 0; j <= i__3; ++j) {
		daxpy_(&i__, &c_b25, &work[ldwork * j + 1], &c__1, &a[(i__ +
			j + 1) * a_dim1 + 1], &c__1);
// L30:
	    }
	    //
	    //          Apply the block reflector H to A(i+1:ihi,i+ib:n) from the
	    //          left
	    //
	    i__3 = *ihi - i__;
	    i__4 = *n - i__ - ib + 1;
	    dlarfb_("Left", "Transpose", "Forward", "Columnwise", &i__3, &
		    i__4, &ib, &a[i__ + 1 + i__ * a_dim1], lda, &work[iwt], &
		    c__65, &a[i__ + 1 + (i__ + ib) * a_dim1], lda, &work[1], &
		    ldwork);
// L40:
	}
    }
    //
    //    Use unblocked code to reduce the rest of the matrix
    //
    dgehd2_(n, &i__, ihi, &a[a_offset], lda, &tau[1], &work[1], &iinfo);
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DGEHRD
    //
} // dgehrd_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DHSEQR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DHSEQR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dhseqr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dhseqr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dhseqr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DHSEQR( JOB, COMPZ, N, ILO, IHI, H, LDH, WR, WI, Z,
//                         LDZ, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, ILO, INFO, LDH, LDZ, LWORK, N
//      CHARACTER          COMPZ, JOB
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), WI( * ), WORK( * ), WR( * ),
//     $                   Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DHSEQR computes the eigenvalues of a Hessenberg matrix H
//>    and, optionally, the matrices T and Z from the Schur decomposition
//>    H = Z T Z**T, where T is an upper quasi-triangular matrix (the
//>    Schur form), and Z is the orthogonal matrix of Schur vectors.
//>
//>    Optionally Z may be postmultiplied into an input orthogonal
//>    matrix Q so that this routine can give the Schur factorization
//>    of a matrix A which has been reduced to the Hessenberg form H
//>    by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOB
//> \verbatim
//>          JOB is CHARACTER*1
//>           = 'E':  compute eigenvalues only;
//>           = 'S':  compute eigenvalues and the Schur form T.
//> \endverbatim
//>
//> \param[in] COMPZ
//> \verbatim
//>          COMPZ is CHARACTER*1
//>           = 'N':  no Schur vectors are computed;
//>           = 'I':  Z is initialized to the unit matrix and the matrix Z
//>                   of Schur vectors of H is returned;
//>           = 'V':  Z must contain an orthogonal matrix Q on entry, and
//>                   the product Q*Z is returned.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           The order of the matrix H.  N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>
//>           It is assumed that H is already upper triangular in rows
//>           and columns 1:ILO-1 and IHI+1:N. ILO and IHI are normally
//>           set by a previous call to DGEBAL, and then passed to ZGEHRD
//>           when the matrix output by DGEBAL is reduced to Hessenberg
//>           form. Otherwise ILO and IHI should be set to 1 and N
//>           respectively.  If N > 0, then 1 <= ILO <= IHI <= N.
//>           If N = 0, then ILO = 1 and IHI = 0.
//> \endverbatim
//>
//> \param[in,out] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>           On entry, the upper Hessenberg matrix H.
//>           On exit, if INFO = 0 and JOB = 'S', then H contains the
//>           upper quasi-triangular matrix T from the Schur decomposition
//>           (the Schur form); 2-by-2 diagonal blocks (corresponding to
//>           complex conjugate pairs of eigenvalues) are returned in
//>           standard form, with H(i,i) = H(i+1,i+1) and
//>           H(i+1,i)*H(i,i+1) < 0. If INFO = 0 and JOB = 'E', the
//>           contents of H are unspecified on exit.  (The output value of
//>           H when INFO > 0 is given under the description of INFO
//>           below.)
//>
//>           Unlike earlier versions of DHSEQR, this subroutine may
//>           explicitly H(i,j) = 0 for i > j and j = 1, 2, ... ILO-1
//>           or j = IHI+1, IHI+2, ... N.
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>           The leading dimension of the array H. LDH >= max(1,N).
//> \endverbatim
//>
//> \param[out] WR
//> \verbatim
//>          WR is DOUBLE PRECISION array, dimension (N)
//> \endverbatim
//>
//> \param[out] WI
//> \verbatim
//>          WI is DOUBLE PRECISION array, dimension (N)
//>
//>           The real and imaginary parts, respectively, of the computed
//>           eigenvalues. If two eigenvalues are computed as a complex
//>           conjugate pair, they are stored in consecutive elements of
//>           WR and WI, say the i-th and (i+1)th, with WI(i) > 0 and
//>           WI(i+1) < 0. If JOB = 'S', the eigenvalues are stored in
//>           the same order as on the diagonal of the Schur form returned
//>           in H, with WR(i) = H(i,i) and, if H(i:i+1,i:i+1) is a 2-by-2
//>           diagonal block, WI(i) = sqrt(-H(i+1,i)*H(i,i+1)) and
//>           WI(i+1) = -WI(i).
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ,N)
//>           If COMPZ = 'N', Z is not referenced.
//>           If COMPZ = 'I', on entry Z need not be set and on exit,
//>           if INFO = 0, Z contains the orthogonal matrix Z of the Schur
//>           vectors of H.  If COMPZ = 'V', on entry Z must contain an
//>           N-by-N matrix Q, which is assumed to be equal to the unit
//>           matrix except for the submatrix Z(ILO:IHI,ILO:IHI). On exit,
//>           if INFO = 0, Z contains Q*Z.
//>           Normally Q is the orthogonal matrix generated by DORGHR
//>           after the call to DGEHRD which formed the Hessenberg matrix
//>           H. (The output value of Z when INFO > 0 is given under
//>           the description of INFO below.)
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>           The leading dimension of the array Z.  if COMPZ = 'I' or
//>           COMPZ = 'V', then LDZ >= MAX(1,N).  Otherwise, LDZ >= 1.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (LWORK)
//>           On exit, if INFO = 0, WORK(1) returns an estimate of
//>           the optimal value for LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>           The dimension of the array WORK.  LWORK >= max(1,N)
//>           is sufficient and delivers very good and sometimes
//>           optimal performance.  However, LWORK as large as 11*N
//>           may be required for optimal performance.  A workspace
//>           query is recommended to determine the optimal workspace
//>           size.
//>
//>           If LWORK = -1, then DHSEQR does a workspace query.
//>           In this case, DHSEQR checks the input parameters and
//>           estimates the optimal workspace size for the given
//>           values of N, ILO and IHI.  The estimate is returned
//>           in WORK(1).  No error message related to LWORK is
//>           issued by XERBLA.  Neither H nor Z are accessed.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>             = 0:  successful exit
//>             < 0:  if INFO = -i, the i-th argument had an illegal
//>                    value
//>             > 0:  if INFO = i, DHSEQR failed to compute all of
//>                the eigenvalues.  Elements 1:ilo-1 and i+1:n of WR
//>                and WI contain those eigenvalues which have been
//>                successfully computed.  (Failures are rare.)
//>
//>                If INFO > 0 and JOB = 'E', then on exit, the
//>                remaining unconverged eigenvalues are the eigen-
//>                values of the upper Hessenberg matrix rows and
//>                columns ILO through INFO of the final, output
//>                value of H.
//>
//>                If INFO > 0 and JOB   = 'S', then on exit
//>
//>           (*)  (initial value of H)*U  = U*(final value of H)
//>
//>                where U is an orthogonal matrix.  The final
//>                value of H is upper Hessenberg and quasi-triangular
//>                in rows and columns INFO+1 through IHI.
//>
//>                If INFO > 0 and COMPZ = 'V', then on exit
//>
//>                  (final value of Z)  =  (initial value of Z)*U
//>
//>                where U is the orthogonal matrix in (*) (regard-
//>                less of the value of JOB.)
//>
//>                If INFO > 0 and COMPZ = 'I', then on exit
//>                      (final value of Z)  = U
//>                where U is the orthogonal matrix in (*) (regard-
//>                less of the value of JOB.)
//>
//>                If INFO > 0 and COMPZ = 'N', then Z is not
//>                accessed.
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
//> \par Contributors:
// ==================
//>
//>       Karen Braman and Ralph Byers, Department of Mathematics,
//>       University of Kansas, USA
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>             Default values supplied by
//>             ILAENV(ISPEC,'DHSEQR',JOB(:1)//COMPZ(:1),N,ILO,IHI,LWORK).
//>             It is suggested that these defaults be adjusted in order
//>             to attain best performance in each particular
//>             computational environment.
//>
//>            ISPEC=12: The DLAHQR vs DLAQR0 crossover point.
//>                      Default: 75. (Must be at least 11.)
//>
//>            ISPEC=13: Recommended deflation window size.
//>                      This depends on ILO, IHI and NS.  NS is the
//>                      number of simultaneous shifts returned
//>                      by ILAENV(ISPEC=15).  (See ISPEC=15 below.)
//>                      The default for (IHI-ILO+1) <= 500 is NS.
//>                      The default for (IHI-ILO+1) >  500 is 3*NS/2.
//>
//>            ISPEC=14: Nibble crossover point. (See IPARMQ for
//>                      details.)  Default: 14% of deflation window
//>                      size.
//>
//>            ISPEC=15: Number of simultaneous shifts in a multishift
//>                      QR iteration.
//>
//>                      If IHI-ILO+1 is ...
//>
//>                      greater than      ...but less    ... the
//>                      or equal to ...      than        default is
//>
//>                           1               30          NS =   2(+)
//>                          30               60          NS =   4(+)
//>                          60              150          NS =  10(+)
//>                         150              590          NS =  **
//>                         590             3000          NS =  64
//>                        3000             6000          NS = 128
//>                        6000             infinity      NS = 256
//>
//>                  (+)  By default some or all matrices of this order
//>                       are passed to the implicit double shift routine
//>                       DLAHQR and this parameter is ignored.  See
//>                       ISPEC=12 above and comments in IPARMQ for
//>                       details.
//>
//>                 (**)  The asterisks (**) indicate an ad-hoc
//>                       function of N increasing from 10 to 64.
//>
//>            ISPEC=16: Select structured matrix multiply.
//>                      If the number of simultaneous shifts (specified
//>                      by ISPEC=15) is less than 14, then the default
//>                      for ISPEC=16 is 0.  Otherwise the default for
//>                      ISPEC=16 is 2.
//> \endverbatim
//
//> \par References:
// ================
//>
//>       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//>       Algorithm Part I: Maintaining Well Focused Shifts, and Level 3
//>       Performance, SIAM Journal of Matrix Analysis, volume 23, pages
//>       929--947, 2002.
//> \n
//>       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//>       Algorithm Part II: Aggressive Early Deflation, SIAM Journal
//>       of Matrix Analysis, volume 23, pages 948--973, 2002.
//
// =====================================================================
/* Subroutine */ int dhseqr_(char *job, char *compz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, double *z__, int *
	ldz, double *work, int *lwork, int *info)
{
    // Table of constant values
    double c_b11 = 0.;
    double c_b12 = 1.;
    int c__12 = 12;
    int c__2 = 2;
    int c__49 = 49;

    // System generated locals
    address a__1[2];
    int h_dim1, h_offset, z_dim1, z_offset, i__1, i__2[2], i__3;
    double d__1;
    char ch__1[2+1]={'\0'};

    // Local variables
    int i__;
    double hl[2401]	/* was [49][49] */;
    int kbot, nmin;
    extern int lsame_(char *, char *);
    int initz;
    double workl[49];
    int wantt, wantz;
    extern /* Subroutine */ int dlaqr0_(int *, int *, int *, int *, int *,
	    double *, int *, double *, double *, int *, int *, double *, int *
	    , double *, int *, int *), dlahqr_(int *, int *, int *, int *,
	    int *, double *, int *, double *, double *, int *, int *, double *
	    , int *, int *), dlacpy_(char *, int *, int *, double *, int *,
	    double *, int *), dlaset_(char *, int *, int *, double *, double *
	    , double *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    int lquery;

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
    //
    //    ==== Matrices of order NTINY or smaller must be processed by
    //    .    DLAHQR because of insufficient subdiagonal scratch space.
    //    .    (This is a hard limit.) ====
    //
    //    ==== NL allocates some local workspace to help small matrices
    //    .    through a rare DLAHQR failure.  NL > NTINY = 11 is
    //    .    required and NL <= NMIN = ILAENV(ISPEC=12,...) is recom-
    //    .    mended.  (The default value of NMIN is 75.)  Using NL = 49
    //    .    allows up to six simultaneous shifts and a 16-by-16
    //    .    deflation window.  ====
    //    ..
    //    .. Local Arrays ..
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
    //    ==== Decode and check the input parameters. ====
    //
    // Parameter adjustments
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    // Function Body
    wantt = lsame_(job, "S");
    initz = lsame_(compz, "I");
    wantz = initz || lsame_(compz, "V");
    work[1] = (double) max(1,*n);
    lquery = *lwork == -1;
    *info = 0;
    if (! lsame_(job, "E") && ! wantt) {
	*info = -1;
    } else if (! lsame_(compz, "N") && ! wantz) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -4;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -5;
    } else if (*ldh < max(1,*n)) {
	*info = -7;
    } else if (*ldz < 1 || wantz && *ldz < max(1,*n)) {
	*info = -11;
    } else if (*lwork < max(1,*n) && ! lquery) {
	*info = -13;
    }
    if (*info != 0) {
	//
	//       ==== Quick return in case of invalid argument. ====
	//
	i__1 = -(*info);
	xerbla_("DHSEQR", &i__1);
	return 0;
    } else if (*n == 0) {
	//
	//       ==== Quick return in case N = 0; nothing to do. ====
	//
	return 0;
    } else if (lquery) {
	//
	//       ==== Quick return in case of a workspace query ====
	//
	dlaqr0_(&wantt, &wantz, n, ilo, ihi, &h__[h_offset], ldh, &wr[1], &wi[
		1], ilo, ihi, &z__[z_offset], ldz, &work[1], lwork, info);
	//       ==== Ensure reported workspace size is backward-compatible with
	//       .    previous LAPACK versions. ====
	// Computing MAX
	d__1 = (double) max(1,*n);
	work[1] = max(d__1,work[1]);
	return 0;
    } else {
	//
	//       ==== copy eigenvalues isolated by DGEBAL ====
	//
	i__1 = *ilo - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    wr[i__] = h__[i__ + i__ * h_dim1];
	    wi[i__] = 0.;
// L10:
	}
	i__1 = *n;
	for (i__ = *ihi + 1; i__ <= i__1; ++i__) {
	    wr[i__] = h__[i__ + i__ * h_dim1];
	    wi[i__] = 0.;
// L20:
	}
	//
	//       ==== Initialize Z, if requested ====
	//
	if (initz) {
	    dlaset_("A", n, n, &c_b11, &c_b12, &z__[z_offset], ldz);
	}
	//
	//       ==== Quick return if possible ====
	//
	if (*ilo == *ihi) {
	    wr[*ilo] = h__[*ilo + *ilo * h_dim1];
	    wi[*ilo] = 0.;
	    return 0;
	}
	//
	//       ==== DLAHQR/DLAQR0 crossover point ====
	//
	// Writing concatenation
	i__2[0] = 1, a__1[0] = job;
	i__2[1] = 1, a__1[1] = compz;
	s_cat(ch__1, a__1, i__2, &c__2);
	nmin = ilaenv_(&c__12, "DHSEQR", ch__1, n, ilo, ihi, lwork);
	nmin = max(11,nmin);
	//
	//       ==== DLAQR0 for big matrices; DLAHQR for small ones ====
	//
	if (*n > nmin) {
	    dlaqr0_(&wantt, &wantz, n, ilo, ihi, &h__[h_offset], ldh, &wr[1],
		    &wi[1], ilo, ihi, &z__[z_offset], ldz, &work[1], lwork,
		    info);
	} else {
	    //
	    //          ==== Small matrix ====
	    //
	    dlahqr_(&wantt, &wantz, n, ilo, ihi, &h__[h_offset], ldh, &wr[1],
		    &wi[1], ilo, ihi, &z__[z_offset], ldz, info);
	    if (*info > 0) {
		//
		//             ==== A rare DLAHQR failure!  DLAQR0 sometimes succeeds
		//             .    when DLAHQR fails. ====
		//
		kbot = *info;
		if (*n >= 49) {
		    //
		    //                ==== Larger matrices have enough subdiagonal scratch
		    //                .    space to call DLAQR0 directly. ====
		    //
		    dlaqr0_(&wantt, &wantz, n, ilo, &kbot, &h__[h_offset],
			    ldh, &wr[1], &wi[1], ilo, ihi, &z__[z_offset],
			    ldz, &work[1], lwork, info);
		} else {
		    //
		    //                ==== Tiny matrices don't have enough subdiagonal
		    //                .    scratch space to benefit from DLAQR0.  Hence,
		    //                .    tiny matrices must be copied into a larger
		    //                .    array before calling DLAQR0. ====
		    //
		    dlacpy_("A", n, n, &h__[h_offset], ldh, hl, &c__49);
		    hl[*n + 1 + *n * 49 - 50] = 0.;
		    i__1 = 49 - *n;
		    dlaset_("A", &c__49, &i__1, &c_b11, &c_b11, &hl[(*n + 1) *
			     49 - 49], &c__49);
		    dlaqr0_(&wantt, &wantz, &c__49, ilo, &kbot, hl, &c__49, &
			    wr[1], &wi[1], ilo, ihi, &z__[z_offset], ldz,
			    workl, &c__49, info);
		    if (wantt || *info != 0) {
			dlacpy_("A", n, n, hl, &c__49, &h__[h_offset], ldh);
		    }
		}
	    }
	}
	//
	//       ==== Clear out the trash, if necessary. ====
	//
	if ((wantt || *info != 0) && *n > 2) {
	    i__1 = *n - 2;
	    i__3 = *n - 2;
	    dlaset_("L", &i__1, &i__3, &c_b11, &c_b11, &h__[h_dim1 + 3], ldh);
	}
	//
	//       ==== Ensure reported workspace size is backward-compatible with
	//       .    previous LAPACK versions. ====
	//
	// Computing MAX
	d__1 = (double) max(1,*n);
	work[1] = max(d__1,work[1]);
    }
    //
    //    ==== End of DHSEQR ====
    //
    return 0;
} // dhseqr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLABAD
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLABAD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlabad.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlabad.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlabad.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLABAD( SMALL, LARGE )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   LARGE, SMALL
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLABAD takes as input the values computed by DLAMCH for underflow and
//> overflow, and returns the square root of each of these values if the
//> log of LARGE is sufficiently large.  This subroutine is intended to
//> identify machines with a large exponent range, such as the Crays, and
//> redefine the underflow and overflow limits to be the square roots of
//> the values computed by DLAMCH.  This subroutine is needed because
//> DLAMCH does not compensate for poor arithmetic in the upper half of
//> the exponent range, as is found on a Cray.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in,out] SMALL
//> \verbatim
//>          SMALL is DOUBLE PRECISION
//>          On entry, the underflow threshold as computed by DLAMCH.
//>          On exit, if LOG10(LARGE) is sufficiently large, the square
//>          root of SMALL, otherwise unchanged.
//> \endverbatim
//>
//> \param[in,out] LARGE
//> \verbatim
//>          LARGE is DOUBLE PRECISION
//>          On entry, the overflow threshold as computed by DLAMCH.
//>          On exit, if LOG10(LARGE) is sufficiently large, the square
//>          root of LARGE, otherwise unchanged.
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
/* Subroutine */ int dlabad_(double *small, double *large)
{
    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    If it looks like we're on a Cray, take the square root of
    //    SMALL and LARGE to avoid overflow and underflow problems.
    //
    if (d_lg10(large) > 2e3) {
	*small = sqrt(*small);
	*large = sqrt(*large);
    }
    return 0;
    //
    //    End of DLABAD
    //
} // dlabad_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLADIV performs complex division in real arithmetic, avoiding unnecessary overflow.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLADIV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dladiv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dladiv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dladiv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLADIV( A, B, C, D, P, Q )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   A, B, C, D, P, Q
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLADIV performs complex division in  real arithmetic
//>
//>                       a + i*b
//>            p + i*q = ---------
//>                       c + i*d
//>
//> The algorithm is due to Michael Baudin and Robert L. Smith
//> and can be found in the paper
//> "A Robust Complex Division in Scilab"
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] B
//> \verbatim
//>          B is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] C
//> \verbatim
//>          C is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION
//>          The scalars a, b, c, and d in the above expression.
//> \endverbatim
//>
//> \param[out] P
//> \verbatim
//>          P is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] Q
//> \verbatim
//>          Q is DOUBLE PRECISION
//>          The scalars p and q in the above expression.
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
//> \date January 2013
//
//> \ingroup doubleOTHERauxiliary
//
// =====================================================================
/* Subroutine */ int dladiv_(double *a, double *b, double *c__, double *d__,
	double *p, double *q)
{
    // System generated locals
    double d__1, d__2;

    // Local variables
    double s, aa, ab, bb, cc, cd, dd, be, un, ov, eps;
    extern double dlamch_(char *);
    extern /* Subroutine */ int dladiv1_(double *, double *, double *, double
	    *, double *, double *);

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    January 2013
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //
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
    aa = *a;
    bb = *b;
    cc = *c__;
    dd = *d__;
    // Computing MAX
    d__1 = abs(*a), d__2 = abs(*b);
    ab = max(d__1,d__2);
    // Computing MAX
    d__1 = abs(*c__), d__2 = abs(*d__);
    cd = max(d__1,d__2);
    s = 1.;
    ov = dlamch_("Overflow threshold");
    un = dlamch_("Safe minimum");
    eps = dlamch_("Epsilon");
    be = 2. / (eps * eps);
    if (ab >= ov * .5) {
	aa *= .5;
	bb *= .5;
	s *= 2.;
    }
    if (cd >= ov * .5) {
	cc *= .5;
	dd *= .5;
	s *= .5;
    }
    if (ab <= un * 2. / eps) {
	aa *= be;
	bb *= be;
	s /= be;
    }
    if (cd <= un * 2. / eps) {
	cc *= be;
	dd *= be;
	s *= be;
    }
    if (abs(*d__) <= abs(*c__)) {
	dladiv1_(&aa, &bb, &cc, &dd, p, q);
    } else {
	dladiv1_(&bb, &aa, &dd, &cc, p, q);
	*q = -(*q);
    }
    *p *= s;
    *q *= s;
    return 0;
    //
    //    End of DLADIV
    //
} // dladiv_

//> \ingroup doubleOTHERauxiliary
/* Subroutine */ int dladiv1_(double *a, double *b, double *c__, double *d__,
	double *p, double *q)
{
    double r__, t;
    extern double dladiv2_(double *, double *, double *, double *, double *,
	    double *);

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    January 2013
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    r__ = *d__ / *c__;
    t = 1. / (*c__ + *d__ * r__);
    *p = dladiv2_(a, b, c__, d__, &r__, &t);
    *a = -(*a);
    *q = dladiv2_(b, a, c__, d__, &r__, &t);
    return 0;
    //
    //    End of DLADIV1
    //
} // dladiv1_

//> \ingroup doubleOTHERauxiliary
double dladiv2_(double *a, double *b, double *c__, double *d__, double *r__,
	double *t)
{
    // System generated locals
    double ret_val;

    // Local variables
    double br;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    January 2013
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    //    .. Parameters ..
    //
    //    .. Local Scalars ..
    //    ..
    //    .. Executable Statements ..
    //
    if (*r__ != 0.) {
	br = *b * *r__;
	if (br != 0.) {
	    ret_val = (*a + br) * *t;
	} else {
	    ret_val = *a * *t + *b * *t * *r__;
	}
    } else {
	ret_val = (*a + *d__ * (*b / *c__)) * *t;
    }
    return ret_val;
    //
    //    End of DLADIV12
    //
} // dladiv2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAEXC swaps adjacent diagonal blocks of a real upper quasi-triangular matrix in Schur canonical form, by an orthogonal similarity transformation.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAEXC + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaexc.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaexc.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaexc.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAEXC( WANTQ, N, T, LDT, Q, LDQ, J1, N1, N2, WORK,
//                         INFO )
//
//      .. Scalar Arguments ..
//      LOGICAL            WANTQ
//      INTEGER            INFO, J1, LDQ, LDT, N, N1, N2
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Q( LDQ, * ), T( LDT, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAEXC swaps adjacent diagonal blocks T11 and T22 of order 1 or 2 in
//> an upper quasi-triangular matrix T by an orthogonal similarity
//> transformation.
//>
//> T must be in Schur canonical form, that is, block upper triangular
//> with 1-by-1 and 2-by-2 diagonal blocks; each 2-by-2 diagonal block
//> has its diagonal elemnts equal and its off-diagonal elements of
//> opposite sign.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] WANTQ
//> \verbatim
//>          WANTQ is LOGICAL
//>          = .TRUE. : accumulate the transformation in the matrix Q;
//>          = .FALSE.: do not accumulate the transformation.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix T. N >= 0.
//> \endverbatim
//>
//> \param[in,out] T
//> \verbatim
//>          T is DOUBLE PRECISION array, dimension (LDT,N)
//>          On entry, the upper quasi-triangular matrix T, in Schur
//>          canonical form.
//>          On exit, the updated matrix T, again in Schur canonical form.
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of the array T. LDT >= max(1,N).
//> \endverbatim
//>
//> \param[in,out] Q
//> \verbatim
//>          Q is DOUBLE PRECISION array, dimension (LDQ,N)
//>          On entry, if WANTQ is .TRUE., the orthogonal matrix Q.
//>          On exit, if WANTQ is .TRUE., the updated matrix Q.
//>          If WANTQ is .FALSE., Q is not referenced.
//> \endverbatim
//>
//> \param[in] LDQ
//> \verbatim
//>          LDQ is INTEGER
//>          The leading dimension of the array Q.
//>          LDQ >= 1; and if WANTQ is .TRUE., LDQ >= N.
//> \endverbatim
//>
//> \param[in] J1
//> \verbatim
//>          J1 is INTEGER
//>          The index of the first row of the first block T11.
//> \endverbatim
//>
//> \param[in] N1
//> \verbatim
//>          N1 is INTEGER
//>          The order of the first block T11. N1 = 0, 1 or 2.
//> \endverbatim
//>
//> \param[in] N2
//> \verbatim
//>          N2 is INTEGER
//>          The order of the second block T22. N2 = 0, 1 or 2.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit
//>          = 1: the transformed matrix T would be too far from Schur
//>               form; the blocks are not swapped and T and Q are
//>               unchanged.
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
/* Subroutine */ int dlaexc_(int *wantq, int *n, double *t, int *ldt, double *
	q, int *ldq, int *j1, int *n1, int *n2, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__4 = 4;
    int c_false = FALSE_;
    int c_n1 = -1;
    int c__2 = 2;
    int c__3 = 3;

    // System generated locals
    int q_dim1, q_offset, t_dim1, t_offset, i__1;
    double d__1, d__2, d__3;

    // Local variables
    double d__[16]	/* was [4][4] */;
    int k;
    double u[3], x[4]	/* was [2][2] */;
    int j2, j3, j4;
    double u1[3], u2[3];
    int nd;
    double cs, t11, t22, t33, sn, wi1, wi2, wr1, wr2, eps, tau, tau1, tau2;
    int ierr;
    double temp;
    extern /* Subroutine */ int drot_(int *, double *, int *, double *, int *,
	     double *, double *);
    double scale, dnorm, xnorm;
    extern /* Subroutine */ int dlanv2_(double *, double *, double *, double *
	    , double *, double *, double *, double *, double *, double *),
	    dlasy2_(int *, int *, int *, int *, int *, double *, int *,
	    double *, int *, double *, int *, double *, double *, int *,
	    double *, int *);
    extern double dlamch_(char *), dlange_(char *, int *, int *, double *,
	    int *, double *);
    extern /* Subroutine */ int dlarfg_(int *, double *, double *, int *,
	    double *), dlacpy_(char *, int *, int *, double *, int *, double *
	    , int *), dlartg_(double *, double *, double *, double *, double *
	    ), dlarfx_(char *, int *, int *, double *, double *, double *,
	    int *, double *);
    double thresh, smlnum;

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
    // Parameter adjustments
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --work;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n == 0 || *n1 == 0 || *n2 == 0) {
	return 0;
    }
    if (*j1 + *n1 > *n) {
	return 0;
    }
    j2 = *j1 + 1;
    j3 = *j1 + 2;
    j4 = *j1 + 3;
    if (*n1 == 1 && *n2 == 1) {
	//
	//       Swap two 1-by-1 blocks.
	//
	t11 = t[*j1 + *j1 * t_dim1];
	t22 = t[j2 + j2 * t_dim1];
	//
	//       Determine the transformation to perform the interchange.
	//
	d__1 = t22 - t11;
	dlartg_(&t[*j1 + j2 * t_dim1], &d__1, &cs, &sn, &temp);
	//
	//       Apply transformation to the matrix T.
	//
	if (j3 <= *n) {
	    i__1 = *n - *j1 - 1;
	    drot_(&i__1, &t[*j1 + j3 * t_dim1], ldt, &t[j2 + j3 * t_dim1],
		    ldt, &cs, &sn);
	}
	i__1 = *j1 - 1;
	drot_(&i__1, &t[*j1 * t_dim1 + 1], &c__1, &t[j2 * t_dim1 + 1], &c__1,
		&cs, &sn);
	t[*j1 + *j1 * t_dim1] = t22;
	t[j2 + j2 * t_dim1] = t11;
	if (*wantq) {
	    //
	    //          Accumulate transformation in the matrix Q.
	    //
	    drot_(n, &q[*j1 * q_dim1 + 1], &c__1, &q[j2 * q_dim1 + 1], &c__1,
		    &cs, &sn);
	}
    } else {
	//
	//       Swapping involves at least one 2-by-2 block.
	//
	//       Copy the diagonal block of order N1+N2 to the local array D
	//       and compute its norm.
	//
	nd = *n1 + *n2;
	dlacpy_("Full", &nd, &nd, &t[*j1 + *j1 * t_dim1], ldt, d__, &c__4);
	dnorm = dlange_("Max", &nd, &nd, d__, &c__4, &work[1]);
	//
	//       Compute machine-dependent threshold for test for accepting
	//       swap.
	//
	eps = dlamch_("P");
	smlnum = dlamch_("S") / eps;
	// Computing MAX
	d__1 = eps * 10. * dnorm;
	thresh = max(d__1,smlnum);
	//
	//       Solve T11*X - X*T22 = scale*T12 for X.
	//
	dlasy2_(&c_false, &c_false, &c_n1, n1, n2, d__, &c__4, &d__[*n1 + 1 +
		(*n1 + 1 << 2) - 5], &c__4, &d__[(*n1 + 1 << 2) - 4], &c__4, &
		scale, x, &c__2, &xnorm, &ierr);
	//
	//       Swap the adjacent diagonal blocks.
	//
	k = *n1 + *n1 + *n2 - 3;
	switch (k) {
	    case 1:  goto L10;
	    case 2:  goto L20;
	    case 3:  goto L30;
	}
L10:
	//
	//       N1 = 1, N2 = 2: generate elementary reflector H so that:
	//
	//       ( scale, X11, X12 ) H = ( 0, 0, * )
	//
	u[0] = scale;
	u[1] = x[0];
	u[2] = x[2];
	dlarfg_(&c__3, &u[2], u, &c__1, &tau);
	u[2] = 1.;
	t11 = t[*j1 + *j1 * t_dim1];
	//
	//       Perform swap provisionally on diagonal block in D.
	//
	dlarfx_("L", &c__3, &c__3, u, &tau, d__, &c__4, &work[1]);
	dlarfx_("R", &c__3, &c__3, u, &tau, d__, &c__4, &work[1]);
	//
	//       Test whether to reject swap.
	//
	// Computing MAX
	d__2 = abs(d__[2]), d__3 = abs(d__[6]), d__2 = max(d__2,d__3), d__3 =
		(d__1 = d__[10] - t11, abs(d__1));
	if (max(d__2,d__3) > thresh) {
	    goto L50;
	}
	//
	//       Accept swap: apply transformation to the entire matrix T.
	//
	i__1 = *n - *j1 + 1;
	dlarfx_("L", &c__3, &i__1, u, &tau, &t[*j1 + *j1 * t_dim1], ldt, &
		work[1]);
	dlarfx_("R", &j2, &c__3, u, &tau, &t[*j1 * t_dim1 + 1], ldt, &work[1])
		;
	t[j3 + *j1 * t_dim1] = 0.;
	t[j3 + j2 * t_dim1] = 0.;
	t[j3 + j3 * t_dim1] = t11;
	if (*wantq) {
	    //
	    //          Accumulate transformation in the matrix Q.
	    //
	    dlarfx_("R", n, &c__3, u, &tau, &q[*j1 * q_dim1 + 1], ldq, &work[
		    1]);
	}
	goto L40;
L20:
	//
	//       N1 = 2, N2 = 1: generate elementary reflector H so that:
	//
	//       H (  -X11 ) = ( * )
	//         (  -X21 ) = ( 0 )
	//         ( scale ) = ( 0 )
	//
	u[0] = -x[0];
	u[1] = -x[1];
	u[2] = scale;
	dlarfg_(&c__3, u, &u[1], &c__1, &tau);
	u[0] = 1.;
	t33 = t[j3 + j3 * t_dim1];
	//
	//       Perform swap provisionally on diagonal block in D.
	//
	dlarfx_("L", &c__3, &c__3, u, &tau, d__, &c__4, &work[1]);
	dlarfx_("R", &c__3, &c__3, u, &tau, d__, &c__4, &work[1]);
	//
	//       Test whether to reject swap.
	//
	// Computing MAX
	d__2 = abs(d__[1]), d__3 = abs(d__[2]), d__2 = max(d__2,d__3), d__3 =
		(d__1 = d__[0] - t33, abs(d__1));
	if (max(d__2,d__3) > thresh) {
	    goto L50;
	}
	//
	//       Accept swap: apply transformation to the entire matrix T.
	//
	dlarfx_("R", &j3, &c__3, u, &tau, &t[*j1 * t_dim1 + 1], ldt, &work[1])
		;
	i__1 = *n - *j1;
	dlarfx_("L", &c__3, &i__1, u, &tau, &t[*j1 + j2 * t_dim1], ldt, &work[
		1]);
	t[*j1 + *j1 * t_dim1] = t33;
	t[j2 + *j1 * t_dim1] = 0.;
	t[j3 + *j1 * t_dim1] = 0.;
	if (*wantq) {
	    //
	    //          Accumulate transformation in the matrix Q.
	    //
	    dlarfx_("R", n, &c__3, u, &tau, &q[*j1 * q_dim1 + 1], ldq, &work[
		    1]);
	}
	goto L40;
L30:
	//
	//       N1 = 2, N2 = 2: generate elementary reflectors H(1) and H(2) so
	//       that:
	//
	//       H(2) H(1) (  -X11  -X12 ) = (  *  * )
	//                 (  -X21  -X22 )   (  0  * )
	//                 ( scale    0  )   (  0  0 )
	//                 (    0  scale )   (  0  0 )
	//
	u1[0] = -x[0];
	u1[1] = -x[1];
	u1[2] = scale;
	dlarfg_(&c__3, u1, &u1[1], &c__1, &tau1);
	u1[0] = 1.;
	temp = -tau1 * (x[2] + u1[1] * x[3]);
	u2[0] = -temp * u1[1] - x[3];
	u2[1] = -temp * u1[2];
	u2[2] = scale;
	dlarfg_(&c__3, u2, &u2[1], &c__1, &tau2);
	u2[0] = 1.;
	//
	//       Perform swap provisionally on diagonal block in D.
	//
	dlarfx_("L", &c__3, &c__4, u1, &tau1, d__, &c__4, &work[1]);
	dlarfx_("R", &c__4, &c__3, u1, &tau1, d__, &c__4, &work[1]);
	dlarfx_("L", &c__3, &c__4, u2, &tau2, &d__[1], &c__4, &work[1]);
	dlarfx_("R", &c__4, &c__3, u2, &tau2, &d__[4], &c__4, &work[1]);
	//
	//       Test whether to reject swap.
	//
	// Computing MAX
	d__1 = abs(d__[2]), d__2 = abs(d__[6]), d__1 = max(d__1,d__2), d__2 =
		abs(d__[3]), d__1 = max(d__1,d__2), d__2 = abs(d__[7]);
	if (max(d__1,d__2) > thresh) {
	    goto L50;
	}
	//
	//       Accept swap: apply transformation to the entire matrix T.
	//
	i__1 = *n - *j1 + 1;
	dlarfx_("L", &c__3, &i__1, u1, &tau1, &t[*j1 + *j1 * t_dim1], ldt, &
		work[1]);
	dlarfx_("R", &j4, &c__3, u1, &tau1, &t[*j1 * t_dim1 + 1], ldt, &work[
		1]);
	i__1 = *n - *j1 + 1;
	dlarfx_("L", &c__3, &i__1, u2, &tau2, &t[j2 + *j1 * t_dim1], ldt, &
		work[1]);
	dlarfx_("R", &j4, &c__3, u2, &tau2, &t[j2 * t_dim1 + 1], ldt, &work[1]
		);
	t[j3 + *j1 * t_dim1] = 0.;
	t[j3 + j2 * t_dim1] = 0.;
	t[j4 + *j1 * t_dim1] = 0.;
	t[j4 + j2 * t_dim1] = 0.;
	if (*wantq) {
	    //
	    //          Accumulate transformation in the matrix Q.
	    //
	    dlarfx_("R", n, &c__3, u1, &tau1, &q[*j1 * q_dim1 + 1], ldq, &
		    work[1]);
	    dlarfx_("R", n, &c__3, u2, &tau2, &q[j2 * q_dim1 + 1], ldq, &work[
		    1]);
	}
L40:
	if (*n2 == 2) {
	    //
	    //          Standardize new 2-by-2 block T11
	    //
	    dlanv2_(&t[*j1 + *j1 * t_dim1], &t[*j1 + j2 * t_dim1], &t[j2 + *
		    j1 * t_dim1], &t[j2 + j2 * t_dim1], &wr1, &wi1, &wr2, &
		    wi2, &cs, &sn);
	    i__1 = *n - *j1 - 1;
	    drot_(&i__1, &t[*j1 + (*j1 + 2) * t_dim1], ldt, &t[j2 + (*j1 + 2)
		    * t_dim1], ldt, &cs, &sn);
	    i__1 = *j1 - 1;
	    drot_(&i__1, &t[*j1 * t_dim1 + 1], &c__1, &t[j2 * t_dim1 + 1], &
		    c__1, &cs, &sn);
	    if (*wantq) {
		drot_(n, &q[*j1 * q_dim1 + 1], &c__1, &q[j2 * q_dim1 + 1], &
			c__1, &cs, &sn);
	    }
	}
	if (*n1 == 2) {
	    //
	    //          Standardize new 2-by-2 block T22
	    //
	    j3 = *j1 + *n2;
	    j4 = j3 + 1;
	    dlanv2_(&t[j3 + j3 * t_dim1], &t[j3 + j4 * t_dim1], &t[j4 + j3 *
		    t_dim1], &t[j4 + j4 * t_dim1], &wr1, &wi1, &wr2, &wi2, &
		    cs, &sn);
	    if (j3 + 2 <= *n) {
		i__1 = *n - j3 - 1;
		drot_(&i__1, &t[j3 + (j3 + 2) * t_dim1], ldt, &t[j4 + (j3 + 2)
			 * t_dim1], ldt, &cs, &sn);
	    }
	    i__1 = j3 - 1;
	    drot_(&i__1, &t[j3 * t_dim1 + 1], &c__1, &t[j4 * t_dim1 + 1], &
		    c__1, &cs, &sn);
	    if (*wantq) {
		drot_(n, &q[j3 * q_dim1 + 1], &c__1, &q[j4 * q_dim1 + 1], &
			c__1, &cs, &sn);
	    }
	}
    }
    return 0;
    //
    //    Exit with INFO = 1 if swap was rejected.
    //
L50:
    *info = 1;
    return 0;
    //
    //    End of DLAEXC
    //
} // dlaexc_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAHQR computes the eigenvalues and Schur factorization of an upper Hessenberg matrix, using the double-shift/single-shift QR algorithm.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAHQR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlahqr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlahqr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlahqr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAHQR( WANTT, WANTZ, N, ILO, IHI, H, LDH, WR, WI,
//                         ILOZ, IHIZ, Z, LDZ, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, IHIZ, ILO, ILOZ, INFO, LDH, LDZ, N
//      LOGICAL            WANTT, WANTZ
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), WI( * ), WR( * ), Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DLAHQR is an auxiliary routine called by DHSEQR to update the
//>    eigenvalues and Schur decomposition already computed by DHSEQR, by
//>    dealing with the Hessenberg submatrix in rows and columns ILO to
//>    IHI.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] WANTT
//> \verbatim
//>          WANTT is LOGICAL
//>          = .TRUE. : the full Schur form T is required;
//>          = .FALSE.: only eigenvalues are required.
//> \endverbatim
//>
//> \param[in] WANTZ
//> \verbatim
//>          WANTZ is LOGICAL
//>          = .TRUE. : the matrix of Schur vectors Z is required;
//>          = .FALSE.: Schur vectors are not required.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix H.  N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>          It is assumed that H is already upper quasi-triangular in
//>          rows and columns IHI+1:N, and that H(ILO,ILO-1) = 0 (unless
//>          ILO = 1). DLAHQR works primarily with the Hessenberg
//>          submatrix in rows and columns ILO to IHI, but applies
//>          transformations to all of H if WANTT is .TRUE..
//>          1 <= ILO <= max(1,IHI); IHI <= N.
//> \endverbatim
//>
//> \param[in,out] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>          On entry, the upper Hessenberg matrix H.
//>          On exit, if INFO is zero and if WANTT is .TRUE., H is upper
//>          quasi-triangular in rows and columns ILO:IHI, with any
//>          2-by-2 diagonal blocks in standard form. If INFO is zero
//>          and WANTT is .FALSE., the contents of H are unspecified on
//>          exit.  The output state of H if INFO is nonzero is given
//>          below under the description of INFO.
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>          The leading dimension of the array H. LDH >= max(1,N).
//> \endverbatim
//>
//> \param[out] WR
//> \verbatim
//>          WR is DOUBLE PRECISION array, dimension (N)
//> \endverbatim
//>
//> \param[out] WI
//> \verbatim
//>          WI is DOUBLE PRECISION array, dimension (N)
//>          The real and imaginary parts, respectively, of the computed
//>          eigenvalues ILO to IHI are stored in the corresponding
//>          elements of WR and WI. If two eigenvalues are computed as a
//>          complex conjugate pair, they are stored in consecutive
//>          elements of WR and WI, say the i-th and (i+1)th, with
//>          WI(i) > 0 and WI(i+1) < 0. If WANTT is .TRUE., the
//>          eigenvalues are stored in the same order as on the diagonal
//>          of the Schur form returned in H, with WR(i) = H(i,i), and, if
//>          H(i:i+1,i:i+1) is a 2-by-2 diagonal block,
//>          WI(i) = sqrt(H(i+1,i)*H(i,i+1)) and WI(i+1) = -WI(i).
//> \endverbatim
//>
//> \param[in] ILOZ
//> \verbatim
//>          ILOZ is INTEGER
//> \endverbatim
//>
//> \param[in] IHIZ
//> \verbatim
//>          IHIZ is INTEGER
//>          Specify the rows of Z to which transformations must be
//>          applied if WANTZ is .TRUE..
//>          1 <= ILOZ <= ILO; IHI <= IHIZ <= N.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ,N)
//>          If WANTZ is .TRUE., on entry Z must contain the current
//>          matrix Z of transformations accumulated by DHSEQR, and on
//>          exit Z has been updated; transformations are applied only to
//>          the submatrix Z(ILOZ:IHIZ,ILO:IHI).
//>          If WANTZ is .FALSE., Z is not referenced.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>          The leading dimension of the array Z. LDZ >= max(1,N).
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>           = 0:  successful exit
//>           > 0:  If INFO = i, DLAHQR failed to compute all the
//>                  eigenvalues ILO to IHI in a total of 30 iterations
//>                  per eigenvalue; elements i+1:ihi of WR and WI
//>                  contain those eigenvalues which have been
//>                  successfully computed.
//>
//>                  If INFO > 0 and WANTT is .FALSE., then on exit,
//>                  the remaining unconverged eigenvalues are the
//>                  eigenvalues of the upper Hessenberg matrix rows
//>                  and columns ILO through INFO of the final, output
//>                  value of H.
//>
//>                  If INFO > 0 and WANTT is .TRUE., then on exit
//>          (*)       (initial value of H)*U  = U*(final value of H)
//>                  where U is an orthogonal matrix.    The final
//>                  value of H is upper Hessenberg and triangular in
//>                  rows and columns INFO+1 through IHI.
//>
//>                  If INFO > 0 and WANTZ is .TRUE., then on exit
//>                      (final value of Z)  = (initial value of Z)*U
//>                  where U is the orthogonal matrix in (*)
//>                  (regardless of the value of WANTT.)
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>     02-96 Based on modifications by
//>     David Day, Sandia National Laboratory, USA
//>
//>     12-04 Further modifications by
//>     Ralph Byers, University of Kansas, USA
//>     This is a modified version of DLAHQR from LAPACK version 3.0.
//>     It is (1) more robust against overflow and underflow and
//>     (2) adopts the more conservative Ahues & Tisseur stopping
//>     criterion (LAWN 122, 1997).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlahqr_(int *wantt, int *wantz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, int *iloz, int *
	ihiz, double *z__, int *ldz, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4;
    double d__1, d__2, d__3, d__4;

    // Local variables
    int i__, j, k, l, m;
    double s, v[3];
    int i1, i2;
    double t1, t2, t3, v2, v3, aa, ab, ba, bb, h11, h12, h21, h22, cs;
    int nh;
    double sn;
    int nr;
    double tr;
    int nz;
    double det, h21s;
    int its;
    double ulp, sum, tst, rt1i, rt2i, rt1r, rt2r;
    extern /* Subroutine */ int drot_(int *, double *, int *, double *, int *,
	     double *, double *), dcopy_(int *, double *, int *, double *,
	    int *);
    int itmax;
    extern /* Subroutine */ int dlanv2_(double *, double *, double *, double *
	    , double *, double *, double *, double *, double *, double *),
	    dlabad_(double *, double *);
    extern double dlamch_(char *);
    extern /* Subroutine */ int dlarfg_(int *, double *, double *, int *,
	    double *);
    double safmin, safmax, rtdisc, smlnum;

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
    // =========================================================
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
    // Parameter adjustments
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	return 0;
    }
    if (*ilo == *ihi) {
	wr[*ilo] = h__[*ilo + *ilo * h_dim1];
	wi[*ilo] = 0.;
	return 0;
    }
    //
    //    ==== clear out the trash ====
    i__1 = *ihi - 3;
    for (j = *ilo; j <= i__1; ++j) {
	h__[j + 2 + j * h_dim1] = 0.;
	h__[j + 3 + j * h_dim1] = 0.;
// L10:
    }
    if (*ilo <= *ihi - 2) {
	h__[*ihi + (*ihi - 2) * h_dim1] = 0.;
    }
    nh = *ihi - *ilo + 1;
    nz = *ihiz - *iloz + 1;
    //
    //    Set machine-dependent constants for the stopping criterion.
    //
    safmin = dlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_("PRECISION");
    smlnum = safmin * ((double) nh / ulp);
    //
    //    I1 and I2 are the indices of the first row and last column of H
    //    to which transformations must be applied. If eigenvalues only are
    //    being computed, I1 and I2 are set inside the main loop.
    //
    if (*wantt) {
	i1 = 1;
	i2 = *n;
    }
    //
    //    ITMAX is the total number of QR iterations allowed.
    //
    itmax = max(10,nh) * 30;
    //
    //    The main loop begins here. I is the loop index and decreases from
    //    IHI to ILO in steps of 1 or 2. Each iteration of the loop works
    //    with the active submatrix in rows and columns L to I.
    //    Eigenvalues I+1 to IHI have already converged. Either L = ILO or
    //    H(L,L-1) is negligible so that the matrix splits.
    //
    i__ = *ihi;
L20:
    l = *ilo;
    if (i__ < *ilo) {
	goto L160;
    }
    //
    //    Perform QR iterations on rows and columns ILO to I until a
    //    submatrix of order 1 or 2 splits off at the bottom because a
    //    subdiagonal element has become negligible.
    //
    i__1 = itmax;
    for (its = 0; its <= i__1; ++its) {
	//
	//       Look for a single small subdiagonal element.
	//
	i__2 = l + 1;
	for (k = i__; k >= i__2; --k) {
	    if ((d__1 = h__[k + (k - 1) * h_dim1], abs(d__1)) <= smlnum) {
		goto L40;
	    }
	    tst = (d__1 = h__[k - 1 + (k - 1) * h_dim1], abs(d__1)) + (d__2 =
		    h__[k + k * h_dim1], abs(d__2));
	    if (tst == 0.) {
		if (k - 2 >= *ilo) {
		    tst += (d__1 = h__[k - 1 + (k - 2) * h_dim1], abs(d__1));
		}
		if (k + 1 <= *ihi) {
		    tst += (d__1 = h__[k + 1 + k * h_dim1], abs(d__1));
		}
	    }
	    //          ==== The following is a conservative small subdiagonal
	    //          .    deflation  criterion due to Ahues & Tisseur (LAWN 122,
	    //          .    1997). It has better mathematical foundation and
	    //          .    improves accuracy in some cases.  ====
	    if ((d__1 = h__[k + (k - 1) * h_dim1], abs(d__1)) <= ulp * tst) {
		// Computing MAX
		d__3 = (d__1 = h__[k + (k - 1) * h_dim1], abs(d__1)), d__4 = (
			d__2 = h__[k - 1 + k * h_dim1], abs(d__2));
		ab = max(d__3,d__4);
		// Computing MIN
		d__3 = (d__1 = h__[k + (k - 1) * h_dim1], abs(d__1)), d__4 = (
			d__2 = h__[k - 1 + k * h_dim1], abs(d__2));
		ba = min(d__3,d__4);
		// Computing MAX
		d__3 = (d__1 = h__[k + k * h_dim1], abs(d__1)), d__4 = (d__2 =
			 h__[k - 1 + (k - 1) * h_dim1] - h__[k + k * h_dim1],
			abs(d__2));
		aa = max(d__3,d__4);
		// Computing MIN
		d__3 = (d__1 = h__[k + k * h_dim1], abs(d__1)), d__4 = (d__2 =
			 h__[k - 1 + (k - 1) * h_dim1] - h__[k + k * h_dim1],
			abs(d__2));
		bb = min(d__3,d__4);
		s = aa + ab;
		// Computing MAX
		d__1 = smlnum, d__2 = ulp * (bb * (aa / s));
		if (ba * (ab / s) <= max(d__1,d__2)) {
		    goto L40;
		}
	    }
// L30:
	}
L40:
	l = k;
	if (l > *ilo) {
	    //
	    //          H(L,L-1) is negligible
	    //
	    h__[l + (l - 1) * h_dim1] = 0.;
	}
	//
	//       Exit from loop if a submatrix of order 1 or 2 has split off.
	//
	if (l >= i__ - 1) {
	    goto L150;
	}
	//
	//       Now the active submatrix is in rows and columns L to I. If
	//       eigenvalues only are being computed, only the active submatrix
	//       need be transformed.
	//
	if (! (*wantt)) {
	    i1 = l;
	    i2 = i__;
	}
	if (its == 10) {
	    //
	    //          Exceptional shift.
	    //
	    s = (d__1 = h__[l + 1 + l * h_dim1], abs(d__1)) + (d__2 = h__[l +
		    2 + (l + 1) * h_dim1], abs(d__2));
	    h11 = s * .75 + h__[l + l * h_dim1];
	    h12 = s * -.4375;
	    h21 = s;
	    h22 = h11;
	} else if (its == 20) {
	    //
	    //          Exceptional shift.
	    //
	    s = (d__1 = h__[i__ + (i__ - 1) * h_dim1], abs(d__1)) + (d__2 =
		    h__[i__ - 1 + (i__ - 2) * h_dim1], abs(d__2));
	    h11 = s * .75 + h__[i__ + i__ * h_dim1];
	    h12 = s * -.4375;
	    h21 = s;
	    h22 = h11;
	} else {
	    //
	    //          Prepare to use Francis' double shift
	    //          (i.e. 2nd degree generalized Rayleigh quotient)
	    //
	    h11 = h__[i__ - 1 + (i__ - 1) * h_dim1];
	    h21 = h__[i__ + (i__ - 1) * h_dim1];
	    h12 = h__[i__ - 1 + i__ * h_dim1];
	    h22 = h__[i__ + i__ * h_dim1];
	}
	s = abs(h11) + abs(h12) + abs(h21) + abs(h22);
	if (s == 0.) {
	    rt1r = 0.;
	    rt1i = 0.;
	    rt2r = 0.;
	    rt2i = 0.;
	} else {
	    h11 /= s;
	    h21 /= s;
	    h12 /= s;
	    h22 /= s;
	    tr = (h11 + h22) / 2.;
	    det = (h11 - tr) * (h22 - tr) - h12 * h21;
	    rtdisc = sqrt((abs(det)));
	    if (det >= 0.) {
		//
		//             ==== complex conjugate shifts ====
		//
		rt1r = tr * s;
		rt2r = rt1r;
		rt1i = rtdisc * s;
		rt2i = -rt1i;
	    } else {
		//
		//             ==== real shifts (use only one of them)  ====
		//
		rt1r = tr + rtdisc;
		rt2r = tr - rtdisc;
		if ((d__1 = rt1r - h22, abs(d__1)) <= (d__2 = rt2r - h22, abs(
			d__2))) {
		    rt1r *= s;
		    rt2r = rt1r;
		} else {
		    rt2r *= s;
		    rt1r = rt2r;
		}
		rt1i = 0.;
		rt2i = 0.;
	    }
	}
	//
	//       Look for two consecutive small subdiagonal elements.
	//
	i__2 = l;
	for (m = i__ - 2; m >= i__2; --m) {
	    //          Determine the effect of starting the double-shift QR
	    //          iteration at row M, and see if this would make H(M,M-1)
	    //          negligible.  (The following uses scaling to avoid
	    //          overflows and most underflows.)
	    //
	    h21s = h__[m + 1 + m * h_dim1];
	    s = (d__1 = h__[m + m * h_dim1] - rt2r, abs(d__1)) + abs(rt2i) +
		    abs(h21s);
	    h21s = h__[m + 1 + m * h_dim1] / s;
	    v[0] = h21s * h__[m + (m + 1) * h_dim1] + (h__[m + m * h_dim1] -
		    rt1r) * ((h__[m + m * h_dim1] - rt2r) / s) - rt1i * (rt2i
		    / s);
	    v[1] = h21s * (h__[m + m * h_dim1] + h__[m + 1 + (m + 1) * h_dim1]
		     - rt1r - rt2r);
	    v[2] = h21s * h__[m + 2 + (m + 1) * h_dim1];
	    s = abs(v[0]) + abs(v[1]) + abs(v[2]);
	    v[0] /= s;
	    v[1] /= s;
	    v[2] /= s;
	    if (m == l) {
		goto L60;
	    }
	    if ((d__1 = h__[m + (m - 1) * h_dim1], abs(d__1)) * (abs(v[1]) +
		    abs(v[2])) <= ulp * abs(v[0]) * ((d__2 = h__[m - 1 + (m -
		    1) * h_dim1], abs(d__2)) + (d__3 = h__[m + m * h_dim1],
		    abs(d__3)) + (d__4 = h__[m + 1 + (m + 1) * h_dim1], abs(
		    d__4)))) {
		goto L60;
	    }
// L50:
	}
L60:
	//
	//       Double-shift QR step
	//
	i__2 = i__ - 1;
	for (k = m; k <= i__2; ++k) {
	    //
	    //          The first iteration of this loop determines a reflection G
	    //          from the vector V and applies it from left and right to H,
	    //          thus creating a nonzero bulge below the subdiagonal.
	    //
	    //          Each subsequent iteration determines a reflection G to
	    //          restore the Hessenberg form in the (K-1)th column, and thus
	    //          chases the bulge one step toward the bottom of the active
	    //          submatrix. NR is the order of G.
	    //
	    // Computing MIN
	    i__3 = 3, i__4 = i__ - k + 1;
	    nr = min(i__3,i__4);
	    if (k > m) {
		dcopy_(&nr, &h__[k + (k - 1) * h_dim1], &c__1, v, &c__1);
	    }
	    dlarfg_(&nr, v, &v[1], &c__1, &t1);
	    if (k > m) {
		h__[k + (k - 1) * h_dim1] = v[0];
		h__[k + 1 + (k - 1) * h_dim1] = 0.;
		if (k < i__ - 1) {
		    h__[k + 2 + (k - 1) * h_dim1] = 0.;
		}
	    } else if (m > l) {
		//              ==== Use the following instead of
		//              .    H( K, K-1 ) = -H( K, K-1 ) to
		//              .    avoid a bug when v(2) and v(3)
		//              .    underflow. ====
		h__[k + (k - 1) * h_dim1] *= 1. - t1;
	    }
	    v2 = v[1];
	    t2 = t1 * v2;
	    if (nr == 3) {
		v3 = v[2];
		t3 = t1 * v3;
		//
		//             Apply G from the left to transform the rows of the matrix
		//             in columns K to I2.
		//
		i__3 = i2;
		for (j = k; j <= i__3; ++j) {
		    sum = h__[k + j * h_dim1] + v2 * h__[k + 1 + j * h_dim1]
			    + v3 * h__[k + 2 + j * h_dim1];
		    h__[k + j * h_dim1] -= sum * t1;
		    h__[k + 1 + j * h_dim1] -= sum * t2;
		    h__[k + 2 + j * h_dim1] -= sum * t3;
// L70:
		}
		//
		//             Apply G from the right to transform the columns of the
		//             matrix in rows I1 to min(K+3,I).
		//
		// Computing MIN
		i__4 = k + 3;
		i__3 = min(i__4,i__);
		for (j = i1; j <= i__3; ++j) {
		    sum = h__[j + k * h_dim1] + v2 * h__[j + (k + 1) * h_dim1]
			     + v3 * h__[j + (k + 2) * h_dim1];
		    h__[j + k * h_dim1] -= sum * t1;
		    h__[j + (k + 1) * h_dim1] -= sum * t2;
		    h__[j + (k + 2) * h_dim1] -= sum * t3;
// L80:
		}
		if (*wantz) {
		    //
		    //                Accumulate transformations in the matrix Z
		    //
		    i__3 = *ihiz;
		    for (j = *iloz; j <= i__3; ++j) {
			sum = z__[j + k * z_dim1] + v2 * z__[j + (k + 1) *
				z_dim1] + v3 * z__[j + (k + 2) * z_dim1];
			z__[j + k * z_dim1] -= sum * t1;
			z__[j + (k + 1) * z_dim1] -= sum * t2;
			z__[j + (k + 2) * z_dim1] -= sum * t3;
// L90:
		    }
		}
	    } else if (nr == 2) {
		//
		//             Apply G from the left to transform the rows of the matrix
		//             in columns K to I2.
		//
		i__3 = i2;
		for (j = k; j <= i__3; ++j) {
		    sum = h__[k + j * h_dim1] + v2 * h__[k + 1 + j * h_dim1];
		    h__[k + j * h_dim1] -= sum * t1;
		    h__[k + 1 + j * h_dim1] -= sum * t2;
// L100:
		}
		//
		//             Apply G from the right to transform the columns of the
		//             matrix in rows I1 to min(K+3,I).
		//
		i__3 = i__;
		for (j = i1; j <= i__3; ++j) {
		    sum = h__[j + k * h_dim1] + v2 * h__[j + (k + 1) * h_dim1]
			    ;
		    h__[j + k * h_dim1] -= sum * t1;
		    h__[j + (k + 1) * h_dim1] -= sum * t2;
// L110:
		}
		if (*wantz) {
		    //
		    //                Accumulate transformations in the matrix Z
		    //
		    i__3 = *ihiz;
		    for (j = *iloz; j <= i__3; ++j) {
			sum = z__[j + k * z_dim1] + v2 * z__[j + (k + 1) *
				z_dim1];
			z__[j + k * z_dim1] -= sum * t1;
			z__[j + (k + 1) * z_dim1] -= sum * t2;
// L120:
		    }
		}
	    }
// L130:
	}
// L140:
    }
    //
    //    Failure to converge in remaining number of iterations
    //
    *info = i__;
    return 0;
L150:
    if (l == i__) {
	//
	//       H(I,I-1) is negligible: one eigenvalue has converged.
	//
	wr[i__] = h__[i__ + i__ * h_dim1];
	wi[i__] = 0.;
    } else if (l == i__ - 1) {
	//
	//       H(I-1,I-2) is negligible: a pair of eigenvalues have converged.
	//
	//       Transform the 2-by-2 submatrix to standard Schur form,
	//       and compute and store the eigenvalues.
	//
	dlanv2_(&h__[i__ - 1 + (i__ - 1) * h_dim1], &h__[i__ - 1 + i__ *
		h_dim1], &h__[i__ + (i__ - 1) * h_dim1], &h__[i__ + i__ *
		h_dim1], &wr[i__ - 1], &wi[i__ - 1], &wr[i__], &wi[i__], &cs,
		&sn);
	if (*wantt) {
	    //
	    //          Apply the transformation to the rest of H.
	    //
	    if (i2 > i__) {
		i__1 = i2 - i__;
		drot_(&i__1, &h__[i__ - 1 + (i__ + 1) * h_dim1], ldh, &h__[
			i__ + (i__ + 1) * h_dim1], ldh, &cs, &sn);
	    }
	    i__1 = i__ - i1 - 1;
	    drot_(&i__1, &h__[i1 + (i__ - 1) * h_dim1], &c__1, &h__[i1 + i__ *
		     h_dim1], &c__1, &cs, &sn);
	}
	if (*wantz) {
	    //
	    //          Apply the transformation to Z.
	    //
	    drot_(&nz, &z__[*iloz + (i__ - 1) * z_dim1], &c__1, &z__[*iloz +
		    i__ * z_dim1], &c__1, &cs, &sn);
	}
    }
    //
    //    return to start of the main loop with new value of I.
    //
    i__ = l - 1;
    goto L20;
L160:
    return 0;
    //
    //    End of DLAHQR
    //
} // dlahqr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAHR2 reduces the specified number of first columns of a general rectangular matrix A so that elements below the specified subdiagonal are zero, and returns auxiliary matrices which are needed to apply the transformation to the unreduced part of A.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAHR2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlahr2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlahr2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlahr2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAHR2( N, K, NB, A, LDA, TAU, T, LDT, Y, LDY )
//
//      .. Scalar Arguments ..
//      INTEGER            K, LDA, LDT, LDY, N, NB
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION  A( LDA, * ), T( LDT, NB ), TAU( NB ),
//     $                   Y( LDY, NB )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAHR2 reduces the first NB columns of A real general n-BY-(n-k+1)
//> matrix A so that elements below the k-th subdiagonal are zero. The
//> reduction is performed by an orthogonal similarity transformation
//> Q**T * A * Q. The routine returns the matrices V and T which determine
//> Q as a block reflector I - V*T*V**T, and also the matrix Y = A * V * T.
//>
//> This is an auxiliary routine called by DGEHRD.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The offset for the reduction. Elements below the k-th
//>          subdiagonal in the first NB columns are reduced to zero.
//>          K < N.
//> \endverbatim
//>
//> \param[in] NB
//> \verbatim
//>          NB is INTEGER
//>          The number of columns to be reduced.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N-K+1)
//>          On entry, the n-by-(n-k+1) general matrix A.
//>          On exit, the elements on and above the k-th subdiagonal in
//>          the first NB columns are overwritten with the corresponding
//>          elements of the reduced matrix; the elements below the k-th
//>          subdiagonal, with the array TAU, represent the matrix Q as a
//>          product of elementary reflectors. The other columns of A are
//>          unchanged. See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (NB)
//>          The scalar factors of the elementary reflectors. See Further
//>          Details.
//> \endverbatim
//>
//> \param[out] T
//> \verbatim
//>          T is DOUBLE PRECISION array, dimension (LDT,NB)
//>          The upper triangular matrix T.
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of the array T.  LDT >= NB.
//> \endverbatim
//>
//> \param[out] Y
//> \verbatim
//>          Y is DOUBLE PRECISION array, dimension (LDY,NB)
//>          The n-by-nb matrix Y.
//> \endverbatim
//>
//> \param[in] LDY
//> \verbatim
//>          LDY is INTEGER
//>          The leading dimension of the array Y. LDY >= N.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The matrix Q is represented as a product of nb elementary reflectors
//>
//>     Q = H(1) H(2) . . . H(nb).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i+k-1) = 0, v(i+k) = 1; v(i+k+1:n) is stored on exit in
//>  A(i+k+1:n,i), and tau in TAU(i).
//>
//>  The elements of the vectors v together form the (n-k+1)-by-nb matrix
//>  V which is needed, with T and Y, to apply the transformation to the
//>  unreduced part of the matrix, using an update of the form:
//>  A := (I - V*T*V**T) * (A - Y*V**T).
//>
//>  The contents of A on exit are illustrated by the following example
//>  with n = 7, k = 3 and nb = 2:
//>
//>     ( a   a   a   a   a )
//>     ( a   a   a   a   a )
//>     ( a   a   a   a   a )
//>     ( h   h   a   a   a )
//>     ( v1  h   a   a   a )
//>     ( v1  v2  a   a   a )
//>     ( v1  v2  a   a   a )
//>
//>  where a denotes an element of the original matrix A, h denotes a
//>  modified element of the upper Hessenberg matrix H, and vi denotes an
//>  element of the vector defining H(i).
//>
//>  This subroutine is a slight modification of LAPACK-3.0's DLAHRD
//>  incorporating improvements proposed by Quintana-Orti and Van de
//>  Gejin. Note that the entries of A(1:K,2:NB) differ from those
//>  returned by the original LAPACK-3.0's DLAHRD routine. (This
//>  subroutine is not backward compatible with LAPACK-3.0's DLAHRD.)
//> \endverbatim
//
//> \par References:
// ================
//>
//>  Gregorio Quintana-Orti and Robert van de Geijn, "Improving the
//>  performance of reduction to Hessenberg form," ACM Transactions on
//>  Mathematical Software, 32(2):180-194, June 2006.
//>
// =====================================================================
/* Subroutine */ int dlahr2_(int *n, int *k, int *nb, double *a, int *lda,
	double *tau, double *t, int *ldt, double *y, int *ldy)
{
    // Table of constant values
    double c_b4 = -1.;
    double c_b5 = 1.;
    int c__1 = 1;
    double c_b38 = 0.;

    // System generated locals
    int a_dim1, a_offset, t_dim1, t_offset, y_dim1, y_offset, i__1, i__2,
	    i__3;
    double d__1;

    // Local variables
    int i__;
    double ei;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *),
	    dgemm_(char *, char *, int *, int *, int *, double *, double *,
	    int *, double *, int *, double *, double *, int *), dgemv_(char *,
	     int *, int *, double *, double *, int *, double *, int *, double
	    *, double *, int *), dcopy_(int *, double *, int *, double *, int
	    *), dtrmm_(char *, char *, char *, char *, int *, int *, double *,
	     double *, int *, double *, int *), daxpy_(int *, double *,
	    double *, int *, double *, int *), dtrmv_(char *, char *, char *,
	    int *, double *, int *, double *, int *), dlarfg_(int *, double *,
	     double *, int *, double *), dlacpy_(char *, int *, int *, double
	    *, int *, double *, int *);

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Quick return if possible
    //
    // Parameter adjustments
    --tau;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    // Function Body
    if (*n <= 1) {
	return 0;
    }
    i__1 = *nb;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (i__ > 1) {
	    //
	    //          Update A(K+1:N,I)
	    //
	    //          Update I-th column of A - Y * V**T
	    //
	    i__2 = *n - *k;
	    i__3 = i__ - 1;
	    dgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b4, &y[*k + 1 + y_dim1],
		    ldy, &a[*k + i__ - 1 + a_dim1], lda, &c_b5, &a[*k + 1 +
		    i__ * a_dim1], &c__1);
	    //
	    //          Apply I - V * T**T * V**T to this column (call it b) from the
	    //          left, using the last column of T as workspace
	    //
	    //          Let  V = ( V1 )   and   b = ( b1 )   (first I-1 rows)
	    //                   ( V2 )             ( b2 )
	    //
	    //          where V1 is unit lower triangular
	    //
	    //          w := V1**T * b1
	    //
	    i__2 = i__ - 1;
	    dcopy_(&i__2, &a[*k + 1 + i__ * a_dim1], &c__1, &t[*nb * t_dim1 +
		    1], &c__1);
	    i__2 = i__ - 1;
	    dtrmv_("Lower", "Transpose", "UNIT", &i__2, &a[*k + 1 + a_dim1],
		    lda, &t[*nb * t_dim1 + 1], &c__1);
	    //
	    //          w := w + V2**T * b2
	    //
	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[*k + i__ + a_dim1],
		    lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b5, &t[*nb *
		    t_dim1 + 1], &c__1);
	    //
	    //          w := T**T * w
	    //
	    i__2 = i__ - 1;
	    dtrmv_("Upper", "Transpose", "NON-UNIT", &i__2, &t[t_offset], ldt,
		     &t[*nb * t_dim1 + 1], &c__1);
	    //
	    //          b2 := b2 - V2*w
	    //
	    i__2 = *n - *k - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b4, &a[*k + i__ + a_dim1],
		     lda, &t[*nb * t_dim1 + 1], &c__1, &c_b5, &a[*k + i__ +
		    i__ * a_dim1], &c__1);
	    //
	    //          b1 := b1 - V1*w
	    //
	    i__2 = i__ - 1;
	    dtrmv_("Lower", "NO TRANSPOSE", "UNIT", &i__2, &a[*k + 1 + a_dim1]
		    , lda, &t[*nb * t_dim1 + 1], &c__1);
	    i__2 = i__ - 1;
	    daxpy_(&i__2, &c_b4, &t[*nb * t_dim1 + 1], &c__1, &a[*k + 1 + i__
		    * a_dim1], &c__1);
	    a[*k + i__ - 1 + (i__ - 1) * a_dim1] = ei;
	}
	//
	//       Generate the elementary reflector H(I) to annihilate
	//       A(K+I+1:N,I)
	//
	i__2 = *n - *k - i__ + 1;
	// Computing MIN
	i__3 = *k + i__ + 1;
	dlarfg_(&i__2, &a[*k + i__ + i__ * a_dim1], &a[min(i__3,*n) + i__ *
		a_dim1], &c__1, &tau[i__]);
	ei = a[*k + i__ + i__ * a_dim1];
	a[*k + i__ + i__ * a_dim1] = 1.;
	//
	//       Compute  Y(K+1:N,I)
	//
	i__2 = *n - *k;
	i__3 = *n - *k - i__ + 1;
	dgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b5, &a[*k + 1 + (i__ + 1) *
		a_dim1], lda, &a[*k + i__ + i__ * a_dim1], &c__1, &c_b38, &y[*
		k + 1 + i__ * y_dim1], &c__1);
	i__2 = *n - *k - i__ + 1;
	i__3 = i__ - 1;
	dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[*k + i__ + a_dim1], lda, &
		a[*k + i__ + i__ * a_dim1], &c__1, &c_b38, &t[i__ * t_dim1 +
		1], &c__1);
	i__2 = *n - *k;
	i__3 = i__ - 1;
	dgemv_("NO TRANSPOSE", &i__2, &i__3, &c_b4, &y[*k + 1 + y_dim1], ldy,
		&t[i__ * t_dim1 + 1], &c__1, &c_b5, &y[*k + 1 + i__ * y_dim1],
		 &c__1);
	i__2 = *n - *k;
	dscal_(&i__2, &tau[i__], &y[*k + 1 + i__ * y_dim1], &c__1);
	//
	//       Compute T(1:I,I)
	//
	i__2 = i__ - 1;
	d__1 = -tau[i__];
	dscal_(&i__2, &d__1, &t[i__ * t_dim1 + 1], &c__1);
	i__2 = i__ - 1;
	dtrmv_("Upper", "No Transpose", "NON-UNIT", &i__2, &t[t_offset], ldt,
		&t[i__ * t_dim1 + 1], &c__1);
	t[i__ + i__ * t_dim1] = tau[i__];
// L10:
    }
    a[*k + *nb + *nb * a_dim1] = ei;
    //
    //    Compute Y(1:K,1:NB)
    //
    dlacpy_("ALL", k, nb, &a[(a_dim1 << 1) + 1], lda, &y[y_offset], ldy);
    dtrmm_("RIGHT", "Lower", "NO TRANSPOSE", "UNIT", k, nb, &c_b5, &a[*k + 1
	    + a_dim1], lda, &y[y_offset], ldy);
    if (*n > *k + *nb) {
	i__1 = *n - *k - *nb;
	dgemm_("NO TRANSPOSE", "NO TRANSPOSE", k, nb, &i__1, &c_b5, &a[(*nb +
		2) * a_dim1 + 1], lda, &a[*k + 1 + *nb + a_dim1], lda, &c_b5,
		&y[y_offset], ldy);
    }
    dtrmm_("RIGHT", "Upper", "NO TRANSPOSE", "NON-UNIT", k, nb, &c_b5, &t[
	    t_offset], ldt, &y[y_offset], ldy);
    return 0;
    //
    //    End of DLAHR2
    //
} // dlahr2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLALN2 solves a 1-by-1 or 2-by-2 linear system of equations of the specified form.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLALN2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaln2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaln2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaln2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLALN2( LTRANS, NA, NW, SMIN, CA, A, LDA, D1, D2, B,
//                         LDB, WR, WI, X, LDX, SCALE, XNORM, INFO )
//
//      .. Scalar Arguments ..
//      LOGICAL            LTRANS
//      INTEGER            INFO, LDA, LDB, LDX, NA, NW
//      DOUBLE PRECISION   CA, D1, D2, SCALE, SMIN, WI, WR, XNORM
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), X( LDX, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLALN2 solves a system of the form  (ca A - w D ) X = s B
//> or (ca A**T - w D) X = s B   with possible scaling ("s") and
//> perturbation of A.  (A**T means A-transpose.)
//>
//> A is an NA x NA real matrix, ca is a real scalar, D is an NA x NA
//> real diagonal matrix, w is a real or complex value, and X and B are
//> NA x 1 matrices -- real if w is real, complex if w is complex.  NA
//> may be 1 or 2.
//>
//> If w is complex, X and B are represented as NA x 2 matrices,
//> the first column of each being the real part and the second
//> being the imaginary part.
//>
//> "s" is a scaling factor (<= 1), computed by DLALN2, which is
//> so chosen that X can be computed without overflow.  X is further
//> scaled if necessary to assure that norm(ca A - w D)*norm(X) is less
//> than overflow.
//>
//> If both singular values of (ca A - w D) are less than SMIN,
//> SMIN*identity will be used instead of (ca A - w D).  If only one
//> singular value is less than SMIN, one element of (ca A - w D) will be
//> perturbed enough to make the smallest singular value roughly SMIN.
//> If both singular values are at least SMIN, (ca A - w D) will not be
//> perturbed.  In any case, the perturbation will be at most some small
//> multiple of max( SMIN, ulp*norm(ca A - w D) ).  The singular values
//> are computed by infinity-norm approximations, and thus will only be
//> correct to a factor of 2 or so.
//>
//> Note: all input quantities are assumed to be smaller than overflow
//> by a reasonable factor.  (See BIGNUM.)
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] LTRANS
//> \verbatim
//>          LTRANS is LOGICAL
//>          =.TRUE.:  A-transpose will be used.
//>          =.FALSE.: A will be used (not transposed.)
//> \endverbatim
//>
//> \param[in] NA
//> \verbatim
//>          NA is INTEGER
//>          The size of the matrix A.  It may (only) be 1 or 2.
//> \endverbatim
//>
//> \param[in] NW
//> \verbatim
//>          NW is INTEGER
//>          1 if "w" is real, 2 if "w" is complex.  It may only be 1
//>          or 2.
//> \endverbatim
//>
//> \param[in] SMIN
//> \verbatim
//>          SMIN is DOUBLE PRECISION
//>          The desired lower bound on the singular values of A.  This
//>          should be a safe distance away from underflow or overflow,
//>          say, between (underflow/machine precision) and  (machine
//>          precision * overflow ).  (See BIGNUM and ULP.)
//> \endverbatim
//>
//> \param[in] CA
//> \verbatim
//>          CA is DOUBLE PRECISION
//>          The coefficient c, which A is multiplied by.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,NA)
//>          The NA x NA matrix A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of A.  It must be at least NA.
//> \endverbatim
//>
//> \param[in] D1
//> \verbatim
//>          D1 is DOUBLE PRECISION
//>          The 1,1 element in the diagonal matrix D.
//> \endverbatim
//>
//> \param[in] D2
//> \verbatim
//>          D2 is DOUBLE PRECISION
//>          The 2,2 element in the diagonal matrix D.  Not used if NA=1.
//> \endverbatim
//>
//> \param[in] B
//> \verbatim
//>          B is DOUBLE PRECISION array, dimension (LDB,NW)
//>          The NA x NW matrix B (right-hand side).  If NW=2 ("w" is
//>          complex), column 1 contains the real part of B and column 2
//>          contains the imaginary part.
//> \endverbatim
//>
//> \param[in] LDB
//> \verbatim
//>          LDB is INTEGER
//>          The leading dimension of B.  It must be at least NA.
//> \endverbatim
//>
//> \param[in] WR
//> \verbatim
//>          WR is DOUBLE PRECISION
//>          The real part of the scalar "w".
//> \endverbatim
//>
//> \param[in] WI
//> \verbatim
//>          WI is DOUBLE PRECISION
//>          The imaginary part of the scalar "w".  Not used if NW=1.
//> \endverbatim
//>
//> \param[out] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension (LDX,NW)
//>          The NA x NW matrix X (unknowns), as computed by DLALN2.
//>          If NW=2 ("w" is complex), on exit, column 1 will contain
//>          the real part of X and column 2 will contain the imaginary
//>          part.
//> \endverbatim
//>
//> \param[in] LDX
//> \verbatim
//>          LDX is INTEGER
//>          The leading dimension of X.  It must be at least NA.
//> \endverbatim
//>
//> \param[out] SCALE
//> \verbatim
//>          SCALE is DOUBLE PRECISION
//>          The scale factor that B must be multiplied by to insure
//>          that overflow does not occur when computing X.  Thus,
//>          (ca A - w D) X  will be SCALE*B, not B (ignoring
//>          perturbations of A.)  It will be at most 1.
//> \endverbatim
//>
//> \param[out] XNORM
//> \verbatim
//>          XNORM is DOUBLE PRECISION
//>          The infinity-norm of X, when X is regarded as an NA x NW
//>          real matrix.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          An error flag.  It will be set to zero if no error occurs,
//>          a negative number if an argument is in error, or a positive
//>          number if  ca A - w D  had to be perturbed.
//>          The possible values are:
//>          = 0: No error occurred, and (ca A - w D) did not have to be
//>                 perturbed.
//>          = 1: (ca A - w D) had to be perturbed to make its smallest
//>               (or only) singular value greater than SMIN.
//>          NOTE: In the interests of speed, this routine does not
//>                check the inputs for errors.
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
/* Subroutine */ int dlaln2_(int *ltrans, int *na, int *nw, double *smin,
	double *ca, double *a, int *lda, double *d1, double *d2, double *b,
	int *ldb, double *wr, double *wi, double *x, int *ldx, double *scale,
	double *xnorm, int *info)
{
    /* Initialized data */

    static int zswap[4] = { FALSE_,FALSE_,TRUE_,TRUE_ };
    static int rswap[4] = { FALSE_,TRUE_,FALSE_,TRUE_ };
    static int ipivot[16]	/* was [4][4] */ = { 1,2,3,4,2,1,4,3,3,4,1,2,
	    4,3,2,1 };

    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, x_dim1, x_offset;
    double d__1, d__2, d__3, d__4, d__5, d__6;
    static double equiv_0[4], equiv_1[4];

    // Local variables
    int j;
#define ci (equiv_0)
#define cr (equiv_1)
    double bi1, bi2, br1, br2, xi1, xi2, xr1, xr2, ci21, ci22, cr21, cr22,
	    li21, csi, ui11, lr21, ui12, ui22;
#define civ (equiv_0)
    double csr, ur11, ur12, ur22;
#define crv (equiv_1)
    double bbnd, cmax, ui11r, ui12s, temp, ur11r, ur12s, u22abs;
    int icmax;
    double bnorm, cnorm, smini;
    extern double dlamch_(char *);
    extern /* Subroutine */ int dladiv_(double *, double *, double *, double *
	    , double *, double *);
    double bignum, smlnum;

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
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Equivalences ..
    //    ..
    //    .. Data statements ..
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;

    // Function Body
    //    ..
    //    .. Executable Statements ..
    //
    //    Compute BIGNUM
    //
    smlnum = 2. * dlamch_("Safe minimum");
    bignum = 1. / smlnum;
    smini = max(*smin,smlnum);
    //
    //    Don't check for input errors
    //
    *info = 0;
    //
    //    Standard Initializations
    //
    *scale = 1.;
    if (*na == 1) {
	//
	//       1 x 1  (i.e., scalar) system   C X = B
	//
	if (*nw == 1) {
	    //
	    //          Real 1x1 system.
	    //
	    //          C = ca A - w D
	    //
	    csr = *ca * a[a_dim1 + 1] - *wr * *d1;
	    cnorm = abs(csr);
	    //
	    //          If | C | < SMINI, use C = SMINI
	    //
	    if (cnorm < smini) {
		csr = smini;
		cnorm = smini;
		*info = 1;
	    }
	    //
	    //          Check scaling for  X = B / C
	    //
	    bnorm = (d__1 = b[b_dim1 + 1], abs(d__1));
	    if (cnorm < 1. && bnorm > 1.) {
		if (bnorm > bignum * cnorm) {
		    *scale = 1. / bnorm;
		}
	    }
	    //
	    //          Compute X
	    //
	    x[x_dim1 + 1] = b[b_dim1 + 1] * *scale / csr;
	    *xnorm = (d__1 = x[x_dim1 + 1], abs(d__1));
	} else {
	    //
	    //          Complex 1x1 system (w is complex)
	    //
	    //          C = ca A - w D
	    //
	    csr = *ca * a[a_dim1 + 1] - *wr * *d1;
	    csi = -(*wi) * *d1;
	    cnorm = abs(csr) + abs(csi);
	    //
	    //          If | C | < SMINI, use C = SMINI
	    //
	    if (cnorm < smini) {
		csr = smini;
		csi = 0.;
		cnorm = smini;
		*info = 1;
	    }
	    //
	    //          Check scaling for  X = B / C
	    //
	    bnorm = (d__1 = b[b_dim1 + 1], abs(d__1)) + (d__2 = b[(b_dim1 <<
		    1) + 1], abs(d__2));
	    if (cnorm < 1. && bnorm > 1.) {
		if (bnorm > bignum * cnorm) {
		    *scale = 1. / bnorm;
		}
	    }
	    //
	    //          Compute X
	    //
	    d__1 = *scale * b[b_dim1 + 1];
	    d__2 = *scale * b[(b_dim1 << 1) + 1];
	    dladiv_(&d__1, &d__2, &csr, &csi, &x[x_dim1 + 1], &x[(x_dim1 << 1)
		     + 1]);
	    *xnorm = (d__1 = x[x_dim1 + 1], abs(d__1)) + (d__2 = x[(x_dim1 <<
		    1) + 1], abs(d__2));
	}
    } else {
	//
	//       2x2 System
	//
	//       Compute the real part of  C = ca A - w D  (or  ca A**T - w D )
	//
	cr[0] = *ca * a[a_dim1 + 1] - *wr * *d1;
	cr[3] = *ca * a[(a_dim1 << 1) + 2] - *wr * *d2;
	if (*ltrans) {
	    cr[2] = *ca * a[a_dim1 + 2];
	    cr[1] = *ca * a[(a_dim1 << 1) + 1];
	} else {
	    cr[1] = *ca * a[a_dim1 + 2];
	    cr[2] = *ca * a[(a_dim1 << 1) + 1];
	}
	if (*nw == 1) {
	    //
	    //          Real 2x2 system  (w is real)
	    //
	    //          Find the largest element in C
	    //
	    cmax = 0.;
	    icmax = 0;
	    for (j = 1; j <= 4; ++j) {
		if ((d__1 = crv[j - 1], abs(d__1)) > cmax) {
		    cmax = (d__1 = crv[j - 1], abs(d__1));
		    icmax = j;
		}
// L10:
	    }
	    //
	    //          If norm(C) < SMINI, use SMINI*identity.
	    //
	    if (cmax < smini) {
		// Computing MAX
		d__3 = (d__1 = b[b_dim1 + 1], abs(d__1)), d__4 = (d__2 = b[
			b_dim1 + 2], abs(d__2));
		bnorm = max(d__3,d__4);
		if (smini < 1. && bnorm > 1.) {
		    if (bnorm > bignum * smini) {
			*scale = 1. / bnorm;
		    }
		}
		temp = *scale / smini;
		x[x_dim1 + 1] = temp * b[b_dim1 + 1];
		x[x_dim1 + 2] = temp * b[b_dim1 + 2];
		*xnorm = temp * bnorm;
		*info = 1;
		return 0;
	    }
	    //
	    //          Gaussian elimination with complete pivoting.
	    //
	    ur11 = crv[icmax - 1];
	    cr21 = crv[ipivot[(icmax << 2) - 3] - 1];
	    ur12 = crv[ipivot[(icmax << 2) - 2] - 1];
	    cr22 = crv[ipivot[(icmax << 2) - 1] - 1];
	    ur11r = 1. / ur11;
	    lr21 = ur11r * cr21;
	    ur22 = cr22 - ur12 * lr21;
	    //
	    //          If smaller pivot < SMINI, use SMINI
	    //
	    if (abs(ur22) < smini) {
		ur22 = smini;
		*info = 1;
	    }
	    if (rswap[icmax - 1]) {
		br1 = b[b_dim1 + 2];
		br2 = b[b_dim1 + 1];
	    } else {
		br1 = b[b_dim1 + 1];
		br2 = b[b_dim1 + 2];
	    }
	    br2 -= lr21 * br1;
	    // Computing MAX
	    d__2 = (d__1 = br1 * (ur22 * ur11r), abs(d__1)), d__3 = abs(br2);
	    bbnd = max(d__2,d__3);
	    if (bbnd > 1. && abs(ur22) < 1.) {
		if (bbnd >= bignum * abs(ur22)) {
		    *scale = 1. / bbnd;
		}
	    }
	    xr2 = br2 * *scale / ur22;
	    xr1 = *scale * br1 * ur11r - xr2 * (ur11r * ur12);
	    if (zswap[icmax - 1]) {
		x[x_dim1 + 1] = xr2;
		x[x_dim1 + 2] = xr1;
	    } else {
		x[x_dim1 + 1] = xr1;
		x[x_dim1 + 2] = xr2;
	    }
	    // Computing MAX
	    d__1 = abs(xr1), d__2 = abs(xr2);
	    *xnorm = max(d__1,d__2);
	    //
	    //          Further scaling if  norm(A) norm(X) > overflow
	    //
	    if (*xnorm > 1. && cmax > 1.) {
		if (*xnorm > bignum / cmax) {
		    temp = cmax / bignum;
		    x[x_dim1 + 1] = temp * x[x_dim1 + 1];
		    x[x_dim1 + 2] = temp * x[x_dim1 + 2];
		    *xnorm = temp * *xnorm;
		    *scale = temp * *scale;
		}
	    }
	} else {
	    //
	    //          Complex 2x2 system  (w is complex)
	    //
	    //          Find the largest element in C
	    //
	    ci[0] = -(*wi) * *d1;
	    ci[1] = 0.;
	    ci[2] = 0.;
	    ci[3] = -(*wi) * *d2;
	    cmax = 0.;
	    icmax = 0;
	    for (j = 1; j <= 4; ++j) {
		if ((d__1 = crv[j - 1], abs(d__1)) + (d__2 = civ[j - 1], abs(
			d__2)) > cmax) {
		    cmax = (d__1 = crv[j - 1], abs(d__1)) + (d__2 = civ[j - 1]
			    , abs(d__2));
		    icmax = j;
		}
// L20:
	    }
	    //
	    //          If norm(C) < SMINI, use SMINI*identity.
	    //
	    if (cmax < smini) {
		// Computing MAX
		d__5 = (d__1 = b[b_dim1 + 1], abs(d__1)) + (d__2 = b[(b_dim1
			<< 1) + 1], abs(d__2)), d__6 = (d__3 = b[b_dim1 + 2],
			abs(d__3)) + (d__4 = b[(b_dim1 << 1) + 2], abs(d__4));
		bnorm = max(d__5,d__6);
		if (smini < 1. && bnorm > 1.) {
		    if (bnorm > bignum * smini) {
			*scale = 1. / bnorm;
		    }
		}
		temp = *scale / smini;
		x[x_dim1 + 1] = temp * b[b_dim1 + 1];
		x[x_dim1 + 2] = temp * b[b_dim1 + 2];
		x[(x_dim1 << 1) + 1] = temp * b[(b_dim1 << 1) + 1];
		x[(x_dim1 << 1) + 2] = temp * b[(b_dim1 << 1) + 2];
		*xnorm = temp * bnorm;
		*info = 1;
		return 0;
	    }
	    //
	    //          Gaussian elimination with complete pivoting.
	    //
	    ur11 = crv[icmax - 1];
	    ui11 = civ[icmax - 1];
	    cr21 = crv[ipivot[(icmax << 2) - 3] - 1];
	    ci21 = civ[ipivot[(icmax << 2) - 3] - 1];
	    ur12 = crv[ipivot[(icmax << 2) - 2] - 1];
	    ui12 = civ[ipivot[(icmax << 2) - 2] - 1];
	    cr22 = crv[ipivot[(icmax << 2) - 1] - 1];
	    ci22 = civ[ipivot[(icmax << 2) - 1] - 1];
	    if (icmax == 1 || icmax == 4) {
		//
		//             Code when off-diagonals of pivoted C are real
		//
		if (abs(ur11) > abs(ui11)) {
		    temp = ui11 / ur11;
		    // Computing 2nd power
		    d__1 = temp;
		    ur11r = 1. / (ur11 * (d__1 * d__1 + 1.));
		    ui11r = -temp * ur11r;
		} else {
		    temp = ur11 / ui11;
		    // Computing 2nd power
		    d__1 = temp;
		    ui11r = -1. / (ui11 * (d__1 * d__1 + 1.));
		    ur11r = -temp * ui11r;
		}
		lr21 = cr21 * ur11r;
		li21 = cr21 * ui11r;
		ur12s = ur12 * ur11r;
		ui12s = ur12 * ui11r;
		ur22 = cr22 - ur12 * lr21;
		ui22 = ci22 - ur12 * li21;
	    } else {
		//
		//             Code when diagonals of pivoted C are real
		//
		ur11r = 1. / ur11;
		ui11r = 0.;
		lr21 = cr21 * ur11r;
		li21 = ci21 * ur11r;
		ur12s = ur12 * ur11r;
		ui12s = ui12 * ur11r;
		ur22 = cr22 - ur12 * lr21 + ui12 * li21;
		ui22 = -ur12 * li21 - ui12 * lr21;
	    }
	    u22abs = abs(ur22) + abs(ui22);
	    //
	    //          If smaller pivot < SMINI, use SMINI
	    //
	    if (u22abs < smini) {
		ur22 = smini;
		ui22 = 0.;
		*info = 1;
	    }
	    if (rswap[icmax - 1]) {
		br2 = b[b_dim1 + 1];
		br1 = b[b_dim1 + 2];
		bi2 = b[(b_dim1 << 1) + 1];
		bi1 = b[(b_dim1 << 1) + 2];
	    } else {
		br1 = b[b_dim1 + 1];
		br2 = b[b_dim1 + 2];
		bi1 = b[(b_dim1 << 1) + 1];
		bi2 = b[(b_dim1 << 1) + 2];
	    }
	    br2 = br2 - lr21 * br1 + li21 * bi1;
	    bi2 = bi2 - li21 * br1 - lr21 * bi1;
	    // Computing MAX
	    d__1 = (abs(br1) + abs(bi1)) * (u22abs * (abs(ur11r) + abs(ui11r))
		    ), d__2 = abs(br2) + abs(bi2);
	    bbnd = max(d__1,d__2);
	    if (bbnd > 1. && u22abs < 1.) {
		if (bbnd >= bignum * u22abs) {
		    *scale = 1. / bbnd;
		    br1 = *scale * br1;
		    bi1 = *scale * bi1;
		    br2 = *scale * br2;
		    bi2 = *scale * bi2;
		}
	    }
	    dladiv_(&br2, &bi2, &ur22, &ui22, &xr2, &xi2);
	    xr1 = ur11r * br1 - ui11r * bi1 - ur12s * xr2 + ui12s * xi2;
	    xi1 = ui11r * br1 + ur11r * bi1 - ui12s * xr2 - ur12s * xi2;
	    if (zswap[icmax - 1]) {
		x[x_dim1 + 1] = xr2;
		x[x_dim1 + 2] = xr1;
		x[(x_dim1 << 1) + 1] = xi2;
		x[(x_dim1 << 1) + 2] = xi1;
	    } else {
		x[x_dim1 + 1] = xr1;
		x[x_dim1 + 2] = xr2;
		x[(x_dim1 << 1) + 1] = xi1;
		x[(x_dim1 << 1) + 2] = xi2;
	    }
	    // Computing MAX
	    d__1 = abs(xr1) + abs(xi1), d__2 = abs(xr2) + abs(xi2);
	    *xnorm = max(d__1,d__2);
	    //
	    //          Further scaling if  norm(A) norm(X) > overflow
	    //
	    if (*xnorm > 1. && cmax > 1.) {
		if (*xnorm > bignum / cmax) {
		    temp = cmax / bignum;
		    x[x_dim1 + 1] = temp * x[x_dim1 + 1];
		    x[x_dim1 + 2] = temp * x[x_dim1 + 2];
		    x[(x_dim1 << 1) + 1] = temp * x[(x_dim1 << 1) + 1];
		    x[(x_dim1 << 1) + 2] = temp * x[(x_dim1 << 1) + 2];
		    *xnorm = temp * *xnorm;
		    *scale = temp * *scale;
		}
	    }
	}
    }
    return 0;
    //
    //    End of DLALN2
    //
} // dlaln2_

#undef crv
#undef civ
#undef cr
#undef ci


/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric matrix in standard form.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLANV2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlanv2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlanv2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlanv2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLANV2( A, B, C, D, RT1R, RT1I, RT2R, RT2I, CS, SN )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   A, B, C, CS, D, RT1I, RT1R, RT2I, RT2R, SN
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLANV2 computes the Schur factorization of a real 2-by-2 nonsymmetric
//> matrix in standard form:
//>
//>      [ A  B ] = [ CS -SN ] [ AA  BB ] [ CS  SN ]
//>      [ C  D ]   [ SN  CS ] [ CC  DD ] [-SN  CS ]
//>
//> where either
//> 1) CC = 0 so that AA and DD are real eigenvalues of the matrix, or
//> 2) AA = DD and BB*CC < 0, so that AA + or - sqrt(BB*CC) are complex
//> conjugate eigenvalues.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] B
//> \verbatim
//>          B is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION
//>          On entry, the elements of the input matrix.
//>          On exit, they are overwritten by the elements of the
//>          standardised Schur form.
//> \endverbatim
//>
//> \param[out] RT1R
//> \verbatim
//>          RT1R is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] RT1I
//> \verbatim
//>          RT1I is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] RT2R
//> \verbatim
//>          RT2R is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] RT2I
//> \verbatim
//>          RT2I is DOUBLE PRECISION
//>          The real and imaginary parts of the eigenvalues. If the
//>          eigenvalues are a complex conjugate pair, RT1I > 0.
//> \endverbatim
//>
//> \param[out] CS
//> \verbatim
//>          CS is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] SN
//> \verbatim
//>          SN is DOUBLE PRECISION
//>          Parameters of the rotation matrix.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Modified by V. Sima, Research Institute for Informatics, Bucharest,
//>  Romania, to reduce the risk of cancellation errors,
//>  when computing real eigenvalues, and to ensure, if possible, that
//>  abs(RT1R) >= abs(RT2R).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlanv2_(double *a, double *b, double *c__, double *d__,
	double *rt1r, double *rt1i, double *rt2r, double *rt2i, double *cs,
	double *sn)
{
    // Table of constant values
    double c_b3 = 1.;

    // System generated locals
    double d__1, d__2;

    // Local variables
    double p, z__, aa, bb, cc, dd, cs1, sn1, sab, sac, eps, tau, temp, scale,
	    bcmax, bcmis, sigma;
    extern double dlapy2_(double *, double *), dlamch_(char *);

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
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
    //    .. Executable Statements ..
    //
    eps = dlamch_("P");
    if (*c__ == 0.) {
	*cs = 1.;
	*sn = 0.;
    } else if (*b == 0.) {
	//
	//       Swap rows and columns
	//
	*cs = 0.;
	*sn = 1.;
	temp = *d__;
	*d__ = *a;
	*a = temp;
	*b = -(*c__);
	*c__ = 0.;
    } else if (*a - *d__ == 0. && d_sign(&c_b3, b) != d_sign(&c_b3, c__)) {
	*cs = 1.;
	*sn = 0.;
    } else {
	temp = *a - *d__;
	p = temp * .5;
	// Computing MAX
	d__1 = abs(*b), d__2 = abs(*c__);
	bcmax = max(d__1,d__2);
	// Computing MIN
	d__1 = abs(*b), d__2 = abs(*c__);
	bcmis = min(d__1,d__2) * d_sign(&c_b3, b) * d_sign(&c_b3, c__);
	// Computing MAX
	d__1 = abs(p);
	scale = max(d__1,bcmax);
	z__ = p / scale * p + bcmax / scale * bcmis;
	//
	//       If Z is of the order of the machine accuracy, postpone the
	//       decision on the nature of eigenvalues
	//
	if (z__ >= eps * 4.) {
	    //
	    //          Real eigenvalues. Compute A and D.
	    //
	    d__1 = sqrt(scale) * sqrt(z__);
	    z__ = p + d_sign(&d__1, &p);
	    *a = *d__ + z__;
	    *d__ -= bcmax / z__ * bcmis;
	    //
	    //          Compute B and the rotation matrix
	    //
	    tau = dlapy2_(c__, &z__);
	    *cs = z__ / tau;
	    *sn = *c__ / tau;
	    *b -= *c__;
	    *c__ = 0.;
	} else {
	    //
	    //          Complex eigenvalues, or real (almost) equal eigenvalues.
	    //          Make diagonal elements equal.
	    //
	    sigma = *b + *c__;
	    tau = dlapy2_(&sigma, &temp);
	    *cs = sqrt((abs(sigma) / tau + 1.) * .5);
	    *sn = -(p / (tau * *cs)) * d_sign(&c_b3, &sigma);
	    //
	    //          Compute [ AA  BB ] = [ A  B ] [ CS -SN ]
	    //                  [ CC  DD ]   [ C  D ] [ SN  CS ]
	    //
	    aa = *a * *cs + *b * *sn;
	    bb = -(*a) * *sn + *b * *cs;
	    cc = *c__ * *cs + *d__ * *sn;
	    dd = -(*c__) * *sn + *d__ * *cs;
	    //
	    //          Compute [ A  B ] = [ CS  SN ] [ AA  BB ]
	    //                  [ C  D ]   [-SN  CS ] [ CC  DD ]
	    //
	    *a = aa * *cs + cc * *sn;
	    *b = bb * *cs + dd * *sn;
	    *c__ = -aa * *sn + cc * *cs;
	    *d__ = -bb * *sn + dd * *cs;
	    temp = (*a + *d__) * .5;
	    *a = temp;
	    *d__ = temp;
	    if (*c__ != 0.) {
		if (*b != 0.) {
		    if (d_sign(&c_b3, b) == d_sign(&c_b3, c__)) {
			//
			//                   Real eigenvalues: reduce to upper triangular form
			//
			sab = sqrt((abs(*b)));
			sac = sqrt((abs(*c__)));
			d__1 = sab * sac;
			p = d_sign(&d__1, c__);
			tau = 1. / sqrt((d__1 = *b + *c__, abs(d__1)));
			*a = temp + p;
			*d__ = temp - p;
			*b -= *c__;
			*c__ = 0.;
			cs1 = sab * tau;
			sn1 = sac * tau;
			temp = *cs * cs1 - *sn * sn1;
			*sn = *cs * sn1 + *sn * cs1;
			*cs = temp;
		    }
		} else {
		    *b = -(*c__);
		    *c__ = 0.;
		    temp = *cs;
		    *cs = -(*sn);
		    *sn = temp;
		}
	    }
	}
    }
    //
    //    Store eigenvalues in (RT1R,RT1I) and (RT2R,RT2I).
    //
    *rt1r = *a;
    *rt2r = *d__;
    if (*c__ == 0.) {
	*rt1i = 0.;
	*rt2i = 0.;
    } else {
	*rt1i = sqrt((abs(*b))) * sqrt((abs(*c__)));
	*rt2i = -(*rt1i);
    }
    return 0;
    //
    //    End of DLANV2
    //
} // dlanv2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAQR0 computes the eigenvalues of a Hessenberg matrix, and optionally the matrices from the Schur decomposition.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAQR0 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaqr0.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaqr0.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaqr0.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAQR0( WANTT, WANTZ, N, ILO, IHI, H, LDH, WR, WI,
//                         ILOZ, IHIZ, Z, LDZ, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, IHIZ, ILO, ILOZ, INFO, LDH, LDZ, LWORK, N
//      LOGICAL            WANTT, WANTZ
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), WI( * ), WORK( * ), WR( * ),
//     $                   Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DLAQR0 computes the eigenvalues of a Hessenberg matrix H
//>    and, optionally, the matrices T and Z from the Schur decomposition
//>    H = Z T Z**T, where T is an upper quasi-triangular matrix (the
//>    Schur form), and Z is the orthogonal matrix of Schur vectors.
//>
//>    Optionally Z may be postmultiplied into an input orthogonal
//>    matrix Q so that this routine can give the Schur factorization
//>    of a matrix A which has been reduced to the Hessenberg form H
//>    by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] WANTT
//> \verbatim
//>          WANTT is LOGICAL
//>          = .TRUE. : the full Schur form T is required;
//>          = .FALSE.: only eigenvalues are required.
//> \endverbatim
//>
//> \param[in] WANTZ
//> \verbatim
//>          WANTZ is LOGICAL
//>          = .TRUE. : the matrix of Schur vectors Z is required;
//>          = .FALSE.: Schur vectors are not required.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           The order of the matrix H.  N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>           It is assumed that H is already upper triangular in rows
//>           and columns 1:ILO-1 and IHI+1:N and, if ILO > 1,
//>           H(ILO,ILO-1) is zero. ILO and IHI are normally set by a
//>           previous call to DGEBAL, and then passed to DGEHRD when the
//>           matrix output by DGEBAL is reduced to Hessenberg form.
//>           Otherwise, ILO and IHI should be set to 1 and N,
//>           respectively.  If N > 0, then 1 <= ILO <= IHI <= N.
//>           If N = 0, then ILO = 1 and IHI = 0.
//> \endverbatim
//>
//> \param[in,out] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>           On entry, the upper Hessenberg matrix H.
//>           On exit, if INFO = 0 and WANTT is .TRUE., then H contains
//>           the upper quasi-triangular matrix T from the Schur
//>           decomposition (the Schur form); 2-by-2 diagonal blocks
//>           (corresponding to complex conjugate pairs of eigenvalues)
//>           are returned in standard form, with H(i,i) = H(i+1,i+1)
//>           and H(i+1,i)*H(i,i+1) < 0. If INFO = 0 and WANTT is
//>           .FALSE., then the contents of H are unspecified on exit.
//>           (The output value of H when INFO > 0 is given under the
//>           description of INFO below.)
//>
//>           This subroutine may explicitly set H(i,j) = 0 for i > j and
//>           j = 1, 2, ... ILO-1 or j = IHI+1, IHI+2, ... N.
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>           The leading dimension of the array H. LDH >= max(1,N).
//> \endverbatim
//>
//> \param[out] WR
//> \verbatim
//>          WR is DOUBLE PRECISION array, dimension (IHI)
//> \endverbatim
//>
//> \param[out] WI
//> \verbatim
//>          WI is DOUBLE PRECISION array, dimension (IHI)
//>           The real and imaginary parts, respectively, of the computed
//>           eigenvalues of H(ILO:IHI,ILO:IHI) are stored in WR(ILO:IHI)
//>           and WI(ILO:IHI). If two eigenvalues are computed as a
//>           complex conjugate pair, they are stored in consecutive
//>           elements of WR and WI, say the i-th and (i+1)th, with
//>           WI(i) > 0 and WI(i+1) < 0. If WANTT is .TRUE., then
//>           the eigenvalues are stored in the same order as on the
//>           diagonal of the Schur form returned in H, with
//>           WR(i) = H(i,i) and, if H(i:i+1,i:i+1) is a 2-by-2 diagonal
//>           block, WI(i) = sqrt(-H(i+1,i)*H(i,i+1)) and
//>           WI(i+1) = -WI(i).
//> \endverbatim
//>
//> \param[in] ILOZ
//> \verbatim
//>          ILOZ is INTEGER
//> \endverbatim
//>
//> \param[in] IHIZ
//> \verbatim
//>          IHIZ is INTEGER
//>           Specify the rows of Z to which transformations must be
//>           applied if WANTZ is .TRUE..
//>           1 <= ILOZ <= ILO; IHI <= IHIZ <= N.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ,IHI)
//>           If WANTZ is .FALSE., then Z is not referenced.
//>           If WANTZ is .TRUE., then Z(ILO:IHI,ILOZ:IHIZ) is
//>           replaced by Z(ILO:IHI,ILOZ:IHIZ)*U where U is the
//>           orthogonal Schur factor of H(ILO:IHI,ILO:IHI).
//>           (The output value of Z when INFO > 0 is given under
//>           the description of INFO below.)
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>           The leading dimension of the array Z.  if WANTZ is .TRUE.
//>           then LDZ >= MAX(1,IHIZ).  Otherwise, LDZ >= 1.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension LWORK
//>           On exit, if LWORK = -1, WORK(1) returns an estimate of
//>           the optimal value for LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>           The dimension of the array WORK.  LWORK >= max(1,N)
//>           is sufficient, but LWORK typically as large as 6*N may
//>           be required for optimal performance.  A workspace query
//>           to determine the optimal workspace size is recommended.
//>
//>           If LWORK = -1, then DLAQR0 does a workspace query.
//>           In this case, DLAQR0 checks the input parameters and
//>           estimates the optimal workspace size for the given
//>           values of N, ILO and IHI.  The estimate is returned
//>           in WORK(1).  No error message related to LWORK is
//>           issued by XERBLA.  Neither H nor Z are accessed.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>             = 0:  successful exit
//>             > 0:  if INFO = i, DLAQR0 failed to compute all of
//>                the eigenvalues.  Elements 1:ilo-1 and i+1:n of WR
//>                and WI contain those eigenvalues which have been
//>                successfully computed.  (Failures are rare.)
//>
//>                If INFO > 0 and WANT is .FALSE., then on exit,
//>                the remaining unconverged eigenvalues are the eigen-
//>                values of the upper Hessenberg matrix rows and
//>                columns ILO through INFO of the final, output
//>                value of H.
//>
//>                If INFO > 0 and WANTT is .TRUE., then on exit
//>
//>           (*)  (initial value of H)*U  = U*(final value of H)
//>
//>                where U is an orthogonal matrix.  The final
//>                value of H is upper Hessenberg and quasi-triangular
//>                in rows and columns INFO+1 through IHI.
//>
//>                If INFO > 0 and WANTZ is .TRUE., then on exit
//>
//>                  (final value of Z(ILO:IHI,ILOZ:IHIZ)
//>                   =  (initial value of Z(ILO:IHI,ILOZ:IHIZ)*U
//>
//>                where U is the orthogonal matrix in (*) (regard-
//>                less of the value of WANTT.)
//>
//>                If INFO > 0 and WANTZ is .FALSE., then Z is not
//>                accessed.
//> \endverbatim
//
//> \par Contributors:
// ==================
//>
//>       Karen Braman and Ralph Byers, Department of Mathematics,
//>       University of Kansas, USA
//
//> \par References:
// ================
//>
//>       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//>       Algorithm Part I: Maintaining Well Focused Shifts, and Level 3
//>       Performance, SIAM Journal of Matrix Analysis, volume 23, pages
//>       929--947, 2002.
//> \n
//>       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//>       Algorithm Part II: Aggressive Early Deflation, SIAM Journal
//>       of Matrix Analysis, volume 23, pages 948--973, 2002.
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
/* Subroutine */ int dlaqr0_(int *wantt, int *wantz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, int *iloz, int *
	ihiz, double *z__, int *ldz, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__13 = 13;
    int c__15 = 15;
    int c_n1 = -1;
    int c__12 = 12;
    int c__14 = 14;
    int c__16 = 16;
    int c_false = FALSE_;
    int c__1 = 1;
    int c__3 = 3;

    // System generated locals
    int h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    double d__1, d__2, d__3, d__4;

    // Local variables
    int i__, k;
    double aa, bb, cc, dd;
    int ld;
    double cs;
    int nh, it, ks, kt;
    double sn;
    int ku, kv, ls, ns;
    double ss;
    int nw, inf, kdu, nho, nve, kwh, nsr, nwr, kwv, ndec, ndfl, kbot, nmin;
    double swap;
    int ktop;
    double zdum[1]	/* was [1][1] */;
    int kacc22, itmax, nsmax, nwmax, kwtop;
    extern /* Subroutine */ int dlanv2_(double *, double *, double *, double *
	    , double *, double *, double *, double *, double *, double *),
	    dlaqr3_(int *, int *, int *, int *, int *, int *, double *, int *,
	     int *, int *, double *, int *, int *, int *, double *, double *,
	    double *, int *, int *, double *, int *, int *, double *, int *,
	    double *, int *), dlaqr4_(int *, int *, int *, int *, int *,
	    double *, int *, double *, double *, int *, int *, double *, int *
	    , double *, int *, int *), dlaqr5_(int *, int *, int *, int *,
	    int *, int *, int *, double *, double *, double *, int *, int *,
	    int *, double *, int *, double *, int *, double *, int *, int *,
	    double *, int *, int *, double *, int *);
    int nibble;
    extern /* Subroutine */ int dlahqr_(int *, int *, int *, int *, int *,
	    double *, int *, double *, double *, int *, int *, double *, int *
	    , int *), dlacpy_(char *, int *, int *, double *, int *, double *,
	     int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    char jbcmpz[2+1]={'\0'};
    int nwupbd;
    int sorted;
    int lwkopt;

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
    // ================================================================
    //
    //    .. Parameters ..
    //
    //    ==== Matrices of order NTINY or smaller must be processed by
    //    .    DLAHQR because of insufficient subdiagonal scratch space.
    //    .    (This is a hard limit.) ====
    //
    //    ==== Exceptional deflation windows:  try to cure rare
    //    .    slow convergence by varying the size of the
    //    .    deflation window after KEXNW iterations. ====
    //
    //    ==== Exceptional shifts: try to cure rare slow convergence
    //    .    with ad-hoc exceptional shifts every KEXSH iterations.
    //    .    ====
    //
    //    ==== The constants WILK1 and WILK2 are used to form the
    //    .    exceptional shifts. ====
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    // Parameter adjustments
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    // Function Body
    *info = 0;
    //
    //    ==== Quick return for N = 0: nothing to do. ====
    //
    if (*n == 0) {
	work[1] = 1.;
	return 0;
    }
    if (*n <= 11) {
	//
	//       ==== Tiny matrices must use DLAHQR. ====
	//
	lwkopt = 1;
	if (*lwork != -1) {
	    dlahqr_(wantt, wantz, n, ilo, ihi, &h__[h_offset], ldh, &wr[1], &
		    wi[1], iloz, ihiz, &z__[z_offset], ldz, info);
	}
    } else {
	//
	//       ==== Use small bulge multi-shift QR with aggressive early
	//       .    deflation on larger-than-tiny matrices. ====
	//
	//       ==== Hope for the best. ====
	//
	*info = 0;
	//
	//       ==== Set up job flags for ILAENV. ====
	//
	if (*wantt) {
	    *(unsigned char *)jbcmpz = 'S';
	} else {
	    *(unsigned char *)jbcmpz = 'E';
	}
	if (*wantz) {
	    *(unsigned char *)&jbcmpz[1] = 'V';
	} else {
	    *(unsigned char *)&jbcmpz[1] = 'N';
	}
	//
	//       ==== NWR = recommended deflation window size.  At this
	//       .    point,  N .GT. NTINY = 11, so there is enough
	//       .    subdiagonal workspace for NWR.GE.2 as required.
	//       .    (In fact, there is enough subdiagonal space for
	//       .    NWR.GE.3.) ====
	//
	nwr = ilaenv_(&c__13, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
	nwr = max(2,nwr);
	// Computing MIN
	i__1 = *ihi - *ilo + 1, i__2 = (*n - 1) / 3, i__1 = min(i__1,i__2);
	nwr = min(i__1,nwr);
	//
	//       ==== NSR = recommended number of simultaneous shifts.
	//       .    At this point N .GT. NTINY = 11, so there is at
	//       .    enough subdiagonal workspace for NSR to be even
	//       .    and greater than or equal to two as required. ====
	//
	nsr = ilaenv_(&c__15, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
	// Computing MIN
	i__1 = nsr, i__2 = (*n + 6) / 9, i__1 = min(i__1,i__2), i__2 = *ihi -
		*ilo;
	nsr = min(i__1,i__2);
	// Computing MAX
	i__1 = 2, i__2 = nsr - nsr % 2;
	nsr = max(i__1,i__2);
	//
	//       ==== Estimate optimal workspace ====
	//
	//       ==== Workspace query call to DLAQR3 ====
	//
	i__1 = nwr + 1;
	dlaqr3_(wantt, wantz, n, ilo, ihi, &i__1, &h__[h_offset], ldh, iloz,
		ihiz, &z__[z_offset], ldz, &ls, &ld, &wr[1], &wi[1], &h__[
		h_offset], ldh, n, &h__[h_offset], ldh, n, &h__[h_offset],
		ldh, &work[1], &c_n1);
	//
	//       ==== Optimal workspace = MAX(DLAQR5, DLAQR3) ====
	//
	// Computing MAX
	i__1 = nsr * 3 / 2, i__2 = (int) work[1];
	lwkopt = max(i__1,i__2);
	//
	//       ==== Quick return in case of workspace query. ====
	//
	if (*lwork == -1) {
	    work[1] = (double) lwkopt;
	    return 0;
	}
	//
	//       ==== DLAHQR/DLAQR0 crossover point ====
	//
	nmin = ilaenv_(&c__12, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
	nmin = max(11,nmin);
	//
	//       ==== Nibble crossover point ====
	//
	nibble = ilaenv_(&c__14, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
	nibble = max(0,nibble);
	//
	//       ==== Accumulate reflections during ttswp?  Use block
	//       .    2-by-2 structure during matrix-matrix multiply? ====
	//
	kacc22 = ilaenv_(&c__16, "DLAQR0", jbcmpz, n, ilo, ihi, lwork);
	kacc22 = max(0,kacc22);
	kacc22 = min(2,kacc22);
	//
	//       ==== NWMAX = the largest possible deflation window for
	//       .    which there is sufficient workspace. ====
	//
	// Computing MIN
	i__1 = (*n - 1) / 3, i__2 = *lwork / 2;
	nwmax = min(i__1,i__2);
	nw = nwmax;
	//
	//       ==== NSMAX = the Largest number of simultaneous shifts
	//       .    for which there is sufficient workspace. ====
	//
	// Computing MIN
	i__1 = (*n + 6) / 9, i__2 = (*lwork << 1) / 3;
	nsmax = min(i__1,i__2);
	nsmax -= nsmax % 2;
	//
	//       ==== NDFL: an iteration count restarted at deflation. ====
	//
	ndfl = 1;
	//
	//       ==== ITMAX = iteration limit ====
	//
	// Computing MAX
	i__1 = 10, i__2 = *ihi - *ilo + 1;
	itmax = max(i__1,i__2) * 30;
	//
	//       ==== Last row and column in the active block ====
	//
	kbot = *ihi;
	//
	//       ==== Main Loop ====
	//
	i__1 = itmax;
	for (it = 1; it <= i__1; ++it) {
	    //
	    //          ==== Done when KBOT falls below ILO ====
	    //
	    if (kbot < *ilo) {
		goto L90;
	    }
	    //
	    //          ==== Locate active block ====
	    //
	    i__2 = *ilo + 1;
	    for (k = kbot; k >= i__2; --k) {
		if (h__[k + (k - 1) * h_dim1] == 0.) {
		    goto L20;
		}
// L10:
	    }
	    k = *ilo;
L20:
	    ktop = k;
	    //
	    //          ==== Select deflation window size:
	    //          .    Typical Case:
	    //          .      If possible and advisable, nibble the entire
	    //          .      active block.  If not, use size MIN(NWR,NWMAX)
	    //          .      or MIN(NWR+1,NWMAX) depending upon which has
	    //          .      the smaller corresponding subdiagonal entry
	    //          .      (a heuristic).
	    //          .
	    //          .    Exceptional Case:
	    //          .      If there have been no deflations in KEXNW or
	    //          .      more iterations, then vary the deflation window
	    //          .      size.   At first, because, larger windows are,
	    //          .      in general, more powerful than smaller ones,
	    //          .      rapidly increase the window to the maximum possible.
	    //          .      Then, gradually reduce the window size. ====
	    //
	    nh = kbot - ktop + 1;
	    nwupbd = min(nh,nwmax);
	    if (ndfl < 5) {
		nw = min(nwupbd,nwr);
	    } else {
		// Computing MIN
		i__2 = nwupbd, i__3 = nw << 1;
		nw = min(i__2,i__3);
	    }
	    if (nw < nwmax) {
		if (nw >= nh - 1) {
		    nw = nh;
		} else {
		    kwtop = kbot - nw + 1;
		    if ((d__1 = h__[kwtop + (kwtop - 1) * h_dim1], abs(d__1))
			    > (d__2 = h__[kwtop - 1 + (kwtop - 2) * h_dim1],
			    abs(d__2))) {
			++nw;
		    }
		}
	    }
	    if (ndfl < 5) {
		ndec = -1;
	    } else if (ndec >= 0 || nw >= nwupbd) {
		++ndec;
		if (nw - ndec < 2) {
		    ndec = 0;
		}
		nw -= ndec;
	    }
	    //
	    //          ==== Aggressive early deflation:
	    //          .    split workspace under the subdiagonal into
	    //          .      - an nw-by-nw work array V in the lower
	    //          .        left-hand-corner,
	    //          .      - an NW-by-at-least-NW-but-more-is-better
	    //          .        (NW-by-NHO) horizontal work array along
	    //          .        the bottom edge,
	    //          .      - an at-least-NW-but-more-is-better (NHV-by-NW)
	    //          .        vertical work array along the left-hand-edge.
	    //          .        ====
	    //
	    kv = *n - nw + 1;
	    kt = nw + 1;
	    nho = *n - nw - 1 - kt + 1;
	    kwv = nw + 2;
	    nve = *n - nw - kwv + 1;
	    //
	    //          ==== Aggressive early deflation ====
	    //
	    dlaqr3_(wantt, wantz, n, &ktop, &kbot, &nw, &h__[h_offset], ldh,
		    iloz, ihiz, &z__[z_offset], ldz, &ls, &ld, &wr[1], &wi[1],
		     &h__[kv + h_dim1], ldh, &nho, &h__[kv + kt * h_dim1],
		    ldh, &nve, &h__[kwv + h_dim1], ldh, &work[1], lwork);
	    //
	    //          ==== Adjust KBOT accounting for new deflations. ====
	    //
	    kbot -= ld;
	    //
	    //          ==== KS points to the shifts. ====
	    //
	    ks = kbot - ls + 1;
	    //
	    //          ==== Skip an expensive QR sweep if there is a (partly
	    //          .    heuristic) reason to expect that many eigenvalues
	    //          .    will deflate without it.  Here, the QR sweep is
	    //          .    skipped if many eigenvalues have just been deflated
	    //          .    or if the remaining active block is small.
	    //
	    if (ld == 0 || ld * 100 <= nw * nibble && kbot - ktop + 1 > min(
		    nmin,nwmax)) {
		//
		//             ==== NS = nominal number of simultaneous shifts.
		//             .    This may be lowered (slightly) if DLAQR3
		//             .    did not provide that many shifts. ====
		//
		// Computing MIN
		// Computing MAX
		i__4 = 2, i__5 = kbot - ktop;
		i__2 = min(nsmax,nsr), i__3 = max(i__4,i__5);
		ns = min(i__2,i__3);
		ns -= ns % 2;
		//
		//             ==== If there have been no deflations
		//             .    in a multiple of KEXSH iterations,
		//             .    then try exceptional shifts.
		//             .    Otherwise use shifts provided by
		//             .    DLAQR3 above or from the eigenvalues
		//             .    of a trailing principal submatrix. ====
		//
		if (ndfl % 6 == 0) {
		    ks = kbot - ns + 1;
		    // Computing MAX
		    i__3 = ks + 1, i__4 = ktop + 2;
		    i__2 = max(i__3,i__4);
		    for (i__ = kbot; i__ >= i__2; i__ += -2) {
			ss = (d__1 = h__[i__ + (i__ - 1) * h_dim1], abs(d__1))
				 + (d__2 = h__[i__ - 1 + (i__ - 2) * h_dim1],
				abs(d__2));
			aa = ss * .75 + h__[i__ + i__ * h_dim1];
			bb = ss;
			cc = ss * -.4375;
			dd = aa;
			dlanv2_(&aa, &bb, &cc, &dd, &wr[i__ - 1], &wi[i__ - 1]
				, &wr[i__], &wi[i__], &cs, &sn);
// L30:
		    }
		    if (ks == ktop) {
			wr[ks + 1] = h__[ks + 1 + (ks + 1) * h_dim1];
			wi[ks + 1] = 0.;
			wr[ks] = wr[ks + 1];
			wi[ks] = wi[ks + 1];
		    }
		} else {
		    //
		    //                ==== Got NS/2 or fewer shifts? Use DLAQR4 or
		    //                .    DLAHQR on a trailing principal submatrix to
		    //                .    get more. (Since NS.LE.NSMAX.LE.(N+6)/9,
		    //                .    there is enough space below the subdiagonal
		    //                .    to fit an NS-by-NS scratch array.) ====
		    //
		    if (kbot - ks + 1 <= ns / 2) {
			ks = kbot - ns + 1;
			kt = *n - ns + 1;
			dlacpy_("A", &ns, &ns, &h__[ks + ks * h_dim1], ldh, &
				h__[kt + h_dim1], ldh);
			if (ns > nmin) {
			    dlaqr4_(&c_false, &c_false, &ns, &c__1, &ns, &h__[
				    kt + h_dim1], ldh, &wr[ks], &wi[ks], &
				    c__1, &c__1, zdum, &c__1, &work[1], lwork,
				     &inf);
			} else {
			    dlahqr_(&c_false, &c_false, &ns, &c__1, &ns, &h__[
				    kt + h_dim1], ldh, &wr[ks], &wi[ks], &
				    c__1, &c__1, zdum, &c__1, &inf);
			}
			ks += inf;
			//
			//                   ==== In case of a rare QR failure use
			//                   .    eigenvalues of the trailing 2-by-2
			//                   .    principal submatrix.  ====
			//
			if (ks >= kbot) {
			    aa = h__[kbot - 1 + (kbot - 1) * h_dim1];
			    cc = h__[kbot + (kbot - 1) * h_dim1];
			    bb = h__[kbot - 1 + kbot * h_dim1];
			    dd = h__[kbot + kbot * h_dim1];
			    dlanv2_(&aa, &bb, &cc, &dd, &wr[kbot - 1], &wi[
				    kbot - 1], &wr[kbot], &wi[kbot], &cs, &sn)
				    ;
			    ks = kbot - 1;
			}
		    }
		    if (kbot - ks + 1 > ns) {
			//
			//                   ==== Sort the shifts (Helps a little)
			//                   .    Bubble sort keeps complex conjugate
			//                   .    pairs together. ====
			//
			sorted = FALSE_;
			i__2 = ks + 1;
			for (k = kbot; k >= i__2; --k) {
			    if (sorted) {
				goto L60;
			    }
			    sorted = TRUE_;
			    i__3 = k - 1;
			    for (i__ = ks; i__ <= i__3; ++i__) {
				if ((d__1 = wr[i__], abs(d__1)) + (d__2 = wi[
					i__], abs(d__2)) < (d__3 = wr[i__ + 1]
					, abs(d__3)) + (d__4 = wi[i__ + 1],
					abs(d__4))) {
				    sorted = FALSE_;
				    swap = wr[i__];
				    wr[i__] = wr[i__ + 1];
				    wr[i__ + 1] = swap;
				    swap = wi[i__];
				    wi[i__] = wi[i__ + 1];
				    wi[i__ + 1] = swap;
				}
// L40:
			    }
// L50:
			}
L60:
			;
		    }
		    //
		    //                ==== Shuffle shifts into pairs of real shifts
		    //                .    and pairs of complex conjugate shifts
		    //                .    assuming complex conjugate shifts are
		    //                .    already adjacent to one another. (Yes,
		    //                .    they are.)  ====
		    //
		    i__2 = ks + 2;
		    for (i__ = kbot; i__ >= i__2; i__ += -2) {
			if (wi[i__] != -wi[i__ - 1]) {
			    swap = wr[i__];
			    wr[i__] = wr[i__ - 1];
			    wr[i__ - 1] = wr[i__ - 2];
			    wr[i__ - 2] = swap;
			    swap = wi[i__];
			    wi[i__] = wi[i__ - 1];
			    wi[i__ - 1] = wi[i__ - 2];
			    wi[i__ - 2] = swap;
			}
// L70:
		    }
		}
		//
		//             ==== If there are only two shifts and both are
		//             .    real, then use only one.  ====
		//
		if (kbot - ks + 1 == 2) {
		    if (wi[kbot] == 0.) {
			if ((d__1 = wr[kbot] - h__[kbot + kbot * h_dim1], abs(
				d__1)) < (d__2 = wr[kbot - 1] - h__[kbot +
				kbot * h_dim1], abs(d__2))) {
			    wr[kbot - 1] = wr[kbot];
			} else {
			    wr[kbot] = wr[kbot - 1];
			}
		    }
		}
		//
		//             ==== Use up to NS of the the smallest magnitude
		//             .    shifts.  If there aren't NS shifts available,
		//             .    then use them all, possibly dropping one to
		//             .    make the number of shifts even. ====
		//
		// Computing MIN
		i__2 = ns, i__3 = kbot - ks + 1;
		ns = min(i__2,i__3);
		ns -= ns % 2;
		ks = kbot - ns + 1;
		//
		//             ==== Small-bulge multi-shift QR sweep:
		//             .    split workspace under the subdiagonal into
		//             .    - a KDU-by-KDU work array U in the lower
		//             .      left-hand-corner,
		//             .    - a KDU-by-at-least-KDU-but-more-is-better
		//             .      (KDU-by-NHo) horizontal work array WH along
		//             .      the bottom edge,
		//             .    - and an at-least-KDU-but-more-is-better-by-KDU
		//             .      (NVE-by-KDU) vertical work WV arrow along
		//             .      the left-hand-edge. ====
		//
		kdu = ns * 3 - 3;
		ku = *n - kdu + 1;
		kwh = kdu + 1;
		nho = *n - kdu - 3 - (kdu + 1) + 1;
		kwv = kdu + 4;
		nve = *n - kdu - kwv + 1;
		//
		//             ==== Small-bulge multi-shift QR sweep ====
		//
		dlaqr5_(wantt, wantz, &kacc22, n, &ktop, &kbot, &ns, &wr[ks],
			&wi[ks], &h__[h_offset], ldh, iloz, ihiz, &z__[
			z_offset], ldz, &work[1], &c__3, &h__[ku + h_dim1],
			ldh, &nve, &h__[kwv + h_dim1], ldh, &nho, &h__[ku +
			kwh * h_dim1], ldh);
	    }
	    //
	    //          ==== Note progress (or the lack of it). ====
	    //
	    if (ld > 0) {
		ndfl = 1;
	    } else {
		++ndfl;
	    }
	    //
	    //          ==== End of main loop ====
// L80:
	}
	//
	//       ==== Iteration limit exceeded.  Set INFO to show where
	//       .    the problem occurred and exit. ====
	//
	*info = kbot;
L90:
	;
    }
    //
    //    ==== Return the optimal value of LWORK. ====
    //
    work[1] = (double) lwkopt;
    //
    //    ==== End of DLAQR0 ====
    //
    return 0;
} // dlaqr0_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAQR1 sets a scalar multiple of the first column of the product of 2-by-2 or 3-by-3 matrix H and specified shifts.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAQR1 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaqr1.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaqr1.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaqr1.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAQR1( N, H, LDH, SR1, SI1, SR2, SI2, V )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   SI1, SI2, SR1, SR2
//      INTEGER            LDH, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), V( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>      Given a 2-by-2 or 3-by-3 matrix H, DLAQR1 sets v to a
//>      scalar multiple of the first column of the product
//>
//>      (*)  K = (H - (sr1 + i*si1)*I)*(H - (sr2 + i*si2)*I)
//>
//>      scaling to avoid overflows and most underflows. It
//>      is assumed that either
//>
//>              1) sr1 = sr2 and si1 = -si2
//>          or
//>              2) si1 = si2 = 0.
//>
//>      This is useful for starting double implicit shift bulges
//>      in the QR algorithm.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>              Order of the matrix H. N must be either 2 or 3.
//> \endverbatim
//>
//> \param[in] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>              The 2-by-2 or 3-by-3 matrix H in (*).
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>              The leading dimension of H as declared in
//>              the calling procedure.  LDH >= N
//> \endverbatim
//>
//> \param[in] SR1
//> \verbatim
//>          SR1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] SI1
//> \verbatim
//>          SI1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] SR2
//> \verbatim
//>          SR2 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] SI2
//> \verbatim
//>          SI2 is DOUBLE PRECISION
//>              The shifts in (*).
//> \endverbatim
//>
//> \param[out] V
//> \verbatim
//>          V is DOUBLE PRECISION array, dimension (N)
//>              A scalar multiple of the first column of the
//>              matrix K in (*).
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
//> \ingroup doubleOTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>       Karen Braman and Ralph Byers, Department of Mathematics,
//>       University of Kansas, USA
//>
// =====================================================================
/* Subroutine */ int dlaqr1_(int *n, double *h__, int *ldh, double *sr1,
	double *si1, double *sr2, double *si2, double *v)
{
    // System generated locals
    int h_dim1, h_offset;
    double d__1, d__2, d__3;

    // Local variables
    double s, h21s, h31s;

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
    // ================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Quick return if possible
    //
    // Parameter adjustments
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --v;

    // Function Body
    if (*n != 2 && *n != 3) {
	return 0;
    }
    if (*n == 2) {
	s = (d__1 = h__[h_dim1 + 1] - *sr2, abs(d__1)) + abs(*si2) + (d__2 =
		h__[h_dim1 + 2], abs(d__2));
	if (s == 0.) {
	    v[1] = 0.;
	    v[2] = 0.;
	} else {
	    h21s = h__[h_dim1 + 2] / s;
	    v[1] = h21s * h__[(h_dim1 << 1) + 1] + (h__[h_dim1 + 1] - *sr1) *
		    ((h__[h_dim1 + 1] - *sr2) / s) - *si1 * (*si2 / s);
	    v[2] = h21s * (h__[h_dim1 + 1] + h__[(h_dim1 << 1) + 2] - *sr1 - *
		    sr2);
	}
    } else {
	s = (d__1 = h__[h_dim1 + 1] - *sr2, abs(d__1)) + abs(*si2) + (d__2 =
		h__[h_dim1 + 2], abs(d__2)) + (d__3 = h__[h_dim1 + 3], abs(
		d__3));
	if (s == 0.) {
	    v[1] = 0.;
	    v[2] = 0.;
	    v[3] = 0.;
	} else {
	    h21s = h__[h_dim1 + 2] / s;
	    h31s = h__[h_dim1 + 3] / s;
	    v[1] = (h__[h_dim1 + 1] - *sr1) * ((h__[h_dim1 + 1] - *sr2) / s)
		    - *si1 * (*si2 / s) + h__[(h_dim1 << 1) + 1] * h21s + h__[
		    h_dim1 * 3 + 1] * h31s;
	    v[2] = h21s * (h__[h_dim1 + 1] + h__[(h_dim1 << 1) + 2] - *sr1 - *
		    sr2) + h__[h_dim1 * 3 + 2] * h31s;
	    v[3] = h31s * (h__[h_dim1 + 1] + h__[h_dim1 * 3 + 3] - *sr1 - *
		    sr2) + h21s * h__[(h_dim1 << 1) + 3];
	}
    }
    return 0;
} // dlaqr1_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAQR2 performs the orthogonal similarity transformation of a Hessenberg matrix to detect and deflate fully converged eigenvalues from a trailing principal submatrix (aggressive early deflation).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAQR2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaqr2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaqr2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaqr2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAQR2( WANTT, WANTZ, N, KTOP, KBOT, NW, H, LDH, ILOZ,
//                         IHIZ, Z, LDZ, NS, ND, SR, SI, V, LDV, NH, T,
//                         LDT, NV, WV, LDWV, WORK, LWORK )
//
//      .. Scalar Arguments ..
//      INTEGER            IHIZ, ILOZ, KBOT, KTOP, LDH, LDT, LDV, LDWV,
//     $                   LDZ, LWORK, N, ND, NH, NS, NV, NW
//      LOGICAL            WANTT, WANTZ
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), SI( * ), SR( * ), T( LDT, * ),
//     $                   V( LDV, * ), WORK( * ), WV( LDWV, * ),
//     $                   Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DLAQR2 is identical to DLAQR3 except that it avoids
//>    recursion by calling DLAHQR instead of DLAQR4.
//>
//>    Aggressive early deflation:
//>
//>    This subroutine accepts as input an upper Hessenberg matrix
//>    H and performs an orthogonal similarity transformation
//>    designed to detect and deflate fully converged eigenvalues from
//>    a trailing principal submatrix.  On output H has been over-
//>    written by a new Hessenberg matrix that is a perturbation of
//>    an orthogonal similarity transformation of H.  It is to be
//>    hoped that the final version of H has many zero subdiagonal
//>    entries.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] WANTT
//> \verbatim
//>          WANTT is LOGICAL
//>          If .TRUE., then the Hessenberg matrix H is fully updated
//>          so that the quasi-triangular Schur factor may be
//>          computed (in cooperation with the calling subroutine).
//>          If .FALSE., then only enough of H is updated to preserve
//>          the eigenvalues.
//> \endverbatim
//>
//> \param[in] WANTZ
//> \verbatim
//>          WANTZ is LOGICAL
//>          If .TRUE., then the orthogonal matrix Z is updated so
//>          so that the orthogonal Schur factor may be computed
//>          (in cooperation with the calling subroutine).
//>          If .FALSE., then Z is not referenced.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix H and (if WANTZ is .TRUE.) the
//>          order of the orthogonal matrix Z.
//> \endverbatim
//>
//> \param[in] KTOP
//> \verbatim
//>          KTOP is INTEGER
//>          It is assumed that either KTOP = 1 or H(KTOP,KTOP-1)=0.
//>          KBOT and KTOP together determine an isolated block
//>          along the diagonal of the Hessenberg matrix.
//> \endverbatim
//>
//> \param[in] KBOT
//> \verbatim
//>          KBOT is INTEGER
//>          It is assumed without a check that either
//>          KBOT = N or H(KBOT+1,KBOT)=0.  KBOT and KTOP together
//>          determine an isolated block along the diagonal of the
//>          Hessenberg matrix.
//> \endverbatim
//>
//> \param[in] NW
//> \verbatim
//>          NW is INTEGER
//>          Deflation window size.  1 <= NW <= (KBOT-KTOP+1).
//> \endverbatim
//>
//> \param[in,out] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>          On input the initial N-by-N section of H stores the
//>          Hessenberg matrix undergoing aggressive early deflation.
//>          On output H has been transformed by an orthogonal
//>          similarity transformation, perturbed, and the returned
//>          to Hessenberg form that (it is to be hoped) has some
//>          zero subdiagonal entries.
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>          Leading dimension of H just as declared in the calling
//>          subroutine.  N <= LDH
//> \endverbatim
//>
//> \param[in] ILOZ
//> \verbatim
//>          ILOZ is INTEGER
//> \endverbatim
//>
//> \param[in] IHIZ
//> \verbatim
//>          IHIZ is INTEGER
//>          Specify the rows of Z to which transformations must be
//>          applied if WANTZ is .TRUE.. 1 <= ILOZ <= IHIZ <= N.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ,N)
//>          IF WANTZ is .TRUE., then on output, the orthogonal
//>          similarity transformation mentioned above has been
//>          accumulated into Z(ILOZ:IHIZ,ILOZ:IHIZ) from the right.
//>          If WANTZ is .FALSE., then Z is unreferenced.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>          The leading dimension of Z just as declared in the
//>          calling subroutine.  1 <= LDZ.
//> \endverbatim
//>
//> \param[out] NS
//> \verbatim
//>          NS is INTEGER
//>          The number of unconverged (ie approximate) eigenvalues
//>          returned in SR and SI that may be used as shifts by the
//>          calling subroutine.
//> \endverbatim
//>
//> \param[out] ND
//> \verbatim
//>          ND is INTEGER
//>          The number of converged eigenvalues uncovered by this
//>          subroutine.
//> \endverbatim
//>
//> \param[out] SR
//> \verbatim
//>          SR is DOUBLE PRECISION array, dimension (KBOT)
//> \endverbatim
//>
//> \param[out] SI
//> \verbatim
//>          SI is DOUBLE PRECISION array, dimension (KBOT)
//>          On output, the real and imaginary parts of approximate
//>          eigenvalues that may be used for shifts are stored in
//>          SR(KBOT-ND-NS+1) through SR(KBOT-ND) and
//>          SI(KBOT-ND-NS+1) through SI(KBOT-ND), respectively.
//>          The real and imaginary parts of converged eigenvalues
//>          are stored in SR(KBOT-ND+1) through SR(KBOT) and
//>          SI(KBOT-ND+1) through SI(KBOT), respectively.
//> \endverbatim
//>
//> \param[out] V
//> \verbatim
//>          V is DOUBLE PRECISION array, dimension (LDV,NW)
//>          An NW-by-NW work array.
//> \endverbatim
//>
//> \param[in] LDV
//> \verbatim
//>          LDV is INTEGER
//>          The leading dimension of V just as declared in the
//>          calling subroutine.  NW <= LDV
//> \endverbatim
//>
//> \param[in] NH
//> \verbatim
//>          NH is INTEGER
//>          The number of columns of T.  NH >= NW.
//> \endverbatim
//>
//> \param[out] T
//> \verbatim
//>          T is DOUBLE PRECISION array, dimension (LDT,NW)
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of T just as declared in the
//>          calling subroutine.  NW <= LDT
//> \endverbatim
//>
//> \param[in] NV
//> \verbatim
//>          NV is INTEGER
//>          The number of rows of work array WV available for
//>          workspace.  NV >= NW.
//> \endverbatim
//>
//> \param[out] WV
//> \verbatim
//>          WV is DOUBLE PRECISION array, dimension (LDWV,NW)
//> \endverbatim
//>
//> \param[in] LDWV
//> \verbatim
//>          LDWV is INTEGER
//>          The leading dimension of W just as declared in the
//>          calling subroutine.  NW <= LDV
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (LWORK)
//>          On exit, WORK(1) is set to an estimate of the optimal value
//>          of LWORK for the given values of N, NW, KTOP and KBOT.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the work array WORK.  LWORK = 2*NW
//>          suffices, but greater efficiency may result from larger
//>          values of LWORK.
//>
//>          If LWORK = -1, then a workspace query is assumed; DLAQR2
//>          only estimates the optimal workspace size for the given
//>          values of N, NW, KTOP and KBOT.  The estimate is returned
//>          in WORK(1).  No error message related to LWORK is issued
//>          by XERBLA.  Neither H nor Z are accessed.
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
//> \ingroup doubleOTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>       Karen Braman and Ralph Byers, Department of Mathematics,
//>       University of Kansas, USA
//>
// =====================================================================
/* Subroutine */ int dlaqr2_(int *wantt, int *wantz, int *n, int *ktop, int *
	kbot, int *nw, double *h__, int *ldh, int *iloz, int *ihiz, double *
	z__, int *ldz, int *ns, int *nd, double *sr, double *si, double *v,
	int *ldv, int *nh, double *t, int *ldt, int *nv, double *wv, int *
	ldwv, double *work, int *lwork)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    double c_b12 = 0.;
    double c_b13 = 1.;
    int c_true = TRUE_;

    // System generated locals
    int h_dim1, h_offset, t_dim1, t_offset, v_dim1, v_offset, wv_dim1,
	    wv_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4;
    double d__1, d__2, d__3, d__4, d__5, d__6;

    // Local variables
    int i__, j, k;
    double s, aa, bb, cc, dd, cs, sn;
    int jw;
    double evi, evk, foo;
    int kln;
    double tau, ulp;
    int lwk1, lwk2;
    double beta;
    int kend, kcol, info, ifst, ilst, ltop, krow;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *), dgemm_(char *, char *, int *
	    , int *, int *, double *, double *, int *, double *, int *,
	    double *, double *, int *);
    int bulge;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int infqr, kwtop;
    extern /* Subroutine */ int dlanv2_(double *, double *, double *, double *
	    , double *, double *, double *, double *, double *, double *),
	    dlabad_(double *, double *);
    extern double dlamch_(char *);
    extern /* Subroutine */ int dgehrd_(int *, int *, int *, double *, int *,
	    double *, double *, int *, int *), dlarfg_(int *, double *,
	    double *, int *, double *), dlahqr_(int *, int *, int *, int *,
	    int *, double *, int *, double *, double *, int *, int *, double *
	    , int *, int *), dlacpy_(char *, int *, int *, double *, int *,
	    double *, int *);
    double safmin;
    extern /* Subroutine */ int dlaset_(char *, int *, int *, double *,
	    double *, double *, int *);
    double safmax;
    extern /* Subroutine */ int dtrexc_(char *, int *, double *, int *,
	    double *, int *, int *, int *, double *, int *), dormhr_(char *,
	    char *, int *, int *, int *, int *, double *, int *, double *,
	    double *, int *, double *, int *, int *);
    int sorted;
    double smlnum;
    int lwkopt;

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
    // ================================================================
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
    //    ==== Estimate optimal workspace. ====
    //
    // Parameter adjustments
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --sr;
    --si;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    --work;

    // Function Body
    // Computing MIN
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = min(i__1,i__2);
    if (jw <= 2) {
	lwkopt = 1;
    } else {
	//
	//       ==== Workspace query call to DGEHRD ====
	//
	i__1 = jw - 1;
	dgehrd_(&jw, &c__1, &i__1, &t[t_offset], ldt, &work[1], &work[1], &
		c_n1, &info);
	lwk1 = (int) work[1];
	//
	//       ==== Workspace query call to DORMHR ====
	//
	i__1 = jw - 1;
	dormhr_("R", "N", &jw, &jw, &c__1, &i__1, &t[t_offset], ldt, &work[1],
		 &v[v_offset], ldv, &work[1], &c_n1, &info);
	lwk2 = (int) work[1];
	//
	//       ==== Optimal workspace ====
	//
	lwkopt = jw + max(lwk1,lwk2);
    }
    //
    //    ==== Quick return in case of workspace query. ====
    //
    if (*lwork == -1) {
	work[1] = (double) lwkopt;
	return 0;
    }
    //
    //    ==== Nothing to do ...
    //    ... for an empty active block ... ====
    *ns = 0;
    *nd = 0;
    work[1] = 1.;
    if (*ktop > *kbot) {
	return 0;
    }
    //    ... nor for an empty deflation window. ====
    if (*nw < 1) {
	return 0;
    }
    //
    //    ==== Machine constants ====
    //
    safmin = dlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_("PRECISION");
    smlnum = safmin * ((double) (*n) / ulp);
    //
    //    ==== Setup deflation window ====
    //
    // Computing MIN
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = min(i__1,i__2);
    kwtop = *kbot - jw + 1;
    if (kwtop == *ktop) {
	s = 0.;
    } else {
	s = h__[kwtop + (kwtop - 1) * h_dim1];
    }
    if (*kbot == kwtop) {
	//
	//       ==== 1-by-1 deflation window: not much to do ====
	//
	sr[kwtop] = h__[kwtop + kwtop * h_dim1];
	si[kwtop] = 0.;
	*ns = 1;
	*nd = 0;
	// Computing MAX
	d__2 = smlnum, d__3 = ulp * (d__1 = h__[kwtop + kwtop * h_dim1], abs(
		d__1));
	if (abs(s) <= max(d__2,d__3)) {
	    *ns = 0;
	    *nd = 1;
	    if (kwtop > *ktop) {
		h__[kwtop + (kwtop - 1) * h_dim1] = 0.;
	    }
	}
	work[1] = 1.;
	return 0;
    }
    //
    //    ==== Convert to spike-triangular form.  (In case of a
    //    .    rare QR failure, this routine continues to do
    //    .    aggressive early deflation using that part of
    //    .    the deflation window that converged using INFQR
    //    .    here and there to keep track.) ====
    //
    dlacpy_("U", &jw, &jw, &h__[kwtop + kwtop * h_dim1], ldh, &t[t_offset],
	    ldt);
    i__1 = jw - 1;
    i__2 = *ldh + 1;
    i__3 = *ldt + 1;
    dcopy_(&i__1, &h__[kwtop + 1 + kwtop * h_dim1], &i__2, &t[t_dim1 + 2], &
	    i__3);
    dlaset_("A", &jw, &jw, &c_b12, &c_b13, &v[v_offset], ldv);
    dlahqr_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sr[kwtop],
	    &si[kwtop], &c__1, &jw, &v[v_offset], ldv, &infqr);
    //
    //    ==== DTREXC needs a clean margin near the diagonal ====
    //
    i__1 = jw - 3;
    for (j = 1; j <= i__1; ++j) {
	t[j + 2 + j * t_dim1] = 0.;
	t[j + 3 + j * t_dim1] = 0.;
// L10:
    }
    if (jw > 2) {
	t[jw + (jw - 2) * t_dim1] = 0.;
    }
    //
    //    ==== Deflation detection loop ====
    //
    *ns = jw;
    ilst = infqr + 1;
L20:
    if (ilst <= *ns) {
	if (*ns == 1) {
	    bulge = FALSE_;
	} else {
	    bulge = t[*ns + (*ns - 1) * t_dim1] != 0.;
	}
	//
	//       ==== Small spike tip test for deflation ====
	//
	if (! bulge) {
	    //
	    //          ==== Real eigenvalue ====
	    //
	    foo = (d__1 = t[*ns + *ns * t_dim1], abs(d__1));
	    if (foo == 0.) {
		foo = abs(s);
	    }
	    // Computing MAX
	    d__2 = smlnum, d__3 = ulp * foo;
	    if ((d__1 = s * v[*ns * v_dim1 + 1], abs(d__1)) <= max(d__2,d__3))
		     {
		//
		//             ==== Deflatable ====
		//
		--(*ns);
	    } else {
		//
		//             ==== Undeflatable.   Move it up out of the way.
		//             .    (DTREXC can not fail in this case.) ====
		//
		ifst = *ns;
		dtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst,
			 &ilst, &work[1], &info);
		++ilst;
	    }
	} else {
	    //
	    //          ==== Complex conjugate pair ====
	    //
	    foo = (d__3 = t[*ns + *ns * t_dim1], abs(d__3)) + sqrt((d__1 = t[*
		    ns + (*ns - 1) * t_dim1], abs(d__1))) * sqrt((d__2 = t[*
		    ns - 1 + *ns * t_dim1], abs(d__2)));
	    if (foo == 0.) {
		foo = abs(s);
	    }
	    // Computing MAX
	    d__3 = (d__1 = s * v[*ns * v_dim1 + 1], abs(d__1)), d__4 = (d__2 =
		     s * v[(*ns - 1) * v_dim1 + 1], abs(d__2));
	    // Computing MAX
	    d__5 = smlnum, d__6 = ulp * foo;
	    if (max(d__3,d__4) <= max(d__5,d__6)) {
		//
		//             ==== Deflatable ====
		//
		*ns += -2;
	    } else {
		//
		//             ==== Undeflatable. Move them up out of the way.
		//             .    Fortunately, DTREXC does the right thing with
		//             .    ILST in case of a rare exchange failure. ====
		//
		ifst = *ns;
		dtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst,
			 &ilst, &work[1], &info);
		ilst += 2;
	    }
	}
	//
	//       ==== End deflation detection loop ====
	//
	goto L20;
    }
    //
    //       ==== Return to Hessenberg form ====
    //
    if (*ns == 0) {
	s = 0.;
    }
    if (*ns < jw) {
	//
	//       ==== sorting diagonal blocks of T improves accuracy for
	//       .    graded matrices.  Bubble sort deals well with
	//       .    exchange failures. ====
	//
	sorted = FALSE_;
	i__ = *ns + 1;
L30:
	if (sorted) {
	    goto L50;
	}
	sorted = TRUE_;
	kend = i__ - 1;
	i__ = infqr + 1;
	if (i__ == *ns) {
	    k = i__ + 1;
	} else if (t[i__ + 1 + i__ * t_dim1] == 0.) {
	    k = i__ + 1;
	} else {
	    k = i__ + 2;
	}
L40:
	if (k <= kend) {
	    if (k == i__ + 1) {
		evi = (d__1 = t[i__ + i__ * t_dim1], abs(d__1));
	    } else {
		evi = (d__3 = t[i__ + i__ * t_dim1], abs(d__3)) + sqrt((d__1 =
			 t[i__ + 1 + i__ * t_dim1], abs(d__1))) * sqrt((d__2 =
			 t[i__ + (i__ + 1) * t_dim1], abs(d__2)));
	    }
	    if (k == kend) {
		evk = (d__1 = t[k + k * t_dim1], abs(d__1));
	    } else if (t[k + 1 + k * t_dim1] == 0.) {
		evk = (d__1 = t[k + k * t_dim1], abs(d__1));
	    } else {
		evk = (d__3 = t[k + k * t_dim1], abs(d__3)) + sqrt((d__1 = t[
			k + 1 + k * t_dim1], abs(d__1))) * sqrt((d__2 = t[k +
			(k + 1) * t_dim1], abs(d__2)));
	    }
	    if (evi >= evk) {
		i__ = k;
	    } else {
		sorted = FALSE_;
		ifst = i__;
		ilst = k;
		dtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst,
			 &ilst, &work[1], &info);
		if (info == 0) {
		    i__ = ilst;
		} else {
		    i__ = k;
		}
	    }
	    if (i__ == kend) {
		k = i__ + 1;
	    } else if (t[i__ + 1 + i__ * t_dim1] == 0.) {
		k = i__ + 1;
	    } else {
		k = i__ + 2;
	    }
	    goto L40;
	}
	goto L30;
L50:
	;
    }
    //
    //    ==== Restore shift/eigenvalue array from T ====
    //
    i__ = jw;
L60:
    if (i__ >= infqr + 1) {
	if (i__ == infqr + 1) {
	    sr[kwtop + i__ - 1] = t[i__ + i__ * t_dim1];
	    si[kwtop + i__ - 1] = 0.;
	    --i__;
	} else if (t[i__ + (i__ - 1) * t_dim1] == 0.) {
	    sr[kwtop + i__ - 1] = t[i__ + i__ * t_dim1];
	    si[kwtop + i__ - 1] = 0.;
	    --i__;
	} else {
	    aa = t[i__ - 1 + (i__ - 1) * t_dim1];
	    cc = t[i__ + (i__ - 1) * t_dim1];
	    bb = t[i__ - 1 + i__ * t_dim1];
	    dd = t[i__ + i__ * t_dim1];
	    dlanv2_(&aa, &bb, &cc, &dd, &sr[kwtop + i__ - 2], &si[kwtop + i__
		    - 2], &sr[kwtop + i__ - 1], &si[kwtop + i__ - 1], &cs, &
		    sn);
	    i__ += -2;
	}
	goto L60;
    }
    if (*ns < jw || s == 0.) {
	if (*ns > 1 && s != 0.) {
	    //
	    //          ==== Reflect spike back into lower triangle ====
	    //
	    dcopy_(ns, &v[v_offset], ldv, &work[1], &c__1);
	    beta = work[1];
	    dlarfg_(ns, &beta, &work[2], &c__1, &tau);
	    work[1] = 1.;
	    i__1 = jw - 2;
	    i__2 = jw - 2;
	    dlaset_("L", &i__1, &i__2, &c_b12, &c_b12, &t[t_dim1 + 3], ldt);
	    dlarf_("L", ns, &jw, &work[1], &c__1, &tau, &t[t_offset], ldt, &
		    work[jw + 1]);
	    dlarf_("R", ns, ns, &work[1], &c__1, &tau, &t[t_offset], ldt, &
		    work[jw + 1]);
	    dlarf_("R", &jw, ns, &work[1], &c__1, &tau, &v[v_offset], ldv, &
		    work[jw + 1]);
	    i__1 = *lwork - jw;
	    dgehrd_(&jw, &c__1, ns, &t[t_offset], ldt, &work[1], &work[jw + 1]
		    , &i__1, &info);
	}
	//
	//       ==== Copy updated reduced window into place ====
	//
	if (kwtop > 1) {
	    h__[kwtop + (kwtop - 1) * h_dim1] = s * v[v_dim1 + 1];
	}
	dlacpy_("U", &jw, &jw, &t[t_offset], ldt, &h__[kwtop + kwtop * h_dim1]
		, ldh);
	i__1 = jw - 1;
	i__2 = *ldt + 1;
	i__3 = *ldh + 1;
	dcopy_(&i__1, &t[t_dim1 + 2], &i__2, &h__[kwtop + 1 + kwtop * h_dim1],
		 &i__3);
	//
	//       ==== Accumulate orthogonal matrix in order update
	//       .    H and Z, if requested.  ====
	//
	if (*ns > 1 && s != 0.) {
	    i__1 = *lwork - jw;
	    dormhr_("R", "N", &jw, ns, &c__1, ns, &t[t_offset], ldt, &work[1],
		     &v[v_offset], ldv, &work[jw + 1], &i__1, &info);
	}
	//
	//       ==== Update vertical slab in H ====
	//
	if (*wantt) {
	    ltop = 1;
	} else {
	    ltop = *ktop;
	}
	i__1 = kwtop - 1;
	i__2 = *nv;
	for (krow = ltop; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow +=
		i__2) {
	    // Computing MIN
	    i__3 = *nv, i__4 = kwtop - krow;
	    kln = min(i__3,i__4);
	    dgemm_("N", "N", &kln, &jw, &jw, &c_b13, &h__[krow + kwtop *
		    h_dim1], ldh, &v[v_offset], ldv, &c_b12, &wv[wv_offset],
		    ldwv);
	    dlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &h__[krow + kwtop *
		    h_dim1], ldh);
// L70:
	}
	//
	//       ==== Update horizontal slab in H ====
	//
	if (*wantt) {
	    i__2 = *n;
	    i__1 = *nh;
	    for (kcol = *kbot + 1; i__1 < 0 ? kcol >= i__2 : kcol <= i__2;
		    kcol += i__1) {
		// Computing MIN
		i__3 = *nh, i__4 = *n - kcol + 1;
		kln = min(i__3,i__4);
		dgemm_("C", "N", &jw, &kln, &jw, &c_b13, &v[v_offset], ldv, &
			h__[kwtop + kcol * h_dim1], ldh, &c_b12, &t[t_offset],
			 ldt);
		dlacpy_("A", &jw, &kln, &t[t_offset], ldt, &h__[kwtop + kcol *
			 h_dim1], ldh);
// L80:
	    }
	}
	//
	//       ==== Update vertical slab in Z ====
	//
	if (*wantz) {
	    i__1 = *ihiz;
	    i__2 = *nv;
	    for (krow = *iloz; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow +=
		     i__2) {
		// Computing MIN
		i__3 = *nv, i__4 = *ihiz - krow + 1;
		kln = min(i__3,i__4);
		dgemm_("N", "N", &kln, &jw, &jw, &c_b13, &z__[krow + kwtop *
			z_dim1], ldz, &v[v_offset], ldv, &c_b12, &wv[
			wv_offset], ldwv);
		dlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &z__[krow +
			kwtop * z_dim1], ldz);
// L90:
	    }
	}
    }
    //
    //    ==== Return the number of deflations ... ====
    //
    *nd = jw - *ns;
    //
    //    ==== ... and the number of shifts. (Subtracting
    //    .    INFQR from the spike length takes care
    //    .    of the case of a rare QR failure while
    //    .    calculating eigenvalues of the deflation
    //    .    window.)  ====
    //
    *ns -= infqr;
    //
    //     ==== Return optimal workspace. ====
    //
    work[1] = (double) lwkopt;
    //
    //    ==== End of DLAQR2 ====
    //
    return 0;
} // dlaqr2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAQR3 performs the orthogonal similarity transformation of a Hessenberg matrix to detect and deflate fully converged eigenvalues from a trailing principal submatrix (aggressive early deflation).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAQR3 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaqr3.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaqr3.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaqr3.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAQR3( WANTT, WANTZ, N, KTOP, KBOT, NW, H, LDH, ILOZ,
//                         IHIZ, Z, LDZ, NS, ND, SR, SI, V, LDV, NH, T,
//                         LDT, NV, WV, LDWV, WORK, LWORK )
//
//      .. Scalar Arguments ..
//      INTEGER            IHIZ, ILOZ, KBOT, KTOP, LDH, LDT, LDV, LDWV,
//     $                   LDZ, LWORK, N, ND, NH, NS, NV, NW
//      LOGICAL            WANTT, WANTZ
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), SI( * ), SR( * ), T( LDT, * ),
//     $                   V( LDV, * ), WORK( * ), WV( LDWV, * ),
//     $                   Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    Aggressive early deflation:
//>
//>    DLAQR3 accepts as input an upper Hessenberg matrix
//>    H and performs an orthogonal similarity transformation
//>    designed to detect and deflate fully converged eigenvalues from
//>    a trailing principal submatrix.  On output H has been over-
//>    written by a new Hessenberg matrix that is a perturbation of
//>    an orthogonal similarity transformation of H.  It is to be
//>    hoped that the final version of H has many zero subdiagonal
//>    entries.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] WANTT
//> \verbatim
//>          WANTT is LOGICAL
//>          If .TRUE., then the Hessenberg matrix H is fully updated
//>          so that the quasi-triangular Schur factor may be
//>          computed (in cooperation with the calling subroutine).
//>          If .FALSE., then only enough of H is updated to preserve
//>          the eigenvalues.
//> \endverbatim
//>
//> \param[in] WANTZ
//> \verbatim
//>          WANTZ is LOGICAL
//>          If .TRUE., then the orthogonal matrix Z is updated so
//>          so that the orthogonal Schur factor may be computed
//>          (in cooperation with the calling subroutine).
//>          If .FALSE., then Z is not referenced.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix H and (if WANTZ is .TRUE.) the
//>          order of the orthogonal matrix Z.
//> \endverbatim
//>
//> \param[in] KTOP
//> \verbatim
//>          KTOP is INTEGER
//>          It is assumed that either KTOP = 1 or H(KTOP,KTOP-1)=0.
//>          KBOT and KTOP together determine an isolated block
//>          along the diagonal of the Hessenberg matrix.
//> \endverbatim
//>
//> \param[in] KBOT
//> \verbatim
//>          KBOT is INTEGER
//>          It is assumed without a check that either
//>          KBOT = N or H(KBOT+1,KBOT)=0.  KBOT and KTOP together
//>          determine an isolated block along the diagonal of the
//>          Hessenberg matrix.
//> \endverbatim
//>
//> \param[in] NW
//> \verbatim
//>          NW is INTEGER
//>          Deflation window size.  1 <= NW <= (KBOT-KTOP+1).
//> \endverbatim
//>
//> \param[in,out] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>          On input the initial N-by-N section of H stores the
//>          Hessenberg matrix undergoing aggressive early deflation.
//>          On output H has been transformed by an orthogonal
//>          similarity transformation, perturbed, and the returned
//>          to Hessenberg form that (it is to be hoped) has some
//>          zero subdiagonal entries.
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>          Leading dimension of H just as declared in the calling
//>          subroutine.  N <= LDH
//> \endverbatim
//>
//> \param[in] ILOZ
//> \verbatim
//>          ILOZ is INTEGER
//> \endverbatim
//>
//> \param[in] IHIZ
//> \verbatim
//>          IHIZ is INTEGER
//>          Specify the rows of Z to which transformations must be
//>          applied if WANTZ is .TRUE.. 1 <= ILOZ <= IHIZ <= N.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ,N)
//>          IF WANTZ is .TRUE., then on output, the orthogonal
//>          similarity transformation mentioned above has been
//>          accumulated into Z(ILOZ:IHIZ,ILOZ:IHIZ) from the right.
//>          If WANTZ is .FALSE., then Z is unreferenced.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>          The leading dimension of Z just as declared in the
//>          calling subroutine.  1 <= LDZ.
//> \endverbatim
//>
//> \param[out] NS
//> \verbatim
//>          NS is INTEGER
//>          The number of unconverged (ie approximate) eigenvalues
//>          returned in SR and SI that may be used as shifts by the
//>          calling subroutine.
//> \endverbatim
//>
//> \param[out] ND
//> \verbatim
//>          ND is INTEGER
//>          The number of converged eigenvalues uncovered by this
//>          subroutine.
//> \endverbatim
//>
//> \param[out] SR
//> \verbatim
//>          SR is DOUBLE PRECISION array, dimension (KBOT)
//> \endverbatim
//>
//> \param[out] SI
//> \verbatim
//>          SI is DOUBLE PRECISION array, dimension (KBOT)
//>          On output, the real and imaginary parts of approximate
//>          eigenvalues that may be used for shifts are stored in
//>          SR(KBOT-ND-NS+1) through SR(KBOT-ND) and
//>          SI(KBOT-ND-NS+1) through SI(KBOT-ND), respectively.
//>          The real and imaginary parts of converged eigenvalues
//>          are stored in SR(KBOT-ND+1) through SR(KBOT) and
//>          SI(KBOT-ND+1) through SI(KBOT), respectively.
//> \endverbatim
//>
//> \param[out] V
//> \verbatim
//>          V is DOUBLE PRECISION array, dimension (LDV,NW)
//>          An NW-by-NW work array.
//> \endverbatim
//>
//> \param[in] LDV
//> \verbatim
//>          LDV is INTEGER
//>          The leading dimension of V just as declared in the
//>          calling subroutine.  NW <= LDV
//> \endverbatim
//>
//> \param[in] NH
//> \verbatim
//>          NH is INTEGER
//>          The number of columns of T.  NH >= NW.
//> \endverbatim
//>
//> \param[out] T
//> \verbatim
//>          T is DOUBLE PRECISION array, dimension (LDT,NW)
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of T just as declared in the
//>          calling subroutine.  NW <= LDT
//> \endverbatim
//>
//> \param[in] NV
//> \verbatim
//>          NV is INTEGER
//>          The number of rows of work array WV available for
//>          workspace.  NV >= NW.
//> \endverbatim
//>
//> \param[out] WV
//> \verbatim
//>          WV is DOUBLE PRECISION array, dimension (LDWV,NW)
//> \endverbatim
//>
//> \param[in] LDWV
//> \verbatim
//>          LDWV is INTEGER
//>          The leading dimension of W just as declared in the
//>          calling subroutine.  NW <= LDV
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (LWORK)
//>          On exit, WORK(1) is set to an estimate of the optimal value
//>          of LWORK for the given values of N, NW, KTOP and KBOT.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the work array WORK.  LWORK = 2*NW
//>          suffices, but greater efficiency may result from larger
//>          values of LWORK.
//>
//>          If LWORK = -1, then a workspace query is assumed; DLAQR3
//>          only estimates the optimal workspace size for the given
//>          values of N, NW, KTOP and KBOT.  The estimate is returned
//>          in WORK(1).  No error message related to LWORK is issued
//>          by XERBLA.  Neither H nor Z are accessed.
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
//> \ingroup doubleOTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>       Karen Braman and Ralph Byers, Department of Mathematics,
//>       University of Kansas, USA
//>
// =====================================================================
/* Subroutine */ int dlaqr3_(int *wantt, int *wantz, int *n, int *ktop, int *
	kbot, int *nw, double *h__, int *ldh, int *iloz, int *ihiz, double *
	z__, int *ldz, int *ns, int *nd, double *sr, double *si, double *v,
	int *ldv, int *nh, double *t, int *ldt, int *nv, double *wv, int *
	ldwv, double *work, int *lwork)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c_true = TRUE_;
    double c_b17 = 0.;
    double c_b18 = 1.;
    int c__12 = 12;

    // System generated locals
    int h_dim1, h_offset, t_dim1, t_offset, v_dim1, v_offset, wv_dim1,
	    wv_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4;
    double d__1, d__2, d__3, d__4, d__5, d__6;

    // Local variables
    int i__, j, k;
    double s, aa, bb, cc, dd, cs, sn;
    int jw;
    double evi, evk, foo;
    int kln;
    double tau, ulp;
    int lwk1, lwk2, lwk3;
    double beta;
    int kend, kcol, info, nmin, ifst, ilst, ltop, krow;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *), dgemm_(char *, char *, int *
	    , int *, int *, double *, double *, int *, double *, int *,
	    double *, double *, int *);
    int bulge;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int infqr, kwtop;
    extern /* Subroutine */ int dlanv2_(double *, double *, double *, double *
	    , double *, double *, double *, double *, double *, double *),
	    dlaqr4_(int *, int *, int *, int *, int *, double *, int *,
	    double *, double *, int *, int *, double *, int *, double *, int *
	    , int *), dlabad_(double *, double *);
    extern double dlamch_(char *);
    extern /* Subroutine */ int dgehrd_(int *, int *, int *, double *, int *,
	    double *, double *, int *, int *), dlarfg_(int *, double *,
	    double *, int *, double *), dlahqr_(int *, int *, int *, int *,
	    int *, double *, int *, double *, double *, int *, int *, double *
	    , int *, int *), dlacpy_(char *, int *, int *, double *, int *,
	    double *, int *);
    double safmin;
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    double safmax;
    extern /* Subroutine */ int dlaset_(char *, int *, int *, double *,
	    double *, double *, int *), dtrexc_(char *, int *, double *, int *
	    , double *, int *, int *, int *, double *, int *), dormhr_(char *,
	     char *, int *, int *, int *, int *, double *, int *, double *,
	    double *, int *, double *, int *, int *);
    int sorted;
    double smlnum;
    int lwkopt;

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // ================================================================
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
    //    ==== Estimate optimal workspace. ====
    //
    // Parameter adjustments
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --sr;
    --si;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    --work;

    // Function Body
    // Computing MIN
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = min(i__1,i__2);
    if (jw <= 2) {
	lwkopt = 1;
    } else {
	//
	//       ==== Workspace query call to DGEHRD ====
	//
	i__1 = jw - 1;
	dgehrd_(&jw, &c__1, &i__1, &t[t_offset], ldt, &work[1], &work[1], &
		c_n1, &info);
	lwk1 = (int) work[1];
	//
	//       ==== Workspace query call to DORMHR ====
	//
	i__1 = jw - 1;
	dormhr_("R", "N", &jw, &jw, &c__1, &i__1, &t[t_offset], ldt, &work[1],
		 &v[v_offset], ldv, &work[1], &c_n1, &info);
	lwk2 = (int) work[1];
	//
	//       ==== Workspace query call to DLAQR4 ====
	//
	dlaqr4_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sr[1],
		&si[1], &c__1, &jw, &v[v_offset], ldv, &work[1], &c_n1, &
		infqr);
	lwk3 = (int) work[1];
	//
	//       ==== Optimal workspace ====
	//
	// Computing MAX
	i__1 = jw + max(lwk1,lwk2);
	lwkopt = max(i__1,lwk3);
    }
    //
    //    ==== Quick return in case of workspace query. ====
    //
    if (*lwork == -1) {
	work[1] = (double) lwkopt;
	return 0;
    }
    //
    //    ==== Nothing to do ...
    //    ... for an empty active block ... ====
    *ns = 0;
    *nd = 0;
    work[1] = 1.;
    if (*ktop > *kbot) {
	return 0;
    }
    //    ... nor for an empty deflation window. ====
    if (*nw < 1) {
	return 0;
    }
    //
    //    ==== Machine constants ====
    //
    safmin = dlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_("PRECISION");
    smlnum = safmin * ((double) (*n) / ulp);
    //
    //    ==== Setup deflation window ====
    //
    // Computing MIN
    i__1 = *nw, i__2 = *kbot - *ktop + 1;
    jw = min(i__1,i__2);
    kwtop = *kbot - jw + 1;
    if (kwtop == *ktop) {
	s = 0.;
    } else {
	s = h__[kwtop + (kwtop - 1) * h_dim1];
    }
    if (*kbot == kwtop) {
	//
	//       ==== 1-by-1 deflation window: not much to do ====
	//
	sr[kwtop] = h__[kwtop + kwtop * h_dim1];
	si[kwtop] = 0.;
	*ns = 1;
	*nd = 0;
	// Computing MAX
	d__2 = smlnum, d__3 = ulp * (d__1 = h__[kwtop + kwtop * h_dim1], abs(
		d__1));
	if (abs(s) <= max(d__2,d__3)) {
	    *ns = 0;
	    *nd = 1;
	    if (kwtop > *ktop) {
		h__[kwtop + (kwtop - 1) * h_dim1] = 0.;
	    }
	}
	work[1] = 1.;
	return 0;
    }
    //
    //    ==== Convert to spike-triangular form.  (In case of a
    //    .    rare QR failure, this routine continues to do
    //    .    aggressive early deflation using that part of
    //    .    the deflation window that converged using INFQR
    //    .    here and there to keep track.) ====
    //
    dlacpy_("U", &jw, &jw, &h__[kwtop + kwtop * h_dim1], ldh, &t[t_offset],
	    ldt);
    i__1 = jw - 1;
    i__2 = *ldh + 1;
    i__3 = *ldt + 1;
    dcopy_(&i__1, &h__[kwtop + 1 + kwtop * h_dim1], &i__2, &t[t_dim1 + 2], &
	    i__3);
    dlaset_("A", &jw, &jw, &c_b17, &c_b18, &v[v_offset], ldv);
    nmin = ilaenv_(&c__12, "DLAQR3", "SV", &jw, &c__1, &jw, lwork);
    if (jw > nmin) {
	dlaqr4_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sr[
		kwtop], &si[kwtop], &c__1, &jw, &v[v_offset], ldv, &work[1],
		lwork, &infqr);
    } else {
	dlahqr_(&c_true, &c_true, &jw, &c__1, &jw, &t[t_offset], ldt, &sr[
		kwtop], &si[kwtop], &c__1, &jw, &v[v_offset], ldv, &infqr);
    }
    //
    //    ==== DTREXC needs a clean margin near the diagonal ====
    //
    i__1 = jw - 3;
    for (j = 1; j <= i__1; ++j) {
	t[j + 2 + j * t_dim1] = 0.;
	t[j + 3 + j * t_dim1] = 0.;
// L10:
    }
    if (jw > 2) {
	t[jw + (jw - 2) * t_dim1] = 0.;
    }
    //
    //    ==== Deflation detection loop ====
    //
    *ns = jw;
    ilst = infqr + 1;
L20:
    if (ilst <= *ns) {
	if (*ns == 1) {
	    bulge = FALSE_;
	} else {
	    bulge = t[*ns + (*ns - 1) * t_dim1] != 0.;
	}
	//
	//       ==== Small spike tip test for deflation ====
	//
	if (! bulge) {
	    //
	    //          ==== Real eigenvalue ====
	    //
	    foo = (d__1 = t[*ns + *ns * t_dim1], abs(d__1));
	    if (foo == 0.) {
		foo = abs(s);
	    }
	    // Computing MAX
	    d__2 = smlnum, d__3 = ulp * foo;
	    if ((d__1 = s * v[*ns * v_dim1 + 1], abs(d__1)) <= max(d__2,d__3))
		     {
		//
		//             ==== Deflatable ====
		//
		--(*ns);
	    } else {
		//
		//             ==== Undeflatable.   Move it up out of the way.
		//             .    (DTREXC can not fail in this case.) ====
		//
		ifst = *ns;
		dtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst,
			 &ilst, &work[1], &info);
		++ilst;
	    }
	} else {
	    //
	    //          ==== Complex conjugate pair ====
	    //
	    foo = (d__3 = t[*ns + *ns * t_dim1], abs(d__3)) + sqrt((d__1 = t[*
		    ns + (*ns - 1) * t_dim1], abs(d__1))) * sqrt((d__2 = t[*
		    ns - 1 + *ns * t_dim1], abs(d__2)));
	    if (foo == 0.) {
		foo = abs(s);
	    }
	    // Computing MAX
	    d__3 = (d__1 = s * v[*ns * v_dim1 + 1], abs(d__1)), d__4 = (d__2 =
		     s * v[(*ns - 1) * v_dim1 + 1], abs(d__2));
	    // Computing MAX
	    d__5 = smlnum, d__6 = ulp * foo;
	    if (max(d__3,d__4) <= max(d__5,d__6)) {
		//
		//             ==== Deflatable ====
		//
		*ns += -2;
	    } else {
		//
		//             ==== Undeflatable. Move them up out of the way.
		//             .    Fortunately, DTREXC does the right thing with
		//             .    ILST in case of a rare exchange failure. ====
		//
		ifst = *ns;
		dtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst,
			 &ilst, &work[1], &info);
		ilst += 2;
	    }
	}
	//
	//       ==== End deflation detection loop ====
	//
	goto L20;
    }
    //
    //       ==== Return to Hessenberg form ====
    //
    if (*ns == 0) {
	s = 0.;
    }
    if (*ns < jw) {
	//
	//       ==== sorting diagonal blocks of T improves accuracy for
	//       .    graded matrices.  Bubble sort deals well with
	//       .    exchange failures. ====
	//
	sorted = FALSE_;
	i__ = *ns + 1;
L30:
	if (sorted) {
	    goto L50;
	}
	sorted = TRUE_;
	kend = i__ - 1;
	i__ = infqr + 1;
	if (i__ == *ns) {
	    k = i__ + 1;
	} else if (t[i__ + 1 + i__ * t_dim1] == 0.) {
	    k = i__ + 1;
	} else {
	    k = i__ + 2;
	}
L40:
	if (k <= kend) {
	    if (k == i__ + 1) {
		evi = (d__1 = t[i__ + i__ * t_dim1], abs(d__1));
	    } else {
		evi = (d__3 = t[i__ + i__ * t_dim1], abs(d__3)) + sqrt((d__1 =
			 t[i__ + 1 + i__ * t_dim1], abs(d__1))) * sqrt((d__2 =
			 t[i__ + (i__ + 1) * t_dim1], abs(d__2)));
	    }
	    if (k == kend) {
		evk = (d__1 = t[k + k * t_dim1], abs(d__1));
	    } else if (t[k + 1 + k * t_dim1] == 0.) {
		evk = (d__1 = t[k + k * t_dim1], abs(d__1));
	    } else {
		evk = (d__3 = t[k + k * t_dim1], abs(d__3)) + sqrt((d__1 = t[
			k + 1 + k * t_dim1], abs(d__1))) * sqrt((d__2 = t[k +
			(k + 1) * t_dim1], abs(d__2)));
	    }
	    if (evi >= evk) {
		i__ = k;
	    } else {
		sorted = FALSE_;
		ifst = i__;
		ilst = k;
		dtrexc_("V", &jw, &t[t_offset], ldt, &v[v_offset], ldv, &ifst,
			 &ilst, &work[1], &info);
		if (info == 0) {
		    i__ = ilst;
		} else {
		    i__ = k;
		}
	    }
	    if (i__ == kend) {
		k = i__ + 1;
	    } else if (t[i__ + 1 + i__ * t_dim1] == 0.) {
		k = i__ + 1;
	    } else {
		k = i__ + 2;
	    }
	    goto L40;
	}
	goto L30;
L50:
	;
    }
    //
    //    ==== Restore shift/eigenvalue array from T ====
    //
    i__ = jw;
L60:
    if (i__ >= infqr + 1) {
	if (i__ == infqr + 1) {
	    sr[kwtop + i__ - 1] = t[i__ + i__ * t_dim1];
	    si[kwtop + i__ - 1] = 0.;
	    --i__;
	} else if (t[i__ + (i__ - 1) * t_dim1] == 0.) {
	    sr[kwtop + i__ - 1] = t[i__ + i__ * t_dim1];
	    si[kwtop + i__ - 1] = 0.;
	    --i__;
	} else {
	    aa = t[i__ - 1 + (i__ - 1) * t_dim1];
	    cc = t[i__ + (i__ - 1) * t_dim1];
	    bb = t[i__ - 1 + i__ * t_dim1];
	    dd = t[i__ + i__ * t_dim1];
	    dlanv2_(&aa, &bb, &cc, &dd, &sr[kwtop + i__ - 2], &si[kwtop + i__
		    - 2], &sr[kwtop + i__ - 1], &si[kwtop + i__ - 1], &cs, &
		    sn);
	    i__ += -2;
	}
	goto L60;
    }
    if (*ns < jw || s == 0.) {
	if (*ns > 1 && s != 0.) {
	    //
	    //          ==== Reflect spike back into lower triangle ====
	    //
	    dcopy_(ns, &v[v_offset], ldv, &work[1], &c__1);
	    beta = work[1];
	    dlarfg_(ns, &beta, &work[2], &c__1, &tau);
	    work[1] = 1.;
	    i__1 = jw - 2;
	    i__2 = jw - 2;
	    dlaset_("L", &i__1, &i__2, &c_b17, &c_b17, &t[t_dim1 + 3], ldt);
	    dlarf_("L", ns, &jw, &work[1], &c__1, &tau, &t[t_offset], ldt, &
		    work[jw + 1]);
	    dlarf_("R", ns, ns, &work[1], &c__1, &tau, &t[t_offset], ldt, &
		    work[jw + 1]);
	    dlarf_("R", &jw, ns, &work[1], &c__1, &tau, &v[v_offset], ldv, &
		    work[jw + 1]);
	    i__1 = *lwork - jw;
	    dgehrd_(&jw, &c__1, ns, &t[t_offset], ldt, &work[1], &work[jw + 1]
		    , &i__1, &info);
	}
	//
	//       ==== Copy updated reduced window into place ====
	//
	if (kwtop > 1) {
	    h__[kwtop + (kwtop - 1) * h_dim1] = s * v[v_dim1 + 1];
	}
	dlacpy_("U", &jw, &jw, &t[t_offset], ldt, &h__[kwtop + kwtop * h_dim1]
		, ldh);
	i__1 = jw - 1;
	i__2 = *ldt + 1;
	i__3 = *ldh + 1;
	dcopy_(&i__1, &t[t_dim1 + 2], &i__2, &h__[kwtop + 1 + kwtop * h_dim1],
		 &i__3);
	//
	//       ==== Accumulate orthogonal matrix in order update
	//       .    H and Z, if requested.  ====
	//
	if (*ns > 1 && s != 0.) {
	    i__1 = *lwork - jw;
	    dormhr_("R", "N", &jw, ns, &c__1, ns, &t[t_offset], ldt, &work[1],
		     &v[v_offset], ldv, &work[jw + 1], &i__1, &info);
	}
	//
	//       ==== Update vertical slab in H ====
	//
	if (*wantt) {
	    ltop = 1;
	} else {
	    ltop = *ktop;
	}
	i__1 = kwtop - 1;
	i__2 = *nv;
	for (krow = ltop; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow +=
		i__2) {
	    // Computing MIN
	    i__3 = *nv, i__4 = kwtop - krow;
	    kln = min(i__3,i__4);
	    dgemm_("N", "N", &kln, &jw, &jw, &c_b18, &h__[krow + kwtop *
		    h_dim1], ldh, &v[v_offset], ldv, &c_b17, &wv[wv_offset],
		    ldwv);
	    dlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &h__[krow + kwtop *
		    h_dim1], ldh);
// L70:
	}
	//
	//       ==== Update horizontal slab in H ====
	//
	if (*wantt) {
	    i__2 = *n;
	    i__1 = *nh;
	    for (kcol = *kbot + 1; i__1 < 0 ? kcol >= i__2 : kcol <= i__2;
		    kcol += i__1) {
		// Computing MIN
		i__3 = *nh, i__4 = *n - kcol + 1;
		kln = min(i__3,i__4);
		dgemm_("C", "N", &jw, &kln, &jw, &c_b18, &v[v_offset], ldv, &
			h__[kwtop + kcol * h_dim1], ldh, &c_b17, &t[t_offset],
			 ldt);
		dlacpy_("A", &jw, &kln, &t[t_offset], ldt, &h__[kwtop + kcol *
			 h_dim1], ldh);
// L80:
	    }
	}
	//
	//       ==== Update vertical slab in Z ====
	//
	if (*wantz) {
	    i__1 = *ihiz;
	    i__2 = *nv;
	    for (krow = *iloz; i__2 < 0 ? krow >= i__1 : krow <= i__1; krow +=
		     i__2) {
		// Computing MIN
		i__3 = *nv, i__4 = *ihiz - krow + 1;
		kln = min(i__3,i__4);
		dgemm_("N", "N", &kln, &jw, &jw, &c_b18, &z__[krow + kwtop *
			z_dim1], ldz, &v[v_offset], ldv, &c_b17, &wv[
			wv_offset], ldwv);
		dlacpy_("A", &kln, &jw, &wv[wv_offset], ldwv, &z__[krow +
			kwtop * z_dim1], ldz);
// L90:
	    }
	}
    }
    //
    //    ==== Return the number of deflations ... ====
    //
    *nd = jw - *ns;
    //
    //    ==== ... and the number of shifts. (Subtracting
    //    .    INFQR from the spike length takes care
    //    .    of the case of a rare QR failure while
    //    .    calculating eigenvalues of the deflation
    //    .    window.)  ====
    //
    *ns -= infqr;
    //
    //     ==== Return optimal workspace. ====
    //
    work[1] = (double) lwkopt;
    //
    //    ==== End of DLAQR3 ====
    //
    return 0;
} // dlaqr3_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAQR4 computes the eigenvalues of a Hessenberg matrix, and optionally the matrices from the Schur decomposition.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAQR4 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaqr4.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaqr4.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaqr4.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAQR4( WANTT, WANTZ, N, ILO, IHI, H, LDH, WR, WI,
//                         ILOZ, IHIZ, Z, LDZ, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, IHIZ, ILO, ILOZ, INFO, LDH, LDZ, LWORK, N
//      LOGICAL            WANTT, WANTZ
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), WI( * ), WORK( * ), WR( * ),
//     $                   Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DLAQR4 implements one level of recursion for DLAQR0.
//>    It is a complete implementation of the small bulge multi-shift
//>    QR algorithm.  It may be called by DLAQR0 and, for large enough
//>    deflation window size, it may be called by DLAQR3.  This
//>    subroutine is identical to DLAQR0 except that it calls DLAQR2
//>    instead of DLAQR3.
//>
//>    DLAQR4 computes the eigenvalues of a Hessenberg matrix H
//>    and, optionally, the matrices T and Z from the Schur decomposition
//>    H = Z T Z**T, where T is an upper quasi-triangular matrix (the
//>    Schur form), and Z is the orthogonal matrix of Schur vectors.
//>
//>    Optionally Z may be postmultiplied into an input orthogonal
//>    matrix Q so that this routine can give the Schur factorization
//>    of a matrix A which has been reduced to the Hessenberg form H
//>    by the orthogonal matrix Q:  A = Q*H*Q**T = (QZ)*T*(QZ)**T.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] WANTT
//> \verbatim
//>          WANTT is LOGICAL
//>          = .TRUE. : the full Schur form T is required;
//>          = .FALSE.: only eigenvalues are required.
//> \endverbatim
//>
//> \param[in] WANTZ
//> \verbatim
//>          WANTZ is LOGICAL
//>          = .TRUE. : the matrix of Schur vectors Z is required;
//>          = .FALSE.: Schur vectors are not required.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           The order of the matrix H.  N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>           It is assumed that H is already upper triangular in rows
//>           and columns 1:ILO-1 and IHI+1:N and, if ILO > 1,
//>           H(ILO,ILO-1) is zero. ILO and IHI are normally set by a
//>           previous call to DGEBAL, and then passed to DGEHRD when the
//>           matrix output by DGEBAL is reduced to Hessenberg form.
//>           Otherwise, ILO and IHI should be set to 1 and N,
//>           respectively.  If N > 0, then 1 <= ILO <= IHI <= N.
//>           If N = 0, then ILO = 1 and IHI = 0.
//> \endverbatim
//>
//> \param[in,out] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>           On entry, the upper Hessenberg matrix H.
//>           On exit, if INFO = 0 and WANTT is .TRUE., then H contains
//>           the upper quasi-triangular matrix T from the Schur
//>           decomposition (the Schur form); 2-by-2 diagonal blocks
//>           (corresponding to complex conjugate pairs of eigenvalues)
//>           are returned in standard form, with H(i,i) = H(i+1,i+1)
//>           and H(i+1,i)*H(i,i+1) < 0. If INFO = 0 and WANTT is
//>           .FALSE., then the contents of H are unspecified on exit.
//>           (The output value of H when INFO > 0 is given under the
//>           description of INFO below.)
//>
//>           This subroutine may explicitly set H(i,j) = 0 for i > j and
//>           j = 1, 2, ... ILO-1 or j = IHI+1, IHI+2, ... N.
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>           The leading dimension of the array H. LDH >= max(1,N).
//> \endverbatim
//>
//> \param[out] WR
//> \verbatim
//>          WR is DOUBLE PRECISION array, dimension (IHI)
//> \endverbatim
//>
//> \param[out] WI
//> \verbatim
//>          WI is DOUBLE PRECISION array, dimension (IHI)
//>           The real and imaginary parts, respectively, of the computed
//>           eigenvalues of H(ILO:IHI,ILO:IHI) are stored in WR(ILO:IHI)
//>           and WI(ILO:IHI). If two eigenvalues are computed as a
//>           complex conjugate pair, they are stored in consecutive
//>           elements of WR and WI, say the i-th and (i+1)th, with
//>           WI(i) > 0 and WI(i+1) < 0. If WANTT is .TRUE., then
//>           the eigenvalues are stored in the same order as on the
//>           diagonal of the Schur form returned in H, with
//>           WR(i) = H(i,i) and, if H(i:i+1,i:i+1) is a 2-by-2 diagonal
//>           block, WI(i) = sqrt(-H(i+1,i)*H(i,i+1)) and
//>           WI(i+1) = -WI(i).
//> \endverbatim
//>
//> \param[in] ILOZ
//> \verbatim
//>          ILOZ is INTEGER
//> \endverbatim
//>
//> \param[in] IHIZ
//> \verbatim
//>          IHIZ is INTEGER
//>           Specify the rows of Z to which transformations must be
//>           applied if WANTZ is .TRUE..
//>           1 <= ILOZ <= ILO; IHI <= IHIZ <= N.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ,IHI)
//>           If WANTZ is .FALSE., then Z is not referenced.
//>           If WANTZ is .TRUE., then Z(ILO:IHI,ILOZ:IHIZ) is
//>           replaced by Z(ILO:IHI,ILOZ:IHIZ)*U where U is the
//>           orthogonal Schur factor of H(ILO:IHI,ILO:IHI).
//>           (The output value of Z when INFO > 0 is given under
//>           the description of INFO below.)
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>           The leading dimension of the array Z.  if WANTZ is .TRUE.
//>           then LDZ >= MAX(1,IHIZ).  Otherwise, LDZ >= 1.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension LWORK
//>           On exit, if LWORK = -1, WORK(1) returns an estimate of
//>           the optimal value for LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>           The dimension of the array WORK.  LWORK >= max(1,N)
//>           is sufficient, but LWORK typically as large as 6*N may
//>           be required for optimal performance.  A workspace query
//>           to determine the optimal workspace size is recommended.
//>
//>           If LWORK = -1, then DLAQR4 does a workspace query.
//>           In this case, DLAQR4 checks the input parameters and
//>           estimates the optimal workspace size for the given
//>           values of N, ILO and IHI.  The estimate is returned
//>           in WORK(1).  No error message related to LWORK is
//>           issued by XERBLA.  Neither H nor Z are accessed.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>             = 0:  successful exit
//>             > 0:  if INFO = i, DLAQR4 failed to compute all of
//>                the eigenvalues.  Elements 1:ilo-1 and i+1:n of WR
//>                and WI contain those eigenvalues which have been
//>                successfully computed.  (Failures are rare.)
//>
//>                If INFO > 0 and WANT is .FALSE., then on exit,
//>                the remaining unconverged eigenvalues are the eigen-
//>                values of the upper Hessenberg matrix rows and
//>                columns ILO through INFO of the final, output
//>                value of H.
//>
//>                If INFO > 0 and WANTT is .TRUE., then on exit
//>
//>           (*)  (initial value of H)*U  = U*(final value of H)
//>
//>                where U is a orthogonal matrix.  The final
//>                value of  H is upper Hessenberg and triangular in
//>                rows and columns INFO+1 through IHI.
//>
//>                If INFO > 0 and WANTZ is .TRUE., then on exit
//>
//>                  (final value of Z(ILO:IHI,ILOZ:IHIZ)
//>                   =  (initial value of Z(ILO:IHI,ILOZ:IHIZ)*U
//>
//>                where U is the orthogonal matrix in (*) (regard-
//>                less of the value of WANTT.)
//>
//>                If INFO > 0 and WANTZ is .FALSE., then Z is not
//>                accessed.
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
//> \par Contributors:
// ==================
//>
//>       Karen Braman and Ralph Byers, Department of Mathematics,
//>       University of Kansas, USA
//
//> \par References:
// ================
//>
//>       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//>       Algorithm Part I: Maintaining Well Focused Shifts, and Level 3
//>       Performance, SIAM Journal of Matrix Analysis, volume 23, pages
//>       929--947, 2002.
//> \n
//>       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//>       Algorithm Part II: Aggressive Early Deflation, SIAM Journal
//>       of Matrix Analysis, volume 23, pages 948--973, 2002.
//>
// =====================================================================
/* Subroutine */ int dlaqr4_(int *wantt, int *wantz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, int *iloz, int *
	ihiz, double *z__, int *ldz, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__13 = 13;
    int c__15 = 15;
    int c_n1 = -1;
    int c__12 = 12;
    int c__14 = 14;
    int c__16 = 16;
    int c_false = FALSE_;
    int c__1 = 1;
    int c__3 = 3;

    // System generated locals
    int h_dim1, h_offset, z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    double d__1, d__2, d__3, d__4;

    // Local variables
    int i__, k;
    double aa, bb, cc, dd;
    int ld;
    double cs;
    int nh, it, ks, kt;
    double sn;
    int ku, kv, ls, ns;
    double ss;
    int nw, inf, kdu, nho, nve, kwh, nsr, nwr, kwv, ndec, ndfl, kbot, nmin;
    double swap;
    int ktop;
    double zdum[1]	/* was [1][1] */;
    int kacc22, itmax, nsmax, nwmax, kwtop;
    extern /* Subroutine */ int dlaqr2_(int *, int *, int *, int *, int *,
	    int *, double *, int *, int *, int *, double *, int *, int *, int
	    *, double *, double *, double *, int *, int *, double *, int *,
	    int *, double *, int *, double *, int *), dlanv2_(double *,
	    double *, double *, double *, double *, double *, double *,
	    double *, double *, double *), dlaqr5_(int *, int *, int *, int *,
	     int *, int *, int *, double *, double *, double *, int *, int *,
	    int *, double *, int *, double *, int *, double *, int *, int *,
	    double *, int *, int *, double *, int *);
    int nibble;
    extern /* Subroutine */ int dlahqr_(int *, int *, int *, int *, int *,
	    double *, int *, double *, double *, int *, int *, double *, int *
	    , int *), dlacpy_(char *, int *, int *, double *, int *, double *,
	     int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    char jbcmpz[2+1]={'\0'};
    int nwupbd;
    int sorted;
    int lwkopt;

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
    // ================================================================
    //    .. Parameters ..
    //
    //    ==== Matrices of order NTINY or smaller must be processed by
    //    .    DLAHQR because of insufficient subdiagonal scratch space.
    //    .    (This is a hard limit.) ====
    //
    //    ==== Exceptional deflation windows:  try to cure rare
    //    .    slow convergence by varying the size of the
    //    .    deflation window after KEXNW iterations. ====
    //
    //    ==== Exceptional shifts: try to cure rare slow convergence
    //    .    with ad-hoc exceptional shifts every KEXSH iterations.
    //    .    ====
    //
    //    ==== The constants WILK1 and WILK2 are used to form the
    //    .    exceptional shifts. ====
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    // Parameter adjustments
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    --wr;
    --wi;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;

    // Function Body
    *info = 0;
    //
    //    ==== Quick return for N = 0: nothing to do. ====
    //
    if (*n == 0) {
	work[1] = 1.;
	return 0;
    }
    if (*n <= 11) {
	//
	//       ==== Tiny matrices must use DLAHQR. ====
	//
	lwkopt = 1;
	if (*lwork != -1) {
	    dlahqr_(wantt, wantz, n, ilo, ihi, &h__[h_offset], ldh, &wr[1], &
		    wi[1], iloz, ihiz, &z__[z_offset], ldz, info);
	}
    } else {
	//
	//       ==== Use small bulge multi-shift QR with aggressive early
	//       .    deflation on larger-than-tiny matrices. ====
	//
	//       ==== Hope for the best. ====
	//
	*info = 0;
	//
	//       ==== Set up job flags for ILAENV. ====
	//
	if (*wantt) {
	    *(unsigned char *)jbcmpz = 'S';
	} else {
	    *(unsigned char *)jbcmpz = 'E';
	}
	if (*wantz) {
	    *(unsigned char *)&jbcmpz[1] = 'V';
	} else {
	    *(unsigned char *)&jbcmpz[1] = 'N';
	}
	//
	//       ==== NWR = recommended deflation window size.  At this
	//       .    point,  N .GT. NTINY = 11, so there is enough
	//       .    subdiagonal workspace for NWR.GE.2 as required.
	//       .    (In fact, there is enough subdiagonal space for
	//       .    NWR.GE.3.) ====
	//
	nwr = ilaenv_(&c__13, "DLAQR4", jbcmpz, n, ilo, ihi, lwork);
	nwr = max(2,nwr);
	// Computing MIN
	i__1 = *ihi - *ilo + 1, i__2 = (*n - 1) / 3, i__1 = min(i__1,i__2);
	nwr = min(i__1,nwr);
	//
	//       ==== NSR = recommended number of simultaneous shifts.
	//       .    At this point N .GT. NTINY = 11, so there is at
	//       .    enough subdiagonal workspace for NSR to be even
	//       .    and greater than or equal to two as required. ====
	//
	nsr = ilaenv_(&c__15, "DLAQR4", jbcmpz, n, ilo, ihi, lwork);
	// Computing MIN
	i__1 = nsr, i__2 = (*n + 6) / 9, i__1 = min(i__1,i__2), i__2 = *ihi -
		*ilo;
	nsr = min(i__1,i__2);
	// Computing MAX
	i__1 = 2, i__2 = nsr - nsr % 2;
	nsr = max(i__1,i__2);
	//
	//       ==== Estimate optimal workspace ====
	//
	//       ==== Workspace query call to DLAQR2 ====
	//
	i__1 = nwr + 1;
	dlaqr2_(wantt, wantz, n, ilo, ihi, &i__1, &h__[h_offset], ldh, iloz,
		ihiz, &z__[z_offset], ldz, &ls, &ld, &wr[1], &wi[1], &h__[
		h_offset], ldh, n, &h__[h_offset], ldh, n, &h__[h_offset],
		ldh, &work[1], &c_n1);
	//
	//       ==== Optimal workspace = MAX(DLAQR5, DLAQR2) ====
	//
	// Computing MAX
	i__1 = nsr * 3 / 2, i__2 = (int) work[1];
	lwkopt = max(i__1,i__2);
	//
	//       ==== Quick return in case of workspace query. ====
	//
	if (*lwork == -1) {
	    work[1] = (double) lwkopt;
	    return 0;
	}
	//
	//       ==== DLAHQR/DLAQR0 crossover point ====
	//
	nmin = ilaenv_(&c__12, "DLAQR4", jbcmpz, n, ilo, ihi, lwork);
	nmin = max(11,nmin);
	//
	//       ==== Nibble crossover point ====
	//
	nibble = ilaenv_(&c__14, "DLAQR4", jbcmpz, n, ilo, ihi, lwork);
	nibble = max(0,nibble);
	//
	//       ==== Accumulate reflections during ttswp?  Use block
	//       .    2-by-2 structure during matrix-matrix multiply? ====
	//
	kacc22 = ilaenv_(&c__16, "DLAQR4", jbcmpz, n, ilo, ihi, lwork);
	kacc22 = max(0,kacc22);
	kacc22 = min(2,kacc22);
	//
	//       ==== NWMAX = the largest possible deflation window for
	//       .    which there is sufficient workspace. ====
	//
	// Computing MIN
	i__1 = (*n - 1) / 3, i__2 = *lwork / 2;
	nwmax = min(i__1,i__2);
	nw = nwmax;
	//
	//       ==== NSMAX = the Largest number of simultaneous shifts
	//       .    for which there is sufficient workspace. ====
	//
	// Computing MIN
	i__1 = (*n + 6) / 9, i__2 = (*lwork << 1) / 3;
	nsmax = min(i__1,i__2);
	nsmax -= nsmax % 2;
	//
	//       ==== NDFL: an iteration count restarted at deflation. ====
	//
	ndfl = 1;
	//
	//       ==== ITMAX = iteration limit ====
	//
	// Computing MAX
	i__1 = 10, i__2 = *ihi - *ilo + 1;
	itmax = max(i__1,i__2) * 30;
	//
	//       ==== Last row and column in the active block ====
	//
	kbot = *ihi;
	//
	//       ==== Main Loop ====
	//
	i__1 = itmax;
	for (it = 1; it <= i__1; ++it) {
	    //
	    //          ==== Done when KBOT falls below ILO ====
	    //
	    if (kbot < *ilo) {
		goto L90;
	    }
	    //
	    //          ==== Locate active block ====
	    //
	    i__2 = *ilo + 1;
	    for (k = kbot; k >= i__2; --k) {
		if (h__[k + (k - 1) * h_dim1] == 0.) {
		    goto L20;
		}
// L10:
	    }
	    k = *ilo;
L20:
	    ktop = k;
	    //
	    //          ==== Select deflation window size:
	    //          .    Typical Case:
	    //          .      If possible and advisable, nibble the entire
	    //          .      active block.  If not, use size MIN(NWR,NWMAX)
	    //          .      or MIN(NWR+1,NWMAX) depending upon which has
	    //          .      the smaller corresponding subdiagonal entry
	    //          .      (a heuristic).
	    //          .
	    //          .    Exceptional Case:
	    //          .      If there have been no deflations in KEXNW or
	    //          .      more iterations, then vary the deflation window
	    //          .      size.   At first, because, larger windows are,
	    //          .      in general, more powerful than smaller ones,
	    //          .      rapidly increase the window to the maximum possible.
	    //          .      Then, gradually reduce the window size. ====
	    //
	    nh = kbot - ktop + 1;
	    nwupbd = min(nh,nwmax);
	    if (ndfl < 5) {
		nw = min(nwupbd,nwr);
	    } else {
		// Computing MIN
		i__2 = nwupbd, i__3 = nw << 1;
		nw = min(i__2,i__3);
	    }
	    if (nw < nwmax) {
		if (nw >= nh - 1) {
		    nw = nh;
		} else {
		    kwtop = kbot - nw + 1;
		    if ((d__1 = h__[kwtop + (kwtop - 1) * h_dim1], abs(d__1))
			    > (d__2 = h__[kwtop - 1 + (kwtop - 2) * h_dim1],
			    abs(d__2))) {
			++nw;
		    }
		}
	    }
	    if (ndfl < 5) {
		ndec = -1;
	    } else if (ndec >= 0 || nw >= nwupbd) {
		++ndec;
		if (nw - ndec < 2) {
		    ndec = 0;
		}
		nw -= ndec;
	    }
	    //
	    //          ==== Aggressive early deflation:
	    //          .    split workspace under the subdiagonal into
	    //          .      - an nw-by-nw work array V in the lower
	    //          .        left-hand-corner,
	    //          .      - an NW-by-at-least-NW-but-more-is-better
	    //          .        (NW-by-NHO) horizontal work array along
	    //          .        the bottom edge,
	    //          .      - an at-least-NW-but-more-is-better (NHV-by-NW)
	    //          .        vertical work array along the left-hand-edge.
	    //          .        ====
	    //
	    kv = *n - nw + 1;
	    kt = nw + 1;
	    nho = *n - nw - 1 - kt + 1;
	    kwv = nw + 2;
	    nve = *n - nw - kwv + 1;
	    //
	    //          ==== Aggressive early deflation ====
	    //
	    dlaqr2_(wantt, wantz, n, &ktop, &kbot, &nw, &h__[h_offset], ldh,
		    iloz, ihiz, &z__[z_offset], ldz, &ls, &ld, &wr[1], &wi[1],
		     &h__[kv + h_dim1], ldh, &nho, &h__[kv + kt * h_dim1],
		    ldh, &nve, &h__[kwv + h_dim1], ldh, &work[1], lwork);
	    //
	    //          ==== Adjust KBOT accounting for new deflations. ====
	    //
	    kbot -= ld;
	    //
	    //          ==== KS points to the shifts. ====
	    //
	    ks = kbot - ls + 1;
	    //
	    //          ==== Skip an expensive QR sweep if there is a (partly
	    //          .    heuristic) reason to expect that many eigenvalues
	    //          .    will deflate without it.  Here, the QR sweep is
	    //          .    skipped if many eigenvalues have just been deflated
	    //          .    or if the remaining active block is small.
	    //
	    if (ld == 0 || ld * 100 <= nw * nibble && kbot - ktop + 1 > min(
		    nmin,nwmax)) {
		//
		//             ==== NS = nominal number of simultaneous shifts.
		//             .    This may be lowered (slightly) if DLAQR2
		//             .    did not provide that many shifts. ====
		//
		// Computing MIN
		// Computing MAX
		i__4 = 2, i__5 = kbot - ktop;
		i__2 = min(nsmax,nsr), i__3 = max(i__4,i__5);
		ns = min(i__2,i__3);
		ns -= ns % 2;
		//
		//             ==== If there have been no deflations
		//             .    in a multiple of KEXSH iterations,
		//             .    then try exceptional shifts.
		//             .    Otherwise use shifts provided by
		//             .    DLAQR2 above or from the eigenvalues
		//             .    of a trailing principal submatrix. ====
		//
		if (ndfl % 6 == 0) {
		    ks = kbot - ns + 1;
		    // Computing MAX
		    i__3 = ks + 1, i__4 = ktop + 2;
		    i__2 = max(i__3,i__4);
		    for (i__ = kbot; i__ >= i__2; i__ += -2) {
			ss = (d__1 = h__[i__ + (i__ - 1) * h_dim1], abs(d__1))
				 + (d__2 = h__[i__ - 1 + (i__ - 2) * h_dim1],
				abs(d__2));
			aa = ss * .75 + h__[i__ + i__ * h_dim1];
			bb = ss;
			cc = ss * -.4375;
			dd = aa;
			dlanv2_(&aa, &bb, &cc, &dd, &wr[i__ - 1], &wi[i__ - 1]
				, &wr[i__], &wi[i__], &cs, &sn);
// L30:
		    }
		    if (ks == ktop) {
			wr[ks + 1] = h__[ks + 1 + (ks + 1) * h_dim1];
			wi[ks + 1] = 0.;
			wr[ks] = wr[ks + 1];
			wi[ks] = wi[ks + 1];
		    }
		} else {
		    //
		    //                ==== Got NS/2 or fewer shifts? Use DLAHQR
		    //                .    on a trailing principal submatrix to
		    //                .    get more. (Since NS.LE.NSMAX.LE.(N+6)/9,
		    //                .    there is enough space below the subdiagonal
		    //                .    to fit an NS-by-NS scratch array.) ====
		    //
		    if (kbot - ks + 1 <= ns / 2) {
			ks = kbot - ns + 1;
			kt = *n - ns + 1;
			dlacpy_("A", &ns, &ns, &h__[ks + ks * h_dim1], ldh, &
				h__[kt + h_dim1], ldh);
			dlahqr_(&c_false, &c_false, &ns, &c__1, &ns, &h__[kt
				+ h_dim1], ldh, &wr[ks], &wi[ks], &c__1, &
				c__1, zdum, &c__1, &inf);
			ks += inf;
			//
			//                   ==== In case of a rare QR failure use
			//                   .    eigenvalues of the trailing 2-by-2
			//                   .    principal submatrix.  ====
			//
			if (ks >= kbot) {
			    aa = h__[kbot - 1 + (kbot - 1) * h_dim1];
			    cc = h__[kbot + (kbot - 1) * h_dim1];
			    bb = h__[kbot - 1 + kbot * h_dim1];
			    dd = h__[kbot + kbot * h_dim1];
			    dlanv2_(&aa, &bb, &cc, &dd, &wr[kbot - 1], &wi[
				    kbot - 1], &wr[kbot], &wi[kbot], &cs, &sn)
				    ;
			    ks = kbot - 1;
			}
		    }
		    if (kbot - ks + 1 > ns) {
			//
			//                   ==== Sort the shifts (Helps a little)
			//                   .    Bubble sort keeps complex conjugate
			//                   .    pairs together. ====
			//
			sorted = FALSE_;
			i__2 = ks + 1;
			for (k = kbot; k >= i__2; --k) {
			    if (sorted) {
				goto L60;
			    }
			    sorted = TRUE_;
			    i__3 = k - 1;
			    for (i__ = ks; i__ <= i__3; ++i__) {
				if ((d__1 = wr[i__], abs(d__1)) + (d__2 = wi[
					i__], abs(d__2)) < (d__3 = wr[i__ + 1]
					, abs(d__3)) + (d__4 = wi[i__ + 1],
					abs(d__4))) {
				    sorted = FALSE_;
				    swap = wr[i__];
				    wr[i__] = wr[i__ + 1];
				    wr[i__ + 1] = swap;
				    swap = wi[i__];
				    wi[i__] = wi[i__ + 1];
				    wi[i__ + 1] = swap;
				}
// L40:
			    }
// L50:
			}
L60:
			;
		    }
		    //
		    //                ==== Shuffle shifts into pairs of real shifts
		    //                .    and pairs of complex conjugate shifts
		    //                .    assuming complex conjugate shifts are
		    //                .    already adjacent to one another. (Yes,
		    //                .    they are.)  ====
		    //
		    i__2 = ks + 2;
		    for (i__ = kbot; i__ >= i__2; i__ += -2) {
			if (wi[i__] != -wi[i__ - 1]) {
			    swap = wr[i__];
			    wr[i__] = wr[i__ - 1];
			    wr[i__ - 1] = wr[i__ - 2];
			    wr[i__ - 2] = swap;
			    swap = wi[i__];
			    wi[i__] = wi[i__ - 1];
			    wi[i__ - 1] = wi[i__ - 2];
			    wi[i__ - 2] = swap;
			}
// L70:
		    }
		}
		//
		//             ==== If there are only two shifts and both are
		//             .    real, then use only one.  ====
		//
		if (kbot - ks + 1 == 2) {
		    if (wi[kbot] == 0.) {
			if ((d__1 = wr[kbot] - h__[kbot + kbot * h_dim1], abs(
				d__1)) < (d__2 = wr[kbot - 1] - h__[kbot +
				kbot * h_dim1], abs(d__2))) {
			    wr[kbot - 1] = wr[kbot];
			} else {
			    wr[kbot] = wr[kbot - 1];
			}
		    }
		}
		//
		//             ==== Use up to NS of the the smallest magnitude
		//             .    shifts.  If there aren't NS shifts available,
		//             .    then use them all, possibly dropping one to
		//             .    make the number of shifts even. ====
		//
		// Computing MIN
		i__2 = ns, i__3 = kbot - ks + 1;
		ns = min(i__2,i__3);
		ns -= ns % 2;
		ks = kbot - ns + 1;
		//
		//             ==== Small-bulge multi-shift QR sweep:
		//             .    split workspace under the subdiagonal into
		//             .    - a KDU-by-KDU work array U in the lower
		//             .      left-hand-corner,
		//             .    - a KDU-by-at-least-KDU-but-more-is-better
		//             .      (KDU-by-NHo) horizontal work array WH along
		//             .      the bottom edge,
		//             .    - and an at-least-KDU-but-more-is-better-by-KDU
		//             .      (NVE-by-KDU) vertical work WV arrow along
		//             .      the left-hand-edge. ====
		//
		kdu = ns * 3 - 3;
		ku = *n - kdu + 1;
		kwh = kdu + 1;
		nho = *n - kdu - 3 - (kdu + 1) + 1;
		kwv = kdu + 4;
		nve = *n - kdu - kwv + 1;
		//
		//             ==== Small-bulge multi-shift QR sweep ====
		//
		dlaqr5_(wantt, wantz, &kacc22, n, &ktop, &kbot, &ns, &wr[ks],
			&wi[ks], &h__[h_offset], ldh, iloz, ihiz, &z__[
			z_offset], ldz, &work[1], &c__3, &h__[ku + h_dim1],
			ldh, &nve, &h__[kwv + h_dim1], ldh, &nho, &h__[ku +
			kwh * h_dim1], ldh);
	    }
	    //
	    //          ==== Note progress (or the lack of it). ====
	    //
	    if (ld > 0) {
		ndfl = 1;
	    } else {
		++ndfl;
	    }
	    //
	    //          ==== End of main loop ====
// L80:
	}
	//
	//       ==== Iteration limit exceeded.  Set INFO to show where
	//       .    the problem occurred and exit. ====
	//
	*info = kbot;
L90:
	;
    }
    //
    //    ==== Return the optimal value of LWORK. ====
    //
    work[1] = (double) lwkopt;
    //
    //    ==== End of DLAQR4 ====
    //
    return 0;
} // dlaqr4_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAQR5 performs a single small-bulge multi-shift QR sweep.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAQR5 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaqr5.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaqr5.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaqr5.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAQR5( WANTT, WANTZ, KACC22, N, KTOP, KBOT, NSHFTS,
//                         SR, SI, H, LDH, ILOZ, IHIZ, Z, LDZ, V, LDV, U,
//                         LDU, NV, WV, LDWV, NH, WH, LDWH )
//
//      .. Scalar Arguments ..
//      INTEGER            IHIZ, ILOZ, KACC22, KBOT, KTOP, LDH, LDU, LDV,
//     $                   LDWH, LDWV, LDZ, N, NH, NSHFTS, NV
//      LOGICAL            WANTT, WANTZ
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   H( LDH, * ), SI( * ), SR( * ), U( LDU, * ),
//     $                   V( LDV, * ), WH( LDWH, * ), WV( LDWV, * ),
//     $                   Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    DLAQR5, called by DLAQR0, performs a
//>    single small-bulge multi-shift QR sweep.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] WANTT
//> \verbatim
//>          WANTT is LOGICAL
//>             WANTT = .true. if the quasi-triangular Schur factor
//>             is being computed.  WANTT is set to .false. otherwise.
//> \endverbatim
//>
//> \param[in] WANTZ
//> \verbatim
//>          WANTZ is LOGICAL
//>             WANTZ = .true. if the orthogonal Schur factor is being
//>             computed.  WANTZ is set to .false. otherwise.
//> \endverbatim
//>
//> \param[in] KACC22
//> \verbatim
//>          KACC22 is INTEGER with value 0, 1, or 2.
//>             Specifies the computation mode of far-from-diagonal
//>             orthogonal updates.
//>        = 0: DLAQR5 does not accumulate reflections and does not
//>             use matrix-matrix multiply to update far-from-diagonal
//>             matrix entries.
//>        = 1: DLAQR5 accumulates reflections and uses matrix-matrix
//>             multiply to update the far-from-diagonal matrix entries.
//>        = 2: DLAQR5 accumulates reflections, uses matrix-matrix
//>             multiply to update the far-from-diagonal matrix entries,
//>             and takes advantage of 2-by-2 block structure during
//>             matrix multiplies.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>             N is the order of the Hessenberg matrix H upon which this
//>             subroutine operates.
//> \endverbatim
//>
//> \param[in] KTOP
//> \verbatim
//>          KTOP is INTEGER
//> \endverbatim
//>
//> \param[in] KBOT
//> \verbatim
//>          KBOT is INTEGER
//>             These are the first and last rows and columns of an
//>             isolated diagonal block upon which the QR sweep is to be
//>             applied. It is assumed without a check that
//>                       either KTOP = 1  or   H(KTOP,KTOP-1) = 0
//>             and
//>                       either KBOT = N  or   H(KBOT+1,KBOT) = 0.
//> \endverbatim
//>
//> \param[in] NSHFTS
//> \verbatim
//>          NSHFTS is INTEGER
//>             NSHFTS gives the number of simultaneous shifts.  NSHFTS
//>             must be positive and even.
//> \endverbatim
//>
//> \param[in,out] SR
//> \verbatim
//>          SR is DOUBLE PRECISION array, dimension (NSHFTS)
//> \endverbatim
//>
//> \param[in,out] SI
//> \verbatim
//>          SI is DOUBLE PRECISION array, dimension (NSHFTS)
//>             SR contains the real parts and SI contains the imaginary
//>             parts of the NSHFTS shifts of origin that define the
//>             multi-shift QR sweep.  On output SR and SI may be
//>             reordered.
//> \endverbatim
//>
//> \param[in,out] H
//> \verbatim
//>          H is DOUBLE PRECISION array, dimension (LDH,N)
//>             On input H contains a Hessenberg matrix.  On output a
//>             multi-shift QR sweep with shifts SR(J)+i*SI(J) is applied
//>             to the isolated diagonal block in rows and columns KTOP
//>             through KBOT.
//> \endverbatim
//>
//> \param[in] LDH
//> \verbatim
//>          LDH is INTEGER
//>             LDH is the leading dimension of H just as declared in the
//>             calling procedure.  LDH >= MAX(1,N).
//> \endverbatim
//>
//> \param[in] ILOZ
//> \verbatim
//>          ILOZ is INTEGER
//> \endverbatim
//>
//> \param[in] IHIZ
//> \verbatim
//>          IHIZ is INTEGER
//>             Specify the rows of Z to which transformations must be
//>             applied if WANTZ is .TRUE.. 1 <= ILOZ <= IHIZ <= N
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ,IHIZ)
//>             If WANTZ = .TRUE., then the QR Sweep orthogonal
//>             similarity transformation is accumulated into
//>             Z(ILOZ:IHIZ,ILOZ:IHIZ) from the right.
//>             If WANTZ = .FALSE., then Z is unreferenced.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>             LDA is the leading dimension of Z just as declared in
//>             the calling procedure. LDZ >= N.
//> \endverbatim
//>
//> \param[out] V
//> \verbatim
//>          V is DOUBLE PRECISION array, dimension (LDV,NSHFTS/2)
//> \endverbatim
//>
//> \param[in] LDV
//> \verbatim
//>          LDV is INTEGER
//>             LDV is the leading dimension of V as declared in the
//>             calling procedure.  LDV >= 3.
//> \endverbatim
//>
//> \param[out] U
//> \verbatim
//>          U is DOUBLE PRECISION array, dimension (LDU,3*NSHFTS-3)
//> \endverbatim
//>
//> \param[in] LDU
//> \verbatim
//>          LDU is INTEGER
//>             LDU is the leading dimension of U just as declared in the
//>             in the calling subroutine.  LDU >= 3*NSHFTS-3.
//> \endverbatim
//>
//> \param[in] NV
//> \verbatim
//>          NV is INTEGER
//>             NV is the number of rows in WV agailable for workspace.
//>             NV >= 1.
//> \endverbatim
//>
//> \param[out] WV
//> \verbatim
//>          WV is DOUBLE PRECISION array, dimension (LDWV,3*NSHFTS-3)
//> \endverbatim
//>
//> \param[in] LDWV
//> \verbatim
//>          LDWV is INTEGER
//>             LDWV is the leading dimension of WV as declared in the
//>             in the calling subroutine.  LDWV >= NV.
//> \endverbatim
//
//> \param[in] NH
//> \verbatim
//>          NH is INTEGER
//>             NH is the number of columns in array WH available for
//>             workspace. NH >= 1.
//> \endverbatim
//>
//> \param[out] WH
//> \verbatim
//>          WH is DOUBLE PRECISION array, dimension (LDWH,NH)
//> \endverbatim
//>
//> \param[in] LDWH
//> \verbatim
//>          LDWH is INTEGER
//>             Leading dimension of WH just as declared in the
//>             calling procedure.  LDWH >= 3*NSHFTS-3.
//> \endverbatim
//>
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
//> \ingroup doubleOTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//>       Karen Braman and Ralph Byers, Department of Mathematics,
//>       University of Kansas, USA
//
//> \par References:
// ================
//>
//>       K. Braman, R. Byers and R. Mathias, The Multi-Shift QR
//>       Algorithm Part I: Maintaining Well Focused Shifts, and Level 3
//>       Performance, SIAM Journal of Matrix Analysis, volume 23, pages
//>       929--947, 2002.
//>
// =====================================================================
/* Subroutine */ int dlaqr5_(int *wantt, int *wantz, int *kacc22, int *n, int
	*ktop, int *kbot, int *nshfts, double *sr, double *si, double *h__,
	int *ldh, int *iloz, int *ihiz, double *z__, int *ldz, double *v, int
	*ldv, double *u, int *ldu, int *nv, double *wv, int *ldwv, int *nh,
	double *wh, int *ldwh)
{
    // Table of constant values
    double c_b7 = 0.;
    double c_b8 = 1.;
    int c__3 = 3;
    int c__1 = 1;
    int c__2 = 2;

    // System generated locals
    int h_dim1, h_offset, u_dim1, u_offset, v_dim1, v_offset, wh_dim1,
	    wh_offset, wv_dim1, wv_offset, z_dim1, z_offset, i__1, i__2, i__3,
	     i__4, i__5, i__6, i__7;
    double d__1, d__2, d__3, d__4, d__5;

    // Local variables
    int i__, j, k, m, i2, j2, i4, j4, k1;
    double h11, h12, h21, h22;
    int m22, ns, nu;
    double vt[3], scl;
    int kdu, kms;
    double ulp;
    int knz, kzs;
    double tst1, tst2, beta;
    int blk22, bmp22;
    int mend, jcol, jlen, jbot, mbot;
    double swap;
    int jtop, jrow, mtop;
    double alpha;
    int accum;
    extern /* Subroutine */ int dgemm_(char *, char *, int *, int *, int *,
	    double *, double *, int *, double *, int *, double *, double *,
	    int *);
    int ndcol, incol, krcol, nbmps;
    extern /* Subroutine */ int dtrmm_(char *, char *, char *, char *, int *,
	    int *, double *, double *, int *, double *, int *), dlaqr1_(int *,
	     double *, int *, double *, double *, double *, double *, double *
	    ), dlabad_(double *, double *);
    extern double dlamch_(char *);
    extern /* Subroutine */ int dlarfg_(int *, double *, double *, int *,
	    double *), dlacpy_(char *, int *, int *, double *, int *, double *
	    , int *);
    double safmin;
    extern /* Subroutine */ int dlaset_(char *, int *, int *, double *,
	    double *, double *, int *);
    double safmax, refsum;
    int mstart;
    double smlnum;

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //    .. Array Arguments ..
    //    ..
    //
    // ================================================================
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Intrinsic Functions ..
    //
    //    ..
    //    .. Local Arrays ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    //    ==== If there are no shifts, then there is nothing to do. ====
    //
    // Parameter adjustments
    --sr;
    --si;
    h_dim1 = *ldh;
    h_offset = 1 + h_dim1;
    h__ -= h_offset;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    u_dim1 = *ldu;
    u_offset = 1 + u_dim1;
    u -= u_offset;
    wv_dim1 = *ldwv;
    wv_offset = 1 + wv_dim1;
    wv -= wv_offset;
    wh_dim1 = *ldwh;
    wh_offset = 1 + wh_dim1;
    wh -= wh_offset;

    // Function Body
    if (*nshfts < 2) {
	return 0;
    }
    //
    //    ==== If the active block is empty or 1-by-1, then there
    //    .    is nothing to do. ====
    //
    if (*ktop >= *kbot) {
	return 0;
    }
    //
    //    ==== Shuffle shifts into pairs of real shifts and pairs
    //    .    of complex conjugate shifts assuming complex
    //    .    conjugate shifts are already adjacent to one
    //    .    another. ====
    //
    i__1 = *nshfts - 2;
    for (i__ = 1; i__ <= i__1; i__ += 2) {
	if (si[i__] != -si[i__ + 1]) {
	    swap = sr[i__];
	    sr[i__] = sr[i__ + 1];
	    sr[i__ + 1] = sr[i__ + 2];
	    sr[i__ + 2] = swap;
	    swap = si[i__];
	    si[i__] = si[i__ + 1];
	    si[i__ + 1] = si[i__ + 2];
	    si[i__ + 2] = swap;
	}
// L10:
    }
    //
    //    ==== NSHFTS is supposed to be even, but if it is odd,
    //    .    then simply reduce it by one.  The shuffle above
    //    .    ensures that the dropped shift is real and that
    //    .    the remaining shifts are paired. ====
    //
    ns = *nshfts - *nshfts % 2;
    //
    //    ==== Machine constants for deflation ====
    //
    safmin = dlamch_("SAFE MINIMUM");
    safmax = 1. / safmin;
    dlabad_(&safmin, &safmax);
    ulp = dlamch_("PRECISION");
    smlnum = safmin * ((double) (*n) / ulp);
    //
    //    ==== Use accumulated reflections to update far-from-diagonal
    //    .    entries ? ====
    //
    accum = *kacc22 == 1 || *kacc22 == 2;
    //
    //    ==== If so, exploit the 2-by-2 block structure? ====
    //
    blk22 = ns > 2 && *kacc22 == 2;
    //
    //    ==== clear trash ====
    //
    if (*ktop + 2 <= *kbot) {
	h__[*ktop + 2 + *ktop * h_dim1] = 0.;
    }
    //
    //    ==== NBMPS = number of 2-shift bulges in the chain ====
    //
    nbmps = ns / 2;
    //
    //    ==== KDU = width of slab ====
    //
    kdu = nbmps * 6 - 3;
    //
    //    ==== Create and chase chains of NBMPS bulges ====
    //
    i__1 = *kbot - 2;
    i__2 = nbmps * 3 - 2;
    for (incol = (1 - nbmps) * 3 + *ktop - 1; i__2 < 0 ? incol >= i__1 :
	    incol <= i__1; incol += i__2) {
	ndcol = incol + kdu;
	if (accum) {
	    dlaset_("ALL", &kdu, &kdu, &c_b7, &c_b8, &u[u_offset], ldu);
	}
	//
	//       ==== Near-the-diagonal bulge chase.  The following loop
	//       .    performs the near-the-diagonal part of a small bulge
	//       .    multi-shift QR sweep.  Each 6*NBMPS-2 column diagonal
	//       .    chunk extends from column INCOL to column NDCOL
	//       .    (including both column INCOL and column NDCOL). The
	//       .    following loop chases a 3*NBMPS column long chain of
	//       .    NBMPS bulges 3*NBMPS-2 columns to the right.  (INCOL
	//       .    may be less than KTOP and and NDCOL may be greater than
	//       .    KBOT indicating phantom columns from which to chase
	//       .    bulges before they are actually introduced or to which
	//       .    to chase bulges beyond column KBOT.)  ====
	//
	// Computing MIN
	i__4 = incol + nbmps * 3 - 3, i__5 = *kbot - 2;
	i__3 = min(i__4,i__5);
	for (krcol = incol; krcol <= i__3; ++krcol) {
	    //
	    //          ==== Bulges number MTOP to MBOT are active double implicit
	    //          .    shift bulges.  There may or may not also be small
	    //          .    2-by-2 bulge, if there is room.  The inactive bulges
	    //          .    (if any) must wait until the active bulges have moved
	    //          .    down the diagonal to make room.  The phantom matrix
	    //          .    paradigm described above helps keep track.  ====
	    //
	    // Computing MAX
	    i__4 = 1, i__5 = (*ktop - 1 - krcol + 2) / 3 + 1;
	    mtop = max(i__4,i__5);
	    // Computing MIN
	    i__4 = nbmps, i__5 = (*kbot - krcol) / 3;
	    mbot = min(i__4,i__5);
	    m22 = mbot + 1;
	    bmp22 = mbot < nbmps && krcol + (m22 - 1) * 3 == *kbot - 2;
	    //
	    //          ==== Generate reflections to chase the chain right
	    //          .    one column.  (The minimum value of K is KTOP-1.) ====
	    //
	    i__4 = mbot;
	    for (m = mtop; m <= i__4; ++m) {
		k = krcol + (m - 1) * 3;
		if (k == *ktop - 1) {
		    dlaqr1_(&c__3, &h__[*ktop + *ktop * h_dim1], ldh, &sr[(m
			    << 1) - 1], &si[(m << 1) - 1], &sr[m * 2], &si[m *
			     2], &v[m * v_dim1 + 1]);
		    alpha = v[m * v_dim1 + 1];
		    dlarfg_(&c__3, &alpha, &v[m * v_dim1 + 2], &c__1, &v[m *
			    v_dim1 + 1]);
		} else {
		    beta = h__[k + 1 + k * h_dim1];
		    v[m * v_dim1 + 2] = h__[k + 2 + k * h_dim1];
		    v[m * v_dim1 + 3] = h__[k + 3 + k * h_dim1];
		    dlarfg_(&c__3, &beta, &v[m * v_dim1 + 2], &c__1, &v[m *
			    v_dim1 + 1]);
		    //
		    //                ==== A Bulge may collapse because of vigilant
		    //                .    deflation or destructive underflow.  In the
		    //                .    underflow case, try the two-small-subdiagonals
		    //                .    trick to try to reinflate the bulge.  ====
		    //
		    if (h__[k + 3 + k * h_dim1] != 0. || h__[k + 3 + (k + 1) *
			     h_dim1] != 0. || h__[k + 3 + (k + 2) * h_dim1] ==
			     0.) {
			//
			//                   ==== Typical case: not collapsed (yet). ====
			//
			h__[k + 1 + k * h_dim1] = beta;
			h__[k + 2 + k * h_dim1] = 0.;
			h__[k + 3 + k * h_dim1] = 0.;
		    } else {
			//
			//                   ==== Atypical case: collapsed.  Attempt to
			//                   .    reintroduce ignoring H(K+1,K) and H(K+2,K).
			//                   .    If the fill resulting from the new
			//                   .    reflector is too large, then abandon it.
			//                   .    Otherwise, use the new one. ====
			//
			dlaqr1_(&c__3, &h__[k + 1 + (k + 1) * h_dim1], ldh, &
				sr[(m << 1) - 1], &si[(m << 1) - 1], &sr[m *
				2], &si[m * 2], vt);
			alpha = vt[0];
			dlarfg_(&c__3, &alpha, &vt[1], &c__1, vt);
			refsum = vt[0] * (h__[k + 1 + k * h_dim1] + vt[1] *
				h__[k + 2 + k * h_dim1]);
			if ((d__1 = h__[k + 2 + k * h_dim1] - refsum * vt[1],
				abs(d__1)) + (d__2 = refsum * vt[2], abs(d__2)
				) > ulp * ((d__3 = h__[k + k * h_dim1], abs(
				d__3)) + (d__4 = h__[k + 1 + (k + 1) * h_dim1]
				, abs(d__4)) + (d__5 = h__[k + 2 + (k + 2) *
				h_dim1], abs(d__5)))) {
			    //
			    //                      ==== Starting a new bulge here would
			    //                      .    create non-negligible fill.  Use
			    //                      .    the old one with trepidation. ====
			    //
			    h__[k + 1 + k * h_dim1] = beta;
			    h__[k + 2 + k * h_dim1] = 0.;
			    h__[k + 3 + k * h_dim1] = 0.;
			} else {
			    //
			    //                      ==== Stating a new bulge here would
			    //                      .    create only negligible fill.
			    //                      .    Replace the old reflector with
			    //                      .    the new one. ====
			    //
			    h__[k + 1 + k * h_dim1] -= refsum;
			    h__[k + 2 + k * h_dim1] = 0.;
			    h__[k + 3 + k * h_dim1] = 0.;
			    v[m * v_dim1 + 1] = vt[0];
			    v[m * v_dim1 + 2] = vt[1];
			    v[m * v_dim1 + 3] = vt[2];
			}
		    }
		}
// L20:
	    }
	    //
	    //          ==== Generate a 2-by-2 reflection, if needed. ====
	    //
	    k = krcol + (m22 - 1) * 3;
	    if (bmp22) {
		if (k == *ktop - 1) {
		    dlaqr1_(&c__2, &h__[k + 1 + (k + 1) * h_dim1], ldh, &sr[(
			    m22 << 1) - 1], &si[(m22 << 1) - 1], &sr[m22 * 2],
			     &si[m22 * 2], &v[m22 * v_dim1 + 1]);
		    beta = v[m22 * v_dim1 + 1];
		    dlarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22
			    * v_dim1 + 1]);
		} else {
		    beta = h__[k + 1 + k * h_dim1];
		    v[m22 * v_dim1 + 2] = h__[k + 2 + k * h_dim1];
		    dlarfg_(&c__2, &beta, &v[m22 * v_dim1 + 2], &c__1, &v[m22
			    * v_dim1 + 1]);
		    h__[k + 1 + k * h_dim1] = beta;
		    h__[k + 2 + k * h_dim1] = 0.;
		}
	    }
	    //
	    //          ==== Multiply H by reflections from the left ====
	    //
	    if (accum) {
		jbot = min(ndcol,*kbot);
	    } else if (*wantt) {
		jbot = *n;
	    } else {
		jbot = *kbot;
	    }
	    i__4 = jbot;
	    for (j = max(*ktop,krcol); j <= i__4; ++j) {
		// Computing MIN
		i__5 = mbot, i__6 = (j - krcol + 2) / 3;
		mend = min(i__5,i__6);
		i__5 = mend;
		for (m = mtop; m <= i__5; ++m) {
		    k = krcol + (m - 1) * 3;
		    refsum = v[m * v_dim1 + 1] * (h__[k + 1 + j * h_dim1] + v[
			    m * v_dim1 + 2] * h__[k + 2 + j * h_dim1] + v[m *
			    v_dim1 + 3] * h__[k + 3 + j * h_dim1]);
		    h__[k + 1 + j * h_dim1] -= refsum;
		    h__[k + 2 + j * h_dim1] -= refsum * v[m * v_dim1 + 2];
		    h__[k + 3 + j * h_dim1] -= refsum * v[m * v_dim1 + 3];
// L30:
		}
// L40:
	    }
	    if (bmp22) {
		k = krcol + (m22 - 1) * 3;
		// Computing MAX
		i__4 = k + 1;
		i__5 = jbot;
		for (j = max(i__4,*ktop); j <= i__5; ++j) {
		    refsum = v[m22 * v_dim1 + 1] * (h__[k + 1 + j * h_dim1] +
			    v[m22 * v_dim1 + 2] * h__[k + 2 + j * h_dim1]);
		    h__[k + 1 + j * h_dim1] -= refsum;
		    h__[k + 2 + j * h_dim1] -= refsum * v[m22 * v_dim1 + 2];
// L50:
		}
	    }
	    //
	    //          ==== Multiply H by reflections from the right.
	    //          .    Delay filling in the last row until the
	    //          .    vigilant deflation check is complete. ====
	    //
	    if (accum) {
		jtop = max(*ktop,incol);
	    } else if (*wantt) {
		jtop = 1;
	    } else {
		jtop = *ktop;
	    }
	    i__5 = mbot;
	    for (m = mtop; m <= i__5; ++m) {
		if (v[m * v_dim1 + 1] != 0.) {
		    k = krcol + (m - 1) * 3;
		    // Computing MIN
		    i__6 = *kbot, i__7 = k + 3;
		    i__4 = min(i__6,i__7);
		    for (j = jtop; j <= i__4; ++j) {
			refsum = v[m * v_dim1 + 1] * (h__[j + (k + 1) *
				h_dim1] + v[m * v_dim1 + 2] * h__[j + (k + 2)
				* h_dim1] + v[m * v_dim1 + 3] * h__[j + (k +
				3) * h_dim1]);
			h__[j + (k + 1) * h_dim1] -= refsum;
			h__[j + (k + 2) * h_dim1] -= refsum * v[m * v_dim1 +
				2];
			h__[j + (k + 3) * h_dim1] -= refsum * v[m * v_dim1 +
				3];
// L60:
		    }
		    if (accum) {
			//
			//                   ==== Accumulate U. (If necessary, update Z later
			//                   .    with with an efficient matrix-matrix
			//                   .    multiply.) ====
			//
			kms = k - incol;
			// Computing MAX
			i__4 = 1, i__6 = *ktop - incol;
			i__7 = kdu;
			for (j = max(i__4,i__6); j <= i__7; ++j) {
			    refsum = v[m * v_dim1 + 1] * (u[j + (kms + 1) *
				    u_dim1] + v[m * v_dim1 + 2] * u[j + (kms
				    + 2) * u_dim1] + v[m * v_dim1 + 3] * u[j
				    + (kms + 3) * u_dim1]);
			    u[j + (kms + 1) * u_dim1] -= refsum;
			    u[j + (kms + 2) * u_dim1] -= refsum * v[m *
				    v_dim1 + 2];
			    u[j + (kms + 3) * u_dim1] -= refsum * v[m *
				    v_dim1 + 3];
// L70:
			}
		    } else if (*wantz) {
			//
			//                   ==== U is not accumulated, so update Z
			//                   .    now by multiplying by reflections
			//                   .    from the right. ====
			//
			i__7 = *ihiz;
			for (j = *iloz; j <= i__7; ++j) {
			    refsum = v[m * v_dim1 + 1] * (z__[j + (k + 1) *
				    z_dim1] + v[m * v_dim1 + 2] * z__[j + (k
				    + 2) * z_dim1] + v[m * v_dim1 + 3] * z__[
				    j + (k + 3) * z_dim1]);
			    z__[j + (k + 1) * z_dim1] -= refsum;
			    z__[j + (k + 2) * z_dim1] -= refsum * v[m *
				    v_dim1 + 2];
			    z__[j + (k + 3) * z_dim1] -= refsum * v[m *
				    v_dim1 + 3];
// L80:
			}
		    }
		}
// L90:
	    }
	    //
	    //          ==== Special case: 2-by-2 reflection (if needed) ====
	    //
	    k = krcol + (m22 - 1) * 3;
	    if (bmp22) {
		if (v[m22 * v_dim1 + 1] != 0.) {
		    // Computing MIN
		    i__7 = *kbot, i__4 = k + 3;
		    i__5 = min(i__7,i__4);
		    for (j = jtop; j <= i__5; ++j) {
			refsum = v[m22 * v_dim1 + 1] * (h__[j + (k + 1) *
				h_dim1] + v[m22 * v_dim1 + 2] * h__[j + (k +
				2) * h_dim1]);
			h__[j + (k + 1) * h_dim1] -= refsum;
			h__[j + (k + 2) * h_dim1] -= refsum * v[m22 * v_dim1
				+ 2];
// L100:
		    }
		    if (accum) {
			kms = k - incol;
			// Computing MAX
			i__5 = 1, i__7 = *ktop - incol;
			i__4 = kdu;
			for (j = max(i__5,i__7); j <= i__4; ++j) {
			    refsum = v[m22 * v_dim1 + 1] * (u[j + (kms + 1) *
				    u_dim1] + v[m22 * v_dim1 + 2] * u[j + (
				    kms + 2) * u_dim1]);
			    u[j + (kms + 1) * u_dim1] -= refsum;
			    u[j + (kms + 2) * u_dim1] -= refsum * v[m22 *
				    v_dim1 + 2];
// L110:
			}
		    } else if (*wantz) {
			i__4 = *ihiz;
			for (j = *iloz; j <= i__4; ++j) {
			    refsum = v[m22 * v_dim1 + 1] * (z__[j + (k + 1) *
				    z_dim1] + v[m22 * v_dim1 + 2] * z__[j + (
				    k + 2) * z_dim1]);
			    z__[j + (k + 1) * z_dim1] -= refsum;
			    z__[j + (k + 2) * z_dim1] -= refsum * v[m22 *
				    v_dim1 + 2];
// L120:
			}
		    }
		}
	    }
	    //
	    //          ==== Vigilant deflation check ====
	    //
	    mstart = mtop;
	    if (krcol + (mstart - 1) * 3 < *ktop) {
		++mstart;
	    }
	    mend = mbot;
	    if (bmp22) {
		++mend;
	    }
	    if (krcol == *kbot - 2) {
		++mend;
	    }
	    i__4 = mend;
	    for (m = mstart; m <= i__4; ++m) {
		// Computing MIN
		i__5 = *kbot - 1, i__7 = krcol + (m - 1) * 3;
		k = min(i__5,i__7);
		//
		//             ==== The following convergence test requires that
		//             .    the tradition small-compared-to-nearby-diagonals
		//             .    criterion and the Ahues & Tisseur (LAWN 122, 1997)
		//             .    criteria both be satisfied.  The latter improves
		//             .    accuracy in some examples. Falling back on an
		//             .    alternate convergence criterion when TST1 or TST2
		//             .    is zero (as done here) is traditional but probably
		//             .    unnecessary. ====
		//
		if (h__[k + 1 + k * h_dim1] != 0.) {
		    tst1 = (d__1 = h__[k + k * h_dim1], abs(d__1)) + (d__2 =
			    h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
		    if (tst1 == 0.) {
			if (k >= *ktop + 1) {
			    tst1 += (d__1 = h__[k + (k - 1) * h_dim1], abs(
				    d__1));
			}
			if (k >= *ktop + 2) {
			    tst1 += (d__1 = h__[k + (k - 2) * h_dim1], abs(
				    d__1));
			}
			if (k >= *ktop + 3) {
			    tst1 += (d__1 = h__[k + (k - 3) * h_dim1], abs(
				    d__1));
			}
			if (k <= *kbot - 2) {
			    tst1 += (d__1 = h__[k + 2 + (k + 1) * h_dim1],
				    abs(d__1));
			}
			if (k <= *kbot - 3) {
			    tst1 += (d__1 = h__[k + 3 + (k + 1) * h_dim1],
				    abs(d__1));
			}
			if (k <= *kbot - 4) {
			    tst1 += (d__1 = h__[k + 4 + (k + 1) * h_dim1],
				    abs(d__1));
			}
		    }
		    // Computing MAX
		    d__2 = smlnum, d__3 = ulp * tst1;
		    if ((d__1 = h__[k + 1 + k * h_dim1], abs(d__1)) <= max(
			    d__2,d__3)) {
			// Computing MAX
			d__3 = (d__1 = h__[k + 1 + k * h_dim1], abs(d__1)),
				d__4 = (d__2 = h__[k + (k + 1) * h_dim1], abs(
				d__2));
			h12 = max(d__3,d__4);
			// Computing MIN
			d__3 = (d__1 = h__[k + 1 + k * h_dim1], abs(d__1)),
				d__4 = (d__2 = h__[k + (k + 1) * h_dim1], abs(
				d__2));
			h21 = min(d__3,d__4);
			// Computing MAX
			d__3 = (d__1 = h__[k + 1 + (k + 1) * h_dim1], abs(
				d__1)), d__4 = (d__2 = h__[k + k * h_dim1] -
				h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
			h11 = max(d__3,d__4);
			// Computing MIN
			d__3 = (d__1 = h__[k + 1 + (k + 1) * h_dim1], abs(
				d__1)), d__4 = (d__2 = h__[k + k * h_dim1] -
				h__[k + 1 + (k + 1) * h_dim1], abs(d__2));
			h22 = min(d__3,d__4);
			scl = h11 + h12;
			tst2 = h22 * (h11 / scl);
			//
			// Computing MAX
			d__1 = smlnum, d__2 = ulp * tst2;
			if (tst2 == 0. || h21 * (h12 / scl) <= max(d__1,d__2))
				 {
			    h__[k + 1 + k * h_dim1] = 0.;
			}
		    }
		}
// L130:
	    }
	    //
	    //          ==== Fill in the last row of each bulge. ====
	    //
	    // Computing MIN
	    i__4 = nbmps, i__5 = (*kbot - krcol - 1) / 3;
	    mend = min(i__4,i__5);
	    i__4 = mend;
	    for (m = mtop; m <= i__4; ++m) {
		k = krcol + (m - 1) * 3;
		refsum = v[m * v_dim1 + 1] * v[m * v_dim1 + 3] * h__[k + 4 + (
			k + 3) * h_dim1];
		h__[k + 4 + (k + 1) * h_dim1] = -refsum;
		h__[k + 4 + (k + 2) * h_dim1] = -refsum * v[m * v_dim1 + 2];
		h__[k + 4 + (k + 3) * h_dim1] -= refsum * v[m * v_dim1 + 3];
// L140:
	    }
	    //
	    //          ==== End of near-the-diagonal bulge chase. ====
	    //
// L150:
	}
	//
	//       ==== Use U (if accumulated) to update far-from-diagonal
	//       .    entries in H.  If required, use U to update Z as
	//       .    well. ====
	//
	if (accum) {
	    if (*wantt) {
		jtop = 1;
		jbot = *n;
	    } else {
		jtop = *ktop;
		jbot = *kbot;
	    }
	    if (! blk22 || incol < *ktop || ndcol > *kbot || ns <= 2) {
		//
		//             ==== Updates not exploiting the 2-by-2 block
		//             .    structure of U.  K1 and NU keep track of
		//             .    the location and size of U in the special
		//             .    cases of introducing bulges and chasing
		//             .    bulges off the bottom.  In these special
		//             .    cases and in case the number of shifts
		//             .    is NS = 2, there is no 2-by-2 block
		//             .    structure to exploit.  ====
		//
		// Computing MAX
		i__3 = 1, i__4 = *ktop - incol;
		k1 = max(i__3,i__4);
		// Computing MAX
		i__3 = 0, i__4 = ndcol - *kbot;
		nu = kdu - max(i__3,i__4) - k1 + 1;
		//
		//             ==== Horizontal Multiply ====
		//
		i__3 = jbot;
		i__4 = *nh;
		for (jcol = min(ndcol,*kbot) + 1; i__4 < 0 ? jcol >= i__3 :
			jcol <= i__3; jcol += i__4) {
		    // Computing MIN
		    i__5 = *nh, i__7 = jbot - jcol + 1;
		    jlen = min(i__5,i__7);
		    dgemm_("C", "N", &nu, &jlen, &nu, &c_b8, &u[k1 + k1 *
			    u_dim1], ldu, &h__[incol + k1 + jcol * h_dim1],
			    ldh, &c_b7, &wh[wh_offset], ldwh);
		    dlacpy_("ALL", &nu, &jlen, &wh[wh_offset], ldwh, &h__[
			    incol + k1 + jcol * h_dim1], ldh);
// L160:
		}
		//
		//             ==== Vertical multiply ====
		//
		i__4 = max(*ktop,incol) - 1;
		i__3 = *nv;
		for (jrow = jtop; i__3 < 0 ? jrow >= i__4 : jrow <= i__4;
			jrow += i__3) {
		    // Computing MIN
		    i__5 = *nv, i__7 = max(*ktop,incol) - jrow;
		    jlen = min(i__5,i__7);
		    dgemm_("N", "N", &jlen, &nu, &nu, &c_b8, &h__[jrow + (
			    incol + k1) * h_dim1], ldh, &u[k1 + k1 * u_dim1],
			    ldu, &c_b7, &wv[wv_offset], ldwv);
		    dlacpy_("ALL", &jlen, &nu, &wv[wv_offset], ldwv, &h__[
			    jrow + (incol + k1) * h_dim1], ldh);
// L170:
		}
		//
		//             ==== Z multiply (also vertical) ====
		//
		if (*wantz) {
		    i__3 = *ihiz;
		    i__4 = *nv;
		    for (jrow = *iloz; i__4 < 0 ? jrow >= i__3 : jrow <= i__3;
			     jrow += i__4) {
			// Computing MIN
			i__5 = *nv, i__7 = *ihiz - jrow + 1;
			jlen = min(i__5,i__7);
			dgemm_("N", "N", &jlen, &nu, &nu, &c_b8, &z__[jrow + (
				incol + k1) * z_dim1], ldz, &u[k1 + k1 *
				u_dim1], ldu, &c_b7, &wv[wv_offset], ldwv);
			dlacpy_("ALL", &jlen, &nu, &wv[wv_offset], ldwv, &z__[
				jrow + (incol + k1) * z_dim1], ldz);
// L180:
		    }
		}
	    } else {
		//
		//             ==== Updates exploiting U's 2-by-2 block structure.
		//             .    (I2, I4, J2, J4 are the last rows and columns
		//             .    of the blocks.) ====
		//
		i2 = (kdu + 1) / 2;
		i4 = kdu;
		j2 = i4 - i2;
		j4 = kdu;
		//
		//             ==== KZS and KNZ deal with the band of zeros
		//             .    along the diagonal of one of the triangular
		//             .    blocks. ====
		//
		kzs = j4 - j2 - (ns + 1);
		knz = ns + 1;
		//
		//             ==== Horizontal multiply ====
		//
		i__4 = jbot;
		i__3 = *nh;
		for (jcol = min(ndcol,*kbot) + 1; i__3 < 0 ? jcol >= i__4 :
			jcol <= i__4; jcol += i__3) {
		    // Computing MIN
		    i__5 = *nh, i__7 = jbot - jcol + 1;
		    jlen = min(i__5,i__7);
		    //
		    //                ==== Copy bottom of H to top+KZS of scratch ====
		    //                 (The first KZS rows get multiplied by zero.) ====
		    //
		    dlacpy_("ALL", &knz, &jlen, &h__[incol + 1 + j2 + jcol *
			    h_dim1], ldh, &wh[kzs + 1 + wh_dim1], ldwh);
		    //
		    //                ==== Multiply by U21**T ====
		    //
		    dlaset_("ALL", &kzs, &jlen, &c_b7, &c_b7, &wh[wh_offset],
			    ldwh);
		    dtrmm_("L", "U", "C", "N", &knz, &jlen, &c_b8, &u[j2 + 1
			    + (kzs + 1) * u_dim1], ldu, &wh[kzs + 1 + wh_dim1]
			    , ldwh);
		    //
		    //                ==== Multiply top of H by U11**T ====
		    //
		    dgemm_("C", "N", &i2, &jlen, &j2, &c_b8, &u[u_offset],
			    ldu, &h__[incol + 1 + jcol * h_dim1], ldh, &c_b8,
			    &wh[wh_offset], ldwh);
		    //
		    //                ==== Copy top of H to bottom of WH ====
		    //
		    dlacpy_("ALL", &j2, &jlen, &h__[incol + 1 + jcol * h_dim1]
			    , ldh, &wh[i2 + 1 + wh_dim1], ldwh);
		    //
		    //                ==== Multiply by U21**T ====
		    //
		    dtrmm_("L", "L", "C", "N", &j2, &jlen, &c_b8, &u[(i2 + 1)
			    * u_dim1 + 1], ldu, &wh[i2 + 1 + wh_dim1], ldwh);
		    //
		    //                ==== Multiply by U22 ====
		    //
		    i__5 = i4 - i2;
		    i__7 = j4 - j2;
		    dgemm_("C", "N", &i__5, &jlen, &i__7, &c_b8, &u[j2 + 1 + (
			    i2 + 1) * u_dim1], ldu, &h__[incol + 1 + j2 +
			    jcol * h_dim1], ldh, &c_b8, &wh[i2 + 1 + wh_dim1],
			     ldwh);
		    //
		    //                ==== Copy it back ====
		    //
		    dlacpy_("ALL", &kdu, &jlen, &wh[wh_offset], ldwh, &h__[
			    incol + 1 + jcol * h_dim1], ldh);
// L190:
		}
		//
		//             ==== Vertical multiply ====
		//
		i__3 = max(incol,*ktop) - 1;
		i__4 = *nv;
		for (jrow = jtop; i__4 < 0 ? jrow >= i__3 : jrow <= i__3;
			jrow += i__4) {
		    // Computing MIN
		    i__5 = *nv, i__7 = max(incol,*ktop) - jrow;
		    jlen = min(i__5,i__7);
		    //
		    //                ==== Copy right of H to scratch (the first KZS
		    //                .    columns get multiplied by zero) ====
		    //
		    dlacpy_("ALL", &jlen, &knz, &h__[jrow + (incol + 1 + j2) *
			     h_dim1], ldh, &wv[(kzs + 1) * wv_dim1 + 1], ldwv)
			    ;
		    //
		    //                ==== Multiply by U21 ====
		    //
		    dlaset_("ALL", &jlen, &kzs, &c_b7, &c_b7, &wv[wv_offset],
			    ldwv);
		    dtrmm_("R", "U", "N", "N", &jlen, &knz, &c_b8, &u[j2 + 1
			    + (kzs + 1) * u_dim1], ldu, &wv[(kzs + 1) *
			    wv_dim1 + 1], ldwv);
		    //
		    //                ==== Multiply by U11 ====
		    //
		    dgemm_("N", "N", &jlen, &i2, &j2, &c_b8, &h__[jrow + (
			    incol + 1) * h_dim1], ldh, &u[u_offset], ldu, &
			    c_b8, &wv[wv_offset], ldwv);
		    //
		    //                ==== Copy left of H to right of scratch ====
		    //
		    dlacpy_("ALL", &jlen, &j2, &h__[jrow + (incol + 1) *
			    h_dim1], ldh, &wv[(i2 + 1) * wv_dim1 + 1], ldwv);
		    //
		    //                ==== Multiply by U21 ====
		    //
		    i__5 = i4 - i2;
		    dtrmm_("R", "L", "N", "N", &jlen, &i__5, &c_b8, &u[(i2 +
			    1) * u_dim1 + 1], ldu, &wv[(i2 + 1) * wv_dim1 + 1]
			    , ldwv);
		    //
		    //                ==== Multiply by U22 ====
		    //
		    i__5 = i4 - i2;
		    i__7 = j4 - j2;
		    dgemm_("N", "N", &jlen, &i__5, &i__7, &c_b8, &h__[jrow + (
			    incol + 1 + j2) * h_dim1], ldh, &u[j2 + 1 + (i2 +
			    1) * u_dim1], ldu, &c_b8, &wv[(i2 + 1) * wv_dim1
			    + 1], ldwv);
		    //
		    //                ==== Copy it back ====
		    //
		    dlacpy_("ALL", &jlen, &kdu, &wv[wv_offset], ldwv, &h__[
			    jrow + (incol + 1) * h_dim1], ldh);
// L200:
		}
		//
		//             ==== Multiply Z (also vertical) ====
		//
		if (*wantz) {
		    i__4 = *ihiz;
		    i__3 = *nv;
		    for (jrow = *iloz; i__3 < 0 ? jrow >= i__4 : jrow <= i__4;
			     jrow += i__3) {
			// Computing MIN
			i__5 = *nv, i__7 = *ihiz - jrow + 1;
			jlen = min(i__5,i__7);
			//
			//                   ==== Copy right of Z to left of scratch (first
			//                   .     KZS columns get multiplied by zero) ====
			//
			dlacpy_("ALL", &jlen, &knz, &z__[jrow + (incol + 1 +
				j2) * z_dim1], ldz, &wv[(kzs + 1) * wv_dim1 +
				1], ldwv);
			//
			//                   ==== Multiply by U12 ====
			//
			dlaset_("ALL", &jlen, &kzs, &c_b7, &c_b7, &wv[
				wv_offset], ldwv);
			dtrmm_("R", "U", "N", "N", &jlen, &knz, &c_b8, &u[j2
				+ 1 + (kzs + 1) * u_dim1], ldu, &wv[(kzs + 1)
				* wv_dim1 + 1], ldwv);
			//
			//                   ==== Multiply by U11 ====
			//
			dgemm_("N", "N", &jlen, &i2, &j2, &c_b8, &z__[jrow + (
				incol + 1) * z_dim1], ldz, &u[u_offset], ldu,
				&c_b8, &wv[wv_offset], ldwv);
			//
			//                   ==== Copy left of Z to right of scratch ====
			//
			dlacpy_("ALL", &jlen, &j2, &z__[jrow + (incol + 1) *
				z_dim1], ldz, &wv[(i2 + 1) * wv_dim1 + 1],
				ldwv);
			//
			//                   ==== Multiply by U21 ====
			//
			i__5 = i4 - i2;
			dtrmm_("R", "L", "N", "N", &jlen, &i__5, &c_b8, &u[(
				i2 + 1) * u_dim1 + 1], ldu, &wv[(i2 + 1) *
				wv_dim1 + 1], ldwv);
			//
			//                   ==== Multiply by U22 ====
			//
			i__5 = i4 - i2;
			i__7 = j4 - j2;
			dgemm_("N", "N", &jlen, &i__5, &i__7, &c_b8, &z__[
				jrow + (incol + 1 + j2) * z_dim1], ldz, &u[j2
				+ 1 + (i2 + 1) * u_dim1], ldu, &c_b8, &wv[(i2
				+ 1) * wv_dim1 + 1], ldwv);
			//
			//                   ==== Copy the result back to Z ====
			//
			dlacpy_("ALL", &jlen, &kdu, &wv[wv_offset], ldwv, &
				z__[jrow + (incol + 1) * z_dim1], ldz);
// L210:
		    }
		}
	    }
	}
// L220:
    }
    //
    //    ==== End of DLAQR5 ====
    //
    return 0;
} // dlaqr5_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARFX applies an elementary reflector to a general rectangular matrix, with loop unrolling when the reflector has order ≤ 10.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARFX + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarfx.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarfx.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarfx.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARFX( SIDE, M, N, V, TAU, C, LDC, WORK )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE
//      INTEGER            LDC, M, N
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
//> DLARFX applies a real elementary reflector H to a real m by n
//> matrix C, from either the left or the right. H is represented in the
//> form
//>
//>       H = I - tau * v * v**T
//>
//> where tau is a real scalar and v is a real vector.
//>
//> If tau = 0, then H is taken to be the unit matrix
//>
//> This version uses inline code if H has order < 11.
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
//>          V is DOUBLE PRECISION array, dimension (M) if SIDE = 'L'
//>                                     or (N) if SIDE = 'R'
//>          The vector v in the representation of H.
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
//>          The leading dimension of the array C. LDC >= (1,M).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension
//>                      (N) if SIDE = 'L'
//>                      or (M) if SIDE = 'R'
//>          WORK is not referenced if H has order < 11.
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
/* Subroutine */ int dlarfx_(char *side, int *m, int *n, double *v, double *
	tau, double *c__, int *ldc, double *work)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int c_dim1, c_offset, i__1;

    // Local variables
    int j;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, v1, v2, v3, v4, v5, v6, v7, v8,
	     v9, t10, v10, sum;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *,
	    double *, double *, int *, double *);
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
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    if (*tau == 0.) {
	return 0;
    }
    if (lsame_(side, "L")) {
	//
	//       Form  H * C, where H has order m.
	//
	switch (*m) {
	    case 1:  goto L10;
	    case 2:  goto L30;
	    case 3:  goto L50;
	    case 4:  goto L70;
	    case 5:  goto L90;
	    case 6:  goto L110;
	    case 7:  goto L130;
	    case 8:  goto L150;
	    case 9:  goto L170;
	    case 10:  goto L190;
	}
	//
	//       Code for general M
	//
	dlarf_(side, m, n, &v[1], &c__1, tau, &c__[c_offset], ldc, &work[1]);
	goto L410;
L10:
	//
	//       Special code for 1 x 1 Householder
	//
	t1 = 1. - *tau * v[1] * v[1];
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    c__[j * c_dim1 + 1] = t1 * c__[j * c_dim1 + 1];
// L20:
	}
	goto L410;
L30:
	//
	//       Special code for 2 x 2 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
// L40:
	}
	goto L410;
L50:
	//
	//       Special code for 3 x 3 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
// L60:
	}
	goto L410;
L70:
	//
	//       Special code for 4 x 4 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
// L80:
	}
	goto L410;
L90:
	//
	//       Special code for 5 x 5 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
// L100:
	}
	goto L410;
L110:
	//
	//       Special code for 6 x 6 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
// L120:
	}
	goto L410;
L130:
	//
	//       Special code for 7 x 7 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
// L140:
	}
	goto L410;
L150:
	//
	//       Special code for 8 x 8 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7] + v8 * c__[j * c_dim1 + 8];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
	    c__[j * c_dim1 + 8] -= sum * t8;
// L160:
	}
	goto L410;
L170:
	//
	//       Special code for 9 x 9 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7] + v8 * c__[j * c_dim1 + 8] + v9 * c__[j *
		    c_dim1 + 9];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
	    c__[j * c_dim1 + 8] -= sum * t8;
	    c__[j * c_dim1 + 9] -= sum * t9;
// L180:
	}
	goto L410;
L190:
	//
	//       Special code for 10 x 10 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	v10 = v[10];
	t10 = *tau * v10;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j * c_dim1 + 1] + v2 * c__[j * c_dim1 + 2] + v3 *
		    c__[j * c_dim1 + 3] + v4 * c__[j * c_dim1 + 4] + v5 * c__[
		    j * c_dim1 + 5] + v6 * c__[j * c_dim1 + 6] + v7 * c__[j *
		    c_dim1 + 7] + v8 * c__[j * c_dim1 + 8] + v9 * c__[j *
		    c_dim1 + 9] + v10 * c__[j * c_dim1 + 10];
	    c__[j * c_dim1 + 1] -= sum * t1;
	    c__[j * c_dim1 + 2] -= sum * t2;
	    c__[j * c_dim1 + 3] -= sum * t3;
	    c__[j * c_dim1 + 4] -= sum * t4;
	    c__[j * c_dim1 + 5] -= sum * t5;
	    c__[j * c_dim1 + 6] -= sum * t6;
	    c__[j * c_dim1 + 7] -= sum * t7;
	    c__[j * c_dim1 + 8] -= sum * t8;
	    c__[j * c_dim1 + 9] -= sum * t9;
	    c__[j * c_dim1 + 10] -= sum * t10;
// L200:
	}
	goto L410;
    } else {
	//
	//       Form  C * H, where H has order n.
	//
	switch (*n) {
	    case 1:  goto L210;
	    case 2:  goto L230;
	    case 3:  goto L250;
	    case 4:  goto L270;
	    case 5:  goto L290;
	    case 6:  goto L310;
	    case 7:  goto L330;
	    case 8:  goto L350;
	    case 9:  goto L370;
	    case 10:  goto L390;
	}
	//
	//       Code for general N
	//
	dlarf_(side, m, n, &v[1], &c__1, tau, &c__[c_offset], ldc, &work[1]);
	goto L410;
L210:
	//
	//       Special code for 1 x 1 Householder
	//
	t1 = 1. - *tau * v[1] * v[1];
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    c__[j + c_dim1] = t1 * c__[j + c_dim1];
// L220:
	}
	goto L410;
L230:
	//
	//       Special code for 2 x 2 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
// L240:
	}
	goto L410;
L250:
	//
	//       Special code for 3 x 3 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
// L260:
	}
	goto L410;
L270:
	//
	//       Special code for 4 x 4 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
// L280:
	}
	goto L410;
L290:
	//
	//       Special code for 5 x 5 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
// L300:
	}
	goto L410;
L310:
	//
	//       Special code for 6 x 6 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
// L320:
	}
	goto L410;
L330:
	//
	//       Special code for 7 x 7 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
// L340:
	}
	goto L410;
L350:
	//
	//       Special code for 8 x 8 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7] + v8 * c__[j + (c_dim1 << 3)];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
	    c__[j + (c_dim1 << 3)] -= sum * t8;
// L360:
	}
	goto L410;
L370:
	//
	//       Special code for 9 x 9 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7] + v8 * c__[j + (c_dim1 << 3)] + v9 * c__[
		    j + c_dim1 * 9];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
	    c__[j + (c_dim1 << 3)] -= sum * t8;
	    c__[j + c_dim1 * 9] -= sum * t9;
// L380:
	}
	goto L410;
L390:
	//
	//       Special code for 10 x 10 Householder
	//
	v1 = v[1];
	t1 = *tau * v1;
	v2 = v[2];
	t2 = *tau * v2;
	v3 = v[3];
	t3 = *tau * v3;
	v4 = v[4];
	t4 = *tau * v4;
	v5 = v[5];
	t5 = *tau * v5;
	v6 = v[6];
	t6 = *tau * v6;
	v7 = v[7];
	t7 = *tau * v7;
	v8 = v[8];
	t8 = *tau * v8;
	v9 = v[9];
	t9 = *tau * v9;
	v10 = v[10];
	t10 = *tau * v10;
	i__1 = *m;
	for (j = 1; j <= i__1; ++j) {
	    sum = v1 * c__[j + c_dim1] + v2 * c__[j + (c_dim1 << 1)] + v3 *
		    c__[j + c_dim1 * 3] + v4 * c__[j + (c_dim1 << 2)] + v5 *
		    c__[j + c_dim1 * 5] + v6 * c__[j + c_dim1 * 6] + v7 * c__[
		    j + c_dim1 * 7] + v8 * c__[j + (c_dim1 << 3)] + v9 * c__[
		    j + c_dim1 * 9] + v10 * c__[j + c_dim1 * 10];
	    c__[j + c_dim1] -= sum * t1;
	    c__[j + (c_dim1 << 1)] -= sum * t2;
	    c__[j + c_dim1 * 3] -= sum * t3;
	    c__[j + (c_dim1 << 2)] -= sum * t4;
	    c__[j + c_dim1 * 5] -= sum * t5;
	    c__[j + c_dim1 * 6] -= sum * t6;
	    c__[j + c_dim1 * 7] -= sum * t7;
	    c__[j + (c_dim1 << 3)] -= sum * t8;
	    c__[j + c_dim1 * 9] -= sum * t9;
	    c__[j + c_dim1 * 10] -= sum * t10;
// L400:
	}
	goto L410;
    }
L410:
    return 0;
    //
    //    End of DLARFX
    //
} // dlarfx_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASY2 solves the Sylvester matrix equation where the matrices are of order 1 or 2.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASY2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasy2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasy2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasy2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASY2( LTRANL, LTRANR, ISGN, N1, N2, TL, LDTL, TR,
//                         LDTR, B, LDB, SCALE, X, LDX, XNORM, INFO )
//
//      .. Scalar Arguments ..
//      LOGICAL            LTRANL, LTRANR
//      INTEGER            INFO, ISGN, LDB, LDTL, LDTR, LDX, N1, N2
//      DOUBLE PRECISION   SCALE, XNORM
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   B( LDB, * ), TL( LDTL, * ), TR( LDTR, * ),
//     $                   X( LDX, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASY2 solves for the N1 by N2 matrix X, 1 <= N1,N2 <= 2, in
//>
//>        op(TL)*X + ISGN*X*op(TR) = SCALE*B,
//>
//> where TL is N1 by N1, TR is N2 by N2, B is N1 by N2, and ISGN = 1 or
//> -1.  op(T) = T or T**T, where T**T denotes the transpose of T.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] LTRANL
//> \verbatim
//>          LTRANL is LOGICAL
//>          On entry, LTRANL specifies the op(TL):
//>             = .FALSE., op(TL) = TL,
//>             = .TRUE., op(TL) = TL**T.
//> \endverbatim
//>
//> \param[in] LTRANR
//> \verbatim
//>          LTRANR is LOGICAL
//>          On entry, LTRANR specifies the op(TR):
//>            = .FALSE., op(TR) = TR,
//>            = .TRUE., op(TR) = TR**T.
//> \endverbatim
//>
//> \param[in] ISGN
//> \verbatim
//>          ISGN is INTEGER
//>          On entry, ISGN specifies the sign of the equation
//>          as described before. ISGN may only be 1 or -1.
//> \endverbatim
//>
//> \param[in] N1
//> \verbatim
//>          N1 is INTEGER
//>          On entry, N1 specifies the order of matrix TL.
//>          N1 may only be 0, 1 or 2.
//> \endverbatim
//>
//> \param[in] N2
//> \verbatim
//>          N2 is INTEGER
//>          On entry, N2 specifies the order of matrix TR.
//>          N2 may only be 0, 1 or 2.
//> \endverbatim
//>
//> \param[in] TL
//> \verbatim
//>          TL is DOUBLE PRECISION array, dimension (LDTL,2)
//>          On entry, TL contains an N1 by N1 matrix.
//> \endverbatim
//>
//> \param[in] LDTL
//> \verbatim
//>          LDTL is INTEGER
//>          The leading dimension of the matrix TL. LDTL >= max(1,N1).
//> \endverbatim
//>
//> \param[in] TR
//> \verbatim
//>          TR is DOUBLE PRECISION array, dimension (LDTR,2)
//>          On entry, TR contains an N2 by N2 matrix.
//> \endverbatim
//>
//> \param[in] LDTR
//> \verbatim
//>          LDTR is INTEGER
//>          The leading dimension of the matrix TR. LDTR >= max(1,N2).
//> \endverbatim
//>
//> \param[in] B
//> \verbatim
//>          B is DOUBLE PRECISION array, dimension (LDB,2)
//>          On entry, the N1 by N2 matrix B contains the right-hand
//>          side of the equation.
//> \endverbatim
//>
//> \param[in] LDB
//> \verbatim
//>          LDB is INTEGER
//>          The leading dimension of the matrix B. LDB >= max(1,N1).
//> \endverbatim
//>
//> \param[out] SCALE
//> \verbatim
//>          SCALE is DOUBLE PRECISION
//>          On exit, SCALE contains the scale factor. SCALE is chosen
//>          less than or equal to 1 to prevent the solution overflowing.
//> \endverbatim
//>
//> \param[out] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension (LDX,2)
//>          On exit, X contains the N1 by N2 solution.
//> \endverbatim
//>
//> \param[in] LDX
//> \verbatim
//>          LDX is INTEGER
//>          The leading dimension of the matrix X. LDX >= max(1,N1).
//> \endverbatim
//>
//> \param[out] XNORM
//> \verbatim
//>          XNORM is DOUBLE PRECISION
//>          On exit, XNORM is the infinity-norm of the solution.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          On exit, INFO is set to
//>             0: successful exit.
//>             1: TL and TR have too close eigenvalues, so TL or
//>                TR is perturbed to get a nonsingular equation.
//>          NOTE: In the interests of speed, this routine does not
//>                check the inputs for errors.
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
//> \ingroup doubleSYauxiliary
//
// =====================================================================
/* Subroutine */ int dlasy2_(int *ltranl, int *ltranr, int *isgn, int *n1,
	int *n2, double *tl, int *ldtl, double *tr, int *ldtr, double *b, int
	*ldb, double *scale, double *x, int *ldx, double *xnorm, int *info)
{
    // Table of constant values
    int c__4 = 4;
    int c__1 = 1;
    int c__16 = 16;
    int c__0 = 0;

    /* Initialized data */

    static int locu12[4] = { 3,4,1,2 };
    static int locl21[4] = { 2,1,4,3 };
    static int locu22[4] = { 4,3,2,1 };
    static int xswpiv[4] = { FALSE_,FALSE_,TRUE_,TRUE_ };
    static int bswpiv[4] = { FALSE_,TRUE_,FALSE_,TRUE_ };

    // System generated locals
    int b_dim1, b_offset, tl_dim1, tl_offset, tr_dim1, tr_offset, x_dim1,
	    x_offset;
    double d__1, d__2, d__3, d__4, d__5, d__6, d__7, d__8;

    // Local variables
    int i__, j, k;
    double x2[2], l21, u11, u12;
    int ip, jp;
    double u22, t16[16]	/* was [4][4] */, gam, bet, eps, sgn, tmp[4], tau1,
	    btmp[4], smin;
    int ipiv;
    double temp;
    int jpiv[4];
    double xmax;
    int ipsv, jpsv;
    int bswap;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    ), dswap_(int *, double *, int *, double *, int *);
    int xswap;
    extern double dlamch_(char *);
    extern int idamax_(int *, double *, int *);
    double smlnum;

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
    //=====================================================================
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
    //    .. Data statements ..
    // Parameter adjustments
    tl_dim1 = *ldtl;
    tl_offset = 1 + tl_dim1;
    tl -= tl_offset;
    tr_dim1 = *ldtr;
    tr_offset = 1 + tr_dim1;
    tr -= tr_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;

    // Function Body
    //    ..
    //    .. Executable Statements ..
    //
    //    Do not check the input parameters for errors
    //
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n1 == 0 || *n2 == 0) {
	return 0;
    }
    //
    //    Set constants to control overflow
    //
    eps = dlamch_("P");
    smlnum = dlamch_("S") / eps;
    sgn = (double) (*isgn);
    k = *n1 + *n1 + *n2 - 2;
    switch (k) {
	case 1:  goto L10;
	case 2:  goto L20;
	case 3:  goto L30;
	case 4:  goto L50;
    }
    //
    //    1 by 1: TL11*X + SGN*X*TR11 = B11
    //
L10:
    tau1 = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    bet = abs(tau1);
    if (bet <= smlnum) {
	tau1 = smlnum;
	bet = smlnum;
	*info = 1;
    }
    *scale = 1.;
    gam = (d__1 = b[b_dim1 + 1], abs(d__1));
    if (smlnum * gam > bet) {
	*scale = 1. / gam;
    }
    x[x_dim1 + 1] = b[b_dim1 + 1] * *scale / tau1;
    *xnorm = (d__1 = x[x_dim1 + 1], abs(d__1));
    return 0;
    //
    //    1 by 2:
    //    TL11*[X11 X12] + ISGN*[X11 X12]*op[TR11 TR12]  = [B11 B12]
    //                                      [TR21 TR22]
    //
L20:
    //
    // Computing MAX
    // Computing MAX
    d__7 = (d__1 = tl[tl_dim1 + 1], abs(d__1)), d__8 = (d__2 = tr[tr_dim1 + 1]
	    , abs(d__2)), d__7 = max(d__7,d__8), d__8 = (d__3 = tr[(tr_dim1 <<
	     1) + 1], abs(d__3)), d__7 = max(d__7,d__8), d__8 = (d__4 = tr[
	    tr_dim1 + 2], abs(d__4)), d__7 = max(d__7,d__8), d__8 = (d__5 =
	    tr[(tr_dim1 << 1) + 2], abs(d__5));
    d__6 = eps * max(d__7,d__8);
    smin = max(d__6,smlnum);
    tmp[0] = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    tmp[3] = tl[tl_dim1 + 1] + sgn * tr[(tr_dim1 << 1) + 2];
    if (*ltranr) {
	tmp[1] = sgn * tr[tr_dim1 + 2];
	tmp[2] = sgn * tr[(tr_dim1 << 1) + 1];
    } else {
	tmp[1] = sgn * tr[(tr_dim1 << 1) + 1];
	tmp[2] = sgn * tr[tr_dim1 + 2];
    }
    btmp[0] = b[b_dim1 + 1];
    btmp[1] = b[(b_dim1 << 1) + 1];
    goto L40;
    //
    //    2 by 1:
    //         op[TL11 TL12]*[X11] + ISGN* [X11]*TR11  = [B11]
    //           [TL21 TL22] [X21]         [X21]         [B21]
    //
L30:
    // Computing MAX
    // Computing MAX
    d__7 = (d__1 = tr[tr_dim1 + 1], abs(d__1)), d__8 = (d__2 = tl[tl_dim1 + 1]
	    , abs(d__2)), d__7 = max(d__7,d__8), d__8 = (d__3 = tl[(tl_dim1 <<
	     1) + 1], abs(d__3)), d__7 = max(d__7,d__8), d__8 = (d__4 = tl[
	    tl_dim1 + 2], abs(d__4)), d__7 = max(d__7,d__8), d__8 = (d__5 =
	    tl[(tl_dim1 << 1) + 2], abs(d__5));
    d__6 = eps * max(d__7,d__8);
    smin = max(d__6,smlnum);
    tmp[0] = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    tmp[3] = tl[(tl_dim1 << 1) + 2] + sgn * tr[tr_dim1 + 1];
    if (*ltranl) {
	tmp[1] = tl[(tl_dim1 << 1) + 1];
	tmp[2] = tl[tl_dim1 + 2];
    } else {
	tmp[1] = tl[tl_dim1 + 2];
	tmp[2] = tl[(tl_dim1 << 1) + 1];
    }
    btmp[0] = b[b_dim1 + 1];
    btmp[1] = b[b_dim1 + 2];
L40:
    //
    //    Solve 2 by 2 system using complete pivoting.
    //    Set pivots less than SMIN to SMIN.
    //
    ipiv = idamax_(&c__4, tmp, &c__1);
    u11 = tmp[ipiv - 1];
    if (abs(u11) <= smin) {
	*info = 1;
	u11 = smin;
    }
    u12 = tmp[locu12[ipiv - 1] - 1];
    l21 = tmp[locl21[ipiv - 1] - 1] / u11;
    u22 = tmp[locu22[ipiv - 1] - 1] - u12 * l21;
    xswap = xswpiv[ipiv - 1];
    bswap = bswpiv[ipiv - 1];
    if (abs(u22) <= smin) {
	*info = 1;
	u22 = smin;
    }
    if (bswap) {
	temp = btmp[1];
	btmp[1] = btmp[0] - l21 * temp;
	btmp[0] = temp;
    } else {
	btmp[1] -= l21 * btmp[0];
    }
    *scale = 1.;
    if (smlnum * 2. * abs(btmp[1]) > abs(u22) || smlnum * 2. * abs(btmp[0]) >
	    abs(u11)) {
	// Computing MAX
	d__1 = abs(btmp[0]), d__2 = abs(btmp[1]);
	*scale = .5 / max(d__1,d__2);
	btmp[0] *= *scale;
	btmp[1] *= *scale;
    }
    x2[1] = btmp[1] / u22;
    x2[0] = btmp[0] / u11 - u12 / u11 * x2[1];
    if (xswap) {
	temp = x2[1];
	x2[1] = x2[0];
	x2[0] = temp;
    }
    x[x_dim1 + 1] = x2[0];
    if (*n1 == 1) {
	x[(x_dim1 << 1) + 1] = x2[1];
	*xnorm = (d__1 = x[x_dim1 + 1], abs(d__1)) + (d__2 = x[(x_dim1 << 1)
		+ 1], abs(d__2));
    } else {
	x[x_dim1 + 2] = x2[1];
	// Computing MAX
	d__3 = (d__1 = x[x_dim1 + 1], abs(d__1)), d__4 = (d__2 = x[x_dim1 + 2]
		, abs(d__2));
	*xnorm = max(d__3,d__4);
    }
    return 0;
    //
    //    2 by 2:
    //    op[TL11 TL12]*[X11 X12] +ISGN* [X11 X12]*op[TR11 TR12] = [B11 B12]
    //      [TL21 TL22] [X21 X22]        [X21 X22]   [TR21 TR22]   [B21 B22]
    //
    //    Solve equivalent 4 by 4 system using complete pivoting.
    //    Set pivots less than SMIN to SMIN.
    //
L50:
    // Computing MAX
    d__5 = (d__1 = tr[tr_dim1 + 1], abs(d__1)), d__6 = (d__2 = tr[(tr_dim1 <<
	    1) + 1], abs(d__2)), d__5 = max(d__5,d__6), d__6 = (d__3 = tr[
	    tr_dim1 + 2], abs(d__3)), d__5 = max(d__5,d__6), d__6 = (d__4 =
	    tr[(tr_dim1 << 1) + 2], abs(d__4));
    smin = max(d__5,d__6);
    // Computing MAX
    d__5 = smin, d__6 = (d__1 = tl[tl_dim1 + 1], abs(d__1)), d__5 = max(d__5,
	    d__6), d__6 = (d__2 = tl[(tl_dim1 << 1) + 1], abs(d__2)), d__5 =
	    max(d__5,d__6), d__6 = (d__3 = tl[tl_dim1 + 2], abs(d__3)), d__5 =
	     max(d__5,d__6), d__6 = (d__4 = tl[(tl_dim1 << 1) + 2], abs(d__4))
	    ;
    smin = max(d__5,d__6);
    // Computing MAX
    d__1 = eps * smin;
    smin = max(d__1,smlnum);
    btmp[0] = 0.;
    dcopy_(&c__16, btmp, &c__0, t16, &c__1);
    t16[0] = tl[tl_dim1 + 1] + sgn * tr[tr_dim1 + 1];
    t16[5] = tl[(tl_dim1 << 1) + 2] + sgn * tr[tr_dim1 + 1];
    t16[10] = tl[tl_dim1 + 1] + sgn * tr[(tr_dim1 << 1) + 2];
    t16[15] = tl[(tl_dim1 << 1) + 2] + sgn * tr[(tr_dim1 << 1) + 2];
    if (*ltranl) {
	t16[4] = tl[tl_dim1 + 2];
	t16[1] = tl[(tl_dim1 << 1) + 1];
	t16[14] = tl[tl_dim1 + 2];
	t16[11] = tl[(tl_dim1 << 1) + 1];
    } else {
	t16[4] = tl[(tl_dim1 << 1) + 1];
	t16[1] = tl[tl_dim1 + 2];
	t16[14] = tl[(tl_dim1 << 1) + 1];
	t16[11] = tl[tl_dim1 + 2];
    }
    if (*ltranr) {
	t16[8] = sgn * tr[(tr_dim1 << 1) + 1];
	t16[13] = sgn * tr[(tr_dim1 << 1) + 1];
	t16[2] = sgn * tr[tr_dim1 + 2];
	t16[7] = sgn * tr[tr_dim1 + 2];
    } else {
	t16[8] = sgn * tr[tr_dim1 + 2];
	t16[13] = sgn * tr[tr_dim1 + 2];
	t16[2] = sgn * tr[(tr_dim1 << 1) + 1];
	t16[7] = sgn * tr[(tr_dim1 << 1) + 1];
    }
    btmp[0] = b[b_dim1 + 1];
    btmp[1] = b[b_dim1 + 2];
    btmp[2] = b[(b_dim1 << 1) + 1];
    btmp[3] = b[(b_dim1 << 1) + 2];
    //
    //    Perform elimination
    //
    for (i__ = 1; i__ <= 3; ++i__) {
	xmax = 0.;
	for (ip = i__; ip <= 4; ++ip) {
	    for (jp = i__; jp <= 4; ++jp) {
		if ((d__1 = t16[ip + (jp << 2) - 5], abs(d__1)) >= xmax) {
		    xmax = (d__1 = t16[ip + (jp << 2) - 5], abs(d__1));
		    ipsv = ip;
		    jpsv = jp;
		}
// L60:
	    }
// L70:
	}
	if (ipsv != i__) {
	    dswap_(&c__4, &t16[ipsv - 1], &c__4, &t16[i__ - 1], &c__4);
	    temp = btmp[i__ - 1];
	    btmp[i__ - 1] = btmp[ipsv - 1];
	    btmp[ipsv - 1] = temp;
	}
	if (jpsv != i__) {
	    dswap_(&c__4, &t16[(jpsv << 2) - 4], &c__1, &t16[(i__ << 2) - 4],
		    &c__1);
	}
	jpiv[i__ - 1] = jpsv;
	if ((d__1 = t16[i__ + (i__ << 2) - 5], abs(d__1)) < smin) {
	    *info = 1;
	    t16[i__ + (i__ << 2) - 5] = smin;
	}
	for (j = i__ + 1; j <= 4; ++j) {
	    t16[j + (i__ << 2) - 5] /= t16[i__ + (i__ << 2) - 5];
	    btmp[j - 1] -= t16[j + (i__ << 2) - 5] * btmp[i__ - 1];
	    for (k = i__ + 1; k <= 4; ++k) {
		t16[j + (k << 2) - 5] -= t16[j + (i__ << 2) - 5] * t16[i__ + (
			k << 2) - 5];
// L80:
	    }
// L90:
	}
// L100:
    }
    if (abs(t16[15]) < smin) {
	*info = 1;
	t16[15] = smin;
    }
    *scale = 1.;
    if (smlnum * 8. * abs(btmp[0]) > abs(t16[0]) || smlnum * 8. * abs(btmp[1])
	     > abs(t16[5]) || smlnum * 8. * abs(btmp[2]) > abs(t16[10]) ||
	    smlnum * 8. * abs(btmp[3]) > abs(t16[15])) {
	// Computing MAX
	d__1 = abs(btmp[0]), d__2 = abs(btmp[1]), d__1 = max(d__1,d__2), d__2
		= abs(btmp[2]), d__1 = max(d__1,d__2), d__2 = abs(btmp[3]);
	*scale = .125 / max(d__1,d__2);
	btmp[0] *= *scale;
	btmp[1] *= *scale;
	btmp[2] *= *scale;
	btmp[3] *= *scale;
    }
    for (i__ = 1; i__ <= 4; ++i__) {
	k = 5 - i__;
	temp = 1. / t16[k + (k << 2) - 5];
	tmp[k - 1] = btmp[k - 1] * temp;
	for (j = k + 1; j <= 4; ++j) {
	    tmp[k - 1] -= temp * t16[k + (j << 2) - 5] * tmp[j - 1];
// L110:
	}
// L120:
    }
    for (i__ = 1; i__ <= 3; ++i__) {
	if (jpiv[4 - i__ - 1] != 4 - i__) {
	    temp = tmp[4 - i__ - 1];
	    tmp[4 - i__ - 1] = tmp[jpiv[4 - i__ - 1] - 1];
	    tmp[jpiv[4 - i__ - 1] - 1] = temp;
	}
// L130:
    }
    x[x_dim1 + 1] = tmp[0];
    x[x_dim1 + 2] = tmp[1];
    x[(x_dim1 << 1) + 1] = tmp[2];
    x[(x_dim1 << 1) + 2] = tmp[3];
    // Computing MAX
    d__1 = abs(tmp[0]) + abs(tmp[2]), d__2 = abs(tmp[1]) + abs(tmp[3]);
    *xnorm = max(d__1,d__2);
    return 0;
    //
    //    End of DLASY2
    //
} // dlasy2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORGHR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORGHR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorghr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorghr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorghr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORGHR( N, ILO, IHI, A, LDA, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IHI, ILO, INFO, LDA, LWORK, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DORGHR generates a real orthogonal matrix Q which is defined as the
//> product of IHI-ILO elementary reflectors of order N, as returned by
//> DGEHRD:
//>
//> Q = H(ilo) H(ilo+1) . . . H(ihi-1).
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix Q. N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>
//>          ILO and IHI must have the same values as in the previous call
//>          of DGEHRD. Q is equal to the unit matrix except in the
//>          submatrix Q(ilo+1:ihi,ilo+1:ihi).
//>          1 <= ILO <= IHI <= N, if N > 0; ILO=1 and IHI=0, if N=0.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the vectors which define the elementary reflectors,
//>          as returned by DGEHRD.
//>          On exit, the N-by-N orthogonal matrix Q.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A. LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (N-1)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGEHRD.
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
//>          The dimension of the array WORK. LWORK >= IHI-ILO.
//>          For optimum performance LWORK >= (IHI-ILO)*NB, where NB is
//>          the optimal blocksize.
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
/* Subroutine */ int dorghr_(int *n, int *ilo, int *ihi, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, nb, nh, iinfo;
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dorgqr_(int *, int *, int *, double *, int *,
	    double *, double *, int *, int *);
    int lwkopt;
    int lquery;

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
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    --work;

    // Function Body
    *info = 0;
    nh = *ihi - *ilo;
    lquery = *lwork == -1;
    if (*n < 0) {
	*info = -1;
    } else if (*ilo < 1 || *ilo > max(1,*n)) {
	*info = -2;
    } else if (*ihi < min(*ilo,*n) || *ihi > *n) {
	*info = -3;
    } else if (*lda < max(1,*n)) {
	*info = -5;
    } else if (*lwork < max(1,nh) && ! lquery) {
	*info = -8;
    }
    if (*info == 0) {
	nb = ilaenv_(&c__1, "DORGQR", " ", &nh, &nh, &nh, &c_n1);
	lwkopt = max(1,nh) * nb;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORGHR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0) {
	work[1] = 1.;
	return 0;
    }
    //
    //    Shift the vectors which define the elementary reflectors one
    //    column to the right, and set the first ilo and the last n-ihi
    //    rows and columns to those of the unit matrix
    //
    i__1 = *ilo + 1;
    for (j = *ihi; j >= i__1; --j) {
	i__2 = j - 1;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.;
// L10:
	}
	i__2 = *ihi;
	for (i__ = j + 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = a[i__ + (j - 1) * a_dim1];
// L20:
	}
	i__2 = *n;
	for (i__ = *ihi + 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.;
// L30:
	}
// L40:
    }
    i__1 = *ilo;
    for (j = 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.;
// L50:
	}
	a[j + j * a_dim1] = 1.;
// L60:
    }
    i__1 = *n;
    for (j = *ihi + 1; j <= i__1; ++j) {
	i__2 = *n;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    a[i__ + j * a_dim1] = 0.;
// L70:
	}
	a[j + j * a_dim1] = 1.;
// L80:
    }
    if (nh > 0) {
	//
	//       Generate Q(ilo+1:ihi,ilo+1:ihi)
	//
	dorgqr_(&nh, &nh, &nh, &a[*ilo + 1 + (*ilo + 1) * a_dim1], lda, &tau[*
		ilo], &work[1], lwork, &iinfo);
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORGHR
    //
} // dorghr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORMHR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORMHR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dormhr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dormhr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dormhr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORMHR( SIDE, TRANS, M, N, ILO, IHI, A, LDA, TAU, C,
//                         LDC, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            IHI, ILO, INFO, LDA, LDC, LWORK, M, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), C( LDC, * ), TAU( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DORMHR overwrites the general real M-by-N matrix C with
//>
//>                 SIDE = 'L'     SIDE = 'R'
//> TRANS = 'N':      Q * C          C * Q
//> TRANS = 'T':      Q**T * C       C * Q**T
//>
//> where Q is a real orthogonal matrix of order nq, with nq = m if
//> SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
//> IHI-ILO elementary reflectors, as returned by DGEHRD:
//>
//> Q = H(ilo) H(ilo+1) . . . H(ihi-1).
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'L': apply Q or Q**T from the Left;
//>          = 'R': apply Q or Q**T from the Right.
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          = 'N':  No transpose, apply Q;
//>          = 'T':  Transpose, apply Q**T.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of rows of the matrix C. M >= 0.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of columns of the matrix C. N >= 0.
//> \endverbatim
//>
//> \param[in] ILO
//> \verbatim
//>          ILO is INTEGER
//> \endverbatim
//>
//> \param[in] IHI
//> \verbatim
//>          IHI is INTEGER
//>
//>          ILO and IHI must have the same values as in the previous call
//>          of DGEHRD. Q is equal to the unit matrix except in the
//>          submatrix Q(ilo+1:ihi,ilo+1:ihi).
//>          If SIDE = 'L', then 1 <= ILO <= IHI <= M, if M > 0, and
//>          ILO = 1 and IHI = 0, if M = 0;
//>          if SIDE = 'R', then 1 <= ILO <= IHI <= N, if N > 0, and
//>          ILO = 1 and IHI = 0, if N = 0.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension
//>                               (LDA,M) if SIDE = 'L'
//>                               (LDA,N) if SIDE = 'R'
//>          The vectors which define the elementary reflectors, as
//>          returned by DGEHRD.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//>          LDA >= max(1,M) if SIDE = 'L'; LDA >= max(1,N) if SIDE = 'R'.
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension
//>                               (M-1) if SIDE = 'L'
//>                               (N-1) if SIDE = 'R'
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGEHRD.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC,N)
//>          On entry, the M-by-N matrix C.
//>          On exit, C is overwritten by Q*C or Q**T*C or C*Q**T or C*Q.
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
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK.
//>          If SIDE = 'L', LWORK >= max(1,N);
//>          if SIDE = 'R', LWORK >= max(1,M).
//>          For optimum performance LWORK >= N*NB if SIDE = 'L', and
//>          LWORK >= M*NB if SIDE = 'R', where NB is the optimal
//>          blocksize.
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
/* Subroutine */ int dormhr_(char *side, char *trans, int *m, int *n, int *
	ilo, int *ihi, double *a, int *lda, double *tau, double *c__, int *
	ldc, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__2 = 2;

    // System generated locals
    address a__1[2];
    int a_dim1, a_offset, c_dim1, c_offset, i__1[2], i__2;
    char ch__1[2+1]={'\0'};

    // Local variables
    int i1, i2, nb, mi, nh, ni, nq, nw;
    int left;
    extern int lsame_(char *, char *);
    int iinfo;
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dormqr_(char *, char *, int *, int *, int *,
	    double *, int *, double *, double *, int *, double *, int *, int *
	    );
    int lwkopt;
    int lquery;

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
    //    Test the input arguments
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --tau;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    // Function Body
    *info = 0;
    nh = *ihi - *ilo;
    left = lsame_(side, "L");
    lquery = *lwork == -1;
    //
    //    NQ is the order of Q and NW is the minimum dimension of WORK
    //
    if (left) {
	nq = *m;
	nw = *n;
    } else {
	nq = *n;
	nw = *m;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ilo < 1 || *ilo > max(1,nq)) {
	*info = -5;
    } else if (*ihi < min(*ilo,nq) || *ihi > nq) {
	*info = -6;
    } else if (*lda < max(1,nq)) {
	*info = -8;
    } else if (*ldc < max(1,*m)) {
	*info = -11;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -13;
    }
    if (*info == 0) {
	if (left) {
	    // Writing concatenation
	    i__1[0] = 1, a__1[0] = side;
	    i__1[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__1, &c__2);
	    nb = ilaenv_(&c__1, "DORMQR", ch__1, &nh, n, &nh, &c_n1);
	} else {
	    // Writing concatenation
	    i__1[0] = 1, a__1[0] = side;
	    i__1[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__1, &c__2);
	    nb = ilaenv_(&c__1, "DORMQR", ch__1, m, &nh, &nh, &c_n1);
	}
	lwkopt = max(1,nw) * nb;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__2 = -(*info);
	xerbla_("DORMHR", &i__2);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || nh == 0) {
	work[1] = 1.;
	return 0;
    }
    if (left) {
	mi = nh;
	ni = *n;
	i1 = *ilo + 1;
	i2 = 1;
    } else {
	mi = *m;
	ni = nh;
	i1 = 1;
	i2 = *ilo + 1;
    }
    dormqr_(side, trans, &mi, &ni, &nh, &a[*ilo + 1 + *ilo * a_dim1], lda, &
	    tau[*ilo], &c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORMHR
    //
} // dormhr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DTREVC3
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DTREVC3 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dtrevc3.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dtrevc3.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dtrevc3.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DTREVC3( SIDE, HOWMNY, SELECT, N, T, LDT, VL, LDVL,
//                          VR, LDVR, MM, M, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          HOWMNY, SIDE
//      INTEGER            INFO, LDT, LDVL, LDVR, LWORK, M, MM, N
//      ..
//      .. Array Arguments ..
//      LOGICAL            SELECT( * )
//      DOUBLE PRECISION   T( LDT, * ), VL( LDVL, * ), VR( LDVR, * ),
//     $                   WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DTREVC3 computes some or all of the right and/or left eigenvectors of
//> a real upper quasi-triangular matrix T.
//> Matrices of this type are produced by the Schur factorization of
//> a real general matrix:  A = Q*T*Q**T, as computed by DHSEQR.
//>
//> The right eigenvector x and the left eigenvector y of T corresponding
//> to an eigenvalue w are defined by:
//>
//>    T*x = w*x,     (y**T)*T = w*(y**T)
//>
//> where y**T denotes the transpose of the vector y.
//> The eigenvalues are not input to this routine, but are read directly
//> from the diagonal blocks of T.
//>
//> This routine returns the matrices X and/or Y of right and left
//> eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
//> input matrix. If Q is the orthogonal factor that reduces a matrix
//> A to Schur form T, then Q*X and Q*Y are the matrices of right and
//> left eigenvectors of A.
//>
//> This uses a Level 3 BLAS version of the back transformation.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'R':  compute right eigenvectors only;
//>          = 'L':  compute left eigenvectors only;
//>          = 'B':  compute both right and left eigenvectors.
//> \endverbatim
//>
//> \param[in] HOWMNY
//> \verbatim
//>          HOWMNY is CHARACTER*1
//>          = 'A':  compute all right and/or left eigenvectors;
//>          = 'B':  compute all right and/or left eigenvectors,
//>                  backtransformed by the matrices in VR and/or VL;
//>          = 'S':  compute selected right and/or left eigenvectors,
//>                  as indicated by the logical array SELECT.
//> \endverbatim
//>
//> \param[in,out] SELECT
//> \verbatim
//>          SELECT is LOGICAL array, dimension (N)
//>          If HOWMNY = 'S', SELECT specifies the eigenvectors to be
//>          computed.
//>          If w(j) is a real eigenvalue, the corresponding real
//>          eigenvector is computed if SELECT(j) is .TRUE..
//>          If w(j) and w(j+1) are the real and imaginary parts of a
//>          complex eigenvalue, the corresponding complex eigenvector is
//>          computed if either SELECT(j) or SELECT(j+1) is .TRUE., and
//>          on exit SELECT(j) is set to .TRUE. and SELECT(j+1) is set to
//>          .FALSE..
//>          Not referenced if HOWMNY = 'A' or 'B'.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix T. N >= 0.
//> \endverbatim
//>
//> \param[in] T
//> \verbatim
//>          T is DOUBLE PRECISION array, dimension (LDT,N)
//>          The upper quasi-triangular matrix T in Schur canonical form.
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of the array T. LDT >= max(1,N).
//> \endverbatim
//>
//> \param[in,out] VL
//> \verbatim
//>          VL is DOUBLE PRECISION array, dimension (LDVL,MM)
//>          On entry, if SIDE = 'L' or 'B' and HOWMNY = 'B', VL must
//>          contain an N-by-N matrix Q (usually the orthogonal matrix Q
//>          of Schur vectors returned by DHSEQR).
//>          On exit, if SIDE = 'L' or 'B', VL contains:
//>          if HOWMNY = 'A', the matrix Y of left eigenvectors of T;
//>          if HOWMNY = 'B', the matrix Q*Y;
//>          if HOWMNY = 'S', the left eigenvectors of T specified by
//>                           SELECT, stored consecutively in the columns
//>                           of VL, in the same order as their
//>                           eigenvalues.
//>          A complex eigenvector corresponding to a complex eigenvalue
//>          is stored in two consecutive columns, the first holding the
//>          real part, and the second the imaginary part.
//>          Not referenced if SIDE = 'R'.
//> \endverbatim
//>
//> \param[in] LDVL
//> \verbatim
//>          LDVL is INTEGER
//>          The leading dimension of the array VL.
//>          LDVL >= 1, and if SIDE = 'L' or 'B', LDVL >= N.
//> \endverbatim
//>
//> \param[in,out] VR
//> \verbatim
//>          VR is DOUBLE PRECISION array, dimension (LDVR,MM)
//>          On entry, if SIDE = 'R' or 'B' and HOWMNY = 'B', VR must
//>          contain an N-by-N matrix Q (usually the orthogonal matrix Q
//>          of Schur vectors returned by DHSEQR).
//>          On exit, if SIDE = 'R' or 'B', VR contains:
//>          if HOWMNY = 'A', the matrix X of right eigenvectors of T;
//>          if HOWMNY = 'B', the matrix Q*X;
//>          if HOWMNY = 'S', the right eigenvectors of T specified by
//>                           SELECT, stored consecutively in the columns
//>                           of VR, in the same order as their
//>                           eigenvalues.
//>          A complex eigenvector corresponding to a complex eigenvalue
//>          is stored in two consecutive columns, the first holding the
//>          real part and the second the imaginary part.
//>          Not referenced if SIDE = 'L'.
//> \endverbatim
//>
//> \param[in] LDVR
//> \verbatim
//>          LDVR is INTEGER
//>          The leading dimension of the array VR.
//>          LDVR >= 1, and if SIDE = 'R' or 'B', LDVR >= N.
//> \endverbatim
//>
//> \param[in] MM
//> \verbatim
//>          MM is INTEGER
//>          The number of columns in the arrays VL and/or VR. MM >= M.
//> \endverbatim
//>
//> \param[out] M
//> \verbatim
//>          M is INTEGER
//>          The number of columns in the arrays VL and/or VR actually
//>          used to store the eigenvectors.
//>          If HOWMNY = 'A' or 'B', M is set to N.
//>          Each selected real eigenvector occupies one column and each
//>          selected complex eigenvector occupies two columns.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of array WORK. LWORK >= max(1,3*N).
//>          For optimum performance, LWORK >= N + 2*N*NB, where NB is
//>          the optimal blocksize.
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
// @precisions fortran d -> s
//
//> \ingroup doubleOTHERcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The algorithm used in this program is basically backward (forward)
//>  substitution, with scaling to make the the code robust against
//>  possible overflow.
//>
//>  Each eigenvector is normalized so that the element of largest
//>  magnitude has magnitude 1; here the magnitude of a complex number
//>  (x,y) is taken to be |x| + |y|.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dtrevc3_(char *side, char *howmny, int *select, int *n,
	double *t, int *ldt, double *vl, int *ldvl, double *vr, int *ldvr,
	int *mm, int *m, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__2 = 2;
    double c_b17 = 0.;
    int c_false = FALSE_;
    double c_b29 = 1.;
    int c_true = TRUE_;

    // System generated locals
    address a__1[2];
    int t_dim1, t_offset, vl_dim1, vl_offset, vr_dim1, vr_offset, i__1[2],
	    i__2, i__3, i__4;
    double d__1, d__2, d__3, d__4;
    char ch__1[2+1]={'\0'};

    // Local variables
    int i__, j, k;
    double x[4]	/* was [2][2] */;
    int j1, j2, iscomplex[128], nb, ii, ki, ip, is, iv;
    double wi, wr;
    int ki2;
    double rec, ulp, beta, emax;
    int pair;
    extern double ddot_(int *, double *, int *, double *, int *);
    int allv;
    int ierr;
    double unfl, ovfl, smin;
    int over;
    double vmax;
    int jnxt;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    double scale;
    extern /* Subroutine */ int dgemm_(char *, char *, int *, int *, int *,
	    double *, double *, int *, double *, int *, double *, double *,
	    int *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, int *, int *, double *, double
	    *, int *, double *, int *, double *, double *, int *);
    double remax;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int leftv, bothv;
    extern /* Subroutine */ int daxpy_(int *, double *, double *, int *,
	    double *, int *);
    double vcrit;
    int somev;
    double xnorm;
    extern /* Subroutine */ int dlaln2_(int *, int *, int *, double *, double
	    *, double *, int *, double *, double *, double *, int *, double *,
	     double *, double *, int *, double *, double *, int *), dlabad_(
	    double *, double *);
    extern double dlamch_(char *);
    extern int idamax_(int *, double *, int *);
    extern /* Subroutine */ int dlaset_(char *, int *, int *, double *,
	    double *, double *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dlacpy_(char *, int *, int *, double *, int *,
	     double *, int *);
    double bignum;
    int rightv;
    int maxwrk;
    double smlnum;
    int lquery;

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
    //    .. Local Arrays ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Decode and test the input parameters
    //
    // Parameter adjustments
    --select;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    vl_dim1 = *ldvl;
    vl_offset = 1 + vl_dim1;
    vl -= vl_offset;
    vr_dim1 = *ldvr;
    vr_offset = 1 + vr_dim1;
    vr -= vr_offset;
    --work;

    // Function Body
    bothv = lsame_(side, "B");
    rightv = lsame_(side, "R") || bothv;
    leftv = lsame_(side, "L") || bothv;
    allv = lsame_(howmny, "A");
    over = lsame_(howmny, "B");
    somev = lsame_(howmny, "S");
    *info = 0;
    // Writing concatenation
    i__1[0] = 1, a__1[0] = side;
    i__1[1] = 1, a__1[1] = howmny;
    s_cat(ch__1, a__1, i__1, &c__2);
    nb = ilaenv_(&c__1, "DTREVC", ch__1, n, &c_n1, &c_n1, &c_n1);
    maxwrk = *n + (*n << 1) * nb;
    work[1] = (double) maxwrk;
    lquery = *lwork == -1;
    if (! rightv && ! leftv) {
	*info = -1;
    } else if (! allv && ! over && ! somev) {
	*info = -2;
    } else if (*n < 0) {
	*info = -4;
    } else if (*ldt < max(1,*n)) {
	*info = -6;
    } else if (*ldvl < 1 || leftv && *ldvl < *n) {
	*info = -8;
    } else if (*ldvr < 1 || rightv && *ldvr < *n) {
	*info = -10;
    } else /* if(complicated condition) */ {
	// Computing MAX
	i__2 = 1, i__3 = *n * 3;
	if (*lwork < max(i__2,i__3) && ! lquery) {
	    *info = -14;
	} else {
	    //
	    //       Set M to the number of columns required to store the selected
	    //       eigenvectors, standardize the array SELECT if necessary, and
	    //       test MM.
	    //
	    if (somev) {
		*m = 0;
		pair = FALSE_;
		i__2 = *n;
		for (j = 1; j <= i__2; ++j) {
		    if (pair) {
			pair = FALSE_;
			select[j] = FALSE_;
		    } else {
			if (j < *n) {
			    if (t[j + 1 + j * t_dim1] == 0.) {
				if (select[j]) {
				    ++(*m);
				}
			    } else {
				pair = TRUE_;
				if (select[j] || select[j + 1]) {
				    select[j] = TRUE_;
				    *m += 2;
				}
			    }
			} else {
			    if (select[*n]) {
				++(*m);
			    }
			}
		    }
// L10:
		}
	    } else {
		*m = *n;
	    }
	    if (*mm < *m) {
		*info = -11;
	    }
	}
    }
    if (*info != 0) {
	i__2 = -(*info);
	xerbla_("DTREVC3", &i__2);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*n == 0) {
	return 0;
    }
    //
    //    Use blocked version of back-transformation if sufficient workspace.
    //    Zero-out the workspace to avoid potential NaN propagation.
    //
    if (over && *lwork >= *n + (*n << 4)) {
	nb = (*lwork - *n) / (*n << 1);
	nb = min(nb,128);
	i__2 = (nb << 1) + 1;
	dlaset_("F", n, &i__2, &c_b17, &c_b17, &work[1], n);
    } else {
	nb = 1;
    }
    //
    //    Set the constants to control overflow.
    //
    unfl = dlamch_("Safe minimum");
    ovfl = 1. / unfl;
    dlabad_(&unfl, &ovfl);
    ulp = dlamch_("Precision");
    smlnum = unfl * (*n / ulp);
    bignum = (1. - ulp) / smlnum;
    //
    //    Compute 1-norm of each column of strictly upper triangular
    //    part of T to control overflow in triangular solver.
    //
    work[1] = 0.;
    i__2 = *n;
    for (j = 2; j <= i__2; ++j) {
	work[j] = 0.;
	i__3 = j - 1;
	for (i__ = 1; i__ <= i__3; ++i__) {
	    work[j] += (d__1 = t[i__ + j * t_dim1], abs(d__1));
// L20:
	}
// L30:
    }
    //
    //    Index IP is used to specify the real or complex eigenvalue:
    //      IP = 0, real eigenvalue,
    //           1, first  of conjugate complex pair: (wr,wi)
    //          -1, second of conjugate complex pair: (wr,wi)
    //      ISCOMPLEX array stores IP for each column in current block.
    //
    if (rightv) {
	//
	//       ============================================================
	//       Compute right eigenvectors.
	//
	//       IV is index of column in current block.
	//       For complex right vector, uses IV-1 for real part and IV for complex part.
	//       Non-blocked version always uses IV=2;
	//       blocked     version starts with IV=NB, goes down to 1 or 2.
	//       (Note the "0-th" column is used for 1-norms computed above.)
	iv = 2;
	if (nb > 2) {
	    iv = nb;
	}
	ip = 0;
	is = *m;
	for (ki = *n; ki >= 1; --ki) {
	    if (ip == -1) {
		//             previous iteration (ki+1) was second of conjugate pair,
		//             so this ki is first of conjugate pair; skip to end of loop
		ip = 1;
		goto L140;
	    } else if (ki == 1) {
		//             last column, so this ki must be real eigenvalue
		ip = 0;
	    } else if (t[ki + (ki - 1) * t_dim1] == 0.) {
		//             zero on sub-diagonal, so this ki is real eigenvalue
		ip = 0;
	    } else {
		//             non-zero on sub-diagonal, so this ki is second of conjugate pair
		ip = -1;
	    }
	    if (somev) {
		if (ip == 0) {
		    if (! select[ki]) {
			goto L140;
		    }
		} else {
		    if (! select[ki - 1]) {
			goto L140;
		    }
		}
	    }
	    //
	    //          Compute the KI-th eigenvalue (WR,WI).
	    //
	    wr = t[ki + ki * t_dim1];
	    wi = 0.;
	    if (ip != 0) {
		wi = sqrt((d__1 = t[ki + (ki - 1) * t_dim1], abs(d__1))) *
			sqrt((d__2 = t[ki - 1 + ki * t_dim1], abs(d__2)));
	    }
	    // Computing MAX
	    d__1 = ulp * (abs(wr) + abs(wi));
	    smin = max(d__1,smlnum);
	    if (ip == 0) {
		//
		//             --------------------------------------------------------
		//             Real right eigenvector
		//
		work[ki + iv * *n] = 1.;
		//
		//             Form right-hand side.
		//
		i__2 = ki - 1;
		for (k = 1; k <= i__2; ++k) {
		    work[k + iv * *n] = -t[k + ki * t_dim1];
// L50:
		}
		//
		//             Solve upper quasi-triangular system:
		//             [ T(1:KI-1,1:KI-1) - WR ]*X = SCALE*WORK.
		//
		jnxt = ki - 1;
		for (j = ki - 1; j >= 1; --j) {
		    if (j > jnxt) {
			goto L60;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j - 1;
		    if (j > 1) {
			if (t[j + (j - 1) * t_dim1] != 0.) {
			    j1 = j - 1;
			    jnxt = j - 2;
			}
		    }
		    if (j1 == j2) {
			//
			//                   1-by-1 diagonal block
			//
			dlaln2_(&c_false, &c__1, &c__1, &smin, &c_b29, &t[j +
				j * t_dim1], ldt, &c_b29, &c_b29, &work[j +
				iv * *n], n, &wr, &c_b17, x, &c__2, &scale, &
				xnorm, &ierr);
			//
			//                   Scale X(1,1) to avoid overflow when updating
			//                   the right-hand side.
			//
			if (xnorm > 1.) {
			    if (work[j] > bignum / xnorm) {
				x[0] /= xnorm;
				scale /= xnorm;
			    }
			}
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    dscal_(&ki, &scale, &work[iv * *n + 1], &c__1);
			}
			work[j + iv * *n] = x[0];
			//
			//                   Update right-hand side
			//
			i__2 = j - 1;
			d__1 = -x[0];
			daxpy_(&i__2, &d__1, &t[j * t_dim1 + 1], &c__1, &work[
				iv * *n + 1], &c__1);
		    } else {
			//
			//                   2-by-2 diagonal block
			//
			dlaln2_(&c_false, &c__2, &c__1, &smin, &c_b29, &t[j -
				1 + (j - 1) * t_dim1], ldt, &c_b29, &c_b29, &
				work[j - 1 + iv * *n], n, &wr, &c_b17, x, &
				c__2, &scale, &xnorm, &ierr);
			//
			//                   Scale X(1,1) and X(2,1) to avoid overflow when
			//                   updating the right-hand side.
			//
			if (xnorm > 1.) {
			    // Computing MAX
			    d__1 = work[j - 1], d__2 = work[j];
			    beta = max(d__1,d__2);
			    if (beta > bignum / xnorm) {
				x[0] /= xnorm;
				x[1] /= xnorm;
				scale /= xnorm;
			    }
			}
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    dscal_(&ki, &scale, &work[iv * *n + 1], &c__1);
			}
			work[j - 1 + iv * *n] = x[0];
			work[j + iv * *n] = x[1];
			//
			//                   Update right-hand side
			//
			i__2 = j - 2;
			d__1 = -x[0];
			daxpy_(&i__2, &d__1, &t[(j - 1) * t_dim1 + 1], &c__1,
				&work[iv * *n + 1], &c__1);
			i__2 = j - 2;
			d__1 = -x[1];
			daxpy_(&i__2, &d__1, &t[j * t_dim1 + 1], &c__1, &work[
				iv * *n + 1], &c__1);
		    }
L60:
		    ;
		}
		//
		//             Copy the vector x or Q*x to VR and normalize.
		//
		if (! over) {
		    //                ------------------------------
		    //                no back-transform: copy x to VR and normalize.
		    dcopy_(&ki, &work[iv * *n + 1], &c__1, &vr[is * vr_dim1 +
			    1], &c__1);
		    ii = idamax_(&ki, &vr[is * vr_dim1 + 1], &c__1);
		    remax = 1. / (d__1 = vr[ii + is * vr_dim1], abs(d__1));
		    dscal_(&ki, &remax, &vr[is * vr_dim1 + 1], &c__1);
		    i__2 = *n;
		    for (k = ki + 1; k <= i__2; ++k) {
			vr[k + is * vr_dim1] = 0.;
// L70:
		    }
		} else if (nb == 1) {
		    //                ------------------------------
		    //                version 1: back-transform each vector with GEMV, Q*x.
		    if (ki > 1) {
			i__2 = ki - 1;
			dgemv_("N", n, &i__2, &c_b29, &vr[vr_offset], ldvr, &
				work[iv * *n + 1], &c__1, &work[ki + iv * *n],
				 &vr[ki * vr_dim1 + 1], &c__1);
		    }
		    ii = idamax_(n, &vr[ki * vr_dim1 + 1], &c__1);
		    remax = 1. / (d__1 = vr[ii + ki * vr_dim1], abs(d__1));
		    dscal_(n, &remax, &vr[ki * vr_dim1 + 1], &c__1);
		} else {
		    //                ------------------------------
		    //                version 2: back-transform block of vectors with GEMM
		    //                zero out below vector
		    i__2 = *n;
		    for (k = ki + 1; k <= i__2; ++k) {
			work[k + iv * *n] = 0.;
		    }
		    iscomplex[iv - 1] = ip;
		    //                back-transform and normalization is done below
		}
	    } else {
		//
		//             --------------------------------------------------------
		//             Complex right eigenvector.
		//
		//             Initial solve
		//             [ ( T(KI-1,KI-1) T(KI-1,KI) ) - (WR + I*WI) ]*X = 0.
		//             [ ( T(KI,  KI-1) T(KI,  KI) )               ]
		//
		if ((d__1 = t[ki - 1 + ki * t_dim1], abs(d__1)) >= (d__2 = t[
			ki + (ki - 1) * t_dim1], abs(d__2))) {
		    work[ki - 1 + (iv - 1) * *n] = 1.;
		    work[ki + iv * *n] = wi / t[ki - 1 + ki * t_dim1];
		} else {
		    work[ki - 1 + (iv - 1) * *n] = -wi / t[ki + (ki - 1) *
			    t_dim1];
		    work[ki + iv * *n] = 1.;
		}
		work[ki + (iv - 1) * *n] = 0.;
		work[ki - 1 + iv * *n] = 0.;
		//
		//             Form right-hand side.
		//
		i__2 = ki - 2;
		for (k = 1; k <= i__2; ++k) {
		    work[k + (iv - 1) * *n] = -work[ki - 1 + (iv - 1) * *n] *
			    t[k + (ki - 1) * t_dim1];
		    work[k + iv * *n] = -work[ki + iv * *n] * t[k + ki *
			    t_dim1];
// L80:
		}
		//
		//             Solve upper quasi-triangular system:
		//             [ T(1:KI-2,1:KI-2) - (WR+i*WI) ]*X = SCALE*(WORK+i*WORK2)
		//
		jnxt = ki - 2;
		for (j = ki - 2; j >= 1; --j) {
		    if (j > jnxt) {
			goto L90;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j - 1;
		    if (j > 1) {
			if (t[j + (j - 1) * t_dim1] != 0.) {
			    j1 = j - 1;
			    jnxt = j - 2;
			}
		    }
		    if (j1 == j2) {
			//
			//                   1-by-1 diagonal block
			//
			dlaln2_(&c_false, &c__1, &c__2, &smin, &c_b29, &t[j +
				j * t_dim1], ldt, &c_b29, &c_b29, &work[j + (
				iv - 1) * *n], n, &wr, &wi, x, &c__2, &scale,
				&xnorm, &ierr);
			//
			//                   Scale X(1,1) and X(1,2) to avoid overflow when
			//                   updating the right-hand side.
			//
			if (xnorm > 1.) {
			    if (work[j] > bignum / xnorm) {
				x[0] /= xnorm;
				x[2] /= xnorm;
				scale /= xnorm;
			    }
			}
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    dscal_(&ki, &scale, &work[(iv - 1) * *n + 1], &
				    c__1);
			    dscal_(&ki, &scale, &work[iv * *n + 1], &c__1);
			}
			work[j + (iv - 1) * *n] = x[0];
			work[j + iv * *n] = x[2];
			//
			//                   Update the right-hand side
			//
			i__2 = j - 1;
			d__1 = -x[0];
			daxpy_(&i__2, &d__1, &t[j * t_dim1 + 1], &c__1, &work[
				(iv - 1) * *n + 1], &c__1);
			i__2 = j - 1;
			d__1 = -x[2];
			daxpy_(&i__2, &d__1, &t[j * t_dim1 + 1], &c__1, &work[
				iv * *n + 1], &c__1);
		    } else {
			//
			//                   2-by-2 diagonal block
			//
			dlaln2_(&c_false, &c__2, &c__2, &smin, &c_b29, &t[j -
				1 + (j - 1) * t_dim1], ldt, &c_b29, &c_b29, &
				work[j - 1 + (iv - 1) * *n], n, &wr, &wi, x, &
				c__2, &scale, &xnorm, &ierr);
			//
			//                   Scale X to avoid overflow when updating
			//                   the right-hand side.
			//
			if (xnorm > 1.) {
			    // Computing MAX
			    d__1 = work[j - 1], d__2 = work[j];
			    beta = max(d__1,d__2);
			    if (beta > bignum / xnorm) {
				rec = 1. / xnorm;
				x[0] *= rec;
				x[2] *= rec;
				x[1] *= rec;
				x[3] *= rec;
				scale *= rec;
			    }
			}
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    dscal_(&ki, &scale, &work[(iv - 1) * *n + 1], &
				    c__1);
			    dscal_(&ki, &scale, &work[iv * *n + 1], &c__1);
			}
			work[j - 1 + (iv - 1) * *n] = x[0];
			work[j + (iv - 1) * *n] = x[1];
			work[j - 1 + iv * *n] = x[2];
			work[j + iv * *n] = x[3];
			//
			//                   Update the right-hand side
			//
			i__2 = j - 2;
			d__1 = -x[0];
			daxpy_(&i__2, &d__1, &t[(j - 1) * t_dim1 + 1], &c__1,
				&work[(iv - 1) * *n + 1], &c__1);
			i__2 = j - 2;
			d__1 = -x[1];
			daxpy_(&i__2, &d__1, &t[j * t_dim1 + 1], &c__1, &work[
				(iv - 1) * *n + 1], &c__1);
			i__2 = j - 2;
			d__1 = -x[2];
			daxpy_(&i__2, &d__1, &t[(j - 1) * t_dim1 + 1], &c__1,
				&work[iv * *n + 1], &c__1);
			i__2 = j - 2;
			d__1 = -x[3];
			daxpy_(&i__2, &d__1, &t[j * t_dim1 + 1], &c__1, &work[
				iv * *n + 1], &c__1);
		    }
L90:
		    ;
		}
		//
		//             Copy the vector x or Q*x to VR and normalize.
		//
		if (! over) {
		    //                ------------------------------
		    //                no back-transform: copy x to VR and normalize.
		    dcopy_(&ki, &work[(iv - 1) * *n + 1], &c__1, &vr[(is - 1)
			    * vr_dim1 + 1], &c__1);
		    dcopy_(&ki, &work[iv * *n + 1], &c__1, &vr[is * vr_dim1 +
			    1], &c__1);
		    emax = 0.;
		    i__2 = ki;
		    for (k = 1; k <= i__2; ++k) {
			// Computing MAX
			d__3 = emax, d__4 = (d__1 = vr[k + (is - 1) * vr_dim1]
				, abs(d__1)) + (d__2 = vr[k + is * vr_dim1],
				abs(d__2));
			emax = max(d__3,d__4);
// L100:
		    }
		    remax = 1. / emax;
		    dscal_(&ki, &remax, &vr[(is - 1) * vr_dim1 + 1], &c__1);
		    dscal_(&ki, &remax, &vr[is * vr_dim1 + 1], &c__1);
		    i__2 = *n;
		    for (k = ki + 1; k <= i__2; ++k) {
			vr[k + (is - 1) * vr_dim1] = 0.;
			vr[k + is * vr_dim1] = 0.;
// L110:
		    }
		} else if (nb == 1) {
		    //                ------------------------------
		    //                version 1: back-transform each vector with GEMV, Q*x.
		    if (ki > 2) {
			i__2 = ki - 2;
			dgemv_("N", n, &i__2, &c_b29, &vr[vr_offset], ldvr, &
				work[(iv - 1) * *n + 1], &c__1, &work[ki - 1
				+ (iv - 1) * *n], &vr[(ki - 1) * vr_dim1 + 1],
				 &c__1);
			i__2 = ki - 2;
			dgemv_("N", n, &i__2, &c_b29, &vr[vr_offset], ldvr, &
				work[iv * *n + 1], &c__1, &work[ki + iv * *n],
				 &vr[ki * vr_dim1 + 1], &c__1);
		    } else {
			dscal_(n, &work[ki - 1 + (iv - 1) * *n], &vr[(ki - 1)
				* vr_dim1 + 1], &c__1);
			dscal_(n, &work[ki + iv * *n], &vr[ki * vr_dim1 + 1],
				&c__1);
		    }
		    emax = 0.;
		    i__2 = *n;
		    for (k = 1; k <= i__2; ++k) {
			// Computing MAX
			d__3 = emax, d__4 = (d__1 = vr[k + (ki - 1) * vr_dim1]
				, abs(d__1)) + (d__2 = vr[k + ki * vr_dim1],
				abs(d__2));
			emax = max(d__3,d__4);
// L120:
		    }
		    remax = 1. / emax;
		    dscal_(n, &remax, &vr[(ki - 1) * vr_dim1 + 1], &c__1);
		    dscal_(n, &remax, &vr[ki * vr_dim1 + 1], &c__1);
		} else {
		    //                ------------------------------
		    //                version 2: back-transform block of vectors with GEMM
		    //                zero out below vector
		    i__2 = *n;
		    for (k = ki + 1; k <= i__2; ++k) {
			work[k + (iv - 1) * *n] = 0.;
			work[k + iv * *n] = 0.;
		    }
		    iscomplex[iv - 2] = -ip;
		    iscomplex[iv - 1] = ip;
		    --iv;
		    //                back-transform and normalization is done below
		}
	    }
	    if (nb > 1) {
		//             --------------------------------------------------------
		//             Blocked version of back-transform
		//             For complex case, KI2 includes both vectors (KI-1 and KI)
		if (ip == 0) {
		    ki2 = ki;
		} else {
		    ki2 = ki - 1;
		}
		//             Columns IV:NB of work are valid vectors.
		//             When the number of vectors stored reaches NB-1 or NB,
		//             or if this was last vector, do the GEMM
		if (iv <= 2 || ki2 == 1) {
		    i__2 = nb - iv + 1;
		    i__3 = ki2 + nb - iv;
		    dgemm_("N", "N", n, &i__2, &i__3, &c_b29, &vr[vr_offset],
			    ldvr, &work[iv * *n + 1], n, &c_b17, &work[(nb +
			    iv) * *n + 1], n);
		    //                normalize vectors
		    i__2 = nb;
		    for (k = iv; k <= i__2; ++k) {
			if (iscomplex[k - 1] == 0) {
			    //                      real eigenvector
			    ii = idamax_(n, &work[(nb + k) * *n + 1], &c__1);
			    remax = 1. / (d__1 = work[ii + (nb + k) * *n],
				    abs(d__1));
			} else if (iscomplex[k - 1] == 1) {
			    //                      first eigenvector of conjugate pair
			    emax = 0.;
			    i__3 = *n;
			    for (ii = 1; ii <= i__3; ++ii) {
				// Computing MAX
				d__3 = emax, d__4 = (d__1 = work[ii + (nb + k)
					 * *n], abs(d__1)) + (d__2 = work[ii
					+ (nb + k + 1) * *n], abs(d__2));
				emax = max(d__3,d__4);
			    }
			    remax = 1. / emax;
			    //                   else if ISCOMPLEX(K).EQ.-1
			    //                      second eigenvector of conjugate pair
			    //                      reuse same REMAX as previous K
			}
			dscal_(n, &remax, &work[(nb + k) * *n + 1], &c__1);
		    }
		    i__2 = nb - iv + 1;
		    dlacpy_("F", n, &i__2, &work[(nb + iv) * *n + 1], n, &vr[
			    ki2 * vr_dim1 + 1], ldvr);
		    iv = nb;
		} else {
		    --iv;
		}
	    }
	    //
	    // blocked back-transform
	    --is;
	    if (ip != 0) {
		--is;
	    }
L140:
	    ;
	}
    }
    if (leftv) {
	//
	//       ============================================================
	//       Compute left eigenvectors.
	//
	//       IV is index of column in current block.
	//       For complex left vector, uses IV for real part and IV+1 for complex part.
	//       Non-blocked version always uses IV=1;
	//       blocked     version starts with IV=1, goes up to NB-1 or NB.
	//       (Note the "0-th" column is used for 1-norms computed above.)
	iv = 1;
	ip = 0;
	is = 1;
	i__2 = *n;
	for (ki = 1; ki <= i__2; ++ki) {
	    if (ip == 1) {
		//             previous iteration (ki-1) was first of conjugate pair,
		//             so this ki is second of conjugate pair; skip to end of loop
		ip = -1;
		goto L260;
	    } else if (ki == *n) {
		//             last column, so this ki must be real eigenvalue
		ip = 0;
	    } else if (t[ki + 1 + ki * t_dim1] == 0.) {
		//             zero on sub-diagonal, so this ki is real eigenvalue
		ip = 0;
	    } else {
		//             non-zero on sub-diagonal, so this ki is first of conjugate pair
		ip = 1;
	    }
	    if (somev) {
		if (! select[ki]) {
		    goto L260;
		}
	    }
	    //
	    //          Compute the KI-th eigenvalue (WR,WI).
	    //
	    wr = t[ki + ki * t_dim1];
	    wi = 0.;
	    if (ip != 0) {
		wi = sqrt((d__1 = t[ki + (ki + 1) * t_dim1], abs(d__1))) *
			sqrt((d__2 = t[ki + 1 + ki * t_dim1], abs(d__2)));
	    }
	    // Computing MAX
	    d__1 = ulp * (abs(wr) + abs(wi));
	    smin = max(d__1,smlnum);
	    if (ip == 0) {
		//
		//             --------------------------------------------------------
		//             Real left eigenvector
		//
		work[ki + iv * *n] = 1.;
		//
		//             Form right-hand side.
		//
		i__3 = *n;
		for (k = ki + 1; k <= i__3; ++k) {
		    work[k + iv * *n] = -t[ki + k * t_dim1];
// L160:
		}
		//
		//             Solve transposed quasi-triangular system:
		//             [ T(KI+1:N,KI+1:N) - WR ]**T * X = SCALE*WORK
		//
		vmax = 1.;
		vcrit = bignum;
		jnxt = ki + 1;
		i__3 = *n;
		for (j = ki + 1; j <= i__3; ++j) {
		    if (j < jnxt) {
			goto L170;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j + 1;
		    if (j < *n) {
			if (t[j + 1 + j * t_dim1] != 0.) {
			    j2 = j + 1;
			    jnxt = j + 2;
			}
		    }
		    if (j1 == j2) {
			//
			//                   1-by-1 diagonal block
			//
			//                   Scale if necessary to avoid overflow when forming
			//                   the right-hand side.
			//
			if (work[j] > vcrit) {
			    rec = 1. / vmax;
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &rec, &work[ki + iv * *n], &c__1);
			    vmax = 1.;
			    vcrit = bignum;
			}
			i__4 = j - ki - 1;
			work[j + iv * *n] -= ddot_(&i__4, &t[ki + 1 + j *
				t_dim1], &c__1, &work[ki + 1 + iv * *n], &
				c__1);
			//
			//                   Solve [ T(J,J) - WR ]**T * X = WORK
			//
			dlaln2_(&c_false, &c__1, &c__1, &smin, &c_b29, &t[j +
				j * t_dim1], ldt, &c_b29, &c_b29, &work[j +
				iv * *n], n, &wr, &c_b17, x, &c__2, &scale, &
				xnorm, &ierr);
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &scale, &work[ki + iv * *n], &c__1);
			}
			work[j + iv * *n] = x[0];
			// Computing MAX
			d__2 = (d__1 = work[j + iv * *n], abs(d__1));
			vmax = max(d__2,vmax);
			vcrit = bignum / vmax;
		    } else {
			//
			//                   2-by-2 diagonal block
			//
			//                   Scale if necessary to avoid overflow when forming
			//                   the right-hand side.
			//
			// Computing MAX
			d__1 = work[j], d__2 = work[j + 1];
			beta = max(d__1,d__2);
			if (beta > vcrit) {
			    rec = 1. / vmax;
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &rec, &work[ki + iv * *n], &c__1);
			    vmax = 1.;
			    vcrit = bignum;
			}
			i__4 = j - ki - 1;
			work[j + iv * *n] -= ddot_(&i__4, &t[ki + 1 + j *
				t_dim1], &c__1, &work[ki + 1 + iv * *n], &
				c__1);
			i__4 = j - ki - 1;
			work[j + 1 + iv * *n] -= ddot_(&i__4, &t[ki + 1 + (j
				+ 1) * t_dim1], &c__1, &work[ki + 1 + iv * *n]
				, &c__1);
			//
			//                   Solve
			//                   [ T(J,J)-WR   T(J,J+1)      ]**T * X = SCALE*( WORK1 )
			//                   [ T(J+1,J)    T(J+1,J+1)-WR ]                ( WORK2 )
			//
			dlaln2_(&c_true, &c__2, &c__1, &smin, &c_b29, &t[j +
				j * t_dim1], ldt, &c_b29, &c_b29, &work[j +
				iv * *n], n, &wr, &c_b17, x, &c__2, &scale, &
				xnorm, &ierr);
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &scale, &work[ki + iv * *n], &c__1);
			}
			work[j + iv * *n] = x[0];
			work[j + 1 + iv * *n] = x[1];
			//
			// Computing MAX
			d__3 = (d__1 = work[j + iv * *n], abs(d__1)), d__4 = (
				d__2 = work[j + 1 + iv * *n], abs(d__2)),
				d__3 = max(d__3,d__4);
			vmax = max(d__3,vmax);
			vcrit = bignum / vmax;
		    }
L170:
		    ;
		}
		//
		//             Copy the vector x or Q*x to VL and normalize.
		//
		if (! over) {
		    //                ------------------------------
		    //                no back-transform: copy x to VL and normalize.
		    i__3 = *n - ki + 1;
		    dcopy_(&i__3, &work[ki + iv * *n], &c__1, &vl[ki + is *
			    vl_dim1], &c__1);
		    i__3 = *n - ki + 1;
		    ii = idamax_(&i__3, &vl[ki + is * vl_dim1], &c__1) + ki -
			    1;
		    remax = 1. / (d__1 = vl[ii + is * vl_dim1], abs(d__1));
		    i__3 = *n - ki + 1;
		    dscal_(&i__3, &remax, &vl[ki + is * vl_dim1], &c__1);
		    i__3 = ki - 1;
		    for (k = 1; k <= i__3; ++k) {
			vl[k + is * vl_dim1] = 0.;
// L180:
		    }
		} else if (nb == 1) {
		    //                ------------------------------
		    //                version 1: back-transform each vector with GEMV, Q*x.
		    if (ki < *n) {
			i__3 = *n - ki;
			dgemv_("N", n, &i__3, &c_b29, &vl[(ki + 1) * vl_dim1
				+ 1], ldvl, &work[ki + 1 + iv * *n], &c__1, &
				work[ki + iv * *n], &vl[ki * vl_dim1 + 1], &
				c__1);
		    }
		    ii = idamax_(n, &vl[ki * vl_dim1 + 1], &c__1);
		    remax = 1. / (d__1 = vl[ii + ki * vl_dim1], abs(d__1));
		    dscal_(n, &remax, &vl[ki * vl_dim1 + 1], &c__1);
		} else {
		    //                ------------------------------
		    //                version 2: back-transform block of vectors with GEMM
		    //                zero out above vector
		    //                could go from KI-NV+1 to KI-1
		    i__3 = ki - 1;
		    for (k = 1; k <= i__3; ++k) {
			work[k + iv * *n] = 0.;
		    }
		    iscomplex[iv - 1] = ip;
		    //                back-transform and normalization is done below
		}
	    } else {
		//
		//             --------------------------------------------------------
		//             Complex left eigenvector.
		//
		//             Initial solve:
		//             [ ( T(KI,KI)    T(KI,KI+1)  )**T - (WR - I* WI) ]*X = 0.
		//             [ ( T(KI+1,KI) T(KI+1,KI+1) )                   ]
		//
		if ((d__1 = t[ki + (ki + 1) * t_dim1], abs(d__1)) >= (d__2 =
			t[ki + 1 + ki * t_dim1], abs(d__2))) {
		    work[ki + iv * *n] = wi / t[ki + (ki + 1) * t_dim1];
		    work[ki + 1 + (iv + 1) * *n] = 1.;
		} else {
		    work[ki + iv * *n] = 1.;
		    work[ki + 1 + (iv + 1) * *n] = -wi / t[ki + 1 + ki *
			    t_dim1];
		}
		work[ki + 1 + iv * *n] = 0.;
		work[ki + (iv + 1) * *n] = 0.;
		//
		//             Form right-hand side.
		//
		i__3 = *n;
		for (k = ki + 2; k <= i__3; ++k) {
		    work[k + iv * *n] = -work[ki + iv * *n] * t[ki + k *
			    t_dim1];
		    work[k + (iv + 1) * *n] = -work[ki + 1 + (iv + 1) * *n] *
			    t[ki + 1 + k * t_dim1];
// L190:
		}
		//
		//             Solve transposed quasi-triangular system:
		//             [ T(KI+2:N,KI+2:N)**T - (WR-i*WI) ]*X = WORK1+i*WORK2
		//
		vmax = 1.;
		vcrit = bignum;
		jnxt = ki + 2;
		i__3 = *n;
		for (j = ki + 2; j <= i__3; ++j) {
		    if (j < jnxt) {
			goto L200;
		    }
		    j1 = j;
		    j2 = j;
		    jnxt = j + 1;
		    if (j < *n) {
			if (t[j + 1 + j * t_dim1] != 0.) {
			    j2 = j + 1;
			    jnxt = j + 2;
			}
		    }
		    if (j1 == j2) {
			//
			//                   1-by-1 diagonal block
			//
			//                   Scale if necessary to avoid overflow when
			//                   forming the right-hand side elements.
			//
			if (work[j] > vcrit) {
			    rec = 1. / vmax;
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &rec, &work[ki + iv * *n], &c__1);
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &rec, &work[ki + (iv + 1) * *n], &
				    c__1);
			    vmax = 1.;
			    vcrit = bignum;
			}
			i__4 = j - ki - 2;
			work[j + iv * *n] -= ddot_(&i__4, &t[ki + 2 + j *
				t_dim1], &c__1, &work[ki + 2 + iv * *n], &
				c__1);
			i__4 = j - ki - 2;
			work[j + (iv + 1) * *n] -= ddot_(&i__4, &t[ki + 2 + j
				* t_dim1], &c__1, &work[ki + 2 + (iv + 1) * *
				n], &c__1);
			//
			//                   Solve [ T(J,J)-(WR-i*WI) ]*(X11+i*X12)= WK+I*WK2
			//
			d__1 = -wi;
			dlaln2_(&c_false, &c__1, &c__2, &smin, &c_b29, &t[j +
				j * t_dim1], ldt, &c_b29, &c_b29, &work[j +
				iv * *n], n, &wr, &d__1, x, &c__2, &scale, &
				xnorm, &ierr);
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &scale, &work[ki + iv * *n], &c__1);
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &scale, &work[ki + (iv + 1) * *n], &
				    c__1);
			}
			work[j + iv * *n] = x[0];
			work[j + (iv + 1) * *n] = x[2];
			// Computing MAX
			d__3 = (d__1 = work[j + iv * *n], abs(d__1)), d__4 = (
				d__2 = work[j + (iv + 1) * *n], abs(d__2)),
				d__3 = max(d__3,d__4);
			vmax = max(d__3,vmax);
			vcrit = bignum / vmax;
		    } else {
			//
			//                   2-by-2 diagonal block
			//
			//                   Scale if necessary to avoid overflow when forming
			//                   the right-hand side elements.
			//
			// Computing MAX
			d__1 = work[j], d__2 = work[j + 1];
			beta = max(d__1,d__2);
			if (beta > vcrit) {
			    rec = 1. / vmax;
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &rec, &work[ki + iv * *n], &c__1);
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &rec, &work[ki + (iv + 1) * *n], &
				    c__1);
			    vmax = 1.;
			    vcrit = bignum;
			}
			i__4 = j - ki - 2;
			work[j + iv * *n] -= ddot_(&i__4, &t[ki + 2 + j *
				t_dim1], &c__1, &work[ki + 2 + iv * *n], &
				c__1);
			i__4 = j - ki - 2;
			work[j + (iv + 1) * *n] -= ddot_(&i__4, &t[ki + 2 + j
				* t_dim1], &c__1, &work[ki + 2 + (iv + 1) * *
				n], &c__1);
			i__4 = j - ki - 2;
			work[j + 1 + iv * *n] -= ddot_(&i__4, &t[ki + 2 + (j
				+ 1) * t_dim1], &c__1, &work[ki + 2 + iv * *n]
				, &c__1);
			i__4 = j - ki - 2;
			work[j + 1 + (iv + 1) * *n] -= ddot_(&i__4, &t[ki + 2
				+ (j + 1) * t_dim1], &c__1, &work[ki + 2 + (
				iv + 1) * *n], &c__1);
			//
			//                   Solve 2-by-2 complex linear equation
			//                   [ (T(j,j)   T(j,j+1)  )**T - (wr-i*wi)*I ]*X = SCALE*B
			//                   [ (T(j+1,j) T(j+1,j+1))                  ]
			//
			d__1 = -wi;
			dlaln2_(&c_true, &c__2, &c__2, &smin, &c_b29, &t[j +
				j * t_dim1], ldt, &c_b29, &c_b29, &work[j +
				iv * *n], n, &wr, &d__1, x, &c__2, &scale, &
				xnorm, &ierr);
			//
			//                   Scale if necessary
			//
			if (scale != 1.) {
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &scale, &work[ki + iv * *n], &c__1);
			    i__4 = *n - ki + 1;
			    dscal_(&i__4, &scale, &work[ki + (iv + 1) * *n], &
				    c__1);
			}
			work[j + iv * *n] = x[0];
			work[j + (iv + 1) * *n] = x[2];
			work[j + 1 + iv * *n] = x[1];
			work[j + 1 + (iv + 1) * *n] = x[3];
			// Computing MAX
			d__1 = abs(x[0]), d__2 = abs(x[2]), d__1 = max(d__1,
				d__2), d__2 = abs(x[1]), d__1 = max(d__1,d__2)
				, d__2 = abs(x[3]), d__1 = max(d__1,d__2);
			vmax = max(d__1,vmax);
			vcrit = bignum / vmax;
		    }
L200:
		    ;
		}
		//
		//             Copy the vector x or Q*x to VL and normalize.
		//
		if (! over) {
		    //                ------------------------------
		    //                no back-transform: copy x to VL and normalize.
		    i__3 = *n - ki + 1;
		    dcopy_(&i__3, &work[ki + iv * *n], &c__1, &vl[ki + is *
			    vl_dim1], &c__1);
		    i__3 = *n - ki + 1;
		    dcopy_(&i__3, &work[ki + (iv + 1) * *n], &c__1, &vl[ki + (
			    is + 1) * vl_dim1], &c__1);
		    emax = 0.;
		    i__3 = *n;
		    for (k = ki; k <= i__3; ++k) {
			// Computing MAX
			d__3 = emax, d__4 = (d__1 = vl[k + is * vl_dim1], abs(
				d__1)) + (d__2 = vl[k + (is + 1) * vl_dim1],
				abs(d__2));
			emax = max(d__3,d__4);
// L220:
		    }
		    remax = 1. / emax;
		    i__3 = *n - ki + 1;
		    dscal_(&i__3, &remax, &vl[ki + is * vl_dim1], &c__1);
		    i__3 = *n - ki + 1;
		    dscal_(&i__3, &remax, &vl[ki + (is + 1) * vl_dim1], &c__1)
			    ;
		    i__3 = ki - 1;
		    for (k = 1; k <= i__3; ++k) {
			vl[k + is * vl_dim1] = 0.;
			vl[k + (is + 1) * vl_dim1] = 0.;
// L230:
		    }
		} else if (nb == 1) {
		    //                ------------------------------
		    //                version 1: back-transform each vector with GEMV, Q*x.
		    if (ki < *n - 1) {
			i__3 = *n - ki - 1;
			dgemv_("N", n, &i__3, &c_b29, &vl[(ki + 2) * vl_dim1
				+ 1], ldvl, &work[ki + 2 + iv * *n], &c__1, &
				work[ki + iv * *n], &vl[ki * vl_dim1 + 1], &
				c__1);
			i__3 = *n - ki - 1;
			dgemv_("N", n, &i__3, &c_b29, &vl[(ki + 2) * vl_dim1
				+ 1], ldvl, &work[ki + 2 + (iv + 1) * *n], &
				c__1, &work[ki + 1 + (iv + 1) * *n], &vl[(ki
				+ 1) * vl_dim1 + 1], &c__1);
		    } else {
			dscal_(n, &work[ki + iv * *n], &vl[ki * vl_dim1 + 1],
				&c__1);
			dscal_(n, &work[ki + 1 + (iv + 1) * *n], &vl[(ki + 1)
				* vl_dim1 + 1], &c__1);
		    }
		    emax = 0.;
		    i__3 = *n;
		    for (k = 1; k <= i__3; ++k) {
			// Computing MAX
			d__3 = emax, d__4 = (d__1 = vl[k + ki * vl_dim1], abs(
				d__1)) + (d__2 = vl[k + (ki + 1) * vl_dim1],
				abs(d__2));
			emax = max(d__3,d__4);
// L240:
		    }
		    remax = 1. / emax;
		    dscal_(n, &remax, &vl[ki * vl_dim1 + 1], &c__1);
		    dscal_(n, &remax, &vl[(ki + 1) * vl_dim1 + 1], &c__1);
		} else {
		    //                ------------------------------
		    //                version 2: back-transform block of vectors with GEMM
		    //                zero out above vector
		    //                could go from KI-NV+1 to KI-1
		    i__3 = ki - 1;
		    for (k = 1; k <= i__3; ++k) {
			work[k + iv * *n] = 0.;
			work[k + (iv + 1) * *n] = 0.;
		    }
		    iscomplex[iv - 1] = ip;
		    iscomplex[iv] = -ip;
		    ++iv;
		    //                back-transform and normalization is done below
		}
	    }
	    if (nb > 1) {
		//             --------------------------------------------------------
		//             Blocked version of back-transform
		//             For complex case, KI2 includes both vectors (KI and KI+1)
		if (ip == 0) {
		    ki2 = ki;
		} else {
		    ki2 = ki + 1;
		}
		//             Columns 1:IV of work are valid vectors.
		//             When the number of vectors stored reaches NB-1 or NB,
		//             or if this was last vector, do the GEMM
		if (iv >= nb - 1 || ki2 == *n) {
		    i__3 = *n - ki2 + iv;
		    dgemm_("N", "N", n, &iv, &i__3, &c_b29, &vl[(ki2 - iv + 1)
			     * vl_dim1 + 1], ldvl, &work[ki2 - iv + 1 + *n],
			    n, &c_b17, &work[(nb + 1) * *n + 1], n);
		    //                normalize vectors
		    i__3 = iv;
		    for (k = 1; k <= i__3; ++k) {
			if (iscomplex[k - 1] == 0) {
			    //                      real eigenvector
			    ii = idamax_(n, &work[(nb + k) * *n + 1], &c__1);
			    remax = 1. / (d__1 = work[ii + (nb + k) * *n],
				    abs(d__1));
			} else if (iscomplex[k - 1] == 1) {
			    //                      first eigenvector of conjugate pair
			    emax = 0.;
			    i__4 = *n;
			    for (ii = 1; ii <= i__4; ++ii) {
				// Computing MAX
				d__3 = emax, d__4 = (d__1 = work[ii + (nb + k)
					 * *n], abs(d__1)) + (d__2 = work[ii
					+ (nb + k + 1) * *n], abs(d__2));
				emax = max(d__3,d__4);
			    }
			    remax = 1. / emax;
			    //                   else if ISCOMPLEX(K).EQ.-1
			    //                      second eigenvector of conjugate pair
			    //                      reuse same REMAX as previous K
			}
			dscal_(n, &remax, &work[(nb + k) * *n + 1], &c__1);
		    }
		    dlacpy_("F", n, &iv, &work[(nb + 1) * *n + 1], n, &vl[(
			    ki2 - iv + 1) * vl_dim1 + 1], ldvl);
		    iv = 1;
		} else {
		    ++iv;
		}
	    }
	    //
	    // blocked back-transform
	    ++is;
	    if (ip != 0) {
		++is;
	    }
L260:
	    ;
	}
    }
    return 0;
    //
    //    End of DTREVC3
    //
} // dtrevc3_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DTREXC
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DTREXC + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dtrexc.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dtrexc.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dtrexc.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DTREXC( COMPQ, N, T, LDT, Q, LDQ, IFST, ILST, WORK,
//                         INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          COMPQ
//      INTEGER            IFST, ILST, INFO, LDQ, LDT, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Q( LDQ, * ), T( LDT, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DTREXC reorders the real Schur factorization of a real matrix
//> A = Q*T*Q**T, so that the diagonal block of T with row index IFST is
//> moved to row ILST.
//>
//> The real Schur form T is reordered by an orthogonal similarity
//> transformation Z**T*T*Z, and optionally the matrix Q of Schur vectors
//> is updated by postmultiplying it with Z.
//>
//> T must be in Schur canonical form (as returned by DHSEQR), that is,
//> block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; each
//> 2-by-2 diagonal block has its diagonal elements equal and its
//> off-diagonal elements of opposite sign.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] COMPQ
//> \verbatim
//>          COMPQ is CHARACTER*1
//>          = 'V':  update the matrix Q of Schur vectors;
//>          = 'N':  do not update Q.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix T. N >= 0.
//>          If N == 0 arguments ILST and IFST may be any value.
//> \endverbatim
//>
//> \param[in,out] T
//> \verbatim
//>          T is DOUBLE PRECISION array, dimension (LDT,N)
//>          On entry, the upper quasi-triangular matrix T, in Schur
//>          Schur canonical form.
//>          On exit, the reordered upper quasi-triangular matrix, again
//>          in Schur canonical form.
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of the array T. LDT >= max(1,N).
//> \endverbatim
//>
//> \param[in,out] Q
//> \verbatim
//>          Q is DOUBLE PRECISION array, dimension (LDQ,N)
//>          On entry, if COMPQ = 'V', the matrix Q of Schur vectors.
//>          On exit, if COMPQ = 'V', Q has been postmultiplied by the
//>          orthogonal transformation matrix Z which reorders T.
//>          If COMPQ = 'N', Q is not referenced.
//> \endverbatim
//>
//> \param[in] LDQ
//> \verbatim
//>          LDQ is INTEGER
//>          The leading dimension of the array Q.  LDQ >= 1, and if
//>          COMPQ = 'V', LDQ >= max(1,N).
//> \endverbatim
//>
//> \param[in,out] IFST
//> \verbatim
//>          IFST is INTEGER
//> \endverbatim
//>
//> \param[in,out] ILST
//> \verbatim
//>          ILST is INTEGER
//>
//>          Specify the reordering of the diagonal blocks of T.
//>          The block with row index IFST is moved to row ILST, by a
//>          sequence of transpositions between adjacent blocks.
//>          On exit, if IFST pointed on entry to the second row of a
//>          2-by-2 block, it is changed to point to the first row; ILST
//>          always points to the first row of the block in its final
//>          position (which may differ from its input value by +1 or -1).
//>          1 <= IFST <= N; 1 <= ILST <= N.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          = 1:  two adjacent blocks were too close to swap (the problem
//>                is very ill-conditioned); T may have been partially
//>                reordered, and ILST points to the first row of the
//>                current position of the block being moved.
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
/* Subroutine */ int dtrexc_(char *compq, int *n, double *t, int *ldt, double
	*q, int *ldq, int *ifst, int *ilst, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__2 = 2;

    // System generated locals
    int q_dim1, q_offset, t_dim1, t_offset, i__1;

    // Local variables
    int nbf, nbl, here;
    extern int lsame_(char *, char *);
    int wantq;
    extern /* Subroutine */ int dlaexc_(int *, int *, double *, int *, double
	    *, int *, int *, int *, int *, double *, int *), xerbla_(char *,
	    int *);
    int nbnext;

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
    //    Decode and test the input arguments.
    //
    // Parameter adjustments
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;
    q_dim1 = *ldq;
    q_offset = 1 + q_dim1;
    q -= q_offset;
    --work;

    // Function Body
    *info = 0;
    wantq = lsame_(compq, "V");
    if (! wantq && ! lsame_(compq, "N")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*ldt < max(1,*n)) {
	*info = -4;
    } else if (*ldq < 1 || wantq && *ldq < max(1,*n)) {
	*info = -6;
    } else if ((*ifst < 1 || *ifst > *n) && *n > 0) {
	*info = -7;
    } else if ((*ilst < 1 || *ilst > *n) && *n > 0) {
	*info = -8;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DTREXC", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n <= 1) {
	return 0;
    }
    //
    //    Determine the first row of specified block
    //    and find out it is 1 by 1 or 2 by 2.
    //
    if (*ifst > 1) {
	if (t[*ifst + (*ifst - 1) * t_dim1] != 0.) {
	    --(*ifst);
	}
    }
    nbf = 1;
    if (*ifst < *n) {
	if (t[*ifst + 1 + *ifst * t_dim1] != 0.) {
	    nbf = 2;
	}
    }
    //
    //    Determine the first row of the final block
    //    and find out it is 1 by 1 or 2 by 2.
    //
    if (*ilst > 1) {
	if (t[*ilst + (*ilst - 1) * t_dim1] != 0.) {
	    --(*ilst);
	}
    }
    nbl = 1;
    if (*ilst < *n) {
	if (t[*ilst + 1 + *ilst * t_dim1] != 0.) {
	    nbl = 2;
	}
    }
    if (*ifst == *ilst) {
	return 0;
    }
    if (*ifst < *ilst) {
	//
	//       Update ILST
	//
	if (nbf == 2 && nbl == 1) {
	    --(*ilst);
	}
	if (nbf == 1 && nbl == 2) {
	    ++(*ilst);
	}
	here = *ifst;
L10:
	//
	//       Swap block with next one below
	//
	if (nbf == 1 || nbf == 2) {
	    //
	    //          Current block either 1 by 1 or 2 by 2
	    //
	    nbnext = 1;
	    if (here + nbf + 1 <= *n) {
		if (t[here + nbf + 1 + (here + nbf) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &here, &
		    nbf, &nbnext, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    here += nbnext;
	    //
	    //          Test if 2 by 2 block breaks into two 1 by 1 blocks
	    //
	    if (nbf == 2) {
		if (t[here + 1 + here * t_dim1] == 0.) {
		    nbf = 3;
		}
	    }
	} else {
	    //
	    //          Current block consists of two 1 by 1 blocks each of which
	    //          must be swapped individually
	    //
	    nbnext = 1;
	    if (here + 3 <= *n) {
		if (t[here + 3 + (here + 2) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    i__1 = here + 1;
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &i__1, &
		    c__1, &nbnext, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    if (nbnext == 1) {
		//
		//             Swap two 1 by 1 blocks, no problems possible
		//
		dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			here, &c__1, &nbnext, &work[1], info);
		++here;
	    } else {
		//
		//             Recompute NBNEXT in case 2 by 2 split
		//
		if (t[here + 2 + (here + 1) * t_dim1] == 0.) {
		    nbnext = 1;
		}
		if (nbnext == 2) {
		    //
		    //                2 by 2 Block did not split
		    //
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    here, &c__1, &nbnext, &work[1], info);
		    if (*info != 0) {
			*ilst = here;
			return 0;
		    }
		    here += 2;
		} else {
		    //
		    //                2 by 2 Block did split
		    //
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    here, &c__1, &c__1, &work[1], info);
		    i__1 = here + 1;
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    i__1, &c__1, &c__1, &work[1], info);
		    here += 2;
		}
	    }
	}
	if (here < *ilst) {
	    goto L10;
	}
    } else {
	here = *ifst;
L20:
	//
	//       Swap block with next one above
	//
	if (nbf == 1 || nbf == 2) {
	    //
	    //          Current block either 1 by 1 or 2 by 2
	    //
	    nbnext = 1;
	    if (here >= 3) {
		if (t[here - 1 + (here - 2) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    i__1 = here - nbnext;
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &i__1, &
		    nbnext, &nbf, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    here -= nbnext;
	    //
	    //          Test if 2 by 2 block breaks into two 1 by 1 blocks
	    //
	    if (nbf == 2) {
		if (t[here + 1 + here * t_dim1] == 0.) {
		    nbf = 3;
		}
	    }
	} else {
	    //
	    //          Current block consists of two 1 by 1 blocks each of which
	    //          must be swapped individually
	    //
	    nbnext = 1;
	    if (here >= 3) {
		if (t[here - 1 + (here - 2) * t_dim1] != 0.) {
		    nbnext = 2;
		}
	    }
	    i__1 = here - nbnext;
	    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &i__1, &
		    nbnext, &c__1, &work[1], info);
	    if (*info != 0) {
		*ilst = here;
		return 0;
	    }
	    if (nbnext == 1) {
		//
		//             Swap two 1 by 1 blocks, no problems possible
		//
		dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			here, &nbnext, &c__1, &work[1], info);
		--here;
	    } else {
		//
		//             Recompute NBNEXT in case 2 by 2 split
		//
		if (t[here + (here - 1) * t_dim1] == 0.) {
		    nbnext = 1;
		}
		if (nbnext == 2) {
		    //
		    //                2 by 2 Block did not split
		    //
		    i__1 = here - 1;
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    i__1, &c__2, &c__1, &work[1], info);
		    if (*info != 0) {
			*ilst = here;
			return 0;
		    }
		    here += -2;
		} else {
		    //
		    //                2 by 2 Block did split
		    //
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    here, &c__1, &c__1, &work[1], info);
		    i__1 = here - 1;
		    dlaexc_(&wantq, n, &t[t_offset], ldt, &q[q_offset], ldq, &
			    i__1, &c__1, &c__1, &work[1], info);
		    here += -2;
		}
	    }
	}
	if (here > *ilst) {
	    goto L20;
	}
    }
    *ilst = here;
    return 0;
    //
    //    End of DTREXC
    //
} // dtrexc_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b IDAMAX
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      INTEGER FUNCTION IDAMAX(N,DX,INCX)
//
//      .. Scalar Arguments ..
//      INTEGER INCX,N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION DX(*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//>    IDAMAX finds the index of the first element having maximum absolute value.
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
//> \param[in] DX
//> \verbatim
//>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
//> \endverbatim
//>
//> \param[in] INCX
//> \verbatim
//>          INCX is INTEGER
//>         storage spacing between elements of DX
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
int idamax_(int *n, double *dx, int *incx)
{
    // System generated locals
    int ret_val, i__1;
    double d__1;

    // Local variables
    int i__, ix;
    double dmax__;

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
    --dx;

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
	dmax__ = abs(dx[1]);
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if ((d__1 = dx[i__], abs(d__1)) > dmax__) {
		ret_val = i__;
		dmax__ = (d__1 = dx[i__], abs(d__1));
	    }
	}
    } else {
	//
	//       code for increment not equal to 1
	//
	ix = 1;
	dmax__ = abs(dx[1]);
	ix += *incx;
	i__1 = *n;
	for (i__ = 2; i__ <= i__1; ++i__) {
	    if ((d__1 = dx[ix], abs(d__1)) > dmax__) {
		ret_val = i__;
		dmax__ = (d__1 = dx[ix], abs(d__1));
	    }
	    ix += *incx;
	}
    }
    return ret_val;
} // idamax_

