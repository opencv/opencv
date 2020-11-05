/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLARFT forms the triangular factor T of a block reflector H = I - vtvH
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARFT + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarft.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarft.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarft.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARFT( DIRECT, STOREV, N, K, V, LDV, TAU, T, LDT )
//
//      .. Scalar Arguments ..
//      CHARACTER          DIRECT, STOREV
//      INTEGER            K, LDT, LDV, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   T( LDT, * ), TAU( * ), V( LDV, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLARFT forms the triangular factor T of a real block reflector H
//> of order n, which is defined as a product of k elementary reflectors.
//>
//> If DIRECT = 'F', H = H(1) H(2) . . . H(k) and T is upper triangular;
//>
//> If DIRECT = 'B', H = H(k) . . . H(2) H(1) and T is lower triangular.
//>
//> If STOREV = 'C', the vector which defines the elementary reflector
//> H(i) is stored in the i-th column of the array V, and
//>
//>    H  =  I - V * T * V**T
//>
//> If STOREV = 'R', the vector which defines the elementary reflector
//> H(i) is stored in the i-th row of the array V, and
//>
//>    H  =  I - V**T * T * V
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] DIRECT
//> \verbatim
//>          DIRECT is CHARACTER*1
//>          Specifies the order in which the elementary reflectors are
//>          multiplied to form the block reflector:
//>          = 'F': H = H(1) H(2) . . . H(k) (Forward)
//>          = 'B': H = H(k) . . . H(2) H(1) (Backward)
//> \endverbatim
//>
//> \param[in] STOREV
//> \verbatim
//>          STOREV is CHARACTER*1
//>          Specifies how the vectors which define the elementary
//>          reflectors are stored (see also Further Details):
//>          = 'C': columnwise
//>          = 'R': rowwise
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the block reflector H. N >= 0.
//> \endverbatim
//>
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The order of the triangular factor T (= the number of
//>          elementary reflectors). K >= 1.
//> \endverbatim
//>
//> \param[in] V
//> \verbatim
//>          V is DOUBLE PRECISION array, dimension
//>                               (LDV,K) if STOREV = 'C'
//>                               (LDV,N) if STOREV = 'R'
//>          The matrix V. See further details.
//> \endverbatim
//>
//> \param[in] LDV
//> \verbatim
//>          LDV is INTEGER
//>          The leading dimension of the array V.
//>          If STOREV = 'C', LDV >= max(1,N); if STOREV = 'R', LDV >= K.
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i).
//> \endverbatim
//>
//> \param[out] T
//> \verbatim
//>          T is DOUBLE PRECISION array, dimension (LDT,K)
//>          The k by k triangular factor T of the block reflector.
//>          If DIRECT = 'F', T is upper triangular; if DIRECT = 'B', T is
//>          lower triangular. The rest of the array is not used.
//> \endverbatim
//>
//> \param[in] LDT
//> \verbatim
//>          LDT is INTEGER
//>          The leading dimension of the array T. LDT >= K.
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
//>  The shape of the matrix V and the storage of the vectors which define
//>  the H(i) is best illustrated by the following example with n = 5 and
//>  k = 3. The elements equal to 1 are not stored.
//>
//>  DIRECT = 'F' and STOREV = 'C':         DIRECT = 'F' and STOREV = 'R':
//>
//>               V = (  1       )                 V = (  1 v1 v1 v1 v1 )
//>                   ( v1  1    )                     (     1 v2 v2 v2 )
//>                   ( v1 v2  1 )                     (        1 v3 v3 )
//>                   ( v1 v2 v3 )
//>                   ( v1 v2 v3 )
//>
//>  DIRECT = 'B' and STOREV = 'C':         DIRECT = 'B' and STOREV = 'R':
//>
//>               V = ( v1 v2 v3 )                 V = ( v1 v1  1       )
//>                   ( v1 v2 v3 )                     ( v2 v2 v2  1    )
//>                   (  1 v2 v3 )                     ( v3 v3 v3 v3  1 )
//>                   (     1 v3 )
//>                   (        1 )
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlarft_(char *direct, char *storev, int *n, int *k,
	double *v, int *ldv, double *tau, double *t, int *ldt)
{
    // Table of constant values
    int c__1 = 1;
    double c_b7 = 1.;

    // System generated locals
    int t_dim1, t_offset, v_dim1, v_offset, i__1, i__2, i__3;
    double d__1;

    // Local variables
    int i__, j, prevlastv;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, int *, int *, double *, double
	    *, int *, double *, int *, double *, double *, int *);
    int lastv;
    extern /* Subroutine */ int dtrmv_(char *, char *, char *, int *, double *
	    , int *, double *, int *);

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
    //    Quick return if possible
    //
    // Parameter adjustments
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --tau;
    t_dim1 = *ldt;
    t_offset = 1 + t_dim1;
    t -= t_offset;

    // Function Body
    if (*n == 0) {
	return 0;
    }
    if (lsame_(direct, "F")) {
	prevlastv = *n;
	i__1 = *k;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    prevlastv = max(i__,prevlastv);
	    if (tau[i__] == 0.) {
		//
		//             H(i)  =  I
		//
		i__2 = i__;
		for (j = 1; j <= i__2; ++j) {
		    t[j + i__ * t_dim1] = 0.;
		}
	    } else {
		//
		//             general case
		//
		if (lsame_(storev, "C")) {
		    //                Skip any trailing zeros.
		    i__2 = i__ + 1;
		    for (lastv = *n; lastv >= i__2; --lastv) {
			if (v[lastv + i__ * v_dim1] != 0.) {
			    break;
			}
		    }
		    i__2 = i__ - 1;
		    for (j = 1; j <= i__2; ++j) {
			t[j + i__ * t_dim1] = -tau[i__] * v[i__ + j * v_dim1];
		    }
		    j = min(lastv,prevlastv);
		    //
		    //                T(1:i-1,i) := - tau(i) * V(i:j,1:i-1)**T * V(i:j,i)
		    //
		    i__2 = j - i__;
		    i__3 = i__ - 1;
		    d__1 = -tau[i__];
		    dgemv_("Transpose", &i__2, &i__3, &d__1, &v[i__ + 1 +
			    v_dim1], ldv, &v[i__ + 1 + i__ * v_dim1], &c__1, &
			    c_b7, &t[i__ * t_dim1 + 1], &c__1);
		} else {
		    //                Skip any trailing zeros.
		    i__2 = i__ + 1;
		    for (lastv = *n; lastv >= i__2; --lastv) {
			if (v[i__ + lastv * v_dim1] != 0.) {
			    break;
			}
		    }
		    i__2 = i__ - 1;
		    for (j = 1; j <= i__2; ++j) {
			t[j + i__ * t_dim1] = -tau[i__] * v[j + i__ * v_dim1];
		    }
		    j = min(lastv,prevlastv);
		    //
		    //                T(1:i-1,i) := - tau(i) * V(1:i-1,i:j) * V(i,i:j)**T
		    //
		    i__2 = i__ - 1;
		    i__3 = j - i__;
		    d__1 = -tau[i__];
		    dgemv_("No transpose", &i__2, &i__3, &d__1, &v[(i__ + 1) *
			     v_dim1 + 1], ldv, &v[i__ + (i__ + 1) * v_dim1],
			    ldv, &c_b7, &t[i__ * t_dim1 + 1], &c__1);
		}
		//
		//             T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
		//
		i__2 = i__ - 1;
		dtrmv_("Upper", "No transpose", "Non-unit", &i__2, &t[
			t_offset], ldt, &t[i__ * t_dim1 + 1], &c__1);
		t[i__ + i__ * t_dim1] = tau[i__];
		if (i__ > 1) {
		    prevlastv = max(prevlastv,lastv);
		} else {
		    prevlastv = lastv;
		}
	    }
	}
    } else {
	prevlastv = 1;
	for (i__ = *k; i__ >= 1; --i__) {
	    if (tau[i__] == 0.) {
		//
		//             H(i)  =  I
		//
		i__1 = *k;
		for (j = i__; j <= i__1; ++j) {
		    t[j + i__ * t_dim1] = 0.;
		}
	    } else {
		//
		//             general case
		//
		if (i__ < *k) {
		    if (lsame_(storev, "C")) {
			//                   Skip any leading zeros.
			i__1 = i__ - 1;
			for (lastv = 1; lastv <= i__1; ++lastv) {
			    if (v[lastv + i__ * v_dim1] != 0.) {
				break;
			    }
			}
			i__1 = *k;
			for (j = i__ + 1; j <= i__1; ++j) {
			    t[j + i__ * t_dim1] = -tau[i__] * v[*n - *k + i__
				    + j * v_dim1];
			}
			j = max(lastv,prevlastv);
			//
			//                   T(i+1:k,i) = -tau(i) * V(j:n-k+i,i+1:k)**T * V(j:n-k+i,i)
			//
			i__1 = *n - *k + i__ - j;
			i__2 = *k - i__;
			d__1 = -tau[i__];
			dgemv_("Transpose", &i__1, &i__2, &d__1, &v[j + (i__
				+ 1) * v_dim1], ldv, &v[j + i__ * v_dim1], &
				c__1, &c_b7, &t[i__ + 1 + i__ * t_dim1], &
				c__1);
		    } else {
			//                   Skip any leading zeros.
			i__1 = i__ - 1;
			for (lastv = 1; lastv <= i__1; ++lastv) {
			    if (v[i__ + lastv * v_dim1] != 0.) {
				break;
			    }
			}
			i__1 = *k;
			for (j = i__ + 1; j <= i__1; ++j) {
			    t[j + i__ * t_dim1] = -tau[i__] * v[j + (*n - *k
				    + i__) * v_dim1];
			}
			j = max(lastv,prevlastv);
			//
			//                   T(i+1:k,i) = -tau(i) * V(i+1:k,j:n-k+i) * V(i,j:n-k+i)**T
			//
			i__1 = *k - i__;
			i__2 = *n - *k + i__ - j;
			d__1 = -tau[i__];
			dgemv_("No transpose", &i__1, &i__2, &d__1, &v[i__ +
				1 + j * v_dim1], ldv, &v[i__ + j * v_dim1],
				ldv, &c_b7, &t[i__ + 1 + i__ * t_dim1], &c__1)
				;
		    }
		    //
		    //                T(i+1:k,i) := T(i+1:k,i+1:k) * T(i+1:k,i)
		    //
		    i__1 = *k - i__;
		    dtrmv_("Lower", "No transpose", "Non-unit", &i__1, &t[i__
			    + 1 + (i__ + 1) * t_dim1], ldt, &t[i__ + 1 + i__ *
			     t_dim1], &c__1);
		    if (i__ > 1) {
			prevlastv = min(prevlastv,lastv);
		    } else {
			prevlastv = lastv;
		    }
		}
		t[i__ + i__ * t_dim1] = tau[i__];
	    }
	}
    }
    return 0;
    //
    //    End of DLARFT
    //
} // dlarft_

