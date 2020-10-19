/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLASQ2 computes all the eigenvalues of the symmetric positive definite tridiagonal matrix associated with the qd Array Z to high relative accuracy. Used by sbdsqr and sstegr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ2( N, Z, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ2 computes all the eigenvalues of the symmetric positive
//> definite tridiagonal matrix associated with the qd array Z to high
//> relative accuracy are computed to high relative accuracy, in the
//> absence of denormalization, underflow and overflow.
//>
//> To see the relation of Z to the tridiagonal matrix, let L be a
//> unit lower bidiagonal matrix with subdiagonals Z(2,4,6,,..) and
//> let U be an upper bidiagonal matrix with 1's above and diagonal
//> Z(1,3,5,,..). The tridiagonal is L*U or, if you prefer, the
//> symmetric tridiagonal to which it is similar.
//>
//> Note : DLASQ2 defines a logical variable, IEEE, which is true
//> on machines which follow ieee-754 floating-point standard in their
//> handling of infinities and NaNs, and false otherwise. This variable
//> is passed to DLASQ3.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>        The number of rows and columns in the matrix. N >= 0.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N )
//>        On entry Z holds the qd array. On exit, entries 1 to N hold
//>        the eigenvalues in decreasing order, Z( 2*N+1 ) holds the
//>        trace, and Z( 2*N+2 ) holds the sum of the eigenvalues. If
//>        N > 2, then Z( 2*N+3 ) holds the iteration count, Z( 2*N+4 )
//>        holds NDIVS/NIN^2, and Z( 2*N+5 ) holds the percentage of
//>        shifts that failed.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>        = 0: successful exit
//>        < 0: if the i-th argument is a scalar and had an illegal
//>             value, then INFO = -i, if the i-th argument is an
//>             array and the j-entry had an illegal value, then
//>             INFO = -(i*100+j)
//>        > 0: the algorithm failed
//>              = 1, a split was marked by a positive value in E
//>              = 2, current block of Z not diagonalized after 100*N
//>                   iterations (in inner while loop).  On exit Z holds
//>                   a qd array with the same eigenvalues as the given Z.
//>              = 3, termination criterion of outer while loop not met
//>                   (program created more than N unreduced blocks)
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
//> \ingroup auxOTHERcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  Local Variables: I0:N0 defines a current unreduced segment of Z.
//>  The shifts are accumulated in SIGMA. Iteration count is in ITER.
//>  Ping-pong is controlled by PP (alternates between 0 and 1).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlasq2_(int *n, double *z__, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__2 = 2;
    int c__10 = 10;
    int c__3 = 3;
    int c__4 = 4;
    int c__11 = 11;

    // System generated locals
    int i__1, i__2, i__3;
    double d__1, d__2;

    // Builtin functions
    double sqrt(double);

    // Local variables
    double d__, e, g;
    int k;
    double s, t;
    int i0, i1, i4, n0, n1;
    double dn;
    int pp;
    double dn1, dn2, dee, eps, tau, tol;
    int ipn4;
    double tol2;
    int ieee;
    int nbig;
    double dmin__, emin, emax;
    int kmin, ndiv, iter;
    double qmin, temp, qmax, zmax;
    int splt;
    double dmin1, dmin2;
    int nfail;
    double desig, trace, sigma;
    int iinfo;
    double tempe, tempq;
    int ttype;
    extern /* Subroutine */ int dlasq3_(int *, int *, double *, int *, double 
	    *, double *, double *, double *, int *, int *, int *, int *, int *
	    , double *, double *, double *, double *, double *, double *, 
	    double *);
    extern double dlamch_(char *);
    double deemin;
    int iwhila, iwhilb;
    double oldemn, safmin;
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dlasrt_(char *, int *, double *, int *);

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
    //    Test the input arguments.
    //    (in case DLASQ2 is not called by DLASQ1)
    //
    // Parameter adjustments
    --z__;

    // Function Body
    *info = 0;
    eps = dlamch_("Precision");
    safmin = dlamch_("Safe minimum");
    tol = eps * 100.;
    // Computing 2nd power
    d__1 = tol;
    tol2 = d__1 * d__1;
    if (*n < 0) {
	*info = -1;
	xerbla_("DLASQ2", &c__1);
	return 0;
    } else if (*n == 0) {
	return 0;
    } else if (*n == 1) {
	//
	//       1-by-1 case.
	//
	if (z__[1] < 0.) {
	    *info = -201;
	    xerbla_("DLASQ2", &c__2);
	}
	return 0;
    } else if (*n == 2) {
	//
	//       2-by-2 case.
	//
	if (z__[2] < 0. || z__[3] < 0.) {
	    *info = -2;
	    xerbla_("DLASQ2", &c__2);
	    return 0;
	} else if (z__[3] > z__[1]) {
	    d__ = z__[3];
	    z__[3] = z__[1];
	    z__[1] = d__;
	}
	z__[5] = z__[1] + z__[2] + z__[3];
	if (z__[2] > z__[3] * tol2) {
	    t = (z__[1] - z__[3] + z__[2]) * .5;
	    s = z__[3] * (z__[2] / t);
	    if (s <= t) {
		s = z__[3] * (z__[2] / (t * (sqrt(s / t + 1.) + 1.)));
	    } else {
		s = z__[3] * (z__[2] / (t + sqrt(t) * sqrt(t + s)));
	    }
	    t = z__[1] + (s + z__[2]);
	    z__[3] *= z__[1] / t;
	    z__[1] = t;
	}
	z__[2] = z__[3];
	z__[6] = z__[2] + z__[1];
	return 0;
    }
    //
    //    Check for negative data and compute sums of q's and e's.
    //
    z__[*n * 2] = 0.;
    emin = z__[2];
    qmax = 0.;
    zmax = 0.;
    d__ = 0.;
    e = 0.;
    i__1 = *n - 1 << 1;
    for (k = 1; k <= i__1; k += 2) {
	if (z__[k] < 0.) {
	    *info = -(k + 200);
	    xerbla_("DLASQ2", &c__2);
	    return 0;
	} else if (z__[k + 1] < 0.) {
	    *info = -(k + 201);
	    xerbla_("DLASQ2", &c__2);
	    return 0;
	}
	d__ += z__[k];
	e += z__[k + 1];
	// Computing MAX
	d__1 = qmax, d__2 = z__[k];
	qmax = max(d__1,d__2);
	// Computing MIN
	d__1 = emin, d__2 = z__[k + 1];
	emin = min(d__1,d__2);
	// Computing MAX
	d__1 = max(qmax,zmax), d__2 = z__[k + 1];
	zmax = max(d__1,d__2);
// L10:
    }
    if (z__[(*n << 1) - 1] < 0.) {
	*info = -((*n << 1) + 199);
	xerbla_("DLASQ2", &c__2);
	return 0;
    }
    d__ += z__[(*n << 1) - 1];
    // Computing MAX
    d__1 = qmax, d__2 = z__[(*n << 1) - 1];
    qmax = max(d__1,d__2);
    zmax = max(qmax,zmax);
    //
    //    Check for diagonality.
    //
    if (e == 0.) {
	i__1 = *n;
	for (k = 2; k <= i__1; ++k) {
	    z__[k] = z__[(k << 1) - 1];
// L20:
	}
	dlasrt_("D", n, &z__[1], &iinfo);
	z__[(*n << 1) - 1] = d__;
	return 0;
    }
    trace = d__ + e;
    //
    //    Check for zero data.
    //
    if (trace == 0.) {
	z__[(*n << 1) - 1] = 0.;
	return 0;
    }
    //
    //    Check whether the machine is IEEE conformable.
    //
    ieee = ilaenv_(&c__10, "DLASQ2", "N", &c__1, &c__2, &c__3, &c__4) == 1 && 
	    ilaenv_(&c__11, "DLASQ2", "N", &c__1, &c__2, &c__3, &c__4) == 1;
    //
    //    Rearrange data for locality: Z=(q1,qq1,e1,ee1,q2,qq2,e2,ee2,...).
    //
    for (k = *n << 1; k >= 2; k += -2) {
	z__[k * 2] = 0.;
	z__[(k << 1) - 1] = z__[k];
	z__[(k << 1) - 2] = 0.;
	z__[(k << 1) - 3] = z__[k - 1];
// L30:
    }
    i0 = 1;
    n0 = *n;
    //
    //    Reverse the qd-array, if warranted.
    //
    if (z__[(i0 << 2) - 3] * 1.5 < z__[(n0 << 2) - 3]) {
	ipn4 = i0 + n0 << 2;
	i__1 = i0 + n0 - 1 << 1;
	for (i4 = i0 << 2; i4 <= i__1; i4 += 4) {
	    temp = z__[i4 - 3];
	    z__[i4 - 3] = z__[ipn4 - i4 - 3];
	    z__[ipn4 - i4 - 3] = temp;
	    temp = z__[i4 - 1];
	    z__[i4 - 1] = z__[ipn4 - i4 - 5];
	    z__[ipn4 - i4 - 5] = temp;
// L40:
	}
    }
    //
    //    Initial split checking via dqd and Li's test.
    //
    pp = 0;
    for (k = 1; k <= 2; ++k) {
	d__ = z__[(n0 << 2) + pp - 3];
	i__1 = (i0 << 2) + pp;
	for (i4 = (n0 - 1 << 2) + pp; i4 >= i__1; i4 += -4) {
	    if (z__[i4 - 1] <= tol2 * d__) {
		z__[i4 - 1] = -0.;
		d__ = z__[i4 - 3];
	    } else {
		d__ = z__[i4 - 3] * (d__ / (d__ + z__[i4 - 1]));
	    }
// L50:
	}
	//
	//       dqd maps Z to ZZ plus Li's test.
	//
	emin = z__[(i0 << 2) + pp + 1];
	d__ = z__[(i0 << 2) + pp - 3];
	i__1 = (n0 - 1 << 2) + pp;
	for (i4 = (i0 << 2) + pp; i4 <= i__1; i4 += 4) {
	    z__[i4 - (pp << 1) - 2] = d__ + z__[i4 - 1];
	    if (z__[i4 - 1] <= tol2 * d__) {
		z__[i4 - 1] = -0.;
		z__[i4 - (pp << 1) - 2] = d__;
		z__[i4 - (pp << 1)] = 0.;
		d__ = z__[i4 + 1];
	    } else if (safmin * z__[i4 + 1] < z__[i4 - (pp << 1) - 2] && 
		    safmin * z__[i4 - (pp << 1) - 2] < z__[i4 + 1]) {
		temp = z__[i4 + 1] / z__[i4 - (pp << 1) - 2];
		z__[i4 - (pp << 1)] = z__[i4 - 1] * temp;
		d__ *= temp;
	    } else {
		z__[i4 - (pp << 1)] = z__[i4 + 1] * (z__[i4 - 1] / z__[i4 - (
			pp << 1) - 2]);
		d__ = z__[i4 + 1] * (d__ / z__[i4 - (pp << 1) - 2]);
	    }
	    // Computing MIN
	    d__1 = emin, d__2 = z__[i4 - (pp << 1)];
	    emin = min(d__1,d__2);
// L60:
	}
	z__[(n0 << 2) - pp - 2] = d__;
	//
	//       Now find qmax.
	//
	qmax = z__[(i0 << 2) - pp - 2];
	i__1 = (n0 << 2) - pp - 2;
	for (i4 = (i0 << 2) - pp + 2; i4 <= i__1; i4 += 4) {
	    // Computing MAX
	    d__1 = qmax, d__2 = z__[i4];
	    qmax = max(d__1,d__2);
// L70:
	}
	//
	//       Prepare for the next iteration on K.
	//
	pp = 1 - pp;
// L80:
    }
    //
    //    Initialise variables to pass to DLASQ3.
    //
    ttype = 0;
    dmin1 = 0.;
    dmin2 = 0.;
    dn = 0.;
    dn1 = 0.;
    dn2 = 0.;
    g = 0.;
    tau = 0.;
    iter = 2;
    nfail = 0;
    ndiv = n0 - i0 << 1;
    i__1 = *n + 1;
    for (iwhila = 1; iwhila <= i__1; ++iwhila) {
	if (n0 < 1) {
	    goto L170;
	}
	//
	//       While array unfinished do
	//
	//       E(N0) holds the value of SIGMA when submatrix in I0:N0
	//       splits from the rest of the array, but is negated.
	//
	desig = 0.;
	if (n0 == *n) {
	    sigma = 0.;
	} else {
	    sigma = -z__[(n0 << 2) - 1];
	}
	if (sigma < 0.) {
	    *info = 1;
	    return 0;
	}
	//
	//       Find last unreduced submatrix's top index I0, find QMAX and
	//       EMIN. Find Gershgorin-type bound if Q's much greater than E's.
	//
	emax = 0.;
	if (n0 > i0) {
	    emin = (d__1 = z__[(n0 << 2) - 5], abs(d__1));
	} else {
	    emin = 0.;
	}
	qmin = z__[(n0 << 2) - 3];
	qmax = qmin;
	for (i4 = n0 << 2; i4 >= 8; i4 += -4) {
	    if (z__[i4 - 5] <= 0.) {
		goto L100;
	    }
	    if (qmin >= emax * 4.) {
		// Computing MIN
		d__1 = qmin, d__2 = z__[i4 - 3];
		qmin = min(d__1,d__2);
		// Computing MAX
		d__1 = emax, d__2 = z__[i4 - 5];
		emax = max(d__1,d__2);
	    }
	    // Computing MAX
	    d__1 = qmax, d__2 = z__[i4 - 7] + z__[i4 - 5];
	    qmax = max(d__1,d__2);
	    // Computing MIN
	    d__1 = emin, d__2 = z__[i4 - 5];
	    emin = min(d__1,d__2);
// L90:
	}
	i4 = 4;
L100:
	i0 = i4 / 4;
	pp = 0;
	if (n0 - i0 > 1) {
	    dee = z__[(i0 << 2) - 3];
	    deemin = dee;
	    kmin = i0;
	    i__2 = (n0 << 2) - 3;
	    for (i4 = (i0 << 2) + 1; i4 <= i__2; i4 += 4) {
		dee = z__[i4] * (dee / (dee + z__[i4 - 2]));
		if (dee <= deemin) {
		    deemin = dee;
		    kmin = (i4 + 3) / 4;
		}
// L110:
	    }
	    if (kmin - i0 << 1 < n0 - kmin && deemin <= z__[(n0 << 2) - 3] * 
		    .5) {
		ipn4 = i0 + n0 << 2;
		pp = 2;
		i__2 = i0 + n0 - 1 << 1;
		for (i4 = i0 << 2; i4 <= i__2; i4 += 4) {
		    temp = z__[i4 - 3];
		    z__[i4 - 3] = z__[ipn4 - i4 - 3];
		    z__[ipn4 - i4 - 3] = temp;
		    temp = z__[i4 - 2];
		    z__[i4 - 2] = z__[ipn4 - i4 - 2];
		    z__[ipn4 - i4 - 2] = temp;
		    temp = z__[i4 - 1];
		    z__[i4 - 1] = z__[ipn4 - i4 - 5];
		    z__[ipn4 - i4 - 5] = temp;
		    temp = z__[i4];
		    z__[i4] = z__[ipn4 - i4 - 4];
		    z__[ipn4 - i4 - 4] = temp;
// L120:
		}
	    }
	}
	//
	//       Put -(initial shift) into DMIN.
	//
	// Computing MAX
	d__1 = 0., d__2 = qmin - sqrt(qmin) * 2. * sqrt(emax);
	dmin__ = -max(d__1,d__2);
	//
	//       Now I0:N0 is unreduced.
	//       PP = 0 for ping, PP = 1 for pong.
	//       PP = 2 indicates that flipping was applied to the Z array and
	//              and that the tests for deflation upon entry in DLASQ3
	//              should not be performed.
	//
	nbig = (n0 - i0 + 1) * 100;
	i__2 = nbig;
	for (iwhilb = 1; iwhilb <= i__2; ++iwhilb) {
	    if (i0 > n0) {
		goto L150;
	    }
	    //
	    //          While submatrix unfinished take a good dqds step.
	    //
	    dlasq3_(&i0, &n0, &z__[1], &pp, &dmin__, &sigma, &desig, &qmax, &
		    nfail, &iter, &ndiv, &ieee, &ttype, &dmin1, &dmin2, &dn, &
		    dn1, &dn2, &g, &tau);
	    pp = 1 - pp;
	    //
	    //          When EMIN is very small check for splits.
	    //
	    if (pp == 0 && n0 - i0 >= 3) {
		if (z__[n0 * 4] <= tol2 * qmax || z__[(n0 << 2) - 1] <= tol2 *
			 sigma) {
		    splt = i0 - 1;
		    qmax = z__[(i0 << 2) - 3];
		    emin = z__[(i0 << 2) - 1];
		    oldemn = z__[i0 * 4];
		    i__3 = n0 - 3 << 2;
		    for (i4 = i0 << 2; i4 <= i__3; i4 += 4) {
			if (z__[i4] <= tol2 * z__[i4 - 3] || z__[i4 - 1] <= 
				tol2 * sigma) {
			    z__[i4 - 1] = -sigma;
			    splt = i4 / 4;
			    qmax = 0.;
			    emin = z__[i4 + 3];
			    oldemn = z__[i4 + 4];
			} else {
			    // Computing MAX
			    d__1 = qmax, d__2 = z__[i4 + 1];
			    qmax = max(d__1,d__2);
			    // Computing MIN
			    d__1 = emin, d__2 = z__[i4 - 1];
			    emin = min(d__1,d__2);
			    // Computing MIN
			    d__1 = oldemn, d__2 = z__[i4];
			    oldemn = min(d__1,d__2);
			}
// L130:
		    }
		    z__[(n0 << 2) - 1] = emin;
		    z__[n0 * 4] = oldemn;
		    i0 = splt + 1;
		}
	    }
// L140:
	}
	*info = 2;
	//
	//       Maximum number of iterations exceeded, restore the shift
	//       SIGMA and place the new d's and e's in a qd array.
	//       This might need to be done for several blocks
	//
	i1 = i0;
	n1 = n0;
L145:
	tempq = z__[(i0 << 2) - 3];
	z__[(i0 << 2) - 3] += sigma;
	i__2 = n0;
	for (k = i0 + 1; k <= i__2; ++k) {
	    tempe = z__[(k << 2) - 5];
	    z__[(k << 2) - 5] *= tempq / z__[(k << 2) - 7];
	    tempq = z__[(k << 2) - 3];
	    z__[(k << 2) - 3] = z__[(k << 2) - 3] + sigma + tempe - z__[(k << 
		    2) - 5];
	}
	//
	//       Prepare to do this on the previous block if there is one
	//
	if (i1 > 1) {
	    n1 = i1 - 1;
	    while(i1 >= 2 && z__[(i1 << 2) - 5] >= 0.) {
		--i1;
	    }
	    sigma = -z__[(n1 << 2) - 1];
	    goto L145;
	}
	i__2 = *n;
	for (k = 1; k <= i__2; ++k) {
	    z__[(k << 1) - 1] = z__[(k << 2) - 3];
	    //
	    //       Only the block 1..N0 is unfinished.  The rest of the e's
	    //       must be essentially zero, although sometimes other data
	    //       has been stored in them.
	    //
	    if (k < n0) {
		z__[k * 2] = z__[(k << 2) - 1];
	    } else {
		z__[k * 2] = 0.;
	    }
	}
	return 0;
	//
	//       end IWHILB
	//
L150:
// L160:
	;
    }
    *info = 3;
    return 0;
    //
    //    end IWHILA
    //
L170:
    //
    //    Move q's to the front.
    //
    i__1 = *n;
    for (k = 2; k <= i__1; ++k) {
	z__[k] = z__[(k << 2) - 3];
// L180:
    }
    //
    //    Sort and compute sum of eigenvalues.
    //
    dlasrt_("D", n, &z__[1], &iinfo);
    e = 0.;
    for (k = *n; k >= 1; --k) {
	e += z__[k];
// L190:
    }
    //
    //    Store trace, sum(eigenvalues) and information on performance.
    //
    z__[(*n << 1) + 1] = trace;
    z__[(*n << 1) + 2] = e;
    z__[(*n << 1) + 3] = (double) iter;
    // Computing 2nd power
    i__1 = *n;
    z__[(*n << 1) + 4] = (double) ndiv / (double) (i__1 * i__1);
    z__[(*n << 1) + 5] = nfail * 100. / (double) iter;
    return 0;
    //
    //    End of DLASQ2
    //
} // dlasq2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ3 checks for deflation, computes a shift and calls dqds. Used by sbdsqr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ3 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq3.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq3.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq3.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ3( I0, N0, Z, PP, DMIN, SIGMA, DESIG, QMAX, NFAIL,
//                         ITER, NDIV, IEEE, TTYPE, DMIN1, DMIN2, DN, DN1,
//                         DN2, G, TAU )
//
//      .. Scalar Arguments ..
//      LOGICAL            IEEE
//      INTEGER            I0, ITER, N0, NDIV, NFAIL, PP
//      DOUBLE PRECISION   DESIG, DMIN, DMIN1, DMIN2, DN, DN1, DN2, G,
//     $                   QMAX, SIGMA, TAU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ3 checks for deflation, computes a shift (TAU) and calls dqds.
//> In case of failure it changes shifts, and tries again until output
//> is positive.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>         First index.
//> \endverbatim
//>
//> \param[in,out] N0
//> \verbatim
//>          N0 is INTEGER
//>         Last index.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N0 )
//>         Z holds the qd array.
//> \endverbatim
//>
//> \param[in,out] PP
//> \verbatim
//>          PP is INTEGER
//>         PP=0 for ping, PP=1 for pong.
//>         PP=2 indicates that flipping was applied to the Z array
//>         and that the initial tests for deflation should not be
//>         performed.
//> \endverbatim
//>
//> \param[out] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>         Minimum value of d.
//> \endverbatim
//>
//> \param[out] SIGMA
//> \verbatim
//>          SIGMA is DOUBLE PRECISION
//>         Sum of shifts used in current segment.
//> \endverbatim
//>
//> \param[in,out] DESIG
//> \verbatim
//>          DESIG is DOUBLE PRECISION
//>         Lower order part of SIGMA
//> \endverbatim
//>
//> \param[in] QMAX
//> \verbatim
//>          QMAX is DOUBLE PRECISION
//>         Maximum value of q.
//> \endverbatim
//>
//> \param[in,out] NFAIL
//> \verbatim
//>          NFAIL is INTEGER
//>         Increment NFAIL by 1 each time the shift was too big.
//> \endverbatim
//>
//> \param[in,out] ITER
//> \verbatim
//>          ITER is INTEGER
//>         Increment ITER by 1 for each iteration.
//> \endverbatim
//>
//> \param[in,out] NDIV
//> \verbatim
//>          NDIV is INTEGER
//>         Increment NDIV by 1 for each division.
//> \endverbatim
//>
//> \param[in] IEEE
//> \verbatim
//>          IEEE is LOGICAL
//>         Flag for IEEE or non IEEE arithmetic (passed to DLASQ5).
//> \endverbatim
//>
//> \param[in,out] TTYPE
//> \verbatim
//>          TTYPE is INTEGER
//>         Shift type.
//> \endverbatim
//>
//> \param[in,out] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DN1
//> \verbatim
//>          DN1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] DN2
//> \verbatim
//>          DN2 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] G
//> \verbatim
//>          G is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in,out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>
//>         These are passed as arguments in order to save their values
//>         between calls to DLASQ3.
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
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasq3_(int *i0, int *n0, double *z__, int *pp, double *
	dmin__, double *sigma, double *desig, double *qmax, int *nfail, int *
	iter, int *ndiv, int *ieee, int *ttype, double *dmin1, double *dmin2, 
	double *dn, double *dn1, double *dn2, double *g, double *tau)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Builtin functions
    double sqrt(double);

    // Local variables
    double s, t;
    int j4, nn;
    double eps, tol;
    int n0in, ipn4;
    double tol2, temp;
    extern /* Subroutine */ int dlasq4_(int *, int *, double *, int *, int *, 
	    double *, double *, double *, double *, double *, double *, 
	    double *, int *, double *), dlasq5_(int *, int *, double *, int *,
	     double *, double *, double *, double *, double *, double *, 
	    double *, double *, int *, double *), dlasq6_(int *, int *, 
	    double *, int *, double *, double *, double *, double *, double *,
	     double *);
    extern double dlamch_(char *);
    extern int disnan_(double *);

    //
    // -- LAPACK computational routine (version 3.7.0) --
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
    //    .. External Subroutines ..
    //    ..
    //    .. External Function ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --z__;

    // Function Body
    n0in = *n0;
    eps = dlamch_("Precision");
    tol = eps * 100.;
    // Computing 2nd power
    d__1 = tol;
    tol2 = d__1 * d__1;
    //
    //    Check for deflation.
    //
L10:
    if (*n0 < *i0) {
	return 0;
    }
    if (*n0 == *i0) {
	goto L20;
    }
    nn = (*n0 << 2) + *pp;
    if (*n0 == *i0 + 1) {
	goto L40;
    }
    //
    //    Check whether E(N0-1) is negligible, 1 eigenvalue.
    //
    if (z__[nn - 5] > tol2 * (*sigma + z__[nn - 3]) && z__[nn - (*pp << 1) - 
	    4] > tol2 * z__[nn - 7]) {
	goto L30;
    }
L20:
    z__[(*n0 << 2) - 3] = z__[(*n0 << 2) + *pp - 3] + *sigma;
    --(*n0);
    goto L10;
    //
    //    Check  whether E(N0-2) is negligible, 2 eigenvalues.
    //
L30:
    if (z__[nn - 9] > tol2 * *sigma && z__[nn - (*pp << 1) - 8] > tol2 * z__[
	    nn - 11]) {
	goto L50;
    }
L40:
    if (z__[nn - 3] > z__[nn - 7]) {
	s = z__[nn - 3];
	z__[nn - 3] = z__[nn - 7];
	z__[nn - 7] = s;
    }
    t = (z__[nn - 7] - z__[nn - 3] + z__[nn - 5]) * .5;
    if (z__[nn - 5] > z__[nn - 3] * tol2 && t != 0.) {
	s = z__[nn - 3] * (z__[nn - 5] / t);
	if (s <= t) {
	    s = z__[nn - 3] * (z__[nn - 5] / (t * (sqrt(s / t + 1.) + 1.)));
	} else {
	    s = z__[nn - 3] * (z__[nn - 5] / (t + sqrt(t) * sqrt(t + s)));
	}
	t = z__[nn - 7] + (s + z__[nn - 5]);
	z__[nn - 3] *= z__[nn - 7] / t;
	z__[nn - 7] = t;
    }
    z__[(*n0 << 2) - 7] = z__[nn - 7] + *sigma;
    z__[(*n0 << 2) - 3] = z__[nn - 3] + *sigma;
    *n0 += -2;
    goto L10;
L50:
    if (*pp == 2) {
	*pp = 0;
    }
    //
    //    Reverse the qd-array, if warranted.
    //
    if (*dmin__ <= 0. || *n0 < n0in) {
	if (z__[(*i0 << 2) + *pp - 3] * 1.5 < z__[(*n0 << 2) + *pp - 3]) {
	    ipn4 = *i0 + *n0 << 2;
	    i__1 = *i0 + *n0 - 1 << 1;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		temp = z__[j4 - 3];
		z__[j4 - 3] = z__[ipn4 - j4 - 3];
		z__[ipn4 - j4 - 3] = temp;
		temp = z__[j4 - 2];
		z__[j4 - 2] = z__[ipn4 - j4 - 2];
		z__[ipn4 - j4 - 2] = temp;
		temp = z__[j4 - 1];
		z__[j4 - 1] = z__[ipn4 - j4 - 5];
		z__[ipn4 - j4 - 5] = temp;
		temp = z__[j4];
		z__[j4] = z__[ipn4 - j4 - 4];
		z__[ipn4 - j4 - 4] = temp;
// L60:
	    }
	    if (*n0 - *i0 <= 4) {
		z__[(*n0 << 2) + *pp - 1] = z__[(*i0 << 2) + *pp - 1];
		z__[(*n0 << 2) - *pp] = z__[(*i0 << 2) - *pp];
	    }
	    // Computing MIN
	    d__1 = *dmin2, d__2 = z__[(*n0 << 2) + *pp - 1];
	    *dmin2 = min(d__1,d__2);
	    // Computing MIN
	    d__1 = z__[(*n0 << 2) + *pp - 1], d__2 = z__[(*i0 << 2) + *pp - 1]
		    , d__1 = min(d__1,d__2), d__2 = z__[(*i0 << 2) + *pp + 3];
	    z__[(*n0 << 2) + *pp - 1] = min(d__1,d__2);
	    // Computing MIN
	    d__1 = z__[(*n0 << 2) - *pp], d__2 = z__[(*i0 << 2) - *pp], d__1 =
		     min(d__1,d__2), d__2 = z__[(*i0 << 2) - *pp + 4];
	    z__[(*n0 << 2) - *pp] = min(d__1,d__2);
	    // Computing MAX
	    d__1 = *qmax, d__2 = z__[(*i0 << 2) + *pp - 3], d__1 = max(d__1,
		    d__2), d__2 = z__[(*i0 << 2) + *pp + 1];
	    *qmax = max(d__1,d__2);
	    *dmin__ = -0.;
	}
    }
    //
    //    Choose a shift.
    //
    dlasq4_(i0, n0, &z__[1], pp, &n0in, dmin__, dmin1, dmin2, dn, dn1, dn2, 
	    tau, ttype, g);
    //
    //    Call dqds until DMIN > 0.
    //
L70:
    dlasq5_(i0, n0, &z__[1], pp, tau, sigma, dmin__, dmin1, dmin2, dn, dn1, 
	    dn2, ieee, &eps);
    *ndiv += *n0 - *i0 + 2;
    ++(*iter);
    //
    //    Check status.
    //
    if (*dmin__ >= 0. && *dmin1 >= 0.) {
	//
	//       Success.
	//
	goto L90;
    } else if (*dmin__ < 0. && *dmin1 > 0. && z__[(*n0 - 1 << 2) - *pp] < tol 
	    * (*sigma + *dn1) && abs(*dn) < tol * *sigma) {
	//
	//       Convergence hidden by negative DN.
	//
	z__[(*n0 - 1 << 2) - *pp + 2] = 0.;
	*dmin__ = 0.;
	goto L90;
    } else if (*dmin__ < 0.) {
	//
	//       TAU too big. Select new TAU and try again.
	//
	++(*nfail);
	if (*ttype < -22) {
	    //
	    //          Failed twice. Play it safe.
	    //
	    *tau = 0.;
	} else if (*dmin1 > 0.) {
	    //
	    //          Late failure. Gives excellent shift.
	    //
	    *tau = (*tau + *dmin__) * (1. - eps * 2.);
	    *ttype += -11;
	} else {
	    //
	    //          Early failure. Divide by 4.
	    //
	    *tau *= .25;
	    *ttype += -12;
	}
	goto L70;
    } else if (disnan_(dmin__)) {
	//
	//       NaN.
	//
	if (*tau == 0.) {
	    goto L80;
	} else {
	    *tau = 0.;
	    goto L70;
	}
    } else {
	//
	//       Possible underflow. Play it safe.
	//
	goto L80;
    }
    //
    //    Risk of underflow.
    //
L80:
    dlasq6_(i0, n0, &z__[1], pp, dmin__, dmin1, dmin2, dn, dn1, dn2);
    *ndiv += *n0 - *i0 + 2;
    ++(*iter);
    *tau = 0.;
L90:
    if (*tau < *sigma) {
	*desig += *tau;
	t = *sigma + *desig;
	*desig -= t - *sigma;
    } else {
	t = *sigma + *tau;
	*desig = *sigma - (t - *tau) + *desig;
    }
    *sigma = t;
    return 0;
    //
    //    End of DLASQ3
    //
} // dlasq3_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ4 computes an approximation to the smallest eigenvalue using values of d from the previous transform. Used by sbdsqr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ4 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq4.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq4.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq4.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ4( I0, N0, Z, PP, N0IN, DMIN, DMIN1, DMIN2, DN,
//                         DN1, DN2, TAU, TTYPE, G )
//
//      .. Scalar Arguments ..
//      INTEGER            I0, N0, N0IN, PP, TTYPE
//      DOUBLE PRECISION   DMIN, DMIN1, DMIN2, DN, DN1, DN2, G, TAU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ4 computes an approximation TAU to the smallest eigenvalue
//> using values of d from the previous transform.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>        First index.
//> \endverbatim
//>
//> \param[in] N0
//> \verbatim
//>          N0 is INTEGER
//>        Last index.
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N0 )
//>        Z holds the qd array.
//> \endverbatim
//>
//> \param[in] PP
//> \verbatim
//>          PP is INTEGER
//>        PP=0 for ping, PP=1 for pong.
//> \endverbatim
//>
//> \param[in] N0IN
//> \verbatim
//>          N0IN is INTEGER
//>        The value of N0 at start of EIGTEST.
//> \endverbatim
//>
//> \param[in] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>        Minimum value of d.
//> \endverbatim
//>
//> \param[in] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ).
//> \endverbatim
//>
//> \param[in] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
//> \endverbatim
//>
//> \param[in] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//>        d(N)
//> \endverbatim
//>
//> \param[in] DN1
//> \verbatim
//>          DN1 is DOUBLE PRECISION
//>        d(N-1)
//> \endverbatim
//>
//> \param[in] DN2
//> \verbatim
//>          DN2 is DOUBLE PRECISION
//>        d(N-2)
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>        This is the shift.
//> \endverbatim
//>
//> \param[out] TTYPE
//> \verbatim
//>          TTYPE is INTEGER
//>        Shift type.
//> \endverbatim
//>
//> \param[in,out] G
//> \verbatim
//>          G is DOUBLE PRECISION
//>        G is passed as an argument in order to save its value between
//>        calls to DLASQ4.
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
//> \ingroup auxOTHERcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  CNST1 = 9/16
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlasq4_(int *i0, int *n0, double *z__, int *pp, int *
	n0in, double *dmin__, double *dmin1, double *dmin2, double *dn, 
	double *dn1, double *dn2, double *tau, int *ttype, double *g)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Builtin functions
    double sqrt(double);

    // Local variables
    double s, a2, b1, b2;
    int i4, nn, np;
    double gam, gap1, gap2;

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    A negative DMIN forces the shift to take that absolute value
    //    TTYPE records the type of shift.
    //
    // Parameter adjustments
    --z__;

    // Function Body
    if (*dmin__ <= 0.) {
	*tau = -(*dmin__);
	*ttype = -1;
	return 0;
    }
    nn = (*n0 << 2) + *pp;
    if (*n0in == *n0) {
	//
	//       No eigenvalues deflated.
	//
	if (*dmin__ == *dn || *dmin__ == *dn1) {
	    b1 = sqrt(z__[nn - 3]) * sqrt(z__[nn - 5]);
	    b2 = sqrt(z__[nn - 7]) * sqrt(z__[nn - 9]);
	    a2 = z__[nn - 7] + z__[nn - 5];
	    //
	    //          Cases 2 and 3.
	    //
	    if (*dmin__ == *dn && *dmin1 == *dn1) {
		gap2 = *dmin2 - a2 - *dmin2 * .25;
		if (gap2 > 0. && gap2 > b2) {
		    gap1 = a2 - *dn - b2 / gap2 * b2;
		} else {
		    gap1 = a2 - *dn - (b1 + b2);
		}
		if (gap1 > 0. && gap1 > b1) {
		    // Computing MAX
		    d__1 = *dn - b1 / gap1 * b1, d__2 = *dmin__ * .5;
		    s = max(d__1,d__2);
		    *ttype = -2;
		} else {
		    s = 0.;
		    if (*dn > b1) {
			s = *dn - b1;
		    }
		    if (a2 > b1 + b2) {
			// Computing MIN
			d__1 = s, d__2 = a2 - (b1 + b2);
			s = min(d__1,d__2);
		    }
		    // Computing MAX
		    d__1 = s, d__2 = *dmin__ * .333;
		    s = max(d__1,d__2);
		    *ttype = -3;
		}
	    } else {
		//
		//             Case 4.
		//
		*ttype = -4;
		s = *dmin__ * .25;
		if (*dmin__ == *dn) {
		    gam = *dn;
		    a2 = 0.;
		    if (z__[nn - 5] > z__[nn - 7]) {
			return 0;
		    }
		    b2 = z__[nn - 5] / z__[nn - 7];
		    np = nn - 9;
		} else {
		    np = nn - (*pp << 1);
		    gam = *dn1;
		    if (z__[np - 4] > z__[np - 2]) {
			return 0;
		    }
		    a2 = z__[np - 4] / z__[np - 2];
		    if (z__[nn - 9] > z__[nn - 11]) {
			return 0;
		    }
		    b2 = z__[nn - 9] / z__[nn - 11];
		    np = nn - 13;
		}
		//
		//             Approximate contribution to norm squared from I < NN-1.
		//
		a2 += b2;
		i__1 = (*i0 << 2) - 1 + *pp;
		for (i4 = np; i4 >= i__1; i4 += -4) {
		    if (b2 == 0.) {
			goto L20;
		    }
		    b1 = b2;
		    if (z__[i4] > z__[i4 - 2]) {
			return 0;
		    }
		    b2 *= z__[i4] / z__[i4 - 2];
		    a2 += b2;
		    if (max(b2,b1) * 100. < a2 || .563 < a2) {
			goto L20;
		    }
// L10:
		}
L20:
		a2 *= 1.05;
		//
		//             Rayleigh quotient residual bound.
		//
		if (a2 < .563) {
		    s = gam * (1. - sqrt(a2)) / (a2 + 1.);
		}
	    }
	} else if (*dmin__ == *dn2) {
	    //
	    //          Case 5.
	    //
	    *ttype = -5;
	    s = *dmin__ * .25;
	    //
	    //          Compute contribution to norm squared from I > NN-2.
	    //
	    np = nn - (*pp << 1);
	    b1 = z__[np - 2];
	    b2 = z__[np - 6];
	    gam = *dn2;
	    if (z__[np - 8] > b2 || z__[np - 4] > b1) {
		return 0;
	    }
	    a2 = z__[np - 8] / b2 * (z__[np - 4] / b1 + 1.);
	    //
	    //          Approximate contribution to norm squared from I < NN-2.
	    //
	    if (*n0 - *i0 > 2) {
		b2 = z__[nn - 13] / z__[nn - 15];
		a2 += b2;
		i__1 = (*i0 << 2) - 1 + *pp;
		for (i4 = nn - 17; i4 >= i__1; i4 += -4) {
		    if (b2 == 0.) {
			goto L40;
		    }
		    b1 = b2;
		    if (z__[i4] > z__[i4 - 2]) {
			return 0;
		    }
		    b2 *= z__[i4] / z__[i4 - 2];
		    a2 += b2;
		    if (max(b2,b1) * 100. < a2 || .563 < a2) {
			goto L40;
		    }
// L30:
		}
L40:
		a2 *= 1.05;
	    }
	    if (a2 < .563) {
		s = gam * (1. - sqrt(a2)) / (a2 + 1.);
	    }
	} else {
	    //
	    //          Case 6, no information to guide us.
	    //
	    if (*ttype == -6) {
		*g += (1. - *g) * .333;
	    } else if (*ttype == -18) {
		*g = .083250000000000005;
	    } else {
		*g = .25;
	    }
	    s = *g * *dmin__;
	    *ttype = -6;
	}
    } else if (*n0in == *n0 + 1) {
	//
	//       One eigenvalue just deflated. Use DMIN1, DN1 for DMIN and DN.
	//
	if (*dmin1 == *dn1 && *dmin2 == *dn2) {
	    //
	    //          Cases 7 and 8.
	    //
	    *ttype = -7;
	    s = *dmin1 * .333;
	    if (z__[nn - 5] > z__[nn - 7]) {
		return 0;
	    }
	    b1 = z__[nn - 5] / z__[nn - 7];
	    b2 = b1;
	    if (b2 == 0.) {
		goto L60;
	    }
	    i__1 = (*i0 << 2) - 1 + *pp;
	    for (i4 = (*n0 << 2) - 9 + *pp; i4 >= i__1; i4 += -4) {
		a2 = b1;
		if (z__[i4] > z__[i4 - 2]) {
		    return 0;
		}
		b1 *= z__[i4] / z__[i4 - 2];
		b2 += b1;
		if (max(b1,a2) * 100. < b2) {
		    goto L60;
		}
// L50:
	    }
L60:
	    b2 = sqrt(b2 * 1.05);
	    // Computing 2nd power
	    d__1 = b2;
	    a2 = *dmin1 / (d__1 * d__1 + 1.);
	    gap2 = *dmin2 * .5 - a2;
	    if (gap2 > 0. && gap2 > b2 * a2) {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - a2 * 1.01 * (b2 / gap2) * b2);
		s = max(d__1,d__2);
	    } else {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - b2 * 1.01);
		s = max(d__1,d__2);
		*ttype = -8;
	    }
	} else {
	    //
	    //          Case 9.
	    //
	    s = *dmin1 * .25;
	    if (*dmin1 == *dn1) {
		s = *dmin1 * .5;
	    }
	    *ttype = -9;
	}
    } else if (*n0in == *n0 + 2) {
	//
	//       Two eigenvalues deflated. Use DMIN2, DN2 for DMIN and DN.
	//
	//       Cases 10 and 11.
	//
	if (*dmin2 == *dn2 && z__[nn - 5] * 2. < z__[nn - 7]) {
	    *ttype = -10;
	    s = *dmin2 * .333;
	    if (z__[nn - 5] > z__[nn - 7]) {
		return 0;
	    }
	    b1 = z__[nn - 5] / z__[nn - 7];
	    b2 = b1;
	    if (b2 == 0.) {
		goto L80;
	    }
	    i__1 = (*i0 << 2) - 1 + *pp;
	    for (i4 = (*n0 << 2) - 9 + *pp; i4 >= i__1; i4 += -4) {
		if (z__[i4] > z__[i4 - 2]) {
		    return 0;
		}
		b1 *= z__[i4] / z__[i4 - 2];
		b2 += b1;
		if (b1 * 100. < b2) {
		    goto L80;
		}
// L70:
	    }
L80:
	    b2 = sqrt(b2 * 1.05);
	    // Computing 2nd power
	    d__1 = b2;
	    a2 = *dmin2 / (d__1 * d__1 + 1.);
	    gap2 = z__[nn - 7] + z__[nn - 9] - sqrt(z__[nn - 11]) * sqrt(z__[
		    nn - 9]) - a2;
	    if (gap2 > 0. && gap2 > b2 * a2) {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - a2 * 1.01 * (b2 / gap2) * b2);
		s = max(d__1,d__2);
	    } else {
		// Computing MAX
		d__1 = s, d__2 = a2 * (1. - b2 * 1.01);
		s = max(d__1,d__2);
	    }
	} else {
	    s = *dmin2 * .25;
	    *ttype = -11;
	}
    } else if (*n0in > *n0 + 2) {
	//
	//       Case 12, more than two eigenvalues deflated. No information.
	//
	s = 0.;
	*ttype = -12;
    }
    *tau = s;
    return 0;
    //
    //    End of DLASQ4
    //
} // dlasq4_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ5 computes one dqds transform in ping-pong form. Used by sbdsqr and sstegr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ5 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq5.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq5.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq5.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ5( I0, N0, Z, PP, TAU, SIGMA, DMIN, DMIN1, DMIN2, DN,
//                         DNM1, DNM2, IEEE, EPS )
//
//      .. Scalar Arguments ..
//      LOGICAL            IEEE
//      INTEGER            I0, N0, PP
//      DOUBLE PRECISION   DMIN, DMIN1, DMIN2, DN, DNM1, DNM2, TAU, SIGMA, EPS
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ5 computes one dqds transform in ping-pong form, one
//> version for IEEE machines another for non IEEE machines.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>        First index.
//> \endverbatim
//>
//> \param[in] N0
//> \verbatim
//>          N0 is INTEGER
//>        Last index.
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N )
//>        Z holds the qd array. EMIN is stored in Z(4*N0) to avoid
//>        an extra argument.
//> \endverbatim
//>
//> \param[in] PP
//> \verbatim
//>          PP is INTEGER
//>        PP=0 for ping, PP=1 for pong.
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION
//>        This is the shift.
//> \endverbatim
//>
//> \param[in] SIGMA
//> \verbatim
//>          SIGMA is DOUBLE PRECISION
//>        This is the accumulated shift up to this step.
//> \endverbatim
//>
//> \param[out] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>        Minimum value of d.
//> \endverbatim
//>
//> \param[out] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ).
//> \endverbatim
//>
//> \param[out] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
//> \endverbatim
//>
//> \param[out] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//>        d(N0), the last value of d.
//> \endverbatim
//>
//> \param[out] DNM1
//> \verbatim
//>          DNM1 is DOUBLE PRECISION
//>        d(N0-1).
//> \endverbatim
//>
//> \param[out] DNM2
//> \verbatim
//>          DNM2 is DOUBLE PRECISION
//>        d(N0-2).
//> \endverbatim
//>
//> \param[in] IEEE
//> \verbatim
//>          IEEE is LOGICAL
//>        Flag for IEEE or non IEEE arithmetic.
//> \endverbatim
//>
//> \param[in] EPS
//> \verbatim
//>          EPS is DOUBLE PRECISION
//>        This is the value of epsilon used.
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
//> \date June 2017
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasq5_(int *i0, int *n0, double *z__, int *pp, double *
	tau, double *sigma, double *dmin__, double *dmin1, double *dmin2, 
	double *dn, double *dnm1, double *dnm2, int *ieee, double *eps)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Local variables
    double d__;
    int j4, j4p2;
    double emin, temp, dthresh;

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
    //    .. Parameter ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --z__;

    // Function Body
    if (*n0 - *i0 - 1 <= 0) {
	return 0;
    }
    dthresh = *eps * (*sigma + *tau);
    if (*tau < dthresh * .5) {
	*tau = 0.;
    }
    if (*tau != 0.) {
	j4 = (*i0 << 2) + *pp - 3;
	emin = z__[j4 + 4];
	d__ = z__[j4] - *tau;
	*dmin__ = d__;
	*dmin1 = -z__[j4];
	if (*ieee) {
	    //
	    //       Code for IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    temp = z__[j4 + 1] / z__[j4 - 2];
		    d__ = d__ * temp - *tau;
		    *dmin__ = min(*dmin__,d__);
		    z__[j4] = z__[j4 - 1] * temp;
		    // Computing MIN
		    d__1 = z__[j4];
		    emin = min(d__1,emin);
// L10:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    temp = z__[j4 + 2] / z__[j4 - 3];
		    d__ = d__ * temp - *tau;
		    *dmin__ = min(*dmin__,d__);
		    z__[j4 - 1] = z__[j4] * temp;
		    // Computing MIN
		    d__1 = z__[j4 - 1];
		    emin = min(d__1,emin);
// L20:
		}
	    }
	    //
	    //       Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dn);
	} else {
	    //
	    //       Code for non IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
			d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4];
		    emin = min(d__1,d__2);
// L30:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
			d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4 - 1];
		    emin = min(d__1,d__2);
// L40:
		}
	    }
	    //
	    //       Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    if (*dnm2 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    if (*dnm1 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dn);
	}
    } else {
	//    This is the version that sets d's to zero if they are small enough
	j4 = (*i0 << 2) + *pp - 3;
	emin = z__[j4 + 4];
	d__ = z__[j4] - *tau;
	*dmin__ = d__;
	*dmin1 = -z__[j4];
	if (*ieee) {
	    //
	    //    Code for IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    temp = z__[j4 + 1] / z__[j4 - 2];
		    d__ = d__ * temp - *tau;
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    z__[j4] = z__[j4 - 1] * temp;
		    // Computing MIN
		    d__1 = z__[j4];
		    emin = min(d__1,emin);
// L50:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    temp = z__[j4 + 2] / z__[j4 - 3];
		    d__ = d__ * temp - *tau;
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    z__[j4 - 1] = z__[j4] * temp;
		    // Computing MIN
		    d__1 = z__[j4 - 1];
		    emin = min(d__1,emin);
// L60:
		}
	    }
	    //
	    //    Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    *dmin__ = min(*dmin__,*dn);
	} else {
	    //
	    //    Code for non IEEE arithmetic.
	    //
	    if (*pp == 0) {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 2] = d__ + z__[j4 - 1];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
			d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		    }
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4];
		    emin = min(d__1,d__2);
// L70:
		}
	    } else {
		i__1 = *n0 - 3 << 2;
		for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		    z__[j4 - 3] = d__ + z__[j4];
		    if (d__ < 0.) {
			return 0;
		    } else {
			z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
			d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		    }
		    if (d__ < dthresh) {
			d__ = 0.;
		    }
		    *dmin__ = min(*dmin__,d__);
		    // Computing MIN
		    d__1 = emin, d__2 = z__[j4 - 1];
		    emin = min(d__1,d__2);
// L80:
		}
	    }
	    //
	    //    Unroll last two steps.
	    //
	    *dnm2 = d__;
	    *dmin2 = *dmin__;
	    j4 = (*n0 - 2 << 2) - *pp;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm2 + z__[j4p2];
	    if (*dnm2 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dnm1);
	    *dmin1 = *dmin__;
	    j4 += 4;
	    j4p2 = j4 + (*pp << 1) - 1;
	    z__[j4 - 2] = *dnm1 + z__[j4p2];
	    if (*dnm1 < 0.) {
		return 0;
	    } else {
		z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
		*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	    }
	    *dmin__ = min(*dmin__,*dn);
	}
    }
    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return 0;
    //
    //    End of DLASQ5
    //
} // dlasq5_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLASQ6 computes one dqd transform in ping-pong form. Used by sbdsqr and sstegr.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASQ6 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasq6.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasq6.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasq6.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASQ6( I0, N0, Z, PP, DMIN, DMIN1, DMIN2, DN,
//                         DNM1, DNM2 )
//
//      .. Scalar Arguments ..
//      INTEGER            I0, N0, PP
//      DOUBLE PRECISION   DMIN, DMIN1, DMIN2, DN, DNM1, DNM2
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLASQ6 computes one dqd (shift equal to zero) transform in
//> ping-pong form, with protection against underflow and overflow.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] I0
//> \verbatim
//>          I0 is INTEGER
//>        First index.
//> \endverbatim
//>
//> \param[in] N0
//> \verbatim
//>          N0 is INTEGER
//>        Last index.
//> \endverbatim
//>
//> \param[in] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension ( 4*N )
//>        Z holds the qd array. EMIN is stored in Z(4*N0) to avoid
//>        an extra argument.
//> \endverbatim
//>
//> \param[in] PP
//> \verbatim
//>          PP is INTEGER
//>        PP=0 for ping, PP=1 for pong.
//> \endverbatim
//>
//> \param[out] DMIN
//> \verbatim
//>          DMIN is DOUBLE PRECISION
//>        Minimum value of d.
//> \endverbatim
//>
//> \param[out] DMIN1
//> \verbatim
//>          DMIN1 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ).
//> \endverbatim
//>
//> \param[out] DMIN2
//> \verbatim
//>          DMIN2 is DOUBLE PRECISION
//>        Minimum value of d, excluding D( N0 ) and D( N0-1 ).
//> \endverbatim
//>
//> \param[out] DN
//> \verbatim
//>          DN is DOUBLE PRECISION
//>        d(N0), the last value of d.
//> \endverbatim
//>
//> \param[out] DNM1
//> \verbatim
//>          DNM1 is DOUBLE PRECISION
//>        d(N0-1).
//> \endverbatim
//>
//> \param[out] DNM2
//> \verbatim
//>          DNM2 is DOUBLE PRECISION
//>        d(N0-2).
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
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasq6_(int *i0, int *n0, double *z__, int *pp, double *
	dmin__, double *dmin1, double *dmin2, double *dn, double *dnm1, 
	double *dnm2)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Local variables
    double d__;
    int j4, j4p2;
    double emin, temp;
    extern double dlamch_(char *);
    double safmin;

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
    //    .. Parameter ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Function ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --z__;

    // Function Body
    if (*n0 - *i0 - 1 <= 0) {
	return 0;
    }
    safmin = dlamch_("Safe minimum");
    j4 = (*i0 << 2) + *pp - 3;
    emin = z__[j4 + 4];
    d__ = z__[j4];
    *dmin__ = d__;
    if (*pp == 0) {
	i__1 = *n0 - 3 << 2;
	for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
	    z__[j4 - 2] = d__ + z__[j4 - 1];
	    if (z__[j4 - 2] == 0.) {
		z__[j4] = 0.;
		d__ = z__[j4 + 1];
		*dmin__ = d__;
		emin = 0.;
	    } else if (safmin * z__[j4 + 1] < z__[j4 - 2] && safmin * z__[j4 
		    - 2] < z__[j4 + 1]) {
		temp = z__[j4 + 1] / z__[j4 - 2];
		z__[j4] = z__[j4 - 1] * temp;
		d__ *= temp;
	    } else {
		z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
		d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]);
	    }
	    *dmin__ = min(*dmin__,d__);
	    // Computing MIN
	    d__1 = emin, d__2 = z__[j4];
	    emin = min(d__1,d__2);
// L10:
	}
    } else {
	i__1 = *n0 - 3 << 2;
	for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
	    z__[j4 - 3] = d__ + z__[j4];
	    if (z__[j4 - 3] == 0.) {
		z__[j4 - 1] = 0.;
		d__ = z__[j4 + 2];
		*dmin__ = d__;
		emin = 0.;
	    } else if (safmin * z__[j4 + 2] < z__[j4 - 3] && safmin * z__[j4 
		    - 3] < z__[j4 + 2]) {
		temp = z__[j4 + 2] / z__[j4 - 3];
		z__[j4 - 1] = z__[j4] * temp;
		d__ *= temp;
	    } else {
		z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
		d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]);
	    }
	    *dmin__ = min(*dmin__,d__);
	    // Computing MIN
	    d__1 = emin, d__2 = z__[j4 - 1];
	    emin = min(d__1,d__2);
// L20:
	}
    }
    //
    //    Unroll last two steps.
    //
    *dnm2 = d__;
    *dmin2 = *dmin__;
    j4 = (*n0 - 2 << 2) - *pp;
    j4p2 = j4 + (*pp << 1) - 1;
    z__[j4 - 2] = *dnm2 + z__[j4p2];
    if (z__[j4 - 2] == 0.) {
	z__[j4] = 0.;
	*dnm1 = z__[j4p2 + 2];
	*dmin__ = *dnm1;
	emin = 0.;
    } else if (safmin * z__[j4p2 + 2] < z__[j4 - 2] && safmin * z__[j4 - 2] < 
	    z__[j4p2 + 2]) {
	temp = z__[j4p2 + 2] / z__[j4 - 2];
	z__[j4] = z__[j4p2] * temp;
	*dnm1 = *dnm2 * temp;
    } else {
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]);
    }
    *dmin__ = min(*dmin__,*dnm1);
    *dmin1 = *dmin__;
    j4 += 4;
    j4p2 = j4 + (*pp << 1) - 1;
    z__[j4 - 2] = *dnm1 + z__[j4p2];
    if (z__[j4 - 2] == 0.) {
	z__[j4] = 0.;
	*dn = z__[j4p2 + 2];
	*dmin__ = *dn;
	emin = 0.;
    } else if (safmin * z__[j4p2 + 2] < z__[j4 - 2] && safmin * z__[j4 - 2] < 
	    z__[j4p2 + 2]) {
	temp = z__[j4p2 + 2] / z__[j4 - 2];
	z__[j4] = z__[j4p2] * temp;
	*dn = *dnm1 * temp;
    } else {
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]);
    }
    *dmin__ = min(*dmin__,*dn);
    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return 0;
    //
    //    End of DLASQ6
    //
} // dlasq6_

