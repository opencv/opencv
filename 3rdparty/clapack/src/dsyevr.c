/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLAE2 computes the eigenvalues of a 2-by-2 symmetric matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAE2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlae2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlae2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlae2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAE2( A, B, C, RT1, RT2 )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   A, B, C, RT1, RT2
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAE2  computes the eigenvalues of a 2-by-2 symmetric matrix
//>    [  A   B  ]
//>    [  B   C  ].
//> On return, RT1 is the eigenvalue of larger absolute value, and RT2
//> is the eigenvalue of smaller absolute value.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION
//>          The (1,1) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] B
//> \verbatim
//>          B is DOUBLE PRECISION
//>          The (1,2) and (2,1) elements of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] C
//> \verbatim
//>          C is DOUBLE PRECISION
//>          The (2,2) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[out] RT1
//> \verbatim
//>          RT1 is DOUBLE PRECISION
//>          The eigenvalue of larger absolute value.
//> \endverbatim
//>
//> \param[out] RT2
//> \verbatim
//>          RT2 is DOUBLE PRECISION
//>          The eigenvalue of smaller absolute value.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  RT1 is accurate to a few ulps barring over/underflow.
//>
//>  RT2 may be inaccurate if there is massive cancellation in the
//>  determinant A*C-B*B; higher precision or correctly rounded or
//>  correctly truncated arithmetic would be needed to compute RT2
//>  accurately in all cases.
//>
//>  Overflow is possible only if RT1 is within a factor of 5 of overflow.
//>  Underflow is harmless if the input data is 0 or exceeds
//>     underflow_threshold / macheps.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlae2_(double *a, double *b, double *c__, double *rt1, 
	double *rt2)
{
    // System generated locals
    double d__1;

    // Builtin functions
    double sqrt(double);

    // Local variables
    double ab, df, tb, sm, rt, adf, acmn, acmx;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Compute the eigenvalues
    //
    sm = *a + *c__;
    df = *a - *c__;
    adf = abs(df);
    tb = *b + *b;
    ab = abs(tb);
    if (abs(*a) > abs(*c__)) {
	acmx = *a;
	acmn = *c__;
    } else {
	acmx = *c__;
	acmn = *a;
    }
    if (adf > ab) {
	// Computing 2nd power
	d__1 = ab / adf;
	rt = adf * sqrt(d__1 * d__1 + 1.);
    } else if (adf < ab) {
	// Computing 2nd power
	d__1 = adf / ab;
	rt = ab * sqrt(d__1 * d__1 + 1.);
    } else {
	//
	//       Includes case AB=ADF=0
	//
	rt = ab * sqrt(2.);
    }
    if (sm < 0.) {
	*rt1 = (sm - rt) * .5;
	//
	//       Order of execution important.
	//       To get fully accurate smaller eigenvalue,
	//       next line needs to be executed in higher precision.
	//
	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else if (sm > 0.) {
	*rt1 = (sm + rt) * .5;
	//
	//       Order of execution important.
	//       To get fully accurate smaller eigenvalue,
	//       next line needs to be executed in higher precision.
	//
	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else {
	//
	//       Includes case RT1 = RT2 = 0
	//
	*rt1 = rt * .5;
	*rt2 = rt * -.5;
    }
    return 0;
    //
    //    End of DLAE2
    //
} // dlae2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAEBZ computes the number of eigenvalues of a real symmetric tridiagonal matrix which are less than or equal to a given value, and performs other tasks required by the routine sstebz.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAEBZ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaebz.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaebz.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaebz.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAEBZ( IJOB, NITMAX, N, MMAX, MINP, NBMIN, ABSTOL,
//                         RELTOL, PIVMIN, D, E, E2, NVAL, AB, C, MOUT,
//                         NAB, WORK, IWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IJOB, INFO, MINP, MMAX, MOUT, N, NBMIN, NITMAX
//      DOUBLE PRECISION   ABSTOL, PIVMIN, RELTOL
//      ..
//      .. Array Arguments ..
//      INTEGER            IWORK( * ), NAB( MMAX, * ), NVAL( * )
//      DOUBLE PRECISION   AB( MMAX, * ), C( * ), D( * ), E( * ), E2( * ),
//     $                   WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAEBZ contains the iteration loops which compute and use the
//> function N(w), which is the count of eigenvalues of a symmetric
//> tridiagonal matrix T less than or equal to its argument  w.  It
//> performs a choice of two types of loops:
//>
//> IJOB=1, followed by
//> IJOB=2: It takes as input a list of intervals and returns a list of
//>         sufficiently small intervals whose union contains the same
//>         eigenvalues as the union of the original intervals.
//>         The input intervals are (AB(j,1),AB(j,2)], j=1,...,MINP.
//>         The output interval (AB(j,1),AB(j,2)] will contain
//>         eigenvalues NAB(j,1)+1,...,NAB(j,2), where 1 <= j <= MOUT.
//>
//> IJOB=3: It performs a binary search in each input interval
//>         (AB(j,1),AB(j,2)] for a point  w(j)  such that
//>         N(w(j))=NVAL(j), and uses  C(j)  as the starting point of
//>         the search.  If such a w(j) is found, then on output
//>         AB(j,1)=AB(j,2)=w.  If no such w(j) is found, then on output
//>         (AB(j,1),AB(j,2)] will be a small interval containing the
//>         point where N(w) jumps through NVAL(j), unless that point
//>         lies outside the initial interval.
//>
//> Note that the intervals are in all cases half-open intervals,
//> i.e., of the form  (a,b] , which includes  b  but not  a .
//>
//> To avoid underflow, the matrix should be scaled so that its largest
//> element is no greater than  overflow**(1/2) * underflow**(1/4)
//> in absolute value.  To assure the most accurate computation
//> of small eigenvalues, the matrix should be scaled to be
//> not much smaller than that, either.
//>
//> See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
//> Matrix", Report CS41, Computer Science Dept., Stanford
//> University, July 21, 1966
//>
//> Note: the arguments are, in general, *not* checked for unreasonable
//> values.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] IJOB
//> \verbatim
//>          IJOB is INTEGER
//>          Specifies what is to be done:
//>          = 1:  Compute NAB for the initial intervals.
//>          = 2:  Perform bisection iteration to find eigenvalues of T.
//>          = 3:  Perform bisection iteration to invert N(w), i.e.,
//>                to find a point which has a specified number of
//>                eigenvalues of T to its left.
//>          Other values will cause DLAEBZ to return with INFO=-1.
//> \endverbatim
//>
//> \param[in] NITMAX
//> \verbatim
//>          NITMAX is INTEGER
//>          The maximum number of "levels" of bisection to be
//>          performed, i.e., an interval of width W will not be made
//>          smaller than 2^(-NITMAX) * W.  If not all intervals
//>          have converged after NITMAX iterations, then INFO is set
//>          to the number of non-converged intervals.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The dimension n of the tridiagonal matrix T.  It must be at
//>          least 1.
//> \endverbatim
//>
//> \param[in] MMAX
//> \verbatim
//>          MMAX is INTEGER
//>          The maximum number of intervals.  If more than MMAX intervals
//>          are generated, then DLAEBZ will quit with INFO=MMAX+1.
//> \endverbatim
//>
//> \param[in] MINP
//> \verbatim
//>          MINP is INTEGER
//>          The initial number of intervals.  It may not be greater than
//>          MMAX.
//> \endverbatim
//>
//> \param[in] NBMIN
//> \verbatim
//>          NBMIN is INTEGER
//>          The smallest number of intervals that should be processed
//>          using a vector loop.  If zero, then only the scalar loop
//>          will be used.
//> \endverbatim
//>
//> \param[in] ABSTOL
//> \verbatim
//>          ABSTOL is DOUBLE PRECISION
//>          The minimum (absolute) width of an interval.  When an
//>          interval is narrower than ABSTOL, or than RELTOL times the
//>          larger (in magnitude) endpoint, then it is considered to be
//>          sufficiently small, i.e., converged.  This must be at least
//>          zero.
//> \endverbatim
//>
//> \param[in] RELTOL
//> \verbatim
//>          RELTOL is DOUBLE PRECISION
//>          The minimum relative width of an interval.  When an interval
//>          is narrower than ABSTOL, or than RELTOL times the larger (in
//>          magnitude) endpoint, then it is considered to be
//>          sufficiently small, i.e., converged.  Note: this should
//>          always be at least radix*machine epsilon.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum absolute value of a "pivot" in the Sturm
//>          sequence loop.
//>          This must be at least  max |e(j)**2|*safe_min  and at
//>          least safe_min, where safe_min is at least
//>          the smallest number that can divide one without overflow.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N)
//>          The offdiagonal elements of the tridiagonal matrix T in
//>          positions 1 through N-1.  E(N) is arbitrary.
//> \endverbatim
//>
//> \param[in] E2
//> \verbatim
//>          E2 is DOUBLE PRECISION array, dimension (N)
//>          The squares of the offdiagonal elements of the tridiagonal
//>          matrix T.  E2(N) is ignored.
//> \endverbatim
//>
//> \param[in,out] NVAL
//> \verbatim
//>          NVAL is INTEGER array, dimension (MINP)
//>          If IJOB=1 or 2, not referenced.
//>          If IJOB=3, the desired values of N(w).  The elements of NVAL
//>          will be reordered to correspond with the intervals in AB.
//>          Thus, NVAL(j) on output will not, in general be the same as
//>          NVAL(j) on input, but it will correspond with the interval
//>          (AB(j,1),AB(j,2)] on output.
//> \endverbatim
//>
//> \param[in,out] AB
//> \verbatim
//>          AB is DOUBLE PRECISION array, dimension (MMAX,2)
//>          The endpoints of the intervals.  AB(j,1) is  a(j), the left
//>          endpoint of the j-th interval, and AB(j,2) is b(j), the
//>          right endpoint of the j-th interval.  The input intervals
//>          will, in general, be modified, split, and reordered by the
//>          calculation.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (MMAX)
//>          If IJOB=1, ignored.
//>          If IJOB=2, workspace.
//>          If IJOB=3, then on input C(j) should be initialized to the
//>          first search point in the binary search.
//> \endverbatim
//>
//> \param[out] MOUT
//> \verbatim
//>          MOUT is INTEGER
//>          If IJOB=1, the number of eigenvalues in the intervals.
//>          If IJOB=2 or 3, the number of intervals output.
//>          If IJOB=3, MOUT will equal MINP.
//> \endverbatim
//>
//> \param[in,out] NAB
//> \verbatim
//>          NAB is INTEGER array, dimension (MMAX,2)
//>          If IJOB=1, then on output NAB(i,j) will be set to N(AB(i,j)).
//>          If IJOB=2, then on input, NAB(i,j) should be set.  It must
//>             satisfy the condition:
//>             N(AB(i,1)) <= NAB(i,1) <= NAB(i,2) <= N(AB(i,2)),
//>             which means that in interval i only eigenvalues
//>             NAB(i,1)+1,...,NAB(i,2) will be considered.  Usually,
//>             NAB(i,j)=N(AB(i,j)), from a previous call to DLAEBZ with
//>             IJOB=1.
//>             On output, NAB(i,j) will contain
//>             max(na(k),min(nb(k),N(AB(i,j)))), where k is the index of
//>             the input interval that the output interval
//>             (AB(j,1),AB(j,2)] came from, and na(k) and nb(k) are the
//>             the input values of NAB(k,1) and NAB(k,2).
//>          If IJOB=3, then on output, NAB(i,j) contains N(AB(i,j)),
//>             unless N(w) > NVAL(i) for all search points  w , in which
//>             case NAB(i,1) will not be modified, i.e., the output
//>             value will be the same as the input value (modulo
//>             reorderings -- see NVAL and AB), or unless N(w) < NVAL(i)
//>             for all search points  w , in which case NAB(i,2) will
//>             not be modified.  Normally, NAB should be set to some
//>             distinctive value(s) before DLAEBZ is called.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MMAX)
//>          Workspace.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (MMAX)
//>          Workspace.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:       All intervals converged.
//>          = 1--MMAX: The last INFO intervals did not converge.
//>          = MMAX+1:  More than MMAX intervals were generated.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>      This routine is intended to be called only by other LAPACK
//>  routines, thus the interface is less user-friendly.  It is intended
//>  for two purposes:
//>
//>  (a) finding eigenvalues.  In this case, DLAEBZ should have one or
//>      more initial intervals set up in AB, and DLAEBZ should be called
//>      with IJOB=1.  This sets up NAB, and also counts the eigenvalues.
//>      Intervals with no eigenvalues would usually be thrown out at
//>      this point.  Also, if not all the eigenvalues in an interval i
//>      are desired, NAB(i,1) can be increased or NAB(i,2) decreased.
//>      For example, set NAB(i,1)=NAB(i,2)-1 to get the largest
//>      eigenvalue.  DLAEBZ is then called with IJOB=2 and MMAX
//>      no smaller than the value of MOUT returned by the call with
//>      IJOB=1.  After this (IJOB=2) call, eigenvalues NAB(i,1)+1
//>      through NAB(i,2) are approximately AB(i,1) (or AB(i,2)) to the
//>      tolerance specified by ABSTOL and RELTOL.
//>
//>  (b) finding an interval (a',b'] containing eigenvalues w(f),...,w(l).
//>      In this case, start with a Gershgorin interval  (a,b).  Set up
//>      AB to contain 2 search intervals, both initially (a,b).  One
//>      NVAL element should contain  f-1  and the other should contain  l
//>      , while C should contain a and b, resp.  NAB(i,1) should be -1
//>      and NAB(i,2) should be N+1, to flag an error if the desired
//>      interval does not lie in (a,b).  DLAEBZ is then called with
//>      IJOB=3.  On exit, if w(f-1) < w(f), then one of the intervals --
//>      j -- will have AB(j,1)=AB(j,2) and NAB(j,1)=NAB(j,2)=f-1, while
//>      if, to the specified tolerance, w(f-k)=...=w(f+r), k > 0 and r
//>      >= 0, then the interval will have  N(AB(j,1))=NAB(j,1)=f-k and
//>      N(AB(j,2))=NAB(j,2)=f+r.  The cases w(l) < w(l+1) and
//>      w(l-r)=...=w(l+k) are handled similarly.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlaebz_(int *ijob, int *nitmax, int *n, int *mmax, int *
	minp, int *nbmin, double *abstol, double *reltol, double *pivmin, 
	double *d__, double *e, double *e2, int *nval, double *ab, double *
	c__, int *mout, int *nab, double *work, int *iwork, int *info)
{
    // System generated locals
    int nab_dim1, nab_offset, ab_dim1, ab_offset, i__1, i__2, i__3, i__4, 
	    i__5, i__6;
    double d__1, d__2, d__3, d__4;

    // Local variables
    int j, kf, ji, kl, jp, jit;
    double tmp1, tmp2;
    int itmp1, itmp2, kfnew, klnew;

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Check for Errors
    //
    // Parameter adjustments
    nab_dim1 = *mmax;
    nab_offset = 1 + nab_dim1;
    nab -= nab_offset;
    ab_dim1 = *mmax;
    ab_offset = 1 + ab_dim1;
    ab -= ab_offset;
    --d__;
    --e;
    --e2;
    --nval;
    --c__;
    --work;
    --iwork;

    // Function Body
    *info = 0;
    if (*ijob < 1 || *ijob > 3) {
	*info = -1;
	return 0;
    }
    //
    //    Initialize NAB
    //
    if (*ijob == 1) {
	//
	//       Compute the number of eigenvalues in the initial intervals.
	//
	*mout = 0;
	i__1 = *minp;
	for (ji = 1; ji <= i__1; ++ji) {
	    for (jp = 1; jp <= 2; ++jp) {
		tmp1 = d__[1] - ab[ji + jp * ab_dim1];
		if (abs(tmp1) < *pivmin) {
		    tmp1 = -(*pivmin);
		}
		nab[ji + jp * nab_dim1] = 0;
		if (tmp1 <= 0.) {
		    nab[ji + jp * nab_dim1] = 1;
		}
		i__2 = *n;
		for (j = 2; j <= i__2; ++j) {
		    tmp1 = d__[j] - e2[j - 1] / tmp1 - ab[ji + jp * ab_dim1];
		    if (abs(tmp1) < *pivmin) {
			tmp1 = -(*pivmin);
		    }
		    if (tmp1 <= 0.) {
			++nab[ji + jp * nab_dim1];
		    }
// L10:
		}
// L20:
	    }
	    *mout = *mout + nab[ji + (nab_dim1 << 1)] - nab[ji + nab_dim1];
// L30:
	}
	return 0;
    }
    //
    //    Initialize for loop
    //
    //    KF and KL have the following meaning:
    //       Intervals 1,...,KF-1 have converged.
    //       Intervals KF,...,KL  still need to be refined.
    //
    kf = 1;
    kl = *minp;
    //
    //    If IJOB=2, initialize C.
    //    If IJOB=3, use the user-supplied starting point.
    //
    if (*ijob == 2) {
	i__1 = *minp;
	for (ji = 1; ji <= i__1; ++ji) {
	    c__[ji] = (ab[ji + ab_dim1] + ab[ji + (ab_dim1 << 1)]) * .5;
// L40:
	}
    }
    //
    //    Iteration loop
    //
    i__1 = *nitmax;
    for (jit = 1; jit <= i__1; ++jit) {
	//
	//       Loop over intervals
	//
	if (kl - kf + 1 >= *nbmin && *nbmin > 0) {
	    //
	    //          Begin of Parallel Version of the loop
	    //
	    i__2 = kl;
	    for (ji = kf; ji <= i__2; ++ji) {
		//
		//             Compute N(c), the number of eigenvalues less than c
		//
		work[ji] = d__[1] - c__[ji];
		iwork[ji] = 0;
		if (work[ji] <= *pivmin) {
		    iwork[ji] = 1;
		    // Computing MIN
		    d__1 = work[ji], d__2 = -(*pivmin);
		    work[ji] = min(d__1,d__2);
		}
		i__3 = *n;
		for (j = 2; j <= i__3; ++j) {
		    work[ji] = d__[j] - e2[j - 1] / work[ji] - c__[ji];
		    if (work[ji] <= *pivmin) {
			++iwork[ji];
			// Computing MIN
			d__1 = work[ji], d__2 = -(*pivmin);
			work[ji] = min(d__1,d__2);
		    }
// L50:
		}
// L60:
	    }
	    if (*ijob <= 2) {
		//
		//             IJOB=2: Choose all intervals containing eigenvalues.
		//
		klnew = kl;
		i__2 = kl;
		for (ji = kf; ji <= i__2; ++ji) {
		    //
		    //                Insure that N(w) is monotone
		    //
		    // Computing MIN
		    // Computing MAX
		    i__5 = nab[ji + nab_dim1], i__6 = iwork[ji];
		    i__3 = nab[ji + (nab_dim1 << 1)], i__4 = max(i__5,i__6);
		    iwork[ji] = min(i__3,i__4);
		    //
		    //                Update the Queue -- add intervals if both halves
		    //                contain eigenvalues.
		    //
		    if (iwork[ji] == nab[ji + (nab_dim1 << 1)]) {
			//
			//                   No eigenvalue in the upper interval:
			//                   just use the lower interval.
			//
			ab[ji + (ab_dim1 << 1)] = c__[ji];
		    } else if (iwork[ji] == nab[ji + nab_dim1]) {
			//
			//                   No eigenvalue in the lower interval:
			//                   just use the upper interval.
			//
			ab[ji + ab_dim1] = c__[ji];
		    } else {
			++klnew;
			if (klnew <= *mmax) {
			    //
			    //                      Eigenvalue in both intervals -- add upper to
			    //                      queue.
			    //
			    ab[klnew + (ab_dim1 << 1)] = ab[ji + (ab_dim1 << 
				    1)];
			    nab[klnew + (nab_dim1 << 1)] = nab[ji + (nab_dim1 
				    << 1)];
			    ab[klnew + ab_dim1] = c__[ji];
			    nab[klnew + nab_dim1] = iwork[ji];
			    ab[ji + (ab_dim1 << 1)] = c__[ji];
			    nab[ji + (nab_dim1 << 1)] = iwork[ji];
			} else {
			    *info = *mmax + 1;
			}
		    }
// L70:
		}
		if (*info != 0) {
		    return 0;
		}
		kl = klnew;
	    } else {
		//
		//             IJOB=3: Binary search.  Keep only the interval containing
		//                     w   s.t. N(w) = NVAL
		//
		i__2 = kl;
		for (ji = kf; ji <= i__2; ++ji) {
		    if (iwork[ji] <= nval[ji]) {
			ab[ji + ab_dim1] = c__[ji];
			nab[ji + nab_dim1] = iwork[ji];
		    }
		    if (iwork[ji] >= nval[ji]) {
			ab[ji + (ab_dim1 << 1)] = c__[ji];
			nab[ji + (nab_dim1 << 1)] = iwork[ji];
		    }
// L80:
		}
	    }
	} else {
	    //
	    //          End of Parallel Version of the loop
	    //
	    //          Begin of Serial Version of the loop
	    //
	    klnew = kl;
	    i__2 = kl;
	    for (ji = kf; ji <= i__2; ++ji) {
		//
		//             Compute N(w), the number of eigenvalues less than w
		//
		tmp1 = c__[ji];
		tmp2 = d__[1] - tmp1;
		itmp1 = 0;
		if (tmp2 <= *pivmin) {
		    itmp1 = 1;
		    // Computing MIN
		    d__1 = tmp2, d__2 = -(*pivmin);
		    tmp2 = min(d__1,d__2);
		}
		i__3 = *n;
		for (j = 2; j <= i__3; ++j) {
		    tmp2 = d__[j] - e2[j - 1] / tmp2 - tmp1;
		    if (tmp2 <= *pivmin) {
			++itmp1;
			// Computing MIN
			d__1 = tmp2, d__2 = -(*pivmin);
			tmp2 = min(d__1,d__2);
		    }
// L90:
		}
		if (*ijob <= 2) {
		    //
		    //                IJOB=2: Choose all intervals containing eigenvalues.
		    //
		    //                Insure that N(w) is monotone
		    //
		    // Computing MIN
		    // Computing MAX
		    i__5 = nab[ji + nab_dim1];
		    i__3 = nab[ji + (nab_dim1 << 1)], i__4 = max(i__5,itmp1);
		    itmp1 = min(i__3,i__4);
		    //
		    //                Update the Queue -- add intervals if both halves
		    //                contain eigenvalues.
		    //
		    if (itmp1 == nab[ji + (nab_dim1 << 1)]) {
			//
			//                   No eigenvalue in the upper interval:
			//                   just use the lower interval.
			//
			ab[ji + (ab_dim1 << 1)] = tmp1;
		    } else if (itmp1 == nab[ji + nab_dim1]) {
			//
			//                   No eigenvalue in the lower interval:
			//                   just use the upper interval.
			//
			ab[ji + ab_dim1] = tmp1;
		    } else if (klnew < *mmax) {
			//
			//                   Eigenvalue in both intervals -- add upper to queue.
			//
			++klnew;
			ab[klnew + (ab_dim1 << 1)] = ab[ji + (ab_dim1 << 1)];
			nab[klnew + (nab_dim1 << 1)] = nab[ji + (nab_dim1 << 
				1)];
			ab[klnew + ab_dim1] = tmp1;
			nab[klnew + nab_dim1] = itmp1;
			ab[ji + (ab_dim1 << 1)] = tmp1;
			nab[ji + (nab_dim1 << 1)] = itmp1;
		    } else {
			*info = *mmax + 1;
			return 0;
		    }
		} else {
		    //
		    //                IJOB=3: Binary search.  Keep only the interval
		    //                        containing  w  s.t. N(w) = NVAL
		    //
		    if (itmp1 <= nval[ji]) {
			ab[ji + ab_dim1] = tmp1;
			nab[ji + nab_dim1] = itmp1;
		    }
		    if (itmp1 >= nval[ji]) {
			ab[ji + (ab_dim1 << 1)] = tmp1;
			nab[ji + (nab_dim1 << 1)] = itmp1;
		    }
		}
// L100:
	    }
	    kl = klnew;
	}
	//
	//       Check for convergence
	//
	kfnew = kf;
	i__2 = kl;
	for (ji = kf; ji <= i__2; ++ji) {
	    tmp1 = (d__1 = ab[ji + (ab_dim1 << 1)] - ab[ji + ab_dim1], abs(
		    d__1));
	    // Computing MAX
	    d__3 = (d__1 = ab[ji + (ab_dim1 << 1)], abs(d__1)), d__4 = (d__2 =
		     ab[ji + ab_dim1], abs(d__2));
	    tmp2 = max(d__3,d__4);
	    // Computing MAX
	    d__1 = max(*abstol,*pivmin), d__2 = *reltol * tmp2;
	    if (tmp1 < max(d__1,d__2) || nab[ji + nab_dim1] >= nab[ji + (
		    nab_dim1 << 1)]) {
		//
		//             Converged -- Swap with position KFNEW,
		//                          then increment KFNEW
		//
		if (ji > kfnew) {
		    tmp1 = ab[ji + ab_dim1];
		    tmp2 = ab[ji + (ab_dim1 << 1)];
		    itmp1 = nab[ji + nab_dim1];
		    itmp2 = nab[ji + (nab_dim1 << 1)];
		    ab[ji + ab_dim1] = ab[kfnew + ab_dim1];
		    ab[ji + (ab_dim1 << 1)] = ab[kfnew + (ab_dim1 << 1)];
		    nab[ji + nab_dim1] = nab[kfnew + nab_dim1];
		    nab[ji + (nab_dim1 << 1)] = nab[kfnew + (nab_dim1 << 1)];
		    ab[kfnew + ab_dim1] = tmp1;
		    ab[kfnew + (ab_dim1 << 1)] = tmp2;
		    nab[kfnew + nab_dim1] = itmp1;
		    nab[kfnew + (nab_dim1 << 1)] = itmp2;
		    if (*ijob == 3) {
			itmp1 = nval[ji];
			nval[ji] = nval[kfnew];
			nval[kfnew] = itmp1;
		    }
		}
		++kfnew;
	    }
// L110:
	}
	kf = kfnew;
	//
	//       Choose Midpoints
	//
	i__2 = kl;
	for (ji = kf; ji <= i__2; ++ji) {
	    c__[ji] = (ab[ji + ab_dim1] + ab[ji + (ab_dim1 << 1)]) * .5;
// L120:
	}
	//
	//       If no more intervals to refine, quit.
	//
	if (kf > kl) {
	    goto L140;
	}
// L130:
    }
    //
    //    Converged
    //
L140:
    // Computing MAX
    i__1 = kl + 1 - kf;
    *info = max(i__1,0);
    *mout = kl;
    return 0;
    //
    //    End of DLAEBZ
    //
} // dlaebz_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAEV2 computes the eigenvalues and eigenvectors of a 2-by-2 symmetric/Hermitian matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAEV2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaev2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaev2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaev2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAEV2( A, B, C, RT1, RT2, CS1, SN1 )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION   A, B, C, CS1, RT1, RT2, SN1
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAEV2 computes the eigendecomposition of a 2-by-2 symmetric matrix
//>    [  A   B  ]
//>    [  B   C  ].
//> On return, RT1 is the eigenvalue of larger absolute value, RT2 is the
//> eigenvalue of smaller absolute value, and (CS1,SN1) is the unit right
//> eigenvector for RT1, giving the decomposition
//>
//>    [ CS1  SN1 ] [  A   B  ] [ CS1 -SN1 ]  =  [ RT1  0  ]
//>    [-SN1  CS1 ] [  B   C  ] [ SN1  CS1 ]     [  0  RT2 ].
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION
//>          The (1,1) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] B
//> \verbatim
//>          B is DOUBLE PRECISION
//>          The (1,2) element and the conjugate of the (2,1) element of
//>          the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[in] C
//> \verbatim
//>          C is DOUBLE PRECISION
//>          The (2,2) element of the 2-by-2 matrix.
//> \endverbatim
//>
//> \param[out] RT1
//> \verbatim
//>          RT1 is DOUBLE PRECISION
//>          The eigenvalue of larger absolute value.
//> \endverbatim
//>
//> \param[out] RT2
//> \verbatim
//>          RT2 is DOUBLE PRECISION
//>          The eigenvalue of smaller absolute value.
//> \endverbatim
//>
//> \param[out] CS1
//> \verbatim
//>          CS1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] SN1
//> \verbatim
//>          SN1 is DOUBLE PRECISION
//>          The vector (CS1, SN1) is a unit right eigenvector for RT1.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  RT1 is accurate to a few ulps barring over/underflow.
//>
//>  RT2 may be inaccurate if there is massive cancellation in the
//>  determinant A*C-B*B; higher precision or correctly rounded or
//>  correctly truncated arithmetic would be needed to compute RT2
//>  accurately in all cases.
//>
//>  CS1 and SN1 are accurate to a few ulps barring over/underflow.
//>
//>  Overflow is possible only if RT1 is within a factor of 5 of overflow.
//>  Underflow is harmless if the input data is 0 or exceeds
//>     underflow_threshold / macheps.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlaev2_(double *a, double *b, double *c__, double *rt1, 
	double *rt2, double *cs1, double *sn1)
{
    // System generated locals
    double d__1;

    // Builtin functions
    double sqrt(double);

    // Local variables
    double ab, df, cs, ct, tb, sm, tn, rt, adf, acs;
    int sgn1, sgn2;
    double acmn, acmx;

    //
    // -- LAPACK auxiliary routine (version 3.7.0) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    December 2016
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Compute the eigenvalues
    //
    sm = *a + *c__;
    df = *a - *c__;
    adf = abs(df);
    tb = *b + *b;
    ab = abs(tb);
    if (abs(*a) > abs(*c__)) {
	acmx = *a;
	acmn = *c__;
    } else {
	acmx = *c__;
	acmn = *a;
    }
    if (adf > ab) {
	// Computing 2nd power
	d__1 = ab / adf;
	rt = adf * sqrt(d__1 * d__1 + 1.);
    } else if (adf < ab) {
	// Computing 2nd power
	d__1 = adf / ab;
	rt = ab * sqrt(d__1 * d__1 + 1.);
    } else {
	//
	//       Includes case AB=ADF=0
	//
	rt = ab * sqrt(2.);
    }
    if (sm < 0.) {
	*rt1 = (sm - rt) * .5;
	sgn1 = -1;
	//
	//       Order of execution important.
	//       To get fully accurate smaller eigenvalue,
	//       next line needs to be executed in higher precision.
	//
	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else if (sm > 0.) {
	*rt1 = (sm + rt) * .5;
	sgn1 = 1;
	//
	//       Order of execution important.
	//       To get fully accurate smaller eigenvalue,
	//       next line needs to be executed in higher precision.
	//
	*rt2 = acmx / *rt1 * acmn - *b / *rt1 * *b;
    } else {
	//
	//       Includes case RT1 = RT2 = 0
	//
	*rt1 = rt * .5;
	*rt2 = rt * -.5;
	sgn1 = 1;
    }
    //
    //    Compute the eigenvector
    //
    if (df >= 0.) {
	cs = df + rt;
	sgn2 = 1;
    } else {
	cs = df - rt;
	sgn2 = -1;
    }
    acs = abs(cs);
    if (acs > ab) {
	ct = -tb / cs;
	*sn1 = 1. / sqrt(ct * ct + 1.);
	*cs1 = ct * *sn1;
    } else {
	if (ab == 0.) {
	    *cs1 = 1.;
	    *sn1 = 0.;
	} else {
	    tn = -cs / tb;
	    *cs1 = 1. / sqrt(tn * tn + 1.);
	    *sn1 = tn * *cs1;
	}
    }
    if (sgn1 == sgn2) {
	tn = *cs1;
	*cs1 = -(*sn1);
	*sn1 = tn;
    }
    return 0;
    //
    //    End of DLAEV2
    //
} // dlaev2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAGTF computes an LU factorization of a matrix T-λI, where T is a general tridiagonal matrix, and λ a scalar, using partial pivoting with row interchanges.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAGTF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlagtf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlagtf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlagtf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAGTF( N, A, LAMBDA, B, C, TOL, D, IN, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, N
//      DOUBLE PRECISION   LAMBDA, TOL
//      ..
//      .. Array Arguments ..
//      INTEGER            IN( * )
//      DOUBLE PRECISION   A( * ), B( * ), C( * ), D( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAGTF factorizes the matrix (T - lambda*I), where T is an n by n
//> tridiagonal matrix and lambda is a scalar, as
//>
//>    T - lambda*I = PLU,
//>
//> where P is a permutation matrix, L is a unit lower tridiagonal matrix
//> with at most one non-zero sub-diagonal elements per column and U is
//> an upper triangular matrix with at most two non-zero super-diagonal
//> elements per column.
//>
//> The factorization is obtained by Gaussian elimination with partial
//> pivoting and implicit row scaling.
//>
//> The parameter LAMBDA is included in the routine so that DLAGTF may
//> be used, in conjunction with DLAGTS, to obtain eigenvectors of T by
//> inverse iteration.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix T.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (N)
//>          On entry, A must contain the diagonal elements of T.
//>
//>          On exit, A is overwritten by the n diagonal elements of the
//>          upper triangular matrix U of the factorization of T.
//> \endverbatim
//>
//> \param[in] LAMBDA
//> \verbatim
//>          LAMBDA is DOUBLE PRECISION
//>          On entry, the scalar lambda.
//> \endverbatim
//>
//> \param[in,out] B
//> \verbatim
//>          B is DOUBLE PRECISION array, dimension (N-1)
//>          On entry, B must contain the (n-1) super-diagonal elements of
//>          T.
//>
//>          On exit, B is overwritten by the (n-1) super-diagonal
//>          elements of the matrix U of the factorization of T.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (N-1)
//>          On entry, C must contain the (n-1) sub-diagonal elements of
//>          T.
//>
//>          On exit, C is overwritten by the (n-1) sub-diagonal elements
//>          of the matrix L of the factorization of T.
//> \endverbatim
//>
//> \param[in] TOL
//> \verbatim
//>          TOL is DOUBLE PRECISION
//>          On entry, a relative tolerance used to indicate whether or
//>          not the matrix (T - lambda*I) is nearly singular. TOL should
//>          normally be chose as approximately the largest relative error
//>          in the elements of T. For example, if the elements of T are
//>          correct to about 4 significant figures, then TOL should be
//>          set to about 5*10**(-4). If TOL is supplied as less than eps,
//>          where eps is the relative machine precision, then the value
//>          eps is used in place of TOL.
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N-2)
//>          On exit, D is overwritten by the (n-2) second super-diagonal
//>          elements of the matrix U of the factorization of T.
//> \endverbatim
//>
//> \param[out] IN
//> \verbatim
//>          IN is INTEGER array, dimension (N)
//>          On exit, IN contains details of the permutation matrix P. If
//>          an interchange occurred at the kth step of the elimination,
//>          then IN(k) = 1, otherwise IN(k) = 0. The element IN(n)
//>          returns the smallest positive integer j such that
//>
//>             abs( u(j,j) ) <= norm( (T - lambda*I)(j) )*TOL,
//>
//>          where norm( A(j) ) denotes the sum of the absolute values of
//>          the jth row of the matrix A. If no such j exists then IN(n)
//>          is returned as zero. If IN(n) is returned as positive, then a
//>          diagonal element of U is small, indicating that
//>          (T - lambda*I) is singular or nearly singular,
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -k, the kth argument had an illegal value
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
/* Subroutine */ int dlagtf_(int *n, double *a, double *lambda, double *b, 
	double *c__, double *tol, double *d__, int *in, int *info)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Local variables
    int k;
    double tl, eps, piv1, piv2, temp, mult, scale1, scale2;
    extern double dlamch_(char *);
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
    //=====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --in;
    --d__;
    --c__;
    --b;
    --a;

    // Function Body
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("DLAGTF", &i__1);
	return 0;
    }
    if (*n == 0) {
	return 0;
    }
    a[1] -= *lambda;
    in[*n] = 0;
    if (*n == 1) {
	if (a[1] == 0.) {
	    in[1] = 1;
	}
	return 0;
    }
    eps = dlamch_("Epsilon");
    tl = max(*tol,eps);
    scale1 = abs(a[1]) + abs(b[1]);
    i__1 = *n - 1;
    for (k = 1; k <= i__1; ++k) {
	a[k + 1] -= *lambda;
	scale2 = (d__1 = c__[k], abs(d__1)) + (d__2 = a[k + 1], abs(d__2));
	if (k < *n - 1) {
	    scale2 += (d__1 = b[k + 1], abs(d__1));
	}
	if (a[k] == 0.) {
	    piv1 = 0.;
	} else {
	    piv1 = (d__1 = a[k], abs(d__1)) / scale1;
	}
	if (c__[k] == 0.) {
	    in[k] = 0;
	    piv2 = 0.;
	    scale1 = scale2;
	    if (k < *n - 1) {
		d__[k] = 0.;
	    }
	} else {
	    piv2 = (d__1 = c__[k], abs(d__1)) / scale2;
	    if (piv2 <= piv1) {
		in[k] = 0;
		scale1 = scale2;
		c__[k] /= a[k];
		a[k + 1] -= c__[k] * b[k];
		if (k < *n - 1) {
		    d__[k] = 0.;
		}
	    } else {
		in[k] = 1;
		mult = a[k] / c__[k];
		a[k] = c__[k];
		temp = a[k + 1];
		a[k + 1] = b[k] - mult * temp;
		if (k < *n - 1) {
		    d__[k] = b[k + 1];
		    b[k + 1] = -mult * d__[k];
		}
		b[k] = temp;
		c__[k] = mult;
	    }
	}
	if (max(piv1,piv2) <= tl && in[*n] == 0) {
	    in[*n] = k;
	}
// L10:
    }
    if ((d__1 = a[*n], abs(d__1)) <= scale1 * tl && in[*n] == 0) {
	in[*n] = *n;
    }
    return 0;
    //
    //    End of DLAGTF
    //
} // dlagtf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAGTS solves the system of equations (T-λI)x = y or (T-λI)Tx = y,where T is a general tridiagonal matrix and λ a scalar, using the LU factorization computed by slagtf.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAGTS + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlagts.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlagts.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlagts.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAGTS( JOB, N, A, B, C, D, IN, Y, TOL, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, JOB, N
//      DOUBLE PRECISION   TOL
//      ..
//      .. Array Arguments ..
//      INTEGER            IN( * )
//      DOUBLE PRECISION   A( * ), B( * ), C( * ), D( * ), Y( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAGTS may be used to solve one of the systems of equations
//>
//>    (T - lambda*I)*x = y   or   (T - lambda*I)**T*x = y,
//>
//> where T is an n by n tridiagonal matrix, for x, following the
//> factorization of (T - lambda*I) as
//>
//>    (T - lambda*I) = P*L*U ,
//>
//> by routine DLAGTF. The choice of equation to be solved is
//> controlled by the argument JOB, and in each case there is an option
//> to perturb zero or very small diagonal elements of U, this option
//> being intended for use in applications such as inverse iteration.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOB
//> \verbatim
//>          JOB is INTEGER
//>          Specifies the job to be performed by DLAGTS as follows:
//>          =  1: The equations  (T - lambda*I)x = y  are to be solved,
//>                but diagonal elements of U are not to be perturbed.
//>          = -1: The equations  (T - lambda*I)x = y  are to be solved
//>                and, if overflow would otherwise occur, the diagonal
//>                elements of U are to be perturbed. See argument TOL
//>                below.
//>          =  2: The equations  (T - lambda*I)**Tx = y  are to be solved,
//>                but diagonal elements of U are not to be perturbed.
//>          = -2: The equations  (T - lambda*I)**Tx = y  are to be solved
//>                and, if overflow would otherwise occur, the diagonal
//>                elements of U are to be perturbed. See argument TOL
//>                below.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix T.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (N)
//>          On entry, A must contain the diagonal elements of U as
//>          returned from DLAGTF.
//> \endverbatim
//>
//> \param[in] B
//> \verbatim
//>          B is DOUBLE PRECISION array, dimension (N-1)
//>          On entry, B must contain the first super-diagonal elements of
//>          U as returned from DLAGTF.
//> \endverbatim
//>
//> \param[in] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (N-1)
//>          On entry, C must contain the sub-diagonal elements of L as
//>          returned from DLAGTF.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N-2)
//>          On entry, D must contain the second super-diagonal elements
//>          of U as returned from DLAGTF.
//> \endverbatim
//>
//> \param[in] IN
//> \verbatim
//>          IN is INTEGER array, dimension (N)
//>          On entry, IN must contain details of the matrix P as returned
//>          from DLAGTF.
//> \endverbatim
//>
//> \param[in,out] Y
//> \verbatim
//>          Y is DOUBLE PRECISION array, dimension (N)
//>          On entry, the right hand side vector y.
//>          On exit, Y is overwritten by the solution vector x.
//> \endverbatim
//>
//> \param[in,out] TOL
//> \verbatim
//>          TOL is DOUBLE PRECISION
//>          On entry, with  JOB < 0, TOL should be the minimum
//>          perturbation to be made to very small diagonal elements of U.
//>          TOL should normally be chosen as about eps*norm(U), where eps
//>          is the relative machine precision, but if TOL is supplied as
//>          non-positive, then it is reset to eps*max( abs( u(i,j) ) ).
//>          If  JOB > 0  then TOL is not referenced.
//>
//>          On exit, TOL is changed as described above, only if TOL is
//>          non-positive on entry. Otherwise TOL is unchanged.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  overflow would occur when computing the INFO(th)
//>                element of the solution vector x. This can only occur
//>                when JOB is supplied as positive and either means
//>                that a diagonal element of U is very small, or that
//>                the elements of the right-hand side vector y are very
//>                large.
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
/* Subroutine */ int dlagts_(int *job, int *n, double *a, double *b, double *
	c__, double *d__, int *in, double *y, double *tol, int *info)
{
    // System generated locals
    int i__1;
    double d__1, d__2, d__3, d__4, d__5;

    // Builtin functions
    double d_sign(double *, double *);

    // Local variables
    int k;
    double ak, eps, temp, pert, absak, sfmin;
    extern double dlamch_(char *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    double bignum;

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --y;
    --in;
    --d__;
    --c__;
    --b;
    --a;

    // Function Body
    *info = 0;
    if (abs(*job) > 2 || *job == 0) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLAGTS", &i__1);
	return 0;
    }
    if (*n == 0) {
	return 0;
    }
    eps = dlamch_("Epsilon");
    sfmin = dlamch_("Safe minimum");
    bignum = 1. / sfmin;
    if (*job < 0) {
	if (*tol <= 0.) {
	    *tol = abs(a[1]);
	    if (*n > 1) {
		// Computing MAX
		d__1 = *tol, d__2 = abs(a[2]), d__1 = max(d__1,d__2), d__2 = 
			abs(b[1]);
		*tol = max(d__1,d__2);
	    }
	    i__1 = *n;
	    for (k = 3; k <= i__1; ++k) {
		// Computing MAX
		d__4 = *tol, d__5 = (d__1 = a[k], abs(d__1)), d__4 = max(d__4,
			d__5), d__5 = (d__2 = b[k - 1], abs(d__2)), d__4 = 
			max(d__4,d__5), d__5 = (d__3 = d__[k - 2], abs(d__3));
		*tol = max(d__4,d__5);
// L10:
	    }
	    *tol *= eps;
	    if (*tol == 0.) {
		*tol = eps;
	    }
	}
    }
    if (abs(*job) == 1) {
	i__1 = *n;
	for (k = 2; k <= i__1; ++k) {
	    if (in[k - 1] == 0) {
		y[k] -= c__[k - 1] * y[k - 1];
	    } else {
		temp = y[k - 1];
		y[k - 1] = y[k];
		y[k] = temp - c__[k - 1] * y[k];
	    }
// L20:
	}
	if (*job == 1) {
	    for (k = *n; k >= 1; --k) {
		if (k <= *n - 2) {
		    temp = y[k] - b[k] * y[k + 1] - d__[k] * y[k + 2];
		} else if (k == *n - 1) {
		    temp = y[k] - b[k] * y[k + 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    *info = k;
			    return 0;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			*info = k;
			return 0;
		    }
		}
		y[k] = temp / ak;
// L30:
	    }
	} else {
	    for (k = *n; k >= 1; --k) {
		if (k <= *n - 2) {
		    temp = y[k] - b[k] * y[k + 1] - d__[k] * y[k + 2];
		} else if (k == *n - 1) {
		    temp = y[k] - b[k] * y[k + 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		pert = d_sign(tol, &ak);
L40:
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    ak += pert;
			    pert *= 2;
			    goto L40;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			ak += pert;
			pert *= 2;
			goto L40;
		    }
		}
		y[k] = temp / ak;
// L50:
	    }
	}
    } else {
	//
	//       Come to here if  JOB = 2 or -2
	//
	if (*job == 2) {
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		if (k >= 3) {
		    temp = y[k] - b[k - 1] * y[k - 1] - d__[k - 2] * y[k - 2];
		} else if (k == 2) {
		    temp = y[k] - b[k - 1] * y[k - 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    *info = k;
			    return 0;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			*info = k;
			return 0;
		    }
		}
		y[k] = temp / ak;
// L60:
	    }
	} else {
	    i__1 = *n;
	    for (k = 1; k <= i__1; ++k) {
		if (k >= 3) {
		    temp = y[k] - b[k - 1] * y[k - 1] - d__[k - 2] * y[k - 2];
		} else if (k == 2) {
		    temp = y[k] - b[k - 1] * y[k - 1];
		} else {
		    temp = y[k];
		}
		ak = a[k];
		pert = d_sign(tol, &ak);
L70:
		absak = abs(ak);
		if (absak < 1.) {
		    if (absak < sfmin) {
			if (absak == 0. || abs(temp) * sfmin > absak) {
			    ak += pert;
			    pert *= 2;
			    goto L70;
			} else {
			    temp *= bignum;
			    ak *= bignum;
			}
		    } else if (abs(temp) > absak * bignum) {
			ak += pert;
			pert *= 2;
			goto L70;
		    }
		}
		y[k] = temp / ak;
// L80:
	    }
	}
	for (k = *n; k >= 2; --k) {
	    if (in[k - 1] == 0) {
		y[k - 1] -= c__[k - 1] * y[k];
	    } else {
		temp = y[k - 1];
		y[k - 1] = y[k];
		y[k] = temp - c__[k - 1] * y[k];
	    }
// L90:
	}
    }
    //
    //    End of DLAGTS
    //
    return 0;
} // dlagts_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLANEG computes the Sturm count.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLANEG + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaneg.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaneg.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaneg.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      INTEGER FUNCTION DLANEG( N, D, LLD, SIGMA, PIVMIN, R )
//
//      .. Scalar Arguments ..
//      INTEGER            N, R
//      DOUBLE PRECISION   PIVMIN, SIGMA
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), LLD( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLANEG computes the Sturm count, the number of negative pivots
//> encountered while factoring tridiagonal T - sigma I = L D L^T.
//> This implementation works directly on the factors without forming
//> the tridiagonal matrix T.  The Sturm count is also the number of
//> eigenvalues of T less than sigma.
//>
//> This routine is called from DLARRB.
//>
//> The current routine does not use the PIVMIN parameter but rather
//> requires IEEE-754 propagation of Infinities and NaNs.  This
//> routine also has no input range restrictions but does require
//> default exception handling such that x/0 produces Inf when x is
//> non-zero, and Inf/Inf produces NaN.  For more information, see:
//>
//>   Marques, Riedy, and Voemel, "Benefits of IEEE-754 Features in
//>   Modern Symmetric Tridiagonal Eigensolvers," SIAM Journal on
//>   Scientific Computing, v28, n5, 2006.  DOI 10.1137/050641624
//>   (Tech report version in LAWN 172 with the same title.)
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The N diagonal elements of the diagonal matrix D.
//> \endverbatim
//>
//> \param[in] LLD
//> \verbatim
//>          LLD is DOUBLE PRECISION array, dimension (N-1)
//>          The (N-1) elements L(i)*L(i)*D(i).
//> \endverbatim
//>
//> \param[in] SIGMA
//> \verbatim
//>          SIGMA is DOUBLE PRECISION
//>          Shift amount in T - sigma I = L D L^T.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot in the Sturm sequence.  May be used
//>          when zero pivots are encountered on non-IEEE-754
//>          architectures.
//> \endverbatim
//>
//> \param[in] R
//> \verbatim
//>          R is INTEGER
//>          The twist index for the twisted factorization that is used
//>          for the negcount.
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
//> \par Contributors:
// ==================
//>
//>     Osni Marques, LBNL/NERSC, USA \n
//>     Christof Voemel, University of California, Berkeley, USA \n
//>     Jason Riedy, University of California, Berkeley, USA \n
//>
// =====================================================================
int dlaneg_(int *n, double *d__, double *lld, double *sigma, double *pivmin, 
	int *r__)
{
    // System generated locals
    int ret_val, i__1, i__2, i__3, i__4;

    // Local variables
    int j;
    double p, t;
    int bj;
    double tmp;
    int neg1, neg2;
    double bsav, gamma, dplus;
    extern int disnan_(double *);
    int negcnt;
    int sawnan;
    double dminus;

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
    //    Some architectures propagate Infinities and NaNs very slowly, so
    //    the code computes counts in BLKLEN chunks.  Then a NaN can
    //    propagate at most BLKLEN columns before being detected.  This is
    //    not a general tuning parameter; it needs only to be just large
    //    enough that the overhead is tiny in common cases.
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    // Parameter adjustments
    --lld;
    --d__;

    // Function Body
    negcnt = 0;
    //    I) upper part: L D L^T - SIGMA I = L+ D+ L+^T
    t = -(*sigma);
    i__1 = *r__ - 1;
    for (bj = 1; bj <= i__1; bj += 128) {
	neg1 = 0;
	bsav = t;
	// Computing MIN
	i__3 = bj + 127, i__4 = *r__ - 1;
	i__2 = min(i__3,i__4);
	for (j = bj; j <= i__2; ++j) {
	    dplus = d__[j] + t;
	    if (dplus < 0.) {
		++neg1;
	    }
	    tmp = t / dplus;
	    t = tmp * lld[j] - *sigma;
// L21:
	}
	sawnan = disnan_(&t);
	//    Run a slower version of the above loop if a NaN is detected.
	//    A NaN should occur only with a zero pivot after an infinite
	//    pivot.  In that case, substituting 1 for T/DPLUS is the
	//    correct limit.
	if (sawnan) {
	    neg1 = 0;
	    t = bsav;
	    // Computing MIN
	    i__3 = bj + 127, i__4 = *r__ - 1;
	    i__2 = min(i__3,i__4);
	    for (j = bj; j <= i__2; ++j) {
		dplus = d__[j] + t;
		if (dplus < 0.) {
		    ++neg1;
		}
		tmp = t / dplus;
		if (disnan_(&tmp)) {
		    tmp = 1.;
		}
		t = tmp * lld[j] - *sigma;
// L22:
	    }
	}
	negcnt += neg1;
// L210:
    }
    //
    //    II) lower part: L D L^T - SIGMA I = U- D- U-^T
    p = d__[*n] - *sigma;
    i__1 = *r__;
    for (bj = *n - 1; bj >= i__1; bj += -128) {
	neg2 = 0;
	bsav = p;
	// Computing MAX
	i__3 = bj - 127;
	i__2 = max(i__3,*r__);
	for (j = bj; j >= i__2; --j) {
	    dminus = lld[j] + p;
	    if (dminus < 0.) {
		++neg2;
	    }
	    tmp = p / dminus;
	    p = tmp * d__[j] - *sigma;
// L23:
	}
	sawnan = disnan_(&p);
	//    As above, run a slower version that substitutes 1 for Inf/Inf.
	//
	if (sawnan) {
	    neg2 = 0;
	    p = bsav;
	    // Computing MAX
	    i__3 = bj - 127;
	    i__2 = max(i__3,*r__);
	    for (j = bj; j >= i__2; --j) {
		dminus = lld[j] + p;
		if (dminus < 0.) {
		    ++neg2;
		}
		tmp = p / dminus;
		if (disnan_(&tmp)) {
		    tmp = 1.;
		}
		p = tmp * d__[j] - *sigma;
// L24:
	    }
	}
	negcnt += neg2;
// L230:
    }
    //
    //    III) Twist index
    //      T was shifted by SIGMA initially.
    gamma = t + *sigma + p;
    if (gamma < 0.) {
	++negcnt;
    }
    ret_val = negcnt;
    return ret_val;
} // dlaneg_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLANSY returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the element of largest absolute value of a real symmetric matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLANSY + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlansy.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlansy.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlansy.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      DOUBLE PRECISION FUNCTION DLANSY( NORM, UPLO, N, A, LDA, WORK )
//
//      .. Scalar Arguments ..
//      CHARACTER          NORM, UPLO
//      INTEGER            LDA, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLANSY  returns the value of the one norm,  or the Frobenius norm, or
//> the  infinity norm,  or the  element of  largest absolute value  of a
//> real symmetric matrix A.
//> \endverbatim
//>
//> \return DLANSY
//> \verbatim
//>
//>    DLANSY = ( max(abs(A(i,j))), NORM = 'M' or 'm'
//>             (
//>             ( norm1(A),         NORM = '1', 'O' or 'o'
//>             (
//>             ( normI(A),         NORM = 'I' or 'i'
//>             (
//>             ( normF(A),         NORM = 'F', 'f', 'E' or 'e'
//>
//> where  norm1  denotes the  one norm of a matrix (maximum column sum),
//> normI  denotes the  infinity norm  of a matrix  (maximum row sum) and
//> normF  denotes the  Frobenius norm of a matrix (square root of sum of
//> squares).  Note that  max(abs(A(i,j)))  is not a consistent matrix norm.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] NORM
//> \verbatim
//>          NORM is CHARACTER*1
//>          Specifies the value to be returned in DLANSY as described
//>          above.
//> \endverbatim
//>
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          Specifies whether the upper or lower triangular part of the
//>          symmetric matrix A is to be referenced.
//>          = 'U':  Upper triangular part of A is referenced
//>          = 'L':  Lower triangular part of A is referenced
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.  N >= 0.  When N = 0, DLANSY is
//>          set to zero.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          The symmetric matrix A.  If UPLO = 'U', the leading n by n
//>          upper triangular part of A contains the upper triangular part
//>          of the matrix A, and the strictly lower triangular part of A
//>          is not referenced.  If UPLO = 'L', the leading n by n lower
//>          triangular part of A contains the lower triangular part of
//>          the matrix A, and the strictly upper triangular part of A is
//>          not referenced.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(N,1).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK)),
//>          where LWORK >= N when NORM = 'I' or '1' or 'O'; otherwise,
//>          WORK is not referenced.
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
//> \ingroup doubleSYauxiliary
//
// =====================================================================
double dlansy_(char *norm, char *uplo, int *n, double *a, int *lda, double *
	work)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2;
    double ret_val, d__1;

    // Builtin functions
    double sqrt(double);

    // Local variables
    extern /* Subroutine */ int dcombssq_(double *, double *);
    int i__, j;
    double sum, ssq[2], absa;
    extern int lsame_(char *, char *);
    double value;
    extern int disnan_(double *);
    extern /* Subroutine */ int dlassq_(int *, double *, int *, double *, 
	    double *);
    double colssq[2];

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
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --work;

    // Function Body
    if (*n == 0) {
	value = 0.;
    } else if (lsame_(norm, "M")) {
	//
	//       Find max(abs(A(i,j))).
	//
	value = 0.;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    sum = (d__1 = a[i__ + j * a_dim1], abs(d__1));
		    if (value < sum || disnan_(&sum)) {
			value = sum;
		    }
// L10:
		}
// L20:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n;
		for (i__ = j; i__ <= i__2; ++i__) {
		    sum = (d__1 = a[i__ + j * a_dim1], abs(d__1));
		    if (value < sum || disnan_(&sum)) {
			value = sum;
		    }
// L30:
		}
// L40:
	    }
	}
    } else if (lsame_(norm, "I") || lsame_(norm, "O") || *(unsigned char *)
	    norm == '1') {
	//
	//       Find normI(A) ( = norm1(A), since A is symmetric).
	//
	value = 0.;
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = 0.;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    absa = (d__1 = a[i__ + j * a_dim1], abs(d__1));
		    sum += absa;
		    work[i__] += absa;
// L50:
		}
		work[j] = sum + (d__1 = a[j + j * a_dim1], abs(d__1));
// L60:
	    }
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		sum = work[i__];
		if (value < sum || disnan_(&sum)) {
		    value = sum;
		}
// L70:
	    }
	} else {
	    i__1 = *n;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		work[i__] = 0.;
// L80:
	    }
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		sum = work[j] + (d__1 = a[j + j * a_dim1], abs(d__1));
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    absa = (d__1 = a[i__ + j * a_dim1], abs(d__1));
		    sum += absa;
		    work[i__] += absa;
// L90:
		}
		if (value < sum || disnan_(&sum)) {
		    value = sum;
		}
// L100:
	    }
	}
    } else if (lsame_(norm, "F") || lsame_(norm, "E")) {
	//
	//       Find normF(A).
	//       SSQ(1) is scale
	//       SSQ(2) is sum-of-squares
	//       For better accuracy, sum each column separately.
	//
	ssq[0] = 0.;
	ssq[1] = 1.;
	//
	//       Sum off-diagonals
	//
	if (lsame_(uplo, "U")) {
	    i__1 = *n;
	    for (j = 2; j <= i__1; ++j) {
		colssq[0] = 0.;
		colssq[1] = 1.;
		i__2 = j - 1;
		dlassq_(&i__2, &a[j * a_dim1 + 1], &c__1, colssq, &colssq[1]);
		dcombssq_(ssq, colssq);
// L110:
	    }
	} else {
	    i__1 = *n - 1;
	    for (j = 1; j <= i__1; ++j) {
		colssq[0] = 0.;
		colssq[1] = 1.;
		i__2 = *n - j;
		dlassq_(&i__2, &a[j + 1 + j * a_dim1], &c__1, colssq, &colssq[
			1]);
		dcombssq_(ssq, colssq);
// L120:
	    }
	}
	ssq[1] *= 2;
	//
	//       Sum diagonal
	//
	colssq[0] = 0.;
	colssq[1] = 1.;
	i__1 = *lda + 1;
	dlassq_(n, &a[a_offset], &i__1, colssq, &colssq[1]);
	dcombssq_(ssq, colssq);
	value = ssq[0] * sqrt(ssq[1]);
    }
    ret_val = value;
    return ret_val;
    //
    //    End of DLANSY
    //
} // dlansy_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAR1V computes the (scaled) r-th column of the inverse of the submatrix in rows b1 through bn of the tridiagonal matrix LDLT - λI.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAR1V + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlar1v.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlar1v.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlar1v.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLAR1V( N, B1, BN, LAMBDA, D, L, LD, LLD,
//                 PIVMIN, GAPTOL, Z, WANTNC, NEGCNT, ZTZ, MINGMA,
//                 R, ISUPPZ, NRMINV, RESID, RQCORR, WORK )
//
//      .. Scalar Arguments ..
//      LOGICAL            WANTNC
//      INTEGER   B1, BN, N, NEGCNT, R
//      DOUBLE PRECISION   GAPTOL, LAMBDA, MINGMA, NRMINV, PIVMIN, RESID,
//     $                   RQCORR, ZTZ
//      ..
//      .. Array Arguments ..
//      INTEGER            ISUPPZ( * )
//      DOUBLE PRECISION   D( * ), L( * ), LD( * ), LLD( * ),
//     $                  WORK( * )
//      DOUBLE PRECISION Z( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLAR1V computes the (scaled) r-th column of the inverse of
//> the sumbmatrix in rows B1 through BN of the tridiagonal matrix
//> L D L**T - sigma I. When sigma is close to an eigenvalue, the
//> computed vector is an accurate eigenvector. Usually, r corresponds
//> to the index where the eigenvector is largest in magnitude.
//> The following steps accomplish this computation :
//> (a) Stationary qd transform,  L D L**T - sigma I = L(+) D(+) L(+)**T,
//> (b) Progressive qd transform, L D L**T - sigma I = U(-) D(-) U(-)**T,
//> (c) Computation of the diagonal elements of the inverse of
//>     L D L**T - sigma I by combining the above transforms, and choosing
//>     r as the index where the diagonal of the inverse is (one of the)
//>     largest in magnitude.
//> (d) Computation of the (scaled) r-th column of the inverse using the
//>     twisted factorization obtained by combining the top part of the
//>     the stationary and the bottom part of the progressive transform.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           The order of the matrix L D L**T.
//> \endverbatim
//>
//> \param[in] B1
//> \verbatim
//>          B1 is INTEGER
//>           First index of the submatrix of L D L**T.
//> \endverbatim
//>
//> \param[in] BN
//> \verbatim
//>          BN is INTEGER
//>           Last index of the submatrix of L D L**T.
//> \endverbatim
//>
//> \param[in] LAMBDA
//> \verbatim
//>          LAMBDA is DOUBLE PRECISION
//>           The shift. In order to compute an accurate eigenvector,
//>           LAMBDA should be a good approximation to an eigenvalue
//>           of L D L**T.
//> \endverbatim
//>
//> \param[in] L
//> \verbatim
//>          L is DOUBLE PRECISION array, dimension (N-1)
//>           The (n-1) subdiagonal elements of the unit bidiagonal matrix
//>           L, in elements 1 to N-1.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>           The n diagonal elements of the diagonal matrix D.
//> \endverbatim
//>
//> \param[in] LD
//> \verbatim
//>          LD is DOUBLE PRECISION array, dimension (N-1)
//>           The n-1 elements L(i)*D(i).
//> \endverbatim
//>
//> \param[in] LLD
//> \verbatim
//>          LLD is DOUBLE PRECISION array, dimension (N-1)
//>           The n-1 elements L(i)*L(i)*D(i).
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>           The minimum pivot in the Sturm sequence.
//> \endverbatim
//>
//> \param[in] GAPTOL
//> \verbatim
//>          GAPTOL is DOUBLE PRECISION
//>           Tolerance that indicates when eigenvector entries are negligible
//>           w.r.t. their contribution to the residual.
//> \endverbatim
//>
//> \param[in,out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (N)
//>           On input, all entries of Z must be set to 0.
//>           On output, Z contains the (scaled) r-th column of the
//>           inverse. The scaling is such that Z(R) equals 1.
//> \endverbatim
//>
//> \param[in] WANTNC
//> \verbatim
//>          WANTNC is LOGICAL
//>           Specifies whether NEGCNT has to be computed.
//> \endverbatim
//>
//> \param[out] NEGCNT
//> \verbatim
//>          NEGCNT is INTEGER
//>           If WANTNC is .TRUE. then NEGCNT = the number of pivots < pivmin
//>           in the  matrix factorization L D L**T, and NEGCNT = -1 otherwise.
//> \endverbatim
//>
//> \param[out] ZTZ
//> \verbatim
//>          ZTZ is DOUBLE PRECISION
//>           The square of the 2-norm of Z.
//> \endverbatim
//>
//> \param[out] MINGMA
//> \verbatim
//>          MINGMA is DOUBLE PRECISION
//>           The reciprocal of the largest (in magnitude) diagonal
//>           element of the inverse of L D L**T - sigma I.
//> \endverbatim
//>
//> \param[in,out] R
//> \verbatim
//>          R is INTEGER
//>           The twist index for the twisted factorization used to
//>           compute Z.
//>           On input, 0 <= R <= N. If R is input as 0, R is set to
//>           the index where (L D L**T - sigma I)^{-1} is largest
//>           in magnitude. If 1 <= R <= N, R is unchanged.
//>           On output, R contains the twist index used to compute Z.
//>           Ideally, R designates the position of the maximum entry in the
//>           eigenvector.
//> \endverbatim
//>
//> \param[out] ISUPPZ
//> \verbatim
//>          ISUPPZ is INTEGER array, dimension (2)
//>           The support of the vector in Z, i.e., the vector Z is
//>           nonzero only in elements ISUPPZ(1) through ISUPPZ( 2 ).
//> \endverbatim
//>
//> \param[out] NRMINV
//> \verbatim
//>          NRMINV is DOUBLE PRECISION
//>           NRMINV = 1/SQRT( ZTZ )
//> \endverbatim
//>
//> \param[out] RESID
//> \verbatim
//>          RESID is DOUBLE PRECISION
//>           The residual of the FP vector.
//>           RESID = ABS( MINGMA )/SQRT( ZTZ )
//> \endverbatim
//>
//> \param[out] RQCORR
//> \verbatim
//>          RQCORR is DOUBLE PRECISION
//>           The Rayleigh Quotient correction to LAMBDA.
//>           RQCORR = MINGMA*TMP
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (4*N)
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
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlar1v_(int *n, int *b1, int *bn, double *lambda, double 
	*d__, double *l, double *ld, double *lld, double *pivmin, double *
	gaptol, double *z__, int *wantnc, int *negcnt, double *ztz, double *
	mingma, int *r__, int *isuppz, double *nrminv, double *resid, double *
	rqcorr, double *work)
{
    // System generated locals
    int i__1;
    double d__1, d__2, d__3;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int i__;
    double s;
    int r1, r2;
    double eps, tmp;
    int neg1, neg2, indp, inds;
    double dplus;
    extern double dlamch_(char *);
    extern int disnan_(double *);
    int indlpl, indumn;
    double dminus;
    int sawnan1, sawnan2;

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --work;
    --isuppz;
    --z__;
    --lld;
    --ld;
    --l;
    --d__;

    // Function Body
    eps = dlamch_("Precision");
    if (*r__ == 0) {
	r1 = *b1;
	r2 = *bn;
    } else {
	r1 = *r__;
	r2 = *r__;
    }
    //    Storage for LPLUS
    indlpl = 0;
    //    Storage for UMINUS
    indumn = *n;
    inds = (*n << 1) + 1;
    indp = *n * 3 + 1;
    if (*b1 == 1) {
	work[inds] = 0.;
    } else {
	work[inds + *b1 - 1] = lld[*b1 - 1];
    }
    //
    //    Compute the stationary transform (using the differential form)
    //    until the index R2.
    //
    sawnan1 = FALSE_;
    neg1 = 0;
    s = work[inds + *b1 - 1] - *lambda;
    i__1 = r1 - 1;
    for (i__ = *b1; i__ <= i__1; ++i__) {
	dplus = d__[i__] + s;
	work[indlpl + i__] = ld[i__] / dplus;
	if (dplus < 0.) {
	    ++neg1;
	}
	work[inds + i__] = s * work[indlpl + i__] * l[i__];
	s = work[inds + i__] - *lambda;
// L50:
    }
    sawnan1 = disnan_(&s);
    if (sawnan1) {
	goto L60;
    }
    i__1 = r2 - 1;
    for (i__ = r1; i__ <= i__1; ++i__) {
	dplus = d__[i__] + s;
	work[indlpl + i__] = ld[i__] / dplus;
	work[inds + i__] = s * work[indlpl + i__] * l[i__];
	s = work[inds + i__] - *lambda;
// L51:
    }
    sawnan1 = disnan_(&s);
L60:
    if (sawnan1) {
	//       Runs a slower version of the above loop if a NaN is detected
	neg1 = 0;
	s = work[inds + *b1 - 1] - *lambda;
	i__1 = r1 - 1;
	for (i__ = *b1; i__ <= i__1; ++i__) {
	    dplus = d__[i__] + s;
	    if (abs(dplus) < *pivmin) {
		dplus = -(*pivmin);
	    }
	    work[indlpl + i__] = ld[i__] / dplus;
	    if (dplus < 0.) {
		++neg1;
	    }
	    work[inds + i__] = s * work[indlpl + i__] * l[i__];
	    if (work[indlpl + i__] == 0.) {
		work[inds + i__] = lld[i__];
	    }
	    s = work[inds + i__] - *lambda;
// L70:
	}
	i__1 = r2 - 1;
	for (i__ = r1; i__ <= i__1; ++i__) {
	    dplus = d__[i__] + s;
	    if (abs(dplus) < *pivmin) {
		dplus = -(*pivmin);
	    }
	    work[indlpl + i__] = ld[i__] / dplus;
	    work[inds + i__] = s * work[indlpl + i__] * l[i__];
	    if (work[indlpl + i__] == 0.) {
		work[inds + i__] = lld[i__];
	    }
	    s = work[inds + i__] - *lambda;
// L71:
	}
    }
    //
    //    Compute the progressive transform (using the differential form)
    //    until the index R1
    //
    sawnan2 = FALSE_;
    neg2 = 0;
    work[indp + *bn - 1] = d__[*bn] - *lambda;
    i__1 = r1;
    for (i__ = *bn - 1; i__ >= i__1; --i__) {
	dminus = lld[i__] + work[indp + i__];
	tmp = d__[i__] / dminus;
	if (dminus < 0.) {
	    ++neg2;
	}
	work[indumn + i__] = l[i__] * tmp;
	work[indp + i__ - 1] = work[indp + i__] * tmp - *lambda;
// L80:
    }
    tmp = work[indp + r1 - 1];
    sawnan2 = disnan_(&tmp);
    if (sawnan2) {
	//       Runs a slower version of the above loop if a NaN is detected
	neg2 = 0;
	i__1 = r1;
	for (i__ = *bn - 1; i__ >= i__1; --i__) {
	    dminus = lld[i__] + work[indp + i__];
	    if (abs(dminus) < *pivmin) {
		dminus = -(*pivmin);
	    }
	    tmp = d__[i__] / dminus;
	    if (dminus < 0.) {
		++neg2;
	    }
	    work[indumn + i__] = l[i__] * tmp;
	    work[indp + i__ - 1] = work[indp + i__] * tmp - *lambda;
	    if (tmp == 0.) {
		work[indp + i__ - 1] = d__[i__] - *lambda;
	    }
// L100:
	}
    }
    //
    //    Find the index (from R1 to R2) of the largest (in magnitude)
    //    diagonal element of the inverse
    //
    *mingma = work[inds + r1 - 1] + work[indp + r1 - 1];
    if (*mingma < 0.) {
	++neg1;
    }
    if (*wantnc) {
	*negcnt = neg1 + neg2;
    } else {
	*negcnt = -1;
    }
    if (abs(*mingma) == 0.) {
	*mingma = eps * work[inds + r1 - 1];
    }
    *r__ = r1;
    i__1 = r2 - 1;
    for (i__ = r1; i__ <= i__1; ++i__) {
	tmp = work[inds + i__] + work[indp + i__];
	if (tmp == 0.) {
	    tmp = eps * work[inds + i__];
	}
	if (abs(tmp) <= abs(*mingma)) {
	    *mingma = tmp;
	    *r__ = i__ + 1;
	}
// L110:
    }
    //
    //    Compute the FP vector: solve N^T v = e_r
    //
    isuppz[1] = *b1;
    isuppz[2] = *bn;
    z__[*r__] = 1.;
    *ztz = 1.;
    //
    //    Compute the FP vector upwards from R
    //
    if (! sawnan1 && ! sawnan2) {
	i__1 = *b1;
	for (i__ = *r__ - 1; i__ >= i__1; --i__) {
	    z__[i__] = -(work[indlpl + i__] * z__[i__ + 1]);
	    if (((d__1 = z__[i__], abs(d__1)) + (d__2 = z__[i__ + 1], abs(
		    d__2))) * (d__3 = ld[i__], abs(d__3)) < *gaptol) {
		z__[i__] = 0.;
		isuppz[1] = i__ + 1;
		goto L220;
	    }
	    *ztz += z__[i__] * z__[i__];
// L210:
	}
L220:
	;
    } else {
	//       Run slower loop if NaN occurred.
	i__1 = *b1;
	for (i__ = *r__ - 1; i__ >= i__1; --i__) {
	    if (z__[i__ + 1] == 0.) {
		z__[i__] = -(ld[i__ + 1] / ld[i__]) * z__[i__ + 2];
	    } else {
		z__[i__] = -(work[indlpl + i__] * z__[i__ + 1]);
	    }
	    if (((d__1 = z__[i__], abs(d__1)) + (d__2 = z__[i__ + 1], abs(
		    d__2))) * (d__3 = ld[i__], abs(d__3)) < *gaptol) {
		z__[i__] = 0.;
		isuppz[1] = i__ + 1;
		goto L240;
	    }
	    *ztz += z__[i__] * z__[i__];
// L230:
	}
L240:
	;
    }
    //    Compute the FP vector downwards from R in blocks of size BLKSIZ
    if (! sawnan1 && ! sawnan2) {
	i__1 = *bn - 1;
	for (i__ = *r__; i__ <= i__1; ++i__) {
	    z__[i__ + 1] = -(work[indumn + i__] * z__[i__]);
	    if (((d__1 = z__[i__], abs(d__1)) + (d__2 = z__[i__ + 1], abs(
		    d__2))) * (d__3 = ld[i__], abs(d__3)) < *gaptol) {
		z__[i__ + 1] = 0.;
		isuppz[2] = i__;
		goto L260;
	    }
	    *ztz += z__[i__ + 1] * z__[i__ + 1];
// L250:
	}
L260:
	;
    } else {
	//       Run slower loop if NaN occurred.
	i__1 = *bn - 1;
	for (i__ = *r__; i__ <= i__1; ++i__) {
	    if (z__[i__] == 0.) {
		z__[i__ + 1] = -(ld[i__ - 1] / ld[i__]) * z__[i__ - 1];
	    } else {
		z__[i__ + 1] = -(work[indumn + i__] * z__[i__]);
	    }
	    if (((d__1 = z__[i__], abs(d__1)) + (d__2 = z__[i__ + 1], abs(
		    d__2))) * (d__3 = ld[i__], abs(d__3)) < *gaptol) {
		z__[i__ + 1] = 0.;
		isuppz[2] = i__;
		goto L280;
	    }
	    *ztz += z__[i__ + 1] * z__[i__ + 1];
// L270:
	}
L280:
	;
    }
    //
    //    Compute quantities for convergence test
    //
    tmp = 1. / *ztz;
    *nrminv = sqrt(tmp);
    *resid = abs(*mingma) * *nrminv;
    *rqcorr = *mingma * tmp;
    return 0;
    //
    //    End of DLAR1V
    //
} // dlar1v_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARNV returns a vector of random numbers from a uniform or normal distribution.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARNV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarnv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarnv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarnv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARNV( IDIST, ISEED, N, X )
//
//      .. Scalar Arguments ..
//      INTEGER            IDIST, N
//      ..
//      .. Array Arguments ..
//      INTEGER            ISEED( 4 )
//      DOUBLE PRECISION   X( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLARNV returns a vector of n random real numbers from a uniform or
//> normal distribution.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] IDIST
//> \verbatim
//>          IDIST is INTEGER
//>          Specifies the distribution of the random numbers:
//>          = 1:  uniform (0,1)
//>          = 2:  uniform (-1,1)
//>          = 3:  normal (0,1)
//> \endverbatim
//>
//> \param[in,out] ISEED
//> \verbatim
//>          ISEED is INTEGER array, dimension (4)
//>          On entry, the seed of the random number generator; the array
//>          elements must be between 0 and 4095, and ISEED(4) must be
//>          odd.
//>          On exit, the seed is updated.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of random numbers to be generated.
//> \endverbatim
//>
//> \param[out] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension (N)
//>          The generated random numbers.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  This routine calls the auxiliary routine DLARUV to generate random
//>  real numbers from a uniform (0,1) distribution, in batches of up to
//>  128 using vectorisable code. The Box-Muller method is used to
//>  transform numbers from a uniform to a normal distribution.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlarnv_(int *idist, int *iseed, int *n, double *x)
{
    // System generated locals
    int i__1, i__2, i__3;

    // Builtin functions
    double log(double), sqrt(double), cos(double);

    // Local variables
    int i__;
    double u[128];
    int il, iv, il2;
    extern /* Subroutine */ int dlaruv_(int *, int *, double *);

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --x;
    --iseed;

    // Function Body
    i__1 = *n;
    for (iv = 1; iv <= i__1; iv += 64) {
	// Computing MIN
	i__2 = 64, i__3 = *n - iv + 1;
	il = min(i__2,i__3);
	if (*idist == 3) {
	    il2 = il << 1;
	} else {
	    il2 = il;
	}
	//
	//       Call DLARUV to generate IL2 numbers from a uniform (0,1)
	//       distribution (IL2 <= LV)
	//
	dlaruv_(&iseed[1], &il2, u);
	if (*idist == 1) {
	    //
	    //          Copy generated numbers
	    //
	    i__2 = il;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		x[iv + i__ - 1] = u[i__ - 1];
// L10:
	    }
	} else if (*idist == 2) {
	    //
	    //          Convert generated numbers to uniform (-1,1) distribution
	    //
	    i__2 = il;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		x[iv + i__ - 1] = u[i__ - 1] * 2. - 1.;
// L20:
	    }
	} else if (*idist == 3) {
	    //
	    //          Convert generated numbers to normal (0,1) distribution
	    //
	    i__2 = il;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		x[iv + i__ - 1] = sqrt(log(u[(i__ << 1) - 2]) * -2.) * cos(u[(
			i__ << 1) - 1] * 6.2831853071795864769252867663);
// L30:
	    }
	}
// L40:
    }
    return 0;
    //
    //    End of DLARNV
    //
} // dlarnv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRA computes the splitting points with the specified threshold.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRA + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarra.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarra.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarra.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRA( N, D, E, E2, SPLTOL, TNRM,
//                          NSPLIT, ISPLIT, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, N, NSPLIT
//      DOUBLE PRECISION    SPLTOL, TNRM
//      ..
//      .. Array Arguments ..
//      INTEGER            ISPLIT( * )
//      DOUBLE PRECISION   D( * ), E( * ), E2( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Compute the splitting points with threshold SPLTOL.
//> DLARRA sets any "small" off-diagonal elements to zero.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix. N > 0.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the N diagonal elements of the tridiagonal
//>          matrix T.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N)
//>          On entry, the first (N-1) entries contain the subdiagonal
//>          elements of the tridiagonal matrix T; E(N) need not be set.
//>          On exit, the entries E( ISPLIT( I ) ), 1 <= I <= NSPLIT,
//>          are set to zero, the other entries of E are untouched.
//> \endverbatim
//>
//> \param[in,out] E2
//> \verbatim
//>          E2 is DOUBLE PRECISION array, dimension (N)
//>          On entry, the first (N-1) entries contain the SQUARES of the
//>          subdiagonal elements of the tridiagonal matrix T;
//>          E2(N) need not be set.
//>          On exit, the entries E2( ISPLIT( I ) ),
//>          1 <= I <= NSPLIT, have been set to zero
//> \endverbatim
//>
//> \param[in] SPLTOL
//> \verbatim
//>          SPLTOL is DOUBLE PRECISION
//>          The threshold for splitting. Two criteria can be used:
//>          SPLTOL<0 : criterion based on absolute off-diagonal value
//>          SPLTOL>0 : criterion that preserves relative accuracy
//> \endverbatim
//>
//> \param[in] TNRM
//> \verbatim
//>          TNRM is DOUBLE PRECISION
//>          The norm of the matrix.
//> \endverbatim
//>
//> \param[out] NSPLIT
//> \verbatim
//>          NSPLIT is INTEGER
//>          The number of blocks T splits into. 1 <= NSPLIT <= N.
//> \endverbatim
//>
//> \param[out] ISPLIT
//> \verbatim
//>          ISPLIT is INTEGER array, dimension (N)
//>          The splitting points, at which T breaks up into blocks.
//>          The first block consists of rows/columns 1 to ISPLIT(1),
//>          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2),
//>          etc., and the NSPLIT-th consists of rows/columns
//>          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
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
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlarra_(int *n, double *d__, double *e, double *e2, 
	double *spltol, double *tnrm, int *nsplit, int *isplit, int *info)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int i__;
    double tmp1, eabs;

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
    // Parameter adjustments
    --isplit;
    --e2;
    --e;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    //
    //    Compute splitting points
    *nsplit = 1;
    if (*spltol < 0.) {
	//       Criterion based on absolute off-diagonal value
	tmp1 = abs(*spltol) * *tnrm;
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    eabs = (d__1 = e[i__], abs(d__1));
	    if (eabs <= tmp1) {
		e[i__] = 0.;
		e2[i__] = 0.;
		isplit[*nsplit] = i__;
		++(*nsplit);
	    }
// L9:
	}
    } else {
	//       Criterion that guarantees relative accuracy
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    eabs = (d__1 = e[i__], abs(d__1));
	    if (eabs <= *spltol * sqrt((d__1 = d__[i__], abs(d__1))) * sqrt((
		    d__2 = d__[i__ + 1], abs(d__2)))) {
		e[i__] = 0.;
		e2[i__] = 0.;
		isplit[*nsplit] = i__;
		++(*nsplit);
	    }
// L10:
	}
    }
    isplit[*nsplit] = *n;
    return 0;
    //
    //    End of DLARRA
    //
} // dlarra_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRB provides limited bisection to locate eigenvalues for more accuracy.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRB + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrb.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrb.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrb.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRB( N, D, LLD, IFIRST, ILAST, RTOL1,
//                         RTOL2, OFFSET, W, WGAP, WERR, WORK, IWORK,
//                         PIVMIN, SPDIAM, TWIST, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IFIRST, ILAST, INFO, N, OFFSET, TWIST
//      DOUBLE PRECISION   PIVMIN, RTOL1, RTOL2, SPDIAM
//      ..
//      .. Array Arguments ..
//      INTEGER            IWORK( * )
//      DOUBLE PRECISION   D( * ), LLD( * ), W( * ),
//     $                   WERR( * ), WGAP( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Given the relatively robust representation(RRR) L D L^T, DLARRB
//> does "limited" bisection to refine the eigenvalues of L D L^T,
//> W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial
//> guesses for these eigenvalues are input in W, the corresponding estimate
//> of the error in these guesses and their gaps are input in WERR
//> and WGAP, respectively. During bisection, intervals
//> [left, right] are maintained by storing their mid-points and
//> semi-widths in the arrays W and WERR respectively.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The N diagonal elements of the diagonal matrix D.
//> \endverbatim
//>
//> \param[in] LLD
//> \verbatim
//>          LLD is DOUBLE PRECISION array, dimension (N-1)
//>          The (N-1) elements L(i)*L(i)*D(i).
//> \endverbatim
//>
//> \param[in] IFIRST
//> \verbatim
//>          IFIRST is INTEGER
//>          The index of the first eigenvalue to be computed.
//> \endverbatim
//>
//> \param[in] ILAST
//> \verbatim
//>          ILAST is INTEGER
//>          The index of the last eigenvalue to be computed.
//> \endverbatim
//>
//> \param[in] RTOL1
//> \verbatim
//>          RTOL1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] RTOL2
//> \verbatim
//>          RTOL2 is DOUBLE PRECISION
//>          Tolerance for the convergence of the bisection intervals.
//>          An interval [LEFT,RIGHT] has converged if
//>          RIGHT-LEFT < MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) )
//>          where GAP is the (estimated) distance to the nearest
//>          eigenvalue.
//> \endverbatim
//>
//> \param[in] OFFSET
//> \verbatim
//>          OFFSET is INTEGER
//>          Offset for the arrays W, WGAP and WERR, i.e., the IFIRST-OFFSET
//>          through ILAST-OFFSET elements of these arrays are to be used.
//> \endverbatim
//>
//> \param[in,out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          On input, W( IFIRST-OFFSET ) through W( ILAST-OFFSET ) are
//>          estimates of the eigenvalues of L D L^T indexed IFIRST through
//>          ILAST.
//>          On output, these estimates are refined.
//> \endverbatim
//>
//> \param[in,out] WGAP
//> \verbatim
//>          WGAP is DOUBLE PRECISION array, dimension (N-1)
//>          On input, the (estimated) gaps between consecutive
//>          eigenvalues of L D L^T, i.e., WGAP(I-OFFSET) is the gap between
//>          eigenvalues I and I+1. Note that if IFIRST = ILAST
//>          then WGAP(IFIRST-OFFSET) must be set to ZERO.
//>          On output, these gaps are refined.
//> \endverbatim
//>
//> \param[in,out] WERR
//> \verbatim
//>          WERR is DOUBLE PRECISION array, dimension (N)
//>          On input, WERR( IFIRST-OFFSET ) through WERR( ILAST-OFFSET ) are
//>          the errors in the estimates of the corresponding elements in W.
//>          On output, these errors are refined.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (2*N)
//>          Workspace.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (2*N)
//>          Workspace.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot in the Sturm sequence.
//> \endverbatim
//>
//> \param[in] SPDIAM
//> \verbatim
//>          SPDIAM is DOUBLE PRECISION
//>          The spectral diameter of the matrix.
//> \endverbatim
//>
//> \param[in] TWIST
//> \verbatim
//>          TWIST is INTEGER
//>          The twist index for the twisted factorization that is used
//>          for the negcount.
//>          TWIST = N: Compute negcount from L D L^T - LAMBDA I = L+ D+ L+^T
//>          TWIST = 1: Compute negcount from L D L^T - LAMBDA I = U- D- U-^T
//>          TWIST = R: Compute negcount from L D L^T - LAMBDA I = N(r) D(r) N(r)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          Error flag.
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
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlarrb_(int *n, double *d__, double *lld, int *ifirst, 
	int *ilast, double *rtol1, double *rtol2, int *offset, double *w, 
	double *wgap, double *werr, double *work, int *iwork, double *pivmin, 
	double *spdiam, int *twist, int *info)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Builtin functions
    double log(double);

    // Local variables
    int i__, k, r__, i1, ii, ip;
    double gap, mid, tmp, back, lgap, rgap, left;
    int iter, nint, prev, next;
    double cvrgd, right, width;
    extern int dlaneg_(int *, double *, double *, double *, double *, int *);
    int negcnt;
    double mnwdth;
    int olnint, maxitr;

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
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --iwork;
    --work;
    --werr;
    --wgap;
    --w;
    --lld;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    maxitr = (int) ((log(*spdiam + *pivmin) - log(*pivmin)) / log(2.)) + 2;
    mnwdth = *pivmin * 2.;
    r__ = *twist;
    if (r__ < 1 || r__ > *n) {
	r__ = *n;
    }
    //
    //    Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
    //    The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
    //    Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 )
    //    for an unconverged interval is set to the index of the next unconverged
    //    interval, and is -1 or 0 for a converged interval. Thus a linked
    //    list of unconverged intervals is set up.
    //
    i1 = *ifirst;
    //    The number of unconverged intervals
    nint = 0;
    //    The last unconverged interval found
    prev = 0;
    rgap = wgap[i1 - *offset];
    i__1 = *ilast;
    for (i__ = i1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	left = w[ii] - werr[ii];
	right = w[ii] + werr[ii];
	lgap = rgap;
	rgap = wgap[ii];
	gap = min(lgap,rgap);
	//       Make sure that [LEFT,RIGHT] contains the desired eigenvalue
	//       Compute negcount from dstqds facto L+D+L+^T = L D L^T - LEFT
	//
	//       Do while( NEGCNT(LEFT).GT.I-1 )
	//
	back = werr[ii];
L20:
	negcnt = dlaneg_(n, &d__[1], &lld[1], &left, pivmin, &r__);
	if (negcnt > i__ - 1) {
	    left -= back;
	    back *= 2.;
	    goto L20;
	}
	//
	//       Do while( NEGCNT(RIGHT).LT.I )
	//       Compute negcount from dstqds facto L+D+L+^T = L D L^T - RIGHT
	//
	back = werr[ii];
L50:
	negcnt = dlaneg_(n, &d__[1], &lld[1], &right, pivmin, &r__);
	if (negcnt < i__) {
	    right += back;
	    back *= 2.;
	    goto L50;
	}
	width = (d__1 = left - right, abs(d__1)) * .5;
	// Computing MAX
	d__1 = abs(left), d__2 = abs(right);
	tmp = max(d__1,d__2);
	// Computing MAX
	d__1 = *rtol1 * gap, d__2 = *rtol2 * tmp;
	cvrgd = max(d__1,d__2);
	if (width <= cvrgd || width <= mnwdth) {
	    //          This interval has already converged and does not need refinement.
	    //          (Note that the gaps might change through refining the
	    //           eigenvalues, however, they can only get bigger.)
	    //          Remove it from the list.
	    iwork[k - 1] = -1;
	    //          Make sure that I1 always points to the first unconverged interval
	    if (i__ == i1 && i__ < *ilast) {
		i1 = i__ + 1;
	    }
	    if (prev >= i1 && i__ <= *ilast) {
		iwork[(prev << 1) - 1] = i__ + 1;
	    }
	} else {
	    //          unconverged interval found
	    prev = i__;
	    ++nint;
	    iwork[k - 1] = i__ + 1;
	    iwork[k] = negcnt;
	}
	work[k - 1] = left;
	work[k] = right;
// L75:
    }
    //
    //    Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
    //    and while (ITER.LT.MAXITR)
    //
    iter = 0;
L80:
    prev = i1 - 1;
    i__ = i1;
    olnint = nint;
    i__1 = olnint;
    for (ip = 1; ip <= i__1; ++ip) {
	k = i__ << 1;
	ii = i__ - *offset;
	rgap = wgap[ii];
	lgap = rgap;
	if (ii > 1) {
	    lgap = wgap[ii - 1];
	}
	gap = min(lgap,rgap);
	next = iwork[k - 1];
	left = work[k - 1];
	right = work[k];
	mid = (left + right) * .5;
	//       semiwidth of interval
	width = right - mid;
	// Computing MAX
	d__1 = abs(left), d__2 = abs(right);
	tmp = max(d__1,d__2);
	// Computing MAX
	d__1 = *rtol1 * gap, d__2 = *rtol2 * tmp;
	cvrgd = max(d__1,d__2);
	if (width <= cvrgd || width <= mnwdth || iter == maxitr) {
	    //          reduce number of unconverged intervals
	    --nint;
	    //          Mark interval as converged.
	    iwork[k - 1] = 0;
	    if (i1 == i__) {
		i1 = next;
	    } else {
		//             Prev holds the last unconverged interval previously examined
		if (prev >= i1) {
		    iwork[(prev << 1) - 1] = next;
		}
	    }
	    i__ = next;
	    goto L100;
	}
	prev = i__;
	//
	//       Perform one bisection step
	//
	negcnt = dlaneg_(n, &d__[1], &lld[1], &mid, pivmin, &r__);
	if (negcnt <= i__ - 1) {
	    work[k - 1] = mid;
	} else {
	    work[k] = mid;
	}
	i__ = next;
L100:
	;
    }
    ++iter;
    //    do another loop if there are still unconverged intervals
    //    However, in the last iteration, all intervals are accepted
    //    since this is the best we can do.
    if (nint > 0 && iter <= maxitr) {
	goto L80;
    }
    //
    //
    //    At this point, all the intervals have converged
    i__1 = *ilast;
    for (i__ = *ifirst; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	//       All intervals marked by '0' have been refined.
	if (iwork[k - 1] == 0) {
	    w[ii] = (work[k - 1] + work[k]) * .5;
	    werr[ii] = work[k] - w[ii];
	}
// L110:
    }
    i__1 = *ilast;
    for (i__ = *ifirst + 1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	// Computing MAX
	d__1 = 0., d__2 = w[ii] - werr[ii] - w[ii - 1] - werr[ii - 1];
	wgap[ii - 1] = max(d__1,d__2);
// L111:
    }
    return 0;
    //
    //    End of DLARRB
    //
} // dlarrb_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRC computes the number of eigenvalues of the symmetric tridiagonal matrix.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRC + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrc.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrc.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrc.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRC( JOBT, N, VL, VU, D, E, PIVMIN,
//                                  EIGCNT, LCNT, RCNT, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          JOBT
//      INTEGER            EIGCNT, INFO, LCNT, N, RCNT
//      DOUBLE PRECISION   PIVMIN, VL, VU
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), E( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Find the number of eigenvalues of the symmetric tridiagonal matrix T
//> that are in the interval (VL,VU] if JOBT = 'T', and of L D L^T
//> if JOBT = 'L'.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOBT
//> \verbatim
//>          JOBT is CHARACTER*1
//>          = 'T':  Compute Sturm count for matrix T.
//>          = 'L':  Compute Sturm count for matrix L D L^T.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix. N > 0.
//> \endverbatim
//>
//> \param[in] VL
//> \verbatim
//>          VL is DOUBLE PRECISION
//>          The lower bound for the eigenvalues.
//> \endverbatim
//>
//> \param[in] VU
//> \verbatim
//>          VU is DOUBLE PRECISION
//>          The upper bound for the eigenvalues.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          JOBT = 'T': The N diagonal elements of the tridiagonal matrix T.
//>          JOBT = 'L': The N diagonal elements of the diagonal matrix D.
//> \endverbatim
//>
//> \param[in] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N)
//>          JOBT = 'T': The N-1 offdiagonal elements of the matrix T.
//>          JOBT = 'L': The N-1 offdiagonal elements of the matrix L.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot in the Sturm sequence for T.
//> \endverbatim
//>
//> \param[out] EIGCNT
//> \verbatim
//>          EIGCNT is INTEGER
//>          The number of eigenvalues of the symmetric tridiagonal matrix T
//>          that are in the interval (VL,VU]
//> \endverbatim
//>
//> \param[out] LCNT
//> \verbatim
//>          LCNT is INTEGER
//> \endverbatim
//>
//> \param[out] RCNT
//> \verbatim
//>          RCNT is INTEGER
//>          The left and right negcounts of the interval.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
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
//> \par Contributors:
// ==================
//>
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlarrc_(char *jobt, int *n, double *vl, double *vu, 
	double *d__, double *e, double *pivmin, int *eigcnt, int *lcnt, int *
	rcnt, int *info)
{
    // System generated locals
    int i__1;
    double d__1;

    // Local variables
    int i__;
    double sl, su, tmp, tmp2;
    int matt;
    extern int lsame_(char *, char *);
    double lpivot, rpivot;

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
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //    ..
    //    .. External Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --e;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    *lcnt = 0;
    *rcnt = 0;
    *eigcnt = 0;
    matt = lsame_(jobt, "T");
    if (matt) {
	//       Sturm sequence count on T
	lpivot = d__[1] - *vl;
	rpivot = d__[1] - *vu;
	if (lpivot <= 0.) {
	    ++(*lcnt);
	}
	if (rpivot <= 0.) {
	    ++(*rcnt);
	}
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    // Computing 2nd power
	    d__1 = e[i__];
	    tmp = d__1 * d__1;
	    lpivot = d__[i__ + 1] - *vl - tmp / lpivot;
	    rpivot = d__[i__ + 1] - *vu - tmp / rpivot;
	    if (lpivot <= 0.) {
		++(*lcnt);
	    }
	    if (rpivot <= 0.) {
		++(*rcnt);
	    }
// L10:
	}
    } else {
	//       Sturm sequence count on L D L^T
	sl = -(*vl);
	su = -(*vu);
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    lpivot = d__[i__] + sl;
	    rpivot = d__[i__] + su;
	    if (lpivot <= 0.) {
		++(*lcnt);
	    }
	    if (rpivot <= 0.) {
		++(*rcnt);
	    }
	    tmp = e[i__] * d__[i__] * e[i__];
	    tmp2 = tmp / lpivot;
	    if (tmp2 == 0.) {
		sl = tmp - *vl;
	    } else {
		sl = sl * tmp2 - *vl;
	    }
	    tmp2 = tmp / rpivot;
	    if (tmp2 == 0.) {
		su = tmp - *vu;
	    } else {
		su = su * tmp2 - *vu;
	    }
// L20:
	}
	lpivot = d__[*n] + sl;
	rpivot = d__[*n] + su;
	if (lpivot <= 0.) {
	    ++(*lcnt);
	}
	if (rpivot <= 0.) {
	    ++(*rcnt);
	}
    }
    *eigcnt = *rcnt - *lcnt;
    return 0;
    //
    //    end of DLARRC
    //
} // dlarrc_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRD computes the eigenvalues of a symmetric tridiagonal matrix to suitable accuracy.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrd.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrd.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrd.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRD( RANGE, ORDER, N, VL, VU, IL, IU, GERS,
//                          RELTOL, D, E, E2, PIVMIN, NSPLIT, ISPLIT,
//                          M, W, WERR, WL, WU, IBLOCK, INDEXW,
//                          WORK, IWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          ORDER, RANGE
//      INTEGER            IL, INFO, IU, M, N, NSPLIT
//      DOUBLE PRECISION    PIVMIN, RELTOL, VL, VU, WL, WU
//      ..
//      .. Array Arguments ..
//      INTEGER            IBLOCK( * ), INDEXW( * ),
//     $                   ISPLIT( * ), IWORK( * )
//      DOUBLE PRECISION   D( * ), E( * ), E2( * ),
//     $                   GERS( * ), W( * ), WERR( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLARRD computes the eigenvalues of a symmetric tridiagonal
//> matrix T to suitable accuracy. This is an auxiliary code to be
//> called from DSTEMR.
//> The user may ask for all eigenvalues, all eigenvalues
//> in the half-open interval (VL, VU], or the IL-th through IU-th
//> eigenvalues.
//>
//> To avoid overflow, the matrix must be scaled so that its
//> largest element is no greater than overflow**(1/2) * underflow**(1/4) in absolute value, and for greatest
//> accuracy, it should not be much smaller than that.
//>
//> See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
//> Matrix", Report CS41, Computer Science Dept., Stanford
//> University, July 21, 1966.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] RANGE
//> \verbatim
//>          RANGE is CHARACTER*1
//>          = 'A': ("All")   all eigenvalues will be found.
//>          = 'V': ("Value") all eigenvalues in the half-open interval
//>                           (VL, VU] will be found.
//>          = 'I': ("Index") the IL-th through IU-th eigenvalues (of the
//>                           entire matrix) will be found.
//> \endverbatim
//>
//> \param[in] ORDER
//> \verbatim
//>          ORDER is CHARACTER*1
//>          = 'B': ("By Block") the eigenvalues will be grouped by
//>                              split-off block (see IBLOCK, ISPLIT) and
//>                              ordered from smallest to largest within
//>                              the block.
//>          = 'E': ("Entire matrix")
//>                              the eigenvalues for the entire matrix
//>                              will be ordered from smallest to
//>                              largest.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the tridiagonal matrix T.  N >= 0.
//> \endverbatim
//>
//> \param[in] VL
//> \verbatim
//>          VL is DOUBLE PRECISION
//>          If RANGE='V', the lower bound of the interval to
//>          be searched for eigenvalues.  Eigenvalues less than or equal
//>          to VL, or greater than VU, will not be returned.  VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] VU
//> \verbatim
//>          VU is DOUBLE PRECISION
//>          If RANGE='V', the upper bound of the interval to
//>          be searched for eigenvalues.  Eigenvalues less than or equal
//>          to VL, or greater than VU, will not be returned.  VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] IL
//> \verbatim
//>          IL is INTEGER
//>          If RANGE='I', the index of the
//>          smallest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[in] IU
//> \verbatim
//>          IU is INTEGER
//>          If RANGE='I', the index of the
//>          largest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[in] GERS
//> \verbatim
//>          GERS is DOUBLE PRECISION array, dimension (2*N)
//>          The N Gerschgorin intervals (the i-th Gerschgorin interval
//>          is (GERS(2*i-1), GERS(2*i)).
//> \endverbatim
//>
//> \param[in] RELTOL
//> \verbatim
//>          RELTOL is DOUBLE PRECISION
//>          The minimum relative width of an interval.  When an interval
//>          is narrower than RELTOL times the larger (in
//>          magnitude) endpoint, then it is considered to be
//>          sufficiently small, i.e., converged.  Note: this should
//>          always be at least radix*machine epsilon.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The n diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          The (n-1) off-diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] E2
//> \verbatim
//>          E2 is DOUBLE PRECISION array, dimension (N-1)
//>          The (n-1) squared off-diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot allowed in the Sturm sequence for T.
//> \endverbatim
//>
//> \param[in] NSPLIT
//> \verbatim
//>          NSPLIT is INTEGER
//>          The number of diagonal blocks in the matrix T.
//>          1 <= NSPLIT <= N.
//> \endverbatim
//>
//> \param[in] ISPLIT
//> \verbatim
//>          ISPLIT is INTEGER array, dimension (N)
//>          The splitting points, at which T breaks up into submatrices.
//>          The first submatrix consists of rows/columns 1 to ISPLIT(1),
//>          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2),
//>          etc., and the NSPLIT-th consists of rows/columns
//>          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N.
//>          (Only the first NSPLIT elements will actually be used, but
//>          since the user cannot know a priori what value NSPLIT will
//>          have, N words must be reserved for ISPLIT.)
//> \endverbatim
//>
//> \param[out] M
//> \verbatim
//>          M is INTEGER
//>          The actual number of eigenvalues found. 0 <= M <= N.
//>          (See also the description of INFO=2,3.)
//> \endverbatim
//>
//> \param[out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          On exit, the first M elements of W will contain the
//>          eigenvalue approximations. DLARRD computes an interval
//>          I_j = (a_j, b_j] that includes eigenvalue j. The eigenvalue
//>          approximation is given as the interval midpoint
//>          W(j)= ( a_j + b_j)/2. The corresponding error is bounded by
//>          WERR(j) = abs( a_j - b_j)/2
//> \endverbatim
//>
//> \param[out] WERR
//> \verbatim
//>          WERR is DOUBLE PRECISION array, dimension (N)
//>          The error bound on the corresponding eigenvalue approximation
//>          in W.
//> \endverbatim
//>
//> \param[out] WL
//> \verbatim
//>          WL is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] WU
//> \verbatim
//>          WU is DOUBLE PRECISION
//>          The interval (WL, WU] contains all the wanted eigenvalues.
//>          If RANGE='V', then WL=VL and WU=VU.
//>          If RANGE='A', then WL and WU are the global Gerschgorin bounds
//>                        on the spectrum.
//>          If RANGE='I', then WL and WU are computed by DLAEBZ from the
//>                        index range specified.
//> \endverbatim
//>
//> \param[out] IBLOCK
//> \verbatim
//>          IBLOCK is INTEGER array, dimension (N)
//>          At each row/column j where E(j) is zero or small, the
//>          matrix T is considered to split into a block diagonal
//>          matrix.  On exit, if INFO = 0, IBLOCK(i) specifies to which
//>          block (from 1 to the number of blocks) the eigenvalue W(i)
//>          belongs.  (DLARRD may use the remaining N-M elements as
//>          workspace.)
//> \endverbatim
//>
//> \param[out] INDEXW
//> \verbatim
//>          INDEXW is INTEGER array, dimension (N)
//>          The indices of the eigenvalues within each block (submatrix);
//>          for example, INDEXW(i)= j and IBLOCK(i)=k imply that the
//>          i-th eigenvalue W(i) is the j-th eigenvalue in block k.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (4*N)
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (3*N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  some or all of the eigenvalues failed to converge or
//>                were not computed:
//>                =1 or 3: Bisection failed to converge for some
//>                        eigenvalues; these eigenvalues are flagged by a
//>                        negative block number.  The effect is that the
//>                        eigenvalues may not be as accurate as the
//>                        absolute and relative tolerances.  This is
//>                        generally caused by unexpectedly inaccurate
//>                        arithmetic.
//>                =2 or 3: RANGE='I' only: Not all of the eigenvalues
//>                        IL:IU were found.
//>                        Effect: M < IU+1-IL
//>                        Cause:  non-monotonic arithmetic, causing the
//>                                Sturm sequence to be non-monotonic.
//>                        Cure:   recalculate, using RANGE='A', and pick
//>                                out eigenvalues IL:IU.  In some cases,
//>                                increasing the PARAMETER "FUDGE" may
//>                                make things work.
//>                = 4:    RANGE='I', and the Gershgorin interval
//>                        initially used was too small.  No eigenvalues
//>                        were computed.
//>                        Probable cause: your machine has sloppy
//>                                        floating-point arithmetic.
//>                        Cure: Increase the PARAMETER "FUDGE",
//>                              recompile, and try again.
//> \endverbatim
//
//> \par Internal Parameters:
// =========================
//>
//> \verbatim
//>  FUDGE   DOUBLE PRECISION, default = 2
//>          A "fudge factor" to widen the Gershgorin intervals.  Ideally,
//>          a value of 1 should work, but on machines with sloppy
//>          arithmetic, this needs to be larger.  The default for
//>          publicly released versions should be large enough to handle
//>          the worst machine around.  Note that this has no effect
//>          on accuracy of the solution.
//> \endverbatim
//>
//> \par Contributors:
// ==================
//>
//>     W. Kahan, University of California, Berkeley, USA \n
//>     Beresford Parlett, University of California, Berkeley, USA \n
//>     Jim Demmel, University of California, Berkeley, USA \n
//>     Inderjit Dhillon, University of Texas, Austin, USA \n
//>     Osni Marques, LBNL/NERSC, USA \n
//>     Christof Voemel, University of California, Berkeley, USA \n
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
/* Subroutine */ int dlarrd_(char *range, char *order, int *n, double *vl, 
	double *vu, int *il, int *iu, double *gers, double *reltol, double *
	d__, double *e, double *e2, double *pivmin, int *nsplit, int *isplit, 
	int *m, double *w, double *werr, double *wl, double *wu, int *iblock, 
	int *indexw, double *work, int *iwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;
    int c__0 = 0;

    // System generated locals
    int i__1, i__2, i__3;
    double d__1, d__2;

    // Builtin functions
    double log(double);

    // Local variables
    int i__, j, ib, ie, je, nb;
    double gl;
    int im, in;
    double gu;
    int iw, jee;
    double eps;
    int nwl;
    double wlu, wul;
    int nwu;
    double tmp1, tmp2;
    int iend, jblk, ioff, iout, itmp1, itmp2, jdisc;
    extern int lsame_(char *, char *);
    int iinfo;
    double atoli;
    int iwoff, itmax;
    double wkill, rtoli, uflow, tnorm;
    extern double dlamch_(char *);
    int ibegin;
    extern /* Subroutine */ int dlaebz_(int *, int *, int *, int *, int *, 
	    int *, double *, double *, double *, double *, double *, double *,
	     int *, double *, double *, int *, int *, double *, int *, int *);
    int irange, idiscl, idumma[1];
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int idiscu;
    int ncnvrg, toofew;

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
    --iwork;
    --work;
    --indexw;
    --iblock;
    --werr;
    --w;
    --isplit;
    --e2;
    --e;
    --d__;
    --gers;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    //
    //    Decode RANGE
    //
    if (lsame_(range, "A")) {
	irange = 1;
    } else if (lsame_(range, "V")) {
	irange = 2;
    } else if (lsame_(range, "I")) {
	irange = 3;
    } else {
	irange = 0;
    }
    //
    //    Check for Errors
    //
    if (irange <= 0) {
	*info = -1;
    } else if (! (lsame_(order, "B") || lsame_(order, "E"))) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (irange == 2) {
	if (*vl >= *vu) {
	    *info = -5;
	}
    } else if (irange == 3 && (*il < 1 || *il > max(1,*n))) {
	*info = -6;
    } else if (irange == 3 && (*iu < min(*n,*il) || *iu > *n)) {
	*info = -7;
    }
    if (*info != 0) {
	return 0;
    }
    //    Initialize error flags
    *info = 0;
    ncnvrg = FALSE_;
    toofew = FALSE_;
    //    Quick return if possible
    *m = 0;
    if (*n == 0) {
	return 0;
    }
    //    Simplification:
    if (irange == 3 && *il == 1 && *iu == *n) {
	irange = 1;
    }
    //    Get machine constants
    eps = dlamch_("P");
    uflow = dlamch_("U");
    //    Special Case when N=1
    //    Treat case of 1x1 matrix for quick return
    if (*n == 1) {
	if (irange == 1 || irange == 2 && d__[1] > *vl && d__[1] <= *vu || 
		irange == 3 && *il == 1 && *iu == 1) {
	    *m = 1;
	    w[1] = d__[1];
	    //          The computation error of the eigenvalue is zero
	    werr[1] = 0.;
	    iblock[1] = 1;
	    indexw[1] = 1;
	}
	return 0;
    }
    //    NB is the minimum vector length for vector bisection, or 0
    //    if only scalar is to be done.
    nb = ilaenv_(&c__1, "DSTEBZ", " ", n, &c_n1, &c_n1, &c_n1);
    if (nb <= 1) {
	nb = 0;
    }
    //    Find global spectral radius
    gl = d__[1];
    gu = d__[1];
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	// Computing MIN
	d__1 = gl, d__2 = gers[(i__ << 1) - 1];
	gl = min(d__1,d__2);
	// Computing MAX
	d__1 = gu, d__2 = gers[i__ * 2];
	gu = max(d__1,d__2);
// L5:
    }
    //    Compute global Gerschgorin bounds and spectral diameter
    // Computing MAX
    d__1 = abs(gl), d__2 = abs(gu);
    tnorm = max(d__1,d__2);
    gl = gl - tnorm * 2. * eps * *n - *pivmin * 4.;
    gu = gu + tnorm * 2. * eps * *n + *pivmin * 4.;
    //    [JAN/28/2009] remove the line below since SPDIAM variable not use
    //    SPDIAM = GU - GL
    //    Input arguments for DLAEBZ:
    //    The relative tolerance.  An interval (a,b] lies within
    //    "relative tolerance" if  b-a < RELTOL*max(|a|,|b|),
    rtoli = *reltol;
    //    Set the absolute tolerance for interval convergence to zero to force
    //    interval convergence based on relative size of the interval.
    //    This is dangerous because intervals might not converge when RELTOL is
    //    small. But at least a very small number should be selected so that for
    //    strongly graded matrices, the code can get relatively accurate
    //    eigenvalues.
    atoli = uflow * 4. + *pivmin * 4.;
    if (irange == 3) {
	//       RANGE='I': Compute an interval containing eigenvalues
	//       IL through IU. The initial interval [GL,GU] from the global
	//       Gerschgorin bounds GL and GU is refined by DLAEBZ.
	itmax = (int) ((log(tnorm + *pivmin) - log(*pivmin)) / log(2.)) + 2;
	work[*n + 1] = gl;
	work[*n + 2] = gl;
	work[*n + 3] = gu;
	work[*n + 4] = gu;
	work[*n + 5] = gl;
	work[*n + 6] = gu;
	iwork[1] = -1;
	iwork[2] = -1;
	iwork[3] = *n + 1;
	iwork[4] = *n + 1;
	iwork[5] = *il - 1;
	iwork[6] = *iu;
	dlaebz_(&c__3, &itmax, n, &c__2, &c__2, &nb, &atoli, &rtoli, pivmin, &
		d__[1], &e[1], &e2[1], &iwork[5], &work[*n + 1], &work[*n + 5]
		, &iout, &iwork[1], &w[1], &iblock[1], &iinfo);
	if (iinfo != 0) {
	    *info = iinfo;
	    return 0;
	}
	//       On exit, output intervals may not be ordered by ascending negcount
	if (iwork[6] == *iu) {
	    *wl = work[*n + 1];
	    wlu = work[*n + 3];
	    nwl = iwork[1];
	    *wu = work[*n + 4];
	    wul = work[*n + 2];
	    nwu = iwork[4];
	} else {
	    *wl = work[*n + 2];
	    wlu = work[*n + 4];
	    nwl = iwork[2];
	    *wu = work[*n + 3];
	    wul = work[*n + 1];
	    nwu = iwork[3];
	}
	//       On exit, the interval [WL, WLU] contains a value with negcount NWL,
	//       and [WUL, WU] contains a value with negcount NWU.
	if (nwl < 0 || nwl >= *n || nwu < 1 || nwu > *n) {
	    *info = 4;
	    return 0;
	}
    } else if (irange == 2) {
	*wl = *vl;
	*wu = *vu;
    } else if (irange == 1) {
	*wl = gl;
	*wu = gu;
    }
    //    Find Eigenvalues -- Loop Over blocks and recompute NWL and NWU.
    //    NWL accumulates the number of eigenvalues .le. WL,
    //    NWU accumulates the number of eigenvalues .le. WU
    *m = 0;
    iend = 0;
    *info = 0;
    nwl = 0;
    nwu = 0;
    i__1 = *nsplit;
    for (jblk = 1; jblk <= i__1; ++jblk) {
	ioff = iend;
	ibegin = ioff + 1;
	iend = isplit[jblk];
	in = iend - ioff;
	if (in == 1) {
	    //          1x1 block
	    if (*wl >= d__[ibegin] - *pivmin) {
		++nwl;
	    }
	    if (*wu >= d__[ibegin] - *pivmin) {
		++nwu;
	    }
	    if (irange == 1 || *wl < d__[ibegin] - *pivmin && *wu >= d__[
		    ibegin] - *pivmin) {
		++(*m);
		w[*m] = d__[ibegin];
		werr[*m] = 0.;
		//             The gap for a single block doesn't matter for the later
		//             algorithm and is assigned an arbitrary large value
		iblock[*m] = jblk;
		indexw[*m] = 1;
	    }
	    //       Disabled 2x2 case because of a failure on the following matrix
	    //       RANGE = 'I', IL = IU = 4
	    //         Original Tridiagonal, d = [
	    //          -0.150102010615740E+00
	    //          -0.849897989384260E+00
	    //          -0.128208148052635E-15
	    //           0.128257718286320E-15
	    //         ];
	    //         e = [
	    //          -0.357171383266986E+00
	    //          -0.180411241501588E-15
	    //          -0.175152352710251E-15
	    //         ];
	    //
	    //        ELSE IF( IN.EQ.2 ) THEN
	    //*           2x2 block
	    //           DISC = SQRT( (HALF*(D(IBEGIN)-D(IEND)))**2 + E(IBEGIN)**2 )
	    //           TMP1 = HALF*(D(IBEGIN)+D(IEND))
	    //           L1 = TMP1 - DISC
	    //           IF( WL.GE. L1-PIVMIN )
	    //    $         NWL = NWL + 1
	    //           IF( WU.GE. L1-PIVMIN )
	    //    $         NWU = NWU + 1
	    //           IF( IRANGE.EQ.ALLRNG .OR. ( WL.LT.L1-PIVMIN .AND. WU.GE.
	    //    $          L1-PIVMIN ) ) THEN
	    //              M = M + 1
	    //              W( M ) = L1
	    //*              The uncertainty of eigenvalues of a 2x2 matrix is very small
	    //              WERR( M ) = EPS * ABS( W( M ) ) * TWO
	    //              IBLOCK( M ) = JBLK
	    //              INDEXW( M ) = 1
	    //           ENDIF
	    //           L2 = TMP1 + DISC
	    //           IF( WL.GE. L2-PIVMIN )
	    //    $         NWL = NWL + 1
	    //           IF( WU.GE. L2-PIVMIN )
	    //    $         NWU = NWU + 1
	    //           IF( IRANGE.EQ.ALLRNG .OR. ( WL.LT.L2-PIVMIN .AND. WU.GE.
	    //    $          L2-PIVMIN ) ) THEN
	    //              M = M + 1
	    //              W( M ) = L2
	    //*              The uncertainty of eigenvalues of a 2x2 matrix is very small
	    //              WERR( M ) = EPS * ABS( W( M ) ) * TWO
	    //              IBLOCK( M ) = JBLK
	    //              INDEXW( M ) = 2
	    //           ENDIF
	} else {
	    //          General Case - block of size IN >= 2
	    //          Compute local Gerschgorin interval and use it as the initial
	    //          interval for DLAEBZ
	    gu = d__[ibegin];
	    gl = d__[ibegin];
	    tmp1 = 0.;
	    i__2 = iend;
	    for (j = ibegin; j <= i__2; ++j) {
		// Computing MIN
		d__1 = gl, d__2 = gers[(j << 1) - 1];
		gl = min(d__1,d__2);
		// Computing MAX
		d__1 = gu, d__2 = gers[j * 2];
		gu = max(d__1,d__2);
// L40:
	    }
	    //          [JAN/28/2009]
	    //          change SPDIAM by TNORM in lines 2 and 3 thereafter
	    //          line 1: remove computation of SPDIAM (not useful anymore)
	    //          SPDIAM = GU - GL
	    //          GL = GL - FUDGE*SPDIAM*EPS*IN - FUDGE*PIVMIN
	    //          GU = GU + FUDGE*SPDIAM*EPS*IN + FUDGE*PIVMIN
	    gl = gl - tnorm * 2. * eps * in - *pivmin * 2.;
	    gu = gu + tnorm * 2. * eps * in + *pivmin * 2.;
	    if (irange > 1) {
		if (gu < *wl) {
		    //                the local block contains none of the wanted eigenvalues
		    nwl += in;
		    nwu += in;
		    goto L70;
		}
		//             refine search interval if possible, only range (WL,WU] matters
		gl = max(gl,*wl);
		gu = min(gu,*wu);
		if (gl >= gu) {
		    goto L70;
		}
	    }
	    //          Find negcount of initial interval boundaries GL and GU
	    work[*n + 1] = gl;
	    work[*n + in + 1] = gu;
	    dlaebz_(&c__1, &c__0, &in, &in, &c__1, &nb, &atoli, &rtoli, 
		    pivmin, &d__[ibegin], &e[ibegin], &e2[ibegin], idumma, &
		    work[*n + 1], &work[*n + (in << 1) + 1], &im, &iwork[1], &
		    w[*m + 1], &iblock[*m + 1], &iinfo);
	    if (iinfo != 0) {
		*info = iinfo;
		return 0;
	    }
	    nwl += iwork[1];
	    nwu += iwork[in + 1];
	    iwoff = *m - iwork[1];
	    //          Compute Eigenvalues
	    itmax = (int) ((log(gu - gl + *pivmin) - log(*pivmin)) / log(2.)) 
		    + 2;
	    dlaebz_(&c__2, &itmax, &in, &in, &c__1, &nb, &atoli, &rtoli, 
		    pivmin, &d__[ibegin], &e[ibegin], &e2[ibegin], idumma, &
		    work[*n + 1], &work[*n + (in << 1) + 1], &iout, &iwork[1],
		     &w[*m + 1], &iblock[*m + 1], &iinfo);
	    if (iinfo != 0) {
		*info = iinfo;
		return 0;
	    }
	    //
	    //          Copy eigenvalues into W and IBLOCK
	    //          Use -JBLK for block number for unconverged eigenvalues.
	    //          Loop over the number of output intervals from DLAEBZ
	    i__2 = iout;
	    for (j = 1; j <= i__2; ++j) {
		//             eigenvalue approximation is middle point of interval
		tmp1 = (work[j + *n] + work[j + in + *n]) * .5;
		//             semi length of error interval
		tmp2 = (d__1 = work[j + *n] - work[j + in + *n], abs(d__1)) * 
			.5;
		if (j > iout - iinfo) {
		    //                Flag non-convergence.
		    ncnvrg = TRUE_;
		    ib = -jblk;
		} else {
		    ib = jblk;
		}
		i__3 = iwork[j + in] + iwoff;
		for (je = iwork[j] + 1 + iwoff; je <= i__3; ++je) {
		    w[je] = tmp1;
		    werr[je] = tmp2;
		    indexw[je] = je - iwoff;
		    iblock[je] = ib;
// L50:
		}
// L60:
	    }
	    *m += im;
	}
L70:
	;
    }
    //    If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU
    //    If NWL+1 < IL or NWU > IU, discard extra eigenvalues.
    if (irange == 3) {
	idiscl = *il - 1 - nwl;
	idiscu = nwu - *iu;
	if (idiscl > 0) {
	    im = 0;
	    i__1 = *m;
	    for (je = 1; je <= i__1; ++je) {
		//             Remove some of the smallest eigenvalues from the left so that
		//             at the end IDISCL =0. Move all eigenvalues up to the left.
		if (w[je] <= wlu && idiscl > 0) {
		    --idiscl;
		} else {
		    ++im;
		    w[im] = w[je];
		    werr[im] = werr[je];
		    indexw[im] = indexw[je];
		    iblock[im] = iblock[je];
		}
// L80:
	    }
	    *m = im;
	}
	if (idiscu > 0) {
	    //          Remove some of the largest eigenvalues from the right so that
	    //          at the end IDISCU =0. Move all eigenvalues up to the left.
	    im = *m + 1;
	    for (je = *m; je >= 1; --je) {
		if (w[je] >= wul && idiscu > 0) {
		    --idiscu;
		} else {
		    --im;
		    w[im] = w[je];
		    werr[im] = werr[je];
		    indexw[im] = indexw[je];
		    iblock[im] = iblock[je];
		}
// L81:
	    }
	    jee = 0;
	    i__1 = *m;
	    for (je = im; je <= i__1; ++je) {
		++jee;
		w[jee] = w[je];
		werr[jee] = werr[je];
		indexw[jee] = indexw[je];
		iblock[jee] = iblock[je];
// L82:
	    }
	    *m = *m - im + 1;
	}
	if (idiscl > 0 || idiscu > 0) {
	    //          Code to deal with effects of bad arithmetic. (If N(w) is
	    //          monotone non-decreasing, this should never happen.)
	    //          Some low eigenvalues to be discarded are not in (WL,WLU],
	    //          or high eigenvalues to be discarded are not in (WUL,WU]
	    //          so just kill off the smallest IDISCL/largest IDISCU
	    //          eigenvalues, by marking the corresponding IBLOCK = 0
	    if (idiscl > 0) {
		wkill = *wu;
		i__1 = idiscl;
		for (jdisc = 1; jdisc <= i__1; ++jdisc) {
		    iw = 0;
		    i__2 = *m;
		    for (je = 1; je <= i__2; ++je) {
			if (iblock[je] != 0 && (w[je] < wkill || iw == 0)) {
			    iw = je;
			    wkill = w[je];
			}
// L90:
		    }
		    iblock[iw] = 0;
// L100:
		}
	    }
	    if (idiscu > 0) {
		wkill = *wl;
		i__1 = idiscu;
		for (jdisc = 1; jdisc <= i__1; ++jdisc) {
		    iw = 0;
		    i__2 = *m;
		    for (je = 1; je <= i__2; ++je) {
			if (iblock[je] != 0 && (w[je] >= wkill || iw == 0)) {
			    iw = je;
			    wkill = w[je];
			}
// L110:
		    }
		    iblock[iw] = 0;
// L120:
		}
	    }
	    //          Now erase all eigenvalues with IBLOCK set to zero
	    im = 0;
	    i__1 = *m;
	    for (je = 1; je <= i__1; ++je) {
		if (iblock[je] != 0) {
		    ++im;
		    w[im] = w[je];
		    werr[im] = werr[je];
		    indexw[im] = indexw[je];
		    iblock[im] = iblock[je];
		}
// L130:
	    }
	    *m = im;
	}
	if (idiscl < 0 || idiscu < 0) {
	    toofew = TRUE_;
	}
    }
    if (irange == 1 && *m != *n || irange == 3 && *m != *iu - *il + 1) {
	toofew = TRUE_;
    }
    //    If ORDER='B', do nothing the eigenvalues are already sorted by
    //       block.
    //    If ORDER='E', sort the eigenvalues from smallest to largest
    if (lsame_(order, "E") && *nsplit > 1) {
	i__1 = *m - 1;
	for (je = 1; je <= i__1; ++je) {
	    ie = 0;
	    tmp1 = w[je];
	    i__2 = *m;
	    for (j = je + 1; j <= i__2; ++j) {
		if (w[j] < tmp1) {
		    ie = j;
		    tmp1 = w[j];
		}
// L140:
	    }
	    if (ie != 0) {
		tmp2 = werr[ie];
		itmp1 = iblock[ie];
		itmp2 = indexw[ie];
		w[ie] = w[je];
		werr[ie] = werr[je];
		iblock[ie] = iblock[je];
		indexw[ie] = indexw[je];
		w[je] = tmp1;
		werr[je] = tmp2;
		iblock[je] = itmp1;
		indexw[je] = itmp2;
	    }
// L150:
	}
    }
    *info = 0;
    if (ncnvrg) {
	++(*info);
    }
    if (toofew) {
	*info += 2;
    }
    return 0;
    //
    //    End of DLARRD
    //
} // dlarrd_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRE given the tridiagonal matrix T, sets small off-diagonal elements to zero and for each unreduced block Ti, finds base representations and eigenvalues.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRE + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarre.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarre.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarre.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRE( RANGE, N, VL, VU, IL, IU, D, E, E2,
//                          RTOL1, RTOL2, SPLTOL, NSPLIT, ISPLIT, M,
//                          W, WERR, WGAP, IBLOCK, INDEXW, GERS, PIVMIN,
//                          WORK, IWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          RANGE
//      INTEGER            IL, INFO, IU, M, N, NSPLIT
//      DOUBLE PRECISION  PIVMIN, RTOL1, RTOL2, SPLTOL, VL, VU
//      ..
//      .. Array Arguments ..
//      INTEGER            IBLOCK( * ), ISPLIT( * ), IWORK( * ),
//     $                   INDEXW( * )
//      DOUBLE PRECISION   D( * ), E( * ), E2( * ), GERS( * ),
//     $                   W( * ),WERR( * ), WGAP( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> To find the desired eigenvalues of a given real symmetric
//> tridiagonal matrix T, DLARRE sets any "small" off-diagonal
//> elements to zero, and for each unreduced block T_i, it finds
//> (a) a suitable shift at one end of the block's spectrum,
//> (b) the base representation, T_i - sigma_i I = L_i D_i L_i^T, and
//> (c) eigenvalues of each L_i D_i L_i^T.
//> The representations and eigenvalues found are then used by
//> DSTEMR to compute the eigenvectors of T.
//> The accuracy varies depending on whether bisection is used to
//> find a few eigenvalues or the dqds algorithm (subroutine DLASQ2) to
//> conpute all and then discard any unwanted one.
//> As an added benefit, DLARRE also outputs the n
//> Gerschgorin intervals for the matrices L_i D_i L_i^T.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] RANGE
//> \verbatim
//>          RANGE is CHARACTER*1
//>          = 'A': ("All")   all eigenvalues will be found.
//>          = 'V': ("Value") all eigenvalues in the half-open interval
//>                           (VL, VU] will be found.
//>          = 'I': ("Index") the IL-th through IU-th eigenvalues (of the
//>                           entire matrix) will be found.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix. N > 0.
//> \endverbatim
//>
//> \param[in,out] VL
//> \verbatim
//>          VL is DOUBLE PRECISION
//>          If RANGE='V', the lower bound for the eigenvalues.
//>          Eigenvalues less than or equal to VL, or greater than VU,
//>          will not be returned.  VL < VU.
//>          If RANGE='I' or ='A', DLARRE computes bounds on the desired
//>          part of the spectrum.
//> \endverbatim
//>
//> \param[in,out] VU
//> \verbatim
//>          VU is DOUBLE PRECISION
//>          If RANGE='V', the upper bound for the eigenvalues.
//>          Eigenvalues less than or equal to VL, or greater than VU,
//>          will not be returned.  VL < VU.
//>          If RANGE='I' or ='A', DLARRE computes bounds on the desired
//>          part of the spectrum.
//> \endverbatim
//>
//> \param[in] IL
//> \verbatim
//>          IL is INTEGER
//>          If RANGE='I', the index of the
//>          smallest eigenvalue to be returned.
//>          1 <= IL <= IU <= N.
//> \endverbatim
//>
//> \param[in] IU
//> \verbatim
//>          IU is INTEGER
//>          If RANGE='I', the index of the
//>          largest eigenvalue to be returned.
//>          1 <= IL <= IU <= N.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the N diagonal elements of the tridiagonal
//>          matrix T.
//>          On exit, the N diagonal elements of the diagonal
//>          matrices D_i.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N)
//>          On entry, the first (N-1) entries contain the subdiagonal
//>          elements of the tridiagonal matrix T; E(N) need not be set.
//>          On exit, E contains the subdiagonal elements of the unit
//>          bidiagonal matrices L_i. The entries E( ISPLIT( I ) ),
//>          1 <= I <= NSPLIT, contain the base points sigma_i on output.
//> \endverbatim
//>
//> \param[in,out] E2
//> \verbatim
//>          E2 is DOUBLE PRECISION array, dimension (N)
//>          On entry, the first (N-1) entries contain the SQUARES of the
//>          subdiagonal elements of the tridiagonal matrix T;
//>          E2(N) need not be set.
//>          On exit, the entries E2( ISPLIT( I ) ),
//>          1 <= I <= NSPLIT, have been set to zero
//> \endverbatim
//>
//> \param[in] RTOL1
//> \verbatim
//>          RTOL1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] RTOL2
//> \verbatim
//>          RTOL2 is DOUBLE PRECISION
//>           Parameters for bisection.
//>           An interval [LEFT,RIGHT] has converged if
//>           RIGHT-LEFT < MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) )
//> \endverbatim
//>
//> \param[in] SPLTOL
//> \verbatim
//>          SPLTOL is DOUBLE PRECISION
//>          The threshold for splitting.
//> \endverbatim
//>
//> \param[out] NSPLIT
//> \verbatim
//>          NSPLIT is INTEGER
//>          The number of blocks T splits into. 1 <= NSPLIT <= N.
//> \endverbatim
//>
//> \param[out] ISPLIT
//> \verbatim
//>          ISPLIT is INTEGER array, dimension (N)
//>          The splitting points, at which T breaks up into blocks.
//>          The first block consists of rows/columns 1 to ISPLIT(1),
//>          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2),
//>          etc., and the NSPLIT-th consists of rows/columns
//>          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N.
//> \endverbatim
//>
//> \param[out] M
//> \verbatim
//>          M is INTEGER
//>          The total number of eigenvalues (of all L_i D_i L_i^T)
//>          found.
//> \endverbatim
//>
//> \param[out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          The first M elements contain the eigenvalues. The
//>          eigenvalues of each of the blocks, L_i D_i L_i^T, are
//>          sorted in ascending order ( DLARRE may use the
//>          remaining N-M elements as workspace).
//> \endverbatim
//>
//> \param[out] WERR
//> \verbatim
//>          WERR is DOUBLE PRECISION array, dimension (N)
//>          The error bound on the corresponding eigenvalue in W.
//> \endverbatim
//>
//> \param[out] WGAP
//> \verbatim
//>          WGAP is DOUBLE PRECISION array, dimension (N)
//>          The separation from the right neighbor eigenvalue in W.
//>          The gap is only with respect to the eigenvalues of the same block
//>          as each block has its own representation tree.
//>          Exception: at the right end of a block we store the left gap
//> \endverbatim
//>
//> \param[out] IBLOCK
//> \verbatim
//>          IBLOCK is INTEGER array, dimension (N)
//>          The indices of the blocks (submatrices) associated with the
//>          corresponding eigenvalues in W; IBLOCK(i)=1 if eigenvalue
//>          W(i) belongs to the first block from the top, =2 if W(i)
//>          belongs to the second block, etc.
//> \endverbatim
//>
//> \param[out] INDEXW
//> \verbatim
//>          INDEXW is INTEGER array, dimension (N)
//>          The indices of the eigenvalues within each block (submatrix);
//>          for example, INDEXW(i)= 10 and IBLOCK(i)=2 imply that the
//>          i-th eigenvalue W(i) is the 10-th eigenvalue in block 2
//> \endverbatim
//>
//> \param[out] GERS
//> \verbatim
//>          GERS is DOUBLE PRECISION array, dimension (2*N)
//>          The N Gerschgorin intervals (the i-th Gerschgorin interval
//>          is (GERS(2*i-1), GERS(2*i)).
//> \endverbatim
//>
//> \param[out] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot in the Sturm sequence for T.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (6*N)
//>          Workspace.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (5*N)
//>          Workspace.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          > 0:  A problem occurred in DLARRE.
//>          < 0:  One of the called subroutines signaled an internal problem.
//>                Needs inspection of the corresponding parameter IINFO
//>                for further information.
//>
//>          =-1:  Problem in DLARRD.
//>          = 2:  No base representation could be found in MAXTRY iterations.
//>                Increasing MAXTRY and recompilation might be a remedy.
//>          =-3:  Problem in DLARRB when computing the refined root
//>                representation for DLASQ2.
//>          =-4:  Problem in DLARRB when preforming bisection on the
//>                desired part of the spectrum.
//>          =-5:  Problem in DLASQ2.
//>          =-6:  Problem in DLASQ2.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  The base representations are required to suffer very little
//>  element growth and consequently define all their eigenvalues to
//>  high relative accuracy.
//> \endverbatim
//
//> \par Contributors:
// ==================
//>
//>     Beresford Parlett, University of California, Berkeley, USA \n
//>     Jim Demmel, University of California, Berkeley, USA \n
//>     Inderjit Dhillon, University of Texas, Austin, USA \n
//>     Osni Marques, LBNL/NERSC, USA \n
//>     Christof Voemel, University of California, Berkeley, USA \n
//>
// =====================================================================
/* Subroutine */ int dlarre_(char *range, int *n, double *vl, double *vu, int 
	*il, int *iu, double *d__, double *e, double *e2, double *rtol1, 
	double *rtol2, double *spltol, int *nsplit, int *isplit, int *m, 
	double *w, double *werr, double *wgap, int *iblock, int *indexw, 
	double *gers, double *pivmin, double *work, int *iwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c__2 = 2;

    // System generated locals
    int i__1, i__2;
    double d__1, d__2, d__3;

    // Builtin functions
    double sqrt(double), log(double);

    // Local variables
    int i__, j;
    double s1, s2;
    int mb;
    double gl;
    int in, mm;
    double gu;
    int cnt;
    double eps, tau, tmp, rtl;
    int cnt1, cnt2;
    double tmp1, eabs;
    int iend, jblk;
    double eold;
    int indl;
    double dmax__, emax;
    int wend, idum, indu;
    double rtol;
    int iseed[4];
    double avgap, sigma;
    extern int lsame_(char *, char *);
    int iinfo;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int norep;
    extern /* Subroutine */ int dlasq2_(int *, double *, int *);
    extern double dlamch_(char *);
    int ibegin;
    int forceb;
    int irange;
    double sgndef;
    extern /* Subroutine */ int dlarra_(int *, double *, double *, double *, 
	    double *, double *, int *, int *, int *), dlarrb_(int *, double *,
	     double *, int *, int *, double *, double *, int *, double *, 
	    double *, double *, double *, int *, double *, double *, int *, 
	    int *), dlarrc_(char *, int *, double *, double *, double *, 
	    double *, double *, int *, int *, int *, int *);
    int wbegin;
    extern /* Subroutine */ int dlarrd_(char *, char *, int *, double *, 
	    double *, int *, int *, double *, double *, double *, double *, 
	    double *, double *, int *, int *, int *, double *, double *, 
	    double *, double *, int *, int *, double *, int *, int *);
    double safmin, spdiam;
    extern /* Subroutine */ int dlarrk_(int *, int *, double *, double *, 
	    double *, double *, double *, double *, double *, double *, int *)
	    ;
    int usedqd;
    double clwdth, isleft;
    extern /* Subroutine */ int dlarnv_(int *, int *, int *, double *);
    double isrght, bsrtol, dpivot;

    //
    // -- LAPACK auxiliary routine (version 3.8.0) --
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
    //    .. External Functions ..
    //    ..
    //    .. External Subroutines ..
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --iwork;
    --work;
    --gers;
    --indexw;
    --iblock;
    --wgap;
    --werr;
    --w;
    --isplit;
    --e2;
    --e;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    //
    //    Decode RANGE
    //
    if (lsame_(range, "A")) {
	irange = 1;
    } else if (lsame_(range, "V")) {
	irange = 3;
    } else if (lsame_(range, "I")) {
	irange = 2;
    }
    *m = 0;
    //    Get machine constants
    safmin = dlamch_("S");
    eps = dlamch_("P");
    //    Set parameters
    rtl = sqrt(eps);
    bsrtol = sqrt(eps);
    //    Treat case of 1x1 matrix for quick return
    if (*n == 1) {
	if (irange == 1 || irange == 3 && d__[1] > *vl && d__[1] <= *vu || 
		irange == 2 && *il == 1 && *iu == 1) {
	    *m = 1;
	    w[1] = d__[1];
	    //          The computation error of the eigenvalue is zero
	    werr[1] = 0.;
	    wgap[1] = 0.;
	    iblock[1] = 1;
	    indexw[1] = 1;
	    gers[1] = d__[1];
	    gers[2] = d__[1];
	}
	//       store the shift for the initial RRR, which is zero in this case
	e[1] = 0.;
	return 0;
    }
    //    General case: tridiagonal matrix of order > 1
    //
    //    Init WERR, WGAP. Compute Gerschgorin intervals and spectral diameter.
    //    Compute maximum off-diagonal entry and pivmin.
    gl = d__[1];
    gu = d__[1];
    eold = 0.;
    emax = 0.;
    e[*n] = 0.;
    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	werr[i__] = 0.;
	wgap[i__] = 0.;
	eabs = (d__1 = e[i__], abs(d__1));
	if (eabs >= emax) {
	    emax = eabs;
	}
	tmp1 = eabs + eold;
	gers[(i__ << 1) - 1] = d__[i__] - tmp1;
	// Computing MIN
	d__1 = gl, d__2 = gers[(i__ << 1) - 1];
	gl = min(d__1,d__2);
	gers[i__ * 2] = d__[i__] + tmp1;
	// Computing MAX
	d__1 = gu, d__2 = gers[i__ * 2];
	gu = max(d__1,d__2);
	eold = eabs;
// L5:
    }
    //    The minimum pivot allowed in the Sturm sequence for T
    // Computing MAX
    // Computing 2nd power
    d__3 = emax;
    d__1 = 1., d__2 = d__3 * d__3;
    *pivmin = safmin * max(d__1,d__2);
    //    Compute spectral diameter. The Gerschgorin bounds give an
    //    estimate that is wrong by at most a factor of SQRT(2)
    spdiam = gu - gl;
    //    Compute splitting points
    dlarra_(n, &d__[1], &e[1], &e2[1], spltol, &spdiam, nsplit, &isplit[1], &
	    iinfo);
    //    Can force use of bisection instead of faster DQDS.
    //    Option left in the code for future multisection work.
    forceb = FALSE_;
    //    Initialize USEDQD, DQDS should be used for ALLRNG unless someone
    //    explicitly wants bisection.
    usedqd = irange == 1 && ! forceb;
    if (irange == 1 && ! forceb) {
	//       Set interval [VL,VU] that contains all eigenvalues
	*vl = gl;
	*vu = gu;
    } else {
	//       We call DLARRD to find crude approximations to the eigenvalues
	//       in the desired range. In case IRANGE = INDRNG, we also obtain the
	//       interval (VL,VU] that contains all the wanted eigenvalues.
	//       An interval [LEFT,RIGHT] has converged if
	//       RIGHT-LEFT.LT.RTOL*MAX(ABS(LEFT),ABS(RIGHT))
	//       DLARRD needs a WORK of size 4*N, IWORK of size 3*N
	dlarrd_(range, "B", n, vl, vu, il, iu, &gers[1], &bsrtol, &d__[1], &e[
		1], &e2[1], pivmin, nsplit, &isplit[1], &mm, &w[1], &werr[1], 
		vl, vu, &iblock[1], &indexw[1], &work[1], &iwork[1], &iinfo);
	if (iinfo != 0) {
	    *info = -1;
	    return 0;
	}
	//       Make sure that the entries M+1 to N in W, WERR, IBLOCK, INDEXW are 0
	i__1 = *n;
	for (i__ = mm + 1; i__ <= i__1; ++i__) {
	    w[i__] = 0.;
	    werr[i__] = 0.;
	    iblock[i__] = 0;
	    indexw[i__] = 0;
// L14:
	}
    }
    //**
    //    Loop over unreduced blocks
    ibegin = 1;
    wbegin = 1;
    i__1 = *nsplit;
    for (jblk = 1; jblk <= i__1; ++jblk) {
	iend = isplit[jblk];
	in = iend - ibegin + 1;
	//       1 X 1 block
	if (in == 1) {
	    if (irange == 1 || irange == 3 && d__[ibegin] > *vl && d__[ibegin]
		     <= *vu || irange == 2 && iblock[wbegin] == jblk) {
		++(*m);
		w[*m] = d__[ibegin];
		werr[*m] = 0.;
		//             The gap for a single block doesn't matter for the later
		//             algorithm and is assigned an arbitrary large value
		wgap[*m] = 0.;
		iblock[*m] = jblk;
		indexw[*m] = 1;
		++wbegin;
	    }
	    //          E( IEND ) holds the shift for the initial RRR
	    e[iend] = 0.;
	    ibegin = iend + 1;
	    goto L170;
	}
	//
	//       Blocks of size larger than 1x1
	//
	//       E( IEND ) will hold the shift for the initial RRR, for now set it =0
	e[iend] = 0.;
	//
	//       Find local outer bounds GL,GU for the block
	gl = d__[ibegin];
	gu = d__[ibegin];
	i__2 = iend;
	for (i__ = ibegin; i__ <= i__2; ++i__) {
	    // Computing MIN
	    d__1 = gers[(i__ << 1) - 1];
	    gl = min(d__1,gl);
	    // Computing MAX
	    d__1 = gers[i__ * 2];
	    gu = max(d__1,gu);
// L15:
	}
	spdiam = gu - gl;
	if (! (irange == 1 && ! forceb)) {
	    //          Count the number of eigenvalues in the current block.
	    mb = 0;
	    i__2 = mm;
	    for (i__ = wbegin; i__ <= i__2; ++i__) {
		if (iblock[i__] == jblk) {
		    ++mb;
		} else {
		    goto L21;
		}
// L20:
	    }
L21:
	    if (mb == 0) {
		//             No eigenvalue in the current block lies in the desired range
		//             E( IEND ) holds the shift for the initial RRR
		e[iend] = 0.;
		ibegin = iend + 1;
		goto L170;
	    } else {
		//             Decide whether dqds or bisection is more efficient
		usedqd = (double) mb > in * .5 && ! forceb;
		wend = wbegin + mb - 1;
		//             Calculate gaps for the current block
		//             In later stages, when representations for individual
		//             eigenvalues are different, we use SIGMA = E( IEND ).
		sigma = 0.;
		i__2 = wend - 1;
		for (i__ = wbegin; i__ <= i__2; ++i__) {
		    // Computing MAX
		    d__1 = 0., d__2 = w[i__ + 1] - werr[i__ + 1] - (w[i__] + 
			    werr[i__]);
		    wgap[i__] = max(d__1,d__2);
// L30:
		}
		// Computing MAX
		d__1 = 0., d__2 = *vu - sigma - (w[wend] + werr[wend]);
		wgap[wend] = max(d__1,d__2);
		//             Find local index of the first and last desired evalue.
		indl = indexw[wbegin];
		indu = indexw[wend];
	    }
	}
	if (irange == 1 && ! forceb || usedqd) {
	    //          Case of DQDS
	    //          Find approximations to the extremal eigenvalues of the block
	    dlarrk_(&in, &c__1, &gl, &gu, &d__[ibegin], &e2[ibegin], pivmin, &
		    rtl, &tmp, &tmp1, &iinfo);
	    if (iinfo != 0) {
		*info = -1;
		return 0;
	    }
	    // Computing MAX
	    d__2 = gl, d__3 = tmp - tmp1 - eps * 100. * (d__1 = tmp - tmp1, 
		    abs(d__1));
	    isleft = max(d__2,d__3);
	    dlarrk_(&in, &in, &gl, &gu, &d__[ibegin], &e2[ibegin], pivmin, &
		    rtl, &tmp, &tmp1, &iinfo);
	    if (iinfo != 0) {
		*info = -1;
		return 0;
	    }
	    // Computing MIN
	    d__2 = gu, d__3 = tmp + tmp1 + eps * 100. * (d__1 = tmp + tmp1, 
		    abs(d__1));
	    isrght = min(d__2,d__3);
	    //          Improve the estimate of the spectral diameter
	    spdiam = isrght - isleft;
	} else {
	    //          Case of bisection
	    //          Find approximations to the wanted extremal eigenvalues
	    // Computing MAX
	    d__2 = gl, d__3 = w[wbegin] - werr[wbegin] - eps * 100. * (d__1 = 
		    w[wbegin] - werr[wbegin], abs(d__1));
	    isleft = max(d__2,d__3);
	    // Computing MIN
	    d__2 = gu, d__3 = w[wend] + werr[wend] + eps * 100. * (d__1 = w[
		    wend] + werr[wend], abs(d__1));
	    isrght = min(d__2,d__3);
	}
	//       Decide whether the base representation for the current block
	//       L_JBLK D_JBLK L_JBLK^T = T_JBLK - sigma_JBLK I
	//       should be on the left or the right end of the current block.
	//       The strategy is to shift to the end which is "more populated"
	//       Furthermore, decide whether to use DQDS for the computation of
	//       the eigenvalue approximations at the end of DLARRE or bisection.
	//       dqds is chosen if all eigenvalues are desired or the number of
	//       eigenvalues to be computed is large compared to the blocksize.
	if (irange == 1 && ! forceb) {
	    //          If all the eigenvalues have to be computed, we use dqd
	    usedqd = TRUE_;
	    //          INDL is the local index of the first eigenvalue to compute
	    indl = 1;
	    indu = in;
	    //          MB =  number of eigenvalues to compute
	    mb = in;
	    wend = wbegin + mb - 1;
	    //          Define 1/4 and 3/4 points of the spectrum
	    s1 = isleft + spdiam * .25;
	    s2 = isrght - spdiam * .25;
	} else {
	    //          DLARRD has computed IBLOCK and INDEXW for each eigenvalue
	    //          approximation.
	    //          choose sigma
	    if (usedqd) {
		s1 = isleft + spdiam * .25;
		s2 = isrght - spdiam * .25;
	    } else {
		tmp = min(isrght,*vu) - max(isleft,*vl);
		s1 = max(isleft,*vl) + tmp * .25;
		s2 = min(isrght,*vu) - tmp * .25;
	    }
	}
	//       Compute the negcount at the 1/4 and 3/4 points
	if (mb > 1) {
	    dlarrc_("T", &in, &s1, &s2, &d__[ibegin], &e[ibegin], pivmin, &
		    cnt, &cnt1, &cnt2, &iinfo);
	}
	if (mb == 1) {
	    sigma = gl;
	    sgndef = 1.;
	} else if (cnt1 - indl >= indu - cnt2) {
	    if (irange == 1 && ! forceb) {
		sigma = max(isleft,gl);
	    } else if (usedqd) {
		//             use Gerschgorin bound as shift to get pos def matrix
		//             for dqds
		sigma = isleft;
	    } else {
		//             use approximation of the first desired eigenvalue of the
		//             block as shift
		sigma = max(isleft,*vl);
	    }
	    sgndef = 1.;
	} else {
	    if (irange == 1 && ! forceb) {
		sigma = min(isrght,gu);
	    } else if (usedqd) {
		//             use Gerschgorin bound as shift to get neg def matrix
		//             for dqds
		sigma = isrght;
	    } else {
		//             use approximation of the first desired eigenvalue of the
		//             block as shift
		sigma = min(isrght,*vu);
	    }
	    sgndef = -1.;
	}
	//       An initial SIGMA has been chosen that will be used for computing
	//       T - SIGMA I = L D L^T
	//       Define the increment TAU of the shift in case the initial shift
	//       needs to be refined to obtain a factorization with not too much
	//       element growth.
	if (usedqd) {
	    //          The initial SIGMA was to the outer end of the spectrum
	    //          the matrix is definite and we need not retreat.
	    tau = spdiam * eps * *n + *pivmin * 2.;
	    // Computing MAX
	    d__1 = tau, d__2 = eps * 2. * abs(sigma);
	    tau = max(d__1,d__2);
	} else {
	    if (mb > 1) {
		clwdth = w[wend] + werr[wend] - w[wbegin] - werr[wbegin];
		avgap = (d__1 = clwdth / (double) (wend - wbegin), abs(d__1));
		if (sgndef == 1.) {
		    // Computing MAX
		    d__1 = wgap[wbegin];
		    tau = max(d__1,avgap) * .5;
		    // Computing MAX
		    d__1 = tau, d__2 = werr[wbegin];
		    tau = max(d__1,d__2);
		} else {
		    // Computing MAX
		    d__1 = wgap[wend - 1];
		    tau = max(d__1,avgap) * .5;
		    // Computing MAX
		    d__1 = tau, d__2 = werr[wend];
		    tau = max(d__1,d__2);
		}
	    } else {
		tau = werr[wbegin];
	    }
	}
	for (idum = 1; idum <= 6; ++idum) {
	    //          Compute L D L^T factorization of tridiagonal matrix T - sigma I.
	    //          Store D in WORK(1:IN), L in WORK(IN+1:2*IN), and reciprocals of
	    //          pivots in WORK(2*IN+1:3*IN)
	    dpivot = d__[ibegin] - sigma;
	    work[1] = dpivot;
	    dmax__ = abs(work[1]);
	    j = ibegin;
	    i__2 = in - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[(in << 1) + i__] = 1. / work[i__];
		tmp = e[j] * work[(in << 1) + i__];
		work[in + i__] = tmp;
		dpivot = d__[j + 1] - sigma - tmp * e[j];
		work[i__ + 1] = dpivot;
		// Computing MAX
		d__1 = dmax__, d__2 = abs(dpivot);
		dmax__ = max(d__1,d__2);
		++j;
// L70:
	    }
	    //          check for element growth
	    if (dmax__ > spdiam * 64.) {
		norep = TRUE_;
	    } else {
		norep = FALSE_;
	    }
	    if (usedqd && ! norep) {
		//             Ensure the definiteness of the representation
		//             All entries of D (of L D L^T) must have the same sign
		i__2 = in;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    tmp = sgndef * work[i__];
		    if (tmp < 0.) {
			norep = TRUE_;
		    }
// L71:
		}
	    }
	    if (norep) {
		//             Note that in the case of IRANGE=ALLRNG, we use the Gerschgorin
		//             shift which makes the matrix definite. So we should end up
		//             here really only in the case of IRANGE = VALRNG or INDRNG.
		if (idum == 5) {
		    if (sgndef == 1.) {
			//                   The fudged Gerschgorin shift should succeed
			sigma = gl - spdiam * 2. * eps * *n - *pivmin * 4.;
		    } else {
			sigma = gu + spdiam * 2. * eps * *n + *pivmin * 4.;
		    }
		} else {
		    sigma -= sgndef * tau;
		    tau *= 2.;
		}
	    } else {
		//             an initial RRR is found
		goto L83;
	    }
// L80:
	}
	//       if the program reaches this point, no base representation could be
	//       found in MAXTRY iterations.
	*info = 2;
	return 0;
L83:
	//       At this point, we have found an initial base representation
	//       T - SIGMA I = L D L^T with not too much element growth.
	//       Store the shift.
	e[iend] = sigma;
	//       Store D and L.
	dcopy_(&in, &work[1], &c__1, &d__[ibegin], &c__1);
	i__2 = in - 1;
	dcopy_(&i__2, &work[in + 1], &c__1, &e[ibegin], &c__1);
	if (mb > 1) {
	    //
	    //          Perturb each entry of the base representation by a small
	    //          (but random) relative amount to overcome difficulties with
	    //          glued matrices.
	    //
	    for (i__ = 1; i__ <= 4; ++i__) {
		iseed[i__ - 1] = 1;
// L122:
	    }
	    i__2 = (in << 1) - 1;
	    dlarnv_(&c__2, iseed, &i__2, &work[1]);
	    i__2 = in - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		d__[ibegin + i__ - 1] *= eps * 8. * work[i__] + 1.;
		e[ibegin + i__ - 1] *= eps * 8. * work[in + i__] + 1.;
// L125:
	    }
	    d__[iend] *= eps * 4. * work[in] + 1.;
	}
	//
	//       Don't update the Gerschgorin intervals because keeping track
	//       of the updates would be too much work in DLARRV.
	//       We update W instead and use it to locate the proper Gerschgorin
	//       intervals.
	//       Compute the required eigenvalues of L D L' by bisection or dqds
	if (! usedqd) {
	    //          If DLARRD has been used, shift the eigenvalue approximations
	    //          according to their representation. This is necessary for
	    //          a uniform DLARRV since dqds computes eigenvalues of the
	    //          shifted representation. In DLARRV, W will always hold the
	    //          UNshifted eigenvalue approximation.
	    i__2 = wend;
	    for (j = wbegin; j <= i__2; ++j) {
		w[j] -= sigma;
		werr[j] += (d__1 = w[j], abs(d__1)) * eps;
// L134:
	    }
	    //          call DLARRB to reduce eigenvalue error of the approximations
	    //          from DLARRD
	    i__2 = iend - 1;
	    for (i__ = ibegin; i__ <= i__2; ++i__) {
		// Computing 2nd power
		d__1 = e[i__];
		work[i__] = d__[i__] * (d__1 * d__1);
// L135:
	    }
	    //          use bisection to find EV from INDL to INDU
	    i__2 = indl - 1;
	    dlarrb_(&in, &d__[ibegin], &work[ibegin], &indl, &indu, rtol1, 
		    rtol2, &i__2, &w[wbegin], &wgap[wbegin], &werr[wbegin], &
		    work[(*n << 1) + 1], &iwork[1], pivmin, &spdiam, &in, &
		    iinfo);
	    if (iinfo != 0) {
		*info = -4;
		return 0;
	    }
	    //          DLARRB computes all gaps correctly except for the last one
	    //          Record distance to VU/GU
	    // Computing MAX
	    d__1 = 0., d__2 = *vu - sigma - (w[wend] + werr[wend]);
	    wgap[wend] = max(d__1,d__2);
	    i__2 = indu;
	    for (i__ = indl; i__ <= i__2; ++i__) {
		++(*m);
		iblock[*m] = jblk;
		indexw[*m] = i__;
// L138:
	    }
	} else {
	    //          Call dqds to get all eigs (and then possibly delete unwanted
	    //          eigenvalues).
	    //          Note that dqds finds the eigenvalues of the L D L^T representation
	    //          of T to high relative accuracy. High relative accuracy
	    //          might be lost when the shift of the RRR is subtracted to obtain
	    //          the eigenvalues of T. However, T is not guaranteed to define its
	    //          eigenvalues to high relative accuracy anyway.
	    //          Set RTOL to the order of the tolerance used in DLASQ2
	    //          This is an ESTIMATED error, the worst case bound is 4*N*EPS
	    //          which is usually too large and requires unnecessary work to be
	    //          done by bisection when computing the eigenvectors
	    rtol = log((double) in) * 4. * eps;
	    j = ibegin;
	    i__2 = in - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		work[(i__ << 1) - 1] = (d__1 = d__[j], abs(d__1));
		work[i__ * 2] = e[j] * e[j] * work[(i__ << 1) - 1];
		++j;
// L140:
	    }
	    work[(in << 1) - 1] = (d__1 = d__[iend], abs(d__1));
	    work[in * 2] = 0.;
	    dlasq2_(&in, &work[1], &iinfo);
	    if (iinfo != 0) {
		//             If IINFO = -5 then an index is part of a tight cluster
		//             and should be changed. The index is in IWORK(1) and the
		//             gap is in WORK(N+1)
		*info = -5;
		return 0;
	    } else {
		//             Test that all eigenvalues are positive as expected
		i__2 = in;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    if (work[i__] < 0.) {
			*info = -6;
			return 0;
		    }
// L149:
		}
	    }
	    if (sgndef > 0.) {
		i__2 = indu;
		for (i__ = indl; i__ <= i__2; ++i__) {
		    ++(*m);
		    w[*m] = work[in - i__ + 1];
		    iblock[*m] = jblk;
		    indexw[*m] = i__;
// L150:
		}
	    } else {
		i__2 = indu;
		for (i__ = indl; i__ <= i__2; ++i__) {
		    ++(*m);
		    w[*m] = -work[i__];
		    iblock[*m] = jblk;
		    indexw[*m] = i__;
// L160:
		}
	    }
	    i__2 = *m;
	    for (i__ = *m - mb + 1; i__ <= i__2; ++i__) {
		//             the value of RTOL below should be the tolerance in DLASQ2
		werr[i__] = rtol * (d__1 = w[i__], abs(d__1));
// L165:
	    }
	    i__2 = *m - 1;
	    for (i__ = *m - mb + 1; i__ <= i__2; ++i__) {
		//             compute the right gap between the intervals
		// Computing MAX
		d__1 = 0., d__2 = w[i__ + 1] - werr[i__ + 1] - (w[i__] + werr[
			i__]);
		wgap[i__] = max(d__1,d__2);
// L166:
	    }
	    // Computing MAX
	    d__1 = 0., d__2 = *vu - sigma - (w[*m] + werr[*m]);
	    wgap[*m] = max(d__1,d__2);
	}
	//       proceed with next block
	ibegin = iend + 1;
	wbegin = wend + 1;
L170:
	;
    }
    return 0;
    //
    //    end of DLARRE
    //
} // dlarre_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRF finds a new relatively robust representation such that at least one of the eigenvalues is relatively isolated.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRF( N, D, L, LD, CLSTRT, CLEND,
//                         W, WGAP, WERR,
//                         SPDIAM, CLGAPL, CLGAPR, PIVMIN, SIGMA,
//                         DPLUS, LPLUS, WORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            CLSTRT, CLEND, INFO, N
//      DOUBLE PRECISION   CLGAPL, CLGAPR, PIVMIN, SIGMA, SPDIAM
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), DPLUS( * ), L( * ), LD( * ),
//     $          LPLUS( * ), W( * ), WGAP( * ), WERR( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Given the initial representation L D L^T and its cluster of close
//> eigenvalues (in a relative measure), W( CLSTRT ), W( CLSTRT+1 ), ...
//> W( CLEND ), DLARRF finds a new relatively robust representation
//> L D L^T - SIGMA I = L(+) D(+) L(+)^T such that at least one of the
//> eigenvalues of L(+) D(+) L(+)^T is relatively isolated.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix (subblock, if the matrix split).
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The N diagonal elements of the diagonal matrix D.
//> \endverbatim
//>
//> \param[in] L
//> \verbatim
//>          L is DOUBLE PRECISION array, dimension (N-1)
//>          The (N-1) subdiagonal elements of the unit bidiagonal
//>          matrix L.
//> \endverbatim
//>
//> \param[in] LD
//> \verbatim
//>          LD is DOUBLE PRECISION array, dimension (N-1)
//>          The (N-1) elements L(i)*D(i).
//> \endverbatim
//>
//> \param[in] CLSTRT
//> \verbatim
//>          CLSTRT is INTEGER
//>          The index of the first eigenvalue in the cluster.
//> \endverbatim
//>
//> \param[in] CLEND
//> \verbatim
//>          CLEND is INTEGER
//>          The index of the last eigenvalue in the cluster.
//> \endverbatim
//>
//> \param[in] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension
//>          dimension is >=  (CLEND-CLSTRT+1)
//>          The eigenvalue APPROXIMATIONS of L D L^T in ascending order.
//>          W( CLSTRT ) through W( CLEND ) form the cluster of relatively
//>          close eigenalues.
//> \endverbatim
//>
//> \param[in,out] WGAP
//> \verbatim
//>          WGAP is DOUBLE PRECISION array, dimension
//>          dimension is >=  (CLEND-CLSTRT+1)
//>          The separation from the right neighbor eigenvalue in W.
//> \endverbatim
//>
//> \param[in] WERR
//> \verbatim
//>          WERR is DOUBLE PRECISION array, dimension
//>          dimension is  >=  (CLEND-CLSTRT+1)
//>          WERR contain the semiwidth of the uncertainty
//>          interval of the corresponding eigenvalue APPROXIMATION in W
//> \endverbatim
//>
//> \param[in] SPDIAM
//> \verbatim
//>          SPDIAM is DOUBLE PRECISION
//>          estimate of the spectral diameter obtained from the
//>          Gerschgorin intervals
//> \endverbatim
//>
//> \param[in] CLGAPL
//> \verbatim
//>          CLGAPL is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] CLGAPR
//> \verbatim
//>          CLGAPR is DOUBLE PRECISION
//>          absolute gap on each end of the cluster.
//>          Set by the calling routine to protect against shifts too close
//>          to eigenvalues outside the cluster.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot allowed in the Sturm sequence.
//> \endverbatim
//>
//> \param[out] SIGMA
//> \verbatim
//>          SIGMA is DOUBLE PRECISION
//>          The shift used to form L(+) D(+) L(+)^T.
//> \endverbatim
//>
//> \param[out] DPLUS
//> \verbatim
//>          DPLUS is DOUBLE PRECISION array, dimension (N)
//>          The N diagonal elements of the diagonal matrix D(+).
//> \endverbatim
//>
//> \param[out] LPLUS
//> \verbatim
//>          LPLUS is DOUBLE PRECISION array, dimension (N-1)
//>          The first (N-1) elements of LPLUS contain the subdiagonal
//>          elements of the unit bidiagonal matrix L(+).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (2*N)
//>          Workspace.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          Signals processing OK (=0) or failure (=1)
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
//> \par Contributors:
// ==================
//>
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlarrf_(int *n, double *d__, double *l, double *ld, int *
	clstrt, int *clend, double *w, double *wgap, double *werr, double *
	spdiam, double *clgapl, double *clgapr, double *pivmin, double *sigma,
	 double *dplus, double *lplus, double *work, int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int i__1;
    double d__1, d__2, d__3;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int i__;
    double s, bestshift, smlgrowth, eps, tmp, max1, max2, rrr1, rrr2, znm2, 
	    growthbound, fail, fact, oldp;
    int indx;
    double prod;
    int ktry;
    double fail2, avgap, ldmax, rdmax;
    int shift;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int dorrr1;
    extern double dlamch_(char *);
    double ldelta;
    int nofail;
    double mingap, lsigma, rdelta;
    extern int disnan_(double *);
    int forcer;
    double rsigma, clwdth;
    int sawnan1, sawnan2, tryrrr1;

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
    // Parameter adjustments
    --work;
    --lplus;
    --dplus;
    --werr;
    --wgap;
    --w;
    --ld;
    --l;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    fact = 2.;
    eps = dlamch_("Precision");
    shift = 0;
    forcer = FALSE_;
    //    Note that we cannot guarantee that for any of the shifts tried,
    //    the factorization has a small or even moderate element growth.
    //    There could be Ritz values at both ends of the cluster and despite
    //    backing off, there are examples where all factorizations tried
    //    (in IEEE mode, allowing zero pivots & infinities) have INFINITE
    //    element growth.
    //    For this reason, we should use PIVMIN in this subroutine so that at
    //    least the L D L^T factorization exists. It can be checked afterwards
    //    whether the element growth caused bad residuals/orthogonality.
    //    Decide whether the code should accept the best among all
    //    representations despite large element growth or signal INFO=1
    //    Setting NOFAIL to .FALSE. for quick fix for bug 113
    nofail = FALSE_;
    //
    //    Compute the average gap length of the cluster
    clwdth = (d__1 = w[*clend] - w[*clstrt], abs(d__1)) + werr[*clend] + werr[
	    *clstrt];
    avgap = clwdth / (double) (*clend - *clstrt);
    mingap = min(*clgapl,*clgapr);
    //    Initial values for shifts to both ends of cluster
    // Computing MIN
    d__1 = w[*clstrt], d__2 = w[*clend];
    lsigma = min(d__1,d__2) - werr[*clstrt];
    // Computing MAX
    d__1 = w[*clstrt], d__2 = w[*clend];
    rsigma = max(d__1,d__2) + werr[*clend];
    //    Use a small fudge to make sure that we really shift to the outside
    lsigma -= abs(lsigma) * 4. * eps;
    rsigma += abs(rsigma) * 4. * eps;
    //    Compute upper bounds for how much to back off the initial shifts
    ldmax = mingap * .25 + *pivmin * 2.;
    rdmax = mingap * .25 + *pivmin * 2.;
    // Computing MAX
    d__1 = avgap, d__2 = wgap[*clstrt];
    ldelta = max(d__1,d__2) / fact;
    // Computing MAX
    d__1 = avgap, d__2 = wgap[*clend - 1];
    rdelta = max(d__1,d__2) / fact;
    //
    //    Initialize the record of the best representation found
    //
    s = dlamch_("S");
    smlgrowth = 1. / s;
    fail = (double) (*n - 1) * mingap / (*spdiam * eps);
    fail2 = (double) (*n - 1) * mingap / (*spdiam * sqrt(eps));
    bestshift = lsigma;
    //
    //    while (KTRY <= KTRYMAX)
    ktry = 0;
    growthbound = *spdiam * 8.;
L5:
    sawnan1 = FALSE_;
    sawnan2 = FALSE_;
    //    Ensure that we do not back off too much of the initial shifts
    ldelta = min(ldmax,ldelta);
    rdelta = min(rdmax,rdelta);
    //    Compute the element growth when shifting to both ends of the cluster
    //    accept the shift if there is no element growth at one of the two ends
    //    Left end
    s = -lsigma;
    dplus[1] = d__[1] + s;
    if (abs(dplus[1]) < *pivmin) {
	dplus[1] = -(*pivmin);
	//       Need to set SAWNAN1 because refined RRR test should not be used
	//       in this case
	sawnan1 = TRUE_;
    }
    max1 = abs(dplus[1]);
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	lplus[i__] = ld[i__] / dplus[i__];
	s = s * lplus[i__] * l[i__] - lsigma;
	dplus[i__ + 1] = d__[i__ + 1] + s;
	if ((d__1 = dplus[i__ + 1], abs(d__1)) < *pivmin) {
	    dplus[i__ + 1] = -(*pivmin);
	    //          Need to set SAWNAN1 because refined RRR test should not be used
	    //          in this case
	    sawnan1 = TRUE_;
	}
	// Computing MAX
	d__2 = max1, d__3 = (d__1 = dplus[i__ + 1], abs(d__1));
	max1 = max(d__2,d__3);
// L6:
    }
    sawnan1 = sawnan1 || disnan_(&max1);
    if (forcer || max1 <= growthbound && ! sawnan1) {
	*sigma = lsigma;
	shift = 1;
	goto L100;
    }
    //    Right end
    s = -rsigma;
    work[1] = d__[1] + s;
    if (abs(work[1]) < *pivmin) {
	work[1] = -(*pivmin);
	//       Need to set SAWNAN2 because refined RRR test should not be used
	//       in this case
	sawnan2 = TRUE_;
    }
    max2 = abs(work[1]);
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[*n + i__] = ld[i__] / work[i__];
	s = s * work[*n + i__] * l[i__] - rsigma;
	work[i__ + 1] = d__[i__ + 1] + s;
	if ((d__1 = work[i__ + 1], abs(d__1)) < *pivmin) {
	    work[i__ + 1] = -(*pivmin);
	    //          Need to set SAWNAN2 because refined RRR test should not be used
	    //          in this case
	    sawnan2 = TRUE_;
	}
	// Computing MAX
	d__2 = max2, d__3 = (d__1 = work[i__ + 1], abs(d__1));
	max2 = max(d__2,d__3);
// L7:
    }
    sawnan2 = sawnan2 || disnan_(&max2);
    if (forcer || max2 <= growthbound && ! sawnan2) {
	*sigma = rsigma;
	shift = 2;
	goto L100;
    }
    //    If we are at this point, both shifts led to too much element growth
    //    Record the better of the two shifts (provided it didn't lead to NaN)
    if (sawnan1 && sawnan2) {
	//       both MAX1 and MAX2 are NaN
	goto L50;
    } else {
	if (! sawnan1) {
	    indx = 1;
	    if (max1 <= smlgrowth) {
		smlgrowth = max1;
		bestshift = lsigma;
	    }
	}
	if (! sawnan2) {
	    if (sawnan1 || max2 <= max1) {
		indx = 2;
	    }
	    if (max2 <= smlgrowth) {
		smlgrowth = max2;
		bestshift = rsigma;
	    }
	}
    }
    //    If we are here, both the left and the right shift led to
    //    element growth. If the element growth is moderate, then
    //    we may still accept the representation, if it passes a
    //    refined test for RRR. This test supposes that no NaN occurred.
    //    Moreover, we use the refined RRR test only for isolated clusters.
    if (clwdth < mingap / 128. && min(max1,max2) < fail2 && ! sawnan1 && ! 
	    sawnan2) {
	dorrr1 = TRUE_;
    } else {
	dorrr1 = FALSE_;
    }
    tryrrr1 = TRUE_;
    if (tryrrr1 && dorrr1) {
	if (indx == 1) {
	    tmp = (d__1 = dplus[*n], abs(d__1));
	    znm2 = 1.;
	    prod = 1.;
	    oldp = 1.;
	    for (i__ = *n - 1; i__ >= 1; --i__) {
		if (prod <= eps) {
		    prod = dplus[i__ + 1] * work[*n + i__ + 1] / (dplus[i__] *
			     work[*n + i__]) * oldp;
		} else {
		    prod *= (d__1 = work[*n + i__], abs(d__1));
		}
		oldp = prod;
		// Computing 2nd power
		d__1 = prod;
		znm2 += d__1 * d__1;
		// Computing MAX
		d__2 = tmp, d__3 = (d__1 = dplus[i__] * prod, abs(d__1));
		tmp = max(d__2,d__3);
// L15:
	    }
	    rrr1 = tmp / (*spdiam * sqrt(znm2));
	    if (rrr1 <= 8.) {
		*sigma = lsigma;
		shift = 1;
		goto L100;
	    }
	} else if (indx == 2) {
	    tmp = (d__1 = work[*n], abs(d__1));
	    znm2 = 1.;
	    prod = 1.;
	    oldp = 1.;
	    for (i__ = *n - 1; i__ >= 1; --i__) {
		if (prod <= eps) {
		    prod = work[i__ + 1] * lplus[i__ + 1] / (work[i__] * 
			    lplus[i__]) * oldp;
		} else {
		    prod *= (d__1 = lplus[i__], abs(d__1));
		}
		oldp = prod;
		// Computing 2nd power
		d__1 = prod;
		znm2 += d__1 * d__1;
		// Computing MAX
		d__2 = tmp, d__3 = (d__1 = work[i__] * prod, abs(d__1));
		tmp = max(d__2,d__3);
// L16:
	    }
	    rrr2 = tmp / (*spdiam * sqrt(znm2));
	    if (rrr2 <= 8.) {
		*sigma = rsigma;
		shift = 2;
		goto L100;
	    }
	}
    }
L50:
    if (ktry < 1) {
	//       If we are here, both shifts failed also the RRR test.
	//       Back off to the outside
	// Computing MAX
	d__1 = lsigma - ldelta, d__2 = lsigma - ldmax;
	lsigma = max(d__1,d__2);
	// Computing MIN
	d__1 = rsigma + rdelta, d__2 = rsigma + rdmax;
	rsigma = min(d__1,d__2);
	ldelta *= 2.;
	rdelta *= 2.;
	++ktry;
	goto L5;
    } else {
	//       None of the representations investigated satisfied our
	//       criteria. Take the best one we found.
	if (smlgrowth < fail || nofail) {
	    lsigma = bestshift;
	    rsigma = bestshift;
	    forcer = TRUE_;
	    goto L5;
	} else {
	    *info = 1;
	    return 0;
	}
    }
L100:
    if (shift == 1) {
    } else if (shift == 2) {
	//       store new L and D back into DPLUS, LPLUS
	dcopy_(n, &work[1], &c__1, &dplus[1], &c__1);
	i__1 = *n - 1;
	dcopy_(&i__1, &work[*n + 1], &c__1, &lplus[1], &c__1);
    }
    return 0;
    //
    //    End of DLARRF
    //
} // dlarrf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRJ performs refinement of the initial estimates of the eigenvalues of the matrix T.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRJ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrj.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrj.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrj.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRJ( N, D, E2, IFIRST, ILAST,
//                         RTOL, OFFSET, W, WERR, WORK, IWORK,
//                         PIVMIN, SPDIAM, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            IFIRST, ILAST, INFO, N, OFFSET
//      DOUBLE PRECISION   PIVMIN, RTOL, SPDIAM
//      ..
//      .. Array Arguments ..
//      INTEGER            IWORK( * )
//      DOUBLE PRECISION   D( * ), E2( * ), W( * ),
//     $                   WERR( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Given the initial eigenvalue approximations of T, DLARRJ
//> does  bisection to refine the eigenvalues of T,
//> W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial
//> guesses for these eigenvalues are input in W, the corresponding estimate
//> of the error in these guesses in WERR. During bisection, intervals
//> [left, right] are maintained by storing their mid-points and
//> semi-widths in the arrays W and WERR respectively.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The N diagonal elements of T.
//> \endverbatim
//>
//> \param[in] E2
//> \verbatim
//>          E2 is DOUBLE PRECISION array, dimension (N-1)
//>          The Squares of the (N-1) subdiagonal elements of T.
//> \endverbatim
//>
//> \param[in] IFIRST
//> \verbatim
//>          IFIRST is INTEGER
//>          The index of the first eigenvalue to be computed.
//> \endverbatim
//>
//> \param[in] ILAST
//> \verbatim
//>          ILAST is INTEGER
//>          The index of the last eigenvalue to be computed.
//> \endverbatim
//>
//> \param[in] RTOL
//> \verbatim
//>          RTOL is DOUBLE PRECISION
//>          Tolerance for the convergence of the bisection intervals.
//>          An interval [LEFT,RIGHT] has converged if
//>          RIGHT-LEFT < RTOL*MAX(|LEFT|,|RIGHT|).
//> \endverbatim
//>
//> \param[in] OFFSET
//> \verbatim
//>          OFFSET is INTEGER
//>          Offset for the arrays W and WERR, i.e., the IFIRST-OFFSET
//>          through ILAST-OFFSET elements of these arrays are to be used.
//> \endverbatim
//>
//> \param[in,out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          On input, W( IFIRST-OFFSET ) through W( ILAST-OFFSET ) are
//>          estimates of the eigenvalues of L D L^T indexed IFIRST through
//>          ILAST.
//>          On output, these estimates are refined.
//> \endverbatim
//>
//> \param[in,out] WERR
//> \verbatim
//>          WERR is DOUBLE PRECISION array, dimension (N)
//>          On input, WERR( IFIRST-OFFSET ) through WERR( ILAST-OFFSET ) are
//>          the errors in the estimates of the corresponding elements in W.
//>          On output, these errors are refined.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (2*N)
//>          Workspace.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (2*N)
//>          Workspace.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot in the Sturm sequence for T.
//> \endverbatim
//>
//> \param[in] SPDIAM
//> \verbatim
//>          SPDIAM is DOUBLE PRECISION
//>          The spectral diameter of T.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          Error flag.
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
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlarrj_(int *n, double *d__, double *e2, int *ifirst, 
	int *ilast, double *rtol, int *offset, double *w, double *werr, 
	double *work, int *iwork, double *pivmin, double *spdiam, int *info)
{
    // System generated locals
    int i__1, i__2;
    double d__1, d__2;

    // Builtin functions
    double log(double);

    // Local variables
    int i__, j, k, p;
    double s;
    int i1, i2, ii;
    double fac, mid;
    int cnt;
    double tmp, left;
    int iter, nint, prev, next, savi1;
    double right, width, dplus;
    int olnint, maxitr;

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
    // =====================================================================
    //
    //    .. Parameters ..
    //    ..
    //    .. Local Scalars ..
    //
    //    ..
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    // Parameter adjustments
    --iwork;
    --work;
    --werr;
    --w;
    --e2;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    maxitr = (int) ((log(*spdiam + *pivmin) - log(*pivmin)) / log(2.)) + 2;
    //
    //    Initialize unconverged intervals in [ WORK(2*I-1), WORK(2*I) ].
    //    The Sturm Count, Count( WORK(2*I-1) ) is arranged to be I-1, while
    //    Count( WORK(2*I) ) is stored in IWORK( 2*I ). The integer IWORK( 2*I-1 )
    //    for an unconverged interval is set to the index of the next unconverged
    //    interval, and is -1 or 0 for a converged interval. Thus a linked
    //    list of unconverged intervals is set up.
    //
    i1 = *ifirst;
    i2 = *ilast;
    //    The number of unconverged intervals
    nint = 0;
    //    The last unconverged interval found
    prev = 0;
    i__1 = i2;
    for (i__ = i1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	left = w[ii] - werr[ii];
	mid = w[ii];
	right = w[ii] + werr[ii];
	width = right - mid;
	// Computing MAX
	d__1 = abs(left), d__2 = abs(right);
	tmp = max(d__1,d__2);
	//       The following test prevents the test of converged intervals
	if (width < *rtol * tmp) {
	    //          This interval has already converged and does not need refinement.
	    //          (Note that the gaps might change through refining the
	    //           eigenvalues, however, they can only get bigger.)
	    //          Remove it from the list.
	    iwork[k - 1] = -1;
	    //          Make sure that I1 always points to the first unconverged interval
	    if (i__ == i1 && i__ < i2) {
		i1 = i__ + 1;
	    }
	    if (prev >= i1 && i__ <= i2) {
		iwork[(prev << 1) - 1] = i__ + 1;
	    }
	} else {
	    //          unconverged interval found
	    prev = i__;
	    //          Make sure that [LEFT,RIGHT] contains the desired eigenvalue
	    //
	    //          Do while( CNT(LEFT).GT.I-1 )
	    //
	    fac = 1.;
L20:
	    cnt = 0;
	    s = left;
	    dplus = d__[1] - s;
	    if (dplus < 0.) {
		++cnt;
	    }
	    i__2 = *n;
	    for (j = 2; j <= i__2; ++j) {
		dplus = d__[j] - s - e2[j - 1] / dplus;
		if (dplus < 0.) {
		    ++cnt;
		}
// L30:
	    }
	    if (cnt > i__ - 1) {
		left -= werr[ii] * fac;
		fac *= 2.;
		goto L20;
	    }
	    //
	    //          Do while( CNT(RIGHT).LT.I )
	    //
	    fac = 1.;
L50:
	    cnt = 0;
	    s = right;
	    dplus = d__[1] - s;
	    if (dplus < 0.) {
		++cnt;
	    }
	    i__2 = *n;
	    for (j = 2; j <= i__2; ++j) {
		dplus = d__[j] - s - e2[j - 1] / dplus;
		if (dplus < 0.) {
		    ++cnt;
		}
// L60:
	    }
	    if (cnt < i__) {
		right += werr[ii] * fac;
		fac *= 2.;
		goto L50;
	    }
	    ++nint;
	    iwork[k - 1] = i__ + 1;
	    iwork[k] = cnt;
	}
	work[k - 1] = left;
	work[k] = right;
// L75:
    }
    savi1 = i1;
    //
    //    Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
    //    and while (ITER.LT.MAXITR)
    //
    iter = 0;
L80:
    prev = i1 - 1;
    i__ = i1;
    olnint = nint;
    i__1 = olnint;
    for (p = 1; p <= i__1; ++p) {
	k = i__ << 1;
	ii = i__ - *offset;
	next = iwork[k - 1];
	left = work[k - 1];
	right = work[k];
	mid = (left + right) * .5;
	//       semiwidth of interval
	width = right - mid;
	// Computing MAX
	d__1 = abs(left), d__2 = abs(right);
	tmp = max(d__1,d__2);
	if (width < *rtol * tmp || iter == maxitr) {
	    //          reduce number of unconverged intervals
	    --nint;
	    //          Mark interval as converged.
	    iwork[k - 1] = 0;
	    if (i1 == i__) {
		i1 = next;
	    } else {
		//             Prev holds the last unconverged interval previously examined
		if (prev >= i1) {
		    iwork[(prev << 1) - 1] = next;
		}
	    }
	    i__ = next;
	    goto L100;
	}
	prev = i__;
	//
	//       Perform one bisection step
	//
	cnt = 0;
	s = mid;
	dplus = d__[1] - s;
	if (dplus < 0.) {
	    ++cnt;
	}
	i__2 = *n;
	for (j = 2; j <= i__2; ++j) {
	    dplus = d__[j] - s - e2[j - 1] / dplus;
	    if (dplus < 0.) {
		++cnt;
	    }
// L90:
	}
	if (cnt <= i__ - 1) {
	    work[k - 1] = mid;
	} else {
	    work[k] = mid;
	}
	i__ = next;
L100:
	;
    }
    ++iter;
    //    do another loop if there are still unconverged intervals
    //    However, in the last iteration, all intervals are accepted
    //    since this is the best we can do.
    if (nint > 0 && iter <= maxitr) {
	goto L80;
    }
    //
    //
    //    At this point, all the intervals have converged
    i__1 = *ilast;
    for (i__ = savi1; i__ <= i__1; ++i__) {
	k = i__ << 1;
	ii = i__ - *offset;
	//       All intervals marked by '0' have been refined.
	if (iwork[k - 1] == 0) {
	    w[ii] = (work[k - 1] + work[k]) * .5;
	    werr[ii] = work[k] - w[ii];
	}
// L110:
    }
    return 0;
    //
    //    End of DLARRJ
    //
} // dlarrj_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRK computes one eigenvalue of a symmetric tridiagonal matrix T to suitable accuracy.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRK + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrk.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrk.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrk.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRK( N, IW, GL, GU,
//                          D, E2, PIVMIN, RELTOL, W, WERR, INFO)
//
//      .. Scalar Arguments ..
//      INTEGER   INFO, IW, N
//      DOUBLE PRECISION    PIVMIN, RELTOL, GL, GU, W, WERR
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), E2( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLARRK computes one eigenvalue of a symmetric tridiagonal
//> matrix T to suitable accuracy. This is an auxiliary code to be
//> called from DSTEMR.
//>
//> To avoid overflow, the matrix must be scaled so that its
//> largest element is no greater than overflow**(1/2) * underflow**(1/4) in absolute value, and for greatest
//> accuracy, it should not be much smaller than that.
//>
//> See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
//> Matrix", Report CS41, Computer Science Dept., Stanford
//> University, July 21, 1966.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the tridiagonal matrix T.  N >= 0.
//> \endverbatim
//>
//> \param[in] IW
//> \verbatim
//>          IW is INTEGER
//>          The index of the eigenvalues to be returned.
//> \endverbatim
//>
//> \param[in] GL
//> \verbatim
//>          GL is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] GU
//> \verbatim
//>          GU is DOUBLE PRECISION
//>          An upper and a lower bound on the eigenvalue.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The n diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] E2
//> \verbatim
//>          E2 is DOUBLE PRECISION array, dimension (N-1)
//>          The (n-1) squared off-diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot allowed in the Sturm sequence for T.
//> \endverbatim
//>
//> \param[in] RELTOL
//> \verbatim
//>          RELTOL is DOUBLE PRECISION
//>          The minimum relative width of an interval.  When an interval
//>          is narrower than RELTOL times the larger (in
//>          magnitude) endpoint, then it is considered to be
//>          sufficiently small, i.e., converged.  Note: this should
//>          always be at least radix*machine epsilon.
//> \endverbatim
//>
//> \param[out] W
//> \verbatim
//>          W is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[out] WERR
//> \verbatim
//>          WERR is DOUBLE PRECISION
//>          The error bound on the corresponding eigenvalue approximation
//>          in W.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:       Eigenvalue converged
//>          = -1:      Eigenvalue did NOT converge
//> \endverbatim
//
//> \par Internal Parameters:
// =========================
//>
//> \verbatim
//>  FUDGE   DOUBLE PRECISION, default = 2
//>          A "fudge factor" to widen the Gershgorin intervals.
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
//> \ingroup OTHERauxiliary
//
// =====================================================================
/* Subroutine */ int dlarrk_(int *n, int *iw, double *gl, double *gu, double *
	d__, double *e2, double *pivmin, double *reltol, double *w, double *
	werr, int *info)
{
    // System generated locals
    int i__1;
    double d__1, d__2;

    // Builtin functions
    double log(double);

    // Local variables
    int i__, it;
    double mid, eps, tmp1, tmp2, left, atoli, right;
    int itmax;
    double rtoli, tnorm;
    extern double dlamch_(char *);
    int negcnt;

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
    //    Quick return if possible
    //
    // Parameter adjustments
    --e2;
    --d__;

    // Function Body
    if (*n <= 0) {
	*info = 0;
	return 0;
    }
    //
    //    Get machine constants
    eps = dlamch_("P");
    // Computing MAX
    d__1 = abs(*gl), d__2 = abs(*gu);
    tnorm = max(d__1,d__2);
    rtoli = *reltol;
    atoli = *pivmin * 4.;
    itmax = (int) ((log(tnorm + *pivmin) - log(*pivmin)) / log(2.)) + 2;
    *info = -1;
    left = *gl - tnorm * 2. * eps * *n - *pivmin * 4.;
    right = *gu + tnorm * 2. * eps * *n + *pivmin * 4.;
    it = 0;
L10:
    //
    //    Check if interval converged or maximum number of iterations reached
    //
    tmp1 = (d__1 = right - left, abs(d__1));
    // Computing MAX
    d__1 = abs(right), d__2 = abs(left);
    tmp2 = max(d__1,d__2);
    // Computing MAX
    d__1 = max(atoli,*pivmin), d__2 = rtoli * tmp2;
    if (tmp1 < max(d__1,d__2)) {
	*info = 0;
	goto L30;
    }
    if (it > itmax) {
	goto L30;
    }
    //
    //    Count number of negative pivots for mid-point
    //
    ++it;
    mid = (left + right) * .5;
    negcnt = 0;
    tmp1 = d__[1] - mid;
    if (abs(tmp1) < *pivmin) {
	tmp1 = -(*pivmin);
    }
    if (tmp1 <= 0.) {
	++negcnt;
    }
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	tmp1 = d__[i__] - e2[i__ - 1] / tmp1 - mid;
	if (abs(tmp1) < *pivmin) {
	    tmp1 = -(*pivmin);
	}
	if (tmp1 <= 0.) {
	    ++negcnt;
	}
// L20:
    }
    if (negcnt >= *iw) {
	right = mid;
    } else {
	left = mid;
    }
    goto L10;
L30:
    //
    //    Converged or maximum number of iterations reached
    //
    *w = (left + right) * .5;
    *werr = (d__1 = right - left, abs(d__1)) * .5;
    return 0;
    //
    //    End of DLARRK
    //
} // dlarrk_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRR performs tests to decide whether the symmetric tridiagonal matrix T warrants expensive computations which guarantee high relative accuracy in the eigenvalues.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRR( N, D, E, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            N, INFO
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), E( * )
//      ..
//
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Perform tests to decide whether the symmetric tridiagonal matrix T
//> warrants expensive computations which guarantee high relative accuracy
//> in the eigenvalues.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix. N > 0.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The N diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N)
//>          On entry, the first (N-1) entries contain the subdiagonal
//>          elements of the tridiagonal matrix T; E(N) is set to ZERO.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          INFO = 0(default) : the matrix warrants computations preserving
//>                              relative accuracy.
//>          INFO = 1          : the matrix warrants computations guaranteeing
//>                              only absolute accuracy.
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
//> \ingroup OTHERauxiliary
//
//> \par Contributors:
// ==================
//>
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlarrr_(int *n, double *d__, double *e, int *info)
{
    // System generated locals
    int i__1;
    double d__1;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int i__;
    double eps, tmp, tmp2, rmin;
    extern double dlamch_(char *);
    double offdig, safmin;
    int yesrel;
    double smlnum, offdig2;

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
    //    Quick return if possible
    //
    // Parameter adjustments
    --e;
    --d__;

    // Function Body
    if (*n <= 0) {
	*info = 0;
	return 0;
    }
    //
    //    As a default, do NOT go for relative-accuracy preserving computations.
    *info = 1;
    safmin = dlamch_("Safe minimum");
    eps = dlamch_("Precision");
    smlnum = safmin / eps;
    rmin = sqrt(smlnum);
    //    Tests for relative accuracy
    //
    //    Test for scaled diagonal dominance
    //    Scale the diagonal entries to one and check whether the sum of the
    //    off-diagonals is less than one
    //
    //    The sdd relative error bounds have a 1/(1- 2*x) factor in them,
    //    x = max(OFFDIG + OFFDIG2), so when x is close to 1/2, no relative
    //    accuracy is promised.  In the notation of the code fragment below,
    //    1/(1 - (OFFDIG + OFFDIG2)) is the condition number.
    //    We don't think it is worth going into "sdd mode" unless the relative
    //    condition number is reasonable, not 1/macheps.
    //    The threshold should be compatible with other thresholds used in the
    //    code. We set  OFFDIG + OFFDIG2 <= .999 =: RELCOND, it corresponds
    //    to losing at most 3 decimal digits: 1 / (1 - (OFFDIG + OFFDIG2)) <= 1000
    //    instead of the current OFFDIG + OFFDIG2 < 1
    //
    yesrel = TRUE_;
    offdig = 0.;
    tmp = sqrt((abs(d__[1])));
    if (tmp < rmin) {
	yesrel = FALSE_;
    }
    if (! yesrel) {
	goto L11;
    }
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	tmp2 = sqrt((d__1 = d__[i__], abs(d__1)));
	if (tmp2 < rmin) {
	    yesrel = FALSE_;
	}
	if (! yesrel) {
	    goto L11;
	}
	offdig2 = (d__1 = e[i__ - 1], abs(d__1)) / (tmp * tmp2);
	if (offdig + offdig2 >= .999) {
	    yesrel = FALSE_;
	}
	if (! yesrel) {
	    goto L11;
	}
	tmp = tmp2;
	offdig = offdig2;
// L10:
    }
L11:
    if (yesrel) {
	*info = 0;
	return 0;
    } else {
    }
    //
    //
    //    *** MORE TO BE IMPLEMENTED ***
    //
    //
    //    Test if the lower bidiagonal matrix L from T = L D L^T
    //    (zero shift facto) is well conditioned
    //
    //
    //    Test if the upper bidiagonal matrix U from T = U D U^T
    //    (zero shift facto) is well conditioned.
    //    In this case, the matrix needs to be flipped and, at the end
    //    of the eigenvector computation, the flip needs to be applied
    //    to the computed eigenvectors (and the support)
    //
    //
    return 0;
    //
    //    END OF DLARRR
    //
} // dlarrr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARRV computes the eigenvectors of the tridiagonal matrix T = L D LT given L, D and the eigenvalues of L D LT.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARRV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlarrv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlarrv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlarrv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARRV( N, VL, VU, D, L, PIVMIN,
//                         ISPLIT, M, DOL, DOU, MINRGP,
//                         RTOL1, RTOL2, W, WERR, WGAP,
//                         IBLOCK, INDEXW, GERS, Z, LDZ, ISUPPZ,
//                         WORK, IWORK, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            DOL, DOU, INFO, LDZ, M, N
//      DOUBLE PRECISION   MINRGP, PIVMIN, RTOL1, RTOL2, VL, VU
//      ..
//      .. Array Arguments ..
//      INTEGER            IBLOCK( * ), INDEXW( * ), ISPLIT( * ),
//     $                   ISUPPZ( * ), IWORK( * )
//      DOUBLE PRECISION   D( * ), GERS( * ), L( * ), W( * ), WERR( * ),
//     $                   WGAP( * ), WORK( * )
//      DOUBLE PRECISION  Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLARRV computes the eigenvectors of the tridiagonal matrix
//> T = L D L**T given L, D and APPROXIMATIONS to the eigenvalues of L D L**T.
//> The input eigenvalues should have been computed by DLARRE.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix.  N >= 0.
//> \endverbatim
//>
//> \param[in] VL
//> \verbatim
//>          VL is DOUBLE PRECISION
//>          Lower bound of the interval that contains the desired
//>          eigenvalues. VL < VU. Needed to compute gaps on the left or right
//>          end of the extremal eigenvalues in the desired RANGE.
//> \endverbatim
//>
//> \param[in] VU
//> \verbatim
//>          VU is DOUBLE PRECISION
//>          Upper bound of the interval that contains the desired
//>          eigenvalues. VL < VU.
//>          Note: VU is currently not used by this implementation of DLARRV, VU is
//>          passed to DLARRV because it could be used compute gaps on the right end
//>          of the extremal eigenvalues. However, with not much initial accuracy in
//>          LAMBDA and VU, the formula can lead to an overestimation of the right gap
//>          and thus to inadequately early RQI 'convergence'. This is currently
//>          prevented this by forcing a small right gap. And so it turns out that VU
//>          is currently not used by this implementation of DLARRV.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the N diagonal elements of the diagonal matrix D.
//>          On exit, D may be overwritten.
//> \endverbatim
//>
//> \param[in,out] L
//> \verbatim
//>          L is DOUBLE PRECISION array, dimension (N)
//>          On entry, the (N-1) subdiagonal elements of the unit
//>          bidiagonal matrix L are in elements 1 to N-1 of L
//>          (if the matrix is not split.) At the end of each block
//>          is stored the corresponding shift as given by DLARRE.
//>          On exit, L is overwritten.
//> \endverbatim
//>
//> \param[in] PIVMIN
//> \verbatim
//>          PIVMIN is DOUBLE PRECISION
//>          The minimum pivot allowed in the Sturm sequence.
//> \endverbatim
//>
//> \param[in] ISPLIT
//> \verbatim
//>          ISPLIT is INTEGER array, dimension (N)
//>          The splitting points, at which T breaks up into blocks.
//>          The first block consists of rows/columns 1 to
//>          ISPLIT( 1 ), the second of rows/columns ISPLIT( 1 )+1
//>          through ISPLIT( 2 ), etc.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The total number of input eigenvalues.  0 <= M <= N.
//> \endverbatim
//>
//> \param[in] DOL
//> \verbatim
//>          DOL is INTEGER
//> \endverbatim
//>
//> \param[in] DOU
//> \verbatim
//>          DOU is INTEGER
//>          If the user wants to compute only selected eigenvectors from all
//>          the eigenvalues supplied, he can specify an index range DOL:DOU.
//>          Or else the setting DOL=1, DOU=M should be applied.
//>          Note that DOL and DOU refer to the order in which the eigenvalues
//>          are stored in W.
//>          If the user wants to compute only selected eigenpairs, then
//>          the columns DOL-1 to DOU+1 of the eigenvector space Z contain the
//>          computed eigenvectors. All other columns of Z are set to zero.
//> \endverbatim
//>
//> \param[in] MINRGP
//> \verbatim
//>          MINRGP is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] RTOL1
//> \verbatim
//>          RTOL1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] RTOL2
//> \verbatim
//>          RTOL2 is DOUBLE PRECISION
//>           Parameters for bisection.
//>           An interval [LEFT,RIGHT] has converged if
//>           RIGHT-LEFT < MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) )
//> \endverbatim
//>
//> \param[in,out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          The first M elements of W contain the APPROXIMATE eigenvalues for
//>          which eigenvectors are to be computed.  The eigenvalues
//>          should be grouped by split-off block and ordered from
//>          smallest to largest within the block ( The output array
//>          W from DLARRE is expected here ). Furthermore, they are with
//>          respect to the shift of the corresponding root representation
//>          for their block. On exit, W holds the eigenvalues of the
//>          UNshifted matrix.
//> \endverbatim
//>
//> \param[in,out] WERR
//> \verbatim
//>          WERR is DOUBLE PRECISION array, dimension (N)
//>          The first M elements contain the semiwidth of the uncertainty
//>          interval of the corresponding eigenvalue in W
//> \endverbatim
//>
//> \param[in,out] WGAP
//> \verbatim
//>          WGAP is DOUBLE PRECISION array, dimension (N)
//>          The separation from the right neighbor eigenvalue in W.
//> \endverbatim
//>
//> \param[in] IBLOCK
//> \verbatim
//>          IBLOCK is INTEGER array, dimension (N)
//>          The indices of the blocks (submatrices) associated with the
//>          corresponding eigenvalues in W; IBLOCK(i)=1 if eigenvalue
//>          W(i) belongs to the first block from the top, =2 if W(i)
//>          belongs to the second block, etc.
//> \endverbatim
//>
//> \param[in] INDEXW
//> \verbatim
//>          INDEXW is INTEGER array, dimension (N)
//>          The indices of the eigenvalues within each block (submatrix);
//>          for example, INDEXW(i)= 10 and IBLOCK(i)=2 imply that the
//>          i-th eigenvalue W(i) is the 10-th eigenvalue in the second block.
//> \endverbatim
//>
//> \param[in] GERS
//> \verbatim
//>          GERS is DOUBLE PRECISION array, dimension (2*N)
//>          The N Gerschgorin intervals (the i-th Gerschgorin interval
//>          is (GERS(2*i-1), GERS(2*i)). The Gerschgorin intervals should
//>          be computed from the original UNshifted matrix.
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ, max(1,M) )
//>          If INFO = 0, the first M columns of Z contain the
//>          orthonormal eigenvectors of the matrix T
//>          corresponding to the input eigenvalues, with the i-th
//>          column of Z holding the eigenvector associated with W(i).
//>          Note: the user must ensure that at least max(1,M) columns are
//>          supplied in the array Z.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>          The leading dimension of the array Z.  LDZ >= 1, and if
//>          JOBZ = 'V', LDZ >= max(1,N).
//> \endverbatim
//>
//> \param[out] ISUPPZ
//> \verbatim
//>          ISUPPZ is INTEGER array, dimension ( 2*max(1,M) )
//>          The support of the eigenvectors in Z, i.e., the indices
//>          indicating the nonzero elements in Z. The I-th eigenvector
//>          is nonzero only in elements ISUPPZ( 2*I-1 ) through
//>          ISUPPZ( 2*I ).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (12*N)
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (7*N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>
//>          > 0:  A problem occurred in DLARRV.
//>          < 0:  One of the called subroutines signaled an internal problem.
//>                Needs inspection of the corresponding parameter IINFO
//>                for further information.
//>
//>          =-1:  Problem in DLARRB when refining a child's eigenvalues.
//>          =-2:  Problem in DLARRF when computing the RRR of a child.
//>                When a child is inside a tight cluster, it can be difficult
//>                to find an RRR. A partial remedy from the user's point of
//>                view is to make the parameter MINRGP smaller and recompile.
//>                However, as the orthogonality of the computed vectors is
//>                proportional to 1/MINRGP, the user should be aware that
//>                he might be trading in precision when he decreases MINRGP.
//>          =-3:  Problem in DLARRB when refining a single eigenvalue
//>                after the Rayleigh correction was rejected.
//>          = 5:  The Rayleigh Quotient Iteration failed to converge to
//>                full accuracy in MAXITR steps.
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
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dlarrv_(int *n, double *vl, double *vu, double *d__, 
	double *l, double *pivmin, int *isplit, int *m, int *dol, int *dou, 
	double *minrgp, double *rtol1, double *rtol2, double *w, double *werr,
	 double *wgap, int *iblock, int *indexw, double *gers, double *z__, 
	int *ldz, int *isuppz, double *work, int *iwork, int *info)
{
    // Table of constant values
    double c_b5 = 0.;
    int c__1 = 1;
    int c__2 = 2;

    // System generated locals
    int z_dim1, z_offset, i__1, i__2, i__3, i__4, i__5;
    double d__1, d__2;
    int L__1;

    // Builtin functions
    double log(double);

    // Local variables
    int minwsize, i__, j, k, p, q, miniwsize, ii;
    double gl;
    int im, in;
    double gu, gap, eps, tau, tol, tmp;
    int zto;
    double ztz;
    int iend, jblk;
    double lgap;
    int done;
    double rgap, left;
    int wend, iter;
    double bstw;
    int itmp1;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    int indld;
    double fudge;
    int idone;
    double sigma;
    int iinfo, iindr;
    double resid;
    int eskip;
    double right;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    );
    int nclus, zfrom;
    double rqtol;
    int iindc1, iindc2;
    extern /* Subroutine */ int dlar1v_(int *, int *, int *, double *, double 
	    *, double *, double *, double *, double *, double *, double *, 
	    int *, int *, double *, double *, int *, int *, double *, double *
	    , double *, double *);
    int stp2ii;
    double lambda;
    extern double dlamch_(char *);
    int ibegin, indeig;
    int needbs;
    int indlld;
    double sgndef, mingma;
    extern /* Subroutine */ int dlarrb_(int *, double *, double *, int *, int 
	    *, double *, double *, int *, double *, double *, double *, 
	    double *, int *, double *, double *, int *, int *);
    int oldien, oldncl, wbegin;
    double spdiam;
    int negcnt;
    extern /* Subroutine */ int dlarrf_(int *, double *, double *, double *, 
	    int *, int *, double *, double *, double *, double *, double *, 
	    double *, double *, double *, double *, double *, double *, int *)
	    ;
    int oldcls;
    double savgap;
    int ndepth;
    double ssigma;
    extern /* Subroutine */ int dlaset_(char *, int *, int *, double *, 
	    double *, double *, int *);
    int usedbs;
    int iindwk, offset;
    double gaptol;
    int newcls, oldfst, indwrk, windex, oldlst;
    int usedrq;
    int newfst, newftt, parity, windmn, windpl, isupmn, newlst, zusedl;
    double bstres;
    int newsiz, zusedu, zusedw;
    double nrminv, rqcorr;
    int tryrqc;
    int isupmx;

    //
    // -- LAPACK auxiliary routine (version 3.8.0) --
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
    //    ..
    // Parameter adjustments
    --d__;
    --l;
    --isplit;
    --w;
    --werr;
    --wgap;
    --iblock;
    --indexw;
    --gers;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --isuppz;
    --work;
    --iwork;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    //
    //    The first N entries of WORK are reserved for the eigenvalues
    indld = *n + 1;
    indlld = (*n << 1) + 1;
    indwrk = *n * 3 + 1;
    minwsize = *n * 12;
    i__1 = minwsize;
    for (i__ = 1; i__ <= i__1; ++i__) {
	work[i__] = 0.;
// L5:
    }
    //    IWORK(IINDR+1:IINDR+N) hold the twist indices R for the
    //    factorization used to compute the FP vector
    iindr = 0;
    //    IWORK(IINDC1+1:IINC2+N) are used to store the clusters of the current
    //    layer and the one above.
    iindc1 = *n;
    iindc2 = *n << 1;
    iindwk = *n * 3 + 1;
    miniwsize = *n * 7;
    i__1 = miniwsize;
    for (i__ = 1; i__ <= i__1; ++i__) {
	iwork[i__] = 0;
// L10:
    }
    zusedl = 1;
    if (*dol > 1) {
	//       Set lower bound for use of Z
	zusedl = *dol - 1;
    }
    zusedu = *m;
    if (*dou < *m) {
	//       Set lower bound for use of Z
	zusedu = *dou + 1;
    }
    //    The width of the part of Z that is used
    zusedw = zusedu - zusedl + 1;
    dlaset_("Full", n, &zusedw, &c_b5, &c_b5, &z__[zusedl * z_dim1 + 1], ldz);
    eps = dlamch_("Precision");
    rqtol = eps * 2.;
    //
    //    Set expert flags for standard code.
    tryrqc = TRUE_;
    if (*dol == 1 && *dou == *m) {
    } else {
	//       Only selected eigenpairs are computed. Since the other evalues
	//       are not refined by RQ iteration, bisection has to compute to full
	//       accuracy.
	*rtol1 = eps * 4.;
	*rtol2 = eps * 4.;
    }
    //    The entries WBEGIN:WEND in W, WERR, WGAP correspond to the
    //    desired eigenvalues. The support of the nonzero eigenvector
    //    entries is contained in the interval IBEGIN:IEND.
    //    Remark that if k eigenpairs are desired, then the eigenvectors
    //    are stored in k contiguous columns of Z.
    //    DONE is the number of eigenvectors already computed
    done = 0;
    ibegin = 1;
    wbegin = 1;
    i__1 = iblock[*m];
    for (jblk = 1; jblk <= i__1; ++jblk) {
	iend = isplit[jblk];
	sigma = l[iend];
	//       Find the eigenvectors of the submatrix indexed IBEGIN
	//       through IEND.
	wend = wbegin - 1;
L15:
	if (wend < *m) {
	    if (iblock[wend + 1] == jblk) {
		++wend;
		goto L15;
	    }
	}
	if (wend < wbegin) {
	    ibegin = iend + 1;
	    goto L170;
	} else if (wend < *dol || wbegin > *dou) {
	    ibegin = iend + 1;
	    wbegin = wend + 1;
	    goto L170;
	}
	//       Find local spectral diameter of the block
	gl = gers[(ibegin << 1) - 1];
	gu = gers[ibegin * 2];
	i__2 = iend;
	for (i__ = ibegin + 1; i__ <= i__2; ++i__) {
	    // Computing MIN
	    d__1 = gers[(i__ << 1) - 1];
	    gl = min(d__1,gl);
	    // Computing MAX
	    d__1 = gers[i__ * 2];
	    gu = max(d__1,gu);
// L20:
	}
	spdiam = gu - gl;
	//       OLDIEN is the last index of the previous block
	oldien = ibegin - 1;
	//       Calculate the size of the current block
	in = iend - ibegin + 1;
	//       The number of eigenvalues in the current block
	im = wend - wbegin + 1;
	//       This is for a 1x1 block
	if (ibegin == iend) {
	    ++done;
	    z__[ibegin + wbegin * z_dim1] = 1.;
	    isuppz[(wbegin << 1) - 1] = ibegin;
	    isuppz[wbegin * 2] = ibegin;
	    w[wbegin] += sigma;
	    work[wbegin] = w[wbegin];
	    ibegin = iend + 1;
	    ++wbegin;
	    goto L170;
	}
	//       The desired (shifted) eigenvalues are stored in W(WBEGIN:WEND)
	//       Note that these can be approximations, in this case, the corresp.
	//       entries of WERR give the size of the uncertainty interval.
	//       The eigenvalue approximations will be refined when necessary as
	//       high relative accuracy is required for the computation of the
	//       corresponding eigenvectors.
	dcopy_(&im, &w[wbegin], &c__1, &work[wbegin], &c__1);
	//       We store in W the eigenvalue approximations w.r.t. the original
	//       matrix T.
	i__2 = im;
	for (i__ = 1; i__ <= i__2; ++i__) {
	    w[wbegin + i__ - 1] += sigma;
// L30:
	}
	//       NDEPTH is the current depth of the representation tree
	ndepth = 0;
	//       PARITY is either 1 or 0
	parity = 1;
	//       NCLUS is the number of clusters for the next level of the
	//       representation tree, we start with NCLUS = 1 for the root
	nclus = 1;
	iwork[iindc1 + 1] = 1;
	iwork[iindc1 + 2] = im;
	//       IDONE is the number of eigenvectors already computed in the current
	//       block
	idone = 0;
	//       loop while( IDONE.LT.IM )
	//       generate the representation tree for the current block and
	//       compute the eigenvectors
L40:
	if (idone < im) {
	    //          This is a crude protection against infinitely deep trees
	    if (ndepth > *m) {
		*info = -2;
		return 0;
	    }
	    //          breadth first processing of the current level of the representation
	    //          tree: OLDNCL = number of clusters on current level
	    oldncl = nclus;
	    //          reset NCLUS to count the number of child clusters
	    nclus = 0;
	    parity = 1 - parity;
	    if (parity == 0) {
		oldcls = iindc1;
		newcls = iindc2;
	    } else {
		oldcls = iindc2;
		newcls = iindc1;
	    }
	    //          Process the clusters on the current level
	    i__2 = oldncl;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		j = oldcls + (i__ << 1);
		//             OLDFST, OLDLST = first, last index of current cluster.
		//                              cluster indices start with 1 and are relative
		//                              to WBEGIN when accessing W, WGAP, WERR, Z
		oldfst = iwork[j - 1];
		oldlst = iwork[j];
		if (ndepth > 0) {
		    //                Retrieve relatively robust representation (RRR) of cluster
		    //                that has been computed at the previous level
		    //                The RRR is stored in Z and overwritten once the eigenvectors
		    //                have been computed or when the cluster is refined
		    if (*dol == 1 && *dou == *m) {
			//                   Get representation from location of the leftmost evalue
			//                   of the cluster
			j = wbegin + oldfst - 1;
		    } else {
			if (wbegin + oldfst - 1 < *dol) {
			    //                      Get representation from the left end of Z array
			    j = *dol - 1;
			} else if (wbegin + oldfst - 1 > *dou) {
			    //                      Get representation from the right end of Z array
			    j = *dou;
			} else {
			    j = wbegin + oldfst - 1;
			}
		    }
		    dcopy_(&in, &z__[ibegin + j * z_dim1], &c__1, &d__[ibegin]
			    , &c__1);
		    i__3 = in - 1;
		    dcopy_(&i__3, &z__[ibegin + (j + 1) * z_dim1], &c__1, &l[
			    ibegin], &c__1);
		    sigma = z__[iend + (j + 1) * z_dim1];
		    //                Set the corresponding entries in Z to zero
		    dlaset_("Full", &in, &c__2, &c_b5, &c_b5, &z__[ibegin + j 
			    * z_dim1], ldz);
		}
		//             Compute DL and DLL of current RRR
		i__3 = iend - 1;
		for (j = ibegin; j <= i__3; ++j) {
		    tmp = d__[j] * l[j];
		    work[indld - 1 + j] = tmp;
		    work[indlld - 1 + j] = tmp * l[j];
// L50:
		}
		if (ndepth > 0) {
		    //                P and Q are index of the first and last eigenvalue to compute
		    //                within the current block
		    p = indexw[wbegin - 1 + oldfst];
		    q = indexw[wbegin - 1 + oldlst];
		    //                Offset for the arrays WORK, WGAP and WERR, i.e., the P-OFFSET
		    //                through the Q-OFFSET elements of these arrays are to be used.
		    //                 OFFSET = P-OLDFST
		    offset = indexw[wbegin] - 1;
		    //                perform limited bisection (if necessary) to get approximate
		    //                eigenvalues to the precision needed.
		    dlarrb_(&in, &d__[ibegin], &work[indlld + ibegin - 1], &p,
			     &q, rtol1, rtol2, &offset, &work[wbegin], &wgap[
			    wbegin], &werr[wbegin], &work[indwrk], &iwork[
			    iindwk], pivmin, &spdiam, &in, &iinfo);
		    if (iinfo != 0) {
			*info = -1;
			return 0;
		    }
		    //                We also recompute the extremal gaps. W holds all eigenvalues
		    //                of the unshifted matrix and must be used for computation
		    //                of WGAP, the entries of WORK might stem from RRRs with
		    //                different shifts. The gaps from WBEGIN-1+OLDFST to
		    //                WBEGIN-1+OLDLST are correctly computed in DLARRB.
		    //                However, we only allow the gaps to become greater since
		    //                this is what should happen when we decrease WERR
		    if (oldfst > 1) {
			// Computing MAX
			d__1 = wgap[wbegin + oldfst - 2], d__2 = w[wbegin + 
				oldfst - 1] - werr[wbegin + oldfst - 1] - w[
				wbegin + oldfst - 2] - werr[wbegin + oldfst - 
				2];
			wgap[wbegin + oldfst - 2] = max(d__1,d__2);
		    }
		    if (wbegin + oldlst - 1 < wend) {
			// Computing MAX
			d__1 = wgap[wbegin + oldlst - 1], d__2 = w[wbegin + 
				oldlst] - werr[wbegin + oldlst] - w[wbegin + 
				oldlst - 1] - werr[wbegin + oldlst - 1];
			wgap[wbegin + oldlst - 1] = max(d__1,d__2);
		    }
		    //                Each time the eigenvalues in WORK get refined, we store
		    //                the newly found approximation with all shifts applied in W
		    i__3 = oldlst;
		    for (j = oldfst; j <= i__3; ++j) {
			w[wbegin + j - 1] = work[wbegin + j - 1] + sigma;
// L53:
		    }
		}
		//             Process the current node.
		newfst = oldfst;
		i__3 = oldlst;
		for (j = oldfst; j <= i__3; ++j) {
		    if (j == oldlst) {
			//                   we are at the right end of the cluster, this is also the
			//                   boundary of the child cluster
			newlst = j;
		    } else if (wgap[wbegin + j - 1] >= *minrgp * (d__1 = work[
			    wbegin + j - 1], abs(d__1))) {
			//                   the right relative gap is big enough, the child cluster
			//                   (NEWFST,..,NEWLST) is well separated from the following
			newlst = j;
		    } else {
			//                   inside a child cluster, the relative gap is not
			//                   big enough.
			goto L140;
		    }
		    //                Compute size of child cluster found
		    newsiz = newlst - newfst + 1;
		    //                NEWFTT is the place in Z where the new RRR or the computed
		    //                eigenvector is to be stored
		    if (*dol == 1 && *dou == *m) {
			//                   Store representation at location of the leftmost evalue
			//                   of the cluster
			newftt = wbegin + newfst - 1;
		    } else {
			if (wbegin + newfst - 1 < *dol) {
			    //                      Store representation at the left end of Z array
			    newftt = *dol - 1;
			} else if (wbegin + newfst - 1 > *dou) {
			    //                      Store representation at the right end of Z array
			    newftt = *dou;
			} else {
			    newftt = wbegin + newfst - 1;
			}
		    }
		    if (newsiz > 1) {
			//
			//                   Current child is not a singleton but a cluster.
			//                   Compute and store new representation of child.
			//
			//
			//                   Compute left and right cluster gap.
			//
			//                   LGAP and RGAP are not computed from WORK because
			//                   the eigenvalue approximations may stem from RRRs
			//                   different shifts. However, W hold all eigenvalues
			//                   of the unshifted matrix. Still, the entries in WGAP
			//                   have to be computed from WORK since the entries
			//                   in W might be of the same order so that gaps are not
			//                   exhibited correctly for very close eigenvalues.
			if (newfst == 1) {
			    // Computing MAX
			    d__1 = 0., d__2 = w[wbegin] - werr[wbegin] - *vl;
			    lgap = max(d__1,d__2);
			} else {
			    lgap = wgap[wbegin + newfst - 2];
			}
			rgap = wgap[wbegin + newlst - 1];
			//
			//                   Compute left- and rightmost eigenvalue of child
			//                   to high precision in order to shift as close
			//                   as possible and obtain as large relative gaps
			//                   as possible
			//
			for (k = 1; k <= 2; ++k) {
			    if (k == 1) {
				p = indexw[wbegin - 1 + newfst];
			    } else {
				p = indexw[wbegin - 1 + newlst];
			    }
			    offset = indexw[wbegin] - 1;
			    dlarrb_(&in, &d__[ibegin], &work[indlld + ibegin 
				    - 1], &p, &p, &rqtol, &rqtol, &offset, &
				    work[wbegin], &wgap[wbegin], &werr[wbegin]
				    , &work[indwrk], &iwork[iindwk], pivmin, &
				    spdiam, &in, &iinfo);
// L55:
			}
			if (wbegin + newlst - 1 < *dol || wbegin + newfst - 1 
				> *dou) {
			    //                      if the cluster contains no desired eigenvalues
			    //                      skip the computation of that branch of the rep. tree
			    //
			    //                      We could skip before the refinement of the extremal
			    //                      eigenvalues of the child, but then the representation
			    //                      tree could be different from the one when nothing is
			    //                      skipped. For this reason we skip at this place.
			    idone = idone + newlst - newfst + 1;
			    goto L139;
			}
			//
			//                   Compute RRR of child cluster.
			//                   Note that the new RRR is stored in Z
			//
			//                   DLARRF needs LWORK = 2*N
			dlarrf_(&in, &d__[ibegin], &l[ibegin], &work[indld + 
				ibegin - 1], &newfst, &newlst, &work[wbegin], 
				&wgap[wbegin], &werr[wbegin], &spdiam, &lgap, 
				&rgap, pivmin, &tau, &z__[ibegin + newftt * 
				z_dim1], &z__[ibegin + (newftt + 1) * z_dim1],
				 &work[indwrk], &iinfo);
			if (iinfo == 0) {
			    //                      a new RRR for the cluster was found by DLARRF
			    //                      update shift and store it
			    ssigma = sigma + tau;
			    z__[iend + (newftt + 1) * z_dim1] = ssigma;
			    //                      WORK() are the midpoints and WERR() the semi-width
			    //                      Note that the entries in W are unchanged.
			    i__4 = newlst;
			    for (k = newfst; k <= i__4; ++k) {
				fudge = eps * 3. * (d__1 = work[wbegin + k - 
					1], abs(d__1));
				work[wbegin + k - 1] -= tau;
				fudge += eps * 4. * (d__1 = work[wbegin + k - 
					1], abs(d__1));
				//                         Fudge errors
				werr[wbegin + k - 1] += fudge;
				//                         Gaps are not fudged. Provided that WERR is small
				//                         when eigenvalues are close, a zero gap indicates
				//                         that a new representation is needed for resolving
				//                         the cluster. A fudge could lead to a wrong decision
				//                         of judging eigenvalues 'separated' which in
				//                         reality are not. This could have a negative impact
				//                         on the orthogonality of the computed eigenvectors.
// L116:
			    }
			    ++nclus;
			    k = newcls + (nclus << 1);
			    iwork[k - 1] = newfst;
			    iwork[k] = newlst;
			} else {
			    *info = -2;
			    return 0;
			}
		    } else {
			//
			//                   Compute eigenvector of singleton
			//
			iter = 0;
			tol = log((double) in) * 4. * eps;
			k = newfst;
			windex = wbegin + k - 1;
			// Computing MAX
			i__4 = windex - 1;
			windmn = max(i__4,1);
			// Computing MIN
			i__4 = windex + 1;
			windpl = min(i__4,*m);
			lambda = work[windex];
			++done;
			//                   Check if eigenvector computation is to be skipped
			if (windex < *dol || windex > *dou) {
			    eskip = TRUE_;
			    goto L125;
			} else {
			    eskip = FALSE_;
			}
			left = work[windex] - werr[windex];
			right = work[windex] + werr[windex];
			indeig = indexw[windex];
			//                   Note that since we compute the eigenpairs for a child,
			//                   all eigenvalue approximations are w.r.t the same shift.
			//                   In this case, the entries in WORK should be used for
			//                   computing the gaps since they exhibit even very small
			//                   differences in the eigenvalues, as opposed to the
			//                   entries in W which might "look" the same.
			if (k == 1) {
			    //                      In the case RANGE='I' and with not much initial
			    //                      accuracy in LAMBDA and VL, the formula
			    //                      LGAP = MAX( ZERO, (SIGMA - VL) + LAMBDA )
			    //                      can lead to an overestimation of the left gap and
			    //                      thus to inadequately early RQI 'convergence'.
			    //                      Prevent this by forcing a small left gap.
			    // Computing MAX
			    d__1 = abs(left), d__2 = abs(right);
			    lgap = eps * max(d__1,d__2);
			} else {
			    lgap = wgap[windmn];
			}
			if (k == im) {
			    //                      In the case RANGE='I' and with not much initial
			    //                      accuracy in LAMBDA and VU, the formula
			    //                      can lead to an overestimation of the right gap and
			    //                      thus to inadequately early RQI 'convergence'.
			    //                      Prevent this by forcing a small right gap.
			    // Computing MAX
			    d__1 = abs(left), d__2 = abs(right);
			    rgap = eps * max(d__1,d__2);
			} else {
			    rgap = wgap[windex];
			}
			gap = min(lgap,rgap);
			if (k == 1 || k == im) {
			    //                      The eigenvector support can become wrong
			    //                      because significant entries could be cut off due to a
			    //                      large GAPTOL parameter in LAR1V. Prevent this.
			    gaptol = 0.;
			} else {
			    gaptol = gap * eps;
			}
			isupmn = in;
			isupmx = 1;
			//                   Update WGAP so that it holds the minimum gap
			//                   to the left or the right. This is crucial in the
			//                   case where bisection is used to ensure that the
			//                   eigenvalue is refined up to the required precision.
			//                   The correct value is restored afterwards.
			savgap = wgap[windex];
			wgap[windex] = gap;
			//                   We want to use the Rayleigh Quotient Correction
			//                   as often as possible since it converges quadratically
			//                   when we are close enough to the desired eigenvalue.
			//                   However, the Rayleigh Quotient can have the wrong sign
			//                   and lead us away from the desired eigenvalue. In this
			//                   case, the best we can do is to use bisection.
			usedbs = FALSE_;
			usedrq = FALSE_;
			//                   Bisection is initially turned off unless it is forced
			needbs = ! tryrqc;
L120:
			//                   Check if bisection should be used to refine eigenvalue
			if (needbs) {
			    //                      Take the bisection as new iterate
			    usedbs = TRUE_;
			    itmp1 = iwork[iindr + windex];
			    offset = indexw[wbegin] - 1;
			    d__1 = eps * 2.;
			    dlarrb_(&in, &d__[ibegin], &work[indlld + ibegin 
				    - 1], &indeig, &indeig, &c_b5, &d__1, &
				    offset, &work[wbegin], &wgap[wbegin], &
				    werr[wbegin], &work[indwrk], &iwork[
				    iindwk], pivmin, &spdiam, &itmp1, &iinfo);
			    if (iinfo != 0) {
				*info = -3;
				return 0;
			    }
			    lambda = work[windex];
			    //                      Reset twist index from inaccurate LAMBDA to
			    //                      force computation of true MINGMA
			    iwork[iindr + windex] = 0;
			}
			//                   Given LAMBDA, compute the eigenvector.
			L__1 = ! usedbs;
			dlar1v_(&in, &c__1, &in, &lambda, &d__[ibegin], &l[
				ibegin], &work[indld + ibegin - 1], &work[
				indlld + ibegin - 1], pivmin, &gaptol, &z__[
				ibegin + windex * z_dim1], &L__1, &negcnt, &
				ztz, &mingma, &iwork[iindr + windex], &isuppz[
				(windex << 1) - 1], &nrminv, &resid, &rqcorr, 
				&work[indwrk]);
			if (iter == 0) {
			    bstres = resid;
			    bstw = lambda;
			} else if (resid < bstres) {
			    bstres = resid;
			    bstw = lambda;
			}
			// Computing MIN
			i__4 = isupmn, i__5 = isuppz[(windex << 1) - 1];
			isupmn = min(i__4,i__5);
			// Computing MAX
			i__4 = isupmx, i__5 = isuppz[windex * 2];
			isupmx = max(i__4,i__5);
			++iter;
			//                   sin alpha <= |resid|/gap
			//                   Note that both the residual and the gap are
			//                   proportional to the matrix, so ||T|| doesn't play
			//                   a role in the quotient
			//
			//                   Convergence test for Rayleigh-Quotient iteration
			//                   (omitted when Bisection has been used)
			//
			if (resid > tol * gap && abs(rqcorr) > rqtol * abs(
				lambda) && ! usedbs) {
			    //                      We need to check that the RQCORR update doesn't
			    //                      move the eigenvalue away from the desired one and
			    //                      towards a neighbor. -> protection with bisection
			    if (indeig <= negcnt) {
				//                         The wanted eigenvalue lies to the left
				sgndef = -1.;
			    } else {
				//                         The wanted eigenvalue lies to the right
				sgndef = 1.;
			    }
			    //                      We only use the RQCORR if it improves the
			    //                      the iterate reasonably.
			    if (rqcorr * sgndef >= 0. && lambda + rqcorr <= 
				    right && lambda + rqcorr >= left) {
				usedrq = TRUE_;
				//                         Store new midpoint of bisection interval in WORK
				if (sgndef == 1.) {
				    //                            The current LAMBDA is on the left of the true
				    //                            eigenvalue
				    left = lambda;
				    //                            We prefer to assume that the error estimate
				    //                            is correct. We could make the interval not
				    //                            as a bracket but to be modified if the RQCORR
				    //                            chooses to. In this case, the RIGHT side should
				    //                            be modified as follows:
				    //                             RIGHT = MAX(RIGHT, LAMBDA + RQCORR)
				} else {
				    //                            The current LAMBDA is on the right of the true
				    //                            eigenvalue
				    right = lambda;
				    //                            See comment about assuming the error estimate is
				    //                            correct above.
				    //                             LEFT = MIN(LEFT, LAMBDA + RQCORR)
				}
				work[windex] = (right + left) * .5;
				//                         Take RQCORR since it has the correct sign and
				//                         improves the iterate reasonably
				lambda += rqcorr;
				//                         Update width of error interval
				werr[windex] = (right - left) * .5;
			    } else {
				needbs = TRUE_;
			    }
			    if (right - left < rqtol * abs(lambda)) {
				//                            The eigenvalue is computed to bisection accuracy
				//                            compute eigenvector and stop
				usedbs = TRUE_;
				goto L120;
			    } else if (iter < 10) {
				goto L120;
			    } else if (iter == 10) {
				needbs = TRUE_;
				goto L120;
			    } else {
				*info = 5;
				return 0;
			    }
			} else {
			    stp2ii = FALSE_;
			    if (usedrq && usedbs && bstres <= resid) {
				lambda = bstw;
				stp2ii = TRUE_;
			    }
			    if (stp2ii) {
				//                         improve error angle by second step
				L__1 = ! usedbs;
				dlar1v_(&in, &c__1, &in, &lambda, &d__[ibegin]
					, &l[ibegin], &work[indld + ibegin - 
					1], &work[indlld + ibegin - 1], 
					pivmin, &gaptol, &z__[ibegin + windex 
					* z_dim1], &L__1, &negcnt, &ztz, &
					mingma, &iwork[iindr + windex], &
					isuppz[(windex << 1) - 1], &nrminv, &
					resid, &rqcorr, &work[indwrk]);
			    }
			    work[windex] = lambda;
			}
			//
			//                   Compute FP-vector support w.r.t. whole matrix
			//
			isuppz[(windex << 1) - 1] += oldien;
			isuppz[windex * 2] += oldien;
			zfrom = isuppz[(windex << 1) - 1];
			zto = isuppz[windex * 2];
			isupmn += oldien;
			isupmx += oldien;
			//                   Ensure vector is ok if support in the RQI has changed
			if (isupmn < zfrom) {
			    i__4 = zfrom - 1;
			    for (ii = isupmn; ii <= i__4; ++ii) {
				z__[ii + windex * z_dim1] = 0.;
// L122:
			    }
			}
			if (isupmx > zto) {
			    i__4 = isupmx;
			    for (ii = zto + 1; ii <= i__4; ++ii) {
				z__[ii + windex * z_dim1] = 0.;
// L123:
			    }
			}
			i__4 = zto - zfrom + 1;
			dscal_(&i__4, &nrminv, &z__[zfrom + windex * z_dim1], 
				&c__1);
L125:
			//                   Update W
			w[windex] = lambda + sigma;
			//                   Recompute the gaps on the left and right
			//                   But only allow them to become larger and not
			//                   smaller (which can only happen through "bad"
			//                   cancellation and doesn't reflect the theory
			//                   where the initial gaps are underestimated due
			//                   to WERR being too crude.)
			if (! eskip) {
			    if (k > 1) {
				// Computing MAX
				d__1 = wgap[windmn], d__2 = w[windex] - werr[
					windex] - w[windmn] - werr[windmn];
				wgap[windmn] = max(d__1,d__2);
			    }
			    if (windex < wend) {
				// Computing MAX
				d__1 = savgap, d__2 = w[windpl] - werr[windpl]
					 - w[windex] - werr[windex];
				wgap[windex] = max(d__1,d__2);
			    }
			}
			++idone;
		    }
		    //                here ends the code for the current child
		    //
L139:
		    //                Proceed to any remaining child nodes
		    newfst = j + 1;
L140:
		    ;
		}
// L150:
	    }
	    ++ndepth;
	    goto L40;
	}
	ibegin = iend + 1;
	wbegin = wend + 1;
L170:
	;
    }
    return 0;
    //
    //    End of DLARRV
    //
} // dlarrv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLARUV returns a vector of n random real numbers from a uniform distribution.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLARUV + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaruv.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaruv.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaruv.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLARUV( ISEED, N, X )
//
//      .. Scalar Arguments ..
//      INTEGER            N
//      ..
//      .. Array Arguments ..
//      INTEGER            ISEED( 4 )
//      DOUBLE PRECISION   X( N )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLARUV returns a vector of n random real numbers from a uniform (0,1)
//> distribution (n <= 128).
//>
//> This is an auxiliary routine called by DLARNV and ZLARNV.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in,out] ISEED
//> \verbatim
//>          ISEED is INTEGER array, dimension (4)
//>          On entry, the seed of the random number generator; the array
//>          elements must be between 0 and 4095, and ISEED(4) must be
//>          odd.
//>          On exit, the seed is updated.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The number of random numbers to be generated. N <= 128.
//> \endverbatim
//>
//> \param[out] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension (N)
//>          The generated random numbers.
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
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  This routine uses a multiplicative congruential method with modulus
//>  2**48 and multiplier 33952834046453 (see G.S.Fishman,
//>  'Multiplicative congruential random number generators with modulus
//>  2**b: an exhaustive analysis for b = 32 and a partial analysis for
//>  b = 48', Math. Comp. 189, pp 331-344, 1990).
//>
//>  48-bit integers are stored in 4 integer array elements with 12 bits
//>  per element. Hence the routine is portable across machines with
//>  integers of 32 bits or more.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlaruv_(int *iseed, int *n, double *x)
{
    /* Initialized data */

    static int mm[512]	/* was [128][4] */ = { 494,2637,255,2008,1253,3344,
	    4084,1739,3143,3468,688,1657,1238,3166,1292,3422,1270,2016,154,
	    2862,697,1706,491,931,1444,444,3577,3944,2184,1661,3482,657,3023,
	    3618,1267,1828,164,3798,3087,2400,2870,3876,1905,1593,1797,1234,
	    3460,328,2861,1950,617,2070,3331,769,1558,2412,2800,189,287,2045,
	    1227,2838,209,2770,3654,3993,192,2253,3491,2889,2857,2094,1818,
	    688,1407,634,3231,815,3524,1914,516,164,303,2144,3480,119,3357,
	    837,2826,2332,2089,3780,1700,3712,150,2000,3375,1621,3090,3765,
	    1149,3146,33,3082,2741,359,3316,1749,185,2784,2202,2199,1364,1244,
	    2020,3160,2785,2772,1217,1822,1245,2252,3904,2774,997,2573,1148,
	    545,322,789,1440,752,2859,123,1848,643,2405,2638,2344,46,3814,913,
	    3649,339,3808,822,2832,3078,3633,2970,637,2249,2081,4019,1478,242,
	    481,2075,4058,622,3376,812,234,641,4005,1122,3135,2640,2302,40,
	    1832,2247,2034,2637,1287,1691,496,1597,2394,2584,1843,336,1472,
	    2407,433,2096,1761,2810,566,442,41,1238,1086,603,840,3168,1499,
	    1084,3438,2408,1589,2391,288,26,512,1456,171,1677,2657,2270,2587,
	    2961,1970,1817,676,1410,3723,2803,3185,184,663,499,3784,1631,1925,
	    3912,1398,1349,1441,2224,2411,1907,3192,2786,382,37,759,2948,1862,
	    3802,2423,2051,2295,1332,1832,2405,3638,3661,327,3660,716,1842,
	    3987,1368,1848,2366,2508,3754,1766,3572,2893,307,1297,3966,758,
	    2598,3406,2922,1038,2934,2091,2451,1580,1958,2055,1507,1078,3273,
	    17,854,2916,3971,2889,3831,2621,1541,893,736,3992,787,2125,2364,
	    2460,257,1574,3912,1216,3248,3401,2124,2762,149,2245,166,466,4018,
	    1399,190,2879,153,2320,18,712,2159,2318,2091,3443,1510,449,1956,
	    2201,3137,3399,1321,2271,3667,2703,629,2365,2431,1113,3922,2554,
	    184,2099,3228,4012,1921,3452,3901,572,3309,3171,817,3039,1696,
	    1256,3715,2077,3019,1497,1101,717,51,981,1978,1813,3881,76,3846,
	    3694,1682,124,1660,3997,479,1141,886,3514,1301,3604,1888,1836,
	    1990,2058,692,1194,20,3285,2046,2107,3508,3525,3801,2549,1145,
	    2253,305,3301,1065,3133,2913,3285,1241,1197,3729,2501,1673,541,
	    2753,949,2361,1165,4081,2725,3305,3069,3617,3733,409,2157,1361,
	    3973,1865,2525,1409,3445,3577,77,3761,2149,1449,3005,225,85,3673,
	    3117,3089,1349,2057,413,65,1845,697,3085,3441,1573,3689,2941,929,
	    533,2841,4077,721,2821,2249,2397,2817,245,1913,1997,3121,997,1833,
	    2877,1633,981,2009,941,2449,197,2441,285,1473,2741,3129,909,2801,
	    421,4073,2813,2337,1429,1177,1901,81,1669,2633,2269,129,1141,249,
	    3917,2481,3941,2217,2749,3041,1877,345,2861,1809,3141,2825,157,
	    2881,3637,1465,2829,2161,3365,361,2685,3745,2325,3609,3821,3537,
	    517,3017,2141,1537 };

    // System generated locals
    int i__1;

    // Local variables
    int i__, i1, i2, i3, i4, it1, it2, it3, it4;

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Data statements ..
    // Parameter adjustments
    --iseed;
    --x;

    // Function Body
    //    ..
    //    .. Executable Statements ..
    //
    i1 = iseed[1];
    i2 = iseed[2];
    i3 = iseed[3];
    i4 = iseed[4];
    i__1 = min(*n,128);
    for (i__ = 1; i__ <= i__1; ++i__) {
L20:
	//
	//       Multiply the seed by i-th power of the multiplier modulo 2**48
	//
	it4 = i4 * mm[i__ + 383];
	it3 = it4 / 4096;
	it4 -= it3 << 12;
	it3 = it3 + i3 * mm[i__ + 383] + i4 * mm[i__ + 255];
	it2 = it3 / 4096;
	it3 -= it2 << 12;
	it2 = it2 + i2 * mm[i__ + 383] + i3 * mm[i__ + 255] + i4 * mm[i__ + 
		127];
	it1 = it2 / 4096;
	it2 -= it1 << 12;
	it1 = it1 + i1 * mm[i__ + 383] + i2 * mm[i__ + 255] + i3 * mm[i__ + 
		127] + i4 * mm[i__ - 1];
	it1 %= 4096;
	//
	//       Convert 48-bit integer to a real number in the interval (0,1)
	//
	x[i__] = ((double) it1 + ((double) it2 + ((double) it3 + (double) it4 
		* 2.44140625e-4) * 2.44140625e-4) * 2.44140625e-4) * 
		2.44140625e-4;
	if (x[i__] == 1.) {
	    //          If a real number has n bits of precision, and the first
	    //          n bits of the 48-bit integer above happen to be all 1 (which
	    //          will occur about once every 2**n calls), then X( I ) will
	    //          be rounded to exactly 1.0.
	    //          Since X( I ) is not supposed to return exactly 0.0 or 1.0,
	    //          the statistically correct thing to do in this situation is
	    //          simply to iterate again.
	    //          N.B. the case X( I ) = 0.0 should not be possible.
	    i1 += 2;
	    i2 += 2;
	    i3 += 2;
	    i4 += 2;
	    goto L20;
	}
// L10:
    }
    //
    //    Return final value of seed
    //
    iseed[1] = it1;
    iseed[2] = it2;
    iseed[3] = it3;
    iseed[4] = it4;
    return 0;
    //
    //    End of DLARUV
    //
} // dlaruv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLATRD reduces the first nb rows and columns of a symmetric/Hermitian matrix A to real tridiagonal form by an orthogonal similarity transformation.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLATRD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlatrd.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlatrd.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlatrd.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLATRD( UPLO, N, NB, A, LDA, E, TAU, W, LDW )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            LDA, LDW, N, NB
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), E( * ), TAU( * ), W( LDW, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DLATRD reduces NB rows and columns of a real symmetric matrix A to
//> symmetric tridiagonal form by an orthogonal similarity
//> transformation Q**T * A * Q, and returns the matrices V and W which are
//> needed to apply the transformation to the unreduced part of A.
//>
//> If UPLO = 'U', DLATRD reduces the last NB rows and columns of a
//> matrix, of which the upper triangle is supplied;
//> if UPLO = 'L', DLATRD reduces the first NB rows and columns of a
//> matrix, of which the lower triangle is supplied.
//>
//> This is an auxiliary routine called by DSYTRD.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          Specifies whether the upper or lower triangular part of the
//>          symmetric matrix A is stored:
//>          = 'U': Upper triangular
//>          = 'L': Lower triangular
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix A.
//> \endverbatim
//>
//> \param[in] NB
//> \verbatim
//>          NB is INTEGER
//>          The number of rows and columns to be reduced.
//> \endverbatim
//>
//> \param[in,out] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,N)
//>          On entry, the symmetric matrix A.  If UPLO = 'U', the leading
//>          n-by-n upper triangular part of A contains the upper
//>          triangular part of the matrix A, and the strictly lower
//>          triangular part of A is not referenced.  If UPLO = 'L', the
//>          leading n-by-n lower triangular part of A contains the lower
//>          triangular part of the matrix A, and the strictly upper
//>          triangular part of A is not referenced.
//>          On exit:
//>          if UPLO = 'U', the last NB columns have been reduced to
//>            tridiagonal form, with the diagonal elements overwriting
//>            the diagonal elements of A; the elements above the diagonal
//>            with the array TAU, represent the orthogonal matrix Q as a
//>            product of elementary reflectors;
//>          if UPLO = 'L', the first NB columns have been reduced to
//>            tridiagonal form, with the diagonal elements overwriting
//>            the diagonal elements of A; the elements below the diagonal
//>            with the array TAU, represent the  orthogonal matrix Q as a
//>            product of elementary reflectors.
//>          See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= (1,N).
//> \endverbatim
//>
//> \param[out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal
//>          elements of the last NB columns of the reduced matrix;
//>          if UPLO = 'L', E(1:nb) contains the subdiagonal elements of
//>          the first NB columns of the reduced matrix.
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (N-1)
//>          The scalar factors of the elementary reflectors, stored in
//>          TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.
//>          See Further Details.
//> \endverbatim
//>
//> \param[out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (LDW,NB)
//>          The n-by-nb matrix W required to update the unreduced part
//>          of A.
//> \endverbatim
//>
//> \param[in] LDW
//> \verbatim
//>          LDW is INTEGER
//>          The leading dimension of the array W. LDW >= max(1,N).
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
//>  If UPLO = 'U', the matrix Q is represented as a product of elementary
//>  reflectors
//>
//>     Q = H(n) H(n-1) . . . H(n-nb+1).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),
//>  and tau in TAU(i-1).
//>
//>  If UPLO = 'L', the matrix Q is represented as a product of elementary
//>  reflectors
//>
//>     Q = H(1) H(2) . . . H(nb).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),
//>  and tau in TAU(i).
//>
//>  The elements of the vectors v together form the n-by-nb matrix V
//>  which is needed, with W, to apply the transformation to the unreduced
//>  part of the matrix, using a symmetric rank-2k update of the form:
//>  A := A - V*W**T - W*V**T.
//>
//>  The contents of A on exit are illustrated by the following examples
//>  with n = 5 and nb = 2:
//>
//>  if UPLO = 'U':                       if UPLO = 'L':
//>
//>    (  a   a   a   v4  v5 )              (  d                  )
//>    (      a   a   v4  v5 )              (  1   d              )
//>    (          a   1   v5 )              (  v1  1   a          )
//>    (              d   1  )              (  v1  v2  a   a      )
//>    (                  d  )              (  v1  v2  a   a   a  )
//>
//>  where d denotes a diagonal element of the reduced matrix, a denotes
//>  an element of the original matrix that is unchanged, and vi denotes
//>  an element of the vector defining H(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dlatrd_(char *uplo, int *n, int *nb, double *a, int *lda,
	 double *e, double *tau, double *w, int *ldw)
{
    // Table of constant values
    double c_b5 = -1.;
    double c_b6 = 1.;
    int c__1 = 1;
    double c_b16 = 0.;

    // System generated locals
    int a_dim1, a_offset, w_dim1, w_offset, i__1, i__2, i__3;

    // Local variables
    int i__, iw;
    extern double ddot_(int *, double *, int *, double *, int *);
    double alpha;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int dgemv_(char *, int *, int *, double *, double 
	    *, int *, double *, int *, double *, double *, int *), daxpy_(int 
	    *, double *, double *, int *, double *, int *), dsymv_(char *, 
	    int *, double *, double *, int *, double *, int *, double *, 
	    double *, int *), dlarfg_(int *, double *, double *, int *, 
	    double *);

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
    //    .. Intrinsic Functions ..
    //    ..
    //    .. Executable Statements ..
    //
    //    Quick return if possible
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --e;
    --tau;
    w_dim1 = *ldw;
    w_offset = 1 + w_dim1;
    w -= w_offset;

    // Function Body
    if (*n <= 0) {
	return 0;
    }
    if (lsame_(uplo, "U")) {
	//
	//       Reduce last NB columns of upper triangle
	//
	i__1 = *n - *nb + 1;
	for (i__ = *n; i__ >= i__1; --i__) {
	    iw = i__ - *n + *nb;
	    if (i__ < *n) {
		//
		//             Update A(1:i,i)
		//
		i__2 = *n - i__;
		dgemv_("No transpose", &i__, &i__2, &c_b5, &a[(i__ + 1) * 
			a_dim1 + 1], lda, &w[i__ + (iw + 1) * w_dim1], ldw, &
			c_b6, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = *n - i__;
		dgemv_("No transpose", &i__, &i__2, &c_b5, &w[(iw + 1) * 
			w_dim1 + 1], ldw, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b6, &a[i__ * a_dim1 + 1], &c__1);
	    }
	    if (i__ > 1) {
		//
		//             Generate elementary reflector H(i) to annihilate
		//             A(1:i-2,i)
		//
		i__2 = i__ - 1;
		dlarfg_(&i__2, &a[i__ - 1 + i__ * a_dim1], &a[i__ * a_dim1 + 
			1], &c__1, &tau[i__ - 1]);
		e[i__ - 1] = a[i__ - 1 + i__ * a_dim1];
		a[i__ - 1 + i__ * a_dim1] = 1.;
		//
		//             Compute W(1:i-1,i)
		//
		i__2 = i__ - 1;
		dsymv_("Upper", &i__2, &c_b6, &a[a_offset], lda, &a[i__ * 
			a_dim1 + 1], &c__1, &c_b16, &w[iw * w_dim1 + 1], &
			c__1);
		if (i__ < *n) {
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("Transpose", &i__2, &i__3, &c_b6, &w[(iw + 1) * 
			    w_dim1 + 1], ldw, &a[i__ * a_dim1 + 1], &c__1, &
			    c_b16, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[(i__ + 1) *
			     a_dim1 + 1], lda, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b6, &w[iw * w_dim1 + 1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("Transpose", &i__2, &i__3, &c_b6, &a[(i__ + 1) * 
			    a_dim1 + 1], lda, &a[i__ * a_dim1 + 1], &c__1, &
			    c_b16, &w[i__ + 1 + iw * w_dim1], &c__1);
		    i__2 = i__ - 1;
		    i__3 = *n - i__;
		    dgemv_("No transpose", &i__2, &i__3, &c_b5, &w[(iw + 1) * 
			    w_dim1 + 1], ldw, &w[i__ + 1 + iw * w_dim1], &
			    c__1, &c_b6, &w[iw * w_dim1 + 1], &c__1);
		}
		i__2 = i__ - 1;
		dscal_(&i__2, &tau[i__ - 1], &w[iw * w_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		alpha = tau[i__ - 1] * -.5 * ddot_(&i__2, &w[iw * w_dim1 + 1],
			 &c__1, &a[i__ * a_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		daxpy_(&i__2, &alpha, &a[i__ * a_dim1 + 1], &c__1, &w[iw * 
			w_dim1 + 1], &c__1);
	    }
// L10:
	}
    } else {
	//
	//       Reduce first NB columns of lower triangle
	//
	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    //
	    //          Update A(i:n,i)
	    //
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ + a_dim1], lda,
		     &w[i__ + w_dim1], ldw, &c_b6, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b5, &w[i__ + w_dim1], ldw,
		     &a[i__ + a_dim1], lda, &c_b6, &a[i__ + i__ * a_dim1], &
		    c__1);
	    if (i__ < *n) {
		//
		//             Generate elementary reflector H(i) to annihilate
		//             A(i+2:n,i)
		//
		i__2 = *n - i__;
		// Computing MIN
		i__3 = i__ + 2;
		dlarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*n) + 
			i__ * a_dim1], &c__1, &tau[i__]);
		e[i__] = a[i__ + 1 + i__ * a_dim1];
		a[i__ + 1 + i__ * a_dim1] = 1.;
		//
		//             Compute W(i+1:n,i)
		//
		i__2 = *n - i__;
		dsymv_("Lower", &i__2, &c_b6, &a[i__ + 1 + (i__ + 1) * a_dim1]
			, lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b6, &w[i__ + 1 + w_dim1],
			 ldw, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, &w[
			i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + 
			a_dim1], lda, &w[i__ * w_dim1 + 1], &c__1, &c_b6, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b6, &a[i__ + 1 + a_dim1],
			 lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, &w[
			i__ * w_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &w[i__ + 1 + 
			w_dim1], ldw, &w[i__ * w_dim1 + 1], &c__1, &c_b6, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		dscal_(&i__2, &tau[i__], &w[i__ + 1 + i__ * w_dim1], &c__1);
		i__2 = *n - i__;
		alpha = tau[i__] * -.5 * ddot_(&i__2, &w[i__ + 1 + i__ * 
			w_dim1], &c__1, &a[i__ + 1 + i__ * a_dim1], &c__1);
		i__2 = *n - i__;
		daxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &w[
			i__ + 1 + i__ * w_dim1], &c__1);
	    }
// L20:
	}
    }
    return 0;
    //
    //    End of DLATRD
    //
} // dlatrd_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORM2L multiplies a general matrix by the orthogonal matrix from a QL factorization determined by sgeqlf (unblocked algorithm).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORM2L + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dorm2l.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dorm2l.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dorm2l.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORM2L( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, M, N
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
//> DORM2L overwrites the general real m by n matrix C with
//>
//>       Q * C  if SIDE = 'L' and TRANS = 'N', or
//>
//>       Q**T * C  if SIDE = 'L' and TRANS = 'T', or
//>
//>       C * Q  if SIDE = 'R' and TRANS = 'N', or
//>
//>       C * Q**T if SIDE = 'R' and TRANS = 'T',
//>
//> where Q is a real orthogonal matrix defined as the product of k
//> elementary reflectors
//>
//>       Q = H(k) . . . H(2) H(1)
//>
//> as returned by DGEQLF. Q is of order m if SIDE = 'L' and of order n
//> if SIDE = 'R'.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIDE
//> \verbatim
//>          SIDE is CHARACTER*1
//>          = 'L': apply Q or Q**T from the Left
//>          = 'R': apply Q or Q**T from the Right
//> \endverbatim
//>
//> \param[in] TRANS
//> \verbatim
//>          TRANS is CHARACTER*1
//>          = 'N': apply Q  (No transpose)
//>          = 'T': apply Q**T (Transpose)
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
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines
//>          the matrix Q.
//>          If SIDE = 'L', M >= K >= 0;
//>          if SIDE = 'R', N >= K >= 0.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,K)
//>          The i-th column must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          DGEQLF in the last k columns of its array argument A.
//>          A is modified by the routine but restored on exit.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//>          If SIDE = 'L', LDA >= max(1,M);
//>          if SIDE = 'R', LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGEQLF.
//> \endverbatim
//>
//> \param[in,out] C
//> \verbatim
//>          C is DOUBLE PRECISION array, dimension (LDC,N)
//>          On entry, the m by n matrix C.
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
//>          WORK is DOUBLE PRECISION array, dimension
//>                                   (N) if SIDE = 'L',
//>                                   (M) if SIDE = 'R'
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit
//>          < 0: if INFO = -i, the i-th argument had an illegal value
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
/* Subroutine */ int dorm2l_(char *side, char *trans, int *m, int *n, int *k, 
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *info)
{
    // Table of constant values
    int c__1 = 1;

    // System generated locals
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2;

    // Local variables
    int i__, i1, i2, i3, mi, ni, nq;
    double aii;
    int left;
    extern /* Subroutine */ int dlarf_(char *, int *, int *, double *, int *, 
	    double *, double *, int *, double *);
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    int notran;

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
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    //
    //    NQ is the order of Q
    //
    if (left) {
	nq = *m;
    } else {
	nq = *n;
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORM2L", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || *k == 0) {
	return 0;
    }
    if (left && notran || ! left && ! notran) {
	i1 = 1;
	i2 = *k;
	i3 = 1;
    } else {
	i1 = *k;
	i2 = 1;
	i3 = -1;
    }
    if (left) {
	ni = *n;
    } else {
	mi = *m;
    }
    i__1 = i2;
    i__2 = i3;
    for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	if (left) {
	    //
	    //          H(i) is applied to C(1:m-k+i,1:n)
	    //
	    mi = *m - *k + i__;
	} else {
	    //
	    //          H(i) is applied to C(1:m,1:n-k+i)
	    //
	    ni = *n - *k + i__;
	}
	//
	//       Apply H(i)
	//
	aii = a[nq - *k + i__ + i__ * a_dim1];
	a[nq - *k + i__ + i__ * a_dim1] = 1.;
	dlarf_(side, &mi, &ni, &a[i__ * a_dim1 + 1], &c__1, &tau[i__], &c__[
		c_offset], ldc, &work[1]);
	a[nq - *k + i__ + i__ * a_dim1] = aii;
// L10:
    }
    return 0;
    //
    //    End of DORM2L
    //
} // dorm2l_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORMQL
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORMQL + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dormql.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dormql.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dormql.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORMQL( SIDE, TRANS, M, N, K, A, LDA, TAU, C, LDC,
//                         WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS
//      INTEGER            INFO, K, LDA, LDC, LWORK, M, N
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
//> DORMQL overwrites the general real M-by-N matrix C with
//>
//>                 SIDE = 'L'     SIDE = 'R'
//> TRANS = 'N':      Q * C          C * Q
//> TRANS = 'T':      Q**T * C       C * Q**T
//>
//> where Q is a real orthogonal matrix defined as the product of k
//> elementary reflectors
//>
//>       Q = H(k) . . . H(2) H(1)
//>
//> as returned by DGEQLF. Q is of order M if SIDE = 'L' and of order N
//> if SIDE = 'R'.
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
//> \param[in] K
//> \verbatim
//>          K is INTEGER
//>          The number of elementary reflectors whose product defines
//>          the matrix Q.
//>          If SIDE = 'L', M >= K >= 0;
//>          if SIDE = 'R', N >= K >= 0.
//> \endverbatim
//>
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension (LDA,K)
//>          The i-th column must contain the vector which defines the
//>          elementary reflector H(i), for i = 1,2,...,k, as returned by
//>          DGEQLF in the last k columns of its array argument A.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.
//>          If SIDE = 'L', LDA >= max(1,M);
//>          if SIDE = 'R', LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (K)
//>          TAU(i) must contain the scalar factor of the elementary
//>          reflector H(i), as returned by DGEQLF.
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
/* Subroutine */ int dormql_(char *side, char *trans, int *m, int *n, int *k, 
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__2 = 2;
    int c__65 = 65;

    // System generated locals
    address a__1[2];
    int a_dim1, a_offset, c_dim1, c_offset, i__1, i__2, i__3[2], i__4, i__5;
    char ch__1[2+1]={'\0'};

    // Builtin functions
    /* Subroutine */ int s_cat(char *, char **, int *, int *);

    // Local variables
    int i__, i1, i2, i3, ib, nb, mi, ni, nq, nw, iwt;
    int left;
    extern int lsame_(char *, char *);
    int nbmin, iinfo;
    extern /* Subroutine */ int dorm2l_(char *, char *, int *, int *, int *, 
	    double *, int *, double *, double *, int *, double *, int *), 
	    dlarfb_(char *, char *, char *, char *, int *, int *, int *, 
	    double *, int *, double *, int *, double *, int *, double *, int *
	    ), dlarft_(char *, char *, int *, int *, double *, int *, double *
	    , double *, int *), xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int notran;
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
    left = lsame_(side, "L");
    notran = lsame_(trans, "N");
    lquery = *lwork == -1;
    //
    //    NQ is the order of Q and NW is the minimum dimension of WORK
    //
    if (left) {
	nq = *m;
	nw = max(1,*n);
    } else {
	nq = *n;
	nw = max(1,*m);
    }
    if (! left && ! lsame_(side, "R")) {
	*info = -1;
    } else if (! notran && ! lsame_(trans, "T")) {
	*info = -2;
    } else if (*m < 0) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*k < 0 || *k > nq) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < nw && ! lquery) {
	*info = -12;
    }
    if (*info == 0) {
	//
	//       Compute the workspace requirements
	//
	if (*m == 0 || *n == 0) {
	    lwkopt = 1;
	} else {
	    // Computing MIN
	    // Writing concatenation
	    i__3[0] = 1, a__1[0] = side;
	    i__3[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__3, &c__2);
	    i__1 = 64, i__2 = ilaenv_(&c__1, "DORMQL", ch__1, m, n, k, &c_n1);
	    nb = min(i__1,i__2);
	    lwkopt = nw * nb + 4160;
	}
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DORMQL", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0) {
	return 0;
    }
    nbmin = 2;
    ldwork = nw;
    if (nb > 1 && nb < *k) {
	if (*lwork < nw * nb + 4160) {
	    nb = (*lwork - 4160) / ldwork;
	    // Computing MAX
	    // Writing concatenation
	    i__3[0] = 1, a__1[0] = side;
	    i__3[1] = 1, a__1[1] = trans;
	    s_cat(ch__1, a__1, i__3, &c__2);
	    i__1 = 2, i__2 = ilaenv_(&c__2, "DORMQL", ch__1, m, n, k, &c_n1);
	    nbmin = max(i__1,i__2);
	}
    }
    if (nb < nbmin || nb >= *k) {
	//
	//       Use unblocked code
	//
	dorm2l_(side, trans, m, n, k, &a[a_offset], lda, &tau[1], &c__[
		c_offset], ldc, &work[1], &iinfo);
    } else {
	//
	//       Use blocked code
	//
	iwt = nw * nb + 1;
	if (left && notran || ! left && ! notran) {
	    i1 = 1;
	    i2 = *k;
	    i3 = nb;
	} else {
	    i1 = (*k - 1) / nb * nb + 1;
	    i2 = 1;
	    i3 = -nb;
	}
	if (left) {
	    ni = *n;
	} else {
	    mi = *m;
	}
	i__1 = i2;
	i__2 = i3;
	for (i__ = i1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += i__2) {
	    // Computing MIN
	    i__4 = nb, i__5 = *k - i__ + 1;
	    ib = min(i__4,i__5);
	    //
	    //          Form the triangular factor of the block reflector
	    //          H = H(i+ib-1) . . . H(i+1) H(i)
	    //
	    i__4 = nq - *k + i__ + ib - 1;
	    dlarft_("Backward", "Columnwise", &i__4, &ib, &a[i__ * a_dim1 + 1]
		    , lda, &tau[i__], &work[iwt], &c__65);
	    if (left) {
		//
		//             H or H**T is applied to C(1:m-k+i+ib-1,1:n)
		//
		mi = *m - *k + i__ + ib - 1;
	    } else {
		//
		//             H or H**T is applied to C(1:m,1:n-k+i+ib-1)
		//
		ni = *n - *k + i__ + ib - 1;
	    }
	    //
	    //          Apply H or H**T
	    //
	    dlarfb_(side, trans, "Backward", "Columnwise", &mi, &ni, &ib, &a[
		    i__ * a_dim1 + 1], lda, &work[iwt], &c__65, &c__[c_offset]
		    , ldc, &work[1], &ldwork);
// L10:
	}
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORMQL
    //
} // dormql_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DORMTR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DORMTR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dormtr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dormtr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dormtr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DORMTR( SIDE, UPLO, TRANS, M, N, A, LDA, TAU, C, LDC,
//                         WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          SIDE, TRANS, UPLO
//      INTEGER            INFO, LDA, LDC, LWORK, M, N
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
//> DORMTR overwrites the general real M-by-N matrix C with
//>
//>                 SIDE = 'L'     SIDE = 'R'
//> TRANS = 'N':      Q * C          C * Q
//> TRANS = 'T':      Q**T * C       C * Q**T
//>
//> where Q is a real orthogonal matrix of order nq, with nq = m if
//> SIDE = 'L' and nq = n if SIDE = 'R'. Q is defined as the product of
//> nq-1 elementary reflectors, as returned by DSYTRD:
//>
//> if UPLO = 'U', Q = H(nq-1) . . . H(2) H(1);
//>
//> if UPLO = 'L', Q = H(1) H(2) . . . H(nq-1).
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
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          = 'U': Upper triangle of A contains elementary reflectors
//>                 from DSYTRD;
//>          = 'L': Lower triangle of A contains elementary reflectors
//>                 from DSYTRD.
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
//> \param[in] A
//> \verbatim
//>          A is DOUBLE PRECISION array, dimension
//>                               (LDA,M) if SIDE = 'L'
//>                               (LDA,N) if SIDE = 'R'
//>          The vectors which define the elementary reflectors, as
//>          returned by DSYTRD.
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
//>          reflector H(i), as returned by DSYTRD.
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
/* Subroutine */ int dormtr_(char *side, char *uplo, char *trans, int *m, int 
	*n, double *a, int *lda, double *tau, double *c__, int *ldc, double *
	work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__2 = 2;

    // System generated locals
    address a__1[2];
    int a_dim1, a_offset, c_dim1, c_offset, i__1[2], i__2, i__3;
    char ch__1[2+1]={'\0'};

    // Builtin functions
    /* Subroutine */ int s_cat(char *, char **, int *, int *);

    // Local variables
    int i1, i2, nb, mi, ni, nq, nw;
    int left;
    extern int lsame_(char *, char *);
    int iinfo;
    int upper;
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int dormql_(char *, char *, int *, int *, int *, 
	    double *, int *, double *, double *, int *, double *, int *, int *
	    ), dormqr_(char *, char *, int *, int *, int *, double *, int *, 
	    double *, double *, int *, double *, int *, int *);
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
    left = lsame_(side, "L");
    upper = lsame_(uplo, "U");
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
    } else if (! upper && ! lsame_(uplo, "L")) {
	*info = -2;
    } else if (! lsame_(trans, "N") && ! lsame_(trans, "T")) {
	*info = -3;
    } else if (*m < 0) {
	*info = -4;
    } else if (*n < 0) {
	*info = -5;
    } else if (*lda < max(1,nq)) {
	*info = -7;
    } else if (*ldc < max(1,*m)) {
	*info = -10;
    } else if (*lwork < max(1,nw) && ! lquery) {
	*info = -12;
    }
    if (*info == 0) {
	if (upper) {
	    if (left) {
		// Writing concatenation
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2);
		i__2 = *m - 1;
		i__3 = *m - 1;
		nb = ilaenv_(&c__1, "DORMQL", ch__1, &i__2, n, &i__3, &c_n1);
	    } else {
		// Writing concatenation
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "DORMQL", ch__1, m, &i__2, &i__3, &c_n1);
	    }
	} else {
	    if (left) {
		// Writing concatenation
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2);
		i__2 = *m - 1;
		i__3 = *m - 1;
		nb = ilaenv_(&c__1, "DORMQR", ch__1, &i__2, n, &i__3, &c_n1);
	    } else {
		// Writing concatenation
		i__1[0] = 1, a__1[0] = side;
		i__1[1] = 1, a__1[1] = trans;
		s_cat(ch__1, a__1, i__1, &c__2);
		i__2 = *n - 1;
		i__3 = *n - 1;
		nb = ilaenv_(&c__1, "DORMQR", ch__1, m, &i__2, &i__3, &c_n1);
	    }
	}
	lwkopt = max(1,nw) * nb;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__2 = -(*info);
	xerbla_("DORMTR", &i__2);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*m == 0 || *n == 0 || nq == 1) {
	work[1] = 1.;
	return 0;
    }
    if (left) {
	mi = *m - 1;
	ni = *n;
    } else {
	mi = *m;
	ni = *n - 1;
    }
    if (upper) {
	//
	//       Q was determined by a call to DSYTRD with UPLO = 'U'
	//
	i__2 = nq - 1;
	dormql_(side, trans, &mi, &ni, &i__2, &a[(a_dim1 << 1) + 1], lda, &
		tau[1], &c__[c_offset], ldc, &work[1], lwork, &iinfo);
    } else {
	//
	//       Q was determined by a call to DSYTRD with UPLO = 'L'
	//
	if (left) {
	    i1 = 2;
	    i2 = 1;
	} else {
	    i1 = 1;
	    i2 = 2;
	}
	i__2 = nq - 1;
	dormqr_(side, trans, &mi, &ni, &i__2, &a[a_dim1 + 2], lda, &tau[1], &
		c__[i1 + i2 * c_dim1], ldc, &work[1], lwork, &iinfo);
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DORMTR
    //
} // dormtr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSTEBZ
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DSTEBZ + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dstebz.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dstebz.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dstebz.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DSTEBZ( RANGE, ORDER, N, VL, VU, IL, IU, ABSTOL, D, E,
//                         M, NSPLIT, W, IBLOCK, ISPLIT, WORK, IWORK,
//                         INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          ORDER, RANGE
//      INTEGER            IL, INFO, IU, M, N, NSPLIT
//      DOUBLE PRECISION   ABSTOL, VL, VU
//      ..
//      .. Array Arguments ..
//      INTEGER            IBLOCK( * ), ISPLIT( * ), IWORK( * )
//      DOUBLE PRECISION   D( * ), E( * ), W( * ), WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSTEBZ computes the eigenvalues of a symmetric tridiagonal
//> matrix T.  The user may ask for all eigenvalues, all eigenvalues
//> in the half-open interval (VL, VU], or the IL-th through IU-th
//> eigenvalues.
//>
//> To avoid overflow, the matrix must be scaled so that its
//> largest element is no greater than overflow**(1/2) * underflow**(1/4) in absolute value, and for greatest
//> accuracy, it should not be much smaller than that.
//>
//> See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
//> Matrix", Report CS41, Computer Science Dept., Stanford
//> University, July 21, 1966.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] RANGE
//> \verbatim
//>          RANGE is CHARACTER*1
//>          = 'A': ("All")   all eigenvalues will be found.
//>          = 'V': ("Value") all eigenvalues in the half-open interval
//>                           (VL, VU] will be found.
//>          = 'I': ("Index") the IL-th through IU-th eigenvalues (of the
//>                           entire matrix) will be found.
//> \endverbatim
//>
//> \param[in] ORDER
//> \verbatim
//>          ORDER is CHARACTER*1
//>          = 'B': ("By Block") the eigenvalues will be grouped by
//>                              split-off block (see IBLOCK, ISPLIT) and
//>                              ordered from smallest to largest within
//>                              the block.
//>          = 'E': ("Entire matrix")
//>                              the eigenvalues for the entire matrix
//>                              will be ordered from smallest to
//>                              largest.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the tridiagonal matrix T.  N >= 0.
//> \endverbatim
//>
//> \param[in] VL
//> \verbatim
//>          VL is DOUBLE PRECISION
//>
//>          If RANGE='V', the lower bound of the interval to
//>          be searched for eigenvalues.  Eigenvalues less than or equal
//>          to VL, or greater than VU, will not be returned.  VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] VU
//> \verbatim
//>          VU is DOUBLE PRECISION
//>
//>          If RANGE='V', the upper bound of the interval to
//>          be searched for eigenvalues.  Eigenvalues less than or equal
//>          to VL, or greater than VU, will not be returned.  VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] IL
//> \verbatim
//>          IL is INTEGER
//>
//>          If RANGE='I', the index of the
//>          smallest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[in] IU
//> \verbatim
//>          IU is INTEGER
//>
//>          If RANGE='I', the index of the
//>          largest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[in] ABSTOL
//> \verbatim
//>          ABSTOL is DOUBLE PRECISION
//>          The absolute tolerance for the eigenvalues.  An eigenvalue
//>          (or cluster) is considered to be located if it has been
//>          determined to lie in an interval whose width is ABSTOL or
//>          less.  If ABSTOL is less than or equal to zero, then ULP*|T|
//>          will be used, where |T| means the 1-norm of T.
//>
//>          Eigenvalues will be computed most accurately when ABSTOL is
//>          set to twice the underflow threshold 2*DLAMCH('S'), not zero.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The n diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          The (n-1) off-diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[out] M
//> \verbatim
//>          M is INTEGER
//>          The actual number of eigenvalues found. 0 <= M <= N.
//>          (See also the description of INFO=2,3.)
//> \endverbatim
//>
//> \param[out] NSPLIT
//> \verbatim
//>          NSPLIT is INTEGER
//>          The number of diagonal blocks in the matrix T.
//>          1 <= NSPLIT <= N.
//> \endverbatim
//>
//> \param[out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          On exit, the first M elements of W will contain the
//>          eigenvalues.  (DSTEBZ may use the remaining N-M elements as
//>          workspace.)
//> \endverbatim
//>
//> \param[out] IBLOCK
//> \verbatim
//>          IBLOCK is INTEGER array, dimension (N)
//>          At each row/column j where E(j) is zero or small, the
//>          matrix T is considered to split into a block diagonal
//>          matrix.  On exit, if INFO = 0, IBLOCK(i) specifies to which
//>          block (from 1 to the number of blocks) the eigenvalue W(i)
//>          belongs.  (DSTEBZ may use the remaining N-M elements as
//>          workspace.)
//> \endverbatim
//>
//> \param[out] ISPLIT
//> \verbatim
//>          ISPLIT is INTEGER array, dimension (N)
//>          The splitting points, at which T breaks up into submatrices.
//>          The first submatrix consists of rows/columns 1 to ISPLIT(1),
//>          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2),
//>          etc., and the NSPLIT-th consists of rows/columns
//>          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N.
//>          (Only the first NSPLIT elements will actually be used, but
//>          since the user cannot know a priori what value NSPLIT will
//>          have, N words must be reserved for ISPLIT.)
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (4*N)
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (3*N)
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  some or all of the eigenvalues failed to converge or
//>                were not computed:
//>                =1 or 3: Bisection failed to converge for some
//>                        eigenvalues; these eigenvalues are flagged by a
//>                        negative block number.  The effect is that the
//>                        eigenvalues may not be as accurate as the
//>                        absolute and relative tolerances.  This is
//>                        generally caused by unexpectedly inaccurate
//>                        arithmetic.
//>                =2 or 3: RANGE='I' only: Not all of the eigenvalues
//>                        IL:IU were found.
//>                        Effect: M < IU+1-IL
//>                        Cause:  non-monotonic arithmetic, causing the
//>                                Sturm sequence to be non-monotonic.
//>                        Cure:   recalculate, using RANGE='A', and pick
//>                                out eigenvalues IL:IU.  In some cases,
//>                                increasing the PARAMETER "FUDGE" may
//>                                make things work.
//>                = 4:    RANGE='I', and the Gershgorin interval
//>                        initially used was too small.  No eigenvalues
//>                        were computed.
//>                        Probable cause: your machine has sloppy
//>                                        floating-point arithmetic.
//>                        Cure: Increase the PARAMETER "FUDGE",
//>                              recompile, and try again.
//> \endverbatim
//
//> \par Internal Parameters:
// =========================
//>
//> \verbatim
//>  RELFAC  DOUBLE PRECISION, default = 2.0e0
//>          The relative tolerance.  An interval (a,b] lies within
//>          "relative tolerance" if  b-a < RELFAC*ulp*max(|a|,|b|),
//>          where "ulp" is the machine precision (distance from 1 to
//>          the next larger floating point number.)
//>
//>  FUDGE   DOUBLE PRECISION, default = 2
//>          A "fudge factor" to widen the Gershgorin intervals.  Ideally,
//>          a value of 1 should work, but on machines with sloppy
//>          arithmetic, this needs to be larger.  The default for
//>          publicly released versions should be large enough to handle
//>          the worst machine around.  Note that this has no effect
//>          on accuracy of the solution.
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
/* Subroutine */ int dstebz_(char *range, char *order, int *n, double *vl, 
	double *vu, int *il, int *iu, double *abstol, double *d__, double *e, 
	int *m, int *nsplit, double *w, int *iblock, int *isplit, double *
	work, int *iwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;
    int c__0 = 0;

    // System generated locals
    int i__1, i__2, i__3;
    double d__1, d__2, d__3, d__4, d__5;

    // Builtin functions
    double sqrt(double), log(double);

    // Local variables
    int j, ib, jb, ie, je, nb;
    double gl;
    int im, in;
    double gu;
    int iw;
    double wl, wu;
    int nwl;
    double ulp, wlu, wul;
    int nwu;
    double tmp1, tmp2;
    int iend, ioff, iout, itmp1, jdisc;
    extern int lsame_(char *, char *);
    int iinfo;
    double atoli;
    int iwoff;
    double bnorm;
    int itmax;
    double wkill, rtoli, tnorm;
    extern double dlamch_(char *);
    int ibegin;
    extern /* Subroutine */ int dlaebz_(int *, int *, int *, int *, int *, 
	    int *, double *, double *, double *, double *, double *, double *,
	     int *, double *, double *, int *, int *, double *, int *, int *);
    int irange, idiscl;
    double safemn;
    int idumma[1];
    extern /* Subroutine */ int xerbla_(char *, int *);
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    int idiscu, iorder;
    int ncnvrg;
    double pivmin;
    int toofew;

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
    --iwork;
    --work;
    --isplit;
    --iblock;
    --w;
    --e;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Decode RANGE
    //
    if (lsame_(range, "A")) {
	irange = 1;
    } else if (lsame_(range, "V")) {
	irange = 2;
    } else if (lsame_(range, "I")) {
	irange = 3;
    } else {
	irange = 0;
    }
    //
    //    Decode ORDER
    //
    if (lsame_(order, "B")) {
	iorder = 2;
    } else if (lsame_(order, "E")) {
	iorder = 1;
    } else {
	iorder = 0;
    }
    //
    //    Check for Errors
    //
    if (irange <= 0) {
	*info = -1;
    } else if (iorder <= 0) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (irange == 2) {
	if (*vl >= *vu) {
	    *info = -5;
	}
    } else if (irange == 3 && (*il < 1 || *il > max(1,*n))) {
	*info = -6;
    } else if (irange == 3 && (*iu < min(*n,*il) || *iu > *n)) {
	*info = -7;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSTEBZ", &i__1);
	return 0;
    }
    //
    //    Initialize error flags
    //
    *info = 0;
    ncnvrg = FALSE_;
    toofew = FALSE_;
    //
    //    Quick return if possible
    //
    *m = 0;
    if (*n == 0) {
	return 0;
    }
    //
    //    Simplifications:
    //
    if (irange == 3 && *il == 1 && *iu == *n) {
	irange = 1;
    }
    //
    //    Get machine constants
    //    NB is the minimum vector length for vector bisection, or 0
    //    if only scalar is to be done.
    //
    safemn = dlamch_("S");
    ulp = dlamch_("P");
    rtoli = ulp * 2.;
    nb = ilaenv_(&c__1, "DSTEBZ", " ", n, &c_n1, &c_n1, &c_n1);
    if (nb <= 1) {
	nb = 0;
    }
    //
    //    Special Case when N=1
    //
    if (*n == 1) {
	*nsplit = 1;
	isplit[1] = 1;
	if (irange == 2 && (*vl >= d__[1] || *vu < d__[1])) {
	    *m = 0;
	} else {
	    w[1] = d__[1];
	    iblock[1] = 1;
	    *m = 1;
	}
	return 0;
    }
    //
    //    Compute Splitting Points
    //
    *nsplit = 1;
    work[*n] = 0.;
    pivmin = 1.;
    i__1 = *n;
    for (j = 2; j <= i__1; ++j) {
	// Computing 2nd power
	d__1 = e[j - 1];
	tmp1 = d__1 * d__1;
	// Computing 2nd power
	d__2 = ulp;
	if ((d__1 = d__[j] * d__[j - 1], abs(d__1)) * (d__2 * d__2) + safemn 
		> tmp1) {
	    isplit[*nsplit] = j - 1;
	    ++(*nsplit);
	    work[j - 1] = 0.;
	} else {
	    work[j - 1] = tmp1;
	    pivmin = max(pivmin,tmp1);
	}
// L10:
    }
    isplit[*nsplit] = *n;
    pivmin *= safemn;
    //
    //    Compute Interval and ATOLI
    //
    if (irange == 3) {
	//
	//       RANGE='I': Compute the interval containing eigenvalues
	//                  IL through IU.
	//
	//       Compute Gershgorin interval for entire (split) matrix
	//       and use it as the initial interval
	//
	gu = d__[1];
	gl = d__[1];
	tmp1 = 0.;
	i__1 = *n - 1;
	for (j = 1; j <= i__1; ++j) {
	    tmp2 = sqrt(work[j]);
	    // Computing MAX
	    d__1 = gu, d__2 = d__[j] + tmp1 + tmp2;
	    gu = max(d__1,d__2);
	    // Computing MIN
	    d__1 = gl, d__2 = d__[j] - tmp1 - tmp2;
	    gl = min(d__1,d__2);
	    tmp1 = tmp2;
// L20:
	}
	//
	// Computing MAX
	d__1 = gu, d__2 = d__[*n] + tmp1;
	gu = max(d__1,d__2);
	// Computing MIN
	d__1 = gl, d__2 = d__[*n] - tmp1;
	gl = min(d__1,d__2);
	// Computing MAX
	d__1 = abs(gl), d__2 = abs(gu);
	tnorm = max(d__1,d__2);
	gl = gl - tnorm * 2.1 * ulp * *n - pivmin * 4.2000000000000002;
	gu = gu + tnorm * 2.1 * ulp * *n + pivmin * 2.1;
	//
	//       Compute Iteration parameters
	//
	itmax = (int) ((log(tnorm + pivmin) - log(pivmin)) / log(2.)) + 2;
	if (*abstol <= 0.) {
	    atoli = ulp * tnorm;
	} else {
	    atoli = *abstol;
	}
	work[*n + 1] = gl;
	work[*n + 2] = gl;
	work[*n + 3] = gu;
	work[*n + 4] = gu;
	work[*n + 5] = gl;
	work[*n + 6] = gu;
	iwork[1] = -1;
	iwork[2] = -1;
	iwork[3] = *n + 1;
	iwork[4] = *n + 1;
	iwork[5] = *il - 1;
	iwork[6] = *iu;
	dlaebz_(&c__3, &itmax, n, &c__2, &c__2, &nb, &atoli, &rtoli, &pivmin, 
		&d__[1], &e[1], &work[1], &iwork[5], &work[*n + 1], &work[*n 
		+ 5], &iout, &iwork[1], &w[1], &iblock[1], &iinfo);
	if (iwork[6] == *iu) {
	    wl = work[*n + 1];
	    wlu = work[*n + 3];
	    nwl = iwork[1];
	    wu = work[*n + 4];
	    wul = work[*n + 2];
	    nwu = iwork[4];
	} else {
	    wl = work[*n + 2];
	    wlu = work[*n + 4];
	    nwl = iwork[2];
	    wu = work[*n + 3];
	    wul = work[*n + 1];
	    nwu = iwork[3];
	}
	if (nwl < 0 || nwl >= *n || nwu < 1 || nwu > *n) {
	    *info = 4;
	    return 0;
	}
    } else {
	//
	//       RANGE='A' or 'V' -- Set ATOLI
	//
	// Computing MAX
	d__3 = abs(d__[1]) + abs(e[1]), d__4 = (d__1 = d__[*n], abs(d__1)) + (
		d__2 = e[*n - 1], abs(d__2));
	tnorm = max(d__3,d__4);
	i__1 = *n - 1;
	for (j = 2; j <= i__1; ++j) {
	    // Computing MAX
	    d__4 = tnorm, d__5 = (d__1 = d__[j], abs(d__1)) + (d__2 = e[j - 1]
		    , abs(d__2)) + (d__3 = e[j], abs(d__3));
	    tnorm = max(d__4,d__5);
// L30:
	}
	if (*abstol <= 0.) {
	    atoli = ulp * tnorm;
	} else {
	    atoli = *abstol;
	}
	if (irange == 2) {
	    wl = *vl;
	    wu = *vu;
	} else {
	    wl = 0.;
	    wu = 0.;
	}
    }
    //
    //    Find Eigenvalues -- Loop Over Blocks and recompute NWL and NWU.
    //    NWL accumulates the number of eigenvalues .le. WL,
    //    NWU accumulates the number of eigenvalues .le. WU
    //
    *m = 0;
    iend = 0;
    *info = 0;
    nwl = 0;
    nwu = 0;
    i__1 = *nsplit;
    for (jb = 1; jb <= i__1; ++jb) {
	ioff = iend;
	ibegin = ioff + 1;
	iend = isplit[jb];
	in = iend - ioff;
	if (in == 1) {
	    //
	    //          Special Case -- IN=1
	    //
	    if (irange == 1 || wl >= d__[ibegin] - pivmin) {
		++nwl;
	    }
	    if (irange == 1 || wu >= d__[ibegin] - pivmin) {
		++nwu;
	    }
	    if (irange == 1 || wl < d__[ibegin] - pivmin && wu >= d__[ibegin] 
		    - pivmin) {
		++(*m);
		w[*m] = d__[ibegin];
		iblock[*m] = jb;
	    }
	} else {
	    //
	    //          General Case -- IN > 1
	    //
	    //          Compute Gershgorin Interval
	    //          and use it as the initial interval
	    //
	    gu = d__[ibegin];
	    gl = d__[ibegin];
	    tmp1 = 0.;
	    i__2 = iend - 1;
	    for (j = ibegin; j <= i__2; ++j) {
		tmp2 = (d__1 = e[j], abs(d__1));
		// Computing MAX
		d__1 = gu, d__2 = d__[j] + tmp1 + tmp2;
		gu = max(d__1,d__2);
		// Computing MIN
		d__1 = gl, d__2 = d__[j] - tmp1 - tmp2;
		gl = min(d__1,d__2);
		tmp1 = tmp2;
// L40:
	    }
	    //
	    // Computing MAX
	    d__1 = gu, d__2 = d__[iend] + tmp1;
	    gu = max(d__1,d__2);
	    // Computing MIN
	    d__1 = gl, d__2 = d__[iend] - tmp1;
	    gl = min(d__1,d__2);
	    // Computing MAX
	    d__1 = abs(gl), d__2 = abs(gu);
	    bnorm = max(d__1,d__2);
	    gl = gl - bnorm * 2.1 * ulp * in - pivmin * 2.1;
	    gu = gu + bnorm * 2.1 * ulp * in + pivmin * 2.1;
	    //
	    //          Compute ATOLI for the current submatrix
	    //
	    if (*abstol <= 0.) {
		// Computing MAX
		d__1 = abs(gl), d__2 = abs(gu);
		atoli = ulp * max(d__1,d__2);
	    } else {
		atoli = *abstol;
	    }
	    if (irange > 1) {
		if (gu < wl) {
		    nwl += in;
		    nwu += in;
		    goto L70;
		}
		gl = max(gl,wl);
		gu = min(gu,wu);
		if (gl >= gu) {
		    goto L70;
		}
	    }
	    //
	    //          Set Up Initial Interval
	    //
	    work[*n + 1] = gl;
	    work[*n + in + 1] = gu;
	    dlaebz_(&c__1, &c__0, &in, &in, &c__1, &nb, &atoli, &rtoli, &
		    pivmin, &d__[ibegin], &e[ibegin], &work[ibegin], idumma, &
		    work[*n + 1], &work[*n + (in << 1) + 1], &im, &iwork[1], &
		    w[*m + 1], &iblock[*m + 1], &iinfo);
	    nwl += iwork[1];
	    nwu += iwork[in + 1];
	    iwoff = *m - iwork[1];
	    //
	    //          Compute Eigenvalues
	    //
	    itmax = (int) ((log(gu - gl + pivmin) - log(pivmin)) / log(2.)) + 
		    2;
	    dlaebz_(&c__2, &itmax, &in, &in, &c__1, &nb, &atoli, &rtoli, &
		    pivmin, &d__[ibegin], &e[ibegin], &work[ibegin], idumma, &
		    work[*n + 1], &work[*n + (in << 1) + 1], &iout, &iwork[1],
		     &w[*m + 1], &iblock[*m + 1], &iinfo);
	    //
	    //          Copy Eigenvalues Into W and IBLOCK
	    //          Use -JB for block number for unconverged eigenvalues.
	    //
	    i__2 = iout;
	    for (j = 1; j <= i__2; ++j) {
		tmp1 = (work[j + *n] + work[j + in + *n]) * .5;
		//
		//             Flag non-convergence.
		//
		if (j > iout - iinfo) {
		    ncnvrg = TRUE_;
		    ib = -jb;
		} else {
		    ib = jb;
		}
		i__3 = iwork[j + in] + iwoff;
		for (je = iwork[j] + 1 + iwoff; je <= i__3; ++je) {
		    w[je] = tmp1;
		    iblock[je] = ib;
// L50:
		}
// L60:
	    }
	    *m += im;
	}
L70:
	;
    }
    //
    //    If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU
    //    If NWL+1 < IL or NWU > IU, discard extra eigenvalues.
    //
    if (irange == 3) {
	im = 0;
	idiscl = *il - 1 - nwl;
	idiscu = nwu - *iu;
	if (idiscl > 0 || idiscu > 0) {
	    i__1 = *m;
	    for (je = 1; je <= i__1; ++je) {
		if (w[je] <= wlu && idiscl > 0) {
		    --idiscl;
		} else if (w[je] >= wul && idiscu > 0) {
		    --idiscu;
		} else {
		    ++im;
		    w[im] = w[je];
		    iblock[im] = iblock[je];
		}
// L80:
	    }
	    *m = im;
	}
	if (idiscl > 0 || idiscu > 0) {
	    //
	    //          Code to deal with effects of bad arithmetic:
	    //          Some low eigenvalues to be discarded are not in (WL,WLU],
	    //          or high eigenvalues to be discarded are not in (WUL,WU]
	    //          so just kill off the smallest IDISCL/largest IDISCU
	    //          eigenvalues, by simply finding the smallest/largest
	    //          eigenvalue(s).
	    //
	    //          (If N(w) is monotone non-decreasing, this should never
	    //              happen.)
	    //
	    if (idiscl > 0) {
		wkill = wu;
		i__1 = idiscl;
		for (jdisc = 1; jdisc <= i__1; ++jdisc) {
		    iw = 0;
		    i__2 = *m;
		    for (je = 1; je <= i__2; ++je) {
			if (iblock[je] != 0 && (w[je] < wkill || iw == 0)) {
			    iw = je;
			    wkill = w[je];
			}
// L90:
		    }
		    iblock[iw] = 0;
// L100:
		}
	    }
	    if (idiscu > 0) {
		wkill = wl;
		i__1 = idiscu;
		for (jdisc = 1; jdisc <= i__1; ++jdisc) {
		    iw = 0;
		    i__2 = *m;
		    for (je = 1; je <= i__2; ++je) {
			if (iblock[je] != 0 && (w[je] > wkill || iw == 0)) {
			    iw = je;
			    wkill = w[je];
			}
// L110:
		    }
		    iblock[iw] = 0;
// L120:
		}
	    }
	    im = 0;
	    i__1 = *m;
	    for (je = 1; je <= i__1; ++je) {
		if (iblock[je] != 0) {
		    ++im;
		    w[im] = w[je];
		    iblock[im] = iblock[je];
		}
// L130:
	    }
	    *m = im;
	}
	if (idiscl < 0 || idiscu < 0) {
	    toofew = TRUE_;
	}
    }
    //
    //    If ORDER='B', do nothing -- the eigenvalues are already sorted
    //       by block.
    //    If ORDER='E', sort the eigenvalues from smallest to largest
    //
    if (iorder == 1 && *nsplit > 1) {
	i__1 = *m - 1;
	for (je = 1; je <= i__1; ++je) {
	    ie = 0;
	    tmp1 = w[je];
	    i__2 = *m;
	    for (j = je + 1; j <= i__2; ++j) {
		if (w[j] < tmp1) {
		    ie = j;
		    tmp1 = w[j];
		}
// L140:
	    }
	    if (ie != 0) {
		itmp1 = iblock[ie];
		w[ie] = w[je];
		iblock[ie] = iblock[je];
		w[je] = tmp1;
		iblock[je] = itmp1;
	    }
// L150:
	}
    }
    *info = 0;
    if (ncnvrg) {
	++(*info);
    }
    if (toofew) {
	*info += 2;
    }
    return 0;
    //
    //    End of DSTEBZ
    //
} // dstebz_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSTEIN
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DSTEIN + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dstein.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dstein.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dstein.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DSTEIN( N, D, E, M, W, IBLOCK, ISPLIT, Z, LDZ, WORK,
//                         IWORK, IFAIL, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, LDZ, M, N
//      ..
//      .. Array Arguments ..
//      INTEGER            IBLOCK( * ), IFAIL( * ), ISPLIT( * ),
//     $                   IWORK( * )
//      DOUBLE PRECISION   D( * ), E( * ), W( * ), WORK( * ), Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSTEIN computes the eigenvectors of a real symmetric tridiagonal
//> matrix T corresponding to specified eigenvalues, using inverse
//> iteration.
//>
//> The maximum number of iterations allowed for each eigenvector is
//> specified by an internal parameter MAXITS (currently set to 5).
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix.  N >= 0.
//> \endverbatim
//>
//> \param[in] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The n diagonal elements of the tridiagonal matrix T.
//> \endverbatim
//>
//> \param[in] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          The (n-1) subdiagonal elements of the tridiagonal matrix
//>          T, in elements 1 to N-1.
//> \endverbatim
//>
//> \param[in] M
//> \verbatim
//>          M is INTEGER
//>          The number of eigenvectors to be found.  0 <= M <= N.
//> \endverbatim
//>
//> \param[in] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          The first M elements of W contain the eigenvalues for
//>          which eigenvectors are to be computed.  The eigenvalues
//>          should be grouped by split-off block and ordered from
//>          smallest to largest within the block.  ( The output array
//>          W from DSTEBZ with ORDER = 'B' is expected here. )
//> \endverbatim
//>
//> \param[in] IBLOCK
//> \verbatim
//>          IBLOCK is INTEGER array, dimension (N)
//>          The submatrix indices associated with the corresponding
//>          eigenvalues in W; IBLOCK(i)=1 if eigenvalue W(i) belongs to
//>          the first submatrix from the top, =2 if W(i) belongs to
//>          the second submatrix, etc.  ( The output array IBLOCK
//>          from DSTEBZ is expected here. )
//> \endverbatim
//>
//> \param[in] ISPLIT
//> \verbatim
//>          ISPLIT is INTEGER array, dimension (N)
//>          The splitting points, at which T breaks up into submatrices.
//>          The first submatrix consists of rows/columns 1 to
//>          ISPLIT( 1 ), the second of rows/columns ISPLIT( 1 )+1
//>          through ISPLIT( 2 ), etc.
//>          ( The output array ISPLIT from DSTEBZ is expected here. )
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ, M)
//>          The computed eigenvectors.  The eigenvector associated
//>          with the eigenvalue W(i) is stored in the i-th column of
//>          Z.  Any vector which fails to converge is set to its current
//>          iterate after MAXITS iterations.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>          The leading dimension of the array Z.  LDZ >= max(1,N).
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (5*N)
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (N)
//> \endverbatim
//>
//> \param[out] IFAIL
//> \verbatim
//>          IFAIL is INTEGER array, dimension (M)
//>          On normal exit, all elements of IFAIL are zero.
//>          If one or more eigenvectors fail to converge after
//>          MAXITS iterations, then their indices are stored in
//>          array IFAIL.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0: successful exit.
//>          < 0: if INFO = -i, the i-th argument had an illegal value
//>          > 0: if INFO = i, then i eigenvectors failed to converge
//>               in MAXITS iterations.  Their indices are stored in
//>               array IFAIL.
//> \endverbatim
//
//> \par Internal Parameters:
// =========================
//>
//> \verbatim
//>  MAXITS  INTEGER, default = 5
//>          The maximum number of iterations performed.
//>
//>  EXTRA   INTEGER, default = 2
//>          The number of iterations performed after norm growth
//>          criterion is satisfied, should be at least 1.
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
/* Subroutine */ int dstein_(int *n, double *d__, double *e, int *m, double *
	w, int *iblock, int *isplit, double *z__, int *ldz, double *work, int 
	*iwork, int *ifail, int *info)
{
    // Table of constant values
    int c__2 = 2;
    int c__1 = 1;
    int c_n1 = -1;

    // System generated locals
    int z_dim1, z_offset, i__1, i__2, i__3;
    double d__1, d__2, d__3, d__4, d__5;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int i__, j, b1, j1, bn;
    double xj, scl, eps, sep, nrm, tol;
    int its;
    double xjm, ztr, eps1;
    int jblk, nblk;
    extern double ddot_(int *, double *, int *, double *, int *);
    int jmax;
    extern double dnrm2_(int *, double *, int *);
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    int iseed[4], gpind, iinfo;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    ), daxpy_(int *, double *, double *, int *, double *, int *);
    double ortol;
    int indrv1, indrv2, indrv3, indrv4, indrv5;
    extern double dlamch_(char *);
    extern /* Subroutine */ int dlagtf_(int *, double *, double *, double *, 
	    double *, double *, double *, int *, int *);
    extern int idamax_(int *, double *, int *);
    extern /* Subroutine */ int xerbla_(char *, int *), dlagts_(int *, int *, 
	    double *, double *, double *, double *, int *, double *, double *,
	     int *);
    int nrmchk;
    extern /* Subroutine */ int dlarnv_(int *, int *, int *, double *);
    int blksiz;
    double onenrm, dtpcrt, pertol;

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
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;
    --e;
    --w;
    --iblock;
    --isplit;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --work;
    --iwork;
    --ifail;

    // Function Body
    *info = 0;
    i__1 = *m;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ifail[i__] = 0;
// L10:
    }
    if (*n < 0) {
	*info = -1;
    } else if (*m < 0 || *m > *n) {
	*info = -4;
    } else if (*ldz < max(1,*n)) {
	*info = -9;
    } else {
	i__1 = *m;
	for (j = 2; j <= i__1; ++j) {
	    if (iblock[j] < iblock[j - 1]) {
		*info = -6;
		goto L30;
	    }
	    if (iblock[j] == iblock[j - 1] && w[j] < w[j - 1]) {
		*info = -5;
		goto L30;
	    }
// L20:
	}
L30:
	;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSTEIN", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n == 0 || *m == 0) {
	return 0;
    } else if (*n == 1) {
	z__[z_dim1 + 1] = 1.;
	return 0;
    }
    //
    //    Get machine constants.
    //
    eps = dlamch_("Precision");
    //
    //    Initialize seed for random number generator DLARNV.
    //
    for (i__ = 1; i__ <= 4; ++i__) {
	iseed[i__ - 1] = 1;
// L40:
    }
    //
    //    Initialize pointers.
    //
    indrv1 = 0;
    indrv2 = indrv1 + *n;
    indrv3 = indrv2 + *n;
    indrv4 = indrv3 + *n;
    indrv5 = indrv4 + *n;
    //
    //    Compute eigenvectors of matrix blocks.
    //
    j1 = 1;
    i__1 = iblock[*m];
    for (nblk = 1; nblk <= i__1; ++nblk) {
	//
	//       Find starting and ending indices of block nblk.
	//
	if (nblk == 1) {
	    b1 = 1;
	} else {
	    b1 = isplit[nblk - 1] + 1;
	}
	bn = isplit[nblk];
	blksiz = bn - b1 + 1;
	if (blksiz == 1) {
	    goto L60;
	}
	gpind = j1;
	//
	//       Compute reorthogonalization criterion and stopping criterion.
	//
	onenrm = (d__1 = d__[b1], abs(d__1)) + (d__2 = e[b1], abs(d__2));
	// Computing MAX
	d__3 = onenrm, d__4 = (d__1 = d__[bn], abs(d__1)) + (d__2 = e[bn - 1],
		 abs(d__2));
	onenrm = max(d__3,d__4);
	i__2 = bn - 1;
	for (i__ = b1 + 1; i__ <= i__2; ++i__) {
	    // Computing MAX
	    d__4 = onenrm, d__5 = (d__1 = d__[i__], abs(d__1)) + (d__2 = e[
		    i__ - 1], abs(d__2)) + (d__3 = e[i__], abs(d__3));
	    onenrm = max(d__4,d__5);
// L50:
	}
	ortol = onenrm * .001;
	dtpcrt = sqrt(.1 / blksiz);
	//
	//       Loop through eigenvalues of block nblk.
	//
L60:
	jblk = 0;
	i__2 = *m;
	for (j = j1; j <= i__2; ++j) {
	    if (iblock[j] != nblk) {
		j1 = j;
		goto L160;
	    }
	    ++jblk;
	    xj = w[j];
	    //
	    //          Skip all the work if the block size is one.
	    //
	    if (blksiz == 1) {
		work[indrv1 + 1] = 1.;
		goto L120;
	    }
	    //
	    //          If eigenvalues j and j-1 are too close, add a relatively
	    //          small perturbation.
	    //
	    if (jblk > 1) {
		eps1 = (d__1 = eps * xj, abs(d__1));
		pertol = eps1 * 10.;
		sep = xj - xjm;
		if (sep < pertol) {
		    xj = xjm + pertol;
		}
	    }
	    its = 0;
	    nrmchk = 0;
	    //
	    //          Get random starting vector.
	    //
	    dlarnv_(&c__2, iseed, &blksiz, &work[indrv1 + 1]);
	    //
	    //          Copy the matrix T so it won't be destroyed in factorization.
	    //
	    dcopy_(&blksiz, &d__[b1], &c__1, &work[indrv4 + 1], &c__1);
	    i__3 = blksiz - 1;
	    dcopy_(&i__3, &e[b1], &c__1, &work[indrv2 + 2], &c__1);
	    i__3 = blksiz - 1;
	    dcopy_(&i__3, &e[b1], &c__1, &work[indrv3 + 1], &c__1);
	    //
	    //          Compute LU factors with partial pivoting  ( PT = LU )
	    //
	    tol = 0.;
	    dlagtf_(&blksiz, &work[indrv4 + 1], &xj, &work[indrv2 + 2], &work[
		    indrv3 + 1], &tol, &work[indrv5 + 1], &iwork[1], &iinfo);
	    //
	    //          Update iteration count.
	    //
L70:
	    ++its;
	    if (its > 5) {
		goto L100;
	    }
	    //
	    //          Normalize and scale the righthand side vector Pb.
	    //
	    jmax = idamax_(&blksiz, &work[indrv1 + 1], &c__1);
	    // Computing MAX
	    d__3 = eps, d__4 = (d__1 = work[indrv4 + blksiz], abs(d__1));
	    scl = blksiz * onenrm * max(d__3,d__4) / (d__2 = work[indrv1 + 
		    jmax], abs(d__2));
	    dscal_(&blksiz, &scl, &work[indrv1 + 1], &c__1);
	    //
	    //          Solve the system LU = Pb.
	    //
	    dlagts_(&c_n1, &blksiz, &work[indrv4 + 1], &work[indrv2 + 2], &
		    work[indrv3 + 1], &work[indrv5 + 1], &iwork[1], &work[
		    indrv1 + 1], &tol, &iinfo);
	    //
	    //          Reorthogonalize by modified Gram-Schmidt if eigenvalues are
	    //          close enough.
	    //
	    if (jblk == 1) {
		goto L90;
	    }
	    if ((d__1 = xj - xjm, abs(d__1)) > ortol) {
		gpind = j;
	    }
	    if (gpind != j) {
		i__3 = j - 1;
		for (i__ = gpind; i__ <= i__3; ++i__) {
		    ztr = -ddot_(&blksiz, &work[indrv1 + 1], &c__1, &z__[b1 + 
			    i__ * z_dim1], &c__1);
		    daxpy_(&blksiz, &ztr, &z__[b1 + i__ * z_dim1], &c__1, &
			    work[indrv1 + 1], &c__1);
// L80:
		}
	    }
	    //
	    //          Check the infinity norm of the iterate.
	    //
L90:
	    jmax = idamax_(&blksiz, &work[indrv1 + 1], &c__1);
	    nrm = (d__1 = work[indrv1 + jmax], abs(d__1));
	    //
	    //          Continue for additional iterations after norm reaches
	    //          stopping criterion.
	    //
	    if (nrm < dtpcrt) {
		goto L70;
	    }
	    ++nrmchk;
	    if (nrmchk < 3) {
		goto L70;
	    }
	    goto L110;
	    //
	    //          If stopping criterion was not satisfied, update info and
	    //          store eigenvector number in array ifail.
	    //
L100:
	    ++(*info);
	    ifail[*info] = j;
	    //
	    //          Accept iterate as jth eigenvector.
	    //
L110:
	    scl = 1. / dnrm2_(&blksiz, &work[indrv1 + 1], &c__1);
	    jmax = idamax_(&blksiz, &work[indrv1 + 1], &c__1);
	    if (work[indrv1 + jmax] < 0.) {
		scl = -scl;
	    }
	    dscal_(&blksiz, &scl, &work[indrv1 + 1], &c__1);
L120:
	    i__3 = *n;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		z__[i__ + j * z_dim1] = 0.;
// L130:
	    }
	    i__3 = blksiz;
	    for (i__ = 1; i__ <= i__3; ++i__) {
		z__[b1 + i__ - 1 + j * z_dim1] = work[indrv1 + i__];
// L140:
	    }
	    //
	    //          Save the shift to check eigenvalue spacing at next
	    //          iteration.
	    //
	    xjm = xj;
// L150:
	}
L160:
	;
    }
    return 0;
    //
    //    End of DSTEIN
    //
} // dstein_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSTEMR
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DSTEMR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dstemr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dstemr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dstemr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DSTEMR( JOBZ, RANGE, N, D, E, VL, VU, IL, IU,
//                         M, W, Z, LDZ, NZC, ISUPPZ, TRYRAC, WORK, LWORK,
//                         IWORK, LIWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          JOBZ, RANGE
//      LOGICAL            TRYRAC
//      INTEGER            IL, INFO, IU, LDZ, NZC, LIWORK, LWORK, M, N
//      DOUBLE PRECISION VL, VU
//      ..
//      .. Array Arguments ..
//      INTEGER            ISUPPZ( * ), IWORK( * )
//      DOUBLE PRECISION   D( * ), E( * ), W( * ), WORK( * )
//      DOUBLE PRECISION   Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSTEMR computes selected eigenvalues and, optionally, eigenvectors
//> of a real symmetric tridiagonal matrix T. Any such unreduced matrix has
//> a well defined set of pairwise different real eigenvalues, the corresponding
//> real eigenvectors are pairwise orthogonal.
//>
//> The spectrum may be computed either completely or partially by specifying
//> either an interval (VL,VU] or a range of indices IL:IU for the desired
//> eigenvalues.
//>
//> Depending on the number of desired eigenvalues, these are computed either
//> by bisection or the dqds algorithm. Numerically orthogonal eigenvectors are
//> computed by the use of various suitable L D L^T factorizations near clusters
//> of close eigenvalues (referred to as RRRs, Relatively Robust
//> Representations). An informal sketch of the algorithm follows.
//>
//> For each unreduced block (submatrix) of T,
//>    (a) Compute T - sigma I  = L D L^T, so that L and D
//>        define all the wanted eigenvalues to high relative accuracy.
//>        This means that small relative changes in the entries of D and L
//>        cause only small relative changes in the eigenvalues and
//>        eigenvectors. The standard (unfactored) representation of the
//>        tridiagonal matrix T does not have this property in general.
//>    (b) Compute the eigenvalues to suitable accuracy.
//>        If the eigenvectors are desired, the algorithm attains full
//>        accuracy of the computed eigenvalues only right before
//>        the corresponding vectors have to be computed, see steps c) and d).
//>    (c) For each cluster of close eigenvalues, select a new
//>        shift close to the cluster, find a new factorization, and refine
//>        the shifted eigenvalues to suitable accuracy.
//>    (d) For each eigenvalue with a large enough relative separation compute
//>        the corresponding eigenvector by forming a rank revealing twisted
//>        factorization. Go back to (c) for any clusters that remain.
//>
//> For more details, see:
//> - Inderjit S. Dhillon and Beresford N. Parlett: "Multiple representations
//>   to compute orthogonal eigenvectors of symmetric tridiagonal matrices,"
//>   Linear Algebra and its Applications, 387(1), pp. 1-28, August 2004.
//> - Inderjit Dhillon and Beresford Parlett: "Orthogonal Eigenvectors and
//>   Relative Gaps," SIAM Journal on Matrix Analysis and Applications, Vol. 25,
//>   2004.  Also LAPACK Working Note 154.
//> - Inderjit Dhillon: "A new O(n^2) algorithm for the symmetric
//>   tridiagonal eigenvalue/eigenvector problem",
//>   Computer Science Division Technical Report No. UCB/CSD-97-971,
//>   UC Berkeley, May 1997.
//>
//> Further Details
//> 1.DSTEMR works only on machines which follow IEEE-754
//> floating-point standard in their handling of infinities and NaNs.
//> This permits the use of efficient inner loops avoiding a check for
//> zero divisors.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOBZ
//> \verbatim
//>          JOBZ is CHARACTER*1
//>          = 'N':  Compute eigenvalues only;
//>          = 'V':  Compute eigenvalues and eigenvectors.
//> \endverbatim
//>
//> \param[in] RANGE
//> \verbatim
//>          RANGE is CHARACTER*1
//>          = 'A': all eigenvalues will be found.
//>          = 'V': all eigenvalues in the half-open interval (VL,VU]
//>                 will be found.
//>          = 'I': the IL-th through IU-th eigenvalues will be found.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the N diagonal elements of the tridiagonal matrix
//>          T. On exit, D is overwritten.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N)
//>          On entry, the (N-1) subdiagonal elements of the tridiagonal
//>          matrix T in elements 1 to N-1 of E. E(N) need not be set on
//>          input, but is used internally as workspace.
//>          On exit, E is overwritten.
//> \endverbatim
//>
//> \param[in] VL
//> \verbatim
//>          VL is DOUBLE PRECISION
//>
//>          If RANGE='V', the lower bound of the interval to
//>          be searched for eigenvalues. VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] VU
//> \verbatim
//>          VU is DOUBLE PRECISION
//>
//>          If RANGE='V', the upper bound of the interval to
//>          be searched for eigenvalues. VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] IL
//> \verbatim
//>          IL is INTEGER
//>
//>          If RANGE='I', the index of the
//>          smallest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[in] IU
//> \verbatim
//>          IU is INTEGER
//>
//>          If RANGE='I', the index of the
//>          largest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[out] M
//> \verbatim
//>          M is INTEGER
//>          The total number of eigenvalues found.  0 <= M <= N.
//>          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
//> \endverbatim
//>
//> \param[out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          The first M elements contain the selected eigenvalues in
//>          ascending order.
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ, max(1,M) )
//>          If JOBZ = 'V', and if INFO = 0, then the first M columns of Z
//>          contain the orthonormal eigenvectors of the matrix T
//>          corresponding to the selected eigenvalues, with the i-th
//>          column of Z holding the eigenvector associated with W(i).
//>          If JOBZ = 'N', then Z is not referenced.
//>          Note: the user must ensure that at least max(1,M) columns are
//>          supplied in the array Z; if RANGE = 'V', the exact value of M
//>          is not known in advance and can be computed with a workspace
//>          query by setting NZC = -1, see below.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>          The leading dimension of the array Z.  LDZ >= 1, and if
//>          JOBZ = 'V', then LDZ >= max(1,N).
//> \endverbatim
//>
//> \param[in] NZC
//> \verbatim
//>          NZC is INTEGER
//>          The number of eigenvectors to be held in the array Z.
//>          If RANGE = 'A', then NZC >= max(1,N).
//>          If RANGE = 'V', then NZC >= the number of eigenvalues in (VL,VU].
//>          If RANGE = 'I', then NZC >= IU-IL+1.
//>          If NZC = -1, then a workspace query is assumed; the
//>          routine calculates the number of columns of the array Z that
//>          are needed to hold the eigenvectors.
//>          This value is returned as the first entry of the Z array, and
//>          no error message related to NZC is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] ISUPPZ
//> \verbatim
//>          ISUPPZ is INTEGER array, dimension ( 2*max(1,M) )
//>          The support of the eigenvectors in Z, i.e., the indices
//>          indicating the nonzero elements in Z. The i-th computed eigenvector
//>          is nonzero only in elements ISUPPZ( 2*i-1 ) through
//>          ISUPPZ( 2*i ). This is relevant in the case when the matrix
//>          is split. ISUPPZ is only accessed when JOBZ is 'V' and N > 0.
//> \endverbatim
//>
//> \param[in,out] TRYRAC
//> \verbatim
//>          TRYRAC is LOGICAL
//>          If TRYRAC = .TRUE., indicates that the code should check whether
//>          the tridiagonal matrix defines its eigenvalues to high relative
//>          accuracy.  If so, the code uses relative-accuracy preserving
//>          algorithms that might be (a bit) slower depending on the matrix.
//>          If the matrix does not define its eigenvalues to high relative
//>          accuracy, the code can uses possibly faster algorithms.
//>          If TRYRAC = .FALSE., the code is not required to guarantee
//>          relatively accurate eigenvalues and can use the fastest possible
//>          techniques.
//>          On exit, a .TRUE. TRYRAC will be set to .FALSE. if the matrix
//>          does not define its eigenvalues to high relative accuracy.
//> \endverbatim
//>
//> \param[out] WORK
//> \verbatim
//>          WORK is DOUBLE PRECISION array, dimension (LWORK)
//>          On exit, if INFO = 0, WORK(1) returns the optimal
//>          (and minimal) LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK. LWORK >= max(1,18*N)
//>          if JOBZ = 'V', and LWORK >= max(1,12*N) if JOBZ = 'N'.
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (LIWORK)
//>          On exit, if INFO = 0, IWORK(1) returns the optimal LIWORK.
//> \endverbatim
//>
//> \param[in] LIWORK
//> \verbatim
//>          LIWORK is INTEGER
//>          The dimension of the array IWORK.  LIWORK >= max(1,10*N)
//>          if the eigenvectors are desired, and LIWORK >= max(1,8*N)
//>          if only the eigenvalues are to be computed.
//>          If LIWORK = -1, then a workspace query is assumed; the
//>          routine only calculates the optimal size of the IWORK array,
//>          returns this value as the first entry of the IWORK array, and
//>          no error message related to LIWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          On exit, INFO
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  if INFO = 1X, internal error in DLARRE,
//>                if INFO = 2X, internal error in DLARRV.
//>                Here, the digit X = ABS( IINFO ) < 10, where IINFO is
//>                the nonzero error code returned by DLARRE or
//>                DLARRV, respectively.
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
//> \ingroup doubleOTHERcomputational
//
//> \par Contributors:
// ==================
//>
//> Beresford Parlett, University of California, Berkeley, USA \n
//> Jim Demmel, University of California, Berkeley, USA \n
//> Inderjit Dhillon, University of Texas, Austin, USA \n
//> Osni Marques, LBNL/NERSC, USA \n
//> Christof Voemel, University of California, Berkeley, USA
//
// =====================================================================
/* Subroutine */ int dstemr_(char *jobz, char *range, int *n, double *d__, 
	double *e, double *vl, double *vu, int *il, int *iu, int *m, double *
	w, double *z__, int *ldz, int *nzc, int *isuppz, int *tryrac, double *
	work, int *lwork, int *iwork, int *liwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    double c_b18 = .001;

    // System generated locals
    int z_dim1, z_offset, i__1, i__2;
    double d__1, d__2;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int i__, j;
    double r1, r2;
    int jj;
    double cs;
    int in;
    double sn, wl, wu;
    int iil, iiu;
    double eps, tmp;
    int indd, iend, jblk, wend;
    double rmin, rmax;
    int itmp;
    double tnrm;
    extern /* Subroutine */ int dlae2_(double *, double *, double *, double *,
	     double *);
    int inde2, itmp2;
    double rtol1, rtol2;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    double scale;
    int indgp;
    extern int lsame_(char *, char *);
    int iinfo, iindw, ilast;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    ), dswap_(int *, double *, int *, double *, int *);
    int lwmin;
    int wantz;
    extern /* Subroutine */ int dlaev2_(double *, double *, double *, double *
	    , double *, double *, double *);
    extern double dlamch_(char *);
    int alleig;
    int ibegin;
    int indeig;
    int iindbl;
    int valeig;
    extern /* Subroutine */ int dlarrc_(char *, int *, double *, double *, 
	    double *, double *, double *, int *, int *, int *, int *), 
	    dlarre_(char *, int *, double *, double *, int *, int *, double *,
	     double *, double *, double *, double *, double *, int *, int *, 
	    int *, double *, double *, double *, int *, int *, double *, 
	    double *, double *, int *, int *);
    int wbegin;
    double safmin;
    extern /* Subroutine */ int dlarrj_(int *, double *, double *, int *, int 
	    *, double *, int *, double *, double *, double *, int *, double *,
	     double *, int *), xerbla_(char *, int *);
    double bignum;
    int inderr, iindwk, indgrs, offset;
    extern double dlanst_(char *, int *, double *, double *);
    extern /* Subroutine */ int dlarrr_(int *, double *, double *, int *), 
	    dlarrv_(int *, double *, double *, double *, double *, double *, 
	    int *, int *, int *, int *, double *, double *, double *, double *
	    , double *, double *, int *, int *, double *, double *, int *, 
	    int *, double *, int *, int *), dlasrt_(char *, int *, double *, 
	    int *);
    double thresh;
    int iinspl, ifirst, indwrk, liwmin, nzcmin;
    double pivmin;
    int nsplit;
    double smlnum;
    int lquery, zquery;

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
    --d__;
    --e;
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --isuppz;
    --work;
    --iwork;

    // Function Body
    wantz = lsame_(jobz, "V");
    alleig = lsame_(range, "A");
    valeig = lsame_(range, "V");
    indeig = lsame_(range, "I");
    lquery = *lwork == -1 || *liwork == -1;
    zquery = *nzc == -1;
    //    DSTEMR needs WORK of size 6*N, IWORK of size 3*N.
    //    In addition, DLARRE needs WORK of size 6*N, IWORK of size 5*N.
    //    Furthermore, DLARRV needs WORK of size 12*N, IWORK of size 7*N.
    if (wantz) {
	lwmin = *n * 18;
	liwmin = *n * 10;
    } else {
	//       need less workspace if only the eigenvalues are wanted
	lwmin = *n * 12;
	liwmin = *n << 3;
    }
    wl = 0.;
    wu = 0.;
    iil = 0;
    iiu = 0;
    nsplit = 0;
    if (valeig) {
	//       We do not reference VL, VU in the cases RANGE = 'I','A'
	//       The interval (WL, WU] contains all the wanted eigenvalues.
	//       It is either given by the user or computed in DLARRE.
	wl = *vl;
	wu = *vu;
    } else if (indeig) {
	//       We do not reference IL, IU in the cases RANGE = 'V','A'
	iil = *il;
	iiu = *iu;
    }
    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (alleig || valeig || indeig)) {
	*info = -2;
    } else if (*n < 0) {
	*info = -3;
    } else if (valeig && *n > 0 && wu <= wl) {
	*info = -7;
    } else if (indeig && (iil < 1 || iil > *n)) {
	*info = -8;
    } else if (indeig && (iiu < iil || iiu > *n)) {
	*info = -9;
    } else if (*ldz < 1 || wantz && *ldz < *n) {
	*info = -13;
    } else if (*lwork < lwmin && ! lquery) {
	*info = -17;
    } else if (*liwork < liwmin && ! lquery) {
	*info = -19;
    }
    //
    //    Get machine constants.
    //
    safmin = dlamch_("Safe minimum");
    eps = dlamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = sqrt(smlnum);
    // Computing MIN
    d__1 = sqrt(bignum), d__2 = 1. / sqrt(sqrt(safmin));
    rmax = min(d__1,d__2);
    if (*info == 0) {
	work[1] = (double) lwmin;
	iwork[1] = liwmin;
	if (wantz && alleig) {
	    nzcmin = *n;
	} else if (wantz && valeig) {
	    dlarrc_("T", n, vl, vu, &d__[1], &e[1], &safmin, &nzcmin, &itmp, &
		    itmp2, info);
	} else if (wantz && indeig) {
	    nzcmin = iiu - iil + 1;
	} else {
	    //          WANTZ .EQ. FALSE.
	    nzcmin = 0;
	}
	if (zquery && *info == 0) {
	    z__[z_dim1 + 1] = (double) nzcmin;
	} else if (*nzc < nzcmin && ! zquery) {
	    *info = -14;
	}
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSTEMR", &i__1);
	return 0;
    } else if (lquery || zquery) {
	return 0;
    }
    //
    //    Handle N = 0, 1, and 2 cases immediately
    //
    *m = 0;
    if (*n == 0) {
	return 0;
    }
    if (*n == 1) {
	if (alleig || indeig) {
	    *m = 1;
	    w[1] = d__[1];
	} else {
	    if (wl < d__[1] && wu >= d__[1]) {
		*m = 1;
		w[1] = d__[1];
	    }
	}
	if (wantz && ! zquery) {
	    z__[z_dim1 + 1] = 1.;
	    isuppz[1] = 1;
	    isuppz[2] = 1;
	}
	return 0;
    }
    if (*n == 2) {
	if (! wantz) {
	    dlae2_(&d__[1], &e[1], &d__[2], &r1, &r2);
	} else if (wantz && ! zquery) {
	    dlaev2_(&d__[1], &e[1], &d__[2], &r1, &r2, &cs, &sn);
	}
	if (alleig || valeig && r2 > wl && r2 <= wu || indeig && iil == 1) {
	    ++(*m);
	    w[*m] = r2;
	    if (wantz && ! zquery) {
		z__[*m * z_dim1 + 1] = -sn;
		z__[*m * z_dim1 + 2] = cs;
		//             Note: At most one of SN and CS can be zero.
		if (sn != 0.) {
		    if (cs != 0.) {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[*m * 2] = 2;
		    } else {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[*m * 2] = 1;
		    }
		} else {
		    isuppz[(*m << 1) - 1] = 2;
		    isuppz[*m * 2] = 2;
		}
	    }
	}
	if (alleig || valeig && r1 > wl && r1 <= wu || indeig && iiu == 2) {
	    ++(*m);
	    w[*m] = r1;
	    if (wantz && ! zquery) {
		z__[*m * z_dim1 + 1] = cs;
		z__[*m * z_dim1 + 2] = sn;
		//             Note: At most one of SN and CS can be zero.
		if (sn != 0.) {
		    if (cs != 0.) {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[*m * 2] = 2;
		    } else {
			isuppz[(*m << 1) - 1] = 1;
			isuppz[*m * 2] = 1;
		    }
		} else {
		    isuppz[(*m << 1) - 1] = 2;
		    isuppz[*m * 2] = 2;
		}
	    }
	}
    } else {
	//    Continue with general N
	indgrs = 1;
	inderr = (*n << 1) + 1;
	indgp = *n * 3 + 1;
	indd = (*n << 2) + 1;
	inde2 = *n * 5 + 1;
	indwrk = *n * 6 + 1;
	iinspl = 1;
	iindbl = *n + 1;
	iindw = (*n << 1) + 1;
	iindwk = *n * 3 + 1;
	//
	//       Scale matrix to allowable range, if necessary.
	//       The allowable range is related to the PIVMIN parameter; see the
	//       comments in DLARRD.  The preference for scaling small values
	//       up is heuristic; we expect users' matrices not to be close to the
	//       RMAX threshold.
	//
	scale = 1.;
	tnrm = dlanst_("M", n, &d__[1], &e[1]);
	if (tnrm > 0. && tnrm < rmin) {
	    scale = rmin / tnrm;
	} else if (tnrm > rmax) {
	    scale = rmax / tnrm;
	}
	if (scale != 1.) {
	    dscal_(n, &scale, &d__[1], &c__1);
	    i__1 = *n - 1;
	    dscal_(&i__1, &scale, &e[1], &c__1);
	    tnrm *= scale;
	    if (valeig) {
		//             If eigenvalues in interval have to be found,
		//             scale (WL, WU] accordingly
		wl *= scale;
		wu *= scale;
	    }
	}
	//
	//       Compute the desired eigenvalues of the tridiagonal after splitting
	//       into smaller subblocks if the corresponding off-diagonal elements
	//       are small
	//       THRESH is the splitting parameter for DLARRE
	//       A negative THRESH forces the old splitting criterion based on the
	//       size of the off-diagonal. A positive THRESH switches to splitting
	//       which preserves relative accuracy.
	//
	if (*tryrac) {
	    //          Test whether the matrix warrants the more expensive relative approach.
	    dlarrr_(n, &d__[1], &e[1], &iinfo);
	} else {
	    //          The user does not care about relative accurately eigenvalues
	    iinfo = -1;
	}
	//       Set the splitting criterion
	if (iinfo == 0) {
	    thresh = eps;
	} else {
	    thresh = -eps;
	    //          relative accuracy is desired but T does not guarantee it
	    *tryrac = FALSE_;
	}
	if (*tryrac) {
	    //          Copy original diagonal, needed to guarantee relative accuracy
	    dcopy_(n, &d__[1], &c__1, &work[indd], &c__1);
	}
	//       Store the squares of the offdiagonal values of T
	i__1 = *n - 1;
	for (j = 1; j <= i__1; ++j) {
	    // Computing 2nd power
	    d__1 = e[j];
	    work[inde2 + j - 1] = d__1 * d__1;
// L5:
	}
	//       Set the tolerance parameters for bisection
	if (! wantz) {
	    //          DLARRE computes the eigenvalues to full precision.
	    rtol1 = eps * 4.;
	    rtol2 = eps * 4.;
	} else {
	    //          DLARRE computes the eigenvalues to less than full precision.
	    //          DLARRV will refine the eigenvalue approximations, and we can
	    //          need less accurate initial bisection in DLARRE.
	    //          Note: these settings do only affect the subset case and DLARRE
	    rtol1 = sqrt(eps);
	    // Computing MAX
	    d__1 = sqrt(eps) * .005, d__2 = eps * 4.;
	    rtol2 = max(d__1,d__2);
	}
	dlarre_(range, n, &wl, &wu, &iil, &iiu, &d__[1], &e[1], &work[inde2], 
		&rtol1, &rtol2, &thresh, &nsplit, &iwork[iinspl], m, &w[1], &
		work[inderr], &work[indgp], &iwork[iindbl], &iwork[iindw], &
		work[indgrs], &pivmin, &work[indwrk], &iwork[iindwk], &iinfo);
	if (iinfo != 0) {
	    *info = abs(iinfo) + 10;
	    return 0;
	}
	//       Note that if RANGE .NE. 'V', DLARRE computes bounds on the desired
	//       part of the spectrum. All desired eigenvalues are contained in
	//       (WL,WU]
	if (wantz) {
	    //
	    //          Compute the desired eigenvectors corresponding to the computed
	    //          eigenvalues
	    //
	    dlarrv_(n, &wl, &wu, &d__[1], &e[1], &pivmin, &iwork[iinspl], m, &
		    c__1, m, &c_b18, &rtol1, &rtol2, &w[1], &work[inderr], &
		    work[indgp], &iwork[iindbl], &iwork[iindw], &work[indgrs],
		     &z__[z_offset], ldz, &isuppz[1], &work[indwrk], &iwork[
		    iindwk], &iinfo);
	    if (iinfo != 0) {
		*info = abs(iinfo) + 20;
		return 0;
	    }
	} else {
	    //          DLARRE computes eigenvalues of the (shifted) root representation
	    //          DLARRV returns the eigenvalues of the unshifted matrix.
	    //          However, if the eigenvectors are not desired by the user, we need
	    //          to apply the corresponding shifts from DLARRE to obtain the
	    //          eigenvalues of the original matrix.
	    i__1 = *m;
	    for (j = 1; j <= i__1; ++j) {
		itmp = iwork[iindbl + j - 1];
		w[j] += e[iwork[iinspl + itmp - 1]];
// L20:
	    }
	}
	if (*tryrac) {
	    //          Refine computed eigenvalues so that they are relatively accurate
	    //          with respect to the original matrix T.
	    ibegin = 1;
	    wbegin = 1;
	    i__1 = iwork[iindbl + *m - 1];
	    for (jblk = 1; jblk <= i__1; ++jblk) {
		iend = iwork[iinspl + jblk - 1];
		in = iend - ibegin + 1;
		wend = wbegin - 1;
		//             check if any eigenvalues have to be refined in this block
L36:
		if (wend < *m) {
		    if (iwork[iindbl + wend] == jblk) {
			++wend;
			goto L36;
		    }
		}
		if (wend < wbegin) {
		    ibegin = iend + 1;
		    goto L39;
		}
		offset = iwork[iindw + wbegin - 1] - 1;
		ifirst = iwork[iindw + wbegin - 1];
		ilast = iwork[iindw + wend - 1];
		rtol2 = eps * 4.;
		dlarrj_(&in, &work[indd + ibegin - 1], &work[inde2 + ibegin - 
			1], &ifirst, &ilast, &rtol2, &offset, &w[wbegin], &
			work[inderr + wbegin - 1], &work[indwrk], &iwork[
			iindwk], &pivmin, &tnrm, &iinfo);
		ibegin = iend + 1;
		wbegin = wend + 1;
L39:
		;
	    }
	}
	//
	//       If matrix was scaled, then rescale eigenvalues appropriately.
	//
	if (scale != 1.) {
	    d__1 = 1. / scale;
	    dscal_(m, &d__1, &w[1], &c__1);
	}
    }
    //
    //    If eigenvalues are not in increasing order, then sort them,
    //    possibly along with eigenvectors.
    //
    if (nsplit > 1 || *n == 2) {
	if (! wantz) {
	    dlasrt_("I", m, &w[1], &iinfo);
	    if (iinfo != 0) {
		*info = 3;
		return 0;
	    }
	} else {
	    i__1 = *m - 1;
	    for (j = 1; j <= i__1; ++j) {
		i__ = 0;
		tmp = w[j];
		i__2 = *m;
		for (jj = j + 1; jj <= i__2; ++jj) {
		    if (w[jj] < tmp) {
			i__ = jj;
			tmp = w[jj];
		    }
// L50:
		}
		if (i__ != 0) {
		    w[i__] = w[j];
		    w[j] = tmp;
		    if (wantz) {
			dswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[j * 
				z_dim1 + 1], &c__1);
			itmp = isuppz[(i__ << 1) - 1];
			isuppz[(i__ << 1) - 1] = isuppz[(j << 1) - 1];
			isuppz[(j << 1) - 1] = itmp;
			itmp = isuppz[i__ * 2];
			isuppz[i__ * 2] = isuppz[j * 2];
			isuppz[j * 2] = itmp;
		    }
		}
// L60:
	    }
	}
    }
    work[1] = (double) lwmin;
    iwork[1] = liwmin;
    return 0;
    //
    //    End of DSTEMR
    //
} // dstemr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSTERF
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DSTERF + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsterf.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsterf.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsterf.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DSTERF( N, D, E, INFO )
//
//      .. Scalar Arguments ..
//      INTEGER            INFO, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * ), E( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSTERF computes all eigenvalues of a symmetric tridiagonal matrix
//> using the Pal-Walker-Kahan variant of the QL or QR algorithm.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The order of the matrix.  N >= 0.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the n diagonal elements of the tridiagonal matrix.
//>          On exit, if INFO = 0, the eigenvalues in ascending order.
//> \endverbatim
//>
//> \param[in,out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          On entry, the (n-1) subdiagonal elements of the tridiagonal
//>          matrix.
//>          On exit, E has been destroyed.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  the algorithm failed to find all of the eigenvalues in
//>                a total of 30*N iterations; if INFO = i, then i
//>                elements of E have not converged to zero.
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
/* Subroutine */ int dsterf_(int *n, double *d__, double *e, int *info)
{
    // Table of constant values
    int c__0 = 0;
    int c__1 = 1;
    double c_b33 = 1.;

    // System generated locals
    int i__1;
    double d__1, d__2, d__3;

    // Builtin functions
    double sqrt(double), d_sign(double *, double *);

    // Local variables
    double c__;
    int i__, l, m;
    double p, r__, s;
    int l1;
    double bb, rt1, rt2, eps, rte;
    int lsv;
    double eps2, oldc;
    int lend;
    double rmax;
    int jtot;
    extern /* Subroutine */ int dlae2_(double *, double *, double *, double *,
	     double *);
    double gamma, alpha, sigma, anorm;
    extern double dlapy2_(double *, double *), dlamch_(char *);
    int iscale;
    extern /* Subroutine */ int dlascl_(char *, int *, int *, double *, 
	    double *, int *, int *, double *, int *, int *);
    double oldgam, safmin;
    extern /* Subroutine */ int xerbla_(char *, int *);
    double safmax;
    extern double dlanst_(char *, int *, double *, double *);
    extern /* Subroutine */ int dlasrt_(char *, int *, double *, int *);
    int lendsv;
    double ssfmin;
    int nmaxit;
    double ssfmax;

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
    --e;
    --d__;

    // Function Body
    *info = 0;
    //
    //    Quick return if possible
    //
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("DSTERF", &i__1);
	return 0;
    }
    if (*n <= 1) {
	return 0;
    }
    //
    //    Determine the unit roundoff for this environment.
    //
    eps = dlamch_("E");
    // Computing 2nd power
    d__1 = eps;
    eps2 = d__1 * d__1;
    safmin = dlamch_("S");
    safmax = 1. / safmin;
    ssfmax = sqrt(safmax) / 3.;
    ssfmin = sqrt(safmin) / eps2;
    rmax = dlamch_("O");
    //
    //    Compute the eigenvalues of the tridiagonal matrix.
    //
    nmaxit = *n * 30;
    sigma = 0.;
    jtot = 0;
    //
    //    Determine where the matrix splits and choose QL or QR iteration
    //    for each block, according to whether top or bottom diagonal
    //    element is smaller.
    //
    l1 = 1;
L10:
    if (l1 > *n) {
	goto L170;
    }
    if (l1 > 1) {
	e[l1 - 1] = 0.;
    }
    i__1 = *n - 1;
    for (m = l1; m <= i__1; ++m) {
	if ((d__3 = e[m], abs(d__3)) <= sqrt((d__1 = d__[m], abs(d__1))) * 
		sqrt((d__2 = d__[m + 1], abs(d__2))) * eps) {
	    e[m] = 0.;
	    goto L30;
	}
// L20:
    }
    m = *n;
L30:
    l = l1;
    lsv = l;
    lend = m;
    lendsv = lend;
    l1 = m + 1;
    if (lend == l) {
	goto L10;
    }
    //
    //    Scale submatrix in rows and columns L to LEND
    //
    i__1 = lend - l + 1;
    anorm = dlanst_("M", &i__1, &d__[l], &e[l]);
    iscale = 0;
    if (anorm == 0.) {
	goto L10;
    }
    if (anorm > ssfmax) {
	iscale = 1;
	i__1 = lend - l + 1;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &d__[l], n, 
		info);
	i__1 = lend - l;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmax, &i__1, &c__1, &e[l], n, 
		info);
    } else if (anorm < ssfmin) {
	iscale = 2;
	i__1 = lend - l + 1;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &d__[l], n, 
		info);
	i__1 = lend - l;
	dlascl_("G", &c__0, &c__0, &anorm, &ssfmin, &i__1, &c__1, &e[l], n, 
		info);
    }
    i__1 = lend - 1;
    for (i__ = l; i__ <= i__1; ++i__) {
	// Computing 2nd power
	d__1 = e[i__];
	e[i__] = d__1 * d__1;
// L40:
    }
    //
    //    Choose between QL and QR iteration
    //
    if ((d__1 = d__[lend], abs(d__1)) < (d__2 = d__[l], abs(d__2))) {
	lend = lsv;
	l = lendsv;
    }
    if (lend >= l) {
	//
	//       QL Iteration
	//
	//       Look for small subdiagonal element.
	//
L50:
	if (l != lend) {
	    i__1 = lend - 1;
	    for (m = l; m <= i__1; ++m) {
		if ((d__2 = e[m], abs(d__2)) <= eps2 * (d__1 = d__[m] * d__[m 
			+ 1], abs(d__1))) {
		    goto L70;
		}
// L60:
	    }
	}
	m = lend;
L70:
	if (m < lend) {
	    e[m] = 0.;
	}
	p = d__[l];
	if (m == l) {
	    goto L90;
	}
	//
	//       If remaining matrix is 2 by 2, use DLAE2 to compute its
	//       eigenvalues.
	//
	if (m == l + 1) {
	    rte = sqrt(e[l]);
	    dlae2_(&d__[l], &rte, &d__[l + 1], &rt1, &rt2);
	    d__[l] = rt1;
	    d__[l + 1] = rt2;
	    e[l] = 0.;
	    l += 2;
	    if (l <= lend) {
		goto L50;
	    }
	    goto L150;
	}
	if (jtot == nmaxit) {
	    goto L150;
	}
	++jtot;
	//
	//       Form shift.
	//
	rte = sqrt(e[l]);
	sigma = (d__[l + 1] - p) / (rte * 2.);
	r__ = dlapy2_(&sigma, &c_b33);
	sigma = p - rte / (sigma + d_sign(&r__, &sigma));
	c__ = 1.;
	s = 0.;
	gamma = d__[m] - sigma;
	p = gamma * gamma;
	//
	//       Inner loop
	//
	i__1 = l;
	for (i__ = m - 1; i__ >= i__1; --i__) {
	    bb = e[i__];
	    r__ = p + bb;
	    if (i__ != m - 1) {
		e[i__ + 1] = s * r__;
	    }
	    oldc = c__;
	    c__ = p / r__;
	    s = bb / r__;
	    oldgam = gamma;
	    alpha = d__[i__];
	    gamma = c__ * (alpha - sigma) - s * oldgam;
	    d__[i__ + 1] = oldgam + (alpha - gamma);
	    if (c__ != 0.) {
		p = gamma * gamma / c__;
	    } else {
		p = oldc * bb;
	    }
// L80:
	}
	e[l] = s * p;
	d__[l] = sigma + gamma;
	goto L50;
	//
	//       Eigenvalue found.
	//
L90:
	d__[l] = p;
	++l;
	if (l <= lend) {
	    goto L50;
	}
	goto L150;
    } else {
	//
	//       QR Iteration
	//
	//       Look for small superdiagonal element.
	//
L100:
	i__1 = lend + 1;
	for (m = l; m >= i__1; --m) {
	    if ((d__2 = e[m - 1], abs(d__2)) <= eps2 * (d__1 = d__[m] * d__[m 
		    - 1], abs(d__1))) {
		goto L120;
	    }
// L110:
	}
	m = lend;
L120:
	if (m > lend) {
	    e[m - 1] = 0.;
	}
	p = d__[l];
	if (m == l) {
	    goto L140;
	}
	//
	//       If remaining matrix is 2 by 2, use DLAE2 to compute its
	//       eigenvalues.
	//
	if (m == l - 1) {
	    rte = sqrt(e[l - 1]);
	    dlae2_(&d__[l], &rte, &d__[l - 1], &rt1, &rt2);
	    d__[l] = rt1;
	    d__[l - 1] = rt2;
	    e[l - 1] = 0.;
	    l += -2;
	    if (l >= lend) {
		goto L100;
	    }
	    goto L150;
	}
	if (jtot == nmaxit) {
	    goto L150;
	}
	++jtot;
	//
	//       Form shift.
	//
	rte = sqrt(e[l - 1]);
	sigma = (d__[l - 1] - p) / (rte * 2.);
	r__ = dlapy2_(&sigma, &c_b33);
	sigma = p - rte / (sigma + d_sign(&r__, &sigma));
	c__ = 1.;
	s = 0.;
	gamma = d__[m] - sigma;
	p = gamma * gamma;
	//
	//       Inner loop
	//
	i__1 = l - 1;
	for (i__ = m; i__ <= i__1; ++i__) {
	    bb = e[i__];
	    r__ = p + bb;
	    if (i__ != m) {
		e[i__ - 1] = s * r__;
	    }
	    oldc = c__;
	    c__ = p / r__;
	    s = bb / r__;
	    oldgam = gamma;
	    alpha = d__[i__ + 1];
	    gamma = c__ * (alpha - sigma) - s * oldgam;
	    d__[i__] = oldgam + (alpha - gamma);
	    if (c__ != 0.) {
		p = gamma * gamma / c__;
	    } else {
		p = oldc * bb;
	    }
// L130:
	}
	e[l - 1] = s * p;
	d__[l] = sigma + gamma;
	goto L100;
	//
	//       Eigenvalue found.
	//
L140:
	d__[l] = p;
	--l;
	if (l >= lend) {
	    goto L100;
	}
	goto L150;
    }
    //
    //    Undo scaling if necessary
    //
L150:
    if (iscale == 1) {
	i__1 = lendsv - lsv + 1;
	dlascl_("G", &c__0, &c__0, &ssfmax, &anorm, &i__1, &c__1, &d__[lsv], 
		n, info);
    }
    if (iscale == 2) {
	i__1 = lendsv - lsv + 1;
	dlascl_("G", &c__0, &c__0, &ssfmin, &anorm, &i__1, &c__1, &d__[lsv], 
		n, info);
    }
    //
    //    Check for no convergence to an eigenvalue after a total
    //    of N*MAXIT iterations.
    //
    if (jtot < nmaxit) {
	goto L10;
    }
    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (e[i__] != 0.) {
	    ++(*info);
	}
// L160:
    }
    goto L180;
    //
    //    Sort eigenvalues in increasing order.
    //
L170:
    dlasrt_("I", n, &d__[1], info);
L180:
    return 0;
    //
    //    End of DSTERF
    //
} // dsterf_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief <b> DSYEVR computes the eigenvalues and, optionally, the left and/or right eigenvectors for SY matrices</b>
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DSYEVR + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsyevr.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsyevr.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsyevr.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DSYEVR( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU,
//                         ABSTOL, M, W, Z, LDZ, ISUPPZ, WORK, LWORK,
//                         IWORK, LIWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          JOBZ, RANGE, UPLO
//      INTEGER            IL, INFO, IU, LDA, LDZ, LIWORK, LWORK, M, N
//      DOUBLE PRECISION   ABSTOL, VL, VU
//      ..
//      .. Array Arguments ..
//      INTEGER            ISUPPZ( * ), IWORK( * )
//      DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * ), Z( LDZ, * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSYEVR computes selected eigenvalues and, optionally, eigenvectors
//> of a real symmetric matrix A.  Eigenvalues and eigenvectors can be
//> selected by specifying either a range of values or a range of
//> indices for the desired eigenvalues.
//>
//> DSYEVR first reduces the matrix A to tridiagonal form T with a call
//> to DSYTRD.  Then, whenever possible, DSYEVR calls DSTEMR to compute
//> the eigenspectrum using Relatively Robust Representations.  DSTEMR
//> computes eigenvalues by the dqds algorithm, while orthogonal
//> eigenvectors are computed from various "good" L D L^T representations
//> (also known as Relatively Robust Representations). Gram-Schmidt
//> orthogonalization is avoided as far as possible. More specifically,
//> the various steps of the algorithm are as follows.
//>
//> For each unreduced block (submatrix) of T,
//>    (a) Compute T - sigma I  = L D L^T, so that L and D
//>        define all the wanted eigenvalues to high relative accuracy.
//>        This means that small relative changes in the entries of D and L
//>        cause only small relative changes in the eigenvalues and
//>        eigenvectors. The standard (unfactored) representation of the
//>        tridiagonal matrix T does not have this property in general.
//>    (b) Compute the eigenvalues to suitable accuracy.
//>        If the eigenvectors are desired, the algorithm attains full
//>        accuracy of the computed eigenvalues only right before
//>        the corresponding vectors have to be computed, see steps c) and d).
//>    (c) For each cluster of close eigenvalues, select a new
//>        shift close to the cluster, find a new factorization, and refine
//>        the shifted eigenvalues to suitable accuracy.
//>    (d) For each eigenvalue with a large enough relative separation compute
//>        the corresponding eigenvector by forming a rank revealing twisted
//>        factorization. Go back to (c) for any clusters that remain.
//>
//> The desired accuracy of the output can be specified by the input
//> parameter ABSTOL.
//>
//> For more details, see DSTEMR's documentation and:
//> - Inderjit S. Dhillon and Beresford N. Parlett: "Multiple representations
//>   to compute orthogonal eigenvectors of symmetric tridiagonal matrices,"
//>   Linear Algebra and its Applications, 387(1), pp. 1-28, August 2004.
//> - Inderjit Dhillon and Beresford Parlett: "Orthogonal Eigenvectors and
//>   Relative Gaps," SIAM Journal on Matrix Analysis and Applications, Vol. 25,
//>   2004.  Also LAPACK Working Note 154.
//> - Inderjit Dhillon: "A new O(n^2) algorithm for the symmetric
//>   tridiagonal eigenvalue/eigenvector problem",
//>   Computer Science Division Technical Report No. UCB/CSD-97-971,
//>   UC Berkeley, May 1997.
//>
//>
//> Note 1 : DSYEVR calls DSTEMR when the full spectrum is requested
//> on machines which conform to the ieee-754 floating point standard.
//> DSYEVR calls DSTEBZ and DSTEIN on non-ieee machines and
//> when partial spectrum requests are made.
//>
//> Normal execution of DSTEMR may create NaNs and infinities and
//> hence may abort due to a floating point exception in environments
//> which do not handle NaNs and infinities in the ieee standard default
//> manner.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] JOBZ
//> \verbatim
//>          JOBZ is CHARACTER*1
//>          = 'N':  Compute eigenvalues only;
//>          = 'V':  Compute eigenvalues and eigenvectors.
//> \endverbatim
//>
//> \param[in] RANGE
//> \verbatim
//>          RANGE is CHARACTER*1
//>          = 'A': all eigenvalues will be found.
//>          = 'V': all eigenvalues in the half-open interval (VL,VU]
//>                 will be found.
//>          = 'I': the IL-th through IU-th eigenvalues will be found.
//>          For RANGE = 'V' or 'I' and IU - IL < N - 1, DSTEBZ and
//>          DSTEIN are called
//> \endverbatim
//>
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
//>          A is DOUBLE PRECISION array, dimension (LDA, N)
//>          On entry, the symmetric matrix A.  If UPLO = 'U', the
//>          leading N-by-N upper triangular part of A contains the
//>          upper triangular part of the matrix A.  If UPLO = 'L',
//>          the leading N-by-N lower triangular part of A contains
//>          the lower triangular part of the matrix A.
//>          On exit, the lower triangle (if UPLO='L') or the upper
//>          triangle (if UPLO='U') of A, including the diagonal, is
//>          destroyed.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[in] VL
//> \verbatim
//>          VL is DOUBLE PRECISION
//>          If RANGE='V', the lower bound of the interval to
//>          be searched for eigenvalues. VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] VU
//> \verbatim
//>          VU is DOUBLE PRECISION
//>          If RANGE='V', the upper bound of the interval to
//>          be searched for eigenvalues. VL < VU.
//>          Not referenced if RANGE = 'A' or 'I'.
//> \endverbatim
//>
//> \param[in] IL
//> \verbatim
//>          IL is INTEGER
//>          If RANGE='I', the index of the
//>          smallest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[in] IU
//> \verbatim
//>          IU is INTEGER
//>          If RANGE='I', the index of the
//>          largest eigenvalue to be returned.
//>          1 <= IL <= IU <= N, if N > 0; IL = 1 and IU = 0 if N = 0.
//>          Not referenced if RANGE = 'A' or 'V'.
//> \endverbatim
//>
//> \param[in] ABSTOL
//> \verbatim
//>          ABSTOL is DOUBLE PRECISION
//>          The absolute error tolerance for the eigenvalues.
//>          An approximate eigenvalue is accepted as converged
//>          when it is determined to lie in an interval [a,b]
//>          of width less than or equal to
//>
//>                  ABSTOL + EPS *   max( |a|,|b| ) ,
//>
//>          where EPS is the machine precision.  If ABSTOL is less than
//>          or equal to zero, then  EPS*|T|  will be used in its place,
//>          where |T| is the 1-norm of the tridiagonal matrix obtained
//>          by reducing A to tridiagonal form.
//>
//>          See "Computing Small Singular Values of Bidiagonal Matrices
//>          with Guaranteed High Relative Accuracy," by Demmel and
//>          Kahan, LAPACK Working Note #3.
//>
//>          If high relative accuracy is important, set ABSTOL to
//>          DLAMCH( 'Safe minimum' ).  Doing so will guarantee that
//>          eigenvalues are computed to high relative accuracy when
//>          possible in future releases.  The current code does not
//>          make any guarantees about high relative accuracy, but
//>          future releases will. See J. Barlow and J. Demmel,
//>          "Computing Accurate Eigensystems of Scaled Diagonally
//>          Dominant Matrices", LAPACK Working Note #7, for a discussion
//>          of which matrices define their eigenvalues to high relative
//>          accuracy.
//> \endverbatim
//>
//> \param[out] M
//> \verbatim
//>          M is INTEGER
//>          The total number of eigenvalues found.  0 <= M <= N.
//>          If RANGE = 'A', M = N, and if RANGE = 'I', M = IU-IL+1.
//> \endverbatim
//>
//> \param[out] W
//> \verbatim
//>          W is DOUBLE PRECISION array, dimension (N)
//>          The first M elements contain the selected eigenvalues in
//>          ascending order.
//> \endverbatim
//>
//> \param[out] Z
//> \verbatim
//>          Z is DOUBLE PRECISION array, dimension (LDZ, max(1,M))
//>          If JOBZ = 'V', then if INFO = 0, the first M columns of Z
//>          contain the orthonormal eigenvectors of the matrix A
//>          corresponding to the selected eigenvalues, with the i-th
//>          column of Z holding the eigenvector associated with W(i).
//>          If JOBZ = 'N', then Z is not referenced.
//>          Note: the user must ensure that at least max(1,M) columns are
//>          supplied in the array Z; if RANGE = 'V', the exact value of M
//>          is not known in advance and an upper bound must be used.
//>          Supplying N columns is always safe.
//> \endverbatim
//>
//> \param[in] LDZ
//> \verbatim
//>          LDZ is INTEGER
//>          The leading dimension of the array Z.  LDZ >= 1, and if
//>          JOBZ = 'V', LDZ >= max(1,N).
//> \endverbatim
//>
//> \param[out] ISUPPZ
//> \verbatim
//>          ISUPPZ is INTEGER array, dimension ( 2*max(1,M) )
//>          The support of the eigenvectors in Z, i.e., the indices
//>          indicating the nonzero elements in Z. The i-th eigenvector
//>          is nonzero only in elements ISUPPZ( 2*i-1 ) through
//>          ISUPPZ( 2*i ). This is an output of DSTEMR (tridiagonal
//>          matrix). The support of the eigenvectors of A is typically
//>          1:N because of the orthogonal transformations applied by DORMTR.
//>          Implemented only for RANGE = 'A' or 'I' and IU - IL = N - 1
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
//>          The dimension of the array WORK.  LWORK >= max(1,26*N).
//>          For optimal efficiency, LWORK >= (NB+6)*N,
//>          where NB is the max of the blocksize for DSYTRD and DORMTR
//>          returned by ILAENV.
//>
//>          If LWORK = -1, then a workspace query is assumed; the routine
//>          only calculates the optimal size of the WORK array, returns
//>          this value as the first entry of the WORK array, and no error
//>          message related to LWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] IWORK
//> \verbatim
//>          IWORK is INTEGER array, dimension (MAX(1,LIWORK))
//>          On exit, if INFO = 0, IWORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LIWORK
//> \verbatim
//>          LIWORK is INTEGER
//>          The dimension of the array IWORK.  LIWORK >= max(1,10*N).
//>
//>          If LIWORK = -1, then a workspace query is assumed; the
//>          routine only calculates the optimal size of the IWORK array,
//>          returns this value as the first entry of the IWORK array, and
//>          no error message related to LIWORK is issued by XERBLA.
//> \endverbatim
//>
//> \param[out] INFO
//> \verbatim
//>          INFO is INTEGER
//>          = 0:  successful exit
//>          < 0:  if INFO = -i, the i-th argument had an illegal value
//>          > 0:  Internal error
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
//> \ingroup doubleSYeigen
//
//> \par Contributors:
// ==================
//>
//>     Inderjit Dhillon, IBM Almaden, USA \n
//>     Osni Marques, LBNL/NERSC, USA \n
//>     Ken Stanley, Computer Science Division, University of
//>       California at Berkeley, USA \n
//>     Jason Riedy, Computer Science Division, University of
//>       California at Berkeley, USA \n
//>
// =====================================================================
/* Subroutine */ int dsyevr_(char *jobz, char *range, char *uplo, int *n, 
	double *a, int *lda, double *vl, double *vu, int *il, int *iu, double 
	*abstol, int *m, double *w, double *z__, int *ldz, int *isuppz, 
	double *work, int *lwork, int *iwork, int *liwork, int *info)
{
    // Table of constant values
    int c__10 = 10;
    int c__1 = 1;
    int c__2 = 2;
    int c__3 = 3;
    int c__4 = 4;
    int c_n1 = -1;

    // System generated locals
    int a_dim1, a_offset, z_dim1, z_offset, i__1, i__2;
    double d__1, d__2;

    // Builtin functions
    double sqrt(double);

    // Local variables
    int i__, j, nb, jj;
    double eps, vll, vuu, tmp1;
    int indd, inde;
    double anrm;
    int imax;
    double rmin, rmax;
    int inddd, indee;
    extern /* Subroutine */ int dscal_(int *, double *, double *, int *);
    double sigma;
    extern int lsame_(char *, char *);
    int iinfo;
    char order[1+1]={'\0'};
    int indwk;
    extern /* Subroutine */ int dcopy_(int *, double *, int *, double *, int *
	    ), dswap_(int *, double *, int *, double *, int *);
    int lwmin;
    int lower, wantz;
    extern double dlamch_(char *);
    int alleig, indeig;
    int iscale, ieeeok, indibl, indifl;
    int valeig;
    double safmin;
    extern int ilaenv_(int *, char *, char *, int *, int *, int *, int *);
    extern /* Subroutine */ int xerbla_(char *, int *);
    double abstll, bignum;
    int indtau, indisp;
    extern /* Subroutine */ int dstein_(int *, double *, double *, int *, 
	    double *, int *, int *, double *, int *, double *, int *, int *, 
	    int *), dsterf_(int *, double *, double *, int *);
    int indiwo, indwkn;
    extern double dlansy_(char *, char *, int *, double *, int *, double *);
    extern /* Subroutine */ int dstebz_(char *, char *, int *, double *, 
	    double *, int *, int *, double *, double *, double *, int *, int *
	    , double *, int *, int *, double *, int *, int *), dstemr_(char *,
	     char *, int *, double *, double *, double *, double *, int *, 
	    int *, int *, double *, double *, int *, int *, int *, int *, 
	    double *, int *, int *, int *, int *);
    int liwmin;
    int tryrac;
    extern /* Subroutine */ int dormtr_(char *, char *, char *, int *, int *, 
	    double *, int *, double *, double *, int *, double *, int *, int *
	    );
    int llwrkn, llwork, nsplit;
    double smlnum;
    extern /* Subroutine */ int dsytrd_(char *, int *, double *, int *, 
	    double *, double *, double *, double *, int *, int *);
    int lwkopt;
    int lquery;

    //
    // -- LAPACK driver routine (version 3.7.1) --
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
    --w;
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --isuppz;
    --work;
    --iwork;

    // Function Body
    ieeeok = ilaenv_(&c__10, "DSYEVR", "N", &c__1, &c__2, &c__3, &c__4);
    lower = lsame_(uplo, "L");
    wantz = lsame_(jobz, "V");
    alleig = lsame_(range, "A");
    valeig = lsame_(range, "V");
    indeig = lsame_(range, "I");
    lquery = *lwork == -1 || *liwork == -1;
    //
    // Computing MAX
    i__1 = 1, i__2 = *n * 26;
    lwmin = max(i__1,i__2);
    // Computing MAX
    i__1 = 1, i__2 = *n * 10;
    liwmin = max(i__1,i__2);
    *info = 0;
    if (! (wantz || lsame_(jobz, "N"))) {
	*info = -1;
    } else if (! (alleig || valeig || indeig)) {
	*info = -2;
    } else if (! (lower || lsame_(uplo, "U"))) {
	*info = -3;
    } else if (*n < 0) {
	*info = -4;
    } else if (*lda < max(1,*n)) {
	*info = -6;
    } else {
	if (valeig) {
	    if (*n > 0 && *vu <= *vl) {
		*info = -8;
	    }
	} else if (indeig) {
	    if (*il < 1 || *il > max(1,*n)) {
		*info = -9;
	    } else if (*iu < min(*n,*il) || *iu > *n) {
		*info = -10;
	    }
	}
    }
    if (*info == 0) {
	if (*ldz < 1 || wantz && *ldz < *n) {
	    *info = -15;
	} else if (*lwork < lwmin && ! lquery) {
	    *info = -18;
	} else if (*liwork < liwmin && ! lquery) {
	    *info = -20;
	}
    }
    if (*info == 0) {
	nb = ilaenv_(&c__1, "DSYTRD", uplo, n, &c_n1, &c_n1, &c_n1);
	// Computing MAX
	i__1 = nb, i__2 = ilaenv_(&c__1, "DORMTR", uplo, n, &c_n1, &c_n1, &
		c_n1);
	nb = max(i__1,i__2);
	// Computing MAX
	i__1 = (nb + 1) * *n;
	lwkopt = max(i__1,lwmin);
	work[1] = (double) lwkopt;
	iwork[1] = liwmin;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYEVR", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }
    //
    //    Quick return if possible
    //
    *m = 0;
    if (*n == 0) {
	work[1] = 1.;
	return 0;
    }
    if (*n == 1) {
	work[1] = 7.;
	if (alleig || indeig) {
	    *m = 1;
	    w[1] = a[a_dim1 + 1];
	} else {
	    if (*vl < a[a_dim1 + 1] && *vu >= a[a_dim1 + 1]) {
		*m = 1;
		w[1] = a[a_dim1 + 1];
	    }
	}
	if (wantz) {
	    z__[z_dim1 + 1] = 1.;
	    isuppz[1] = 1;
	    isuppz[2] = 1;
	}
	return 0;
    }
    //
    //    Get machine constants.
    //
    safmin = dlamch_("Safe minimum");
    eps = dlamch_("Precision");
    smlnum = safmin / eps;
    bignum = 1. / smlnum;
    rmin = sqrt(smlnum);
    // Computing MIN
    d__1 = sqrt(bignum), d__2 = 1. / sqrt(sqrt(safmin));
    rmax = min(d__1,d__2);
    //
    //    Scale matrix to allowable range, if necessary.
    //
    iscale = 0;
    abstll = *abstol;
    if (valeig) {
	vll = *vl;
	vuu = *vu;
    }
    anrm = dlansy_("M", uplo, n, &a[a_offset], lda, &work[1]);
    if (anrm > 0. && anrm < rmin) {
	iscale = 1;
	sigma = rmin / anrm;
    } else if (anrm > rmax) {
	iscale = 1;
	sigma = rmax / anrm;
    }
    if (iscale == 1) {
	if (lower) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = *n - j + 1;
		dscal_(&i__2, &sigma, &a[j + j * a_dim1], &c__1);
// L10:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		dscal_(&j, &sigma, &a[j * a_dim1 + 1], &c__1);
// L20:
	    }
	}
	if (*abstol > 0.) {
	    abstll = *abstol * sigma;
	}
	if (valeig) {
	    vll = *vl * sigma;
	    vuu = *vu * sigma;
	}
    }
    //    Initialize indices into workspaces.  Note: The IWORK indices are
    //    used only if DSTERF or DSTEMR fail.
    //    WORK(INDTAU:INDTAU+N-1) stores the scalar factors of the
    //    elementary reflectors used in DSYTRD.
    indtau = 1;
    //    WORK(INDD:INDD+N-1) stores the tridiagonal's diagonal entries.
    indd = indtau + *n;
    //    WORK(INDE:INDE+N-1) stores the off-diagonal entries of the
    //    tridiagonal matrix from DSYTRD.
    inde = indd + *n;
    //    WORK(INDDD:INDDD+N-1) is a copy of the diagonal entries over
    //    -written by DSTEMR (the DSTERF path copies the diagonal to W).
    inddd = inde + *n;
    //    WORK(INDEE:INDEE+N-1) is a copy of the off-diagonal entries over
    //    -written while computing the eigenvalues in DSTERF and DSTEMR.
    indee = inddd + *n;
    //    INDWK is the starting offset of the left-over workspace, and
    //    LLWORK is the remaining workspace size.
    indwk = indee + *n;
    llwork = *lwork - indwk + 1;
    //    IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and
    //    stores the block indices of each of the M<=N eigenvalues.
    indibl = 1;
    //    IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and
    //    stores the starting and finishing indices of each block.
    indisp = indibl + *n;
    //    IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
    //    that corresponding to eigenvectors that fail to converge in
    //    DSTEIN.  This information is discarded; if any fail, the driver
    //    returns INFO > 0.
    indifl = indisp + *n;
    //    INDIWO is the offset of the remaining integer workspace.
    indiwo = indifl + *n;
    //
    //    Call DSYTRD to reduce symmetric matrix to tridiagonal form.
    //
    dsytrd_(uplo, n, &a[a_offset], lda, &work[indd], &work[inde], &work[
	    indtau], &work[indwk], &llwork, &iinfo);
    //
    //    If all eigenvalues are desired
    //    then call DSTERF or DSTEMR and DORMTR.
    //
    if ((alleig || indeig && *il == 1 && *iu == *n) && ieeeok == 1) {
	if (! wantz) {
	    dcopy_(n, &work[indd], &c__1, &w[1], &c__1);
	    i__1 = *n - 1;
	    dcopy_(&i__1, &work[inde], &c__1, &work[indee], &c__1);
	    dsterf_(n, &w[1], &work[indee], info);
	} else {
	    i__1 = *n - 1;
	    dcopy_(&i__1, &work[inde], &c__1, &work[indee], &c__1);
	    dcopy_(n, &work[indd], &c__1, &work[inddd], &c__1);
	    if (*abstol <= *n * 2. * eps) {
		tryrac = TRUE_;
	    } else {
		tryrac = FALSE_;
	    }
	    dstemr_(jobz, "A", n, &work[inddd], &work[indee], vl, vu, il, iu, 
		    m, &w[1], &z__[z_offset], ldz, n, &isuppz[1], &tryrac, &
		    work[indwk], lwork, &iwork[1], liwork, info);
	    //
	    //
	    //
	    //       Apply orthogonal matrix used in reduction to tridiagonal
	    //       form to eigenvectors returned by DSTEMR.
	    //
	    if (wantz && *info == 0) {
		indwkn = inde;
		llwrkn = *lwork - indwkn + 1;
		dormtr_("L", uplo, "N", n, m, &a[a_offset], lda, &work[indtau]
			, &z__[z_offset], ldz, &work[indwkn], &llwrkn, &iinfo)
			;
	    }
	}
	if (*info == 0) {
	    //          Everything worked.  Skip DSTEBZ/DSTEIN.  IWORK(:) are
	    //          undefined.
	    *m = *n;
	    goto L30;
	}
	*info = 0;
    }
    //
    //    Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN.
    //    Also call DSTEBZ and DSTEIN if DSTEMR fails.
    //
    if (wantz) {
	*(unsigned char *)order = 'B';
    } else {
	*(unsigned char *)order = 'E';
    }
    dstebz_(range, order, n, &vll, &vuu, il, iu, &abstll, &work[indd], &work[
	    inde], m, &nsplit, &w[1], &iwork[indibl], &iwork[indisp], &work[
	    indwk], &iwork[indiwo], info);
    if (wantz) {
	dstein_(n, &work[indd], &work[inde], m, &w[1], &iwork[indibl], &iwork[
		indisp], &z__[z_offset], ldz, &work[indwk], &iwork[indiwo], &
		iwork[indifl], info);
	//
	//       Apply orthogonal matrix used in reduction to tridiagonal
	//       form to eigenvectors returned by DSTEIN.
	//
	indwkn = inde;
	llwrkn = *lwork - indwkn + 1;
	dormtr_("L", uplo, "N", n, m, &a[a_offset], lda, &work[indtau], &z__[
		z_offset], ldz, &work[indwkn], &llwrkn, &iinfo);
    }
    //
    //    If matrix was scaled, then rescale eigenvalues appropriately.
    //
    // Jump here if DSTEMR/DSTEIN succeeded.
L30:
    if (iscale == 1) {
	if (*info == 0) {
	    imax = *m;
	} else {
	    imax = *info - 1;
	}
	d__1 = 1. / sigma;
	dscal_(&imax, &d__1, &w[1], &c__1);
    }
    //
    //    If eigenvalues are not in order, then sort them, along with
    //    eigenvectors.  Note: We do not sort the IFAIL portion of IWORK.
    //    It may not be initialized (if DSTEMR/DSTEIN succeeded), and we do
    //    not return this detailed information to the user.
    //
    if (wantz) {
	i__1 = *m - 1;
	for (j = 1; j <= i__1; ++j) {
	    i__ = 0;
	    tmp1 = w[j];
	    i__2 = *m;
	    for (jj = j + 1; jj <= i__2; ++jj) {
		if (w[jj] < tmp1) {
		    i__ = jj;
		    tmp1 = w[jj];
		}
// L40:
	    }
	    if (i__ != 0) {
		w[i__] = w[j];
		w[j] = tmp1;
		dswap_(n, &z__[i__ * z_dim1 + 1], &c__1, &z__[j * z_dim1 + 1],
			 &c__1);
	    }
// L50:
	}
    }
    //
    //    Set WORK(1) to optimal workspace size.
    //
    work[1] = (double) lwkopt;
    iwork[1] = liwmin;
    return 0;
    //
    //    End of DSYEVR
    //
} // dsyevr_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSYMV
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DSYMV(UPLO,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION ALPHA,BETA
//      INTEGER INCX,INCY,LDA,N
//      CHARACTER UPLO
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
//> DSYMV  performs the matrix-vector  operation
//>
//>    y := alpha*A*x + beta*y,
//>
//> where alpha and beta are scalars, x and y are n element vectors and
//> A is an n by n symmetric matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>           On entry, UPLO specifies whether the upper or lower
//>           triangular part of the array A is to be referenced as
//>           follows:
//>
//>              UPLO = 'U' or 'u'   Only the upper triangular part of A
//>                                  is to be referenced.
//>
//>              UPLO = 'L' or 'l'   Only the lower triangular part of A
//>                                  is to be referenced.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry, N specifies the order of the matrix A.
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
//>           Before entry with  UPLO = 'U' or 'u', the leading n by n
//>           upper triangular part of the array A must contain the upper
//>           triangular part of the symmetric matrix and the strictly
//>           lower triangular part of A is not referenced.
//>           Before entry with UPLO = 'L' or 'l', the leading n by n
//>           lower triangular part of the array A must contain the lower
//>           triangular part of the symmetric matrix and the strictly
//>           upper triangular part of A is not referenced.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in the calling (sub) program. LDA must be at least
//>           max( 1, n ).
//> \endverbatim
//>
//> \param[in] X
//> \verbatim
//>          X is DOUBLE PRECISION array, dimension at least
//>           ( 1 + ( n - 1 )*abs( INCX ) ).
//>           Before entry, the incremented array X must contain the n
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
//>           ( 1 + ( n - 1 )*abs( INCY ) ).
//>           Before entry, the incremented array Y must contain the n
//>           element vector y. On exit, Y is overwritten by the updated
//>           vector y.
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
/* Subroutine */ int dsymv_(char *uplo, int *n, double *alpha, double *a, int 
	*lda, double *x, int *incx, double *beta, double *y, int *incy)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, ix, iy, jx, jy, kx, ky, info;
    double temp1, temp2;
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
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*lda < max(1,*n)) {
	info = 5;
    } else if (*incx == 0) {
	info = 7;
    } else if (*incy == 0) {
	info = 10;
    }
    if (info != 0) {
	xerbla_("DSYMV ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*n == 0 || *alpha == 0. && *beta == 1.) {
	return 0;
    }
    //
    //    Set up the start points in  X  and  Y.
    //
    if (*incx > 0) {
	kx = 1;
    } else {
	kx = 1 - (*n - 1) * *incx;
    }
    if (*incy > 0) {
	ky = 1;
    } else {
	ky = 1 - (*n - 1) * *incy;
    }
    //
    //    Start the operations. In this version the elements of A are
    //    accessed sequentially with one pass through the triangular part
    //    of A.
    //
    //    First form  y := beta*y.
    //
    if (*beta != 1.) {
	if (*incy == 1) {
	    if (*beta == 0.) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = 0.;
// L10:
		}
	    } else {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[i__] = *beta * y[i__];
// L20:
		}
	    }
	} else {
	    iy = ky;
	    if (*beta == 0.) {
		i__1 = *n;
		for (i__ = 1; i__ <= i__1; ++i__) {
		    y[iy] = 0.;
		    iy += *incy;
// L30:
		}
	    } else {
		i__1 = *n;
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
    if (lsame_(uplo, "U")) {
	//
	//       Form  y  when A is stored in upper triangle.
	//
	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[j];
		temp2 = 0.;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    y[i__] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[i__];
// L50:
		}
		y[j] = y[j] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
// L60:
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[jx];
		temp2 = 0.;
		ix = kx;
		iy = ky;
		i__2 = j - 1;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    y[iy] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[ix];
		    ix += *incx;
		    iy += *incy;
// L70:
		}
		y[jy] = y[jy] + temp1 * a[j + j * a_dim1] + *alpha * temp2;
		jx += *incx;
		jy += *incy;
// L80:
	    }
	}
    } else {
	//
	//       Form  y  when A is stored in lower triangle.
	//
	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[j];
		temp2 = 0.;
		y[j] += temp1 * a[j + j * a_dim1];
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    y[i__] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[i__];
// L90:
		}
		y[j] += *alpha * temp2;
// L100:
	    }
	} else {
	    jx = kx;
	    jy = ky;
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		temp1 = *alpha * x[jx];
		temp2 = 0.;
		y[jy] += temp1 * a[j + j * a_dim1];
		ix = jx;
		iy = jy;
		i__2 = *n;
		for (i__ = j + 1; i__ <= i__2; ++i__) {
		    ix += *incx;
		    iy += *incy;
		    y[iy] += temp1 * a[i__ + j * a_dim1];
		    temp2 += a[i__ + j * a_dim1] * x[ix];
// L110:
		}
		y[jy] += *alpha * temp2;
		jx += *incx;
		jy += *incy;
// L120:
	    }
	}
    }
    return 0;
    //
    //    End of DSYMV .
    //
} // dsymv_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSYR2
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DSYR2(UPLO,N,ALPHA,X,INCX,Y,INCY,A,LDA)
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION ALPHA
//      INTEGER INCX,INCY,LDA,N
//      CHARACTER UPLO
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
//> DSYR2  performs the symmetric rank 2 operation
//>
//>    A := alpha*x*y**T + alpha*y*x**T + A,
//>
//> where alpha is a scalar, x and y are n element vectors and A is an n
//> by n symmetric matrix.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>           On entry, UPLO specifies whether the upper or lower
//>           triangular part of the array A is to be referenced as
//>           follows:
//>
//>              UPLO = 'U' or 'u'   Only the upper triangular part of A
//>                                  is to be referenced.
//>
//>              UPLO = 'L' or 'l'   Only the lower triangular part of A
//>                                  is to be referenced.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>           On entry, N specifies the order of the matrix A.
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
//>           ( 1 + ( n - 1 )*abs( INCX ) ).
//>           Before entry, the incremented array X must contain the n
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
//>           Before entry with  UPLO = 'U' or 'u', the leading n by n
//>           upper triangular part of the array A must contain the upper
//>           triangular part of the symmetric matrix and the strictly
//>           lower triangular part of A is not referenced. On exit, the
//>           upper triangular part of the array A is overwritten by the
//>           upper triangular part of the updated matrix.
//>           Before entry with UPLO = 'L' or 'l', the leading n by n
//>           lower triangular part of the array A must contain the lower
//>           triangular part of the symmetric matrix and the strictly
//>           upper triangular part of A is not referenced. On exit, the
//>           lower triangular part of the array A is overwritten by the
//>           lower triangular part of the updated matrix.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>           On entry, LDA specifies the first dimension of A as declared
//>           in the calling (sub) program. LDA must be at least
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
/* Subroutine */ int dsyr2_(char *uplo, int *n, double *alpha, double *x, int 
	*incx, double *y, int *incy, double *a, int *lda)
{
    // System generated locals
    int a_dim1, a_offset, i__1, i__2;

    // Local variables
    int i__, j, ix, iy, jx, jy, kx, ky, info;
    double temp1, temp2;
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
    --x;
    --y;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    // Function Body
    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*incy == 0) {
	info = 7;
    } else if (*lda < max(1,*n)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("DSYR2 ", &info);
	return 0;
    }
    //
    //    Quick return if possible.
    //
    if (*n == 0 || *alpha == 0.) {
	return 0;
    }
    //
    //    Set up the start points in X and Y if the increments are not both
    //    unity.
    //
    if (*incx != 1 || *incy != 1) {
	if (*incx > 0) {
	    kx = 1;
	} else {
	    kx = 1 - (*n - 1) * *incx;
	}
	if (*incy > 0) {
	    ky = 1;
	} else {
	    ky = 1 - (*n - 1) * *incy;
	}
	jx = kx;
	jy = ky;
    }
    //
    //    Start the operations. In this version the elements of A are
    //    accessed sequentially with one pass through the triangular part
    //    of A.
    //
    if (lsame_(uplo, "U")) {
	//
	//       Form  A  when A is stored in the upper triangle.
	//
	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[j] != 0. || y[j] != 0.) {
		    temp1 = *alpha * y[j];
		    temp2 = *alpha * x[j];
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] * 
				temp1 + y[i__] * temp2;
// L10:
		    }
		}
// L20:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0. || y[jy] != 0.) {
		    temp1 = *alpha * y[jy];
		    temp2 = *alpha * x[jx];
		    ix = kx;
		    iy = ky;
		    i__2 = j;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] * 
				temp1 + y[iy] * temp2;
			ix += *incx;
			iy += *incy;
// L30:
		    }
		}
		jx += *incx;
		jy += *incy;
// L40:
	    }
	}
    } else {
	//
	//       Form  A  when A is stored in the lower triangle.
	//
	if (*incx == 1 && *incy == 1) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[j] != 0. || y[j] != 0.) {
		    temp1 = *alpha * y[j];
		    temp2 = *alpha * x[j];
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[i__] * 
				temp1 + y[i__] * temp2;
// L50:
		    }
		}
// L60:
	    }
	} else {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		if (x[jx] != 0. || y[jy] != 0.) {
		    temp1 = *alpha * y[jy];
		    temp2 = *alpha * x[jx];
		    ix = jx;
		    iy = jy;
		    i__2 = *n;
		    for (i__ = j; i__ <= i__2; ++i__) {
			a[i__ + j * a_dim1] = a[i__ + j * a_dim1] + x[ix] * 
				temp1 + y[iy] * temp2;
			ix += *incx;
			iy += *incy;
// L70:
		    }
		}
		jx += *incx;
		jy += *incy;
// L80:
	    }
	}
    }
    return 0;
    //
    //    End of DSYR2 .
    //
} // dsyr2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSYR2K
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
// Definition:
// ===========
//
//      SUBROUTINE DSYR2K(UPLO,TRANS,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION ALPHA,BETA
//      INTEGER K,LDA,LDB,LDC,N
//      CHARACTER TRANS,UPLO
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSYR2K  performs one of the symmetric rank 2k operations
//>
//>    C := alpha*A*B**T + alpha*B*A**T + beta*C,
//>
//> or
//>
//>    C := alpha*A**T*B + alpha*B**T*A + beta*C,
//>
//> where  alpha and beta  are scalars, C is an  n by n  symmetric matrix
//> and  A and B  are  n by k  matrices  in the  first  case  and  k by n
//> matrices in the second case.
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
//>              TRANS = 'N' or 'n'   C := alpha*A*B**T + alpha*B*A**T +
//>                                        beta*C.
//>
//>              TRANS = 'T' or 't'   C := alpha*A**T*B + alpha*B**T*A +
//>                                        beta*C.
//>
//>              TRANS = 'C' or 'c'   C := alpha*A**T*B + alpha*B**T*A +
//>                                        beta*C.
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
//>           of  columns  of the  matrices  A and B,  and on  entry  with
//>           TRANS = 'T' or 't' or 'C' or 'c',  K  specifies  the  number
//>           of rows of the matrices  A and B.  K must be at least  zero.
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
//> \param[in] B
//> \verbatim
//>          B is DOUBLE PRECISION array, dimension ( LDB, kb ), where kb is
//>           k  when  TRANS = 'N' or 'n',  and is  n  otherwise.
//>           Before entry with  TRANS = 'N' or 'n',  the  leading  n by k
//>           part of the array  B  must contain the matrix  B,  otherwise
//>           the leading  k by n  part of the array  B  must contain  the
//>           matrix B.
//> \endverbatim
//>
//> \param[in] LDB
//> \verbatim
//>          LDB is INTEGER
//>           On entry, LDB specifies the first dimension of B as declared
//>           in  the  calling  (sub)  program.   When  TRANS = 'N' or 'n'
//>           then  LDB must be at least  max( 1, n ), otherwise  LDB must
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
//>
//>  -- Written on 8-February-1989.
//>     Jack Dongarra, Argonne National Laboratory.
//>     Iain Duff, AERE Harwell.
//>     Jeremy Du Croz, Numerical Algorithms Group Ltd.
//>     Sven Hammarling, Numerical Algorithms Group Ltd.
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dsyr2k_(char *uplo, char *trans, int *n, int *k, double *
	alpha, double *a, int *lda, double *b, int *ldb, double *beta, double 
	*c__, int *ldc)
{
    // System generated locals
    int a_dim1, a_offset, b_dim1, b_offset, c_dim1, c_offset, i__1, i__2, 
	    i__3;

    // Local variables
    int i__, j, l, info;
    double temp1, temp2;
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
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;
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
    } else if (*ldb < max(1,nrowa)) {
	info = 9;
    } else if (*ldc < max(1,*n)) {
	info = 12;
    }
    if (info != 0) {
	xerbla_("DSYR2K", &info);
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
	//       Form  C := alpha*A*B**T + alpha*B*A**T + C.
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
		    if (a[j + l * a_dim1] != 0. || b[j + l * b_dim1] != 0.) {
			temp1 = *alpha * b[j + l * b_dim1];
			temp2 = *alpha * a[j + l * a_dim1];
			i__3 = j;
			for (i__ = 1; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
				    i__ + l * a_dim1] * temp1 + b[i__ + l * 
				    b_dim1] * temp2;
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
		    if (a[j + l * a_dim1] != 0. || b[j + l * b_dim1] != 0.) {
			temp1 = *alpha * b[j + l * b_dim1];
			temp2 = *alpha * a[j + l * a_dim1];
			i__3 = *n;
			for (i__ = j; i__ <= i__3; ++i__) {
			    c__[i__ + j * c_dim1] = c__[i__ + j * c_dim1] + a[
				    i__ + l * a_dim1] * temp1 + b[i__ + l * 
				    b_dim1] * temp2;
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
	//       Form  C := alpha*A**T*B + alpha*B**T*A + C.
	//
	if (upper) {
	    i__1 = *n;
	    for (j = 1; j <= i__1; ++j) {
		i__2 = j;
		for (i__ = 1; i__ <= i__2; ++i__) {
		    temp1 = 0.;
		    temp2 = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
			temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
// L190:
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha * 
				temp2;
		    } else {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1] 
				+ *alpha * temp1 + *alpha * temp2;
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
		    temp1 = 0.;
		    temp2 = 0.;
		    i__3 = *k;
		    for (l = 1; l <= i__3; ++l) {
			temp1 += a[l + i__ * a_dim1] * b[l + j * b_dim1];
			temp2 += b[l + i__ * b_dim1] * a[l + j * a_dim1];
// L220:
		    }
		    if (*beta == 0.) {
			c__[i__ + j * c_dim1] = *alpha * temp1 + *alpha * 
				temp2;
		    } else {
			c__[i__ + j * c_dim1] = *beta * c__[i__ + j * c_dim1] 
				+ *alpha * temp1 + *alpha * temp2;
		    }
// L230:
		}
// L240:
	    }
	}
    }
    return 0;
    //
    //    End of DSYR2K.
    //
} // dsyr2k_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSYTD2 reduces a symmetric matrix to real symmetric tridiagonal form by an orthogonal similarity transformation (unblocked algorithm).
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DSYTD2 + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsytd2.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsytd2.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsytd2.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DSYTD2( UPLO, N, A, LDA, D, E, TAU, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDA, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), D( * ), E( * ), TAU( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSYTD2 reduces a real symmetric matrix A to symmetric tridiagonal
//> form T by an orthogonal similarity transformation: Q**T * A * Q = T.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] UPLO
//> \verbatim
//>          UPLO is CHARACTER*1
//>          Specifies whether the upper or lower triangular part of the
//>          symmetric matrix A is stored:
//>          = 'U':  Upper triangular
//>          = 'L':  Lower triangular
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
//>          n-by-n upper triangular part of A contains the upper
//>          triangular part of the matrix A, and the strictly lower
//>          triangular part of A is not referenced.  If UPLO = 'L', the
//>          leading n-by-n lower triangular part of A contains the lower
//>          triangular part of the matrix A, and the strictly upper
//>          triangular part of A is not referenced.
//>          On exit, if UPLO = 'U', the diagonal and first superdiagonal
//>          of A are overwritten by the corresponding elements of the
//>          tridiagonal matrix T, and the elements above the first
//>          superdiagonal, with the array TAU, represent the orthogonal
//>          matrix Q as a product of elementary reflectors; if UPLO
//>          = 'L', the diagonal and first subdiagonal of A are over-
//>          written by the corresponding elements of the tridiagonal
//>          matrix T, and the elements below the first subdiagonal, with
//>          the array TAU, represent the orthogonal matrix Q as a product
//>          of elementary reflectors. See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The diagonal elements of the tridiagonal matrix T:
//>          D(i) = A(i,i).
//> \endverbatim
//>
//> \param[out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          The off-diagonal elements of the tridiagonal matrix T:
//>          E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.
//> \endverbatim
//>
//> \param[out] TAU
//> \verbatim
//>          TAU is DOUBLE PRECISION array, dimension (N-1)
//>          The scalar factors of the elementary reflectors (see Further
//>          Details).
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
//> \ingroup doubleSYcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  If UPLO = 'U', the matrix Q is represented as a product of elementary
//>  reflectors
//>
//>     Q = H(n-1) . . . H(2) H(1).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
//>  A(1:i-1,i+1), and tau in TAU(i).
//>
//>  If UPLO = 'L', the matrix Q is represented as a product of elementary
//>  reflectors
//>
//>     Q = H(1) H(2) . . . H(n-1).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
//>  and tau in TAU(i).
//>
//>  The contents of A on exit are illustrated by the following examples
//>  with n = 5:
//>
//>  if UPLO = 'U':                       if UPLO = 'L':
//>
//>    (  d   e   v2  v3  v4 )              (  d                  )
//>    (      d   e   v3  v4 )              (  e   d              )
//>    (          d   e   v4 )              (  v1  e   d          )
//>    (              d   e  )              (  v1  v2  e   d      )
//>    (                  d  )              (  v1  v2  v3  e   d  )
//>
//>  where d and e denote diagonal and off-diagonal elements of T, and vi
//>  denotes an element of the vector defining H(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dsytd2_(char *uplo, int *n, double *a, int *lda, double *
	d__, double *e, double *tau, int *info)
{
    // Table of constant values
    int c__1 = 1;
    double c_b8 = 0.;
    double c_b14 = -1.;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__;
    extern double ddot_(int *, double *, int *, double *, int *);
    double taui;
    extern /* Subroutine */ int dsyr2_(char *, int *, double *, double *, int 
	    *, double *, int *, double *, int *);
    double alpha;
    extern int lsame_(char *, char *);
    extern /* Subroutine */ int daxpy_(int *, double *, double *, int *, 
	    double *, int *);
    int upper;
    extern /* Subroutine */ int dsymv_(char *, int *, double *, double *, int 
	    *, double *, int *, double *, double *, int *), dlarfg_(int *, 
	    double *, double *, int *, double *), xerbla_(char *, int *);

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
    //    Test the input parameters
    //
    // Parameter adjustments
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tau;

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
	xerbla_("DSYTD2", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n <= 0) {
	return 0;
    }
    if (upper) {
	//
	//       Reduce the upper triangle of A
	//
	for (i__ = *n - 1; i__ >= 1; --i__) {
	    //
	    //          Generate elementary reflector H(i) = I - tau * v * v**T
	    //          to annihilate A(1:i-1,i+1)
	    //
	    dlarfg_(&i__, &a[i__ + (i__ + 1) * a_dim1], &a[(i__ + 1) * a_dim1 
		    + 1], &c__1, &taui);
	    e[i__] = a[i__ + (i__ + 1) * a_dim1];
	    if (taui != 0.) {
		//
		//             Apply H(i) from both sides to A(1:i,1:i)
		//
		a[i__ + (i__ + 1) * a_dim1] = 1.;
		//
		//             Compute  x := tau * A * v  storing x in TAU(1:i)
		//
		dsymv_(uplo, &i__, &taui, &a[a_offset], lda, &a[(i__ + 1) * 
			a_dim1 + 1], &c__1, &c_b8, &tau[1], &c__1);
		//
		//             Compute  w := x - 1/2 * tau * (x**T * v) * v
		//
		alpha = taui * -.5 * ddot_(&i__, &tau[1], &c__1, &a[(i__ + 1) 
			* a_dim1 + 1], &c__1);
		daxpy_(&i__, &alpha, &a[(i__ + 1) * a_dim1 + 1], &c__1, &tau[
			1], &c__1);
		//
		//             Apply the transformation as a rank-2 update:
		//                A := A - v * w**T - w * v**T
		//
		dsyr2_(uplo, &i__, &c_b14, &a[(i__ + 1) * a_dim1 + 1], &c__1, 
			&tau[1], &c__1, &a[a_offset], lda);
		a[i__ + (i__ + 1) * a_dim1] = e[i__];
	    }
	    d__[i__ + 1] = a[i__ + 1 + (i__ + 1) * a_dim1];
	    tau[i__] = taui;
// L10:
	}
	d__[1] = a[a_dim1 + 1];
    } else {
	//
	//       Reduce the lower triangle of A
	//
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    //
	    //          Generate elementary reflector H(i) = I - tau * v * v**T
	    //          to annihilate A(i+2:n,i)
	    //
	    i__2 = *n - i__;
	    // Computing MIN
	    i__3 = i__ + 2;
	    dlarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3,*n) + i__ *
		     a_dim1], &c__1, &taui);
	    e[i__] = a[i__ + 1 + i__ * a_dim1];
	    if (taui != 0.) {
		//
		//             Apply H(i) from both sides to A(i+1:n,i+1:n)
		//
		a[i__ + 1 + i__ * a_dim1] = 1.;
		//
		//             Compute  x := tau * A * v  storing y in TAU(i:n-1)
		//
		i__2 = *n - i__;
		dsymv_(uplo, &i__2, &taui, &a[i__ + 1 + (i__ + 1) * a_dim1], 
			lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b8, &tau[
			i__], &c__1);
		//
		//             Compute  w := x - 1/2 * tau * (x**T * v) * v
		//
		i__2 = *n - i__;
		alpha = taui * -.5 * ddot_(&i__2, &tau[i__], &c__1, &a[i__ + 
			1 + i__ * a_dim1], &c__1);
		i__2 = *n - i__;
		daxpy_(&i__2, &alpha, &a[i__ + 1 + i__ * a_dim1], &c__1, &tau[
			i__], &c__1);
		//
		//             Apply the transformation as a rank-2 update:
		//                A := A - v * w**T - w * v**T
		//
		i__2 = *n - i__;
		dsyr2_(uplo, &i__2, &c_b14, &a[i__ + 1 + i__ * a_dim1], &c__1,
			 &tau[i__], &c__1, &a[i__ + 1 + (i__ + 1) * a_dim1], 
			lda);
		a[i__ + 1 + i__ * a_dim1] = e[i__];
	    }
	    d__[i__] = a[i__ + i__ * a_dim1];
	    tau[i__] = taui;
// L20:
	}
	d__[*n] = a[*n + *n * a_dim1];
    }
    return 0;
    //
    //    End of DSYTD2
    //
} // dsytd2_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DSYTRD
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DSYTRD + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dsytrd.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dsytrd.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dsytrd.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DSYTRD( UPLO, N, A, LDA, D, E, TAU, WORK, LWORK, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          UPLO
//      INTEGER            INFO, LDA, LWORK, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   A( LDA, * ), D( * ), E( * ), TAU( * ),
//     $                   WORK( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DSYTRD reduces a real symmetric matrix A to real symmetric
//> tridiagonal form T by an orthogonal similarity transformation:
//> Q**T * A * Q = T.
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
//>          On exit, if UPLO = 'U', the diagonal and first superdiagonal
//>          of A are overwritten by the corresponding elements of the
//>          tridiagonal matrix T, and the elements above the first
//>          superdiagonal, with the array TAU, represent the orthogonal
//>          matrix Q as a product of elementary reflectors; if UPLO
//>          = 'L', the diagonal and first subdiagonal of A are over-
//>          written by the corresponding elements of the tridiagonal
//>          matrix T, and the elements below the first subdiagonal, with
//>          the array TAU, represent the orthogonal matrix Q as a product
//>          of elementary reflectors. See Further Details.
//> \endverbatim
//>
//> \param[in] LDA
//> \verbatim
//>          LDA is INTEGER
//>          The leading dimension of the array A.  LDA >= max(1,N).
//> \endverbatim
//>
//> \param[out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          The diagonal elements of the tridiagonal matrix T:
//>          D(i) = A(i,i).
//> \endverbatim
//>
//> \param[out] E
//> \verbatim
//>          E is DOUBLE PRECISION array, dimension (N-1)
//>          The off-diagonal elements of the tridiagonal matrix T:
//>          E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.
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
//>          WORK is DOUBLE PRECISION array, dimension (MAX(1,LWORK))
//>          On exit, if INFO = 0, WORK(1) returns the optimal LWORK.
//> \endverbatim
//>
//> \param[in] LWORK
//> \verbatim
//>          LWORK is INTEGER
//>          The dimension of the array WORK.  LWORK >= 1.
//>          For optimum performance LWORK >= N*NB, where NB is the
//>          optimal blocksize.
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
//> \ingroup doubleSYcomputational
//
//> \par Further Details:
// =====================
//>
//> \verbatim
//>
//>  If UPLO = 'U', the matrix Q is represented as a product of elementary
//>  reflectors
//>
//>     Q = H(n-1) . . . H(2) H(1).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in
//>  A(1:i-1,i+1), and tau in TAU(i).
//>
//>  If UPLO = 'L', the matrix Q is represented as a product of elementary
//>  reflectors
//>
//>     Q = H(1) H(2) . . . H(n-1).
//>
//>  Each H(i) has the form
//>
//>     H(i) = I - tau * v * v**T
//>
//>  where tau is a real scalar, and v is a real vector with
//>  v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),
//>  and tau in TAU(i).
//>
//>  The contents of A on exit are illustrated by the following examples
//>  with n = 5:
//>
//>  if UPLO = 'U':                       if UPLO = 'L':
//>
//>    (  d   e   v2  v3  v4 )              (  d                  )
//>    (      d   e   v3  v4 )              (  e   d              )
//>    (          d   e   v4 )              (  v1  e   d          )
//>    (              d   e  )              (  v1  v2  e   d      )
//>    (                  d  )              (  v1  v2  v3  e   d  )
//>
//>  where d and e denote diagonal and off-diagonal elements of T, and vi
//>  denotes an element of the vector defining H(i).
//> \endverbatim
//>
// =====================================================================
/* Subroutine */ int dsytrd_(char *uplo, int *n, double *a, int *lda, double *
	d__, double *e, double *tau, double *work, int *lwork, int *info)
{
    // Table of constant values
    int c__1 = 1;
    int c_n1 = -1;
    int c__3 = 3;
    int c__2 = 2;
    double c_b22 = -1.;
    double c_b23 = 1.;

    // System generated locals
    int a_dim1, a_offset, i__1, i__2, i__3;

    // Local variables
    int i__, j, nb, kk, nx, iws;
    extern int lsame_(char *, char *);
    int nbmin, iinfo;
    int upper;
    extern /* Subroutine */ int dsytd2_(char *, int *, double *, int *, 
	    double *, double *, double *, int *), dsyr2k_(char *, char *, int 
	    *, int *, double *, double *, int *, double *, int *, double *, 
	    double *, int *), dlatrd_(char *, int *, int *, double *, int *, 
	    double *, double *, double *, int *), xerbla_(char *, int *);
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
    --d__;
    --e;
    --tau;
    --work;

    // Function Body
    *info = 0;
    upper = lsame_(uplo, "U");
    lquery = *lwork == -1;
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*lwork < 1 && ! lquery) {
	*info = -9;
    }
    if (*info == 0) {
	//
	//       Determine the block size.
	//
	nb = ilaenv_(&c__1, "DSYTRD", uplo, n, &c_n1, &c_n1, &c_n1);
	lwkopt = *n * nb;
	work[1] = (double) lwkopt;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYTRD", &i__1);
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
    nx = *n;
    iws = 1;
    if (nb > 1 && nb < *n) {
	//
	//       Determine when to cross over from blocked to unblocked code
	//       (last block is always handled by unblocked code).
	//
	// Computing MAX
	i__1 = nb, i__2 = ilaenv_(&c__3, "DSYTRD", uplo, n, &c_n1, &c_n1, &
		c_n1);
	nx = max(i__1,i__2);
	if (nx < *n) {
	    //
	    //          Determine if workspace is large enough for blocked code.
	    //
	    ldwork = *n;
	    iws = ldwork * nb;
	    if (*lwork < iws) {
		//
		//             Not enough workspace to use optimal NB:  determine the
		//             minimum value of NB, and reduce NB or force use of
		//             unblocked code by setting NX = N.
		//
		// Computing MAX
		i__1 = *lwork / ldwork;
		nb = max(i__1,1);
		nbmin = ilaenv_(&c__2, "DSYTRD", uplo, n, &c_n1, &c_n1, &c_n1)
			;
		if (nb < nbmin) {
		    nx = *n;
		}
	    }
	} else {
	    nx = *n;
	}
    } else {
	nb = 1;
    }
    if (upper) {
	//
	//       Reduce the upper triangle of A.
	//       Columns 1:kk are handled by the unblocked method.
	//
	kk = *n - (*n - nx + nb - 1) / nb * nb;
	i__1 = kk + 1;
	i__2 = -nb;
	for (i__ = *n - nb + 1; i__2 < 0 ? i__ >= i__1 : i__ <= i__1; i__ += 
		i__2) {
	    //
	    //          Reduce columns i:i+nb-1 to tridiagonal form and form the
	    //          matrix W which is needed to update the unreduced part of
	    //          the matrix
	    //
	    i__3 = i__ + nb - 1;
	    dlatrd_(uplo, &i__3, &nb, &a[a_offset], lda, &e[1], &tau[1], &
		    work[1], &ldwork);
	    //
	    //          Update the unreduced submatrix A(1:i-1,1:i-1), using an
	    //          update of the form:  A := A - V*W**T - W*V**T
	    //
	    i__3 = i__ - 1;
	    dsyr2k_(uplo, "No transpose", &i__3, &nb, &c_b22, &a[i__ * a_dim1 
		    + 1], lda, &work[1], &ldwork, &c_b23, &a[a_offset], lda);
	    //
	    //          Copy superdiagonal elements back into A, and diagonal
	    //          elements into D
	    //
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j - 1 + j * a_dim1] = e[j - 1];
		d__[j] = a[j + j * a_dim1];
// L10:
	    }
// L20:
	}
	//
	//       Use unblocked code to reduce the last or only block
	//
	dsytd2_(uplo, &kk, &a[a_offset], lda, &d__[1], &e[1], &tau[1], &iinfo)
		;
    } else {
	//
	//       Reduce the lower triangle of A
	//
	i__2 = *n - nx;
	i__1 = nb;
	for (i__ = 1; i__1 < 0 ? i__ >= i__2 : i__ <= i__2; i__ += i__1) {
	    //
	    //          Reduce columns i:i+nb-1 to tridiagonal form and form the
	    //          matrix W which is needed to update the unreduced part of
	    //          the matrix
	    //
	    i__3 = *n - i__ + 1;
	    dlatrd_(uplo, &i__3, &nb, &a[i__ + i__ * a_dim1], lda, &e[i__], &
		    tau[i__], &work[1], &ldwork);
	    //
	    //          Update the unreduced submatrix A(i+ib:n,i+ib:n), using
	    //          an update of the form:  A := A - V*W**T - W*V**T
	    //
	    i__3 = *n - i__ - nb + 1;
	    dsyr2k_(uplo, "No transpose", &i__3, &nb, &c_b22, &a[i__ + nb + 
		    i__ * a_dim1], lda, &work[nb + 1], &ldwork, &c_b23, &a[
		    i__ + nb + (i__ + nb) * a_dim1], lda);
	    //
	    //          Copy subdiagonal elements back into A, and diagonal
	    //          elements into D
	    //
	    i__3 = i__ + nb - 1;
	    for (j = i__; j <= i__3; ++j) {
		a[j + 1 + j * a_dim1] = e[j];
		d__[j] = a[j + j * a_dim1];
// L30:
	    }
// L40:
	}
	//
	//       Use unblocked code to reduce the last or only block
	//
	i__1 = *n - i__ + 1;
	dsytd2_(uplo, &i__1, &a[i__ + i__ * a_dim1], lda, &d__[i__], &e[i__], 
		&tau[i__], &iinfo);
    }
    work[1] = (double) lwkopt;
    return 0;
    //
    //    End of DSYTRD
    //
} // dsytrd_

