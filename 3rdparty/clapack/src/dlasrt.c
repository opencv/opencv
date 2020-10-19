/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DLASRT sorts numbers in increasing or decreasing order.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLASRT + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlasrt.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlasrt.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlasrt.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      SUBROUTINE DLASRT( ID, N, D, INFO )
//
//      .. Scalar Arguments ..
//      CHARACTER          ID
//      INTEGER            INFO, N
//      ..
//      .. Array Arguments ..
//      DOUBLE PRECISION   D( * )
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> Sort the numbers in D in increasing order (if ID = 'I') or
//> in decreasing order (if ID = 'D' ).
//>
//> Use Quick Sort, reverting to Insertion sort on arrays of
//> size <= 20. Dimension of STACK limits N to about 2**32.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] ID
//> \verbatim
//>          ID is CHARACTER*1
//>          = 'I': sort D in increasing order;
//>          = 'D': sort D in decreasing order.
//> \endverbatim
//>
//> \param[in] N
//> \verbatim
//>          N is INTEGER
//>          The length of the array D.
//> \endverbatim
//>
//> \param[in,out] D
//> \verbatim
//>          D is DOUBLE PRECISION array, dimension (N)
//>          On entry, the array to be sorted.
//>          On exit, D has been sorted into increasing order
//>          (D(1) <= ... <= D(N) ) or into decreasing order
//>          (D(1) >= ... >= D(N) ), depending on ID.
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
//> \date June 2016
//
//> \ingroup auxOTHERcomputational
//
// =====================================================================
/* Subroutine */ int dlasrt_(char *id, int *n, double *d__, int *info)
{
    // System generated locals
    int i__1, i__2;

    // Local variables
    int i__, j;
    double d1, d2, d3;
    int dir;
    double tmp;
    int endd;
    extern int lsame_(char *, char *);
    int stack[64]	/* was [2][32] */;
    double dmnmx;
    int start;
    extern /* Subroutine */ int xerbla_(char *, int *);
    int stkpnt;

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
    //    .. Executable Statements ..
    //
    //    Test the input parameters.
    //
    // Parameter adjustments
    --d__;

    // Function Body
    *info = 0;
    dir = -1;
    if (lsame_(id, "D")) {
	dir = 0;
    } else if (lsame_(id, "I")) {
	dir = 1;
    }
    if (dir == -1) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DLASRT", &i__1);
	return 0;
    }
    //
    //    Quick return if possible
    //
    if (*n <= 1) {
	return 0;
    }
    stkpnt = 1;
    stack[0] = 1;
    stack[1] = *n;
L10:
    start = stack[(stkpnt << 1) - 2];
    endd = stack[(stkpnt << 1) - 1];
    --stkpnt;
    if (endd - start <= 20 && endd - start > 0) {
	//
	//       Do Insertion sort on D( START:ENDD )
	//
	if (dir == 0) {
	    //
	    //          Sort into decreasing order
	    //
	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] > d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L30;
		    }
// L20:
		}
L30:
		;
	    }
	} else {
	    //
	    //          Sort into increasing order
	    //
	    i__1 = endd;
	    for (i__ = start + 1; i__ <= i__1; ++i__) {
		i__2 = start + 1;
		for (j = i__; j >= i__2; --j) {
		    if (d__[j] < d__[j - 1]) {
			dmnmx = d__[j];
			d__[j] = d__[j - 1];
			d__[j - 1] = dmnmx;
		    } else {
			goto L50;
		    }
// L40:
		}
L50:
		;
	    }
	}
    } else if (endd - start > 20) {
	//
	//       Partition D( START:ENDD ) and stack parts, largest one first
	//
	//       Choose partition entry as median of 3
	//
	d1 = d__[start];
	d2 = d__[endd];
	i__ = (start + endd) / 2;
	d3 = d__[i__];
	if (d1 < d2) {
	    if (d3 < d1) {
		dmnmx = d1;
	    } else if (d3 < d2) {
		dmnmx = d3;
	    } else {
		dmnmx = d2;
	    }
	} else {
	    if (d3 < d2) {
		dmnmx = d2;
	    } else if (d3 < d1) {
		dmnmx = d3;
	    } else {
		dmnmx = d1;
	    }
	}
	if (dir == 0) {
	    //
	    //          Sort into decreasing order
	    //
	    i__ = start - 1;
	    j = endd + 1;
L60:
L70:
	    --j;
	    if (d__[j] < dmnmx) {
		goto L70;
	    }
L80:
	    ++i__;
	    if (d__[i__] > dmnmx) {
		goto L80;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L60;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
	    } else {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
	    }
	} else {
	    //
	    //          Sort into increasing order
	    //
	    i__ = start - 1;
	    j = endd + 1;
L90:
L100:
	    --j;
	    if (d__[j] > dmnmx) {
		goto L100;
	    }
L110:
	    ++i__;
	    if (d__[i__] < dmnmx) {
		goto L110;
	    }
	    if (i__ < j) {
		tmp = d__[i__];
		d__[i__] = d__[j];
		d__[j] = tmp;
		goto L90;
	    }
	    if (j - start > endd - j - 1) {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
	    } else {
		++stkpnt;
		stack[(stkpnt << 1) - 2] = j + 1;
		stack[(stkpnt << 1) - 1] = endd;
		++stkpnt;
		stack[(stkpnt << 1) - 2] = start;
		stack[(stkpnt << 1) - 1] = j;
	    }
	}
    }
    if (stkpnt > 0) {
	goto L10;
    }
    return 0;
    //
    //    End of DLASRT
    //
} // dlasrt_

