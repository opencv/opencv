/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

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

