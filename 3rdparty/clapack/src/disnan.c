/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b DISNAN tests input for NaN.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DISNAN + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/disnan.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/disnan.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/disnan.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      LOGICAL FUNCTION DISNAN( DIN )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION, INTENT(IN) :: DIN
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> DISNAN returns .TRUE. if its argument is NaN, and .FALSE.
//> otherwise.  To be replaced by the Fortran 2003 intrinsic in the
//> future.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] DIN
//> \verbatim
//>          DIN is DOUBLE PRECISION
//>          Input to test for NaN.
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
int disnan_(double *din)
{
    // System generated locals
    int ret_val;

    // Local variables
    extern int dlaisnan_(double *, double *);

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    // .. External Functions ..
    // ..
    // .. Executable Statements ..
    ret_val = dlaisnan_(din, din);
    return ret_val;
} // disnan_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b DLAISNAN tests input for NaN by comparing two arguments for inequality.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download DLAISNAN + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/dlaisnan.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/dlaisnan.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/dlaisnan.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      LOGICAL FUNCTION DLAISNAN( DIN1, DIN2 )
//
//      .. Scalar Arguments ..
//      DOUBLE PRECISION, INTENT(IN) :: DIN1, DIN2
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> This routine is not for general use.  It exists solely to avoid
//> over-optimization in DISNAN.
//>
//> DLAISNAN checks for NaNs by comparing its two arguments for
//> inequality.  NaN is the only floating-point value where NaN != NaN
//> returns .TRUE.  To check for NaNs, pass the same variable as both
//> arguments.
//>
//> A compiler must assume that the two arguments are
//> not the same variable, and the test will not be optimized away.
//> Interprocedural or whole-program optimization may delete this
//> test.  The ISNAN functions will be replaced by the correct
//> Fortran 03 intrinsic once the intrinsic is widely available.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] DIN1
//> \verbatim
//>          DIN1 is DOUBLE PRECISION
//> \endverbatim
//>
//> \param[in] DIN2
//> \verbatim
//>          DIN2 is DOUBLE PRECISION
//>          Two numbers to compare for inequality.
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
int dlaisnan_(double *din1, double *din2)
{
    // System generated locals
    int ret_val;

    //
    // -- LAPACK auxiliary routine (version 3.7.1) --
    // -- LAPACK is a software package provided by Univ. of Tennessee,    --
    // -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
    //    June 2017
    //
    //    .. Scalar Arguments ..
    //    ..
    //
    // =====================================================================
    //
    // .. Executable Statements ..
    ret_val = *din1 != *din2;
    return ret_val;
} // dlaisnan_

