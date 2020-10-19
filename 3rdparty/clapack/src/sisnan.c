/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

#include "f2c.h"

//> \brief \b SISNAN tests input for NaN.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SISNAN + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/sisnan.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/sisnan.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/sisnan.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      LOGICAL FUNCTION SISNAN( SIN )
//
//      .. Scalar Arguments ..
//      REAL, INTENT(IN) :: SIN
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> SISNAN returns .TRUE. if its argument is NaN, and .FALSE.
//> otherwise.  To be replaced by the Fortran 2003 intrinsic in the
//> future.
//> \endverbatim
//
// Arguments:
// ==========
//
//> \param[in] SIN
//> \verbatim
//>          SIN is REAL
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
int sisnan_(float *sin__)
{
    // System generated locals
    int ret_val;

    // Local variables
    extern int slaisnan_(float *, float *);

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
    ret_val = slaisnan_(sin__, sin__);
    return ret_val;
} // sisnan_

/*  -- translated by f2c (version 20201020 (for_lapack)). -- */

//> \brief \b SLAISNAN tests input for NaN by comparing two arguments for inequality.
//
// =========== DOCUMENTATION ===========
//
// Online html documentation available at
//           http://www.netlib.org/lapack/explore-html/
//
//> \htmlonly
//> Download SLAISNAN + dependencies
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.tgz?format=tgz&filename=/lapack/lapack_routine/slaisnan.f">
//> [TGZ]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.zip?format=zip&filename=/lapack/lapack_routine/slaisnan.f">
//> [ZIP]</a>
//> <a href="http://www.netlib.org/cgi-bin/netlibfiles.txt?format=txt&filename=/lapack/lapack_routine/slaisnan.f">
//> [TXT]</a>
//> \endhtmlonly
//
// Definition:
// ===========
//
//      LOGICAL FUNCTION SLAISNAN( SIN1, SIN2 )
//
//      .. Scalar Arguments ..
//      REAL, INTENT(IN) :: SIN1, SIN2
//      ..
//
//
//> \par Purpose:
// =============
//>
//> \verbatim
//>
//> This routine is not for general use.  It exists solely to avoid
//> over-optimization in SISNAN.
//>
//> SLAISNAN checks for NaNs by comparing its two arguments for
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
//> \param[in] SIN1
//> \verbatim
//>          SIN1 is REAL
//> \endverbatim
//>
//> \param[in] SIN2
//> \verbatim
//>          SIN2 is REAL
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
int slaisnan_(float *sin1, float *sin2)
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
    ret_val = *sin1 != *sin2;
    return ret_val;
} // slaisnan_

