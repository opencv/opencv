#include "clapack.h"

logical dlaisnan_(doublereal *din1, doublereal *din2)
{
    /* System generated locals */
    logical ret_val;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This routine is not for general use.  It exists solely to avoid */
/*  over-optimization in DISNAN. */

/*  DLAISNAN checks for NaNs by comparing its two arguments for */
/*  inequality.  NaN is the only floating-point value where NaN != NaN */
/*  returns .TRUE.  To check for NaNs, pass the same variable as both */
/*  arguments. */

/*  Strictly speaking, Fortran does not allow aliasing of function */
/*  arguments. So a compiler must assume that the two arguments are */
/*  not the same variable, and the test will not be optimized away. */
/*  Interprocedural or whole-program optimization may delete this */
/*  test.  The ISNAN functions will be replaced by the correct */
/*  Fortran 03 intrinsic once the intrinsic is widely available. */

/*  Arguments */
/*  ========= */

/*  DIN1     (input) DOUBLE PRECISION */
/*  DIN2     (input) DOUBLE PRECISION */
/*          Two numbers to compare for inequality. */

/*  ===================================================================== */

/*  .. Executable Statements .. */
    ret_val = *din1 != *din2;
    return ret_val;
} /* dlaisnan_ */
