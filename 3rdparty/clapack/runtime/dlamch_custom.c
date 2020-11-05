#include "f2c.h"
#include <float.h>
#include <stdio.h>

/* *********************************************************************** */

double dlamc3_(double *a, double *b)
{
    /*  -- LAPACK auxiliary routine (version 3.1) -- */
    /*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
    /*     November 2006 */

    /*     .. Scalar Arguments .. */
    /*     .. */

    /*  Purpose */
    /*  ======= */

    /*  DLAMC3  is intended to force  A  and  B  to be stored prior to doing */
    /*  the addition of  A  and  B ,  for use in situations where optimizers */
    /*  might hold one of these in a register. */

    /*  Arguments */
    /*  ========= */

    /*  A       (input) DOUBLE PRECISION */
    /*  B       (input) DOUBLE PRECISION */
    /*          The values A and B. */

    /* ===================================================================== */

    /*     .. Executable Statements .. */

    double ret_val = *a + *b;

    return ret_val;

    /*     End of DLAMC3 */

} /* dlamc3_ */


/* simpler version of dlamch for the case of IEEE754-compliant FPU module by Piotr Luszczek S.
   taken from http://www.mail-archive.com/numpy-discussion@lists.sourceforge.net/msg02448.html */

#ifndef DBL_DIGITS
#define DBL_DIGITS 53
#endif

static const unsigned char lapack_dlamch_tab0[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 0, 8, 9, 0, 10, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 7, 0, 8, 9,
    0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

const double lapack_dlamch_tab1[] =
{
    0, FLT_RADIX, DBL_EPSILON, DBL_MAX_EXP, DBL_MIN_EXP, DBL_DIGITS, DBL_MAX,
    DBL_EPSILON*FLT_RADIX, 1, DBL_MIN*(1 + DBL_EPSILON), DBL_MIN
};

double dlamch_(char* cmach)
{
    return lapack_dlamch_tab1[lapack_dlamch_tab0[(unsigned char)cmach[0]]];
}
