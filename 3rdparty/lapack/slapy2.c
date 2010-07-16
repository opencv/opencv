/* slapy2.f -- translated by f2c (version 20061008).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "clapack.h"


doublereal slapy2_(real *x, real *y)
{
    /* System generated locals */
    real ret_val, r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    real w, z__, xabs, yabs;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAPY2 returns sqrt(x**2+y**2), taking care not to cause unnecessary */
/*  overflow. */

/*  Arguments */
/*  ========= */

/*  X       (input) REAL */
/*  Y       (input) REAL */
/*          X and Y specify the values x and y. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    xabs = dabs(*x);
    yabs = dabs(*y);
    w = dmax(xabs,yabs);
    z__ = dmin(xabs,yabs);
    if (z__ == 0.f) {
	ret_val = w;
    } else {
/* Computing 2nd power */
	r__1 = z__ / w;
	ret_val = w * sqrt(r__1 * r__1 + 1.f);
    }
    return ret_val;

/*     End of SLAPY2 */

} /* slapy2_ */
