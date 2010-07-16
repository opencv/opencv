/* slasq5.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int slasq5_(integer *i0, integer *n0, real *z__, integer *pp, 
	 real *tau, real *dmin__, real *dmin1, real *dmin2, real *dn, real *
	dnm1, real *dnm2, logical *ieee)
{
    /* System generated locals */
    integer i__1;
    real r__1, r__2;

    /* Local variables */
    real d__;
    integer j4, j4p2;
    real emin, temp;


/*  -- LAPACK routine (version 3.2)                                    -- */

/*  -- Contributed by Osni Marques of the Lawrence Berkeley National   -- */
/*  -- Laboratory and Beresford Parlett of the Univ. of California at  -- */
/*  -- Berkeley                                                        -- */
/*  -- November 2008                                                   -- */

/*  -- LAPACK is a software package provided by Univ. of Tennessee,    -- */
/*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..-- */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLASQ5 computes one dqds transform in ping-pong form, one */
/*  version for IEEE machines another for non IEEE machines. */

/*  Arguments */
/*  ========= */

/*  I0    (input) INTEGER */
/*        First index. */

/*  N0    (input) INTEGER */
/*        Last index. */

/*  Z     (input) REAL array, dimension ( 4*N ) */
/*        Z holds the qd array. EMIN is stored in Z(4*N0) to avoid */
/*        an extra argument. */

/*  PP    (input) INTEGER */
/*        PP=0 for ping, PP=1 for pong. */

/*  TAU   (input) REAL */
/*        This is the shift. */

/*  DMIN  (output) REAL */
/*        Minimum value of d. */

/*  DMIN1 (output) REAL */
/*        Minimum value of d, excluding D( N0 ). */

/*  DMIN2 (output) REAL */
/*        Minimum value of d, excluding D( N0 ) and D( N0-1 ). */

/*  DN    (output) REAL */
/*        d(N0), the last value of d. */

/*  DNM1  (output) REAL */
/*        d(N0-1). */

/*  DNM2  (output) REAL */
/*        d(N0-2). */

/*  IEEE  (input) LOGICAL */
/*        Flag for IEEE or non IEEE arithmetic. */

/*  ===================================================================== */

/*     .. Parameter .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --z__;

    /* Function Body */
    if (*n0 - *i0 - 1 <= 0) {
	return 0;
    }

    j4 = (*i0 << 2) + *pp - 3;
    emin = z__[j4 + 4];
    d__ = z__[j4] - *tau;
    *dmin__ = d__;
    *dmin1 = -z__[j4];

    if (*ieee) {

/*        Code for IEEE arithmetic. */

	if (*pp == 0) {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 2] = d__ + z__[j4 - 1];
		temp = z__[j4 + 1] / z__[j4 - 2];
		d__ = d__ * temp - *tau;
		*dmin__ = dmin(*dmin__,d__);
		z__[j4] = z__[j4 - 1] * temp;
/* Computing MIN */
		r__1 = z__[j4];
		emin = dmin(r__1,emin);
/* L10: */
	    }
	} else {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 3] = d__ + z__[j4];
		temp = z__[j4 + 2] / z__[j4 - 3];
		d__ = d__ * temp - *tau;
		*dmin__ = dmin(*dmin__,d__);
		z__[j4 - 1] = z__[j4] * temp;
/* Computing MIN */
		r__1 = z__[j4 - 1];
		emin = dmin(r__1,emin);
/* L20: */
	    }
	}

/*        Unroll last two steps. */

	*dnm2 = d__;
	*dmin2 = *dmin__;
	j4 = (*n0 - 2 << 2) - *pp;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm2 + z__[j4p2];
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	*dmin__ = dmin(*dmin__,*dnm1);

	*dmin1 = *dmin__;
	j4 += 4;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm1 + z__[j4p2];
	z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	*dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	*dmin__ = dmin(*dmin__,*dn);

    } else {

/*        Code for non IEEE arithmetic. */

	if (*pp == 0) {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 2] = d__ + z__[j4 - 1];
		if (d__ < 0.f) {
		    return 0;
		} else {
		    z__[j4] = z__[j4 + 1] * (z__[j4 - 1] / z__[j4 - 2]);
		    d__ = z__[j4 + 1] * (d__ / z__[j4 - 2]) - *tau;
		}
		*dmin__ = dmin(*dmin__,d__);
/* Computing MIN */
		r__1 = emin, r__2 = z__[j4];
		emin = dmin(r__1,r__2);
/* L30: */
	    }
	} else {
	    i__1 = *n0 - 3 << 2;
	    for (j4 = *i0 << 2; j4 <= i__1; j4 += 4) {
		z__[j4 - 3] = d__ + z__[j4];
		if (d__ < 0.f) {
		    return 0;
		} else {
		    z__[j4 - 1] = z__[j4 + 2] * (z__[j4] / z__[j4 - 3]);
		    d__ = z__[j4 + 2] * (d__ / z__[j4 - 3]) - *tau;
		}
		*dmin__ = dmin(*dmin__,d__);
/* Computing MIN */
		r__1 = emin, r__2 = z__[j4 - 1];
		emin = dmin(r__1,r__2);
/* L40: */
	    }
	}

/*        Unroll last two steps. */

	*dnm2 = d__;
	*dmin2 = *dmin__;
	j4 = (*n0 - 2 << 2) - *pp;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm2 + z__[j4p2];
	if (*dnm2 < 0.f) {
	    return 0;
	} else {
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dnm1 = z__[j4p2 + 2] * (*dnm2 / z__[j4 - 2]) - *tau;
	}
	*dmin__ = dmin(*dmin__,*dnm1);

	*dmin1 = *dmin__;
	j4 += 4;
	j4p2 = j4 + (*pp << 1) - 1;
	z__[j4 - 2] = *dnm1 + z__[j4p2];
	if (*dnm1 < 0.f) {
	    return 0;
	} else {
	    z__[j4] = z__[j4p2 + 2] * (z__[j4p2] / z__[j4 - 2]);
	    *dn = z__[j4p2 + 2] * (*dnm1 / z__[j4 - 2]) - *tau;
	}
	*dmin__ = dmin(*dmin__,*dn);

    }

    z__[j4 + 2] = *dn;
    z__[(*n0 << 2) - *pp] = emin;
    return 0;

/*     End of SLASQ5 */

} /* slasq5_ */
