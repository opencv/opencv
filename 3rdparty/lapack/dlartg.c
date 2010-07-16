/* dlartg.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int dlartg_(doublereal *f, doublereal *g, doublereal *cs, 
	doublereal *sn, doublereal *r__)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    double log(doublereal), pow_di(doublereal *, integer *), sqrt(doublereal);

    /* Local variables */
    integer i__;
    doublereal f1, g1, eps, scale;
    integer count;
    doublereal safmn2, safmx2;
    extern doublereal dlamch_(char *);
    doublereal safmin;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLARTG generate a plane rotation so that */

/*     [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS**2 + SN**2 = 1. */
/*     [ -SN  CS  ]     [ G ]     [ 0 ] */

/*  This is a slower, more accurate version of the BLAS1 routine DROTG, */
/*  with the following other differences: */
/*     F and G are unchanged on return. */
/*     If G=0, then CS=1 and SN=0. */
/*     If F=0 and (G .ne. 0), then CS=0 and SN=1 without doing any */
/*        floating point operations (saves work in DBDSQR when */
/*        there are zeros on the diagonal). */

/*  If F exceeds G in magnitude, CS will be positive. */

/*  Arguments */
/*  ========= */

/*  F       (input) DOUBLE PRECISION */
/*          The first component of vector to be rotated. */

/*  G       (input) DOUBLE PRECISION */
/*          The second component of vector to be rotated. */

/*  CS      (output) DOUBLE PRECISION */
/*          The cosine of the rotation. */

/*  SN      (output) DOUBLE PRECISION */
/*          The sine of the rotation. */

/*  R       (output) DOUBLE PRECISION */
/*          The nonzero component of the rotated vector. */

/*  This version has a few statements commented out for thread safety */
/*  (machine parameters are computed on each entry). 10 feb 03, SJH. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     LOGICAL            FIRST */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Save statement .. */
/*     SAVE               FIRST, SAFMX2, SAFMIN, SAFMN2 */
/*     .. */
/*     .. Data statements .. */
/*     DATA               FIRST / .TRUE. / */
/*     .. */
/*     .. Executable Statements .. */

/*     IF( FIRST ) THEN */
    safmin = dlamch_("S");
    eps = dlamch_("E");
    d__1 = dlamch_("B");
    i__1 = (integer) (log(safmin / eps) / log(dlamch_("B")) / 2.);
    safmn2 = pow_di(&d__1, &i__1);
    safmx2 = 1. / safmn2;
/*        FIRST = .FALSE. */
/*     END IF */
    if (*g == 0.) {
	*cs = 1.;
	*sn = 0.;
	*r__ = *f;
    } else if (*f == 0.) {
	*cs = 0.;
	*sn = 1.;
	*r__ = *g;
    } else {
	f1 = *f;
	g1 = *g;
/* Computing MAX */
	d__1 = abs(f1), d__2 = abs(g1);
	scale = max(d__1,d__2);
	if (scale >= safmx2) {
	    count = 0;
L10:
	    ++count;
	    f1 *= safmn2;
	    g1 *= safmn2;
/* Computing MAX */
	    d__1 = abs(f1), d__2 = abs(g1);
	    scale = max(d__1,d__2);
	    if (scale >= safmx2) {
		goto L10;
	    }
/* Computing 2nd power */
	    d__1 = f1;
/* Computing 2nd power */
	    d__2 = g1;
	    *r__ = sqrt(d__1 * d__1 + d__2 * d__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmx2;
/* L20: */
	    }
	} else if (scale <= safmn2) {
	    count = 0;
L30:
	    ++count;
	    f1 *= safmx2;
	    g1 *= safmx2;
/* Computing MAX */
	    d__1 = abs(f1), d__2 = abs(g1);
	    scale = max(d__1,d__2);
	    if (scale <= safmn2) {
		goto L30;
	    }
/* Computing 2nd power */
	    d__1 = f1;
/* Computing 2nd power */
	    d__2 = g1;
	    *r__ = sqrt(d__1 * d__1 + d__2 * d__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	    i__1 = count;
	    for (i__ = 1; i__ <= i__1; ++i__) {
		*r__ *= safmn2;
/* L40: */
	    }
	} else {
/* Computing 2nd power */
	    d__1 = f1;
/* Computing 2nd power */
	    d__2 = g1;
	    *r__ = sqrt(d__1 * d__1 + d__2 * d__2);
	    *cs = f1 / *r__;
	    *sn = g1 / *r__;
	}
	if (abs(*f) > abs(*g) && *cs < 0.) {
	    *cs = -(*cs);
	    *sn = -(*sn);
	    *r__ = -(*r__);
	}
    }
    return 0;

/*     End of DLARTG */

} /* dlartg_ */
