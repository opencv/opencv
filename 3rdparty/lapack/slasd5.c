/* slasd5.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int slasd5_(integer *i__, real *d__, real *z__, real *delta, 
	real *rho, real *dsigma, real *work)
{
    /* System generated locals */
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    real b, c__, w, del, tau, delsq;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  This subroutine computes the square root of the I-th eigenvalue */
/*  of a positive symmetric rank-one modification of a 2-by-2 diagonal */
/*  matrix */

/*             diag( D ) * diag( D ) +  RHO *  Z * transpose(Z) . */

/*  The diagonal entries in the array D are assumed to satisfy */

/*             0 <= D(i) < D(j)  for  i < j . */

/*  We also assume RHO > 0 and that the Euclidean norm of the vector */
/*  Z is one. */

/*  Arguments */
/*  ========= */

/*  I      (input) INTEGER */
/*         The index of the eigenvalue to be computed.  I = 1 or I = 2. */

/*  D      (input) REAL array, dimension (2) */
/*         The original eigenvalues.  We assume 0 <= D(1) < D(2). */

/*  Z      (input) REAL array, dimension (2) */
/*         The components of the updating vector. */

/*  DELTA  (output) REAL array, dimension (2) */
/*         Contains (D(j) - sigma_I) in its  j-th component. */
/*         The vector DELTA contains the information necessary */
/*         to construct the eigenvectors. */

/*  RHO    (input) REAL */
/*         The scalar in the symmetric updating formula. */

/*  DSIGMA (output) REAL */
/*         The computed sigma_I, the I-th updated eigenvalue. */

/*  WORK   (workspace) REAL array, dimension (2) */
/*         WORK contains (D(j) + sigma_I) in its  j-th component. */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Ren-Cang Li, Computer Science Division, University of California */
/*     at Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --work;
    --delta;
    --z__;
    --d__;

    /* Function Body */
    del = d__[2] - d__[1];
    delsq = del * (d__[2] + d__[1]);
    if (*i__ == 1) {
	w = *rho * 4.f * (z__[2] * z__[2] / (d__[1] + d__[2] * 3.f) - z__[1] *
		 z__[1] / (d__[1] * 3.f + d__[2])) / del + 1.f;
	if (w > 0.f) {
	    b = delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[1] * z__[1] * delsq;

/*           B > ZERO, always */

/*           The following TAU is DSIGMA * DSIGMA - D( 1 ) * D( 1 ) */

	    tau = c__ * 2.f / (b + sqrt((r__1 = b * b - c__ * 4.f, dabs(r__1))
		    ));

/*           The following TAU is DSIGMA - D( 1 ) */

	    tau /= d__[1] + sqrt(d__[1] * d__[1] + tau);
	    *dsigma = d__[1] + tau;
	    delta[1] = -tau;
	    delta[2] = del - tau;
	    work[1] = d__[1] * 2.f + tau;
	    work[2] = d__[1] + tau + d__[2];
/*           DELTA( 1 ) = -Z( 1 ) / TAU */
/*           DELTA( 2 ) = Z( 2 ) / ( DEL-TAU ) */
	} else {
	    b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	    c__ = *rho * z__[2] * z__[2] * delsq;

/*           The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 ) */

	    if (b > 0.f) {
		tau = c__ * -2.f / (b + sqrt(b * b + c__ * 4.f));
	    } else {
		tau = (b - sqrt(b * b + c__ * 4.f)) / 2.f;
	    }

/*           The following TAU is DSIGMA - D( 2 ) */

	    tau /= d__[2] + sqrt((r__1 = d__[2] * d__[2] + tau, dabs(r__1)));
	    *dsigma = d__[2] + tau;
	    delta[1] = -(del + tau);
	    delta[2] = -tau;
	    work[1] = d__[1] + tau + d__[2];
	    work[2] = d__[2] * 2.f + tau;
/*           DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU ) */
/*           DELTA( 2 ) = -Z( 2 ) / TAU */
	}
/*        TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) ) */
/*        DELTA( 1 ) = DELTA( 1 ) / TEMP */
/*        DELTA( 2 ) = DELTA( 2 ) / TEMP */
    } else {

/*        Now I=2 */

	b = -delsq + *rho * (z__[1] * z__[1] + z__[2] * z__[2]);
	c__ = *rho * z__[2] * z__[2] * delsq;

/*        The following TAU is DSIGMA * DSIGMA - D( 2 ) * D( 2 ) */

	if (b > 0.f) {
	    tau = (b + sqrt(b * b + c__ * 4.f)) / 2.f;
	} else {
	    tau = c__ * 2.f / (-b + sqrt(b * b + c__ * 4.f));
	}

/*        The following TAU is DSIGMA - D( 2 ) */

	tau /= d__[2] + sqrt(d__[2] * d__[2] + tau);
	*dsigma = d__[2] + tau;
	delta[1] = -(del + tau);
	delta[2] = -tau;
	work[1] = d__[1] + tau + d__[2];
	work[2] = d__[2] * 2.f + tau;
/*        DELTA( 1 ) = -Z( 1 ) / ( DEL+TAU ) */
/*        DELTA( 2 ) = -Z( 2 ) / TAU */
/*        TEMP = SQRT( DELTA( 1 )*DELTA( 1 )+DELTA( 2 )*DELTA( 2 ) ) */
/*        DELTA( 1 ) = DELTA( 1 ) / TEMP */
/*        DELTA( 2 ) = DELTA( 2 ) / TEMP */
    }
    return 0;

/*     End of SLASD5 */

} /* slasd5_ */
