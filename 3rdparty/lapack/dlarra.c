/* dlarra.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int dlarra_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *e2, doublereal *spltol, doublereal *tnrm, integer *nsplit, 
	 integer *isplit, integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__;
    doublereal tmp1, eabs;


/*  -- LAPACK auxiliary routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  Compute the splitting points with threshold SPLTOL. */
/*  DLARRA sets any "small" off-diagonal elements to zero. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix. N > 0. */

/*  D       (input) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the N diagonal elements of the tridiagonal */
/*          matrix T. */

/*  E       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the first (N-1) entries contain the subdiagonal */
/*          elements of the tridiagonal matrix T; E(N) need not be set. */
/*          On exit, the entries E( ISPLIT( I ) ), 1 <= I <= NSPLIT, */
/*          are set to zero, the other entries of E are untouched. */

/*  E2      (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, the first (N-1) entries contain the SQUARES of the */
/*          subdiagonal elements of the tridiagonal matrix T; */
/*          E2(N) need not be set. */
/*          On exit, the entries E2( ISPLIT( I ) ), */
/*          1 <= I <= NSPLIT, have been set to zero */

/*  SPLTOL (input) DOUBLE PRECISION */
/*          The threshold for splitting. Two criteria can be used: */
/*          SPLTOL<0 : criterion based on absolute off-diagonal value */
/*          SPLTOL>0 : criterion that preserves relative accuracy */

/*  TNRM (input) DOUBLE PRECISION */
/*          The norm of the matrix. */

/*  NSPLIT  (output) INTEGER */
/*          The number of blocks T splits into. 1 <= NSPLIT <= N. */

/*  ISPLIT  (output) INTEGER array, dimension (N) */
/*          The splitting points, at which T breaks up into blocks. */
/*          The first block consists of rows/columns 1 to ISPLIT(1), */
/*          the second of rows/columns ISPLIT(1)+1 through ISPLIT(2), */
/*          etc., and the NSPLIT-th consists of rows/columns */
/*          ISPLIT(NSPLIT-1)+1 through ISPLIT(NSPLIT)=N. */


/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */

/*  Further Details */
/*  =============== */

/*  Based on contributions by */
/*     Beresford Parlett, University of California, Berkeley, USA */
/*     Jim Demmel, University of California, Berkeley, USA */
/*     Inderjit Dhillon, University of Texas, Austin, USA */
/*     Osni Marques, LBNL/NERSC, USA */
/*     Christof Voemel, University of California, Berkeley, USA */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --isplit;
    --e2;
    --e;
    --d__;

    /* Function Body */
    *info = 0;
/*     Compute splitting points */
    *nsplit = 1;
    if (*spltol < 0.) {
/*        Criterion based on absolute off-diagonal value */
	tmp1 = abs(*spltol) * *tnrm;
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    eabs = (d__1 = e[i__], abs(d__1));
	    if (eabs <= tmp1) {
		e[i__] = 0.;
		e2[i__] = 0.;
		isplit[*nsplit] = i__;
		++(*nsplit);
	    }
/* L9: */
	}
    } else {
/*        Criterion that guarantees relative accuracy */
	i__1 = *n - 1;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    eabs = (d__1 = e[i__], abs(d__1));
	    if (eabs <= *spltol * sqrt((d__1 = d__[i__], abs(d__1))) * sqrt((
		    d__2 = d__[i__ + 1], abs(d__2)))) {
		e[i__] = 0.;
		e2[i__] = 0.;
		isplit[*nsplit] = i__;
		++(*nsplit);
	    }
/* L10: */
	}
    }
    isplit[*nsplit] = *n;
    return 0;

/*     End of DLARRA */

} /* dlarra_ */
