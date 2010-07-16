/* slamrg.f -- translated by f2c (version 20061008).
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


/* Subroutine */ int slamrg_(integer *n1, integer *n2, real *a, integer *
	strd1, integer *strd2, integer *index)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    integer i__, ind1, ind2, n1sv, n2sv;


/*  -- LAPACK routine (version 3.2) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLAMRG will create a permutation list which will merge the elements */
/*  of A (which is composed of two independently sorted sets) into a */
/*  single set which is sorted in ascending order. */

/*  Arguments */
/*  ========= */

/*  N1     (input) INTEGER */
/*  N2     (input) INTEGER */
/*         These arguements contain the respective lengths of the two */
/*         sorted lists to be merged. */

/*  A      (input) REAL array, dimension (N1+N2) */
/*         The first N1 elements of A contain a list of numbers which */
/*         are sorted in either ascending or descending order.  Likewise */
/*         for the final N2 elements. */

/*  STRD1  (input) INTEGER */
/*  STRD2  (input) INTEGER */
/*         These are the strides to be taken through the array A. */
/*         Allowable strides are 1 and -1.  They indicate whether a */
/*         subset of A is sorted in ascending (STRDx = 1) or descending */
/*         (STRDx = -1) order. */

/*  INDEX  (output) INTEGER array, dimension (N1+N2) */
/*         On exit this array will contain a permutation such that */
/*         if B( I ) = A( INDEX( I ) ) for I=1,N1+N2, then B will be */
/*         sorted in ascending order. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --index;
    --a;

    /* Function Body */
    n1sv = *n1;
    n2sv = *n2;
    if (*strd1 > 0) {
	ind1 = 1;
    } else {
	ind1 = *n1;
    }
    if (*strd2 > 0) {
	ind2 = *n1 + 1;
    } else {
	ind2 = *n1 + *n2;
    }
    i__ = 1;
/*     while ( (N1SV > 0) & (N2SV > 0) ) */
L10:
    if (n1sv > 0 && n2sv > 0) {
	if (a[ind1] <= a[ind2]) {
	    index[i__] = ind1;
	    ++i__;
	    ind1 += *strd1;
	    --n1sv;
	} else {
	    index[i__] = ind2;
	    ++i__;
	    ind2 += *strd2;
	    --n2sv;
	}
	goto L10;
    }
/*     end while */
    if (n1sv == 0) {
	i__1 = n2sv;
	for (n1sv = 1; n1sv <= i__1; ++n1sv) {
	    index[i__] = ind2;
	    ++i__;
	    ind2 += *strd2;
/* L20: */
	}
    } else {
/*     N2SV .EQ. 0 */
	i__1 = n1sv;
	for (n2sv = 1; n2sv <= i__1; ++n2sv) {
	    index[i__] = ind1;
	    ++i__;
	    ind1 += *strd1;
/* L30: */
	}
    }

    return 0;

/*     End of SLAMRG */

} /* slamrg_ */
