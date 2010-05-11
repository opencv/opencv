#include "clapack.h"

/* Subroutine */ int slarrr_(integer *n, real *d__, real *e, integer *info)
{
    /* System generated locals */
    integer i__1;
    real r__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    integer i__;
    real eps, tmp, tmp2, rmin, offdig;
    extern doublereal slamch_(char *);
    real safmin;
    logical yesrel;
    real smlnum, offdig2;


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */


/*  Purpose */
/*  ======= */

/*  Perform tests to decide whether the symmetric tridiagonal matrix T */
/*  warrants expensive computations which guarantee high relative accuracy */
/*  in the eigenvalues. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix. N > 0. */

/*  D       (input) REAL             array, dimension (N) */
/*          The N diagonal elements of the tridiagonal matrix T. */

/*  E       (input/output) REAL             array, dimension (N) */
/*          On entry, the first (N-1) entries contain the subdiagonal */
/*          elements of the tridiagonal matrix T; E(N) is set to ZERO. */

/*  INFO    (output) INTEGER */
/*          INFO = 0(default) : the matrix warrants computations preserving */
/*                              relative accuracy. */
/*          INFO = 1          : the matrix warrants computations guaranteeing */
/*                              only absolute accuracy. */

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
/*     .. External Functions .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     As a default, do NOT go for relative-accuracy preserving computations. */
    /* Parameter adjustments */
    --e;
    --d__;

    /* Function Body */
    *info = 1;
    safmin = slamch_("Safe minimum");
    eps = slamch_("Precision");
    smlnum = safmin / eps;
    rmin = sqrt(smlnum);
/*     Tests for relative accuracy */

/*     Test for scaled diagonal dominance */
/*     Scale the diagonal entries to one and check whether the sum of the */
/*     off-diagonals is less than one */

/*     The sdd relative error bounds have a 1/(1- 2*x) factor in them, */
/*     x = max(OFFDIG + OFFDIG2), so when x is close to 1/2, no relative */
/*     accuracy is promised.  In the notation of the code fragment below, */
/*     1/(1 - (OFFDIG + OFFDIG2)) is the condition number. */
/*     We don't think it is worth going into "sdd mode" unless the relative */
/*     condition number is reasonable, not 1/macheps. */
/*     The threshold should be compatible with other thresholds used in the */
/*     code. We set  OFFDIG + OFFDIG2 <= .999 =: RELCOND, it corresponds */
/*     to losing at most 3 decimal digits: 1 / (1 - (OFFDIG + OFFDIG2)) <= 1000 */
/*     instead of the current OFFDIG + OFFDIG2 < 1 */

    yesrel = TRUE_;
    offdig = 0.f;
    tmp = sqrt((dabs(d__[1])));
    if (tmp < rmin) {
	yesrel = FALSE_;
    }
    if (! yesrel) {
	goto L11;
    }
    i__1 = *n;
    for (i__ = 2; i__ <= i__1; ++i__) {
	tmp2 = sqrt((r__1 = d__[i__], dabs(r__1)));
	if (tmp2 < rmin) {
	    yesrel = FALSE_;
	}
	if (! yesrel) {
	    goto L11;
	}
	offdig2 = (r__1 = e[i__ - 1], dabs(r__1)) / (tmp * tmp2);
	if (offdig + offdig2 >= .999f) {
	    yesrel = FALSE_;
	}
	if (! yesrel) {
	    goto L11;
	}
	tmp = tmp2;
	offdig = offdig2;
/* L10: */
    }
L11:
    if (yesrel) {
	*info = 0;
	return 0;
    } else {
    }


/*     *** MORE TO BE IMPLEMENTED *** */


/*     Test if the lower bidiagonal matrix L from T = L D L^T */
/*     (zero shift facto) is well conditioned */


/*     Test if the upper bidiagonal matrix U from T = U D U^T */
/*     (zero shift facto) is well conditioned. */
/*     In this case, the matrix needs to be flipped and, at the end */
/*     of the eigenvector computation, the flip needs to be applied */
/*     to the computed eigenvectors (and the support) */


    return 0;

/*     END OF SLARRR */

} /* slarrr_ */
