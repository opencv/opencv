#include "clapack.h"

/* Table of constant values */

static real c_b4 = 1.f;
static real c_b5 = 0.f;
static integer c__1 = 1;

/* Subroutine */ int slarf_(char *side, integer *m, integer *n, real *v, 
	integer *incv, real *tau, real *c__, integer *ldc, real *work)
{
    /* System generated locals */
    integer c_dim1, c_offset;
    real r__1;

    /* Local variables */
    extern /* Subroutine */ int sger_(integer *, integer *, real *, real *, 
	    integer *, real *, integer *, real *, integer *);
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *, 
	    real *, integer *, real *, integer *, real *, real *, integer *);


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLARF applies a real elementary reflector H to a real m by n matrix */
/*  C, from either the left or the right. H is represented in the form */

/*        H = I - tau * v * v' */

/*  where tau is a real scalar and v is a real vector. */

/*  If tau = 0, then H is taken to be the unit matrix. */

/*  Arguments */
/*  ========= */

/*  SIDE    (input) CHARACTER*1 */
/*          = 'L': form  H * C */
/*          = 'R': form  C * H */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix C. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix C. */

/*  V       (input) REAL array, dimension */
/*                     (1 + (M-1)*abs(INCV)) if SIDE = 'L' */
/*                  or (1 + (N-1)*abs(INCV)) if SIDE = 'R' */
/*          The vector v in the representation of H. V is not used if */
/*          TAU = 0. */

/*  INCV    (input) INTEGER */
/*          The increment between elements of v. INCV <> 0. */

/*  TAU     (input) REAL */
/*          The value tau in the representation of H. */

/*  C       (input/output) REAL array, dimension (LDC,N) */
/*          On entry, the m by n matrix C. */
/*          On exit, C is overwritten by the matrix H * C if SIDE = 'L', */
/*          or C * H if SIDE = 'R'. */

/*  LDC     (input) INTEGER */
/*          The leading dimension of the array C. LDC >= max(1,M). */

/*  WORK    (workspace) REAL array, dimension */
/*                         (N) if SIDE = 'L' */
/*                      or (M) if SIDE = 'R' */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --v;
    c_dim1 = *ldc;
    c_offset = 1 + c_dim1;
    c__ -= c_offset;
    --work;

    /* Function Body */
    if (lsame_(side, "L")) {

/*        Form  H * C */

	if (*tau != 0.f) {

/*           w := C' * v */

	    sgemv_("Transpose", m, n, &c_b4, &c__[c_offset], ldc, &v[1], incv, 
		     &c_b5, &work[1], &c__1);

/*           C := C - v * w' */

	    r__1 = -(*tau);
	    sger_(m, n, &r__1, &v[1], incv, &work[1], &c__1, &c__[c_offset], 
		    ldc);
	}
    } else {

/*        Form  C * H */

	if (*tau != 0.f) {

/*           w := C * v */

	    sgemv_("No transpose", m, n, &c_b4, &c__[c_offset], ldc, &v[1], 
		    incv, &c_b5, &work[1], &c__1);

/*           C := C - w * v' */

	    r__1 = -(*tau);
	    sger_(m, n, &r__1, &work[1], &c__1, &v[1], incv, &c__[c_offset], 
		    ldc);
	}
    }
    return 0;

/*     End of SLARF */

} /* slarf_ */
