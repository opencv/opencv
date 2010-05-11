#include "clapack.h"

/* Table of constant values */

static doublereal c_b4 = -1.;
static doublereal c_b5 = 1.;
static integer c__1 = 1;
static doublereal c_b16 = 0.;

/* Subroutine */ int dlabrd_(integer *m, integer *n, integer *nb, doublereal *
	a, integer *lda, doublereal *d__, doublereal *e, doublereal *tauq, 
	doublereal *taup, doublereal *x, integer *ldx, doublereal *y, integer 
	*ldy)
{
    /* System generated locals */
    integer a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__1, i__2, 
	    i__3;

    /* Local variables */
    integer i__;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *), dgemv_(char *, integer *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, integer *), dlarfg_(integer *, doublereal *, 
	     doublereal *, integer *, doublereal *);


/*  -- LAPACK auxiliary routine (version 3.1) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley and NAG Ltd.. */
/*     November 2006 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DLABRD reduces the first NB rows and columns of a real general */
/*  m by n matrix A to upper or lower bidiagonal form by an orthogonal */
/*  transformation Q' * A * P, and returns the matrices X and Y which */
/*  are needed to apply the transformation to the unreduced part of A. */

/*  If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower */
/*  bidiagonal form. */

/*  This is an auxiliary routine called by DGEBRD */

/*  Arguments */
/*  ========= */

/*  M       (input) INTEGER */
/*          The number of rows in the matrix A. */

/*  N       (input) INTEGER */
/*          The number of columns in the matrix A. */

/*  NB      (input) INTEGER */
/*          The number of leading rows and columns of A to be reduced. */

/*  A       (input/output) DOUBLE PRECISION array, dimension (LDA,N) */
/*          On entry, the m by n general matrix to be reduced. */
/*          On exit, the first NB rows and columns of the matrix are */
/*          overwritten; the rest of the array is unchanged. */
/*          If m >= n, elements on and below the diagonal in the first NB */
/*            columns, with the array TAUQ, represent the orthogonal */
/*            matrix Q as a product of elementary reflectors; and */
/*            elements above the diagonal in the first NB rows, with the */
/*            array TAUP, represent the orthogonal matrix P as a product */
/*            of elementary reflectors. */
/*          If m < n, elements below the diagonal in the first NB */
/*            columns, with the array TAUQ, represent the orthogonal */
/*            matrix Q as a product of elementary reflectors, and */
/*            elements on and above the diagonal in the first NB rows, */
/*            with the array TAUP, represent the orthogonal matrix P as */
/*            a product of elementary reflectors. */
/*          See Further Details. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  D       (output) DOUBLE PRECISION array, dimension (NB) */
/*          The diagonal elements of the first NB rows and columns of */
/*          the reduced matrix.  D(i) = A(i,i). */

/*  E       (output) DOUBLE PRECISION array, dimension (NB) */
/*          The off-diagonal elements of the first NB rows and columns of */
/*          the reduced matrix. */

/*  TAUQ    (output) DOUBLE PRECISION array dimension (NB) */
/*          The scalar factors of the elementary reflectors which */
/*          represent the orthogonal matrix Q. See Further Details. */

/*  TAUP    (output) DOUBLE PRECISION array, dimension (NB) */
/*          The scalar factors of the elementary reflectors which */
/*          represent the orthogonal matrix P. See Further Details. */

/*  X       (output) DOUBLE PRECISION array, dimension (LDX,NB) */
/*          The m-by-nb matrix X required to update the unreduced part */
/*          of A. */

/*  LDX     (input) INTEGER */
/*          The leading dimension of the array X. LDX >= M. */

/*  Y       (output) DOUBLE PRECISION array, dimension (LDY,NB) */
/*          The n-by-nb matrix Y required to update the unreduced part */
/*          of A. */

/*  LDY     (input) INTEGER */
/*          The leading dimension of the array Y. LDY >= N. */

/*  Further Details */
/*  =============== */

/*  The matrices Q and P are represented as products of elementary */
/*  reflectors: */

/*     Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb) */

/*  Each H(i) and G(i) has the form: */

/*     H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u' */

/*  where tauq and taup are real scalars, and v and u are real vectors. */

/*  If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in */
/*  A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in */
/*  A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i). */

/*  If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in */
/*  A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in */
/*  A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i). */

/*  The elements of the vectors v and u together form the m-by-nb matrix */
/*  V and the nb-by-n matrix U' which are needed, with X and Y, to apply */
/*  the transformation to the unreduced part of the matrix, using a block */
/*  update of the form:  A := A - V*Y' - X*U'. */

/*  The contents of A on exit are illustrated by the following examples */
/*  with nb = 2: */

/*  m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n): */

/*    (  1   1   u1  u1  u1 )           (  1   u1  u1  u1  u1  u1 ) */
/*    (  v1  1   1   u2  u2 )           (  1   1   u2  u2  u2  u2 ) */
/*    (  v1  v2  a   a   a  )           (  v1  1   a   a   a   a  ) */
/*    (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  ) */
/*    (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  ) */
/*    (  v1  v2  a   a   a  ) */

/*  where a denotes an element of the original matrix which is unchanged, */
/*  vi denotes an element of the vector defining H(i), and ui an element */
/*  of the vector defining G(i). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Quick return if possible */

    /* Parameter adjustments */
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --d__;
    --e;
    --tauq;
    --taup;
    x_dim1 = *ldx;
    x_offset = 1 + x_dim1;
    x -= x_offset;
    y_dim1 = *ldy;
    y_offset = 1 + y_dim1;
    y -= y_offset;

    /* Function Body */
    if (*m <= 0 || *n <= 0) {
	return 0;
    }

    if (*m >= *n) {

/*        Reduce to upper bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i:m,i) */

	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b4, &a[i__ + a_dim1], lda, 
		     &y[i__ + y_dim1], ldy, &c_b5, &a[i__ + i__ * a_dim1], &
		    c__1);
	    i__2 = *m - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b4, &x[i__ + x_dim1], ldx, 
		     &a[i__ * a_dim1 + 1], &c__1, &c_b5, &a[i__ + i__ * 
		    a_dim1], &c__1);

/*           Generate reflection Q(i) to annihilate A(i+1:m,i) */

	    i__2 = *m - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[min(i__3, *m)+ i__ * 
		    a_dim1], &c__1, &tauq[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    if (i__ < *n) {
		a[i__ + i__ * a_dim1] = 1.;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__ + 1;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + (i__ + 1) * 
			a_dim1], lda, &a[i__ + i__ * a_dim1], &c__1, &c_b16, &
			y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + a_dim1], 
			lda, &a[i__ + i__ * a_dim1], &c__1, &c_b16, &y[i__ * 
			y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &y[i__ + 1 + 
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b5, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__ + 1;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &x[i__ + x_dim1], 
			ldx, &a[i__ + i__ * a_dim1], &c__1, &c_b16, &y[i__ * 
			y_dim1 + 1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b4, &a[(i__ + 1) * 
			a_dim1 + 1], lda, &y[i__ * y_dim1 + 1], &c__1, &c_b5, 
			&y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		dscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);

/*              Update A(i,i+1:n) */

		i__2 = *n - i__;
		dgemv_("No transpose", &i__2, &i__, &c_b4, &y[i__ + 1 + 
			y_dim1], ldy, &a[i__ + a_dim1], lda, &c_b5, &a[i__ + (
			i__ + 1) * a_dim1], lda);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b4, &a[(i__ + 1) * 
			a_dim1 + 1], lda, &x[i__ + x_dim1], ldx, &c_b5, &a[
			i__ + (i__ + 1) * a_dim1], lda);

/*              Generate reflection P(i) to annihilate A(i,i+2:n) */

		i__2 = *n - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		dlarfg_(&i__2, &a[i__ + (i__ + 1) * a_dim1], &a[i__ + min(
			i__3, *n)* a_dim1], lda, &taup[i__]);
		e[i__] = a[i__ + (i__ + 1) * a_dim1];
		a[i__ + (i__ + 1) * a_dim1] = 1.;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + (i__ 
			+ 1) * a_dim1], lda, &a[i__ + (i__ + 1) * a_dim1], 
			lda, &c_b16, &x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__;
		dgemv_("Transpose", &i__2, &i__, &c_b5, &y[i__ + 1 + y_dim1], 
			ldy, &a[i__ + (i__ + 1) * a_dim1], lda, &c_b16, &x[
			i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		dgemv_("No transpose", &i__2, &i__, &c_b4, &a[i__ + 1 + 
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[(i__ + 1) * 
			a_dim1 + 1], lda, &a[i__ + (i__ + 1) * a_dim1], lda, &
			c_b16, &x[i__ * x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &x[i__ + 1 + 
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		dscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);
	    }
/* L10: */
	}
    } else {

/*        Reduce to lower bidiagonal form */

	i__1 = *nb;
	for (i__ = 1; i__ <= i__1; ++i__) {

/*           Update A(i,i:n) */

	    i__2 = *n - i__ + 1;
	    i__3 = i__ - 1;
	    dgemv_("No transpose", &i__2, &i__3, &c_b4, &y[i__ + y_dim1], ldy, 
		     &a[i__ + a_dim1], lda, &c_b5, &a[i__ + i__ * a_dim1], 
		    lda);
	    i__2 = i__ - 1;
	    i__3 = *n - i__ + 1;
	    dgemv_("Transpose", &i__2, &i__3, &c_b4, &a[i__ * a_dim1 + 1], 
		    lda, &x[i__ + x_dim1], ldx, &c_b5, &a[i__ + i__ * a_dim1], 
		     lda);

/*           Generate reflection P(i) to annihilate A(i,i+1:n) */

	    i__2 = *n - i__ + 1;
/* Computing MIN */
	    i__3 = i__ + 1;
	    dlarfg_(&i__2, &a[i__ + i__ * a_dim1], &a[i__ + min(i__3, *n)* 
		    a_dim1], lda, &taup[i__]);
	    d__[i__] = a[i__ + i__ * a_dim1];
	    if (i__ < *m) {
		a[i__ + i__ * a_dim1] = 1.;

/*              Compute X(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = *n - i__ + 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + i__ *
			 a_dim1], lda, &a[i__ + i__ * a_dim1], lda, &c_b16, &
			x[i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *n - i__ + 1;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &y[i__ + y_dim1], 
			ldy, &a[i__ + i__ * a_dim1], lda, &c_b16, &x[i__ * 
			x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &a[i__ + 1 + 
			a_dim1], lda, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = i__ - 1;
		i__3 = *n - i__ + 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b5, &a[i__ * a_dim1 + 
			1], lda, &a[i__ + i__ * a_dim1], lda, &c_b16, &x[i__ *
			 x_dim1 + 1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &x[i__ + 1 + 
			x_dim1], ldx, &x[i__ * x_dim1 + 1], &c__1, &c_b5, &x[
			i__ + 1 + i__ * x_dim1], &c__1);
		i__2 = *m - i__;
		dscal_(&i__2, &taup[i__], &x[i__ + 1 + i__ * x_dim1], &c__1);

/*              Update A(i+1:m,i) */

		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &a[i__ + 1 + 
			a_dim1], lda, &y[i__ + y_dim1], ldy, &c_b5, &a[i__ + 
			1 + i__ * a_dim1], &c__1);
		i__2 = *m - i__;
		dgemv_("No transpose", &i__2, &i__, &c_b4, &x[i__ + 1 + 
			x_dim1], ldx, &a[i__ * a_dim1 + 1], &c__1, &c_b5, &a[
			i__ + 1 + i__ * a_dim1], &c__1);

/*              Generate reflection Q(i) to annihilate A(i+2:m,i) */

		i__2 = *m - i__;
/* Computing MIN */
		i__3 = i__ + 2;
		dlarfg_(&i__2, &a[i__ + 1 + i__ * a_dim1], &a[min(i__3, *m)+ 
			i__ * a_dim1], &c__1, &tauq[i__]);
		e[i__] = a[i__ + 1 + i__ * a_dim1];
		a[i__ + 1 + i__ * a_dim1] = 1.;

/*              Compute Y(i+1:n,i) */

		i__2 = *m - i__;
		i__3 = *n - i__;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + (i__ + 
			1) * a_dim1], lda, &a[i__ + 1 + i__ * a_dim1], &c__1, 
			&c_b16, &y[i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		i__3 = i__ - 1;
		dgemv_("Transpose", &i__2, &i__3, &c_b5, &a[i__ + 1 + a_dim1], 
			 lda, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, &y[
			i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		i__3 = i__ - 1;
		dgemv_("No transpose", &i__2, &i__3, &c_b4, &y[i__ + 1 + 
			y_dim1], ldy, &y[i__ * y_dim1 + 1], &c__1, &c_b5, &y[
			i__ + 1 + i__ * y_dim1], &c__1);
		i__2 = *m - i__;
		dgemv_("Transpose", &i__2, &i__, &c_b5, &x[i__ + 1 + x_dim1], 
			ldx, &a[i__ + 1 + i__ * a_dim1], &c__1, &c_b16, &y[
			i__ * y_dim1 + 1], &c__1);
		i__2 = *n - i__;
		dgemv_("Transpose", &i__, &i__2, &c_b4, &a[(i__ + 1) * a_dim1 
			+ 1], lda, &y[i__ * y_dim1 + 1], &c__1, &c_b5, &y[i__ 
			+ 1 + i__ * y_dim1], &c__1);
		i__2 = *n - i__;
		dscal_(&i__2, &tauq[i__], &y[i__ + 1 + i__ * y_dim1], &c__1);
	    }
/* L20: */
	}
    }
    return 0;

/*     End of DLABRD */

} /* dlabrd_ */
