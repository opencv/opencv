#include "clapack.h"

/* Subroutine */ int dsytri_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *work, integer *info)
{
/*  -- LAPACK routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       March 31, 1993   


    Purpose   
    =======   

    DSYTRI computes the inverse of a real symmetric indefinite matrix   
    A using the factorization A = U*D*U**T or A = L*D*L**T computed by   
    DSYTRF.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            Specifies whether the details of the factorization are stored   
            as an upper or lower triangular matrix.   
            = 'U':  Upper triangular, form is A = U*D*U**T;   
            = 'L':  Lower triangular, form is A = L*D*L**T.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)   
            On entry, the block diagonal matrix D and the multipliers   
            used to obtain the factor U or L as computed by DSYTRF.   

            On exit, if INFO = 0, the (symmetric) inverse of the original   
            matrix.  If UPLO = 'U', the upper triangular part of the   
            inverse is formed and the part of A below the diagonal is not   
            referenced; if UPLO = 'L' the lower triangular part of the   
            inverse is formed and the part of A above the diagonal is   
            not referenced.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    IPIV    (input) INTEGER array, dimension (N)   
            Details of the interchanges and the block structure of D   
            as determined by DSYTRF.   

    WORK    (workspace) DOUBLE PRECISION array, dimension (N)   

    INFO    (output) INTEGER   
            = 0: successful exit   
            < 0: if INFO = -i, the i-th argument had an illegal value   
            > 0: if INFO = i, D(i,i) = 0; the matrix is singular and its   
                 inverse could not be computed.   

    =====================================================================   


       Test the input parameters.   

       Parameter adjustments */
    /* Table of constant values */
    static integer c__1 = 1;
    static doublereal c_b11 = -1.;
    static doublereal c_b13 = 0.;
    
    /* System generated locals */
    integer a_dim1, a_offset, i__1;
    doublereal d__1;
    /* Local variables */
    extern doublereal ddot_(integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    static doublereal temp, akkp1, d__;
    static integer k;
    static doublereal t;
    extern logical lsame_(char *, char *);
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *), dswap_(integer *, doublereal *, integer 
	    *, doublereal *, integer *);
    static integer kstep;
    static logical upper;
    extern /* Subroutine */ int dsymv_(char *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    doublereal *, integer *);
    static doublereal ak;
    static integer kp;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    static doublereal akp1;
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --ipiv;
    --work;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYTRI", &i__1);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Check that the diagonal matrix D is nonsingular. */

    if (upper) {

/*        Upper triangular storage: examine D from bottom to top */

	for (*info = *n; *info >= 1; --(*info)) {
	    if (ipiv[*info] > 0 && a_ref(*info, *info) == 0.) {
		return 0;
	    }
/* L10: */
	}
    } else {

/*        Lower triangular storage: examine D from top to bottom. */

	i__1 = *n;
	for (*info = 1; *info <= i__1; ++(*info)) {
	    if (ipiv[*info] > 0 && a_ref(*info, *info) == 0.) {
		return 0;
	    }
/* L20: */
	}
    }
    *info = 0;

    if (upper) {

/*        Compute inv(A) from the factorization A = U*D*U'.   

          K is the main loop index, increasing from 1 to N in steps of   
          1 or 2, depending on the size of the diagonal blocks. */

	k = 1;
L30:

/*        If K > N, exit from loop. */

	if (k > *n) {
	    goto L40;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block   

             Invert the diagonal block. */

	    a_ref(k, k) = 1. / a_ref(k, k);

/*           Compute column K of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		dcopy_(&i__1, &a_ref(1, k), &c__1, &work[1], &c__1);
		i__1 = k - 1;
		dsymv_(uplo, &i__1, &c_b11, &a[a_offset], lda, &work[1], &
			c__1, &c_b13, &a_ref(1, k), &c__1);
		i__1 = k - 1;
		a_ref(k, k) = a_ref(k, k) - ddot_(&i__1, &work[1], &c__1, &
			a_ref(1, k), &c__1);
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block   

             Invert the diagonal block. */

	    t = (d__1 = a_ref(k, k + 1), abs(d__1));
	    ak = a_ref(k, k) / t;
	    akp1 = a_ref(k + 1, k + 1) / t;
	    akkp1 = a_ref(k, k + 1) / t;
	    d__ = t * (ak * akp1 - 1.);
	    a_ref(k, k) = akp1 / d__;
	    a_ref(k + 1, k + 1) = ak / d__;
	    a_ref(k, k + 1) = -akkp1 / d__;

/*           Compute columns K and K+1 of the inverse. */

	    if (k > 1) {
		i__1 = k - 1;
		dcopy_(&i__1, &a_ref(1, k), &c__1, &work[1], &c__1);
		i__1 = k - 1;
		dsymv_(uplo, &i__1, &c_b11, &a[a_offset], lda, &work[1], &
			c__1, &c_b13, &a_ref(1, k), &c__1);
		i__1 = k - 1;
		a_ref(k, k) = a_ref(k, k) - ddot_(&i__1, &work[1], &c__1, &
			a_ref(1, k), &c__1);
		i__1 = k - 1;
		a_ref(k, k + 1) = a_ref(k, k + 1) - ddot_(&i__1, &a_ref(1, k),
			 &c__1, &a_ref(1, k + 1), &c__1);
		i__1 = k - 1;
		dcopy_(&i__1, &a_ref(1, k + 1), &c__1, &work[1], &c__1);
		i__1 = k - 1;
		dsymv_(uplo, &i__1, &c_b11, &a[a_offset], lda, &work[1], &
			c__1, &c_b13, &a_ref(1, k + 1), &c__1);
		i__1 = k - 1;
		a_ref(k + 1, k + 1) = a_ref(k + 1, k + 1) - ddot_(&i__1, &
			work[1], &c__1, &a_ref(1, k + 1), &c__1);
	    }
	    kstep = 2;
	}

	kp = (i__1 = ipiv[k], abs(i__1));
	if (kp != k) {

/*           Interchange rows and columns K and KP in the leading   
             submatrix A(1:k+1,1:k+1) */

	    i__1 = kp - 1;
	    dswap_(&i__1, &a_ref(1, k), &c__1, &a_ref(1, kp), &c__1);
	    i__1 = k - kp - 1;
	    dswap_(&i__1, &a_ref(kp + 1, k), &c__1, &a_ref(kp, kp + 1), lda);
	    temp = a_ref(k, k);
	    a_ref(k, k) = a_ref(kp, kp);
	    a_ref(kp, kp) = temp;
	    if (kstep == 2) {
		temp = a_ref(k, k + 1);
		a_ref(k, k + 1) = a_ref(kp, k + 1);
		a_ref(kp, k + 1) = temp;
	    }
	}

	k += kstep;
	goto L30;
L40:

	;
    } else {

/*        Compute inv(A) from the factorization A = L*D*L'.   

          K is the main loop index, increasing from 1 to N in steps of   
          1 or 2, depending on the size of the diagonal blocks. */

	k = *n;
L50:

/*        If K < 1, exit from loop. */

	if (k < 1) {
	    goto L60;
	}

	if (ipiv[k] > 0) {

/*           1 x 1 diagonal block   

             Invert the diagonal block. */

	    a_ref(k, k) = 1. / a_ref(k, k);

/*           Compute column K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		dcopy_(&i__1, &a_ref(k + 1, k), &c__1, &work[1], &c__1);
		i__1 = *n - k;
		dsymv_(uplo, &i__1, &c_b11, &a_ref(k + 1, k + 1), lda, &work[
			1], &c__1, &c_b13, &a_ref(k + 1, k), &c__1)
			;
		i__1 = *n - k;
		a_ref(k, k) = a_ref(k, k) - ddot_(&i__1, &work[1], &c__1, &
			a_ref(k + 1, k), &c__1);
	    }
	    kstep = 1;
	} else {

/*           2 x 2 diagonal block   

             Invert the diagonal block. */

	    t = (d__1 = a_ref(k, k - 1), abs(d__1));
	    ak = a_ref(k - 1, k - 1) / t;
	    akp1 = a_ref(k, k) / t;
	    akkp1 = a_ref(k, k - 1) / t;
	    d__ = t * (ak * akp1 - 1.);
	    a_ref(k - 1, k - 1) = akp1 / d__;
	    a_ref(k, k) = ak / d__;
	    a_ref(k, k - 1) = -akkp1 / d__;

/*           Compute columns K-1 and K of the inverse. */

	    if (k < *n) {
		i__1 = *n - k;
		dcopy_(&i__1, &a_ref(k + 1, k), &c__1, &work[1], &c__1);
		i__1 = *n - k;
		dsymv_(uplo, &i__1, &c_b11, &a_ref(k + 1, k + 1), lda, &work[
			1], &c__1, &c_b13, &a_ref(k + 1, k), &c__1)
			;
		i__1 = *n - k;
		a_ref(k, k) = a_ref(k, k) - ddot_(&i__1, &work[1], &c__1, &
			a_ref(k + 1, k), &c__1);
		i__1 = *n - k;
		a_ref(k, k - 1) = a_ref(k, k - 1) - ddot_(&i__1, &a_ref(k + 1,
			 k), &c__1, &a_ref(k + 1, k - 1), &c__1);
		i__1 = *n - k;
		dcopy_(&i__1, &a_ref(k + 1, k - 1), &c__1, &work[1], &c__1);
		i__1 = *n - k;
		dsymv_(uplo, &i__1, &c_b11, &a_ref(k + 1, k + 1), lda, &work[
			1], &c__1, &c_b13, &a_ref(k + 1, k - 1), &c__1);
		i__1 = *n - k;
		a_ref(k - 1, k - 1) = a_ref(k - 1, k - 1) - ddot_(&i__1, &
			work[1], &c__1, &a_ref(k + 1, k - 1), &c__1);
	    }
	    kstep = 2;
	}

	kp = (i__1 = ipiv[k], abs(i__1));
	if (kp != k) {

/*           Interchange rows and columns K and KP in the trailing   
             submatrix A(k-1:n,k-1:n) */

	    if (kp < *n) {
		i__1 = *n - kp;
		dswap_(&i__1, &a_ref(kp + 1, k), &c__1, &a_ref(kp + 1, kp), &
			c__1);
	    }
	    i__1 = kp - k - 1;
	    dswap_(&i__1, &a_ref(k + 1, k), &c__1, &a_ref(kp, k + 1), lda);
	    temp = a_ref(k, k);
	    a_ref(k, k) = a_ref(kp, kp);
	    a_ref(kp, kp) = temp;
	    if (kstep == 2) {
		temp = a_ref(k, k - 1);
		a_ref(k, k - 1) = a_ref(kp, k - 1);
		a_ref(kp, k - 1) = temp;
	    }
	}

	k -= kstep;
	goto L50;
L60:
	;
    }

    return 0;

/*     End of DSYTRI */

} /* dsytri_ */

#undef a_ref


