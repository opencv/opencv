#include "clapack.h"

/* Subroutine */ int dsytrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *work, integer *lwork, integer *info)
{
/*  -- LAPACK routine (version 3.0) --   
       Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd.,   
       Courant Institute, Argonne National Lab, and Rice University   
       June 30, 1999   


    Purpose   
    =======   

    DSYTRF computes the factorization of a real symmetric matrix A using   
    the Bunch-Kaufman diagonal pivoting method.  The form of the   
    factorization is   

       A = U*D*U**T  or  A = L*D*L**T   

    where U (or L) is a product of permutation and unit upper (lower)   
    triangular matrices, and D is symmetric and block diagonal with   
    1-by-1 and 2-by-2 diagonal blocks.   

    This is the blocked version of the algorithm, calling Level 3 BLAS.   

    Arguments   
    =========   

    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) DOUBLE PRECISION array, dimension (LDA,N)   
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   

            On exit, the block diagonal matrix D and the multipliers used   
            to obtain the factor U or L (see below for further details).   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    IPIV    (output) INTEGER array, dimension (N)   
            Details of the interchanges and the block structure of D.   
            If IPIV(k) > 0, then rows and columns k and IPIV(k) were   
            interchanged and D(k,k) is a 1-by-1 diagonal block.   
            If UPLO = 'U' and IPIV(k) = IPIV(k-1) < 0, then rows and   
            columns k-1 and -IPIV(k) were interchanged and D(k-1:k,k-1:k)   
            is a 2-by-2 diagonal block.  If UPLO = 'L' and IPIV(k) =   
            IPIV(k+1) < 0, then rows and columns k+1 and -IPIV(k) were   
            interchanged and D(k:k+1,k:k+1) is a 2-by-2 diagonal block.   

    WORK    (workspace/output) DOUBLE PRECISION array, dimension (LWORK)   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The length of WORK.  LWORK >=1.  For best performance   
            LWORK >= N*NB, where NB is the block size returned by ILAENV.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   
            > 0:  if INFO = i, D(i,i) is exactly zero.  The factorization   
                  has been completed, but the block diagonal matrix D is   
                  exactly singular, and division by zero will occur if it   
                  is used to solve a system of equations.   

    Further Details   
    ===============   

    If UPLO = 'U', then A = U*D*U', where   
       U = P(n)*U(n)* ... *P(k)U(k)* ...,   
    i.e., U is a product of terms P(k)*U(k), where k decreases from n to   
    1 in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1   
    and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as   
    defined by IPIV(k), and U(k) is a unit upper triangular matrix, such   
    that if the diagonal block D(k) is of order s (s = 1 or 2), then   

               (   I    v    0   )   k-s   
       U(k) =  (   0    I    0   )   s   
               (   0    0    I   )   n-k   
                  k-s   s   n-k   

    If s = 1, D(k) overwrites A(k,k), and v overwrites A(1:k-1,k).   
    If s = 2, the upper triangle of D(k) overwrites A(k-1,k-1), A(k-1,k),   
    and A(k,k), and v overwrites A(1:k-2,k-1:k).   

    If UPLO = 'L', then A = L*D*L', where   
       L = P(1)*L(1)* ... *P(k)*L(k)* ...,   
    i.e., L is a product of terms P(k)*L(k), where k increases from 1 to   
    n in steps of 1 or 2, and D is a block diagonal matrix with 1-by-1   
    and 2-by-2 diagonal blocks D(k).  P(k) is a permutation matrix as   
    defined by IPIV(k), and L(k) is a unit lower triangular matrix, such   
    that if the diagonal block D(k) is of order s (s = 1 or 2), then   

               (   I    0     0   )  k-1   
       L(k) =  (   0    I     0   )  s   
               (   0    v     I   )  n-k-s+1   
                  k-1   s  n-k-s+1   

    If s = 1, D(k) overwrites A(k,k), and v overwrites A(k+1:n,k).   
    If s = 2, the lower triangle of D(k) overwrites A(k,k), A(k+1,k),   
    and A(k+1,k+1), and v overwrites A(k+2:n,k:k+1).   

    =====================================================================   


       Test the input parameters.   

       Parameter adjustments */
    /* Table of constant values */
    static integer c__1 = 1;
    static integer c_n1 = -1;
    static integer c__2 = 2;
    
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;
    /* Local variables */
    static integer j, k;
    extern logical lsame_(char *, char *);
    static integer nbmin, iinfo;
    static logical upper;
    extern /* Subroutine */ int dsytf2_(char *, integer *, doublereal *, 
	    integer *, integer *, integer *);
    static integer kb, nb;
    extern /* Subroutine */ int xerbla_(char *, integer *);
    extern integer ilaenv_(integer *, char *, char *, integer *, integer *, 
	    integer *, integer *, ftnlen, ftnlen);
    extern /* Subroutine */ int dlasyf_(char *, integer *, integer *, integer 
	    *, doublereal *, integer *, integer *, doublereal *, integer *, 
	    integer *);
    static integer ldwork, lwkopt;
    static logical lquery;
    static integer iws;
#define a_ref(a_1,a_2) a[(a_2)*a_dim1 + a_1]


    a_dim1 = *lda;
    a_offset = 1 + a_dim1 * 1;
    a -= a_offset;
    --ipiv;
    --work;

    /* Function Body */
    *info = 0;
    upper = lsame_(uplo, "U");
    lquery = *lwork == -1;
    if (! upper && ! lsame_(uplo, "L")) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*lda < max(1,*n)) {
	*info = -4;
    } else if (*lwork < 1 && ! lquery) {
	*info = -7;
    }

    if (*info == 0) {

/*        Determine the block size */

	nb = ilaenv_(&c__1, "DSYTRF", uplo, n, &c_n1, &c_n1, &c_n1, (ftnlen)6,
		 (ftnlen)1);
	lwkopt = *n * nb;
	work[1] = (doublereal) lwkopt;
    }

    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DSYTRF", &i__1);
	return 0;
    } else if (lquery) {
	return 0;
    }

    nbmin = 2;
    ldwork = *n;
    if (nb > 1 && nb < *n) {
	iws = ldwork * nb;
	if (*lwork < iws) {
/* Computing MAX */
	    i__1 = *lwork / ldwork;
	    nb = max(i__1,1);
/* Computing MAX */
	    i__1 = 2, i__2 = ilaenv_(&c__2, "DSYTRF", uplo, n, &c_n1, &c_n1, &
		    c_n1, (ftnlen)6, (ftnlen)1);
	    nbmin = max(i__1,i__2);
	}
    } else {
	iws = 1;
    }
    if (nb < nbmin) {
	nb = *n;
    }

    if (upper) {

/*        Factorize A as U*D*U' using the upper triangle of A   

          K is the main loop index, decreasing from N to 1 in steps of   
          KB, where KB is the number of columns factorized by DLASYF;   
          KB is either NB or NB-1, or K for the last block */

	k = *n;
L10:

/*        If K < 1, exit from loop */

	if (k < 1) {
	    goto L40;
	}

	if (k > nb) {

/*           Factorize columns k-kb+1:k of A and use blocked code to   
             update columns 1:k-kb */

	    dlasyf_(uplo, &k, &nb, &kb, &a[a_offset], lda, &ipiv[1], &work[1],
		     &ldwork, &iinfo);
	} else {

/*           Use unblocked code to factorize columns 1:k of A */

	    dsytf2_(uplo, &k, &a[a_offset], lda, &ipiv[1], &iinfo);
	    kb = k;
	}

/*        Set INFO on the first occurrence of a zero pivot */

	if (*info == 0 && iinfo > 0) {
	    *info = iinfo;
	}

/*        Decrease K and return to the start of the main loop */

	k -= kb;
	goto L10;

    } else {

/*        Factorize A as L*D*L' using the lower triangle of A   

          K is the main loop index, increasing from 1 to N in steps of   
          KB, where KB is the number of columns factorized by DLASYF;   
          KB is either NB or NB-1, or N-K+1 for the last block */

	k = 1;
L20:

/*        If K > N, exit from loop */

	if (k > *n) {
	    goto L40;
	}

	if (k <= *n - nb) {

/*           Factorize columns k:k+kb-1 of A and use blocked code to   
             update columns k+kb:n */

	    i__1 = *n - k + 1;
	    dlasyf_(uplo, &i__1, &nb, &kb, &a_ref(k, k), lda, &ipiv[k], &work[
		    1], &ldwork, &iinfo);
	} else {

/*           Use unblocked code to factorize columns k:n of A */

	    i__1 = *n - k + 1;
	    dsytf2_(uplo, &i__1, &a_ref(k, k), lda, &ipiv[k], &iinfo);
	    kb = *n - k + 1;
	}

/*        Set INFO on the first occurrence of a zero pivot */

	if (*info == 0 && iinfo > 0) {
	    *info = iinfo + k - 1;
	}

/*        Adjust IPIV */

	i__1 = k + kb - 1;
	for (j = k; j <= i__1; ++j) {
	    if (ipiv[j] > 0) {
		ipiv[j] = ipiv[j] + k - 1;
	    } else {
		ipiv[j] = ipiv[j] - k + 1;
	    }
/* L30: */
	}

/*        Increase K and return to the start of the main loop */

	k += kb;
	goto L20;

    }

L40:
    work[1] = (doublereal) lwkopt;
    return 0;

/*     End of DSYTRF */

} /* dsytrf_ */

#undef a_ref


