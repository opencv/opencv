
/*  -- translated by f2c (version 19940927).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "clapack.h"

/* Subroutine */ int dsyr_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *a, integer *lda)
{


    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer info;
    static doublereal temp;
    static integer i, j;
    extern logical lsame_(char *, char *);
    static integer ix, jx, kx;
    extern /* Subroutine */ int xerbla_(char *, integer *);


/*  Purpose   
    =======   

    DSYR   performs the symmetric rank 1 operation   

       A := alpha*x*x' + A,   

    where alpha is a real scalar, x is an n element vector and A is an   
    n by n symmetric matrix.   

    Parameters   
    ==========   

    UPLO   - CHARACTER*1.   
             On entry, UPLO specifies whether the upper or lower   
             triangular part of the array A is to be referenced as   
             follows:   

                UPLO = 'U' or 'u'   Only the upper triangular part of A   
                                    is to be referenced.   

                UPLO = 'L' or 'l'   Only the lower triangular part of A   
                                    is to be referenced.   

             Unchanged on exit.   

    N      - INTEGER.   
             On entry, N specifies the order of the matrix A.   
             N must be at least zero.   
             Unchanged on exit.   

    ALPHA  - DOUBLE PRECISION.   
             On entry, ALPHA specifies the scalar alpha.   
             Unchanged on exit.   

    X      - DOUBLE PRECISION array of dimension at least   
             ( 1 + ( n - 1 )*abs( INCX ) ).   
             Before entry, the incremented array X must contain the n   
             element vector x.   
             Unchanged on exit.   

    INCX   - INTEGER.   
             On entry, INCX specifies the increment for the elements of   
             X. INCX must not be zero.   
             Unchanged on exit.   

    A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).   
             Before entry with  UPLO = 'U' or 'u', the leading n by n   
             upper triangular part of the array A must contain the upper 
  
             triangular part of the symmetric matrix and the strictly   
             lower triangular part of A is not referenced. On exit, the   
             upper triangular part of the array A is overwritten by the   
             upper triangular part of the updated matrix.   
             Before entry with UPLO = 'L' or 'l', the leading n by n   
             lower triangular part of the array A must contain the lower 
  
             triangular part of the symmetric matrix and the strictly   
             upper triangular part of A is not referenced. On exit, the   
             lower triangular part of the array A is overwritten by the   
             lower triangular part of the updated matrix.   

    LDA    - INTEGER.   
             On entry, LDA specifies the first dimension of A as declared 
  
             in the calling (sub) program. LDA must be at least   
             max( 1, n ).   
             Unchanged on exit.   


    Level 2 Blas routine.   

    -- Written on 22-October-1986.   
       Jack Dongarra, Argonne National Lab.   
       Jeremy Du Croz, Nag Central Office.   
       Sven Hammarling, Nag Central Office.   
       Richard Hanson, Sandia National Labs.   



       Test the input parameters.   

    
   Parameter adjustments   
       Function Body */
#define X(I) x[(I)-1]

#define A(I,J) a[(I)-1 + ((J)-1)* ( *lda)]

    info = 0;
    if (! lsame_(uplo, "U") && ! lsame_(uplo, "L")) {
	info = 1;
    } else if (*n < 0) {
	info = 2;
    } else if (*incx == 0) {
	info = 5;
    } else if (*lda < max(1,*n)) {
	info = 7;
    }
    if (info != 0) {
	xerbla_("DSYR  ", &info);
	return 0;
    }

/*     Quick return if possible. */

    if (*n == 0 || *alpha == 0.) {
	return 0;
    }

/*     Set the start point in X if the increment is not unity. */

    if (*incx <= 0) {
	kx = 1 - (*n - 1) * *incx;
    } else if (*incx != 1) {
	kx = 1;
    }

/*     Start the operations. In this version the elements of A are   
       accessed sequentially with one pass through the triangular part   
       of A. */

    if (lsame_(uplo, "U")) {

/*        Form  A  when A is stored in upper triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		if (X(j) != 0.) {
		    temp = *alpha * X(j);
		    i__2 = j;
		    for (i = 1; i <= j; ++i) {
			A(i,j) += X(i) * temp;
/* L10: */
		    }
		}
/* L20: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		if (X(jx) != 0.) {
		    temp = *alpha * X(jx);
		    ix = kx;
		    i__2 = j;
		    for (i = 1; i <= j; ++i) {
			A(i,j) += X(ix) * temp;
			ix += *incx;
/* L30: */
		    }
		}
		jx += *incx;
/* L40: */
	    }
	}
    } else {

/*        Form  A  when A is stored in lower triangle. */

	if (*incx == 1) {
	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		if (X(j) != 0.) {
		    temp = *alpha * X(j);
		    i__2 = *n;
		    for (i = j; i <= *n; ++i) {
			A(i,j) += X(i) * temp;
/* L50: */
		    }
		}
/* L60: */
	    }
	} else {
	    jx = kx;
	    i__1 = *n;
	    for (j = 1; j <= *n; ++j) {
		if (X(jx) != 0.) {
		    temp = *alpha * X(jx);
		    ix = jx;
		    i__2 = *n;
		    for (i = j; i <= *n; ++i) {
			A(i,j) += X(ix) * temp;
			ix += *incx;
/* L70: */
		    }
		}
		jx += *incx;
/* L80: */
	    }
	}
    }

    return 0;

/*     End of DSYR  . */

} /* dsyr_ */

