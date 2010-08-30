/* header file for clapack 3.2.1 */

#ifndef __CLAPACK_H
#define __CLAPACK_H

#include "f2c.h"
#include "cblas.h"

#ifdef __cplusplus
extern "C" {
#endif

doublereal dasum_(integer *n, doublereal *dx, integer *incx);

/* Subroutine */ int daxpy_(integer *n, doublereal *da, doublereal *dx, 
	integer *incx, doublereal *dy, integer *incy);

doublereal dcabs1_(doublecomplex *z__);

/* Subroutine */ int dcopy_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy);

doublereal ddot_(integer *n, doublereal *dx, integer *incx, doublereal *dy, 
	integer *incy);

/* Subroutine */ int dgbmv_(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, doublereal *alpha, doublereal *a, integer *lda, 
	doublereal *x, integer *incx, doublereal *beta, doublereal *y, 
	integer *incy);

/* Subroutine */ int dgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, doublereal *alpha, doublereal *a, integer *lda, 
	doublereal *b, integer *ldb, doublereal *beta, doublereal *c__, 
	integer *ldc);

/* Subroutine */ int dgemv_(char *trans, integer *m, integer *n, doublereal *
	alpha, doublereal *a, integer *lda, doublereal *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

/* Subroutine */ int dger_(integer *m, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *y, integer *incy, 
	doublereal *a, integer *lda);

doublereal dnrm2_(integer *n, doublereal *x, integer *incx);

/* Subroutine */ int drot_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy, doublereal *c__, doublereal *s);

/* Subroutine */ int drotg_(doublereal *da, doublereal *db, doublereal *c__, 
	doublereal *s);

/* Subroutine */ int drotm_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy, doublereal *dparam);

/* Subroutine */ int drotmg_(doublereal *dd1, doublereal *dd2, doublereal *
	dx1, doublereal *dy1, doublereal *dparam);

/* Subroutine */ int dsbmv_(char *uplo, integer *n, integer *k, doublereal *
	alpha, doublereal *a, integer *lda, doublereal *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

/* Subroutine */ int dscal_(integer *n, doublereal *da, doublereal *dx, 
	integer *incx);

doublereal dsdot_(integer *n, real *sx, integer *incx, real *sy, integer *
	incy);

/* Subroutine */ int dspmv_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *ap, doublereal *x, integer *incx, doublereal *beta, 
	doublereal *y, integer *incy);

/* Subroutine */ int dspr_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *ap);

/* Subroutine */ int dspr2_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *y, integer *incy, 
	doublereal *ap);

/* Subroutine */ int dswap_(integer *n, doublereal *dx, integer *incx, 
	doublereal *dy, integer *incy);

/* Subroutine */ int dsymm_(char *side, char *uplo, integer *m, integer *n, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *beta, doublereal *c__, integer *ldc);

/* Subroutine */ int dsymv_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *a, integer *lda, doublereal *x, integer *incx, doublereal 
	*beta, doublereal *y, integer *incy);

/* Subroutine */ int dsyr_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *a, integer *lda);

/* Subroutine */ int dsyr2_(char *uplo, integer *n, doublereal *alpha, 
	doublereal *x, integer *incx, doublereal *y, integer *incy, 
	doublereal *a, integer *lda);

/* Subroutine */ int dsyr2k_(char *uplo, char *trans, integer *n, integer *k, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *beta, doublereal *c__, integer *ldc);

/* Subroutine */ int dsyrk_(char *uplo, char *trans, integer *n, integer *k, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *beta, 
	doublereal *c__, integer *ldc);

/* Subroutine */ int dtbmv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *x, integer *incx);

/* Subroutine */ int dtbsv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *x, integer *incx);

/* Subroutine */ int dtpmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *ap, doublereal *x, integer *incx);

/* Subroutine */ int dtpsv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *ap, doublereal *x, integer *incx);

/* Subroutine */ int dtrmm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
	lda, doublereal *b, integer *ldb);

/* Subroutine */ int dtrmv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *a, integer *lda, doublereal *x, integer *incx);

/* Subroutine */ int dtrsm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, doublereal *alpha, doublereal *a, integer *
	lda, doublereal *b, integer *ldb);

/* Subroutine */ int dtrsv_(char *uplo, char *trans, char *diag, integer *n, 
	doublereal *a, integer *lda, doublereal *x, integer *incx);

integer idamax_(integer *n, doublereal *dx, integer *incx);

integer isamax_(integer *n, real *sx, integer *incx);

doublereal sasum_(integer *n, real *sx, integer *incx);

/* Subroutine */ int saxpy_(integer *n, real *sa, real *sx, integer *incx, 
	real *sy, integer *incy);

/* Subroutine */ int scopy_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy);

doublereal sdot_(integer *n, real *sx, integer *incx, real *sy, integer *incy);

doublereal sdsdot_(integer *n, real *sb, real *sx, integer *incx, real *sy, 
	integer *incy);

/* Subroutine */ int sgbmv_(char *trans, integer *m, integer *n, integer *kl, 
	integer *ku, real *alpha, real *a, integer *lda, real *x, integer *
	incx, real *beta, real *y, integer *incy);

/* Subroutine */ int sgemm_(char *transa, char *transb, integer *m, integer *
	n, integer *k, real *alpha, real *a, integer *lda, real *b, integer *
	ldb, real *beta, real *c__, integer *ldc);

/* Subroutine */ int sgemv_(char *trans, integer *m, integer *n, real *alpha, 
	real *a, integer *lda, real *x, integer *incx, real *beta, real *y, 
	integer *incy);

/* Subroutine */ int sger_(integer *m, integer *n, real *alpha, real *x, 
	integer *incx, real *y, integer *incy, real *a, integer *lda);

doublereal snrm2_(integer *n, real *x, integer *incx);

/* Subroutine */ int srot_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy, real *c__, real *s);

/* Subroutine */ int srotg_(real *sa, real *sb, real *c__, real *s);

/* Subroutine */ int srotm_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy, real *sparam);

/* Subroutine */ int srotmg_(real *sd1, real *sd2, real *sx1, real *sy1, real 
	*sparam);

/* Subroutine */ int ssbmv_(char *uplo, integer *n, integer *k, real *alpha, 
	real *a, integer *lda, real *x, integer *incx, real *beta, real *y, 
	integer *incy);

/* Subroutine */ int sscal_(integer *n, real *sa, real *sx, integer *incx);

/* Subroutine */ int sspmv_(char *uplo, integer *n, real *alpha, real *ap, 
	real *x, integer *incx, real *beta, real *y, integer *incy);

/* Subroutine */ int sspr_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *ap);

/* Subroutine */ int sspr2_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *y, integer *incy, real *ap);

/* Subroutine */ int sswap_(integer *n, real *sx, integer *incx, real *sy, 
	integer *incy);

/* Subroutine */ int ssymm_(char *side, char *uplo, integer *m, integer *n, 
	real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta, 
	 real *c__, integer *ldc);

/* Subroutine */ int ssymv_(char *uplo, integer *n, real *alpha, real *a, 
	integer *lda, real *x, integer *incx, real *beta, real *y, integer *
	incy);

/* Subroutine */ int ssyr_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *a, integer *lda);

/* Subroutine */ int ssyr2_(char *uplo, integer *n, real *alpha, real *x, 
	integer *incx, real *y, integer *incy, real *a, integer *lda);

/* Subroutine */ int ssyr2k_(char *uplo, char *trans, integer *n, integer *k, 
	real *alpha, real *a, integer *lda, real *b, integer *ldb, real *beta, 
	 real *c__, integer *ldc);

/* Subroutine */ int ssyrk_(char *uplo, char *trans, integer *n, integer *k, 
	real *alpha, real *a, integer *lda, real *beta, real *c__, integer *
	ldc);

/* Subroutine */ int stbmv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int stbsv_(char *uplo, char *trans, char *diag, integer *n, 
	integer *k, real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int stpmv_(char *uplo, char *trans, char *diag, integer *n, 
	real *ap, real *x, integer *incx);

/* Subroutine */ int stpsv_(char *uplo, char *trans, char *diag, integer *n, 
	real *ap, real *x, integer *incx);

/* Subroutine */ int strmm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, real *alpha, real *a, integer *lda, real *b, 
	integer *ldb);

/* Subroutine */ int strmv_(char *uplo, char *trans, char *diag, integer *n, 
	real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int strsm_(char *side, char *uplo, char *transa, char *diag, 
	integer *m, integer *n, real *alpha, real *a, integer *lda, real *b, 
	integer *ldb);

/* Subroutine */ int strsv_(char *uplo, char *trans, char *diag, integer *n, 
	real *a, integer *lda, real *x, integer *incx);

/* Subroutine */ int xerbla_(char *srname, integer *info);

/* Subroutine */ int xerbla_array__(char *srname_array__, integer *
	srname_len__, integer *info, ftnlen srname_array_len);

/* Subroutine */ int dbdsdc_(char *uplo, char *compq, integer *n, doublereal *
	d__, doublereal *e, doublereal *u, integer *ldu, doublereal *vt, 
	integer *ldvt, doublereal *q, integer *iq, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dbdsqr_(char *uplo, integer *n, integer *ncvt, integer *
	nru, integer *ncc, doublereal *d__, doublereal *e, doublereal *vt, 
	integer *ldvt, doublereal *u, integer *ldu, doublereal *c__, integer *
	ldc, doublereal *work, integer *info);

/* Subroutine */ int ddisna_(char *job, integer *m, integer *n, doublereal *
	d__, doublereal *sep, integer *info);

/* Subroutine */ int dgbbrd_(char *vect, integer *m, integer *n, integer *ncc, 
	 integer *kl, integer *ku, doublereal *ab, integer *ldab, doublereal *
	d__, doublereal *e, doublereal *q, integer *ldq, doublereal *pt, 
	integer *ldpt, doublereal *c__, integer *ldc, doublereal *work, 
	integer *info);

/* Subroutine */ int dgbcon_(char *norm, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, integer *ipiv, doublereal *anorm, 
	doublereal *rcond, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dgbequ_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, integer *
	info);

/* Subroutine */ int dgbequb_(integer *m, integer *n, integer *kl, integer *
	ku, doublereal *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, integer *
	info);

/* Subroutine */ int dgbrfs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, doublereal *ab, integer *ldab, doublereal *afb, 
	integer *ldafb, integer *ipiv, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dgbrfsx_(char *trans, char *equed, integer *n, integer *
	kl, integer *ku, integer *nrhs, doublereal *ab, integer *ldab, 
	doublereal *afb, integer *ldafb, integer *ipiv, doublereal *r__, 
	doublereal *c__, doublereal *b, integer *ldb, doublereal *x, integer *
	ldx, doublereal *rcond, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dgbsv_(integer *n, integer *kl, integer *ku, integer *
	nrhs, doublereal *ab, integer *ldab, integer *ipiv, doublereal *b, 
	integer *ldb, integer *info);

/* Subroutine */ int dgbsvx_(char *fact, char *trans, integer *n, integer *kl, 
	 integer *ku, integer *nrhs, doublereal *ab, integer *ldab, 
	doublereal *afb, integer *ldafb, integer *ipiv, char *equed, 
	doublereal *r__, doublereal *c__, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dgbsvxx_(char *fact, char *trans, integer *n, integer *
	kl, integer *ku, integer *nrhs, doublereal *ab, integer *ldab, 
	doublereal *afb, integer *ldafb, integer *ipiv, char *equed, 
	doublereal *r__, doublereal *c__, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *rcond, doublereal *rpvgrw, 
	doublereal *berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, 
	doublereal *err_bnds_comp__, integer *nparams, doublereal *params, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dgbtf2_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int dgbtrf_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int dgbtrs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, doublereal *ab, integer *ldab, integer *ipiv, 
	doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dgebak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, doublereal *scale, integer *m, doublereal *v, integer *
	ldv, integer *info);

/* Subroutine */ int dgebal_(char *job, integer *n, doublereal *a, integer *
	lda, integer *ilo, integer *ihi, doublereal *scale, integer *info);

/* Subroutine */ int dgebd2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tauq, doublereal *
	taup, doublereal *work, integer *info);

/* Subroutine */ int dgebrd_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tauq, doublereal *
	taup, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dgecon_(char *norm, integer *n, doublereal *a, integer *
	lda, doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dgeequ_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, doublereal 
	*colcnd, doublereal *amax, integer *info);

/* Subroutine */ int dgeequb_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, doublereal 
	*colcnd, doublereal *amax, integer *info);

/* Subroutine */ int dgees_(char *jobvs, char *sort, L_fp select, integer *n, 
	doublereal *a, integer *lda, integer *sdim, doublereal *wr, 
	doublereal *wi, doublereal *vs, integer *ldvs, doublereal *work, 
	integer *lwork, logical *bwork, integer *info);

/* Subroutine */ int dgeesx_(char *jobvs, char *sort, L_fp select, char *
	sense, integer *n, doublereal *a, integer *lda, integer *sdim, 
	doublereal *wr, doublereal *wi, doublereal *vs, integer *ldvs, 
	doublereal *rconde, doublereal *rcondv, doublereal *work, integer *
	lwork, integer *iwork, integer *liwork, logical *bwork, integer *info);

/* Subroutine */ int dgeev_(char *jobvl, char *jobvr, integer *n, doublereal *
	a, integer *lda, doublereal *wr, doublereal *wi, doublereal *vl, 
	integer *ldvl, doublereal *vr, integer *ldvr, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int dgeevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, doublereal *a, integer *lda, doublereal *wr, 
	doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, 
	integer *ldvr, integer *ilo, integer *ihi, doublereal *scale, 
	doublereal *abnrm, doublereal *rconde, doublereal *rcondv, doublereal 
	*work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int dgegs_(char *jobvsl, char *jobvsr, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	alphar, doublereal *alphai, doublereal *beta, doublereal *vsl, 
	integer *ldvsl, doublereal *vsr, integer *ldvsr, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int dgegv_(char *jobvl, char *jobvr, integer *n, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *alphar, 
	doublereal *alphai, doublereal *beta, doublereal *vl, integer *ldvl, 
	doublereal *vr, integer *ldvr, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dgehd2_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *info);

/* Subroutine */ int dgehrd_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int dgejsv_(char *joba, char *jobu, char *jobv, char *jobr, 
	char *jobt, char *jobp, integer *m, integer *n, doublereal *a, 
	integer *lda, doublereal *sva, doublereal *u, integer *ldu, 
	doublereal *v, integer *ldv, doublereal *work, integer *lwork, 
	integer *iwork, integer *info);

/* Subroutine */ int dgelq2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dgelqf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dgels_(char *trans, integer *m, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dgelsd_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	s, doublereal *rcond, integer *rank, doublereal *work, integer *lwork, 
	 integer *iwork, integer *info);

/* Subroutine */ int dgelss_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	s, doublereal *rcond, integer *rank, doublereal *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int dgelsx_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	jpvt, doublereal *rcond, integer *rank, doublereal *work, integer *
	info);

/* Subroutine */ int dgelsy_(integer *m, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	jpvt, doublereal *rcond, integer *rank, doublereal *work, integer *
	lwork, integer *info);

/* Subroutine */ int dgeql2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dgeqlf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dgeqp3_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *jpvt, doublereal *tau, doublereal *work, integer *lwork, 
	 integer *info);

/* Subroutine */ int dgeqpf_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *jpvt, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dgeqr2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dgeqrf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dgerfs_(char *trans, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *af, integer *ldaf, integer *
	ipiv, doublereal *b, integer *ldb, doublereal *x, integer *ldx, 
	doublereal *ferr, doublereal *berr, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dgerfsx_(char *trans, char *equed, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, doublereal *r__, doublereal *c__, doublereal *b, 
	integer *ldb, doublereal *x, integer *ldx, doublereal *rcond, 
	doublereal *berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, 
	doublereal *err_bnds_comp__, integer *nparams, doublereal *params, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dgerq2_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dgerqf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dgesc2_(integer *n, doublereal *a, integer *lda, 
	doublereal *rhs, integer *ipiv, integer *jpiv, doublereal *scale);

/* Subroutine */ int dgesdd_(char *jobz, integer *m, integer *n, doublereal *
	a, integer *lda, doublereal *s, doublereal *u, integer *ldu, 
	doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, 
	integer *iwork, integer *info);

/* Subroutine */ int dgesv_(integer *n, integer *nrhs, doublereal *a, integer 
	*lda, integer *ipiv, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dgesvd_(char *jobu, char *jobvt, integer *m, integer *n, 
	doublereal *a, integer *lda, doublereal *s, doublereal *u, integer *
	ldu, doublereal *vt, integer *ldvt, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dgesvj_(char *joba, char *jobu, char *jobv, integer *m, 
	integer *n, doublereal *a, integer *lda, doublereal *sva, integer *mv, 
	 doublereal *v, integer *ldv, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dgesvx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, char *equed, doublereal *r__, doublereal *c__, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dgesvxx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, char *equed, doublereal *r__, doublereal *c__, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *rpvgrw, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dgetc2_(integer *n, doublereal *a, integer *lda, integer 
	*ipiv, integer *jpiv, integer *info);

/* Subroutine */ int dgetf2_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

/* Subroutine */ int dgetrf_(integer *m, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

/* Subroutine */ int dgetri_(integer *n, doublereal *a, integer *lda, integer 
	*ipiv, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dgetrs_(char *trans, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *
	ldb, integer *info);

/* Subroutine */ int dggbak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, doublereal *lscale, doublereal *rscale, integer *m, 
	doublereal *v, integer *ldv, integer *info);

/* Subroutine */ int dggbal_(char *job, integer *n, doublereal *a, integer *
	lda, doublereal *b, integer *ldb, integer *ilo, integer *ihi, 
	doublereal *lscale, doublereal *rscale, doublereal *work, integer *
	info);

/* Subroutine */ int dgges_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, integer *n, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, integer *sdim, doublereal *alphar, doublereal *alphai, 
	doublereal *beta, doublereal *vsl, integer *ldvsl, doublereal *vsr, 
	integer *ldvsr, doublereal *work, integer *lwork, logical *bwork, 
	integer *info);

/* Subroutine */ int dggesx_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, char *sense, integer *n, doublereal *a, integer *lda, 
	doublereal *b, integer *ldb, integer *sdim, doublereal *alphar, 
	doublereal *alphai, doublereal *beta, doublereal *vsl, integer *ldvsl, 
	 doublereal *vsr, integer *ldvsr, doublereal *rconde, doublereal *
	rcondv, doublereal *work, integer *lwork, integer *iwork, integer *
	liwork, logical *bwork, integer *info);

/* Subroutine */ int dggev_(char *jobvl, char *jobvr, integer *n, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *alphar, 
	doublereal *alphai, doublereal *beta, doublereal *vl, integer *ldvl, 
	doublereal *vr, integer *ldvr, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dggevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, 
	integer *ilo, integer *ihi, doublereal *lscale, doublereal *rscale, 
	doublereal *abnrm, doublereal *bbnrm, doublereal *rconde, doublereal *
	rcondv, doublereal *work, integer *lwork, integer *iwork, logical *
	bwork, integer *info);

/* Subroutine */ int dggglm_(integer *n, integer *m, integer *p, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *d__, 
	doublereal *x, doublereal *y, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dgghrd_(char *compq, char *compz, integer *n, integer *
	ilo, integer *ihi, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *q, integer *ldq, doublereal *z__, integer *
	ldz, integer *info);

/* Subroutine */ int dgglse_(integer *m, integer *n, integer *p, doublereal *
	a, integer *lda, doublereal *b, integer *ldb, doublereal *c__, 
	doublereal *d__, doublereal *x, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dggqrf_(integer *n, integer *m, integer *p, doublereal *
	a, integer *lda, doublereal *taua, doublereal *b, integer *ldb, 
	doublereal *taub, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dggrqf_(integer *m, integer *p, integer *n, doublereal *
	a, integer *lda, doublereal *taua, doublereal *b, integer *ldb, 
	doublereal *taub, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dggsvd_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *n, integer *p, integer *k, integer *l, doublereal *a, 
	integer *lda, doublereal *b, integer *ldb, doublereal *alpha, 
	doublereal *beta, doublereal *u, integer *ldu, doublereal *v, integer 
	*ldv, doublereal *q, integer *ldq, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dggsvp_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *tola, doublereal *tolb, integer *k, integer 
	*l, doublereal *u, integer *ldu, doublereal *v, integer *ldv, 
	doublereal *q, integer *ldq, integer *iwork, doublereal *tau, 
	doublereal *work, integer *info);

/* Subroutine */ int dgsvj0_(char *jobv, integer *m, integer *n, doublereal *
	a, integer *lda, doublereal *d__, doublereal *sva, integer *mv, 
	doublereal *v, integer *ldv, doublereal *eps, doublereal *sfmin, 
	doublereal *tol, integer *nsweep, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dgsvj1_(char *jobv, integer *m, integer *n, integer *n1, 
	doublereal *a, integer *lda, doublereal *d__, doublereal *sva, 
	integer *mv, doublereal *v, integer *ldv, doublereal *eps, doublereal 
	*sfmin, doublereal *tol, integer *nsweep, doublereal *work, integer *
	lwork, integer *info);

/* Subroutine */ int dgtcon_(char *norm, integer *n, doublereal *dl, 
	doublereal *d__, doublereal *du, doublereal *du2, integer *ipiv, 
	doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dgtrfs_(char *trans, integer *n, integer *nrhs, 
	doublereal *dl, doublereal *d__, doublereal *du, doublereal *dlf, 
	doublereal *df, doublereal *duf, doublereal *du2, integer *ipiv, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	ferr, doublereal *berr, doublereal *work, integer *iwork, integer *
	info);

/* Subroutine */ int dgtsv_(integer *n, integer *nrhs, doublereal *dl, 
	doublereal *d__, doublereal *du, doublereal *b, integer *ldb, integer 
	*info);

/* Subroutine */ int dgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, doublereal *dl, doublereal *d__, doublereal *du, doublereal *
	dlf, doublereal *df, doublereal *duf, doublereal *du2, integer *ipiv, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dgttrf_(integer *n, doublereal *dl, doublereal *d__, 
	doublereal *du, doublereal *du2, integer *ipiv, integer *info);

/* Subroutine */ int dgttrs_(char *trans, integer *n, integer *nrhs, 
	doublereal *dl, doublereal *d__, doublereal *du, doublereal *du2, 
	integer *ipiv, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dgtts2_(integer *itrans, integer *n, integer *nrhs, 
	doublereal *dl, doublereal *d__, doublereal *du, doublereal *du2, 
	integer *ipiv, doublereal *b, integer *ldb);

/* Subroutine */ int dhgeqz_(char *job, char *compq, char *compz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*t, integer *ldt, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *q, integer *ldq, doublereal *z__, integer *ldz, 
	doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dhsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, doublereal *h__, integer *ldh, doublereal *wr, 
	doublereal *wi, doublereal *vl, integer *ldvl, doublereal *vr, 
	integer *ldvr, integer *mm, integer *m, doublereal *work, integer *
	ifaill, integer *ifailr, integer *info);

/* Subroutine */ int dhseqr_(char *job, char *compz, integer *n, integer *ilo, 
	 integer *ihi, doublereal *h__, integer *ldh, doublereal *wr, 
	doublereal *wi, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *info);

logical disnan_(doublereal *din);

/* Subroutine */ int dla_gbamv__(integer *trans, integer *m, integer *n, 
	integer *kl, integer *ku, doublereal *alpha, doublereal *ab, integer *
	ldab, doublereal *x, integer *incx, doublereal *beta, doublereal *y, 
	integer *incy);

doublereal dla_gbrcond__(char *trans, integer *n, integer *kl, integer *ku, 
	doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb, 
	integer *ipiv, integer *cmode, doublereal *c__, integer *info, 
	doublereal *work, integer *iwork, ftnlen trans_len);

/* Subroutine */ int dla_gbrfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *kl, integer *ku, integer *nrhs, 
	doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb, 
	integer *ipiv, logical *colequ, doublereal *c__, doublereal *b, 
	integer *ldb, doublereal *y, integer *ldy, doublereal *berr_out__, 
	integer *n_norms__, doublereal *errs_n__, doublereal *errs_c__, 
	doublereal *res, doublereal *ayb, doublereal *dy, doublereal *
	y_tail__, doublereal *rcond, integer *ithresh, doublereal *rthresh, 
	doublereal *dz_ub__, logical *ignore_cwise__, integer *info);

doublereal dla_gbrpvgrw__(integer *n, integer *kl, integer *ku, integer *
	ncols, doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb);

/* Subroutine */ int dla_geamv__(integer *trans, integer *m, integer *n, 
	doublereal *alpha, doublereal *a, integer *lda, doublereal *x, 
	integer *incx, doublereal *beta, doublereal *y, integer *incy);

doublereal dla_gercond__(char *trans, integer *n, doublereal *a, integer *lda,
	 doublereal *af, integer *ldaf, integer *ipiv, integer *cmode, 
	doublereal *c__, integer *info, doublereal *work, integer *iwork, 
	ftnlen trans_len);

/* Subroutine */ int dla_gerfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *nrhs, doublereal *a, integer *lda, 
	doublereal *af, integer *ldaf, integer *ipiv, logical *colequ, 
	doublereal *c__, doublereal *b, integer *ldb, doublereal *y, integer *
	ldy, doublereal *berr_out__, integer *n_norms__, doublereal *errs_n__,
	 doublereal *errs_c__, doublereal *res, doublereal *ayb, doublereal *
	dy, doublereal *y_tail__, doublereal *rcond, integer *ithresh, 
	doublereal *rthresh, doublereal *dz_ub__, logical *ignore_cwise__, 
	integer *info);

/* Subroutine */ int dla_lin_berr__(integer *n, integer *nz, integer *nrhs, 
	doublereal *res, doublereal *ayb, doublereal *berr);

doublereal dla_porcond__(char *uplo, integer *n, doublereal *a, integer *lda, 
	doublereal *af, integer *ldaf, integer *cmode, doublereal *c__, 
	integer *info, doublereal *work, integer *iwork, ftnlen uplo_len);

/* Subroutine */ int dla_porfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *
	af, integer *ldaf, logical *colequ, doublereal *c__, doublereal *b, 
	integer *ldb, doublereal *y, integer *ldy, doublereal *berr_out__, 
	integer *n_norms__, doublereal *errs_n__, doublereal *errs_c__, 
	doublereal *res, doublereal *ayb, doublereal *dy, doublereal *
	y_tail__, doublereal *rcond, integer *ithresh, doublereal *rthresh, 
	doublereal *dz_ub__, logical *ignore_cwise__, integer *info, ftnlen 
	uplo_len);

doublereal dla_porpvgrw__(char *uplo, integer *ncols, doublereal *a, integer *
	lda, doublereal *af, integer *ldaf, doublereal *work, ftnlen uplo_len);

doublereal dla_rpvgrw__(integer *n, integer *ncols, doublereal *a, integer *
	lda, doublereal *af, integer *ldaf);

/* Subroutine */ int dla_syamv__(integer *uplo, integer *n, doublereal *alpha,
	 doublereal *a, integer *lda, doublereal *x, integer *incx, 
	doublereal *beta, doublereal *y, integer *incy);

doublereal dla_syrcond__(char *uplo, integer *n, doublereal *a, integer *lda, 
	doublereal *af, integer *ldaf, integer *ipiv, integer *cmode, 
	doublereal *c__, integer *info, doublereal *work, integer *iwork, 
	ftnlen uplo_len);

/* Subroutine */ int dla_syrfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, doublereal *a, integer *lda, doublereal *
	af, integer *ldaf, integer *ipiv, logical *colequ, doublereal *c__, 
	doublereal *b, integer *ldb, doublereal *y, integer *ldy, doublereal *
	berr_out__, integer *n_norms__, doublereal *errs_n__, doublereal *
	errs_c__, doublereal *res, doublereal *ayb, doublereal *dy, 
	doublereal *y_tail__, doublereal *rcond, integer *ithresh, doublereal 
	*rthresh, doublereal *dz_ub__, logical *ignore_cwise__, integer *info,
	 ftnlen uplo_len);

doublereal dla_syrpvgrw__(char *uplo, integer *n, integer *info, doublereal *
	a, integer *lda, doublereal *af, integer *ldaf, integer *ipiv, 
	doublereal *work, ftnlen uplo_len);

/* Subroutine */ int dla_wwaddw__(integer *n, doublereal *x, doublereal *y, 
	doublereal *w);

/* Subroutine */ int dlabad_(doublereal *small, doublereal *large);

/* Subroutine */ int dlabrd_(integer *m, integer *n, integer *nb, doublereal *
	a, integer *lda, doublereal *d__, doublereal *e, doublereal *tauq, 
	doublereal *taup, doublereal *x, integer *ldx, doublereal *y, integer 
	*ldy);

/* Subroutine */ int dlacn2_(integer *n, doublereal *v, doublereal *x, 
	integer *isgn, doublereal *est, integer *kase, integer *isave);

/* Subroutine */ int dlacon_(integer *n, doublereal *v, doublereal *x, 
	integer *isgn, doublereal *est, integer *kase);

/* Subroutine */ int dlacpy_(char *uplo, integer *m, integer *n, doublereal *
	a, integer *lda, doublereal *b, integer *ldb);

/* Subroutine */ int dladiv_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *d__, doublereal *p, doublereal *q);

/* Subroutine */ int dlae2_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *rt1, doublereal *rt2);

/* Subroutine */ int dlaebz_(integer *ijob, integer *nitmax, integer *n, 
	integer *mmax, integer *minp, integer *nbmin, doublereal *abstol, 
	doublereal *reltol, doublereal *pivmin, doublereal *d__, doublereal *
	e, doublereal *e2, integer *nval, doublereal *ab, doublereal *c__, 
	integer *mout, integer *nab, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dlaed0_(integer *icompq, integer *qsiz, integer *n, 
	doublereal *d__, doublereal *e, doublereal *q, integer *ldq, 
	doublereal *qstore, integer *ldqs, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dlaed1_(integer *n, doublereal *d__, doublereal *q, 
	integer *ldq, integer *indxq, doublereal *rho, integer *cutpnt, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dlaed2_(integer *k, integer *n, integer *n1, doublereal *
	d__, doublereal *q, integer *ldq, integer *indxq, doublereal *rho, 
	doublereal *z__, doublereal *dlamda, doublereal *w, doublereal *q2, 
	integer *indx, integer *indxc, integer *indxp, integer *coltyp, 
	integer *info);

/* Subroutine */ int dlaed3_(integer *k, integer *n, integer *n1, doublereal *
	d__, doublereal *q, integer *ldq, doublereal *rho, doublereal *dlamda, 
	 doublereal *q2, integer *indx, integer *ctot, doublereal *w, 
	doublereal *s, integer *info);

/* Subroutine */ int dlaed4_(integer *n, integer *i__, doublereal *d__, 
	doublereal *z__, doublereal *delta, doublereal *rho, doublereal *dlam, 
	 integer *info);

/* Subroutine */ int dlaed5_(integer *i__, doublereal *d__, doublereal *z__, 
	doublereal *delta, doublereal *rho, doublereal *dlam);

/* Subroutine */ int dlaed6_(integer *kniter, logical *orgati, doublereal *
	rho, doublereal *d__, doublereal *z__, doublereal *finit, doublereal *
	tau, integer *info);

/* Subroutine */ int dlaed7_(integer *icompq, integer *n, integer *qsiz, 
	integer *tlvls, integer *curlvl, integer *curpbm, doublereal *d__, 
	doublereal *q, integer *ldq, integer *indxq, doublereal *rho, integer 
	*cutpnt, doublereal *qstore, integer *qptr, integer *prmptr, integer *
	perm, integer *givptr, integer *givcol, doublereal *givnum, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dlaed8_(integer *icompq, integer *k, integer *n, integer 
	*qsiz, doublereal *d__, doublereal *q, integer *ldq, integer *indxq, 
	doublereal *rho, integer *cutpnt, doublereal *z__, doublereal *dlamda, 
	 doublereal *q2, integer *ldq2, doublereal *w, integer *perm, integer 
	*givptr, integer *givcol, doublereal *givnum, integer *indxp, integer 
	*indx, integer *info);

/* Subroutine */ int dlaed9_(integer *k, integer *kstart, integer *kstop, 
	integer *n, doublereal *d__, doublereal *q, integer *ldq, doublereal *
	rho, doublereal *dlamda, doublereal *w, doublereal *s, integer *lds, 
	integer *info);

/* Subroutine */ int dlaeda_(integer *n, integer *tlvls, integer *curlvl, 
	integer *curpbm, integer *prmptr, integer *perm, integer *givptr, 
	integer *givcol, doublereal *givnum, doublereal *q, integer *qptr, 
	doublereal *z__, doublereal *ztemp, integer *info);

/* Subroutine */ int dlaein_(logical *rightv, logical *noinit, integer *n, 
	doublereal *h__, integer *ldh, doublereal *wr, doublereal *wi, 
	doublereal *vr, doublereal *vi, doublereal *b, integer *ldb, 
	doublereal *work, doublereal *eps3, doublereal *smlnum, doublereal *
	bignum, integer *info);

/* Subroutine */ int dlaev2_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *rt1, doublereal *rt2, doublereal *cs1, doublereal *sn1);

/* Subroutine */ int dlaexc_(logical *wantq, integer *n, doublereal *t, 
	integer *ldt, doublereal *q, integer *ldq, integer *j1, integer *n1, 
	integer *n2, doublereal *work, integer *info);

/* Subroutine */ int dlag2_(doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *safmin, doublereal *scale1, doublereal *
	scale2, doublereal *wr1, doublereal *wr2, doublereal *wi);

/* Subroutine */ int dlag2s_(integer *m, integer *n, doublereal *a, integer *
	lda, real *sa, integer *ldsa, integer *info);

/* Subroutine */ int dlags2_(logical *upper, doublereal *a1, doublereal *a2, 
	doublereal *a3, doublereal *b1, doublereal *b2, doublereal *b3, 
	doublereal *csu, doublereal *snu, doublereal *csv, doublereal *snv, 
	doublereal *csq, doublereal *snq);

/* Subroutine */ int dlagtf_(integer *n, doublereal *a, doublereal *lambda, 
	doublereal *b, doublereal *c__, doublereal *tol, doublereal *d__, 
	integer *in, integer *info);

/* Subroutine */ int dlagtm_(char *trans, integer *n, integer *nrhs, 
	doublereal *alpha, doublereal *dl, doublereal *d__, doublereal *du, 
	doublereal *x, integer *ldx, doublereal *beta, doublereal *b, integer 
	*ldb);

/* Subroutine */ int dlagts_(integer *job, integer *n, doublereal *a, 
	doublereal *b, doublereal *c__, doublereal *d__, integer *in, 
	doublereal *y, doublereal *tol, integer *info);

/* Subroutine */ int dlagv2_(doublereal *a, integer *lda, doublereal *b, 
	integer *ldb, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *csl, doublereal *snl, doublereal *csr, doublereal *
	snr);

/* Subroutine */ int dlahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, integer *info);

/* Subroutine */ int dlahr2_(integer *n, integer *k, integer *nb, doublereal *
	a, integer *lda, doublereal *tau, doublereal *t, integer *ldt, 
	doublereal *y, integer *ldy);

/* Subroutine */ int dlahrd_(integer *n, integer *k, integer *nb, doublereal *
	a, integer *lda, doublereal *tau, doublereal *t, integer *ldt, 
	doublereal *y, integer *ldy);

/* Subroutine */ int dlaic1_(integer *job, integer *j, doublereal *x, 
	doublereal *sest, doublereal *w, doublereal *gamma, doublereal *
	sestpr, doublereal *s, doublereal *c__);

logical dlaisnan_(doublereal *din1, doublereal *din2);

/* Subroutine */ int dlaln2_(logical *ltrans, integer *na, integer *nw, 
	doublereal *smin, doublereal *ca, doublereal *a, integer *lda, 
	doublereal *d1, doublereal *d2, doublereal *b, integer *ldb, 
	doublereal *wr, doublereal *wi, doublereal *x, integer *ldx, 
	doublereal *scale, doublereal *xnorm, integer *info);

/* Subroutine */ int dlals0_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *nrhs, doublereal *b, integer *ldb, doublereal 
	*bx, integer *ldbx, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, doublereal *givnum, integer *ldgnum, doublereal *
	poles, doublereal *difl, doublereal *difr, doublereal *z__, integer *
	k, doublereal *c__, doublereal *s, doublereal *work, integer *info);

/* Subroutine */ int dlalsa_(integer *icompq, integer *smlsiz, integer *n, 
	integer *nrhs, doublereal *b, integer *ldb, doublereal *bx, integer *
	ldbx, doublereal *u, integer *ldu, doublereal *vt, integer *k, 
	doublereal *difl, doublereal *difr, doublereal *z__, doublereal *
	poles, integer *givptr, integer *givcol, integer *ldgcol, integer *
	perm, doublereal *givnum, doublereal *c__, doublereal *s, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int dlalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, doublereal *d__, doublereal *e, doublereal *b, integer *ldb, 
	doublereal *rcond, integer *rank, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dlamrg_(integer *n1, integer *n2, doublereal *a, integer 
	*dtrd1, integer *dtrd2, integer *index);

integer dlaneg_(integer *n, doublereal *d__, doublereal *lld, doublereal *
	sigma, doublereal *pivmin, integer *r__);

doublereal dlangb_(char *norm, integer *n, integer *kl, integer *ku, 
	doublereal *ab, integer *ldab, doublereal *work);

doublereal dlange_(char *norm, integer *m, integer *n, doublereal *a, integer 
	*lda, doublereal *work);

doublereal dlangt_(char *norm, integer *n, doublereal *dl, doublereal *d__, 
	doublereal *du);

doublereal dlanhs_(char *norm, integer *n, doublereal *a, integer *lda, 
	doublereal *work);

doublereal dlansb_(char *norm, char *uplo, integer *n, integer *k, doublereal 
	*ab, integer *ldab, doublereal *work);

doublereal dlansf_(char *norm, char *transr, char *uplo, integer *n, 
	doublereal *a, doublereal *work);

doublereal dlansp_(char *norm, char *uplo, integer *n, doublereal *ap, 
	doublereal *work);

doublereal dlanst_(char *norm, integer *n, doublereal *d__, doublereal *e);

doublereal dlansy_(char *norm, char *uplo, integer *n, doublereal *a, integer 
	*lda, doublereal *work);

doublereal dlantb_(char *norm, char *uplo, char *diag, integer *n, integer *k, 
	 doublereal *ab, integer *ldab, doublereal *work);

doublereal dlantp_(char *norm, char *uplo, char *diag, integer *n, doublereal 
	*ap, doublereal *work);

doublereal dlantr_(char *norm, char *uplo, char *diag, integer *m, integer *n, 
	 doublereal *a, integer *lda, doublereal *work);

/* Subroutine */ int dlanv2_(doublereal *a, doublereal *b, doublereal *c__, 
	doublereal *d__, doublereal *rt1r, doublereal *rt1i, doublereal *rt2r, 
	 doublereal *rt2i, doublereal *cs, doublereal *sn);

/* Subroutine */ int dlapll_(integer *n, doublereal *x, integer *incx, 
	doublereal *y, integer *incy, doublereal *ssmin);

/* Subroutine */ int dlapmt_(logical *forwrd, integer *m, integer *n, 
	doublereal *x, integer *ldx, integer *k);

doublereal dlapy2_(doublereal *x, doublereal *y);

doublereal dlapy3_(doublereal *x, doublereal *y, doublereal *z__);

/* Subroutine */ int dlaqgb_(integer *m, integer *n, integer *kl, integer *ku, 
	 doublereal *ab, integer *ldab, doublereal *r__, doublereal *c__, 
	doublereal *rowcnd, doublereal *colcnd, doublereal *amax, char *equed);

/* Subroutine */ int dlaqge_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *r__, doublereal *c__, doublereal *rowcnd, doublereal 
	*colcnd, doublereal *amax, char *equed);

/* Subroutine */ int dlaqp2_(integer *m, integer *n, integer *offset, 
	doublereal *a, integer *lda, integer *jpvt, doublereal *tau, 
	doublereal *vn1, doublereal *vn2, doublereal *work);

/* Subroutine */ int dlaqps_(integer *m, integer *n, integer *offset, integer 
	*nb, integer *kb, doublereal *a, integer *lda, integer *jpvt, 
	doublereal *tau, doublereal *vn1, doublereal *vn2, doublereal *auxv, 
	doublereal *f, integer *ldf);

/* Subroutine */ int dlaqr0_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dlaqr1_(integer *n, doublereal *h__, integer *ldh, 
	doublereal *sr1, doublereal *si1, doublereal *sr2, doublereal *si2, 
	doublereal *v);

/* Subroutine */ int dlaqr2_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublereal *h__, integer *
	ldh, integer *iloz, integer *ihiz, doublereal *z__, integer *ldz, 
	integer *ns, integer *nd, doublereal *sr, doublereal *si, doublereal *
	v, integer *ldv, integer *nh, doublereal *t, integer *ldt, integer *
	nv, doublereal *wv, integer *ldwv, doublereal *work, integer *lwork);

/* Subroutine */ int dlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, doublereal *h__, integer *
	ldh, integer *iloz, integer *ihiz, doublereal *z__, integer *ldz, 
	integer *ns, integer *nd, doublereal *sr, doublereal *si, doublereal *
	v, integer *ldv, integer *nh, doublereal *t, integer *ldt, integer *
	nv, doublereal *wv, integer *ldwv, doublereal *work, integer *lwork);

/* Subroutine */ int dlaqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dlaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, doublereal 
	*sr, doublereal *si, doublereal *h__, integer *ldh, integer *iloz, 
	integer *ihiz, doublereal *z__, integer *ldz, doublereal *v, integer *
	ldv, doublereal *u, integer *ldu, integer *nv, doublereal *wv, 
	integer *ldwv, integer *nh, doublereal *wh, integer *ldwh);

/* Subroutine */ int dlaqsb_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, doublereal *s, doublereal *scond, doublereal *amax, 
	 char *equed);

/* Subroutine */ int dlaqsp_(char *uplo, integer *n, doublereal *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, char *equed);

/* Subroutine */ int dlaqsy_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *s, doublereal *scond, doublereal *amax, char *equed);

/* Subroutine */ int dlaqtr_(logical *ltran, logical *lreal, integer *n, 
	doublereal *t, integer *ldt, doublereal *b, doublereal *w, doublereal 
	*scale, doublereal *x, doublereal *work, integer *info);

/* Subroutine */ int dlar1v_(integer *n, integer *b1, integer *bn, doublereal 
	*lambda, doublereal *d__, doublereal *l, doublereal *ld, doublereal *
	lld, doublereal *pivmin, doublereal *gaptol, doublereal *z__, logical 
	*wantnc, integer *negcnt, doublereal *ztz, doublereal *mingma, 
	integer *r__, integer *isuppz, doublereal *nrminv, doublereal *resid, 
	doublereal *rqcorr, doublereal *work);

/* Subroutine */ int dlar2v_(integer *n, doublereal *x, doublereal *y, 
	doublereal *z__, integer *incx, doublereal *c__, doublereal *s, 
	integer *incc);

/* Subroutine */ int dlarf_(char *side, integer *m, integer *n, doublereal *v, 
	 integer *incv, doublereal *tau, doublereal *c__, integer *ldc, 
	doublereal *work);

/* Subroutine */ int dlarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, doublereal *v, integer *
	ldv, doublereal *t, integer *ldt, doublereal *c__, integer *ldc, 
	doublereal *work, integer *ldwork);

/* Subroutine */ int dlarfg_(integer *n, doublereal *alpha, doublereal *x, 
	integer *incx, doublereal *tau);

/* Subroutine */ int dlarfp_(integer *n, doublereal *alpha, doublereal *x, 
	integer *incx, doublereal *tau);

/* Subroutine */ int dlarft_(char *direct, char *storev, integer *n, integer *
	k, doublereal *v, integer *ldv, doublereal *tau, doublereal *t, 
	integer *ldt);

/* Subroutine */ int dlarfx_(char *side, integer *m, integer *n, doublereal *
	v, doublereal *tau, doublereal *c__, integer *ldc, doublereal *work);

/* Subroutine */ int dlargv_(integer *n, doublereal *x, integer *incx, 
	doublereal *y, integer *incy, doublereal *c__, integer *incc);

/* Subroutine */ int dlarnv_(integer *idist, integer *iseed, integer *n, 
	doublereal *x);

/* Subroutine */ int dlarra_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *e2, doublereal *spltol, doublereal *tnrm, integer *nsplit, 
	 integer *isplit, integer *info);

/* Subroutine */ int dlarrb_(integer *n, doublereal *d__, doublereal *lld, 
	integer *ifirst, integer *ilast, doublereal *rtol1, doublereal *rtol2, 
	 integer *offset, doublereal *w, doublereal *wgap, doublereal *werr, 
	doublereal *work, integer *iwork, doublereal *pivmin, doublereal *
	spdiam, integer *twist, integer *info);

/* Subroutine */ int dlarrc_(char *jobt, integer *n, doublereal *vl, 
	doublereal *vu, doublereal *d__, doublereal *e, doublereal *pivmin, 
	integer *eigcnt, integer *lcnt, integer *rcnt, integer *info);

/* Subroutine */ int dlarrd_(char *range, char *order, integer *n, doublereal 
	*vl, doublereal *vu, integer *il, integer *iu, doublereal *gers, 
	doublereal *reltol, doublereal *d__, doublereal *e, doublereal *e2, 
	doublereal *pivmin, integer *nsplit, integer *isplit, integer *m, 
	doublereal *w, doublereal *werr, doublereal *wl, doublereal *wu, 
	integer *iblock, integer *indexw, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dlarre_(char *range, integer *n, doublereal *vl, 
	doublereal *vu, integer *il, integer *iu, doublereal *d__, doublereal 
	*e, doublereal *e2, doublereal *rtol1, doublereal *rtol2, doublereal *
	spltol, integer *nsplit, integer *isplit, integer *m, doublereal *w, 
	doublereal *werr, doublereal *wgap, integer *iblock, integer *indexw, 
	doublereal *gers, doublereal *pivmin, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dlarrf_(integer *n, doublereal *d__, doublereal *l, 
	doublereal *ld, integer *clstrt, integer *clend, doublereal *w, 
	doublereal *wgap, doublereal *werr, doublereal *spdiam, doublereal *
	clgapl, doublereal *clgapr, doublereal *pivmin, doublereal *sigma, 
	doublereal *dplus, doublereal *lplus, doublereal *work, integer *info);

/* Subroutine */ int dlarrj_(integer *n, doublereal *d__, doublereal *e2, 
	integer *ifirst, integer *ilast, doublereal *rtol, integer *offset, 
	doublereal *w, doublereal *werr, doublereal *work, integer *iwork, 
	doublereal *pivmin, doublereal *spdiam, integer *info);

/* Subroutine */ int dlarrk_(integer *n, integer *iw, doublereal *gl, 
	doublereal *gu, doublereal *d__, doublereal *e2, doublereal *pivmin, 
	doublereal *reltol, doublereal *w, doublereal *werr, integer *info);

/* Subroutine */ int dlarrr_(integer *n, doublereal *d__, doublereal *e, 
	integer *info);

/* Subroutine */ int dlarrv_(integer *n, doublereal *vl, doublereal *vu, 
	doublereal *d__, doublereal *l, doublereal *pivmin, integer *isplit, 
	integer *m, integer *dol, integer *dou, doublereal *minrgp, 
	doublereal *rtol1, doublereal *rtol2, doublereal *w, doublereal *werr, 
	 doublereal *wgap, integer *iblock, integer *indexw, doublereal *gers, 
	 doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *iwork, integer *info);

/* Subroutine */ int dlarscl2_(integer *m, integer *n, doublereal *d__, 
	doublereal *x, integer *ldx);

/* Subroutine */ int dlartg_(doublereal *f, doublereal *g, doublereal *cs, 
	doublereal *sn, doublereal *r__);

/* Subroutine */ int dlartv_(integer *n, doublereal *x, integer *incx, 
	doublereal *y, integer *incy, doublereal *c__, doublereal *s, integer 
	*incc);

/* Subroutine */ int dlaruv_(integer *iseed, integer *n, doublereal *x);

/* Subroutine */ int dlarz_(char *side, integer *m, integer *n, integer *l, 
	doublereal *v, integer *incv, doublereal *tau, doublereal *c__, 
	integer *ldc, doublereal *work);

/* Subroutine */ int dlarzb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, integer *l, doublereal *v, 
	 integer *ldv, doublereal *t, integer *ldt, doublereal *c__, integer *
	ldc, doublereal *work, integer *ldwork);

/* Subroutine */ int dlarzt_(char *direct, char *storev, integer *n, integer *
	k, doublereal *v, integer *ldv, doublereal *tau, doublereal *t, 
	integer *ldt);

/* Subroutine */ int dlas2_(doublereal *f, doublereal *g, doublereal *h__, 
	doublereal *ssmin, doublereal *ssmax);

/* Subroutine */ int dlascl_(char *type__, integer *kl, integer *ku, 
	doublereal *cfrom, doublereal *cto, integer *m, integer *n, 
	doublereal *a, integer *lda, integer *info);

/* Subroutine */ int dlascl2_(integer *m, integer *n, doublereal *d__, 
	doublereal *x, integer *ldx);

/* Subroutine */ int dlasd0_(integer *n, integer *sqre, doublereal *d__, 
	doublereal *e, doublereal *u, integer *ldu, doublereal *vt, integer *
	ldvt, integer *smlsiz, integer *iwork, doublereal *work, integer *
	info);

/* Subroutine */ int dlasd1_(integer *nl, integer *nr, integer *sqre, 
	doublereal *d__, doublereal *alpha, doublereal *beta, doublereal *u, 
	integer *ldu, doublereal *vt, integer *ldvt, integer *idxq, integer *
	iwork, doublereal *work, integer *info);

/* Subroutine */ int dlasd2_(integer *nl, integer *nr, integer *sqre, integer 
	*k, doublereal *d__, doublereal *z__, doublereal *alpha, doublereal *
	beta, doublereal *u, integer *ldu, doublereal *vt, integer *ldvt, 
	doublereal *dsigma, doublereal *u2, integer *ldu2, doublereal *vt2, 
	integer *ldvt2, integer *idxp, integer *idx, integer *idxc, integer *
	idxq, integer *coltyp, integer *info);

/* Subroutine */ int dlasd3_(integer *nl, integer *nr, integer *sqre, integer 
	*k, doublereal *d__, doublereal *q, integer *ldq, doublereal *dsigma, 
	doublereal *u, integer *ldu, doublereal *u2, integer *ldu2, 
	doublereal *vt, integer *ldvt, doublereal *vt2, integer *ldvt2, 
	integer *idxc, integer *ctot, doublereal *z__, integer *info);

/* Subroutine */ int dlasd4_(integer *n, integer *i__, doublereal *d__, 
	doublereal *z__, doublereal *delta, doublereal *rho, doublereal *
	sigma, doublereal *work, integer *info);

/* Subroutine */ int dlasd5_(integer *i__, doublereal *d__, doublereal *z__, 
	doublereal *delta, doublereal *rho, doublereal *dsigma, doublereal *
	work);

/* Subroutine */ int dlasd6_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, doublereal *d__, doublereal *vf, doublereal *vl, 
	doublereal *alpha, doublereal *beta, integer *idxq, integer *perm, 
	integer *givptr, integer *givcol, integer *ldgcol, doublereal *givnum, 
	 integer *ldgnum, doublereal *poles, doublereal *difl, doublereal *
	difr, doublereal *z__, integer *k, doublereal *c__, doublereal *s, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dlasd7_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *k, doublereal *d__, doublereal *z__, 
	doublereal *zw, doublereal *vf, doublereal *vfw, doublereal *vl, 
	doublereal *vlw, doublereal *alpha, doublereal *beta, doublereal *
	dsigma, integer *idx, integer *idxp, integer *idxq, integer *perm, 
	integer *givptr, integer *givcol, integer *ldgcol, doublereal *givnum, 
	 integer *ldgnum, doublereal *c__, doublereal *s, integer *info);

/* Subroutine */ int dlasd8_(integer *icompq, integer *k, doublereal *d__, 
	doublereal *z__, doublereal *vf, doublereal *vl, doublereal *difl, 
	doublereal *difr, integer *lddifr, doublereal *dsigma, doublereal *
	work, integer *info);

/* Subroutine */ int dlasda_(integer *icompq, integer *smlsiz, integer *n, 
	integer *sqre, doublereal *d__, doublereal *e, doublereal *u, integer 
	*ldu, doublereal *vt, integer *k, doublereal *difl, doublereal *difr, 
	doublereal *z__, doublereal *poles, integer *givptr, integer *givcol, 
	integer *ldgcol, integer *perm, doublereal *givnum, doublereal *c__, 
	doublereal *s, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dlasdq_(char *uplo, integer *sqre, integer *n, integer *
	ncvt, integer *nru, integer *ncc, doublereal *d__, doublereal *e, 
	doublereal *vt, integer *ldvt, doublereal *u, integer *ldu, 
	doublereal *c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int dlasdt_(integer *n, integer *lvl, integer *nd, integer *
	inode, integer *ndiml, integer *ndimr, integer *msub);

/* Subroutine */ int dlaset_(char *uplo, integer *m, integer *n, doublereal *
	alpha, doublereal *beta, doublereal *a, integer *lda);

/* Subroutine */ int dlasq1_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *work, integer *info);

/* Subroutine */ int dlasq2_(integer *n, doublereal *z__, integer *info);

/* Subroutine */ int dlasq3_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, doublereal *dmin__, doublereal *sigma, doublereal *desig, 
	 doublereal *qmax, integer *nfail, integer *iter, integer *ndiv, 
	logical *ieee, integer *ttype, doublereal *dmin1, doublereal *dmin2, 
	doublereal *dn, doublereal *dn1, doublereal *dn2, doublereal *g, 
	doublereal *tau);

/* Subroutine */ int dlasq4_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, integer *n0in, doublereal *dmin__, doublereal *dmin1, 
	doublereal *dmin2, doublereal *dn, doublereal *dn1, doublereal *dn2, 
	doublereal *tau, integer *ttype, doublereal *g);

/* Subroutine */ int dlasq5_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, doublereal *tau, doublereal *dmin__, doublereal *dmin1, 
	doublereal *dmin2, doublereal *dn, doublereal *dnm1, doublereal *dnm2, 
	 logical *ieee);

/* Subroutine */ int dlasq6_(integer *i0, integer *n0, doublereal *z__, 
	integer *pp, doublereal *dmin__, doublereal *dmin1, doublereal *dmin2, 
	 doublereal *dn, doublereal *dnm1, doublereal *dnm2);

/* Subroutine */ int dlasr_(char *side, char *pivot, char *direct, integer *m, 
	 integer *n, doublereal *c__, doublereal *s, doublereal *a, integer *
	lda);

/* Subroutine */ int dlasrt_(char *id, integer *n, doublereal *d__, integer *
	info);

/* Subroutine */ int dlassq_(integer *n, doublereal *x, integer *incx, 
	doublereal *scale, doublereal *sumsq);

/* Subroutine */ int dlasv2_(doublereal *f, doublereal *g, doublereal *h__, 
	doublereal *ssmin, doublereal *ssmax, doublereal *snr, doublereal *
	csr, doublereal *snl, doublereal *csl);

/* Subroutine */ int dlaswp_(integer *n, doublereal *a, integer *lda, integer 
	*k1, integer *k2, integer *ipiv, integer *incx);

/* Subroutine */ int dlasy2_(logical *ltranl, logical *ltranr, integer *isgn, 
	integer *n1, integer *n2, doublereal *tl, integer *ldtl, doublereal *
	tr, integer *ldtr, doublereal *b, integer *ldb, doublereal *scale, 
	doublereal *x, integer *ldx, doublereal *xnorm, integer *info);

/* Subroutine */ int dlasyf_(char *uplo, integer *n, integer *nb, integer *kb, 
	 doublereal *a, integer *lda, integer *ipiv, doublereal *w, integer *
	ldw, integer *info);

/* Subroutine */ int dlat2s_(char *uplo, integer *n, doublereal *a, integer *
	lda, real *sa, integer *ldsa, integer *info);

/* Subroutine */ int dlatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, doublereal *ab, integer *ldab, 
	doublereal *x, doublereal *scale, doublereal *cnorm, integer *info);

/* Subroutine */ int dlatdf_(integer *ijob, integer *n, doublereal *z__, 
	integer *ldz, doublereal *rhs, doublereal *rdsum, doublereal *rdscal, 
	integer *ipiv, integer *jpiv);

/* Subroutine */ int dlatps_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, doublereal *ap, doublereal *x, doublereal *scale, 
	doublereal *cnorm, integer *info);

/* Subroutine */ int dlatrd_(char *uplo, integer *n, integer *nb, doublereal *
	a, integer *lda, doublereal *e, doublereal *tau, doublereal *w, 
	integer *ldw);

/* Subroutine */ int dlatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, doublereal *a, integer *lda, doublereal *x, 
	doublereal *scale, doublereal *cnorm, integer *info);

/* Subroutine */ int dlatrz_(integer *m, integer *n, integer *l, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work);

/* Subroutine */ int dlatzm_(char *side, integer *m, integer *n, doublereal *
	v, integer *incv, doublereal *tau, doublereal *c1, doublereal *c2, 
	integer *ldc, doublereal *work);

/* Subroutine */ int dlauu2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int dlauum_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int dopgtr_(char *uplo, integer *n, doublereal *ap, 
	doublereal *tau, doublereal *q, integer *ldq, doublereal *work, 
	integer *info);

/* Subroutine */ int dopmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, doublereal *ap, doublereal *tau, doublereal *c__, integer 
	*ldc, doublereal *work, integer *info);

/* Subroutine */ int dorg2l_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dorg2r_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dorgbr_(char *vect, integer *m, integer *n, integer *k, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int dorghr_(integer *n, integer *ilo, integer *ihi, 
	doublereal *a, integer *lda, doublereal *tau, doublereal *work, 
	integer *lwork, integer *info);

/* Subroutine */ int dorgl2_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dorglq_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dorgql_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dorgqr_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dorgr2_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *info);

/* Subroutine */ int dorgrq_(integer *m, integer *n, integer *k, doublereal *
	a, integer *lda, doublereal *tau, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dorgtr_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dorm2l_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int dorm2r_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int dormbr_(char *vect, char *side, char *trans, integer *m, 
	integer *n, integer *k, doublereal *a, integer *lda, doublereal *tau, 
	doublereal *c__, integer *ldc, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dormhr_(char *side, char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, doublereal *a, integer *lda, doublereal *
	tau, doublereal *c__, integer *ldc, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dorml2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int dormlq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dormql_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dormqr_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dormr2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int dormr3_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, doublereal *a, integer *lda, doublereal *tau, 
	doublereal *c__, integer *ldc, doublereal *work, integer *info);

/* Subroutine */ int dormrq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dormrz_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, doublereal *a, integer *lda, doublereal *tau, 
	doublereal *c__, integer *ldc, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dormtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, doublereal *a, integer *lda, doublereal *tau, doublereal *
	c__, integer *ldc, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dpbcon_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, doublereal *anorm, doublereal *rcond, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int dpbequ_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, doublereal *s, doublereal *scond, doublereal *amax, 
	 integer *info);

/* Subroutine */ int dpbrfs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublereal *ab, integer *ldab, doublereal *afb, integer *ldafb, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	ferr, doublereal *berr, doublereal *work, integer *iwork, integer *
	info);

/* Subroutine */ int dpbstf_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, integer *info);

/* Subroutine */ int dpbsv_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublereal *ab, integer *ldab, doublereal *b, integer *ldb, 
	integer *info);

/* Subroutine */ int dpbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, doublereal *ab, integer *ldab, doublereal *afb, 
	integer *ldafb, char *equed, doublereal *s, doublereal *b, integer *
	ldb, doublereal *x, integer *ldx, doublereal *rcond, doublereal *ferr, 
	 doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dpbtf2_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, integer *info);

/* Subroutine */ int dpbtrf_(char *uplo, integer *n, integer *kd, doublereal *
	ab, integer *ldab, integer *info);

/* Subroutine */ int dpbtrs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, doublereal *ab, integer *ldab, doublereal *b, integer *ldb, 
	integer *info);

/* Subroutine */ int dpftrf_(char *transr, char *uplo, integer *n, doublereal 
	*a, integer *info);

/* Subroutine */ int dpftri_(char *transr, char *uplo, integer *n, doublereal 
	*a, integer *info);

/* Subroutine */ int dpftrs_(char *transr, char *uplo, integer *n, integer *
	nrhs, doublereal *a, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dpocon_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dpoequ_(integer *n, doublereal *a, integer *lda, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int dpoequb_(integer *n, doublereal *a, integer *lda, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int dporfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	ferr, doublereal *berr, doublereal *work, integer *iwork, integer *
	info);

/* Subroutine */ int dporfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	doublereal *s, doublereal *b, integer *ldb, doublereal *x, integer *
	ldx, doublereal *rcond, doublereal *berr, integer *n_err_bnds__, 
	doublereal *err_bnds_norm__, doublereal *err_bnds_comp__, integer *
	nparams, doublereal *params, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dposv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*a, integer *lda, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dposvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	char *equed, doublereal *s, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *rcond, doublereal *ferr, doublereal *
	berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dposvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	char *equed, doublereal *s, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *rcond, doublereal *rpvgrw, doublereal *
	berr, integer *n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int dpotf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int dpotrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int dpotri_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *info);

/* Subroutine */ int dpotrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int dppcon_(char *uplo, integer *n, doublereal *ap, 
	doublereal *anorm, doublereal *rcond, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dppequ_(char *uplo, integer *n, doublereal *ap, 
	doublereal *s, doublereal *scond, doublereal *amax, integer *info);

/* Subroutine */ int dpprfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, doublereal *afp, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dppsv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*ap, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dppsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *ap, doublereal *afp, char *equed, doublereal *s, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	iwork, integer *info);

/* Subroutine */ int dpptrf_(char *uplo, integer *n, doublereal *ap, integer *
	info);

/* Subroutine */ int dpptri_(char *uplo, integer *n, doublereal *ap, integer *
	info);

/* Subroutine */ int dpptrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dpstf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *piv, integer *rank, doublereal *tol, doublereal *work, 
	integer *info);

/* Subroutine */ int dpstrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *piv, integer *rank, doublereal *tol, doublereal *work, 
	integer *info);

/* Subroutine */ int dptcon_(integer *n, doublereal *d__, doublereal *e, 
	doublereal *anorm, doublereal *rcond, doublereal *work, integer *info);

/* Subroutine */ int dpteqr_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int dptrfs_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *df, doublereal *ef, doublereal *b, integer 
	*ldb, doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	 doublereal *work, integer *info);

/* Subroutine */ int dptsv_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dptsvx_(char *fact, integer *n, integer *nrhs, 
	doublereal *d__, doublereal *e, doublereal *df, doublereal *ef, 
	doublereal *b, integer *ldb, doublereal *x, integer *ldx, doublereal *
	rcond, doublereal *ferr, doublereal *berr, doublereal *work, integer *
	info);

/* Subroutine */ int dpttrf_(integer *n, doublereal *d__, doublereal *e, 
	integer *info);

/* Subroutine */ int dpttrs_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dptts2_(integer *n, integer *nrhs, doublereal *d__, 
	doublereal *e, doublereal *b, integer *ldb);

/* Subroutine */ int drscl_(integer *n, doublereal *sa, doublereal *sx, 
	integer *incx);

/* Subroutine */ int dsbev_(char *jobz, char *uplo, integer *n, integer *kd, 
	doublereal *ab, integer *ldab, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *info);

/* Subroutine */ int dsbevd_(char *jobz, char *uplo, integer *n, integer *kd, 
	doublereal *ab, integer *ldab, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int dsbevx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *kd, doublereal *ab, integer *ldab, doublereal *q, integer *
	ldq, doublereal *vl, doublereal *vu, integer *il, integer *iu, 
	doublereal *abstol, integer *m, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *iwork, integer *ifail, 
	integer *info);

/* Subroutine */ int dsbgst_(char *vect, char *uplo, integer *n, integer *ka, 
	integer *kb, doublereal *ab, integer *ldab, doublereal *bb, integer *
	ldbb, doublereal *x, integer *ldx, doublereal *work, integer *info);

/* Subroutine */ int dsbgv_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, doublereal *ab, integer *ldab, doublereal *bb, integer *
	ldbb, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int dsbgvd_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, doublereal *ab, integer *ldab, doublereal *bb, integer *
	ldbb, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dsbgvx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *ka, integer *kb, doublereal *ab, integer *ldab, doublereal *
	bb, integer *ldbb, doublereal *q, integer *ldq, doublereal *vl, 
	doublereal *vu, integer *il, integer *iu, doublereal *abstol, integer 
	*m, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int dsbtrd_(char *vect, char *uplo, integer *n, integer *kd, 
	doublereal *ab, integer *ldab, doublereal *d__, doublereal *e, 
	doublereal *q, integer *ldq, doublereal *work, integer *info);

/* Subroutine */ int dsfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, doublereal *alpha, doublereal *a, integer *lda, 
	doublereal *beta, doublereal *c__);

/* Subroutine */ int dsgesv_(integer *n, integer *nrhs, doublereal *a, 
	integer *lda, integer *ipiv, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *work, real *swork, integer *iter, 
	integer *info);

/* Subroutine */ int dspcon_(char *uplo, integer *n, doublereal *ap, integer *
	ipiv, doublereal *anorm, doublereal *rcond, doublereal *work, integer 
	*iwork, integer *info);

/* Subroutine */ int dspev_(char *jobz, char *uplo, integer *n, doublereal *
	ap, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int dspevd_(char *jobz, char *uplo, integer *n, doublereal *
	ap, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dspevx_(char *jobz, char *range, char *uplo, integer *n, 
	doublereal *ap, doublereal *vl, doublereal *vu, integer *il, integer *
	iu, doublereal *abstol, integer *m, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *iwork, integer *ifail, 
	integer *info);

/* Subroutine */ int dspgst_(integer *itype, char *uplo, integer *n, 
	doublereal *ap, doublereal *bp, integer *info);

/* Subroutine */ int dspgv_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *ap, doublereal *bp, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *info);

/* Subroutine */ int dspgvd_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *ap, doublereal *bp, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int dspgvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, doublereal *ap, doublereal *bp, doublereal *vl, 
	doublereal *vu, integer *il, integer *iu, doublereal *abstol, integer 
	*m, doublereal *w, doublereal *z__, integer *ldz, doublereal *work, 
	integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int dsposv_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	x, integer *ldx, doublereal *work, real *swork, integer *iter, 
	integer *info);

/* Subroutine */ int dsprfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, doublereal *afp, integer *ipiv, doublereal *b, 
	integer *ldb, doublereal *x, integer *ldx, doublereal *ferr, 
	doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dspsv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*ap, integer *ipiv, doublereal *b, integer *ldb, integer *info);

/* Subroutine */ int dspsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *ap, doublereal *afp, integer *ipiv, doublereal *b, 
	integer *ldb, doublereal *x, integer *ldx, doublereal *rcond, 
	doublereal *ferr, doublereal *berr, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dsptrd_(char *uplo, integer *n, doublereal *ap, 
	doublereal *d__, doublereal *e, doublereal *tau, integer *info);

/* Subroutine */ int dsptrf_(char *uplo, integer *n, doublereal *ap, integer *
	ipiv, integer *info);

/* Subroutine */ int dsptri_(char *uplo, integer *n, doublereal *ap, integer *
	ipiv, doublereal *work, integer *info);

/* Subroutine */ int dsptrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *ap, integer *ipiv, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int dstebz_(char *range, char *order, integer *n, doublereal 
	*vl, doublereal *vu, integer *il, integer *iu, doublereal *abstol, 
	doublereal *d__, doublereal *e, integer *m, integer *nsplit, 
	doublereal *w, integer *iblock, integer *isplit, doublereal *work, 
	integer *iwork, integer *info);

/* Subroutine */ int dstedc_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dstegr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dstein_(integer *n, doublereal *d__, doublereal *e, 
	integer *m, doublereal *w, integer *iblock, integer *isplit, 
	doublereal *z__, integer *ldz, doublereal *work, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int dstemr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, integer *m, doublereal *w, doublereal *z__, integer *ldz, 
	 integer *nzc, integer *isuppz, logical *tryrac, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dsteqr_(char *compz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int dsterf_(integer *n, doublereal *d__, doublereal *e, 
	integer *info);

/* Subroutine */ int dstev_(char *jobz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *info);

/* Subroutine */ int dstevd_(char *jobz, integer *n, doublereal *d__, 
	doublereal *e, doublereal *z__, integer *ldz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dstevr_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dstevx_(char *jobz, char *range, integer *n, doublereal *
	d__, doublereal *e, doublereal *vl, doublereal *vu, integer *il, 
	integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, doublereal *work, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int dsycon_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *anorm, doublereal *rcond, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int dsyequb_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *s, doublereal *scond, doublereal *amax, doublereal *
	work, integer *info);

/* Subroutine */ int dsyev_(char *jobz, char *uplo, integer *n, doublereal *a, 
	 integer *lda, doublereal *w, doublereal *work, integer *lwork, 
	integer *info);

/* Subroutine */ int dsyevd_(char *jobz, char *uplo, integer *n, doublereal *
	a, integer *lda, doublereal *w, doublereal *work, integer *lwork, 
	integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dsyevr_(char *jobz, char *range, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *vl, doublereal *vu, integer *
	il, integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, integer *isuppz, doublereal *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int dsyevx_(char *jobz, char *range, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *vl, doublereal *vu, integer *
	il, integer *iu, doublereal *abstol, integer *m, doublereal *w, 
	doublereal *z__, integer *ldz, doublereal *work, integer *lwork, 
	integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int dsygs2_(integer *itype, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int dsygst_(integer *itype, char *uplo, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int dsygv_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *w, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dsygvd_(integer *itype, char *jobz, char *uplo, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *w, doublereal *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int dsygvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, doublereal *a, integer *lda, doublereal *b, integer 
	*ldb, doublereal *vl, doublereal *vu, integer *il, integer *iu, 
	doublereal *abstol, integer *m, doublereal *w, doublereal *z__, 
	integer *ldz, doublereal *work, integer *lwork, integer *iwork, 
	integer *ifail, integer *info);

/* Subroutine */ int dsyrfs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, doublereal *af, integer *ldaf, integer *
	ipiv, doublereal *b, integer *ldb, doublereal *x, integer *ldx, 
	doublereal *ferr, doublereal *berr, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dsyrfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, doublereal *s, doublereal *b, integer *ldb, doublereal 
	*x, integer *ldx, doublereal *rcond, doublereal *berr, integer *
	n_err_bnds__, doublereal *err_bnds_norm__, doublereal *
	err_bnds_comp__, integer *nparams, doublereal *params, doublereal *
	work, integer *iwork, integer *info);

/* Subroutine */ int dsysv_(char *uplo, integer *n, integer *nrhs, doublereal 
	*a, integer *lda, integer *ipiv, doublereal *b, integer *ldb, 
	doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dsysvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, doublereal *b, integer *ldb, doublereal *x, integer *
	ldx, doublereal *rcond, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int dsysvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, doublereal *a, integer *lda, doublereal *af, integer *ldaf, 
	integer *ipiv, char *equed, doublereal *s, doublereal *b, integer *
	ldb, doublereal *x, integer *ldx, doublereal *rcond, doublereal *
	rpvgrw, doublereal *berr, integer *n_err_bnds__, doublereal *
	err_bnds_norm__, doublereal *err_bnds_comp__, integer *nparams, 
	doublereal *params, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dsytd2_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tau, integer *info);

/* Subroutine */ int dsytf2_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, integer *info);

/* Subroutine */ int dsytrd_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *d__, doublereal *e, doublereal *tau, doublereal *
	work, integer *lwork, integer *info);

/* Subroutine */ int dsytrf_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dsytri_(char *uplo, integer *n, doublereal *a, integer *
	lda, integer *ipiv, doublereal *work, integer *info);

/* Subroutine */ int dsytrs_(char *uplo, integer *n, integer *nrhs, 
	doublereal *a, integer *lda, integer *ipiv, doublereal *b, integer *
	ldb, integer *info);

/* Subroutine */ int dtbcon_(char *norm, char *uplo, char *diag, integer *n, 
	integer *kd, doublereal *ab, integer *ldab, doublereal *rcond, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dtbrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, doublereal *ab, integer *ldab, doublereal 
	*b, integer *ldb, doublereal *x, integer *ldx, doublereal *ferr, 
	doublereal *berr, doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dtbtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, doublereal *ab, integer *ldab, doublereal 
	*b, integer *ldb, integer *info);

/* Subroutine */ int dtfsm_(char *transr, char *side, char *uplo, char *trans, 
	 char *diag, integer *m, integer *n, doublereal *alpha, doublereal *a, 
	 doublereal *b, integer *ldb);

/* Subroutine */ int dtftri_(char *transr, char *uplo, char *diag, integer *n, 
	 doublereal *a, integer *info);

/* Subroutine */ int dtfttp_(char *transr, char *uplo, integer *n, doublereal 
	*arf, doublereal *ap, integer *info);

/* Subroutine */ int dtfttr_(char *transr, char *uplo, integer *n, doublereal 
	*arf, doublereal *a, integer *lda, integer *info);

/* Subroutine */ int dtgevc_(char *side, char *howmny, logical *select, 
	integer *n, doublereal *s, integer *lds, doublereal *p, integer *ldp, 
	doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, integer 
	*mm, integer *m, doublereal *work, integer *info);

/* Subroutine */ int dtgex2_(logical *wantq, logical *wantz, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	q, integer *ldq, doublereal *z__, integer *ldz, integer *j1, integer *
	n1, integer *n2, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dtgexc_(logical *wantq, logical *wantz, integer *n, 
	doublereal *a, integer *lda, doublereal *b, integer *ldb, doublereal *
	q, integer *ldq, doublereal *z__, integer *ldz, integer *ifst, 
	integer *ilst, doublereal *work, integer *lwork, integer *info);

/* Subroutine */ int dtgsen_(integer *ijob, logical *wantq, logical *wantz, 
	logical *select, integer *n, doublereal *a, integer *lda, doublereal *
	b, integer *ldb, doublereal *alphar, doublereal *alphai, doublereal *
	beta, doublereal *q, integer *ldq, doublereal *z__, integer *ldz, 
	integer *m, doublereal *pl, doublereal *pr, doublereal *dif, 
	doublereal *work, integer *lwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int dtgsja_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, integer *k, integer *l, doublereal *a, 
	integer *lda, doublereal *b, integer *ldb, doublereal *tola, 
	doublereal *tolb, doublereal *alpha, doublereal *beta, doublereal *u, 
	integer *ldu, doublereal *v, integer *ldv, doublereal *q, integer *
	ldq, doublereal *work, integer *ncycle, integer *info);

/* Subroutine */ int dtgsna_(char *job, char *howmny, logical *select, 
	integer *n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *vl, integer *ldvl, doublereal *vr, integer *ldvr, 
	doublereal *s, doublereal *dif, integer *mm, integer *m, doublereal *
	work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int dtgsy2_(char *trans, integer *ijob, integer *m, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *c__, integer *ldc, doublereal *d__, integer *ldd, 
	doublereal *e, integer *lde, doublereal *f, integer *ldf, doublereal *
	scale, doublereal *rdsum, doublereal *rdscal, integer *iwork, integer 
	*pq, integer *info);

/* Subroutine */ int dtgsyl_(char *trans, integer *ijob, integer *m, integer *
	n, doublereal *a, integer *lda, doublereal *b, integer *ldb, 
	doublereal *c__, integer *ldc, doublereal *d__, integer *ldd, 
	doublereal *e, integer *lde, doublereal *f, integer *ldf, doublereal *
	scale, doublereal *dif, doublereal *work, integer *lwork, integer *
	iwork, integer *info);

/* Subroutine */ int dtpcon_(char *norm, char *uplo, char *diag, integer *n, 
	doublereal *ap, doublereal *rcond, doublereal *work, integer *iwork, 
	integer *info);

/* Subroutine */ int dtprfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *ap, doublereal *b, integer *ldb, 
	doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dtptri_(char *uplo, char *diag, integer *n, doublereal *
	ap, integer *info);

/* Subroutine */ int dtptrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *ap, doublereal *b, integer *ldb, integer *
	info);

/* Subroutine */ int dtpttf_(char *transr, char *uplo, integer *n, doublereal 
	*ap, doublereal *arf, integer *info);

/* Subroutine */ int dtpttr_(char *uplo, integer *n, doublereal *ap, 
	doublereal *a, integer *lda, integer *info);

/* Subroutine */ int dtrcon_(char *norm, char *uplo, char *diag, integer *n, 
	doublereal *a, integer *lda, doublereal *rcond, doublereal *work, 
	integer *iwork, integer *info);

/* Subroutine */ int dtrevc_(char *side, char *howmny, logical *select, 
	integer *n, doublereal *t, integer *ldt, doublereal *vl, integer *
	ldvl, doublereal *vr, integer *ldvr, integer *mm, integer *m, 
	doublereal *work, integer *info);

/* Subroutine */ int dtrexc_(char *compq, integer *n, doublereal *t, integer *
	ldt, doublereal *q, integer *ldq, integer *ifst, integer *ilst, 
	doublereal *work, integer *info);

/* Subroutine */ int dtrrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *
	ldb, doublereal *x, integer *ldx, doublereal *ferr, doublereal *berr, 
	doublereal *work, integer *iwork, integer *info);

/* Subroutine */ int dtrsen_(char *job, char *compq, logical *select, integer 
	*n, doublereal *t, integer *ldt, doublereal *q, integer *ldq, 
	doublereal *wr, doublereal *wi, integer *m, doublereal *s, doublereal 
	*sep, doublereal *work, integer *lwork, integer *iwork, integer *
	liwork, integer *info);

/* Subroutine */ int dtrsna_(char *job, char *howmny, logical *select, 
	integer *n, doublereal *t, integer *ldt, doublereal *vl, integer *
	ldvl, doublereal *vr, integer *ldvr, doublereal *s, doublereal *sep, 
	integer *mm, integer *m, doublereal *work, integer *ldwork, integer *
	iwork, integer *info);

/* Subroutine */ int dtrsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, doublereal *a, integer *lda, doublereal *b, integer *
	ldb, doublereal *c__, integer *ldc, doublereal *scale, integer *info);

/* Subroutine */ int dtrti2_(char *uplo, char *diag, integer *n, doublereal *
	a, integer *lda, integer *info);

/* Subroutine */ int dtrtri_(char *uplo, char *diag, integer *n, doublereal *
	a, integer *lda, integer *info);

/* Subroutine */ int dtrtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, doublereal *a, integer *lda, doublereal *b, integer *
	ldb, integer *info);

/* Subroutine */ int dtrttf_(char *transr, char *uplo, integer *n, doublereal 
	*a, integer *lda, doublereal *arf, integer *info);

/* Subroutine */ int dtrttp_(char *uplo, integer *n, doublereal *a, integer *
	lda, doublereal *ap, integer *info);

/* Subroutine */ int dtzrqf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, integer *info);

/* Subroutine */ int dtzrzf_(integer *m, integer *n, doublereal *a, integer *
	lda, doublereal *tau, doublereal *work, integer *lwork, integer *info);

doublereal dzsum1_(integer *n, doublecomplex *cx, integer *incx);

integer icmax1_(integer *n, complex *cx, integer *incx);

integer ieeeck_(integer *ispec, real *zero, real *one);

integer ilaclc_(integer *m, integer *n, complex *a, integer *lda);

integer ilaclr_(integer *m, integer *n, complex *a, integer *lda);

integer iladiag_(char *diag);

integer iladlc_(integer *m, integer *n, doublereal *a, integer *lda);

integer iladlr_(integer *m, integer *n, doublereal *a, integer *lda);

integer ilaenv_(integer *ispec, char *name__, char *opts, integer *n1, 
	integer *n2, integer *n3, integer *n4);

integer ilaprec_(char *prec);

integer ilaslc_(integer *m, integer *n, real *a, integer *lda);

integer ilaslr_(integer *m, integer *n, real *a, integer *lda);

integer ilatrans_(char *trans);

integer ilauplo_(char *uplo);

/* Subroutine */ int ilaver_(integer *vers_major__, integer *vers_minor__, 
	integer *vers_patch__);

integer ilazlc_(integer *m, integer *n, doublecomplex *a, integer *lda);

integer ilazlr_(integer *m, integer *n, doublecomplex *a, integer *lda);

integer iparmq_(integer *ispec, char *name__, char *opts, integer *n, integer 
	*ilo, integer *ihi, integer *lwork);

integer izmax1_(integer *n, doublecomplex *cx, integer *incx);

logical lsamen_(integer *n, char *ca, char *cb);

integer smaxloc_(real *a, integer *dimm);

/* Subroutine */ int sbdsdc_(char *uplo, char *compq, integer *n, real *d__, 
	real *e, real *u, integer *ldu, real *vt, integer *ldvt, real *q, 
	integer *iq, real *work, integer *iwork, integer *info);

/* Subroutine */ int sbdsqr_(char *uplo, integer *n, integer *ncvt, integer *
	nru, integer *ncc, real *d__, real *e, real *vt, integer *ldvt, real *
	u, integer *ldu, real *c__, integer *ldc, real *work, integer *info);

doublereal scsum1_(integer *n, complex *cx, integer *incx);

/* Subroutine */ int sdisna_(char *job, integer *m, integer *n, real *d__, 
	real *sep, integer *info);

/* Subroutine */ int sgbbrd_(char *vect, integer *m, integer *n, integer *ncc, 
	 integer *kl, integer *ku, real *ab, integer *ldab, real *d__, real *
	e, real *q, integer *ldq, real *pt, integer *ldpt, real *c__, integer 
	*ldc, real *work, integer *info);

/* Subroutine */ int sgbcon_(char *norm, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, integer *ipiv, real *anorm, real *rcond, 
	real *work, integer *iwork, integer *info);

/* Subroutine */ int sgbequ_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real *
	colcnd, real *amax, integer *info);

/* Subroutine */ int sgbequb_(integer *m, integer *n, integer *kl, integer *
	ku, real *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real 
	*colcnd, real *amax, integer *info);

/* Subroutine */ int sgbrfs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, real *ab, integer *ldab, real *afb, integer *ldafb, 
	 integer *ipiv, real *b, integer *ldb, real *x, integer *ldx, real *
	ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int sgbrfsx_(char *trans, char *equed, integer *n, integer *
	kl, integer *ku, integer *nrhs, real *ab, integer *ldab, real *afb, 
	integer *ldafb, integer *ipiv, real *r__, real *c__, real *b, integer 
	*ldb, real *x, integer *ldx, real *rcond, real *berr, integer *
	n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, integer *
	nparams, real *params, real *work, integer *iwork, integer *info);

/* Subroutine */ int sgbsv_(integer *n, integer *kl, integer *ku, integer *
	nrhs, real *ab, integer *ldab, integer *ipiv, real *b, integer *ldb, 
	integer *info);

/* Subroutine */ int sgbsvx_(char *fact, char *trans, integer *n, integer *kl, 
	 integer *ku, integer *nrhs, real *ab, integer *ldab, real *afb, 
	integer *ldafb, integer *ipiv, char *equed, real *r__, real *c__, 
	real *b, integer *ldb, real *x, integer *ldx, real *rcond, real *ferr, 
	 real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int sgbsvxx_(char *fact, char *trans, integer *n, integer *
	kl, integer *ku, integer *nrhs, real *ab, integer *ldab, real *afb, 
	integer *ldafb, integer *ipiv, char *equed, real *r__, real *c__, 
	real *b, integer *ldb, real *x, integer *ldx, real *rcond, real *
	rpvgrw, real *berr, integer *n_err_bnds__, real *err_bnds_norm__, 
	real *err_bnds_comp__, integer *nparams, real *params, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int sgbtf2_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int sgbtrf_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, integer *ipiv, integer *info);

/* Subroutine */ int sgbtrs_(char *trans, integer *n, integer *kl, integer *
	ku, integer *nrhs, real *ab, integer *ldab, integer *ipiv, real *b, 
	integer *ldb, integer *info);

/* Subroutine */ int sgebak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, real *scale, integer *m, real *v, integer *ldv, integer 
	*info);

/* Subroutine */ int sgebal_(char *job, integer *n, real *a, integer *lda, 
	integer *ilo, integer *ihi, real *scale, integer *info);

/* Subroutine */ int sgebd2_(integer *m, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tauq, real *taup, real *work, integer *info);

/* Subroutine */ int sgebrd_(integer *m, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tauq, real *taup, real *work, integer *
	lwork, integer *info);

/* Subroutine */ int sgecon_(char *norm, integer *n, real *a, integer *lda, 
	real *anorm, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int sgeequ_(integer *m, integer *n, real *a, integer *lda, 
	real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, integer 
	*info);

/* Subroutine */ int sgeequb_(integer *m, integer *n, real *a, integer *lda, 
	real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, integer 
	*info);

/* Subroutine */ int sgees_(char *jobvs, char *sort, L_fp select, integer *n, 
	real *a, integer *lda, integer *sdim, real *wr, real *wi, real *vs, 
	integer *ldvs, real *work, integer *lwork, logical *bwork, integer *
	info);

/* Subroutine */ int sgeesx_(char *jobvs, char *sort, L_fp select, char *
	sense, integer *n, real *a, integer *lda, integer *sdim, real *wr, 
	real *wi, real *vs, integer *ldvs, real *rconde, real *rcondv, real *
	work, integer *lwork, integer *iwork, integer *liwork, logical *bwork, 
	 integer *info);

/* Subroutine */ int sgeev_(char *jobvl, char *jobvr, integer *n, real *a, 
	integer *lda, real *wr, real *wi, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgeevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, real *a, integer *lda, real *wr, real *wi, real *
	vl, integer *ldvl, real *vr, integer *ldvr, integer *ilo, integer *
	ihi, real *scale, real *abnrm, real *rconde, real *rcondv, real *work, 
	 integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int sgegs_(char *jobvsl, char *jobvsr, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, real *alphar, real *alphai, real 
	*beta, real *vsl, integer *ldvsl, real *vsr, integer *ldvsr, real *
	work, integer *lwork, integer *info);

/* Subroutine */ int sgegv_(char *jobvl, char *jobvr, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, real *alphar, real *alphai, real 
	*beta, real *vl, integer *ldvl, real *vr, integer *ldvr, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int sgehd2_(integer *n, integer *ilo, integer *ihi, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int sgehrd_(integer *n, integer *ilo, integer *ihi, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgejsv_(char *joba, char *jobu, char *jobv, char *jobr, 
	char *jobt, char *jobp, integer *m, integer *n, real *a, integer *lda, 
	 real *sva, real *u, integer *ldu, real *v, integer *ldv, real *work, 
	integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int sgelq2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int sgelqf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgels_(char *trans, integer *m, integer *n, integer *
	nrhs, real *a, integer *lda, real *b, integer *ldb, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int sgelsd_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, real *s, real *rcond, integer *
	rank, real *work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int sgelss_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, real *s, real *rcond, integer *
	rank, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgelsx_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *jpvt, real *rcond, 
	integer *rank, real *work, integer *info);

/* Subroutine */ int sgelsy_(integer *m, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *jpvt, real *rcond, 
	integer *rank, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgeql2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int sgeqlf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgeqp3_(integer *m, integer *n, real *a, integer *lda, 
	integer *jpvt, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgeqpf_(integer *m, integer *n, real *a, integer *lda, 
	integer *jpvt, real *tau, real *work, integer *info);

/* Subroutine */ int sgeqr2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int sgeqrf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgerfs_(char *trans, integer *n, integer *nrhs, real *a, 
	integer *lda, real *af, integer *ldaf, integer *ipiv, real *b, 
	integer *ldb, real *x, integer *ldx, real *ferr, real *berr, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int sgerfsx_(char *trans, char *equed, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	real *r__, real *c__, real *b, integer *ldb, real *x, integer *ldx, 
	real *rcond, real *berr, integer *n_err_bnds__, real *err_bnds_norm__, 
	 real *err_bnds_comp__, integer *nparams, real *params, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int sgerq2_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *info);

/* Subroutine */ int sgerqf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgesc2_(integer *n, real *a, integer *lda, real *rhs, 
	integer *ipiv, integer *jpiv, real *scale);

/* Subroutine */ int sgesdd_(char *jobz, integer *m, integer *n, real *a, 
	integer *lda, real *s, real *u, integer *ldu, real *vt, integer *ldvt, 
	 real *work, integer *lwork, integer *iwork, integer *info);

/* Subroutine */ int sgesv_(integer *n, integer *nrhs, real *a, integer *lda, 
	integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int sgesvd_(char *jobu, char *jobvt, integer *m, integer *n, 
	real *a, integer *lda, real *s, real *u, integer *ldu, real *vt, 
	integer *ldvt, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgesvj_(char *joba, char *jobu, char *jobv, integer *m, 
	integer *n, real *a, integer *lda, real *sva, integer *mv, real *v, 
	integer *ldv, real *work, integer *lwork, integer *info);

/* Subroutine */ int sgesvx_(char *fact, char *trans, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	char *equed, real *r__, real *c__, real *b, integer *ldb, real *x, 
	integer *ldx, real *rcond, real *ferr, real *berr, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int sgesvxx_(char *fact, char *trans, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	char *equed, real *r__, real *c__, real *b, integer *ldb, real *x, 
	integer *ldx, real *rcond, real *rpvgrw, real *berr, integer *
	n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, integer *
	nparams, real *params, real *work, integer *iwork, integer *info);

/* Subroutine */ int sgetc2_(integer *n, real *a, integer *lda, integer *ipiv, 
	 integer *jpiv, integer *info);

/* Subroutine */ int sgetf2_(integer *m, integer *n, real *a, integer *lda, 
	integer *ipiv, integer *info);

/* Subroutine */ int sgetrf_(integer *m, integer *n, real *a, integer *lda, 
	integer *ipiv, integer *info);

/* Subroutine */ int sgetri_(integer *n, real *a, integer *lda, integer *ipiv, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int sgetrs_(char *trans, integer *n, integer *nrhs, real *a, 
	integer *lda, integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int sggbak_(char *job, char *side, integer *n, integer *ilo, 
	integer *ihi, real *lscale, real *rscale, integer *m, real *v, 
	integer *ldv, integer *info);

/* Subroutine */ int sggbal_(char *job, integer *n, real *a, integer *lda, 
	real *b, integer *ldb, integer *ilo, integer *ihi, real *lscale, real 
	*rscale, real *work, integer *info);

/* Subroutine */ int sgges_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, integer *n, real *a, integer *lda, real *b, integer *ldb, 
	integer *sdim, real *alphar, real *alphai, real *beta, real *vsl, 
	integer *ldvsl, real *vsr, integer *ldvsr, real *work, integer *lwork, 
	 logical *bwork, integer *info);

/* Subroutine */ int sggesx_(char *jobvsl, char *jobvsr, char *sort, L_fp 
	selctg, char *sense, integer *n, real *a, integer *lda, real *b, 
	integer *ldb, integer *sdim, real *alphar, real *alphai, real *beta, 
	real *vsl, integer *ldvsl, real *vsr, integer *ldvsr, real *rconde, 
	real *rcondv, real *work, integer *lwork, integer *iwork, integer *
	liwork, logical *bwork, integer *info);

/* Subroutine */ int sggev_(char *jobvl, char *jobvr, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, real *alphar, real *alphai, real 
	*beta, real *vl, integer *ldvl, real *vr, integer *ldvr, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int sggevx_(char *balanc, char *jobvl, char *jobvr, char *
	sense, integer *n, real *a, integer *lda, real *b, integer *ldb, real 
	*alphar, real *alphai, real *beta, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, integer *ilo, integer *ihi, real *lscale, real *rscale, 
	 real *abnrm, real *bbnrm, real *rconde, real *rcondv, real *work, 
	integer *lwork, integer *iwork, logical *bwork, integer *info);

/* Subroutine */ int sggglm_(integer *n, integer *m, integer *p, real *a, 
	integer *lda, real *b, integer *ldb, real *d__, real *x, real *y, 
	real *work, integer *lwork, integer *info);

/* Subroutine */ int sgghrd_(char *compq, char *compz, integer *n, integer *
	ilo, integer *ihi, real *a, integer *lda, real *b, integer *ldb, real 
	*q, integer *ldq, real *z__, integer *ldz, integer *info);

/* Subroutine */ int sgglse_(integer *m, integer *n, integer *p, real *a, 
	integer *lda, real *b, integer *ldb, real *c__, real *d__, real *x, 
	real *work, integer *lwork, integer *info);

/* Subroutine */ int sggqrf_(integer *n, integer *m, integer *p, real *a, 
	integer *lda, real *taua, real *b, integer *ldb, real *taub, real *
	work, integer *lwork, integer *info);

/* Subroutine */ int sggrqf_(integer *m, integer *p, integer *n, real *a, 
	integer *lda, real *taua, real *b, integer *ldb, real *taub, real *
	work, integer *lwork, integer *info);

/* Subroutine */ int sggsvd_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *n, integer *p, integer *k, integer *l, real *a, integer *lda, 
	 real *b, integer *ldb, real *alpha, real *beta, real *u, integer *
	ldu, real *v, integer *ldv, real *q, integer *ldq, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int sggsvp_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, real *a, integer *lda, real *b, integer *ldb, 
	real *tola, real *tolb, integer *k, integer *l, real *u, integer *ldu, 
	 real *v, integer *ldv, real *q, integer *ldq, integer *iwork, real *
	tau, real *work, integer *info);

/* Subroutine */ int sgsvj0_(char *jobv, integer *m, integer *n, real *a, 
	integer *lda, real *d__, real *sva, integer *mv, real *v, integer *
	ldv, real *eps, real *sfmin, real *tol, integer *nsweep, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int sgsvj1_(char *jobv, integer *m, integer *n, integer *n1, 
	real *a, integer *lda, real *d__, real *sva, integer *mv, real *v, 
	integer *ldv, real *eps, real *sfmin, real *tol, integer *nsweep, 
	real *work, integer *lwork, integer *info);

/* Subroutine */ int sgtcon_(char *norm, integer *n, real *dl, real *d__, 
	real *du, real *du2, integer *ipiv, real *anorm, real *rcond, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int sgtrfs_(char *trans, integer *n, integer *nrhs, real *dl, 
	 real *d__, real *du, real *dlf, real *df, real *duf, real *du2, 
	integer *ipiv, real *b, integer *ldb, real *x, integer *ldx, real *
	ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int sgtsv_(integer *n, integer *nrhs, real *dl, real *d__, 
	real *du, real *b, integer *ldb, integer *info);

/* Subroutine */ int sgtsvx_(char *fact, char *trans, integer *n, integer *
	nrhs, real *dl, real *d__, real *du, real *dlf, real *df, real *duf, 
	real *du2, integer *ipiv, real *b, integer *ldb, real *x, integer *
	ldx, real *rcond, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int sgttrf_(integer *n, real *dl, real *d__, real *du, real *
	du2, integer *ipiv, integer *info);

/* Subroutine */ int sgttrs_(char *trans, integer *n, integer *nrhs, real *dl, 
	 real *d__, real *du, real *du2, integer *ipiv, real *b, integer *ldb, 
	 integer *info);

/* Subroutine */ int sgtts2_(integer *itrans, integer *n, integer *nrhs, real 
	*dl, real *d__, real *du, real *du2, integer *ipiv, real *b, integer *
	ldb);

/* Subroutine */ int shgeqz_(char *job, char *compq, char *compz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *t, integer 
	*ldt, real *alphar, real *alphai, real *beta, real *q, integer *ldq, 
	real *z__, integer *ldz, real *work, integer *lwork, integer *info);

/* Subroutine */ int shsein_(char *side, char *eigsrc, char *initv, logical *
	select, integer *n, real *h__, integer *ldh, real *wr, real *wi, real 
	*vl, integer *ldvl, real *vr, integer *ldvr, integer *mm, integer *m, 
	real *work, integer *ifaill, integer *ifailr, integer *info);

/* Subroutine */ int shseqr_(char *job, char *compz, integer *n, integer *ilo, 
	 integer *ihi, real *h__, integer *ldh, real *wr, real *wi, real *z__, 
	 integer *ldz, real *work, integer *lwork, integer *info);

logical sisnan_(real *sin__);

/* Subroutine */ int sla_gbamv__(integer *trans, integer *m, integer *n, 
	integer *kl, integer *ku, real *alpha, real *ab, integer *ldab, real *
	x, integer *incx, real *beta, real *y, integer *incy);

doublereal sla_gbrcond__(char *trans, integer *n, integer *kl, integer *ku, 
	real *ab, integer *ldab, real *afb, integer *ldafb, integer *ipiv, 
	integer *cmode, real *c__, integer *info, real *work, integer *iwork, 
	ftnlen trans_len);

/* Subroutine */ int sla_gbrfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *kl, integer *ku, integer *nrhs, 
	real *ab, integer *ldab, real *afb, integer *ldafb, integer *ipiv, 
	logical *colequ, real *c__, real *b, integer *ldb, real *y, integer *
	ldy, real *berr_out__, integer *n_norms__, real *errs_n__, real *
	errs_c__, real *res, real *ayb, real *dy, real *y_tail__, real *rcond,
	 integer *ithresh, real *rthresh, real *dz_ub__, logical *
	ignore_cwise__, integer *info);

doublereal sla_gbrpvgrw__(integer *n, integer *kl, integer *ku, integer *
	ncols, real *ab, integer *ldab, real *afb, integer *ldafb);

/* Subroutine */ int sla_geamv__(integer *trans, integer *m, integer *n, real 
	*alpha, real *a, integer *lda, real *x, integer *incx, real *beta, 
	real *y, integer *incy);

doublereal sla_gercond__(char *trans, integer *n, real *a, integer *lda, real 
	*af, integer *ldaf, integer *ipiv, integer *cmode, real *c__, integer 
	*info, real *work, integer *iwork, ftnlen trans_len);

/* Subroutine */ int sla_gerfsx_extended__(integer *prec_type__, integer *
	trans_type__, integer *n, integer *nrhs, real *a, integer *lda, real *
	af, integer *ldaf, integer *ipiv, logical *colequ, real *c__, real *b,
	 integer *ldb, real *y, integer *ldy, real *berr_out__, integer *
	n_norms__, real *errs_n__, real *errs_c__, real *res, real *ayb, real 
	*dy, real *y_tail__, real *rcond, integer *ithresh, real *rthresh, 
	real *dz_ub__, logical *ignore_cwise__, integer *info);

/* Subroutine */ int sla_lin_berr__(integer *n, integer *nz, integer *nrhs, 
	real *res, real *ayb, real *berr);

doublereal sla_porcond__(char *uplo, integer *n, real *a, integer *lda, real *
	af, integer *ldaf, integer *cmode, real *c__, integer *info, real *
	work, integer *iwork, ftnlen uplo_len);

/* Subroutine */ int sla_porfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, real *a, integer *lda, real *af, integer *
	ldaf, logical *colequ, real *c__, real *b, integer *ldb, real *y, 
	integer *ldy, real *berr_out__, integer *n_norms__, real *errs_n__, 
	real *errs_c__, real *res, real *ayb, real *dy, real *y_tail__, real *
	rcond, integer *ithresh, real *rthresh, real *dz_ub__, logical *
	ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal sla_porpvgrw__(char *uplo, integer *ncols, real *a, integer *lda, 
	real *af, integer *ldaf, real *work, ftnlen uplo_len);

doublereal sla_rpvgrw__(integer *n, integer *ncols, real *a, integer *lda, 
	real *af, integer *ldaf);

/* Subroutine */ int sla_syamv__(integer *uplo, integer *n, real *alpha, real 
	*a, integer *lda, real *x, integer *incx, real *beta, real *y, 
	integer *incy);

doublereal sla_syrcond__(char *uplo, integer *n, real *a, integer *lda, real *
	af, integer *ldaf, integer *ipiv, integer *cmode, real *c__, integer *
	info, real *work, integer *iwork, ftnlen uplo_len);

/* Subroutine */ int sla_syrfsx_extended__(integer *prec_type__, char *uplo, 
	integer *n, integer *nrhs, real *a, integer *lda, real *af, integer *
	ldaf, integer *ipiv, logical *colequ, real *c__, real *b, integer *
	ldb, real *y, integer *ldy, real *berr_out__, integer *n_norms__, 
	real *errs_n__, real *errs_c__, real *res, real *ayb, real *dy, real *
	y_tail__, real *rcond, integer *ithresh, real *rthresh, real *dz_ub__,
	 logical *ignore_cwise__, integer *info, ftnlen uplo_len);

doublereal sla_syrpvgrw__(char *uplo, integer *n, integer *info, real *a, 
	integer *lda, real *af, integer *ldaf, integer *ipiv, real *work, 
	ftnlen uplo_len);

/* Subroutine */ int sla_wwaddw__(integer *n, real *x, real *y, real *w);

/* Subroutine */ int slabad_(real *small, real *large);

/* Subroutine */ int slabrd_(integer *m, integer *n, integer *nb, real *a, 
	integer *lda, real *d__, real *e, real *tauq, real *taup, real *x, 
	integer *ldx, real *y, integer *ldy);

/* Subroutine */ int slacn2_(integer *n, real *v, real *x, integer *isgn, 
	real *est, integer *kase, integer *isave);

/* Subroutine */ int slacon_(integer *n, real *v, real *x, integer *isgn, 
	real *est, integer *kase);

/* Subroutine */ int slacpy_(char *uplo, integer *m, integer *n, real *a, 
	integer *lda, real *b, integer *ldb);

/* Subroutine */ int sladiv_(real *a, real *b, real *c__, real *d__, real *p, 
	real *q);

/* Subroutine */ int slae2_(real *a, real *b, real *c__, real *rt1, real *rt2);

/* Subroutine */ int slaebz_(integer *ijob, integer *nitmax, integer *n, 
	integer *mmax, integer *minp, integer *nbmin, real *abstol, real *
	reltol, real *pivmin, real *d__, real *e, real *e2, integer *nval, 
	real *ab, real *c__, integer *mout, integer *nab, real *work, integer 
	*iwork, integer *info);

/* Subroutine */ int slaed0_(integer *icompq, integer *qsiz, integer *n, real 
	*d__, real *e, real *q, integer *ldq, real *qstore, integer *ldqs, 
	real *work, integer *iwork, integer *info);

/* Subroutine */ int slaed1_(integer *n, real *d__, real *q, integer *ldq, 
	integer *indxq, real *rho, integer *cutpnt, real *work, integer *
	iwork, integer *info);

/* Subroutine */ int slaed2_(integer *k, integer *n, integer *n1, real *d__, 
	real *q, integer *ldq, integer *indxq, real *rho, real *z__, real *
	dlamda, real *w, real *q2, integer *indx, integer *indxc, integer *
	indxp, integer *coltyp, integer *info);

/* Subroutine */ int slaed3_(integer *k, integer *n, integer *n1, real *d__, 
	real *q, integer *ldq, real *rho, real *dlamda, real *q2, integer *
	indx, integer *ctot, real *w, real *s, integer *info);

/* Subroutine */ int slaed4_(integer *n, integer *i__, real *d__, real *z__, 
	real *delta, real *rho, real *dlam, integer *info);

/* Subroutine */ int slaed5_(integer *i__, real *d__, real *z__, real *delta, 
	real *rho, real *dlam);

/* Subroutine */ int slaed6_(integer *kniter, logical *orgati, real *rho, 
	real *d__, real *z__, real *finit, real *tau, integer *info);

/* Subroutine */ int slaed7_(integer *icompq, integer *n, integer *qsiz, 
	integer *tlvls, integer *curlvl, integer *curpbm, real *d__, real *q, 
	integer *ldq, integer *indxq, real *rho, integer *cutpnt, real *
	qstore, integer *qptr, integer *prmptr, integer *perm, integer *
	givptr, integer *givcol, real *givnum, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int slaed8_(integer *icompq, integer *k, integer *n, integer 
	*qsiz, real *d__, real *q, integer *ldq, integer *indxq, real *rho, 
	integer *cutpnt, real *z__, real *dlamda, real *q2, integer *ldq2, 
	real *w, integer *perm, integer *givptr, integer *givcol, real *
	givnum, integer *indxp, integer *indx, integer *info);

/* Subroutine */ int slaed9_(integer *k, integer *kstart, integer *kstop, 
	integer *n, real *d__, real *q, integer *ldq, real *rho, real *dlamda, 
	 real *w, real *s, integer *lds, integer *info);

/* Subroutine */ int slaeda_(integer *n, integer *tlvls, integer *curlvl, 
	integer *curpbm, integer *prmptr, integer *perm, integer *givptr, 
	integer *givcol, real *givnum, real *q, integer *qptr, real *z__, 
	real *ztemp, integer *info);

/* Subroutine */ int slaein_(logical *rightv, logical *noinit, integer *n, 
	real *h__, integer *ldh, real *wr, real *wi, real *vr, real *vi, real 
	*b, integer *ldb, real *work, real *eps3, real *smlnum, real *bignum, 
	integer *info);

/* Subroutine */ int slaev2_(real *a, real *b, real *c__, real *rt1, real *
	rt2, real *cs1, real *sn1);

/* Subroutine */ int slaexc_(logical *wantq, integer *n, real *t, integer *
	ldt, real *q, integer *ldq, integer *j1, integer *n1, integer *n2, 
	real *work, integer *info);

/* Subroutine */ int slag2_(real *a, integer *lda, real *b, integer *ldb, 
	real *safmin, real *scale1, real *scale2, real *wr1, real *wr2, real *
	wi);

/* Subroutine */ int slag2d_(integer *m, integer *n, real *sa, integer *ldsa, 
	doublereal *a, integer *lda, integer *info);

/* Subroutine */ int slags2_(logical *upper, real *a1, real *a2, real *a3, 
	real *b1, real *b2, real *b3, real *csu, real *snu, real *csv, real *
	snv, real *csq, real *snq);

/* Subroutine */ int slagtf_(integer *n, real *a, real *lambda, real *b, real 
	*c__, real *tol, real *d__, integer *in, integer *info);

/* Subroutine */ int slagtm_(char *trans, integer *n, integer *nrhs, real *
	alpha, real *dl, real *d__, real *du, real *x, integer *ldx, real *
	beta, real *b, integer *ldb);

/* Subroutine */ int slagts_(integer *job, integer *n, real *a, real *b, real 
	*c__, real *d__, integer *in, real *y, real *tol, integer *info);

/* Subroutine */ int slagv2_(real *a, integer *lda, real *b, integer *ldb, 
	real *alphar, real *alphai, real *beta, real *csl, real *snl, real *
	csr, real *snr);

/* Subroutine */ int slahqr_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *
	info);

/* Subroutine */ int slahr2_(integer *n, integer *k, integer *nb, real *a, 
	integer *lda, real *tau, real *t, integer *ldt, real *y, integer *ldy);

/* Subroutine */ int slahrd_(integer *n, integer *k, integer *nb, real *a, 
	integer *lda, real *tau, real *t, integer *ldt, real *y, integer *ldy);

/* Subroutine */ int slaic1_(integer *job, integer *j, real *x, real *sest, 
	real *w, real *gamma, real *sestpr, real *s, real *c__);

logical slaisnan_(real *sin1, real *sin2);

/* Subroutine */ int slaln2_(logical *ltrans, integer *na, integer *nw, real *
	smin, real *ca, real *a, integer *lda, real *d1, real *d2, real *b, 
	integer *ldb, real *wr, real *wi, real *x, integer *ldx, real *scale, 
	real *xnorm, integer *info);

/* Subroutine */ int slals0_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *nrhs, real *b, integer *ldb, real *bx, 
	integer *ldbx, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, real *givnum, integer *ldgnum, real *poles, real *
	difl, real *difr, real *z__, integer *k, real *c__, real *s, real *
	work, integer *info);

/* Subroutine */ int slalsa_(integer *icompq, integer *smlsiz, integer *n, 
	integer *nrhs, real *b, integer *ldb, real *bx, integer *ldbx, real *
	u, integer *ldu, real *vt, integer *k, real *difl, real *difr, real *
	z__, real *poles, integer *givptr, integer *givcol, integer *ldgcol, 
	integer *perm, real *givnum, real *c__, real *s, real *work, integer *
	iwork, integer *info);

/* Subroutine */ int slalsd_(char *uplo, integer *smlsiz, integer *n, integer 
	*nrhs, real *d__, real *e, real *b, integer *ldb, real *rcond, 
	integer *rank, real *work, integer *iwork, integer *info);

/* Subroutine */ int slamrg_(integer *n1, integer *n2, real *a, integer *
	strd1, integer *strd2, integer *index);

integer slaneg_(integer *n, real *d__, real *lld, real *sigma, real *pivmin, 
	integer *r__);

doublereal slangb_(char *norm, integer *n, integer *kl, integer *ku, real *ab, 
	 integer *ldab, real *work);

doublereal slange_(char *norm, integer *m, integer *n, real *a, integer *lda, 
	real *work);

doublereal slangt_(char *norm, integer *n, real *dl, real *d__, real *du);

doublereal slanhs_(char *norm, integer *n, real *a, integer *lda, real *work);

doublereal slansb_(char *norm, char *uplo, integer *n, integer *k, real *ab, 
	integer *ldab, real *work);

doublereal slansf_(char *norm, char *transr, char *uplo, integer *n, real *a, 
	real *work);

doublereal slansp_(char *norm, char *uplo, integer *n, real *ap, real *work);

doublereal slanst_(char *norm, integer *n, real *d__, real *e);

doublereal slansy_(char *norm, char *uplo, integer *n, real *a, integer *lda, 
	real *work);

doublereal slantb_(char *norm, char *uplo, char *diag, integer *n, integer *k, 
	 real *ab, integer *ldab, real *work);

doublereal slantp_(char *norm, char *uplo, char *diag, integer *n, real *ap, 
	real *work);

doublereal slantr_(char *norm, char *uplo, char *diag, integer *m, integer *n, 
	 real *a, integer *lda, real *work);

/* Subroutine */ int slanv2_(real *a, real *b, real *c__, real *d__, real *
	rt1r, real *rt1i, real *rt2r, real *rt2i, real *cs, real *sn);

/* Subroutine */ int slapll_(integer *n, real *x, integer *incx, real *y, 
	integer *incy, real *ssmin);

/* Subroutine */ int slapmt_(logical *forwrd, integer *m, integer *n, real *x, 
	 integer *ldx, integer *k);

doublereal slapy2_(real *x, real *y);

doublereal slapy3_(real *x, real *y, real *z__);

/* Subroutine */ int slaqgb_(integer *m, integer *n, integer *kl, integer *ku, 
	 real *ab, integer *ldab, real *r__, real *c__, real *rowcnd, real *
	colcnd, real *amax, char *equed);

/* Subroutine */ int slaqge_(integer *m, integer *n, real *a, integer *lda, 
	real *r__, real *c__, real *rowcnd, real *colcnd, real *amax, char *
	equed);

/* Subroutine */ int slaqp2_(integer *m, integer *n, integer *offset, real *a, 
	 integer *lda, integer *jpvt, real *tau, real *vn1, real *vn2, real *
	work);

/* Subroutine */ int slaqps_(integer *m, integer *n, integer *offset, integer 
	*nb, integer *kb, real *a, integer *lda, integer *jpvt, real *tau, 
	real *vn1, real *vn2, real *auxv, real *f, integer *ldf);

/* Subroutine */ int slaqr0_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, real *work, 
	 integer *lwork, integer *info);

/* Subroutine */ int slaqr1_(integer *n, real *h__, integer *ldh, real *sr1, 
	real *si1, real *sr2, real *si2, real *v);

/* Subroutine */ int slaqr2_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, real *h__, integer *ldh, 
	integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *ns, 
	integer *nd, real *sr, real *si, real *v, integer *ldv, integer *nh, 
	real *t, integer *ldt, integer *nv, real *wv, integer *ldwv, real *
	work, integer *lwork);

/* Subroutine */ int slaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ktop, integer *kbot, integer *nw, real *h__, integer *ldh, 
	integer *iloz, integer *ihiz, real *z__, integer *ldz, integer *ns, 
	integer *nd, real *sr, real *si, real *v, integer *ldv, integer *nh, 
	real *t, integer *ldt, integer *nv, real *wv, integer *ldwv, real *
	work, integer *lwork);

/* Subroutine */ int slaqr4_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, real *h__, integer *ldh, real *wr, real *
	wi, integer *iloz, integer *ihiz, real *z__, integer *ldz, real *work, 
	 integer *lwork, integer *info);

/* Subroutine */ int slaqr5_(logical *wantt, logical *wantz, integer *kacc22, 
	integer *n, integer *ktop, integer *kbot, integer *nshfts, real *sr, 
	real *si, real *h__, integer *ldh, integer *iloz, integer *ihiz, real 
	*z__, integer *ldz, real *v, integer *ldv, real *u, integer *ldu, 
	integer *nv, real *wv, integer *ldwv, integer *nh, real *wh, integer *
	ldwh);

/* Subroutine */ int slaqsb_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int slaqsp_(char *uplo, integer *n, real *ap, real *s, real *
	scond, real *amax, char *equed);

/* Subroutine */ int slaqsy_(char *uplo, integer *n, real *a, integer *lda, 
	real *s, real *scond, real *amax, char *equed);

/* Subroutine */ int slaqtr_(logical *ltran, logical *lreal, integer *n, real 
	*t, integer *ldt, real *b, real *w, real *scale, real *x, real *work, 
	integer *info);

/* Subroutine */ int slar1v_(integer *n, integer *b1, integer *bn, real *
	lambda, real *d__, real *l, real *ld, real *lld, real *pivmin, real *
	gaptol, real *z__, logical *wantnc, integer *negcnt, real *ztz, real *
	mingma, integer *r__, integer *isuppz, real *nrminv, real *resid, 
	real *rqcorr, real *work);

/* Subroutine */ int slar2v_(integer *n, real *x, real *y, real *z__, integer 
	*incx, real *c__, real *s, integer *incc);

/* Subroutine */ int slarf_(char *side, integer *m, integer *n, real *v, 
	integer *incv, real *tau, real *c__, integer *ldc, real *work);

/* Subroutine */ int slarfb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, real *v, integer *ldv, 
	real *t, integer *ldt, real *c__, integer *ldc, real *work, integer *
	ldwork);

/* Subroutine */ int slarfg_(integer *n, real *alpha, real *x, integer *incx, 
	real *tau);

/* Subroutine */ int slarfp_(integer *n, real *alpha, real *x, integer *incx, 
	real *tau);

/* Subroutine */ int slarft_(char *direct, char *storev, integer *n, integer *
	k, real *v, integer *ldv, real *tau, real *t, integer *ldt);

/* Subroutine */ int slarfx_(char *side, integer *m, integer *n, real *v, 
	real *tau, real *c__, integer *ldc, real *work);

/* Subroutine */ int slargv_(integer *n, real *x, integer *incx, real *y, 
	integer *incy, real *c__, integer *incc);

/* Subroutine */ int slarnv_(integer *idist, integer *iseed, integer *n, real 
	*x);

/* Subroutine */ int slarra_(integer *n, real *d__, real *e, real *e2, real *
	spltol, real *tnrm, integer *nsplit, integer *isplit, integer *info);

/* Subroutine */ int slarrb_(integer *n, real *d__, real *lld, integer *
	ifirst, integer *ilast, real *rtol1, real *rtol2, integer *offset, 
	real *w, real *wgap, real *werr, real *work, integer *iwork, real *
	pivmin, real *spdiam, integer *twist, integer *info);

/* Subroutine */ int slarrc_(char *jobt, integer *n, real *vl, real *vu, real 
	*d__, real *e, real *pivmin, integer *eigcnt, integer *lcnt, integer *
	rcnt, integer *info);

/* Subroutine */ int slarrd_(char *range, char *order, integer *n, real *vl, 
	real *vu, integer *il, integer *iu, real *gers, real *reltol, real *
	d__, real *e, real *e2, real *pivmin, integer *nsplit, integer *
	isplit, integer *m, real *w, real *werr, real *wl, real *wu, integer *
	iblock, integer *indexw, real *work, integer *iwork, integer *info);

/* Subroutine */ int slarre_(char *range, integer *n, real *vl, real *vu, 
	integer *il, integer *iu, real *d__, real *e, real *e2, real *rtol1, 
	real *rtol2, real *spltol, integer *nsplit, integer *isplit, integer *
	m, real *w, real *werr, real *wgap, integer *iblock, integer *indexw, 
	real *gers, real *pivmin, real *work, integer *iwork, integer *info);

/* Subroutine */ int slarrf_(integer *n, real *d__, real *l, real *ld, 
	integer *clstrt, integer *clend, real *w, real *wgap, real *werr, 
	real *spdiam, real *clgapl, real *clgapr, real *pivmin, real *sigma, 
	real *dplus, real *lplus, real *work, integer *info);

/* Subroutine */ int slarrj_(integer *n, real *d__, real *e2, integer *ifirst, 
	 integer *ilast, real *rtol, integer *offset, real *w, real *werr, 
	real *work, integer *iwork, real *pivmin, real *spdiam, integer *info);

/* Subroutine */ int slarrk_(integer *n, integer *iw, real *gl, real *gu, 
	real *d__, real *e2, real *pivmin, real *reltol, real *w, real *werr, 
	integer *info);

/* Subroutine */ int slarrr_(integer *n, real *d__, real *e, integer *info);

/* Subroutine */ int slarrv_(integer *n, real *vl, real *vu, real *d__, real *
	l, real *pivmin, integer *isplit, integer *m, integer *dol, integer *
	dou, real *minrgp, real *rtol1, real *rtol2, real *w, real *werr, 
	real *wgap, integer *iblock, integer *indexw, real *gers, real *z__, 
	integer *ldz, integer *isuppz, real *work, integer *iwork, integer *
	info);

/* Subroutine */ int slarscl2_(integer *m, integer *n, real *d__, real *x, 
	integer *ldx);

/* Subroutine */ int slartg_(real *f, real *g, real *cs, real *sn, real *r__);

/* Subroutine */ int slartv_(integer *n, real *x, integer *incx, real *y, 
	integer *incy, real *c__, real *s, integer *incc);

/* Subroutine */ int slaruv_(integer *iseed, integer *n, real *x);

/* Subroutine */ int slarz_(char *side, integer *m, integer *n, integer *l, 
	real *v, integer *incv, real *tau, real *c__, integer *ldc, real *
	work);

/* Subroutine */ int slarzb_(char *side, char *trans, char *direct, char *
	storev, integer *m, integer *n, integer *k, integer *l, real *v, 
	integer *ldv, real *t, integer *ldt, real *c__, integer *ldc, real *
	work, integer *ldwork);

/* Subroutine */ int slarzt_(char *direct, char *storev, integer *n, integer *
	k, real *v, integer *ldv, real *tau, real *t, integer *ldt);

/* Subroutine */ int slas2_(real *f, real *g, real *h__, real *ssmin, real *
	ssmax);

/* Subroutine */ int slascl_(char *type__, integer *kl, integer *ku, real *
	cfrom, real *cto, integer *m, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int slascl2_(integer *m, integer *n, real *d__, real *x, 
	integer *ldx);

/* Subroutine */ int slasd0_(integer *n, integer *sqre, real *d__, real *e, 
	real *u, integer *ldu, real *vt, integer *ldvt, integer *smlsiz, 
	integer *iwork, real *work, integer *info);

/* Subroutine */ int slasd1_(integer *nl, integer *nr, integer *sqre, real *
	d__, real *alpha, real *beta, real *u, integer *ldu, real *vt, 
	integer *ldvt, integer *idxq, integer *iwork, real *work, integer *
	info);

/* Subroutine */ int slasd2_(integer *nl, integer *nr, integer *sqre, integer 
	*k, real *d__, real *z__, real *alpha, real *beta, real *u, integer *
	ldu, real *vt, integer *ldvt, real *dsigma, real *u2, integer *ldu2, 
	real *vt2, integer *ldvt2, integer *idxp, integer *idx, integer *idxc, 
	 integer *idxq, integer *coltyp, integer *info);

/* Subroutine */ int slasd3_(integer *nl, integer *nr, integer *sqre, integer 
	*k, real *d__, real *q, integer *ldq, real *dsigma, real *u, integer *
	ldu, real *u2, integer *ldu2, real *vt, integer *ldvt, real *vt2, 
	integer *ldvt2, integer *idxc, integer *ctot, real *z__, integer *
	info);

/* Subroutine */ int slasd4_(integer *n, integer *i__, real *d__, real *z__, 
	real *delta, real *rho, real *sigma, real *work, integer *info);

/* Subroutine */ int slasd5_(integer *i__, real *d__, real *z__, real *delta, 
	real *rho, real *dsigma, real *work);

/* Subroutine */ int slasd6_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, real *d__, real *vf, real *vl, real *alpha, real *beta, 
	 integer *idxq, integer *perm, integer *givptr, integer *givcol, 
	integer *ldgcol, real *givnum, integer *ldgnum, real *poles, real *
	difl, real *difr, real *z__, integer *k, real *c__, real *s, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int slasd7_(integer *icompq, integer *nl, integer *nr, 
	integer *sqre, integer *k, real *d__, real *z__, real *zw, real *vf, 
	real *vfw, real *vl, real *vlw, real *alpha, real *beta, real *dsigma, 
	 integer *idx, integer *idxp, integer *idxq, integer *perm, integer *
	givptr, integer *givcol, integer *ldgcol, real *givnum, integer *
	ldgnum, real *c__, real *s, integer *info);

/* Subroutine */ int slasd8_(integer *icompq, integer *k, real *d__, real *
	z__, real *vf, real *vl, real *difl, real *difr, integer *lddifr, 
	real *dsigma, real *work, integer *info);

/* Subroutine */ int slasda_(integer *icompq, integer *smlsiz, integer *n, 
	integer *sqre, real *d__, real *e, real *u, integer *ldu, real *vt, 
	integer *k, real *difl, real *difr, real *z__, real *poles, integer *
	givptr, integer *givcol, integer *ldgcol, integer *perm, real *givnum, 
	 real *c__, real *s, real *work, integer *iwork, integer *info);

/* Subroutine */ int slasdq_(char *uplo, integer *sqre, integer *n, integer *
	ncvt, integer *nru, integer *ncc, real *d__, real *e, real *vt, 
	integer *ldvt, real *u, integer *ldu, real *c__, integer *ldc, real *
	work, integer *info);

/* Subroutine */ int slasdt_(integer *n, integer *lvl, integer *nd, integer *
	inode, integer *ndiml, integer *ndimr, integer *msub);

/* Subroutine */ int slaset_(char *uplo, integer *m, integer *n, real *alpha, 
	real *beta, real *a, integer *lda);

/* Subroutine */ int slasq1_(integer *n, real *d__, real *e, real *work, 
	integer *info);

/* Subroutine */ int slasq2_(integer *n, real *z__, integer *info);

/* Subroutine */ int slasq3_(integer *i0, integer *n0, real *z__, integer *pp, 
	 real *dmin__, real *sigma, real *desig, real *qmax, integer *nfail, 
	integer *iter, integer *ndiv, logical *ieee, integer *ttype, real *
	dmin1, real *dmin2, real *dn, real *dn1, real *dn2, real *g, real *
	tau);

/* Subroutine */ int slasq4_(integer *i0, integer *n0, real *z__, integer *pp, 
	 integer *n0in, real *dmin__, real *dmin1, real *dmin2, real *dn, 
	real *dn1, real *dn2, real *tau, integer *ttype, real *g);

/* Subroutine */ int slasq5_(integer *i0, integer *n0, real *z__, integer *pp, 
	 real *tau, real *dmin__, real *dmin1, real *dmin2, real *dn, real *
	dnm1, real *dnm2, logical *ieee);

/* Subroutine */ int slasq6_(integer *i0, integer *n0, real *z__, integer *pp, 
	 real *dmin__, real *dmin1, real *dmin2, real *dn, real *dnm1, real *
	dnm2);

/* Subroutine */ int slasr_(char *side, char *pivot, char *direct, integer *m, 
	 integer *n, real *c__, real *s, real *a, integer *lda);

/* Subroutine */ int slasrt_(char *id, integer *n, real *d__, integer *info);

/* Subroutine */ int slassq_(integer *n, real *x, integer *incx, real *scale, 
	real *sumsq);

/* Subroutine */ int slasv2_(real *f, real *g, real *h__, real *ssmin, real *
	ssmax, real *snr, real *csr, real *snl, real *csl);

/* Subroutine */ int slaswp_(integer *n, real *a, integer *lda, integer *k1, 
	integer *k2, integer *ipiv, integer *incx);

/* Subroutine */ int slasy2_(logical *ltranl, logical *ltranr, integer *isgn, 
	integer *n1, integer *n2, real *tl, integer *ldtl, real *tr, integer *
	ldtr, real *b, integer *ldb, real *scale, real *x, integer *ldx, real 
	*xnorm, integer *info);

/* Subroutine */ int slasyf_(char *uplo, integer *n, integer *nb, integer *kb, 
	 real *a, integer *lda, integer *ipiv, real *w, integer *ldw, integer 
	*info);

/* Subroutine */ int slatbs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, integer *kd, real *ab, integer *ldab, real *x, 
	real *scale, real *cnorm, integer *info);

/* Subroutine */ int slatdf_(integer *ijob, integer *n, real *z__, integer *
	ldz, real *rhs, real *rdsum, real *rdscal, integer *ipiv, integer *
	jpiv);

/* Subroutine */ int slatps_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, real *ap, real *x, real *scale, real *cnorm, 
	integer *info);

/* Subroutine */ int slatrd_(char *uplo, integer *n, integer *nb, real *a, 
	integer *lda, real *e, real *tau, real *w, integer *ldw);

/* Subroutine */ int slatrs_(char *uplo, char *trans, char *diag, char *
	normin, integer *n, real *a, integer *lda, real *x, real *scale, real 
	*cnorm, integer *info);

/* Subroutine */ int slatrz_(integer *m, integer *n, integer *l, real *a, 
	integer *lda, real *tau, real *work);

/* Subroutine */ int slatzm_(char *side, integer *m, integer *n, real *v, 
	integer *incv, real *tau, real *c1, real *c2, integer *ldc, real *
	work);

/* Subroutine */ int slauu2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int slauum_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int sopgtr_(char *uplo, integer *n, real *ap, real *tau, 
	real *q, integer *ldq, real *work, integer *info);

/* Subroutine */ int sopmtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, real *ap, real *tau, real *c__, integer *ldc, real *work, 
	integer *info);

/* Subroutine */ int sorg2l_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int sorg2r_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int sorgbr_(char *vect, integer *m, integer *n, integer *k, 
	real *a, integer *lda, real *tau, real *work, integer *lwork, integer 
	*info);

/* Subroutine */ int sorghr_(integer *n, integer *ilo, integer *ihi, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sorgl2_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int sorglq_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sorgql_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sorgqr_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sorgr2_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *info);

/* Subroutine */ int sorgrq_(integer *m, integer *n, integer *k, real *a, 
	integer *lda, real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sorgtr_(char *uplo, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int sorm2l_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int sorm2r_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int sormbr_(char *vect, char *side, char *trans, integer *m, 
	integer *n, integer *k, real *a, integer *lda, real *tau, real *c__, 
	integer *ldc, real *work, integer *lwork, integer *info);

/* Subroutine */ int sormhr_(char *side, char *trans, integer *m, integer *n, 
	integer *ilo, integer *ihi, real *a, integer *lda, real *tau, real *
	c__, integer *ldc, real *work, integer *lwork, integer *info);

/* Subroutine */ int sorml2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int sormlq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int sormql_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int sormqr_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int sormr2_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *info);

/* Subroutine */ int sormr3_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, real *a, integer *lda, real *tau, real *c__, 
	integer *ldc, real *work, integer *info);

/* Subroutine */ int sormrq_(char *side, char *trans, integer *m, integer *n, 
	integer *k, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int sormrz_(char *side, char *trans, integer *m, integer *n, 
	integer *k, integer *l, real *a, integer *lda, real *tau, real *c__, 
	integer *ldc, real *work, integer *lwork, integer *info);

/* Subroutine */ int sormtr_(char *side, char *uplo, char *trans, integer *m, 
	integer *n, real *a, integer *lda, real *tau, real *c__, integer *ldc, 
	 real *work, integer *lwork, integer *info);

/* Subroutine */ int spbcon_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, real *anorm, real *rcond, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int spbequ_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, real *s, real *scond, real *amax, integer *info);

/* Subroutine */ int spbrfs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, real *ab, integer *ldab, real *afb, integer *ldafb, real *b, 
	integer *ldb, real *x, integer *ldx, real *ferr, real *berr, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int spbstf_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, integer *info);

/* Subroutine */ int spbsv_(char *uplo, integer *n, integer *kd, integer *
	nrhs, real *ab, integer *ldab, real *b, integer *ldb, integer *info);

/* Subroutine */ int spbsvx_(char *fact, char *uplo, integer *n, integer *kd, 
	integer *nrhs, real *ab, integer *ldab, real *afb, integer *ldafb, 
	char *equed, real *s, real *b, integer *ldb, real *x, integer *ldx, 
	real *rcond, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int spbtf2_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, integer *info);

/* Subroutine */ int spbtrf_(char *uplo, integer *n, integer *kd, real *ab, 
	integer *ldab, integer *info);

/* Subroutine */ int spbtrs_(char *uplo, integer *n, integer *kd, integer *
	nrhs, real *ab, integer *ldab, real *b, integer *ldb, integer *info);

/* Subroutine */ int spftrf_(char *transr, char *uplo, integer *n, real *a, 
	integer *info);

/* Subroutine */ int spftri_(char *transr, char *uplo, integer *n, real *a, 
	integer *info);

/* Subroutine */ int spftrs_(char *transr, char *uplo, integer *n, integer *
	nrhs, real *a, real *b, integer *ldb, integer *info);

/* Subroutine */ int spocon_(char *uplo, integer *n, real *a, integer *lda, 
	real *anorm, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int spoequ_(integer *n, real *a, integer *lda, real *s, real 
	*scond, real *amax, integer *info);

/* Subroutine */ int spoequb_(integer *n, real *a, integer *lda, real *s, 
	real *scond, real *amax, integer *info);

/* Subroutine */ int sporfs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *af, integer *ldaf, real *b, integer *ldb, real *x, 
	 integer *ldx, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int sporfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, real *s, real *
	b, integer *ldb, real *x, integer *ldx, real *rcond, real *berr, 
	integer *n_err_bnds__, real *err_bnds_norm__, real *err_bnds_comp__, 
	integer *nparams, real *params, real *work, integer *iwork, integer *
	info);

/* Subroutine */ int sposv_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int sposvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, char *equed, 
	real *s, real *b, integer *ldb, real *x, integer *ldx, real *rcond, 
	real *ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int sposvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, char *equed, 
	real *s, real *b, integer *ldb, real *x, integer *ldx, real *rcond, 
	real *rpvgrw, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, real *work, integer *iwork, integer *info);

/* Subroutine */ int spotf2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int spotrf_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int spotri_(char *uplo, integer *n, real *a, integer *lda, 
	integer *info);

/* Subroutine */ int spotrs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int sppcon_(char *uplo, integer *n, real *ap, real *anorm, 
	real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int sppequ_(char *uplo, integer *n, real *ap, real *s, real *
	scond, real *amax, integer *info);

/* Subroutine */ int spprfs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *afp, real *b, integer *ldb, real *x, integer *ldx, real *ferr, 
	real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int sppsv_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int sppsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *ap, real *afp, char *equed, real *s, real *b, integer *
	ldb, real *x, integer *ldx, real *rcond, real *ferr, real *berr, real 
	*work, integer *iwork, integer *info);

/* Subroutine */ int spptrf_(char *uplo, integer *n, real *ap, integer *info);

/* Subroutine */ int spptri_(char *uplo, integer *n, real *ap, integer *info);

/* Subroutine */ int spptrs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int spstf2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *piv, integer *rank, real *tol, real *work, integer *info);

/* Subroutine */ int spstrf_(char *uplo, integer *n, real *a, integer *lda, 
	integer *piv, integer *rank, real *tol, real *work, integer *info);

/* Subroutine */ int sptcon_(integer *n, real *d__, real *e, real *anorm, 
	real *rcond, real *work, integer *info);

/* Subroutine */ int spteqr_(char *compz, integer *n, real *d__, real *e, 
	real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int sptrfs_(integer *n, integer *nrhs, real *d__, real *e, 
	real *df, real *ef, real *b, integer *ldb, real *x, integer *ldx, 
	real *ferr, real *berr, real *work, integer *info);

/* Subroutine */ int sptsv_(integer *n, integer *nrhs, real *d__, real *e, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int sptsvx_(char *fact, integer *n, integer *nrhs, real *d__, 
	 real *e, real *df, real *ef, real *b, integer *ldb, real *x, integer 
	*ldx, real *rcond, real *ferr, real *berr, real *work, integer *info);

/* Subroutine */ int spttrf_(integer *n, real *d__, real *e, integer *info);

/* Subroutine */ int spttrs_(integer *n, integer *nrhs, real *d__, real *e, 
	real *b, integer *ldb, integer *info);

/* Subroutine */ int sptts2_(integer *n, integer *nrhs, real *d__, real *e, 
	real *b, integer *ldb);

/* Subroutine */ int srscl_(integer *n, real *sa, real *sx, integer *incx);

/* Subroutine */ int ssbev_(char *jobz, char *uplo, integer *n, integer *kd, 
	real *ab, integer *ldab, real *w, real *z__, integer *ldz, real *work, 
	 integer *info);

/* Subroutine */ int ssbevd_(char *jobz, char *uplo, integer *n, integer *kd, 
	real *ab, integer *ldab, real *w, real *z__, integer *ldz, real *work, 
	 integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int ssbevx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *kd, real *ab, integer *ldab, real *q, integer *ldq, real *vl, 
	 real *vu, integer *il, integer *iu, real *abstol, integer *m, real *
	w, real *z__, integer *ldz, real *work, integer *iwork, integer *
	ifail, integer *info);

/* Subroutine */ int ssbgst_(char *vect, char *uplo, integer *n, integer *ka, 
	integer *kb, real *ab, integer *ldab, real *bb, integer *ldbb, real *
	x, integer *ldx, real *work, integer *info);

/* Subroutine */ int ssbgv_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, real *ab, integer *ldab, real *bb, integer *ldbb, real *
	w, real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int ssbgvd_(char *jobz, char *uplo, integer *n, integer *ka, 
	integer *kb, real *ab, integer *ldab, real *bb, integer *ldbb, real *
	w, real *z__, integer *ldz, real *work, integer *lwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int ssbgvx_(char *jobz, char *range, char *uplo, integer *n, 
	integer *ka, integer *kb, real *ab, integer *ldab, real *bb, integer *
	ldbb, real *q, integer *ldq, real *vl, real *vu, integer *il, integer 
	*iu, real *abstol, integer *m, real *w, real *z__, integer *ldz, real 
	*work, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int ssbtrd_(char *vect, char *uplo, integer *n, integer *kd, 
	real *ab, integer *ldab, real *d__, real *e, real *q, integer *ldq, 
	real *work, integer *info);

/* Subroutine */ int ssfrk_(char *transr, char *uplo, char *trans, integer *n, 
	 integer *k, real *alpha, real *a, integer *lda, real *beta, real *
	c__);

/* Subroutine */ int sspcon_(char *uplo, integer *n, real *ap, integer *ipiv, 
	real *anorm, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int sspev_(char *jobz, char *uplo, integer *n, real *ap, 
	real *w, real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int sspevd_(char *jobz, char *uplo, integer *n, real *ap, 
	real *w, real *z__, integer *ldz, real *work, integer *lwork, integer 
	*iwork, integer *liwork, integer *info);

/* Subroutine */ int sspevx_(char *jobz, char *range, char *uplo, integer *n, 
	real *ap, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, real *work, integer *
	iwork, integer *ifail, integer *info);

/* Subroutine */ int sspgst_(integer *itype, char *uplo, integer *n, real *ap, 
	 real *bp, integer *info);

/* Subroutine */ int sspgv_(integer *itype, char *jobz, char *uplo, integer *
	n, real *ap, real *bp, real *w, real *z__, integer *ldz, real *work, 
	integer *info);

/* Subroutine */ int sspgvd_(integer *itype, char *jobz, char *uplo, integer *
	n, real *ap, real *bp, real *w, real *z__, integer *ldz, real *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int sspgvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, real *ap, real *bp, real *vl, real *vu, integer *il, 
	 integer *iu, real *abstol, integer *m, real *w, real *z__, integer *
	ldz, real *work, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int ssprfs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	real *afp, integer *ipiv, real *b, integer *ldb, real *x, integer *
	ldx, real *ferr, real *berr, real *work, integer *iwork, integer *
	info);

/* Subroutine */ int sspsv_(char *uplo, integer *n, integer *nrhs, real *ap, 
	integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int sspsvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *ap, real *afp, integer *ipiv, real *b, integer *ldb, real 
	*x, integer *ldx, real *rcond, real *ferr, real *berr, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int ssptrd_(char *uplo, integer *n, real *ap, real *d__, 
	real *e, real *tau, integer *info);

/* Subroutine */ int ssptrf_(char *uplo, integer *n, real *ap, integer *ipiv, 
	integer *info);

/* Subroutine */ int ssptri_(char *uplo, integer *n, real *ap, integer *ipiv, 
	real *work, integer *info);

/* Subroutine */ int ssptrs_(char *uplo, integer *n, integer *nrhs, real *ap, 
	integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int sstebz_(char *range, char *order, integer *n, real *vl, 
	real *vu, integer *il, integer *iu, real *abstol, real *d__, real *e, 
	integer *m, integer *nsplit, real *w, integer *iblock, integer *
	isplit, real *work, integer *iwork, integer *info);

/* Subroutine */ int sstedc_(char *compz, integer *n, real *d__, real *e, 
	real *z__, integer *ldz, real *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int sstegr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, integer *isuppz, real *
	work, integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int sstein_(integer *n, real *d__, real *e, integer *m, real 
	*w, integer *iblock, integer *isplit, real *z__, integer *ldz, real *
	work, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int sstemr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, integer *m, 
	real *w, real *z__, integer *ldz, integer *nzc, integer *isuppz, 
	logical *tryrac, real *work, integer *lwork, integer *iwork, integer *
	liwork, integer *info);

/* Subroutine */ int ssteqr_(char *compz, integer *n, real *d__, real *e, 
	real *z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int ssterf_(integer *n, real *d__, real *e, integer *info);

/* Subroutine */ int sstev_(char *jobz, integer *n, real *d__, real *e, real *
	z__, integer *ldz, real *work, integer *info);

/* Subroutine */ int sstevd_(char *jobz, integer *n, real *d__, real *e, real 
	*z__, integer *ldz, real *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int sstevr_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, integer *isuppz, real *
	work, integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int sstevx_(char *jobz, char *range, integer *n, real *d__, 
	real *e, real *vl, real *vu, integer *il, integer *iu, real *abstol, 
	integer *m, real *w, real *z__, integer *ldz, real *work, integer *
	iwork, integer *ifail, integer *info);

/* Subroutine */ int ssycon_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, real *anorm, real *rcond, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int ssyequb_(char *uplo, integer *n, real *a, integer *lda, 
	real *s, real *scond, real *amax, real *work, integer *info);

/* Subroutine */ int ssyev_(char *jobz, char *uplo, integer *n, real *a, 
	integer *lda, real *w, real *work, integer *lwork, integer *info);

/* Subroutine */ int ssyevd_(char *jobz, char *uplo, integer *n, real *a, 
	integer *lda, real *w, real *work, integer *lwork, integer *iwork, 
	integer *liwork, integer *info);

/* Subroutine */ int ssyevr_(char *jobz, char *range, char *uplo, integer *n, 
	real *a, integer *lda, real *vl, real *vu, integer *il, integer *iu, 
	real *abstol, integer *m, real *w, real *z__, integer *ldz, integer *
	isuppz, real *work, integer *lwork, integer *iwork, integer *liwork, 
	integer *info);

/* Subroutine */ int ssyevx_(char *jobz, char *range, char *uplo, integer *n, 
	real *a, integer *lda, real *vl, real *vu, integer *il, integer *iu, 
	real *abstol, integer *m, real *w, real *z__, integer *ldz, real *
	work, integer *lwork, integer *iwork, integer *ifail, integer *info);

/* Subroutine */ int ssygs2_(integer *itype, char *uplo, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int ssygst_(integer *itype, char *uplo, integer *n, real *a, 
	integer *lda, real *b, integer *ldb, integer *info);

/* Subroutine */ int ssygv_(integer *itype, char *jobz, char *uplo, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *w, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int ssygvd_(integer *itype, char *jobz, char *uplo, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *w, real *work, 
	integer *lwork, integer *iwork, integer *liwork, integer *info);

/* Subroutine */ int ssygvx_(integer *itype, char *jobz, char *range, char *
	uplo, integer *n, real *a, integer *lda, real *b, integer *ldb, real *
	vl, real *vu, integer *il, integer *iu, real *abstol, integer *m, 
	real *w, real *z__, integer *ldz, real *work, integer *lwork, integer 
	*iwork, integer *ifail, integer *info);

/* Subroutine */ int ssyrfs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, real *af, integer *ldaf, integer *ipiv, real *b, 
	integer *ldb, real *x, integer *ldx, real *ferr, real *berr, real *
	work, integer *iwork, integer *info);

/* Subroutine */ int ssyrfsx_(char *uplo, char *equed, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	real *s, real *b, integer *ldb, real *x, integer *ldx, real *rcond, 
	real *berr, integer *n_err_bnds__, real *err_bnds_norm__, real *
	err_bnds_comp__, integer *nparams, real *params, real *work, integer *
	iwork, integer *info);

/* Subroutine */ int ssysv_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, integer *ipiv, real *b, integer *ldb, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int ssysvx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	real *b, integer *ldb, real *x, integer *ldx, real *rcond, real *ferr, 
	 real *berr, real *work, integer *lwork, integer *iwork, integer *
	info);

/* Subroutine */ int ssysvxx_(char *fact, char *uplo, integer *n, integer *
	nrhs, real *a, integer *lda, real *af, integer *ldaf, integer *ipiv, 
	char *equed, real *s, real *b, integer *ldb, real *x, integer *ldx, 
	real *rcond, real *rpvgrw, real *berr, integer *n_err_bnds__, real *
	err_bnds_norm__, real *err_bnds_comp__, integer *nparams, real *
	params, real *work, integer *iwork, integer *info);

/* Subroutine */ int ssytd2_(char *uplo, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tau, integer *info);

/* Subroutine */ int ssytf2_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, integer *info);

/* Subroutine */ int ssytrd_(char *uplo, integer *n, real *a, integer *lda, 
	real *d__, real *e, real *tau, real *work, integer *lwork, integer *
	info);

/* Subroutine */ int ssytrf_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, real *work, integer *lwork, integer *info);

/* Subroutine */ int ssytri_(char *uplo, integer *n, real *a, integer *lda, 
	integer *ipiv, real *work, integer *info);

/* Subroutine */ int ssytrs_(char *uplo, integer *n, integer *nrhs, real *a, 
	integer *lda, integer *ipiv, real *b, integer *ldb, integer *info);

/* Subroutine */ int stbcon_(char *norm, char *uplo, char *diag, integer *n, 
	integer *kd, real *ab, integer *ldab, real *rcond, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int stbrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, real *ab, integer *ldab, real *b, integer 
	*ldb, real *x, integer *ldx, real *ferr, real *berr, real *work, 
	integer *iwork, integer *info);

/* Subroutine */ int stbtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *kd, integer *nrhs, real *ab, integer *ldab, real *b, integer 
	*ldb, integer *info);

/* Subroutine */ int stfsm_(char *transr, char *side, char *uplo, char *trans, 
	 char *diag, integer *m, integer *n, real *alpha, real *a, real *b, 
	integer *ldb);

/* Subroutine */ int stftri_(char *transr, char *uplo, char *diag, integer *n, 
	 real *a, integer *info);

/* Subroutine */ int stfttp_(char *transr, char *uplo, integer *n, real *arf, 
	real *ap, integer *info);

/* Subroutine */ int stfttr_(char *transr, char *uplo, integer *n, real *arf, 
	real *a, integer *lda, integer *info);

/* Subroutine */ int stgevc_(char *side, char *howmny, logical *select, 
	integer *n, real *s, integer *lds, real *p, integer *ldp, real *vl, 
	integer *ldvl, real *vr, integer *ldvr, integer *mm, integer *m, real 
	*work, integer *info);

/* Subroutine */ int stgex2_(logical *wantq, logical *wantz, integer *n, real 
	*a, integer *lda, real *b, integer *ldb, real *q, integer *ldq, real *
	z__, integer *ldz, integer *j1, integer *n1, integer *n2, real *work, 
	integer *lwork, integer *info);

/* Subroutine */ int stgexc_(logical *wantq, logical *wantz, integer *n, real 
	*a, integer *lda, real *b, integer *ldb, real *q, integer *ldq, real *
	z__, integer *ldz, integer *ifst, integer *ilst, real *work, integer *
	lwork, integer *info);

/* Subroutine */ int stgsen_(integer *ijob, logical *wantq, logical *wantz, 
	logical *select, integer *n, real *a, integer *lda, real *b, integer *
	ldb, real *alphar, real *alphai, real *beta, real *q, integer *ldq, 
	real *z__, integer *ldz, integer *m, real *pl, real *pr, real *dif, 
	real *work, integer *lwork, integer *iwork, integer *liwork, integer *
	info);

/* Subroutine */ int stgsja_(char *jobu, char *jobv, char *jobq, integer *m, 
	integer *p, integer *n, integer *k, integer *l, real *a, integer *lda, 
	 real *b, integer *ldb, real *tola, real *tolb, real *alpha, real *
	beta, real *u, integer *ldu, real *v, integer *ldv, real *q, integer *
	ldq, real *work, integer *ncycle, integer *info);

/* Subroutine */ int stgsna_(char *job, char *howmny, logical *select, 
	integer *n, real *a, integer *lda, real *b, integer *ldb, real *vl, 
	integer *ldvl, real *vr, integer *ldvr, real *s, real *dif, integer *
	mm, integer *m, real *work, integer *lwork, integer *iwork, integer *
	info);

/* Subroutine */ int stgsy2_(char *trans, integer *ijob, integer *m, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *c__, integer *
	ldc, real *d__, integer *ldd, real *e, integer *lde, real *f, integer 
	*ldf, real *scale, real *rdsum, real *rdscal, integer *iwork, integer 
	*pq, integer *info);

/* Subroutine */ int stgsyl_(char *trans, integer *ijob, integer *m, integer *
	n, real *a, integer *lda, real *b, integer *ldb, real *c__, integer *
	ldc, real *d__, integer *ldd, real *e, integer *lde, real *f, integer 
	*ldf, real *scale, real *dif, real *work, integer *lwork, integer *
	iwork, integer *info);

/* Subroutine */ int stpcon_(char *norm, char *uplo, char *diag, integer *n, 
	real *ap, real *rcond, real *work, integer *iwork, integer *info);

/* Subroutine */ int stprfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *ap, real *b, integer *ldb, real *x, integer *ldx, 
	 real *ferr, real *berr, real *work, integer *iwork, integer *info);

/* Subroutine */ int stptri_(char *uplo, char *diag, integer *n, real *ap, 
	integer *info);

/* Subroutine */ int stptrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *ap, real *b, integer *ldb, integer *info);

/* Subroutine */ int stpttf_(char *transr, char *uplo, integer *n, real *ap, 
	real *arf, integer *info);

/* Subroutine */ int stpttr_(char *uplo, integer *n, real *ap, real *a, 
	integer *lda, integer *info);

/* Subroutine */ int strcon_(char *norm, char *uplo, char *diag, integer *n, 
	real *a, integer *lda, real *rcond, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int strevc_(char *side, char *howmny, logical *select, 
	integer *n, real *t, integer *ldt, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, integer *mm, integer *m, real *work, integer *info);

/* Subroutine */ int strexc_(char *compq, integer *n, real *t, integer *ldt, 
	real *q, integer *ldq, integer *ifst, integer *ilst, real *work, 
	integer *info);

/* Subroutine */ int strrfs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *a, integer *lda, real *b, integer *ldb, real *x, 
	integer *ldx, real *ferr, real *berr, real *work, integer *iwork, 
	integer *info);

/* Subroutine */ int strsen_(char *job, char *compq, logical *select, integer 
	*n, real *t, integer *ldt, real *q, integer *ldq, real *wr, real *wi, 
	integer *m, real *s, real *sep, real *work, integer *lwork, integer *
	iwork, integer *liwork, integer *info);

/* Subroutine */ int strsna_(char *job, char *howmny, logical *select, 
	integer *n, real *t, integer *ldt, real *vl, integer *ldvl, real *vr, 
	integer *ldvr, real *s, real *sep, integer *mm, integer *m, real *
	work, integer *ldwork, integer *iwork, integer *info);

/* Subroutine */ int strsyl_(char *trana, char *tranb, integer *isgn, integer 
	*m, integer *n, real *a, integer *lda, real *b, integer *ldb, real *
	c__, integer *ldc, real *scale, integer *info);

/* Subroutine */ int strti2_(char *uplo, char *diag, integer *n, real *a, 
	integer *lda, integer *info);

/* Subroutine */ int strtri_(char *uplo, char *diag, integer *n, real *a, 
	integer *lda, integer *info);

/* Subroutine */ int strtrs_(char *uplo, char *trans, char *diag, integer *n, 
	integer *nrhs, real *a, integer *lda, real *b, integer *ldb, integer *
	info);

/* Subroutine */ int strttf_(char *transr, char *uplo, integer *n, real *a, 
	integer *lda, real *arf, integer *info);

/* Subroutine */ int strttp_(char *uplo, integer *n, real *a, integer *lda, 
	real *ap, integer *info);

/* Subroutine */ int stzrqf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, integer *info);

/* Subroutine */ int stzrzf_(integer *m, integer *n, real *a, integer *lda, 
	real *tau, real *work, integer *lwork, integer *info);

/* Subroutine */ int xerbla_(char *srname, integer *info);

/* Subroutine */ int xerbla_array__(char *srname_array__, integer *
	srname_len__, integer *info, ftnlen srname_array_len);


/* Subroutine */ int dlamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1);

doublereal dsecnd_();

/* Subroutine */ int ilaver_(integer *vers_major__, integer *vers_minor__, 
	integer *vers_patch__);

doublereal second_();

/* Subroutine */ int slamc1_(integer *beta, integer *t, logical *rnd, logical 
	*ieee1);

/* Subroutine */ int slamc2_(integer *beta, integer *t, logical *rnd, real *
		    eps, integer *emin, real *rmin, integer *emax, real *rmax);

doublereal slamc3_(real *a, real *b);

/* Subroutine */ int slamc4_(integer *emin, real *start, integer *base);

/* Subroutine */ int slamc5_(integer *beta, integer *p, integer *emin,
		    logical *ieee, integer *emax, real *rmax);


/* Subroutine */ int dlamc1_(integer *beta, integer *t, logical *rnd, logical
		    *ieee1);

/* Subroutine */ int dlamc2_(integer *beta, integer *t, logical *rnd,
		    doublereal *eps, integer *emin, doublereal *rmin, integer *emax,
			    doublereal *rmax);

doublereal dlamc3_(doublereal *a, doublereal *b);

/* Subroutine */ int dlamc4_(integer *emin, doublereal *start, integer *base);

/* Subroutine */ int dlamc5_(integer *beta, integer *p, integer *emin,
		    logical *ieee, integer *emax, doublereal *rmax);

#ifdef __cplusplus
}
#endif

#endif /* __CLAPACK_H */
