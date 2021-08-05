// this is auto-generated header for Lapack subset
#ifndef __CLAPACK_H__
#define __CLAPACK_H__

#include "cblas.h"

#ifdef __cplusplus
extern "C" {
#endif

int cgemm_(char *transa, char *transb, int *m, int *n, int *
	k, lapack_complex *alpha, lapack_complex *a, int *lda, lapack_complex *b, int *ldb,
	lapack_complex *beta, lapack_complex *c__, int *ldc);

int daxpy_(int *n, double *da, double *dx, int *incx, double
	*dy, int *incy);

int dbdsdc_(char *uplo, char *compq, int *n, double *d__,
	double *e, double *u, int *ldu, double *vt, int *ldvt, double *q, int
	*iq, double *work, int *iwork, int *info);

int dbdsqr_(char *uplo, int *n, int *ncvt, int *nru, int *
	ncc, double *d__, double *e, double *vt, int *ldvt, double *u, int *
	ldu, double *c__, int *ldc, double *work, int *info);

int dcombssq_(double *v1, double *v2);

int dcopy_(int *n, double *dx, int *incx, double *dy, int *
	incy);

double ddot_(int *n, double *dx, int *incx, double *dy, int *incy);

int dgebak_(char *job, char *side, int *n, int *ilo, int *
	ihi, double *scale, int *m, double *v, int *ldv, int *info);

int dgebal_(char *job, int *n, double *a, int *lda, int *ilo,
	 int *ihi, double *scale, int *info);

int dgebd2_(int *m, int *n, double *a, int *lda, double *d__,
	 double *e, double *tauq, double *taup, double *work, int *info);

int dgebrd_(int *m, int *n, double *a, int *lda, double *d__,
	 double *e, double *tauq, double *taup, double *work, int *lwork, int
	*info);

int dgeev_(char *jobvl, char *jobvr, int *n, double *a, int *
	lda, double *wr, double *wi, double *vl, int *ldvl, double *vr, int *
	ldvr, double *work, int *lwork, int *info);

int dgehd2_(int *n, int *ilo, int *ihi, double *a, int *lda,
	double *tau, double *work, int *info);

int dgehrd_(int *n, int *ilo, int *ihi, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info);

int dgelq2_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *info);

int dgelqf_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *lwork, int *info);

int dgemm_(char *transa, char *transb, int *m, int *n, int *
	k, double *alpha, double *a, int *lda, double *b, int *ldb, double *
	beta, double *c__, int *ldc);

int dgemv_(char *trans, int *m, int *n, double *alpha,
	double *a, int *lda, double *x, int *incx, double *beta, double *y,
	int *incy);

int dgeqr2_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *info);

int dgeqrf_(int *m, int *n, double *a, int *lda, double *tau,
	 double *work, int *lwork, int *info);

int dger_(int *m, int *n, double *alpha, double *x, int *
	incx, double *y, int *incy, double *a, int *lda);

int dgesdd_(char *jobz, int *m, int *n, double *a, int *lda,
	double *s, double *u, int *ldu, double *vt, int *ldvt, double *work,
	int *lwork, int *iwork, int *info);

int dhseqr_(char *job, char *compz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, double *z__, int *
	ldz, double *work, int *lwork, int *info);

int disnan_(double *din);

// "small" is a macro defined in Windows headers: https://stackoverflow.com/a/27794577
#ifdef small
#undef small
#endif

int dlabad_(double *small, double *large);

int dlabrd_(int *m, int *n, int *nb, double *a, int *lda,
	double *d__, double *e, double *tauq, double *taup, double *x, int *
	ldx, double *y, int *ldy);

int dlacpy_(char *uplo, int *m, int *n, double *a, int *lda,
	double *b, int *ldb);

int dladiv1_(double *a, double *b, double *c__, double *d__,
	double *p, double *q);

double dladiv2_(double *a, double *b, double *c__, double *d__, double *r__,
	double *t);

int dladiv_(double *a, double *b, double *c__, double *d__,
	double *p, double *q);

int dlaed6_(int *kniter, int *orgati, double *rho, double *
	d__, double *z__, double *finit, double *tau, int *info);

int dlaexc_(int *wantq, int *n, double *t, int *ldt, double *
	q, int *ldq, int *j1, int *n1, int *n2, double *work, int *info);

int dlahqr_(int *wantt, int *wantz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, int *iloz, int *
	ihiz, double *z__, int *ldz, int *info);

int dlahr2_(int *n, int *k, int *nb, double *a, int *lda,
	double *tau, double *t, int *ldt, double *y, int *ldy);

int dlaisnan_(double *din1, double *din2);

int dlaln2_(int *ltrans, int *na, int *nw, double *smin,
	double *ca, double *a, int *lda, double *d1, double *d2, double *b,
	int *ldb, double *wr, double *wi, double *x, int *ldx, double *scale,
	double *xnorm, int *info);

int dlamrg_(int *n1, int *n2, double *a, int *dtrd1, int *
	dtrd2, int *index);

double dlange_(char *norm, int *m, int *n, double *a, int *lda, double *work);

double dlanst_(char *norm, int *n, double *d__, double *e);

int dlanv2_(double *a, double *b, double *c__, double *d__,
	double *rt1r, double *rt1i, double *rt2r, double *rt2i, double *cs,
	double *sn);

double dlapy2_(double *x, double *y);

int dlaqr0_(int *wantt, int *wantz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, int *iloz, int *
	ihiz, double *z__, int *ldz, double *work, int *lwork, int *info);

int dlaqr1_(int *n, double *h__, int *ldh, double *sr1,
	double *si1, double *sr2, double *si2, double *v);

int dlaqr2_(int *wantt, int *wantz, int *n, int *ktop, int *
	kbot, int *nw, double *h__, int *ldh, int *iloz, int *ihiz, double *
	z__, int *ldz, int *ns, int *nd, double *sr, double *si, double *v,
	int *ldv, int *nh, double *t, int *ldt, int *nv, double *wv, int *
	ldwv, double *work, int *lwork);

int dlaqr3_(int *wantt, int *wantz, int *n, int *ktop, int *
	kbot, int *nw, double *h__, int *ldh, int *iloz, int *ihiz, double *
	z__, int *ldz, int *ns, int *nd, double *sr, double *si, double *v,
	int *ldv, int *nh, double *t, int *ldt, int *nv, double *wv, int *
	ldwv, double *work, int *lwork);

int dlaqr4_(int *wantt, int *wantz, int *n, int *ilo, int *
	ihi, double *h__, int *ldh, double *wr, double *wi, int *iloz, int *
	ihiz, double *z__, int *ldz, double *work, int *lwork, int *info);

int dlaqr5_(int *wantt, int *wantz, int *kacc22, int *n, int
	*ktop, int *kbot, int *nshfts, double *sr, double *si, double *h__,
	int *ldh, int *iloz, int *ihiz, double *z__, int *ldz, double *v, int
	*ldv, double *u, int *ldu, int *nv, double *wv, int *ldwv, int *nh,
	double *wh, int *ldwh);

int dlarf_(char *side, int *m, int *n, double *v, int *incv,
	double *tau, double *c__, int *ldc, double *work);

int dlarfb_(char *side, char *trans, char *direct, char *
	storev, int *m, int *n, int *k, double *v, int *ldv, double *t, int *
	ldt, double *c__, int *ldc, double *work, int *ldwork);

int dlarfg_(int *n, double *alpha, double *x, int *incx,
	double *tau);

int dlarft_(char *direct, char *storev, int *n, int *k,
	double *v, int *ldv, double *tau, double *t, int *ldt);

int dlarfx_(char *side, int *m, int *n, double *v, double *
	tau, double *c__, int *ldc, double *work);

int dlartg_(double *f, double *g, double *cs, double *sn,
	double *r__);

int dlas2_(double *f, double *g, double *h__, double *ssmin,
	double *ssmax);

int dlascl_(char *type__, int *kl, int *ku, double *cfrom,
	double *cto, int *m, int *n, double *a, int *lda, int *info);

int dlasd0_(int *n, int *sqre, double *d__, double *e,
	double *u, int *ldu, double *vt, int *ldvt, int *smlsiz, int *iwork,
	double *work, int *info);

int dlasd1_(int *nl, int *nr, int *sqre, double *d__, double
	*alpha, double *beta, double *u, int *ldu, double *vt, int *ldvt, int
	*idxq, int *iwork, double *work, int *info);

int dlasd2_(int *nl, int *nr, int *sqre, int *k, double *d__,
	 double *z__, double *alpha, double *beta, double *u, int *ldu,
	double *vt, int *ldvt, double *dsigma, double *u2, int *ldu2, double *
	vt2, int *ldvt2, int *idxp, int *idx, int *idxc, int *idxq, int *
	coltyp, int *info);

int dlasd3_(int *nl, int *nr, int *sqre, int *k, double *d__,
	 double *q, int *ldq, double *dsigma, double *u, int *ldu, double *u2,
	 int *ldu2, double *vt, int *ldvt, double *vt2, int *ldvt2, int *idxc,
	 int *ctot, double *z__, int *info);

int dlasd4_(int *n, int *i__, double *d__, double *z__,
	double *delta, double *rho, double *sigma, double *work, int *info);

int dlasd5_(int *i__, double *d__, double *z__, double *
	delta, double *rho, double *dsigma, double *work);

int dlasd6_(int *icompq, int *nl, int *nr, int *sqre, double
	*d__, double *vf, double *vl, double *alpha, double *beta, int *idxq,
	int *perm, int *givptr, int *givcol, int *ldgcol, double *givnum, int
	*ldgnum, double *poles, double *difl, double *difr, double *z__, int *
	k, double *c__, double *s, double *work, int *iwork, int *info);

int dlasd7_(int *icompq, int *nl, int *nr, int *sqre, int *k,
	 double *d__, double *z__, double *zw, double *vf, double *vfw,
	double *vl, double *vlw, double *alpha, double *beta, double *dsigma,
	int *idx, int *idxp, int *idxq, int *perm, int *givptr, int *givcol,
	int *ldgcol, double *givnum, int *ldgnum, double *c__, double *s, int
	*info);

int dlasd8_(int *icompq, int *k, double *d__, double *z__,
	double *vf, double *vl, double *difl, double *difr, int *lddifr,
	double *dsigma, double *work, int *info);

int dlasda_(int *icompq, int *smlsiz, int *n, int *sqre,
	double *d__, double *e, double *u, int *ldu, double *vt, int *k,
	double *difl, double *difr, double *z__, double *poles, int *givptr,
	int *givcol, int *ldgcol, int *perm, double *givnum, double *c__,
	double *s, double *work, int *iwork, int *info);

int dlasdq_(char *uplo, int *sqre, int *n, int *ncvt, int *
	nru, int *ncc, double *d__, double *e, double *vt, int *ldvt, double *
	u, int *ldu, double *c__, int *ldc, double *work, int *info);

int dlasdt_(int *n, int *lvl, int *nd, int *inode, int *
	ndiml, int *ndimr, int *msub);

int dlaset_(char *uplo, int *m, int *n, double *alpha,
	double *beta, double *a, int *lda);

int dlasq1_(int *n, double *d__, double *e, double *work,
	int *info);

int dlasq2_(int *n, double *z__, int *info);

int dlasq3_(int *i0, int *n0, double *z__, int *pp, double *
	dmin__, double *sigma, double *desig, double *qmax, int *nfail, int *
	iter, int *ndiv, int *ieee, int *ttype, double *dmin1, double *dmin2,
	double *dn, double *dn1, double *dn2, double *g, double *tau);

int dlasq4_(int *i0, int *n0, double *z__, int *pp, int *
	n0in, double *dmin__, double *dmin1, double *dmin2, double *dn,
	double *dn1, double *dn2, double *tau, int *ttype, double *g);

int dlasq5_(int *i0, int *n0, double *z__, int *pp, double *
	tau, double *sigma, double *dmin__, double *dmin1, double *dmin2,
	double *dn, double *dnm1, double *dnm2, int *ieee, double *eps);

int dlasq6_(int *i0, int *n0, double *z__, int *pp, double *
	dmin__, double *dmin1, double *dmin2, double *dn, double *dnm1,
	double *dnm2);

int dlasr_(char *side, char *pivot, char *direct, int *m,
	int *n, double *c__, double *s, double *a, int *lda);

int dlasrt_(char *id, int *n, double *d__, int *info);

int dlassq_(int *n, double *x, int *incx, double *scale,
	double *sumsq);

int dlasv2_(double *f, double *g, double *h__, double *ssmin,
	 double *ssmax, double *snr, double *csr, double *snl, double *csl);

int dlasy2_(int *ltranl, int *ltranr, int *isgn, int *n1,
	int *n2, double *tl, int *ldtl, double *tr, int *ldtr, double *b, int
	*ldb, double *scale, double *x, int *ldx, double *xnorm, int *info);

double dnrm2_(int *n, double *x, int *incx);

int dorg2r_(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *info);

int dorgbr_(char *vect, int *m, int *n, int *k, double *a,
	int *lda, double *tau, double *work, int *lwork, int *info);

int dorghr_(int *n, int *ilo, int *ihi, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info);

int dorgl2_(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *info);

int dorglq_(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info);

int dorgqr_(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info);

int dorm2r_(char *side, char *trans, int *m, int *n, int *k,
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *info);

int dormbr_(char *vect, char *side, char *trans, int *m, int
	*n, int *k, double *a, int *lda, double *tau, double *c__, int *ldc,
	double *work, int *lwork, int *info);

int dormhr_(char *side, char *trans, int *m, int *n, int *
	ilo, int *ihi, double *a, int *lda, double *tau, double *c__, int *
	ldc, double *work, int *lwork, int *info);

int dorml2_(char *side, char *trans, int *m, int *n, int *k,
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *info);

int dormlq_(char *side, char *trans, int *m, int *n, int *k,
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *lwork, int *info);

int dormqr_(char *side, char *trans, int *m, int *n, int *k,
	double *a, int *lda, double *tau, double *c__, int *ldc, double *work,
	 int *lwork, int *info);

int drot_(int *n, double *dx, int *incx, double *dy, int *
	incy, double *c__, double *s);

int dscal_(int *n, double *da, double *dx, int *incx);

int dswap_(int *n, double *dx, int *incx, double *dy, int *
	incy);

int dtrevc3_(char *side, char *howmny, int *select, int *n,
	double *t, int *ldt, double *vl, int *ldvl, double *vr, int *ldvr,
	int *mm, int *m, double *work, int *lwork, int *info);

int dtrexc_(char *compq, int *n, double *t, int *ldt, double
	*q, int *ldq, int *ifst, int *ilst, double *work, int *info);

int dtrmm_(char *side, char *uplo, char *transa, char *diag,
	int *m, int *n, double *alpha, double *a, int *lda, double *b, int *
	ldb);

int dtrmv_(char *uplo, char *trans, char *diag, int *n,
	double *a, int *lda, double *x, int *incx);

int idamax_(int *n, double *dx, int *incx);

int ieeeck_(int *ispec, float *zero, float *one);

int iladlc_(int *m, int *n, double *a, int *lda);

int iladlr_(int *m, int *n, double *a, int *lda);

int ilaenv_(int *ispec, char *name__, char *opts, int *n1, int *n2, int *n3,
	int *n4);

int iparmq_(int *ispec, char *name__, char *opts, int *n, int *ilo, int *ihi,
	int *lwork);

int sgemm_(char *transa, char *transb, int *m, int *n, int *
	k, float *alpha, float *a, int *lda, float *b, int *ldb, float *beta,
	float *c__, int *ldc);

int zgemm_(char *transa, char *transb, int *m, int *n, int *
	k, lapack_doublecomplex *alpha, lapack_doublecomplex *a, int *lda, lapack_doublecomplex *b,
	 int *ldb, lapack_doublecomplex *beta, lapack_doublecomplex *c__, int *ldc);

#ifdef __cplusplus
}
#endif

#endif
