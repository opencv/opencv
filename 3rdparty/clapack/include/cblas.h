#ifndef __CBLAS_H__
#define __CBLAS_H__

/* most of the stuff is in lapacke.h */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lapack_complex
{
    float r, i;
} lapack_complex;

typedef struct lapack_doublecomplex
{
    double r, i;
} lapack_doublecomplex;

typedef enum {CblasRowMajor=101, CblasColMajor=102} CBLAS_LAYOUT;
typedef enum {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113} CBLAS_TRANSPOSE;

void cblas_xerbla(const CBLAS_LAYOUT layout, int info,
                  const char *rout, const char *form, ...);

void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                 CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void cblas_dgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
              CBLAS_TRANSPOSE TransB, const int M, const int N,
              const int K, const double alpha, const double *A,
              const int lda, const double *B, const int ldb,
              const double beta, double *C, const int ldc);

void cblas_cgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
               CBLAS_TRANSPOSE TransB, const int M, const int N,
               const int K, const void *alpha, const void *A,
               const int lda, const void *B, const int ldb,
               const void *beta, void *C, const int ldc);

void cblas_zgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA,
                CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const void *alpha, const void *A,
                const int lda, const void *B, const int ldb,
                const void *beta, void *C, const int ldc);

int xerbla_(char *, int *);
int lsame_(char *, char *);
double slamch_(char* cmach);
double slamc3_(float *a, float *b);
double dlamch_(char* cmach);
double dlamc3_(double *a, double *b);

int dgels_(char *trans, int *m, int *n, int *nrhs, double *a,
	int *lda, double *b, int *ldb, double *work, int *lwork, int *info);

int dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv,
	double *b, int *ldb, int *info);

int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv,
	int *info);

int dposv_(char *uplo, int *n, int *nrhs, double *a, int *
	lda, double *b, int *ldb, int *info);

int dpotrf_(char *uplo, int *n, double *a, int *lda, int *
	info);

int sgels_(char *trans, int *m, int *n, int *nrhs, float *a,
	int *lda, float *b, int *ldb, float *work, int *lwork, int *info);

int sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *
	lda, float *wr, float *wi, float *vl, int *ldvl, float *vr, int *
	ldvr, float *work, int *lwork, int *info);

int sgeqrf_(int *m, int *n, float *a, int *lda, float *tau,
	float *work, int *lwork, int *info);

int sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv,
	float *b, int *ldb, int *info);

int sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv,
	int *info);

int sposv_(char *uplo, int *n, int *nrhs, float *a, int *
	lda, float *b, int *ldb, int *info);

int spotrf_(char *uplo, int *n, float *a, int *lda, int *
	info);

int sgesdd_(char *jobz, int *m, int *n, float *a, int *lda,
	float *s, float *u, int *ldu, float *vt, int *ldvt, float *work,
	int *lwork, int *iwork, int *info);

#ifdef __cplusplus
}
#endif

#endif /* __CBLAS_H__ */
