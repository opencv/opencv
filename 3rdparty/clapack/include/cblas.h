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

int sgesdd_(char *jobz, int *m, int *n, float *a, int *lda,
	float *s, float *u, int *ldu, float *vt, int *ldvt, float *work,
	int *lwork, int *iwork, int *info);

#ifdef __cplusplus
}
#endif

#endif /* __CBLAS_H__ */
