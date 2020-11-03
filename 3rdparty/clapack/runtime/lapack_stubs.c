#include "f2c.h"

static const int CLAPACK_NOT_IMPLEMENTED = -1024;

int sgesdd_(char *jobz, int *m, int *n, float *a, int *lda,
	float *s, float *u, int *ldu, float *vt, int *ldvt, float *work,
	int *lwork, int *iwork, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int dgels_(char *trans, int *m, int *n, int *nrhs, double *a,
	int *lda, double *b, int *ldb, double *work, int *lwork, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int dgesv_(int *n, int *nrhs, double *a, int *lda, int *ipiv,
	double *b, int *ldb, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int dgetrf_(int *m, int *n, double *a, int *lda, int *ipiv,
	int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int dposv_(char *uplo, int *n, int *nrhs, double *a, int *
	lda, double *b, int *ldb, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int dpotrf_(char *uplo, int *n, double *a, int *lda, int *
	info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int sgels_(char *trans, int *m, int *n, int *nrhs, float *a,
	int *lda, float *b, int *ldb, float *work, int *lwork, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int sgeev_(char *jobvl, char *jobvr, int *n, float *a, int *
	lda, float *wr, float *wi, float *vl, int *ldvl, float *vr, int *
	ldvr, float *work, int *lwork, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int sgeqrf_(int *m, int *n, float *a, int *lda, float *tau,
	float *work, int *lwork, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int sgesv_(int *n, int *nrhs, float *a, int *lda, int *ipiv,
	float *b, int *ldb, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}        

int sgetrf_(int *m, int *n, float *a, int *lda, int *ipiv,
	int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}

int sposv_(char *uplo, int *n, int *nrhs, float *a, int *
	lda, float *b, int *ldb, int *info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}        

int spotrf_(char *uplo, int *n, float *a, int *lda, int *
	info)
{
    *info = CLAPACK_NOT_IMPLEMENTED;
    return 0;
}
