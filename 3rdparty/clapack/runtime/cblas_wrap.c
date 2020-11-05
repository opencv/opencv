#include "f2c.h"
#include <stdarg.h>

void cblas_cgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void  *A,
                 const int lda, const void  *B, const int ldb,
                 const void *beta, void  *C, const int ldc)
{
   char TA, TB;

   if( layout == CblasColMajor )
   {
      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_cgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 3, "cblas_cgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      cgemm_(&TA, &TB, (int*)&M, (int*)&N, (int*)&K, (complex*)alpha, (complex*)A, (int*)&lda,
             (complex*)B, (int*)&ldb, (complex*)beta, (complex*)C, (int*)&ldc);
   }
   else if (layout == CblasRowMajor)
   {
      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_cgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_cgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      cgemm_(&TA, &TB, (int*)&N, (int*)&M, (int*)&K, (complex*)alpha, (complex*)B, (int*)&ldb,
             (complex*)A, (int*)&lda, (complex*)beta, (complex*)C, (int*)&ldc);
   }
   else cblas_xerbla(layout, 1, "cblas_cgemm", "Illegal layout setting, %d\n", layout);
}

void cblas_dgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A,
                 const int lda, const double  *B, const int ldb,
                 const double beta, double  *C, const int ldc)
{
   char TA, TB;

   if( layout == CblasColMajor )
   {
      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_dgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 3, "cblas_dgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      dgemm_(&TA, &TB, (int*)&M, (int*)&N, (int*)&K, (double*)&alpha, (double*)A, (int*)&lda,
             (double*)B, (int*)&ldb, (double*)&beta, (double*)C, (int*)&ldc);
   }
   else if (layout == CblasRowMajor)
   {
      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_dgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_dgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      dgemm_(&TA, &TB, (int*)&N, (int*)&M, (int*)&K, (double*)&alpha, (double*)B, (int*)&ldb,
             (double*)A, (int*)&lda, (double*)&beta, (double*)C, (int*)&ldc);
   }
   else cblas_xerbla(layout, 1, "cblas_dgemm", "Illegal layout setting, %d\n", layout);
}


void cblas_sgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float  *A,
                 const int lda, const float  *B, const int ldb,
                 const float beta, float  *C, const int ldc)
{
   char TA, TB;

   if( layout == CblasColMajor )
   {
      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_sgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 3, "cblas_sgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      sgemm_(&TA, &TB, (int*)&M, (int*)&N, (int*)&K, (float*)&alpha, (float*)A, (int*)&lda,
             (float*)B, (int*)&ldb, (float*)&beta, (float*)C, (int*)&ldc);
   }
   else if (layout == CblasRowMajor)
   {
      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_sgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_sgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      sgemm_(&TA, &TB, (int*)&N, (int*)&M, (int*)&K, (float*)&alpha, (float*)B, (int*)&ldb,
             (float*)A, (int*)&lda, (float*)&beta, (float*)C, (int*)&ldc);
   }
   else cblas_xerbla(layout, 1, "cblas_sgemm", "Illegal layout setting, %d\n", layout);
}

void cblas_zgemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const void *alpha, const void  *A,
                 const int lda, const void  *B, const int ldb,
                 const void *beta, void  *C, const int ldc)
{
   char TA, TB;

   if( layout == CblasColMajor )
   {
      if(TransA == CblasTrans) TA='T';
      else if ( TransA == CblasConjTrans ) TA='C';
      else if ( TransA == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_zgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }

      if(TransB == CblasTrans) TB='T';
      else if ( TransB == CblasConjTrans ) TB='C';
      else if ( TransB == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 3, "cblas_zgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      zgemm_(&TA, &TB, (int*)&M, (int*)&N, (int*)&K, (doublecomplex*)alpha, (doublecomplex*)A, (int*)&lda,
             (doublecomplex*)B, (int*)&ldb, (doublecomplex*)beta, (doublecomplex*)C, (int*)&ldc);
   }
   else if (layout == CblasRowMajor)
   {
      if(TransA == CblasTrans) TB='T';
      else if ( TransA == CblasConjTrans ) TB='C';
      else if ( TransA == CblasNoTrans )   TB='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_zgemm", "Illegal TransA setting, %d\n", TransA);
         return;
      }
      if(TransB == CblasTrans) TA='T';
      else if ( TransB == CblasConjTrans ) TA='C';
      else if ( TransB == CblasNoTrans )   TA='N';
      else
      {
         cblas_xerbla(layout, 2, "cblas_zgemm", "Illegal TransB setting, %d\n", TransB);
         return;
      }

      zgemm_(&TA, &TB, (int*)&N, (int*)&M, (int*)&K, (doublecomplex*)alpha, (doublecomplex*)B, (int*)&ldb,
             (doublecomplex*)A, (int*)&lda, (doublecomplex*)beta, (doublecomplex*)C, (int*)&ldc);
   }
   else cblas_xerbla(layout, 1, "cblas_zgemm", "Illegal layout setting, %d\n", layout);
}

void cblas_xerbla(const CBLAS_LAYOUT layout, int info, const char *rout, const char *form, ...)
{
   extern int RowMajorStrg;
   char empty[1] = "";
   va_list argptr;

   va_start(argptr, form);

   if (layout == CblasRowMajor)
   {
      if (strstr(rout,"gemm") != 0)
      {
         if      (info == 5 ) info =  4;
         else if (info == 4 ) info =  5;
         else if (info == 11) info =  9;
         else if (info == 9 ) info = 11;
      }
      else if (strstr(rout,"symm") != 0 || strstr(rout,"hemm") != 0)
      {
         if      (info == 5 ) info =  4;
         else if (info == 4 ) info =  5;
      }
      else if (strstr(rout,"trmm") != 0 || strstr(rout,"trsm") != 0)
      {
         if      (info == 7 ) info =  6;
         else if (info == 6 ) info =  7;
      }
      else if (strstr(rout,"gemv") != 0)
      {
         if      (info == 4)  info = 3;
         else if (info == 3)  info = 4;
      }
      else if (strstr(rout,"gbmv") != 0)
      {
         if      (info == 4)  info = 3;
         else if (info == 3)  info = 4;
         else if (info == 6)  info = 5;
         else if (info == 5)  info = 6;
      }
      else if (strstr(rout,"ger") != 0)
      {
         if      (info == 3) info = 2;
         else if (info == 2) info = 3;
         else if (info == 8) info = 6;
         else if (info == 6) info = 8;
      }
      else if ( (strstr(rout,"her2") != 0 || strstr(rout,"hpr2") != 0)
                 && strstr(rout,"her2k") == 0 )
      {
         if      (info == 8) info = 6;
         else if (info == 6) info = 8;
      }
   }
   if (info)
      fprintf(stderr, "Parameter %d to routine %s was incorrect\n", info, rout);
   vfprintf(stderr, form, argptr);
   va_end(argptr);
   if (info && !info)
      xerbla_(empty, &info); /* Force link of our F77 error handler */
   exit(-1);
}
