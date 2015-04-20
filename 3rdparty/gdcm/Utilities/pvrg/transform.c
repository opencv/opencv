/*************************************************************
Copyright (C) 1990, 1991, 1993 Andy C. Hung, all rights reserved.
PUBLIC DOMAIN LICENSE: Stanford University Portable Video Research
Group. If you use this software, you agree to the following: This
program package is purely experimental, and is licensed "as is".
Permission is granted to use, modify, and distribute this program
without charge for any purpose, provided this license/ disclaimer
notice appears in the copies.  No warranty or maintenance is given,
either expressed or implied.  In no event shall the author(s) be
liable to you or a third party for any special, incidental,
consequential, or other damages, arising out of the use or inability
to use the program for any purpose (or the loss of data), even if we
have been advised of such possibilities.  Any public reference or
advertisement of this source code should refer to it as the Portable
Video Research Group (PVRG) code, and not by any author(s) (or
Stanford University) name.
*************************************************************/

/*
************************************************************
transform.c

This file contains the reference DCT, the zig-zag and quantization
algorithms.

************************************************************
*/

/*LABEL transform.c */

/* Include files */

#include "globals.h"
#include "dct.h"
#include <math.h>
#include <stdlib.h> /* exit */

/*PUBLIC*/

extern void ReferenceDct();
extern void ReferenceIDct();
extern void TransposeMatrix();
extern void Quantize();
extern void IQuantize();
extern void PreshiftDctMatrix();
extern void PostshiftIDctMatrix();
extern void BoundDctMatrix();
extern void BoundIDctMatrix();
extern void ZigzagMatrix();
extern void IZigzagMatrix();
extern int *ScaleMatrix();
extern void PrintMatrix();
extern void ClearMatrix();

static void DoubleReferenceDct1D();
static void DoubleReferenceIDct1D();
static void DoubleTransposeMatrix();

/*PRIVATE*/

/* The transposition indices */

int transpose_index[] =          /* Is a transpose map for matrix transp. */
{0,  8, 16, 24, 32, 40, 48, 56,
 1,  9, 17, 25, 33, 41, 49, 57,
 2, 10, 18, 26, 34, 42, 50, 58,
 3, 11, 19, 27, 35, 43, 51, 59,
 4, 12, 20, 28, 36, 44, 52, 60,
 5, 13, 21, 29, 37, 45, 53, 61,
 6, 14, 22, 30, 38, 46, 54, 62,
 7, 15, 23, 31, 39, 47, 55, 63};

int zigzag_index[] =              /* Is zig-zag map for matrix -> scan array */
{0,  1,  5,  6, 14, 15, 27, 28,
 2,  4,  7, 13, 16, 26, 29, 42,
 3,  8, 12, 17, 25, 30, 41, 43,
 9, 11, 18, 24, 31, 40, 44, 53,
10, 19, 23, 32, 39, 45, 52, 54,
20, 22, 33, 38, 46, 51, 55, 60,
21, 34, 37, 47, 50, 56, 59, 61,
35, 36, 48, 49, 57, 58, 62, 63};

int izigzag_index[] =
{0,  1,  8, 16,  9,  2,  3, 10,
17, 24, 32, 25, 18, 11,  4,  5,
12, 19, 26, 33, 40, 48, 41, 34,
27, 20, 13,  6,  7, 14, 21, 28,
35, 42, 49, 56, 57, 50, 43, 36,
29, 22, 15, 23, 30, 37, 44, 51,
58, 59, 52, 45, 38, 31, 39, 46,
53, 60, 61, 54, 47, 55, 62, 63};

/*Some definitions */

#define MakeMatrix() (int *) calloc(BLOCKSIZE,sizeof(int))
#define FixedMultiply(s,x,y)  x = ((x * y) >> s);
#define DCT_OFFSET 128

/*START*/

/*BFUNC

ReferenceDct() does a reference DCT on the input (matrix) and output
(new matrix).

EFUNC*/

void ReferenceDct(matrix,newmatrix)
     int *matrix;
     int *newmatrix;
{
  BEGIN("ReferenceDct")
  int *mptr;
  double *sptr,*dptr;
  double sourcematrix[BLOCKSIZE],destmatrix[BLOCKSIZE];

  for(sptr=sourcematrix,mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {                             /* Convert to doubles */
      *(sptr++) = (double) *mptr;
    }
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)
    {                             /* Do DCT on rows */
      DoubleReferenceDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);  /* Transpose */
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)
    {                             /* Do DCT on columns */
      DoubleReferenceDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);  /* Transpose */
  for(sptr = sourcematrix,mptr=newmatrix;
      mptr<newmatrix+BLOCKSIZE;sptr++)
    {    /* NB: Inversion on counter */
      *(mptr++) = (int) (*sptr > 0 ? (*(sptr)+0.5):(*(sptr)-0.5));
    }
}

/*BFUNC

DoubleReferenceDCT1D() does a 8 point dct on an array of double
input and places the result in a double output.

EFUNC*/

static void DoubleReferenceDct1D(ivect,ovect)
     double *ivect;
     double *ovect;
{
  BEGIN("DoubleReferenceDct1D")
  double *mptr,*iptr,*optr;

  for(mptr=DctMatrix,optr=ovect;optr<ovect+BLOCKWIDTH;optr++)
    {                           /* 1d dct is just matrix multiply */
      for(*optr=0,iptr=ivect;iptr<ivect+BLOCKWIDTH;iptr++)
  {
    *optr += *iptr*(*(mptr++));
  }
    }
}

/*BFUNC

ReferenceIDct() is used to perform a reference 8x8 inverse dct.  It is
a balanced IDCT. It takes the input (matrix) and puts it into the
output (newmatrix).

EFUNC*/

void ReferenceIDct(matrix,newmatrix)
     int *matrix;
     int *newmatrix;
{
  BEGIN("ReferenceIDct")
  int *mptr;
  double *sptr,*dptr;
  double sourcematrix[BLOCKSIZE],destmatrix[BLOCKSIZE];

  for(sptr = sourcematrix,mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      *(sptr++) = (double) *mptr;
    }
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)
    {
      DoubleReferenceIDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);
  for(dptr = destmatrix,sptr=sourcematrix;
      sptr<sourcematrix+BLOCKSIZE;sptr+=BLOCKWIDTH)
    {
      DoubleReferenceIDct1D(sptr,dptr);
      dptr+=BLOCKWIDTH;
    }
  DoubleTransposeMatrix(destmatrix,sourcematrix);
  for(sptr = sourcematrix,mptr=newmatrix;mptr<newmatrix+BLOCKSIZE;sptr++)
    {    /* NB: Inversion on counter */
      *(mptr++) = (int) (*sptr > 0 ? (*(sptr)+0.5):(*(sptr)-0.5));
    }
}

/*BFUNC

DoubleReferenceIDct1D() does an 8 point inverse dct on ivect and
puts the output in ovect.

EFUNC*/

static void DoubleReferenceIDct1D(ivect,ovect)
     double *ivect;
     double *ovect;
{
  BEGIN("DoubleReferenceIDct1D")
  double *mptr,*iptr,*optr;

  for(mptr = IDctMatrix,optr=ovect;optr<ovect+BLOCKWIDTH;optr++)
    {
      for(*optr=0,iptr=ivect;iptr<ivect+BLOCKWIDTH;iptr++)
  {
    *optr += *iptr*(*(mptr++));
  }
    }
}

/*BFUNC

TransposeMatrix transposes an input matrix and puts the output in
newmatrix.

EFUNC*/

void TransposeMatrix(matrix,newmatrix)
     int *matrix;
     int *newmatrix;
{
  BEGIN("TransposeMatrix")
  int *tptr;

  for(tptr=transpose_index;tptr<transpose_index+BLOCKSIZE;tptr++)
    {
      *(newmatrix++) = matrix[*tptr];
    }
}

/*BFUNC

DoubleTransposeMatrix transposes a double input matrix and puts the
double output in newmatrix.

EFUNC*/

static void DoubleTransposeMatrix(matrix,newmatrix)
     double *matrix;
     double *newmatrix;
{
  BEGIN("DoubleTransposeMatrix")
  int *tptr;

  for(tptr=transpose_index;tptr<transpose_index+BLOCKSIZE;tptr++)
    {
      *(newmatrix++) = matrix[*tptr];
    }
}

/*BFUNC

Quantize() quantizes an input matrix and puts the output in qmatrix.

EFUNC*/

void Quantize(matrix,qmatrix)
     int *matrix;
     int *qmatrix;
{
  BEGIN("Quantize")
  int *mptr;

  if (!qmatrix)
    {
      WHEREAMI();
      printf("No quantization matrix specified!\n");
      exit(ERROR_BOUNDS);
    }
  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      if (*mptr > 0)          /* Rounding is different for +/- coeffs */
  {
    *mptr = (*mptr + *qmatrix/2)/ (*qmatrix);
    qmatrix++;
  }
      else
  {
    *mptr = (*mptr - *qmatrix/2)/ (*qmatrix);
    qmatrix++;
  }
    }
}

/*BFUNC

IQuantize() takes an input matrix and does an inverse quantization
and puts the output int qmatrix.

EFUNC*/

void IQuantize(matrix,qmatrix)
     int *matrix;
     int *qmatrix;
{
  BEGIN("IQuantize")
  int *mptr;

  if (!qmatrix)
    {
      WHEREAMI();
      printf("No quantization matrix specified!\n");
      exit(ERROR_BOUNDS);
    }
  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      *mptr = *mptr*(*qmatrix);
      qmatrix++;
    }
}

/*BFUNC

PreshiftDctMatrix() subtracts 128 (2048) from all 64 elements of an 8x8
matrix.  This results in a balanced DCT without any DC bias.

EFUNC*/


void PreshiftDctMatrix(matrix,shift)
     int *matrix;
     int shift;
{
  BEGIN("PreshiftDctMatrix")
  int *mptr;

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++) {*mptr -= shift;}
}

/*BFUNC

PostshiftIDctMatrix() adds 128 (2048) to all 64 elements of an 8x8 matrix.
This results in strictly positive values for all pixel coefficients.

EFUNC*/

void PostshiftIDctMatrix(matrix,shift)
     int *matrix;
     int shift;
{
  BEGIN("PostshiftIDctMatrix")
  int *mptr;

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++) {*mptr += shift;}
}

/*BFUNC

BoundDctMatrix() clips the Dct matrix such that it is no larger than
a 10 (1023) bit word or 14 bit word (4095).

EFUNC*/

void BoundDctMatrix(matrix,Bound)
     int *matrix;
     int Bound;
{
  BEGIN("BoundDctMatrix")
  int *mptr;

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      if (*mptr+Bound < 0)
  *mptr = -Bound;
      else if (*mptr-Bound > 0)
  *mptr = Bound;
    }
}

/*BFUNC

BoundIDctMatrix bounds the inverse dct matrix so that no pixel has a
value greater than 255 (4095) or less than 0.

EFUNC*/

void BoundIDctMatrix(matrix,Bound)
     int *matrix;
     int Bound;
{
  BEGIN("BoundIDctMatrix")
  int *mptr;

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++)
    {
      if (*mptr < 0) {*mptr = 0;}
      else if (*mptr > Bound) {*mptr = Bound;}
    }
}

/*BFUNC

IZigzagMatrix() performs an inverse zig-zag translation on the
input imatrix and places the output in omatrix.

EFUNC*/

void IZigzagMatrix(imatrix,omatrix)
     int *imatrix;
     int *omatrix;
{
  BEGIN("IZigzagMatrix")
  int *tptr;

  for(tptr=zigzag_index;tptr<zigzag_index+BLOCKSIZE;tptr++)
    {
      *(omatrix++) = imatrix[*tptr];
    }
}


/*BFUNC

ZigzagMatrix() performs a zig-zag translation on the input imatrix
and puts the output in omatrix.

EFUNC*/

void ZigzagMatrix(imatrix,omatrix)
     int *imatrix;
     int *omatrix;
{
  BEGIN("ZigzagMatrix")
  int *tptr;

  for(tptr=zigzag_index;tptr<zigzag_index+BLOCKSIZE;tptr++)
    {
      omatrix[*tptr] = *(imatrix++);
    }
}


/*BFUNC

ScaleMatrix() does a matrix scale appropriate to the old Q-factor.  It
returns the matrix created.

EFUNC*/

int *ScaleMatrix(Numerator,Denominator,LongFlag,Matrix)
     int Numerator;
     int Denominator;
     int LongFlag;
     int *Matrix;
{
  BEGIN("ScaleMatrix")
  int *Temp,*tptr;
  int Limit;

  Limit=((LongFlag)?65535:255);
  if (!(Temp = MakeMatrix()))
    {
      WHEREAMI();
      printf("Cannot allocate space for new matrix.\n");
      exit(ERROR_MEMORY);
    }
  for(tptr=Temp;tptr<Temp+BLOCKSIZE;tptr++)
    {
      *tptr = (*(Matrix++) * Numerator)/Denominator;
      if (*tptr > Limit)
  *tptr = Limit;
      else if (*tptr < 1)
  *tptr = 1;
    }
  return(Temp);
}

/*BFUNC

PrintMatrix() prints an 8x8 matrix in row/column form.

EFUNC*/

void PrintMatrix(matrix)
     int *matrix;
{
  BEGIN("PrintMatrix")
  int i,j;

  if (matrix)
    {
      for(i=0;i<BLOCKHEIGHT;i++)
  {
    for(j=0;j<BLOCKWIDTH;j++) {printf("%6d ",*(matrix++));}
    printf("\n");
  }
    }
  else {printf("Null\n");}
}


/*BFUNC

ClearMatrix() sets all the elements of a matrix to be zero.

EFUNC*/

void ClearMatrix(matrix)
     int *matrix;
{
  BEGIN("ClearMatrix")
  int *mptr;

  for(mptr=matrix;mptr<matrix+BLOCKSIZE;mptr++) {*mptr = 0;}
}

/*END*/
