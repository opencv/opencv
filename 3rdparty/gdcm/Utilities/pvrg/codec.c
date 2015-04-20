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
codec.c

This file contains much of the transform coding routines to manipulate
the Huffman stream.

************************************************************
*/

/*LABEL codec.c */

/* Include files. */
#include "globals.h"
#include "csize.h"

#include <stdlib.h> /* abs */

/* Definitions for renaming functions. */

#define fputv meputv
#define fgetv megetv

/* Exportable functions. */

/*PUBLIC*/

extern void FrequencyAC();
extern void EncodeAC();
extern void DecodeAC();
extern int DecodeDC();
extern void FrequencyDC();
extern void EncodeDC();
extern void ResetCodec();
extern void ClearFrameFrequency();
extern void AddFrequency();
extern void InstallFrequency();
extern void InstallPrediction();
extern void PrintACEhuff();
extern void PrintDCEhuff();
extern int SizeACEhuff();
extern int SizeDCEhuff();

extern int LosslessDecodeDC();
extern void LosslessFrequencyDC();
extern void LosslessEncodeDC();

/*PRIVATE*/

/* Imported Variables. */

extern int bit_set_mask[]; /* Used for testing sign extension. */
extern int Loud;           /* General debug level. */
extern FRAME *CFrame;      /* Frame parameter. */
extern IMAGE *CImage;      /* Image parameter. */
extern SCAN *CScan;        /* Scan parameter. */

/* Local Variables */

static int *LastDC=NULL;         /* Last DC value for DPCM. */
static int *ACFrequency=NULL;    /* AC Frequency table to accum. statistics.*/
static int *DCFrequency=NULL;    /* DC Frequency table to accum. statistics.*/
static int extend_mask[]={       /* Used for sign extensions. */
0xFFFFFFFE,
0xFFFFFFFC,
0xFFFFFFF8,
0xFFFFFFF0,
0xFFFFFFE0,
0xFFFFFFC0,
0xFFFFFF80,
0xFFFFFF00,
0xFFFFFE00,
0xFFFFFC00,
0xFFFFF800,
0xFFFFF000,
0xFFFFE000,
0xFFFFC000,
0xFFFF8000,
0xFFFF0000,
0xFFFE0000,
0xFFFC0000,
0xFFF80000,
0xFFF00000
};

/*START*/

/*BFUNC

FrequencyAC() is used to accumulate the Huffman codes for the input
matrix. The Huffman codes are not actually stored but rather the count
of each code is stored so that construction of a Custom Table is
possible.

EFUNC*/

void FrequencyAC(matrix)
     int *matrix;
{
  BEGIN("FrequencyAC")
  int i,k,r,ssss,cofac;

  for(k=r=0;++k < BLOCKSIZE;)  /* Like EncodeAC below except don't write out */
    {
      cofac = abs(matrix[k]);           /* Find absolute size */
      if (cofac < 256)
  {
    ssss = csize[cofac];
  }
      else
  {
    cofac = cofac >> 8;
    ssss = csize[cofac] + 8;
  }
      if (matrix[k] == 0)               /* Check for zeroes */
  {
    if (k == BLOCKSIZE-1)         /* If end of block, then process */
      {
#ifdef CODEC_DEBUG
        printf("AC FEncoding EOB %d\n",0);
#endif
        ACFrequency[0]++;         /* Increment EOB frequency */
        break;
      }
    r++;
  }
      else
  {
    while(r > 15)                 /* Convert, r, ssss, into RLE */
      {
#ifdef CODEC_DEBUG
        printf("AC FEncoding OVFL %d\n",240);
#endif
        ACFrequency[240]++;       /* Increment ZRL extender freq */
        r -= 16;
      }
    i = 16*r + ssss;              /* Make code */
    r = 0;
#ifdef CODEC_DEBUG
    printf("AC FEncoding nnnnssss %d\n",i);
#endif
    ACFrequency[i]++;             /* Increment frequency of such code. */
  }
    }
}

/*BFUNC

EncodeAC() takes the matrix and encodes it by passing the values
of the codes found to the Huffman package.

EFUNC*/

void EncodeAC(matrix)
     int *matrix;
{
  BEGIN("EncodeAC")
  int i,k,r,ssss,cofac;

  for(k=r=0;++k<BLOCKSIZE;)
    {
      cofac = abs(matrix[k]);             /* Find absolute size */
      if (cofac < 256)
  {
    ssss = csize[cofac];
  }
      else
  {
    cofac = cofac >> 8;
    ssss = csize[cofac] + 8;
  }
      if (matrix[k] == 0)                /* Check for zeroes */
  {
    if (k == BLOCKSIZE-1)
      {
#ifdef CODEC_DEBUG
        printf("AC Encoding EOB %d\n",0);
#endif
        EncodeHuffman(0);
        break;
      }
    r++;                           /* Increment run-length of zeroes */
  }
      else
  {
    while(r > 15)                 /* If run-length > 15, time for  */
      {                           /* Run-length extension */
#ifdef CODEC_DEBUG
        printf("AC Encoding OVFL %d\n",240);
#endif
        EncodeHuffman(240);
        r -= 16;
      }
    i = 16*r + ssss;              /* Now we can find code byte */
#ifdef CODEC_DEBUG
    printf("AC Encoding nnnnssss %d\n",i);
#endif
    r = 0;
    EncodeHuffman(i);             /* Encode RLE code */
    if (matrix[k]< 0)             /* Follow by significant bits */
      {
        fputv(ssss,matrix[k]-1);
      }
    else
      {
        fputv(ssss,matrix[k]);
      }

  }
    }
}

/*BFUNC

DecodeAC() is used to decode the AC coefficients from the stream in
the stream package. The information generated is stored in the matrix
passed to it.

EFUNC*/

void DecodeAC(matrix)
     int *matrix;
{
  BEGIN("DecodeAC")
  int k,r,s,n;
  register int *mptr;

  for(mptr=matrix+1;mptr<matrix+BLOCKSIZE;mptr++)  /* Set all values to zero */
    {
      *mptr=0;
    }

  for(k=1;k<BLOCKSIZE;)  /* JPEG Mistake */
    {
      r = DecodeHuffman();                         /* Decode Huffman */
#ifdef CODEC_DEBUG
      printf("Raw AC Input: %d\n",r);
#endif
      s = r & 0xf;                                 /* Find significant bits */
      n = (r >> 4) & 0xf;                          /* n = run-length */
      if (s)
  {
    if ((k += n)>=BLOCKSIZE) break;          /* JPEG Mistake */
    matrix[k] = fgetv(s);                    /* Get s bits */

    s--; /* Align s */
    if ((matrix[k] & bit_set_mask[s]) == 0)  /* Also (1 << s) */
      {
        matrix[k] |= extend_mask[s];         /* Also  (-1 << s) + 1 */
        matrix[k]++;                         /* Increment 2's c */
      }
    k++;                                     /* Goto next element */
  }
      else if (n == 15)                      /* Zero run length code extnd */
  k += 16;
      else
  {
    break;
  }
    }
}

/*BFUNC

DecodeDC() is used to decode a DC value from the input stream.
It returns the actual number found.

EFUNC*/

int DecodeDC()
{
  BEGIN("DecodeDC")
  int s,diff;

  s = DecodeHuffman();
#ifdef CODEC_DEBUG
  printf("DC Decode sig. %d\n",s);
#endif

  if (s)
    {
      diff = fgetv(s);
      s--;                                  /* 2's Bit Align */
#ifdef CODEC_DEBUG
      printf("Raw DC Decode %d\n",diff);
#endif
      if ((diff & bit_set_mask[s]) == 0)
  {
    diff |= extend_mask[s];
    diff++;
  }
      diff += *LastDC;                      /* Change the last DC */
      *LastDC = diff;
    }
  return(*LastDC);
}

/*BFUNC

FrequencyDC() is used to accumulate statistics on what DC codes occur
most frequently.

EFUNC*/

void FrequencyDC(coef)
     int coef;
{
  BEGIN("FrequencyDC")
  int s,diff,cofac;

  diff = coef - *LastDC;         /* Do DPCM */
  *LastDC = coef;
  cofac = abs(diff);
  if (cofac < 256)               /* Find "code" */
    {
      s = csize[cofac];
    }
  else
    {
      cofac = cofac >> 8;
      s = csize[cofac] + 8;
    }
#ifdef CODEC_DEBUG
  printf("DC FEncoding Difference %d Size %d\n",diff,s);
#endif
  DCFrequency[s]++;              /* Increment frequency of such code */
}

/*BFUNC

EncodeDC() encodes the input coefficient to the stream using the
currently installed DC Huffman table.

EFUNC*/

void EncodeDC(coef)
     int coef;
{
  BEGIN("EncodeDC")
  int s,diff,cofac;

  diff = coef - *LastDC;
  *LastDC = coef;                /* Do DPCM */
  cofac = abs(diff);
  if (cofac < 256)
    {
      s = csize[cofac];          /* Find true size */
    }
  else
    {
      cofac = cofac >> 8;
      s = csize[cofac] + 8;
    }
#ifdef CODEC_DEBUG
  printf("DC Encoding Difference %d Size %d\n",diff,s);
#endif
  EncodeHuffman(s);              /* Encode size */
  if (diff < 0)                  /* Encode difference */
    {
      diff--;
    }
  fputv(s,diff);
}

/*BFUNC

ResetCodec() is used to reset all the DC prediction values. This
function is primarily used for initialization and resynchronization.

EFUNC*/

void ResetCodec()
{
  BEGIN("ResetCodec")
  int i;

  for(i=0;i<CScan->NumberComponents;i++)
    {
      *CScan->LastDC[i] = 0;                /* Sets all DC predictions to 0 */
    }
}

/*BFUNC

ClearFrameFrequency() clears all current statistics.

EFUNC*/

void ClearFrameFrequency()
{
  int i;
  int *iptr;

  for(i=0;i<CScan->NumberComponents;i++)
    {
      *CScan->LastDC[i] = 0;
      for(iptr=CScan->ACFrequency[i];
    iptr<CScan->ACFrequency[i]+257;iptr++)
  {
    *iptr = 0;
  }
      for(iptr=CScan->DCFrequency[i];
    iptr<CScan->DCFrequency[i]+257;iptr++)
  {
    *iptr = 0;
  }
    }
}

/*BFUNC

AddFrequency() is used to combine the first set of frequencies denoted
by the first pointer to the second set of frequencies denoted by the
second pointer.

EFUNC*/

void AddFrequency(ptr1,ptr2)
     int *ptr1;
     int *ptr2;
{
  BEGIN("AddFrequency")
  int i;

  for(i=0;i<256;i++)
    {
      *(ptr1) = *(ptr1) + *(ptr2);
      ptr1++;
      ptr2++;
    }
  *(ptr1) = MAX(*(ptr1),*(ptr2));
}

/*BFUNC

InstallFrequency() is used to install a particular frequency set of
arrays (denoted by the [index] scan component from the Scan
parameters).

EFUNC*/

void InstallFrequency(index)
     int index;
{
  BEGIN("InstallFrequency")
  ACFrequency = CScan->ACFrequency[index];  /* Set the right pointers */
  DCFrequency = CScan->DCFrequency[index];
  LastDC = CScan->LastDC[index];
}


/*BFUNC

InstallPrediction() is used to install a particular DC prediction for
use in frequency counting, encoding and decoding.

EFUNC*/

void InstallPrediction(index)
     int index;
{
  BEGIN("InstallPrediction")

  LastDC = CScan->LastDC[index];   /* Set the right pointer */
}

/*BFUNC

PrintACEhuff() prints out the [index] AC Huffman encoding structure in
the Image structure.

EFUNC*/

void PrintACEhuff(index)
     int index;
{
  BEGIN("PrintACEhuff")
  int place;
  EHUFF *eh;
  int *freq;
  int i,j;

  freq = CScan->ACFrequency[index];
  eh = CImage->ACEhuff[index];
  printf("Code:[Frequency:Size]:TotalBits\n");
  for(place=0,i=0;i<8;i++)
    {
      for(j=0;j<8;j++)
  {
    printf("%2x:[%d:%d]:%d ",
     place,freq[place],eh->ehufsi[place],
     freq[place]*eh->ehufsi[place]);
    place++;
  }
      printf("\n");
    }
}

/*BFUNC

SizeACEhuff() returns the size in bits necessary to code the
particular frequency spectrum by the indexed ehuff.

EFUNC*/

int SizeACEhuff(index)
     int index;
{
  BEGIN("SizeACEhuff")
  int place,sumbits;
  EHUFF *eh;
  int *freq;

  freq = CScan->ACFrequency[index];
  eh = CImage->ACEhuff[index];
  for(sumbits=0,place=0;place<256;place++)  /* For all codes, */
    {                                       /* return freq * codelength */
      sumbits += freq[place]*(eh->ehufsi[place] + (place & 0x0f));
    }
  return(sumbits);
}

/*BFUNC

PrintDCEhuff() prints out the DC encoding Huffman structure in the
CImage structure according to the position specified by [index].

EFUNC*/

void PrintDCEhuff(index)
     int index;
{
  BEGIN("PrintDCEhuff")
  int place;
  EHUFF *eh;
  int *freq;
  int i,j;

  freq = CScan->DCFrequency[index];
  eh = CImage->DCEhuff[index];
  printf("Code:[Frequency:Size]:TotalBits\n");
  for(place=0,i=0;i<8;i++)
    {
      for(j=0;j<8;j++)
  {
    printf("%2x:[%d:%d]:%d ",
     place,freq[place],eh->ehufsi[place],
     freq[place]*eh->ehufsi[place]);
    place++;
  }
      printf("\n");
    }
}


/*BFUNC

SizeDCEhuff() returns the bit size of the frequency and codes held by
the indexed dc codebook and frequency.

EFUNC*/

int SizeDCEhuff(index)
     int index;
{
  BEGIN("SizeDCEhuff")
  int place,sumbits;
  EHUFF *eh;
  int *freq;

  freq = CScan->DCFrequency[index];
  eh = CImage->DCEhuff[index];
  for(sumbits=0,place=0;place<256;place++) /* For all codes */
    {                                      /* Return freq * codelength */
      sumbits += freq[place]*(eh->ehufsi[place] + place);
    }
  return(sumbits);
}


/*BFUNC

LosslessFrequencyDC() is used to accumulate statistics on what DC codes occur
most frequently.

EFUNC*/

void LosslessFrequencyDC(coef)
     int coef;
{
  BEGIN("FrequencyDC")
  int s,cofac;

  cofac = coef&0xffff;               /* Take modulo */
  if (cofac & 0x8000)                /* if signed, then get absoulte val*/
    cofac = 0x10000-cofac;

  for(s=0;cofac>=256;s+=8,cofac>>=8);               /* Find "code" */
  s += csize[cofac];

#ifdef CODEC_DEBUG
  printf("DC FEncoding Difference %d Size %d\n",diff,s);
#endif
  DCFrequency[s]++;              /* Increment frequency of such code */
}

/*BFUNC

LosslessEncodeDC() encodes the input coefficient to the stream using
the currently installed DC Huffman table.  The only exception is the
SSSS value of 16.

EFUNC*/

void LosslessEncodeDC(coef)
     int coef;
{
  BEGIN("EncodeDC")
  int s,cofac;

  cofac = coef&0xffff;               /* Take modulo */
  if (cofac & 0x8000)                /* if signed, then get absoulte val*/
    cofac = 0x10000-cofac;

  for(s=0;cofac>=256;s+=8,cofac>>=8);               /* Find "code" */
  s += csize[cofac];

#ifdef CODEC_DEBUG
  printf("DC Encoding Difference %d Size %d\n",coeff,s);
#endif
  EncodeHuffman(s);              /* Encode size */
  if (coef &0x8000)                  /* Encode difference */
    coef--;
  if (s!=16) fputv(s,coef);
}

/*BFUNC

LosslessDecodeDC() is used to decode a DC value from the input stream.
It returns the actual number found.

EFUNC*/

int LosslessDecodeDC()
{
  BEGIN("DecodeDC")
  int s,coef;

  s = DecodeHuffman();
#ifdef CODEC_DEBUG
  printf("DC Decode sig. %d\n",s);
#endif

  /* FIXME begin bug http://groups.google.com/group/comp.protocols.dicom/msg/6d90002f734a12eb?dmode=source */
  if (s==16)  return(32768);
  /* end bug */
  else if (s)
    {
      coef = fgetv(s);
      s--;                                /* 2's Bit Align */
#ifdef CODEC_DEBUG
      printf("Raw DC Decode %d\n",coef);
#endif
      if ((coef & bit_set_mask[s]) == 0)
  {
    coef |= extend_mask[s];
    coef++;
  }
      return(coef);
    }
  else return(0);
}

/*END*/
