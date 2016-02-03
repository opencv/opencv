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
leedct.c

This is the Byeong Gi Lee algorithm from IEEE Trans. Accoustics,
Speech, and Signal Processing, Vol ASSP-32, No. 6, December 1984, pp.
1243 -1245.

************************************************************
*/

/*LABEL leedct.c */

/*PUBLIC*/

extern void LeeIDct();
extern void LeeDct();

/*PRIVATE*/

/* Standard Macros */

#define LS(r,s) ((r) << (s))
#define RS(r,s) ((r) >> (s)) /* Caution with rounding... */

#define MSCALE(expr)  RS((expr),9)

#define IDCTSCALE(x) (((x<0) ? (x-8) : (x+8))/16);
#define DCTSCALE(x) (((x<0) ? (x-8) : (x+8))/16);

/* Cos Table */

#define twoc1d4 724L
#define twoc1d8 946L
#define twoc3d8 392L
#define twoc1d16 1004L
#define twoc3d16 851L
#define twoc5d16 569L
#define twoc7d16 200L
#define sqrt2 724L

/* 1/Cos Table */

#define itwoc1d4 362L
#define itwoc1d8 277L
#define itwoc3d8 669L
#define itwoc1d16 261L
#define itwoc3d16 308L
#define itwoc5d16 461L
#define itwoc7d16 1312L
#define isqrt2 362L

#define x0 tx0
#define x1 tx1
#define x2 tx2
#define x3 tx3
#define x4 ex4
#define x5 ex5
#define x6 ex6
#define x7 ex7

#define r0 rx0
#define r1 rx1
#define r2 rx2
#define r3 rx3

#define s0 rx0
#define s1 rx1
#define s2 rx2
#define s3 rx3

#define f0 ex0
#define f1 ex1
#define f2 ex2
#define f3 ex3

#define g0 ex4
#define g1 ex5
#define g2 ex6
#define g3 ex7

#define b1 gx0
#define b2 gx0
#define b3 gx1

#define a1 gx2
#define a3 gx2
#define c1 gx2
#define c3 gx2

#define ihold gx1

/*START*/

/*BFUNC

LeeIDct is implemented according to the inverse dct flow diagram in
the paper.  It takes two input arrays that must be defined before the
call.

EFUNC*/

void LeeIDct(x,y)
     int *x;
     int *y;
{
  register int ex0,ex1,ex2,ex3,ex4,ex5,ex6,ex7;
  register int tx0,tx1,tx2,tx3;
  register int rx0,rx1,rx2,rx3;
  register int gx0,gx1,gx2;
  register int *iptr,*jptr;
  register int i;

  /* Do rows */

  for(jptr=y,iptr=x,i=0;i<8;i++)
    {
      x0 = MSCALE(isqrt2*LS(*(iptr++),2));
      x1 = LS(*(iptr++),2);
      x2 = LS(*(iptr++),2);
      x3 = LS(*(iptr++),2);
      x4 = LS(*(iptr++),2);
      x5 = LS(*(iptr++),2);
      x6 = LS(*(iptr++),2);
      x7 = LS(*(iptr++),2);

      a1 = MSCALE(itwoc1d4*x4);
      r0 = x0+a1;
      r1 = x0-a1;

      a3 = MSCALE(itwoc1d4*(x2+x6));
      r2 = MSCALE(itwoc1d8*(x2+a3));
      r3 = MSCALE(itwoc3d8*(x2-a3));

      f0 = r0+r2;
      f1 = r1+r3;
      f2 = r0-r2;
      f3 = r1-r3;

      b1 = x3+x5;
      c1 = MSCALE(itwoc1d4*b1);
      s0 = x1+c1;
      s1 = x1-c1;

      b2 = x1+x3;
      b3 = x5+x7;
      c3 = MSCALE(itwoc1d4*(b2+b3));
      s2 = MSCALE(itwoc1d8*(b2+c3));
      s3 = MSCALE(itwoc3d8*(b2-c3));

      g0 = MSCALE(itwoc1d16*(s0+s2));
      g1 = MSCALE(itwoc3d16*(s1+s3));
      g2 = MSCALE(itwoc7d16*(s0-s2));
      g3 = MSCALE(itwoc5d16*(s1-s3));

      *(jptr++) = f0+g0;
      *(jptr++) = f1+g1;
      *(jptr++) = f3+g3;
      *(jptr++) = f2+g2;

      *(jptr++) = f2-g2;
      *(jptr++) = f3-g3;
      *(jptr++) = f1-g1;
      *(jptr++) = f0-g0;
    }


  /* Do columns */

  for(i=0;i<8;i++)
    {
      jptr = iptr = y+i;


#ifdef PVERSION

      x0 = MSCALE(isqrt2*(*(iptr)));
      iptr += 8;
      x1 = *(iptr);
      iptr += 8;
      x2 = *(iptr);
      iptr += 8;
      x3 = *(iptr);
      iptr += 8;
      x4 = *(iptr);
      iptr += 8;
      x5 = *(iptr);
      iptr += 8;
      x6 = *(iptr);
      iptr += 8;
      x7 = *(iptr);

#else

#undef x1
#undef x2
#undef x3
#undef x4
#undef x5
#undef x6
#undef x7

#define x1 iptr[8]
#define x2 iptr[16]
#define x3 iptr[24]
#define x4 iptr[32]
#define x5 iptr[40]
#define x6 iptr[48]
#define x7 iptr[56]

      x0 = MSCALE(isqrt2*(*iptr));

#endif

      a1 = MSCALE(itwoc1d4*x4);
      r0 = x0+a1;
      r1 = x0-a1;

      a3 = MSCALE(itwoc1d4*(x2+x6));
      r2 = MSCALE(itwoc1d8*(x2+a3));
      r3 = MSCALE(itwoc3d8*(x2-a3));

      f0 = r0+r2;
      f1 = r1+r3;
      f2 = r0-r2;
      f3 = r1-r3;

      b1 = x3+x5;
      c1 = MSCALE(itwoc1d4*b1);
      s0 = x1+c1;
      s1 = x1-c1;

      b2 = x1+x3;
      b3 = x5+x7;
      c3 = MSCALE(itwoc1d4*(b2+b3));
      s2 = MSCALE(itwoc1d8*(b2+c3));
      s3 = MSCALE(itwoc3d8*(b2-c3));

      g0 = MSCALE(itwoc1d16*(s0+s2));
      g1 = MSCALE(itwoc3d16*(s1+s3));
      g2 = MSCALE(itwoc7d16*(s0-s2));
      g3 = MSCALE(itwoc5d16*(s1-s3));

      ihold = f0+g0;
      (*jptr) = IDCTSCALE(ihold);
      jptr += 8;
      ihold = f1+g1;
      (*jptr) = IDCTSCALE(ihold);
      jptr += 8;
      ihold = f3+g3;
      (*jptr) = IDCTSCALE(ihold);
      jptr += 8;
      ihold = f2+g2;
      (*jptr) = IDCTSCALE(ihold);
      jptr += 8;
      ihold = f2-g2;
      (*jptr) = IDCTSCALE(ihold);
      jptr += 8;
      ihold = f3-g3;
      (*jptr) = IDCTSCALE(ihold);
      jptr += 8;
      ihold = f1-g1;
      (*jptr) = IDCTSCALE(ihold);
      jptr += 8;
      ihold = f0-g0;
      (*jptr) = IDCTSCALE(ihold);

    }
}

#undef f0
#undef f1
#undef f2
#undef f3

#undef g0
#undef g1
#undef g2
#undef g3

#undef r0
#undef r1
#undef r2
#undef r3

#undef s0
#undef s1
#undef s2
#undef s3

#define f0 rx0
#define f1 rx1
#define f2 rx2
#define f3 rx3

#define r0 rx4
#define r1 rx5
#define r2 rx6
#define r3 rx7

#define g0 sx0
#define g1 sx1
#define g2 sx2
#define g3 sx3

#define s0 rx4
#define s1 rx5
#define s2 rx6
#define s3 rx7


/*BFUNC

LeeDct is implemented by reversing the arrows in the inverse dct flow
diagram.  It takes two input arrays that must be defined before the
call.

EFUNC*/

void LeeDct(x,y)
     int *x;
     int *y;
{
  register int rx0,rx1,rx2,rx3,rx4,rx5,rx6,rx7;
  register int sx0,sx1,sx2,sx3;
  register int hold,c2;
  register int *iptr,*jptr;

#undef x0
#undef x1
#undef x2
#undef x3
#undef x4
#undef x5
#undef x6
#undef x7

#define x0 iptr[0]
#define x1 iptr[1]
#define x2 iptr[2]
#define x3 iptr[3]
#define x4 iptr[4]
#define x5 iptr[5]
#define x6 iptr[6]
#define x7 iptr[7]

  for(jptr=y,iptr=x;iptr<x+64;jptr+=8,iptr+=8)
    {
      f0 = x0+x7;
      g0 = MSCALE(twoc1d16*(x0-x7));
      f1 = x1+x6;
      g1 = MSCALE(twoc3d16*(x1-x6));
      f3 = x2+x5;
      g3 = MSCALE(twoc5d16*(x2-x5));
      f2 = x3+x4;
      g2 = MSCALE(twoc7d16*(x3-x4));

      r0 = f0+f2;
      r1 = f1+f3;
      r2 = MSCALE(twoc1d8*(f0-f2));
      r3 = MSCALE(twoc3d8*(f1-f3));

      jptr[0] = MSCALE(sqrt2*(r0+r1));
      jptr[4] = MSCALE(twoc1d4*(r0-r1));
      jptr[2] = hold = r2+r3;
      jptr[6] = MSCALE(twoc1d4*(r2-r3))-hold;

      s0 = g0+g2;
      s1 = g1+g3;
      s2 = MSCALE(twoc1d8*(g0-g2));
      s3 = MSCALE(twoc3d8*(g1-g3));

      jptr[1] = hold = s0+s1;
      c2 = s2+s3;
      jptr[3] = hold = c2-hold;
      jptr[5] = hold = MSCALE(twoc1d4*(s0-s1))-hold;
      jptr[7] = MSCALE(twoc1d4*(s2-s3))-c2-hold;
    }

#undef x0
#undef x1
#undef x2
#undef x3
#undef x4
#undef x5
#undef x6
#undef x7

#define x0 iptr[0]
#define x1 iptr[8]
#define x2 iptr[16]
#define x3 iptr[24]
#define x4 iptr[32]
#define x5 iptr[40]
#define x6 iptr[48]
#define x7 iptr[56]

#define y0 rx0
#define y1 rx0
#define y2 rx1
#define y3 rx1
#define y4 rx2
#define y5 rx2
#define y6 rx3
#define y7 rx3

  for(jptr=y,iptr=y;iptr<y+8;jptr++,iptr++)
    {
      f0 = x0+x7;
      g0 = MSCALE(twoc1d16*(x0-x7));
      f1 = x1+x6;
      g1 = MSCALE(twoc3d16*(x1-x6));
      f3 = x2+x5;
      g3 = MSCALE(twoc5d16*(x2-x5));
      f2 = x3+x4;
      g2 = MSCALE(twoc7d16*(x3-x4));

      r0 = f0+f2;
      r1 = f1+f3;
      r2 = MSCALE(twoc1d8*(f0-f2));
      r3 = MSCALE(twoc3d8*(f1-f3));

      y0 = MSCALE(sqrt2*(r0+r1));
      y4 = MSCALE(twoc1d4*(r0-r1));
      y2 = r2+r3;
      y6 = MSCALE(twoc1d4*(r2-r3))-y2;

      jptr[0] = DCTSCALE(y0);
      jptr[16] = DCTSCALE(y2);
      jptr[32] = DCTSCALE(y4);
      jptr[48] = DCTSCALE(y6);

      s0 = g0+g2;
      s1 = g1+g3;
      s2 = MSCALE(twoc1d8*(g0-g2));
      s3 = MSCALE(twoc3d8*(g1-g3));

      y1 = s0+s1;
      c2 = s2+s3;
      y3 = c2-y1;
      y5 = MSCALE(twoc1d4*(s0-s1))-y3;
      y7 = MSCALE(twoc1d4*(s2-s3))-c2-y5;

      jptr[8] = DCTSCALE(y1);
      jptr[24] = DCTSCALE(y3);
      jptr[40] = DCTSCALE(y5);
      jptr[56] = DCTSCALE(y7);
    }
}

/*END*/
