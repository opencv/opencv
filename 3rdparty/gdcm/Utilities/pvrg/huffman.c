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
huffman.c

This file represents the core Huffman routines, most of them
implemented with the JPEG reference. These routines are not very fast
and can be improved, but comprise very little of software run-time.

************************************************************
*/

/*LABEL huffman.c */

/* Include files */

#include "globals.h"
#include "stream.h"
#include <stdlib.h> /* exit */

/*PUBLIC*/

static void CodeSize();
static void CountBits();
static void AdjustBits();
static void SortInput();
static void SizeTable();
static void CodeTable();
static void OrderCodes();
static void DecoderTables();

extern void MakeHuffman();
extern void SpecifiedHuffman();
extern void MakeDecoderHuffman();
extern void ReadHuffman();
extern void WriteHuffman();
extern int DecodeHuffman();
extern void EncodeHuffman();
extern void MakeXhuff();
extern void MakeEhuff();
extern void MakeDhuff();
extern void UseACHuffman();
extern void UseDCHuffman();
extern void SetACHuffman();
extern void SetDCHuffman();
extern void PrintHuffman();
extern void PrintTable();

/*PRIVATE*/

extern int Loud;
extern int ErrorValue;
extern IMAGE *CImage;
extern FRAME *CFrame;
extern SCAN *CScan;


static int frequency[257];
static int codesize[257];
static int huffsize[257];
static int huffcode[257];
static int lastp;
static int others[257];
static XHUFF *Xhuff=NULL;
static DHUFF *Dhuff=NULL;
static EHUFF *Ehuff=NULL;

#define fgetb megetb
#define fputv meputv

#define ClearFrequency() \
{int *cfip; for(cfip=frequency;cfip<frequency+257;*(cfip++)=0);}
#define ClearCodeSize() \
{int *ccsip; for(ccsip=codesize;ccsip<codesize+257;*(ccsip++)=0);}
#define ClearOthers() \
{int *coip; for(coip=others;coip<others+257;*(coip++)= -1);}
#define ClearBits() \
{int *cbip; for(cbip=Xhuff->bits;cbip<Xhuff->bits+36;*(cbip++)=0);}
#define ClearEcodes() \
{int *cip,*dip;dip=Ehuff->ehufsi;cip=Ehuff->ehufco;\
 while(cip<codesize+257){*(cip++)=0; *(dip++)=0;}}

/*START*/

/*BFUNC

CodeSize() is used to size up which codes are found. This part merely
generates a series of code lengths of which any particular usage is
determined by the order of frequency of access. Note that the code
word associated with 0xffff has been restricted.

EFUNC*/

static void CodeSize()
{
  BEGIN("CodeSize")
  int *cfip,i;
  int least_value,next_least_value;
  int least_value_index,next_least_value_index;

  frequency[256] = 1; /* Add an extra code to ensure 0xffff not taken. */
  ClearCodeSize();
  ClearOthers();
  while(1)
    {
      least_value = next_least_value = 0x7fffffff;  /* largest word */
      least_value_index = next_least_value_index = -1;
      cfip = frequency;
      for(i=0;i<257;i++)                      /* Find two smallest values */
  {
    if (*cfip)
      {
        if (*cfip <= least_value)
    {
      next_least_value = least_value;
      least_value = *cfip;
      next_least_value_index = least_value_index;
      least_value_index = i;
    }
        else if (*cfip <= next_least_value)
    {
      next_least_value = *cfip;
      next_least_value_index = i;
    }
      }
    cfip++;
  }
      if (next_least_value_index == -1)      /* If only one value, finished */
  {
    break;
  }
      frequency[least_value_index] += frequency[next_least_value_index];
      frequency[next_least_value_index] = 0;
      codesize[least_value_index]++;
      while(others[least_value_index] != -1)
  {
    least_value_index = others[least_value_index];
    codesize[least_value_index]++;
  }
      others[least_value_index] = next_least_value_index;
      do
  {
    codesize[next_least_value_index]++;
  }
      while((next_least_value_index = others[next_least_value_index]) != -1);
    }
}

/*BFUNC

CountBits() tabulates a histogram of the number of codes with a give
bit-length.

EFUNC*/

static void CountBits()
{
  BEGIN("CountBits")
  int *csptr;

  ClearBits();
  for(csptr=codesize+256;csptr>=codesize;csptr--)
    {
      if (*csptr)
  {
    Xhuff->bits[*csptr]++;
  }
    }
}

/*BFUNC

AdjustBits() is used to trim the Huffman code tree into 16 bit code
words only.

EFUNC*/

static void AdjustBits()
{
  BEGIN("AdjustBits")
  int i,j;

  i=32;
  while(1)
    {
      if (Xhuff->bits[i]>0)
  {
    j = i-1;
    while(!Xhuff->bits[--j]);  /* Change from JPEG Manual */
    Xhuff->bits[i] -= 2;       /* Remove 2 of the longest hufco */
    Xhuff->bits[i-1]++;        /* Add one hufco to its prefix */
    Xhuff->bits[j]--;          /* Remove hufco from next length */
    Xhuff->bits[j+1] += 2;     /* to be prefix to one hufco */
  }                            /* from j term and the one */
                                     /* hufco from the i (longest) term.*/
      else if (--i==16)
  {
    break;
  }
    }
  while(!Xhuff->bits[i])             /* If fortunate enough not to use */
    {                                /* any 16 bit codes, then find out */
      i--;                           /* where last codes are. */
    }
  Xhuff->bits[i]--;  /* Get rid of the extra code that generated 0xffff */
}

/*BFUNC

SortInput() assembles the codes in increasing order with code length.
Since we know the bit-lengths in increasing order, they will
correspond to the codes with decreasing frequency. This sort is O(mn),),
not the greatest.

EFUNC*/

static void SortInput()
{
  BEGIN("SortInput")
  int i,j,p;

  for(p=0,i=1;i<33;i++)  /* Designate a length in i. */
    {
      for(j=0;j<256;j++) /* Find all codes with a given length. */
  {
    if (codesize[j]==i)
      {
        Xhuff->huffval[p++] = j;  /* Add that value to be associated */
      }                           /* with the next largest code. */
  }
    }
}

/*BFUNC

SizeTable() is used to associate a size with the code in increasing
length. For example, it would be 44556677... in huffsize[].  Lastp is
the number of codes used.

EFUNC*/

static void SizeTable()
{
  BEGIN("SizeTable")
  int i,j,p;

  for(p=0,i=1;i<17;i++)
    {
      for(j=1;j<=Xhuff->bits[i];j++)
  {
    huffsize[p++] = i;
  }
    }
  huffsize[p] = 0;
  lastp = p;
}


/*BFUNC

CodeTable() is used to generate the codes once the hufsizes are known.

EFUNC*/

static void CodeTable()
{
  BEGIN("CodeTable")
  int p,code,size;

  p=0;
  code=0;
  size = huffsize[0];
  while(1)
    {
      do
  {
    huffcode[p++] = code++;
  }
      while((huffsize[p]==size)&&(p<257)); /* Overflow Detection */
      if (!huffsize[p]) /* All finished. */
  {
    break;
  }
      do                /* Shift next code to expand prefix. */
  {
    code <<= 1;
    size++;
  }
      while(huffsize[p] != size);
    }
}

/*BFUNC

OrderCodes() reorders from the monotonically increasing Huffman-code
words into an array which is indexed on the actual value represented
by the codes. This converts the Xhuff structure into an Ehuff
structure.

EFUNC*/

static void OrderCodes()
{
  BEGIN("OrderCodes")
  int index,p;

  for(p=0;p<lastp;p++)
    {
      index = Xhuff->huffval[p];
      Ehuff->ehufco[index] = huffcode[p];
      Ehuff->ehufsi[index] = huffsize[p];
    }
}

/*BFUNC

DecoderTables() takes the Xhuff and converts it to a form suitable for
the JPEG suggested decoder. This is not the fastest method but it is
the reference method.

EFUNC*/

static void DecoderTables()
{
  BEGIN("DecoderTables")
  int l,p;

  for(Dhuff->ml=1,p=0,l=1;l<=16;l++)
    {
      if (Xhuff->bits[l]==0)
  {
    Dhuff->maxcode[l] = -1; /* Watch out JPEG is wrong here */
  }                         /* We use -1 to indicate skipping. */
      else
  {
    Dhuff->valptr[l]=p;
    Dhuff->mincode[l]=huffcode[p];
    p+=Xhuff->bits[l]-1;
    Dhuff->maxcode[l]=huffcode[p];
    Dhuff->ml = l;
    p++;
  }
    }
  Dhuff->maxcode[Dhuff->ml]++;
}

/*BFUNC

MakeHuffman() is used to create the Huffman table from the frequency
passed into it.

EFUNC*/

void MakeHuffman(freq)
     int *freq;
{
  BEGIN("MakeHuffman")
  int *ptr;

  for(ptr=frequency;ptr<frequency+256;ptr++)
    *ptr= *(freq++);

  CodeSize();
  CountBits();
  AdjustBits();
  SortInput();
  SizeTable();         /*From Xhuff to Ehuff */
  CodeTable();
  OrderCodes();
}

/*BFUNC

SpecifiedHuffman() is used to create the Huffman table from the bits
and the huffvals passed into it.

EFUNC*/

void SpecifiedHuffman(bts,hvls)
     int *bts;
     int *hvls;
{
  BEGIN("MakeHuffman")
  int i;
  int accum;

  for(accum=0,i=0;i<16;i++)
    {
      accum+= bts[i];
      Xhuff->bits[i+1] = bts[i];  /* Shift offset for internal specs.*/
    }
  for(i=0;i<accum;i++)
    {
      Xhuff->huffval[i] = hvls[i];
    }
  SizeTable();         /*From Xhuff to Ehuff */
  CodeTable();
  OrderCodes();
}

/*BFUNC

MakeDecoderHuffman() creates the decoder tables from the Xhuff structure.

EFUNC*/

void MakeDecoderHuffman()
{
  BEGIN("MakeDecoderHuffman")

  SizeTable();
  CodeTable();
  DecoderTables();
}

/*BFUNC

ReadHuffman() reads in a Huffman structure from the currently open
stream.

EFUNC*/

void ReadHuffman()
{
  BEGIN("ReadHuffman")
  int i,accum;

  for(accum=0,i=1;i<=16;i++)
    {
      Xhuff->bits[i]=bgetc();
      accum += Xhuff->bits[i];
    }
  if (Loud > NOISY)
    {
      printf("Huffman Read In:\n");
      printf("NUMBER OF CODES %d\n",accum);
    }
  for(i=0;i<accum;i++)
    {
      Xhuff->huffval[i] = bgetc();
    }
  SizeTable();
  CodeTable();
  DecoderTables();
  if (Loud > NOISY)
    {
      printf("Huffman Read In:\n");
      for(i=1;i<=16;i++)
  {
    printf("DHUFF->MAXCODE DHUFF->MINCODE DHUFF->VALPTR %d %d %d\n",
     Dhuff->maxcode[i],Dhuff->mincode[i],Dhuff->valptr[i]);
  }
    }
}

/*BFUNC

WriteHuffman() writes the Huffman out to the stream. This Huffman
structure is written from the Xhuff structure.

EFUNC*/

void WriteHuffman()
{
  BEGIN("WriteHuffman")
  int i,accum;

  if (Xhuff)
    {
      for(accum=0,i=1;i<=16;i++)
  {
    bputc(Xhuff->bits[i]);
    accum += Xhuff->bits[i];
  }
      for(i=0;i<accum;i++)
  {
    bputc(Xhuff->huffval[i]);
  }
    }
  else
    {
      WHEREAMI();
      printf("Null Huffman table found.\n");
    }
}

/*BFUNC

DecodeHuffman() returns the value decoded from the Huffman stream.
The Dhuff must be loaded before this function be called.

EFUNC*/

int DecodeHuffman()
{
  BEGIN("DecodeHuffman")
  int code,l,p;

  if (!Dhuff)
    {
      WHEREAMI();
      printf("Unreferenced decoder Huffman table!\n");
      exit(ERROR_HUFFMAN_READ);
    }
  code = fgetb();
  for(l=1;code>Dhuff->maxcode[l];l++)
    {
      if (Loud > WHISPER)
  {
    WHEREAMI();
    printf("CurrentCode=%d Length=%d Dhuff->Maxcode=%d\n",
     code,l,Dhuff->maxcode[l]);
  }
      code= (code<<1)+fgetb();
    }
  if(code<Dhuff->maxcode[Dhuff->ml])
    {
      p = Dhuff->valptr[l] + code - Dhuff->mincode[l];
      if (Loud > WHISPER)
  {
    WHEREAMI();
    printf("HuffmanDecoded code: %d  value: %d\n",p,Xhuff->huffval[p]);
  }
      return(Xhuff->huffval[p]);
    }
  else
    {
      WHEREAMI();
      /*printf("Huffman read error: l=%d code=%d\n");*/
      Resync();
      ErrorValue = ERROR_HUFFMAN_READ;
      return(0);
    }
}

/*BFUNC

EncodeHuffman() places the Huffman code for the value onto the stream.

EFUNC*/

void EncodeHuffman(value)
     int value;
{
  BEGIN("EncodeHuffman")

  if (Loud > WHISPER)
    {
      WHEREAMI();
      printf("HUFFMAN_OUTPUT value=%d Ehuff->ehufsi=%d Ehuff->ehufco=%d\n",
       value,Ehuff->ehufsi[value],Ehuff->ehufco[value]);
    }
  if (!Ehuff)
    {
      WHEREAMI();
      printf("Encoding with Null Huffman table.\n");
      exit(ERROR_HUFFMAN_ENCODE);
    }
  if (Ehuff->ehufsi[value])
    {
      fputv(Ehuff->ehufsi[value],Ehuff->ehufco[value]);
    }
  else
    {
      WHEREAMI();
      printf("Null Code for [%d] Encountered:\n",value);
      printf("*** Dumping Huffman Table ***\n");
      PrintHuffman();
      printf("***\n");
      ErrorValue = ERROR_HUFFMAN_ENCODE;
      exit(ErrorValue);
    }
}

/*BFUNC

MakeXhuff() creates a Huffman structure and puts it into the current
slot.

EFUNC*/

void MakeXhuff()
{
  BEGIN("MakeXhuff")

  if (!(Xhuff = MakeStructure(XHUFF)))
    {
      WHEREAMI();
      printf("Cannot allocate memory for Xhuff structure.\n");
      exit(ERROR_MEMORY);
    }
}

/*BFUNC

MakeEhuff() creates a Huffman structure and puts it into the current
slot.

EFUNC*/

void MakeEhuff()
{
  BEGIN("MakeEhuff")

  if (!(Ehuff = MakeStructure(EHUFF)))
    {
      WHEREAMI();
      printf("Cannot allocate memory for Ehuff structure.\n");
      exit(ERROR_MEMORY);
    }
}

/*BFUNC

MakeDhuff() creates a Huffman structure and puts it into the current
slot.

EFUNC*/

void MakeDhuff()
{
  BEGIN("MakeDhuff")

  if (!(Dhuff = MakeStructure(DHUFF)))
    {
      WHEREAMI();
      printf("Cannot allocate memory for Dhuff structure.\n");
      exit(ERROR_MEMORY);
    }
}

/*BFUNC

UseACHuffman() installs the appropriate Huffman structure from the
CImage structure.

EFUNC*/

void UseACHuffman(index)
     int index;
{
  BEGIN("UseACHuffman")

  Xhuff = CImage->ACXhuff[index];
  Dhuff = CImage->ACDhuff[index];
  Ehuff = CImage->ACEhuff[index];
  if (!Dhuff && !Ehuff)
    {
      WHEREAMI();
      printf("Reference to nonexistent table %d.\n",index);
    }
}

/*BFUNC

UseDCHuffman() installs the DC Huffman structure from the CImage
structure.

EFUNC*/

void UseDCHuffman(index)
     int index;
{
  BEGIN("UseDCHuffman")

  Xhuff = CImage->DCXhuff[index];
  Dhuff = CImage->DCDhuff[index];
  Ehuff = CImage->DCEhuff[index];
  if (!Dhuff && !Ehuff)
    {
      WHEREAMI();
      printf("Reference to nonexistent table %d.\n",index);
    }
}

/*BFUNC

SetACHuffman() sets the CImage structure contents to be the current
Huffman structure.

EFUNC*/

void SetACHuffman(index)
     int index;
{
  BEGIN("SetACHuffman")

  CImage->ACXhuff[index] = Xhuff;
  CImage->ACDhuff[index] = Dhuff;
  CImage->ACEhuff[index] = Ehuff;
}

/*BFUNC

SetDCHuffman() sets the CImage structure contents to be the current
Huffman structure.

EFUNC*/

void SetDCHuffman(index)
     int index;
{
  BEGIN("SetDCHuffman")

  CImage->DCXhuff[index] = Xhuff;
  CImage->DCDhuff[index] = Dhuff;
  CImage->DCEhuff[index] = Ehuff;
}

/*BFUNC

PrintHuffman() prints out the current Huffman structure.

EFUNC*/

void PrintHuffman()
{
  BEGIN("PrintHuffman")
  int i;

  if (Xhuff)
    {
      printf("Xhuff ID: %p\n",(void*)Xhuff);
      printf("Bits: [length:number]\n");
      for(i=1;i<9;i++)
  {
    printf("[%d:%d]",i,Xhuff->bits[i]);
  }
      printf("\n");
      for(i=9;i<17;i++)
  {
    printf("[%d:%d]",i,Xhuff->bits[i]);
  }
      printf("\n");

      printf("Huffval:\n");
      PrintTable(Xhuff->huffval);
    }
  if (Ehuff)
    {
      printf("Ehuff ID: %p\n",(void*)Ehuff);
      printf("Ehufco:\n");
      PrintTable(Ehuff->ehufco);
      printf("Ehufsi:\n");
      PrintTable(Ehuff->ehufsi);
    }
  if (Dhuff)
    {
      printf("Dhuff ID: %p\n",(void*)Dhuff);
      printf("MaxLength: %d\n",Dhuff->ml);
      printf("[index:MaxCode:MinCode:ValPtr]\n");
      for(i=1;i<5;i++)
  {
    printf("[%d:%2x:%2x:%2x]",
     i,
     Dhuff->maxcode[i],
     Dhuff->mincode[i],
     Dhuff->valptr[i]);
  }
      printf("\n");
      for(i=5;i<9;i++)
  {
    printf("[%d:%2x:%2x:%2x]",
     i,
     Dhuff->maxcode[i],
     Dhuff->mincode[i],
     Dhuff->valptr[i]);
  }
      printf("\n");
      for(i=9;i<13;i++)
  {
    printf("[%d:%2x:%2x:%2x]",
     i,
     Dhuff->maxcode[i],
     Dhuff->mincode[i],
     Dhuff->valptr[i]);
  }
      printf("\n");
      for(i=13;i<17;i++)
  {
    printf("[%d:%2x:%2x:%2x]",
     i,
     Dhuff->maxcode[i],
     Dhuff->mincode[i],
     Dhuff->valptr[i]);
  }
      printf("\n");
    }
}

/*BFUNC

PrintTable() prints out a table to the screen. The table is assumed to
be a 16x16 matrix represented by a single integer pointer.

EFUNC*/

void PrintTable(table)
     int *table;
{
  BEGIN("PrintTable")
  int i,j;

  for(i=0;i<16;i++)
    {
      for(j=0;j<16;j++)
  {
    printf("%2x ",*(table++));
  }
      printf("\n");
    }
}



/*END*/
