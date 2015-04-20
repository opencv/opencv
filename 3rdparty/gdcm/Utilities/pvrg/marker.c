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
marker.c

This file contains the Marker library which uses the direct buffer
access routines bgetc...

************************************************************
*/

/*LABEL marker.c */

/*Include files */
#include "globals.h"
#include "stream.h"
#include "marker.h"
#ifdef SYSV
#include <sys/fcntl.h>
#endif
#include <stdlib.h> /* exit */

/*PUBLIC*/
extern void WriteSoi();
extern void WriteEoi();
extern void WriteJfif();
extern void WriteSof();
extern void WriteDri();
extern void WriteDqt();
extern void WriteSos();
extern void WriteDht();
extern void ReadSof();
extern void ReadDqt();
extern void ReadDht();
extern void ReadDri();
extern void ReadDnl();
extern int CheckMarker();
extern void CheckScan();
extern void ReadSos();
extern void MakeConsistentFrameSize();
/*PRIVATE*/

/* External marker definition */

extern FRAME *CFrame;
extern IMAGE *CImage;
extern SCAN *CScan;
extern int NumberMDU;
extern int Loud;
extern int izigzag_index[];

#define Zigzag(i) izigzag_index[i]

/*START*/

/*BFUNC

WriteSoi() puts an SOI marker onto the stream.

EFUNC*/

void WriteSoi()
{
  BEGIN("WriteSoi")

  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_SOI);
}

/*BFUNC

WriteEoi() puts an EOI marker onto the stream.

EFUNC*/

void WriteEoi()
{
  BEGIN("WriteEoi")

  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_EOI);
}

/*BFUNC

WriteJfif() puts an JFIF APP0 marker onto the stream.  This is a
generic 1x1 aspect ratio, no thumbnail specification.

EFUNC*/

void WriteJfif()
{
  BEGIN("WriteJfif")
  int Start,End;

  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_APP);
  Start = swtell();              /* Find out the start of position. */
  bputw(0);                      /* Put a 0 down onto the stream. */
  bputc(0x4a); bputc(0x46); bputc(0x49); bputc(0x46); bputc(0x00);
  bputc(0x01); bputc(0x02);   /*Version 1.02*/
  bputc(0x00); /* No absolute DPI */
  bputw(1);    /* Aspect ratio */
  bputw(1);
  bputc(0x00); /* No thumbnails */
  bputc(0x00);

  End = swtell();                /* Find out end of the marker. */
  swseek(Start);                 /* Rewind */
  bputw((End-Start) >> 3);       /* Put marker there. */
  if ((Start-End) & 0x7)         /* if not byte flush, then problems. */
    {
      WHEREAMI();
      printf("Bad frame marker, not byte flush.\n");
    }
  swseek(End);
}
/*BFUNC

WriteSof() puts an SOF marker onto the stream.

EFUNC*/

void WriteSof()
{
  BEGIN("WriteSof")
  int i,j,Start,End;

  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_SOF|(CFrame->Type&0xf));
  Start = swtell();              /* Find out the start of position. */
  bputw(0);                      /* Put a 0 down onto the stream. */
  bputc(CFrame->DataPrecision);
  if (!CFrame->InsertDnl) {bputw(CFrame->GlobalHeight);}
  else {bputw(0);}
  bputw(CFrame->GlobalWidth);
  bputc(CFrame->GlobalNumberComponents);
  for(i=0;i<CFrame->GlobalNumberComponents;i++)
    {
      bputc(j=CFrame->cn[i]);               /* Store off in index */
      bputn(CFrame->hf[j],CFrame->vf[j]);
      bputc(CFrame->tq[j]);
    }
  End = swtell();                /* Find out end of the marker. */
  swseek(Start);                 /* Rewind */
  bputw((End-Start) >> 3);       /* Put marker there. */
  if ((Start-End) & 0x7)         /* if not byte flush, then problems. */
    {
      WHEREAMI();
      printf("Bad frame marker, not byte flush.\n");
    }
  swseek(End);
}

/*BFUNC

WriteDri() writes out a resync (or restart) interval out to the
stream. If unspecified, resync is not enabled.

EFUNC*/

void WriteDri()
{
  BEGIN("WriteDri")

  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_DRI);
  bputw(4); /* Constant length of 4 */
  bputw(CFrame->ResyncInterval);
}

/*BFUNC

WriteDnl() writes out a number of line marker out to the stream.  Note
that we must have defined number of lines before as 0.

EFUNC*/


void WriteDnl()
{
  BEGIN("WriteDnl")

  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_DNL);
  bputw(4); /* Constant length of 4 */
  bputw(CFrame->GlobalHeight);
}

/*BFUNC

WriteDqt() writes out the quantization matrices in the CImage
structure.

EFUNC*/

void WriteDqt()
{
  BEGIN("WriteDqt")
  int i,j,bignum_p,Start,End,*qmatrix;

  if (!(CScan->NumberQTablesSend))
    return;   /* No tables to transmit, then ignore. */
  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_DQT);
  Start = swtell();
  bputw(0);
  for(i=0;i<CScan->NumberQTablesSend;i++)
    {
      qmatrix = CImage->QuantizationMatrices[CScan->sq[i]];
      for(bignum_p=0,j=63;j>=0;j--)
  {
    if(qmatrix[j]>255)
      {
        bignum_p=0x10;
        break;
      }
  }
      bputc((bignum_p|CScan->sq[i])); /* Precision defined for big numbers */
      if (bignum_p)
  {
    for(j=0;j<64;j++)
      bputw(qmatrix[Zigzag(j)]);
  }
      else
  {
    for(j=0;j<64;j++)
      bputc(qmatrix[Zigzag(j)]);
  }
    }
  CScan->NumberQTablesSend=0; /* Clear out queue */
  End = swtell();       /* Assume a marker code will follow.*/
  swseek(Start);        /* bputc(END_QUANTIZATION_TABLE);*/
  bputw((End-Start) >> 3);
  if ((Start-End) & 0x7)
    {
      WHEREAMI();
      printf("DQT marker not byte flush.\n");
    }
  swseek(End);
}

/*BFUNC

WriteSos() writes a start of scan marker.

EFUNC*/

void WriteSos()
{
  BEGIN("WriteSos")
  int i,Start,End;

  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_SOS);
  Start = swtell();
  bputw(0);
  bputc(CScan->NumberComponents);
  for(i=0;i<CScan->NumberComponents;i++)
    {
      bputc(CScan->ci[i]);
      bputn(CScan->td[i],CScan->ta[i]);
    }
  bputc(CScan->SSS);
  bputc(CScan->SSE);
  bputn(CScan->SAH,CScan->SAL);
  End = swtell();
  swseek(Start);
  bputw((End-Start) >> 3);
  if ((Start-End) & 0x7)
    {
      WHEREAMI();
      printf("Bad scan marker not byte flush.\n");
    }
  swseek(End);
}

/*BFUNC

WriteDht() writes out the Huffman tables to send.

EFUNC*/

void WriteDht()
{
  BEGIN("WriteDht")
  int i,Start,End;

  if (!(CScan->NumberDCTablesSend) && !(CScan->NumberACTablesSend))
    return;   /* No tables to transmit, then ignore. */
  swbytealign();
  bputc(MARKER_MARKER);
  bputc(MARKER_DHT);
  Start = swtell();
  bputw(0);
  for(i=0;i<CScan->NumberDCTablesSend;i++)
    {
      bputc(CScan->sd[i]);
      UseDCHuffman(CScan->sd[i]);
      WriteHuffman();
    }
  for(i=0;i<CScan->NumberACTablesSend;i++)
    {
      bputc(CScan->sa[i]|0x10);
      UseACHuffman(CScan->sa[i]);
      WriteHuffman();
    }
  CScan->NumberDCTablesSend=0; /* Clear out send queue */
  CScan->NumberACTablesSend=0;
  /*
    We end on a new marker... so an end of code table is unnecessary.
    bputc(END_CODE_TABLE);
    */
  End = swtell();
  swseek(Start);
  bputw((End-Start) >> 3);
  if ((Start-End) & 0x7)
    {
      WHEREAMI();
      printf("Bad scan marker not byte flush.\n");
    }
  swseek(End);
}


/*BFUNC

  ReadSof() reads a start of frame marker from the stream. We assume that
  the first two bytes (marker prefix) have already been stripped.

  EFUNC*/

void ReadSof(Type)
     int Type;
{
  BEGIN("ReadSof")
  int i,j,Length,Start,End,rb;

  Start = srtell();
  Length = bgetw();
  if (Loud > MUTE)
    printf("Frame Length %d\n",Length);
  CFrame->Type=Type;
  CFrame->DataPrecision = bgetc();
  CFrame->GlobalHeight = bgetw();
  CFrame->GlobalWidth = bgetw();

  for(i=0;i<MAXIMUM_COMPONENTS;i++)
    CFrame->hf[i]=CFrame->vf[i]=CFrame->tq[i]=0;
  CFrame->GlobalNumberComponents = bgetc();
  for(i=0;i<CFrame->GlobalNumberComponents;i++)
    {
      j = bgetc();
      rb = bgetc();
      CFrame->cn[i] = j;
      CFrame->hf[j] = hinyb(rb);
      CFrame->vf[j] = lonyb(rb);
      CFrame->tq[j] = bgetc();
    }
  MakeConsistentFrameSize();
  End = srtell();
  if ((End-Start) != (Length<<3))
    {
      WHEREAMI();
      printf("Bad read frame length.\n");
    }
  if (Loud > MUTE)
    {
      PrintImage();
      PrintFrame();
    }
}

/*BFUNC

  ReadDqt() reads a quantization table marker from the stream.
  The first two bytes have been stripped off.

  EFUNC*/

void ReadDqt()
{
  BEGIN("ReadDqt")
  int i,Length,Qget,Index,Precision,Start,End;

  Start = srtell();
  Length = bgetw();
  if (Loud > MUTE)
    printf("Quantization Length %d\n",Length);
  while((Qget=bgetc()) != END_QUANTIZATION_TABLE)
    {
      Index = Qget & 0xf;
      Precision = (Qget >> 4)&0xf;
      if (Precision > 1)
  {
    printf("Bad Precision: %d  in Quantization Download\n",
     Precision);
    printf("*** Dumping Image ***\n");
    PrintImage();
    printf("*** Dumping Frame ***\n");
    PrintFrame();
    exit(ERROR_MARKER);
  }                             /* Load in q-matrices */
      CImage->QuantizationMatrices[Index] = (int *) calloc(65,sizeof(int));
      if (Precision)               /* If precision then word quantization*/
  {
    for(i=0;i<64;i++)
      {
        if (!(CImage->QuantizationMatrices[Index][Zigzag(i)]=bgetw()))
    {
      printf("marker.c:ReadDqt: Quantization value of zero.\n");
      if (i)
        {
          printf("marker.c:ReadDqt: Changing to i-1.\n");
          CImage->QuantizationMatrices[Index][Zigzag(i)]=
      CImage->QuantizationMatrices[Index][Zigzag(i-1)];
        }
      else
        {
          printf("marker.c:ReadDqt: Changing to 16.\n");
          CImage->QuantizationMatrices[Index][Zigzag(i)]=16;
        }
    }
      }
  }
      else                       /* Otherwise byte quantization */
  {
    for(i=0;i<64;i++)
      {
        if (!(CImage->QuantizationMatrices[Index][Zigzag(i)]=bgetc()))
    {
      printf("marker.c:ReadDqt: Quantization value of zero.\n");
      if (i)
        {
          printf("marker.c:ReadDqt: Changing to i-1.\n");
          CImage->QuantizationMatrices[Index][Zigzag(i)]=
      CImage->QuantizationMatrices[Index][Zigzag(i-1)];
        }
      else
        {
          printf("marker.c:ReadDqt: Changing to 16.\n");
          CImage->QuantizationMatrices[Index][Zigzag(i)]=16;
        }
    }
      }
  }
    }
  bpushc(END_QUANTIZATION_TABLE);
  End = srtell();
  if ((End-Start) != (Length<<3))
    {
      WHEREAMI();
      printf("Bad DQT read length.\n");
    }
  if (Loud > MUTE)
    {
      PrintImage();
      PrintFrame();
    }
}


/*BFUNC

  ReadDht() reads a Huffman marker from the stream. We assume that the
  first two bytes have been stripped off.

  EFUNC*/

void ReadDht()
{
  BEGIN("ReadDht")
  int Index,Where,Length,Start,End;

  Start = srtell();
  Length = bgetw();
  if (Loud > MUTE)
    printf("Define Huffman length %d\n",Length);
  while((Index = bgetc()) != END_CODE_TABLE)
    {
      Where = (Index >> 4) & 0x0f;       /* Find location to place it in */
      Index = Index & 0x0f;
      MakeXhuff();                       /* Make Huffman table */
      MakeDhuff();
      ReadHuffman();
      if (Where)
  {
    SetACHuffman(Index);           /* Set current Huffman limit */
    CImage->NumberACTables = MAX(CImage->NumberACTables,(Index+1));
  }
      else
  {
    SetDCHuffman(Index);
    CImage->NumberDCTables = MAX(CImage->NumberDCTables,(Index+1));
  }
    }
  bpushc(END_CODE_TABLE);
  End = srtell();
  if ((End-Start) != (Length<<3))
    {
      WHEREAMI();
      printf("Bad DHT length.\n");
    }
  if (Loud > MUTE)
    PrintImage();
}

/*BFUNC

  ReadDri() reads a resync interval marker from the stream. We assume
  the first two bytes are stripped off.

  EFUNC*/

void ReadDri()
{
  BEGIN("ReadDri")
  int Length;

  if ((Length=bgetw())!=4)            /* Constant length of 4 */
    {
      WHEREAMI();
      printf("Bad length %d, should be 4.\n",Length);
    }
  CFrame->ResyncInterval = bgetw();
}

/*BFUNC

  ReadDnl() reads a number of lines marker from the stream. The first
  two bytes should be stripped off.

  EFUNC*/

void ReadDnl()
{
  BEGIN("ReadDnl")
  int Length;

  if ((Length=bgetw())!=4)             /* Constant length of 4 */
    printf("marker.c:ReadDnl: Bad length %d, should be 4.\n",Length);
  CFrame->GlobalHeight = bgetw();
  if (CScan->NumberComponents)
    {
      MakeConsistentFrameSize();
      CheckScan();
      ResizeIob();
      if (CFrame->GlobalHeight)
  {
    InstallIob(0);
    if (CFrame->Type==3)
      NumberMDU = CScan->MDUWide*CScan->MDUHigh;
    else
      NumberMDU = CScan->MDUWide*CScan->MDUHigh;
  }
      else
  NumberMDU = -1;
    }
}

/*BFUNC

CheckMarker() checks to see if there is a marker in the stream ahead.
This function presumes that ungetc is not allowed to push more than
one byte back.

EFUNC*/

int CheckMarker()
{
  BEGIN("CheckMarker")
  int Length;
  int v1;

  Length = brtell();
  v1=bgetw();

  if (v1>=0xffc0)
    {
      brseek(Length,0L);
      return(v1&0xff);
    }
  brseek(Length,0L);
  return(0);
}

/*BFUNC

ReadSos() reads in a start of scan from the stream. The first two
bytes should have been stripped off.

EFUNC*/

void ReadSos()
{
  BEGIN("ReadSos")
  int i,Length,Start,End,rb;

  Start = srtell();
  Length = bgetw();
  if (Loud > MUTE)
    {
      WHEREAMI();
      printf("Scan length %d\n",Length);
    }
  CScan->NumberComponents = bgetc();
  for(i=0;i<CScan->NumberComponents;i++)
    {
      CScan->ci[i] = bgetc();
      rb = bgetc();
      CScan->td[i] =  hinyb(rb);
      CScan->ta[i] = lonyb(rb);
    }
  CScan->SSS = bgetc();
  CScan->SSE = bgetc();
  rb = bgetc();
  CScan->SAH = hinyb(rb);
  CScan->SAL = lonyb(rb);

  End = srtell();
  if ((End-Start) != (Length<<3))
    {
      WHEREAMI();
      printf("Bad scan length.\n");
    }
  if (Loud > MUTE)
    PrintScan();
  MakeConsistentFileNames();   /* A Scan marker always makes new files */
  CheckValidity();
  CheckBaseline();
  CheckScan();
  /* Create the io buffer structure */

  if (CFrame->Type==3)
    {
      MakeIob(IOB_LINE,O_RDWR | O_CREAT,
        ((CFrame->DataPrecision>8)?2:1));
      if (CFrame->GlobalHeight)
  {
    InstallIob(0);
    NumberMDU = CScan->MDUWide*CScan->MDUHigh;
  }
      else NumberMDU = -1;
    }
  else
    {
      MakeIob(IOB_BLOCK,O_RDWR | O_CREAT | O_TRUNC,
        ((CFrame->DataPrecision>8)?2:1));
      if (CFrame->GlobalHeight)
  {
    InstallIob(0);
    NumberMDU = CScan->MDUWide*CScan->MDUHigh;
  }
      else NumberMDU = -1;
    }

  /* Sometimes rewinding is necessary */
  /* for(i=0;i<CScan->NumberComponents;i++)
    {
      InstallIob(i);
      RewindIob();
    } */
  ResetCodec();                /* Reset codec for information */
}

/*BFUNC

CheckScan() sets the MDU dimensions for the CScan structure.

EFUNC*/

void CheckScan()
{
  int i;

  if (CScan->NumberComponents==1)
    {
      i = (((CFrame->GlobalWidth*CFrame->hf[CScan->ci[0]])-1)/CFrame->Maxh)+1;
      if (CFrame->Type!=3)
  i = ((i-1)/8)+1;
      CScan->MDUWide = i;

      i = (((CFrame->GlobalHeight*CFrame->vf[CScan->ci[0]])-1)/CFrame->Maxv)+1;
      if (CFrame->Type!=3)
  i = ((i-1)/8)+1;
      CScan->MDUHigh = i;
    }
  else
    {
      CScan->MDUWide=CFrame->MDUWide;
      CScan->MDUHigh=CFrame->MDUHigh;
    }
}

/*BFUNC

MakeConsistentFrameSize() makes a consistent frame size for all of the
horizontal and vertical frequencies read.

EFUNC*/

void MakeConsistentFrameSize()
{
  BEGIN("MakeConsistentFrameSize")
  int i,Maxh,Maxv;
  int TestWide, TestHigh;

  Maxv = Maxh = 1;
  for(i=0;i<MAXIMUM_COMPONENTS;i++)
    {
      if (CFrame->vf[i] > Maxv)
  Maxv = CFrame->vf[i];
      if (CFrame->hf[i] > Maxh)
  Maxh = CFrame->hf[i];
    }

  for(i=0;i<MAXIMUM_COMPONENTS;i++)       /* Define estimated actual width */
    {                                     /* ignoring replications */
      if (CFrame->hf[i])
  {
    if (!CFrame->Width[i])
      CFrame->Width[i] =
        (((CFrame->GlobalWidth*CFrame->hf[i])-1)/Maxh)+1;
    if (!CFrame->Height[i])
      CFrame->Height[i] =
        (((CFrame->GlobalHeight*CFrame->vf[i])-1)/Maxv)+1;
  }
    }

  CFrame->Maxv = Maxv;  CFrame->Maxh = Maxh;

  CFrame->MDUWide = (CFrame->GlobalWidth-1)/Maxh +1;
  if (CFrame->GlobalHeight)
    CFrame->MDUHigh = (CFrame->GlobalHeight-1)/Maxv +1;
  else
    CFrame->MDUHigh = 0;

  if (CFrame->Type!=3)
    {
      CFrame->MDUWide= (CFrame->MDUWide-1)/8 +1;
      if (CFrame->MDUHigh)
  CFrame->MDUHigh= (CFrame->MDUHigh-1)/8 +1;
    }

  for(i=0;i<MAXIMUM_COMPONENTS;i++)
    {
      if (CFrame->hf[i])
  {
    TestWide = ((CFrame->Width[i]-1)/(CFrame->hf[i]))+1;
    if (CFrame->Type!=3) TestWide= (TestWide-1)/8 +1;

    if (CFrame->MDUWide!=TestWide)
      {
        WHEREAMI();
        printf("Inconsistent frame width.\n");
        printf("Component[%dx%d]\n",
         CFrame->Width[i],CFrame->Height[i]);
        printf("MDU Wide: Image, Component %d!= %d.\n",
         CFrame->MDUWide,TestWide);
      }
    if (CFrame->MDUHigh)
      {
        TestHigh = ((CFrame->Height[i]-1)/(CFrame->vf[i]))+1;
        if (CFrame->Type!=3) TestHigh= (TestHigh-1)/8 +1;
        if (CFrame->MDUHigh!=TestHigh)
    {
      WHEREAMI();
      printf("Inconsistent frame height.\n");
      printf("Component[%dx%d]\n",
       CFrame->Width[i],CFrame->Height[i]);
      printf("MDU High: Image, Component %d!= %d.\n",
       CFrame->MDUHigh,TestHigh);
    }
      }
  }
    }
}



/*END*/
