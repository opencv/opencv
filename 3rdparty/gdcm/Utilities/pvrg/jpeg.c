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
jpeg.c

This file contains the main calling routines for the JPEG coder.

************************************************************
*/

/*LABEL jpeg.c */

/* Include files. */

#include "tables.h"
#include "marker.h"
#include "globals.h"
#ifdef SYSV
#include <sys/fcntl.h>
#endif
#include <stdlib.h> /* exit */
#include <string.h> /* strlen */

/*
  Define the functions to be used with ANSI prototyping.
  */

/*PUBLIC*/

static void JpegEncodeFrame();
static void JpegDecodeFrame();
static void JpegDecodeScan();
static void JpegLosslessDecodeScan();
static void Help();

extern void PrintImage();
extern void PrintFrame();
extern void PrintScan();
extern void MakeImage();
extern void MakeFrame();
extern void MakeScanFrequency();
extern void MakeScan();
extern void MakeConsistentFileNames();
extern void CheckValidity();
extern int CheckBaseline();
extern void ConfirmFileSize();
extern void JpegQuantizationFrame();
extern void JpegDefaultHuffmanScan();
extern void JpegFrequencyScan();
extern void JpegCustomScan();
extern void JpegEncodeScan();

extern void JpegLosslessFrequencyScan();
extern void JpegLosslessEncodeScan();

/*PRIVATE*/

/* These variables occur in the stream definition. */

extern int CleartoResync;
extern int LastKnownResync;
extern int ResyncEnable;
extern int ResyncCount;
extern int EndofFile;
extern int EndofImage;

/* Define the parameter passing structures. */
IMAGE *CImage=NULL;           /* Current Image variables structure */
FRAME *CFrame=NULL;           /* Current Frame variables structure */
SCAN *CScan=NULL;             /* Current Scan variables structure */

/* Define the MDU counters. */
int CurrentMDU=0;             /* Holds the value of the current MDU */
int NumberMDU=0;              /* This number is the number of MDU's */

/* Define Lossless info */

int LosslessPredictorType=0;  /* The lossless predictor used */
int PointTransform=0;         /* This parameter affects the shifting in io.c */

/* How we break things up */

int ScanComponentThreshold=SCAN_COMPONENT_THRESHOLD;

/* Define the support/utility variables.*/
int ErrorValue=0;             /* Holds error upon return */
int Loud=MUTE;                /* Loudness gives level of debug traces */
int HuffmanTrace=0;        /* When set, dumps Huffman statistics */
int Notify=1;                 /* When set, gives image size feedback */
int Robust=0;
static int LargeQ=0;          /* When set, large quantization is enabled */

/* We default to the Chen DCT algorithm. */
vFunc *UseDct = ChenDct;       /* This is the DCT algorithm to use */
vFunc *UseIDct = ChenIDct;     /* This is the inverse DCT algorithm to use */

/* Add some macros to ease readability. */
#define DefaultDct (*UseDct)
#define DefaultIDct (*UseIDct)

/*START*/

/*BFUNC

main() is first called by the shell routine upon execution of the
program.

EFUNC*/

int main(argc,argv)
     int argc;
     char **argv;
{
  BEGIN("main")
  int i,ComponentIndex;
  int Oracle=0;     /* Oracle means that we use the lexer interactively */

  MakeImage();      /* Construct the image structures */
  MakeFrame();
  MakeScan();

  if (argc == 1)    /* No arguments then print help info */
    {
      Help();
      exit(-1);
    }

  ComponentIndex=1;  /* Start with index 1 (Could be zero, but JFIF compat) */
  for(i=1;i<argc;i++)  /* Else loop through all arguments. */
    {
    if (!strcmp(argv[i],"-JFIF"))
      CImage->Jfif=1;
    else if (!strcmp(argv[i],"-ci"))
      ComponentIndex=atoi(argv[++i]);
    else if (*(argv[i]) == '-')       /* Strip off first "dash" */
      {
      switch(*(++argv[i]))
        {
      case 'a':                      /* -a Reference DCT */
        UseDct = ReferenceDct;
        UseIDct = ReferenceIDct;
        break;
      case 'b':                      /* -b Lee DCT */
        UseDct = LeeDct;
        UseIDct = LeeIDct;
        break;
      case 'd':                      /* -d Decode */
        CImage->JpegMode = J_DECODER;
        break;
      case 'k':                      /* -k Lossless mode */
        CImage->JpegMode = J_LOSSLESS;
        CFrame->Type=3;
        LosslessPredictorType = atoi(argv[++i]);
        break;
      case 'f':
        switch(*(++argv[i]))
          {
        case 'w':                 /* -fw Frame width */
          CFrame->Width[ComponentIndex] =
            atoi(argv[++i]);
          break;
        case 'h':                 /* -fh Frame height */
          CFrame->Height[ComponentIndex] =
            atoi(argv[++i]);
          break;
        default:
          WHEREAMI();
          printf("Illegal option: f%c.\n",
            *argv[i]);
          exit(ERROR_BOUNDS);
          break;
          }
        break;
      case 'i':
        switch(*(++argv[i]))
          {
        case 'w':                /* -iw Image width */
          CFrame->GlobalWidth = atoi(argv[++i]);
          break;
        case 'h':                /* -ih Image height */
          CFrame->GlobalHeight = atoi(argv[++i]);
          break;
        default:
          WHEREAMI();
          printf("Illegal option: i%c.\n",
            *argv[i]);
          exit(ERROR_BOUNDS);
          break;
          }
        break;
      case 'h':                    /* -h horizontal frequency */
        CFrame->hf[ComponentIndex] =
          atoi(argv[++i]);
        break;
#ifndef PRODUCTION_VERSION
      case 'l':                    /* -l loudness for debugging */
        Loud = atoi(argv[++i]);
        break;
#endif
      case 'n':                    /* Set non-interleaved mode */
        ScanComponentThreshold=1;
        break;
      case 'o':                    /* -o Oracle mode (input parsing)*/
        Oracle=1;
        break;
      case 'p':
        CFrame->DataPrecision = atoi(argv[++i]);
        if (!CFrame->Type) CFrame->Type = 1;
        break;
      case 'r':                    /* -r resynchronization */
        CFrame->ResyncInterval = atoi(argv[++i]);
        break;
      case 'q':                    /* -q Q factor */
        if (*(++argv[i])=='l') LargeQ=1;
        CFrame->Q = atoi(argv[++i]);
        break;
      case 'v':                    /* -v vertical frequency */
        CFrame->vf[ComponentIndex] = atoi(argv[++i]);
        break;
      case 's':                    /* -s stream file name */
        CImage->StreamFileName = argv[++i];
        break;
      case 't':
        PointTransform=atoi(argv[++i]);
        break;
#ifndef PRODUCTION_VERSION
      case 'x':                    /* -x trace */
        HuffmanTrace = 1;
        break;
#endif
      case 'u':                    /* -u disable width/size output */
        Notify=0;
        break;
      case 'y':
        Robust=1;
        break;
      case 'z':                    /* -z use default Huffman */
        CImage->JpegMode |= J_DEFAULTHUFFMAN;
        break;
      case 'g':                    /* -g GDCM secret option */
        CFrame->tmpfile = atoi(argv[++i]); /* very bad programming but should work :) */
        break;
      default:
        WHEREAMI();
        printf("Illegal option in command line: %c.\n",
          *argv[i]);
        exit(ERROR_BOUNDS);
        break;
        }
      }
    else                               /* If not a "-" then a filename */
      {
      CFrame->cn[CFrame->GlobalNumberComponents++]= ComponentIndex;
      if (!CFrame->vf[ComponentIndex])
        CFrame->vf[ComponentIndex]=1;
      if (!CFrame->hf[ComponentIndex])
        CFrame->hf[ComponentIndex]=1;
      CFrame->ComponentFileName[ComponentIndex] = argv[i];
      ComponentIndex++;
      }
    }

  if (Oracle)            /* If Oracle set */
    {
      initparser();      /* Initialize interactive parser */
      parser();          /* parse input from stdin */
      exit(ErrorValue);
    }

  /* Otherwise act on information */

  if (!(GetFlag(CImage->JpegMode,J_DECODER)) &&  /* Check for files */
      (CFrame->GlobalNumberComponents == 0))
    {
      WHEREAMI();
      printf("No component file specified.\n");
      exit(ERROR_BOUNDS);
    }
  if (CImage->StreamFileName == NULL)            /* Check for stream name */
    {
    if (CFrame->ComponentFileName[CFrame->cn[0]])  /* If doesn't exist */
      {                                            /* Create one. */
      CImage->StreamFileName =
        (char *) calloc(strlen(CFrame->ComponentFileName[CFrame->cn[0]])+6,
          sizeof(char));
      sprintf(CImage->StreamFileName,"%s.jpg",
        CFrame->ComponentFileName[CFrame->cn[0]]);
      }
    else
      {
      WHEREAMI();
      printf("No stream filename.\n");
      exit(ERROR_BOUNDS);
      }
    }
  if (GetFlag(CImage->JpegMode,J_DECODER))       /* If decoder flag set then */
    {                                            /* decode frame. */
      JpegDecodeFrame();
    }
  else
    {
    if (!(CFrame->GlobalWidth) || !(CFrame->GlobalHeight)) /* Dimensions ? */
      {
      WHEREAMI();
      printf("Unspecified frame size.\n");
      exit(ERROR_BOUNDS);
      }
    swopen(CImage->StreamFileName,0);          /* Open output file, index 0*/
    JpegEncodeFrame();                         /* Encode the frame */
    swclose();                                 /* Flush remaining bits */
    }
  /*exit(ErrorValue);*/
  return(ErrorValue);
}

/*BFUNC

JpegEncodeFrame() handles the basic encoding of the routines provided that
CFrame and CImage are set up properly. It creates the appropriate
CScan to handle the intermediate variables.

EFUNC*/

static void JpegEncodeFrame()
{
  BEGIN("JpegEncodeFrame")
  int i,CurrentComponent;

  CurrentComponent=0;           /* Write start of image, start of frame */
  WriteSoi();
  if (CImage->Jfif) WriteJfif(); /* Write JFIF marker if necessary */
  MakeConsistentFrameSize();     /* Do it here when everything defined */
  JpegQuantizationFrame();       /* Set up quantization matrices */
  WriteSof();
  if (CFrame->ResyncInterval)   /* If resync enabled put DRI marker */
    {
      WriteDri();
    }
  while(1)                      /* This loop breaks up a large number of */
    {                           /* components into small scans */
    if (CFrame->GlobalNumberComponents<=CurrentComponent)
      {
      break;                /* All encoded */
      }
    else if (CFrame->GlobalNumberComponents-CurrentComponent <=
      ScanComponentThreshold)
      {                       /* If less/equal to (SCT) components do it */
      CScan->NumberComponents =
        CFrame->GlobalNumberComponents-CurrentComponent;
      for(i=0;CurrentComponent<CFrame->GlobalNumberComponents;
        CurrentComponent++,i++)
        {
        CScan->ci[i]=CFrame->cn[CurrentComponent];
        }
      }
    else
      {                       /* Break into (SCT) componets */
      CScan->NumberComponents = ScanComponentThreshold;
      for(i=0;i<ScanComponentThreshold;CurrentComponent++,i++)
        {
        CScan->ci[i]=CFrame->cn[CurrentComponent];
        }
      }
    CheckValidity();                  /* Check validity */
    CheckBaseline();                  /* See if type is correct */

    if (Loud > MUTE)
      {
      PrintImage();
      PrintFrame();
      PrintScan();
      }
    ConfirmFileSize();                /* Does files on disk agree? */
    if (GetFlag(CImage->JpegMode,J_LOSSLESS))
      {
      MakeIob(IOB_LINE,O_RDONLY,
        ((CFrame->DataPrecision>8)?2:1));  /* Make IO read struct*/
      JpegLosslessFrequencyScan();        /* Else make custom stables */
      JpegCustomScan(CUSTOM_DO_DC);
      WriteDht();                         /* write Huffman tables */
      JpegLosslessEncodeScan();
      }
    else if (GetFlag(CImage->JpegMode,J_DEFAULTHUFFMAN))
      {
      MakeIob(IOB_BLOCK,O_RDONLY,
        ((CFrame->DataPrecision>8)?2:1));  /* Make IO read struct */

      JpegDefaultHuffmanScan();     /* If default tables, then set up */
      WriteDqt();                       /* Write out quantization */
      WriteDht();                       /* and Huffman tables */
      JpegEncodeScan();                 /* Encode the scan */
      }
    else
      {
      MakeIob(IOB_BLOCK,O_RDONLY,
        ((CFrame->DataPrecision>8)?2:1));  /* Make IO read struct*/
      JpegFrequencyScan();              /* Else make custom tables */
      JpegCustomScan(CUSTOM_DO_AC|CUSTOM_DO_DC);
      WriteDqt();                       /* Write out quantization */
      WriteDht();                       /* and Huffman tables */
      JpegEncodeScan();                 /* Encode the scan */
      }
    for(i=0;i<CScan->NumberComponents;i++)  /* Close all components */
      {
      InstallIob(i);
      CloseIob();
      }
    }
  WriteEoi();                              /* All finished, Write eoi */
}

/*BFUNC

JpegQuantizationFrame() sets up the default quantization matrices to be
used in the scan. Not to be used with user-specified quantization.

EFUNC*/

void JpegQuantizationFrame()
{
  BEGIN("JpegQuantizationFrame")
  int i;

  if (CFrame->Q)                    /* if Q  rescale quantization matrices */
    CImage->QuantizationMatrices[0] =
      ScaleMatrix(CFrame->Q,Q_PRECISION,LargeQ,LuminanceQuantization);
  else
    CImage->QuantizationMatrices[0] = LuminanceQuantization;

  CScan->sq[CScan->NumberQTablesSend++] = 0; /* Queue luminance to send */
  if (CFrame->GlobalNumberComponents>1)
    {
    if (CFrame->Q)                 /* rescale quantization matrices */
      CImage->QuantizationMatrices[1] =
        ScaleMatrix(CFrame->Q,Q_PRECISION,LargeQ,ChrominanceQuantization);
    else
      CImage->QuantizationMatrices[1] = ChrominanceQuantization;
    CScan->sq[CScan->NumberQTablesSend++] = 1; /* Queue table to send */
    }
  for(i=0;i<CFrame->GlobalNumberComponents;i++)
    {
    if (i%ScanComponentThreshold)
      CFrame->tq[CFrame->cn[i]]=1; /* chrominance q for non-primaries */
    else
      CFrame->tq[CFrame->cn[i]]=0; /* luminance q starts each scan */
    }
}

/*BFUNC

JpegDefaultHuffmanScan() creates the default tables for baseline use.

EFUNC*/

void JpegDefaultHuffmanScan()
{
  BEGIN("JpegDefaultScan")
  int i;

  if (CFrame->DataPrecision>8)
    {
      WHEREAMI();
      printf("Default tables attempted with precision > 8.\n");
      exit(ERROR_BOUNDS);
    }
  MakeXhuff();                      /* Make luminance DC Huffman */
  MakeEhuff();
  SpecifiedHuffman(LuminanceDCBits,LuminanceDCValues);
  SetDCHuffman(0);
  MakeXhuff();                      /* Make luminance AC Huffman */
  MakeEhuff();
  SpecifiedHuffman(LuminanceACBits,LuminanceACValues);
  SetACHuffman(0);
  MakeXhuff();
  MakeEhuff();
  CScan->td[0] = 0;
  CScan->ta[0] = 0;
  CScan->sa[CScan->NumberACTablesSend++] = 0;  /* Queue to transmit table */
  CScan->sd[CScan->NumberDCTablesSend++] = 0;
  if(CScan->NumberComponents>1)     /* Make chrominance Huffman tables */
    {                               /* Only if necessary */
      SpecifiedHuffman(ChrominanceDCBits,ChrominanceDCValues);
      SetDCHuffman(1);
      MakeXhuff();
      MakeEhuff();
      SpecifiedHuffman(ChrominanceACBits,ChrominanceACValues);
      SetACHuffman(1);
      for(i=1;i<CScan->NumberComponents;i++)
        {
        CScan->td[i] = 1;
        CScan->ta[i] = 1;
        }
      CScan->sa[CScan->NumberACTablesSend++] = 1;
      CScan->sd[CScan->NumberDCTablesSend++] = 1;
      CImage->NumberACTables = MAX(CImage->NumberACTables,2);
      CImage->NumberDCTables = MAX(CImage->NumberDCTables,2);
    }
  else
    {
      CImage->NumberACTables = MAX(CImage->NumberACTables,1);
      CImage->NumberDCTables = MAX(CImage->NumberDCTables,1);
    }
}

/*BFUNC

JpegFrequencyScan() assembles the frequency statistics for the given
scan, making one AC Freq, DC Freq statistic per component specified.
This function should be used before making custom quantization tables.

EFUNC*/

void JpegFrequencyScan()
{
  BEGIN("JpegFrequencyScan")
  int i,j,h,v,dohf,dovf;
  int input[64],output[64];
  int DCTBound,DCTShift;

  InstallIob(0);                 /* Zero out for fast single-component */
  InstallPrediction(0);          /* operation. */
  InstallFrequency(0);
  CheckScan();
  NumberMDU = CScan->MDUWide*CScan->MDUHigh;
  ClearFrameFrequency();
  ResetCodec();
  DCTBound = ((CFrame->DataPrecision>8)?16383:1023);
  DCTShift = ((CFrame->DataPrecision>8)?2048:128);
  for(i=0;i<NumberMDU;i++)         /* Do for all MDU in image */
    {
    if ( i && (CFrame->ResyncInterval))
      {
      if (!(i % CFrame->ResyncInterval)) /* Resync the codec */
        ResetCodec();
      }
    for(j=0;j<CScan->NumberComponents;j++)
      {
      InstallIob(j);
      InstallPrediction(j);    /* Install statistics tables */
      InstallFrequency(j);
      if (CScan->NumberComponents==1)
        dohf=dovf=1;
      else
        {
        dohf = CFrame->hf[CScan->ci[j]];
        dovf = CFrame->vf[CScan->ci[j]];
        }
      for(v=0;v<dovf;v++)  /* Do encoding */
        {                                      /* and accum. stats */
        for(h=0;h<dohf;h++)
          {
          ReadBlock(input);
          PreshiftDctMatrix(input,DCTShift);
          DefaultDct(input,output);
          BoundDctMatrix(output,DCTBound);
          Quantize(output,
            CImage->
            QuantizationMatrices[CFrame->
            tq[CScan->ci[j]]]);
          ZigzagMatrix(output,input);
          FrequencyDC(*input);           /* Freq accumulates */
          FrequencyAC(input);            /* stats w/o encoding */
          }
        }
      }
    }
  for(i=0;i<CScan->NumberComponents;i++)  /* Rewind to start */
    {
      InstallIob(i);
      RewindIob();
    }
}

/*BFUNC

JpegCustomScan() assembles custom Huffman tables for the input.
It defaults to baseline unless FULLHUFFMAN flag is set.

EFUNC*/

void JpegCustomScan(flags)
     int flags;
{
  BEGIN("JpegCustomScan")
  int i,Sumbits;

  if ((GetFlag(CImage->JpegMode,J_FULLHUFFMAN)) ||
    (CScan->NumberComponents < 3))
    {
    for(i=0;i<CScan->NumberComponents;i++)
      {
      if (GetFlag(flags,CUSTOM_DO_DC))
        {
        MakeXhuff();
        MakeEhuff();
        MakeHuffman(CScan->DCFrequency[i]);
        SetDCHuffman(i);
        CScan->td[i] = i;
        CScan->sd[CScan->NumberDCTablesSend++] = i;
        }
      if (GetFlag(flags,CUSTOM_DO_AC))
        {
        MakeXhuff();
        MakeEhuff();
        MakeHuffman(CScan->ACFrequency[i]);
        SetACHuffman(i);
        CScan->ta[i] = i;
        CScan->sa[CScan->NumberACTablesSend++] = i;
        }
      InstallIob(i);
      RewindIob();
      }
    CImage->NumberACTables = MAX(CImage->NumberACTables,
      CScan->NumberComponents);
    CImage->NumberDCTables = MAX(CImage->NumberDCTables,
      CScan->NumberComponents);
    }
  else
    {
    if (GetFlag(flags,CUSTOM_DO_DC))
      {
      MakeXhuff();                   /* 0 Component has custom Huffman */
      MakeEhuff();
      MakeHuffman(CScan->DCFrequency[0]);
      SetDCHuffman(0);
      CScan->td[0] = 0;              /* 0 component uses tables 0 */
      CScan->sd[CScan->NumberDCTablesSend++] = 0; /* Queue to send */
      }
    if (GetFlag(flags,CUSTOM_DO_AC))
      {
      MakeXhuff();
      MakeEhuff();
      MakeHuffman(CScan->ACFrequency[0]);
      SetACHuffman(0);
      CScan->ta[0] = 0;
      CScan->sa[CScan->NumberACTablesSend++] = 0; /* Queue table send */
      }
    if (CScan->NumberComponents > 1)
      {
      if (GetFlag(flags,CUSTOM_DO_DC))
        {
        for(i=2;i<CScan->NumberComponents;i++) /* Rest share Huffman*/
          {                                    /* Accum. frequencies */
          AddFrequency(CScan->DCFrequency[1],CScan->DCFrequency[i]);
          }
        MakeXhuff();
        MakeEhuff();
        MakeHuffman(CScan->DCFrequency[1]);
        SetDCHuffman(1);
        for(i=1;i<CScan->NumberComponents;i++) /* Rest use table 1 */
          CScan->td[i] = 1;
        CScan->sd[CScan->NumberDCTablesSend++] = 1;/* Queue to send */
        }
      if (GetFlag(flags,CUSTOM_DO_AC))
        {
        for(i=2;i<CScan->NumberComponents;i++) /*Accum. frequencies */
          {
          AddFrequency(CScan->ACFrequency[1],CScan->ACFrequency[i]);
          }
        MakeXhuff();
        MakeEhuff();
        MakeHuffman(CScan->ACFrequency[1]);
        SetACHuffman(1);
        for(i=1;i<CScan->NumberComponents;i++) /* Rest use table 1 */
          CScan->ta[i] = 1;
        CScan->sa[CScan->NumberACTablesSend++] = 1;  /* Queue to send */
        }
      CImage->NumberACTables = MAX(CImage->NumberACTables,2);/*reset */
      CImage->NumberDCTables = MAX(CImage->NumberDCTables,2);/* limits */
      }
    else
      {
      CImage->NumberACTables = MAX(CImage->NumberACTables,1); /* Reset */
      CImage->NumberDCTables = MAX(CImage->NumberDCTables,1); /*  limits */
      }
    }
  if (HuffmanTrace)     /* If trace flag, then dump out frequency tables */
    {
    Sumbits = 0;
    for(i=0;i<CImage->NumberACTables;i++)
      {
      WHEREAMI();
      printf("AC Code Frequency: Table %d\n",i);
      PrintACEhuff(i);
      Sumbits += SizeACEhuff(i);
      }
    for(i=0;i<CImage->NumberDCTables;i++)
      {
      WHEREAMI();
      printf("DC Code Frequency: Table %d\n",i);
      PrintDCEhuff(i);
      Sumbits +=  SizeDCEhuff(i);
      }
    WHEREAMI();
    printf("Total bits: %d  bytes: %d\n",
      Sumbits,(Sumbits+7)/8);
    }
}

/*BFUNC

JpegEncodeScan() encodes the scan that is given to it. We assume that
the quantization and the Huffman tables have already been specified.

EFUNC*/

void JpegEncodeScan()
{
  BEGIN("JpegEncodeScan")
  int i,j,h,v,dohf,dovf;
  int input[64],output[64];
  int DCTBound,DCTShift;

  InstallIob(0);
  CheckScan();
  NumberMDU = CScan->MDUWide*CScan->MDUHigh;
  ClearFrameFrequency();
  ResetCodec();
  DCTBound = ((CFrame->DataPrecision>8)?16383:1023);
  DCTShift = ((CFrame->DataPrecision>8)?2048:128);
  ResyncCount=0;                /* Reset the resync counter for every scan */
  if (CFrame->InsertDnl>0) /* If DNL is greater than 0, insert */
    {                           /* into according Resync interval */
    if  (!(CFrame->ResyncInterval))
      WriteDnl();  /* Automatically write a dnl if no resync is enabled.*/
    else                      /* If DNL > MDU, then put in last resync */
      CFrame->InsertDnl = MAX(CFrame->InsertDnl,      /* interval */
        NumberMDU/CFrame->ResyncInterval);
    }
  WriteSos();                  /* Start of Scan */
  for(i=0;i<NumberMDU;i++)
    {
    if ( i && (CFrame->ResyncInterval))
      {
      if (!(i % CFrame->ResyncInterval)) /* Check for resync */
        {
        if ((i/CFrame->ResyncInterval)==CFrame->InsertDnl)
          {
          WriteDnl();                /* If resync matches use DNL */
          CFrame->InsertDnl=0;       /* Mission accomplished. */
          }
        WriteResync();                 /* Write resync */
        ResetCodec();
        }
      }
    for(j=0;j<CScan->NumberComponents;j++)
      {
      if (Loud > MUTE)
        {
        WHEREAMI();
        printf("[Pass 2 [Component:MDU] [%d:%d]]\n",j,i);
        }
      InstallIob(j);                    /* Install component j */
      InstallPrediction(j);
      if (CScan->NumberComponents==1)
        dohf=dovf=1;
      else
        {
        dohf = CFrame->hf[CScan->ci[j]];
        dovf = CFrame->vf[CScan->ci[j]];
        }
      for(v=0;v<dovf;v++)  /* loop thru MDU */
        {
        for(h=0;h<dohf;h++)
          {
          ReadBlock(input);                /* Read in */
          if (Loud > WHISPER)
            {
            WHEREAMI();
            printf("Raw input:\n");
            PrintMatrix(input);
            }
          PreshiftDctMatrix(input,DCTShift);        /* Shift */
          DefaultDct(input,output);        /* DCT */
          BoundDctMatrix(output,DCTBound); /* Bound, limit */
          Quantize(output,                 /* Quantize */
            CImage->
            QuantizationMatrices[CFrame->
            tq[CScan->ci[j]]]);
          ZigzagMatrix(output,input);      /* Zigzag trace */
          if (Loud > TALK)
            {
            WHEREAMI();
            printf("Cooked Output:\n");
            PrintMatrix(input);
            }
          UseDCHuffman(CScan->td[j]);
          EncodeDC(*input);               /* Encode DC component */
          UseACHuffman(CScan->ta[j]);
          EncodeAC(input);                /* Encode AC component */
          }
        }
      }
    }
  if (CFrame->InsertDnl==-2)    /* -2 is automatic DNL insertion */
    {
      WriteDnl();                 /* Put DNL here */
      CFrame->InsertDnl=0;
    }
  for(i=0;i<CScan->NumberComponents;i++)          /* Rewind to start */
    {
      InstallIob(i);
      RewindIob();
    }
}

/*BFUNC

JpegLosslessFrequencyScan() accumulates the frequencies into the DC
frequency index.

EFUNC*/

void JpegLosslessFrequencyScan()
{
  BEGIN("JpegLosslessFrequencyScan")
  int x,y,j,h,v,px;
  int height,width,horfreq,value;
  int MaxElem,CurrentElem,NumberElem;
  int StartofLine=1,UseType=1;              /* Start with type 1 coding */
  int *input;

  CheckScan();
  for(j=0;j<CScan->NumberComponents;j++)          /* Rewind to start */
    {
      InstallIob(j);
      RewindIob();
    }
  if (CScan->NumberComponents==1)       /* Calculate maximum number of */
    MaxElem= LOSSLESSBUFFERSIZE/4;      /* elements can be loaded in */
  else
    {
    MaxElem= LOSSLESSBUFFERSIZE/
      ((CFrame->vf[CScan->ci[0]]+1)*(CFrame->hf[CScan->ci[0]]+1));
    for(j=1;j<CScan->NumberComponents;j++)          /* Rewind to start */
      {
      x=LOSSLESSBUFFERSIZE/
        ((CFrame->vf[CScan->ci[j]]+1)*(CFrame->hf[CScan->ci[j]]+1));
      if (x < MaxElem) MaxElem=x;
      }
    }
  CScan->SSS=LosslessPredictorType;
  CScan->SAL=PointTransform;
  ClearFrameFrequency();
  InstallIob(0);             /* Set up values for fast non-interleaved mode */
  InstallFrequency(0);
  if (CScan->NumberComponents==1)
    height=horfreq=1;
  else
    {
      height=CFrame->vf[CScan->ci[0]];
      horfreq=CFrame->hf[CScan->ci[0]];
    }
  NumberMDU = CScan->MDUWide*CScan->MDUHigh;
  CurrentMDU=0;
  if ((CFrame->ResyncInterval)&&(CFrame->ResyncInterval % CScan->MDUWide))
    {
      WHEREAMI();
      printf("Resync Interval not an integer multiple of MDU's wide.\n");
      printf("Proceeding anyways.\n");
      if (MaxElem>=CFrame->ResyncInterval)
        MaxElem=CFrame->ResyncInterval;       /* Reduce to resync interval */
      else
        MaxElem=1;                            /* Can't proceed quickly */
    }
  CurrentElem=NumberElem=0;
  for(y=0;y<CScan->MDUHigh;y++)
    {
    for(x=0;x<CScan->MDUWide;x++)
      {
      if (CurrentMDU && (CFrame->ResyncInterval))
        {
        if (!(CurrentMDU % CFrame->ResyncInterval)) /* Check resync */
          {
          UseType=1;                     /* Reset codec */
          for(j=0;j<CScan->NumberComponents;j++)
            {
            InstallIob(j);
            LineResetBuffers();
            }
          }
        }
      if (!(CurrentMDU%CScan->MDUWide)&&(CurrentMDU)) /* Reset CScan type */
        {
        UseType=2;                           /* Start of line */
        StartofLine=1;                       /* uses top pel predictor */
        }
      CurrentElem++;
      if (CurrentElem>=NumberElem)
        {
        NumberElem = MIN((CScan->MDUWide-x),MaxElem);
        CurrentElem=0;
        for(j=0;j<CScan->NumberComponents;j++)
          {
          InstallIob(j);                    /* Install component j */
          ReadLine(NumberElem,              /* Read in some elements*/
            CScan->LosslessBuffer[j]);
          }
        }
      if (CScan->NumberComponents==1)
        {
        width=horfreq*NumberElem+1;
        input = &CScan->LosslessBuffer[0][CurrentElem];
        if (Loud > NOISY)
          {
          WHEREAMI();
          printf("[Pass 1 [Component:MDU:Total] [%d:%d:%d]]\n",
            0,CurrentMDU,NumberMDU);
          }
        switch(UseType) /* Same as lossless coding predictor*/
          {
        case 1:
          px = input[width];
          break;
        case 2:
          px = input[1];
          break;
        case 3:
          px = input[0];
          break;
        case 4:
          px = input[width] + input[1] - input[0];
          break;
        case 5:
          px = input[width] + ((input[1] - input[0])>>1);
          break;
        case 6:
          px = input[1] + ((input[width] - input[0])>>1);
          break;
        case 7:
          px = (input[1]+input[width])>>1;  /* No rounding */
          break;
        default:
          WHEREAMI();
          printf("Lossless mode %d not supported.\n",UseType);
          break;
          }
        value=input[width+1]-px;
        if (Loud > NOISY)
          printf("IN=%d  PX=%d  FRE: %d\n",
            input[width+1],px,value);
        LosslessFrequencyDC(value);
        }
      else
        {
        for(j=0;j<CScan->NumberComponents;j++)
          {
          if (Loud > NOISY)
            {
            WHEREAMI();
            printf("[Pass 1 [Component:MDU:Total] [%d:%d:%d]]\n",
              j,CurrentMDU,NumberMDU);
            }
          InstallFrequency(j);
          height=CFrame->vf[CScan->ci[j]];
          horfreq=CFrame->hf[CScan->ci[j]];
          width=horfreq*NumberElem+1;
          input = &CScan->LosslessBuffer[j][CurrentElem*horfreq];
          for(v=1;v<=height;v++)
            {
            for(h=1;h<=horfreq;h++)
              {
              switch(UseType) /* lossless coding predictor*/
                {
              case 1:
                px = input[(v*(width))+h-1];
                break;
              case 2:
                px = input[((v-1)*(width))+h];
                break;
              case 3:
                px = input[((v-1)*(width))+h-1];
                break;
              case 4:
                px = input[(v*(width))+h-1] +
                  input[((v-1)*(width))+h] -
                  input[((v-1)*(width))+h-1];
                break;
              case 5:
                px = input[(v*(width))+h-1] +
                  ((input[((v-1)*(width))+h] -
                    input[((v-1)*(width))+h-1])>>1);
                break;
              case 6:
                px = input[((v-1)*(width))+h] +
                  ((input[(v*(width))+h-1] -
                    input[((v-1)*(width))+h-1])>>1);
                break;
              case 7:
                px = (input[((v-1)*(width))+h] +
                  input[(v*(width))+h-1])>>1;
                break;
              default:
                WHEREAMI();
                printf("Lossless mode: %d not supported.\n",
                  UseType);
                break;
                }
              value=input[(v*(width))+h]-px;
              if (Loud > NOISY)
                printf("IN=%d  PX=%d  FRE: %d\n",
                  input[(v*(width))+h],px,value);
              LosslessFrequencyDC(value);
              }
            }
          }
        }
      CurrentMDU++;
      if (StartofLine)
        {
        UseType=CScan->SSS;
        StartofLine=0;
        }
      }
    }
  for(j=0;j<CScan->NumberComponents;j++)          /* Rewind to start */
    {
      InstallIob(j);
      RewindIob();
    }
}

/*BFUNC

JpegEncodeLosslessScan() encodes the scan that is given to it by lossless
techniques. The Huffman table should already be specified.

EFUNC*/

void JpegLosslessEncodeScan()
{
  BEGIN("JpegEncodeLosslessScan")
  int x,y,j,h,v,px;
  int height,width,horfreq,value;
  int MaxElem,CurrentElem,NumberElem;
  int StartofLine=1,UseType=1;              /* Start with type 1 coding */
  int *input;

  CheckScan();
  for(j=0;j<CScan->NumberComponents;j++)    /* Important to rewind to start */
    {                                       /* for lossless coding... */
      InstallIob(j);
      RewindIob();
    }
  if (CScan->NumberComponents==1)       /* Calculate maximum number of */
    MaxElem= LOSSLESSBUFFERSIZE/4;      /* elements can be loaded in */
  else
    {
    MaxElem= LOSSLESSBUFFERSIZE/
      ((CFrame->vf[CScan->ci[0]]+1)*(CFrame->hf[CScan->ci[0]]+1));
    for(j=1;j<CScan->NumberComponents;j++)          /* Rewind to start */
      {
      x=LOSSLESSBUFFERSIZE/
        ((CFrame->vf[CScan->ci[j]]+1)*(CFrame->hf[CScan->ci[j]]+1));
      if (x < MaxElem) MaxElem=x;
      }
    }
  CScan->SSS=LosslessPredictorType;
  CScan->SAL=PointTransform;
  InstallIob(0);
  UseDCHuffman(CScan->td[0]);          /* Install DC table */
  if (CScan->NumberComponents==1)
    height=horfreq=1;
  else
    {
      height=CFrame->vf[CScan->ci[0]];
      horfreq=CFrame->hf[CScan->ci[0]];
    }
  NumberMDU = CScan->MDUWide*CScan->MDUHigh;
  ResyncCount=0;                /* Reset the resync counter for every scan */
  if (CFrame->InsertDnl>0) /* If DNL is greater than 0, insert */
    {                           /* into according Resync interval */
    if  (!(CFrame->ResyncInterval))
      WriteDnl();  /* Automatically write a dnl if no resync is enabled.*/
    else                      /* If DNL > MDU, then put in last resync */
      CFrame->InsertDnl = MAX(CFrame->InsertDnl,         /* interval */
        NumberMDU/CFrame->ResyncInterval);
    }
  WriteSos();                  /* Start of Scan */
  CurrentMDU=0;
  if ((CFrame->ResyncInterval)&&(CFrame->ResyncInterval % CScan->MDUWide))
    {
    WHEREAMI();
    printf("Resync Interval not an integer multiple of MDU's wide.\n");
    printf("Proceeding anyways.\n");
    if (MaxElem>=CFrame->ResyncInterval)
      MaxElem=CFrame->ResyncInterval;       /* Reduce to resync interval */
    else
      MaxElem=1;                            /* Can't proceed quickly */
    }
  CurrentElem=NumberElem=0;
  for(y=0;y<CScan->MDUHigh;y++)
    {
    for(x=0;x<CScan->MDUWide;x++)
      {
      if (CurrentMDU && (CFrame->ResyncInterval))
        {
        if (!(CurrentMDU % CFrame->ResyncInterval)) /* Check resync */
          {
          if ((CurrentMDU/CFrame->ResyncInterval)==CFrame->InsertDnl)
            {
            WriteDnl();              /* If resync matches use DNL */
            CFrame->InsertDnl=0;     /* Mission accomplished. */
            }
          WriteResync();                 /* Write resync */
          UseType=1;                     /* Reset codec */
          for(j=0;j<CScan->NumberComponents;j++)
            {
            InstallIob(j);
            LineResetBuffers();
            }
          }
        }
      if (!(CurrentMDU%CScan->MDUWide)&&(CurrentMDU)) /* Reset CScan type */
        {
        UseType=2;                           /* Start of line */
        StartofLine=1;                       /* uses top pel predictor */
        }
      CurrentElem++;
      if (CurrentElem>=NumberElem)
        {
        NumberElem = MIN((CScan->MDUWide-x),MaxElem);
        CurrentElem=0;
        for(j=0;j<CScan->NumberComponents;j++)
          {
          InstallIob(j);                    /* Install component j */
          ReadLine(NumberElem,              /* Read in some elements*/
            CScan->LosslessBuffer[j]);
          }
        }
      if (CScan->NumberComponents==1)
        {
        if (Loud > MUTE)
          {
          WHEREAMI();
          printf("[Pass 2 [Component:MDU:Total] [%d:%d:%d]]\n",
            0,CurrentMDU,NumberMDU);
          }
        input = &CScan->LosslessBuffer[0][CurrentElem];
        width=horfreq*NumberElem+1;
        switch(UseType) /* Same as lossless coding predictor*/
          {
        case 1:
          px = input[width];
          break;
        case 2:
          px = input[1];
          break;
        case 3:
          px = input[0];
          break;
        case 4:
          px = input[width] + input[1] - input[0];
          break;
        case 5:
          px = input[width] + ((input[1] - input[0])>>1);
          break;
        case 6:
          px = input[1] + ((input[width] - input[0])>>1);
          break;
        case 7:
          px = (input[1] + input[width])>>1;  /* No rounding */
          break;
        default:
          WHEREAMI();
          printf("Lossless mode %d not supported.\n",UseType);
          break;
          }
        value=input[width+1]-px;
        if (Loud > MUTE)
          printf("IN=%d  PX=%d  FRE: %d\n",
            input[width+1],px,value);
        LosslessEncodeDC(value);
        }
      else
        {
        for(j=0;j<CScan->NumberComponents;j++)
          {
          if (Loud > MUTE)
            {
            WHEREAMI();
            printf("[Pass 2 [Component:MDU] [%d:%d]]\n",
              j,CurrentMDU);
            }
          height=CFrame->vf[CScan->ci[j]];
          horfreq=CFrame->hf[CScan->ci[j]];
          width=horfreq*NumberElem+1;
          input = &CScan->LosslessBuffer[j][CurrentElem*horfreq];
          UseDCHuffman(CScan->td[j]);
          for(v=1;v<=height;v++)
            {
            for(h=1;h<=horfreq;h++)
              {
              switch(UseType)   /* Same as lossless predictor*/
                {
              case 1:
                px = input[(v*(width))+h-1];
                break;
              case 2:
                px = input[((v-1)*(width))+h];
                break;
              case 3:
                px = input[((v-1)*(width))+h-1];
                break;
              case 4:
                px = input[(v*(width))+h-1] +
                  input[((v-1)*(width))+h] -
                  input[((v-1)*(width))+h-1];
                break;
              case 5:
                px = input[(v*(width))+h-1] +
                  ((input[((v-1)*(width))+h] -
                    input[((v-1)*(width))+h-1])>>1);
                break;
              case 6:
                px = input[((v-1)*(width))+h] +
                  ((input[(v*(width))+h-1] -
                    input[((v-1)*(width))+h-1])>>1);
                break;
              case 7:
                px = (input[((v-1)*(width))+h] +
                  input[(v*(width))+h-1])>>1;
                break;
              default:
                WHEREAMI();
                printf("Lossless mode %d not supported.\n",
                  UseType);
                break;
                }
              value=input[(v*(width))+h]-px;
              if (Loud > MUTE)
                {
                printf("IN=%d  PX=%d  ENC: %d\n",
                  input[(v*(width))+h],px,value);
                }
              LosslessEncodeDC(value); /* Encode as DC component */
              }
            }
          }
        }
      CurrentMDU++;
      if (StartofLine)
        {
        UseType=CScan->SSS;
        StartofLine=0;
        }
      }
    }
  if (CFrame->InsertDnl==-2)    /* -2 is automatic DNL insertion */
    {
    WriteDnl();
    CFrame->InsertDnl=0;
    }

  for(j=0;j<CScan->NumberComponents;j++)          /* Rewind to start */
    {
    InstallIob(j);
    RewindIob();
    }
}

/*BFUNC

JpegDecodeFrame(general,)
     ) is used to decode a file. In general; is used to decode a file. In general, CFrame should
hold just enough information to set up the file structure; that is,
which file is to be opened for what component.

EFUNC*/

static void JpegDecodeFrame()
{
  BEGIN("JpegDecodeFrame")
  int i;

  sropen(CImage->StreamFileName,0);   /* Zero index */
  if (ScreenAllMarker() < 0)          /* Do all markers pending */
    {
      WHEREAMI();
      printf("No initial marker found!\n");
      exit(-1);
    }
  while(1)
    {
    if (NumberMDU>=0)               /* If NumberMDU is positive proceed */
      {
      if (CurrentMDU >= NumberMDU) /* If all decoded */
        {
        if (Notify)             /* Print statistics */
          {
          printf("> GW:%d  GH:%d  R:%d\n",
            CFrame->GlobalWidth,
            CFrame->GlobalHeight,
            CFrame->ResyncInterval);
          }
        for(i=0;i<CScan->NumberComponents;i++)  /* Print Scan info */
          {
          if (Notify)
            {
            printf(">> C:%d  N:%s  W:%d  H:%d  hf:%d  vf:%d\n",
              CScan->ci[i],
              CFrame->ComponentFileName[CScan->ci[i]],
              CFrame->Width[CScan->ci[i]],
              CFrame->Height[CScan->ci[i]],
              CFrame->hf[CScan->ci[i]],
              CFrame->vf[CScan->ci[i]]);
            }
          InstallIob(i);
          FlushIob();                        /* Close image files */
          SeekEndIob();
          CloseIob();
          }
        CurrentMDU=0;
        if (ScreenAllMarker()<0)            /* See if any more images*/
          {
          WHEREAMI();
          printf("No trailing marker found!\n");
          exit(-1);
          }
        if ((EndofFile)||(EndofImage))      /* Nothing, then return */
          {
          srclose();
          break;
          }
        }
      }
    if (CFrame->Type==3)
      JpegLosslessDecodeScan();
    else
      JpegDecodeScan();
    }
}

/*BFUNC

JpegLosslessDecodeScan() is used to losslessly decode a portion of the
image called the scan.  This routine uses the internal lossless
buffers to reduce the overhead in writing.  However, one must note
that the overhead is mostly in the Huffman decoding.

EFUNC*/

static void JpegLosslessDecodeScan()
{
  BEGIN("JpegLosslessDecodeScan")
  int j,v,h,value,px;
  int height,horfreq,width;
  int MaxElem,CurrentElem,NumberElem;
  int StartofLine=1,UseType=1;              /* Start with type 1 coding */
  int *input;

  PointTransform=CScan->SAL;
  for(j=0;j<CScan->NumberComponents;j++)    /* Important to rewind to start */
    {                                       /* for lossless coding... */
      InstallIob(j);
      RewindIob();
    }
  if (CScan->NumberComponents==1)       /* Calculate maximum number of */
    MaxElem= LOSSLESSBUFFERSIZE/4;      /* elements can be loaded in */
  else
    {
    MaxElem= LOSSLESSBUFFERSIZE/
      ((CFrame->vf[CScan->ci[0]]+1)*(CFrame->hf[CScan->ci[0]]+1));
    for(j=1;j<CScan->NumberComponents;j++)          /* Rewind to start */
      {
      v=LOSSLESSBUFFERSIZE/
        ((CFrame->vf[CScan->ci[j]]+1)*(CFrame->hf[CScan->ci[j]]+1));
      if (v < MaxElem) MaxElem=v;
      }
    }
  InstallIob(0);
  UseDCHuffman(CScan->td[0]);          /* Install DC table */
  if (CScan->NumberComponents==1)
    height=horfreq=1;
  else
    {
      height=CFrame->vf[CScan->ci[0]];
      horfreq=CFrame->hf[CScan->ci[0]];
    }
  if ((CFrame->ResyncInterval)&&(CFrame->ResyncInterval % CScan->MDUWide))
    {
      WHEREAMI();
      printf("Resync Interval not an integer multiple of MDU's wide.\n");
      printf("Proceeding anyways.\n");
      if (MaxElem>=CFrame->ResyncInterval)
        MaxElem=CFrame->ResyncInterval;       /* Reduce to resync interval */
      else
        MaxElem=1;                            /* Can't proceed quickly */
    }
  CurrentElem=NumberElem=0;
  while(1)
    {
    if ((NumberMDU<0)&&(!(CurrentMDU%CScan->MDUWide)))
      {
      if (CheckMarker()==0xdc)
        ScreenMarker();
      }
    if (NumberMDU>=0)               /* If NumberMDU is positive proceed */
      {
      if (CurrentMDU >= NumberMDU) /* If all decoded */
        return;
      }

    if (CFrame->ResyncInterval)                /* Flag to decoder stream */
      ResyncEnable = 1;
    if (CurrentMDU && (CFrame->ResyncInterval))
      {                                    /* If resync interval */
      if ((CurrentMDU % CFrame->ResyncInterval)==0)
        {
        if (!CleartoResync)               /* If not in error recovery*/
          ReadResync();                          /* read resync. */
        if (CleartoResync)
          {
          /*
          Clear until we have LastKnownResync:
          the offset is by 1 because we add the resync i%8
          _after_ we code the ith resync interval...
           */
          if (((CurrentMDU/CFrame->ResyncInterval)&0x07)==
            ((LastKnownResync+1)&0x07))
            CleartoResync = 0;   /* Finished with resync clearing */
          }
        UseType=1;                             /* Reset codec */
        for(j=0;j<CScan->NumberComponents;j++) /* reset line buffers */
          {                                    /* Type is previous pel */
          InstallIob(j);
          LineResetBuffers();
          }
        }
      }
    if (!(CurrentMDU%CScan->MDUWide)&&(CurrentMDU))  /* Reset CScan type */
      {
      UseType=2;                            /* Start of line */
      StartofLine=1;                        /* uses top pel predictor */
      }

    if (CurrentElem>=NumberElem)
      {
      NumberElem = MIN((CScan->MDUWide-(CurrentMDU%CScan->MDUWide)),
        MaxElem);
      CurrentElem=0;
      for(j=0;j<CScan->NumberComponents;j++)
        {
        InstallIob(j);                    /* Install component j */
        ReadPreambleLine(NumberElem,     /* Read in some elements*/
          CScan->LosslessBuffer[j]);
        }
      }
    if (CScan->NumberComponents==1)
      {
      width=horfreq*NumberElem+1;
      input = &CScan->LosslessBuffer[0][CurrentElem];
      switch(UseType) /* Same as lossless coding predictor*/
        {
      case 1:
        px = input[width];
        break;
      case 2:
        px = input[1];
        break;
      case 3:
        px = input[0];
        break;
      case 4:
        px = input[width] + input[1] - input[0];
        break;
      case 5:
        px = input[width] + ((input[1] - input[0])>>1);
        break;
      case 6:
        px = input[1] + ((input[width] - input[0])>>1);
        break;
      case 7:
        px = (input[1] + input[width])>>1;  /* No rounding */
        break;
      default:
        WHEREAMI();
        printf("Lossless mode %d not supported.\n",UseType);
        break;
        }
      if (CleartoResync)         /* If CleartoResync, flush */
        input[width+1] = 0;
      else
        {
        value = LosslessDecodeDC();
        input[width+1] = (value+px)&0xffff;
        if (Loud > MUTE)
          {
          printf("OUT=%d  PX=%d  VAL: %d\n",
            input[width+1],px,value);
          }
        }
      }
    else
      {
      for(j=0;j<CScan->NumberComponents;j++)   /* Decode MDU */
        {
        if (Loud > MUTE)
          {
          WHEREAMI();
          printf("[Decoder Pass [Component:MDU:#MDU] [%d:%d:%d]]\n",
            j,CurrentMDU,NumberMDU);
          }
        InstallIob(j);                     /* Install component */
        height=CFrame->vf[CScan->ci[j]];
        horfreq=CFrame->hf[CScan->ci[j]];
        width=horfreq*NumberElem+1;
        input = &CScan->LosslessBuffer[j][CurrentElem*horfreq];
        UseDCHuffman(CScan->td[j]);          /* Install DC table */
        for(v=1;v<=height;v++)
          {
          for(h=1;h<=horfreq;h++)
            {
            switch(UseType) /* Same as lossless coding predictor*/
              {
            case 1:
              px = input[(v*(width))+h-1];
              break;
            case 2:
              px = input[((v-1)*(width))+h];
              break;
            case 3:
              px = input[((v-1)*(width))+h-1];
              break;
            case 4:
              px = input[(v*(width))+h-1] +
                input[((v-1)*(width))+h] -
                input[((v-1)*(width))+h-1];
              break;
            case 5:
              px = input[(v*(width))+h-1] +
                ((input[((v-1)*(width))+h] -
                  input[((v-1)*(width))+h-1])>>1);
              break;
            case 6:
              px = input[((v-1)*(width))+h] +
                ((input[(v*(width))+h-1] -
                  input[((v-1)*(width))+h-1])>>1);
              break;
            case 7:
              px = (input[((v-1)*(width))+h] +
                input[(v*(width))+h-1])>>1;
              break;
            default:
              WHEREAMI();
              printf("Lossless mode %d not supported.\n",
                UseType);
              break;
              }
            if (CleartoResync)         /* If CleartoResync, flush */
              input[(v*(width))+h] = 0;
            else
              {
              value = LosslessDecodeDC();
              input[(v*(width))+h] = (value+px)&0xffff;
              if (Loud > MUTE)
                {
                printf("OUT=%d  PX=%d  VAL: %d\n",
                  input[(v*(width))+h],px,value);
                }
              }
            }
          }
        }
      }
    CurrentElem++;
    if (CurrentElem>=NumberElem)
      {
      for(j=0;j<CScan->NumberComponents;j++)
        {
        InstallIob(j);                    /* Install component j */
        WriteLine(NumberElem,             /* Write out elements*/
          CScan->LosslessBuffer[j]);
        }
      }
    CurrentMDU++;
    if (StartofLine)
      {
      UseType=CScan->SSS;
      StartofLine=0;
      }
    }
}

/*BFUNC

JpegDecodeScan() is used to decode a portion of the image called the
scan.  Everything  is read upon getting to this stage.

EFUNC*/

static void JpegDecodeScan()
{
  BEGIN("JpegDecodeScan")
  int j,v,h,dovf,dohf;
  int input[64],output[64];
  int IDCTBound,IDCTShift;

  while(1)
    {
    if ((NumberMDU<0)&&(!(CurrentMDU%CScan->MDUWide)))
      {
      if (CheckMarker()==0xdc)
        ScreenMarker();
      }
    if (NumberMDU>=0)               /* If NumberMDU is positive proceed */
      {
      if (CurrentMDU >= NumberMDU) /* If all decoded */
        return;
      }
    if (CFrame->ResyncInterval)                /* Flag to decoder stream */
      {
      ResyncEnable = 1;
      }
    if (CurrentMDU && (CFrame->ResyncInterval))
      {                                    /* If resync interval */
      if ((CurrentMDU % CFrame->ResyncInterval)==0)
        {
        if (!CleartoResync)               /* If not in error recovery*/
          {                               /* read resync. */
          ReadResync();
          }
        if (CleartoResync)
          {
          /*
          Clear until we have LastKnownResync:
          the offset is by 1 because we add the resync i%8
          _after_ we code the ith resync interval...
           */
          if (((CurrentMDU/CFrame->ResyncInterval)&0x07)==
            ((LastKnownResync+1)&0x07))
            {
            CleartoResync = 0;   /* Finished with resync clearing */
            }
          }
        ResetCodec();                /* Reset codec */
        }
      }
    IDCTBound=((CFrame->DataPrecision>8)?4095:255);
    IDCTShift=((CFrame->DataPrecision>8)?2048:128);
    for(j=0;j<CScan->NumberComponents;j++)   /* Decode MDU */
      {
      if (Loud > MUTE)
        {
        WHEREAMI();
        printf("[Decoder Pass [Component:MDU:#MDU] [%d:%d:%d]]\n",
          j,CurrentMDU,NumberMDU);
        }
      InstallPrediction(j);             /* Install component */
      InstallIob(j);
      if (CScan->NumberComponents==1) /* Check for non-interleaved mode */
        dohf=dovf=1;
      else
        {
        dohf = CFrame->hf[CScan->ci[j]];
        dovf = CFrame->vf[CScan->ci[j]];
        }
      for(v=0;v<dovf;v++) /* Do for blocks in MDU*/
        {
        for(h=0;h<dohf;h++)
          {
          if (CleartoResync)             /* CleartoResync, flush */
            ClearMatrix(input);
          else
            {
            UseDCHuffman(CScan->td[j]);  /* Install DC table */
            *input = DecodeDC();         /* Decode DC */
            UseACHuffman(CScan->ta[j]);  /* Install AC table */
            DecodeAC(input);             /* Decode AC */
            if (Loud > TALK)
              {
              printf("Cooked Input\n");
              PrintMatrix(input);
              }
            IZigzagMatrix(input,output);   /* Inverse zigzag */
            IQuantize(output,              /* Inverse quantize */
              CImage->
              QuantizationMatrices[CFrame->
              tq[CScan->ci[j]]]);
            DefaultIDct(output,input);     /* Inverse DCT */
            PostshiftIDctMatrix(input,IDCTShift);
            /* Shift (all positive)*/
            BoundIDctMatrix(input,IDCTBound); /* Bound */
            if (Loud > WHISPER)
              {
              printf("Raw Output\n");
              PrintMatrix(input);
              }
            }
          WriteBlock(input);                 /* Write out */
          }
        }
      }
    CurrentMDU++;
    }
}

/*BFUNC

PrintImage() prints out the Image structure of the CURRENT image.  It
is primarily used for debugging. The image structure consists of the
data that is held to be fixed even though multiple scans (or multiple
frames, even though it is not advertised as such by JPEG) are
received.

EFUNC*/

void PrintImage()
{
  BEGIN("PrintImage")
  int i;

  printf("*** Image ID: %p ***\n",(void*)CImage); /* %p should work ... */
  if (CImage)
    {
    if (CImage->StreamFileName)
      {
      printf("StreamFileName %s\n",(CImage->StreamFileName ?
          CImage->StreamFileName :
          "Null"));
      }
    printf("InternalMode: %d   ImageSequence: %d\n",
      CImage->JpegMode,CImage->ImageSequence);
    printf("NumberQuantizationMatrices %d\n",
      CImage->NumberQuantizationMatrices);
    for(i=0;i<CImage->NumberQuantizationMatrices;i++)
      {
      printf("Quantization Matrix [%d]\n",i);
      PrintMatrix(CImage->QuantizationMatrices[i]);
      }
    printf("NumberDCTables %d\n",
      CImage->NumberDCTables);
    for(i=0;i<CImage->NumberDCTables;i++)
      {
      printf("DC Huffman Table[%d]\n",i);
      UseDCHuffman(i);
      PrintHuffman();
      }
    printf("NumberACTables %d\n",
      CImage->NumberACTables);
    for(i=0;i<CImage->NumberACTables;i++)
      {
      printf("AC Huffman Table[%d]\n",i);
      UseACHuffman(i);
      PrintHuffman();
      }
    }
}

/*BFUNC

PrintFrame() is used to print the information specific to loading in
the frame. This corresponds roughly to the information received by the
SOF marker code.

EFUNC*/

void PrintFrame()
{
  BEGIN("PrintFrame")
  int i;

  printf("*** Frame ID: %p *** (TYPE: %d)\n",(void*)CFrame,CFrame->Type);
  if (CFrame)
    {
    printf("DataPrecision: %d  ResyncInterval: %d\n",
      CFrame->DataPrecision,CFrame->ResyncInterval);
    printf("Height: %d   Width: %d\n",
      CFrame->GlobalHeight,CFrame->GlobalWidth);
    printf("BufferSize: %d  Image: %p\n",CFrame->BufferSize,(void*)CFrame->Image);
    printf("NumberComponents %d\n",
      CFrame->GlobalNumberComponents);
    for(i=0;i<CFrame->GlobalNumberComponents;i++)
      {
      printf("ComponentFileName %s\n",
        ((CFrame->ComponentFileName[CFrame->cn[i]]) ?
         CFrame->ComponentFileName[CFrame->cn[i]] : "Null"));
      printf("HorizontalFrequency: %d  VerticalFrequency: %d\n",
        CFrame->hf[CFrame->cn[i]],CFrame->vf[CFrame->cn[i]]);
      printf("Height: %d  Width: %d\n",
        CFrame->Height[CFrame->cn[i]],CFrame->Width[CFrame->cn[i]]);
      InstallIob(i);
      PrintIob();
      }
    }
}

/*BFUNC

PrintScan() is used to print the information in the CScan structure.
This roughly corresponds to the information received by the Scan
marker code.

EFUNC*/

void PrintScan()
{
  BEGIN("PrintScan")
  int i;

  printf("*** Scan ID: %p ***\n",(void*)CScan);
  if (CScan)
    {
    printf("NumberComponents %d\n",
      CScan->NumberComponents);
    for(i=0;i<CScan->NumberComponents;i++)
      {
      printf("Component: %d  Index: %d\n",
        i,CScan->ci[i]);
      printf("DC Huffman Table: %d  AC Huffman Table: %d\n",
        CScan->td[i],CScan->ta[i]);
      printf("LastDC: %d  Iob: %p\n",
        *(CScan->LastDC[i]),(void*)CScan->Iob[i]);
      }
    printf("NumberACSend: %d  NumberDCSend: %d  NumberQSend: %d\n",
      CScan->NumberACTablesSend,
      CScan->NumberDCTablesSend,
      CScan->NumberQTablesSend);
    }
}

/*BFUNC

MakeImage() makes an image and puts it into the Current Image pointer
(CImage). It initializes the structure appropriate to the JPEG initial
specifications.

EFUNC*/

void MakeImage()
{
  BEGIN("MakeImage")

  if (!(CImage = MakeStructure(IMAGE)))
    {
      WHEREAMI();
      printf("Cannot allocate memory for Image structure.\n");
      exit(ERROR_MEMORY);
    }
  CImage->StreamFileName = NULL;
  CImage->JpegMode = 0;
  CImage->Jfif=0;
  CImage->ImageSequence = -1;        /* First element in sequence is 0 */
  CImage->NumberQuantizationMatrices = 2;  /* Default # matrices is 2 */
  CImage->QuantizationMatrices[0] = LuminanceQuantization;
  CImage->QuantizationMatrices[1] = ChrominanceQuantization;
  CImage->NumberACTables = 0;       /* No tables defined yet */
  CImage->NumberDCTables = 0;
}

/*BFUNC

MakeFrame() constructs a Frame Structure and puts it in the Current
Frame pointer (CFrame).

EFUNC*/

void MakeFrame()
{
  BEGIN("MakeFrame")
  int i;

  if (!(CFrame = MakeStructure(FRAME)))
    {
      WHEREAMI();
      printf("Cannot allocate memory for Frame structure.\n");
      exit(ERROR_MEMORY);
    }
  CFrame->Type=0;                   /* Baseline type */
  CFrame->InsertDnl = 0;            /* Set to default position */
  CFrame->Q = 0;
  CFrame->tmpfile = 0;
  CFrame->GlobalHeight = 0;
  CFrame->GlobalWidth = 0;
  CFrame->DataPrecision = 8;         /* Default 8 precision */
  CFrame->ResyncInterval = 0;
  CFrame->GlobalNumberComponents = 0;
  for(i=0;i<MAXIMUM_COMPONENTS;i++)
    {
      CFrame->cn[i] = 0;           /* Clean out all slots */
      CFrame->hf[i] = 0;
      CFrame->vf[i] = 0;
      CFrame->tq[i] = 0;
      CFrame->Height[i] = 0;
      CFrame->Width[i] = 0;
      CFrame->ComponentFileName[i] = 0;
    }
  CFrame->BufferSize = BUFFERSIZE;
  CFrame->Image = CImage;
}

/*BFUNC

MakeScanFrequency() constructs a set of scan information for the
current variables. These frequency markers are used for creating the
JPEG custom matrices.

EFUNC*/

void MakeScanFrequency()
{
  BEGIN("MakeScanFrequency")
  int i;

  for(i=0;i<MAXIMUM_SOURCES;i++)
    {
    if (!(CScan->LastDC[i] = MakeStructure(int)))
      {
      WHEREAMI();
      printf("Cannot allocate LastDC integer store.\n");
      exit(ERROR_MEMORY);
      }
    if (!(CScan->ACFrequency[i] = (int *) calloc(257,sizeof(int))))
      {
      WHEREAMI();
      printf("Cannot allocate AC Frequency array.\n");
      exit(ERROR_MEMORY);
      }
    if (!(CScan->DCFrequency[i] = (int *) calloc(257,sizeof(int))))
      {
      WHEREAMI();
      printf("Cannot allocate DC Frequency array.\n");
      exit(ERROR_MEMORY);
      }
    }
}

/*BFUNC

MakeScan() is used for creating the Scan structure which holds most of
the information in the Scan marker code.

EFUNC*/

void MakeScan()
{
  BEGIN("MakeScan")
  int i;

  if (!(CScan = MakeStructure(SCAN)))
    {
      WHEREAMI();
      printf("Cannot allocate memory for Scan structure.\n");
      exit(ERROR_MEMORY);
    }
  CScan->NumberACTablesSend = 0;    /* Install with default values */
  CScan->NumberDCTablesSend = 0;
  CScan->NumberComponents = 0;
  for(i=0;i<MAXIMUM_SOURCES;i++)
    {
      CScan->ci[i] = 0;
      CScan->ta[i] = 0;
      CScan->td[i] = 0;
      CScan->sa[i] = 0;
      CScan->sd[i] = 0;
      CScan->sq[i] = 0;
    }
  CScan->SSS=0;
  CScan->SSE=0;
  CScan->SAH=0;
  CScan->SAL=0;
  MakeScanFrequency();
}

/*BFUNC

MakeConsistentFileNames() is used to construct consistent filenames
for opening and closing of data storage. It is used primarily by the
decoder when all the files may not necessarily be specified.

EFUNC*/

void MakeConsistentFileNames()
{
  BEGIN("MakeConsistentFileNames")
  int i;

  for(i=0;i<CScan->NumberComponents;i++)
    {
    if (CImage->ImageSequence)  /* If in sequence, must add sequence */
      {                         /* identifier */
      CFrame->ComponentFileName[CScan->ci[i]] =
        (char *) calloc(strlen(CImage->StreamFileName)+16,sizeof(char));
      sprintf(CFrame->ComponentFileName[CScan->ci[i]],"%s.%d.%d",
        CImage->StreamFileName,CImage->ImageSequence,CScan->ci[i]);
      }
    else if (CFrame->ComponentFileName[CScan->ci[i]] == NULL)
      {                        /* Otherwise if none specified, create. */
      CFrame->ComponentFileName[CScan->ci[i]] =
        (char *) calloc(strlen(CImage->StreamFileName)+8,sizeof(char));
      sprintf(CFrame->ComponentFileName[CScan->ci[i]],"%s.%d",
        CImage->StreamFileName,CScan->ci[i]);
      }
    }
}

/*BFUNC

CheckValidity() checks whether the current values in CFrame and CScan
meet the internal specifications for correctness and the algorithm
can guarantee completion.

EFUNC*/

void CheckValidity()
{
  BEGIN("CheckValidity")
  int i;

  ErrorValue = 0;           /* Check if within internal specs */
  InBounds(CFrame->GlobalWidth,0,MAXIMUM_IMAGE_WIDTH,"Bad Image Width");
  InBounds(CFrame->GlobalHeight,0,MAXIMUM_IMAGE_HEIGHT,"Bad Image Height");
  if (CFrame->Q<0)
    {
      WHEREAMI();
      printf("Q factor is negative - must be positive\n");
    }
  if ((CFrame->DataPrecision!=8)&&(CFrame->DataPrecision!=12))
    {
    if (CImage->JpegMode == J_LOSSLESS)
      {
      if (CFrame->DataPrecision<=16)
        printf("Precision type: %d\n",CFrame->DataPrecision);
      else
        printf("Caution: precision type: %d greater than 16.\n",
          CFrame->DataPrecision);
      }
    else
      printf("Caution: precision type: %d not 8 or 12.\n",
        CFrame->DataPrecision);
    }
  InBounds(CScan->NumberComponents,1,15,"Bad Number of Components");
  for(i=0;i<CScan->NumberComponents;i++)
    {
    InBounds(CFrame->Width[CScan->ci[i]],0,MAXIMUM_IMAGE_WIDTH,
      "Bad Frame Width");
    InBounds(CFrame->Height[CScan->ci[i]],0,MAXIMUM_IMAGE_HEIGHT,
      "Bad Frame Height");
    InBounds(CFrame->hf[CScan->ci[i]],1,MAXIMUM_HORIZONTAL_FREQUENCY,
      "Bad Horizontal Frequency");
    InBounds(CFrame->vf[CScan->ci[i]],1,MAXIMUM_VERTICAL_FREQUENCY,
      "Bad Vertical Frequency");
    }
  InBounds(LosslessPredictorType,0,7,"Bad Lossless Predictor Type");
  if (PointTransform)
    {
    if (!(LosslessPredictorType))
      {
      WHEREAMI();
      printf("Point Transform specified without lossless prediction.\n");
      printf("Shifting of input/output should be anticipated.\n");
      }
    else
      InBounds(PointTransform,0,14,"Bad Point Transform");
    }
  if (ErrorValue)
    {
    WHEREAMI();
    printf("Invalid input detected.\n");
    exit(ErrorValue);
    }
}

/*BFUNC

CheckBaseline() checks whether the internal values meet JPEG Baseline
specifications.

EFUNC*/

int CheckBaseline()
{
  BEGIN("CheckBaseline")
  int i;

  ErrorValue = 0;         /* Check for JPEG specs */
  InBounds(CFrame->GlobalWidth,0,MAXIMUM_IMAGE_WIDTH,"Bad Image Width");
  InBounds(CFrame->GlobalHeight,0,MAXIMUM_IMAGE_HEIGHT,"Bad Image Height");
  if (CFrame->Q<0)
    {
      WHEREAMI();
      printf("Q factor is negative - must be positive\n");
    }
  InBounds(CScan->NumberComponents,1,4,"Bad Number of Components");
  for(i=0;i<CScan->NumberComponents;i++)
    {
    InBounds(CFrame->Width[CScan->ci[i]],0,MAXIMUM_IMAGE_WIDTH,
      "Bad Frame Width");
    InBounds(CFrame->Height[CScan->ci[i]],0,MAXIMUM_IMAGE_HEIGHT,
      "Bad Frame Height");
    InBounds(CFrame->hf[CScan->ci[i]],1,MAXIMUM_JPEG_HORIZONTAL_FREQUENCY,
      "Bad Horizontal Frequency");
    InBounds(CFrame->vf[CScan->ci[i]],1,MAXIMUM_JPEG_VERTICAL_FREQUENCY,
      "Bad Vertical Frequency");
    }
  if (ErrorValue)
    {
      printf("Caution: JPEG++ Mode.\n");
      ErrorValue = 0;
    }
  return 0;
}

/*BFUNC

ConfirmFileSize() checks to see if the files used in the scan actually
exist and correspond in size to the input given.

EFUNC*/

void ConfirmFileSize()
{
  BEGIN("ConfirmFileSize")
  int i,FileSize;
  FILE *test;

  for(i=0;i<CScan->NumberComponents;i++)  /* Do for all components in scan*/
    {
    if (CFrame->ComponentFileName[CScan->ci[i]])
      {
      if ((test = fopen(CFrame->ComponentFileName[CScan->ci[i]],
            "rb")) == NULL)
        {
        WHEREAMI();
        printf("Cannot open filename %s\n",
          CFrame->ComponentFileName[CScan->ci[i]]);
        exit(ERROR_BOUNDS);
        }
      fseek(test,0,2);                /* Go to end */
      FileSize = ftell(test);         /* Find number of bytes */
      rewind(test);
      if (CFrame->Height[CScan->ci[i]] == 0)  /* Must have good dimens*/
        {
        if (CFrame->Width[CScan->ci[i]] == 0)
          {
          WHEREAMI();
          printf("Bad file specification in %s.\n",
            CFrame->ComponentFileName[CScan->ci[i]]);
          }
        else
          {
          CFrame->Height[CScan->ci[i]] = FileSize /
            (CFrame->Width[CScan->ci[i]]*
             ((CFrame->DataPrecision>8)?2:1));
          WHEREAMI();
          printf("Autosizing height to %d\n",
            CFrame->Height[CScan->ci[i]]);
          }
        }                                 /* Dimensions must conform */
      if (FileSize !=
        CFrame->Width[CScan->ci[i]] * CFrame->Height[CScan->ci[i]]*
        ((CFrame->DataPrecision>8)?2:1))
        {
        WHEREAMI();
        printf("File size conflict in %s, est: %d  act: %d \n",
          CFrame->ComponentFileName[CScan->ci[i]],
          CFrame->Width[CScan->ci[i]]*CFrame->Height[CScan->ci[i]]*
          ((CFrame->DataPrecision>8)?2:1),
          FileSize);
        exit(ERROR_BOUNDS);
        }
      fclose(test);
      }
    }
}

/*BFUNC

Help() prints out general information regarding the use of this
JPEG software.

EFUNC*/

static void Help()
{
  BEGIN("Help")

  printf("jpeg -iw ImageWidth -ih ImageHeight [-JFIF] [-q(l) Q-Factor]\n");
  printf("     [-a] [-b] [-d] [-k predictortype] [-n] [-o] [-y] [-z]\n");
  printf("     [-p PrecisionValue] [-t pointtransform]\n");
  printf("     [-r ResyncInterval] [-s StreamName]\n");
  printf("     [[-ci ComponentIndex1] [-fw FrameWidth1] [-fh FrameHeight1]\n");
  printf("      [-hf HorizontalFrequency1] [-vf VerticalFrequency1]\n");
  printf("      ComponentFile1]\n");
  printf("     [[-ci ComponentIndex2] [-fw FrameWidth2] [-fh FrameHeight2]\n");
  printf("      [-hf HorizontalFrequency2] [-vf VerticalFrequency2]\n");
  printf("      ComponentFile1]\n");
  printf("     ....\n\n");
  printf("-JFIF puts a JFIF marker. Don't change component indices.\n");
  printf("-a enables Reference DCT.\n");
  printf("-b enables Lee DCT.\n");
  printf("-d decoder enable.\n");
  printf("-[k predictortype] enables lossless mode.\n");
  printf("-q specifies quantization factor; -ql specifies can be long.\n");
  printf("-n enables non-interleaved mode.\n");
  printf("-[t pointtransform] is the number of bits for the PT shift.\n");
  printf("-o enables the Command Interpreter.\n");
  printf("-p specifies precision.\n");
  printf("-y run in robust mode against errors (cannot be used with DNL).\n");
  printf("-z uses default Huffman tables.\n");
}

/*END*/
