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
globals.h

This file contains the global includes and other definitions.

************************************************************
*/

#ifndef GLOBAL_DONE
#define GLOBAL_DONE

#include <stdio.h>
#include "prototypes.h"
#include "param.h"
#include "system.h"


/* Map stream functions to those used in stream.c (MSB) */
/* Makes for easy alterations for least-significant bit non-JPEG defns. */

#define sropen mropen
#define srclose mrclose
#define swopen mwopen
#define swclose mwclose

#define sgetb megetb
#define sgetv megetv
#define sputv meputv

#define swtell mwtell
#define srtell mrtell

#define swseek mwseek
#define srseek mrseek

#define IMAGE struct Image_Definition
#define FRAME struct Frame_Definition
#define SCAN struct Scan_Definition

#define MUTE 0
#define WHISPER 1
#define TALK 2
#define NOISY 3
#define SCREAM 4


/* The defined flag for encoding/decoding. */
#define J_DECODER 1
#define J_FULLHUFFMAN 2
#define J_DEFAULTHUFFMAN 4
#define J_LOSSLESS 8

/* Some flags for JpegCustomScan() */

#define CUSTOM_DO_DC 1
#define CUSTOM_DO_AC 2

/* Error flags */

#define ERROR_NONE 0
#define ERROR_BOUNDS 1            /*Input Values out of bounds */
#define ERROR_HUFFMAN_READ 2      /*Huffman Decoder finds bad code */
#define ERROR_HUFFMAN_ENCODE 3    /*Undefined value in encoder */
#define ERROR_MARKER 4            /*Error Found in Marker */
#define ERROR_INIT_FILE 5         /*Cannot initialize files */
#define ERROR_UNRECOVERABLE 6     /*No recovery mode specified */
#define ERROR_PREMATURE_EOF 7     /*End of file unexpected */
#define ERROR_MARKER_STRUCTURE 8  /*Bad Marker Structure */
#define ERROR_WRITE 9             /*Cannot write output */
#define ERROR_READ 10             /*Cannot write input */
#define ERROR_PARAMETER 11        /*System Parameter Error */
#define ERROR_MEMORY 12           /*Memory exceeded */

typedef int iFunc();
typedef void vFunc();

/* A flag obtaining macro */
#define GetFlag(value,flag) (((value) & (flag)) ? 1:0)

/* MAX and MIN macros */
#define MAX(x,y) ((x > y) ? x:y)
#define MIN(x,y) ((x > y) ? y:x)

/* BEGIN is used to start most routines. It sets up the Routine Name */
/* which is used in the WHEREAMI() macro */
#ifdef CODEC_DEBUG
#define BEGIN(name) static char RoutineName[]= name;
#else
#define BEGIN(name)
#endif /*CODEC_DEBUG*/
/* WHEREAMI prints out current location in code. */
#ifdef CODEC_DEBUG
#define WHEREAMI() printf("F>%s:R>%s:L>%d: ",\
        __FILE__,RoutineName,__LINE__)
#else
#define WHEREAMI()
#endif /* CODEC_DEBUG */

/* InBounds is used to test whether a value is in or out of bounds. */
#define InBounds(var,lo,hi,str)\
{if (((var) < (lo)) || ((var) > (hi)))\
{WHEREAMI(); printf("%s in %d\n",(str),(var));ErrorValue=ERROR_BOUNDS;}}

/* MakeStructure makes the named structure */
#define MakeStructure(named_st) ((named_st *) malloc(sizeof(named_st)))

IMAGE {
char *StreamFileName;            /* Name of compressed stream file */
int JpegMode;                    /* Mode of JPEG encoder */
int Jfif;                        /* If set, automatically drop JFIF marker */
int ImageSequence;               /* Index in image sequence */
int NumberQuantizationMatrices;  /* Number of quantization matrices */
int *QuantizationMatrices[MAXIMUM_DEVICES]; /* Pointers to q-matrices */
int NumberACTables;              /* Number of AC Huffman tables */
DHUFF *ACDhuff[MAXIMUM_DEVICES]; /* Decoder huffman tables */
EHUFF *ACEhuff[MAXIMUM_DEVICES]; /* Encoder huffman tables */
XHUFF *ACXhuff[MAXIMUM_DEVICES]; /* Transmittable huffman tables */
int NumberDCTables;              /* Number of DC Huffman tables */
DHUFF *DCDhuff[MAXIMUM_DEVICES]; /* Decoder huffman tables */
EHUFF *DCEhuff[MAXIMUM_DEVICES]; /* Encoder huffman tables */
XHUFF *DCXhuff[MAXIMUM_DEVICES]; /* Transmittable huffman tables */
};

FRAME {
int Type;                       /* SOF(X) where X is type (4 bits) */
char *ComponentFileName[MAXIMUM_COMPONENTS]; /* image component file names */
int InsertDnl;                  /* DNL flag (-2 = AUTO) (-1 = ENABLE) (>0 ) */
int Q;                          /* Q Factor (0 disables) */
int DataPrecision;              /* Data Precision (not used) */
int GlobalHeight;               /* Dimensions of overall image */
int GlobalWidth;
int ResyncInterval;             /* Resync interval (0 disables) */
int GlobalNumberComponents;     /* Global number of components */
int cn[MAXIMUM_COMPONENTS];     /* Translation index used */
int hf[MAXIMUM_COMPONENTS];     /* Horizontal frequency */
int vf[MAXIMUM_COMPONENTS];     /* Vertical frequency */
int tq[MAXIMUM_COMPONENTS];     /* Quantization table used by */
int Width[MAXIMUM_COMPONENTS];  /* Dimensions of component files */
int Height[MAXIMUM_COMPONENTS];
int BufferSize;                 /* Buffer sizes used */
int Maxv, Maxh;                 /* Max Sampling Freq */
int MDUWide, MDUHigh;           /* Number MDU wide */
int tmpfile;
IMAGE *Image;
};

SCAN {
int NumberComponents;               /* Number of components in scan */
int SSS;                            /* Spectral Selection Start (not used) */
int SSE;                            /* Spectral Selection End (not used) */
int SAH;                            /* Spectral approximation (not used) */
int SAL;                            /* Spectral approximation (not used) */
int *LastDC[MAXIMUM_SOURCES];       /* LastDC DPCM predictor */
int *ACFrequency[MAXIMUM_SOURCES];  /* Frequency charts for custom huffman */
int *DCFrequency[MAXIMUM_SOURCES];  /* table building */
int LosslessBuffer[MAXIMUM_SOURCES][LOSSLESSBUFFERSIZE];
int MDUWide, MDUHigh;
                                    /* a integer buffer for lossless coding */
IOBUF *Iob[MAXIMUM_SOURCES];        /* IOB per scan index  */
int ci[MAXIMUM_SOURCES];            /* Index */
int ta[MAXIMUM_SOURCES];            /* AC Tables for that scan index */
int td[MAXIMUM_SOURCES];            /* DC Tables for scan index */
int NumberACTablesSend;             /* Number of tables to send */
int NumberDCTablesSend;
int NumberQTablesSend;
int sa[MAXIMUM_SOURCES];            /* AC table indices to send */
int sd[MAXIMUM_SOURCES];            /* DC table indices to send */
int sq[MAXIMUM_SOURCES];            /* Quantization table indices to send */
};

#endif
