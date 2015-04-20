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
stream.c

This file is used for management of bit-aligned files.

************************************************************
*/

/*LABEL stream.c */

/* Include files */

#include "globals.h"
#include "marker.h"
#include "stream.h"
#include <stdlib.h> /* exit */

/*PUBLIC*/

extern void initstream();
extern void pushstream();
extern void popstream();
extern void bpushc();
extern int bgetc();
extern void bputc();
extern void mropen();
extern void mrclose();
extern void mwopen();
extern void swbytealign();
extern void mwclose();
extern long mwtell();
extern long mrtell();
extern void mwseek();
extern void mrseek();
extern int megetb();
extern void meputv();
extern int megetv();
extern int DoMarker();
extern int ScreenMarker();
extern void Resync();
extern void WriteResync();
extern int ReadResync();
extern int ScreenAllMarker();
extern int DoAllMarker();

static int pgetc();

/*PRIVATE*/

/* External values */

extern int ErrorValue;
extern IMAGE *CImage;
extern int Robust;         /* Whether to ignore scan markers and such */

/* Masks */
int bit_set_mask[] = { /* This is 2^i at ith position */
0x00000001,0x00000002,0x00000004,0x00000008,
0x00000010,0x00000020,0x00000040,0x00000080,
0x00000100,0x00000200,0x00000400,0x00000800,
0x00001000,0x00002000,0x00004000,0x00008000,
0x00010000,0x00020000,0x00040000,0x00080000,
0x00100000,0x00200000,0x00400000,0x00800000,
0x01000000,0x02000000,0x04000000,0x08000000,
0x10000000,0x20000000,0x40000000,0x80000000};
int lmask[] = {        /* This is 2^{i+1}-1 */
0x00000001,0x00000003,0x00000007,0x0000000f,
0x0000001f,0x0000003f,0x0000007f,0x000000ff,
0x000001ff,0x000003ff,0x000007ff,0x00000fff,
0x00001fff,0x00003fff,0x00007fff,0x0000ffff,
0x0001ffff,0x0003ffff,0x0007ffff,0x000fffff,
0x001fffff,0x003fffff,0x007fffff,0x00ffffff,
0x01ffffff,0x03ffffff,0x07ffffff,0x0fffffff,
0x1fffffff,0x3fffffff,0x7fffffff,0xffffffff};

#ifdef __OLD
int umask[] = {        /* This is -1 XOR 2^{i+1}-1 */
0xfffffffe,0xfffffffc,0xfffffff8,0xfffffff0,
0xffffffe0,0xffffffc0,0xffffff80,0xffffff00,
0xfffffe00,0xfffffc00,0xfffff800,0xfffff000,
0xffffe000,0xffffc000,0xffff8000,0xffff0000,
0xfffe0000,0xfffc0000,0xfff80000,0xfff00000,
0xffe00000,0xffc00000,0xff800000,0xff000000,
0xfe000000,0xfc000000,0xf8000000,0xf0000000,
0xe0000000,0xc0000000,0x80000000,0x00000000};
#endif

/* Internally kept variables for global flag communication */

int CleartoResync=0;       /* Return black blocks until last Resync reached*/
int ResyncEnable=0;         /* This enables the resync feature */
int ResyncCount=0;     /* This is the resync marker count */
int LastKnownResync=0;     /* This is the index of the next Resync */
int EndofFile=0;     /* End of file means read stream exhausted */
int EndofImage=0;     /* End of image means EOI image marker found */

/* Static variables that keep internal state. */

static FILE *swout;
static FILE *srin;
static unsigned int current_write_byte;
static unsigned int current_read_byte;
static unsigned int marker_read_byte;
static int read_position;
static int write_position;
static int InResync=0;

/* Stack of variables to handle multiple streams. */

static int Stack_Stream_Current= -1;
static int Stack_Stream_Active[NUMBER_OF_STREAMS];
static int Stack_Stream_CleartoResync[NUMBER_OF_STREAMS];
static int Stack_Stream_ResyncEnable[NUMBER_OF_STREAMS];
static int Stack_Stream_ResyncCount[NUMBER_OF_STREAMS];
static int Stack_Stream_LastKnownResync[NUMBER_OF_STREAMS];
static int Stack_Stream_EndofFile[NUMBER_OF_STREAMS];
static int Stack_Stream_EndofImage[NUMBER_OF_STREAMS];
static FILE * Stack_Stream_swout[NUMBER_OF_STREAMS];
static FILE * Stack_Stream_srin[NUMBER_OF_STREAMS];
static unsigned int Stack_Stream_current_write_byte[NUMBER_OF_STREAMS];
static unsigned int Stack_Stream_current_read_byte[NUMBER_OF_STREAMS];
static unsigned int Stack_Stream_marker_read_byte[NUMBER_OF_STREAMS];
static int Stack_Stream_read_position[NUMBER_OF_STREAMS];
static int Stack_Stream_write_position[NUMBER_OF_STREAMS];

/*START*/

/* STACK STREAM LIBRARY */

/*BFUNC

initstream() initializes all of the stream variables-- especially the
stack. Not necessary to call unless you wish to use more than one
stream variable.

EFUNC*/

void initstream()
{
  BEGIN("initstream")
  int i;

  Stack_Stream_Current= -1;
  for(i=0;i<NUMBER_OF_STREAMS;i++)
    {
      Stack_Stream_Active[i]=0;
      Stack_Stream_swout[i]=NULL;
      Stack_Stream_srin[i]=NULL;
    }
}

/*BFUNC

pushstream() pushes the currently active stream into its predefined
location.

EFUNC*/

void pushstream()
{
  BEGIN("pushstream")

  if (Stack_Stream_Current < 0) return;
  Stack_Stream_CleartoResync[Stack_Stream_Current]=CleartoResync;
  Stack_Stream_ResyncEnable[Stack_Stream_Current]=ResyncEnable;
  Stack_Stream_ResyncCount[Stack_Stream_Current]=ResyncCount;
  Stack_Stream_LastKnownResync[Stack_Stream_Current]=LastKnownResync;
  Stack_Stream_EndofFile[Stack_Stream_Current]=EndofFile;
  Stack_Stream_EndofImage[Stack_Stream_Current]=EndofImage;
  Stack_Stream_swout[Stack_Stream_Current]=swout;
  Stack_Stream_srin[Stack_Stream_Current]=srin;
  Stack_Stream_current_write_byte[Stack_Stream_Current]=current_write_byte;
  Stack_Stream_current_read_byte[Stack_Stream_Current]=current_read_byte;
  Stack_Stream_marker_read_byte[Stack_Stream_Current]=marker_read_byte;
  Stack_Stream_read_position[Stack_Stream_Current]=read_position;
  Stack_Stream_write_position[Stack_Stream_Current]=write_position;
}

/*BFUNC

popstream() gets the specified stream from the location.  If there
is already a current active stream, it removes it.

EFUNC*/

void popstream(index)
     int index;
{
  BEGIN("popstream")

  if ((index < 0)||(!Stack_Stream_Active[index]))
    {
      WHEREAMI();
      printf("Cannot pop non-existent stream.\n");
      exit(ERROR_BOUNDS);
    }
  if (Stack_Stream_Current >=0) pushstream();
  CleartoResync=Stack_Stream_CleartoResync[index];
  ResyncEnable=Stack_Stream_ResyncEnable[index];
  ResyncCount=Stack_Stream_ResyncCount[index];
  LastKnownResync=Stack_Stream_LastKnownResync[index];
  EndofFile=Stack_Stream_EndofFile[index];
  EndofImage=Stack_Stream_EndofImage[index];
  swout=Stack_Stream_swout[index];
  srin=Stack_Stream_srin[index];
  current_write_byte=Stack_Stream_current_write_byte[index];
  current_read_byte=Stack_Stream_current_read_byte[index];
  marker_read_byte=Stack_Stream_marker_read_byte[index];
  read_position=Stack_Stream_read_position[index];
  write_position=Stack_Stream_write_position[index];
}

/* THAT'S ALL FOR THE STACK STREAM LIBRARY! */

/* BUFFER LIBRARY */

/*BFUNC

brtell() is used to find the location in the read stream.

EFUNC*/

int brtell()
  {BEGIN("brtell") return(ftell(srin));}

/*BFUNC

brseek() is used to find the location in the read stream.

EFUNC*/

int brseek(offset,ptr)
     int offset;
     int ptr;
  {BEGIN("brseek") return(fseek(srin,offset,ptr));}

/*BFUNC

bpushc() is used to unget a character value from the current stream.

EFUNC*/

void bpushc(value)
     int value;
  {BEGIN("bpushc") ungetc(value,srin);}

/*BFUNC

bgetc() gets a character from the stream. It is byte aligned and
bypasses bit buffering.

EFUNC*/

int bgetc()
  {BEGIN("bgetc") return(getc(srin));}

/*BFUNC

bgetw() gets a msb word from the stream.

EFUNC*/

int bgetw()
  {BEGIN("bgetw") int fu; fu=getc(srin); return ((fu << 8)| getc(srin));}

/*BFUNC

bputc() puts a character into the stream. It is byte aligned and
bypasses the bit buffering.

EFUNC*/

void bputc(c)
     int c;
  {BEGIN("bputc") putc(c,swout);}

/* PROTECTED MARKER GETS AND FETCHES */

/*BFUNC

pgetc() gets a character onto the stream but it checks to see
if there are any marker conflicts.

EFUNC*/

static int pgetc()
{
  BEGIN("pgetc")
  int temp;

  if (CleartoResync)           /* If cleartoresync do not read from stream */
    {
      return(0);
    }
  if ((temp = bgetc())==MARKER_MARKER)   /* If MARKER then */
    {
      if ((temp = bgetc()))              /* if next is not 0xff, then marker */
  {
    WHEREAMI();
    printf("Unanticipated marker detected.\n");
    if (!ResyncEnable) DoAllMarker(); /* If no resync enabled */
  }                                   /* could be marker */
      else
  {
    return(MARKER_MARKER);        /* else truly 0xff */
  }
    }
  return(temp);
}

/*BMACRO

pputc(stream,)
     ) puts a value onto the stream; puts a value onto the stream, appending an extra '0' if it
matches the marker code.

EMACRO*/

#define pputc(val) {bputc(val); if (val==MARKER_MARKER) bputc(0);}

/* MAIN ROUTINES */

/*BFUNC

mropen() opens a given filename as the input read stream.

EFUNC*/

void mropen(filename,index)
     char *filename;
     int index;
{
  BEGIN("mropen")

  if (Stack_Stream_Active[index])
    {
      WHEREAMI();
      printf("%s cannot be opened because %d stream slot filled.\n",
       filename,index);
      exit(ERROR_BOUNDS);
    }
  if (Stack_Stream_Current!=index) pushstream();
  current_read_byte=0;
  read_position = -1;
  if ((srin = fopen(filename,"rb"))==NULL)
    {
      WHEREAMI();
      printf("Cannot read input file %s.\n",
       filename);
      exit(ERROR_INIT_FILE);
    }
  CleartoResync=0;
  ResyncEnable=0;
  ResyncCount=0;
  LastKnownResync=0;
  EndofFile=0;
  EndofImage=1;    /* We start after "virtual" end of previous image */
  Stack_Stream_Current= index;
  Stack_Stream_Active[index]=1;
}

/*BFUNC

mrclose() closes the input read stream.

EFUNC*/

void mrclose()
{
  BEGIN("mrclose")
  fclose(srin);
  srin=NULL;
  if (swout==NULL)
    {
      Stack_Stream_Active[Stack_Stream_Current]=0;
      Stack_Stream_Current= -1;
    }
}

/*BFUNC

mwopen() opens the stream for writing. Note that reading and
writing can occur simultaneously because the read and write
routines are independently buffered.

EFUNC*/

void mwopen(filename,index)
     char *filename;
     int index;
{
  BEGIN("mwopen")

  if (Stack_Stream_Active[index])
    {
      WHEREAMI();
      printf("%s cannot be opened because %d stream slot filled.\n",
       filename,index);
      exit(ERROR_BOUNDS);
    }
  if ((Stack_Stream_Current!=index)) pushstream();
  current_write_byte=0;
  write_position=7;
  if ((swout = fopen(filename,"wb+"))==NULL)
    {
      WHEREAMI();
      printf("Cannot open output file %s.\n",filename);
      exit(ERROR_INIT_FILE);
    }
  Stack_Stream_Current= index;
  Stack_Stream_Active[index]=1;
}

/*BFUNC

swbytealign() flushes the current bit-buffered byte out to the stream.
This is used before marker codes.

EFUNC*/

void swbytealign()
{
  BEGIN("swbytealign")

  if (write_position !=7)
    {
      current_write_byte |= lmask[write_position];
      pputc(current_write_byte);
      write_position=7;
      current_write_byte=0;
    }
}

/*BFUNC

mwclose() closes the stream that has been opened for writing.

EFUNC*/

void mwclose()
{
  BEGIN("mwclose")

  swbytealign();
  fclose(swout);
  swout=NULL;
  if (srin==NULL)
    {
      Stack_Stream_Active[Stack_Stream_Current]=0;
      Stack_Stream_Current= -1;
    }
}

/*BFUNC

mwtell() returns the bit position on the write stream.

EFUNC*/

long mwtell()
{
  BEGIN("mwtell")

  return((ftell(swout)<<3) + (7 - write_position));
}

/*BFUNC

mrtell() returns the bit position on the read stream.

EFUNC*/

long mrtell()
{
  BEGIN("mrtell")

  return((ftell(srin)<<3) - (read_position+1));
}

/*BFUNC

mwseek returns the bit position on the write stream.

EFUNC*/

void mwseek(distance)
     long distance;
{
  BEGIN("mwseek")
  int length;

  if (write_position!=7)             /* Must flush out current byte */
    {
      putc(current_write_byte,swout);
    }
  fseek(swout,0,2L);                 /* Find end */
  length = ftell(swout);
  fseek(swout,((distance+7)>>3),0L);
  if ((length<<3) <= distance)       /* Make sure we read clean stuff */
    {
      current_write_byte = 0;
      write_position = 7 - (distance & 0x7);
    }
  else
    {
      current_write_byte = getc(swout);  /* if within bounds, then read byte */
      write_position = 7 - (distance & 0x7);
      fseek(swout,((distance+7)>>3),0L); /* Reset seek pointer for write */
    }
}


/*BFUNC

mrseek() jumps to a bit position on the read stream.

EFUNC*/

void mrseek(distance)
     long distance;
{
  BEGIN("mrseek")

  fseek(srin,(distance>>3),0L);       /* Go to location */
  current_read_byte = bgetc();        /* read byte in */
  read_position = 7 - (distance % 8);
}


/*BFUNC

megetb() gets a bit from the read stream.

EFUNC*/

int megetb()
{
  BEGIN("megetb")

  if (read_position < 0)
    {
      current_read_byte = pgetc();
      read_position=7;
    }
  if (current_read_byte&bit_set_mask[read_position--])
    {
      return(1);
    }
  return(0);
}

/*BFUNC

meputv() puts n bits from b onto the writer stream.

EFUNC*/

void meputv(n,b)
     int n;
     int b;
{
  BEGIN("meputv")
  int p;

  n--;
  b &= lmask[n];
  p = n - write_position;
  if (!p)                           /* Can do parallel save immediately */
    {
      current_write_byte |= b;
      pputc(current_write_byte);
      current_write_byte = 0;
      write_position = 7;
      return;
    }
  else if (p < 0)                   /* if can fit, we have to shift byte */
    {
      p = -p;
      current_write_byte |= (b << p);
      write_position = p-1;
      return;
    }
  current_write_byte |= (b >> p);  /* cannot fit. we must do putc's */
  pputc(current_write_byte);       /* Save off  remainder */
  while(p > 7)                     /* Save off bytes while remaining > 7 */
    {
      p -= 8;
      current_write_byte = (b >> p) & lmask[7];
      pputc(current_write_byte);
    }
  if (!p)                          /* If zero then reset position */
    {
      write_position = 7;
      current_write_byte = 0;
    }
  else                             /* Otherwise reset write byte buffer */
    {
      write_position = 8-p;
      current_write_byte = (b << write_position) & lmask[7];
      write_position--;
    }
}

/*BFUNC

megetv() gets n bits from the read stream and returns it.

EFUNC*/

int megetv(n)
     int n;
{
  BEGIN("megetv")
  int p,rv;

  n--;
  p = n-read_position;
  while(p > 0)
    {
      if (read_position>23)  /* If byte buffer contains almost entire word */
  {
    rv = (current_read_byte << p);  /* Manipulate buffer */
    current_read_byte = pgetc();    /* Change read bytes */
    rv |= (current_read_byte >> (8-p));
    read_position = 7-p;
    return(rv & lmask[n]);          /* Can return pending residual val */
  }
      current_read_byte = (current_read_byte << 8) | pgetc();
      read_position += 8;                 /* else shift in new information */
      p -= 8;
    }
  if (!p)                                 /* If position is zero */
    {
      read_position = -1;                 /* Can return current byte */
      return(current_read_byte & lmask[n]);
    }
  p = -p;                                 /* Else reverse position and shift */
  read_position = p-1;
  return((current_read_byte >> p) & lmask[n]);
}


/*BFUNC

DoMarker() performs marker analysis. We assume that the Current Marker
head has been read (0xFF) plus top information is at
marker\_read\_byte.

EFUNC*/

int DoMarker()
{
  BEGIN("DoMarker")
  int i,hin,lon,marker,length;

  current_read_byte = 0;
  read_position= -1;                    /* Make sure we are byte-flush. */
  while(marker_read_byte==MARKER_FIL)   /* Get rid of FIL markers */
    {
#ifdef VERSION_1_0
      if ((marker_read_byte = bgetc())!=MARKER_MARKER)
  {
    WHEREAMI();
    printf("Unknown FIL marker. Bypassing.\n");
    ErrorValue = ERROR_MARKER;
    return(0);
  }
#endif
      marker_read_byte = bgetc();
    }
  lon = marker_read_byte & 0x0f;         /* Segregate between hi and lo */
  hin = (marker_read_byte>>4) & 0x0f;    /* nybbles for the marker read byte */
  marker = marker_read_byte;

  if (InResync)
    {
      if ((marker <0xd0)||(marker>0xd7))
  {
    WHEREAMI();
    printf("Illegal resync marker found.\n");
    return(0);
  }
    }
  switch(hin)                            /* Pretty much self explanatory */
    {
    case 0x0c:                           /* Frame Style Marker */
      switch(lon)
  {
  case 0x04:
    ReadDht();
    break;
  case 0x00:
  case 0x01:
  case 0x03:
    ReadSof(lon);
    break;
  case 0x08:
  case 0x09:
  case 0x0a:
  case 0x0b:
  case 0x0c:
  case 0x0d:
  case 0x0e:
  case 0x0f:
    WHEREAMI();
    printf("Arithmetic coding not supported.\n");
    length = bgetw();
    for(i=2;i<length;i++)          /* Length adds 2 bytes itself */
      bgetc();
    break;
  case 0x02:
  case 0x05:
  case 0x06:
  case 0x07:
  default:
    WHEREAMI();
    printf("Frame type %x not supported.\n",lon);
    length = bgetw();
    for(i=2;i<length;i++)          /* Length adds 2 bytes itself */
      bgetc();
    break;
  }
      break;
    case 0x0d:  /* Resync Marker */
      if (lon > 7)
  {
    switch(lon)
      {
      case 0x08:                    /* Start of Image */
        EndofImage=0;               /* If End of Image occurs */
        CImage->ImageSequence++;    /* reset, and increment sequence */
        break;
      case 0x09:                    /* End of Image */
        EndofImage=1;
        break;
      case 0x0a:
        ResyncCount=0;              /* SOS clears the resync count */
        ReadSos();
        break;
      case 0x0b:
        ReadDqt();
        break;
      case 0x0c:
        ReadDnl();
        break;
      case 0x0d:
        ReadDri();
        break;
      default:
        WHEREAMI();
        printf("Hierarchical markers found.\n");
        length = bgetw();
        for(i=2;i<length;i++)      /* Length adds 2 bytes itself */
    {
      bgetc();
    }
        break;
      }
  }
      break;
    case 0x0e: /* Application Specific */
      length = bgetw();
      for(i=2;i<length;i++) /* Length adds 2 bytes itself */
  bgetc();
      break;
    case 0x0f: /* JPEG Specific */
      length = bgetw();
      for(i=2;i<length;i++) /* Length adds 2 bytes itself */
  bgetc();
      break;
    default:
      WHEREAMI();
      printf("Bad marker byte %d.\n",marker);
      Resync();
      ErrorValue = ERROR_MARKER;
      return(-1);
      break;
    }
  return(marker);
}

/*BFUNC

ScreenMarker() looks to see what marker is present on the stream.  It
returns with the marker value read.

EFUNC*/

int ScreenMarker()
{
  BEGIN("ScreenMarker")

  if (read_position!=7)                  /* Already read byte */
    {
      current_read_byte = 0;
      read_position= -1;                 /* Consume byte to be flush */
      if ((marker_read_byte=bgetc())==(unsigned int)EOF)
  {
    EndofFile=2;
    return(EOF);
  }
    }
  else                /* If flush, then marker byte is current read byte */
    {
      marker_read_byte = current_read_byte;
    }
  if (marker_read_byte!=MARKER_MARKER)    /* Not a marker, return -1. */
    {
      current_read_byte = marker_read_byte;
      read_position=7;
      return(-1);
    }
  while((marker_read_byte = bgetc())==MARKER_FIL)
    {                                      /* Get rid of FIL markers */
      if ((marker_read_byte = bgetc())!=MARKER_MARKER)
  {
    WHEREAMI();
    printf("Unattached FIL marker.\n");
    ErrorValue = ERROR_MARKER;
    return(-1);
  }
      if (marker_read_byte == (unsigned int)EOF)        /* Found end of file */
  {
    EndofFile=2;
    return(EOF);
  }
      marker_read_byte = bgetc();         /* Otherwise read another byte */
    }                                     /* Call processor for markers */
  if (marker_read_byte)  return(DoMarker());
  else                                    /* Is a FF00 so don't process */
    {
      current_read_byte=MARKER_MARKER;     /* 255 actually read */
      read_position=7;
      return(-1);
    }
}

/*BFUNC

Resync() does a resync action on the stream. This involves searching
for the next resync byte.

EFUNC*/

void Resync()
{
  BEGIN("Resync")

  if (!ResyncEnable)
    {
      WHEREAMI();
      printf("Resync without resync enabled\n");
      printf("Fatal error.\n");
      TerminateFile();
      exit(ERROR_UNRECOVERABLE);
    }
  WHEREAMI();
  printf("Attempting resynchronization.\n");
  do
    {
      while((marker_read_byte = bgetc())!=MARKER_MARKER)
  {
    if (marker_read_byte==(unsigned int)EOF)
      {
        WHEREAMI();
        printf("Attempt to resync at end of file.\n");
        printf("Sorry.\n");
        TerminateFile();
        exit(ERROR_PREMATURE_EOF);
      }
  }
    }
  while(((marker_read_byte = bgetc()) & MARKER_RSC_MASK)!=MARKER_RSC);
  LastKnownResync = marker_read_byte & 0x07;  /* Set up currently read */
  WHEREAMI();                                 /* resync byte as future ref */
  printf("Resync successful!\n");
  /*
    In general, we assume that we must add black space
    until resynchronization. This is consistent under both
    byte loss, byte gain, and byte corruption.
    We assume corruption does not create new markers with
    an RSC value--if so, we are probably dead, anyways.
    */
  CleartoResync=1;
  ResyncCount = (LastKnownResync+1)&0x07;
  current_read_byte = 0;
  read_position = -1;
  ResetCodec();  /* Reset the codec incase in a non-local jump. */

  printf("ResyncCount: %d  LastKnownResync: %d\n",
   ResyncCount,LastKnownResync);
}

/*BFUNC

WriteResync() writes a resync marker out to the write stream.

EFUNC*/

void WriteResync()
{
  BEGIN("WriteResync")

  swbytealign();                   /* This procedure writes a byte-aligned */
  bputc(MARKER_MARKER);            /* resync marker. */
  bputc((MARKER_RSC|(ResyncCount & 0x07)));
  ResyncCount = (ResyncCount + 1) & 0x07;
}

/*BFUNC

ReadResync() looks for a resync marker on the stream. It returns a 0
if successful and a -1 if a search pass was required.

EFUNC*/

int ReadResync()
{
  BEGIN("ReadResync")
  int ValueRead;

  if (Robust) InResync=1;
  while((ValueRead = ScreenMarker()) >= 0)
    {
      if ((ValueRead & MARKER_RSC_MASK)!=MARKER_RSC) /* Strange marker found */
  {
    if (ValueRead != MARKER_DNL)  /* DNL only other possibility */
      {                           /* actually excluded, never reached */
        WHEREAMI();               /* 11/19/91 ACH */
        printf("Non-Resync marker found for resync.\n");
        printf("Trying again.\n");
      }
  }
      else
  {
    ValueRead = ValueRead & 0x07;  /* If so, then check resync count */
    if (ValueRead != ResyncCount)
      {
        WHEREAMI();
        printf("Bad resync counter. No search done.\n");
      }
    ResyncCount = (ResyncCount+1)&0x07;
                                         /* Flush spurious markers. */
    while((ValueRead = ScreenMarker()) >= 0);
    InResync=0;
    return(0);
  }
    }
  WHEREAMI();
  printf("Anticipated resync not found.\n");
  Resync();
  InResync=0;
  return(-1);
}

/*BFUNC

ScreenAllMarker() looks for all the markers on the stream. It returns
a 0 if a marker has been found, -1 if no markers exist.

EFUNC*/

int ScreenAllMarker()
{
  BEGIN("ScreenAllMarker")

  if (ScreenMarker()<0)
    {
      return(-1);
    }
  while(ScreenMarker()>=0);  /* Flush out all markers */
  return(0);
}

/*BFUNC

DoAllMarker() is the same as ScreenAllMarker except we assume that the
prefix markerbyte (0xff) has been read and the second byte of the
prefix is in the marker\_byte variable. It returns a -1 if there is an
error in reading the marker.

EFUNC*/

int DoAllMarker()
{
  BEGIN("DoAllMarker")

  if (DoMarker()<0)
    {
      return(-1);
    }
  while(ScreenMarker()>=0);   /* Flush out all markers */
  return(0);
}

/*END*/
