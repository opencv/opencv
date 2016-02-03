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
io.c

This package is used to manipulate the raw image files.

There are two standards: block based, assumed to be in sizes of the
DCT block, herein defined as BlockWidth and BlockHeight; and a special
case, two-line-based, assumed to be of two lines per.

************************************************************
*/

/*LABEL io.c */

/* Include definitions. */
#include "globals.h"
#ifdef SYSV
#include <sys/fcntl.h>
#include <sys/unistd.h>
#endif
#include <stdlib.h> /* malloc */
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h> /* memcpy */
#include <assert.h> /* memcpy */
#ifdef WIN32
#include <io.h> /* lseek on win32 */
#endif


/* Functions which are local and which are exported. */

/*PUBLIC*/

static BUFFER *MakeXBuffer();
static void WriteXBuffer();
static void ReadXBuffer();
static void ReadResizeBuffer();
static void FlushBuffer();
static void BlockMoveTo();
static void ReadXBound();
static void WriteXBound();

static void LineMoveTo();

extern void ReadBlock();
extern void WriteBlock();
extern void ResizeIob();
extern void RewindIob();
extern void FlushIob();
extern void SeekEndIob();
extern void CloseIob();
extern void MakeIob();
extern void PrintIob();
extern void InstallIob();
extern void TerminateFile();

extern void ReadLine();
extern void ReadPreambleLine();
extern void WriteLine();
extern void LineResetBuffers();


/*PRIVATE*/

/* External variables */
extern int Loud;
extern int PointTransform;   /* Used for shifting the pels from the I/O */
extern IMAGE *CImage;
extern FRAME *CFrame;
extern SCAN *CScan;

/* Internal variables. */
static IOBUF *Iob=NULL;               /* Internal I/O buffer. */
static int BlockWidth = BLOCKWIDTH;   /* Block width. */
static int BlockHeight = BLOCKHEIGHT; /* Block height. */

/* Buffer calculation information */

#define BufferIndex(i,iobuf) (iobuf)->blist[(i)]
#define TrueBufferPos(buffer) (((buffer)->currentoffs - \
((buffer)->tptr - (buffer)->bptr))/buffer->wsize)


/*START*/
/*BFUNC

MakeXBuffer() constructs a holding buffer for the stream input. It
takes up a size passed into it and returns the appropriate buffer
structure.

EFUNC*/

static BUFFER *MakeXBuffer(nelem,wsize)
     int nelem;
     int wsize;
{
  BEGIN("MakeXBuffer")
  BUFFER *temp;

  if (!(temp = MakeStructure(BUFFER)))     /* Make structure */
    {
      WHEREAMI();
      printf("Cannot allocate buffer structure.\n");
      exit(ERROR_MEMORY);
    }
  temp->disable=0;                          /* Not disabled */
  temp->wsize = wsize;                      /* Set up word size */
  temp->size=nelem*wsize;                   /* Set up size, offset */
  temp->currentoffs = 0;
  temp->streamoffs = 0;
  if (!(temp->space =(unsigned char *)     /* Allocate buffer space */
  calloc(temp->size+1,sizeof(unsigned char))))
    {
      WHEREAMI();
      printf("Cannot allocate buffer memory.\n");
      exit(ERROR_MEMORY);
    }
  temp->tptr = temp->bptr = temp->space;
  return(temp);
}

/*BFUNC

ResizeIob() is used to resize the Iob height and width to conform to
that of the CScan. This is used for the dynamic Number-of-lines
rescaling.

EFUNC*/

void ResizeIob()
{
  BEGIN("ResizeIob")
  int index;

  for(index=0;index<CScan->NumberComponents;index++)
    {
      CScan->Iob[index]->width = CFrame->Width[CScan->ci[index]];
      CScan->Iob[index]->height = CFrame->Height[CScan->ci[index]];
    }
}

/*BFUNC

MakeIob() is used to create an Iob structure for use in the CScan
structure. An Iob consists of several Buffer structures with some
additional sizing information. The input flags set up the parameters
of the stream.

EFUNC*/

void MakeIob(type,flags,wsize)
     int type;
     int flags;
     int wsize;
{
  BEGIN("MakeIob")
  int index,sofs;
  BUFFER **current;
  IOBUF *temp;

  for(index=0;index<CScan->NumberComponents;index++) /* Make IOBUF */
    {                                                /* For each component */
      if (!(temp = MakeStructure(IOBUF)))
  {
    WHEREAMI();
    printf("Cannot allocate IOBUF structure.\n");
    exit(ERROR_MEMORY);
  }
      temp->linelastdefault=(1<<(CFrame->DataPrecision-PointTransform-1));
      temp->type = type;
      temp->wsize = wsize;
      temp->hpos=0;
      temp->vpos=0;
      temp->width = CFrame->Width[CScan->ci[index]];  /* Set up widthxheight */
      temp->height = CFrame->Height[CScan->ci[index]];
      if (CScan->NumberComponents==1)
  {
    temp->hor = 1;               /* For non-interleaved mode the freq */
    temp->ver = 1;               /* is always 1x1 */
  }
      else
  {
    temp->hor = CFrame->hf[CScan->ci[index]];       /* and hf x vf */
    temp->ver = CFrame->vf[CScan->ci[index]];
  }
      switch(temp->type)
  {
  case IOB_BLOCK:             /* BLOCK TYPE */
    temp->num = temp->ver*BlockHeight;
    break;
  case IOB_LINE:             /* LINE TYPE */
    temp->num = temp->ver + 1;
    break;
  default:
    WHEREAMI();
    printf("Illegal type specified: %d.\n",type);
    exit(ERROR_BOUNDS);
  }
      temp->flags = flags;                            /* and also flags */
      if (!(temp->blist =                             /*Set up buffer list */
      (BUFFER **) calloc(temp->num,sizeof(BUFFER *))))
  {
    WHEREAMI();
    printf("Cannot allocate Iob bufferlist.\n");
    exit(ERROR_MEMORY);
  }
     if( CFrame->tmpfile )
  {
      temp->file = CFrame->tmpfile;
  }
    else
  {
      if ((temp->file =                               /* Open file */
     open(CFrame->ComponentFileName[CScan->ci[index]],
    flags,UMASK)) < 0)
  {
    WHEREAMI();
    printf("Cannot open file %s.\n",
     CFrame->ComponentFileName[CScan->ci[index]]);
    exit(ERROR_INIT_FILE);
  }               /* Make buffer for every line of component in MDU */
  }

      for(sofs=0,current=temp->blist;current<temp->blist+temp->num;current++)
  {
    *current = MakeXBuffer(CFrame->BufferSize, wsize);
    (*current)->streamoffs = sofs;
    (*current)->iob = temp;
    (*current)->data_linelast = temp->linelastdefault;
    if (!temp->height || (current - temp->blist) < temp->height-1)
      {
        sofs += CFrame->Width[CScan->ci[index]]*wsize;
      }
  }
      CScan->Iob[index] = temp;
    }
}

/*BFUNC

PrintIob() is used to print the current input buffer to the stdio
stream.

EFUNC*/

void PrintIob()
{
  BEGIN("PrintIob")

  if (Iob)
    {
      printf("*** Iob ID: %p ***\n",(void*)Iob);
      printf("Number of Buffers: %d  Width: %d  Height: %d\n",
       Iob->num,Iob->width,Iob->height);
      printf("hpos: %d  vpos: %d  hor-freq: %d  ver-freq: %d\n",
       Iob->hpos,Iob->vpos,Iob->hor,Iob->ver);
      printf("filed: %d  flags: %d  BufferListId: %p\n",
       Iob->file,Iob->flags,(void*)Iob->blist);
    }
  else
    {
      printf("*** Iob ID: NULL ***\n");
    }
}

/*BFUNC

WriteXBuffer() writes out len elements from storage out to the buffer
structure specified.  This is can result in a multiple of len bytes
being written out depending on the element structure.

EFUNC*/

static void WriteXBuffer(len,storage,buffer)
     int len;
     int *storage;
     BUFFER *buffer;
{
  BEGIN("WriteXBuffer")
  int diff,wout;

  if (buffer->disable)
    {
      WHEREAMI();
      printf("Attempting to write to disabled buffer!\n");
    }
  /* printf("Writing:%d bytes\n",len);*/
  diff = buffer->size - (buffer->bptr - buffer->space); /* Find room left */
  diff = diff/buffer->wsize;                        /* Scale by element # */
  if(len > diff)
    {                                     /* Put as many elems in */
      WriteXBuffer(diff,storage,buffer);  /* If no room, then flush current */
      FlushBuffer(buffer);                /* buffer out to disk */
      len -= diff;
      storage += diff;
    }
  switch(buffer->wsize)   /* Because of compatibility differences between */
    {                     /* UNIX implementations, we are forced to do this */
    case 1:               /* explicit ordering of bytes... */
      while(len--)        /* Write the rest of the buffer out to the disk */
  {
    wout = *(storage++)<<PointTransform;
  *(buffer->bptr++) = (unsigned char) wout;
  }
      break;
    case 2:
      while(len--)        /* Write the rest of the buffer out to the disk */
  {
    wout = *(storage++)<<PointTransform;
    *(buffer->bptr++) = (unsigned char) (wout>>8)&0xff;
    *(buffer->bptr++) = (unsigned char) wout&0xff;
  }
      break;
    case 3:
      while(len--)        /* Write the rest of the buffer out to the disk */
  {
    wout = *(storage++)<<PointTransform;
    *(buffer->bptr++) = (unsigned char) (wout>>16)&0xff;
    *(buffer->bptr++) = (unsigned char) (wout>>8)&0xff;
    *(buffer->bptr++) = (unsigned char) wout&0xff;
  }
      break;
    case 4:
      while(len--)        /* Write the rest of the buffer out to the disk */
  {
    wout = *(storage++)<<PointTransform;
    *(buffer->bptr++) = (unsigned char) (wout>>24)&0xff;
    *(buffer->bptr++) = (unsigned char) (wout>>16)&0xff;
    *(buffer->bptr++) = (unsigned char) (wout>>8)&0xff;
    *(buffer->bptr++) = (unsigned char) wout&0xff;
  }
      break;
    default:
      WHEREAMI();
      printf("Illegal word size in characters %d.\n",buffer->wsize);
      exit(ERROR_BOUNDS);
      break;
    }
}

/*BFUNC

ReadXBuffer() is fetches len amount of elements into storage from the
buffer structure.  This may actually amount to an arbitrary number of
characters depending on the word size.

EFUNC*/

static void ReadXBuffer(len,storage,buffer)
     int len;
     int *storage;
     BUFFER *buffer;
{
  BEGIN("ReadXBuffer")
  int i,numchars,maxelem,rin;

  if (buffer->disable)
    {
      for(i=0;i<len;i++) *(storage++)=buffer->data_linelast;
      return;
    }

  numchars = len*buffer->wsize;
                                   /* The following command recurses because */
                                   /* it's slightly more efficient that way */
                                   /* when the probability of recursion is */
                                   /* negligible. */
  while (numchars > buffer->size)  /* We ask more than the buffer can handle */
    {                              /* Inefficient for small buffer sizes */
      maxelem = buffer->size/buffer->wsize;
      ReadXBuffer(maxelem, storage, buffer); /* Split up into several reads */
      storage += maxelem;
      len -= maxelem;
      numchars -= maxelem*buffer->wsize;
    }
  if(numchars > (buffer->tptr - buffer->bptr)) /* If we request > bytes */
    ReadResizeBuffer(numchars,buffer);       /* Read those bytes in */
  switch(buffer->wsize)                 /* Again, explicit input of bytes */
    {
    case 1:
      while(len--)                      /* Now copy over to storage */
  {
    rin = (int) *(buffer->bptr++);
    *(storage++) = rin >> PointTransform;
  }
      break;
    case 2:
      while(len--)                      /* Now copy over to storage */
  {
    rin = (((int)*(buffer->bptr++))<<8);
    rin |= *(buffer->bptr++);
    *(storage++) = rin >> PointTransform;
  }
      break;
    case 3:
      while(len--)                      /* Now copy over to storage */
  {
    rin = (((int)*(buffer->bptr++))<<16);
    rin |= (((int)*(buffer->bptr++))<<8);
    rin |= (*(buffer->bptr++));
    *(storage++) = rin >> PointTransform;
  }
      break;
    case 4:
      while(len--)                      /* Now copy over to storage */
  {
    rin = (((int)*(buffer->bptr++))<<24);
    rin |= (((int)*(buffer->bptr++))<<16);
    rin |= (((int)*(buffer->bptr++))<<8);
    rin |= (*(buffer->bptr++));
    *(storage++) = rin >> PointTransform;
  }
      break;
    default:
      WHEREAMI();
      printf("Illegal word size in characters %d.\n",buffer->wsize);
      exit(ERROR_BOUNDS);
      break;
    }
#ifdef IO_DEBUG
  WHEREAMI();
  printf("last read: %d",*(storage-1));
  printf("\n");
#endif
}

/*BFUNC

ReadResizeBuffer() reads len bytes from the stream and puts it
into the buffer.

EFUNC*/

static void ReadResizeBuffer(len,buffer)
     int len;
     BUFFER *buffer;
{
  BEGIN("ReadResizeBuffer")
  int retval,diff,location,amount;

  diff = buffer->tptr - buffer->bptr;        /* Find out the current usage */
  if (len > buffer->size-1)                  /* calculate if we can hold it */
    {
      WHEREAMI();
      printf("Length Request Too Large.\n");
      exit(ERROR_PARAMETER);
    }
#ifdef IO_DEBUG
  printf("SPACE: %x BPTR: %x DIFF: %d\n",buffer->space,buffer->bptr,diff);
  printf("ReadLseek %d\n",buffer->streamoffs+buffer->currentoffs);
#endif
  assert( diff >= 0 );
  memcpy(buffer->space,buffer->bptr,diff);   /* Move buffer down. */
  buffer->bptr = buffer->space;              /* Reset pointers. */
  buffer->tptr = buffer->space + diff;

  location = buffer->streamoffs+buffer->currentoffs;
  amount = buffer->size-(buffer->tptr - buffer->space);
  lseek(buffer->iob->file,location,SEEK_SET);
#ifdef IO_DEBUG
  printf("Read: Filed %d  Buf: %x  NBytes: %d\n",
   buffer->iob->file,buffer->tptr,amount);
#endif
  if ((retval = read(buffer->iob->file,      /* Do the read */
         buffer->tptr,
         amount)) < 0)
    {
      WHEREAMI();
      printf("Cannot Resize.\n");
      exit(ERROR_READ);
    }
#ifdef IO_DEBUG
  printf("ReadReturn numbytes %d\n",retval);
#endif
  buffer->tptr += retval;                   /* Alter pointers */
  buffer->currentoffs += retval;
}

/*BFUNC

FlushBuffer() saves the rest of the bytes in the buffer out to the
disk.

EFUNC*/

static void FlushBuffer(buffer)
     BUFFER *buffer;
{
  BEGIN("FlushBuffer")
  int retval;

#ifdef IO_DEBUG
  printf("WriteLseek %d\n",buffer->streamoffs+buffer->currentoffs);
#endif
  lseek(buffer->iob->file,buffer->streamoffs+buffer->currentoffs,SEEK_SET);
  if ((retval = write(buffer->iob->file,
          buffer->space,
          (buffer->bptr - buffer->space))) < 0)
    {
      WHEREAMI();
      printf("Cannot flush buffer.\n");
      exit(ERROR_WRITE);
    }
  buffer->currentoffs += (buffer->bptr - buffer->space);
  buffer->bptr = buffer->space;
}

/*BFUNC

ReadBlock() is used to get a block from the current Iob. This function
returns (for the JPEG case) 64 bytes in the store integer array.  It
is stored in row-major form; that is, the row index changes least
rapidly.

EFUNC*/

void ReadBlock(store)
     int *store;
{
  BEGIN("ReadBlock")
  int i,voffs;

  voffs = (Iob->vpos % Iob->ver)*BlockHeight;  /* Find current v offset*/
#ifdef IO_DEBUG
  for(i=0;i<BlockHeight;i++)
    {
      printf("%d Iob %x\n",i,Iob->blist[i]->Iob);
    }
#endif
  for(i=voffs;i<voffs+BlockHeight;i++)         /* Use voffs to index into */
    {                                          /* Buffer list of IOB */
#ifdef IO_DEBUG
      printf("%d Iob %x\n",i,Iob->blist[i]->Iob);
#endif
      ReadXBound(BlockWidth,store,Iob->blist[i]);  /* get blockwidth elms */
      store+=BlockWidth;                    /* Storage array & increment */
    }                                       /* by blockwidth */
  if ((++Iob->hpos % Iob->hor)==0)          /* Increment MDU block pos */
    {
      if ((++Iob->vpos % Iob->ver) == 0)
  {
    if (Iob->hpos < CScan->MDUWide*Iob->hor)
      {
        Iob->vpos -= Iob->ver;
      }
    else
      {
        Iob->hpos = 0;                /* If at end of raster width*/
        BlockMoveTo();                /* Reload buffers from start */
      }                               /* of next line. */
  }
      else
  {
    Iob->hpos -= Iob->hor;
  }
    }
}

/*BFUNC

WriteBlock() writes an array of data in the integer array pointed to
by store out to the driver specified by the IOB.  The integer array is
stored in row-major form, that is, the first row of (8) elements, the
second row of (8) elements....

EFUNC*/

void WriteBlock(store)
     int *store;
{
  BEGIN("WriteBlock")
  int i,voffs;

  voffs = (Iob->vpos % Iob->ver)*BlockHeight;  /* Find vertical buffer offs. */
  for(i=voffs;i<voffs+BlockHeight;i++)
    {
      if (!Iob->height || (((Iob->vpos/Iob->ver)*BlockHeight + i) <
         Iob->height))
  {
    WriteXBound(BlockWidth,store,Iob->blist[i]); /* write Block elms */
    store+=BlockWidth;                   /* Iob indexed by offset */
  }
    }
  if ((++Iob->hpos % Iob->hor)==0)             /* Increment block position */
    {                                          /* in MDU. */
      if ((++Iob->vpos % Iob->ver) == 0)
  {
    if (Iob->hpos < CScan->MDUWide*Iob->hor)
      {
        Iob->vpos -= Iob->ver;
      }
    else
      {
        Iob->hpos = 0;                  /* If at end of image (width) */
        FlushIob();                     /* Flush current IOB and */
        BlockMoveTo();                  /* Move to next lower MDU line */
      }
  }
      else
  {
    Iob->hpos -= Iob->hor;
  }
    }
}


/*BFUNC

BlockMoveTo() is used to move to a specific vertical and horizontal
location (block wise) specified by the current Iob. That means you set
the current Iob parameters and then call BlockMoveTo().

EFUNC*/

static void BlockMoveTo()
{
  BEGIN("BlockMoveTo")
  int i,vertical,horizontal;

  if (Loud > MUTE)
    {
      WHEREAMI();
      printf("%p  Moving To [Horizontal:Vertical] [%d:%d] \n",
       (void*)Iob,Iob->hpos,Iob->vpos);
    }
  horizontal =  Iob->hpos * BlockWidth;    /* Calculate actual */
  vertical = Iob->vpos * BlockHeight;      /* Pixel position */
  for(i=0;i<Iob->ver*BlockHeight;i++)
    {
      if (Iob->height)
  {
    vertical =
      ((vertical < Iob->height) ?
       vertical : Iob->height-1);
  }
      Iob->blist[i]->tptr =                /* Reset pointer space */
  Iob->blist[i]->bptr =              /* To show no contents */
    Iob->blist[i]->space;
      Iob->blist[i]->currentoffs = horizontal* Iob->wsize;/* reset h offset */
      Iob->blist[i]->streamoffs = vertical * Iob->width *
  Iob->wsize;                                       /* Reset v offset */
      vertical++;
    }
}

/*BFUNC

RewindIob() brings all the pointers to the start of the file. The reset
does not flush the buffers if writing.

EFUNC*/

void RewindIob()
{
  BEGIN("RewindIob")
  int i;

  switch(Iob->type)
    {
    case IOB_BLOCK:
      BlockWidth = BLOCKWIDTH;   /* Block width. */
      BlockHeight = BLOCKHEIGHT; /* Block height. */
      for(i=0;i<Iob->ver*BlockHeight;i++)
  {
    Iob->blist[i]->tptr =
      Iob->blist[i]->bptr =
        Iob->blist[i]->space;
    Iob->blist[i]->currentoffs = 0;
    Iob->blist[i]->streamoffs = i * Iob->width * Iob->wsize;
  }
      Iob->hpos = Iob->vpos = 0;
      break;
    case IOB_LINE:
      Iob->linelastdefault=(1<<(CFrame->DataPrecision-PointTransform-1));
      for(i= 0;i<Iob->ver+1;i++)
  {
    Iob->blist[i]->tptr =
      Iob->blist[i]->bptr =
        Iob->blist[i]->space;
    Iob->blist[i]->currentoffs = 0;
    if (!i)
      {
        Iob->blist[i]->streamoffs = 0;
        Iob->blist[i]->disable=1;
      }
    else
      {
        Iob->blist[i]->streamoffs = (i-1) * Iob->width * Iob->wsize;
        Iob->blist[i]->disable=0;
      }
    Iob->blist[i]->data_linelast = Iob->linelastdefault;
  }
      Iob->hpos = 0;
      Iob->vpos = -1;
      break;
    default:
      WHEREAMI();
      printf("Bad IOB type: %d\n",Iob->type);
      break;
    }
}

/*BFUNC

FlushIob() is used to flush all the buffers in the current Iob. This
is done at the conclusion of a write on the current buffers.

EFUNC*/

void FlushIob()
{
  BEGIN("FlushIob")
  int i;

  if (Loud > MUTE)
    printf("IOB: %p  Flushing buffers\n",(void*)Iob);
  switch(Iob->type)
    {
    case IOB_BLOCK:
      for(i=0;i<Iob->ver*BlockHeight;i++)
  FlushBuffer(Iob->blist[i]);
      break;
    case IOB_LINE:
      Iob->blist[0]->data_linelast=Iob->linelastdefault;
      for(i=1;i<Iob->ver+1;i++)
  {
    Iob->blist[i]->data_linelast=Iob->linelastdefault;
    FlushBuffer(Iob->blist[i]);
  }
      break;
    default:
      WHEREAMI();
      printf("Illegal IOB type: %d.\n",Iob->type);
      break;
    }
}

/*BFUNC

SeekEndIob() is used to seek the end of all the buffers in the current
Iob. This is done at the conclusion of a write, to avoid DNL problems.

EFUNC*/

void SeekEndIob()
{
  BEGIN("SeekEndIob")
  int size,tsize;
  static unsigned char Terminator[] = {0x80,0x00};

  size = lseek(Iob->file,0,SEEK_END);
  tsize = Iob->width*Iob->height*Iob->wsize;
  if (size !=  tsize)
    {
      WHEREAMI();
      printf("End not flush, making flush (actual: %d != target:%d)\n",
       size,tsize);

      if (size<tsize)
  {
    lseek(Iob->file,tsize-1,SEEK_SET);         /* Seek and terminate */
    write(Iob->file,Terminator,1);
  }
      else if (size > tsize)
  {
#ifdef NOTRUNCATE
    WHEREAMI();
    printf("file is too large, only first %d bytes valid\n",
     tsize);
#else
#ifdef WIN32
    chsize(Iob->file,tsize); /* no ftruncate on WIN32... */
#else
    ftruncate(Iob->file,tsize);                   /* simply truncate*/
#endif
#endif
  }
    }
}


/*BFUNC

CloseIob() is used to close the current Iob.

EFUNC*/

void CloseIob()
{
  BEGIN("CloseIob")

  if( !CFrame->tmpfile ) /* if file is closed we loose it for good */
{
  close(Iob->file);
}
}

/*BFUNC

ReadXBound() reads nelem elements of information from the specified
buffer.  It detects to see whether a load is necessary or not, or
whether the current buffer is out of the image width bounds.

EFUNC*/

static void ReadXBound(nelem,cstore,buffer)
     int nelem;
     int *cstore;
     BUFFER *buffer;
{
  BEGIN("ReadXBound")
  int i,diff;

  if ((diff = buffer->iob->width - TrueBufferPos(buffer)) <= nelem)
    {
#ifdef IO_DEBUG
      printf("ReadBound: Trailing Edge Detected. Diff: %d\n",diff);
#endif
      if (diff <= 0)
  {
    for(i=0;i<nelem;i++)       /* Pure outside bounds */
      *(cstore++) = buffer->overflow;
  }
      else
  {
    ReadXBuffer(diff,cstore,buffer);
    buffer->overflow = (unsigned int) cstore[diff-1];
    for(i=diff;i<nelem;i++)     /* Replicate to bounds */
      cstore[i] = cstore[i-1];
  }
    }
  else
    ReadXBuffer(nelem,cstore,buffer);
}

/*BFUNC

WriteXBound() writes the integer array input to the buffer. It checks
to see whether the bounds of the image width are exceeded, if so, the
excess information is ignored.

EFUNC*/

static void WriteXBound(nelem,cstore,buffer)
     int nelem;
     int *cstore;
     BUFFER *buffer;
{
  BEGIN("WriteXBound")
  int diff;

  if ((diff = buffer->iob->width - TrueBufferPos(buffer)) <= nelem)
    {                           /* Diff is balance to write to disk */
      if (diff > 0)             /* Write balance out to disk */
  WriteXBuffer(diff,cstore,buffer);
    }
  else                          /* If more than numberelem, then can put all */
    WriteXBuffer(nelem,cstore,buffer);       /* to the buffer. */
}

/*BFUNC

InstallIob() is used to install the Iob in the current scan as the
real Iob.

EFUNC*/

void InstallIob(index)
     int index;
{
  BEGIN("InstallIob")

  if (!(Iob = CScan->Iob[index]))
    {
      WHEREAMI();
      printf("Warning, NULL Iob installed.\n");
    }
}


/*BFUNC

TerminateFile() is a function that ensures that the entire file
defined by the Iob is properly flush with the filesize specifications.
This function is used when some fatal error occurs.

EFUNC*/


void TerminateFile()
{
  BEGIN("TerminateFile")
  int i,size;
  static unsigned char Terminator[] = {0x80,0x00};

  if (CFrame->GlobalHeight)
    {
      printf("> GH:%d  GW:%d  R:%d\n",
       CFrame->GlobalHeight,
       CFrame->GlobalWidth,
       CFrame->ResyncInterval);
      for(i=0;i<CScan->NumberComponents;i++)
  {
    if (CScan->Iob[i])
      {
        printf(">> C:%d  N:%s  H:%d  W:%d  hf:%d  vf:%d\n",
         CScan->ci[i],
         CFrame->ComponentFileName[CScan->ci[i]],
         CFrame->Height[CScan->ci[i]],
         CFrame->Width[CScan->ci[i]],
         CFrame->hf[CScan->ci[i]],
         CFrame->vf[CScan->ci[i]]);
        InstallIob(i);
        FlushIob();
        size = lseek(CScan->Iob[i]->file,0,SEEK_END);
        if (size !=
      CFrame->Width[CScan->ci[i]]*CFrame->Height[CScan->ci[i]]*
      CScan->Iob[i]->wsize)
    {                                      /* Terminate file */
      lseek(CScan->Iob[i]->file,           /* by seeking to end */
      (CFrame->Width[CScan->ci[i]]*  /* And writing byte */
       CFrame->Height[CScan->ci[i]]*
       CScan->Iob[i]->wsize)-1,      /* Making flush with */
      SEEK_SET);                           /* Original size  */
      write(CScan->Iob[i]->file,Terminator,1);
    }
      }
  }
    }
  else
    {
      WHEREAMI();
      printf("Unknown number of lines. Cannot flush file.\n");
    }
}


/*BFUNC

ReadLine() reads in the lines required by the lossless function.  The
array *store should be large enough to handle the line information
read.

In total, there should be (HORIZONTALFREQUENCY+1) * nelem
(VERTICALFREQUENCY+1) elements in the *store array.  This forms a
matrix with each line consisting of:

[PastPredictor 1 element]  nelem* [HORIZONTALFREQUENCY elements]

And there are (VERTICALFREQUENCY+1) of such lines in the matrix:

Previous line (2**Precision-1) if beyond specifications of window
Active line 1...
...
Active line VERTICALFREQUENCY...


EFUNC*/

void ReadLine(nelem,store)
     int nelem;
     int *store;
{
  BEGIN("ReadLine")
  int i;

  for(i=0;i<Iob->ver+1;i++)         /* Use voffs to index into */
    {                                          /* Buffer list of IOB */
#ifdef IO_DEBUG
      printf("%d Iob %x\n",i,Iob->blist[i]->Iob);
#endif
      *(store++)=Iob->blist[i]->data_linelast;
      ReadXBound(Iob->hor*nelem,store,Iob->blist[i]);
      store+=Iob->hor*nelem;
      Iob->blist[i]->data_linelast = *(store-1);
    }
  Iob->hpos += Iob->hor*nelem;
  if (Iob->hpos >= CScan->MDUWide*Iob->hor)
    {
      Iob->vpos += Iob->ver;
      Iob->hpos = 0;                /* If at end of raster width*/
      LineMoveTo();                 /* Reload buffers from start */
    }                               /* of next line. */
}

/*BFUNC

ReadPreambleLine() reads the first line of the *store array for the
WriteLine() companion command.  It reads it so that prediction can be
accomplished with minimum effort and storage.

This command is executed before decoding a particular line for the
prediction values; WriteLine() is called after the decoding is done.

EFUNC*/

void ReadPreambleLine(nelem,store)
     int nelem;
     int *store;
{
  BEGIN("ReadPreambleLine")
  int i;
  int preamblelength=1;

  for(i=0;i<Iob->ver+1;i++)         /* Use voffs to index into */
    {                                          /* Buffer list of IOB */
#ifdef IO_DEBUG
      printf("%d Iob %x\n",i,Iob->blist[i]->Iob);
#endif
      if (i<preamblelength)
  {
    *(store++)=Iob->blist[i]->data_linelast;
    ReadXBound(Iob->hor*nelem,store,Iob->blist[i]);
    store+=Iob->hor*nelem;
    Iob->blist[i]->data_linelast = *(store-1);
  }
      else
  {
    *(store) = Iob->blist[i]->data_linelast;
    store += Iob->hor*nelem+1;
  }
    }
}


/*BFUNC

WriteLine() is used to write a particular line out to the IOB.  The
line must be of the proper form in the array for this function to
work.

In total, there should be (HORIZONTALFREQUENCY+1) * nelem
(VERTICALFREQUENCY+1) elements in the *store array.  This forms a
matrix with each line consisting of:

[PastPredictor 1 element]  nelem* [HORIZONTALFREQUENCY elements]

And there are (VERTICALFREQUENCY+1) of such lines in the matrix:

Previous line (2**Precision-1) if beyond specifications of window
Active line 1...
...
Active line VERTICALFREQUENCY...


EFUNC*/

void WriteLine(nelem,store)
     int nelem;
     int *store;
{
  BEGIN("WriteLine")
  int i;

  store += Iob->hor*nelem+1;        /* Get rid of first line */
  for(i=1;i<Iob->ver+1;i++)         /* Use voffs to index into */
    {                               /* Buffer list of IOB */
#ifdef IO_DEBUG
      printf("WriteLine: %d  Store: %d Iobblist: %x\n",
       i,*(store+1),Iob->blist[i]);
#endif

      WriteXBound(Iob->hor*nelem,store+1,Iob->blist[i]);
      store+=(Iob->hor*nelem)+1;
      Iob->blist[i]->data_linelast = *(store-1);
    }
  Iob->hpos += Iob->hor*nelem;
  if (Iob->hpos >= CScan->MDUWide*Iob->hor)
    {
      Iob->vpos += Iob->ver;
      Iob->hpos = 0;                /* If at end of raster width*/
      FlushIob();                   /* Flush current IOB and */
      LineMoveTo();                 /* Reload buffers from start */
    }                               /* of next line. */
}

/*BFUNC

LineResetBuffers() resets all of the line buffers to the
(2\^DataPrecision-1) state.  The previous state is the default
prediction.  This commmand is used for resynchronization. The
implementation here does a trivial resetting.

EFUNC */

extern void LineResetBuffers()
{
  BEGIN("LineResetBuffers")
  int i;

  if (Iob->type!=IOB_LINE)
    {
      WHEREAMI();
      printf("Attempting to line reset a non-line buffer!\n");
      exit(ERROR_PARAMETER);
    }
  for(i=0;i<Iob->ver+1;i++)
    Iob->blist[i]->data_linelast = Iob->linelastdefault;
}

/*BFUNC

LineMoveTo() is used to move to a specific vertical and horizontal
location (line wise) specified by the current Iob. That means you set
the current Iob parameters and then call LineMoveTo().

EFUNC*/

static void LineMoveTo()
{
  BEGIN("LineMoveTo")
  int i,vertical,horizontal;

  if (Loud > MUTE)
    {
      WHEREAMI();
      printf("%p  Moving To [Horizontal:Vertical] [%d:%d] \n",
       (void*)Iob,Iob->hpos,Iob->vpos);
    }
  horizontal =  Iob->hpos;
  vertical = Iob->vpos;
  for(i=0;i<Iob->ver+1;i++)
    {                                     /* Reset last element read */
      if (vertical<0)
  {
    Iob->blist[i]->disable=1;
    continue;
  }
      Iob->blist[i]->disable=0;
      Iob->blist[i]->data_linelast=Iob->linelastdefault;
      if (Iob->height)
  {
    vertical =
      ((vertical < Iob->height) ?
       vertical : Iob->height-1);
  }
      Iob->blist[i]->tptr =                /* Reset pointer space */
  Iob->blist[i]->bptr =              /* To show no contents */
    Iob->blist[i]->space;
      Iob->blist[i]->currentoffs = horizontal* Iob->wsize; /* Reset h offset */
      Iob->blist[i]->streamoffs = vertical * Iob->width *
  Iob->wsize;                                        /* Reset v offset */
      vertical++;
    }
}



/*END*/
