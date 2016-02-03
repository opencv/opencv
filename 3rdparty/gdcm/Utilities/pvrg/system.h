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
system.h

This file contains the miscellaneous definitions for running
the JPEG coder.

************************************************************
*/

#ifndef SYSTEM_DONE
#define SYSTEM_DONE
/*#include <sys/file.h>*/
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#define IOB_BLOCK 0
#define IOB_LINE 1
#define IOB_OVERSAMPLEBLOCK 2

#define BUFFER struct io_buffer
#define IOBUF struct io_buffer_list

#define XHUFF struct huffman_standard_structure
#define EHUFF struct huffman_encoder
#define DHUFF struct huffman_decoder

BUFFER {
unsigned int overflow;   /* The last buffer character on line overflow */
int data_linelast;       /* The last element read out for line buffering */
int disable;             /* Stream is disabled! */
int wsize;               /* Element word size in characters */
int size;                /* Size of buffer in characters */
long currentoffs;        /* Current offset from left edge of image */
long streamoffs;         /* Stream offset (the pixel index of left edge) */
unsigned char *space;    /* Space is the raw buffer pointer */
unsigned char *bptr;     /* Current base pointer of buffer */
unsigned char *tptr;     /* Current top pointer of buffer */
IOBUF *iob;              /* References own IOB */
};


IOBUF {
int type;                     /* Iob type */
int num;                      /* Number of buffers */
int wsize;                    /* Element word size in characters */
int hpos;                     /* Current block position in image */
int vpos;
int hor;                      /* Sampling frequency */
int ver;
int width;                    /* Width and height of image */
int height;
int file;                     /* File descriptor */
int flags;                    /* File mode flags */
int linelastdefault;          /* Last line element default */
BUFFER **blist;               /* A list of buffers */
};

/* XHUFF contains all the information that needs be transmitted */
/* EHUFF and DHUFF are derivable from XHUFF */

XHUFF {
int bits[36];          /* Bit-length frequency (indexed on length  */
int huffval[257];      /* Huffman value index */
};

/* Encoder tables */

EHUFF {
int ehufco[257];      /* Encoder huffman code indexed on code word */
int ehufsi[257];      /* Encoder huffman code-size indexed on code word */
};

/* Decoder tables */

DHUFF {
int ml;               /* Maximum length */
int maxcode[36];      /* Max code for a given bit length -1 if no codes */
int mincode[36];      /* Min code for a given bit length */
int valptr[36];       /* First index (of min-code) for a given bit-length */
};


#endif
