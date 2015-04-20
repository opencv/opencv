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
stream.h

This file contains simple macros for better access to the direct
stream.

************************************************************
*/

#ifndef STREAM_DONE
#define STREAM_DONE

#define bputw(word) {bputc((word>>8)&0xff); bputc(word&0xff);}
#define bputn(nybbleh,nybblel) {bputc(((nybbleh&0x0f)<<4)|(nybblel&0x0f));}
#define hinyb(val) ((val>>4)&0x0f)
#define lonyb(val) (val & 0x0f)
int bgetw();
int brtell();
int brseek(int offset,int ptr);

#endif
