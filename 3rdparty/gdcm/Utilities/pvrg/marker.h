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
marker.h

Some basic definitions of commonly occurring markers.

************************************************************
*/

#ifndef MARKER_DONE
#define MARKER_DONE

#define END_QUANTIZATION_TABLE 0xFF
#define END_CODE_TABLE 0xFF

#define MARKER_MARKER 0xff
#define MARKER_FIL 0xff

#define MARKER_SOI 0xd8
#define MARKER_EOI 0xd9
#define MARKER_SOS 0xda
#define MARKER_DQT 0xdb
#define MARKER_DNL 0xdc
#define MARKER_DRI 0xdd
#define MARKER_DHP 0xde
#define MARKER_EXP 0xdf

#define MARKER_DHT 0xc4

#define MARKER_SOF 0xc0
#define MARKER_RSC 0xd0
#define MARKER_APP 0xe0
#define MARKER_JPG 0xf0

#define MARKER_RSC_MASK 0xf8

int CheckMarker();

#endif
