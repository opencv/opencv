/* $Id: tif_fax3.h,v 1.13 2016-12-14 18:36:27 faxguy Exp $ */

/*
 * Copyright (c) 1990-1997 Sam Leffler
 * Copyright (c) 1991-1997 Silicon Graphics, Inc.
 *
 * Permission to use, copy, modify, distribute, and sell this software and 
 * its documentation for any purpose is hereby granted without fee, provided
 * that (i) the above copyright notices and this permission notice appear in
 * all copies of the software and related documentation, and (ii) the names of
 * Sam Leffler and Silicon Graphics may not be used in any advertising or
 * publicity relating to the software without the specific, prior written
 * permission of Sam Leffler and Silicon Graphics.
 * 
 * THE SOFTWARE IS PROVIDED "AS-IS" AND WITHOUT WARRANTY OF ANY KIND, 
 * EXPRESS, IMPLIED OR OTHERWISE, INCLUDING WITHOUT LIMITATION, ANY 
 * WARRANTY OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  
 * 
 * IN NO EVENT SHALL SAM LEFFLER OR SILICON GRAPHICS BE LIABLE FOR
 * ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
 * OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF 
 * LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE 
 * OF THIS SOFTWARE.
 */

#ifndef _FAX3_
#define	_FAX3_
/*
 * TIFF Library.
 *
 * CCITT Group 3 (T.4) and Group 4 (T.6) Decompression Support.
 *
 * Decoder support is derived, with permission, from the code
 * in Frank Cringle's viewfax program;
 *      Copyright (C) 1990, 1995  Frank D. Cringle.
 */
#include "tiff.h"

/*
 * To override the default routine used to image decoded
 * spans one can use the pseudo tag TIFFTAG_FAXFILLFUNC.
 * The routine must have the type signature given below;
 * for example:
 *
 * fillruns(unsigned char* buf, uint32* runs, uint32* erun, uint32 lastx)
 *
 * where buf is place to set the bits, runs is the array of b&w run
 * lengths (white then black), erun is the last run in the array, and
 * lastx is the width of the row in pixels.  Fill routines can assume
 * the run array has room for at least lastx runs and can overwrite
 * data in the run array as needed (e.g. to append zero runs to bring
 * the count up to a nice multiple).
 */
typedef void (*TIFFFaxFillFunc)(unsigned char*, uint32*, uint32*, uint32);

/*
 * The default run filler; made external for other decoders.
 */
#if defined(__cplusplus)
extern "C" {
#endif
extern void _TIFFFax3fillruns(unsigned char*, uint32*, uint32*, uint32);
#if defined(__cplusplus)
}
#endif


/* finite state machine codes */
#define S_Null     0
#define S_Pass     1
#define S_Horiz    2
#define S_V0       3
#define S_VR       4
#define S_VL       5
#define S_Ext      6
#define S_TermW    7
#define S_TermB    8
#define S_MakeUpW  9
#define S_MakeUpB  10
#define S_MakeUp   11
#define S_EOL      12

/* WARNING: do not change the layout of this structure as the HylaFAX software */
/* really depends on it. See http://bugzilla.maptools.org/show_bug.cgi?id=2636 */
typedef struct {                /* state table entry */
	unsigned char State;    /* see above */
	unsigned char Width;    /* width of code in bits */
	uint32 Param;           /* unsigned 32-bit run length in bits (holds on 16 bit actually, but cannot be changed. See above warning) */
} TIFFFaxTabEnt;

extern const TIFFFaxTabEnt TIFFFaxMainTable[];
extern const TIFFFaxTabEnt TIFFFaxWhiteTable[];
extern const TIFFFaxTabEnt TIFFFaxBlackTable[];

/*
 * The following macros define the majority of the G3/G4 decoder
 * algorithm using the state tables defined elsewhere.  To build
 * a decoder you need some setup code and some glue code. Note
 * that you may also need/want to change the way the NeedBits*
 * macros get input data if, for example, you know the data to be
 * decoded is properly aligned and oriented (doing so before running
 * the decoder can be a big performance win).
 *
 * Consult the decoder in the TIFF library for an idea of what you
 * need to define and setup to make use of these definitions.
 *
 * NB: to enable a debugging version of these macros define FAX3_DEBUG
 *     before including this file.  Trace output goes to stdout.
 */

#ifndef EndOfData
#define EndOfData()	(cp >= ep)
#endif
/*
 * Need <=8 or <=16 bits of input data.  Unlike viewfax we
 * cannot use/assume a word-aligned, properly bit swizzled
 * input data set because data may come from an arbitrarily
 * aligned, read-only source such as a memory-mapped file.
 * Note also that the viewfax decoder does not check for
 * running off the end of the input data buffer.  This is
 * possible for G3-encoded data because it prescans the input
 * data to count EOL markers, but can cause problems for G4
 * data.  In any event, we don't prescan and must watch for
 * running out of data since we can't permit the library to
 * scan past the end of the input data buffer.
 *
 * Finally, note that we must handle remaindered data at the end
 * of a strip specially.  The coder asks for a fixed number of
 * bits when scanning for the next code.  This may be more bits
 * than are actually present in the data stream.  If we appear
 * to run out of data but still have some number of valid bits
 * remaining then we makeup the requested amount with zeros and
 * return successfully.  If the returned data is incorrect then
 * we should be called again and get a premature EOF error;
 * otherwise we should get the right answer.
 */
#ifndef NeedBits8
#define NeedBits8(n,eoflab) do {					\
    if (BitsAvail < (n)) {						\
	if (EndOfData()) {						\
	    if (BitsAvail == 0)			/* no valid bits */	\
		goto eoflab;						\
	    BitsAvail = (n);			/* pad with zeros */	\
	} else {							\
	    BitAcc |= ((uint32) bitmap[*cp++])<<BitsAvail;		\
	    BitsAvail += 8;						\
	}								\
    }									\
} while (0)
#endif
#ifndef NeedBits16
#define NeedBits16(n,eoflab) do {					\
    if (BitsAvail < (n)) {						\
	if (EndOfData()) {						\
	    if (BitsAvail == 0)			/* no valid bits */	\
		goto eoflab;						\
	    BitsAvail = (n);			/* pad with zeros */	\
	} else {							\
	    BitAcc |= ((uint32) bitmap[*cp++])<<BitsAvail;		\
	    if ((BitsAvail += 8) < (n)) {				\
		if (EndOfData()) {					\
		    /* NB: we know BitsAvail is non-zero here */	\
		    BitsAvail = (n);		/* pad with zeros */	\
		} else {						\
		    BitAcc |= ((uint32) bitmap[*cp++])<<BitsAvail;	\
		    BitsAvail += 8;					\
		}							\
	    }								\
	}								\
    }									\
} while (0)
#endif
#define GetBits(n)	(BitAcc & ((1<<(n))-1))
#define ClrBits(n) do {							\
    BitsAvail -= (n);							\
    BitAcc >>= (n);							\
} while (0)

#ifdef FAX3_DEBUG
static const char* StateNames[] = {
    "Null   ",
    "Pass   ",
    "Horiz  ",
    "V0     ",
    "VR     ",
    "VL     ",
    "Ext    ",
    "TermW  ",
    "TermB  ",
    "MakeUpW",
    "MakeUpB",
    "MakeUp ",
    "EOL    ",
};
#define DEBUG_SHOW putchar(BitAcc & (1 << t) ? '1' : '0')
#define LOOKUP8(wid,tab,eoflab) do {					\
    int t;								\
    NeedBits8(wid,eoflab);						\
    TabEnt = tab + GetBits(wid);					\
    printf("%08lX/%d: %s%5d\t", (long) BitAcc, BitsAvail,		\
	   StateNames[TabEnt->State], TabEnt->Param);			\
    for (t = 0; t < TabEnt->Width; t++)					\
	DEBUG_SHOW;							\
    putchar('\n');							\
    fflush(stdout);							\
    ClrBits(TabEnt->Width);						\
} while (0)
#define LOOKUP16(wid,tab,eoflab) do {					\
    int t;								\
    NeedBits16(wid,eoflab);						\
    TabEnt = tab + GetBits(wid);					\
    printf("%08lX/%d: %s%5d\t", (long) BitAcc, BitsAvail,		\
	   StateNames[TabEnt->State], TabEnt->Param);			\
    for (t = 0; t < TabEnt->Width; t++)					\
	DEBUG_SHOW;							\
    putchar('\n');							\
    fflush(stdout);							\
    ClrBits(TabEnt->Width);						\
} while (0)

#define SETVALUE(x) do {							\
    *pa++ = RunLength + (x);						\
    printf("SETVALUE: %d\t%d\n", RunLength + (x), a0);			\
    a0 += x;								\
    RunLength = 0;							\
} while (0)
#else
#define LOOKUP8(wid,tab,eoflab) do {					\
    NeedBits8(wid,eoflab);						\
    TabEnt = tab + GetBits(wid);					\
    ClrBits(TabEnt->Width);						\
} while (0)
#define LOOKUP16(wid,tab,eoflab) do {					\
    NeedBits16(wid,eoflab);						\
    TabEnt = tab + GetBits(wid);					\
    ClrBits(TabEnt->Width);						\
} while (0)

/*
 * Append a run to the run length array for the
 * current row and reset decoding state.
 */
#define SETVALUE(x) do {							\
    *pa++ = RunLength + (x);						\
    a0 += (x);								\
    RunLength = 0;							\
} while (0)
#endif

/*
 * Synchronize input decoding at the start of each
 * row by scanning for an EOL (if appropriate) and
 * skipping any trash data that might be present
 * after a decoding error.  Note that the decoding
 * done elsewhere that recognizes an EOL only consumes
 * 11 consecutive zero bits.  This means that if EOLcnt
 * is non-zero then we still need to scan for the final flag
 * bit that is part of the EOL code.
 */
#define	SYNC_EOL(eoflab) do {						\
    if (EOLcnt == 0) {							\
	for (;;) {							\
	    NeedBits16(11,eoflab);					\
	    if (GetBits(11) == 0)					\
		break;							\
	    ClrBits(1);							\
	}								\
    }									\
    for (;;) {								\
	NeedBits8(8,eoflab);						\
	if (GetBits(8))							\
	    break;							\
	ClrBits(8);							\
    }									\
    while (GetBits(1) == 0)						\
	ClrBits(1);							\
    ClrBits(1);				/* EOL bit */			\
    EOLcnt = 0;				/* reset EOL counter/flag */	\
} while (0)

/*
 * Cleanup the array of runs after decoding a row.
 * We adjust final runs to insure the user buffer is not
 * overwritten and/or undecoded area is white filled.
 */
#define	CLEANUP_RUNS() do {						\
    if (RunLength)							\
	SETVALUE(0);							\
    if (a0 != lastx) {							\
	badlength(a0, lastx);						\
	while (a0 > lastx && pa > thisrun)				\
	    a0 -= *--pa;						\
	if (a0 < lastx) {						\
	    if (a0 < 0)							\
		a0 = 0;							\
	    if ((pa-thisrun)&1)						\
		SETVALUE(0);						\
	    SETVALUE(lastx - a0);						\
	} else if (a0 > lastx) {					\
	    SETVALUE(lastx);						\
	    SETVALUE(0);							\
	}								\
    }									\
} while (0)

/*
 * Decode a line of 1D-encoded data.
 *
 * The line expanders are written as macros so that they can be reused
 * but still have direct access to the local variables of the "calling"
 * function.
 *
 * Note that unlike the original version we have to explicitly test for
 * a0 >= lastx after each black/white run is decoded.  This is because
 * the original code depended on the input data being zero-padded to
 * insure the decoder recognized an EOL before running out of data.
 */
#define EXPAND1D(eoflab) do {						\
    for (;;) {								\
	for (;;) {							\
	    LOOKUP16(12, TIFFFaxWhiteTable, eof1d);			\
	    switch (TabEnt->State) {					\
	    case S_EOL:							\
		EOLcnt = 1;						\
		goto done1d;						\
	    case S_TermW:						\
		SETVALUE(TabEnt->Param);					\
		goto doneWhite1d;					\
	    case S_MakeUpW:						\
	    case S_MakeUp:						\
		a0 += TabEnt->Param;					\
		RunLength += TabEnt->Param;				\
		break;							\
	    default:							\
		unexpected("WhiteTable", a0);				\
		goto done1d;						\
	    }								\
	}								\
    doneWhite1d:							\
	if (a0 >= lastx)						\
	    goto done1d;						\
	for (;;) {							\
	    LOOKUP16(13, TIFFFaxBlackTable, eof1d);			\
	    switch (TabEnt->State) {					\
	    case S_EOL:							\
		EOLcnt = 1;						\
		goto done1d;						\
	    case S_TermB:						\
		SETVALUE(TabEnt->Param);					\
		goto doneBlack1d;					\
	    case S_MakeUpB:						\
	    case S_MakeUp:						\
		a0 += TabEnt->Param;					\
		RunLength += TabEnt->Param;				\
		break;							\
	    default:							\
		unexpected("BlackTable", a0);				\
		goto done1d;						\
	    }								\
	}								\
    doneBlack1d:							\
	if (a0 >= lastx)						\
	    goto done1d;						\
        if( *(pa-1) == 0 && *(pa-2) == 0 )				\
            pa -= 2;                                                    \
    }									\
eof1d:									\
    prematureEOF(a0);							\
    CLEANUP_RUNS();							\
    goto eoflab;							\
done1d:									\
    CLEANUP_RUNS();							\
} while (0)

/*
 * Update the value of b1 using the array
 * of runs for the reference line.
 */
#define CHECK_b1 do {							\
    if (pa != thisrun) while (b1 <= a0 && b1 < lastx) {			\
	b1 += pb[0] + pb[1];						\
	pb += 2;							\
    }									\
} while (0)

/*
 * Expand a row of 2D-encoded data.
 */
#define EXPAND2D(eoflab) do {						\
    while (a0 < lastx) {						\
	LOOKUP8(7, TIFFFaxMainTable, eof2d);				\
	switch (TabEnt->State) {					\
	case S_Pass:							\
	    CHECK_b1;							\
	    b1 += *pb++;						\
	    RunLength += b1 - a0;					\
	    a0 = b1;							\
	    b1 += *pb++;						\
	    break;							\
	case S_Horiz:							\
	    if ((pa-thisrun)&1) {					\
		for (;;) {	/* black first */			\
		    LOOKUP16(13, TIFFFaxBlackTable, eof2d);		\
		    switch (TabEnt->State) {				\
		    case S_TermB:					\
			SETVALUE(TabEnt->Param);				\
			goto doneWhite2da;				\
		    case S_MakeUpB:					\
		    case S_MakeUp:					\
			a0 += TabEnt->Param;				\
			RunLength += TabEnt->Param;			\
			break;						\
		    default:						\
			goto badBlack2d;				\
		    }							\
		}							\
	    doneWhite2da:;						\
		for (;;) {	/* then white */			\
		    LOOKUP16(12, TIFFFaxWhiteTable, eof2d);		\
		    switch (TabEnt->State) {				\
		    case S_TermW:					\
			SETVALUE(TabEnt->Param);				\
			goto doneBlack2da;				\
		    case S_MakeUpW:					\
		    case S_MakeUp:					\
			a0 += TabEnt->Param;				\
			RunLength += TabEnt->Param;			\
			break;						\
		    default:						\
			goto badWhite2d;				\
		    }							\
		}							\
	    doneBlack2da:;						\
	    } else {							\
		for (;;) {	/* white first */			\
		    LOOKUP16(12, TIFFFaxWhiteTable, eof2d);		\
		    switch (TabEnt->State) {				\
		    case S_TermW:					\
			SETVALUE(TabEnt->Param);				\
			goto doneWhite2db;				\
		    case S_MakeUpW:					\
		    case S_MakeUp:					\
			a0 += TabEnt->Param;				\
			RunLength += TabEnt->Param;			\
			break;						\
		    default:						\
			goto badWhite2d;				\
		    }							\
		}							\
	    doneWhite2db:;						\
		for (;;) {	/* then black */			\
		    LOOKUP16(13, TIFFFaxBlackTable, eof2d);		\
		    switch (TabEnt->State) {				\
		    case S_TermB:					\
			SETVALUE(TabEnt->Param);				\
			goto doneBlack2db;				\
		    case S_MakeUpB:					\
		    case S_MakeUp:					\
			a0 += TabEnt->Param;				\
			RunLength += TabEnt->Param;			\
			break;						\
		    default:						\
			goto badBlack2d;				\
		    }							\
		}							\
	    doneBlack2db:;						\
	    }								\
	    CHECK_b1;							\
	    break;							\
	case S_V0:							\
	    CHECK_b1;							\
	    SETVALUE(b1 - a0);						\
	    b1 += *pb++;						\
	    break;							\
	case S_VR:							\
	    CHECK_b1;							\
	    SETVALUE(b1 - a0 + TabEnt->Param);				\
	    b1 += *pb++;						\
	    break;							\
	case S_VL:							\
	    CHECK_b1;							\
	    if (b1 <= (int) (a0 + TabEnt->Param)) {			\
		if (b1 < (int) (a0 + TabEnt->Param) || pa != thisrun) {	\
		    unexpected("VL", a0);				\
		    goto eol2d;						\
		}							\
	    }								\
	    SETVALUE(b1 - a0 - TabEnt->Param);				\
	    b1 -= *--pb;						\
	    break;							\
	case S_Ext:							\
	    *pa++ = lastx - a0;						\
	    extension(a0);						\
	    goto eol2d;							\
	case S_EOL:							\
	    *pa++ = lastx - a0;						\
	    NeedBits8(4,eof2d);						\
	    if (GetBits(4))						\
		unexpected("EOL", a0);					\
            ClrBits(4);                                                 \
	    EOLcnt = 1;							\
	    goto eol2d;							\
	default:							\
	badMain2d:							\
	    unexpected("MainTable", a0);				\
	    goto eol2d;							\
	badBlack2d:							\
	    unexpected("BlackTable", a0);				\
	    goto eol2d;							\
	badWhite2d:							\
	    unexpected("WhiteTable", a0);				\
	    goto eol2d;							\
	eof2d:								\
	    prematureEOF(a0);						\
	    CLEANUP_RUNS();						\
	    goto eoflab;						\
	}								\
    }									\
    if (RunLength) {							\
	if (RunLength + a0 < lastx) {					\
	    /* expect a final V0 */					\
	    NeedBits8(1,eof2d);						\
	    if (!GetBits(1))						\
		goto badMain2d;						\
	    ClrBits(1);							\
	}								\
	SETVALUE(0);							\
    }									\
eol2d:									\
    CLEANUP_RUNS();							\
} while (0)
#endif /* _FAX3_ */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
