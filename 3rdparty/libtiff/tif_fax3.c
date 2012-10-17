/* $Id: tif_fax3.c,v 1.73 2012-06-13 00:27:20 fwarmerdam Exp $ */

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

#include "tiffiop.h"
#ifdef CCITT_SUPPORT
/*
 * TIFF Library.
 *
 * CCITT Group 3 (T.4) and Group 4 (T.6) Compression Support.
 *
 * This file contains support for decoding and encoding TIFF
 * compression algorithms 2, 3, 4, and 32771.
 *
 * Decoder support is derived, with permission, from the code
 * in Frank Cringle's viewfax program;
 *      Copyright (C) 1990, 1995  Frank D. Cringle.
 */
#include "tif_fax3.h"
#define	G3CODES
#include "t4.h"
#include <stdio.h>

/*
 * Compression+decompression state blocks are
 * derived from this ``base state'' block.
 */
typedef struct {
    int      rw_mode;                /* O_RDONLY for decode, else encode */
    int      mode;                   /* operating mode */
    tmsize_t rowbytes;               /* bytes in a decoded scanline */
    uint32   rowpixels;              /* pixels in a scanline */

    uint16   cleanfaxdata;           /* CleanFaxData tag */
    uint32   badfaxrun;              /* BadFaxRun tag */
    uint32   badfaxlines;            /* BadFaxLines tag */
    uint32   groupoptions;           /* Group 3/4 options tag */

    TIFFVGetMethod  vgetparent;      /* super-class method */
    TIFFVSetMethod  vsetparent;      /* super-class method */
    TIFFPrintMethod printdir;        /* super-class method */
} Fax3BaseState;
#define	Fax3State(tif)		((Fax3BaseState*) (tif)->tif_data)

typedef enum { G3_1D, G3_2D } Ttag;
typedef struct {
    Fax3BaseState b;

    /* Decoder state info */
    const unsigned char* bitmap;	/* bit reversal table */
    uint32	data;			/* current i/o byte/word */
    int	bit;			/* current i/o bit in byte */
    int	EOLcnt;			/* count of EOL codes recognized */
    TIFFFaxFillFunc fill;		/* fill routine */
    uint32*	runs;			/* b&w runs for current/previous row */
    uint32*	refruns;		/* runs for reference line */
    uint32*	curruns;		/* runs for current line */

    /* Encoder state info */
    Ttag    tag;			/* encoding state */
    unsigned char*	refline;	/* reference line for 2d decoding */
    int	k;			/* #rows left that can be 2d encoded */
    int	maxk;			/* max #rows that can be 2d encoded */

    int line;
} Fax3CodecState;
#define DecoderState(tif) ((Fax3CodecState*) Fax3State(tif))
#define EncoderState(tif) ((Fax3CodecState*) Fax3State(tif))

#define is2DEncoding(sp) (sp->b.groupoptions & GROUP3OPT_2DENCODING)
#define isAligned(p,t) ((((size_t)(p)) & (sizeof (t)-1)) == 0)

/*
 * Group 3 and Group 4 Decoding.
 */

/*
 * These macros glue the TIFF library state to
 * the state expected by Frank's decoder.
 */
#define	DECLARE_STATE(tif, sp, mod)					\
    static const char module[] = mod;					\
    Fax3CodecState* sp = DecoderState(tif);				\
    int a0;				/* reference element */		\
    int lastx = sp->b.rowpixels;	/* last element in row */	\
    uint32 BitAcc;			/* bit accumulator */		\
    int BitsAvail;			/* # valid bits in BitAcc */	\
    int RunLength;			/* length of current run */	\
    unsigned char* cp;			/* next byte of input data */	\
    unsigned char* ep;			/* end of input data */		\
    uint32* pa;				/* place to stuff next run */	\
    uint32* thisrun;			/* current row's run array */	\
    int EOLcnt;				/* # EOL codes recognized */	\
    const unsigned char* bitmap = sp->bitmap;	/* input data bit reverser */	\
    const TIFFFaxTabEnt* TabEnt
#define	DECLARE_STATE_2D(tif, sp, mod)					\
    DECLARE_STATE(tif, sp, mod);					\
    int b1;				/* next change on prev line */	\
    uint32* pb				/* next run in reference line */\
/*
 * Load any state that may be changed during decoding.
 */
#define	CACHE_STATE(tif, sp) do {					\
    BitAcc = sp->data;							\
    BitsAvail = sp->bit;						\
    EOLcnt = sp->EOLcnt;						\
    cp = (unsigned char*) tif->tif_rawcp;				\
    ep = cp + tif->tif_rawcc;						\
} while (0)
/*
 * Save state possibly changed during decoding.
 */
#define	UNCACHE_STATE(tif, sp) do {					\
    sp->bit = BitsAvail;						\
    sp->data = BitAcc;							\
    sp->EOLcnt = EOLcnt;						\
    tif->tif_rawcc -= (tmsize_t)((uint8*) cp - tif->tif_rawcp);		\
    tif->tif_rawcp = (uint8*) cp;					\
} while (0)

/*
 * Setup state for decoding a strip.
 */
static int
Fax3PreDecode(TIFF* tif, uint16 s)
{
    Fax3CodecState* sp = DecoderState(tif);

    (void) s;
    assert(sp != NULL);
    sp->bit = 0;			/* force initial read */
    sp->data = 0;
    sp->EOLcnt = 0;			/* force initial scan for EOL */
    /*
     * Decoder assumes lsb-to-msb bit order.  Note that we select
     * this here rather than in Fax3SetupState so that viewers can
     * hold the image open, fiddle with the FillOrder tag value,
     * and then re-decode the image.  Otherwise they'd need to close
     * and open the image to get the state reset.
     */
    sp->bitmap =
        TIFFGetBitRevTable(tif->tif_dir.td_fillorder != FILLORDER_LSB2MSB);
    if (sp->refruns) {		/* init reference line to white */
        sp->refruns[0] = (uint32) sp->b.rowpixels;
        sp->refruns[1] = 0;
    }
    sp->line = 0;
    return (1);
}

/*
 * Routine for handling various errors/conditions.
 * Note how they are "glued into the decoder" by
 * overriding the definitions used by the decoder.
 */

static void
Fax3Unexpected(const char* module, TIFF* tif, uint32 line, uint32 a0)
{
    TIFFErrorExt(tif->tif_clientdata, module, "Bad code word at line %u of %s %u (x %u)",
        line, isTiled(tif) ? "tile" : "strip",
        (isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip),
        a0);
}
#define	unexpected(table, a0)	Fax3Unexpected(module, tif, sp->line, a0)

static void
Fax3Extension(const char* module, TIFF* tif, uint32 line, uint32 a0)
{
    TIFFErrorExt(tif->tif_clientdata, module,
        "Uncompressed data (not supported) at line %u of %s %u (x %u)",
        line, isTiled(tif) ? "tile" : "strip",
        (isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip),
        a0);
}
#define	extension(a0)	Fax3Extension(module, tif, sp->line, a0)

static void
Fax3BadLength(const char* module, TIFF* tif, uint32 line, uint32 a0, uint32 lastx)
{
    TIFFWarningExt(tif->tif_clientdata, module, "%s at line %u of %s %u (got %u, expected %u)",
        a0 < lastx ? "Premature EOL" : "Line length mismatch",
        line, isTiled(tif) ? "tile" : "strip",
        (isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip),
        a0, lastx);
}
#define	badlength(a0,lastx)	Fax3BadLength(module, tif, sp->line, a0, lastx)

static void
Fax3PrematureEOF(const char* module, TIFF* tif, uint32 line, uint32 a0)
{
    TIFFWarningExt(tif->tif_clientdata, module, "Premature EOF at line %u of %s %u (x %u)",
        line, isTiled(tif) ? "tile" : "strip",
        (isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip),
        a0);
}
#define	prematureEOF(a0)	Fax3PrematureEOF(module, tif, sp->line, a0)

#define	Nop

/*
 * Decode the requested amount of G3 1D-encoded data.
 */
static int
Fax3Decode1D(TIFF* tif, uint8* buf, tmsize_t occ, uint16 s)
{
    DECLARE_STATE(tif, sp, "Fax3Decode1D");
    (void) s;
    if (occ % sp->b.rowbytes)
    {
        TIFFErrorExt(tif->tif_clientdata, module, "Fractional scanlines cannot be read");
        return (-1);
    }
    CACHE_STATE(tif, sp);
    thisrun = sp->curruns;
    while (occ > 0) {
        a0 = 0;
        RunLength = 0;
        pa = thisrun;
#ifdef FAX3_DEBUG
        printf("\nBitAcc=%08X, BitsAvail = %d\n", BitAcc, BitsAvail);
        printf("-------------------- %d\n", tif->tif_row);
        fflush(stdout);
#endif
        SYNC_EOL(EOF1D);
        EXPAND1D(EOF1Da);
        (*sp->fill)(buf, thisrun, pa, lastx);
        buf += sp->b.rowbytes;
        occ -= sp->b.rowbytes;
        sp->line++;
        continue;
    EOF1D:				/* premature EOF */
        CLEANUP_RUNS();
    EOF1Da:				/* premature EOF */
        (*sp->fill)(buf, thisrun, pa, lastx);
        UNCACHE_STATE(tif, sp);
        return (-1);
    }
    UNCACHE_STATE(tif, sp);
    return (1);
}

#define	SWAP(t,a,b)	{ t x; x = (a); (a) = (b); (b) = x; }
/*
 * Decode the requested amount of G3 2D-encoded data.
 */
static int
Fax3Decode2D(TIFF* tif, uint8* buf, tmsize_t occ, uint16 s)
{
    DECLARE_STATE_2D(tif, sp, "Fax3Decode2D");
    int is1D;			/* current line is 1d/2d-encoded */
    (void) s;
    if (occ % sp->b.rowbytes)
    {
        TIFFErrorExt(tif->tif_clientdata, module, "Fractional scanlines cannot be read");
        return (-1);
    }
    CACHE_STATE(tif, sp);
    while (occ > 0) {
        a0 = 0;
        RunLength = 0;
        pa = thisrun = sp->curruns;
#ifdef FAX3_DEBUG
        printf("\nBitAcc=%08X, BitsAvail = %d EOLcnt = %d",
            BitAcc, BitsAvail, EOLcnt);
#endif
        SYNC_EOL(EOF2D);
        NeedBits8(1, EOF2D);
        is1D = GetBits(1);	/* 1D/2D-encoding tag bit */
        ClrBits(1);
#ifdef FAX3_DEBUG
        printf(" %s\n-------------------- %d\n",
            is1D ? "1D" : "2D", tif->tif_row);
        fflush(stdout);
#endif
        pb = sp->refruns;
        b1 = *pb++;
        if (is1D)
            EXPAND1D(EOF2Da);
        else
            EXPAND2D(EOF2Da);
        (*sp->fill)(buf, thisrun, pa, lastx);
        SETVALUE(0);		/* imaginary change for reference */
        SWAP(uint32*, sp->curruns, sp->refruns);
        buf += sp->b.rowbytes;
        occ -= sp->b.rowbytes;
        sp->line++;
        continue;
    EOF2D:				/* premature EOF */
        CLEANUP_RUNS();
    EOF2Da:				/* premature EOF */
        (*sp->fill)(buf, thisrun, pa, lastx);
        UNCACHE_STATE(tif, sp);
        return (-1);
    }
    UNCACHE_STATE(tif, sp);
    return (1);
}
#undef SWAP

/*
 * The ZERO & FILL macros must handle spans < 2*sizeof(long) bytes.
 * For machines with 64-bit longs this is <16 bytes; otherwise
 * this is <8 bytes.  We optimize the code here to reflect the
 * machine characteristics.
 */
#if SIZEOF_UNSIGNED_LONG == 8
# define FILL(n, cp)							    \
    switch (n) {							    \
    case 15:(cp)[14] = 0xff; case 14:(cp)[13] = 0xff; case 13: (cp)[12] = 0xff;\
    case 12:(cp)[11] = 0xff; case 11:(cp)[10] = 0xff; case 10: (cp)[9] = 0xff;\
    case  9: (cp)[8] = 0xff; case  8: (cp)[7] = 0xff; case  7: (cp)[6] = 0xff;\
    case  6: (cp)[5] = 0xff; case  5: (cp)[4] = 0xff; case  4: (cp)[3] = 0xff;\
    case  3: (cp)[2] = 0xff; case  2: (cp)[1] = 0xff;			      \
    case  1: (cp)[0] = 0xff; (cp) += (n); case 0:  ;			      \
    }
# define ZERO(n, cp)							\
    switch (n) {							\
    case 15:(cp)[14] = 0; case 14:(cp)[13] = 0; case 13: (cp)[12] = 0;	\
    case 12:(cp)[11] = 0; case 11:(cp)[10] = 0; case 10: (cp)[9] = 0;	\
    case  9: (cp)[8] = 0; case  8: (cp)[7] = 0; case  7: (cp)[6] = 0;	\
    case  6: (cp)[5] = 0; case  5: (cp)[4] = 0; case  4: (cp)[3] = 0;	\
    case  3: (cp)[2] = 0; case  2: (cp)[1] = 0;				\
    case  1: (cp)[0] = 0; (cp) += (n); case 0:  ;			\
    }
#else
# define FILL(n, cp)							    \
    switch (n) {							    \
    case 7: (cp)[6] = 0xff; case 6: (cp)[5] = 0xff; case 5: (cp)[4] = 0xff; \
    case 4: (cp)[3] = 0xff; case 3: (cp)[2] = 0xff; case 2: (cp)[1] = 0xff; \
    case 1: (cp)[0] = 0xff; (cp) += (n); case 0:  ;			    \
    }
# define ZERO(n, cp)							\
    switch (n) {							\
    case 7: (cp)[6] = 0; case 6: (cp)[5] = 0; case 5: (cp)[4] = 0;	\
    case 4: (cp)[3] = 0; case 3: (cp)[2] = 0; case 2: (cp)[1] = 0;	\
    case 1: (cp)[0] = 0; (cp) += (n); case 0:  ;			\
    }
#endif

/*
 * Bit-fill a row according to the white/black
 * runs generated during G3/G4 decoding.
 */
void
_TIFFFax3fillruns(unsigned char* buf, uint32* runs, uint32* erun, uint32 lastx)
{
    static const unsigned char _fillmasks[] =
        { 0x00, 0x80, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe, 0xff };
    unsigned char* cp;
    uint32 x, bx, run;
    int32 n, nw;
    long* lp;

    if ((erun-runs)&1)
        *erun++ = 0;
    x = 0;
    for (; runs < erun; runs += 2) {
        run = runs[0];
        if (x+run > lastx || run > lastx )
        run = runs[0] = (uint32) (lastx - x);
        if (run) {
        cp = buf + (x>>3);
        bx = x&7;
        if (run > 8-bx) {
            if (bx) {			/* align to byte boundary */
            *cp++ &= 0xff << (8-bx);
            run -= 8-bx;
            }
            if( (n = run >> 3) != 0 ) {	/* multiple bytes to fill */
            if ((n/sizeof (long)) > 1) {
                /*
                 * Align to longword boundary and fill.
                 */
                for (; n && !isAligned(cp, long); n--)
                    *cp++ = 0x00;
                lp = (long*) cp;
                nw = (int32)(n / sizeof (long));
                n -= nw * sizeof (long);
                do {
                    *lp++ = 0L;
                } while (--nw);
                cp = (unsigned char*) lp;
            }
            ZERO(n, cp);
            run &= 7;
            }
            if (run)
            cp[0] &= 0xff >> run;
        } else
            cp[0] &= ~(_fillmasks[run]>>bx);
        x += runs[0];
        }
        run = runs[1];
        if (x+run > lastx || run > lastx )
        run = runs[1] = lastx - x;
        if (run) {
        cp = buf + (x>>3);
        bx = x&7;
        if (run > 8-bx) {
            if (bx) {			/* align to byte boundary */
            *cp++ |= 0xff >> bx;
            run -= 8-bx;
            }
            if( (n = run>>3) != 0 ) {	/* multiple bytes to fill */
            if ((n/sizeof (long)) > 1) {
                /*
                 * Align to longword boundary and fill.
                 */
                for (; n && !isAligned(cp, long); n--)
                *cp++ = 0xff;
                lp = (long*) cp;
                nw = (int32)(n / sizeof (long));
                n -= nw * sizeof (long);
                do {
                *lp++ = -1L;
                } while (--nw);
                cp = (unsigned char*) lp;
            }
            FILL(n, cp);
            run &= 7;
            }
            if (run)
            cp[0] |= 0xff00 >> run;
        } else
            cp[0] |= _fillmasks[run]>>bx;
        x += runs[1];
        }
    }
    assert(x == lastx);
}
#undef	ZERO
#undef	FILL

static int
Fax3FixupTags(TIFF* tif)
{
    (void) tif;
    return (1);
}

/*
 * Setup G3/G4-related compression/decompression state
 * before data is processed.  This routine is called once
 * per image -- it sets up different state based on whether
 * or not decoding or encoding is being done and whether
 * 1D- or 2D-encoded data is involved.
 */
static int
Fax3SetupState(TIFF* tif)
{
    static const char module[] = "Fax3SetupState";
    TIFFDirectory* td = &tif->tif_dir;
    Fax3BaseState* sp = Fax3State(tif);
    int needsRefLine;
    Fax3CodecState* dsp = (Fax3CodecState*) Fax3State(tif);
    tmsize_t rowbytes;
    uint32 rowpixels, nruns;

    if (td->td_bitspersample != 1) {
        TIFFErrorExt(tif->tif_clientdata, module,
            "Bits/sample must be 1 for Group 3/4 encoding/decoding");
        return (0);
    }
    /*
     * Calculate the scanline/tile widths.
     */
    if (isTiled(tif)) {
        rowbytes = TIFFTileRowSize(tif);
        rowpixels = td->td_tilewidth;
    } else {
        rowbytes = TIFFScanlineSize(tif);
        rowpixels = td->td_imagewidth;
    }
    sp->rowbytes = rowbytes;
    sp->rowpixels = rowpixels;
    /*
     * Allocate any additional space required for decoding/encoding.
     */
    needsRefLine = (
        (sp->groupoptions & GROUP3OPT_2DENCODING) ||
        td->td_compression == COMPRESSION_CCITTFAX4
    );

    /*
      Assure that allocation computations do not overflow.

      TIFFroundup and TIFFSafeMultiply return zero on integer overflow
    */
    dsp->runs=(uint32*) NULL;
    nruns = TIFFroundup_32(rowpixels,32);
    if (needsRefLine) {
        nruns = TIFFSafeMultiply(uint32,nruns,2);
    }
    if ((nruns == 0) || (TIFFSafeMultiply(uint32,nruns,2) == 0)) {
        TIFFErrorExt(tif->tif_clientdata, tif->tif_name,
                 "Row pixels integer overflow (rowpixels %u)",
                 rowpixels);
        return (0);
    }
    dsp->runs = (uint32*) _TIFFCheckMalloc(tif,
                           TIFFSafeMultiply(uint32,nruns,2),
                           sizeof (uint32),
                           "for Group 3/4 run arrays");
    if (dsp->runs == NULL)
        return (0);
    memset( dsp->runs, 0, TIFFSafeMultiply(uint32,nruns,2));
    dsp->curruns = dsp->runs;
    if (needsRefLine)
        dsp->refruns = dsp->runs + nruns;
    else
        dsp->refruns = NULL;
    if (td->td_compression == COMPRESSION_CCITTFAX3
        && is2DEncoding(dsp)) {	/* NB: default is 1D routine */
        tif->tif_decoderow = Fax3Decode2D;
        tif->tif_decodestrip = Fax3Decode2D;
        tif->tif_decodetile = Fax3Decode2D;
    }

    if (needsRefLine) {		/* 2d encoding */
        Fax3CodecState* esp = EncoderState(tif);
        /*
         * 2d encoding requires a scanline
         * buffer for the ``reference line''; the
         * scanline against which delta encoding
         * is referenced.  The reference line must
         * be initialized to be ``white'' (done elsewhere).
         */
        esp->refline = (unsigned char*) _TIFFmalloc(rowbytes);
        if (esp->refline == NULL) {
            TIFFErrorExt(tif->tif_clientdata, module,
                "No space for Group 3/4 reference line");
            return (0);
        }
    } else					/* 1d encoding */
        EncoderState(tif)->refline = NULL;

    return (1);
}

/*
 * CCITT Group 3 FAX Encoding.
 */

#define	Fax3FlushBits(tif, sp) {				\
    if ((tif)->tif_rawcc >= (tif)->tif_rawdatasize)		\
        (void) TIFFFlushData1(tif);			\
    *(tif)->tif_rawcp++ = (uint8) (sp)->data;		\
    (tif)->tif_rawcc++;					\
    (sp)->data = 0, (sp)->bit = 8;				\
}
#define	_FlushBits(tif) {					\
    if ((tif)->tif_rawcc >= (tif)->tif_rawdatasize)		\
        (void) TIFFFlushData1(tif);			\
    *(tif)->tif_rawcp++ = (uint8) data;		\
    (tif)->tif_rawcc++;					\
    data = 0, bit = 8;					\
}
static const int _msbmask[9] =
    { 0x00, 0x01, 0x03, 0x07, 0x0f, 0x1f, 0x3f, 0x7f, 0xff };
#define	_PutBits(tif, bits, length) {				\
    while (length > bit) {					\
        data |= bits >> (length - bit);			\
        length -= bit;					\
        _FlushBits(tif);				\
    }							\
        assert( length < 9 );                                   \
    data |= (bits & _msbmask[length]) << (bit - length);	\
    bit -= length;						\
    if (bit == 0)						\
        _FlushBits(tif);				\
}

/*
 * Write a variable-length bit-value to
 * the output stream.  Values are
 * assumed to be at most 16 bits.
 */
static void
Fax3PutBits(TIFF* tif, unsigned int bits, unsigned int length)
{
    Fax3CodecState* sp = EncoderState(tif);
    unsigned int bit = sp->bit;
    int data = sp->data;

    _PutBits(tif, bits, length);

    sp->data = data;
    sp->bit = bit;
}

/*
 * Write a code to the output stream.
 */
#define putcode(tif, te)	Fax3PutBits(tif, (te)->code, (te)->length)

#ifdef FAX3_DEBUG
#define	DEBUG_COLOR(w) (tab == TIFFFaxWhiteCodes ? w "W" : w "B")
#define	DEBUG_PRINT(what,len) {						\
    int t;								\
    printf("%08X/%-2d: %s%5d\t", data, bit, DEBUG_COLOR(what), len);	\
    for (t = length-1; t >= 0; t--)					\
    putchar(code & (1<<t) ? '1' : '0');				\
    putchar('\n');							\
}
#endif

/*
 * Write the sequence of codes that describes
 * the specified span of zero's or one's.  The
 * appropriate table that holds the make-up and
 * terminating codes is supplied.
 */
static void
putspan(TIFF* tif, int32 span, const tableentry* tab)
{
    Fax3CodecState* sp = EncoderState(tif);
    unsigned int bit = sp->bit;
    int data = sp->data;
    unsigned int code, length;

    while (span >= 2624) {
        const tableentry* te = &tab[63 + (2560>>6)];
        code = te->code, length = te->length;
#ifdef FAX3_DEBUG
        DEBUG_PRINT("MakeUp", te->runlen);
#endif
        _PutBits(tif, code, length);
        span -= te->runlen;
    }
    if (span >= 64) {
        const tableentry* te = &tab[63 + (span>>6)];
        assert(te->runlen == 64*(span>>6));
        code = te->code, length = te->length;
#ifdef FAX3_DEBUG
        DEBUG_PRINT("MakeUp", te->runlen);
#endif
        _PutBits(tif, code, length);
        span -= te->runlen;
    }
    code = tab[span].code, length = tab[span].length;
#ifdef FAX3_DEBUG
    DEBUG_PRINT("  Term", tab[span].runlen);
#endif
    _PutBits(tif, code, length);

    sp->data = data;
    sp->bit = bit;
}

/*
 * Write an EOL code to the output stream.  The zero-fill
 * logic for byte-aligning encoded scanlines is handled
 * here.  We also handle writing the tag bit for the next
 * scanline when doing 2d encoding.
 */
static void
Fax3PutEOL(TIFF* tif)
{
    Fax3CodecState* sp = EncoderState(tif);
    unsigned int bit = sp->bit;
    int data = sp->data;
    unsigned int code, length, tparm;

    if (sp->b.groupoptions & GROUP3OPT_FILLBITS) {
        /*
         * Force bit alignment so EOL will terminate on
         * a byte boundary.  That is, force the bit alignment
         * to 16-12 = 4 before putting out the EOL code.
         */
        int align = 8 - 4;
        if (align != sp->bit) {
            if (align > sp->bit)
                align = sp->bit + (8 - align);
            else
                align = sp->bit - align;
            code = 0;
            tparm=align;
            _PutBits(tif, 0, tparm);
        }
    }
    code = EOL, length = 12;
    if (is2DEncoding(sp))
        code = (code<<1) | (sp->tag == G3_1D), length++;
    _PutBits(tif, code, length);

    sp->data = data;
    sp->bit = bit;
}

/*
 * Reset encoding state at the start of a strip.
 */
static int
Fax3PreEncode(TIFF* tif, uint16 s)
{
    Fax3CodecState* sp = EncoderState(tif);

    (void) s;
    assert(sp != NULL);
    sp->bit = 8;
    sp->data = 0;
    sp->tag = G3_1D;
    /*
     * This is necessary for Group 4; otherwise it isn't
     * needed because the first scanline of each strip ends
     * up being copied into the refline.
     */
    if (sp->refline)
        _TIFFmemset(sp->refline, 0x00, sp->b.rowbytes);
    if (is2DEncoding(sp)) {
        float res = tif->tif_dir.td_yresolution;
        /*
         * The CCITT spec says that when doing 2d encoding, you
         * should only do it on K consecutive scanlines, where K
         * depends on the resolution of the image being encoded
         * (2 for <= 200 lpi, 4 for > 200 lpi).  Since the directory
         * code initializes td_yresolution to 0, this code will
         * select a K of 2 unless the YResolution tag is set
         * appropriately.  (Note also that we fudge a little here
         * and use 150 lpi to avoid problems with units conversion.)
         */
        if (tif->tif_dir.td_resolutionunit == RESUNIT_CENTIMETER)
            res *= 2.54f;		/* convert to inches */
        sp->maxk = (res > 150 ? 4 : 2);
        sp->k = sp->maxk-1;
    } else
        sp->k = sp->maxk = 0;
    sp->line = 0;
    return (1);
}

static const unsigned char zeroruns[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,	/* 0x00 - 0x0f */
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,	/* 0x10 - 0x1f */
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,	/* 0x20 - 0x2f */
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,	/* 0x30 - 0x3f */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0x40 - 0x4f */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0x50 - 0x5f */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0x60 - 0x6f */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0x70 - 0x7f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x80 - 0x8f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x90 - 0x9f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0xa0 - 0xaf */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0xb0 - 0xbf */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0xc0 - 0xcf */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0xd0 - 0xdf */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0xe0 - 0xef */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0xf0 - 0xff */
};
static const unsigned char oneruns[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x00 - 0x0f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x10 - 0x1f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x20 - 0x2f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x30 - 0x3f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x40 - 0x4f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x50 - 0x5f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x60 - 0x6f */
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,	/* 0x70 - 0x7f */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0x80 - 0x8f */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0x90 - 0x9f */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0xa0 - 0xaf */
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,	/* 0xb0 - 0xbf */
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,	/* 0xc0 - 0xcf */
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,	/* 0xd0 - 0xdf */
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,	/* 0xe0 - 0xef */
    4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8,	/* 0xf0 - 0xff */
};

/*
 * On certain systems it pays to inline
 * the routines that find pixel spans.
 */
#ifdef VAXC
static	int32 find0span(unsigned char*, int32, int32);
static	int32 find1span(unsigned char*, int32, int32);
#pragma inline(find0span,find1span)
#endif

/*
 * Find a span of ones or zeros using the supplied
 * table.  The ``base'' of the bit string is supplied
 * along with the start+end bit indices.
 */
static int32
find0span(unsigned char* bp, int32 bs, int32 be)
{
    int32 bits = be - bs;
    int32 n, span;

    bp += bs>>3;
    /*
     * Check partial byte on lhs.
     */
    if (bits > 0 && (n = (bs & 7))) {
        span = zeroruns[(*bp << n) & 0xff];
        if (span > 8-n)		/* table value too generous */
            span = 8-n;
        if (span > bits)	/* constrain span to bit range */
            span = bits;
        if (n+span < 8)		/* doesn't extend to edge of byte */
            return (span);
        bits -= span;
        bp++;
    } else
        span = 0;
    if (bits >= (int32)(2 * 8 * sizeof(long))) {
        long* lp;
        /*
         * Align to longword boundary and check longwords.
         */
        while (!isAligned(bp, long)) {
            if (*bp != 0x00)
                return (span + zeroruns[*bp]);
            span += 8, bits -= 8;
            bp++;
        }
        lp = (long*) bp;
        while ((bits >= (int32)(8 * sizeof(long))) && (0 == *lp)) {
            span += 8*sizeof (long), bits -= 8*sizeof (long);
            lp++;
        }
        bp = (unsigned char*) lp;
    }
    /*
     * Scan full bytes for all 0's.
     */
    while (bits >= 8) {
        if (*bp != 0x00)	/* end of run */
            return (span + zeroruns[*bp]);
        span += 8, bits -= 8;
        bp++;
    }
    /*
     * Check partial byte on rhs.
     */
    if (bits > 0) {
        n = zeroruns[*bp];
        span += (n > bits ? bits : n);
    }
    return (span);
}

static int32
find1span(unsigned char* bp, int32 bs, int32 be)
{
    int32 bits = be - bs;
    int32 n, span;

    bp += bs>>3;
    /*
     * Check partial byte on lhs.
     */
    if (bits > 0 && (n = (bs & 7))) {
        span = oneruns[(*bp << n) & 0xff];
        if (span > 8-n)		/* table value too generous */
            span = 8-n;
        if (span > bits)	/* constrain span to bit range */
            span = bits;
        if (n+span < 8)		/* doesn't extend to edge of byte */
            return (span);
        bits -= span;
        bp++;
    } else
        span = 0;
    if (bits >= (int32)(2 * 8 * sizeof(long))) {
        long* lp;
        /*
         * Align to longword boundary and check longwords.
         */
        while (!isAligned(bp, long)) {
            if (*bp != 0xff)
                return (span + oneruns[*bp]);
            span += 8, bits -= 8;
            bp++;
        }
        lp = (long*) bp;
        while ((bits >= (int32)(8 * sizeof(long))) && (~0 == *lp)) {
            span += 8*sizeof (long), bits -= 8*sizeof (long);
            lp++;
        }
        bp = (unsigned char*) lp;
    }
    /*
     * Scan full bytes for all 1's.
     */
    while (bits >= 8) {
        if (*bp != 0xff)	/* end of run */
            return (span + oneruns[*bp]);
        span += 8, bits -= 8;
        bp++;
    }
    /*
     * Check partial byte on rhs.
     */
    if (bits > 0) {
        n = oneruns[*bp];
        span += (n > bits ? bits : n);
    }
    return (span);
}

/*
 * Return the offset of the next bit in the range
 * [bs..be] that is different from the specified
 * color.  The end, be, is returned if no such bit
 * exists.
 */
#define	finddiff(_cp, _bs, _be, _color)	\
    (_bs + (_color ? find1span(_cp,_bs,_be) : find0span(_cp,_bs,_be)))
/*
 * Like finddiff, but also check the starting bit
 * against the end in case start > end.
 */
#define	finddiff2(_cp, _bs, _be, _color) \
    (_bs < _be ? finddiff(_cp,_bs,_be,_color) : _be)

/*
 * 1d-encode a row of pixels.  The encoding is
 * a sequence of all-white or all-black spans
 * of pixels encoded with Huffman codes.
 */
static int
Fax3Encode1DRow(TIFF* tif, unsigned char* bp, uint32 bits)
{
    Fax3CodecState* sp = EncoderState(tif);
    int32 span;
        uint32 bs = 0;

    for (;;) {
        span = find0span(bp, bs, bits);		/* white span */
        putspan(tif, span, TIFFFaxWhiteCodes);
        bs += span;
        if (bs >= bits)
            break;
        span = find1span(bp, bs, bits);		/* black span */
        putspan(tif, span, TIFFFaxBlackCodes);
        bs += span;
        if (bs >= bits)
            break;
    }
    if (sp->b.mode & (FAXMODE_BYTEALIGN|FAXMODE_WORDALIGN)) {
        if (sp->bit != 8)			/* byte-align */
            Fax3FlushBits(tif, sp);
        if ((sp->b.mode&FAXMODE_WORDALIGN) &&
            !isAligned(tif->tif_rawcp, uint16))
            Fax3FlushBits(tif, sp);
    }
    return (1);
}

static const tableentry horizcode =
    { 3, 0x1, 0 };	/* 001 */
static const tableentry passcode =
    { 4, 0x1, 0 };	/* 0001 */
static const tableentry vcodes[7] = {
    { 7, 0x03, 0 },	/* 0000 011 */
    { 6, 0x03, 0 },	/* 0000 11 */
    { 3, 0x03, 0 },	/* 011 */
    { 1, 0x1, 0 },	/* 1 */
    { 3, 0x2, 0 },	/* 010 */
    { 6, 0x02, 0 },	/* 0000 10 */
    { 7, 0x02, 0 }	/* 0000 010 */
};

/*
 * 2d-encode a row of pixels.  Consult the CCITT
 * documentation for the algorithm.
 */
static int
Fax3Encode2DRow(TIFF* tif, unsigned char* bp, unsigned char* rp, uint32 bits)
{
#define	PIXEL(buf,ix)	((((buf)[(ix)>>3]) >> (7-((ix)&7))) & 1)
        uint32 a0 = 0;
    uint32 a1 = (PIXEL(bp, 0) != 0 ? 0 : finddiff(bp, 0, bits, 0));
    uint32 b1 = (PIXEL(rp, 0) != 0 ? 0 : finddiff(rp, 0, bits, 0));
    uint32 a2, b2;

    for (;;) {
        b2 = finddiff2(rp, b1, bits, PIXEL(rp,b1));
        if (b2 >= a1) {
            int32 d = b1 - a1;
            if (!(-3 <= d && d <= 3)) {	/* horizontal mode */
                a2 = finddiff2(bp, a1, bits, PIXEL(bp,a1));
                putcode(tif, &horizcode);
                if (a0+a1 == 0 || PIXEL(bp, a0) == 0) {
                    putspan(tif, a1-a0, TIFFFaxWhiteCodes);
                    putspan(tif, a2-a1, TIFFFaxBlackCodes);
                } else {
                    putspan(tif, a1-a0, TIFFFaxBlackCodes);
                    putspan(tif, a2-a1, TIFFFaxWhiteCodes);
                }
                a0 = a2;
            } else {			/* vertical mode */
                putcode(tif, &vcodes[d+3]);
                a0 = a1;
            }
        } else {				/* pass mode */
            putcode(tif, &passcode);
            a0 = b2;
        }
        if (a0 >= bits)
            break;
        a1 = finddiff(bp, a0, bits, PIXEL(bp,a0));
        b1 = finddiff(rp, a0, bits, !PIXEL(bp,a0));
        b1 = finddiff(rp, b1, bits, PIXEL(bp,a0));
    }
    return (1);
#undef PIXEL
}

/*
 * Encode a buffer of pixels.
 */
static int
Fax3Encode(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s)
{
    static const char module[] = "Fax3Encode";
    Fax3CodecState* sp = EncoderState(tif);
    (void) s;
    if (cc % sp->b.rowbytes)
    {
        TIFFErrorExt(tif->tif_clientdata, module, "Fractional scanlines cannot be written");
        return (0);
    }
    while (cc > 0) {
        if ((sp->b.mode & FAXMODE_NOEOL) == 0)
            Fax3PutEOL(tif);
        if (is2DEncoding(sp)) {
            if (sp->tag == G3_1D) {
                if (!Fax3Encode1DRow(tif, bp, sp->b.rowpixels))
                    return (0);
                sp->tag = G3_2D;
            } else {
                if (!Fax3Encode2DRow(tif, bp, sp->refline,
                    sp->b.rowpixels))
                    return (0);
                sp->k--;
            }
            if (sp->k == 0) {
                sp->tag = G3_1D;
                sp->k = sp->maxk-1;
            } else
                _TIFFmemcpy(sp->refline, bp, sp->b.rowbytes);
        } else {
            if (!Fax3Encode1DRow(tif, bp, sp->b.rowpixels))
                return (0);
        }
        bp += sp->b.rowbytes;
        cc -= sp->b.rowbytes;
    }
    return (1);
}

static int
Fax3PostEncode(TIFF* tif)
{
    Fax3CodecState* sp = EncoderState(tif);

    if (sp->bit != 8)
        Fax3FlushBits(tif, sp);
    return (1);
}

static void
Fax3Close(TIFF* tif)
{
    if ((Fax3State(tif)->mode & FAXMODE_NORTC) == 0) {
        Fax3CodecState* sp = EncoderState(tif);
        unsigned int code = EOL;
        unsigned int length = 12;
        int i;

        if (is2DEncoding(sp))
            code = (code<<1) | (sp->tag == G3_1D), length++;
        for (i = 0; i < 6; i++)
            Fax3PutBits(tif, code, length);
        Fax3FlushBits(tif, sp);
    }
}

static void
Fax3Cleanup(TIFF* tif)
{
    Fax3CodecState* sp = DecoderState(tif);

    assert(sp != 0);

    tif->tif_tagmethods.vgetfield = sp->b.vgetparent;
    tif->tif_tagmethods.vsetfield = sp->b.vsetparent;
    tif->tif_tagmethods.printdir = sp->b.printdir;

    if (sp->runs)
        _TIFFfree(sp->runs);
    if (sp->refline)
        _TIFFfree(sp->refline);

    _TIFFfree(tif->tif_data);
    tif->tif_data = NULL;

    _TIFFSetDefaultCompressionState(tif);
}

#define	FIELD_BADFAXLINES	(FIELD_CODEC+0)
#define	FIELD_CLEANFAXDATA	(FIELD_CODEC+1)
#define	FIELD_BADFAXRUN		(FIELD_CODEC+2)

#define	FIELD_OPTIONS		(FIELD_CODEC+7)

static const TIFFField faxFields[] = {
    { TIFFTAG_FAXMODE, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT, TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, FALSE, FALSE, "FaxMode", NULL },
    { TIFFTAG_FAXFILLFUNC, 0, 0, TIFF_ANY, 0, TIFF_SETGET_OTHER, TIFF_SETGET_UNDEFINED, FIELD_PSEUDO, FALSE, FALSE, "FaxFillFunc", NULL },
    { TIFFTAG_BADFAXLINES, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UINT32, FIELD_BADFAXLINES, TRUE, FALSE, "BadFaxLines", NULL },
    { TIFFTAG_CLEANFAXDATA, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UINT16, FIELD_CLEANFAXDATA, TRUE, FALSE, "CleanFaxData", NULL },
    { TIFFTAG_CONSECUTIVEBADFAXLINES, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UINT32, FIELD_BADFAXRUN, TRUE, FALSE, "ConsecutiveBadFaxLines", NULL }};
static const TIFFField fax3Fields[] = {
    { TIFFTAG_GROUP3OPTIONS, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UINT32, FIELD_OPTIONS, FALSE, FALSE, "Group3Options", NULL },
};
static const TIFFField fax4Fields[] = {
    { TIFFTAG_GROUP4OPTIONS, 1, 1, TIFF_LONG, 0, TIFF_SETGET_UINT32, TIFF_SETGET_UINT32, FIELD_OPTIONS, FALSE, FALSE, "Group4Options", NULL },
};

static int
Fax3VSetField(TIFF* tif, uint32 tag, va_list ap)
{
    Fax3BaseState* sp = Fax3State(tif);
    const TIFFField* fip;

    assert(sp != 0);
    assert(sp->vsetparent != 0);

    switch (tag) {
    case TIFFTAG_FAXMODE:
        sp->mode = (int) va_arg(ap, int);
        return 1;			/* NB: pseudo tag */
    case TIFFTAG_FAXFILLFUNC:
        DecoderState(tif)->fill = va_arg(ap, TIFFFaxFillFunc);
        return 1;			/* NB: pseudo tag */
    case TIFFTAG_GROUP3OPTIONS:
        /* XXX: avoid reading options if compression mismatches. */
        if (tif->tif_dir.td_compression == COMPRESSION_CCITTFAX3)
            sp->groupoptions = (uint32) va_arg(ap, uint32);
        break;
    case TIFFTAG_GROUP4OPTIONS:
        /* XXX: avoid reading options if compression mismatches. */
        if (tif->tif_dir.td_compression == COMPRESSION_CCITTFAX4)
            sp->groupoptions = (uint32) va_arg(ap, uint32);
        break;
    case TIFFTAG_BADFAXLINES:
        sp->badfaxlines = (uint32) va_arg(ap, uint32);
        break;
    case TIFFTAG_CLEANFAXDATA:
        sp->cleanfaxdata = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_CONSECUTIVEBADFAXLINES:
        sp->badfaxrun = (uint32) va_arg(ap, uint32);
        break;
    default:
        return (*sp->vsetparent)(tif, tag, ap);
    }

    if ((fip = TIFFFieldWithTag(tif, tag)))
        TIFFSetFieldBit(tif, fip->field_bit);
    else
        return 0;

    tif->tif_flags |= TIFF_DIRTYDIRECT;
    return 1;
}

static int
Fax3VGetField(TIFF* tif, uint32 tag, va_list ap)
{
    Fax3BaseState* sp = Fax3State(tif);

    assert(sp != 0);

    switch (tag) {
    case TIFFTAG_FAXMODE:
        *va_arg(ap, int*) = sp->mode;
        break;
    case TIFFTAG_FAXFILLFUNC:
        *va_arg(ap, TIFFFaxFillFunc*) = DecoderState(tif)->fill;
        break;
    case TIFFTAG_GROUP3OPTIONS:
    case TIFFTAG_GROUP4OPTIONS:
        *va_arg(ap, uint32*) = sp->groupoptions;
        break;
    case TIFFTAG_BADFAXLINES:
        *va_arg(ap, uint32*) = sp->badfaxlines;
        break;
    case TIFFTAG_CLEANFAXDATA:
        *va_arg(ap, uint16*) = sp->cleanfaxdata;
        break;
    case TIFFTAG_CONSECUTIVEBADFAXLINES:
        *va_arg(ap, uint32*) = sp->badfaxrun;
        break;
    default:
        return (*sp->vgetparent)(tif, tag, ap);
    }
    return (1);
}

static void
Fax3PrintDir(TIFF* tif, FILE* fd, long flags)
{
    Fax3BaseState* sp = Fax3State(tif);

    assert(sp != 0);

    (void) flags;
    if (TIFFFieldSet(tif,FIELD_OPTIONS)) {
        const char* sep = " ";
        if (tif->tif_dir.td_compression == COMPRESSION_CCITTFAX4) {
            fprintf(fd, "  Group 4 Options:");
            if (sp->groupoptions & GROUP4OPT_UNCOMPRESSED)
                fprintf(fd, "%suncompressed data", sep);
        } else {

            fprintf(fd, "  Group 3 Options:");
            if (sp->groupoptions & GROUP3OPT_2DENCODING)
                fprintf(fd, "%s2-d encoding", sep), sep = "+";
            if (sp->groupoptions & GROUP3OPT_FILLBITS)
                fprintf(fd, "%sEOL padding", sep), sep = "+";
            if (sp->groupoptions & GROUP3OPT_UNCOMPRESSED)
                fprintf(fd, "%suncompressed data", sep);
        }
        fprintf(fd, " (%lu = 0x%lx)\n",
                        (unsigned long) sp->groupoptions,
                        (unsigned long) sp->groupoptions);
    }
    if (TIFFFieldSet(tif,FIELD_CLEANFAXDATA)) {
        fprintf(fd, "  Fax Data:");
        switch (sp->cleanfaxdata) {
        case CLEANFAXDATA_CLEAN:
            fprintf(fd, " clean");
            break;
        case CLEANFAXDATA_REGENERATED:
            fprintf(fd, " receiver regenerated");
            break;
        case CLEANFAXDATA_UNCLEAN:
            fprintf(fd, " uncorrected errors");
            break;
        }
        fprintf(fd, " (%u = 0x%x)\n",
            sp->cleanfaxdata, sp->cleanfaxdata);
    }
    if (TIFFFieldSet(tif,FIELD_BADFAXLINES))
        fprintf(fd, "  Bad Fax Lines: %lu\n",
                        (unsigned long) sp->badfaxlines);
    if (TIFFFieldSet(tif,FIELD_BADFAXRUN))
        fprintf(fd, "  Consecutive Bad Fax Lines: %lu\n",
            (unsigned long) sp->badfaxrun);
    if (sp->printdir)
        (*sp->printdir)(tif, fd, flags);
}

static int
InitCCITTFax3(TIFF* tif)
{
    static const char module[] = "InitCCITTFax3";
    Fax3BaseState* sp;

    /*
     * Merge codec-specific tag information.
     */
    if (!_TIFFMergeFields(tif, faxFields, TIFFArrayCount(faxFields))) {
        TIFFErrorExt(tif->tif_clientdata, "InitCCITTFax3",
            "Merging common CCITT Fax codec-specific tags failed");
        return 0;
    }

    /*
     * Allocate state block so tag methods have storage to record values.
     */
    tif->tif_data = (uint8*)
        _TIFFmalloc(sizeof (Fax3CodecState));

    if (tif->tif_data == NULL) {
        TIFFErrorExt(tif->tif_clientdata, module,
            "No space for state block");
        return (0);
    }

    sp = Fax3State(tif);
        sp->rw_mode = tif->tif_mode;

    /*
     * Override parent get/set field methods.
     */
    sp->vgetparent = tif->tif_tagmethods.vgetfield;
    tif->tif_tagmethods.vgetfield = Fax3VGetField; /* hook for codec tags */
    sp->vsetparent = tif->tif_tagmethods.vsetfield;
    tif->tif_tagmethods.vsetfield = Fax3VSetField; /* hook for codec tags */
    sp->printdir = tif->tif_tagmethods.printdir;
    tif->tif_tagmethods.printdir = Fax3PrintDir;   /* hook for codec tags */
    sp->groupoptions = 0;

    if (sp->rw_mode == O_RDONLY) /* FIXME: improve for in place update */
        tif->tif_flags |= TIFF_NOBITREV; /* decoder does bit reversal */
    DecoderState(tif)->runs = NULL;
    TIFFSetField(tif, TIFFTAG_FAXFILLFUNC, _TIFFFax3fillruns);
    EncoderState(tif)->refline = NULL;

    /*
     * Install codec methods.
     */
    tif->tif_fixuptags = Fax3FixupTags;
    tif->tif_setupdecode = Fax3SetupState;
    tif->tif_predecode = Fax3PreDecode;
    tif->tif_decoderow = Fax3Decode1D;
    tif->tif_decodestrip = Fax3Decode1D;
    tif->tif_decodetile = Fax3Decode1D;
    tif->tif_setupencode = Fax3SetupState;
    tif->tif_preencode = Fax3PreEncode;
    tif->tif_postencode = Fax3PostEncode;
    tif->tif_encoderow = Fax3Encode;
    tif->tif_encodestrip = Fax3Encode;
    tif->tif_encodetile = Fax3Encode;
    tif->tif_close = Fax3Close;
    tif->tif_cleanup = Fax3Cleanup;

    return (1);
}

int
TIFFInitCCITTFax3(TIFF* tif, int scheme)
{
    (void) scheme;
    if (InitCCITTFax3(tif)) {
        /*
         * Merge codec-specific tag information.
         */
        if (!_TIFFMergeFields(tif, fax3Fields,
                      TIFFArrayCount(fax3Fields))) {
            TIFFErrorExt(tif->tif_clientdata, "TIFFInitCCITTFax3",
            "Merging CCITT Fax 3 codec-specific tags failed");
            return 0;
        }

        /*
         * The default format is Class/F-style w/o RTC.
         */
        return TIFFSetField(tif, TIFFTAG_FAXMODE, FAXMODE_CLASSF);
    } else
        return 01;
}

/*
 * CCITT Group 4 (T.6) Facsimile-compatible
 * Compression Scheme Support.
 */

#define SWAP(t,a,b) { t x; x = (a); (a) = (b); (b) = x; }
/*
 * Decode the requested amount of G4-encoded data.
 */
static int
Fax4Decode(TIFF* tif, uint8* buf, tmsize_t occ, uint16 s)
{
    DECLARE_STATE_2D(tif, sp, "Fax4Decode");
    (void) s;
    if (occ % sp->b.rowbytes)
    {
        TIFFErrorExt(tif->tif_clientdata, module, "Fractional scanlines cannot be read");
        return (-1);
    }
    CACHE_STATE(tif, sp);
    while (occ > 0) {
        a0 = 0;
        RunLength = 0;
        pa = thisrun = sp->curruns;
        pb = sp->refruns;
        b1 = *pb++;
#ifdef FAX3_DEBUG
        printf("\nBitAcc=%08X, BitsAvail = %d\n", BitAcc, BitsAvail);
        printf("-------------------- %d\n", tif->tif_row);
        fflush(stdout);
#endif
        EXPAND2D(EOFG4);
                if (EOLcnt)
                    goto EOFG4;
        (*sp->fill)(buf, thisrun, pa, lastx);
        SETVALUE(0);		/* imaginary change for reference */
        SWAP(uint32*, sp->curruns, sp->refruns);
        buf += sp->b.rowbytes;
        occ -= sp->b.rowbytes;
        sp->line++;
        continue;
    EOFG4:
                NeedBits16( 13, BADG4 );
        BADG4:
#ifdef FAX3_DEBUG
                if( GetBits(13) != 0x1001 )
                    fputs( "Bad EOFB\n", stderr );
#endif
                ClrBits( 13 );
        (*sp->fill)(buf, thisrun, pa, lastx);
        UNCACHE_STATE(tif, sp);
        return ( sp->line ? 1 : -1);	/* don't error on badly-terminated strips */
    }
    UNCACHE_STATE(tif, sp);
    return (1);
}
#undef	SWAP

/*
 * Encode the requested amount of data.
 */
static int
Fax4Encode(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s)
{
    static const char module[] = "Fax4Encode";
    Fax3CodecState *sp = EncoderState(tif);
    (void) s;
    if (cc % sp->b.rowbytes)
    {
        TIFFErrorExt(tif->tif_clientdata, module, "Fractional scanlines cannot be written");
        return (0);
    }
    while (cc > 0) {
        if (!Fax3Encode2DRow(tif, bp, sp->refline, sp->b.rowpixels))
            return (0);
        _TIFFmemcpy(sp->refline, bp, sp->b.rowbytes);
        bp += sp->b.rowbytes;
        cc -= sp->b.rowbytes;
    }
    return (1);
}

static int
Fax4PostEncode(TIFF* tif)
{
    Fax3CodecState *sp = EncoderState(tif);

    /* terminate strip w/ EOFB */
    Fax3PutBits(tif, EOL, 12);
    Fax3PutBits(tif, EOL, 12);
    if (sp->bit != 8)
        Fax3FlushBits(tif, sp);
    return (1);
}

int
TIFFInitCCITTFax4(TIFF* tif, int scheme)
{
    (void) scheme;
    if (InitCCITTFax3(tif)) {		/* reuse G3 support */
        /*
         * Merge codec-specific tag information.
         */
        if (!_TIFFMergeFields(tif, fax4Fields,
                      TIFFArrayCount(fax4Fields))) {
            TIFFErrorExt(tif->tif_clientdata, "TIFFInitCCITTFax4",
            "Merging CCITT Fax 4 codec-specific tags failed");
            return 0;
        }

        tif->tif_decoderow = Fax4Decode;
        tif->tif_decodestrip = Fax4Decode;
        tif->tif_decodetile = Fax4Decode;
        tif->tif_encoderow = Fax4Encode;
        tif->tif_encodestrip = Fax4Encode;
        tif->tif_encodetile = Fax4Encode;
        tif->tif_postencode = Fax4PostEncode;
        /*
         * Suppress RTC at the end of each strip.
         */
        return TIFFSetField(tif, TIFFTAG_FAXMODE, FAXMODE_NORTC);
    } else
        return (0);
}

/*
 * CCITT Group 3 1-D Modified Huffman RLE Compression Support.
 * (Compression algorithms 2 and 32771)
 */

/*
 * Decode the requested amount of RLE-encoded data.
 */
static int
Fax3DecodeRLE(TIFF* tif, uint8* buf, tmsize_t occ, uint16 s)
{
    DECLARE_STATE(tif, sp, "Fax3DecodeRLE");
    int mode = sp->b.mode;
    (void) s;
    if (occ % sp->b.rowbytes)
    {
        TIFFErrorExt(tif->tif_clientdata, module, "Fractional scanlines cannot be read");
        return (-1);
    }
    CACHE_STATE(tif, sp);
    thisrun = sp->curruns;
    while (occ > 0) {
        a0 = 0;
        RunLength = 0;
        pa = thisrun;
#ifdef FAX3_DEBUG
        printf("\nBitAcc=%08X, BitsAvail = %d\n", BitAcc, BitsAvail);
        printf("-------------------- %d\n", tif->tif_row);
        fflush(stdout);
#endif
        EXPAND1D(EOFRLE);
        (*sp->fill)(buf, thisrun, pa, lastx);
        /*
         * Cleanup at the end of the row.
         */
        if (mode & FAXMODE_BYTEALIGN) {
            int n = BitsAvail - (BitsAvail &~ 7);
            ClrBits(n);
        } else if (mode & FAXMODE_WORDALIGN) {
            int n = BitsAvail - (BitsAvail &~ 15);
            ClrBits(n);
            if (BitsAvail == 0 && !isAligned(cp, uint16))
                cp++;
        }
        buf += sp->b.rowbytes;
        occ -= sp->b.rowbytes;
        sp->line++;
        continue;
    EOFRLE:				/* premature EOF */
        (*sp->fill)(buf, thisrun, pa, lastx);
        UNCACHE_STATE(tif, sp);
        return (-1);
    }
    UNCACHE_STATE(tif, sp);
    return (1);
}

int
TIFFInitCCITTRLE(TIFF* tif, int scheme)
{
    (void) scheme;
    if (InitCCITTFax3(tif)) {		/* reuse G3 support */
        tif->tif_decoderow = Fax3DecodeRLE;
        tif->tif_decodestrip = Fax3DecodeRLE;
        tif->tif_decodetile = Fax3DecodeRLE;
        /*
         * Suppress RTC+EOLs when encoding and byte-align data.
         */
        return TIFFSetField(tif, TIFFTAG_FAXMODE,
            FAXMODE_NORTC|FAXMODE_NOEOL|FAXMODE_BYTEALIGN);
    } else
        return (0);
}

int
TIFFInitCCITTRLEW(TIFF* tif, int scheme)
{
    (void) scheme;
    if (InitCCITTFax3(tif)) {		/* reuse G3 support */
        tif->tif_decoderow = Fax3DecodeRLE;
        tif->tif_decodestrip = Fax3DecodeRLE;
        tif->tif_decodetile = Fax3DecodeRLE;
        /*
         * Suppress RTC+EOLs when encoding and word-align data.
         */
        return TIFFSetField(tif, TIFFTAG_FAXMODE,
            FAXMODE_NORTC|FAXMODE_NOEOL|FAXMODE_WORDALIGN);
    } else
        return (0);
}
#endif /* CCITT_SUPPORT */

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
