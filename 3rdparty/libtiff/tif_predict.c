/*
 * Copyright (c) 1988-1997 Sam Leffler
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

/*
 * TIFF Library.
 *
 * Predictor Tag Support (used by multiple codecs).
 */
#include "tiffiop.h"
#include "tif_predict.h"

#define	PredictorState(tif)	((TIFFPredictorState*) (tif)->tif_data)

static int horAcc8(TIFF* tif, uint8* cp0, tmsize_t cc);
static int horAcc16(TIFF* tif, uint8* cp0, tmsize_t cc);
static int horAcc32(TIFF* tif, uint8* cp0, tmsize_t cc);
static int swabHorAcc16(TIFF* tif, uint8* cp0, tmsize_t cc);
static int swabHorAcc32(TIFF* tif, uint8* cp0, tmsize_t cc);
static int horDiff8(TIFF* tif, uint8* cp0, tmsize_t cc);
static int horDiff16(TIFF* tif, uint8* cp0, tmsize_t cc);
static int horDiff32(TIFF* tif, uint8* cp0, tmsize_t cc);
static int swabHorDiff16(TIFF* tif, uint8* cp0, tmsize_t cc);
static int swabHorDiff32(TIFF* tif, uint8* cp0, tmsize_t cc);
static int fpAcc(TIFF* tif, uint8* cp0, tmsize_t cc);
static int fpDiff(TIFF* tif, uint8* cp0, tmsize_t cc);
static int PredictorDecodeRow(TIFF* tif, uint8* op0, tmsize_t occ0, uint16 s);
static int PredictorDecodeTile(TIFF* tif, uint8* op0, tmsize_t occ0, uint16 s);
static int PredictorEncodeRow(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s);
static int PredictorEncodeTile(TIFF* tif, uint8* bp0, tmsize_t cc0, uint16 s);

static int
PredictorSetup(TIFF* tif)
{
	static const char module[] = "PredictorSetup";

	TIFFPredictorState* sp = PredictorState(tif);
	TIFFDirectory* td = &tif->tif_dir;

	switch (sp->predictor)		/* no differencing */
	{
		case PREDICTOR_NONE:
			return 1;
		case PREDICTOR_HORIZONTAL:
			if (td->td_bitspersample != 8
			    && td->td_bitspersample != 16
			    && td->td_bitspersample != 32) {
				TIFFErrorExt(tif->tif_clientdata, module,
				    "Horizontal differencing \"Predictor\" not supported with %d-bit samples",
				    td->td_bitspersample);
				return 0;
			}
			break;
		case PREDICTOR_FLOATINGPOINT:
			if (td->td_sampleformat != SAMPLEFORMAT_IEEEFP) {
				TIFFErrorExt(tif->tif_clientdata, module,
				    "Floating point \"Predictor\" not supported with %d data format",
				    td->td_sampleformat);
				return 0;
			}
                        if (td->td_bitspersample != 16
                            && td->td_bitspersample != 24
                            && td->td_bitspersample != 32
                            && td->td_bitspersample != 64) { /* Should 64 be allowed? */
                                TIFFErrorExt(tif->tif_clientdata, module,
                                             "Floating point \"Predictor\" not supported with %d-bit samples",
                                             td->td_bitspersample);
				return 0;
                            }
			break;
		default:
			TIFFErrorExt(tif->tif_clientdata, module,
			    "\"Predictor\" value %d not supported",
			    sp->predictor);
			return 0;
	}
	sp->stride = (td->td_planarconfig == PLANARCONFIG_CONTIG ?
	    td->td_samplesperpixel : 1);
	/*
	 * Calculate the scanline/tile-width size in bytes.
	 */
	if (isTiled(tif))
		sp->rowsize = TIFFTileRowSize(tif);
	else
		sp->rowsize = TIFFScanlineSize(tif);
	if (sp->rowsize == 0)
		return 0;

	return 1;
}

static int
PredictorSetupDecode(TIFF* tif)
{
	TIFFPredictorState* sp = PredictorState(tif);
	TIFFDirectory* td = &tif->tif_dir;

	/* Note: when PredictorSetup() fails, the effets of setupdecode() */
	/* will not be "cancelled" so setupdecode() might be robust to */
	/* be called several times. */
	if (!(*sp->setupdecode)(tif) || !PredictorSetup(tif))
		return 0;

	if (sp->predictor == 2) {
		switch (td->td_bitspersample) {
			case 8:  sp->decodepfunc = horAcc8; break;
			case 16: sp->decodepfunc = horAcc16; break;
			case 32: sp->decodepfunc = horAcc32; break;
		}
		/*
		 * Override default decoding method with one that does the
		 * predictor stuff.
		 */
                if( tif->tif_decoderow != PredictorDecodeRow )
                {
                    sp->decoderow = tif->tif_decoderow;
                    tif->tif_decoderow = PredictorDecodeRow;
                    sp->decodestrip = tif->tif_decodestrip;
                    tif->tif_decodestrip = PredictorDecodeTile;
                    sp->decodetile = tif->tif_decodetile;
                    tif->tif_decodetile = PredictorDecodeTile;
                }

		/*
		 * If the data is horizontally differenced 16-bit data that
		 * requires byte-swapping, then it must be byte swapped before
		 * the accumulation step.  We do this with a special-purpose
		 * routine and override the normal post decoding logic that
		 * the library setup when the directory was read.
		 */
		if (tif->tif_flags & TIFF_SWAB) {
			if (sp->decodepfunc == horAcc16) {
				sp->decodepfunc = swabHorAcc16;
				tif->tif_postdecode = _TIFFNoPostDecode;
            } else if (sp->decodepfunc == horAcc32) {
				sp->decodepfunc = swabHorAcc32;
				tif->tif_postdecode = _TIFFNoPostDecode;
            }
		}
	}

	else if (sp->predictor == 3) {
		sp->decodepfunc = fpAcc;
		/*
		 * Override default decoding method with one that does the
		 * predictor stuff.
		 */
                if( tif->tif_decoderow != PredictorDecodeRow )
                {
                    sp->decoderow = tif->tif_decoderow;
                    tif->tif_decoderow = PredictorDecodeRow;
                    sp->decodestrip = tif->tif_decodestrip;
                    tif->tif_decodestrip = PredictorDecodeTile;
                    sp->decodetile = tif->tif_decodetile;
                    tif->tif_decodetile = PredictorDecodeTile;
                }
		/*
		 * The data should not be swapped outside of the floating
		 * point predictor, the accumulation routine should return
		 * byres in the native order.
		 */
		if (tif->tif_flags & TIFF_SWAB) {
			tif->tif_postdecode = _TIFFNoPostDecode;
		}
		/*
		 * Allocate buffer to keep the decoded bytes before
		 * rearranging in the right order
		 */
	}

	return 1;
}

static int
PredictorSetupEncode(TIFF* tif)
{
	TIFFPredictorState* sp = PredictorState(tif);
	TIFFDirectory* td = &tif->tif_dir;

	if (!(*sp->setupencode)(tif) || !PredictorSetup(tif))
		return 0;

	if (sp->predictor == 2) {
		switch (td->td_bitspersample) {
			case 8:  sp->encodepfunc = horDiff8; break;
			case 16: sp->encodepfunc = horDiff16; break;
			case 32: sp->encodepfunc = horDiff32; break;
		}
		/*
		 * Override default encoding method with one that does the
		 * predictor stuff.
		 */
                if( tif->tif_encoderow != PredictorEncodeRow )
                {
                    sp->encoderow = tif->tif_encoderow;
                    tif->tif_encoderow = PredictorEncodeRow;
                    sp->encodestrip = tif->tif_encodestrip;
                    tif->tif_encodestrip = PredictorEncodeTile;
                    sp->encodetile = tif->tif_encodetile;
                    tif->tif_encodetile = PredictorEncodeTile;
                }

                /*
                 * If the data is horizontally differenced 16-bit data that
                 * requires byte-swapping, then it must be byte swapped after
                 * the differentiation step.  We do this with a special-purpose
                 * routine and override the normal post decoding logic that
                 * the library setup when the directory was read.
                 */
                if (tif->tif_flags & TIFF_SWAB) {
                    if (sp->encodepfunc == horDiff16) {
                            sp->encodepfunc = swabHorDiff16;
                            tif->tif_postdecode = _TIFFNoPostDecode;
                    } else if (sp->encodepfunc == horDiff32) {
                            sp->encodepfunc = swabHorDiff32;
                            tif->tif_postdecode = _TIFFNoPostDecode;
                    }
                }
        }

	else if (sp->predictor == 3) {
		sp->encodepfunc = fpDiff;
		/*
		 * Override default encoding method with one that does the
		 * predictor stuff.
		 */
                if( tif->tif_encoderow != PredictorEncodeRow )
                {
                    sp->encoderow = tif->tif_encoderow;
                    tif->tif_encoderow = PredictorEncodeRow;
                    sp->encodestrip = tif->tif_encodestrip;
                    tif->tif_encodestrip = PredictorEncodeTile;
                    sp->encodetile = tif->tif_encodetile;
                    tif->tif_encodetile = PredictorEncodeTile;
                }
	}

	return 1;
}

#define REPEAT4(n, op)		\
    switch (n) {		\
    default: { \
        tmsize_t i; for (i = n-4; i > 0; i--) { op; } }  /*-fallthrough*/  \
    case 4:  op; /*-fallthrough*/ \
    case 3:  op; /*-fallthrough*/ \
    case 2:  op; /*-fallthrough*/ \
    case 1:  op; /*-fallthrough*/ \
    case 0:  ;			\
    }

/* Remarks related to C standard compliance in all below functions : */
/* - to avoid any undefined behaviour, we only operate on unsigned types */
/*   since the behaviour of "overflows" is defined (wrap over) */
/* - when storing into the byte stream, we explicitly mask with 0xff so */
/*   as to make icc -check=conversions happy (not necessary by the standard) */

TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static int
horAcc8(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	tmsize_t stride = PredictorState(tif)->stride;

	unsigned char* cp = (unsigned char*) cp0;
    if((cc%stride)!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "horAcc8",
                     "%s", "(cc%stride)!=0");
        return 0;
    }

	if (cc > stride) {
		/*
		 * Pipeline the most common cases.
		 */
		if (stride == 3)  {
			unsigned int cr = cp[0];
			unsigned int cg = cp[1];
			unsigned int cb = cp[2];
			cc -= 3;
			cp += 3;
			while (cc>0) {
				cp[0] = (unsigned char) ((cr += cp[0]) & 0xff);
				cp[1] = (unsigned char) ((cg += cp[1]) & 0xff);
				cp[2] = (unsigned char) ((cb += cp[2]) & 0xff);
				cc -= 3;
				cp += 3;
			}
		} else if (stride == 4)  {
			unsigned int cr = cp[0];
			unsigned int cg = cp[1];
			unsigned int cb = cp[2];
			unsigned int ca = cp[3];
			cc -= 4;
			cp += 4;
			while (cc>0) {
				cp[0] = (unsigned char) ((cr += cp[0]) & 0xff);
				cp[1] = (unsigned char) ((cg += cp[1]) & 0xff);
				cp[2] = (unsigned char) ((cb += cp[2]) & 0xff);
				cp[3] = (unsigned char) ((ca += cp[3]) & 0xff);
				cc -= 4;
				cp += 4;
			}
		} else  {
			cc -= stride;
			do {
				REPEAT4(stride, cp[stride] =
					(unsigned char) ((cp[stride] + *cp) & 0xff); cp++)
				cc -= stride;
			} while (cc>0);
		}
	}
	return 1;
}

static int
swabHorAcc16(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	uint16* wp = (uint16*) cp0;
	tmsize_t wc = cc / 2;

        TIFFSwabArrayOfShort(wp, wc);
        return horAcc16(tif, cp0, cc);
}

TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static int
horAcc16(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	tmsize_t stride = PredictorState(tif)->stride;
	uint16* wp = (uint16*) cp0;
	tmsize_t wc = cc / 2;

    if((cc%(2*stride))!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "horAcc16",
                     "%s", "cc%(2*stride))!=0");
        return 0;
    }

	if (wc > stride) {
		wc -= stride;
		do {
			REPEAT4(stride, wp[stride] = (uint16)(((unsigned int)wp[stride] + (unsigned int)wp[0]) & 0xffff); wp++)
			wc -= stride;
		} while (wc > 0);
	}
	return 1;
}

static int
swabHorAcc32(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	uint32* wp = (uint32*) cp0;
	tmsize_t wc = cc / 4;

        TIFFSwabArrayOfLong(wp, wc);
	return horAcc32(tif, cp0, cc);
}

TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static int
horAcc32(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	tmsize_t stride = PredictorState(tif)->stride;
	uint32* wp = (uint32*) cp0;
	tmsize_t wc = cc / 4;

    if((cc%(4*stride))!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "horAcc32",
                     "%s", "cc%(4*stride))!=0");
        return 0;
    }

	if (wc > stride) {
		wc -= stride;
		do {
			REPEAT4(stride, wp[stride] += wp[0]; wp++)
			wc -= stride;
		} while (wc > 0);
	}
	return 1;
}

/*
 * Floating point predictor accumulation routine.
 */
static int
fpAcc(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	tmsize_t stride = PredictorState(tif)->stride;
	uint32 bps = tif->tif_dir.td_bitspersample / 8;
	tmsize_t wc = cc / bps;
	tmsize_t count = cc;
	uint8 *cp = (uint8 *) cp0;
	uint8 *tmp;

    if(cc%(bps*stride)!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "fpAcc",
                     "%s", "cc%(bps*stride))!=0");
        return 0;
    }

    tmp = (uint8 *)_TIFFmalloc(cc);
	if (!tmp)
		return 0;

	while (count > stride) {
		REPEAT4(stride, cp[stride] =
                        (unsigned char) ((cp[stride] + cp[0]) & 0xff); cp++)
		count -= stride;
	}

	_TIFFmemcpy(tmp, cp0, cc);
	cp = (uint8 *) cp0;
	for (count = 0; count < wc; count++) {
		uint32 byte;
		for (byte = 0; byte < bps; byte++) {
			#if WORDS_BIGENDIAN
			cp[bps * count + byte] = tmp[byte * wc + count];
			#else
			cp[bps * count + byte] =
				tmp[(bps - byte - 1) * wc + count];
			#endif
		}
	}
	_TIFFfree(tmp);
    return 1;
}

/*
 * Decode a scanline and apply the predictor routine.
 */
static int
PredictorDecodeRow(TIFF* tif, uint8* op0, tmsize_t occ0, uint16 s)
{
	TIFFPredictorState *sp = PredictorState(tif);

	assert(sp != NULL);
	assert(sp->decoderow != NULL);
	assert(sp->decodepfunc != NULL);  

	if ((*sp->decoderow)(tif, op0, occ0, s)) {
		return (*sp->decodepfunc)(tif, op0, occ0);
	} else
		return 0;
}

/*
 * Decode a tile/strip and apply the predictor routine.
 * Note that horizontal differencing must be done on a
 * row-by-row basis.  The width of a "row" has already
 * been calculated at pre-decode time according to the
 * strip/tile dimensions.
 */
static int
PredictorDecodeTile(TIFF* tif, uint8* op0, tmsize_t occ0, uint16 s)
{
	TIFFPredictorState *sp = PredictorState(tif);

	assert(sp != NULL);
	assert(sp->decodetile != NULL);

	if ((*sp->decodetile)(tif, op0, occ0, s)) {
		tmsize_t rowsize = sp->rowsize;
		assert(rowsize > 0);
		if((occ0%rowsize) !=0)
        {
            TIFFErrorExt(tif->tif_clientdata, "PredictorDecodeTile",
                         "%s", "occ0%rowsize != 0");
            return 0;
        }
		assert(sp->decodepfunc != NULL);
		while (occ0 > 0) {
			if( !(*sp->decodepfunc)(tif, op0, rowsize) )
                return 0;
			occ0 -= rowsize;
			op0 += rowsize;
		}
		return 1;
	} else
		return 0;
}

TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static int
horDiff8(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	TIFFPredictorState* sp = PredictorState(tif);
	tmsize_t stride = sp->stride;
	unsigned char* cp = (unsigned char*) cp0;

    if((cc%stride)!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "horDiff8",
                     "%s", "(cc%stride)!=0");
        return 0;
    }

	if (cc > stride) {
		cc -= stride;
		/*
		 * Pipeline the most common cases.
		 */
		if (stride == 3) {
			unsigned int r1, g1, b1;
			unsigned int r2 = cp[0];
			unsigned int g2 = cp[1];
			unsigned  int b2 = cp[2];
			do {
				r1 = cp[3]; cp[3] = (unsigned char)((r1-r2)&0xff); r2 = r1;
				g1 = cp[4]; cp[4] = (unsigned char)((g1-g2)&0xff); g2 = g1;
				b1 = cp[5]; cp[5] = (unsigned char)((b1-b2)&0xff); b2 = b1;
				cp += 3;
			} while ((cc -= 3) > 0);
		} else if (stride == 4) {
			unsigned int r1, g1, b1, a1;
			unsigned int r2 = cp[0];
			unsigned int g2 = cp[1];
			unsigned int b2 = cp[2];
			unsigned int a2 = cp[3];
			do {
				r1 = cp[4]; cp[4] = (unsigned char)((r1-r2)&0xff); r2 = r1;
				g1 = cp[5]; cp[5] = (unsigned char)((g1-g2)&0xff); g2 = g1;
				b1 = cp[6]; cp[6] = (unsigned char)((b1-b2)&0xff); b2 = b1;
				a1 = cp[7]; cp[7] = (unsigned char)((a1-a2)&0xff); a2 = a1;
				cp += 4;
			} while ((cc -= 4) > 0);
		} else {
			cp += cc - 1;
			do {
				REPEAT4(stride, cp[stride] = (unsigned char)((cp[stride] - cp[0])&0xff); cp--)
			} while ((cc -= stride) > 0);
		}
	}
	return 1;
}

TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static int
horDiff16(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	TIFFPredictorState* sp = PredictorState(tif);
	tmsize_t stride = sp->stride;
	uint16 *wp = (uint16*) cp0;
	tmsize_t wc = cc/2;

    if((cc%(2*stride))!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "horDiff8",
                     "%s", "(cc%(2*stride))!=0");
        return 0;
    }

	if (wc > stride) {
		wc -= stride;
		wp += wc - 1;
		do {
			REPEAT4(stride, wp[stride] = (uint16)(((unsigned int)wp[stride] - (unsigned int)wp[0]) & 0xffff); wp--)
			wc -= stride;
		} while (wc > 0);
	}
	return 1;
}

static int
swabHorDiff16(TIFF* tif, uint8* cp0, tmsize_t cc)
{
    uint16* wp = (uint16*) cp0;
    tmsize_t wc = cc / 2;

    if( !horDiff16(tif, cp0, cc) )
        return 0;

    TIFFSwabArrayOfShort(wp, wc);
    return 1;
}

TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static int
horDiff32(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	TIFFPredictorState* sp = PredictorState(tif);
	tmsize_t stride = sp->stride;
	uint32 *wp = (uint32*) cp0;
	tmsize_t wc = cc/4;

    if((cc%(4*stride))!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "horDiff32",
                     "%s", "(cc%(4*stride))!=0");
        return 0;
    }

	if (wc > stride) {
		wc -= stride;
		wp += wc - 1;
		do {
			REPEAT4(stride, wp[stride] -= wp[0]; wp--)
			wc -= stride;
		} while (wc > 0);
	}
	return 1;
}

static int
swabHorDiff32(TIFF* tif, uint8* cp0, tmsize_t cc)
{
    uint32* wp = (uint32*) cp0;
    tmsize_t wc = cc / 4;

    if( !horDiff32(tif, cp0, cc) )
        return 0;

    TIFFSwabArrayOfLong(wp, wc);
    return 1;
}

/*
 * Floating point predictor differencing routine.
 */
TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
static int
fpDiff(TIFF* tif, uint8* cp0, tmsize_t cc)
{
	tmsize_t stride = PredictorState(tif)->stride;
	uint32 bps = tif->tif_dir.td_bitspersample / 8;
	tmsize_t wc = cc / bps;
	tmsize_t count;
	uint8 *cp = (uint8 *) cp0;
	uint8 *tmp;

    if((cc%(bps*stride))!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "fpDiff",
                     "%s", "(cc%(bps*stride))!=0");
        return 0;
    }

    tmp = (uint8 *)_TIFFmalloc(cc);
	if (!tmp)
		return 0;

	_TIFFmemcpy(tmp, cp0, cc);
	for (count = 0; count < wc; count++) {
		uint32 byte;
		for (byte = 0; byte < bps; byte++) {
			#if WORDS_BIGENDIAN
			cp[byte * wc + count] = tmp[bps * count + byte];
			#else
			cp[(bps - byte - 1) * wc + count] =
				tmp[bps * count + byte];
			#endif
		}
	}
	_TIFFfree(tmp);

	cp = (uint8 *) cp0;
	cp += cc - stride - 1;
	for (count = cc; count > stride; count -= stride)
		REPEAT4(stride, cp[stride] = (unsigned char)((cp[stride] - cp[0])&0xff); cp--)
    return 1;
}

static int
PredictorEncodeRow(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s)
{
	TIFFPredictorState *sp = PredictorState(tif);

	assert(sp != NULL);
	assert(sp->encodepfunc != NULL);
	assert(sp->encoderow != NULL);

	/* XXX horizontal differencing alters user's data XXX */
	if( !(*sp->encodepfunc)(tif, bp, cc) )
        return 0;
	return (*sp->encoderow)(tif, bp, cc, s);
}

static int
PredictorEncodeTile(TIFF* tif, uint8* bp0, tmsize_t cc0, uint16 s)
{
	static const char module[] = "PredictorEncodeTile";
	TIFFPredictorState *sp = PredictorState(tif);
        uint8 *working_copy;
	tmsize_t cc = cc0, rowsize;
	unsigned char* bp;
        int result_code;

	assert(sp != NULL);
	assert(sp->encodepfunc != NULL);
	assert(sp->encodetile != NULL);

        /* 
         * Do predictor manipulation in a working buffer to avoid altering
         * the callers buffer. http://trac.osgeo.org/gdal/ticket/1965
         */
        working_copy = (uint8*) _TIFFmalloc(cc0);
        if( working_copy == NULL )
        {
            TIFFErrorExt(tif->tif_clientdata, module, 
                         "Out of memory allocating " TIFF_SSIZE_FORMAT " byte temp buffer.",
                         cc0 );
            return 0;
        }
        memcpy( working_copy, bp0, cc0 );
        bp = working_copy;

	rowsize = sp->rowsize;
	assert(rowsize > 0);
	if((cc0%rowsize)!=0)
    {
        TIFFErrorExt(tif->tif_clientdata, "PredictorEncodeTile",
                     "%s", "(cc0%rowsize)!=0");
        _TIFFfree( working_copy );
        return 0;
    }
	while (cc > 0) {
		(*sp->encodepfunc)(tif, bp, rowsize);
		cc -= rowsize;
		bp += rowsize;
	}
	result_code = (*sp->encodetile)(tif, working_copy, cc0, s);

        _TIFFfree( working_copy );

        return result_code;
}

#define	FIELD_PREDICTOR	(FIELD_CODEC+0)		/* XXX */

static const TIFFField predictFields[] = {
    { TIFFTAG_PREDICTOR, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16, TIFF_SETGET_UINT16, FIELD_PREDICTOR, FALSE, FALSE, "Predictor", NULL },
};

static int
PredictorVSetField(TIFF* tif, uint32 tag, va_list ap)
{
	TIFFPredictorState *sp = PredictorState(tif);

	assert(sp != NULL);
	assert(sp->vsetparent != NULL);

	switch (tag) {
	case TIFFTAG_PREDICTOR:
		sp->predictor = (uint16) va_arg(ap, uint16_vap);
		TIFFSetFieldBit(tif, FIELD_PREDICTOR);
		break;
	default:
		return (*sp->vsetparent)(tif, tag, ap);
	}
	tif->tif_flags |= TIFF_DIRTYDIRECT;
	return 1;
}

static int
PredictorVGetField(TIFF* tif, uint32 tag, va_list ap)
{
	TIFFPredictorState *sp = PredictorState(tif);

	assert(sp != NULL);
	assert(sp->vgetparent != NULL);

	switch (tag) {
	case TIFFTAG_PREDICTOR:
		*va_arg(ap, uint16*) = (uint16)sp->predictor;
		break;
	default:
		return (*sp->vgetparent)(tif, tag, ap);
	}
	return 1;
}

static void
PredictorPrintDir(TIFF* tif, FILE* fd, long flags)
{
	TIFFPredictorState* sp = PredictorState(tif);

	(void) flags;
	if (TIFFFieldSet(tif,FIELD_PREDICTOR)) {
		fprintf(fd, "  Predictor: ");
		switch (sp->predictor) {
			case 1: fprintf(fd, "none "); break;
			case 2: fprintf(fd, "horizontal differencing "); break;
			case 3: fprintf(fd, "floating point predictor "); break;
		}
		fprintf(fd, "%d (0x%x)\n", sp->predictor, sp->predictor);
	}
	if (sp->printdir)
		(*sp->printdir)(tif, fd, flags);
}

int
TIFFPredictorInit(TIFF* tif)
{
	TIFFPredictorState* sp = PredictorState(tif);

	assert(sp != 0);

	/*
	 * Merge codec-specific tag information.
	 */
	if (!_TIFFMergeFields(tif, predictFields,
			      TIFFArrayCount(predictFields))) {
		TIFFErrorExt(tif->tif_clientdata, "TIFFPredictorInit",
		    "Merging Predictor codec-specific tags failed");
		return 0;
	}

	/*
	 * Override parent get/set field methods.
	 */
	sp->vgetparent = tif->tif_tagmethods.vgetfield;
	tif->tif_tagmethods.vgetfield =
            PredictorVGetField;/* hook for predictor tag */
	sp->vsetparent = tif->tif_tagmethods.vsetfield;
	tif->tif_tagmethods.vsetfield =
	    PredictorVSetField;/* hook for predictor tag */
	sp->printdir = tif->tif_tagmethods.printdir;
	tif->tif_tagmethods.printdir =
            PredictorPrintDir;	/* hook for predictor tag */

	sp->setupdecode = tif->tif_setupdecode;
	tif->tif_setupdecode = PredictorSetupDecode;
	sp->setupencode = tif->tif_setupencode;
	tif->tif_setupencode = PredictorSetupEncode;

	sp->predictor = 1;			/* default value */
	sp->encodepfunc = NULL;			/* no predictor routine */
	sp->decodepfunc = NULL;			/* no predictor routine */
	return 1;
}

int
TIFFPredictorCleanup(TIFF* tif)
{
	TIFFPredictorState* sp = PredictorState(tif);

	assert(sp != 0);

	tif->tif_tagmethods.vgetfield = sp->vgetparent;
	tif->tif_tagmethods.vsetfield = sp->vsetparent;
	tif->tif_tagmethods.printdir = sp->printdir;
	tif->tif_setupdecode = sp->setupdecode;
	tif->tif_setupencode = sp->setupencode;

	return 1;
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
