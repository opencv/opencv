/* $Id: tif_dir.c,v 1.113 2012-06-14 20:32:53 fwarmerdam Exp $ */

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
 * Directory Tag Get & Set Routines.
 * (and also some miscellaneous stuff)
 */
#include "tiffiop.h"

/*
 * These are used in the backwards compatibility code...
 */
#define DATATYPE_VOID		0       /* !untyped data */
#define DATATYPE_INT		1       /* !signed integer data */
#define DATATYPE_UINT		2       /* !unsigned integer data */
#define DATATYPE_IEEEFP		3       /* !IEEE floating point data */

static void
setByteArray(void** vpp, void* vp, size_t nmemb, size_t elem_size)
{
    if (*vpp)
        _TIFFfree(*vpp), *vpp = 0;
    if (vp) {
        tmsize_t bytes = (tmsize_t)(nmemb * elem_size);
        if (elem_size && bytes / elem_size == nmemb)
            *vpp = (void*) _TIFFmalloc(bytes);
        if (*vpp)
            _TIFFmemcpy(*vpp, vp, bytes);
    }
}
void _TIFFsetByteArray(void** vpp, void* vp, uint32 n)
    { setByteArray(vpp, vp, n, 1); }
void _TIFFsetString(char** cpp, char* cp)
    { setByteArray((void**) cpp, (void*) cp, strlen(cp)+1, 1); }
void _TIFFsetNString(char** cpp, char* cp, uint32 n)
    { setByteArray((void**) cpp, (void*) cp, n, 1); }
void _TIFFsetShortArray(uint16** wpp, uint16* wp, uint32 n)
    { setByteArray((void**) wpp, (void*) wp, n, sizeof (uint16)); }
void _TIFFsetLongArray(uint32** lpp, uint32* lp, uint32 n)
    { setByteArray((void**) lpp, (void*) lp, n, sizeof (uint32)); }
void _TIFFsetLong8Array(uint64** lpp, uint64* lp, uint32 n)
    { setByteArray((void**) lpp, (void*) lp, n, sizeof (uint64)); }
void _TIFFsetFloatArray(float** fpp, float* fp, uint32 n)
    { setByteArray((void**) fpp, (void*) fp, n, sizeof (float)); }
void _TIFFsetDoubleArray(double** dpp, double* dp, uint32 n)
    { setByteArray((void**) dpp, (void*) dp, n, sizeof (double)); }

static void
setDoubleArrayOneValue(double** vpp, double value, size_t nmemb)
{
    if (*vpp)
        _TIFFfree(*vpp);
    *vpp = _TIFFmalloc(nmemb*sizeof(double));
    if (*vpp)
    {
        while (nmemb--)
            ((double*)*vpp)[nmemb] = value;
    }
}

/*
 * Install extra samples information.
 */
static int
setExtraSamples(TIFFDirectory* td, va_list ap, uint32* v)
{
/* XXX: Unassociated alpha data == 999 is a known Corel Draw bug, see below */
#define EXTRASAMPLE_COREL_UNASSALPHA 999

    uint16* va;
    uint32 i;

    *v = (uint16) va_arg(ap, uint16_vap);
    if ((uint16) *v > td->td_samplesperpixel)
        return 0;
    va = va_arg(ap, uint16*);
    if (*v > 0 && va == NULL)		/* typically missing param */
        return 0;
    for (i = 0; i < *v; i++) {
        if (va[i] > EXTRASAMPLE_UNASSALPHA) {
            /*
             * XXX: Corel Draw is known to produce incorrect
             * ExtraSamples tags which must be patched here if we
             * want to be able to open some of the damaged TIFF
             * files:
             */
            if (va[i] == EXTRASAMPLE_COREL_UNASSALPHA)
                va[i] = EXTRASAMPLE_UNASSALPHA;
            else
                return 0;
        }
    }
    td->td_extrasamples = (uint16) *v;
    _TIFFsetShortArray(&td->td_sampleinfo, va, td->td_extrasamples);
    return 1;

#undef EXTRASAMPLE_COREL_UNASSALPHA
}

/*
 * Confirm we have "samplesperpixel" ink names separated by \0.  Returns
 * zero if the ink names are not as expected.
 */
static uint32
checkInkNamesString(TIFF* tif, uint32 slen, const char* s)
{
    TIFFDirectory* td = &tif->tif_dir;
    uint16 i = td->td_samplesperpixel;

    if (slen > 0) {
        const char* ep = s+slen;
        const char* cp = s;
        for (; i > 0; i--) {
            for (; cp < ep && *cp != '\0'; cp++) {}
            if (cp >= ep)
                goto bad;
            cp++;				/* skip \0 */
        }
        return ((uint32)(cp-s));
    }
bad:
    TIFFErrorExt(tif->tif_clientdata, "TIFFSetField",
        "%s: Invalid InkNames value; expecting %d names, found %d",
        tif->tif_name,
        td->td_samplesperpixel,
        td->td_samplesperpixel-i);
    return (0);
}

static int
_TIFFVSetField(TIFF* tif, uint32 tag, va_list ap)
{
    static const char module[] = "_TIFFVSetField";

    TIFFDirectory* td = &tif->tif_dir;
    int status = 1;
    uint32 v32, i, v;
    char* s;
    const TIFFField *fip = TIFFFindField(tif, tag, TIFF_ANY);
    uint32 standard_tag = tag;

    /*
     * We want to force the custom code to be used for custom
     * fields even if the tag happens to match a well known
     * one - important for reinterpreted handling of standard
     * tag values in custom directories (ie. EXIF)
     */
    if (fip->field_bit == FIELD_CUSTOM) {
        standard_tag = 0;
    }

    switch (standard_tag) {
    case TIFFTAG_SUBFILETYPE:
        td->td_subfiletype = (uint32) va_arg(ap, uint32);
        break;
    case TIFFTAG_IMAGEWIDTH:
        td->td_imagewidth = (uint32) va_arg(ap, uint32);
        break;
    case TIFFTAG_IMAGELENGTH:
        td->td_imagelength = (uint32) va_arg(ap, uint32);
        break;
    case TIFFTAG_BITSPERSAMPLE:
        td->td_bitspersample = (uint16) va_arg(ap, uint16_vap);
        /*
         * If the data require post-decoding processing to byte-swap
         * samples, set it up here.  Note that since tags are required
         * to be ordered, compression code can override this behaviour
         * in the setup method if it wants to roll the post decoding
         * work in with its normal work.
         */
        if (tif->tif_flags & TIFF_SWAB) {
            if (td->td_bitspersample == 8)
                tif->tif_postdecode = _TIFFNoPostDecode;
            else if (td->td_bitspersample == 16)
                tif->tif_postdecode = _TIFFSwab16BitData;
            else if (td->td_bitspersample == 24)
                tif->tif_postdecode = _TIFFSwab24BitData;
            else if (td->td_bitspersample == 32)
                tif->tif_postdecode = _TIFFSwab32BitData;
            else if (td->td_bitspersample == 64)
                tif->tif_postdecode = _TIFFSwab64BitData;
            else if (td->td_bitspersample == 128) /* two 64's */
                tif->tif_postdecode = _TIFFSwab64BitData;
        }
        break;
    case TIFFTAG_COMPRESSION:
        v = (uint16) va_arg(ap, uint16_vap);
        /*
         * If we're changing the compression scheme, the notify the
         * previous module so that it can cleanup any state it's
         * setup.
         */
        if (TIFFFieldSet(tif, FIELD_COMPRESSION)) {
            if ((uint32)td->td_compression == v)
                break;
            (*tif->tif_cleanup)(tif);
            tif->tif_flags &= ~TIFF_CODERSETUP;
        }
        /*
         * Setup new compression routine state.
         */
        if( (status = TIFFSetCompressionScheme(tif, v)) != 0 )
            td->td_compression = (uint16) v;
        else
            status = 0;
        break;
    case TIFFTAG_PHOTOMETRIC:
        td->td_photometric = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_THRESHHOLDING:
        td->td_threshholding = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_FILLORDER:
        v = (uint16) va_arg(ap, uint16_vap);
        if (v != FILLORDER_LSB2MSB && v != FILLORDER_MSB2LSB)
            goto badvalue;
        td->td_fillorder = (uint16) v;
        break;
    case TIFFTAG_ORIENTATION:
        v = (uint16) va_arg(ap, uint16_vap);
        if (v < ORIENTATION_TOPLEFT || ORIENTATION_LEFTBOT < v)
            goto badvalue;
        else
            td->td_orientation = (uint16) v;
        break;
    case TIFFTAG_SAMPLESPERPIXEL:
        v = (uint16) va_arg(ap, uint16_vap);
        if (v == 0)
            goto badvalue;
        td->td_samplesperpixel = (uint16) v;
        break;
    case TIFFTAG_ROWSPERSTRIP:
        v32 = (uint32) va_arg(ap, uint32);
        if (v32 == 0)
            goto badvalue32;
        td->td_rowsperstrip = v32;
        if (!TIFFFieldSet(tif, FIELD_TILEDIMENSIONS)) {
            td->td_tilelength = v32;
            td->td_tilewidth = td->td_imagewidth;
        }
        break;
    case TIFFTAG_MINSAMPLEVALUE:
        td->td_minsamplevalue = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_MAXSAMPLEVALUE:
        td->td_maxsamplevalue = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_SMINSAMPLEVALUE:
        if (tif->tif_flags & TIFF_PERSAMPLE)
            _TIFFsetDoubleArray(&td->td_sminsamplevalue, va_arg(ap, double*), td->td_samplesperpixel);
        else
            setDoubleArrayOneValue(&td->td_sminsamplevalue, va_arg(ap, double), td->td_samplesperpixel);
        break;
    case TIFFTAG_SMAXSAMPLEVALUE:
        if (tif->tif_flags & TIFF_PERSAMPLE)
            _TIFFsetDoubleArray(&td->td_smaxsamplevalue, va_arg(ap, double*), td->td_samplesperpixel);
        else
            setDoubleArrayOneValue(&td->td_smaxsamplevalue, va_arg(ap, double), td->td_samplesperpixel);
        break;
    case TIFFTAG_XRESOLUTION:
        td->td_xresolution = (float) va_arg(ap, double);
        break;
    case TIFFTAG_YRESOLUTION:
        td->td_yresolution = (float) va_arg(ap, double);
        break;
    case TIFFTAG_PLANARCONFIG:
        v = (uint16) va_arg(ap, uint16_vap);
        if (v != PLANARCONFIG_CONTIG && v != PLANARCONFIG_SEPARATE)
            goto badvalue;
        td->td_planarconfig = (uint16) v;
        break;
    case TIFFTAG_XPOSITION:
        td->td_xposition = (float) va_arg(ap, double);
        break;
    case TIFFTAG_YPOSITION:
        td->td_yposition = (float) va_arg(ap, double);
        break;
    case TIFFTAG_RESOLUTIONUNIT:
        v = (uint16) va_arg(ap, uint16_vap);
        if (v < RESUNIT_NONE || RESUNIT_CENTIMETER < v)
            goto badvalue;
        td->td_resolutionunit = (uint16) v;
        break;
    case TIFFTAG_PAGENUMBER:
        td->td_pagenumber[0] = (uint16) va_arg(ap, uint16_vap);
        td->td_pagenumber[1] = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_HALFTONEHINTS:
        td->td_halftonehints[0] = (uint16) va_arg(ap, uint16_vap);
        td->td_halftonehints[1] = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_COLORMAP:
        v32 = (uint32)(1L<<td->td_bitspersample);
        _TIFFsetShortArray(&td->td_colormap[0], va_arg(ap, uint16*), v32);
        _TIFFsetShortArray(&td->td_colormap[1], va_arg(ap, uint16*), v32);
        _TIFFsetShortArray(&td->td_colormap[2], va_arg(ap, uint16*), v32);
        break;
    case TIFFTAG_EXTRASAMPLES:
        if (!setExtraSamples(td, ap, &v))
            goto badvalue;
        break;
    case TIFFTAG_MATTEING:
        td->td_extrasamples =  (((uint16) va_arg(ap, uint16_vap)) != 0);
        if (td->td_extrasamples) {
            uint16 sv = EXTRASAMPLE_ASSOCALPHA;
            _TIFFsetShortArray(&td->td_sampleinfo, &sv, 1);
        }
        break;
    case TIFFTAG_TILEWIDTH:
        v32 = (uint32) va_arg(ap, uint32);
        if (v32 % 16) {
            if (tif->tif_mode != O_RDONLY)
                goto badvalue32;
            TIFFWarningExt(tif->tif_clientdata, tif->tif_name,
                "Nonstandard tile width %d, convert file", v32);
        }
        td->td_tilewidth = v32;
        tif->tif_flags |= TIFF_ISTILED;
        break;
    case TIFFTAG_TILELENGTH:
        v32 = (uint32) va_arg(ap, uint32);
        if (v32 % 16) {
            if (tif->tif_mode != O_RDONLY)
                goto badvalue32;
            TIFFWarningExt(tif->tif_clientdata, tif->tif_name,
                "Nonstandard tile length %d, convert file", v32);
        }
        td->td_tilelength = v32;
        tif->tif_flags |= TIFF_ISTILED;
        break;
    case TIFFTAG_TILEDEPTH:
        v32 = (uint32) va_arg(ap, uint32);
        if (v32 == 0)
            goto badvalue32;
        td->td_tiledepth = v32;
        break;
    case TIFFTAG_DATATYPE:
        v = (uint16) va_arg(ap, uint16_vap);
        switch (v) {
        case DATATYPE_VOID:	v = SAMPLEFORMAT_VOID;	break;
        case DATATYPE_INT:	v = SAMPLEFORMAT_INT;	break;
        case DATATYPE_UINT:	v = SAMPLEFORMAT_UINT;	break;
        case DATATYPE_IEEEFP:	v = SAMPLEFORMAT_IEEEFP;break;
        default:		goto badvalue;
        }
        td->td_sampleformat = (uint16) v;
        break;
    case TIFFTAG_SAMPLEFORMAT:
        v = (uint16) va_arg(ap, uint16_vap);
        if (v < SAMPLEFORMAT_UINT || SAMPLEFORMAT_COMPLEXIEEEFP < v)
            goto badvalue;
        td->td_sampleformat = (uint16) v;

        /*  Try to fix up the SWAB function for complex data. */
        if( td->td_sampleformat == SAMPLEFORMAT_COMPLEXINT
            && td->td_bitspersample == 32
            && tif->tif_postdecode == _TIFFSwab32BitData )
            tif->tif_postdecode = _TIFFSwab16BitData;
        else if( (td->td_sampleformat == SAMPLEFORMAT_COMPLEXINT
              || td->td_sampleformat == SAMPLEFORMAT_COMPLEXIEEEFP)
             && td->td_bitspersample == 64
             && tif->tif_postdecode == _TIFFSwab64BitData )
            tif->tif_postdecode = _TIFFSwab32BitData;
        break;
    case TIFFTAG_IMAGEDEPTH:
        td->td_imagedepth = (uint32) va_arg(ap, uint32);
        break;
    case TIFFTAG_SUBIFD:
        if ((tif->tif_flags & TIFF_INSUBIFD) == 0) {
            td->td_nsubifd = (uint16) va_arg(ap, uint16_vap);
            _TIFFsetLong8Array(&td->td_subifd, (uint64*) va_arg(ap, uint64*),
                (long) td->td_nsubifd);
        } else {
            TIFFErrorExt(tif->tif_clientdata, module,
                     "%s: Sorry, cannot nest SubIFDs",
                     tif->tif_name);
            status = 0;
        }
        break;
    case TIFFTAG_YCBCRPOSITIONING:
        td->td_ycbcrpositioning = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_YCBCRSUBSAMPLING:
        td->td_ycbcrsubsampling[0] = (uint16) va_arg(ap, uint16_vap);
        td->td_ycbcrsubsampling[1] = (uint16) va_arg(ap, uint16_vap);
        break;
    case TIFFTAG_TRANSFERFUNCTION:
        v = (td->td_samplesperpixel - td->td_extrasamples) > 1 ? 3 : 1;
        for (i = 0; i < v; i++)
            _TIFFsetShortArray(&td->td_transferfunction[i],
                va_arg(ap, uint16*), 1L<<td->td_bitspersample);
        break;
    case TIFFTAG_REFERENCEBLACKWHITE:
        /* XXX should check for null range */
        _TIFFsetFloatArray(&td->td_refblackwhite, va_arg(ap, float*), 6);
        break;
    case TIFFTAG_INKNAMES:
        v = (uint16) va_arg(ap, uint16_vap);
        s = va_arg(ap, char*);
        v = checkInkNamesString(tif, v, s);
        status = v > 0;
        if( v > 0 ) {
            _TIFFsetNString(&td->td_inknames, s, v);
            td->td_inknameslen = v;
        }
        break;
    case TIFFTAG_PERSAMPLE:
        v = (uint16) va_arg(ap, uint16_vap);
        if( v == PERSAMPLE_MULTI )
            tif->tif_flags |= TIFF_PERSAMPLE;
        else
            tif->tif_flags &= ~TIFF_PERSAMPLE;
        break;
    default: {
        TIFFTagValue *tv;
        int tv_size, iCustom;

        /*
         * This can happen if multiple images are open with different
         * codecs which have private tags.  The global tag information
         * table may then have tags that are valid for one file but not
         * the other. If the client tries to set a tag that is not valid
         * for the image's codec then we'll arrive here.  This
         * happens, for example, when tiffcp is used to convert between
         * compression schemes and codec-specific tags are blindly copied.
         */
        if(fip == NULL || fip->field_bit != FIELD_CUSTOM) {
            TIFFErrorExt(tif->tif_clientdata, module,
                "%s: Invalid %stag \"%s\" (not supported by codec)",
                tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "",
                fip ? fip->field_name : "Unknown");
            status = 0;
            break;
        }

        /*
         * Find the existing entry for this custom value.
         */
        tv = NULL;
        for (iCustom = 0; iCustom < td->td_customValueCount; iCustom++) {
            if (td->td_customValues[iCustom].info->field_tag == tag) {
                tv = td->td_customValues + iCustom;
                if (tv->value != NULL) {
                    _TIFFfree(tv->value);
                    tv->value = NULL;
                }
                break;
            }
        }

        /*
         * Grow the custom list if the entry was not found.
         */
        if(tv == NULL) {
            TIFFTagValue *new_customValues;

            td->td_customValueCount++;
            new_customValues = (TIFFTagValue *)
                _TIFFrealloc(td->td_customValues,
                sizeof(TIFFTagValue) * td->td_customValueCount);
            if (!new_customValues) {
                TIFFErrorExt(tif->tif_clientdata, module,
                    "%s: Failed to allocate space for list of custom values",
                    tif->tif_name);
                status = 0;
                goto end;
            }

            td->td_customValues = new_customValues;

            tv = td->td_customValues + (td->td_customValueCount - 1);
            tv->info = fip;
            tv->value = NULL;
            tv->count = 0;
        }

        /*
         * Set custom value ... save a copy of the custom tag value.
         */
        tv_size = _TIFFDataSize(fip->field_type);
        if (tv_size == 0) {
            status = 0;
            TIFFErrorExt(tif->tif_clientdata, module,
                "%s: Bad field type %d for \"%s\"",
                tif->tif_name, fip->field_type,
                fip->field_name);
            goto end;
        }

        if (fip->field_type == TIFF_ASCII)
        {
            uint32 ma;
            char* mb;
            if (fip->field_passcount)
            {
                assert(fip->field_writecount==TIFF_VARIABLE2);
                ma=(uint32)va_arg(ap,uint32);
                mb=(char*)va_arg(ap,char*);
            }
            else
            {
                mb=(char*)va_arg(ap,char*);
                ma=(uint32)(strlen(mb)+1);
            }
            tv->count=ma;
            setByteArray(&tv->value,mb,ma,1);
        }
        else
        {
            if (fip->field_passcount) {
                if (fip->field_writecount == TIFF_VARIABLE2)
                    tv->count = (uint32) va_arg(ap, uint32);
                else
                    tv->count = (int) va_arg(ap, int);
            } else if (fip->field_writecount == TIFF_VARIABLE
               || fip->field_writecount == TIFF_VARIABLE2)
                tv->count = 1;
            else if (fip->field_writecount == TIFF_SPP)
                tv->count = td->td_samplesperpixel;
            else
                tv->count = fip->field_writecount;

            if (tv->count == 0) {
                status = 0;
                TIFFErrorExt(tif->tif_clientdata, module,
                         "%s: Null count for \"%s\" (type "
                         "%d, writecount %d, passcount %d)",
                         tif->tif_name,
                         fip->field_name,
                         fip->field_type,
                         fip->field_writecount,
                         fip->field_passcount);
                goto end;
            }

            tv->value = _TIFFCheckMalloc(tif, tv->count, tv_size,
                "custom tag binary object");
            if (!tv->value) {
                status = 0;
                goto end;
            }

            if (fip->field_tag == TIFFTAG_DOTRANGE
                && strcmp(fip->field_name,"DotRange") == 0) {
                /* TODO: This is an evil exception and should not have been
                   handled this way ... likely best if we move it into
                   the directory structure with an explicit field in
                   libtiff 4.1 and assign it a FIELD_ value */
                uint16 v[2];
                v[0] = (uint16)va_arg(ap, int);
                v[1] = (uint16)va_arg(ap, int);
                _TIFFmemcpy(tv->value, &v, 4);
            }

            else if (fip->field_passcount
                  || fip->field_writecount == TIFF_VARIABLE
                  || fip->field_writecount == TIFF_VARIABLE2
                  || fip->field_writecount == TIFF_SPP
                  || tv->count > 1) {
                _TIFFmemcpy(tv->value, va_arg(ap, void *),
                    tv->count * tv_size);
            } else {
                char *val = (char *)tv->value;
                assert( tv->count == 1 );

                switch (fip->field_type) {
                case TIFF_BYTE:
                case TIFF_UNDEFINED:
                    {
                        uint8 v = (uint8)va_arg(ap, int);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_SBYTE:
                    {
                        int8 v = (int8)va_arg(ap, int);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_SHORT:
                    {
                        uint16 v = (uint16)va_arg(ap, int);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_SSHORT:
                    {
                        int16 v = (int16)va_arg(ap, int);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_LONG:
                case TIFF_IFD:
                    {
                        uint32 v = va_arg(ap, uint32);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_SLONG:
                    {
                        int32 v = va_arg(ap, int32);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_LONG8:
                case TIFF_IFD8:
                    {
                        uint64 v = va_arg(ap, uint64);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_SLONG8:
                    {
                        int64 v = va_arg(ap, int64);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_RATIONAL:
                case TIFF_SRATIONAL:
                case TIFF_FLOAT:
                    {
                        float v = (float)va_arg(ap, double);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                case TIFF_DOUBLE:
                    {
                        double v = va_arg(ap, double);
                        _TIFFmemcpy(val, &v, tv_size);
                    }
                    break;
                default:
                    _TIFFmemset(val, 0, tv_size);
                    status = 0;
                    break;
                }
            }
        }
    }
    }
    if (status) {
        const TIFFField* fip=TIFFFieldWithTag(tif,tag);
        if (fip)
            TIFFSetFieldBit(tif, fip->field_bit);
        tif->tif_flags |= TIFF_DIRTYDIRECT;
    }

end:
    va_end(ap);
    return (status);
badvalue:
        {
        const TIFFField* fip=TIFFFieldWithTag(tif,tag);
        TIFFErrorExt(tif->tif_clientdata, module,
             "%s: Bad value %u for \"%s\" tag",
             tif->tif_name, v,
             fip ? fip->field_name : "Unknown");
        va_end(ap);
        }
    return (0);
badvalue32:
        {
        const TIFFField* fip=TIFFFieldWithTag(tif,tag);
        TIFFErrorExt(tif->tif_clientdata, module,
             "%s: Bad value %u for \"%s\" tag",
             tif->tif_name, v32,
             fip ? fip->field_name : "Unknown");
        va_end(ap);
        }
    return (0);
}

/*
 * Return 1/0 according to whether or not
 * it is permissible to set the tag's value.
 * Note that we allow ImageLength to be changed
 * so that we can append and extend to images.
 * Any other tag may not be altered once writing
 * has commenced, unless its value has no effect
 * on the format of the data that is written.
 */
static int
OkToChangeTag(TIFF* tif, uint32 tag)
{
    const TIFFField* fip = TIFFFindField(tif, tag, TIFF_ANY);
    if (!fip) {			/* unknown tag */
        TIFFErrorExt(tif->tif_clientdata, "TIFFSetField", "%s: Unknown %stag %u",
            tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "", tag);
        return (0);
    }
    if (tag != TIFFTAG_IMAGELENGTH && (tif->tif_flags & TIFF_BEENWRITING) &&
        !fip->field_oktochange) {
        /*
         * Consult info table to see if tag can be changed
         * after we've started writing.  We only allow changes
         * to those tags that don't/shouldn't affect the
         * compression and/or format of the data.
         */
        TIFFErrorExt(tif->tif_clientdata, "TIFFSetField",
            "%s: Cannot modify tag \"%s\" while writing",
            tif->tif_name, fip->field_name);
        return (0);
    }
    return (1);
}

/*
 * Record the value of a field in the
 * internal directory structure.  The
 * field will be written to the file
 * when/if the directory structure is
 * updated.
 */
int
TIFFSetField(TIFF* tif, uint32 tag, ...)
{
    va_list ap;
    int status;

    va_start(ap, tag);
    status = TIFFVSetField(tif, tag, ap);
    va_end(ap);
    return (status);
}

/*
 * Clear the contents of the field in the internal structure.
 */
int
TIFFUnsetField(TIFF* tif, uint32 tag)
{
    const TIFFField *fip =  TIFFFieldWithTag(tif, tag);
    TIFFDirectory* td = &tif->tif_dir;

    if( !fip )
        return 0;

    if( fip->field_bit != FIELD_CUSTOM )
        TIFFClrFieldBit(tif, fip->field_bit);
    else
    {
        TIFFTagValue *tv = NULL;
        int i;

        for (i = 0; i < td->td_customValueCount; i++) {

            tv = td->td_customValues + i;
            if( tv->info->field_tag == tag )
                break;
        }

        if( i < td->td_customValueCount )
        {
            _TIFFfree(tv->value);
            for( ; i < td->td_customValueCount-1; i++) {
                td->td_customValues[i] = td->td_customValues[i+1];
            }
            td->td_customValueCount--;
        }
    }

    tif->tif_flags |= TIFF_DIRTYDIRECT;

    return (1);
}

/*
 * Like TIFFSetField, but taking a varargs
 * parameter list.  This routine is useful
 * for building higher-level interfaces on
 * top of the library.
 */
int
TIFFVSetField(TIFF* tif, uint32 tag, va_list ap)
{
    return OkToChangeTag(tif, tag) ?
        (*tif->tif_tagmethods.vsetfield)(tif, tag, ap) : 0;
}

static int
_TIFFVGetField(TIFF* tif, uint32 tag, va_list ap)
{
    TIFFDirectory* td = &tif->tif_dir;
    int ret_val = 1;
    uint32 standard_tag = tag;
    const TIFFField* fip = TIFFFindField(tif, tag, TIFF_ANY);

    /*
     * We want to force the custom code to be used for custom
     * fields even if the tag happens to match a well known
     * one - important for reinterpreted handling of standard
     * tag values in custom directories (ie. EXIF)
     */
    if (fip->field_bit == FIELD_CUSTOM) {
        standard_tag = 0;
    }

    switch (standard_tag) {
        case TIFFTAG_SUBFILETYPE:
            *va_arg(ap, uint32*) = td->td_subfiletype;
            break;
        case TIFFTAG_IMAGEWIDTH:
            *va_arg(ap, uint32*) = td->td_imagewidth;
            break;
        case TIFFTAG_IMAGELENGTH:
            *va_arg(ap, uint32*) = td->td_imagelength;
            break;
        case TIFFTAG_BITSPERSAMPLE:
            *va_arg(ap, uint16*) = td->td_bitspersample;
            break;
        case TIFFTAG_COMPRESSION:
            *va_arg(ap, uint16*) = td->td_compression;
            break;
        case TIFFTAG_PHOTOMETRIC:
            *va_arg(ap, uint16*) = td->td_photometric;
            break;
        case TIFFTAG_THRESHHOLDING:
            *va_arg(ap, uint16*) = td->td_threshholding;
            break;
        case TIFFTAG_FILLORDER:
            *va_arg(ap, uint16*) = td->td_fillorder;
            break;
        case TIFFTAG_ORIENTATION:
            *va_arg(ap, uint16*) = td->td_orientation;
            break;
        case TIFFTAG_SAMPLESPERPIXEL:
            *va_arg(ap, uint16*) = td->td_samplesperpixel;
            break;
        case TIFFTAG_ROWSPERSTRIP:
            *va_arg(ap, uint32*) = td->td_rowsperstrip;
            break;
        case TIFFTAG_MINSAMPLEVALUE:
            *va_arg(ap, uint16*) = td->td_minsamplevalue;
            break;
        case TIFFTAG_MAXSAMPLEVALUE:
            *va_arg(ap, uint16*) = td->td_maxsamplevalue;
            break;
        case TIFFTAG_SMINSAMPLEVALUE:
            if (tif->tif_flags & TIFF_PERSAMPLE)
                *va_arg(ap, double**) = td->td_sminsamplevalue;
            else
            {
                /* libtiff historially treats this as a single value. */
                uint16 i;
                double v = td->td_sminsamplevalue[0];
                for (i=1; i < td->td_samplesperpixel; ++i)
                    if( td->td_sminsamplevalue[i] < v )
                        v = td->td_sminsamplevalue[i];
                *va_arg(ap, double*) = v;
            }
            break;
        case TIFFTAG_SMAXSAMPLEVALUE:
            if (tif->tif_flags & TIFF_PERSAMPLE)
                *va_arg(ap, double**) = td->td_smaxsamplevalue;
            else
            {
                /* libtiff historially treats this as a single value. */
                uint16 i;
                double v = td->td_smaxsamplevalue[0];
                for (i=1; i < td->td_samplesperpixel; ++i)
                    if( td->td_smaxsamplevalue[i] > v )
                        v = td->td_smaxsamplevalue[i];
                *va_arg(ap, double*) = v;
            }
            break;
        case TIFFTAG_XRESOLUTION:
            *va_arg(ap, float*) = td->td_xresolution;
            break;
        case TIFFTAG_YRESOLUTION:
            *va_arg(ap, float*) = td->td_yresolution;
            break;
        case TIFFTAG_PLANARCONFIG:
            *va_arg(ap, uint16*) = td->td_planarconfig;
            break;
        case TIFFTAG_XPOSITION:
            *va_arg(ap, float*) = td->td_xposition;
            break;
        case TIFFTAG_YPOSITION:
            *va_arg(ap, float*) = td->td_yposition;
            break;
        case TIFFTAG_RESOLUTIONUNIT:
            *va_arg(ap, uint16*) = td->td_resolutionunit;
            break;
        case TIFFTAG_PAGENUMBER:
            *va_arg(ap, uint16*) = td->td_pagenumber[0];
            *va_arg(ap, uint16*) = td->td_pagenumber[1];
            break;
        case TIFFTAG_HALFTONEHINTS:
            *va_arg(ap, uint16*) = td->td_halftonehints[0];
            *va_arg(ap, uint16*) = td->td_halftonehints[1];
            break;
        case TIFFTAG_COLORMAP:
            *va_arg(ap, uint16**) = td->td_colormap[0];
            *va_arg(ap, uint16**) = td->td_colormap[1];
            *va_arg(ap, uint16**) = td->td_colormap[2];
            break;
        case TIFFTAG_STRIPOFFSETS:
        case TIFFTAG_TILEOFFSETS:
            _TIFFFillStriles( tif );
            *va_arg(ap, uint64**) = td->td_stripoffset;
            break;
        case TIFFTAG_STRIPBYTECOUNTS:
        case TIFFTAG_TILEBYTECOUNTS:
            _TIFFFillStriles( tif );
            *va_arg(ap, uint64**) = td->td_stripbytecount;
            break;
        case TIFFTAG_MATTEING:
            *va_arg(ap, uint16*) =
                (td->td_extrasamples == 1 &&
                td->td_sampleinfo[0] == EXTRASAMPLE_ASSOCALPHA);
            break;
        case TIFFTAG_EXTRASAMPLES:
            *va_arg(ap, uint16*) = td->td_extrasamples;
            *va_arg(ap, uint16**) = td->td_sampleinfo;
            break;
        case TIFFTAG_TILEWIDTH:
            *va_arg(ap, uint32*) = td->td_tilewidth;
            break;
        case TIFFTAG_TILELENGTH:
            *va_arg(ap, uint32*) = td->td_tilelength;
            break;
        case TIFFTAG_TILEDEPTH:
            *va_arg(ap, uint32*) = td->td_tiledepth;
            break;
        case TIFFTAG_DATATYPE:
            switch (td->td_sampleformat) {
                case SAMPLEFORMAT_UINT:
                    *va_arg(ap, uint16*) = DATATYPE_UINT;
                    break;
                case SAMPLEFORMAT_INT:
                    *va_arg(ap, uint16*) = DATATYPE_INT;
                    break;
                case SAMPLEFORMAT_IEEEFP:
                    *va_arg(ap, uint16*) = DATATYPE_IEEEFP;
                    break;
                case SAMPLEFORMAT_VOID:
                    *va_arg(ap, uint16*) = DATATYPE_VOID;
                    break;
            }
            break;
        case TIFFTAG_SAMPLEFORMAT:
            *va_arg(ap, uint16*) = td->td_sampleformat;
            break;
        case TIFFTAG_IMAGEDEPTH:
            *va_arg(ap, uint32*) = td->td_imagedepth;
            break;
        case TIFFTAG_SUBIFD:
            *va_arg(ap, uint16*) = td->td_nsubifd;
            *va_arg(ap, uint64**) = td->td_subifd;
            break;
        case TIFFTAG_YCBCRPOSITIONING:
            *va_arg(ap, uint16*) = td->td_ycbcrpositioning;
            break;
        case TIFFTAG_YCBCRSUBSAMPLING:
            *va_arg(ap, uint16*) = td->td_ycbcrsubsampling[0];
            *va_arg(ap, uint16*) = td->td_ycbcrsubsampling[1];
            break;
        case TIFFTAG_TRANSFERFUNCTION:
            *va_arg(ap, uint16**) = td->td_transferfunction[0];
            if (td->td_samplesperpixel - td->td_extrasamples > 1) {
                *va_arg(ap, uint16**) = td->td_transferfunction[1];
                *va_arg(ap, uint16**) = td->td_transferfunction[2];
            }
            break;
        case TIFFTAG_REFERENCEBLACKWHITE:
            *va_arg(ap, float**) = td->td_refblackwhite;
            break;
        case TIFFTAG_INKNAMES:
            *va_arg(ap, char**) = td->td_inknames;
            break;
        default:
            {
                int i;

                /*
                 * This can happen if multiple images are open
                 * with different codecs which have private
                 * tags.  The global tag information table may
                 * then have tags that are valid for one file
                 * but not the other. If the client tries to
                 * get a tag that is not valid for the image's
                 * codec then we'll arrive here.
                 */
                if( fip == NULL || fip->field_bit != FIELD_CUSTOM )
                {
                    TIFFErrorExt(tif->tif_clientdata, "_TIFFVGetField",
                        "%s: Invalid %stag \"%s\" "
                        "(not supported by codec)",
                        tif->tif_name,
                        isPseudoTag(tag) ? "pseudo-" : "",
                        fip ? fip->field_name : "Unknown");
                    ret_val = 0;
                    break;
                }

                /*
                 * Do we have a custom value?
                 */
                ret_val = 0;
                for (i = 0; i < td->td_customValueCount; i++) {
                    TIFFTagValue *tv = td->td_customValues + i;

                    if (tv->info->field_tag != tag)
                        continue;

                    if (fip->field_passcount) {
                        if (fip->field_readcount == TIFF_VARIABLE2)
                            *va_arg(ap, uint32*) = (uint32)tv->count;
                        else  /* Assume TIFF_VARIABLE */
                            *va_arg(ap, uint16*) = (uint16)tv->count;
                        *va_arg(ap, void **) = tv->value;
                        ret_val = 1;
                    } else if (fip->field_tag == TIFFTAG_DOTRANGE
                           && strcmp(fip->field_name,"DotRange") == 0) {
                        /* TODO: This is an evil exception and should not have been
                           handled this way ... likely best if we move it into
                           the directory structure with an explicit field in
                           libtiff 4.1 and assign it a FIELD_ value */
                        *va_arg(ap, uint16*) = ((uint16 *)tv->value)[0];
                        *va_arg(ap, uint16*) = ((uint16 *)tv->value)[1];
                        ret_val = 1;
                    } else {
                        if (fip->field_type == TIFF_ASCII
                            || fip->field_readcount == TIFF_VARIABLE
                            || fip->field_readcount == TIFF_VARIABLE2
                            || fip->field_readcount == TIFF_SPP
                            || tv->count > 1) {
                            *va_arg(ap, void **) = tv->value;
                            ret_val = 1;
                        } else {
                            char *val = (char *)tv->value;
                            assert( tv->count == 1 );
                            switch (fip->field_type) {
                            case TIFF_BYTE:
                            case TIFF_UNDEFINED:
                                *va_arg(ap, uint8*) =
                                    *(uint8 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SBYTE:
                                *va_arg(ap, int8*) =
                                    *(int8 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SHORT:
                                *va_arg(ap, uint16*) =
                                    *(uint16 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SSHORT:
                                *va_arg(ap, int16*) =
                                    *(int16 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_LONG:
                            case TIFF_IFD:
                                *va_arg(ap, uint32*) =
                                    *(uint32 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SLONG:
                                *va_arg(ap, int32*) =
                                    *(int32 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_LONG8:
                            case TIFF_IFD8:
                                *va_arg(ap, uint64*) =
                                    *(uint64 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_SLONG8:
                                *va_arg(ap, int64*) =
                                    *(int64 *)val;
                                ret_val = 1;
                                break;
                            case TIFF_RATIONAL:
                            case TIFF_SRATIONAL:
                            case TIFF_FLOAT:
                                *va_arg(ap, float*) =
                                    *(float *)val;
                                ret_val = 1;
                                break;
                            case TIFF_DOUBLE:
                                *va_arg(ap, double*) =
                                    *(double *)val;
                                ret_val = 1;
                                break;
                            default:
                                ret_val = 0;
                                break;
                            }
                        }
                    }
                    break;
                }
            }
    }
    return(ret_val);
}

/*
 * Return the value of a field in the
 * internal directory structure.
 */
int
TIFFGetField(TIFF* tif, uint32 tag, ...)
{
    int status;
    va_list ap;

    va_start(ap, tag);
    status = TIFFVGetField(tif, tag, ap);
    va_end(ap);
    return (status);
}

/*
 * Like TIFFGetField, but taking a varargs
 * parameter list.  This routine is useful
 * for building higher-level interfaces on
 * top of the library.
 */
int
TIFFVGetField(TIFF* tif, uint32 tag, va_list ap)
{
    const TIFFField* fip = TIFFFindField(tif, tag, TIFF_ANY);
    return (fip && (isPseudoTag(tag) || TIFFFieldSet(tif, fip->field_bit)) ?
        (*tif->tif_tagmethods.vgetfield)(tif, tag, ap) : 0);
}

#define	CleanupField(member) {		\
    if (td->member) {			\
    _TIFFfree(td->member);		\
    td->member = 0;			\
    }					\
}

/*
 * Release storage associated with a directory.
 */
void
TIFFFreeDirectory(TIFF* tif)
{
    TIFFDirectory *td = &tif->tif_dir;
    int            i;

    _TIFFmemset(td->td_fieldsset, 0, FIELD_SETLONGS);
    CleanupField(td_sminsamplevalue);
    CleanupField(td_smaxsamplevalue);
    CleanupField(td_colormap[0]);
    CleanupField(td_colormap[1]);
    CleanupField(td_colormap[2]);
    CleanupField(td_sampleinfo);
    CleanupField(td_subifd);
    CleanupField(td_inknames);
    CleanupField(td_refblackwhite);
    CleanupField(td_transferfunction[0]);
    CleanupField(td_transferfunction[1]);
    CleanupField(td_transferfunction[2]);
    CleanupField(td_stripoffset);
    CleanupField(td_stripbytecount);
    TIFFClrFieldBit(tif, FIELD_YCBCRSUBSAMPLING);
    TIFFClrFieldBit(tif, FIELD_YCBCRPOSITIONING);

    /* Cleanup custom tag values */
    for( i = 0; i < td->td_customValueCount; i++ ) {
        if (td->td_customValues[i].value)
            _TIFFfree(td->td_customValues[i].value);
    }

    td->td_customValueCount = 0;
    CleanupField(td_customValues);

#if defined(DEFER_STRILE_LOAD)
        _TIFFmemset( &(td->td_stripoffset_entry), 0, sizeof(TIFFDirEntry));
        _TIFFmemset( &(td->td_stripbytecount_entry), 0, sizeof(TIFFDirEntry));
#endif
}
#undef CleanupField

/*
 * Client Tag extension support (from Niles Ritter).
 */
static TIFFExtendProc _TIFFextender = (TIFFExtendProc) NULL;

TIFFExtendProc
TIFFSetTagExtender(TIFFExtendProc extender)
{
    TIFFExtendProc prev = _TIFFextender;
    _TIFFextender = extender;
    return (prev);
}

/*
 * Setup for a new directory.  Should we automatically call
 * TIFFWriteDirectory() if the current one is dirty?
 *
 * The newly created directory will not exist on the file till
 * TIFFWriteDirectory(), TIFFFlush() or TIFFClose() is called.
 */
int
TIFFCreateDirectory(TIFF* tif)
{
    TIFFDefaultDirectory(tif);
    tif->tif_diroff = 0;
    tif->tif_nextdiroff = 0;
    tif->tif_curoff = 0;
    tif->tif_row = (uint32) -1;
    tif->tif_curstrip = (uint32) -1;

    return 0;
}

int
TIFFCreateCustomDirectory(TIFF* tif, const TIFFFieldArray* infoarray)
{
    TIFFDefaultDirectory(tif);

    /*
     * Reset the field definitions to match the application provided list.
     * Hopefully TIFFDefaultDirectory() won't have done anything irreversable
     * based on it's assumption this is an image directory.
     */
    _TIFFSetupFields(tif, infoarray);

    tif->tif_diroff = 0;
    tif->tif_nextdiroff = 0;
    tif->tif_curoff = 0;
    tif->tif_row = (uint32) -1;
    tif->tif_curstrip = (uint32) -1;

    return 0;
}

int
TIFFCreateEXIFDirectory(TIFF* tif)
{
    const TIFFFieldArray* exifFieldArray;
    exifFieldArray = _TIFFGetExifFields();
    return TIFFCreateCustomDirectory(tif, exifFieldArray);
}

/*
 * Setup a default directory structure.
 */
int
TIFFDefaultDirectory(TIFF* tif)
{
    register TIFFDirectory* td = &tif->tif_dir;
    const TIFFFieldArray* tiffFieldArray;

    tiffFieldArray = _TIFFGetFields();
    _TIFFSetupFields(tif, tiffFieldArray);

    _TIFFmemset(td, 0, sizeof (*td));
    td->td_fillorder = FILLORDER_MSB2LSB;
    td->td_bitspersample = 1;
    td->td_threshholding = THRESHHOLD_BILEVEL;
    td->td_orientation = ORIENTATION_TOPLEFT;
    td->td_samplesperpixel = 1;
    td->td_rowsperstrip = (uint32) -1;
    td->td_tilewidth = 0;
    td->td_tilelength = 0;
    td->td_tiledepth = 1;
    td->td_stripbytecountsorted = 1; /* Our own arrays always sorted. */
    td->td_resolutionunit = RESUNIT_INCH;
    td->td_sampleformat = SAMPLEFORMAT_UINT;
    td->td_imagedepth = 1;
    td->td_ycbcrsubsampling[0] = 2;
    td->td_ycbcrsubsampling[1] = 2;
    td->td_ycbcrpositioning = YCBCRPOSITION_CENTERED;
    tif->tif_postdecode = _TIFFNoPostDecode;
    tif->tif_foundfield = NULL;
    tif->tif_tagmethods.vsetfield = _TIFFVSetField;
    tif->tif_tagmethods.vgetfield = _TIFFVGetField;
    tif->tif_tagmethods.printdir = NULL;
    /*
     *  Give client code a chance to install their own
     *  tag extensions & methods, prior to compression overloads.
     */
    if (_TIFFextender)
        (*_TIFFextender)(tif);
    (void) TIFFSetField(tif, TIFFTAG_COMPRESSION, COMPRESSION_NONE);
    /*
     * NB: The directory is marked dirty as a result of setting
     * up the default compression scheme.  However, this really
     * isn't correct -- we want TIFF_DIRTYDIRECT to be set only
     * if the user does something.  We could just do the setup
     * by hand, but it seems better to use the normal mechanism
     * (i.e. TIFFSetField).
     */
    tif->tif_flags &= ~TIFF_DIRTYDIRECT;

    /*
     * As per http://bugzilla.remotesensing.org/show_bug.cgi?id=19
     * we clear the ISTILED flag when setting up a new directory.
     * Should we also be clearing stuff like INSUBIFD?
     */
    tif->tif_flags &= ~TIFF_ISTILED;

    return (1);
}

static int
TIFFAdvanceDirectory(TIFF* tif, uint64* nextdir, uint64* off)
{
    static const char module[] = "TIFFAdvanceDirectory";
    if (isMapped(tif))
    {
        uint64 poff=*nextdir;
        if (!(tif->tif_flags&TIFF_BIGTIFF))
        {
            tmsize_t poffa,poffb,poffc,poffd;
            uint16 dircount;
            uint32 nextdir32;
            poffa=(tmsize_t)poff;
            poffb=poffa+sizeof(uint16);
            if (((uint64)poffa!=poff)||(poffb<poffa)||(poffb<(tmsize_t)sizeof(uint16))||(poffb>tif->tif_size))
            {
                TIFFErrorExt(tif->tif_clientdata,module,"Error fetching directory count");
                return(0);
            }
            _TIFFmemcpy(&dircount,tif->tif_base+poffa,sizeof(uint16));
            if (tif->tif_flags&TIFF_SWAB)
                TIFFSwabShort(&dircount);
            poffc=poffb+dircount*12;
            poffd=poffc+sizeof(uint32);
            if ((poffc<poffb)||(poffc<dircount*12)||(poffd<poffc)||(poffd<(tmsize_t)sizeof(uint32))||(poffd>tif->tif_size))
            {
                TIFFErrorExt(tif->tif_clientdata,module,"Error fetching directory link");
                return(0);
            }
            if (off!=NULL)
                *off=(uint64)poffc;
            _TIFFmemcpy(&nextdir32,tif->tif_base+poffc,sizeof(uint32));
            if (tif->tif_flags&TIFF_SWAB)
                TIFFSwabLong(&nextdir32);
            *nextdir=nextdir32;
        }
        else
        {
            tmsize_t poffa,poffb,poffc,poffd;
            uint64 dircount64;
            uint16 dircount16;
            poffa=(tmsize_t)poff;
            poffb=poffa+sizeof(uint64);
            if (((uint64)poffa!=poff)||(poffb<poffa)||(poffb<(tmsize_t)sizeof(uint64))||(poffb>tif->tif_size))
            {
                TIFFErrorExt(tif->tif_clientdata,module,"Error fetching directory count");
                return(0);
            }
            _TIFFmemcpy(&dircount64,tif->tif_base+poffa,sizeof(uint64));
            if (tif->tif_flags&TIFF_SWAB)
                TIFFSwabLong8(&dircount64);
            if (dircount64>0xFFFF)
            {
                TIFFErrorExt(tif->tif_clientdata,module,"Sanity check on directory count failed");
                return(0);
            }
            dircount16=(uint16)dircount64;
            poffc=poffb+dircount16*20;
            poffd=poffc+sizeof(uint64);
            if ((poffc<poffb)||(poffc<dircount16*20)||(poffd<poffc)||(poffd<(tmsize_t)sizeof(uint64))||(poffd>tif->tif_size))
            {
                TIFFErrorExt(tif->tif_clientdata,module,"Error fetching directory link");
                return(0);
            }
            if (off!=NULL)
                *off=(uint64)poffc;
            _TIFFmemcpy(nextdir,tif->tif_base+poffc,sizeof(uint64));
            if (tif->tif_flags&TIFF_SWAB)
                TIFFSwabLong8(nextdir);
        }
        return(1);
    }
    else
    {
        if (!(tif->tif_flags&TIFF_BIGTIFF))
        {
            uint16 dircount;
            uint32 nextdir32;
            if (!SeekOK(tif, *nextdir) ||
                !ReadOK(tif, &dircount, sizeof (uint16))) {
                TIFFErrorExt(tif->tif_clientdata, module, "%s: Error fetching directory count",
                    tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabShort(&dircount);
            if (off != NULL)
                *off = TIFFSeekFile(tif,
                    dircount*12, SEEK_CUR);
            else
                (void) TIFFSeekFile(tif,
                    dircount*12, SEEK_CUR);
            if (!ReadOK(tif, &nextdir32, sizeof (uint32))) {
                TIFFErrorExt(tif->tif_clientdata, module, "%s: Error fetching directory link",
                    tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong(&nextdir32);
            *nextdir=nextdir32;
        }
        else
        {
            uint64 dircount64;
            uint16 dircount16;
            if (!SeekOK(tif, *nextdir) ||
                !ReadOK(tif, &dircount64, sizeof (uint64))) {
                TIFFErrorExt(tif->tif_clientdata, module, "%s: Error fetching directory count",
                    tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(&dircount64);
            if (dircount64>0xFFFF)
            {
                TIFFErrorExt(tif->tif_clientdata, module, "Error fetching directory count");
                return(0);
            }
            dircount16 = (uint16)dircount64;
            if (off != NULL)
                *off = TIFFSeekFile(tif,
                    dircount16*20, SEEK_CUR);
            else
                (void) TIFFSeekFile(tif,
                    dircount16*20, SEEK_CUR);
            if (!ReadOK(tif, nextdir, sizeof (uint64))) {
                TIFFErrorExt(tif->tif_clientdata, module, "%s: Error fetching directory link",
                    tif->tif_name);
                return (0);
            }
            if (tif->tif_flags & TIFF_SWAB)
                TIFFSwabLong8(nextdir);
        }
        return (1);
    }
}

/*
 * Count the number of directories in a file.
 */
uint16
TIFFNumberOfDirectories(TIFF* tif)
{
    uint64 nextdir;
    uint16 n;
    if (!(tif->tif_flags&TIFF_BIGTIFF))
        nextdir = tif->tif_header.classic.tiff_diroff;
    else
        nextdir = tif->tif_header.big.tiff_diroff;
    n = 0;
    while (nextdir != 0 && TIFFAdvanceDirectory(tif, &nextdir, NULL))
        n++;
    return (n);
}

/*
 * Set the n-th directory as the current directory.
 * NB: Directories are numbered starting at 0.
 */
int
TIFFSetDirectory(TIFF* tif, uint16 dirn)
{
    uint64 nextdir;
    uint16 n;

    if (!(tif->tif_flags&TIFF_BIGTIFF))
        nextdir = tif->tif_header.classic.tiff_diroff;
    else
        nextdir = tif->tif_header.big.tiff_diroff;
    for (n = dirn; n > 0 && nextdir != 0; n--)
        if (!TIFFAdvanceDirectory(tif, &nextdir, NULL))
            return (0);
    tif->tif_nextdiroff = nextdir;
    /*
     * Set curdir to the actual directory index.  The
     * -1 is because TIFFReadDirectory will increment
     * tif_curdir after successfully reading the directory.
     */
    tif->tif_curdir = (dirn - n) - 1;
    /*
     * Reset tif_dirnumber counter and start new list of seen directories.
     * We need this to prevent IFD loops.
     */
    tif->tif_dirnumber = 0;
    return (TIFFReadDirectory(tif));
}

/*
 * Set the current directory to be the directory
 * located at the specified file offset.  This interface
 * is used mainly to access directories linked with
 * the SubIFD tag (e.g. thumbnail images).
 */
int
TIFFSetSubDirectory(TIFF* tif, uint64 diroff)
{
    tif->tif_nextdiroff = diroff;
    /*
     * Reset tif_dirnumber counter and start new list of seen directories.
     * We need this to prevent IFD loops.
     */
    tif->tif_dirnumber = 0;
    return (TIFFReadDirectory(tif));
}

/*
 * Return file offset of the current directory.
 */
uint64
TIFFCurrentDirOffset(TIFF* tif)
{
    return (tif->tif_diroff);
}

/*
 * Return an indication of whether or not we are
 * at the last directory in the file.
 */
int
TIFFLastDirectory(TIFF* tif)
{
    return (tif->tif_nextdiroff == 0);
}

/*
 * Unlink the specified directory from the directory chain.
 */
int
TIFFUnlinkDirectory(TIFF* tif, uint16 dirn)
{
    static const char module[] = "TIFFUnlinkDirectory";
    uint64 nextdir;
    uint64 off;
    uint16 n;

    if (tif->tif_mode == O_RDONLY) {
        TIFFErrorExt(tif->tif_clientdata, module,
                             "Can not unlink directory in read-only file");
        return (0);
    }
    /*
     * Go to the directory before the one we want
     * to unlink and nab the offset of the link
     * field we'll need to patch.
     */
    if (!(tif->tif_flags&TIFF_BIGTIFF))
    {
        nextdir = tif->tif_header.classic.tiff_diroff;
        off = 4;
    }
    else
    {
        nextdir = tif->tif_header.big.tiff_diroff;
        off = 8;
    }
    for (n = dirn-1; n > 0; n--) {
        if (nextdir == 0) {
            TIFFErrorExt(tif->tif_clientdata, module, "Directory %d does not exist", dirn);
            return (0);
        }
        if (!TIFFAdvanceDirectory(tif, &nextdir, &off))
            return (0);
    }
    /*
     * Advance to the directory to be unlinked and fetch
     * the offset of the directory that follows.
     */
    if (!TIFFAdvanceDirectory(tif, &nextdir, NULL))
        return (0);
    /*
     * Go back and patch the link field of the preceding
     * directory to point to the offset of the directory
     * that follows.
     */
    (void) TIFFSeekFile(tif, off, SEEK_SET);
    if (!(tif->tif_flags&TIFF_BIGTIFF))
    {
        uint32 nextdir32;
        nextdir32=(uint32)nextdir;
        assert((uint64)nextdir32==nextdir);
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(&nextdir32);
        if (!WriteOK(tif, &nextdir32, sizeof (uint32))) {
            TIFFErrorExt(tif->tif_clientdata, module, "Error writing directory link");
            return (0);
        }
    }
    else
    {
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong8(&nextdir);
        if (!WriteOK(tif, &nextdir, sizeof (uint64))) {
            TIFFErrorExt(tif->tif_clientdata, module, "Error writing directory link");
            return (0);
        }
    }
    /*
     * Leave directory state setup safely.  We don't have
     * facilities for doing inserting and removing directories,
     * so it's safest to just invalidate everything.  This
     * means that the caller can only append to the directory
     * chain.
     */
    (*tif->tif_cleanup)(tif);
    if ((tif->tif_flags & TIFF_MYBUFFER) && tif->tif_rawdata) {
        _TIFFfree(tif->tif_rawdata);
        tif->tif_rawdata = NULL;
        tif->tif_rawcc = 0;
                tif->tif_rawdataoff = 0;
                tif->tif_rawdataloaded = 0;
    }
    tif->tif_flags &= ~(TIFF_BEENWRITING|TIFF_BUFFERSETUP|TIFF_POSTENCODE|TIFF_BUF4WRITE);
    TIFFFreeDirectory(tif);
    TIFFDefaultDirectory(tif);
    tif->tif_diroff = 0;			/* force link on next write */
    tif->tif_nextdiroff = 0;		/* next write must be at end */
    tif->tif_curoff = 0;
    tif->tif_row = (uint32) -1;
    tif->tif_curstrip = (uint32) -1;
    return (1);
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
