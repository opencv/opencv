/* $Id: tif_dir.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

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
		tsize_t	bytes = nmemb * elem_size;
		if (elem_size && bytes / elem_size == nmemb)
			*vpp = (void*) _TIFFmalloc(bytes);
		if (*vpp)
			_TIFFmemcpy(*vpp, vp, bytes);
	}
}
void _TIFFsetByteArray(void** vpp, void* vp, long n)
    { setByteArray(vpp, vp, n, 1); }
void _TIFFsetString(char** cpp, char* cp)
    { setByteArray((void**) cpp, (void*) cp, strlen(cp)+1, 1); }
void _TIFFsetNString(char** cpp, char* cp, long n)
    { setByteArray((void**) cpp, (void*) cp, n, 1); }
void _TIFFsetShortArray(uint16** wpp, uint16* wp, long n)
    { setByteArray((void**) wpp, (void*) wp, n, sizeof (uint16)); }
void _TIFFsetLongArray(uint32** lpp, uint32* lp, long n)
    { setByteArray((void**) lpp, (void*) lp, n, sizeof (uint32)); }
void _TIFFsetFloatArray(float** fpp, float* fp, long n)
    { setByteArray((void**) fpp, (void*) fp, n, sizeof (float)); }
void _TIFFsetDoubleArray(double** dpp, double* dp, long n)
    { setByteArray((void**) dpp, (void*) dp, n, sizeof (double)); }

/*
 * Install extra samples information.
 */
static int
setExtraSamples(TIFFDirectory* td, va_list ap, int* v)
{
	uint16* va;
	int i;

	*v = va_arg(ap, int);
	if ((uint16) *v > td->td_samplesperpixel)
		return (0);
	va = va_arg(ap, uint16*);
	if (*v > 0 && va == NULL)		/* typically missing param */
		return (0);
	for (i = 0; i < *v; i++)
		if (va[i] > EXTRASAMPLE_UNASSALPHA)
			return (0);
	td->td_extrasamples = (uint16) *v;
	_TIFFsetShortArray(&td->td_sampleinfo, va, td->td_extrasamples);
	return (1);
}

static int
checkInkNamesString(TIFF* tif, int slen, const char* s)
{
	TIFFDirectory* td = &tif->tif_dir;
	int i = td->td_samplesperpixel;

	if (slen > 0) {
		const char* ep = s+slen;
		const char* cp = s;
		for (; i > 0; i--) {
			for (; *cp != '\0'; cp++)
				if (cp >= ep)
					goto bad;
			cp++;				/* skip \0 */
		}
		return (cp-s);
	}
bad:
	TIFFError("TIFFSetField",
	    "%s: Invalid InkNames value; expecting %d names, found %d",
	    tif->tif_name,
	    td->td_samplesperpixel,
	    td->td_samplesperpixel-i);
	return (0);
}

static int
_TIFFVSetField(TIFF* tif, ttag_t tag, va_list ap)
{
	static const char module[] = "_TIFFVSetField";
	
	TIFFDirectory* td = &tif->tif_dir;
	int status = 1;
	uint32 v32;
	int i, v;
	double d;
	char* s;

	switch (tag) {
	case TIFFTAG_SUBFILETYPE:
		td->td_subfiletype = va_arg(ap, uint32);
		break;
	case TIFFTAG_IMAGEWIDTH:
		td->td_imagewidth = va_arg(ap, uint32);
		break;
	case TIFFTAG_IMAGELENGTH:
		td->td_imagelength = va_arg(ap, uint32);
		break;
	case TIFFTAG_BITSPERSAMPLE:
		td->td_bitspersample = (uint16) va_arg(ap, int);
		/*
		 * If the data require post-decoding processing
		 * to byte-swap samples, set it up here.  Note
		 * that since tags are required to be ordered,
		 * compression code can override this behaviour
		 * in the setup method if it wants to roll the
		 * post decoding work in with its normal work.
		 */
		if (tif->tif_flags & TIFF_SWAB) {
			if (td->td_bitspersample == 16)
				tif->tif_postdecode = _TIFFSwab16BitData;
			else if (td->td_bitspersample == 32)
				tif->tif_postdecode = _TIFFSwab32BitData;
			else if (td->td_bitspersample == 64)
				tif->tif_postdecode = _TIFFSwab64BitData;
		}
		break;
	case TIFFTAG_COMPRESSION:
		v = va_arg(ap, int) & 0xffff;
		/*
		 * If we're changing the compression scheme,
		 * the notify the previous module so that it
		 * can cleanup any state it's setup.
		 */
		if (TIFFFieldSet(tif, FIELD_COMPRESSION)) {
			if (td->td_compression == v)
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
		td->td_photometric = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_THRESHHOLDING:
		td->td_threshholding = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_FILLORDER:
		v = va_arg(ap, int);
		if (v != FILLORDER_LSB2MSB && v != FILLORDER_MSB2LSB)
			goto badvalue;
		td->td_fillorder = (uint16) v;
		break;
	case TIFFTAG_DOCUMENTNAME:
		_TIFFsetString(&td->td_documentname, va_arg(ap, char*));
		break;
	case TIFFTAG_ARTIST:
		_TIFFsetString(&td->td_artist, va_arg(ap, char*));
		break;
	case TIFFTAG_DATETIME:
		_TIFFsetString(&td->td_datetime, va_arg(ap, char*));
		break;
	case TIFFTAG_HOSTCOMPUTER:
		_TIFFsetString(&td->td_hostcomputer, va_arg(ap, char*));
		break;
	case TIFFTAG_IMAGEDESCRIPTION:
		_TIFFsetString(&td->td_imagedescription, va_arg(ap, char*));
		break;
	case TIFFTAG_MAKE:
		_TIFFsetString(&td->td_make, va_arg(ap, char*));
		break;
	case TIFFTAG_MODEL:
		_TIFFsetString(&td->td_model, va_arg(ap, char*));
		break;
	case TIFFTAG_COPYRIGHT:
		_TIFFsetString(&td->td_copyright, va_arg(ap, char*));
		break;
	case TIFFTAG_ORIENTATION:
		v = va_arg(ap, int);
		if (v < ORIENTATION_TOPLEFT || ORIENTATION_LEFTBOT < v) {
			TIFFWarning(tif->tif_name,
			    "Bad value %ld for \"%s\" tag ignored",
			    v, _TIFFFieldWithTag(tif, tag)->field_name);
		} else
			td->td_orientation = (uint16) v;
		break;
	case TIFFTAG_SAMPLESPERPIXEL:
		/* XXX should cross check -- e.g. if pallette, then 1 */
		v = va_arg(ap, int);
		if (v == 0)
			goto badvalue;
		td->td_samplesperpixel = (uint16) v;
		break;
	case TIFFTAG_ROWSPERSTRIP:
		v32 = va_arg(ap, uint32);
		if (v32 == 0)
			goto badvalue32;
		td->td_rowsperstrip = v32;
		if (!TIFFFieldSet(tif, FIELD_TILEDIMENSIONS)) {
			td->td_tilelength = v32;
			td->td_tilewidth = td->td_imagewidth;
		}
		break;
	case TIFFTAG_MINSAMPLEVALUE:
		td->td_minsamplevalue = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_MAXSAMPLEVALUE:
		td->td_maxsamplevalue = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_SMINSAMPLEVALUE:
		td->td_sminsamplevalue = (double) va_arg(ap, dblparam_t);
		break;
	case TIFFTAG_SMAXSAMPLEVALUE:
		td->td_smaxsamplevalue = (double) va_arg(ap, dblparam_t);
		break;
	case TIFFTAG_XRESOLUTION:
		td->td_xresolution = (float) va_arg(ap, dblparam_t);
		break;
	case TIFFTAG_YRESOLUTION:
		td->td_yresolution = (float) va_arg(ap, dblparam_t);
		break;
	case TIFFTAG_PLANARCONFIG:
		v = va_arg(ap, int);
		if (v != PLANARCONFIG_CONTIG && v != PLANARCONFIG_SEPARATE)
			goto badvalue;
		td->td_planarconfig = (uint16) v;
		break;
	case TIFFTAG_PAGENAME:
		_TIFFsetString(&td->td_pagename, va_arg(ap, char*));
		break;
	case TIFFTAG_XPOSITION:
		td->td_xposition = (float) va_arg(ap, dblparam_t);
		break;
	case TIFFTAG_YPOSITION:
		td->td_yposition = (float) va_arg(ap, dblparam_t);
		break;
	case TIFFTAG_RESOLUTIONUNIT:
		v = va_arg(ap, int);
		if (v < RESUNIT_NONE || RESUNIT_CENTIMETER < v)
			goto badvalue;
		td->td_resolutionunit = (uint16) v;
		break;
	case TIFFTAG_PAGENUMBER:
		td->td_pagenumber[0] = (uint16) va_arg(ap, int);
		td->td_pagenumber[1] = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_HALFTONEHINTS:
		td->td_halftonehints[0] = (uint16) va_arg(ap, int);
		td->td_halftonehints[1] = (uint16) va_arg(ap, int);
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
		td->td_extrasamples = (uint16) (va_arg(ap, int) != 0);
		if (td->td_extrasamples) {
			uint16 sv = EXTRASAMPLE_ASSOCALPHA;
			_TIFFsetShortArray(&td->td_sampleinfo, &sv, 1);
		}
		break;
	case TIFFTAG_TILEWIDTH:
		v32 = va_arg(ap, uint32);
		if (v32 % 16) {
			if (tif->tif_mode != O_RDONLY)
				goto badvalue32;
			TIFFWarning(tif->tif_name,
			    "Nonstandard tile width %d, convert file", v32);
		}
		td->td_tilewidth = v32;
		tif->tif_flags |= TIFF_ISTILED;
		break;
	case TIFFTAG_TILELENGTH:
		v32 = va_arg(ap, uint32);
		if (v32 % 16) {
			if (tif->tif_mode != O_RDONLY)
				goto badvalue32;
			TIFFWarning(tif->tif_name,
			    "Nonstandard tile length %d, convert file", v32);
		}
		td->td_tilelength = v32;
		tif->tif_flags |= TIFF_ISTILED;
		break;
	case TIFFTAG_TILEDEPTH:
		v32 = va_arg(ap, uint32);
		if (v32 == 0)
			goto badvalue32;
		td->td_tiledepth = v32;
		break;
	case TIFFTAG_DATATYPE:
		v = va_arg(ap, int);
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
		v = va_arg(ap, int);
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
                else if( td->td_sampleformat == SAMPLEFORMAT_COMPLEXIEEEFP
                         && td->td_bitspersample == 128
                         && tif->tif_postdecode == NULL )
                    tif->tif_postdecode = _TIFFSwab64BitData;
		break;
	case TIFFTAG_IMAGEDEPTH:
		td->td_imagedepth = va_arg(ap, uint32);
		break;
	case TIFFTAG_STONITS:
		d = va_arg(ap, dblparam_t);
		if (d <= 0.)
			goto badvaluedbl;
		td->td_stonits = d;
		break;
	/* Begin Pixar Tags */
 	case TIFFTAG_PIXAR_IMAGEFULLWIDTH:
 		td->td_imagefullwidth = va_arg(ap, uint32);
 		break;
 	case TIFFTAG_PIXAR_IMAGEFULLLENGTH:
 		td->td_imagefulllength = va_arg(ap, uint32);
 		break;
 	case TIFFTAG_PIXAR_TEXTUREFORMAT:
 		_TIFFsetString(&td->td_textureformat, va_arg(ap, char*));
 		break;
 	case TIFFTAG_PIXAR_WRAPMODES:
 		_TIFFsetString(&td->td_wrapmodes, va_arg(ap, char*));
 		break;
 	case TIFFTAG_PIXAR_FOVCOT:
 		td->td_fovcot = (float) va_arg(ap, dblparam_t);
 		break;
 	case TIFFTAG_PIXAR_MATRIX_WORLDTOSCREEN:
 		_TIFFsetFloatArray(&td->td_matrixWorldToScreen,
 			va_arg(ap, float*), 16);
 		break;
 	case TIFFTAG_PIXAR_MATRIX_WORLDTOCAMERA:
 		_TIFFsetFloatArray(&td->td_matrixWorldToCamera,
 			va_arg(ap, float*), 16);
 		break;
 	/* End Pixar Tags */	       

	case TIFFTAG_SUBIFD:
		if ((tif->tif_flags & TIFF_INSUBIFD) == 0) {
			td->td_nsubifd = (uint16) va_arg(ap, int);
			_TIFFsetLongArray(&td->td_subifd, va_arg(ap, uint32*),
			    (long) td->td_nsubifd);
		} else {
			TIFFError(module, "%s: Sorry, cannot nest SubIFDs",
				  tif->tif_name);
			status = 0;
		}
		break;
	case TIFFTAG_YCBCRCOEFFICIENTS:
		_TIFFsetFloatArray(&td->td_ycbcrcoeffs, va_arg(ap, float*), 3);
		break;
	case TIFFTAG_YCBCRPOSITIONING:
		td->td_ycbcrpositioning = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_YCBCRSUBSAMPLING:
		td->td_ycbcrsubsampling[0] = (uint16) va_arg(ap, int);
		td->td_ycbcrsubsampling[1] = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_WHITEPOINT:
		_TIFFsetFloatArray(&td->td_whitepoint, va_arg(ap, float*), 2);
		break;
	case TIFFTAG_PRIMARYCHROMATICITIES:
		_TIFFsetFloatArray(&td->td_primarychromas, va_arg(ap, float*), 6);
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
	case TIFFTAG_INKSET:
		td->td_inkset = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_DOTRANGE:
		/* XXX should check for null range */
		td->td_dotrange[0] = (uint16) va_arg(ap, int);
		td->td_dotrange[1] = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_INKNAMES:
		i = va_arg(ap, int);
		s = va_arg(ap, char*);
		i = checkInkNamesString(tif, i, s);
                status = i > 0;
		if( i > 0 ) {
			_TIFFsetNString(&td->td_inknames, s, i);
			td->td_inknameslen = i;
		}
		break;
	case TIFFTAG_NUMBEROFINKS:
		td->td_ninks = (uint16) va_arg(ap, int);
		break;
	case TIFFTAG_TARGETPRINTER:
		_TIFFsetString(&td->td_targetprinter, va_arg(ap, char*));
		break;
	case TIFFTAG_ICCPROFILE:
		td->td_profileLength = (uint32) va_arg(ap, uint32);
		_TIFFsetByteArray(&td->td_profileData, va_arg(ap, void*),
		    td->td_profileLength);
		break;
 	case TIFFTAG_PHOTOSHOP:
  		td->td_photoshopLength = (uint32) va_arg(ap, uint32);
  		_TIFFsetByteArray (&td->td_photoshopData, va_arg(ap, void*),
 			td->td_photoshopLength);
 		break;
	case TIFFTAG_RICHTIFFIPTC: 
  		td->td_richtiffiptcLength = (uint32) va_arg(ap, uint32);
  		_TIFFsetLongArray ((uint32**)&td->td_richtiffiptcData,
				   va_arg(ap, uint32*),
				   td->td_richtiffiptcLength);
 		break;
	case TIFFTAG_XMLPACKET:
		td->td_xmlpacketLength = (uint32) va_arg(ap, uint32);
		_TIFFsetByteArray(&td->td_xmlpacketData, va_arg(ap, void*),
		    td->td_xmlpacketLength);
		break;
        default: {
            const TIFFFieldInfo* fip = _TIFFFindFieldInfo(tif, tag, TIFF_ANY);
            TIFFTagValue *tv;
            int tv_size, iCustom;

            /*
             * This can happen if multiple images are open with
             * different codecs which have private tags.  The
             * global tag information table may then have tags
             * that are valid for one file but not the other. 
             * If the client tries to set a tag that is not valid
             * for the image's codec then we'll arrive here.  This
             * happens, for example, when tiffcp is used to convert
             * between compression schemes and codec-specific tags
             * are blindly copied.
             */
            if( fip == NULL || fip->field_bit != FIELD_CUSTOM )
            {
		TIFFError(module,
		    "%s: Invalid %stag \"%s\" (not supported by codec)",
		    tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "",
		    _TIFFFieldWithTag(tif, tag)->field_name);
		status = 0;
		break;
            }

            /*
             * Find the existing entry for this custom value.
             */
            tv = NULL;
            for( iCustom = 0; iCustom < td->td_customValueCount; iCustom++ )
            {
                if( td->td_customValues[iCustom].info == fip )
                {
                    tv = td->td_customValues + iCustom;
                    if( tv->value != NULL )
                        _TIFFfree( tv->value );
                    break;
                }
            }

            /*
             * Grow the custom list if the entry was not found.
             */
            if( tv == NULL )
            {
		TIFFTagValue	*new_customValues;
		
		td->td_customValueCount++;
		new_customValues = (TIFFTagValue *)
			_TIFFrealloc(td->td_customValues,
				     sizeof(TIFFTagValue) * td->td_customValueCount);
		if (!new_customValues) {
			TIFFError(module,
		"%s: Failed to allocate space for list of custom values",
				  tif->tif_name);
			status = 0;
			goto end;
		}

		td->td_customValues = new_customValues;

                tv = td->td_customValues + (td->td_customValueCount-1);
                tv->info = fip;
                tv->value = NULL;
                tv->count = 0;
            }

            /*
             * Set custom value ... save a copy of the custom tag value.
             */
	    switch (fip->field_type) {
		    /*
		     * XXX: We can't use TIFFDataWidth() to determine the
		     * space needed to store the value. For TIFF_RATIONAL
		     * values TIFFDataWidth() returns 8, but we use 4-byte
		     * float to represent rationals.
		     */
		    case TIFF_BYTE:
		    case TIFF_ASCII:
		    case TIFF_SBYTE:
		    case TIFF_UNDEFINED:
			tv_size = 1;
			break;

		    case TIFF_SHORT:
		    case TIFF_SSHORT:
			tv_size = 2;
			break;

		    case TIFF_LONG:
		    case TIFF_SLONG:
		    case TIFF_FLOAT:
		    case TIFF_IFD:
		    case TIFF_RATIONAL:
		    case TIFF_SRATIONAL:
			tv_size = 4;
			break;

		    case TIFF_DOUBLE:
			tv_size = 8;
			break;

		    default:
			status = 0;
			TIFFError(module, "%s: Bad field type %d for \"%s\"",
				  tif->tif_name, fip->field_type,
				  fip->field_name);
			goto end;
		    
	    }
            
            if(fip->field_passcount)
                tv->count = (int) va_arg(ap, int);
            else
                tv->count = 1;
            
	    if (fip->field_passcount) {
                tv->value = _TIFFmalloc(tv_size * tv->count);
		if ( !tv->value ) {
			status = 0;
			goto end;
		}
                _TIFFmemcpy(tv->value, (void *) va_arg(ap,void*),
                            tv->count * tv_size);
            } else if (fip->field_type == TIFF_ASCII) {
                const char *value = (const char *) va_arg(ap,const char *);
                tv->count = strlen(value)+1;
                tv->value = _TIFFmalloc(tv->count);
		if (!tv->value) {
			status = 0;
			goto end;
		}
                strcpy(tv->value, value);
            } else {
                /* not supporting "pass by value" types yet */
		TIFFError(module,
			  "%s: Pass by value is not implemented.",
			  tif->tif_name);

                tv->value = _TIFFmalloc(tv_size * tv->count);
		if (!tv->value) {
			status = 0;
			goto end;
		}
                _TIFFmemset(tv->value, 0, tv->count * tv_size);
                status = 0;
            }
          }
	}
	if (status) {
            TIFFSetFieldBit(tif, _TIFFFieldWithTag(tif, tag)->field_bit);
            tif->tif_flags |= TIFF_DIRTYDIRECT;
	}

end:
	va_end(ap);
	return (status);
badvalue:
	TIFFError(module, "%s: Bad value %d for \"%s\"",
		  tif->tif_name, v, _TIFFFieldWithTag(tif, tag)->field_name);
	va_end(ap);
	return (0);
badvalue32:
	TIFFError(module, "%s: Bad value %ld for \"%s\"",
		   tif->tif_name, v32, _TIFFFieldWithTag(tif, tag)->field_name);
	va_end(ap);
	return (0);
badvaluedbl:
	TIFFError(module, "%s: Bad value %f for \"%s\"",
		  tif->tif_name, d, _TIFFFieldWithTag(tif, tag)->field_name);
	va_end(ap);
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
OkToChangeTag(TIFF* tif, ttag_t tag)
{
	const TIFFFieldInfo* fip = _TIFFFindFieldInfo(tif, tag, TIFF_ANY);
	if (!fip) {			/* unknown tag */
		TIFFError("TIFFSetField", "%s: Unknown %stag %u",
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
		TIFFError("TIFFSetField",
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
TIFFSetField(TIFF* tif, ttag_t tag, ...)
{
	va_list ap;
	int status;

	va_start(ap, tag);
	status = TIFFVSetField(tif, tag, ap);
	va_end(ap);
	return (status);
}

/*
 * Like TIFFSetField, but taking a varargs
 * parameter list.  This routine is useful
 * for building higher-level interfaces on
 * top of the library.
 */
int
TIFFVSetField(TIFF* tif, ttag_t tag, va_list ap)
{
	return OkToChangeTag(tif, tag) ?
	    (*tif->tif_tagmethods.vsetfield)(tif, tag, ap) : 0;
}

static int
_TIFFVGetField(TIFF* tif, ttag_t tag, va_list ap)
{
    TIFFDirectory* td = &tif->tif_dir;
    int            ret_val = 1;

    switch (tag) {
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
	case TIFFTAG_DOCUMENTNAME:
            *va_arg(ap, char**) = td->td_documentname;
            break;
	case TIFFTAG_ARTIST:
            *va_arg(ap, char**) = td->td_artist;
            break;
	case TIFFTAG_DATETIME:
            *va_arg(ap, char**) = td->td_datetime;
            break;
	case TIFFTAG_HOSTCOMPUTER:
            *va_arg(ap, char**) = td->td_hostcomputer;
            break;
	case TIFFTAG_IMAGEDESCRIPTION:
            *va_arg(ap, char**) = td->td_imagedescription;
            break;
	case TIFFTAG_MAKE:
            *va_arg(ap, char**) = td->td_make;
            break;
	case TIFFTAG_MODEL:
            *va_arg(ap, char**) = td->td_model;
            break;
	case TIFFTAG_COPYRIGHT:
            *va_arg(ap, char**) = td->td_copyright;
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
            *va_arg(ap, double*) = td->td_sminsamplevalue;
            break;
	case TIFFTAG_SMAXSAMPLEVALUE:
            *va_arg(ap, double*) = td->td_smaxsamplevalue;
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
	case TIFFTAG_PAGENAME:
            *va_arg(ap, char**) = td->td_pagename;
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
            *va_arg(ap, uint32**) = td->td_stripoffset;
            break;
	case TIFFTAG_STRIPBYTECOUNTS:
	case TIFFTAG_TILEBYTECOUNTS:
            *va_arg(ap, uint32**) = td->td_stripbytecount;
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
	case TIFFTAG_STONITS:
            *va_arg(ap, double*) = td->td_stonits;
            break;
	case TIFFTAG_SUBIFD:
            *va_arg(ap, uint16*) = td->td_nsubifd;
            *va_arg(ap, uint32**) = td->td_subifd;
            break;
	case TIFFTAG_YCBCRCOEFFICIENTS:
            *va_arg(ap, float**) = td->td_ycbcrcoeffs;
            break;
	case TIFFTAG_YCBCRPOSITIONING:
            *va_arg(ap, uint16*) = td->td_ycbcrpositioning;
            break;
	case TIFFTAG_YCBCRSUBSAMPLING:
            *va_arg(ap, uint16*) = td->td_ycbcrsubsampling[0];
            *va_arg(ap, uint16*) = td->td_ycbcrsubsampling[1];
            break;
	case TIFFTAG_WHITEPOINT:
            *va_arg(ap, float**) = td->td_whitepoint;
            break;
	case TIFFTAG_PRIMARYCHROMATICITIES:
            *va_arg(ap, float**) = td->td_primarychromas;
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
	case TIFFTAG_INKSET:
            *va_arg(ap, uint16*) = td->td_inkset;
            break;
	case TIFFTAG_DOTRANGE:
            *va_arg(ap, uint16*) = td->td_dotrange[0];
            *va_arg(ap, uint16*) = td->td_dotrange[1];
            break;
	case TIFFTAG_INKNAMES:
            *va_arg(ap, char**) = td->td_inknames;
            break;
	case TIFFTAG_NUMBEROFINKS:
            *va_arg(ap, uint16*) = td->td_ninks;
            break;
	case TIFFTAG_TARGETPRINTER:
            *va_arg(ap, char**) = td->td_targetprinter;
            break;
	case TIFFTAG_ICCPROFILE:
            *va_arg(ap, uint32*) = td->td_profileLength;
            *va_arg(ap, void**) = td->td_profileData;
            break;
 	case TIFFTAG_PHOTOSHOP:
            *va_arg(ap, uint32*) = td->td_photoshopLength;
            *va_arg(ap, void**) = td->td_photoshopData;
            break;
 	case TIFFTAG_RICHTIFFIPTC:
            *va_arg(ap, uint32*) = td->td_richtiffiptcLength;
            *va_arg(ap, void**) = td->td_richtiffiptcData;
            break;
	case TIFFTAG_XMLPACKET:
            *va_arg(ap, uint32*) = td->td_xmlpacketLength;
            *va_arg(ap, void**) = td->td_xmlpacketData;
            break;
            /* Begin Pixar Tags */
 	case TIFFTAG_PIXAR_IMAGEFULLWIDTH:
            *va_arg(ap, uint32*) = td->td_imagefullwidth;
            break;
 	case TIFFTAG_PIXAR_IMAGEFULLLENGTH:
            *va_arg(ap, uint32*) = td->td_imagefulllength;
            break;
 	case TIFFTAG_PIXAR_TEXTUREFORMAT:
            *va_arg(ap, char**) = td->td_textureformat;
            break;
 	case TIFFTAG_PIXAR_WRAPMODES:
            *va_arg(ap, char**) = td->td_wrapmodes;
            break;
 	case TIFFTAG_PIXAR_FOVCOT:
            *va_arg(ap, float*) = td->td_fovcot;
            break;
 	case TIFFTAG_PIXAR_MATRIX_WORLDTOSCREEN:
            *va_arg(ap, float**) = td->td_matrixWorldToScreen;
            break;
 	case TIFFTAG_PIXAR_MATRIX_WORLDTOCAMERA:
            *va_arg(ap, float**) = td->td_matrixWorldToCamera;
            break;
            /* End Pixar Tags */

        default:
        {
            const TIFFFieldInfo* fip = _TIFFFindFieldInfo(tif, tag, TIFF_ANY);
            int           i;
            
            /*
             * This can happen if multiple images are open with
             * different codecs which have private tags.  The
             * global tag information table may then have tags
             * that are valid for one file but not the other. 
             * If the client tries to get a tag that is not valid
             * for the image's codec then we'll arrive here.
             */
            if( fip == NULL || fip->field_bit != FIELD_CUSTOM )
            {
                TIFFError("_TIFFVGetField",
                          "%s: Invalid %stag \"%s\" (not supported by codec)",
                          tif->tif_name, isPseudoTag(tag) ? "pseudo-" : "",
                          _TIFFFieldWithTag(tif, tag)->field_name);
                ret_val = 0;
                break;
            }

            /*
            ** Do we have a custom value?
            */
            ret_val = 0;
            for (i = 0; i < td->td_customValueCount; i++)
            {
		TIFFTagValue *tv = td->td_customValues + i;

		if (tv->info->field_tag != tag)
			continue;
                
		if (fip->field_passcount) {
			if (fip->field_readcount == TIFF_VARIABLE2) 
				*va_arg(ap, uint32*) = (uint32)tv->count;
			else	/* Assume TIFF_VARIABLE */
				*va_arg(ap, uint16*) = (uint16)tv->count;
			*va_arg(ap, void **) = tv->value;
			ret_val = 1;
			break;
                } else if (fip->field_type == TIFF_ASCII) {
			*va_arg(ap, void **) = tv->value;
			ret_val = 1;
			break;
                } else {
			TIFFError("_TIFFVGetField",
				  "%s: Pass by value is not implemented.",
				  tif->tif_name);
			break;
                }
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
TIFFGetField(TIFF* tif, ttag_t tag, ...)
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
TIFFVGetField(TIFF* tif, ttag_t tag, va_list ap)
{
	const TIFFFieldInfo* fip = _TIFFFindFieldInfo(tif, tag, TIFF_ANY);
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

	CleanupField(td_colormap[0]);
	CleanupField(td_colormap[1]);
	CleanupField(td_colormap[2]);
	CleanupField(td_documentname);
	CleanupField(td_artist);
	CleanupField(td_datetime);
	CleanupField(td_hostcomputer);
	CleanupField(td_imagedescription);
	CleanupField(td_make);
	CleanupField(td_model);
	CleanupField(td_copyright);
	CleanupField(td_pagename);
	CleanupField(td_sampleinfo);
	CleanupField(td_subifd);
	CleanupField(td_ycbcrcoeffs);
	CleanupField(td_inknames);
	CleanupField(td_targetprinter);
	CleanupField(td_whitepoint);
	CleanupField(td_primarychromas);
	CleanupField(td_refblackwhite);
	CleanupField(td_transferfunction[0]);
	CleanupField(td_transferfunction[1]);
	CleanupField(td_transferfunction[2]);
	CleanupField(td_profileData);
	CleanupField(td_photoshopData);
	CleanupField(td_richtiffiptcData);
	CleanupField(td_xmlpacketData);
	CleanupField(td_stripoffset);
	CleanupField(td_stripbytecount);
	/* Begin Pixar Tags */
	CleanupField(td_textureformat);
	CleanupField(td_wrapmodes);
	CleanupField(td_matrixWorldToScreen);
	CleanupField(td_matrixWorldToCamera);
	/* End Pixar Tags */

	/* Cleanup custom tag values */
	for( i = 0; i < td->td_customValueCount; i++ ) {
		if (td->td_customValues[i].value)
			_TIFFfree(td->td_customValues[i].value);
	}

	td->td_customValueCount = 0;
	CleanupField(td_customValues);
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
    tif->tif_curstrip = (tstrip_t) -1;

    return 0;
}

/*
 * Setup a default directory structure.
 */
int
TIFFDefaultDirectory(TIFF* tif)
{
	register TIFFDirectory* td = &tif->tif_dir;

	_TIFFSetupFieldInfo(tif);
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
	td->td_inkset = INKSET_CMYK;
	td->td_ninks = 4;
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
TIFFAdvanceDirectory(TIFF* tif, uint32* nextdir, toff_t* off)
{
    static const char module[] = "TIFFAdvanceDirectory";
    uint16 dircount;
    if (isMapped(tif))
    {
        toff_t poff=*nextdir;
        if (poff+sizeof(uint16) > tif->tif_size)
        {
            TIFFError(module, "%s: Error fetching directory count",
                      tif->tif_name);
            return (0);
        }
        _TIFFmemcpy(&dircount, tif->tif_base+poff, sizeof (uint16));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabShort(&dircount);
        poff+=sizeof (uint16)+dircount*sizeof (TIFFDirEntry);
        if (off != NULL)
            *off = poff;
        if (((toff_t) (poff+sizeof (uint32))) > tif->tif_size)
        {
            TIFFError(module, "%s: Error fetching directory link",
                      tif->tif_name);
            return (0);
        }
        _TIFFmemcpy(nextdir, tif->tif_base+poff, sizeof (uint32));
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(nextdir);
        return (1);
    }
    else
    {
        if (!SeekOK(tif, *nextdir) ||
            !ReadOK(tif, &dircount, sizeof (uint16))) {
            TIFFError(module, "%s: Error fetching directory count",
                      tif->tif_name);
            return (0);
        }
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabShort(&dircount);
        if (off != NULL)
            *off = TIFFSeekFile(tif,
                                dircount*sizeof (TIFFDirEntry), SEEK_CUR);
        else
            (void) TIFFSeekFile(tif,
                                dircount*sizeof (TIFFDirEntry), SEEK_CUR);
        if (!ReadOK(tif, nextdir, sizeof (uint32))) {
            TIFFError(module, "%s: Error fetching directory link",
                      tif->tif_name);
            return (0);
        }
        if (tif->tif_flags & TIFF_SWAB)
            TIFFSwabLong(nextdir);
        return (1);
    }
}

/*
 * Count the number of directories in a file.
 */
tdir_t
TIFFNumberOfDirectories(TIFF* tif)
{
    toff_t nextdir = tif->tif_header.tiff_diroff;
    tdir_t n = 0;
    
    while (nextdir != 0 && TIFFAdvanceDirectory(tif, &nextdir, NULL))
        n++;
    return (n);
}

/*
 * Set the n-th directory as the current directory.
 * NB: Directories are numbered starting at 0.
 */
int
TIFFSetDirectory(TIFF* tif, tdir_t dirn)
{
	toff_t nextdir;
	tdir_t n;

	nextdir = tif->tif_header.tiff_diroff;
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
TIFFSetSubDirectory(TIFF* tif, uint32 diroff)
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
uint32
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
TIFFUnlinkDirectory(TIFF* tif, tdir_t dirn)
{
	static const char module[] = "TIFFUnlinkDirectory";
	toff_t nextdir;
	toff_t off;
	tdir_t n;

	if (tif->tif_mode == O_RDONLY) {
		TIFFError(module, "Can not unlink directory in read-only file");
		return (0);
	}
	/*
	 * Go to the directory before the one we want
	 * to unlink and nab the offset of the link
	 * field we'll need to patch.
	 */
	nextdir = tif->tif_header.tiff_diroff;
	off = sizeof (uint16) + sizeof (uint16);
	for (n = dirn-1; n > 0; n--) {
		if (nextdir == 0) {
			TIFFError(module, "Directory %d does not exist", dirn);
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
	if (tif->tif_flags & TIFF_SWAB)
		TIFFSwabLong(&nextdir);
	if (!WriteOK(tif, &nextdir, sizeof (uint32))) {
		TIFFError(module, "Error writing directory link");
		return (0);
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
	}
	tif->tif_flags &= ~(TIFF_BEENWRITING|TIFF_BUFFERSETUP|TIFF_POSTENCODE);
	TIFFFreeDirectory(tif);
	TIFFDefaultDirectory(tif);
	tif->tif_diroff = 0;			/* force link on next write */
	tif->tif_nextdiroff = 0;		/* next write must be at end */
	tif->tif_curoff = 0;
	tif->tif_row = (uint32) -1;
	tif->tif_curstrip = (tstrip_t) -1;
	return (1);
}

/*			[BFC]
 *
 * Author: Bruce Cameron <cameron@petris.com>
 *
 * Set a table of tags that are to be replaced during directory process by the
 * 'IGNORE' state - or return TRUE/FALSE for the requested tag such that
 * 'ReadDirectory' can use the stored information.
 *
 * FIXME: this is never used properly. Should be removed in the future.
 */
int
TIFFReassignTagToIgnore (enum TIFFIgnoreSense task, int TIFFtagID)
{
    static int TIFFignoretags [FIELD_LAST];
    static int tagcount = 0 ;
    int		i;					/* Loop index */
    int		j;					/* Loop index */

    switch (task)
    {
      case TIS_STORE:
        if ( tagcount < (FIELD_LAST - 1) )
        {
            for ( j = 0 ; j < tagcount ; ++j )
            {					/* Do not add duplicate tag */
                if ( TIFFignoretags [j] == TIFFtagID )
                    return (TRUE) ;
            }
            TIFFignoretags [tagcount++] = TIFFtagID ;
            return (TRUE) ;
        }
        break ;
        
      case TIS_EXTRACT:
        for ( i = 0 ; i < tagcount ; ++i )
        {
            if ( TIFFignoretags [i] == TIFFtagID )
                return (TRUE) ;
        }
        break;
        
      case TIS_EMPTY:
        tagcount = 0 ;			/* Clear the list */
        return (TRUE) ;
        
      default:
        break;
    }
    
    return (FALSE);
}

/* vim: set ts=8 sts=8 sw=8 noet: */
