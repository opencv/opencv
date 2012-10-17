/* $Id: tif_print.c,v 1.59 2012-06-13 01:08:51 fwarmerdam Exp $ */

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
 * Directory Printing Support
 */
#include "tiffiop.h"
#include <stdio.h>

#include <ctype.h>

static void
_TIFFprintAsciiBounded(FILE* fd, const char* cp, int max_chars);

static const char *photoNames[] = {
    "min-is-white",				/* PHOTOMETRIC_MINISWHITE */
    "min-is-black",				/* PHOTOMETRIC_MINISBLACK */
    "RGB color",				/* PHOTOMETRIC_RGB */
    "palette color (RGB from colormap)",	/* PHOTOMETRIC_PALETTE */
    "transparency mask",			/* PHOTOMETRIC_MASK */
    "separated",				/* PHOTOMETRIC_SEPARATED */
    "YCbCr",					/* PHOTOMETRIC_YCBCR */
    "7 (0x7)",
    "CIE L*a*b*",				/* PHOTOMETRIC_CIELAB */
};
#define	NPHOTONAMES	(sizeof (photoNames) / sizeof (photoNames[0]))

static const char *orientNames[] = {
    "0 (0x0)",
    "row 0 top, col 0 lhs",			/* ORIENTATION_TOPLEFT */
    "row 0 top, col 0 rhs",			/* ORIENTATION_TOPRIGHT */
    "row 0 bottom, col 0 rhs",			/* ORIENTATION_BOTRIGHT */
    "row 0 bottom, col 0 lhs",			/* ORIENTATION_BOTLEFT */
    "row 0 lhs, col 0 top",			/* ORIENTATION_LEFTTOP */
    "row 0 rhs, col 0 top",			/* ORIENTATION_RIGHTTOP */
    "row 0 rhs, col 0 bottom",			/* ORIENTATION_RIGHTBOT */
    "row 0 lhs, col 0 bottom",			/* ORIENTATION_LEFTBOT */
};
#define	NORIENTNAMES	(sizeof (orientNames) / sizeof (orientNames[0]))

static void
_TIFFPrintField(FILE* fd, const TIFFField *fip,
        uint32 value_count, void *raw_data)
{
    uint32 j;

    fprintf(fd, "  %s: ", fip->field_name);

    for(j = 0; j < value_count; j++) {
        if(fip->field_type == TIFF_BYTE)
            fprintf(fd, "%u", ((uint8 *) raw_data)[j]);
        else if(fip->field_type == TIFF_UNDEFINED)
            fprintf(fd, "0x%x",
                (unsigned int) ((unsigned char *) raw_data)[j]);
        else if(fip->field_type == TIFF_SBYTE)
            fprintf(fd, "%d", ((int8 *) raw_data)[j]);
        else if(fip->field_type == TIFF_SHORT)
            fprintf(fd, "%u", ((uint16 *) raw_data)[j]);
        else if(fip->field_type == TIFF_SSHORT)
            fprintf(fd, "%d", ((int16 *) raw_data)[j]);
        else if(fip->field_type == TIFF_LONG)
            fprintf(fd, "%lu",
                (unsigned long)((uint32 *) raw_data)[j]);
        else if(fip->field_type == TIFF_SLONG)
            fprintf(fd, "%ld", (long)((int32 *) raw_data)[j]);
        else if(fip->field_type == TIFF_IFD)
            fprintf(fd, "0x%lx",
                (unsigned long)((uint32 *) raw_data)[j]);
        else if(fip->field_type == TIFF_RATIONAL
            || fip->field_type == TIFF_SRATIONAL
            || fip->field_type == TIFF_FLOAT)
            fprintf(fd, "%f", ((float *) raw_data)[j]);
        else if(fip->field_type == TIFF_LONG8)
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
            fprintf(fd, "%I64u",
                (unsigned __int64)((uint64 *) raw_data)[j]);
#else
            fprintf(fd, "%llu",
                (unsigned long long)((uint64 *) raw_data)[j]);
#endif
        else if(fip->field_type == TIFF_SLONG8)
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
            fprintf(fd, "%I64d", (__int64)((int64 *) raw_data)[j]);
#else
            fprintf(fd, "%lld", (long long)((int64 *) raw_data)[j]);
#endif
        else if(fip->field_type == TIFF_IFD8)
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
            fprintf(fd, "0x%I64x",
                (unsigned __int64)((uint64 *) raw_data)[j]);
#else
            fprintf(fd, "0x%llx",
                (unsigned long long)((uint64 *) raw_data)[j]);
#endif
        else if(fip->field_type == TIFF_FLOAT)
            fprintf(fd, "%f", ((float *)raw_data)[j]);
        else if(fip->field_type == TIFF_DOUBLE)
            fprintf(fd, "%f", ((double *) raw_data)[j]);
        else if(fip->field_type == TIFF_ASCII) {
            fprintf(fd, "%s", (char *) raw_data);
            break;
        }
        else {
            fprintf(fd, "<unsupported data type in TIFFPrint>");
            break;
        }

        if(j < value_count - 1)
            fprintf(fd, ",");
    }

    fprintf(fd, "\n");
}

static int
_TIFFPrettyPrintField(TIFF* tif, const TIFFField *fip, FILE* fd, uint32 tag,
              uint32 value_count, void *raw_data)
{
        (void) tif;

    /* do not try to pretty print auto-defined fields */
    if (strncmp(fip->field_name,"Tag ", 4) == 0) {
        return 0;
    }

    switch (tag)
    {
        case TIFFTAG_INKSET:
            if (value_count == 2 && fip->field_type == TIFF_SHORT) {
                fprintf(fd, "  Ink Set: ");
                switch (*((uint16*)raw_data)) {
                case INKSET_CMYK:
                    fprintf(fd, "CMYK\n");
                    break;
                default:
                    fprintf(fd, "%u (0x%x)\n",
                        *((uint16*)raw_data),
                        *((uint16*)raw_data));
                    break;
                }
                return 1;
            }
            return 0;

        case TIFFTAG_DOTRANGE:
            if (value_count == 2 && fip->field_type == TIFF_SHORT) {
                fprintf(fd, "  Dot Range: %u-%u\n",
                    ((uint16*)raw_data)[0], ((uint16*)raw_data)[1]);
                return 1;
            }
            return 0;

        case TIFFTAG_WHITEPOINT:
            if (value_count == 2 && fip->field_type == TIFF_RATIONAL) {
                fprintf(fd, "  White Point: %g-%g\n",
                    ((float *)raw_data)[0], ((float *)raw_data)[1]);
                return 1;
            }
            return 0;

        case TIFFTAG_XMLPACKET:
        {
            uint32 i;

            fprintf(fd, "  XMLPacket (XMP Metadata):\n" );
            for(i = 0; i < value_count; i++)
                fputc(((char *)raw_data)[i], fd);
            fprintf( fd, "\n" );
            return 1;
        }
        case TIFFTAG_RICHTIFFIPTC:
            /*
             * XXX: for some weird reason RichTIFFIPTC tag
             * defined as array of LONG values.
             */
            fprintf(fd,
                "  RichTIFFIPTC Data: <present>, %lu bytes\n",
                (unsigned long) value_count * 4);
            return 1;

        case TIFFTAG_PHOTOSHOP:
            fprintf(fd, "  Photoshop Data: <present>, %lu bytes\n",
                (unsigned long) value_count);
            return 1;

        case TIFFTAG_ICCPROFILE:
            fprintf(fd, "  ICC Profile: <present>, %lu bytes\n",
                (unsigned long) value_count);
            return 1;

        case TIFFTAG_STONITS:
            if (value_count == 1 && fip->field_type == TIFF_DOUBLE) {
                fprintf(fd,
                    "  Sample to Nits conversion factor: %.4e\n",
                    *((double*)raw_data));
                return 1;
            }
            return 0;
    }

    return 0;
}

/*
 * Print the contents of the current directory
 * to the specified stdio file stream.
 */
void
TIFFPrintDirectory(TIFF* tif, FILE* fd, long flags)
{
    TIFFDirectory *td = &tif->tif_dir;
    char *sep;
    uint16 i;
    long l, n;

#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
    fprintf(fd, "TIFF Directory at offset 0x%I64x (%I64u)\n",
        (unsigned __int64) tif->tif_diroff,
        (unsigned __int64) tif->tif_diroff);
#else
    fprintf(fd, "TIFF Directory at offset 0x%llx (%llu)\n",
        (unsigned long long) tif->tif_diroff,
        (unsigned long long) tif->tif_diroff);
#endif
    if (TIFFFieldSet(tif,FIELD_SUBFILETYPE)) {
        fprintf(fd, "  Subfile Type:");
        sep = " ";
        if (td->td_subfiletype & FILETYPE_REDUCEDIMAGE) {
            fprintf(fd, "%sreduced-resolution image", sep);
            sep = "/";
        }
        if (td->td_subfiletype & FILETYPE_PAGE) {
            fprintf(fd, "%smulti-page document", sep);
            sep = "/";
        }
        if (td->td_subfiletype & FILETYPE_MASK)
            fprintf(fd, "%stransparency mask", sep);
        fprintf(fd, " (%lu = 0x%lx)\n",
            (long) td->td_subfiletype, (long) td->td_subfiletype);
    }
    if (TIFFFieldSet(tif,FIELD_IMAGEDIMENSIONS)) {
        fprintf(fd, "  Image Width: %lu Image Length: %lu",
            (unsigned long) td->td_imagewidth, (unsigned long) td->td_imagelength);
        if (TIFFFieldSet(tif,FIELD_IMAGEDEPTH))
            fprintf(fd, " Image Depth: %lu",
                (unsigned long) td->td_imagedepth);
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif,FIELD_TILEDIMENSIONS)) {
        fprintf(fd, "  Tile Width: %lu Tile Length: %lu",
            (unsigned long) td->td_tilewidth, (unsigned long) td->td_tilelength);
        if (TIFFFieldSet(tif,FIELD_TILEDEPTH))
            fprintf(fd, " Tile Depth: %lu",
                (unsigned long) td->td_tiledepth);
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif,FIELD_RESOLUTION)) {
        fprintf(fd, "  Resolution: %g, %g",
            td->td_xresolution, td->td_yresolution);
        if (TIFFFieldSet(tif,FIELD_RESOLUTIONUNIT)) {
            switch (td->td_resolutionunit) {
            case RESUNIT_NONE:
                fprintf(fd, " (unitless)");
                break;
            case RESUNIT_INCH:
                fprintf(fd, " pixels/inch");
                break;
            case RESUNIT_CENTIMETER:
                fprintf(fd, " pixels/cm");
                break;
            default:
                fprintf(fd, " (unit %u = 0x%x)",
                    td->td_resolutionunit,
                    td->td_resolutionunit);
                break;
            }
        }
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif,FIELD_POSITION))
        fprintf(fd, "  Position: %g, %g\n",
            td->td_xposition, td->td_yposition);
    if (TIFFFieldSet(tif,FIELD_BITSPERSAMPLE))
        fprintf(fd, "  Bits/Sample: %u\n", td->td_bitspersample);
    if (TIFFFieldSet(tif,FIELD_SAMPLEFORMAT)) {
        fprintf(fd, "  Sample Format: ");
        switch (td->td_sampleformat) {
        case SAMPLEFORMAT_VOID:
            fprintf(fd, "void\n");
            break;
        case SAMPLEFORMAT_INT:
            fprintf(fd, "signed integer\n");
            break;
        case SAMPLEFORMAT_UINT:
            fprintf(fd, "unsigned integer\n");
            break;
        case SAMPLEFORMAT_IEEEFP:
            fprintf(fd, "IEEE floating point\n");
            break;
        case SAMPLEFORMAT_COMPLEXINT:
            fprintf(fd, "complex signed integer\n");
            break;
        case SAMPLEFORMAT_COMPLEXIEEEFP:
            fprintf(fd, "complex IEEE floating point\n");
            break;
        default:
            fprintf(fd, "%u (0x%x)\n",
                td->td_sampleformat, td->td_sampleformat);
            break;
        }
    }
    if (TIFFFieldSet(tif,FIELD_COMPRESSION)) {
        const TIFFCodec* c = TIFFFindCODEC(td->td_compression);
        fprintf(fd, "  Compression Scheme: ");
        if (c)
            fprintf(fd, "%s\n", c->name);
        else
            fprintf(fd, "%u (0x%x)\n",
                td->td_compression, td->td_compression);
    }
    if (TIFFFieldSet(tif,FIELD_PHOTOMETRIC)) {
        fprintf(fd, "  Photometric Interpretation: ");
        if (td->td_photometric < NPHOTONAMES)
            fprintf(fd, "%s\n", photoNames[td->td_photometric]);
        else {
            switch (td->td_photometric) {
            case PHOTOMETRIC_LOGL:
                fprintf(fd, "CIE Log2(L)\n");
                break;
            case PHOTOMETRIC_LOGLUV:
                fprintf(fd, "CIE Log2(L) (u',v')\n");
                break;
            default:
                fprintf(fd, "%u (0x%x)\n",
                    td->td_photometric, td->td_photometric);
                break;
            }
        }
    }
    if (TIFFFieldSet(tif,FIELD_EXTRASAMPLES) && td->td_extrasamples) {
        fprintf(fd, "  Extra Samples: %u<", td->td_extrasamples);
        sep = "";
        for (i = 0; i < td->td_extrasamples; i++) {
            switch (td->td_sampleinfo[i]) {
            case EXTRASAMPLE_UNSPECIFIED:
                fprintf(fd, "%sunspecified", sep);
                break;
            case EXTRASAMPLE_ASSOCALPHA:
                fprintf(fd, "%sassoc-alpha", sep);
                break;
            case EXTRASAMPLE_UNASSALPHA:
                fprintf(fd, "%sunassoc-alpha", sep);
                break;
            default:
                fprintf(fd, "%s%u (0x%x)", sep,
                    td->td_sampleinfo[i], td->td_sampleinfo[i]);
                break;
            }
            sep = ", ";
        }
        fprintf(fd, ">\n");
    }
    if (TIFFFieldSet(tif,FIELD_INKNAMES)) {
        char* cp;
        fprintf(fd, "  Ink Names: ");
        i = td->td_samplesperpixel;
        sep = "";
        for (cp = td->td_inknames;
             i > 0 && cp < td->td_inknames + td->td_inknameslen;
             cp = strchr(cp,'\0')+1, i--) {
            int max_chars =
                td->td_inknameslen - (cp - td->td_inknames);
            fputs(sep, fd);
            _TIFFprintAsciiBounded(fd, cp, max_chars);
            sep = ", ";
        }
                fputs("\n", fd);
    }
    if (TIFFFieldSet(tif,FIELD_THRESHHOLDING)) {
        fprintf(fd, "  Thresholding: ");
        switch (td->td_threshholding) {
        case THRESHHOLD_BILEVEL:
            fprintf(fd, "bilevel art scan\n");
            break;
        case THRESHHOLD_HALFTONE:
            fprintf(fd, "halftone or dithered scan\n");
            break;
        case THRESHHOLD_ERRORDIFFUSE:
            fprintf(fd, "error diffused\n");
            break;
        default:
            fprintf(fd, "%u (0x%x)\n",
                td->td_threshholding, td->td_threshholding);
            break;
        }
    }
    if (TIFFFieldSet(tif,FIELD_FILLORDER)) {
        fprintf(fd, "  FillOrder: ");
        switch (td->td_fillorder) {
        case FILLORDER_MSB2LSB:
            fprintf(fd, "msb-to-lsb\n");
            break;
        case FILLORDER_LSB2MSB:
            fprintf(fd, "lsb-to-msb\n");
            break;
        default:
            fprintf(fd, "%u (0x%x)\n",
                td->td_fillorder, td->td_fillorder);
            break;
        }
    }
    if (TIFFFieldSet(tif,FIELD_YCBCRSUBSAMPLING))
        {
        fprintf(fd, "  YCbCr Subsampling: %u, %u\n",
            td->td_ycbcrsubsampling[0], td->td_ycbcrsubsampling[1] );
    }
    if (TIFFFieldSet(tif,FIELD_YCBCRPOSITIONING)) {
        fprintf(fd, "  YCbCr Positioning: ");
        switch (td->td_ycbcrpositioning) {
        case YCBCRPOSITION_CENTERED:
            fprintf(fd, "centered\n");
            break;
        case YCBCRPOSITION_COSITED:
            fprintf(fd, "cosited\n");
            break;
        default:
            fprintf(fd, "%u (0x%x)\n",
                td->td_ycbcrpositioning, td->td_ycbcrpositioning);
            break;
        }
    }
    if (TIFFFieldSet(tif,FIELD_HALFTONEHINTS))
        fprintf(fd, "  Halftone Hints: light %u dark %u\n",
            td->td_halftonehints[0], td->td_halftonehints[1]);
    if (TIFFFieldSet(tif,FIELD_ORIENTATION)) {
        fprintf(fd, "  Orientation: ");
        if (td->td_orientation < NORIENTNAMES)
            fprintf(fd, "%s\n", orientNames[td->td_orientation]);
        else
            fprintf(fd, "%u (0x%x)\n",
                td->td_orientation, td->td_orientation);
    }
    if (TIFFFieldSet(tif,FIELD_SAMPLESPERPIXEL))
        fprintf(fd, "  Samples/Pixel: %u\n", td->td_samplesperpixel);
    if (TIFFFieldSet(tif,FIELD_ROWSPERSTRIP)) {
        fprintf(fd, "  Rows/Strip: ");
        if (td->td_rowsperstrip == (uint32) -1)
            fprintf(fd, "(infinite)\n");
        else
            fprintf(fd, "%lu\n", (unsigned long) td->td_rowsperstrip);
    }
    if (TIFFFieldSet(tif,FIELD_MINSAMPLEVALUE))
        fprintf(fd, "  Min Sample Value: %u\n", td->td_minsamplevalue);
    if (TIFFFieldSet(tif,FIELD_MAXSAMPLEVALUE))
        fprintf(fd, "  Max Sample Value: %u\n", td->td_maxsamplevalue);
    if (TIFFFieldSet(tif,FIELD_SMINSAMPLEVALUE)) {
        int count = (tif->tif_flags & TIFF_PERSAMPLE) ? td->td_samplesperpixel : 1;
        fprintf(fd, "  SMin Sample Value:");
        for (i = 0; i < count; ++i)
            fprintf(fd, " %g", td->td_sminsamplevalue[i]);
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif,FIELD_SMAXSAMPLEVALUE)) {
        int count = (tif->tif_flags & TIFF_PERSAMPLE) ? td->td_samplesperpixel : 1;
        fprintf(fd, "  SMax Sample Value:");
        for (i = 0; i < count; ++i)
            fprintf(fd, " %g", td->td_smaxsamplevalue[i]);
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif,FIELD_PLANARCONFIG)) {
        fprintf(fd, "  Planar Configuration: ");
        switch (td->td_planarconfig) {
        case PLANARCONFIG_CONTIG:
            fprintf(fd, "single image plane\n");
            break;
        case PLANARCONFIG_SEPARATE:
            fprintf(fd, "separate image planes\n");
            break;
        default:
            fprintf(fd, "%u (0x%x)\n",
                td->td_planarconfig, td->td_planarconfig);
            break;
        }
    }
    if (TIFFFieldSet(tif,FIELD_PAGENUMBER))
        fprintf(fd, "  Page Number: %u-%u\n",
            td->td_pagenumber[0], td->td_pagenumber[1]);
    if (TIFFFieldSet(tif,FIELD_COLORMAP)) {
        fprintf(fd, "  Color Map: ");
        if (flags & TIFFPRINT_COLORMAP) {
            fprintf(fd, "\n");
            n = 1L<<td->td_bitspersample;
            for (l = 0; l < n; l++)
                fprintf(fd, "   %5lu: %5u %5u %5u\n",
                    l,
                    td->td_colormap[0][l],
                    td->td_colormap[1][l],
                    td->td_colormap[2][l]);
        } else
            fprintf(fd, "(present)\n");
    }
    if (TIFFFieldSet(tif,FIELD_REFBLACKWHITE)) {
        fprintf(fd, "  Reference Black/White:\n");
        for (i = 0; i < 3; i++)
        fprintf(fd, "    %2d: %5g %5g\n", i,
            td->td_refblackwhite[2*i+0],
            td->td_refblackwhite[2*i+1]);
    }
    if (TIFFFieldSet(tif,FIELD_TRANSFERFUNCTION)) {
        fprintf(fd, "  Transfer Function: ");
        if (flags & TIFFPRINT_CURVES) {
            fprintf(fd, "\n");
            n = 1L<<td->td_bitspersample;
            for (l = 0; l < n; l++) {
                fprintf(fd, "    %2lu: %5u",
                    l, td->td_transferfunction[0][l]);
                for (i = 1; i < td->td_samplesperpixel; i++)
                    fprintf(fd, " %5u",
                        td->td_transferfunction[i][l]);
                fputc('\n', fd);
            }
        } else
            fprintf(fd, "(present)\n");
    }
    if (TIFFFieldSet(tif, FIELD_SUBIFD) && (td->td_subifd)) {
        fprintf(fd, "  SubIFD Offsets:");
        for (i = 0; i < td->td_nsubifd; i++)
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
            fprintf(fd, " %5I64u",
                (unsigned __int64) td->td_subifd[i]);
#else
            fprintf(fd, " %5llu",
                (unsigned long long) td->td_subifd[i]);
#endif
        fputc('\n', fd);
    }

    /*
    ** Custom tag support.
    */
    {
        int  i;
        short count;

        count = (short) TIFFGetTagListCount(tif);
        for(i = 0; i < count; i++) {
            uint32 tag = TIFFGetTagListEntry(tif, i);
            const TIFFField *fip;
            uint32 value_count;
            int mem_alloc = 0;
            void *raw_data;

            fip = TIFFFieldWithTag(tif, tag);
            if(fip == NULL)
                continue;

            if(fip->field_passcount) {
                if (fip->field_readcount == TIFF_VARIABLE2 ) {
                    if(TIFFGetField(tif, tag, &value_count, &raw_data) != 1)
                        continue;
                } else if (fip->field_readcount == TIFF_VARIABLE2 ) {
                    uint16 small_value_count;
                    if(TIFFGetField(tif, tag, &small_value_count, &raw_data) != 1)
                        continue;
                    value_count = small_value_count;
                } else {
                    assert (fip->field_readcount == TIFF_VARIABLE
                        || fip->field_readcount == TIFF_VARIABLE2);
                    continue;
                }
            } else {
                if (fip->field_readcount == TIFF_VARIABLE
                    || fip->field_readcount == TIFF_VARIABLE2)
                    value_count = 1;
                else if (fip->field_readcount == TIFF_SPP)
                    value_count = td->td_samplesperpixel;
                else
                    value_count = fip->field_readcount;
                if (fip->field_tag == TIFFTAG_DOTRANGE
                    && strcmp(fip->field_name,"DotRange") == 0) {
                    /* TODO: This is an evil exception and should not have been
                       handled this way ... likely best if we move it into
                       the directory structure with an explicit field in
                       libtiff 4.1 and assign it a FIELD_ value */
                    static uint16 dotrange[2];
                    raw_data = dotrange;
                    TIFFGetField(tif, tag, dotrange+0, dotrange+1);
                } else if (fip->field_type == TIFF_ASCII
                       || fip->field_readcount == TIFF_VARIABLE
                       || fip->field_readcount == TIFF_VARIABLE2
                       || fip->field_readcount == TIFF_SPP
                       || value_count > 1) {
                    if(TIFFGetField(tif, tag, &raw_data) != 1)
                        continue;
                } else {
                    raw_data = _TIFFmalloc(
                        _TIFFDataSize(fip->field_type)
                        * value_count);
                    mem_alloc = 1;
                    if(TIFFGetField(tif, tag, raw_data) != 1) {
                        _TIFFfree(raw_data);
                        continue;
                    }
                }
            }

            /*
             * Catch the tags which needs to be specially handled
             * and pretty print them. If tag not handled in
             * _TIFFPrettyPrintField() fall down and print it as
             * any other tag.
             */
            if (!_TIFFPrettyPrintField(tif, fip, fd, tag, value_count, raw_data))
                _TIFFPrintField(fd, fip, value_count, raw_data);

            if(mem_alloc)
                _TIFFfree(raw_data);
        }
    }

    if (tif->tif_tagmethods.printdir)
        (*tif->tif_tagmethods.printdir)(tif, fd, flags);

        _TIFFFillStriles( tif );

    if ((flags & TIFFPRINT_STRIPS) &&
        TIFFFieldSet(tif,FIELD_STRIPOFFSETS)) {
        uint32 s;

        fprintf(fd, "  %lu %s:\n",
            (long) td->td_nstrips,
            isTiled(tif) ? "Tiles" : "Strips");
        for (s = 0; s < td->td_nstrips; s++)
#if defined(__WIN32__) && (defined(_MSC_VER) || defined(__MINGW32__))
            fprintf(fd, "    %3lu: [%8I64u, %8I64u]\n",
                (unsigned long) s,
                (unsigned __int64) td->td_stripoffset[s],
                (unsigned __int64) td->td_stripbytecount[s]);
#else
            fprintf(fd, "    %3lu: [%8llu, %8llu]\n",
                (unsigned long) s,
                (unsigned long long) td->td_stripoffset[s],
                (unsigned long long) td->td_stripbytecount[s]);
#endif
    }
}

void
_TIFFprintAscii(FILE* fd, const char* cp)
{
    _TIFFprintAsciiBounded( fd, cp, strlen(cp));
}

static void
_TIFFprintAsciiBounded(FILE* fd, const char* cp, int max_chars)
{
    for (; max_chars > 0 && *cp != '\0'; cp++, max_chars--) {
        const char* tp;

        if (isprint((int)*cp)) {
            fputc(*cp, fd);
            continue;
        }
        for (tp = "\tt\bb\rr\nn\vv"; *tp; tp++)
            if (*tp++ == *cp)
                break;
        if (*tp)
            fprintf(fd, "\\%c", *tp);
        else
            fprintf(fd, "\\%03o", *cp & 0xff);
    }
}

void
_TIFFprintAsciiTag(FILE* fd, const char* name, const char* value)
{
    fprintf(fd, "  %s: \"", name);
    _TIFFprintAscii(fd, value);
    fprintf(fd, "\"\n");
}

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
