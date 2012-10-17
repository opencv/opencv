/* $Id: tif_dir.h,v 1.54 2011-02-18 20:53:05 fwarmerdam Exp $ */

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

#ifndef _TIFFDIR_
#define	_TIFFDIR_
/*
 * ``Library-private'' Directory-related Definitions.
 */

typedef struct {
    const TIFFField *info;
    int             count;
    void           *value;
} TIFFTagValue;

/*
 * TIFF Image File Directories are comprised of a table of field
 * descriptors of the form shown below.  The table is sorted in
 * ascending order by tag.  The values associated with each entry are
 * disjoint and may appear anywhere in the file (so long as they are
 * placed on a word boundary).
 *
 * If the value is 4 bytes or less, in ClassicTIFF, or 8 bytes or less in
 * BigTIFF, then it is placed in the offset field to save space. If so,
 * it is left-justified in the offset field.
 */
typedef struct {
    uint16 tdir_tag;        /* see below */
    uint16 tdir_type;       /* data type; see below */
    uint64 tdir_count;      /* number of items; length in spec */
    union {
        uint16 toff_short;
        uint32 toff_long;
        uint64 toff_long8;
    } tdir_offset;		/* either offset or the data itself if fits */
} TIFFDirEntry;

/*
 * Internal format of a TIFF directory entry.
 */
typedef struct {
#define FIELD_SETLONGS 4
    /* bit vector of fields that are set */
    unsigned long td_fieldsset[FIELD_SETLONGS];

    uint32  td_imagewidth, td_imagelength, td_imagedepth;
    uint32  td_tilewidth, td_tilelength, td_tiledepth;
    uint32  td_subfiletype;
    uint16  td_bitspersample;
    uint16  td_sampleformat;
    uint16  td_compression;
    uint16  td_photometric;
    uint16  td_threshholding;
    uint16  td_fillorder;
    uint16  td_orientation;
    uint16  td_samplesperpixel;
    uint32  td_rowsperstrip;
    uint16  td_minsamplevalue, td_maxsamplevalue;
    double* td_sminsamplevalue;
    double* td_smaxsamplevalue;
    float   td_xresolution, td_yresolution;
    uint16  td_resolutionunit;
    uint16  td_planarconfig;
    float   td_xposition, td_yposition;
    uint16  td_pagenumber[2];
    uint16* td_colormap[3];
    uint16  td_halftonehints[2];
    uint16  td_extrasamples;
    uint16* td_sampleinfo;
    /* even though the name is misleading, td_stripsperimage is the number
     * of striles (=strips or tiles) per plane, and td_nstrips the total
     * number of striles */
    uint32  td_stripsperimage;
    uint32  td_nstrips;              /* size of offset & bytecount arrays */
    uint64* td_stripoffset;
    uint64* td_stripbytecount;
    int     td_stripbytecountsorted; /* is the bytecount array sorted ascending? */
#if defined(DEFER_STRILE_LOAD)
        TIFFDirEntry td_stripoffset_entry;    /* for deferred loading */
        TIFFDirEntry td_stripbytecount_entry; /* for deferred loading */
#endif
    uint16  td_nsubifd;
    uint64* td_subifd;
    /* YCbCr parameters */
    uint16  td_ycbcrsubsampling[2];
    uint16  td_ycbcrpositioning;
    /* Colorimetry parameters */
    uint16* td_transferfunction[3];
    float*	td_refblackwhite;
    /* CMYK parameters */
    int     td_inknameslen;
    char*   td_inknames;

    int     td_customValueCount;
        TIFFTagValue *td_customValues;
} TIFFDirectory;

/*
 * Field flags used to indicate fields that have been set in a directory, and
 * to reference fields when manipulating a directory.
 */

/*
 * FIELD_IGNORE is used to signify tags that are to be processed but otherwise
 * ignored.  This permits antiquated tags to be quietly read and discarded.
 * Note that a bit *is* allocated for ignored tags; this is understood by the
 * directory reading logic which uses this fact to avoid special-case handling
 */
#define FIELD_IGNORE                   0

/* multi-item fields */
#define FIELD_IMAGEDIMENSIONS          1
#define FIELD_TILEDIMENSIONS           2
#define FIELD_RESOLUTION               3
#define FIELD_POSITION                 4

/* single-item fields */
#define FIELD_SUBFILETYPE              5
#define FIELD_BITSPERSAMPLE            6
#define FIELD_COMPRESSION              7
#define FIELD_PHOTOMETRIC              8
#define FIELD_THRESHHOLDING            9
#define FIELD_FILLORDER                10
#define FIELD_ORIENTATION              15
#define FIELD_SAMPLESPERPIXEL          16
#define FIELD_ROWSPERSTRIP             17
#define FIELD_MINSAMPLEVALUE           18
#define FIELD_MAXSAMPLEVALUE           19
#define FIELD_PLANARCONFIG             20
#define FIELD_RESOLUTIONUNIT           22
#define FIELD_PAGENUMBER               23
#define FIELD_STRIPBYTECOUNTS          24
#define FIELD_STRIPOFFSETS             25
#define FIELD_COLORMAP                 26
#define FIELD_EXTRASAMPLES             31
#define FIELD_SAMPLEFORMAT             32
#define FIELD_SMINSAMPLEVALUE          33
#define FIELD_SMAXSAMPLEVALUE          34
#define FIELD_IMAGEDEPTH               35
#define FIELD_TILEDEPTH                36
#define FIELD_HALFTONEHINTS            37
#define FIELD_YCBCRSUBSAMPLING         39
#define FIELD_YCBCRPOSITIONING         40
#define	FIELD_REFBLACKWHITE            41
#define FIELD_TRANSFERFUNCTION         44
#define FIELD_INKNAMES                 46
#define FIELD_SUBIFD                   49
/*      FIELD_CUSTOM (see tiffio.h)    65 */
/* end of support for well-known tags; codec-private tags follow */
#define FIELD_CODEC                    66  /* base of codec-private tags */


/*
 * Pseudo-tags don't normally need field bits since they are not written to an
 * output file (by definition). The library also has express logic to always
 * query a codec for a pseudo-tag so allocating a field bit for one is a
 * waste.   If codec wants to promote the notion of a pseudo-tag being ``set''
 * or ``unset'' then it can do using internal state flags without polluting
 * the field bit space defined for real tags.
 */
#define FIELD_PSEUDO			0

#define FIELD_LAST			(32*FIELD_SETLONGS-1)

#define BITn(n)				(((unsigned long)1L)<<((n)&0x1f))
#define BITFIELDn(tif, n)		((tif)->tif_dir.td_fieldsset[(n)/32])
#define TIFFFieldSet(tif, field)	(BITFIELDn(tif, field) & BITn(field))
#define TIFFSetFieldBit(tif, field)	(BITFIELDn(tif, field) |= BITn(field))
#define TIFFClrFieldBit(tif, field)	(BITFIELDn(tif, field) &= ~BITn(field))

#define FieldSet(fields, f)		(fields[(f)/32] & BITn(f))
#define ResetFieldBit(fields, f)	(fields[(f)/32] &= ~BITn(f))

typedef enum {
    TIFF_SETGET_UNDEFINED = 0,
    TIFF_SETGET_ASCII = 1,
    TIFF_SETGET_UINT8 = 2,
    TIFF_SETGET_SINT8 = 3,
    TIFF_SETGET_UINT16 = 4,
    TIFF_SETGET_SINT16 = 5,
    TIFF_SETGET_UINT32 = 6,
    TIFF_SETGET_SINT32 = 7,
    TIFF_SETGET_UINT64 = 8,
    TIFF_SETGET_SINT64 = 9,
    TIFF_SETGET_FLOAT = 10,
    TIFF_SETGET_DOUBLE = 11,
    TIFF_SETGET_IFD8 = 12,
    TIFF_SETGET_INT = 13,
    TIFF_SETGET_UINT16_PAIR = 14,
    TIFF_SETGET_C0_ASCII = 15,
    TIFF_SETGET_C0_UINT8 = 16,
    TIFF_SETGET_C0_SINT8 = 17,
    TIFF_SETGET_C0_UINT16 = 18,
    TIFF_SETGET_C0_SINT16 = 19,
    TIFF_SETGET_C0_UINT32 = 20,
    TIFF_SETGET_C0_SINT32 = 21,
    TIFF_SETGET_C0_UINT64 = 22,
    TIFF_SETGET_C0_SINT64 = 23,
    TIFF_SETGET_C0_FLOAT = 24,
    TIFF_SETGET_C0_DOUBLE = 25,
    TIFF_SETGET_C0_IFD8 = 26,
    TIFF_SETGET_C16_ASCII = 27,
    TIFF_SETGET_C16_UINT8 = 28,
    TIFF_SETGET_C16_SINT8 = 29,
    TIFF_SETGET_C16_UINT16 = 30,
    TIFF_SETGET_C16_SINT16 = 31,
    TIFF_SETGET_C16_UINT32 = 32,
    TIFF_SETGET_C16_SINT32 = 33,
    TIFF_SETGET_C16_UINT64 = 34,
    TIFF_SETGET_C16_SINT64 = 35,
    TIFF_SETGET_C16_FLOAT = 36,
    TIFF_SETGET_C16_DOUBLE = 37,
    TIFF_SETGET_C16_IFD8 = 38,
    TIFF_SETGET_C32_ASCII = 39,
    TIFF_SETGET_C32_UINT8 = 40,
    TIFF_SETGET_C32_SINT8 = 41,
    TIFF_SETGET_C32_UINT16 = 42,
    TIFF_SETGET_C32_SINT16 = 43,
    TIFF_SETGET_C32_UINT32 = 44,
    TIFF_SETGET_C32_SINT32 = 45,
    TIFF_SETGET_C32_UINT64 = 46,
    TIFF_SETGET_C32_SINT64 = 47,
    TIFF_SETGET_C32_FLOAT = 48,
    TIFF_SETGET_C32_DOUBLE = 49,
    TIFF_SETGET_C32_IFD8 = 50,
    TIFF_SETGET_OTHER = 51
} TIFFSetGetFieldType;

#if defined(__cplusplus)
extern "C" {
#endif

extern const TIFFFieldArray* _TIFFGetFields(void);
extern const TIFFFieldArray* _TIFFGetExifFields(void);
extern void _TIFFSetupFields(TIFF* tif, const TIFFFieldArray* infoarray);
extern void _TIFFPrintFieldInfo(TIFF*, FILE*);

extern int _TIFFFillStriles(TIFF*);

typedef enum {
    tfiatImage,
    tfiatExif,
    tfiatOther
} TIFFFieldArrayType;

struct _TIFFFieldArray {
    TIFFFieldArrayType type;    /* array type, will be used to determine if IFD is image and such */
    uint32 allocated_size;      /* 0 if array is constant, other if modified by future definition extension support */
    uint32 count;               /* number of elements in fields array */
    TIFFField* fields;          /* actual field info */
};

struct _TIFFField {
    uint32 field_tag;                       /* field's tag */
    short field_readcount;                  /* read count/TIFF_VARIABLE/TIFF_SPP */
    short field_writecount;                 /* write count/TIFF_VARIABLE */
    TIFFDataType field_type;                /* type of associated data */
    uint32 reserved;                        /* reserved for future extension */
    TIFFSetGetFieldType set_field_type;     /* type to be passed to TIFFSetField */
    TIFFSetGetFieldType get_field_type;     /* type to be passed to TIFFGetField */
    unsigned short field_bit;               /* bit in fieldsset bit vector */
    unsigned char field_oktochange;         /* if true, can change while writing */
    unsigned char field_passcount;          /* if true, pass dir count on set */
    char* field_name;                       /* ASCII name */
    TIFFFieldArray* field_subfields;        /* if field points to child ifds, child ifd field definition array */
};

extern int _TIFFMergeFields(TIFF*, const TIFFField[], uint32);
extern const TIFFField* _TIFFFindOrRegisterField(TIFF *, uint32, TIFFDataType);
extern  TIFFField* _TIFFCreateAnonField(TIFF *, uint32, TIFFDataType);

#if defined(__cplusplus)
}
#endif
#endif /* _TIFFDIR_ */

/* vim: set ts=8 sts=8 sw=8 noet: */

/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
