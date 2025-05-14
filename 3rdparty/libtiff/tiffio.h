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

#ifndef _TIFFIO_
#define _TIFFIO_

/*
 * TIFF I/O Library Definitions.
 */
#include "tiff.h"
#include "tiffvers.h"

/*
 * TIFF is defined as an incomplete type to hide the
 * library's internal data structures from clients.
 */
typedef struct tiff TIFF;

/*
 * The following typedefs define the intrinsic size of
 * data types used in the *exported* interfaces.  These
 * definitions depend on the proper definition of types
 * in tiff.h.  Note also that the varargs interface used
 * to pass tag types and values uses the types defined in
 * tiff.h directly.
 *
 * NB: ttag_t is unsigned int and not unsigned short because
 *     ANSI C requires that the type before the ellipsis be a
 *     promoted type (i.e. one of int, unsigned int, pointer,
 *     or double) and because we defined pseudo-tags that are
 *     outside the range of legal Aldus-assigned tags.
 * NB: tsize_t is signed and not unsigned because some functions
 *     return -1.
 * NB: toff_t is not off_t for many reasons; TIFFs max out at
 *     32-bit file offsets, and BigTIFF maxes out at 64-bit
 *     offsets being the most important, and to ensure use of
 *     a consistently unsigned type across architectures.
 *     Prior to libtiff 4.0, this was an unsigned 32 bit type.
 */
/*
 * this is the machine addressing size type, only it's signed, so make it
 * int32_t on 32bit machines, int64_t on 64bit machines
 */
typedef TIFF_SSIZE_T tmsize_t;
#define TIFF_TMSIZE_T_MAX (tmsize_t)(SIZE_MAX >> 1)

typedef uint64_t toff_t; /* file offset */
/* the following are deprecated and should be replaced by their defining
   counterparts */
typedef uint32_t ttag_t;    /* directory tag */
typedef uint32_t tdir_t;    /* directory index */
typedef uint16_t tsample_t; /* sample number */
typedef uint32_t tstrile_t; /* strip or tile number */
typedef tstrile_t tstrip_t; /* strip number */
typedef tstrile_t ttile_t;  /* tile number */
typedef tmsize_t tsize_t;   /* i/o size in bytes */
typedef void *tdata_t;      /* image data ref */

#if !defined(__WIN32__) && (defined(_WIN32) || defined(WIN32))
#define __WIN32__
#endif

/*
 * On windows you should define USE_WIN32_FILEIO if you are using tif_win32.c
 * or AVOID_WIN32_FILEIO if you are using something else (like tif_unix.c).
 *
 * By default tif_unix.c is assumed.
 */

#if defined(_WINDOWS) || defined(__WIN32__) || defined(_Windows)
#if !defined(__CYGWIN) && !defined(AVOID_WIN32_FILEIO) &&                      \
    !defined(USE_WIN32_FILEIO)
#define AVOID_WIN32_FILEIO
#endif
#endif

#if defined(USE_WIN32_FILEIO)
#define VC_EXTRALEAN
#include <windows.h>
#ifdef __WIN32__
DECLARE_HANDLE(thandle_t); /* Win32 file handle */
#else
typedef HFILE thandle_t; /* client data handle */
#endif /* __WIN32__ */
#else
typedef void *thandle_t; /* client data handle */
#endif /* USE_WIN32_FILEIO */

/*
 * Flags to pass to TIFFPrintDirectory to control
 * printing of data structures that are potentially
 * very large.   Bit-or these flags to enable printing
 * multiple items.
 */
#define TIFFPRINT_NONE 0x0           /* no extra info */
#define TIFFPRINT_STRIPS 0x1         /* strips/tiles info */
#define TIFFPRINT_CURVES 0x2         /* color/gray response curves */
#define TIFFPRINT_COLORMAP 0x4       /* colormap */
#define TIFFPRINT_JPEGQTABLES 0x100  /* JPEG Q matrices */
#define TIFFPRINT_JPEGACTABLES 0x200 /* JPEG AC tables */
#define TIFFPRINT_JPEGDCTABLES 0x200 /* JPEG DC tables */

/*
 * Colour conversion stuff
 */

/* reference white */
#define D65_X0 (95.0470F)
#define D65_Y0 (100.0F)
#define D65_Z0 (108.8827F)

#define D50_X0 (96.4250F)
#define D50_Y0 (100.0F)
#define D50_Z0 (82.4680F)

/* Structure for holding information about a display device. */

typedef unsigned char TIFFRGBValue; /* 8-bit samples */

typedef struct
{
    float d_mat[3][3]; /* XYZ -> luminance matrix */
    float d_YCR;       /* Light o/p for reference white */
    float d_YCG;
    float d_YCB;
    uint32_t d_Vrwr; /* Pixel values for ref. white */
    uint32_t d_Vrwg;
    uint32_t d_Vrwb;
    float d_Y0R; /* Residual light for black pixel */
    float d_Y0G;
    float d_Y0B;
    float d_gammaR; /* Gamma values for the three guns */
    float d_gammaG;
    float d_gammaB;
} TIFFDisplay;

typedef struct
{                           /* YCbCr->RGB support */
    TIFFRGBValue *clamptab; /* range clamping table */
    int *Cr_r_tab;
    int *Cb_b_tab;
    int32_t *Cr_g_tab;
    int32_t *Cb_g_tab;
    int32_t *Y_tab;
} TIFFYCbCrToRGB;

typedef struct
{              /* CIE Lab 1976->RGB support */
    int range; /* Size of conversion table */
#define CIELABTORGB_TABLE_RANGE 1500
    float rstep, gstep, bstep;
    float X0, Y0, Z0; /* Reference white point */
    TIFFDisplay display;
    float Yr2r[CIELABTORGB_TABLE_RANGE + 1]; /* Conversion of Yr to r */
    float Yg2g[CIELABTORGB_TABLE_RANGE + 1]; /* Conversion of Yg to g */
    float Yb2b[CIELABTORGB_TABLE_RANGE + 1]; /* Conversion of Yb to b */
} TIFFCIELabToRGB;

/*
 * RGBA-style image support.
 */
typedef struct _TIFFRGBAImage TIFFRGBAImage;
/*
 * The image reading and conversion routines invoke
 * ``put routines'' to copy/image/whatever tiles of
 * raw image data.  A default set of routines are
 * provided to convert/copy raw image data to 8-bit
 * packed ABGR format rasters.  Applications can supply
 * alternate routines that unpack the data into a
 * different format or, for example, unpack the data
 * and draw the unpacked raster on the display.
 */
typedef void (*tileContigRoutine)(TIFFRGBAImage *, uint32_t *, uint32_t,
                                  uint32_t, uint32_t, uint32_t, int32_t,
                                  int32_t, unsigned char *);
typedef void (*tileSeparateRoutine)(TIFFRGBAImage *, uint32_t *, uint32_t,
                                    uint32_t, uint32_t, uint32_t, int32_t,
                                    int32_t, unsigned char *, unsigned char *,
                                    unsigned char *, unsigned char *);
/*
 * RGBA-reader state.
 */
struct _TIFFRGBAImage
{
    TIFF *tif;                /* image handle */
    int stoponerr;            /* stop on read error */
    int isContig;             /* data is packed/separate */
    int alpha;                /* type of alpha data present */
    uint32_t width;           /* image width */
    uint32_t height;          /* image height */
    uint16_t bitspersample;   /* image bits/sample */
    uint16_t samplesperpixel; /* image samples/pixel */
    uint16_t orientation;     /* image orientation */
    uint16_t req_orientation; /* requested orientation */
    uint16_t photometric;     /* image photometric interp */
    uint16_t *redcmap;        /* colormap palette */
    uint16_t *greencmap;
    uint16_t *bluecmap;
    /* get image data routine */
    int (*get)(TIFFRGBAImage *, uint32_t *, uint32_t, uint32_t);
    /* put decoded strip/tile */
    union
    {
        void (*any)(TIFFRGBAImage *);
        tileContigRoutine contig;
        tileSeparateRoutine separate;
    } put;
    TIFFRGBValue *Map;       /* sample mapping array */
    uint32_t **BWmap;        /* black&white map */
    uint32_t **PALmap;       /* palette image map */
    TIFFYCbCrToRGB *ycbcr;   /* YCbCr conversion state */
    TIFFCIELabToRGB *cielab; /* CIE L*a*b conversion state */

    uint8_t *UaToAa; /* Unassociated alpha to associated alpha conversion LUT */
    uint8_t *Bitdepth16To8; /* LUT for conversion from 16bit to 8bit values */

    int row_offset;
    int col_offset;
};

/*
 * Macros for extracting components from the
 * packed ABGR form returned by TIFFReadRGBAImage.
 */
#define TIFFGetR(abgr) ((abgr)&0xff)
#define TIFFGetG(abgr) (((abgr) >> 8) & 0xff)
#define TIFFGetB(abgr) (((abgr) >> 16) & 0xff)
#define TIFFGetA(abgr) (((abgr) >> 24) & 0xff)

/*
 * A CODEC is a software package that implements decoding,
 * encoding, or decoding+encoding of a compression algorithm.
 * The library provides a collection of builtin codecs.
 * More codecs may be registered through calls to the library
 * and/or the builtin implementations may be overridden.
 */
typedef int (*TIFFInitMethod)(TIFF *, int);
typedef struct
{
    char *name;
    uint16_t scheme;
    TIFFInitMethod init;
} TIFFCodec;

typedef struct
{
    uint32_t uNum;
    uint32_t uDenom;
} TIFFRational_t;

#include <stdarg.h>
#include <stdio.h>

/* share internal LogLuv conversion routines? */
#ifndef LOGLUV_PUBLIC
#define LOGLUV_PUBLIC 1
#endif

#if defined(__GNUC__) || defined(__clang__) || defined(__attribute__)
#define TIFF_ATTRIBUTE(x) __attribute__(x)
#else
#define TIFF_ATTRIBUTE(x) /*nothing*/
#endif

#if defined(c_plusplus) || defined(__cplusplus)
extern "C"
{
#endif
    typedef void (*TIFFErrorHandler)(const char *, const char *, va_list);
    typedef void (*TIFFErrorHandlerExt)(thandle_t, const char *, const char *,
                                        va_list);
    typedef int (*TIFFErrorHandlerExtR)(TIFF *, void *user_data, const char *,
                                        const char *, va_list);
    typedef tmsize_t (*TIFFReadWriteProc)(thandle_t, void *, tmsize_t);
    typedef toff_t (*TIFFSeekProc)(thandle_t, toff_t, int);
    typedef int (*TIFFCloseProc)(thandle_t);
    typedef toff_t (*TIFFSizeProc)(thandle_t);
    typedef int (*TIFFMapFileProc)(thandle_t, void **base, toff_t *size);
    typedef void (*TIFFUnmapFileProc)(thandle_t, void *base, toff_t size);
    typedef void (*TIFFExtendProc)(TIFF *);

    extern const char *TIFFGetVersion(void);

    extern const TIFFCodec *TIFFFindCODEC(uint16_t);
    extern TIFFCodec *TIFFRegisterCODEC(uint16_t, const char *, TIFFInitMethod);
    extern void TIFFUnRegisterCODEC(TIFFCodec *);
    extern int TIFFIsCODECConfigured(uint16_t);
    extern TIFFCodec *TIFFGetConfiguredCODECs(void);

    /*
     * Auxiliary functions.
     */

    extern void *_TIFFmalloc(tmsize_t s);
    extern void *_TIFFcalloc(tmsize_t nmemb, tmsize_t siz);
    extern void *_TIFFrealloc(void *p, tmsize_t s);
    extern void _TIFFmemset(void *p, int v, tmsize_t c);
    extern void _TIFFmemcpy(void *d, const void *s, tmsize_t c);
    extern int _TIFFmemcmp(const void *p1, const void *p2, tmsize_t c);
    extern void _TIFFfree(void *p);

    /*
    ** Stuff, related to tag handling and creating custom tags.
    */
    extern int TIFFGetTagListCount(TIFF *);
    extern uint32_t TIFFGetTagListEntry(TIFF *, int tag_index);

#define TIFF_ANY TIFF_NOTYPE /* for field descriptor searching */
#define TIFF_VARIABLE -1     /* marker for variable length tags */
#define TIFF_SPP -2          /* marker for SamplesPerPixel tags */
#define TIFF_VARIABLE2 -3    /* marker for uint32_t var-length tags */

#define FIELD_CUSTOM 65

    typedef struct _TIFFField TIFFField;
    typedef struct _TIFFFieldArray TIFFFieldArray;

    extern const TIFFField *TIFFFindField(TIFF *, uint32_t, TIFFDataType);
    extern const TIFFField *TIFFFieldWithTag(TIFF *, uint32_t);
    extern const TIFFField *TIFFFieldWithName(TIFF *, const char *);

    extern uint32_t TIFFFieldTag(const TIFFField *);
    extern const char *TIFFFieldName(const TIFFField *);
    extern TIFFDataType TIFFFieldDataType(const TIFFField *);
    extern int TIFFFieldPassCount(const TIFFField *);
    extern int TIFFFieldReadCount(const TIFFField *);
    extern int TIFFFieldWriteCount(const TIFFField *);
    extern int
    TIFFFieldSetGetSize(const TIFFField *); /* returns internal storage size of
                                               TIFFSetGetFieldType in bytes. */
    extern int TIFFFieldSetGetCountSize(
        const TIFFField *); /* returns size of count parameter 0=none,
                               2=uint16_t, 4=uint32_t */
    extern int TIFFFieldIsAnonymous(const TIFFField *);

    typedef int (*TIFFVSetMethod)(TIFF *, uint32_t, va_list);
    typedef int (*TIFFVGetMethod)(TIFF *, uint32_t, va_list);
    typedef void (*TIFFPrintMethod)(TIFF *, FILE *, long);

    typedef struct
    {
        TIFFVSetMethod vsetfield; /* tag set routine */
        TIFFVGetMethod vgetfield; /* tag get routine */
        TIFFPrintMethod printdir; /* directory print routine */
    } TIFFTagMethods;

    extern TIFFTagMethods *TIFFAccessTagMethods(TIFF *);
    extern void *TIFFGetClientInfo(TIFF *, const char *);
    extern void TIFFSetClientInfo(TIFF *, void *, const char *);

    extern void TIFFCleanup(TIFF *tif);
    extern void TIFFClose(TIFF *tif);
    extern int TIFFFlush(TIFF *tif);
    extern int TIFFFlushData(TIFF *tif);
    extern int TIFFGetField(TIFF *tif, uint32_t tag, ...);
    extern int TIFFVGetField(TIFF *tif, uint32_t tag, va_list ap);
    extern int TIFFGetFieldDefaulted(TIFF *tif, uint32_t tag, ...);
    extern int TIFFVGetFieldDefaulted(TIFF *tif, uint32_t tag, va_list ap);
    extern int TIFFReadDirectory(TIFF *tif);
    extern int TIFFReadCustomDirectory(TIFF *tif, toff_t diroff,
                                       const TIFFFieldArray *infoarray);
    extern int TIFFReadEXIFDirectory(TIFF *tif, toff_t diroff);
    extern int TIFFReadGPSDirectory(TIFF *tif, toff_t diroff);
    extern uint64_t TIFFScanlineSize64(TIFF *tif);
    extern tmsize_t TIFFScanlineSize(TIFF *tif);
    extern uint64_t TIFFRasterScanlineSize64(TIFF *tif);
    extern tmsize_t TIFFRasterScanlineSize(TIFF *tif);
    extern uint64_t TIFFStripSize64(TIFF *tif);
    extern tmsize_t TIFFStripSize(TIFF *tif);
    extern uint64_t TIFFRawStripSize64(TIFF *tif, uint32_t strip);
    extern tmsize_t TIFFRawStripSize(TIFF *tif, uint32_t strip);
    extern uint64_t TIFFVStripSize64(TIFF *tif, uint32_t nrows);
    extern tmsize_t TIFFVStripSize(TIFF *tif, uint32_t nrows);
    extern uint64_t TIFFTileRowSize64(TIFF *tif);
    extern tmsize_t TIFFTileRowSize(TIFF *tif);
    extern uint64_t TIFFTileSize64(TIFF *tif);
    extern tmsize_t TIFFTileSize(TIFF *tif);
    extern uint64_t TIFFVTileSize64(TIFF *tif, uint32_t nrows);
    extern tmsize_t TIFFVTileSize(TIFF *tif, uint32_t nrows);
    extern uint32_t TIFFDefaultStripSize(TIFF *tif, uint32_t request);
    extern void TIFFDefaultTileSize(TIFF *, uint32_t *, uint32_t *);
    extern int TIFFFileno(TIFF *);
    extern int TIFFSetFileno(TIFF *, int);
    extern thandle_t TIFFClientdata(TIFF *);
    extern thandle_t TIFFSetClientdata(TIFF *, thandle_t);
    extern int TIFFGetMode(TIFF *);
    extern int TIFFSetMode(TIFF *, int);
    extern int TIFFIsTiled(TIFF *);
    extern int TIFFIsByteSwapped(TIFF *);
    extern int TIFFIsUpSampled(TIFF *);
    extern int TIFFIsMSB2LSB(TIFF *);
    extern int TIFFIsBigEndian(TIFF *);
    extern int TIFFIsBigTIFF(TIFF *);
    extern TIFFReadWriteProc TIFFGetReadProc(TIFF *);
    extern TIFFReadWriteProc TIFFGetWriteProc(TIFF *);
    extern TIFFSeekProc TIFFGetSeekProc(TIFF *);
    extern TIFFCloseProc TIFFGetCloseProc(TIFF *);
    extern TIFFSizeProc TIFFGetSizeProc(TIFF *);
    extern TIFFMapFileProc TIFFGetMapFileProc(TIFF *);
    extern TIFFUnmapFileProc TIFFGetUnmapFileProc(TIFF *);
    extern uint32_t TIFFCurrentRow(TIFF *);
    extern tdir_t TIFFCurrentDirectory(TIFF *);
    extern tdir_t TIFFNumberOfDirectories(TIFF *);
    extern uint64_t TIFFCurrentDirOffset(TIFF *);
    extern uint32_t TIFFCurrentStrip(TIFF *);
    extern uint32_t TIFFCurrentTile(TIFF *tif);
    extern int TIFFReadBufferSetup(TIFF *tif, void *bp, tmsize_t size);
    extern int TIFFWriteBufferSetup(TIFF *tif, void *bp, tmsize_t size);
    extern int TIFFSetupStrips(TIFF *);
    extern int TIFFWriteCheck(TIFF *, int, const char *);
    extern void TIFFFreeDirectory(TIFF *);
    extern int TIFFCreateDirectory(TIFF *);
    extern int TIFFCreateCustomDirectory(TIFF *, const TIFFFieldArray *);
    extern int TIFFCreateEXIFDirectory(TIFF *);
    extern int TIFFCreateGPSDirectory(TIFF *);
    extern int TIFFLastDirectory(TIFF *);
    extern int TIFFSetDirectory(TIFF *, tdir_t);
    extern int TIFFSetSubDirectory(TIFF *, uint64_t);
    extern int TIFFUnlinkDirectory(TIFF *, tdir_t);
    extern int TIFFSetField(TIFF *, uint32_t, ...);
    extern int TIFFVSetField(TIFF *, uint32_t, va_list);
    extern int TIFFUnsetField(TIFF *, uint32_t);
    extern int TIFFWriteDirectory(TIFF *);
    extern int TIFFWriteCustomDirectory(TIFF *, uint64_t *);
    extern int TIFFCheckpointDirectory(TIFF *);
    extern int TIFFRewriteDirectory(TIFF *);
    extern int TIFFDeferStrileArrayWriting(TIFF *);
    extern int TIFFForceStrileArrayWriting(TIFF *);

#if defined(c_plusplus) || defined(__cplusplus)
    extern void TIFFPrintDirectory(TIFF *, FILE *, long = 0);
    extern int TIFFReadScanline(TIFF *tif, void *buf, uint32_t row,
                                uint16_t sample = 0);
    extern int TIFFWriteScanline(TIFF *tif, void *buf, uint32_t row,
                                 uint16_t sample = 0);
    extern int TIFFReadRGBAImage(TIFF *, uint32_t, uint32_t, uint32_t *,
                                 int = 0);
    extern int TIFFReadRGBAImageOriented(TIFF *, uint32_t, uint32_t, uint32_t *,
                                         int = ORIENTATION_BOTLEFT, int = 0);
#else
extern void TIFFPrintDirectory(TIFF *, FILE *, long);
extern int TIFFReadScanline(TIFF *tif, void *buf, uint32_t row,
                            uint16_t sample);
extern int TIFFWriteScanline(TIFF *tif, void *buf, uint32_t row,
                             uint16_t sample);
extern int TIFFReadRGBAImage(TIFF *, uint32_t, uint32_t, uint32_t *, int);
extern int TIFFReadRGBAImageOriented(TIFF *, uint32_t, uint32_t, uint32_t *,
                                     int, int);
#endif

    extern int TIFFReadRGBAStrip(TIFF *, uint32_t, uint32_t *);
    extern int TIFFReadRGBATile(TIFF *, uint32_t, uint32_t, uint32_t *);
    extern int TIFFReadRGBAStripExt(TIFF *, uint32_t, uint32_t *,
                                    int stop_on_error);
    extern int TIFFReadRGBATileExt(TIFF *, uint32_t, uint32_t, uint32_t *,
                                   int stop_on_error);
    extern int TIFFRGBAImageOK(TIFF *, char[1024]);
    extern int TIFFRGBAImageBegin(TIFFRGBAImage *, TIFF *, int, char[1024]);
    extern int TIFFRGBAImageGet(TIFFRGBAImage *, uint32_t *, uint32_t,
                                uint32_t);
    extern void TIFFRGBAImageEnd(TIFFRGBAImage *);

    extern const char *TIFFFileName(TIFF *);
    extern const char *TIFFSetFileName(TIFF *, const char *);
    extern void TIFFError(const char *, const char *, ...)
        TIFF_ATTRIBUTE((__format__(__printf__, 2, 3)));
    extern void TIFFErrorExt(thandle_t, const char *, const char *, ...)
        TIFF_ATTRIBUTE((__format__(__printf__, 3, 4)));
    extern void TIFFWarning(const char *, const char *, ...)
        TIFF_ATTRIBUTE((__format__(__printf__, 2, 3)));
    extern void TIFFWarningExt(thandle_t, const char *, const char *, ...)
        TIFF_ATTRIBUTE((__format__(__printf__, 3, 4)));
    extern TIFFErrorHandler TIFFSetErrorHandler(TIFFErrorHandler);
    extern TIFFErrorHandlerExt TIFFSetErrorHandlerExt(TIFFErrorHandlerExt);
    extern TIFFErrorHandler TIFFSetWarningHandler(TIFFErrorHandler);
    extern TIFFErrorHandlerExt TIFFSetWarningHandlerExt(TIFFErrorHandlerExt);

    extern void TIFFWarningExtR(TIFF *, const char *, const char *, ...)
        TIFF_ATTRIBUTE((__format__(__printf__, 3, 4)));
    extern void TIFFErrorExtR(TIFF *, const char *, const char *, ...)
        TIFF_ATTRIBUTE((__format__(__printf__, 3, 4)));

    typedef struct TIFFOpenOptions TIFFOpenOptions;
    extern TIFFOpenOptions *TIFFOpenOptionsAlloc(void);
    extern void TIFFOpenOptionsFree(TIFFOpenOptions *);
    extern void
    TIFFOpenOptionsSetMaxSingleMemAlloc(TIFFOpenOptions *opts,
                                        tmsize_t max_single_mem_alloc);
    extern void
    TIFFOpenOptionsSetErrorHandlerExtR(TIFFOpenOptions *opts,
                                       TIFFErrorHandlerExtR handler,
                                       void *errorhandler_user_data);
    extern void
    TIFFOpenOptionsSetWarningHandlerExtR(TIFFOpenOptions *opts,
                                         TIFFErrorHandlerExtR handler,
                                         void *warnhandler_user_data);

    extern TIFF *TIFFOpen(const char *, const char *);
    extern TIFF *TIFFOpenExt(const char *, const char *, TIFFOpenOptions *opts);
#ifdef __WIN32__
    extern TIFF *TIFFOpenW(const wchar_t *, const char *);
    extern TIFF *TIFFOpenWExt(const wchar_t *, const char *,
                              TIFFOpenOptions *opts);
#endif /* __WIN32__ */
    extern TIFF *TIFFFdOpen(int, const char *, const char *);
    extern TIFF *TIFFFdOpenExt(int, const char *, const char *,
                               TIFFOpenOptions *opts);
    extern TIFF *TIFFClientOpen(const char *, const char *, thandle_t,
                                TIFFReadWriteProc, TIFFReadWriteProc,
                                TIFFSeekProc, TIFFCloseProc, TIFFSizeProc,
                                TIFFMapFileProc, TIFFUnmapFileProc);
    extern TIFF *TIFFClientOpenExt(const char *, const char *, thandle_t,
                                   TIFFReadWriteProc, TIFFReadWriteProc,
                                   TIFFSeekProc, TIFFCloseProc, TIFFSizeProc,
                                   TIFFMapFileProc, TIFFUnmapFileProc,
                                   TIFFOpenOptions *opts);
    extern TIFFExtendProc TIFFSetTagExtender(TIFFExtendProc);
    extern uint32_t TIFFComputeTile(TIFF *tif, uint32_t x, uint32_t y,
                                    uint32_t z, uint16_t s);
    extern int TIFFCheckTile(TIFF *tif, uint32_t x, uint32_t y, uint32_t z,
                             uint16_t s);
    extern uint32_t TIFFNumberOfTiles(TIFF *);
    extern tmsize_t TIFFReadTile(TIFF *tif, void *buf, uint32_t x, uint32_t y,
                                 uint32_t z, uint16_t s);
    extern tmsize_t TIFFWriteTile(TIFF *tif, void *buf, uint32_t x, uint32_t y,
                                  uint32_t z, uint16_t s);
    extern uint32_t TIFFComputeStrip(TIFF *, uint32_t, uint16_t);
    extern uint32_t TIFFNumberOfStrips(TIFF *);
    extern tmsize_t TIFFReadEncodedStrip(TIFF *tif, uint32_t strip, void *buf,
                                         tmsize_t size);
    extern tmsize_t TIFFReadRawStrip(TIFF *tif, uint32_t strip, void *buf,
                                     tmsize_t size);
    extern tmsize_t TIFFReadEncodedTile(TIFF *tif, uint32_t tile, void *buf,
                                        tmsize_t size);
    extern tmsize_t TIFFReadRawTile(TIFF *tif, uint32_t tile, void *buf,
                                    tmsize_t size);
    extern int TIFFReadFromUserBuffer(TIFF *tif, uint32_t strile, void *inbuf,
                                      tmsize_t insize, void *outbuf,
                                      tmsize_t outsize);
    extern tmsize_t TIFFWriteEncodedStrip(TIFF *tif, uint32_t strip, void *data,
                                          tmsize_t cc);
    extern tmsize_t TIFFWriteRawStrip(TIFF *tif, uint32_t strip, void *data,
                                      tmsize_t cc);
    extern tmsize_t TIFFWriteEncodedTile(TIFF *tif, uint32_t tile, void *data,
                                         tmsize_t cc);
    extern tmsize_t TIFFWriteRawTile(TIFF *tif, uint32_t tile, void *data,
                                     tmsize_t cc);
    extern int TIFFDataWidth(
        TIFFDataType); /* table of tag datatype widths within TIFF file. */
    extern void TIFFSetWriteOffset(TIFF *tif, toff_t off);
    extern void TIFFSwabShort(uint16_t *);
    extern void TIFFSwabLong(uint32_t *);
    extern void TIFFSwabLong8(uint64_t *);
    extern void TIFFSwabFloat(float *);
    extern void TIFFSwabDouble(double *);
    extern void TIFFSwabArrayOfShort(uint16_t *wp, tmsize_t n);
    extern void TIFFSwabArrayOfTriples(uint8_t *tp, tmsize_t n);
    extern void TIFFSwabArrayOfLong(uint32_t *lp, tmsize_t n);
    extern void TIFFSwabArrayOfLong8(uint64_t *lp, tmsize_t n);
    extern void TIFFSwabArrayOfFloat(float *fp, tmsize_t n);
    extern void TIFFSwabArrayOfDouble(double *dp, tmsize_t n);
    extern void TIFFReverseBits(uint8_t *cp, tmsize_t n);
    extern const unsigned char *TIFFGetBitRevTable(int);

    extern uint64_t TIFFGetStrileOffset(TIFF *tif, uint32_t strile);
    extern uint64_t TIFFGetStrileByteCount(TIFF *tif, uint32_t strile);
    extern uint64_t TIFFGetStrileOffsetWithErr(TIFF *tif, uint32_t strile,
                                               int *pbErr);
    extern uint64_t TIFFGetStrileByteCountWithErr(TIFF *tif, uint32_t strile,
                                                  int *pbErr);

#ifdef LOGLUV_PUBLIC
#define U_NEU 0.210526316
#define V_NEU 0.473684211
#define UVSCALE 410.
    extern double LogL16toY(int);
    extern double LogL10toY(int);
    extern void XYZtoRGB24(float *, uint8_t *);
    extern int uv_decode(double *, double *, int);
    extern void LogLuv24toXYZ(uint32_t, float *);
    extern void LogLuv32toXYZ(uint32_t, float *);
#if defined(c_plusplus) || defined(__cplusplus)
    extern int LogL16fromY(double, int = SGILOGENCODE_NODITHER);
    extern int LogL10fromY(double, int = SGILOGENCODE_NODITHER);
    extern int uv_encode(double, double, int = SGILOGENCODE_NODITHER);
    extern uint32_t LogLuv24fromXYZ(float *, int = SGILOGENCODE_NODITHER);
    extern uint32_t LogLuv32fromXYZ(float *, int = SGILOGENCODE_NODITHER);
#else
    extern int LogL16fromY(double, int);
    extern int LogL10fromY(double, int);
    extern int uv_encode(double, double, int);
    extern uint32_t LogLuv24fromXYZ(float *, int);
    extern uint32_t LogLuv32fromXYZ(float *, int);
#endif
#endif /* LOGLUV_PUBLIC */

    extern int TIFFCIELabToRGBInit(TIFFCIELabToRGB *, const TIFFDisplay *,
                                   float *);
    extern void TIFFCIELabToXYZ(TIFFCIELabToRGB *, uint32_t, int32_t, int32_t,
                                float *, float *, float *);
    extern void TIFFXYZToRGB(TIFFCIELabToRGB *, float, float, float, uint32_t *,
                             uint32_t *, uint32_t *);

    extern int TIFFYCbCrToRGBInit(TIFFYCbCrToRGB *, float *, float *);
    extern void TIFFYCbCrtoRGB(TIFFYCbCrToRGB *, uint32_t, int32_t, int32_t,
                               uint32_t *, uint32_t *, uint32_t *);

    /****************************************************************************
     *               O B S O L E T E D    I N T E R F A C E S
     *
     * Don't use this stuff in your applications, it may be removed in the
     *future libtiff versions.
     ****************************************************************************/
    typedef struct
    {
        ttag_t field_tag;               /* field's tag */
        short field_readcount;          /* read count/TIFF_VARIABLE/TIFF_SPP */
        short field_writecount;         /* write count/TIFF_VARIABLE */
        TIFFDataType field_type;        /* type of associated data */
        unsigned short field_bit;       /* bit in fieldsset bit vector */
        unsigned char field_oktochange; /* if true, can change while writing */
        unsigned char field_passcount;  /* if true, pass dir count on set */
        char *field_name;               /* ASCII name */
    } TIFFFieldInfo;

    extern int TIFFMergeFieldInfo(TIFF *, const TIFFFieldInfo[], uint32_t);

#if defined(c_plusplus) || defined(__cplusplus)
}
#endif

#endif /* _TIFFIO_ */
