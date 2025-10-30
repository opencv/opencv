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

#ifndef _TIFFIOP_
#define _TIFFIOP_
/*
 * ``Library-private'' definitions.
 */

#include "tif_config.h"

#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#include <string.h>

#ifdef HAVE_ASSERT_H
#include <assert.h>
#else
#define assert(x)
#endif

#include "tif_hash_set.h"
#include "tiffio.h"

#include "tif_dir.h"

#include <limits.h>

#ifndef STRIP_SIZE_DEFAULT
#define STRIP_SIZE_DEFAULT 8192
#endif

#ifndef TIFF_MAX_DIR_COUNT
#define TIFF_MAX_DIR_COUNT 1048576
#endif

#define TIFF_NON_EXISTENT_DIR_NUMBER UINT_MAX

#define streq(a, b) (strcmp(a, b) == 0)
#define strneq(a, b, n) (strncmp(a, b, n) == 0)

#ifndef TRUE
#define TRUE 1
#define FALSE 0
#endif

typedef struct client_info
{
    struct client_info *next;
    void *data;
    char *name;
} TIFFClientInfoLink;

/*
 * Typedefs for ``method pointers'' used internally.
 * these are deprecated and provided only for backwards compatibility.
 */
typedef unsigned char tidataval_t; /* internal image data value type */
typedef tidataval_t *tidata_t;     /* reference to internal image data */

typedef void (*TIFFVoidMethod)(TIFF *);
typedef int (*TIFFBoolMethod)(TIFF *);
typedef int (*TIFFPreMethod)(TIFF *, uint16_t);
typedef int (*TIFFCodeMethod)(TIFF *tif, uint8_t *buf, tmsize_t size,
                              uint16_t sample);
typedef int (*TIFFSeekMethod)(TIFF *, uint32_t);
typedef void (*TIFFPostMethod)(TIFF *tif, uint8_t *buf, tmsize_t size);
typedef uint32_t (*TIFFStripMethod)(TIFF *, uint32_t);
typedef void (*TIFFTileMethod)(TIFF *, uint32_t *, uint32_t *);

struct TIFFOffsetAndDirNumber
{
    uint64_t offset;
    tdir_t dirNumber;
};
typedef struct TIFFOffsetAndDirNumber TIFFOffsetAndDirNumber;

typedef union
{
    TIFFHeaderCommon common;
    TIFFHeaderClassic classic;
    TIFFHeaderBig big;
} TIFFHeaderUnion;

struct tiff
{
    char *tif_name; /* name of open file */
    int tif_fd;     /* open file descriptor */
    int tif_mode;   /* open mode (O_*) */
    uint32_t tif_flags;
#define TIFF_FILLORDER 0x00003U   /* natural bit fill order for machine */
#define TIFF_DIRTYHEADER 0x00004U /* header must be written on close */
#define TIFF_DIRTYDIRECT 0x00008U /* current directory must be written */
#define TIFF_BUFFERSETUP 0x00010U /* data buffers setup */
#define TIFF_CODERSETUP 0x00020U  /* encoder/decoder setup done */
#define TIFF_BEENWRITING 0x00040U /* written 1+ scanlines to file */
#define TIFF_SWAB 0x00080U        /* byte swap file information */
#define TIFF_NOBITREV 0x00100U    /* inhibit bit reversal logic */
#define TIFF_MYBUFFER 0x00200U    /* my raw data buffer; free on close */
#define TIFF_ISTILED 0x00400U     /* file is tile, not strip- based */
#define TIFF_MAPPED 0x00800U      /* file is mapped into memory */
#define TIFF_POSTENCODE 0x01000U  /* need call to postencode routine */
#define TIFF_INSUBIFD 0x02000U    /* currently writing a subifd */
#define TIFF_UPSAMPLED 0x04000U   /* library is doing data up-sampling */
#define TIFF_STRIPCHOP 0x08000U   /* enable strip chopping support */
#define TIFF_HEADERONLY                                                        \
    0x10000U /* read header only, do not process the first directory */
#define TIFF_NOREADRAW                                                         \
    0x20000U /* skip reading of raw uncompressed image data */
#define TIFF_INCUSTOMIFD 0x40000U /* currently writing a custom IFD */
#define TIFF_BIGTIFF 0x80000U     /* read/write bigtiff */
#define TIFF_BUF4WRITE 0x100000U  /* rawcc bytes are for writing */
#define TIFF_DIRTYSTRIP 0x200000U /* stripoffsets/stripbytecount dirty*/
#define TIFF_PERSAMPLE 0x400000U  /* get/set per sample tags as arrays */
#define TIFF_BUFFERMMAP                                                        \
    0x800000U /* read buffer (tif_rawdata) points into mmap() memory */
#define TIFF_DEFERSTRILELOAD                                                   \
    0x1000000U /* defer strip/tile offset/bytecount array loading. */
#define TIFF_LAZYSTRILELOAD                                                    \
    0x2000000U /* lazy/ondemand loading of strip/tile offset/bytecount values. \
                  Only used if TIFF_DEFERSTRILELOAD is set and in read-only    \
                  mode */
#define TIFF_CHOPPEDUPARRAYS                                                   \
    0x4000000U /* set when allocChoppedUpStripArrays() has modified strip      \
                  array */
    uint64_t tif_diroff;     /* file offset of current directory */
    uint64_t tif_nextdiroff; /* file offset of following directory */
    uint64_t tif_lastdiroff; /* file offset of last directory written so far */
    TIFFHashSet *tif_map_dir_offset_to_number;
    TIFFHashSet *tif_map_dir_number_to_offset;
    int tif_setdirectory_force_absolute; /* switch between relative and absolute
                                            stepping in TIFFSetDirectory() */
    TIFFDirectory tif_dir;               /* internal rep of current directory */
    TIFFDirectory
        tif_customdir; /* custom IFDs are separated from the main ones */
    TIFFHeaderUnion tif_header; /* file's header block Classic/BigTIFF union */
    uint16_t tif_header_size;   /* file's header block and its length */
    uint32_t tif_row;           /* current scanline */

    /* There are IFDs in the file and an "active" IFD in memory,
     * from which fields are "set" and "get".
     * tif_curdir is set to:
     *   a) TIFF_NON_EXISTENT_DIR_NUMBER if there is no IFD in the file
     *      or the state is unknown,
     *      or the last read (i.e. TIFFFetchDirectory()) failed,
     *      or a custom directory was written.
     *   b) IFD index of last IFD written in the file. In this case the
     *      active IFD is a new (empty) one and tif_diroff is zero.
     *      If writing fails, tif_curdir is not changed.
     *   c) IFD index of IFD read from file into memory (=active IFD),
     *      even if IFD is corrupt and TIFFReadDirectory() returns 0.
     *      Then tif_diroff contains the offset of the IFD in the file.
     *   d) IFD index 0, whenever a custom directory or an unchained SubIFD
     *      was read. */
    tdir_t tif_curdir; /* current directory (index) */
    /* tif_curdircount: number of directories (main-IFDs) in file:
     * - TIFF_NON_EXISTENT_DIR_NUMBER means 'dont know number of IFDs'.
     * - 0 means 'empty file opened for writing, but no IFD written yet' */
    tdir_t tif_curdircount;
    uint32_t tif_curstrip;     /* current strip for read/write */
    uint64_t tif_curoff;       /* current offset for read/write */
    uint64_t tif_lastvalidoff; /* last valid offset allowed for rewrite in
                                  place. Used only by TIFFAppendToStrip() */
    uint64_t tif_dataoff;      /* current offset for writing dir (IFD) */
    /* SubIFD support */
    uint16_t tif_nsubifd;   /* remaining subifds to write */
    uint64_t tif_subifdoff; /* offset for patching SubIFD link */
    /* tiling support */
    uint32_t tif_col;      /* current column (offset by row too) */
    uint32_t tif_curtile;  /* current tile for read/write */
    tmsize_t tif_tilesize; /* # of bytes in a tile */
    /* compression scheme hooks */
    int tif_decodestatus;
    TIFFBoolMethod tif_fixuptags;   /* called in TIFFReadDirectory */
    TIFFBoolMethod tif_setupdecode; /* called once before predecode */
    TIFFPreMethod tif_predecode;    /* pre- row/strip/tile decoding */
    TIFFBoolMethod tif_setupencode; /* called once before preencode */
    int tif_encodestatus;
    TIFFPreMethod tif_preencode;      /* pre- row/strip/tile encoding */
    TIFFBoolMethod tif_postencode;    /* post- row/strip/tile encoding */
    TIFFCodeMethod tif_decoderow;     /* scanline decoding routine */
    TIFFCodeMethod tif_encoderow;     /* scanline encoding routine */
    TIFFCodeMethod tif_decodestrip;   /* strip decoding routine */
    TIFFCodeMethod tif_encodestrip;   /* strip encoding routine */
    TIFFCodeMethod tif_decodetile;    /* tile decoding routine */
    TIFFCodeMethod tif_encodetile;    /* tile encoding routine */
    TIFFVoidMethod tif_close;         /* cleanup-on-close routine */
    TIFFSeekMethod tif_seek;          /* position within a strip routine */
    TIFFVoidMethod tif_cleanup;       /* cleanup state routine */
    TIFFStripMethod tif_defstripsize; /* calculate/constrain strip size */
    TIFFTileMethod tif_deftilesize;   /* calculate/constrain tile size */
    uint8_t *tif_data;                /* compression scheme private data */
    /* input/output buffering */
    tmsize_t tif_scanlinesize;  /* # of bytes in a scanline */
    tmsize_t tif_scanlineskew;  /* scanline skew for reading strips */
    uint8_t *tif_rawdata;       /* raw data buffer */
    tmsize_t tif_rawdatasize;   /* # of bytes in raw data buffer */
    tmsize_t tif_rawdataoff;    /* rawdata offset within strip */
    tmsize_t tif_rawdataloaded; /* amount of data in rawdata */
    uint8_t *tif_rawcp;         /* current spot in raw buffer */
    tmsize_t tif_rawcc;         /* bytes unread from raw buffer */
    /* memory-mapped file support */
    uint8_t *tif_base; /* base of mapped file */
    tmsize_t tif_size; /* size of mapped file region (bytes, thus tmsize_t) */
    TIFFMapFileProc tif_mapproc;     /* map file method */
    TIFFUnmapFileProc tif_unmapproc; /* unmap file method */
    /* input/output callback methods */
    thandle_t tif_clientdata;        /* callback parameter */
    TIFFReadWriteProc tif_readproc;  /* read method */
    TIFFReadWriteProc tif_writeproc; /* write method */
    TIFFSeekProc tif_seekproc;       /* lseek method */
    TIFFCloseProc tif_closeproc;     /* close method */
    TIFFSizeProc tif_sizeproc;       /* filesize method */
    /* post-decoding support */
    TIFFPostMethod tif_postdecode; /* post decoding routine */
    /* tag support */
    TIFFField **tif_fields;          /* sorted table of registered tags */
    size_t tif_nfields;              /* # entries in registered tag table */
    const TIFFField *tif_foundfield; /* cached pointer to already found tag */
    TIFFTagMethods tif_tagmethods;   /* tag get/set/print routines */
    TIFFClientInfoLink *tif_clientinfo; /* extra client information. */
    /* Backward compatibility stuff. We need these two fields for
     * setting up an old tag extension scheme. */
    TIFFFieldArray *tif_fieldscompat;
    size_t tif_nfieldscompat;
    /* Error handler support */
    TIFFErrorHandlerExtR tif_errorhandler;
    void *tif_errorhandler_user_data;
    TIFFErrorHandlerExtR tif_warnhandler;
    void *tif_warnhandler_user_data;
    tmsize_t tif_max_single_mem_alloc;    /* in bytes. 0 for unlimited */
    tmsize_t tif_max_cumulated_mem_alloc; /* in bytes. 0 for unlimited */
    tmsize_t tif_cur_cumulated_mem_alloc; /* in bytes */
    int tif_warn_about_unknown_tags;
};

struct TIFFOpenOptions
{
    TIFFErrorHandlerExtR errorhandler; /* may be NULL */
    void *errorhandler_user_data;      /* may be NULL */
    TIFFErrorHandlerExtR warnhandler;  /* may be NULL */
    void *warnhandler_user_data;       /* may be NULL */
    tmsize_t max_single_mem_alloc;     /* in bytes. 0 for unlimited */
    tmsize_t max_cumulated_mem_alloc;  /* in bytes. 0 for unlimited */
    int warn_about_unknown_tags;
};

#define isPseudoTag(t) (t > 0xffff) /* is tag value normal or pseudo */

#define isTiled(tif) (((tif)->tif_flags & TIFF_ISTILED) != 0)
#define isMapped(tif) (((tif)->tif_flags & TIFF_MAPPED) != 0)
#define isFillOrder(tif, o) (((tif)->tif_flags & (o)) != 0)
#define isUpSampled(tif) (((tif)->tif_flags & TIFF_UPSAMPLED) != 0)
#define TIFFReadFile(tif, buf, size)                                           \
    ((*(tif)->tif_readproc)((tif)->tif_clientdata, (buf), (size)))
#define TIFFWriteFile(tif, buf, size)                                          \
    ((*(tif)->tif_writeproc)((tif)->tif_clientdata, (buf), (size)))
#define TIFFSeekFile(tif, off, whence)                                         \
    ((*(tif)->tif_seekproc)((tif)->tif_clientdata, (off), (whence)))
#define TIFFCloseFile(tif) ((*(tif)->tif_closeproc)((tif)->tif_clientdata))
#define TIFFGetFileSize(tif) ((*(tif)->tif_sizeproc)((tif)->tif_clientdata))
#define TIFFMapFileContents(tif, paddr, psize)                                 \
    ((*(tif)->tif_mapproc)((tif)->tif_clientdata, (paddr), (psize)))
#define TIFFUnmapFileContents(tif, addr, size)                                 \
    ((*(tif)->tif_unmapproc)((tif)->tif_clientdata, (addr), (size)))

/*
 * Default Read/Seek/Write definitions.
 */
#ifndef ReadOK
#define ReadOK(tif, buf, size) (TIFFReadFile((tif), (buf), (size)) == (size))
#endif
#ifndef SeekOK
#define SeekOK(tif, off) _TIFFSeekOK(tif, off)
#endif
#ifndef WriteOK
#define WriteOK(tif, buf, size) (TIFFWriteFile((tif), (buf), (size)) == (size))
#endif

/* NB: the uint32_t casts are to silence certain ANSI-C compilers */
#define TIFFhowmany_32(x, y)                                                   \
    (((uint32_t)x < (0xffffffff - (uint32_t)(y - 1)))                          \
         ? ((((uint32_t)(x)) + (((uint32_t)(y)) - 1)) / ((uint32_t)(y)))       \
         : 0U)
/* Variant of TIFFhowmany_32() that doesn't return 0 if x close to MAXUINT. */
/* Caution: TIFFhowmany_32_maxuint_compat(x,y)*y might overflow */
#define TIFFhowmany_32_maxuint_compat(x, y)                                    \
    (((uint32_t)(x) / (uint32_t)(y)) +                                         \
     ((((uint32_t)(x) % (uint32_t)(y)) != 0) ? 1 : 0))
#define TIFFhowmany8_32(x)                                                     \
    (((x)&0x07) ? ((uint32_t)(x) >> 3) + 1 : (uint32_t)(x) >> 3)
#define TIFFroundup_32(x, y) (TIFFhowmany_32(x, y) * (y))
#define TIFFhowmany_64(x, y)                                                   \
    ((((uint64_t)(x)) + (((uint64_t)(y)) - 1)) / ((uint64_t)(y)))
#define TIFFhowmany8_64(x)                                                     \
    (((x)&0x07) ? ((uint64_t)(x) >> 3) + 1 : (uint64_t)(x) >> 3)
#define TIFFroundup_64(x, y) (TIFFhowmany_64(x, y) * (y))

/* Safe multiply which returns zero if there is an *unsigned* integer overflow.
 * This macro is not safe for *signed* integer types */
#define TIFFSafeMultiply(t, v, m)                                              \
    ((((t)(m) != (t)0) && (((t)(((v) * (m)) / (m))) == (t)(v)))                \
         ? (t)((v) * (m))                                                      \
         : (t)0)

#define TIFFmax(A, B) ((A) > (B) ? (A) : (B))
#define TIFFmin(A, B) ((A) < (B) ? (A) : (B))

#define TIFFArrayCount(a) (sizeof(a) / sizeof((a)[0]))

/*
  Support for large files.

  Windows read/write APIs support only 'unsigned int' rather than 'size_t'.
  Windows off_t is only 32-bit, even in 64-bit builds.
*/
#if defined(HAVE_FSEEKO)
/*
  Use fseeko() and ftello() if they are available since they use
  'off_t' rather than 'long'.  It is wrong to use fseeko() and
  ftello() only on systems with special LFS support since some systems
  (e.g. FreeBSD) support a 64-bit off_t by default.

  For MinGW, __MSVCRT_VERSION__ must be at least 0x800 to expose these
  interfaces. The MinGW compiler must support the requested version.  MinGW
  does not distribute the CRT (it is supplied by Microsoft) so the correct CRT
  must be available on the target computer in order for the program to run.
*/
#if defined(HAVE_FSEEKO)
#define fseek(stream, offset, whence) fseeko(stream, offset, whence)
#define ftell(stream, offset, whence) ftello(stream, offset, whence)
#endif
#endif
#if defined(_WIN32) &&                                                         \
    !(defined(__MSVCRT_VERSION__) && __MSVCRT_VERSION__ < 0x800)
typedef unsigned int TIFFIOSize_t;
#define _TIFF_lseek_f(fildes, offset, whence)                                  \
    _lseeki64(fildes, /* __int64 */ offset, whence)
/* #define _TIFF_tell_f(fildes) /\* __int64 *\/ _telli64(fildes) */
#define _TIFF_fseek_f(stream, offset, whence)                                  \
    _fseeki64(stream, /* __int64 */ offset, whence)
#define _TIFF_fstat_f(fildes, stat_buff)                                       \
    _fstati64(fildes, /* struct _stati64 */ stat_buff)
/* #define _TIFF_ftell_f(stream) /\* __int64 *\/ _ftelli64(stream) */
/* #define _TIFF_stat_f(path,stat_buff) _stati64(path,/\* struct _stati64 *\/
 * stat_buff) */
#define _TIFF_stat_s struct _stati64
#define _TIFF_off_t __int64
#else
typedef size_t TIFFIOSize_t;
#define _TIFF_lseek_f(fildes, offset, whence) lseek(fildes, offset, whence)
/* #define _TIFF_tell_f(fildes) (_TIFF_lseek_f(fildes,0,SEEK_CUR)) */
#define _TIFF_fseek_f(stream, offset, whence) fseek(stream, offset, whence)
#define _TIFF_fstat_f(fildes, stat_buff) fstat(fildes, stat_buff)
/* #define _TIFF_ftell_f(stream) ftell(stream) */
/* #define _TIFF_stat_f(path,stat_buff) stat(path,stat_buff) */
#define _TIFF_stat_s struct stat
#define _TIFF_off_t off_t
#endif

#if defined(__has_attribute) && defined(__clang__)
#if __has_attribute(no_sanitize)
#define TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW                                  \
    __attribute__((no_sanitize("unsigned-integer-overflow")))
#else
#define TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
#endif
#else
#define TIFF_NOSANITIZE_UNSIGNED_INT_OVERFLOW
#endif

#if defined(__cplusplus)
extern "C"
{
#endif
    extern int _tiffDummyMapProc(thandle_t fd, void **pbase, toff_t *psize);
    extern void _tiffDummyUnmapProc(thandle_t fd, void *base, toff_t size);
    extern int _TIFFgetMode(TIFFOpenOptions *opts, thandle_t clientdata,
                            const char *mode, const char *module);
    extern int _TIFFNoRowEncode(TIFF *tif, uint8_t *pp, tmsize_t cc,
                                uint16_t s);
    extern int _TIFFNoStripEncode(TIFF *tif, uint8_t *pp, tmsize_t cc,
                                  uint16_t s);
    extern int _TIFFNoTileEncode(TIFF *, uint8_t *pp, tmsize_t cc, uint16_t s);
    extern int _TIFFNoRowDecode(TIFF *tif, uint8_t *pp, tmsize_t cc,
                                uint16_t s);
    extern int _TIFFNoStripDecode(TIFF *tif, uint8_t *pp, tmsize_t cc,
                                  uint16_t s);
    extern int _TIFFNoTileDecode(TIFF *, uint8_t *pp, tmsize_t cc, uint16_t s);
    extern void _TIFFNoPostDecode(TIFF *tif, uint8_t *buf, tmsize_t cc);
    extern int _TIFFNoPreCode(TIFF *tif, uint16_t s);
    extern int _TIFFNoSeek(TIFF *tif, uint32_t off);
    extern void _TIFFSwab16BitData(TIFF *tif, uint8_t *buf, tmsize_t cc);
    extern void _TIFFSwab24BitData(TIFF *tif, uint8_t *buf, tmsize_t cc);
    extern void _TIFFSwab32BitData(TIFF *tif, uint8_t *buf, tmsize_t cc);
    extern void _TIFFSwab64BitData(TIFF *tif, uint8_t *buf, tmsize_t cc);
    extern int TIFFFlushData1(TIFF *tif);
    extern int TIFFDefaultDirectory(TIFF *tif);
    extern void _TIFFSetDefaultCompressionState(TIFF *tif);
    extern int _TIFFRewriteField(TIFF *, uint16_t, TIFFDataType, tmsize_t,
                                 void *);
    extern int TIFFSetCompressionScheme(TIFF *tif, int scheme);
    extern int TIFFSetDefaultCompressionState(TIFF *tif);
    extern uint32_t _TIFFDefaultStripSize(TIFF *tif, uint32_t s);
    extern void _TIFFDefaultTileSize(TIFF *tif, uint32_t *tw, uint32_t *th);

    extern void _TIFFsetByteArray(void **, const void *, uint32_t);
    extern void _TIFFsetByteArrayExt(TIFF *, void **, const void *, uint32_t);
    extern void _TIFFsetShortArray(uint16_t **, const uint16_t *, uint32_t);
    extern void _TIFFsetShortArrayExt(TIFF *, uint16_t **, const uint16_t *,
                                      uint32_t);
    extern void _TIFFsetLongArray(uint32_t **, const uint32_t *, uint32_t);
    extern void _TIFFsetLongArrayExt(TIFF *, uint32_t **, const uint32_t *,
                                     uint32_t);
    extern void _TIFFsetFloatArray(float **, const float *, uint32_t);
    extern void _TIFFsetFloatArrayExt(TIFF *, float **, const float *,
                                      uint32_t);
    extern void _TIFFsetDoubleArray(double **, const double *, uint32_t);
    extern void _TIFFsetDoubleArrayExt(TIFF *, double **, const double *,
                                       uint32_t);

    extern void _TIFFprintAscii(FILE *, const char *);
    extern void _TIFFprintAsciiTag(FILE *, const char *, const char *);

    extern TIFFErrorHandler _TIFFwarningHandler;
    extern TIFFErrorHandler _TIFFerrorHandler;
    extern TIFFErrorHandlerExt _TIFFwarningHandlerExt;
    extern TIFFErrorHandlerExt _TIFFerrorHandlerExt;
    void _TIFFErrorEarly(TIFFOpenOptions *opts, thandle_t clientdata,
                         const char *module, const char *fmt, ...)
        TIFF_ATTRIBUTE((__format__(__printf__, 4, 5)));

    extern uint32_t _TIFFMultiply32(TIFF *, uint32_t, uint32_t, const char *);
    extern uint64_t _TIFFMultiply64(TIFF *, uint64_t, uint64_t, const char *);
    extern tmsize_t _TIFFMultiplySSize(TIFF *, tmsize_t, tmsize_t,
                                       const char *);
    extern tmsize_t _TIFFCastUInt64ToSSize(TIFF *, uint64_t, const char *);
    extern void *_TIFFCheckMalloc(TIFF *, tmsize_t, tmsize_t, const char *);
    extern void *_TIFFCheckRealloc(TIFF *, void *, tmsize_t, tmsize_t,
                                   const char *);

    extern float _TIFFClampDoubleToFloat(double);
    extern uint32_t _TIFFClampDoubleToUInt32(double);

    extern void _TIFFCleanupIFDOffsetAndNumberMaps(TIFF *tif);

    extern tmsize_t _TIFFReadEncodedStripAndAllocBuffer(TIFF *tif,
                                                        uint32_t strip,
                                                        void **buf,
                                                        tmsize_t bufsizetoalloc,
                                                        tmsize_t size_to_read);
    extern tmsize_t _TIFFReadEncodedTileAndAllocBuffer(TIFF *tif, uint32_t tile,
                                                       void **buf,
                                                       tmsize_t bufsizetoalloc,
                                                       tmsize_t size_to_read);
    extern tmsize_t _TIFFReadTileAndAllocBuffer(TIFF *tif, void **buf,
                                                tmsize_t bufsizetoalloc,
                                                uint32_t x, uint32_t y,
                                                uint32_t z, uint16_t s);
    extern int _TIFFSeekOK(TIFF *tif, toff_t off);

    extern int TIFFInitDumpMode(TIFF *, int);
#ifdef PACKBITS_SUPPORT
    extern int TIFFInitPackBits(TIFF *, int);
#endif
#ifdef CCITT_SUPPORT
    extern int TIFFInitCCITTRLE(TIFF *, int), TIFFInitCCITTRLEW(TIFF *, int);
    extern int TIFFInitCCITTFax3(TIFF *, int), TIFFInitCCITTFax4(TIFF *, int);
#endif
#ifdef THUNDER_SUPPORT
    extern int TIFFInitThunderScan(TIFF *, int);
#endif
#ifdef NEXT_SUPPORT
    extern int TIFFInitNeXT(TIFF *, int);
#endif
#ifdef LZW_SUPPORT
    extern int TIFFInitLZW(TIFF *, int);
#endif
#ifdef OJPEG_SUPPORT
    extern int TIFFInitOJPEG(TIFF *, int);
#endif
#ifdef JPEG_SUPPORT
    extern int TIFFInitJPEG(TIFF *, int);
    extern int TIFFJPEGIsFullStripRequired(TIFF *);
#endif
#ifdef JBIG_SUPPORT
    extern int TIFFInitJBIG(TIFF *, int);
#endif
#ifdef ZIP_SUPPORT
    extern int TIFFInitZIP(TIFF *, int);
#endif
#ifdef PIXARLOG_SUPPORT
    extern int TIFFInitPixarLog(TIFF *, int);
#endif
#ifdef LOGLUV_SUPPORT
    extern int TIFFInitSGILog(TIFF *, int);
#endif
#ifdef LERC_SUPPORT
    extern int TIFFInitLERC(TIFF *tif, int);
#endif
#ifdef LZMA_SUPPORT
    extern int TIFFInitLZMA(TIFF *, int);
#endif
#ifdef ZSTD_SUPPORT
    extern int TIFFInitZSTD(TIFF *, int);
#endif
#ifdef WEBP_SUPPORT
    extern int TIFFInitWebP(TIFF *, int);
#endif
    extern const TIFFCodec _TIFFBuiltinCODECS[];
    extern void TIFFCIELab16ToXYZ(TIFFCIELabToRGB *, uint32_t l, int32_t a,
                                  int32_t b, float *, float *, float *);

    extern void *_TIFFmallocExt(TIFF *tif, tmsize_t s);
    extern void *_TIFFcallocExt(TIFF *tif, tmsize_t nmemb, tmsize_t siz);
    extern void *_TIFFreallocExt(TIFF *tif, void *p, tmsize_t s);
    extern void _TIFFfreeExt(TIFF *tif, void *p);

#if defined(__cplusplus)
}
#endif
#endif /* _TIFFIOP_ */
