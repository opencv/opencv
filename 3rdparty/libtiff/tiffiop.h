/* $Id: tiffiop.h,v 1.84 2012-05-30 01:50:17 fwarmerdam Exp $ */

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
#define	_TIFFIOP_
/*
 * ``Library-private'' definitions.
 */

#include "tif_config.h"

#ifdef HAVE_FCNTL_H
# include <fcntl.h>
#endif

#ifdef HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif

#ifdef HAVE_STRING_H
# include <string.h>
#endif

#ifdef HAVE_ASSERT_H
# include <assert.h>
#else
# define assert(x)
#endif

#ifdef HAVE_SEARCH_H
# include <search.h>
#else
extern void *lfind(const void *, const void *, size_t *, size_t,
           int (*)(const void *, const void *));
#endif

#include "tiffio.h"

#include "tif_dir.h"

#ifndef STRIP_SIZE_DEFAULT
# define STRIP_SIZE_DEFAULT 8192
#endif

#define    streq(a,b)      (strcmp(a,b) == 0)

#ifndef TRUE
#define	TRUE	1
#define	FALSE	0
#endif

typedef struct client_info {
    struct client_info *next;
    void *data;
    char *name;
} TIFFClientInfoLink;

/*
 * Typedefs for ``method pointers'' used internally.
 * these are depriciated and provided only for backwards compatibility
 */
typedef unsigned char tidataval_t;    /* internal image data value type */
typedef tidataval_t* tidata_t;        /* reference to internal image data */

typedef void (*TIFFVoidMethod)(TIFF*);
typedef int (*TIFFBoolMethod)(TIFF*);
typedef int (*TIFFPreMethod)(TIFF*, uint16);
typedef int (*TIFFCodeMethod)(TIFF* tif, uint8* buf, tmsize_t size, uint16 sample);
typedef int (*TIFFSeekMethod)(TIFF*, uint32);
typedef void (*TIFFPostMethod)(TIFF* tif, uint8* buf, tmsize_t size);
typedef uint32 (*TIFFStripMethod)(TIFF*, uint32);
typedef void (*TIFFTileMethod)(TIFF*, uint32*, uint32*);

struct tiff {
    char*                tif_name;         /* name of open file */
    int                  tif_fd;           /* open file descriptor */
    int                  tif_mode;         /* open mode (O_*) */
    uint32               tif_flags;
    #define TIFF_FILLORDER   0x00003 /* natural bit fill order for machine */
    #define TIFF_DIRTYHEADER 0x00004 /* header must be written on close */
    #define TIFF_DIRTYDIRECT 0x00008 /* current directory must be written */
    #define TIFF_BUFFERSETUP 0x00010 /* data buffers setup */
    #define TIFF_CODERSETUP  0x00020 /* encoder/decoder setup done */
    #define TIFF_BEENWRITING 0x00040 /* written 1+ scanlines to file */
    #define TIFF_SWAB        0x00080 /* byte swap file information */
    #define TIFF_NOBITREV    0x00100 /* inhibit bit reversal logic */
    #define TIFF_MYBUFFER    0x00200 /* my raw data buffer; free on close */
    #define TIFF_ISTILED     0x00400 /* file is tile, not strip- based */
    #define TIFF_MAPPED      0x00800 /* file is mapped into memory */
    #define TIFF_POSTENCODE  0x01000 /* need call to postencode routine */
    #define TIFF_INSUBIFD    0x02000 /* currently writing a subifd */
    #define TIFF_UPSAMPLED   0x04000 /* library is doing data up-sampling */
    #define TIFF_STRIPCHOP   0x08000 /* enable strip chopping support */
    #define TIFF_HEADERONLY  0x10000 /* read header only, do not process the first directory */
    #define TIFF_NOREADRAW   0x20000 /* skip reading of raw uncompressed image data */
    #define TIFF_INCUSTOMIFD 0x40000 /* currently writing a custom IFD */
    #define TIFF_BIGTIFF     0x80000 /* read/write bigtiff */
        #define TIFF_BUF4WRITE  0x100000 /* rawcc bytes are for writing */
        #define TIFF_DIRTYSTRIP 0x200000 /* stripoffsets/stripbytecount dirty*/
        #define TIFF_PERSAMPLE  0x400000 /* get/set per sample tags as arrays */
        #define TIFF_BUFFERMMAP 0x800000 /* read buffer (tif_rawdata) points into mmap() memory */
    uint64               tif_diroff;       /* file offset of current directory */
    uint64               tif_nextdiroff;   /* file offset of following directory */
    uint64*              tif_dirlist;      /* list of offsets to already seen directories to prevent IFD looping */
    uint16               tif_dirlistsize;  /* number of entires in offset list */
    uint16               tif_dirnumber;    /* number of already seen directories */
    TIFFDirectory        tif_dir;          /* internal rep of current directory */
    TIFFDirectory        tif_customdir;    /* custom IFDs are separated from the main ones */
    union {
        TIFFHeaderCommon common;
        TIFFHeaderClassic classic;
        TIFFHeaderBig big;
    } tif_header;
    uint16               tif_header_size;  /* file's header block and its length */
    uint32               tif_row;          /* current scanline */
    uint16               tif_curdir;       /* current directory (index) */
    uint32               tif_curstrip;     /* current strip for read/write */
    uint64               tif_curoff;       /* current offset for read/write */
    uint64               tif_dataoff;      /* current offset for writing dir */
    /* SubIFD support */
    uint16               tif_nsubifd;      /* remaining subifds to write */
    uint64               tif_subifdoff;    /* offset for patching SubIFD link */
    /* tiling support */
    uint32               tif_col;          /* current column (offset by row too) */
    uint32               tif_curtile;      /* current tile for read/write */
    tmsize_t             tif_tilesize;     /* # of bytes in a tile */
    /* compression scheme hooks */
    int                  tif_decodestatus;
    TIFFBoolMethod       tif_fixuptags;    /* called in TIFFReadDirectory */
    TIFFBoolMethod       tif_setupdecode;  /* called once before predecode */
    TIFFPreMethod        tif_predecode;    /* pre- row/strip/tile decoding */
    TIFFBoolMethod       tif_setupencode;  /* called once before preencode */
    int                  tif_encodestatus;
    TIFFPreMethod        tif_preencode;    /* pre- row/strip/tile encoding */
    TIFFBoolMethod       tif_postencode;   /* post- row/strip/tile encoding */
    TIFFCodeMethod       tif_decoderow;    /* scanline decoding routine */
    TIFFCodeMethod       tif_encoderow;    /* scanline encoding routine */
    TIFFCodeMethod       tif_decodestrip;  /* strip decoding routine */
    TIFFCodeMethod       tif_encodestrip;  /* strip encoding routine */
    TIFFCodeMethod       tif_decodetile;   /* tile decoding routine */
    TIFFCodeMethod       tif_encodetile;   /* tile encoding routine */
    TIFFVoidMethod       tif_close;        /* cleanup-on-close routine */
    TIFFSeekMethod       tif_seek;         /* position within a strip routine */
    TIFFVoidMethod       tif_cleanup;      /* cleanup state routine */
    TIFFStripMethod      tif_defstripsize; /* calculate/constrain strip size */
    TIFFTileMethod       tif_deftilesize;  /* calculate/constrain tile size */
    uint8*               tif_data;         /* compression scheme private data */
    /* input/output buffering */
    tmsize_t             tif_scanlinesize; /* # of bytes in a scanline */
    tmsize_t             tif_scanlineskew; /* scanline skew for reading strips */
    uint8*               tif_rawdata;      /* raw data buffer */
    tmsize_t             tif_rawdatasize;  /* # of bytes in raw data buffer */
        tmsize_t             tif_rawdataoff;   /* rawdata offset within strip */
        tmsize_t             tif_rawdataloaded;/* amount of data in rawdata */
    uint8*               tif_rawcp;        /* current spot in raw buffer */
    tmsize_t             tif_rawcc;        /* bytes unread from raw buffer */
    /* memory-mapped file support */
    uint8*               tif_base;         /* base of mapped file */
    tmsize_t             tif_size;         /* size of mapped file region (bytes, thus tmsize_t) */
    TIFFMapFileProc      tif_mapproc;      /* map file method */
    TIFFUnmapFileProc    tif_unmapproc;    /* unmap file method */
    /* input/output callback methods */
    thandle_t            tif_clientdata;   /* callback parameter */
    TIFFReadWriteProc    tif_readproc;     /* read method */
    TIFFReadWriteProc    tif_writeproc;    /* write method */
    TIFFSeekProc         tif_seekproc;     /* lseek method */
    TIFFCloseProc        tif_closeproc;    /* close method */
    TIFFSizeProc         tif_sizeproc;     /* filesize method */
    /* post-decoding support */
    TIFFPostMethod       tif_postdecode;   /* post decoding routine */
    /* tag support */
    TIFFField**          tif_fields;       /* sorted table of registered tags */
    size_t               tif_nfields;      /* # entries in registered tag table */
    const TIFFField*     tif_foundfield;   /* cached pointer to already found tag */
    TIFFTagMethods       tif_tagmethods;   /* tag get/set/print routines */
    TIFFClientInfoLink*  tif_clientinfo;   /* extra client information. */
    /* Backward compatibility stuff. We need these two fields for
     * setting up an old tag extension scheme. */
    TIFFFieldArray*      tif_fieldscompat;
    size_t               tif_nfieldscompat;
};

#define isPseudoTag(t) (t > 0xffff)            /* is tag value normal or pseudo */

#define isTiled(tif) (((tif)->tif_flags & TIFF_ISTILED) != 0)
#define isMapped(tif) (((tif)->tif_flags & TIFF_MAPPED) != 0)
#define isFillOrder(tif, o) (((tif)->tif_flags & (o)) != 0)
#define isUpSampled(tif) (((tif)->tif_flags & TIFF_UPSAMPLED) != 0)
#define TIFFReadFile(tif, buf, size) \
    ((*(tif)->tif_readproc)((tif)->tif_clientdata,(buf),(size)))
#define TIFFWriteFile(tif, buf, size) \
    ((*(tif)->tif_writeproc)((tif)->tif_clientdata,(buf),(size)))
#define TIFFSeekFile(tif, off, whence) \
    ((*(tif)->tif_seekproc)((tif)->tif_clientdata,(off),(whence)))
#define TIFFCloseFile(tif) \
    ((*(tif)->tif_closeproc)((tif)->tif_clientdata))
#define TIFFGetFileSize(tif) \
    ((*(tif)->tif_sizeproc)((tif)->tif_clientdata))
#define TIFFMapFileContents(tif, paddr, psize) \
    ((*(tif)->tif_mapproc)((tif)->tif_clientdata,(paddr),(psize)))
#define TIFFUnmapFileContents(tif, addr, size) \
    ((*(tif)->tif_unmapproc)((tif)->tif_clientdata,(addr),(size)))

/*
 * Default Read/Seek/Write definitions.
 */
#ifndef ReadOK
#define ReadOK(tif, buf, size) \
    (TIFFReadFile((tif),(buf),(size))==(size))
#endif
#ifndef SeekOK
#define SeekOK(tif, off) \
    (TIFFSeekFile((tif),(off),SEEK_SET)==(off))
#endif
#ifndef WriteOK
#define WriteOK(tif, buf, size) \
    (TIFFWriteFile((tif),(buf),(size))==(size))
#endif

/* NB: the uint32 casts are to silence certain ANSI-C compilers */
#define TIFFhowmany_32(x, y) (((uint32)x < (0xffffffff - (uint32)(y-1))) ? \
               ((((uint32)(x))+(((uint32)(y))-1))/((uint32)(y))) : \
               0U)
#define TIFFhowmany8_32(x) (((x)&0x07)?((uint32)(x)>>3)+1:(uint32)(x)>>3)
#define TIFFroundup_32(x, y) (TIFFhowmany_32(x,y)*(y))
#define TIFFhowmany_64(x, y) ((((uint64)(x))+(((uint64)(y))-1))/((uint64)(y)))
#define TIFFhowmany8_64(x) (((x)&0x07)?((uint64)(x)>>3)+1:(uint64)(x)>>3)
#define TIFFroundup_64(x, y) (TIFFhowmany_64(x,y)*(y))

/* Safe multiply which returns zero if there is an integer overflow */
#define TIFFSafeMultiply(t,v,m) ((((t)(m) != (t)0) && (((t)(((v)*(m))/(m))) == (t)(v))) ? (t)((v)*(m)) : (t)0)

#define TIFFmax(A,B) ((A)>(B)?(A):(B))
#define TIFFmin(A,B) ((A)<(B)?(A):(B))

#define TIFFArrayCount(a) (sizeof (a) / sizeof ((a)[0]))

#if defined(__cplusplus)
extern "C" {
#endif
extern int _TIFFgetMode(const char* mode, const char* module);
extern int _TIFFNoRowEncode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s);
extern int _TIFFNoStripEncode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s);
extern int _TIFFNoTileEncode(TIFF*, uint8* pp, tmsize_t cc, uint16 s);
extern int _TIFFNoRowDecode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s);
extern int _TIFFNoStripDecode(TIFF* tif, uint8* pp, tmsize_t cc, uint16 s);
extern int _TIFFNoTileDecode(TIFF*, uint8* pp, tmsize_t cc, uint16 s);
extern void _TIFFNoPostDecode(TIFF* tif, uint8* buf, tmsize_t cc);
extern int _TIFFNoPreCode(TIFF* tif, uint16 s);
extern int _TIFFNoSeek(TIFF* tif, uint32 off);
extern void _TIFFSwab16BitData(TIFF* tif, uint8* buf, tmsize_t cc);
extern void _TIFFSwab24BitData(TIFF* tif, uint8* buf, tmsize_t cc);
extern void _TIFFSwab32BitData(TIFF* tif, uint8* buf, tmsize_t cc);
extern void _TIFFSwab64BitData(TIFF* tif, uint8* buf, tmsize_t cc);
extern int TIFFFlushData1(TIFF* tif);
extern int TIFFDefaultDirectory(TIFF* tif);
extern void _TIFFSetDefaultCompressionState(TIFF* tif);
extern int _TIFFRewriteField(TIFF *, uint16, TIFFDataType, tmsize_t, void *);
extern int TIFFSetCompressionScheme(TIFF* tif, int scheme);
extern int TIFFSetDefaultCompressionState(TIFF* tif);
extern uint32 _TIFFDefaultStripSize(TIFF* tif, uint32 s);
extern void _TIFFDefaultTileSize(TIFF* tif, uint32* tw, uint32* th);
extern int _TIFFDataSize(TIFFDataType type);

extern void _TIFFsetByteArray(void**, void*, uint32);
extern void _TIFFsetString(char**, char*);
extern void _TIFFsetShortArray(uint16**, uint16*, uint32);
extern void _TIFFsetLongArray(uint32**, uint32*, uint32);
extern void _TIFFsetFloatArray(float**, float*, uint32);
extern void _TIFFsetDoubleArray(double**, double*, uint32);

extern void _TIFFprintAscii(FILE*, const char*);
extern void _TIFFprintAsciiTag(FILE*, const char*, const char*);

extern TIFFErrorHandler _TIFFwarningHandler;
extern TIFFErrorHandler _TIFFerrorHandler;
extern TIFFErrorHandlerExt _TIFFwarningHandlerExt;
extern TIFFErrorHandlerExt _TIFFerrorHandlerExt;

extern uint32 _TIFFMultiply32(TIFF*, uint32, uint32, const char*);
extern uint64 _TIFFMultiply64(TIFF*, uint64, uint64, const char*);
extern void* _TIFFCheckMalloc(TIFF*, tmsize_t, tmsize_t, const char*);
extern void* _TIFFCheckRealloc(TIFF*, void*, tmsize_t, tmsize_t, const char*);

extern double _TIFFUInt64ToDouble(uint64);
extern float _TIFFUInt64ToFloat(uint64);

extern int TIFFInitDumpMode(TIFF*, int);
#ifdef PACKBITS_SUPPORT
extern int TIFFInitPackBits(TIFF*, int);
#endif
#ifdef CCITT_SUPPORT
extern int TIFFInitCCITTRLE(TIFF*, int), TIFFInitCCITTRLEW(TIFF*, int);
extern int TIFFInitCCITTFax3(TIFF*, int), TIFFInitCCITTFax4(TIFF*, int);
#endif
#ifdef THUNDER_SUPPORT
extern int TIFFInitThunderScan(TIFF*, int);
#endif
#ifdef NEXT_SUPPORT
extern int TIFFInitNeXT(TIFF*, int);
#endif
#ifdef LZW_SUPPORT
extern int TIFFInitLZW(TIFF*, int);
#endif
#ifdef OJPEG_SUPPORT
extern int TIFFInitOJPEG(TIFF*, int);
#endif
#ifdef JPEG_SUPPORT
extern int TIFFInitJPEG(TIFF*, int);
#endif
#ifdef JBIG_SUPPORT
extern int TIFFInitJBIG(TIFF*, int);
#endif
#ifdef ZIP_SUPPORT
extern int TIFFInitZIP(TIFF*, int);
#endif
#ifdef PIXARLOG_SUPPORT
extern int TIFFInitPixarLog(TIFF*, int);
#endif
#ifdef LOGLUV_SUPPORT
extern int TIFFInitSGILog(TIFF*, int);
#endif
#ifdef LZMA_SUPPORT
extern int TIFFInitLZMA(TIFF*, int);
#endif
#ifdef VMS
extern const TIFFCodec _TIFFBuiltinCODECS[];
#else
extern TIFFCodec _TIFFBuiltinCODECS[];
#endif

#if defined(__cplusplus)
}
#endif
#endif /* _TIFFIOP_ */

/* vim: set ts=8 sts=8 sw=8 noet: */
/*
 * Local Variables:
 * mode: c
 * c-basic-offset: 8
 * fill-column: 78
 * End:
 */
