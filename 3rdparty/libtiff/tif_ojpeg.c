/* $Id: tif_ojpeg.c,v 1.1 2005-06-17 13:54:52 vp153 Exp $ */

#include "tiffiop.h"
#ifdef OJPEG_SUPPORT

/* JPEG Compression support, as per the original TIFF 6.0 specification.

   WARNING: KLUDGE ALERT!  The type of JPEG encapsulation defined by the TIFF
                           Version 6.0 specification is now totally obsolete and
   deprecated for new applications and images.  This file is an unsupported hack
   that was created solely in order to read (but NOT write!) a few old,
   unconverted images still present on some users' computer systems.  The code
   isn't pretty or robust, and it won't read every "old format" JPEG-in-TIFF
   file (see Samuel Leffler's draft "TIFF Technical Note No. 2" for a long and
   incomplete list of known problems), but it seems to work well enough in the
   few cases of practical interest to the author; so, "caveat emptor"!  This
   file should NEVER be enhanced to write new images using anything other than
   the latest approved JPEG-in-TIFF encapsulation method, implemented by the
   "tif_jpeg.c" file elsewhere in this library.

   This file interfaces with Release 6B of the JPEG Library written by theu
   Independent JPEG Group, which you can find on the Internet at:
   ftp://ftp.uu.net:/graphics/jpeg/.

   The "C" Preprocessor macros, "[CD]_LOSSLESS_SUPPORTED", are defined by your
   JPEG Library Version 6B only if you have applied a (massive!) patch by Ken
   Murchison of Oceana Matrix Ltd. <ken@oceana.com> to support lossless Huffman
   encoding (TIFF "JPEGProc" tag value = 14).  This patch can be found on the
   Internet at: ftp://ftp.oceana.com/pub/ljpeg-6b.tar.gz.

   Some old files produced by the Wang Imaging application for Microsoft Windows
   apparently can be decoded only with a special patch to the JPEG Library,
   which defines a subroutine named "jpeg_reset_huff_decode()" in its "jdhuff.c"
   module (the "jdshuff.c" module, if Ken Murchison's patch has been applied).
   Unfortunately the patch differs slightly in each case, and some TIFF Library
   have reported problems finding the code, so both versions appear below; you
   should carefully extract and apply only the version that applies to your JPEG
   Library!

   Contributed by Scott Marovich <marovich@hpl.hp.com> with considerable help
   from Charles Auer <Bumble731@msn.com> to unravel the mysteries of image files
   created by the Wang Imaging application for Microsoft Windows.
*/
#if 0  /* Patch for JPEG Library WITHOUT lossless Huffman coding */
*** jdhuff.c.orig	Mon Oct 20 17:51:10 1997
--- jdhuff.c	Sun Nov 11 17:33:58 2001
***************
*** 648,651 ****
--- 648,683 ----
    for (i = 0; i < NUM_HUFF_TBLS; i++) {
      entropy->dc_derived_tbls[i] = entropy->ac_derived_tbls[i] = NULL;
    }
  }
+ 
+ /*
+  * BEWARE OF KLUDGE:  This subroutine is a hack for decoding illegal JPEG-in-
+  *                    TIFF encapsulations produced by Microsoft's Wang Imaging
+  * for Windows application with the public-domain TIFF Library.  Based upon an
+  * examination of selected output files, this program apparently divides a JPEG
+  * bit-stream into consecutive horizontal TIFF "strips", such that the JPEG
+  * encoder's/decoder's DC coefficients for each image component are reset before
+  * each "strip".  Moreover, a "strip" is not necessarily encoded in a multiple
+  * of 8 bits, so one must sometimes discard 1-7 bits at the end of each "strip"
+  * for alignment to the next input-Byte storage boundary.  IJG JPEG Library
+  * decoder state is not normally exposed to client applications, so this sub-
+  * routine provides the TIFF Library with a "hook" to make these corrections.
+  * It should be called after "jpeg_start_decompress()" and before
+  * "jpeg_finish_decompress()", just before decoding each "strip" using
+  * "jpeg_read_raw_data()" or "jpeg_read_scanlines()".
+  *
+  * This kludge is not sanctioned or supported by the Independent JPEG Group, and
+  * future changes to the IJG JPEG Library might invalidate it.  Do not send bug
+  * reports about this code to IJG developers.  Instead, contact the author for
+  * advice: Scott B. Marovich <marovich@hpl.hp.com>, Hewlett-Packard Labs, 6/01.
+  */
+ GLOBAL(void)
+ jpeg_reset_huff_decode (register j_decompress_ptr cinfo)
+ { register huff_entropy_ptr entropy = (huff_entropy_ptr)cinfo->entropy;
+   register int ci = 0;
+ 
+   /* Discard encoded input bits, up to the next Byte boundary */
+   entropy->bitstate.bits_left &= ~7;
+   /* Re-initialize DC predictions to 0 */
+   do entropy->saved.last_dc_val[ci] = 0; while (++ci < cinfo->comps_in_scan);
+ }
#endif /* Patch for JPEG Library WITHOUT lossless Huffman coding */
#if 0  /* Patch for JPEG Library WITH lossless Huffman coding */
*** jdshuff.c.orig	Mon Mar 11 16:44:54 2002
--- jdshuff.c	Mon Mar 11 16:44:54 2002
***************
*** 357,360 ****
--- 357,393 ----
    for (i = 0; i < NUM_HUFF_TBLS; i++) {
      entropy->dc_derived_tbls[i] = entropy->ac_derived_tbls[i] = NULL;
    }
  }
+ 
+ /*
+  * BEWARE OF KLUDGE:  This subroutine is a hack for decoding illegal JPEG-in-
+  *                    TIFF encapsulations produced by Microsoft's Wang Imaging
+  * for Windows application with the public-domain TIFF Library.  Based upon an
+  * examination of selected output files, this program apparently divides a JPEG
+  * bit-stream into consecutive horizontal TIFF "strips", such that the JPEG
+  * encoder's/decoder's DC coefficients for each image component are reset before
+  * each "strip".  Moreover, a "strip" is not necessarily encoded in a multiple
+  * of 8 bits, so one must sometimes discard 1-7 bits at the end of each "strip"
+  * for alignment to the next input-Byte storage boundary.  IJG JPEG Library
+  * decoder state is not normally exposed to client applications, so this sub-
+  * routine provides the TIFF Library with a "hook" to make these corrections.
+  * It should be called after "jpeg_start_decompress()" and before
+  * "jpeg_finish_decompress()", just before decoding each "strip" using
+  * "jpeg_read_raw_data()" or "jpeg_read_scanlines()".
+  *
+  * This kludge is not sanctioned or supported by the Independent JPEG Group, and
+  * future changes to the IJG JPEG Library might invalidate it.  Do not send bug
+  * reports about this code to IJG developers.  Instead, contact the author for
+  * advice: Scott B. Marovich <marovich@hpl.hp.com>, Hewlett-Packard Labs, 6/01.
+  */
+ GLOBAL(void)
+ jpeg_reset_huff_decode (register j_decompress_ptr cinfo)
+ { register shuff_entropy_ptr entropy = (shuff_entropy_ptr)
+                                        ((j_lossy_d_ptr)cinfo->codec)->entropy_private;
+   register int ci = 0;
+ 
+   /* Discard encoded input bits, up to the next Byte boundary */
+   entropy->bitstate.bits_left &= ~7;
+   /* Re-initialize DC predictions to 0 */
+   do entropy->saved.last_dc_val[ci] = 0; while (++ci < cinfo->comps_in_scan);
+ }
#endif /* Patch for JPEG Library WITH lossless Huffman coding */
#include <setjmp.h>
#include <stdio.h>
#ifdef FAR
#undef FAR /* Undefine FAR to avoid conflict with JPEG definition */
#endif
#define JPEG_INTERNALS /* Include "jpegint.h" for "DSTATE_*" symbols */
#define JPEG_CJPEG_DJPEG /* Include all Version 6B+ "jconfig.h" options */
#undef INLINE
#include "jpeglib.h"
#undef JPEG_CJPEG_DJPEG
#undef JPEG_INTERNALS

/* Hack for files produced by Wang Imaging application on Microsoft Windows */
extern void jpeg_reset_huff_decode(j_decompress_ptr);

/* On some machines, it may be worthwhile to use "_setjmp()" or "sigsetjmp()"
   instead of "setjmp()".  These macros make it easier:
*/
#define SETJMP(jbuf)setjmp(jbuf)
#define LONGJMP(jbuf,code)longjmp(jbuf,code)
#define JMP_BUF jmp_buf

#define TIFFTAG_WANG_PAGECONTROL 32934

/* Bit-vector offsets for keeping track of TIFF records that we've parsed. */

#define FIELD_JPEGPROC FIELD_CODEC
#define FIELD_JPEGIFOFFSET (FIELD_CODEC+1)
#define FIELD_JPEGIFBYTECOUNT (FIELD_CODEC+2)
#define FIELD_JPEGRESTARTINTERVAL (FIELD_CODEC+3)
#define FIELD_JPEGTABLES (FIELD_CODEC+4) /* New, post-6.0 JPEG-in-TIFF tag! */
#define FIELD_JPEGLOSSLESSPREDICTORS (FIELD_CODEC+5)
#define FIELD_JPEGPOINTTRANSFORM (FIELD_CODEC+6)
#define FIELD_JPEGQTABLES (FIELD_CODEC+7)
#define FIELD_JPEGDCTABLES (FIELD_CODEC+8)
#define FIELD_JPEGACTABLES (FIELD_CODEC+9)
#define FIELD_WANG_PAGECONTROL (FIELD_CODEC+10)
#define FIELD_JPEGCOLORMODE (FIELD_CODEC+11)

typedef struct jpeg_destination_mgr jpeg_destination_mgr;
typedef struct jpeg_source_mgr jpeg_source_mgr;
typedef struct jpeg_error_mgr jpeg_error_mgr;

/* State variable for each open TIFF file that uses "libjpeg" for JPEG
   decompression.  (Note:  This file should NEVER perform JPEG compression
   except in the manner implemented by the "tif_jpeg.c" file, elsewhere in this
   library; see comments above.)  JPEG Library internal state is recorded in a
   "jpeg_{de}compress_struct", while a "jpeg_common_struct" records a few items
   common to both compression and expansion.  The "cinfo" field containing JPEG
   Library state MUST be the 1st member of our own state variable, so that we
   can safely "cast" pointers back and forth.
*/
typedef struct             /* This module's private, per-image state variable */
  {
    union         /* JPEG Library state variable; this MUST be our 1st field! */
      {
        struct jpeg_compress_struct c;
        struct jpeg_decompress_struct d;
        struct jpeg_common_struct comm;
      } cinfo;
    jpeg_error_mgr err;                         /* JPEG Library error manager */
    JMP_BUF exit_jmpbuf;             /* ...for catching JPEG Library failures */
#   ifdef never

 /* (The following two fields could be a "union", but they're small enough that
    it's not worth the effort.)
 */
    jpeg_destination_mgr dest;             /* Destination for compressed data */
#   endif
    jpeg_source_mgr src;                           /* Source of expanded data */
    JSAMPARRAY ds_buffer[MAX_COMPONENTS]; /* ->Temporary downsampling buffers */
    TIFF *tif;                        /* Reverse pointer, needed by some code */
    TIFFVGetMethod vgetparent;                    /* "Super class" methods... */
    TIFFVSetMethod vsetparent;
    TIFFStripMethod defsparent;
    TIFFTileMethod deftparent;
    void *jpegtables;           /* ->"New" JPEG tables, if we synthesized any */
    uint32 is_WANG,    /* <=> Wang Imaging for Microsoft Windows output file? */
           jpegtables_length;   /* Length of "new" JPEG tables, if they exist */
    tsize_t bytesperline;          /* No. of decompressed Bytes per scan line */
    int jpegquality,                             /* Compression quality level */
        jpegtablesmode,                          /* What to put in JPEGTables */
        samplesperclump,
        scancount;                           /* No. of scan lines accumulated */
    J_COLOR_SPACE photometric;          /* IJG JPEG Library's photometry code */
    unsigned char h_sampling,                          /* Luminance sampling factors */
           v_sampling,
           jpegcolormode;           /* Who performs RGB <-> YCbCr conversion? */
			/* JPEGCOLORMODE_RAW <=> TIFF Library or its client */
			/* JPEGCOLORMODE_RGB <=> JPEG Library               */
    /* These fields are added to support TIFFGetField */
    uint16 jpegproc;
    uint32 jpegifoffset;
    uint32 jpegifbytecount;
    uint32 jpegrestartinterval;
    void* jpeglosslesspredictors;
    uint16 jpeglosslesspredictors_length;
    void* jpegpointtransform;
    uint32 jpegpointtransform_length;
    void* jpegqtables;
    uint32 jpegqtables_length;
    void* jpegdctables;
    uint32 jpegdctables_length;
    void* jpegactables;
    uint32 jpegactables_length;

  } OJPEGState;
#define OJState(tif)((OJPEGState*)(tif)->tif_data)

static const TIFFFieldInfo ojpegFieldInfo[]=/* JPEG-specific TIFF-record tags */
  {

 /* This is the current JPEG-in-TIFF metadata-encapsulation tag, and its
    treatment in this file is idiosyncratic.  It should never appear in a
    "source" image conforming to the TIFF Version 6.0 specification, so we
    arrange to report an error if it appears.  But in order to support possible
    future conversion of "old" JPEG-in-TIFF encapsulations to "new" ones, we
    might wish to synthesize an equivalent value to be returned by the TIFF
    Library's "getfield" method.  So, this table tells the TIFF Library to pass
    these records to us in order to filter them below.
 */
    {
      TIFFTAG_JPEGTABLES            ,TIFF_VARIABLE2,TIFF_VARIABLE2,
      TIFF_UNDEFINED,FIELD_JPEGTABLES            ,FALSE,TRUE ,"JPEGTables"
    },

 /* These tags are defined by the TIFF Version 6.0 specification and are now
    obsolete.  This module reads them from an old "source" image, but it never
    writes them to a new "destination" image.
 */
    {
      TIFFTAG_JPEGPROC              ,1            ,1            ,
      TIFF_SHORT    ,FIELD_JPEGPROC              ,FALSE,FALSE,"JPEGProc"
    },
    {
      TIFFTAG_JPEGIFOFFSET          ,1            ,1            ,
      TIFF_LONG     ,FIELD_JPEGIFOFFSET          ,FALSE,FALSE,"JPEGInterchangeFormat"
    },
    {
      TIFFTAG_JPEGIFBYTECOUNT       ,1            ,1            ,
      TIFF_LONG     ,FIELD_JPEGIFBYTECOUNT       ,FALSE,FALSE,"JPEGInterchangeFormatLength"
    },
    {
      TIFFTAG_JPEGRESTARTINTERVAL   ,1            ,1            ,
      TIFF_SHORT    ,FIELD_JPEGRESTARTINTERVAL   ,FALSE,FALSE,"JPEGRestartInterval"
    },
    {
      TIFFTAG_JPEGLOSSLESSPREDICTORS,TIFF_VARIABLE,TIFF_VARIABLE,
      TIFF_SHORT    ,FIELD_JPEGLOSSLESSPREDICTORS,FALSE,TRUE ,"JPEGLosslessPredictors"
    },
    {
      TIFFTAG_JPEGPOINTTRANSFORM    ,TIFF_VARIABLE,TIFF_VARIABLE,
      TIFF_SHORT    ,FIELD_JPEGPOINTTRANSFORM    ,FALSE,TRUE ,"JPEGPointTransforms"
    },
    {
      TIFFTAG_JPEGQTABLES           ,TIFF_VARIABLE,TIFF_VARIABLE,
      TIFF_LONG     ,FIELD_JPEGQTABLES           ,FALSE,TRUE ,"JPEGQTables"
    },
    {
      TIFFTAG_JPEGDCTABLES          ,TIFF_VARIABLE,TIFF_VARIABLE,
      TIFF_LONG     ,FIELD_JPEGDCTABLES          ,FALSE,TRUE ,"JPEGDCTables"
    },
    {
      TIFFTAG_JPEGACTABLES          ,TIFF_VARIABLE,TIFF_VARIABLE,
      TIFF_LONG     ,FIELD_JPEGACTABLES          ,FALSE,TRUE ,"JPEGACTables"
    },
    {
      TIFFTAG_WANG_PAGECONTROL      ,TIFF_VARIABLE,1            ,
      TIFF_LONG     ,FIELD_WANG_PAGECONTROL      ,FALSE,FALSE,"WANG PageControl"
    },

 /* This is a pseudo tag intended for internal use only by the TIFF Library and
    its clients, which should never appear in an input/output image file.  It
    specifies whether the TIFF Library (or its client) should do YCbCr <-> RGB
    color-space conversion (JPEGCOLORMODE_RAW <=> 0) or whether we should ask
    the JPEG Library to do it (JPEGCOLORMODE_RGB <=> 1).
 */
    {
      TIFFTAG_JPEGCOLORMODE         ,0            ,0            ,
      TIFF_ANY      ,FIELD_PSEUDO                ,FALSE,FALSE,"JPEGColorMode"
    }
  };
static const char JPEGLib_name[]={"JPEG Library"},
                  bad_bps[]={"%u BitsPerSample not allowed for JPEG"},
                  bad_photometry[]={"PhotometricInterpretation %u not allowed for JPEG"},
                  bad_subsampling[]={"invalid YCbCr subsampling factor(s)"},
#                 ifdef never
                  no_write_frac[]={"fractional scan line discarded"},
#                 endif
                  no_read_frac[]={"fractional scan line not read"},
                  no_jtable_space[]={"No space for JPEGTables"};

/* The following diagnostic subroutines interface with and replace default
   subroutines in the JPEG Library.  Our basic strategy is to use "setjmp()"/
   "longjmp()" in order to return control to the TIFF Library when the JPEG
   library detects an error, and to use TIFF Library subroutines for displaying
   diagnostic messages to a client application.
*/
static void
TIFFojpeg_error_exit(register j_common_ptr cinfo)
{
    char buffer[JMSG_LENGTH_MAX];
    int code = cinfo->err->msg_code;

    if (((OJPEGState *)cinfo)->is_WANG) {
	if (code == JERR_SOF_DUPLICATE || code == JERR_SOI_DUPLICATE)
	    return;	    /* ignore it */
    }

    (*cinfo->err->format_message)(cinfo,buffer);
    TIFFError(JPEGLib_name,buffer); /* Display error message */
    jpeg_abort(cinfo); /* Clean up JPEG Library state */
    LONGJMP(((OJPEGState *)cinfo)->exit_jmpbuf,1); /* Return to TIFF client */
}

static void
TIFFojpeg_output_message(register j_common_ptr cinfo)
  { char buffer[JMSG_LENGTH_MAX];

 /* This subroutine is invoked only for warning messages, since the JPEG
    Library's "error_exit" method does its own thing and "trace_level" is never
    set > 0.
 */
    (*cinfo->err->format_message)(cinfo,buffer);
    TIFFWarning(JPEGLib_name,buffer);
  }

/* The following subroutines, which also interface with the JPEG Library, exist
   mainly in limit the side effects of "setjmp()" and convert JPEG normal/error
   conditions into TIFF Library return codes.
*/
#define CALLJPEG(sp,fail,op)(SETJMP((sp)->exit_jmpbuf)?(fail):(op))
#define CALLVJPEG(sp,op)CALLJPEG(sp,0,((op),1))
#ifdef never

static int
TIFFojpeg_create_compress(register OJPEGState *sp)
  {
    sp->cinfo.c.err = jpeg_std_error(&sp->err); /* Initialize error handling */
    sp->err.error_exit = TIFFojpeg_error_exit;
    sp->err.output_message = TIFFojpeg_output_message;
    return CALLVJPEG(sp,jpeg_create_compress(&sp->cinfo.c));
  }

/* The following subroutines comprise a JPEG Library "destination" data manager
   by directing compressed data from the JPEG Library to a TIFF Library output
   buffer.
*/
static void
std_init_destination(register j_compress_ptr cinfo){} /* "Dummy" stub */

static boolean
std_empty_output_buffer(register j_compress_ptr cinfo)
  {
#   define sp ((OJPEGState *)cinfo)
    register TIFF *tif = sp->tif;

    tif->tif_rawcc = tif->tif_rawdatasize; /* Entire buffer has been filled */
    TIFFFlushData1(tif);
    sp->dest.next_output_byte = (JOCTET *)tif->tif_rawdata;
    sp->dest.free_in_buffer = (size_t)tif->tif_rawdatasize;
    return TRUE;
#   undef sp
  }

static void
std_term_destination(register j_compress_ptr cinfo)
  {
#   define sp ((OJPEGState *)cinfo)
    register TIFF *tif = sp->tif;

 /* NB: The TIFF Library does the final buffer flush. */
    tif->tif_rawcp = (tidata_t)sp->dest.next_output_byte;
    tif->tif_rawcc = tif->tif_rawdatasize - (tsize_t)sp->dest.free_in_buffer;
#   undef sp
  }

/* Alternate destination manager to output JPEGTables field: */

static void
tables_init_destination(register j_compress_ptr cinfo)
  {
#   define sp ((OJPEGState *)cinfo)
 /* The "jpegtables_length" field is the allocated buffer size while building */
    sp->dest.next_output_byte = (JOCTET *)sp->jpegtables;
    sp->dest.free_in_buffer = (size_t)sp->jpegtables_length;
#   undef sp
  }

static boolean
tables_empty_output_buffer(register j_compress_ptr cinfo)
  { void *newbuf;
#   define sp ((OJPEGState *)cinfo)

 /* The entire buffer has been filled, so enlarge it by 1000 bytes. */
    if (!( newbuf = _TIFFrealloc( (tdata_t)sp->jpegtables
                                , (tsize_t)(sp->jpegtables_length + 1000)
                                )
         )
       ) ERREXIT1(cinfo,JERR_OUT_OF_MEMORY,100);
    sp->dest.next_output_byte = (JOCTET *)newbuf + sp->jpegtables_length;
    sp->dest.free_in_buffer = (size_t)1000;
    sp->jpegtables = newbuf;
    sp->jpegtables_length += 1000;
    return TRUE;
#   undef sp
  }

static void
tables_term_destination(register j_compress_ptr cinfo)
  {
#   define sp ((OJPEGState *)cinfo)
 /* Set tables length to no. of Bytes actually emitted. */
    sp->jpegtables_length -= sp->dest.free_in_buffer;
#   undef sp
  }

/*ARGSUSED*/ static int
TIFFojpeg_tables_dest(register OJPEGState *sp, TIFF *tif)
  {

 /* Allocate a working buffer for building tables.  The initial size is 1000
    Bytes, which is usually adequate.
 */
    if (sp->jpegtables) _TIFFfree(sp->jpegtables);
    if (!(sp->jpegtables = (void*)
                           _TIFFmalloc((tsize_t)(sp->jpegtables_length = 1000))
         )
       )
      {
        sp->jpegtables_length = 0;
        TIFFError("TIFFojpeg_tables_dest",no_jtable_space);
        return 0;
      };
    sp->cinfo.c.dest = &sp->dest;
    sp->dest.init_destination = tables_init_destination;
    sp->dest.empty_output_buffer = tables_empty_output_buffer;
    sp->dest.term_destination = tables_term_destination;
    return 1;
  }
#else /* well, hardly ever */

static int
_notSupported(register TIFF *tif)
  { const TIFFCodec *c = TIFFFindCODEC(tif->tif_dir.td_compression);

    TIFFError(tif->tif_name,"%s compression not supported",c->name);
    return 0;
  }
#endif /* never */

/* The following subroutines comprise a JPEG Library "source" data manager by
   by directing compressed data to the JPEG Library from a TIFF Library input
   buffer.
*/
static void
std_init_source(register j_decompress_ptr cinfo)
  {
#   define sp ((OJPEGState *)cinfo)
    register TIFF *tif = sp->tif;

    if (sp->src.bytes_in_buffer == 0)
      {
        sp->src.next_input_byte = (const JOCTET *)tif->tif_rawdata;
        sp->src.bytes_in_buffer = (size_t)tif->tif_rawcc;
      };
#   undef sp
  }

static boolean
std_fill_input_buffer(register j_decompress_ptr cinfo)
  { static const JOCTET dummy_EOI[2]={0xFF,JPEG_EOI};
#   define sp ((OJPEGState *)cinfo)

 /* Control should never get here, since an entire strip/tile is read into
    memory before the decompressor is called; thus, data should have been
    supplied by the "init_source" method.  ...But, sometimes things fail.
 */
    WARNMS(cinfo,JWRN_JPEG_EOF);
    sp->src.next_input_byte = dummy_EOI; /* Insert a fake EOI marker */
    sp->src.bytes_in_buffer = sizeof dummy_EOI;
    return TRUE;
#   undef sp
  }

static void
std_skip_input_data(register j_decompress_ptr cinfo, long num_bytes)
  {
#   define sp ((OJPEGState *)cinfo)

    if (num_bytes > 0)
    {
      if (num_bytes > (long)sp->src.bytes_in_buffer) /* oops: buffer overrun */
        (void)std_fill_input_buffer(cinfo);
      else
        {
          sp->src.next_input_byte += (size_t)num_bytes;
          sp->src.bytes_in_buffer -= (size_t)num_bytes;
        }
    }
#   undef sp
  }

/*ARGSUSED*/ static void
std_term_source(register j_decompress_ptr cinfo){} /* "Dummy" stub */

/* Allocate temporary I/O buffers for downsampled data, using values computed in
   "jpeg_start_{de}compress()".  We use the JPEG Library's allocator so that
   buffers will be released automatically when done with a strip/tile.  This is
   also a handy place to compute samplesperclump, bytesperline, etc.
*/
static int
alloc_downsampled_buffers(TIFF *tif,jpeg_component_info *comp_info,
                          int num_components)
  { register OJPEGState *sp = OJState(tif);

    sp->samplesperclump = 0;
    if (num_components > 0)
      { tsize_t size = sp->cinfo.comm.is_decompressor
#                    ifdef D_LOSSLESS_SUPPORTED
                     ? sp->cinfo.d.min_codec_data_unit
#                    else
                     ? DCTSIZE
#                    endif
#                    ifdef C_LOSSLESS_SUPPORTED
                     : sp->cinfo.c.data_unit;
#                    else
                     : DCTSIZE;
#                    endif
        int ci = 0;
        register jpeg_component_info *compptr = comp_info;

        do
          { JSAMPARRAY buf;

            sp->samplesperclump +=
              compptr->h_samp_factor * compptr->v_samp_factor;
#           if defined(C_LOSSLESS_SUPPORTED) || defined(D_LOSSLESS_SUPPORTED)
            if (!(buf = CALLJPEG(sp,0,(*sp->cinfo.comm.mem->alloc_sarray)(&sp->cinfo.comm,JPOOL_IMAGE,compptr->width_in_data_units*size,compptr->v_samp_factor*size))))
#           else
            if (!(buf = CALLJPEG(sp,0,(*sp->cinfo.comm.mem->alloc_sarray)(&sp->cinfo.comm,JPOOL_IMAGE,compptr->width_in_blocks*size,compptr->v_samp_factor*size))))
#           endif
              return 0;
            sp->ds_buffer[ci] = buf;
          }
        while (++compptr,++ci < num_components);
      };
    return 1;
  }
#ifdef never

/* JPEG Encoding begins here. */

/*ARGSUSED*/ static int
OJPEGEncode(register TIFF *tif,tidata_t buf,tsize_t cc,tsample_t s)
  { tsize_t rows;                          /* No. of unprocessed rows in file */
    register OJPEGState *sp = OJState(tif);

 /* Encode a chunk of pixels, where returned data is NOT down-sampled (the
    standard case).  The data is expected to be written in scan-line multiples.
 */
    if (cc % sp->bytesperline) TIFFWarning(tif->tif_name,no_write_frac);
    if ( (cc /= bytesperline)      /* No. of complete rows in caller's buffer */
       > (rows = sp->cinfo.c.image_height - sp->cinfo.c.next_scanline)
       ) cc = rows;
    while (--cc >= 0)
      {
        if (   CALLJPEG(sp,-1,jpeg_write_scanlines(&sp->cinfo.c,(JSAMPARRAY)&buf,1))
            != 1
           ) return 0;
        ++tif->tif_row;
        buf += sp->bytesperline;
      };
    return 1;
  }

/*ARGSUSED*/ static int
OJPEGEncodeRaw(register TIFF *tif,tidata_t buf,tsize_t cc,tsample_t s)
  { tsize_t rows;                          /* No. of unprocessed rows in file */
    JDIMENSION lines_per_MCU, size;
    register OJPEGState *sp = OJState(tif);

 /* Encode a chunk of pixels, where returned data is down-sampled as per the
    sampling factors.  The data is expected to be written in scan-line
    multiples.
 */
    cc /= sp->bytesperline;
    if (cc % sp->bytesperline) TIFFWarning(tif->tif_name,no_write_frac);
    if ( (cc /= bytesperline)      /* No. of complete rows in caller's buffer */
       > (rows = sp->cinfo.c.image_height - sp->cinfo.c.next_scanline)
       ) cc = rows;
#   ifdef C_LOSSLESS_SUPPORTED
    lines_per_MCU = sp->cinfo.c.max_samp_factor*(size = sp->cinfo.d.data_unit);
#   else
    lines_per_MCU = sp->cinfo.c.max_samp_factor*(size = DCTSIZE);
#   endif
    while (--cc >= 0)
      { int ci = 0, clumpoffset = 0;
        register jpeg_component_info *compptr = sp->cinfo.c.comp_info;

     /* The fastest way to separate the data is to make 1 pass over the scan
        line for each row of each component.
     */
        do
          { int ypos = 0;

            do
              { int padding;
                register JSAMPLE *inptr = (JSAMPLE*)buf + clumpoffset,
                                 *outptr =
                  sp->ds_buffer[ci][sp->scancount*compptr->v_samp_factor+ypos];
             /* Cb,Cr both have sampling factors 1, so this is correct */
                register int clumps_per_line =
                  sp->cinfo.c.comp_info[1].downsampled_width,
                             xpos;

                padding = (int)
#                         ifdef C_LOSSLESS_SUPPORTED
                          ( compptr->width_in_data_units * size
#                         else
                          ( compptr->width_in_blocks * size
#                         endif
                          - clumps_per_line * compptr->h_samp_factor
                          );
                if (compptr->h_samp_factor == 1) /* Cb & Cr fast path */
                  do *outptr++ = *inptr;
                  while ((inptr += sp->samplesperclump),--clumps_per_line > 0);
                else /* general case */
                  do
                    {
                      xpos = 0;
                      do *outptr++ = inptr[xpos];
                      while (++xpos < compptr->h_samp_factor);
                    }
                  while ((inptr += sp->samplesperclump),--clumps_per_line > 0);
                xpos = 0; /* Pad each scan line as needed */
                do outptr[0] = outptr[-1]; while (++outptr,++xpos < padding);
                clumpoffset += compptr->h_samp_factor;
              }
            while (++ypos < compptr->v_samp_factor);
          }
        while (++compptr,++ci < sp->cinfo.c.num_components);
        if (++sp->scancount >= size)
          {
            if (   CALLJPEG(sp,-1,jpeg_write_raw_data(&sp->cinfo.c,sp->ds_buffer,lines_per_MCU))
                != lines_per_MCU
               ) return 0;
            sp->scancount = 0;
          };
        ++tif->tif_row++
        buf += sp->bytesperline;
      };
    return 1;
  }

static int
OJPEGSetupEncode(register TIFF *tif)
  { static const char module[]={"OJPEGSetupEncode"};
    uint32 segment_height, segment_width;
    int status = 1;                              /* Assume success by default */
    register OJPEGState *sp = OJState(tif);
#   define td (&tif->tif_dir)

 /* Verify miscellaneous parameters.  This will need work if the TIFF Library
    ever supports different depths for different components, or if the JPEG
    Library ever supports run-time depth selection.  Neither seems imminent.
 */
    if (td->td_bitspersample != 8)
      {
        TIFFError(module,bad_bps,td->td_bitspersample);
        status = 0;
      };

 /* The TIFF Version 6.0 specification and IJG JPEG Library accept different
    sets of color spaces, so verify that our image belongs to the common subset
    and map its photometry code, then initialize to handle subsampling and
    optional JPEG Library YCbCr <-> RGB color-space conversion.
 */
    switch (td->td_photometric)
      {
        case PHOTOMETRIC_YCBCR     :

       /* ISO IS 10918-1 requires that JPEG subsampling factors be 1-4, but
          TIFF Version 6.0 is more restrictive: only 1, 2, and 4 are allowed.
       */
          if (   (   td->td_ycbcrsubsampling[0] == 1
                  || td->td_ycbcrsubsampling[0] == 2
                  || td->td_ycbcrsubsampling[0] == 4
                 )
              && (   td->td_ycbcrsubsampling[1] == 1
                  || td->td_ycbcrsubsampling[1] == 2
                  || td->td_ycbcrsubsampling[1] == 4
                 )
             )
            sp->cinfo.c.raw_data_in =
              ( (sp->h_sampling = td->td_ycbcrsubsampling[0]) << 3
              | (sp->v_sampling = td->td_ycbcrsubsampling[1])
              ) != 011;
          else
            {
              TIFFError(module,bad_subsampling);
              status = 0;
            };

       /* A ReferenceBlackWhite field MUST be present, since the default value
          is inapproriate for YCbCr.  Fill in the proper value if the
          application didn't set it.
       */
          if (!TIFFFieldSet(tif,FIELD_REFBLACKWHITE))
            { float refbw[6];
              long top = 1L << td->td_bitspersample;
 
              refbw[0] = 0;
              refbw[1] = (float)(top-1L);
              refbw[2] = (float)(top>>1);
              refbw[3] = refbw[1];
              refbw[4] = refbw[2];
              refbw[5] = refbw[1];
              TIFFSetField(tif,TIFFTAG_REFERENCEBLACKWHITE,refbw);
            };
          sp->cinfo.c.jpeg_color_space = JCS_YCbCr;
          if (sp->jpegcolormode == JPEGCOLORMODE_RGB)
            {
              sp->cinfo.c.raw_data_in = FALSE;
              sp->in_color_space = JCS_RGB;
              break;
            };
          goto L2;
        case PHOTOMETRIC_MINISBLACK:
          sp->cinfo.c.jpeg_color_space = JCS_GRAYSCALE;
          goto L1;
        case PHOTOMETRIC_RGB       :
          sp->cinfo.c.jpeg_color_space = JCS_RGB;
          goto L1;
        case PHOTOMETRIC_SEPARATED :
          sp->cinfo.c.jpeg_color_space = JCS_CMYK;
      L1: sp->jpegcolormode = JPEGCOLORMODE_RAW; /* No JPEG Lib. conversion */
      L2: sp->cinfo.d.in_color_space = sp->cinfo.d.jpeg_color-space;
          break;
        default                    :
          TIFFError(module,bad_photometry,td->td_photometric);
          status = 0;
      };
    tif->tif_encoderow = tif->tif_encodestrip = tif->tif_encodetile =
      sp->cinfo.c.raw_data_in ? OJPEGEncodeRaw : OJPEGEncode;
    if (isTiled(tif))
      { tsize_t size;

#       ifdef C_LOSSLESS_SUPPORTED
        if ((size = sp->v_sampling*sp->cinfo.c.data_unit) < 16) size = 16;
#       else
        if ((size = sp->v_sampling*DCTSIZE) < 16) size = 16;
#       endif
        if ((segment_height = td->td_tilelength) % size)
          {
            TIFFError(module,"JPEG tile height must be multiple of %d",size);
            status = 0;
          };
#       ifdef C_LOSSLESS_SUPPORTED
        if ((size = sp->h_sampling*sp->cinfo.c.data_unit) < 16) size = 16;
#       else
        if ((size = sp->h_sampling*DCTSIZE) < 16) size = 16;
#       endif
        if ((segment_width = td->td_tilewidth) % size)
          {
            TIFFError(module,"JPEG tile width must be multiple of %d",size);
            status = 0;
          };
        sp->bytesperline = TIFFTileRowSize(tif);
      }
    else
      { tsize_t size;

#       ifdef C_LOSSLESS_SUPPORTED
        if ((size = sp->v_sampling*sp->cinfo.c.data_unit) < 16) size = 16;
#       else
        if ((size = sp->v_sampling*DCTSIZE) < 16) size = 16;
#       endif
        if (td->td_rowsperstrip < (segment_height = td->td_imagelength))
          {
            if (td->td_rowsperstrip % size)
              {
                TIFFError(module,"JPEG RowsPerStrip must be multiple of %d",size);
                status = 0;
              };
            segment_height = td->td_rowsperstrip;
          };
        segment_width = td->td_imagewidth;
        sp->bytesperline = tif->tif_scanlinesize;
      };
    if (segment_width > 65535 || segment_height > 65535)
      {
        TIFFError(module,"Strip/tile too large for JPEG");
        status = 0;
      };

 /* Initialize all JPEG parameters to default values.  Note that the JPEG
    Library's "jpeg_set_defaults()" method needs legal values for the
    "in_color_space" and "input_components" fields.
 */
    sp->cinfo.c.input_components = 1; /* Default for JCS_UNKNOWN */
    if (!CALLVJPEG(sp,jpeg_set_defaults(&sp->cinfo.c))) status = 0;
    switch (sp->jpegtablesmode & (JPEGTABLESMODE_HUFF|JPEGTABLESMODE_QUANT))
      { register JHUFF_TBL *htbl;
        register JQUANT_TBL *qtbl;

        case 0                                       :
          sp->cinfo.c.optimize_coding = TRUE;
        case JPEGTABLESMODE_HUFF                     :
          if (!CALLVJPEG(sp,jpeg_set_quality(&sp->cinfo.c,sp->jpegquality,FALSE)))
            return 0;
          if (qtbl = sp->cinfo.c.quant_tbl_ptrs[0]) qtbl->sent_table = FALSE;
          if (qtbl = sp->cinfo.c.quant_tbl_ptrs[1]) qtbl->sent_table = FALSE;
          goto L3;
        case JPEGTABLESMODE_QUANT                    :
          sp->cinfo.c.optimize_coding = TRUE;

       /* We do not support application-supplied JPEG tables, so mark the field
          "not present".
       */
      L3: TIFFClrFieldBit(tif,FIELD_JPEGTABLES);
          break;
        case JPEGTABLESMODE_HUFF|JPEGTABLESMODE_QUANT:
          if (   !CALLVJPEG(sp,jpeg_set_quality(&sp->cinfo.c,sp->jpegquality,FALSE))
              || !CALLVJPEG(sp,jpeg_suppress_tables(&sp->cinfo.c,TRUE))
             )
            {
              status = 0;
              break;
            };
          if (qtbl = sp->cinfo.c.quant_tbl_ptrs[0]) qtbl->sent_table = FALSE;
          if (htbl = sp->cinfo.c.dc_huff_tbl_ptrs[0]) htbl->sent_table = FALSE;
          if (htbl = sp->cinfo.c.ac_huff_tbl_ptrs[0]) htbl->sent_table = FALSE;
          if (sp->cinfo.c.jpeg_color_space == JCS_YCbCr)
            {
              if (qtbl = sp->cinfo.c.quant_tbl_ptrs[1])
                qtbl->sent_table = FALSE;
              if (htbl = sp->cinfo.c.dc_huff_tbl_ptrs[1])
                htbl->sent_table = FALSE;
              if (htbl = sp->cinfo.c.ac_huff_tbl_ptrs[1])
                htbl->sent_table = FALSE;
            };
          if (   TIFFojpeg_tables_dest(sp,tif)
              && CALLVJPEG(sp,jpeg_write_tables(&sp->cinfo.c))
             )
            {
    
           /* Mark the field "present".  We can't use "TIFFSetField()" because
              "BEENWRITING" is already set!
           */
              TIFFSetFieldBit(tif,FIELD_JPEGTABLES);
              tif->tif_flags |= TIFF_DIRTYDIRECT;
            }
          else status = 0;
      };
    if (   sp->cinfo.c.raw_data_in
        && !alloc_downsampled_buffers(tif,sp->cinfo.c.comp_info,
                                      sp->cinfo.c.num_components)
       ) status = 0;
    if (status == 0) return 0; /* If TIFF errors, don't bother to continue */
 /* Grab parameters that are same for all strips/tiles. */

    sp->dest.init_destination = std_init_destination;
    sp->dest.empty_output_buffer = std_empty_output_buffer;
    sp->dest.term_destination = std_term_destination;
    sp->cinfo.c.dest = &sp->dest;
    sp->cinfo.c.data_precision = td->td_bitspersample;
    sp->cinfo.c.write_JFIF_header = /* Don't write extraneous markers */
    sp->cinfo.c.write_Adobe_marker = FALSE;
    sp->cinfo.c.image_width = segment_width;
    sp->cinfo.c.image_height = segment_height;
    sp->cinfo.c.comp_info[0].h_samp_factor =
    sp->cinfo.c.comp_info[0].v_samp_factor = 1;
    return CALLVJPEG(sp,jpeg_start_compress(&sp->cinfo.c,FALSE));
#   undef td
  }

static int
OJPEGPreEncode(register TIFF *tif,tsample_t s)
  { register OJPEGState *sp = OJState(tif);
#   define td (&tif->tif_dir)

 /* If we are about to write the first row of an image plane, which should
    coincide with a JPEG "scan", reset the JPEG Library's compressor.  Otherwise
    let the compressor run "as is" and return a "success" status without further
    ado.
 */
    if (     (isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip)
           % td->td_stripsperimage
        == 0
       )
      {
        if (   (sp->cinfo.c.comp_info[0].component_id = s) == 1)
            && sp->cinfo.c.jpeg_color_space == JCS_YCbCr
           )
          {
            sp->cinfo.c.comp_info[0].quant_tbl_no =
            sp->cinfo.c.comp_info[0].dc_tbl_no =
            sp->cinfo.c.comp_info[0].ac_tbl_no = 1;
            sp->cinfo.c.comp_info[0].h_samp_factor = sp->h_sampling;
            sp->cinfo.c.comp_info[0].v_samp_factor = sp->v_sampling;
    
         /* Scale expected strip/tile size to match a downsampled component. */
    
            sp->cinfo.c.image_width = TIFFhowmany(segment_width,sp->h_sampling);
            sp->cinfo.c.image_height=TIFFhowmany(segment_height,sp->v_sampling);
          };
        sp->scancount = 0; /* Mark subsampling buffer(s) empty */
      };
    return 1;
#   undef td
  }

static int
OJPEGPostEncode(register TIFF *tif)
  { register OJPEGState *sp = OJState(tif);

 /* Finish up at the end of a strip or tile. */

    if (sp->scancount > 0) /* emit partial buffer of down-sampled data */
      { JDIMENSION n;

#       ifdef C_LOSSLESS_SUPPORTED
        if (   sp->scancount < sp->cinfo.c.data_unit
            && sp->cinfo.c.num_components > 0
           )
#       else
        if (sp->scancount < DCTSIZE && sp->cinfo.c.num_components > 0)
#       endif
          { int ci = 0,                            /* Pad the data vertically */
#           ifdef C_LOSSLESS_SUPPORTED
                size = sp->cinfo.c.data_unit;
#           else
                size = DCTSIZE;
#           endif
            register jpeg_component_info *compptr = sp->cinfo.c.comp_info;

            do
#              ifdef C_LOSSLESS_SUPPORTED
               { tsize_t row_width = compptr->width_in_data_units
#              else
                 tsize_t row_width = compptr->width_in_blocks
#              endif
                   *size*sizeof(JSAMPLE);
                 int ypos = sp->scancount*compptr->v_samp_factor;

                 do _TIFFmemcpy( (tdata_t)sp->ds_buffer[ci][ypos]
                               , (tdata_t)sp->ds_buffer[ci][ypos-1]
                               , row_width
                               );
                 while (++ypos < compptr->v_samp_factor*size);
               }
            while (++compptr,++ci < sp->cinfo.c.num_components);
          };
        n = sp->cinfo.c.max_v_samp_factor*size;
        if (CALLJPEG(sp,-1,jpeg_write_raw_data(&sp->cinfo.c,sp->ds_buffer,n)) != n)
          return 0;
      };
    return CALLVJPEG(sp,jpeg_finish_compress(&sp->cinfo.c));
  }
#endif /* never */

/* JPEG Decoding begins here. */

/*ARGSUSED*/ static int
OJPEGDecode(register TIFF *tif,tidata_t buf,tsize_t cc,tsample_t s)
  { tsize_t bytesperline = isTiled(tif)
                         ? TIFFTileRowSize(tif)
                         : tif->tif_scanlinesize,
            rows;                          /* No. of unprocessed rows in file */
    register OJPEGState *sp = OJState(tif);

 /* Decode a chunk of pixels, where the input data has not NOT been down-
    sampled, or else the TIFF Library's client has used the "JPEGColorMode" TIFF
    pseudo-tag to request that the JPEG Library do color-space conversion; this
    is the normal case.  The data is expected to be read in scan-line multiples,
    and this subroutine is called for both pixel-interleaved and separate color
    planes.

    WARNING:  Unlike "OJPEGDecodeRawContig()", below, the no. of Bytes in each
              decoded row is calculated here as "bytesperline" instead of
    using "sp->bytesperline", which might be a little smaller.  This can
    occur for an old tiled image whose width isn't a multiple of 8 pixels.
    That's illegal according to the TIFF Version 6 specification, but some
    test files, like "zackthecat.tif", were built that way.  In those cases,
    we want to embed the image's true width in our caller's buffer (which is
    presumably allocated according to the expected tile width) by
    effectively "padding" it with unused Bytes at the end of each row.
 */
    if ( (cc /= bytesperline)      /* No. of complete rows in caller's buffer */
       > (rows = sp->cinfo.d.output_height - sp->cinfo.d.output_scanline)
       ) cc = rows;
    while (--cc >= 0)
      {
        if (   CALLJPEG(sp,-1,jpeg_read_scanlines(&sp->cinfo.d,(JSAMPARRAY)&buf,1))
            != 1
           ) return 0;
        buf += bytesperline;
        ++tif->tif_row;
      };

 /* BEWARE OF KLUDGE:  If our input file was produced by Microsoft's Wang
                       Imaging for Windows application, the DC coefficients of
    each JPEG image component (Y,Cb,Cr) must be reset at the end of each TIFF
    "strip", and any JPEG data bits remaining in the current Byte of the
    decoder's input buffer must be discarded.  To do so, we create an "ad hoc"
    interface in the "jdhuff.c" module of IJG JPEG Library Version 6 (module
    "jdshuff.c", if Ken Murchison's lossless-Huffman patch is applied), and we
    invoke that interface here after decoding each "strip".
 */
    if (sp->is_WANG) jpeg_reset_huff_decode(&sp->cinfo.d);
    return 1;
  }

/*ARGSUSED*/ static int
OJPEGDecodeRawContig(register TIFF *tif,tidata_t buf,tsize_t cc,tsample_t s)
  { tsize_t rows;                          /* No. of unprocessed rows in file */
    JDIMENSION lines_per_MCU, size;
    register OJPEGState *sp = OJState(tif);

 /* Decode a chunk of pixels, where the input data has pixel-interleaved color
    planes, some of which have been down-sampled, but the TIFF Library's client
    has NOT used the "JPEGColorMode" TIFF pseudo-tag to request that the JPEG
    Library do color-space conversion.  In other words, we must up-sample/
    expand/duplicate image components according to the image's sampling factors,
    without changing its color space.  The data is expected to be read in scan-
    line multiples.
 */
    if ( (cc /= sp->bytesperline)  /* No. of complete rows in caller's buffer */
       > (rows = sp->cinfo.d.output_height - sp->cinfo.d.output_scanline)
       ) cc = rows;
    lines_per_MCU = sp->cinfo.d.max_v_samp_factor
#   ifdef D_LOSSLESS_SUPPORTED
                  * (size = sp->cinfo.d.min_codec_data_unit);
#   else
                  * (size = DCTSIZE);
#   endif
    while (--cc >= 0)
      { int clumpoffset, ci;
        register jpeg_component_info *compptr;

        if (sp->scancount >= size) /* reload downsampled-data buffers */
          {
            if (   CALLJPEG(sp,-1,jpeg_read_raw_data(&sp->cinfo.d,sp->ds_buffer,lines_per_MCU))
                != lines_per_MCU
               ) return 0;
            sp->scancount = 0;
          };

     /* The fastest way to separate the data is: make 1 pass over the scan
        line for each row of each component.
     */
        clumpoffset = ci = 0;
        compptr = sp->cinfo.d.comp_info;
        do
          { int ypos = 0;

            if (compptr->h_samp_factor == 1) /* fast path */
              do
                { register JSAMPLE *inptr =
                    sp->ds_buffer[ci][sp->scancount*compptr->v_samp_factor+ypos],
                                   *outptr = (JSAMPLE *)buf + clumpoffset;
                  register int clumps_per_line = compptr->downsampled_width;

                  do *outptr = *inptr++;
                  while ((outptr += sp->samplesperclump),--clumps_per_line > 0);
                }
              while ( (clumpoffset += compptr->h_samp_factor)
                    , ++ypos < compptr->v_samp_factor
                    );
            else /* general case */
              do
                { register JSAMPLE *inptr =
                    sp->ds_buffer[ci][sp->scancount*compptr->v_samp_factor+ypos],
                                   *outptr = (JSAMPLE *)buf + clumpoffset;
                  register int clumps_per_line = compptr->downsampled_width;

                  do
                    { register int xpos = 0;

                      do outptr[xpos] = *inptr++;
                      while (++xpos < compptr->h_samp_factor);
                    }
                  while ((outptr += sp->samplesperclump),--clumps_per_line > 0);
                }
              while ( (clumpoffset += compptr->h_samp_factor)
                    , ++ypos < compptr->v_samp_factor
                    );
          }
        while (++compptr,++ci < sp->cinfo.d.num_components);
        ++sp->scancount;
        buf += sp->bytesperline;
        ++tif->tif_row;
      };

 /* BEWARE OF KLUDGE:  If our input file was produced by Microsoft's Wang
                       Imaging for Windows application, the DC coefficients of
    each JPEG image component (Y,Cb,Cr) must be reset at the end of each TIFF
    "strip", and any JPEG data bits remaining in the current Byte of the
    decoder's input buffer must be discarded.  To do so, we create an "ad hoc"
    interface in the "jdhuff.c" module of IJG JPEG Library Version 6 (module
    "jdshuff.c", if Ken Murchison's lossless-Huffman patch is applied), and we
    invoke that interface here after decoding each "strip".
 */
    if (sp->is_WANG) jpeg_reset_huff_decode(&sp->cinfo.d);
    return 1;
  }

/*ARGSUSED*/ static int
OJPEGDecodeRawSeparate(TIFF *tif,register tidata_t buf,tsize_t cc,tsample_t s)
  { tsize_t rows;                          /* No. of unprocessed rows in file */
    JDIMENSION lines_per_MCU,
               size,                                             /* ...of MCU */
               v;                   /* Component's vertical up-sampling ratio */
    register OJPEGState *sp = OJState(tif);
    register jpeg_component_info *compptr = sp->cinfo.d.comp_info + s;

 /* Decode a chunk of pixels, where the input data has separate color planes,
    some of which have been down-sampled, but the TIFF Library's client has NOT
    used the "JPEGColorMode" TIFF pseudo-tag to request that the JPEG Library
    do color-space conversion.  The data is expected to be read in scan-line
    multiples.
 */
    v = sp->cinfo.d.max_v_samp_factor/compptr->v_samp_factor;
    if ( (cc /= compptr->downsampled_width) /* No. of rows in caller's buffer */
       > (rows = (sp->cinfo.d.output_height-sp->cinfo.d.output_scanline+v-1)/v)
       ) cc = rows; /* No. of rows of "clumps" to read */
    lines_per_MCU = sp->cinfo.d.max_v_samp_factor
#   ifdef D_LOSSLESS_SUPPORTED
                  * (size = sp->cinfo.d.min_codec_data_unit);
#   else
                  * (size = DCTSIZE);
#   endif
 L: if (sp->scancount >= size) /* reload downsampled-data buffers */
      {
        if (   CALLJPEG(sp,-1,jpeg_read_raw_data(&sp->cinfo.d,sp->ds_buffer,lines_per_MCU))
            != lines_per_MCU
           ) return 0;
        sp->scancount = 0;
      };
    rows = 0;
    do
      { register JSAMPLE *inptr =
          sp->ds_buffer[s][sp->scancount*compptr->v_samp_factor + rows];
        register int clumps_per_line = compptr->downsampled_width;

        do *buf++ = *inptr++; while (--clumps_per_line > 0); /* Copy scanline */
        tif->tif_row += v;
        if (--cc <= 0) return 1; /* End of caller's buffer? */
      }
    while (++rows < compptr->v_samp_factor);
    ++sp->scancount;
    goto L;
  }

/* "OJPEGSetupDecode()" temporarily forces the JPEG Library to use the following
   subroutine as a "dummy" input reader in order to fool the library into
   thinking that it has read the image's first "Start of Scan" (SOS) marker, so
   that it initializes accordingly.
*/
/*ARGSUSED*/ METHODDEF(int)
fake_SOS_marker(j_decompress_ptr cinfo){return JPEG_REACHED_SOS;}

/*ARGSUSED*/ METHODDEF(int)
suspend(j_decompress_ptr cinfo){return JPEG_SUSPENDED;}

/* The JPEG Library's "null" color-space converter actually re-packs separate
   color planes (it's native image representation) into a pixel-interleaved,
   contiguous plane.  But if our TIFF Library client is tryng to process a
   PLANARCONFIG_SEPARATE image, we don't want that; so here are modifications of
   code in the JPEG Library's "jdcolor.c" file, which simply copy Bytes to a
   color plane specified by the current JPEG "scan".
*/
METHODDEF(void)
ycc_rgb_convert(register j_decompress_ptr cinfo,JSAMPIMAGE in,JDIMENSION row,
                register JSAMPARRAY out,register int nrows)
  { typedef struct                /* "jdcolor.c" color-space conversion state */
      {

     /* WARNING:  This declaration is ugly and dangerous!  It's supposed to be
                  private to the JPEG Library's "jdcolor.c" module, but we also
        need it here.  Since the library's copy might change without notice, be
        sure to keep this one synchronized or the following code will break!
     */
        struct jpeg_color_deconverter pub; /* Public fields */
     /* Private state for YCC->RGB conversion */
        int *Cr_r_tab,   /* ->Cr to R conversion table */
            *Cb_b_tab;   /* ->Cb to B conversion table */
        INT32 *Cr_g_tab, /* ->Cr to G conversion table */
              *Cb_g_tab; /* ->Cb to G conversion table */
      } *my_cconvert_ptr;
    my_cconvert_ptr cconvert = (my_cconvert_ptr)cinfo->cconvert;
    JSAMPARRAY irow0p = in[0] + row;
    register JSAMPLE *range_limit = cinfo->sample_range_limit;
    register JSAMPROW outp, Y;

    switch (cinfo->output_scan_number - 1)
      { JSAMPARRAY irow1p, irow2p;
        register INT32 *table0, *table1;
        SHIFT_TEMPS

        case RGB_RED  : irow2p = in[2] + row;
                        table0 = (INT32 *)cconvert->Cr_r_tab;
                        while (--nrows >= 0)
                          { register JSAMPROW Cr = *irow2p++;
                             register int i = cinfo->output_width;

                             Y = *irow0p++;
                             outp = *out++;
                             while (--i >= 0)
                               *outp++ = range_limit[*Y++ + table0[*Cr++]];
                          };
                        return;
        case RGB_GREEN: irow1p = in[1] + row;
                        irow2p = in[2] + row;
                        table0 = cconvert->Cb_g_tab;
                        table1 = cconvert->Cr_g_tab;
                        while (--nrows >= 0)
                          { register JSAMPROW Cb = *irow1p++,
                                              Cr = *irow2p++;
                             register int i = cinfo->output_width;

                             Y = *irow0p++;
                             outp = *out++;
                             while (--i >= 0)
                               *outp++ =
                                 range_limit[ *Y++
                                            + RIGHT_SHIFT(table0[*Cb++]+table1[*Cr++],16)
                                            ];
                          };
                        return;
        case RGB_BLUE : irow1p = in[1] + row;
                        table0 = (INT32 *)cconvert->Cb_b_tab;
                        while (--nrows >= 0)
                          { register JSAMPROW Cb = *irow1p++;
                             register int i = cinfo->output_width;

                             Y = *irow0p++;
                             outp = *out++;
                             while (--i >= 0)
                               *outp++ = range_limit[*Y++ + table0[*Cb++]];
                          }
      }
  }

METHODDEF(void)
null_convert(register j_decompress_ptr cinfo,JSAMPIMAGE in,JDIMENSION row,
             register JSAMPARRAY out,register int nrows)
  { register JSAMPARRAY irowp = in[cinfo->output_scan_number - 1] + row;

    while (--nrows >= 0) _TIFFmemcpy(*out++,*irowp++,cinfo->output_width);
  }

static int
OJPEGSetupDecode(register TIFF *tif)
  { static char module[]={"OJPEGSetupDecode"};
    J_COLOR_SPACE jpeg_color_space,   /* Color space of JPEG-compressed image */
                  out_color_space;       /* Color space of decompressed image */
    uint32 segment_width;
    int status = 1;                              /* Assume success by default */
    boolean downsampled_output=FALSE, /* <=> Want JPEG Library's "raw" image? */
            is_JFIF;                                       /* <=> JFIF image? */
    register OJPEGState *sp = OJState(tif);
#   define td (&tif->tif_dir)

 /* Verify miscellaneous parameters.  This will need work if the TIFF Library
    ever supports different depths for different components, or if the JPEG
    Library ever supports run-time depth selection.  Neither seems imminent.
 */
    if (td->td_bitspersample != sp->cinfo.d.data_precision)
      {
        TIFFError(module,bad_bps,td->td_bitspersample);
        status = 0;
      };

 /* The TIFF Version 6.0 specification and IJG JPEG Library accept different
    sets of color spaces, so verify that our image belongs to the common subset
    and map its photometry code, then initialize to handle subsampling and
    optional JPEG Library YCbCr <-> RGB color-space conversion.
 */
    switch (td->td_photometric)
      {
        case PHOTOMETRIC_YCBCR     :

       /* ISO IS 10918-1 requires that JPEG subsampling factors be 1-4, but
          TIFF Version 6.0 is more restrictive: only 1, 2, and 4 are allowed.
       */
          if (   (   td->td_ycbcrsubsampling[0] == 1
                  || td->td_ycbcrsubsampling[0] == 2
                  || td->td_ycbcrsubsampling[0] == 4
                 )
              && (   td->td_ycbcrsubsampling[1] == 1
                  || td->td_ycbcrsubsampling[1] == 2
                  || td->td_ycbcrsubsampling[1] == 4
                 )
             )
            downsampled_output =
              (
                (sp->h_sampling = td->td_ycbcrsubsampling[0]) << 3
              | (sp->v_sampling = td->td_ycbcrsubsampling[1])
              ) != 011;
          else
            {
              TIFFError(module,bad_subsampling);
              status = 0;
            };
          jpeg_color_space = JCS_YCbCr;
          if (sp->jpegcolormode == JPEGCOLORMODE_RGB)
            {
              downsampled_output = FALSE;
              out_color_space = JCS_RGB;
              break;
            };
          goto L2;
        case PHOTOMETRIC_MINISBLACK:
          jpeg_color_space = JCS_GRAYSCALE;
          goto L1;
        case PHOTOMETRIC_RGB       :
          jpeg_color_space = JCS_RGB;
          goto L1;
        case PHOTOMETRIC_SEPARATED :
          jpeg_color_space = JCS_CMYK;
      L1: sp->jpegcolormode = JPEGCOLORMODE_RAW; /* No JPEG Lib. conversion */
      L2: out_color_space = jpeg_color_space;
          break;
        default                    :
          TIFFError(module,bad_photometry,td->td_photometric);
          status = 0;
      };
    if (status == 0) return 0; /* If TIFF errors, don't bother to continue */

 /* Set parameters that are same for all strips/tiles. */

    sp->cinfo.d.src = &sp->src;
    sp->src.init_source = std_init_source;
    sp->src.fill_input_buffer = std_fill_input_buffer;
    sp->src.skip_input_data = std_skip_input_data;
    sp->src.resync_to_restart = jpeg_resync_to_restart;
    sp->src.term_source = std_term_source;

 /* BOGOSITY ALERT!  The Wang Imaging application for Microsoft Windows produces
                     images containing "JPEGInterchangeFormat[Length]" TIFF
    records that resemble JFIF-in-TIFF encapsulations but, in fact, violate the
    TIFF Version 6 specification in several ways; nevertheless, we try to handle
    them gracefully because there are apparently a lot of them around.  The
    purported "JFIF" data stream in one of these files vaguely resembles a JPEG
    "tables only" data stream, except that there's no trailing EOI marker.  The
    rest of the JPEG data stream lies in a discontiguous file region, identified
    by the 0th Strip offset (which is *also* illegal!), where it begins with an
    SOS marker and apparently continues to the end of the file.  There is no
    trailing EOI marker here, either.
 */
    is_JFIF = !sp->is_WANG && TIFFFieldSet(tif,FIELD_JPEGIFOFFSET);

 /* Initialize decompression parameters that won't be overridden by JPEG Library
    defaults set during the "jpeg_read_header()" call, below.
 */
    segment_width = td->td_imagewidth;
    if (isTiled(tif))
      {
        if (sp->is_WANG) /* we don't know how to handle it */
          {
            TIFFError(module,"Tiled Wang image not supported");
            return 0;
          };

     /* BOGOSITY ALERT!  "TIFFTileRowSize()" seems to work fine for modern JPEG-
                         in-TIFF encapsulations where the image width--like the
        tile width--is a multiple of 8 or 16 pixels.  But image widths and
        heights are aren't restricted to 8- or 16-bit multiples, and we need
        the exact Byte count of decompressed scan lines when we call the JPEG
        Library.  At least one old file ("zackthecat.tif") in the TIFF Library
        test suite has widths and heights slightly less than the tile sizes, and
        it apparently used the bogus computation below to determine the number
        of Bytes per scan line (was this due to an old, broken version of
        "TIFFhowmany()"?).  Before we get here, "OJPEGSetupDecode()" verified
        that our image uses 8-bit samples, so the following check appears to
        return the correct answer in all known cases tested to date.
     */
        if (is_JFIF || (segment_width & 7) == 0)
          sp->bytesperline = TIFFTileRowSize(tif); /* Normal case */
        else
          {
            /* Was the file-encoder's segment-width calculation bogus? */
            segment_width = (segment_width/sp->h_sampling + 1) * sp->h_sampling;
            sp->bytesperline = segment_width * td->td_samplesperpixel;
          }
      }
    else sp->bytesperline = TIFFVStripSize(tif,1);

 /* BEWARE OF KLUDGE:  If we have JPEG Interchange File Format (JFIF) image,
                       then we want to read "metadata" in the bit-stream's
    header and validate it against corresponding information in TIFF records.
    But if we have a *really old* JPEG file that's not JFIF, then we simply
    assign TIFF-record values to JPEG Library variables without checking.
 */
    if (is_JFIF) /* JFIF image */
      { unsigned char *end_of_data;
        int subsampling_factors;
        register unsigned char *p;
        register int i;

     /* WARNING:  Although the image file contains a JFIF bit stream, it might
                  also contain some old TIFF records causing "OJPEGVSetField()"
        to have allocated quantization or Huffman decoding tables.  But when the
        JPEG Library reads and parses the JFIF header below, it reallocate these
        tables anew without checking for "dangling" pointers, thereby causing a
        memory "leak".  We have enough information to potentially deallocate the
        old tables here, but unfortunately JPEG Library Version 6B uses a "pool"
        allocator for small objects, with no deallocation procedure; instead, it
        reclaims a whole pool when an image is closed/destroyed, so well-behaved
        TIFF client applications (i.e., those which close their JPEG images as
        soon as they're no longer needed) will waste memory for a short time but
        recover it eventually.  But ill-behaved TIFF clients (i.e., those which
        keep many JPEG images open gratuitously) can exhaust memory prematurely.
        If the JPEG Library ever implements a deallocation procedure, insert
        this clean-up code:
     */
#       ifdef someday
        if (sp->jpegtablesmode & JPEGTABLESMODE_QUANT) /* free quant. tables */
          { register int i = 0;

            do
              { register JQUANT_TBL *q;

                if (q = sp->cinfo.d.quant_tbl_ptrs[i])
                  {
                    jpeg_free_small(&sp->cinfo.comm,q,sizeof *q);
                    sp->cinfo.d.quant_tbl_ptrs[i] = 0;
                  }
              }
            while (++i < NUM_QUANT_TBLS);
          };
        if (sp->jpegtablesmode & JPEGTABLESMODE_HUFF) /* free Huffman tables */
          { register int i = 0;

            do
              { register JHUFF_TBL *h;

                if (h = sp->cinfo.d.dc_huff_tbl_ptrs[i])
                  {
                    jpeg_free_small(&sp->cinfo.comm,h,sizeof *h);
                    sp->cinfo.d.dc_huff_tbl_ptrs[i] = 0;
                  };
                if (h = sp->cinfo.d.ac_huff_tbl_ptrs[i])
                  {
                    jpeg_free_small(&sp->cinfo.comm,h,sizeof *h);
                    sp->cinfo.d.ac_huff_tbl_ptrs[i] = 0;
                  }
              }
            while (++i < NUM_HUFF_TBLS);
          };
#       endif /* someday */

     /* Since we might someday wish to try rewriting "old format" JPEG-in-TIFF
        encapsulations in "new format" files, try to synthesize the value of a
        modern "JPEGTables" TIFF record by scanning the JPEG data from just past
        the "Start of Information" (SOI) marker until something other than a
        legitimate "table" marker is found, as defined in ISO IS 10918-1
        Appending B.2.4; namely:

        -- Define Quantization Table (DQT)
        -- Define Huffman Table (DHT)
        -- Define Arithmetic Coding table (DAC)
        -- Define Restart Interval (DRI)
        -- Comment (COM)
        -- Application data (APPn)

        For convenience, we also accept "Expansion" (EXP) markers, although they
        are apparently not a part of normal "table" data.
     */
        sp->jpegtables = p = (unsigned char *)sp->src.next_input_byte;
        end_of_data = p + sp->src.bytes_in_buffer;
        p += 2;
        while (p < end_of_data && p[0] == 0xFF)
          switch (p[1])
            {
              default  : goto L;
              case 0xC0: /* SOF0  */
              case 0xC1: /* SOF1  */
              case 0xC2: /* SOF2  */
              case 0xC3: /* SOF3  */
              case 0xC4: /* DHT   */
              case 0xC5: /* SOF5  */
              case 0xC6: /* SOF6  */
              case 0xC7: /* SOF7  */
              case 0xC9: /* SOF9  */
              case 0xCA: /* SOF10 */
              case 0xCB: /* SOF11 */
              case 0xCC: /* DAC   */
              case 0xCD: /* SOF13 */
              case 0xCE: /* SOF14 */
              case 0xCF: /* SOF15 */
              case 0xDB: /* DQT   */
              case 0xDD: /* DRI   */
              case 0xDF: /* EXP   */
              case 0xE0: /* APP0  */
              case 0xE1: /* APP1  */
              case 0xE2: /* APP2  */
              case 0xE3: /* APP3  */
              case 0xE4: /* APP4  */
              case 0xE5: /* APP5  */
              case 0xE6: /* APP6  */
              case 0xE7: /* APP7  */
              case 0xE8: /* APP8  */
              case 0xE9: /* APP9  */
              case 0xEA: /* APP10 */
              case 0xEB: /* APP11 */
              case 0xEC: /* APP12 */
              case 0xED: /* APP13 */
              case 0xEE: /* APP14 */
              case 0xEF: /* APP15 */
              case 0xFE: /* COM   */
                         p += (p[2] << 8 | p[3]) + 2;
            };
     L: if (p - (unsigned char *)sp->jpegtables > 2) /* fake "JPEGTables" */
          {

         /* In case our client application asks, pretend that this image file
            contains a modern "JPEGTables" TIFF record by copying to a buffer
            the initial part of the JFIF bit-stream that we just scanned, from
            the SOI marker through the "metadata" tables, then append an EOI
            marker and flag the "JPEGTables" TIFF record as "present".
         */
            sp->jpegtables_length = p - (unsigned char*)sp->jpegtables + 2;
            p = sp->jpegtables;
            if (!(sp->jpegtables = _TIFFmalloc(sp->jpegtables_length)))
              {
                TIFFError(module,no_jtable_space);
                return 0;
              };
            _TIFFmemcpy(sp->jpegtables,p,sp->jpegtables_length-2);
            p = (unsigned char *)sp->jpegtables + sp->jpegtables_length;
            p[-2] = 0xFF; p[-1] = JPEG_EOI; /* Append EOI marker */
            TIFFSetFieldBit(tif,FIELD_JPEGTABLES);
            tif->tif_flags |= TIFF_DIRTYDIRECT;
          }
        else sp->jpegtables = 0; /* Don't simulate "JPEGTables" */
        if (   CALLJPEG(sp,-1,jpeg_read_header(&sp->cinfo.d,TRUE))
            != JPEG_HEADER_OK
           ) return 0;
        if (   sp->cinfo.d.image_width  != segment_width
            || sp->cinfo.d.image_height != td->td_imagelength 
           )
          {
            TIFFError(module,"Improper JPEG strip/tile size");
            return 0;
          };
        if (sp->cinfo.d.num_components != td->td_samplesperpixel)
          {
            TIFFError(module,"Improper JPEG component count");
            return 0;
          };
        if (sp->cinfo.d.data_precision != td->td_bitspersample)
          {
            TIFFError(module,"Improper JPEG data precision");
            return 0;
          };

     /* Check that JPEG image components all have the same subsampling factors
        declared (or defaulted) in the TIFF file, since TIFF Version 6.0 is more
        restrictive than JPEG:  Only the 0th component may have horizontal and
        vertical subsampling factors other than <1,1>.
     */
        subsampling_factors = sp->h_sampling << 3 | sp->v_sampling;
        i = 0;
        do
          {
            if (   ( sp->cinfo.d.comp_info[i].h_samp_factor << 3
                   | sp->cinfo.d.comp_info[i].v_samp_factor
                   )
                != subsampling_factors
               )
              {
                TIFFError(module,"Improper JPEG subsampling factors");
                return 0;
              };
            subsampling_factors = 011; /* Required for image components > 0 */
          }
        while (++i < sp->cinfo.d.num_components);
      }
    else /* not JFIF image */
      { int (*save)(j_decompress_ptr cinfo) = sp->cinfo.d.marker->read_markers;
        register int i;

     /* We're not assuming that this file's JPEG bit stream has any header
        "metadata", so fool the JPEG Library into thinking that we read a
        "Start of Input" (SOI) marker and a "Start of Frame" (SOFx) marker, then
        force it to read a simulated "Start of Scan" (SOS) marker when we call
        "jpeg_read_header()" below.  This should cause the JPEG Library to
        establish reasonable defaults.
     */
        sp->cinfo.d.marker->saw_SOI =       /* Pretend we saw SOI marker */
        sp->cinfo.d.marker->saw_SOF = TRUE; /* Pretend we saw SOF marker */
        sp->cinfo.d.marker->read_markers =
          sp->is_WANG ? suspend : fake_SOS_marker;
        sp->cinfo.d.global_state = DSTATE_INHEADER;
        sp->cinfo.d.Se = DCTSIZE2-1; /* Suppress JPEG Library warning */
        sp->cinfo.d.image_width  = segment_width;
        sp->cinfo.d.image_height = td->td_imagelength;

     /* The following color-space initialization, including the complicated
        "switch"-statement below, essentially duplicates the logic used by the
        JPEG Library's "jpeg_init_colorspace()" subroutine during compression.
     */
        sp->cinfo.d.num_components = td->td_samplesperpixel;
        sp->cinfo.d.comp_info = (jpeg_component_info *)
          (*sp->cinfo.d.mem->alloc_small)
            ( &sp->cinfo.comm
            , JPOOL_IMAGE
            , sp->cinfo.d.num_components * sizeof *sp->cinfo.d.comp_info
            );
        i = 0;
        do
          {
            sp->cinfo.d.comp_info[i].component_index = i;
            sp->cinfo.d.comp_info[i].component_needed = TRUE;
            sp->cinfo.d.cur_comp_info[i] = &sp->cinfo.d.comp_info[i];
          }
        while (++i < sp->cinfo.d.num_components);
        switch (jpeg_color_space)
          {
            case JCS_UNKNOWN  :
              i = 0;
              do
                {
                  sp->cinfo.d.comp_info[i].component_id = i;
                  sp->cinfo.d.comp_info[i].h_samp_factor =
                  sp->cinfo.d.comp_info[i].v_samp_factor = 1;
                }
              while (++i < sp->cinfo.d.num_components);
              break;
            case JCS_GRAYSCALE:
              sp->cinfo.d.comp_info[0].component_id =
              sp->cinfo.d.comp_info[0].h_samp_factor =
              sp->cinfo.d.comp_info[0].v_samp_factor = 1;
              break;
            case JCS_RGB      :
              sp->cinfo.d.comp_info[0].component_id = 'R';
              sp->cinfo.d.comp_info[1].component_id = 'G';
              sp->cinfo.d.comp_info[2].component_id = 'B';
              i = 0;
              do sp->cinfo.d.comp_info[i].h_samp_factor =
                 sp->cinfo.d.comp_info[i].v_samp_factor = 1;
              while (++i < sp->cinfo.d.num_components);
              break;
            case JCS_CMYK     :
              sp->cinfo.d.comp_info[0].component_id = 'C';
              sp->cinfo.d.comp_info[1].component_id = 'M';
              sp->cinfo.d.comp_info[2].component_id = 'Y';
              sp->cinfo.d.comp_info[3].component_id = 'K';
              i = 0;
              do sp->cinfo.d.comp_info[i].h_samp_factor =
                 sp->cinfo.d.comp_info[i].v_samp_factor = 1;
              while (++i < sp->cinfo.d.num_components);
              break;
            case JCS_YCbCr    :
              i = 0;
              do
                {
                  sp->cinfo.d.comp_info[i].component_id = i+1;
                  sp->cinfo.d.comp_info[i].h_samp_factor =
                  sp->cinfo.d.comp_info[i].v_samp_factor = 1;
                  sp->cinfo.d.comp_info[i].quant_tbl_no =
                  sp->cinfo.d.comp_info[i].dc_tbl_no =
                  sp->cinfo.d.comp_info[i].ac_tbl_no = i > 0;
                }
              while (++i < sp->cinfo.d.num_components);
              sp->cinfo.d.comp_info[0].h_samp_factor = sp->h_sampling;
              sp->cinfo.d.comp_info[0].v_samp_factor = sp->v_sampling;
          };
        sp->cinfo.d.comps_in_scan = td->td_planarconfig == PLANARCONFIG_CONTIG
                                  ? sp->cinfo.d.num_components
                                  : 1;
        i = CALLJPEG(sp,-1,jpeg_read_header(&sp->cinfo.d,!sp->is_WANG));
        sp->cinfo.d.marker->read_markers = save; /* Restore input method */
        if (sp->is_WANG) /* produced by Wang Imaging on Microsoft Windows */
          {
            if (i != JPEG_SUSPENDED) return 0;

         /* BOGOSITY ALERT!  Files prooduced by the Wang Imaging application for
                             Microsoft Windows are a special--and, technically
            illegal--case.  A JPEG SOS marker and rest of the data stream should
            be located at the end of the file, in a position identified by the
            0th Strip offset.
         */
            i = td->td_nstrips - 1;
            sp->src.next_input_byte = tif->tif_base + td->td_stripoffset[0];
            sp->src.bytes_in_buffer = td->td_stripoffset[i] -
              td->td_stripoffset[0] + td->td_stripbytecount[i];
            i = CALLJPEG(sp,-1,jpeg_read_header(&sp->cinfo.d,TRUE));
          };
        if (i != JPEG_HEADER_OK) return 0;
      };

 /* Some of our initialization must wait until the JPEG Library is initialized
    above, in order to override its defaults.
 */
    if (   (sp->cinfo.d.raw_data_out = downsampled_output)
        && !alloc_downsampled_buffers(tif,sp->cinfo.d.comp_info,
                                      sp->cinfo.d.num_components)
       ) return 0;
    sp->cinfo.d.jpeg_color_space = jpeg_color_space;
    sp->cinfo.d.out_color_space = out_color_space;
    sp->cinfo.d.dither_mode = JDITHER_NONE; /* Reduce image "noise" */
    sp->cinfo.d.two_pass_quantize = FALSE;

 /* If the image consists of separate, discontiguous TIFF "samples" (= color
    planes, hopefully = JPEG "scans"), then we must use the JPEG Library's
    "buffered image" mode to decompress the entire image into temporary buffers,
    because the JPEG Library must parse the entire JPEG bit-stream in order to
    be satsified that it has a complete set of color components for each pixel,
    but the TIFF Library must allow our client to extract 1 component at a time.
    Initializing the JPEG Library's "buffered image" mode is tricky:  First, we
    start its decompressor, then we tell the decompressor to "consume" (i.e.,
    buffer) the entire bit-stream.

    WARNING:  Disabling "fancy" up-sampling seems to slightly reduce "noise" for
              certain old Wang Imaging files, but it absolutely *must* be
    enabled if the image has separate color planes, since in that case, the JPEG
    Library doesn't use an "sp->cinfo.d.cconvert" structure (so de-referencing
    this pointer below will cause a fatal crash) but writing our own code to up-
    sample separate color planes is too much work for right now.  Maybe someday?
 */
    sp->cinfo.d.do_fancy_upsampling = /* Always let this default (to TRUE)? */
    sp->cinfo.d.buffered_image = td->td_planarconfig == PLANARCONFIG_SEPARATE;
    if (!CALLJPEG(sp,0,jpeg_start_decompress(&sp->cinfo.d))) return 0;
    if (sp->cinfo.d.buffered_image) /* separate color planes */
      {
        if (sp->cinfo.d.raw_data_out)
          tif->tif_decoderow = tif->tif_decodestrip = tif->tif_decodetile =
            OJPEGDecodeRawSeparate;
        else
          {
            tif->tif_decoderow = tif->tif_decodestrip = tif->tif_decodetile =
              OJPEGDecode;

         /* In JPEG Library Version 6B, color-space conversion isn't implemented
            for separate color planes, so we must do it ourself if our TIFF
            client doesn't want to:
         */
            sp->cinfo.d.cconvert->color_convert =
              sp->cinfo.d.jpeg_color_space == sp->cinfo.d.out_color_space
              ? null_convert : ycc_rgb_convert;
          };
    L3: switch (CALLJPEG(sp,0,jpeg_consume_input(&sp->cinfo.d)))
          {
            default              : goto L3;

         /* If no JPEG "End of Information" (EOI) marker is found when bit-
            stream parsing ends, check whether we have enough data to proceed
            before reporting an error.
         */
            case JPEG_SUSPENDED  : if (  sp->cinfo.d.input_scan_number
                                        *sp->cinfo.d.image_height
                                       + sp->cinfo.d.input_iMCU_row
                                        *sp->cinfo.d.max_v_samp_factor
#                                       ifdef D_LOSSLESS_SUPPORTED
                                        *sp->cinfo.d.data_units_in_MCU
                                        *sp->cinfo.d.min_codec_data_unit
#                                       else
                                        *sp->cinfo.d.blocks_in_MCU
                                        *DCTSIZE
#                                       endif
                                      < td->td_samplesperpixel
                                       *sp->cinfo.d.image_height
                                      )
                                     {
                                       TIFFError(tif->tif_name,
                                         "Premature end of JPEG bit-stream");
                                       return 0;
                                     }
            case JPEG_REACHED_EOI: ;
          }
      }
    else /* pixel-interleaved color planes */
      tif->tif_decoderow = tif->tif_decodestrip = tif->tif_decodetile =
        downsampled_output ? OJPEGDecodeRawContig : OJPEGDecode;
    return 1;
#   undef td
  }

static int
OJPEGPreDecode(register TIFF *tif,tsample_t s)
  { register OJPEGState *sp = OJState(tif);
#   define td (&tif->tif_dir)

 /* If we are about to read the first row of an image plane (hopefully, these
    are coincident with JPEG "scans"!), reset the JPEG Library's decompressor
    appropriately.  Otherwise, let the decompressor run "as is" and return a
    "success" status without further ado.
 */
    if (     (isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip)
           % td->td_stripsperimage
        == 0
       )
      {
        if (   sp->cinfo.d.buffered_image
            && !CALLJPEG(sp,0,jpeg_start_output(&sp->cinfo.d,s+1))
           ) return 0;
        sp->cinfo.d.output_scanline = 0;

     /* Mark subsampling buffers "empty". */

#       ifdef D_LOSSLESS_SUPPORTED
        sp->scancount = sp->cinfo.d.min_codec_data_unit;
#       else
        sp->scancount = DCTSIZE;
#       endif
      };
    return 1;
#   undef td
  }

/*ARGSUSED*/ static void
OJPEGPostDecode(register TIFF *tif,tidata_t buf,tsize_t cc)
  { register OJPEGState *sp = OJState(tif);
#   define td (&tif->tif_dir)

 /* The JPEG Library decompressor has reached the end of a strip/tile.  If this
    is the end of a TIFF image "sample" (= JPEG "scan") in a file with separate
    components (color planes), then end the "scan".  If it ends the image's last
    sample/scan, then also stop the JPEG Library's decompressor.
 */
    if (sp->cinfo.d.output_scanline >= sp->cinfo.d.output_height)
      {
        if (sp->cinfo.d.buffered_image)
          CALLJPEG(sp,-1,jpeg_finish_output(&sp->cinfo.d)); /* End JPEG scan */
        if (   (isTiled(tif) ? tif->tif_curtile : tif->tif_curstrip)
            >= td->td_nstrips-1
           ) CALLJPEG(sp,0,jpeg_finish_decompress(&sp->cinfo.d));
      }
#   undef td
  }

static int
OJPEGVSetField(register TIFF *tif,ttag_t tag,va_list ap)
{
    uint32 v32;
    register OJPEGState *sp = OJState(tif);
#   define td (&tif->tif_dir)
    toff_t tiffoff=0;
    uint32 bufoff=0;
    uint32 code_count=0;
    int i2=0;
    int k2=0;

    switch (tag)
      {

     /* If a "ReferenceBlackWhite" TIFF tag appears in the file explicitly, undo
        any modified default definition that we might have installed below, then
        install the real one.
     */
        case TIFFTAG_REFERENCEBLACKWHITE   : if (td->td_refblackwhite)
                                               {
                                                 _TIFFfree(td->td_refblackwhite);
                                                 td->td_refblackwhite = 0;
                                               };
        default                            : return
                                               (*sp->vsetparent)(tif,tag,ap);

     /* BEWARE OF KLUDGE:  Some old-format JPEG-in-TIFF files, including those
                           produced by the Wang Imaging application for Micro-
        soft Windows, illegally omit a "ReferenceBlackWhite" TIFF tag, even
        though the TIFF specification's default is intended for the RGB color
        space and is inappropriate for the YCbCr color space ordinarily used for
        JPEG images.  Since many TIFF client applications request the value of
        this tag immediately after a TIFF image directory is parsed, and before
        any other code in this module receives control, we are forced to fix
        this problem very early in image-file processing.  Fortunately, legal
        TIFF files are supposed to store their tags in numeric order, so a
        mandatory "PhotometricInterpretation" tag should always appear before
        an optional "ReferenceBlackWhite" tag.  Hence, we slyly peek ahead when
        we discover the desired photometry, by installing modified black and
        white reference levels.
     */
        case TIFFTAG_PHOTOMETRIC           :
          if (   (v32 = (*sp->vsetparent)(tif,tag,ap))
              && td->td_photometric == PHOTOMETRIC_YCBCR
             )
	  {
            if ( (td->td_refblackwhite = _TIFFmalloc(6*sizeof(float))) )
              { register long top = 1 << td->td_bitspersample;

                td->td_refblackwhite[0] = 0;
                td->td_refblackwhite[1] = td->td_refblackwhite[3] =
                td->td_refblackwhite[5] = top - 1;
                td->td_refblackwhite[2] = td->td_refblackwhite[4] = top >> 1;
              }
            else
              {
                TIFFError(tif->tif_name,
                  "Cannot set default reference black and white levels");
                v32 = 0;
              };
	  }
          return v32;

     /* BEWARE OF KLUDGE:  According to Charles Auer <Bumble731@msn.com>, if our
                           input is a multi-image (multi-directory) JPEG-in-TIFF
        file is produced by the Wang Imaging application on Microsoft Windows,
        for some reason the first directory excludes the vendor-specific "WANG
        PageControl" tag (32934) that we check below, so the only other way to
        identify these directories is apparently to look for a software-
        identification tag with the substring, "Wang Labs".  Single-image files
        can apparently pass both tests, which causes no harm here, but what a
        mess this is!
     */
        case TIFFTAG_SOFTWARE              :
        {
            char *software;

            v32 = (*sp->vsetparent)(tif,tag,ap);
            if( TIFFGetField( tif, TIFFTAG_SOFTWARE, &software )
                && strstr( software, "Wang Labs" ) )
                sp->is_WANG = 1;
            return v32;
        }

        case TIFFTAG_JPEGPROC              :
        case TIFFTAG_JPEGIFOFFSET          :
        case TIFFTAG_JPEGIFBYTECOUNT       :
        case TIFFTAG_JPEGRESTARTINTERVAL   :
        case TIFFTAG_JPEGLOSSLESSPREDICTORS:
        case TIFFTAG_JPEGPOINTTRANSFORM    :
        case TIFFTAG_JPEGQTABLES           :
        case TIFFTAG_JPEGDCTABLES          :
        case TIFFTAG_JPEGACTABLES          :
        case TIFFTAG_WANG_PAGECONTROL      :
        case TIFFTAG_JPEGCOLORMODE         : ;
      };
    v32 = va_arg(ap,uint32); /* No. of values in this TIFF record */

    /* This switch statement is added for OJPEGVSetField */
    if(v32 !=0){
        switch(tag){
            case TIFFTAG_JPEGPROC:
                sp->jpegproc=v32;
                break;
            case TIFFTAG_JPEGIFOFFSET:
                sp->jpegifoffset=v32;
		break;
            case TIFFTAG_JPEGIFBYTECOUNT:
		sp->jpegifbytecount=v32;
		break;
            case TIFFTAG_JPEGRESTARTINTERVAL:
		sp->jpegrestartinterval=v32;
		break;
            case TIFFTAG_JPEGLOSSLESSPREDICTORS:
		sp->jpeglosslesspredictors_length=v32;
		break;
            case TIFFTAG_JPEGPOINTTRANSFORM:
		sp->jpegpointtransform_length=v32;
		break;
            case TIFFTAG_JPEGQTABLES:
		sp->jpegqtables_length=v32;
		break;
            case TIFFTAG_JPEGACTABLES:
		sp->jpegactables_length=v32;
		break;
            case TIFFTAG_JPEGDCTABLES:
		sp->jpegdctables_length=v32;
		break;
            default:
		break;
        }
    }

 /* BEWARE:  The following actions apply only if we are reading a "source" TIFF
             image to be decompressed for a client application program.  If we
    ever enhance this file's CODEC to write "destination" JPEG-in-TIFF images,
    we'll need an "if"- and another "switch"-statement below, because we'll
    probably want to store these records' values in some different places.  Most
    of these need not be parsed here in order to decode JPEG bit stream, so we
    set boolean flags to note that they have been seen, but we otherwise ignore
    them.
 */
    switch (tag)
      { JHUFF_TBL **h;

     /* Validate the JPEG-process code. */

        case TIFFTAG_JPEGPROC              :
          switch (v32)
            {
              default               : TIFFError(tif->tif_name,
                                        "Unknown JPEG process");
                                      return 0;
#             ifdef C_LOSSLESS_SUPPORTED

           /* Image uses (lossy) baseline sequential coding. */

              case JPEGPROC_BASELINE: sp->cinfo.d.process = JPROC_SEQUENTIAL;
                                      sp->cinfo.d.data_unit = DCTSIZE;
                                      break;

           /* Image uses (lossless) Huffman coding. */

              case JPEGPROC_LOSSLESS: sp->cinfo.d.process = JPROC_LOSSLESS;
                                      sp->cinfo.d.data_unit = 1;
#             else /* not C_LOSSLESS_SUPPORTED */
              case JPEGPROC_LOSSLESS: TIFFError(JPEGLib_name,
                                        "Does not support lossless Huffman coding");
                                      return 0;
              case JPEGPROC_BASELINE: ;
#             endif /* C_LOSSLESS_SUPPORTED */
            };
          break;

     /* The TIFF Version 6.0 specification says that if the value of a TIFF
        "JPEGInterchangeFormat" record is 0, then we are to behave as if this
        record were absent; i.e., the data does *not* represent a JPEG Inter-
        change Format File (JFIF), so don't even set the boolean "I've been
        here" flag below.  Otherwise, the field's value represents the file
        offset of the JPEG SOI marker.
     */
        case TIFFTAG_JPEGIFOFFSET          :
          if (v32)
            {
              sp->src.next_input_byte = tif->tif_base + v32;
              break;
            };
          return 1;
        case TIFFTAG_JPEGIFBYTECOUNT       :
          sp->src.bytes_in_buffer = v32;
          break;

     /* The TIFF Version 6.0 specification says that if the JPEG "Restart"
        marker interval is 0, then the data has no "Restart" markers; i.e., we
        must behave as if this TIFF record were absent.  So, don't even set the
        boolean "I've been here" flag below.
     */
     /*
      * Instead, set the field bit so TIFFGetField can get whether or not
      * it was set.
      */
        case TIFFTAG_JPEGRESTARTINTERVAL   :
          if (v32)
              sp->cinfo.d.restart_interval = v32;
              break;
     /* The TIFF Version 6.0 specification says that this tag is supposed to be
        a vector containing a value for each image component, but for lossless
        Huffman coding (the only JPEG process defined by the specification for
        which this tag should be needed), ISO IS 10918-1 uses only a single
        value, equivalent to the "Ss" field in a JPEG bit-stream's "Start of
        Scan" (SOS) marker.  So, we extract the first vector element and ignore
        the rest.  (I hope this is correct!)
     */
        case TIFFTAG_JPEGLOSSLESSPREDICTORS:
           if (v32)
             {
               sp->cinfo.d.Ss = *va_arg(ap,uint16 *);
               sp->jpeglosslesspredictors = 
		    _TIFFmalloc(sp->jpeglosslesspredictors_length
				* sizeof(uint16));
               if(sp->jpeglosslesspredictors==NULL){return(0);}
               for(i2=0;i2<sp->jpeglosslesspredictors_length;i2++){
                ((uint16*)sp->jpeglosslesspredictors)[i2] =
			((uint16*)sp->cinfo.d.Ss)[i2];
               }
               sp->jpeglosslesspredictors_length*=sizeof(uint16);
               break;
             };
           return v32;

     /* The TIFF Version 6.0 specification says that this tag is supposed to be
        a vector containing a value for each image component, but for lossless
        Huffman coding (the only JPEG process defined by the specification for
        which this tag should be needed), ISO IS 10918-1 uses only a single
        value, equivalent to the "Al" field in a JPEG bit-stream's "Start of
        Scan" (SOS) marker.  So, we extract the first vector element and ignore
        the rest.  (I hope this is correct!)
     */
        case TIFFTAG_JPEGPOINTTRANSFORM    :
           if (v32)
             {
               sp->cinfo.d.Al = *va_arg(ap,uint16 *);
               sp->jpegpointtransform =
		    _TIFFmalloc(sp->jpegpointtransform_length*sizeof(uint16));
               if(sp->jpegpointtransform==NULL){return(0);}
               for(i2=0;i2<sp->jpegpointtransform_length;i2++) {
                ((uint16*)sp->jpegpointtransform)[i2] =
			((uint16*)sp->cinfo.d.Al)[i2];
               }
               sp->jpegpointtransform_length*=sizeof(uint16);
               break;
             }
           return v32;

     /* We have a vector of offsets to quantization tables, so load 'em! */

        case TIFFTAG_JPEGQTABLES           :
          if (v32)
            { uint32 *v;
              int i;
              if (v32 > NUM_QUANT_TBLS)
                {
                  TIFFError(tif->tif_name,"Too many quantization tables");
                  return 0;
                };
              i = 0;
              v = va_arg(ap,uint32 *);
                sp->jpegqtables=_TIFFmalloc(64*sp->jpegqtables_length);
                if(sp->jpegqtables==NULL){return(0);}
                tiffoff = TIFFSeekFile(tif, 0, SEEK_CUR);
                bufoff=0;
                for(i2=0;i2<sp->jpegqtables_length;i2++){
                    TIFFSeekFile(tif, v[i2], SEEK_SET);
                    TIFFReadFile(tif, &(((unsigned char*)(sp->jpegqtables))[bufoff]),
				 64);
                    bufoff+=64;
                }
                sp->jpegqtables_length=bufoff;
                TIFFSeekFile(tif, tiffoff, SEEK_SET);

              do /* read quantization table */
                { register UINT8 *from = tif->tif_base + *v++;
                  register UINT16 *to;
                  register int j = DCTSIZE2;

                  if (!( sp->cinfo.d.quant_tbl_ptrs[i]
                       = CALLJPEG(sp,0,jpeg_alloc_quant_table(&sp->cinfo.comm))
                       )
                     )
                    {
                      TIFFError(JPEGLib_name,"No space for quantization table");
                      return 0;
                    };
                  to = sp->cinfo.d.quant_tbl_ptrs[i]->quantval;
                  do *to++ = *from++; while (--j > 0);
                }
              while (++i < v32);
              sp->jpegtablesmode |= JPEGTABLESMODE_QUANT;
            };
          break;

     /* We have a vector of offsets to DC Huffman tables, so load 'em! */

        case TIFFTAG_JPEGDCTABLES          :
          h = sp->cinfo.d.dc_huff_tbl_ptrs;
          goto L;

     /* We have a vector of offsets to AC Huffman tables, so load 'em! */

        case TIFFTAG_JPEGACTABLES          :
          h = sp->cinfo.d.ac_huff_tbl_ptrs;
       L: if (v32)
            { uint32 *v;
              int i;
              if (v32 > NUM_HUFF_TBLS)
                {
                  TIFFError(tif->tif_name,"Too many Huffman tables");
                  return 0;
                };
              v = va_arg(ap,uint32 *);
                if(tag == TIFFTAG_JPEGDCTABLES) {
                    sp->jpegdctables=_TIFFmalloc(272*sp->jpegdctables_length);
                    if(sp->jpegdctables==NULL){return(0);}
                    tiffoff = TIFFSeekFile(tif, 0, SEEK_CUR);
                    bufoff=0;
                    code_count=0;                
                    for(i2=0;i2<sp->jpegdctables_length;i2++){
                        TIFFSeekFile(tif, v[i2], SEEK_SET);
                        TIFFReadFile(tif,
				     &(((unsigned char*)(sp->jpegdctables))[bufoff]),
				     16);
                        code_count=0;
                        for(k2=0;k2<16;k2++){
                            code_count+=((unsigned char*)(sp->jpegdctables))[k2+bufoff];
                        }
                        TIFFReadFile(tif,
				     &(((unsigned char*)(sp->jpegdctables))[bufoff+16]),
				     code_count);
                        bufoff+=16;
                        bufoff+=code_count;
                    }
                    sp->jpegdctables_length=bufoff;
                    TIFFSeekFile(tif, tiffoff, SEEK_SET);
                }
                if(tag==TIFFTAG_JPEGACTABLES){
                    sp->jpegactables=_TIFFmalloc(272*sp->jpegactables_length);
                    if(sp->jpegactables==NULL){return(0);}
                    tiffoff = TIFFSeekFile(tif, 0, SEEK_CUR);
                    bufoff=0;
                    code_count=0;                
                    for(i2=0;i2<sp->jpegactables_length;i2++){
                        TIFFSeekFile(tif, v[i2], SEEK_SET);
                        TIFFReadFile(tif, &(((unsigned char*)(sp->jpegactables))[bufoff]), 16);
                        code_count=0;
                        for(k2=0;k2<16;k2++){
                            code_count+=((unsigned char*)(sp->jpegactables))[k2+bufoff];
                        }
                        TIFFReadFile(tif, &(((unsigned char*)(sp->jpegactables))[bufoff+16]), code_count);
                        bufoff+=16;
                        bufoff+=code_count;
                    }
                    sp->jpegactables_length=bufoff;
                    TIFFSeekFile(tif, tiffoff, SEEK_SET);
                }
              i = 0;
              do /* copy each Huffman table */
                { int size = 0;
                  register UINT8 *from = tif->tif_base + *v++, *to;
                  register int j = sizeof (*h)->bits;

               /* WARNING:  This code relies on the fact that an image file not
                            "memory mapped" was read entirely into a single
                  buffer by "TIFFInitOJPEG()", so we can do a fast memory-to-
                  memory copy here.  Each table consists of 16 Bytes, which are
                  suffixed to a 0 Byte when copied, followed by a variable
                  number of Bytes whose length is the sum of the first 16.
               */
                  if (!( *h
                       = CALLJPEG(sp,0,jpeg_alloc_huff_table(&sp->cinfo.comm))
                       )
                     )
                    {
                      TIFFError(JPEGLib_name,"No space for Huffman table");
                      return 0;
                    };
                  to = (*h++)->bits;
                  *to++ = 0;
                  while (--j > 0) size += *to++ = *from++; /* Copy 16 Bytes */
                  if (size > sizeof (*h)->huffval/sizeof *(*h)->huffval)
                    {
                      TIFFError(tif->tif_name,"Huffman table too big");
                      return 0;
                    };
                  if ((j = size) > 0) do *to++ = *from++; while (--j > 0);
                  while (++size <= sizeof (*h)->huffval/sizeof *(*h)->huffval)
                    *to++ = 0; /* Zero the rest of the table for cleanliness */
                }
              while (++i < v32);
              sp->jpegtablesmode |= JPEGTABLESMODE_HUFF;
            };
          break;

     /* The following vendor-specific TIFF tag occurs in (highly illegal) files
        produced by the Wang Imaging application for Microsoft Windows.  These
        can apparently have several "pages", in which case this tag specifies
        the offset of a "page control" structure, which we don't currently know
        how to handle.  0 indicates a 1-page image with no "page control", which
        we make a feeble effort to handle.
     */
        case TIFFTAG_WANG_PAGECONTROL      :
          if (v32 == 0) v32 = -1;
          sp->is_WANG = v32;
          tag = TIFFTAG_JPEGPROC+FIELD_WANG_PAGECONTROL-FIELD_JPEGPROC;
          break;

     /* This pseudo tag indicates whether our caller is expected to do YCbCr <->
        RGB color-space conversion (JPEGCOLORMODE_RAW <=> 0) or whether we must
        ask the JPEG Library to do it (JPEGCOLORMODE_RGB <=> 1).
     */
        case TIFFTAG_JPEGCOLORMODE         :
          sp->jpegcolormode = v32;

       /* Mark the image to indicate whether returned data is up-sampled, so
          that "TIFF{Strip,Tile}Size()" reflect the true amount of data present.
       */
          v32 = tif->tif_flags; /* Save flags temporarily */
          tif->tif_flags &= ~TIFF_UPSAMPLED;
          if (   td->td_photometric == PHOTOMETRIC_YCBCR
              &&    (td->td_ycbcrsubsampling[0]<<3 | td->td_ycbcrsubsampling[1])
                 != 011
              && sp->jpegcolormode == JPEGCOLORMODE_RGB
             ) tif->tif_flags |= TIFF_UPSAMPLED;

       /* If the up-sampling state changed, re-calculate tile size. */

          if ((tif->tif_flags ^ v32) & TIFF_UPSAMPLED)
            {
              tif->tif_tilesize = isTiled(tif) ? TIFFTileSize(tif) : (tsize_t) -1;
              tif->tif_flags |= TIFF_DIRTYDIRECT;
            };
          return 1;
      };
    TIFFSetFieldBit(tif,tag-TIFFTAG_JPEGPROC+FIELD_JPEGPROC);
    return 1;
#   undef td
  }

static int
OJPEGVGetField(register TIFF *tif,ttag_t tag,va_list ap)
  { register OJPEGState *sp = OJState(tif);

    switch (tag)
      {

     /* If this file has managed to synthesize a set of consolidated "metadata"
        tables for the current (post-TIFF Version 6.0 specification) JPEG-in-
        TIFF encapsulation strategy, then tell our caller about them; otherwise,
        keep mum.
     */
        case TIFFTAG_JPEGTABLES            :
          if (sp->jpegtables_length) /* we have "new"-style JPEG tables */
            {
              *va_arg(ap,uint32 *) = sp->jpegtables_length;
              *va_arg(ap,char **) = sp->jpegtables;
              return 1;
            };

     /* This pseudo tag indicates whether our caller is expected to do YCbCr <->
        RGB color-space conversion (JPEGCOLORMODE_RAW <=> 0) or whether we must
        ask the JPEG Library to do it (JPEGCOLORMODE_RGB <=> 1).
     */
        case TIFFTAG_JPEGCOLORMODE         :
          *va_arg(ap,uint32 *) = sp->jpegcolormode;
          return 1;

     /* The following tags are defined by the TIFF Version 6.0 specification
        and are obsolete.  If our caller asks for information about them, do not
        return anything, even if we parsed them in an old-format "source" image.
     */
        case TIFFTAG_JPEGPROC              :
		*va_arg(ap, uint16*)=sp->jpegproc;
		return(1);
		break;
        case TIFFTAG_JPEGIFOFFSET          :
		*va_arg(ap, uint32*)=sp->jpegifoffset;
		return(1);
		break;
        case TIFFTAG_JPEGIFBYTECOUNT       :
		*va_arg(ap, uint32*)=sp->jpegifbytecount;
		return(1);
		break;
        case TIFFTAG_JPEGRESTARTINTERVAL   :
		*va_arg(ap, uint32*)=sp->jpegrestartinterval;
		return(1);
		break;
        case TIFFTAG_JPEGLOSSLESSPREDICTORS:
                *va_arg(ap, uint32*)=sp->jpeglosslesspredictors_length;
                *va_arg(ap, void**)=sp->jpeglosslesspredictors;
                return(1);
                break;
        case TIFFTAG_JPEGPOINTTRANSFORM    :
                *va_arg(ap, uint32*)=sp->jpegpointtransform_length;
                *va_arg(ap, void**)=sp->jpegpointtransform;
                return(1);
                break;
        case TIFFTAG_JPEGQTABLES           :
                *va_arg(ap, uint32*)=sp->jpegqtables_length;
                *va_arg(ap, void**)=sp->jpegqtables;
                return(1);
                break;
        case TIFFTAG_JPEGDCTABLES          :
                *va_arg(ap, uint32*)=sp->jpegdctables_length;
                *va_arg(ap, void**)=sp->jpegdctables;
                return(1);
                break;
        case TIFFTAG_JPEGACTABLES          : 
                *va_arg(ap, uint32*)=sp->jpegactables_length;
                *va_arg(ap, void**)=sp->jpegactables;
                return(1);
                break;
      };
    return (*sp->vgetparent)(tif,tag,ap);
  }

static void
OJPEGPrintDir(register TIFF *tif,FILE *fd,long flags)
  { register OJPEGState *sp = OJState(tif);

    if (   ( flags
           & (TIFFPRINT_JPEGQTABLES|TIFFPRINT_JPEGDCTABLES|TIFFPRINT_JPEGACTABLES)
           )
        && sp->jpegtables_length
       )
      fprintf(fd,"  JPEG Table Data: <present>, %lu bytes\n",
        sp->jpegtables_length);
  }

static uint32
OJPEGDefaultStripSize(register TIFF *tif,register uint32 s)
  { register OJPEGState *sp = OJState(tif);
#   define td (&tif->tif_dir)

    if ((s = (*sp->defsparent)(tif,s)) < td->td_imagelength)
      { register tsize_t size = sp->cinfo.comm.is_decompressor
#                             ifdef D_LOSSLESS_SUPPORTED
                              ? sp->cinfo.d.min_codec_data_unit
#                             else
                              ? DCTSIZE
#                             endif
#                             ifdef C_LOSSLESS_SUPPORTED
                              : sp->cinfo.c.data_unit;
#                             else
                              : DCTSIZE;
#                             endif

        size = TIFFroundup(size,16);
        s = TIFFroundup(s,td->td_ycbcrsubsampling[1]*size);
      };
    return s;
#   undef td
  }

static void
OJPEGDefaultTileSize(register TIFF *tif,register uint32 *tw,register uint32 *th)
  { register OJPEGState *sp = OJState(tif);
    register tsize_t size;
#   define td (&tif->tif_dir)

    size = sp->cinfo.comm.is_decompressor
#        ifdef D_LOSSLESS_SUPPORTED
         ? sp->cinfo.d.min_codec_data_unit
#        else
         ? DCTSIZE
#        endif
#        ifdef C_LOSSLESS_SUPPORTED
         : sp->cinfo.c.data_unit;
#        else
         : DCTSIZE;
#        endif
    size = TIFFroundup(size,16);
    (*sp->deftparent)(tif,tw,th);
    *tw = TIFFroundup(*tw,td->td_ycbcrsubsampling[0]*size);
    *th = TIFFroundup(*th,td->td_ycbcrsubsampling[1]*size);
#   undef td
  }

static void
OJPEGCleanUp(register TIFF *tif)
  { register OJPEGState *sp;

    if ( (sp = OJState(tif)) )
      {
        CALLVJPEG(sp,jpeg_destroy(&sp->cinfo.comm)); /* Free JPEG Lib. vars. */
        if (sp->jpegtables) {_TIFFfree(sp->jpegtables);sp->jpegtables=0;}
        if (sp->jpeglosslesspredictors) {
		_TIFFfree(sp->jpeglosslesspredictors);
		sp->jpeglosslesspredictors = 0;
	}
        if (sp->jpegpointtransform) {
		_TIFFfree(sp->jpegpointtransform);
		sp->jpegpointtransform=0;
	}
        if (sp->jpegqtables) {_TIFFfree(sp->jpegqtables);sp->jpegqtables=0;}
        if (sp->jpegactables) {_TIFFfree(sp->jpegactables);sp->jpegactables=0;}
        if (sp->jpegdctables) {_TIFFfree(sp->jpegdctables);sp->jpegdctables=0;}
     /* If the image file isn't "memory mapped" and we read it all into a
        single, large memory buffer, free the buffer now.
     */
        if (!isMapped(tif) && tif->tif_base) /* free whole-file buffer */
          {
            _TIFFfree(tif->tif_base);
            tif->tif_base = 0;
            tif->tif_size = 0;
          };
        _TIFFfree(sp); /* Release local variables */
        tif->tif_data = 0;
      }
  }

int
TIFFInitOJPEG(register TIFF *tif,int scheme)
  { register OJPEGState *sp;
#   define td (&tif->tif_dir)
#   ifndef never

 /* This module supports a decompression-only CODEC, which is intended strictly
    for viewing old image files using the obsolete JPEG-in-TIFF encapsulation
    specified by the TIFF Version 6.0 specification.  It does not, and never
    should, support compression for new images.  If a client application asks us
    to, refuse and complain loudly!
 */
    if (tif->tif_mode != O_RDONLY) return _notSupported(tif);
#   endif /* never */
    if (!isMapped(tif))
      {

     /* BEWARE OF KLUDGE:  If our host operating-system doesn't let an image
                           file be "memory mapped", then we want to read the
        entire file into a single (possibly large) memory buffer as if it had
        been "memory mapped".  Although this is likely to waste space, because
        analysis of the file's content might cause parts of it to be read into
        smaller buffers duplicatively, it appears to be the lesser of several
        evils.  Very old JPEG-in-TIFF encapsulations aren't guaranteed to be
        JFIF bit streams, or to have a TIFF "JPEGTables" record or much other
        "metadata" to help us locate the decoding tables and entropy-coded data,
        so we're likely do a lot of random-access grokking around, and we must
        ultimately tell the JPEG Library to sequentially scan much of the file
        anyway.  This is all likely to be easier if we use "brute force" to
        read the entire file, once, and don't use incremental disc I/O.  If our
        client application tries to process a file so big that we can't buffer
        it entirely, then tough shit: we'll give up and exit!
     */
        if (!(tif->tif_base = _TIFFmalloc(tif->tif_size=TIFFGetFileSize(tif))))
          {
            TIFFError(tif->tif_name,"Cannot allocate file buffer");
            return 0;
          };
        if (!SeekOK(tif,0) || !ReadOK(tif,tif->tif_base,tif->tif_size))
          {
            TIFFError(tif->tif_name,"Cannot read file");
            return 0;
          }
      };

 /* Allocate storage for this module's per-file variables. */

    if (!(tif->tif_data = (tidata_t)_TIFFmalloc(sizeof *sp)))
      {
        TIFFError("TIFFInitOJPEG","No space for JPEG state block");
        return 0;
      };
    (sp = OJState(tif))->tif = tif; /* Initialize reverse pointer */
    sp->cinfo.d.err = jpeg_std_error(&sp->err); /* Initialize error handling */
    sp->err.error_exit = TIFFojpeg_error_exit;
    sp->err.output_message = TIFFojpeg_output_message;
    if (!CALLVJPEG(sp,jpeg_create_decompress(&sp->cinfo.d))) return 0;

 /* Install CODEC-specific tag information and override default TIFF Library
    "method" subroutines with our own, CODEC-specific methods.  Like all good
    members of an object-class, we save some of these subroutine pointers for
    "fall back" in case our own methods fail.
 */
    _TIFFMergeFieldInfo(tif,ojpegFieldInfo,
      sizeof ojpegFieldInfo/sizeof *ojpegFieldInfo);
    sp->defsparent = tif->tif_defstripsize;
    sp->deftparent = tif->tif_deftilesize;
    sp->vgetparent = tif->tif_tagmethods.vgetfield;
    sp->vsetparent = tif->tif_tagmethods.vsetfield;
    tif->tif_defstripsize = OJPEGDefaultStripSize;
    tif->tif_deftilesize = OJPEGDefaultTileSize;
    tif->tif_tagmethods.vgetfield = OJPEGVGetField;
    tif->tif_tagmethods.vsetfield = OJPEGVSetField;
    tif->tif_tagmethods.printdir = OJPEGPrintDir;
#   ifdef never
    tif->tif_setupencode = OJPEGSetupEncode;
    tif->tif_preencode = OJPEGPreEncode;
    tif->tif_postencode = OJPEGPostEncode;
#   else /* well, hardly ever */
    tif->tif_setupencode = tif->tif_postencode = _notSupported;
    tif->tif_preencode = (TIFFPreMethod)_notSupported;
#   endif /* never */
    tif->tif_setupdecode = OJPEGSetupDecode;
    tif->tif_predecode = OJPEGPreDecode;
    tif->tif_postdecode = OJPEGPostDecode;
    tif->tif_cleanup = OJPEGCleanUp;

 /* If the image file doesn't have "JPEGInterchangeFormat[Length]" TIFF records
    to guide us, we have few clues about where its encapsulated JPEG bit stream
    is located, so establish intelligent defaults:  If the Image File Directory
    doesn't immediately follow the TIFF header, assume that the JPEG data lies
    in between; otherwise, assume that it follows the Image File Directory.
 */
    if (tif->tif_header.tiff_diroff > sizeof tif->tif_header)
      {
        sp->src.next_input_byte = tif->tif_base + sizeof tif->tif_header;
        sp->src.bytes_in_buffer = tif->tif_header.tiff_diroff
                                - sizeof tif->tif_header;
      }
    else /* this case is ugly! */
      { uint32 maxoffset = tif->tif_size;
        uint16 dircount;

     /* Calculate the offset to the next Image File Directory, if there is one,
        or to the end of the file, if not.  Then arrange to read the file from
        the end of the Image File Directory to that offset.
     */
        if (tif->tif_nextdiroff) maxoffset = tif->tif_nextdiroff; /* Not EOF */
        _TIFFmemcpy(&dircount,(const tdata_t)
          (sp->src.next_input_byte = tif->tif_base+tif->tif_header.tiff_diroff),
          sizeof dircount);
        if (tif->tif_flags & TIFF_SWAB) TIFFSwabShort(&dircount);
        sp->src.next_input_byte += dircount*sizeof(TIFFDirEntry)
                                + sizeof maxoffset + sizeof dircount;
        sp->src.bytes_in_buffer = tif->tif_base - sp->src.next_input_byte
                                + maxoffset;
      };

 /* IJG JPEG Library Version 6B can be configured for either 8- or 12-bit sample
    precision, but we assume that "old JPEG" TIFF clients only need 8 bits.
 */
    sp->cinfo.d.data_precision = 8;
#   ifdef C_LOSSLESS_SUPPORTED

 /* If the "JPEGProc" TIFF tag is missing from the Image File Dictionary, the
    JPEG Library will use its (lossy) baseline sequential process by default.
 */
    sp->cinfo.d.data_unit = DCTSIZE;
#   endif /* C_LOSSLESS_SUPPORTED */

 /* Initialize other CODEC-specific variables requiring default values. */

    tif->tif_flags |= TIFF_NOBITREV; /* No bit-reversal within data bytes */
    sp->h_sampling = sp->v_sampling = 1; /* No subsampling by default */
    sp->is_WANG = 0; /* Assume not a MS Windows Wang Imaging file by default */
    sp->jpegtables = 0; /* No "new"-style JPEG tables synthesized yet */
    sp->jpegtables_length = 0;
    sp->jpegquality = 75; /* Default IJG quality */
    sp->jpegcolormode = JPEGCOLORMODE_RAW;
    sp->jpegtablesmode = 0; /* No tables found yet */
    sp->jpeglosslesspredictors=0;
    sp->jpeglosslesspredictors_length=0;
    sp->jpegpointtransform=0;
    sp->jpegpointtransform_length=0;
    sp->jpegqtables=0;
    sp->jpegqtables_length=0;
    sp->jpegdctables=0;
    sp->jpegdctables_length=0;
    sp->jpegactables=0;
    sp->jpegactables_length=0;
    return 1;
#   undef td
  }
#endif /* OJPEG_SUPPORT */

/* vim: set ts=8 sts=8 sw=8 noet: */
