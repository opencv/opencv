/* WARNING: The type of JPEG encapsulation defined by the TIFF Version 6.0
   specification is now totally obsolete and deprecated for new applications and
   images. This file was was created solely in order to read unconverted images
   still present on some users' computer systems. It will never be extended
   to write such files. Writing new-style JPEG compressed TIFFs is implemented
   in tif_jpeg.c.

   The code is carefully crafted to robustly read all gathered JPEG-in-TIFF
   testfiles, and anticipate as much as possible all other... But still, it may
   fail on some. If you encounter problems, please report them on the TIFF
   mailing list and/or to Joris Van Damme <info@awaresystems.be>.

   Please read the file called "TIFF Technical Note #2" if you need to be
   convinced this compression scheme is bad and breaks TIFF. That document
   is linked to from the LibTiff site <http://www.remotesensing.org/libtiff/>
   and from AWare Systems' TIFF section
   <http://www.awaresystems.be/imaging/tiff.html>. It is also absorbed
   in Adobe's specification supplements, marked "draft" up to this day, but
   supported by the TIFF community.

   This file interfaces with Release 6B of the JPEG Library written by the
   Independent JPEG Group. Previous versions of this file required a hack inside
   the LibJpeg library. This version no longer requires that. Remember to
   remove the hack if you update from the old version.

   Copyright (c) Joris Van Damme <info@awaresystems.be>
   Copyright (c) AWare Systems <http://www.awaresystems.be/>

   The licence agreement for this file is the same as the rest of the LibTiff
   library.

   IN NO EVENT SHALL JORIS VAN DAMME OR AWARE SYSTEMS BE LIABLE FOR
   ANY SPECIAL, INCIDENTAL, INDIRECT OR CONSEQUENTIAL DAMAGES OF ANY KIND,
   OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
   WHETHER OR NOT ADVISED OF THE POSSIBILITY OF DAMAGE, AND ON ANY THEORY OF
   LIABILITY, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
   OF THIS SOFTWARE.

   Joris Van Damme and/or AWare Systems may be available for custom
   development. If you like what you see, and need anything similar or related,
   contact <info@awaresystems.be>.
*/

/* What is what, and what is not?

   This decoder starts with an input stream, that is essentially the
   JpegInterchangeFormat stream, if any, followed by the strile data, if any.
   This stream is read in OJPEGReadByte and related functions.

   It analyzes the start of this stream, until it encounters non-marker data,
   i.e. compressed image data. Some of the header markers it sees have no actual
   content, like the SOI marker, and APP/COM markers that really shouldn't even
   be there. Some other markers do have content, and the valuable bits and
   pieces of information in these markers are saved, checking all to verify that
   the stream is more or less within expected bounds. This happens inside the
   OJPEGReadHeaderInfoSecStreamXxx functions.

   Some OJPEG imagery contains no valid JPEG header markers. This situation is
   picked up on if we've seen no SOF marker when we're at the start of the
   compressed image data. In this case, the tables are read from JpegXxxTables
   tags, and the other bits and pieces of information is initialized to its most
   basic value. This is implemented in the OJPEGReadHeaderInfoSecTablesXxx
   functions.

   When this is complete, a good and valid JPEG header can be assembled, and
   this is passed through to LibJpeg. When that's done, the remainder of the
   input stream, i.e. the compressed image data, can be passed through
   unchanged. This is done in OJPEGWriteStream functions.

   LibTiff rightly expects to know the subsampling values before decompression.
   Just like in new-style JPEG-in-TIFF, though, or even more so, actually, the
   YCbCrsubsampling tag is notoriously unreliable. To correct these tag values
   with the ones inside the JPEG stream, the first part of the input stream is
   pre-scanned in OJPEGSubsamplingCorrect, making no note of any other data,
   reporting no warnings or errors, up to the point where either these values
   are read, or it's clear they aren't there. This means that some of the data
   is read twice, but we feel speed in correcting these values is important
   enough to warrant this sacrifice. Although there is currently no define or
   other configuration mechanism to disable this behavior, the actual header
   scanning is build to robustly respond with error report if it should
   encounter an uncorrected mismatch of subsampling values. See
   OJPEGReadHeaderInfoSecStreamSof.

   The restart interval and restart markers are the most tricky part... The
   restart interval can be specified in a tag. It can also be set inside the
   input JPEG stream. It can be used inside the input JPEG stream. If reading
   from strile data, we've consistently discovered the need to insert restart
   markers in between the different striles, as is also probably the most likely
   interpretation of the original TIFF 6.0 specification. With all this setting
   of interval, and actual use of markers that is not predictable at the time of
   valid JPEG header assembly, the restart thing may turn out the Achilles heel
   of this implementation. Fortunately, most OJPEG writer vendors succeed in
   reading back what they write, which may be the reason why we've been able to
   discover ways that seem to work.

   Some special provision is made for planarconfig separate OJPEG files. These
   seem to consistently contain header info, a SOS marker, a plane, SOS marker,
   plane, SOS, and plane. This may or may not be a valid JPEG configuration, we
   don't know and don't care. We want LibTiff to be able to access the planes
   individually, without huge buffering inside LibJpeg, anyway. So we compose
   headers to feed to LibJpeg, in this case, that allow us to pass a single
   plane such that LibJpeg sees a valid single-channel JPEG stream. Locating
   subsequent SOS markers, and thus subsequent planes, is done inside
   OJPEGReadSecondarySos.

   The benefit of the scheme is... that it works, basically. We know of no other
   that does. It works without checking software tag, or otherwise going about
   things in an OJPEG flavor specific manner. Instead, it is a single scheme,
   that covers the cases with and without JpegInterchangeFormat, with and
   without striles, with part of the header in JpegInterchangeFormat and
   remainder in first strile, etc. It is forgiving and robust, may likely work
   with OJPEG flavors we've not seen yet, and makes most out of the data.

   Another nice side-effect is that a complete JPEG single valid stream is build
   if planarconfig is not separate (vast majority). We may one day use that to
   build converters to JPEG, and/or to new-style JPEG compression inside TIFF.

   A disadvantage is the lack of random access to the individual striles. This
   is the reason for much of the complicated restart-and-position stuff inside
   OJPEGPreDecode. Applications would do well accessing all striles in order, as
   this will result in a single sequential scan of the input stream, and no
   restarting of LibJpeg decoding session.
*/

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN

#include "tiffiop.h"
#ifdef OJPEG_SUPPORT

/* Configuration defines here are:
 * JPEG_ENCAP_EXTERNAL: The normal way to call libjpeg, uses longjump. In some
 * environments, like eg LibTiffDelphi, this is not possible. For this reason,
 * the actual calls to libjpeg, with longjump stuff, are encapsulated in
 * dedicated functions. When JPEG_ENCAP_EXTERNAL is defined, these encapsulating
 * functions are declared external to this unit, and can be defined elsewhere to
 * use stuff other then longjump. The default mode, without JPEG_ENCAP_EXTERNAL,
 * implements the call encapsulators here, internally, with normal longjump.
 * SETJMP, LONGJMP, JMP_BUF: On some machines/environments a longjump equivalent
 * is conveniently available, but still it may be worthwhile to use _setjmp or
 * sigsetjmp in place of plain setjmp. These macros will make it easier. It is
 * useless to fiddle with these if you define JPEG_ENCAP_EXTERNAL. OJPEG_BUFFER:
 * Define the size of the desired buffer here. Should be small enough so as to
 * guarantee instant processing, optimal streaming and optimal use of processor
 * cache, but also big enough so as to not result in significant call overhead.
 * It should be at least a few bytes to accommodate some structures (this is
 * verified in asserts), but it would not be sensible to make it this small
 * anyway, and it should be at most 64K since it is indexed with uint16_t. We
 * recommend 2K. EGYPTIANWALK: You could also define EGYPTIANWALK here, but it
 * is not used anywhere and has absolutely no effect. That is why most people
 * insist the EGYPTIANWALK is a bit silly.
 */

/* define LIBJPEG_ENCAP_EXTERNAL */
#define SETJMP(jbuf) setjmp(jbuf)
#define LONGJMP(jbuf, code) longjmp(jbuf, code)
#define JMP_BUF jmp_buf
#define OJPEG_BUFFER 2048
/* define EGYPTIANWALK */

#define JPEG_MARKER_SOF0 0xC0
#define JPEG_MARKER_SOF1 0xC1
#define JPEG_MARKER_SOF3 0xC3
#define JPEG_MARKER_DHT 0xC4
#define JPEG_MARKER_RST0 0XD0
#define JPEG_MARKER_SOI 0xD8
#define JPEG_MARKER_EOI 0xD9
#define JPEG_MARKER_SOS 0xDA
#define JPEG_MARKER_DQT 0xDB
#define JPEG_MARKER_DRI 0xDD
#define JPEG_MARKER_APP0 0xE0
#define JPEG_MARKER_COM 0xFE

#define FIELD_OJPEG_JPEGINTERCHANGEFORMAT (FIELD_CODEC + 0)
#define FIELD_OJPEG_JPEGINTERCHANGEFORMATLENGTH (FIELD_CODEC + 1)
#define FIELD_OJPEG_JPEGQTABLES (FIELD_CODEC + 2)
#define FIELD_OJPEG_JPEGDCTABLES (FIELD_CODEC + 3)
#define FIELD_OJPEG_JPEGACTABLES (FIELD_CODEC + 4)
#define FIELD_OJPEG_JPEGPROC (FIELD_CODEC + 5)
#define FIELD_OJPEG_JPEGRESTARTINTERVAL (FIELD_CODEC + 6)

static const TIFFField ojpegFields[] = {
    {TIFFTAG_JPEGIFOFFSET, 1, 1, TIFF_LONG8, 0, TIFF_SETGET_UINT64,
     TIFF_SETGET_UNDEFINED, FIELD_OJPEG_JPEGINTERCHANGEFORMAT, TRUE, FALSE,
     "JpegInterchangeFormat", NULL},
    {TIFFTAG_JPEGIFBYTECOUNT, 1, 1, TIFF_LONG8, 0, TIFF_SETGET_UINT64,
     TIFF_SETGET_UNDEFINED, FIELD_OJPEG_JPEGINTERCHANGEFORMATLENGTH, TRUE,
     FALSE, "JpegInterchangeFormatLength", NULL},
    {TIFFTAG_JPEGQTABLES, TIFF_VARIABLE2, TIFF_VARIABLE2, TIFF_LONG8, 0,
     TIFF_SETGET_C32_UINT64, TIFF_SETGET_UNDEFINED, FIELD_OJPEG_JPEGQTABLES,
     FALSE, TRUE, "JpegQTables", NULL},
    {TIFFTAG_JPEGDCTABLES, TIFF_VARIABLE2, TIFF_VARIABLE2, TIFF_LONG8, 0,
     TIFF_SETGET_C32_UINT64, TIFF_SETGET_UNDEFINED, FIELD_OJPEG_JPEGDCTABLES,
     FALSE, TRUE, "JpegDcTables", NULL},
    {TIFFTAG_JPEGACTABLES, TIFF_VARIABLE2, TIFF_VARIABLE2, TIFF_LONG8, 0,
     TIFF_SETGET_C32_UINT64, TIFF_SETGET_UNDEFINED, FIELD_OJPEG_JPEGACTABLES,
     FALSE, TRUE, "JpegAcTables", NULL},
    {TIFFTAG_JPEGPROC, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16,
     TIFF_SETGET_UNDEFINED, FIELD_OJPEG_JPEGPROC, FALSE, FALSE, "JpegProc",
     NULL},
    {TIFFTAG_JPEGRESTARTINTERVAL, 1, 1, TIFF_SHORT, 0, TIFF_SETGET_UINT16,
     TIFF_SETGET_UNDEFINED, FIELD_OJPEG_JPEGRESTARTINTERVAL, FALSE, FALSE,
     "JpegRestartInterval", NULL},
};

#ifndef LIBJPEG_ENCAP_EXTERNAL
#include <setjmp.h>
#endif

/* We undefine FAR to avoid conflict with JPEG definition */

#ifdef FAR
#undef FAR
#endif

/*
  Libjpeg's jmorecfg.h defines INT16 and INT32, but only if XMD_H is
  not defined.  Unfortunately, the MinGW and Borland compilers include
  a typedef for INT32, which causes a conflict.  MSVC does not include
  a conflicting typedef given the headers which are included.
*/
#if defined(__BORLANDC__) || defined(__MINGW32__)
#define XMD_H 1
#endif

/* Define "boolean" as unsigned char, not int, per Windows custom. */
#if defined(__WIN32__) && !defined(__MINGW32__)
#ifndef __RPCNDR_H__ /* don't conflict if rpcndr.h already read */
typedef unsigned char boolean;
#endif
#define HAVE_BOOLEAN /* prevent jmorecfg.h from redefining it */
#endif

#include "jerror.h"
#include "jpeglib.h"

typedef struct jpeg_error_mgr jpeg_error_mgr;
typedef struct jpeg_common_struct jpeg_common_struct;
typedef struct jpeg_decompress_struct jpeg_decompress_struct;
typedef struct jpeg_source_mgr jpeg_source_mgr;

typedef enum
{
    osibsNotSetYet,
    osibsJpegInterchangeFormat,
    osibsStrile,
    osibsEof
} OJPEGStateInBufferSource;

typedef enum
{
    ososSoi,
    ososQTable0,
    ososQTable1,
    ososQTable2,
    ososQTable3,
    ososDcTable0,
    ososDcTable1,
    ososDcTable2,
    ososDcTable3,
    ososAcTable0,
    ososAcTable1,
    ososAcTable2,
    ososAcTable3,
    ososDri,
    ososSof,
    ososSos,
    ososCompressed,
    ososRst,
    ososEoi
} OJPEGStateOutState;

typedef struct
{
    TIFF *tif;
    int decoder_ok;
    int error_in_raw_data_decoding;
#ifndef LIBJPEG_ENCAP_EXTERNAL
    JMP_BUF exit_jmpbuf;
#endif
    TIFFVGetMethod vgetparent;
    TIFFVSetMethod vsetparent;
    TIFFPrintMethod printdir;
    uint64_t file_size;
    uint32_t image_width;
    uint32_t image_length;
    uint32_t strile_width;
    uint32_t strile_length;
    uint32_t strile_length_total;
    uint8_t samples_per_pixel;
    uint8_t plane_sample_offset;
    uint8_t samples_per_pixel_per_plane;
    uint64_t jpeg_interchange_format;
    uint64_t jpeg_interchange_format_length;
    uint8_t jpeg_proc;
    uint8_t subsamplingcorrect;
    uint8_t subsamplingcorrect_done;
    uint8_t subsampling_tag;
    uint8_t subsampling_hor;
    uint8_t subsampling_ver;
    uint8_t subsampling_force_desubsampling_inside_decompression;
    uint8_t qtable_offset_count;
    uint8_t dctable_offset_count;
    uint8_t actable_offset_count;
    uint64_t qtable_offset[3];
    uint64_t dctable_offset[3];
    uint64_t actable_offset[3];
    uint8_t *qtable[4];
    uint8_t *dctable[4];
    uint8_t *actable[4];
    uint16_t restart_interval;
    uint8_t restart_index;
    uint8_t sof_log;
    uint8_t sof_marker_id;
    uint32_t sof_x;
    uint32_t sof_y;
    uint8_t sof_c[3];
    uint8_t sof_hv[3];
    uint8_t sof_tq[3];
    uint8_t sos_cs[3];
    uint8_t sos_tda[3];
    struct
    {
        uint8_t log;
        OJPEGStateInBufferSource in_buffer_source;
        uint32_t in_buffer_next_strile;
        uint64_t in_buffer_file_pos;
        uint64_t in_buffer_file_togo;
    } sos_end[3];
    uint8_t readheader_done;
    uint8_t writeheader_done;
    uint16_t write_cursample;
    uint32_t write_curstrile;
    uint8_t libjpeg_session_active;
    uint8_t libjpeg_jpeg_query_style;
    jpeg_error_mgr libjpeg_jpeg_error_mgr;
    jpeg_decompress_struct libjpeg_jpeg_decompress_struct;
    jpeg_source_mgr libjpeg_jpeg_source_mgr;
    uint8_t subsampling_convert_log;
    uint32_t subsampling_convert_ylinelen;
    uint32_t subsampling_convert_ylines;
    uint32_t subsampling_convert_clinelen;
    uint32_t subsampling_convert_clines;
    uint32_t subsampling_convert_ybuflen;
    uint32_t subsampling_convert_cbuflen;
    uint32_t subsampling_convert_ycbcrbuflen;
    uint8_t *subsampling_convert_ycbcrbuf;
    uint8_t *subsampling_convert_ybuf;
    uint8_t *subsampling_convert_cbbuf;
    uint8_t *subsampling_convert_crbuf;
    uint32_t subsampling_convert_ycbcrimagelen;
    uint8_t **subsampling_convert_ycbcrimage;
    uint32_t subsampling_convert_clinelenout;
    uint32_t subsampling_convert_state;
    uint32_t bytes_per_line;   /* if the codec outputs subsampled data, a 'line'
                                  in bytes_per_line */
    uint32_t lines_per_strile; /* and lines_per_strile means subsampling_ver
                                  desubsampled rows     */
    OJPEGStateInBufferSource in_buffer_source;
    uint32_t in_buffer_next_strile;
    uint32_t in_buffer_strile_count;
    uint64_t in_buffer_file_pos;
    uint8_t in_buffer_file_pos_log;
    uint64_t in_buffer_file_togo;
    uint16_t in_buffer_togo;
    uint8_t *in_buffer_cur;
    uint8_t in_buffer[OJPEG_BUFFER];
    OJPEGStateOutState out_state;
    uint8_t out_buffer[OJPEG_BUFFER];
    uint8_t *skip_buffer;
} OJPEGState;

static int OJPEGVGetField(TIFF *tif, uint32_t tag, va_list ap);
static int OJPEGVSetField(TIFF *tif, uint32_t tag, va_list ap);
static void OJPEGPrintDir(TIFF *tif, FILE *fd, long flags);

static int OJPEGFixupTags(TIFF *tif);
static int OJPEGSetupDecode(TIFF *tif);
static int OJPEGPreDecode(TIFF *tif, uint16_t s);
static int OJPEGPreDecodeSkipRaw(TIFF *tif);
static int OJPEGPreDecodeSkipScanlines(TIFF *tif);
static int OJPEGDecode(TIFF *tif, uint8_t *buf, tmsize_t cc, uint16_t s);
static int OJPEGDecodeRaw(TIFF *tif, uint8_t *buf, tmsize_t cc);
static int OJPEGDecodeScanlines(TIFF *tif, uint8_t *buf, tmsize_t cc);
static void OJPEGPostDecode(TIFF *tif, uint8_t *buf, tmsize_t cc);
static int OJPEGSetupEncode(TIFF *tif);
static int OJPEGPreEncode(TIFF *tif, uint16_t s);
static int OJPEGEncode(TIFF *tif, uint8_t *buf, tmsize_t cc, uint16_t s);
static int OJPEGPostEncode(TIFF *tif);
static void OJPEGCleanup(TIFF *tif);

static void OJPEGSubsamplingCorrect(TIFF *tif);
static int OJPEGReadHeaderInfo(TIFF *tif);
static int OJPEGReadSecondarySos(TIFF *tif, uint16_t s);
static int OJPEGWriteHeaderInfo(TIFF *tif);
static void OJPEGLibjpegSessionAbort(TIFF *tif);

static int OJPEGReadHeaderInfoSec(TIFF *tif);
static int OJPEGReadHeaderInfoSecStreamDri(TIFF *tif);
static int OJPEGReadHeaderInfoSecStreamDqt(TIFF *tif);
static int OJPEGReadHeaderInfoSecStreamDht(TIFF *tif);
static int OJPEGReadHeaderInfoSecStreamSof(TIFF *tif, uint8_t marker_id);
static int OJPEGReadHeaderInfoSecStreamSos(TIFF *tif);
static int OJPEGReadHeaderInfoSecTablesQTable(TIFF *tif);
static int OJPEGReadHeaderInfoSecTablesDcTable(TIFF *tif);
static int OJPEGReadHeaderInfoSecTablesAcTable(TIFF *tif);

static int OJPEGReadBufferFill(OJPEGState *sp);
static int OJPEGReadByte(OJPEGState *sp, uint8_t *byte);
static int OJPEGReadBytePeek(OJPEGState *sp, uint8_t *byte);
static void OJPEGReadByteAdvance(OJPEGState *sp);
static int OJPEGReadWord(OJPEGState *sp, uint16_t *word);
static int OJPEGReadBlock(OJPEGState *sp, uint16_t len, void *mem);
static void OJPEGReadSkip(OJPEGState *sp, uint16_t len);

static int OJPEGWriteStream(TIFF *tif, void **mem, uint32_t *len);
static void OJPEGWriteStreamSoi(TIFF *tif, void **mem, uint32_t *len);
static void OJPEGWriteStreamQTable(TIFF *tif, uint8_t table_index, void **mem,
                                   uint32_t *len);
static void OJPEGWriteStreamDcTable(TIFF *tif, uint8_t table_index, void **mem,
                                    uint32_t *len);
static void OJPEGWriteStreamAcTable(TIFF *tif, uint8_t table_index, void **mem,
                                    uint32_t *len);
static void OJPEGWriteStreamDri(TIFF *tif, void **mem, uint32_t *len);
static void OJPEGWriteStreamSof(TIFF *tif, void **mem, uint32_t *len);
static void OJPEGWriteStreamSos(TIFF *tif, void **mem, uint32_t *len);
static int OJPEGWriteStreamCompressed(TIFF *tif, void **mem, uint32_t *len);
static void OJPEGWriteStreamRst(TIFF *tif, void **mem, uint32_t *len);
static void OJPEGWriteStreamEoi(TIFF *tif, void **mem, uint32_t *len);

#ifdef LIBJPEG_ENCAP_EXTERNAL
extern int jpeg_create_decompress_encap(OJPEGState *sp,
                                        jpeg_decompress_struct *cinfo);
extern int jpeg_read_header_encap(OJPEGState *sp, jpeg_decompress_struct *cinfo,
                                  uint8_t require_image);
extern int jpeg_start_decompress_encap(OJPEGState *sp,
                                       jpeg_decompress_struct *cinfo);
extern int jpeg_read_scanlines_encap(OJPEGState *sp,
                                     jpeg_decompress_struct *cinfo,
                                     void *scanlines, uint32_t max_lines);
extern int jpeg_read_raw_data_encap(OJPEGState *sp,
                                    jpeg_decompress_struct *cinfo, void *data,
                                    uint32_t max_lines);
extern void jpeg_encap_unwind(TIFF *tif);
#else
static int jpeg_create_decompress_encap(OJPEGState *sp,
                                        jpeg_decompress_struct *j);
static int jpeg_read_header_encap(OJPEGState *sp, jpeg_decompress_struct *cinfo,
                                  uint8_t require_image);
static int jpeg_start_decompress_encap(OJPEGState *sp,
                                       jpeg_decompress_struct *cinfo);
static int jpeg_read_scanlines_encap(OJPEGState *sp,
                                     jpeg_decompress_struct *cinfo,
                                     void *scanlines, uint32_t max_lines);
static int jpeg_read_raw_data_encap(OJPEGState *sp,
                                    jpeg_decompress_struct *cinfo, void *data,
                                    uint32_t max_lines);
static void jpeg_encap_unwind(TIFF *tif);
#endif

static void OJPEGLibjpegJpegErrorMgrOutputMessage(jpeg_common_struct *cinfo);
static void OJPEGLibjpegJpegErrorMgrErrorExit(jpeg_common_struct *cinfo);
static void OJPEGLibjpegJpegSourceMgrInitSource(jpeg_decompress_struct *cinfo);
static boolean
OJPEGLibjpegJpegSourceMgrFillInputBuffer(jpeg_decompress_struct *cinfo);
static void
OJPEGLibjpegJpegSourceMgrSkipInputData(jpeg_decompress_struct *cinfo,
                                       long num_bytes);
static boolean
OJPEGLibjpegJpegSourceMgrResyncToRestart(jpeg_decompress_struct *cinfo,
                                         int desired);
static void OJPEGLibjpegJpegSourceMgrTermSource(jpeg_decompress_struct *cinfo);

int TIFFInitOJPEG(TIFF *tif, int scheme)
{
    static const char module[] = "TIFFInitOJPEG";
    OJPEGState *sp;

    (void)scheme;
    assert(scheme == COMPRESSION_OJPEG);

    /*
     * Merge codec-specific tag information.
     */
    if (!_TIFFMergeFields(tif, ojpegFields, TIFFArrayCount(ojpegFields)))
    {
        TIFFErrorExtR(tif, module,
                      "Merging Old JPEG codec-specific tags failed");
        return 0;
    }

    /* state block */
    sp = _TIFFmallocExt(tif, sizeof(OJPEGState));
    if (sp == NULL)
    {
        TIFFErrorExtR(tif, module, "No space for OJPEG state block");
        return (0);
    }
    _TIFFmemset(sp, 0, sizeof(OJPEGState));
    sp->tif = tif;
    sp->jpeg_proc = 1;
    sp->subsampling_hor = 2;
    sp->subsampling_ver = 2;
    TIFFSetField(tif, TIFFTAG_YCBCRSUBSAMPLING, 2, 2);
    /* tif codec methods */
    tif->tif_fixuptags = OJPEGFixupTags;
    tif->tif_setupdecode = OJPEGSetupDecode;
    tif->tif_predecode = OJPEGPreDecode;
    tif->tif_postdecode = OJPEGPostDecode;
    tif->tif_decoderow = OJPEGDecode;
    tif->tif_decodestrip = OJPEGDecode;
    tif->tif_decodetile = OJPEGDecode;
    tif->tif_setupencode = OJPEGSetupEncode;
    tif->tif_preencode = OJPEGPreEncode;
    tif->tif_postencode = OJPEGPostEncode;
    tif->tif_encoderow = OJPEGEncode;
    tif->tif_encodestrip = OJPEGEncode;
    tif->tif_encodetile = OJPEGEncode;
    tif->tif_cleanup = OJPEGCleanup;
    tif->tif_data = (uint8_t *)sp;
    /* tif tag methods */
    sp->vgetparent = tif->tif_tagmethods.vgetfield;
    tif->tif_tagmethods.vgetfield = OJPEGVGetField;
    sp->vsetparent = tif->tif_tagmethods.vsetfield;
    tif->tif_tagmethods.vsetfield = OJPEGVSetField;
    sp->printdir = tif->tif_tagmethods.printdir;
    tif->tif_tagmethods.printdir = OJPEGPrintDir;
    /* Some OJPEG files don't have strip or tile offsets or bytecounts tags.
       Some others do, but have totally meaningless or corrupt values
       in these tags. In these cases, the JpegInterchangeFormat stream is
       reliable. In any case, this decoder reads the compressed data itself,
       from the most reliable locations, and we need to notify encapsulating
       LibTiff not to read raw strips or tiles for us. */
    tif->tif_flags |= TIFF_NOREADRAW;
    return (1);
}

static int OJPEGVGetField(TIFF *tif, uint32_t tag, va_list ap)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    switch (tag)
    {
        case TIFFTAG_JPEGIFOFFSET:
            *va_arg(ap, uint64_t *) = (uint64_t)sp->jpeg_interchange_format;
            break;
        case TIFFTAG_JPEGIFBYTECOUNT:
            *va_arg(ap, uint64_t *) =
                (uint64_t)sp->jpeg_interchange_format_length;
            break;
        case TIFFTAG_YCBCRSUBSAMPLING:
            if (sp->subsamplingcorrect_done == 0)
                OJPEGSubsamplingCorrect(tif);
            *va_arg(ap, uint16_t *) = (uint16_t)sp->subsampling_hor;
            *va_arg(ap, uint16_t *) = (uint16_t)sp->subsampling_ver;
            break;
        case TIFFTAG_JPEGQTABLES:
            *va_arg(ap, uint32_t *) = (uint32_t)sp->qtable_offset_count;
            *va_arg(ap, const void **) = (const void *)sp->qtable_offset;
            break;
        case TIFFTAG_JPEGDCTABLES:
            *va_arg(ap, uint32_t *) = (uint32_t)sp->dctable_offset_count;
            *va_arg(ap, const void **) = (const void *)sp->dctable_offset;
            break;
        case TIFFTAG_JPEGACTABLES:
            *va_arg(ap, uint32_t *) = (uint32_t)sp->actable_offset_count;
            *va_arg(ap, const void **) = (const void *)sp->actable_offset;
            break;
        case TIFFTAG_JPEGPROC:
            *va_arg(ap, uint16_t *) = (uint16_t)sp->jpeg_proc;
            break;
        case TIFFTAG_JPEGRESTARTINTERVAL:
            *va_arg(ap, uint16_t *) = sp->restart_interval;
            break;
        default:
            return (*sp->vgetparent)(tif, tag, ap);
    }
    return (1);
}

static int OJPEGVSetField(TIFF *tif, uint32_t tag, va_list ap)
{
    static const char module[] = "OJPEGVSetField";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint32_t ma;
    uint64_t *mb;
    uint32_t n;
    const TIFFField *fip;

    switch (tag)
    {
        case TIFFTAG_JPEGIFOFFSET:
            sp->jpeg_interchange_format = (uint64_t)va_arg(ap, uint64_t);
            break;
        case TIFFTAG_JPEGIFBYTECOUNT:
            sp->jpeg_interchange_format_length = (uint64_t)va_arg(ap, uint64_t);
            break;
        case TIFFTAG_YCBCRSUBSAMPLING:
            sp->subsampling_tag = 1;
            sp->subsampling_hor = (uint8_t)va_arg(ap, uint16_vap);
            sp->subsampling_ver = (uint8_t)va_arg(ap, uint16_vap);
            tif->tif_dir.td_ycbcrsubsampling[0] = sp->subsampling_hor;
            tif->tif_dir.td_ycbcrsubsampling[1] = sp->subsampling_ver;
            break;
        case TIFFTAG_JPEGQTABLES:
            ma = (uint32_t)va_arg(ap, uint32_t);
            if (ma != 0)
            {
                if (ma > 3)
                {
                    TIFFErrorExtR(tif, module,
                                  "JpegQTables tag has incorrect count");
                    return (0);
                }
                sp->qtable_offset_count = (uint8_t)ma;
                mb = (uint64_t *)va_arg(ap, uint64_t *);
                for (n = 0; n < ma; n++)
                    sp->qtable_offset[n] = mb[n];
            }
            break;
        case TIFFTAG_JPEGDCTABLES:
            ma = (uint32_t)va_arg(ap, uint32_t);
            if (ma != 0)
            {
                if (ma > 3)
                {
                    TIFFErrorExtR(tif, module,
                                  "JpegDcTables tag has incorrect count");
                    return (0);
                }
                sp->dctable_offset_count = (uint8_t)ma;
                mb = (uint64_t *)va_arg(ap, uint64_t *);
                for (n = 0; n < ma; n++)
                    sp->dctable_offset[n] = mb[n];
            }
            break;
        case TIFFTAG_JPEGACTABLES:
            ma = (uint32_t)va_arg(ap, uint32_t);
            if (ma != 0)
            {
                if (ma > 3)
                {
                    TIFFErrorExtR(tif, module,
                                  "JpegAcTables tag has incorrect count");
                    return (0);
                }
                sp->actable_offset_count = (uint8_t)ma;
                mb = (uint64_t *)va_arg(ap, uint64_t *);
                for (n = 0; n < ma; n++)
                    sp->actable_offset[n] = mb[n];
            }
            break;
        case TIFFTAG_JPEGPROC:
            sp->jpeg_proc = (uint8_t)va_arg(ap, uint16_vap);
            break;
        case TIFFTAG_JPEGRESTARTINTERVAL:
            sp->restart_interval = (uint16_t)va_arg(ap, uint16_vap);
            break;
        default:
            return (*sp->vsetparent)(tif, tag, ap);
    }
    fip = TIFFFieldWithTag(tif, tag);
    if (fip == NULL) /* shouldn't happen */
        return (0);
    TIFFSetFieldBit(tif, fip->field_bit);
    tif->tif_flags |= TIFF_DIRTYDIRECT;
    return (1);
}

static void OJPEGPrintDir(TIFF *tif, FILE *fd, long flags)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    (void)flags;
    assert(sp != NULL);
    if (TIFFFieldSet(tif, FIELD_OJPEG_JPEGINTERCHANGEFORMAT))
        fprintf(fd, "  JpegInterchangeFormat: %" PRIu64 "\n",
                (uint64_t)sp->jpeg_interchange_format);
    if (TIFFFieldSet(tif, FIELD_OJPEG_JPEGINTERCHANGEFORMATLENGTH))
        fprintf(fd, "  JpegInterchangeFormatLength: %" PRIu64 "\n",
                (uint64_t)sp->jpeg_interchange_format_length);
    if (TIFFFieldSet(tif, FIELD_OJPEG_JPEGQTABLES))
    {
        fprintf(fd, "  JpegQTables:");
        for (m = 0; m < sp->qtable_offset_count; m++)
            fprintf(fd, " %" PRIu64, (uint64_t)sp->qtable_offset[m]);
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif, FIELD_OJPEG_JPEGDCTABLES))
    {
        fprintf(fd, "  JpegDcTables:");
        for (m = 0; m < sp->dctable_offset_count; m++)
            fprintf(fd, " %" PRIu64, (uint64_t)sp->dctable_offset[m]);
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif, FIELD_OJPEG_JPEGACTABLES))
    {
        fprintf(fd, "  JpegAcTables:");
        for (m = 0; m < sp->actable_offset_count; m++)
            fprintf(fd, " %" PRIu64, (uint64_t)sp->actable_offset[m]);
        fprintf(fd, "\n");
    }
    if (TIFFFieldSet(tif, FIELD_OJPEG_JPEGPROC))
        fprintf(fd, "  JpegProc: %" PRIu8 "\n", sp->jpeg_proc);
    if (TIFFFieldSet(tif, FIELD_OJPEG_JPEGRESTARTINTERVAL))
        fprintf(fd, "  JpegRestartInterval: %" PRIu16 "\n",
                sp->restart_interval);
    if (sp->printdir)
        (*sp->printdir)(tif, fd, flags);
}

static int OJPEGFixupTags(TIFF *tif)
{
    (void)tif;
    return (1);
}

static int OJPEGSetupDecode(TIFF *tif)
{
    static const char module[] = "OJPEGSetupDecode";
    TIFFWarningExtR(tif, module,
                    "Deprecated and troublesome old-style JPEG compression "
                    "mode, please convert to new-style JPEG compression and "
                    "notify vendor of writing software");
    return (1);
}

static int OJPEGPreDecode(TIFF *tif, uint16_t s)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint32_t m;
    if (sp->subsamplingcorrect_done == 0)
        OJPEGSubsamplingCorrect(tif);
    if (sp->readheader_done == 0)
    {
        if (OJPEGReadHeaderInfo(tif) == 0)
            return (0);
    }
    if (sp->sos_end[s].log == 0)
    {
        if (OJPEGReadSecondarySos(tif, s) == 0)
            return (0);
    }
    if (isTiled(tif))
        m = tif->tif_curtile;
    else
        m = tif->tif_curstrip;
    if ((sp->writeheader_done != 0) &&
        ((sp->write_cursample != s) || (sp->write_curstrile > m)))
    {
        if (sp->libjpeg_session_active != 0)
            OJPEGLibjpegSessionAbort(tif);
        sp->writeheader_done = 0;
    }
    if (sp->writeheader_done == 0)
    {
        sp->plane_sample_offset = (uint8_t)s;
        sp->write_cursample = s;
        sp->write_curstrile = s * tif->tif_dir.td_stripsperimage;
        if ((sp->in_buffer_file_pos_log == 0) ||
            (sp->in_buffer_file_pos - sp->in_buffer_togo !=
             sp->sos_end[s].in_buffer_file_pos))
        {
            sp->in_buffer_source = sp->sos_end[s].in_buffer_source;
            sp->in_buffer_next_strile = sp->sos_end[s].in_buffer_next_strile;
            sp->in_buffer_file_pos = sp->sos_end[s].in_buffer_file_pos;
            sp->in_buffer_file_pos_log = 0;
            sp->in_buffer_file_togo = sp->sos_end[s].in_buffer_file_togo;
            sp->in_buffer_togo = 0;
            sp->in_buffer_cur = 0;
        }
        if (OJPEGWriteHeaderInfo(tif) == 0)
            return (0);
    }
    while (sp->write_curstrile < m)
    {
        if (sp->libjpeg_jpeg_query_style == 0)
        {
            if (OJPEGPreDecodeSkipRaw(tif) == 0)
                return (0);
        }
        else
        {
            if (OJPEGPreDecodeSkipScanlines(tif) == 0)
                return (0);
        }
        sp->write_curstrile++;
    }
    sp->decoder_ok = 1;
    return (1);
}

static int OJPEGPreDecodeSkipRaw(TIFF *tif)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint32_t m;
    m = sp->lines_per_strile;
    if (sp->subsampling_convert_state != 0)
    {
        if (sp->subsampling_convert_clines - sp->subsampling_convert_state >= m)
        {
            sp->subsampling_convert_state += m;
            if (sp->subsampling_convert_state == sp->subsampling_convert_clines)
                sp->subsampling_convert_state = 0;
            return (1);
        }
        m -= sp->subsampling_convert_clines - sp->subsampling_convert_state;
        sp->subsampling_convert_state = 0;
        sp->error_in_raw_data_decoding = 0;
    }
    while (m >= sp->subsampling_convert_clines)
    {
        if (jpeg_read_raw_data_encap(sp, &(sp->libjpeg_jpeg_decompress_struct),
                                     sp->subsampling_convert_ycbcrimage,
                                     sp->subsampling_ver * 8) == 0)
            return (0);
        m -= sp->subsampling_convert_clines;
    }
    if (m > 0)
    {
        if (jpeg_read_raw_data_encap(sp, &(sp->libjpeg_jpeg_decompress_struct),
                                     sp->subsampling_convert_ycbcrimage,
                                     sp->subsampling_ver * 8) == 0)
            return (0);
        sp->subsampling_convert_state = m;
    }
    return (1);
}

static int OJPEGPreDecodeSkipScanlines(TIFF *tif)
{
    static const char module[] = "OJPEGPreDecodeSkipScanlines";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint32_t m;
    if (sp->skip_buffer == NULL)
    {
        sp->skip_buffer = _TIFFmallocExt(tif, sp->bytes_per_line);
        if (sp->skip_buffer == NULL)
        {
            TIFFErrorExtR(tif, module, "Out of memory");
            return (0);
        }
    }
    for (m = 0; m < sp->lines_per_strile; m++)
    {
        if (jpeg_read_scanlines_encap(sp, &(sp->libjpeg_jpeg_decompress_struct),
                                      &sp->skip_buffer, 1) == 0)
            return (0);
    }
    return (1);
}

static int OJPEGDecode(TIFF *tif, uint8_t *buf, tmsize_t cc, uint16_t s)
{
    static const char module[] = "OJPEGDecode";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    (void)s;
    if (!sp->decoder_ok)
    {
        TIFFErrorExtR(tif, module,
                      "Cannot decode: decoder not correctly initialized");
        return 0;
    }
    if (sp->libjpeg_session_active == 0)
    {
        /* This should normally not happen, except that it does when */
        /* using TIFFReadScanline() which calls OJPEGPostDecode() for */
        /* each scanline, which assumes that a whole strile was read */
        /* and may thus incorrectly consider it has read the whole image,
         * causing */
        /* OJPEGLibjpegSessionAbort() to be called prematurely. */
        /* Triggered by https://gitlab.com/libtiff/libtiff/-/issues/337 */
        TIFFErrorExtR(tif, module,
                      "Cannot decode: libjpeg_session_active == 0");
        return 0;
    }
    if (sp->error_in_raw_data_decoding)
    {
        return 0;
    }
    if (sp->libjpeg_jpeg_query_style == 0)
    {
        if (OJPEGDecodeRaw(tif, buf, cc) == 0)
            return (0);
    }
    else
    {
        if (OJPEGDecodeScanlines(tif, buf, cc) == 0)
            return (0);
    }
    return (1);
}

static int OJPEGDecodeRaw(TIFF *tif, uint8_t *buf, tmsize_t cc)
{
    static const char module[] = "OJPEGDecodeRaw";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t *m;
    tmsize_t n;
    uint8_t *oy;
    uint8_t *ocb;
    uint8_t *ocr;
    uint8_t *p;
    uint32_t q;
    uint8_t *r;
    uint8_t sx, sy;
    if (cc % sp->bytes_per_line != 0)
    {
        TIFFErrorExtR(tif, module, "Fractional scanline not read");
        return (0);
    }
    assert(cc > 0);
    m = buf;
    n = cc;
    do
    {
        if (sp->subsampling_convert_state == 0)
        {
            if (jpeg_read_raw_data_encap(sp,
                                         &(sp->libjpeg_jpeg_decompress_struct),
                                         sp->subsampling_convert_ycbcrimage,
                                         sp->subsampling_ver * 8) == 0)
            {
                sp->error_in_raw_data_decoding = 1;
                return (0);
            }
        }
        oy = sp->subsampling_convert_ybuf +
             sp->subsampling_convert_state * sp->subsampling_ver *
                 sp->subsampling_convert_ylinelen;
        ocb = sp->subsampling_convert_cbbuf +
              sp->subsampling_convert_state * sp->subsampling_convert_clinelen;
        ocr = sp->subsampling_convert_crbuf +
              sp->subsampling_convert_state * sp->subsampling_convert_clinelen;
        p = m;
        for (q = 0; q < sp->subsampling_convert_clinelenout; q++)
        {
            r = oy;
            for (sy = 0; sy < sp->subsampling_ver; sy++)
            {
                for (sx = 0; sx < sp->subsampling_hor; sx++)
                    *p++ = *r++;
                r += sp->subsampling_convert_ylinelen - sp->subsampling_hor;
            }
            oy += sp->subsampling_hor;
            *p++ = *ocb++;
            *p++ = *ocr++;
        }
        sp->subsampling_convert_state++;
        if (sp->subsampling_convert_state == sp->subsampling_convert_clines)
            sp->subsampling_convert_state = 0;
        m += sp->bytes_per_line;
        n -= sp->bytes_per_line;
    } while (n > 0);
    return (1);
}

static int OJPEGDecodeScanlines(TIFF *tif, uint8_t *buf, tmsize_t cc)
{
    static const char module[] = "OJPEGDecodeScanlines";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t *m;
    tmsize_t n;
    if (cc % sp->bytes_per_line != 0)
    {
        TIFFErrorExtR(tif, module, "Fractional scanline not read");
        return (0);
    }
    assert(cc > 0);
    m = buf;
    n = cc;
    do
    {
        if (jpeg_read_scanlines_encap(sp, &(sp->libjpeg_jpeg_decompress_struct),
                                      &m, 1) == 0)
            return (0);
        m += sp->bytes_per_line;
        n -= sp->bytes_per_line;
    } while (n > 0);
    return (1);
}

static void OJPEGPostDecode(TIFF *tif, uint8_t *buf, tmsize_t cc)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    (void)buf;
    (void)cc;
    /* This function somehow incorrectly assumes that a whole strile was read,
     */
    /* which is not true when TIFFReadScanline() is called, */
    /* and may thus incorrectly consider it has read the whole image, causing */
    /* OJPEGLibjpegSessionAbort() to be called prematurely. */
    /* So this logic should be fixed to take into account cc, or disable */
    /* the scan line reading interface. */
    /* Triggered by https://gitlab.com/libtiff/libtiff/-/issues/337 */
    sp->write_curstrile++;
    if (sp->write_curstrile % tif->tif_dir.td_stripsperimage == 0)
    {
        assert(sp->libjpeg_session_active != 0);
        OJPEGLibjpegSessionAbort(tif);
        sp->writeheader_done = 0;
    }
}

static int OJPEGSetupEncode(TIFF *tif)
{
    static const char module[] = "OJPEGSetupEncode";
    TIFFErrorExtR(
        tif, module,
        "OJPEG encoding not supported; use new-style JPEG compression instead");
    return (0);
}

static int OJPEGPreEncode(TIFF *tif, uint16_t s)
{
    static const char module[] = "OJPEGPreEncode";
    (void)s;
    TIFFErrorExtR(
        tif, module,
        "OJPEG encoding not supported; use new-style JPEG compression instead");
    return (0);
}

static int OJPEGEncode(TIFF *tif, uint8_t *buf, tmsize_t cc, uint16_t s)
{
    static const char module[] = "OJPEGEncode";
    (void)buf;
    (void)cc;
    (void)s;
    TIFFErrorExtR(
        tif, module,
        "OJPEG encoding not supported; use new-style JPEG compression instead");
    return (0);
}

static int OJPEGPostEncode(TIFF *tif)
{
    static const char module[] = "OJPEGPostEncode";
    TIFFErrorExtR(
        tif, module,
        "OJPEG encoding not supported; use new-style JPEG compression instead");
    return (0);
}

static void OJPEGCleanup(TIFF *tif)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    if (sp != 0)
    {
        tif->tif_tagmethods.vgetfield = sp->vgetparent;
        tif->tif_tagmethods.vsetfield = sp->vsetparent;
        tif->tif_tagmethods.printdir = sp->printdir;
        if (sp->qtable[0] != 0)
            _TIFFfreeExt(tif, sp->qtable[0]);
        if (sp->qtable[1] != 0)
            _TIFFfreeExt(tif, sp->qtable[1]);
        if (sp->qtable[2] != 0)
            _TIFFfreeExt(tif, sp->qtable[2]);
        if (sp->qtable[3] != 0)
            _TIFFfreeExt(tif, sp->qtable[3]);
        if (sp->dctable[0] != 0)
            _TIFFfreeExt(tif, sp->dctable[0]);
        if (sp->dctable[1] != 0)
            _TIFFfreeExt(tif, sp->dctable[1]);
        if (sp->dctable[2] != 0)
            _TIFFfreeExt(tif, sp->dctable[2]);
        if (sp->dctable[3] != 0)
            _TIFFfreeExt(tif, sp->dctable[3]);
        if (sp->actable[0] != 0)
            _TIFFfreeExt(tif, sp->actable[0]);
        if (sp->actable[1] != 0)
            _TIFFfreeExt(tif, sp->actable[1]);
        if (sp->actable[2] != 0)
            _TIFFfreeExt(tif, sp->actable[2]);
        if (sp->actable[3] != 0)
            _TIFFfreeExt(tif, sp->actable[3]);
        if (sp->libjpeg_session_active != 0)
            OJPEGLibjpegSessionAbort(tif);
        if (sp->subsampling_convert_ycbcrbuf != 0)
            _TIFFfreeExt(tif, sp->subsampling_convert_ycbcrbuf);
        if (sp->subsampling_convert_ycbcrimage != 0)
            _TIFFfreeExt(tif, sp->subsampling_convert_ycbcrimage);
        if (sp->skip_buffer != 0)
            _TIFFfreeExt(tif, sp->skip_buffer);
        _TIFFfreeExt(tif, sp);
        tif->tif_data = NULL;
        _TIFFSetDefaultCompressionState(tif);
    }
}

static void OJPEGSubsamplingCorrect(TIFF *tif)
{
    static const char module[] = "OJPEGSubsamplingCorrect";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t mh;
    uint8_t mv;

    assert(sp->subsamplingcorrect_done == 0);
    if ((tif->tif_dir.td_samplesperpixel != 3) ||
        ((tif->tif_dir.td_photometric != PHOTOMETRIC_YCBCR) &&
         (tif->tif_dir.td_photometric != PHOTOMETRIC_ITULAB)))
    {
        if (sp->subsampling_tag != 0)
            TIFFWarningExtR(tif, module,
                            "Subsampling tag not appropriate for this "
                            "Photometric and/or SamplesPerPixel");
        sp->subsampling_hor = 1;
        sp->subsampling_ver = 1;
        sp->subsampling_force_desubsampling_inside_decompression = 0;
    }
    else
    {
        sp->subsamplingcorrect_done = 1;
        mh = sp->subsampling_hor;
        mv = sp->subsampling_ver;
        sp->subsamplingcorrect = 1;
        OJPEGReadHeaderInfoSec(tif);
        if (sp->subsampling_force_desubsampling_inside_decompression != 0)
        {
            sp->subsampling_hor = 1;
            sp->subsampling_ver = 1;
        }
        sp->subsamplingcorrect = 0;
        if (((sp->subsampling_hor != mh) || (sp->subsampling_ver != mv)) &&
            (sp->subsampling_force_desubsampling_inside_decompression == 0))
        {
            if (sp->subsampling_tag == 0)
                TIFFWarningExtR(
                    tif, module,
                    "Subsampling tag is not set, yet subsampling inside JPEG "
                    "data [%" PRIu8 ",%" PRIu8
                    "] does not match default values [2,2]; assuming "
                    "subsampling inside JPEG data is correct",
                    sp->subsampling_hor, sp->subsampling_ver);
            else
                TIFFWarningExtR(
                    tif, module,
                    "Subsampling inside JPEG data [%" PRIu8 ",%" PRIu8
                    "] does not match subsampling tag values [%" PRIu8
                    ",%" PRIu8
                    "]; assuming subsampling inside JPEG data is correct",
                    sp->subsampling_hor, sp->subsampling_ver, mh, mv);
        }
        if (sp->subsampling_force_desubsampling_inside_decompression != 0)
        {
            if (sp->subsampling_tag == 0)
                TIFFWarningExtR(
                    tif, module,
                    "Subsampling tag is not set, yet subsampling inside JPEG "
                    "data does not match default values [2,2] (nor any other "
                    "values allowed in TIFF); assuming subsampling inside JPEG "
                    "data is correct and desubsampling inside JPEG "
                    "decompression");
            else
                TIFFWarningExtR(
                    tif, module,
                    "Subsampling inside JPEG data does not match subsampling "
                    "tag values [%" PRIu8 ",%" PRIu8
                    "] (nor any other values allowed in TIFF); assuming "
                    "subsampling inside JPEG data is correct and desubsampling "
                    "inside JPEG decompression",
                    mh, mv);
        }
        if (sp->subsampling_force_desubsampling_inside_decompression == 0)
        {
            if (sp->subsampling_hor < sp->subsampling_ver)
                TIFFWarningExtR(tif, module,
                                "Subsampling values [%" PRIu8 ",%" PRIu8
                                "] are not allowed in TIFF",
                                sp->subsampling_hor, sp->subsampling_ver);
        }
    }
    sp->subsamplingcorrect_done = 1;
}

static int OJPEGReadHeaderInfo(TIFF *tif)
{
    static const char module[] = "OJPEGReadHeaderInfo";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    assert(sp->readheader_done == 0);
    sp->image_width = tif->tif_dir.td_imagewidth;
    sp->image_length = tif->tif_dir.td_imagelength;
    if (isTiled(tif))
    {
        sp->strile_width = tif->tif_dir.td_tilewidth;
        sp->strile_length = tif->tif_dir.td_tilelength;
        sp->strile_length_total =
            ((sp->image_length + sp->strile_length - 1) / sp->strile_length) *
            sp->strile_length;
    }
    else
    {
        sp->strile_width = sp->image_width;
        sp->strile_length = tif->tif_dir.td_rowsperstrip;
        if (sp->strile_length == (uint32_t)-1)
            sp->strile_length = sp->image_length;
        sp->strile_length_total = sp->image_length;
    }
    if (tif->tif_dir.td_samplesperpixel == 1)
    {
        sp->samples_per_pixel = 1;
        sp->plane_sample_offset = 0;
        sp->samples_per_pixel_per_plane = sp->samples_per_pixel;
        sp->subsampling_hor = 1;
        sp->subsampling_ver = 1;
    }
    else
    {
        if (tif->tif_dir.td_samplesperpixel != 3)
        {
            TIFFErrorExtR(tif, module,
                          "SamplesPerPixel %" PRIu8
                          " not supported for this compression scheme",
                          sp->samples_per_pixel);
            return (0);
        }
        sp->samples_per_pixel = 3;
        sp->plane_sample_offset = 0;
        if (tif->tif_dir.td_planarconfig == PLANARCONFIG_CONTIG)
            sp->samples_per_pixel_per_plane = 3;
        else
            sp->samples_per_pixel_per_plane = 1;
    }
    if (sp->strile_length < sp->image_length)
    {
        if (((sp->subsampling_hor != 1) && (sp->subsampling_hor != 2) &&
             (sp->subsampling_hor != 4)) ||
            ((sp->subsampling_ver != 1) && (sp->subsampling_ver != 2) &&
             (sp->subsampling_ver != 4)))
        {
            TIFFErrorExtR(tif, module, "Invalid subsampling values");
            return (0);
        }
        if (sp->strile_length % (sp->subsampling_ver * 8) != 0)
        {
            TIFFErrorExtR(tif, module,
                          "Incompatible vertical subsampling and image "
                          "strip/tile length");
            return (0);
        }
        sp->restart_interval =
            (uint16_t)(((sp->strile_width + sp->subsampling_hor * 8 - 1) /
                        (sp->subsampling_hor * 8)) *
                       (sp->strile_length / (sp->subsampling_ver * 8)));
    }
    if (OJPEGReadHeaderInfoSec(tif) == 0)
        return (0);
    sp->sos_end[0].log = 1;
    sp->sos_end[0].in_buffer_source = sp->in_buffer_source;
    sp->sos_end[0].in_buffer_next_strile = sp->in_buffer_next_strile;
    sp->sos_end[0].in_buffer_file_pos =
        sp->in_buffer_file_pos - sp->in_buffer_togo;
    sp->sos_end[0].in_buffer_file_togo =
        sp->in_buffer_file_togo + sp->in_buffer_togo;
    sp->readheader_done = 1;
    return (1);
}

static int OJPEGReadSecondarySos(TIFF *tif, uint16_t s)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    assert(s > 0);
    assert(s < 3);
    assert(sp->sos_end[0].log != 0);
    assert(sp->sos_end[s].log == 0);
    sp->plane_sample_offset = (uint8_t)(s - 1);
    while (sp->sos_end[sp->plane_sample_offset].log == 0)
        sp->plane_sample_offset--;
    sp->in_buffer_source =
        sp->sos_end[sp->plane_sample_offset].in_buffer_source;
    sp->in_buffer_next_strile =
        sp->sos_end[sp->plane_sample_offset].in_buffer_next_strile;
    sp->in_buffer_file_pos =
        sp->sos_end[sp->plane_sample_offset].in_buffer_file_pos;
    sp->in_buffer_file_pos_log = 0;
    sp->in_buffer_file_togo =
        sp->sos_end[sp->plane_sample_offset].in_buffer_file_togo;
    sp->in_buffer_togo = 0;
    sp->in_buffer_cur = 0;
    while (sp->plane_sample_offset < s)
    {
        do
        {
            if (OJPEGReadByte(sp, &m) == 0)
                return (0);
            if (m == 255)
            {
                do
                {
                    if (OJPEGReadByte(sp, &m) == 0)
                        return (0);
                    if (m != 255)
                        break;
                } while (1);
                if (m == JPEG_MARKER_SOS)
                    break;
            }
        } while (1);
        sp->plane_sample_offset++;
        if (OJPEGReadHeaderInfoSecStreamSos(tif) == 0)
            return (0);
        sp->sos_end[sp->plane_sample_offset].log = 1;
        sp->sos_end[sp->plane_sample_offset].in_buffer_source =
            sp->in_buffer_source;
        sp->sos_end[sp->plane_sample_offset].in_buffer_next_strile =
            sp->in_buffer_next_strile;
        sp->sos_end[sp->plane_sample_offset].in_buffer_file_pos =
            sp->in_buffer_file_pos - sp->in_buffer_togo;
        sp->sos_end[sp->plane_sample_offset].in_buffer_file_togo =
            sp->in_buffer_file_togo + sp->in_buffer_togo;
    }
    return (1);
}

static int OJPEGWriteHeaderInfo(TIFF *tif)
{
    static const char module[] = "OJPEGWriteHeaderInfo";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t **m;
    uint32_t n;
    /* if a previous attempt failed, don't try again */
    if (sp->libjpeg_session_active != 0)
        return 0;
    sp->out_state = ososSoi;
    sp->restart_index = 0;
    jpeg_std_error(&(sp->libjpeg_jpeg_error_mgr));
    sp->libjpeg_jpeg_error_mgr.output_message =
        OJPEGLibjpegJpegErrorMgrOutputMessage;
    sp->libjpeg_jpeg_error_mgr.error_exit = OJPEGLibjpegJpegErrorMgrErrorExit;
    sp->libjpeg_jpeg_decompress_struct.err = &(sp->libjpeg_jpeg_error_mgr);
    sp->libjpeg_jpeg_decompress_struct.client_data = (void *)tif;
    if (jpeg_create_decompress_encap(
            sp, &(sp->libjpeg_jpeg_decompress_struct)) == 0)
        return (0);
    sp->libjpeg_session_active = 1;
    sp->libjpeg_jpeg_source_mgr.bytes_in_buffer = 0;
    sp->libjpeg_jpeg_source_mgr.init_source =
        OJPEGLibjpegJpegSourceMgrInitSource;
    sp->libjpeg_jpeg_source_mgr.fill_input_buffer =
        OJPEGLibjpegJpegSourceMgrFillInputBuffer;
    sp->libjpeg_jpeg_source_mgr.skip_input_data =
        OJPEGLibjpegJpegSourceMgrSkipInputData;
    sp->libjpeg_jpeg_source_mgr.resync_to_restart =
        OJPEGLibjpegJpegSourceMgrResyncToRestart;
    sp->libjpeg_jpeg_source_mgr.term_source =
        OJPEGLibjpegJpegSourceMgrTermSource;
    sp->libjpeg_jpeg_decompress_struct.src = &(sp->libjpeg_jpeg_source_mgr);
    if (jpeg_read_header_encap(sp, &(sp->libjpeg_jpeg_decompress_struct), 1) ==
        0)
        return (0);
    if ((sp->subsampling_force_desubsampling_inside_decompression == 0) &&
        (sp->samples_per_pixel_per_plane > 1))
    {
        sp->libjpeg_jpeg_decompress_struct.raw_data_out = 1;
#if JPEG_LIB_VERSION >= 70
        sp->libjpeg_jpeg_decompress_struct.do_fancy_upsampling = FALSE;
#endif
        sp->libjpeg_jpeg_query_style = 0;
        if (sp->subsampling_convert_log == 0)
        {
            assert(sp->subsampling_convert_ycbcrbuf == 0);
            assert(sp->subsampling_convert_ycbcrimage == 0);
            /* Check for division by zero. */
            if (sp->subsampling_hor == 0 || sp->subsampling_ver == 0)
                return (0);
            sp->subsampling_convert_ylinelen =
                ((sp->strile_width + sp->subsampling_hor * 8 - 1) /
                 (sp->subsampling_hor * 8) * sp->subsampling_hor * 8);
            sp->subsampling_convert_ylines = sp->subsampling_ver * 8;
            sp->subsampling_convert_clinelen =
                sp->subsampling_convert_ylinelen / sp->subsampling_hor;
            sp->subsampling_convert_clines = 8;
            sp->subsampling_convert_ybuflen = sp->subsampling_convert_ylinelen *
                                              sp->subsampling_convert_ylines;
            sp->subsampling_convert_cbuflen = sp->subsampling_convert_clinelen *
                                              sp->subsampling_convert_clines;
            sp->subsampling_convert_ycbcrbuflen =
                sp->subsampling_convert_ybuflen +
                2 * sp->subsampling_convert_cbuflen;
            /* The calloc is not normally necessary, except in some edge/broken
             * cases */
            /* for example for a tiled image of height 1 with a tile height of 1
             * and subsampling_hor=subsampling_ver=2 */
            /* In that case, libjpeg will only fill the 8 first lines of the 16
             * lines */
            /* See https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=16844
             */
            /* Even if this case is allowed (?), its handling is broken because
             * OJPEGPreDecode() should also likely */
            /* reset subsampling_convert_state to 0 when changing tile. */
            sp->subsampling_convert_ycbcrbuf =
                _TIFFcallocExt(tif, 1, sp->subsampling_convert_ycbcrbuflen);
            if (sp->subsampling_convert_ycbcrbuf == 0)
            {
                TIFFErrorExtR(tif, module, "Out of memory");
                return (0);
            }
            sp->subsampling_convert_ybuf = sp->subsampling_convert_ycbcrbuf;
            sp->subsampling_convert_cbbuf =
                sp->subsampling_convert_ybuf + sp->subsampling_convert_ybuflen;
            sp->subsampling_convert_crbuf =
                sp->subsampling_convert_cbbuf + sp->subsampling_convert_cbuflen;
            sp->subsampling_convert_ycbcrimagelen =
                3 + sp->subsampling_convert_ylines +
                2 * sp->subsampling_convert_clines;
            sp->subsampling_convert_ycbcrimage = _TIFFmallocExt(
                tif, sp->subsampling_convert_ycbcrimagelen * sizeof(uint8_t *));
            if (sp->subsampling_convert_ycbcrimage == 0)
            {
                TIFFErrorExtR(tif, module, "Out of memory");
                return (0);
            }
            m = sp->subsampling_convert_ycbcrimage;
            *m++ = (uint8_t *)(sp->subsampling_convert_ycbcrimage + 3);
            *m++ = (uint8_t *)(sp->subsampling_convert_ycbcrimage + 3 +
                               sp->subsampling_convert_ylines);
            *m++ = (uint8_t *)(sp->subsampling_convert_ycbcrimage + 3 +
                               sp->subsampling_convert_ylines +
                               sp->subsampling_convert_clines);
            for (n = 0; n < sp->subsampling_convert_ylines; n++)
                *m++ = sp->subsampling_convert_ybuf +
                       n * sp->subsampling_convert_ylinelen;
            for (n = 0; n < sp->subsampling_convert_clines; n++)
                *m++ = sp->subsampling_convert_cbbuf +
                       n * sp->subsampling_convert_clinelen;
            for (n = 0; n < sp->subsampling_convert_clines; n++)
                *m++ = sp->subsampling_convert_crbuf +
                       n * sp->subsampling_convert_clinelen;
            sp->subsampling_convert_clinelenout =
                sp->strile_width / sp->subsampling_hor +
                ((sp->strile_width % sp->subsampling_hor) != 0 ? 1 : 0);
            sp->subsampling_convert_state = 0;
            sp->error_in_raw_data_decoding = 0;
            sp->bytes_per_line =
                sp->subsampling_convert_clinelenout *
                (sp->subsampling_ver * sp->subsampling_hor + 2);
            sp->lines_per_strile =
                sp->strile_length / sp->subsampling_ver +
                ((sp->strile_length % sp->subsampling_ver) != 0 ? 1 : 0);
            sp->subsampling_convert_log = 1;
        }
    }
    else
    {
        sp->libjpeg_jpeg_decompress_struct.jpeg_color_space = JCS_UNKNOWN;
        sp->libjpeg_jpeg_decompress_struct.out_color_space = JCS_UNKNOWN;
        sp->libjpeg_jpeg_query_style = 1;
        sp->bytes_per_line = sp->samples_per_pixel_per_plane * sp->strile_width;
        sp->lines_per_strile = sp->strile_length;
    }
    if (jpeg_start_decompress_encap(sp,
                                    &(sp->libjpeg_jpeg_decompress_struct)) == 0)
        return (0);
    if (sp->libjpeg_jpeg_decompress_struct.image_width != sp->strile_width)
    {
        TIFFErrorExtR(tif, module,
                      "jpeg_start_decompress() returned image_width = %u, "
                      "expected %" PRIu32,
                      sp->libjpeg_jpeg_decompress_struct.image_width,
                      sp->strile_width);
        return 0;
    }
    if (sp->libjpeg_jpeg_decompress_struct.max_h_samp_factor !=
            sp->subsampling_hor ||
        sp->libjpeg_jpeg_decompress_struct.max_v_samp_factor !=
            sp->subsampling_ver)
    {
        TIFFErrorExtR(tif, module,
                      "jpeg_start_decompress() returned max_h_samp_factor = %d "
                      "and max_v_samp_factor = %d, expected %" PRIu8
                      " and %" PRIu8,
                      sp->libjpeg_jpeg_decompress_struct.max_h_samp_factor,
                      sp->libjpeg_jpeg_decompress_struct.max_v_samp_factor,
                      sp->subsampling_hor, sp->subsampling_ver);
        return 0;
    }

    sp->writeheader_done = 1;
    return (1);
}

static void OJPEGLibjpegSessionAbort(TIFF *tif)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    assert(sp->libjpeg_session_active != 0);
    jpeg_destroy((jpeg_common_struct *)(&(sp->libjpeg_jpeg_decompress_struct)));
    sp->libjpeg_session_active = 0;
}

static int OJPEGReadHeaderInfoSec(TIFF *tif)
{
    static const char module[] = "OJPEGReadHeaderInfoSec";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    uint16_t n;
    uint8_t o;
    if (sp->file_size == 0)
        sp->file_size = TIFFGetFileSize(tif);
    if (sp->jpeg_interchange_format != 0)
    {
        if (sp->jpeg_interchange_format >= sp->file_size)
        {
            sp->jpeg_interchange_format = 0;
            sp->jpeg_interchange_format_length = 0;
        }
        else
        {
            if ((sp->jpeg_interchange_format_length == 0) ||
                (sp->jpeg_interchange_format >
                 UINT64_MAX - sp->jpeg_interchange_format_length) ||
                (sp->jpeg_interchange_format +
                     sp->jpeg_interchange_format_length >
                 sp->file_size))
                sp->jpeg_interchange_format_length =
                    sp->file_size - sp->jpeg_interchange_format;
        }
    }
    sp->in_buffer_source = osibsNotSetYet;
    sp->in_buffer_next_strile = 0;
    sp->in_buffer_strile_count = tif->tif_dir.td_nstrips;
    sp->in_buffer_file_togo = 0;
    sp->in_buffer_togo = 0;
    do
    {
        if (OJPEGReadBytePeek(sp, &m) == 0)
            return (0);
        if (m != 255)
            break;
        OJPEGReadByteAdvance(sp);
        do
        {
            if (OJPEGReadByte(sp, &m) == 0)
                return (0);
        } while (m == 255);
        switch (m)
        {
            case JPEG_MARKER_SOI:
                /* this type of marker has no data, and should be skipped */
                break;
            case JPEG_MARKER_COM:
            case JPEG_MARKER_APP0:
            case JPEG_MARKER_APP0 + 1:
            case JPEG_MARKER_APP0 + 2:
            case JPEG_MARKER_APP0 + 3:
            case JPEG_MARKER_APP0 + 4:
            case JPEG_MARKER_APP0 + 5:
            case JPEG_MARKER_APP0 + 6:
            case JPEG_MARKER_APP0 + 7:
            case JPEG_MARKER_APP0 + 8:
            case JPEG_MARKER_APP0 + 9:
            case JPEG_MARKER_APP0 + 10:
            case JPEG_MARKER_APP0 + 11:
            case JPEG_MARKER_APP0 + 12:
            case JPEG_MARKER_APP0 + 13:
            case JPEG_MARKER_APP0 + 14:
            case JPEG_MARKER_APP0 + 15:
                /* this type of marker has data, but it has no use to us (and no
                 * place here) and should be skipped */
                if (OJPEGReadWord(sp, &n) == 0)
                    return (0);
                if (n < 2)
                {
                    if (sp->subsamplingcorrect == 0)
                        TIFFErrorExtR(tif, module, "Corrupt JPEG data");
                    return (0);
                }
                if (n > 2)
                    OJPEGReadSkip(sp, n - 2);
                break;
            case JPEG_MARKER_DRI:
                if (OJPEGReadHeaderInfoSecStreamDri(tif) == 0)
                    return (0);
                break;
            case JPEG_MARKER_DQT:
                if (OJPEGReadHeaderInfoSecStreamDqt(tif) == 0)
                    return (0);
                break;
            case JPEG_MARKER_DHT:
                if (OJPEGReadHeaderInfoSecStreamDht(tif) == 0)
                    return (0);
                break;
            case JPEG_MARKER_SOF0:
            case JPEG_MARKER_SOF1:
            case JPEG_MARKER_SOF3:
                if (OJPEGReadHeaderInfoSecStreamSof(tif, m) == 0)
                    return (0);
                if (sp->subsamplingcorrect != 0)
                    return (1);
                break;
            case JPEG_MARKER_SOS:
                if (sp->subsamplingcorrect != 0)
                    return (1);
                assert(sp->plane_sample_offset == 0);
                if (OJPEGReadHeaderInfoSecStreamSos(tif) == 0)
                    return (0);
                break;
            default:
                TIFFErrorExtR(tif, module,
                              "Unknown marker type %" PRIu8 " in JPEG data", m);
                return (0);
        }
    } while (m != JPEG_MARKER_SOS);
    if (sp->subsamplingcorrect)
        return (1);
    if (sp->sof_log == 0)
    {
        if (OJPEGReadHeaderInfoSecTablesQTable(tif) == 0)
            return (0);
        sp->sof_marker_id = JPEG_MARKER_SOF0;
        for (o = 0; o < sp->samples_per_pixel; o++)
            sp->sof_c[o] = o;
        sp->sof_hv[0] = ((sp->subsampling_hor << 4) | sp->subsampling_ver);
        for (o = 1; o < sp->samples_per_pixel; o++)
            sp->sof_hv[o] = 17;
        sp->sof_x = sp->strile_width;
        sp->sof_y = sp->strile_length_total;
        sp->sof_log = 1;
        if (OJPEGReadHeaderInfoSecTablesDcTable(tif) == 0)
            return (0);
        if (OJPEGReadHeaderInfoSecTablesAcTable(tif) == 0)
            return (0);
        for (o = 1; o < sp->samples_per_pixel; o++)
            sp->sos_cs[o] = o;
    }
    return (1);
}

static int OJPEGReadHeaderInfoSecStreamDri(TIFF *tif)
{
    /* This could easily cause trouble in some cases... but no such cases have
       occurred so far */
    static const char module[] = "OJPEGReadHeaderInfoSecStreamDri";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint16_t m;
    if (OJPEGReadWord(sp, &m) == 0)
        return (0);
    if (m != 4)
    {
        TIFFErrorExtR(tif, module, "Corrupt DRI marker in JPEG data");
        return (0);
    }
    if (OJPEGReadWord(sp, &m) == 0)
        return (0);
    sp->restart_interval = m;
    return (1);
}

static int OJPEGReadHeaderInfoSecStreamDqt(TIFF *tif)
{
    /* this is a table marker, and it is to be saved as a whole for exact
     * pushing on the jpeg stream later on */
    static const char module[] = "OJPEGReadHeaderInfoSecStreamDqt";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint16_t m;
    uint32_t na;
    uint8_t *nb;
    uint8_t o;
    if (OJPEGReadWord(sp, &m) == 0)
        return (0);
    if (m <= 2)
    {
        if (sp->subsamplingcorrect == 0)
            TIFFErrorExtR(tif, module, "Corrupt DQT marker in JPEG data");
        return (0);
    }
    if (sp->subsamplingcorrect != 0)
        OJPEGReadSkip(sp, m - 2);
    else
    {
        m -= 2;
        do
        {
            if (m < 65)
            {
                TIFFErrorExtR(tif, module, "Corrupt DQT marker in JPEG data");
                return (0);
            }
            na = sizeof(uint32_t) + 69;
            nb = _TIFFmallocExt(tif, na);
            if (nb == 0)
            {
                TIFFErrorExtR(tif, module, "Out of memory");
                return (0);
            }
            *(uint32_t *)nb = na;
            nb[sizeof(uint32_t)] = 255;
            nb[sizeof(uint32_t) + 1] = JPEG_MARKER_DQT;
            nb[sizeof(uint32_t) + 2] = 0;
            nb[sizeof(uint32_t) + 3] = 67;
            if (OJPEGReadBlock(sp, 65, &nb[sizeof(uint32_t) + 4]) == 0)
            {
                _TIFFfreeExt(tif, nb);
                return (0);
            }
            o = nb[sizeof(uint32_t) + 4] & 15;
            if (3 < o)
            {
                TIFFErrorExtR(tif, module, "Corrupt DQT marker in JPEG data");
                _TIFFfreeExt(tif, nb);
                return (0);
            }
            if (sp->qtable[o] != 0)
                _TIFFfreeExt(tif, sp->qtable[o]);
            sp->qtable[o] = nb;
            m -= 65;
        } while (m > 0);
    }
    return (1);
}

static int OJPEGReadHeaderInfoSecStreamDht(TIFF *tif)
{
    /* this is a table marker, and it is to be saved as a whole for exact
     * pushing on the jpeg stream later on */
    /* TODO: the following assumes there is only one table in this marker... but
     * i'm not quite sure that assumption is guaranteed correct */
    static const char module[] = "OJPEGReadHeaderInfoSecStreamDht";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint16_t m;
    uint32_t na;
    uint8_t *nb;
    uint8_t o;
    if (OJPEGReadWord(sp, &m) == 0)
        return (0);
    if (m <= 2)
    {
        if (sp->subsamplingcorrect == 0)
            TIFFErrorExtR(tif, module, "Corrupt DHT marker in JPEG data");
        return (0);
    }
    if (sp->subsamplingcorrect != 0)
    {
        OJPEGReadSkip(sp, m - 2);
    }
    else
    {
        na = sizeof(uint32_t) + 2 + m;
        nb = _TIFFmallocExt(tif, na);
        if (nb == 0)
        {
            TIFFErrorExtR(tif, module, "Out of memory");
            return (0);
        }
        *(uint32_t *)nb = na;
        nb[sizeof(uint32_t)] = 255;
        nb[sizeof(uint32_t) + 1] = JPEG_MARKER_DHT;
        nb[sizeof(uint32_t) + 2] = (m >> 8);
        nb[sizeof(uint32_t) + 3] = (m & 255);
        if (OJPEGReadBlock(sp, m - 2, &nb[sizeof(uint32_t) + 4]) == 0)
        {
            _TIFFfreeExt(tif, nb);
            return (0);
        }
        o = nb[sizeof(uint32_t) + 4];
        if ((o & 240) == 0)
        {
            if (3 < o)
            {
                TIFFErrorExtR(tif, module, "Corrupt DHT marker in JPEG data");
                _TIFFfreeExt(tif, nb);
                return (0);
            }
            if (sp->dctable[o] != 0)
                _TIFFfreeExt(tif, sp->dctable[o]);
            sp->dctable[o] = nb;
        }
        else
        {
            if ((o & 240) != 16)
            {
                TIFFErrorExtR(tif, module, "Corrupt DHT marker in JPEG data");
                _TIFFfreeExt(tif, nb);
                return (0);
            }
            o &= 15;
            if (3 < o)
            {
                TIFFErrorExtR(tif, module, "Corrupt DHT marker in JPEG data");
                _TIFFfreeExt(tif, nb);
                return (0);
            }
            if (sp->actable[o] != 0)
                _TIFFfreeExt(tif, sp->actable[o]);
            sp->actable[o] = nb;
        }
    }
    return (1);
}

static int OJPEGReadHeaderInfoSecStreamSof(TIFF *tif, uint8_t marker_id)
{
    /* this marker needs to be checked, and part of its data needs to be saved
     * for regeneration later on */
    static const char module[] = "OJPEGReadHeaderInfoSecStreamSof";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint16_t m;
    uint16_t n;
    uint8_t o;
    uint16_t p;
    uint16_t q;
    if (sp->sof_log != 0)
    {
        TIFFErrorExtR(tif, module, "Corrupt JPEG data");
        return (0);
    }
    if (sp->subsamplingcorrect == 0)
        sp->sof_marker_id = marker_id;
    /* Lf: data length */
    if (OJPEGReadWord(sp, &m) == 0)
        return (0);
    if (m < 11)
    {
        if (sp->subsamplingcorrect == 0)
            TIFFErrorExtR(tif, module, "Corrupt SOF marker in JPEG data");
        return (0);
    }
    m -= 8;
    if (m % 3 != 0)
    {
        if (sp->subsamplingcorrect == 0)
            TIFFErrorExtR(tif, module, "Corrupt SOF marker in JPEG data");
        return (0);
    }
    n = m / 3;
    if (sp->subsamplingcorrect == 0)
    {
        if (n != sp->samples_per_pixel)
        {
            TIFFErrorExtR(
                tif, module,
                "JPEG compressed data indicates unexpected number of samples");
            return (0);
        }
    }
    /* P: Sample precision */
    if (OJPEGReadByte(sp, &o) == 0)
        return (0);
    if (o != 8)
    {
        if (sp->subsamplingcorrect == 0)
            TIFFErrorExtR(tif, module,
                          "JPEG compressed data indicates unexpected number of "
                          "bits per sample");
        return (0);
    }
    /* Y: Number of lines, X: Number of samples per line */
    if (sp->subsamplingcorrect)
        OJPEGReadSkip(sp, 4);
    else
    {
        /* Y: Number of lines */
        if (OJPEGReadWord(sp, &p) == 0)
            return (0);
        if (((uint32_t)p < sp->image_length) &&
            ((uint32_t)p < sp->strile_length_total))
        {
            TIFFErrorExtR(tif, module,
                          "JPEG compressed data indicates unexpected height");
            return (0);
        }
        sp->sof_y = p;
        /* X: Number of samples per line */
        if (OJPEGReadWord(sp, &p) == 0)
            return (0);
        if (((uint32_t)p < sp->image_width) && ((uint32_t)p < sp->strile_width))
        {
            TIFFErrorExtR(tif, module,
                          "JPEG compressed data indicates unexpected width");
            return (0);
        }
        if ((uint32_t)p > sp->strile_width)
        {
            TIFFErrorExtR(tif, module,
                          "JPEG compressed data image width exceeds expected "
                          "image width");
            return (0);
        }
        sp->sof_x = p;
    }
    /* Nf: Number of image components in frame */
    if (OJPEGReadByte(sp, &o) == 0)
        return (0);
    if (o != n)
    {
        if (sp->subsamplingcorrect == 0)
            TIFFErrorExtR(tif, module, "Corrupt SOF marker in JPEG data");
        return (0);
    }
    /* per component stuff */
    /* TODO: double-check that flow implies that n cannot be as big as to make
     * us overflow sof_c, sof_hv and sof_tq arrays */
    for (q = 0; q < n; q++)
    {
        /* C: Component identifier */
        if (OJPEGReadByte(sp, &o) == 0)
            return (0);
        if (sp->subsamplingcorrect == 0)
            sp->sof_c[q] = o;
        /* H: Horizontal sampling factor, and V: Vertical sampling factor */
        if (OJPEGReadByte(sp, &o) == 0)
            return (0);
        if (sp->subsamplingcorrect != 0)
        {
            if (q == 0)
            {
                sp->subsampling_hor = (o >> 4);
                sp->subsampling_ver = (o & 15);
                if (((sp->subsampling_hor != 1) && (sp->subsampling_hor != 2) &&
                     (sp->subsampling_hor != 4)) ||
                    ((sp->subsampling_ver != 1) && (sp->subsampling_ver != 2) &&
                     (sp->subsampling_ver != 4)))
                    sp->subsampling_force_desubsampling_inside_decompression =
                        1;
            }
            else
            {
                if (o != 17)
                    sp->subsampling_force_desubsampling_inside_decompression =
                        1;
            }
        }
        else
        {
            sp->sof_hv[q] = o;
            if (sp->subsampling_force_desubsampling_inside_decompression == 0)
            {
                if (q == 0)
                {
                    if (o != ((sp->subsampling_hor << 4) | sp->subsampling_ver))
                    {
                        TIFFErrorExtR(tif, module,
                                      "JPEG compressed data indicates "
                                      "unexpected subsampling values");
                        return (0);
                    }
                }
                else
                {
                    if (o != 17)
                    {
                        TIFFErrorExtR(tif, module,
                                      "JPEG compressed data indicates "
                                      "unexpected subsampling values");
                        return (0);
                    }
                }
            }
        }
        /* Tq: Quantization table destination selector */
        if (OJPEGReadByte(sp, &o) == 0)
            return (0);
        if (sp->subsamplingcorrect == 0)
            sp->sof_tq[q] = o;
    }
    if (sp->subsamplingcorrect == 0)
        sp->sof_log = 1;
    return (1);
}

static int OJPEGReadHeaderInfoSecStreamSos(TIFF *tif)
{
    /* this marker needs to be checked, and part of its data needs to be saved
     * for regeneration later on */
    static const char module[] = "OJPEGReadHeaderInfoSecStreamSos";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint16_t m;
    uint8_t n;
    uint8_t o;
    assert(sp->subsamplingcorrect == 0);
    if (sp->sof_log == 0)
    {
        TIFFErrorExtR(tif, module, "Corrupt SOS marker in JPEG data");
        return (0);
    }
    /* Ls */
    if (OJPEGReadWord(sp, &m) == 0)
        return (0);
    if (m != 6 + sp->samples_per_pixel_per_plane * 2)
    {
        TIFFErrorExtR(tif, module, "Corrupt SOS marker in JPEG data");
        return (0);
    }
    /* Ns */
    if (OJPEGReadByte(sp, &n) == 0)
        return (0);
    if (n != sp->samples_per_pixel_per_plane)
    {
        TIFFErrorExtR(tif, module, "Corrupt SOS marker in JPEG data");
        return (0);
    }
    /* Cs, Td, and Ta */
    for (o = 0; o < sp->samples_per_pixel_per_plane; o++)
    {
        /* Cs */
        if (OJPEGReadByte(sp, &n) == 0)
            return (0);
        sp->sos_cs[sp->plane_sample_offset + o] = n;
        /* Td and Ta */
        if (OJPEGReadByte(sp, &n) == 0)
            return (0);
        sp->sos_tda[sp->plane_sample_offset + o] = n;
    }
    /* skip Ss, Se, Ah, en Al -> no check, as per Tom Lane recommendation, as
     * per LibJpeg source */
    OJPEGReadSkip(sp, 3);
    return (1);
}

static int OJPEGReadHeaderInfoSecTablesQTable(TIFF *tif)
{
    static const char module[] = "OJPEGReadHeaderInfoSecTablesQTable";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    uint8_t n;
    uint32_t oa;
    uint8_t *ob;
    uint32_t p;
    if (sp->qtable_offset[0] == 0)
    {
        TIFFErrorExtR(tif, module, "Missing JPEG tables");
        return (0);
    }
    sp->in_buffer_file_pos_log = 0;
    for (m = 0; m < sp->samples_per_pixel; m++)
    {
        if ((sp->qtable_offset[m] != 0) &&
            ((m == 0) || (sp->qtable_offset[m] != sp->qtable_offset[m - 1])))
        {
            for (n = 0; n < m - 1; n++)
            {
                if (sp->qtable_offset[m] == sp->qtable_offset[n])
                {
                    TIFFErrorExtR(tif, module, "Corrupt JpegQTables tag value");
                    return (0);
                }
            }
            oa = sizeof(uint32_t) + 69;
            ob = _TIFFmallocExt(tif, oa);
            if (ob == 0)
            {
                TIFFErrorExtR(tif, module, "Out of memory");
                return (0);
            }
            *(uint32_t *)ob = oa;
            ob[sizeof(uint32_t)] = 255;
            ob[sizeof(uint32_t) + 1] = JPEG_MARKER_DQT;
            ob[sizeof(uint32_t) + 2] = 0;
            ob[sizeof(uint32_t) + 3] = 67;
            ob[sizeof(uint32_t) + 4] = m;
            TIFFSeekFile(tif, sp->qtable_offset[m], SEEK_SET);
            p = (uint32_t)TIFFReadFile(tif, &ob[sizeof(uint32_t) + 5], 64);
            if (p != 64)
            {
                _TIFFfreeExt(tif, ob);
                return (0);
            }
            if (sp->qtable[m] != 0)
                _TIFFfreeExt(tif, sp->qtable[m]);
            sp->qtable[m] = ob;
            sp->sof_tq[m] = m;
        }
        else
            sp->sof_tq[m] = sp->sof_tq[m - 1];
    }
    return (1);
}

static int OJPEGReadHeaderInfoSecTablesDcTable(TIFF *tif)
{
    static const char module[] = "OJPEGReadHeaderInfoSecTablesDcTable";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    uint8_t n;
    uint8_t o[16];
    uint32_t p;
    uint32_t q;
    uint32_t ra;
    uint8_t *rb;
    if (sp->dctable_offset[0] == 0)
    {
        TIFFErrorExtR(tif, module, "Missing JPEG tables");
        return (0);
    }
    sp->in_buffer_file_pos_log = 0;
    for (m = 0; m < sp->samples_per_pixel; m++)
    {
        if ((sp->dctable_offset[m] != 0) &&
            ((m == 0) || (sp->dctable_offset[m] != sp->dctable_offset[m - 1])))
        {
            for (n = 0; n < m - 1; n++)
            {
                if (sp->dctable_offset[m] == sp->dctable_offset[n])
                {
                    TIFFErrorExtR(tif, module,
                                  "Corrupt JpegDcTables tag value");
                    return (0);
                }
            }
            TIFFSeekFile(tif, sp->dctable_offset[m], SEEK_SET);
            p = (uint32_t)TIFFReadFile(tif, o, 16);
            if (p != 16)
                return (0);
            q = 0;
            for (n = 0; n < 16; n++)
                q += o[n];
            ra = sizeof(uint32_t) + 21 + q;
            rb = _TIFFmallocExt(tif, ra);
            if (rb == 0)
            {
                TIFFErrorExtR(tif, module, "Out of memory");
                return (0);
            }
            *(uint32_t *)rb = ra;
            rb[sizeof(uint32_t)] = 255;
            rb[sizeof(uint32_t) + 1] = JPEG_MARKER_DHT;
            rb[sizeof(uint32_t) + 2] = (uint8_t)((19 + q) >> 8);
            rb[sizeof(uint32_t) + 3] = ((19 + q) & 255);
            rb[sizeof(uint32_t) + 4] = m;
            for (n = 0; n < 16; n++)
                rb[sizeof(uint32_t) + 5 + n] = o[n];
            p = (uint32_t)TIFFReadFile(tif, &(rb[sizeof(uint32_t) + 21]), q);
            if (p != q)
            {
                _TIFFfreeExt(tif, rb);
                return (0);
            }
            if (sp->dctable[m] != 0)
                _TIFFfreeExt(tif, sp->dctable[m]);
            sp->dctable[m] = rb;
            sp->sos_tda[m] = (m << 4);
        }
        else
            sp->sos_tda[m] = sp->sos_tda[m - 1];
    }
    return (1);
}

static int OJPEGReadHeaderInfoSecTablesAcTable(TIFF *tif)
{
    static const char module[] = "OJPEGReadHeaderInfoSecTablesAcTable";
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    uint8_t n;
    uint8_t o[16];
    uint32_t p;
    uint32_t q;
    uint32_t ra;
    uint8_t *rb;
    if (sp->actable_offset[0] == 0)
    {
        TIFFErrorExtR(tif, module, "Missing JPEG tables");
        return (0);
    }
    sp->in_buffer_file_pos_log = 0;
    for (m = 0; m < sp->samples_per_pixel; m++)
    {
        if ((sp->actable_offset[m] != 0) &&
            ((m == 0) || (sp->actable_offset[m] != sp->actable_offset[m - 1])))
        {
            for (n = 0; n < m - 1; n++)
            {
                if (sp->actable_offset[m] == sp->actable_offset[n])
                {
                    TIFFErrorExtR(tif, module,
                                  "Corrupt JpegAcTables tag value");
                    return (0);
                }
            }
            TIFFSeekFile(tif, sp->actable_offset[m], SEEK_SET);
            p = (uint32_t)TIFFReadFile(tif, o, 16);
            if (p != 16)
                return (0);
            q = 0;
            for (n = 0; n < 16; n++)
                q += o[n];
            ra = sizeof(uint32_t) + 21 + q;
            rb = _TIFFmallocExt(tif, ra);
            if (rb == 0)
            {
                TIFFErrorExtR(tif, module, "Out of memory");
                return (0);
            }
            *(uint32_t *)rb = ra;
            rb[sizeof(uint32_t)] = 255;
            rb[sizeof(uint32_t) + 1] = JPEG_MARKER_DHT;
            rb[sizeof(uint32_t) + 2] = (uint8_t)((19 + q) >> 8);
            rb[sizeof(uint32_t) + 3] = ((19 + q) & 255);
            rb[sizeof(uint32_t) + 4] = (16 | m);
            for (n = 0; n < 16; n++)
                rb[sizeof(uint32_t) + 5 + n] = o[n];
            p = (uint32_t)TIFFReadFile(tif, &(rb[sizeof(uint32_t) + 21]), q);
            if (p != q)
            {
                _TIFFfreeExt(tif, rb);
                return (0);
            }
            if (sp->actable[m] != 0)
                _TIFFfreeExt(tif, sp->actable[m]);
            sp->actable[m] = rb;
            sp->sos_tda[m] = (sp->sos_tda[m] | m);
        }
        else
            sp->sos_tda[m] = (sp->sos_tda[m] | (sp->sos_tda[m - 1] & 15));
    }
    return (1);
}

static int OJPEGReadBufferFill(OJPEGState *sp)
{
    uint16_t m;
    tmsize_t n;
    /* TODO: double-check: when subsamplingcorrect is set, no call to
     * TIFFErrorExt or TIFFWarningExt should be made in any other case, seek or
     * read errors should be passed through */
    do
    {
        if (sp->in_buffer_file_togo != 0)
        {
            if (sp->in_buffer_file_pos_log == 0)
            {
                TIFFSeekFile(sp->tif, sp->in_buffer_file_pos, SEEK_SET);
                sp->in_buffer_file_pos_log = 1;
            }
            m = OJPEG_BUFFER;
            if ((uint64_t)m > sp->in_buffer_file_togo)
                m = (uint16_t)sp->in_buffer_file_togo;
            n = TIFFReadFile(sp->tif, sp->in_buffer, (tmsize_t)m);
            if (n == 0)
                return (0);
            assert(n > 0);
            assert(n <= OJPEG_BUFFER);
            assert(n < 65536);
            assert((uint64_t)n <= sp->in_buffer_file_togo);
            m = (uint16_t)n;
            sp->in_buffer_togo = m;
            sp->in_buffer_cur = sp->in_buffer;
            sp->in_buffer_file_togo -= m;
            sp->in_buffer_file_pos += m;
            break;
        }
        sp->in_buffer_file_pos_log = 0;
        switch (sp->in_buffer_source)
        {
            case osibsNotSetYet:
                if (sp->jpeg_interchange_format != 0)
                {
                    sp->in_buffer_file_pos = sp->jpeg_interchange_format;
                    sp->in_buffer_file_togo =
                        sp->jpeg_interchange_format_length;
                }
                sp->in_buffer_source = osibsJpegInterchangeFormat;
                break;
            case osibsJpegInterchangeFormat:
                sp->in_buffer_source = osibsStrile;
                break;
            case osibsStrile:
                if (sp->in_buffer_next_strile == sp->in_buffer_strile_count)
                    sp->in_buffer_source = osibsEof;
                else
                {
                    int err = 0;
                    sp->in_buffer_file_pos = TIFFGetStrileOffsetWithErr(
                        sp->tif, sp->in_buffer_next_strile, &err);
                    if (err)
                        return 0;
                    if (sp->in_buffer_file_pos != 0)
                    {
                        uint64_t bytecount = TIFFGetStrileByteCountWithErr(
                            sp->tif, sp->in_buffer_next_strile, &err);
                        if (err)
                            return 0;
                        if (sp->in_buffer_file_pos >= sp->file_size)
                            sp->in_buffer_file_pos = 0;
                        else if (bytecount == 0)
                            sp->in_buffer_file_togo =
                                sp->file_size - sp->in_buffer_file_pos;
                        else
                        {
                            sp->in_buffer_file_togo = bytecount;
                            if (sp->in_buffer_file_togo == 0)
                                sp->in_buffer_file_pos = 0;
                            else if (sp->in_buffer_file_pos >
                                         UINT64_MAX - sp->in_buffer_file_togo ||
                                     sp->in_buffer_file_pos +
                                             sp->in_buffer_file_togo >
                                         sp->file_size)
                                sp->in_buffer_file_togo =
                                    sp->file_size - sp->in_buffer_file_pos;
                        }
                    }
                    sp->in_buffer_next_strile++;
                }
                break;
            default:
                return (0);
        }
    } while (1);
    return (1);
}

static int OJPEGReadByte(OJPEGState *sp, uint8_t *byte)
{
    if (sp->in_buffer_togo == 0)
    {
        if (OJPEGReadBufferFill(sp) == 0)
            return (0);
        assert(sp->in_buffer_togo > 0);
    }
    *byte = *(sp->in_buffer_cur);
    sp->in_buffer_cur++;
    sp->in_buffer_togo--;
    return (1);
}

static int OJPEGReadBytePeek(OJPEGState *sp, uint8_t *byte)
{
    if (sp->in_buffer_togo == 0)
    {
        if (OJPEGReadBufferFill(sp) == 0)
            return (0);
        assert(sp->in_buffer_togo > 0);
    }
    *byte = *(sp->in_buffer_cur);
    return (1);
}

static void OJPEGReadByteAdvance(OJPEGState *sp)
{
    assert(sp->in_buffer_togo > 0);
    sp->in_buffer_cur++;
    sp->in_buffer_togo--;
}

static int OJPEGReadWord(OJPEGState *sp, uint16_t *word)
{
    uint8_t m;
    if (OJPEGReadByte(sp, &m) == 0)
        return (0);
    *word = (m << 8);
    if (OJPEGReadByte(sp, &m) == 0)
        return (0);
    *word |= m;
    return (1);
}

static int OJPEGReadBlock(OJPEGState *sp, uint16_t len, void *mem)
{
    uint16_t mlen;
    uint8_t *mmem;
    uint16_t n;
    assert(len > 0);
    mlen = len;
    mmem = mem;
    do
    {
        if (sp->in_buffer_togo == 0)
        {
            if (OJPEGReadBufferFill(sp) == 0)
                return (0);
            assert(sp->in_buffer_togo > 0);
        }
        n = mlen;
        if (n > sp->in_buffer_togo)
            n = sp->in_buffer_togo;
        _TIFFmemcpy(mmem, sp->in_buffer_cur, n);
        sp->in_buffer_cur += n;
        sp->in_buffer_togo -= n;
        mlen -= n;
        mmem += n;
    } while (mlen > 0);
    return (1);
}

static void OJPEGReadSkip(OJPEGState *sp, uint16_t len)
{
    uint16_t m;
    uint16_t n;
    m = len;
    n = m;
    if (n > sp->in_buffer_togo)
        n = sp->in_buffer_togo;
    sp->in_buffer_cur += n;
    sp->in_buffer_togo -= n;
    m -= n;
    if (m > 0)
    {
        assert(sp->in_buffer_togo == 0);
        n = m;
        if ((uint64_t)n > sp->in_buffer_file_togo)
            n = (uint16_t)sp->in_buffer_file_togo;
        sp->in_buffer_file_pos += n;
        sp->in_buffer_file_togo -= n;
        sp->in_buffer_file_pos_log = 0;
        /* we don't skip past jpeginterchangeformat/strile block...
         * if that is asked from us, we're dealing with totally bazurk
         * data anyway, and we've not seen this happening on any
         * testfile, so we might as well likely cause some other
         * meaningless error to be passed at some later time
         */
    }
}

static int OJPEGWriteStream(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    *len = 0;
    do
    {
        assert(sp->out_state <= ososEoi);
        switch (sp->out_state)
        {
            case ososSoi:
                OJPEGWriteStreamSoi(tif, mem, len);
                break;
            case ososQTable0:
                OJPEGWriteStreamQTable(tif, 0, mem, len);
                break;
            case ososQTable1:
                OJPEGWriteStreamQTable(tif, 1, mem, len);
                break;
            case ososQTable2:
                OJPEGWriteStreamQTable(tif, 2, mem, len);
                break;
            case ososQTable3:
                OJPEGWriteStreamQTable(tif, 3, mem, len);
                break;
            case ososDcTable0:
                OJPEGWriteStreamDcTable(tif, 0, mem, len);
                break;
            case ososDcTable1:
                OJPEGWriteStreamDcTable(tif, 1, mem, len);
                break;
            case ososDcTable2:
                OJPEGWriteStreamDcTable(tif, 2, mem, len);
                break;
            case ososDcTable3:
                OJPEGWriteStreamDcTable(tif, 3, mem, len);
                break;
            case ososAcTable0:
                OJPEGWriteStreamAcTable(tif, 0, mem, len);
                break;
            case ososAcTable1:
                OJPEGWriteStreamAcTable(tif, 1, mem, len);
                break;
            case ososAcTable2:
                OJPEGWriteStreamAcTable(tif, 2, mem, len);
                break;
            case ososAcTable3:
                OJPEGWriteStreamAcTable(tif, 3, mem, len);
                break;
            case ososDri:
                OJPEGWriteStreamDri(tif, mem, len);
                break;
            case ososSof:
                OJPEGWriteStreamSof(tif, mem, len);
                break;
            case ososSos:
                OJPEGWriteStreamSos(tif, mem, len);
                break;
            case ososCompressed:
                if (OJPEGWriteStreamCompressed(tif, mem, len) == 0)
                    return (0);
                break;
            case ososRst:
                OJPEGWriteStreamRst(tif, mem, len);
                break;
            case ososEoi:
                OJPEGWriteStreamEoi(tif, mem, len);
                break;
        }
    } while (*len == 0);
    return (1);
}

static void OJPEGWriteStreamSoi(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    assert(OJPEG_BUFFER >= 2);
    sp->out_buffer[0] = 255;
    sp->out_buffer[1] = JPEG_MARKER_SOI;
    *len = 2;
    *mem = (void *)sp->out_buffer;
    sp->out_state++;
}

static void OJPEGWriteStreamQTable(TIFF *tif, uint8_t table_index, void **mem,
                                   uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    if (sp->qtable[table_index] != 0)
    {
        *mem = (void *)(sp->qtable[table_index] + sizeof(uint32_t));
        *len = *((uint32_t *)sp->qtable[table_index]) - sizeof(uint32_t);
    }
    sp->out_state++;
}

static void OJPEGWriteStreamDcTable(TIFF *tif, uint8_t table_index, void **mem,
                                    uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    if (sp->dctable[table_index] != 0)
    {
        *mem = (void *)(sp->dctable[table_index] + sizeof(uint32_t));
        *len = *((uint32_t *)sp->dctable[table_index]) - sizeof(uint32_t);
    }
    sp->out_state++;
}

static void OJPEGWriteStreamAcTable(TIFF *tif, uint8_t table_index, void **mem,
                                    uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    if (sp->actable[table_index] != 0)
    {
        *mem = (void *)(sp->actable[table_index] + sizeof(uint32_t));
        *len = *((uint32_t *)sp->actable[table_index]) - sizeof(uint32_t);
    }
    sp->out_state++;
}

static void OJPEGWriteStreamDri(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    assert(OJPEG_BUFFER >= 6);
    if (sp->restart_interval != 0)
    {
        sp->out_buffer[0] = 255;
        sp->out_buffer[1] = JPEG_MARKER_DRI;
        sp->out_buffer[2] = 0;
        sp->out_buffer[3] = 4;
        sp->out_buffer[4] = (sp->restart_interval >> 8);
        sp->out_buffer[5] = (sp->restart_interval & 255);
        *len = 6;
        *mem = (void *)sp->out_buffer;
    }
    sp->out_state++;
}

static void OJPEGWriteStreamSof(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    assert(OJPEG_BUFFER >= 2 + 8 + sp->samples_per_pixel_per_plane * 3);
    assert(255 >= 8 + sp->samples_per_pixel_per_plane * 3);
    sp->out_buffer[0] = 255;
    sp->out_buffer[1] = sp->sof_marker_id;
    /* Lf */
    sp->out_buffer[2] = 0;
    sp->out_buffer[3] = 8 + sp->samples_per_pixel_per_plane * 3;
    /* P */
    sp->out_buffer[4] = 8;
    /* Y */
    sp->out_buffer[5] = (uint8_t)(sp->sof_y >> 8);
    sp->out_buffer[6] = (sp->sof_y & 255);
    /* X */
    sp->out_buffer[7] = (uint8_t)(sp->sof_x >> 8);
    sp->out_buffer[8] = (sp->sof_x & 255);
    /* Nf */
    sp->out_buffer[9] = sp->samples_per_pixel_per_plane;
    for (m = 0; m < sp->samples_per_pixel_per_plane; m++)
    {
        /* C */
        sp->out_buffer[10 + m * 3] = sp->sof_c[sp->plane_sample_offset + m];
        /* H and V */
        sp->out_buffer[10 + m * 3 + 1] =
            sp->sof_hv[sp->plane_sample_offset + m];
        /* Tq */
        sp->out_buffer[10 + m * 3 + 2] =
            sp->sof_tq[sp->plane_sample_offset + m];
    }
    *len = 10 + sp->samples_per_pixel_per_plane * 3;
    *mem = (void *)sp->out_buffer;
    sp->out_state++;
}

static void OJPEGWriteStreamSos(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    uint8_t m;
    assert(OJPEG_BUFFER >= 2 + 6 + sp->samples_per_pixel_per_plane * 2);
    assert(255 >= 6 + sp->samples_per_pixel_per_plane * 2);
    sp->out_buffer[0] = 255;
    sp->out_buffer[1] = JPEG_MARKER_SOS;
    /* Ls */
    sp->out_buffer[2] = 0;
    sp->out_buffer[3] = 6 + sp->samples_per_pixel_per_plane * 2;
    /* Ns */
    sp->out_buffer[4] = sp->samples_per_pixel_per_plane;
    for (m = 0; m < sp->samples_per_pixel_per_plane; m++)
    {
        /* Cs */
        sp->out_buffer[5 + m * 2] = sp->sos_cs[sp->plane_sample_offset + m];
        /* Td and Ta */
        sp->out_buffer[5 + m * 2 + 1] =
            sp->sos_tda[sp->plane_sample_offset + m];
    }
    /* Ss */
    sp->out_buffer[5 + sp->samples_per_pixel_per_plane * 2] = 0;
    /* Se */
    sp->out_buffer[5 + sp->samples_per_pixel_per_plane * 2 + 1] = 63;
    /* Ah and Al */
    sp->out_buffer[5 + sp->samples_per_pixel_per_plane * 2 + 2] = 0;
    *len = 8 + sp->samples_per_pixel_per_plane * 2;
    *mem = (void *)sp->out_buffer;
    sp->out_state++;
}

static int OJPEGWriteStreamCompressed(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    if (sp->in_buffer_togo == 0)
    {
        if (OJPEGReadBufferFill(sp) == 0)
            return (0);
        assert(sp->in_buffer_togo > 0);
    }
    *len = sp->in_buffer_togo;
    *mem = (void *)sp->in_buffer_cur;
    sp->in_buffer_togo = 0;
    if (sp->in_buffer_file_togo == 0)
    {
        switch (sp->in_buffer_source)
        {
            case osibsStrile:
                if (sp->in_buffer_next_strile < sp->in_buffer_strile_count)
                    sp->out_state = ososRst;
                else
                    sp->out_state = ososEoi;
                break;
            case osibsEof:
                sp->out_state = ososEoi;
                break;
            default:
                break;
        }
    }
    return (1);
}

static void OJPEGWriteStreamRst(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    assert(OJPEG_BUFFER >= 2);
    sp->out_buffer[0] = 255;
    sp->out_buffer[1] = JPEG_MARKER_RST0 + sp->restart_index;
    sp->restart_index++;
    if (sp->restart_index == 8)
        sp->restart_index = 0;
    *len = 2;
    *mem = (void *)sp->out_buffer;
    sp->out_state = ososCompressed;
}

static void OJPEGWriteStreamEoi(TIFF *tif, void **mem, uint32_t *len)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    assert(OJPEG_BUFFER >= 2);
    sp->out_buffer[0] = 255;
    sp->out_buffer[1] = JPEG_MARKER_EOI;
    *len = 2;
    *mem = (void *)sp->out_buffer;
}

#ifndef LIBJPEG_ENCAP_EXTERNAL
static int jpeg_create_decompress_encap(OJPEGState *sp,
                                        jpeg_decompress_struct *cinfo)
{
    if (SETJMP(sp->exit_jmpbuf))
        return 0;
    else
    {
        jpeg_create_decompress(cinfo);
        return 1;
    }
}
#endif

#ifndef LIBJPEG_ENCAP_EXTERNAL
static int jpeg_read_header_encap(OJPEGState *sp, jpeg_decompress_struct *cinfo,
                                  uint8_t require_image)
{
    if (SETJMP(sp->exit_jmpbuf))
        return 0;
    else
    {
        jpeg_read_header(cinfo, require_image);
        return 1;
    }
}
#endif

#ifndef LIBJPEG_ENCAP_EXTERNAL
static int jpeg_start_decompress_encap(OJPEGState *sp,
                                       jpeg_decompress_struct *cinfo)
{
    if (SETJMP(sp->exit_jmpbuf))
        return 0;
    else
    {
        jpeg_start_decompress(cinfo);
        return 1;
    }
}
#endif

#ifndef LIBJPEG_ENCAP_EXTERNAL
static int jpeg_read_scanlines_encap(OJPEGState *sp,
                                     jpeg_decompress_struct *cinfo,
                                     void *scanlines, uint32_t max_lines)
{
    if (SETJMP(sp->exit_jmpbuf))
        return 0;
    else
    {
        jpeg_read_scanlines(cinfo, scanlines, max_lines);
        return 1;
    }
}
#endif

#ifndef LIBJPEG_ENCAP_EXTERNAL
static int jpeg_read_raw_data_encap(OJPEGState *sp,
                                    jpeg_decompress_struct *cinfo, void *data,
                                    uint32_t max_lines)
{
    if (SETJMP(sp->exit_jmpbuf))
        return 0;
    else
    {
        jpeg_read_raw_data(cinfo, data, max_lines);
        return 1;
    }
}
#endif

#ifndef LIBJPEG_ENCAP_EXTERNAL
static void jpeg_encap_unwind(TIFF *tif)
{
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    LONGJMP(sp->exit_jmpbuf, 1);
}
#endif

static void OJPEGLibjpegJpegErrorMgrOutputMessage(jpeg_common_struct *cinfo)
{
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    TIFFWarningExtR(((TIFF *)(cinfo->client_data)), "LibJpeg", "%s", buffer);
}

static void OJPEGLibjpegJpegErrorMgrErrorExit(jpeg_common_struct *cinfo)
{
    char buffer[JMSG_LENGTH_MAX];
    (*cinfo->err->format_message)(cinfo, buffer);
    TIFFErrorExtR(((TIFF *)(cinfo->client_data)), "LibJpeg", "%s", buffer);
    jpeg_encap_unwind((TIFF *)(cinfo->client_data));
}

static void OJPEGLibjpegJpegSourceMgrInitSource(jpeg_decompress_struct *cinfo)
{
    (void)cinfo;
}

static boolean
OJPEGLibjpegJpegSourceMgrFillInputBuffer(jpeg_decompress_struct *cinfo)
{
    TIFF *tif = (TIFF *)cinfo->client_data;
    OJPEGState *sp = (OJPEGState *)tif->tif_data;
    void *mem = 0;
    uint32_t len = 0U;
    if (OJPEGWriteStream(tif, &mem, &len) == 0)
    {
        TIFFErrorExtR(tif, "LibJpeg", "Premature end of JPEG data");
        jpeg_encap_unwind(tif);
    }
    sp->libjpeg_jpeg_source_mgr.bytes_in_buffer = len;
    sp->libjpeg_jpeg_source_mgr.next_input_byte = mem;
    return (1);
}

static void
OJPEGLibjpegJpegSourceMgrSkipInputData(jpeg_decompress_struct *cinfo,
                                       long num_bytes)
{
    TIFF *tif = (TIFF *)cinfo->client_data;
    (void)num_bytes;
    TIFFErrorExtR(tif, "LibJpeg", "Unexpected error");
    jpeg_encap_unwind(tif);
}

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702) /* unreachable code */
#endif
static boolean
OJPEGLibjpegJpegSourceMgrResyncToRestart(jpeg_decompress_struct *cinfo,
                                         int desired)
{
    TIFF *tif = (TIFF *)cinfo->client_data;
    (void)desired;
    TIFFErrorExtR(tif, "LibJpeg", "Unexpected error");
    jpeg_encap_unwind(tif);
    return (0);
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

static void OJPEGLibjpegJpegSourceMgrTermSource(jpeg_decompress_struct *cinfo)
{
    (void)cinfo;
}

#endif
