/*
* Copyright (c) 2018, Mapbox
* Author: <norman.barker at mapbox.com>
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

#include "tiffiop.h"
#ifdef WEBP_SUPPORT
/*
 * TIFF Library.
 *
 * WEBP Compression Support
 *
 */

#include "webp/decode.h"
#include "webp/encode.h"

#include <stdio.h>

#define LSTATE_INIT_DECODE 0x01
#define LSTATE_INIT_ENCODE 0x02
/*
 * State block for each open TIFF
 * file using WEBP compression/decompression.
 */
typedef struct {
  uint16           nSamples;               /* number of samples per pixel */
  
  int              lossless;               /* lossy/lossless compression */
  int              quality_level;          /* compression level */
  WebPPicture      sPicture;               /* WebP Picture */
  WebPConfig       sEncoderConfig;         /* WebP encoder config */
  uint8*           pBuffer;                /* buffer to hold raw data on encoding */
  unsigned int     buffer_offset;          /* current offset into the buffer */
  unsigned int     buffer_size;
  
  WebPIDecoder*    psDecoder;              /* WebPIDecoder */
  WebPDecBuffer    sDecBuffer;             /* Decoder buffer */
  int              last_y;                 /* Last row decoded */
  
  int              state;                  /* state flags */
  
	TIFFVGetMethod   vgetparent;             /* super-class method */
	TIFFVSetMethod   vsetparent;             /* super-class method */
} WebPState;

#define LState(tif)            ((WebPState*) (tif)->tif_data)
#define DecoderState(tif)       LState(tif)
#define EncoderState(tif)       LState(tif)

static int TWebPEncode(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s);
static int TWebPDecode(TIFF* tif, uint8* op, tmsize_t occ, uint16 s);

static
int TWebPDatasetWriter(const uint8_t* data, size_t data_size,
                      const WebPPicture* const picture)
{
  static const char module[] = "TWebPDatasetWriter";
  TIFF* tif = (TIFF*)(picture->custom_ptr);
  
  if ( (tif->tif_rawcc + (tmsize_t)data_size) > tif->tif_rawdatasize ) {
    TIFFErrorExt(tif->tif_clientdata, module,
                 "Buffer too small by " TIFF_SIZE_FORMAT " bytes.",
                 (size_t) (tif->tif_rawcc + data_size - tif->tif_rawdatasize));
    return 0;
  } else {
    _TIFFmemcpy(tif->tif_rawcp, data, data_size);
    tif->tif_rawcc += data_size;
    tif->tif_rawcp += data_size;
    return 1;    
  }
}

/*
 * Encode a chunk of pixels.
 */
static int
TWebPEncode(TIFF* tif, uint8* bp, tmsize_t cc, uint16 s)
{
  static const char module[] = "TWebPEncode";
  WebPState *sp = EncoderState(tif);
  (void) s;

  assert(sp != NULL);
  assert(sp->state == LSTATE_INIT_ENCODE);
    
  if( (uint64)sp->buffer_offset +
                            (uint64)cc > sp->buffer_size )
  {
      TIFFErrorExt(tif->tif_clientdata, module,
                   "Too many bytes to be written");
      return 0;
  }

  memcpy(sp->pBuffer + sp->buffer_offset,
         bp, cc);
  sp->buffer_offset += (unsigned)cc;

  return 1;
  
}

static int
TWebPDecode(TIFF* tif, uint8* op, tmsize_t occ, uint16 s)
{
  static const char module[] = "WebPDecode";
  VP8StatusCode status = VP8_STATUS_OK;
  WebPState *sp = DecoderState(tif);
  (void) s;  

  assert(sp != NULL);
  assert(sp->state == LSTATE_INIT_DECODE);
  
  if (occ % sp->sDecBuffer.u.RGBA.stride)
  {
    TIFFErrorExt(tif->tif_clientdata, module,
                 "Fractional scanlines cannot be read");
    return 0;
  }

  status = WebPIAppend(sp->psDecoder, tif->tif_rawcp, tif->tif_rawcc);

  if (status != VP8_STATUS_OK && status != VP8_STATUS_SUSPENDED) {
    if (status == VP8_STATUS_INVALID_PARAM) {
       TIFFErrorExt(tif->tif_clientdata, module,
         "Invalid parameter used.");      
    } else if (status == VP8_STATUS_OUT_OF_MEMORY) {
      TIFFErrorExt(tif->tif_clientdata, module,
        "Out of memory.");         
    } else {
      TIFFErrorExt(tif->tif_clientdata, module,
        "Unrecognized error.");   
    }
    return 0;
  } else {
    int current_y, stride;
    uint8_t* buf;

    /* Returns the RGB/A image decoded so far */
    buf = WebPIDecGetRGB(sp->psDecoder, &current_y, NULL, NULL, &stride);
    
    if ((buf != NULL) &&
        (occ <= stride * (current_y - sp->last_y))) {
      memcpy(op,   
         buf + (sp->last_y * stride),
         occ);

      tif->tif_rawcp += tif->tif_rawcc;
      tif->tif_rawcc = 0;
      sp->last_y += occ / sp->sDecBuffer.u.RGBA.stride;
      return 1;
    } else {
      TIFFErrorExt(tif->tif_clientdata, module, "Unable to decode WebP data."); 
      return 0;
    }
  }
}

static int
TWebPFixupTags(TIFF* tif)
{
  (void) tif;
  if (tif->tif_dir.td_planarconfig != PLANARCONFIG_CONTIG) {
    static const char module[] = "TWebPFixupTags";
    TIFFErrorExt(tif->tif_clientdata, module,
      "TIFF WEBP requires data to be stored contiguously in RGB e.g. RGBRGBRGB "
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
      "or RGBARGBARGBA"
#endif
    );
    return 0;
  }
  return 1;
}

static int
TWebPSetupDecode(TIFF* tif)
{
  static const char module[] = "WebPSetupDecode";
  uint16 nBitsPerSample = tif->tif_dir.td_bitspersample;
  uint16 sampleFormat = tif->tif_dir.td_sampleformat;

  WebPState* sp = DecoderState(tif);
  assert(sp != NULL);

  sp->nSamples = tif->tif_dir.td_samplesperpixel;

  /* check band count */
  if ( sp->nSamples != 3
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
    && sp->nSamples != 4
#endif
  )
  {
    TIFFErrorExt(tif->tif_clientdata, module,
      "WEBP driver doesn't support %d bands. Must be 3 (RGB) "
  #if WEBP_ENCODER_ABI_VERSION >= 0x0100
      "or 4 (RGBA) "
  #endif
    "bands.",
    sp->nSamples );
    return 0;
  }

  /* check bits per sample and data type */
  if ((nBitsPerSample != 8) && (sampleFormat != 1)) {
    TIFFErrorExt(tif->tif_clientdata, module,
                "WEBP driver requires 8 bit unsigned data");
    return 0;
  }
  
  /* if we were last encoding, terminate this mode */
  if (sp->state & LSTATE_INIT_ENCODE) {
      WebPPictureFree(&sp->sPicture);
      if (sp->pBuffer != NULL) {
        _TIFFfree(sp->pBuffer);
        sp->pBuffer = NULL;
      }
      sp->buffer_offset = 0;
      sp->state = 0;
  }

  sp->state |= LSTATE_INIT_DECODE;

  return 1;
}

/*
* Setup state for decoding a strip.
*/
static int
TWebPPreDecode(TIFF* tif, uint16 s)
{
  static const char module[] = "TWebPPreDecode";
  uint32 segment_width, segment_height;
  WebPState* sp = DecoderState(tif);
  TIFFDirectory* td = &tif->tif_dir;
  (void) s;
  assert(sp != NULL);
  
  if (isTiled(tif)) {
    segment_width = td->td_tilewidth;
    segment_height = td->td_tilelength;
  } else {
    segment_width = td->td_imagewidth;
    segment_height = td->td_imagelength - tif->tif_row;
    if (segment_height > td->td_rowsperstrip)
      segment_height = td->td_rowsperstrip;
  }

  if( (sp->state & LSTATE_INIT_DECODE) == 0 )
      tif->tif_setupdecode(tif);
      
  if (sp->psDecoder != NULL) {
    WebPIDelete(sp->psDecoder);
    WebPFreeDecBuffer(&sp->sDecBuffer);
    sp->psDecoder = NULL;
  }

  sp->last_y = 0;
  
  WebPInitDecBuffer(&sp->sDecBuffer);
  
  sp->sDecBuffer.is_external_memory = 0;
  sp->sDecBuffer.width = segment_width;
  sp->sDecBuffer.height = segment_height;
  sp->sDecBuffer.u.RGBA.stride = segment_width * sp->nSamples;
  sp->sDecBuffer.u.RGBA.size = segment_width * sp->nSamples * segment_height;
  
  if (sp->nSamples > 3) {
    sp->sDecBuffer.colorspace = MODE_RGBA;
  } else {
    sp->sDecBuffer.colorspace = MODE_RGB;
  }
  
  sp->psDecoder = WebPINewDecoder(&sp->sDecBuffer);
  
  if (sp->psDecoder == NULL) {
    TIFFErrorExt(tif->tif_clientdata, module,
                "Unable to allocate WebP decoder.");
    return 0;
  }
  
  return 1;
}

static int
TWebPSetupEncode(TIFF* tif)
{
  static const char module[] = "WebPSetupEncode";
  uint16 nBitsPerSample = tif->tif_dir.td_bitspersample;
  uint16 sampleFormat = tif->tif_dir.td_sampleformat;
  
  WebPState* sp = EncoderState(tif);
  assert(sp != NULL);

  sp->nSamples = tif->tif_dir.td_samplesperpixel;

  /* check band count */
  if ( sp->nSamples != 3
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
    && sp->nSamples != 4
#endif
  )
  {
    TIFFErrorExt(tif->tif_clientdata, module,
      "WEBP driver doesn't support %d bands. Must be 3 (RGB) "
#if WEBP_ENCODER_ABI_VERSION >= 0x0100
      "or 4 (RGBA) "
#endif
    "bands.",
    sp->nSamples );
    return 0;
  }
  
  /* check bits per sample and data type */
  if ((nBitsPerSample != 8) && (sampleFormat != 1)) {
    TIFFErrorExt(tif->tif_clientdata, module,
                "WEBP driver requires 8 bit unsigned data");
    return 0;
  }
  
  if (sp->state & LSTATE_INIT_DECODE) {
    WebPIDelete(sp->psDecoder);
    WebPFreeDecBuffer(&sp->sDecBuffer);
    sp->psDecoder = NULL;
    sp->last_y = 0;
    sp->state = 0;
  }

  sp->state |= LSTATE_INIT_ENCODE;

  if (!WebPPictureInit(&sp->sPicture)) {
    TIFFErrorExt(tif->tif_clientdata, module,
        "Error initializing WebP picture.");
    return 0;
  }

  if (!WebPConfigInitInternal(&sp->sEncoderConfig, WEBP_PRESET_DEFAULT,
                              sp->quality_level,
                              WEBP_ENCODER_ABI_VERSION)) {
    TIFFErrorExt(tif->tif_clientdata, module,
      "Error creating WebP encoder configuration.");
    return 0;
  }

  // WebPConfigInitInternal above sets lossless to false
  #if WEBP_ENCODER_ABI_VERSION >= 0x0100
    sp->sEncoderConfig.lossless = sp->lossless;
    if (sp->lossless) {
      sp->sPicture.use_argb = 1;
    }
  #endif

  if (!WebPValidateConfig(&sp->sEncoderConfig)) {
    TIFFErrorExt(tif->tif_clientdata, module,
      "Error with WebP encoder configuration.");
    return 0;
  }

  return 1;
}

/*
* Reset encoding state at the start of a strip.
*/
static int
TWebPPreEncode(TIFF* tif, uint16 s)
{
  static const char module[] = "TWebPPreEncode";
  uint32 segment_width, segment_height;
  WebPState *sp = EncoderState(tif);
  TIFFDirectory* td = &tif->tif_dir;

  (void) s;

  assert(sp != NULL);
  if( sp->state != LSTATE_INIT_ENCODE )
    tif->tif_setupencode(tif);

  /*
   * Set encoding parameters for this strip/tile.
   */
  if (isTiled(tif)) {
    segment_width = td->td_tilewidth;
    segment_height = td->td_tilelength;
  } else {
    segment_width = td->td_imagewidth;
    segment_height = td->td_imagelength - tif->tif_row;
    if (segment_height > td->td_rowsperstrip)
      segment_height = td->td_rowsperstrip;
  }

  if( segment_width > 16383 || segment_height > 16383 ) {
      TIFFErrorExt(tif->tif_clientdata, module, 
                   "WEBP maximum image dimensions are 16383 x 16383.");
      return 0;
  }

  /* set up buffer for raw data */
  /* given above check and that nSamples <= 4, buffer_size is <= 1 GB */
  sp->buffer_size = segment_width * segment_height * sp->nSamples;
  
  if (sp->pBuffer != NULL) {
      _TIFFfree(sp->pBuffer);
      sp->pBuffer = NULL;    
  }
  
  sp->pBuffer = _TIFFmalloc(sp->buffer_size);
  if( !sp->pBuffer) {
      TIFFErrorExt(tif->tif_clientdata, module, "Cannot allocate buffer");
      return 0;
  }
  sp->buffer_offset = 0;

  sp->sPicture.width = segment_width;
  sp->sPicture.height = segment_height;
  sp->sPicture.writer = TWebPDatasetWriter;
  sp->sPicture.custom_ptr = tif;

  return 1;
}

/*
* Finish off an encoded strip by flushing it.
*/
static int
TWebPPostEncode(TIFF* tif)
{
  static const char module[] = "WebPPostEncode";
  int64_t stride;
  WebPState *sp = EncoderState(tif);
  assert(sp != NULL);

  assert(sp->state == LSTATE_INIT_ENCODE);

  stride = (int64_t)sp->sPicture.width * sp->nSamples;

#if WEBP_ENCODER_ABI_VERSION >= 0x0100
  if (sp->nSamples == 4) {
      if (!WebPPictureImportRGBA(&sp->sPicture, sp->pBuffer, (int)stride)) {
          TIFFErrorExt(tif->tif_clientdata, module,
                    "WebPPictureImportRGBA() failed" );
          return 0;
      }
  }
  else
#endif
  if (!WebPPictureImportRGB(&sp->sPicture, sp->pBuffer, (int)stride)) {
      TIFFErrorExt(tif->tif_clientdata, module,
                    "WebPPictureImportRGB() failed");
      return 0;
  }
  
  if (!WebPEncode(&sp->sEncoderConfig, &sp->sPicture)) {

#if WEBP_ENCODER_ABI_VERSION >= 0x0100
    const char* pszErrorMsg = NULL;
    switch(sp->sPicture.error_code) {
    case VP8_ENC_ERROR_OUT_OF_MEMORY:
        pszErrorMsg = "Out of memory"; break;
    case VP8_ENC_ERROR_BITSTREAM_OUT_OF_MEMORY:
        pszErrorMsg = "Out of memory while flushing bits"; break;
    case VP8_ENC_ERROR_NULL_PARAMETER:
        pszErrorMsg = "A pointer parameter is NULL"; break;
    case VP8_ENC_ERROR_INVALID_CONFIGURATION:
        pszErrorMsg = "Configuration is invalid"; break;
    case VP8_ENC_ERROR_BAD_DIMENSION:
        pszErrorMsg = "Picture has invalid width/height"; break;
    case VP8_ENC_ERROR_PARTITION0_OVERFLOW:
        pszErrorMsg = "Partition is bigger than 512k. Try using less "
            "SEGMENTS, or increase PARTITION_LIMIT value";
        break;
    case VP8_ENC_ERROR_PARTITION_OVERFLOW:
        pszErrorMsg = "Partition is bigger than 16M";
        break;
    case VP8_ENC_ERROR_BAD_WRITE:
        pszErrorMsg = "Error while fludshing bytes"; break;
    case VP8_ENC_ERROR_FILE_TOO_BIG:
        pszErrorMsg = "File is bigger than 4G"; break;
    case VP8_ENC_ERROR_USER_ABORT:
        pszErrorMsg = "User interrupted";
        break;
    default:
        TIFFErrorExt(tif->tif_clientdata, module,
                "WebPEncode returned an unknown error code: %d",
                sp->sPicture.error_code);
        pszErrorMsg = "Unknown WebP error type.";
        break;
    }
    TIFFErrorExt(tif->tif_clientdata, module,
             "WebPEncode() failed : %s", pszErrorMsg);
#else
    TIFFErrorExt(tif->tif_clientdata, module,
             "Error in WebPEncode()");
#endif
    return 0;
  }

  sp->sPicture.custom_ptr = NULL;

  if (!TIFFFlushData1(tif))
  {
    TIFFErrorExt(tif->tif_clientdata, module,
      "Error flushing TIFF WebP encoder.");
    return 0;
  }

  return 1;
}

static void
TWebPCleanup(TIFF* tif)
{
  WebPState* sp = LState(tif);

  assert(sp != 0);

  tif->tif_tagmethods.vgetfield = sp->vgetparent;
  tif->tif_tagmethods.vsetfield = sp->vsetparent;

  if (sp->state & LSTATE_INIT_ENCODE) {
    WebPPictureFree(&sp->sPicture);
  }

  if (sp->psDecoder != NULL) {
    WebPIDelete(sp->psDecoder);
    WebPFreeDecBuffer(&sp->sDecBuffer);
    sp->psDecoder = NULL;
    sp->last_y = 0;
  }
  
  if (sp->pBuffer != NULL) {
      _TIFFfree(sp->pBuffer);
      sp->pBuffer = NULL;    
  }

  _TIFFfree(tif->tif_data);
  tif->tif_data = NULL;

  _TIFFSetDefaultCompressionState(tif);
}

static int
TWebPVSetField(TIFF* tif, uint32 tag, va_list ap)
{
	static const char module[] = "WebPVSetField";
  WebPState* sp = LState(tif);

  switch (tag) {
  case TIFFTAG_WEBP_LEVEL:
    sp->quality_level = (int) va_arg(ap, int);
    if( sp->quality_level <= 0 ||
        sp->quality_level > 100.0f ) {
      TIFFWarningExt(tif->tif_clientdata, module,
                     "WEBP_LEVEL should be between 1 and 100");
    }
    return 1;
  case TIFFTAG_WEBP_LOSSLESS:
    #if WEBP_ENCODER_ABI_VERSION >= 0x0100
    sp->lossless = va_arg(ap, int);
    if (sp->lossless){
      sp->quality_level = 100.0f;      
    }
    return 1;
    #else
      TIFFErrorExt(tif->tif_clientdata, module,
                  "Need to upgrade WEBP driver, this version doesn't support "
                  "lossless compression.");
      return 0;
    #endif 
  default:
    return (*sp->vsetparent)(tif, tag, ap);
  }
  /*NOTREACHED*/
}

static int
TWebPVGetField(TIFF* tif, uint32 tag, va_list ap)
{
  WebPState* sp = LState(tif);

  switch (tag) {
  case TIFFTAG_WEBP_LEVEL:
    *va_arg(ap, int*) = sp->quality_level;
    break;
  case TIFFTAG_WEBP_LOSSLESS:
    *va_arg(ap, int*) = sp->lossless;
    break;
  default:
    return (*sp->vgetparent)(tif, tag, ap);
  }
  return 1;
}

static const TIFFField TWebPFields[] = {
  { TIFFTAG_WEBP_LEVEL, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
    TIFF_SETGET_UNDEFINED,
    FIELD_PSEUDO, TRUE, FALSE, "WEBP quality", NULL },
  { TIFFTAG_WEBP_LOSSLESS, 0, 0, TIFF_ANY, 0, TIFF_SETGET_INT,
    TIFF_SETGET_UNDEFINED,
    FIELD_PSEUDO, TRUE, FALSE, "WEBP lossless/lossy", NULL
  },
};

int
TIFFInitWebP(TIFF* tif, int scheme)
{
  static const char module[] = "TIFFInitWebP";
  WebPState* sp;

  assert( scheme == COMPRESSION_WEBP );

  /*
  * Merge codec-specific tag information.
  */
  if ( !_TIFFMergeFields(tif, TWebPFields, TIFFArrayCount(TWebPFields)) ) {
    TIFFErrorExt(tif->tif_clientdata, module,
                "Merging WebP codec-specific tags failed");
    return 0;
  }

  /*
  * Allocate state block so tag methods have storage to record values.
  */
  tif->tif_data = (uint8*) _TIFFmalloc(sizeof(WebPState));
  if (tif->tif_data == NULL)
    goto bad;
  sp = LState(tif);

  /*
  * Override parent get/set field methods.
  */
  sp->vgetparent = tif->tif_tagmethods.vgetfield;
  tif->tif_tagmethods.vgetfield = TWebPVGetField;	/* hook for codec tags */
  sp->vsetparent = tif->tif_tagmethods.vsetfield;
  tif->tif_tagmethods.vsetfield = TWebPVSetField;	/* hook for codec tags */

  /* Default values for codec-specific fields */
  sp->quality_level = 75.0f;		/* default comp. level */
  sp->lossless = 0; /* default to false */
  sp->state = 0;
  sp->nSamples = 0;
  sp->psDecoder = NULL;
  sp->last_y = 0;
  
  sp->buffer_offset = 0;
  sp->pBuffer = NULL;

  /*
  * Install codec methods.
  * Notes:
  * encoderow is not supported
  */
  tif->tif_fixuptags = TWebPFixupTags;
  tif->tif_setupdecode = TWebPSetupDecode;
  tif->tif_predecode = TWebPPreDecode;
  tif->tif_decoderow = TWebPDecode;
  tif->tif_decodestrip = TWebPDecode;
  tif->tif_decodetile = TWebPDecode;
  tif->tif_setupencode = TWebPSetupEncode;
  tif->tif_preencode = TWebPPreEncode;
  tif->tif_postencode = TWebPPostEncode;
  tif->tif_encoderow = TWebPEncode;
  tif->tif_encodestrip = TWebPEncode;
  tif->tif_encodetile = TWebPEncode;
  tif->tif_cleanup = TWebPCleanup;

  return 1;
bad:
  TIFFErrorExt(tif->tif_clientdata, module,
  	     "No space for WebP state block");
  return 0;
}

#endif /* WEBP_SUPPORT */
