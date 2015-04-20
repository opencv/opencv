/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmJPEG2000Codec.h"
#include "gdcmTransferSyntax.h"
#include "gdcmTrace.h"
#include "gdcmDataElement.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSwapper.h"

#include <cstring>
#include <numeric>

#ifdef OPENJPEG_MAJOR_VERSION
#if OPENJPEG_MAJOR_VERSION == 1
#include "gdcm_openjpeg.h"
#elif OPENJPEG_MAJOR_VERSION == 2
#define USE_OPJ_DEPRECATED // opj_setup_decoder
#include "gdcm_openjpeg2.h"
#else
#error should not happen
#endif
#else
#error should not happen
#endif

namespace gdcm
{

/**
sample error callback expecting a FILE* client object
*/
void error_callback(const char *msg, void *) {
  (void)msg;
  gdcmErrorMacro( "Error in gdcmopenjpeg" << msg );
}
/**
sample warning callback expecting a FILE* client object
*/
void warning_callback(const char *msg, void *) {
  (void)msg;
  gdcmWarningMacro( "Warning in gdcmopenjpeg" << msg );
}
/**
sample debug callback expecting no client object
*/
void info_callback(const char *msg, void *) {
  (void)msg;
  gdcmDebugMacro( "Info in gdcmopenjpeg" << msg );
}

#define J2K_CFMT 0
#define JP2_CFMT 1
#define JPT_CFMT 2

#if OPENJPEG_MAJOR_VERSION == 1
#define MJ2_CFMT 3
#define PXM_DFMT 0
#define PGX_DFMT 1
#define BMP_DFMT 2
#define YUV_DFMT 3
#elif OPENJPEG_MAJOR_VERSION == 2
#define PXM_DFMT 10
#define PGX_DFMT 11
#define BMP_DFMT 12
#define YUV_DFMT 13
#define TIF_DFMT 14
#define RAW_DFMT 15
#define TGA_DFMT 16
#define PNG_DFMT 17
#endif // OPENJPEG_MAJOR_VERSION == 1

#if OPENJPEG_MAJOR_VERSION == 2
struct myfile
{
  char *mem;
  char *cur;
  size_t len;
};

void gdcm_error_callback(const char* msg, void* f)
{
  if( strcmp( msg, "Cannot read data with no size known, giving up\n" ) == 0 )
    {
    OPJ_UINT32 **s = (OPJ_UINT32**)f;
    *s[1] = *s[0];
    gdcmWarningMacro( "Recovering from odd J2K file" );
    }
//  else
//    {
//    fprintf( stderr, msg );
//    }
}


OPJ_UINT32 opj_read_from_memory(void * p_buffer, OPJ_UINT32 p_nb_bytes, myfile* p_file)
{
  //OPJ_UINT32 l_nb_read = fread(p_buffer,1,p_nb_bytes,p_file);
  OPJ_UINT32 l_nb_read;
  if( p_file->cur + p_nb_bytes < p_file->mem + p_file->len )
    {
    l_nb_read = 1*p_nb_bytes;
    }
  else
    {
    l_nb_read = (OPJ_UINT32)(p_file->mem + p_file->len - p_file->cur);
    assert( l_nb_read < p_nb_bytes );
    }
  memcpy(p_buffer,p_file->cur,l_nb_read);
  p_file->cur += l_nb_read;
  assert( p_file->cur <= p_file->mem + p_file->len );
  //std::cout << "l_nb_read: " << l_nb_read << std::endl;
  return l_nb_read ? l_nb_read : ((OPJ_UINT32)-1);
}

OPJ_UINT32 opj_write_from_memory (void * p_buffer, OPJ_UINT32 p_nb_bytes, myfile* p_file)
{
  //return fwrite(p_buffer,1,p_nb_bytes,p_file);
  OPJ_UINT32 l_nb_write;
  //if( p_file->cur + p_nb_bytes < p_file->mem + p_file->len )
  //  {
  l_nb_write = 1*p_nb_bytes;
  //  }
  //else
  //  {
  //  l_nb_write = p_file->mem + p_file->len - p_file->cur;
  //  assert( l_nb_write < p_nb_bytes );
  //  }
  memcpy(p_file->cur,p_buffer,l_nb_write);
  p_file->cur += l_nb_write;
  p_file->len += l_nb_write;
  //assert( p_file->cur < p_file->mem + p_file->len );
  return l_nb_write;
  //return p_nb_bytes;
}

OPJ_SIZE_T opj_skip_from_memory (OPJ_SIZE_T p_nb_bytes, myfile * p_file)
{
  //if (fseek(p_user_data,p_nb_bytes,SEEK_CUR))
  //  {
  //  return -1;
  //  }
  if( p_file->cur + p_nb_bytes < p_file->mem + p_file->len )
    {
    p_file->cur += p_nb_bytes;
    return p_nb_bytes;
    }

  p_file->cur = p_file->mem + p_file->len;
  return (OPJ_SIZE_T)-1;
}

bool opj_seek_from_memory (OPJ_SIZE_T p_nb_bytes, myfile * p_file)
{
  //if (fseek(p_user_data,p_nb_bytes,SEEK_SET))
  //  {
  //  return false;
  //  }
  //return true;
  if( p_file->cur + p_nb_bytes < p_file->mem + p_file->len )
    {
    p_file->cur += p_nb_bytes;
    return true;
    }
  p_file->cur = p_file->mem + p_file->len;
  return false;
}

opj_stream_t* OPJ_CALLCONV opj_stream_create_memory_stream (myfile* p_mem,OPJ_UINT32 p_size,bool p_is_read_stream)
{
  opj_stream_t* l_stream = 00;
  if
    (! p_mem)
  {
    return 00;
  }
  l_stream = opj_stream_create(p_size,p_is_read_stream);
  if
    (! l_stream)
  {
    return 00;
  }
  opj_stream_set_user_data(l_stream,p_mem);
  opj_stream_set_read_function(l_stream,(opj_stream_read_fn) opj_read_from_memory);
  opj_stream_set_write_function(l_stream, (opj_stream_write_fn) opj_write_from_memory);
  opj_stream_set_skip_function(l_stream, (opj_stream_skip_fn) opj_skip_from_memory);
  opj_stream_set_seek_function(l_stream, (opj_stream_seek_fn) opj_seek_from_memory);
  return l_stream;
}

#endif // OPENJPEG_MAJOR_VERSION == 2

/*
 * Divide an integer by a power of 2 and round upwards.
 *
 * a divided by 2^b
 */
inline int int_ceildivpow2(int a, int b) {
  return (a + (1 << b) - 1) >> b;
}

class JPEG2000Internals
{
public:
  JPEG2000Internals()
    {
    memset(&coder_param, 0, sizeof(coder_param));
    opj_set_default_encoder_parameters(&coder_param);
    }

  opj_cparameters coder_param;
};

void JPEG2000Codec::SetRate(unsigned int idx, double rate)
{
  Internals->coder_param.tcp_rates[idx] = (float)rate;
  if( Internals->coder_param.tcp_numlayers <= (int)idx )
    {
    Internals->coder_param.tcp_numlayers = idx + 1;
    }
  Internals->coder_param.cp_disto_alloc = 1;
}

double JPEG2000Codec::GetRate(unsigned int idx ) const
{
  return Internals->coder_param.tcp_rates[idx];
}

void JPEG2000Codec::SetQuality(unsigned int idx, double q)
{
  Internals->coder_param.tcp_distoratio[idx] = (float)q;
  if( Internals->coder_param.tcp_numlayers <= (int)idx )
    {
    Internals->coder_param.tcp_numlayers = idx + 1;
    }
  Internals->coder_param.cp_fixed_quality = 1;
}

double JPEG2000Codec::GetQuality(unsigned int idx) const
{
  return Internals->coder_param.tcp_distoratio[idx];
}

void JPEG2000Codec::SetTileSize(unsigned int tx, unsigned int ty)
{
  Internals->coder_param.cp_tdx = tx;
  Internals->coder_param.cp_tdy = ty;
  Internals->coder_param.tile_size_on = true;
}

void JPEG2000Codec::SetNumberOfResolutions(unsigned int nres)
{
  Internals->coder_param.numresolution = nres;
}

void JPEG2000Codec::SetReversible(bool res)
{
  LossyFlag = !res;
  Internals->coder_param.irreversible = !res;
}

JPEG2000Codec::JPEG2000Codec()
{
  Internals = new JPEG2000Internals;
}

JPEG2000Codec::~JPEG2000Codec()
{
  delete Internals;
}

bool JPEG2000Codec::CanDecode(TransferSyntax const &ts) const
{
  return ts == TransferSyntax::JPEG2000Lossless
      || ts == TransferSyntax::JPEG2000
      || ts == TransferSyntax::JPEG2000Part2Lossless
      || ts == TransferSyntax::JPEG2000Part2;
}

bool JPEG2000Codec::CanCode(TransferSyntax const &ts) const
{
  return ts == TransferSyntax::JPEG2000Lossless
      || ts == TransferSyntax::JPEG2000
      || ts == TransferSyntax::JPEG2000Part2Lossless
      || ts == TransferSyntax::JPEG2000Part2;
}

/*
A.4.4 JPEG 2000 image compression

  If the object allows multi-frame images in the pixel data field, then for these JPEG 2000 Part 1 Transfer
  Syntaxes, each frame shall be encoded separately. Each fragment shall contain encoded data from a
  single frame.
  Note: That is, the processes defined in ISO/IEC 15444-1 shall be applied on a per-frame basis. The proposal
  for encapsulation of multiple frames in a non-DICOM manner in so-called 'Motion-JPEG' or 'M-JPEG'
  defined in 15444-3 is not used.
*/
bool JPEG2000Codec::Decode(DataElement const &in, DataElement &out)
{
  if( NumberOfDimensions == 2 )
    {
    const SequenceOfFragments *sf = in.GetSequenceOfFragments();
    const ByteValue *j2kbv = in.GetByteValue();
    if( !sf && !j2kbv ) return false;
    SmartPointer<SequenceOfFragments> sf_bug = new SequenceOfFragments;
    if ( j2kbv )
      {
      gdcmWarningMacro( "Pixel Data is not encapsulated correctly. Continuing anyway" );
      assert( !sf );
      std::stringstream is;
      size_t j2kbv_len = j2kbv->GetLength();
      char *mybuffer = new char[j2kbv_len];
      bool b = j2kbv->GetBuffer(mybuffer, j2kbv_len);
      assert( b );
      if( b ) is.write(mybuffer, j2kbv_len);
      delete[] mybuffer;

      try {
        sf_bug->Read<SwapperNoOp>(is,true);
      } catch ( ... ) {
        return false;
      }
      sf = &*sf_bug;
      }
    if( !sf ) return false;
    std::stringstream is;
    unsigned long totalLen = sf->ComputeByteLength();
    char *buffer = new char[totalLen];
    sf->GetBuffer(buffer, totalLen);
    is.write(buffer, totalLen);
    delete[] buffer;
    std::stringstream os;
    bool r = DecodeByStreams(is, os);
    if(!r) return false;
    out = in;
    std::string str = os.str();
    out.SetByteValue( &str[0], (uint32_t)str.size() );
    //memcpy(buffer, os.str().c_str(), len);
    return r;
    }
  else if ( NumberOfDimensions == 3 )
    {
    /* I cannot figure out how to use openjpeg to support multiframes
     * as encoded in DICOM
     * MM: Hack. If we are lucky enough the number of encapsulated fragments actually match
     * the number of Z frames.
     * MM: hopefully this is the standard so people are following it ...
     */
    //#ifdef SUPPORT_MULTIFRAMESJ2K_ONLY
    const SequenceOfFragments *sf = in.GetSequenceOfFragments();
    if( !sf ) return false;
    std::stringstream os;
    if( sf->GetNumberOfFragments() != Dimensions[2] )
      {
      gdcmErrorMacro( "Not handled" );
      return false;
      }
    for(unsigned int i = 0; i < sf->GetNumberOfFragments(); ++i)
      {
      std::stringstream is;
      const Fragment &frag = sf->GetFragment(i);
      if( frag.IsEmpty() ) return false;
      const ByteValue *bv = frag.GetByteValue();
      assert( bv );
      size_t bv_len = bv->GetLength();
      char *mybuffer = new char[bv_len];
      bv->GetBuffer(mybuffer, bv->GetLength());
      is.write(mybuffer, bv->GetLength());
      delete[] mybuffer;
      bool r = DecodeByStreams(is, os);
      if(!r) return false;
      assert( r == true );
      }
    std::string str = os.str();
    assert( str.size() );
    out.SetByteValue( &str[0], (uint32_t)str.size() );

    return true;
    }
  // else
  return false;
}

std::pair<char *, size_t> JPEG2000Codec::DecodeByStreamsCommon(char *dummy_buffer, size_t buf_size)
{
  opj_dparameters_t parameters;  /* decompression parameters */
#if OPENJPEG_MAJOR_VERSION == 1
  opj_event_mgr_t event_mgr;    /* event manager */
  opj_dinfo_t* dinfo;  /* handle to a decompressor */
  opj_cio_t *cio;
#elif OPENJPEG_MAJOR_VERSION == 2
  opj_codec_t* dinfo = NULL;  /* handle to a decompressor */
  opj_stream_t *cio = NULL;
#endif // OPENJPEG_MAJOR_VERSION == 1
  opj_image_t *image = NULL;

  unsigned char *src = (unsigned char*)dummy_buffer;
  uint32_t file_length = (uint32_t)buf_size; // 32bits truncation should be ok since DICOM cannot have larger than 2Gb image

  // WARNING: OpenJPEG is very picky when there is a trailing 00 at the end of the JPC
  // so we need to make sure to remove it:
  // See for example: DX_J2K_0Padding.dcm
  //             and D_CLUNIE_CT1_J2KR.dcm
    //  Marker 0xffd9 EOI End of Image (JPEG 2000 EOC End of codestream)
    // gdcmData/D_CLUNIE_CT1_J2KR.dcm contains a trailing 0xFF which apparently is ok...
  while( file_length > 0 && src[file_length-1] != 0xd9 )
    {
    file_length--;
    }
  // what if 0xd9 is never found ?
  assert( file_length > 0 && src[file_length-1] == 0xd9 );

#if OPENJPEG_MAJOR_VERSION == 1
  /* configure the event callbacks (not required) */
  memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
  event_mgr.error_handler = error_callback;
  event_mgr.warning_handler = warning_callback;
  event_mgr.info_handler = info_callback;
#endif // OPENJPEG_MAJOR_VERSION == 1

  /* set decoding parameters to default values */
  opj_set_default_decoder_parameters(&parameters);

#if OPENJPEG_MAJOR_VERSION == 1
  // default blindly copied
  parameters.cp_layer=0;
  parameters.cp_reduce=0;
  //   parameters.decod_format=-1;
  //   parameters.cod_format=-1;
#endif

  const char jp2magic[] = "\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A";
  if( memcmp( src, jp2magic, sizeof(jp2magic) ) == 0 )
    {
    /* JPEG-2000 compressed image data ... sigh */
    // gdcmData/ELSCINT1_JP2vsJ2K.dcm
    // gdcmData/MAROTECH_CT_JP2Lossy.dcm
    gdcmWarningMacro( "J2K start like JPEG-2000 compressed image data instead of codestream" );
    parameters.decod_format = JP2_CFMT;
    assert(parameters.decod_format == JP2_CFMT);
    }
  else
    {
    /* JPEG-2000 codestream */
    parameters.decod_format = J2K_CFMT;
    assert(parameters.decod_format == J2K_CFMT);
    }
  parameters.cod_format = PGX_DFMT;
  assert(parameters.cod_format == PGX_DFMT);

  /* get a decoder handle */
  switch(parameters.decod_format)
    {
  case J2K_CFMT:
    dinfo = opj_create_decompress(CODEC_J2K);
    break;
  case JP2_CFMT:
    dinfo = opj_create_decompress(CODEC_JP2);
    break;
  default:
    gdcmErrorMacro( "Impossible happen" );
    return std::make_pair<char*,size_t>(0,0);
    }

  int reversible;
#if OPENJPEG_MAJOR_VERSION == 1
  /* catch events using our callbacks and give a local context */
  opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, NULL);

  /* setup the decoder decoding parameters using user parameters */
  opj_setup_decoder(dinfo, &parameters);

  /* open a byte stream */
  cio = opj_cio_open((opj_common_ptr)dinfo, src, file_length);

  /* decode the stream and fill the image structure */
  image = opj_decode(dinfo, cio);
  if(!image) {
    opj_destroy_decompress(dinfo);
    opj_cio_close(cio);
    gdcmErrorMacro( "opj_decode failed" );
    return std::make_pair<char*,size_t>(0,0);
  }
#elif OPENJPEG_MAJOR_VERSION == 2
  myfile mysrc;
  myfile *fsrc = &mysrc;
  fsrc->mem = fsrc->cur = (char*)src;
  fsrc->len = file_length;

  OPJ_UINT32 *s[2];
  // the following hack is used for the file: DX_J2K_0Padding.dcm
  // see the function j2k_read_sot in openjpeg (line: 5946)
  // to deal with zero length Psot
  OPJ_UINT32 fl = file_length - 100;
  s[0] = &fl;
  s[1] = 0;
  opj_set_error_handler(dinfo, gdcm_error_callback, s);

  cio = opj_stream_create_memory_stream(fsrc,J2K_STREAM_CHUNK_SIZE, true);

  /* setup the decoder decoding parameters using user parameters */
  opj_setup_decoder(dinfo, &parameters);
  bool bResult;
  OPJ_INT32 l_tile_x0,l_tile_y0;
  OPJ_UINT32 l_tile_width,l_tile_height,l_nb_tiles_x,l_nb_tiles_y;
  bResult = opj_read_header(
    dinfo,
    &image,
    &l_tile_x0,
    &l_tile_y0,
    &l_tile_width,
    &l_tile_height,
    &l_nb_tiles_x,
    &l_nb_tiles_y,
    cio);
  assert( bResult );

#if OPENJPEG_MAJOR_VERSION == 1
#else
  // needs to be before call to opj_decode...
  reversible = opj_get_reversible(dinfo, &parameters );
  assert( reversible == 0 || reversible == 1 );
#endif

  image = opj_decode(dinfo, cio);
  //assert( image );
  bResult = bResult && (image != 00);
  bResult = bResult && opj_end_decompress(dinfo,cio);
  if (!image || !check_comp_valid(image) )
    {
    opj_destroy_codec(dinfo);
    opj_stream_destroy(cio);
    gdcmErrorMacro( "opj_decode failed" );
    return std::make_pair<char*,size_t>(0,0);
    }
#endif // OPENJPEG_MAJOR_VERSION == 1

#if 0
  if( image->color_space )
    {
    if( image->color_space == CLRSPC_GRAY )
      {
      assert( this->GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME2
        || this->GetPhotometricInterpretation() == PhotometricInterpretation::MONOCHROME1
        || this->GetPhotometricInterpretation() == PhotometricInterpretation::PALETTE_COLOR );
      }
    else if( image->color_space == CLRSPC_SRGB )
      {
      assert( this->GetPhotometricInterpretation() == PhotometricInterpretation::RGB );
      }
    else
      {
      assert(0);
      }
    }
#endif

#if OPENJPEG_MAJOR_VERSION == 1
  opj_j2k_t* j2k = NULL;
  opj_jp2_t* jp2 = NULL;

  switch(parameters.decod_format)
    {
  case J2K_CFMT:
    j2k = (opj_j2k_t*)dinfo->j2k_handle;
    assert( j2k );
    reversible = j2k->cp->tcps->tccps->qmfbid;
    break;
  case JP2_CFMT:
    jp2 = (opj_jp2_t*)dinfo->jp2_handle;
    assert( jp2 );
    reversible = jp2->j2k->cp->tcps->tccps->qmfbid;
    break;
  default:
    gdcmErrorMacro( "Impossible happen" );
    return std::make_pair<char*,size_t>(0,0);
    }
#endif // OPENJPEG_MAJOR_VERSION == 1
  LossyFlag = !reversible;

#if 0
#ifndef GDCM_USE_SYSTEM_OPENJPEG
  if( j2k )
    j2k_dump_cp(stdout, image, j2k->cp);
  if( jp2 )
    j2k_dump_cp(stdout, image, jp2->j2k->cp);
#endif
#endif

  assert( image->numcomps == this->GetPixelFormat().GetSamplesPerPixel() );
  assert( image->numcomps == this->GetPhotometricInterpretation().GetSamplesPerPixel() );

#if OPENJPEG_MAJOR_VERSION == 1
  /* close the byte stream */
  opj_cio_close(cio);
#elif OPENJPEG_MAJOR_VERSION == 2
  /* close the byte stream */
  opj_stream_destroy(cio);
#endif // OPENJPEG_MAJOR_VERSION == 1

  // Copy buffer
  unsigned long len = Dimensions[0]*Dimensions[1] * (PF.GetBitsAllocated() / 8) * image->numcomps;
  char *raw = new char[len];
  //assert( len == fsrc->len );
  for (unsigned int compno = 0; compno < (unsigned int)image->numcomps; compno++)
    {
    opj_image_comp_t *comp = &image->comps[compno];

    int w = image->comps[compno].w;
    int wr = int_ceildivpow2(image->comps[compno].w, image->comps[compno].factor);

    //int h = image.comps[compno].h;
    int hr = int_ceildivpow2(image->comps[compno].h, image->comps[compno].factor);
    //assert(  wr * hr * 1 * image->numcomps * (comp->prec/8) == len );

    // ELSCINT1_JP2vsJ2K.dcm
    // -> prec = 12, bpp = 0, sgnd = 0
    //assert( wr == Dimensions[0] );
    //assert( hr == Dimensions[1] );
#if OPENJPEG_MAJOR_VERSION == 1
    if( comp->bpp == PF.GetBitsAllocated() )
      {
      gdcmWarningMacro( "BPP = " << comp->bpp << " vs BitsAllocated = " << PF.GetBitsAllocated() );
      }
#endif // OPENJPEG_MAJOR_VERSION == 1

    if( comp->sgnd != PF.GetPixelRepresentation() )
      {
      PF.SetPixelRepresentation( (uint16_t)comp->sgnd );
      }
#ifndef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    assert( comp->prec == PF.GetBitsStored()); // D_CLUNIE_RG3_JPLY.dcm
    assert( comp->prec - 1 == PF.GetHighBit());
#endif
    //assert( comp->prec >= PF.GetBitsStored());
    if( comp->prec != PF.GetBitsStored() )
      {
      if( comp->prec <= 8 )
        PF.SetBitsAllocated( 8 );
      else if( comp->prec <= 16 )
        PF.SetBitsAllocated( 16 );
      else if( comp->prec <= 32 )
        PF.SetBitsAllocated( 32 );
      PF.SetBitsStored( (unsigned short)comp->prec );
      PF.SetHighBit( (unsigned short)(comp->prec - 1) ); // ??
      }
    assert( PF.IsValid() );
    assert( comp->prec <= 32 );

    if (comp->prec <= 8)
      {
      uint8_t *data8 = (uint8_t*)raw + compno;
      for (int i = 0; i < wr * hr; i++)
        {
        int v = image->comps[compno].data[i / wr * w + i % wr];
        *data8 = (uint8_t)v;
        data8 += image->numcomps;
        }
      }
    else if (comp->prec <= 16)
      {
      // ELSCINT1_JP2vsJ2K.dcm is a 12bits image
      uint16_t *data16 = (uint16_t*)raw + compno;
      for (int i = 0; i < wr * hr; i++)
        {
        int v = image->comps[compno].data[i / wr * w + i % wr];
        *data16 = (uint16_t)v;
        data16 += image->numcomps;
        }
      }
    else
      {
      uint32_t *data32 = (uint32_t*)raw + compno;
      for (int i = 0; i < wr * hr; i++)
        {
        int v = image->comps[compno].data[i / wr * w + i % wr];
        *data32 = (uint32_t)v;
        data32 += image->numcomps;
        }
      }
    }

#if OPENJPEG_MAJOR_VERSION == 1
  /* free remaining structures */
  if(dinfo) {
    opj_destroy_decompress(dinfo);
  }
#elif OPENJPEG_MAJOR_VERSION == 2
  /* free remaining structures */
  if (dinfo)
    {
    opj_destroy_codec(dinfo);
    }
#endif // OPENJPEG_MAJOR_VERSION == 1

  /* free image data structure */
  opj_image_destroy(image);

  return std::make_pair(raw,len);
}

bool JPEG2000Codec::DecodeByStreams(std::istream &is, std::ostream &os)
{
  // FIXME: Do some stupid work:
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  char *dummy_buffer = new char[buf_size];
  is.seekg(0, std::ios::beg);
  is.read( dummy_buffer, buf_size);

  std::pair<char*,size_t> raw_len = this->DecodeByStreamsCommon(dummy_buffer, buf_size);
  /* free the memory containing the code-stream */
  delete[] dummy_buffer;

  if( !raw_len.first || !raw_len.second ) return false;
  os.write( raw_len.first, raw_len.second);
  delete[] raw_len.first;
  return true;
}

template<typename T>
void rawtoimage_fill(T *inputbuffer, int w, int h, int numcomps, opj_image_t *image, int pc)
{
  T *p = inputbuffer;
  if( pc )
    {
    for(int compno = 0; compno < numcomps; compno++)
      {
      for (int i = 0; i < w * h; i++)
        {
        /* compno : 0 = GREY, (0, 1, 2) = (R, G, B) */
        image->comps[compno].data[i] = *p;
        ++p;
        }
      }
    }
  else
    {
    for (int i = 0; i < w * h; i++)
      {
      for(int compno = 0; compno < numcomps; compno++)
        {
        /* compno : 0 = GREY, (0, 1, 2) = (R, G, B) */
        image->comps[compno].data[i] = *p;
        ++p;
        }
      }
    }
}

opj_image_t* rawtoimage(char *inputbuffer, opj_cparameters_t *parameters,
  int fragment_size, int image_width, int image_height, int sample_pixel,
  int bitsallocated, int bitsstored, int sign, int quality, int pc)
{
  (void)quality;
  (void)fragment_size;
  int w, h;
  int numcomps;
  OPJ_COLOR_SPACE color_space;
  opj_image_cmptparm_t cmptparm[3]; /* maximum of 3 components */
  opj_image_t * image = NULL;

  assert( sample_pixel == 1 || sample_pixel == 3 );
  if( sample_pixel == 1 )
    {
    numcomps = 1;
    color_space = CLRSPC_GRAY;
    }
  else // sample_pixel == 3
    {
    numcomps = 3;
    color_space = CLRSPC_SRGB;
    /* Does OpenJPEg support: CLRSPC_SYCC ?? */
    }
  if( bitsallocated % 8 != 0 )
    {
    gdcmDebugMacro( "BitsAllocated is not % 8" );
    return 0;
    }
  assert( bitsallocated % 8 == 0 );
  // eg. fragment_size == 63532 and 181 * 117 * 3 * 8 == 63531 ...
  assert( ((fragment_size + 1)/2 ) * 2 == ((image_height * image_width * numcomps * (bitsallocated/8) + 1)/ 2 )* 2 );
  int subsampling_dx = parameters->subsampling_dx;
  int subsampling_dy = parameters->subsampling_dy;

  // FIXME
  w = image_width;
  h = image_height;

  /* initialize image components */
  memset(&cmptparm[0], 0, 3 * sizeof(opj_image_cmptparm_t));
  //assert( bitsallocated == 8 );
  for(int i = 0; i < numcomps; i++) {
    cmptparm[i].prec = bitsstored;
    cmptparm[i].bpp = bitsallocated;
    cmptparm[i].sgnd = sign;
    cmptparm[i].dx = subsampling_dx;
    cmptparm[i].dy = subsampling_dy;
    cmptparm[i].w = w;
    cmptparm[i].h = h;
  }

  /* create the image */
  image = opj_image_create(numcomps, &cmptparm[0], color_space);
  if(!image) {
    return NULL;
  }
  /* set image offset and reference grid */
  image->x0 = parameters->image_offset_x0;
  image->y0 = parameters->image_offset_y0;
  image->x1 = parameters->image_offset_x0 + (w - 1) * subsampling_dx + 1;
  image->y1 = parameters->image_offset_y0 + (h - 1) * subsampling_dy + 1;

  /* set image data */

  //assert( fragment_size == numcomps*w*h*(bitsallocated/8) );
  if (bitsallocated <= 8)
    {
    if( sign )
      {
      rawtoimage_fill<int8_t>((int8_t*)inputbuffer,w,h,numcomps,image,pc);
      }
    else
      {
      rawtoimage_fill<uint8_t>((uint8_t*)inputbuffer,w,h,numcomps,image,pc);
      }
    }
  else if (bitsallocated <= 16)
    {
    if( sign )
      {
      rawtoimage_fill<int16_t>((int16_t*)inputbuffer,w,h,numcomps,image,pc);
      }
    else
      {
      rawtoimage_fill<uint16_t>((uint16_t*)inputbuffer,w,h,numcomps,image,pc);
      }
    }
  else if (bitsallocated <= 32)
    {
    if( sign )
      {
      rawtoimage_fill<int32_t>((int32_t*)inputbuffer,w,h,numcomps,image,pc);
      }
    else
      {
      rawtoimage_fill<uint32_t>((uint32_t*)inputbuffer,w,h,numcomps,image,pc);
      }
    }
  else
    {
    return NULL;
    }

  return image;
}

bool JPEG2000Codec::CodeFrameIntoBuffer(char * outdata, size_t outlen, size_t & complen, const char * inputdata, size_t inputlength )
{
  complen = 0; // default init
  if( NeedOverlayCleanup )
    {
    gdcmErrorMacro( "TODO" );
    return false;
    }
  const unsigned int *dims = this->GetDimensions();
  int image_width = dims[0];
  int image_height = dims[1];
  int numZ = 0; //dims[2];
  const PixelFormat &pf = this->GetPixelFormat();
  int sample_pixel = pf.GetSamplesPerPixel();
  int bitsallocated = pf.GetBitsAllocated();
#ifndef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
  int bitsstored = pf.GetBitsStored();
#else
  // Usual D_CLUNIE_RG3_JPLY.dcm kludge:
  int bitsstored = pf.GetBitsAllocated();
#endif
  int sign = pf.GetPixelRepresentation();
  int quality = 100;

  //// input_buffer is ONE image
  //// fragment_size is the size of this image (fragment)
  (void)numZ;
  bool bSuccess;
  //bool delete_comment = true;
  opj_cparameters_t parameters;  /* compression parameters */
#if OPENJPEG_MAJOR_VERSION == 1
  opj_event_mgr_t event_mgr;    /* event manager */
#endif // OPENJPEG_MAJOR_VERSION == 1
  opj_image_t *image = NULL;
  //quality = 100;

#if OPENJPEG_MAJOR_VERSION == 1
  /*
  configure the event callbacks (not required)
  setting of each callback is optionnal
   */
  memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
  event_mgr.error_handler = error_callback;
  event_mgr.warning_handler = warning_callback;
  event_mgr.info_handler = info_callback;
#endif // OPENJPEG_MAJOR_VERSION == 1

  /* set encoding parameters to default values */
  //memset(&parameters, 0, sizeof(parameters));
  //opj_set_default_encoder_parameters(&parameters);

  memcpy(&parameters, &(Internals->coder_param), sizeof(parameters));

  if ((parameters.cp_disto_alloc || parameters.cp_fixed_alloc || parameters.cp_fixed_quality)
    && (!(parameters.cp_disto_alloc ^ parameters.cp_fixed_alloc ^ parameters.cp_fixed_quality)))
    {
    gdcmErrorMacro( "Error: options -r -q and -f cannot be used together." );
    return false;
    }        /* mod fixed_quality */

  /* if no rate entered, lossless by default */
  if (parameters.tcp_numlayers == 0)
    {
    parameters.tcp_rates[0] = 0;
    parameters.tcp_numlayers = 1;
    parameters.cp_disto_alloc = 1;
    }

  if(parameters.cp_comment == NULL) {
    const char comment[] = "Created by GDCM/OpenJPEG version 2.0";
    parameters.cp_comment = (char*)malloc(strlen(comment) + 1);
    strcpy(parameters.cp_comment, comment);
    /* no need to delete parameters.cp_comment on exit */
    //delete_comment = false;
  }

  // Compute the proper number of resolutions to use.
  // This is mostly done for images smaller than 64 pixels
  // along any dimension.
  unsigned int numberOfResolutions = 0;

  unsigned int tw = image_width >> 1;
  unsigned int th = image_height >> 1;

  while( tw && th )
    {
    numberOfResolutions++;
    tw >>= 1;
    th >>= 1;
    }

  // Clamp the number of resolutions to 6.
  if( numberOfResolutions > 6 )
    {
    numberOfResolutions = 6;
    }

  parameters.numresolution = numberOfResolutions;


  /* decode the source image */
  /* ----------------------- */

  image = rawtoimage((char*)inputdata, &parameters,
    static_cast<int>( inputlength ),
    image_width, image_height,
    sample_pixel, bitsallocated, bitsstored, sign, quality, this->GetPlanarConfiguration() );
  if (!image) {
    return false;
  }

  /* encode the destination image */
  /* ---------------------------- */
  parameters.cod_format = J2K_CFMT; /* J2K format output */
  size_t codestream_length;
#if OPENJPEG_MAJOR_VERSION == 1
  opj_cio_t *cio = NULL;

  /* get a J2K compressor handle */
  opj_cinfo_t* cinfo = opj_create_compress(CODEC_J2K);

  /* catch events using our callbacks and give a local context */
  opj_set_event_mgr((opj_common_ptr)cinfo, &event_mgr, stderr);

  /* setup the encoder parameters using the current image and using user parameters */
  opj_setup_encoder(cinfo, &parameters, image);

  /* open a byte stream for writing */
  /* allocate memory for all tiles */
  cio = opj_cio_open((opj_common_ptr)cinfo, NULL, 0);

  /* encode the image */
  bSuccess = opj_encode(cinfo, cio, image, parameters.index);
  if (!bSuccess) {
    opj_cio_close(cio);
    fprintf(stderr, "failed to encode image\n");
    return false;
  }
  codestream_length = cio_tell(cio);
#elif OPENJPEG_MAJOR_VERSION == 2
  opj_codec_t* cinfo = 00;
  opj_stream_t *cio = 00;

  /* get a J2K compressor handle */
  cinfo = opj_create_compress(CODEC_J2K);

  /* setup the encoder parameters using the current image and using user parameters */
  opj_setup_encoder(cinfo, &parameters, image);

  myfile mysrc;
  myfile *fsrc = &mysrc;
  char *buffer_j2k = new char[image_len]; // overallocated
  fsrc->mem = fsrc->cur = buffer_j2k;
  fsrc->len = 0;

  /* open a byte stream for writing */
  /* allocate memory for all tiles */
  cio = opj_stream_create_memory_stream(fsrc,J2K_STREAM_CHUNK_SIZE,false);
  if (! cio)
    {
    return false;
    }
  /* encode the image */
  /*if (*indexfilename)          // If need to extract codestream information
    bSuccess = opj_encode_with_info(cinfo, cio, image, &cstr_info);
    else*/
  bSuccess = opj_start_compress(cinfo,image,cio);
  bSuccess = bSuccess && opj_encode(cinfo, cio);
  bSuccess = bSuccess && opj_end_compress(cinfo, cio);

  if (!bSuccess)
    {
    opj_stream_destroy(cio);
    return false;
    }
  codestream_length = mysrc.len;
#endif // OPENJPEG_MAJOR_VERSION == 1

  /* write the buffer to disk */
  //f = fopen(parameters.outfile, "wb");
  //if (!f) {
  //  fprintf(stderr, "failed to open %s for writing\n", parameters.outfile);
  //  return 1;
  //}
  //fwrite(cio->buffer, 1, codestream_length, f);
  //#define MDEBUG
#ifdef MDEBUG
  static int c = 0;
  std::ostringstream os;
  os << "/tmp/debug";
  os << c;
  c++;
  os << ".j2k";
  std::ofstream debug(os.str().c_str(), std::ios::binary);
  debug.write((char*)(cio->buffer), codestream_length);
  debug.close();
#endif

  bool success = false;
#if OPENJPEG_MAJOR_VERSION == 1
  if( codestream_length <= outlen )
    {
    success = true;
    memcpy(outdata, (char*)(cio->buffer), codestream_length);
    }

  /* close and free the byte stream */
  opj_cio_close(cio);

  /* free remaining compression structures */
  opj_destroy_compress(cinfo);
#elif OPENJPEG_MAJOR_VERSION == 2
  if( codestream_length <= outlen )
    {
    success = true;
    memcpy(outdata, (char*)(mysrc.mem), codestream_length);
    }
  delete [] buffer_j2k;

  /* close and free the byte stream */
  opj_stream_destroy(cio);

  /* free remaining compression structures */
  opj_destroy_codec(cinfo);
#endif // OPENJPEG_MAJOR_VERSION == 1
  complen = codestream_length;

  /* free user parameters structure */
  if(parameters.cp_comment) free(parameters.cp_comment);
  if(parameters.cp_matrice) free(parameters.cp_matrice);

  /* free image data */
  opj_image_destroy(image);

  return success;
}

// Compress into JPEG
bool JPEG2000Codec::Code(DataElement const &in, DataElement &out)
{
  out = in;
  //
  // Create a Sequence Of Fragments:
  SmartPointer<SequenceOfFragments> sq = new SequenceOfFragments;

  const unsigned int *dims = this->GetDimensions();
  int image_width = dims[0];
  int image_height = dims[1];

  const ByteValue *bv = in.GetByteValue();
  const char *input = bv->GetPointer();
  unsigned long len = bv->GetLength();
  unsigned long image_len = len / dims[2];
  size_t inputlength = image_len;

  for(unsigned int dim = 0; dim < dims[2]; ++dim)
    {
    const char *inputdata = input + dim * image_len;

    std::vector<char> rgbyteCompressed;
    rgbyteCompressed.resize(image_width * image_height * 4);

    size_t cbyteCompressed;
    const bool b = this->CodeFrameIntoBuffer((char*)&rgbyteCompressed[0], rgbyteCompressed.size(), cbyteCompressed, inputdata, inputlength );
    if( !b ) return false;

    Fragment frag;
    assert( cbyteCompressed <= rgbyteCompressed.size() ); // default alloc would be bogus
    frag.SetByteValue( &rgbyteCompressed[0], (uint32_t)cbyteCompressed );
    sq->AddFragment( frag );
    }

  assert( sq->GetNumberOfFragments() == dims[2] );
  out.SetValue( *sq );

  return true;
}

bool JPEG2000Codec::GetHeaderInfo(std::istream &is, TransferSyntax &ts)
{
  // FIXME: Do some stupid work:
  is.seekg( 0, std::ios::end);
  size_t buf_size = (size_t)is.tellg();
  char *dummy_buffer = new char[buf_size];
  is.seekg(0, std::ios::beg);
  is.read( dummy_buffer, buf_size);
  bool b = GetHeaderInfo( dummy_buffer, (size_t)buf_size, ts );
  delete[] dummy_buffer;
  return b;
}

static inline bool check_comp_valid( opj_image_t *image )
{
  int compno = 0;
  opj_image_comp_t *comp = &image->comps[compno];
  if( comp->prec > 32 ) // I doubt openjpeg will reach here.
    return false;

  bool invalid = false;
  if( image->numcomps == 3 )
    {
    opj_image_comp_t *comp1 = &image->comps[1];
    opj_image_comp_t *comp2 = &image->comps[2];
#if OPENJPEG_MAJOR_VERSION == 1
    if( comp->bpp  != comp1->bpp  ) invalid = true;
    if( comp->bpp  != comp2->bpp  ) invalid = true;
#endif // OPENJPEG_MAJOR_VERSION == 1
    if( comp->prec != comp1->prec ) invalid = true;
    if( comp->prec != comp2->prec ) invalid = true;
    if( comp->sgnd != comp1->sgnd ) invalid = true;
    if( comp->sgnd != comp2->sgnd ) invalid = true;
    if( comp->h != comp1->h ) invalid = true;
    if( comp->h != comp2->h ) invalid = true;
    if( comp->w != comp1->w ) invalid = true;
    if( comp->w != comp2->w ) invalid = true;
    }
  return !invalid;
}

bool JPEG2000Codec::GetHeaderInfo(const char * dummy_buffer, size_t buf_size, TransferSyntax &ts)
{
  opj_dparameters_t parameters;  /* decompression parameters */
#if OPENJPEG_MAJOR_VERSION == 1
  opj_event_mgr_t event_mgr;    /* event manager */
  opj_dinfo_t* dinfo;  /* handle to a decompressor */
  opj_cio_t *cio;
#elif OPENJPEG_MAJOR_VERSION == 2
  opj_codec_t* dinfo = NULL;  /* handle to a decompressor */
  opj_stream_t *cio = NULL;
#endif // OPENJPEG_MAJOR_VERSION == 1
  opj_image_t *image = NULL;
  unsigned char *src = (unsigned char*)dummy_buffer;
  size_t file_length = buf_size;

#if OPENJPEG_MAJOR_VERSION == 1
  /* configure the event callbacks (not required) */
  memset(&event_mgr, 0, sizeof(opj_event_mgr_t));
  event_mgr.error_handler = error_callback;
  event_mgr.warning_handler = warning_callback;
  event_mgr.info_handler = info_callback;
#endif // OPENJPEG_MAJOR_VERSION == 1

  /* set decoding parameters to default values */
  opj_set_default_decoder_parameters(&parameters);

#if OPENJPEG_MAJOR_VERSION == 1
  // default blindly copied
  parameters.cp_layer=0;
  parameters.cp_reduce=0;
  //   parameters.decod_format=-1;
  //   parameters.cod_format=-1;
#endif

  const char jp2magic[] = "\x00\x00\x00\x0C\x6A\x50\x20\x20\x0D\x0A\x87\x0A";
  if( memcmp( src, jp2magic, sizeof(jp2magic) ) == 0 )
    {
    /* JPEG-2000 compressed image data */
    // gdcmData/ELSCINT1_JP2vsJ2K.dcm
    gdcmWarningMacro( "J2K start like JPEG-2000 compressed image data instead of codestream" );
    parameters.decod_format = JP2_CFMT;
    assert(parameters.decod_format == JP2_CFMT);
    }
  else
    {
    /* JPEG-2000 codestream */
    parameters.decod_format = J2K_CFMT;
    assert(parameters.decod_format == J2K_CFMT);
    }
  parameters.cod_format = PGX_DFMT;
  assert(parameters.cod_format == PGX_DFMT);

  /* get a decoder handle */
  switch(parameters.decod_format )
    {
  case J2K_CFMT:
    dinfo = opj_create_decompress(CODEC_J2K);
    break;
  case JP2_CFMT:
    dinfo = opj_create_decompress(CODEC_JP2);
    break;
  default:
    gdcmErrorMacro( "Impossible happen" );
    return false;
    }

#if OPENJPEG_MAJOR_VERSION == 1
  /* catch events using our callbacks and give a local context */
  opj_set_event_mgr((opj_common_ptr)dinfo, &event_mgr, NULL);

  /* setup the decoder decoding parameters using user parameters */
  opj_setup_decoder(dinfo, &parameters);

  /* open a byte stream */
  cio = opj_cio_open((opj_common_ptr)dinfo, src, (int)file_length);

  /* decode the stream and fill the image structure */
  image = opj_decode(dinfo, cio);
  if(!image) {
    opj_destroy_decompress(dinfo);
    opj_cio_close(cio);
    gdcmErrorMacro( "opj_decode failed" );
    return false;
  }
#elif OPENJPEG_MAJOR_VERSION == 2
  myfile mysrc;
  myfile *fsrc = &mysrc;
  fsrc->mem = fsrc->cur = (char*)src;
  fsrc->len = file_length;

  // the hack is not used when reading meta-info of a j2k stream:
  opj_set_error_handler(dinfo, gdcm_error_callback, NULL);

  cio = opj_stream_create_memory_stream(fsrc,J2K_STREAM_CHUNK_SIZE, true);

  /* setup the decoder decoding parameters using user parameters */
  opj_setup_decoder(dinfo, &parameters);
  bool bResult;
  OPJ_INT32 l_tile_x0,l_tile_y0;
  OPJ_UINT32 l_tile_width,l_tile_height,l_nb_tiles_x,l_nb_tiles_y;
  bResult = opj_read_header(
    dinfo,
    &image,
    &l_tile_x0,
    &l_tile_y0,
    &l_tile_width,
    &l_tile_height,
    &l_nb_tiles_x,
    &l_nb_tiles_y,
    cio);
  //image = opj_decode(dinfo, cio);
  //bResult = bResult && (image != 00);
  //bResult = bResult && opj_end_decompress(dinfo,cio);
  //if (!image)
  //  {
  //  opj_destroy_codec(dinfo);
  //  opj_stream_destroy(cio);
  //  gdcmErrorMacro( "opj_decode failed" );
  //  return false;
  //  }
#endif // OPENJPEG_MAJOR_VERSION == 1

  int reversible;
#if OPENJPEG_MAJOR_VERSION == 1
  opj_j2k_t* j2k = NULL;
  opj_jp2_t* jp2 = NULL;

  switch(parameters.decod_format)
    {
  case J2K_CFMT:
    j2k = (opj_j2k_t*)dinfo->j2k_handle;
    assert( j2k );
    reversible = j2k->cp->tcps->tccps->qmfbid;
    break;
  case JP2_CFMT:
    jp2 = (opj_jp2_t*)dinfo->jp2_handle;
    assert( jp2 );
    reversible = jp2->j2k->cp->tcps->tccps->qmfbid;
    break;
  default:
    gdcmErrorMacro( "Impossible happen" );
    return false;
    }
#else
  reversible = opj_get_reversible(dinfo, &parameters );
  assert( reversible == 0 || reversible == 1 );
#endif // OPENJPEG_MAJOR_VERSION == 1
  LossyFlag = !reversible;

#if 0
#ifndef GDCM_USE_SYSTEM_OPENJPEG
  if( j2k )
    j2k_dump_cp(stdout, image, j2k->cp);
  if( jp2 )
    j2k_dump_cp(stdout, image, jp2->j2k->cp);
#endif
#endif

  int compno = 0;
  opj_image_comp_t *comp = &image->comps[compno];

  if( !check_comp_valid( image ) )
    {
    gdcmErrorMacro( "Invalid test failed" );
    return false;
    }

  this->Dimensions[0] = comp->w;
  this->Dimensions[1] = comp->h;

  if( comp->prec <= 8 )
    {
#if OPENJPEG_MAJOR_VERSION == 1
    if( comp->bpp ) assert( comp->bpp == 8 );
#endif // OPENJPEG_MAJOR_VERSION == 1
    this->PF = PixelFormat( PixelFormat::UINT8 );
    }
  else if( comp->prec <= 16 )
    {
#if OPENJPEG_MAJOR_VERSION == 1
    if( comp->bpp ) assert( comp->bpp == 16 );
#endif // OPENJPEG_MAJOR_VERSION == 1
    this->PF = PixelFormat( PixelFormat::UINT16 );
    }
  else if( comp->prec <= 32 )
    {
#if OPENJPEG_MAJOR_VERSION == 1
    if( comp->bpp ) assert( comp->bpp == 32 );
#endif // OPENJPEG_MAJOR_VERSION == 1
    this->PF = PixelFormat( PixelFormat::UINT32 );
    }
  else
    {
    gdcmErrorMacro( "do not handle precision: " << comp->prec );
    return false;
    }
  this->PF.SetBitsStored( (unsigned short)comp->prec );
  this->PF.SetHighBit( (unsigned short)(comp->prec - 1) );
  this->PF.SetPixelRepresentation( (unsigned short)comp->sgnd );

  if( image->numcomps == 1 )
    {
    // normally we have codec only, but in some case we have a JP2 with
    // color space info:
    // - gdcmData/MAROTECH_CT_JP2Lossy.dcm
    // - gdcmData/D_CLUNIE_CT1_J2KI.dcm -> color_space = 32767
    //assert( image->color_space == 0 || image->color_space == CLRSPC_GRAY );
    PI = PhotometricInterpretation::MONOCHROME2;
    this->PF.SetSamplesPerPixel( 1 );
    }
  else if( image->numcomps == 3 )
    {
    //assert( image->color_space == 0 );
    //PI = PhotometricInterpretation::RGB;
    /*
    8.2.4 JPEG 2000 IMAGE COMPRESSION
    The JPEG 2000 bit stream specifies whether or not a reversible or irreversible
    multi-component (color) transformation, if any, has been applied. If no
    multi-component transformation has been applied, then the components shall
    correspond to those specified by the DICOM Attribute Photometric Interpretation
    (0028,0004). If the JPEG 2000 Part 1 reversible multi-component transformation
    has been applied then the DICOM Attribute Photometric Interpretation
    (0028,0004) shall be YBR_RCT. If the JPEG 2000 Part 1 irreversible
    multi-component transformation has been applied then the DICOM Attribute
    Photometric Interpretation (0028,0004) shall be YBR_ICT.  Notes: 1. For
    example, single component may be present, and the Photometric Interpretation
    (0028,0004) may be MONOCHROME2.  2. Though it would be unusual, would not take
    advantage of correlation between the red, green and blue components, and would
    not achieve effective compression, a Photometric Interpretation of RGB could be
    specified as long as no multi-component transformation was specified by the
    JPEG 2000 bit stream.  3. Despite the application of a multi-component color
    transformation and its reflection in the Photometric Interpretation attribute,
    the color space remains undefined.  There is currently no means of conveying
    standard color spaces either by fixed values (such as sRGB) or by ICC
    profiles. Note in particular that the JP2 file header is not sent in the JPEG
    2000 bitstream that is encapsulated in DICOM.
     */
    PI = PhotometricInterpretation::YBR_RCT;
    this->PF.SetSamplesPerPixel( 3 );
    }
  else if( image->numcomps == 4 )
    {
    /* Yes this is legal */
    // http://www.crc.ricoh.com/~gormish/jpeg2000conformance/
    // jpeg2000testimages/Part4TestStreams/codestreams_profile0/p0_06.j2k
    gdcmErrorMacro( "Image is 4 components which is not supported anymore in DICOM (ARGB is retired)" );
    // TODO: How about I get the 3 comps and set the alpha plane in the overlay ?
    return false;
    }
  else
    {
    // jpeg2000testimages/Part4TestStreams/codestreams_profile0/p0_13.j2k
    gdcmErrorMacro( "Image is " << image->numcomps << " components which is not supported in DICOM" );
    return false;
    }

  assert( PI != PhotometricInterpretation::UNKNOW );

  bool mct = false;
  if( mct )
    {
    if( reversible )
      {
      ts = TransferSyntax::JPEG2000Part2Lossless;
      }
    else
      {
      ts = TransferSyntax::JPEG2000Part2;
      if( PI == PhotometricInterpretation::YBR_RCT )
        {
        // FIXME ???
        PI = PhotometricInterpretation::YBR_ICT;
        }
      }
    }
  else
    {
    if( reversible )
      {
      ts = TransferSyntax::JPEG2000Lossless;
      }
    else
      {
      ts = TransferSyntax::JPEG2000;
      if( PI == PhotometricInterpretation::YBR_RCT )
        {
        // FIXME ???
        PI = PhotometricInterpretation::YBR_ICT;
        }
      }
    }

  //assert( ts.IsLossy() == this->GetPhotometricInterpretation().IsLossy() );
  //assert( ts.IsLossless() == this->GetPhotometricInterpretation().IsLossless() );
  if( this->GetPhotometricInterpretation().IsLossy() )
    {
    assert( ts.IsLossy() );
    }
  if( ts.IsLossless() && !ts.IsLossy() )
    {
    assert( this->GetPhotometricInterpretation().IsLossless() );
    }

#if OPENJPEG_MAJOR_VERSION == 1
  /* close the byte stream */
  opj_cio_close(cio);

  /* free the memory containing the code-stream */
  //delete[] src;  //FIXME

  /* free remaining structures */
  if(dinfo) {
    opj_destroy_decompress(dinfo);
  }
#elif OPENJPEG_MAJOR_VERSION == 2
  /* close the byte stream */
  opj_stream_destroy(cio);
  /* free remaining structures */
  if (dinfo)
    {
    opj_destroy_codec(dinfo);
    }
#endif // OPENJPEG_MAJOR_VERSION == 1

  /* free image data structure */
  opj_image_destroy(image);


  return true;
}

bool JPEG2000Codec::DecodeExtent(
  char *buffer,
  unsigned int xmin, unsigned int xmax,
  unsigned int ymin, unsigned int ymax,
  unsigned int zmin, unsigned int zmax,
  std::istream & is
)
{
  BasicOffsetTable bot;
  bot.Read<SwapperNoOp>( is );

  const unsigned int * dimensions = this->GetDimensions();
  // retrieve pixel format *after* DecodeByStreamsCommon !
  const PixelFormat pf = this->GetPixelFormat(); // make a copy !
  assert( pf.GetBitsAllocated() % 8 == 0 );
  assert( pf != PixelFormat::SINGLEBIT );
  assert( pf != PixelFormat::UINT12 && pf != PixelFormat::INT12 );

  if( NumberOfDimensions == 2 )
    {
    char *dummy_buffer = NULL;
    std::vector<char> vdummybuffer;
    size_t buf_size = 0;

    const Tag seqDelItem(0xfffe,0xe0dd);
    Fragment frag;
    while( frag.ReadPreValue<SwapperNoOp>(is) && frag.GetTag() != seqDelItem )
      {
      size_t fraglen = frag.GetVL();
      size_t oldlen = vdummybuffer.size();
      // update
      buf_size = fraglen + oldlen;
      vdummybuffer.resize( buf_size );
      dummy_buffer = &vdummybuffer[0];
      // read J2K
      is.read( &vdummybuffer[oldlen], fraglen );
      }
    assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
    assert( zmin == zmax );
    assert( zmin == 0 );

    std::pair<char*,size_t> raw_len = this->DecodeByStreamsCommon(dummy_buffer, buf_size);
    if( !raw_len.first || !raw_len.second ) return false;
    // check pixel format *after* DecodeByStreamsCommon !
    const PixelFormat & pf2 = this->GetPixelFormat();
    // SC16BitsAllocated_8BitsStoredJ2K.dcm
    if( pf.GetSamplesPerPixel() != pf2.GetSamplesPerPixel()
     || pf.GetBitsAllocated() != pf2.GetBitsAllocated()
/*
     || pf.GetPixelRepresentation() != pf2.GetPixelRepresentation() // TODO, we are a bit too agressive here
*/
    )
      {
      gdcmErrorMacro( "Invalid PixelFormat found (mismatch DICOM vs J2K)" );
      return false;
      }

    char *raw = raw_len.first;
    const unsigned int rowsize = xmax - xmin + 1;
    const unsigned int colsize = ymax - ymin + 1;
    const unsigned int bytesPerPixel = pf.GetPixelSize();

    const char *tmpBuffer1 = raw;
    unsigned int z = 0;
    for (unsigned int y = ymin; y <= ymax; ++y)
      {
      size_t theOffset = 0 + (z*dimensions[1]*dimensions[0] + y*dimensions[0] + xmin)*bytesPerPixel;
      tmpBuffer1 = raw + theOffset;
      memcpy(&(buffer[((z-zmin)*rowsize*colsize +
            (y-ymin)*rowsize)*bytesPerPixel]),
        tmpBuffer1, rowsize*bytesPerPixel);
      }
    delete[] raw_len.first;

    }
  else if ( NumberOfDimensions == 3 )
    {
    const Tag seqDelItem(0xfffe,0xe0dd);
    Fragment frag;
    std::streamoff thestart = is.tellg();
    unsigned int numfrags = 0;
    std::vector< size_t > offsets;
    while( frag.ReadPreValue<SwapperNoOp>(is) && frag.GetTag() != seqDelItem )
      {
      //std::streamoff relstart = is.tellg();
      //assert( relstart - thestart == 8 );
      std::streamoff off = frag.GetVL();
      offsets.push_back( off );
      is.seekg( off, std::ios::cur );
      ++numfrags;
      }
    assert( frag.GetTag() == seqDelItem && frag.GetVL() == 0 );
    assert( numfrags == offsets.size() );
    if( numfrags != Dimensions[2] )
      {
      gdcmErrorMacro( "Not handled" );
      return false;
      }

    for( unsigned int z = zmin; z <= zmax; ++z )
      {
      size_t curoffset = std::accumulate( offsets.begin(), offsets.begin() + z, 0 );
      is.seekg( thestart + curoffset + 8 * z, std::ios::beg );
      is.seekg( 8, std::ios::cur );

      const size_t buf_size = offsets[z];
      char *dummy_buffer = new char[ buf_size ];
      is.read( dummy_buffer, buf_size );
      std::pair<char*,size_t> raw_len = this->DecodeByStreamsCommon(dummy_buffer, buf_size);
      /* free the memory containing the code-stream */
      delete[] dummy_buffer;
      if( !raw_len.first || !raw_len.second ) return false;
      // check pixel format *after* DecodeByStreamsCommon !
      const PixelFormat & pf2 = this->GetPixelFormat();
      if( pf != pf2 ) return false;

      char *raw = raw_len.first;
      const unsigned int rowsize = xmax - xmin + 1;
      const unsigned int colsize = ymax - ymin + 1;
      const unsigned int bytesPerPixel = pf.GetPixelSize();

      const char *tmpBuffer1 = raw;
      for (unsigned int y = ymin; y <= ymax; ++y)
        {
        size_t theOffset = 0 + (0*dimensions[1]*dimensions[0] + y*dimensions[0] + xmin)*bytesPerPixel;
        tmpBuffer1 = raw + theOffset;
        memcpy(&(buffer[((z-zmin)*rowsize*colsize +
              (y-ymin)*rowsize)*bytesPerPixel]),
          tmpBuffer1, rowsize*bytesPerPixel);
        }
      delete[] raw_len.first;
      }
    }
  return true;
}

ImageCodec * JPEG2000Codec::Clone() const
{
  JPEG2000Codec * copy = new JPEG2000Codec;
  return copy;
}

bool JPEG2000Codec::StartEncode( std::ostream & )
{
  return true;
}
bool JPEG2000Codec::IsRowEncoder()
{
  return false;
}

bool JPEG2000Codec::IsFrameEncoder()
{
  return true;
}

bool JPEG2000Codec::AppendRowEncode( std::ostream & , const char * , size_t )
{
  return false;
}

bool JPEG2000Codec::AppendFrameEncode( std::ostream & out, const char * data, size_t datalen )
{
  const unsigned int * dimensions = this->GetDimensions();
  const PixelFormat & pf = this->GetPixelFormat();
  assert( datalen == dimensions[0] * dimensions[1] * pf.GetPixelSize() );

  std::vector<char> rgbyteCompressed;
  rgbyteCompressed.resize(dimensions[0] * dimensions[1] * 4);

  size_t cbyteCompressed;
  const bool b = this->CodeFrameIntoBuffer((char*)&rgbyteCompressed[0], rgbyteCompressed.size(), cbyteCompressed, data, datalen );
  if( !b ) return false;

  out.write( (char*)&rgbyteCompressed[0], cbyteCompressed );

  return true;
}

bool JPEG2000Codec::StopEncode( std::ostream & )
{
  return true;
}

} // end namespace gdcm
