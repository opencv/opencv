/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileChangeTransferSyntax.h"

#include "gdcmImageCodec.h"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmImageRegionReader.h"
#include "gdcmJPEGCodec.h"
#include "gdcmJPEGLSCodec.h"
#include "gdcmJPEG2000Codec.h"
#include "gdcmRLECodec.h"
#include "gdcmImageHelper.h"
#include "gdcmEvent.h"
#include "gdcmProgressEvent.h"
#include "gdcmUIDGenerator.h"
#include "gdcmAttribute.h"
#include "gdcmFileDerivation.h"
#include "gdcmFileAnonymizer.h"

namespace gdcm
{

class FileChangeTransferSyntaxInternals
{
public:
  FileChangeTransferSyntaxInternals():
    IC(NULL),
    InitializeCopy(false)
  {}
  ~FileChangeTransferSyntaxInternals()
    {
    delete IC;
    }
  ImageCodec *IC;
  bool InitializeCopy;
  std::streampos PixelDataPos;
  std::string InFilename;
  std::string OutFilename;
  TransferSyntax TS;
  std::vector<unsigned int> Dims;
  PixelFormat PF;
  PhotometricInterpretation PI;
  unsigned int PC;
  bool Needbyteswap;
  double Progress;
};

FileChangeTransferSyntax::FileChangeTransferSyntax()
{
  Internals = new FileChangeTransferSyntaxInternals;
}

FileChangeTransferSyntax::~FileChangeTransferSyntax()
{
  delete Internals;
}

bool FileChangeTransferSyntax::Change()
{
  this->InvokeEvent( StartEvent() );
  if( !InitializeCopy() )
    {
    gdcmDebugMacro( "Could not InitializeCopy" );
    return false;
    }

  const char *filename = this->Internals->InFilename.c_str();
  std::ifstream is( filename, std::ios::binary );
  is.seekg( Internals->PixelDataPos, std::ios::beg );
  const char *outfilename = this->Internals->OutFilename.c_str();
  std::fstream os( outfilename, std::ofstream::in | std::ofstream::out | std::ios::binary );
  os.seekp( 0, std::ios::end );

  const std::vector<unsigned int> & dims = Internals->Dims;
  const PixelFormat &pf = Internals->PF;
  const PhotometricInterpretation &pi = Internals->PI;
  unsigned int pc = Internals->PC;
  const int pixsize = pf.GetPixelSize();

  const bool needbyteswap = Internals->Needbyteswap;
  ImageCodec *codec = Internals->IC;
  codec->SetDimensions( dims );
  codec->SetNumberOfDimensions( 2 );
  codec->SetPlanarConfiguration( pc );
  codec->SetPhotometricInterpretation( pi );
  codec->SetNeedByteSwap( needbyteswap );
  codec->SetPixelFormat( pf ); // need to be last !

  VL vl;
  vl.SetToUndefined();
  DataElement de( Tag(0x7fe0,0x0010), vl, VR::OB );
  switch ( pf.GetBitsAllocated() )
    {
  case 8:
    de.SetVR( VR::OB );
    break;
    //case 12:
  case 16:
  case 32:
    de.SetVR( VR::OW );
    break;
  default:
    return false;
    }
  de.GetTag().Write<SwapperNoOp>( os );
  de.GetVR().Write( os );
  de.GetVL().Write<SwapperNoOp>( os );

  Fragment frag;
  // Basic Offset Table
  frag.Write<SwapperNoOp>( os );

  Internals->Progress = 0;
  bool b = Internals->IC->StartEncode(os);
  assert( b );
  size_t len = 0; // actual size compressed:
  if( Internals->IC->IsRowEncoder() )
    {
    std::vector<char> vbuffer;
    vbuffer.resize( dims[0] * pixsize );

    char *data = &vbuffer[0];
    const size_t datalen = vbuffer.size();

    const size_t nscanlines = dims[2] * dims[1];
    const double progresstick = 1. / (double)nscanlines;

    for( unsigned int z = 0; z < dims[2]; ++z )
      {
      // frag header:
      frag.Write<SwapperNoOp>( os );
      std::streampos start = os.tellp();
      for( unsigned int y = 0; y < dims[1]; ++y )
        {
        is.read( data, datalen );
        assert( is.good() );
        b = Internals->IC->AppendRowEncode(os, data, datalen);
        if( !b ) return false;
        Internals->Progress += progresstick;
        ProgressEvent pe;
        pe.SetProgress( Internals->Progress );
        this->InvokeEvent( pe );
        }
      const std::streampos end = os.tellp();

      // Compute JPEG length:
      const VL jpegvl = (uint32_t)(end - start);
      len += jpegvl;
      start -= 4;
      if( jpegvl.IsOdd() )
        {
        // 0 - padding:
        os.put( 0 );
        }
      os.seekp( start, std::ios::beg );
      jpegvl.Write<SwapperNoOp>( os );
      os.seekp( 0, std::ios::end );
      }
    }
  else if( Internals->IC->IsFrameEncoder() )
    {
    std::vector<char> vbuffer;
    vbuffer.resize( dims[0] * dims[1] * pixsize );

    char *data = &vbuffer[0];
    const size_t datalen = vbuffer.size();

    const size_t nscanlines = dims[2];
    const double progresstick = 1. / (double)nscanlines;

    for( unsigned int z = 0; z < dims[2]; ++z )
      {
      // frag header:
      frag.Write<SwapperNoOp>( os );
      std::streampos start = os.tellp();
        {
        is.read( data, datalen );
        assert( is.good() );
        b = Internals->IC->AppendFrameEncode(os, data, datalen);
        if( !b ) return false;
        Internals->Progress += progresstick;
        ProgressEvent pe;
        pe.SetProgress( Internals->Progress );
        this->InvokeEvent( pe );
        }
      const std::streampos end = os.tellp();

      // Compute JPEG length:
      const VL jpegvl = (uint32_t)(end - start);
      len += jpegvl;
      start -= 4;
      if( jpegvl.IsOdd() )
        {
        // 0 - padding:
        os.put( 0 );
        }
      os.seekp( start, std::ios::beg );
      jpegvl.Write<SwapperNoOp>( os );
      os.seekp( 0, std::ios::end );
      }
    }
  else
    {
    // dead code ?
    return false;
    }
  b = Internals->IC->StopEncode(os);
  assert( b );

  const Tag seqDelItem(0xfffe,0xe0dd);
  seqDelItem.Write<SwapperNoOp>(os);
  VL zero = 0;
  zero.Write<SwapperNoOp>(os);

  is.close();
  os.close();

  size_t reflen = dims[0] * dims[1] * pixsize;
  double level = (double)reflen / (double)len;

  if( !UpdateCompressionLevel(level) )
    {
    gdcmDebugMacro( "Could not UpdateCompressionLevel" );
    return false;
    }

  this->InvokeEvent( EndEvent() );
  return true;
}

bool FileChangeTransferSyntax::UpdateCompressionLevel(double level)
{
  // no need to update if not lossy:
  if( !Internals->IC->GetLossyFlag() ) return true;

  double ratio = level;
  Attribute<0x0028,0x2112> at2;
  at2.SetValues( &ratio, 1);
  DataElement de = at2.GetAsDataElement();

  // file has been flushed to disk at this point, need to update the compression level now:
  const char * fn = Internals->OutFilename.c_str();

  FileAnonymizer fa;
  fa.SetInputFileName(fn);
  fa.SetOutputFileName(fn);
  const ByteValue * bv = de.GetByteValue();
  fa.Replace( de.GetTag(), bv->GetPointer(), 16/* bv->GetLength()*/ );
  if( !fa.Write() )
    {
    gdcmDebugMacro( "FileAnonymizer failure" );
    return false;
    }

  return true;
}

void FileChangeTransferSyntax::SetTransferSyntax( TransferSyntax const & ts )
{
  Internals->TS = ts;
  delete Internals->IC;

  JPEGCodec jpeg;
  JPEGLSCodec jpegls;
  JPEG2000Codec jpeg2000;
  RLECodec rle;

  ImageCodec *codecs[] = {
    &jpeg,
    &jpegls,
    &jpeg2000,
    &rle
  };
  const int n = sizeof( codecs ) / sizeof( *codecs );
  for( int i = 0; i < n; ++i )
    {
    if( codecs[i]->CanCode( ts ) )
      {
      Internals->IC = codecs[i]->Clone();
      }
    }
  assert( Internals->TS );
}

ImageCodec * FileChangeTransferSyntax::GetCodec()
{
  return Internals->IC;
}

void FileChangeTransferSyntax::SetInputFileName(const char *filename_native)
{
  if( filename_native )
    Internals->InFilename = filename_native;
}

void FileChangeTransferSyntax::SetOutputFileName(const char *filename_native)
{
  if( filename_native )
    Internals->OutFilename = filename_native;
}

bool FileChangeTransferSyntax::InitializeCopy()
{
  if( !Internals->IC )
    {
    return false;
    }
  if( !this->Internals->InitializeCopy )
    {
    const char *filename = this->Internals->InFilename.c_str();
    const char *outfilename = this->Internals->OutFilename.c_str();
      {
      // Make sure to update image meta data:
      std::ifstream is( filename, std::ios::binary );
      ImageRegionReader reader;
      reader.SetStream( is );
      if( !reader.ReadInformation() )
        {
        gdcmDebugMacro( "ImageRegionReader::ReadInformation failed" );
        return false;
        }
      is.clear(); // important
      Internals->PixelDataPos = is.tellg();
      File & file = reader.GetFile();
      DataSet & ds = file.GetDataSet();
      if( ds.FindDataElement( Tag(0x7fe0,0x0010) ) )
        {
        const DataElement & de = ds.GetDataElement( Tag(0x7fe0,0x0010) );
        gdcmAssertAlwaysMacro( "Impossible happen"); (void)de;
        return false;
        }
      FileMetaInformation & fmi = file.GetHeader();
      const TransferSyntax &ts = fmi.GetDataSetTransferSyntax();
      if( ts.IsEncapsulated() )
        {
        gdcmDebugMacro( "Dont know how to handle encapsulated TS: " << ts );
        return false;
        }
      if( ts == TransferSyntax::ImplicitVRBigEndianPrivateGE
       || ts == TransferSyntax::ExplicitVRBigEndian )
        {
        gdcmDebugMacro( "Dont know how to handle TS: " << ts );
        return false;
        }
      Internals->Needbyteswap = (ts == TransferSyntax::ImplicitVRBigEndianPrivateGE
        || ts == TransferSyntax::ExplicitVRBigEndian );
      Internals->Dims = ImageHelper::GetDimensionsValue(file);
      Internals->PF = ImageHelper::GetPixelFormatValue(file);
      Internals->PI = ImageHelper::GetPhotometricInterpretationValue(file);
      Internals->PC = ImageHelper::GetPlanarConfigurationValue(file);
      if( Internals->PC )
        {
        gdcmDebugMacro( "Dont know how to handle Planar Configuration" );
        return false;
        }
      is.close();

      // do the lossy transfer syntax handling:
      if( Internals->IC->GetLossyFlag() )
        {
        if( !ds.FindDataElement( Tag(0x0008,0x0016) )
          || ds.GetDataElement( Tag(0x0008,0x0016) ).IsEmpty() )
          {
          gdcmErrorMacro( "Missing Tag" );
          return false;
          }
        if( !ds.FindDataElement( Tag(0x0008,0x0018) )
          || ds.GetDataElement( Tag(0x0008,0x0018) ).IsEmpty() )
          {
          gdcmErrorMacro( "Missing Tag" );
          return false;
          }
        const DataElement &sopclassuid = ds.GetDataElement( Tag(0x0008,0x0016) );
        const DataElement &sopinstanceuid = ds.GetDataElement( Tag(0x0008,0x0018) );
        // Make sure that const char* pointer will be properly padded with \0 char:
        std::string sopclassuid_str( sopclassuid.GetByteValue()->GetPointer(), sopclassuid.GetByteValue()->GetLength() );
        std::string sopinstanceuid_str( sopinstanceuid.GetByteValue()->GetPointer(), sopinstanceuid.GetByteValue()->GetLength() );

        UIDGenerator uid;
        //ds.Remove( Tag(0x8,0x18) );
          {
          const char *sop = uid.Generate();
          DataElement de( Tag(0x0008,0x0018) );
          VL::Type strlenSOP = (VL::Type) strlen(sop);
          de.SetByteValue( sop, strlenSOP );
          de.SetVR( Attribute<0x0008, 0x0018>::GetVR() );
          ds.Replace( de );
          }

        FileDerivation fd;
        fd.SetFile( file );
        fd.AddReference( sopclassuid_str.c_str(), sopinstanceuid_str.c_str() );

        // CID 7202 Source Image Purposes of Reference
        // {"DCM",121320,"Uncompressed predecessor"},
        fd.SetPurposeOfReferenceCodeSequenceCodeValue( 121320 );

        // CID 7203 Image Derivation
        // { "DCM",113040,"Lossy Compression" },
        fd.SetDerivationCodeSequenceCodeValue( 113040 );
        fd.SetDerivationDescription( "lossy conversion" );
        if( !fd.Derive() )
          {
          gdcmErrorMacro( "could not derive using input info" );
          return false;
          }
        /*
        (0028,2110) CS [01]                                     #   2, 1 LossyImageCompression
        (0028,2112) DS [15.95]                                  #   6, 1 LossyImageCompressionRatio
        (0028,2114) CS [ISO_10918_1]                            #  12, 1 LossyImageCompressionMethod
         */
        Attribute<0x0028,0x2110> at1;
        at1.SetValue( "01" );
        ds.Replace( at1.GetAsDataElement() );
#if 0
        double ratio = 1234567890123456; // DS is 16bytes
        Attribute<0x0028,0x2112> at2;
        at2.SetValues( &ratio, 1);
        ds.Replace( at2.GetAsDataElement() );
#else
        char buf[16+1];
        buf[16] = 0;
        memset( buf, ' ', 16 );
        DataElement de1( Tag(0x0028,0x2112) );
        de1.SetByteValue( buf, (uint32_t)strlen( buf ) );
        ds.Replace( de1 );
#endif
        /*
        The Defined Terms for Lossy Image Compression Method (0028,2114) are :
        ISO_10918_1 = JPEG Lossy Compression
        ISO_14495_1 = JPEG-LS Near-lossless Compression
        ISO_15444_1 = JPEG 2000 Irreversible Compression
        ISO_13818_2 = MPEG2 Compression
         */
        Attribute<0x0028,0x2114> at3;
        const TransferSyntax ts_orig = Internals->TS;
        if( ts_orig == TransferSyntax::JPEG2000 )
          {
          static const CSComp newvalues2[] = {"ISO_15444_1"};
          at3.SetValues( newvalues2, 1 );
          }
        else if( ts_orig == TransferSyntax::JPEGLSNearLossless )
          {
          static const CSComp newvalues2[] = {"ISO_14495_1"};
          at3.SetValues( newvalues2, 1 );
          }
        else if (
          ts_orig == TransferSyntax::JPEGBaselineProcess1 ||
          ts_orig == TransferSyntax::JPEGExtendedProcess2_4 ||
          ts_orig == TransferSyntax::JPEGExtendedProcess3_5 ||
          ts_orig == TransferSyntax::JPEGSpectralSelectionProcess6_8 ||
          ts_orig == TransferSyntax::JPEGFullProgressionProcess10_12 )
          {
          static const CSComp newvalues2[] = {"ISO_10918_1"};
          at3.SetValues( newvalues2, 1 );
          }
        else
          {
          gdcmErrorMacro(
            "Pixel Data is lossy but I cannot find the original transfer syntax" );
          return false;
          }
        ds.Replace( at3.GetAsDataElement() );
        }
      Writer writer;
      fmi.Clear();
      fmi.SetDataSetTransferSyntax( Internals->TS );
      writer.SetFileName( outfilename );
      writer.SetFile( file );
      if( !writer.Write() ) return false;
      }
    this->Internals->InitializeCopy = true;
    }
  return true;
}

} // end namespace gdcm
