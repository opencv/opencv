/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmImageWriter.h"
#include "gdcmImageReader.h"
#include "gdcmImage.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmAttribute.h"

/*
Let's test the relation in between lossless transfer syntax and lossless compressed stream.

D_CLUNIE_CT1_J2KR.dcm
D_CLUNIE_CT1_J2KI.dcm

D_CLUNIE_MR1_JPLL.dcm
D_CLUNIE_MR1_JPLY.dcm

D_CLUNIE_CT1_JLSL.dcm
D_CLUNIE_CT1_JLSN.dcm

*/

struct LossLessTest
{
  const char *lossyfile;
  const char *losslessfile;
  gdcm::TransferSyntax::TSType lossyts;
  gdcm::TransferSyntax::TSType losslessts;
};

static const LossLessTest LossLessTestArray[] = {
 { "/D_CLUNIE_CT1_J2KI.dcm", "/D_CLUNIE_CT1_J2KR.dcm", gdcm::TransferSyntax::JPEG2000, gdcm::TransferSyntax::JPEG2000Lossless },
 { "/D_CLUNIE_MR1_JPLY.dcm", "/D_CLUNIE_MR1_JPLL.dcm", gdcm::TransferSyntax::JPEGExtendedProcess2_4, gdcm::TransferSyntax::JPEGLosslessProcess14_1 },
 { "/D_CLUNIE_CT1_JLSN.dcm", "/D_CLUNIE_CT1_JLSL.dcm", gdcm::TransferSyntax::JPEGLSNearLossless, gdcm::TransferSyntax::JPEGLSLossless }
};

int TestImageWriter2(int , char *[])
{
  const char *directory = gdcm::Testing::GetDataRoot();
  gdcm::Trace::WarningOff();
  static const unsigned int ntests = sizeof( LossLessTestArray ) / sizeof( *LossLessTestArray );
  for( unsigned int test = 0; test < ntests; ++test )
    {
    const LossLessTest &ltest = LossLessTestArray[test];

    const char *j2k_filename1 = ltest.lossyfile;
    const char *j2k_filename2 = ltest.losslessfile;
    //const char j2k_filename1[] = "/D_CLUNIE_CT1_J2KI.dcm";
    //const char j2k_filename2[] = "/D_CLUNIE_CT1_J2KR.dcm";
    std::string filename_lossy = directory;
    filename_lossy += j2k_filename1;
    std::string filename_lossless = directory;
    filename_lossless += j2k_filename2;

    // Create directory first:
    const char subdir[] = "TestImageWriter2";
    std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
    if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
      {
      gdcm::System::MakeDirectory( tmpdir.c_str() );
      //return 1;
      }
    std::string outfilename_lossy = gdcm::Testing::GetTempFilename( filename_lossy.c_str(), subdir );
    std::string outfilename_lossless = gdcm::Testing::GetTempFilename( filename_lossless.c_str(), subdir );

      {
      gdcm::ImageReader reader;
      reader.SetFileName( filename_lossy.c_str() );
      if( !reader.Read() )
        {
        return 1;
        }

      gdcm::Image &ir = reader.GetImage();
      if( !ir.IsLossy() )
        {
        std::cerr << "Image is not lossy" << std::endl;
        std::cerr << filename_lossy << std::endl;
        return 1;
        }

      //if( ir.GetTransferSyntax() != gdcm::TransferSyntax::JPEG2000 )
      if( ir.GetTransferSyntax() != ltest.lossyts )
        {
        std::cerr << filename_lossy << " " << filename_lossless << std::endl;
        std::cerr << ir.GetTransferSyntax() << " vs " << gdcm::TransferSyntax::GetTSString( ltest.lossyts ) << std::endl;
        return 1;
        }

      //ir.SetTransferSyntax( gdcm::TransferSyntax::JPEG2000Lossless );
      ir.SetTransferSyntax( ltest.losslessts );

      gdcm::ImageWriter writer;
      writer.SetImage( ir );
      writer.SetFileName( outfilename_lossy.c_str() );
      // We should never authorized writing of image that was lossy compress and declare as lossless
      if( writer.Write() )
        {
        gdcm::Attribute<0x0028,0x2110> at;
        at.Set( writer.GetFile().GetDataSet() );
        if( at.GetValue() != "01" )
          {
          std::cerr << "We should never authorize writing of image that was lossy "
            "compress and declare as lossless" << std::endl;
          std::cerr << filename_lossy << " " << filename_lossless << std::endl;
          return 1;
          }
        }
      }

    // But the contrary is ok:
      {
      gdcm::ImageReader reader;
      reader.SetFileName( filename_lossless.c_str() );
      if( !reader.Read() )
        {
        return 1;
        }

      gdcm::Image &ir = reader.GetImage();
      //if( ir.GetTransferSyntax() != gdcm::TransferSyntax::JPEG2000Lossless )
      if( ir.GetTransferSyntax() != ltest.losslessts )
        {
        return 1;
        }

      //ir.SetTransferSyntax( gdcm::TransferSyntax::JPEG2000 );
      ir.SetTransferSyntax( ltest.lossyts );

      gdcm::ImageWriter writer;
      writer.SetImage( ir );
      writer.SetFileName( outfilename_lossless.c_str() );
      // It is ok to save a lossless file and declare transfer syntax to JPEG2000
      if( !writer.Write() )
        {
        return 1;
        }
      }
    }

#if 0
  const gdcm::Image &ir = reader.GetImage();

  gdcm::Image image;
  image.SetNumberOfDimensions( ir.GetNumberOfDimensions() );

  const unsigned int *dims = ir.GetDimensions();
  image.SetDimension(0, dims[0] );
  image.SetDimension(1, dims[1] );

  const gdcm::PixelFormat &pixeltype = ir.GetPixelFormat();
  image.SetPixelFormat( pixeltype );

  const gdcm::PhotometricInterpretation &pi = ir.GetPhotometricInterpretation();
  image.SetPhotometricInterpretation( pi );

  unsigned long len = image.GetBufferLength();
  assert( len = ir.GetBufferLength() );
  std::vector<char> buffer;
  buffer.resize(len); // black image

  gdcm::ByteValue *bv = new gdcm::ByteValue(buffer);
  gdcm::DataElement pixeldata( gdcm::Tag(0x7fe0,0x0010) );
  pixeldata.SetValue( *bv );
  image.SetDataElement( pixeldata );

  const char filename[] = "toto.dcm";
  gdcm::ImageWriter writer;
  writer.SetImage( image );
  writer.SetFileName( filename );

  gdcm::File& file = writer.GetFile();
  gdcm::DataSet& ds = file.GetDataSet();

    gdcm::DataElement de( gdcm::Tag(0x0010,0x0010) );
    const char s[] = "GDCM^Rocks";
    de.SetByteValue( s, strlen( s ) );
    ds.Insert( de );

  if( !writer.Write() )
    {
    return 1;
    }

  return 0;
#endif
  return 0;
}
