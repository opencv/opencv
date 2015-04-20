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

#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmReader.h"
#include "gdcmImageReader.h"
#include "gdcmFilename.h"
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmByteSwap.h"

static int TestFileChangeTransferSyntax1Func(const char *filename, bool verbose = false)
{
  using namespace gdcm;
  if( verbose )
    std::cout << "Processing: " << filename << std::endl;

  // Create directory first:
  const char subdir[] = "TestFileChangeTransferSyntax1";
  std::string tmpdir = Testing::GetTempDirectory( subdir );
  if( !System::FileIsDirectory( tmpdir.c_str() ) )
    {
    System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string outfilename = Testing::GetTempFilename( filename, subdir );
  if( verbose )
    std::cout << "Generating: " << outfilename << std::endl;

  const gdcm::TransferSyntax ts( TransferSyntax::JPEGLosslessProcess14_1 );

  gdcm::FileChangeTransferSyntax fcts;
  fcts.SetTransferSyntax( ts );
  fcts.SetInputFileName( filename );
  fcts.SetOutputFileName( outfilename.c_str() );

  ImageCodec *ic = fcts.GetCodec();
  if( !ic )
    {
    return 1;
    }

  if( !fcts.Change() )
    {
    gdcm::Reader reader;
    reader.SetFileName( filename );
    if( !reader.Read() )
      {
      std::cerr << "not dicom" << std::endl;
      return 1;
      }
    const gdcm::File & file = reader.GetFile();
    const gdcm::FileMetaInformation & fmi = file.GetHeader();
    const TransferSyntax &tsref = fmi.GetDataSetTransferSyntax();
    if( tsref.IsEncapsulated() )
      {
      if( verbose )
        std::cout << "Will not generate (encaps): " << outfilename << std::endl;
      return 0;
      }

    gdcm::Filename fn( filename );
    const char *name = fn.GetName();
    // Special handling:
    if( strcmp(name, "CT-MONO2-12-lomb-an2.acr" ) == 0
    || strcmp(name, "LIBIDO-8-ACR_NEMA-Lena_128_128.acr") == 0
    || strcmp(name, "gdcm-ACR-LibIDO.acr") == 0
    || strcmp(name, "gdcm-MR-SIEMENS-16-2.acr") == 0
    || strcmp(name, "libido1.0-vol.acr") == 0
    || strcmp(name, "test.acr") == 0
    || strcmp(name, "LIBIDO-24-ACR_NEMA-Rectangle.dcm") == 0
    || strcmp(name, "MR_Spectroscopy_SIEMENS_OF.dcm") == 0 // not an image
    || strcmp(name, "ELSCINT1_PMSCT_RLE1.dcm") == 0
    || strcmp(name, "ELSCINT1_PMSCT_RLE1_priv.dcm") == 0
    || strcmp(name, "gdcm-CR-DCMTK-16-NonSamplePerPix.dcm") == 0
    || strcmp(name, "SIEMENS_MAGNETOM-12-ACR_NEMA_2-Modern.dcm") == 0
    || strcmp(name, "PrivateGEImplicitVRBigEndianTransferSyntax16Bits.dcm") == 0 // Implicit VR Big Endian DLX (G.E Private)
    || strcmp(name, "MR_GE_with_Private_Compressed_Icon_0009_1110.dcm") == 0 // Explicit VR Big Endian
    || strcmp(name, "US-RGB-8-epicard.dcm") == 0 // Explicit VR Big Endian
    || strcmp(name, "GE_DLX-8-MONO2-PrivateSyntax.dcm") == 0 // Implicit VR Big Endian DLX (G.E Private)
    || strcmp(name, "GE_CT_With_Private_compressed-icon.dcm") == 0 // Explicit VR Big Endian
    || strcmp(name, "JDDICOM_Sample2-dcmdjpeg.dcm") == 0 // cannot recreate FMI
    || strcmp(name, "DMCPACS_ExplicitImplicit_BogusIOP.dcm") == 0 // ImageRegionReader does not handle it
    || strcmp(name, "unreadable.dcm") == 0 // No Pixel Data (old ACR-NEMA)
    || strncmp(name, "DICOMDIR", 8) == 0 // DICOMDIR*
    || strncmp(name, "dicomdir", 8) == 0 // dicomdir*
    )
      {
      if( verbose )
        std::cout << "Will not generate: " << outfilename << std::endl;
      return 0;
      }
    std::cerr << "Could not change: " << filename << std::endl;
    return 1;
    }

  // Let's read that file back in !
  gdcm::ImageReader reader2;
  reader2.SetFileName( outfilename.c_str() );
  if ( !reader2.Read() )
    {
    std::cerr << "Could not even reread our generated file : " << outfilename << std::endl;
    return 1;
    }
  // Check that after decompression we still find the same thing:
  int res = 0;
  gdcm::Image img = reader2.GetImage();
  int pc = 0;

  // When recompressing: US-RGB-8-epicard.dcm, make sure to compute the md5 using the
  // same original Planar Configuration...
  if( (int)img.GetPlanarConfiguration() !=  pc )
    {
    gdcm::ImageChangePlanarConfiguration icpc;
    icpc.SetInput( reader2.GetImage() );
    icpc.SetPlanarConfiguration( pc );
    icpc.Change();
    img = icpc.GetOutput();
    }
  //std::cerr << "Success to read image from file: " << filename << std::endl;
  unsigned long len = img.GetBufferLength();
  char* buffer = new char[len];
  bool res2 = img.GetBuffer(buffer);
  if( !res2 )
    {
    std::cerr << "could not get buffer: " << outfilename << std::endl;
    return 1;
    }
  // On big Endian system we have byteswapped the buffer (duh!)
  // Since the md5sum is byte based there is now way it would detect
  // that the file is written in big endian word, so comparing against
  // a md5sum computed on LittleEndian would fail. Thus we need to
  // byteswap (again!) here:
#ifdef GDCM_WORDS_BIGENDIAN
  if( img.GetPixelFormat().GetBitsAllocated() == 16 )
    {
    assert( !(len % 2) );
    assert( img.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1
      || img.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME2 );
    gdcm::ByteSwap<unsigned short>::SwapRangeFromSwapCodeIntoSystem(
      (unsigned short*)buffer, gdcm::SwapCode::LittleEndian, len/2);
    }
#endif
  const char *ref = gdcm::Testing::GetMD5FromFile(filename);

  char digest[33];
  gdcm::Testing::ComputeMD5(buffer, len, digest);
  if( !ref )
    {
    // new regression image needs a md5 sum
    std::cerr << "Missing md5 " << digest << " for: " << filename <<  std::endl;
    //assert(0);
    res = 1;
    }
  else if( strcmp(digest, ref) )
    {
    std::cerr << "Problem reading image from: " << filename << std::endl;
    std::cerr << "Found " << digest << " instead of " << ref << std::endl;
    res = 1;
    }
  if(res)
    {
    std::cerr << "problem with: " << outfilename << std::endl;
    }
  if( verbose )
    {
    std::cout << "file was written in: " << outfilename << std::endl;
    }

  delete[] buffer;
  return res;
}

int TestFileChangeTransferSyntax1(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestFileChangeTransferSyntax1Func(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestFileChangeTransferSyntax1Func( filename );
    ++i;
    }

  return r;
}
