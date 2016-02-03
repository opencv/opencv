/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMImageReader2.h"
#include "vtkGDCMTesting.h"
#include "vtkMedicalImageProperties.h"

#include "vtkMetaImageWriter.h"
#include "vtkImageData.h"
#include "vtkStringArray.h"
//#include <vtksys/SystemTools.hxx>

#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmDirectory.h"
#include "gdcmMediaStorage.h"

static int TestvtkGDCMMetaImageWrite(const char *filename, bool verbose)
{
  if( verbose )
    std::cerr << "Reading : " << filename << std::endl;

  vtkGDCMImageReader2 * reader = vtkGDCMImageReader2::New();
  reader->FileLowerLeftOn();
  reader->DebugOff();

  int canread = reader->CanReadFile( filename );
  if( !canread )
    {
    reader->Delete();
    gdcm::Filename fn( filename );
    if( fn.GetName() == std::string("ELSCINT1_PMSCT_RLE1.dcm" ) )
      {
      // No Pixel Data...
      return 0;
      }
    const char *refms = gdcm::Testing::GetMediaStorageFromFile(filename);
    if( gdcm::MediaStorage::IsImage( gdcm::MediaStorage::GetMSType(refms) ) )
      {
      std::cerr << "Problem with: " << filename << std::endl;
      return 1;
      }
    // not an image
    return 0;
    }

  const char *refms = gdcm::Testing::GetMediaStorageFromFile(filename);
  if( !gdcm::MediaStorage::IsImage( gdcm::MediaStorage::GetMSType(refms) ) )
    {
    if( !refms )
      {
      std::cerr << "Missing SOP Class: " << filename << std::endl;
      return 1;
      }
    }

  reader->SetFileName( filename );
  reader->Update();

  if( verbose )
    {
    reader->GetOutput()->Print( cout );
    reader->GetMedicalImageProperties()->Print( cout );
    }

//  if( verbose )
    {
    // Create directory first:
    const char subdir[] = "TestvtkGDCMMetaImageWriter2";
    std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
    if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
      {
      gdcm::System::MakeDirectory( tmpdir.c_str() );
      //return 1;
      }
    std::string mhdfile = gdcm::Testing::GetTempFilename( filename, subdir );
    std::string rawfile = mhdfile;

    vtkMetaImageWriter *writer = vtkMetaImageWriter::New();
    writer->SetCompression( false );
    writer->SetInputConnection( reader->GetOutputPort() );
    mhdfile += ".mhd";
    rawfile += ".raw";
    writer->SetFileName( mhdfile.c_str() );
    writer->Write();
    writer->Delete();
    if( verbose )
      cout << "Wrote MHD output into: " << mhdfile << std::endl;
    char digestmhd[33] = {};
    char digestraw[33] = {};
    bool bmhd = gdcm::Testing::ComputeFileMD5( mhdfile.c_str() , digestmhd );
    bool braw = gdcm::Testing::ComputeFileMD5( rawfile.c_str() , digestraw );
    assert( bmhd && braw ); (void)bmhd; (void)braw;
    const char * mhdref = vtkGDCMTesting::GetMHDMD5FromFile(filename);
    const char * rawref = vtkGDCMTesting::GetRAWMD5FromFile(filename);
    if( !mhdref || !rawref )
      {
      std::cout << "Found: \"" << filename << "\",\"" << digestmhd << "\", \"" << digestraw << "\"" << std::endl;
      return 1;
      }
    else if( strcmp(digestraw, rawref) )
      {
      std::cerr << "Problem reading RAW from: " << rawfile << std::endl;
      std::cerr << "Found " << digestraw << " instead of " << rawref << std::endl;

      return 1;
      }
    else if( strcmp(digestmhd, mhdref) )
      {
      std::cerr << "Problem reading MHD from: " << mhdfile << std::endl;
      std::cerr << "Found " << digestmhd << " instead of " << mhdref << std::endl;

      return 1;
      }
    }

  reader->Delete();
  return 0;
}

int TestvtkGDCMMetaImageWriter2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMMetaImageWrite(filename, true);
    }

#if 0
  unsigned int n = vtkGDCMTesting::GetNumberOfMD5MetaImages();
  for( unsigned int i = 0; i < n; ++i )
    {
    const char * const * p = vtkGDCMTesting::GetMD5MetaImage(i);
    std::cout << p[1] << "  " << p[0] << ".mhd" <<  std::endl;
    std::cout << p[2] << "  " << p[0] << ".raw" << std::endl;
    }
  return 0;
#endif

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestvtkGDCMMetaImageWrite( filename, false );
    ++i;
    }

  return r;
}
