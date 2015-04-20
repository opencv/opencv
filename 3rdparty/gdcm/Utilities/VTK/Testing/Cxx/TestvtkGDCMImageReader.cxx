/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMImageReader.h"
#include "vtkMedicalImageProperties.h"

#include "vtkPNGWriter.h"
#include "vtkImageData.h"
#include "vtkStringArray.h"

#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmDirectory.h"

static int TestvtkGDCMImageRead(const char *filename, bool verbose)
{
  if( verbose )
    std::cerr << "Reading : " << filename << std::endl;

  vtkGDCMImageReader *reader = vtkGDCMImageReader::New();
  if( gdcm::System::FileIsDirectory( filename ) )
    {
    verbose = false;
    gdcm::Directory d;
    d.Load( filename );
    gdcm::Directory::FilenamesType l = d.GetFilenames();
    const size_t nfiles = l.size();
    vtkStringArray *sarray = vtkStringArray::New();
    for(unsigned int i = 0; i < nfiles; ++i)
      {
      sarray->InsertNextValue( l[i] );
      }
    assert( sarray->GetNumberOfValues() == (int)nfiles );
    reader->SetFileNames( sarray );
    sarray->Delete();
    }
  else
    {
    reader->SetFileName( filename );
    }

  //int canread = reader->CanReadFile( filename );
  reader->Update();

  if( verbose )
    {
    reader->GetOutput()->Print( cout );
    reader->GetMedicalImageProperties()->Print( cout );
    }

  if( verbose && false )
    {
    // Create directory first:
    const char subdir[] = "TestvtkGDCMImageReader";
    std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
    if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
      {
      gdcm::System::MakeDirectory( tmpdir.c_str() );
      //return 1;
      }
    std::string pngfile = gdcm::Testing::GetTempFilename( filename, subdir );

    vtkPNGWriter *writer = vtkPNGWriter::New();
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
    writer->SetInputConnection( reader->GetOutputPort() );
#else
    writer->SetInput( reader->GetOutput() );
#endif
    pngfile += ".png";
    writer->SetFileName( pngfile.c_str() );
    //writer->Write();
    writer->Delete();
    cout << "Wrote PNG output into:" << pngfile << std::endl;
    }

  reader->Delete();
  return 0;
}

int TestvtkGDCMImageReader(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMImageRead(filename, true);
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
    r += TestvtkGDCMImageRead( filename, false );
    ++i;
    }

  return r;
}
