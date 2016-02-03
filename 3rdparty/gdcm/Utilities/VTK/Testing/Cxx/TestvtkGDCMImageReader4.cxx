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
#include "vtkGDCMImageWriter.h"
#include "vtkMedicalImageProperties.h"

#include "vtkPNGWriter.h"
#include "vtkImageData.h"
#include "vtkStringArray.h"
//#include <vtksys/SystemTools.hxx>

#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmDirectory.h"
#include "gdcmScanner.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"

/*
 * This test shows how can one extent value stored in the vtkMedicalImageProperties
 * For instance we will add the following two value in the struct:
 * (0008,0005) CS [ISO_IR 100]                             #  10, 1 SpecificCharacterSet
 * (0008,0008) CS [ORIGINAL\PRIMARY\AXIAL]                 #  22, 3 ImageType
 */
int TestvtkGDCMImageRead4(const char *filename, bool verbose)
{
  if( verbose )
    std::cerr << "Reading : " << filename << std::endl;

  gdcm::Directory::FilenamesType l;
  vtkGDCMImageReader *reader = vtkGDCMImageReader::New();
  if( gdcm::System::FileIsDirectory( filename ) )
    {
    verbose = false;
    gdcm::Directory d;
    d.Load( filename );
    l = d.GetFilenames();
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
    l.push_back( filename );
    }

  //int canread = reader->CanReadFile( filename );
  reader->Update();

  gdcm::Scanner scanner;
  // (0008,0005) CS [ISO_IR 100]                             #  10, 1 SpecificCharacterSet
  // (0008,0008) CS [ORIGINAL\PRIMARY\AXIAL]                 #  22, 3 ImageType
  const gdcm::Tag t1(0x0008,0x0005);
  const gdcm::Tag t2(0x0008,0x0008);
  scanner.AddTag( t1 );
  scanner.AddTag( t2 );
  const gdcm::Global& g = gdcm::Global::GetInstance();
  const gdcm::Dicts &ds = g.GetDicts();

  bool b = scanner.Scan( l );
  if( !b )
    {
    return 1;
    }

  vtkMedicalImageProperties * medprop = reader->GetMedicalImageProperties();

#if ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
  const char *value1 =  scanner.GetValue( filename, t1 );
  const gdcm::DictEntry& de1 = ds.GetDictEntry( t1 );
  medprop->AddUserDefinedValue(de1.GetName(), value1);

  const char *value2 =  scanner.GetValue( filename, t2 );
  const gdcm::DictEntry& de2 = ds.GetDictEntry( t2 );
  medprop->AddUserDefinedValue(de2.GetName(), value2);
#endif

  if( verbose )
    {
    reader->GetOutput()->Print( cout );
    reader->GetMedicalImageProperties()->Print( cout );
    }

  // Create directory first:
  const char subdir[] = "TestvtkGDCMImageReader4";
  std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }
  std::string gdcmfile = gdcm::Testing::GetTempFilename( filename, subdir );

  vtkGDCMImageWriter *writer = vtkGDCMImageWriter::New();
#if (VTK_MAJOR_VERSION >= 6)
  writer->SetInputConnection( reader->GetOutputPort() );
#else
  writer->SetInput( reader->GetOutput() );
#endif
  writer->SetFileLowerLeft( reader->GetFileLowerLeft() );
  writer->SetDirectionCosines( reader->GetDirectionCosines() );
  writer->SetImageFormat( reader->GetImageFormat() );
  writer->SetFileDimensionality( reader->GetFileDimensionality() );
  writer->SetMedicalImageProperties( reader->GetMedicalImageProperties() );
  writer->SetShift( reader->GetShift() );
  writer->SetScale( reader->GetScale() );
  writer->SetFileName( gdcmfile.c_str() );
  writer->Write();
  if( verbose )  std::cerr << "Write out: " << gdcmfile << std::endl;

  reader->Delete();
  writer->Delete();
  return 0;
}

int TestvtkGDCMImageReader4(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMImageRead4(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestvtkGDCMImageRead4( filename, false );
    ++i;
    }

  return r;
}
