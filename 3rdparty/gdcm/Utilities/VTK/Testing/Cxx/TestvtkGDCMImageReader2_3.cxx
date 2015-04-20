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
#include "vtkGDCMImageWriter.h"
#include "vtkStringArray.h"
#include "vtkImageData.h"
#include "vtkImageChangeInformation.h"

#include "gdcmFilename.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmDirectory.h"
#include "gdcmIPPSorter.h"
#include "gdcmFilenameGenerator.h"

#ifndef vtkFloatingPointType
#define vtkFloatingPointType float
#endif
/*
 * Test to show the pipeline for
 * IPPSorter -> vtkGDCMImageReader2 -> vtkImageChangeInformation
 */
int TestvtkGDCMImageReader2_3(int , char *[])
{
  const char *directory = gdcm::Testing::GetDataRoot();
  std::vector<std::string> filenames;
#if 1
  std::string file0 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq0.dcm";
  std::string file1 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq1.dcm";
  std::string file2 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq2.dcm";
  std::string file3 = std::string(directory) + "/SIEMENS_MAGNETOM-12-MONO2-FileSeq3.dcm";
#else
  std::string file0 = "/tmp/0.dcm";
  std::string file1 = "/tmp/1.dcm";
  std::string file2 = "/tmp/2.dcm";
  std::string file3 = "/tmp/3.dcm";
#endif
  filenames.push_back( file1 );
  filenames.push_back( file3 );
  filenames.push_back( file2 );
  filenames.push_back( file0 );
  gdcm::IPPSorter s;
  s.SetComputeZSpacing( true );
  s.SetZSpacingTolerance( 1e-10 );
  bool b = s.Sort( filenames );
  if( !b )
    {
    std::cerr << "Failed to sort:" << directory << std::endl;
    return 1;
    }
  std::cout << "Sorting succeeded:" << std::endl;
  s.Print( std::cout );

  std::cout << "Found z-spacing:" << std::endl;
  std::cout << s.GetZSpacing() << std::endl;
  double ippzspacing = s.GetZSpacing();
  if( ippzspacing != 5.5 )
    {
    // This should be test in another specific test ...
    return 1;
    }

  const std::vector<std::string> & sorted = s.GetFilenames();
  vtkGDCMImageReader2 * reader = vtkGDCMImageReader2::New();
  vtkStringArray *files = vtkStringArray::New();
  std::vector< std::string >::const_iterator it = sorted.begin();
  for( ; it != sorted.end(); ++it)
    {
    const std::string &f = *it;
    files->InsertNextValue( f.c_str() );
    }
  reader->SetFileNames( files );
  reader->Update();

  const vtkFloatingPointType *spacing = reader->GetOutput()->GetSpacing();
  std::cout << spacing[0] << "," << spacing[1] << "," << spacing[2] << std::endl;
  int ret = 0;
  if( spacing[2] != 0.5 )
    {
    // Spacing Between Slice is set to 0.5 in those files
    ret++;
    }

  // try again but this time we want 5.5 to be the spacing
  vtkGDCMImageReader2 * reader2 = vtkGDCMImageReader2::New();
  reader2->SetDataSpacing( spacing[0], spacing[1], ippzspacing );
  reader2->SetFileNames( files );
  //reader2->FileLowerLeftOn(); // TODO
  reader2->Update();
  const vtkFloatingPointType *spacing2 = reader2->GetOutput()->GetSpacing();
  std::cout << spacing2[0] << "," << spacing2[1] << "," << spacing2[2] << std::endl;
  // You need to use this class to preserve spacing
  // across pipeline re-execution
  vtkImageChangeInformation *change = vtkImageChangeInformation::New();
#if (VTK_MAJOR_VERSION >= 6)
  change->SetInputConnection( reader2->GetOutputPort() );
#else
  change->SetInput( reader2->GetOutput() );
#endif
  change->SetOutputSpacing( spacing2[0], spacing2[1], ippzspacing );
  change->Update();

  const vtkFloatingPointType *spacing3 = change->GetOutput()->GetSpacing();
  std::cout << spacing3[0] << "," << spacing3[1] << "," << spacing3[2] << std::endl;
  if( spacing3[2] != 5.5 )
    {
    ret++;
    }

  // Ok Let's try to write this volume back to disk:
  vtkGDCMImageWriter *writer = vtkGDCMImageWriter::New();
#if (VTK_MAJOR_VERSION >= 6)
  writer->SetInputData( change->GetOutput() );
#else
  writer->SetInput( change->GetOutput() );
#endif
  writer->SetFileDimensionality( 2 );
  //writer->SetFileLowerLeft( reader2->GetFileLowerLeft() ); // TODO
  writer->SetMedicalImageProperties( reader2->GetMedicalImageProperties() ); // nasty
  writer->SetDirectionCosines( reader2->GetDirectionCosines() );
  writer->SetImageFormat( reader2->GetImageFormat() );
  const char subdir[] = "TestvtkGDCMImageReader2_3";
  std::string tmpdir = gdcm::Testing::GetTempDirectory(subdir);
  if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
    {
    gdcm::System::MakeDirectory( tmpdir.c_str() );
    //return 1;
    }

  tmpdir += "/";
  const char tfilename[] = "SIEMENS_MAGNETOM-12-MONO2-FileSeq%01d.dcm";
  tmpdir += tfilename;
  gdcm::FilenameGenerator fg;
  fg.SetPattern( tmpdir.c_str() );
  fg.SetNumberOfFilenames( files->GetNumberOfValues() );
  bool bb = fg.Generate();
  if( !bb )
    {
    std::cerr << "FilenameGenerator::Generate failed" << std::endl;
    return 1;
    }
  if( !fg.GetNumberOfFilenames() )
    {
    std::cerr << "No filenames generated" << std::endl;
    return 1;
    }
  vtkStringArray *wfilenames = vtkStringArray::New();
  for(unsigned int i = 0; i < fg.GetNumberOfFilenames(); ++i)
    {
    wfilenames->InsertNextValue( fg.GetFilename(i) );
    std::cerr << fg.GetFilename(i) << std::endl;
    }
  assert( (gdcm::FilenameGenerator::SizeType)wfilenames->GetNumberOfValues() == fg.GetNumberOfFilenames() );
  writer->SetFileNames( wfilenames );
  wfilenames->Delete();
  writer->Write();

  change->Delete();
  reader->Delete();
  reader2->Delete();
  writer->Delete();
  files->Delete();

  return ret;
}
