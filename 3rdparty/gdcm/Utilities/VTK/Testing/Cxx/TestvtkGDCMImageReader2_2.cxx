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
#include "vtkMedicalImageProperties.h"

#include "vtkInformation.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkStringArray.h"
// DEBUG
#include "vtkImageColorViewer.h"
#include "vtkRenderWindowInteractor.h"

#include "gdcmSystem.h"
#include "gdcmTesting.h"
#include "gdcmDirectory.h"

static int TestvtkGDCMImageRead(const char *filename, bool verbose)
{
//  if( verbose )
    std::cerr << "Reading : " << filename << std::endl;
  vtkGDCMImageReader2 *reader = vtkGDCMImageReader2::New();
  if( gdcm::System::FileIsDirectory( filename ) )
    {
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

  reader->UpdateInformation();
  if( reader->GetErrorCode() )
    {
    return 1;
    }
  int wext[6];
  reader->GetOutputInformation(0)->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), wext);
  //reader->GetOutput()->Print( std::cout );
  const int mid = (wext[5] - wext[0]) / 2;
  wext[4] = wext[5] = mid;
  reader->SetUpdateExtent( wext );
  //reader->Update();
  int ret = reader->GetExecutive()->Update();
  if( !ret )
    {
    std::cerr << "Problem with: " << filename << std::endl;
    }

  if( verbose )
    {
    reader->GetOutput()->Print( cout );
    reader->GetMedicalImageProperties()->Print( cout );
    }

  if( 0 )
    {
    vtkImageColorViewer *viewer = vtkImageColorViewer::New();
    viewer->SetInputConnection( reader->GetOutputPort() );

    vtkRenderWindowInteractor *iren = vtkRenderWindowInteractor::New();

    viewer->SetupInteractor( iren );
    viewer->Render();

    iren->Initialize();
    iren->Start();
    }

  reader->Delete();
  return ret;
}

int TestvtkGDCMImageReader2_2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMImageRead(filename, true);
    }

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
