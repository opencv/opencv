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

#include "vtkDirectory.h"
#include "vtkImageActor.h"
#include "vtkImageChangeInformation.h"
#include "vtkImageMapToWindowLevelColors.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkRenderer.h"

int TestvtkImageChangeInformation(int argc, char *argv[])
{
  if( argc < 2 )
    {
    std::cerr << "Need arg" << std::endl;
    return 1;
    }
  const char *dirname = argv[1];

  vtkDirectory * dir = vtkDirectory::New();
  int r = dir->Open( dirname );
  if( !r )
    {
    std::cerr << "Cannot open dir:" << dirname << std::endl;
    return 1;
    }
  vtkIdType nfiles = dir->GetNumberOfFiles();

  vtkImageChangeInformation *ici = vtkImageChangeInformation::New();
  ici->SetOutputOrigin(0, 0, 0);
  vtkImageMapToWindowLevelColors * windowlevel = vtkImageMapToWindowLevelColors::New();
  windowlevel->SetInput( ici->GetOutput() );
  vtkImageActor *imageactor = vtkImageActor::New();
  imageactor->SetInput( windowlevel->GetOutput() );

  // Create the RenderWindow, Renderer and both Actors
  vtkRenderer *ren = vtkRenderer::New();
  vtkRenderWindow *renWin = vtkRenderWindow::New();
  renWin->AddRenderer (ren);

  // Add the actors to the renderer, set the background and size
  ren->AddActor (imageactor);

  double range[2];
  for ( vtkIdType file = 0; file < nfiles; ++file )
    {
    vtkGDCMImageReader * reader = vtkGDCMImageReader::New();
    ici->SetInput( reader->GetOutput() );
    std::string filename = dir->GetFile(file);
    if( filename.find( "dcm" ) != std::string::npos )
      {
      std::string fullpath = dirname;
      fullpath += "/";
      fullpath += filename;
      std::cerr << "Processing: " << fullpath << std::endl;

      reader->SetFileName( fullpath.c_str() );
      //reader->Update();
      //ici->GetOutput()->Update(); // bad !
      ici->GetOutput()->GetScalarRange(range);
      //reader->GetOutput()->GetScalarRange(range);
      renWin->Render();
      std::cerr << "Range: " << range[0] << " " << range[1] << std::endl;
      }
    reader->Delete();
    }

  dir->Delete();
  ici->Delete();
  windowlevel->Delete();
  imageactor->Delete();

  return 0;
}
