/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMThreadedImageReader2.h"
#include "vtkGDCMImageReader.h"
#include "vtkCommand.h"

#include "gdcmDirectory.h"
#include "gdcmSystem.h"
#include "gdcmImageReader.h"
#include "gdcmTesting.h"

#include "vtkPNGWriter.h"
#include "vtkStringArray.h"
#include "vtkStructuredPointsWriter.h"
#include "vtkImageData.h"
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#include <vtksys/SystemTools.hxx>
#endif

class ProgressObserver : public vtkCommand
{
public:
  static ProgressObserver* New() {
    return new ProgressObserver;
  }

  virtual void Execute(vtkObject* caller, unsigned long event, void *callData)
    {
    (void)callData;
    if( event == vtkCommand::ProgressEvent )
      {
      std::cout << ((vtkGDCMThreadedImageReader2*)caller)->GetProgress() << std::endl;
      }
    }
};


template <typename TReader>
int ExecuteInformation(const char *filename, TReader *vtkreader)
{
  gdcm::ImageReader reader;
  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    return 0;
    }
  const gdcm::Image &image = reader.GetImage();
  const unsigned int *dims = image.GetDimensions();

  // Set the Extents.
  assert( image.GetNumberOfDimensions() >= 2 );
  int dataextent[6] = {};
  dataextent[0] = 0;
  dataextent[1] = dims[0] - 1;
  dataextent[2] = 0;
  dataextent[3] = dims[1] - 1;
  if( image.GetNumberOfDimensions() == 2 )
    {
    vtkStringArray *filenames = vtkreader->GetFileNames();
    if ( filenames && filenames->GetNumberOfValues() > 1 )
      {
      dataextent[4] = 0;
      dataextent[5] = (int)filenames->GetNumberOfValues() - 1;
      }
    else
      {
      dataextent[4] = 0;
      dataextent[5] = 0;
      }
    }
  else
    {
    assert( image.GetNumberOfDimensions() == 3 );
    //this->FileDimensionality = 3;
    dataextent[4] = 0;
    dataextent[5] = dims[2] - 1;
    }
  //this->DataSpacing[0] = 1.;
  //this->DataSpacing[1] = -1.;
  //this->DataSpacing[2] = 1.;

  gdcm::PixelFormat pixeltype = image.GetPixelFormat();
  int datascalartype = VTK_VOID;
  switch( pixeltype )
    {
  case gdcm::PixelFormat::INT8:
    datascalartype = VTK_CHAR;
    break;
  case gdcm::PixelFormat::UINT8:
    datascalartype = VTK_UNSIGNED_CHAR;
    break;
  case gdcm::PixelFormat::INT16:
    datascalartype = VTK_SHORT;
    break;
  case gdcm::PixelFormat::UINT16:
    datascalartype = VTK_UNSIGNED_SHORT;
    break;
  case gdcm::PixelFormat::INT32:
    datascalartype = VTK_INT;
    break;
  case gdcm::PixelFormat::UINT32:
    datascalartype = VTK_UNSIGNED_INT;
    break;
  default:
    ;
    }
  if( datascalartype == VTK_VOID )
    {
    return 0;
    }

  unsigned int numberOfScalarComponents = pixeltype.GetSamplesPerPixel();

  vtkreader->SetDataExtent( dataextent );
  vtkreader->SetDataScalarType ( datascalartype );
  //vtkreader->SetShift( image.GetIntercept() );
  //vtkreader->SetScale( image.GetSlope() );
  vtkreader->SetNumberOfScalarComponents( numberOfScalarComponents );
  vtkreader->LoadOverlaysOff();
  if( image.GetNumberOfOverlays() )
    {
    vtkreader->LoadOverlaysOn();
    }

  return 1;
}

template <typename TReader>
int TestvtkGDCMThreadedImageRead2(const char *filename, bool verbose = false)
{
  TReader *reader = TReader::New();
  reader->FileLowerLeftOn();
  //reader->CanReadFile( filename );
  if( verbose) std::cerr << "Reading : " << filename << std::endl;

  const char *refimage = NULL;
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
    refimage = sarray->GetValue( 0 ); // Ok since sarray is ref count
    }
  else
    {
    reader->SetFileName( filename );
    refimage = filename;
    }

  // In all cases we need to explicitly say what the image type is:
  if( !ExecuteInformation<TReader>(refimage, reader) )
    {
    std::cerr << "file: " << refimage << " is not an image. giving up" << std::endl;
    reader->Delete();
    return 0;
    }


  ProgressObserver *obs = ProgressObserver::New();
  if( verbose )
    {
    reader->AddObserver( vtkCommand::ProgressEvent, obs);
    }

  reader->Update();
  obs->Delete();

  //reader->GetOutput()->Print( cout );
  //reader->GetOutput(1)->Print( cout );

  if( reader->GetNumberOfOverlays() )
    {
    vtkPNGWriter *writer = vtkPNGWriter::New();
#if (VTK_MAJOR_VERSION >= 6)
    writer->SetInputConnection( reader->GetOutputPort(1) );
#else
    writer->SetInput( reader->GetOutput(1) );
#endif
    const char subdir[] = "TestvtkGDCMThreadedImageReader2";
    // Create directory first:
    std::string tmpdir = gdcm::Testing::GetTempDirectory( subdir );
    if( !gdcm::System::FileIsDirectory( tmpdir.c_str() ) )
      {
      gdcm::System::MakeDirectory( tmpdir.c_str() );
      //return 1;
      }
    std::string pngfile = gdcm::Testing::GetTempFilename( filename, subdir );
    //pngfile += vtksys::SystemTools::GetFilenameWithoutExtension( filename );
    pngfile += ".png";
    writer->SetFileName( pngfile.c_str() );
    if( verbose ) std::cerr << pngfile << std::endl;
    //writer->Write();
    writer->Delete();
    }

/*
  vtkStructuredPointsWriter *writer = vtkStructuredPointsWriter::New();
  writer->SetInput( reader->GetOutput() );
  writer->SetFileName( "TestvtkGDCMThreadedImageReader2.vtk" );
  writer->SetFileTypeToBinary();
  //writer->Write();
  writer->Delete();
*/

  bool compute = false;
  if( verbose && compute )
    {
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 2 )
    double *s = reader->GetOutput()->GetScalarRange();
#else
    float *s = reader->GetOutput()->GetScalarRange();
#endif
    std::cout << s[0] << " " << s[1] << std::endl;
    }

  reader->Delete();

  return 0;
}

int TestvtkGDCMThreadedImageReader2(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMThreadedImageRead2<vtkGDCMThreadedImageReader2>(filename, true);
    }

  // else
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestvtkGDCMThreadedImageRead2<vtkGDCMThreadedImageReader2>( filename );
    ++i;
    }

  return r;
}
