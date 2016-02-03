/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMThreadedImageReader.h"

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkMedicalImageProperties.h"
#include "vtkStringArray.h"
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDemandDrivenPipeline.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#endif /* (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 ) */

#include "gdcmImageReader.h"
#include "gdcmDataElement.h"
#include "gdcmByteValue.h"
#include "gdcmSwapper.h"

#include <sstream>

#include <pthread.h>
#include <unistd.h> // sysconf

#ifdef _WIN32
#include <windows.h> // SYSTEM_INFO (mingw)
#endif

#ifdef __APPLE__
// For some reason sysconf + _SC_NPROCESSORS_ONLN is documented on macosx tiger, but it does not compile
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

vtkCxxRevisionMacro(vtkGDCMThreadedImageReader, "$Revision: 1.1 $")
vtkStandardNewMacro(vtkGDCMThreadedImageReader)

// Output Ports are as follow:
// #0: The image/volume (root PixelData element)
// #1: (if present): the Icon Image (0088,0200)
// #2-xx: (if present): the Overlay (60xx,3000)

#define IconImagePortNumber 1
#define OverlayPortNumber   2

vtkGDCMThreadedImageReader::vtkGDCMThreadedImageReader()
{
  this->LoadIconImage = 0;
  this->UseShiftScale = 1;
}

vtkGDCMThreadedImageReader::~vtkGDCMThreadedImageReader()
{
}

#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#else /* (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 ) */
void vtkGDCMThreadedImageReader::ExecuteInformation()
{
  //std::cerr << "ExecuteInformation" << std::endl;
  // This reader only implement case where image is flipped upside down
  if( !this->FileLowerLeft )
    {
    vtkErrorMacro( "You need to set the FileLowerLeft flag to On" );
    }
  if( this->LoadIconImage )
    {
    vtkErrorMacro( "Icon are not supported" );
    }
  //int * updateExtent = this->Outputs[0]->GetUpdateExtent();
  //std::cout << "UpdateExtent:" << updateExtent[4] << " " << updateExtent[5] << std::endl;

  vtkImageData *output = this->GetOutput();
  output->SetUpdateExtentToWholeExtent(); // pipeline is not reexecuting properly without that...

  int numvol = 1;
  if( this->LoadIconImage)
    {
    numvol = 2;
    }
  if( this->LoadOverlays )
    {
    this->NumberOfOverlays = 1;
    numvol = 3;
    }
  this->SetNumberOfOutputs(numvol);
  assert( numvol == 1 || numvol == 3 );

  // vtkImageReader2::ExecuteInformation only allocate first output
  this->vtkImageReader2::ExecuteInformation();
  // Let's do the other ones ourselves:
  for (int i=1; i<numvol; i++)
    {
    if (!this->Outputs[i])
      {
      vtkImageData * img = vtkImageData::New();
      this->SetNthOutput(i, img);
      img->Delete();
      }
    vtkImageData *output = this->GetOutput(i);
    switch(i)
      {
    case 0:
      output->SetWholeExtent(this->DataExtent);
      output->SetSpacing(this->DataSpacing);
      output->SetOrigin(this->DataOrigin);

      output->SetScalarType(this->DataScalarType);
      output->SetNumberOfScalarComponents(this->NumberOfScalarComponents);
      break;
    case IconImagePortNumber:
      output->SetWholeExtent(this->IconImageDataExtent);
      output->SetScalarType( VTK_UNSIGNED_CHAR );
      output->SetNumberOfScalarComponents( 1 );
      break;
    //case OverlayPortNumber:
    default:
      output->SetWholeExtent(this->DataExtent[0],this->DataExtent[1],
        this->DataExtent[2],this->DataExtent[3],
        0,0
      );
      //output->SetSpacing(this->DataSpacing);
      //output->SetOrigin(this->DataOrigin);
      output->SetScalarType(VTK_UNSIGNED_CHAR);
      output->SetNumberOfScalarComponents(1);
      break;
      }
    }
}

void vtkGDCMThreadedImageReader::ExecuteData(vtkDataObject *output)
{
  //std::cerr << "ExecuteData" << std::endl;
  // In VTK 4.2 AllocateOutputData is reexecuting ExecuteInformation which is bad !
  //vtkImageData *data = this->AllocateOutputData(output);
  vtkImageData *res = vtkImageData::SafeDownCast(output);
  res->SetExtent(res->GetUpdateExtent());
  res->AllocateScalars();

  if( this->LoadIconImage )
    {
/*
    vtkImageData *res = vtkImageData::SafeDownCast(this->Outputs[IconImagePortNumber]);
    res->SetUpdateExtentToWholeExtent();

    res->SetExtent(res->GetUpdateExtent());
    res->AllocateScalars();
*/
    vtkErrorMacro( "IconImage are not supported" );
    }
  if( this->LoadOverlays )
    {
    vtkImageData *res = vtkImageData::SafeDownCast(this->Outputs[OverlayPortNumber]);
    res->SetUpdateExtentToWholeExtent();

    res->SetExtent(res->GetUpdateExtent());
    res->AllocateScalars();
    }

//  if( data->UpdateExtentIsEmpty() )
//    {
//    return;
//    }
  //int * updateExtent = data->GetUpdateExtent();
  //std::cout << "UpdateExtent:" << updateExtent[4] << " " << updateExtent[5] << std::endl;
  RequestDataCompat();
}
#endif /* (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 ) */

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
int vtkGDCMThreadedImageReader::RequestInformation(vtkInformation *request,
                                      vtkInformationVector **inputVector,
                                      vtkInformationVector *outputVector)
{
  (void)request;(void)inputVector;(void)outputVector;
  // Some information need to have been set outside (user specified)
  //assert( this->GetOutput(0)->GetNumberOfPoints() != 0 );
  // For now only handles series:
  if( !this->FileNames && !this->FileName )
    {
    return 0;
    }

  // This reader only implement case where image is flipped upside down
  if( !this->FileLowerLeft )
    {
    vtkErrorMacro( "You need to set the FileLowerLeft flag to On" );
    return 0;
    }

  if( this->FileNames )
    {
    int zmin = 0;
    int zmax = 0;
    zmax = (int)this->FileNames->GetNumberOfValues() - 1;
    if( this->DataExtent[4] != zmin || this->DataExtent[5] != zmax )
      {
      vtkErrorMacro( "Problem with extent" );
      return 0;
      }
    }
  // Cannot deduce anything else otherwise...

  int numvol = 1;
  if( this->LoadIconImage )
    {
    numvol = 2;
    return 0;
    }
  if( this->LoadOverlays )
    {
    this->NumberOfOverlays = 1;
    numvol = 3;
    }
  assert( numvol == 1 || numvol == 3 );
  this->SetNumberOfOutputPorts(numvol);
  assert( this->DataScalarType != VTK_VOID );
  // For each output:
  for(int i = 0; i < numvol; ++i)
    {
    // Allocate !
    if( !this->GetOutput(i) )
      {
      vtkImageData *img = vtkImageData::New();
      this->GetExecutive()->SetOutputData(i, img );
      img->Delete();
      }
    vtkInformation *outInfo = outputVector->GetInformationObject(i);
    switch(i)
      {
    // root Pixel Data
    case 0:
      outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->DataExtent, 6);
      //outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), this->DataExtent, 6);
      outInfo->Set(vtkDataObject::SPACING(), this->DataSpacing, 3);
      outInfo->Set(vtkDataObject::ORIGIN(), this->DataOrigin, 3);
      vtkDataObject::SetPointDataActiveScalarInfo(outInfo, this->DataScalarType, this->NumberOfScalarComponents);
      break;
    // Icon Image
    case IconImagePortNumber:
      outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->IconImageDataExtent, 6);
      vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_UNSIGNED_CHAR, 1);
      break;
    // Overlays:
    //case OverlayPortNumber:
    default:
      outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
        this->DataExtent[0], this->DataExtent[1],
        this->DataExtent[2], this->DataExtent[3],
        0,0 );
      vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_UNSIGNED_CHAR, 1);
      break;
      }

    }

  // Ok let's fill in the 'extra' info:
  //FillMedicalImageInformation(reader);

  return 1;
}
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/

struct threadparams
{
  unsigned int threadid;
  unsigned int nthreads;
  const char **filenames;             // array of filenames thread will process (order is important!)
  unsigned int nfiles;                // number of files the thread will process
  char *scalarpointer;                // start of the image buffer affected to the thread
  char *overlayscalarpointer;
  unsigned long len;                  // This is not required but useful to check if files are consistant
  unsigned long overlaylen;
  unsigned long totalfiles;           // total number of files being processed (needed to compute progress)
  pthread_mutex_t lock;               // critial section for updating progress
  vtkGDCMThreadedImageReader *reader; // needed for calling updateprogress
};

void *ReadFilesThread(void *voidparams)
{
  threadparams *params = static_cast<threadparams *> (voidparams);
  assert( params );

  const unsigned int nfiles = params->nfiles;
  assert( nfiles ); //
  // pre compute progress delta for one file:
  assert( params->totalfiles );
  const double progressdelta = 1. / (double)params->totalfiles;
  for(unsigned int file = 0; file < nfiles; ++file)
    {
    const char *filename = params->filenames[file];
    //std::cerr << filename << std::endl;

    gdcm::ImageReader reader;
    reader.SetFileName( filename );
    if( !reader.Read() )
      {
      return 0;
      }

    // Update progress
    // We are done reading one file, let's shout it loud:
    assert( params->reader->GetDebug() == 0 );
    const double progress = params->reader->GetProgress(); // other thread might have updated it also...
    if( params->threadid == 0 )
      {
      // IMPLEMENTATION NOTE (WARNING)
      // I think this is ok to assume that thread are equally distributed and the progress of thread 0
      // actually represent nthreads times the local progress...
      params->reader->UpdateProgress( progress + params->nthreads*progressdelta );
      }
    // BUG:
    //const double shift = params->reader->GetShift();
    //const double scale = params->reader->GetScale();
    // This is NOT safe to assume that shift/scale is constant thoughout the Series, this is better to
    // read the shift/scale from the image
    const gdcm::Image &image = reader.GetImage();

    const double shift = image.GetIntercept();
    const double scale = image.GetSlope();

    unsigned long len = image.GetBufferLength();
    // When not applying a transform:
    // len -> sizeof stored image
    // params->len sizeof world value image (after transform)
    if( shift == 1 && scale == 0 )
      assert( len == params->len ); // that would be very bad

    char * pointer = params->scalarpointer;
    //memcpy(pointer + file*len, tempimage, len);
    // image
    char *tempimage = pointer + file*params->len;
    image.GetBuffer(tempimage);
    // overlay
    size_t numoverlays = image.GetNumberOfOverlays();
    //if( numoverlays && !params->reader->GetLoadOverlays() )
    //params->reader->SetNumberOfOverlays( numoverlays );
    if( numoverlays )
      {
      const gdcm::Overlay& ov = image.GetOverlay();
      char * overlaypointer = params->overlayscalarpointer;
      char *tempimage2 = overlaypointer + file*params->overlaylen;
      memset(tempimage2,0,params->overlaylen);
      assert( (unsigned long)ov.GetRows()*ov.GetColumns() <= params->overlaylen );
      if( !ov.GetUnpackBuffer(tempimage2, params->overlaylen) )
        {
        vtkGenericWarningMacro( "Problem in GetUnpackBuffer" );
        }
      }
    //if( params->reader->GetShift() != 1 || params->reader->GetScale() != 0 )
    if( params->reader->GetUseShiftScale() && (shift != 1 || scale != 0) )
      {
      const int shift_int = (int)shift;
      const int scale_int = (int)scale;
      if( scale == 1 && shift == (double)shift_int )
        {
        unsigned short *out = (unsigned short*)(pointer + file * params->len);
        unsigned short *pout = out;
        for( ; pout != out + params->len / sizeof(unsigned short); ++pout )
          {
          *pout = (unsigned short)(*pout + (short)shift);
          }
        }
      else if ( shift == 0 && scale != (double)scale_int )
        {
        // FIXME TODO tempimage stored the DICOM image at the beginning of the buffer,
        // we could avoid duplicating the memory by iterating over the buffer starting
        // from the end and filling out the target buffer by the end...
        // scale is a float !!
        char * duplicate = new char[len];
        memcpy(duplicate,tempimage,len);
        const unsigned short *in = (unsigned short*)duplicate;
        const unsigned short *pin = in;
        float *out = (float*)(pointer + file * params->len);
        float *pout = out;
        for( ; pout != out + params->len / sizeof(float); ++pout )
          {
          // scale is a double, but DICOM specify 32bits for floating point value
          *pout = (float)((double)*pin * (float)scale);
          ++pin;
          }
        //assert( pin == in + len / sizeof(unsigned short) );
        delete[] duplicate;
        }
      else
        {
        //assert( 0 && "Not Implemented" );
        vtkGenericWarningMacro( "Not Implemented" );
        }
      }
    }

  return voidparams;
}

void ShowFilenames(const threadparams &params)
{
  std::cout << "start" << std::endl;
  for(unsigned int i = 0; i < params.nfiles; ++i)
    {
    const char *filename = params.filenames[i];
    std::cout << filename << std::endl;
    }
  std::cout << "end" << std::endl;
}

//----------------------------------------------------------------------------
void vtkGDCMThreadedImageReader::ReadFiles(unsigned int nfiles, const char *filenames[])
{
  // image data:
  vtkImageData *output = this->GetOutput(0);
  assert( output->GetNumberOfPoints() % nfiles == 0 );
  const unsigned long len = output->GetNumberOfPoints() * output->GetScalarSize() / nfiles;
  const unsigned long overlaylen = output->GetNumberOfPoints() / nfiles;
  char * scalarpointer = static_cast<char*>(output->GetScalarPointer());
  // overlay data:
  char * overlayscalarpointer = 0;
  if( this->LoadOverlays )
    {
    vtkImageData *overlayoutput = this->GetOutput(OverlayPortNumber);
#if (VTK_MAJOR_VERSION >= 6)
    // allocation is done in RequestData
#else
    overlayoutput->SetScalarTypeToUnsignedChar();
    overlayoutput->AllocateScalars();
#endif
    overlayscalarpointer = static_cast<char*>(overlayoutput->GetScalarPointer());
    }

#ifdef _WIN32
  // mingw
  SYSTEM_INFO info;
  GetSystemInfo (&info);
  const unsigned int nprocs = info.dwNumberOfProcessors;
#else
#ifdef _SC_NPROCESSORS_ONLN
  const unsigned int nprocs = (unsigned int)sysconf( _SC_NPROCESSORS_ONLN );
#else
#ifdef __APPLE__
  int count = 1;
  size_t size = sizeof(count);
  int res = sysctlbyname("hw.ncpu",&count,&size,NULL,0);
  if( res == -1 )
    {
    count = 1;
    }
  const unsigned int nprocs = (unsigned int)count;
#endif // __APPLE__
#endif // _SC_NPROCESSORS_ONLN
#endif // _WIN32
  const unsigned int nthreads = std::min( nprocs, nfiles );
  threadparams *params = new threadparams[nthreads];

  pthread_mutex_t lock;
  pthread_mutex_init(&lock, NULL);

  pthread_t *pthread = new pthread_t[nthreads];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setscope(&attr, PTHREAD_SCOPE_PROCESS);
  this->Debug = 0;

  // There is nfiles, and nThreads
  assert( nfiles >= nthreads );
  const unsigned int partition = nfiles / nthreads;
  assert( partition );
  for (unsigned int thread=0; thread < nthreads; ++thread)
    {
    params[thread].filenames = filenames + thread * partition;
    params[thread].nfiles = partition;
    if( thread == nthreads - 1 )
      {
      // There is slightly more files to process in this thread:
      params[thread].nfiles += nfiles % nthreads;
      }
    assert( thread * partition < nfiles );
    //ShowFilenames(params[thread]);
    params[thread].scalarpointer = scalarpointer + thread * partition * len;
    params[thread].overlayscalarpointer = overlayscalarpointer + thread * partition * len;
    params[thread].len = len;
    params[thread].overlaylen = overlaylen;
    params[thread].totalfiles = nfiles;
    params[thread].threadid = thread;
    params[thread].nthreads = nthreads;
    params[thread].lock = lock;
    assert( this->Debug == 0 );
    params[thread].reader = this;
    assert( params[thread].reader->Debug == 0 );
    // start thread:
    //int res = pthread_create( &pthread[thread], NULL, ReadFilesThread, &params[thread]);
    int res = pthread_create( &pthread[thread], &attr, ReadFilesThread, &params[thread]);
    if( res )
      {
      std::cerr << "Unable to start a new thread, pthread returned: " << res << std::endl;
      assert(0);
      }
    }
// DEBUG
  unsigned int total = 0;
  for (unsigned int thread=0; thread < nthreads; ++thread)
    {
    total += params[thread].nfiles;
    }
  assert( total == nfiles );
// END DEBUG

  for (unsigned int thread=0;thread<nthreads;thread++)
    {
    pthread_join( pthread[thread], NULL);
    }
  delete[] pthread;

  pthread_mutex_destroy(&lock);
  delete[] params;

#if 0
  // For some reason writing down the file is painfully slow...
  vtkStructuredPointsWriter *writer = vtkStructuredPointsWriter::New();
  writer->SetInput( output );
  writer->SetFileName( "/tmp/threadgdcm.vtk" );
  writer->SetFileTypeToBinary();
  //writer->Write();
  writer->Delete();
#endif

  //output->Print( std::cout );
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
int vtkGDCMThreadedImageReader::RequestData(vtkInformation *vtkNotUsed(request),
                                vtkInformationVector **vtkNotUsed(inputVector),
                                vtkInformationVector *outputVector)
{
  //this->UpdateProgress(0.2);

  // Make sure the output dimension is OK, and allocate its scalars
  for(int i = 0; i < this->GetNumberOfOutputPorts(); ++i)
    {
#if (VTK_MAJOR_VERSION >= 6)
    vtkInformation* outInfo = outputVector->GetInformationObject(i);
    vtkImageData *data = static_cast<vtkImageData *>(outInfo->Get(vtkDataObject::DATA_OBJECT()));
    // Make sure that this output is an image
    if (data)
      {
      int extent[6];
      outInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent);
      this->AllocateOutputData(data, outInfo, extent);
      }
#else
    (void)outputVector;
    // Copy/paste from vtkImageAlgorithm::AllocateScalars. Cf. "this needs to be fixed -Ken"
    vtkStreamingDemandDrivenPipeline *sddp =
      vtkStreamingDemandDrivenPipeline::SafeDownCast(this->GetExecutive());
    if (sddp)
      {
      int extent[6];
      sddp->GetOutputInformation(i)->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),extent);
      this->GetOutput(i)->SetExtent(extent);
      }
    this->GetOutput(i)->AllocateScalars();
#endif
    }
  RequestDataCompat();
  return 1;
}
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/

void vtkGDCMThreadedImageReader::RequestDataCompat()
{
  int *dext = this->GetDataExtent();
  if( this->FileNames )
    {
    // Make sure that each file is single slice
    assert( dext[5] - dext[4] == this->FileNames->GetNumberOfValues() - 1 ); (void)dext;
    const vtkIdType nfiles = this->FileNames->GetNumberOfValues();
    const char **filenames = new const char* [ nfiles ];
    for(unsigned int i = 0; i < nfiles; ++i)
      {
      filenames[i] = this->FileNames->GetValue( i );
      //std::cerr << filenames[i] << std::endl;
      }
    ReadFiles((unsigned int)nfiles, filenames);
    delete[] filenames;
    }
  else if( this->FileName )
    {
    // File can be a volume
    const char *filename = this->FileName;
    ReadFiles(1, &filename);
    }
  else
    {
    // Impossible case since ExecuteInformation would have failed earlier...
    assert( 0 && "Impossible happen" );
    }

}

//----------------------------------------------------------------------------
void vtkGDCMThreadedImageReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
