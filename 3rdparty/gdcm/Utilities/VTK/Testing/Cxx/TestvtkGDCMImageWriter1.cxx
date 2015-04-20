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

#include "vtkImageData.h"
#include "vtkMultiThreader.h"
#include "vtkMedicalImageProperties.h"

#include "gdcmTesting.h"
#include "gdcmFilename.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmImageReader.h"

#ifndef vtkFloatingPointType
#define vtkFloatingPointType float
#endif

int TestvtkGDCMImageWrite(const char *filename, bool verbose = false)
{
  int res = 0; // no error
  if( verbose )
    std::cerr << "Reading : " << filename << std::endl;
  vtkGDCMImageReader *reader = vtkGDCMImageReader::New();
  reader->FileLowerLeftOn();
  int canread = reader->CanReadFile( filename );
  if( canread )
    {
    reader->SetFileName( filename );
    reader->Update();
    if( verbose )
      {
      reader->GetOutput()->Print( cout );
      reader->GetMedicalImageProperties()->Print( cout );
      }

    // Create directory first:
    const char subdir[] = "TestvtkGDCMImageWriter";
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
    writer->SetPlanarConfiguration( reader->GetPlanarConfiguration() );
    writer->SetShift( reader->GetShift() );
    writer->SetScale( reader->GetScale() );
    writer->SetFileName( gdcmfile.c_str() );
    writer->Write();
    if( verbose )  std::cerr << "Write out: " << gdcmfile << std::endl;

    writer->Delete();

    // Need to check we can still read this image back:
    gdcm::ImageReader r;
    if( gdcm::System::FileExists( gdcmfile.c_str() ) )
      {
      r.SetFileName( gdcmfile.c_str() );
      }
    if( !r.Read() )
      {
      std::cerr << "failed to read back:" << gdcmfile << std::endl;
      res = 1;
      }
    else
      {
      // ok could read the file, now check origin is ok:
      const gdcm::Image &image = r.GetImage();
      const double *origin = image.GetOrigin();
      if( origin )
        {
        vtkImageData * vtkimg = reader->GetOutput();
        const vtkFloatingPointType *vtkorigin = vtkimg->GetOrigin();
        if( fabs(vtkorigin[0] - origin[0]) > 1.e-3
          || fabs(vtkorigin[1] - origin[1]) > 1.e-3
          || fabs(vtkorigin[2] - origin[2]) > 1.e-3 )
          {
          std::cerr << "Problem:" << vtkorigin[0] << "," << vtkorigin[1] << "," << vtkorigin[2] ;
          std::cerr << " should be:" << origin[0] << "," << origin[1] << "," << origin[2] << std::endl ;
          std::cerr << filename << std::endl;
          res = 1;
          }
        }

      gdcm::ImageReader r2;
      r2.SetFileName( filename );
      if( !r2.Read() )
        {
        std::cerr << "failed to re-read initial image...how is that possible ?:" << filename << std::endl;
        res = 1;
        }
      const gdcm::Image &compimage = r2.GetImage();
      // Check that Media Storage is still correct:
      // Well this is difficult to implement as Retired class are replaced with newer one automatically
      gdcm::MediaStorage ms1;
      ms1.SetFromFile( r.GetFile() ); // our rewritten file
      gdcm::MediaStorage ms2;
      ms2.SetFromFile( r2.GetFile() ); // original file
      if( ms1 != ms2 )
        {
        if( ms1 == gdcm::MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage )
          {
          // Hum I have this weird case when reading libido1.0-vol.acr...
          }
        else if( ms1 == gdcm::MediaStorage::XRayAngiographicImageStorage && ms2 == gdcm::MediaStorage::SecondaryCaptureImageStorage  )
          {
          // FIXME: D_CLUNIE_XA1_JPLL.dcm
          }
        else if( ms2 == gdcm::MediaStorage::XRayRadiofluoroscopingImageStorage )
          {
          // gdcmData/JDDICOM_Sample5.dcm
          }
        else if( ms1 == gdcm::MediaStorage::EnhancedMRImageStorage && ms2 == gdcm::MediaStorage::MRImageStorage )
          {
          // gdcmData/MR-MONO2-8-16x-heart.dcm
          }
        else if( ms1 == gdcm::MediaStorage::UltrasoundImageStorage && ms2 == gdcm::MediaStorage::UltrasoundImageStorageRetired )
          {
          // gdcmData/US-RGB-8-esopecho.dcm
          }
        else if( ms1 == gdcm::MediaStorage::UltrasoundMultiFrameImageStorage && ms2 == gdcm::MediaStorage::UltrasoundMultiFrameImageStorageRetired )
          {
          // gdcmData/US-MONO2-8-8x-execho.dcm
          }
        else if ( ms1 == gdcm::MediaStorage::NuclearMedicineImageStorage && ms2 == gdcm::MediaStorage::SecondaryCaptureImageStorage )
          {
          // gdcmData/Renal_Flow.dcm
          }
        else if ( ms1 == gdcm::MediaStorage::CTImageStorage && ms2 == gdcm::MediaStorage::SecondaryCaptureImageStorage )
          {
          // gdcmData/D_CLUNIE_SC1_JPLY.dcm
          }
        else if ( ms1 == gdcm::MediaStorage::EnhancedCTImageStorage && ms2 == gdcm::MediaStorage::CTImageStorage && compimage.GetNumberOfDimensions() == 3 )
          {
          // gdcmData/CroppedArm.dcm
          }
        else if( ms1 == gdcm::MediaStorage::MRImageStorage && ms2 == gdcm::MediaStorage::GeneralElectricMagneticResonanceImageStorage )
          {
          // gdcmData/MR00010001.dcm
          }
        else if( ms1 == gdcm::MediaStorage::UltrasoundImageStorage && ms2 == gdcm::MediaStorage::SecondaryCaptureImageStorage )
          {
          // gdcmData/GE_LOGIQBook-8-RGB-HugePreview.dcm
          }
        else if( ms1 == gdcm::MediaStorage::DigitalXRayImageStorageForProcessing
          && ms2 == gdcm::MediaStorage::DigitalXRayImageStorageForPresentation
        )
          {
          // gdcmData/DX_GE_FALCON_SNOWY-VOI.dcm
          }
        else
          {
          std::cerr << "MediaStorage incompatible: " << ms1 << " vs " << ms2 << " for file: " << filename << std::endl;
          res = 1;
          }
        }
      // Make sure that md5 is still ok:
      unsigned long len = image.GetBufferLength();
      char* buffer = new char[len];
      bool res2 = image.GetBuffer(buffer);
      if( !res2 )
        {
        std::cerr << "Could not get buffer" << std::endl;
        res = 1;
        }
      const char *ref = gdcm::Testing::GetMD5FromFile(filename);
      char digest[33];
      gdcm::Testing::ComputeMD5(buffer, len, digest);
      if( !ref )
        {
        std::cerr << "Could not compute md5" << std::endl;
        res = 1;
        }
      const gdcm::PixelFormat &comppf = compimage.GetPixelFormat();
      if( !ref )
        {
        std::cerr << "Missing md5: " << digest << std::endl;
        }
      else if( strcmp(digest, ref) != 0
        // I do not support rewritting 12Bits pack image (illegal anyway)
        && comppf != gdcm::PixelFormat::UINT12
      )
        {
#if 0
{
unsigned long len = compimage.GetBufferLength();
char* buffer = new char[len];
bool res2 = compimage.GetBuffer(buffer);
std::ofstream out("/tmp/debug.raw", std::ios::binary);
out.write( buffer, len );
out.close();
}
#endif
        std::cerr << "Problem reading image from: " << filename << std::endl;
        std::cerr << "Found " << digest << " instead of " << ref << std::endl;
        std::cerr << "Original TransferSyntax was: " << compimage.GetTransferSyntax() << std::endl;
        std::cerr << "Output image: " << gdcmfile << std::endl;
        res = 1;
        }
      delete[] buffer;
      }
    }
  else
    {
    if( verbose )
      std::cerr << "vtkGDCMImageReader cannot read: " << filename << std::endl;
    }
  reader->Delete();

  return res;
}

int TestvtkGDCMImageWriter1(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestvtkGDCMImageWrite(filename, true);
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
    r += TestvtkGDCMImageWrite( filename );
    ++i;
    }

  return r;
}
