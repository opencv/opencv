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

#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkErrorCode.h"
#include "vtkMath.h"
#include "vtkPolyData.h"
#include "vtkCellArray.h"
#include "vtkPoints.h"
#include "vtkMedicalImageProperties.h"
#include "vtkGDCMMedicalImageProperties.h"
#include "vtkStringArray.h"
#include "vtkPointData.h"
#include "vtkLookupTable.h"
#include "vtkWindowLevelLookupTable.h"
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#include "vtkLookupTable16.h"
#include "vtkInformationVector.h"
#include "vtkInformation.h"
#include "vtkDemandDrivenPipeline.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/
#include "vtkMatrix4x4.h"
#include "vtkUnsignedCharArray.h"
//#include "vtkUnsignedShortArray.h"
#include "vtkBitArray.h"

#include "gdcmImageReader.h"
#include "gdcmDataElement.h"
#include "gdcmByteValue.h"
#include "gdcmSwapper.h"
#include "gdcmUnpacker12Bits.h"
#include "gdcmRescaler.h"
#include "gdcmOrientation.h"
#include "gdcmTrace.h"
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmDirectoryHelper.h"

#include <sstream>

vtkCxxRevisionMacro(vtkGDCMImageReader, "$Revision: 1.1 $")
vtkStandardNewMacro(vtkGDCMImageReader)

static inline bool vtkGDCMImageReader_IsCharTypeSigned()
{
#ifndef VTK_TYPE_CHAR_IS_SIGNED
  unsigned char uc = 255;
  return (*reinterpret_cast<char*>(&uc) < 0) ? true : false;
#else
  return VTK_TYPE_CHAR_IS_SIGNED;
#endif
}

// Output Ports are as follow:
// #0: The image/volume (root PixelData element)
// #1: (if present): the Icon Image (0088,0200)
// #2-xx: (if present): the Overlay (60xx,3000)

#define ICONIMAGEPORTNUMBER 1
#define OVERLAYPORTNUMBER   2

vtkCxxSetObjectMacro(vtkGDCMImageReader,Curve,vtkPolyData)
vtkCxxSetObjectMacro(vtkGDCMImageReader,MedicalImageProperties,vtkMedicalImageProperties)

//----------------------------------------------------------------------------
vtkGDCMImageReader::vtkGDCMImageReader()
{
  // vtkDataArray has an internal vtkLookupTable why not used it ?
  // vtkMedicalImageProperties is in the parent class
  //this->FileLowerLeft = 1;
  this->DirectionCosines = vtkMatrix4x4::New();
  this->DirectionCosines->Identity();
  //this->DirectionCosines->SetElement(0,0,1); // x0
  //this->DirectionCosines->SetElement(1,0,0); // x1
  //this->DirectionCosines->SetElement(2,0,0); // x2
  //this->DirectionCosines->SetElement(3,0,0); //
  //this->DirectionCosines->SetElement(0,1,0); // y0
  //this->DirectionCosines->SetElement(1,1,1); // y1
  //this->DirectionCosines->SetElement(2,1,0); // y2
  //this->DirectionCosines->SetElement(3,1,0); //
  //this->DirectionCosines->SetElement(0,2,0); // y0
  //this->DirectionCosines->SetElement(1,2,0); // y1
  //this->DirectionCosines->SetElement(2,2,1); // y2
  //this->DirectionCosines->SetElement(3,2,0); //
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#else
  this->MedicalImageProperties = vtkMedicalImageProperties::New();
#endif
#if (VTK_MAJOR_VERSION > 5) || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
  //this->SetNumberOfInputPorts(0);
#else
  this->FileNames = NULL; //vtkStringArray::New();
#endif
  this->LoadOverlays = 1;
  this->LoadIconImage = 1;
  this->NumberOfOverlays = 0;
  this->NumberOfIconImages = 0;
  memset(this->IconImageDataExtent,0,6*sizeof(int));
  this->ImageFormat = 0; // INVALID
  this->ApplyInverseVideo = 0;
  this->ApplyLookupTable = 0;
  this->ApplyYBRToRGB = 0;
  this->ApplyPlanarConfiguration = 1;
  this->ApplyShiftScale = 1;
  memset(this->ImagePositionPatient,0,3*sizeof(double));
  memset(this->ImageOrientationPatient,0,6*sizeof(double));
  this->Curve = 0;
  this->Shift = 0.;
  this->Scale = 1.;
  this->IconDataScalarType = VTK_CHAR;
  this->IconNumberOfScalarComponents = 1;
  this->PlanarConfiguration = 0;
  this->LossyFlag = 0;

  // DirectionCosine was added after 5.2
#if ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 2 )
  this->MedicalImageProperties->SetDirectionCosine(1,0,0,0,1,0);
#endif
  this->SetImageOrientationPatient(1,0,0,0,1,0);

  //  this->SetMedicalImageProperties( vtkGDCMMedicalImageProperties::New() );
  this->ForceRescale = 0;
}

//----------------------------------------------------------------------------
vtkGDCMImageReader::~vtkGDCMImageReader()
{
  //delete this->Internals;
  this->DirectionCosines->Delete();
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#else
  this->MedicalImageProperties->Delete();
#endif
#if (VTK_MAJOR_VERSION > 5) || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
#else
  if( this->FileNames )
    {
    this->FileNames->Delete();
    }
#endif
  if( this->Curve )
    {
    this->Curve->Delete();
    }
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION > 5) || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
#else
void vtkGDCMImageReader::SetFileNames(vtkStringArray *filenames)
{
  if (filenames == this->FileNames)
    {
    return;
    }
  if (this->FileNames)
    {
    this->FileNames->Delete();
    this->FileNames = 0;
    }
  if (filenames)
    {
    this->FileNames = filenames;
    this->FileNames->Register(this);
    if (this->FileNames->GetNumberOfValues() > 0)
      {
      this->DataExtent[4] = 0;
      this->DataExtent[5] = this->FileNames->GetNumberOfValues() - 1;
      }
    if (this->FilePrefix)
      {
      delete [] this->FilePrefix;
      this->FilePrefix = NULL;
      }
    if (this->FileName)
      {
      delete [] this->FileName;
      this->FileName = NULL;
      }
    }

  this->Modified();
}
#endif

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#else
void vtkGDCMImageReader::ExecuteInformation()
{
  //std::cerr << "ExecuteInformation" << std::endl;

  // FIXME: I think it only apply to VTK 4.2...
  vtkImageData *output = this->GetOutput();
  output->SetUpdateExtentToWholeExtent(); // pipeline is not reexecuting properly without that...

  int res = RequestInformationCompat();
  if( !res )
    {
    vtkErrorMacro( "ExecuteInformation failed" );
    this->SetErrorCode(vtkErrorCode::FileFormatError);
    return;
    }

  int numvol = 1;
  if( this->LoadIconImage )
    {
    numvol = 2;
    }
  if( this->LoadOverlays )
    {
    // If not icon found, we still need to be associated to port #2:
    numvol = 3;
    }
  this->SetNumberOfOutputs(numvol);

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
#ifdef GDCMV2_0_COMPATIBILITY
      output->SetOrigin(this->DataOrigin);
#endif

      output->SetScalarType(this->DataScalarType);
      output->SetNumberOfScalarComponents(this->NumberOfScalarComponents);
      break;
    case ICONIMAGEPORTNUMBER:
      output->SetWholeExtent(this->IconImageDataExtent);
      output->SetScalarType( this->IconDataScalarType );
      output->SetNumberOfScalarComponents( this->IconNumberOfScalarComponents );
      break;
    //case OVERLAYPORTNUMBER:
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

//----------------------------------------------------------------------------
void vtkGDCMImageReader::ExecuteData(vtkDataObject *output)
{
  //std::cerr << "ExecuteData" << std::endl;
  // In VTK 4.2 AllocateOutputData is reexecuting ExecuteInformation which is bad !
  //vtkImageData *data = this->AllocateOutputData(output);
  vtkImageData *res = vtkImageData::SafeDownCast(output);
  res->SetExtent(res->GetUpdateExtent());
  res->AllocateScalars();

  if( this->LoadIconImage )
    {
    vtkImageData *res = vtkImageData::SafeDownCast(this->Outputs[ICONIMAGEPORTNUMBER]);
    res->SetUpdateExtentToWholeExtent();

    res->SetExtent(res->GetUpdateExtent());
    res->AllocateScalars();
    }
  if( this->LoadOverlays )
    {
    vtkImageData *res = vtkImageData::SafeDownCast(this->Outputs[OVERLAYPORTNUMBER]);
    res->SetUpdateExtentToWholeExtent();

    res->SetExtent(res->GetUpdateExtent());
    res->AllocateScalars();
    }
  //int * updateExtent = data->GetUpdateExtent();
  //std::cout << "UpdateExtent:" << updateExtent[4] << " " << updateExtent[5] << std::endl;
  RequestDataCompat();

}

#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/

//----------------------------------------------------------------------------
int vtkGDCMImageReader::CanReadFile(const char* fname)
{
  gdcm::ImageReader reader;
  reader.SetFileName( fname );
  if( !reader.Read() )
    {
    return 0;
    }
  // 3 means: I might be able to read...
  return 3;
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
int vtkGDCMImageReader::ProcessRequest(vtkInformation* request,
                                 vtkInformationVector** inputVector,
                                 vtkInformationVector* outputVector)
{
  // generate the data
  if(request->Has(vtkDemandDrivenPipeline::REQUEST_DATA()))
    {
    return this->RequestData(request, inputVector, outputVector);
    }

  // execute information
  if(request->Has(vtkDemandDrivenPipeline::REQUEST_INFORMATION()))
    {
    return this->RequestInformation(request, inputVector, outputVector);
    }

  return this->Superclass::ProcessRequest(request, inputVector, outputVector);
}
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/




//----------------------------------------------------------------------------
void vtkGDCMImageReader::FillMedicalImageInformation(const gdcm::ImageReader &reader)
{
  const gdcm::File &file = reader.GetFile();
  const gdcm::DataSet &ds = file.GetDataSet();

  // $ grep "vtkSetString\|DICOM" vtkMedicalImageProperties.h
  // For ex: DICOM (0010,0010) = DOE,JOHN
  this->MedicalImageProperties->SetPatientName( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0010,0x0010), ds).c_str() );
  // For ex: DICOM (0010,0020) = 1933197
  this->MedicalImageProperties->SetPatientID( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0010,0x0020), ds).c_str() );
  // For ex: DICOM (0010,1010) = 031Y
  this->MedicalImageProperties->SetPatientAge( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0010,0x1010), ds).c_str() );
  // For ex: DICOM (0010,0040) = M
  this->MedicalImageProperties->SetPatientSex( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0010,0x0040), ds).c_str() );
  // For ex: DICOM (0010,0030) = 19680427
  this->MedicalImageProperties->SetPatientBirthDate( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0010,0x0030), ds).c_str() );
#if (VTK_MAJOR_VERSION > 5) || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
  // For ex: DICOM (0008,0020) = 20030617
  this->MedicalImageProperties->SetStudyDate( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0020), ds).c_str() );
#endif
  // For ex: DICOM (0008,0022) = 20030617
  this->MedicalImageProperties->SetAcquisitionDate( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0022), ds).c_str() );
#if (VTK_MAJOR_VERSION > 5) || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
  // For ex: DICOM (0008,0030) = 162552.0705 or 230012, or 0012
  this->MedicalImageProperties->SetStudyTime( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0030), ds).c_str() );
#endif
  // For ex: DICOM (0008,0032) = 162552.0705 or 230012, or 0012
  this->MedicalImageProperties->SetAcquisitionTime( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0032), ds).c_str() );
  // For ex: DICOM (0008,0023) = 20030617
  this->MedicalImageProperties->SetImageDate( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0023), ds).c_str() );
  // For ex: DICOM (0008,0033) = 162552.0705 or 230012, or 0012
  this->MedicalImageProperties->SetImageTime( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0033), ds).c_str() );
  // For ex: DICOM (0020,0013) = 1
  this->MedicalImageProperties->SetImageNumber( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0020,0x0013), ds).c_str() );
  // For ex: DICOM (0020,0011) = 902
  this->MedicalImageProperties->SetSeriesNumber( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0020,0x0011), ds).c_str() );
  // For ex: DICOM (0008,103e) = SCOUT
  this->MedicalImageProperties->SetSeriesDescription( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x103e), ds).c_str() );
  // For ex: DICOM (0020,0010) = 37481
  this->MedicalImageProperties->SetStudyID( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0020,0x0010), ds).c_str() );
  // For ex: DICOM (0008,1030) = BRAIN/C-SP/FACIAL
  this->MedicalImageProperties->SetStudyDescription( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x1030), ds).c_str() );
  // For ex: DICOM (0008,0060)= CT
  this->MedicalImageProperties->SetModality( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0060), ds).c_str() );
  // For ex: DICOM (0008,0070) = Siemens
  this->MedicalImageProperties->SetManufacturer( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0070), ds).c_str() );
  // For ex: DICOM (0008,1090) = LightSpeed QX/i
  this->MedicalImageProperties->SetManufacturerModelName( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x1090), ds).c_str() );
  // For ex: DICOM (0008,1010) = LSPD_OC8
  this->MedicalImageProperties->SetStationName( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x1010), ds).c_str() );
  // For ex: DICOM (0008,0080) = FooCity Medical Center
  this->MedicalImageProperties->SetInstitutionName( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0008,0x0080), ds).c_str() );
  // For ex: DICOM (0018,1210) = Bone
  this->MedicalImageProperties->SetConvolutionKernel( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x1210), ds).c_str() );
  // For ex: DICOM (0018,0050) = 0.273438
  this->MedicalImageProperties->SetSliceThickness( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x0050), ds).c_str() );
  // For ex: DICOM (0018,0060) = 120
  this->MedicalImageProperties->SetKVP( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x0060), ds).c_str() );
  // For ex: DICOM (0018,1120) = 15
  this->MedicalImageProperties->SetGantryTilt( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x1120), ds).c_str() );
  // For ex: DICOM (0018,0081) = 105
  this->MedicalImageProperties->SetEchoTime( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x0081), ds).c_str() );
  // For ex: DICOM (0018,0091) = 35
  this->MedicalImageProperties->SetEchoTrainLength( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x0091), ds).c_str() );
  // For ex: DICOM (0018,0080) = 2040
  this->MedicalImageProperties->SetRepetitionTime( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x0080), ds).c_str() );
  // For ex: DICOM (0018,1150) = 5
  this->MedicalImageProperties->SetExposureTime( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x1150), ds).c_str() );
  // For ex: DICOM (0018,1151) = 400
  this->MedicalImageProperties->SetXRayTubeCurrent( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x1151), ds).c_str() );
  // For ex: DICOM (0018,1152) = 114
  this->MedicalImageProperties->SetExposure( gdcm::DirectoryHelper::GetStringValueFromTag( gdcm::Tag(0x0018,0x1152), ds).c_str() );

  // virtual void AddWindowLevelPreset(double w, double l);
  // (0028,1050) DS [   498\  498]                           #  12, 2 WindowCenter
  // (0028,1051) DS [  1063\ 1063]                           #  12, 2 WindowWidth
  gdcm::Tag twindowcenter(0x0028,0x1050);
  gdcm::Tag twindowwidth(0x0028,0x1051);
  if( ds.FindDataElement( twindowcenter ) && ds.FindDataElement( twindowwidth) )
    {
    const gdcm::DataElement& windowcenter = ds.GetDataElement( twindowcenter );
    const gdcm::DataElement& windowwidth = ds.GetDataElement( twindowwidth );
    const gdcm::ByteValue *bvwc = windowcenter.GetByteValue();
    const gdcm::ByteValue *bvww = windowwidth.GetByteValue();
    if( bvwc && bvww ) // Can be Type 2
      {
      //gdcm::Attributes<0x0028,0x1050> at;
      gdcm::Element<gdcm::VR::DS,gdcm::VM::VM1_n> elwc;
      std::stringstream ss1;
      std::string swc = std::string( bvwc->GetPointer(), bvwc->GetLength() );
      ss1.str( swc );
      gdcm::VR vr = gdcm::VR::DS;
      unsigned int vrsize = vr.GetSizeof();
      unsigned int count = gdcm::VM::GetNumberOfElementsFromArray(swc.c_str(), (unsigned int)swc.size());
      elwc.SetLength( count * vrsize );
      elwc.Read( ss1 );
      std::stringstream ss2;
      std::string sww = std::string( bvww->GetPointer(), bvww->GetLength() );
      ss2.str( sww );
      gdcm::Element<gdcm::VR::DS,gdcm::VM::VM1_n> elww;
      elww.SetLength( count * vrsize );
      elww.Read( ss2 );
      //assert( elww.GetLength() == elwc.GetLength() );
      for(unsigned int i = 0; i < elwc.GetLength(); ++i)
        {
        this->MedicalImageProperties->AddWindowLevelPreset( elww.GetValue(i), elwc.GetValue(i) );
        }
      }
    }
  gdcm::Tag twindowexplanation(0x0028,0x1055);
  if( ds.FindDataElement( twindowexplanation ) )
    {
    const gdcm::DataElement& windowexplanation = ds.GetDataElement( twindowexplanation );
    const gdcm::ByteValue *bvwe = windowexplanation.GetByteValue();
    if( bvwe ) // Can be Type 2
      {
      unsigned int n = this->MedicalImageProperties->GetNumberOfWindowLevelPresets();
      gdcm::Element<gdcm::VR::LO,gdcm::VM::VM1_n> elwe; // window explanation
      gdcm::VR vr = gdcm::VR::LO;
      std::stringstream ss;
      ss.str( "" );
      std::string swe = std::string( bvwe->GetPointer(), bvwe->GetLength() );
      unsigned int count = gdcm::VM::GetNumberOfElementsFromArray(swe.c_str(), (unsigned int)swe.size()); (void)count;
      // I found a case with only one W/L but two comments: WINDOW1\WINDOW2
      // SIEMENS-IncompletePixelData.dcm
      // oh wait but what if we have the countrary...
      //assert( count >= (unsigned int)n );
      elwe.SetLength( count * vr.GetSizeof() );
      ss.str( swe );
      elwe.Read( ss );
      unsigned int c = std::min(n, count);
      for(unsigned int i = 0; i < c; ++i)
        {
        this->MedicalImageProperties->SetNthWindowLevelPresetComment(i, elwe.GetValue(i).c_str() );
        }
      }
    }

#if 0
  // gdcmData/JDDICOM_Sample4.dcm
  // -> (0008,0060) CS [DM  Digital microscopy]                 #  24, 1 Modality
  gdcm::MediaStorage ms1 = gdcm::MediaStorage::SecondaryCaptureImageStorage;
  ms1.GuessFromModality( this->MedicalImageProperties->GetModality(), this->FileDimensionality );
  gdcm::MediaStorage ms2;
  ms2.SetFromFile( reader.GetFile() );
  if( ms2 != ms1 && ms2 != gdcm::MediaStorage::SecondaryCaptureImageStorage )
    {
    vtkWarningMacro( "SHOULD NOT HAPPEN. Unrecognized Modality: " << this->MedicalImageProperties->GetModality()
      << " Will be set instead to the known one: " << ms2.GetModality() )
    this->MedicalImageProperties->SetModality( ms2.GetModality() );
    }
#endif

  // Add more info:
  vtkGDCMMedicalImageProperties *gdcmmip =
    dynamic_cast<vtkGDCMMedicalImageProperties*>( this->MedicalImageProperties );
  if( gdcmmip )
    {
    gdcmmip->PushBackFile( file );
    }
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
int vtkGDCMImageReader::RequestInformation(vtkInformation *request,
                                      vtkInformationVector **inputVector,
                                      vtkInformationVector *outputVector)
{
  (void)request;(void)inputVector;
  int res = RequestInformationCompat();
  if( !res )
    {
    vtkErrorMacro( "RequestInformationCompat failed: " << res );
    this->SetErrorCode(vtkErrorCode::FileFormatError);
    return 0;
    }

  int numvol = 1;
  if( this->LoadIconImage )
    {
    numvol = 2;
    }
  if( this->LoadOverlays )
    {
    // If no icon found, we still need to be associated to port #2:
    numvol = 2 + this->NumberOfOverlays;
    }
  this->SetNumberOfOutputPorts(numvol);
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
#ifdef GDCMV2_0_COMPATIBILITY
      outInfo->Set(vtkDataObject::ORIGIN(), this->DataOrigin, 3);
#endif
      vtkDataObject::SetPointDataActiveScalarInfo(outInfo, this->DataScalarType, this->NumberOfScalarComponents);
      break;
    // Icon Image
    case ICONIMAGEPORTNUMBER:
      outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->IconImageDataExtent, 6);
      vtkDataObject::SetPointDataActiveScalarInfo(outInfo, this->IconDataScalarType, this->IconNumberOfScalarComponents );
      break;
    // Overlays:
    //case OVERLAYPORTNUMBER:
    default:
      outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),
        this->DataExtent[0], this->DataExtent[1],
        this->DataExtent[2], this->DataExtent[3],
        0,0 );
      vtkDataObject::SetPointDataActiveScalarInfo(outInfo, VTK_UNSIGNED_CHAR, 1);
      break;
      }
    }

  //return res;
  return 1;
}
#endif

static gdcm::PixelFormat::ScalarType
ComputePixelTypeFromFiles(const char *inputfilename, vtkStringArray *filenames,
  gdcm::Image const & imageref)
{
  gdcm::PixelFormat::ScalarType outputpt;
  outputpt = gdcm::PixelFormat::UNKNOWN;
  // there is a very subtle bug here. Let's imagine we have a collection of files
  // they can all have different Rescale Slope / Intercept. In this case we should:
  // 1. Make sure to read each Rescale Slope / Intercept individually
  // 2. Make sure to decide which Pixel Type to use using *all* slices:
  if( inputfilename )
    {
    const gdcm::Image &image = imageref;
    const gdcm::PixelFormat &pixeltype = image.GetPixelFormat();
    double shift = image.GetIntercept();
    double scale = image.GetSlope();

    gdcm::Rescaler r;
    r.SetIntercept( shift );
    r.SetSlope( scale );
    r.SetPixelFormat( pixeltype );
    outputpt = r.ComputeInterceptSlopePixelType();
    }
  else if ( filenames && filenames->GetNumberOfValues() > 0 )
    {
    std::set< gdcm::PixelFormat::ScalarType > pixeltypes;
    std::set< unsigned short > samplesperpixel;
    // FIXME a gdcm::Scanner would be much faster here:
    for(int i = 0; i < filenames->GetNumberOfValues(); ++i )
      {
      const char *filename = filenames->GetValue( i );
      gdcm::ImageReader reader;
      reader.SetFileName( filename );
      if( !reader.Read() )
        {
        vtkGenericWarningMacro( "ImageReader failed: " << filename );
        return gdcm::PixelFormat::UNKNOWN;
        }
      const gdcm::Image &image = reader.GetImage();
      const gdcm::PixelFormat &pixeltype = image.GetPixelFormat();
      samplesperpixel.insert( pixeltype.GetSamplesPerPixel() );

      double shift = image.GetIntercept();
      double scale = image.GetSlope();

      gdcm::PixelFormat::ScalarType outputpt2 = pixeltype;
      gdcm::Rescaler r;
      r.SetIntercept( shift );
      r.SetSlope( scale );
      r.SetPixelFormat( pixeltype );
      outputpt2 = r.ComputeInterceptSlopePixelType();
      //std::cout << "Found: " << outputpt << std::endl;
      pixeltypes.insert( outputpt2 );
      }
    if( pixeltypes.size() == 1 )
      {
      assert( samplesperpixel.size() == 1 );
      // Ok easy case
      outputpt = *pixeltypes.begin();
      }
    else if( samplesperpixel.size() == 1 )
      {
      // Hardcoded. If Pixel Type found is the maximum (as of PS 3.5 - 2008)
      // There is nothing bigger that FLOAT64
      if( pixeltypes.count( gdcm::PixelFormat::FLOAT64 ) != 0 )
        {
        outputpt = gdcm::PixelFormat::FLOAT64;
        }
      else
        {
        // should I just take the biggest value ?
        // MM: I am not sure UINT16 and INT16 are really compatible
        // so taking the biggest value might not be the solution
        // In this case we could use INT32, but FLOAT64 also works...
        // oh well, let's just use FLOAT64 always.
        vtkGenericWarningMacro( "This may not always be optimized. Sorry" );
        outputpt = gdcm::PixelFormat::FLOAT64;
        }
      }
    else
      {
      vtkGenericWarningMacro( "Could not compute Pixel Type. Sorry" );
      }
    }
  else
    {
    assert( 0 ); // I do not think this is possible
    }
  //gdcmAssertMacro( outputpt >= pixeltype_ref );
  //(void)pixeltype_ref;

  return outputpt;
}

//----------------------------------------------------------------------------
int vtkGDCMImageReader::RequestInformationCompat()
{
  // FIXME, need to implement the other modes too:
  if( this->ApplyLookupTable || this->ApplyYBRToRGB || this->ApplyInverseVideo )
    {
    vtkErrorMacro( "ApplyLookupTable/ApplyYBRToRGB/ApplyInverseVideo not compatible" );
    return 0;
    }
  // I do not think this is a good idea anyway to let the user decide
  // wether or not she wants *not* to apply shift/scale...
  if( !this->ApplyShiftScale )
  {
    vtkErrorMacro("ApplyShiftScale not compatible" );
    return 0;
   }
  // I do not think this one will ever be implemented:
  if( !this->ApplyPlanarConfiguration )
    {
    vtkErrorMacro("ApplyPlanarConfiguration not compatible" );
    return 0;
    }

  // Let's read the first file :
  const char *filename;
  if( this->FileName )
    {
    filename = this->FileName;
    }
  else if ( this->FileNames && this->FileNames->GetNumberOfValues() > 0 )
    {
    filename = this->FileNames->GetValue( 0 );
    }
  else
    {
    // hey! I need at least one file to schew on !
    vtkErrorMacro( "You did not specify any filenames" );
    return 0;
    }
  gdcm::ImageReader reader;
  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    vtkErrorMacro( "ImageReader failed on " << filename );
    return 0;
    }
  const gdcm::Image &image = reader.GetImage();
  this->LossyFlag = image.IsLossy();
  const unsigned int *dims = image.GetDimensions();

  // Set the Extents.
  assert( image.GetNumberOfDimensions() >= 2 );
  this->DataExtent[0] = 0;
  this->DataExtent[1] = dims[0] - 1;
  this->DataExtent[2] = 0;
  this->DataExtent[3] = dims[1] - 1;
  if( image.GetNumberOfDimensions() == 2 )
    {
    // This is just so much painful to deal with DICOM / VTK
    // they simply assume that number of file is equal to the dimension
    // of the last axe (see vtkImageReader2::SetFileNames )
    if ( this->FileNames && this->FileNames->GetNumberOfValues() > 1 )
      {
      this->DataExtent[4] = 0;
      //this->DataExtent[5] = this->FileNames->GetNumberOfValues() - 1;
      }
    else
      {
      this->DataExtent[4] = 0;
      this->DataExtent[5] = 0;
      }
    }
  else
    {
    assert( image.GetNumberOfDimensions() == 3 );
    this->FileDimensionality = 3;
    this->DataExtent[4] = 0;
    this->DataExtent[5] = dims[2] - 1;
    }
  gdcm::MediaStorage ms;
  ms.SetFromFile( reader.GetFile() );
  assert( gdcm::MediaStorage::IsImage( ms ) || ms == gdcm::MediaStorage::MRSpectroscopyStorage );
  // There is no point in adding world info to a SC object since noone but GDCM can use this info...
  //if( ms != gdcm::MediaStorage::SecondaryCaptureImageStorage )

  const double *spacing = image.GetSpacing();
  if( spacing )
    {
    this->DataSpacing[0] = spacing[0];
    this->DataSpacing[1] = spacing[1];
    this->DataSpacing[2] = image.GetSpacing(2);
    }

  const double *origin = image.GetOrigin();
  if( origin )
    {
    this->ImagePositionPatient[0] = image.GetOrigin(0);
    this->ImagePositionPatient[1] = image.GetOrigin(1);
    this->ImagePositionPatient[2] = image.GetOrigin(2);
    }

  const double *dircos = image.GetDirectionCosines();
  if( dircos )
    {
    this->DirectionCosines->SetElement(0,0, dircos[0]);
    this->DirectionCosines->SetElement(1,0, dircos[1]);
    this->DirectionCosines->SetElement(2,0, dircos[2]);
    this->DirectionCosines->SetElement(3,0, 0);
    this->DirectionCosines->SetElement(0,1, dircos[3]);
    this->DirectionCosines->SetElement(1,1, dircos[4]);
    this->DirectionCosines->SetElement(2,1, dircos[5]);
    this->DirectionCosines->SetElement(3,1, 0);
    double dircosz[3];
    vtkMath::Cross(dircos, dircos+3, dircosz);
    this->DirectionCosines->SetElement(0,2, dircosz[0]);
    this->DirectionCosines->SetElement(1,2, dircosz[1]);
    this->DirectionCosines->SetElement(2,2, dircosz[2]);
    this->DirectionCosines->SetElement(3,2, 0);
    //std::cout << "Det: " << this->DirectionCosines->Determinant() << std::endl;

    for(int i=0;i<6;++i)
      this->ImageOrientationPatient[i] = dircos[i];
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 2 )
    this->MedicalImageProperties->SetDirectionCosine( this->ImageOrientationPatient );
#endif
    }
  // Apply transform:
#ifdef GDCMV2_0_COMPATIBILITY
  if( dircos && origin )
    {
    if( this->FileLowerLeft )
      {
      // Since we are not doing the VTK Y-flipping operation, Origin and Image Position (Patient)
      // are the same:
      this->DataOrigin[0] = origin[0];
      this->DataOrigin[1] = origin[1];
      this->DataOrigin[2] = origin[2];
      }
    else
      {
      // We are doing the Y-flip:
      // translate Image Position (Patient) along the Y-vector of the Image Orientation (Patient):
      // Step 1: Compute norm of translation vector:
      // Because position is in the center of the pixel, we need to substract 1 to the dimY:
      assert( dims[1] >=1 );
      double norm = (dims[1] - 1) * this->DataSpacing[1];
      // Step 2: translate:
      this->DataOrigin[0] = origin[0] + norm * dircos[3+0];
      this->DataOrigin[1] = origin[1] + norm * dircos[3+1];
      this->DataOrigin[2] = origin[2] + norm * dircos[3+2];
      }
    }
  // Need to set the rest to 0 ???
#endif

  const gdcm::PixelFormat &pixeltype = image.GetPixelFormat();
  this->Shift = image.GetIntercept();
  this->Scale = image.GetSlope();

  //gdcm::PixelFormat::ScalarType outputpt = pixeltype;
  gdcm::PixelFormat::ScalarType outputpt =
    ComputePixelTypeFromFiles(this->FileName, this->FileNames, image);
  if( this->FileName )
    {
    // We should test that outputpt is 8 when BitsAllocated = 16 / Bits Stored = 8
    // BUT we should test that output is 16 when BitsAllocated = 16 / BitsStored = 12
    // assert( outputpt == pixeltype );
    }

  // Compute output pixel format when Rescaling:
//  if( this->Shift != 0 || this->Scale != 1. )
//    {
//    gdcm::Rescaler r;
//    r.SetIntercept( this->Shift );
//    r.SetSlope( this->Scale );
//    r.SetPixelFormat( pixeltype );
//    outputpt = r.ComputeInterceptSlopePixelType();
//    assert( pixeltype <= outputpt );
//    assert( pixeltype.GetSamplesPerPixel() == 1 && image.GetPhotometricInterpretation().GetSamplesPerPixel() == 1 );
//    assert( image.GetPhotometricInterpretation() != gdcm::PhotometricInterpretation::PALETTE_COLOR );
//    }
  //if( pixeltype != outputpt ) assert( Shift != 0. || Scale != 1 );

  this->ForceRescale = 0; // always reset this thing
  // gdcmData/DCMTK_JPEGExt_12Bits.dcm
  if( pixeltype != outputpt && pixeltype.GetBitsAllocated() != 12 )
    {
    this->ForceRescale = 1;
    }

  switch( outputpt )
    {
  case gdcm::PixelFormat::INT8:
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
    this->DataScalarType = VTK_SIGNED_CHAR;
#else
    if( !vtkGDCMImageReader_IsCharTypeSigned() )
      {
      vtkErrorMacro( "Output Pixel Type will be incorrect, go get a newer VTK version" );
      }
    this->DataScalarType = VTK_CHAR;
#endif
    break;
  case gdcm::PixelFormat::UINT8:
    this->DataScalarType = VTK_UNSIGNED_CHAR;
    break;
  case gdcm::PixelFormat::INT16:
    this->DataScalarType = VTK_SHORT;
    break;
  case gdcm::PixelFormat::UINT16:
    this->DataScalarType = VTK_UNSIGNED_SHORT;
    break;
  // RT / SC have 32bits
  case gdcm::PixelFormat::INT32:
    this->DataScalarType = VTK_INT;
    break;
  case gdcm::PixelFormat::UINT32:
    this->DataScalarType = VTK_UNSIGNED_INT;
    break;
  case gdcm::PixelFormat::INT12:
    this->DataScalarType = VTK_SHORT;
    break;
  case gdcm::PixelFormat::UINT12:
    this->DataScalarType = VTK_UNSIGNED_SHORT;
    break;
  //case gdcm::PixelFormat::FLOAT16: // TODO
  case gdcm::PixelFormat::FLOAT32:
    this->DataScalarType = VTK_FLOAT;
    break;
  case gdcm::PixelFormat::FLOAT64:
    this->DataScalarType = VTK_DOUBLE;
    break;
  case gdcm::PixelFormat::SINGLEBIT:
    this->DataScalarType = VTK_BIT;
    break;
  default:
    this->SetErrorCode(vtkErrorCode::FileFormatError);
    vtkErrorMacro( "Do not support this Pixel Type: " << pixeltype.GetScalarTypeAsString()
      << " with " << gdcm::PixelFormat(outputpt).GetScalarTypeAsString()  );
    return 0;
    }
  this->NumberOfScalarComponents = pixeltype.GetSamplesPerPixel();

  // Ok let's fill in the 'extra' info:
  this->FillMedicalImageInformation(reader);

  // Do the IconImage if requested:
  const gdcm::IconImage& icon = image.GetIconImage();
  if( this->LoadIconImage && !icon.IsEmpty() )
    {
    this->IconImageDataExtent[0] = 0;
    this->IconImageDataExtent[1] = icon.GetColumns() - 1;
    this->IconImageDataExtent[2] = 0;
    this->IconImageDataExtent[3] = icon.GetRows() - 1;
    //
    const gdcm::PixelFormat &iconpixelformat = icon.GetPixelFormat();
    switch(iconpixelformat)
      {
    case gdcm::PixelFormat::INT8:
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
      this->IconDataScalarType = VTK_SIGNED_CHAR;
#else
      this->IconDataScalarType = VTK_CHAR;
#endif
      break;
    case gdcm::PixelFormat::UINT8:
      this->IconDataScalarType = VTK_UNSIGNED_CHAR;
      break;
    case gdcm::PixelFormat::INT16:
      this->IconDataScalarType = VTK_SHORT;
      break;
    case gdcm::PixelFormat::UINT16:
      this->IconDataScalarType = VTK_UNSIGNED_SHORT;
      break;
    default:
      vtkErrorMacro( "Do not support this Icon Pixel Type: " << (int)iconpixelformat.GetScalarType() );
      return 0;
      }
    this->IconNumberOfScalarComponents = iconpixelformat.GetSamplesPerPixel();
    }

  // Overlay!
  size_t numoverlays = image.GetNumberOfOverlays();
  if( this->LoadOverlays && numoverlays )
    {
    // Do overlay specific stuff...
    // what if overlay do not have the same data extent as image ?
    for( unsigned int ovidx = 0; ovidx < numoverlays; ++ovidx )
      {
      const gdcm::Overlay& ov = image.GetOverlay(ovidx);
      assert( (unsigned int)ov.GetRows() == image.GetRows() ); (void)ov;
      assert( (unsigned int)ov.GetColumns() == image.GetColumns() );
      }
    this->NumberOfOverlays = (int)numoverlays;
    }

//  return this->Superclass::RequestInformation(
//    request, inputVector, outputVector);

  return 1;
}

//----------------------------------------------------------------------------
template <class T>
static inline unsigned long vtkImageDataGetTypeSize(T*, int a = 0,int b = 0)
{
  (void)a;(void)b;
  return sizeof(T);
}

//----------------------------------------------------------------------------
static void InPlaceYFlipImage(vtkImageData* data, const int dext[6])
{
  unsigned long outsize = data->GetNumberOfScalarComponents();
  if( dext[1] == dext[0] && dext[0] == 0 ) return;

  // Multiply by the number of bytes per scalar
  switch (data->GetScalarType())
    {
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
    case VTK_BIT: { outsize /= 8; }; break;
    vtkTemplateMacro(
      outsize *= vtkImageDataGetTypeSize(static_cast<VTK_TT*>(0))
    );
#else
    case VTK_BIT: { outsize /= 8; }; break;
    vtkTemplateMacro3(
      outsize *= vtkImageDataGetTypeSize, static_cast<VTK_TT*>(0), 0, 0
    );
#endif
    default:
      //vtkErrorMacro("do not support scalar type: " << data->GetScalarType() );
      assert(0);
    }
  outsize *= (dext[1] - dext[0] + 1);
  char * ref = static_cast<char*>(data->GetScalarPointer());
  char * pointer = static_cast<char*>(data->GetScalarPointer());
  assert( pointer );

  char *line = new char[outsize];

  for(int j = dext[4]; j <= dext[5]; ++j)
    {
    char *start = pointer;
    assert( start == ref + j * outsize * (dext[3] - dext[2] + 1) ); (void)ref;
    // Swap two-lines at a time
    // when Rows is odd number (359) then dext[3] == 178
    // so we should avoid copying the line right in the center of the image
    // since memcpy does not like copying on itself...
    for(int i = dext[2]; i < (dext[3]+1) / 2; ++i)
      {
      // image:
      char * end = start+(dext[3] - i)*outsize;
      assert( (end - pointer) >= (int)outsize );
      memcpy(line,end,outsize); // duplicate line
      memcpy(end,pointer,outsize);
      memcpy(pointer,line,outsize);
      pointer += outsize;
      }
    // because the for loop iterated only over 1/2 all lines, skip to the next slice:
    assert( dext[2] == 0 );
    pointer += (dext[3] + 1 - (dext[3]+1)/2 )*outsize;
    }
  // Did we reach the end ?
  assert( pointer == ref + (dext[5]-dext[4]+1)*(dext[3]-dext[2]+1)*outsize );
  delete[] line;
}

//----------------------------------------------------------------------------
int vtkGDCMImageReader::LoadSingleFile(const char *filename, char *pointer, unsigned long &outlen)
{
  int *dext = this->GetDataExtent();
  vtkImageData *data = this->GetOutput(0);
  //bool filelowerleft = this->FileLowerLeft ? true : false;

  //char * pointer = static_cast<char*>(data->GetScalarPointer());

  gdcm::ImageReader reader;
  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    vtkErrorMacro( "ImageReader failed: " << filename );
    return 0;
    }
  gdcm::Image &image = reader.GetImage();
  this->LossyFlag = image.IsLossy();
  //VTK does not cope with Planar Configuration, so let's schew the work to please it
  assert( this->PlanarConfiguration == 0 || this->PlanarConfiguration == 1 );
  // Store the PlanarConfiguration before inverting it !
  this->PlanarConfiguration = image.GetPlanarConfiguration();
  //assert( this->PlanarConfiguration == 0 || this->PlanarConfiguration == 1 );
  if( image.GetPlanarConfiguration() == 1 )
    {
    gdcm::ImageChangePlanarConfiguration icpc;
    icpc.SetInput( image );
    icpc.SetPlanarConfiguration( 0 );
    icpc.Change();
    image = icpc.GetOutput();
    }

  const gdcm::PixelFormat &pixeltype = image.GetPixelFormat();
  assert( image.GetNumberOfDimensions() == 2 || image.GetNumberOfDimensions() == 3 );
  /*const*/ unsigned long len = image.GetBufferLength();
  outlen = len;
  unsigned long overlaylen = 0;
  //image.GetBuffer(pointer);
  // HACK: Make sure that Shift/Scale are the one from the file:
  this->Shift = image.GetIntercept();
  this->Scale = image.GetSlope();

  if( (this->Scale != 1.0 || this->Shift != 0.0) || this->ForceRescale )
  {
    assert( pixeltype.GetSamplesPerPixel() == 1 );
    gdcm::Rescaler r;
    r.SetIntercept( Shift ); // FIXME
    r.SetSlope( Scale ); // FIXME
    gdcm::PixelFormat::ScalarType targetpixeltype = gdcm::PixelFormat::UNKNOWN;
    // r.SetTargetPixelType( gdcm::PixelFormat::FLOAT64 );
    int scalarType = data->GetScalarType();
   switch( scalarType )
    {
  case VTK_CHAR:
    if( vtkGDCMImageReader_IsCharTypeSigned() )
      targetpixeltype = gdcm::PixelFormat::INT8;
    else
      targetpixeltype = gdcm::PixelFormat::UINT8;
    break;
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
  case VTK_SIGNED_CHAR:
    targetpixeltype = gdcm::PixelFormat::INT8;
    break;
#endif
  case VTK_UNSIGNED_CHAR:
    targetpixeltype = gdcm::PixelFormat::UINT8;
    break;
  case VTK_SHORT:
    targetpixeltype = gdcm::PixelFormat::INT16;
    break;
  case VTK_UNSIGNED_SHORT:
    targetpixeltype = gdcm::PixelFormat::UINT16;
    break;
  case VTK_INT:
    targetpixeltype = gdcm::PixelFormat::INT32;
    break;
  case VTK_UNSIGNED_INT:
    targetpixeltype = gdcm::PixelFormat::UINT32;
    break;
  case VTK_FLOAT:
    targetpixeltype = gdcm::PixelFormat::FLOAT32;
    break;
  case VTK_DOUBLE:
    targetpixeltype = gdcm::PixelFormat::FLOAT64;
    break;
  case VTK_BIT:
    targetpixeltype = gdcm::PixelFormat::SINGLEBIT;
    break;
  default:
    vtkErrorMacro( "Do not support this Pixel Type: " << scalarType );
    assert( 0 );
    return 0;
    }
    r.SetTargetPixelType( targetpixeltype );

    r.SetUseTargetPixelType(true);
    r.SetPixelFormat( pixeltype );
    char * copy = new char[len];
    //memcpy(copy, pointer, len);
    image.GetBuffer(copy);
    if( !r.Rescale(pointer,copy,len) )
      {
      vtkErrorMacro( "Could not Rescale" );
      // problem with gdcmData/3E768EB7.dcm
      return 0;
      }
    delete[] copy;
    // WARNING: sizeof(Real World Value) != sizeof(Stored Pixel)
    outlen = data->GetScalarSize() * data->GetNumberOfPoints() / data->GetDimensions()[2];
    assert( data->GetNumberOfScalarComponents() == 1 );
  }
  else
    {
    image.GetBuffer(pointer);
    }

  // Do the Icon Image:
  if( this->LoadIconImage )
    {
    this->NumberOfIconImages = image.GetIconImage().IsEmpty() ? 0 : 1;
    if( this->NumberOfIconImages )
      {
      char * iconpointer = static_cast<char*>(this->GetOutput(ICONIMAGEPORTNUMBER)->GetScalarPointer());
      assert( iconpointer );
      image.GetIconImage().GetBuffer( iconpointer );
      if ( image.GetIconImage().GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::PALETTE_COLOR )
        {
        const gdcm::LookupTable &lut = image.GetIconImage().GetLUT();
        assert( lut.GetBitSample() == 8 );
          {
          vtkLookupTable *vtklut = vtkLookupTable::New();
          vtklut->SetNumberOfTableValues(256);
          // SOLVED: GetPointer(0) is skrew up, need to replace it with WritePointer(0,4) ...
          if( !lut.GetBufferAsRGBA( vtklut->WritePointer(0,4) ) )
            {
            vtkWarningMacro( "Could not get values from LUT" );
            return 0;
            }
          vtklut->SetRange(0,255);
          this->GetOutput(ICONIMAGEPORTNUMBER)->GetPointData()->GetScalars()->SetLookupTable( vtklut );
          vtklut->Delete();
          }
        }
      }
    }

  // Do the Curve:
  size_t numcurves = image.GetNumberOfCurves();
  if( numcurves )
    {
    const gdcm::Curve& curve = image.GetCurve();
    //curve.Print( std::cout );
    vtkPoints * pts = vtkPoints::New();
    pts->SetNumberOfPoints( curve.GetNumberOfPoints() );
    curve.GetAsPoints( (float*)pts->GetVoidPointer(0) );
    vtkCellArray *polys = vtkCellArray::New();
    for(unsigned int i = 0; i < curve.GetNumberOfPoints(); i+=2 )
      {
      polys->InsertNextCell(2);
      polys->InsertCellPoint(i);
      polys->InsertCellPoint(i+1);
      }
    vtkPolyData *cube = vtkPolyData::New();
    cube->SetPoints(pts);
    pts->Delete();
    cube->SetLines(polys);
    polys->Delete();
    SetCurve(cube);
    cube->Delete();
    }

  // Do the Overlay:
  if( this->LoadOverlays )
    {
    size_t numoverlays = image.GetNumberOfOverlays();
    long overlayoutsize = (dext[1] - dext[0] + 1);
    //this->NumberOfOverlays = numoverlays;
    //if( numoverlays )
    if( !this->LoadOverlays ) assert( this->NumberOfOverlays == 0 );
    for( int ovidx = 0;  ovidx < this->NumberOfOverlays; ++ovidx )
      {
      vtkImageData *vtkimage = this->GetOutput(OVERLAYPORTNUMBER + ovidx);
      // vtkOpenGLImageMapper::RenderData does not support bit array (since OpenGL does not either)
      // we have to decompress the bit overlay into an unsigned char array to please everybody:
      const gdcm::Overlay& ov1 = image.GetOverlay(ovidx);
      vtkUnsignedCharArray *chararray = vtkUnsignedCharArray::New();
      chararray->SetNumberOfTuples( overlayoutsize * ( dext[3] - dext[2] + 1 ) );
      overlaylen = overlayoutsize * ( dext[3] - dext[2] + 1 );
      assert( (unsigned long)ov1.GetRows()*ov1.GetColumns() <= overlaylen );
      const signed short *origin = ov1.GetOrigin();
      if( (unsigned long)ov1.GetRows()*ov1.GetColumns() != overlaylen )
        {
        vtkWarningMacro( "vtkImageData Overlay have an extent that do not match the one of the image" );
        }
      if( origin[0] != 1 || origin[1] != 1 )
        {
        // Table C.9-2 OVERLAY PLANE MODULE ATTRIBUTES
        vtkWarningMacro( "Overlay with origin are not supported right now" );
        }
      vtkimage->GetPointData()->SetScalars( chararray );
      vtkimage->GetPointData()->GetScalars()->SetName( ov1.GetDescription() );
      chararray->Delete();

      assert( vtkimage->GetScalarType() == VTK_UNSIGNED_CHAR );
      char * overlaypointer = static_cast<char*>(vtkimage->GetScalarPointer());
      assert( overlaypointer );
      //assert( image->GetPointData()->GetScalars() != 0 );

      //memset(overlaypointer,0,overlaylen); // FIXME: can be optimized
      if( !ov1.GetUnpackBuffer( overlaypointer, overlaylen ) )
        {
        vtkErrorMacro( "Problem in GetUnpackBuffer" );
        }
      }
    if( numoverlays ) assert( (unsigned long)overlayoutsize * ( dext[3] - dext[2] + 1 ) == overlaylen );
    }

  //const gdcm::PixelFormat &pixeltype = image.GetPixelFormat();
  // Do the LUT
  if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::PALETTE_COLOR )
    {
    this->ImageFormat = VTK_LOOKUP_TABLE;
    const gdcm::LookupTable &lut = image.GetLUT();
    if( lut.GetBitSample() == 8 )
      {
      vtkLookupTable *vtklut = vtkLookupTable::New();
      vtklut->SetNumberOfTableValues(256);
      // SOLVED: GetPointer(0) is skrew up, need to replace it with WritePointer(0,4) ...
      if( !lut.GetBufferAsRGBA( vtklut->WritePointer(0,4) ) )
        {
        vtkWarningMacro( "Could not get values from LUT" );
        return 0;
        }
      vtklut->SetRange(0,255);
      data->GetPointData()->GetScalars()->SetLookupTable( vtklut );
      vtklut->Delete();
      }
    else
      {
#if (VTK_MAJOR_VERSION >= 5)
      assert( lut.GetBitSample() == 16 );
      vtkLookupTable16 *vtklut = vtkLookupTable16::New();
      vtklut->SetNumberOfTableValues(256*256);
      // SOLVED: GetPointer(0) is skrew up, need to replace it with WritePointer(0,4) ...
      if( !lut.GetBufferAsRGBA( (unsigned char*)vtklut->WritePointer(0,4) ) )
        {
        vtkWarningMacro( "Could not get values from LUT" );
        return 0;
        }
      vtklut->SetRange(0,256*256-1);
      data->GetPointData()->GetScalars()->SetLookupTable( vtklut );
      vtklut->Delete();
#else
      vtkWarningMacro( "Unhandled" );
#endif
      }
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME1 )
    {
    this->ImageFormat = VTK_INVERSE_LUMINANCE;
    vtkWindowLevelLookupTable *vtklut = vtkWindowLevelLookupTable::New();
    // Technically we could also use the first of the Window Width / Window Center
    // oh well, if they are missing let's just compute something:
    const double min = (double)pixeltype.GetMin();
    const double max = (double)pixeltype.GetMax();
    vtklut->SetWindow( max - min );
    vtklut->SetLevel( 0.5 * (max + min) );
    //vtklut->SetWindow(1024); // WindowWidth
    //vtklut->SetLevel(550); // WindowCenter
    vtklut->InverseVideoOn();
    data->GetPointData()->GetScalars()->SetLookupTable( vtklut );
    vtklut->Delete();
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::YBR_FULL_422 )
    {
    this->ImageFormat = VTK_YBR;
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::YBR_FULL )
    {
    this->ImageFormat = VTK_YBR;
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::RGB )
    {
    this->ImageFormat = VTK_RGB;
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::MONOCHROME2 )
    {
    this->ImageFormat = VTK_LUMINANCE;
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::YBR_RCT )
    {
    this->ImageFormat = VTK_RGB;
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::YBR_ICT )
    {
    this->ImageFormat = VTK_RGB;
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::CMYK )
    {
    this->ImageFormat = VTK_CMYK;
    }
  else if ( image.GetPhotometricInterpretation() == gdcm::PhotometricInterpretation::ARGB )
    {
    this->ImageFormat = VTK_RGBA;
    }
  else
    {
    // HSV / CMYK ???
    // let's just give up for now
    vtkErrorMacro( "Does not handle: " << image.GetPhotometricInterpretation().GetString() );
    //return 0;
    }
  //assert( this->ImageFormat );

  long outsize;
  if( data->GetScalarType() == VTK_BIT )
    {
    outsize = (dext[1] - dext[0] + 1) / 8;
    }
  else
    {
    outsize = pixeltype.GetPixelSize()*(dext[1] - dext[0] + 1);
    }

  if( this->FileName) assert( (unsigned long)outsize * (dext[3] - dext[2]+1) * (dext[5]-dext[4]+1) == len );

  return 1; // success
}


//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
int vtkGDCMImageReader::RequestData(vtkInformation *vtkNotUsed(request),
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
  int res = RequestDataCompat();
  return res;
}
#endif

//----------------------------------------------------------------------------
int vtkGDCMImageReader::RequestDataCompat()
{
  vtkImageData *output = this->GetOutput(0);
  output->GetPointData()->GetScalars()->SetName("GDCMImage");

  // The outExt is the allocated data extent
  int outExt[6];
  output->GetExtent(outExt);
  // The dext is the whole extent (includes not-loaded data)
  int *dext = this->GetDataExtent();
  //vtkIdType outInc[3];
  //data->GetIncrements(outInc);
  //int outSize[3];
  //data->GetDimensions(outSize);

  //void *outPtr = data->GetScalarPointerForExtent(outExt);

  char * pointer = static_cast<char*>(output->GetScalarPointerForExtent(outExt));
  if( this->FileName )
    {
    const char *filename = this->FileName;
    unsigned long len;
    int load = this->LoadSingleFile( filename, pointer, len ); (void)len;
    if( !load )
      {
      // FIXME: I need to fill the buffer with 0, shouldn't I ?
      return 0;
      }
    }
  else if( this->FileNames && this->FileNames->GetNumberOfValues() >= 1 )
    {
    // Load each 2D files
    // HACK: len is moved out of the loop so that when file > 1 start failing we can still know
    // the len of the buffer...technically all files should have the same len (not checked for now)
    unsigned long len = 0;
    for(int j = dext[4]; !this->AbortExecute && j <= dext[5]; ++j)
      {
      assert( j >= 0 && j <= this->FileNames->GetNumberOfValues() );
      const char *filename = this->FileNames->GetValue( j );
      int load = this->LoadSingleFile( filename, pointer, len );
      if( !load )
        {
        // hum... we could not read this file within the series, let's just fill
        // the slice with 0 value, hopefully this should be the right thing to do
        memset( pointer, 0, len);
        }
      assert( len );
      pointer += len;
      this->UpdateProgress( (double)(j - dext[4] ) / ( dext[5] - dext[4] ));
      }
    }
  else
    {
    return 0;
    }
  // Y-flip image
  if (!this->FileLowerLeft)
    {
    InPlaceYFlipImage(this->GetOutput(0), dext);
    if( this->LoadIconImage )
      {
      int *iiext = this->IconImageDataExtent;
      InPlaceYFlipImage(this->GetOutput(ICONIMAGEPORTNUMBER), iiext);
      }
    for( int ovidx = 0;  ovidx < this->NumberOfOverlays; ++ovidx )
      {
      assert( this->LoadOverlays );
      int oext[6];
      this->GetDataExtent(oext);
      oext[4] = 0;
      oext[5] = 0;
      InPlaceYFlipImage(this->GetOutput(OVERLAYPORTNUMBER+ovidx), oext);
      }
    }

  return 1;
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
vtkAlgorithmOutput* vtkGDCMImageReader::GetOverlayPort(int index)
{
  if( index >= 0 && index < this->NumberOfOverlays)
    return this->GetOutputPort(index+OVERLAYPORTNUMBER);
  return NULL;
}
vtkAlgorithmOutput* vtkGDCMImageReader::GetIconImagePort()
{
  const int index = 0;
  if( index >= 0 && index < this->NumberOfIconImages)
    return this->GetOutputPort(index+ICONIMAGEPORTNUMBER);
  return NULL;
}
#endif

//----------------------------------------------------------------------------
vtkImageData* vtkGDCMImageReader::GetOverlay(int i)
{
  if( i >= 0 && i < this->NumberOfOverlays)
    return this->GetOutput(i+OVERLAYPORTNUMBER);
  return NULL;
}

//----------------------------------------------------------------------------
vtkImageData* vtkGDCMImageReader::GetIconImage()
{
  const int i = 0;
  if( i >= 0 && i < this->NumberOfIconImages)
    return this->GetOutput(i+ICONIMAGEPORTNUMBER);
  return NULL;
}

//----------------------------------------------------------------------------
void vtkGDCMImageReader::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
