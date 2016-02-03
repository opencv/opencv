/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkGDCMImageWriter.h"

#include "vtkVersion.h"
#include "vtkObjectFactory.h"
#include "vtkImageData.h"
#include "vtkLookupTable.h"
#include "vtkLookupTable16.h"
#include "vtkMath.h"
#include "vtkMatrix4x4.h"
#include "vtkMedicalImageProperties.h"
#include "vtkGDCMMedicalImageProperties.h"
#include "vtkStringArray.h"
#include "vtkPointData.h"
#include "vtkGDCMImageReader.h"
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
#include "vtkInformationVector.h"
#include "vtkStreamingDemandDrivenPipeline.h"
#include "vtkInformation.h"
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/

#include "gdcmImageWriter.h"
#include "gdcmByteValue.h"
#include "gdcmUIDGenerator.h"
#include "gdcmAnonymizer.h"
#include "gdcmAttribute.h"
#include "gdcmRescaler.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"
#include "gdcmDict.h"
#include "gdcmTag.h"
#include "gdcmImageHelper.h"
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmImageChangeTransferSyntax.h"

#include <limits>

vtkCxxRevisionMacro(vtkGDCMImageWriter, "$Revision: 1.1 $")
vtkStandardNewMacro(vtkGDCMImageWriter)

//vtkCxxSetObjectMacro(vtkGDCMImageWriter,LookupTable,vtkLookupTable)
vtkCxxSetObjectMacro(vtkGDCMImageWriter,MedicalImageProperties,vtkMedicalImageProperties)
vtkCxxSetObjectMacro(vtkGDCMImageWriter,FileNames,vtkStringArray)
vtkCxxSetObjectMacro(vtkGDCMImageWriter,DirectionCosines,vtkMatrix4x4)

inline bool vtkGDCMImageWriter_IsCharTypeSigned()
{
#ifndef VTK_TYPE_CHAR_IS_SIGNED
  unsigned char uc = 255;
  return (*reinterpret_cast<char*>(&uc) < 0) ? true : false;
#else
  return VTK_TYPE_CHAR_IS_SIGNED;
#endif
}

#ifndef vtkFloatingPointType
#define vtkFloatingPointType float
#endif

//----------------------------------------------------------------------------
vtkGDCMImageWriter::vtkGDCMImageWriter()
{
  this->DataUpdateExtent[0] = 0;
  this->DataUpdateExtent[1] = 0;
  this->DataUpdateExtent[2] = 0;
  this->DataUpdateExtent[3] = 0;
  this->DataUpdateExtent[4] = 0;
  this->DataUpdateExtent[5] = 0;

  //this->LookupTable = vtkLookupTable::New();
  this->MedicalImageProperties = vtkMedicalImageProperties::New();
  this->FileNames = vtkStringArray::New();
  this->StudyUID = 0;
  this->SeriesUID = 0;
  this->DirectionCosines = vtkMatrix4x4::New();
  this->DirectionCosines->SetElement(0,0,1);
  this->DirectionCosines->SetElement(1,0,0);
  this->DirectionCosines->SetElement(2,0,0);
  this->DirectionCosines->SetElement(0,1,0);
  this->DirectionCosines->SetElement(1,1,1);
  this->DirectionCosines->SetElement(2,1,0);

  // This is the same root as ITK, but implementation version will be different...
  gdcm::UIDGenerator::SetRoot( "1.2.826.0.1.3680043.2.1125" );

  // echo "VTK" | od -b
  gdcm::FileMetaInformation::AppendImplementationClassUID( "126.124.113" );
  const std::string project_name = std::string("GDCM/VTK ") + vtkVersion::GetVTKVersion();
  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( project_name.c_str() );

  this->ImageFormat = 0; // invalid

  this->Shift = 0.;
  this->Scale = 1.;
  this->FileLowerLeft = 0; // same default as vtkImageReader2
  this->PlanarConfiguration = 0;
  this->LossyFlag = 0;
  this->CompressionType = NO_COMPRESSION;

  // For both case (2d file or 3d file) we need a common uid for the Series/Study:
  gdcm::UIDGenerator uidgen;
  const char *uid = uidgen.Generate();
  this->SetStudyUID(uid);
  uid = uidgen.Generate();
  this->SetSeriesUID(uid);
}

//----------------------------------------------------------------------------
vtkGDCMImageWriter::~vtkGDCMImageWriter()
{
  //this->LookupTable->Delete();
  this->MedicalImageProperties->Delete();
  this->FileNames->Delete();
  this->SetStudyUID(NULL);
  this->SetSeriesUID(NULL);
  this->DirectionCosines->Delete();
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
int vtkGDCMImageWriter::FillInputPortInformation(
  int port, vtkInformation *info)
{
  if (!this->Superclass::FillInputPortInformation(port, info))
    {
    return 0;
    }
  info->Set(vtkAlgorithm::INPUT_IS_REPEATABLE(), 1);
  return 1;
}
//---------------------------------------------------------------------------
int vtkGDCMImageWriter::RequestInformation(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *vtkNotUsed(outputVector))
{
  // Check to make sure that all input information agrees
  int mismatchedInputs = 0;

  double spacing[3];
  double origin[3];
  int extent[6];
  int components = 0;
  int dataType = 0;

  // For each connection on port 0, check against the first connection
  for (int i = 0; i < this->GetNumberOfInputConnections(0); i++)
    {
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(i);
    if (i == 0)
      {
      inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent);
      inInfo->Get(vtkDataObject::SPACING(), spacing);
      inInfo->Get(vtkDataObject::ORIGIN(), origin);
      components = inInfo->Get(vtkDataObject::FIELD_NUMBER_OF_COMPONENTS());
      dataType = inInfo->Get(vtkDataObject::FIELD_ARRAY_TYPE());
      continue;
      }

    if (memcmp(inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT()),
               extent, sizeof(extent)) ||
        memcmp(inInfo->Get(vtkDataObject::SPACING()), spacing,
               sizeof(spacing)) ||
        memcmp(inInfo->Get(vtkDataObject::ORIGIN()), origin,
               sizeof(origin)) ||
        inInfo->Get(vtkDataObject::FIELD_NUMBER_OF_COMPONENTS())
          != components ||
        inInfo->Get(vtkDataObject::FIELD_ARRAY_TYPE()) != dataType)
      {
      mismatchedInputs = 1;
      return 0;
      }
    }

  // Technically we should be much more paranoid with the shift scale (like value bigger
  // then stored pixel type: does this even make sense ?)
  // Let's do the easy one here:
  // do I really need to comment on this one:
  if( this->Scale == 0 )
    {
    return 0;
    }

  return 1;
}

//--------------------------------------------------------------------------
int vtkGDCMImageWriter::RequestUpdateExtent(
  vtkInformation *vtkNotUsed(request),
  vtkInformationVector **inputVector,
  vtkInformationVector *vtkNotUsed(outputVector))
{
  // Set the UpdateExtent from the DataUpdateExtent for the current slice
  int n = inputVector[0]->GetNumberOfInformationObjects();
  for (int i = 0; i < n; i++)
    {
    vtkInformation *inInfo = inputVector[0]->GetInformationObject(i);
    inInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(),
                this->DataUpdateExtent, 6);
    }

  return 1;
}

//--------------------------------------------------------------------------
int vtkGDCMImageWriter::RequestData(
  vtkInformation* vtkNotUsed(request),
  vtkInformationVector** inputVector,
  vtkInformationVector* vtkNotUsed(outputVector))
{
  // Go through the inputs and write the data for each
  int numTimeSteps = inputVector[0]->GetNumberOfInformationObjects();

  for (int timeStep = 0; timeStep < numTimeSteps; timeStep++)
    {
    vtkInformation *inInfo =
      inputVector[0]->GetInformationObject(timeStep);
    vtkImageData *input =
      vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));

    // Error checking
    if (input == NULL)
      {
      // Close file, set GDCMFileID to zero
      //this->CloseFile(this->GDCMFileId);
      //this->GDCMFileId = 0;
      vtkErrorMacro(<<"Write: Please specify an input!");
      return 0;
      }
    // Call WriteGDCMData for each input
    if (this->WriteGDCMData(input, timeStep) == 0)
      {
      return 0;
      }
    }

  return 1;
}
#endif /*(VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )*/

//----------------------------------------------------------------------------
/*const*/ char *vtkGDCMImageWriter::GetFileName()
{
  if( this->FileNames->GetNumberOfValues() )
    {
    const char *filename = this->FileNames->GetValue(0);
    return (char*)filename;
    }
  return this->Superclass::GetFileName();
}

//----------------------------------------------------------------------------
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
void vtkGDCMImageWriter::Write()
{
  if (this->GetFileName() == 0)
    {
    vtkErrorMacro("Write: You must supply a file name.");
    return;
    }

  // Get the first input and update its information.
  vtkImageData *input = this->GetImageDataInput(0);

  if (input == 0)
    {
    vtkErrorMacro("Write: No input supplied.");
    return;
    }
#if (VTK_MAJOR_VERSION >= 6)
#else
  input->UpdateInformation();
#endif

  // Update the rest.
  this->UpdateInformation();

  // Get the whole extent of the input
#if (VTK_MAJOR_VERSION >= 6)
  vtkInformation *inInfo = this->GetInputInformation(0, 0);
  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), this->DataUpdateExtent);
#else
  input->GetWholeExtent(this->DataUpdateExtent);
#endif

  if (this->DataUpdateExtent[0] == (this->DataUpdateExtent[1] + 1) ||
      this->DataUpdateExtent[2] == (this->DataUpdateExtent[3] + 1) ||
      this->DataUpdateExtent[4] == (this->DataUpdateExtent[5] + 1))
    {
    vtkErrorMacro("Write: Empty input supplied.");
    return;
    }


  // For both case (2d file or 3d file) we need a common uid for the Series/Study:
  //gdcm::UIDGenerator uidgen;
  //const char *uid = uidgen.Generate();
  //this->SetStudyUID(uid);
  //uid = uidgen.Generate();
  //this->SetSeriesUID(uid);

  // Did the user specified dim of output file to be 2 ?
  if( this->FileDimensionality == 2 )
    {
    int dimIndex = 2;
    int firstSlice = this->DataUpdateExtent[2*dimIndex];
    int lastSlice = this->DataUpdateExtent[2*dimIndex+1];
    assert( lastSlice >= firstSlice );
    if( lastSlice - firstSlice ) // will be == 0 when only a single slice
      {
      if( lastSlice - firstSlice + 1 != this->FileNames->GetNumberOfValues() )
        {
        vtkErrorMacro("Wrong number of filenames: " << this->FileNames->GetNumberOfValues()
          << " should be " << lastSlice - firstSlice + 1);
        return;
        }
      }

    // Go through data slice-by-slice using file-order slices
    for (int slice = firstSlice; slice <= lastSlice; slice++)
      {
      //std::cerr << "Slice:" << slice << std::endl;
      // Set the DataUpdateExtent to the slice extent we want to write
      this->DataUpdateExtent[2*dimIndex] = slice;
      this->DataUpdateExtent[2*dimIndex+1] = slice;
      this->Modified();

      // Call Update to execute pipeline and write slice to disk.
      this->Update();
      }
    }
  else if( this->FileDimensionality == 3 )
    {
    // Call Update to execute pipeline and write slice to disk.
    this->Update();
    }
  else
    {
    vtkErrorMacro( "Unhandled: " << this->FileDimensionality );
    }

}
#else
//----------------------------------------------------------------------------
// Writes all the data from the input.
void vtkGDCMImageWriter::Write()
{
  // Error checking
  if ( this->GetInput() == NULL )
    {
    vtkErrorMacro(<<"Write:Please specify an input!");
    return;
    }
//  if (!this->WriteToMemory && !this->FileName && !this->FilePattern)
//    {
//    vtkErrorMacro(<<"Write:Please specify either a FileName or a file prefix and pattern");
//    return;
//    }

  // Make sure the file name is allocated
  this->InternalFileName =  0;
//    new char[(this->FileName ? strlen(this->FileName) : 1) +
//            (this->FilePrefix ? strlen(this->FilePrefix) : 1) +
//            (this->FilePattern ? strlen(this->FilePattern) : 1) + 10];

  // Fill in image information.
  this->GetInput()->UpdateInformation();
  int *wExtent;
  wExtent = this->GetInput()->GetWholeExtent();
  this->FileNumber = this->GetInput()->GetWholeExtent()[4];
  this->UpdateProgress(0.0);
  // loop over the z axis and write the slices
  for (this->FileNumber = wExtent[4]; this->FileNumber <= wExtent[5];
       ++this->FileNumber)
    {
    this->GetInput()->SetUpdateExtent(wExtent[0], wExtent[1],
                                      wExtent[2], wExtent[3],
                                      this->FileNumber,
                                      this->FileNumber);
    // determine the name
/*
    if (this->FileName)
      {
      sprintf(this->InternalFileName,"%s",this->FileName);
      }
    else
      {
      if (this->FilePrefix)
        {
        sprintf(this->InternalFileName, this->FilePattern,
                this->FilePrefix, this->FileNumber);
        }
      else
        {
        sprintf(this->InternalFileName, this->FilePattern,this->FileNumber);
        }
      }
*/
    this->GetInput()->UpdateData();
    this->WriteSlice(this->GetInput());
    this->UpdateProgress((this->FileNumber - wExtent[4])/
                         (wExtent[5] - wExtent[4] + 1.0));
    }
  //delete [] this->InternalFileName;
  //this->InternalFileName = NULL;
}

//----------------------------------------------------------------------------
void vtkGDCMImageWriter::WriteSlice(vtkImageData *data)
{
  this->WriteGDCMData(data, 0);
}

#endif

//----------------------------------------------------------------------------
//void SetStringValueFromTag(const char *s, const gdcm::Tag& t, gdcm::DataSet& ds)
static void SetStringValueFromTag(const char *s, const gdcm::Tag& t, gdcm::Anonymizer & ano)
{
  if( s && *s )
    {
#if 0
    gdcm::DataElement de( t );
    de.SetByteValue( s, strlen( s ) );
    const gdcm::Global& g = gdcm::Global::GetInstance();
    const gdcm::Dicts &dicts = g.GetDicts();
    // FIXME: we know the tag at compile time we could save some time
    // Using the static dict instead of the run-time one:
    const gdcm::DictEntry &dictentry = dicts.GetDictEntry( t );
    de.SetVR( dictentry.GetVR() );
    ds.Insert( de );
#else
    ano.Replace(t, s);
#endif
    }
}


//----------------------------------------------------------------------------
int vtkGDCMImageWriter::WriteGDCMData(vtkImageData *data, int timeStep)
{
  //std::cerr << "Calling WriteGDCMData" << std::endl;
  assert( timeStep >= 0 );
#if (VTK_MAJOR_VERSION >= 6)
  vtkInformation *inInfo = this->GetInputInformation(0, timeStep);
  int inWholeExt[6];
  inInfo->Get(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), inWholeExt);
  int inExt[6];
  inInfo->Get(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), inExt);
#else
  int inWholeExt[6];
  data->GetWholeExtent(inWholeExt);
  int inExt[6];
  data->GetUpdateExtent(inExt);
#endif
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 2 )
  vtkIdType inInc[3];
#else
  int inInc[3];
#endif
  data->GetIncrements(inInc);

  //data->Update();
  //data->Print( std::cout );
  //const char * filename = this->GetFileName();
  //std::cerr << data->GetDataDimension() << std::endl;

  gdcm::ImageWriter writer;
  //writer.SetImage( image );
  gdcm::ImageChangeTransferSyntax change;

  //gdcm::Image &image = writer.GetImage();
  gdcm::Image &image = change.GetInput();

  image.SetLossyFlag( this->LossyFlag );
  // Nowadays this is the default one:
#ifdef GDCM_WORDS_BIGENDIAN
  // FIXME: this is not the default syntax, but should be a little faster on big endian machine
  // let see if people complain dataset cannot be sent
  image.SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRBigEndian );
#else
  // that's the default syntax AND it is the fastest syntax to write to disk.
  image.SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
#endif
  image.SetNumberOfDimensions( 2 ); // good default
  const int *dims = data->GetDimensions();
  assert( dims[0] >= 0 && dims[1] >= 0 && dims[2] >= 0 );
  image.SetDimension(0, dims[0] );
  image.SetDimension(1, dims[1] );
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 2 )
  const double *spacing = data->GetSpacing();
#else
  const float *spacing = data->GetSpacing();
#endif
  image.SetSpacing(0, spacing[0] );
  image.SetSpacing(1, spacing[1] );
  if( dims[2] > 1 && this->FileDimensionality == 3 )
    {
    // resize num of dim to 3:
    image.SetNumberOfDimensions( 3 );
    image.SetDimension(2, dims[2] );
    }
  // Even in case of 2D image, pass the 3rd dimension spacing, this might
  // Be needed for example in MR : Spacing Between Slice tag
  image.SetSpacing(2, spacing[2] ); // should always be valid...
  // TODO: need to do Origin / Image Position (Patient)
  // For now FileDimensionality should match File Dimension
  //this->FileDimensionality
  int scalarType = data->GetScalarType();
  gdcm::PixelFormat pixeltype = gdcm::PixelFormat::UNKNOWN;
  bool forcerescale = false;
  switch( scalarType )
    {
  case VTK_BIT:
    pixeltype = gdcm::PixelFormat::SINGLEBIT;
    break;
  case VTK_CHAR:
    if( vtkGDCMImageWriter_IsCharTypeSigned() )
      pixeltype = gdcm::PixelFormat::INT8;
    else
      pixeltype = gdcm::PixelFormat::UINT8;
    break;
#if (VTK_MAJOR_VERSION >= 5) || ( VTK_MAJOR_VERSION == 4 && VTK_MINOR_VERSION > 5 )
  case VTK_SIGNED_CHAR:
    pixeltype = gdcm::PixelFormat::INT8;
    break;
#endif
  case VTK_UNSIGNED_CHAR:
    pixeltype = gdcm::PixelFormat::UINT8;
    break;
  case VTK_SHORT:
    pixeltype = gdcm::PixelFormat::INT16;
    break;
  case VTK_UNSIGNED_SHORT:
    pixeltype = gdcm::PixelFormat::UINT16;
    break;
  case VTK_INT:
    pixeltype = gdcm::PixelFormat::INT32;
    break;
  case VTK_UNSIGNED_INT:
    pixeltype = gdcm::PixelFormat::UINT32;
    break;
  case VTK_FLOAT:
    if( this->Shift == (int)this->Shift && this->Scale == (int)this->Scale )
      {
      // I cannot consider that this is a problem, afterall a floating point type image
      // could in fact really be only integer type, only print a warning to inform dummy user
      vtkWarningMacro( "Image is floating point type, but rescale type is integer type. Rescaling anyway" );
      }
    /*
    Note to myself: should I allow people to squeeze into unsigned char ? Or can I assume most people
    will be doing unsigned short anyway...
    */
    pixeltype = gdcm::PixelFormat::FLOAT32;
    forcerescale = true;
    break;
  case VTK_DOUBLE:
    if( this->Shift == (int)this->Shift && this->Scale == (int)this->Scale )
      {
      // I cannot consider that this is a problem, afterall a floating point type image
      // could in fact really be only integer type, only print a warning to inform dummy user
      vtkWarningMacro( "Image is floating point type, but rescale type is integer type. Rescaling anyway" );
      }
    /*
    Note to myself: should I allow people to squeeze into unsigned char ? Or can I assume most people
    will be doing unsigned short anyway...
    */
    pixeltype = gdcm::PixelFormat::FLOAT64;
    forcerescale = true;
    break;
  default:
    vtkErrorMacro( "Do not support this Pixel Type: " << scalarType );
    return 0;
    }

  gdcm::PhotometricInterpretation pi;
  if( this->ImageFormat )
    {
    // We have been passed the proper image format, let's use it !
    switch( this->ImageFormat )
      {
      case VTK_LUMINANCE:
        pi = gdcm::PhotometricInterpretation::MONOCHROME2;
        break;
      case VTK_RGB:
        pi = gdcm::PhotometricInterpretation::RGB;
        break;
      case VTK_RGBA:
        pi = gdcm::PhotometricInterpretation::ARGB;
        break;
      case VTK_INVERSE_LUMINANCE:
        pi = gdcm::PhotometricInterpretation::MONOCHROME1;
        break;
      case VTK_LOOKUP_TABLE:
        pi = gdcm::PhotometricInterpretation::PALETTE_COLOR;
        break;
      case VTK_YBR:
        pi = gdcm::PhotometricInterpretation::YBR_FULL;
        break;
      default:
        vtkErrorMacro( "Unknown ImageFormat:" << this->ImageFormat );
        return 0;
      }
    }
  else
    {
    // Attempt a guess
    if( data->GetNumberOfScalarComponents() == 1 )
      {
      pi = gdcm::PhotometricInterpretation::MONOCHROME2;
      }
    else if( data->GetNumberOfScalarComponents() == 3 )
      {
      // It could well be YBR ... oh well
      pi = gdcm::PhotometricInterpretation::RGB;
      // (0028,0006) US 0                                        #   2, 1 PlanarConfiguration
      }
    else if( data->GetNumberOfScalarComponents() == 4 )
      {
      // It could well be CMYK ... oh well
      pi = gdcm::PhotometricInterpretation::ARGB;
      }
    else
      {
      return 0;
      }
    }

  // Let's try to fake out the SOP Class UID here:
  gdcm::MediaStorage ms = gdcm::MediaStorage::SecondaryCaptureImageStorage;
  ms.GuessFromModality( this->MedicalImageProperties->GetModality(), this->FileDimensionality ); // Will override SC only if something is found...

  // store in a safe place the 'raw' pixeltype from vtk
  gdcm::PixelFormat savepixeltype = pixeltype;
  // fast path, when (shift,scale) is (0,1) we do not need no rescale function
  // at all. Pay attention in some case users really wants to store floating
  // point values anyway, be nice with them
  if( !forcerescale && (this->Shift == 0 && this->Scale == 1) )
    {
    //assert( pixeltype == outputpt );
    }
  else
    {
    gdcm::Rescaler ir2;
    ir2.SetIntercept( this->Shift );
    ir2.SetSlope( this->Scale );
    ir2.SetPixelFormat( pixeltype );
    // TODO: Hum...ScalarRange is -I believe- computed on the WholeExtent...
    vtkFloatingPointType srange[2];
    data->GetScalarRange(srange);
    // HACK !!!
    // MR Image Storage cannot have Shift / Rescale , however it looks like people are doing it
    // anyway, so let's make GDCM just as bad as any other library, by providing a fix:
    if( ms == gdcm::MediaStorage::MRImageStorage /*&& pixeltype.GetBitsAllocated() == 8*/ )
      {
      srange[1] = std::numeric_limits<uint16_t>::max() * this->Scale + this->Shift;
      }
    ir2.SetMinMaxForPixelType( srange[0], srange[1] );
    //gdcm::PixelFormat::ScalarType outputpt = ir2.ComputeInterceptSlopePixelType();
    gdcm::PixelFormat outputpt = ir2.ComputePixelTypeFromMinMax();
    // override pixeltype with what is found by Rescaler
    pixeltype = outputpt;
    }

  pixeltype.SetSamplesPerPixel( (unsigned short)data->GetNumberOfScalarComponents() );
  image.SetPhotometricInterpretation( pi );
  image.SetPixelFormat( pixeltype );
  image.SetPlanarConfiguration( 0 ); // VTK default

  // Setup LUT if any:
  if( pi == gdcm::PhotometricInterpretation::PALETTE_COLOR )
    {
    assert( pixeltype.GetSamplesPerPixel() == 1 );
    vtkLookupTable * vtklut = data->GetPointData()->GetScalars()->GetLookupTable();
    //vtkLookupTable * vtklut = this->LookupTable;
    assert( vtklut );
    //const char *name = vtklut->GetClassName ();
    vtkLookupTable16 * vtklut16 = vtkLookupTable16::SafeDownCast( vtklut );
    //assert( vtklut->GetNumberOfTableValues() == 256 );
    //vtkIdType vtknumcolors = vtklut->GetNumberOfTableValues();
    unsigned int lutlen = 256;
    assert( pixeltype.GetBitsAllocated() == 8 || pixeltype.GetBitsAllocated() == 16 );
    if( pixeltype.GetBitsAllocated() == 8 )
      {
      lutlen = 256;
      }
    else
      {
      //assert( pixeltype.GetBitsAllocated() == 16 );
      lutlen = 65536;
      }
    gdcm::SmartPointer<gdcm::LookupTable> lut = new gdcm::LookupTable;
    lut->Allocate( pixeltype.GetBitsAllocated() );
    lut->InitializeLUT( gdcm::LookupTable::RED, (unsigned short)lutlen, 0, 16 );
    lut->InitializeLUT( gdcm::LookupTable::GREEN, (unsigned short)lutlen, 0, 16 );
    lut->InitializeLUT( gdcm::LookupTable::BLUE, (unsigned short)lutlen, 0, 16 );
    bool b;
    if( vtklut16 )
      b = lut->WriteBufferAsRGBA( vtklut16->WritePointer(0,4) );
    else
      b = lut->WriteBufferAsRGBA( vtklut->WritePointer(0,4) );
    if( !b )
      {
      vtkWarningMacro( "Could not get values from LUT" );
      return 0;
      }

    image.SetLUT( *lut );
    }

  unsigned long len = image.GetBufferLength();
  vtkIdType npts = (vtkIdType)(inExt[5] - inExt[4] + 1) * (inExt[3] - inExt[2] + 1) * (inExt[1] - inExt[0] + 1);
  if( npts < 0 )
    {
    vtkErrorMacro( "Could not Get number of points" );
    return 0;
    }
  //assert( npts >= 0 );
  //assert( npts == data->GetNumberOfPoints() );
  int ssize = data->GetScalarSize();
  unsigned long vtklen = npts * ssize;
  if( ssize == 0 )
    {
    assert( data->GetScalarType() == VTK_BIT );
    vtklen = npts / 8;
    }
  else
    {
    vtklen = npts * ssize;
    assert( vtklen >= (unsigned long)npts );
    }
  //unsigned long vtklen = npts * ssize;
  //assert( vtklen == len * ssize );

  gdcm::DataElement pixeldata( gdcm::Tag(0x7fe0,0x0010) );
  gdcm::ByteValue *bv = new gdcm::ByteValue(); // (char*)data->GetScalarPointer(), len );
  bv->SetLength( (uint32_t)len ); // allocate !

//  std::ofstream of( "/tmp/bla.raw", std::ios::binary );
//  of.write( (char*)data->GetScalarPointer(), len);
//  of.close();
  // re shuffle the line within ByteValue:
  //
  char *pointer = (char*)bv->GetPointer();
  //const char *tempimage = (char*)data->GetScalarPointer();
  const char *tempimage = (char*)data->GetScalarPointerForExtent(inExt);
  //std::cerr << "Pointer:" << (unsigned int)tempimage << std::endl;
  int *dext = data->GetExtent();
  long outsize;
  if( data->GetScalarType() == VTK_BIT )
    {
    outsize = (dext[1] - dext[0] + 1) / 8;
    }
  else
    {
    outsize = pixeltype.GetPixelSize()*(dext[1] - dext[0] + 1);
    }
  int jj = dext[4];


  bool rescaled = false;
  char * copy = NULL;
  // Whenever shift / scale is needed... do it !
  if( this->Shift != 0 || this->Scale != 1 || forcerescale )
    {
    assert( this->PlanarConfiguration == 0 );
    // rescale from float to unsigned short
    gdcm::Rescaler ir;
    ir.SetIntercept( this->Shift );
    ir.SetSlope( this->Scale );
    ir.SetPixelFormat( savepixeltype );
    vtkFloatingPointType srange[2];
    data->GetScalarRange(srange);
    // HACK !!!
    // MR Image Storage cannot have Shift / Rescale , however it looks like people are doing it
    // anyway, so let's make GDCM just as bad as any other library, by providing a fix:
    if( ms == gdcm::MediaStorage::MRImageStorage /*&& pixeltype.GetBitsAllocated() == 8*/ )
      {
      srange[1] = std::numeric_limits<uint16_t>::max() * this->Scale + this->Shift;
      }
    ir.SetMinMaxForPixelType( srange[0], srange[1] );
    image.SetIntercept( this->Shift );
    image.SetSlope( this->Scale );
    copy = new char[len];
    ir.InverseRescale(copy,tempimage,vtklen);
    rescaled = true;
    tempimage = copy;
    }

  //std::cerr << "dext[4]:" << j << std::endl;
  //std::cerr << "inExt[4]:" << inExt[4] << std::endl;
  if( this->FileLowerLeft )
    {
    memcpy(pointer,tempimage,len);
    }
  else
    {
    if( dims[2] > 1 && this->FileDimensionality == 3 )
      {
      for(int j = dext[4]; j <= dext[5]; ++j)
        {
        for(int i = dext[2]; i <= dext[3]; ++i)
          {
          memcpy(pointer,
            tempimage+((dext[3] - i)+j*(dext[3]+1))*outsize, outsize);
          pointer += outsize;
          }
        }
      }
    else
      {
      for(int i = dext[2]; i <= dext[3]; ++i)
        {
        memcpy(pointer,
          tempimage+((dext[3] - i)+jj*(dext[3]+1))*outsize, outsize);
        pointer += outsize;
        }
      }
    }
  if( rescaled )
    {
    delete[] copy;
    }

  pixeldata.SetValue( *bv );
  image.SetDataElement( pixeldata );

// DEBUG
#ifndef NDEBUG
  const gdcm::DataElement &pixeldata2 = image.GetDataElement();
  //const gdcm::Value &v = image.GetValue();
  //const gdcm::ByteValue *bv1 = dynamic_cast<const gdcm::ByteValue*>(&v);
  const gdcm::ByteValue *bv1 = pixeldata2.GetByteValue();
  assert( bv1 && bv1 == bv );
  //image.Print( std::cerr );
#endif
// END DEBUG

  // Do PlanarConfiguration
  if( this->PlanarConfiguration )
    {
    gdcm::ImageChangePlanarConfiguration icpc;
    icpc.SetInput( image );
    icpc.SetPlanarConfiguration( 1 );
    icpc.Change();
    image = icpc.GetOutput();
    assert( image.GetPlanarConfiguration() == 1 );
    }


  int year, month, day;
  gdcm::File& file = writer.GetFile();
  gdcm::DataSet& ds = file.GetDataSet();
  vtkGDCMMedicalImageProperties *gdcmmip =
    dynamic_cast<vtkGDCMMedicalImageProperties*>( this->MedicalImageProperties );
  gdcm::Anonymizer ano;
  if( gdcmmip )
    {
    gdcm::File const &f = gdcmmip->GetFile(timeStep);
    writer.SetFile( f );
    ano.SetFile( writer.GetFile() );
    }
  else
    {
    ano.SetFile( file );
    // For ex: DICOM (0010,0010) = DOE,JOHN
    SetStringValueFromTag(this->MedicalImageProperties->GetPatientName(), gdcm::Tag(0x0010,0x0010), ano);
    // For ex: DICOM (0010,0020) = 1933197
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientID(), gdcm::Tag(0x0010,0x0020), ano);
    // For ex: DICOM (0010,1010) = 031Y
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientAge(), gdcm::Tag(0x0010,0x1010), ano);
    // For ex: DICOM (0010,0040) = M
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientSex(), gdcm::Tag(0x0010,0x0040), ano);
    // For ex: DICOM (0010,0030) = 19680427
    SetStringValueFromTag( this->MedicalImageProperties->GetPatientBirthDate(), gdcm::Tag(0x0010,0x0030), ano);
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
    // For ex: DICOM (0008,0020) = 20030617
    if( vtkMedicalImageProperties::GetDateAsFields( this->MedicalImageProperties->GetStudyDate(), year, month, day ) )
      SetStringValueFromTag( this->MedicalImageProperties->GetStudyDate(), gdcm::Tag(0x0008,0x0020), ano);
#endif
    // For ex: DICOM (0008,0022) = 20030617
    SetStringValueFromTag( this->MedicalImageProperties->GetAcquisitionDate(), gdcm::Tag(0x0008,0x0022), ano);
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
    // For ex: DICOM (0008,0030) = 162552.0705 or 230012, or 0012
#if 0
    int hour, minute, second;
    if( vtkMedicalImageProperties::GetTimeAsFields( this->MedicalImageProperties->GetStudyTime(), hour, minute, second ) )
#else
    time_t studytime;
    char date[22] = { ' ' };
    strcpy( date, "19000101" );
    if( this->MedicalImageProperties->GetStudyTime() )
      strncpy( date + 8 , this->MedicalImageProperties->GetStudyTime(), 22 - 8 );
    date[21] = 0;
    if( gdcm::System::ParseDateTime(studytime, date ) )
#endif
      SetStringValueFromTag( this->MedicalImageProperties->GetStudyTime(), gdcm::Tag(0x0008,0x0030), ano);
#endif
    // For ex: DICOM (0008,0032) = 162552.0705 or 230012, or 0012
    SetStringValueFromTag( this->MedicalImageProperties->GetAcquisitionTime(), gdcm::Tag(0x0008,0x0032), ano);
    // For ex: DICOM (0008,0023) = 20030617
    SetStringValueFromTag( this->MedicalImageProperties->GetImageDate(), gdcm::Tag(0x0008,0x0023), ano);
    // For ex: DICOM (0008,0033) = 162552.0705 or 230012, or 0012
    SetStringValueFromTag( this->MedicalImageProperties->GetImageTime(), gdcm::Tag(0x0008,0x0033), ano);
    // For ex: DICOM (0020,0013) = 1
    SetStringValueFromTag( this->MedicalImageProperties->GetImageNumber(), gdcm::Tag(0x0020,0x0013), ano);
    // For ex: DICOM (0020,0011) = 902
    SetStringValueFromTag( this->MedicalImageProperties->GetSeriesNumber(), gdcm::Tag(0x0020,0x0011), ano);
    // For ex: DICOM (0008,103e) = SCOUT
    SetStringValueFromTag( this->MedicalImageProperties->GetSeriesDescription(), gdcm::Tag(0x0008,0x103e), ano);
    // For ex: DICOM (0020,0010) = 37481
    SetStringValueFromTag( this->MedicalImageProperties->GetStudyID(), gdcm::Tag(0x0020,0x0010), ano);
    // For ex: DICOM (0008,1030) = BRAIN/C-SP/FACIAL
    SetStringValueFromTag( this->MedicalImageProperties->GetStudyDescription(), gdcm::Tag(0x0008,0x1030), ano);
    // For ex: DICOM (0008,0060)= CT
    SetStringValueFromTag( this->MedicalImageProperties->GetModality(), gdcm::Tag(0x0008,0x0060), ano);
    // For ex: DICOM (0008,0070) = Siemens
    SetStringValueFromTag( this->MedicalImageProperties->GetManufacturer(), gdcm::Tag(0x0008,0x0070), ano);
    // For ex: DICOM (0008,1090) = LightSpeed QX/i
    SetStringValueFromTag( this->MedicalImageProperties->GetManufacturerModelName(), gdcm::Tag(0x0008,0x1090), ano);
    // For ex: DICOM (0008,1010) = LSPD_OC8
    SetStringValueFromTag( this->MedicalImageProperties->GetStationName(), gdcm::Tag(0x0008,0x1010), ano);
    // For ex: DICOM (0008,0080) = FooCity Medical Center
    SetStringValueFromTag( this->MedicalImageProperties->GetInstitutionName(), gdcm::Tag(0x0008,0x0080), ano);
    // For ex: DICOM (0018,1210) = Bone
    SetStringValueFromTag( this->MedicalImageProperties->GetConvolutionKernel(), gdcm::Tag(0x0018,0x1210), ano);
    // For ex: DICOM (0018,0050) = 0.273438
    SetStringValueFromTag( this->MedicalImageProperties->GetSliceThickness(), gdcm::Tag(0x0018,0x0050), ano);
    // For ex: DICOM (0018,0060) = 120
    SetStringValueFromTag( this->MedicalImageProperties->GetKVP(), gdcm::Tag(0x0018,0x0060), ano);
    // For ex: DICOM (0018,1120) = 15
    SetStringValueFromTag( this->MedicalImageProperties->GetGantryTilt(), gdcm::Tag(0x0018,0x1120), ano);
    // For ex: DICOM (0018,0081) = 105
    SetStringValueFromTag( this->MedicalImageProperties->GetEchoTime(), gdcm::Tag(0x0018,0x0081), ano);
    // For ex: DICOM (0018,0091) = 35
    SetStringValueFromTag( this->MedicalImageProperties->GetEchoTrainLength(), gdcm::Tag(0x0018,0x0091), ano);
    // For ex: DICOM (0018,0080) = 2040
    SetStringValueFromTag( this->MedicalImageProperties->GetRepetitionTime(), gdcm::Tag(0x0018,0x0080), ano);
    // For ex: DICOM (0018,1150) = 5
    SetStringValueFromTag( this->MedicalImageProperties->GetExposureTime(), gdcm::Tag(0x0018,0x1150), ano);
    // For ex: DICOM (0018,1151) = 400
    SetStringValueFromTag( this->MedicalImageProperties->GetXRayTubeCurrent(), gdcm::Tag(0x0018,0x1151), ano);
    // For ex: DICOM (0018,1152) = 114
    SetStringValueFromTag( this->MedicalImageProperties->GetExposure(), gdcm::Tag(0x0018,0x1152), ano);

    // Window Level / Window Center
    int numwl = this->MedicalImageProperties->GetNumberOfWindowLevelPresets();
    if( numwl )
      {
      gdcm::VR vr = gdcm::VR::DS;
      gdcm::Element<gdcm::VR::DS,gdcm::VM::VM1_n> elwc;
      elwc.SetLength( numwl * vr.GetSizeof() );
      gdcm::Element<gdcm::VR::DS,gdcm::VM::VM1_n> elww;
      elww.SetLength( numwl * vr.GetSizeof() );
      vr = gdcm::VR::LO;
      gdcm::Element<gdcm::VR::LO,gdcm::VM::VM1_n> elwe;
      elwe.SetLength( numwl * vr.GetSizeof() );
      for(int i = 0; i < numwl; ++i)
        {
        const double *wl = this->MedicalImageProperties->GetNthWindowLevelPreset(i);
        elww.SetValue( wl[0], i );
        elwc.SetValue( wl[1], i );
        const char* we = this->MedicalImageProperties->GetNthWindowLevelPresetComment(i);
        elwe.SetValue( we, i );
        }
        {
        gdcm::DataElement de = elwc.GetAsDataElement();
        de.SetTag( gdcm::Tag(0x0028,0x1050) );
        ds.Insert( de );
        }
        {
        gdcm::DataElement de = elww.GetAsDataElement();
        de.SetTag( gdcm::Tag(0x0028,0x1051) );
        ds.Insert( de );
        }
        {
        gdcm::DataElement de = elwe.GetAsDataElement();
        de.SetTag( gdcm::Tag(0x0028,0x1055) );
        ds.Insert( de );
        }
      }
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 0 )
    // User defined value
    // Remap any user defined value from the DICOM name to the DICOM tag
    unsigned int nvalues = this->MedicalImageProperties->GetNumberOfUserDefinedValues();
    for(unsigned int i = 0; i < nvalues; ++i)
      {
      const char *name = this->MedicalImageProperties->GetUserDefinedNameByIndex(i);
      const char *value = this->MedicalImageProperties->GetUserDefinedValueByIndex(i);
      assert( name && value && *name && *value );
      // Only deal with public elements:
      const gdcm::Global& g = gdcm::Global::GetInstance();
      const gdcm::Dicts &dicts = g.GetDicts();
      const gdcm::Dict &pubdict = dicts.GetPublicDict();
      gdcm::Tag t;
      // Lookup up tag by name is truly inefficient : 0(n)
      const gdcm::DictEntry &de = pubdict.GetDictEntryByName(name, t); (void)de;
      SetStringValueFromTag( value, t, ano);
      }
#endif
    }

  ms = gdcm::ImageHelper::ComputeMediaStorageFromModality(
    this->MedicalImageProperties->GetModality(), this->FileDimensionality,
    pixeltype, pi, this->Shift, this->Scale );
  if( ms == gdcm::MediaStorage::MS_END )
    {
    vtkErrorMacro( "Incompatible MediaStorage" );
    return 0;
    }


  // FIXME: new Secondary object handle multi frames...
  assert( gdcm::MediaStorage::IsImage( ms ) );
    {
    gdcm::DataElement de( gdcm::Tag(0x0008, 0x0016) );
    const char* msstr = gdcm::MediaStorage::GetMSString(ms);
    assert( msstr );
    de.SetByteValue( msstr, (uint32_t)strlen(msstr) );
    de.SetVR( gdcm::Attribute<0x0008, 0x0016>::GetVR() );
    ds.Insert( de );
    }

  // Image Type is pretty much always required:
  gdcm::Attribute<0x0008,0x0008> imagetype;
  const gdcm::CSComp values[] = { "ORIGINAL", "PRIMARY" };
  imagetype.SetValues( values, 2 );
  ds.Insert( imagetype.GetAsDataElement() );

  // Image Orientation (Patient)
  //gdcm::Attribute<0x0020,0x0037> iop = {{1,0,0,0,1,0}}; // default value
  std::vector<double> iop;
  iop.resize(6);
  const vtkMatrix4x4 *dircos = this->DirectionCosines;
  for(int i = 0; i < 3; ++i)
    {
    iop[i] = dircos->GetElement(i,0);
    }

  for(int i = 0; i < 3; ++i)
    {
    iop[i+3] = dircos->GetElement(i,1);
    }
#if VTK_MAJOR_VERSION >= 6 || ( VTK_MAJOR_VERSION == 5 && VTK_MINOR_VERSION > 2 )
   const double *iop_mip = this->MedicalImageProperties->GetDirectionCosine();
   if( iop[0] != iop_mip[0]
    || iop[1] != iop_mip[1]
    || iop[2] != iop_mip[2]
    || iop[3] != iop_mip[3]
    || iop[4] != iop_mip[4]
    || iop[5] != iop_mip[5]
   )
     {
     vtkErrorMacro( "DirectionCosines is not compatible with vtkMedicalImageProperties::DirectionCosine" );
     return 0;
     }
#endif

  image.SetDirectionCosines( &iop[0] );

  std::vector<double> ipp;
  ipp.resize(3);
  // Image Position (Patient)
  // cross product of direction cosines gives the direction along
  // which the slices are stacked
  const double *iop1 = &iop[0];
  const double *iop2 = iop1+3;
  double zaxis[3];
  vtkMath::Cross(iop1, iop2, zaxis);

  // determine the relative index of the current slice
  // in the case of a single volume, this will be 0
  // since inExt (UpdateExtent) and WholeExt are the same
  int n = inExt[4] - inWholeExt[4];
  const vtkFloatingPointType *vtkorigin = data->GetOrigin();
  vtkFloatingPointType origin[3];
  if( this->FileLowerLeft )
    {
    origin[0] = vtkorigin[0];
    origin[1] = vtkorigin[1];
    origin[2] = vtkorigin[2];
    }
  else
    {
    double norm = (dims[1] - 1) * spacing[1];
    origin[0] = vtkorigin[0] - norm * iop[3+0];
    origin[1] = vtkorigin[1] - norm * iop[3+1];
    origin[2] = vtkorigin[2] - norm * iop[3+2];
    }
  double new_origin[3];
  // In order to compute the newer Image Position (Patient) we need to have a valid spacing along
  // 3rd dimension. Simply give up in case 0:
  // FIXME: Actually if user decides to write a series of SC object it is ok...
  if( spacing[2] == 0. && dims[2] > 1 )
    {
    vtkErrorMacro( "Z-spacing cannot be 0 for multiframe image" );
    return 0;
    }
  for (int i = 0; i < 3; i++)
    {
    // the n'th slice is n * z-spacing aloung the IOP-derived
    // z-axis
    new_origin[i] = origin[i] + zaxis[i] * n * spacing[2];
    }

  for(int i = 0; i < 3; ++i)
    ipp[i] = new_origin[i];

  image.SetOrigin(0, ipp[0] );
  image.SetOrigin(1, ipp[1] );
  image.SetOrigin(2, ipp[2] );
  assert( ipp.size() < 3 || image.GetOrigin(2) == ipp[2] );
  //gdcm::ImageHelper::SetOriginValue(ds, ipp, dims[2], spacing[2]);


  // Here come the important part: generate proper UID for Series/Study so that people knows this is the same Study/Series
  const char *studyuid = this->StudyUID;
  assert( studyuid ); // programmer error
    {
    gdcm::DataElement de( gdcm::Tag(0x0020,0x000d) ); // Study
    de.SetByteValue( studyuid, (uint32_t)strlen(studyuid) );
    de.SetVR( gdcm::Attribute<0x0020, 0x000d>::GetVR() );
    ds.Insert( de );
    }
  const char *seriesuid = this->SeriesUID;
  assert( seriesuid ); // programmer error
    {
    gdcm::DataElement de( gdcm::Tag(0x0020,0x000e) ); // Series
    de.SetByteValue( seriesuid, (uint32_t)strlen(seriesuid) );
    de.SetVR( gdcm::Attribute<0x0020, 0x000e>::GetVR() );
    ds.Insert( de );
    }

  const char *filename = NULL;
  int k = inExt[4];
  if( this->FileNames->GetNumberOfValues() )
    {
    //int n = this->FileNames->GetNumberOfValues();
    filename = this->FileNames->GetValue(k);
    }
  else
    {
    filename = this->GetFileName();
    }
  assert( filename );

  // Let's add an Instance Number just for fun, unless we have a vtkGDCMMedicalImageProperties
  if( !gdcmmip )
    {
    std::ostringstream os;
    os << k;
    // Will only be added if none found
    SetStringValueFromTag(os.str().c_str(), gdcm::Tag(0x0020,0x0013), ano);
    }

  switch( this->CompressionType )
    {
    /*
     * 10.1 DICOM DEFAULT TRANSFER SYNTAX
     *  DICOM defines a default Transfer Syntax, the DICOM Implicit VR Little Endian Transfer Syntax (UID =
     *  "1.2.840.10008.1.2"), which shall be supported by every conformant DICOM Implementation.
     */
    case NO_COMPRESSION:
      change.SetTransferSyntax( gdcm::TransferSyntax::ImplicitVRLittleEndian );
      break;
    case JPEG_COMPRESSION:
      change.SetTransferSyntax( gdcm::TransferSyntax::JPEGLosslessProcess14_1 );
      break;
    case JPEG2000_COMPRESSION:
      change.SetTransferSyntax( gdcm::TransferSyntax::JPEG2000Lossless );
      break;
    case JPEGLS_COMPRESSION:
      change.SetTransferSyntax( gdcm::TransferSyntax::JPEGLSLossless );
      break;
    case RLE_COMPRESSION:
      change.SetTransferSyntax( gdcm::TransferSyntax::RLELossless );
      break;
    }
  if( !change.Change() )
    {
    vtkErrorMacro( "Could not change the Transfer Syntax for Compression Type: " );
    return 0;
    }
  writer.SetImage( change.GetOutput() );
  writer.SetFileName( filename );
  if( !writer.Write() )
    {
    vtkErrorMacro( "Could not write" );
    return 0;
    }

  return 1;
}

//void vtkGDCMImageWriter::SetCompressionTypeFromString(const char *)
//{
//}
//
//const char *vtkGDCMImageWriter::GetCompressionTypeAsString()
//{
//    NO_COMPRESSION = 0,   // raw (default)
//    JPEG_COMPRESSION,     // JPEG
//    JPEG2000_COMPRESSION, // J2K
//    JPEGLS_COMPRESSION,   // JPEG-LS
//    RLE_COMPRESSION       // RLE
//}
void vtkGDCMImageWriter::SetDirectionCosinesFromImageOrientationPatient(const double dircos[6])
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
}

//----------------------------------------------------------------------------
void vtkGDCMImageWriter::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os,indent);
}
