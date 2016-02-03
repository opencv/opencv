/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmMediaStorage.h"
#include "gdcmTag.h"
#include "gdcmByteValue.h"
#include "gdcmDataSet.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmFile.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmCodeString.h"

namespace gdcm
{

static const char *MSStrings[] = {
  "1.2.840.10008.1.3.10",
  "1.2.840.10008.5.1.4.1.1.1",
  "1.2.840.10008.5.1.4.1.1.1.1",
  "1.2.840.10008.5.1.4.1.1.1.1.1",
  "1.2.840.10008.5.1.4.1.1.1.2",
  "1.2.840.10008.5.1.4.1.1.1.2.1",
  "1.2.840.10008.5.1.4.1.1.1.3",
  "1.2.840.10008.5.1.4.1.1.1.3.1",
  "1.2.840.10008.5.1.4.1.1.2",
  "1.2.840.10008.5.1.4.1.1.2.1",
  "1.2.840.10008.5.1.4.1.1.6",
  "1.2.840.10008.5.1.4.1.1.6.1",  // Ultrasound Image Storage
  "1.2.840.10008.5.1.4.1.1.3",
  "1.2.840.10008.5.1.4.1.1.3.1",
  "1.2.840.10008.5.1.4.1.1.4",
  "1.2.840.10008.5.1.4.1.1.4.1",
  "1.2.840.10008.5.1.4.1.1.4.2",
  "1.2.840.10008.5.1.4.1.1.5",     // NuclearMedicineImageStorageRetired
  "1.2.840.10008.5.1.4.1.1.7",
  "1.2.840.10008.5.1.4.1.1.7.1",
  "1.2.840.10008.5.1.4.1.1.7.2",
  "1.2.840.10008.5.1.4.1.1.7.3",
  "1.2.840.10008.5.1.4.1.1.7.4",
  "1.2.840.10008.5.1.4.1.1.8",
  "1.2.840.10008.5.1.4.1.1.9",
  "1.2.840.10008.5.1.4.1.1.9.1.1",
  "1.2.840.10008.5.1.4.1.1.9.1.2",
  "1.2.840.10008.5.1.4.1.1.9.1.3",
  "1.2.840.10008.5.1.4.1.1.9.2.1",
  "1.2.840.10008.5.1.4.1.1.9.3.1",
  "1.2.840.10008.5.1.4.1.1.9.4.1",
  "1.2.840.10008.5.1.4.1.1.10",
  "1.2.840.10008.5.1.4.1.1.11",
  "1.2.840.10008.5.1.4.1.1.11.1",
  "1.2.840.10008.5.1.4.1.1.12.1",
  "1.2.840.10008.5.1.4.1.1.12.2",
  "1.2.840.10008.5.1.4.1.1.12.3",
  "1.2.840.10008.5.1.4.1.1.20",
  "1.2.840.10008.5.1.4.1.1.66",
  "1.2.840.10008.5.1.4.1.1.66.1",
  "1.2.840.10008.5.1.4.1.1.66.2",

  // See PETAt001_PT001.dcm
  "1.2.840.10008.5.1.4.1.1.128",
  // SYNGORTImage.dcm
  "1.2.840.10008.5.1.4.1.1.481.1",
  // eclipse_dose.dcm
  "1.2.840.10008.5.1.4.1.1.481.2",
  // exRT_Structure_Set_Storage.dcm
  "1.2.840.10008.5.1.4.1.1.481.3",
  // eclipse_plan.dcm
  "1.2.840.10008.5.1.4.1.1.481.5",
  // exCSA_Non-Image_Storage.dcm
  "1.3.12.2.1107.5.9.1",
  // 3DDCM011.dcm
  "1.2.840.113543.6.6.1.3.10002",
  // EnhancedSR
  "1.2.840.10008.5.1.4.1.1.88.22",
  // BasicTextSR:
  "1.2.840.10008.5.1.4.1.1.88.11",
  // HardcopyGrayscaleImageStorage
  "1.2.840.10008.5.1.1.29",
  // ComprehensiveSR
  "1.2.840.10008.5.1.4.1.1.88.33",
  // DetachedStudyManagementSOPClass,
  "1.2.840.10008.3.1.2.3.1",
  // EncapsulatedPDFStorage
  "1.2.840.10008.5.1.4.1.1.104.1",
  // EncapsulatedCDAStorage
  "1.2.840.10008.5.1.4.1.1.104.2",
  // StudyComponentManagementSOPClass
  "1.2.840.10008.3.1.2.3.2",
  // DetachedVisitManagementSOPClass
  "1.2.840.10008.3.1.2.2.1",
  // DetachedPatientManagementSOPClass
  "1.2.840.10008.3.1.2.1.1",
  // VideoEndoscopicImageStorage
  "1.2.840.10008.5.1.4.1.1.77.1.1.1",
  // GeneralElectricMagneticResonanceImageStorage
  "1.2.840.113619.4.2",
  // GEPrivate3DModelStorage
  "1.2.840.113619.4.26",
  // Toshiba Private Data Storage
  "1.2.392.200036.9116.7.8.1.1.1",
  // MammographyCADSR,
  "1.2.840.10008.5.1.4.1.1.88.50",
  // KeyObjectSelectionDocument
  "1.2.840.10008.5.1.4.1.1.88.59",
  // HangingProtocolStorage
  "1.2.840.10008.5.1.4.38.1",
  // Modality Performed Procedure Step SOP Class
  "1.2.840.10008.3.1.2.3.3",
  // Philips Private MR Synthetic Image Storage
  "1.3.46.670589.5.0.10",
  "1.2.840.10008.5.1.4.1.1.77.1.4", // "VL Photographic Image Storage",
  "1.2.840.10008.5.1.4.1.1.66.4", // Segmentation Storage
  "1.2.840.10008.5.1.4.1.1.481.8", // RT Ion Plan Storage
  "1.2.840.10008.5.1.4.1.1.13.1.1", // XRay3DAngiographicImageStorage,
  "1.2.840.10008.5.1.4.1.1.12.1.1", // Enhanced XA Image Storage
  "1.2.840.10008.5.1.4.1.1.481.9", //  RTIonBeamsTreatmentRecordStorage
  "1.2.840.10008.5.1.4.1.1.66.5", // Surface Segmentation Storage
  "1.2.840.10008.5.1.4.1.1.77.1.6", // VLWholeSlideMicroscopyImageStorage
  "1.2.840.10008.5.1.4.1.1.481.7", // RTTreatmentSummaryRecordStorage
  "1.2.840.10008.5.1.4.1.1.6.2", // EnhancedUSVolumeStorage
  "1.2.840.10008.5.1.4.1.1.88.67", // XRayRadiationDoseSR
  "1.2.840.10008.5.1.4.1.1.77.1.1", // VLEndoscopicImageStorage
  "1.2.840.10008.5.1.4.1.1.13.1.3", // BreastTomosynthesisImageStorage
  "1.2.392.200036.9125.1.1.2",  // FujiPrivateCRImageStorage
  "1.2.840.10008.5.1.4.1.1.77.1.5.1", // OphthalmicPhotography8BitImageStorage
  "1.2.840.10008.5.1.4.1.1.77.1.5.4", // OphthalmicTomographyImageStorage
  "1.2.840.10008.5.1.4.1.1.77.1.2",   // VL Microscopic Image Storage
  0
};

MediaStorage::MSType MediaStorage::GetMSType(const char *str)
{
  if(!str) return MS_END;

  for(unsigned int i = 0; MSStrings[i] != 0; ++i)
    {
    if( strcmp(str, MSStrings[i]) == 0 )
      {
      return (MSType)i;
      }
    }
  // Ouch ! We did not find anything, that's pretty bad, let's hope that
  // the toolkit which wrote the image is buggy and tolerate space padded binary
  // string
  CodeString codestring = str;
  std::string cs = codestring.GetAsString();
  for(unsigned int i = 0; MSStrings[i] != 0; ++i)
    {
    if( strcmp(cs.c_str(), MSStrings[i]) == 0 )
      {
      return (MSType)i;
      }
    }

  //assert(0);
  return MS_END;
}

const char* MediaStorage::GetMSString(MSType ms)
{
  assert( ms <= MS_END );
  return MSStrings[(int)ms];
}

const char* MediaStorage::GetString() const
{
  return GetMSString(MSField);
}

// FIXME
// Currently the implementation is bogus it only define the TS which
// are associated with an image so indeed the implementation of IsImage
// is only the verification of TSType is != TS_END
bool MediaStorage::IsImage(MSType ms)
{
  if ( ms == MS_END // most frequent first
    // lexicographical order then...
    || ms == BasicVoiceAudioWaveformStorage
    || ms == CSANonImageStorage
    || ms == HemodynamicWaveformStorage
    || ms == MediaStorageDirectoryStorage
    || ms == RTPlanStorage
    || ms == GrayscaleSoftcopyPresentationStateStorageSOPClass
    || ms == CardiacElectrophysiologyWaveformStorage
    || ms == ToshibaPrivateDataStorage // not an image I think...
    || ms == EnhancedSR
    || ms == BasicTextSR
    || ms == ComprehensiveSR
    || ms == StudyComponentManagementSOPClass
    || ms == DetachedVisitManagementSOPClass
    || ms == DetachedStudyManagementSOPClass
    || ms == EncapsulatedPDFStorage
    || ms == EncapsulatedCDAStorage
    || ms == XRayRadiationDoseSR
    || ms == KeyObjectSelectionDocument
    || ms == HangingProtocolStorage
    || ms == MRSpectroscopyStorage
    || ms == ModalityPerformedProcedureStepSOPClass
    || ms == RawDataStorage
    || ms == RTIonPlanStorage
    || ms == LeadECGWaveformStorage
    || ms == GeneralECGWaveformStorage
    || ms == RTIonBeamsTreatmentRecordStorage
    || ms == RTStructureSetStorage
    || ms == MammographyCADSR
    || ms == SurfaceSegmentationStorage )
    {
    return false;
    }
  return true;
}

struct MSModalityType
{
  const char *Modality;
  const unsigned char Dimension;
  const bool Retired;
};

static const MSModalityType MSModalityTypes[] = {
  {"00", 0, 0},//MediaStorageDirectoryStorage,
  {"CR", 2, 0},//ComputedRadiographyImageStorage,
  {"DX", 2, 0},//DigitalXRayImageStorageForPresentation,
  {"DX", 2, 0},//DigitalXRayImageStorageForProcessing,
  {"  ", 2, 0},//DigitalMammographyImageStorageForPresentation,
  {"MG", 2, 0},//DigitalMammographyImageStorageForProcessing,
  {"  ", 2, 0},//DigitalIntraoralXrayImageStorageForPresentation,
  {"  ", 2, 0},//DigitalIntraoralXRayImageStorageForProcessing,
  {"CT", 2, 0},//CTImageStorage,
  {"CT", 3, 0},//EnhancedCTImageStorage,
  {"US", 2, 1},//UltrasoundImageStorageRetired,
  {"US", 2, 0},//UltrasoundImageStorage,
  {"US", 3, 1},//UltrasoundMultiFrameImageStorageRetired,
  {"US", 3, 0},//UltrasoundMultiFrameImageStorage,
  {"MR", 2, 0},//MRImageStorage,
  {"MR", 3, 0},//EnhancedMRImageStorage,
  {"MR", 2, 0},//MRSpectroscopyStorage,
  {"  ", 2, 1},//NuclearMedicineImageStorageRetired,
  {"OT", 2, 0},//SecondaryCaptureImageStorage,
  {"OT", 3, 0},//MultiframeSingleBitSecondaryCaptureImageStorage,
  {"OT", 3, 0},//MultiframeGrayscaleByteSecondaryCaptureImageStorage,
  {"OT", 3, 0},//MultiframeGrayscaleWordSecondaryCaptureImageStorage,
  {"OT", 3, 0},//MultiframeTrueColorSecondaryCaptureImageStorage,
  {"  ", 2, 0},//StandaloneOverlayStorage,
  {"  ", 2, 0},//StandaloneCurveStorage,
  {"  ", 2, 0},//LeadECGWaveformStorage, // 12-
  {"  ", 2, 0},//GeneralECGWaveformStorage,
  {"  ", 2, 0},//AmbulatoryECGWaveformStorage,
  {"  ", 2, 0},//HemodynamicWaveformStorage,
  {"  ", 2, 0},//CardiacElectrophysiologyWaveformStorage,
  {"  ", 2, 0},//BasicVoiceAudioWaveformStorage,
  {"  ", 2, 0},//StandaloneModalityLUTStorage,
  {"  ", 2, 0},//StandaloneVOILUTStorage,
  {"  ", 2, 0},//GrayscaleSoftcopyPresentationStateStorageSOPClass,
  {"XA", 3, 0},//XRayAngiographicImageStorage,
  {"RF", 2, 0},//XRayRadiofluoroscopingImageStorage,
  {"  ", 2, 0},//XRayAngiographicBiPlaneImageStorageRetired,
  {"NM", 3, 0},//NuclearMedicineImageStorage,
  {"  ", 2, 0},//RawDataStorage,
  {"  ", 2, 0},//SpacialRegistrationStorage,
  {"  ", 2, 0},//SpacialFiducialsStorage,
  {"PT", 2, 0},//PETImageStorage,
  {"RTIMAGE ", 2, 0},//RTImageStorage, // FIXME
  {"RTDOSE", 3, 0},//RTDoseStorage,
  {"  ", 2, 0},//RTStructureSetStorage,
  {"  ", 2, 0},//RTPlanStorage,
  {"  ", 2, 0},//CSANonImageStorage,
  {"  ", 2, 0},//Philips3D,
  {"  ", 2, 0},//EnhancedSR
  {"  ", 2, 0},//BasicTextSR
  {"  ", 2, 0},//HardcopyGrayscaleImageStorage
  {"  ", 2, 0},//ComprehensiveSR
  {"  ", 2, 0},//DetachedStudyManagementSOPClass
  {"  ", 2, 0},//EncapsulatedPDFStorage
  {"  ", 2, 0},//EncapsulatedCDAStorage
  {"  ", 2, 0},//StudyComponentManagementSOPClass
  {"  ", 2, 0},//DetachedVisitManagementSOPClass
  {"  ", 2, 0},//DetachedPatientManagementSOPClass
  {"ES", 3, 0},//VideoEndoscopicImageStorage
  {"  ", 2, 0},//GeneralElectricMagneticResonanceImageStorage
  {"  ", 2, 0},//GEPrivate3DModelStorage
  {"  ", 2, 0},//ToshibaPrivateDataStorage
  {"  ", 2, 0},//MammographyCADSR
  {"  ", 2, 0},//KeyObjectSelectionDocument
  {"  ", 2, 0},//HangingProtocolStorage
  {"  ", 2, 0},//ModalityPerformedProcedureStepSOPClass
  {"  ", 2, 0},//PhilipsPrivateMRSyntheticImageStorage
  {"XC", 2, 0},//VLPhotographicImageStorage
  {"SEG ", 3, 0},// Segmentation Storage
  {"  ", 2, 0},// RT Ion Plan Storage
  {"XA", 3, 0},// XRay3DAngiographicImageStorage,
  {"XA", 3, 0},// Enhanced XA Image Storage
  {"  ", 2, 0},// RTIonBeamsTreatmentRecordStorage
  {"SEG", 3, 0},// Surface Segmentation Storage
  {"SM", 2, 0},// VLWholeSlideMicroscopyImageStorage
  {"RTRECORD", 2, 0},//RTTreatmentSummaryRecordStorage
  {"US", 3, 0},// EnhancedUSVolumeStorage
  {"  ", 2, 0},// XRayRadiationDoseSR
  {"ES", 2, 0},// VLEndoscopicImageStorage
  {"MG", 3, 0},// BreastTomosynthesisImageStorage
  {"CR", 2, 0},// FujiPrivateCRImageStorage
  {"OP", 2, 0},// OphthalmicPhotography8BitImageStorage
  {"OPT", 3, 0},// OphthalmicTomographyImageStorage
  {"GM", 3, 0},// VLMicroscopicImageStorage

  {NULL, 0, 0} //MS_END
};

unsigned int MediaStorage::GetNumberOfMSType()
{
  const unsigned int n = MS_END;
  assert( n > 0 );
  return n;
}

unsigned int MediaStorage::GetNumberOfMSString()
{
  static const unsigned int n = sizeof( MSStrings ) / sizeof( *MSStrings );
  assert( n > 0 );
  return n - 1;
}

unsigned int MediaStorage::GetNumberOfModality()
{
  static const unsigned int n = sizeof( MSModalityTypes ) / sizeof( *MSModalityTypes );
  assert( n > 0 );
  return n - 1;
}

const char *MediaStorage::GetModality() const
{
  if (!MSModalityTypes[MSField].Modality)
    return NULL;
  assert( MSModalityTypes[MSField].Modality[0] != ' ' ); // FIXME
  return MSModalityTypes[MSField].Modality;
}

unsigned int MediaStorage::GetModalityDimension() const
{
  if (!MSModalityTypes[MSField].Modality)
    return 0;
  assert( MSModalityTypes[MSField].Dimension );
  return MSModalityTypes[MSField].Dimension;
}

void MediaStorage::GuessFromModality(const char *modality, unsigned int dim )
{
  // no default value is set, it is up to the user to decide initial value
  if( !modality || !dim ) return;
  //if( strlen(modality) != 2 ) return;
  int i = 0;
  while( MSModalityTypes[i].Modality &&
    (strcmp(modality, MSModalityTypes[i].Modality) != 0
     || MSModalityTypes[i].Retired
     || MSModalityTypes[i].Dimension < dim ))
    {
    ++i;
    }
  if( MSModalityTypes[i].Modality )
    {
    // Ok we found something...
    MSField = (MSType)i;
    }
}

const char* MediaStorage::GetFromDataSetOrHeader(DataSet const &ds, const Tag & tag)
{
  static std::string ret;
  if( ds.FindDataElement( tag ) )
    {
    const ByteValue *sopclassuid = ds.GetDataElement( tag ).GetByteValue();
    // Empty SOP Class UID:
    // lifetechmed/A0038329.DCM
    if( !sopclassuid || !sopclassuid->GetPointer() ) return 0;
    std::string sopclassuid_str(
      sopclassuid->GetPointer(),
      sopclassuid->GetLength() );
    if( sopclassuid_str.find( ' ' ) != std::string::npos )
      {
      gdcmWarningMacro( "UI contains a space character discarding" );
      std::string::size_type pos = sopclassuid_str.find_last_of(' ');
      sopclassuid_str = sopclassuid_str.substr(0,pos);
      }
    ret = sopclassuid_str.c_str();
    return ret.c_str();
    }
  return 0;
}

bool MediaStorage::SetFromDataSetOrHeader(DataSet const &ds, const Tag & tag)
{
  const char * ms_str = GetFromDataSetOrHeader(ds,tag);
  if( ms_str )
    {
    MediaStorage ms = MediaStorage::GetMSType(ms_str);
    MSField = ms;
    if( ms == MS_END )
      {
      // weird something was found, but we not find the MS anyway...
      gdcmWarningMacro( "Does not know what: " << ms_str << " is..." );
      }
    return true;
    }
  return false;
}

const char* MediaStorage::GetFromHeader(FileMetaInformation const &fmi)
{
  const Tag tmediastoragesopclassuid(0x0002, 0x0002);
  return GetFromDataSetOrHeader(fmi, tmediastoragesopclassuid);
}

bool MediaStorage::SetFromHeader(FileMetaInformation const &fmi)
{
  const Tag tmediastoragesopclassuid(0x0002, 0x0002);
  return SetFromDataSetOrHeader(fmi, tmediastoragesopclassuid);
}

const char* MediaStorage::GetFromDataSet(DataSet const &ds)
{
  const Tag tsopclassuid(0x0008, 0x0016);
  return GetFromDataSetOrHeader(ds, tsopclassuid);
}


bool MediaStorage::SetFromDataSet(DataSet const &ds)
{
  const Tag tsopclassuid(0x0008, 0x0016);
  return SetFromDataSetOrHeader(ds, tsopclassuid);
}

void MediaStorage::SetFromSourceImageSequence(DataSet const &ds)
{
  const Tag sourceImageSequenceTag(0x0008,0x2112);
  if( ds.FindDataElement( sourceImageSequenceTag ) )
    {
    const DataElement &sourceImageSequencesq = ds.GetDataElement( sourceImageSequenceTag );
    //const SequenceOfItems* sq = sourceImageSequencesq.GetSequenceOfItems();
    SmartPointer<SequenceOfItems> sq = sourceImageSequencesq.GetValueAsSQ();
    if( !sq ) return;
    SequenceOfItems::ConstIterator it = sq->Begin();
    const DataSet &subds = it->GetNestedDataSet();
    // (0008,1150) UI =MRImageStorage                          #  26, 1 ReferencedSOPClassUID
    const Tag referencedSOPClassUIDTag(0x0008,0x1150);
    if( subds.FindDataElement( referencedSOPClassUIDTag ) )
      {
      const DataElement& de = subds.GetDataElement( referencedSOPClassUIDTag );
      const ByteValue *sopclassuid = de.GetByteValue();
      // LEADTOOLS_FLOWERS-8-PAL-Uncompressed.dcm
      //assert( sopclassuid );
      if( sopclassuid )
        {
        std::string sopclassuid_str(
          sopclassuid->GetPointer(),
          sopclassuid->GetLength() );
        if( sopclassuid_str.find( ' ' ) != std::string::npos )
          {
          gdcmWarningMacro( "UI contains a space character discarding" );
          std::string::size_type pos = sopclassuid_str.find_last_of(' ');
          sopclassuid_str = sopclassuid_str.substr(0,pos);
          }
        MediaStorage ms = MediaStorage::GetMSType(sopclassuid_str.c_str());
        assert( ms != MS_END );
        MSField = ms;
        }
      }
    }
}

bool MediaStorage::SetFromModality(DataSet const &ds)
{
  // Ok let's try againg with little luck it contains a pixel data...
  if( ds.FindDataElement( Tag(0x7fe0,0x0010) ) )
    {
    // Pixel Data found !
    // Attempt to recover from the modality (0008,0060):
    if( ds.FindDataElement( Tag(0x0008,0x0060) ) )
      {
      // gdcm-CR-DCMTK-16-NonSamplePerPix.dcm
      // Someone defined the Transfer Syntax but I have no clue what
      // it is. Since there is Pixel Data element, let's try to read
      // that as a buggy DICOM Image file...
      const ByteValue *bv = ds.GetDataElement( Tag(0x0008,0x0060) ).GetByteValue();
      if( bv )
        {
        std::string modality = std::string( bv->GetPointer(), bv->GetLength() );
        GuessFromModality( modality.c_str() );
        }
      }
    // We know there is a Pixel Data element, so make sure not to return without a default
    // to SC Object:
    if( MSField == MediaStorage::MS_END )
      {
      gdcmWarningMacro( "Unknown/Unhandle MediaStorage, but Pixel Data element found" );
      // BUG: Need to check Col*Row*Bit*NSample == PixelSize (uncompressed)
      MSField = MediaStorage::SecondaryCaptureImageStorage;
      return false;
      }
    }
  return true;
}

bool MediaStorage::SetFromFile(File const &file)
{
  /*
   * DICOMDIR usually have group 0002 present, but no 0008,0016 (doh!)
   * So we first check in header, if found, assumed it is ok (we should
   * check that consistent with 0008,0016 ...)
   * A lot of DICOM image file are still missing the group header
   * this is why we check 0008,0016, and to preserve compat with ACR-NEMA
   * we also check Modality element to guess a fake Media Storage UID
   * file such as:
   * gdcmData/SIEMENS-MR-RGB-16Bits.dcm
   * are a pain to handle ...
   */
  const FileMetaInformation &header = file.GetHeader();
  const char* header_ms_ptr = GetFromHeader(header);
  std::string copy1;
  const char *header_ms_str = 0;
  if( header_ms_ptr )
    {
    copy1 = header_ms_ptr;
    header_ms_str = copy1.c_str();
    }
  const DataSet &ds = file.GetDataSet();
  const char* ds_ms_ptr = GetFromDataSet(ds);
  std::string copy2;
  const char *ds_ms_str = 0;
  if( ds_ms_ptr )
    {
    copy2 = ds_ms_ptr;
    ds_ms_str = copy2.c_str();
    }

  // Easy case:
  if( header_ms_str && ds_ms_str && strcmp(header_ms_str, ds_ms_str) == 0 )
    {
    return SetFromHeader( header );
    }

  if( ds_ms_str )
    {
    // means either no header ms or different, take from dataset just in case
    return SetFromDataSet( ds );
    }

  // Looks suspicious or DICOMDIR...
  if( header_ms_str )
    {
    return SetFromHeader( header );
    }

  // old fall back
  if( !SetFromHeader( header ) )
    {
    // try again but from dataset this time:
    gdcmDebugMacro( "No MediaStorage found in Header, looking up in DataSet" );
    if( !SetFromDataSet( ds ) )
      {
      // ACR-NEMA compat:
      gdcmDebugMacro( "No MediaStorage found neither in DataSet nor in FileMetaInformation, trying from Modality" );
      // Attempt to read what's in Modality:
      if( !SetFromModality( ds ) )
        {
        return false;
        }
      }
    }
// BEGIN SPECIAL HANDLING FOR GDCM 1.2.x 'ReWrite'n files
#if 0
  else if( MSField == MediaStorage::SecondaryCaptureImageStorage )
    {
    /*
     * BEGIN HACK:
     * Technically it should be enough to know that the image is a SecondaryCaptureImageStorage ... BUT GDCM 1.x
     * used to rewrite everything by default as SecondaryCaptureImageStorage so when you would look carefully
     * this DataSet would in fact contains *everything* from the MR Image Storage, therefore, we prefer to use
     * the Source Image Sequence to detect the *real* IOD...I am pretty sure this will bite us one day...
     */
    MediaStorage ms2;
    ms2.SetFromSourceImageSequence(ds);
    if( MSField != ms2 && ms2 != MediaStorage::MS_END )
      {
      assert( MediaStorage::IsImage( ms2 ) );
      gdcmWarningMacro( "Object is declared as SecondaryCaptureImageStorage but according"
        " to Source Image Sequence it was derived from " << ms2 << ". Using it instead" );
      MSField = ms2;
      }
    }
#endif
  return true;
}

} // end namespace gdcm
