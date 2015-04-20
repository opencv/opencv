/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMMEDIASTORAGE_H
#define GDCMMEDIASTORAGE_H

#include "gdcmTransferSyntax.h"

namespace gdcm
{

class DataSet;
class Tag;
class FileMetaInformation;
class File;

// WARNING: This class will be deprecated in the future. There is no reason to extend this class.
// Please check the gdcm::UIDs class if adding new well known UID.

/**
 * \brief MediaStorage
 *
 * \note
 * FIXME There should not be any notion of Image and/or PDF at that point
 * Only the codec can answer yes I support this Media Storage or not...
 * For instance an ImageCodec will answer yes to most of them
 * while a PDFCodec will answer only for the Encapsulated PDF
 *
 * \see UIDs
 */
class GDCM_EXPORT MediaStorage
{
public:
  typedef enum {
    MediaStorageDirectoryStorage = 0,
    ComputedRadiographyImageStorage,
    DigitalXRayImageStorageForPresentation,
    DigitalXRayImageStorageForProcessing,
    DigitalMammographyImageStorageForPresentation,
    DigitalMammographyImageStorageForProcessing,
    DigitalIntraoralXrayImageStorageForPresentation,
    DigitalIntraoralXRayImageStorageForProcessing,
    CTImageStorage,
    EnhancedCTImageStorage,
    UltrasoundImageStorageRetired,
    UltrasoundImageStorage,
    UltrasoundMultiFrameImageStorageRetired,
    UltrasoundMultiFrameImageStorage,
    MRImageStorage,
    EnhancedMRImageStorage,
    MRSpectroscopyStorage,
    NuclearMedicineImageStorageRetired,
    SecondaryCaptureImageStorage,
    MultiframeSingleBitSecondaryCaptureImageStorage,
    MultiframeGrayscaleByteSecondaryCaptureImageStorage,
    MultiframeGrayscaleWordSecondaryCaptureImageStorage,
    MultiframeTrueColorSecondaryCaptureImageStorage,
    StandaloneOverlayStorage,
    StandaloneCurveStorage,
    LeadECGWaveformStorage, // 12-
    GeneralECGWaveformStorage,
    AmbulatoryECGWaveformStorage,
    HemodynamicWaveformStorage,
    CardiacElectrophysiologyWaveformStorage,
    BasicVoiceAudioWaveformStorage,
    StandaloneModalityLUTStorage,
    StandaloneVOILUTStorage,
    GrayscaleSoftcopyPresentationStateStorageSOPClass,
    XRayAngiographicImageStorage,
    XRayRadiofluoroscopingImageStorage,
    XRayAngiographicBiPlaneImageStorageRetired,
    NuclearMedicineImageStorage,
    RawDataStorage,
    SpacialRegistrationStorage, // Spatial
    SpacialFiducialsStorage, // Spatial..
    PETImageStorage,
    RTImageStorage,
    RTDoseStorage,
    RTStructureSetStorage,
    RTPlanStorage,
    CSANonImageStorage,
    Philips3D,
    EnhancedSR,
    BasicTextSR,
    HardcopyGrayscaleImageStorage,
    ComprehensiveSR,
    DetachedStudyManagementSOPClass,
    EncapsulatedPDFStorage,
    EncapsulatedCDAStorage,
    StudyComponentManagementSOPClass,
    DetachedVisitManagementSOPClass,
    DetachedPatientManagementSOPClass,
    VideoEndoscopicImageStorage,
    GeneralElectricMagneticResonanceImageStorage,
    GEPrivate3DModelStorage,
    ToshibaPrivateDataStorage,
    MammographyCADSR,
    KeyObjectSelectionDocument,
    HangingProtocolStorage,
    ModalityPerformedProcedureStepSOPClass,
    PhilipsPrivateMRSyntheticImageStorage,
    VLPhotographicImageStorage,
    SegmentationStorage, // "1.2.840.10008.5.1.4.1.1.66.4"
    RTIonPlanStorage, // 1.2.840.10008.5.1.4.1.1.481.8
    XRay3DAngiographicImageStorage, // 1.2.840.10008.5.1.4.1.1.13.1.1
    EnhancedXAImageStorage,
    RTIonBeamsTreatmentRecordStorage, // 1.2.840.10008.5.1.4.1.1.481.9
    SurfaceSegmentationStorage, // "1.2.840.10008.5.1.4.1.1.66.5"
    VLWholeSlideMicroscopyImageStorage, // 1.2.840.10008.5.1.4.1.1.77.1.6
    RTTreatmentSummaryRecordStorage, // 1.2.840.10008.5.1.4.1.1.481.7
    EnhancedUSVolumeStorage, // 1.2.840.10008.5.1.4.1.1.6.2
    XRayRadiationDoseSR, // 1.2.840.10008.5.1.4.1.1.88.67
    VLEndoscopicImageStorage, // 1.2.840.10008.5.1.4.1.1.77.1.1
    BreastTomosynthesisImageStorage, // 1.2.840.10008.5.1.4.1.1.13.1.3
    FujiPrivateCRImageStorage, // 1.2.392.200036.9125.1.1.2
    OphthalmicPhotography8BitImageStorage, // 1.2.840.10008.5.1.4.1.1.77.1.5.1
    OphthalmicTomographyImageStorage, // 1.2.840.10008.5.1.4.1.1.77.1.5.4
    VLMicroscopicImageStorage,
    MS_END
  } MSType; // Media Storage Type

typedef enum {
    NoObject = 0, // DICOMDIR
    Video, // Most common, include image, video and volume
    Waveform, // Isn't it simply a 1D video ?
    Audio, // ???
    PDF,
    URI, // URL...
    Segmentation, // TODO
    ObjectEnd
  } ObjectType;

  /// Return the Media String associated. Will return NULL for MS_END
  static const char* GetMSString(MSType ts);

  /// Return the Media String of the object.
  const char* GetString() const;
  static MSType GetMSType(const char *str);

  MediaStorage(MSType type = MS_END):MSField(type) {}

  /// Returns whether DICOM has a Pixel Data element (7fe0,0010)
  /// \warning MRSpectroscopyStorage could be image but are not
  static bool IsImage(MSType ts);

  operator MSType () const { return MSField; }

  const char *GetModality() const;
  unsigned int GetModalityDimension() const;

  static unsigned int GetNumberOfMSType();
  static unsigned int GetNumberOfMSString();
  static unsigned int GetNumberOfModality();


  /// Attempt to set the MediaStorage from a file:
  /// WARNING: When no MediaStorage & Modality are found BUT a PixelData element is found
  /// then MediaStorage is set to the default SecondaryCaptureImageStorage (return value is
  /// false in this case)
  bool SetFromFile(File const &file);

  /// Advanced user only (functions should be protected level...)
  /// Those function are lower level than SetFromFile
  bool SetFromDataSet(DataSet const &ds); // Will get the SOP Class UID
  bool SetFromHeader(FileMetaInformation const &fmi); // Will get the Media Storage SOP Class UID
  bool SetFromModality(DataSet const &ds);
  void GuessFromModality(const char *modality, unsigned int dimension = 2);

  friend std::ostream &operator<<(std::ostream &os, const MediaStorage &ms);

  bool IsUndefined() const { return MSField == MS_END; }

protected:
  void SetFromSourceImageSequence(DataSet const &ds);

private:
  bool SetFromDataSetOrHeader(DataSet const &ds, const Tag & tag);
  /// NOT THREAD SAFE
  const char* GetFromDataSetOrHeader(DataSet const &ds, const Tag & tag);
  /// NOT THREAD SAFE
  const char* GetFromHeader(FileMetaInformation const &fmi);
  /// NOT THREAD SAFE
  const char* GetFromDataSet(DataSet const &ds);

private:
  MSType MSField;
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &_os, const MediaStorage &ms)
{
  const char *msstring = MediaStorage::GetMSString(ms);
  _os << (msstring ? msstring : "INVALID MEDIA STORAGE");
  return _os;

}

} // end namespace gdcm

#endif // GDCMMEDIASTORAGE_H
