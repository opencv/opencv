/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDefs.h"
#include "gdcmTableReader.h"
#include "gdcmMediaStorage.h"
//#include "gdcmGlobal.h"
#include "gdcmTrace.h"
#include "gdcmFile.h"

#include <stdlib.h>

namespace gdcm
{

Defs::Defs()
{
}

Defs::~Defs()
{
}

void Defs::LoadDefaults()
{
//  TableReader tr(*this);
//  const Global &g = Global::GetInstance();
//  const char *filename = g.Locate( "Part3.xml" );
//
//  if( filename )
//    {
//    tr.SetFilename(filename);
//    tr.Read();
//    }
//  else
//    {
//    gdcmErrorMacro( "Could not find Part3.xml file. Please report" );
//    throw Exception( "Impossible" );
//    }
}

void Defs::LoadFromFile(const char *filename)
{
  assert( filename );
  TableReader tr(*this);
  tr.SetFilename(filename);
  tr.Read();
}

const char *Defs::GetIODNameFromMediaStorage(MediaStorage const &ms)
{
  const char *iodname;
  switch(ms)
    {
    case MediaStorage::MediaStorageDirectoryStorage:
      iodname = "Basic Directory IOD Modules";
      break;
    case MediaStorage::MRImageStorage:
      iodname = "MR Image IOD Modules";
      break;
    case MediaStorage::EnhancedMRImageStorage:
      iodname = "Enhanced MR Image IOD Modules";
      break;
    case MediaStorage::CTImageStorage:
      iodname = "CT Image IOD Modules";
      break;
    case MediaStorage::EnhancedCTImageStorage:
      iodname = "Enhanced CT Image IOD Modules";
      break;
    case MediaStorage::ComputedRadiographyImageStorage:
      iodname = "CR Image IOD Modules";
      break;
    case MediaStorage::XRayAngiographicImageStorage:
    //case MediaStorage::XRayAngiographicBiPlaneImageStorageRetired: // FIXME ???
      iodname = "X Ray Angiographic Image IOD Modules";
      break;
    case MediaStorage::UltrasoundImageStorageRetired:
    case MediaStorage::UltrasoundImageStorage:
      iodname = "US Image IOD Modules";
      break;
    case MediaStorage::UltrasoundMultiFrameImageStorageRetired:
    case MediaStorage::UltrasoundMultiFrameImageStorage:
      iodname ="US Multi Frame Image IOD Modules";
      break;
    case MediaStorage::SecondaryCaptureImageStorage:
      iodname = "SC Image IOD Modules";
      break;
    case MediaStorage::XRayRadiofluoroscopingImageStorage:
      iodname = "XRF Image IOD Modules";
      break;
    case MediaStorage::MRSpectroscopyStorage:
      iodname = "MR Spectroscopy IOD Modules";
      break;
    case MediaStorage::NuclearMedicineImageStorageRetired:
    case MediaStorage::NuclearMedicineImageStorage:
      iodname = "NM Image IOD Modules";
      break;
    case MediaStorage::MultiframeSingleBitSecondaryCaptureImageStorage:
      iodname = "Multi Frame Single Bit SC Image IOD Modules";
      break;
    case MediaStorage::MultiframeGrayscaleByteSecondaryCaptureImageStorage:
      iodname = "Multi Frame Grayscale Byte SC Image IOD Modules";
      break;
    case MediaStorage::MultiframeGrayscaleWordSecondaryCaptureImageStorage:
      iodname = "Multi Frame Grayscale Word SC Image IOD Modules";
      break;
    case MediaStorage::MultiframeTrueColorSecondaryCaptureImageStorage:
      iodname = "Multi Frame True Color SC Image IOD Modules";
      break;
    case MediaStorage::EncapsulatedPDFStorage:
      iodname = "Encapsulated PDF IOD Modules";
      break;
    case MediaStorage::EncapsulatedCDAStorage:
      iodname = "Encapsulated CDA IOD Modules";
      break;
    case MediaStorage::VLPhotographicImageStorage:
      iodname = "VL Photographic Image IOD Modules";
      break;
    case MediaStorage::SegmentationStorage:
      iodname = "Segmentation IOD Modules";
      break;
    case MediaStorage::RawDataStorage:
      iodname = "Raw Data IOD Modules";
      break;
    case MediaStorage::MammographyCADSR:
      iodname = "Mammography CAD SR IOD Modules";
      break;
    case MediaStorage::VideoEndoscopicImageStorage:
      iodname = "Video Endoscopic Image IOD Modules";
      break;
    case MediaStorage::RTImageStorage:
      iodname = "RT Image IOD Modules";
      break;
    case MediaStorage::RTDoseStorage:
      iodname = "RT Dose IOD Modules";
      break;
    case MediaStorage::RTStructureSetStorage:
      iodname = "RT Structure Set IOD Modules";
      break;
    case MediaStorage::RTPlanStorage:
      iodname = "RT Plan IOD Modules";
      break;
    case MediaStorage::ModalityPerformedProcedureStepSOPClass:
      iodname = "Modality Performed Procedure Step IOD Modules";
      break;
    case MediaStorage::HangingProtocolStorage:
      iodname = "Hanging Protocol IOD Modules";
      break;
    case MediaStorage::KeyObjectSelectionDocument:
      iodname = "Key Object Selection Document IOD Modules";
      break;
    case MediaStorage::ComprehensiveSR:
      iodname = "Comprehensive SR IOD Modules";
      break;
    case MediaStorage::HemodynamicWaveformStorage:
      iodname = "Hemodynamic IOD Modules";
      break;
    case MediaStorage::DigitalIntraoralXrayImageStorageForPresentation:
    case MediaStorage::DigitalIntraoralXRayImageStorageForProcessing:
      iodname = "Digital Intra Oral X Ray Image IOD Modules";
      break;
    case MediaStorage::DigitalXRayImageStorageForPresentation:
    case MediaStorage::DigitalXRayImageStorageForProcessing:
      iodname = "Digital X Ray Image IOD Modules";
      break;
    case MediaStorage::DigitalMammographyImageStorageForPresentation:
    case MediaStorage::DigitalMammographyImageStorageForProcessing:
      iodname = "Digital Mammography X Ray Image IOD Modules";
      break;
    case MediaStorage::GrayscaleSoftcopyPresentationStateStorageSOPClass:
      iodname = "Grayscale Softcopy Presentation State IOD Modules";
      break;
    case MediaStorage::LeadECGWaveformStorage:
      iodname = "12 Lead ECG IOD Modules";
      break;
    case MediaStorage::GeneralECGWaveformStorage:
      iodname = "General ECG IOD Modules";
      break;
    case MediaStorage::AmbulatoryECGWaveformStorage:
      iodname = "Ambulatory ECG IOD Modules";
      break;
    case MediaStorage::BasicVoiceAudioWaveformStorage:
      iodname = "Basic Voice Audio IOD Modules";
      break;
    case MediaStorage::SpacialFiducialsStorage:
      iodname = "Spatial Fiducials IOD Modules";
      break;
    case MediaStorage::BasicTextSR:
      iodname = "Basic Text SR IOD Modules";
      break;
    case MediaStorage::CardiacElectrophysiologyWaveformStorage:
      iodname = "Basic Cardiac EP IOD Modules";
      break;
    case MediaStorage::PETImageStorage:
      iodname = "PET Image IOD Modules";
      break;
    case MediaStorage::EnhancedSR:
      iodname = "Enhanced SR IOD Modules";
      break;
    case MediaStorage::SpacialRegistrationStorage:
      iodname = "Spatial Registration IOD Modules";
      break;
    case MediaStorage::RTIonPlanStorage:
      iodname = "RT Ion Plan IOD Modules";
      break;
    case MediaStorage::XRay3DAngiographicImageStorage:
      iodname = "X Ray 3D Angiographic Image IOD Modules";
      break;
    case MediaStorage::EnhancedXAImageStorage:
      iodname = "Enhanced X Ray Angiographic Image IOD Modules";
      break;
    case MediaStorage::RTIonBeamsTreatmentRecordStorage:
      iodname = "RT Ion Beams Treatment Record IOD Modules";
      break;
    case MediaStorage::RTTreatmentSummaryRecordStorage:
      iodname = "RT Treatment Summary Record IOD Modules";
      break;
    case MediaStorage::VLEndoscopicImageStorage:
      iodname = "VL Endoscopic Image IOD Modules";
      break;
    case MediaStorage::XRayRadiationDoseSR:
      iodname = "X Ray Radiation Dose SR IOD Modules";
      break;
    case MediaStorage::CSANonImageStorage:
      iodname = "Siemens Non-image IOD Modules";
      break;
    case MediaStorage::VLMicroscopicImageStorage:
      iodname = "VL Microscopic Image IOD Modules";
      break;
    default:
      iodname = 0;
    }
  return iodname;
}

const IOD& Defs::GetIODFromFile(const File& file) const
{
  MediaStorage ms;
  ms.SetFromFile(file); // SetFromDataSet does not handle DICOMDIR

  const IODs &iods = GetIODs();
  const char *iodname = GetIODNameFromMediaStorage( ms );
  if( !iodname )
    {
    gdcmErrorMacro( "Not implemented: " << ms );
    throw "Not Implemented";
    }
  const IOD &iod = iods.GetIOD( iodname );
  return iod;
}

Type Defs::GetTypeFromTag(const File& file, const Tag& tag) const
{
  Type ret;
  MediaStorage ms;
  ms.SetFromFile(file); // SetFromDataSet does not handle DICOMDIR

  const IODs &iods = GetIODs();
  const Modules &modules = GetModules();
  const char *iodname = GetIODNameFromMediaStorage( ms );
  if( !iodname )
    {
    gdcmErrorMacro( "Not implemented: " << ms );
    return ret;
    }
  const IOD &iod = iods.GetIOD( iodname );
  const Macros &macros = GetMacros();

  IOD::SizeType niods = iod.GetNumberOfIODs();
  // Iterate over each iod entry in order:
  for(unsigned int idx = 0; idx < niods; ++idx)
    {
    const IODEntry &iodentry = iod.GetIODEntry(idx);
    const char *ref = iodentry.GetRef();
    //Usage::UsageType ut = iodentry.GetUsageType();

    const Module &module = modules.GetModule( ref );
    if( module.FindModuleEntryInMacros(macros, tag ) )
      {
      const ModuleEntry &module_entry = module.GetModuleEntryInMacros(macros,tag);
      ret = module_entry.GetType();
      }
    }

  return ret;
}

bool Defs::Verify(const File& file) const
{
  MediaStorage ms;
  ms.SetFromFile(file);

  const IODs &iods = GetIODs();
  const Modules &modules = GetModules();
  const char *iodname = GetIODNameFromMediaStorage( ms );
  if( !iodname )
    {
    gdcmErrorMacro( "Not implemented" );
    return false;
    }
  const IOD &iod = iods.GetIOD( iodname );

  //std::cout << iod << std::endl;
  //std::cout << iod.GetIODEntry(14) << std::endl;
  IOD::SizeType niods = iod.GetNumberOfIODs();
  bool v = true;
  // Iterate over each iod entry in order:
  for(unsigned int idx = 0; idx < niods; ++idx)
    {
    const IODEntry &iodentry = iod.GetIODEntry(idx);
    const char *ref = iodentry.GetRef();
    Usage::UsageType ut = iodentry.GetUsageType();

    const Module &module = modules.GetModule( ref );
    //std::cout << module << std::endl;
    v = v && module.Verify( file.GetDataSet(), ut );
    }

  return v;

}

bool Defs::Verify(const DataSet& ds) const
{
  MediaStorage ms;
  ms.SetFromDataSet(ds);

  const IODs &iods = GetIODs();
  const Modules &modules = GetModules();
  const char *iodname = GetIODNameFromMediaStorage( ms );
  if( !iodname )
    {
    gdcmErrorMacro( "Not implemented" );
    return false;
    }
  const IOD &iod = iods.GetIOD( iodname );

  //std::cout << iod << std::endl;
  //std::cout << iod.GetIODEntry(14) << std::endl;
  IOD::SizeType niods = iod.GetNumberOfIODs();
  bool v = true;
  // Iterate over each iod entry in order:
  for(unsigned int idx = 0; idx < niods; ++idx)
    {
    const IODEntry &iodentry = iod.GetIODEntry(idx);
    const char *ref = iodentry.GetRef();
    Usage::UsageType ut = iodentry.GetUsageType();

    const Module &module = modules.GetModule( ref );
    //std::cout << module << std::endl;
    v = v && module.Verify( ds, ut );
    }

  return v;

}


} // end namespace gdcm
