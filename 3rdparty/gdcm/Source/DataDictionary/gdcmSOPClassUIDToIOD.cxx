/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSOPClassUIDToIOD.h"

#include <cstring>

namespace gdcm
{
  static const char * const SOPClassUIDToIODStrings[][2] = {
{"1.2.840.10008.1.3.10" , "Basic Directory IOD Modules"}, // IOD defined in PS 3.3
{"1.2.840.10008.5.1.4.1.1.1" , "CR Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.1.1" , "Digital X Ray Image IOD Modules"}, //  DX IOD (see B.5.1.1)
{"1.2.840.10008.5.1.4.1.1.1.1.1" , "Digital X Ray Image IOD Modules"}, // DX IOD (see B.5.1.1)
{"1.2.840.10008.5.1.4.1.1.1.2" , "Digital Mammography X Ray Image IOD Modules"}, // (see B.5.1.2)
{"1.2.840.10008.5.1.4.1.1.1.2.1" , "Digital Mammography X Ray Image IOD Modules"}, // (see B.5.1.2)
{"1.2.840.10008.5.1.4.1.1.1.3" , "Digital Intra Oral X Ray Image IOD Modules"}, // (see B.5.1.3)
{"1.2.840.10008.5.1.4.1.1.1.3.1" , "Digital Intra Oral X Ray Image IOD Modules"}, // (see B.5.1.3)
{"1.2.840.10008.5.1.4.1.1.2" , "CT Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.2.1" , "Enhanced CT Image IOD Modules"}, // (see B.5.1.7)
{"1.2.840.10008.5.1.4.1.1.3.1" , "US Multi Frame Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.4" , "MR Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.4.1" , "Enhanced MR Image IOD Modules"}, // (see B.5.1.6)
{"1.2.840.10008.5.1.4.1.1.4.2" , "MR Spectroscopy IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.4.3" , "Enhanced MR Color Image"},
{"1.2.840.10008.5.1.4.1.1.6.1" , "US Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.6.2" , "Enhanced US Volume"},
{"1.2.840.10008.5.1.4.1.1.7" , "SC Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.7.1" , "Multi Frame Single Bit SC Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.7.2" , "Multi Frame Grayscale Byte SC Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.7.3" , "Multi Frame Grayscale Word SC Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.7.4" , "Multi Frame True Color SC Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.9.1.1" , "12 Lead ECG IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.9.1.2" , "General ECG IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.9.1.3" , "Ambulatory ECG IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.9.2.1" , "Hemodynamic IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.9.3.1" , "Basic Cardiac EP IOD Modules"}, // Cardiac Electrophysiology Waveform
{"1.2.840.10008.5.1.4.1.1.9.4.1" , "Basic Voice Audio IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.9.4.2" , "General Audio Waveform"},
{"1.2.840.10008.5.1.4.1.1.9.5.1" , "Arterial Pulse Waveform"},
{"1.2.840.10008.5.1.4.1.1.9.6.1" , "Respiratory Waveform"},
{"1.2.840.10008.5.1.4.1.1.11.1" , "Grayscale Softcopy Presentation State IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.11.2" , "Color Softcopy Presentation State"},
{"1.2.840.10008.5.1.4.1.1.11.3" , "Pseudo-Color Softcopy Presentation State"},
{"1.2.840.10008.5.1.4.1.1.11.4" , "Blending Softcopy Presentation State"},
{"1.2.840.10008.5.1.4.1.1.11.5" , "IOD defined in PS 3.3"},
{"1.2.840.10008.5.1.4.1.1.12.1" , "X Ray Angiographic Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.12.1.1" , "Enhanced X Ray Angiographic Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.12.2" , "XRF Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.12.2.1" , "Enhanced X Ray RF Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.13.1.1" , "X Ray 3D Angiographic Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.13.1.2" , "X-Ray 3D Craniofacial Image"},
{"1.2.840.10008.5.1.4.1.1.13.1.3" , "IOD defined in PS 3.3"},
{"1.2.840.10008.5.1.4.1.1.20" , "NM Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.66" , "Raw Data IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.66.1" , "Spatial Registration IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.66.2" , "Spatial Fiducials IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.66.3" , "Deformable Spatial Registration"},
{"1.2.840.10008.5.1.4.1.1.66.4" , "Segmentation IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.66.5" , "Surface Segmentation"},
{"1.2.840.10008.5.1.4.1.1.67" , "Real World Value Mapping"},
{"1.2.840.10008.5.1.4.1.1.77.1.1" , "VL Endoscopic Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.77.1.1.1" , "Video Endoscopic Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.77.1.2" , "VL Microscopic Image"},
{"1.2.840.10008.5.1.4.1.1.77.1.2.1" , "Video Microscopic Image"},
{"1.2.840.10008.5.1.4.1.1.77.1.3" , "VL Slide-Coordinates Microscopic Image"},
{"1.2.840.10008.5.1.4.1.1.77.1.4" , "VL Photographic Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.77.1.4.1" , "Video Photographic Image"},
{"1.2.840.10008.5.1.4.1.1.77.1.5.1" , "Ophthalmic Photography 8 Bit Image"},
{"1.2.840.10008.5.1.4.1.1.77.1.5.2" , "Ophthalmic Photography 16 Bit Image"},
{"1.2.840.10008.5.1.4.1.1.77.1.5.3" , "Stereometric Relationship"},
{"1.2.840.10008.5.1.4.1.1.77.1.5.4" , "Ophthalmic Tomography Image"},
{"1.2.840.10008.5.1.4.1.1.78.1" , "Lensometry Measurements"},
{"1.2.840.10008.5.1.4.1.1.78.2" , "Autorefraction Measurements"},
{"1.2.840.10008.5.1.4.1.1.78.3" , "Keratometry Measurements"},
{"1.2.840.10008.5.1.4.1.1.78.4" , "Subjective Refraction Measurements"},
{"1.2.840.10008.5.1.4.1.1.78.5" , "Visual Acuity Measurements"},
{"1.2.840.10008.5.1.4.1.1.78.6" , "Spectacle Prescription Report"},
{"1.2.840.10008.5.1.4.1.1.79.1" , "Macular Grid Thickness and Volume Report"},
{"1.2.840.10008.5.1.4.1.1.88.11" , "Basic Text SR IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.88.22" , "Enhanced SR IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.88.33" , "Comprehensive SR IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.88.40" , "Procedure Log"},
{"1.2.840.10008.5.1.4.1.1.88.50" , "Mammography CAD SR IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.88.59" , "Key Object Selection Document IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.88.65" , "Chest CAD SR IOD"},
{"1.2.840.10008.5.1.4.1.1.88.67" , "X Ray Radiation Dose SR IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.88.69" , "Colon CAD SR IOD"},
{"1.2.840.10008.5.1.4.1.1.104.1" , "Encapsulated PDF IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.104.2" , "Encapsulated CDA IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.128" , "PET Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.130" , "IOD defined in PS 3.3"},
{"1.2.840.10008.5.1.4.1.1.131" , "Basic Structured Display IOD"},
{"1.2.840.10008.5.1.4.1.1.481.1" , "RT Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.2" , "RT Dose IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.3" , "RT Structure Set IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.4" , "RT Beams Treatment Record IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.5" , "RT Plan IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.6" , "RT Brachy Treatment Record IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.7" , "RT Treatment Summary Record IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.8" , "RT Ion Plan IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.481.9" , "RT Ion Beams Treatment Record IOD Modules"},
{"1.2.840.10008.5.1.4.38.1" , "Hanging Protocol IOD Modules"},
{"1.2.840.10008.5.1.4.39.1" , "Color Palette IOD"},
// Deprecated:
{"1.2.840.10008.3.1.2.3.3" , "Modality Performed Procedure Step IOD Modules" },
{"1.2.840.10008.5.1.4.1.1.5" , "NM Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.6" , "US Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.3" , "US Multi Frame Image IOD Modules"},
{"1.2.840.10008.5.1.4.1.1.12.3" , ""}, // XRayAngiographicBiplaneImageStorage
// private:
{ "1.3.12.2.1107.5.9.1" , "Siemens Non-image IOD Modules"}, // CSA Non-Image Storage

{ 0, 0 }
};

unsigned int SOPClassUIDToIOD::GetNumberOfSOPClassToIOD()
{
  static const unsigned int n = sizeof( SOPClassUIDToIODStrings ) / sizeof( *SOPClassUIDToIODStrings );
  assert( n > 0 );
  return n - 1;
}

const char *SOPClassUIDToIOD::GetIOD(UIDs const & uid)
{
//  std::ifstream is(  );
//
//  char buf[BUFSIZ];
//  XML_Parser parser = XML_ParserCreate(NULL);
//  int done;
//  //int depth = 0;
//  XML_SetUserData(parser, this);
//  XML_SetElementHandler(parser, startElement, endElement);
//  XML_SetCharacterDataHandler(parser, characterDataHandler);
//  int ret = 0;
//  do {
//    is.read(buf, sizeof(buf));
//    size_t len = is.gcount();
//    done = len < sizeof(buf);
//    if (XML_Parse(parser, buf, len, done) == XML_STATUS_ERROR) {
//      fprintf(stderr,
//        "%s at line %" XML_FMT_INT_MOD "u\n",
//        XML_ErrorString(XML_GetErrorCode(parser)),
//        XML_GetCurrentLineNumber(parser));
//      ret = 1; // Mark as error
//      done = 1; // exit while
//    }
//  } while (!done);
//  XML_ParserFree(parser);
//  is.close();
  //typedef const char* const (*SOPClassUIDToIODType)[2];
  SOPClassUIDToIOD::SOPClassUIDToIODType *p = SOPClassUIDToIODStrings;
  const char *sopclassuid = uid.GetString();

  // FIXME I think we can do binary search
  while( (*p)[0] && strcmp( (*p)[0] , sopclassuid ) != 0 )
    {
    ++p;
    }
  return (*p)[1];
}

SOPClassUIDToIOD::SOPClassUIDToIODType *SOPClassUIDToIOD::GetSOPClassUIDToIODs()
{
  return SOPClassUIDToIODStrings;
}

SOPClassUIDToIOD::SOPClassUIDToIODType& SOPClassUIDToIOD::GetSOPClassUIDToIOD(unsigned int i)
{
  if( i < SOPClassUIDToIOD::GetNumberOfSOPClassToIOD() )
    return SOPClassUIDToIODStrings[i];
  // else return the {0x0, 0x0} sentinel:
  assert( *SOPClassUIDToIODStrings[ SOPClassUIDToIOD::GetNumberOfSOPClassToIOD() ] == 0 );
  return SOPClassUIDToIODStrings[ SOPClassUIDToIOD::GetNumberOfSOPClassToIOD() ];

}

const char *SOPClassUIDToIOD::GetSOPClassUIDFromIOD(const char *iod)
{
  if(!iod) return NULL;
  unsigned int i = 0;
  SOPClassUIDToIODType *sopclassuidtoiods = GetSOPClassUIDToIODs();
  const char *p = sopclassuidtoiods[i][1];
  while( p != 0 )
    {
    if( strcmp( iod, p ) == 0 )
      {
      break;
      }
    ++i;
    p = sopclassuidtoiods[i][1];
    }
  // \postcondition always valid (before sentinel)
  assert( i <= GetNumberOfSOPClassToIOD() );
  return sopclassuidtoiods[i][0];
}

const char *SOPClassUIDToIOD::GetIODFromSOPClassUID(const char *sopclassuid)
{
  if(!sopclassuid) return NULL;
  unsigned int i = 0;
  SOPClassUIDToIODType *sopclassuidtoiods = GetSOPClassUIDToIODs();
  const char *p = sopclassuidtoiods[i][0];
  while( p != 0 )
    {
    if( strcmp( sopclassuid, p ) == 0 )
      {
      break;
      }
    ++i;
    p = sopclassuidtoiods[i][0];
    }
  // \postcondition always valid (before sentinel)
  assert( i <= GetNumberOfSOPClassToIOD() );
  return sopclassuidtoiods[i][1];
}

}
