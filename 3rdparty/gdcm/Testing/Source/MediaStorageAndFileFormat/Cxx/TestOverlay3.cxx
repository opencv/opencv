/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmOverlay.h"
#include "gdcmImageReader.h"
#include "gdcmTesting.h"
#include "gdcmSystem.h"
#include "gdcmFilename.h"

using namespace gdcm;

struct ovel
{
  const char *md5;
  const char *fn;
  unsigned int idx;
  Overlay::OverlayType type;
};

static const ovel overlay3array[] = {
// gdcmData
    {"d42bff3545ed8c5fccb39d9a61828992", "MR-SIEMENS-DICOM-WithOverlays-extracted-overlays.dcm", 0, Overlay::Graphics },
    {"2cf60257b75a034fbdc98e560881184e", "PHILIPS_Brilliance_ExtraBytesInOverlay.dcm", 0, Overlay::Graphics  },
    {"b2dd1007e018b3b9691761cf93f77552", "05115014-mr-siemens-avanto-syngo-with-palette-icone.dcm", 0, Overlay::Graphics  },
    {"d42bff3545ed8c5fccb39d9a61828992", "MR-SIEMENS-DICOM-WithOverlays.dcm", 0, Overlay::Graphics  },
// gdcmDataExtra
    {"4b0240033afba211eeac42a44417d4c9", "05119848_IS_Black_hasOverlayData.dcm", 0, Overlay::Graphics  },
    {"349d1f9510f64467ecf73eeea46c9c6e", "45909476", 0, Overlay::Graphics  },
    {"6a5f8038cc8cf753bf74422164adc24c", "45909517", 0, Overlay::Graphics  },
    {"1a3bf73e42b0f6dc282a9be59c054027", "OverlayDICOMDataSet.dcm", 0, Overlay::Graphics  },
// gdcmConformanceTests
    {"040560796c1a53ffce0d2f7e90c9dc26", "CT_OSIRIX_OddOverlay.dcm", 0, Overlay::Graphics  },
  // random
//    {"f7e43de189a1bc08044c13aefac73fed", "1.dcm", 0 },
//    {"e7859c818f26202fb63a2b205ff16297", "1.dcm", 1 },
//    {"aa4c726bc52e13b750ac8c94c7b06e07", "0.dcm", 0 },
//    {"31d58476326722793379fbcda55a4856", "0.dcm", 1 },

    // sentinel
    { 0, 0, 0, Overlay::Invalid }
};

static int TestReadOverlay(const char* filename, bool verbose = false)
{
  if( verbose )
    std::cerr << "Reading: " << filename << std::endl;
  gdcm::ImageReader reader;

  reader.SetFileName( filename );
  int ret = 0;
  if ( reader.Read() )
    {
    gdcm::Filename fn( filename );
    const char *name = fn.GetName();

    std::vector<char> overlay;
    const gdcm::Image &img = reader.GetImage();
    size_t numoverlays = img.GetNumberOfOverlays();
    for( size_t ovidx = 0; ovidx < numoverlays; ++ovidx )
      {
      const gdcm::Overlay& ov = img.GetOverlay(ovidx);
      size_t len = ov.GetUnpackBufferLength();
      overlay.resize( len );
      if( !ov.GetUnpackBuffer(&overlay[0], len) )
        {
        std::cerr << "GetUnpackBuffer: Problem with Overlay: #" << ovidx << std::endl;
        ++ret;
        }
      char digest1[33];
      if( !gdcm::Testing::ComputeMD5(&overlay[0], len, digest1) )
        {
        std::cerr << "ComputeMD5: Problem with Overlay: #" << ovidx << std::endl;
        ++ret;
        }
      std::stringstream overlay2;
      ov.Decompress(overlay2);
      Overlay::OverlayType type = ov.GetTypeAsEnum();
      const std::string soverlay2 = overlay2.str();
      if( soverlay2.size() != len )
        {
        std::cerr << "Decompress: Problem with Overlay: #" << ovidx << std::endl;
        std::cerr << "Size is: " << soverlay2.size() << " vs " << len << std::endl;
        ++ret;
        }
      char digest2[33];
      if( !gdcm::Testing::ComputeMD5(soverlay2.c_str(), soverlay2.size(), digest2) )
        {
        std::cerr << "ComputeMD5: Problem with Overlay: #" << ovidx << std::endl;
        ++ret;
        }

      Overlay::OverlayType reftype = Overlay::Invalid;
      const char *refmd5 = NULL;
        {
        unsigned int i = 0;
        const char *p = overlay3array[i].fn;
        unsigned int idx = overlay3array[i].idx;
        while( p != 0 )
          {
          if( strcmp( name, p ) == 0 && ovidx == idx )
            {
            break;
            }
          ++i;
          p = overlay3array[i].fn;
          idx = overlay3array[i].idx;
          }
        refmd5 = overlay3array[i].md5;
        reftype = overlay3array[i].type;
        }

      if( !refmd5 )
        {
        std::cerr << "refmd5: Problem with Overlay: #" << ovidx << std::endl;
        std::cerr << name << std::endl;
        ++ret;
        }
      if( refmd5 && strcmp(digest1, refmd5) )
        {
        std::cerr << "strcmp/ref: Problem with Overlay: #" << ovidx << std::endl;
        std::cerr << "ref: " << refmd5 << " vs " << digest1 << std::endl;
        ++ret;
        }
      if( strcmp(digest1, digest2) )
        {
        std::cerr << "strcmp/1/2: Problem with Overlay: #" << ovidx << std::endl;
        std::cerr << "digest1: " << digest1 << " vs " << digest2 << std::endl;
        ++ret;
        }
      if( reftype != type )
        {
        std::cerr << "OverlayType: Problem with Overlay: #" << ovidx << std::endl;
        std::cerr << "reftype: " << (int)reftype << " vs " << (int)type << std::endl;
        std::cerr << name << std::endl;
        ++ret;
        }
      }
    }

  return ret;
}

int TestOverlay3(int argc, char *argv[])
{
  if( argc == 2 )
    {
    const char *filename = argv[1];
    return TestReadOverlay(filename, true);
    }

  // else
  // First of get rid of warning/debug message
  gdcm::Trace::DebugOff();
  gdcm::Trace::WarningOff();
  gdcm::Trace::ErrorOff();
  int r = 0, i = 0;
  const char *filename;
  const char * const *filenames = gdcm::Testing::GetFileNames();
  while( (filename = filenames[i]) )
    {
    r += TestReadOverlay( filename);
    ++i;
    }

  return r;
}
