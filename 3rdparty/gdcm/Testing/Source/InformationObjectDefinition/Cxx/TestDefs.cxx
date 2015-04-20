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
#include "gdcmUIDs.h"
#include "gdcmGlobal.h"
#include "gdcmMediaStorage.h"
#include "gdcmSOPClassUIDToIOD.h"

int TestDefs(int, char *[])
{
  using gdcm::MediaStorage;
  gdcm::Global& g = gdcm::Global::GetInstance();
  if( !g.LoadResourcesFiles() )
    {
    std::cerr << "Could not LoadResourcesFiles" << std::endl;
    return 1;
    }

  const gdcm::Defs &defs = g.GetDefs();
  //std::cout << defs.GetMacros() << std::endl;

  int ret = 0;
  gdcm::MediaStorage::MSType mst;
  for ( mst = gdcm::MediaStorage::MediaStorageDirectoryStorage;
    mst < gdcm::MediaStorage::MS_END; mst = (gdcm::MediaStorage::MSType)(mst + 1) )
    {
    const char *iod = defs.GetIODNameFromMediaStorage(mst);
    gdcm::UIDs uid;
    uid.SetFromUID( gdcm::MediaStorage::GetMSString(mst) /*mst.GetString()*/ );
    if( !iod )
      {
      // We do not support Private IODs (for now??)
      if( mst != MediaStorage::PhilipsPrivateMRSyntheticImageStorage
        && mst != MediaStorage::ToshibaPrivateDataStorage
        && mst != MediaStorage::GEPrivate3DModelStorage
        && mst != MediaStorage::Philips3D
        && mst != MediaStorage::CSANonImageStorage
        && mst != MediaStorage::GeneralElectricMagneticResonanceImageStorage )
        {
        std::cerr << "Missing iod for MS: " << (int)mst << " " <<
          gdcm::MediaStorage::GetMSString(mst) << "  "; //std::endl;
        std::cerr << "MediaStorage is " << (int)mst << " [" << uid.GetName() << "]" << std::endl;
        ++ret;
        }
      }
    else
      {
      const char *iod_ref = gdcm::SOPClassUIDToIOD::GetIOD(uid);
      if( !iod_ref )
        {
        std::cerr << "Could not find IOD for SOPClass: " << uid << std::endl;
        ++ret;
        }
      else
        {
        std::string iod_ref_str = iod_ref;
        //iod_ref_str += " IOD Modules";
        if( iod_ref_str != iod )
          {
          std::cerr << "UID: " << uid << "   ";
          std::cerr << "Incompatible IODs: [" << iod << "] versus ref= [" <<
            iod_ref_str << "]" << std::endl;
          ++ret;
          }
        }
      }
    }

  unsigned int nm = MediaStorage::GetNumberOfModality();
  unsigned int nsop = gdcm::SOPClassUIDToIOD::GetNumberOfSOPClassToIOD();
  if( nm != nsop )
    {
    std::cerr << "Incompatible MediaStorage knows: " << nm <<
      " SOP Classes while SOPClassUIDToIOD knows: " << nsop << " classes" << std::endl;
    ++ret;
    }

  //return ret;
  return 0;
}
