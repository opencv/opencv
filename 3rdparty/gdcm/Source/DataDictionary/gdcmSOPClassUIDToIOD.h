/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMSOPCLASSUIDTOIOD_H
#define GDCMSOPCLASSUIDTOIOD_H

#include "gdcmUIDs.h"

namespace gdcm
{

/**
 * \brief Class convert a class SOP Class UID into IOD
 *
 * Reference PS 3.4 Table B.5-1 STANDARD SOP CLASSES
 */
class GDCM_EXPORT SOPClassUIDToIOD
{
public:
  /// Return the associated IOD based on a SOP Class UID uid
  /// (there is a one-to-one mapping from SOP Class UID to matching IOD)
  static const char *GetIOD(UIDs const & uid);

  /// Return the number of SOP Class UID listed internally
  static unsigned int GetNumberOfSOPClassToIOD();

  typedef const char* const (SOPClassUIDToIODType)[2];
  static SOPClassUIDToIODType* GetSOPClassUIDToIODs();

  static SOPClassUIDToIODType& GetSOPClassUIDToIOD(unsigned int i);

  static const char *GetSOPClassUIDFromIOD(const char *iod);
  static const char *GetIODFromSOPClassUID(const char *sopclassuid);
};

} // end namespace gdcm

#endif //GDCMSOPCLASSUIDTOIOD_H
