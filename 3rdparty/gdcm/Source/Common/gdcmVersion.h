/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMVERSION_H
#define GDCMVERSION_H

#include "gdcmTypes.h"
#include <iostream>

namespace gdcm
{
/**
 * \class Version
 * \brief major/minor and build version
 */
//-----------------------------------------------------------------------------
class GDCM_EXPORT Version
{
  friend std::ostream& operator<<(std::ostream &_os, const Version &v);
public :
  static const char *GetVersion();
  static int GetMajorVersion();
  static int GetMinorVersion();
  static int GetBuildVersion();

  void Print(std::ostream &os = std::cout) const;

//protected:
  Version() {};
  ~Version() {};
};
//-----------------------------------------------------------------------------
inline std::ostream& operator<<(std::ostream &os, const Version &v)
{
  v.Print( os );
  return os;
}

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMVERSION_H
