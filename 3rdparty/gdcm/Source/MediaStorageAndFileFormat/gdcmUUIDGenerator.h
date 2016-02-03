/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMUUIDGENERATOR_H
#define GDCMUUIDGENERATOR_H

#include "gdcmTypes.h"

namespace gdcm
{

/**
 * \brief Class for generating unique UUID
 * generate DCE 1.1 uid
 */
class GDCM_EXPORT UUIDGenerator
{
public:
  /// Return the generated uuid
  /// NOT THREAD SAFE
  const char* Generate();

  /// Find out if the string is a valid UUID or not
  static bool IsValid(const char *uid);

private:
  std::string Unique; // Buffer
};

} // end namespace gdcm

#endif //GDCMUUIDGENERATOR_H
