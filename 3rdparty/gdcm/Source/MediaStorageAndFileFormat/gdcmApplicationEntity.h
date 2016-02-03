/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMAPPLICATIONENTITY_H
#define GDCMAPPLICATIONENTITY_H

#include "gdcmTypes.h"
#include <vector>
#include <stdlib.h> // abort

namespace gdcm
{

/**
 * \brief ApplicationEntity
 * - AE Application Entity
 * - A string of characters that identifies an Application Entity with leading
 *   and trailing spaces (20H) being non-significant. A value consisting solely
 *   of spaces shall not be used.
 * - Default Character Repertoire excluding character code 5CH (the BACKSLASH \ in
 *   ISO-IR 6), and control characters LF, FF, CR and ESC.
 * - 16 bytes maximum
 */
class GDCM_EXPORT ApplicationEntity
{
public:
  static const unsigned int MaxNumberOfComponents = 1;
  static const unsigned int MaxLength = 16;
  std::string Internal;
  static const char Separator = ' ';
  static const char Padding   = ' ';
  //static const char Excluded[5] = { '\\' /* 5CH */, '\n' /* LF */, '\f', /* FF */, '\r' /* CR */, 0x1b /* ESC */};

  bool IsValid() const {
    return true;
  }
  void Squeeze() {
    // trim leading and trailing white spaces
  }
  void SetBlob(const std::vector<char>& v) {
    (void)v;
    assert(0); //TODO
  }
  void Print(std::ostream &os) const {
  (void)os;
    assert(0); //TODO
  }
};

} // end namespace gdcm

#endif //GDCMAPPLICATIONENTITY_H
