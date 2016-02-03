/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMVALIDATE_H
#define GDCMVALIDATE_H

#include "gdcmFile.h"

namespace gdcm
{

/**
 * \brief Validate class
 */
class GDCM_EXPORT Validate
{
public:
  Validate();
  ~Validate();

  void SetFile(File const &f) { F = &f; }
  const File& GetValidatedFile() { return V; }

  void Validation();

protected:
  const File *F;
  File V; // Validated file
};

} // end namespace gdcm

#endif //GDCMVALIDATE_H
