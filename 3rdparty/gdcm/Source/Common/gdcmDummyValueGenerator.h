/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMDUMMYVALUEGENERATOR_H
#define GDCMDUMMYVALUEGENERATOR_H

#include "gdcmTypes.h"

namespace gdcm
{

/**
 * \brief Class for generating dummy value
 * \see Anonymizer
 */
class GDCM_EXPORT DummyValueGenerator
{
public:

  /** Generate a dummy value from an input value. This is guarantee to always
   * return the same output value when input is identical.  Return an array of
   * bytes that can be used for anonymization purpose, return NULL on error
   * NOT THREAD SAFE
   */
  static const char* Generate(const char *input);

private:
};


} // end namespace gdcm

#endif //GDCMDUMMYVALUEGENERATOR_H
