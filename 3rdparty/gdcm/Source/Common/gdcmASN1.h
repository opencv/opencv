/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMASN1_H
#define GDCMASN1_H

#include "gdcmTypes.h"


namespace gdcm
{
//-----------------------------------------------------------------------------
class ASN1Internals;
/**
 * \brief Class for ASN1
 *
 */
class GDCM_EXPORT ASN1
{
public :
  ASN1();
  ~ASN1();

  static bool ParseDumpFile(const char *filename);

  static bool ParseDump(const char *array, size_t length);

protected:
  int TestPBKDF2();

private:
  ASN1Internals *Internals;
private:
  ASN1(const ASN1&);  // Not implemented.
  void operator=(const ASN1&);  // Not implemented.
};
} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMASN1_H
