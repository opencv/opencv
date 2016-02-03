/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSHA1_H
#define GDCMSHA1_H

#include "gdcmTypes.h"

namespace gdcm
{
//-----------------------------------------------------------------------------
class SHA1Internals;
/**
 * \brief Class for SHA1
 *
 * \warning this class is able to pick from one implementation:
 *
 * 1. the one from OpenSSL (when GDCM_USE_SYSTEM_OPENSSL is turned ON)
 *
 * In all other cases it will return an error
 */
class GDCM_EXPORT SHA1
{
public :
  SHA1();
  ~SHA1();

  static bool Compute(const char *buffer, unsigned long buf_len, char digest_str[20*2+1]);

  static bool ComputeFile(const char *filename, char digest_str[20*2+1]);

private:
  SHA1Internals *Internals;
private:
  SHA1(const SHA1&);  // Not implemented.
  void operator=(const SHA1&);  // Not implemented.
};
} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMSHA1_H
