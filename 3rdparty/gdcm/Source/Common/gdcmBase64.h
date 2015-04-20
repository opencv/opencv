/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMBASE64_H
#define GDCMBASE64_H

#include "gdcmTypes.h"

namespace gdcm
{
/**
 * \brief Class for Base64
 *
 */
class GDCM_EXPORT Base64
{
public:

  /**
   * Call this function to obtain the required buffer size
   */
  static size_t GetEncodeLength(const char *src, size_t srclen );

  /**
   * \brief          Encode a buffer into base64 format
   *
   * \param dst      destination buffer
   * \param dlen     size of the buffer
   * \param src      source buffer
   * \param slen     amount of data to be encoded
   *
   * \return         0 if not successful, size of encoded otherwise
   *
   */
  static size_t Encode( char *dst, size_t dlen, const char *src, size_t slen );

  /**
   * Call this function to obtain the required buffer size
   */
  static size_t GetDecodeLength( const char *src, size_t len );

  /**
   * \brief          Decode a base64-formatted buffer
   *
   * \param dst      destination buffer
   * \param dlen     size of the buffer
   * \param src      source buffer
   * \param slen     amount of data to be decoded
   *
   * \return         0 if not successful, size of decoded otherwise
   */
  static size_t Decode( char *dst, size_t dlen, const char *src, size_t slen );

private:
  Base64(const Base64&);  // Not implemented.
  void operator=(const Base64&);  // Not implemented.
};

} // end namespace gdcm

#endif // GDCMBASE64_H
