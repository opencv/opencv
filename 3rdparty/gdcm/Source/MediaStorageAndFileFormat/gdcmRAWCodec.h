/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMRAWCODEC_H
#define GDCMRAWCODEC_H

#include "gdcmImageCodec.h"

namespace gdcm
{

class RAWInternals;
/**
 * \brief RAWCodec class
 */
class GDCM_EXPORT RAWCodec : public ImageCodec
{
public:
  RAWCodec();
  ~RAWCodec();
  bool CanCode(TransferSyntax const &ts) const;
  bool CanDecode(TransferSyntax const &ts) const;
  bool Decode(DataElement const &is, DataElement &os);
  bool Code(DataElement const &in, DataElement &out);

  bool GetHeaderInfo(std::istream &is, TransferSyntax &ts);
  virtual ImageCodec * Clone() const;

  /// Used by the ImageStreamReader-- converts a read in 
  /// buffer into one with the proper encodings.
  bool DecodeBytes(const char* inBytes, size_t inBufferLength,
    char* outBytes, size_t inOutBufferLength);

protected:
  bool DecodeByStreams(std::istream &is, std::ostream &os);

private:
  RAWInternals *Internals;
};

} // end namespace gdcm

#endif // GDCMRAWCODEC_H
