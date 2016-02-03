/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMJPEG16CODEC_H
#define GDCMJPEG16CODEC_H

#include "gdcmJPEGCodec.h"

namespace gdcm
{

class JPEGInternals;
class ByteValue;
/**
 * \brief Class to do JPEG 16bits (lossless)
 * \note internal class
 */
class JPEG16Codec : public JPEGCodec
{
public:
  JPEG16Codec();
  ~JPEG16Codec();

  bool DecodeByStreams(std::istream &is, std::ostream &os);
  bool InternalCode(const char *input, unsigned long len, std::ostream &os);

  bool GetHeaderInfo(std::istream &is, TransferSyntax &ts);

protected:
  bool IsStateSuspension() const;
  virtual bool EncodeBuffer(std::ostream &os, const char *data, size_t datalen);

private:
  JPEGInternals *Internals;
};

} // end namespace gdcm

#endif //GDCMJPEG16CODEC_H
