/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMRLECODEC_H
#define GDCMRLECODEC_H

#include "gdcmImageCodec.h"

namespace gdcm
{

class Fragment;
class RLEInternals;
/**
 * \brief Class to do RLE
 * \note
 * ANSI X3.9
 * A.4.2 RLE Compression
 * Annex G defines a RLE Compression Transfer Syntax. This transfer Syntax is
 * identified by the UID value "1.2.840.10008.1.2.5". If the object allows
 * multi-frame images in the pixel data field, then each frame shall be encoded
 * separately. Each frame shall be encoded in one and only one Fragment (see PS
 * 3.5.8.2).
 *
 */
class GDCM_EXPORT RLECodec : public ImageCodec
{
friend class ImageRegionReader;
public:
  RLECodec();
  ~RLECodec();
  bool CanCode(TransferSyntax const &ts) const;
  bool CanDecode(TransferSyntax const &ts) const;
  bool Decode(DataElement const &is, DataElement &os);
  unsigned long GetBufferLength() const { return BufferLength; }
  void SetBufferLength(unsigned long l) { BufferLength = l; }

  bool Code(DataElement const &in, DataElement &out);
  bool GetHeaderInfo(std::istream &is, TransferSyntax &ts);
  virtual ImageCodec * Clone() const;

protected:
  bool DecodeExtent(
    char *buffer,
    unsigned int XMin, unsigned int XMax,
    unsigned int YMin, unsigned int YMax,
    unsigned int ZMin, unsigned int ZMax,
    std::istream & is
  );

  bool DecodeByStreams(std::istream &is, std::ostream &os);
public:

  void SetLength(unsigned long l)
    {
    Length = l;
    }

protected:
  bool StartEncode( std::ostream & );
  bool IsRowEncoder();
  bool IsFrameEncoder();
  bool AppendRowEncode( std::ostream & out, const char * data, size_t datalen );
  bool AppendFrameEncode( std::ostream & out, const char * data, size_t datalen );
  bool StopEncode( std::ostream & );

private:
  bool DecodeByStreamsCommon(std::istream &is, std::ostream &os);
  RLEInternals *Internals;
  unsigned long Length;
  unsigned long BufferLength;
  size_t DecodeFragment(Fragment const & frag, char *buffer, unsigned long llen);
};

} // end namespace gdcm

#endif //GDCMRLECODEC_H
