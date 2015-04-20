/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTRANSFERSYNTAX_H
#define GDCMTRANSFERSYNTAX_H

#include "gdcmSwapCode.h"

namespace gdcm
{

/**
 * \brief Class to manipulate Transfer Syntax
 * \note
 * TRANSFER SYNTAX (Standard and Private): A set of encoding rules that allow
 * Application Entities to unambiguously negotiate the encoding techniques
 * (e.g., Data Element structure, byte ordering, compression) they are able to
 * support, thereby allowing these Application Entities to communicate.
 *
 * \todo: The implementation is completely retarded -> see gdcm::UIDs for a replacement
 * We need: IsSupported
 * We need preprocess of raw/xml file
 * We need GetFullName()
 *
 * Need a notion of Private Syntax. As defined in PS 3.5. Section 9.2
 *
 * \see UIDs
 */
class GDCM_EXPORT TransferSyntax
{
public:
  typedef enum {
    Unknown = 0,
    Explicit,
    Implicit
  } NegociatedType;

#if 0
  //NOT FLEXIBLE, since force user to update lib everytime new module
  //comes out...
  // TODO
  typedef enum {
    NoSpacing = 0,
    PixelSpacing,
    ImagerPixelSpacing,
    PixelAspectRatio
  } ImageSpacingType;
  ImageSpacingType GetImageSpacing();
#endif

  typedef enum {
    ImplicitVRLittleEndian = 0,
    ImplicitVRBigEndianPrivateGE,
    ExplicitVRLittleEndian,
    DeflatedExplicitVRLittleEndian,
    ExplicitVRBigEndian,
    JPEGBaselineProcess1,
    JPEGExtendedProcess2_4,
    JPEGExtendedProcess3_5,
    JPEGSpectralSelectionProcess6_8,
    JPEGFullProgressionProcess10_12,
    JPEGLosslessProcess14,
    JPEGLosslessProcess14_1,
    JPEGLSLossless,
    JPEGLSNearLossless,
    JPEG2000Lossless,
    JPEG2000,
    JPEG2000Part2Lossless,
    JPEG2000Part2,
    RLELossless,
    MPEG2MainProfile,
    ImplicitVRBigEndianACRNEMA,
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
    WeirdPapryus,
#endif
    CT_private_ELE,
    JPIPReferenced,
    TS_END
  } TSType;

  // Return the string as written in the official DICOM dict from
  // a custom enum type
  static const char* GetTSString(TSType ts);
  static TSType GetTSType(const char *str);

  NegociatedType GetNegociatedType() const;

  /// \deprecated Return the SwapCode associated with the Transfer Syntax. Be careful with
  /// the special GE private syntax the DataSet is written in little endian but
  /// the Pixel Data is in Big Endian.
  SwapCode GetSwapCode() const;

  bool IsValid() const { return TSField != TS_END; }

  operator TSType () const { return TSField; }

  // FIXME: ImplicitVRLittleEndian used to be the default, but nowadays
  // this is rather the ExplicitVRLittleEndian instead...should be change the default ?
  TransferSyntax(TSType type = ImplicitVRLittleEndian):TSField(type) {}

  // return if dataset is encoded or not (Deflate Explicit VR)
  bool IsEncoded() const;

  bool IsImplicit() const;
  bool IsExplicit() const;

  bool IsEncapsulated() const;

  /** Return true if the transfer syntax algorithm is a lossy algorithm */
  bool IsLossy() const;
  /** Return true if the transfer syntax algorithm is a lossless algorithm */
  bool IsLossless() const;
  /** return true if TransFer Syntax Allow storing of Lossy Pixel Data */
  bool CanStoreLossy() const;

  const char *GetString() const { return TransferSyntax::GetTSString(TSField); }

  friend std::ostream &operator<<(std::ostream &os, const TransferSyntax &ts);
private:
  // DO NOT EXPOSE the following. Internal details of TransferSyntax
bool IsImplicit(TSType ts) const;
bool IsExplicit(TSType ts) const;
bool IsLittleEndian(TSType ts) const;
bool IsBigEndian(TSType ts) const;

  TSType TSField;
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &_os, const TransferSyntax &ts)
{
  _os << TransferSyntax::GetTSString(ts);
  return _os;

}

} // end namespace gdcm

#endif //GDCMTRANSFERSYNTAX_H
