/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMPRESENTATIONCONTEXTAC_H
#define GDCMPRESENTATIONCONTEXTAC_H

#include "gdcmTypes.h"
#include "gdcmTransferSyntaxSub.h"

namespace gdcm
{

namespace network
{

/**
 * \brief PresentationContextAC
 * Table 9-18
 * PRESENTATION CONTEXT ITEM FIELDS
 * \see PresentationContext
 */
class PresentationContextAC
{
public:
  PresentationContextAC();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  size_t Size() const;

  void SetTransferSyntax( TransferSyntaxSub const &ts );
  void SetPresentationContextID( uint8_t id );

  void Print(std::ostream &os) const;

  uint8_t GetPresentationContextID() const
    {
    return ID;
    }
  TransferSyntaxSub const & GetTransferSyntax() const { return SubItems; }

  void SetReason( uint8_t r ) { Result = r; }
  uint8_t GetReason() const { return Result; }

private:
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength; // len of last transfer syntax
  uint8_t /*PresentationContext*/ID;
  static const uint8_t Reserved6;
  uint8_t /*Reason*/Result;
  static const uint8_t Reserved8;
  TransferSyntaxSub SubItems;
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMPRESENTATIONCONTEXTAC_H
