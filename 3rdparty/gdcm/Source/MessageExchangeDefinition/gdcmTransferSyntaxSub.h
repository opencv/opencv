/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTRANSFERSYNTAXSUB_H
#define GDCMTRANSFERSYNTAXSUB_H

#include "gdcmTypes.h"
#include "gdcmTransferSyntax.h"
#include "gdcmUIDs.h"

namespace gdcm
{

namespace network
{

/**
 * \brief TransferSyntaxSub
 * Table 9-15
 * TRANSFER SYNTAX SUB-ITEM FIELDS
 *
 * TODO what is the goal of :
 *
 * Table 9-19
 * TRANSFER SYNTAX SUB-ITEM FIELDS
 */
class TransferSyntaxSub
{
public:
  TransferSyntaxSub();
  void SetName( const char *name );
  const char *GetName() const { return Name.c_str(); }

  // accept a UIDs::TSType also...
  void SetNameFromUID( UIDs::TSName tsname );

  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;
  size_t Size() const;
  void Print(std::ostream &os) const;

  bool operator==(const TransferSyntaxSub & ts) const
    {
    return Name == ts.Name;
    }

private:
  void UpdateName( const char *name );
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength; // len of
  std::string /*TransferSyntaxSub*/ Name; // UID
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMTRANSFERSYNTAXSUB_H
