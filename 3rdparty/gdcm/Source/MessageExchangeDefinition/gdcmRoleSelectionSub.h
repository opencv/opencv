/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMROLESELECTIONSUB_H
#define GDCMROLESELECTIONSUB_H

#include "gdcmTypes.h"

namespace gdcm
{

namespace network
{

/**
 * \brief RoleSelectionSub
 * PS 3.7
 * Table D.3-9
 * SCP/SCU ROLE SELECTION SUB-ITEM FIELDS (A-ASSOCIATE-RQ)
 */
class RoleSelectionSub
{
public:
  RoleSelectionSub();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  size_t Size() const;
  void Print(std::ostream &os) const;

  void SetTuple(const char *uid, uint8_t scurole, uint8_t scprole);

private:
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength;
  uint16_t UIDLength;
  std::string /*SOP-class-uid*/ Name; // UID
  uint8_t SCURole;
  uint8_t SCPRole;
};

} // end namespace network

} // end namespace gdcm

#endif // GDCMROLESELECTIONSUB_H
