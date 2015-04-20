/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMUSERINFORMATION_H
#define GDCMUSERINFORMATION_H

#include "gdcmTypes.h"
#include "gdcmMaximumLengthSub.h"
#include "gdcmImplementationVersionNameSub.h"
#include "gdcmImplementationClassUIDSub.h"

namespace gdcm
{

namespace network
{

class AsynchronousOperationsWindowSub;
class RoleSelectionSub;
struct RoleSelectionSubItems;
class SOPClassExtendedNegociationSub;
struct SOPClassExtendedNegociationSubItems;
/**
 * \brief UserInformation
 * Table 9-16
 * USER INFORMATION ITEM FIELDS
 *
 * TODO what is the goal of :
 *
 * Table 9-20
 * USER INFORMATION ITEM FIELDS
 */
class UserInformation
{
public:
  UserInformation();
  ~UserInformation();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;
  size_t Size() const;

  void Print(std::ostream &os) const;

  const MaximumLengthSub &GetMaximumLengthSub() const { return MLS; }
  MaximumLengthSub &GetMaximumLengthSub() { return MLS; }

  void AddRoleSelectionSub( RoleSelectionSub const & r );
  void AddSOPClassExtendedNegociationSub( SOPClassExtendedNegociationSub const & s );

private:
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength; // len of
  MaximumLengthSub MLS;
  ImplementationClassUIDSub ICUID;
  AsynchronousOperationsWindowSub *AOWS;
  RoleSelectionSubItems *RSSI;
  SOPClassExtendedNegociationSubItems *SOPCENSI;
  ImplementationVersionNameSub IVNS;

  UserInformation(const UserInformation&); // Not implemented
public:
  UserInformation &operator=(const UserInformation&);
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMUSERINFORMATION_H
