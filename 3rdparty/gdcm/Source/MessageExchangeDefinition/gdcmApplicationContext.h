/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMAPPLICATIONCONTEXT_H
#define GDCMAPPLICATIONCONTEXT_H

#include "gdcmTypes.h"

namespace gdcm
{

namespace network
{

/**
 * \brief ApplicationContext
 * Table 9-12
 * APPLICATION CONTEXT ITEM FIELDS
 * \todo
 * Looks like Application Context can only be 64 bytes at max (see Figure 9-1 / PS 3.8 - 2009 )
 */
class ApplicationContext
{
public:
  ApplicationContext();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  void SetName( const char *name ) { UpdateName( name ); }
  const char *GetName() const { return Name.c_str(); }
  size_t Size() const;

  //static const uint8_t GetItemType() { return ItemType; }
  void Print(std::ostream &os) const;

private:
  void UpdateName( const char *name );
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength; // len of application context name
  std::string /*ApplicationContext*/ Name; // UID
};

} // end namespace network

} // end namespace gdcm

#endif //GDCMAPPLICATIONCONTEXT_H
