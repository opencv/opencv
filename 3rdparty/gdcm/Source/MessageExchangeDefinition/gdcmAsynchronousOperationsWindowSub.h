/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMASYNCHRONOUSOPERATIONSWINDOWSUB_H
#define GDCMASYNCHRONOUSOPERATIONSWINDOWSUB_H

#include "gdcmTypes.h"

namespace gdcm
{

namespace network
{

/**
 * \brief AsynchronousOperationsWindowSub
 * PS 3.7
 * Table D.3-7
 * ASYNCHRONOUS OPERATIONS WINDOW SUB-ITEM FIELDS
 * (A-ASSOCIATE-RQ)
 */
class AsynchronousOperationsWindowSub
{
public:
  AsynchronousOperationsWindowSub();
  std::istream &Read(std::istream &is);
  const std::ostream &Write(std::ostream &os) const;

  size_t Size() const;
  void Print(std::ostream &os) const;

private:
  static const uint8_t ItemType;
  static const uint8_t Reserved2;
  uint16_t ItemLength;
  uint16_t MaximumNumberOperationsInvoked;
  uint16_t MaximumNumberOperationsPerformed;
};

} // end namespace network

} // end namespace gdcm

#endif // GDCMASYNCHRONOUSOPERATIONSWINDOWSUB_H
