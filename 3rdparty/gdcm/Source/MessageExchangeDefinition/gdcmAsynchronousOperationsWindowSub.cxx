/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmAsynchronousOperationsWindowSub.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmSwapper.h"

namespace gdcm
{
namespace network
{
const uint8_t AsynchronousOperationsWindowSub::ItemType = 0x53;
const uint8_t AsynchronousOperationsWindowSub::Reserved2 = 0x00;

AsynchronousOperationsWindowSub::AsynchronousOperationsWindowSub()
{
  ItemLength = 0;
  MaximumNumberOperationsInvoked = 0;
  MaximumNumberOperationsPerformed = 0;

  ItemLength = (uint16_t)(Size() - 4);
  assert( (size_t)ItemLength + 4 == Size() );
}

std::istream &AsynchronousOperationsWindowSub::Read(std::istream &is)
{
  //uint8_t itemtype = 0x0;
  //is.read( (char*)&itemtype, sizeof(ItemType) );
  //assert( itemtype == ItemType );
  uint8_t reserved2;
  is.read( (char*)&reserved2, sizeof(Reserved2) );
  uint16_t itemlength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;

  uint16_t maximumnumberoperationsinvoked;
  is.read( (char*)&maximumnumberoperationsinvoked, sizeof(MaximumNumberOperationsInvoked) );
  SwapperDoOp::SwapArray(&maximumnumberoperationsinvoked,1);
  MaximumNumberOperationsInvoked = maximumnumberoperationsinvoked;

  uint16_t maximumnumberoperationsperformed;
  is.read( (char*)&maximumnumberoperationsperformed, sizeof(MaximumNumberOperationsPerformed) );
  SwapperDoOp::SwapArray(&maximumnumberoperationsperformed,1);
  MaximumNumberOperationsPerformed = maximumnumberoperationsperformed;

  return is;
}

const std::ostream &AsynchronousOperationsWindowSub::Write(std::ostream &os) const
{
  os.write( (char*)&ItemType, sizeof(ItemType) );
  os.write( (char*)&Reserved2, sizeof(Reserved2) );
  //os.write( (char*)&ItemLength, sizeof(ItemLength) );
  uint16_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );

  uint16_t maximumnumberoperationsinvoked = MaximumNumberOperationsInvoked;
  SwapperDoOp::SwapArray(&maximumnumberoperationsinvoked,1);
  os.write( (char*)&maximumnumberoperationsinvoked, sizeof(MaximumNumberOperationsInvoked) );

  uint16_t maximumnumberoperationsperformed = MaximumNumberOperationsPerformed;
  SwapperDoOp::SwapArray(&maximumnumberoperationsperformed,1);
  os.write( (char*)&maximumnumberoperationsperformed, sizeof(MaximumNumberOperationsPerformed) );

  return os;
}

size_t AsynchronousOperationsWindowSub::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemType);
  ret += sizeof(Reserved2);
  ret += sizeof(ItemLength);
  ret += sizeof(MaximumNumberOperationsInvoked);
  ret += sizeof(MaximumNumberOperationsPerformed);

  return ret;
}

void AsynchronousOperationsWindowSub::Print(std::ostream &os) const
{
  os << "MaximumNumberOperationsInvoked: " << MaximumNumberOperationsInvoked << std::endl;
  os << "MaximumNumberOperationsPerformed: " << MaximumNumberOperationsPerformed << std::endl;
}

} // end namespace network
} // end namespace gdcm
