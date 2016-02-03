/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSequenceOfItems.h"

namespace gdcm
{

void SequenceOfItems::AddItem(Item const &item)
{
  Items.push_back( item );
  if( !SequenceLengthField.IsUndefined() )
    {
    assert(0); // TODO
    }
}

void SequenceOfItems::Clear()
{
  Items.clear();
  assert( SequenceLengthField.IsUndefined() );
}

bool SequenceOfItems::RemoveItemByIndex( const SizeType position )
{
  if( position < 1 || position > Items.size() )
    {
    return false;
    }
  Items.erase (Items.begin() + position);
  return true;
}

Item &SequenceOfItems::GetItem(SizeType position)
{
  if( position < 1 || position > Items.size() )
    {
    throw Exception( "Out of Range" );
    }
  return Items[position-1];
}

const Item &SequenceOfItems::GetItem(SizeType position) const
{
  if( position < 1 || position > Items.size() )
    {
    throw Exception( "Out of Range" );
    }
  return Items[position-1];
}

void SequenceOfItems::SetLengthToUndefined()
{
  SequenceLengthField = 0xFFFFFFFF;
}

bool SequenceOfItems::FindDataElement(const Tag &t) const
{
  ConstIterator it = Begin();
  bool found = false;
  for(; it != End() && !found; ++it)
    {
    const Item & item = *it;
    found = item.FindDataElement( t );
    }
  return found;
}

} // end namespace gdcm
