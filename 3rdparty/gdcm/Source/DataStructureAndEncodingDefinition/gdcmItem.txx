/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMITEM_TXX
#define GDCMITEM_TXX

#include "gdcmItem.h"

namespace gdcm
{

template <typename TDE>
VL Item::GetLength() const
{
  if( ValueLengthField.IsUndefined() )
    {
    assert( !NestedDataSet.GetLength<TDE>().IsUndefined() );
    // Item Start             4
    // Item Length            4
    // DataSet                ?
    // Item End Delimitation  4
    // Item End Length        4
    assert( NestedDataSet.GetLength<TDE>() % 2 == 0 );
    return TagField.GetLength() /* 4 */ + ValueLengthField.GetLength() /* 4 */
      + NestedDataSet.GetLength<TDE>() + 4 + 4;
    }
  else
    {
    // Item Start             4
    // Item Length            4
    // DataSet                ?
    const VL nestedlen = NestedDataSet.GetLength<TDE>();
    // The following line is commented out, since we could be in the case where
    // we are computing the actual length of an implicit dataset, from an
    // initially read explicit dataset in which case the two length cannot
    // related to each other
    //gdcmAssertAlwaysMacro( ValueLengthField == nestedlen );
    assert( nestedlen % 2 == 0 );
    return TagField.GetLength() /* 4 */ + ValueLengthField.GetLength() /* 4 */
      //+ ValueLengthField;
      + nestedlen;
    }
}

} // end namespace gdcm

#endif // GDCMITEM_TXX
