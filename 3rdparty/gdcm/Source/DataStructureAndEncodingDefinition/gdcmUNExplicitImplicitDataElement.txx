/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMUNEXPLICITIMPLICITDATAELEMENT_TXX
#define GDCMUNEXPLICITIMPLICITDATAELEMENT_TXX

#include "gdcmSequenceOfItems.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmVL.h"
#include "gdcmUNExplicitDataElement.h"
#include "gdcmImplicitDataElement.h"

#include "gdcmValueIO.h"
#include "gdcmSwapper.h"

namespace gdcm
{
//-----------------------------------------------------------------------------
template <typename TSwap>
std::istream &UNExplicitImplicitDataElement::Read(std::istream &is)
{
  ReadPreValue<TSwap>(is);
  return ReadValue<TSwap>(is);
}

template <typename TSwap>
std::istream &UNExplicitImplicitDataElement::ReadPreValue(std::istream &is)
{
  assert(0);
  //DataElement &de = *this;//unused de, fires a warning on macos
  //de.template ReadPreValue<UNExplicitDataElement,TSwap>( is );
  return is;
}

template <typename TSwap>
std::istream &UNExplicitImplicitDataElement::ReadValue(std::istream &is)
{
  if( is.eof() ) return is;
  DataElement &de = *this;
  try
    {
    //de.template ReadValue<UNExplicitDataElement,TSwap>( is );
    }
  catch(ParseException &ex)
    {
    de.SetVR( VR::INVALID ); // EXTREMELY IMPORTANT !
    if( ex.GetLastElement().GetTag() == Tag(0xfffe,0xe0dd) )
      {
      // We have never read the 2 bytes for the VR, since exception raised earlier
      is.seekg( -4, std::ios::cur );
      }
    else
      {
      is.seekg( -6, std::ios::cur );
      }
    //de.template ReadValue<ImplicitDataElement,TSwap>( is );
    }
  return is;
}


} // end namespace gdcm

#endif // GDCMUNEXPLICITIMPLICITDATAELEMENT_TXX
