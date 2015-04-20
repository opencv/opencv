/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmSequenceOfFragments.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmByteValue.h"

namespace gdcm
{

void SequenceOfFragments::Clear()
{
  Table.SetByteValue( "", 0 );
  Fragments.clear();
}

SequenceOfFragments::SizeType SequenceOfFragments::GetNumberOfFragments() const
{
  // Do not count the last fragment
  //assert( SequenceLengthField.IsUndefined() );
  return Fragments.size();
}

void SequenceOfFragments::AddFragment(Fragment const &item)
{
  Fragments.push_back(item);
}

VL SequenceOfFragments::ComputeLength() const
{
  VL length = 0;
  // First the table
  length += Table.GetLength();
  // Then all the fragments
  FragmentVector::const_iterator it = Fragments.begin();
  for(;it != Fragments.end(); ++it)
    {
    const VL fraglen = it->ComputeLength();
    assert( fraglen % 2 == 0 );
    length += fraglen;
    }
  assert( SequenceLengthField.IsUndefined() );
  length += 8; // seq end delimitor (tag + vl)
  return length;
}

unsigned long SequenceOfFragments::ComputeByteLength() const
{
  unsigned long r = 0;
  FragmentVector::const_iterator it = Fragments.begin();
  for(;it != Fragments.end(); ++it)
    {
    assert( !it->GetVL().IsUndefined() );
    r += it->GetVL();
    }
  return r;
}

bool SequenceOfFragments::GetFragBuffer(unsigned int fragNb, char *buffer, unsigned long &length) const
{
  FragmentVector::const_iterator it = Fragments.begin();
    {
    const Fragment &frag = *(it+fragNb);
    const ByteValue &bv = dynamic_cast<const ByteValue&>(frag.GetValue());
    const VL len = frag.GetVL();
    bv.GetBuffer(buffer, len);
    length = len;
    }
  return true;
}

const Fragment& SequenceOfFragments::GetFragment(SizeType num) const
{
  assert( num < Fragments.size() );
  FragmentVector::const_iterator it = Fragments.begin();
  const Fragment &frag = *(it+num);
  return frag;
}

bool SequenceOfFragments::GetBuffer(char *buffer, unsigned long length) const
{
  FragmentVector::const_iterator it = Fragments.begin();
  unsigned long total = 0;
  for(;it != Fragments.end(); ++it)
    {
    const Fragment &frag = *it;
    const ByteValue &bv = dynamic_cast<const ByteValue&>(frag.GetValue());
    const VL len = frag.GetVL();
    bv.GetBuffer(buffer, len);
    buffer += len;
    total += len;
    }
  if( total != length )
    {
    //std::cerr << " DEBUG: " << total << " " << length << std::endl;
    assert(0);
    return false;
    }
  return true;
}

bool SequenceOfFragments::WriteBuffer(std::ostream &os) const
{
  FragmentVector::const_iterator it = Fragments.begin();
  unsigned long total = 0;
  for(;it != Fragments.end(); ++it)
    {
    const Fragment &frag = *it;
    const ByteValue *bv = frag.GetByteValue();
    assert( bv );
    const VL len = frag.GetVL();
    bv->WriteBuffer(os);
    total += len;
    }
  //if( total != length )
  //  {
  //  //std::cerr << " DEBUG: " << total << " " << length << std::endl;
  //  assert(0);
  //  return false;
  //  }
  return true;
}

bool SequenceOfFragments::FillFragmentWithJPEG( Fragment & frag, std::istream & is )
{
  std::vector<unsigned char> jfif;
  unsigned char byte;
  // begin /simple/ JPEG parser:
  while( is.read( (char*)&byte, 1 ) )
    {
    jfif.push_back( byte );
    if( byte == 0xd9 && jfif[ jfif.size() - 2 ] == 0xff ) break;
    }
  const uint32_t len = static_cast<uint32_t>(jfif.size());
  frag.SetByteValue( (char*)&jfif[0], len );
  return true;
}

} // end namespace gdcm
