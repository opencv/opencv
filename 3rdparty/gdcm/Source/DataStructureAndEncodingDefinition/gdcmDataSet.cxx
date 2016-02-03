/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDataSet.h"
#include "gdcmPrivateTag.h"

namespace gdcm
{
DataElement DataSet::DEEnd = DataElement( Tag(0xffff,0xffff) );

const DataElement& DataSet::GetDEEnd() const
{
  return DEEnd;
}

std::string DataSet::GetPrivateCreator(const Tag &t) const
{
  if( t.IsPrivate() && !t.IsPrivateCreator() )
    {
    Tag pc = t.GetPrivateCreator();
    if( pc.GetElement() )
      {
      const DataElement r(pc);
      ConstIterator it = DES.find(r);
      if( it == DES.end() )
        {
        // FIXME, could this happen ?
        return "";
        }
      const DataElement &de = *it;
      if( de.IsEmpty() ) return "";
      const ByteValue *bv = de.GetByteValue();
      assert( bv );
      std::string owner = std::string(bv->GetPointer(),bv->GetLength());
      // There should not be any trailing space character...
      // TODO: tmp.erase(tmp.find_last_not_of(' ') + 1);
      while( owner.size() && owner[owner.size()-1] == ' ' )
        {
        // osirix/AbdominalCT/36382443
        owner.erase(owner.size()-1,1);
        }
      assert( owner.size() == 0 || owner[owner.size()-1] != ' ' );
      return owner;
      }
    }
  return "";
}

Tag DataSet::ComputeDataElement(const PrivateTag & t) const
{
  gdcmDebugMacro( "Entering ComputeDataElement" );
  //assert( t.IsPrivateCreator() ); // No this is wrong to do the assert: eg. (0x07a1,0x000a,"ELSCINT1")
  // is valid because we have not yet done the mapping, so 0xa < 0x10 fails but might not later on
  const Tag start(t.GetGroup(), 0x0010 ); // First possible private creator (0x0 -> 0x9 are reserved...)
  const DataElement r(start);
  ConstIterator it = DES.lower_bound(r);
  const char *refowner = t.GetOwner();
  assert( refowner );
  bool found = false;
  while( it != DES.end() && it->GetTag().GetGroup() == t.GetGroup() && it->GetTag().GetElement() < 0x100 )
    {
    //assert( it->GetTag().GetOwner() );
    const ByteValue * bv = it->GetByteValue();
    if( bv )
      {
      //if( strcmp( bv->GetPointer(), refowner ) == 0 )
      std::string tmp(bv->GetPointer(),bv->GetLength());
      // trim trailing whitespaces:
      tmp.erase(tmp.find_last_not_of(' ') + 1);
      assert( tmp.size() == 0 || tmp[ tmp.size() - 1 ] != ' ' ); // FIXME
      if( System::StrCaseCmp( tmp.c_str(), refowner ) == 0 )
        {
        // found !
        found = true;
        break;
        }
      }
    ++it;
    }
  gdcmDebugMacro( "In compute found is:" << found );
  if (!found) return GetDEEnd().GetTag();
  // else
  // ok we found the Private Creator Data Element, let's construct the proper data element
  Tag copy = t;
  copy.SetPrivateCreator( it->GetTag() );
  gdcmDebugMacro( "Compute found:" << copy );
  return copy;
}

bool DataSet::FindDataElement(const PrivateTag &t) const
{
  return FindDataElement( ComputeDataElement(t) );
}

const DataElement& DataSet::GetDataElement(const PrivateTag &t) const
{
  return GetDataElement( ComputeDataElement(t) );
}

MediaStorage DataSet::GetMediaStorage() const
{
  // Let's check 0008,0016:
  // D 0008|0016 [UI] [SOP Class UID] [1.2.840.10008.5.1.4.1.1.7 ]
  // ==> [Secondary Capture Image Storage]
  const Tag tsopclassuid(0x0008, 0x0016);
  if( !FindDataElement( tsopclassuid) )
    {
    gdcmDebugMacro( "No SOP Class UID" );
    return MediaStorage::MS_END;
    }
  const DataElement &de = GetDataElement(tsopclassuid);
  if( de.IsEmpty() )
    {
    gdcmDebugMacro( "Empty SOP Class UID" );
    return MediaStorage::MS_END;
    }
  std::string ts;
    {
    const ByteValue *bv = de.GetByteValue();
    assert( bv );
    if( bv->GetPointer() && bv->GetLength() )
      {
      // Pad string with a \0
      ts = std::string(bv->GetPointer(), bv->GetLength());
      }
    }
  // Paranoid check: if last character of a VR=UI is space let's pretend this is a \0
  if( ts.size() )
    {
    char &last = ts[ts.size()-1];
    if( last == ' ' )
      {
      gdcmWarningMacro( "Media Storage Class UID: " << ts << " contained a trailing space character" );
      last = '\0';
      }
    }
  gdcmDebugMacro( "TS: " << ts );
  MediaStorage ms = MediaStorage::GetMSType(ts.c_str());
  if( ms == MediaStorage::MS_END )
    {
    gdcmWarningMacro( "Media Storage Class UID: " << ts << " is unknow" );
    }
  return ms;
}

} // end namespace gdcm
