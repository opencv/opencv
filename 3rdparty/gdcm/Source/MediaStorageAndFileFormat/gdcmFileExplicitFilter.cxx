/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileExplicitFilter.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmFragment.h"
#include "gdcmGlobal.h"
#include "gdcmDict.h"
#include "gdcmDicts.h"
#include "gdcmGroupDict.h"
#include "gdcmVR.h"
#include "gdcmVM.h"
#include "gdcmDataSetHelper.h"

namespace gdcm
{

void FileExplicitFilter::SetRecomputeItemLength(bool b)
{
  RecomputeItemLength = b;
}

void FileExplicitFilter::SetRecomputeSequenceLength(bool b)
{
  RecomputeSequenceLength = b;
}

bool FileExplicitFilter::ChangeFMI()
{
/*
    FileMetaInformation &fmi = F->GetHeader();
    TransferSyntax ts = TransferSyntax::ImplicitVRLittleEndian;
      {
      ts = TransferSyntax::ExplicitVRLittleEndian;
      }
    const char *tsuid = TransferSyntax::GetTSString( ts );
    DataElement de( Tag(0x0002,0x0010) );
    de.SetByteValue( tsuid, strlen(tsuid) );
    de.SetVR( Attribute<0x0002, 0x0010>::GetVR() );
    fmi.Replace( de );
    //fmi.Remove( Tag(0x0002,0x0012) ); // will be regenerated
    //fmi.Remove( Tag(0x0002,0x0013) ); //  '   '    '
    fmi.SetDataSetTransferSyntax(ts);
*/

  return true;
}

bool FileExplicitFilter::ProcessDataSet(DataSet &ds, Dicts const & dicts)
{
  if( RecomputeSequenceLength || RecomputeItemLength )
    {
    gdcmWarningMacro( "Not implemented sorry" );
    return false;
    }
  DataSet::Iterator it = ds.Begin();
  for( ; it != ds.End(); )
    {
    DataElement de = *it;
    std::string strowner;
    const char *owner = 0;
    const Tag& t = de.GetTag();
    if( t.IsPrivate() && !ChangePrivateTags
    // As a special exception we convert to proper VR :
    // - Private Group Length
    // - Private Creator
    // This makes the output more readable (and this should be relative safe)
      && !t.IsGroupLength() && !t.IsPrivateCreator()
    )
      {
      // nothing to do ! just skip
      ++it;
      continue;
      }
    if( t.IsPrivate() && !t.IsPrivateCreator() )
      {
      strowner = ds.GetPrivateCreator(t);
      owner = strowner.c_str();
      }
    const DictEntry &entry = dicts.GetDictEntry(t,owner);
    const VR &vr = entry.GetVR();

    //assert( de.GetVR() == VR::INVALID );
    VR cvr = DataSetHelper::ComputeVR(*F,ds, t);
    VR oldvr = de.GetVR();
    //SequenceOfItems *sqi = de.GetSequenceOfItems();
    //SequenceOfItems *sqi = dynamic_cast<SequenceOfItems*>(&de.GetValue());
    SmartPointer<SequenceOfItems> sqi = 0;
    if( vr == VR::SQ )
      {
      sqi = de.GetValueAsSQ();
      if(!sqi)
        {
        assert( de.IsEmpty() );
        }
      }
    if( de.GetByteValue() && !sqi )
      {
      // all set
      //assert( cvr != VR::SQ /*&& cvr != VR::UN*/ );
      assert( cvr != VR::INVALID );
      if( cvr != VR::UN )
        {
        // about to change , make some paranoid test:
        //assert( cvr.Compatible( oldvr ) ); // LT != LO but there are somewhat compatible
        if( cvr & VR::VRASCII )
          {
          //assert( oldvr & VR::VRASCII || oldvr == VR::INVALID || oldvr == VR::UN );
          // gdcm-JPEG-Extended.dcm has a couple of VR::OB private field
          // is this a good idea to change them to an ASCII when we know this might not work ?
          if( !(oldvr & VR::VRASCII || oldvr == VR::INVALID || oldvr == VR::UN) )
            {
            gdcmErrorMacro( "Cannot convert VR for tag: " << t << " " << oldvr << " is incompatible with " << cvr << " as given by ref. dict." );
            return false;
            }
          }
        else if( cvr & VR::VRBINARY )
          {
          // PHILIPS_Gyroscan-12-MONO2-Jpeg_Lossless.dcm
          if( !( oldvr & VR::VRBINARY || oldvr == VR::INVALID || oldvr == VR::UN ) )
            {
            gdcmErrorMacro( "Cannot convert VR for tag: " << t << " " << oldvr << " is incompatible with " << cvr << " as given by ref. dict." );
            return false;
            }
          }
        else
          {
          assert( 0 ); // programmer error
          }

        // let's do one more check we are going to make this attribute explicit VR, there is
        // still a special case, when VL is > uint16_max then we must give up:
        if( !(cvr & VR::VL32) && de.GetVL() > UINT16_MAX )
          {
          cvr = VR::UN;
          }
        de.SetVR( cvr );
        }
      }
    else if( sqi )
      {
      assert( cvr == VR::SQ || cvr == VR::UN );
      de.SetVR( VR::SQ );
      if( de.GetByteValue() )
        {
        de.SetValue( *sqi );
        //de.SetVL( sqi->ComputeLength<ExplicitDataElement>() );
        }
      de.SetVLToUndefined();
      assert( sqi->GetLength().IsUndefined() );
      // recursive
      SequenceOfItems::ItemVector::iterator sit = sqi->Items.begin();
      for(; sit != sqi->Items.end(); ++sit)
        {
        //Item &item = const_cast<Item&>(*sit);
        Item &item = *sit;
        item.SetVLToUndefined();
        DataSet &nds = item.GetNestedDataSet();
        //const DataElement &deitem = item;
        ProcessDataSet(nds, dicts);
        item.SetVL( item.GetLength<ExplicitDataElement>() );
        }
      }
    else if( de.GetSequenceOfFragments() )
      {
      assert( cvr & VR::OB_OW );
      }
    else
      {
      // Ok length is 0, it can be a 0 length explicit SQ (implicit) or a ByteValue...
      // we cannot make any error here, simply change the VR
      de.SetVR( cvr );
      }
    ++it;
    ds.Replace( de );
    }
  return true;
}

bool FileExplicitFilter::Change()
{
  //if( !UseVRUN)
  //  {
  //  gdcmErrorMacro( "Not implemented" );
  //  return false;
  //  }
  const Global& g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();

  DataSet &ds = F->GetDataSet();

  bool b = ProcessDataSet(ds, dicts);

  return b;
}


} // end namespace gdcm
