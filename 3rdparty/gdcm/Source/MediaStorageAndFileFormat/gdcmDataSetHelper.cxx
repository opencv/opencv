/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDataSetHelper.h"
#include "gdcmFile.h"
#include "gdcmDataSet.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"
#include "gdcmDict.h"
#include "gdcmAttribute.h"

namespace gdcm
{
/*
    See PS 3.5 - 2008
    Annex A (Normative) Transfer Syntax Specifications
*/

VR ComputeVRImplicitLittleEndian(DataSet const &ds, const Tag& tag)
{
  (void)ds;
  (void)tag;
    /*
    A.1 DICOM IMPLICIT VR LITTLE ENDIAN TRANSFER SYNTAX
    a) The Data Elements contained in the Data Set structure shall be encoded with Implicit VR
    (without a VR Field) as specified in Section 7.1.3.
    b) The encoding of the overall Data Set structure (Data Element Tags, Value Length, and Value)
    shall be in Little Endian as specified in Section 7.3.
    c) The encoding of the Data Elements of the Data Set shall be as follows according to their Value
    Representations:
      - For all Value Representations defined in this part, except for the Value Representations
      OB and OW, the encoding shall be in Little Endian as specified in Section 7.3
      - For the Value Representations OB and OW, the encoding shall meet the following
      specification depending on the Data Element Tag:
        - Data Element (7FE0,0010) Pixel Data has the Value Representation OW and shall
        be encoded in Little Endian.
        - Data Element (60xx,3000) Overlay Data has the Value Representation OW and shall
        be encoded in Little Endian.
        - Data Element (5400,1010) Waveform Data shall have Value Representation OW
        and shall be encoded in Little Endian.
        - Data Elements (0028,1201), (0028,1202),(0028,1203) Red, Green, Blue Palette
        Lookup Table Data have the Value Representation OW and shall be encoded in
        Little Endian.
        Note: Previous versions of the Standard either did not specify the encoding of these Data
        Elements in this Part, but specified a VR of US or SS in PS 3.6 (1993), or specified
        OW in this Part but a VR of US, SS or OW in PS 3.6 (1996). The actual encoding
        of the values and their byte order would be identical in each case.
        - Data Elements (0028,1101), (0028,1102),(0028,1103) Red, Green, Blue Palette
        Lookup Table Descriptor have the Value Representation SS or US (depending on
        rules specified in the IOD in PS 3.3), and shall be encoded in Little Endian. The first
        and third values are always interpreted as unsigned, regardless of the Value
        Representation.
        - Data Elements (0028,1221),(0028,1222),(0028,1223) Segmented Red, Green, Blue
        Palette Color Lookup table Data have the Value Representation OW and shall be
        encoded in Little Endian.
        - Data Element (0028,3006) Lookup Table Data has the Value Representation US, SS
        or OW and shall be encoded in Little Endian.
        - Data Element (0028,3002) Lookup Table Descriptor has the Value Representation
        SS or US (depending on rules specified in the IOD in PS 3.3), and shall be encoded
        in Little Endian. The first and third values are always interpreted as unsigned,
        regardless of the Value Representation.
    */
  VR vr = VR::INVALID;
  return vr;
}

VR DataSetHelper::ComputeVR(File const &file, DataSet const &ds, const Tag& tag)
{
  const Global& g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();
  //const Dict &d = dicts.GetPublicDict();

  std::string strowner;
  const char *owner = 0;
  const Tag& t = tag;
  if( t.IsPrivate() && !t.IsPrivateCreator() )
    {
    strowner = ds.GetPrivateCreator(t);
    owner = strowner.c_str();
    }
  const DictEntry &entry = dicts.GetDictEntry(t,owner);
  const VR &refvr = entry.GetVR();
  //const VM &vm = entry.GetVM();

  // not much we can do...
  if( refvr == VR::INVALID || refvr == VR::UN )
    {
    // postcondition says it cannot be VR::INVALID, so return VR::UN
    return VR::UN;
    }

  VR vr = refvr;

  // Special handling of US or SS vr:
  if( vr == VR::US_SS )
    {
    // I believe all US_SS VR derived from the value from 0028,0103 ... except 0028,0071
    if( t != Tag(0x0028,0x0071) )
      {
      // In case of SAX parser, we would have had to process Pixel Representation already:
      Attribute<0x0028,0x0103> at;
      const Tag &pixelrep = at.GetTag();
      assert( pixelrep < t );
      const DataSet &rootds = file.GetDataSet();
      // FIXME
      // PhilipsWith15Overlays.dcm has a Private SQ with public elements such as
      // 0028,3002, so we cannot look up element in current dataset, but have to get the root dataset
      // to loop up...

      // FIXME:
      // gdcmDataExtra/gdcmSampleData/ImagesPapyrus/TestImages/wristb.pap
      // It's the contrary: root dataset does not have a Pixel Representation, but each SQ do...
      assert( rootds.FindDataElement( pixelrep ) || ds.FindDataElement( pixelrep ) );
      if( ds.FindDataElement( pixelrep ) )
        {
        at.SetFromDataElement( ds.GetDataElement( pixelrep ) );
        }
      else if( rootds.FindDataElement( pixelrep ) )
        {
        at.SetFromDataElement( rootds.GetDataElement( pixelrep ) );
        }
      else
        {
        //throw Exception( "Unhandled" );
        gdcmWarningMacro( "Unhandled" );
        vr = VR::INVALID;
        }
      //assert( at.GetValue() == 0 || at.GetValue() == 1 );
      if( at.GetValue() )
        {
        vr = VR::SS;
        }
      else
        {
        vr = VR::US;
        }
      }
    else
      {
      // FIXME ???
      vr = VR::US;
      }
    }
  else if( vr == VR::OB_OW )
    {
    Tag pixeldata(0x7fe0,0x0010);
    Tag waveformpaddingvalue(0x5400,0x100a);
    Tag waveformdata(0x5400,0x1010);
    Tag overlaydata(0x6000,0x3000);
    Tag curvedata(0x5000,0x3000);
    Tag audiodata(0x5000,0x200c);
    Tag variablepixeldata(0x7f00,0x0010);
    Tag bitsallocated(0x0028,0x0100);
    Tag channelminval(0x5400,0x0110);
    Tag channelmaxval(0x5400,0x0112);
    //assert( ds.FindDataElement( pixeldata ) );
    int v = -1;
    if( waveformdata == t || waveformpaddingvalue == t )
      {
      Tag waveformbitsallocated(0x5400,0x1004);
      // For Waveform Data:
      // (5400,1004) US 16                                             # 2,1 Waveform Bits Allocated
      assert( ds.FindDataElement( waveformbitsallocated ) );
      Attribute<0x5400,0x1004> at;
      at.SetFromDataElement( ds.GetDataElement( waveformbitsallocated ) );
      v = at.GetValue();
      }
    else // ( pixeldata == t  )
      {
      // For Pixel Data:
      assert( ds.FindDataElement( bitsallocated ) );
      Attribute<0x0028,0x0100> at;
      at.SetFromDataElement( ds.GetDataElement( bitsallocated ) );
      }
    (void)v;

    if( pixeldata == t || t.IsGroupXX(overlaydata) )
      {
      vr = VR::OW;
      }
    else if( waveformdata == t || waveformpaddingvalue == t )
      {
      //assert( v == 8 || v == 16 );
      vr = VR::OW;
      }
    else if ( t.IsGroupXX(audiodata) )
      {
      vr = VR::OB;
      }
    else if ( t.IsGroupXX(curvedata) )
      {
      vr = VR::OB;
      }
    else if ( t.IsGroupXX(variablepixeldata) )
      {
      vr = VR::OB;
      }
    else if ( t == channelminval || t == channelmaxval )
      {
      vr = VR::OB;
      }
    else
      {
      assert( 0 && "Should not happen" );
      vr = VR::INVALID;
      }
    }
  else if( vr == VR::US_SS_OW )
    {
    vr = VR::OW;
    }
  // TODO need to treat US_SS_OW too

  // \postcondition:
  assert( vr.IsVRFile() );
  assert( vr != VR::INVALID );

  if( tag.IsGroupLength() )
    {
    assert( vr == VR::UL );
    }
  if( tag.IsPrivateCreator() )
    {
    assert( vr == VR::LO );
    }
  return vr;
}


/*
SequenceOfItems* DataSetHelper::ComputeSQFromByteValue(File const & file, DataSet const &ds, const Tag &tag)
{
  const TransferSyntax &ts = file.GetHeader().GetDataSetTransferSyntax();
  assert( ts != TransferSyntax::DeflatedExplicitVRLittleEndian );
  const DataElement &de = ds.GetDataElement( tag );
  if( de.IsEmpty() )
    {
    return 0;
    }
  Value &v = const_cast<Value&>(de.GetValue());
  SequenceOfItems *sq = dynamic_cast<SequenceOfItems*>(&v);
  if( sq ) // all set !
    {
    SmartPointer<SequenceOfItems> sqi = sq;
    return sqi;
    }

  try
    {
    if( ts.GetSwapCode() == SwapCode::BigEndian )
      {
      assert(0);
      }
    else
      {
      if( ts.GetNegociatedType() == TransferSyntax::Implicit )
        {
        assert( de.GetVR() == VR::INVALID );
        const ByteValue *bv = de.GetByteValue();
        assert( bv );
        SequenceOfItems *sqi = new SequenceOfItems;
        sqi->SetLength( bv->GetLength() );
        std::stringstream ss;
        ss.str( std::string( bv->GetPointer(), bv->GetLength() ) );
        sqi->Read<ImplicitDataElement,SwapperNoOp>( ss );
        return sqi;
        }
      else
        {
        assert( de.GetVR() == VR::UN ); // cp 246, IVRLE SQ
        const ByteValue *bv = de.GetByteValue();
        assert( bv );
        SequenceOfItems *sqi = new SequenceOfItems;
        sqi->SetLength( bv->GetLength() );
        std::stringstream ss;
        ss.str( std::string( bv->GetPointer(), bv->GetLength() ) );
        sqi->Read<ImplicitDataElement,SwapperNoOp>( ss );
        return sqi;
        }
      }
    }
  catch( ParseException &pex )
    {
    gdcmDebugMacro( pex.what() );
    }
  catch( Exception &ex )
    {
    gdcmDebugMacro( ex.what() );
    }
  catch( ... )
    {
    gdcmWarningMacro( "Unknown exception" );
    }

  return 0;
}
*/

}
