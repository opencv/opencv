/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#include "gdcmCStoreMessages.h"

#include "gdcmUIDs.h"
#include "gdcmAttribute.h"
#include "gdcmFile.h"
#include "gdcmImplicitDataElement.h"
#include "gdcmPresentationContextRQ.h"
#include "gdcmCommandDataSet.h"
#include "gdcmBasePDU.h"
#include "gdcmPDataTFPDU.h"
#include "gdcmMediaStorage.h"
#include "gdcmULConnection.h"
#include "gdcmWriter.h"

#include <limits>

namespace gdcm{

class DataSetWriter : public Writer
{
public:
  DataSetWriter()
    {
    SetWriteDataSetOnly( true );
    }
};

namespace network{

std::vector<PresentationDataValue> CStoreRQ::ConstructPDV(
const ULConnection &inConnection, File const & file )
{
const DataSet* inDataSet = &file.GetDataSet();

  std::vector<PresentationDataValue> thePDVs;
  PresentationContextRQ pc( UIDs::VerificationSOPClass );
  uint8_t prescontid;
{
  assert( inDataSet );
  PresentationDataValue thePDV;
#if 0
  std::string UIDString;
  thePDV.SetPresentationContextID(
    PresentationContextRQ::AssignPresentationContextID(*inDataSet, UIDString));
#else
{
  MediaStorage mst;
  if (!mst.SetFromDataSet(*inDataSet))
    {
    throw Exception("Missing MediaStorage");
    }
  UIDs uid;
  uid.SetFromUID( MediaStorage::GetMSString(mst) );
  //as.SetNameFromUID( uid );
  pc.GetAbstractSyntax().SetName( uid.GetString() );
}

//  prescontid = inConnection.GetPresentationContextIDFromAbstractSyntax(as);
//  thePDV.SetPresentationContextID( prescontid );
#endif

  thePDV.SetCommand(true);
  thePDV.SetLastFragment(true);
  //ignore incoming data set, make your own

  CommandDataSet ds;
  {
  DataElement de( Tag(0x0,0x2) );
  de.SetVR( VR::UI );
  if( !inDataSet->FindDataElement( Tag(0x0008, 0x0016) ) ||
    inDataSet->GetDataElement( Tag(0x0008, 0x0016) ).IsEmpty() )
    {
    throw Exception("Missing SOP Class UID");
    }
  const DataElement& msclass = inDataSet->GetDataElement( Tag(0x0008, 0x0016) );
  const char *uid = msclass.GetByteValue()->GetPointer();
  assert( uid );
  std::string suid = std::string(uid, msclass.GetByteValue()->GetLength());

  // self check
//  const PresentationContextAC * pc = inConnection.GetPresentationContextACByID(prescontid);
//  assert( pc );
//  TransferSyntaxSub const &tssub = pc->GetTransferSyntax();
  const TransferSyntax & fmits = file.GetHeader().GetDataSetTransferSyntax();
  const char *tsuidvalue = fmits.GetString();
//  TransferSyntaxSub tssub;
//  tssub.SetName( tsuidvalue );
  pc.GetTransferSyntax(0).SetName( tsuidvalue );

  std::string tsuid = fmits.GetString();

  prescontid = inConnection.GetPresentationContextIDFromPresentationContext(pc);
  // prescontid cannot possibly be unknown since we are only looking in our own
  // AAssociateRQPDU
  assert( prescontid != 0 );
  const PresentationContextRQ * rqpc = inConnection.GetPresentationContextRQByID(prescontid);
  assert( rqpc );

  // Now let's see if this best matching PresentationContextRQ can be found in the AC
  // section of the AAssociateACPDU
  const PresentationContextAC * acpc = inConnection.GetPresentationContextACByID(prescontid);

  // the following make sure that the accepted Presentation Context match the actual encoding
  // of the current File
  // ADV: technically we could use an explicit VR encoded dataset and send it over
  // an implicit TS accecpted Transfer syntax. However thing do not interchange well
  // so we really need a filter to check whether conversion is ok or not.
  if( acpc == 0 )
    {
    // Technically we should fallback to something else. Anyway lets' give up
    // and hope the user will convert the encapsulated stream to something else...
    throw Exception("Server side refuse our proposed PC.");
    }

  TransferSyntaxSub const & actssub = acpc->GetTransferSyntax();
  assert( rqpc->GetNumberOfTransferSyntaxes() == 1 ); // TODO FIXME
  TransferSyntaxSub const & rqtssub = rqpc->GetTransferSyntax(0);
  if( !(actssub == rqtssub) )
    {
    gdcmDebugMacro( "Faulty Presentation Context : "
      << (int)acpc->GetPresentationContextID() );
    throw Exception("Server side refuse our proposed PC for context id" );
    }

#if 0
  // For some reason using a dcmtk 3.5.4 server. The PresCont even if refused returned
  // filled with the default Implicit Little Endian. So make sure TS matches
  TransferSyntaxSub const & actssub = acpc->GetTransferSyntax();
  TransferSyntaxSub const & dummy0 = pc.GetTransferSyntax(0);
  if( !(actssub == pc.GetTransferSyntax(0)) )
    {
    gdcmDebugMacro( "Faulty Presentation Context : "
      << (int)acpc->GetPresentationContextID() );
    throw Exception("Server side refuse our proposed PC for context id" );
    }
#endif

  thePDV.SetPresentationContextID( prescontid );

  assert(suid.size() < std::numeric_limits<uint32_t>::max());
  de.SetByteValue( suid.c_str(), (uint32_t)suid.size()  );
  ds.Insert( de );
  }

  {
  assert( inDataSet->FindDataElement( Tag(0x0008, 0x0018) ) );
  const DataElement& msinst = inDataSet->GetDataElement( Tag(0x0008, 0x0018) );
  std::string suid;
  DataElement de( Tag(0x0,0x1000) );
  de.SetVR( VR::UI );
  if( !msinst.IsEmpty() )
    {
    const ByteValue* bv = msinst.GetByteValue();
    if( bv )
      {
      const char *uid = bv->GetPointer();
      assert( uid );
      suid = std::string(uid, bv->GetLength() );
      assert(suid.size() < std::numeric_limits<uint32_t>::max());
      }
    }
  de.SetByteValue( suid.c_str(), (uint32_t)suid.size()  );
  ds.Insert( de );
  }

  {
  Attribute<0x0,0x100> at = { 1 };
  ds.Insert( at.GetAsDataElement() );
  }
static uint32_t messageid = 1;
  {
  Attribute<0x0,0x110> at = { 0 };
  at.SetValue( (unsigned short)messageid++ );
  assert( messageid < std::numeric_limits<uint32_t>::max());
  ds.Insert( at.GetAsDataElement() );
  }
  {
  Attribute<0x0,0x700> at = { 2 };
  ds.Insert( at.GetAsDataElement() );
  }
  {
  Attribute<0x0,0x800> at = { 1 };
  ds.Insert( at.GetAsDataElement() );
  }
  {
  Attribute<0x0,0x0> at = { 0 };
  unsigned int glen = ds.GetLength<ImplicitDataElement>();
  assert( (glen % 2) == 0 );
  at.SetValue( glen );
  ds.Insert( at.GetAsDataElement() );
  }

  thePDV.SetDataSet(ds);

  //!!!Mathieu, I assume you'll want to fix this
  thePDVs.push_back(thePDV);
}

  // now let's chunk'ate the dataset:
{
  std::stringstream ss;
#if 0
  inDataSet->Write<ImplicitDataElement,SwapperNoOp>( ss );
#else
  DataSetWriter writer;
  writer.SetStream( ss );
  writer.SetFile( file );
  writer.Write();
#endif

  std::string ds_copy = ss.str();
  // E: 0006:0308 DUL Illegal PDU Length 16390.  Max expected 16384
  //const size_t maxpdu = 16384 - 6;
  size_t maxpdu = 16378;
  maxpdu = inConnection.GetMaxPDUSize() - 6;
  size_t len = ds_copy.size();
  const char *begin = ds_copy.c_str();
  const char *end = begin + len;
  const char *cur = begin;
  while( cur < end )
    {
    size_t remaining = std::min( maxpdu , (size_t)(end - cur) );
    std::string sub( cur, remaining );

    PresentationDataValue thePDV;
    std::string UIDString;
    thePDV.SetPresentationContextID( prescontid );

    thePDV.SetBlob( sub );
    cur += remaining;

    if( cur < end )
      thePDV.SetMessageHeader( 0 );
    else
      {
      assert( cur == end );
      thePDV.SetMessageHeader( 2 );
      }
    thePDVs.push_back(thePDV);
    }
}

  return thePDVs;

}

//private hack
std::vector<PresentationDataValue> CStoreRQ::ConstructPDV(
const ULConnection &inConnection, const BaseRootQuery* inRootQuery)
{
  std::vector<PresentationDataValue> thePDVs;
  (void)inRootQuery;
  (void)inConnection;
  assert( 0 && "TODO" );
  return thePDVs;
}

//private hack
std::vector<PresentationDataValue>  CStoreRSP::ConstructPDV(const ULConnection &, const BaseRootQuery* inRootQuery)
{
  std::vector<PresentationDataValue> thePDVs;
  (void)inRootQuery;
  assert( 0 && "TODO" );
  return thePDVs;
}

std::vector<PresentationDataValue> CStoreRSP::ConstructPDV(const DataSet* inDataSet, const BasePDU* inPDU){
  std::vector<PresentationDataValue> thePDVs;

///should be passed the received dataset, ie, the cstorerq, so that
///the cstorersp contains the appropriate SOP instance UIDs.
  CommandDataSet ds;

  uint32_t theMessageID = 0;
  Attribute<0x0,0x0110> at3 = { 0 };
  at3.SetFromDataSet( *inDataSet );
  theMessageID = at3.GetValue();

  const DataElement &de1 = inDataSet->GetDataElement( Tag( 0x0000,0x0002 ) );
  const DataElement &de2 = inDataSet->GetDataElement( Tag( 0x0000,0x1000 ) );
  //pass back the instance UIDs in the response
  ds.Insert(de1);
  ds.Insert(de2);

  //code is from the presentationdatavalue::myinit2
    {
    // Command Field
    Attribute<0x0,0x100> at = { 32769 };
    ds.Insert( at.GetAsDataElement() );
    }
    {
    // Message ID Being Responded To
    Attribute<0x0,0x120> at = { 1 };
    at.SetValue( (unsigned short)theMessageID );
    ds.Insert( at.GetAsDataElement() );
    }
    {
    Attribute<0x0,0x800> at = { 257 };
    ds.Insert( at.GetAsDataElement() );
    }
    {
    Attribute<0x0,0x900> at = { 0 };
    ds.Insert( at.GetAsDataElement() );
    }
    {
    Attribute<0x0,0x0> at = { 0 };
    unsigned int glen = ds.GetLength<ImplicitDataElement>();
    assert( (glen % 2) == 0 );
    at.SetValue( glen );
    ds.Insert( at.GetAsDataElement() );
    }

  PresentationDataValue pdv;

  // FIXME
  // how do we retrieve the actual PresID from the AAssociate?
  const PDataTFPDU* theDataPDU = dynamic_cast<const PDataTFPDU*>(inPDU);
  assert (theDataPDU);
  uint8_t thePDVValue;
  PresentationDataValue const &input_pdv = theDataPDU->GetPresentationDataValue(0);
  thePDVValue = input_pdv.GetPresentationContextID();

  pdv.SetPresentationContextID( thePDVValue );

  pdv.SetDataSet(ds);

  pdv.SetMessageHeader(3);
  thePDVs.push_back(pdv);

  return thePDVs;
}

}//namespace network
}//namespace gdcm
