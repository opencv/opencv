/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPresentationDataValue.h"
#include "gdcmSwapper.h"
#include "gdcmFile.h"
#include "gdcmAttribute.h"
#include "gdcmCommandDataSet.h"
#include "gdcmPrinter.h"

#include <limits>

namespace gdcm
{
namespace network
{

PresentationDataValue::PresentationDataValue()
{
  MessageHeader = 0;
  PresentationContextID = 0; //MUST BE SET BY THE CALLER!

  // postcondition
  assert(Size() < std::numeric_limits<uint32_t>::max());
  ItemLength = (uint32_t)Size() - 4;
  assert (ItemLength + 4 == Size() );
}

std::istream &PresentationDataValue::Read(std::istream &is)
{
  uint32_t itemlength = ItemLength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;
  is.read( (char*)&PresentationContextID, sizeof(PresentationContextID) );

  uint8_t mh;
  is.read( (char*)&mh, 1 );
  MessageHeader = mh;
  if ( MessageHeader > 3 )
    {
    gdcmDebugMacro( "Bizarre MessageHeader: " << MessageHeader );
    }

  assert( ItemLength > 2 );
  uint32_t vl = ItemLength - 2;
  Blob.resize( vl );
  is.read( &Blob[0], vl );

  assert (ItemLength + 4 == Size() );
  return is;
}

std::istream &PresentationDataValue::ReadInto(std::istream &is, std::ostream &os)
{
  uint32_t itemlength = ItemLength;
  is.read( (char*)&itemlength, sizeof(ItemLength) );
  SwapperDoOp::SwapArray(&itemlength,1);
  ItemLength = itemlength;
  is.read( (char*)&PresentationContextID, sizeof(PresentationContextID) );

  uint8_t mh;
  is.read( (char*)&mh, 1 );
  MessageHeader = mh;
  if ( MessageHeader > 3 )
    {
    gdcmDebugMacro( "Bizarre MessageHeader: " << MessageHeader );
    }

  assert( ItemLength > 2 );
  uint32_t vl = ItemLength - 2;
  Blob.resize( vl );
  is.read( &Blob[0], vl );
  os.write( &Blob[0], vl );

  assert (ItemLength + 4 == Size() );
  return is;
}

const std::ostream &PresentationDataValue::Write(std::ostream &os) const
{
  uint32_t copy = ItemLength;
  SwapperDoOp::SwapArray(&copy,1);
  os.write( (char*)&copy, sizeof(ItemLength) );
  assert( os.good() );
  os.write( (char*)&PresentationContextID, sizeof(PresentationContextID) );
  assert( os.good() );

  assert( MessageHeader <= 3 );
  uint8_t t = MessageHeader;
  os.write( (char*)&t, 1 );
  assert( os.good() );

  os.write( Blob.c_str(), Blob.size() );

  assert( Blob.size() == ItemLength - 2 );
  assert (ItemLength + 4 == Size() );

  return os;
}

size_t PresentationDataValue::Size() const
{
  size_t ret = 0;
  ret += sizeof(ItemLength);
  ret += sizeof(PresentationContextID);
  ret += sizeof(MessageHeader); // MESSAGE CONTROL HEADER ENCODING
  ret += Blob.size();

  return ret;
}

void PresentationDataValue::SetBlob(const std::string & partialblob)
{
  assert( !partialblob.empty() );
  Blob = partialblob;
  assert(Size() < std::numeric_limits<uint32_t>::max());
  ItemLength = (uint32_t)Size() - 4;
  assert (ItemLength + 4 == Size() );
}

void PresentationDataValue::SetDataSet(const DataSet & ds)
{
  std::stringstream ss;
  //!!FIXME-- have to make sure that the transfer syntax is known and accounted for!
  ds.Write<ImplicitDataElement,SwapperNoOp>( ss );
  SetBlob( ss.str() );
}

const std::string &PresentationDataValue::GetBlob() const
{
  return Blob;
}

//should only be one data set per chunk of pdvs.  So, only return one; the
//loop that gets the results from the scp will be clever and only enter this function
//when the pdu has its 'last bit' set to true (ie, when all the pdvs can be sent in at once,
//but the are all part of the same data set)
DataSet PresentationDataValue::ConcatenatePDVBlobs(const std::vector<PresentationDataValue>& inPDVs)
{
  //size_t s = inPDVs.size();

  std::string theEntireBuffer;//could do it as streams.  but apparently, std isn't letting me
  std::vector<PresentationDataValue>::const_iterator itor;
  for (itor = inPDVs.begin(); itor < inPDVs.end(); itor++){
    const std::string & theBlobString = itor->GetBlob();
    theEntireBuffer.insert(theEntireBuffer.end(), theBlobString.begin(), theBlobString.end());
  }

  DataSet outDataSet;

  std::stringstream ss;
  ss.str( theEntireBuffer );

#if 0
  char fn[512];
  static int i = 0;
  sprintf( fn, "/tmp/debugimp%d", i++ );
  std::ofstream d( fn, std::ios::binary );
  d.write( theEntireBuffer.c_str(), theEntireBuffer.size() );
  d.close();
#endif

  outDataSet.Read<ImplicitDataElement,SwapperNoOp>( ss );
  //outDataSet.Read<ExplicitDataElement,SwapperNoOp>( ss );


  return outDataSet;
}

DataSet PresentationDataValue::ConcatenatePDVBlobsAsExplicit(const std::vector<PresentationDataValue>& inPDVs)
{
#if 0
  std::stringstream ss;
  std::vector<PresentationDataValue>::const_iterator itor;
  for (itor = inPDVs.begin(); itor < inPDVs.end(); itor++)
    {
    const std::string & theBlobString = itor->GetBlob();
    ss.write( &theBlobString[0], theBlobString.size() );
    }
#else
  std::string theEntireBuffer;//could do it as streams.  but apparently, std isn't letting me
  std::vector<PresentationDataValue>::const_iterator itor;
  for (itor = inPDVs.begin(); itor < inPDVs.end(); itor++){
    const std::string & theBlobString = itor->GetBlob();
    theEntireBuffer.insert(theEntireBuffer.end(), theBlobString.begin(), theBlobString.end());
  }


  std::stringstream ss;
  ss.str( theEntireBuffer );

#endif

#if 0
  char fn[512];
  static int i = 0;
  sprintf( fn, "/tmp/debugex%d", i++ );
  std::ofstream d( fn, std::ios::binary );
  d.write( theEntireBuffer.c_str(), theEntireBuffer.size() );
  d.close();
#endif

  DataSet outDataSet;
  //outDataSet.Read<ExplicitDataElement,SwapperNoOp>( ss );
  VL length = (uint32_t)theEntireBuffer.size();
  //gdcm::Trace::DebugOn();
  outDataSet.ReadWithLength<ExplicitDataElement,SwapperNoOp>( ss, length );

  return outDataSet;
}

void PresentationDataValue::Print(std::ostream &os) const
{
  os << "ItemLength: " << ItemLength << std::endl;
  os << "PresentationContextID: " << (int)PresentationContextID << std::endl;
  os << "MessageHeader: " << (int)MessageHeader << std::endl;
  std::vector<PresentationDataValue> thePDVs;
  thePDVs.push_back(*this);
  DataSet ds = ConcatenatePDVBlobs(thePDVs);
  Printer thePrinter;
  thePrinter.PrintDataSet(ds, os);
}

void PresentationDataValue::SetCommand(bool inCommand)
{
  const uint8_t flipped = ~1;
  if (inCommand)
    {
    MessageHeader |= 1;
    }
  else
    {
    MessageHeader &= flipped;
    }
}

void PresentationDataValue::SetLastFragment(bool inLast)
{
  const uint8_t flipped = ~2;
  if (inLast)
    {
    MessageHeader |= 2;
    }
  else
    {
    MessageHeader &= flipped;//set the second field to zero
    }
}

bool PresentationDataValue::GetIsCommand() const
{
  return (MessageHeader & 1) == 1;
}

bool PresentationDataValue::GetIsLastFragment() const
{
  return (MessageHeader & 2) == 2;
}

} // end namespace network
} // end namespace gdcm
