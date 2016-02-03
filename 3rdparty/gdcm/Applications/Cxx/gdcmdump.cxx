/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*
 * Simple command line tool to dump the layout/values of a DICOM file
 * This is largely inspired by other tools available from other toolkit, namely:
 * - dcdump (dicom3tools)
 * - dcmdump (dcmtk)
 * - dcmInfo (SIEMENS)
 * - PrintFile (GDCM 1.x)
 *
 * For now all layout are harcoded (see --color/--xml-dict for instance)
 *
 * gdcmdump has some feature not described in the DICOM standard:
 *   --csa : to print CSA information (dcmInfo.exe compatible)
 *   --pdb : to print PDB information (GEMS private info)
 *   --elscint : to print ELSCINT information (ELSCINT private info)
 *
 *
 * TODO: it would be nice to have custom printing, namely printing as HTML/XML
 *       it would be nice to have runtime dict (instead of compile time)
 */

#include "gdcmReader.h"
#include "gdcmVersion.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmDataSet.h"
#include "gdcmPrivateTag.h"
#include "gdcmPrinter.h"
#include "gdcmDumper.h"
#include "gdcmDictPrinter.h"
#include "gdcmValidate.h"
#include "gdcmWriter.h"
#include "gdcmSystem.h"
#include "gdcmDirectory.h"
#include "gdcmCSAHeader.h"
#include "gdcmPDBHeader.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmASN1.h"
#include "gdcmAttribute.h"

#include <string>
#include <iostream>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

static int color = 0;

static int ignoreerrors = 0;

namespace cleanup
{
// {"1.3.46.670589.11.0.0.12.2"       ,"Philips Private MR Series Data Storage"},
enum {
  TYPE_FLOAT  = 0, // float
  TYPE_INT32  = 1, // int32
  TYPE_STRING = 2, // 80 bytes string (+1)
  TYPE_UINT32 = 4  // uint32
};

template <typename T>
static void printvaluet(std::istream & is, uint32_t numels)
{
  T buffer;
  for( uint32_t i = 0; i < numels; ++i )
    {
    if( i ) std::cout << "\\";
    is.read( (char*)&buffer, sizeof(T) );
    std::cout << buffer;
    }
}

static void printvalue(std::istream &is, uint32_t type, uint32_t numels, uint32_t pos)
{
  assert( numels > 0 );
  std::streampos start = is.tellg();
  is.seekg( pos );
  std::cout << "[";
  typedef char (string81)[81]; // 80'th byte == 0
  assert( sizeof( string81 ) == 81 );
  switch( type )
    {
  case TYPE_FLOAT:
    printvaluet<float>(is, numels);
    break;
  case TYPE_INT32:
    printvaluet<int32_t>(is, numels);
    break;
  case TYPE_STRING:
    printvaluet<string81>(is, numels);
    break;
  case TYPE_UINT32:
    printvaluet<uint32_t>(is, numels);
    break;
  default:
    assert( 0 );
    }
  std::cout << "]";
  std::cout << " # " << numels;
  is.seekg( start );
}

struct PDFElement
{
  const char *getname() const { return name; }
  uint32_t gettype() const { return getvalue(0); }
  uint32_t getnumelems() const { return getvalue(1); }
  uint32_t getdummy() const { return getvalue(2); }
  uint32_t getoffset() const { return getvalue(3); }
private:
  char name[50];
  // type , numel and offset needs to be read starting from the end
  // the data in between name and those value can contains garbage stuff
  uint32_t getvalue(int n) const {
    uint32_t val = 0;
    memcpy( (char*)&val, name + 50 - 16 + n * 4, sizeof( val ) );
    return val;
  }
};

static void printbinary(std::istream &is, PDFElement const & pdfel )
{
  const char *bufferref = pdfel.getname();
  std::cout << "  " << bufferref << " ";
  uint32_t type = pdfel.gettype();
  uint32_t numels = pdfel.getnumelems();
  uint32_t dummy = pdfel.getdummy();
  assert( dummy == 0 ); (void)dummy;
  uint32_t offset = pdfel.getoffset();
  uint32_t pos = (uint32_t)(offset + is.tellg() - 4);
  printvalue(is, type, numels, pos);
}

static void ProcessSDSData( std::istream & is )
{
  // havent been able to figure out what was the begin meant for
  is.seekg( 0x20 - 8 );
  uint32_t version = 0;
  is.read( (char*)&version, sizeof(version) );
  assert( version == 8 );
  uint32_t numel = 0;
  is.read( (char*)&numel, sizeof(numel) );
  for( uint32_t el = 0; el < numel; ++el )
    {
    PDFElement pdfel;
    assert( sizeof(pdfel) == 50 );
    is.read( (char*)&pdfel, 50 );
    if( *pdfel.getname() )
      {
      printbinary( is, pdfel );
      std::cout << std::endl;
      }
    }

}
// PMS MR Series Data Storage
static int DumpPMS_MRSDS(const gdcm::DataSet & ds)
{
  const gdcm::PrivateTag tdata(0x2005,0x32,"Philips MR Imaging DD 002");
  if( !ds.FindDataElement( tdata ) ) return 1;
  const gdcm::DataElement &data = ds.GetDataElement( tdata );
  gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = data.GetValueAsSQ();
  if( !sqi ) return 1;
  std::cout << "PMS Dumping info from tag " << tdata << std::endl;
  gdcm::SequenceOfItems::ConstIterator it = sqi->Begin();
  for( ; it != sqi->End(); ++it )
    {
    const gdcm::Item & item = *it;
    const gdcm::DataSet & nestedds = item.GetNestedDataSet();
    const gdcm::PrivateTag tprotocoldataname(0x2005,0x37,"Philips MR Imaging DD 002");
    const gdcm::DataElement & protocoldataname = nestedds.GetDataElement( tprotocoldataname );
    const gdcm::ByteValue *bv1 = protocoldataname.GetByteValue();
    const gdcm::PrivateTag tprotocoldatatype(0x2005,0x39,"Philips MR Imaging DD 002");
    const gdcm::DataElement & protocoldatatype = nestedds.GetDataElement( tprotocoldatatype );
    const gdcm::ByteValue *bv2 = protocoldatatype.GetByteValue();
    const gdcm::PrivateTag tprotocoldatablock(0x2005,0x44,"Philips MR Imaging DD 002");
    const gdcm::DataElement & protocoldatablock = nestedds.GetDataElement( tprotocoldatablock );
    const gdcm::ByteValue *bv3 = protocoldatablock.GetByteValue();
    const gdcm::PrivateTag tprotocoldatabool(0x2005,0x47,"Philips MR Imaging DD 002");
    const gdcm::DataElement & protocoldatabool = nestedds.GetDataElement( tprotocoldatabool );
    const gdcm::ByteValue *bv4 = protocoldatabool.GetByteValue();
    std::string s1;
    if( bv1 )
      {
      s1 = std::string( bv1->GetPointer(), bv1->GetLength() );
      }
    std::string s2;
    if( bv2 )
      {
      s2 = std::string( bv2->GetPointer(), bv2->GetLength() );
      }
    std::string s3;
    if( bv3 )
      {
      s3 = std::string( bv3->GetPointer(), bv3->GetLength() );
      }
    std::string s4;
    if( bv4 )
      {
      s4 = std::string( bv4->GetPointer(), bv4->GetLength() );
      }
    std::istringstream is( s3 );
    std::cout << "PMS/Item name: [" << s1 << "/" << s2 << "/" << s4 << "]" << std::endl;
    ProcessSDSData( is );
    }
  return 0;
}

static int DumpTOSHIBA_MEC_CT3(const gdcm::DataSet & ds)
{
  const gdcm::PrivateTag tdata(0x7005,0x10,"TOSHIBA_MEC_CT3");
  if( !ds.FindDataElement( tdata ) ) return 1;
  const gdcm::DataElement &data = ds.GetDataElement( tdata );

  const gdcm::ByteValue *bv = data.GetByteValue();
  if( !bv ) return 1;

  const int offset = 24;
  if( bv->GetLength() < offset )
    {
    std::cerr << "Not enough header" << std::endl;
    return 1;
    }
  std::istringstream is0;
  const std::string str0 = std::string( bv->GetPointer(), offset );
  is0.str( str0 );
  gdcm::ImplicitDataElement ide0;
  ide0.Read<gdcm::SwapperNoOp>(is0);
  gdcm::ImplicitDataElement ide1;
  ide1.Read<gdcm::SwapperNoOp>(is0);

  gdcm::Attribute<0x0,0x0> at0;
  at0.SetFromDataElement( ide0 );
  if( at0.GetValue() != 12 )
    {
    std::cerr << "Bogus header value #0" << std::endl;
    return 1;
    }
  gdcm::Attribute<0x0,0x1> at1;
  at1.SetFromDataElement( ide1 );

  const unsigned int dlen = bv->GetLength() - offset;
  if( at1.GetValue() != dlen )
    {
    std::cerr << "Bogus header value #1" << std::endl;
    return 1;
    }
  std::istringstream is1;
  const std::string str1 = std::string( bv->GetPointer() + offset, bv->GetLength() - offset);
  is1.str( str1 );

  gdcm::Reader r;
  r.SetStream( is1 );
  if( !r.Read() )
    {
    std::cerr << "Could not read CT Private Data 2" << std::endl;
    return 1;
    }

  gdcm::Printer printer;
  printer.SetFile ( r.GetFile() );
  printer.SetColor( color != 0 );
  printer.Print( std::cout );

  return 0;
}

// VEPRO
/*
[VIMDATA2]
PrivateCreator = VEPRO VIM 5.0 DATA
Group = 0x0055
Element = 0x0020
Data.ID     = C|0|3
Data.Version     = C|3|3
Data.UserName    = C|6|32
Data.UserAdress1  = C|38|32
Data.UserAdress2  = C|70|32
Data.UserAdress3  = C|102|32
Data.UserAdress4  = C|134|32
Data.UserAdress5  = C|166|32
Data.RecDate        = C|198|8
Data.RecTime        = C|206|6
Data.RecPlace        = C|212|64
Data.RecSource        = C|276|64
Data.DF1          = C|340|64
Data.DF2          = C|404|64
Data.DF3          = C|468|64
Data.DF4          = C|532|64
Data.DF5          = C|596|64
Data.DF6          = C|660|64
Data.DF7          = C|724|64
Data.DF8          = C|788|64
Data.DF9          = C|852|64
Data.DF10          = C|916|64
Data.DF11          = C|980|64
Data.DF12          = C|1044|64
Data.DF13          = C|1108|64
Data.DF14          = C|1172|64
Data.DF15          = C|1236|64
Data.DF16          = C|1300|64
Data.DF17          = C|1364|64
Data.DF18          = C|1428|64
Data.DF19          = C|1492|64
Data.DF20          = C|1556|64
Data.StudyUID          = C|1642|64
Data.SeriesUID          = C|1706|64
Data.Modality          = C|1770|16
*/
// TYPE[C/I] / OFFSET / LENGTH (in bytes)
struct Data2
{
 char ID[3]; // Data.ID     = C|0|3
 char Version[3]; // Data.Version     = C|3|3
 char UserName[32]; // Data.UserName    = C|6|32
 char UserAdress1[32]; // Data.UserAdress1  = C|38|32
 char UserAdress2[32]; // Data.UserAdress2  = C|70|32
 char UserAdress3[32]; // Data.UserAdress3  = C|102|32
 char UserAdress4[32]; // Data.UserAdress4  = C|134|32
 char UserAdress5[32]; // Data.UserAdress5  = C|166|32
 char RecDate[8];  // Data.RecDate        = C|198|8
 char RecTime[6];  // Data.RecTime        = C|206|6
 char RecPlace[64]; // Data.RecPlace        = C|212|64
 char RecSource[64];// Data.RecSource        = C|276|64
 char DF1[64];       // Data.DF1          = C|340|64
 char DF2[64];       // Data.DF2          = C|404|64
 char DF3[64];       // Data.DF3          = C|468|64
 char DF4[64];       // Data.DF4          = C|532|64
 char DF5[64];       // Data.DF5          = C|596|64
 char DF6[64];       // Data.DF6          = C|660|64
 char DF7[64];       // Data.DF7          = C|724|64
 char DF8[64];       // Data.DF8          = C|788|64
 char DF9[64];       // Data.DF9          = C|852|64
 char DF10[64];     // Data.DF10          = C|916|64
 char DF11[64];     // Data.DF11          = C|980|64
 char DF12[64];     // Data.DF12          = C|1044|64
 char DF13[64];     // Data.DF13          = C|1108|64
 char DF14[64];     // Data.DF14          = C|1172|64
 char DF15[64];     // Data.DF15          = C|1236|64
 char DF16[64];     // Data.DF16          = C|1300|64
 char DF17[64];     // Data.DF17          = C|1364|64
 char DF18[64];     // Data.DF18          = C|1428|64
 char DF19[64];     // Data.DF19          = C|1492|64
 char DF20[64];     // Data.DF20          = C|1556|64
 char Padding[22]; // ?????
 char StudyUID[64];  // Data.StudyUID          = C|1642|64
 char SeriesUID[64];  // Data.SeriesUID          = C|1706|64
 char Modality[16];  // Data.Modality          = C|1770|16

 void Print( std::ostream &os )
   {
   os << "  ID: "          << std::string(ID,3) << "\n";
   os << "  Version: "     << std::string(Version,3) << "\n";
   os << "  UserName: "    << std::string(UserName,32) << "\n";
   os << "  UserAdress1: " << std::string(UserAdress1,32) << "\n";
   os << "  UserAdress2: " << std::string(UserAdress2,32) << "\n";
   os << "  UserAdress3: " << std::string(UserAdress3,32) << "\n";
   os << "  UserAdress4: " << std::string(UserAdress4,32) << "\n";
   os << "  UserAdress5: " << std::string(UserAdress5,32) << "\n";
   os << "  RecDate: "     << std::string(RecDate,8) << "\n";
   os << "  RecTime: "     << std::string(RecTime,64) << "\n";
   os << "  RecPlace: "    << std::string(RecPlace,64) << "\n";
   os << "  RecSource: "   << std::string(RecSource,64) << "\n";
   os << "  DF1: "         << std::string(DF1,64) << "\n";
   os << "  DF2: "         << std::string(DF2,64) << "\n";
   os << "  DF3: "         << std::string(DF3,64) << "\n";
   os << "  DF4: "         << std::string(DF4,64) << "\n";
   os << "  DF5: "         << std::string(DF5,64) << "\n";
   os << "  DF6: "         << std::string(DF6,64) << "\n";
   os << "  DF7: "         << std::string(DF7,64) << "\n";
   os << "  DF8: "         << std::string(DF8,64) << "\n";
   os << "  DF9: "         << std::string(DF9,64) << "\n";
   os << "  DF10: "        << std::string(DF10,64) << "\n";
   os << "  DF11: "        << std::string(DF11,64) << "\n";
   os << "  DF12: "        << std::string(DF12,64) << "\n";
   os << "  DF13: "        << std::string(DF13,64) << "\n";
   os << "  DF14: "        << std::string(DF14,64) << "\n";
   os << "  DF15: "        << std::string(DF15,64) << "\n";
   os << "  DF16: "        << std::string(DF16,64) << "\n";
   os << "  DF17: "        << std::string(DF17,64) << "\n";
   os << "  DF18: "        << std::string(DF18,64) << "\n";
   os << "  DF19: "        << std::string(DF19,64) << "\n";
   os << "  DF20: "        << std::string(DF20,64) << "\n";
   //os << "  Padding: " <<   std::string(Padding,22) << "\n";
   os << "  StudyUID: "    << std::string(StudyUID,64) << "\n";
   os << "  SeriesUID: "   << std::string(SeriesUID,64) << "\n";
   os << "  Modality: "    << std::string(Modality,16) << "\n";
   }
};

static bool ProcessData( const char *buf, size_t len )
{
  Data2 data2;
  const size_t s = sizeof(data2);
  assert( len >= s); (void)len;
  // VIMDATA2 is generally 2048 bytes, while s = 1786
  // the end is filled with \0 bytes
  memcpy(&data2, buf, s);

  data2.Print( std::cout );
  return true;
}

static int DumpVEPRO(const gdcm::DataSet & ds)
{
  // 01f7,1026
  const gdcm::ByteValue *bv2 = NULL;
  const gdcm::PrivateTag tdata1(0x55,0x0020,"VEPRO VIF 3.0 DATA");
  const gdcm::PrivateTag tdata2(0x55,0x0020,"VEPRO VIM 5.0 DATA");
  // Prefer VIF over VIM ?
  if( ds.FindDataElement( tdata1 ) )
    {
    std::cout  << "VIF DATA: " << tdata1 << "\n";
    const gdcm::DataElement &data = ds.GetDataElement( tdata1 );
    bv2 = data.GetByteValue();
    }
  else if( ds.FindDataElement( tdata2 ) )
    {
    std::cout  << "VIMDATA2: " << tdata2 << "\n";
    const gdcm::DataElement &data = ds.GetDataElement( tdata2 );
    bv2 = data.GetByteValue();
    }

  if( bv2 )
    {
    ProcessData( bv2->GetPointer(), bv2->GetLength() );
    return 0;
    }

  return 1;
}

// ELSCINT1
static bool readastring(std::string &out, const char *input )
{
  out.clear();
  while( *input )
    {
    out.push_back( *input++ );
    }
  return true;
}

struct el
{
  std::string name;
  uint32_t pad;
  std::vector<std::string> values;
  size_t Size() const
    {
    size_t s = 0;
    s += name.size() + 1;
    s += sizeof(pad);
    for( std::vector<std::string>::const_iterator it = values.begin(); it != values.end(); ++it )
      s += it->size() + 1;
    return s;
    }
  void ReadFromString( const char * input )
    {
    readastring( name, input );
    const char *p = input + 1+  name.size();
    memcpy( &pad, p, sizeof( pad ) );
    //assert( pad == 1 || pad == 2 || pad == 3 || pad == 6 );
    values.resize( pad );
    const char *pp = p + sizeof(uint32_t);
    for( uint32_t pidx = 0; pidx < pad; ++pidx )
      {
      readastring( values[pidx], pp );
      pp = pp + values[pidx].size() + 1;
      }
    }
  void Print() const
    {
    //std::cout << "  " << name << " : " << pad << " : (";
    std::cout << "  " << name << " [";
      {
      std::vector<std::string>::const_iterator it = values.begin();
      std::cout << *it++;
      for(; it != values.end(); ++it )
        {
        std::cout << "\\";
        std::cout << *it;
        }
      }
    std::cout << "]" << std::endl;
    }
};


struct info
{
  size_t Read(const char *in )
    {
    const char *m = in;
    uint32_t h;
    memcpy( &h, in, sizeof(h) );
    in += sizeof(h);
    std::string dummy;
    readastring( dummy, in );
    in += dummy.size();
    in += 1;
    if( h == 432154 ) // 0x6981a
      {
      // Single item
      uint32_t nels;
      memcpy( &nels, in, sizeof(nels) );
      in += sizeof(nels);
      //std::cout << "  ELSCINT1/Item name: " << dummy << " : " << nels << std::endl;
      std::cout << "ELSCINT1/Item name: [" << dummy << "]" << std::endl;
      for( uint32_t i = 0; i < nels; ++i )
        {
        el e;
        e.ReadFromString( in );
        e.Print();
        in += e.Size();
        }
      }
    else if( h == 2341 ) // 0x925
      {
      // Multiple Item(s)
      uint32_t d;
      memcpy( &d, in, sizeof(d) );
      in += sizeof(d);
      //std::cout << "  Info Name: " << dummy << " : " << d << std::endl;
      std::cout << "ELSCINT1/Item name: " << dummy << std::endl;
      for( uint32_t dix = 0; dix < d; ++dix )
        {
        uint32_t fixme;
        memcpy( &fixme, in, sizeof(fixme) );
        in += 4; //
        //std::cout << "  number of  Subitems " << fixme << std::endl;
        uint32_t nels = fixme;
        if( nels )
          std::cout << " SubItems #" << dix << std::endl;
        else
          std::cout << " No SubItems (Empty)" << std::endl;
        for( uint32_t i = 0; i < nels; ++i )
          {
          el e;
          e.ReadFromString( in );
          e.Print();
          in += e.Size();
          }
        }
      // postcondition
      uint32_t fixme;
      memcpy( &fixme, in, sizeof(fixme) );
      assert( fixme == 0x0006981A );
      }
    else
      {
      assert( 0 );
      }
    return in - m;
    }
};

static int DumpEl2_new(const gdcm::DataSet & ds)
{
  // 01f7,1026
  const gdcm::PrivateTag t01f7_26(0x01f7,0x1026,"ELSCINT1");
  if( !ds.FindDataElement( t01f7_26 ) ) return 1;
  const gdcm::DataElement& de01f7_26 = ds.GetDataElement( t01f7_26 );
  if ( de01f7_26.IsEmpty() ) return 1;
  const gdcm::ByteValue * bv = de01f7_26.GetByteValue();

  const char *begin = bv->GetPointer();
  uint32_t val0[3];
  memcpy( &val0, begin, sizeof( val0 ) );
  assert( val0[0] == 0xF22D );
  begin += sizeof( val0 );

  // 1A 98 06 00 -> start element
  // Next is a string (can be NULL)
  // then number of (nested) elements

  std::cout << "ELSCINT1 Dumping info from tag " << t01f7_26 << std::endl;
  info i;
  size_t o;
  assert( val0[1] == 0x1 );
  for( uint32_t idx = 0; idx < val0[2]; ++idx )
    {
    o = i.Read( begin );
    std::cout << std::endl;
    begin += o;
    }

  return 0;
}

} // end namespace cleanup

template <typename TPrinter>
static int DoOperation(const std::string & filename)
{
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  bool success = reader.Read();
  if( !success && !ignoreerrors )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  TPrinter printer;
  printer.SetFile ( reader.GetFile() );
  printer.SetColor( color != 0);
  printer.Print( std::cout );

  // Only return success when file read succeeded not depending whether or not we printed it
  return success ? 0 : 1;
}

static int PrintASN1(const std::string & filename, bool verbose)
{
  (void)verbose;
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }
  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();
  gdcm::Tag tencryptedattributessequence(0x0400,0x0500);
  if( !ds.FindDataElement( tencryptedattributessequence ) )
    {
    return 1;
    }
  const gdcm::DataElement &encryptedattributessequence = ds.GetDataElement( tencryptedattributessequence );
  //const gdcm::SequenceOfItems * sqi = encryptedattributessequence.GetSequenceOfItems();
  gdcm::SmartPointer<gdcm::SequenceOfItems> sqi = encryptedattributessequence.GetValueAsSQ();
  if( !sqi->GetNumberOfItems() )
    {
    return 1;
    }
  const gdcm::Item &item1 = sqi->GetItem(1);
  const gdcm::DataSet &subds = item1.GetNestedDataSet();

  gdcm::Tag tencryptedcontent(0x0400,0x0520);
  if( !subds.FindDataElement( tencryptedcontent) )
    {
    return 1;
    }
  const gdcm::DataElement &encryptedcontent = subds.GetDataElement( tencryptedcontent );
  const gdcm::ByteValue *bv = encryptedcontent.GetByteValue();

  bool b = gdcm::ASN1::ParseDump( bv->GetPointer(), bv->GetLength() );
  if( !b ) return 1;
  return 0;
}

static int PrintELSCINT(const std::string & filename, bool verbose)
{
  (void)verbose;
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();
  int ret = cleanup::DumpEl2_new( ds );

  return ret;
}

static int PrintVEPRO(const std::string & filename, bool verbose)
{
  (void)verbose;
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();
  int ret = cleanup::DumpVEPRO( ds );

  return ret;
}

static int PrintSDS(const std::string & filename, bool verbose)
{
  (void)verbose;
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();
  int ret = cleanup::DumpPMS_MRSDS( ds );

  return ret;
}

static int PrintCT3(const std::string & filename, bool verbose)
{
  (void)verbose;
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();
  int ret = cleanup::DumpTOSHIBA_MEC_CT3( ds );

  return ret;
}

static int PrintPDB(const std::string & filename, bool verbose)
{
  (void)verbose;
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  gdcm::PDBHeader pdb;
  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();

  const gdcm::PrivateTag &t1 = pdb.GetPDBInfoTag();

  bool found = false;
  int ret = 0;
  if( ds.FindDataElement( t1 ) )
    {
    pdb.LoadFromDataElement( ds.GetDataElement( t1 ) );
    pdb.Print( std::cout );
    found = true;
    }
  if( !found )
    {
    std::cout << "no pdb tag found" << std::endl;
    ret = 1;
    }

  return ret;
}

static int PrintCSA(const std::string & filename)
{
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  gdcm::CSAHeader csa;
  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();

  const gdcm::PrivateTag &t1 = csa.GetCSAImageHeaderInfoTag();
  const gdcm::PrivateTag &t2 = csa.GetCSASeriesHeaderInfoTag();
  const gdcm::PrivateTag &t3 = csa.GetCSADataInfo();

  bool found = false;
  int ret = 0;
  if( ds.FindDataElement( t1 ) )
    {
    csa.LoadFromDataElement( ds.GetDataElement( t1 ) );
    csa.Print( std::cout );
    found = true;
    if( csa.GetFormat() == gdcm::CSAHeader::ZEROED_OUT )
      {
      std::cout << "CSA Header has been zero-out (contains only 0)" << std::endl;
      ret = 1;
      }
    else if( csa.GetFormat() == gdcm::CSAHeader::DATASET_FORMAT )
      {
      gdcm::Printer p;
      gdcm::File f;
      f.SetDataSet( csa.GetDataSet() );
      p.SetFile( f );
      p.Print( std::cout );
      }
    }
  if( ds.FindDataElement( t2 ) )
    {
    csa.LoadFromDataElement( ds.GetDataElement( t2 ) );
    csa.Print( std::cout );
    found = true;
    if( csa.GetFormat() == gdcm::CSAHeader::ZEROED_OUT )
      {
      std::cout << "CSA Header has been zero-out (contains only 0)" << std::endl;
      ret = 1;
      }
    else if( csa.GetFormat() == gdcm::CSAHeader::DATASET_FORMAT )
      {
      gdcm::Printer p;
      gdcm::File f;
      f.SetDataSet( csa.GetDataSet() );
      p.SetFile( f );
      p.Print( std::cout );
      }
    }
  if( ds.FindDataElement( t3 ) )
    {
    csa.LoadFromDataElement( ds.GetDataElement( t3 ) );
    csa.Print( std::cout );
    found = true;
    if( csa.GetFormat() == gdcm::CSAHeader::ZEROED_OUT )
      {
      std::cout << "CSA Header has been zero-out (contains only 0)" << std::endl;
      ret = 1;
      }
    else if( csa.GetFormat() == gdcm::CSAHeader::INTERFILE )
      {
      const char *interfile = csa.GetInterfile();
      if( interfile ) std::cout << interfile << std::endl;
      }
    else if( csa.GetFormat() == gdcm::CSAHeader::DATASET_FORMAT )
      {
      gdcm::Printer p;
      gdcm::File f;
      f.SetDataSet( csa.GetDataSet() );
      p.SetFile( f );
      p.Print( std::cout );
      }
    }
  if( !found )
    {
    std::cout << "no csa tag found" << std::endl;
    ret = 1;
    }

  return ret;
}



static void PrintVersion()
{
  std::cout << "gdcmdump: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmdump [OPTION]... FILE..." << std::endl;
  std::cout << "dumps a DICOM file, it will display the structure and values contained in the specified DICOM file\n";
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input     DICOM filename or directory" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -x --xml-dict       generate the XML dict (only private elements for now)." << std::endl;
  std::cout << "  -r --recursive      recursive." << std::endl;
  std::cout << "  -d --dump           dump value (limited use)." << std::endl;
  std::cout << "  -p --print          print value instead of simply dumping (default)." << std::endl;
  std::cout << "  -c --color          print in color." << std::endl;
  std::cout << "  -C --csa            print SIEMENS CSA Header (0029,[12]0,SIEMENS CSA HEADER)." << std::endl;
  std::cout << "  -P --pdb            print GEMS Protocol Data Block (0025,1b,GEMS_SERS_01)." << std::endl;
  std::cout << "     --elscint        print ELSCINT Protocol Information (01f7,26,ELSCINT1)." << std::endl;
  std::cout << "     --vepro          print VEPRO Protocol Information (0055,20,VEPRO VIF 3.0 DATA)." << std::endl;
  std::cout << "                         or VEPRO Protocol Information (0055,20,VEPRO VIM 5.0 DATA)." << std::endl;
  std::cout << "     --sds            print Philips MR Series Data Storage (1.3.46.670589.11.0.0.12.2) Information (2005,32,Philips MR Imaging DD 002)." << std::endl;
  std::cout << "     --ct3            print CT Private Data 2 (7005,10,TOSHIBA_MEC_CT3)." << std::endl;
  std::cout << "  -A --asn1           print encapsulated ASN1 structure >(0400,0520)." << std::endl;
  std::cout << "     --map-uid-names  map UID to names." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose   more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning   print warning info." << std::endl;
  std::cout << "  -D --debug     print debug info." << std::endl;
  std::cout << "  -E --error     print error info." << std::endl;
  std::cout << "  -h --help      print help." << std::endl;
  std::cout << "  -v --version   print version." << std::endl;
  std::cout << "Special Options:" << std::endl;
  std::cout << "  -I --ignore-errors   print even if file is corrupted." << std::endl;
}

int main (int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;

  std::string filename;
  int printdict = 0;
  int dump = 0;
  int print = 0;
  int printcsa = 0;
  int printpdb = 0;
  int printelscint = 0;
  int printvepro = 0;
  int printsds = 0; // MR Series Data Storage
  int printct3 = 0; // TOSHIBA_MEC_CT3
  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;
  int recursive = 0;
  int printasn1 = 0;
  int mapuidnames = 0;
  while (1) {
    //int this_option_optind = optind ? optind : 1;
    int option_index = 0;
/*
   struct option {
              const char *name;
              int has_arg;
              int *flag;
              int val;
          };
*/
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"xml-dict", 0, &printdict, 1},
        {"recursive", 0, &recursive, 1},
        {"print", 0, &print, 1},
        {"dump", 0, &dump, 1},
        {"color", 0, &color, 1},
        {"csa", 0, &printcsa, 1},
        {"pdb", 0, &printpdb, 1},
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {"ignore-errors", 0, &ignoreerrors, 1},
        {"asn1", 0, &printasn1, 1},
        {"map-uid-names", 0, &mapuidnames, 1},
        {"elscint", 0, &printelscint, 1},
        {"vepro", 0, &printvepro, 1},
        {"sds", 0, &printsds, 1},
        {"ct3", 0, &printct3, 1},
        {0, 0, 0, 0} // required
    };
    static const char short_options[] = "i:xrpdcCPAVWDEhvI";
    c = getopt_long (argc, argv, short_options,
      long_options, &option_index);
    if (c == -1)
      {
      break;
      }

    switch (c)
      {
    case 0:
    case '-':
        {
        const char *s = long_options[option_index].name; (void)s;
        //printf ("option %s", s);
        if (optarg)
          {
          if( option_index == 0 ) /* input */
            {
            assert( strcmp(s, "input") == 0 );
            assert( filename.empty() );
            filename = optarg;
            }
          //printf (" with arg %s", optarg);
          }
        //printf ("\n");
        }
      break;

    case 'i':
      //printf ("option i with value '%s'\n", optarg);
      assert( filename.empty() );
      filename = optarg;
      break;

    case 'x':
      //printf ("option d with value '%s'\n", optarg);
      printdict = 1;
      break;

    case 'r':
      recursive = 1;
      break;

    case 'p':
      //printf ("option p with value '%s'\n", optarg);
      print = 1;
      break;

    case 'd':
      dump = 1;
      break;

    case 'c':
      color = 1;
      break;

    case 'C':
      printcsa = 1;
      break;

    case 'A':
      printasn1 = 1;
      break;

    case 'P':
      printpdb = 1;
      break;

    case 'V':
      verbose = 1;
      break;

    case 'W':
      warning = 1;
      break;

    case 'D':
      debug = 1;
      break;

    case 'E':
      error = 1;
      break;

    case 'h':
      help = 1;
      break;

    case 'v':
      version = 1;
      break;

    case 'I':
      ignoreerrors = 1;
      break;

    case '?':
      break;

    default:
      printf ("?? getopt returned character code 0%o ??\n", c);
      }
  }

  if (optind < argc)
    {
    //printf ("non-option ARGV-elements: %d", optind );
    //while (optind < argc)
    //  {
    //  printf ("%s\n", argv[optind++]);
    //  }
    //printf ("\n");
    // Ok there is only one arg, easy, it's the filename:
    int v = argc - optind;
    if( v == 1 )
      {
      filename = argv[optind];
      }
    }

  //
  //gdcm::System::SetArgv0( argv[0] );

  if( version )
    {
    //std::cout << "version" << std::endl;
    PrintVersion();
    return 0;
    }

  if( help )
    {
    //std::cout << "help" << std::endl;
    PrintHelp();
    return 0;
    }

  // check if d or p are passed, only one at a time
  if( print || printdict )
    {
    if ( print && printdict )
      {
      std::cerr << "d or p" << std::endl;
      return 1;
      }
    }
  if( filename.empty() )
    {
    //std::cerr << "Need input file (-i)\n";
    PrintHelp();
    return 1;
    }
  // Debug is a little too verbose
  gdcm::Trace::SetDebug( debug != 0);
  gdcm::Trace::SetWarning( warning != 0);
  gdcm::Trace::SetError( error != 0);
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( verbose != 0);
    gdcm::Trace::SetError( verbose!= 0);
    }

  if( mapuidnames )
    {
    std::cerr << "Not handled for now" << std::endl;
    }

  // else
  int res = 0;
  if( !gdcm::System::FileExists(filename.c_str()) )
    {
    std::cerr << "no such file: " << filename << std::endl;
    return 1;
    }
  else if( gdcm::System::FileIsDirectory( filename.c_str() ) )
    {
    gdcm::Directory d;
    d.Load(filename, recursive!= 0);
    gdcm::Directory::FilenamesType const &filenames = d.GetFilenames();
    for( gdcm::Directory::FilenamesType::const_iterator it = filenames.begin(); it != filenames.end(); ++it )
      {
      if( printdict )
        {
        res += DoOperation<gdcm::DictPrinter>(*it);
        }
      else if( printasn1 )
        {
        res += PrintASN1(*it, verbose!= 0);
        }
      else if( printvepro )
        {
        res += PrintVEPRO(*it, verbose!= 0);
        }
      else if( printsds )
        {
        res += PrintSDS(*it, verbose!= 0);
        }
      else if( printct3 )
        {
        res += PrintCT3(*it, verbose!= 0);
        }
      else if( printelscint )
        {
        res += PrintELSCINT(*it, verbose!= 0);
        }
      else if( printpdb )
        {
        res += PrintPDB(*it, verbose!= 0);
        }
      else if( printcsa )
        {
        res += PrintCSA(*it);
        }
      else if( dump )
        {
        res += DoOperation<gdcm::Dumper>(*it);
        }
      else
        {
        res += DoOperation<gdcm::Printer>(*it);
        }
      if( verbose ) std::cerr << *it << std::endl;
      }
    if( verbose ) std::cerr << "Total: " << filenames.size() << " files were processed" << std::endl;
    }
  else
    {
    assert( gdcm::System::FileExists(filename.c_str()) );
    if( printdict )
      {
      res += DoOperation<gdcm::DictPrinter>(filename);
      }
    else if( printasn1 )
      {
      res += PrintASN1(filename, verbose!= 0);
      }
    else if( printvepro )
      {
      res += PrintVEPRO(filename, verbose!= 0);
      }
    else if( printsds )
      {
      res += PrintSDS(filename, verbose!= 0);
      }
    else if( printct3 )
      {
      res += PrintCT3(filename, verbose!= 0);
      }
    else if( printelscint )
      {
      res += PrintELSCINT(filename, verbose!= 0);
      }
    else if( printpdb )
      {
      res += PrintPDB(filename, verbose!= 0);
      }
    else if( printcsa )
      {
      res += PrintCSA(filename);
      }
    else if( dump )
      {
      res += DoOperation<gdcm::Dumper>(filename);
      }
    else
      {
      res += DoOperation<gdcm::Printer>(filename);
      }
    // ...
    if ( verbose )
      std::cerr << "Filename: " << filename << std::endl;
    }

  return res;
}

/*
 * Harvested data:
 * A lot of them are still non-obvious

Most obvious ones:
ETL -> Echo Train Length
FLIPANG -> Flip Angle
MATRIXX / MATRIXY ->  Acquisition Matrix
SLTHICK ->  Slice Thickness

   ENTRY "Feet First"
+  POSITION "Supine"
--------------------
=  Patient Position


Full list:

ANREF "IC"
ANREF "NA"
ANREF "SN"
AUTOCF "Water"
AUTOSCIC "0"
AUTOSCIC "2"
AUTOSHIM "Auto"
AUTOSHIM "Off"
AUTOSHIM "Yes"
AUTOSUBOPTIONS "0"
AUTOTRGTYPE "0"
AUTOTRIGWIN "0"
AUTOVOICE "0"
B4PAUSE "0"
BPMMODE "0"
BWRT "0"
BWRT "-1"
CLOC1 "0.0"
CLOC1 "L4.7"
CLOC1 "L5.9"
CLOC2 "0.0"
CLOC2 "P20.0"
CLOC2 "P42.2"
CLOC2 "P44.5"
CLOC3 "0.0"
CLOC3 "S7.0"
CLOC3 "S8.2"
COIL "5GP"
COIL "8HRBRAIN"
COIL "HEAD"
COIL "LOOP2CM"
CONTAG "GAD"
CONTAM "10    GAD"
CONTAM "No    "
CONTAM "Yes   "
CONTRAST "No"
CONTRAST "Yes"
DELACQ "Minimum"
DUMACQ "0"
ELOC1 "L12.4"
ELOC1 "L142.9"
ELOC1 "L1.6"
ELOC1 "L2.2"
ELOC1 "L4.9"
ELOC1 "L5.9"
ELOC1 "L80.1"
ELOC1 "L84.1"
ELOC1 "L99.3"
ELOC1 "S65.4"
ELOC1 "S66.5"
ELOC1 "S89.0"
ELOC2 "0.0"
ELOC2 "A18.3"
ELOC2 "A43.5"
ELOC2 "A79.2"
ELOC2 "A87.6"
ELOC2 "L6.9"
ELOC2 "P27.4"
ELOC2 "P38.8"
ELOC2 "P48.8"
ELOC2 "P49.4"
ELOC3 "A12.8"
ELOC3 "I111.9"
ELOC3 "I27.7"
ELOC3 "I7.1"
ELOC3 "P21.2"
ELOC3 "P31.1"
ELOC3 "S12.3"
ELOC3 "S1.7"
ELOC3 "S31.8"
ELOC3 "S5.4"
ELOC3 "S7.0"
ELOC3 "S9.8"
ENTRY "Head First"
ETL "17"
ETL "2"
ETL "24"
ETL "3"
ETL "6"
ETL "8"
ETL "9"
FILTCHOICE "None"
FLDIR "Slice"
FLIPANG "12"
FLIPANG "17"
FLIPANG "20"
FLIPANG "36"
FLIPANG "8"
FLIPANG "90"
FOV "12"
FOV "14"
FOV "24"
FOV "24.0"
FOV "3"
FOV "30"
FOV "4"
FOV "6"
FOV "8"
FOVCNT1 "0.0"
FOVCNT2 "0.0"
FOVCNT2 "P21.2"
FOVCNT2 "P31.1"
GRADMODE "WHOLE"
GRADMODE "ZOOM"
GRIP_NUMPSCVOL "0"
GRIP_NUMSLGROUPS "0"
GRIP_NUMSLGROUPS "1"
GRIP_PSCVOL1 "0"
GRIP_PSCVOL2 "0"
GRIP_PSCVOLFOV "0"
GRIP_PSCVOLFOV "0.000000"
GRIP_PSCVOLTHICK "0"
GRIP_PSCVOLTHICK "0.000000"
GRIP_SATGROUP1 "0"
GRIP_SATGROUP2 "0"
GRIP_SATGROUP3 "0"
GRIP_SATGROUP4 "0"
GRIP_SATGROUP5 "0"
GRIP_SATGROUP6 "0"
GRIP_SLGROUP1 "0.000000 -21.170785 -13.463666 0.000000 0.000000 1.000000 1.000000 0.000000 0.000000 0.000000 -1.000000 0.000000 1 0.000000 1 0"
GRIP_SLGROUP1 "0.000000 -31.122005 2.926577 0.000000 0.000000 1.000000 0.000000 1.000000 0.000000 1.000000 0.000000 0.000000 26 0.000000 1 0"
GRIP_SLGROUP1 "-13.163267 0.000000 25.592358 0.005670 0.000000 -0.999984 0.999984 0.000000 0.005670 0.000000 -1.000000 0.000000 56 0.000000 1 0 1"
GRIP_SLGROUP1 "3.135807 14.667716 -32.340976 -0.997518 0.043626 0.055276 -0.056814 -0.962372 -0.265728 0.041603 -0.268209 0.962462 1 0.000000 1 0 1"
GRIP_SPECTRO "0"
GRIP_TRACKER "0"
GRXOPT "0"
GRXOPT "2"
IEC_ACCEPT "ON"
IMODE "2D"
IMODE "3D"
INITSTATE "0"
IOPT "EDR, Fast, IrP"
IOPT "EPI, FMRI"
IOPT "Fast, IrP"
IOPT "Fast, ZIP512, FR"
IOPT "FC, EDR, TRF, Fast, ZIP512"
IOPT "FC, VBw, EDR"
IOPT "None"
IOPT "NPW, Seq, VBw, TRF, Fast"
IOPT "NPW, TRF, Fast, ZIP512, FR"
IOPT "NPW, VBw, EDR, Fast, ZIP2"
IOPT "NPW, VBw, Fast"
IOPT "NPW, ZIP512"
IOPT "TRF, Fast"
IOPT "VBw, EDR, Fast"
IOPT "VBw, Fast"
MASKPAUSE "0"
MASKPHASE "0"
MATRIXX "192"
MATRIXX "256"
MATRIXX "288"
MATRIXX "320"
MATRIXX "416"
MATRIXX "512"
MATRIXX "64"
MATRIXY "128"
MATRIXY "160"
MATRIXY "192"
MATRIXY "224"
MATRIXY "256"
MATRIXY "320"
MATRIXY "64"
MONSAR "y"
NECHO "1"
NEX "1.00"
NEX "1.50"
NEX "2.00"
NEX "3.00"
NEX "4.00"
NOSLC "1"
NOSLC "12"
NOSLC "15"
NOSLC "19"
NOSLC "20"
NOSLC "21"
NOSLC "24"
NOSLC "26"
NOSLC "56"
NOTES ".pn/_2"
NOTES ".pn/_3"
NOTES ".pn/_4"
NUMACCELFACTOR "1.00"
NUMACCELFACTOR "Recommended"
NUMACQS "0"
NUMACQS "2"
NUMSHOTS "1"
OVLPLOC "0"
PAUSEDELMASKACQ "1"
PDGMSTR "None"
PHASEASSET "1.00"
PHASECORR "No"
PHASECORR "Yes"
PHASEFOV "0.75"
PHASEFOV "1.00"
PLANE "3-PLANE"
PLANE "AXIAL"
PLANE "OBLIQUE"
PLUG "0"
PLUG "11"
PLUG "14"
PLUG "22"
PLUG "23"
PLUG "45"
PLUG "5"
PLUG "6"
PLUG "9"
POSITION "Prone"
POSITION "Supine"
PRESETDELAY "0.0"
PSDNAME "fse-xl"
PSDTRIG "0"
PSEQ "FRFSE-XL"
PSEQ "FSE-XL"
PSEQ "Gradient Echo"
PSEQ "IR"
PSEQ "Localizer"
PSEQ "SPGR"
PSEQ "Spin Echo"
RBW "12.50"
RBW "14.71"
RBW "15.63"
RBW "17.86"
RBW "20.83"
RBW "22.73"
RBW "25.00"
RBW "31.25"
SATLOCZ1 "9990"
SATLOCZ2 "9990"
SATTHICKZ1 "40.0"
SATTHICKZ2 "40.0"
SEDESC "3D FSPGR IR"
SEDESC "3DIR PREP"
SEDESC "3 plane loc"
SEDESC "AX FSE T1"
SEDESC "AX FSE T2"
SEDESC "AX T2*"
SEDESC "FATSAT T2 FSE Scout"
SEDESCFLAG "1"
SEDESC "LOCALIZER-RATCOIL"
SEDESC "O-Ax FATSAT T2 FSE high Res"
SEDESC "Oblique PD AX"
SEDESC "Oblique STIR"
SEDESC "Oblique T1 AX +C"
SEDESC "Oblique T1-SAG"
SEDESC "Oblique T1-SAG+C"
SEDESC "Oblique T2 AX."
SEDESC "O-Cor T1 "
SEDESC "RUN 1"
SEDESC "SPGR3D-HRES-Brasch"
SEDESC "SPGR3D-LRES-Brasch"
SEPSERIES "0"
SL3PLANE "0"
SL3PLANE "1"
SL3PLANE1 "0"
SL3PLANE1 "5"
SL3PLANE2 "0"
SL3PLANE2 "5"
SL3PLANE3 "0"
SL3PLANE3 "5"
SLABLOC "128"
SLABLOC "144"
SLABLOC "64"
SLABLOC "80"
SLICEASSET "1.00"
SLICEORDER "1"
SLOC1 "I35.2"
SLOC1 "I59.6"
SLOC1 "I93.4"
SLOC1 "L13.9"
SLOC1 "L1.6"
SLOC1 "L2.1"
SLOC1 "L5.0"
SLOC1 "L5.9"
SLOC1 "L84.1"
SLOC1 "L85.9"
SLOC1 "L99.2"
SLOC1 "R86.3"
SLOC2 "0.0"
SLOC2 "A11.0"
SLOC2 "A115.0"
SLOC2 "A79.2"
SLOC2 "A80.8"
SLOC2 "P34.4"
SLOC2 "P37.0"
SLOC2 "P46.5"
SLOC2 "P50.2"
SLOC3 "I27.8"
SLOC3 "I37.0"
SLOC3 "I7.1"
SLOC3 "S12.3"
SLOC3 "S163.1"
SLOC3 "S1.7"
SLOC3 "S5.4"
SLOC3 "S7.0"
SLPERLOC "274"
SLTHICK "0.2"
SLTHICK "0.7"
SLTHICK "1.2"
SLTHICK "1.3"
SLTHICK "3.0"
SLTHICK "4.0"
SLTHICK "5.0"
SLTHICK "5.5"
SPC "0.0"
SPC "1.5"
SPCPERPLANE1 "0.0"
SPCPERPLANE1 "1.5"
SPCPERPLANE2 "0.0"
SPCPERPLANE2 "1.5"
SPCPERPLANE3 "0.0"
SPCPERPLANE3 "1.5"
STATION "0"
SUPPTQ "1"
SWAPPF "A/P"
SWAPPF "R/L"
SWAPPF "S/I"
SWAPPF "Unswap"
TAG_SPACE "7"
TAG_TYPE "None"
TBLDELTA "0.00"
TE "100.0"
TE "102.0"
TE "15.0"
TE "30.0"
TE "50.0"
TE "Min Full"
TE "Minimum"
TI "150"
TI "450"
TI "500"
TOTALNOSTATION "0"
TR "2000.0"
TR "3000.0"
TR "4000.0"
TR "4125.0"
TR "4575.0"
TR "475.0"
TR "500.0"
TR "5200.0"
TR "525.0"
TR "5325.0"
TR "6600.0"
TRACKLEN "200.0"
TRACKTHICK "20.0"
TRACTIVE "0"
TRACTIVE "4"
TRICKSIMG "1"
TRREST "0"
TRREST "4"
USERCV0 "0.00"
USERCV0 "1.00"
USERCV21 "0.00"
USERCV23 "100.00"
USERCV4 "0.00"
USERCV6 "0.00"
USERCV7 "0.00"
USERCV_MASK "0"
USERCV_MASK "1"
USERCV_MASK "128"
USERCV_MASK "192"
USERCV_MASK "2097344"
USERCV_MASK "6144"
USERCV_MASK "64"
USERCV_MASK "8388688"
VIEWORDER "1"

*/
