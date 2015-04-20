/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPDBHeader.h"
#include "gdcmPrivateTag.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmExplicitDataElement.h"
#include "gdcmSwapper.h"
#include "gdcmDeflateStream.h"

namespace gdcm
{

/*
 * In some GE MEDICAL SYSTEMS image one can find a Data Element: 0025,xx1b,GEMS_SERS_01
 * which is documented as Protocol Data Block (compressed).
 * in fact this is a simple text format compressed using the gzip algorithm
 *
 * Typically one could do:
 *
 *   $ gdcmraw -i input.dcm -o output.raw -t 0025,101b
 *
 * Skip the binary length (little endian encoding):
 *
 *   $ dd bs=4 skip=1 if=output.raw of=foo
 *
 * Check file type:
 *
 *   $ file foo
 *   foo: gzip compressed data, was "Ex421Ser8Scan1", from Unix
 *
 * Gunzip !
 *   $ gzip -dc < foo > bar
 *   $ cat bar
 *
 * THANKS to: John Reiser (BitWagon.com) for hints
 *
 * For sample see:
 * GE_MR_0025xx1bProtocolDataBlock.dcm
 * ( <=> http://server.oersted.dtu.dk/personal/jw/jwpublic/courses/31540/mri/second_set/dicom/t2/b17.dcm)
 */

/* Typical output:
 *
 * ENTRY "Head First"
 * POSITION "Supine"
 * ANREF "NA"
 * COIL "HEAD"
 * PLANE "OBLIQUE"
 * SEDESCFLAG "1"
 * SEDESC "AX FSE T2"
 * IMODE "2D"
 * PSEQ "FSE-XL"
 * IOPT "FC, EDR, TRF, Fast"
 * PLUG "22"
 * FILTCHOICE "None"
 * BWRT "-1"
 * TRICKSIMG "1"
 * TAG_SPACE "7"
 * TAG_TYPE "None"
 * USERCV0 "0.00"
 * USERCV6 "0.00"
 * USERCV7 "0.00"
 * USERCV21 "0.00"
 * USERCV_MASK "2097344"
 * TE "102.0"
 * NECHO "1"
 * TR "5720.0"
 * NUMACQS "1"
 * ETL "17"
 * BPMMODE "0"
 * AUTOTRGTYPE "0"
 * PSDTRIG "0"
 * SLICEORDER "1"
 * VIEWORDER "1"
 * TRREST "0"
 * TRACTIVE "0"
 * SLICEASSET "1.00"
 * PHASEASSET "1.00"
 * SEPSERIES "0"
 * AUTOTRIGWIN "0"
 * FOV "24.0"
 * SLTHICK "2.0"
 * SPC "2.0"
 * GRXOPT "0"
 * SLOC1 "L11.8"
 * SLOC2 "P29.9"
 * SLOC3 "I50.0"
 * ELOC1 "L11.6"
 * ELOC2 "P29.4"
 * ELOC3 "S53.9"
 * NOSLC "27"
 * SL3PLANE "0"
 * SL3PLANE1 "0"
 * SL3PLANE2 "0"
 * SL3PLANE3 "0"
 * SPCPERPLANE1 "0.0"
 * SPCPERPLANE2 "0.0"
 * SPCPERPLANE3 "0.0"
 * MATRIXX "448"
 * MATRIXY "224"
 * SWAPPF "A/P"
 * NEX "4.00"
 * CONTRAST "No"
 * CONTAM "Yes   "
 * TBLDELTA "0.00"
 * PHASEFOV "0.75"
 * RBW "31.25"
 * AUTOSHIM "Auto"
 * PHASECORR "Yes"
 * FLDIR "Freq"
 * NUMACCELFACTOR "1.00"
 * PAUSEDELMASKACQ "1"
 * NOTES ".pn/_3"
 * GRIP_NUMSLGROUPS "1"
 * GRIP_SLGROUP1 "-11.703952 -29.677423 1.949659 0.002380 0.004775 0.999985 0.999997 0.000175 -0.002380 0.000186 -0.999988 0.004775 27 0.000000 1 0 0"
 * GRIP_SATGROUP1 "0"
 * GRIP_SATGROUP2 "0"
 * GRIP_SATGROUP3 "0"
 * GRIP_SATGROUP4 "0"
 * GRIP_SATGROUP5 "0"
 * GRIP_SATGROUP6 "0"
 * GRIP_TRACKER "0"
 * GRIP_SPECTRO "0"
 * GRIP_NUMPSCVOL "0"
 * GRIP_PSCVOL1 "0"
 * GRIP_PSCVOL2 "0"
 * GRIP_PSCVOLFOV "0.000000"
 * GRIP_PSCVOLTHICK "0.000000"
 * AUTOSUBOPTIONS "0"
 * AUTOSCIC "0"
 * AUTOVOICE "0"
 * PRESETDELAY "0.0"
 * MASKPHASE "0"
 * MASKPAUSE "0"
 * TOTALNOSTATION "0"
 * STATION "0"
 */

PDBElement PDBHeader::PDBEEnd = PDBElement( );

  const PDBElement& PDBHeader::GetPDBEEnd() const
  {
    return PDBEEnd;
  }

int PDBHeader::readprotocoldatablock(const char *input, size_t inputlen, bool verbose)
{
  (void)verbose;
  // First 4 bytes are the length (again)
  uint32_t len = *(uint32_t*)input;
  SwapperNoOp::SwapArray(&len,1);
  //if( verbose )
  //  std::cout << len << "," << inputlen << std::endl;
  if( len + 4 + 1 == inputlen )
    {
    //if( verbose )
    //  std::cout << "gzip stream was padded with an extra 0 \n";
    }
  else if( len + 4 == inputlen )
    {
    //if( verbose )
    //  std::cout << "gzip stream was not padded with an extra 0 \n";
    }
  else
    {
    //std::cerr << "Found the Protocol Data Block but could not read length..." << std::endl;
    return 1;
    }
  // Alright we need to check if the binary blob was padded, if padded we need to
  // discard the trailing \0 to please gzip:
  std::string str( input + 4, input + len );
  std::istringstream is( str );

  zlib_stream::zip_istream gzis( is );

//  if (gzis.is_gzip())
//    {
//    std::cout<<"crc check: "<<( gzis.check_crc() ? "ok" : "failed");
//    std::cout << std::endl;
//    }

  std::string out;
  //while( gzis >> out )
  while( std::getline(gzis , out ) )
    {
    PDBElement pdbel;
    //std::cout << out << std::endl;
    std::istringstream is2( out );
    std::string name, value;
    is2 >> name;
    std::getline(is2, value);
    pdbel.SetName( name.c_str() );
    // remove the first space character and the first & last " character
    std::string value2( value.begin()+2, value.end()-1);
    pdbel.SetValue( value2.c_str() );
    InternalPDBDataSet.push_back( pdbel );
    }
  //std::cout << out.size();

  return 0;
}

/*
int DumpProtocolDataBlock(const std::string & filename, bool verbose)
{
  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }
  const gdcm::DataSet& ds = reader.GetFile().GetDataSet();

  const gdcm::PrivateTag tprotocoldatablock(0x0025,0x1b,"GEMS_SERS_01");
  if( !ds.FindDataElement( tprotocoldatablock) )
    {
    std::cerr << "Could not find tag: " << tprotocoldatablock << std::endl;
    return 1;
    }
  const gdcm::DataElement& protocoldatablock= ds.GetDataElement( tprotocoldatablock);
  if ( protocoldatablock.IsEmpty() ) return 1;
  const gdcm::ByteValue * bv = protocoldatablock.GetByteValue();

  std::cout << "Dumping: " << tprotocoldatablock << std::endl;
  int ret = readprotocoldatablock( bv->GetPointer(), bv->GetLength(), verbose );

  return ret;
}
*/

bool PDBHeader::LoadFromDataElement(DataElement const &protocoldatablock)
{
  InternalPDBDataSet.clear();
  if ( protocoldatablock.IsEmpty() ) return false;
  const gdcm::ByteValue * bv = protocoldatablock.GetByteValue();

  //std::cout << "Dumping: " << tprotocoldatablock << std::endl;
  int ret = readprotocoldatablock( bv->GetPointer(), bv->GetLength(), false);

  if(ret) return false;
  return true;
}

void PDBHeader::Print(std::ostream &os) const
{
  std::vector<PDBElement>::const_iterator it = InternalPDBDataSet.begin();

  for(; it != InternalPDBDataSet.end(); ++it)
    {
    os << *it << std::endl;
    }
}

const PDBElement &PDBHeader::GetPDBElementByName(const char *name)
{
  std::vector<PDBElement>::const_iterator it = InternalPDBDataSet.begin();
  for(; it != InternalPDBDataSet.end(); ++it)
    {
    const char *itname = it->GetName();
    if( strcmp(name, itname) == 0 )
      {
      return *it;
      }
    }
    return GetPDBEEnd();
}

bool PDBHeader::FindPDBElementByName(const char *name)
{
  std::vector<PDBElement>::const_iterator it = InternalPDBDataSet.begin();
  for(; it != InternalPDBDataSet.end(); ++it)
    {
    const char *itname = it->GetName();
    if( strcmp(name, itname) == 0 )
      {
      return true;
      }
    }
  return false;
}

static const char pdbheader[] = "GEMS_SERS_01";
static const gdcm::PrivateTag t1(0x0025,0x001b,pdbheader);

const PrivateTag & PDBHeader::GetPDBInfoTag()
{
  return t1;
}

} // end namespace gdcm
