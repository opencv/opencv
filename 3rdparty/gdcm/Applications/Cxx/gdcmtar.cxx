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
 * tar is a UNIX program for archiving.
 * Two types of operations are possible: concatenate / extract
 * Thus the name of 'gdcmtar' to concatenate a list of 2D slices into a multi frames
 * and the other way around: extract 2D slices from a multi frames image
 * It also support the fake multi frame image (CSA MOSAIC)
 */

#include "gdcmReader.h"
#include "gdcmVersion.h"
#include "gdcmImageReader.h"
#include "gdcmDataElement.h"
#include "gdcmImageWriter.h"
#include "gdcmSplitMosaicFilter.h"
#include "gdcmFilename.h"
#include "gdcmFilenameGenerator.h"
#include "gdcmDirectionCosines.h"
#include "gdcmImageHelper.h"
#include "gdcmUIDGenerator.h"
#include "gdcmUIDs.h"
#include "gdcmGlobal.h"
#include "gdcmDirectory.h"
#include "gdcmScanner.h"
#include "gdcmIPPSorter.h"
#include "gdcmAttribute.h"

#include <string>
#include <iostream>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

static void PrintVersion()
{
  std::cout << "gdcmtar: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmtar [OPTION] [FILE]" << std::endl;
  std::cout << "Concatenate/Extract DICOM files.\n";
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input     DICOM filename" << std::endl;
  std::cout << "  -o --output    DICOM filename" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     --enhance    enhance (default)" << std::endl;
  std::cout << "  -U --unenhance  unenhance" << std::endl;
  std::cout << "  -M --mosaic     Split SIEMENS Mosaic image into multiple frames." << std::endl;
  std::cout << "  -p --pattern    Specify trailing file pattern." << std::endl;
  std::cout << "     --root-uid        Root UID." << std::endl;
  //std::cout << "     --resources-path     Resources path." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose    more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning    print warning info." << std::endl;
  std::cout << "  -D --debug      print debug info." << std::endl;
  std::cout << "  -E --error      print error info." << std::endl;
  std::cout << "  -h --help       print help." << std::endl;
  std::cout << "  -v --version    print version." << std::endl;
  std::cout << "Env var:" << std::endl;
  std::cout << "  GDCM_ROOT_UID Root UID" << std::endl;
  //std::cout << "  GDCM_RESOURCES_PATH path pointing to resources files (Part3.xml, ...)" << std::endl;
}

/*
 * The following example is a basic sorted which should work in generic cases.
 * It sort files based on:
 * Study Instance UID
 *   Series Instance UID
 *     Frame of Reference UID
 *       Image Orientation (Patient)
 *         Image Position (Patient) (Sorting based on IPP + IOP)
 */

namespace gdcm {
  const Tag T0(0x0008,0x0016); // SOP Class UID
  const Tag T1(0x0020,0x000d); // Study Instance UID
  const Tag T2(0x0020,0x000e); // Series Instance UID
  const Tag T3(0x0020,0x0052); // Frame of Reference UID
  const Tag T4(0x0020,0x0037); // Image Orientation (Patient)

class DiscriminateVolume
{
private:
  static const bool debuggdcmtar = false;
  std::vector< Directory::FilenamesType > SortedFiles;
  std::vector< Directory::FilenamesType > UnsortedFiles;

  Directory::FilenamesType GetAllFilenamesFromTagToValue(
    Scanner const & s, Directory::FilenamesType const &filesubset, Tag const &t, const char *valueref)
{
  Directory::FilenamesType theReturn;
  if( valueref )
    {
    size_t len = strlen( valueref );
    Directory::FilenamesType::const_iterator file = filesubset.begin();
    for(; file != filesubset.end(); ++file)
      {
      const char *filename = file->c_str();
      const char * value = s.GetValue(filename, t);
      if( value && strncmp(value, valueref, len ) == 0 )
        {
        theReturn.push_back( filename );
        }
      }
    }
  return theReturn;
}

void ProcessAIOP(Scanner const & , Directory::FilenamesType const & subset, const char *iopval)
{
  if( debuggdcmtar )
  std::cout << "IOP: " << iopval << std::endl;
  IPPSorter ipp;
  ipp.SetComputeZSpacing( true );
  ipp.SetZSpacingTolerance( 1e-3 ); // ??
  bool b = ipp.Sort( subset );
  if( !b )
    {
    // If you reach here this means you need one more parameter to discriminiat this
    // series. Eg. T1 / T2 intertwinted. Multiple Echo (0018,0081)
    if( debuggdcmtar )
      {
      std::cerr << "Failed to sort: " << subset.begin()->c_str() << std::endl;
      for(
        Directory::FilenamesType::const_iterator file = subset.begin();
        file != subset.end(); ++file)
        {
        std::cerr << *file << std::endl;
        }
      }
    UnsortedFiles.push_back( subset );
    return ;
    }
  if( debuggdcmtar )
  ipp.Print( std::cout );
  SortedFiles.push_back( ipp.GetFilenames() );
}

void ProcessAFrameOfRef(Scanner const & s, Directory::FilenamesType const & subset, const char * frameuid)
{
  // In this subset of files (belonging to same series), let's find those
  // belonging to the same Frame ref UID:
  Directory::FilenamesType files = GetAllFilenamesFromTagToValue(
    s, subset, T3, frameuid);

  std::set< std::string > iopset;

  for(
    Directory::FilenamesType::const_iterator file = files.begin();
    file != files.end(); ++file)
    {
    //std::cout << *file << std::endl;
    const char * value = s.GetValue(file->c_str(), gdcm::T4 );
    assert( value );
    iopset.insert( value );
    }
  size_t n = iopset.size();
  if ( n == 0 )
    {
    assert( files.empty() );
    return;
    }

  if( debuggdcmtar )
  std::cout << "Frame of Ref: " << frameuid << std::endl;
  if ( n == 1 )
    {
    ProcessAIOP(s, files, iopset.begin()->c_str() );
    }
  else
    {
    const char *f = files.begin()->c_str();
    if( debuggdcmtar )
    std::cerr << "More than one IOP: " << f << std::endl;
    // Make sure that there is actually 'n' different IOP
    gdcm::DirectionCosines ref;
    gdcm::DirectionCosines dc;
    for(
      std::set< std::string >::const_iterator it = iopset.begin();
      it != iopset.end(); ++it )
      {
      ref.SetFromString( it->c_str() );
      for(
        Directory::FilenamesType::const_iterator file = files.begin();
        file != files.end(); ++file)
        {
        std::string value = s.GetValue(file->c_str(), gdcm::T4 );
        if( value != it->c_str() )
          {
          dc.SetFromString( value.c_str() );
          const double crossdot = ref.CrossDot(dc);
          const double eps = std::fabs( 1. - crossdot );
          if( eps < 1e-6 )
            {
            std::cerr << "Problem with IOP discrimination: " << file->c_str()
              << " " << it->c_str() << std::endl;
            return;
            }
          }
        }
      }
      // If we reach here this means there is actually 'n' different IOP
    for(
      std::set< std::string >::const_iterator it = iopset.begin();
      it != iopset.end(); ++it )
      {
      const char *iopvalue = it->c_str();
      Directory::FilenamesType iopfiles = GetAllFilenamesFromTagToValue(
        s, files, T4, iopvalue );
      ProcessAIOP(s, iopfiles, iopvalue );
      }
    }
}

void ProcessASeries(Scanner const & s, const char * seriesuid)
{
  if( debuggdcmtar )
    std::cout << "Series: " << seriesuid << std::endl;
  // let's find all files belonging to this series:
  Directory::FilenamesType seriesfiles = GetAllFilenamesFromTagToValue(
    s, s.GetFilenames(), T2, seriesuid);

  gdcm::Scanner::ValuesType vt3 = s.GetValues(T3);
  for(
    gdcm::Scanner::ValuesType::const_iterator it = vt3.begin()
    ; it != vt3.end(); ++it )
    {
    ProcessAFrameOfRef(s, seriesfiles, it->c_str());
    }
}

void ProcessAStudy(Scanner const & s, const char * studyuid)
{
  if( debuggdcmtar )
    std::cout << "Study: " << studyuid << std::endl;
  gdcm::Scanner::ValuesType vt2 = s.GetValues(T2);
  if( vt2.empty() )
    std::cerr << "No Series Found" << std::endl;
  for(
    gdcm::Scanner::ValuesType::const_iterator it = vt2.begin()
    ; it != vt2.end(); ++it )
    {
    ProcessASeries(s, it->c_str());
    }
}
public:

void Print( std::ostream & os )
{
  os << "Sorted Files: " << std::endl;
  for(
    std::vector< Directory::FilenamesType >::const_iterator it = SortedFiles.begin();
    it != SortedFiles.end(); ++it )
    {
    os << "Group: " << std::endl;
    for(
      Directory::FilenamesType::const_iterator file = it->begin();
      file != it->end(); ++file)
      {
      os << *file << std::endl;
      }
    }
  os << "Unsorted Files: " << std::endl;
  for(
    std::vector< Directory::FilenamesType >::const_iterator it = UnsortedFiles.begin();
    it != UnsortedFiles.end(); ++it )
    {
    os << "Group: " << std::endl;
    for(
      Directory::FilenamesType::const_iterator file = it->begin();
      file != it->end(); ++file)
      {
      os << *file << std::endl;
      }
    }

}

  std::vector< Directory::FilenamesType > const & GetSortedFiles() const { return SortedFiles; }
  std::vector< Directory::FilenamesType > const & GetUnsortedFiles() const { return UnsortedFiles; }

void ProcessIntoVolume( Scanner const & s )
{
  gdcm::Scanner::ValuesType vt1 = s.GetValues( gdcm::T1 );
  for(
    gdcm::Scanner::ValuesType::const_iterator it = vt1.begin()
    ; it != vt1.end(); ++it )
    {
    ProcessAStudy( s, it->c_str() );
    }

}

};

static bool ConcatenateImages(Image &im1, Image const &im2)
{
  DataElement& de1 = im1.GetDataElement();
  if( de1.GetByteValue() )
    {
    const ByteValue *bv1 = de1.GetByteValue();
    std::vector<char> v1 = *bv1;
    const DataElement& de2 = im2.GetDataElement();
    const ByteValue *bv2 = de2.GetByteValue();
    const std::vector<char> & v2 = *bv2;
    v1.insert( v1.end(), v2.begin(), v2.end() );

    de1.SetByteValue(&v1[0], (uint32_t)v1.size());
    }
  else if( de1.GetSequenceOfFragments() )
    {
    SequenceOfFragments *sqf1 = de1.GetSequenceOfFragments();
    assert( sqf1 );
    const DataElement& de2 = im2.GetDataElement();
    const SequenceOfFragments *sqf2 = de2.GetSequenceOfFragments();
    assert( sqf2 );
    assert( sqf2->GetNumberOfFragments() == 1 );
    const Fragment& frag = sqf2->GetFragment(0);
    sqf1->AddFragment(frag);
    }
  else
    {
    return false;
    }

  // update meta info
  unsigned int z = im1.GetDimension(2);
  im1.SetDimension(2, z + 1 );
  return true;
}

} // namespace gdcm


static int MakeImageEnhanced( std::string const & filename, std::string const &outfilename )
{
  if( !gdcm::System::FileIsDirectory(filename.c_str()) )
    {
    std::cerr << "Input needs to be directory" << std::endl;
    return 1;
    }

  gdcm::Directory d;
  d.Load( filename.c_str(), true ); // recursive !

  gdcm::Scanner s;
  s.AddTag( gdcm::T0 );
  s.AddTag( gdcm::T1 );
  s.AddTag( gdcm::T2 );
  s.AddTag( gdcm::T3 );
  s.AddTag( gdcm::T4 );
  bool b = s.Scan( d.GetFilenames() );
  if( !b )
    {
    std::cerr << "Scanner failed" << std::endl;
    return 1;
    }

  // For now only accept MR Image Storage
  gdcm::Scanner::ValuesType vt = s.GetValues(gdcm::T0);
  if( vt.size() != 1 ) return 1;

  const char *sop = vt.begin()->c_str();
  gdcm::MediaStorage msorig = gdcm::MediaStorage::GetMSType( sop );
  if( msorig != gdcm::MediaStorage::MRImageStorage
   && msorig != gdcm::MediaStorage::CTImageStorage )
    {
    std::cerr << "Sorry MediaStorage not supported: [" << sop << "]" << std::endl;
    return 1;
    }

  gdcm::DiscriminateVolume dv;
  dv.ProcessIntoVolume( s );
//  dv.Print( std::cout );

  // gdcm::DataElement &de = im.GetImage().GetDataElement();
  std::vector< gdcm::Directory::FilenamesType > const &sorted = dv.GetSortedFiles();
  if( !gdcm::System::MakeDirectory( outfilename.c_str() ) )
    {
    std::cerr << "Could not create dir: " << outfilename << std::endl;
    return 1;
    }
  for(
    std::vector< gdcm::Directory::FilenamesType >::const_iterator it = sorted.begin();
    it != sorted.end(); ++it )
    {
    gdcm::ImageWriter im0;

    gdcm::Directory::FilenamesType const & files = *it;
    gdcm::Directory::FilenamesType::const_iterator file = files.begin();

    const char *reffile = file->c_str();
    // construct the target dir:
    const char* studyuid = s.GetValue(reffile, gdcm::T1);
    const char* seriesuid = s.GetValue(reffile, gdcm::T2);
    const char* frameuid = s.GetValue(reffile, gdcm::T3);
    std::string targetdir = outfilename;
    targetdir += '/';
    targetdir += studyuid;
    targetdir += '/';
    targetdir += seriesuid;
    targetdir += '/';
    targetdir += frameuid;
    // construct the target name:
    std::string targetname = targetdir;

    targetdir += "/old/";

    // make sure the dir exist first:
    if( !gdcm::System::MakeDirectory( targetdir.c_str() ) )
      {
      std::cerr << "Could not create dir: " << targetdir << std::endl;
      return 1;
      }

    gdcm::FilenameGenerator fg;
    fg.SetNumberOfFilenames( files.size() );
    fg.SetPrefix( targetdir.c_str() );
    fg.SetPattern( "%04d.dcm" );
    if( !fg.Generate() )
      {
      assert( 0 );
      }

    gdcm::ImageReader reader0;
    reader0.SetFileName( reffile );
    if( !reader0.Read() )
      {
      assert( 0 );
      }
    gdcm::Image &currentim = reader0.GetImage();
    assert( currentim.GetNumberOfDimensions( ) == 2 );
    currentim.SetNumberOfDimensions( 3 );
    size_t count = 0;

    //gdcm::ImageWriter writer;
    gdcm::Writer writer0;
    writer0.SetFileName( fg.GetFilename( count ) );
    writer0.SetFile( reader0.GetFile() );
    writer0.GetFile().GetHeader().Clear();
    if( !writer0.Write() )
      {
      assert( 0 );
      }
    ++file;
    ++count;

    for( ; file != files.end(); ++file, ++count )
      {
      gdcm::ImageReader reader;
      reader.SetFileName( file->c_str() );
      if( !reader.Read() )
        {
        assert( 0 );
        }
      const gdcm::Image &im = reader.GetImage();

      //gdcm::ImageWriter writer;
      gdcm::Writer writer;
      writer.SetFileName( fg.GetFilename( count ) );
      writer.SetFile( reader.GetFile() );
      writer.GetFile().GetHeader().Clear();
      if( !writer.Write() )
        {
        assert( 0 );
        }

      if( !ConcatenateImages(currentim, im) )
        {
        assert( 0 );
        }
      }

    im0.SetFileName( (targetname + "/new.dcm").c_str() );
    //  im.SetFile( reader.GetFile() );

    gdcm::DataSet &ds = im0.GetFile().GetDataSet();

    gdcm::MediaStorage ms;
    switch( msorig )
      {
    case gdcm::MediaStorage::CTImageStorage:
      ms = gdcm::MediaStorage::EnhancedCTImageStorage;
      break;
    case gdcm::MediaStorage::MRImageStorage:
      ms = gdcm::MediaStorage::EnhancedMRImageStorage;
      break;
    default:
      return 1;
      }

    gdcm::DataElement de( gdcm::Tag(0x0008, 0x0016) );
    const char* msstr = gdcm::MediaStorage::GetMSString(ms);
    de.SetByteValue( msstr, (uint32_t)strlen(msstr) );
    de.SetVR( gdcm::Attribute<0x0008, 0x0016>::GetVR() );
    ds.Insert( de );

    im0.SetImage( currentim );
    if( !im0.Write() )
      {
      std::cerr << "Could not write: " << std::endl;
      return 1;
      }

    }

  std::vector< gdcm::Directory::FilenamesType > const &unsorted = dv.GetUnsortedFiles();
  if( !unsorted.empty() )
    {
    std::string targetdir3 = outfilename;
    targetdir3 += "/unhandled/";
    if( !gdcm::System::MakeDirectory( targetdir3.c_str() ) )
      {
      std::cerr << "Could not create dir: " << outfilename << std::endl;
      return 1;
      }
    std::cerr << "Could not process the following files (please report): " << std::endl;
    std::vector< gdcm::Directory::FilenamesType >::const_iterator it = unsorted.begin();
    for( ; it != unsorted.end(); ++it )
      {
      gdcm::Directory::FilenamesType const & files = *it;
      gdcm::Directory::FilenamesType::const_iterator file = files.begin();
      for( ; file != files.end(); ++file )
        {
        const char *f = file->c_str();
        std::string targetdir2 = outfilename;
        targetdir2 += "/unhandled/";
        gdcm::Filename fn2( f );
        const char *outfn2 = fn2.GetName();
        targetdir2 += outfn2;
        //std::cerr << f << " -> " << targetdir2 << std::endl;
        std::ifstream in( f, std::ios::binary );
        std::ofstream out( targetdir2.c_str() , std::ios::binary );
        out << in.rdbuf();
        }
      }
    }

  return 0;
}

namespace gdcm
{

static const DataElement &GetNestedDataElement( const DataSet &ds, const Tag & t1, const Tag & t2 )
{
  assert( ds.FindDataElement( t1 ) );
  SmartPointer<SequenceOfItems> sqi1 = ds.GetDataElement( t1 ).GetValueAsSQ();
  assert( sqi1 );
  const Item &item1 = sqi1->GetItem(1);
  const DataSet & ds1 = item1.GetNestedDataSet();
  assert( ds1.FindDataElement( t2 ) );
  return ds1.GetDataElement( t2 );
}

static bool RemapSharedIntoOld( gdcm::DataSet & ds,
 SequenceOfItems *sfgs,
 SequenceOfItems *pffgs,
 unsigned int index )
{
  assert( sfgs );
  assert( pffgs );

  assert( sfgs->GetNumberOfItems() == 1 );
  Item const &item1 = sfgs->GetItem( 1 );
  const DataSet & sfgs_ds = item1.GetNestedDataSet();
#if 1
  // Repetition Time
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9112), Tag(0x0018,0x0080) ) );
  // Echo Train Length
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9112), Tag(0x0018,0x0091) ) );
  // Flip Angle
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9112), Tag(0x0018,0x1314) ) );
  // Number of Averages
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9119), Tag(0x0018,0x0083) ) );

  // Percent Sampling
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9125), Tag(0x0018,0x0093) ) );
  // Percent Phase Field of View
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9125), Tag(0x0018,0x0094) ) );
  // Receive Coil Name
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9042), Tag(0x0018,0x1250) ) );
  // Transmit Coil Name
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9049), Tag(0x0018,0x1251) ) );
  // InPlanePhaseEncodingDirection
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9125), Tag(0x0018,0x1312) ) );
  // TransmitterFrequency
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9006), Tag(0x0018,0x9098) ) );
  // InversionRecovery
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9115), Tag(0x0018,0x9009) ) );
  // FlowCompensation
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9115), Tag(0x0018,0x9010) ) );
  // ReceiveCoilType
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9042), Tag(0x0018,0x9043) ) );
  // QuadratureReceiveCoil
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9042), Tag(0x0018,0x9044) ) );
  // SlabThickness
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9107), Tag(0x0018,0x9104) ) );
  // MultiCoilDefinitionSequence
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9042), Tag(0x0018,0x9045) ) );
  // SlabOrientation
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9107), Tag(0x0018,0x9105) ) );
  // MidSlabPosition
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9107), Tag(0x0018,0x9106) ) );
  // OperatingModeSequence
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9112), Tag(0x0018,0x9176) ) );
  // MRAcquisitionPhaseEncodingStepsOutOf
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9125), Tag(0x0018,0x9232) ) );
  // SpecificAbsorptionRateSequence
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0018,0x9112), Tag(0x0018,0x9239) ) );
  // AnatomicRegionSequence
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0020,0x9071), Tag(0x0008,0x2218) ) );
  // Purpose of Reference Code Sequence
  // FIXME what if there is multiple purpose of rcs ?
  ds.Replace( GetNestedDataElement(sfgs_ds, Tag(0x0008,0x1140), Tag(0x0040,0xa170) ) );
#else
  for(
    DataSet::ConstIterator it = sfgs_ds.Begin();
    it != sfgs_ds.End(); ++it )
    {
    ds.Replace( *it );
    }
#endif

  Item const &item2 = pffgs->GetItem( index + 1 );
  const DataSet & pffgs_ds = item2.GetNestedDataSet();

#if 1
  // Effective Echo Time
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0018,0x9114), Tag(0x0018,0x9082) ) );
  // -> should also be Echo Time
  // Nominal Cardiac Trigger Delay Time
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0018,0x9118), Tag(0x0020,0x9153) ) );
  // Metabolite Map Description
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0018,0x9152), Tag(0x0018,0x9080) ) );
  // IPP
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9113), Tag(0x0020,0x0032) ) );
  // IOP
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9116), Tag(0x0020,0x0037) ) );
  // Slice Thickness
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0028,0x9110), Tag(0x0018,0x0050) ) );
  // Pixel Spacing
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0028,0x9110), Tag(0x0028,0x0030) ) );

  // window level
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0028,0x9132), Tag(0x0028,0x1050) ) );
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0028,0x9132), Tag(0x0028,0x1051) ) );

  // rescale slope/intercept
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0028,0x9145), Tag(0x0028,0x1052) ) );
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0028,0x9145), Tag(0x0028,0x1053) ) );
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0028,0x9145), Tag(0x0028,0x1054) ) );

  // FrameReferenceDateTime
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9111), Tag(0x0018,0x9151) ) );
  // FrameAcquisitionDuration
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9111), Tag(0x0018,0x9220) ) );
  // TemporalPositionIndex
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9111), Tag(0x0020,0x9128) ) );
  // InStackPositionNumber
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9111), Tag(0x0020,0x9057) ) );
  // FrameType
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0018,0x9226), Tag(0x0008,0x9007) ) );
  // DimensionIndexValues
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9111), Tag(0x0020,0x9157) ) );
  // FrameAcquisitionDateTime
  ds.Replace( GetNestedDataElement(pffgs_ds, Tag(0x0020,0x9111), Tag(0x0018,0x9074) ) );
  // Nominal Cardiac Trigger Delay Time -> Trigger Time
  //const DataElement &NominalCardiacTriggerDelayTime =
  //  GetNestedDataElement(pffgs_ds, Tag(0x0018,0x9226), Tag(0x0008,0x9007) );
#endif

  // (0020,9228) UL 158                                      #   4, 1 ConcatenationFrameOffsetNumber
  gdcm::Attribute<0x0020,0x9228> at = { index };
  ds.Replace( at.GetAsDataElement() );

  return true;
}

} // namespace gdcm

int main (int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;

  int rootuid = 0;
  std::string filename;
  std::string outfilename;
  std::string root;
  int resourcespath = 0;
  int mosaic = 0;
  int enhance = 1;
  int unenhance = 0;
  std::string xmlpath;

  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;

  std::string pattern;
  while (1) {
    //int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"output", 1, 0, 0},
        {"mosaic", 0, &mosaic, 1}, // split siemens mosaic into multiple frames
        {"pattern", 1, 0, 0},               // p
        {"enhance", 0, &enhance, 1},               // unenhance
        {"unenhance", 0, &unenhance, 1},               // unenhance
        {"root-uid", 1, &rootuid, 1}, // specific Root (not GDCM)
        //{"resources-path", 0, &resourcespath, 1},

// General options !
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},

        {0, 0, 0, 0}
    };

    c = getopt_long (argc, argv, "i:o:MUp:VWDEhv",
      long_options, &option_index);
    if (c == -1)
      {
      break;
      }

    switch (c)
      {
    case 0:
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
          else if( option_index == 3 ) /* pattern */
            {
            assert( strcmp(s, "pattern") == 0 );
            assert( pattern.empty() );
            pattern = optarg;
            }
           else if( option_index == 6 ) /* root uid */
            {
            assert( strcmp(s, "root-uid") == 0 );
            root = optarg;
            }
            else if( option_index == 7 ) /* resourcespath */
            {
            assert( strcmp(s, "resources-path") == 0 );
            assert( xmlpath.empty() );
            xmlpath = optarg;
            }
          else
            {
            printf (" with arg %s, index = %d", optarg, option_index);
            assert(0);
            }
          //printf (" with arg %s, index = %d", optarg, option_index);
          }
        //printf ("\n");
        }
      break;

    case 'i':
      assert( filename.empty() );
      filename = optarg;
      break;

    case 'o':
      assert( outfilename.empty() );
      outfilename = optarg;
      break;

    case 'U':
      //assert( outfilename.empty() );
      //outfilename = optarg;
      //printf ("option unenhance \n");
      unenhance = 1;
      break;

    case 'M':
      //assert( outfilename.empty() );
      //outfilename = optarg;
      mosaic = 1;
      break;

    case 'p':
      assert( pattern.empty() );
      pattern = optarg;
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

    case '?':
      break;

    default:
      printf ("?? getopt returned character code 0%o ??\n", c);
      }
  }

  // For now only support one input / one output
  if (optind < argc)
    {
    std::vector<std::string> files;
    while (optind < argc)
      {
      //printf ("%s\n", argv[optind++]);
      files.push_back( argv[optind++] );
      }
    //printf ("\n");
    if( files.size() == 2
      && filename.empty()
      && outfilename.empty()
    )
      {
      filename = files[0];
      outfilename = files[1];
      }
    else
      {
      PrintHelp();
      return 1;
      }
    }

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

  if( filename.empty() )
    {
    //std::cerr << "Need input file (-i)\n";
    PrintHelp();
    return 1;
    }
  if( outfilename.empty() )
    {
    //std::cerr << "Need output file (-o)\n";
    PrintHelp();
    return 1;
    }

  // Debug is a little too verbose
  gdcm::Trace::SetDebug( (debug  > 0 ? true : false));
  gdcm::Trace::SetWarning(  (warning  > 0 ? true : false));
  gdcm::Trace::SetError(  (error  > 0 ? true : false));
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( (verbose  > 0 ? true : false) );
    gdcm::Trace::SetError( (verbose  > 0 ? true : false) );
    }

  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( "gdcmtar" );
  if( !rootuid )
    {
    // only read the env var is no explicit cmd line option
    // maybe there is an env var defined... let's check
    const char *rootuid_env = getenv("GDCM_ROOT_UID");
    if( rootuid_env )
      {
      rootuid = 1;
      root = rootuid_env;
      }
    }
  if( rootuid )
    {
    if( !gdcm::UIDGenerator::IsValid( root.c_str() ) )
      {
      std::cerr << "specified Root UID is not valid: " << root << std::endl;
      return 1;
      }
    gdcm::UIDGenerator::SetRoot( root.c_str() );
    }

  if( unenhance && false )
    {
    gdcm::Global& g = gdcm::Global::GetInstance();
    // First thing we need to locate the XML dict
    // did the user requested to look XML file in a particular directory ?
    if( !resourcespath )
      {
      const char *xmlpathenv = getenv("GDCM_RESOURCES_PATH");
      if( xmlpathenv )
        {
        // Make sure to look for XML dict in user explicitly specified dir first:
        xmlpath = xmlpathenv;
        resourcespath = 1;
        }
      }
    if( resourcespath )
      {
      // xmlpath is set either by the cmd line option or the env var
      if( !g.Prepend( xmlpath.c_str() ) )
        {
        std::cerr << "specified Resources Path is not valid: " << xmlpath << std::endl;
        return 1;
        }
      }

    // All set, then load the XML files:
    if( !g.LoadResourcesFiles() )
      {
      return 1;
      }

    //const gdcm::Defs &defs = g.GetDefs();
    }


  if( mosaic )
    {
    gdcm::ImageReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "could not read: " << filename << std::endl;
      return 1;
      }

    gdcm::SplitMosaicFilter filter;
    filter.SetImage( reader.GetImage() );
    filter.SetFile( reader.GetFile() );
    bool b = filter.Split();
    if( !b )
      {
      std::cerr << "Could not split : " << filename << std::endl;
      return 1;
      }

    const gdcm::Image &image = filter.GetImage();
    const unsigned int *dims = image.GetDimensions();
    const gdcm::DataElement &pixeldata = image.GetDataElement();
    const gdcm::ByteValue *bv = pixeldata.GetByteValue();
    unsigned long slice_len = image.GetBufferLength() / dims[2];
    //assert( image.GetBufferLength() == bv->GetLength() );

    gdcm::FilenameGenerator fg;
    fg.SetNumberOfFilenames( dims[2] );
    fg.SetPrefix( outfilename.c_str() );
    fg.SetPattern( pattern.c_str() );
    if( !fg.Generate() )
      {
      std::cerr << "could not generate filenames" << std::endl;
      return 1;
      }
    const double *cosines = image.GetDirectionCosines();
    gdcm::DirectionCosines dc( cosines );
    double normal[3];
    dc.Cross( normal );
    const double *origin = image.GetOrigin();
    double zspacing = image.GetSpacing(2);

    for(unsigned int i = 0; i < dims[2]; ++i)
      {
      double new_origin[3];
      for (int j = 0; j < 3; j++)
        {
        // the n'th slice is n * z-spacing aloung the IOP-derived
        // z-axis
        new_origin[j] = origin[j] + normal[j] * i * zspacing;
        }

      const char *outfilenamei = fg.GetFilename(i);
      gdcm::ImageWriter writer;
      writer.SetFileName( outfilenamei );
      //writer.SetFile( filter.GetFile() );
      writer.SetFile( reader.GetFile() );

      //
      //writer.SetImage( filter.GetImage() );
      gdcm::Image &slice = writer.GetImage();
      slice = filter.GetImage();
      slice.SetOrigin( new_origin );
      slice.SetNumberOfDimensions( 2 );
      assert( slice.GetPixelFormat() == filter.GetImage().GetPixelFormat() );
      slice.SetSpacing(2, filter.GetImage().GetSpacing(2) );
      //slice.Print( std::cout );
      gdcm::DataElement &pd = slice.GetDataElement();
      const char *sliceptr = bv->GetPointer() + i * slice_len;
      pd.SetByteValue( sliceptr, (uint32_t)slice_len);

      if( !writer.Write() )
        {
        std::cerr << "Failed to write: " << outfilename << std::endl;
        return 1;
        }
      }

    return 0;
    }
  else if ( unenhance )
    {
    gdcm::ImageReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "could not read: " << filename << std::endl;
      return 1;
      }

    gdcm::File &file = reader.GetFile();
    gdcm::DataSet &ds = file.GetDataSet();
    gdcm::MediaStorage ms;
    ms.SetFromFile(file);
    if( ms.IsUndefined() )
      {
      std::cerr << "Unknown MediaStorage" << std::endl;
      return 1;
      }

    gdcm::UIDs uid;
    uid.SetFromUID( ms.GetString() );

    if( uid != gdcm::UIDs::EnhancedMRImageStorage )
      {
      std::cerr << "MediaStorage is not handled " << ms << " [" << uid.GetName() << "]" << std::endl;
      return 1;
      }

  // Preserve info:
  gdcm::DataElement oldsopclassuid = ds.GetDataElement( gdcm::Tag(0x8,0x16) );
  gdcm::DataElement oldinstanceuid = ds.GetDataElement( gdcm::Tag(0x8,0x18) );

  // Ok then change it old Old MR Image Storage
    gdcm::DataElement de( gdcm::Tag(0x0008, 0x0016) );
    ms = gdcm::MediaStorage::MRImageStorage;
    const char* msstr = gdcm::MediaStorage::GetMSString(ms);
    de.SetByteValue( msstr, (uint32_t)strlen(msstr) );
    de.SetVR( gdcm::Attribute<0x0008, 0x0016>::GetVR() );
    ds.Replace( de );

    const gdcm::Image &image = reader.GetImage();
    const unsigned int *dims = image.GetDimensions();
    //std::cout << image << std::endl;
    const gdcm::DataElement &pixeldata = image.GetDataElement();
    //const gdcm::ByteValue *bv = pixeldata.GetByteValue();
    gdcm::SmartPointer<gdcm::ByteValue> bv = (gdcm::ByteValue*)pixeldata.GetByteValue();
    unsigned long slice_len = image.GetBufferLength() / dims[2];
    assert( slice_len * dims[2] == image.GetBufferLength() );
    //assert( image.GetBufferLength() == bv->GetLength() );

    gdcm::FilenameGenerator fg;
    fg.SetNumberOfFilenames( dims[2] );
    fg.SetPrefix( outfilename.c_str() );
    fg.SetPattern( pattern.c_str() );
    if( !fg.Generate() )
      {
      std::cerr << "could not generate" << std::endl;
      return 1;
      }
    const double *cosines = image.GetDirectionCosines();
    gdcm::DirectionCosines dc( cosines );
    double normal[3];
    dc.Cross( normal );
    //const double *origin = image.GetOrigin();
    //double zspacing = image.GetSpacing(2);

    // Remove SharedFunctionalGroupsSequence
    gdcm::SmartPointer<gdcm::SequenceOfItems> sfgs =
      ds.GetDataElement( gdcm::Tag( 0x5200,0x9229 ) ).GetValueAsSQ();
    ds.Remove( gdcm::Tag( 0x5200,0x9229 ) );
    assert( ds.FindDataElement( gdcm::Tag( 0x5200,0x9229 ) ) == false );
    // Remove PerFrameFunctionalGroupsSequence
    gdcm::SmartPointer<gdcm::SequenceOfItems> pffgs =
      ds.GetDataElement( gdcm::Tag( 0x5200,0x9230 ) ).GetValueAsSQ();
    ds.Remove( gdcm::Tag( 0x5200,0x9230 ) );
    assert( ds.FindDataElement( gdcm::Tag( 0x5200,0x9230 ) ) == false );
    ds.Remove( gdcm::Tag( 0x28,0x8) );
    ds.Remove( gdcm::Tag( 0x7fe0,0x0010) );
    assert( ds.FindDataElement( gdcm::Tag( 0x7fe0,0x0010) ) == false );
    //ds.Remove( gdcm::Tag( 0x0008,0x0012) );
    //ds.Remove( gdcm::Tag( 0x0008,0x0013) );

  // reference the old instance:
  // PS 3.3-2009 C.7.6.16.1.3
#if 0
  assert( ds.FindDataElement( gdcm::Tag(0x0008,0x1150) ) == false );
  assert( ds.FindDataElement( gdcm::Tag(0x0008,0x1155) ) == false );
  assert( ds.FindDataElement( gdcm::Tag(0x0008,0x1160) ) == false );
  oldsopclassuid.SetTag( gdcm::Tag(0x8,0x1150) );
  oldinstanceuid.SetTag( gdcm::Tag(0x8,0x1155) );
  ds.Insert( oldsopclassuid );
  ds.Insert( oldinstanceuid );
#endif

  char date[22];
  const size_t datelen = 8;
  //int res = gdcm::System::GetCurrentDateTime(date);
  gdcm::Attribute<0x8,0x12> instcreationdate;
  instcreationdate.SetValue( gdcm::DTComp( date, datelen ) );
  ds.Replace( instcreationdate.GetAsDataElement() );
  gdcm::Attribute<0x8,0x13> instcreationtime;
  instcreationtime.SetValue( gdcm::DTComp( date + datelen, 13 ) );
  ds.Replace( instcreationtime.GetAsDataElement() );
  const char *offset = gdcm::System::GetTimezoneOffsetFromUTC();
  gdcm::Attribute<0x8,0x201> timezoneoffsetfromutc;
  timezoneoffsetfromutc.SetValue( offset );
  ds.Replace( timezoneoffsetfromutc.GetAsDataElement() );

    for(unsigned int i = 0; i < dims[2]; ++i)
      {
#if 0
      double new_origin[3];
      for (int j = 0; j < 3; j++)
        {
        // the n'th slice is n * z-spacing aloung the IOP-derived
        // z-axis
        new_origin[j] = origin[j] + normal[j] * i * zspacing;
        }
#endif

      const char *outfilenamei = fg.GetFilename(i);
      //gdcm::ImageWriter writer;
      gdcm::Writer writer;
      writer.SetFileName( outfilenamei );
      //writer.SetFile( filter.GetFile() );
      writer.SetFile( reader.GetFile() );

      if ( !gdcm::RemapSharedIntoOld( ds, sfgs, pffgs, i ) )
        {
        return 1;
        }

      //
      //writer.SetImage( filter.GetImage() );
      //gdcm::Image & //slice = writer.GetImage();
      //slice = reader.GetImage();
//      slice.SetOrigin( new_origin );
//      slice.SetNumberOfDimensions( 2 );
//      assert( slice.GetPixelFormat() == reader.GetImage().GetPixelFormat() );
//      slice.SetSpacing(2, reader.GetImage().GetSpacing(2) );
      //slice.Print( std::cout );
//      gdcm::DataElement &pd = slice.GetDataElement();
      const char *sliceptr = bv->GetPointer() + i * slice_len;
      gdcm::DataElement newpixeldata( gdcm::Tag(0x7fe0,0x0010) );
      newpixeldata.SetByteValue( sliceptr, (uint32_t)slice_len); // slow !
      ds.Replace( newpixeldata );

      if( !writer.Write() )
        {
        std::cerr << "Failed to write: " << outfilenamei << std::endl;
        return 1;
        }
      //else
      //  {
      //  std::cout << "Success to write: " << outfilenamei << std::endl;
      //  }
      }

    return 0;
    }
  else
    {
    assert( enhance );
    return MakeImageEnhanced( filename, outfilename );
#if 0
    std::cerr << "Not implemented" << std::endl;
    return 1;
    gdcm::ImageReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "could not read: " << filename << std::endl;
      return 1;
      }

    const gdcm::Image &image = reader.GetImage();
    const unsigned int *dims = image.GetDimensions();
    std::cout << image << std::endl;
    const gdcm::DataElement &pixeldata = image.GetDataElement();
    const gdcm::ByteValue *bv = pixeldata.GetByteValue();
    unsigned long slice_len = image.GetBufferLength() / dims[2];
    //assert( image.GetBufferLength() == bv->GetLength() );

    gdcm::FilenameGenerator fg;
    fg.SetNumberOfFilenames( dims[2] );
    fg.SetPrefix( outfilename.c_str() );
    fg.SetPattern( pattern.c_str() );
    if( !fg.Generate() )
      {
      std::cerr << "could not generate" << std::endl;
      return 1;
      }
    const double *cosines = image.GetDirectionCosines();
    gdcm::DirectionCosines dc( cosines );
    double normal[3];
    dc.Cross( normal );
    const double *origin = image.GetOrigin();
    double zspacing = image.GetSpacing(2);

    for(unsigned int i = 0; i < dims[2]; ++i)
      {
      double new_origin[3];
      for (int j = 0; j < 3; j++)
        {
        // the n'th slice is n * z-spacing aloung the IOP-derived
        // z-axis
        new_origin[j] = origin[j] + normal[j] * i * zspacing;
        }

      const char *outfilenamei = fg.GetFilename(i);
      gdcm::ImageWriter writer;
      writer.SetFileName( outfilenamei );
      //writer.SetFile( filter.GetFile() );
      writer.SetFile( reader.GetFile() );

      //
      //writer.SetImage( filter.GetImage() );
      gdcm::Image &slice = writer.GetImage();
      slice = reader.GetImage();
      slice.SetOrigin( new_origin );
      slice.SetNumberOfDimensions( 2 );
      assert( slice.GetPixelFormat() == reader.GetImage().GetPixelFormat() );
      slice.SetSpacing(2, reader.GetImage().GetSpacing(2) );
      //slice.Print( std::cout );
      gdcm::DataElement &pd = slice.GetDataElement();
      const char *sliceptr = bv->GetPointer() + i * slice_len;
      pd.SetByteValue( sliceptr, slice_len); // slow !

      if( !writer.Write() )
        {
        std::cerr << "Failed to write: " << outfilenamei << std::endl;
        return 1;
        }
      else
        {
        std::cout << "Success to write: " << outfilenamei << std::endl;
        }
      }

    return 0;
#endif
    }
}
