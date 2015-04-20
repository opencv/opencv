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
 * TODO: Merging (burnin) of overlay would be nice (merge 0x60xx overlay into PixelData)
 * TODO: --add-thumbnail / --remove-thumbnail
 * convert: -thumbnail geometry  create a thumbnail of the image
 * convert: -crop geometry       cut out a rectangular region of the image
 * -floodfill geometry color
 *                               floodfill the image with color
 * 1. Create a DICOM file from a 'raw' input:
 * 2. Create a blob (jpeg,pgm/pnm,j2k,rle) from input
 * - binary blob(s) (grayscale / RGB) input
 * - jpeg(s)
 * - j2k(s)
 *
 *   Mapping is:
 *
 *   DICOM RAW  <->  pnm/pgm
 *   DICOM jpg  <->  jpg
 *   DICOM ljpg <->  ljpg
 *   DICOM jls  <->  jls
 *   DICOM j2k  <->  j2k
 *   DICOM rle  <->  Utah RLE ??
 *
 * ??:
 *   DICOM avi  <->  avi
 *   DICOM wav  <->  wav
 *   DICOM pdf  <->  pdf
 * Todo: check compat API with jhead
 */
#include "gdcmFilename.h"
#include "gdcmImageHelper.h"
#include "gdcmDirectory.h"
#include "gdcmMediaStorage.h"
#include "gdcmSmartPointer.h"
#include "gdcmUIDGenerator.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmSystem.h"
#include "gdcmReader.h"
#include "gdcmPixmapWriter.h"
#include "gdcmPixmapReader.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmDataSet.h"
#include "gdcmAttribute.h"
#include "gdcmPNMCodec.h"
#include "gdcmPGXCodec.h"
#include "gdcmJPEGCodec.h"
#include "gdcmJPEGLSCodec.h"
#include "gdcmJPEG2000Codec.h"
#include "gdcmRLECodec.h"
#include "gdcmRAWCodec.h"
#include "gdcmVersion.h"

#include <string>
#include <iostream>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

#ifdef _MSC_VER
#define atoll _atoi64
#endif

static unsigned int readsize(const char *str, unsigned int * size)
{
  int n = sscanf( str, "%i,%i,%i", size, size+1, size+2);
  return n == EOF ? 0 : (unsigned int)n;
}

static bool readgeometry(const char *geometry, unsigned int * region)
{
  int n = sscanf( geometry, "%i,%i,%i,%i,%i,%i", region, region+1, region+2, region+3, region+4, region+5);
  if( n != 6 ) return false;
  return true;
}

template <typename T>
static void FillRegionWithColor(T *p, const unsigned int *dims, const unsigned int * region, unsigned int color, unsigned int nsamples)
{
    unsigned int xmin = region[0];
    unsigned int xmax = region[1];
    unsigned int ymin = region[2];
    unsigned int ymax = region[3];
    unsigned int zmin = region[4];
    unsigned int zmax = region[5];

    for( unsigned int x = xmin; x <= xmax; ++x)
      {
      for( unsigned int y = ymin; y <= ymax; ++y)
        {
        for( unsigned int z = zmin; z <= zmax; ++z)
          {
          for( unsigned int sample = 0; sample < nsamples; ++sample)
            {
            p[x*nsamples+y*dims[0]*nsamples+z*dims[0]*dims[1]*nsamples+sample] = (T)color;
            }
          }
        }
      }
}

static void PrintVersion()
{
  std::cout << "gdcmimg: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmimg [OPTION]... FILE..." << std::endl;
  std::cout << "Manipulate DICOM image file:" << std::endl;
  std::cout << " 1. Convert to and from other file format (jpg, jp2, pnm...)" << std::endl;
  std::cout << " 2. Anonymize burn-in annotation (rect region fill with pixel value)" << std::endl;
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input     Input filename" << std::endl;
  std::cout << "  -o --output    Output filename" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "     --endian %s       Endianness (LSB/MSB)." << std::endl;
  std::cout << "  -d --depth %d        Depth (Either 8/16/32 or BitsAllocated eg. 12 when known)." << std::endl;
  std::cout << "     --sign %s         Pixel sign (0/1)." << std::endl;
  std::cout << "     --spp  %d         Sample Per Pixel (1/3)." << std::endl;
  std::cout << "     --pc [01]         Change planar configuration." << std::endl;
  std::cout << "     --pi [str]        Change photometric interpretation." << std::endl;
  std::cout << "     --pf %d,%d,%d     Change pixel format: (BA,BS,HB)." << std::endl;
  std::cout << "  -s --size %d,%d,%d   Size." << std::endl;
  std::cout << "     --offset %ull     Start Offset." << std::endl;
  std::cout << "  -C --sop-class-uid   SOP Class UID (name or value)." << std::endl;
  std::cout << "  -T --study-uid       Study UID." << std::endl;
  std::cout << "  -S --series-uid      Series UID." << std::endl;
  std::cout << "     --root-uid        Root UID." << std::endl;
  std::cout << "Fill Options:" << std::endl;
  std::cout << "  -R --region %d,%d    Region." << std::endl;
  std::cout << "  -F --fill %d         Fill with pixel value specified." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose   more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning   print warning info." << std::endl;
  std::cout << "  -D --debug     print debug info." << std::endl;
  std::cout << "  -E --error     print error info." << std::endl;
  std::cout << "  -h --help      print help." << std::endl;
  std::cout << "  -v --version   print version." << std::endl;
  std::cout << "Env var:" << std::endl;
  std::cout << "  GDCM_ROOT_UID Root UID" << std::endl;
/*
 * Default behavior for root UID is:
 * By default the GDCM one is used
 * If GDCM_ROOT_UID is set, then use this one instead
 * If --root-uid is explicitly set on the command line, it will override any other defined behavior
 */
}

static bool AddContentDateTime(gdcm::DataSet &ds, const char *filename )
{
  time_t studydatetime = gdcm::System::FileTime( filename );
  char date[22];
  gdcm::System::FormatDateTime(date, studydatetime);
  const size_t datelen = 8;
    {
    gdcm::DataElement de( gdcm::Tag(0x0008,0x0023) ); // Content Date
    // Do not copy the whole cstring:
    de.SetByteValue( date, datelen );
    de.SetVR( gdcm::Attribute<0x0008,0x0023>::GetVR() );
    ds.Insert( de );
    }
  // StudyTime
  const size_t timelen = 6; // get rid of milliseconds
    {
    gdcm::DataElement de( gdcm::Tag(0x0008,0x0033) ); // Content Time
    // Do not copy the whole cstring:
    de.SetByteValue( date+datelen, timelen );
    de.SetVR( gdcm::Attribute<0x0008,0x0033>::GetVR() );
    ds.Insert( de );
    }
  return true;
}
// Set Study Date/Time to the file time:
static bool AddStudyDateTime(gdcm::DataSet &ds, const char *filename )
{
  // StudyDate
  char date[22];
  const size_t datelen = 8;
  int res = gdcm::System::GetCurrentDateTime(date);
  if( !res ) return false;

  {
  gdcm::DataElement de( gdcm::Tag(0x0008,0x0020) );
  // Do not copy the whole cstring:
  de.SetByteValue( date, datelen );
  de.SetVR( gdcm::Attribute<0x0008,0x0020>::GetVR() );
  ds.Insert( de );
  }

  // StudyTime
  const size_t timelen = 6; // get rid of milliseconds
  {
  gdcm::DataElement de( gdcm::Tag(0x0008,0x0030) );
  // Do not copy the whole cstring:
  de.SetByteValue( date+datelen, timelen );
  de.SetVR( gdcm::Attribute<0x0008,0x0030>::GetVR() );
  ds.Insert( de );
  }
  return AddContentDateTime(ds, filename);
}


static bool AddUIDs(int sopclassuid, std::string const & sopclass, std::string const & study_uid, std::string const & series_uid, gdcm::PixmapWriter& writer)
{
  gdcm::DataSet & ds = writer.GetFile().GetDataSet();
  gdcm::MediaStorage ms = gdcm::MediaStorage::MS_END;
  gdcm::Pixmap &image = writer.GetPixmap();
  if( sopclassuid )
    {
    // Is it by value or by name ?
    if( gdcm::UIDGenerator::IsValid( sopclass.c_str() ) )
      {
      ms = gdcm::MediaStorage::GetMSType( sopclass.c_str() );
      }
    else
      {
      std::cerr << "not implemented" << std::endl;
      }
    }
  else
    { // guess a default
    ms = gdcm::ImageHelper::ComputeMediaStorageFromModality(
      "OT", image.GetNumberOfDimensions(),
      image.GetPixelFormat(), image.GetPhotometricInterpretation() );
    }

  if( !gdcm::MediaStorage::IsImage(ms) )
    {
    std::cerr << "invalid media storage (no pixel data): " << sopclass << std::endl;
    return false;
    }
  if( ms.GetModalityDimension() < image.GetNumberOfDimensions() )
    {
    std::cerr << "Could not find Modality" << std::endl;
    return false;
    }
  const char* msstr = gdcm::MediaStorage::GetMSString(ms);
  if( !msstr )
    {
    std::cerr << "problem with media storage: " << sopclass << std::endl;
    return false;
    }
  gdcm::DataElement de( gdcm::Tag(0x0008, 0x0016 ) );
  de.SetByteValue( msstr, (uint32_t)strlen(msstr) );
  de.SetVR( gdcm::Attribute<0x0008, 0x0016>::GetVR() );
  ds.Insert( de );

    {
    gdcm::DataElement de( gdcm::Tag(0x0020,0x000d) ); // Study
    de.SetByteValue( study_uid.c_str(), (uint32_t)study_uid.size() );
    de.SetVR( gdcm::Attribute<0x0020, 0x000d>::GetVR() );
    ds.Insert( de );
    }

    {
    gdcm::DataElement de( gdcm::Tag(0x0020,0x000e) ); // Series
    de.SetByteValue( series_uid.c_str(), (uint32_t)series_uid.size() );
    de.SetVR( gdcm::Attribute<0x0020, 0x000e>::GetVR() );
    ds.Insert( de );
    }

  return true;
}

// Append data to either sq or bv depending whether encapsulated or not
static bool PopulateSingeFile( gdcm::PixmapWriter & writer,
  gdcm::SequenceOfFragments *sq , gdcm::ByteValue * bv, gdcm::ImageCodec & jpeg,
  const char *filename, std::streampos const pos = 0 )
{
  /*
   * FIXME: when JPEG contains JFIF marker, we should only read them
   * during header parsing but discard them when copying the JPG byte stream into
   * the encapsulated Pixel Data Element...
   */
  std::ifstream is(filename, std::ios::binary);
  gdcm::TransferSyntax ts;
  bool b = jpeg.GetHeaderInfo( is, ts );
  if( !b )
    {
    std::cerr << "Could not read: " << filename << std::endl;
    return false;
    }

  gdcm::Pixmap &image = writer.GetPixmap();
  image.SetDimensions( jpeg.GetDimensions() );
  image.SetPixelFormat( jpeg.GetPixelFormat() );
  image.SetPhotometricInterpretation( jpeg.GetPhotometricInterpretation() );
  image.SetPlanarConfiguration( jpeg.GetPlanarConfiguration() );
  image.SetTransferSyntax( ts );

  AddStudyDateTime( writer.GetFile().GetDataSet(), filename );

  size_t len = gdcm::System::FileSize(filename);
  if( ts.IsEncapsulated() )
    {
    is.seekg(0, std::ios::beg );// rewind !
    }
  else
    {
    len = image.GetBufferLength();
    // do not rewind file should be just at right offset
    }
  char *buf = new char[len];
  if( pos )
    {
    is.seekg( pos, std::ios::beg );
    }
  is.read(buf, len);
  gdcm::DataElement pixeldata( gdcm::Tag(0x7fe0,0x0010) );

  if( ts.IsEncapsulated() )
    {
    gdcm::Fragment frag;
    frag.SetByteValue( buf, (uint32_t)len );
    sq->AddFragment( frag );
    pixeldata.SetValue( *sq );
    }
  else
    {
    gdcm::ByteValue frame( buf, (uint32_t)len );
    bv->Append( frame );
    pixeldata.SetValue( *bv );
    }
  delete[] buf;
  image.SetDataElement( pixeldata );

  return true;
}

static bool Populate( gdcm::PixmapWriter & writer, gdcm::ImageCodec & jpeg, gdcm::Directory::FilenamesType const & filenames, unsigned int ndim = 2, std::streampos const & pos = 0 )
{
  std::vector<std::string>::const_iterator it = filenames.begin();
  bool b = true;
  gdcm::Pixmap &image = writer.GetPixmap();
  image.SetNumberOfDimensions( ndim );

  gdcm::SmartPointer<gdcm::SequenceOfFragments> sq = new gdcm::SequenceOfFragments;
  gdcm::SmartPointer<gdcm::ByteValue> bv = new gdcm::ByteValue;
  for(; it != filenames.end(); ++it)
    {
    b = b && PopulateSingeFile( writer, sq, bv, jpeg, it->c_str(), pos );
    }
  if( filenames.size() > 1 )
    {
    image.SetNumberOfDimensions( 3 );
    image.SetDimension(2,  (unsigned int)filenames.size() );
    }

  return b;
}


static bool GetPixelFormat( gdcm::PixelFormat & pf, int depth, int bpp, int sign, int pixelsign, int spp = 0, int pixelspp = 1 )
{
  if( depth )
    {
    if( bpp <= 8 )
      {
      pf = gdcm::PixelFormat::UINT8;
      }
    else if( bpp > 8 && bpp <= 16 )
      {
      pf = gdcm::PixelFormat::UINT16;
      }
    else if( bpp > 16 && bpp <= 32 )
      {
      pf = gdcm::PixelFormat::UINT32;
      }
    else
      {
      std::cerr << "Invalid depth: << " << bpp << std::endl;
      return false;
      }
    pf.SetBitsStored( (short)bpp );
    }
  if( sign )
    {
    pf.SetPixelRepresentation( (unsigned short)pixelsign );
    }
  if( spp )
    {
    pf.SetSamplesPerPixel( (unsigned short)pixelspp );
    }

  return true;
}

int main (int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;

  std::string root;
  int rootuid = 0;
  gdcm::Filename filename;
  gdcm::Directory::FilenamesType filenames;
  gdcm::Filename outfilename;
  unsigned int region[6] = {}; // Rows & Columns are VR=US anyway...
  unsigned int color = 0;
  int bregion = 0;
  int fill = 0;
  int sign = 0;
  int spp = 0;
  int pconf = 0; // planar configuration
  int studyuid = 0;
  int seriesuid = 0;
  unsigned int size[3] = {0,0,0};
  unsigned int ndimension = 2;
  int depth = 0;
  int endian = 0;
  int bpp = 0;
  int pixelsign = 0;
  int pixelspp = 0;
  std::string sopclass;
  std::string lsb_msb;
  int sopclassuid = 0;
  int pinter = 0;
  std::string pinterstr;
  int pformat = 0;
  std::string pformatstr;
  int poffset = 0;
  size_t start_pos = 0;

  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;

  gdcm::UIDGenerator uid;
  // Too early for UID Generation
  std::string series_uid; // = uid.Generate();
  std::string study_uid; // = uid.Generate();
  while (1) {
    //int this_option_optind = optind ? optind : 1;
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"output", 1, 0, 0},
        // provide convert-like command line args:
        {"depth", 1, &depth, 1},
        {"size", 1, 0, 0},
        {"region", 1, &bregion, 1},
        {"fill", 1, &fill, 1},
        {"study-uid", 1, &studyuid, 1},
        {"series-uid", 1, &seriesuid, 1},
        {"root-uid", 1, &rootuid, 1}, // specific Root (not GDCM)
        {"sop-class-uid", 1, &sopclassuid, 1}, // specific SOP Class UID
        {"endian", 1, &endian, 1}, //
        {"sign", 1, &sign, 1}, //
        {"spp", 1, &spp, 1}, //
        {"pc", 1, &pconf, 1}, //
        {"pi", 1, &pinter, 1}, //
        {"pf", 1, &pformat, 1}, //
        {"offset", 1, &poffset, 1}, //

// General options !
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {0, 0, 0, 0}
    };

    // i -> input file
    // I -> input directory
    // o -> output file
    // O -> output directory
    c = getopt_long (argc, argv, "i:o:I:O:d:s:R:C:F:T:S:VWDEhv",
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
            assert( filename.IsEmpty() );
            filename = optarg;
            }
          else if( option_index == 2 ) /* depth */
            {
            assert( strcmp(s, "depth") == 0 );
            bpp = atoi(optarg);
            }
          else if( option_index == 3 ) /* size */
            {
            assert( strcmp(s, "size") == 0 );
            ndimension = readsize(optarg, size);
            }
          else if( option_index == 4 ) /* region */
            {
            assert( strcmp(s, "region") == 0 );
            readgeometry(optarg, region);
            }
          else if( option_index == 5 ) /* fill */
            {
            assert( strcmp(s, "fill") == 0 );
            color = atoi(optarg);
            }
          else if( option_index == 6 ) /* study-uid */
            {
            assert( strcmp(s, "study-uid") == 0 );
            study_uid = optarg;
            }
          else if( option_index == 7 ) /* series-uid */
            {
            assert( strcmp(s, "series-uid") == 0 );
            series_uid = optarg;
            }
          else if( option_index == 8 ) /* root-uid */
            {
            assert( strcmp(s, "root-uid") == 0 );
            root = optarg;
            }
          else if( option_index == 9 ) /* sop-class-uid */
            {
            assert( strcmp(s, "sop-class-uid") == 0 );
            sopclass = optarg;
            }
          else if( option_index == 10 ) /* endian */
            {
            assert( strcmp(s, "endian") == 0 );
            lsb_msb = optarg;
            }
          else if( option_index == 11 ) /* sign */
            {
            assert( strcmp(s, "sign") == 0 );
            pixelsign = atoi(optarg);
            }
          else if( option_index == 12 ) /* spp */
            {
            assert( strcmp(s, "spp") == 0 );
            pixelspp = atoi(optarg);
            }
          else if( option_index == 13 ) /* pconf */
            {
            assert( strcmp(s, "pc") == 0 );
            pconf = atoi(optarg);
            }
          else if( option_index == 14 ) /* pinter */
            {
            assert( strcmp(s, "pi") == 0 );
            pinter = 1;
            pinterstr = optarg;
            }
          else if( option_index == 15 ) /* pformat */
            {
            assert( strcmp(s, "pf") == 0 );
            pformat = 1;
            pformatstr = optarg;
            }
          else if( option_index == 16 ) /* start_pos */
            {
            assert( strcmp(s, "offset") == 0 );
            poffset = 1;
            start_pos = (size_t)atoll(optarg);
            }
          //printf (" with arg %s", optarg);
          }
        //printf ("\n");
        }
      break;

    case 'i':
      //printf ("option i with value '%s'\n", optarg);
      assert( filename.IsEmpty() );
      filename = optarg;
      break;

    case 'o':
      //printf ("option o with value '%s'\n", optarg);
      assert( outfilename.IsEmpty() );
      outfilename = optarg;
      break;

    case 'd': // depth
      bpp = atoi(optarg);
      depth = 1;
      break;

    case 's': // size
      ndimension = readsize(optarg, size);
      break;

    case 'T':
      studyuid = 1;
      study_uid = optarg;
      break;

    case 'S':
      seriesuid = 1;
      series_uid = optarg;
      break;

    case 'C':
      sopclassuid = 1;
      sopclass = optarg;
      break;

    case 'R': // region
      //outfilename = optarg;
      readgeometry(optarg, region);
      break;

    case 'F': // fill
      color = atoi( optarg );
      fill = 1;
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
    //printf ("non-option ARGV-elements: ");
    std::vector<std::string> files;
    while (optind < argc)
      {
      //printf ("%s\n", argv[optind++]);
      files.push_back( argv[optind++] );
      }
    //printf ("\n");
    if( files.size() >= 2
      && filename.IsEmpty()
      && outfilename.IsEmpty()
    )
      {
      filename = files[0].c_str();
      filenames = files;
      outfilename = files[ files.size() - 1 ].c_str();
      filenames.pop_back();
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

  if( filenames.empty() && filename.IsEmpty() )
    {
    //std::cerr << "Need input file (-i)\n";
    PrintHelp();
    return 1;
    }
  if( outfilename.IsEmpty() )
    {
    //std::cerr << "Need output file (-o)\n";
    PrintHelp();
    return 1;
    }

  // Ok so we are about to write a DICOM file, do not forget to stamp it GDCM !
  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( "gdcmimg" );
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
  if( study_uid.empty() )
    {
    study_uid = uid.Generate();
    }
  if( !gdcm::UIDGenerator::IsValid( study_uid.c_str() ) )
    {
    std::cerr << "Invalid UID for Study UID: " << study_uid << std::endl;
    return 1;
    }

  if( series_uid.empty() )
    {
    series_uid  = uid.Generate();
    }
  if( !gdcm::UIDGenerator::IsValid( series_uid.c_str() ) )
    {
    std::cerr << "Invalid UID for Series UID: " << series_uid << std::endl;
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

  if( depth )
    {
    if( bpp < 1 || bpp > 32 )
      {
      std::cerr << "Invalid depth for pixel: " << bpp << std::endl;
      return 1;
      }
    }
  if( sign )
    {
    if( pixelsign != 0 && pixelsign != 1 ) return 1;
    }
  if( spp )
    {
    if( pixelspp != 1 && pixelspp != 3 ) return 1;
    }
  if( pconf != 0 && pconf != 1 ) return 1;
  if( pconf )
    {
    if( pixelspp != 3 ) return 1;
    }
  gdcm::PixelFormat pfref = gdcm::PixelFormat::UINT8;
  if( pformat )
    {
    int ba, bs, hb;
    int n = sscanf( pformatstr.c_str(), "%d,%d,%d", &ba, &bs, &hb );
    if( n != 3 ) return 1;
    pfref.SetBitsAllocated( (unsigned short)ba );
    pfref.SetBitsStored( (unsigned short)bs );
    pfref.SetHighBit( (unsigned short)hb );
    if( spp )
      pfref.SetSamplesPerPixel( (unsigned short)pixelspp );
    if( sign )
      pfref.SetPixelRepresentation( (unsigned short)pixelsign );
    }
  gdcm::PhotometricInterpretation::PIType refpi = gdcm::PhotometricInterpretation::MONOCHROME2;
  if( pinter )
    {
    refpi = gdcm::PhotometricInterpretation::GetPIType( pinterstr.c_str() );
    if( refpi == gdcm::PhotometricInterpretation::UNKNOW
      || refpi == gdcm::PhotometricInterpretation::PI_END )
      {
      std::cerr << "Invalid PI: " << pinterstr << std::endl;
      return 1;
      }
    }

  const char *inputextension = filename.GetExtension();
  const char *outputextension = outfilename.GetExtension();
  //if( !inputextension || !outputextension ) return 1;
  if( inputextension )
    {
    if(  gdcm::System::StrCaseCmp(inputextension,".raw") == 0   // watch out that .raw for kakadu means big-endian
      || gdcm::System::StrCaseCmp(inputextension,".rawl") == 0  // kakadu convention for raw little endian
      || gdcm::System::StrCaseCmp(inputextension,".gray") == 0  // imagemagick convention
      || gdcm::System::StrCaseCmp(inputextension,".bin") == 0   // openjp3d convention for raw little endian
      || gdcm::System::StrCaseCmp(inputextension,".rgb") == 0 ) // imagemagick convention
      {
      if( !size[0] || !size[1] )
        {
        std::cerr << "need to specify size of image stored in RAW file" << std::endl;
        return 1;
        }
      gdcm::RAWCodec raw;
      gdcm::PixmapWriter writer;
      // Because the RAW stream is not self sufficient, we need to pass in some extra
      // user info:
      unsigned int dims[3] = {};
      dims[0] = size[0];
      dims[1] = size[1];
      if( ndimension == 3 )
        {
        dims[2] = size[2];
        }
      raw.SetDimensions( dims );
      gdcm::PixelFormat pf = gdcm::PixelFormat::UINT8;
      gdcm::PhotometricInterpretation pi = refpi;
      if( gdcm::System::StrCaseCmp(inputextension,".rgb") == 0 )
        {
        pi = gdcm::PhotometricInterpretation::RGB;
        spp = 1;
        pixelspp = 3;
        }
      if( !GetPixelFormat( pf, depth, bpp, sign, pixelsign, spp, pixelspp ) ) return 1;
      raw.SetPixelFormat( pf );
      if( spp )
        {
        if( pixelspp == 3 ) pi = gdcm::PhotometricInterpretation::RGB;
        }
      raw.SetPhotometricInterpretation( pi );
      raw.SetNeedByteSwap( false );
      raw.SetPlanarConfiguration( pconf );
      if( endian )
        {
        if( lsb_msb == "LSB" || lsb_msb == "MSB" )
          {
          if( lsb_msb == "MSB" )
            {
            raw.SetNeedByteSwap( true );
            }
          }
        else
          {
          std::cerr << "Unrecognized endian: " << lsb_msb << std::endl;
          return 1;
          }
        }

      if( !Populate( writer, raw, filenames, ndimension, start_pos ) ) return 1;
      if( !AddUIDs(sopclassuid, sopclass, study_uid, series_uid, writer ) ) return 1;

      writer.SetFileName( outfilename );
      if( !writer.Write() )
        {
        std::cerr << "Failed to write: " << outfilename << std::endl;
        return 1;
        }

      return 0;
      }

    if(  gdcm::System::StrCaseCmp(inputextension,".rle") == 0 )
      {
      if( !size[0] || !size[1] )
        {
        std::cerr << "need to specify size of image stored in RLE file" << std::endl;
        return 1;
        }
      gdcm::RLECodec rle;
      gdcm::PixmapWriter writer;
      // Because the RLE stream is not self sufficient, we need to pass in some extra
      // user info:
      unsigned int dims[3] = {};
      dims[0] = size[0];
      dims[1] = size[1];
      rle.SetDimensions( dims );
      gdcm::PixelFormat pf = gdcm::PixelFormat::UINT8;
      if( !GetPixelFormat( pf, depth, bpp, sign, pixelsign, spp, pixelspp ) ) return 1;
      rle.SetPixelFormat( pf );
      gdcm::PhotometricInterpretation pi = refpi;
      if( spp )
        {
        if( pixelspp == 3 ) pi = gdcm::PhotometricInterpretation::RGB;
        }
      rle.SetPhotometricInterpretation( pi );

      if( !Populate( writer, rle, filenames ) ) return 1;
      if( !AddUIDs(sopclassuid, sopclass, study_uid, series_uid, writer ) ) return 1;

      writer.SetFileName( outfilename );
      if( !writer.Write() )
        {
        return 1;
        }

      return 0;
      }

    if(  gdcm::System::StrCaseCmp(inputextension,".pgm") == 0
      || gdcm::System::StrCaseCmp(inputextension,".pnm") == 0
      || gdcm::System::StrCaseCmp(inputextension,".ppm") == 0 )
      {
      gdcm::PNMCodec pnm;
      // Let's handle the case where user really wants to specify the data:
      gdcm::PixelFormat pf = gdcm::PixelFormat::UINT8;
      if( !GetPixelFormat( pf, depth, bpp, sign, pixelsign ) ) return 1;
      pnm.SetPixelFormat( pf );
      gdcm::PixmapWriter writer;
      if( !Populate( writer, pnm, filenames ) ) return 1;
      // populate will guess pixel format and photometric inter from file, need
      // to override after calling Populate:
      if( pformat )
        {
        writer.GetPixmap().SetPixelFormat( pfref );
        }
      if( pinter )
        {
        writer.GetPixmap().SetPhotometricInterpretation( refpi );
        }
      // HACK
      if( endian && lsb_msb == "LSB" )
        writer.GetPixmap().SetTransferSyntax( gdcm::TransferSyntax::ImplicitVRLittleEndian );

      if( !AddUIDs(sopclassuid, sopclass, study_uid, series_uid, writer ) ) return 1;

      writer.SetFileName( outfilename );
      if( !writer.Write() )
        {
        return 1;
        }

      return 0;
      }

    if(  gdcm::System::StrCaseCmp(inputextension,".pgx") == 0 )
      {
      gdcm::PGXCodec pnm;
      gdcm::PixmapWriter writer;
      if( !Populate( writer, pnm, filenames ) ) return 1;
      if( !AddUIDs(sopclassuid, sopclass, study_uid, series_uid, writer ) ) return 1;

      writer.SetFileName( outfilename );
      if( !writer.Write() )
        {
        return 1;
        }

      return 0;
      }

    if(  gdcm::System::StrCaseCmp(inputextension,".jls") == 0 )
      {
      gdcm::JPEGLSCodec jpeg;
      gdcm::PixmapWriter writer;
      if( !Populate( writer, jpeg, filenames ) ) return 1;
      if( !AddUIDs(sopclassuid, sopclass, study_uid, series_uid, writer ) ) return 1;

      writer.SetFileName( outfilename );
      if( !writer.Write() )
        {
        return 1;
        }

      return 0;
      }

    if(  gdcm::System::StrCaseCmp(inputextension,".jp2") == 0
      || gdcm::System::StrCaseCmp(inputextension,".j2k") == 0
      || gdcm::System::StrCaseCmp(inputextension,".j2c") == 0
      || gdcm::System::StrCaseCmp(inputextension,".jpx") == 0
      || gdcm::System::StrCaseCmp(inputextension,".jpc") == 0 )
      {
      /*
       * FIXME: Same problem as in classic JPEG: JP2 is NOT a J2K byte stream
       * need to chop off all extra header information...
       */
      gdcm::JPEG2000Codec jpeg;
      gdcm::PixmapWriter writer;
      if( !Populate( writer, jpeg, filenames ) ) return 1;
      if( !AddUIDs(sopclassuid, sopclass, study_uid, series_uid, writer ) ) return 1;

      writer.SetFileName( outfilename );
      if( !writer.Write() )
        {
        return 1;
        }

      return 0;

      }

    if(  gdcm::System::StrCaseCmp(inputextension,".jpg") == 0
      || gdcm::System::StrCaseCmp(inputextension,".jpeg") == 0
      || gdcm::System::StrCaseCmp(inputextension,".ljpg") == 0
      || gdcm::System::StrCaseCmp(inputextension,".ljpeg") == 0 )
      {
      gdcm::JPEGCodec jpeg;
      // Let's handle the case where user really wants to specify signess of data:
      gdcm::PixelFormat pf = gdcm::PixelFormat::UINT8;
      if( !GetPixelFormat( pf, depth, bpp, sign, pixelsign ) ) return 1;
      jpeg.SetPixelFormat( pf );
      gdcm::PixmapWriter writer;
      if( !Populate( writer, jpeg, filenames ) ) return 1;
      if( !AddUIDs(sopclassuid, sopclass, study_uid, series_uid, writer ) ) return 1;

      writer.SetFileName( outfilename );
      if( !writer.Write() )
        {
        std::cerr << "Problem during DICOM steps" << std::endl;
        return 1;
        }

      return 0;
      }
    }
// else safely assume that if no inputextension matched then it is a DICOM file

  gdcm::PixmapReader reader;
  reader.SetFileName( filename );
  if( !reader.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }

  const gdcm::Pixmap &imageori = reader.GetPixmap();
  const gdcm::File &file = reader.GetFile();

  if ( outputextension )
    {
    if(  gdcm::System::StrCaseCmp(outputextension,".pgm") == 0
      || gdcm::System::StrCaseCmp(outputextension,".pnm") == 0
      || gdcm::System::StrCaseCmp(outputextension,".ppm") == 0 )
      {
      gdcm::PNMCodec pnm;
      pnm.SetDimensions( imageori.GetDimensions() );
      pnm.SetPixelFormat( imageori.GetPixelFormat() );
      pnm.SetPhotometricInterpretation( imageori.GetPhotometricInterpretation() );
      pnm.SetPlanarConfiguration( imageori.GetPlanarConfiguration() );
      pnm.SetLUT( imageori.GetLUT() );
      const gdcm::DataElement& in = imageori.GetDataElement();
      bool b = pnm.Write( outfilename, in );
      if( !b )
        {
        std::cerr << "Problem writing PNM file" << std::endl;
        return 1;
        }

      return 0;
      }
    if(  gdcm::System::StrCaseCmp(outputextension,".pgx") == 0 )
      {
      gdcm::PGXCodec pnm;
      pnm.SetDimensions( imageori.GetDimensions() );
      pnm.SetPixelFormat( imageori.GetPixelFormat() );
      pnm.SetPhotometricInterpretation( imageori.GetPhotometricInterpretation() );
      pnm.SetPlanarConfiguration( imageori.GetPlanarConfiguration() );
      pnm.SetLUT( imageori.GetLUT() );
      const gdcm::DataElement& in = imageori.GetDataElement();
      bool b = pnm.Write( outfilename, in );
      if( !b )
        {
        std::cerr << "Problem writing PNM file" << std::endl;
        return 1;
        }

      return 0;
      }
    }

// else safely assume that if no outputextension matched then it is a DICOM file

  gdcm::PixmapWriter writer;
  writer.SetFile( file );
  writer.SetImage( imageori );
  writer.SetFileName( outfilename );


  gdcm::DataSet &ds = writer.GetFile().GetDataSet();
  if( fill )
    {
    const gdcm::PixelFormat &pixeltype = imageori.GetPixelFormat();
    assert( imageori.GetNumberOfDimensions() == 2 || imageori.GetNumberOfDimensions() == 3 );
    unsigned long len = imageori.GetBufferLength();
    gdcm::SmartPointer<gdcm::Pixmap> image = new gdcm::Pixmap;
    image->SetNumberOfDimensions( 2 ); // good default
    const unsigned int *dims = imageori.GetDimensions();
    if ( region[0] > region[1]
      || region[2] > region[3]
      || region[4] > region[5]
      || region[1] > dims[0]
      || region[3] > dims[1]
      || (imageori.GetNumberOfDimensions() > 2 && region[5] > dims[2]) )
      {
      if( imageori.GetNumberOfDimensions() == 2 )
        {
        std::cerr << "bogus region. Should be at most: (" << dims[0] << "," << dims[1] << ","
          /*<< dims[2]*/ << ")" << std::endl;
        }
      else
        {
        std::cerr << "bogus region. Should be at most: (" << dims[0] << "," << dims[1] << ","
          << dims[2] << ")" << std::endl;
        }
      return 1;
      }
    image->SetDimension(0, dims[0] );
    image->SetDimension(1, dims[1] );
    if( imageori.GetNumberOfDimensions() == 3 )
      {
      image->SetNumberOfDimensions( 3 );
      image->SetDimension(2, dims[2] );
      }
    image->SetPhotometricInterpretation( imageori.GetPhotometricInterpretation() );
    image->SetPixelFormat( imageori.GetPixelFormat() );
    image->SetPlanarConfiguration( imageori.GetPlanarConfiguration() );
    image->SetLUT( imageori.GetLUT() );
    image->SetLossyFlag( imageori.IsLossy() );
    // FIXME what is overlay is in pixel data ?
    gdcm::DataElement pixeldata( gdcm::Tag(0x7fe0,0x0010) );
    gdcm::ByteValue *bv = new gdcm::ByteValue();
    bv->SetLength( (uint32_t)len );
    //memcpy( bv->GetPointer(), imageori
    imageori.GetBuffer( (char*)bv->GetPointer() );
    // Rub out pixels:
    char *p = (char*)bv->GetPointer();
    switch(pixeltype)
      {
    case gdcm::PixelFormat::UINT8:
      FillRegionWithColor<uint8_t> ((uint8_t*)p, dims, region, color, pixeltype.GetSamplesPerPixel());
      break;
    case gdcm::PixelFormat::INT8:
      FillRegionWithColor<int8_t>  ((int8_t*)p, dims, region, color, pixeltype.GetSamplesPerPixel());
      break;
    case gdcm::PixelFormat::UINT16:
      FillRegionWithColor<uint16_t>((uint16_t*)p, dims, region, color, pixeltype.GetSamplesPerPixel());
      break;
    case gdcm::PixelFormat::INT16:
      FillRegionWithColor<int16_t> ((int16_t*)p, dims, region, color, pixeltype.GetSamplesPerPixel());
      break;
    default:
      std::cerr << "not implemented" << std::endl;
      return 1;
      }

    pixeldata.SetValue( *bv );
    image->SetDataElement( pixeldata );
    const gdcm::TransferSyntax &ts = imageori.GetTransferSyntax();
    // FIXME: for now we do not know how to recompress the image...
    if( ts.IsExplicit() )
      {
      image->SetTransferSyntax( gdcm::TransferSyntax::ExplicitVRLittleEndian );
      }
    else
      {
      assert( ts.IsImplicit() );
      image->SetTransferSyntax( gdcm::TransferSyntax::ImplicitVRLittleEndian );
      }
    //imageori.Print( std::cout );
    //image.Print( std::cout );

    // Set our filled image instead:
    writer.SetImage( *image );
#if 0
    // <entry group="0028" element="0301" vr="CS" vm="1" name="Burned In Annotation"/>
    gdcm::Attribute<0x0028,0x0301> at;
    at.SetValue( "NO" ); // 'YES'
    ds.Replace( at.GetAsDataElement() );
    // (0008,2111) ST [MedCom Resample v]                      #  18, 1 DerivationDescriptio
    gdcm::Attribute<0x0008,0x2111> at2;
    std::ostringstream os;
    os << "Fill Region ["
      << region[0] << "," << region[1] << ","
      << region[2] << "," << region[3] << ","
      << region[4] << "," << region[5] << "] with color value=" << std::hex << (int)color;
    at2.SetValue( os.str() );
    ds.Replace( at2.GetAsDataElement() );
#else
#endif
/*
> 1. Replace Value #1 of Image Type by 'DERIVED'

Don't do that ... leave Image Type alone (unless you are changing
the UID ... vide infra).
*/
#if 0
    // (0008,0008) CS [ORIGINAL\SECONDARY]                     #  18, 2 ImageType
    gdcm::Attribute<0x0008,0x0008> at3;
    static const gdcm::CSComp values[] = {"DERIVED","SECONDARY"};
    at3.SetValues( values, 2, true ); // true => copy data !
    if( ds.FindDataElement( at3.GetTag() ) )
      {
      const gdcm::DataElement &de = ds.GetDataElement( at3.GetTag() );
      at3.SetFromDataElement( de );
      // Make sure that value #1 is at least 'DERIVED', so override in all cases:
      at3.SetValue( 0, values[0] );
      }
    ds.Replace( at3.GetAsDataElement() );
#endif
    // Make sure to recompute Planar Configuration:
    ds.Remove( gdcm::Tag(0x0028, 0x0004) );
    }
  //  ds.Remove( gdcm::Tag(0x0,0x0) ); // FIXME

  if( !writer.Write() )
    {
    std::cerr << "Failed to write: " << outfilename << std::endl;
    return 1;
    }

  return 0;
}
