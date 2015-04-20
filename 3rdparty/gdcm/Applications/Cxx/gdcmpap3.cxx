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
 * Command line tool to deal with legacy PAPYRUS 3.0 file
 * The command line tool can be compiled in two flavour:
 *
 * 1. without papyrus 3.0 (more portable)
 * 2. with papyrus 3.0
 *
 * The (2) is only required when dealing with invalid JPEG Lossless compressed
 * PAPYRUS 3.0 files
 */
#include "gdcmReader.h"
#include "gdcmDirectionCosines.h"
#include "gdcmUIDGenerator.h"
#include "gdcmVersion.h"
#include "gdcmWriter.h"
#include "gdcmAttribute.h"
#include "gdcmTrace.h"
#include "gdcmImageHelper.h"
#include "gdcmSequenceOfItems.h"

#include <getopt.h>

#ifdef GDCM_USE_SYSTEM_PAPYRUS3
extern "C" {
#include <Papyrus3.h>
}
#endif

static void PrintVersion()
{
  std::cout << "gdcmpap3: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmpap3 [OPTION] input.pa3 output.dcm" << std::endl;
  std::cout << "Convert a PAPYRUS 3.0 file into another DICOM file.\n";
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input      PAPYRUS 3.0 filename" << std::endl;
  std::cout << "  -o --output     DICOM filename" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -S --split           Split single PAPYRUS 3.0 file into multiples DICOM files." << std::endl;
  std::cout << "     --decomp-pap3     Use PAPYRUS 3.0 for decompressing (can be combined with --split)." << std::endl;
  std::cout << "     --check-iop       Check that the Image Orientation (Patient) Attribute is ok (see --split)." << std::endl;
  std::cout << "     --root-uid        Specify Root UID." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose    more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning    print warning info." << std::endl;
  std::cout << "  -D --debug      print debug info." << std::endl;
  std::cout << "  -E --error      print error info." << std::endl;
  std::cout << "  -h --help       print help." << std::endl;
  std::cout << "  -v --version    print version." << std::endl;
  std::cout << "Env var:" << std::endl;
  std::cout << "  GDCM_ROOT_UID Root UID" << std::endl;
}

static bool InitPapyrus3( const char *filename, int & outfileNb)
{
  outfileNb = -1;
#ifdef GDCM_USE_SYSTEM_PAPYRUS3
  /* initialisation of the Papyrus toolkit v3.6 */
  Papy3Init();

  /* open the pap3 file */
  PapyShort fileNb = Papy3FileOpen ((char*)filename, (PAPY_FILE) 0, TRUE, 0);
  if( fileNb < 0 )
    {
    PAPY3PRINTERRMSG();
    return false;
    }
  outfileNb = fileNb;
  return true;
#else
  (void)filename;
  (void)outfileNb;
  std::cerr << "No PAPYRUS 3.0 library found" << std::endl;
  return false;
#endif
}

static bool DecompressPapyrus3( int pap3handle, int itemnum, gdcm::TransferSyntax const & ts, gdcm::File & file )
{
#ifdef GDCM_USE_SYSTEM_PAPYRUS3
  PapyShort fileNb = (PapyShort)pap3handle;
  PapyShort imageNb = (PapyShort)(itemnum + 1);

  if( ts == gdcm::TransferSyntax::JPEGLosslessProcess14_1 )
    {
    SElement *group;
    PapyUShort *theImage;

    std::vector<unsigned int> dims = gdcm::ImageHelper::GetDimensionsValue(file);
    gdcm::PixelFormat pf = gdcm::ImageHelper::GetPixelFormatValue(file);

    gdcm::DataSet & nested = file.GetDataSet();

    /* position the file pointer to the begining of the data set */
    PapyShort err = Papy3GotoNumber (fileNb, (PapyShort)imageNb, DataSetID);

    gdcm::DataElement pixeldata( gdcm::Tag(0x7fe0,0x0010) );

    /* then goto group 0x7FE0 */
    if ((err = Papy3GotoGroupNb (fileNb, 0x7FE0)) == 0)
      {
      /* read group 0x7FE0 from the file */
      if ((err = Papy3GroupRead (fileNb, &group)) > 0)
        {
        /* PIXEL DATA */
        theImage = (PapyUShort *)Papy3GetPixelData (fileNb, imageNb, group, ImagePixel);

        //assert( dims[0] == 512 );
        //assert( dims[1] == 512 );
        //assert( pf.GetPixelSize() == 2 );
        const size_t imglen = dims[0] * dims[1] * pf.GetPixelSize();
        pixeldata.SetByteValue( (char*)theImage, (uint32_t)imglen );

        /* free group 7FE0 */
        err = Papy3GroupFree (&group, TRUE);
        } /* endif ...group 7FE0 read */
      else
        {
        PAPY3PRINTERRMSG ();
        }
      } /* endif ...group 7FE0 found */
    else
      {
      PAPY3PRINTERRMSG ();
      }
    nested.Replace( pixeldata );
    }
  else
    {
    std::cerr << "TransferSyntax: " << ts << " is not handled at this point" << std::endl;
    return false;
    }
  return true;
#else
  (void)pap3handle; (void)itemnum; (void)ts; (void)file;
  std::cerr << "No PAPYRUS 3.0 library found" << std::endl;
  return false;
#endif
}

static bool CleanupPapyrus3( int pap3handle )
{
#ifdef GDCM_USE_SYSTEM_PAPYRUS3
  PapyShort fileNb = (PapyShort)pap3handle;
  /* close and free the file and the associated allocated memory */
  Papy3FileClose (fileNb, TRUE);

  /* free the allocated global value in the toolkit */
  Papy3FreeDataSetModules ();

  return true;
#else
  (void)pap3handle;
  return false;
#endif
}

int main(int argc, char *argv[])
{
  int c;

  std::string filename;
  std::string outfilename;
  std::string root;
  int rootuid = 0;
  int split = 0;
  int decomp_pap3 = 0;
  int check_iop = 0;

  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;

  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"output", 1, 0, 0},
        {"root-uid", 1, &rootuid, 1}, // specific Root (not GDCM)
        {"split", 0, &split, 1},
        {"decomp-pap3", 0, &decomp_pap3, 1},
        {"check-iop", 0, &check_iop, 1},

// General options !
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},

        {0, 0, 0, 0}
    };

    c = getopt_long (argc, argv, "i:o:S:VWDEhv",
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
          else if( option_index == 1 ) /* output */
            {
            assert( strcmp(s, "output") == 0 );
            assert( outfilename.empty() );
            outfilename = optarg;
            }
          else if( option_index == 2 ) /* root-uid */
            {
            assert( strcmp(s, "root-uid") == 0 );
            root = optarg;
            }
          //printf (" with arg %s, index = %d", optarg, option_index);
          }
        //printf ("\n");
        }
      break;

    case 'i':
      //printf ("option i with value '%s'\n", optarg);
      assert( filename.empty() );
      filename = optarg;
      break;

    case 'o':
      //printf ("option o with value '%s'\n", optarg);
      assert( outfilename.empty() );
      outfilename = optarg;
      break;

    //
    case 'S':
      split = 1;
      break;

    // General option
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

  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( "gdcmpap3" );
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
    // root is set either by the cmd line option or the env var
    if( !gdcm::UIDGenerator::IsValid( root.c_str() ) )
      {
      std::cerr << "specified Root UID is not valid: " << root << std::endl;
      return 1;
      }
    gdcm::UIDGenerator::SetRoot( root.c_str() );
    }

  gdcm::Reader reader;
  reader.SetFileName( filename.c_str() );
  if( !reader.Read() )
    {
    std::cerr << "Could not read: " << filename << std::endl;
    return 1;
    }

  gdcm::File & file = reader.GetFile();
  gdcm::FileMetaInformation & header = file.GetHeader();
  gdcm::DataSet & ds = file.GetDataSet();

  gdcm::MediaStorage ms = header.GetMediaStorage(); (void)ms;
  const gdcm::TransferSyntax & ts = header.GetDataSetTransferSyntax();
  //std::cout << ts << std::endl;
  std::string msstr = header.GetMediaStorageAsString();
  //std::cout << msstr << std::endl;

  int pap3handle;
  if( decomp_pap3 )
    {
    if( !InitPapyrus3( filename.c_str(), pap3handle ) )
      {
      std::cerr << "Problem during init of PAPYRUS 3.0. File was: " << filename << std::endl;
      return 1;
      }
    }

  gdcm::PrivateTag pt(0x0041,0x50,"PAPYRUS 3.0");
  const gdcm::DataElement &depap = ds.GetDataElement( pt );
  gdcm::SmartPointer<gdcm::SequenceOfItems> sq = depap.GetValueAsSQ();

  if( !split )
    {
    gdcm::Writer w;
    w.CheckFileMetaInformationOff();
    w.SetFileName( outfilename.c_str() );
    w.SetFile( reader.GetFile() );

    if( decomp_pap3 )
      {
      gdcm::TransferSyntax outts = ts;
      for( gdcm::SequenceOfItems::SizeType i = 0; i < sq->GetNumberOfItems(); ++i )
        {
        gdcm::Item & it = sq->GetItem( i + 1 );
        gdcm::DataSet & nested = it.GetNestedDataSet();
        gdcm::File f;
        f.SetDataSet( nested );
        if( !DecompressPapyrus3( pap3handle, i, ts, f ) )
          {
          std::cerr << "Could not decompress frame #" << i << " from file: " << filename << std::endl;
          return 1;
          }
        const gdcm::DataElement & pixeldata = f.GetDataSet().GetDataElement( gdcm::Tag(0x7fe0,0x0010) );
        nested.Replace( pixeldata );
        }

      // make sq as undefined length (avoid length computation):
      gdcm::DataElement de_dup = depap;
      de_dup.SetValue( *sq );
      de_dup.SetVLToUndefined();
      ds.Replace( de_dup );

      gdcm::FileMetaInformation & h = w.GetFile().GetHeader();
      // pap3 returns image as decompressed:
      outts = gdcm::TransferSyntax::ExplicitVRLittleEndian;
      gdcm::Attribute<0x0002, 0x0010> TransferSyntaxUID;
      const char *tsstr = gdcm::TransferSyntax::GetTSString( outts );
      TransferSyntaxUID.SetValue( tsstr );
      h.Replace( TransferSyntaxUID.GetAsDataElement() );
      gdcm::Attribute<0x0002, 0x0000> filemetagrouplength;
      h.Remove( filemetagrouplength.GetTag() ); // important
      unsigned int glen = h.GetLength<gdcm::ExplicitDataElement>();
      filemetagrouplength.SetValue( glen );
      h.Insert( filemetagrouplength.GetAsDataElement() );
      }

    if( !w.Write() )
      {
      std::cerr << "Could not write output file: " << outfilename << std::endl;
      return 1;
      }
    }
  else
    {
    if( !gdcm::System::FileIsDirectory(outfilename.c_str()) )
      {
      std::cerr << "Output is not a directory: " << outfilename << std::endl;
      return 1;
      }
#if 1
    gdcm::UIDGenerator uid;

    const std::string seriesstr = uid.Generate();

    for( gdcm::SequenceOfItems::SizeType i = 0; i < sq->GetNumberOfItems(); ++i )
      {
      gdcm::Item & it = sq->GetItem( i + 1 );
      gdcm::DataSet & nested = it.GetNestedDataSet();

      std::stringstream ss;
      ss << outfilename;
      ss << "/IMG";
      ss << std::setw(4) << std::setfill( '0') << i;
      ss << ".dcm";
      gdcm::Writer w;
      // 1.2.840.10008.1.2.4.70
      w.CheckFileMetaInformationOn();
      const std::string & outfn = ss.str();
      w.SetFileName( outfn.c_str() );
      gdcm::TransferSyntax outts;
#ifdef GDCM_SUPPORT_BROKEN_IMPLEMENTATION
      if( ts == gdcm::TransferSyntax::WeirdPapryus )
        {
        outts = gdcm::TransferSyntax::ImplicitVRLittleEndian;
        }
      else
#endif
        {
        outts = ts;
        }

      w.GetFile().SetDataSet( nested );

      if( decomp_pap3 )
        {
        if( !DecompressPapyrus3( pap3handle, i, ts, w.GetFile() ) )
          {
          std::cerr << "Could not decompress frame #" << i << " from file: " << filename << std::endl;
          return 1;
          }
        // pap3 returns image as decompressed:
        outts = gdcm::TransferSyntax::ExplicitVRLittleEndian;
        }
      w.GetFile().GetHeader().SetDataSetTransferSyntax( outts );

      if( check_iop )
        {
        bool erroriop = false;
        std::vector<double> iop_orig;
        iop_orig.resize( 6 );
        // gdcm::ImageHelper::GetDirectionCosinesValue( w.GetFile() );
        if( !gdcm::ImageHelper::GetDirectionCosinesFromDataSet(w.GetFile().GetDataSet(), iop_orig) )
          {
          erroriop = true;
          gdcm::DirectionCosines dc( &iop_orig[0] );
          assert( !dc.IsValid() );
            {
            gdcm::Attribute<0x0008,0x0008> imagetype;
            imagetype.Set( w.GetFile().GetDataSet() );
            if( imagetype.GetNumberOfValues() > 2 )
              {
              const std::string &str = imagetype.GetValue( 2 );
              gdcm::Attribute<0x0020,0x0037> at_axial = {{1,0,0,0,1,0}}; // default value for AXIAL
              if( str == "AXIAL" )
                {
                w.GetFile().GetDataSet().Replace( at_axial.GetAsDataElement() );
                erroriop = false; // error has been corrected
                }
              else if( str == "LOCALIZER" )
                {
                static const double fake_axial[] = { 1, 0, 0, 0, 0, 0 };
                assert( memcmp( &iop_orig[0], fake_axial, 6 * sizeof( double ) ) == 0 ); (void)fake_axial;
                w.GetFile().GetDataSet().Replace( at_axial.GetAsDataElement() );
                erroriop = false; // error has been corrected
                }
              }
            assert( !erroriop ); // did our heuristic failed ?
            }
          }
        if( erroriop )
          {
          std::cerr << "Error IOP (could not correct) for frame #" << i << " value : ("
            << iop_orig[0] << ","
            << iop_orig[1] << ","
            << iop_orig[2] << ","
            << iop_orig[3] << ","
            << iop_orig[4] << ","
            << iop_orig[5] << ")" << std::endl;
          return 1;
          }
        }

#if 0
      gdcm::Attribute<0x0008,0x0016> outms;
      outms.SetValue( "1.2.840.10008.5.1.4.1.1.2" );
      nested.Replace( outms.GetAsDataElement() );

      gdcm::Attribute<0x028,0x0102> highbits = { 15 };
      nested.Replace( highbits.GetAsDataElement() );

      gdcm::Attribute<0x0028,0x0030> pixelspacing = { 0.57, 0.57 };
      nested.Replace( pixelspacing.GetAsDataElement() );

      gdcm::Attribute<0x0020,0x000e> seriesuid;
      seriesuid.SetValue( seriesstr );
      nested.Insert( seriesuid.GetAsDataElement() ); // do not replace if exists

      gdcm::Attribute<0x0008,0x0018> instanceuid;
      instanceuid.SetValue( uid.Generate() );
      nested.Replace( instanceuid.GetAsDataElement() );

      // ???
      gdcm::Attribute<0x0020,0x0032> ipp = {{0,0, i * 0.57}}; // default value
      nested.Replace( ipp.GetAsDataElement() );

      gdcm::Attribute<0x0020,0x0037> iop = {{1,0,0,0,1,0}}; // default value
      nested.Replace( iop.GetAsDataElement() );
#endif

      //std::cout << w.GetFile().GetDataSet( ) << std::endl;
      if( !w.Write() )
        {
        std::cerr << "Problem writing output file: " << outfn << std::endl;
        return 1;
        }
      }
#endif
    }

  if( decomp_pap3 )
    {
    if( !CleanupPapyrus3( pap3handle ) )
      {
      std::cerr << "Problem during PAPYRUS 3.0 cleanup" << std::endl;
      return 1;
      }
    }

  return 0;
}
