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
 * Implementation of General Purpose CD-R Interchange / STD-GEN-CD DICOMDIR in GDCM
 */
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmVersion.h"
#include "gdcmSystem.h"
#include "gdcmUIDGenerator.h"
#include "gdcmGlobal.h"
#include "gdcmDefs.h"
#include "gdcmDirectory.h"
#include "gdcmDICOMDIRGenerator.h"

#include <getopt.h>


static void PrintVersion()
{
  std::cout << "gdcmgendir: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmgendir [OPTION]... FILE..." << std::endl;
  std::cout << "create DICOMDIR" << std::endl;
  std::cout << "Parameter:" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -i --input     DICOM filename or directory" << std::endl;
  std::cout << "  -o --output    DICOM filename or directory" << std::endl;
  std::cout << "  -r --recursive          recursive." << std::endl;
  std::cout << "     --descriptor          descriptor." << std::endl;
  std::cout << "     --root-uid               Root UID." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose   more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning   print warning info." << std::endl;
  std::cout << "  -D --debug     print debug info." << std::endl;
  std::cout << "  -E --error     print error info." << std::endl;
  std::cout << "  -h --help      print help." << std::endl;
  std::cout << "  -v --version   print version." << std::endl;
  std::cout << "Env var:" << std::endl;
  std::cout << "  GDCM_ROOT_UID Root UID" << std::endl;
}

int main(int argc, char *argv[])
{
  int c;
  std::string filename;
  std::string outfilename;
  gdcm::Directory::FilenamesType filenames;
  std::string xmlpath;
  int verbose = 0;
  int warning = 0;
  int help = 0;
  int recursive = 0;
  int version = 0;
  int debug = 0;
  int error = 0;
  int resourcespath = 0;
  int rootuid = 0;
  int descriptor = 0;
  std::string descriptor_str;
  std::string root;
  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"output", 1, 0, 0},                // o
        {"recursive", 0, &recursive, 1},
        {"root-uid", 1, &rootuid, 1}, // specific Root (not GDCM)
        {"resources-path", 1, &resourcespath, 1},
        {"descriptor", 1, &descriptor, 1},

        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {0, 0, 0, 0} // required
    };
    static const char short_options[] = "i:o:rVWDEhv";
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
          else if( option_index == 1 ) /* output */
            {
            assert( strcmp(s, "output") == 0 );
            assert( outfilename.empty() );
            outfilename = optarg;
            }
          else if( option_index == 3 ) /* root-uid */
            {
            assert( strcmp(s, "root-uid") == 0 );
            root = optarg;
            }
          else if( option_index == 4 ) /* resources-path */
            {
            assert( strcmp(s, "resources-path") == 0 );
            assert( xmlpath.empty() );
            xmlpath = optarg;
            }
          else if( option_index == 5 ) /* descriptor */
            {
            assert( strcmp(s, "descriptor") == 0 );
            assert( descriptor_str.empty() );
            descriptor_str = optarg;
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

    case 'o':
      assert( outfilename.empty() );
      outfilename = optarg;
      break;

    case 'r':
      recursive = 1;
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

  if (optind < argc)
    {
    //printf ("non-option ARGV-elements: %d", optind );
    std::vector<std::string> files;
    while (optind < argc)
      {
      //printf ("%s\n", argv[optind++]);
      files.push_back( argv[optind++] );
      }
    //printf ("\n");
    if( files.size() >= 2
      && filename.empty()
      && outfilename.empty()
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

  if( filename.empty()
    || outfilename.empty() )
    {
    //std::cerr << "Need input file (-i)\n";
    PrintHelp();
    return 1;
    }

  // Debug is a little too verbose
  gdcm::Trace::SetDebug( debug != 0 );
  gdcm::Trace::SetWarning( warning != 0 );
  gdcm::Trace::SetError( error != 0 );
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( verbose != 0 );
    gdcm::Trace::SetError( verbose != 0);
    }

  if( !gdcm::System::FileExists(filename.c_str()) )
    {
    return 1;
    }

/*
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

  const gdcm::Defs &defs = g.GetDefs();
*/

  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( "gdcmgendir" );
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

  int res = 0;
  unsigned int nfiles = 1;
  gdcm::DICOMDIRGenerator gen;
  if( gdcm::System::FileIsDirectory(filename.c_str()) )
    {
    gdcm::Directory dir;
    nfiles = dir.Load(filename, recursive!= 0); (void)nfiles;
    filenames = dir.GetFilenames();
    gen.SetRootDirectory( filename );
    }
  else
    {
    // should be all set !
    }
  (void)nfiles;

  gen.SetFilenames( filenames );
  gen.SetDescriptor( descriptor_str.c_str() );
  if( !gen.Generate() )
    {
    std::cerr << "Problem during generation" << std::endl;
    return 1;
    }

  gdcm::Writer writer;
  writer.SetFile( gen.GetFile() );
  writer.SetFileName( outfilename.c_str() );
  if( !writer.Write() )
    {
    return 1;
    }

  return res;
}
