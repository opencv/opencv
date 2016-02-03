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
 * Simple command line tool to echo/store/find/move DICOM using
 * DICOM Query/Retrieve
 * This is largely inspired by other tool available from other toolkit, namely:
 * echoscu (DCMTK)
 * findscu (DCMTK)
 * movescu (DCMTK)
 * storescu (DCMTK)
 */

#include "gdcmCompositeNetworkFunctions.h"

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <getopt.h>
#include "gdcmVersion.h"
#include "gdcmGlobal.h"
#include "gdcmSystem.h"
#include "gdcmDirectory.h"
#include "gdcmDataSet.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmUIDGenerator.h"

#include "gdcmBaseRootQuery.h"
#include "gdcmQueryFactory.h"
#include "gdcmPrinter.h"


static void PrintVersion()
{
  std::cout << "gdcmscu: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmscu [OPTION]...[OPERATION]...HOSTNAME...[PORT]..." << std::endl;
  std::cout << "Execute a DICOM Q/R operation to HOSTNAME, using port PORT (104 when not specified)\n";
  std::cout << "Options:" << std::endl;
  std::cout << "  -H --hostname       Hostname." << std::endl;
  std::cout << "  -p --port           Port number." << std::endl;
  std::cout << "     --aetitle        Set Calling AE Title." << std::endl;
  std::cout << "     --call           Set Called AE Title." << std::endl;
  std::cout << "Mode Options:" << std::endl;
  std::cout << "     --echo           C-ECHO (default when none)." << std::endl;
  std::cout << "     --store          C-STORE." << std::endl;
  std::cout << "     --find           C-FIND." << std::endl;
  std::cout << "     --move           C-MOVE." << std::endl;
  std::cout << "     --get            C-GET." << std::endl;
  std::cout << "C-STORE Options:" << std::endl;
  std::cout << "  -i --input          DICOM filename" << std::endl;
  std::cout << "  -r --recursive      recursively process (sub-)directories." << std::endl;
  std::cout << "     --store-query    Store constructed query in file." << std::endl;
  std::cout << "C-FIND Options:" << std::endl;
  //std::cout << "     --worklist       C-FIND Worklist Model." << std::endl;//!!not supported atm
  std::cout << "     --patientroot    C-FIND Patient Root Model." << std::endl;
  std::cout << "     --studyroot      C-FIND Study Root Model." << std::endl;
  std::cout << "     --patient        C-FIND Query on Patient Info (cannot be used with --studyroot)" << std::endl;
  std::cout << "     --study          C-FIND Query on Study Info." << std::endl;
  std::cout << "     --series         C-FIND Query on Series Info." << std::endl;
  std::cout << "     --image          C-FIND Query on Image Info." << std::endl;
  //std::cout << "     --psonly         C-FIND Patient/Study Only Model." << std::endl;
  std::cout << "     --key            0123,4567=VALUE for specifying search criteria (wildcard allowed)." << std::endl;
  std::cout << "                      With --key, leave blank (ie, --key 10,10="") to retrieve values" << std::endl;
  std::cout << "C-MOVE Options:" << std::endl;
  std::cout << "  -o --output         DICOM output directory." << std::endl;
  std::cout << "     --port-scp       Port used for incoming association." << std::endl;
  std::cout << "     --key            0123,4567=VALUE for specifying search criteria (wildcard not allowed)." << std::endl;
  std::cout << "  Note that C-MOVE supports the same queries as C-FIND, but no wildcards are allowed." << std::endl;
  std::cout << "C-GET Options:" << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "     --root-uid               Root UID." << std::endl;
  std::cout << "  -V --verbose   more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning   print warning info." << std::endl;
  std::cout << "  -D --debug     print debug info." << std::endl;
  std::cout << "  -E --error     print error info." << std::endl;
  std::cout << "  -h --help      print help." << std::endl;
  std::cout << "     --queryhelp print query help." << std::endl;
  std::cout << "  -v --version   print version." << std::endl;
  std::cout << "  -L --log-file  set log file (instead of cout)." << std::endl;

  try
    {
    std::locale l("");
    std::string loc = l.name();
    std::cout << std::endl;
    std::cout << "Local Name: " << loc << std::endl;
    }
  catch( const std::exception& e)
    {
    std::cerr << e.what() << std::endl;
    }
  std::cout << "Local Character Set: " << gdcm::System::GetLocaleCharset() << std::endl;
  std::vector<gdcm::ECharSet> charsettype;
  charsettype.push_back( gdcm::QueryFactory::GetCharacterFromCurrentLocale() );
  gdcm::DataElement de = gdcm::QueryFactory::ProduceCharacterSetDataElement(charsettype);
  const gdcm::ByteValue *bv = de.GetByteValue();
  std::string s( bv->GetPointer(), bv->GetLength() );
  std::cout << "DICOM Character Set: [" << s << "]" << std::endl;
}

static void PrintQueryHelp(int inFindPatientRoot)
{
  gdcm::BaseRootQuery* theBase;
  if (inFindPatientRoot)
  {
    std::cout << "To find the help for a study-level query, type" <<std::endl;
    std::cout << " --queryhelp --studyroot" << std::endl;
    theBase = gdcm::QueryFactory::ProduceQuery(gdcm::ePatientRootType, gdcm::eFind, gdcm::ePatient);
    theBase->WriteHelpFile(std::cout);
    delete theBase;
  }
  else
  {
    std::cout << "To find the help for a patient-level query, type" <<std::endl;
    std::cout << " --queryhelp --patientroot" << std::endl;
    std::cout << "These are the study level, study root queries: " << std::endl;
    theBase = gdcm::QueryFactory::ProduceQuery(gdcm::eStudyRootType, gdcm::eFind, gdcm::eStudy);
    theBase->WriteHelpFile(std::cout);
    delete theBase;
  }
}

int main(int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;
  
  std::string shostname;
  std::string callingaetitle = "GDCMSCU";
  std::string callaetitle = "ANY-SCP";
  int port = 104; // default
  int portscp = 0;
  int outputopt = 0;
  int portscpnum = 0;
  gdcm::Directory::FilenamesType filenames;
  std::string outputdir;
  int storequery = 0;
  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int queryhelp = 0;
  int version = 0;
  int echomode = 0;
  int storemode = 0;
  int findmode = 0;
  int movemode = 0;
  int getmode = 0;
  int findworklist = 0;
  int findpatientroot = 0;
  int findstudyroot = 0;
  int patientquery = 0;
  int studyquery = 0;
  int seriesquery = 0;
  int imagequery = 0;
  int findpsonly = 0;
  std::string queryfile;
  std::string root;
  int rootuid = 0;
  int recursive = 0;
  int logfile = 0;
  std::string logfilename;
  gdcm::Tag tag;
  std::vector< std::pair<gdcm::Tag, std::string> > keys;
  
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
      {"verbose", 0, &verbose, 1},
      {"warning", 0, &warning, 1},
      {"debug", 0, &debug, 1},
      {"error", 0, &error, 1},
      {"help", 0, &help, 1},
      {"version", 0, &version, 1},
      {"hostname", 1, 0, 0},     // -h
      {"aetitle", 1, 0, 0},     //
      {"call", 1, 0, 0},     //
      {"port", 0, &port, 1}, // -p
      {"input", 1, 0, 0}, // dcmfile-in
      {"echo", 0, &echomode, 1}, // --echo
      {"store", 0, &storemode, 1}, // --store
      {"find", 0, &findmode, 1}, // --find
      {"move", 0, &movemode, 1}, // --move
      {"key", 1, 0, 0}, // (15) --key
      {"worklist", 0, &findworklist, 1}, // --worklist
      {"patientroot", 0, &findpatientroot, 1}, // --patientroot
      {"studyroot", 0, &findstudyroot, 1}, // --studyroot
      {"psonly", 0, &findpsonly, 1}, // --psonly
      {"port-scp", 1, &portscp, 1}, // (20) --port-scp
      {"output", 1, &outputopt, 1}, // --output
      {"recursive", 0, &recursive, 1},
      {"store-query", 1, &storequery, 1},
      {"queryhelp", 0, &queryhelp, 1},
      {"patient", 0, &patientquery, 1}, // --patient
      {"study", 0, &studyquery, 1}, // --study
      {"series", 0, &seriesquery, 1}, // --series
      {"image", 0, &imagequery, 1}, // --image
      {"log-file", 1, &logfile, 1}, // --log-file
      {"get", 0, &getmode, 1}, // --get
      {0, 0, 0, 0} // required
    };
    static const char short_options[] = "i:H:p:L:VWDEhvk:o:r";
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
            filenames.push_back( optarg );
          }
          else if( option_index == 7 ) /* calling aetitle */
          {
            assert( strcmp(s, "aetitle") == 0 );
            //assert( callingaetitle.empty() );
            callingaetitle = optarg;
          }
          else if( option_index == 8 ) /* called aetitle */
          {
            assert( strcmp(s, "call") == 0 );
            //assert( callaetitle.empty() );
            callaetitle = optarg;
          }
          else if( option_index == 15 ) /* key */
          {
            assert( strcmp(s, "key") == 0 );
            if( !tag.ReadFromCommaSeparatedString(optarg) )
            {
              std::cerr << "Could not read Tag: " << optarg << std::endl;
              return 1;
            }
            std::stringstream ss;
            ss.str( optarg );
            uint16_t dummy;
            char cdummy; // comma
            ss >> std::hex >> dummy;
            assert( tag.GetGroup() == dummy );
            ss >> cdummy;
            assert( cdummy == ',' );
            ss >> std::hex >> dummy;
            assert( tag.GetElement() == dummy );
            ss >> cdummy;
            assert( cdummy == ',' || cdummy == '=' );
            std::string str;
            //ss >> str;
            std::getline(ss, str); // do not skip whitespace
            keys.push_back( std::make_pair(tag, str) );
          }
          else if( option_index == 20 ) /* port-scp */
          {
            assert( strcmp(s, "port-scp") == 0 );
            portscpnum = atoi(optarg);
          }
          else if( option_index == 21 ) /* output */
          {
            assert( strcmp(s, "output") == 0 );
            outputdir = optarg;
          }
          else if( option_index == 23 ) /* store-query */
          {
            assert( strcmp(s, "store-query") == 0 );
            queryfile = optarg;
          }
          else if( option_index == 29 ) /* log-file */
          {
            assert( strcmp(s, "log-file") == 0 );
            logfilename = optarg;
          }
          else
          {
            // If you reach here someone mess-up the index and the argument in
            // the getopt table
            assert( 0 );
          }
          //printf (" with arg %s", optarg);
        }
        //printf ("\n");
      }
        break;
        
      case 'k':
      {
        if( !tag.ReadFromCommaSeparatedString(optarg) )
        {
          std::cerr << "Could not read Tag: " << optarg << std::endl;
          return 1;
        }
        std::stringstream ss;
        ss.str( optarg );
        uint16_t dummy;
        char cdummy; // comma
        ss >> std::hex >> dummy;
        assert( tag.GetGroup() == dummy );
        ss >> cdummy;
        assert( cdummy == ',' );
        ss >> std::hex >> dummy;
        assert( tag.GetElement() == dummy );
        ss >> cdummy;
        assert( cdummy == ',' || cdummy == '=' );
        std::string str;
        std::getline(ss, str); // do not skip whitespace
        keys.push_back( std::make_pair(tag, str) );
      }
        break;
        
      case 'i':
        //printf ("option i with value '%s'\n", optarg);
        filenames.push_back( optarg );
        break;
        
      case 'r':
        recursive = 1;
        break;
        
      case 'o':
        assert( outputdir.empty() );
        outputdir = optarg;
        break;
        
      case 'H':
        shostname = optarg;
        break;
        
      case 'p':
        port = atoi( optarg );
        break;
        
      case 'L':
        logfile = 1;
        logfilename = optarg;
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
        
      case 'q':
        queryhelp = 1;
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
    int v = argc - optind;
    // hostname port filename
    if( v == 1 )
      {
      shostname = argv[optind++];
      }
    else if( v == 2 )
      {
      shostname = argv[optind++];
      port = atoi( argv[optind++] );
      }
    else if( v >= 3 )
      {
      shostname = argv[optind++];
      port = atoi( argv[optind++] );
      std::vector<std::string> files;
      while (optind < argc)
        {
        files.push_back( argv[optind++] );
        }
      filenames = files;
      }
    else
      {
      return 1;
      }
    assert( optind == argc );
    }
  
  if( version )
  {
    PrintVersion();
    return 0;
  }
  
  if( help )
    {
    PrintHelp();
    return 0;
    }
  if(queryhelp)
    {
    PrintQueryHelp(findpatientroot);
    return 0;
    }
  const bool theDebug = debug != 0;
  const bool theWarning = warning != 0;
  const bool theError = error != 0;
  const bool theVerbose = verbose != 0;
  const bool theRecursive = recursive != 0;
  // Debug is a little too verbose
  gdcm::Trace::SetDebug( theDebug );
  gdcm::Trace::SetWarning( theWarning );
  gdcm::Trace::SetError( theError );
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( theVerbose );
    gdcm::Trace::SetError( theVerbose);
    }
  if( logfile )
    {
    gdcm::Trace::SetStreamToFile( logfilename.c_str() );
    }
  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( callaetitle.c_str() );
  if( !rootuid )
    {
    // only read the env var if no explicit cmd line option
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
  
  if( shostname.empty() )
    {
    //std::cerr << "Hostname missing" << std::endl;
    PrintHelp(); // needed to display help message when no arg
    return 1;
    }
  if( port == 0 )
    {
    std::cerr << "Problem with port number" << std::endl;
    return 1;
    }
  // checkout outputdir opt:
  if( outputopt )
    {
    if( !gdcm::System::FileIsDirectory( outputdir.c_str()) )
      {
      if( !gdcm::System::MakeDirectory( outputdir.c_str() ) )
        {
        std::cerr << "Sorry: " << outputdir << " is not a valid directory.";
        std::cerr << std::endl;
        std::cerr << "and I could not create it.";
        std::cerr << std::endl;
        return 1;
        }
      }
    }
  
  const char *hostname = shostname.c_str();
  std::string mode = "echo";
  if ( echomode )
    {
    mode = "echo";
    }
  else if ( storemode )
    {
    mode = "store";
    }
  else if ( findmode )
    {
    mode = "find";
    }
  else if ( movemode )
    {
    mode = "move";
    }
  else if ( getmode )
    {
    mode = "get";
    }
  
  //this class contains the networking calls
  
  if ( mode == "server" ) // C-STORE SCP
    {
    // MM: Do not expose that to user for now (2010/10/11).
    //CStoreServer( port );
    return 1;
    }
  else if ( mode == "echo" ) // C-ECHO SCU
    {
    // ./bin/gdcmscu mi2b2.slicer.org 11112  --aetitle ACME1 --call MI2B2
    // ./bin/gdcmscu --echo mi2b2.slicer.org 11112  --aetitle ACME1 --call MI2B2
    bool didItWork = gdcm::CompositeNetworkFunctions::CEcho( hostname, (uint16_t)port,
      callingaetitle.c_str(), callaetitle.c_str() );
    gdcmDebugMacro( (didItWork ? "Echo succeeded." : "Echo failed.") );
    return didItWork ? 0 : 1;
    }
  else if ( mode == "move" ) // C-FIND SCU
    {
    // ./bin/gdcmscu --move --patient dhcp-67-183 5678 move
    // ./bin/gdcmscu --move --patient mi2b2.slicer.org 11112 move
    gdcm::ERootType theRoot = gdcm::eStudyRootType;
    if (findpatientroot)
      theRoot = gdcm::ePatientRootType;
    gdcm::EQueryLevel theLevel = gdcm::eStudy;
    if (patientquery)
      theLevel = gdcm::ePatient;
    if (seriesquery)
      theLevel = gdcm::eSeries;
    if (imagequery)
      theLevel = gdcm::eImage;

    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys, true);

    if (findstudyroot == 0 && findpatientroot == 0)
      {
      if (gdcm::Trace::GetErrorFlag())
        {
        std::cerr << "Need to explicitly choose query retrieve level, --patientroot or --studyroot" << std::endl;      
        }
      std::cerr << "Move failed." << std::endl;
      return 1;
      }

    if( !portscp )
      {
      std::cerr << "Need to set explicitly port number for SCP association"
        " --port-scp" << std::endl;
      //std::cerr << "Move failed." << std::endl;
      return 1;
      }

    if( storequery )
      {
      if (!theQuery->WriteQuery(queryfile))
        {
        std::cerr << "Could not write out query to: " << queryfile << std::endl;
        std::cerr << "Move failed." << std::endl;
        return 1;
        }
      }

    if (!theQuery->ValidateQuery(false))
      {
      std::cerr << "You have not constructed a valid find query."
        " Please try again." << std::endl;
      return 1;
      }

    //!!! added the boolean to 'interleave writing', which basically writes
    //each file out as it comes across, rather than all at once at the end.
    //Turn off the boolean to have it written all at once at the end.
    bool didItWork = gdcm::CompositeNetworkFunctions::CMove( hostname, (uint16_t)port,
      theQuery, (uint16_t)portscpnum,
      callingaetitle.c_str(), callaetitle.c_str(), outputdir.c_str() );
    gdcmDebugMacro( (didItWork ? "Move succeeded." : "Move failed.") );
    return didItWork ? 0 : 1;
    }
  else if ( mode == "find" ) // C-FIND SCU
    {
    // Construct C-FIND DataSet:
    // ./bin/gdcmscu --find --patient dhcp-67-183 5678
    // ./bin/gdcmscu --find --patient mi2b2.slicer.org 11112  --aetitle ACME1 --call MI2B2
    // findscu -aec MI2B2 -P -k 0010,0010=F* mi2b2.slicer.org 11112 patqry.dcm

    // PATIENT query:
    // ./bin/gdcmscu --find --patient mi2b2.slicer.org 11112  --aetitle ACME1 --call MI2B2 --key 10,10="F*" -V
    gdcm::ERootType theRoot = gdcm::eStudyRootType;
    if (findpatientroot)
      theRoot = gdcm::ePatientRootType;
    gdcm::EQueryLevel theLevel = gdcm::eStudy;
    if (patientquery)
      theLevel = gdcm::ePatient;
    if (seriesquery)
      theLevel = gdcm::eSeries;
    if (imagequery)
      theLevel = gdcm::eImage;

    gdcm::SmartPointer<gdcm::BaseRootQuery> theQuery =
      gdcm::CompositeNetworkFunctions::ConstructQuery(theRoot, theLevel ,keys);

    if (findstudyroot == 0 && findpatientroot == 0)
      {
      if (gdcm::Trace::GetErrorFlag())
        {
        std::cerr << "Need to explicitly choose query retrieve level, --patientroot or --studyroot" << std::endl;      
        }
      std::cerr << "Find failed." << std::endl;
      return 1;
      }
    if (!theQuery)
      {
      std::cerr << "Query construction failed." <<std::endl;
      return 1;
      }

    if( storequery )
      {
      if (!theQuery->WriteQuery(queryfile))
        {
        std::cerr << "Could not write out query to: " << queryfile << std::endl;
        return 1;
        }
      }

    //doing a non-strict query, the second parameter there.
    //look at the base query comments
    if (!theQuery->ValidateQuery(false))
      {
      std::cerr << "You have not constructed a valid find query."
        " Please try again." << std::endl;
      return 1;
      }
    //the value in that tag corresponds to the query type
    std::vector<gdcm::DataSet> theDataSet;
    if( !gdcm::CompositeNetworkFunctions::CFind(hostname, (uint16_t)port, theQuery, theDataSet,
        callingaetitle.c_str(), callaetitle.c_str()) )
      {
      gdcmDebugMacro( "Problem in CFind." );
      return 1;
      }

    gdcm::Printer p;
    std::ostream &os = gdcm::Trace::GetStream();
    for( std::vector<gdcm::DataSet>::iterator itor
      = theDataSet.begin(); itor != theDataSet.end(); itor++)
      {
      os << "Find Response: " << (itor - theDataSet.begin() + 1) << std::endl;
      p.PrintDataSet( *itor, os );
      os << std::endl;
      }

    if( gdcm::Trace::GetWarningFlag() ) // == verbose flag
      {
      os << "Find was successful." << std::endl;
      }
    return 0;
    }
  else if ( mode == "store" ) // C-STORE SCU
    {
    // mode == directory
    gdcm::Directory::FilenamesType thefiles;
    for( gdcm::Directory::FilenamesType::const_iterator file = filenames.begin();
      file != filenames.end(); ++file )
      {
      if( gdcm::System::FileIsDirectory(file->c_str()) )
        {
        gdcm::Directory::FilenamesType files;
        gdcm::Directory dir;
        dir.Load(*file, theRecursive);
        files = dir.GetFilenames();
        thefiles.insert(thefiles.end(), files.begin(), files.end());
        }
      else
        {
        // This is a file simply add it
        thefiles.push_back(*file);
        }
      }
    bool didItWork = 
      gdcm::CompositeNetworkFunctions::CStore(hostname, (uint16_t)port, thefiles,
        callingaetitle.c_str(), callaetitle.c_str());

    gdcmDebugMacro( (didItWork ? "Store was successful." : "Store failed.") );
    return didItWork ? 0 : 1;
    }
  else if ( mode == "get" ) // C-GET SCU
    {
    return 1;
    }
  else
    {
    assert( 0 );
    return 1;
    }

  return 0;
}
