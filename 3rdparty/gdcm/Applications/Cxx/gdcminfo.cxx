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
 * TODO:
 * Should implement the gdcmiodvfy here
 * I need to implement gdcmoverlay here (print info on overlay / img / LUT ...)
 */
#include "gdcmReader.h"
#include "gdcmImageReader.h"
#include "gdcmMediaStorage.h"
#include "gdcmFile.h"
#include "gdcmDataSet.h"
#include "gdcmUIDs.h"
#include "gdcmGlobal.h"
#include "gdcmModules.h"
#include "gdcmDefs.h"
#include "gdcmOrientation.h"
#include "gdcmVersion.h"
#include "gdcmMD5.h"
#include "gdcmSystem.h"
#include "gdcmDirectory.h"

#ifdef GDCM_USE_SYSTEM_POPPLER
#include <poppler/poppler-config.h>
#include <poppler/PDFDoc.h>
#include <poppler/UnicodeMap.h>
#include <poppler/PDFDocEncoding.h>
#include <poppler/GlobalParams.h>
#endif // GDCM_USE_SYSTEM_POPPLER

#include "puff.h"

#include <iostream>

#include <stdio.h>     /* for printf */
#include <stdint.h>
#include <stdlib.h>    /* for exit */
#include <string.h>
#include <getopt.h>
#include <sys/types.h>
#include <sys/stat.h>


static int checkmagick(unsigned char *input)
{
  if( input[128+0] == 'D'
   && input[128+1] == 'I'
   && input[128+2] == 'C'
   && input[128+3] == 'M' )
    {
    return 1;
    }
  return 0;
}

static int checkdeflated(const char *name)
{
  int ret;
  unsigned char *source;
  unsigned long len, sourcelen, destlen;

  unsigned long size;
  unsigned long size1;
  unsigned char *buf;
  FILE *in;
  struct stat s;
  //const char *name = 0;
  union { uint32_t tag; uint16_t tags[2]; char bytes[4]; } tag;
  char vr[3];
  uint16_t vl;
  uint32_t value;

  //if (argc < 2) return 2;
  //name = argv[1];

  len = 0;
  if (stat(name, &s))
    {
    fprintf( stderr, "Cannot stat: %s\n", name );
    return 1;
    }
  if ((s.st_mode & S_IFMT) != S_IFREG)
    {
    fprintf( stderr, "not a regular file\n" );
    return 1;
    }
  size = (unsigned long)(s.st_size);
  if (size == 0 || (off_t)size != s.st_size)
    {
    fprintf( stderr, "size mismatch\n" );
    return 1;
    }
  in = fopen(name, "r");
  if (in == NULL)
    {
    fprintf( stderr, "in is NULL\n" );
    return 1;
    }
  buf = (unsigned char*)malloc(size);
  if (buf != NULL && (size1 = fread(buf, 1, size, in)) != size) {
    free(buf);
    buf = NULL;
    fprintf( stderr, "could not fread: %lu bytes != %lu\n", size, size1 );
    fprintf( stderr, "feof: %i ferror %i\n", feof(in), ferror(in) );
  }
  fclose(in);
  len = size;
  source = buf;
  if( source == NULL ) {
    fprintf( stderr, "source is NULL\n" );
    return 1;
  }
  sourcelen = len;

  if( !checkmagick(source) )
    {
    fprintf( stderr, "checkmagick failed\n" );
    return 1;
    }
  // magick succeed so skip header:
  source += 128 + 4;
  sourcelen -= 128 + 4;

  memcpy(&tag, source, sizeof(tag) );
  fprintf( stdout,"tag: %d, %d\n", tag.tags[0], tag.tags[1] );
  source += sizeof(tag);
  sourcelen -= sizeof(tag);

  vr[2] = 0;
  memcpy(vr, source, 2);
  printf( "vr: %s\n", vr);

  source += 2;
  sourcelen -= 2;

  memcpy(&vl, source, sizeof(vl));
  printf( "vl: %d\n", vl);

  source += sizeof(vl);
  sourcelen -= sizeof(vl);

  memcpy(&value, source, sizeof(value));
  printf( "value: %d\n", value);

  source += sizeof(value);
  sourcelen -= sizeof(value);

  source += value;
  sourcelen -= value;

  len = sourcelen;
  if( len % 2 )
    {
    printf( "len of bit stream is odd: %lu. Continuing anyway\n", len );
    }
  else
    {
    printf( "deflate stream has proper length: %lu\n", len );
    }

  ret = puff(NULL, &destlen, source, &sourcelen);

  if (ret)
    fprintf(stdout,"puff() failed with return code %d\n", ret);
  else {
    fprintf(stdout,"puff() succeeded uncompressing %lu bytes\n", destlen);
    if (sourcelen < len) printf("%lu compressed bytes unused\n",
      len - sourcelen);
  }
  free(buf);
  return ret;
}

#ifdef GDCM_USE_SYSTEM_POPPLER
static std::string getInfoDate(Dict *infoDict, const char *key)
{
  Object obj;
  char *s;
  int year, mon, day, hour, min, sec, n;
  struct tm tmStruct;
  //char buf[256];
  std::string out;

  if (infoDict->lookup((char*)key, &obj)->isString())
    {
    s = obj.getString()->getCString();
    if (s[0] == 'D' && s[1] == ':')
      {
      s += 2;
      }
    if ((n = sscanf(s, "%4d%2d%2d%2d%2d%2d",
          &year, &mon, &day, &hour, &min, &sec)) >= 1)
      {
      switch (n)
        {
      case 1: mon = 1;
      case 2: day = 1;
      case 3: hour = 0;
      case 4: min = 0;
      case 5: sec = 0;
        }
      tmStruct.tm_year = year - 1900;
      tmStruct.tm_mon = mon - 1;
      tmStruct.tm_mday = day;
      tmStruct.tm_hour = hour;
      tmStruct.tm_min = min;
      tmStruct.tm_sec = sec;
      tmStruct.tm_wday = -1;
      tmStruct.tm_yday = -1;
      tmStruct.tm_isdst = -1;
/*
      // compute the tm_wday and tm_yday fields
      if (mktime(&tmStruct) != (time_t)-1 &&
        strftime(buf, sizeof(buf), "%c", &tmStruct)) {
        fputs(buf, stdout);
      } else {
        fputs(s, stdout);
      }
      } else {
        fputs(s, stdout);
*/
      }
    //fputc('\n', stdout);
    char date[22];
    time_t t = mktime(&tmStruct);
    if( t != -1 )
      {
      if( gdcm::System::FormatDateTime(date, t) )
        out = date;
      }
    }
  obj.free();
  return out;
}

static std::string getInfoString(Dict *infoDict, const char *key, UnicodeMap *uMap)
{
  Object obj;
  GooString *s1;
  GBool isUnicode;
  Unicode u;
  char buf[8];
  int i, n;
  std::string out;

  if (infoDict->lookup((char*)key, &obj)->isString())
    {
    s1 = obj.getString();
    if ((s1->getChar(0) & 0xff) == 0xfe &&
      (s1->getChar(1) & 0xff) == 0xff)
      {
      isUnicode = gTrue;
      i = 2;
      }
    else
      {
      isUnicode = gFalse;
      i = 0;
      }
    while (i < obj.getString()->getLength())
      {
      if (isUnicode)
        {
        u = ((s1->getChar(i) & 0xff) << 8) |
          (s1->getChar(i+1) & 0xff);
        i += 2;
        }
      else
        {
        u = pdfDocEncoding[s1->getChar(i) & 0xff];
        ++i;
        }
      n = uMap->mapUnicode(u, buf, sizeof(buf));
      //fwrite(buf,1,n,stdout);
      out.append( std::string(buf, n) );
      }
    }
  obj.free();
  return out;
}
#endif


static void PrintVersion()
{
  std::cout << "gdcminfo: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcminfo [OPTION]... FILE..." << std::endl;
  std::cout << "display meta info about the input DICOM file" << std::endl;
  std::cout << "Parameter:" << std::endl;
  std::cout << "  -i --input     DICOM filename or directory" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -r --recursive          recursive." << std::endl;
  std::cout << "  -d --check-deflated     check if file is proper deflated syntax." << std::endl;
  std::cout << "     --resources-path     Resources path." << std::endl;
  std::cout << "     --md5sum             Compute md5sum of Pixel Data attribute value." << std::endl;
  std::cout << "     --check-compression  check the encapsulated stream compression (lossless/lossy)." << std::endl;
  // the following options would require an advanced MediaStorage::SetFromFile ... sigh
  //std::cout << "     --media-storage-uid   return media storage uid only." << std::endl;
  //std::cout << "     --media-storage-name  return media storage name only (when possible)." << std::endl;
//  std::cout << "  -b --check-big-endian   check if file is ." << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose   more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning   print warning info." << std::endl;
  std::cout << "  -D --debug     print debug info." << std::endl;
  std::cout << "  -E --error     print error info." << std::endl;
  std::cout << "  -h --help      print help." << std::endl;
  std::cout << "  -v --version   print version." << std::endl;
  std::cout << "Env var:" << std::endl;
  std::cout << "  GDCM_RESOURCES_PATH path pointing to resources files (Part3.xml, ...)" << std::endl;
}

  int deflated = 0; // check deflated
  int checkcompression = 0;
  int md5sum = 0;

static int ProcessOneFile( std::string const & filename, gdcm::Defs const & defs )
{
  (void)defs;
  if( deflated )
    {
    return checkdeflated(filename.c_str());
    }

  //const char *filename = argv[1];
  //std::cout << "filename: " << filename << std::endl;
  gdcm::Reader reader0;
  reader0.SetFileName( filename.c_str() );
  if( !reader0.Read() )
    {
    std::cerr << "Failed to read: " << filename << std::endl;
    return 1;
    }
  const gdcm::File &file = reader0.GetFile();
  gdcm::MediaStorage ms;
  ms.SetFromFile(file);
  /*
   * Until gdcm::MediaStorage is fixed only *compile* time constant will be handled
   * see -> http://chuckhahm.com/Ischem/Zurich/XX_0134
   * which make gdcm::UIDs useless :(
   */
  if( ms.IsUndefined() )
    {
    std::cerr << "Unknown MediaStorage" << std::endl;
    return 1;
    }

  gdcm::UIDs uid;
  uid.SetFromUID( ms.GetString() );
  std::cout << "MediaStorage is " << ms << " [" << uid.GetName() << "]" << std::endl;
  const gdcm::TransferSyntax &ts = file.GetHeader().GetDataSetTransferSyntax();
  uid.SetFromUID( ts.GetString() );
  std::cout << "TransferSyntax is " << ts << " [" << uid.GetName() <<  "]" << std::endl;

  if( gdcm::MediaStorage::IsImage( ms ) )
    {
    gdcm::ImageReader reader;
    reader.SetFileName( filename.c_str() );
    if( !reader.Read() )
      {
      std::cerr << "Could not read image from: " << filename << std::endl;
      return 1;
      }
    //const gdcm::File &file = reader.GetFile();
    //const gdcm::DataSet &ds = file.GetDataSet();
    const gdcm::Image &image = reader.GetImage();
    const double *dircos = image.GetDirectionCosines();
    gdcm::Orientation::OrientationType type = gdcm::Orientation::GetType(dircos);
    const char *label = gdcm::Orientation::GetLabel( type );
    image.Print( std::cout );
    std::cout << "Orientation Label: " << label << std::endl;
    if( checkcompression )
      {
      bool lossy = image.IsLossy();
      std::cout << "Encapsulated Stream was found to be: " << (lossy ? "lossy" : "lossless") << std::endl;
      }

    if( md5sum )
      {
      char *buffer = new char[ image.GetBufferLength() ];
      if( image.GetBuffer( buffer ) )
        {
        char digest[33] = {};
        gdcm::MD5::Compute( buffer, image.GetBufferLength(), digest );
        std::cout << "md5sum: " << digest << std::endl;
        }
      else
        {
        std::cout << "Problem decompressing file: " << filename << std::endl;
        }
      delete[] buffer;
      }
    }
  else if ( ms == gdcm::MediaStorage::EncapsulatedPDFStorage )
    {
#ifdef GDCM_USE_SYSTEM_POPPLER
    const gdcm::DataSet &ds = file.GetDataSet();
    const gdcm::DataElement& de = ds.GetDataElement( gdcm::Tag(0x42,0x11) );
    const gdcm::ByteValue* bv = de.GetByteValue();
    const char *p = bv->GetPointer(); (void)p;
    Object appearDict;
    //appearDict.initDict(xref);
    //appearDict.dictAdd(copyString("Length"),
    //	     obj1.initInt(appearBuf->getLength()));
    //appearDict.dictAdd(copyString("Subtype"), obj1.initName("Form"));
    //obj1.initArray(xref);
    //obj1.arrayAdd(obj2.initReal(0));
    //obj1.arrayAdd(obj2.initReal(0));
    //obj1.arrayAdd(obj2.initReal(xMax - xMin));
    //obj1.arrayAdd(obj2.initReal(yMax - yMin));
    //appearDict.dictAdd(copyString("BBox"), &obj1);

    MemStream *appearStream;

    appearStream = new MemStream((char*)bv->GetPointer(), 0,
      bv->GetLength(), &appearDict);
    GooString *ownerPW, *userPW;
    ownerPW = NULL;
    userPW = NULL;

    PDFDoc *doc;
    doc = new PDFDoc(appearStream, ownerPW, userPW);

    std::string title;
    std::string subject;
    std::string keywords;
    std::string author;
    std::string creator;
    std::string producer;
    std::string creationdate;
    std::string moddate;

    UnicodeMap *uMap;
#ifdef LIBPOPPLER_GLOBALPARAMS_CSTOR_HAS_PARAM
    globalParams = new GlobalParams(0);
#else
    globalParams = new GlobalParams();
#endif
    uMap = globalParams->getTextEncoding();

    Object info;
    if (doc->isOk())
      {
      doc->getDocInfo(&info);
      if (info.isDict())
        {
        title        = getInfoString(info.getDict(), "Title",    uMap);
        subject      = getInfoString(info.getDict(), "Subject",  uMap);
        keywords     = getInfoString(info.getDict(), "Keywords", uMap);
        author       = getInfoString(info.getDict(), "Author",   uMap);
        creator      = getInfoString(info.getDict(), "Creator",  uMap);
        producer     = getInfoString(info.getDict(), "Producer", uMap);
        creationdate = getInfoDate(  info.getDict(), "CreationDate"  );
        moddate      = getInfoDate(  info.getDict(), "ModDate"       );
        info.free();
        }
#ifdef LIBPOPPLER_CATALOG_HAS_STRUCTTREEROOT
      const char *tagged = doc->getStructTreeRoot() ? "yes" : "no";
#else
      const char *tagged = doc->getStructTreeRoot()->isDict() ? "yes" : "no";
#endif
      int pages = doc->getNumPages();
      const char *encrypted = doc->isEncrypted() ? "yes" : "no";
      //  printf("yes (print:%s copy:%s change:%s addNotes:%s)\n",
      //   doc->okToPrint(gTrue) ? "yes" : "no",
      //   doc->okToCopy(gTrue) ? "yes" : "no",
      //   doc->okToChange(gTrue) ? "yes" : "no",
      //   doc->okToAddNotes(gTrue) ? "yes" : "no");

      // print linearization info
      const char *optimized = doc->isLinearized() ? "yes" : "no";

      // print PDF version
#ifdef LIBPOPPLER_PDFDOC_HAS_PDFVERSION
      float pdfversion = doc->getPDFVersion();
#else
      const double pdfversion = doc->getPDFMajorVersion() + 0.1 * doc->getPDFMinorVersion();
#endif


      // print page count
      printf("Pages:          %d\n", doc->getNumPages());

      std::cout << "PDF Info:" << std::endl;
      std::cout << "  Title:          " << title << std::endl;
      std::cout << "  Subject:        " << subject << std::endl;
      std::cout << "  Keywords:       " << keywords << std::endl;
      std::cout << "  Author:         " << author << std::endl;
      std::cout << "  Creator:        " << creator << std::endl;
      std::cout << "  Producer:       " << producer << std::endl;
      std::cout << "  CreationDate:   " << creationdate << std::endl;
      std::cout << "  ModDate:        " << moddate << std::endl;
      std::cout << "  Tagged:         " << tagged << std::endl;
      std::cout << "  Pages:          " << pages << std::endl;
      std::cout << "  Encrypted:      " << encrypted << std::endl;
      //std::cout << "Page size:      " << subject << std::endl;
      std::cout << "  File size:      " << bv->GetLength() << std::endl;
      std::cout << "  Optimized:      " << optimized << std::endl;
      std::cout << "  PDF version:    " << pdfversion << std::endl;
      }
    else
      {
      std::cout << "Problem reading Encapsulated PDF " << std::endl;
      }

#else // GDCM_USE_SYSTEM_POPPLER
    std::cout << "  Encapsulated PDF File" << std::endl;
#endif // GDCM_USE_SYSTEM_POPPLER
    }
  // Do the IOD verification !
  //bool v = defs.Verify( file );
  //std::cerr << "IOD Verification: " << (v ? "succeed" : "failed") << std::endl;

  return 0;
}


int main(int argc, char *argv[])
{
  int c;
  std::string filename;
  std::string xmlpath;
  int resourcespath = 0;
  int verbose = 0;
  int warning = 0;
  int help = 0;
  int recursive = 0;
  int version = 0;
  int debug = 0;
  int error = 0;
  while (1) {
    int option_index = 0;
    static struct option long_options[] = {
        {"input", 1, 0, 0},
        {"recursive", 0, &recursive, 1},
        {"check-deflated", 0, &deflated, 1},
        {"resources-path", 0, &resourcespath, 1},
        {"md5sum", 0, &md5sum, 1},
        {"check-compression", 0, &checkcompression, 1},

        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {0, 0, 0, 0} // required
    };
    static const char short_options[] = "i:rdVWDEhv";
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
          else if( option_index == 3 ) /* resources-path */
            {
            assert( strcmp(s, "resources-path") == 0 );
            assert( xmlpath.empty() );
            xmlpath = optarg;
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

    case 'r':
      recursive = 1;
      break;

    case 'd':
      deflated = 1;
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

  // Debug is a little too verbose
  gdcm::Trace::SetDebug( debug != 0);
  gdcm::Trace::SetWarning( warning != 0);
  gdcm::Trace::SetError( error != 0);
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( verbose != 0);
    gdcm::Trace::SetError( verbose != 0);
    }

  if( !gdcm::System::FileExists(filename.c_str()) )
    {
    return 1;
    }

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
    std::cerr << "Could not load XML file from specified path" << std::endl;
    return 1;
    }

  const gdcm::Defs &defs = g.GetDefs();

  int res = 0;
  if( gdcm::System::FileIsDirectory(filename.c_str()) )
    {
    gdcm::Directory d;
    d.Load(filename, recursive!= 0);
    gdcm::Directory::FilenamesType const &filenames = d.GetFilenames();
    for( gdcm::Directory::FilenamesType::const_iterator it = filenames.begin(); it != filenames.end(); ++it )
      {
      std::cout << "filename: " << *it << std::endl;
      res += ProcessOneFile(*it, defs);
      }
    }
  else
    {
    res += ProcessOneFile( filename, defs );
    }


  return res;
}
