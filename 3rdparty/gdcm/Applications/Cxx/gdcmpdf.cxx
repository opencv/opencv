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
 */
#include "gdcmVersion.h"
#include "gdcmUIDGenerator.h"
#include "gdcmWriter.h"
#include "gdcmAttribute.h"
#include "gdcmSystem.h"

#ifdef GDCM_USE_SYSTEM_POPPLER
#include <poppler/poppler-config.h>
#include <poppler/PDFDoc.h>
#include <poppler/UnicodeMap.h>
#include <poppler/PDFDocEncoding.h>
#include <poppler/GlobalParams.h>
#endif

#include <string>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

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

static std::string getInfoString(Dict *infoDict, const char *key, UnicodeMap *uMap, GBool & unicode)
{
  Object obj;
  GooString *s1;
  GBool isUnicode = gFalse;
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
  unicode = unicode || isUnicode;
  return out;
}

static void PrintVersion()
{
  std::cout << "gdcmpdf: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmpdf [OPTION]... FILE..." << std::endl;
  std::cout << "Convert a PDF file to DICOM/PDF\n";
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input     PDF filename" << std::endl;
  std::cout << "  -o --output    DICOM filename" << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose   more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning   print warning info." << std::endl;
  std::cout << "  -D --debug     print debug info." << std::endl;
  std::cout << "  -E --error     print error info." << std::endl;
  std::cout << "  -h --help      print help." << std::endl;
  std::cout << "  -v --version   print version." << std::endl;
}

int main (int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;

  std::string filename;
  std::string outfilename;
  int verbose = 0;
  int warning = 0;
  int debug = 0;
  int error = 0;
  int help = 0;
  int version = 0;
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
        {"output", 1, 0, 0},
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {0, 0, 0, 0} // required
    };
    static const char short_options[] = "i:o:VWDEhv";
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
    int v = argc - optind;
    if( v == 2 )
      {
      filename = argv[optind];
      outfilename = argv[optind+1];
      }
    else
      {
      PrintHelp();
      return 1;
      }
    }
  if( filename.empty() || outfilename.empty() )
    {
    PrintHelp();
    return 1;
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

  GooString *ownerPW, *userPW;
  GooString *fileName;
  PDFDoc *doc;
  Object info;
  UnicodeMap *uMap;
  ownerPW = NULL;
  userPW = NULL;
#ifdef LIBPOPPLER_GLOBALPARAMS_CSTOR_HAS_PARAM
  globalParams = new GlobalParams(0);
#else
  globalParams = new GlobalParams();
#endif
  uMap = globalParams->getTextEncoding();

  //const char *filename = argv[1];
  if( !gdcm::System::FileExists(filename.c_str()) )
    {
    return 1;
    }
  // get length of file:
  size_t length = gdcm::System::FileSize(filename.c_str());
  // PDF doc is stored in an OB element, check that 32bits length is fine:
  if( length > gdcm::VL::GetVL32Max() )
    {
    return 1;
    }

  //const char *outfilename = argv[2];
  fileName = new GooString( filename.c_str() );
  //ownerPW = new GooString( "toto" );
  Object obj;

  obj.initNull();
  doc = new PDFDoc(fileName, ownerPW, userPW);

  if (doc->isEncrypted())
    {
    std::string password;
    std::cout << "Enter password:" << std::endl;
    //  http://www.daniweb.com/code/snippet1174.html
    std::cin >> password;
    //std::cout << "Enter password:" << password << std::endl;
/*
#include <termios.h>
#include <unistd.h>

int mygetch(void)
{
struct termios oldt,
newt;
int ch;
tcgetattr( STDIN_FILENO, &oldt );
newt = oldt;
newt.c_lflag &= ~( ICANON | ECHO );
tcsetattr( STDIN_FILENO, TCSANOW, &newt );
ch = getchar();
tcsetattr( STDIN_FILENO, TCSANOW, &oldt );
return ch;

http://msdn.microsoft.com/en-us/library/078sfkak(VS.80).aspx
}
 */
    ownerPW = new GooString( password.c_str() );
    doc = new PDFDoc(fileName, ownerPW, userPW);
    }

  std::string title;
  std::string subject;
  std::string keywords;
  std::string author;
  std::string creator;
  std::string producer;
  std::string creationdate;
  std::string moddate;

  GBool isUnicode = gFalse;
  if (doc->isOk())
    {
    doc->getDocInfo(&info);
    if (info.isDict())
      {
      title        = getInfoString(info.getDict(), "Title",    uMap, isUnicode);
      subject      = getInfoString(info.getDict(), "Subject",  uMap, isUnicode);
      keywords     = getInfoString(info.getDict(), "Keywords", uMap, isUnicode);
      author       = getInfoString(info.getDict(), "Author",   uMap, isUnicode);
      creator      = getInfoString(info.getDict(), "Creator",  uMap, isUnicode);
      producer     = getInfoString(info.getDict(), "Producer", uMap, isUnicode);
      creationdate = getInfoDate(  info.getDict(), "CreationDate"  );
      moddate      = getInfoDate(  info.getDict(), "ModDate"       );
      info.free();
      }
    }

  gdcm::Writer writer;
  gdcm::DataSet &ds = writer.GetFile().GetDataSet();
{
  gdcm::DataElement de( gdcm::Tag(0x42,0x11) );
  de.SetVR( gdcm::VR::OB );
  std::ifstream is;
  is.open (filename.c_str(), std::ios::binary );

  char *buffer = new char [length];

  // read data as a block:
  is.read (buffer,length);
  is.close();

  de.SetByteValue( buffer, (uint32_t)length );
  delete[] buffer;

  gdcm::FileMetaInformation &fmi = writer.GetFile().GetHeader();
  gdcm::TransferSyntax ts = gdcm::TransferSyntax::ExplicitVRLittleEndian;
  fmi.SetDataSetTransferSyntax( ts );
  ds.Insert( de );
}


{
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
  const size_t timelen = 6 + 1 + 6; // time + milliseconds
  (void)timelen;
    {
    gdcm::Attribute<0x8,0x30> at;
    at.SetValue( date+datelen );
    ds.Insert( at.GetAsDataElement() );
    //gdcm::DataElement de( gdcm::Tag(0x0008,0x0030) );
    // Do not copy the whole cstring:
    //de.SetByteValue( date+datelen, timelen );
    //de.SetVR( gdcm::Attribute<0x0008,0x0030>::GetVR() );
    //ds.Insert( de );
    }

}

  gdcm::UIDGenerator uid;
{
    const char *sop = uid.Generate();
    gdcm::DataElement de( gdcm::Tag(0x0008,0x0018) );
    de.SetByteValue( sop, (uint32_t)strlen(sop) );
    de.SetVR( gdcm::Attribute<0x0008, 0x0018>::GetVR() );
    ds.Insert( de );
}

  gdcm::MediaStorage ms = gdcm::MediaStorage::EncapsulatedPDFStorage;
    {
    gdcm::DataElement de( gdcm::Tag(0x0008, 0x0016) );
    const char* msstr = gdcm::MediaStorage::GetMSString(ms);
    de.SetByteValue( msstr, (uint32_t)strlen(msstr) );
    de.SetVR( gdcm::Attribute<0x0008, 0x0016>::GetVR() );
    ds.Insert( de );
    }

  gdcm::FileMetaInformation::SetSourceApplicationEntityTitle( "gdcmpdf" );

  char date[22];
  const size_t datelen = 8;
  bool b = gdcm::System::GetCurrentDateTime(date); (void)b;
  //std::cout << date << std::endl;

{
  gdcm::Attribute<0x0008, 0x0005> at;
  const char s[] = "ISO_IR 100";
  const char s_unicode[] = "ISO_IR 192";
  at.SetNumberOfValues( 1 );
  if( isUnicode )
  at.SetValue( s_unicode );
  else
  at.SetValue( s );
  ds.Insert( at.GetAsDataElement() );
}
{
  gdcm::Attribute<0x0008, 0x0012> at;
  std::string tmp( date, datelen );
  at.SetValue( tmp.c_str() );
  ds.Insert( at.GetAsDataElement() );
}

  const size_t timelen = 6 + 1 + 6; // TM + milliseconds
{
  gdcm::Attribute<0x0008, 0x0013> at;
  std::string tmp( date+datelen, timelen);
  at.SetValue( tmp.c_str() );
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0020) DA (no value available)                     #   0, 0 StudyDate
{
  gdcm::Attribute<0x0008, 0x0020> at;
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0023) DA (no value available)                     #   0, 0 ContentDate
{
  gdcm::Attribute<0x0008, 0x0023> at;
  std::string tmp( creationdate.c_str(), datelen );
  at.SetValue( tmp.c_str() );
  ds.Insert( at.GetAsDataElement() );
}
//(0008,002a) DT (no value available)                     #   0, 0 AcquisitionDatetime
{
  gdcm::Attribute<0x0008, 0x002a> at;
  time_t studydatetime = gdcm::System::FileTime( filename.c_str() );
  char date2[22];
  gdcm::System::FormatDateTime(date2, studydatetime);
  at.SetValue( date2 );
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0030) TM (no value available)                     #   0, 0 StudyTime
{
  gdcm::Attribute<0x0008, 0x0030> at;
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0033) TM (no value available)                     #   0, 0 ContentTime
{
  gdcm::Attribute<0x0008, 0x0033> at;
  std::string tmp( creationdate.c_str() + datelen, timelen);
  at.SetValue( tmp.c_str() );
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0050) SH (no value available)                     #   0, 0 AccessionNumber
{
  gdcm::Attribute<0x0008, 0x0050> at;
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0060) CS [DOC]                                     #   2, 1 Modality
{
  gdcm::Attribute<0x0008, 0x0060> at;
  at.SetValue( "DOC " );
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0064) CS [WSD]                                    #   4, 1 ConversionType
{
  gdcm::Attribute<0x0008, 0x0064> at;
  at.SetValue( "WSD" );
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0070) LO (no value available)                     #   0, 0 Manufacturer
{
  gdcm::Attribute<0x0008, 0x0070> at;
  at.SetValue( creator.c_str() );
  ds.Insert( at.GetAsDataElement() );
}
//(0008,0090) PN (no value available)                     #   0, 0 ReferringPhysiciansName
{
  gdcm::Attribute<0x0008, 0x0090> at;
  ds.Insert( at.GetAsDataElement() );
}

// In past DICOM implementation there used to be those neat tags:
// (0088,0904) Topic Title TopicTitle LO 1 RET
// (0088,0906) Topic Subject TopicSubject ST 1 RET
// (0088,0910) Topic Author TopicAuthor LO 1 RET
// (0088,0912) Topic Keywords TopicKeywords LO 1-32 RET
// However they are now deprecated...

//(0010,0010) PN (no value available)                     #   0, 0 PatientsName
{
  gdcm::Attribute<0x0010, 0x0010> at;
  at.SetValue( author.c_str() );
  ds.Insert( at.GetAsDataElement() );
}
//(0010,0020) LO (no value available)                     #   0, 0 PatientID
{
  gdcm::Attribute<0x0010, 0x0020> at;
  ds.Insert( at.GetAsDataElement() );
}
//(0010,0030) DA (no value available)                     #   0, 0 PatientsBirthDate
{
  gdcm::Attribute<0x0010, 0x0030> at;
  ds.Insert( at.GetAsDataElement() );
}
//(0010,0040) CS (no value available)                     #   0, 0 PatientsSex
{
  gdcm::Attribute<0x0010, 0x0040> at;
  ds.Insert( at.GetAsDataElement() );
}
{
  gdcm::Attribute<0x0018, 0x1020> at;
  at.SetNumberOfValues( 1 );
  at.SetValue( producer.c_str() );
  ds.Insert( at.GetAsDataElement() );
}
//(0020,000d) UI [1.2.276.0.7230010.3.1.4.8323329.511.1228064157.1] #  48, 1 StudyInstanceUID
{
  gdcm::Attribute<0x0020, 0x000d> at;
  at.SetValue( uid.Generate() );
  ds.Insert( at.GetAsDataElement() );
}
//(0020,000e) UI [1.2.276.0.7230010.3.1.4.8323329.511.1228064157.2] #  48, 1 SeriesInstanceUID
{
  gdcm::Attribute<0x0020, 0x000e> at;
  at.SetValue( uid.Generate() );
  ds.Insert( at.GetAsDataElement() );
}
//(0020,0010) SH (no value available)                     #   0, 0 StudyID
{
  gdcm::Attribute<0x0020, 0x0010> at;
  ds.Insert( at.GetAsDataElement() );
}
//(0020,0011) IS [1]                                      #   2, 1 SeriesNumber
{
  gdcm::Attribute<0x0020, 0x0011> at = { 1 };
  ds.Insert( at.GetAsDataElement() );
}
//(0020,0013) IS [1]                                      #   2, 1 InstanceNumber
{
  gdcm::Attribute<0x0020, 0x0013> at = { 1 };
  ds.Insert( at.GetAsDataElement() );
}
//(0028,0301) CS [YES]                                    #   4, 1 BurnedInAnnotation
{
  gdcm::Attribute<0x0028, 0x0301> at;
  at.SetValue( "YES" );
  ds.Insert( at.GetAsDataElement() );
}
//(0040,a043) SQ (Sequence with explicit length #=0)      #   0, 1 ConceptNameCodeSequence
//(fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
{
  gdcm::Attribute<0x0040, 0xa043> at;
  gdcm::DataElement de( at.GetTag() );
  de.SetVR( at.GetVR() );
  //ds.Insert( at.GetAsDataElement() );
  ds.Insert( de );
}
//(0042,0010) ST (no value available)                     #   0, 0 DocumentTitle
{
  gdcm::Attribute<0x0042, 0x0010> at;
  at.SetValue( title.c_str() );
  ds.Insert( at.GetAsDataElement() );
}
//(0042,0011) OB 25\50\44\46\2d\31\2e\34\0a\25\e7\f3\cf\d3\0a\32\34\35\38\38\20\30... # 6861900, 1 EncapsulatedDocument
//(0042,0012) LO [application/pdf]                        #  16, 1 MIMETypeOfEncapsulatedDocument
{
  gdcm::Attribute<0x0042, 0x0012> at;
  at.SetValue( "application/pdf" );
  ds.Insert( at.GetAsDataElement() );
}


  writer.SetFileName( outfilename.c_str() );
  if( !writer.Write() )
  {
  return 1;
  }

  return 0;
}
