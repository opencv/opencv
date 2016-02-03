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
 * This is an implementation of Application Hosting: DICOM Native Model
 */
#include "gdcmFilename.h"
#include "gdcmReader.h"
#include "gdcmVersion.h"
#include "gdcmFileMetaInformation.h"
#include "gdcmDataSet.h"
#include "gdcmDataElement.h"
#include "gdcmAttribute.h"
#include "gdcmPrivateTag.h"
#include "gdcmValidate.h"
#include "gdcmWriter.h"
#include "gdcmSystem.h"
#include "gdcmDirectory.h"
#include "gdcmCSAHeader.h"
#include "gdcmPDBHeader.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmASN1.h"
#include "gdcmFile.h"
#include "gdcmXMLPrinter.h"
#include "gdcmPrinter.h"

#ifdef GDCM_USE_SYSTEM_LIBXML2
#include <libxml/xmlreader.h>
#endif

#include <string>
#include <iostream>
#include <fstream>

#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include <string.h>

using namespace gdcm;

// This is a very dumb implementation for getData handling
// by default GDCM simply drop the BulkData so we need to store
// the actual BulkData somewhere for proper implementation of getData
class SimpleFileXMLPrinter : public XMLPrinter
{
public:
  void HandleBulkData(const char *uuid, const TransferSyntax & ts,
    const char *bulkdata, size_t bulklen)
    {
    // Store Bulk Data
    std::ofstream out( uuid, std::ios::binary );
    out.write( bulkdata, bulklen );
    out.close();
    std::string tsfn = uuid;
    tsfn += ".ts";
    // Need to store Transfer Syntax for later getData() implementation
    // See Sup118 for details
    const char *tsstring = ts.GetString();
    assert( tsstring );
    std::ofstream out2( tsfn.c_str(), std::ios::binary );
    out2.write( tsstring, strlen(tsstring) );
    out2.close();
    }
};

//Global Variables
int loadBulkData = 0;
int loadTransferSyntax = 0;
TransferSyntax ts;

static void PrintVersion()
{
  std::cout << "gdcmxml: gdcm " << gdcm::Version::GetVersion() << " ";
  const char date[] = "$Date$";
  std::cout << date << std::endl;
}

static void PrintHelp()
{
  PrintVersion();
  std::cout << "Usage: gdcmxml [OPTION]... FILE..." << std::endl;
  std::cout << "Convert a DICOM file into an XML file or vice-versa \n";
  std::cout << "Parameter (required):" << std::endl;
  std::cout << "  -i --input     Filename1" << std::endl;
  std::cout << "  -o --output    Filename2" << std::endl;
  std::cout << "General Options:" << std::endl;
  std::cout << "  -V --verbose        more verbose (warning+error)." << std::endl;
  std::cout << "  -W --warning        print warning info." << std::endl;
  std::cout << "  -D --debug          print debug info." << std::endl;
  std::cout << "  -E --error          print error info." << std::endl;
  std::cout << "  -h --help           print help." << std::endl;
  std::cout << "  -v --version        print version." << std::endl;
  std::cout << "Options for Dicom to XML:" << std::endl;
  std::cout << "  -B --loadBulkData   Loads bulk data into a binary file named \"UUID\"(by default UUID are written)." << std::endl;
  std::cout << "Options for XML to Dicom:" << std::endl;
  std::cout << "  -B --loadBulkData   Loads bulk data from a binary file named as the \"UUID\" in XML file(by default UUID are written)."<< std::endl;
  std::cout << "  -T --TransferSyntax Loads transfer syntax from file (default is LittleEndianImplicit)" << std::endl; 
}

#ifdef GDCM_USE_SYSTEM_LIBXML2

#define CHECK_READER \
  if(ret == -1) \
    assert(0 && "unable to read");

#define READ_NEXT\
  ret = xmlTextReaderRead(reader);\
  CHECK_READER\
  if(xmlTextReaderNodeType(reader) == 14 || xmlTextReaderNodeType(reader) == 13)\
    ret = xmlTextReaderRead(reader);\
  CHECK_READER   

#define CHECK_NAME(value)\
  strcmp((const char*)xmlTextReaderConstName(reader),value)

static void HandleBulkData(const char *uuid, DataElement &de)
  {
  // Load Bulk Data
  if(loadBulkData)
    {
    std::ifstream file( uuid, std::ios::in|std::ios::binary|std::ios::ate );//open file with pointer at file end  
      
    if (file.is_open())
      {
      std::ifstream::pos_type size = file.tellg();    
      char *bulkData = new char [size];
      file.seekg (0, std::ios::beg);
      file.read (bulkData, size);
      file.close();
      
      ByteValue *bv = new ByteValue(bulkData,(int)size);
      de.SetValue(*bv);
      }  
    }
    
  if(loadTransferSyntax)
     {
     std::string tsfn = uuid;
    tsfn += ".ts";
    std::ifstream file( tsfn.c_str(), std::ios::in|std::ios::binary|std::ios::ate );//open file with pointer at file end  
    if (file.is_open())
      {
      std::ifstream::pos_type size = file.tellg();    
      char *tsstring = new char [size];
      file.seekg (0, std::ios::beg);
      file.read (tsstring, size);
      file.close();
      TransferSyntax::TSType tsType = TransferSyntax::GetTSType(tsstring);
      const TransferSyntax ts_temp(tsType);
      
      if(ts_temp.IsValid())
        {        
        ts = tsType;
        }      
      }    
     }  
  }

static void HandlePN(xmlTextReaderPtr reader,DataElement &de)
{
  if(CHECK_NAME("DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15)
    return;//empty element
  else if(!(CHECK_NAME("PersonName") == 0))
    assert(0 && "Invalid XML");
    
  int depth_curr = xmlTextReaderDepth(reader);
  (void)depth_curr;
  int ret;
  std::string name;
  READ_NEXT;
  READ_NEXT;
  while(!(CHECK_NAME("DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15))
    {
    READ_NEXT
    if(xmlTextReaderNodeType(reader) == 3)
      {
      name += (char*)xmlTextReaderConstValue(reader);
      name += "^";
      }
    if((CHECK_NAME("Ideographic") == 0 || CHECK_NAME("Phonetic") == 0)  && xmlTextReaderNodeType(reader) == 1)
      {
      name += "=";
      }
      
    }

  de.SetByteValue( name.c_str(), (uint32_t)name.size() );
  return;
  
  /*
  READ_NEXT//at ByteValue
  
  

  //char *temp_name=new char[500];
  std::string name;
  int count =0;
  //while(!(CHECK_NAME("PersonName") == 0 && xmlTextReaderDepth(reader) == depth_curr && xmlTextReaderNodeType(reader) == 15))
    //{
  if(CHECK_NAME("SingleByte") == 0)
    {
      
      READ_NEXT
      
      if(CHECK_NAME("FamilyName") == 0)
        {
        READ_NEXT
        if(xmlTextReaderNodeType(reader) == 3)
          {
          //append here
          count +=strlen((char*)xmlTextReaderConstValue(reader));
          name += (char*)xmlTextReaderConstValue(reader);
          name += "^";
          }
        READ_NEXT//at </FamilyName>
        READ_NEXT//at </SingleByte>
        //READ_NEXT//may be new name or at </PersonName>
        //if(CHECK_NAME("ByteValue") == 0)
          //READ_NEXT   
        }
      if(CHECK_NAME("GivenName") == 0)
        {
        READ_NEXT
        if(xmlTextReaderNodeType(reader) == 3)
          {
          //append here
          name += (char*)xmlTextReaderConstValue(reader);
          name += "^";
          }
        READ_NEXT
        READ_NEXT
        }
      if(CHECK_NAME("MiddleName") == 0)
        {
        READ_NEXT
        if(xmlTextReaderNodeType(reader) == 3)
          {
          //append here
          name += (char*)xmlTextReaderConstValue(reader);
          name += "^";
          }
        READ_NEXT
        READ_NEXT
        }
      if(CHECK_NAME("NamePrefix") == 0)
        {
        READ_NEXT
        if(xmlTextReaderNodeType(reader) == 3)
          {
          //append here
          name += (char*)xmlTextReaderConstValue(reader);
          name += "^";
          }
        READ_NEXT
        READ_NEXT
        }
      if(CHECK_NAME("NameSuffix") == 0)
        {
        READ_NEXT
        if(xmlTextReaderNodeType(reader) == 3)
          {
          //append here
          name += (char*)xmlTextReaderConstValue(reader);
          name += "^";
          }
        READ_NEXT
        READ_NEXT
        }
      READ_NEXT//starts new or end tag PersonName  
      //name += "=";
    }
  if(CHECK_NAME("Ideographic") == 0)
    {
    
    }  
  if(CHECK_NAME("Phonetic") == 0)
    {
    }  
  
  //Set Value in de
  Element<VR::PN,VM::VM1_n> el;
  el.SetValue(name.c_str(),0);
  de = el.GetAsDataElement();
  
  READ_NEXT//at correct place for Populate DataSet
  return;  
    */
    
}
  
static void HandleSequence(SequenceOfItems *sqi,xmlTextReaderPtr reader,int depth);

static void PopulateDataSet(xmlTextReaderPtr reader,DataSet &DS, int depth, bool SetSQ )
{    
  (void)depth;
   int ret;  
   const char *name = (const char*)xmlTextReaderConstName(reader);
   //printf("%s\n",name);        

#define LoadValueASCII(type) \
  case type: \
        { \
        int count =0; \
        name = (const char*)xmlTextReaderConstName(reader); \
        if(strcmp(name,"DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15)\
          break;\
        char values[10][100] = {"","","","","","","","","",""}; \
        Element<type,VM::VM1_n> el; \
        while(strcmp(name,"Value") == 0) \
          { \
          READ_NEXT \
          if(CHECK_NAME("Value")  == 0)\
          {READ_NEXT;}\
          else{\
          char *value = (char*)xmlTextReaderConstValue(reader); \
          strcpy((char *)values[count++],value); \
          READ_NEXT /*Value ending tag*/ \
          name = (const char*)xmlTextReaderConstName(reader); \
          READ_NEXT \
          name = (const char*)xmlTextReaderConstName(reader); }\
          } \
        assert(CHECK_NAME("DicomAttribute") == 0);\
        el.SetLength( (count) * vr.GetSizeof() ); \
        int total = 0; \
        while(total < count) \
          { \
          el.SetValue(values[total],total); \
          total++; \
          } \
        de = el.GetAsDataElement(); \
        }break

#define LoadValueInteger(type) \
  case type: \
        { \
        int count =0; \
        name = (const char*)xmlTextReaderConstName(reader); \
        if(strcmp(name,"DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15)\
          break;\
        int values[10]; \
        Element<type,VM::VM1_n> el; \
        while(strcmp(name,"Value") == 0) \
          { \
          READ_NEXT \
          char *value_char = (char*)xmlTextReaderConstValue(reader); \
          int nvalue = sscanf(value_char,"%d",&(values[count++]));  \
          assert( nvalue == 1 );  \
          READ_NEXT /*Value ending tag*/ \
          name = (const char*)xmlTextReaderConstName(reader); \
          READ_NEXT \
          name = (const char*)xmlTextReaderConstName(reader); \
          } \
        el.SetLength( (count) * vr.GetSizeof() ); \
        int total = 0; \
        while(total < count) \
          { \
          el.SetValue( (VRToType<VR::type>::Type)(values[total]),total); \
          total++; \
          } \
        de = el.GetAsDataElement(); \
        }break

#define LoadValueFloat(type) \
  case type: \
        { \
        int count =0; \
        name = (const char*)xmlTextReaderConstName(reader); \
        if(strcmp(name,"DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15)\
          break;\
        float values[10]; \
        Element<type,VM::VM1_n> el; \
        while(strcmp(name,"Value") == 0) \
          { \
          READ_NEXT \
          char *value_char = (char*)xmlTextReaderConstValue(reader); \
          sscanf(value_char,"%f",&(values[count++]));  \
          READ_NEXT /*Value ending tag*/ \
          name = (const char*)xmlTextReaderConstName(reader); \
          READ_NEXT \
          name = (const char*)xmlTextReaderConstName(reader); \
          } \
        el.SetLength( (count) * vr.GetSizeof() ); \
        int total = 0; \
        while(total < count) \
          { \
          el.SetValue(values[total],total); \
          total++; \
          } \
        de = el.GetAsDataElement(); \
        }break

#define LoadValueDouble(type) \
  case type: \
        { \
        int count =0; \
        name = (const char*)xmlTextReaderConstName(reader); \
        if(strcmp(name,"DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15)\
          break;\
        double values[10]; \
        Element<type,VM::VM1_n> el; \
        while(strcmp(name,"Value") == 0) \
          { \
          READ_NEXT \
          char *value_char = (char*)xmlTextReaderConstValue(reader); \
          sscanf(value_char,"%lf",&(values[count++]));  \
          READ_NEXT/*Value ending tag*/ \
          name = (const char*)xmlTextReaderConstName(reader); \
          READ_NEXT \
          name = (const char*)xmlTextReaderConstName(reader); \
          } \
        el.SetLength( (count) * vr.GetSizeof() ); \
        int total = 0; \
        while(total < count) \
          { \
          el.SetValue(values[total],total); \
          total++; \
          } \
        de = el.GetAsDataElement(); \
        }break


#define LoadValueAT(type)\
  case type: \
    		{ \
				int count =0; \
      	name = (const char*)xmlTextReaderConstName(reader);\
      	if(strcmp(name,"DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15)\
      		break;\
      	Tag tags[10];\
      	unsigned int group = 0, element = 0;\
      	Element<type,VM::VM1_n> el;\
    		while(strcmp(name,"Value") == 0)\
    			{\
    			READ_NEXT\
    			char *value = (char*)xmlTextReaderConstValue(reader); \
    			if( sscanf(value, "(%04x,%04x)", &group , &element) != 2 )\
      			{\
      				gdcmDebugMacro( "Problem reading AT: ");\
      			} \
      		tags[count].SetGroup( (uint16_t)group );\
    			tags[count].SetElement( (uint16_t)element );count++;\
    			READ_NEXT/*Value ending tag*/ \
    			name = (const char*)xmlTextReaderConstName(reader); \
    			READ_NEXT \
    			name = (const char*)xmlTextReaderConstName(reader); \
    			}\
    		el.SetLength( (count) * vr.GetSizeof() ); \
    		int total = 0; \
    		while(total < count) \
    			{ \
    			el.SetValue(tags[total],total); \
    			total++; \
    			} \
    		de = el.GetAsDataElement();\
    		}break
        
          
   while( xmlTextReaderDepth(reader)!=0 )
    {
    if(SetSQ && (xmlTextReaderNodeType(reader) == 15) && CHECK_NAME("Item") == 0 )
      return;
     if(CHECK_NAME("DicomAttribute") == 0)
      {
      DataElement de;
      
      /* Reading Tag */
      char *tag_read =(char *)xmlTextReaderGetAttribute(reader,(const unsigned char*)"tag");
      Tag t;
      if(!t.ReadFromContinuousString((const char *)tag_read))
        assert(0 && "Invalid Tag!");
      
      /* Reading VR */
      char vr_read[3] = "";
      strcpy(vr_read, (const char *)xmlTextReaderGetAttribute(reader,(const unsigned char*)"vr"));
      vr_read[2]='\0';
      const gdcm::VR vr = gdcm::VR::GetVRType(vr_read);
      
      READ_NEXT  /* should be at Value tag or BulkData tag or Item Tag */
      
      /* Load Value */
      switch(vr)
        {
        
        LoadValueAT(VR::AT);
        LoadValueASCII(VR::AE);
        LoadValueASCII(VR::AS);
        LoadValueASCII(VR::CS);
        LoadValueASCII(VR::DA);
        LoadValueFloat(VR::DS);
        LoadValueASCII(VR::DT);
        LoadValueInteger(VR::IS);
        LoadValueASCII(VR::LO);
        LoadValueASCII(VR::LT);        
        LoadValueASCII(VR::SH);
        LoadValueASCII(VR::ST);
        LoadValueASCII(VR::TM);
        LoadValueASCII(VR::UI);
        LoadValueASCII(VR::UT);
        LoadValueInteger(VR::SS);
        LoadValueInteger(VR::UL);
        LoadValueInteger(VR::SL);
        LoadValueInteger(VR::US);
        LoadValueFloat(VR::FL);
        LoadValueDouble(VR::FD);
        case VR::SQ:
          {
          SequenceOfItems *sqi = new SequenceOfItems();
          HandleSequence(sqi,reader,xmlTextReaderDepth(reader));
          de.SetValue(*sqi);
          }break;
        
        case VR::PN:
          {
          //Current node must be Person Name
          HandlePN(reader,de);
          }break;
        
        case VR::OF:
        case VR::OB:
        case VR::OW:
          {
          //Presently should be at BulkData
          assert(((CHECK_NAME("BulkData")) == 0));
          char * uuid = (char *)xmlTextReaderGetAttribute(reader,(const unsigned char*)"uuid");
          HandleBulkData(uuid,de);
          READ_NEXT
          }break;    
        
        case VR::UN:
          {
          int depth_UN=xmlTextReaderDepth(reader);
          while(!(CHECK_NAME("DicomAttribute") == 0 && xmlTextReaderNodeType(reader) == 15 && (depth_UN-1)  == xmlTextReaderDepth(reader)))
            {READ_NEXT}
          //assert(0 && "UN not Handled yet");
          }break;          
        default:
          assert(0 && "Unknown VR");  
        };
      
      /*Modify de before insert*/
      
      de.SetTag(t);    
      
      DS.Insert(de);
      
      READ_NEXT // To next Element
            
      }  
           
     }
}

static void HandleSequence(SequenceOfItems *sqi, xmlTextReaderPtr reader,int depth)
{
  int ret;
  while(!(  CHECK_NAME("DicomAttribute") == 0  && xmlTextReaderDepth(reader) == (depth - 1)  &&  xmlTextReaderNodeType(reader) == 15 )  )
    {
    gdcmDebugMacro( "HandleSequence (while loop)" );
    if(   CHECK_NAME("Item") == 0  &&  xmlTextReaderDepth(reader) == depth && xmlTextReaderNodeType(reader) == 1)
      {
      //at Item
      READ_NEXT  
      //assert(0 && "Hi1");
      //at DicomAtt
      if(   CHECK_NAME("DicomAttribute") == 0  &&  xmlTextReaderDepth(reader) == (depth + 1) && xmlTextReaderNodeType(reader) == 1)
        {
        //start of an item
        //Create Nested DataSet
        //assert(0 && "Hi2");
        Item *item = new Item();
        DataSet *NestedDS = new DataSet() ;
        PopulateDataSet(reader,*NestedDS,xmlTextReaderDepth(reader),true);
        item->SetNestedDataSet(*NestedDS);
        sqi->AddItem(*item);
        //assert(0 && "Hi3");
        
        }        
      else
        assert("Empty Item or Invalid XML");
      
      READ_NEXT    
      }
    else
      assert("Expected Item");  
    }
}

static void WriteDICOM(xmlTextReaderPtr reader, gdcm::Filename file2)
{  
  int ret;
  
  READ_NEXT
  
  READ_NEXT   // at first element "DicomAttribute"
  
  //populate DS
  DataSet DS;
  if(xmlTextReaderDepth(reader) == 1 && strcmp((const char*)xmlTextReaderConstName(reader),"DicomAttribute") == 0)  
    PopulateDataSet(reader,DS,1,false);
  
  //DataElement de;
  //Tag t(0xFFFe,0xE0DD);
  //de.SetTag(t);
  //DS.Insert(de);
  //add to File 
  
  //store in heap
  File *F = new File();
  F->SetDataSet(DS);
  
  //Validate - possibly from gdcmValidate Class
   FileMetaInformation meta = F->GetHeader();
   meta.SetDataSetTransferSyntax(ts);
  F->SetHeader(meta);  
  
  //Validate - possibly from gdcmValidate Class
  
  //Validate V;
  //V.SetFile(F);
  //V.Validation();
  //F = V.GetValidatedFile(); 
  //add to Writer
  
  if(!file2.IsEmpty())
    {
    Writer W;    
    W.SetFileName(file2.GetFileName());
    W.SetFile(*F);      

    //finally write to file
    W.Write(); 
    }
  else
    {
    Printer printer;
    printer.SetFile ( *F );
    printer.SetColor(1);
    printer.Print( std::cout );
    }      
}

static void XMLtoDICOM(gdcm::Filename file1, gdcm::Filename file2)
{
  xmlTextReaderPtr reader;  
  FILE *in;
  char *buffer;
  size_t numBytes;
  in = fopen(file1.GetFileName(), "r");
  
  if(in == NULL)
    return ;
    
  fseek(in, 0L, SEEK_END);
  numBytes = ftell(in);
  fseek(in, 0L, SEEK_SET);
  buffer = (char*)calloc(numBytes, sizeof(char));

  if(buffer == NULL)
    return ;

  size_t ret = fread(buffer, sizeof(char), numBytes, in);
  if( numBytes != ret )
    {
    // FIXME how to return error code ?
    return;
    }
  fclose(in);
  reader = xmlReaderForMemory  (buffer, (int)numBytes, NULL, NULL, 0);
  //reader = xmlReaderForFile(filename, "UTF-8", 0); Not Working!!
  if (reader != NULL) 
    {
    WriteDICOM(reader, file2);
    } 
  else 
    {
    fprintf(stderr, "Unable to open %s\n", file1.GetFileName());
    }
}
#endif // GDCM_USE_SYSTEM_LIBXML2

int main (int argc, char *argv[])
{
  int c;
  //int digit_optind = 0;
  gdcm::Filename file1;
  gdcm::Filename file2;  
//  int loadTransferSyntax = 0;
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
        {"loadBulkData", 0, &loadBulkData, 1},
        {"TransferSyntax", 0, &loadTransferSyntax, 1},
        {"verbose", 0, &verbose, 1},
        {"warning", 0, &warning, 1},
        {"debug", 0, &debug, 1},
        {"error", 0, &error, 1},
        {"help", 0, &help, 1},
        {"version", 0, &version, 1},
        {0, 0, 0, 0} // required
    };
    static const char short_options[] = "i:o:BTVWDEhv";
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
        if (optarg)
          {
          if( option_index == 0 ) /* input */
            {
            assert( strcmp(s, "input") == 0 );
            assert( file1.IsEmpty() );
            file1 = optarg;
            }
          }
        }
      break;

    case 'i':
      //printf ("option i with value '%s'\n", optarg);
      assert( file1.IsEmpty() );
      file1 = optarg;
      break;

    case 'o':
      assert( file2.IsEmpty() );
      file2 = optarg;
      break;

    case 'B':
      loadBulkData = 1;
      break;
    
    case 'T':
      loadTransferSyntax = 1;
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
    int v = argc - optind;
    if( v == 2 )
      {
      file1 = argv[optind];
      file2 = argv[optind+1];
      }
    else if( v == 1 )
      {
      file1 = argv[optind];
      }
    else
      {
      PrintHelp();
      return 1;
      }
    }

  if( file1.IsEmpty() )
    {
    PrintHelp();
    return 1;
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
    
  gdcm::Trace::SetDebug( debug != 0);
  gdcm::Trace::SetWarning( warning != 0);
  gdcm::Trace::SetError( error != 0);  
  // when verbose is true, make sure warning+error are turned on:
  if( verbose )
    {
    gdcm::Trace::SetWarning( verbose != 0);
    gdcm::Trace::SetError( verbose!= 0);
    }
 
  const char *file1extension = file1.GetExtension();
  //const char *file2extension = file2.GetExtension();

  if(gdcm::System::StrCaseCmp(file1extension,".xml") != 0)// by default we assume it is a DICOM file-- as no extension is required for it
    {
    gdcm::Reader reader;
    reader.SetFileName( file1.GetFileName() );
    bool success = reader.Read();
    if( !success )//!ignoreerrors )
      {
      std::cerr << "Failed to read: " << file1 << std::endl;
      return 1;
      }
    
    if(loadBulkData)
    {
      SimpleFileXMLPrinter printer;      
      printer.SetFile ( reader.GetFile() );
      if( file2.IsEmpty() )
        {
        printer.Print( std::cout );
        }
      else
        {
        std::ofstream outfile;
        outfile.open (file2.GetFileName());
        printer.Print( outfile );
        outfile.close();
        }
    } 
    else
    {
      XMLPrinter printer;
      printer.SetFile ( reader.GetFile() );
      if( file2.IsEmpty() )
        {
        printer.Print( std::cout );
        }
      else
        {
        std::ofstream outfile;
        outfile.open (file2.GetFileName());
        printer.Print( outfile );
        outfile.close();
        }
    }       
    return 0;
    }
  else
    {
#ifdef GDCM_USE_SYSTEM_LIBXML2
    /*
     * This initializes the library and checks potential ABI mismatches
     * between the version it was compiled for and the actual shared
     * library used.
     */
    LIBXML_TEST_VERSION

    XMLtoDICOM(file1,file2);

    /*
     * Cleanup function for the XML library.
     */
    xmlCleanupParser();
    /*
     * This is to debug memory for regression tests.
     */
    xmlMemoryDump();
#else
    printf("\nPlease configure Cmake options with GDCM_USE_SYSTEM_LIBXML2 as ON and compile!\n");    
#endif
    }
}
