/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmDictConverter.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <assert.h>
#include <ctype.h> // isdigit
#include <string.h> // for strncmp
#include <stdio.h> // for sscanf

namespace gdcm
{

// PIMPL Implementation
struct DictConverterInternal
{
  // Filename to the .dic file
  std::string InputFilename;
  // Path to the output file (either XML or CXX code)
  std::string OutputFilename;
  //
  std::ifstream InputStream;
  std::ofstream OutputStream;

  std::string DictName;
};


DictConverter::DictConverter()
{
  Internal = new DictConverterInternal;
  Internal->DictName = "Default";
  OutputType = DICT_DEFAULT;
}

DictConverter::~DictConverter()
{
  delete Internal;
}

const std::string &DictConverter::GetDictName() const
{
  return Internal->DictName;
}

void DictConverter::SetDictName(const char *name)
{
  Internal->DictName = name;
}

void DictConverter::SetInputFileName(const char* filename)
{
  Internal->InputFilename = filename;
}

const std::string &DictConverter::GetInputFilename() const
{
  return Internal->InputFilename;
}

void DictConverter::SetOutputFileName(const char* filename)
{
  Internal->OutputFilename = filename;
}

const std::string &DictConverter::GetOutputFilename() const
{
  return Internal->OutputFilename;
}

// \TODO this code reexecute...
void DictConverter::Convert()
{
  // Open files:
  std::ifstream &from = Internal->InputStream; //(filename, std::ios::binary);
  // Keep binary mode !!!
  from.open( Internal->InputFilename.c_str(), std::ios::binary);
  std::ofstream &into = Internal->OutputStream; //(outfilename);
  into.open( Internal->OutputFilename.c_str(), std::ios::binary );

  WriteHeader();

  std::string line;
  std::ostringstream os;
  while(std::getline(from, line))
    {
    if( !line.empty() )
      {
      std::string::iterator e(line.end()-1);
      if( *e == '\r' ) line.erase(e);
      }
    if( *(line.c_str()) == '\0' )
      {
      // If an empty line is found then a new group starts
      AddGroupLength();
      continue;
      }
    if( line[2] == 'x' && line[3] == 'x' )
      {
      if(line[0] == '5')
        {
        uint16_t start = 0x5000;
        uint16_t end   = 0x50ff;
        for(uint16_t i= start; i<=end; i+=2)
          {
          std::string s1 = line.c_str()+4;
          os.str("");
          os << std::hex << i << s1;
          if( OutputType == DICT_XML )
            {
            ConvertToXML(os.str().c_str(), s1);
            }
          else
            {
            ConvertToCXX(os.str().c_str(), s1);
            }
          into << s1 << std::endl;
          }
        }
      else if(line[0] == '6')
        {
        uint16_t start = 0x6000;
        uint16_t end   = 0x60ff;
        for(uint16_t i= start; i<=end; i+=2)
          {
          std::string s1 = line.c_str()+4;
          os.str("");
          os << std::hex << i << s1;
          if( OutputType == DICT_XML )
            {
            ConvertToXML(os.str().c_str(), s1);
            }
          else
            {
            ConvertToCXX(os.str().c_str(), s1);
            }
          into << s1 << std::endl;
          }
        }
      else
        assert(0);
      }
    else if ( line[7] == 'x' && line[8] == 'x' )
      {
      if( line[5] == '3' && line[6] == '1' )
        {
        uint16_t start = 0x3100;
        uint16_t end   = 0x31ff;
        for(uint16_t i= start; i<=end; ++i)
          {
          std::string s1(line.c_str(), line.c_str()+5);
          std::string s = line.c_str()+9;
          os.str("");
          os << s1 << std::hex << i << s;
          if( OutputType == DICT_XML )
            {
            ConvertToXML(os.str().c_str(), s);
            }
          else
            {
            ConvertToCXX(os.str().c_str(), s);
            }
          into << s << std::endl;
          }
        }
      else
        assert(0);
      }
    else
      {
      std::string s;
      if( OutputType == DICT_XML )
        {
        ConvertToXML(line.c_str(), s);
        }
      else
        {
        ConvertToCXX(line.c_str(), s);
        }
      into << s << std::endl;
      }
    }

  WriteFooter();
}

bool DictConverter::ReadVR(const char *raw, VR::VRType &type)
{
  int i = 0;
  while( !isdigit(*(raw+i)))
    ++i;
  std::string vm(raw, raw+i-1);
  type = VR::GetVRType(vm.c_str());
  assert( type != VR::VR_END );
  return true;
}

bool DictConverter::ReadVM(const char *raw, VM::VMType &type)
{
  char vm[8];
  int r = sscanf(raw, "%s", vm);
  assert( r == 1 );
  type = VM::GetVMType(vm);
  assert( type != VM::VM_END );
  return true;
}

bool DictConverter::Readuint16(const char *raw, uint16_t &ov)
{
  unsigned int v;
  int r = sscanf(raw, "%04x", &v);
  assert( r == 1 && "Wrong Value read for uint16");
  char sv[4+1];
  r = sprintf(sv, "%04x", v);
  assert( r == 4 && "Wrong Value printed for uint16");
  assert( strncmp(raw, sv, 4) == 0 );
  ov = v;
  return true;
}

void DictConverter::WriteHeader()
{
  std::ofstream &of = Internal->OutputStream;
  const std::string &name = Internal->DictName;
  of << "#ifndef __gdcm" << name << "Dicts_cxx\n";
  of << "#define __gdcm" << name << "Dicts_cxx\n\n";
  of << "#include \"gdcmVR.h\"\n";
  of << "#include \"gdcmDict.h\"\n";
  of << "#include \"gdcmDictEntry.h\"\n\n";
  of << "namespace gdcm\n{\n\n";
  of << "typedef struct\n{\n";
  of << "  uint16_t group;\n";
  of << "  uint16_t element;\n";
  if( OutputType == DICT_DEBUG )
    {
    of << "  const char *vr;\n";
    of << "  const char *vm;\n";
    }
  else
    {
    of << "  VR::VRType vr;\n";
    of << "  VM::VMType vm;\n";
    }
  of << "  const char *name;\n";
  of << "} DICT_ENTRY;\n\n";
  of << "static const DICT_ENTRY " << name << "DataDict [] = {\n";
}

void DictConverter::WriteFooter()
{
  std::ofstream &of = Internal->OutputStream;
  const std::string &name = Internal->DictName;
  if(OutputType == DICT_DEBUG )
    of << "   {0,0,0,0,0}\n";
  else
    of << "   {0,0,VR::VR_END,VM::VM_END,0}\n";
  of << "};\n\n";
  of << "void Dict::Fill" << name << "DataDict()\n";
  of << "{\n";
  of << "   unsigned int i = 0;\n";
  of << "   DICT_ENTRY n = " << name << "DataDict[i];\n";
  of << "   while( n.name != 0 )\n";
  of << "   {  \n";
  of << "      Tag t(n.group, n.element);\n";
  of << "      DictEntry e( n.name, n.vr, n.vm );\n";
  of << "      AddDictEntry( t, e );\n";
  of << "      n = " << name << "DataDict[++i];\n";
  of << "   }\n";
  of << "   //Tag t(0, 0);\n";
  of << "   //DictEntry e( \"\", (VR::VRType)0, (VM::VMType)0);\n";
  of << "   //AddDictEntry( t, e );\n";
  of << "}\n\n";
  of << "} //end gdcm namespace\n";
  of << "\n#endif\n";
}

// Takes as input a raw text file, and converts it into a xml line
// Template is:
// From
// 0000 0000 UL 1 Group Length
// into:
// <DataElement Tag="0000,0000" VR="UL" VM="1" Name="Group Length">
bool DictConverter::ConvertToXML(const char *raw, std::string &cxx)
{
  uint16_t group;
  uint16_t element;
  VR::VRType vr;
  VM::VMType vm;
  Readuint16(raw, group);
  assert( !(group%2) );
  Readuint16(raw+5, element);
  ReadVR(raw+10, vr);
  int len = 11+strlen(VR::GetVRString(vr));
  ReadVM(raw+len, vm);
  len += strlen(VM::GetVMString(vm))+1;
  std::ostringstream os;
  os << "  <DataElement Tag=\"" << std::hex << std::setw(4)
    << std::setfill('0') << group << ","
    << std::setw(4) << std::setfill('0')
    << element << "\" VR=\"" << vr << "\" VM=\"" << vm << "\"\tName=\""
    << (raw+len) << "\">";
  cxx = os.str();
  return true;
}

// Takes as input a raw text file, and converts it into a c++ line
// Template is:
// From
// 0000 0000 UL 1 Group Length
// into:
// {0x0000, 0x0000, "UL" , "1" , "Group Length"},
bool DictConverter::ConvertToCXX(const char *raw, std::string &cxx)
{
  uint16_t group;
  uint16_t element;
  VR::VRType vr;
  VM::VMType vm;
  Readuint16(raw, group);
  //assert( !(group%2) );
  //
  Readuint16(raw+5, element);
  ReadVR(raw+10, vr);
  int len = 11+strlen(VR::GetVRString(vr));
  ReadVM(raw+len, vm);
  len += strlen(VM::GetVMString(vm))+1;
  std::ostringstream os;
  os << "   {0x" << std::hex << std::setw(4) << std::setfill('0')
    << group << ", 0x"
    << std::setw(4) << std::setfill('0');
  if( OutputType == DICT_DEBUG )
    {
    os << element << ", \"" << vr << "\" , \"" << vm << "\" , \""
      << (raw+len) << "\"}, ";
    }
  else
    {
    os << element << std::dec << ",(VR::VRType)" << (int)vr
      << ",(VM::VMType)" << (int)vm << ",\"" << (raw+len) << "\"},";
    }
  cxx = os.str();
  return true;
}


void DictConverter::AddGroupLength()
{
  std::ifstream &from = Internal->InputStream;
  std::ofstream &into = Internal->OutputStream;

  std::streampos p = from.tellg();
  char group[5];
  from.get(group, 5);
  std::string in = group;
  const char group_str[] = " 0000 UL 1 Group Length";
  std::ostringstream os;
  std::string out;
  if( in[2] == 'x' && in[3] == 'x' )
    {
    std::string line = in;
    if(line[0] == '5')
      {
      uint16_t start = 0x5000;
      uint16_t end   = 0x50ff;
      for(uint16_t i= start; i<=end; i+=2)
        {
        os.str("");
        os << std::hex << i << group_str;
        if( OutputType != DICT_XML )
          {
          ConvertToCXX(os.str().c_str(), out);
          into << out << std::endl;
          }
        }
      }
    else if(line[0] == '6')
      {
      uint16_t start = 0x6000;
      uint16_t end   = 0x60ff;
      for(uint16_t i= start; i<=end; i+=2)
        {
        os.str("");
        os << std::hex << i << group_str;
        if( OutputType != DICT_XML )
          {
          ConvertToCXX(os.str().c_str(), out);
          into << out << std::endl;
          }
        }
      }
    }
  else
    {
    in += group_str;
    if( OutputType == DICT_XML )
      {
      ConvertToXML(in.c_str(), out);
      }
    else
      {
      ConvertToCXX(in.c_str(), out);
      }
    into << out << std::endl;
    }
  // seek back
  from.seekg(p);
}

} // end namespace gdcm
