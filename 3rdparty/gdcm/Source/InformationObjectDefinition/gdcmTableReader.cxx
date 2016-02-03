/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmTableReader.h"
#include "gdcmTable.h"

#include <iostream>
#include <fstream>
#include "gdcm_expat.h"

#include <stdio.h> // for stderr
#include <string.h>

namespace gdcm
{
#if 1
#ifdef XML_LARGE_SIZE
#if defined(XML_USE_MSC_EXTENSIONS) && _MSC_VER < 1400
#define XML_FMT_INT_MOD "I64"
#else
#define XML_FMT_INT_MOD "ll"
#endif
#else
#define XML_FMT_INT_MOD "l"
#endif
#else
#define XML_FMT_INT_MOD ""
#endif

#ifndef BUFSIZ
#define BUFSIZ 4096
#endif

static void XMLCALL startElement(void *userData, const char *name, const char **atts)
{
  TableReader *tr = reinterpret_cast<TableReader*>(userData);
  tr->StartElement(name, atts);
}

static void XMLCALL endElement(void *userData, const char *name)
{
  TableReader *tr = reinterpret_cast<TableReader*>(userData);
  tr->EndElement(name);
}

static void XMLCALL characterDataHandler(void* userData, const char* data,
  int length)
{
  TableReader *tr = reinterpret_cast<TableReader*>(userData);
  tr->CharacterDataHandler(data,length);
}

void TableReader::HandleMacroEntryDescription(const char **atts)
{
  (void)atts;
  assert( ParsingMacroEntryDescription == false );
  ParsingMacroEntryDescription = true;
  assert( *atts == NULL );
  assert( Description == "" );
}

void TableReader::HandleModuleInclude(const char **atts)
{
  const char *ref = *atts;
  assert( strcmp(ref, "ref") == 0 );
  (void)ref; //removing warning
  const char *include = *(atts+1);
  CurrentModule.AddMacro( include );
  //assert( *(atts+2) == 0 ); // description ?
}

void TableReader::HandleModuleEntryDescription(const char **atts)
{
  (void)atts;
  assert( ParsingModuleEntryDescription == false );
  ParsingModuleEntryDescription = true;
  assert( *atts == NULL );
  assert( Description == "" );
}

void TableReader::HandleMacroEntry(const char **atts)
{
  std::string strgrp = "group";
  std::string strelt = "element";
  std::string strname = "name";
  std::string strtype = "type";
  Tag &tag = CurrentTag;
  MacroEntry &moduleentry = CurrentMacroEntry;
  const char **current = atts;
  while(*current /*&& current+1*/)
    {
    if( strgrp == *current )
      {
      unsigned int v;
      const char *raw = *(current+1);
      int r = sscanf(raw, "%04x", &v);
      assert( r == 1 );
      assert( v <= 0xFFFF );
      (void)r; //removing warning
      tag.SetGroup( (uint16_t)v );
      }
    else if( strelt == *current )
      {
      unsigned int v;
      const char *raw = *(current+1);
      int r = sscanf(raw, "%04x", &v);
      assert( r == 1 );
      assert( v <= 0xFFFF );
      (void)r; //removing warning
      tag.SetElement( (uint16_t)v );
      }
    else if( strname == *current )
      {
      const char *raw = *(current+1);
      moduleentry.SetName( raw );
      }
    else if( strtype == *current )
      {
      const char *raw = *(current+1);
      moduleentry.SetType( Type::GetTypeType(raw) );
      }
    else
      {
      assert(0);
      }
    ++current;
    ++current;
    }
}

void TableReader::HandleModuleEntry(const char **atts)
{
  std::string strgrp = "group";
  std::string strelt = "element";
  std::string strname = "name";
  std::string strtype = "type";
  Tag &tag = CurrentTag;
  ModuleEntry &moduleentry = CurrentModuleEntry;
  const char **current = atts;
  while(*current /*&& current+1*/)
    {
    if( strgrp == *current )
      {
      unsigned int v;
      const char *raw = *(current+1);
      int r = sscanf(raw, "%04x", &v);
      assert( r == 1 );
      assert( v <= 0xFFFF );
      (void)r; //removing warning
      tag.SetGroup( (uint16_t)v );
      }
    else if( strelt == *current )
      {
      unsigned int v;
      const char *raw = *(current+1);
      int r = sscanf(raw, "%04x", &v);
      assert( r == 1 );
      assert( v <= 0xFFFF );
      (void)r; //removing warning
      tag.SetElement( (uint16_t)v );
      }
    else if( strname == *current )
      {
      const char *raw = *(current+1);
      moduleentry.SetName( raw );
      }
    else if( strtype == *current )
      {
      const char *raw = *(current+1);
      moduleentry.SetType( Type::GetTypeType(raw) );
      }
    else
      {
      assert(0);
      }
    ++current;
    ++current;
    }
}

void TableReader::HandleIODEntry(const char **atts)
{
  std::string strie = "ie";
  std::string strname  = "name";
  std::string strref = "ref";
  std::string strusage = "usage";
  // <iod ref="Table B.7-1" name="Film Session IOD Modules">
  std::string strdesc = "description";
  IODEntry &iodentry = CurrentIODEntry;
  const char **current = atts;
  while(*current /*&& current+1*/)
    {
    const char *raw = *(current+1);
    if( strie == *current )
      {
      iodentry.SetIE( raw );
      }
    else if( strname == *current )
      {
      iodentry.SetName( raw );
      }
    else if( strref == *current )
      {
      iodentry.SetRef( raw );
      }
    else if( strusage == *current )
      {
      iodentry.SetUsage( raw );
      }
    else if( strdesc == *current )
      {
      //iodentry.SetDescription( raw );
      }
    else
      {
      assert(0);
      }
    ++current;
    ++current;
    }
}

void TableReader::HandleIOD(const char **atts)
{
  HandleModule(atts);
}

void TableReader::HandleMacro(const char **atts)
{
  HandleModule(atts);
}

void TableReader::HandleModule(const char **atts)
{
  std::string strref = "ref";
  std::string strname = "name";
  std::string strtable = "table";
  const char **current = atts;
  while(*current /*&& current+1*/)
    {
    if( strref == *current )
      {
      CurrentModuleRef = *(current+1);
      }
    else if( strtable == *current )
      {
      // ref to table is needed for referencing Macro
      CurrentMacroRef = *(current+1);
      }
    else if( strname == *current )
      {
      CurrentModuleName = *(current+1);
      }
    else
      {
      assert(0);
      }
    ++current;
    ++current;
    }
}

void TableReader::StartElement(const char *name, const char **atts)
{
  //int i;
  //int *depthPtr = (int *)userData;
//  for (i = 0; i < *depthPtr; i++)
//    putchar('\t');
  //std::cout << name << /*" : " << atts[0] << "=" << atts[1] <<*/ std::endl;
  if( strcmp(name, "tables" ) == 0 )
    {
    //*depthPtr += 1;
    }
  else if( strcmp(name, "macro" ) == 0 )
    {
    ParsingMacro = true;
    HandleMacro(atts);
    }
  else if( strcmp(name, "module" ) == 0 )
    {
    //std::cout << "Start Module" << std::endl;
    ParsingModule = true;
    HandleModule(atts);
    }
  else if( strcmp(name, "iod" ) == 0 )
    {
    ParsingIOD = true;
    HandleIOD(atts);
    }
  else if( strcmp(name, "entry" ) == 0 )
    {
    if( ParsingModule )
      {
      ParsingModuleEntry = true;
      HandleModuleEntry(atts);
      }
    else if( ParsingMacro )
      {
      ParsingMacroEntry = true;
      HandleMacroEntry(atts);
      }
    else if( ParsingIOD )
      {
      ParsingIODEntry = true;
      HandleIODEntry(atts);
      }
    }
  else if( strcmp(name, "description" ) == 0 )
    {
    if( ParsingModuleEntry )
      {
      HandleModuleEntryDescription(atts);
      }
    else if( ParsingMacroEntry )
      {
      HandleMacroEntryDescription(atts);
      }
    else /*if( ParsingIODoEntry )*/
      {
      assert(0);
      }
    }
  else if( strcmp(name, "section" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "include" ) == 0 )
    {
    // TODO !
    HandleModuleInclude(atts);
    }
  else if ( strcmp(name,"standard-sop-classes") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"mapping") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"unrecognized-rows") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"retired-defined-terms") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"enumerated-values") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"defined-terms") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"term") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"sop-classes") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"standard-and-related-general-sop-classes") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"media-storage-standard-sop-classes") == 0 )
    {
    // TODO !
    }
  else
    {
    assert(0);
    }
}

void TableReader::EndElement(const char *name)
{
//  int *depthPtr = (int *)userData;
//  *depthPtr -= 1;
  if( strcmp(name, "tables" ) == 0 )
    {
    }
  else if( strcmp(name, "macro" ) == 0 )
    {
    //std::cout << "Start Macro" << std::endl;
    CurrentMacro.SetName( CurrentModuleName.c_str() );
    CurrentDefs.GetMacros().AddMacro( CurrentMacroRef.c_str(), CurrentMacro);
    CurrentMacroRef.clear();
    CurrentModuleName.clear();
    CurrentMacro.Clear();
    ParsingMacro = false;
    }
  else if( strcmp( "module", name) == 0 )
    {
    CurrentModule.SetName( CurrentModuleName.c_str() );
    CurrentDefs.GetModules().AddModule( CurrentModuleRef.c_str(), CurrentModule);
    //std::cout << "End Module: " << CurrentModuleRef << "," << CurrentModuleName << std::endl;
    CurrentModuleRef.clear();
    CurrentModuleName.clear();
    CurrentModule.Clear();
    ParsingModule = false;
    }
  else if( strcmp(name, "iod" ) == 0 )
    {
    CurrentDefs.GetIODs().AddIOD( CurrentModuleName.c_str(), CurrentIOD);
    CurrentModuleName.clear();
    CurrentIOD.Clear();
    ParsingIOD = false;
    }
  else if( strcmp(name, "entry" ) == 0 )
    {
    if( ParsingModule )
      {
      ParsingModuleEntry = false;
      CurrentModule.AddModuleEntry( CurrentTag, CurrentModuleEntry);
      }
    else if( ParsingMacro )
      {
      ParsingMacroEntry = false;
      CurrentMacro.AddMacroEntry( CurrentTag, CurrentMacroEntry);
      }
    else if( ParsingIOD )
      {
      ParsingIODEntry = false;
      CurrentIOD.AddIODEntry( CurrentIODEntry);
      }
    }
  else if( strcmp(name, "description" ) == 0 )
    {
    if( ParsingModuleEntry )
      {
      ParsingModuleEntryDescription = false;
      CurrentModuleEntry.SetDescription( Description.c_str() );
      Description = "";
      }
    else if( ParsingMacroEntry )
      {
      ParsingMacroEntryDescription = false;
      //assert( !Description.empty() );
      CurrentMacroEntry.SetDescription( Description.c_str() );
      Description = "";
      }
    else
      {
      assert(0);
      }
    }
  else if( strcmp(name, "mapping" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "standard-sop-classes" ) == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"standard-and-related-general-sop-classes") == 0 )
    {
    // TODO !
    }
  else if ( strcmp(name,"media-storage-standard-sop-classes") == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "section" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "unrecognized-rows" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "retired-defined-terms" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "enumerated-values" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "defined-terms" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "term" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "sop-classes" ) == 0 )
    {
    // TODO !
    }
  else if( strcmp(name, "include" ) == 0 )
    {
    if( ParsingModule )
      {
      }
    else if( ParsingMacro )
      {
      //abort();
      }
    else
      {
      assert(0);
      }
    }
  else
    {
    assert(0);
    }
}

void TableReader::CharacterDataHandler(const char *data, int length)
{
  if( ParsingModuleEntryDescription )
    {
    std::string name( data, length);
    assert( (unsigned int)length == strlen( name.c_str() ) );
    Description.append( name );
    }
  else if( ParsingMacroEntryDescription )
    {
    std::string name( data, length);
    assert( (unsigned int)length == strlen( name.c_str() ) );
    Description.append( name );
    }
  else
    {
    //assert(0);
    }
}

int TableReader::Read()
{
  std::ifstream is( Filename.c_str(), std::ios::binary );

  char buf[BUFSIZ];
  XML_Parser parser = XML_ParserCreate(NULL);
  int done;
  //int depth = 0;
  XML_SetUserData(parser, this);
  XML_SetElementHandler(parser, startElement, endElement);
  XML_SetCharacterDataHandler(parser, characterDataHandler);
  int ret = 0;
  do {
    is.read(buf, sizeof(buf));
	  std::streamsize len = is.gcount();
    done = (unsigned int)len < sizeof(buf);
    if (XML_Parse(parser, buf, (int)len, done) == XML_STATUS_ERROR) {
      fprintf(stderr,
        "%s at line %" XML_FMT_INT_MOD "u\n",
        XML_ErrorString(XML_GetErrorCode(parser)),
        XML_GetCurrentLineNumber(parser));
      ret = 1; // Mark as error
      done = 1; // exit while
    }
  } while (!done);
  XML_ParserFree(parser);
  is.close();
  return ret;
}

} // end namespace gdcm
