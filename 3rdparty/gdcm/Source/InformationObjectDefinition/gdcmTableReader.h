/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMTABLEREADER_H
#define GDCMTABLEREADER_H

#include "gdcmTypes.h"
#include "gdcmDefs.h"
//#include "gdcmModule.h"
//#include "gdcmIOD.h"
//#include "gdcmIODs.h"
//#include "gdcmModules.h"

#include <string>
#include <vector>
#include <map>

namespace gdcm
{
/**
 * \brief Class for representing a TableReader
 * \note This class is an empty shell meant to be derived
 */
class GDCM_EXPORT TableReader
{
public:
  TableReader(Defs &defs):CurrentDefs(defs),ParsingModule(false),ParsingModuleEntry(false),
  ParsingModuleEntryDescription(false),
  ParsingMacro(false),
  ParsingMacroEntry(false),
  ParsingMacroEntryDescription(false),
  ParsingIOD(false),
  ParsingIODEntry(false),
  Description() {}
  virtual ~TableReader() {}

  // Set/Get filename
  void SetFilename(const char *filename) { Filename = filename; }
  const char *GetFilename() { return Filename.c_str(); }

  int Read();

//protected:
  // You need to override those function in your subclasses:
  virtual void StartElement(const char *name, const char **atts);
  virtual void EndElement(const char *name);
  virtual void CharacterDataHandler(const char *data, int length);

void HandleModuleEntry(const char **atts);
void HandleModule(const char **atts);
void HandleModuleEntryDescription(const char **atts);
void HandleMacroEntry(const char **atts);
void HandleMacro(const char **atts);
void HandleMacroEntryDescription(const char **atts);
void HandleModuleInclude(const char **atts);
void HandleIODEntry(const char **atts);
void HandleIOD(const char **atts);

  //const Modules & GetModules() const { return CurrentModules; }
  //const Macros & GetMacros() const { return CurrentMacros; }
  //const IODs & GetIODs() const { return CurrentIODs; }
  const Defs & GetDefs() const { return CurrentDefs; }

private:
  std::string Filename;
  Defs &CurrentDefs;
  //Macros CurrentMacros;
  //Modules CurrentModules;
  //IODs CurrentIODs;
  Macro CurrentMacro;
  Module CurrentModule;
  IOD CurrentIOD;
  MacroEntry CurrentMacroEntry;
  ModuleEntry CurrentModuleEntry;
  IODEntry CurrentIODEntry;
  std::string CurrentModuleName;
  std::string CurrentModuleRef;
  std::string CurrentMacroRef;
  bool ParsingModule;
  bool ParsingModuleEntry;
  bool ParsingModuleEntryDescription;
  bool ParsingMacro;
  bool ParsingMacroEntry;
  bool ParsingMacroEntryDescription;
  bool ParsingIOD;
  bool ParsingIODEntry;
  Tag CurrentTag;
  std::string Description;
};

} // end namespace gdcm

#endif //GDCMTABLEREADER_H
