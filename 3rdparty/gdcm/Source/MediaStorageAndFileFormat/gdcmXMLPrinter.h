/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMXMLPRINTER_H
#define GDCMXMLPRINTER_H

/*

The Normative version of the XML Schema for the Native DICOM Model follows:



start = element NativeDicomModel { DicomDataSet }

# A DICOM Data Set is as defined in PS3.5. It does not appear
# as an XML Element, since it does not appear in the binary encoded
# DICOM objects. It exists here merely as a documentation aid.

DicomDataSet = DicomAttribute*
DicomAttribute = element DicomAttribute {
Tag, VR, Keyword?, PrivateCreator?,
( BulkData | Value+ | Item+ | PersonName+ )?
}

BulkData = element BulkData{ UUID }
Value = element Value { Number, xsd:string }
Item = element Item { Number, DicomDataSet }
PersonName = element PersonName {
Number,
element SingleByte { NameComponents }?,
element Ideographic { NameComponents }?,
element Phonetic
{ NameComponents }?
}

NameComponents =
element FamilyName {xsd:string}?,
element GivenName {xsd:string}?,
element MiddleName {xsd:string}?,
element NamePrefix {xsd:string}?,
element NameSuffix {xsd:string}?

# keyword is the attribute tag from PS3.6
# (derived from the DICOM Attribute's name)
Keyword = attribute keyword { xsd:token }
# canonical XML definition of Hex, with lowercase letters disallowed
Tag = attribute tag { xsd:string{ minLength="8" maxLength="8" pattern="[0-9A-F]{8}" } }
VR = attribute vr { "AE" | "AS" | "AT"| "CS" | "DA" | "DS" | "DT" | "FL" | "FD"
| "IS" | "LO" | "LT" | "OB" | "OF" | "OW" | "PN" | "SH" | "SL"
| "SQ" | "SS" | "ST" | "TM" | "UI" | "UL" | "UN" | "US" | "UT" }
PrivateCreator = attribute privateCreator{ xsd:string }
UUID = attribute uuid { xsd:string }
Number = attribute number { xsd:positiveInteger }


*/

#include "gdcmFile.h"
#include "gdcmDataElement.h"

namespace gdcm
{

class DataSet;
class DictEntry;
class Dicts;

class GDCM_EXPORT XMLPrinter
{
public:
  XMLPrinter();
  virtual ~XMLPrinter();

  // Set file
  void SetFile(File const &f) { F = &f; }

  

  typedef enum {
  
    OnlyUUID = 0 ,
    LOADBULKDATA = 1     
    
  } PrintStyles;

  // Set PrintStyle value
  void SetStyle(PrintStyles ps)
  {
    PrintStyle = ps;
  }
  
  // Get PrintStyle value
  PrintStyles GetPrintStyle() const 
  {
    return PrintStyle;
  }

  // Print
  void Print(std::ostream& os);

  // Print an individual dataset
  void PrintDataSet(const DataSet &ds, const TransferSyntax & ts, std::ostream& os);
  
  //void PrintUID(std::ostream &os);

  /// Virtual function mecanism to allow application programmer to
  /// override the default mecanism for BulkData handling. By default
  /// GDCM will simply discard the BulkData and only write the UUID
  virtual void HandleBulkData(const char *uuid, const TransferSyntax &ts,
    const char *bulkdata, size_t bulklen);

protected:

  VR PrintDataElement(std::ostream &os, const Dicts &dicts, const DataSet & ds, const DataElement &de, const TransferSyntax & ts);
  
  void PrintSQ(const SequenceOfItems *sqi, const TransferSyntax & ts, std::ostream &os);
    
  PrintStyles PrintStyle;
  
  const File *F;
  
};

} // end namespace gdcm

#endif //GDCMXMLPRINTER_H
