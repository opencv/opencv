/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmFileDerivation.h"
#include "gdcmFile.h"
#include "gdcmAttribute.h"
#include "gdcmUIDGenerator.h"

#include <vector>
#include <string>

/*
PS 3.16 - 2008
CID 7202 Source Image Purposes of Reference
Context ID 7202
Source Image Purposes of Reference
Type: Extensible Version: 20051101
Coding Scheme
Designator
(0008,0102)
Code Value
(0008,0100)
Code Meaning
(0008,0104)
DCM 121320 Uncompressed predecessor
DCM 121321 Mask image for image processing operation
DCM 121322 Source image for image processing operation
DCM 121329 Source image for montage
DCM 121330 Lossy compressed predecessor
*/

/*
CID 7203 Image Derivation
Context ID 7203
Image Derivation
Type: Extensible Version: 20050822
Coding Scheme
Designator
(0008,0102)
Code Value
(0008,0100)
Code Meaning
(0008,0104)
DCM 113040 Lossy Compression
DCM 113041 Apparent Diffusion Coefficient
DCM 113042 Pixel by pixel addition
DCM 113043 Diffusion weighted
DCM 113044 Diffusion Anisotropy
DCM 113045 Diffusion Attenuated
DCM 113046 Pixel by pixel division
DCM 113047 Pixel by pixel mask
DCM 113048 Pixel by pixel Maximum
DCM 113049 Pixel by pixel mean
DCM 113050 Metabolite Maps from spectroscopy data
DCM 113051 Pixel by pixel Minimum
DCM 113052 Mean Transit Time
DCM 113053 Pixel by pixel multiplication
DCM 113054 Negative Enhancement Integral
DCM 113055 Regional Cerebral Blood Flow
DCM 113056 Regional Cerebral Blood Volume
DCM 113057 R-Coefficient Map
DCM 113058 Proton Density map
DCM 113059 Signal Change Map
DCM 113060 Signal to Noise Map
DCM 113061 Standard Deviation
DCM 113062 Pixel by pixel subtraction
DCM 113063 T1 Map
DCM 113064 T2* Map
DCM 113065 T2 Map
DCM 113066 Time Course of Signal
DCM 113067 Temperature encoded
DCM 113068 Student's T-Test
DCM 113069 Time To Peak map
DCM 113070 Velocity encoded
DCM 113071 Z-Score Map
DCM 113072 Multiplanar reformatting
DCM 113073 Curved multiplanar reformatting
DCM 113074 Volume rendering
DCM 113075 Surface rendering
DCM 113076 Segmentation
DCM 113077 Volume editing
DCM 113078 Maximum intensity projection
DCM 113079 Minimum intensity projection
DCM 113085 Spatial resampling
DCM 113086 Edge enhancement
DCM 113087 Smoothing
DCM 113088 Gaussian blur
DCM 113089 Unsharp mask
DCM 113090 Image stitching
*/
namespace gdcm
{
/*
Annex D DICOM Controlled Terminology Definitions (Normative)
This Annex specifies the meanings of codes defined in DICOM, either explicitly or by reference to another
part of DICOM or an external reference document or standard.
DICOM Code Definitions (Coding Scheme Designator 'DCM' Coding Scheme Version '01')
*/

struct CodeDefinition
{
const char *CodingSchemeDesignator;
unsigned int CodeValue;
const char *CodeMeaning;
};

// CID 7202 Source Image Purposes of Reference
static const CodeDefinition SourceImagePurposesofReference[] = {
{"DCM",121320,"Uncompressed predecessor"},
{"DCM",121321,"Mask image for image processing operation"},
{"DCM",121322,"Source image for image processing operation"},
{"DCM",121329,"Source image for montage"},
{"DCM",121330,"Lossy compressed predecessor"},
{NULL,0,NULL} // sentinel
};

// CID 7203 Image Derivation
static const CodeDefinition ImageDerivation[] = {
{ "DCM",113040,"Lossy Compression" },
{ "DCM",113041,"Apparent Diffusion Coefficient" },
{ "DCM",113042,"Pixel by pixel addition" },
{ "DCM",113043,"Diffusion weighted" },
{ "DCM",113044,"Diffusion Anisotropy" },
{ "DCM",113045,"Diffusion Attenuated" },
{ "DCM",113046,"Pixel by pixel division" },
{ "DCM",113047,"Pixel by pixel mask" },
{ "DCM",113048,"Pixel by pixel Maximum" },
{ "DCM",113049,"Pixel by pixel mean" },
{ "DCM",113050,"Metabolite Maps from spectroscopy data" },
{ "DCM",113051,"Pixel by pixel Minimum" },
{ "DCM",113052,"Mean Transit Time" },
{ "DCM",113053,"Pixel by pixel multiplication" },
{ "DCM",113054,"Negative Enhancement Integral" },
{ "DCM",113055,"Regional Cerebral Blood Flow" },
{ "DCM",113056,"Regional Cerebral Blood Volume" },
{ "DCM",113057,"R-Coefficient Map" },
{ "DCM",113058,"Proton Density map" },
{ "DCM",113059,"Signal Change Map" },
{ "DCM",113060,"Signal to Noise Map" },
{ "DCM",113061,"Standard Deviation" },
{ "DCM",113062,"Pixel by pixel subtraction" },
{ "DCM",113063,"T1 Map" },
{ "DCM",113064,"T2* Map" },
{ "DCM",113065,"T2 Map" },
{ "DCM",113066,"Time Course of Signal" },
{ "DCM",113067,"Temperature encoded" },
{ "DCM",113068,"Student's T-Test" },
{ "DCM",113069,"Time To Peak map" },
{ "DCM",113070,"Velocity encoded" },
{ "DCM",113071,"Z-Score Map" },
{ "DCM",113072,"Multiplanar reformatting" },
{ "DCM",113073,"Curved multiplanar reformatting" },
{ "DCM",113074,"Volume rendering" },
{ "DCM",113075,"Surface rendering" },
{ "DCM",113076,"Segmentation" },
{ "DCM",113077,"Volume editing" },
{ "DCM",113078,"Maximum intensity projection" },
{ "DCM",113079,"Minimum intensity projection" },
{ "DCM",113085,"Spatial resampling" },
{ "DCM",113086,"Edge enhancement" },
{ "DCM",113087,"Smoothing" },
{ "DCM",113088,"Gaussian blur" },
{ "DCM",113089,"Unsharp mask" },
{ "DCM",113090,"Image stitching" },
{NULL,0,NULL} // sentinel
};

// CID 7205 Purpose Of Reference to Alternate Representation
static const CodeDefinition PurposeOfReferencetoAlternateRepresentation[] = {
{ "DCM",121324,"Source image" },
{ "DCM",121325,"Lossy compressed image" },
{ "DCM",121326,"Alternate SOP Class instance" },
{ "DCM",121327,"Full fidelity image" },
{ "DCM",121328,"Alternate Photometric Interpretation image" },
{NULL,0,NULL} // sentinel
};

class FileDerivationInternals
{
public:
  FileDerivationInternals():References(),DerivationCodeSequenceCodeValue(0),PurposeOfReferenceCodeSequenceCodeValue(0),DerivationDescription() {}
  std::vector< std::pair< std::string, std::string > > References;
  unsigned int DerivationCodeSequenceCodeValue;
  unsigned int PurposeOfReferenceCodeSequenceCodeValue;
  std::string DerivationDescription;
};

FileDerivation::FileDerivation():F(new File),Internals(new FileDerivationInternals)
{
}

FileDerivation::~FileDerivation()
{
  delete Internals;
}

const CodeDefinition * GetCodeDefinition( unsigned int codevalue, const CodeDefinition list[] )
{
  const CodeDefinition * cds = list;
  while( cds->CodeMeaning && cds->CodeValue != codevalue )
    {
    ++cds;
    }
  if( cds->CodeValue )
    return cds;
  // else
  return NULL;
}

void FileDerivation::SetDerivationDescription( const char *dd )
{
  this->Internals->DerivationDescription = dd;
}

void FileDerivation::SetDerivationCodeSequenceCodeValue(unsigned int codevalue)
{
  this->Internals->DerivationCodeSequenceCodeValue = codevalue;
}

void FileDerivation::SetPurposeOfReferenceCodeSequenceCodeValue(unsigned int codevalue)
{
  this->Internals->PurposeOfReferenceCodeSequenceCodeValue = codevalue;
}

bool FileDerivation::AddReference(const char *referencedsopclassuid, const char *referencedsopinstanceuid)
{
  if( !UIDGenerator::IsValid(referencedsopclassuid) )
    {
    return false;
    }
  if( !UIDGenerator::IsValid(referencedsopinstanceuid) )
    {
    return false;
    }
  //
  Internals->References.push_back( std::make_pair( referencedsopclassuid, referencedsopinstanceuid) );
  return true;
}

/*
PS 3.3 - 2008 C.7.6.1.1.3 Derivation Description
If an Image is identified to be a derived image (see C.7.6.1.1.2 Image Type), Derivation
Description (0008,2111) and Derivation Code Sequence (0008,9215) describe the way in which
the image was derived. They may be used whether or not the Source Image Sequence
(0008,2112) is provided. They may also be used in cases when the Derived Image pixel data is
not significantly changed from one of the source images and the SOP Instance UID of the Derived
Image is the same as the one used for the source image.
*/
bool FileDerivation::AddDerivationDescription()
{
  File &file = GetFile();
  DataSet &ds = file.GetDataSet();

  if( !this->Internals->DerivationDescription.empty() )
    {
    Attribute<0x0008,0x2111> at1;
    at1.SetValue( this->Internals->DerivationDescription.c_str() );
    ds.Replace( at1.GetAsDataElement() );
    }

// ADD_DERIV: should we append the derivation after any existing one ?
// For compat reason: always override the existing one
//#define ADD_DERIV
  const Tag sisq(0x8,0x9215);
  SmartPointer<SequenceOfItems> sqi;
  sqi = new SequenceOfItems;
#ifdef ADD_DERIV
  if( ds.FindDataElement( sisq ) )
    sqi = ds.GetDataElement( sisq ).GetValueAsSQ();
#endif
  sqi->SetLengthToUndefined();

  Item item;
  item.SetVLToUndefined();

  //Item &item1 = sqi->GetItem(1);
  DataSet &subds3 = item.GetNestedDataSet();

  unsigned int codevalue = this->Internals->DerivationCodeSequenceCodeValue;
  const CodeDefinition *cd = GetCodeDefinition( codevalue, ImageDerivation );
  if ( !cd )
    {
    return false;
    }

  std::ostringstream os;
  os << cd->CodeValue;

  Attribute<0x0008,0x0100> at1;
  at1.SetValue( os.str() );
  subds3.Replace( at1.GetAsDataElement() );
  Attribute<0x0008,0x0102> at2;
  at2.SetValue( cd->CodingSchemeDesignator );
  subds3.Replace( at2.GetAsDataElement() );
  Attribute<0x0008,0x0104> at3;
  at3.SetValue( cd->CodeMeaning );
  subds3.Replace( at3.GetAsDataElement() );

  sqi->AddItem( item );

  DataElement de( sisq );
  de.SetVR( VR::SQ );
  de.SetValue( *sqi );
  de.SetVLToUndefined();
  ds.Replace( de );

  return true;
}

/*
PS 3.3 - 2008 C.7.6.1.1.4 Source image sequence
If an Image is identified to be a Derived image (see C.7.6.1.1.2 Image Type), Source Image
Sequence (0008,2112) is an optional list of Referenced SOP Class UID (0008,1150)/ Referenced
SOP Instance UID (0008,1150) pairs that identify the source images used to create the Derived
image. It may be used whether or not there is a description of the way the image was derived in
Derivation Description (0008,2111) or Derivation Code Sequence (0008,9215).
*/
bool FileDerivation::AddSourceImageSequence()
{
  File &file = GetFile();

  const Tag sisq(0x8,0x2112);
  SmartPointer<SequenceOfItems> sqi;
  sqi = new SequenceOfItems;
  DataElement de( sisq);
  de.SetVR( VR::SQ );
  de.SetValue( *sqi );
  de.SetVLToUndefined();

  DataSet &ds = file.GetDataSet();
  ds.Insert( de );

  //sqi = (SequenceOfItems*)ds.GetDataElement( sisq ).GetSequenceOfItems();
  sqi = ds.GetDataElement( sisq ).GetValueAsSQ();
  sqi->SetLengthToUndefined();

  if( sqi->GetNumberOfItems() )
    {
    gdcmWarningMacro( "Do not support appending Referenced Image" );
    return false;
    }
  std::vector< std::pair< std::string, std::string > >::const_iterator it =
    Internals->References.begin();
  for( ; it != Internals->References.end(); ++it )
    {
    Item item1;
    item1.SetVLToUndefined();

    DataSet &subds = item1.GetNestedDataSet();
    /*
    (0008,1150) UI =UltrasoundImageStorage                  #  28, 1 ReferencedSOPClassUID
    (0008,1155) UI [1.2.840.1136190195280574824680000700.3.0.1.19970424140438] #  58, 1 ReferencedSOPInstanceUID
     */
      {
      Attribute<0x8,0x1150> sopinstanceuid;
      sopinstanceuid.SetValue( it->first );
      subds.Replace( sopinstanceuid.GetAsDataElement() );
      Attribute<0x8,0x1155> sopclassuid;
      sopclassuid.SetValue( it->second );
      subds.Replace( sopclassuid.GetAsDataElement() );
      }
    if( !AddPurposeOfReferenceCodeSequence(subds) )
      {
      return false;
      }
    sqi->AddItem( item1 );
    }


  return true;
}

bool FileDerivation::AddPurposeOfReferenceCodeSequence(DataSet &subds)
{
/*
    (0040,a170) SQ (Sequence with undefined length #=1)     # u/l, 1 PurposeOfReferenceCodeSequence
      (fffe,e000) na (Item with undefined length #=3)         # u/l, 1 Item
        (0008,0100) SH [121320]                                 #   6, 1 CodeValue
        (0008,0102) SH [DCM]                                    #   4, 1 CodingSchemeDesignator
        (0008,0104) LO [Uncompressed predecessor]               #  24, 1 CodeMeaning
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
*/
  const Tag prcs(0x0040,0xa170);
  if( !subds.FindDataElement( prcs) )
    {
    SequenceOfItems *sqi2 = new SequenceOfItems;
    DataElement de( prcs );
    de.SetVR( VR::SQ );
    de.SetValue( *sqi2 );
    de.SetVLToUndefined();
    subds.Insert( de );
    }

  //SequenceOfItems *sqi = (SequenceOfItems*)subds.GetDataElement( prcs ).GetSequenceOfItems();
  SmartPointer<SequenceOfItems> sqi = subds.GetDataElement( prcs ).GetValueAsSQ();
  sqi->SetLengthToUndefined();

  if( !sqi->GetNumberOfItems() )
    {
    Item item; //( Tag(0xfffe,0xe000) );
    item.SetVLToUndefined();
    sqi->AddItem( item );
    }
  Item &item2 = sqi->GetItem(1);
  DataSet &subds2 = item2.GetNestedDataSet();

  /*
  (0008,0100) SH [121320]                                 #   6, 1 CodeValue
  (0008,0102) SH [DCM]                                    #   4, 1 CodingSchemeDesignator
  (0008,0104) LO [Uncompressed predecessor]               #  24, 1 CodeMeaning
   */

  unsigned int codevalue = this->Internals->PurposeOfReferenceCodeSequenceCodeValue;
  const CodeDefinition *cd = GetCodeDefinition( codevalue, SourceImagePurposesofReference );
  if( !cd )
    {
    cd = GetCodeDefinition( codevalue, PurposeOfReferencetoAlternateRepresentation );
    }
  if (!cd)
    {
    return false;
    }

  std::ostringstream os;
  os << cd->CodeValue;

  Attribute<0x0008,0x0100> at1;
  at1.SetValue( os.str() );
  subds2.Replace( at1.GetAsDataElement() );
  Attribute<0x0008,0x0102> at2;
  at2.SetValue( cd->CodingSchemeDesignator );
  subds2.Replace( at2.GetAsDataElement() );
  Attribute<0x0008,0x0104> at3;
  at3.SetValue( cd->CodeMeaning );
  subds2.Replace( at3.GetAsDataElement() );

  return true;
}


bool FileDerivation::Derive()
{
  File &file = GetFile();

  DataSet &ds = file.GetDataSet();
    {
    // (0008,0008) CS [ORIGINAL\SECONDARY]                     #  18, 2 ImageType
    Attribute<0x0008,0x0008> at3;
    static const CSComp values[] = {"DERIVED","SECONDARY"};
    at3.SetValues( values, 2, true ); // true => copy data !
    if( ds.FindDataElement( at3.GetTag() ) )
      {
      const DataElement &de = ds.GetDataElement( at3.GetTag() );
      if( !de.IsEmpty() )
        at3.SetFromDataElement( de );
      // Make sure that value #1 is at least 'DERIVED', so override in all cases:
      at3.SetValue( 0, values[0] );
      }
    ds.Replace( at3.GetAsDataElement() );
    }
//    {
//    Attribute<0x0008,0x2111> at1;
//    at1.SetValue( "lossy conversion" );
//    ds.Replace( at1.GetAsDataElement() );
//    }

  bool b = AddSourceImageSequence();
  if( !b )
    {
    gdcmDebugMacro( "Could not AddSourceImageSequence" );
    return false;
    }

  b = AddDerivationDescription();
  if( !b )
    {
    gdcmDebugMacro( "Could not AddDerivationDescription" );
    return false;
    }

  return true;
}


} // end namespace gdcm
