/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMVR_H
#define GDCMVR_H

#include "gdcmTag.h"
#include "gdcmTrace.h"
#include "gdcmString.h"

#include <iostream>
#include <fstream>
#include <assert.h>

//these defines are here to ensure compilation on sunos gcc
#if defined (CS)
# undef CS
#endif
#if defined (DS)
# undef DS
#endif
#if defined (SS)
# undef SS
#endif


namespace gdcm
{

/**
 * \brief VR class
 * This is adapted from DICOM standard
 * The biggest difference is the INVALID VR
 * and the composite one that differ from standard (more like an addition)
 * This allow us to represent all the possible case express in the DICOMV3 dict
 * \note
 * VALUE REPRESENTATION (VR)
 * Specifies the data type and format of the Value(s) contained in the
 * Value Field of a Data Element.
 * VALUE REPRESENTATION FIELD:
 * The field where the Value Representation of a Data Element is
 * stored in the encoding of a Data Element structure with explicit VR.
 */
class GDCM_EXPORT VR
{
public:
  typedef enum {
    // Warning: Do not write if ( vr & VR::INVALID ) but if ( vr == VR::INVALID )
    INVALID = 0, // For Item/(Seq) Item Delimitation Item
    AE = 1,
    AS = 2,
    AT = 4,
    CS = 8,
    DA = 16,
    DS = 32,
    DT = 64,
    FD = 128,
    FL = 256,
    IS = 512,
    LO = 1024,
    LT = 2048,
    OB = 4096,
    OD = 134217728, // 2^27
    OF = 8192,
    OW = 16384,
    PN = 32768,
    SH = 65536,
    SL = 131072,
    SQ = 262144,
    SS = 524288,
    ST = 1048576,
    TM = 2097152,
    UI = 4194304,
    UL = 8388608,
    UN = 16777216,
    US = 33554432,
    UT = 67108864,
    OB_OW = OB | OW,
    US_SS = US | SS,
    US_SS_OW = US | SS | OW,
    // The following do not have a VRString equivalent (ie cannot be found in PS 3.6)
    VL16 = AE | AS | AT | CS | DA | DS | DT | FD | FL | IS | LO | LT | PN | SH | SL | SS | ST | TM | UI | UL | US, // if( VR & VL16 ) => VR has its VL coded over 16bits
    VL32 = OB | OW | OD | OF | SQ | UN | UT, // if( VR & VL32 ) => VR has its VL coded over 32bits
    VRASCII = AE | AS | CS | DA | DS | DT | IS | LO | LT | PN | SH | ST | TM | UI | UT,
    VRBINARY = AT | FL | FD | OB | OD | OF | OW | SL | SQ | SS | UL | UN | US, // FIXME: UN ?
    // PS 3.5:
    // Data Elements with a VR of SQ, OD, OF, OW, OB or UN shall always have a Value Multiplicity of one.
    // GDCM is adding a couple more: AS, LT, ST, UT
    VR_VM1 = AS | LT | ST | UT | SQ | OF | OD | OW | OB | UN, // All those VR have a VM1
    VRALL = VRASCII | VRBINARY,
    VR_END = UT+1  // Invalid VR, need to be max(VRType)+1
  } VRType;

  static const char *GetVRString(VRType vr);

  // This function will only look at the very first two chars nothing else
  static VRType GetVRTypeFromFile(const char *vr);

  // You need to make sure end of string is \0
  static VRType GetVRType(const char *vr);
  static const char *GetVRStringFromFile(VRType vr);

  static bool IsValid(const char *vr);
  // Check if vr1 is valid against vr2,
  // Typically vr1 is read from the file and vr2 is taken from the dict
  static bool IsValid(const char *vr1, VRType vr2);
  //static bool IsValid(const VRType &vr1, const VRType &vr2);
  // Find out if the string read is byte swapped
  static bool IsSwap(const char *vr);

  // Size read on disk
  // FIXME: int ?
  int GetLength() const {
    return VR::GetLength(VRField);
  }
  unsigned int GetSizeof() const;
  static uint32_t GetLength(VRType vr) {
    //if( vr == VR::INVALID ) return 4;
    if( vr & VL32 )
      {
      return 4;
      }
    else
      return 2;
  }

  // Some use of template metaprograming with ugly macro
  static bool IsBinary(VRType vr);
  static bool IsASCII(VRType vr);
  // TODO: REMOVE ME
  static bool CanDisplay(VRType vr);
  // TODO: REMOVE ME
  static bool IsBinary2(VRType vr);
  // TODO: REMOVE ME
  static bool IsASCII2(VRType vr);

  VR(VRType vr = INVALID):VRField(vr) { }
  //VR(VR const &vr):VRField(vr.VRField) { }
  std::istream &Read(std::istream &is)
    {
    char vr[2];
    is.read(vr, 2);
    VRField = GetVRTypeFromFile(vr);
    assert( VRField != VR::VR_END );
    //assert( VRField != VR::INVALID );
    if( VRField == VR::INVALID ) throw Exception( "INVALID VR" );
    if( VRField & VL32 )
      {
#if 0
      // For some reason this seems slower on my linux box...
      is.seekg(2, std::ios::cur );
#else
      char dum[2];
      is.read(dum, 2);
      if( !(dum[0] == 0 && dum[1] == 0 ))
        {
        // JDDICOM_Sample4.dcm
        gdcmDebugMacro( "32bits VR contains non zero bytes. Skipped" );
        }
#endif
      }
    return is;
    }

  const std::ostream &Write(std::ostream &os) const
    {
    VRType vrfield = VRField;
    gdcmAssertAlwaysMacro( !IsDual() );
    if( vrfield == VR::INVALID )
      {
      //vrfield = VR::UN;
      }
    const char *vr = GetVRString(vrfield);
    //assert( strlen( vr ) == 2 );
    assert( vr[0] && vr[1] && vr[2] == 0 );
    os.write(vr, 2);
    // See PS 3.5, Data Element Structure With Explicit VR
    if( vrfield & VL32 )
      {
      const char dum[2] = {0, 0};
      os.write(dum,2);
      }
    return os;
    }
  friend std::ostream &operator<<(std::ostream &os, const VR &vr);

  operator VRType () const { return VRField; }

  unsigned int GetSize() const;

  bool Compatible(VR const &vr) const;

  bool IsVRFile() const;

  bool IsDual() const;

private:
  // Internal function that map a VRType to an index in the VRStrings table
  static int GetIndex(VRType vr);
  VRType VRField;
};
//-----------------------------------------------------------------------------
inline std::ostream &operator<<(std::ostream &_os, const VR &val)
{
  //_os << VR::GetVRStringFromFile(val.VRField);
  _os << VR::GetVRString(val.VRField);
  return _os;
}

// Apparently SWIG is not happy with something, somewhere below...
#ifndef SWIG

// Tells whether VR Type is ASCII or Binary
template<int T> struct VRToEncoding;
// Convert from VR Type to real underlying type
template<int T> struct VRToType;
#define TYPETOENCODING(type,rep, rtype)         \
  template<> struct VRToEncoding<VR::type>    \
  { enum { Mode = VR::rep }; };                 \
  template<> struct VRToType<VR::type>        \
  { typedef rtype Type; };


// Do not use me
struct UI { char Internal[64+1];
  friend std::ostream& operator<<(std::ostream &_os, const UI &_val);
};
inline std::ostream& operator<<(std::ostream &_os, const UI &_val)
{
  _os << _val.Internal;
  return _os;
}

typedef String<'\\',16> AEComp;
typedef String<'\\',64> ASComp;
typedef String<'\\',16> CSComp;
typedef String<'\\',64> DAComp;
typedef String<'\\',64> DTComp;
typedef String<'\\',64> LOComp;
typedef String<'\\',64> LTComp;
typedef String<'\\',64> PNComp;
typedef String<'\\',64> SHComp;
typedef String<'\\',64> STComp;
typedef String<'\\',16> TMComp;
typedef String<'\\',64,0> UIComp;
typedef String<'\\',64> UTComp;


// TODO: Could be generated from XML file
TYPETOENCODING(AE,VRASCII ,AEComp)
TYPETOENCODING(AS,VRASCII ,ASComp)
TYPETOENCODING(AT,VRBINARY,Tag)
TYPETOENCODING(CS,VRASCII ,CSComp)
TYPETOENCODING(DA,VRASCII ,DAComp)
TYPETOENCODING(DS,VRASCII ,double)
TYPETOENCODING(DT,VRASCII ,DTComp)
TYPETOENCODING(FL,VRBINARY,float)
TYPETOENCODING(FD,VRBINARY,double)
TYPETOENCODING(IS,VRASCII ,int32_t)
TYPETOENCODING(LO,VRASCII ,LOComp)
TYPETOENCODING(LT,VRASCII ,LTComp)
TYPETOENCODING(OB,VRBINARY,uint8_t)
TYPETOENCODING(OF,VRBINARY,float)
TYPETOENCODING(OW,VRBINARY,uint16_t)
TYPETOENCODING(PN,VRASCII ,PNComp)
TYPETOENCODING(SH,VRASCII ,SHComp)
TYPETOENCODING(SL,VRBINARY,int32_t)
TYPETOENCODING(SQ,VRBINARY,unsigned char) // FIXME
TYPETOENCODING(SS,VRBINARY,int16_t)
TYPETOENCODING(ST,VRASCII ,STComp)
TYPETOENCODING(TM,VRASCII ,TMComp)
TYPETOENCODING(UI,VRASCII ,UIComp)
TYPETOENCODING(UL,VRBINARY,uint32_t)
TYPETOENCODING(UN,VRBINARY,uint8_t) // FIXME ?
TYPETOENCODING(US,VRBINARY,uint16_t)
TYPETOENCODING(UT,VRASCII ,UTComp)

#define VRTypeTemplateCase(type) \
  case VR::type: \
    return sizeof ( VRToType<VR::type>::Type );

inline unsigned int VR::GetSize() const
{
  switch(VRField)
  {
    VRTypeTemplateCase(AE)
    VRTypeTemplateCase(AS)
    VRTypeTemplateCase(AT)
    VRTypeTemplateCase(CS)
    VRTypeTemplateCase(DA)
    VRTypeTemplateCase(DS)
    VRTypeTemplateCase(DT)
    VRTypeTemplateCase(FL)
    VRTypeTemplateCase(FD)
    VRTypeTemplateCase(IS)
    VRTypeTemplateCase(LO)
    VRTypeTemplateCase(LT)
    VRTypeTemplateCase(OB)
    VRTypeTemplateCase(OF)
    VRTypeTemplateCase(OW)
    VRTypeTemplateCase(PN)
    VRTypeTemplateCase(SH)
    VRTypeTemplateCase(SL)
    VRTypeTemplateCase(SQ)
    VRTypeTemplateCase(SS)
    VRTypeTemplateCase(ST)
    VRTypeTemplateCase(TM)
    VRTypeTemplateCase(UI)
    VRTypeTemplateCase(UL)
    VRTypeTemplateCase(UN)
    VRTypeTemplateCase(US)
    VRTypeTemplateCase(UT)
    case VR::US_SS:
      return 2;
    default:
       assert( 0 && "should not" );
  }
  return 0;
}
#endif // SWIG


} // end namespace gdcm

#endif //GDCMVR_H
