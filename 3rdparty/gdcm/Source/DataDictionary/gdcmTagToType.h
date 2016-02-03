
// GENERATED FILE DO NOT EDIT
// $ xsltproc TagToType.xsl Part6.xml > gdcmTagToType.h

/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef GDCMTAGTOTYPE_H
#define GDCMTAGTOTYPE_H

#include "gdcmVR.h"
#include "gdcmVM.h"
#include "gdcmStaticAssert.h"

namespace gdcm {
// default template: the compiler should only pick it up when the element is private:
template <uint16_t group,uint16_t element> struct TagToType {
//GDCM_STATIC_ASSERT( group % 2 );
enum { VRType = VR::VRALL };
enum { VMType = VM::VM1_n };
};
// template for group length:
template <uint16_t group> struct TagToType<group,0x0000> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0000> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0001> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0002> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0003> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0100> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0110> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0120> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0200> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0300> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0400> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0600> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0700> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0800> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0850> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0860> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0900> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0901> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0000,0x0902> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x0903> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1000> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1001> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1002> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1005> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0000,0x1008> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1020> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1021> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1022> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1023> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1030> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x1031> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x4010> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5020> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5110> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5120> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5130> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5140> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5150> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5160> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5170> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5180> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x5190> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x51a0> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0000,0x51b0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0002,0x0000> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0001> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0002> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0003> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0010> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0012> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0013> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0016> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0100> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0002,0x0102> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1130> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1141> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_8 };
static const char* GetVMString() { return "1-8"; }
};
template <> struct TagToType<0x0004,0x1142> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1200> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1202> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1212> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1400> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1410> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1420> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1430> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1432> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1500> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_8 };
static const char* GetVMString() { return "1-8"; }
};
template <> struct TagToType<0x0004,0x1504> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1510> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1511> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x1512> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0004,0x151a> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0004,0x1600> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0001> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0005> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x0006> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0008> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM2_n };
static const char* GetVMString() { return "2-n"; }
};
template <> struct TagToType<0x0008,0x0010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0012> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0013> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0014> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0016> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0018> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x001a> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x001b> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0020> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0021> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0022> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0023> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0024> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0025> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x002a> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0030> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0031> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0032> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0033> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0034> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0035> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0040> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0041> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0042> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0050> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0051> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0052> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0054> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x0056> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0058> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x0060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0061> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x0062> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x0064> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0068> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0070> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0080> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0081> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0082> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0090> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0092> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0094> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x0096> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0100> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0102> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0103> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0104> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0105> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0106> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0107> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x010b> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x010c> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x010d> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x010f> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0112> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0114> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0115> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0116> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0117> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x0201> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1000> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1030> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1032> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x103e> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x103f> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1040> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1048> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1049> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1050> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1052> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1060> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1062> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1070> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1072> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1080> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1084> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1090> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1111> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1115> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1120> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1125> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1130> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1134> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x113a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1140> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1145> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x114a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x114b> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1150> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1155> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x115a> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1160> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1161> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x1162> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM3_3n };
static const char* GetVMString() { return "3-3n"; }
};
template <> struct TagToType<0x0008,0x1163> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0008,0x1164> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1167> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1195> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1197> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1198> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1199> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1200> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x1250> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2110> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2111> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2112> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2120> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2122> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2124> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2127> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2128> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2129> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x212a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2130> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x2132> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0008,0x2133> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2134> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2135> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2142> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2143> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2144> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2200> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2204> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2208> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2218> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2228> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2229> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2230> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2240> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2242> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2244> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2246> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2251> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2253> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2255> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2256> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2257> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2258> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x2259> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x225a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x225c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x3001> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x3010> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9007> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0008,0x9092> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9121> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9123> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9124> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9154> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9205> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9206> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9207> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9208> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9209> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9215> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9237> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9410> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9458> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9459> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0008,0x9460> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0010> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0021> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0022> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0024> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0030> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0032> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0101> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x0102> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1000> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0010,0x1001> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0010,0x1002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1005> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1010> {
static const char* GetVRString() { return "AS"; }
typedef VRToType<VR::AS>::Type Type;
enum { VRType = VR::AS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1020> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1021> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1040> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1050> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0010,0x1060> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1080> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1081> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x1090> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2000> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0010,0x2110> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0010,0x2150> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2152> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2154> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0010,0x2160> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2180> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x21a0> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x21b0> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x21c0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x21d0> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x21f0> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2201> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2202> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2203> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2210> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2292> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2293> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2294> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2295> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2296> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2297> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2298> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x2299> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0010,0x9431> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0010> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0021> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0030> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0031> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0040> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0042> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0050> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0051> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0060> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0062> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0063> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0012,0x0064> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0071> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0072> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0081> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0082> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0083> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0084> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0012,0x0085> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x0023> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0024> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0025> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0028> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0032> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0034> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0042> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0044> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0045> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x0046> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x0050> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x0052> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x0054> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x0056> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x1010> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x1020> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x1040> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2004> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2006> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2008> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2014> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2016> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2018> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x201a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x201c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x201e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2032> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2202> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2204> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2206> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2208> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x220a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x220c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x220e> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2210> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2222> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2224> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x2226> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x2228> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x222a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x222c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x3011> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3012> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3022> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3024> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3026> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3028> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3071> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3072> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3073> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3074> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3075> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3076> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3077> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3080> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x3099> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4006> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x400a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x400c> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x400e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x400f> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4011> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4012> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4013> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4014> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4015> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4016> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4017> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4018> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4019> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x401a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x401b> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x401c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4022> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4024> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4026> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4028> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4031> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4032> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4033> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4034> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4035> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4036> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4038> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x403a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x403c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x4040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4051> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4052> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4054> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4056> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4057> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4058> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4059> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x405a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x405c> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4062> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4064> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4070> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4072> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4074> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4076> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x4078> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x407a> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x407c> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x407e> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0014,0x5002> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0014,0x5004> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0010> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0014> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0015> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x0021> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x0022> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x0023> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0024> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0025> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0026> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0027> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0028> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0029> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x002a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0030> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x0031> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0032> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0033> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x0034> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0035> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0036> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0037> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0038> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0039> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x003a> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0040> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0042> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0050> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0060> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0070> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0071> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0072> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0073> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0074> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0075> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0080> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0081> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0082> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0083> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0084> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0085> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0086> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x0087> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0088> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0089> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0090> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0091> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0093> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0094> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x0095> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1000> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1002> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1003> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1004> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1005> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1006> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1007> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1008> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1010> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1011> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1012> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1014> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1016> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1017> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1018> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1019> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x101a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x101b> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1022> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1023> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1030> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1040> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1041> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1042> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1043> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1044> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1045> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1046> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1047> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1048> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1049> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1050> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1060> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1061> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1062> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1063> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1064> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1065> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1066> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1067> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1068> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1069> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x106a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x106c> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x106e> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1070> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1071> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1072> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1073> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1074> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1075> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1076> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1077> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1078> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1079> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1081> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1082> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1083> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1084> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1085> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1086> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1088> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1090> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1094> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1100> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1110> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1111> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1114> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1120> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1121> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1130> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1131> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1134> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1135> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1136> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1137> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1138> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x113a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1140> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1141> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1142> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1143> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1144> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1145> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1146> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1147> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1149> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x1150> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1151> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1152> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1153> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1154> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1155> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1156> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x115a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x115e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1160> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1161> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1162> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1164> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x1166> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1170> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1180> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1181> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1182> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x1183> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x1184> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x1190> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1191> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x11a0> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x11a2> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1200> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1201> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1210> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1240> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1242> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1243> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1244> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1250> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1251> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1260> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1261> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1300> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1301> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1302> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1310> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0018,0x1312> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1314> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1315> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1316> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1318> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1400> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1401> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1402> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1403> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1404> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1405> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1411> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1412> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1413> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1450> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1460> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1470> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1480> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1490> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1491> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1495> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1500> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1508> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1510> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1511> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1520> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1521> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x1530> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1531> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1600> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_3 };
static const char* GetVMString() { return "1-3"; }
};
template <> struct TagToType<0x0018,0x1602> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1604> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1606> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1608> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1610> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x1612> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1620> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0018,0x1622> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1623> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1624> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x1700> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_3 };
static const char* GetVMString() { return "1-3"; }
};
template <> struct TagToType<0x0018,0x1702> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1704> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1706> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1708> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1710> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x1712> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1720> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0018,0x1800> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1801> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1802> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x1803> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x2001> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x2002> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x2003> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x2004> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x2005> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x2006> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x2010> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x2020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x2030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x3100> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x3101> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x3102> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x3103> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x3104> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x3105> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5000> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x5010> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x5012> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5021> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5022> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5024> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5026> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5027> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5028> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5029> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5040> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5050> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5100> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5101> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5104> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x5210> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x0018,0x5212> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x6000> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6011> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6012> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6014> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6016> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6018> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x601a> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x601c> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x601e> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6020> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6022> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6024> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6026> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6028> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x602a> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x602c> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x602e> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6030> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6031> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6032> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6034> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6036> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6038> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6039> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x603a> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x603b> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x603c> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x603d> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x603e> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x603f> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6040> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6041> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6042> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6043> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6044> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6046> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6048> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x604a> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x604c> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x604e> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6050> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6052> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x6054> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x6056> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x6058> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x605a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x6060> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x7000> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7001> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7005> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7006> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7008> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x700a> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x700c> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x700e> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7010> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7011> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7012> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7014> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7016> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x701a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x7020> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x7022> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x7024> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7026> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x7028> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x702a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x702b> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x7032> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7034> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7036> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x7038> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7040> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7041> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7042> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7044> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7046> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x7048> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x704c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7050> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x7052> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x7054> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x7056> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x7058> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x7060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7062> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7064> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x7065> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x8150> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x8151> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9005> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9006> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9008> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9009> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9011> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9012> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9014> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9015> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9016> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9017> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9018> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9019> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9021> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9022> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9024> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9025> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9026> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9027> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9028> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9029> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9030> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9032> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9033> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9034> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9035> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9036> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9037> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9041> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9042> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9043> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9044> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9045> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9046> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9047> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9048> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9049> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9050> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9051> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9052> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9053> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9054> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9058> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9059> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9061> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9062> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9063> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9064> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9065> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9066> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9067> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9069> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9070> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9073> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9074> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9075> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9076> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9077> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9078> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9079> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x9080> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9081> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9082> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9083> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9084> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9085> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9087> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9089> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9090> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9091> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9092> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9093> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9094> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9095> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9096> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9098> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9100> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9101> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9103> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9104> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9105> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9106> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9107> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9112> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9114> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9115> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9117> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9118> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9119> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9125> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9126> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9127> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9147> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9151> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9152> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9155> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9159> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9166> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9168> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9169> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9170> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9171> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9172> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9173> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9174> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9175> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9176> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9177> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9178> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9179> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9180> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9181> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9182> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9183> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9184> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9185> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9186> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9195> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9196> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9197> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9198> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9199> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9200> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9214> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9217> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9218> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9219> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9220> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9226> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9227> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9231> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9232> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9234> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9236> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9239> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9240> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9241> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9250> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9251> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9252> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9253> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9254> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9255> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9256> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9257> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9258> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9259> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x925a> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x925b> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x925c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x925d> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x925e> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x925f> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9260> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9295> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9296> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9301> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9302> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9303> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9304> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9305> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9306> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9307> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9308> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9309> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9310> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9311> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9312> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9313> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9314> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9315> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9316> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9317> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x9318> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9319> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9320> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9321> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9322> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x9323> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9324> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9325> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9326> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9327> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9328> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9329> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9330> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9332> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9333> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9334> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9335> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9337> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9338> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9340> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9341> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9342> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9343> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9344> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9345> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9346> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9351> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9352> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0018,0x9353> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9360> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9401> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9402> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9403> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9404> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x9405> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9406> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9407> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9410> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9412> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9417> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9420> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9423> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9424> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9425> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9426> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9427> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9428> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9429> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x9430> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x9432> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9433> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9434> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9435> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9436> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9437> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9438> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9439> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9440> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0018,0x9441> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9442> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM2_n };
static const char* GetVMString() { return "2-n"; }
};
template <> struct TagToType<0x0018,0x9447> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9449> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9451> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9452> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9455> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9456> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9457> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9461> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0018,0x9462> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9463> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9464> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9465> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9466> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9467> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9468> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9469> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9470> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9471> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9472> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9473> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9474> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9476> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9477> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9504> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9506> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9507> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9508> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9509> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9510> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9511> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9514> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9515> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9516> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9517> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9524> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9525> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9526> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9527> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9528> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9530> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9531> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9538> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9601> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9602> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9603> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9604> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9605> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9606> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9607> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9701> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9715> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9716> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9717> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9718> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9719> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9720> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9721> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9722> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9723> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9724> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9725> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9726> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9727> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9729> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9732> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9733> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9734> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9735> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9736> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9737> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9738> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9739> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9740> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9749> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9751> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9755> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9756> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9758> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9759> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9760> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9761> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9762> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9763> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9764> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9765> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9766> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9767> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9768> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9769> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9770> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9771> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9772> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9801> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0018,0x9803> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9804> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9805> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9806> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9807> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9808> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x9809> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x980b> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x980c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x980d> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x980e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0x980f> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0xa001> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0xa002> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0018,0xa003> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x000d> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x000e> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0011> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0012> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0013> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0014> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0015> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0016> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0017> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0018> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0019> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0020,0x0022> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0024> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0026> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0020,0x0032> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0020,0x0035> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x0020,0x0037> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x0020,0x0050> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0052> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0062> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0070> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0020,0x00aa> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0100> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0105> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0110> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0200> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x0242> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1000> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1001> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1002> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1003> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1004> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1005> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0020,0x1040> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1041> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1070> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0020,0x1200> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1202> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1204> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1206> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1208> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x1209> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x3401> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x3402> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x3403> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x3404> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x3405> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x3406> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x5000> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0020,0x5002> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0020,0x9056> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9057> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9071> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9072> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9111> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9113> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9116> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9128> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9153> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9154> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9155> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9156> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9157> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0020,0x9158> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9161> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9162> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9163> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9164> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9165> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9167> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9213> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9221> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9222> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9228> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9238> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9241> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9245> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9246> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9247> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9248> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9249> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9250> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9251> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9252> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9253> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9254> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9255> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9256> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9257> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9301> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0020,0x9302> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x0020,0x9307> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9308> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0020,0x9309> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM16 };
static const char* GetVMString() { return "16"; }
};
template <> struct TagToType<0x0020,0x930a> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM16 };
static const char* GetVMString() { return "16"; }
};
template <> struct TagToType<0x0020,0x930c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x930d> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x930e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x930f> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9310> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9311> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9312> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9313> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9421> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9450> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9453> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9518> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0020,0x9529> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0020,0x9536> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0001> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0002> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0022,0x0003> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0004> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0022,0x0005> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0006> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0007> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0008> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0009> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x000a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x000b> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x000c> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x000d> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x000e> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0010> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0011> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0012> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0013> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0014> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0015> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0016> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0017> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0018> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0019> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x001a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x001b> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x001c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x001d> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x001e> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0021> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0022> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0030> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0031> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0032> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0022,0x0035> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0036> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0037> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0038> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0039> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0041> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0042> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0048> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0049> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x004e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0055> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0056> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0057> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x0058> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1007> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1009> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1019> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1024> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1025> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1028> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1029> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1033> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1035> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1037> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1039> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1044> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1053> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1054> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1059> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1065> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1066> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1090> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1092> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1093> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1094> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1095> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1096> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1097> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1101> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1103> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1121> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1122> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1125> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1127> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1128> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1130> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1131> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1132> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1133> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1134> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1135> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1140> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1150> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1153> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1155> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1159> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1210> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1211> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1212> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1225> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1230> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1250> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1255> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1257> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1260> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1262> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1273> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1300> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1310> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0022,0x1330> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0010> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0011> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0012> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0016> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0018> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0020> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0021> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0024> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0025> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0028> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0032> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0033> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0034> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0035> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0036> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0037> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0038> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0039> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0042> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0044> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0045> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0046> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0048> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0050> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0051> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0052> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0053> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0054> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0055> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0056> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0057> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0058> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0059> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0060> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0061> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0062> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0063> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0064> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0065> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0066> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0067> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0068> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0069> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0070> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0071> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0072> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0073> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0074> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0075> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0076> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0077> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0078> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0079> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0081> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0083> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0085> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0086> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0087> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0088> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0089> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0090> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0091> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0092> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0093> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0094> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0095> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0096> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0097> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0098> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0100> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0102> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0103> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0104> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0105> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0106> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0107> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0108> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0112> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0113> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0114> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0115> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0117> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0118> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0120> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0122> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0124> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0126> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0202> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0306> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0307> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0308> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0309> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0317> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0320> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0325> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0338> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0341> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0024,0x0344> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0002> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0003> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0005> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0006> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0008> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0009> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x000a> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0011> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0012> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0014> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0028,0x0031> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0028,0x0032> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0028,0x0034> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0028,0x0040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0050> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0051> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x005f> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0061> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0062> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0063> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0065> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0066> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0068> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0069> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0070> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0080> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0081> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0082> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0090> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0091> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0092> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0093> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0094> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0100> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0101> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0102> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0103> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0200> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0300> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0301> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0302> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0303> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0304> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0400> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0401> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0402> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0403> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0404> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0700> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0701> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0702> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0710> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0720> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0721> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x0722> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0730> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0740> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0a02> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x0a04> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1041> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1050> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x1051> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x1052> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1053> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1054> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1055> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x1056> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1090> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1104> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0028,0x1199> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1201> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1202> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1203> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1204> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1211> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1212> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1213> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1214> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1221> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1222> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1223> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1300> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1350> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1351> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1352> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x135a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1401> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1402> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1403> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1404> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1405> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1406> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1407> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0028,0x1408> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x140b> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x140c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x140d> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x140e> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x140f> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x1410> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x2000> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x2110> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x2112> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x2114> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x3000> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x3003> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x3004> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x3010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x3110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x5000> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x6010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x6020> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x6022> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x6023> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x6030> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x6040> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x6100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x6101> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x6102> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0028,0x6110> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x6112> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x6114> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0028,0x6120> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x6190> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x7fe0> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9001> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9002> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9003> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9099> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9108> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9132> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9145> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9235> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9411> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9415> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9416> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9422> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9443> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9444> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9445> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9446> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0028,0x9454> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9474> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9478> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9501> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9502> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9503> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0028,0x9505> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0028,0x9506> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0028,0x9507> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0028,0x9520> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM16 };
static const char* GetVMString() { return "16"; }
};
template <> struct TagToType<0x0028,0x9537> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x000a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x000c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x0012> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x0032> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x0033> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x0034> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x0035> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1000> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1001> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1010> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1011> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1021> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0032,0x1030> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1031> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1032> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1033> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1034> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1040> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1041> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1050> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1051> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1055> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1060> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1064> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x1070> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0032,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0004> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0008> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0010> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0011> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0014> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0016> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x001a> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x001b> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x001c> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x001d> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x001e> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0020> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0021> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0030> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0032> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0040> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0044> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0050> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0060> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0061> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0062> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0064> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0300> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0400> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0500> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x0502> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0038,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0005> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0010> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x001a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0020> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0200> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0202> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0203> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0205> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x003a,0x0208> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0209> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x020a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x020c> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0210> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0211> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0212> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0213> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0214> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0215> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0218> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x021a> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0220> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0221> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0222> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0223> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0230> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0231> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x003a,0x0240> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0241> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0242> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0244> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x003a,0x0245> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0246> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0247> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0248> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0300> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0301> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x003a,0x0302> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0001> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0x0002> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0003> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0004> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0005> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0006> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0007> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0009> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x000a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x000b> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0x0011> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0012> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0026> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0027> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0031> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0032> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0033> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0035> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0036> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0039> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x003a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0241> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0242> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0243> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0244> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0245> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0250> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0251> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0252> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0253> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0254> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0255> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0260> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0261> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0270> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0275> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0280> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0281> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0293> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0294> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0295> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0296> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0300> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0301> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0302> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0303> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_2 };
static const char* GetVMString() { return "1-2"; }
};
template <> struct TagToType<0x0040,0x0306> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0307> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x030e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0310> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0312> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0314> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0316> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0318> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0320> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0321> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0324> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0330> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0340> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0400> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0440> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0441> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x050a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0512> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0513> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0515> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0518> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x051a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0520> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0550> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0551> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0552> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0553> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0554> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0555> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0556> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0560> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0562> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x059a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0600> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0602> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0610> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0612> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x0620> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x06fa> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x071a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x072a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x073a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x074a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x08d8> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x08da> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x08ea> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x09f8> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1001> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1002> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1003> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1004> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1005> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1006> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1007> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1008> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1009> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x100a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1010> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0x1011> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1060> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1101> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1102> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x1103> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0x1400> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2001> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2004> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2005> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2006> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2007> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2008> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2009> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2016> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2017> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x2400> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x3001> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4002> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4003> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4004> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4005> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4006> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4007> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4009> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4010> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4011> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4015> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4016> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4018> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4019> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4021> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4022> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4023> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4025> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4026> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4027> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4028> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4029> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4031> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4032> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4033> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4034> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4035> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4036> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4037> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4041> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4050> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4051> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x4052> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x8302> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x9094> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x9096> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x9098> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x9210> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x9212> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0x9224> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0x9225> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa007> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa021> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa022> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa023> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa024> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa026> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa027> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa028> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa030> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa032> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa043> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa047> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa050> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa057> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0xa060> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa066> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa067> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa068> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa070> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa073> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa074> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa075> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa076> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa078> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa07a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa07c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa082> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa084> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa085> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa088> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa089> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa090> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa0b0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0040,0xa110> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa112> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa120> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa121> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa122> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa123> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa124> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa125> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0040,0xa130> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa132> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0xa136> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0xa138> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0xa13a> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0xa160> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa167> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa168> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa16a> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa170> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa171> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa172> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa173> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa174> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa180> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa192> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa193> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa194> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa195> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa224> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa290> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa296> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa297> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa29a> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x0040,0xa300> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa301> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa307> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa30a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0xa313> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa33a> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa340> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa352> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa353> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa354> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa358> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa360> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa370> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa372> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa375> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa380> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa385> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa390> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa402> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa403> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa404> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa491> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa492> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa493> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa494> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa496> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa504> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa525> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa600> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa601> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa603> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa730> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa731> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa732> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa744> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xa992> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xb020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xdb00> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xdb06> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xdb07> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xdb0b> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xdb0c> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xdb0d> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xdb73> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0040,0xe001> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe004> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe006> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe010> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe011> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe021> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe022> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe023> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe024> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe030> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0040,0xe031> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0042,0x0010> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0042,0x0011> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0042,0x0012> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0042,0x0013> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0042,0x0014> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0044,0x0001> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0002> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0003> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0004> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0007> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0008> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0044,0x0009> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x000a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x000b> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0010> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0011> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0012> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0013> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0044,0x0019> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0012> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0014> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0015> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0016> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0018> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0028> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0030> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0032> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0034> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0036> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0038> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0040> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0042> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0044> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0046> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0052> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0060> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0062> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0063> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0064> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0070> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0071> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0074> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0075> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0076> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0077> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0080> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0092> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0094> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0095> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0097> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0098> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0101> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0102> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0104> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0106> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0121> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0122> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0123> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0124> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0125> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0135> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0046,0x0137> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0139> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0145> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0146> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0046,0x0147> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0001> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0002> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0003> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0006> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0007> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0011> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0012> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0013> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0014> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0015> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0048,0x0100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0102> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x0048,0x0105> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0106> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0107> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0108> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0111> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0112> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0113> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0120> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0200> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0201> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0048,0x0202> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0048,0x0207> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x021a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x021e> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x021f> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0048,0x0301> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0013> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0014> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0015> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0016> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0017> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0018> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0019> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x001a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x001b> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x001c> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x001d> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x001e> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0050,0x0020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0001> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0002> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0003> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0004> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0006> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0007> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0008> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0009> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0011> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0012> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0013> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0014> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0016> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0025> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0026> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0027> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0028> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0029> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0030> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0031> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0033> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0034> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0036> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0038> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x0039> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0052,0x003a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0011> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0013> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0014> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0015> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0016> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0017> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0018> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0020> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0021> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0022> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0030> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0031> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0032> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0033> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0036> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0038> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0039> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0050> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0051> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0052> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0053> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0060> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0061> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0062> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0063> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0070> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0071> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0072> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0073> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0080> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0081> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0090> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0100> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0101> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0200> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0202> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0210> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x0211> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0222> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0300> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0302> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0304> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0306> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0308> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0400> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0410> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0412> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0414> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x0500> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1000> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0054,0x1001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1002> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1006> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1100> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1101> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1102> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1103> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1104> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1105> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1200> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1201> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0054,0x1202> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1203> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0054,0x1210> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1220> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x1300> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1310> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1311> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x1320> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1321> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1322> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1323> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1324> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1330> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0054,0x1400> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0054,0x1401> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0060,0x3000> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0060,0x3002> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0060,0x3008> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0060,0x3010> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0060,0x3020> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0062,0x0001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0003> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0004> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0005> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0006> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0008> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0009> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x000a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x000b> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0062,0x000c> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x000d> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0062,0x000e> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x000f> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0062,0x0010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0064,0x0002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0064,0x0003> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0064,0x0005> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0064,0x0007> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0064,0x0008> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0064,0x0009> {
static const char* GetVRString() { return "OF"; }
typedef VRToType<VR::OF>::Type Type;
enum { VRType = VR::OF };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0064,0x000f> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0064,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0001> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0003> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0004> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0009> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x000a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x000b> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x000c> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x000d> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x000e> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0011> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0013> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0015> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0016> {
static const char* GetVRString() { return "OF"; }
typedef VRToType<VR::OF>::Type Type;
enum { VRType = VR::OF };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0017> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0066,0x0018> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0019> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x001a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x0066,0x001b> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0066,0x001c> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0066,0x001e> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x001f> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0020> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0066,0x0021> {
static const char* GetVRString() { return "OF"; }
typedef VRToType<VR::OF>::Type Type;
enum { VRType = VR::OF };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0023> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0024> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0025> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0026> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0027> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0028> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0029> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x002a> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x002b> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x002c> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x002d> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x002e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x002f> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0031> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0032> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0034> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0035> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0066,0x0036> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6210> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6221> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6222> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6223> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6224> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6225> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6226> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6230> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6260> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6265> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6270> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6280> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x62a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x62a5> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x62c0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x62d0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x62d5> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x62e0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x62f0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM9 };
static const char* GetVMString() { return "9"; }
};
template <> struct TagToType<0x0068,0x62f2> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6300> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6310> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6320> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6330> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6340> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6345> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6346> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0068,0x6347> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0068,0x6350> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0068,0x6360> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6380> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6390> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63a4> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63a8> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63ac> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63b0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63c0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63d0> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63e0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x63f0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6400> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6410> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6420> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6430> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6440> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6450> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0068,0x6460> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0068,0x6470> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6490> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0068,0x64a0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0068,0x64c0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0068,0x64d0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM9 };
static const char* GetVMString() { return "9"; }
};
template <> struct TagToType<0x0068,0x64f0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0068,0x6500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6510> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6520> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6530> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6540> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6545> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6550> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x6560> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0068,0x6590> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0068,0x65a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x65b0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0068,0x65d0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x0068,0x65e0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0068,0x65f0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0068,0x6610> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0068,0x6620> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0070,0x0001> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0002> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0003> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0005> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0006> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0009> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0010> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0011> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0012> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0014> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0015> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0020> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0021> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0022> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2_n };
static const char* GetVMString() { return "2-n"; }
};
template <> struct TagToType<0x0070,0x0023> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0024> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0040> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0041> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0042> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0050> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0051> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0052> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0053> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x005a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0062> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0066> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0067> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0070,0x0068> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0081> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0082> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0083> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0084> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0086> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0087> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0100> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0101> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0102> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0103> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0207> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0208> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0209> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0226> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0227> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0228> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0229> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0230> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0231> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0232> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0233> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0234> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0241> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0070,0x0242> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0243> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0244> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0245> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0246> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0247> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0070,0x0248> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0249> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0250> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0251> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0070,0x0252> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0070,0x0253> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0254> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0255> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0256> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0257> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0258> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0261> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0262> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0273> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0070,0x0274> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0278> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0279> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0282> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0284> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0285> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0287> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0288> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0289> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0294> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0295> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0306> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0308> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0309> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x030a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x030c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x030d> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x030f> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0310> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0311> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0312> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0314> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0318> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x031a> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x031c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x031e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0401> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0070,0x0402> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0403> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0404> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0070,0x0405> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0002> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0004> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0006> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0008> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x000a> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x000c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x000e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0010> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0014> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0022> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0024> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0026> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0028> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0032> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0034> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0038> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0072,0x003a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x003c> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0072,0x003e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0040> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0050> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0052> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0054> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0056> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0060> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0062> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0064> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0066> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0068> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x006a> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x006c> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x006e> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0070> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0072> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0074> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0076> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0078> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x007a> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x007c> {
static const char* GetVRString() { return "SL"; }
typedef VRToType<VR::SL>::Type Type;
enum { VRType = VR::SL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x007e> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0080> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0100> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0102> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0104> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0106> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0108> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0072,0x010a> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x010c> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x010e> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0200> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0202> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0203> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0204> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0206> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0208> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0210> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0212> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2_n };
static const char* GetVMString() { return "2-n"; }
};
template <> struct TagToType<0x0072,0x0214> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0216> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0218> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0300> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0302> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0304> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0306> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0308> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0310> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0312> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0314> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0316> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0318> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0320> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0330> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0400> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0402> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0404> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0406> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0420> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0072,0x0421> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0072,0x0422> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0424> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0427> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0430> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0432> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM2_n };
static const char* GetVMString() { return "2-n"; }
};
template <> struct TagToType<0x0072,0x0434> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0500> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0510> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0512> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0514> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0516> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0520> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0072,0x0600> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0602> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0604> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0700> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0072,0x0702> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0704> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0705> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0706> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0710> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0712> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0714> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0716> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0717> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0072,0x0718> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x0120> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x0121> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1000> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1004> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1006> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x100a> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x100c> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x100e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1022> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1024> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1026> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1027> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1028> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x102a> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x102b> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x102c> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x102d> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1032> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1034> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1036> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1038> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x103a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0074,0x1040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1042> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1044> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1046> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1048> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x104a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x104c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x104e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1052> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1054> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1056> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1057> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0074,0x1200> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1202> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1204> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1210> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1212> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1216> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1222> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1224> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1230> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1234> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1236> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1238> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1242> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1244> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1246> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1324> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x1338> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0074,0x133a> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0076,0x0001> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0003> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0006> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0008> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x000a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x000c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x000e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0030> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0032> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0034> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0036> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0038> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0055> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0070> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0080> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x0090> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x00a0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x00b0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0076,0x00c0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0001> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0010> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0024> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0026> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0028> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x002a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x002e> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0050> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x0078,0x0060> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM9 };
static const char* GetVMString() { return "9"; }
};
template <> struct TagToType<0x0078,0x0070> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x0090> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x0078,0x00a0> {
static const char* GetVRString() { return "FD"; }
typedef VRToType<VR::FD>::Type Type;
enum { VRType = VR::FD };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x0078,0x00b0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x00b2> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x00b4> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x00b6> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0078,0x00b8> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0088,0x0130> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0088,0x0140> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0088,0x0200> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0088,0x0904> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0088,0x0906> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0088,0x0910> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0088,0x0912> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_32 };
static const char* GetVMString() { return "1-32"; }
};
template <> struct TagToType<0x0100,0x0410> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0100,0x0420> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0100,0x0424> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0100,0x0426> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0005> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0010> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0015> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0020> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x0400,0x0100> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0105> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0110> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0115> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0120> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0305> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0310> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0401> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0402> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0403> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0404> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0510> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0520> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0550> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0561> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0562> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0563> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0564> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x0400,0x0565> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0010> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x001e> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0030> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0050> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0060> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0061> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0062> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0063> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0065> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0067> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0069> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x006a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x00a0> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x00a1> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x00a2> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x00a4> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x00a8> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2000,0x0510> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0010> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0030> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0050> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0052> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0054> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x00a6> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x00a7> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x2010,0x00a8> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x00a9> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x2010,0x0100> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0110> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0120> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0130> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0140> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0150> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0152> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0154> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x015e> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0160> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0376> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x2010,0x0500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0510> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2010,0x0520> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0030> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0050> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x00a0> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x00a2> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0111> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0130> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2020,0x0140> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2030,0x0010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2030,0x0020> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0011> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_99 };
static const char* GetVMString() { return "1-99"; }
};
template <> struct TagToType<0x2040,0x0020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0070> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0072> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0074> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0080> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0082> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0090> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0100> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2040,0x0500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2050,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2050,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2050,0x0500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0010> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0030> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0040> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0050> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0070> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0140> {
static const char* GetVRString() { return "AE"; }
typedef VRToType<VR::AE>::Type Type;
enum { VRType = VR::AE };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0160> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0170> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2100,0x0500> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2110,0x0010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2110,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2110,0x0030> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2110,0x0099> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2120,0x0010> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2120,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2120,0x0070> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x0015> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x0030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x0040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x0060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x0080> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x00a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2130,0x00c0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0002> {
static const char* GetVRString() { return "UT"; }
typedef VRToType<VR::UT>::Type Type;
enum { VRType = VR::UT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0003> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0004> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0005> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0006> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0007> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0008> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0009> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x000a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x000b> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x000c> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x000d> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x000e> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x2200,0x000f> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x2200,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0002> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0003> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0004> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x000a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x000c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x000d> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x3002,0x000e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0010> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM6 };
static const char* GetVMString() { return "6"; }
};
template <> struct TagToType<0x3002,0x0011> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x3002,0x0012> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x3002,0x0020> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0022> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0024> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0026> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0028> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0029> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0032> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0034> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM4 };
static const char* GetVMString() { return "4"; }
};
template <> struct TagToType<0x3002,0x0040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0041> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0042> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0051> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3002,0x0052> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0002> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0004> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0006> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0008> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x3004,0x000a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x000c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2_n };
static const char* GetVMString() { return "2-n"; }
};
template <> struct TagToType<0x3004,0x000e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0012> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0014> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_3 };
static const char* GetVMString() { return "1-3"; }
};
template <> struct TagToType<0x3004,0x0040> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x3004,0x0042> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0052> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0054> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0056> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0058> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x3004,0x0060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0062> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0070> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0072> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3004,0x0074> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0002> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0004> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0006> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0008> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0009> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0012> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0014> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0016> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0022> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0024> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0026> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0028> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x002a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x3006,0x002c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0033> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0036> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0038> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0039> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0042> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0044> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0045> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x3006,0x0046> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0048> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0049> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x3006,0x0050> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3_3n };
static const char* GetVMString() { return "3-3n"; }
};
template <> struct TagToType<0x3006,0x0080> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0082> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0084> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0085> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0086> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x0088> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00a4> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00a6> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00b0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00b2> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00b4> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00b6> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00b7> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00b8> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00c0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00c2> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00c4> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3006,0x00c6> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM16 };
static const char* GetVMString() { return "16"; }
};
template <> struct TagToType<0x3006,0x00c8> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0012> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0014> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0016> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0021> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0022> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0024> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0025> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x002a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x002b> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x002c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0030> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0032> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0033> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0036> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0037> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x003a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x003b> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0041> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0042> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0044> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0045> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0046> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0047> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x3008,0x0048> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0052> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0054> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0056> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x005a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0061> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0062> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0063> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0064> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0065> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0066> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0068> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x006a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0070> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0072> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0074> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0076> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0078> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x007a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0080> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0082> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0090> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0092> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00b0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00c0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00d0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00e0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00f0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00f2> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00f4> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x00f6> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0105> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0110> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0116> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0120> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0122> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0130> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0132> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0134> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0136> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0138> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x013a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x013c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0140> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0142> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0150> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0152> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0160> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0162> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0164> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0166> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0168> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0200> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0202> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0220> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0223> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0224> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0230> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0240> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0250> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x3008,0x0251> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0002> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0003> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0004> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0006> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0007> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0009> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x000a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x000b> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x000c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x000e> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0010> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0012> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0013> {
static const char* GetVRString() { return "UI"; }
typedef VRToType<VR::UI>::Type Type;
enum { VRType = VR::UI };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0014> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0015> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0016> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0018> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x300a,0x001a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0021> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0022> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0023> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0025> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0026> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0027> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0028> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x002a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x002b> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x002c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x002d> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0042> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0043> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0044> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0046> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0048> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x004a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x004b> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x004c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x004e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x004f> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0050> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0051> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0052> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0053> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0055> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0070> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0071> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0072> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0078> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0079> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x007a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x007b> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0080> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0082> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x300a,0x0084> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0086> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0088> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0089> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x008a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00a0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00a2> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x300a,0x00a4> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00b0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00b2> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00b3> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00b4> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00b6> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00b8> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00ba> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00bb> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00bc> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00be> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3_n };
static const char* GetVMString() { return "3-n"; }
};
template <> struct TagToType<0x300a,0x00c0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00c2> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00c3> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00c4> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00c6> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00c7> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00c8> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00ca> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00cc> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x00ce> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d1> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d2> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d3> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d4> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d5> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d6> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d7> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d8> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00d9> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00da> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00db> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00dc> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00dd> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e1> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e2> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e3> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e4> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e5> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e6> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e7> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e8> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00e9> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x300a,0x00ea> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x300a,0x00eb> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x00ec> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x00ed> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00ee> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f2> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f3> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f4> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f5> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f6> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f7> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f8> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00f9> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00fa> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00fb> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00fc> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x00fe> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0100> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0102> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0104> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0106> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x300a,0x0107> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0108> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0109> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x010a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x010c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x010e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0110> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0111> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0112> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0114> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0115> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0116> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0118> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x011a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x011c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM2_2n };
static const char* GetVMString() { return "2-2n"; }
};
template <> struct TagToType<0x300a,0x011e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x011f> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0120> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0121> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0122> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0123> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0124> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0125> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0126> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0128> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0129> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x012a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x012c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x300a,0x012e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x300a,0x0130> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0134> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0140> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0142> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0144> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0146> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0148> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x014a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x014c> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x014e> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0180> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0182> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0183> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0184> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0190> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0192> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0194> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0196> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0198> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0199> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x019a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01a2> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01a4> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01a6> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01a8> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01b0> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01b2> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01b4> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01b6> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01b8> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01ba> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01bc> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01d0> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01d2> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01d4> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x01d6> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0200> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0202> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0206> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0210> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0212> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0214> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0216> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0218> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x021a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0222> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0224> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0226> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0228> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0229> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x022a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x022b> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x022c> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x022e> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0230> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0232> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0234> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0236> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0238> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0240> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0242> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0244> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0250> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0260> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0262> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0263> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0264> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0266> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x026a> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x026c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0280> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0282> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0284> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0286> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0288> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x028a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x028c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0290> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0291> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0292> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0294> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0296> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0298> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x029c> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x029e> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02a0> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02a2> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02a4> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02b0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02b2> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02b3> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02b4> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02b8> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02ba> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02c8> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02d0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02d2> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02d4> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x300a,0x02d6> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02e0> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02e1> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02e2> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x02e3> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02e4> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02e5> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02e6> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x02e7> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02e8> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02ea> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x02eb> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0302> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0304> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0306> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0308> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x030a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x300a,0x030c> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x030d> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x030f> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0312> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0314> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0316> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0318> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0320> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0322> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0330> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0332> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0334> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0336> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0338> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x033a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x033c> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0340> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0342> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0344> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0346> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0348> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x034a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x034c> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0350> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0352> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0354> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0356> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0358> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x035a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0360> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0362> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0364> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0366> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0370> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0372> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0374> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0380> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0382> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0384> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0386> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0388> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x038a> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0390> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0392> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0394> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x0396> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x300a,0x0398> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x300a,0x039a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x03a0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x03a2> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x03a4> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x03a6> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x03a8> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x03aa> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x03ac> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0401> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0402> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0410> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0412> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x300a,0x0420> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0421> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0422> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0423> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0424> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0431> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0432> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0433> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0434> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0435> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300a,0x0436> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0002> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0004> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0006> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0007> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0008> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0009> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x000a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x000c> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x000e> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0020> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0022> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0040> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0042> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0051> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0055> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0060> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x006a> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0080> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00a0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00b0> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00c0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00d0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00e0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00f0> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00f2> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00f4> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x00f6> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0100> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0102> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300c,0x0104> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300e,0x0002> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300e,0x0004> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300e,0x0005> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x300e,0x0008> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4000,0x0010> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4000,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0040> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0042> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0050> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x00ff> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0100> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0101> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0102> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0103> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0108> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0109> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x010a> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x010b> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x010c> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0111> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0112> {
static const char* GetVRString() { return "DA"; }
typedef VRToType<VR::DA>::Type Type;
enum { VRType = VR::DA };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0113> {
static const char* GetVRString() { return "TM"; }
typedef VRToType<VR::TM>::Type Type;
enum { VRType = VR::TM };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0114> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0115> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0117> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0118> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0119> {
static const char* GetVRString() { return "PN"; }
typedef VRToType<VR::PN>::Type Type;
enum { VRType = VR::PN };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x011a> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0200> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0202> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0210> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0212> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x0300> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4008,0x4000> {
static const char* GetVRString() { return "ST"; }
typedef VRToType<VR::ST>::Type Type;
enum { VRType = VR::ST };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x0001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x0002> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x0004> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1001> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1004> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x4010,0x1005> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x4010,0x1006> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1007> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1008> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1009> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x100a> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1011> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1012> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1013> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1014> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1015> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1016> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1017> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1018> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1019> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x101a> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x101b> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x4010,0x101c> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x4010,0x101d> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM6_6n };
static const char* GetVMString() { return "6-n"; }
};
template <> struct TagToType<0x4010,0x101e> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x101f> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1021> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x4010,0x1023> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1024> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1025> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1026> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1027> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1028> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1029> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x4010,0x102a> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x102b> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1031> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1033> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1034> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1037> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1038> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1039> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x103a> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1041> {
static const char* GetVRString() { return "DT"; }
typedef VRToType<VR::DT>::Type Type;
enum { VRType = VR::DT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1042> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1043> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x4010,0x1044> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1045> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1046> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1047> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1048> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1051> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1052> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1053> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1054> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1055> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1056> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1058> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1059> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1060> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x4010,0x1061> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM3 };
static const char* GetVMString() { return "3"; }
};
template <> struct TagToType<0x4010,0x1062> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1064> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1067> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1068> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x1069> {
static const char* GetVRString() { return "FL"; }
typedef VRToType<VR::FL>::Type Type;
enum { VRType = VR::FL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4010,0x106c> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x4ffe,0x0001> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x0005> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x0010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x0020> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x0022> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x0030> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x0040> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x0103> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x0104> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x0105> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x0106> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x0110> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x0112> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x0114> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x5000,0x1001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2000> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2002> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2004> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2006> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2008> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x200a> {
static const char* GetVRString() { return "UL"; }
typedef VRToType<VR::UL>::Type Type;
enum { VRType = VR::UL };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x200e> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2500> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2600> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5000,0x2610> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5200,0x9229> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5200,0x9230> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5400,0x0100> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5400,0x1004> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5400,0x1006> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5600,0x0010> {
static const char* GetVRString() { return "OF"; }
typedef VRToType<VR::OF>::Type Type;
enum { VRType = VR::OF };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x5600,0x0020> {
static const char* GetVRString() { return "OF"; }
typedef VRToType<VR::OF>::Type Type;
enum { VRType = VR::OF };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0010> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0011> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0012> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0015> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0022> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0040> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0045> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0050> {
static const char* GetVRString() { return "SS"; }
typedef VRToType<VR::SS>::Type Type;
enum { VRType = VR::SS };
enum { VMType = VM::VM2 };
static const char* GetVMString() { return "2"; }
};
template <> struct TagToType<0x6000,0x0051> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0052> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0060> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0061> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0062> {
static const char* GetVRString() { return "SH"; }
typedef VRToType<VR::SH>::Type Type;
enum { VRType = VR::SH };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0063> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0066> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x6000,0x0068> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0069> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0100> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0102> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0110> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0200> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0800> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x6000,0x0802> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x0803> {
static const char* GetVRString() { return "AT"; }
typedef VRToType<VR::AT>::Type Type;
enum { VRType = VR::AT };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x6000,0x0804> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1001> {
static const char* GetVRString() { return "CS"; }
typedef VRToType<VR::CS>::Type Type;
enum { VRType = VR::CS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1100> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1101> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1102> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1103> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1200> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x6000,0x1201> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x6000,0x1202> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x6000,0x1203> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1_n };
static const char* GetVMString() { return "1-n"; }
};
template <> struct TagToType<0x6000,0x1301> {
static const char* GetVRString() { return "IS"; }
typedef VRToType<VR::IS>::Type Type;
enum { VRType = VR::IS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1302> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1303> {
static const char* GetVRString() { return "DS"; }
typedef VRToType<VR::DS>::Type Type;
enum { VRType = VR::DS };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x1500> {
static const char* GetVRString() { return "LO"; }
typedef VRToType<VR::LO>::Type Type;
enum { VRType = VR::LO };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x6000,0x4000> {
static const char* GetVRString() { return "LT"; }
typedef VRToType<VR::LT>::Type Type;
enum { VRType = VR::LT };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x7fe0,0x0020> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x7fe0,0x0030> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x7fe0,0x0040> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x7f00,0x0011> {
static const char* GetVRString() { return "US"; }
typedef VRToType<VR::US>::Type Type;
enum { VRType = VR::US };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x7f00,0x0020> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x7f00,0x0030> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0x7f00,0x0040> {
static const char* GetVRString() { return "OW"; }
typedef VRToType<VR::OW>::Type Type;
enum { VRType = VR::OW };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0xfffa,0xfffa> {
static const char* GetVRString() { return "SQ"; }
typedef VRToType<VR::SQ>::Type Type;
enum { VRType = VR::SQ };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};
template <> struct TagToType<0xfffc,0xfffc> {
static const char* GetVRString() { return "OB"; }
typedef VRToType<VR::OB>::Type Type;
enum { VRType = VR::OB };
enum { VMType = VM::VM1 };
static const char* GetVMString() { return "1"; }
};

} // end namespace gdcm
#endif // GDCMTAGTOTYPE_H
