/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// See docs:
// http://www.swig.org/Doc1.3/Php.html
// http://www.swig.org/Doc1.3/SWIGPlus.html#SWIGPlus

%module(docstring="A DICOM library",directors=1) gdcm
#pragma SWIG nowarn=302,303,312,362,383,389,401,503,504,509,510,514,516,842

#if defined(SWIGPHP)
%{
#define SWIGPHP
%}
#endif

%{
#include "gdcmTypes.h"
#include "gdcmASN1.h"
#include "gdcmSmartPointer.h"
#include "gdcmSwapCode.h"
#include "gdcmEvent.h"
#include "gdcmProgressEvent.h"
#include "gdcmAnonymizeEvent.h"
#include "gdcmDirectory.h"
#ifdef GDCM_BUILD_TESTING
#include "gdcmTesting.h"
#endif
#include "gdcmObject.h"
#include "gdcmPixelFormat.h"
#include "gdcmMediaStorage.h"
#include "gdcmTag.h"
#include "gdcmPrivateTag.h"
#include "gdcmVL.h"
#include "gdcmVR.h"
#include "gdcmVM.h"
#include "gdcmObject.h"
#include "gdcmValue.h"
#include "gdcmByteValue.h"
#include "gdcmDataElement.h"
#include "gdcmItem.h"
#include "gdcmSequenceOfItems.h"
#include "gdcmDataSet.h"
//#include "gdcmString.h"
//#include "gdcmCodeString.h"
#include "gdcmPreamble.h"
#include "gdcmFile.h"
#include "gdcmBitmap.h"
#include "gdcmIconImage.h"
#include "gdcmPixmap.h"
#include "gdcmImage.h"
#include "gdcmFragment.h"
#include "gdcmCSAHeader.h"
#include "gdcmPDBHeader.h"
#include "gdcmSequenceOfFragments.h"
#include "gdcmTransferSyntax.h"
#include "gdcmBasicOffsetTable.h"
//#include "gdcmLO.h"
#include "gdcmCSAElement.h"
#include "gdcmPDBElement.h"
#include "gdcmFileSet.h"

#include "gdcmReader.h"
#include "gdcmPixmapReader.h"
#include "gdcmImageReader.h"
#include "gdcmWriter.h"
#include "gdcmPixmapWriter.h"
#include "gdcmImageWriter.h"
#include "gdcmStringFilter.h"
#include "gdcmGlobal.h"
#include "gdcmDicts.h"
#include "gdcmDict.h"
#include "gdcmCSAHeaderDict.h"
#include "gdcmDictEntry.h"
#include "gdcmCSAHeaderDictEntry.h"
#include "gdcmUIDGenerator.h"
#include "gdcmUUIDGenerator.h"
//#include "gdcmConstCharWrapper.h"
#include "gdcmScanner.h"
#include "gdcmAttribute.h"
#include "gdcmSubject.h"
#include "gdcmCommand.h"
#include "gdcmAnonymizer.h"
#include "gdcmFileAnonymizer.h"
#include "gdcmFileStreamer.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmUIDs.h"
#include "gdcmSorter.h"
#include "gdcmIPPSorter.h"
#include "gdcmSpectroscopy.h"
#include "gdcmPrinter.h"
#include "gdcmXMLPrinter.h"
#include "gdcmDumper.h"
#include "gdcmOrientation.h"
#include "gdcmFiducials.h"
#include "gdcmWaveform.h"
#include "gdcmPersonName.h"
#include "gdcmCurve.h"
#include "gdcmDICOMDIR.h"
#include "gdcmValidate.h"
#include "gdcmApplicationEntity.h"
#include "gdcmDictPrinter.h"
#include "gdcmFilenameGenerator.h"
#include "gdcmVersion.h"
#include "gdcmFilename.h"
#include "gdcmEnumeratedValues.h"
#include "gdcmPatient.h"
#include "gdcmStudy.h"
#include "gdcmUsage.h"
#include "gdcmMacroEntry.h"
#include "gdcmModuleEntry.h"
#include "gdcmNestedModuleEntries.h"
#include "gdcmMacro.h"
#include "gdcmMacros.h"
#include "gdcmModule.h"
#include "gdcmModules.h"
#include "gdcmDefs.h"
#include "gdcmIOD.h"
#include "gdcmIODs.h"
#include "gdcmTableEntry.h"
#include "gdcmDefinedTerms.h"
#include "gdcmSeries.h"
#include "gdcmIODEntry.h"
#include "gdcmRescaler.h"
#include "gdcmSegmentedPaletteColorLookupTable.h"
#include "gdcmUnpacker12Bits.h"
#include "gdcmDirectionCosines.h"
#include "gdcmTagPath.h"
#include "gdcmBitmapToBitmapFilter.h"
#include "gdcmPixmapToPixmapFilter.h"
#include "gdcmImageToImageFilter.h"
#include "gdcmSOPClassUIDToIOD.h"
#include "gdcmCoder.h"
#include "gdcmDecoder.h"
#include "gdcmCodec.h"
#include "gdcmImageCodec.h"
#include "gdcmJPEGCodec.h"
#include "gdcmJPEGLSCodec.h"
#include "gdcmJPEG2000Codec.h"
#include "gdcmPNMCodec.h"
#include "gdcmImageChangeTransferSyntax.h"
#include "gdcmFileChangeTransferSyntax.h"
#include "gdcmImageApplyLookupTable.h"
#include "gdcmSplitMosaicFilter.h"
#include "gdcmImageChangePhotometricInterpretation.h"
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmImageFragmentSplitter.h"
#include "gdcmDataSetHelper.h"
#include "gdcmFileExplicitFilter.h"
#include "gdcmImageHelper.h"
#include "gdcmMD5.h"
#include "gdcmDummyValueGenerator.h"
#include "gdcmSHA1.h"
#include "gdcmBase64.h"
#include "gdcmCryptographicMessageSyntax.h"
#include "gdcmCryptoFactory.h"
#include "gdcmSpacing.h"
#include "gdcmIconImageGenerator.h"
#include "gdcmIconImageFilter.h"

#include "gdcmSimpleSubjectWatcher.h"
#include "gdcmDICOMDIRGenerator.h"
#include "gdcmFileDerivation.h"

#include "gdcmQueryBase.h"
#include "gdcmQueryFactory.h"
#include "gdcmBaseRootQuery.h"
#include "gdcmPresentationContext.h"
#include "gdcmPresentationContextGenerator.h"
#include "gdcmCompositeNetworkFunctions.h"
#include "gdcmServiceClassUser.h"

#include "gdcmStreamImageReader.h"

#include "gdcmRegion.h"
#include "gdcmBoxRegion.h"
#include "gdcmImageRegionReader.h"

using namespace gdcm;
%}

//%include "docstrings.i"

// swig need to know what are uint16_t, uint8_t...
%include "stdint.i"

// gdcm does not use std::string in its interface, but we do need it for the
// %extend (see below)
%include "std_string.i"
//%include "std_set.i"
%include "std_vector.i"
%include "std_pair.i"
%include "std_map.i"
%include "exception.i"

// operator= is not needed in python AFAIK
%ignore operator=;                      // Ignore = everywhere.
%ignore operator++;                     // Ignore

%define EXTEND_CLASS_PRINT_GENERAL(classfuncname,classname)
%extend classname
{
  const char *classfuncname() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%enddef

#if defined(SWIGPHP)
%define EXTEND_CLASS_PRINT(classname)
EXTEND_CLASS_PRINT_GENERAL(__toString,classname)
%enddef
#endif

//%feature("autodoc", "1")
%include "gdcmConfigure.h"
//%include "gdcmTypes.h"
//%include "gdcmWin32.h"
// I cannot include gdcmWin32.h without gdcmTypes.h, first. But gdcmTypes.h needs to know _MSC_VER at swig time...
#define GDCM_EXPORT
%include "gdcmLegacyMacro.h"

// The following must be define early on as gdcmVL.h get included real early
%rename(GetValueLength) gdcm::VL::operator uint32_t;
//%csmethodmodifiers gdcm::VL::GetValueLength "private"
//%csmethodmodifiers GetValueLength "private"
//%rename(GetValue) VL::operator uint32_t ();
//  public static implicit operator int( MyType a )
//        {
//            return a.value;
//        }
%include "gdcmSwapCode.h"

//%feature("director") Event;
//%feature("director") AnyEvent;
%include "gdcmEvent.h"

%include "gdcmPixelFormat.h"
EXTEND_CLASS_PRINT(gdcm::PixelFormat)
%include "gdcmMediaStorage.h"
EXTEND_CLASS_PRINT(gdcm::MediaStorage)
//%rename(__getitem__) gdcm::Tag::operator[];
//%rename(this ) gdcm::Tag::operator[];
%include "gdcmTag.h"
EXTEND_CLASS_PRINT(gdcm::Tag)
%include "gdcmPrivateTag.h"
EXTEND_CLASS_PRINT(gdcm::PrivateTag)

%include "gdcmProgressEvent.h"
//%feature("director") AnonymizeEvent;
%include "gdcmAnonymizeEvent.h"

%include "gdcmVL.h"
EXTEND_CLASS_PRINT(gdcm::VL)
%include "gdcmVR.h"
EXTEND_CLASS_PRINT(gdcm::VR)
%include "gdcmVM.h"
EXTEND_CLASS_PRINT(gdcm::VM)
//%template (FilenameType) std::string;
%template (FilenamesType) std::vector<std::string>;
%include "gdcmDirectory.h"
EXTEND_CLASS_PRINT(gdcm::Directory)
%include "gdcmObject.h"
%include "gdcmValue.h"
EXTEND_CLASS_PRINT(gdcm::Value)
// Array marshaling for arrays of primitives




%apply char[] { char* buffer }
%ignore gdcm::ByteValue::WriteBuffer(std::ostream &os) const;
//%ignore gdcm::ByteValue::GetPointer() const;
//%ignore gdcm::ByteValue::GetBuffer(char *buffer, unsigned long length) const;
%include "gdcmByteValue.h"
EXTEND_CLASS_PRINT(gdcm::ByteValue)
%clear char* buffer;

%apply char[] { const char* array }

%include "gdcmASN1.h"
%include "gdcmSmartPointer.h"
%template(SmartPtrSQ) gdcm::SmartPointer<gdcm::SequenceOfItems>;
%template(SmartPtrFrag) gdcm::SmartPointer<gdcm::SequenceOfFragments>;
%include "gdcmDataElement.h"
EXTEND_CLASS_PRINT(gdcm::DataElement)

%clear const char* array;
%extend gdcm::DataElement
{
 void SetArray(unsigned char array[], unsigned int nitems) {
   $self->SetByteValue((char*)array, nitems * sizeof(unsigned char) );
 }
 void SetArray(char array[], unsigned int nitems) {
   $self->SetByteValue((char*)array, nitems * sizeof(char) );
 }
 void SetArray(unsigned short array[], unsigned int nitems) {
   $self->SetByteValue((char*)array, nitems * sizeof(unsigned short) );
 }
 void SetArray(short array[], unsigned int nitems) {
   $self->SetByteValue((char*)array, nitems * sizeof(short) );
 }
 void SetArray(float array[], unsigned int nitems) {
   $self->SetByteValue((char*)array, nitems * sizeof(float) );
 }
 void SetArray(double array[], unsigned int nitems) {
   $self->SetByteValue((char*)array, nitems * sizeof(double) );
 }
};

%include "gdcmItem.h"
EXTEND_CLASS_PRINT(gdcm::Item)
/*
 The following line is very important it properly convert :
SWIGTYPE_p_std__vectorT_int_t__size_type -> uint
*/
%template() std::vector< gdcm::Item >;
%include "gdcmSequenceOfItems.h"
EXTEND_CLASS_PRINT(gdcm::SequenceOfItems)
%rename (PHPDataSet) SWIGDataSet;
%rename (PHPTagToValue) SWIGTagToValue;
%include "gdcmDataSet.h"
EXTEND_CLASS_PRINT(gdcm::DataSet)
//%include "gdcmString.h"
//%include "gdcmCodeString.h"
//%include "gdcmTransferSyntax.h"
%include "gdcmPhotometricInterpretation.h"
EXTEND_CLASS_PRINT(gdcm::PhotometricInterpretation)
%include "gdcmObject.h"
%include "gdcmLookupTable.h"
EXTEND_CLASS_PRINT(gdcm::LookupTable)
%include "gdcmOverlay.h"
EXTEND_CLASS_PRINT(gdcm::Overlay)
//%include "gdcmVR.h"
//%template (DataElementSet) std::set<gdcm::DataElement>;
%include "gdcmPreamble.h"
EXTEND_CLASS_PRINT(gdcm::Preamble)
%include "gdcmTransferSyntax.h"
EXTEND_CLASS_PRINT(gdcm::TransferSyntax)
%include "gdcmFileMetaInformation.h"
EXTEND_CLASS_PRINT(gdcm::FileMetaInformation)

//%template(File) gdcm::SmartPointer<gdcm::File>;
//%ignore gdcm::File;

%include "gdcmFile.h"
EXTEND_CLASS_PRINT(gdcm::File)
//%include "gdcm_arrays_csharp.i"

%apply char[] { char* buffer }
%apply unsigned int[] { unsigned int dims[2] }
%apply unsigned int[] { unsigned int dims[3] }

//%apply byte OUTPUT[] { char* buffer } ;
//%ignore gdcm::Pixmap::GetBuffer(char*) const;
//%apply byte FIXED[] { char *buffer }
//%csmethodmodifiers gdcm::Pixmap::GetBuffer "public unsafe";
//%define %cs_marshal_array(TYPE, CSTYPE)
//       %typemap(ctype)  TYPE[] "void*"
//       %typemap(imtype, inattributes="[MarshalAs(UnmanagedType.LPArray)]") TYPE[] "CSTYPE[]"
//       %typemap(cstype) TYPE[] "CSTYPE[]"
//       %typemap(in)     TYPE[] %{ $1 = (TYPE*)$input; %}
//       %typemap(csin)   TYPE[] "$csinput"
//%enddef
//%cs_marshal_array(char, byte)
%include "gdcmBitmap.h"
EXTEND_CLASS_PRINT(gdcm::Bitmap)

%include "gdcmIconImage.h"
EXTEND_CLASS_PRINT(gdcm::IconImage)
%include "gdcmPixmap.h"
EXTEND_CLASS_PRINT(gdcm::Pixmap)

%include "gdcmImage.h"
EXTEND_CLASS_PRINT(gdcm::Image)
%include "gdcmFragment.h"
EXTEND_CLASS_PRINT(gdcm::Fragment)
%include "gdcmPDBElement.h"
EXTEND_CLASS_PRINT(gdcm::PDBElement)
%include "gdcmPDBHeader.h"
EXTEND_CLASS_PRINT(gdcm::PDBHeader)
%include "gdcmCSAElement.h"
EXTEND_CLASS_PRINT(gdcm::CSAElement)
%include "gdcmCSAHeader.h"
EXTEND_CLASS_PRINT(gdcm::CSAHeader)
%include "gdcmSequenceOfFragments.h"
EXTEND_CLASS_PRINT(gdcm::SequenceOfFragments)
%include "gdcmBasicOffsetTable.h"
EXTEND_CLASS_PRINT(gdcm::BasicOffsetTable)
//%include "gdcmLO.h"
%include "gdcmFileSet.h"
EXTEND_CLASS_PRINT(gdcm::FileSet)

%include "gdcmGlobal.h"
EXTEND_CLASS_PRINT(gdcm::Global)

%include "gdcmDictEntry.h"
EXTEND_CLASS_PRINT(gdcm::DictEntry)
%include "gdcmCSAHeaderDictEntry.h"
EXTEND_CLASS_PRINT(gdcm::CSAHeaderDictEntry)

%include "gdcmDict.h"
EXTEND_CLASS_PRINT(gdcm::Dict)
%include "gdcmCSAHeaderDict.h"
EXTEND_CLASS_PRINT(gdcm::CSAHeaderDictEntry)
%include "gdcmDicts.h"
EXTEND_CLASS_PRINT(gdcm::Dicts)

//%template (TagSetType) std::set<gdcm::Tag>;
%ignore gdcm::Reader::SetStream;
%include "gdcmReader.h"
//EXTEND_CLASS_PRINT(gdcm::Reader)
%include "gdcmPixmapReader.h"
//EXTEND_CLASS_PRINT(gdcm::PixmapReader)
%include "gdcmImageReader.h"
//EXTEND_CLASS_PRINT(gdcm::ImageReader)
%include "gdcmWriter.h"
//EXTEND_CLASS_PRINT(gdcm::Writer)
%include "gdcmPixmapWriter.h"
//EXTEND_CLASS_PRINT(gdcm::PixmapWriter)
%include "gdcmImageWriter.h"
//EXTEND_CLASS_PRINT(gdcm::ImageWriter)
%template (PairString) std::pair<std::string,std::string>;
//%template (MyM) std::map<gdcm::Tag,gdcm::ConstCharWrapper>;
%include "gdcmStringFilter.h"
//EXTEND_CLASS_PRINT(gdcm::StringFilter)
%include "gdcmUIDGenerator.h"
//EXTEND_CLASS_PRINT(gdcm::UIDGenerator)
%include "gdcmUUIDGenerator.h"
//EXTEND_CLASS_PRINT(gdcm::UUIDGenerator)
//%template (ValuesType)      std::set<std::string>;
%rename (CSharpTagToValue) SWIGTagToValue;
#define GDCM_STATIC_ASSERT(x)
%include "gdcmAttribute.h"
%include "gdcmSubject.h"
%include "gdcmCommand.h"

%template(SmartPtrScan) gdcm::SmartPointer<gdcm::Scanner>;
%include "gdcmScanner.h"
EXTEND_CLASS_PRINT(gdcm::Scanner)

%template(SmartPtrAno) gdcm::SmartPointer<gdcm::Anonymizer>;
//%ignore gdcm::Anonymizer::Anonymizer;


//%template(Anonymizer) gdcm::SmartPointer<gdcm::Anonymizer>;
//
//%ignore gdcm::Anonymizer;
//%feature("unref") Anonymizer "coucou $this->Delete();"
// http://www.swig.org/Doc1.3/SWIGPlus.html#SWIGPlus%5Fnn34
%include "gdcmAnonymizer.h"
%apply char[] { char* value_data }
%include "gdcmFileAnonymizer.h"
%clear char* value_data;

%apply char[] { char* array }
%template(SmartPtrFStreamer) gdcm::SmartPointer<gdcm::FileStreamer>;
%include "gdcmFileStreamer.h"
%clear char* array;

//EXTEND_CLASS_PRINT(gdcm::Anonymizer)
//%extend gdcm::Anonymizer
//{
//%typemap(cscode) gdcm::Anonymizer
//%{
//  public Anonymizer() : this(gdcmPINVOKE.Anonymizer_New(), false) {
//  }
//%}
//};

// System is a namespace in C#, need to rename to something different
%rename (PosixEmulation) System;
%include "gdcmSystem.h"
//EXTEND_CLASS_PRINT(gdcm::System)

%include "gdcmTrace.h"
//EXTEND_CLASS_PRINT(gdcm::Trace)
%include "gdcmUIDs.h"
EXTEND_CLASS_PRINT(gdcm::UIDs)
//%feature("director") gdcm::IPPSorter;

////////////////////////////////////////////////////////////////////////////
// cs_callback is used to marshall callbacks. It allows a C# function to
// be passed to C++ as a function pointer through P/Invoke, which has the
// ability to make unmanaged-to-managed thunks. It does NOT allow you to
// pass C++ function pointers to C#.
//
// I would have liked to support FastDelegate<...> as the C++ argument
// type; this would have involved the cs_callback2 macro... but it turns
// out not to work under default project settings because .NET functions
// use the __stdcall calling convention, but FastDelegate uses the default
// convention which tends to be something else (__fastcall?). So nevermind.
//
// Anyway, to use this macro you need to declare the function pointer type
// TYPE in the appropriate header file (including the calling convention),
// declare a delegate named after CSTYPE in your C# project, and use this
// macro in your .i file. Here is an example:
//
// in C++ header file (%include this header in your .i file):
// typedef void (__stdcall *Callback)(PCWSTR);
// void Foo(Callback c);
//
// in C# code:
// public delegate void CppCallback([MarshalAs(UnmanagedType.LPWStr)] string message);
//
// in your .i file:
// %cs_callback(Callback, CppCallback)
//
// Remember to invoke %cs_callback before any code involving Callback.
%define %cs_callback(TYPE, CSTYPE)
       %typemap(ctype) TYPE, TYPE& "void*"
       %typemap(in) TYPE  %{ $1 = (TYPE)$input; %}
       %typemap(in) TYPE& %{ $1 = (TYPE*)&$input; %}
       %typemap(imtype, out="IntPtr") TYPE, TYPE& "CSTYPE"
       %typemap(cstype, out="IntPtr") TYPE, TYPE& "CSTYPE"
       %typemap(csin) TYPE, TYPE& "$csinput"
%enddef
%define %cs_callback2(TYPE, CTYPE, CSTYPE)
       %typemap(ctype) TYPE "CTYPE"
       %typemap(in) TYPE %{ $1 = (TYPE)$input; %}
       %typemap(imtype, out="IntPtr") TYPE "CSTYPE"
       %typemap(cstype, out="IntPtr") TYPE "CSTYPE"
       %typemap(csin) TYPE "$csinput"
%enddef

%cs_callback(Sorter::SortFunction, Sorter::CppSortFunction)

%include "gdcmSorter.h"
EXTEND_CLASS_PRINT(gdcm::Sorter)
%include "gdcmIPPSorter.h"
EXTEND_CLASS_PRINT(gdcm::IPPSorter)
%include "gdcmSpectroscopy.h"
//EXTEND_CLASS_PRINT(gdcm::Spectroscopy)
%include "gdcmPrinter.h"
//EXTEND_CLASS_PRINT(gdcm::Printer)
%include "gdcmXMLPrinter.h"
//EXTEND_CLASS_PRINT(gdcm::XMLPrinter)
%include "gdcmDumper.h"
//EXTEND_CLASS_PRINT(gdcm::Dumper)
%include "gdcmOrientation.h"
EXTEND_CLASS_PRINT(gdcm::Orientation)
%include "gdcmDirectionCosines.h"
EXTEND_CLASS_PRINT(gdcm::DirectionCosines)

%include "gdcmFiducials.h"
%include "gdcmWaveform.h"
%include "gdcmPersonName.h"
%include "gdcmCurve.h"
%include "gdcmDICOMDIR.h"
%include "gdcmValidate.h"
%include "gdcmApplicationEntity.h"
%include "gdcmDictPrinter.h"
%include "gdcmFilenameGenerator.h"
%include "gdcmVersion.h"
EXTEND_CLASS_PRINT(gdcm::Version)
%include "gdcmFilename.h"
%include "gdcmEnumeratedValues.h"
%include "gdcmPatient.h"
%include "gdcmStudy.h"
%include "gdcmUsage.h"
%include "gdcmMacroEntry.h"
%include "gdcmModuleEntry.h"
EXTEND_CLASS_PRINT(gdcm::ModuleEntry)
%include "gdcmNestedModuleEntries.h"
%include "gdcmMacro.h"
%include "gdcmMacros.h"
%include "gdcmModule.h"
%include "gdcmModules.h"
%include "gdcmDefs.h"
%include "gdcmIOD.h"
%include "gdcmIODs.h"
%include "gdcmTableEntry.h"
%include "gdcmDefinedTerms.h"
%include "gdcmSeries.h"
%include "gdcmIODEntry.h"

%apply char[] { char* out }
%apply char[] { char* in }
%include "gdcmRescaler.h"
//EXTEND_CLASS_PRINT(gdcm::Rescaler)
%extend gdcm::Rescaler
{
  bool Rescale(double out[], const short in[], size_t n) {
    return $self->Rescale((char*)out, (char*)in, n);
  }
}
%clear char* out;
%clear char* in;

%include "gdcmSegmentedPaletteColorLookupTable.h"
%include "gdcmUnpacker12Bits.h"

%include "gdcmConfigure.h"
#ifdef GDCM_BUILD_TESTING
%include "gdcmTesting.h"
%ignore gdcm::Testing::ComputeMD5(const char *, const unsigned long , char []);
%ignore gdcm::Testing::ComputeFileMD5(const char*, char []);
%extend gdcm::Testing
{
  static const char *ComputeFileMD5(const char *filename) {
    static char buffer[33];
    gdcm::Testing::ComputeFileMD5(filename, buffer);
    return buffer;
  }
};
#endif
%include "gdcmTagPath.h"
%include "gdcmBitmapToBitmapFilter.h"
%include "gdcmPixmapToPixmapFilter.h"
//%ignore gdcm::ImageToImageFilter::GetOutput() const;
%include "gdcmImageToImageFilter.h"
%include "gdcmSOPClassUIDToIOD.h"
//%feature("director") Coder;
//%include "gdcmCoder.h"
//%feature("director") Decoder;
//%include "gdcmDecoder.h"
//%feature("director") Codec;
//%include "gdcmCodec.h"
%feature("director") ImageCodec;
%include "gdcmImageCodec.h"
%include "gdcmJPEGCodec.h"
%include "gdcmJPEGLSCodec.h"
%include "gdcmJPEG2000Codec.h"
%include "gdcmPNMCodec.h"
%include "gdcmImageChangeTransferSyntax.h"
%template(SmartPtrFCTS) gdcm::SmartPointer<gdcm::FileChangeTransferSyntax>;
%include "gdcmFileChangeTransferSyntax.h"
%include "gdcmImageApplyLookupTable.h"
%include "gdcmSplitMosaicFilter.h"
%include "gdcmImageChangePhotometricInterpretation.h"
%include "gdcmImageChangePlanarConfiguration.h"
%include "gdcmImageFragmentSplitter.h"
%include "gdcmDataSetHelper.h"
%include "gdcmFileExplicitFilter.h"
%template (DoubleArrayType) std::vector<double>;
%template (UShortArrayType) std::vector<unsigned short>;
%template (UIntArrayType) std::vector<unsigned int>;
%include "gdcmImageHelper.h"
%include "gdcmMD5.h"
%include "gdcmDummyValueGenerator.h"
%include "gdcmSHA1.h"
//%include "gdcmBase64.h"
%include "gdcmCryptographicMessageSyntax.h"
%include "gdcmCryptoFactory.h"
%include "gdcmSpacing.h"
%include "gdcmIconImageGenerator.h"
%include "gdcmIconImageFilter.h"

%feature("director") SimpleSubjectWatcher;
%include "gdcmSimpleSubjectWatcher.h"
%include "gdcmDICOMDIRGenerator.h"
%include "gdcmFileDerivation.h"

// MEXD:
%template(DataSetArrayType) std::vector< gdcm::DataSet >;
%template(FileArrayType) std::vector< gdcm::File >;
%template(PresentationContextArrayType) std::vector< gdcm::PresentationContext >;
%template(KeyValuePairType) std::pair< gdcm::Tag, std::string>;
%template(KeyValuePairArrayType) std::vector< std::pair< gdcm::Tag, std::string> >;
%template(TagArrayType) std::vector< gdcm::Tag >;
%include "gdcmQueryBase.h"
%include "gdcmBaseRootQuery.h"
%include "gdcmQueryFactory.h"
%template(CharSetArrayType) std::vector< gdcm::ECharSet >;
%include "gdcmCompositeNetworkFunctions.h"
%include "gdcmPresentationContext.h"
//EXTEND_CLASS_PRINT(gdcm::PresentationContext)
%include "gdcmPresentationContextGenerator.h"
typedef int64_t time_t; // FIXME
%include "gdcmServiceClassUser.h"
%apply char[] { char* inReadBuffer }
%include "gdcmStreamImageReader.h"
%clear char* inReadBuffer;
%include "gdcmRegion.h"
EXTEND_CLASS_PRINT(gdcm::Region)
%include "gdcmBoxRegion.h"
EXTEND_CLASS_PRINT(gdcm::BoxRegion)
%apply char[] { char* inreadbuffer }
%include "gdcmImageRegionReader.h"
%clear char* inreadbuffer;
