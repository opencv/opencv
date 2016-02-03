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
// http://www.swig.org/Doc1.3/Python.html
// http://www.swig.org/Doc1.3/SWIGPlus.html#SWIGPlus
// cstring_output_allocate_size:
// http://www.swig.org/Doc1.3/Library.html
// http://www.geocities.com/foetsch/python/extending_python.htm
// http://www.ddj.com/cpp/184401747
// http://www.ddj.com/article/printableArticle.jhtml;jsessionid=VM4IXCQG5KM10QSNDLRSKH0CJUNN2JVN?articleID=184401747&dept_url=/cpp/
// http://matt.eifelle.com/2008/11/04/exposing-an-array-interface-with-swig-for-a-cc-structure/

%module(docstring="A DICOM library",directors=1) gdcmswig
// http://www.swig.org/Doc1.3/Warnings.html
// "There is no option to suppress all SWIG warning messages."
#pragma SWIG nowarn=302,303,312,362,383,389,401,503,504,509,510,514,516
%{
#include <cstddef> // ptrdiff_t
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
#include "gdcmPythonFilter.h"
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
#include "gdcmJSON.h"

using namespace gdcm;
%}

//%insert("runtime") %{
//#include "myheader.h"
//%}

%include "docstrings.i"

// swig need to know what are uint16_t, uint8_t...
%include "stdint.i"
//typedef int gdcm::DataSet::SizeType; // FIXME
//%include "typemaps.i"

// gdcm does not use std::string in its interface, but we do need it for the
// %extend (see below)
%include "std_string.i"
%include "std_set.i"
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

#if defined(SWIGPYTHON)
%define EXTEND_CLASS_PRINT(classname)
EXTEND_CLASS_PRINT_GENERAL(__str__,classname)
%enddef
#endif

//%feature("autodoc", "1")
%include "gdcmConfigure.h"
//%include "gdcmTypes.h"
//%include "gdcmWin32.h"
// I cannot include gdcmWin32.h without gdcmTypes.h, first. But gdcmTypes.h needs to know _MSC_VER at swig time...
#define GDCM_EXPORT
%include "gdcmLegacyMacro.h"
%rename(__add__) gdcm::VL::operator+=;
%include "gdcmSwapCode.h"

//%feature("director") Event;
//%feature("director") AnyEvent;
%include "gdcmEvent.h"

%include "gdcmPixelFormat.h"
EXTEND_CLASS_PRINT(gdcm::PixelFormat)
%include "gdcmMediaStorage.h"
EXTEND_CLASS_PRINT(gdcm::MediaStorage)
%rename(__getitem__) gdcm::Tag::operator[];
//%rename(__getattr__) gdcm::Tag::operator[];
%include "gdcmTag.h"
EXTEND_CLASS_PRINT(gdcm::Tag)
%include "gdcmPrivateTag.h"
EXTEND_CLASS_PRINT(gdcm::PrivateTag)

%include "gdcmProgressEvent.h"
%extend gdcm::ProgressEvent {
  static ProgressEvent *Cast(Event *event) {
    return dynamic_cast<ProgressEvent*>(event);
  }
};
//%feature("director") AnonymizeEvent;
%include "gdcmAnonymizeEvent.h"
%extend gdcm::AnonymizeEvent {
  static AnonymizeEvent *Cast(Event *event) {
    return dynamic_cast<AnonymizeEvent*>(event);
  }
};

%include "gdcmVL.h"
EXTEND_CLASS_PRINT(gdcm::VL)
//%typemap(out) int
//{
//    $result = SWIG_NewPointerObj($1,SWIGTYPE_p_gdcm__VL,0);
//}
%include "gdcmVR.h"
EXTEND_CLASS_PRINT(gdcm::VR)
%include "gdcmVM.h"
EXTEND_CLASS_PRINT(gdcm::VM)
//%template (FilenameType) std::string;
%template (FilenamesType) std::vector<std::string>;
%include "gdcmDirectory.h"
EXTEND_CLASS_PRINT(gdcm::Directory)
//%clear FilenameType;
%clear FilenamesType;
%include "gdcmObject.h"
%include "gdcmValue.h"
EXTEND_CLASS_PRINT(gdcm::Value)
%ignore gdcm::ByteValue::WriteBuffer(std::ostream &os) const;
%ignore gdcm::ByteValue::GetPointer() const;
%ignore gdcm::ByteValue::GetBuffer(char *buffer, unsigned long length) const;
%include "gdcmByteValue.h"
EXTEND_CLASS_PRINT(gdcm::ByteValue)
%extend gdcm::ByteValue
{
  std::string WriteBuffer() const {
    std::ostringstream os;
    self->WriteBuffer(os);
    return os.str();
  }
  std::string GetBuffer() const {
    std::ostringstream os;
    self->WriteBuffer(os);
    return os.str();
  }
  std::string GetBuffer(unsigned long length) const {
    std::ostringstream os;
    self->WriteBuffer(os);
    std::string copy( os.str().c_str(), length);
    return copy;
  }
};
%include "gdcmASN1.h"
%include "gdcmSmartPointer.h"
%template(SmartPtrSQ) gdcm::SmartPointer<gdcm::SequenceOfItems>;
%template(SmartPtrFrag) gdcm::SmartPointer<gdcm::SequenceOfFragments>;
%include "gdcmDataElement.h"
EXTEND_CLASS_PRINT(gdcm::DataElement)
%include "gdcmItem.h"
EXTEND_CLASS_PRINT(gdcm::Item)
%template() std::vector< gdcm::Item >;
%include "gdcmSequenceOfItems.h"
EXTEND_CLASS_PRINT(gdcm::SequenceOfItems)
%rename (PythonDataSet) SWIGDataSet;
%rename (PythonTagToValue) SWIGTagToValue;
%include "gdcmDataSet.h"
//namespace std {
//  //struct lttag
//  //  {
//  //  bool operator()(const gdcm::DataElement &s1,
//  //    const gdcm::DataElement &s2) const
//  //    {
//  //    return s1.GetTag() < s2.GetTag();
//  //    }
//  //  };
//
//  //%template(DataElementSet) gdcm::DataSet::DataElementSet;
//  %template(DataElementSet) set<DataElement, lttag>;
//}
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
//%rename(DataElementSetPython) std::set<DataElement, lttag>;
//%rename(DataElementSetPython2) DataSet::DataElementSet;
%template (DataElementSet) std::set<gdcm::DataElement>;
//%rename (SetString2) gdcm::DataElementSet;
%include "gdcmPreamble.h"
EXTEND_CLASS_PRINT(gdcm::Preamble)
%include "gdcmTransferSyntax.h"
EXTEND_CLASS_PRINT(gdcm::TransferSyntax)
%include "gdcmFileMetaInformation.h"
EXTEND_CLASS_PRINT(gdcm::FileMetaInformation)
%include "gdcmFile.h"
EXTEND_CLASS_PRINT(gdcm::File)
//%newobject gdcm::Image::GetBuffer;
%include "cstring.i"
%typemap(out) const unsigned int *GetDimensions {
 int i;
 int n = arg1->GetNumberOfDimensions();
 $result = PyList_New(n);
 for (i = 0; i < n; i++) {
   PyObject *o = PyInt_FromLong((long) $1[i]);
   PyList_SetItem($result,i,o);
 }
}
// Grab a 3 element array as a Python 3-tuple
%typemap(in) const unsigned int dims[3] (unsigned int temp[3]) {   // temp[3] becomes a local variable
  int i;
  if (PyTuple_Check($input)) {
    if (!PyArg_ParseTuple($input,"iii",temp,temp+1,temp+2)) {
      PyErr_SetString(PyExc_TypeError,"tuple must have 3 elements");
      return NULL;
    }
    $1 = &temp[0];
  } else {
    PyErr_SetString(PyExc_TypeError,"expected a tuple.");
    return NULL;
  }
}
%ignore gdcm::Bitmap::GetBuffer(char*) const;
%include "gdcmBitmap.h"
%clear const unsigned int dims[3];
EXTEND_CLASS_PRINT(gdcm::Bitmap)
%extend gdcm::Bitmap
{
  // http://mail.python.org/pipermail/python-list/2006-January/361540.html
  %cstring_output_allocate_size(char **buffer, unsigned int *size, free(*$1) );
  void GetBuffer(char **buffer, unsigned int *size) {
    *size = self->GetBufferLength();
    *buffer = (char*)malloc(*size);
    self->GetBuffer(*buffer);
  }
};
%include "gdcmIconImage.h"
EXTEND_CLASS_PRINT(gdcm::IconImage)
%include "gdcmPixmap.h"
EXTEND_CLASS_PRINT(gdcm::Pixmap)
%typemap(out) const double *GetOrigin, const double *GetSpacing {
 int i;
 $result = PyList_New(3);
 for (i = 0; i < 3; i++) {
   PyObject *o = PyFloat_FromDouble((double) $1[i]);
   PyList_SetItem($result,i,o);
 }
}
%typemap(out) const double *GetDirectionCosines {
 int i;
 $result = PyList_New(6);
 for (i = 0; i < 6; i++) {
   PyObject *o = PyFloat_FromDouble((double) $1[i]);
   PyList_SetItem($result,i,o);
 }
}
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

%template(DictEntryTagPairType) std::pair< gdcm::DictEntry, gdcm::Tag>;
%include "gdcmDict.h"
EXTEND_CLASS_PRINT(gdcm::Dict)
%extend gdcm::Dict
{
  std::pair<gdcm::DictEntry,gdcm::Tag> GetDictEntryByKeyword(const char *keyword) const {
    std::pair<gdcm::DictEntry,gdcm::Tag> ret;
    ret.first = self->GetDictEntryByKeyword(keyword, ret.second);
    return ret;
  }
}
%ignore gdcm::Dict::GetDictEntryByKeyword(const char *keyword, Tag & tag) const;
%include "gdcmCSAHeaderDict.h"
EXTEND_CLASS_PRINT(gdcm::CSAHeaderDictEntry)
%include "gdcmDicts.h"
EXTEND_CLASS_PRINT(gdcm::Dicts)

%template (TagSetType) std::set<gdcm::Tag>;
%ignore gdcm::Reader::SetStream;
%exception ReadFooBar {
   try {
      $action
   } catch (std::exception &e) {
      PyErr_SetString(PyExc_IndexError, const_cast<char*>(e.what()));
      return false;
   } catch ( ... ) {
      PyErr_SetString(PyExc_IndexError, "foobarstuff");
      return false;
   }
}
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
//%include "gdcmConstCharWrapper.h"
//%{
//  typedef char * PString;   // copied to wrapper code
//%}
//%template (FilenameToValue) std::map<const char*,const char*>;
//%template (FilenameToValue) std::map<PString,PString>;
//%template (FilenameToValue) std::map<std::string,std::string>;
//%template (MappingType)     std::map<gdcm::Tag,FilenameToValue>;
//%template (StringArray)     std::vector<const char*>;
%template (ValuesType)      std::set<std::string>;
%rename (PythonTagToValue) SWIGTagToValue;
//%template (TagToValue)      std::map<gdcm::Tag,const char*>;
//%template (TagToValue)      std::map<gdcm::Tag,gdcm::ConstCharWrapper>;
//%template (stdFilenameToValue) std::map<const char*,const char*>;
//namespace gdcm
//{
//  class FilenameToValue : public std::map<const char*, const char*>
//  {
//    void foo();
//  };
//}
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
%include "gdcmSystem.h"
//EXTEND_CLASS_PRINT(gdcm::System)

%include "gdcmTrace.h"
//EXTEND_CLASS_PRINT(gdcm::Trace)
%include "gdcmUIDs.h"
EXTEND_CLASS_PRINT(gdcm::UIDs)
//%feature("director") gdcm::IPPSorter;

%{
static bool callback_helper(gdcm::DataSet const & ds1, gdcm::DataSet const & ds2)
{
  PyObject *func, *arglist, *result;
  func = 0; //(PyObject *)data;
  if (!(arglist = Py_BuildValue("()"))) {
    /* fail */
    assert(0);
  }
  result = PyEval_CallObject(func, arglist);
  Py_DECREF(arglist);
  if (result && result != Py_None) {
    PyErr_SetString(PyExc_TypeError,
                    "Callback function should return nothing");
    Py_DECREF(result);
    /* fail */
    assert(0);
  } else if (!result) {
    /* fail: a Python exception was raised */
    assert(0);
  }
  return true;
}
%}
//%{
//static void callback_decref(void *data)
//{
//  /* Lose the reference to the Python callback */
//  Py_DECREF(data);
//}
//%}
%typemap(in) (gdcm::Sorter::SortFunction f) {
  if (!PyCallable_Check($input)) {
    PyErr_SetString(PyExc_TypeError, "Need a callable object!");
    SWIG_fail;
  }
  $1 = callback_helper;
//  $2 = (void *)$input;
//  $2 = callback_decref;
//  $3 = (void *)$input;
  /* Keep a reference to the Python callback */
  Py_INCREF($input);
}

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

// Grab a 6 element array as a Python 6-tuple
%typemap(in) const double dircos[6] (double temp[6]) {   // temp[6] becomes a local variable
  int i;
  if (PyTuple_Check($input) /*|| PyList_Check($input)*/) {
    if (!PyArg_ParseTuple($input,"dddddd",temp,temp+1,temp+2,temp+3,temp+4,temp+5)) {
      PyErr_SetString(PyExc_TypeError,"list must have 6 elements");
      return NULL;
    }
    $1 = &temp[0];
  } else {
    PyErr_SetString(PyExc_TypeError,"expected a list.");
    return NULL;
  }
}
%include "gdcmOrientation.h"
EXTEND_CLASS_PRINT(gdcm::Orientation)
//%typemap(argout) double z[3] {   // temp[6] becomes a local variable
// int i;
// $result = PyList_New(3);
// for (i = 0; i < 3; i++) {
//   PyObject *o = PyFloat_FromDouble((double) $1[i]);
//   PyList_SetItem($result,i,o);
// }
//}
//%typemap(in,numinputs=0) double z[3] (double temp[3]) {
//    $1[0] = temp[0];
//    $1[1] = temp[1];
//    $1[2] = temp[2];
//}
%include "gdcmDirectionCosines.h"
EXTEND_CLASS_PRINT(gdcm::DirectionCosines)
//%clear const double dircos[6];

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
%include "gdcmRescaler.h"
%include "gdcmSegmentedPaletteColorLookupTable.h"
%include "gdcmUnpacker12Bits.h"

%include "gdcmConfigure.h"
#ifdef GDCM_BUILD_TESTING
%include "gdcmTesting.h"
%ignore gdcm::Testing::ComputeMD5(const char *, const unsigned long , char []);
%ignore gdcm::Testing::ComputeFileMD5(const char*, char []);
%extend gdcm::Testing
{
  //static const char *ComputeMD5(const char *buffer) {
  //  static char buffer[33];
  //  gdcm::Testing::ComputeFileMD5(filename, buffer);
  //  return buffer;
  //}
  static const char *ComputeFileMD5(const char *filename) {
    static char buffer[33];
    gdcm::Testing::ComputeFileMD5(filename, buffer);
    return buffer;
  }
};
#endif
%include "gdcmPythonFilter.h"
%include "gdcmTagPath.h"
%include "gdcmBitmapToBitmapFilter.h"
%include "gdcmPixmapToPixmapFilter.h"
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
%include "gdcmBase64.h"
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
%include "gdcmJSON.h"
