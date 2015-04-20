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

%module(directors="1",docstring="A DICOM library") gdcmswig
#pragma SWIG nowarn=504,510
%{
#include "gdcmTypes.h"
#include "gdcmSmartPointer.h"
#include "gdcmSwapCode.h"
#include "gdcmDirectory.h"
#include "gdcmTesting.h"
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
#include "gdcmPreamble.h"
#include "gdcmFile.h"
#include "gdcmBitmap.h"
#include "gdcmPixmap.h"
#include "gdcmImage.h"
#include "gdcmIconImage.h"
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
//#include "gdcmConstCharWrapper.h"
#include "gdcmScanner.h"
#include "gdcmAttribute.h"
#include "gdcmAnonymizer.h"
#include "gdcmSystem.h"
#include "gdcmTrace.h"
#include "gdcmUIDs.h"
#include "gdcmSorter.h"
#include "gdcmIPPSorter.h"
#include "gdcmSpectroscopy.h"
#include "gdcmPrinter.h"
#include "gdcmDumper.h"
#include "gdcmOrientation.h"
#include "gdcmFiducials.h"
#include "gdcmWaveform.h"
#include "gdcmPersonName.h"
#include "gdcmIconImage.h"
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
#include "gdcmModule.h"
#include "gdcmModules.h"
#include "gdcmDefs.h"
#include "gdcmIOD.h"
#include "gdcmIODs.h"
#include "gdcmTableEntry.h"
#include "gdcmDefinedTerms.h"
#include "gdcmSeries.h"
#include "gdcmModuleEntry.h"
#include "gdcmNestedModuleEntries.h"
#include "gdcmIODEntry.h"
#include "gdcmRescaler.h"
#include "gdcmSegmentedPaletteColorLookupTable.h"
#include "gdcmUnpacker12Bits.h"
#include "gdcmPythonFilter.h"
#include "gdcmDirectionCosines.h"
#include "gdcmTagPath.h"
#include "gdcmPixmapToPixmapFilter.h"
#include "gdcmImageToImageFilter.h"
#include "gdcmSOPClassUIDToIOD.h"
#include "gdcmImageChangeTransferSyntax.h"
#include "gdcmImageApplyLookupTable.h"
#include "gdcmSplitMosaicFilter.h"
//#include "gdcmImageChangePhotometricInterpretation.h"
#include "gdcmImageChangePlanarConfiguration.h"
#include "gdcmImageFragmentSplitter.h"
#include "gdcmDataSetHelper.h"
#include "gdcmFileExplicitFilter.h"
#include "gdcmImageHelper.h"
#include "gdcmMD5.h"
#include "gdcmDummyValueGenerator.h"
#include "gdcmSHA1.h"
//#include "gdcmBase64.h"
#include "gdcmSpacing.h"

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

//%feature("autodoc", "1")
//%include "gdcmTypes.h" // define GDCM_EXPORT so need to be the first one...
#define GDCM_EXPORT
%rename(__add__) gdcm::VL::operator+=;
%include "gdcmSwapCode.h"
%include "gdcmPixelFormat.h"
%extend gdcm::PixelFormat
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmMediaStorage.h"
%rename(__getitem__) gdcm::Tag::operator[];
//%rename(__getattr__) gdcm::Tag::operator[];
%include "gdcmTag.h"
%extend gdcm::Tag
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmPrivateTag.h"
%extend gdcm::PrivateTag
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmVL.h"
%extend gdcm::VL
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
//%typemap(out) int
//{
//    $result = SWIG_NewPointerObj($1,SWIGTYPE_p_gdcm__VL,0);
//}
%include "gdcmVR.h"
%extend gdcm::VR
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmVM.h"
//%template (FilenameType) std::string;
%template (FilenamesType) std::vector<std::string>;
%include "gdcmDirectory.h"
%extend gdcm::Directory
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
%include "gdcmObject.h"
%include "gdcmValue.h"
%extend gdcm::Value
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%ignore gdcm::ByteValue::WriteBuffer(std::ostream &os) const;
%ignore gdcm::ByteValue::GetPointer() const;
%ignore gdcm::ByteValue::GetBuffer(char *buffer, unsigned long length) const;
%include "gdcmByteValue.h"
%extend gdcm::ByteValue
{
  const char *__str__() const {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
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
%include "gdcmSmartPointer.h"
%template(SmartPtrSQ) gdcm::SmartPointer<gdcm::SequenceOfItems>;
%template(SmartPtrFrag) gdcm::SmartPointer<gdcm::SequenceOfFragments>;
%include "gdcmDataElement.h"
%extend gdcm::DataElement
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmItem.h"
%extend gdcm::Item
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmSequenceOfItems.h"
%extend gdcm::SequenceOfItems
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
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
%extend gdcm::DataSet
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
    }
};
//%include "gdcmString.h"
//%include "gdcmTransferSyntax.h"
%include "gdcmPhotometricInterpretation.h"
%include "gdcmObject.h"
%include "gdcmLookupTable.h"
%include "gdcmOverlay.h"
//%include "gdcmVR.h"
//%rename(DataElementSetPython) std::set<DataElement, lttag>;
//%rename(DataElementSetPython2) DataSet::DataElementSet;
%template (DataElementSet) std::set<gdcm::DataElement>;
//%rename (SetString2) gdcm::DataElementSet;
%include "gdcmPreamble.h"
%include "gdcmTransferSyntax.h"
%extend gdcm::TransferSyntax
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmFileMetaInformation.h"
%extend gdcm::FileMetaInformation
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmFile.h"
%extend gdcm::File
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
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
%extend gdcm::Bitmap
{
  // http://mail.python.org/pipermail/python-list/2006-January/361540.html
  %cstring_output_allocate_size(char **buffer, unsigned int *size, free(*$1) );
  void GetBuffer(char **buffer, unsigned int *size) {
    *size = self->GetBufferLength();
    *buffer = (char*)malloc(*size);
    self->GetBuffer(*buffer);
  }

  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }

};
%include "gdcmPixmap.h"
%extend gdcm::Pixmap
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};

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
%extend gdcm::Image
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
%include "gdcmIconImage.h"
%include "gdcmFragment.h"
%include "gdcmPDBElement.h"
%extend gdcm::PDBElement
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmPDBHeader.h"
%extend gdcm::PDBHeader
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
%include "gdcmCSAElement.h"
%extend gdcm::CSAElement
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmCSAHeader.h"
%extend gdcm::CSAHeader
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
%include "gdcmSequenceOfFragments.h"
%include "gdcmBasicOffsetTable.h"
//%include "gdcmLO.h"
%include "gdcmFileSet.h"

%include "gdcmGlobal.h"

%include "gdcmDictEntry.h"
%extend gdcm::DictEntry
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmCSAHeaderDictEntry.h"
%extend gdcm::CSAHeaderDictEntry
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};

%include "gdcmDict.h"
%include "gdcmCSAHeaderDict.h"
%include "gdcmDicts.h"

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
%include "gdcmPixmapReader.h"
%include "gdcmImageReader.h"
%include "gdcmWriter.h"
%include "gdcmPixmapWriter.h"
%include "gdcmImageWriter.h"
%template (PairString) std::pair<std::string,std::string>;
//%template (MyM) std::map<gdcm::Tag,gdcm::ConstCharWrapper>;
%include "gdcmStringFilter.h"
%include "gdcmUIDGenerator.h"
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
//%template (TagToValue)      std::map<gdcm::Tag,const char*>;
//%template (TagToValue)      std::map<gdcm::Tag,gdcm::ConstCharWrapper>;
%include "gdcmScanner.h"
%extend gdcm::Scanner
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
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
%include "gdcmAnonymizer.h"
%include "gdcmSystem.h"
%include "gdcmTrace.h"
%include "gdcmUIDs.h"
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
%extend gdcm::Sorter
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
%include "gdcmIPPSorter.h"
%include "gdcmSpectroscopy.h"
%include "gdcmPrinter.h"
%include "gdcmDumper.h"

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
%extend gdcm::Orientation
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
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
%extend gdcm::DirectionCosines
{
  const char *__str__() {
    static std::string buffer;
    std::stringstream s;
    self->Print(s);
    buffer = s.str();
    return buffer.c_str();
  }
};
//%clear const double dircos[6];

%include "gdcmFiducials.h"
%include "gdcmWaveform.h"
%include "gdcmPersonName.h"
%include "gdcmIconImage.h"
%include "gdcmCurve.h"
%include "gdcmDICOMDIR.h"
%include "gdcmValidate.h"
%include "gdcmApplicationEntity.h"
%include "gdcmDictPrinter.h"
%include "gdcmFilenameGenerator.h"
%include "gdcmVersion.h"
%include "gdcmFilename.h"
%include "gdcmEnumeratedValues.h"
%include "gdcmPatient.h"
%include "gdcmStudy.h"
%include "gdcmModuleEntry.h"
%extend gdcm::ModuleEntry
{
  const char *__str__() {
    static std::string buffer;
    std::ostringstream os;
    os << *self;
    buffer = os.str();
    return buffer.c_str();
  }
};
%include "gdcmNestedModuleEntries.h"
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
%include "gdcmPixmapToPixmapFilter.h"
%include "gdcmImageToImageFilter.h"
%include "gdcmSOPClassUIDToIOD.h"
%include "gdcmImageChangeTransferSyntax.h"
%include "gdcmImageApplyLookupTable.h"
%include "gdcmSplitMosaicFilter.h"
//%include "gdcmImageChangePhotometricInterpretation.h"
%include "gdcmImageChangePlanarConfiguration.h"
%include "gdcmImageFragmentSplitter.h"
%include "gdcmDataSetHelper.h"
%include "gdcmFileExplicitFilter.h"
%template (DoubleType) std::vector<double>;
%include "gdcmImageHelper.h"
%include "gdcmMD5.h"
%include "gdcmDummyValueGenerator.h"
%include "gdcmSHA1.h"
//%include "gdcmBase64.h"
%include "gdcmSpacing.h"
