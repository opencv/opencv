/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
%module(docstring="A VTK/GDCM binding") vtkgdcm
#pragma SWIG nowarn=504,510

//%pragma(csharp) moduleimports=%{
//using Kitware.VTK;
//%}

#if defined(SWIGCSHARP)
%{
#define SWIGCSHARP
%}
#endif

#if defined(SWIGPHP)
%{
#define SWIGPHP
%}
#endif

%{
//#define VTK_MAJOR_VERSION 5
//#define VTK_MINOR_VERSION 4
//#define VTK_BUILD_VERSION 0
//#define VTK_VERSION "5.4.0"
%}


%{
// Let's reproduce the stack of include, when one would include vtkSetGet:
#include "vtkConfigure.h"
#include "vtkType.h"
#include "vtkSystemIncludes.h"
#include "vtkSetGet.h"
#include <sstream>

// Common stuff
#include "vtkObjectBase.h"
#include "vtkObject.h"

#include "vtkStringArray.h"
#include "vtkMatrix4x4.h"
#include "vtkMedicalImageProperties.h"

// Wrap vtkImageData
#include "vtkDataObject.h"
#include "vtkDataSet.h"
#include "vtkImageData.h"
#include "vtkPointSet.h"
#include "vtkPolyData.h"

#include "vtkGDCMTesting.h"

// same for vtkGDCMImageReader / vtkGDCMImageWriter so that we get all
// parent's member class functions properly wrapped. (Update, SetFileName ...)
#include "vtkAlgorithm.h"
#include "vtkImageAlgorithm.h"
#include "vtkThreadedImageAlgorithm.h"
#include "vtkImageWriter.h"
#include "vtkImageReader2.h"
#include "vtkMedicalImageReader2.h"
#include "vtkGDCMImageReader.h"
#include "vtkGDCMImageWriter.h"

#include "vtkImageExport.h"
#include "vtkImageImport.h"
#include "vtkImageCast.h"
#include "vtkVolumeReader.h"
#include "vtkVolume16Reader.h"

#include "vtkWindowToImageFilter.h"

#include "vtkToolkits.h" // VTK_DATA_ROOT
%}

//%typemap(csimports) vtkGDCMImageWriter %{
//%typemap(csimports) SWIGTYPE %{
//// I need to duplicate those also:
//using System;
//using System.Runtime.InteropServices;
//// my special import:
//using Kitware.VTK;
//using Kitware.mummy.Runtime;
//%}

//%pragma(csharp) imclassimports=%{
//using System;
//using System.Runtime.InteropServices;
//using My.Own.Namespace;
//%}

#ifdef USEACTIVIZ
%typemap(csimports) SWIGTYPE %{
// I need to duplicate those also:
using System;
using System.Runtime.InteropServices;
// my special import:
using Kitware.VTK;
//using Kitware.mummy.Runtime;
%}
#endif

#define GDCM_EXPORT
#define VTK_EXPORT
#define VTK_COMMON_EXPORT
#define VTK_FILTERING_EXPORT
#define VTK_IO_EXPORT
#define VTK_IMAGING_EXPORT
#define VTK_RENDERING_EXPORT


// FIXME. Including #include vtkSetGet would not work on siwg 1.3.33 ...
#define vtkGetMacro(name,type) virtual type Get##name ();
#define vtkSetMacro(name,type) virtual void Set##name (type _arg);
#define vtkBooleanMacro(name,type) \
  virtual void name##On (); \
  virtual void name##Off ();
#define vtkGetVector3Macro(name,type) virtual type *Get##name ();
#define vtkGetVector6Macro(name,type) virtual type *Get##name ();
#define vtkGetObjectMacro(name,type)  virtual type *Get##name ();
#define vtkSetClampMacro(name,type,min,max) virtual void Set##name (type _arg);
#define vtkSetStringMacro(name) virtual void Set##name (const char* _arg);
#define vtkGetStringMacro(name) virtual char* Get##name ();
#define vtkGetVectorMacro(name,type,count) virtual type *Get##name ();
#define vtkNotUsed(x) x
#define vtkGetVector2Macro(name,type) virtual type *Get##name ();
#define vtkSetVector2Macro(name,type) virtual void Set##name (type _arg1, type _arg2);
#define vtkSetVector3Macro(name,type) virtual void Set##name (type _arg1, type _arg2, type _arg3);


//%include "vtkConfigure.h"

//%ignore vtkGDCMImageReader::GetOverlay;
//%ignore vtkGDCMImageReader::GetIconImage;
//
//%ignore vtkAlgorithm::GetOutputDataObject;
//%ignore vtkAlgorithm::GetInputDataObject;
//
//%ignore vtkImageAlgorithm::GetOutput;
//%ignore vtkImageAlgorithm::GetInput;
//%ignore vtkImageAlgorithm::GetImageDataInput;

%ignore operator<<(ostream& os, vtkObjectBase& o);

%ignore vtkMatrix4x4::operator[];
%ignore vtkMatrix4x4::Determinant(vtkMatrix4x4 &);
%ignore vtkMatrix4x4::Adjoint(vtkMatrix4x4 *in, vtkMatrix4x4 *out);
%ignore vtkMatrix4x4::Invert(vtkMatrix4x4 *in, vtkMatrix4x4 *out);
%ignore vtkMatrix4x4::Transpose(vtkMatrix4x4 *in, vtkMatrix4x4 *out);
// In VTK 5.8 we have to ignore the const variant:
%ignore vtkMatrix4x4::Invert(const vtkMatrix4x4 *in, vtkMatrix4x4 *out);
%ignore vtkMatrix4x4::Transpose(const vtkMatrix4x4 *in, vtkMatrix4x4 *out);

%ignore vtkImageWriter::GetInput; // I am getting a warning on swig 1.3.33 because of vtkImageAlgorithm.GetInput

// Let's wrap the following constants:
// this is only a subset of vtkSystemIncludes.h :
#define VTK_LUMINANCE       1
#define VTK_LUMINANCE_ALPHA 2
#define VTK_RGB             3
#define VTK_RGBA            4

//#include "vtkConfigure.h"
//#define VTK_USE_64BIT_IDS
//
//#ifdef VTK_USE_64BIT_IDS
//typedef long long vtkIdType;
//#else
//typedef int vtkIdType;
//#endif
//typedef vtkIdType2 vtkIdType;
//%apply vtkIdType { vtkIdType }
//#define vtkIdType vtkIdType;
//%include "vtkType.h"

#ifdef USEACTIVIZ

%typemap(cstype) vtkDataObject * "vtkDataObject"
%typemap(csin) vtkDataObject * "$csinput.GetCppThis()"
/*
  public vtkDataObject GetOutputDataObject(int port) {
    IntPtr cPtr = vtkgdcmPINVOKE.vtkAlgorithm_GetOutputDataObject(swigCPtr, port);
    SWIGTYPE_p_vtkDataObject ret = (cPtr == IntPtr.Zero) ? null : new SWIGTYPE_p_vtkDataObject(cPtr, false);
    return ret;
  }
*/
%typemap(csout) (vtkDataObject*) {
  IntPtr rawCppThisSwig = $imcall;
  vtkDataObject data = new vtkDataObject( rawCppThisSwig, false, false );
  return data;
}

%typemap(cstype) vtkStringArray * "vtkStringArray"
%typemap(csin) vtkStringArray * "$csinput.GetCppThis()"
%typemap(csout) (vtkStringArray*) {
  IntPtr rawCppThisSwig = $imcall;
  vtkStringArray data = new vtkStringArray( rawCppThisSwig, false, false );
  return data;
}

%typemap(cstype) vtkPolyData * "vtkPolyData"
%typemap(csin) vtkPolyData * "$csinput.GetCppThis()"
%typemap(csout) (vtkPolyData*) {
  IntPtr rawCppThisSwig = $imcall;
  vtkPolyData data = new vtkPolyData( rawCppThisSwig, false, false );
  return data;
}

%typemap(cstype) vtkMatrix4x4 * "vtkMatrix4x4"
%typemap(csin) vtkMatrix4x4 * "$csinput.GetCppThis()"
%typemap(csout) (vtkMatrix4x4*) {
  IntPtr rawCppThisSwig = $imcall;
  vtkMatrix4x4 data = new vtkMatrix4x4( rawCppThisSwig, false, false );
  return data;
}

%typemap(cstype) vtkMedicalImageProperties * "vtkMedicalImageProperties"
%typemap(csin) vtkMedicalImageProperties * "$csinput.GetCppThis()"
%typemap(csout) (vtkMedicalImageProperties*) {
  IntPtr rawCppThisSwig = $imcall;
  vtkMedicalImageProperties data = new vtkMedicalImageProperties( rawCppThisSwig, false, false );
  return data;
}

%typemap(cstype) vtkImageData * "vtkImageData"
%typemap(csin) vtkImageData * "$csinput.GetCppThis()"
%typemap(csout) (vtkImageData *) {
  IntPtr rawCppThisSwig = $imcall;
  vtkImageData data = new vtkImageData( rawCppThisSwig, false, false );
  //vtkImageData data = null;
  //bool created;
  //if( IntPtr.Zero != rawCppThisSwig )
  //  {
  //  data = (vtkImageData) Kitware.mummy.Runtime.Methods.CreateWrappedObject(
  //    vtkImageData.MRClassNameKey, rawCppThisSwig, false, out created);
  //  // created is true if the C# object was created by this call, false if it was already cached in the table
  //  }
  return data;
}
//
//%typemap(csout) (vtkDataObject *) {
//    vtkImageData data = null;
////    uint mteStatus = 0;
////    uint maxValue = uint.MaxValue;
////    uint rawRefCount = 0;
////    IntPtr rawCppThis =
////vtkImageAlgorithm_GetOutput_06(base.GetCppThis(), ref mteStatus, ref
////maxValue, ref rawRefCount);
////    IntPtr rawCppThisSwig = $imcall;
////    if (IntPtr.Zero != rawCppThisSwig)
////    {
////        bool flag;
////        data = (vtkImageData) Methods.CreateWrappedObject(mteStatus,
////maxValue, rawRefCount, rawCppThisSwig, true, out flag);
////        if (flag)
////        {
////            data.Register(null);
////        }
////    }
//    return data;
//}
//
#endif //USEACTIVIZ

#ifdef USEACTIVIZ
// By hiding all New operator I make sure that no-one will ever be
// able to create a swig wrap object I did not decide to allow.
// For instance the only two objects allowed for now are:
// - vtkGDCMImageReader
// - vtkGDCMImageWriter
// BUG:
// when using %ignore vtkObjectBase::New()
// the vtkObjectBase_New() function is not generated, which is used
// internally in the new cstor that I provide
%csmethodmodifiers vtkObjectBase::New() "internal new"
%csmethodmodifiers vtkObject::New() "internal new"
%csmethodmodifiers vtkAlgorithm::New() "internal new"
%csmethodmodifiers vtkImageAlgorithm::New() "internal new"
%csmethodmodifiers vtkImageWriter::New() "internal new"
%csmethodmodifiers vtkImageReader2::New() "internal new"
%csmethodmodifiers vtkMedicalImageReader2::New() "internal new"

%csmethodmodifiers vtkGDCMImageReader::New() "public new"
%csmethodmodifiers vtkGDCMImageWriter::New() "public new"
%csmethodmodifiers vtkGDCMTesting::New() "public new"

#endif

%newobject vtkGDCMTesting::New();
%newobject vtkGDCMImageWriter::New();
%newobject vtkGDCMImageReader::New();

%delobject vtkObjectBase::Delete();

// TODO: I need to fix Delete and make sure SWIG owns the C++ ptr (call ->Delete in the Dispose layer)
//%ignore vtkObjectBase::Delete;
%ignore vtkObjectBase::FastDelete;
%ignore vtkObjectBase::PrintSelf;
%ignore vtkObjectBase::PrintHeader;
%ignore vtkObjectBase::PrintTrailer;
%ignore vtkObjectBase::Print;
%ignore vtkObjectBase::PrintRevisions;
%ignore vtkObject::PrintSelf;
%ignore vtkAlgorithm::PrintSelf;
%ignore vtkImageAlgorithm::PrintSelf;
%ignore vtkImageAlgorithm::ProcessRequest;
%ignore vtkImageWriter::PrintSelf;
%ignore vtkImageReader2::PrintSelf;
%ignore vtkMedicalImageReader2::PrintSelf;
%ignore vtkGDCMImageReader::PrintSelf;
%ignore vtkGDCMImageWriter::PrintSelf;

%typemap(csdestruct_derived, methodname="Dispose", methodmodifiers="public") vtkGDCMTesting {
  lock(this) {
    if(swigCPtr.Handle != IntPtr.Zero && swigCMemOwn) {
      swigCMemOwn = false;
      vtkgdcmPINVOKE.vtkObjectBase_Delete(swigCPtr);
    }
    swigCPtr = new HandleRef(null, IntPtr.Zero);
    GC.SuppressFinalize(this);
    base.Dispose();
  }
}
%typemap(csdestruct_derived, methodname="Dispose", methodmodifiers="public") vtkGDCMImageReader {
  lock(this) {
    if(swigCPtr.Handle != IntPtr.Zero && swigCMemOwn) {
      swigCMemOwn = false;
      vtkgdcmPINVOKE.vtkObjectBase_Delete(swigCPtr);
    }
    swigCPtr = new HandleRef(null, IntPtr.Zero);
    GC.SuppressFinalize(this);
    base.Dispose();
  }
}
%typemap(csdestruct_derived, methodname="Dispose", methodmodifiers="public") vtkGDCMImageWriter {
  lock(this) {
    if(swigCPtr.Handle != IntPtr.Zero && swigCMemOwn) {
      swigCMemOwn = false;
      vtkgdcmPINVOKE.vtkObjectBase_Delete(swigCPtr);
    }
    swigCPtr = new HandleRef(null, IntPtr.Zero);
    GC.SuppressFinalize(this);
    base.Dispose();
  }
}

%include "vtkObjectBase.h"
#ifdef SWIGCSHARP
%csmethodmodifiers vtkObjectBase::ToString() "public override"
#endif
%extend vtkObjectBase
{
  const char *ToString()
    {
    static std::string buffer;
    std::ostringstream os;
    self->Print( os );
    buffer = os.str();
    return buffer.c_str();
    }
};

%include "vtkObject.h"

%defaultdtor vtkGDCMTesting; // FIXME does not seems to be working
%include "vtkGDCMTesting.h"

#ifndef USEACTIVIZ
%include "vtkStringArray.h"
%include "vtkMatrix4x4.h"
%include "vtkMedicalImageProperties.h"
%include "vtkDataObject.h"
%include "vtkDataSet.h"
%include "vtkImageData.h"
%include "vtkPointSet.h"
%include "vtkPolyData.h"
#endif

%include "vtkAlgorithm.h"
%include "vtkImageAlgorithm.h"
#ifndef USEACTIVIZ
%include "vtkThreadedImageAlgorithm.h"
#endif
%include "vtkImageWriter.h"

/*
By default swig generates:
  public virtual SWIGTYPE_p_double GetImageOrientationPatient() {
    IntPtr cPtr = vtkgdcmPINVOKE.vtkGDCMImageReader_GetImageOrientationPatient(swigCPtr);
    SWIGTYPE_p_double ret = (cPtr == IntPtr.Zero) ? null : new SWIGTYPE_p_double(cPtr, false);
    return ret;
  }
while we would want:
  public virtual double[] GetImageOrientationPatient() {
    IntPtr source = vtkgdcmPINVOKE.vtkGDCMImageReader_GetImageOrientationPatient(swigCPtr);
    double[] ret = null;
    if (IntPtr.Zero != source)
    {
        ret = new double[6];
        Marshal.Copy(source, destination, 0, destination.Length);
    }
    return ret;
  }

*/

//%typemap(ctype) double[] "double*"
%typemap(cstype) double * "double[]"
%typemap(csout) double* GetImagePositionPatient() {
    IntPtr source = $imcall;
    double[] destination = null;
    if (IntPtr.Zero != source) {
      destination = new double[3];
      Marshal.Copy(source, destination, 0, destination.Length);
    }
    return destination;
  }

%typemap(csout) double* GetImageOrientationPatient() {
    IntPtr source = $imcall;
    double[] destination = null;
    if (IntPtr.Zero != source) {
      destination = new double[6];
      Marshal.Copy(source, destination, 0, destination.Length);
    }
    return destination;
  }

%typemap(csout) double* GetDataSpacing() {
    IntPtr source = $imcall;
    double[] destination = null;
    if (IntPtr.Zero != source) {
      destination = new double[3];
      Marshal.Copy(source, destination, 0, destination.Length);
    }
    return destination;
  }

%typemap(csout) double* GetDataOrigin() {
    IntPtr source = $imcall;
    double[] destination = null;
    if (IntPtr.Zero != source) {
      destination = new double[3];
      Marshal.Copy(source, destination, 0, destination.Length);
    }
    return destination;
  }

%include "vtkImageReader2.h"
%include "vtkMedicalImageReader2.h"

//%rename (vtkGDCMImageReaderInternal) vtkGDCMImageReader;
//%rename (vtkGDCMImageWriterInternal) vtkGDCMImageWriter;

%include "vtkGDCMImageReader.h"
%include "vtkGDCMImageWriter.h"
%extend vtkGDCMTesting
{
%typemap(cscode) vtkGDCMTesting
%{
  public vtkGDCMTesting() : this(vtkgdcmPINVOKE.vtkGDCMTesting_New(), true) {
  }
  ~vtkGDCMTesting() {
    Dispose();
  }
%}
};

%extend vtkGDCMImageReader
{
%typemap(cscode) vtkGDCMImageReader
%{
  public vtkGDCMImageReader() : this(vtkgdcmPINVOKE.vtkGDCMImageReader_New(), true) {
  }
  ~vtkGDCMImageReader() {
    Dispose();
  }
%}
};

#ifdef SWIGPHP
%extend vtkGDCMImageReader
{
//public function __construct2($res=null) {
//  $this->_cPtr=vtkGDCMImageReader_Create();
//}

//%typemap(out) vtkGDCMImageReader* (vtkGDCMImageReader::New)
//%{
//public function __construct($res=null) {
//  $this->_cPtr=vtkGDCMImageReader_Create();
//}
//%}
};
#endif

%extend vtkGDCMImageWriter
{
%typemap(cscode) vtkGDCMImageWriter
%{
  public vtkGDCMImageWriter() : this(vtkgdcmPINVOKE.vtkGDCMImageWriter_New(), true) {
  }
  ~vtkGDCMImageWriter() {
    Dispose();
  }
%}
};
%clear double*;
%clear double* GetDataSpacing();
%clear double* GetDataOrigin();

#ifdef SWIGPHP
%include "vtkWindowToImageFilter.h"
#endif

#ifndef USEACTIVIZ
%include "vtkImageExport.h"
%include "vtkImageImport.h"
%include "vtkImageCast.h"
%include "vtkVolumeReader.h"
%include "vtkVolume16Reader.h"
#endif
