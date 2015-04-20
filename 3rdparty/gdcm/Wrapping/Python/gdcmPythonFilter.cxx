/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "gdcmPythonFilter.h"
#include "gdcmGlobal.h"
#include "gdcmElement.h"
#include "gdcmByteValue.h"
#include "gdcmAttribute.h"
#include "gdcmVR.h"

#include <sstream>

namespace gdcm
{
// Py_BuildValue:
// http://www.python.org/doc/1.5.2p2/ext/buildValue.html

PythonFilter::PythonFilter():F(new File)
{
}
//-----------------------------------------------------------------------------
PythonFilter::~PythonFilter()
{
}

void PythonFilter::SetDicts(const Dicts &dicts)
{
  assert(0); // FIXME
}

static const char *PythonTypesFromVR[] = {
0, //  "??",        // 0
"s", //  "AE",        // 1
"s", //  "AS",        // 2
"(ii)", //  "AT",        // 3
"s", //  "CS",        // 4
"s", //  "DA",        // 5
"s", //  "DS",        // 6
"s", //  "DT",        // 7
"d", //  "FD",        // 8
"d", //  "FL",        // 9
"i", //  "IS",        // 10
"s", //  "LO",        // 11
"s", //  "LT",        // 12
"s", //  "OB",        // 13
"d", //  "OF",        // 14
"s", //  "OW",        // 15
"s", //  "PN",        // 16
"s", //  "SH",        // 17
"i", //  "SL",        // 18
"s", //  "SQ",        // 19
"i", //  "SS",        // 20
"s", //  "ST",        // 21
"s", //  "TM",        // 22
"s", //  "UI",        // 23
"i", //  "UL",        // 24
"s", //  "UN",        // 25
"i", //  "US",        // 26
"s", //  "UT",        // 27
};
const char *GetPythonTypeFromVR(VR const &vr)
{
//  return PythonTypesFromVR[ (int)vr ];
  const char *s;
  switch(vr)
    {
    case VR::INVALID:
      s = 0;
      break;
    case VR::AE:
      s = "s";
      break;
    case VR::AS:
      s = "s";
      break;
    case VR::AT:
      s = "(ii)";
      break;
    case VR::CS:
      s = "s";
      break;
    case VR::DA:
      s = "s";
      break;
    case VR::DS:
      s = "d";
      break;
    case VR::DT:
      s = "s";
      break;
    case VR::FD:
      s = "d";
      break;
    case VR::FL:
      s = "d";
      break;
    case VR::IS:
      s = "i";
      break;
    case VR::LO:
      s = "s";
      break;
    case VR::LT:
      s = "s";
      break;
    case VR::OB:
      s = "s";
      break;
    case VR::OF:
      s = "d";
      break;
    case VR::OW:
      s = "s";
      break;
    case VR::PN:
      s = "s";
      break;
    case VR::SH:
      s = "s";
      break;
    case VR::SL:
      s = "i";
      break;
    case VR::SQ:
      s = "s";
      break;
    case VR::SS:
      s = "i";
      break;
    case VR::ST:
      s = "s";
      break;
    case VR::TM:
      s = "s";
      break;
    case VR::UI:
      s = "s";
      break;
    case VR::UL:
      s = "i";
      break;
    case VR::UN:
      s = "s";
      break;
    case VR::US:
      s = "i";
      break;
    case VR::UT:
      s = "s";
      break;
    default:
      assert( 0 );
      s = 0;
    }
  return s;
}

template <int T, typename helper /*= VR::VRToType<T>::Type*/ >
PyObject *DataElementToPyObject(DataElement const &de, VR const &vr)
{
      const ByteValue *bv = de.GetByteValue();
      std::string s( bv->GetPointer(), bv->GetLength() );
      s.resize( std::min( s.size(), strlen( s.c_str() ) ) ); // strlen is garantee to be lower or equal to ::size()
      // http://www.python.org/doc/current/ext/buildValue.html
      // http://mail.python.org/pipermail/python-list/2002-April/137612.html
      unsigned int count;
      if( vr & VR::VRASCII )
             count = VM::GetNumberOfElementsFromArray(bv->GetPointer(), bv->GetLength());
      else /*( vr & VR::VRASCII ) */
             count = bv->GetLength() / vr.GetSize();
      const char *ptype = GetPythonTypeFromVR( vr );
//std::cout << "DEBUG:" << ptype << std::endl;
      Element<T,VM::VM1_n> el;
      //el.SetLength( count * sizeof(typename Element<T,VM::VM1_n>::Type) );
      el.Set( de.GetValue() );
      PyObject *o;
      if( count == 0 )
      {
              o = 0;
      }
      else if( count == 1 )
      {
        helper s = el[0];
        o = Py_BuildValue((char*)ptype, s);
       }
      else
      {

      PyObject* tuple = PyTuple_New(count);

      for (int i = 0; i < count; i++) {
        //double rVal = data[i];
        //PyTuple_SetItem(tuple, i, Py_BuildValue("d", rVal));
        helper s = el[i];
        //PyTuple_SetItem(tuple, i, Py_BuildValue("s", s));
        PyTuple_SetItem(tuple, i, Py_BuildValue((char*)ptype, s));
      }
      o = tuple;
      }


      Py_INCREF(o);
  return o;
}

PyObject *PythonFilter::ToPyObject(const Tag& t) const
{
  const Global &g = GlobalInstance;
  const Dicts &dicts = g.GetDicts();
  const DataSet &ds = GetFile().GetDataSet();
  if( ds.IsEmpty() || !ds.FindDataElement(t) )
    {
    gdcmWarningMacro( "DataSet is empty or does not contains tag:" );
    return 0;
    }
  if( t.IsPrivate() )
    {
    return 0;
    }

  const DataElement &de = ds.GetDataElement( t );
  assert( de.GetTag().IsPublic() );
  const DictEntry &entry = dicts.GetDictEntry(de.GetTag());
  if( entry.GetVR() == VR::INVALID )
    {
    // FIXME This is a public element we do not support...
    //throw Exception();
    return 0;
    }

  VR vr = entry.GetVR();
  VM vm = entry.GetVM();
  // If Explicit override with coded VR:
  if( de.GetVR() != VR::INVALID && de.GetVR() != VR::UN )
    {
    vr = de.GetVR();
    }
  assert( vr != VR::UN && vr != VR::INVALID );
  //std::cerr << "Found " << vr << " for " << de.GetTag() << std::endl;
  //if( VR::IsASCII( vr ) )
    {
    //assert( vr & VR::VRASCII );
    if( de.IsEmpty() )
      {
      return 0;
      }
    else
      {
      PyObject *o;
      switch(vr)
        {
      case VR::CS:
      o = DataElementToPyObject<VR::CS, const char *>(de, vr);
        break;
      case VR::DS:
      o = DataElementToPyObject<VR::DS, double>(de, vr);
        break;
      case VR::SH:
      o = DataElementToPyObject<VR::SH, const char *>(de, vr);
        break;
      case VR::US:
      o = DataElementToPyObject<VR::US, unsigned short>(de, vr);
        break;
        }
      return o;
      }
    }

  PyObject *o = Py_BuildValue("s", "unhandled" );
  Py_INCREF(o);

   return o;
}

}
