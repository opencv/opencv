//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Weta Digital, Ltd and Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class StringVectorAttribute
//
//-----------------------------------------------------------------------------

#define COMPILING_IMF_STRING_VECTOR_ATTRIBUTE

#include "ImfStringVectorAttribute.h"


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


using namespace OPENEXR_IMF_INTERNAL_NAMESPACE;


template <>
IMF_EXPORT const char *
StringVectorAttribute::staticTypeName ()
{
    return "stringvector";
}


template <>
IMF_EXPORT void
StringVectorAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    int size = _value.size();

    for (int i = 0; i < size; i++)
    {
        int strSize = _value[i].size();
        Xdr::write <StreamIO> (os, strSize);
	Xdr::write <StreamIO> (os, &_value[i][0], strSize);
    }
}


template <>
IMF_EXPORT void
StringVectorAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    int read = 0;

    while (read < size)
    {   
       int strSize;
       Xdr::read <StreamIO> (is, strSize);
       read += Xdr::size<int>();       

       // check there is enough space remaining in attribute to
       // contain claimed string length
       if( strSize < 0 ||  strSize > size - read)
       {
           throw IEX_NAMESPACE::InputExc("Invalid size field reading stringvector attribute");
       }

       std::string str;
       str.resize (strSize);
  
       if( strSize>0 )
       {
           Xdr::read<StreamIO> (is, &str[0], strSize);
       }
       
       read += strSize;

       _value.push_back (str);
    }
}

template class IMF_EXPORT_TEMPLATE_INSTANCE TypedAttribute<OPENEXR_IMF_INTERNAL_NAMESPACE::StringVector>;


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT 
