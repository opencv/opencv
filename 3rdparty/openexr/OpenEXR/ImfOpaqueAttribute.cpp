//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class OpaqueAttribute
//
//-----------------------------------------------------------------------------

#include <ImfOpaqueAttribute.h>
#include "Iex.h"
#include <string.h>
#include "ImfNamespace.h"

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OpaqueAttribute::OpaqueAttribute (const char typeName[]):
    _typeName (typeName),
    _dataSize (0)
{
}


OpaqueAttribute::OpaqueAttribute (const OpaqueAttribute &other):
    _typeName (other._typeName),
    _dataSize (other._dataSize),
    _data (other._dataSize)
{
    _data.resizeErase (other._dataSize);
    memcpy ((char *) _data, (const char *) other._data, other._dataSize);
}


OpaqueAttribute::~OpaqueAttribute ()
{
    // empty
}


const char *
OpaqueAttribute::typeName () const
{
    return _typeName.c_str();
}


Attribute *	
OpaqueAttribute::copy () const
{
    return new OpaqueAttribute (*this);
}


void	
OpaqueAttribute::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os, int version) const
{
    Xdr::write <StreamIO> (os, _data, _dataSize);
}


void	
OpaqueAttribute::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is, int size, int version)
{
    _data.resizeErase (size);
    _dataSize = size;
    Xdr::read <StreamIO> (is, _data, size);
}


void	
OpaqueAttribute::copyValueFrom (const Attribute &other)
{
    const OpaqueAttribute *oa = dynamic_cast <const OpaqueAttribute *> (&other);

    if (oa == 0 || _typeName != oa->_typeName)
    {
	THROW (IEX_NAMESPACE::TypeExc, "Cannot copy the value of an "
			     "image file attribute of type "
			     "\"" << other.typeName() << "\" "
			     "to an attribute of type "
			     "\"" << _typeName << "\".");
    }

    _data.resizeErase (oa->_dataSize);
    _dataSize = oa->_dataSize;
    memcpy ((char *) _data, (const char *) oa->_data, oa->_dataSize);
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
