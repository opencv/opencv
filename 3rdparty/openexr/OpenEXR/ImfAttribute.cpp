//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//-----------------------------------------------------------------------------
//
//	class Attribute
//
//-----------------------------------------------------------------------------

#include <ImfAttribute.h>
#include <Iex.h>
#include <string.h>
#include <map>

#include <IlmThreadConfig.h>

#if ILMTHREAD_THREADING_ENABLED
#include <mutex>
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_ENTER 

Attribute::Attribute () {}


Attribute::~Attribute () {}


namespace {

struct NameCompare
{
    bool
    operator () (const char *x, const char *y) const
    {
	return strcmp (x, y) < 0;
    }
};


typedef Attribute* (*Constructor)();
typedef std::map <const char *, Constructor, NameCompare> TypeMap;


class LockedTypeMap: public TypeMap
{
  public:

#if ILMTHREAD_THREADING_ENABLED
    std::mutex mutex;
#endif
};


LockedTypeMap &
typeMap ()
{
    static LockedTypeMap tMap;
    return tMap;
}


} // namespace


bool		
Attribute::knownType (const char typeName[])
{
    LockedTypeMap& tMap = typeMap();
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (tMap.mutex);
#endif
    return tMap.find (typeName) != tMap.end();
}


void	
Attribute::registerAttributeType (const char typeName[],
			          Attribute *(*newAttribute)())
{
    LockedTypeMap& tMap = typeMap();
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (tMap.mutex);
#endif
    if (tMap.find (typeName) != tMap.end())
	THROW (IEX_NAMESPACE::ArgExc, "Cannot register image file attribute "
			    "type \"" << typeName << "\". "
			    "The type has already been registered.");

    tMap.insert (TypeMap::value_type (typeName, newAttribute));
}


void
Attribute::unRegisterAttributeType (const char typeName[])
{
    LockedTypeMap& tMap = typeMap();
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (tMap.mutex);
#endif
    tMap.erase (typeName);
}


Attribute *
Attribute::newAttribute (const char typeName[])
{
    LockedTypeMap& tMap = typeMap();
#if ILMTHREAD_THREADING_ENABLED
    std::lock_guard<std::mutex> lock (tMap.mutex);
#endif
    TypeMap::const_iterator i = tMap.find (typeName);

    if (i == tMap.end())
	THROW (IEX_NAMESPACE::ArgExc, "Cannot create image file attribute of "
			    "unknown type \"" << typeName << "\".");

    return (i->second)();
}


OPENEXR_IMF_INTERNAL_NAMESPACE_SOURCE_EXIT
