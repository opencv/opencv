//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_ATTRIBUTE_H
#define INCLUDED_IMF_ATTRIBUTE_H

//-----------------------------------------------------------------------------
//
//	class Attribute
//
//-----------------------------------------------------------------------------

#include "ImfForward.h"

#include "ImfIO.h"
#include "ImfXdr.h"

#include "IexBaseExc.h"

#include <typeinfo>
#include <cstring>


#if defined(_MSC_VER)
// suppress warning about non-exported base classes
#pragma warning (push)
#pragma warning (disable : 4251)
#pragma warning (disable : 4275)
#endif

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class IMF_EXPORT_TYPE Attribute
{
  public:

    //---------------------------
    // Constructor and destructor
    //---------------------------

    IMF_EXPORT Attribute ();
    IMF_EXPORT virtual ~Attribute ();


    //-------------------------------
    // Get this attribute's type name
    //-------------------------------

    virtual const char *	typeName () const = 0;


    //------------------------------
    // Make a copy of this attribute
    //------------------------------

    virtual Attribute *		copy () const = 0;


    //----------------------------------------
    // Type-specific attribute I/O and copying
    //----------------------------------------

    virtual void		writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
					      int version) const = 0;

    virtual void		readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
					       int size,
					       int version) = 0;

    virtual void		copyValueFrom (const Attribute &other) = 0;


    //------------------
    // Attribute factory
    //------------------

    IMF_EXPORT static Attribute *		newAttribute (const char typeName[]);


    //-----------------------------------------------------------
    // Test if a given attribute type has already been registered
    //-----------------------------------------------------------

    IMF_EXPORT static bool			knownType (const char typeName[]);

  protected:

    //--------------------------------------------------
    // Register an attribute type so that newAttribute()
    // knows how to make objects of this type.
    //--------------------------------------------------
    IMF_EXPORT
    static void		registerAttributeType (const char typeName[],
					       Attribute *(*newAttribute)());

    //------------------------------------------------------
    // Un-register an attribute type so that newAttribute()
    // no longer knows how to make objects of this type (for
    // debugging only).
    //------------------------------------------------------
    IMF_EXPORT
    static void		unRegisterAttributeType (const char typeName[]);
};

//-------------------------------------------------
// Class template for attributes of a specific type
//-------------------------------------------------
    
template <class T>
class IMF_EXPORT_TEMPLATE_TYPE TypedAttribute: public Attribute
{
  public:

    //------------------------------------------------------------
    // Constructors and destructor: default behavior. This assumes
    // that the type T is copyable/assignable/moveable.
    //------------------------------------------------------------

    TypedAttribute () = default;
    TypedAttribute (const T &value);
    TypedAttribute (const TypedAttribute<T> &other) = default;
    TypedAttribute (TypedAttribute<T> &&other) = default;
    //NB: if we use a default destructor, it wreaks havoc with where the vtable and such end up
    //at least under mingw+windows, and since we are providing extern template instantiations
    //this will be pretty trim and should reduce code bloat
    virtual ~TypedAttribute ();

    TypedAttribute& operator = (const TypedAttribute<T>& other) = default;
    TypedAttribute& operator = (TypedAttribute<T>&& other) = default;
    
    //--------------------------------
    // Access to the attribute's value
    //--------------------------------

    T &      value ();
    const T &value () const;


    //--------------------------------
    // Get this attribute's type name.
    //--------------------------------

    virtual const char *		typeName () const;
    

    //---------------------------------------------------------
    // Static version of typeName()
    // This function must be specialized for each value type T.
    //---------------------------------------------------------

    static const char *			staticTypeName ();
    

    //---------------------
    // Make a new attribute
    //---------------------

    static Attribute *			makeNewAttribute ();


    //------------------------------
    // Make a copy of this attribute
    //------------------------------

    virtual Attribute *			copy () const;


    //-----------------------------------------------------------------
    // Type-specific attribute I/O and copying.
    // Depending on type T, these functions may have to be specialized.
    //-----------------------------------------------------------------

    virtual void	writeValueTo (
        OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
        int version) const;

    virtual void	readValueFrom (
        OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
        int size,
        int version);

    virtual void	copyValueFrom (const Attribute &other);


    //------------------------------------------------------------
    // Dynamic casts that throw exceptions instead of returning 0.
    //------------------------------------------------------------

    static TypedAttribute *       cast (Attribute *attribute);
    static const TypedAttribute * cast (const Attribute *attribute);
    static TypedAttribute &       cast (Attribute &attribute);
    static const TypedAttribute & cast (const Attribute &attribute);


    //---------------------------------------------------------------
    // Register this attribute type so that Attribute::newAttribute()
    // knows how to make objects of this type.
    //
    // Note that this function is not thread-safe because it modifies
    // a global variable in the IlmIlm library.  A thread in a multi-
    // threaded program may call registerAttributeType() only when no
    // other thread is accessing any functions or classes in the
    // OpenEXR library.
    //
    //---------------------------------------------------------------

    static void registerAttributeType ();


    //-----------------------------------------------------
    // Un-register this attribute type (for debugging only)
    //-----------------------------------------------------

    static void unRegisterAttributeType ();


  private:

    T					_value;
};

//------------------------------------
// Implementation of TypedAttribute<T>
//------------------------------------

template <class T>
TypedAttribute<T>::TypedAttribute (const T & value):
    Attribute (),
    _value (value)
{
    // empty
}

template <class T>
TypedAttribute<T>::~TypedAttribute ()
{
    // empty
}

template <class T>
inline T &
TypedAttribute<T>::value ()
{
    return _value;
}


template <class T>
inline const T &
TypedAttribute<T>::value () const
{
    return _value;
}


template <class T>
const char *	
TypedAttribute<T>::typeName () const
{
    return staticTypeName();
}


template <class T>
Attribute *
TypedAttribute<T>::makeNewAttribute ()
{
    return new TypedAttribute<T>();
}


template <class T>
Attribute *
TypedAttribute<T>::copy () const
{
    Attribute * attribute = new TypedAttribute<T>();
    attribute->copyValueFrom (*this);
    return attribute;
}


template <class T>
void		
TypedAttribute<T>::writeValueTo (OPENEXR_IMF_INTERNAL_NAMESPACE::OStream &os,
                                    int version) const
{
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::write <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (os, _value);
}


template <class T>
void		
TypedAttribute<T>::readValueFrom (OPENEXR_IMF_INTERNAL_NAMESPACE::IStream &is,
                                     int size,
                                     int version)
{
    OPENEXR_IMF_INTERNAL_NAMESPACE::Xdr::read <OPENEXR_IMF_INTERNAL_NAMESPACE::StreamIO> (is, _value);
}


template <class T>
void		
TypedAttribute<T>::copyValueFrom (const Attribute &other)
{
    _value = cast(other)._value;
}


template <class T>
TypedAttribute<T> *
TypedAttribute<T>::cast (Attribute *attribute)
{
    TypedAttribute<T> *t =
	dynamic_cast <TypedAttribute<T> *> (attribute);

    if (t == 0)
	throw IEX_NAMESPACE::TypeExc ("Unexpected attribute type.");

    return t;
}


template <class T>
const TypedAttribute<T> *
TypedAttribute<T>::cast (const Attribute *attribute)
{
    const TypedAttribute<T> *t =
	dynamic_cast <const TypedAttribute<T> *> (attribute);

    if (t == 0)
	throw IEX_NAMESPACE::TypeExc ("Unexpected attribute type.");

    return t;
}


template <class T>
inline TypedAttribute<T> &
TypedAttribute<T>::cast (Attribute &attribute)
{
    return *cast (&attribute);
}


template <class T>
inline const TypedAttribute<T> &
TypedAttribute<T>::cast (const Attribute &attribute)
{
    return *cast (&attribute);
}


template <class T>
inline void
TypedAttribute<T>::registerAttributeType ()
{
    Attribute::registerAttributeType (staticTypeName(), makeNewAttribute);
}


template <class T>
inline void
TypedAttribute<T>::unRegisterAttributeType ()
{
    Attribute::unRegisterAttributeType (staticTypeName());
}


OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#if defined(_MSC_VER)
#pragma warning (pop)
#endif

#endif
