//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IMF_FLAT_IMAGE_LEVEL_H
#define INCLUDED_IMF_FLAT_IMAGE_LEVEL_H

//----------------------------------------------------------------------------
//
//      class FlatImageLevel
//      class FlatImageLevel::Iterator
//      class FlatImageLevel::ConstIterator
//
//      For an explanation of images, levels and channels,
//      see the comments in header file Image.h.
//
//----------------------------------------------------------------------------

#include "ImfFlatImageChannel.h"
#include "ImfImageLevel.h"
#include "ImfUtilExport.h"
#include <ImathBox.h>
#include <map>
#include <string>

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

class FlatImage;

class IMFUTIL_EXPORT_TYPE FlatImageLevel : public ImageLevel
{
public:
    //
    // Access to the flat image to which the level belongs.
    //

    IMFUTIL_EXPORT
    FlatImage& flatImage ();
    IMFUTIL_EXPORT
    const FlatImage& flatImage () const;

    //
    // Accessing channels by name:
    //
    // findChannel(n)           returns a pointer to the image channel with
    //                          name n, or 0 if no such channel exists.
    //
    // channel(n)               returns a reference to the image channel with
    //                          name n, or throws an Iex::ArgExc exception if
    //                          no such channel exists.
    //
    // findTypedChannel<T>(n)   returns a pointer to the image channel with
    //                          name n and type T, or 0 if no such channel
    //                          exists.
    //
    // typedChannel(n)          returns a reference to the image channel with
    //                          name n and type T, or throws an Iex::ArgExc
    //                          exception if no such channel exists.
    //

    IMFUTIL_EXPORT
    FlatImageChannel* findChannel (const std::string& name);
    IMFUTIL_EXPORT
    const FlatImageChannel* findChannel (const std::string& name) const;

    IMFUTIL_EXPORT
    FlatImageChannel& channel (const std::string& name);
    IMFUTIL_EXPORT
    const FlatImageChannel& channel (const std::string& name) const;

    template <class T>
    TypedFlatImageChannel<T>* findTypedChannel (const std::string& name);

    template <class T>
    const TypedFlatImageChannel<T>*
    findTypedChannel (const std::string& name) const;

    template <class T>
    TypedFlatImageChannel<T>& typedChannel (const std::string& name);

    template <class T>
    const TypedFlatImageChannel<T>&
    typedChannel (const std::string& name) const;

    //
    // Iterator-style access to channels
    //

    typedef std::map<std::string, FlatImageChannel*> ChannelMap;

    class Iterator;
    class ConstIterator;

    IMFUTIL_EXPORT
    Iterator begin ();
    IMFUTIL_EXPORT
    ConstIterator begin () const;

    IMFUTIL_EXPORT
    Iterator end ();
    IMFUTIL_EXPORT
    ConstIterator end () const;

private:
    friend class FlatImage;

    //
    // The constructor and destructor are private.
    // Image levels exist only as part of an image.
    //
    IMFUTIL_HIDDEN
    FlatImageLevel (
        FlatImage&                    image,
        int                           xLevelNumber,
        int                           yLevelNumber,
        const IMATH_NAMESPACE::Box2i& dataWindow);

    IMFUTIL_HIDDEN
    virtual ~FlatImageLevel ();

    IMFUTIL_HIDDEN
    virtual void resize (const IMATH_NAMESPACE::Box2i& dataWindow);

    IMFUTIL_HIDDEN
    virtual void shiftPixels (int dx, int dy);

    IMFUTIL_HIDDEN
    virtual void insertChannel (
        const std::string& name,
        PixelType          type,
        int                xSampling,
        int                ySampling,
        bool               pLinear);

    IMFUTIL_HIDDEN
    virtual void eraseChannel (const std::string& name);

    IMFUTIL_HIDDEN
    virtual void clearChannels ();

    IMFUTIL_HIDDEN
    virtual void
    renameChannel (const std::string& oldName, const std::string& newName);

    IMFUTIL_HIDDEN
    virtual void renameChannels (const RenamingMap& oldToNewNames);

    ChannelMap _channels;
};

class IMFUTIL_EXPORT_TYPE FlatImageLevel::Iterator
{
public:
    IMFUTIL_EXPORT
    Iterator ();
    IMFUTIL_EXPORT
    Iterator (const FlatImageLevel::ChannelMap::iterator& i);

    //
    // Advance the iterator
    //

    IMFUTIL_EXPORT
    Iterator& operator++ ();
    IMFUTIL_EXPORT
    Iterator operator++ (int);

    //
    // Access to the channel to which the iterator points,
    // and to the name of that channel.
    //

    IMFUTIL_EXPORT
    const std::string& name () const;
    IMFUTIL_EXPORT
    FlatImageChannel& channel () const;

private:
    friend class FlatImageLevel::ConstIterator;

    FlatImageLevel::ChannelMap::iterator _i;
};

class IMFUTIL_EXPORT_TYPE FlatImageLevel::ConstIterator
{
public:
    IMFUTIL_EXPORT
    ConstIterator ();
    IMFUTIL_EXPORT
    ConstIterator (const FlatImageLevel::ChannelMap::const_iterator& i);
    IMFUTIL_EXPORT
    ConstIterator (const FlatImageLevel::Iterator& other);

    //
    // Advance the iterator
    //

    IMFUTIL_EXPORT
    ConstIterator& operator++ ();
    IMFUTIL_EXPORT
    ConstIterator operator++ (int);

    //
    // Access to the channel to which the iterator points,
    // and to the name of that channel.
    //

    IMFUTIL_EXPORT
    const std::string& name () const;
    IMFUTIL_EXPORT
    const FlatImageChannel& channel () const;

private:
    friend bool operator== (const ConstIterator&, const ConstIterator&);

    friend bool operator!= (const ConstIterator&, const ConstIterator&);

    FlatImageLevel::ChannelMap::const_iterator _i;
};

//-----------------------------------------------------------------------------
// Implementation of templates and inline functions
//-----------------------------------------------------------------------------

template <class T>
TypedFlatImageChannel<T>*
FlatImageLevel::findTypedChannel (const std::string& name)
{
    return dynamic_cast<TypedFlatImageChannel<T>*> (findChannel (name));
}

template <class T>
const TypedFlatImageChannel<T>*
FlatImageLevel::findTypedChannel (const std::string& name) const
{
    return dynamic_cast<const TypedFlatImageChannel<T>*> (findChannel (name));
}

template <class T>
TypedFlatImageChannel<T>&
FlatImageLevel::typedChannel (const std::string& name)
{
    TypedFlatImageChannel<T>* ptr = findTypedChannel<T> (name);

    if (ptr == 0) throwBadChannelNameOrType (name);

    return *ptr;
}

template <class T>
const TypedFlatImageChannel<T>&
FlatImageLevel::typedChannel (const std::string& name) const
{
    const TypedFlatImageChannel<T>* ptr = findTypedChannel<T> (name);

    if (ptr == 0) throwBadChannelNameOrType (name);

    return *ptr;
}

inline FlatImageLevel::Iterator::Iterator () : _i ()
{
    // empty
}

inline FlatImageLevel::Iterator::Iterator (
    const FlatImageLevel::ChannelMap::iterator& i)
    : _i (i)
{
    // empty
}

inline FlatImageLevel::Iterator&
FlatImageLevel::Iterator::operator++ ()
{
    ++_i;
    return *this;
}

inline FlatImageLevel::Iterator
FlatImageLevel::Iterator::operator++ (int)
{
    Iterator tmp = *this;
    ++_i;
    return tmp;
}

inline const std::string&
FlatImageLevel::Iterator::name () const
{
    return _i->first;
}

inline FlatImageChannel&
FlatImageLevel::Iterator::channel () const
{
    return *_i->second;
}

inline FlatImageLevel::ConstIterator::ConstIterator () : _i ()
{
    // empty
}

inline FlatImageLevel::ConstIterator::ConstIterator (
    const FlatImageLevel::ChannelMap::const_iterator& i)
    : _i (i)
{
    // empty
}

inline FlatImageLevel::ConstIterator::ConstIterator (
    const FlatImageLevel::Iterator& other)
    : _i (other._i)
{
    // empty
}

inline FlatImageLevel::ConstIterator&
FlatImageLevel::ConstIterator::operator++ ()
{
    ++_i;
    return *this;
}

inline FlatImageLevel::ConstIterator
FlatImageLevel::ConstIterator::operator++ (int)
{
    ConstIterator tmp = *this;
    ++_i;
    return tmp;
}

inline const std::string&
FlatImageLevel::ConstIterator::name () const
{
    return _i->first;
}

inline const FlatImageChannel&
FlatImageLevel::ConstIterator::channel () const
{
    return *_i->second;
}

inline bool
operator== (
    const FlatImageLevel::ConstIterator& x,
    const FlatImageLevel::ConstIterator& y)
{
    return x._i == y._i;
}

inline bool
operator!= (
    const FlatImageLevel::ConstIterator& x,
    const FlatImageLevel::ConstIterator& y)
{
    return !(x == y);
}

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
