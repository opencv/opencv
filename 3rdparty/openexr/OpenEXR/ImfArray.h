//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


#ifndef INCLUDED_IMF_ARRAY_H
#define INCLUDED_IMF_ARRAY_H

#include "ImfForward.h"

//-------------------------------------------------------------------------
//
// class Array
// class Array2D
//
// "Arrays of T" whose sizes are not known at compile time.
// When an array goes out of scope, its elements are automatically
// deleted.
//
// Usage example:
//
//	struct C
//	{
//	    C ()		{std::cout << "C::C  (" << this << ")\n";};
//	    virtual ~C ()	{std::cout << "C::~C (" << this << ")\n";};
//	};
// 
//	int
//	main ()
//	{
//	    Array <C> a(3);
// 
//	    C &b = a[1];
//	    const C &c = a[1];
//	    C *d = a + 2;
//	    const C *e = a;
// 
//	    return 0;
//	}
//
//-------------------------------------------------------------------------

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_ENTER

template <class T>
class IMF_EXPORT_TEMPLATE_TYPE Array
{
  public:

    //-----------------------------
    // Constructors and destructors
    //-----------------------------

     Array ()				{_data = 0; _size = 0;}
     Array (long size)			{_data = new T[size]; _size = size;}
    ~Array ()				{delete [] _data;}


    //-----------------------------
    // Access to the array elements
    //-----------------------------

    operator T * ()			{return _data;}
    operator const T * () const		{return _data;}


    //------------------------------------------------------
    // Resize and clear the array (the contents of the array
    // are not preserved across the resize operation).
    //
    // resizeEraseUnsafe() is more memory efficient than
    // resizeErase() because it deletes the old memory block
    // before allocating a new one, but if allocating the
    // new block throws an exception, resizeEraseUnsafe()
    // leaves the array in an unusable state.
    //
    //------------------------------------------------------

    void resizeErase (long size);
    void resizeEraseUnsafe (long size);


    //-------------------------------
    // Return the size of this array.
    //-------------------------------

    long size() const   {return _size;}


  private:

    Array (const Array &) = delete;
    Array & operator = (const Array &) = delete;
    Array (Array &&) = delete;
    Array & operator = (Array &&) = delete;

    long _size;
    T * _data;
};


template <class T>
class IMF_EXPORT_TEMPLATE_TYPE Array2D
{
  public:

    //-----------------------------
    // Constructors and destructors
    //-----------------------------

     Array2D ();			// empty array, 0 by 0 elements
     Array2D (long sizeX, long sizeY);	// sizeX by sizeY elements
    ~Array2D ();


    //-----------------------------
    // Access to the array elements
    //-----------------------------

    T *		operator [] (long x);
    const T *	operator [] (long x) const;


    //------------------------------------------------------
    // Resize and clear the array (the contents of the array
    // are not preserved across the resize operation).
    //
    // resizeEraseUnsafe() is more memory efficient than
    // resizeErase() because it deletes the old memory block
    // before allocating a new one, but if allocating the
    // new block throws an exception, resizeEraseUnsafe()
    // leaves the array in an unusable state.
    //
    //------------------------------------------------------

    void resizeErase (long sizeX, long sizeY);
    void resizeEraseUnsafe (long sizeX, long sizeY);


    //-------------------------------
    // Return the size of this array.
    //-------------------------------

    long height() const  {return _sizeX;}
    long width() const   {return _sizeY;}


  private:

    Array2D (const Array2D &) = delete;
    Array2D & operator = (const Array2D &) = delete;
    Array2D (Array2D &&) = delete;
    Array2D & operator = (Array2D &&) = delete;

    long        _sizeX;
    long	_sizeY;
    T *		_data;
};


//---------------
// Implementation
//---------------

template <class T>
inline void
Array<T>::resizeErase (long size)
{
    T *tmp = new T[size];
    delete [] _data;
    _size = size;
    _data = tmp;
}


template <class T>
inline void
Array<T>::resizeEraseUnsafe (long size)
{
    delete [] _data;
    _data = 0;
    _size = 0;
    _data = new T[size];
    _size = size;
}


template <class T>
inline
Array2D<T>::Array2D ():
    _sizeX(0), _sizeY (0), _data (0)
{
    // emtpy
}


template <class T>
inline
Array2D<T>::Array2D (long sizeX, long sizeY):
    _sizeX (sizeX), _sizeY (sizeY), _data (new T[sizeX * sizeY])
{
    // emtpy
}


template <class T>
inline
Array2D<T>::~Array2D ()
{
    delete [] _data;
}


template <class T>
inline T *	
Array2D<T>::operator [] (long x)
{
    return _data + x * _sizeY;
}


template <class T>
inline const T *
Array2D<T>::operator [] (long x) const
{
    return _data + x * _sizeY;
}


template <class T>
inline void
Array2D<T>::resizeErase (long sizeX, long sizeY)
{
    T *tmp = new T[sizeX * sizeY];
    delete [] _data;
    _sizeX = sizeX;
    _sizeY = sizeY;
    _data = tmp;
}


template <class T>
inline void
Array2D<T>::resizeEraseUnsafe (long sizeX, long sizeY)
{
    delete [] _data;
    _data = 0;
    _sizeX = 0;
    _sizeY = 0;
    _data = new T[sizeX * sizeY];
    _sizeX = sizeX;
    _sizeY = sizeY;
}

OPENEXR_IMF_INTERNAL_NAMESPACE_HEADER_EXIT


#endif
