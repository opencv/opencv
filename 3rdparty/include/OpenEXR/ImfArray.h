///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002, Industrial Light & Magic, a division of Lucas
// Digital Ltd. LLC
// 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Industrial Light & Magic nor the names of
// its contributors may be used to endorse or promote products derived
// from this software without specific prior written permission. 
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
///////////////////////////////////////////////////////////////////////////



#ifndef INCLUDED_IMF_ARRAY_H
#define INCLUDED_IMF_ARRAY_H

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

namespace Imf {


template <class T>
class Array
{
  public:

    //-----------------------------
    // Constructors and destructors
    //-----------------------------

     Array ()				{_data = 0;}
     Array (long size)			{_data = new T[size];}
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


  private:

    Array (const Array &);		// Copying and assignment
    Array & operator = (const Array &);	// are not implemented

    T * _data;
};


template <class T>
class Array2D
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


  private:

    Array2D (const Array2D &);			// Copying and assignment
    Array2D & operator = (const Array2D &);	// are not implemented

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
    _data = tmp;
}


template <class T>
inline void
Array<T>::resizeEraseUnsafe (long size)
{
    delete [] _data;
    _data = 0;
    _data = new T[size];
}


template <class T>
inline
Array2D<T>::Array2D ():
    _sizeY (0), _data (0)
{
    // emtpy
}


template <class T>
inline
Array2D<T>::Array2D (long sizeX, long sizeY):
    _sizeY (sizeY), _data (new T[sizeX * sizeY])
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
    _sizeY = sizeY;
    _data = tmp;
}


template <class T>
inline void
Array2D<T>::resizeEraseUnsafe (long sizeX, long sizeY)
{
    delete [] _data;
    _data = 0;
    _sizeY = 0;
    _data = new T[sizeX * sizeY];
    _sizeY = sizeY;
}


} // namespace Imf

#endif
