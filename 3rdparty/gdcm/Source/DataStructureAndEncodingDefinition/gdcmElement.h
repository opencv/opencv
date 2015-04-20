/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMELEMENT_H
#define GDCMELEMENT_H

#include "gdcmTypes.h"
#include "gdcmVR.h"
#include "gdcmTag.h"
#include "gdcmVM.h"
#include "gdcmByteValue.h"
#include "gdcmDataElement.h"
#include "gdcmSwapper.h"

#include <string>
#include <vector>
#include <sstream>
#include <limits>
#include <cmath>
#include <cstring>

namespace gdcm
{

// Forward declaration
/**
 * \brief EncodingImplementation
 *
 * \note TODO
 */
template<int T> class EncodingImplementation;


/**
 *  \brief A class which is used to produce compile errors for an
 * invalid combination of template parameters.
 *
 * Invalid combinations have specialized declarations with no
 * definition.
 */
template <int TVR, int TVM>
class ElementDisableCombinations {};
template <>
class  ElementDisableCombinations<VR::OB, VM::VM1_n> {};
template <>
class  ElementDisableCombinations<VR::OW, VM::VM1_n> {};
// Make it impossible to compile these other cases
template <int TVM>
class  ElementDisableCombinations<VR::OB, TVM>;
template <int TVM>
class ElementDisableCombinations<VR::OW, TVM>;

/**
 * \brief Element class
 *
 * \note TODO
 */
template<int TVR, int TVM>
class Element
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<TVR, TVM> ) };
public:
  typename VRToType<TVR>::Type Internal[VMToLength<TVM>::Length];
  typedef typename VRToType<TVR>::Type Type;

  static VR  GetVR()  { return (VR::VRType)TVR; }
  static VM  GetVM()  { return (VM::VMType)TVM; }

  unsigned long GetLength() const {
    return VMToLength<TVM>::Length;
  }
  // Implementation of Print is common to all Mode (ASCII/Binary)
  // TODO: Can we print a \ when in ASCII...well I don't think so
  // it would mean we used a bad VM then, right ?
  void Print(std::ostream &_os) const {
    _os << Internal[0]; // VM is at least garantee to be one
    for(int i=1; i<VMToLength<TVM>::Length; ++i)
      _os << "," << Internal[i];
    }

  const typename VRToType<TVR>::Type *GetValues() const {
    return Internal;
  }
  const typename VRToType<TVR>::Type &GetValue(unsigned int idx = 0) const {
    assert( idx < VMToLength<TVM>::Length );
    return Internal[idx];
  }
  typename VRToType<TVR>::Type &GetValue(unsigned int idx = 0) {
    assert( idx < VMToLength<TVM>::Length );
    return Internal[idx];
  }
  typename VRToType<TVR>::Type operator[] (unsigned int idx) const {
    return GetValue(idx);
  }
  void SetValue(typename VRToType<TVR>::Type v, unsigned int idx = 0) {
    assert( idx < VMToLength<TVM>::Length );
    Internal[idx] = v;
  }

  void SetFromDataElement(DataElement const &de) {
    const ByteValue *bv = de.GetByteValue();
    if( !bv ) return;
#ifdef GDCM_WORDS_BIGENDIAN
    if( de.GetVR() == VR::UN /*|| de.GetVR() == VR::INVALID*/ )
#else
    if( de.GetVR() == VR::UN || de.GetVR() == VR::INVALID )
#endif
      {
      Set(de.GetValue());
      }
    else
      {
      SetNoSwap(de.GetValue());
      }
  }

  DataElement GetAsDataElement() const {
    DataElement ret;
    std::ostringstream os;
    EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),os);
    ret.SetVR( (VR::VRType)TVR );
    assert( ret.GetVR() != VR::SQ );
    if( (VR::VRType)VRToEncoding<TVR>::Mode == VR::VRASCII )
      {
      if( GetVR() != VR::UI )
        {
        if( os.str().size() % 2 )
          {
          os << " ";
          }
        }
      }
    VL::Type osStrSize = (VL::Type)os.str().size();
    ret.SetByteValue( os.str().c_str(), osStrSize );

    return ret;
  }

  void Read(std::istream &_is) {
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
      GetLength(),_is);
    }
  void Write(std::ostream &_os) const {
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),_os);
    }

  // FIXME: remove this function
  // this is only used in gdcm::SplitMosaicFilter / to pass value of a CSAElement
  void Set(Value const &v) {
    const ByteValue *bv = dynamic_cast<const ByteValue*>(&v);
    if( bv ) {
      //memcpy(Internal, bv->GetPointer(), bv->GetLength());
      std::stringstream ss;
      std::string s = std::string( bv->GetPointer(), bv->GetLength() );
      ss.str( s );
      EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
        GetLength(),ss);
    }
  }
protected:
  void SetNoSwap(Value const &v) {
    const ByteValue *bv = dynamic_cast<const ByteValue*>(&v);
    assert( bv ); // That would be bad...
    //memcpy(Internal, bv->GetPointer(), bv->GetLength());
    std::stringstream ss;
    std::string s = std::string( bv->GetPointer(), bv->GetLength() );
    ss.str( s );
    EncodingImplementation<VRToEncoding<TVR>::Mode>::ReadNoSwap(Internal,
      GetLength(),ss);
  }
};

struct ignore_char {
  ignore_char(char c): m_char(c) {}
  char m_char;
};
ignore_char const backslash('\\');

  inline std::istream& operator>> (std::istream& in, ignore_char const& ic) {
    if (!in.eof())
      in.clear(in.rdstate() & ~std::ios_base::failbit);
    if (in.get() != ic.m_char)
      in.setstate(std::ios_base::failbit);
    return in;
  }


// Implementation to perform formatted read and write
template<> class EncodingImplementation<VR::VRASCII> {
public:
  template<typename T> // FIXME this should be VRToType<TVR>::Type
  static inline void ReadComputeLength(T* data, unsigned int &length,
                          std::istream &_is) {
    assert( data );
    //assert( length ); // != 0
    length = 0;
    assert( _is );
#if 0
    char sep;
    while( _is >> data[length++] )
      {
      // Get the separator in between the values
      assert( _is );
      _is.get(sep);
      assert( sep == '\\' || sep == ' ' ); // FIXME: Bad use of assert
      if( sep == ' ' ) length--; // FIXME
      }
#else
  while( _is >> std::ws >> data[length++] >> std::ws >> backslash )
    {
    }
#endif
    }

  template<typename T> // FIXME this should be VRToType<TVR>::Type
  static inline void Read(T* data, unsigned long length,
                          std::istream &_is) {
    assert( data );
    assert( length ); // != 0
    assert( _is );
    // FIXME BUG: what if >> operation fails ?
    // gdcmData/MR00010001.dcm / SpacingBetweenSlices
    _is >> std::ws >> data[0];
    char sep;
    //std::cout << "GetLength: " << af->GetLength() << std::endl;
    for(unsigned long i=1; i<length;++i) {
      assert( _is );
      // Get the separator in between the values
      _is >> std::ws >> sep; //_is.get(sep);
      assert( sep == '\\' ); // FIXME: Bad use of assert
      _is >> std::ws >> data[i];
      }
    }

  template<typename T>
  static inline void ReadNoSwap(T* data, unsigned long length,
                          std::istream &_is) {
    Read(data,length,_is);
}
  template<typename T>
  static inline void Write(const T* data, unsigned long length,
                           std::ostream &_os)  {
    assert( data );
    assert( length );
    assert( _os );
    _os << data[0];
    for(unsigned long i=1; i<length; ++i) {
      assert( _os );
      _os << "\\" << data[i];
      }
    }
};

template < typename Float >
std::string to_string ( Float data ) {
  std::stringstream in;
  // in.imbue(std::locale::classic()); // This is not required AFAIK
  int const digits =
    static_cast< int >(
    - std::log( std::numeric_limits<Float>::epsilon() )
    / std::log( 10.0 ) );
  if ( in << std::dec << std::setprecision(/*2+*/digits) << data ) {
    return ( in.str() );
  } else {
    throw "Impossible Conversion"; // should not happen ...
  }
}

/* Writing VR::DS is not that easy after all */
// http://groups.google.com/group/comp.lang.c++/browse_thread/thread/69ccd26f000a0802
template<> inline void EncodingImplementation<VR::VRASCII>::Write(const float * data, unsigned long length, std::ostream &_os)  {
    assert( data );
    assert( length );
    assert( _os );
    _os << to_string(data[0]);
    for(unsigned long i=1; i<length; ++i) {
      assert( _os );
      _os << "\\" << to_string(data[i]);
      }
    }

template<> inline void EncodingImplementation<VR::VRASCII>::Write(const double* data, unsigned long length, std::ostream &_os)  {
    assert( data );
    assert( length );
    assert( _os );
    _os << to_string(data[0]);
    for(unsigned long i=1; i<length; ++i) {
      assert( _os );
      _os << "\\" << to_string(data[i]);
      }
    }


// Implementation to perform binary read and write
// TODO rewrite operation so that either:
// #1. dummy implementation use a pointer to Internal and do ++p (faster)
// #2. Actually do some meta programming to unroll the loop
// (no notion of order in VM ...)
template<> class EncodingImplementation<VR::VRBINARY> {
public:
  template<typename T> // FIXME this should be VRToType<TVR>::Type
    static inline void ReadComputeLength(T* data, unsigned int &length,
      std::istream &_is) {
    const unsigned int type_size = sizeof(T);
    assert( data ); // Can we read from pointer ?
    //assert( length );
    length /= type_size;
    assert( _is ); // Is stream valid ?
    _is.read( reinterpret_cast<char*>(data+0), type_size);
    for(unsigned long i=1; i<length; ++i) {
      assert( _is );
      _is.read( reinterpret_cast<char*>(data+i), type_size );
    }
    }
  template<typename T>
  static inline void ReadNoSwap(T* data, unsigned long length,
    std::istream &_is) {
    const unsigned int type_size = sizeof(T);
    assert( data ); // Can we read from pointer ?
    assert( length );
    assert( _is ); // Is stream valid ?
    _is.read( reinterpret_cast<char*>(data+0), type_size);
    for(unsigned long i=1; i<length; ++i) {
      assert( _is );
      _is.read( reinterpret_cast<char*>(data+i), type_size );
    }
    //ByteSwap<T>::SwapRangeFromSwapCodeIntoSystem(data,
    //  _is.GetSwapCode(), length);
    //SwapperNoOp::SwapArray(data,length);
  }
  template<typename T>
  static inline void Read(T* data, unsigned long length,
    std::istream &_is) {
    const unsigned int type_size = sizeof(T);
    assert( data ); // Can we read from pointer ?
    assert( length );
    assert( _is ); // Is stream valid ?
    _is.read( reinterpret_cast<char*>(data+0), type_size);
    for(unsigned long i=1; i<length; ++i) {
      assert( _is );
      _is.read( reinterpret_cast<char*>(data+i), type_size );
    }
    //ByteSwap<T>::SwapRangeFromSwapCodeIntoSystem(data,
    //  _is.GetSwapCode(), length);
    SwapperNoOp::SwapArray(data,length);
  }
  template<typename T>
  static inline void Write(const T* data, unsigned long length,
    std::ostream &_os) {
    const unsigned int type_size = sizeof(T);
    assert( data ); // Can we write into pointer ?
    assert( length );
    assert( _os ); // Is stream valid ?
    //ByteSwap<T>::SwapRangeFromSwapCodeIntoSystem((T*)data,
    //  _os.GetSwapCode(), length);
    T swappedData = SwapperNoOp::Swap(data[0]);
    _os.write( reinterpret_cast<const char*>(&swappedData), type_size);
    for(unsigned long i=1; i<length;++i) {
      assert( _os );
      swappedData = SwapperNoOp::Swap(data[i]);
      _os.write( reinterpret_cast<const char*>(&swappedData), type_size );
    }
    //ByteSwap<T>::SwapRangeFromSwapCodeIntoSystem((T*)data,
    //  _os.GetSwapCode(), length);
  }
};

// For particular case for ASCII string
// WARNING: This template explicitely instanciates a particular
// EncodingImplementation THEREFORE it is required to be declared after the
// EncodingImplementation is needs (doh!)
#if 0
template<int TVM>
class Element<TVM>
{
public:
  Element(const char array[])
    {
    unsigned int i = 0;
    const char sep = '\\';
    std::string sarray = array;
    std::string::size_type pos1 = 0;
    std::string::size_type pos2 = sarray.find(sep, pos1+1);
    while(pos2 != std::string::npos)
      {
      Internal[i++] = sarray.substr(pos1, pos2-pos1);
      pos1 = pos2+1;
      pos2 = sarray.find(sep, pos1+1);
      }
    Internal[i] = sarray.substr(pos1, pos2-pos1);
    // Shouldn't we do the contrary, since we know how many separators
    // (and default behavior is to discard anything after the VM declared
    assert( GetLength()-1 == i );
    }

  unsigned long GetLength() const {
    return VMToLength<TVM>::Length;
  }
  // Implementation of Print is common to all Mode (ASCII/Binary)
  void Print(std::ostream &_os) const {
    _os << Internal[0]; // VM is at least garantee to be one
    for(int i=1; i<VMToLength<TVM>::Length; ++i)
      _os << "," << Internal[i];
    }

  void Read(std::istream &_is) {
    EncodingImplementation<VR::VRASCII>::Read(Internal, GetLength(),_is);
    }
  void Write(std::ostream &_os) const {
    EncodingImplementation<VR::VRASCII>::Write(Internal, GetLength(),_os);
    }
private:
  typename String Internal[VMToLength<TVM>::Length];
};

template< int TVM>
class Element<VR::PN, TVM> : public StringElement<TVM>
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<VR::PN, TVM> ) };
};
#endif

// Implementation for the undefined length (dynamically allocated array)
template<int TVR>
class Element<TVR, VM::VM1_n>
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<TVR, VM::VM1_n> ) };
public:
  // This the way to prevent default initialization
  explicit Element() { Internal=0; Length=0; Save = false; }
  ~Element() {
    if( Save ) {
      delete[] Internal;
    }
    Internal = 0;
  }

  static VR  GetVR()  { return (VR::VRType)TVR; }
  static VM  GetVM()  { return VM::VM1_n; }

  // Length manipulation
  // SetLength should really be protected anyway...all operation
  // should go through SetArray
  unsigned long GetLength() const { return Length; }
  typedef typename VRToType<TVR>::Type Type;

  void SetLength(unsigned long len) {
    const unsigned int size = sizeof(Type);
    if( len ) {
      if( len > Length ) {
        // perform realloc
        assert( (len / size) * size == len );
        Type *internal = new Type[len / size];
        assert( Save == false );
        Save = true; // ????
        if( Internal )
          {
          memcpy(internal, Internal, len);
          delete[] Internal;
          }
        Internal = internal;
        }
      }
    Length = len / size;
  }

  // If save is set to zero user should not delete the pointer
  //void SetArray(const typename VRToType<TVR>::Type *array, int len, bool save = false)
  void SetArray(const Type *array, unsigned long len,
    bool save = false) {
    if( save ) {
      SetLength(len); // realloc
      memcpy(Internal, array, len/*/sizeof(Type)*/);
      assert( Save == false );
      }
    else {
      // TODO rewrite this stupid code:
      assert( Length == 0 );
      assert( Internal == 0 );
      assert( Save == false );
      Length = len / sizeof(Type);
      //assert( (len / sizeof(Type)) * sizeof(Type) == len );
      // MR00010001.dcm is a tough kid: 0019,105a is supposed to be VR::FL, VM::VM3 but
      // length is 14 bytes instead of 12 bytes. Simply consider value is total garbage.
      if( (len / sizeof(Type)) * sizeof(Type) != len ) { Internal = 0; Length = 0; }
      else Internal = const_cast<Type*>(array);
      }
      Save = save;
  }
  void SetValue(typename VRToType<TVR>::Type v, unsigned int idx = 0) {
    assert( idx < Length );
    Internal[idx] = v;
  }
  const typename VRToType<TVR>::Type &GetValue(unsigned int idx = 0) const {
    assert( idx < Length );
    return Internal[idx];
  }
  typename VRToType<TVR>::Type &GetValue(unsigned int idx = 0) {
    //assert( idx < Length );
    return Internal[idx];
  }
  typename VRToType<TVR>::Type operator[] (unsigned int idx) const {
    return GetValue(idx);
  }
  void Set(Value const &v) {
    const ByteValue *bv = dynamic_cast<const ByteValue*>(&v);
    assert( bv ); // That would be bad...
    if( (VR::VRType)(VRToEncoding<TVR>::Mode) == VR::VRBINARY )
      {
      const Type* array = (Type*)bv->GetPointer();
      if( array ) {
        assert( array ); // That would be bad...
        assert( Internal == 0 );
        SetArray(array, bv->GetLength() ); }
      }
    else
      {
      std::stringstream ss;
      std::string s = std::string( bv->GetPointer(), bv->GetLength() );
      ss.str( s );
      EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
        GetLength(),ss);
      }
  }
  void SetFromDataElement(DataElement const &de) {
    const ByteValue *bv = de.GetByteValue();
    if( !bv ) return;
#ifdef GDCM_WORDS_BIGENDIAN
    if( de.GetVR() == VR::UN /*|| de.GetVR() == VR::INVALID*/ )
#else
    if( de.GetVR() == VR::UN || de.GetVR() == VR::INVALID )
#endif
      {
      Set(de.GetValue());
      }
    else
      {
      SetNoSwap(de.GetValue());
      }
  }


  // Need to be placed after definition of EncodingImplementation<VR::VRASCII>
  void WriteASCII(std::ostream &os) const {
    return EncodingImplementation<VR::VRASCII>::Write(Internal, GetLength(), os);
    }

  // Implementation of Print is common to all Mode (ASCII/Binary)
  void Print(std::ostream &_os) const {
    assert( Length );
    assert( Internal );
    _os << Internal[0]; // VM is at least garantee to be one
    const unsigned long length = GetLength() < 25 ? GetLength() : 25;
    for(unsigned long i=1; i<length; ++i)
      _os << "," << Internal[i];
    }
  void Read(std::istream &_is) {
    if( !Internal ) return;
    EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
      GetLength(),_is);
    }
  //void ReadComputeLength(std::istream &_is) {
  //  if( !Internal ) return;
  //  EncodingImplementation<VRToEncoding<TVR>::Mode>::ReadComputeLength(Internal,
  //    Length,_is);
  //  }
  void Write(std::ostream &_os) const {
    EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),_os);
    }

  DataElement GetAsDataElement() const {
    DataElement ret;
    ret.SetVR( (VR::VRType)TVR );
    assert( ret.GetVR() != VR::SQ );
    if( Internal )
      {
      std::ostringstream os;
      EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
        GetLength(),os);
      if( (VR::VRType)VRToEncoding<TVR>::Mode == VR::VRASCII )
        {
        if( GetVR() != VR::UI )
          {
          if( os.str().size() % 2 )
            {
            os << " ";
            }
          }
        }
      VL::Type osStrSize = (VL::Type)os.str().size();
      ret.SetByteValue( os.str().c_str(), osStrSize );
      }
    return ret;
  }

  Element(const Element&_val) {
    if( this != &_val) {
      *this = _val;
      }
    }

  Element &operator=(const Element &_val) {
    Length = 0; // SYITF
    Internal = 0;
    SetArray(_val.Internal, _val.Length, true);
    return *this;
    }
protected:
  void SetNoSwap(Value const &v) {
    const ByteValue *bv = dynamic_cast<const ByteValue*>(&v);
    assert( bv ); // That would be bad...
    if( (VR::VRType)(VRToEncoding<TVR>::Mode) == VR::VRBINARY )
      {
      const Type* array = (Type*)bv->GetPointer();
      if( array ) {
        assert( array ); // That would be bad...
        assert( Internal == 0 );
        SetArray(array, bv->GetLength() ); }
      }
    else
      {
      std::stringstream ss;
      std::string s = std::string( bv->GetPointer(), bv->GetLength() );
      ss.str( s );
      EncodingImplementation<VRToEncoding<TVR>::Mode>::ReadNoSwap(Internal,
        GetLength(),ss);
      }
  }

private:
  typename VRToType<TVR>::Type *Internal;
  unsigned long Length; // unsigned int ??
  bool Save;
};

//template <int TVM = VM::VM1_n>
//class Element<VR::OB, TVM > : public Element<VR::OB, VM::VM1_n> {};

// Partial specialization for derivatives of 1-n : 2-n, 3-n ...
template<int TVR>
class Element<TVR, VM::VM1_2> : public Element<TVR, VM::VM1_n>
{
public:
  typedef Element<TVR, VM::VM1_n> Parent;
  void SetLength(int len) {
    if( len != 1 || len != 2 ) return;
    Parent::SetLength(len);
  }
};
template<int TVR>
class Element<TVR, VM::VM2_n> : public Element<TVR, VM::VM1_n>
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<TVR, VM::VM2_n> ) };
public:
  typedef Element<TVR, VM::VM1_n> Parent;
  void SetLength(int len) {
    if( len <= 1 ) return;
    Parent::SetLength(len);
  }
};
template<int TVR>
class Element<TVR, VM::VM2_2n> : public Element<TVR, VM::VM2_n>
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<TVR, VM::VM2_2n> ) };
public:
  typedef Element<TVR, VM::VM2_n> Parent;
  void SetLength(int len) {
    if( len % 2 ) return;
    Parent::SetLength(len);
  }
};
template<int TVR>
class Element<TVR, VM::VM3_n> : public Element<TVR, VM::VM1_n>
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<TVR, VM::VM3_n> ) };
public:
  typedef Element<TVR, VM::VM1_n> Parent;
  void SetLength(int len) {
    if( len <= 2 ) return;
    Parent::SetLength(len);
  }
};
template<int TVR>
class Element<TVR, VM::VM3_3n> : public Element<TVR, VM::VM3_n>
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<TVR, VM::VM3_3n> ) };
public:
  typedef Element<TVR, VM::VM3_n> Parent;
  void SetLength(int len) {
    if( len % 3 ) return;
    Parent::SetLength(len);
  }
};


//template<int T> struct VRToLength;
//template <> struct VRToLength<VR::AS>
//{ enum { Length  = VM::VM1 }; }
//template<>
//class Element<VR::AS> : public Element<VR::AS, VRToLength<VR::AS>::Length >

// only 0010 1010 AS 1 Patient's Age
template<>
class Element<VR::AS, VM::VM5>
{
  enum { ElementDisableCombinationsCheck = sizeof ( ElementDisableCombinations<VR::AS, VM::VM5> ) };
public:
  char Internal[VMToLength<VM::VM5>::Length * sizeof( VRToType<VR::AS>::Type )];
  void Print(std::ostream &_os) const {
    _os << Internal;
    }
  unsigned long GetLength() const {
    return VMToLength<VM::VM5>::Length;
  }
};


template <>
class Element<VR::OB, VM::VM1> : public Element<VR::OB, VM::VM1_n> {};

// Same for OW:
template <>
class Element<VR::OW, VM::VM1> : public Element<VR::OW, VM::VM1_n> {};


} // namespace gdcm

#endif //GDCMELEMENT_H
