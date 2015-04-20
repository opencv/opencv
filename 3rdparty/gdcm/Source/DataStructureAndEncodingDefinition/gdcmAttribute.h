/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMATTRIBUTE_H
#define GDCMATTRIBUTE_H

#include "gdcmTypes.h"
#include "gdcmVR.h"
#include "gdcmTagToType.h"
#include "gdcmVM.h"
#include "gdcmElement.h"
#include "gdcmDataElement.h"
#include "gdcmDataSet.h"
#include "gdcmStaticAssert.h"

#include <string>
#include <vector>
#include <sstream>

namespace gdcm
{

struct void_;

// Declaration, also serve as forward declaration
template<int T> class VRVLSize;

// Implementation when VL is coded on 16 bits:
template<> class VRVLSize<0> {
public:
  static inline uint16_t Read(std::istream &_is) {
    uint16_t l;
    _is.read((char*)&l, 2);
    return l;
    }

  static inline void Write(std::ostream &os)  { (void)os;
    }
};
// Implementation when VL is coded on 32 bits:
template<> class VRVLSize<1> {
public:
  static inline uint32_t Read(std::istream &_is) {
    char dummy[2];
    _is.read(dummy, 2);

    uint32_t l;
    _is.read((char*)&l, 4);
    return l;
    }

  static inline void Write(std::ostream &os)  { (void)os;
    }
};

/**
 * \brief Attribute class
 * This class use template metaprograming tricks to let the user know when the template
 * instanciation does not match the public dictionary.
 *
 * Typical example that compile is:
 * Attribute<0x0008,0x9007> a = {"ORIGINAL","PRIMARY","T1","NONE"};
 *
 * Examples that will NOT compile are:
 *
 * Attribute<0x0018,0x1182, VR::IS, VM::VM1> fd1 = {}; // not enough parameters
 * Attribute<0x0018,0x1182, VR::IS, VM::VM2> fd2 = {0,1,2}; // too many initializers
 * Attribute<0x0018,0x1182, VR::IS, VM::VM3> fd3 = {0,1,2}; // VM3 is not valid
 * Attribute<0x0018,0x1182, VR::UL, VM::VM2> fd3 = {0,1}; // UL is not valid VR
 */
template<uint16_t Group, uint16_t Element,
   int TVR = TagToType<Group, Element>::VRType, // can the user override this value ?
   int TVM = TagToType<Group, Element>::VMType // can the user override this value ?
   /*typename SQAttribute = void_*/ > // if only I had variadic template...
class Attribute
{
public:
  typedef typename VRToType<TVR>::Type ArrayType;
  enum { VMType = VMToLength<TVM>::Length };
  ArrayType Internal[VMToLength<TVM>::Length];

  // Make sure that user specified VR/VM are compatible with the public dictionary:
  GDCM_STATIC_ASSERT( ((VR::VRType)TVR & (VR::VRType)(TagToType<Group, Element>::VRType)) );
  GDCM_STATIC_ASSERT( ((VM::VMType)TVM & (VM::VMType)(TagToType<Group, Element>::VMType)) );
  GDCM_STATIC_ASSERT( ((((VR::VRType)TVR & VR::VR_VM1) && ((VM::VMType)TVM == VM::VM1) )
                    || !((VR::VRType)TVR & VR::VR_VM1) ) );

  static Tag GetTag() { return Tag(Group,Element); }
  static VR  GetVR()  { return (VR::VRType)TVR; }
  static VM  GetVM()  { return (VM::VMType)TVM; }

  // The following two methods do make sense only in case of public element,
  // when the template is intanciated with private element the VR/VM are simply
  // defaulted to allow everything (see gdcmTagToType.h default template for TagToType)
  static VR  GetDictVR() { return (VR::VRType)(TagToType<Group, Element>::VRType); }
  static VM  GetDictVM() { return (VM::VMType)(TagToType<Group, Element>::VMType); }

  // Some extra dummy checks:
  // Data Elements with a VR of SQ, OF, OW, OB or UN shall always have a Value Multiplicity of one.

  unsigned int GetNumberOfValues() const {
    return VMToLength<TVM>::Length;
  }
  // Implementation of Print is common to all Mode (ASCII/Binary)
  // TODO: Can we print a \ when in ASCII...well I don't think so
  // it would mean we used a bad VM then, right ?
  void Print(std::ostream &os) const {
    os << GetTag() << " ";
    os << TagToType<Group,Element>::GetVRString()  << " ";
    os << TagToType<Group,Element>::GetVMString()  << " ";
    os << Internal[0]; // VM is at least garantee to be one
    for(unsigned int i=1; i<GetNumberOfValues(); ++i)
      os << "," << Internal[i];
    }

  // copy:
  //ArrayType GetValue(unsigned int idx = 0) {
  //  assert( idx < GetNumberOfValues() );
  //  return Internal[idx];
  //}
  //ArrayType operator[] (unsigned int idx) {
  //  return GetValue(idx);
  //}
  // FIXME: is this always a good idea ?
  // I do not think so, I prefer operator
  //operator ArrayType () const { return Internal[0]; }

  bool operator==(const Attribute &att) const
    {
    return std::equal(Internal, Internal+GetNumberOfValues(),
      att.GetValues());
    }
  bool operator!=(const Attribute &att) const
    {
    return !std::equal(Internal, Internal+GetNumberOfValues(),
      att.GetValues());
    }
  bool operator<(const Attribute &att) const
    {
    return std::lexicographical_compare(Internal, Internal+GetNumberOfValues(),
      att.GetValues(), att.GetValues() + att.GetNumberOfValues() );
    }

  ArrayType &GetValue(unsigned int idx = 0) {
    assert( idx < GetNumberOfValues() );
    return Internal[idx];
  }
  ArrayType & operator[] (unsigned int idx) {
    return GetValue(idx);
  }
  // const reference
  ArrayType const &GetValue(unsigned int idx = 0) const {
    assert( idx < GetNumberOfValues() );
    return Internal[idx];
  }
  ArrayType const & operator[] (unsigned int idx) const {
    return GetValue(idx);
  }
  void SetValue(ArrayType v, unsigned int idx = 0) {
    assert( idx < GetNumberOfValues() );
    Internal[idx] = v;
  }
  void SetValues(const ArrayType* array, unsigned int numel = VMType ) {
    assert( array && numel && numel == GetNumberOfValues() );
    // std::copy is smarted than a memcpy, and will call memcpy when POD type
    std::copy(array, array+numel, Internal);
  }
  const ArrayType* GetValues() const {
    return Internal;
  }

  // API to talk to the run-time layer: gdcm::DataElement
  DataElement GetAsDataElement() const {
    DataElement ret( GetTag() );
    std::ostringstream os;
    // os.imbue(std::locale::classic()); // This is not required AFAIK
    EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetNumberOfValues(),os);
    ret.SetVR( GetVR() );
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

  void SetFromDataElement(DataElement const &de) {
    // This is kind of hackish but since I do not generate other element than the first one: 0x6000 I should be ok:
    assert( GetTag() == de.GetTag() || GetTag().GetGroup() == 0x6000 || GetTag().GetGroup() == 0x5000 );
    assert( GetVR() != VR::INVALID );
    assert( GetVR().Compatible( de.GetVR() ) || de.GetVR() == VR::INVALID ); // In case of VR::INVALID cannot use the & operator
    if( de.IsEmpty() ) return;
    const ByteValue *bv = de.GetByteValue();
#ifdef GDCM_WORDS_BIGENDIAN
    if( de.GetVR() == VR::UN /*|| de.GetVR() == VR::INVALID*/ )
#else
    if( de.GetVR() == VR::UN || de.GetVR() == VR::INVALID )
#endif
      {
      SetByteValue(bv);
      }
    else
      {
      SetByteValueNoSwap(bv);
      }
  }
  void Set(DataSet const &ds) {
    SetFromDataElement( ds.GetDataElement( GetTag() ) );
  }
  void SetFromDataSet(DataSet const &ds) {
    if( ds.FindDataElement( GetTag() ) &&
      !ds.GetDataElement( GetTag() ).IsEmpty() )
      {
      SetFromDataElement( ds.GetDataElement( GetTag() ) );
      }
  }
protected:
  void SetByteValueNoSwap(const ByteValue *bv) {
    if( !bv ) return; // That would be bad...
    assert( bv->GetPointer() && bv->GetLength() ); // [123]C element can be empty
    //if( VRToEncoding<TVR>::Mode == VR::VRBINARY )
    //  {
    //  // always do a copy !
    //  SetValues(bv->GetPointer(), bv->GetLength());
    //  }
    //else
      {
      std::stringstream ss;
      std::string s = std::string( bv->GetPointer(), bv->GetLength() );
      ss.str( s );
      EncodingImplementation<VRToEncoding<TVR>::Mode>::ReadNoSwap(Internal,
        GetNumberOfValues(),ss);
      }
  }
  void SetByteValue(const ByteValue *bv) {
    if( !bv ) return; // That would be bad...
    assert( bv->GetPointer() && bv->GetLength() ); // [123]C element can be empty
    //if( VRToEncoding<TVR>::Mode == VR::VRBINARY )
    //  {
    //  // always do a copy !
    //  SetValues(bv->GetPointer(), bv->GetLength());
    //  }
    //else
      {
      std::stringstream ss;
      std::string s = std::string( bv->GetPointer(), bv->GetLength() );
      ss.str( s );
      EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
        GetNumberOfValues(),ss);
      }
  }
#if 0 // TODO  FIXME the implicit way:
  // explicit:
  void Read(std::istream &_is) {
    const uint16_t cref[] = { Group, Element };
    uint16_t c[2];
    _is.read((char*)&c, sizeof(c));
    assert( c[0] == cref[0] && c[1] == cref[1] );
    char vr[2];
    _is.read(vr, 2); // Check consistency ?
    const uint32_t lref = GetLength() * sizeof( typename VRToType<TVR>::Type );
    uint32_t l = VRVLSize< (TVR & VR::VL32) >::Read(_is);
    l /= sizeof( typename VRToType<TVR>::Type );
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
      l,_is);
  }
  void Write(std::ostream &_os) const {
    uint16_t c[] = { Group, Element };
    _os.write((char*)&c, 4);
    uint32_t l = GetLength() * sizeof( typename VRToType<TVR>::Type );
    _os.write((char*)&l, 4);
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),_os);
    }
  void Read(std::istream &_is) {
    uint16_t cref[] = { Group, Element };
    uint16_t c[2];
    _is.read((char*)&c, 4);
    const uint32_t lref = GetLength() * sizeof( typename VRToType<TVR>::Type );
    uint32_t l;
    _is.read((char*)&l, 4);
    l /= sizeof( typename VRToType<TVR>::Type );
     return EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
      l,_is);
    }
  void Write(std::ostream &_os) const {
    uint16_t c[] = { Group, Element };
    _os.write((char*)&c, 4);
    uint32_t l = GetLength() * sizeof( typename VRToType<TVR>::Type );
    _os.write((char*)&l, 4);
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),_os);
    }
#endif

};

template<uint16_t Group, uint16_t Element, int TVR >
class Attribute<Group,Element,TVR,VM::VM1>
{
public:
  typedef typename VRToType<TVR>::Type ArrayType;
  enum { VMType = VMToLength<VM::VM1>::Length };
  //ArrayType Internal[VMToLength<TVM>::Length];
  ArrayType Internal;
  GDCM_STATIC_ASSERT( VMToLength<VM::VM1>::Length == 1 );

  // Make sure that user specified VR/VM are compatible with the public dictionary:
  GDCM_STATIC_ASSERT( ((VR::VRType)TVR & (VR::VRType)(TagToType<Group, Element>::VRType)) );
  GDCM_STATIC_ASSERT( ((VM::VMType)VM::VM1 & (VM::VMType)(TagToType<Group, Element>::VMType)) );
  GDCM_STATIC_ASSERT( ((((VR::VRType)TVR & VR::VR_VM1) && ((VM::VMType)VM::VM1 == VM::VM1) )
                    || !((VR::VRType)TVR & VR::VR_VM1) ) );

  static Tag GetTag() { return Tag(Group,Element); }
  static VR  GetVR()  { return (VR::VRType)TVR; }
  static VM  GetVM()  { return (VM::VMType)VM::VM1; }

  // The following two methods do make sense only in case of public element,
  // when the template is intanciated with private element the VR/VM are simply
  // defaulted to allow everything (see gdcmTagToType.h default template for TagToType)
  static VR  GetDictVR() { return (VR::VRType)(TagToType<Group, Element>::VRType); }
  static VM  GetDictVM() { return (VM::VMType)(TagToType<Group, Element>::VMType); }

  // Some extra dummy checks:
  // Data Elements with a VR of SQ, OF, OW, OB or UN shall always have a Value Multiplicity of one.

  unsigned int GetNumberOfValues() const {
    return VMToLength<VM::VM1>::Length;
  }
  // Implementation of Print is common to all Mode (ASCII/Binary)
  // TODO: Can we print a \ when in ASCII...well I don't think so
  // it would mean we used a bad VM then, right ?
  void Print(std::ostream &os) const {
    os << GetTag() << " ";
    os << TagToType<Group,Element>::GetVRString()  << " ";
    os << TagToType<Group,Element>::GetVMString()  << " ";
    os << Internal; // VM is at least garantee to be one
  }
  // copy:
  //ArrayType GetValue(unsigned int idx = 0) {
  //  assert( idx < GetNumberOfValues() );
  //  return Internal[idx];
  //}
  //ArrayType operator[] (unsigned int idx) {
  //  return GetValue(idx);
  //}
  // FIXME: is this always a good idea ?
  // I do not think so, I prefer operator
  //operator ArrayType () const { return Internal[0]; }

  bool operator==(const Attribute &att) const
    {
    return std::equal(&Internal, &Internal+GetNumberOfValues(),
      att.GetValues());
    }
  bool operator!=(const Attribute &att) const
    {
    return !std::equal(&Internal, &Internal+GetNumberOfValues(),
      att.GetValues());
    }
  bool operator<(const Attribute &att) const
    {
    return std::lexicographical_compare(&Internal, &Internal+GetNumberOfValues(),
      att.GetValues(), att.GetValues() + att.GetNumberOfValues() );
    }

  ArrayType &GetValue() {
//    assert( idx < GetNumberOfValues() );
    return Internal;
  }
//  ArrayType & operator[] (unsigned int idx) {
//    return GetValue(idx);
//  }
  // const reference
  ArrayType const &GetValue() const {
    //assert( idx < GetNumberOfValues() );
    return Internal;
  }
  //ArrayType const & operator[] () const {
  //  return GetValue();
  //}
  void SetValue(ArrayType v) {
//    assert( idx < GetNumberOfValues() );
    Internal = v;
  }
/*  void SetValues(const ArrayType* array, unsigned int numel = VMType ) {
    assert( array && numel && numel == GetNumberOfValues() );
    // std::copy is smarted than a memcpy, and will call memcpy when POD type
    std::copy(array, array+numel, Internal);
  }
*/

  // FIXME Should we remove this function ?
  const ArrayType* GetValues() const {
    return &Internal;
  }

  // API to talk to the run-time layer: gdcm::DataElement
  DataElement GetAsDataElement() const {
    DataElement ret( GetTag() );
    std::ostringstream os;
    // os.imbue(std::locale::classic()); // This is not required AFAIK
    EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(&Internal,
      GetNumberOfValues(),os);
    ret.SetVR( GetVR() );
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

  void SetFromDataElement(DataElement const &de) {
    // This is kind of hackish but since I do not generate other element than the first one: 0x6000 I should be ok:
    assert( GetTag() == de.GetTag() || GetTag().GetGroup() == 0x6000 || GetTag().GetGroup() == 0x5000 );
    assert( GetVR() != VR::INVALID );
    assert( GetVR().Compatible( de.GetVR() ) || de.GetVR() == VR::INVALID ); // In case of VR::INVALID cannot use the & operator
    if( de.IsEmpty() ) return;
    const ByteValue *bv = de.GetByteValue();
#ifdef GDCM_WORDS_BIGENDIAN
    if( de.GetVR() == VR::UN /*|| de.GetVR() == VR::INVALID*/ )
#else
    if( de.GetVR() == VR::UN || de.GetVR() == VR::INVALID )
#endif
      {
      SetByteValue(bv);
      }
    else
      {
      SetByteValueNoSwap(bv);
      }
  }
  void Set(DataSet const &ds) {
    SetFromDataElement( ds.GetDataElement( GetTag() ) );
  }
  void SetFromDataSet(DataSet const &ds) {
    if( ds.FindDataElement( GetTag() ) &&
      !ds.GetDataElement( GetTag() ).IsEmpty() )
      {
      SetFromDataElement( ds.GetDataElement( GetTag() ) );
      }
  }
protected:
  void SetByteValueNoSwap(const ByteValue *bv) {
    if( !bv ) return; // That would be bad...
    assert( bv->GetPointer() && bv->GetLength() ); // [123]C element can be empty
    //if( VRToEncoding<TVR>::Mode == VR::VRBINARY )
    //  {
    //  // always do a copy !
    //  SetValues(bv->GetPointer(), bv->GetLength());
    //  }
    //else
      {
      std::stringstream ss;
      std::string s = std::string( bv->GetPointer(), bv->GetLength() );
      ss.str( s );
      EncodingImplementation<VRToEncoding<TVR>::Mode>::ReadNoSwap(&Internal,
        GetNumberOfValues(),ss);
      }
  }
  void SetByteValue(const ByteValue *bv) {
    if( !bv ) return; // That would be bad...
    assert( bv->GetPointer() && bv->GetLength() ); // [123]C element can be empty
    //if( VRToEncoding<TVR>::Mode == VR::VRBINARY )
    //  {
    //  // always do a copy !
    //  SetValues(bv->GetPointer(), bv->GetLength());
    //  }
    //else
      {
      std::stringstream ss;
      std::string s = std::string( bv->GetPointer(), bv->GetLength() );
      ss.str( s );
      EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(&Internal,
        GetNumberOfValues(),ss);
      }
  }
#if 0 // TODO  FIXME the implicit way:
  // explicit:
  void Read(std::istream &_is) {
    const uint16_t cref[] = { Group, Element };
    uint16_t c[2];
    _is.read((char*)&c, sizeof(c));
    assert( c[0] == cref[0] && c[1] == cref[1] );
    char vr[2];
    _is.read(vr, 2); // Check consistency ?
    const uint32_t lref = GetLength() * sizeof( typename VRToType<TVR>::Type );
    uint32_t l = VRVLSize< (TVR & VR::VL32) >::Read(_is);
    l /= sizeof( typename VRToType<TVR>::Type );
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
      l,_is);
  }
  void Write(std::ostream &_os) const {
    uint16_t c[] = { Group, Element };
    _os.write((char*)&c, 4);
    uint32_t l = GetLength() * sizeof( typename VRToType<TVR>::Type );
    _os.write((char*)&l, 4);
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),_os);
    }
  void Read(std::istream &_is) {
    uint16_t cref[] = { Group, Element };
    uint16_t c[2];
    _is.read((char*)&c, 4);
    const uint32_t lref = GetLength() * sizeof( typename VRToType<TVR>::Type );
    uint32_t l;
    _is.read((char*)&l, 4);
    l /= sizeof( typename VRToType<TVR>::Type );
     return EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
      l,_is);
    }
  void Write(std::ostream &_os) const {
    uint16_t c[] = { Group, Element };
    _os.write((char*)&c, 4);
    uint32_t l = GetLength() * sizeof( typename VRToType<TVR>::Type );
    _os.write((char*)&l, 4);
    return EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),_os);
    }
#endif

};

// No need to repeat default template arg, since primary template
// will be used to generate the default arguments
template<uint16_t Group, uint16_t Element, int TVR >
class Attribute<Group,Element,TVR,VM::VM1_n>
{
public:
  typedef typename VRToType<TVR>::Type ArrayType;

  // Make sure that user specified VR/VM are compatible with the public dictionary:
  GDCM_STATIC_ASSERT( ((VR::VRType)TVR & (VR::VRType)(TagToType<Group, Element>::VRType)) );
  GDCM_STATIC_ASSERT( (VM::VM1_n & (VM::VMType)(TagToType<Group, Element>::VMType)) );
  GDCM_STATIC_ASSERT( ((((VR::VRType)TVR & VR::VR_VM1) && ((VM::VMType)TagToType<Group,Element>::VMType == VM::VM1) )
                    || !((VR::VRType)TVR & VR::VR_VM1) ) );

  static Tag GetTag() { return Tag(Group,Element); }
  static VR  GetVR()  { return (VR::VRType)TVR; }
  static VM  GetVM()  { return VM::VM1_n; }

  static VR  GetDictVR() { return (VR::VRType)(TagToType<Group, Element>::VRType); }
  static VM  GetDictVM() { return GetVM(); }

  // This the way to prevent default initialization
  explicit Attribute() { Internal=0; Length=0; Own = true; }
  ~Attribute() {
    if( Own ) {
      delete[] Internal;
    }
    Internal = 0; // paranoid
  }

  unsigned int GetNumberOfValues() const { return Length; }

  void SetNumberOfValues(unsigned int numel)
    {
    SetValues(NULL, numel, true);
    }

  const ArrayType* GetValues() const {
    return Internal;
  }
  void Print(std::ostream &os) const {
    os << GetTag() << " ";
    os << GetVR()  << " ";
    os << GetVM()  << " ";
    os << Internal[0]; // VM is at least garantee to be one
    for(unsigned int i=1; i<GetNumberOfValues(); ++i)
      os << "," << Internal[i];
    }
  ArrayType &GetValue(unsigned int idx = 0) {
    assert( idx < GetNumberOfValues() );
    return Internal[idx];
  }
  ArrayType &operator[] (unsigned int idx) {
    return GetValue(idx);
  }
  // const reference
  ArrayType const &GetValue(unsigned int idx = 0) const {
    assert( idx < GetNumberOfValues() );
    return Internal[idx];
  }
  ArrayType const & operator[] (unsigned int idx) const {
    return GetValue(idx);
  }
  void SetValue(unsigned int idx, ArrayType v) {
    assert( idx < GetNumberOfValues() );
    Internal[idx] = v;
  }
  void SetValue(ArrayType v) { SetValue(0, v); }

  void SetValues(const ArrayType *array, unsigned int numel, bool own = false)
    {
    if( Internal ) // were we used before ?
      {
      // yes !
      if( Own ) delete[] Internal;
      Internal = 0;
      }
    Own = own;
    Length = numel;
    assert( Internal == 0 );
    if( own ) // make a copy:
      {
      assert( /*array &&*/ numel );
      Internal = new ArrayType[numel];
      if( array && numel )
        std::copy(array, array+numel, Internal);
      }
    else // pass pointer
      {
      Internal = const_cast<ArrayType*>(array);
      }
    // postcondition
    assert( numel == GetNumberOfValues() );
    }

  DataElement GetAsDataElement() const {
    DataElement ret( GetTag() );
    std::ostringstream os;
    if( Internal )
      {
      EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
        GetNumberOfValues(),os);
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
      }
    ret.SetVR( GetVR() );
    assert( ret.GetVR() != VR::SQ );
    VL::Type osStrSize = (VL::Type) os.str().size();
    ret.SetByteValue( os.str().c_str(), osStrSize);
    return ret;
  }
  void SetFromDataElement(DataElement const &de) {
    // This is kind of hackish but since I do not generate other element than the first one: 0x6000 I should be ok:
    assert( GetTag() == de.GetTag() || GetTag().GetGroup() == 0x6000
      || GetTag().GetGroup() == 0x5000 );
    assert( GetVR().Compatible( de.GetVR() ) ); // In case of VR::INVALID cannot use the & operator
    assert( !de.IsEmpty() );
    const ByteValue *bv = de.GetByteValue();
    SetByteValue(bv);
  }
  void Set(DataSet const &ds) {
    SetFromDataElement( ds.GetDataElement( GetTag() ) );
  }
  void SetFromDataSet(DataSet const &ds) {
    if( ds.FindDataElement( GetTag() ) &&
      !ds.GetDataElement( GetTag() ).IsEmpty() )
      {
      SetFromDataElement( ds.GetDataElement( GetTag() ) );
      }
  }
protected:
  void SetByteValue(const ByteValue *bv) {
    assert( bv ); // FIXME
    std::stringstream ss;
    std::string s = std::string( bv->GetPointer(), bv->GetLength() );
    Length = bv->GetLength(); // HACK FIXME
    ss.str( s );
    ArrayType *internal;
    ArrayType buffer[256];
    if( bv->GetLength() < 256 )
      {
      internal = buffer;
      }
    else
      {
      internal = new ArrayType[(VL::Type)bv->GetLength()]; // over allocation
      }
    EncodingImplementation<VRToEncoding<TVR>::Mode>::ReadComputeLength(internal, Length, ss);
    SetValues( internal, Length, true );
    if( !(bv->GetLength() < 256) )
      {
      delete[] internal;
      }
    //EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
    //  GetNumberOfValues(),ss);
  }

private:
  ArrayType *Internal;
  unsigned int Length;
  bool Own : 1;
};

template<uint16_t Group, uint16_t Element, int TVR>
class Attribute<Group,Element,TVR,VM::VM1_3> : public Attribute<Group,Element,TVR,VM::VM1_n>
{
public:
  VM  GetVM() const { return VM::VM1_3; }
};

template<uint16_t Group, uint16_t Element, int TVR>
class Attribute<Group,Element,TVR,VM::VM1_8> : public Attribute<Group,Element,TVR,VM::VM1_n>
{
public:
  VM  GetVM() const { return VM::VM1_8; }
};

template<uint16_t Group, uint16_t Element, int TVR>
class Attribute<Group,Element,TVR,VM::VM2_n> : public Attribute<Group,Element,TVR,VM::VM1_n>
{
public:
  VM  GetVM() const { return VM::VM2_n; }
};

template<uint16_t Group, uint16_t Element, int TVR>
class Attribute<Group,Element,TVR,VM::VM2_2n> : public Attribute<Group,Element,TVR,VM::VM2_n>
{
public:
  static VM  GetVM() { return VM::VM2_2n; }
};

template<uint16_t Group, uint16_t Element, int TVR>
class Attribute<Group,Element,TVR,VM::VM3_n> : public Attribute<Group,Element,TVR,VM::VM1_n>
{
public:
  static VM  GetVM() { return VM::VM3_n; }
};

template<uint16_t Group, uint16_t Element, int TVR>
class Attribute<Group,Element,TVR,VM::VM3_3n> : public Attribute<Group,Element,TVR,VM::VM3_n>
{
public:
  static VM  GetVM() { return VM::VM3_3n; }
};


// For particular case for ASCII string
// WARNING: This template explicitely instanciates a particular
// EncodingImplementation THEREFORE it is required to be declared after the
// EncodingImplementation is needs (doh!)
#if 0
template<int TVM>
class Attribute<TVM>
{
public:
  Attribute(const char array[])
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
class Attribute<VR::PN, TVM> : public StringAttribute<TVM>
{
};
#endif

#if 0

// Implementation for the undefined length (dynamically allocated array)
template<int TVR>
class Attribute<TVR, VM::VM1_n>
{
public:
  // This the way to prevent default initialization
  explicit Attribute() { Internal=0; Length=0; }
  ~Attribute() {
    delete[] Internal;
    Internal = 0;
  }

  // Length manipulation
  // SetLength should really be protected anyway...all operation
  // should go through SetArray
  unsigned long GetLength() const { return Length; }
  typedef typename VRToType<TVR>::Type ArrayType;
  void SetLength(unsigned long len) {
    const unsigned int size = sizeof(ArrayType);
    if( len ) {
      if( len > Length ) {
        // perform realloc
        assert( (len / size) * size == len );
        ArrayType *internal = new ArrayType[len / size];
        memcpy(internal, Internal, Length * size);
        delete[] Internal;
        Internal = internal;
        }
      }
    Length = len / size;
  }

  // If save is set to zero user should not delete the pointer
  //void SetArray(const typename VRToType<TVR>::Type *array, int len, bool save = false)
  void SetArray(const ArrayType *array, unsigned long len,
    bool save = false) {
    if( save ) {
      SetLength(len); // realloc
      memcpy(Internal, array, len/*/sizeof(ArrayType)*/);
      }
    else {
      // TODO rewrite this stupid code:
      Length = len;
      //Internal = array;
      assert(0);
      }
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
    EncodingImplementation<VRToEncoding<TVR>::Mode>::Read(Internal,
      GetLength(),_is);
    }
  void Write(std::ostream &_os) const {
    EncodingImplementation<VRToEncoding<TVR>::Mode>::Write(Internal,
      GetLength(),_os);
    }

  Attribute(const Attribute&_val) {
    if( this != &_val) {
      *this = _val;
      }
    }

  Attribute &operator=(const Attribute &_val) {
    Length = 0; // SYITF
    Internal = 0;
    SetArray(_val.Internal, _val.Length, true);
    return *this;
    }

private:
  typename VRToType<TVR>::Type *Internal;
  unsigned long Length; // unsigned int ??
};

//template <int TVM = VM::VM1_n>
//class Attribute<VR::OB, TVM > : public Attribute<VR::OB, VM::VM1_n> {};

// Partial specialization for derivatives of 1-n : 2-n, 3-n ...
template<int TVR>
class Attribute<TVR, VM::VM2_n> : public Attribute<TVR, VM::VM1_n>
{
public:
  typedef Attribute<TVR, VM::VM1_n> Parent;
  void SetLength(int len) {
    if( len <= 1 ) return;
    Parent::SetLength(len);
  }
};
template<int TVR>
class Attribute<TVR, VM::VM2_2n> : public Attribute<TVR, VM::VM2_n>
{
public:
  typedef Attribute<TVR, VM::VM2_n> Parent;
  void SetLength(int len) {
    if( len % 2 ) return;
    Parent::SetLength(len);
  }
};
template<int TVR>
class Attribute<TVR, VM::VM3_n> : public Attribute<TVR, VM::VM1_n>
{
public:
  typedef Attribute<TVR, VM::VM1_n> Parent;
  void SetLength(int len) {
    if( len <= 2 ) return;
    Parent::SetLength(len);
  }
};
template<int TVR>
class Attribute<TVR, VM::VM3_3n> : public Attribute<TVR, VM::VM3_n>
{
public:
  typedef Attribute<TVR, VM::VM3_n> Parent;
  void SetLength(int len) {
    if( len % 3 ) return;
    Parent::SetLength(len);
  }
};


//template<int T> struct VRToLength;
//template <> struct VRToLength<VR::AS>
//{ enum { Length  = VM::VM1 }; }
//template<>
//class Attribute<VR::AS> : public Attribute<VR::AS, VRToLength<VR::AS>::Length >

// only 0010 1010 AS 1 Patient's Age
template<>
class Attribute<VR::AS, VM::VM5>
{
public:
  char Internal[VMToLength<VM::VM5>::Length];
  void Print(std::ostream &_os) const {
    _os << Internal;
    }
};

template <>
class Attribute<VR::OB, VM::VM1> : public Attribute<VR::OB, VM::VM1_n> {};
// Make it impossible to compile any other cases:
template <int TVM> class Attribute<VR::OB, TVM>;

// Same for OW:
template <>
class Attribute<VR::OW, VM::VM1> : public Attribute<VR::OW, VM::VM1_n> {};
// Make it impossible to compile any other cases:
template <int TVM> class Attribute<VR::OW, TVM>;
#endif

#if 0
template<>
class Attribute<0x7fe0,0x0010, VR::OW, VM::VM1>
{
public:
  char *Internal;
  unsigned long Length; // unsigned int ??

  void Print(std::ostream &_os) const {
    _os << Internal[0];
    }
  void SetBytes(char *bytes, unsigned long length) {
    Internal = bytes;
    Length = length;
  }
  void Read(std::istream &_is) {
     uint16_t c[2];
    _is.read((char*)&c, 4);
    uint32_t l;
    _is.read((char*)&l, 4);
    Length = l;
    _is.read( Internal, Length );
    }
  void Write(std::ostream &_os) const {
     uint16_t c[] = {0x7fe0, 0x0010};
    _os.write((char*)&c, 4);
    _os.write((char*)&Length, 4);
    _os.write( Internal, Length );
    }
};
#endif

/*
// Removing Attribute for SQ for now...
template<uint16_t Group, uint16_t Element, typename SQA>
class Attribute<Group,Element, VR::SQ, VM::VM1, SQA>
{
public:
  SQA sqa;
  void Print(std::ostream &_os) const {
    _os << Tag(Group,Element);
    sqa.Print(_os << std::endl << '\t');
    }
 void Write(std::ostream &_os) const {
    uint16_t c[] = {Group, Element};
    _os.write((char*)&c, 4);
    uint32_t undef = 0xffffffff;
    _os.write((char*)&undef, 4);
    uint16_t item_beg[] = {0xfffe,0xe000};
    _os.write((char*)&item_beg, 4);
    _os.write((char*)&undef, 4);
    sqa.Write(_os);
    uint16_t item_end[] = {0xfffe,0xe00d};
    _os.write((char*)&item_end, 4);
    uint32_t zero = 0x0;
    _os.write((char*)&zero, 4);
    uint16_t seq_end[] = {0xfffe, 0xe0dd};
    _os.write((char*)&seq_end, 4);
    _os.write((char*)&zero, 4);
    }
};
*/

/**
 * \example PatchFile.cxx
 * This is a C++ example on how to use gdcm::Attribute
 */

} // namespace gdcm

#endif //GDCMATTRIBUTE_H
