/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMOBJECT_H
#define GDCMOBJECT_H

#include "gdcmTypes.h"

#include <assert.h>
#include <iostream> // grrrr

//namespace std { class ostream; }
namespace gdcm
{

template<class ObjectType> class SmartPointer;

/**
 * \brief Object
 *
 * \note main superclass for object that want to use SmartPointer
 * invasive ref counting system
 *
 * \see SmartPointer
 */
class GDCM_EXPORT Object
{
  template <class ObjectType> friend class SmartPointer;
  friend std::ostream& operator<<(std::ostream &os, const Object &obj);

public:
  Object():ReferenceCount(0) {}

  // Implementation note:
  // If I move ~Object in the protected section I can prevent people
  // from writing:
  // SmartPointer<Object> p = new Object;
  // delete p; // due to SmartPointer::operator ObjectType * () const
  // but on the other hand one could not define an Object on the stack
  // Object obj;
  // Furthermore it would not prevent anyone from doing:
  // class MyObject : public Object {};
  // SmartPointer<MyObject> o = new MyObject;
  // delete o; // grrrrrr
  virtual ~Object() {
    // If your debugger reach here it means you are doing something silly
    // like using SmartPointer on object allocated on the stack (vs heap)
    assert(ReferenceCount == 0);
    }

  // http://www.parashift.com/c++-faq-lite/freestore-mgmt.html#faq-16.24
  // Set the ref count to 0
  // Do NOT copy the reference count !
  /// Special requirement for copy/cstor, assignment operator
  Object(const Object&):ReferenceCount(0){}
  void operator=(const Object&){}

  //static Object* New() { return new Object; }

protected:
  // For the purpose of the invasive SmartPointer implementation
  void Register() {
    ReferenceCount++;
    assert( ReferenceCount > 0 );
  }
  void UnRegister() {
    assert( ReferenceCount > 0 );
    ReferenceCount--;
    if(!ReferenceCount)
      {
      delete this;
      }
    }

public:
  // For dealing with printing of object and polymorphism
  virtual void Print(std::ostream &) const {}

private:
  long ReferenceCount;
};

//----------------------------------------------------------------------------
// function do not carry vtable. Thus define in the base class the operator
// and use the member function ->Print() to call the appropriate function
// NOTE: All subclass of Object needs to implement the Print function
inline std::ostream& operator<<(std::ostream &os, const Object &obj)
{
  obj.Print(os);
  return os;
}

} // end namespace gdcm

#endif //GDCMOBJECT_H
