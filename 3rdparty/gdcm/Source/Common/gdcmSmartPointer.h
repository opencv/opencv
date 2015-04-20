/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMSMARTPOINTER_H
#define GDCMSMARTPOINTER_H

#include "gdcmObject.h"

namespace gdcm
{
/**
 * \brief Class for Smart Pointer
 * \details
 * Will only work for subclass of gdcm::Object
 * See tr1/shared_ptr for a more general approach (not invasive)
 * #include <tr1/memory>
 * {
 *   shared_ptr<Bla> b(new Bla);
 * }
 * \note
 * Class partly based on post by Bill Hubauer:
 * http://groups.google.com/group/comp.lang.c++/msg/173ddc38a827a930
 * \see
 * http://www.davethehat.com/articles/smartp.htm
 *
 * and itk::SmartPointer
 */
template<class ObjectType>
class SmartPointer
{
public:
  SmartPointer():Pointer(0) {}
  SmartPointer(const SmartPointer<ObjectType>& p):Pointer(p.Pointer)
    { Register(); }
  SmartPointer(ObjectType* p):Pointer(p)
    { Register(); }
  SmartPointer(ObjectType const & p)
    {
    Pointer = const_cast<ObjectType*>(&p);
    Register();
    }
  ~SmartPointer() {
    UnRegister();
    Pointer = 0;
  }

  /// Overload operator ->
  ObjectType *operator -> () const
    { return Pointer; }

  ObjectType& operator * () const
    { return *Pointer; }

  /// Return pointer to object.
  operator ObjectType * () const
    { return Pointer; }

  /// Overload operator assignment.
  SmartPointer &operator = (SmartPointer const &r)
    { return operator = (r.Pointer); }

  /// Overload operator assignment.
  SmartPointer &operator = (ObjectType *r)
    {
    // http://www.parashift.com/c++-faq-lite/freestore-mgmt.html#faq-16.22
    // DO NOT CHANGE THE ORDER OF THESE STATEMENTS!
    // (This order properly handles self-assignment)
    // (This order also properly handles recursion, e.g., if a ObjectType contains SmartPointer<ObjectType>s)
    if( Pointer != r )
      {
      ObjectType* old = Pointer;
      Pointer = r;
      Register();
      if ( old ) { old->UnRegister(); }
      }
    return *this;
    }

  SmartPointer &operator = (ObjectType const &r)
    {
    ObjectType* tmp = const_cast<ObjectType*>(&r);
    return operator = (tmp);
    }

  /// Explicit function to retrieve the pointer
  ObjectType *GetPointer() const
    { return Pointer; }

private:
  void Register()
    {
    if(Pointer) Pointer->Register();
    }

  void UnRegister()
    {
    if(Pointer) Pointer->UnRegister();
    }

  ObjectType* Pointer;
};

} // end namespace gdcm

#endif //GDCMSMARTPOINTER_H
