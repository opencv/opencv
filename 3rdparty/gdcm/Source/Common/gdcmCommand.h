/*=========================================================================

  Program: GDCM (Grassroots DICOM). A DICOM library

  Copyright (c) 2006-2011 Mathieu Malaterre
  All rights reserved.
  See Copyright.txt or http://gdcm.sourceforge.net/Copyright.html for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef GDCMCOMMAND_H
#define GDCMCOMMAND_H

#include "gdcmSubject.h"

namespace gdcm
{
class Event;

/**
 * \brief Command superclass for callback/observer methods
 * \see Subject
 */
class GDCM_EXPORT Command : public Subject
{
public :
  /// Abstract method that defines the action to be taken by the command.
  virtual void Execute(Subject *caller, const Event & event ) = 0;

  /** Abstract method that defines the action to be taken by the command.
   * This variant is expected to be used when requests comes from a
   * const Object
   */
  virtual void Execute(const Subject *caller, const Event & event ) = 0;

protected:
  Command();
  ~Command();

private:
  Command(const Command&);  // Not implemented.
  void operator=(const Command&);  // Not implemented.
};

/** \class MemberCommand
 *  \brief Command subclass that calls a pointer to a member function
 *
 *  MemberCommand calls a pointer to a member function with the same
 *  arguments as Execute on Command.
 *
 */
template <class T>
class MemberCommand : public Command
{
public:
  /** pointer to a member function that takes a Subject* and the event */
  typedef  void (T::*TMemberFunctionPointer)(Subject*, const Event &);
  typedef  void (T::*TConstMemberFunctionPointer)(const Subject*,
                                                  const Event &);

  /** Standard class typedefs. */
  typedef MemberCommand       Self;
  //typedef SmartPointer<Self>  Pointer;

  /** Method for creation through the object factory. */
  static SmartPointer<MemberCommand> New()
    {
    return new MemberCommand;
    }

  /** Run-time type information (and related methods). */
  //gdcmTypeMacro(MemberCommand,Command);

  /**  Set the callback function along with the object that it will
   *  be invoked on. */
  void SetCallbackFunction(T* object,
                           TMemberFunctionPointer memberFunction)
    {
    m_This = object;
    m_MemberFunction = memberFunction;
    }
  void SetCallbackFunction(T* object,
                           TConstMemberFunctionPointer memberFunction)
    {
    m_This = object;
    m_ConstMemberFunction = memberFunction;
    }

  /**  Invoke the member function. */
  virtual void Execute(Subject *caller, const Event & event )
    {
    if( m_MemberFunction )
      {
      ((*m_This).*(m_MemberFunction))(caller, event);
      }
    }

  /**  Invoke the member function with a const object. */
  virtual void Execute( const Subject *caller, const Event & event )
    {
    if( m_ConstMemberFunction )
      {
      ((*m_This).*(m_ConstMemberFunction))(caller, event);
      }
    }

protected:

  T* m_This;
  TMemberFunctionPointer m_MemberFunction;
  TConstMemberFunctionPointer m_ConstMemberFunction;
  MemberCommand():m_MemberFunction(0),m_ConstMemberFunction(0) {}
  virtual ~MemberCommand(){}

private:
  MemberCommand(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

/** \class SimpleMemberCommand
 *  \brief Command subclass that calls a pointer to a member function
 *
 *  SimpleMemberCommand calls a pointer to a member function with no
 *  arguments.
 */
template <typename T>
class SimpleMemberCommand : public Command
{
public:
  /** A method callback. */
  typedef  void (T::*TMemberFunctionPointer)();

  /** Standard class typedefs. */
  typedef SimpleMemberCommand   Self;
  //typedef SmartPointer<Self>    Pointer;

  /** Run-time type information (and related methods). */
  //gdcmTypeMacro(SimpleMemberCommand,Command);

  /** Method for creation through the object factory. */
  static SmartPointer<SimpleMemberCommand> New()
    {
    return new SimpleMemberCommand;
    }

  /** Specify the callback function. */
  void SetCallbackFunction(T* object,
                           TMemberFunctionPointer memberFunction)
    {
    m_This = object;
    m_MemberFunction = memberFunction;
    }

  /** Invoke the callback function. */
  virtual void Execute(Subject *,const Event & )
    {
    if( m_MemberFunction )
      {
      ((*m_This).*(m_MemberFunction))();
      }
    }
  virtual void Execute(const Subject *,const Event & )
    {
    if( m_MemberFunction )
      {
      ((*m_This).*(m_MemberFunction))();
      }
    }

protected:
  T* m_This;
  TMemberFunctionPointer m_MemberFunction;
  SimpleMemberCommand():m_MemberFunction(0) {}
  virtual ~SimpleMemberCommand() {}

private:
  SimpleMemberCommand(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace gdcm
//-----------------------------------------------------------------------------
#endif //GDCMCOMMAND_H
