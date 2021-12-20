//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

#ifndef INCLUDED_IEXBASEEXC_H
#define INCLUDED_IEXBASEEXC_H

#include "IexNamespace.h"
#include "IexExport.h"

//----------------------------------------------------------
//
//	A general exception base class, and a few
//	useful exceptions derived from the base class.
//
//----------------------------------------------------------

#include <string>
#include <exception>
#include <sstream>

IEX_INTERNAL_NAMESPACE_HEADER_ENTER


//-------------------------------
// Our most basic exception class
//-------------------------------

class IEX_EXPORT_TYPE BaseExc: public std::exception
{
  public:

    //----------------------------
    // Constructors and destructor
    //----------------------------

    IEX_EXPORT BaseExc (const char *s = nullptr);
    IEX_EXPORT BaseExc (const std::string &s);
    IEX_EXPORT BaseExc (std::string &&s); // not noexcept because of stacktrace
    IEX_EXPORT BaseExc (std::stringstream &s);

    IEX_EXPORT BaseExc (const BaseExc &be);
    IEX_EXPORT BaseExc (BaseExc &&be) noexcept;
    IEX_EXPORT virtual ~BaseExc () noexcept;

    IEX_EXPORT BaseExc & operator = (const BaseExc& be);
    IEX_EXPORT BaseExc & operator = (BaseExc&& be) noexcept;

    //---------------------------------------------------
    // what() method -- e.what() returns _message.c_str()
    //---------------------------------------------------

    IEX_EXPORT virtual const char * what () const noexcept;


    //--------------------------------------------------
    // Convenient methods to change the exception's text
    //--------------------------------------------------

    IEX_EXPORT BaseExc &            assign (std::stringstream &s);	// assign (s.str())
    IEX_EXPORT BaseExc &            operator = (std::stringstream &s);

    IEX_EXPORT BaseExc &            append (std::stringstream &s);	// append (s.str())
    IEX_EXPORT BaseExc &            operator += (std::stringstream &s);


    //--------------------------------------------------
    // These methods from the base class get obscured by
    // the definitions above.
    //--------------------------------------------------

    IEX_EXPORT BaseExc &            assign (const char *s);
    IEX_EXPORT BaseExc &            operator = (const char *s);

    IEX_EXPORT BaseExc &            append (const char *s);
    IEX_EXPORT BaseExc &            operator += (const char *s);

    //---------------------------------------------------
    // Access to the string representation of the message
    //---------------------------------------------------

    IEX_EXPORT const std::string &  message () const noexcept;

    //--------------------------------------------------
    // Stack trace for the point at which the exception
    // was thrown.  The stack trace will be an empty
    // string unless a working stack-tracing routine
    // has been installed (see below, setStackTracer()).
    //--------------------------------------------------

    IEX_EXPORT const std::string &  stackTrace () const noexcept;

  private:

    std::string _message;
    std::string _stackTrace;
};


//-----------------------------------------------------
// A macro to save typing when declararing an exception
// class derived directly or indirectly from BaseExc:
//-----------------------------------------------------

#define DEFINE_EXC_EXP(exp, name, base)                             \
    class IEX_EXPORT_TYPE name: public base                         \
    {                                                               \
      public:                                                       \
        exp name();                                                 \
        exp name (const char* text);                                \
        exp name (const std::string &text);                         \
        exp name (std::string &&text);                              \
        exp name (std::stringstream &text);                         \
        exp name (const name &other);                               \
        exp name (name &&other) noexcept;                           \
        exp name& operator = (name &other);                         \
        exp name& operator = (name &&other) noexcept;               \
        exp ~name() noexcept;                                       \
    };

#define DEFINE_EXC_EXP_IMPL(exp, name, base)                     \
exp name::name () : base () {}                                   \
exp name::name (const char* text) : base (text) {}               \
exp name::name (const std::string& text) : base (text) {}        \
exp name::name (std::string&& text) : base (std::move (text)) {} \
exp name::name (std::stringstream& text) : base (text) {}        \
exp name::name (const name &other) : base (other) {}             \
exp name::name (name &&other) noexcept : base (other) {}         \
exp name& name::operator = (name &other) { base::operator=(other); return *this; } \
exp name& name::operator = (name &&other) noexcept { base::operator=(other); return *this; } \
exp name::~name () noexcept {}

// For backward compatibility.
#define DEFINE_EXC(name, base) DEFINE_EXC_EXP(, name, base)


//--------------------------------------------------------
// Some exceptions which should be useful in most programs
//--------------------------------------------------------
DEFINE_EXC_EXP (IEX_EXPORT, ArgExc, BaseExc)    // Invalid arguments to a function call

DEFINE_EXC_EXP (IEX_EXPORT, LogicExc, BaseExc)  // General error in a program's logic,
                                                // for example, a function was called
                                                // in a context where the call does
                                                // not make sense.

DEFINE_EXC_EXP (IEX_EXPORT, InputExc, BaseExc)  // Invalid input data, e.g. from a file

DEFINE_EXC_EXP (IEX_EXPORT, IoExc, BaseExc)     // Input or output operation failed

DEFINE_EXC_EXP (IEX_EXPORT, MathExc, BaseExc) 	// Arithmetic exception; more specific
                                                // exceptions derived from this class
                                                // are defined in ExcMath.h

DEFINE_EXC_EXP (IEX_EXPORT, ErrnoExc, BaseExc)  // Base class for exceptions corresponding
                                                // to errno values (see errno.h); more
                                                // specific exceptions derived from this
                                                // class are defined in ExcErrno.h

DEFINE_EXC_EXP (IEX_EXPORT, NoImplExc, BaseExc) // Missing method exception e.g. from a
                                                // call to a method that is only partially
                                                // or not at all implemented. A reminder
                                                // to lazy software people to get back
                                                // to work.

DEFINE_EXC_EXP (IEX_EXPORT, NullExc, BaseExc)   // A pointer is inappropriately null.

DEFINE_EXC_EXP (IEX_EXPORT, TypeExc, BaseExc)   // An object is an inappropriate type,
                                                // i.e. a dynamnic_cast failed.


//----------------------------------------------------------------------
// Stack-tracing support:
// 
// setStackTracer(st)
//
//	installs a stack-tracing routine, st, which will be called from
//	class BaseExc's constructor every time an exception derived from
//	BaseExc is thrown.  The stack-tracing routine should return a
//	string that contains a printable representation of the program's
//	current call stack.  This string will be stored in the BaseExc
//	object; the string is accesible via the BaseExc::stackTrace()
//	method.
//
// setStackTracer(0)
//
//	removes the current stack tracing routine.  When an exception
//	derived from BaseExc is thrown, the stack trace string stored
//	in the BaseExc object will be empty.
//
// stackTracer()
//
//	returns a pointer to the current stack-tracing routine, or 0
//	if there is no current stack stack-tracing routine.
// 
//----------------------------------------------------------------------

typedef std::string (* StackTracer) ();

IEX_EXPORT void        setStackTracer (StackTracer stackTracer);
IEX_EXPORT StackTracer stackTracer ();


IEX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // INCLUDED_IEXBASEEXC_H
