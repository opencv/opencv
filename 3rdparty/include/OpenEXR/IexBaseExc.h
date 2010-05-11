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



#ifndef INCLUDED_IEXBASEEXC_H
#define INCLUDED_IEXBASEEXC_H


//----------------------------------------------------------
//
//	A general exception base class, and a few
//	useful exceptions derived from the base class.
//
//----------------------------------------------------------

#include <string>
#include <exception>
#include <sstream>

namespace Iex {

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
// Tell MS VC++ to suppress exception specification warnings
#pragma warning(disable:4290)
#endif

//-------------------------------
// Our most basic exception class
//-------------------------------

class BaseExc: public std::string, public std::exception
{
  public:

    //----------------------------
    // Constructors and destructor
    //----------------------------

    BaseExc (const char *s = 0) throw();	// std::string (s)
    BaseExc (const std::string &s) throw();	// std::string (s)
    BaseExc (std::stringstream &s) throw();	// std::string (s.str())

    BaseExc (const BaseExc &be) throw();
    virtual ~BaseExc () throw ();

    //--------------------------------------------
    // what() method -- e.what() returns e.c_str()
    //--------------------------------------------

    virtual const char * what () const throw ();


    //--------------------------------------------------
    // Convenient methods to change the exception's text
    //--------------------------------------------------

    BaseExc &		assign (std::stringstream &s);	// assign (s.str())
    BaseExc &		operator = (std::stringstream &s);

    BaseExc &		append (std::stringstream &s);	// append (s.str())
    BaseExc &		operator += (std::stringstream &s);


    //--------------------------------------------------
    // These methods from the base class get obscured by
    // the definitions above.
    //--------------------------------------------------

    BaseExc &		assign (const char *s);
    BaseExc &		operator = (const char *s);

    BaseExc &		append (const char *s);
    BaseExc &		operator += (const char *s);


    //--------------------------------------------------
    // Stack trace for the point at which the exception
    // was thrown.  The stack trace will be an empty
    // string unless a working stack-tracing routine
    // has been installed (see below, setStackTracer()).
    //--------------------------------------------------

    const std::string &	stackTrace () const;

  private:

    std::string		_stackTrace;
};


//-----------------------------------------------------
// A macro to save typing when declararing an exception
// class derived directly or indirectly from BaseExc:
//-----------------------------------------------------

#define DEFINE_EXC(name, base)				        \
    class name: public base				        \
    {							        \
      public:                                                   \
	name (const char* text=0)      throw(): base (text) {}	\
	name (const std::string &text) throw(): base (text) {}	\
	name (std::stringstream &text) throw(): base (text) {}	\
    };


//--------------------------------------------------------
// Some exceptions which should be useful in most programs
//--------------------------------------------------------

DEFINE_EXC (ArgExc,   BaseExc) 	 // Invalid arguments to a function call

DEFINE_EXC (LogicExc, BaseExc) 	 // General error in a program's logic,
				 // for example, a function was called
				 // in a context where the call does
				 // not make sense.

DEFINE_EXC (InputExc, BaseExc) 	 // Invalid input data, e.g. from a file

DEFINE_EXC (IoExc, BaseExc) 	 // Input or output operation failed

DEFINE_EXC (MathExc,  BaseExc) 	 // Arithmetic exception; more specific
				 // exceptions derived from this class
				 // are defined in ExcMath.h

DEFINE_EXC (ErrnoExc, BaseExc) 	 // Base class for exceptions corresponding
				 // to errno values (see errno.h); more
				 // specific exceptions derived from this
				 // class are defined in ExcErrno.h

DEFINE_EXC (NoImplExc, BaseExc)  // Missing method exception e.g. from a
				 // call to a method that is only partially
				 // or not at all implemented. A reminder
				 // to lazy software people to get back
				 // to work.

DEFINE_EXC (NullExc, BaseExc) 	 // A pointer is inappropriately null.

DEFINE_EXC (TypeExc, BaseExc) 	 // An object is an inappropriate type,
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

void		setStackTracer (StackTracer stackTracer);
StackTracer	stackTracer ();


//-----------------
// Inline functions
//-----------------

inline BaseExc &
BaseExc::operator = (std::stringstream &s)
{
    return assign (s);
}


inline BaseExc &
BaseExc::operator += (std::stringstream &s)
{
    return append (s);
}


inline BaseExc &
BaseExc::assign (const char *s)
{
    std::string::assign(s);
    return *this;
}


inline BaseExc &
BaseExc::operator = (const char *s)
{
    return assign(s);
}


inline BaseExc &
BaseExc::append (const char *s)
{
    std::string::append(s);
    return *this;
}


inline BaseExc &
BaseExc::operator += (const char *s)
{
    return append(s);
}


inline const std::string &
BaseExc::stackTrace () const
{
    return _stackTrace;
}

#if (defined _WIN32 || defined _WIN64) && defined _MSC_VER
#pragma warning(default:4290)
#endif

} // namespace Iex

#endif
