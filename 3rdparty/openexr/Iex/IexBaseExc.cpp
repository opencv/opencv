//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//


//---------------------------------------------------------------------
//
//	Constructors and destructors for our exception base class.
//
//---------------------------------------------------------------------

#include "IexExport.h"
#include "IexBaseExc.h"
#include "IexMacros.h"
#include "IexErrnoExc.h"
#include "IexMathExc.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdlib.h>

#if defined(_MSC_VER)
#pragma warning (disable : 4996)
#endif


IEX_INTERNAL_NAMESPACE_SOURCE_ENTER


namespace {

StackTracer currentStackTracer = 0;

} // namespace


void	
setStackTracer (StackTracer stackTracer)
{
    currentStackTracer = stackTracer;
}


StackTracer
stackTracer ()
{
    return currentStackTracer;
}


BaseExc::BaseExc (const char* s) :
    _message (s? s : ""),
    _stackTrace (currentStackTracer? currentStackTracer(): std::string())
{
}


BaseExc::BaseExc (const std::string &s) :
    _message (s),
    _stackTrace (currentStackTracer? currentStackTracer(): std::string())
{
    // empty
}


BaseExc::BaseExc (std::string &&s) :
    _message (std::move (s)),
    _stackTrace (currentStackTracer? currentStackTracer(): std::string())
{
    // empty
}


BaseExc::BaseExc (std::stringstream &s) :
    _message (s.str()),
    _stackTrace (currentStackTracer? currentStackTracer(): std::string())
{
    // empty
}

BaseExc::BaseExc (const BaseExc &be)
    : _message (be._message),
      _stackTrace (be._stackTrace)
{
}

BaseExc::~BaseExc () noexcept
{
}

BaseExc &
BaseExc::operator = (const BaseExc& be)
{
    if (this != &be)
    {
        _message = be._message;
        _stackTrace = be._stackTrace;
    }

    return *this;
}

BaseExc &
BaseExc::operator = (BaseExc&& be) noexcept
{
    if (this != &be)
    {
        _message = std::move (be._message);
        _stackTrace = std::move (be._stackTrace);
    }
    return *this;
}

const char *
BaseExc::what () const noexcept
{
    return _message.c_str();
}


BaseExc &
BaseExc::assign (std::stringstream &s)
{
    _message.assign (s.str());
    return *this;
}

BaseExc &
BaseExc::append (std::stringstream &s)
{
    _message.append (s.str());
    return *this;
}

const std::string &
BaseExc::message() const noexcept
{
	return _message;
}

BaseExc &
BaseExc::operator = (std::stringstream &s)
{
    return assign (s);
}


BaseExc &
BaseExc::operator += (std::stringstream &s)
{
    return append (s);
}


BaseExc &
BaseExc::assign (const char *s)
{
    _message.assign(s);
    return *this;
}


BaseExc &
BaseExc::operator = (const char *s)
{
    return assign(s);
}


BaseExc &
BaseExc::append (const char *s)
{
    _message.append(s);
    return *this;
}


BaseExc &
BaseExc::operator += (const char *s)
{
    return append(s);
}


const std::string &
BaseExc::stackTrace () const noexcept
{
    return _stackTrace;
}

/// @cond Doxygen_Suppress

DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ArgExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, LogicExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, InputExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, IoExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, MathExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ErrnoExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, NoImplExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, NullExc, BaseExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, TypeExc, BaseExc)

DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EpermExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoentExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EsrchExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EintrExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EioExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnxioExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, E2bigExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoexecExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbadfExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EchildExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EagainExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnomemExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EaccesExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EfaultExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotblkExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbusyExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EexistExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ExdevExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnodevExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotdirExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EisdirExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EinvalExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnfileExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EmfileExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnottyExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EtxtbsyExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EfbigExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnospcExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EspipeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ErofsExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EmlinkExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EpipeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdomExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ErangeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnomsgExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EidrmExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EchrngExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, El2nsyncExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, El3hltExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, El3rstExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ElnrngExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EunatchExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnocsiExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, El2hltExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdeadlkExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnolckExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbadeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbadrExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ExfullExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoanoExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbadrqcExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbadsltExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdeadlockExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbfontExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnostrExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnodataExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EtimeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnosrExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnonetExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnopkgExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EremoteExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnolinkExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EadvExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EsrmntExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EcommExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EprotoExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EmultihopExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbadmsgExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnametoolongExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EoverflowExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotuniqExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbadfdExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EremchgExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ElibaccExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ElibbadExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ElibscnExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ElibmaxExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ElibexecExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EilseqExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnosysExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EloopExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, ErestartExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EstrpipeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotemptyExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EusersExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotsockExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdestaddrreqExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EmsgsizeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EprototypeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoprotooptExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EprotonosupportExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EsocktnosupportExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EopnotsuppExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EpfnosupportExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EafnosupportExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EaddrinuseExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EaddrnotavailExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnetdownExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnetunreachExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnetresetExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EconnabortedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EconnresetExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnobufsExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EisconnExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotconnExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EshutdownExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EtoomanyrefsExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EtimedoutExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EconnrefusedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EhostdownExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EhostunreachExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EalreadyExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EinprogressExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EstaleExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EioresidExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EucleanExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotnamExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnavailExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EisnamExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EremoteioExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EinitExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EremdevExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EcanceledExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnolimfileExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EproclimExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdisjointExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnologinExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EloginlimExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EgrouploopExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoattachExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotsupExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoattrExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdircorruptedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdquotExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnfsremoteExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EcontrollerExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotcontrollerExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EenqueuedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotenqueuedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EjoinedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotjoinedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoprocExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EmustrunExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnotstoppedExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EclockcpuExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EinvalstateExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnoexistExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EendofminorExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EbufsizeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EemptyExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EnointrgroupExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EinvalmodeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EcantextentExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EinvaltimeExc, ErrnoExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, EdestroyedExc, ErrnoExc)

DEFINE_EXC_EXP_IMPL (IEX_EXPORT, OverflowExc, MathExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, UnderflowExc, MathExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, DivzeroExc, MathExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, InexactExc, MathExc)
DEFINE_EXC_EXP_IMPL (IEX_EXPORT, InvalidFpOpExc, MathExc)

/// @endcond Doxygen_Suppress

IEX_INTERNAL_NAMESPACE_SOURCE_EXIT


#ifdef _WIN32

#pragma optimize("", off)
void
iex_debugTrap()
{
    if (0 != getenv("IEXDEBUGTHROW"))
        ::DebugBreak();
}
#else
void
iex_debugTrap()
{
    // how to in Linux?
    if (0 != ::getenv("IEXDEBUGTHROW"))
        __builtin_trap();
}
#endif
