//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.
//

//------------------------------------------------------------------------
//
//	Functions to control floating point exceptions.
//
//------------------------------------------------------------------------

#include "IexMathFpu.h"

#include <stdint.h>
#include <IexConfig.h>
#include <stdio.h>

#if 0
    #include <iostream>
    #define debug(x) (std::cout << x << std::flush)
#else
    #define debug(x)
#endif

#include <IexConfigInternal.h>
#if defined(HAVE_UCONTEXT_H) &&                                                \
    (defined(IEX_HAVE_SIGCONTEXT_CONTROL_REGISTER_SUPPORT) ||              \
     defined(IEX_HAVE_CONTROL_REGISTER_SUPPORT))

#        include <ucontext.h>
#        include <signal.h>
#        include <iostream>
#        include <stdint.h>


IEX_INTERNAL_NAMESPACE_SOURCE_ENTER



namespace FpuControl
{

//-------------------------------------------------------------------
//
//    Modern x86 processors and all AMD64 processors have two
//    sets of floating-point control/status registers: cw and sw
//    for legacy x87 stack-based arithmetic, and mxcsr for
//    SIMD arithmetic.  When setting exception masks or checking
//    for exceptions, we must set/check all relevant registers,
//    since applications may contain code that uses either FP
//    model.
//
//    These functions handle both FP models for x86 and AMD64.
//
//-------------------------------------------------------------------

//-------------------------------------------------------------------
//
//    Restore the control register state from a signal handler
//    user context, optionally clearing the exception bits
//    in the restored control register, if applicable.
//
//-------------------------------------------------------------------

void restoreControlRegs (const ucontext_t & ucon,
			 bool clearExceptions = false);


//------------------------------------------------------------
//
//    Set exception mask bits in the control register state.
//    A value of 1 means the exception is masked, a value of
//    0 means the exception is enabled.
//
//    setExceptionMask returns the previous mask value.  If
//    the 'exceptions' pointer is non-null, it returns in 
//    this argument the FPU exception bits.
//
//------------------------------------------------------------

const int INVALID_EXC   = (1<<0);
const int DENORMAL_EXC  = (1<<1);
const int DIVZERO_EXC   = (1<<2);
const int OVERFLOW_EXC  = (1<<3);
const int UNDERFLOW_EXC = (1<<4);
const int INEXACT_EXC   = (1<<5);
const int ALL_EXC       = INVALID_EXC  | DENORMAL_EXC  | DIVZERO_EXC |
                          OVERFLOW_EXC | UNDERFLOW_EXC | INEXACT_EXC;

int setExceptionMask (int mask, int * exceptions = 0);
int getExceptionMask ();


//---------------------------------------------
//
//    Get/clear the exception bits in the FPU.
//
//---------------------------------------------

int  getExceptions ();
void clearExceptions ();


//------------------------------------------------------------------
//
//    Everything below here is implementation.  Do not use these
//    constants or functions in your applications or libraries.
//    This is not the code you're looking for.  Move along.
//
//    Optimization notes -- on a Pentium 4, at least, it appears
//    to be faster to get the mxcsr first and then the cw; and to
//    set the cw first and then the mxcsr.  Also, it seems to
//    be faster to clear the sw exception bits after setting
//    cw and mxcsr.
//
//------------------------------------------------------------------

static inline uint16_t
getSw ()
{
    uint16_t sw;
    asm volatile ("fnstsw %0" : "=m" (sw) : );
    return sw;
}

static inline void
setCw (uint16_t cw)
{
    asm volatile ("fldcw %0" : : "m" (cw) );
}

static inline uint16_t
getCw ()
{
    uint16_t cw;
    asm volatile ("fnstcw %0" : "=m" (cw) : );
    return cw;
}

static inline void
setMxcsr (uint32_t mxcsr, bool clearExceptions)
{
    mxcsr &= clearExceptions ? 0xffffffc0 : 0xffffffff;
    asm volatile ("ldmxcsr %0" : : "m" (mxcsr) );
}

static inline uint32_t
getMxcsr ()
{
    uint32_t mxcsr;
    asm volatile ("stmxcsr %0" : "=m" (mxcsr) : );
    return mxcsr;
}

static inline int
calcMask (uint16_t cw, uint32_t mxcsr)
{
    //
    // Hopefully, if the user has been using FpuControl functions,
    // the masks are the same, but just in case they're not, we
    // AND them together to report the proper subset of the masks.
    //

    return (cw & ALL_EXC) & ((mxcsr >> 7) & ALL_EXC);
}

inline int
setExceptionMask (int mask, int * exceptions)
{
    uint16_t cw = getCw ();
    uint32_t mxcsr = getMxcsr ();
    
    if (exceptions)
	*exceptions = (mxcsr & ALL_EXC) | (getSw () & ALL_EXC);

    int oldmask = calcMask (cw, mxcsr);

    //
    // The exception constants are chosen very carefully so that
    // we can do a simple mask and shift operation to insert
    // them into the control words.  The mask operation is for 
    // safety, in case the user accidentally set some other
    // bits in the exception mask.
    //

    mask &= ALL_EXC;
    cw = (cw & ~ALL_EXC) | mask;
    mxcsr = (mxcsr & ~(ALL_EXC << 7)) | (mask << 7);

    setCw (cw);
    setMxcsr (mxcsr, false);

    return oldmask;
}

inline int
getExceptionMask ()
{
    uint32_t mxcsr = getMxcsr ();
    uint16_t cw = getCw ();
    return calcMask (cw, mxcsr);
}

inline int
getExceptions ()
{
    return (getMxcsr () | getSw ()) & ALL_EXC;
}

void
clearExceptions ()
{
    uint32_t mxcsr = getMxcsr () & 0xffffffc0;
    asm volatile ("ldmxcsr %0\n"
		  "fnclex"
		  : : "m" (mxcsr) );
}

// If the fpe was taken while doing a float-to-int cast using the x87,
// the rounding mode and possibly the precision will be wrong.  So instead
// of restoring to the state as of the fault, we force the rounding mode
// to be 'nearest' and the precision to be double extended.
//
// rounding mode is in bits 10-11, value 00 == round to nearest
// precision is in bits 8-9, value 11 == double extended (80-bit)
//
const uint16_t cwRestoreMask = ~((3 << 10) | (3 << 8));
const uint16_t cwRestoreVal = (0 << 10) | (3 << 8);


#ifdef IEX_HAVE_CONTROL_REGISTER_SUPPORT

inline void
restoreControlRegs (const ucontext_t & ucon, bool clearExceptions)
{
    setCw ((ucon.uc_mcontext.fpregs->cwd & cwRestoreMask) | cwRestoreVal);
    setMxcsr (ucon.uc_mcontext.fpregs->mxcsr, clearExceptions);
}

#else

//
// Ugly, the mxcsr isn't defined in GNU libc ucontext_t, but
// it's passed to the signal handler by the kernel.  Use
// the kernel's version of the ucontext to get it, see
// <asm/sigcontext.h>
//

#include <asm/sigcontext.h>

inline void
restoreControlRegs (const ucontext_t & ucon, bool clearExceptions)
{
#if defined(__GLIBC__) || defined(__i386__)
    setCw ((ucon.uc_mcontext.fpregs->cw & cwRestoreMask) | cwRestoreVal);
#else
    setCw ((ucon.uc_mcontext.fpregs->cwd & cwRestoreMask) | cwRestoreVal);
#endif
    
    _fpstate * kfp = reinterpret_cast<_fpstate *> (ucon.uc_mcontext.fpregs);
#if defined(__GLIBC__) || defined(__i386__)
    setMxcsr (kfp->magic == 0 ? kfp->mxcsr : 0, clearExceptions);
#else
    setMxcsr (kfp->mxcsr, clearExceptions);
#endif
}

#endif

} // namespace FpuControl


namespace {

volatile FpExceptionHandler fpeHandler = 0;

extern "C" void
catchSigFpe (int sig, siginfo_t *info, ucontext_t *ucon)
{
    debug ("catchSigFpe (sig = "<< sig << ", ...)\n");

    FpuControl::restoreControlRegs (*ucon, true);

    if (fpeHandler == 0)
	return;

    if (info->si_code == SI_USER)
    {
	fpeHandler (0, "Floating-point exception, caused by "
		       "a signal sent from another process.");
	return;
    }

    if (sig == SIGFPE)
    {
	switch (info->si_code)
	{
	  //
	  // IEEE 754 floating point exceptions:
	  //

	  case FPE_FLTDIV:
	    fpeHandler (IEEE_DIVZERO, "Floating-point division by zero.");
	    return;

	  case FPE_FLTOVF:
	    fpeHandler (IEEE_OVERFLOW, "Floating-point overflow.");
	    return;

	  case FPE_FLTUND:
	    fpeHandler (IEEE_UNDERFLOW, "Floating-point underflow.");
	    return;

	  case FPE_FLTRES:
	    fpeHandler (IEEE_INEXACT, "Inexact floating-point result.");
	    return;

	  case FPE_FLTINV:
	    fpeHandler (IEEE_INVALID, "Invalid floating-point operation.");
	    return;

	  //
	  // Other arithmetic exceptions which can also
	  // be trapped by the operating system:
	  //

	  case FPE_INTDIV:
	    fpeHandler (0, "Integer division by zero.");
	    break;

	  case FPE_INTOVF:
	    fpeHandler (0, "Integer overflow.");
	    break;

	  case FPE_FLTSUB:
	    fpeHandler (0, "Subscript out of range.");
	    break;
	}
    }

    fpeHandler (0, "Floating-point exception.");
}

} // namespace

void
setFpExceptions (int when)
{
    int mask = FpuControl::ALL_EXC;

    if (when & IEEE_OVERFLOW)
	mask &= ~FpuControl::OVERFLOW_EXC;
    if (when & IEEE_UNDERFLOW)
	mask &= ~FpuControl::UNDERFLOW_EXC;
    if (when & IEEE_DIVZERO)
	mask &= ~FpuControl::DIVZERO_EXC;
    if (when & IEEE_INEXACT)
	mask &= ~FpuControl::INEXACT_EXC;
    if (when & IEEE_INVALID)
	mask &= ~FpuControl::INVALID_EXC;

    //
    // The Linux kernel apparently sometimes passes
    // incorrect si_info to signal handlers unless
    // the exception flags are cleared.
    //
    // XXX is this still true on 2.4+ kernels?
    //
    
    FpuControl::setExceptionMask (mask);
    FpuControl::clearExceptions ();
}


int
fpExceptions ()
{
    int mask = FpuControl::getExceptionMask ();

    int when = 0;

    if (!(mask & FpuControl::OVERFLOW_EXC))
	when |= IEEE_OVERFLOW;
    if (!(mask & FpuControl::UNDERFLOW_EXC))
	when |= IEEE_UNDERFLOW;
    if (!(mask & FpuControl::DIVZERO_EXC))
	when |= IEEE_DIVZERO;
    if (!(mask & FpuControl::INEXACT_EXC))
	when |= IEEE_INEXACT;
    if (!(mask & FpuControl::INVALID_EXC))
	when |= IEEE_INVALID;

    return when;
}

void
handleExceptionsSetInRegisters()
{
    if (fpeHandler == 0)
	return;

    int mask = FpuControl::getExceptionMask ();

    int exc = FpuControl::getExceptions();

    if (!(mask & FpuControl::DIVZERO_EXC) && (exc & FpuControl::DIVZERO_EXC))
    {
        fpeHandler(IEEE_DIVZERO, "Floating-point division by zero.");
        return;
    }

    if (!(mask & FpuControl::OVERFLOW_EXC) && (exc & FpuControl::OVERFLOW_EXC))
    {
        fpeHandler(IEEE_OVERFLOW, "Floating-point overflow.");
        return;
    }

    if (!(mask & FpuControl::UNDERFLOW_EXC) && (exc & FpuControl::UNDERFLOW_EXC))
    {
        fpeHandler(IEEE_UNDERFLOW, "Floating-point underflow.");
        return;
    }

    if (!(mask & FpuControl::INEXACT_EXC) && (exc & FpuControl::INEXACT_EXC))
    {
        fpeHandler(IEEE_INEXACT, "Inexact floating-point result.");
        return;
    }

    if (!(mask & FpuControl::INVALID_EXC) && (exc & FpuControl::INVALID_EXC))
    {
        fpeHandler(IEEE_INVALID, "Invalid floating-point operation.");
        return;
    }
}


void
setFpExceptionHandler (FpExceptionHandler handler)
{
    if (fpeHandler == 0)
    {
	struct sigaction action;
	sigemptyset (&action.sa_mask);
	action.sa_flags = SA_SIGINFO | SA_NOMASK;
	action.sa_sigaction = (void (*) (int, siginfo_t *, void *)) catchSigFpe;
	action.sa_restorer = 0;

	sigaction (SIGFPE, &action, 0);
    }

    fpeHandler = handler;
}


IEX_INTERNAL_NAMESPACE_SOURCE_EXIT


#else

#include <signal.h>
#include <assert.h>

IEX_INTERNAL_NAMESPACE_SOURCE_ENTER


namespace 
{
	volatile FpExceptionHandler fpeHandler = 0;
	void fpExc_(int x)
	{
	    if (fpeHandler != 0)
	    {
		fpeHandler(x, "");
	    }
	    else
	    {
		assert(0 != "Floating point exception");
	    }
	}
}

void
setFpExceptions( int )
{
}


void
setFpExceptionHandler (FpExceptionHandler handler)
{
    // improve floating point exception handling nanoscopically above "nothing at all"
    fpeHandler = handler;
    signal(SIGFPE, fpExc_);
}

int
fpExceptions()
{
    return 0;
}

void
handleExceptionsSetInRegisters()
{
    // No implementation on this platform
}

IEX_INTERNAL_NAMESPACE_SOURCE_EXIT

#endif
