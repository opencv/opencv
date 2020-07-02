///////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2002-2012, Industrial Light & Magic, a division of Lucas
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



#ifndef INCLUDED_IEXERRNOEXC_H
#define INCLUDED_IEXERRNOEXC_H

//----------------------------------------------------------------
//
//	Exceptions which correspond to "errno" error codes.
//
//----------------------------------------------------------------

#include "IexBaseExc.h"

IEX_INTERNAL_NAMESPACE_HEADER_ENTER

DEFINE_EXC (EpermExc, ErrnoExc)
DEFINE_EXC (EnoentExc, ErrnoExc)
DEFINE_EXC (EsrchExc, ErrnoExc)
DEFINE_EXC (EintrExc, ErrnoExc)
DEFINE_EXC (EioExc, ErrnoExc)
DEFINE_EXC (EnxioExc, ErrnoExc)
DEFINE_EXC (E2bigExc, ErrnoExc)
DEFINE_EXC (EnoexecExc, ErrnoExc)
DEFINE_EXC (EbadfExc, ErrnoExc)
DEFINE_EXC (EchildExc, ErrnoExc)
DEFINE_EXC (EagainExc, ErrnoExc)
DEFINE_EXC (EnomemExc, ErrnoExc)
DEFINE_EXC (EaccesExc, ErrnoExc)
DEFINE_EXC (EfaultExc, ErrnoExc)
DEFINE_EXC (EnotblkExc, ErrnoExc)
DEFINE_EXC (EbusyExc, ErrnoExc)
DEFINE_EXC (EexistExc, ErrnoExc)
DEFINE_EXC (ExdevExc, ErrnoExc)
DEFINE_EXC (EnodevExc, ErrnoExc)
DEFINE_EXC (EnotdirExc, ErrnoExc)
DEFINE_EXC (EisdirExc, ErrnoExc)
DEFINE_EXC (EinvalExc, ErrnoExc)
DEFINE_EXC (EnfileExc, ErrnoExc)
DEFINE_EXC (EmfileExc, ErrnoExc)
DEFINE_EXC (EnottyExc, ErrnoExc)
DEFINE_EXC (EtxtbsyExc, ErrnoExc)
DEFINE_EXC (EfbigExc, ErrnoExc)
DEFINE_EXC (EnospcExc, ErrnoExc)
DEFINE_EXC (EspipeExc, ErrnoExc)
DEFINE_EXC (ErofsExc, ErrnoExc)
DEFINE_EXC (EmlinkExc, ErrnoExc)
DEFINE_EXC (EpipeExc, ErrnoExc)
DEFINE_EXC (EdomExc, ErrnoExc)
DEFINE_EXC (ErangeExc, ErrnoExc)
DEFINE_EXC (EnomsgExc, ErrnoExc)
DEFINE_EXC (EidrmExc, ErrnoExc)
DEFINE_EXC (EchrngExc, ErrnoExc)
DEFINE_EXC (El2nsyncExc, ErrnoExc)
DEFINE_EXC (El3hltExc, ErrnoExc)
DEFINE_EXC (El3rstExc, ErrnoExc)
DEFINE_EXC (ElnrngExc, ErrnoExc)
DEFINE_EXC (EunatchExc, ErrnoExc)
DEFINE_EXC (EnocsiExc, ErrnoExc)
DEFINE_EXC (El2hltExc, ErrnoExc)
DEFINE_EXC (EdeadlkExc, ErrnoExc)
DEFINE_EXC (EnolckExc, ErrnoExc)
DEFINE_EXC (EbadeExc, ErrnoExc)
DEFINE_EXC (EbadrExc, ErrnoExc)
DEFINE_EXC (ExfullExc, ErrnoExc)
DEFINE_EXC (EnoanoExc, ErrnoExc)
DEFINE_EXC (EbadrqcExc, ErrnoExc)
DEFINE_EXC (EbadsltExc, ErrnoExc)
DEFINE_EXC (EdeadlockExc, ErrnoExc)
DEFINE_EXC (EbfontExc, ErrnoExc)
DEFINE_EXC (EnostrExc, ErrnoExc)
DEFINE_EXC (EnodataExc, ErrnoExc)
DEFINE_EXC (EtimeExc, ErrnoExc)
DEFINE_EXC (EnosrExc, ErrnoExc)
DEFINE_EXC (EnonetExc, ErrnoExc)
DEFINE_EXC (EnopkgExc, ErrnoExc)
DEFINE_EXC (EremoteExc, ErrnoExc)
DEFINE_EXC (EnolinkExc, ErrnoExc)
DEFINE_EXC (EadvExc, ErrnoExc)
DEFINE_EXC (EsrmntExc, ErrnoExc)
DEFINE_EXC (EcommExc, ErrnoExc)
DEFINE_EXC (EprotoExc, ErrnoExc)
DEFINE_EXC (EmultihopExc, ErrnoExc)
DEFINE_EXC (EbadmsgExc, ErrnoExc)
DEFINE_EXC (EnametoolongExc, ErrnoExc)
DEFINE_EXC (EoverflowExc, ErrnoExc)
DEFINE_EXC (EnotuniqExc, ErrnoExc)
DEFINE_EXC (EbadfdExc, ErrnoExc)
DEFINE_EXC (EremchgExc, ErrnoExc)
DEFINE_EXC (ElibaccExc, ErrnoExc)
DEFINE_EXC (ElibbadExc, ErrnoExc)
DEFINE_EXC (ElibscnExc, ErrnoExc)
DEFINE_EXC (ElibmaxExc, ErrnoExc)
DEFINE_EXC (ElibexecExc, ErrnoExc)
DEFINE_EXC (EilseqExc, ErrnoExc)
DEFINE_EXC (EnosysExc, ErrnoExc)
DEFINE_EXC (EloopExc, ErrnoExc)
DEFINE_EXC (ErestartExc, ErrnoExc)
DEFINE_EXC (EstrpipeExc, ErrnoExc)
DEFINE_EXC (EnotemptyExc, ErrnoExc)
DEFINE_EXC (EusersExc, ErrnoExc)
DEFINE_EXC (EnotsockExc, ErrnoExc)
DEFINE_EXC (EdestaddrreqExc, ErrnoExc)
DEFINE_EXC (EmsgsizeExc, ErrnoExc)
DEFINE_EXC (EprototypeExc, ErrnoExc)
DEFINE_EXC (EnoprotooptExc, ErrnoExc)
DEFINE_EXC (EprotonosupportExc, ErrnoExc)
DEFINE_EXC (EsocktnosupportExc, ErrnoExc)
DEFINE_EXC (EopnotsuppExc, ErrnoExc)
DEFINE_EXC (EpfnosupportExc, ErrnoExc)
DEFINE_EXC (EafnosupportExc, ErrnoExc)
DEFINE_EXC (EaddrinuseExc, ErrnoExc)
DEFINE_EXC (EaddrnotavailExc, ErrnoExc)
DEFINE_EXC (EnetdownExc, ErrnoExc)
DEFINE_EXC (EnetunreachExc, ErrnoExc)
DEFINE_EXC (EnetresetExc, ErrnoExc)
DEFINE_EXC (EconnabortedExc, ErrnoExc)
DEFINE_EXC (EconnresetExc, ErrnoExc)
DEFINE_EXC (EnobufsExc, ErrnoExc)
DEFINE_EXC (EisconnExc, ErrnoExc)
DEFINE_EXC (EnotconnExc, ErrnoExc)
DEFINE_EXC (EshutdownExc, ErrnoExc)
DEFINE_EXC (EtoomanyrefsExc, ErrnoExc)
DEFINE_EXC (EtimedoutExc, ErrnoExc)
DEFINE_EXC (EconnrefusedExc, ErrnoExc)
DEFINE_EXC (EhostdownExc, ErrnoExc)
DEFINE_EXC (EhostunreachExc, ErrnoExc)
DEFINE_EXC (EalreadyExc, ErrnoExc)
DEFINE_EXC (EinprogressExc, ErrnoExc)
DEFINE_EXC (EstaleExc, ErrnoExc)
DEFINE_EXC (EioresidExc, ErrnoExc)
DEFINE_EXC (EucleanExc, ErrnoExc)
DEFINE_EXC (EnotnamExc, ErrnoExc)
DEFINE_EXC (EnavailExc, ErrnoExc)
DEFINE_EXC (EisnamExc, ErrnoExc)
DEFINE_EXC (EremoteioExc, ErrnoExc)
DEFINE_EXC (EinitExc, ErrnoExc)
DEFINE_EXC (EremdevExc, ErrnoExc)
DEFINE_EXC (EcanceledExc, ErrnoExc)
DEFINE_EXC (EnolimfileExc, ErrnoExc)
DEFINE_EXC (EproclimExc, ErrnoExc)
DEFINE_EXC (EdisjointExc, ErrnoExc)
DEFINE_EXC (EnologinExc, ErrnoExc)
DEFINE_EXC (EloginlimExc, ErrnoExc)
DEFINE_EXC (EgrouploopExc, ErrnoExc)
DEFINE_EXC (EnoattachExc, ErrnoExc)
DEFINE_EXC (EnotsupExc, ErrnoExc)
DEFINE_EXC (EnoattrExc, ErrnoExc)
DEFINE_EXC (EdircorruptedExc, ErrnoExc)
DEFINE_EXC (EdquotExc, ErrnoExc)
DEFINE_EXC (EnfsremoteExc, ErrnoExc)
DEFINE_EXC (EcontrollerExc, ErrnoExc)
DEFINE_EXC (EnotcontrollerExc, ErrnoExc)
DEFINE_EXC (EenqueuedExc, ErrnoExc)
DEFINE_EXC (EnotenqueuedExc, ErrnoExc)
DEFINE_EXC (EjoinedExc, ErrnoExc)
DEFINE_EXC (EnotjoinedExc, ErrnoExc)
DEFINE_EXC (EnoprocExc, ErrnoExc)
DEFINE_EXC (EmustrunExc, ErrnoExc)
DEFINE_EXC (EnotstoppedExc, ErrnoExc)
DEFINE_EXC (EclockcpuExc, ErrnoExc)
DEFINE_EXC (EinvalstateExc, ErrnoExc)
DEFINE_EXC (EnoexistExc, ErrnoExc)
DEFINE_EXC (EendofminorExc, ErrnoExc)
DEFINE_EXC (EbufsizeExc, ErrnoExc)
DEFINE_EXC (EemptyExc, ErrnoExc)
DEFINE_EXC (EnointrgroupExc, ErrnoExc)
DEFINE_EXC (EinvalmodeExc, ErrnoExc)
DEFINE_EXC (EcantextentExc, ErrnoExc)
DEFINE_EXC (EinvaltimeExc, ErrnoExc)
DEFINE_EXC (EdestroyedExc, ErrnoExc)

IEX_INTERNAL_NAMESPACE_HEADER_EXIT

#endif
