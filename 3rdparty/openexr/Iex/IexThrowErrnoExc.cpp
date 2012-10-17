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



//----------------------------------------------------------------
//
//	Exceptions that correspond to "errno" error codes,
//	and a function to make throwing those exceptions easy.
//
//----------------------------------------------------------------

#include "IexThrowErrnoExc.h"
#include "IexErrnoExc.h"
#include <string.h>
#include <errno.h>

namespace Iex {


void throwErrnoExc (const std::string &text, int errnum)
{
    const char *entext = strerror (errnum);
    std::string tmp (text);
    std::string::size_type pos;

    while (std::string::npos != (pos = tmp.find ("%T")))
    tmp.replace (pos, 2, entext, strlen (entext));

    switch (errnum)
    {
      #if defined (EPERM)
      case EPERM:
        throw EpermExc (tmp);
      #endif

      #if defined (ENOENT)
      case ENOENT:
        throw EnoentExc (tmp);
      #endif

      #if defined (ESRCH)
      case ESRCH:
        throw EsrchExc (tmp);
      #endif

      #if defined (EINTR)
      case EINTR:
        throw EintrExc (tmp);
      #endif

      #if defined (EIO)
      case EIO:
        throw EioExc (tmp);
      #endif

      #if defined (ENXIO)
      case ENXIO:
        throw EnxioExc (tmp);
      #endif

      #if defined (E2BIG)
      case E2BIG:
        throw E2bigExc (tmp);
      #endif

      #if defined (ENOEXEC)
      case ENOEXEC:
        throw EnoexecExc (tmp);
      #endif

      #if defined (EBADF)
      case EBADF:
        throw EbadfExc (tmp);
      #endif

      #if defined (ECHILD)
      case ECHILD:
        throw EchildExc (tmp);
      #endif

      #if defined (EAGAIN)
      case EAGAIN:
        throw EagainExc (tmp);
      #endif

      #if defined (ENOMEM)
      case ENOMEM:
        throw EnomemExc (tmp);
      #endif

      #if defined (EACCES)
      case EACCES:
        throw EaccesExc (tmp);
      #endif

      #if defined (EFAULT)
      case EFAULT:
        throw EfaultExc (tmp);
      #endif

      #if defined (ENOTBLK)
      case ENOTBLK:
        throw EnotblkExc (tmp);
      #endif

      #if defined (EBUSY)
      case EBUSY:
        throw EbusyExc (tmp);
      #endif

      #if defined (EEXIST)
      case EEXIST:
        throw EexistExc (tmp);
      #endif

      #if defined (EXDEV)
      case EXDEV:
        throw ExdevExc (tmp);
      #endif

      #if defined (ENODEV)
      case ENODEV:
        throw EnodevExc (tmp);
      #endif

      #if defined (ENOTDIR)
      case ENOTDIR:
        throw EnotdirExc (tmp);
      #endif

      #if defined (EISDIR)
      case EISDIR:
        throw EisdirExc (tmp);
      #endif

      #if defined (EINVAL)
      case EINVAL:
        throw EinvalExc (tmp);
      #endif

      #if defined (ENFILE)
      case ENFILE:
        throw EnfileExc (tmp);
      #endif

      #if defined (EMFILE)
      case EMFILE:
        throw EmfileExc (tmp);
      #endif

      #if defined (ENOTTY)
      case ENOTTY:
        throw EnottyExc (tmp);
      #endif

      #if defined (ETXTBSY)
      case ETXTBSY:
        throw EtxtbsyExc (tmp);
      #endif

      #if defined (EFBIG)
      case EFBIG:
        throw EfbigExc (tmp);
      #endif

      #if defined (ENOSPC)
      case ENOSPC:
        throw EnospcExc (tmp);
      #endif

      #if defined (ESPIPE)
      case ESPIPE:
        throw EspipeExc (tmp);
      #endif

      #if defined (EROFS)
      case EROFS:
        throw ErofsExc (tmp);
      #endif

      #if defined (EMLINK)
      case EMLINK:
        throw EmlinkExc (tmp);
      #endif

      #if defined (EPIPE)
      case EPIPE:
        throw EpipeExc (tmp);
      #endif

      #if defined (EDOM)
      case EDOM:
        throw EdomExc (tmp);
      #endif

      #if defined (ERANGE)
      case ERANGE:
        throw ErangeExc (tmp);
      #endif

      #if defined (ENOMSG)
      case ENOMSG:
        throw EnomsgExc (tmp);
      #endif

      #if defined (EIDRM)
      case EIDRM:
        throw EidrmExc (tmp);
      #endif

      #if defined (ECHRNG)
      case ECHRNG:
        throw EchrngExc (tmp);
      #endif

      #if defined (EL2NSYNC)
      case EL2NSYNC:
        throw El2nsyncExc (tmp);
      #endif

      #if defined (EL3HLT)
      case EL3HLT:
        throw El3hltExc (tmp);
      #endif

      #if defined (EL3RST)
      case EL3RST:
        throw El3rstExc (tmp);
      #endif

      #if defined (ELNRNG)
      case ELNRNG:
        throw ElnrngExc (tmp);
      #endif

      #if defined (EUNATCH)
      case EUNATCH:
        throw EunatchExc (tmp);
      #endif

      #if defined (ENOSCI)
      case ENOCSI:
        throw EnocsiExc (tmp);
      #endif

      #if defined (EL2HLT)
      case EL2HLT:
        throw El2hltExc (tmp);
      #endif

      #if defined (EDEADLK)
      case EDEADLK:
        throw EdeadlkExc (tmp);
      #endif

      #if defined (ENOLCK)
      case ENOLCK:
        throw EnolckExc (tmp);
      #endif

      #if defined (EBADE)
      case EBADE:
        throw EbadeExc (tmp);
      #endif

      #if defined (EBADR)
      case EBADR:
        throw EbadrExc (tmp);
      #endif

      #if defined (EXFULL)
      case EXFULL:
        throw ExfullExc (tmp);
      #endif

      #if defined (ENOANO)
      case ENOANO:
        throw EnoanoExc (tmp);
      #endif

      #if defined (EBADRQC)
      case EBADRQC:
        throw EbadrqcExc (tmp);
      #endif

      #if defined (EBADSLT)
      case EBADSLT:
        throw EbadsltExc (tmp);
      #endif

      #if defined (EDEADLOCK) && defined (EDEADLK)
      #if EDEADLOCK != EDEADLK
          case EDEADLOCK:
        throw EdeadlockExc (tmp);
      #endif
      #elif defined (EDEADLOCK)
      case EDEADLOCK:
        throw EdeadlockExc (tmp);
      #endif

      #if defined (EBFONT)
      case EBFONT:
        throw EbfontExc (tmp);
      #endif

      #if defined (ENOSTR)
      case ENOSTR:
        throw EnostrExc (tmp);
      #endif

      #if defined (ENODATA)
      case ENODATA:
        throw EnodataExc (tmp);
      #endif

      #if defined (ETIME)
      case ETIME:
        throw EtimeExc (tmp);
      #endif

      #if defined (ENOSR)
      case ENOSR:
        throw EnosrExc (tmp);
      #endif

      #if defined (ENONET)
      case ENONET:
        throw EnonetExc (tmp);
      #endif

      #if defined (ENOPKG)
      case ENOPKG:
        throw EnopkgExc (tmp);
      #endif

      #if defined (EREMOTE)
      case EREMOTE:
        throw EremoteExc (tmp);
      #endif

      #if defined (ENOLINK)
      case ENOLINK:
        throw EnolinkExc (tmp);
      #endif

      #if defined (EADV)
      case EADV:
        throw EadvExc (tmp);
      #endif

      #if defined (ESRMNT)
      case ESRMNT:
        throw EsrmntExc (tmp);
      #endif

      #if defined (ECOMM)
      case ECOMM:
        throw EcommExc (tmp);
      #endif

      #if defined (EPROTO)
      case EPROTO:
        throw EprotoExc (tmp);
      #endif

      #if defined (EMULTIHOP)
      case EMULTIHOP:
        throw EmultihopExc (tmp);
      #endif

      #if defined (EBADMSG)
      case EBADMSG:
        throw EbadmsgExc (tmp);
      #endif

      #if defined (ENAMETOOLONG)
      case ENAMETOOLONG:
        throw EnametoolongExc (tmp);
      #endif

      #if defined (EOVERFLOW)
      case EOVERFLOW:
        throw EoverflowExc (tmp);
      #endif

      #if defined (ENOTUNIQ)
      case ENOTUNIQ:
        throw EnotuniqExc (tmp);
      #endif

      #if defined (EBADFD)
      case EBADFD:
        throw EbadfdExc (tmp);
      #endif

      #if defined (EREMCHG)
      case EREMCHG:
        throw EremchgExc (tmp);
      #endif

      #if defined (ELIBACC)
      case ELIBACC:
        throw ElibaccExc (tmp);
      #endif

      #if defined (ELIBBAD)
      case ELIBBAD:
        throw ElibbadExc (tmp);
      #endif

      #if defined (ELIBSCN)
      case ELIBSCN:
        throw ElibscnExc (tmp);
      #endif

      #if defined (ELIBMAX)
      case ELIBMAX:
        throw ElibmaxExc (tmp);
      #endif

      #if defined (ELIBEXEC)
      case ELIBEXEC:
        throw ElibexecExc (tmp);
      #endif

      #if defined (EILSEQ)
      case EILSEQ:
        throw EilseqExc (tmp);
      #endif

      #if defined (ENOSYS)
      case ENOSYS:
        throw EnosysExc (tmp);
      #endif

      #if defined (ELOOP)
      case ELOOP:
        throw EloopExc (tmp);
      #endif

      #if defined (ERESTART)
      case ERESTART:
        throw ErestartExc (tmp);
      #endif

      #if defined (ESTRPIPE)
      case ESTRPIPE:
        throw EstrpipeExc (tmp);
      #endif

      #if defined (ENOTEMPTY)
      case ENOTEMPTY:
        throw EnotemptyExc (tmp);
      #endif

      #if defined (EUSERS)
      case EUSERS:
        throw EusersExc (tmp);
      #endif

      #if defined (ENOTSOCK)
      case ENOTSOCK:
        throw EnotsockExc (tmp);
      #endif

      #if defined (EDESTADDRREQ)
      case EDESTADDRREQ:
        throw EdestaddrreqExc (tmp);
      #endif

      #if defined (EMSGSIZE)
      case EMSGSIZE:
        throw EmsgsizeExc (tmp);
      #endif

      #if defined (EPROTOTYPE)
      case EPROTOTYPE:
        throw EprototypeExc (tmp);
      #endif

      #if defined (ENOPROTOOPT)
      case ENOPROTOOPT:
        throw EnoprotooptExc (tmp);
      #endif

      #if defined (EPROTONOSUPPORT)
      case EPROTONOSUPPORT:
        throw EprotonosupportExc (tmp);
      #endif

      #if defined (ESOCKTNOSUPPORT)
      case ESOCKTNOSUPPORT:
        throw EsocktnosupportExc (tmp);
      #endif

      #if defined (EOPNOTSUPP)
      case EOPNOTSUPP:
        throw EopnotsuppExc (tmp);
      #endif

      #if defined (EPFNOSUPPORT)
      case EPFNOSUPPORT:
        throw EpfnosupportExc (tmp);
      #endif

      #if defined (EAFNOSUPPORT)
      case EAFNOSUPPORT:
        throw EafnosupportExc (tmp);
      #endif

      #if defined (EADDRINUSE)
      case EADDRINUSE:
        throw EaddrinuseExc (tmp);
      #endif

      #if defined (EADDRNOTAVAIL)
      case EADDRNOTAVAIL:
        throw EaddrnotavailExc (tmp);
      #endif

      #if defined (ENETDOWN)
      case ENETDOWN:
        throw EnetdownExc (tmp);
      #endif

      #if defined (ENETUNREACH)
      case ENETUNREACH:
        throw EnetunreachExc (tmp);
      #endif

      #if defined (ENETRESET)
      case ENETRESET:
        throw EnetresetExc (tmp);
      #endif

      #if defined (ECONNABORTED)
      case ECONNABORTED:
        throw EconnabortedExc (tmp);
      #endif

      #if defined (ECONNRESET)
      case ECONNRESET:
        throw EconnresetExc (tmp);
      #endif

      #if defined (ENOBUFS)
      case ENOBUFS:
        throw EnobufsExc (tmp);
      #endif

      #if defined (EISCONN)
      case EISCONN:
        throw EisconnExc (tmp);
      #endif

      #if defined (ENOTCONN)
      case ENOTCONN:
        throw EnotconnExc (tmp);
      #endif

      #if defined (ESHUTDOWN)
      case ESHUTDOWN:
        throw EshutdownExc (tmp);
      #endif

      #if defined (ETOOMANYREFS)
      case ETOOMANYREFS:
        throw EtoomanyrefsExc (tmp);
      #endif

      #if defined (ETIMEDOUT)
      case ETIMEDOUT:
        throw EtimedoutExc (tmp);
      #endif

      #if defined (ECONNREFUSED)
      case ECONNREFUSED:
        throw EconnrefusedExc (tmp);
      #endif

      #if defined (EHOSTDOWN)
      case EHOSTDOWN:
        throw EhostdownExc (tmp);
      #endif

      #if defined (EHOSTUNREACH)
      case EHOSTUNREACH:
        throw EhostunreachExc (tmp);
      #endif

      #if defined (EALREADY)
      case EALREADY:
        throw EalreadyExc (tmp);
      #endif

      #if defined (EINPROGRESS)
      case EINPROGRESS:
        throw EinprogressExc (tmp);
      #endif

      #if defined (ESTALE)
      case ESTALE:
        throw EstaleExc (tmp);
      #endif

      #if defined (EIORESID)
      case EIORESID:
        throw EioresidExc (tmp);
      #endif

      #if defined (EUCLEAN)
      case EUCLEAN:
        throw EucleanExc (tmp);
      #endif

      #if defined (ENOTNAM)
      case ENOTNAM:
        throw EnotnamExc (tmp);
      #endif

      #if defined (ENAVAIL)
      case ENAVAIL:
        throw EnavailExc (tmp);
      #endif

      #if defined (EISNAM)
      case EISNAM:
        throw EisnamExc (tmp);
      #endif

      #if defined (EREMOTEIO)
      case EREMOTEIO:
        throw EremoteioExc (tmp);
      #endif

      #if defined (EINIT)
      case EINIT:
        throw EinitExc (tmp);
      #endif

      #if defined (EREMDEV)
      case EREMDEV:
        throw EremdevExc (tmp);
      #endif

      #if defined (ECANCELED)
      case ECANCELED:
        throw EcanceledExc (tmp);
      #endif

      #if defined (ENOLIMFILE)
      case ENOLIMFILE:
        throw EnolimfileExc (tmp);
      #endif

      #if defined (EPROCLIM)
      case EPROCLIM:
        throw EproclimExc (tmp);
      #endif

      #if defined (EDISJOINT)
      case EDISJOINT:
        throw EdisjointExc (tmp);
      #endif

      #if defined (ENOLOGIN)
      case ENOLOGIN:
        throw EnologinExc (tmp);
      #endif

      #if defined (ELOGINLIM)
      case ELOGINLIM:
        throw EloginlimExc (tmp);
      #endif

      #if defined (EGROUPLOOP)
      case EGROUPLOOP:
        throw EgrouploopExc (tmp);
      #endif

      #if defined (ENOATTACH)
      case ENOATTACH:
        throw EnoattachExc (tmp);
      #endif

      #if defined (ENOTSUP) && defined (EOPNOTSUPP)
      #if ENOTSUP != EOPNOTSUPP
          case ENOTSUP:
        throw EnotsupExc (tmp);
      #endif
      #elif defined (ENOTSUP)
      case ENOTSUP:
        throw EnotsupExc (tmp);
      #endif

      #if defined (ENOATTR)
      case ENOATTR:
        throw EnoattrExc (tmp);
      #endif

      #if defined (EDIRCORRUPTED)
      case EDIRCORRUPTED:
        throw EdircorruptedExc (tmp);
      #endif

      #if defined (EDQUOT)
      case EDQUOT:
        throw EdquotExc (tmp);
      #endif

      #if defined (ENFSREMOTE)
      case ENFSREMOTE:
        throw EnfsremoteExc (tmp);
      #endif

      #if defined (ECONTROLLER)
      case ECONTROLLER:
        throw EcontrollerExc (tmp);
      #endif

      #if defined (ENOTCONTROLLER)
      case ENOTCONTROLLER:
        throw EnotcontrollerExc (tmp);
      #endif

      #if defined (EENQUEUED)
      case EENQUEUED:
        throw EenqueuedExc (tmp);
      #endif

      #if defined (ENOTENQUEUED)
      case ENOTENQUEUED:
        throw EnotenqueuedExc (tmp);
      #endif

      #if defined (EJOINED)
      case EJOINED:
        throw EjoinedExc (tmp);
      #endif

      #if defined (ENOTJOINED)
      case ENOTJOINED:
        throw EnotjoinedExc (tmp);
      #endif

      #if defined (ENOPROC)
      case ENOPROC:
        throw EnoprocExc (tmp);
      #endif

      #if defined (EMUSTRUN)
      case EMUSTRUN:
        throw EmustrunExc (tmp);
      #endif

      #if defined (ENOTSTOPPED)
      case ENOTSTOPPED:
        throw EnotstoppedExc (tmp);
      #endif

      #if defined (ECLOCKCPU)
      case ECLOCKCPU:
        throw EclockcpuExc (tmp);
      #endif

      #if defined (EINVALSTATE)
      case EINVALSTATE:
        throw EinvalstateExc (tmp);
      #endif

      #if defined (ENOEXIST)
      case ENOEXIST:
        throw EnoexistExc (tmp);
      #endif

      #if defined (EENDOFMINOR)
      case EENDOFMINOR:
        throw EendofminorExc (tmp);
      #endif

      #if defined (EBUFSIZE)
      case EBUFSIZE:
        throw EbufsizeExc (tmp);
      #endif

      #if defined (EEMPTY)
      case EEMPTY:
        throw EemptyExc (tmp);
      #endif

      #if defined (ENOINTRGROUP)
      case ENOINTRGROUP:
        throw EnointrgroupExc (tmp);
      #endif

      #if defined (EINVALMODE)
      case EINVALMODE:
        throw EinvalmodeExc (tmp);
      #endif

      #if defined (ECANTEXTENT)
      case ECANTEXTENT:
        throw EcantextentExc (tmp);
      #endif

      #if defined (EINVALTIME)
      case EINVALTIME:
        throw EinvaltimeExc (tmp);
      #endif

      #if defined (EDESTROYED)
      case EDESTROYED:
        throw EdestroyedExc (tmp);
      #endif
    }

    throw ErrnoExc (tmp);
}


void throwErrnoExc (const std::string &text)
{
    throwErrnoExc (text, errno);
}


} // namespace Iex
