// protocol.h -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11

#include <config.h>
#include <protocol.h>

#ifdef WIN32
#ifndef EPROTONOSUPPORT
#	define EPROTONOSUPPORT				WSAEPROTONOSUPPORT
#endif
#endif

const char* protocol::protocolbuf::protocol_name () const
{
  const char* ret = "";
  if (pn == protocol::tcp)
    ret = "tcp";
  if (pn == protocol::udp)
    ret = "udp";
  return ret;
}

void protocol::protocolbuf::connect ()
{
  if (pn == protocol::nil) throw sockerr (EPROTONOSUPPORT);
  sockinetbuf::connect (localhost (), rfc_name (), protocol_name ());
}

void protocol::protocolbuf::connect (unsigned long addr)
     // addr is in host byte order
{
  if (pn == protocol::nil) throw sockerr (EPROTONOSUPPORT);
  sockinetbuf::connect (addr, rfc_name (), protocol_name ());
}

void protocol::protocolbuf::connect (const char* host)
{
  if (pn == protocol::nil) throw sockerr (EPROTONOSUPPORT);
  sockinetbuf::connect (host, rfc_name (), protocol_name ());
}

void protocol::protocolbuf::connect (const char* host, int portno)
{
  if (pn == protocol::nil) throw sockerr (EPROTONOSUPPORT);
  sockinetbuf::connect (host, portno);
}
