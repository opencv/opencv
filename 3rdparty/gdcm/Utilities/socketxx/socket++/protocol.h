// protocol.h -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11

#ifndef PROTOCOL_H
#define PROTOCOL_H

#include <socket++/sockinet.h>

class MY_API protocol: public iosockstream {
public:
  enum p_name {
    nil = 0,
    tcp = sockbuf::sock_stream,
    udp = sockbuf::sock_dgram
  };

  class MY_API protocolbuf: public sockinetbuf {
  private:
    p_name pn;

    void bind (sockAddr& sa) { sockbuf::bind (sa); }
    void connect (sockAddr& sa) { sockbuf::connect (sa); }

  public:
    protocolbuf (sockinetbuf& si): sockinetbuf (si), pn (protocol::nil) {}
    protocolbuf (p_name pname)
      : sockinetbuf ((sockbuf::type) pname, 0), pn (pname) {}


    void                bind () { serve_clients (); }
    void                connect ();
    void                connect (unsigned long addr);
    void                connect (const char* host);
    void                connect (const char* host, int portno);

    const char*         protocol_name () const;

    virtual void        serve_clients (int portno = -1) = 0;
    virtual const char* rfc_name () const = 0;
    virtual const char* rfc_doc  () const = 0;
  };

  protocol (): std::ios (0), iosockstream(NULL) {}  // NULL seems like a very bad idea
};

#endif // PROTOCOL_H
