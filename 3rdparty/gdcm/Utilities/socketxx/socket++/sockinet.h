// sockinet.h -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11

#ifndef _SOCKINET_H
#define _SOCKINET_H

#include "config.h"
#include "sockstream.h"

#if defined(__CYGWIN__) || !defined(WIN32)
#  include <netinet/in.h>
#endif // !WIN32

class sockinetaddr: public sockAddr, public sockaddr_in
{
    protected:
        void setport (const char* sn, const char* pn="tcp");
        void setaddr (const char* hn);

    public:
        ~sockinetaddr () {}
        sockinetaddr ();
        sockinetaddr (unsigned long addr, int port_no=0);
        sockinetaddr (const char* host_name, int port_no=0);
        sockinetaddr (unsigned long addr,
                      const char* service_name,
                      const char* protocol_name="tcp");
        sockinetaddr (const char* host_name,
                      const char* service_name,
                      const char* protocol_name="tcp");
        sockinetaddr (const sockinetaddr& sina);

        operator void* () const { return addr_in (); }

        sockaddr_in*        addr_in () const { return (sockaddr_in*) this; }
        int                 size  () const { return sizeof (sockaddr_in); }
        int                 family() const { return sin_family; }
        sockaddr*           addr  () const { return (sockaddr*) addr_in (); }

        int                 getport    () const;
        const char*         gethostname() const;
};

class MY_API sockinetbuf: public sockbuf
{
    public:
        enum domain { af_inet = AF_INET };

        sockinetbuf (const sockbuf::sockdesc& sd);
        sockinetbuf (const sockinetbuf& si): sockbuf (si) {}
        sockinetbuf (sockbuf::type ty, int proto=0);
        //sockinetbuf& operator=(const sockinetbuf& si);
        ~sockinetbuf () {}

        sockinetaddr        localaddr() const;
        int                 localport() const;
        const char*         localhost() const;

        sockinetaddr        peeraddr() const;
        int                 peerport() const;
        const char*         peerhost() const;

        void                bind_until_success (int portno);

        virtual void        bind (sockAddr& sa);
        void                bind (int port_no=0); // addr is assumed to be INADDR_ANY
                                                  // and thus defaults to local host

        void                bind (unsigned long addr, int port_no);
        void                bind (const char* host_name, int port_no=0);
        void                bind (unsigned long addr,
                                  const char* service_name,
                                  const char* protocol_name="tcp");
        void                bind (const char* host_name,
                                  const char* service_name,
                                  const char* protocol_name="tcp");

        virtual void        connect (sockAddr& sa);
        void                connect (unsigned long addr, int port_no);
        void                connect (const char* host_name, int port_no);
        void                connect (unsigned long addr,
                                     const char* service_name,
                                     const char* protocol_name="tcp");
        void                connect (const char* host_name,
                                     const char* service_name,
                                     const char* protocol_name="tcp");

        virtual sockdesc    accept ();
        virtual sockdesc    accept (sockAddr& sa);
        sockdesc            accept (unsigned long addr, int port_no);
        sockdesc            accept (const char* host_name, int port_no);

        bool                tcpnodelay () const;
        bool                tcpnodelay (bool set) const;
};

class MY_API isockinet: public isockstream
{
    public:
        isockinet (const sockbuf::sockdesc& sd);
        isockinet (const sockinetbuf& sb);
        isockinet (sockbuf::type ty=sockbuf::sock_stream, int proto=0);
        ~isockinet ();

        sockinetbuf* rdbuf () { return (sockinetbuf*)std::ios::rdbuf (); }
        sockinetbuf* operator -> () { return rdbuf (); }
};

class osockinet: public osockstream
{
    public:
        osockinet (const sockbuf::sockdesc& sd);
        osockinet (const sockinetbuf& sb);
        osockinet (sockbuf::type ty=sockbuf::sock_stream, int proto=0);
        ~osockinet ();

        sockinetbuf* rdbuf () { return (sockinetbuf*)std::ios::rdbuf (); }
};

class MY_API iosockinet: public iosockstream
{
    public:
        iosockinet (const sockbuf::sockdesc& sd);
        iosockinet (const sockinetbuf& sb);
        iosockinet (sockbuf::type ty=sockbuf::sock_stream, int proto=0);
        ~iosockinet ();

        sockinetbuf* rdbuf () { return (sockinetbuf*)std::ios::rdbuf (); }
};

#endif    // _SOCKINET_H
