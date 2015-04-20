// sockstream.h -*- C++ -*- socket library
// Copyright (C) 2002 Herbert Straub
//
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11
//
// Version: 1.2 2002-07-25 Herbert Straub
//     Improved Error Handling - extending the sockerr class by cOperation
// 2003-03-06 Herbert Straub
//     adding sockbuf::getname und setname (sockname)
//     sockbuf methods throw method name + sockname

#ifndef _SOCKSTREAM_H
#define    _SOCKSTREAM_H

#include "config.h"

#include <iostream> // must be ANSI compatible
#include <exception> // must be ANSI compatible
//#include <cstddef>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <string>
//#include <cstdio>
#if defined(__CYGWIN__) || !defined(WIN32)
#  include <sys/types.h>
#  include <sys/uio.h>
#  include <sys/socket.h>
#  define SOCKET int
#  define SOCKET_ERROR -1
#else
#  include <windows.h>
#  include <wininet.h>
#  include <errno.h>
#ifdef _MSC_VER
#  pragma comment(lib, "Wininet")
#endif
#endif


#if defined(__linux__) || defined(__CYGWIN__)
#  define MSG_MAXIOVLEN     16
#endif // __linux__

//this class gets rid of the C4251 warning by internalizing the string.
//that way, if something else links to this library (and it should!), no linker conflicts should happen
//see http://www.unknownroad.com/rtfm/VisualStudio/warningC4251.html
class StringWrapper { 
public:
	std::string text;
};

// socket exception classes
class MY_API sockerr : public std::exception
{
    int  err;
    StringWrapper text;
    public:
        sockerr (int e, const char *theop = NULL): err (e)
        {
            if (theop != NULL)
            {
                text.text = theop;
            }
        }
        sockerr (int e, const char *theop, const char *specification) : err (e)
        {
            if (theop != NULL)
                text.text = theop;
            if (specification != NULL)
            {
                text.text += "(";
                text.text += specification;
                text.text += ")";
            }
        }
        sockerr (int e, const std::string &theoperation): err (e)
        {
            text.text = theoperation;
        }
        sockerr (const sockerr &O): std::exception(O)
        {
            err = O.err;
            text = O.text;
        }
        virtual ~sockerr() throw() {}

        const char* what () const throw() { return "sockerr"; }
        const char* operation () const { return text.text.c_str(); }

//      int errno () const { return err; }
        int serrno () const { return err; } // LN
        const char* errstr () const;
        bool error (int eno) const { return eno == err; }

        bool io () const; // non-blocking and interrupt io recoverable error.
        bool arg () const; // incorrect argument supplied. recoverable error.
        bool op () const; // operational error. recovery difficult.

        bool conn () const;   // connection error
        bool addr () const;   // address error
        bool benign () const; // recoverable read/write error like EINTR etc.
};

class sockoob
{
    public:
        const char* what () const { return "sockoob"; }
};

// socket address classes
struct sockaddr;

class sockAddr
{
    public:
        virtual ~sockAddr() {}
        virtual operator void* () const =0;
        operator sockaddr* () const { return addr (); }
        virtual int size() const =0;
        virtual int family() const =0;
        virtual sockaddr* addr      () const =0;
};

struct msghdr;

// socket buffer class
class MY_API sockbuf: public std::streambuf
{
    public:
        enum type {
            sock_stream            = SOCK_STREAM,
            sock_dgram            = SOCK_DGRAM,
            sock_raw            = SOCK_RAW,
            sock_rdm            = SOCK_RDM,
            sock_seqpacket      = SOCK_SEQPACKET
        };
        enum option {
            so_debug            = SO_DEBUG,
            so_reuseaddr    = SO_REUSEADDR,
            so_keepalive    = SO_KEEPALIVE,
            so_dontroute    = SO_DONTROUTE,
            so_broadcast    = SO_BROADCAST,
            so_linger            = SO_LINGER,
            so_oobinline    = SO_OOBINLINE,
            so_sndbuf        = SO_SNDBUF,
            so_rcvbuf        = SO_RCVBUF,
            so_error        = SO_ERROR,
            so_type        = SO_TYPE
        };
        enum level {
            sol_socket          = SOL_SOCKET
        };
        enum msgflag {
            msg_oob        = MSG_OOB,
            msg_peek            = MSG_PEEK,
            msg_dontroute    = MSG_DONTROUTE

#if !(defined(__FreeBSD__) || defined(__FreeBSD_kernel__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__APPLE__))
            ,msg_maxiovlen    = MSG_MAXIOVLEN
#endif
        };
        enum shuthow {
            shut_read,
            shut_write,
            shut_readwrite
        };
        enum { somaxconn    = SOMAXCONN };
        struct socklinger {
            int    l_onoff;    // option on/off
            int    l_linger;    // linger time

            socklinger (int a, int b): l_onoff (a), l_linger (b) {}
        };

        typedef char          char_type;
        typedef std::streampos     pos_type;
        typedef std::streamoff     off_type;
        typedef int           int_type;
        typedef int           seekdir;
        //  const int_type eof = EOF;
        enum { eof = EOF }; // LN

        struct sockdesc {
            int sock;
            sockdesc (int d): sock (d) {}
        };

    protected:
        struct sockcnt {
            SOCKET    sock;
            int            cnt;
            int            stmo; // -1==block, 0==poll, >0 == waiting time in secs
            int            rtmo; // -1==block, 0==poll, >0 == waiting time in secs
            bool        oob;    // check for out-of-band byte while reading
            void*        gend; // end of input buffer
            void*        pend; // end of output buffer

            sockcnt(SOCKET s):
                sock(s), cnt(1), stmo (-1), rtmo (-1), oob (false),
                gend (0), pend (0) {}
        };

        sockcnt* rep;  // counts the # refs to sock
        StringWrapper        sockname; // name of sockbuf - Herbert Straub

#if 0
        virtual sockbuf*      setbuf (char_type* s, int_type* n);
        virtual pos_type      seekoff (off_type off,
                                       seekdir way,
                                       ios::openmode which = ios::in|ios::out);
        virtual pos_type      seekpos (pos_type sp,
                                       ios::openmode which = ios::in|ios::out);
#endif

        virtual int           sync ();

        virtual std::streamsize    showmanyc ();
        virtual std::streamsize    xsgetn (char_type* s, std::streamsize n);
        virtual int_type      underflow ();
        virtual int_type      uflow ();

        virtual int_type      pbackfail (int_type c = eof);

        virtual std::streamsize    xsputn (const char_type* s, std::streamsize n);
        virtual int_type      overflow (int_type c = eof);

    public:
        sockbuf (const sockdesc& sd);
        sockbuf (int domain, type, int proto);
        sockbuf (const sockbuf&);
//      sockbuf&        operator = (const sockbuf&);
        virtual ~sockbuf ();

        SOCKET sd () const { return rep->sock; }
        int pubsync () { return sync (); }
        virtual bool is_open () const;

        virtual void bind    (sockAddr&);
        virtual void connect    (sockAddr&);

        void listen    (int num=somaxconn);
        virtual sockdesc accept();
        virtual sockdesc accept(sockAddr& sa);

        int read(void* buf, int len);
        int recv    (void* buf, int len, int msgf=0);
        int recvfrom(sockAddr& sa, void* buf, int len, int msgf=0);

#if    !defined(__linux__) && !defined(WIN32)
        int recvmsg(msghdr* msg, int msgf=0);
        int sendmsg(msghdr* msg, int msgf=0);
#endif

        int write(const void* buf, int len);
        int send(const void* buf, int len, int msgf=0);
        int sendto    (sockAddr& sa, const void* buf, int len, int msgf=0);

        int sendtimeout (int wp=-1);
        int recvtimeout (int wp=-1);
        int is_readready (int wp_sec, int wp_usec=0) const;
        int is_writeready (int wp_sec, int wp_usec=0) const;
        int is_exceptionpending (int wp_sec, int wp_usec=0) const;

        void shutdown (shuthow sh);

        int getopt(int op, void* buf, int len,
                   int level=sol_socket) const;
        void setopt(int op, void* buf, int len,
                    int level=sol_socket) const;

        type gettype () const;
        int  clearerror () const;
        bool debug      () const;
        bool debug      (bool set) const;
        bool reuseaddr () const;
        bool reuseaddr (bool set) const;
        bool keepalive () const;
        bool keepalive (bool set) const;
        bool dontroute () const;
        bool dontroute (bool set) const;
        bool broadcast () const;
        bool broadcast (bool set) const;
        bool oobinline () const;
        bool oobinline (bool set) const;
        bool oob       () const { return rep->oob; }
        bool oob       (bool b);
        int  sendbufsz () const;
        int  sendbufsz (int sz)   const;
        int  recvbufsz () const;
        int  recvbufsz (int sz)   const;
        socklinger linger() const;
        socklinger linger(socklinger opt) const;
        socklinger linger(int onoff, int tm) const
        { return linger (socklinger (onoff, tm)); }

        bool atmark() const;
        long nread() const;
        long howmanyc();
        void nbio(bool set=true) const;
        inline void setname(const char *name);
        inline void setname(const std::string &name);
        inline const std::string& getname();

#if defined(__CYGWIN__) || !defined(WIN32)
        void async(bool set=true) const;
#endif
#if !defined(WIN32)
        int  pgrp() const;
        int  pgrp(int new_pgrp) const;
        void closeonexec(bool set=true) const;
#endif
};

class MY_API isockstream: public std::istream
{
    protected:
        //isockstream (): istream(rdbuf()), ios (0) {}

    public:
        isockstream(sockbuf* sb): std::ios (sb) , std::istream(sb) {}
        virtual ~isockstream () {}

        sockbuf* rdbuf () { return (sockbuf*)std::ios::rdbuf(); }
        sockbuf* operator -> () { return rdbuf(); }
};

class osockstream: public std::ostream
{
    protected:
        //osockstream (): ostream(static_cast<>rdbuf()), ios (0) {}
    public:
        osockstream(sockbuf* sb): std::ios (sb) , std::ostream(sb) {}
        virtual ~osockstream () {}
        sockbuf* rdbuf () { return (sockbuf*)std::ios::rdbuf(); }
        sockbuf* operator -> () { return rdbuf(); }
};

class MY_API iosockstream: public std::iostream
{
    protected:
        iosockstream ();
    public:
        iosockstream(sockbuf* sb): std::ios(sb), std::iostream(sb) {}
        virtual ~iosockstream () {}

        sockbuf* rdbuf () { return (sockbuf*)std::ios::rdbuf(); }
        sockbuf* operator -> () { return rdbuf(); }
};

// manipulators
extern osockstream& crlf (osockstream&);
extern osockstream& lfcr (osockstream&);

// inline

void sockbuf::setname (const char *name)
{
    sockname.text = name;
}

void sockbuf::setname (const std::string &name)
{
    sockname.text = name;
}

const std::string& sockbuf::getname ()
{
    return sockname.text;
}

#endif    // _SOCKSTREAM_H
