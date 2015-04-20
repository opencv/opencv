// pipestream.cpp  -*- C++ -*- socket library
// Copyright (C) 2002 Herbert Straub
//
// pipestream.C -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11
// 2002-07-28 Version 1.2 (C) Herbert Straub
//  Eliminating sorry_about_global_temp inititialisation. This don't work
//  in combination with NewsCache. My idea is: initializing the classes with (0)
//  and in the second step call ios::init (sockbuf *) and iosockstream::init ...
//  The constructors of ipipestream, opipestream and iopipestream are changed.


#include <config.h>

#include <pipestream.h>
#include <iostream> // ios
using namespace std;

#ifdef _WIN32
# include <wininet.h>
# include <windows.h>
# include <io.h>
#else
# include <unistd.h>
# include <sys/socket.h>
# include <sys/types.h>
# include <sys/socket.h>
#endif

// environ is not given a declaration in sun's <unistd.h>
#ifndef __APPLE__
extern char** environ;
#else
#include <crt_externs.h>
#endif //__APPLE__

// child closes s2 and uses s1
// parent closes s1 and uses s2

enum domain { af_unix = 1 };

iopipestream* iopipestream::head = 0;

static sockbuf* createpipestream (const char* cmd, int mode)
{
  int sockets[2];
#ifndef WIN32
  //FIXME!!! this code needs to work
  if (::socketpair (af_unix, sockbuf::sock_stream, 0, sockets) == -1)
    throw sockerr (errno);

  pid_t pid = ::vfork ();
  if (pid == -1) throw sockerr (errno);

  if (pid == 0) {
    // child process
    if (::close (sockets[1]) == -1) throw sockerr (errno);

    if ((mode & ios::in) && ::dup2 (sockets[0], 1) == -1)
      throw sockerr (errno);
    if ((mode & ios::out) && ::dup2 (sockets[0], 0) == -1)
      throw sockerr (errno);
    if (::close (sockets[0]) == -1) throw sockerr (errno);

    const char*	argv[4];
    argv[0] = "/bin/sh";
    argv[1] = "-c";
    argv[2] = cmd;
    argv[3] = 0;
#ifndef __APPLE__
    execve ("/bin/sh", (char**) argv, environ);
#else
    execve("/bin/sh", (char**)argv, *_NSGetEnviron());
#endif
    throw sockerr (errno);
  }
#endif //wow, none of that above code works for windows at all
  // parent process
  if (::close (sockets[0]) == -1) throw sockerr (errno);

  sockbuf* s = new sockbuf (sockbuf::sockdesc(sockets[1]));
  if (!(mode & ios::out)) s->shutdown (sockbuf::shut_write);
  if (!(mode & ios::in)) s->shutdown (sockbuf::shut_read);
  return s;
}

ipipestream::ipipestream (const char* cmd)
  : ios (0), isockstream(0)
{
	sockbuf *t = createpipestream (cmd, ios::in);

	ios::init (t);
	isockstream::init (t);
}

opipestream::opipestream (const char* cmd)
  : ios (0), osockstream(0)
{
	sockbuf *t = createpipestream (cmd, ios::out);

	ios::init (t);
	osockstream::init (t);
}

iopipestream::iopipestream (const char* cmd)
  : ios (0), iosockstream(0),
    cpid (-1), next (0)
{
	sockbuf *t = createpipestream (cmd, ios::in|ios::out);

	ios::init (t);
	iosockstream::init (t);
}

iopipestream::iopipestream(sockbuf::type ty, int proto)
  : ios (0), iosockstream(NULL), cpid (-1), next (head)  // probably NULL is not a good idea //LN
{
#ifndef WIN32
  if (::socketpair(af_unix, ty, proto, sp) == -1)
    throw sockerr (errno);
  head = this;	
#endif
}

#ifndef WIN32
pid_t iopipestream::fork ()
{
  pid_t pid = ::fork (); // donot use vfork here
  if (pid == -1) throw sockerr (errno);

  if (pid > 0) {
    // parent process
    while (head) {
      if (::close (head->sp[1]) == -1) throw sockerr (errno);
      head->cpid = pid;
      head->init (new sockbuf (sockbuf::sockdesc(head->sp[0])));
      head = head->next;
    }
  } else {
    // child process
    while (head) {
      if (::close (head->sp[0]) == -1) throw sockerr (errno);
      head->cpid = 0;
      head->init (new sockbuf (sockbuf::sockdesc(head->sp[1])));
      head = head->next;
    }
  }
  return pid;
}	
#endif
