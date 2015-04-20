// pipestream.h -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11

#ifndef _PIPESTREAM_H
#define	_PIPESTREAM_H

#include <socket++/sockstream.h>

class ipipestream: public isockstream {
protected:
//                  ipipestream (): std::ios (0) {}
public:
                  ipipestream (const char* cmd);
                  ~ipipestream () { delete std::ios::rdbuf (); }
};

class opipestream: public osockstream {
protected:
//                  opipestream (): std::ios(0) {}
public:
                  opipestream (const char* cmd);
                  ~opipestream () { delete std::ios::rdbuf (); }
};

class iopipestream: public iosockstream {
private:
  iopipestream(const iopipestream& sp); // no defintion provided
  iopipestream&	operator = (iopipestream&); // no definition provided

protected:
  int		sp[2]; // socket pair

  // if iopipstream (sockbuf::type, int) created this object,
  // then cpid is significant. Otherwise it is set to -1.
  //   cpid is child pid if this is parent
  //   cpid is 0 if this is child
#ifdef _WIN32
  typedef int pid_t;
#endif
  pid_t	        cpid;
  iopipestream* next;  // next in the chain. Used only by
                       // iopipstream (sockbuf::type, int)

  static iopipestream* head; // list to take care of by fork()

public:
  iopipestream(sockbuf::type ty=sockbuf::sock_stream, int proto=0);
  iopipestream(const char* cmd);
  ~iopipestream () { delete std::ios::rdbuf (); }

  pid_t        pid () const { return cpid; } // returns cpid
#ifndef WIN32
  static pid_t fork(); // sets cpid of all iopipestream* in the head
#endif
};

#endif	// _PIPESTREAM_H
