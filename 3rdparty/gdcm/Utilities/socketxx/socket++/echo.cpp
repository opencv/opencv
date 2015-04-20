// echo.C -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11


#include <config.h>

#include <socket++/echo.h>
#ifndef WIN32
#include <socket++/fork.h>
#endif
#include <stdlib.h>

using namespace std;

void echo::echobuf::serve_clients (int portno)
{
  if (protocol_name ()) {
    if (portno < 0)
      sockinetbuf::bind ((unsigned long) INADDR_ANY, "echo", protocol_name ());
    else if (portno <= 1024) {
      sockinetbuf::bind ();
    } else
      sockinetbuf::bind ((unsigned long) INADDR_ANY, portno);

    // act as a server now
    listen (sockbuf::somaxconn);

    // commit suicide when we receive SIGTERM
    //but only if you're not on windows, which doesn't have forking
#ifndef WIN32
    Fork::suicide_signal (SIGTERM);
#endif

    for (;;) {
      sockbuf s = accept ();
      //!!! FIXME this is most definitely broken for windows.
#ifndef WIN32
        Fork f (1, 1); // kill my children when I get terminated.
	if (f.is_child ()) {
#else //win32 has no forking
      {//if win32, no forking, in the main process
#endif
	  char buf [1024];
	  int  rcnt;

	  while ((rcnt = s.read (buf, 1024)) > 0)
	    while (rcnt != 0) {
	      int wcnt = s.write (buf, rcnt);
	      if (wcnt == -1) throw sockerr (errno);
	      rcnt -= wcnt;
	    }
#ifndef WIN32
	  sleep (300);
	  exit (0);
#endif
	}
    }
  }
}

