// Fork.h -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11

#ifndef FORK_H
#define FORK_H

#ifndef WIN32
#include <unistd.h>
#include <sys/types.h>
#include <sys/signal.h>

class Fork {
 public:
  class KillForks {
  public:
    KillForks () {}
    ~KillForks ();
  };

  class ForkProcess {
    friend Fork::KillForks::~KillForks ();

    static void infanticide_reason (pid_t pid, int status);
    static void reaper_nohang (int);

    static ForkProcess* list;

  public:
    pid_t          pid;
    const bool     kill_child;
    const bool     reason;
    ForkProcess*   next;

    ForkProcess (bool kill, bool give_reason);
    ~ForkProcess ();

    void           kill_process () const;
    void           reap_child () const;

    static void    commit_suicide (int);
  };
 private:
  static KillForks killall;

  ForkProcess* process;

  Fork (Fork&);             // no copy constructor definition provided
  Fork& operator = (Fork&); // no assignment operator definition provided

 public:
  Fork (bool kill = 0, bool reason = 0)
    : process (new ForkProcess (kill, reason)) {}
  ~Fork ();

  int  is_child () const { return process->pid == 0; }
  int  is_parent () const { return process->pid > 0; }
  int  process_id () const { return process->pid; }

  static void suicide_signal (int signo = SIGTERM);
};

#endif//windows does not define fork, and never will.
#endif // FORK_H
