// socket++ library. sig.h
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.

#ifndef SIG_H
#define SIG_H

#include <list>
#include <map>
#include <algorithm>
#ifndef WIN32
#include <sys/signal.h>

#else
#include <windows.h>
#include <wininet.h>
#endif

//from http://bugs.scribus.net/view.php?id=2213
#ifndef sigset_t
  #define sigset_t int
#endif
#ifndef sigemptyset
  #define sigemptyset(sig)
#endif
#ifndef sigaddset
  #define sigaddset( set, sig)
#endif
#ifndef sigprocmask
  #define sigprocmask(a, b, c)
#endif

// all signal handlers must be derived from
// class sig::hnd. class signal will
// maintain a list of pointers to sig::hnd
// objects for a signal. If a signal occurs,
// all sig::hnds associated with the
// signal are invoked.

// sig::hnd object will insert itself into
// the signal handler list for a signo. Its
// dtor will delete the signal handler object
// from the signal handler list for a signo.
// Thus if a user wants to add a signal handler,
// all that needs to be done is to simply
// instantiate the signal handler object,
// and if the user wants to remove the signal
// handler, all that needs to done is to
// delete the object and its dtor will remove
// itself from the signal handler list.

// Note: you cannot mix sig with other means
//       of setting signal handlers.

class sig;
class siginit;
class sigerr {};

class sig {
public:
  friend class siginit;

  class hnd {
    int signo;
  public:
    hnd (int signo);
    virtual ~hnd ();
    virtual void operator () (int s) = 0;
  };

  typedef hnd* phnd;
  typedef std::list<phnd> phndlist;
  typedef std::map<int, phndlist, std::less<int> > sigmap;
private:
  sigmap smap;

  sig () {}
  ~sig () {}
public:

  // add signal handler h for signal signo
  // return true on success. false otherwise.
  bool set (int signo, hnd* h);

  // remove signal handler h for signal signo
  // return true on success. false otherwise.
  // Note: the user needs to delete h.
  bool unset (int signo, hnd* h);

  // remove all signal handers for signal signo
  void unset (int signo);

  // mask signal signo. Prevent signo from being seen
  // by our process. Note: not all signals can be
  // blocked.
  void mask (int signo) const;

  // block signal signo_b when inside a signo_a handler
  // Note: the process will see signo_b once signo_a handler
  // is finished
  void mask (int signo_a, int signo_b) const;

  // unmask signal signo. Enable signo to be seen by
  // our process.
  void unmask (int signo) const;

  // unblock signal signo_b when inside a signo_a handler
  void unmask (int signo_a, int signo_b) const;

  // is signal signo pending?
  bool ispending (int signo) const;

  // set of signals pending
  sigset_t pending () const;

  // make some system calls to terminate after they are
  // interrupted (set == false). Otherwise resume system
  // call (set == true). Not available on all systems.
  void sysresume (int signo, bool set) const;

  // process a software signal signo
  void kill (int signo);

  static sig& nal; // sig::nal is the only object of class sig
};

class siginit {
  friend class sig;
  static siginit init;
  sig* s;
public:
  siginit (): s (0) { if (this == &init) s = new sig; }
  ~siginit () { if (this == &init) delete s; }
};

#endif // SIG_H
