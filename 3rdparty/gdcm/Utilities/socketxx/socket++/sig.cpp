// socket++ library. sig.C
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.

#include <signal.h>
#include <sig.h>

//explicit template instantiation.
typedef sig::phnd phnd;
typedef sig::phndlist phndlist;
//template class list<phnd>;
//template class map<int, phndlist, less<int> >;

//static sigerr se; //commended out by Herbert Straub
// Change all se to sigerr

siginit siginit::init;
sig& sig::nal = *siginit::init.s;

typedef void (*sighnd_type) (int);

extern "C" {
  static void sighandler (int signo) {
    sig::nal.kill (signo);
  }
}

sig::hnd::hnd (int s)
  : signo (s)
{
  sig::nal.set (signo, this);
}

sig::hnd::~hnd ()
{
  sig::nal.unset (signo, this);
}

bool sig::set (int signo, sig::hnd* hnd)
{
  if (hnd == 0) return false;

  phndlist& v = smap [signo];

  if (v.empty ()) {
#ifndef WIN32
    struct sigaction sa;
    if (sigaction (signo, 0, &sa) == -1) throw sigerr();
    if (sa.sa_handler != sighnd_type (&sighandler)) {
      // setting for the first time
      sa.sa_handler = (void(*)(int)) sighnd_type (&sighandler);
      if (sigemptyset (&sa.sa_mask) == -1) throw sigerr();
      sa.sa_flags = 0;
      if (sigaction (signo, &sa, 0) == -1) throw sigerr();
    }
#endif //windows does not define sigaction
    //basically, this is a way to handle some kind of error that can't exist on windows.
    //see http://svn.haxx.se/dev/archive-2004-01/0685.shtml
    v.push_back (hnd);
    return true;
  }

  phndlist::iterator j = find (v.begin(), v.end (), hnd);
  if (j == v.end ()) {
    v.push_back (hnd);
    return true;
  }
  return false;
}

bool sig::unset (int signo, sig::hnd* hnd)
{
  if (hnd == 0) return false;

  phndlist& v = smap [signo];

  phndlist::iterator j = find (v.begin(), v.end (), hnd);
  if (j != v.end ()) {
    v.erase (j);
    return true;
  }

  return false;
}

void sig::unset (int signo)
{
  phndlist& v = smap [signo];
  v.erase (v.begin (), v.end ());
#ifndef WIN32
  struct sigaction sa;
  if (sigaction (signo, 0, &sa) == -1) throw sigerr();
  if (sa.sa_handler == sighnd_type (&sighandler)) {
    sa.sa_handler = (void(*)(int)) sighnd_type (SIG_DFL);
    if (sigemptyset (&sa.sa_mask) == -1) throw sigerr();
    sa.sa_flags = 0;
    if (sigaction (signo, &sa, 0) == -1) throw sigerr();
  }
#endif //windows does not define sigaction
}

void sig::mask (int signo) const
{
#ifndef WIN32
  sigset_t s;
  if (sigemptyset (&s) == -1) throw sigerr();
  if (sigaddset (&s, signo) == -1) throw sigerr();

  if (sigprocmask (SIG_BLOCK, &s, 0) == -1) throw sigerr();
#endif //windows does not define sigset_t
}

void sig::unmask (int signo) const
{
#ifndef WIN32
  sigset_t s;
  if (sigemptyset (&s) == -1) throw sigerr();
  if (sigaddset (&s, signo) == -1) throw sigerr();

  if (sigprocmask (SIG_UNBLOCK, &s, 0) == -1) throw sigerr();
#endif //windows does not define sigset_t
}

void sig::mask (int siga, int sigb) const
{
#ifndef WIN32
  struct sigaction sa;
  if (sigaction (siga, 0, &sa) == -1) throw sigerr();
  if (sa.sa_handler != sighnd_type (&sighandler)) {
    sa.sa_handler = (void(*)(int)) sighnd_type (&sighandler);
    if (sigemptyset (&sa.sa_mask) == -1) throw sigerr();
    sa.sa_flags = 0;
  }
  if (sigaddset (&sa.sa_mask, sigb) == -1) throw sigerr();
  if (sigaction (siga, &sa, 0) == -1) throw sigerr();
#endif //windows does not define sigaction
}

void sig::unmask (int siga, int sigb) const
{
#ifndef WIN32
  struct sigaction sa;
  if (sigaction (siga, 0, &sa) == -1) throw sigerr();
  if (sa.sa_handler != sighnd_type (&sighandler)) {
    sa.sa_handler = (void(*)(int)) sighnd_type (&sighandler);
    if (sigemptyset (&sa.sa_mask) == -1) throw sigerr();
    sa.sa_flags = 0;
  } else {
    if (sigdelset (&sa.sa_mask, sigb) == -1) throw sigerr();
  }
  if (sigaction (siga, &sa, 0) == -1) throw sigerr();
#endif //windows does not define sigaction
}

void sig::sysresume (int signo, bool set) const
{
#ifndef WIN32
  struct sigaction sa;
  if (sigaction (signo, 0, &sa) == -1) throw sigerr();
  if (sa.sa_handler != sighnd_type (&sighandler)) {
    sa.sa_handler = (void(*)(int)) sighnd_type (&sighandler);
    if (sigemptyset (&sa.sa_mask) == -1) throw sigerr();
    sa.sa_flags = 0;
  }

#if !(defined(__FreeBSD__) || defined(__NetBSD__) || defined(__OpenBSD__) || defined(__bsdi__) || defined(__sun__) || defined(__linux__) || defined(__APPLE__))
// Early SunOS versions may have SA_INTERRUPT. I can't confirm.
  if (set == false)
    sa.sa_flags |= SA_INTERRUPT;
  else
    sa.sa_flags &= ~SA_INTERRUPT;
  if (sigaction (signo, &sa, 0) == -1) throw sigerr();
#endif
#endif //windows does not define sigaction

}

struct procsig {
  int signo;
  procsig (int s): signo (s) {}
  void operator () (phnd& ph) { (*ph) (signo); }
};

void sig::kill (int signo)
{
  phndlist& v = smap [signo];

  // struct procsig used to be here // LN

  for_each (v.begin (), v.end (), procsig (signo));
}

sigset_t sig::pending () const
{
#ifndef WIN32
  sigset_t s;
  if (sigemptyset (&s) == -1) throw sigerr();
  if (sigpending (&s) == -1) throw sigerr();
  return s;
#else
  return true;//is this the right behavior for windows?
#endif //windows does not define sigset_t
}

bool sig::ispending (int signo) const
{
#ifndef WIN32
  sigset_t s = pending ();
  switch (sigismember (&s, signo)) {
  case 0: return false;
  case 1: return true;
  }
  throw sigerr();
#else
  return true;//is this the right behavior for windows?
#endif //windows does not define sigset_t
}
