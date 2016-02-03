// Fork.C -*- C++ -*- socket library
// Copyright (C) 1992-1996 Gnanasekaran Swaminathan <gs4t@virginia.edu>
//
// Permission is granted to use at your own risk and distribute this software
// in source and  binary forms provided  the above copyright notice and  this
// paragraph are  preserved on all copies.  This software is provided "as is"
// with no express or implied warranty.
//
// Version: 12Jan97 1.11

#ifndef WIN32

#include <config.h>

#include <iostream>
#include <stdio.h> // perror in solaris2.3 is declared here
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <signal.h>
#include <sys/wait.h>

#include <socket++/fork.h>

using std::cerr;
using std::endl;

Fork::ForkProcess* Fork::ForkProcess::list = 0;
Fork::KillForks Fork::killall;

Fork::~Fork ()
{
  if (process->pid <= 0)
    delete process;
}

Fork::KillForks::~KillForks ()
  // First, kill all children whose kill_child flag is set.
  // Second, wait for other children to die.
{
  for (ForkProcess* cur = Fork::ForkProcess::list; cur; cur = cur->next)
    if (cur->kill_child)
      delete cur;

  while (Fork::ForkProcess::list && wait (0) > 0) {}
}

Fork::ForkProcess::ForkProcess (bool kill, bool give_reason)
  : kill_child (kill), reason (give_reason), next (0)
{
  if (list == 0) {
    struct sigaction sa;
    sa.sa_handler = (void(*)(int)) sighnd (&Fork::ForkProcess::reaper_nohang);
    sigemptyset (&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction (SIGCHLD, &sa, 0);
  }

  pid = fork ();

  if (pid > 0) {
    next = list;
    list = this;
  } else if (pid == 0) {
    // child process. clear list
    ForkProcess* p = list;
    while (p) {
      ForkProcess* nxt = p->next;
      p->pid = 0;
      delete p;
      p = nxt;
    }
    list = 0;

    if (kill_child) {
      struct sigaction sa;
      sa.sa_handler = (void(*)(int)) sighnd (&Fork::ForkProcess::commit_suicide);
      sigemptyset (&sa.sa_mask);
      sa.sa_flags = SA_RESTART;
      sigaction (SIGTERM, &sa, 0);
    }
  }
}

Fork::ForkProcess::~ForkProcess ()
{
  if (pid > 0) {
    if (kill_child)
      kill (pid, SIGTERM);
    reap_child ();

    // I remove myself from list
    if (list == this)
      list = list->next;
    else {
      for (ForkProcess* p = list; p; p = p->next)
	if (p->next == this) {
	  p->next = next;
	  break;
	}
    }
  }
}

void Fork::ForkProcess::kill_process () const
{
  if (pid > 0) {
    kill (pid, SIGKILL);
    reap_child ();
  }
}

void Fork::ForkProcess::reap_child () const
{
  int status;
  if (pid > 0 && waitpid (pid, &status, 0) == pid && reason)
    infanticide_reason (pid, status);
}

void Fork::ForkProcess::infanticide_reason (pid_t pid, int status)
{
  if (pid <= 0)
    return;

#ifdef SOCKETXX_HAVE_STRSIGNAL
  if (WIFSTOPPED (status))
    cerr << "process " << pid << " gets "
      << strsignal(WSTOPSIG (status)) << endl;
  else if (WIFEXITED (status))
    cerr << "process " << pid << " exited with status "
      << WEXITSTATUS (status) << endl;
  else if (WIFSIGNALED (status))
    cerr << "process " << pid << " got "
      << strsignal(WTERMSIG (status)) << endl;
#else
  if (WIFSTOPPED (status))
    cerr << "process " << pid << " gets "
      << SYS_SIGLIST [WSTOPSIG (status)] << endl;
  else if (WIFEXITED (status))
    cerr << "process " << pid << " exited with status "
      << WEXITSTATUS (status) << endl;
  else if (WIFSIGNALED (status))
    cerr << "process " << pid << " got "
      << SYS_SIGLIST [WTERMSIG (status)] << endl;
#endif // SOCKETXX_HAVE_STRSIGNAL
}

void Fork::ForkProcess::reaper_nohang (int signo)
{
  if (signo != SIGCHLD)
    return;

  int status;
  pid_t wpid;
  if ((wpid = waitpid (-1, &status, WNOHANG)) > 0) {
    ForkProcess* prev = 0;
    ForkProcess* cur  = list;
    while (cur) {
      if (cur->pid == wpid) {
	cur->pid = -1;
	if (prev)
	  prev->next = cur->next;
	else
	  list = list->next;
	
	if (cur->reason)
	  infanticide_reason (wpid, status);

	delete cur;
	break;
      }
      prev = cur;
      cur  = cur->next;
    }
  }
}

void Fork::ForkProcess::commit_suicide (int)
{
  // if this process has any children we kill them.

  ForkProcess* p = list;
  while (p) {
    ForkProcess* next = p->next;
    if (!p->kill_child) // otherwise ForkProcess::~ForkProcess will take care
      kill (p->pid, SIGKILL);
    delete p; // ForkProcess::~ForkProcess will call reap_child ().
    p = next;
  }

  exit (0x0f);
}

void Fork::suicide_signal (int signo)
     // commit suicide at the signal signo
{
  struct sigaction sa;
  sa.sa_handler = (void(*)(int)) sighnd (&Fork::ForkProcess::commit_suicide);
  sigemptyset (&sa.sa_mask);
  sa.sa_flags = 0;
  if (sigaction (signo, &sa, 0) == -1)
    perror ("Fork: Cannot commit suicide with the specified signal");
}
#endif //windows does not get fork
