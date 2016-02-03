// sockunix.cpp  -*- C++ -*- socket library
// Copyright (C) 2002 Herbert Straub
//
// sockunix.C -*- C++ -*- socket library
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
//  and in the second step call ios::init (sockinetbuf *) and iosockstream::init ...
//  The constructors of isockunix, osockunix and iosockunix are changed.

//#include <windows.h>
//using namespace std;

#include <sockunix.h>
#include <sys/socket.h>
#include <string.h>

sockunixaddr::sockunixaddr (const char* path)
{
  sun_family = sockunixbuf::af_unix;
  ::strcpy (sun_path, path);
}

sockunixaddr::sockunixaddr (const sockunixaddr& suna)
{
  sun_family = sockunixbuf::af_unix;
  ::strcpy (sun_path, suna.sun_path);
}

sockunixbuf::sockunixbuf (const sockbuf::sockdesc& sd)
  : sockbuf (sd.sock)
{}

sockunixbuf::sockunixbuf (const sockunixbuf& su)
  : sockbuf (su)
{}

sockunixbuf::sockunixbuf (sockbuf::type ty, int proto)
  : sockbuf (af_unix, ty, proto)
{}

/*sockunixbuf& sockunixbuf::operator = (const sockunixbuf& su)
{
  sockbuf::operator = (su);
  return *this;
}*/

void sockunixbuf::bind (sockAddr& sa)
{
  sockbuf::bind (sa);
}

void sockunixbuf::bind (const char* path)
{
  sockunixaddr sa (path);
  bind (sa);
}

void sockunixbuf::connect (sockAddr& sa)
{
  sockbuf::connect (sa);
}

void sockunixbuf::connect (const char* path)
{
  sockunixaddr sa (path);
  connect (sa);
}

isockunix::isockunix (const sockbuf::sockdesc& sd)
  : ios (0), isockstream(0)
{
	sockunixbuf *t = new sockunixbuf (sd);

	ios::init (t);
	isockstream::init (t);
}

isockunix::isockunix (sockbuf::type ty, int proto)
  : ios (0), isockstream(0)
{
	sockunixbuf *t = new sockunixbuf (ty, proto);

	ios::init (t);
	isockstream::init (t);
}

isockunix::isockunix (const sockunixbuf& sb)
  : ios (0), isockstream(0)
{
	sockunixbuf *t = new sockunixbuf (sb);

	ios::init (t);
	isockstream::init (t);
}

isockunix::~isockunix ()
{
  delete ios::rdbuf ();
}

osockunix::osockunix (const sockbuf::sockdesc& sd)
  : ios (0), osockstream(0)
{
	sockunixbuf *t = new sockunixbuf (sd);

	ios::init (t);
	osockstream::init (t);
}

osockunix::osockunix (sockbuf::type ty, int proto)
  : ios (0), osockstream(0)
{
	sockunixbuf *t = new sockunixbuf (ty, proto);

	ios::init (t);
	osockstream::init (t);
}

osockunix::osockunix (const sockunixbuf& sb)
  : ios (0), osockstream(0)
{
	sockunixbuf *t= new sockunixbuf (sb);

	ios::init (t);
	osockstream::init (t);
}

osockunix::~osockunix ()
{
  delete ios::rdbuf ();
}

iosockunix::iosockunix (const sockbuf::sockdesc& sd)
  : ios (0), iosockstream(0)
{
	sockunixbuf *t = new sockunixbuf (sd);

	ios::init (t);
	iosockstream::init (t);
}

iosockunix::iosockunix (sockbuf::type ty, int proto)
  : ios (0), iosockstream(0)
{
	sockunixbuf *t = new sockunixbuf (ty, proto);

	ios::init (t);
	iosockstream::init (t);
}

iosockunix::iosockunix (const sockunixbuf& sb)
  : ios (0), iosockstream(0)
{
	sockunixbuf *t = new sockunixbuf (sb);

	ios::init (t);
	iosockstream::init (t);
}

iosockunix::~iosockunix ()
{
  delete ios::rdbuf ();
}
