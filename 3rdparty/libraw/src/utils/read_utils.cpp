/* -*- C++ -*-
 * Copyright 2019-2021 LibRaw LLC (info@libraw.org)
 *
 LibRaw uses code from dcraw.c -- Dave Coffin's raw photo decoder,
 dcraw.c is copyright 1997-2018 by Dave Coffin, dcoffin a cybercom o net.
 LibRaw do not use RESTRICTED code from dcraw.c

 LibRaw is free software; you can redistribute it and/or modify
 it under the terms of the one of two licenses as you choose:

1. GNU LESSER GENERAL PUBLIC LICENSE version 2.1
   (See file LICENSE.LGPL provided in LibRaw distribution archive for details).

2. COMMON DEVELOPMENT AND DISTRIBUTION LICENSE (CDDL) Version 1.0
   (See file LICENSE.CDDL provided in LibRaw distribution archive for details).

 */

#include "../../internal/dcraw_defs.h"

ushort LibRaw::sget2Rev(uchar *s) // specific to some Canon Makernotes fields,
                                  // where they have endian in reverse
{
  if (order == 0x4d4d) /* "II" means little-endian, and we reverse to "MM" - big
                          endian */
    return s[0] | s[1] << 8;
  else /* "MM" means big-endian... */
    return s[0] << 8 | s[1];
}

ushort libraw_sget2_static(short _order, uchar *s)
{
    if (_order == 0x4949) /* "II" means little-endian */
        return s[0] | s[1] << 8;
    else /* "MM" means big-endian */
        return s[0] << 8 | s[1];
}

ushort LibRaw::sget2(uchar *s)
{
    return libraw_sget2_static(order, s);
}


ushort LibRaw::get2()
{
  uchar str[2] = {0xff, 0xff};
  fread(str, 1, 2, ifp);
  return sget2(str);
}

unsigned LibRaw::sget4(uchar *s)
{
    return libraw_sget4_static(order, s);
}


unsigned libraw_sget4_static(short _order, uchar *s)
{
  if (_order == 0x4949)
    return s[0] | s[1] << 8 | s[2] << 16 | s[3] << 24;
  else
    return s[0] << 24 | s[1] << 16 | s[2] << 8 | s[3];
}

unsigned LibRaw::get4()
{
  uchar str[4] = {0xff, 0xff, 0xff, 0xff};
  fread(str, 1, 4, ifp);
  return sget4(str);
}

unsigned LibRaw::getint(int type) { return tagtypeIs(LIBRAW_EXIFTAG_TYPE_SHORT) ? get2() : get4(); }

float libraw_int_to_float(int i)
{
  union {
    int i;
    float f;
  } u;
  u.i = i;
  return u.f;
}

float LibRaw::int_to_float(int i) { return libraw_int_to_float(i); }

double LibRaw::getreal(int type)
{
  union {
    char c[8];
    double d;
  } u, v;
  int i, rev;

  switch (type)
  {
  case LIBRAW_EXIFTAG_TYPE_SHORT:
    return (unsigned short)get2();
  case LIBRAW_EXIFTAG_TYPE_LONG:
    return (unsigned int)get4();
  case LIBRAW_EXIFTAG_TYPE_RATIONAL: // (unsigned, unsigned)
    u.d = (unsigned int)get4();
    v.d = (unsigned int)get4();
    return u.d / (v.d ? v.d : 1);
  case LIBRAW_EXIFTAG_TYPE_SSHORT:
    return (signed short)get2();
  case LIBRAW_EXIFTAG_TYPE_SLONG:
    return (signed int)get4();
  case LIBRAW_EXIFTAG_TYPE_SRATIONAL: // (int, int)
    u.d = (signed int)get4();
    v.d = (signed int)get4();
    return u.d / (v.d ? v.d : 1);
  case LIBRAW_EXIFTAG_TYPE_FLOAT:
    return int_to_float(get4());
  case LIBRAW_EXIFTAG_TYPE_DOUBLE:
    rev = 7 * ((order == 0x4949) == (ntohs(0x1234) == 0x1234));
    for (i = 0; i < 8; i++)
      u.c[i ^ rev] = fgetc(ifp);
    return u.d;
  default:
    return fgetc(ifp);
  }
}

double LibRaw::sgetreal(int type, uchar *s)
{
    return libraw_sgetreal_static(order, type, s);
}


double libraw_sgetreal_static(short _order, int type, uchar *s)
{
  union {
    char c[8];
    double d;
  } u, v;
  int i, rev;

  switch (type)
  {
  case LIBRAW_EXIFTAG_TYPE_SHORT:
    return (unsigned short) libraw_sget2_static(_order,s);
  case LIBRAW_EXIFTAG_TYPE_LONG:
      return (unsigned int)libraw_sget4_static(_order, s);
  case LIBRAW_EXIFTAG_TYPE_RATIONAL: // (unsigned, unsigned)
    u.d = (unsigned int)libraw_sget4_static(_order,s);
    v.d = (unsigned int)libraw_sget4_static(_order,s+4);
    return u.d / (v.d ? v.d : 1);
  case LIBRAW_EXIFTAG_TYPE_SSHORT:
    return (signed short)libraw_sget2_static(_order,s);
  case LIBRAW_EXIFTAG_TYPE_SLONG:
    return (signed int) libraw_sget4_static(_order,s);
  case LIBRAW_EXIFTAG_TYPE_SRATIONAL: // (int, int)
    u.d = (signed int)libraw_sget4_static(_order,s);
    v.d = (signed int)libraw_sget4_static(_order,s+4);
    return u.d / (v.d ? v.d : 1);
  case LIBRAW_EXIFTAG_TYPE_FLOAT:
    return libraw_int_to_float(libraw_sget4_static(_order,s));
  case LIBRAW_EXIFTAG_TYPE_DOUBLE:
    rev = 7 * ((_order == 0x4949) == (ntohs(0x1234) == 0x1234));
    for (i = 0; i < 8; i++)
      u.c[i ^ rev] = *(s+1);
    return u.d;
  default:
    return *(s+1);
  }
}


void LibRaw::read_shorts(ushort *pixel, unsigned count)
{
  if ((unsigned)fread(pixel, 2, count, ifp) < count)
    derror();
  if ((order == 0x4949) == (ntohs(0x1234) == 0x1234))
    libraw_swab(pixel, count * 2);
}
