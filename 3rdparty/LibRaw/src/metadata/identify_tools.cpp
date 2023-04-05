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

short LibRaw::guess_byte_order(int words)
{
  uchar test[4][2];
  int t = 2, msb;
  double diff, sum[2] = {0, 0};

  fread(test[0], 2, 2, ifp);
  for (words -= 2; words--;)
  {
    fread(test[t], 2, 1, ifp);
    for (msb = 0; msb < 2; msb++)
    {
      diff = (test[t ^ 2][msb] << 8 | test[t ^ 2][!msb]) -
             (test[t][msb] << 8 | test[t][!msb]);
      sum[msb] += diff * diff;
    }
    t = (t + 1) & 3;
  }
  return sum[0] < sum[1] ? 0x4d4d : 0x4949;
}

float LibRaw::find_green(int bps, int bite, int off0, int off1)
{
  UINT64 bitbuf = 0;
  int vbits, col, i, c;
  ushort img[2][2064];
  double sum[] = {0, 0};
  if (width > 2064)
    return 0.f; // too wide

  FORC(2)
  {
    fseek(ifp, c ? off1 : off0, SEEK_SET);
    for (vbits = col = 0; col < width; col++)
    {
      for (vbits -= bps; vbits < 0; vbits += bite)
      {
        bitbuf <<= bite;
        for (i = 0; i < bite; i += 8)
          bitbuf |= (unsigned)(fgetc(ifp) << i);
      }
      img[c][col] = bitbuf << (64 - bps - vbits) >> (64 - bps);
    }
  }
  FORC(width - 1)
  {
    sum[c & 1] += ABS(img[0][c] - img[1][c + 1]);
    sum[~c & 1] += ABS(img[1][c] - img[0][c + 1]);
  }
  if (sum[0] >= 1.0 && sum[1] >= 1.0)
    return 100 * log(sum[0] / sum[1]);
  else
    return 0.f;
}

void LibRaw::trimSpaces(char *s)
{
  char *p = s;
  int l = int(strlen(p));
  if (!l)
    return;
  while (isspace(p[l - 1]))
    p[--l] = 0; /* trim trailing spaces */
  while (*p && isspace(*p))
    ++p, --l;   /* trim leading spaces */
  memmove(s, p, l + 1);
}

void LibRaw::remove_trailing_spaces(char *string, size_t len)
{
  if (len < 1)
    return; // not needed, b/c sizeof of make/model is 64
  string[len - 1] = 0;
  if (len < 3)
    return; // also not needed
  len = strnlen(string, len - 1);
  for (size_t i = len - 1; i >= 0; i--)
  {
    if (isspace((unsigned char)string[i]))
      string[i] = 0;
    else
      break;
  }
}

void LibRaw::remove_caseSubstr(char *string, char *subStr) // replace a substring with an equal length of spaces
{
  char *found;
  while ((found = strcasestr(string,subStr))) {
    if (!found) return;
    int fill_len = int(strlen(subStr));
    int p = found - string;
    for (int i=p; i<p+fill_len; i++) {
      string[i] = 32;
    }
  }
  trimSpaces (string);
}

void LibRaw::removeExcessiveSpaces(char *string) // replace repeating spaces with one space
{
	int orig_len = int(strlen(string));
	int i = 0;   // counter for resulting string
	int j = -1;
	bool prev_char_is_space = false;
	while (++j < orig_len && string[j] == ' ');
	while (j < orig_len)  {
		if (string[j] != ' ')  {
				string[i++] = string[j++];
				prev_char_is_space = false;
		} else if (string[j++] == ' ') {
			if (!prev_char_is_space) {
				string[i++] = ' ';
				prev_char_is_space = true;
			}
		}
	}
	if (string[i-1] == ' ')
    string[i-1] = 0;
}
