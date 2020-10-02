/*
 * Copyright © 2019  Ebrahim Byagowi
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 */

#ifndef HB_NUMBER_PARSER_HH
#define HB_NUMBER_PARSER_HH

#include "hb.hh"

%%{

machine double_parser;
alphtype unsigned char;
write data;

action see_neg { neg = true; }
action see_exp_neg { exp_neg = true; }

action add_int  {
	value = value * 10. + (fc - '0');
}
action add_frac {
	if (likely (frac <= MAX_FRACT / 10))
	{
	  frac = frac * 10. + (fc - '0');
	  ++frac_count;
	}
}
action add_exp  {
	if (likely (exp * 10 + (fc - '0') <= MAX_EXP))
	  exp = exp * 10 + (fc - '0');
	else
	  exp_overflow = true;
}

num = [0-9]+;

main := (
	(
		(('+'|'-'@see_neg)? num @add_int) ('.' num @add_frac)?
		|
		(('+'|'-'@see_neg)? '.' num @add_frac)
	)
	(('e'|'E') (('+'|'-'@see_exp_neg)? num @add_exp))?
);

}%%

/* Works only for n < 512 */
static inline double
_pow10 (unsigned exponent)
{
  static const double _powers_of_10[] =
  {
    1.0e+256,
    1.0e+128,
    1.0e+64,
    1.0e+32,
    1.0e+16,
    1.0e+8,
    10000.,
    100.,
    10.
  };
  unsigned mask = 1 << (ARRAY_LENGTH (_powers_of_10) - 1);
  double result = 1;
  for (const double *power = _powers_of_10; mask; ++power, mask >>= 1)
    if (exponent & mask) result *= *power;
  return result;
}

/* a variant of strtod that also gets end of buffer in its second argument */
static inline double
strtod_rl (const char *p, const char **end_ptr /* IN/OUT */)
{
  double value = 0;
  double frac = 0;
  double frac_count = 0;
  unsigned exp = 0;
  bool neg = false, exp_neg = false, exp_overflow = false;
  const unsigned long long MAX_FRACT = 0xFFFFFFFFFFFFFull; /* 2^52-1 */
  const unsigned MAX_EXP = 0x7FFu; /* 2^11-1 */

  const char *pe = *end_ptr;
  while (p < pe && ISSPACE (*p))
    p++;

  int cs;
  %%{
    write init;
    write exec;
  }%%

  *end_ptr = p;

  if (frac_count) value += frac / _pow10 (frac_count);
  if (neg) value *= -1.;

  if (unlikely (exp_overflow))
  {
    if (value == 0) return value;
    if (exp_neg)    return neg ? -DBL_MIN : DBL_MIN;
    else            return neg ? -DBL_MAX : DBL_MAX;
  }

  if (exp)
  {
    if (exp_neg) value /= _pow10 (exp);
    else         value *= _pow10 (exp);
  }

  return value;
}

#endif /* HB_NUMBER_PARSER_HH */
