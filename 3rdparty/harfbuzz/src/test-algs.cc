/*
 * Copyright © 2019  Facebook, Inc.
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
 * Facebook Author(s): Behdad Esfahbod
 */

#include "hb.hh"
#include "hb-algs.hh"


static char *
test_func (int a, char **b)
{
  return b ? b[a] : nullptr;
}

struct A
{
  void a () {}
};

int
main (int argc, char **argv)
{
  int i = 1;
  auto p = hb_pair (1, i);

  p.second = 2;
  assert (i == 2);

  const int c = 3;
  auto pc = hb_pair (1, c);
  assert (pc.second == 3);

  auto q = p;
  assert (&q != &p);
  q.second = 4;
  assert (i == 4);

  hb_invoke (test_func, 0, nullptr);

  A a;
  hb_invoke (&A::a, a);

  assert (1 == hb_min (8, 1));
  assert (8 == hb_max (8, 1));

  int x = 1, y = 2;
  hb_min (x, 3);
  hb_min (3, x);
  hb_min (x, 4 + 3);
  int &z = hb_min (x, y);
  z = 3;
  assert (x == 3);

  hb_pair_t<const int*, int> xp = hb_pair_t<int *, long> (nullptr, 0);
  xp = hb_pair_t<int *, double> (nullptr, 1);
  xp = hb_pair_t<const int*, int> (nullptr, 1);

  assert (3 == hb_partial (hb_min, 3) (4));
  assert (3 == hb_partial<1> (hb_min, 4) (3));

  auto M0 = hb_partial<2> (hb_max, 0);
  assert (M0 (-2) == 0);
  assert (M0 (+2) == 2);

  assert (hb_add (2) (5) == 7);
  assert (hb_add (5) (2) == 7);

  x = 1;
  assert (++hb_inc (x) == 3);
  assert (x == 3);

  return 0;
}
