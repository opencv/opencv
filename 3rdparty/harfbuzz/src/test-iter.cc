/*
 * Copyright © 2018  Google, Inc.
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
 * Google Author(s): Behdad Esfahbod
 */

#include "hb.hh"
#include "hb-iter.hh"

#include "hb-array.hh"
#include "hb-set.hh"
#include "hb-ot-layout-common.hh"


template <typename T>
struct array_iter_t : hb_iter_with_fallback_t<array_iter_t<T>, T&>
{
  array_iter_t (hb_array_t<T> arr_) : arr (arr_) {}

  typedef T& __item_t__;
  static constexpr bool is_random_access_iterator = true;
  T& __item_at__ (unsigned i) const { return arr[i]; }
  void __forward__ (unsigned n) { arr += n; }
  void __rewind__ (unsigned n) { arr -= n; }
  unsigned __len__ () const { return arr.length; }
  bool operator != (const array_iter_t& o) { return arr != o.arr; }

  private:
  hb_array_t<T> arr;
};

template <typename T>
struct some_array_t
{
  some_array_t (hb_array_t<T> arr_) : arr (arr_) {}

  typedef array_iter_t<T> iter_t;
  array_iter_t<T> iter () { return array_iter_t<T> (arr); }
  operator array_iter_t<T> () { return iter (); }
  operator hb_iter_t<array_iter_t<T>> () { return iter (); }

  private:
  hb_array_t<T> arr;
};


template <typename Iter,
	  hb_requires (hb_is_iterator (Iter))>
static void
test_iterator_non_default_constructable (Iter it)
{
  /* Iterate over a copy of it. */
  for (auto c = it.iter (); c; c++)
    *c;

  /* Same. */
  for (auto c = +it; c; c++)
    *c;

  /* Range-based for over a copy. */
  for (auto _ : +it)
    (void) _;

  it += it.len ();
  it = it + 10;
  it = 10 + it;

  assert (*it == it[0]);

  static_assert (true || it.is_random_access_iterator, "");
  static_assert (true || it.is_sorted_iterator, "");
}

template <typename Iter,
	  hb_requires (hb_is_iterator (Iter))>
static void
test_iterator (Iter it)
{
  Iter default_constructed;
  assert (!default_constructed);

  test_iterator_non_default_constructable (it);
}

template <typename Iterable,
	  hb_requires (hb_is_iterable (Iterable))>
static void
test_iterable (const Iterable &lst = Null (Iterable))
{
  for (auto _ : lst)
    (void) _;

  // Test that can take iterator from.
  test_iterator (lst.iter ());
}

int
main (int argc, char **argv)
{
  const int src[10] = {};
  int dst[20];
  hb_vector_t<int> v;

  array_iter_t<const int> s (src); /* Implicit conversion from static array. */
  array_iter_t<const int> s2 (v); /* Implicit conversion from vector. */
  array_iter_t<int> t (dst);

  static_assert (array_iter_t<int>::is_random_access_iterator, "");

  some_array_t<const int> a (src);

  s2 = s;

  hb_iter (src);
  hb_iter (src, 2);

  hb_fill (t, 42);
  hb_copy (s, t);
  hb_copy (a.iter (), t);

  test_iterable (v);
  hb_set_t st;
  st << 1 << 15 << 43;
  test_iterable (st);
  hb_sorted_array_t<int> sa;
  (void) static_cast<hb_iter_t<hb_sorted_array_t<int>, hb_sorted_array_t<int>::item_t>&> (sa);
  (void) static_cast<hb_iter_t<hb_sorted_array_t<int>, hb_sorted_array_t<int>::__item_t__>&> (sa);
  (void) static_cast<hb_iter_t<hb_sorted_array_t<int>, int&>&>(sa);
  (void) static_cast<hb_iter_t<hb_sorted_array_t<int>>&>(sa);
  (void) static_cast<hb_iter_t<hb_array_t<int>, int&>&> (sa);
  test_iterable (sa);

  test_iterable<hb_array_t<int>> ();
  test_iterable<hb_sorted_array_t<const int>> ();
  test_iterable<hb_vector_t<float>> ();
  test_iterable<hb_set_t> ();
  test_iterable<OT::Coverage> ();

  test_iterator (hb_zip (st, v));
  test_iterator_non_default_constructable (hb_enumerate (st));
  test_iterator_non_default_constructable (hb_enumerate (st, -5));
  test_iterator_non_default_constructable (hb_enumerate (hb_iter (st)));
  test_iterator_non_default_constructable (hb_enumerate (hb_iter (st) + 1));
  test_iterator_non_default_constructable (hb_iter (st) | hb_filter ());
  test_iterator_non_default_constructable (hb_iter (st) | hb_map (hb_lidentity));

  assert (true == hb_all (st));
  assert (false == hb_all (st, 42u));
  assert (true == hb_any (st));
  assert (false == hb_any (st, 14u));
  assert (true == hb_any (st, 14u, [] (unsigned _) { return _ - 1u; }));
  assert (true == hb_any (st, [] (unsigned _) { return _ == 15u; }));
  assert (true == hb_any (st, 15u));
  assert (false == hb_none (st));
  assert (false == hb_none (st, 15u));
  assert (true == hb_none (st, 17u));

  hb_array_t<hb_vector_t<int>> pa;
  pa->as_array ();

  hb_map_t m;

  hb_iter (st);
  hb_iter (&st);

  + hb_iter (src)
  | hb_map (m)
  | hb_map (&m)
  | hb_filter ()
  | hb_filter (st)
  | hb_filter (&st)
  | hb_filter (hb_bool)
  | hb_filter (hb_bool, hb_identity)
  | hb_sink (st)
  ;

  + hb_iter (src)
  | hb_sink (hb_array (dst))
  ;

  + hb_iter (src)
  | hb_apply (&st)
  ;

  + hb_iter (src)
  | hb_map ([] (int i) { return 1; })
  | hb_reduce ([=] (int acc, int value) { return acc; }, 2)
  ;

  using map_pair_t = hb_item_type<hb_map_t>;
  + hb_iter (m)
  | hb_map ([] (map_pair_t p) { return p.first * p.second; })
  ;

  m.keys ();
  using map_key_t = decltype (*m.keys());
  + hb_iter (m.keys ())
  | hb_filter ([] (map_key_t k) { return k < 42; })
  | hb_drain
  ;

  m.values ();
  using map_value_t = decltype (*m.values());
  + hb_iter (m.values ())
  | hb_filter ([] (map_value_t k) { return k < 42; })
  | hb_drain
  ;

  unsigned int temp1 = 10;
  unsigned int temp2 = 0;
  hb_map_t *result =
  + hb_iter (src)
  | hb_map ([&] (int i) -> hb_set_t *
	    {
	      hb_set_t *set = hb_set_create ();
	      for (unsigned int i = 0; i < temp1; ++i)
		hb_set_add (set, i);
	      temp1++;
	      return set;
	    })
  | hb_reduce ([&] (hb_map_t *acc, hb_set_t *value) -> hb_map_t *
	       {
		 hb_map_set (acc, temp2++, hb_set_get_population (value));
		 /* This is not a memory managed language, take care! */
		 hb_set_destroy (value);
		 return acc;
	       }, hb_map_create ())
  ;
  /* The result should be something like 0->10, 1->11, ..., 9->19 */
  assert (hb_map_get (result, 9) == 19);

  unsigned int temp3 = 0;
  + hb_iter(src)
  | hb_map([&] (int i) { return ++temp3; })
  | hb_reduce([&] (float acc, int value) { return acc + value; }, 0)
  ;
  hb_map_destroy (result);

  + hb_iter (src)
  | hb_drain
  ;

  t << 1;
  long vl;
  s >> vl;

  hb_iota ();
  hb_iota (3);
  hb_iota (3, 2);
  assert ((&vl) + 1 == *++hb_iota (&vl, hb_inc));
  hb_range ();
  hb_repeat (7u);
  hb_repeat (nullptr);
  hb_repeat (vl) | hb_chop (3);
  assert (hb_len (hb_range (10) | hb_take (3)) == 3);
  assert (hb_range (9).len () == 9);
  assert (hb_range (2, 9).len () == 7);
  assert (hb_range (2, 9, 3).len () == 3);
  assert (hb_range (2, 8, 3).len () == 2);
  assert (hb_range (2, 7, 3).len () == 2);
  assert (hb_range (-2, -9, -3).len () == 3);
  assert (hb_range (-2, -8, -3).len () == 2);
  assert (hb_range (-2, -7, -3).len () == 2);

  return 0;
}
