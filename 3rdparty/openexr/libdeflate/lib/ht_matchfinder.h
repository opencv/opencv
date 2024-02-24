/*
 * ht_matchfinder.h - Lempel-Ziv matchfinding with a hash table
 *
 * Copyright 2022 Eric Biggers
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * ---------------------------------------------------------------------------
 *
 * This is a Hash Table (ht) matchfinder.
 *
 * This is a variant of the Hash Chains (hc) matchfinder that is optimized for
 * very fast compression.  The ht_matchfinder stores the hash chains inline in
 * the hash table, whereas the hc_matchfinder stores them in a separate array.
 * Storing the hash chains inline is the faster method when max_search_depth
 * (the maximum chain length) is very small.  It is not appropriate when
 * max_search_depth is larger, as then it uses too much memory.
 *
 * Due to its focus on speed, the ht_matchfinder doesn't support length 3
 * matches.  It also doesn't allow max_search_depth to vary at runtime; it is
 * fixed at build time as HT_MATCHFINDER_BUCKET_SIZE.
 *
 * See hc_matchfinder.h for more information.
 */

#ifndef LIB_HT_MATCHFINDER_H
#define LIB_HT_MATCHFINDER_H

#include "matchfinder_common.h"

#define HT_MATCHFINDER_HASH_ORDER	15
#define HT_MATCHFINDER_BUCKET_SIZE	2

#define HT_MATCHFINDER_MIN_MATCH_LEN	4
/* Minimum value of max_len for ht_matchfinder_longest_match() */
#define HT_MATCHFINDER_REQUIRED_NBYTES	5

struct MATCHFINDER_ALIGNED ht_matchfinder {
	mf_pos_t hash_tab[1UL << HT_MATCHFINDER_HASH_ORDER]
			 [HT_MATCHFINDER_BUCKET_SIZE];
};

static forceinline void
ht_matchfinder_init(struct ht_matchfinder *mf)
{
	STATIC_ASSERT(sizeof(*mf) % MATCHFINDER_SIZE_ALIGNMENT == 0);

	matchfinder_init((mf_pos_t *)mf, sizeof(*mf));
}

static forceinline void
ht_matchfinder_slide_window(struct ht_matchfinder *mf)
{
	matchfinder_rebase((mf_pos_t *)mf, sizeof(*mf));
}

/* Note: max_len must be >= HT_MATCHFINDER_REQUIRED_NBYTES */
static forceinline u32
ht_matchfinder_longest_match(struct ht_matchfinder * const mf,
			     const u8 ** const in_base_p,
			     const u8 * const in_next,
			     const u32 max_len,
			     const u32 nice_len,
			     u32 * const next_hash,
			     u32 * const offset_ret)
{
	u32 best_len = 0;
	const u8 *best_matchptr = in_next;
	u32 cur_pos = in_next - *in_base_p;
	const u8 *in_base;
	mf_pos_t cutoff;
	u32 hash;
	u32 seq;
	mf_pos_t cur_node;
	const u8 *matchptr;
#if HT_MATCHFINDER_BUCKET_SIZE > 1
	mf_pos_t to_insert;
	u32 len;
#endif
#if HT_MATCHFINDER_BUCKET_SIZE > 2
	int i;
#endif

	/* This is assumed throughout this function. */
	STATIC_ASSERT(HT_MATCHFINDER_MIN_MATCH_LEN == 4);

	if (cur_pos == MATCHFINDER_WINDOW_SIZE) {
		ht_matchfinder_slide_window(mf);
		*in_base_p += MATCHFINDER_WINDOW_SIZE;
		cur_pos = 0;
	}
	in_base = *in_base_p;
	cutoff = cur_pos - MATCHFINDER_WINDOW_SIZE;

	hash = *next_hash;
	STATIC_ASSERT(HT_MATCHFINDER_REQUIRED_NBYTES == 5);
	*next_hash = lz_hash(get_unaligned_le32(in_next + 1),
			     HT_MATCHFINDER_HASH_ORDER);
	seq = load_u32_unaligned(in_next);
	prefetchw(&mf->hash_tab[*next_hash]);
#if HT_MATCHFINDER_BUCKET_SIZE == 1
	/* Hand-unrolled version for BUCKET_SIZE == 1 */
	cur_node = mf->hash_tab[hash][0];
	mf->hash_tab[hash][0] = cur_pos;
	if (cur_node <= cutoff)
		goto out;
	matchptr = &in_base[cur_node];
	if (load_u32_unaligned(matchptr) == seq) {
		best_len = lz_extend(in_next, matchptr, 4, max_len);
		best_matchptr = matchptr;
	}
#elif HT_MATCHFINDER_BUCKET_SIZE == 2
	/*
	 * Hand-unrolled version for BUCKET_SIZE == 2.  The logic here also
	 * differs slightly in that it copies the first entry to the second even
	 * if nice_len is reached on the first, as this can be slightly faster.
	 */
	cur_node = mf->hash_tab[hash][0];
	mf->hash_tab[hash][0] = cur_pos;
	if (cur_node <= cutoff)
		goto out;
	matchptr = &in_base[cur_node];

	to_insert = cur_node;
	cur_node = mf->hash_tab[hash][1];
	mf->hash_tab[hash][1] = to_insert;

	if (load_u32_unaligned(matchptr) == seq) {
		best_len = lz_extend(in_next, matchptr, 4, max_len);
		best_matchptr = matchptr;
		if (cur_node <= cutoff || best_len >= nice_len)
			goto out;
		matchptr = &in_base[cur_node];
		if (load_u32_unaligned(matchptr) == seq &&
		    load_u32_unaligned(matchptr + best_len - 3) ==
		    load_u32_unaligned(in_next + best_len - 3)) {
			len = lz_extend(in_next, matchptr, 4, max_len);
			if (len > best_len) {
				best_len = len;
				best_matchptr = matchptr;
			}
		}
	} else {
		if (cur_node <= cutoff)
			goto out;
		matchptr = &in_base[cur_node];
		if (load_u32_unaligned(matchptr) == seq) {
			best_len = lz_extend(in_next, matchptr, 4, max_len);
			best_matchptr = matchptr;
		}
	}
#else
	/* Generic version for HT_MATCHFINDER_BUCKET_SIZE > 2 */
	to_insert = cur_pos;
	for (i = 0; i < HT_MATCHFINDER_BUCKET_SIZE; i++) {
		cur_node = mf->hash_tab[hash][i];
		mf->hash_tab[hash][i] = to_insert;
		if (cur_node <= cutoff)
			goto out;
		matchptr = &in_base[cur_node];
		if (load_u32_unaligned(matchptr) == seq) {
			len = lz_extend(in_next, matchptr, 4, max_len);
			if (len > best_len) {
				best_len = len;
				best_matchptr = matchptr;
				if (best_len >= nice_len)
					goto out;
			}
		}
		to_insert = cur_node;
	}
#endif
out:
	*offset_ret = in_next - best_matchptr;
	return best_len;
}

static forceinline void
ht_matchfinder_skip_bytes(struct ht_matchfinder * const mf,
			  const u8 ** const in_base_p,
			  const u8 *in_next,
			  const u8 * const in_end,
			  const u32 count,
			  u32 * const next_hash)
{
	s32 cur_pos = in_next - *in_base_p;
	u32 hash;
	u32 remaining = count;
	int i;

	if (unlikely(count + HT_MATCHFINDER_REQUIRED_NBYTES > in_end - in_next))
		return;

	if (cur_pos + count - 1 >= MATCHFINDER_WINDOW_SIZE) {
		ht_matchfinder_slide_window(mf);
		*in_base_p += MATCHFINDER_WINDOW_SIZE;
		cur_pos -= MATCHFINDER_WINDOW_SIZE;
	}

	hash = *next_hash;
	do {
		for (i = HT_MATCHFINDER_BUCKET_SIZE - 1; i > 0; i--)
			mf->hash_tab[hash][i] = mf->hash_tab[hash][i - 1];
		mf->hash_tab[hash][0] = cur_pos;

		hash = lz_hash(get_unaligned_le32(++in_next),
			       HT_MATCHFINDER_HASH_ORDER);
		cur_pos++;
	} while (--remaining);

	prefetchw(&mf->hash_tab[hash]);
	*next_hash = hash;
}

#endif /* LIB_HT_MATCHFINDER_H */
