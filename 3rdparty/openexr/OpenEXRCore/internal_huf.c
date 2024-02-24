/*
** SPDX-License-Identifier: BSD-3-Clause
** Copyright Contributors to the OpenEXR Project.
*/

#include "internal_huf.h"

#include "internal_memory.h"
#include "internal_xdr.h"
#include "internal_structs.h"

#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define HUF_ENCBITS 16
#define HUF_DECBITS 14

#define HUF_ENCSIZE ((1 << HUF_ENCBITS) + 1)
#define HUF_DECSIZE (1 << HUF_DECBITS)
#define HUF_DECMASK (HUF_DECSIZE - 1)

#define SHORT_ZEROCODE_RUN 59
#define LONG_ZEROCODE_RUN 63
#define SHORTEST_LONG_RUN (2 + LONG_ZEROCODE_RUN - SHORT_ZEROCODE_RUN)
#define LONGEST_LONG_RUN (255 + SHORTEST_LONG_RUN)

typedef struct _HufDec
{
    int32_t   len;
    uint32_t  lit;
    uint32_t* p;
} HufDec;

/**************************************/

static inline int
hufLength (uint64_t code)
{
    return (int) (code & 63);
}

static inline uint64_t
hufCode (uint64_t code)
{
    return code >> 6;
}

static inline exr_result_t
outputBits (
    int       nBits,
    uint64_t  bits,
    uint64_t* c,
    int*      lc,
    uint8_t** outptr,
    uint8_t*  outend)
{
    uint8_t* out = *outptr;
    *c <<= nBits;
    *lc += nBits;
    *c |= bits;

    while (*lc >= 8)
    {
        if (out >= outend) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;
        *out++ = (uint8_t) (*c >> (*lc -= 8));
    }
    *outptr = out;
    return EXR_ERR_SUCCESS;
}

static inline uint64_t
getBits (uint32_t nBits, uint64_t* c, uint32_t* lc, const uint8_t** inptr)
{
    const uint8_t* in = *inptr;
    while (*lc < nBits)
    {
        *c = (*c << 8) | (uint64_t) (*in++);
        *lc += 8;
    }

    *inptr = in;
    *lc -= nBits;
    return (*c >> *lc) & ((1 << nBits) - 1);
}

//
// ENCODING TABLE BUILDING & (UN)PACKING
//

//
// Build a "canonical" Huffman code table:
//	- for each (uncompressed) symbol, hcode contains the length
//	  of the corresponding code (in the compressed data)
//	- canonical codes are computed and stored in hcode
//	- the rules for constructing canonical codes are as follows:
//	  * shorter codes (if filled with zeroes to the right)
//	    have a numerically higher value than longer codes
//	  * for codes with the same length, numerical values
//	    increase with numerical symbol values
//	- because the canonical code table can be constructed from
//	  symbol lengths alone, the code table can be transmitted
//	  without sending the actual code values
//	- see http://www.compressconsult.com/huffman/
//

static void
hufCanonicalCodeTable (uint64_t* hcode)
{
    uint64_t n[59];
    uint64_t c;

    //
    // For each i from 0 through 58, count the
    // number of different codes of length i, and
    // store the count in n[i].
    //

    for (int i = 0; i <= 58; ++i)
        n[i] = 0;

    for (int i = 0; i < HUF_ENCSIZE; ++i)
        n[hcode[i]] += 1;

    //
    // For each i from 58 through 1, compute the
    // numerically lowest code with length i, and
    // store that code in n[i].
    //

    c = 0;

    for (int i = 58; i > 0; --i)
    {
        uint64_t nc = ((c + n[i]) >> 1);
        n[i]        = c;
        c           = nc;
    }

    //
    // hcode[i] contains the length, l, of the
    // code for symbol i.  Assign the next available
    // code of length l to the symbol and store both
    // l and the code in hcode[i].
    //

    for (int i = 0; i < HUF_ENCSIZE; ++i)
    {
        uint64_t l = hcode[i];

        if (l > 0) hcode[i] = l | (n[l]++ << 6);
    }
}

//
// Compute Huffman codes (based on frq input) and store them in frq:
//	- code structure is : [63:lsb - 6:msb] | [5-0: bit length];
//	- max code length is 58 bits;
//	- codes outside the range [im-iM] have a null length (unused values);
//	- original frequencies are destroyed;
//	- encoding tables are used by hufEncode() and hufBuildDecTable();
//
// NB: The following code "(*a == *b) && (a > b))" was added to ensure
//     elements in the heap with the same value are sorted by index.
//     This is to ensure, the STL make_heap()/pop_heap()/push_heap() methods
//     produced a resultant sorted heap that is identical across OSes.
//

static inline int
FHeapCompare (uint64_t* a, uint64_t* b)
{
    return ((*a > *b) || ((*a == *b) && (a > b)));
}

static inline void
intern_push_heap (
    uint64_t** first, size_t holeIndex, size_t topIndex, uint64_t* value)
{
    size_t parent = (holeIndex - 1) / 2;
    while (holeIndex > topIndex && FHeapCompare (*(first + parent), value))
    {
        *(first + holeIndex) = *(first + parent);
        holeIndex            = parent;
        parent               = (holeIndex - 1) / 2;
    }
    *(first + holeIndex) = value;
}

static inline void
adjust_heap (uint64_t** first, size_t holeIndex, size_t len, uint64_t* value)
{
    const size_t topIndex    = holeIndex;
    size_t       secondChild = holeIndex;

    while (secondChild < (len - 1) / 2)
    {
        secondChild = 2 * (secondChild + 1);
        if (FHeapCompare (*(first + secondChild), *(first + (secondChild - 1))))
            --secondChild;
        *(first + holeIndex) = *(first + secondChild);
        holeIndex            = secondChild;
    }

    if ((len & 1) == 0 && secondChild == (len - 2) / 2)
    {
        secondChild          = 2 * (secondChild + 1);
        *(first + holeIndex) = *(first + (secondChild - 1));
        holeIndex            = secondChild - 1;
    }

    intern_push_heap (first, holeIndex, topIndex, value);
}

static inline void
push_heap (uint64_t** first, uint64_t** last)
{
    uint64_t* value = *(last - 1);
    intern_push_heap (first, (size_t) (last - first) - 1, 0, value);
}

static inline void
intern_pop_heap (uint64_t** first, uint64_t** last, uint64_t** result)
{
    uint64_t* value = *result;
    *result         = *first;
    adjust_heap (first, 0, (size_t) (last - first), value);
}

static inline void
pop_heap (uint64_t** first, uint64_t** last)
{
    if (last - first > 1)
    {
        --last;
        intern_pop_heap (first, last, last);
    }
}

static void
make_heap (uint64_t** first, uint64_t len)
{
    size_t parent;

    if (len < 2) return;
    parent = (len - 2) / 2;

    while (1)
    {
        uint64_t* value = *(first + parent);
        adjust_heap (first, parent, len, value);
        if (parent == 0) return;
        --parent;
    }
}

static void
hufBuildEncTable (
    uint64_t*  frq,
    uint32_t*  im,
    uint32_t*  iM,
    uint32_t*  hlink,
    uint64_t** fHeap,
    uint64_t*  scode)
{
    //
    // This function assumes that when it is called, array frq
    // indicates the frequency of all possible symbols in the data
    // that are to be Huffman-encoded.  (frq[i] contains the number
    // of occurrences of symbol i in the data.)
    //
    // The loop below does three things:
    //
    // 1) Finds the minimum and maximum indices that point
    //    to non-zero entries in frq:
    //
    //     frq[im] != 0, and frq[i] == 0 for all i < im
    //     frq[iM] != 0, and frq[i] == 0 for all i > iM
    //
    // 2) Fills array fHeap with pointers to all non-zero
    //    entries in frq.
    //
    // 3) Initializes array hlink such that hlink[i] == i
    //    for all array entries.
    //
    uint32_t nf = 0;

    *im = 0;

    while (!frq[*im])
        (*im)++;

    for (uint32_t i = *im; i < HUF_ENCSIZE; i++)
    {
        hlink[i] = i;

        if (frq[i])
        {
            fHeap[nf] = &frq[i];
            ++nf;
            *iM = i;
        }
    }

    //
    // Add a pseudo-symbol, with a frequency count of 1, to frq;
    // adjust the fHeap and hlink array accordingly.  Function
    // hufEncode() uses the pseudo-symbol for run-length encoding.
    //

    (*iM)++;
    frq[*iM]  = 1;
    fHeap[nf] = &frq[*iM];
    ++nf;

    //
    // Build an array, scode, such that scode[i] contains the number
    // of bits assigned to symbol i.  Conceptually this is done by
    // constructing a tree whose leaves are the symbols with non-zero
    // frequency:
    //
    //     Make a heap that contains all symbols with a non-zero frequency,
    //     with the least frequent symbol on top.
    //
    //     Repeat until only one symbol is left on the heap:
    //
    //         Take the two least frequent symbols off the top of the heap.
    //         Create a new node that has first two nodes as children, and
    //         whose frequency is the sum of the frequencies of the first
    //         two nodes.  Put the new node back into the heap.
    //
    // The last node left on the heap is the root of the tree.  For each
    // leaf node, the distance between the root and the leaf is the length
    // of the code for the corresponding symbol.
    //
    // The loop below doesn't actually build the tree; instead we compute
    // the distances of the leaves from the root on the fly.  When a new
    // node is added to the heap, then that node's descendants are linked
    // into a single linear list that starts at the new node, and the code
    // lengths of the descendants (that is, their distance from the root
    // of the tree) are incremented by one.
    //

    make_heap (fHeap, nf);

    memset (scode, 0, sizeof (uint64_t) * HUF_ENCSIZE);

    while (nf > 1)
    {
        uint32_t mm, m;
        //
        // Find the indices, mm and m, of the two smallest non-zero frq
        // values in fHeap, add the smallest frq to the second-smallest
        // frq, and remove the smallest frq value from fHeap.
        //

        mm = (uint32_t) (fHeap[0] - frq);
        pop_heap (&fHeap[0], &fHeap[nf]);
        --nf;

        m = (uint32_t) (fHeap[0] - frq);
        pop_heap (&fHeap[0], &fHeap[nf]);

        frq[m] += frq[mm];
        push_heap (&fHeap[0], &fHeap[nf]);

        //
        // The entries in scode are linked into lists with the
        // entries in hlink serving as "next" pointers and with
        // the end of a list marked by hlink[j] == j.
        //
        // Traverse the lists that start at scode[m] and scode[mm].
        // For each element visited, increment the length of the
        // corresponding code by one bit. (If we visit scode[j]
        // during the traversal, then the code for symbol j becomes
        // one bit longer.)
        //
        // Merge the lists that start at scode[m] and scode[mm]
        // into a single list that starts at scode[m].
        //

        //
        // Add a bit to all codes in the first list.
        //

        for (uint32_t j = m;; j = hlink[j])
        {
            scode[j]++;

            if (hlink[j] == j)
            {
                //
                // Merge the two lists.
                //

                hlink[j] = mm;
                break;
            }
        }

        //
        // Add a bit to all codes in the second list
        //

        for (uint32_t j = mm;; j = hlink[j])
        {
            scode[j]++;

            if (hlink[j] == j) break;
        }
    }

    //
    // Build a canonical Huffman code table, replacing the code
    // lengths in scode with (code, code length) pairs.  Copy the
    // code table from scode into frq.
    //

    hufCanonicalCodeTable (scode);
    memcpy (frq, scode, sizeof (uint64_t) * HUF_ENCSIZE);
}

//
// Pack an encoding table:
//	- only code lengths, not actual codes, are stored
//	- runs of zeroes are compressed as follows:
//
//	  unpacked		packed
//	  --------------------------------
//	  1 zero		0	(6 bits)
//	  2 zeroes		59
//	  3 zeroes		60
//	  4 zeroes		61
//	  5 zeroes		62
//	  n zeroes (6 or more)	63 n-6	(6 + 8 bits)
//

static exr_result_t
hufPackEncTable (
    const uint64_t* hcode, // i : encoding table [HUF_ENCSIZE]
    uint32_t        im,    // i : min hcode index
    uint32_t        iM,    // i : max hcode index
    uint8_t**       pcode, // o : ptr to packed table (updated)
    uint8_t*        pend)         // i : max size of table
{
    exr_result_t rv;
    uint8_t*     out = *pcode;
    uint64_t     c   = 0;
    int          lc  = 0;

    for (; im <= iM; im++)
    {
        int l = hufLength (hcode[im]);

        if (l == 0)
        {
            uint64_t zerun = 1;

            while ((im < iM) && (zerun < LONGEST_LONG_RUN))
            {
                if (hufLength (hcode[im + 1]) > 0) break;
                im++;
                zerun++;
            }

            if (zerun >= 2)
            {
                if (zerun >= SHORTEST_LONG_RUN)
                {
                    rv = outputBits (6, LONG_ZEROCODE_RUN, &c, &lc, &out, pend);
                    if (rv != EXR_ERR_SUCCESS) return rv;
                    rv = outputBits (
                        8, zerun - SHORTEST_LONG_RUN, &c, &lc, &out, pend);
                    if (rv != EXR_ERR_SUCCESS) return rv;
                }
                else
                {
                    rv = outputBits (
                        6, SHORT_ZEROCODE_RUN + zerun - 2, &c, &lc, &out, pend);
                    if (rv != EXR_ERR_SUCCESS) return rv;
                }
                continue;
            }
        }

        rv = outputBits (6, (uint64_t) l, &c, &lc, &out, pend);
        if (rv != EXR_ERR_SUCCESS) return rv;
    }

    if (lc > 0)
    {
        if (out >= pend) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;
        *out++ = (uint8_t) (c << (8 - lc));
    }

    *pcode = out;
    return EXR_ERR_SUCCESS;
}

//
// Unpack an encoding table packed by hufPackEncTable():
//

static exr_result_t
hufUnpackEncTable (
    const uint8_t** pcode, // io: ptr to packed table (updated)
    uint64_t*       nLeft, // io: input size (in bytes), bytes left
    uint32_t        im,    // i : min hcode index
    uint32_t        iM,    // i : max hcode index
    uint64_t*       hcode)       // o : encoding table [HUF_ENCSIZE]
{
    const uint8_t* p  = *pcode;
    uint64_t       c  = 0;
    uint64_t       ni = *nLeft;
    uint64_t       nr;
    uint32_t       lc = 0;
    uint64_t       l, zerun;

    memset (hcode, 0, sizeof (uint64_t) * HUF_ENCSIZE);
    for (; im <= iM; im++)
    {
        nr = (((uintptr_t) p) - ((uintptr_t) *pcode));
        if (lc < 6 && nr >= ni) return EXR_ERR_OUT_OF_MEMORY;

        l = hcode[im] = getBits (6, &c, &lc, &p); // code length

        if (l == (uint64_t) LONG_ZEROCODE_RUN)
        {
            nr = (((uintptr_t) p) - ((uintptr_t) *pcode));
            if (lc < 8 && nr >= ni) return EXR_ERR_OUT_OF_MEMORY;

            zerun = getBits (8, &c, &lc, &p) + SHORTEST_LONG_RUN;

            if (im + zerun > iM + 1) return EXR_ERR_CORRUPT_CHUNK;

            while (zerun--)
                hcode[im++] = 0;

            im--;
        }
        else if (l >= (uint64_t) SHORT_ZEROCODE_RUN)
        {
            zerun = l - SHORT_ZEROCODE_RUN + 2;

            if (im + zerun > iM + 1) return EXR_ERR_CORRUPT_CHUNK;

            while (zerun--)
                hcode[im++] = 0;

            im--;
        }
    }

    nr = (((uintptr_t) p) - ((uintptr_t) *pcode));
    *nLeft -= nr;
    *pcode = p;

    hufCanonicalCodeTable (hcode);
    return EXR_ERR_SUCCESS;
}

//
// DECODING TABLE BUILDING
//

//
// Clear a newly allocated decoding table so that it contains only zeroes.
//

static void
hufClearDecTable (HufDec* hdecod)
{
    memset (hdecod, 0, sizeof (HufDec) * HUF_DECSIZE);
}

//
// Build a decoding hash table based on the encoding table hcode:
//	- short codes (<= HUF_DECBITS) are resolved with a single table access;
//	- long code entry allocations are not optimized, because long codes are
//	  unfrequent;
//	- decoding tables are used by hufDecode();
//

static exr_result_t
hufBuildDecTable (
    const struct _internal_exr_context* pctxt,
    const uint64_t*                     hcode,
    uint32_t                            im,
    uint32_t                            iM,
    HufDec*                             hdecod)
{
    void* (*alloc_fn) (size_t) = pctxt ? pctxt->alloc_fn : internal_exr_alloc;
    void (*free_fn) (void*)    = pctxt ? pctxt->free_fn : internal_exr_free;

    //
    // Init hashtable & loop on all codes.
    // Assumes that hufClearDecTable(hdecod) has already been called.
    //

    for (; im <= iM; im++)
    {
        uint64_t c = hufCode (hcode[im]);
        int      l = hufLength (hcode[im]);

        if (c >> l)
        {
            //
            // Error: c is supposed to be an l-bit code,
            // but c contains a value that is greater
            // than the largest l-bit number.
            //

            return EXR_ERR_CORRUPT_CHUNK;
        }

        if (l > HUF_DECBITS)
        {
            //
            // Long code: add a secondary entry
            //

            HufDec* pl = hdecod + (c >> (l - HUF_DECBITS));

            if (pl->len)
            {
                //
                // Error: a short code has already
                // been stored in table entry *pl.
                //

                return EXR_ERR_CORRUPT_CHUNK;
            }

            pl->lit++;

            if (pl->p)
            {
                uint32_t* p = pl->p;
                pl->p = (uint32_t*) alloc_fn (sizeof (uint32_t) * pl->lit);

                if (pl->p)
                {
                    for (uint32_t i = 0; i < pl->lit - 1; ++i)
                        pl->p[i] = p[i];
                }

                free_fn (p);
            }
            else { pl->p = (uint32_t*) alloc_fn (sizeof (uint32_t)); }

            if (!pl->p) return EXR_ERR_OUT_OF_MEMORY;

            pl->p[pl->lit - 1] = im;
        }
        else if (l)
        {
            //
            // Short code: init all primary entries
            //

            HufDec* pl = hdecod + (c << (HUF_DECBITS - l));

            for (uint64_t i = ((uint64_t) 1) << (HUF_DECBITS - l); i > 0;
                 i--, pl++)
            {
                if (pl->len || pl->p)
                {
                    //
                    // Error: a short code or a long code has
                    // already been stored in table entry *pl.
                    //

                    return EXR_ERR_CORRUPT_CHUNK;
                }

                pl->len = (int32_t) l;
                pl->lit = im;
            }
        }
    }
    return EXR_ERR_SUCCESS;
}

//
// Free the long code entries of a decoding table built by hufBuildDecTable()
//

static void
hufFreeDecTable (const struct _internal_exr_context* pctxt, HufDec* hdecod)
{
    void (*free_fn) (void*) = pctxt ? pctxt->free_fn : internal_exr_free;
    for (int i = 0; i < HUF_DECSIZE; i++)
    {
        if (hdecod[i].p)
        {
            free_fn (hdecod[i].p);
            hdecod[i].p = NULL;
        }
    }
}

//
// ENCODING
//

static inline exr_result_t
outputCode (uint64_t code, uint64_t* c, int* lc, uint8_t** out, uint8_t* outend)
{
    return outputBits (hufLength (code), hufCode (code), c, lc, out, outend);
}

static inline exr_result_t
sendCode (
    uint64_t  sCode,
    int       runCount,
    uint64_t  runCode,
    uint64_t* c,
    int*      lc,
    uint8_t** out,
    uint8_t*  outend)
{
    exr_result_t rv;
    if (hufLength (sCode) + hufLength (runCode) + 8 <
        hufLength (sCode) * runCount)
    {
        rv = outputCode (sCode, c, lc, out, outend);
        if (rv == EXR_ERR_SUCCESS)
            rv = outputCode (runCode, c, lc, out, outend);
        if (rv == EXR_ERR_SUCCESS)
            rv = outputBits (8, (uint64_t) runCount, c, lc, out, outend);
    }
    else
    {
        rv = EXR_ERR_SUCCESS;
        while (runCount-- >= 0)
        {
            rv = outputCode (sCode, c, lc, out, outend);
            if (rv != EXR_ERR_SUCCESS) break;
        }
    }
    return rv;
}

//
// Encode (compress) ni values based on the Huffman encoding table hcode:
//

static inline exr_result_t
hufEncode (
    const uint64_t* hcode,
    const uint16_t* in,
    const uint64_t  ni,
    uint32_t        rlc,
    uint8_t*        out,
    uint8_t*        outend,
    uint32_t*       outbytes)
{
    exr_result_t rv = EXR_ERR_SUCCESS;

    uint8_t* outStart = out;
    uint64_t c        = 0; // bits not yet written to out
    int      lc       = 0; // number of valid bits in c (LSB)
    uint16_t s        = in[0];
    int      cs       = 0;

    //
    // Loop on input values
    //

    for (uint64_t i = 1; i < ni; i++)
    {
        //
        // Count same values or send code
        //

        if (s == in[i] && cs < 255) { cs++; }
        else
        {
            rv = sendCode (hcode[s], cs, hcode[rlc], &c, &lc, &out, outend);
            if (rv != EXR_ERR_SUCCESS) break;
            cs = 0;
        }

        s = in[i];
    }

    //
    // Send remaining code
    //

    if (rv == EXR_ERR_SUCCESS)
        rv = sendCode (hcode[s], cs, hcode[rlc], &c, &lc, &out, outend);

    if (rv == EXR_ERR_SUCCESS)
    {
        if (lc)
        {
            if (out >= outend) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;
            *out = (c << (8 - lc)) & 0xff;
        }

        c = (((uintptr_t) out) - ((uintptr_t) outStart)) * 8 + (uint64_t) (lc);
        if (c > (uint64_t) UINT32_MAX) return EXR_ERR_ARGUMENT_OUT_OF_RANGE;
        *outbytes = (uint32_t) c;
    }

    return rv;
}

//
// DECODING
//

//
// In order to force the compiler to inline them,
// getChar() and getCode() are implemented as macros
// instead of "inline" functions.
//

#define getChar(c, lc, in)                                                     \
    c = (c << 8) | (uint64_t) (*in++);                                         \
    lc += 8

#define getCode(po, rlc, c, lc, in, ie, out, ob, oe)                           \
    do                                                                         \
    {                                                                          \
        if (po == rlc)                                                         \
        {                                                                      \
            uint8_t  cs;                                                       \
            uint16_t s;                                                        \
            if (lc < 8)                                                        \
            {                                                                  \
                if (in >= ie) return EXR_ERR_OUT_OF_MEMORY;                    \
                getChar (c, lc, in);                                           \
            }                                                                  \
                                                                               \
            lc -= 8;                                                           \
                                                                               \
            cs = (uint8_t) (c >> lc);                                          \
                                                                               \
            if (out + cs > oe)                                                 \
                return EXR_ERR_CORRUPT_CHUNK;                                  \
            else if (out - 1 < ob)                                             \
                return EXR_ERR_OUT_OF_MEMORY;                                  \
                                                                               \
            s = out[-1];                                                       \
                                                                               \
            while (cs-- > 0)                                                   \
                *out++ = s;                                                    \
        }                                                                      \
        else if (out < oe) { *out++ = (uint16_t) po; }                         \
        else { return EXR_ERR_CORRUPT_CHUNK; }                                 \
    } while (0)

//
// Decode (uncompress) ni bits based on encoding & decoding tables:
//

static exr_result_t
hufDecode (
    const uint64_t* hcode,  // i : encoding table
    const HufDec*   hdecod, // i : decoding table
    const uint8_t*  in,     // i : compressed input buffer
    uint64_t        ni,     // i : input size (in bits)
    uint32_t        rlc,    // i : run-length code
    uint64_t        no,     // i : expected output size (count of uint16 items)
    uint16_t*       out)
{
    uint64_t       i;
    uint64_t       c    = 0;
    int            lc   = 0;
    uint16_t*      outb = out;
    uint16_t*      oe   = out + no;
    const uint8_t* ie   = in + (ni + 7) / 8; // input byte size

    //
    // Loop on input bytes
    //

    while (in < ie)
    {
        getChar (c, lc, in);

        //
        // Access decoding table
        //

        while (lc >= HUF_DECBITS)
        {
            uint64_t      decoffset = (c >> (lc - HUF_DECBITS)) & HUF_DECMASK;
            const HufDec* pl        = hdecod + decoffset;

            if (pl->len)
            {
                //
                // Get short code
                //

                if (pl->len > lc) return EXR_ERR_CORRUPT_CHUNK;

                lc -= pl->len;
                getCode (pl->lit, rlc, c, lc, in, ie, out, outb, oe);
            }
            else
            {
                uint32_t        j;
                const uint32_t* decbuf = pl->p;
                if (!pl->p) return EXR_ERR_CORRUPT_CHUNK; // wrong code

                //
                // Search long code
                //

                for (j = 0; j < pl->lit; j++)
                {
                    int l = hufLength (hcode[decbuf[j]]);

                    while (lc < l && in < ie) // get more bits
                    {
                        getChar (c, lc, in);
                    }

                    if (lc >= l)
                    {
                        if (hufCode (hcode[decbuf[j]]) ==
                            ((c >> (lc - l)) & (((uint64_t) (1) << l) - 1)))
                        {
                            //
                            // Found : get long code
                            //

                            lc -= l;
                            getCode (
                                decbuf[j], rlc, c, lc, in, ie, out, outb, oe);
                            break;
                        }
                    }
                }

                if (j == pl->lit) return EXR_ERR_CORRUPT_CHUNK;
            }
        }
    }

    //
    // Get remaining (short) codes
    //

    i = (8 - ni) & 7;
    c >>= i;
    lc -= (int) i;

    while (lc > 0)
    {
        uint64_t      decoffset = (c << (HUF_DECBITS - lc)) & HUF_DECMASK;
        const HufDec* pl        = hdecod + decoffset;

        if (pl->len)
        {
            if (pl->len > lc) return EXR_ERR_CORRUPT_CHUNK;
            lc -= pl->len;
            getCode (pl->lit, rlc, c, lc, in, ie, out, outb, oe);
        }
        else
            return EXR_ERR_CORRUPT_CHUNK;
    }

    if (out != oe) return EXR_ERR_OUT_OF_MEMORY;
    return EXR_ERR_SUCCESS;
}

static inline void
countFrequencies (uint64_t* freq, const uint16_t* data, uint64_t n)
{
    memset (freq, 0, HUF_ENCSIZE * sizeof (uint64_t));
    for (uint64_t i = 0; i < n; ++i)
        ++freq[data[i]];
}

static inline void
writeUInt (uint8_t* b, uint32_t i)
{
    b[0] = (uint8_t) (i);
    b[1] = (uint8_t) (i >> 8);
    b[2] = (uint8_t) (i >> 16);
    b[3] = (uint8_t) (i >> 24);
}

static inline uint32_t
readUInt (const uint8_t* b)
{
    return (
        ((uint32_t) b[0]) | (((uint32_t) b[1]) << 8u) |
        (((uint32_t) b[2]) << 16u) | (((uint32_t) b[3]) << 24u));
}

/**************************************/

// Longest compressed code length that ImfHuf supports (58 bits)
#define MAX_CODE_LEN 58

// Number of bits in our acceleration table. Should match all
// codes up to TABLE_LOOKUP_BITS in length.
#define TABLE_LOOKUP_BITS 14

#include <inttypes.h>

#ifdef __APPLE__
#    include <libkern/OSByteOrder.h>
#    define READ64(c) OSSwapInt64 (*(const uint64_t*) (c))
#elif defined(linux)
#    include <byteswap.h>
#    define READ64(c) bswap_64 (*(const uint64_t*) (c))
#elif defined(_MSC_VER)
#    include <stdlib.h>
#    define READ64(c) _byteswap_uint64 (*(const uint64_t*) (c))
#else
#    define READ64(c)                                                          \
        ((uint64_t) (c)[0] << 56) | ((uint64_t) (c)[1] << 48) |                \
            ((uint64_t) (c)[2] << 40) | ((uint64_t) (c)[3] << 32) |            \
            ((uint64_t) (c)[4] << 24) | ((uint64_t) (c)[5] << 16) |            \
            ((uint64_t) (c)[6] << 8) | ((uint64_t) (c)[7])
#endif

typedef struct FastHufDecoder
{
    // RLE symbol written by the encoder.
    // This could be 65536, so beware
    // when you use shorts to hold things.
    int _rleSymbol;

    // Number of symbols in the codebook.
    uint32_t _numSymbols;

    uint8_t _minCodeLength; // Minimum code length, in bits.
    uint8_t _maxCodeLength; // Maximum code length, in bits.
    uint8_t _pad[2];

    int _idToSymbol[65536 + 1]; // Maps Ids to symbols. Ids are a symbol
                                // ordering sorted first in terms of
                                // code length, and by code within
                                // the same length. Ids run from 0
                                // to mNumSymbols-1.

    uint64_t _ljBase[MAX_CODE_LEN + 1 + 1]; // the 'left justified base' table.
                                            // Takes base[i] (i = code length)
    // and 'left justifies' it into an uint64_t
    // Also includes a sentinel terminator

    uint64_t _ljOffset[MAX_CODE_LEN + 1]; // There are some other terms that can
        // be folded into constants when taking
        // the 'left justified' decode path. This
        // holds those constants, indexed by
        // code length

    //
    // We can accelerate the 'left justified' processing by running the
    // top TABLE_LOOKUP_BITS through a LUT, to find the symbol and code
    // length. These are those acceleration tables.
    //
    // Even though our eventual 'symbols' are ushort's, the encoder adds
    // a symbol to indicate RLE. So with a dense code book, we could
    // have 2^16+1 codes, hence 'symbol' could  be bigger than 16 bits.
    //
    int _lookupSymbol
        [1 << TABLE_LOOKUP_BITS]; /* value = (codeLen << 24) | symbol */

    uint64_t _tableMin;
} FastHufDecoder;

static exr_result_t
FastHufDecoder_buildTables (
    const struct _internal_exr_context* pctxt,
    FastHufDecoder*                     fhd,
    uint64_t*                           base,
    uint64_t*                           offset)
{
    int minIdx = TABLE_LOOKUP_BITS;

    //
    // Build the 'left justified' base table, by shifting base left..
    //

    for (int i = 0; i <= MAX_CODE_LEN; ++i)
    {
        if (base[i] != 0xffffffffffffffffULL)
        {
            fhd->_ljBase[i] = base[i] << (64 - i);
        }
        else
        {
            //
            // Unused code length - insert dummy values
            //

            fhd->_ljBase[i] = 0xffffffffffffffffULL;
        }
    }
    fhd->_ljBase[MAX_CODE_LEN + 1] = 0; /* sentinel for brute force lookup */

    //
    // Combine some terms into a big fat constant, which for
    // lack of a better term we'll call the 'left justified'
    // offset table (because it serves the same function
    // as 'offset', when using the left justified base table.
    //

    fhd->_ljOffset[0] = offset[0] - fhd->_ljBase[0];
    for (int i = 1; i <= MAX_CODE_LEN; ++i)
        fhd->_ljOffset[i] = offset[i] - (fhd->_ljBase[i] >> (64 - i));

    //
    // Build the acceleration tables for the lookups of
    // short codes ( <= TABLE_LOOKUP_BITS long)
    //

    for (uint64_t i = 0; i < 1 << TABLE_LOOKUP_BITS; ++i)
    {
        uint64_t value = i << (64 - TABLE_LOOKUP_BITS);

        fhd->_lookupSymbol[i] = 0xffff;

        for (int codeLen = fhd->_minCodeLength; codeLen <= fhd->_maxCodeLength;
             ++codeLen)
        {
            if (fhd->_ljBase[codeLen] <= value)
            {
                uint64_t id =
                    fhd->_ljOffset[codeLen] + (value >> (64 - codeLen));
                if (id < (uint64_t) (fhd->_numSymbols))
                {
                    fhd->_lookupSymbol[i] =
                        (fhd->_idToSymbol[id] | (codeLen << 24));
                }
                else
                {
                    if (pctxt)
                        pctxt->print_error (
                            pctxt,
                            EXR_ERR_CORRUPT_CHUNK,
                            "Huffman decode error (Overrun)");
                    return EXR_ERR_CORRUPT_CHUNK;
                }
                break;
            }
        }
    }

    //
    // Store the smallest value in the table that points to real data.
    // This should be the entry for the largest length that has
    // valid data (in our case, non-dummy _ljBase)
    //

    while (minIdx > 0 && fhd->_ljBase[minIdx] == 0xffffffffffffffffULL)
        minIdx--;

    if (minIdx < 0)
    {
        //
        // Error, no codes with lengths 0-TABLE_LOOKUP_BITS used.
        // Set the min value such that the table is never tested.
        //

        fhd->_tableMin = 0xffffffffffffffffULL;
    }
    else { fhd->_tableMin = fhd->_ljBase[minIdx]; }
    return EXR_ERR_SUCCESS;
}

static inline void
FastHufDecoder_refill (
    uint64_t*       buffer,
    int             numBits,           // number of bits to refill
    uint64_t*       bufferBack,        // the next 64-bits, to refill from
    int*            bufferBackNumBits, // number of bits left in bufferBack
    const uint8_t** currByte,          // current byte in the bitstream
    uint64_t*       currBitsLeft)
{
    //
    // Refill bits into the bottom of buffer, from the top of bufferBack.
    // Always top up buffer to be completely full.
    //

    *buffer |= (*bufferBack) >> (64 - numBits);

    if (*bufferBackNumBits < numBits)
    {
        numBits -= *bufferBackNumBits;

        //
        // Refill all of bufferBack from the bitstream. Either grab
        // a full 64-bit chunk, or whatever bytes are left. If we
        // don't have 64-bits left, pad with 0's.
        //

        if (*currBitsLeft >= 64)
        {
            *bufferBack        = READ64 (*currByte);
            *bufferBackNumBits = 64;
            *currByte += sizeof (uint64_t);
            *currBitsLeft -= 8 * sizeof (uint64_t);
        }
        else
        {
            uint64_t shift = 56;

            *bufferBack        = 0;
            *bufferBackNumBits = 64;

            while (*currBitsLeft >= 8)
            {
                *bufferBack |= ((uint64_t) (**currByte)) << shift;

                (*currByte)++;
                shift -= 8;
                *currBitsLeft -= 8;
            }

            if (*currBitsLeft > 0)
            {
                *bufferBack |= ((uint64_t) (**currByte)) << shift;

                (*currByte)++;
                shift -= 8;
                *currBitsLeft = 0;
            }
        }

        *buffer |= (*bufferBack) >> (64 - numBits);
    }

    //
    // We can have cases where the previous shift of bufferBack is << 64 -
    // this is an undefined operation but tends to create just zeroes.
    // so if we won't have any bits left, zero out bufferBack instead of computing the shift
    //

    if (*bufferBackNumBits <= numBits) { *bufferBack = 0; }
    else { *bufferBack = (*bufferBack) << numBits; }
    *bufferBackNumBits -= numBits;
}

static inline uint64_t
fasthuf_read_bits (
    int numBits, uint64_t* buffer, int* bufferNumBits, const uint8_t** currByte)
{
    while (*bufferNumBits < numBits)
    {
        *buffer = ((*buffer) << 8) | *((*currByte)++);
        *bufferNumBits += 8;
    }

    *bufferNumBits -= numBits;
    return ((*buffer) >> (*bufferNumBits)) & ((1 << numBits) - 1);
}

static exr_result_t
fasthuf_initialize (
    const struct _internal_exr_context* pctxt,
    FastHufDecoder*                     fhd,
    const uint8_t**                     table,
    uint64_t                            numBytes,
    uint32_t                            minSymbol,
    uint32_t                            maxSymbol,
    int                                 rleSymbol)
{
    //
    // The 'base' table is the minimum code at each code length. base[i]
    // is the smallest code (numerically) of length i.
    //

    uint64_t base[MAX_CODE_LEN + 1];

    //
    // The 'offset' table is the position (in sorted order) of the first id
    // of a given code length. Array is indexed by code length, like base.
    //

    uint64_t offset[MAX_CODE_LEN + 1];

    //
    // Count of how many codes at each length there are. Array is
    // indexed by code length, like base and offset.
    //

    size_t codeCount[MAX_CODE_LEN + 1];

    const uint8_t* currByte     = *table;
    uint64_t       currBits     = 0;
    int            currBitCount = 0;

    uint64_t       codeLen;
    const uint8_t* topByte = *table + numBytes;

    uint64_t mapping[MAX_CODE_LEN + 1];

    fhd->_rleSymbol     = rleSymbol;
    fhd->_numSymbols    = 0;
    fhd->_minCodeLength = 255;
    fhd->_maxCodeLength = 0;

    for (int i = 0; i <= MAX_CODE_LEN; ++i)
    {
        codeCount[i] = 0;
        base[i]      = 0xffffffffffffffffULL;
        offset[i]    = 0;
    }

    //
    // Count the number of codes, the min/max code lengths, the number of
    // codes with each length, and record symbols with non-zero code
    // length as we find them.
    //

    for (uint64_t symbol = (uint64_t) minSymbol; symbol <= (uint64_t) maxSymbol;
         symbol++)
    {
        if (currByte >= topByte)
        {
            if (pctxt)
                pctxt->print_error (
                    pctxt,
                    EXR_ERR_CORRUPT_CHUNK,
                    "Error decoding Huffman table (Truncated table data).");
            return EXR_ERR_CORRUPT_CHUNK;
        }

        //
        // Next code length - either:
        //       0-58  (literal code length)
        //       59-62 (various lengths runs of 0)
        //       63    (run of n 0's, with n is the next 8 bits)
        //

        codeLen = fasthuf_read_bits (6, &currBits, &currBitCount, &currByte);

        if (codeLen < (uint64_t) SHORT_ZEROCODE_RUN)
        {
            if (codeLen == 0) continue;

            if (codeLen < fhd->_minCodeLength)
                fhd->_minCodeLength = (uint8_t) codeLen;

            if (codeLen > fhd->_maxCodeLength)
                fhd->_maxCodeLength = (uint8_t) codeLen;

            codeCount[codeLen]++;
        }
        else if (codeLen == (uint64_t) LONG_ZEROCODE_RUN)
        {
            if (currByte >= topByte)
            {
                if (pctxt)
                    pctxt->print_error (
                        pctxt,
                        EXR_ERR_CORRUPT_CHUNK,
                        "Error decoding Huffman table (Truncated table data).");
                return EXR_ERR_CORRUPT_CHUNK;
            }

            symbol +=
                fasthuf_read_bits (8, &currBits, &currBitCount, &currByte) +
                SHORTEST_LONG_RUN - 1;
        }
        else
            symbol += codeLen - SHORT_ZEROCODE_RUN + 1;

        if (symbol > (uint64_t) maxSymbol)
        {
            if (pctxt)
                pctxt->print_error (
                    pctxt,
                    EXR_ERR_CORRUPT_CHUNK,
                    "Error decoding Huffman table (Run beyond end of table).");
            return EXR_ERR_CORRUPT_CHUNK;
        }
    }

    for (int i = 0; i < MAX_CODE_LEN; ++i)
        fhd->_numSymbols += (uint32_t) codeCount[i];

    if ((size_t) fhd->_numSymbols > sizeof (fhd->_idToSymbol) / sizeof (int))
    {
        if (pctxt)
            pctxt->print_error (
                pctxt,
                EXR_ERR_CORRUPT_CHUNK,
                "Error decoding Huffman table (Too many symbols).");
        return EXR_ERR_CORRUPT_CHUNK;
    }

    //
    // Compute base - once we have the code length counts, there
    //                is a closed form solution for this
    //

    {
        double* countTmp = (double*) offset; /* temp space */

        for (int l = fhd->_minCodeLength; l <= fhd->_maxCodeLength; ++l)
        {
            countTmp[l] = (double) codeCount[l] *
                          (double) (2ll << (fhd->_maxCodeLength - l));
        }

        for (int l = fhd->_minCodeLength; l <= fhd->_maxCodeLength; ++l)
        {
            double tmp = 0;

            for (int k = l + 1; k <= fhd->_maxCodeLength; ++k)
                tmp += countTmp[k];

            tmp /= (double) (2ll << (fhd->_maxCodeLength - l));

            base[l] = (uint64_t) (ceil (tmp));
        }
    }

    //
    // Compute offset - these are the positions of the first
    //                  id (not symbol) that has length [i]
    //

    offset[fhd->_maxCodeLength] = 0;

    for (int i = fhd->_maxCodeLength - 1; i >= fhd->_minCodeLength; i--)
        offset[i] = offset[i + 1] + codeCount[i + 1];

    //
    // Allocate and fill the symbol-to-id mapping. Smaller Ids should be
    // mapped to less-frequent symbols (which have longer codes). Use
    // the offset table to tell us where the id's for a given code
    // length start off.
    //

    for (int i = 0; i < MAX_CODE_LEN + 1; ++i)
        mapping[i] = (uint64_t) -1;
    for (int i = fhd->_minCodeLength; i <= fhd->_maxCodeLength; ++i)
        mapping[i] = offset[i];

    currByte     = *table;
    currBits     = 0;
    currBitCount = 0;

    //
    // Although we could have created an uncompressed list of symbols in our
    // decoding loop above, it's faster to decode the compressed data again
    //
    for (uint64_t symbol = (uint64_t) minSymbol; symbol <= (uint64_t) maxSymbol;
         symbol++)
    {
        codeLen = fasthuf_read_bits (6, &currBits, &currBitCount, &currByte);

        if (codeLen < (uint64_t) SHORT_ZEROCODE_RUN)
        {
            if (codeLen == 0) continue;

            if (mapping[codeLen] >= (uint64_t) fhd->_numSymbols)
            {
                if (pctxt)
                    pctxt->print_error (
                        pctxt,
                        EXR_ERR_CORRUPT_CHUNK,
                        "Huffman decode error (Invalid symbol in header)");
                return EXR_ERR_CORRUPT_CHUNK;
            }
            fhd->_idToSymbol[mapping[codeLen]] = (int) symbol;
            mapping[codeLen]++;
        }
        else if (codeLen == (uint64_t) LONG_ZEROCODE_RUN)
            symbol +=
                fasthuf_read_bits (8, &currBits, &currBitCount, &currByte) +
                SHORTEST_LONG_RUN - 1;
        else
            symbol += codeLen - SHORT_ZEROCODE_RUN + 1;
    }

    *table = currByte;

    return FastHufDecoder_buildTables (pctxt, fhd, base, offset);
}

static inline int
fasthuf_decode_enabled (void)
{
#if defined(__INTEL_COMPILER) || defined(__GNUC__)

    //
    // Enabled for ICC, GCC:
    //       __i386__   -> x86
    //       __x86_64__ -> 64-bit x86
    //       __e2k__    -> e2k (MCST Elbrus 2000)

#    if defined(__i386__) || defined(__x86_64__) || defined(__e2k__)
    return 1;
#    else
    return 0;
#    endif

#elif defined(_MSC_VER)

    //
    // Enabled for Visual Studio:
    //        _M_IX86 -> x86
    //        _M_X64  -> 64bit x86

#    if defined(_M_IX86) || defined(_M_X64)
    return 1;
#    else
    return 0;
#    endif

#else

    //
    // Unknown compiler - Be safe and disable.
    //
    return 0;
#endif
}

static exr_result_t
fasthuf_decode (
    const struct _internal_exr_context* pctxt,
    FastHufDecoder*                     fhd,
    const uint8_t*                      src,
    uint64_t                            numSrcBits,
    uint16_t*                           dst,
    uint64_t                            numDstElems)
{
    //
    // Current position (byte/bit) in the src data stream
    // (after the first buffer fill)
    //
    uint64_t             buffer, bufferBack, dstIdx;
    int                  bufferNumBits, bufferBackNumBits;
    const unsigned char* currByte = src + 2 * sizeof (uint64_t);

    numSrcBits -= 8 * 2 * (int) sizeof (uint64_t);

    //
    // 64-bit buffer holding the current bits in the stream
    //

    buffer        = READ64 (src);
    bufferNumBits = 64;

    //
    // 64-bit buffer holding the next bits in the stream
    //

    bufferBack        = READ64 ((src + sizeof (uint64_t)));
    bufferBackNumBits = 64;
    dstIdx            = 0;

    while (dstIdx < numDstElems)
    {
        int codeLen;
        int symbol;
        int rleCount;

        //
        // Test if we can be table accelerated. If so, directly
        // lookup the output symbol. Otherwise, we need to fall
        // back to searching for the code.
        //
        // If we're doing table lookups, we don't really need
        // a re-filled buffer, so long as we have TABLE_LOOKUP_BITS
        // left. But for a search, we do need a refilled table.
        //

        if (fhd->_tableMin <= buffer)
        {
            int tableIdx =
                fhd->_lookupSymbol[buffer >> (64 - TABLE_LOOKUP_BITS)];

            //
            // For invalid codes, _tableCodeLen[] should return 0. This
            // will cause the decoder to get stuck in the current spot
            // until we run out of elements, then barf that the codestream
            // is bad.  So we don't need to stick a condition like
            //     if (codeLen > _maxCodeLength) in this inner.
            //

            codeLen = tableIdx >> 24;
            symbol  = tableIdx & 0xffffff;
        }
        else
        {
            uint64_t id;
            //
            // Brute force search:
            // Find the smallest length where _ljBase[length] <= buffer
            //

            codeLen = TABLE_LOOKUP_BITS + 1;

            /* sentinel zero can never be greater than buffer */
            while (fhd->_ljBase[codeLen] >
                   buffer /* && codeLen <= _maxCodeLength */)
                codeLen++;

            if (codeLen > fhd->_maxCodeLength)
            {
                if (pctxt)
                    pctxt->print_error (
                        pctxt,
                        EXR_ERR_CORRUPT_CHUNK,
                        "Huffman decode error (Decoded an invalid symbol)");
                return EXR_ERR_CORRUPT_CHUNK;
            }

            id = fhd->_ljOffset[codeLen] + (buffer >> (64 - codeLen));
            if (id < (uint64_t) fhd->_numSymbols)
            {
                symbol = fhd->_idToSymbol[id];
            }
            else
            {
                if (pctxt)
                    pctxt->print_error (
                        pctxt,
                        EXR_ERR_CORRUPT_CHUNK,
                        "Huffman decode error (Decoded an invalid symbol)");
                return EXR_ERR_CORRUPT_CHUNK;
            }
        }

        //
        // Shift over bit stream, and update the bit count in the buffer
        //

        buffer = buffer << codeLen;
        bufferNumBits -= codeLen;

        //
        // If we received a RLE symbol (_rleSymbol), then we need
        // to read ahead 8 bits to know how many times to repeat
        // the previous symbol. Need to ensure we at least have
        // 8 bits of data in the buffer
        //

        if (symbol == fhd->_rleSymbol)
        {
            if (bufferNumBits < 8)
            {
                FastHufDecoder_refill (
                    &buffer,
                    64 - bufferNumBits,
                    &bufferBack,
                    &bufferBackNumBits,
                    &currByte,
                    &numSrcBits);

                bufferNumBits = 64;
            }

            rleCount = buffer >> 56;

            if (dstIdx < 1)
            {
                if (pctxt)
                    pctxt->print_error (
                        pctxt,
                        EXR_ERR_CORRUPT_CHUNK,
                        "Huffman decode error (RLE code with no previous symbol)");
                return EXR_ERR_CORRUPT_CHUNK;
            }

            if (dstIdx + (uint64_t) rleCount > numDstElems)
            {
                if (pctxt)
                    pctxt->print_error (
                        pctxt,
                        EXR_ERR_CORRUPT_CHUNK,
                        "Huffman decode error (Symbol run beyond expected output buffer length)");
                return EXR_ERR_CORRUPT_CHUNK;
            }

            if (rleCount <= 0)
            {
                if (pctxt)
                    pctxt->print_error (
                        pctxt,
                        EXR_ERR_CORRUPT_CHUNK,
                        "Huffman decode error (Invalid RLE length)");
                return EXR_ERR_CORRUPT_CHUNK;
            }

            for (int i = 0; i < rleCount; ++i)
                dst[dstIdx + (uint64_t) i] = dst[dstIdx - 1];

            dstIdx += (uint64_t) rleCount;

            buffer = buffer << 8;
            bufferNumBits -= 8;
        }
        else
        {
            dst[dstIdx] = (uint16_t) symbol;
            dstIdx++;
        }

        //
        // refill bit stream buffer if we're below the number of
        // bits needed for a table lookup
        //

        if (bufferNumBits < 64)
        {
            FastHufDecoder_refill (
                &buffer,
                64 - bufferNumBits,
                &bufferBack,
                &bufferBackNumBits,
                &currByte,
                &numSrcBits);

            bufferNumBits = 64;
        }
    }

    if (numSrcBits != 0)
    {
        if (pctxt)
            pctxt->print_error (
                pctxt,
                EXR_ERR_CORRUPT_CHUNK,
                "Huffman decode error (%d bits of compressed data remains after filling expected output buffer)",
                (int) numSrcBits);
        return EXR_ERR_CORRUPT_CHUNK;
    }

    return EXR_ERR_SUCCESS;
}

/**************************************/

uint64_t
internal_exr_huf_compress_spare_bytes (void)
{
    uint64_t ret = 0;
    ret += HUF_ENCSIZE * sizeof (uint64_t);  // freq
    ret += HUF_ENCSIZE * sizeof (uint64_t);  // scode
    ret += HUF_ENCSIZE * sizeof (uint64_t*); // fheap
    ret += HUF_ENCSIZE * sizeof (uint32_t);  // hlink
    return ret;
}

uint64_t
internal_exr_huf_decompress_spare_bytes (void)
{
    uint64_t ret = 0;
    ret += HUF_ENCSIZE * sizeof (uint64_t); // freq
    ret += HUF_DECSIZE * sizeof (HufDec);   // hdec
    //    ret += HUF_ENCSIZE * sizeof (uint64_t*); // fheap
    //    ret += HUF_ENCSIZE * sizeof (uint64_t);  // scode
    if (sizeof (FastHufDecoder) > ret) ret = sizeof (FastHufDecoder);
    return ret;
}

exr_result_t
internal_huf_compress (
    uint64_t*       encbytes,
    void*           out,
    uint64_t        outsz,
    const uint16_t* raw,
    uint64_t        nRaw,
    void*           spare,
    uint64_t        sparebytes)
{
    exr_result_t rv;
    uint64_t*    freq;
    uint32_t*    hlink;
    uint64_t**   fHeap;
    uint64_t*    scode;
    uint32_t     im = 0;
    uint32_t     iM = 0;
    uint32_t     tableLength, nBits, dataLength;
    uint8_t*     dataStart;
    uint8_t*     compressed = (uint8_t*) out;
    uint8_t*     tableStart = compressed + 20;
    uint8_t*     tableEnd   = tableStart;
    uint8_t*     maxcompout = compressed + outsz;

    if (nRaw == 0)
    {
        *encbytes = 0;
        return EXR_ERR_SUCCESS;
    }

    if (outsz < 20) return EXR_ERR_INVALID_ARGUMENT;
    if (sparebytes != internal_exr_huf_compress_spare_bytes ())
        return EXR_ERR_INVALID_ARGUMENT;

    freq  = (uint64_t*) spare;
    scode = freq + HUF_ENCSIZE;
    fHeap = (uint64_t**) (scode + HUF_ENCSIZE);
    hlink = (uint32_t*) (fHeap + HUF_ENCSIZE);

    countFrequencies (freq, raw, nRaw);

    hufBuildEncTable (freq, &im, &iM, hlink, fHeap, scode);

    rv = hufPackEncTable (freq, im, iM, &tableEnd, maxcompout);

    if (rv != EXR_ERR_SUCCESS) return rv;
    tableLength =
        (uint32_t) (((uintptr_t) tableEnd) - ((uintptr_t) tableStart));
    dataStart = tableEnd;

    rv = hufEncode (freq, raw, nRaw, iM, dataStart, maxcompout, &nBits);
    if (rv != EXR_ERR_SUCCESS) return rv;

    dataLength = (nBits + 7) / 8;

    writeUInt (compressed, im);
    writeUInt (compressed + 4, iM);
    writeUInt (compressed + 8, tableLength);
    writeUInt (compressed + 12, nBits);
    writeUInt (compressed + 16, 0); // room for future extensions

    *encbytes =
        (((uintptr_t) dataStart) + ((uintptr_t) dataLength) -
         ((uintptr_t) compressed));
    return EXR_ERR_SUCCESS;
}

exr_result_t
internal_huf_decompress (
    exr_decode_pipeline_t* decode,
    const uint8_t*         compressed,
    uint64_t               nCompressed,
    uint16_t*              raw,
    uint64_t               nRaw,
    void*                  spare,
    uint64_t               sparebytes)
{
    uint32_t                            im, iM, nBits;
    uint64_t                            nBytes;
    const uint8_t*                      ptr;
    exr_result_t                        rv;
    const struct _internal_exr_context* pctxt = NULL;
    const uint64_t hufInfoBlockSize           = 5 * sizeof (uint32_t);

    if (decode) pctxt = EXR_CCTXT (decode->context);
    //
    // need at least 20 bytes for header
    //
    if (nCompressed < 20)
    {
        if (nRaw != 0) return EXR_ERR_INVALID_ARGUMENT;
        return EXR_ERR_SUCCESS;
    }

    if (sparebytes != internal_exr_huf_decompress_spare_bytes ())
        return EXR_ERR_INVALID_ARGUMENT;

    im = readUInt (compressed);
    iM = readUInt (compressed + 4);
    // uint32_t tableLength = readUInt (compressed + 8);
    nBits = readUInt (compressed + 12);
    // uint32_t future = readUInt (compressed + 16);

    if (im >= HUF_ENCSIZE || iM >= HUF_ENCSIZE) return EXR_ERR_CORRUPT_CHUNK;

    ptr = compressed + hufInfoBlockSize;

    nBytes = (((uint64_t) (nBits) + 7)) / 8;

    // must be nBytes remaining in buffer
    if (hufInfoBlockSize + nBytes > nCompressed) return EXR_ERR_OUT_OF_MEMORY;

    //
    // Fast decoder needs at least 2x64-bits of compressed data, and
    // needs to be run-able on this platform. Otherwise, fall back
    // to the original decoder
    //
    if (fasthuf_decode_enabled () && nBits > 128)
    {
        FastHufDecoder* fhd = (FastHufDecoder*) spare;

        rv = fasthuf_initialize (
            pctxt, fhd, &ptr, nCompressed - hufInfoBlockSize, im, iM, (int) iM);
        if (rv == EXR_ERR_SUCCESS)
        {
            if ((uint64_t) (ptr - compressed) + nBytes > nCompressed)
                return EXR_ERR_OUT_OF_MEMORY;
            rv = fasthuf_decode (pctxt, fhd, ptr, nBits, raw, nRaw);
        }
    }
    else
    {
        uint64_t* freq  = (uint64_t*) spare;
        HufDec*   hdec  = (HufDec*) (freq + HUF_ENCSIZE);
        uint64_t  nLeft = nCompressed - 20;

        hufClearDecTable (hdec);
        hufUnpackEncTable (&ptr, &nLeft, im, iM, freq);

        if (nBits > 8 * nLeft) return EXR_ERR_CORRUPT_CHUNK;

        rv = hufBuildDecTable (pctxt, freq, im, iM, hdec);
        if (rv == EXR_ERR_SUCCESS)
            rv = hufDecode (freq, hdec, ptr, nBits, iM, nRaw, raw);

        hufFreeDecTable (pctxt, hdec);
    }
    return rv;
}
