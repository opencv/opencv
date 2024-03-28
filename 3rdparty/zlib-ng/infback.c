/* infback.c -- inflate using a call-back interface
 * Copyright (C) 1995-2022 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/*
   This code is largely copied from inflate.c.  Normally either infback.o or
   inflate.o would be linked into an application--not both.  The interface
   with inffast.c is retained so that optimized assembler-coded versions of
   inflate_fast() can be used with either inflate.c or infback.c.
 */

#include "zbuild.h"
#include "zutil.h"
#include "inftrees.h"
#include "inflate.h"
#include "inflate_p.h"
#include "functable.h"

/* Avoid conflicts with zlib.h macros */
#ifdef ZLIB_COMPAT
# undef inflateBackInit
#endif

/*
   strm provides memory allocation functions in zalloc and zfree, or
   NULL to use the library memory allocation functions.

   windowBits is in the range 8..15, and window is a user-supplied
   window and output buffer that is 2**windowBits bytes.

   This function is hidden in ZLIB_COMPAT builds.
 */
int32_t ZNG_CONDEXPORT PREFIX(inflateBackInit)(PREFIX3(stream) *strm, int32_t windowBits, uint8_t *window) {
    struct inflate_state *state;

    if (strm == NULL || window == NULL || windowBits < MIN_WBITS || windowBits > MAX_WBITS)
        return Z_STREAM_ERROR;
    strm->msg = NULL;                   /* in case we return an error */
    if (strm->zalloc == NULL) {
        strm->zalloc = PREFIX(zcalloc);
        strm->opaque = NULL;
    }
    if (strm->zfree == NULL)
        strm->zfree = PREFIX(zcfree);
    state = ZALLOC_INFLATE_STATE(strm);
    if (state == NULL)
        return Z_MEM_ERROR;
    Tracev((stderr, "inflate: allocated\n"));
    strm->state = (struct internal_state *)state;
    state->dmax = 32768U;
    state->wbits = (unsigned int)windowBits;
    state->wsize = 1U << windowBits;
    state->window = window;
    state->wnext = 0;
    state->whave = 0;
    state->sane = 1;
    state->chunksize = functable.chunksize();
    return Z_OK;
}

/* Function used by zlib.h and zlib-ng version 2.0 macros */
int32_t Z_EXPORT PREFIX(inflateBackInit_)(PREFIX3(stream) *strm, int32_t windowBits, uint8_t *window,
                              const char *version, int32_t stream_size) {
    if (CHECK_VER_STSIZE(version, stream_size))
        return Z_VERSION_ERROR;
    return PREFIX(inflateBackInit)(strm, windowBits, window);
}

/*
   Private macros for inflateBack()
   Look in inflate_p.h for macros shared with inflate()
*/

/* Assure that some input is available.  If input is requested, but denied,
   then return a Z_BUF_ERROR from inflateBack(). */
#define PULL() \
    do { \
        if (have == 0) { \
            have = in(in_desc, &next); \
            if (have == 0) { \
                next = NULL; \
                ret = Z_BUF_ERROR; \
                goto inf_leave; \
            } \
        } \
    } while (0)

/* Get a byte of input into the bit accumulator, or return from inflateBack()
   with an error if there is no input available. */
#define PULLBYTE() \
    do { \
        PULL(); \
        have--; \
        hold += ((unsigned)(*next++) << bits); \
        bits += 8; \
    } while (0)

/* Assure that some output space is available, by writing out the window
   if it's full.  If the write fails, return from inflateBack() with a
   Z_BUF_ERROR. */
#define ROOM() \
    do { \
        if (left == 0) { \
            put = state->window; \
            left = state->wsize; \
            state->whave = left; \
            if (out(out_desc, put, left)) { \
                ret = Z_BUF_ERROR; \
                goto inf_leave; \
            } \
        } \
    } while (0)

/*
   strm provides the memory allocation functions and window buffer on input,
   and provides information on the unused input on return.  For Z_DATA_ERROR
   returns, strm will also provide an error message.

   in() and out() are the call-back input and output functions.  When
   inflateBack() needs more input, it calls in().  When inflateBack() has
   filled the window with output, or when it completes with data in the
   window, it calls out() to write out the data.  The application must not
   change the provided input until in() is called again or inflateBack()
   returns.  The application must not change the window/output buffer until
   inflateBack() returns.

   in() and out() are called with a descriptor parameter provided in the
   inflateBack() call.  This parameter can be a structure that provides the
   information required to do the read or write, as well as accumulated
   information on the input and output such as totals and check values.

   in() should return zero on failure.  out() should return non-zero on
   failure.  If either in() or out() fails, than inflateBack() returns a
   Z_BUF_ERROR.  strm->next_in can be checked for NULL to see whether it
   was in() or out() that caused in the error.  Otherwise, inflateBack()
   returns Z_STREAM_END on success, Z_DATA_ERROR for an deflate format
   error, or Z_MEM_ERROR if it could not allocate memory for the state.
   inflateBack() can also return Z_STREAM_ERROR if the input parameters
   are not correct, i.e. strm is NULL or the state was not initialized.
 */
int32_t Z_EXPORT PREFIX(inflateBack)(PREFIX3(stream) *strm, in_func in, void *in_desc, out_func out, void *out_desc) {
    struct inflate_state *state;
    z_const unsigned char *next; /* next input */
    unsigned char *put;          /* next output */
    unsigned have, left;         /* available input and output */
    uint32_t hold;               /* bit buffer */
    unsigned bits;               /* bits in bit buffer */
    unsigned copy;               /* number of stored or match bytes to copy */
    unsigned char *from;         /* where to copy match bytes from */
    code here;                   /* current decoding table entry */
    code last;                   /* parent table entry */
    unsigned len;                /* length to copy for repeats, bits to drop */
    int32_t ret;                 /* return code */
    static const uint16_t order[19] = /* permutation of code lengths */
        {16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

    /* Check that the strm exists and that the state was initialized */
    if (strm == NULL || strm->state == NULL)
        return Z_STREAM_ERROR;
    state = (struct inflate_state *)strm->state;

    /* Reset the state */
    strm->msg = NULL;
    state->mode = TYPE;
    state->last = 0;
    state->whave = 0;
    next = strm->next_in;
    have = next != NULL ? strm->avail_in : 0;
    hold = 0;
    bits = 0;
    put = state->window;
    left = state->wsize;

    /* Inflate until end of block marked as last */
    for (;;)
        switch (state->mode) {
        case TYPE:
            /* determine and dispatch block type */
            if (state->last) {
                BYTEBITS();
                state->mode = DONE;
                break;
            }
            NEEDBITS(3);
            state->last = BITS(1);
            DROPBITS(1);
            switch (BITS(2)) {
            case 0:                             /* stored block */
                Tracev((stderr, "inflate:     stored block%s\n", state->last ? " (last)" : ""));
                state->mode = STORED;
                break;
            case 1:                             /* fixed block */
                PREFIX(fixedtables)(state);
                Tracev((stderr, "inflate:     fixed codes block%s\n", state->last ? " (last)" : ""));
                state->mode = LEN;              /* decode codes */
                break;
            case 2:                             /* dynamic block */
                Tracev((stderr, "inflate:     dynamic codes block%s\n", state->last ? " (last)" : ""));
                state->mode = TABLE;
                break;
            case 3:
                SET_BAD("invalid block type");
            }
            DROPBITS(2);
            break;

        case STORED:
            /* get and verify stored block length */
            BYTEBITS();                         /* go to byte boundary */
            NEEDBITS(32);
            if ((hold & 0xffff) != ((hold >> 16) ^ 0xffff)) {
                SET_BAD("invalid stored block lengths");
                break;
            }
            state->length = (uint16_t)hold;
            Tracev((stderr, "inflate:       stored length %u\n", state->length));
            INITBITS();

            /* copy stored block from input to output */
            while (state->length != 0) {
                copy = state->length;
                PULL();
                ROOM();
                copy = MIN(copy, have);
                copy = MIN(copy, left);
                memcpy(put, next, copy);
                have -= copy;
                next += copy;
                left -= copy;
                put += copy;
                state->length -= copy;
            }
            Tracev((stderr, "inflate:       stored end\n"));
            state->mode = TYPE;
            break;

        case TABLE:
            /* get dynamic table entries descriptor */
            NEEDBITS(14);
            state->nlen = BITS(5) + 257;
            DROPBITS(5);
            state->ndist = BITS(5) + 1;
            DROPBITS(5);
            state->ncode = BITS(4) + 4;
            DROPBITS(4);
#ifndef PKZIP_BUG_WORKAROUND
            if (state->nlen > 286 || state->ndist > 30) {
                SET_BAD("too many length or distance symbols");
                break;
            }
#endif
            Tracev((stderr, "inflate:       table sizes ok\n"));
            state->have = 0;

            /* get code length code lengths (not a typo) */
            while (state->have < state->ncode) {
                NEEDBITS(3);
                state->lens[order[state->have++]] = (uint16_t)BITS(3);
                DROPBITS(3);
            }
            while (state->have < 19)
                state->lens[order[state->have++]] = 0;
            state->next = state->codes;
            state->lencode = (const code *)(state->next);
            state->lenbits = 7;
            ret = zng_inflate_table(CODES, state->lens, 19, &(state->next), &(state->lenbits), state->work);
            if (ret) {
                SET_BAD("invalid code lengths set");
                break;
            }
            Tracev((stderr, "inflate:       code lengths ok\n"));
            state->have = 0;

            /* get length and distance code code lengths */
            while (state->have < state->nlen + state->ndist) {
                for (;;) {
                    here = state->lencode[BITS(state->lenbits)];
                    if (here.bits <= bits) break;
                    PULLBYTE();
                }
                if (here.val < 16) {
                    DROPBITS(here.bits);
                    state->lens[state->have++] = here.val;
                } else {
                    if (here.val == 16) {
                        NEEDBITS(here.bits + 2);
                        DROPBITS(here.bits);
                        if (state->have == 0) {
                            SET_BAD("invalid bit length repeat");
                            break;
                        }
                        len = state->lens[state->have - 1];
                        copy = 3 + BITS(2);
                        DROPBITS(2);
                    } else if (here.val == 17) {
                        NEEDBITS(here.bits + 3);
                        DROPBITS(here.bits);
                        len = 0;
                        copy = 3 + BITS(3);
                        DROPBITS(3);
                    } else {
                        NEEDBITS(here.bits + 7);
                        DROPBITS(here.bits);
                        len = 0;
                        copy = 11 + BITS(7);
                        DROPBITS(7);
                    }
                    if (state->have + copy > state->nlen + state->ndist) {
                        SET_BAD("invalid bit length repeat");
                        break;
                    }
                    while (copy) {
                        --copy;
                        state->lens[state->have++] = (uint16_t)len;
                    }
                }
            }

            /* handle error breaks in while */
            if (state->mode == BAD)
                break;

            /* check for end-of-block code (better have one) */
            if (state->lens[256] == 0) {
                SET_BAD("invalid code -- missing end-of-block");
                break;
            }

            /* build code tables -- note: do not change the lenbits or distbits
               values here (10 and 9) without reading the comments in inftrees.h
               concerning the ENOUGH constants, which depend on those values */
            state->next = state->codes;
            state->lencode = (const code *)(state->next);
            state->lenbits = 10;
            ret = zng_inflate_table(LENS, state->lens, state->nlen, &(state->next), &(state->lenbits), state->work);
            if (ret) {
                SET_BAD("invalid literal/lengths set");
                break;
            }
            state->distcode = (const code *)(state->next);
            state->distbits = 9;
            ret = zng_inflate_table(DISTS, state->lens + state->nlen, state->ndist,
                                &(state->next), &(state->distbits), state->work);
            if (ret) {
                SET_BAD("invalid distances set");
                break;
            }
            Tracev((stderr, "inflate:       codes ok\n"));
            state->mode = LEN;
            Z_FALLTHROUGH;

        case LEN:
            /* use inflate_fast() if we have enough input and output */
            if (have >= INFLATE_FAST_MIN_HAVE &&
                left >= INFLATE_FAST_MIN_LEFT) {
                RESTORE();
                if (state->whave < state->wsize)
                    state->whave = state->wsize - left;
                functable.inflate_fast(strm, state->wsize);
                LOAD();
                break;
            }

            /* get a literal, length, or end-of-block code */
            for (;;) {
                here = state->lencode[BITS(state->lenbits)];
                if (here.bits <= bits)
                    break;
                PULLBYTE();
            }
            if (here.op && (here.op & 0xf0) == 0) {
                last = here;
                for (;;) {
                    here = state->lencode[last.val + (BITS(last.bits + last.op) >> last.bits)];
                    if ((unsigned)last.bits + (unsigned)here.bits <= bits)
                        break;
                    PULLBYTE();
                }
                DROPBITS(last.bits);
            }
            DROPBITS(here.bits);
            state->length = here.val;

            /* process literal */
            if ((int)(here.op) == 0) {
                Tracevv((stderr, here.val >= 0x20 && here.val < 0x7f ?
                        "inflate:         literal '%c'\n" :
                        "inflate:         literal 0x%02x\n", here.val));
                ROOM();
                *put++ = (unsigned char)(state->length);
                left--;
                state->mode = LEN;
                break;
            }

            /* process end of block */
            if (here.op & 32) {
                Tracevv((stderr, "inflate:         end of block\n"));
                state->mode = TYPE;
                break;
            }

            /* invalid code */
            if (here.op & 64) {
                SET_BAD("invalid literal/length code");
                break;
            }

            /* length code -- get extra bits, if any */
            state->extra = (here.op & MAX_BITS);
            if (state->extra) {
                NEEDBITS(state->extra);
                state->length += BITS(state->extra);
                DROPBITS(state->extra);
            }
            Tracevv((stderr, "inflate:         length %u\n", state->length));

            /* get distance code */
            for (;;) {
                here = state->distcode[BITS(state->distbits)];
                if (here.bits <= bits)
                    break;
                PULLBYTE();
            }
            if ((here.op & 0xf0) == 0) {
                last = here;
                for (;;) {
                    here = state->distcode[last.val + (BITS(last.bits + last.op) >> last.bits)];
                    if ((unsigned)last.bits + (unsigned)here.bits <= bits)
                        break;
                    PULLBYTE();
                }
                DROPBITS(last.bits);
            }
            DROPBITS(here.bits);
            if (here.op & 64) {
                SET_BAD("invalid distance code");
                break;
            }
            state->offset = here.val;
            state->extra = (here.op & MAX_BITS);

            /* get distance extra bits, if any */
            if (state->extra) {
                NEEDBITS(state->extra);
                state->offset += BITS(state->extra);
                DROPBITS(state->extra);
            }
#ifdef INFLATE_STRICT
            if (state->offset > state->wsize - (state->whave < state->wsize ? left : 0)) {
                SET_BAD("invalid distance too far back");
                break;
            }
#endif
            Tracevv((stderr, "inflate:         distance %u\n", state->offset));

            /* copy match from window to output */
            do {
                ROOM();
                copy = state->wsize - state->offset;
                if (copy < left) {
                    from = put + copy;
                    copy = left - copy;
                } else {
                    from = put - state->offset;
                    copy = left;
                }
                copy = MIN(copy, state->length);
                state->length -= copy;
                left -= copy;
                do {
                    *put++ = *from++;
                } while (--copy);
            } while (state->length != 0);
            break;

        case DONE:
            /* inflate stream terminated properly */
            ret = Z_STREAM_END;
            goto inf_leave;

        case BAD:
            ret = Z_DATA_ERROR;
            goto inf_leave;

        default:                /* can't happen, but makes compilers happy */
            ret = Z_STREAM_ERROR;
            goto inf_leave;
        }

    /* Write leftover output and return unused input */
  inf_leave:
    if (left < state->wsize) {
        if (out(out_desc, state->window, state->wsize - left) && (ret == Z_STREAM_END)) {
            ret = Z_BUF_ERROR;
        }
    }
    strm->next_in = next;
    strm->avail_in = have;
    return ret;
}

int32_t Z_EXPORT PREFIX(inflateBackEnd)(PREFIX3(stream) *strm) {
    if (strm == NULL || strm->state == NULL || strm->zfree == NULL)
        return Z_STREAM_ERROR;
    ZFREE_STATE(strm, strm->state);
    strm->state = NULL;
    Tracev((stderr, "inflate: end\n"));
    return Z_OK;
}
