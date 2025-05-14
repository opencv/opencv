/* inflate.h -- internal inflate state definition
 * Copyright (C) 1995-2019 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the compression library and is
   subject to change. Applications should only use zlib.h.
 */

#ifndef INFLATE_H_
#define INFLATE_H_

#include "crc32.h"

#ifdef S390_DFLTCC_INFLATE
#  include "arch/s390/dfltcc_common.h"
#  define HAVE_ARCH_INFLATE_STATE
#endif

/* define NO_GZIP when compiling if you want to disable gzip header and trailer decoding by inflate().
   NO_GZIP would be used to avoid linking in the crc code when it is not needed.
   For shared libraries, gzip decoding should be left enabled. */
#ifndef NO_GZIP
#  define GUNZIP
#endif

/* Possible inflate modes between inflate() calls */
typedef enum {
    HEAD = 16180,   /* i: waiting for magic header */
    FLAGS,      /* i: waiting for method and flags (gzip) */
    TIME,       /* i: waiting for modification time (gzip) */
    OS,         /* i: waiting for extra flags and operating system (gzip) */
    EXLEN,      /* i: waiting for extra length (gzip) */
    EXTRA,      /* i: waiting for extra bytes (gzip) */
    NAME,       /* i: waiting for end of file name (gzip) */
    COMMENT,    /* i: waiting for end of comment (gzip) */
    HCRC,       /* i: waiting for header crc (gzip) */
    DICTID,     /* i: waiting for dictionary check value */
    DICT,       /* waiting for inflateSetDictionary() call */
        TYPE,       /* i: waiting for type bits, including last-flag bit */
        TYPEDO,     /* i: same, but skip check to exit inflate on new block */
        STORED,     /* i: waiting for stored size (length and complement) */
        COPY_,      /* i/o: same as COPY below, but only first time in */
        COPY,       /* i/o: waiting for input or output to copy stored block */
        TABLE,      /* i: waiting for dynamic block table lengths */
        LENLENS,    /* i: waiting for code length code lengths */
        CODELENS,   /* i: waiting for length/lit and distance code lengths */
            LEN_,       /* i: same as LEN below, but only first time in */
            LEN,        /* i: waiting for length/lit/eob code */
            LENEXT,     /* i: waiting for length extra bits */
            DIST,       /* i: waiting for distance code */
            DISTEXT,    /* i: waiting for distance extra bits */
            MATCH,      /* o: waiting for output space to copy string */
            LIT,        /* o: waiting for output space to write literal */
    CHECK,      /* i: waiting for 32-bit check value */
    LENGTH,     /* i: waiting for 32-bit length (gzip) */
    DONE,       /* finished check, done -- remain here until reset */
    BAD,        /* got a data error -- remain here until reset */
    SYNC        /* looking for synchronization bytes to restart inflate() */
} inflate_mode;

/*
    State transitions between above modes -

    (most modes can go to BAD on error -- not shown for clarity)

    Process header:
        HEAD -> (gzip) or (zlib) or (raw)
        (gzip) -> FLAGS -> TIME -> OS -> EXLEN -> EXTRA -> NAME -> COMMENT ->
                  HCRC -> TYPE
        (zlib) -> DICTID or TYPE
        DICTID -> DICT -> TYPE
        (raw) -> TYPEDO
    Read deflate blocks:
            TYPE -> TYPEDO -> STORED or TABLE or LEN_ or CHECK
            STORED -> COPY_ -> COPY -> TYPE
            TABLE -> LENLENS -> CODELENS -> LEN_
            LEN_ -> LEN
    Read deflate codes in fixed or dynamic block:
                LEN -> LENEXT or LIT or TYPE
                LENEXT -> DIST -> DISTEXT -> MATCH -> LEN
                LIT -> LEN
    Process trailer:
        CHECK -> LENGTH -> DONE
 */
typedef struct inflate_state inflate_state;

/* Struct for memory allocation handling */
typedef struct inflate_allocs_s {
    char            *buf_start;
    free_func        zfree;
    inflate_state   *state;
    unsigned char   *window;
} inflate_allocs;

/* State maintained between inflate() calls -- approximately 7K bytes, not
   including the allocated sliding window, which is up to 32K bytes. */
struct ALIGNED_(64) inflate_state {
    PREFIX3(stream) *strm;             /* pointer back to this zlib stream */
    inflate_mode mode;          /* current inflate mode */
    int last;                   /* true if processing last block */
    int wrap;                   /* bit 0 true for zlib, bit 1 true for gzip,
                                   bit 2 true to validate check value */
    int havedict;               /* true if dictionary provided */
    int flags;                  /* gzip header method and flags, 0 if zlib, or
                                   -1 if raw or no header yet */
    unsigned dmax;              /* zlib header max distance (INFLATE_STRICT) */
    unsigned long check;        /* protected copy of check value */
    unsigned long total;        /* protected copy of output count */
    PREFIX(gz_headerp) head;    /* where to save gzip header information */
        /* sliding window */
    unsigned wbits;             /* log base 2 of requested window size */
    uint32_t wsize;             /* window size or zero if not using window */
    uint32_t whave;             /* valid bytes in the window */
    uint32_t wnext;             /* window write index */
    unsigned char *window;      /* allocated sliding window, if needed */

    struct crc32_fold_s ALIGNED_(16) crc_fold;

        /* bit accumulator */
    uint32_t hold;              /* input bit accumulator */
    unsigned bits;              /* number of bits in "in" */
        /* for string and stored block copying */
    uint32_t length;            /* literal or length of data to copy */
    unsigned offset;            /* distance back to copy string from */
        /* for table and code decoding */
    unsigned extra;             /* extra bits needed */
        /* fixed and dynamic code tables */
    code const *lencode;        /* starting table for length/literal codes */
    code const *distcode;       /* starting table for distance codes */
    unsigned lenbits;           /* index bits for lencode */
    unsigned distbits;          /* index bits for distcode */
        /* dynamic table building */
    unsigned ncode;             /* number of code length code lengths */
    unsigned nlen;              /* number of length code lengths */
    unsigned ndist;             /* number of distance code lengths */
    uint32_t have;              /* number of code lengths in lens[] */
    code *next;                 /* next available space in codes[] */
    uint16_t lens[320];         /* temporary storage for code lengths */
    uint16_t work[288];         /* work area for code table building */
    code codes[ENOUGH];         /* space for code tables */
    int sane;                   /* if false, allow invalid distance too far */
    int back;                   /* bits back of last unprocessed length/lit */
    unsigned was;               /* initial length of match */
    uint32_t chunksize;         /* size of memory copying chunk */
    inflate_allocs *alloc_bufs; /* struct for handling memory allocations */
#ifdef HAVE_ARCH_INFLATE_STATE
    arch_inflate_state arch;    /* architecture-specific extensions */
#endif
};

void Z_INTERNAL PREFIX(fixedtables)(struct inflate_state *state);
Z_INTERNAL inflate_allocs* alloc_inflate(PREFIX3(stream) *strm);
Z_INTERNAL void free_inflate(PREFIX3(stream) *strm);

#endif /* INFLATE_H_ */
