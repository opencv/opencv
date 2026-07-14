#ifndef DFLTCC_INFLATE_H
#define DFLTCC_INFLATE_H

#include "dfltcc_common.h"

void Z_INTERNAL PREFIX(dfltcc_reset_inflate_state)(PREFIX3(streamp) strm);
int Z_INTERNAL PREFIX(dfltcc_can_inflate)(PREFIX3(streamp) strm);
typedef enum {
    DFLTCC_INFLATE_CONTINUE,
    DFLTCC_INFLATE_BREAK,
    DFLTCC_INFLATE_SOFTWARE,
} dfltcc_inflate_action;
dfltcc_inflate_action Z_INTERNAL PREFIX(dfltcc_inflate)(PREFIX3(streamp) strm, int flush, int *ret);
int Z_INTERNAL PREFIX(dfltcc_was_inflate_used)(PREFIX3(streamp) strm);
int Z_INTERNAL PREFIX(dfltcc_inflate_disable)(PREFIX3(streamp) strm);
int Z_INTERNAL PREFIX(dfltcc_inflate_set_dictionary)(PREFIX3(streamp) strm,
                                                     const unsigned char *dictionary, uInt dict_length);
int Z_INTERNAL PREFIX(dfltcc_inflate_get_dictionary)(PREFIX3(streamp) strm,
                                                     unsigned char *dictionary, uInt* dict_length);

#define INFLATE_RESET_KEEP_HOOK PREFIX(dfltcc_reset_inflate_state)

#define INFLATE_PRIME_HOOK(strm, bits, value) \
    do { if (PREFIX(dfltcc_inflate_disable)((strm))) return Z_STREAM_ERROR; } while (0)

#define INFLATE_TYPEDO_HOOK(strm, flush) \
    if (PREFIX(dfltcc_can_inflate)((strm))) { \
        dfltcc_inflate_action action; \
\
        RESTORE(); \
        action = PREFIX(dfltcc_inflate)((strm), (flush), &ret); \
        LOAD(); \
        if (action == DFLTCC_INFLATE_CONTINUE) \
            break; \
        else if (action == DFLTCC_INFLATE_BREAK) \
            goto inf_leave; \
    }

#define INFLATE_NEED_CHECKSUM(strm) (!PREFIX(dfltcc_can_inflate)((strm)))

#define INFLATE_NEED_UPDATEWINDOW(strm) (!PREFIX(dfltcc_can_inflate)((strm)))

#define INFLATE_MARK_HOOK(strm) \
    do { \
        if (PREFIX(dfltcc_was_inflate_used)((strm))) return -(1L << 16); \
    } while (0)

#define INFLATE_SYNC_POINT_HOOK(strm) \
    do { \
        if (PREFIX(dfltcc_was_inflate_used)((strm))) return Z_STREAM_ERROR; \
    } while (0)

#define INFLATE_SET_DICTIONARY_HOOK(strm, dict, dict_len) \
    do { \
        if (PREFIX(dfltcc_can_inflate)((strm))) \
            return PREFIX(dfltcc_inflate_set_dictionary)((strm), (dict), (dict_len)); \
    } while (0)

#define INFLATE_GET_DICTIONARY_HOOK(strm, dict, dict_len) \
    do { \
        if (PREFIX(dfltcc_can_inflate)((strm))) \
            return PREFIX(dfltcc_inflate_get_dictionary)((strm), (dict), (dict_len)); \
    } while (0)

#define INFLATE_ADJUST_WINDOW_SIZE(n) MAX(n, HB_SIZE)

#endif
