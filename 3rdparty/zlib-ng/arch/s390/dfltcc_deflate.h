#ifndef DFLTCC_DEFLATE_H
#define DFLTCC_DEFLATE_H

#include "deflate.h"
#include "dfltcc_common.h"

void Z_INTERNAL PREFIX(dfltcc_reset_deflate_state)(PREFIX3(streamp));
int Z_INTERNAL PREFIX(dfltcc_can_deflate)(PREFIX3(streamp) strm);
int Z_INTERNAL PREFIX(dfltcc_deflate)(PREFIX3(streamp) strm, int flush, block_state *result);
int Z_INTERNAL PREFIX(dfltcc_deflate_params)(PREFIX3(streamp) strm, int level, int strategy, int *flush);
int Z_INTERNAL PREFIX(dfltcc_deflate_done)(PREFIX3(streamp) strm, int flush);
int Z_INTERNAL PREFIX(dfltcc_can_set_reproducible)(PREFIX3(streamp) strm, int reproducible);
int Z_INTERNAL PREFIX(dfltcc_deflate_set_dictionary)(PREFIX3(streamp) strm,
                                                const unsigned char *dictionary, uInt dict_length);
int Z_INTERNAL PREFIX(dfltcc_deflate_get_dictionary)(PREFIX3(streamp) strm, unsigned char *dictionary, uInt* dict_length);

#define DEFLATE_SET_DICTIONARY_HOOK(strm, dict, dict_len) \
    do { \
        if (PREFIX(dfltcc_can_deflate)((strm))) \
            return PREFIX(dfltcc_deflate_set_dictionary)((strm), (dict), (dict_len)); \
    } while (0)

#define DEFLATE_GET_DICTIONARY_HOOK(strm, dict, dict_len) \
    do { \
        if (PREFIX(dfltcc_can_deflate)((strm))) \
            return PREFIX(dfltcc_deflate_get_dictionary)((strm), (dict), (dict_len)); \
    } while (0)

#define DEFLATE_RESET_KEEP_HOOK PREFIX(dfltcc_reset_deflate_state)

#define DEFLATE_PARAMS_HOOK(strm, level, strategy, hook_flush) \
    do { \
        int err; \
\
        err = PREFIX(dfltcc_deflate_params)((strm), (level), (strategy), (hook_flush)); \
        if (err == Z_STREAM_ERROR) \
            return err; \
    } while (0)

#define DEFLATE_DONE PREFIX(dfltcc_deflate_done)

#define DEFLATE_BOUND_ADJUST_COMPLEN(strm, complen, source_len) \
    do { \
        if (deflateStateCheck((strm)) || PREFIX(dfltcc_can_deflate)((strm))) \
            (complen) = DEFLATE_BOUND_COMPLEN(source_len); \
    } while (0)

#define DEFLATE_NEED_CONSERVATIVE_BOUND(strm) (PREFIX(dfltcc_can_deflate)((strm)))

#define DEFLATE_HOOK PREFIX(dfltcc_deflate)

#define DEFLATE_NEED_CHECKSUM(strm) (!PREFIX(dfltcc_can_deflate)((strm)))

#define DEFLATE_CAN_SET_REPRODUCIBLE PREFIX(dfltcc_can_set_reproducible)

#define DEFLATE_ADJUST_WINDOW_SIZE(n) MAX(n, HB_SIZE)

#endif
