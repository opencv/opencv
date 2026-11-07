/* dfltcc_deflate.c - IBM Z DEFLATE CONVERSION CALL compression support. */

/*
   Use the following commands to build zlib-ng with DFLTCC compression support:

        $ ./configure --with-dfltcc-deflate
   or

        $ cmake -DWITH_DFLTCC_DEFLATE=1 .

   and then

        $ make
*/

#include "zbuild.h"
#include "deflate.h"
#include "trees_emit.h"
#include "dfltcc_deflate.h"
#include "dfltcc_detail.h"

void Z_INTERNAL PREFIX(dfltcc_reset_deflate_state)(PREFIX3(streamp) strm) {
    deflate_state *state = (deflate_state *)strm->state;
    arch_deflate_state *dfltcc_state = &state->arch;

    dfltcc_reset_state(&dfltcc_state->common);

    /* Initialize tuning parameters */
    dfltcc_state->level_mask = DFLTCC_LEVEL_MASK;
    dfltcc_state->block_size = DFLTCC_BLOCK_SIZE;
    dfltcc_state->block_threshold = DFLTCC_FIRST_FHT_BLOCK_SIZE;
    dfltcc_state->dht_threshold = DFLTCC_DHT_MIN_SAMPLE_SIZE;
}

static inline int dfltcc_can_deflate_with_params(PREFIX3(streamp) strm, int level, uInt window_bits, int strategy,
                                       int reproducible) {
    deflate_state *state = (deflate_state *)strm->state;
    arch_deflate_state *dfltcc_state = &state->arch;

    /* Unsupported compression settings */
    if ((dfltcc_state->level_mask & (1 << level)) == 0)
        return 0;
    if (window_bits != HB_BITS)
        return 0;
    if (strategy != Z_FIXED && strategy != Z_DEFAULT_STRATEGY)
        return 0;
    if (reproducible)
        return 0;

    /* Unsupported hardware */
    if (!is_bit_set(dfltcc_state->common.af.fns, DFLTCC_GDHT) ||
            !is_bit_set(dfltcc_state->common.af.fns, DFLTCC_CMPR) ||
            !is_bit_set(dfltcc_state->common.af.fmts, DFLTCC_FMT0))
        return 0;

    return 1;
}

int Z_INTERNAL PREFIX(dfltcc_can_deflate)(PREFIX3(streamp) strm) {
    deflate_state *state = (deflate_state *)strm->state;

    return dfltcc_can_deflate_with_params(strm, state->level, state->w_bits, state->strategy, state->reproducible);
}

static inline void dfltcc_gdht(PREFIX3(streamp) strm) {
    deflate_state *state = (deflate_state *)strm->state;
    struct dfltcc_param_v0 *param = &state->arch.common.param;
    size_t avail_in = strm->avail_in;

    dfltcc(DFLTCC_GDHT, param, NULL, NULL, &strm->next_in, &avail_in, NULL);
}

static inline dfltcc_cc dfltcc_cmpr(PREFIX3(streamp) strm) {
    deflate_state *state = (deflate_state *)strm->state;
    struct dfltcc_param_v0 *param = &state->arch.common.param;
    size_t avail_in = strm->avail_in;
    size_t avail_out = strm->avail_out;
    dfltcc_cc cc;

    cc = dfltcc(DFLTCC_CMPR | HBT_CIRCULAR,
                param, &strm->next_out, &avail_out,
                &strm->next_in, &avail_in, state->window);
    strm->total_in += (strm->avail_in - avail_in);
    strm->total_out += (strm->avail_out - avail_out);
    strm->avail_in = avail_in;
    strm->avail_out = avail_out;
    return cc;
}

static inline void send_eobs(PREFIX3(streamp) strm, const struct dfltcc_param_v0 *param) {
    deflate_state *state = (deflate_state *)strm->state;

    send_bits(state, PREFIX(bi_reverse)(param->eobs >> (15 - param->eobl), param->eobl), param->eobl, state->bi_buf, state->bi_valid);
    PREFIX(flush_pending)(strm);
    if (state->pending != 0) {
        /* The remaining data is located in pending_out[0:pending]. If someone
         * calls put_byte() - this might happen in deflate() - the byte will be
         * placed into pending_buf[pending], which is incorrect. Move the
         * remaining data to the beginning of pending_buf so that put_byte() is
         * usable again.
         */
        memmove(state->pending_buf, state->pending_out, state->pending);
        state->pending_out = state->pending_buf;
    }
#ifdef ZLIB_DEBUG
    state->compressed_len += param->eobl;
#endif
}

int Z_INTERNAL PREFIX(dfltcc_deflate)(PREFIX3(streamp) strm, int flush, block_state *result) {
    deflate_state *state = (deflate_state *)strm->state;
    arch_deflate_state *dfltcc_state = &state->arch;
    struct dfltcc_param_v0 *param = &dfltcc_state->common.param;
    uInt masked_avail_in;
    dfltcc_cc cc;
    int need_empty_block;
    int soft_bcc;
    int no_flush;

    if (!PREFIX(dfltcc_can_deflate)(strm)) {
        /* Clear history. */
        if (flush == Z_FULL_FLUSH)
            param->hl = 0;
        return 0;
    }

again:
    masked_avail_in = 0;
    soft_bcc = 0;
    no_flush = flush == Z_NO_FLUSH;

    /* No input data. Return, except when Continuation Flag is set, which means
     * that DFLTCC has buffered some output in the parameter block and needs to
     * be called again in order to flush it.
     */
    if (strm->avail_in == 0 && !param->cf) {
        /* A block is still open, and the hardware does not support closing
         * blocks without adding data. Thus, close it manually.
         */
        if (!no_flush && param->bcf) {
            send_eobs(strm, param);
            param->bcf = 0;
        }
        /* Let one of deflate_* functions write a trailing empty block. */
        if (flush == Z_FINISH)
            return 0;
        /* Clear history. */
        if (flush == Z_FULL_FLUSH)
            param->hl = 0;
        /* Trigger block post-processing if necessary. */
        *result = no_flush ? need_more : block_done;
        return 1;
    }

    /* There is an open non-BFINAL block, we are not going to close it just
     * yet, we have compressed more than DFLTCC_BLOCK_SIZE bytes and we see
     * more than DFLTCC_DHT_MIN_SAMPLE_SIZE bytes. Open a new block with a new
     * DHT in order to adapt to a possibly changed input data distribution.
     */
    if (param->bcf && no_flush &&
            strm->total_in > dfltcc_state->block_threshold &&
            strm->avail_in >= dfltcc_state->dht_threshold) {
        if (param->cf) {
            /* We need to flush the DFLTCC buffer before writing the
             * End-of-block Symbol. Mask the input data and proceed as usual.
             */
            masked_avail_in += strm->avail_in;
            strm->avail_in = 0;
            no_flush = 0;
        } else {
            /* DFLTCC buffer is empty, so we can manually write the
             * End-of-block Symbol right away.
             */
            send_eobs(strm, param);
            param->bcf = 0;
            dfltcc_state->block_threshold = strm->total_in + dfltcc_state->block_size;
        }
    }

    /* No space for compressed data. If we proceed, dfltcc_cmpr() will return
     * DFLTCC_CC_OP1_TOO_SHORT without buffering header bits, but we will still
     * set BCF=1, which is wrong. Avoid complications and return early.
     */
    if (strm->avail_out == 0) {
        *result = need_more;
        return 1;
    }

    /* The caller gave us too much data. Pass only one block worth of
     * uncompressed data to DFLTCC and mask the rest, so that on the next
     * iteration we start a new block.
     */
    if (no_flush && strm->avail_in > dfltcc_state->block_size) {
        masked_avail_in += (strm->avail_in - dfltcc_state->block_size);
        strm->avail_in = dfltcc_state->block_size;
    }

    /* When we have an open non-BFINAL deflate block and caller indicates that
     * the stream is ending, we need to close an open deflate block and open a
     * BFINAL one.
     */
    need_empty_block = flush == Z_FINISH && param->bcf && !param->bhf;

    /* Translate stream to parameter block */
    param->cvt = state->wrap == 2 ? CVT_CRC32 : CVT_ADLER32;
    if (!no_flush)
        /* We need to close a block. Always do this in software - when there is
         * no input data, the hardware will not honor BCC. */
        soft_bcc = 1;
    if (flush == Z_FINISH && !param->bcf)
        /* We are about to open a BFINAL block, set Block Header Final bit
         * until the stream ends.
         */
        param->bhf = 1;
    /* DFLTCC-CMPR will write to next_out, so make sure that buffers with
     * higher precedence are empty.
     */
    Assert(state->pending == 0, "There must be no pending bytes");
    Assert(state->bi_valid < 8, "There must be less than 8 pending bits");
    param->sbb = (unsigned int)state->bi_valid;
    if (param->sbb > 0)
        *strm->next_out = (unsigned char)state->bi_buf;
    /* Honor history and check value */
    param->nt = 0;
    if (state->wrap == 1)
        param->cv = strm->adler;
    else if (state->wrap == 2)
        param->cv = ZSWAP32(state->crc_fold.value);

    /* When opening a block, choose a Huffman-Table Type */
    if (!param->bcf) {
        if (state->strategy == Z_FIXED || (strm->total_in == 0 && dfltcc_state->block_threshold > 0))
            param->htt = HTT_FIXED;
        else {
            param->htt = HTT_DYNAMIC;
            dfltcc_gdht(strm);
        }
    }

    /* Deflate */
    do {
        cc = dfltcc_cmpr(strm);
        if (strm->avail_in < 4096 && masked_avail_in > 0)
            /* We are about to call DFLTCC with a small input buffer, which is
             * inefficient. Since there is masked data, there will be at least
             * one more DFLTCC call, so skip the current one and make the next
             * one handle more data.
             */
            break;
    } while (cc == DFLTCC_CC_AGAIN);

    /* Translate parameter block to stream */
    strm->msg = oesc_msg(dfltcc_state->common.msg, param->oesc);
    state->bi_valid = param->sbb;
    if (state->bi_valid == 0)
        state->bi_buf = 0; /* Avoid accessing next_out */
    else
        state->bi_buf = *strm->next_out & ((1 << state->bi_valid) - 1);
    if (state->wrap == 1)
        strm->adler = param->cv;
    else if (state->wrap == 2)
        state->crc_fold.value = ZSWAP32(param->cv);

    /* Unmask the input data */
    strm->avail_in += masked_avail_in;
    masked_avail_in = 0;

    /* If we encounter an error, it means there is a bug in DFLTCC call */
    Assert(cc != DFLTCC_CC_OP2_CORRUPT || param->oesc == 0, "BUG");

    /* Update Block-Continuation Flag. It will be used to check whether to call
     * GDHT the next time.
     */
    if (cc == DFLTCC_CC_OK) {
        if (soft_bcc) {
            send_eobs(strm, param);
            param->bcf = 0;
            dfltcc_state->block_threshold = strm->total_in + dfltcc_state->block_size;
        } else
            param->bcf = 1;
        if (flush == Z_FINISH) {
            if (need_empty_block)
                /* Make the current deflate() call also close the stream */
                return 0;
            else {
                bi_windup(state);
                *result = finish_done;
            }
        } else {
            if (flush == Z_FULL_FLUSH)
                param->hl = 0; /* Clear history */
            *result = flush == Z_NO_FLUSH ? need_more : block_done;
        }
    } else {
        param->bcf = 1;
        *result = need_more;
    }
    if (strm->avail_in != 0 && strm->avail_out != 0)
        goto again; /* deflate() must use all input or all output */
    return 1;
}

/*
   Switching between hardware and software compression.

   DFLTCC does not support all zlib settings, e.g. generation of non-compressed
   blocks or alternative window sizes. When such settings are applied on the
   fly with deflateParams, we need to convert between hardware and software
   window formats.
*/
static int dfltcc_was_deflate_used(PREFIX3(streamp) strm) {
    deflate_state *state = (deflate_state *)strm->state;
    struct dfltcc_param_v0 *param = &state->arch.common.param;

    return strm->total_in > 0 || param->nt == 0 || param->hl > 0;
}

int Z_INTERNAL PREFIX(dfltcc_deflate_params)(PREFIX3(streamp) strm, int level, int strategy, int *flush) {
    deflate_state *state = (deflate_state *)strm->state;
    int could_deflate = PREFIX(dfltcc_can_deflate)(strm);
    int can_deflate = dfltcc_can_deflate_with_params(strm, level, state->w_bits, strategy, state->reproducible);

    if (can_deflate == could_deflate)
        /* We continue to work in the same mode - no changes needed */
        return Z_OK;

    if (!dfltcc_was_deflate_used(strm))
        /* DFLTCC was not used yet - no changes needed */
        return Z_OK;

    /* For now, do not convert between window formats - simply get rid of the old data instead */
    *flush = Z_FULL_FLUSH;
    return Z_OK;
}

int Z_INTERNAL PREFIX(dfltcc_deflate_done)(PREFIX3(streamp) strm, int flush) {
    deflate_state *state = (deflate_state *)strm->state;
    struct dfltcc_param_v0 *param = &state->arch.common.param;

    /* When deflate(Z_FULL_FLUSH) is called with small avail_out, it might
     * close the block without resetting the compression state. Detect this
     * situation and return that deflation is not done.
     */
    if (flush == Z_FULL_FLUSH && strm->avail_out == 0)
        return 0;

    /* Return that deflation is not done if DFLTCC is used and either it
     * buffered some data (Continuation Flag is set), or has not written EOBS
     * yet (Block-Continuation Flag is set).
     */
    return !PREFIX(dfltcc_can_deflate)(strm) || (!param->cf && !param->bcf);
}

int Z_INTERNAL PREFIX(dfltcc_can_set_reproducible)(PREFIX3(streamp) strm, int reproducible) {
    deflate_state *state = (deflate_state *)strm->state;

    return reproducible != state->reproducible && !dfltcc_was_deflate_used(strm);
}

/*
   Preloading history.
*/
int Z_INTERNAL PREFIX(dfltcc_deflate_set_dictionary)(PREFIX3(streamp) strm,
                                                const unsigned char *dictionary, uInt dict_length) {
    deflate_state *state = (deflate_state *)strm->state;
    struct dfltcc_param_v0 *param = &state->arch.common.param;

    append_history(param, state->window, dictionary, dict_length);
    state->strstart = 1; /* Add FDICT to zlib header */
    state->block_start = state->strstart; /* Make deflate_stored happy */
    return Z_OK;
}

int Z_INTERNAL PREFIX(dfltcc_deflate_get_dictionary)(PREFIX3(streamp) strm, unsigned char *dictionary, uInt *dict_length) {
    deflate_state *state = (deflate_state *)strm->state;
    struct dfltcc_param_v0 *param = &state->arch.common.param;

    if (dictionary)
        get_history(param, state->window, dictionary);
    if (dict_length)
        *dict_length = param->hl;
    return Z_OK;
}
