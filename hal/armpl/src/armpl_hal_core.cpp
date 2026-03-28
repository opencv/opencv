#ifdef HAVE_ARMPL

#include "armpl_hal_core.hpp"

#include <fftw3.h>
#include <cstring>
#include <cstdio>
#include <cmath>
enum ArmPLDFTMode
{
    ARMPL_DFT_C2C,
    ARMPL_DFT_R2C,
    ARMPL_DFT_C2C_ROW,
    ARMPL_DFT_R_ROW,
    ARMPL_DFT_1D_C2C_FWD,
    ARMPL_DFT_1D_C2C_FWD_64,
    ARMPL_DFT_1D_C2C_INV,
    ARMPL_DFT_1D_C2C_INV_64,
    ARMPL_DFT_1D_R2C_32,
    ARMPL_DFT_1D_R2C_64,
    ARMPL_DFT_1D_R2C_ROWS_32,
    ARMPL_DFT_1D_R2C_ROWS_64,
    ARMPL_DCT_2D,
    ARMPL_DCT_2D_64,
    ARMPL_DCT_ROW,
    ARMPL_DCT_ROW_64,
};

struct ArmPLDCT2DContext
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         inv, no_scale;
    fftwf_plan   plan_fwd, plan_inv;
    float        scale_dc, scale_axis, scale_rest;
    float       *buf;

    ArmPLDCT2DContext()
        : mode(ARMPL_DCT_2D), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0),
          scale_dc(1.f), scale_axis(1.f), scale_rest(1.f), buf(0) {}
};

struct ArmPLDCT2DContext64
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         inv, no_scale;
    fftw_plan    plan_fwd, plan_inv;
    double       scale_dc, scale_axis, scale_rest;

    ArmPLDCT2DContext64()
        : mode(ARMPL_DCT_2D_64), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0),
          scale_dc(1.0), scale_axis(1.0), scale_rest(1.0) {}
};

struct ArmPLDCTRowContext
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         inv, no_scale;
    fftwf_plan   plan_fwd, plan_inv;
    float        scale_dc, scale_rest;
    float       *fftw_buf;

    ArmPLDCTRowContext()
        : mode(ARMPL_DCT_ROW), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0),
          scale_dc(1.f), scale_rest(1.f), fftw_buf(0) {}
};

struct ArmPLDCTRowContext64
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         inv, no_scale;
    fftw_plan    plan_fwd, plan_inv;
    double       scale_dc, scale_rest;
    double      *fftw_buf;

    ArmPLDCTRowContext64()
        : mode(ARMPL_DCT_ROW_64), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0),
          scale_dc(1.0), scale_rest(1.0), fftw_buf(0) {}
};

struct ArmPLC2CDFTContext
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         inv, no_scale;
    fftwf_plan   plan_fwd, plan_inv;
    float        scale;

    ArmPLC2CDFTContext()
        : mode(ARMPL_DFT_C2C), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0), scale(1.f) {}
};

struct ArmPLR2CDFTContext
{
    ArmPLDFTMode mode;
    int          width, height;
    bool         col_wise, no_scale;
    fftwf_plan   plan;
    float        scale;

    ArmPLR2CDFTContext()
        : mode(ARMPL_DFT_R2C), width(0), height(0), col_wise(false), no_scale(true),
          plan(0), scale(1.f) {}
};

struct ArmPLC2CRowDFTContext
{
    ArmPLDFTMode   mode;
    int            width, height;
    bool           inv, no_scale;
    fftwf_plan     plan_fwd, plan_inv;
    float          scale;
    fftwf_complex *fftw_buf;

    ArmPLC2CRowDFTContext()
        : mode(ARMPL_DFT_C2C_ROW), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0), scale(1.f), fftw_buf(0) {}
};

struct ArmPLRRowDFTContext
{
    ArmPLDFTMode   mode;
    int            width, height;
    bool           inv, no_scale;
    fftwf_plan     plan_fwd, plan_inv;
    float          scale;
    float         *fftw_in_r;
    fftwf_complex *fftw_out_c;
    fftwf_complex *fftw_in_c;
    float         *fftw_out_r;

    ArmPLRRowDFTContext()
        : mode(ARMPL_DFT_R_ROW), width(0), height(0), inv(false), no_scale(true),
          plan_fwd(0), plan_inv(0), scale(1.f),
          fftw_in_r(0), fftw_out_c(0), fftw_in_c(0), fftw_out_r(0) {}
};

struct ArmPL1DC2CFwdContext
{
    ArmPLDFTMode mode;
    int          len;
    bool         no_scale;
    fftwf_plan   plan;
    float        scale;

    ArmPL1DC2CFwdContext()
        : mode(ARMPL_DFT_1D_C2C_FWD), len(0), no_scale(true),
          plan(0), scale(1.f) {}
};

struct ArmPL1DC2CFwdContext64
{
    ArmPLDFTMode mode;
    int          len;
    bool         no_scale;
    fftw_plan    plan;
    double       scale;

    ArmPL1DC2CFwdContext64()
        : mode(ARMPL_DFT_1D_C2C_FWD_64), len(0), no_scale(true),
          plan(0), scale(1.0) {}
};

struct ArmPL1DR2CContext32
{
    ArmPLDFTMode mode;
    int          len;
    bool         no_scale;
    float        scale;
    fftwf_plan   plan;

    ArmPL1DR2CContext32()
        : mode(ARMPL_DFT_1D_R2C_32), len(0), no_scale(true),
          scale(1.f), plan(0) {}
};

struct ArmPL1DR2CContext64
{
    ArmPLDFTMode mode;
    int          len;
    bool         no_scale;
    double       scale;
    fftw_plan    plan;

    ArmPL1DR2CContext64()
        : mode(ARMPL_DFT_1D_R2C_64), len(0), no_scale(true),
          scale(1.0), plan(0) {}
};

struct ArmPL1DR2CRowsContext32
{
    ArmPLDFTMode mode;
    int          len;
    int          count;
    bool         no_scale;
    float        scale;
    fftwf_plan   plan;

    ArmPL1DR2CRowsContext32()
        : mode(ARMPL_DFT_1D_R2C_ROWS_32), len(0), count(0), no_scale(true),
          scale(1.f), plan(0) {}
};

struct ArmPL1DR2CRowsContext64
{
    ArmPLDFTMode mode;
    int          len;
    int          count;
    bool         no_scale;
    double       scale;
    fftw_plan    plan;

    ArmPL1DR2CRowsContext64()
        : mode(ARMPL_DFT_1D_R2C_ROWS_64), len(0), count(0), no_scale(true),
          scale(1.0), plan(0) {}
};

int armpl_hal_dftInit2D(cvhalDFT **context,
                        int width, int height,
                        int depth,
                        int src_channels, int dst_channels,
                        int flags, int nonzero_rows)
{
    if (depth        != CV_32F)       return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (nonzero_rows != 0)            return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if ((size_t)width * height <= 64) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const bool isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
    const bool isScaled  = (flags & CV_HAL_DFT_SCALE)   != 0;
    const bool isRowWise = (flags & CV_HAL_DFT_ROWS)    != 0;

    if (!isRowWise && src_channels == 2 && dst_channels == 2)
    {
        const int   norm_flag = !isScaled ? 8 : (isInverse ? 2 : 1);
        float       scale     = 1.0f;
        const float inv_total = 1.0f / (float)(width * height);
        if (isInverse) { if (norm_flag == 1 || norm_flag == 2) scale = inv_total; }
        else           { if (norm_flag == 1)                   scale = inv_total; }

        const size_t total = (size_t)width * height;
        fftwf_complex *tmp_in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * total);
        fftwf_complex *tmp_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * total);
        if (!tmp_in || !tmp_out)
        {
            if (tmp_in)  fftwf_free(tmp_in);
            if (tmp_out) fftwf_free(tmp_out);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        fftwf_plan pf = fftwf_plan_dft_2d(height, width, tmp_in, tmp_out, FFTW_FORWARD,  FFTW_ESTIMATE);
        fftwf_plan pi = fftwf_plan_dft_2d(height, width, tmp_in, tmp_out, FFTW_BACKWARD, FFTW_ESTIMATE);
        fftwf_free(tmp_in);
        fftwf_free(tmp_out);
        if (!pf || !pi)
        {
            if (pf) fftwf_destroy_plan(pf);
            if (pi) fftwf_destroy_plan(pi);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        ArmPLC2CDFTContext *ctx = new ArmPLC2CDFTContext();
        ctx->width = width; ctx->height = height;
        ctx->inv = isInverse; ctx->no_scale = (scale == 1.0f);
        ctx->plan_fwd = pf; ctx->plan_inv = pi; ctx->scale = scale;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }
    if (!isRowWise && src_channels == 1 && dst_channels == 1)
    {
        if (isInverse) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const bool col_wise = (width == 1);
        float      scale    = 1.0f;
        if (isScaled)
            scale = col_wise ? (1.0f / height) : (1.0f / (float)(width * height));

        fftwf_plan plan = 0;
        if (col_wise)
        {
            float         *dr = (float*)        fftwf_malloc(sizeof(float)         * height);
            fftwf_complex *dc = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (height/2 + 1));
            if (!dr || !dc) { if (dr) fftwf_free(dr); if (dc) fftwf_free(dc); return CV_HAL_ERROR_NOT_IMPLEMENTED; }
            plan = fftwf_plan_dft_r2c_1d(height, dr, dc, FFTW_ESTIMATE);
            fftwf_free(dr); fftwf_free(dc);
        }
        else
        {
            float         *dr = (float*)        fftwf_malloc(sizeof(float)         * (size_t)width * height);
            fftwf_complex *dc = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)height * (width/2 + 1));
            if (!dr || !dc) { if (dr) fftwf_free(dr); if (dc) fftwf_free(dc); return CV_HAL_ERROR_NOT_IMPLEMENTED; }
            plan = fftwf_plan_dft_r2c_2d(height, width, dr, dc, FFTW_ESTIMATE);
            fftwf_free(dr); fftwf_free(dc);
        }
        if (!plan) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        ArmPLR2CDFTContext *ctx = new ArmPLR2CDFTContext();
        ctx->width = width; ctx->height = height;
        ctx->col_wise = col_wise; ctx->no_scale = (scale == 1.0f);
        ctx->plan = plan; ctx->scale = scale;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }
    if (isRowWise && src_channels == 2 && dst_channels == 2)
    {
        const int   norm_flag = !isScaled ? 8 : (isInverse ? 2 : 1);
        float       scale     = 1.0f;
        const float inv_w     = 1.0f / (float)width;
        if (isInverse) { if (norm_flag == 1 || norm_flag == 2) scale = inv_w; }
        else           { if (norm_flag == 1)                   scale = inv_w; }

        fftwf_complex *fftw_buf = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * width);
        if (!fftw_buf) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        fftwf_plan pf = fftwf_plan_dft_1d(width, fftw_buf, fftw_buf, FFTW_FORWARD,  FFTW_ESTIMATE);
        fftwf_plan pi = fftwf_plan_dft_1d(width, fftw_buf, fftw_buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!pf || !pi)
        {
            if (pf) fftwf_destroy_plan(pf);
            if (pi) fftwf_destroy_plan(pi);
            fftwf_free(fftw_buf);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        ArmPLC2CRowDFTContext *ctx = new ArmPLC2CRowDFTContext();
        ctx->width = width; ctx->height = height; ctx->inv = isInverse;
        ctx->plan_fwd = pf; ctx->plan_inv = pi;
        ctx->scale = scale; ctx->no_scale = (scale == 1.0f);
        ctx->fftw_buf = fftw_buf;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }

    if (isRowWise && src_channels == 1 && dst_channels == 1)
    {
        float scale = 1.0f;
        if (isScaled) scale = 1.0f / (float)width;

        float         *fftw_in_r  = (float*)        fftwf_malloc(sizeof(float)         * width);
        fftwf_complex *fftw_out_c = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (width/2 + 1));
        fftwf_complex *fftw_in_c  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (width/2 + 1));
        float         *fftw_out_r = (float*)        fftwf_malloc(sizeof(float)         * width);
        if (!fftw_in_r || !fftw_out_c || !fftw_in_c || !fftw_out_r)
        {
            if (fftw_in_r)  fftwf_free(fftw_in_r);
            if (fftw_out_c) fftwf_free(fftw_out_c);
            if (fftw_in_c)  fftwf_free(fftw_in_c);
            if (fftw_out_r) fftwf_free(fftw_out_r);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        fftwf_plan pf = fftwf_plan_dft_r2c_1d(width, fftw_in_r, fftw_out_c, FFTW_ESTIMATE);
        fftwf_plan pi = fftwf_plan_dft_c2r_1d(width, fftw_in_c, fftw_out_r, FFTW_ESTIMATE);
        if (!pf || !pi)
        {
            if (pf) fftwf_destroy_plan(pf);
            if (pi) fftwf_destroy_plan(pi);
            fftwf_free(fftw_in_r); fftwf_free(fftw_out_c);
            fftwf_free(fftw_in_c); fftwf_free(fftw_out_r);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        ArmPLRRowDFTContext *ctx = new ArmPLRRowDFTContext();
        ctx->width = width; ctx->height = height; ctx->inv = isInverse;
        ctx->plan_fwd = pf; ctx->plan_inv = pi;
        ctx->scale = scale; ctx->no_scale = (scale == 1.0f);
        ctx->fftw_in_r = fftw_in_r; ctx->fftw_out_c = fftw_out_c;
        ctx->fftw_in_c = fftw_in_c; ctx->fftw_out_r = fftw_out_r;
        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int armpl_hal_dft2D(cvhalDFT *context,
                    const unsigned char *src_data, size_t src_step,
                    unsigned char       *dst_data, size_t dst_step)
{
    if (!context || !src_data || !dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);

    if (mode == ARMPL_DFT_C2C)
    {
        ArmPLC2CDFTContext *ctx = reinterpret_cast<ArmPLC2CDFTContext*>(context);
        if (!ctx->plan_fwd || !ctx->plan_inv) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int    W        = ctx->width;
        const int    H        = ctx->height;
        const float  sc       = ctx->scale;
        const bool   no_scale = ctx->no_scale;
        const size_t row_cb   = (size_t)W * sizeof(fftwf_complex);

        fftwf_complex *in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)W * H);
        fftwf_complex *out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)W * H);
        if (!in || !out) { if (in) fftwf_free(in); if (out) fftwf_free(out); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        if (src_step == row_cb)
            memcpy(in, src_data, (size_t)W * H * sizeof(fftwf_complex));
        else
        {
            for (int y = 0; y < H; y++)
            {
                const float   *sr = reinterpret_cast<const float*>(src_data + (size_t)y * src_step);
                fftwf_complex *ir = in + (size_t)y * W;
                int x = 0;
#ifdef CV_NEON
                for (; x + 3 < W; x += 4)
                {
                    vst1q_f32((float*)(ir+x),   vld1q_f32(sr + x*2));
                    vst1q_f32((float*)(ir+x+2), vld1q_f32(sr + (x+2)*2));
                }
#endif
                for (; x < W; x++) { ir[x][0] = sr[x*2]; ir[x][1] = sr[x*2+1]; }
            }
        }

        fftwf_execute_dft(ctx->inv ? ctx->plan_inv : ctx->plan_fwd, in, out);

        if (no_scale)
        {
            if (dst_step == row_cb)
                memcpy(dst_data, out, (size_t)W * H * sizeof(fftwf_complex));
            else
                for (int y = 0; y < H; y++)
                    memcpy(dst_data + (size_t)y * dst_step, out + (size_t)y * W, row_cb);
        }
        else
        {
#ifdef CV_NEON
            const float32x4_t sv = vdupq_n_f32(sc);
#endif
            for (int y = 0; y < H; y++)
            {
                float               *dr  = reinterpret_cast<float*>(dst_data + (size_t)y * dst_step);
                const fftwf_complex *or_ = out + (size_t)y * W;
                int x = 0;
#ifdef CV_NEON
                for (; x + 3 < W; x += 4)
                {
                    vst1q_f32(dr + x*2,     vmulq_f32(vld1q_f32((const float*)(or_+x)),   sv));
                    vst1q_f32(dr + (x+2)*2, vmulq_f32(vld1q_f32((const float*)(or_+x+2)), sv));
                }
#endif
                for (; x < W; x++) { dr[x*2] = or_[x][0]*sc; dr[x*2+1] = or_[x][1]*sc; }
            }
        }

        fftwf_free(in);
        fftwf_free(out);
        return CV_HAL_ERROR_OK;
    }

    if (mode == ARMPL_DFT_R2C)
    {
        ArmPLR2CDFTContext *ctx = reinterpret_cast<ArmPLR2CDFTContext*>(context);
        if (!ctx->plan) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int   W        = ctx->width;
        const int   H        = ctx->height;
        const float sc       = ctx->scale;
        const bool  no_scale = ctx->no_scale;

        if (ctx->col_wise)
        {
            float         *in  = (float*)        fftwf_malloc(sizeof(float)         * H);
            fftwf_complex *out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (H/2 + 1));
            if (!in || !out) { if (in) fftwf_free(in); if (out) fftwf_free(out); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

            for (int y = 0; y < H; y++)
                in[y] = reinterpret_cast<const float*>(src_data + (size_t)y * src_step)[0];
            fftwf_execute_dft_r2c(ctx->plan, in, out);

            const int pairs = (H - 1) / 2;
            if (no_scale)
            {
                reinterpret_cast<float*>(dst_data)[0] = out[0][0];
                for (int k = 1; k <= pairs; k++)
                {
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k-1)*dst_step)[0] = out[k][0];
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k)  *dst_step)[0] = out[k][1];
                }
                if ((H & 1) == 0)
                    reinterpret_cast<float*>(dst_data + (size_t)(H-1)*dst_step)[0] = out[H/2][0];
            }
            else
            {
                reinterpret_cast<float*>(dst_data)[0] = out[0][0] * sc;
                for (int k = 1; k <= pairs; k++)
                {
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k-1)*dst_step)[0] = out[k][0] * sc;
                    reinterpret_cast<float*>(dst_data + (size_t)(2*k)  *dst_step)[0] = out[k][1] * sc;
                }
                if ((H & 1) == 0)
                    reinterpret_cast<float*>(dst_data + (size_t)(H-1)*dst_step)[0] = out[H/2][0] * sc;
            }
            fftwf_free(in);
            fftwf_free(out);
            return CV_HAL_ERROR_OK;
        }

        float         *in  = (float*)        fftwf_malloc(sizeof(float)         * (size_t)W * H);
        fftwf_complex *out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * (size_t)H * (W/2 + 1));
        if (!in || !out) { if (in) fftwf_free(in); if (out) fftwf_free(out); return CV_HAL_ERROR_NOT_IMPLEMENTED; }

        if (src_step == (size_t)W * sizeof(float))
            memcpy(in, src_data, (size_t)W * H * sizeof(float));
        else
            for (int y = 0; y < H; y++)
                memcpy(in + (size_t)y * W,
                       reinterpret_cast<const float*>(src_data + (size_t)y * src_step),
                       (size_t)W * sizeof(float));

        fftwf_execute_dft_r2c(ctx->plan, in, out);

        const int half1  = W/2 + 1;
        const int pairs  = (W - 1) / 2;
        const int even_W = (W & 1) == 0;

#define PACK_BINS(dr_, fi_)                                                      \
        do {                                                                     \
            float *_d = (dr_); int _fi = (fi_);                                  \
            if (no_scale) {                                                       \
                for (int k = 1; k <= pairs; k++) {                               \
                    _d[2*k-1] = out[_fi+k][0]; _d[2*k] = out[_fi+k][1];        \
                }                                                                 \
            } else {                                                              \
                for (int k = 1; k <= pairs; k++) {                               \
                    _d[2*k-1] = out[_fi+k][0]*sc; _d[2*k] = out[_fi+k][1]*sc;  \
                }                                                                 \
            }                                                                     \
        } while(0)

        for (int y = 0; y < H; y++)
        {
            float     *dr = reinterpret_cast<float*>(dst_data + (size_t)y * dst_step);
            const int  fi = y * half1;

            if (y == 0 || y == 1)
            {
                dr[0] = no_scale ? out[fi][0] : out[fi][0] * sc;
            }
            else if ((y & 1) == 0)
            {
                const int fi_c0 = (y / 2) * half1;
                dr[0] = no_scale ? out[fi_c0][1] : out[fi_c0][1] * sc;
            }
            else
            {
                const int fi_c0 = ((y + 1) / 2) * half1;
                dr[0] = no_scale ? out[fi_c0][0] : out[fi_c0][0] * sc;
            }

            PACK_BINS(dr, fi);

            if (even_W)
            {
                if (y == 0 || y == 1)
                {
                    dr[W - 1] = no_scale ? out[fi + W/2][0] : out[fi + W/2][0] * sc;
                }
                else if ((y & 1) == 0)
                {
                    const int fi_c0 = (y / 2) * half1;
                    dr[W - 1] = no_scale ? out[fi_c0 + W/2][1] : out[fi_c0 + W/2][1] * sc;
                }
                else
                {
                    const int fi_c0 = ((y + 1) / 2) * half1;
                    dr[W - 1] = no_scale ? out[fi_c0 + W/2][0] : out[fi_c0 + W/2][0] * sc;
                }
            }
        }

#undef PACK_BINS

        fftwf_free(in);
        fftwf_free(out);
        return CV_HAL_ERROR_OK;
    }

    if (mode == ARMPL_DFT_C2C_ROW)
    {
        ArmPLC2CRowDFTContext *ctx = reinterpret_cast<ArmPLC2CRowDFTContext*>(context);
        if (!ctx->plan_fwd || !ctx->plan_inv) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int      W        = ctx->width;
        const int      H        = ctx->height;
        const float    sc       = ctx->scale;
        const bool     no_scale = ctx->no_scale;
        const size_t   rb       = (size_t)W * sizeof(fftwf_complex);
        fftwf_complex *buf      = ctx->fftw_buf;

        if (!ctx->inv)
        {
            for (int i = 0; i < H; i++)
            {
                const unsigned char *sb = src_data + (size_t)i * src_step;
                unsigned char       *db = dst_data + (size_t)i * dst_step;
                if (sb != db) memcpy(db, sb, rb);
                fftwf_execute_dft(ctx->plan_fwd,
                                  reinterpret_cast<fftwf_complex*>(db),
                                  reinterpret_cast<fftwf_complex*>(db));
                if (!no_scale)
                {
                    float *f = reinterpret_cast<float*>(db);
                    int j = 0;
#ifdef CV_NEON
                    const float32x4_t sv = vdupq_n_f32(sc);
                    for (; j + 7 < W*2; j += 8)
                    {
                        vst1q_f32(f+j,   vmulq_f32(vld1q_f32(f+j),   sv));
                        vst1q_f32(f+j+4, vmulq_f32(vld1q_f32(f+j+4), sv));
                    }
#endif
                    for (; j < W*2; j++) f[j] *= sc;
                }
            }
        }
        else
        {
            for (int i = 0; i < H; i++)
            {
                const unsigned char *sb = src_data + (size_t)i * src_step;
                unsigned char       *db = dst_data + (size_t)i * dst_step;
                memcpy(buf, sb, rb);
                fftwf_execute(ctx->plan_inv);
                if (no_scale)
                    memcpy(db, buf, rb);
                else
                {
                    float       *df = reinterpret_cast<float*>(db);
                    const float *bf = reinterpret_cast<const float*>(buf);
                    int j = 0;
#ifdef CV_NEON
                    const float32x4_t sv = vdupq_n_f32(sc);
                    for (; j + 7 < W*2; j += 8)
                    {
                        vst1q_f32(df+j,   vmulq_f32(vld1q_f32(bf+j),   sv));
                        vst1q_f32(df+j+4, vmulq_f32(vld1q_f32(bf+j+4), sv));
                    }
#endif
                    for (; j < W*2; j++) df[j] = bf[j]*sc;
                }
            }
        }
        return CV_HAL_ERROR_OK;
    }

    if (mode == ARMPL_DFT_R_ROW)
    {
        ArmPLRRowDFTContext *ctx = reinterpret_cast<ArmPLRRowDFTContext*>(context);
        if (!ctx->plan_fwd || !ctx->plan_inv) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int   W        = ctx->width;
        const int   H        = ctx->height;
        const float sc       = ctx->scale;
        const bool  no_scale = ctx->no_scale;

        if (!ctx->inv)
        {
            float         *fin  = ctx->fftw_in_r;
            fftwf_complex *fout = ctx->fftw_out_c;
            const int      ncf  = (W - 1) / 2;
            const bool     hnyq = (W & 1) == 0;

            for (int i = 0; i < H; i++)
            {
                const float *sr = reinterpret_cast<const float*>(src_data + (size_t)i * src_step);
                float       *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);

                memcpy(fin, sr, (size_t)W * sizeof(float));
                fftwf_execute(ctx->plan_fwd);

                if (no_scale)
                {
                    dr[0] = fout[0][0];
                    int j = 1;
                    for (; j + 3 <= ncf; j += 4)
                    {
                        dr[j*2-1]     = fout[j][0];   dr[j*2]     = fout[j][1];
                        dr[(j+1)*2-1] = fout[j+1][0]; dr[(j+1)*2] = fout[j+1][1];
                        dr[(j+2)*2-1] = fout[j+2][0]; dr[(j+2)*2] = fout[j+2][1];
                        dr[(j+3)*2-1] = fout[j+3][0]; dr[(j+3)*2] = fout[j+3][1];
                    }
                    for (; j <= ncf; j++) { dr[j*2-1] = fout[j][0]; dr[j*2] = fout[j][1]; }
                    if (hnyq) dr[W-1] = fout[W/2][0];
                }
                else
                {
                    dr[0] = fout[0][0] * sc;
                    int j = 1;
                    for (; j + 3 <= ncf; j += 4)
                    {
                        dr[j*2-1]     = fout[j][0]*sc;   dr[j*2]     = fout[j][1]*sc;
                        dr[(j+1)*2-1] = fout[j+1][0]*sc; dr[(j+1)*2] = fout[j+1][1]*sc;
                        dr[(j+2)*2-1] = fout[j+2][0]*sc; dr[(j+2)*2] = fout[j+2][1]*sc;
                        dr[(j+3)*2-1] = fout[j+3][0]*sc; dr[(j+3)*2] = fout[j+3][1]*sc;
                    }
                    for (; j <= ncf; j++) { dr[j*2-1] = fout[j][0]*sc; dr[j*2] = fout[j][1]*sc; }
                    if (hnyq) dr[W-1] = fout[W/2][0] * sc;
                }
            }
        }
        else
        {
            fftwf_complex *fin  = ctx->fftw_in_c;
            float         *fout = ctx->fftw_out_r;
            const bool     hnyq = (W & 1) == 0;

            for (int i = 0; i < H; i++)
            {
                const float *sr = reinterpret_cast<const float*>(src_data + (size_t)i * src_step);
                float       *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);

                fin[0][0] = sr[0]; fin[0][1] = 0.f;

                if (hnyq)
                {
                    int j = 1, end = W/2;
                    for (; j + 3 < end; j += 4)
                    {
                        fin[j][0]   = sr[j*2-1];     fin[j][1]   = sr[j*2];
                        fin[j+1][0] = sr[(j+1)*2-1]; fin[j+1][1] = sr[(j+1)*2];
                        fin[j+2][0] = sr[(j+2)*2-1]; fin[j+2][1] = sr[(j+2)*2];
                        fin[j+3][0] = sr[(j+3)*2-1]; fin[j+3][1] = sr[(j+3)*2];
                    }
                    for (; j < end; j++) { fin[j][0] = sr[j*2-1]; fin[j][1] = sr[j*2]; }
                    fin[W/2][0] = sr[W-1]; fin[W/2][1] = 0.f;
                }
                else
                {
                    int j = 1, end = W/2 + 1;
                    for (; j + 3 < end; j += 4)
                    {
                        fin[j][0]   = sr[j*2-1];     fin[j][1]   = sr[j*2];
                        fin[j+1][0] = sr[(j+1)*2-1]; fin[j+1][1] = sr[(j+1)*2];
                        fin[j+2][0] = sr[(j+2)*2-1]; fin[j+2][1] = sr[(j+2)*2];
                        fin[j+3][0] = sr[(j+3)*2-1]; fin[j+3][1] = sr[(j+3)*2];
                    }
                    for (; j < end; j++) { fin[j][0] = sr[j*2-1]; fin[j][1] = sr[j*2]; }
                }

                fftwf_execute(ctx->plan_inv);

                if (no_scale)
                    memcpy(dr, fout, (size_t)W * sizeof(float));
                else
                {
                    int j = 0;
#ifdef CV_NEON
                    const float32x4_t sv = vdupq_n_f32(sc);
                    for (; j + 3 < W; j += 4)
                        vst1q_f32(dr+j, vmulq_f32(vld1q_f32(fout+j), sv));
#endif
                    for (; j < W; j++) dr[j] = fout[j] * sc;
                }
            }
        }
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int armpl_hal_dftFree2D(cvhalDFT *context)
{
    if (!context) return CV_HAL_ERROR_OK;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);

    if (mode == ARMPL_DFT_C2C)
    {
        ArmPLC2CDFTContext *ctx = reinterpret_cast<ArmPLC2CDFTContext*>(context);
        if (ctx->plan_fwd) fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->plan_inv) fftwf_destroy_plan(ctx->plan_inv);
        delete ctx;
    }
    else if (mode == ARMPL_DFT_R2C)
    {
        ArmPLR2CDFTContext *ctx = reinterpret_cast<ArmPLR2CDFTContext*>(context);
        if (ctx->plan) fftwf_destroy_plan(ctx->plan);
        delete ctx;
    }
    else if (mode == ARMPL_DFT_C2C_ROW)
    {
        ArmPLC2CRowDFTContext *ctx = reinterpret_cast<ArmPLC2CRowDFTContext*>(context);
        if (ctx->plan_fwd) fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->plan_inv) fftwf_destroy_plan(ctx->plan_inv);
        if (ctx->fftw_buf) fftwf_free(ctx->fftw_buf);
        delete ctx;
    }
    else if (mode == ARMPL_DFT_R_ROW)
    {
        ArmPLRRowDFTContext *ctx = reinterpret_cast<ArmPLRRowDFTContext*>(context);
        if (ctx->plan_fwd)   fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->plan_inv)   fftwf_destroy_plan(ctx->plan_inv);
        if (ctx->fftw_in_r)  fftwf_free(ctx->fftw_in_r);
        if (ctx->fftw_out_c) fftwf_free(ctx->fftw_out_c);
        if (ctx->fftw_in_c)  fftwf_free(ctx->fftw_in_c);
        if (ctx->fftw_out_r) fftwf_free(ctx->fftw_out_r);
        delete ctx;
    }

    return CV_HAL_ERROR_OK;
}

int armpl_hal_dftInit1D(cvhalDFT **context, int len, int count,
                        int depth, int flags, bool *needBuffer)
{
    const bool isInverse    = (flags & CV_HAL_DFT_INVERSE)        != 0;
    const bool isScaled     = (flags & CV_HAL_DFT_SCALE)          != 0;
    const bool isRows       = (flags & CV_HAL_DFT_ROWS)           != 0;
    const bool isRealOut    = (flags & CV_HAL_DFT_REAL_OUTPUT)    != 0;
    const bool isComplexOut = (flags & CV_HAL_DFT_COMPLEX_OUTPUT) != 0;
    const bool isTwoStage   = (flags & CV_HAL_DFT_TWO_STAGE)      != 0;
    const bool isStageCol   = (flags & CV_HAL_DFT_STAGE_COLS)     != 0;

    const bool isRealTransform = isRealOut;
    if (isRealTransform)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    if (isScaled && !isRows && isRealTransform)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const double scale_d = isScaled ? (1.0 / (double)len) : 1.0;

    const int fftw_dir = isInverse ? FFTW_BACKWARD : FFTW_FORWARD;

    if (depth == CV_32F)
    {
        fftwf_complex *tmp_in  = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * len);
        fftwf_complex *tmp_out = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * len);
        if (!tmp_in || !tmp_out)
        {
            if (tmp_in)  fftwf_free(tmp_in);
            if (tmp_out) fftwf_free(tmp_out);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        fftwf_plan plan = fftwf_plan_dft_1d(len, tmp_in, tmp_out, fftw_dir, FFTW_ESTIMATE);
        fftwf_free(tmp_in);
        fftwf_free(tmp_out);
        if (!plan) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        ArmPL1DC2CFwdContext *ctx = new ArmPL1DC2CFwdContext();
        ctx->mode     = isInverse ? ARMPL_DFT_1D_C2C_INV : ARMPL_DFT_1D_C2C_FWD;
        ctx->len      = len;
        ctx->plan     = plan;
        ctx->scale    = (float)scale_d;
        ctx->no_scale = (scale_d == 1.0);
        *context = reinterpret_cast<cvhalDFT*>(ctx);
    }
    else // CV_64F
    {
        fftw_complex *tmp_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * len);
        fftw_complex *tmp_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * len);
        if (!tmp_in || !tmp_out)
        {
            if (tmp_in)  fftw_free(tmp_in);
            if (tmp_out) fftw_free(tmp_out);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }
        fftw_plan plan = fftw_plan_dft_1d(len, tmp_in, tmp_out, fftw_dir, FFTW_ESTIMATE);
        fftw_free(tmp_in);
        fftw_free(tmp_out);
        if (!plan) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        ArmPL1DC2CFwdContext64 *ctx = new ArmPL1DC2CFwdContext64();
        ctx->mode     = isInverse ? ARMPL_DFT_1D_C2C_INV_64 : ARMPL_DFT_1D_C2C_FWD_64;
        ctx->len      = len;
        ctx->plan     = plan;
        ctx->scale    = scale_d;
        ctx->no_scale = (scale_d == 1.0);
        *context = reinterpret_cast<cvhalDFT*>(ctx);
    }

    if (needBuffer) *needBuffer = false;
    return CV_HAL_ERROR_OK;
}

int armpl_hal_dft1D(cvhalDFT *context,
                    const unsigned char *src, unsigned char *dst)
{
    if (!context || !src || !dst)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);
    if (mode == ARMPL_DFT_1D_C2C_FWD || mode == ARMPL_DFT_1D_C2C_INV)
    {
        ArmPL1DC2CFwdContext *ctx = reinterpret_cast<ArmPL1DC2CFwdContext*>(context);
        if (!ctx->plan)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int   len      = ctx->len;
        const float sc       = ctx->scale;
        const bool  no_scale = ctx->no_scale;
        fftwf_execute_dft(
            ctx->plan,
            reinterpret_cast<fftwf_complex*>(const_cast<unsigned char*>(src)),
            reinterpret_cast<fftwf_complex*>(dst));
        if (!no_scale)
        {
            float *df = reinterpret_cast<float*>(dst);
            int i = 0;
            for (; i + 3 < len; i += 4)
            {
                df[i*2]     *= sc;  df[i*2+1]     *= sc;
                df[(i+1)*2] *= sc;  df[(i+1)*2+1] *= sc;
                df[(i+2)*2] *= sc;  df[(i+2)*2+1] *= sc;
                df[(i+3)*2] *= sc;  df[(i+3)*2+1] *= sc;
            }
            for (; i < len; i++) { df[i*2] *= sc;  df[i*2+1] *= sc; }
        }
        return CV_HAL_ERROR_OK;
    }

    if (mode == ARMPL_DFT_1D_C2C_FWD_64 || mode == ARMPL_DFT_1D_C2C_INV_64)
    {
        ArmPL1DC2CFwdContext64 *ctx = reinterpret_cast<ArmPL1DC2CFwdContext64*>(context);
        if (!ctx->plan)
            return CV_HAL_ERROR_NOT_IMPLEMENTED;

        const int    len      = ctx->len;
        const double sc       = ctx->scale;
        const bool   no_scale = ctx->no_scale;
        fftw_execute_dft(
            ctx->plan,
            reinterpret_cast<fftw_complex*>(const_cast<unsigned char*>(src)),
            reinterpret_cast<fftw_complex*>(dst));

        if (!no_scale)
        {
            double *df = reinterpret_cast<double*>(dst);
            int i = 0;
            for (; i + 3 < len; i += 4)
            {
                df[i*2]     *= sc;  df[i*2+1]     *= sc;
                df[(i+1)*2] *= sc;  df[(i+1)*2+1] *= sc;
                df[(i+2)*2] *= sc;  df[(i+2)*2+1] *= sc;
                df[(i+3)*2] *= sc;  df[(i+3)*2+1] *= sc;
            }
            for (; i < len; i++) { df[i*2] *= sc;  df[i*2+1] *= sc; }
        }
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int armpl_hal_dftFree1D(cvhalDFT *context)
{
    if (!context)
        return CV_HAL_ERROR_OK;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);

    if (mode == ARMPL_DFT_1D_C2C_FWD || mode == ARMPL_DFT_1D_C2C_INV)
    {
        ArmPL1DC2CFwdContext *ctx = reinterpret_cast<ArmPL1DC2CFwdContext*>(context);
        if (ctx->plan) fftwf_destroy_plan(ctx->plan);
        delete ctx;
    }
    else if (mode == ARMPL_DFT_1D_C2C_FWD_64 || mode == ARMPL_DFT_1D_C2C_INV_64)
    {
        ArmPL1DC2CFwdContext64 *ctx = reinterpret_cast<ArmPL1DC2CFwdContext64*>(context);
        if (ctx->plan) fftw_destroy_plan(ctx->plan);
        delete ctx;
    }

    return CV_HAL_ERROR_OK;
}

int armplDFTFwd_RToPack(const float* src, float* dst,
                        const void* spec_, unsigned char* /*buf*/)
{
    const ArmplDFTSpec_R_32f* spec = static_cast<const ArmplDFTSpec_R_32f*>(spec_);
    const int n = spec->n;
    fftwf_complex* tmp = reinterpret_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * (n / 2 + 1)));
    if (!tmp) return -1;

    fftwf_execute_dft_r2c(spec->plan, const_cast<float*>(src), tmp);

    dst[0] = tmp[0][0];

    const int num_complex = (n - 1) / 2;
    int dst_idx = 1;
    int i = 1;

#ifdef CV_NEON
    const int simd_end = 1 + (num_complex / 4) * 4;
    for (; i < simd_end; i += 4)
    {
        float32x4x2_t cd = vld2q_f32(reinterpret_cast<const float*>(&tmp[i]));
        vst2q_f32(&dst[dst_idx], cd);
        dst_idx += 8;
    }
#endif
    for (; i <= num_complex; i++)
    {
        dst[dst_idx++] = tmp[i][0];
        dst[dst_idx++] = tmp[i][1];
    }

    if ((n & 1) == 0)
        dst[n - 1] = tmp[n / 2][0];

    fftwf_free(tmp);
    return 0;
}

int armplDFTFwd_RToPack(const double* src, double* dst,
                        const void* spec_, unsigned char* /*buf*/)
{
    const ArmplDFTSpec_R_64f* spec = static_cast<const ArmplDFTSpec_R_64f*>(spec_);
    const int n = spec->n;
    fftw_complex* tmp = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1)));
    if (!tmp) return -1;

    fftw_execute_dft_r2c(spec->plan, const_cast<double*>(src), tmp);

    dst[0] = tmp[0][0];  // DC

    const int num_complex = (n - 1) / 2;
    int dst_idx = 1;
    for (int i = 1; i <= num_complex; i++)
    {
        dst[dst_idx++] = tmp[i][0];
        dst[dst_idx++] = tmp[i][1];
    }

    if ((n & 1) == 0)
        dst[n - 1] = tmp[n / 2][0];

    fftw_free(tmp);
    return 0;
}

int armplDFTInv_PackToR(const float* src, float* dst,
                        const void* spec_, unsigned char* /*buf*/)
{
    const ArmplDFTSpec_R_32f* spec = static_cast<const ArmplDFTSpec_R_32f*>(spec_);
    const int n = spec->n;

    fftwf_complex* tmp = reinterpret_cast<fftwf_complex*>(
        fftwf_malloc(sizeof(fftwf_complex) * (n / 2 + 1)));
    if (!tmp) return -1;

    tmp[0][0] = src[0];
    tmp[0][1] = 0.f;

    const int num_complex = (n - 1) / 2;
    int src_idx = 1;
    for (int i = 1; i <= num_complex; i++)
    {
        tmp[i][0] = src[src_idx++];
        tmp[i][1] = src[src_idx++];
    }

    if ((n & 1) == 0)
    {
        tmp[n / 2][0] = src[n - 1];
        tmp[n / 2][1] = 0.f;
    }

    fftwf_execute_dft_c2r(spec->plan, tmp, dst);
    fftwf_free(tmp);
    return 0;
}

int armplDFTInv_PackToR(const double* src, double* dst,
                        const void* spec_, unsigned char* /*buf*/)
{
    const ArmplDFTSpec_R_64f* spec = static_cast<const ArmplDFTSpec_R_64f*>(spec_);
    const int n = spec->n;
    fftw_complex* tmp = reinterpret_cast<fftw_complex*>(
        fftw_malloc(sizeof(fftw_complex) * (n / 2 + 1)));
    if (!tmp) return -1;

    tmp[0][0] = src[0];
    tmp[0][1] = 0.0;

    const int num_complex = (n - 1) / 2;
    int src_idx = 1;
    for (int i = 1; i <= num_complex; i++)
    {
        tmp[i][0] = src[src_idx++];
        tmp[i][1] = src[src_idx++];
    }

    if ((n & 1) == 0)
    {
        tmp[n / 2][0] = src[n - 1];
        tmp[n / 2][1] = 0.0;
    }

    fftw_execute_dft_c2r(spec->plan, tmp, dst);
    fftw_free(tmp);
    return 0;
}

class DctHalRowInvoker : public cv::ParallelLoopBody
{
public:
    DctHalRowInvoker(const uchar *_src, size_t _src_step,
                           uchar *_dst, size_t _dst_step,
                     int _width, fftwf_plan _plan, bool *_ok)
        : src(_src), src_step(_src_step),
          dst(_dst), dst_step(_dst_step),
          width(_width), plan(_plan), ok(_ok)
    { *ok = true; }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        if (!*ok) return;
        cv::AutoBuffer<float> temp_src(width);
        cv::AutoBuffer<float> temp_dst(width);
        for (int i = range.start; i < range.end; ++i)
        {
            const float *sr = reinterpret_cast<const float*>(src + (size_t)i * src_step);
            float *dr = reinterpret_cast<float*>(dst + (size_t)i * dst_step);
            memcpy(temp_src.data(), sr, (size_t)width * sizeof(float));
            fftwf_execute_r2r(plan, temp_src.data(), temp_dst.data());
            memcpy(dr, temp_dst.data(), (size_t)width * sizeof(float));
        }
    }

private:
    const uchar *src;
    size_t       src_step;
    uchar       *dst;
    size_t       dst_step;
    int          width;
    fftwf_plan   plan;
    bool        *ok;
};

class DctHalColInvoker : public cv::ParallelLoopBody
{
public:
    DctHalColInvoker(const uchar *_src, size_t _src_step,
                           uchar *_dst, size_t _dst_step,
                     int _height, fftwf_plan _plan, bool *_ok)
        : src(_src), src_step(_src_step),
          dst(_dst), dst_step(_dst_step),
          height(_height), plan(_plan), ok(_ok)
    { *ok = true; }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        if (!*ok) return;
        cv::AutoBuffer<float> temp_src(height);
        cv::AutoBuffer<float> temp_dst(height);
        for (int j = range.start; j < range.end; ++j)
        {
            for (int i = 0; i < height; ++i)
            {
                const float *sr = reinterpret_cast<const float*>(src + (size_t)i * src_step);
                temp_src[i] = sr[j];
            }
            fftwf_execute_r2r(plan, temp_src.data(), temp_dst.data());
            for (int i = 0; i < height; ++i)
            {
                float *dr = reinterpret_cast<float*>(dst + (size_t)i * dst_step);
                dr[j] = temp_dst[i];
            }
        }
    }

private:
    const uchar *src;
    size_t       src_step;
    uchar       *dst;
    size_t       dst_step;
    int          height;
    fftwf_plan   plan;
    bool        *ok;
};

int armpl_hal_dctInit2D(cvhalDFT **context,
                        int width, int height,
                        int depth, int flags)
{
    const bool isInverse = (flags & CV_HAL_DFT_INVERSE) != 0;
    const bool isRowWise = (flags & CV_HAL_DFT_ROWS)    != 0;

    if (depth != CV_32F)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const fftw_r2r_kind kind = isInverse ? FFTW_REDFT01
                                         : FFTW_REDFT10;

    if (!isRowWise)
    {
        float *row_buf = (float*)fftwf_malloc(sizeof(float) * (size_t)width * height);
        if (!row_buf) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        float *tmp_r = (float*)fftwf_malloc(sizeof(float) * width);
        float *tmp_c = (float*)fftwf_malloc(sizeof(float) * height);
        if (!tmp_r || !tmp_c)
        {
            fftwf_free(row_buf);
            if (tmp_r) fftwf_free(tmp_r);
            if (tmp_c) fftwf_free(tmp_c);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        fftwf_plan prow = fftwf_plan_r2r_1d(width,  tmp_r, tmp_r, kind, FFTW_MEASURE);
        fftwf_plan pcol = fftwf_plan_r2r_1d(height, tmp_c, tmp_c, kind, FFTW_MEASURE);
        fftwf_free(tmp_r);
        fftwf_free(tmp_c);

        if (!prow || !pcol)
        {
            if (prow) fftwf_destroy_plan(prow);
            if (pcol) fftwf_destroy_plan(pcol);
            fftwf_free(row_buf);
            return CV_HAL_ERROR_NOT_IMPLEMENTED;
        }

        ArmPLDCT2DContext *ctx = new ArmPLDCT2DContext();
        ctx->width    = width;
        ctx->height   = height;
        ctx->inv      = isInverse;
        ctx->plan_fwd = prow;
        ctx->plan_inv = pcol;
        ctx->buf      = row_buf;

        const float w = (float)width;
        const float h = (float)height;
        const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
        const float base = 1.0f / (2.0f * std::sqrt(w * h));

        ctx->scale_dc   = base * inv_sqrt2 * inv_sqrt2;
        ctx->scale_axis = base * inv_sqrt2;
        ctx->scale_rest = base;

        *context = reinterpret_cast<cvhalDFT*>(ctx);
        return CV_HAL_ERROR_OK;
    }

    float *fftw_buf = (float*)fftwf_malloc(sizeof(float) * width);
    if (!fftw_buf) return CV_HAL_ERROR_NOT_IMPLEMENTED;

    fftwf_plan pf = fftwf_plan_r2r_1d(width, fftw_buf, fftw_buf, kind, FFTW_MEASURE);
    if (!pf)
    {
        fftwf_free(fftw_buf);
        return CV_HAL_ERROR_NOT_IMPLEMENTED;
    }

    ArmPLDCTRowContext *ctx = new ArmPLDCTRowContext();
    ctx->width    = width;
    ctx->height   = height;
    ctx->inv      = isInverse;
    ctx->plan_fwd = pf;
    ctx->fftw_buf = fftw_buf;

    const float w = (float)width;
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    const float base = 1.0f / std::sqrt(2.0f * w);
    ctx->scale_dc   = base * inv_sqrt2;
    ctx->scale_rest = base;

    *context = reinterpret_cast<cvhalDFT*>(ctx);
    return CV_HAL_ERROR_OK;
}

int armpl_hal_dct2D(cvhalDFT *context,
                    const unsigned char *src_data, size_t src_step,
                          unsigned char *dst_data, size_t dst_step)
{
    if (!context || !src_data || !dst_data)
        return CV_HAL_ERROR_NOT_IMPLEMENTED;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);

    if (mode == ARMPL_DCT_2D)
    {
        ArmPLDCT2DContext *ctx = reinterpret_cast<ArmPLDCT2DContext*>(context);
        const int W = ctx->width;
        const int H = ctx->height;
        float *buf  = ctx->buf;

        const float sc_dc   = ctx->scale_dc;
        const float sc_axis = ctx->scale_axis;
        const float sc_rest = ctx->scale_rest;

        if (!ctx->inv)
        {
            bool ok_row = true;
            cv::parallel_for_(cv::Range(0, H),
                DctHalRowInvoker(src_data, src_step,
                                 reinterpret_cast<uchar*>(buf),
                                 (size_t)W * sizeof(float),
                                 W, ctx->plan_fwd, &ok_row),
                H / (double)(1 << 4));
            if (!ok_row) return CV_HAL_ERROR_NOT_IMPLEMENTED;

            bool ok_col = true;
            cv::parallel_for_(cv::Range(0, W),
                DctHalColInvoker(reinterpret_cast<const uchar*>(buf),
                                 (size_t)W * sizeof(float),
                                 dst_data, dst_step,
                                 H, ctx->plan_inv, &ok_col),
                W / (double)(1 << 4));
            if (!ok_col) return CV_HAL_ERROR_NOT_IMPLEMENTED;

            for (int i = 0; i < H; ++i)
            {
                float *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);
                const float sc_row = (i == 0) ? sc_axis : sc_rest;
                dr[0] *= (i == 0) ? sc_dc : sc_axis;
                for (int j = 1; j < W; ++j)
                    dr[j] *= sc_row;
            }
            return CV_HAL_ERROR_OK;
        }

        const float inv_sc_dc   = 1.0f / sc_dc;
        const float inv_sc_axis = 1.0f / sc_axis;
        const float inv_sc_rest = 1.0f / sc_rest;
        const float postscale   = 1.0f / (4.0f * static_cast<float>(W) * static_cast<float>(H));

        for (int i = 0; i < H; ++i)
        {
            const float *sr = reinterpret_cast<const float*>(src_data + (size_t)i * src_step);
            float       *br = buf + (size_t)i * W;
            const float  row_inv_sc = (i == 0) ? inv_sc_axis : inv_sc_rest;
            br[0] = sr[0] * ((i == 0) ? inv_sc_dc : inv_sc_axis);
            for (int j = 1; j < W; ++j)
                br[j] = sr[j] * row_inv_sc;
        }

        bool ok_col = true;
        cv::parallel_for_(cv::Range(0, W),
            DctHalColInvoker(reinterpret_cast<const uchar*>(buf),
                             (size_t)W * sizeof(float),
                             dst_data, dst_step,
                             H, ctx->plan_inv, &ok_col),
            W / (double)(1 << 4));
        if (!ok_col) return CV_HAL_ERROR_NOT_IMPLEMENTED;
        bool ok_row = true;
        cv::parallel_for_(cv::Range(0, H),
            DctHalRowInvoker(dst_data, dst_step,
                             dst_data, dst_step,
                             W, ctx->plan_fwd, &ok_row),
            H / (double)(1 << 4));
        if (!ok_row) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        for (int i = 0; i < H; ++i)
        {
            float *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);
            for (int j = 0; j < W; ++j)
                dr[j] *= postscale;
        }
        return CV_HAL_ERROR_OK;
    }

    if (mode == ARMPL_DCT_ROW)
    {
        ArmPLDCTRowContext *ctx = reinterpret_cast<ArmPLDCTRowContext*>(context);
        const int W = ctx->width;
        const int H = ctx->height;
        const float sc_dc   = ctx->scale_dc;
        const float sc_rest = ctx->scale_rest;

        if (!ctx->inv)
        {
            bool ok_row = true;
            cv::parallel_for_(
                cv::Range(0, H),
                DctHalRowInvoker(src_data, src_step,
                                 dst_data, dst_step,
                                 W, ctx->plan_fwd, &ok_row),
                H / (double)(1 << 4));
            if (!ok_row) return CV_HAL_ERROR_NOT_IMPLEMENTED;

            for (int i = 0; i < H; ++i)
            {
                float *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);
                dr[0] *= sc_dc;
                for (int j = 1; j < W; ++j)
                    dr[j] *= sc_rest;
            }
            return CV_HAL_ERROR_OK;
        }

        const float inv_sc_dc   = 1.0f / sc_dc;
        const float inv_sc_rest = 1.0f / sc_rest;
        const float postscale   = 1.0f / (2.0f * static_cast<float>(W));

        for (int i = 0; i < H; ++i)
        {
            const float *sr = reinterpret_cast<const float*>(src_data + (size_t)i * src_step);
            float       *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);
            dr[0] = sr[0] * inv_sc_dc;
            for (int j = 1; j < W; ++j)
                dr[j] = sr[j] * inv_sc_rest;
        }

        bool ok_row = true;
        cv::parallel_for_(
            cv::Range(0, H),
            DctHalRowInvoker(dst_data, dst_step,
                             dst_data, dst_step,
                             W, ctx->plan_fwd, &ok_row),
            H / (double)(1 << 4));
        if (!ok_row) return CV_HAL_ERROR_NOT_IMPLEMENTED;

        for (int i = 0; i < H; ++i)
        {
            float *dr = reinterpret_cast<float*>(dst_data + (size_t)i * dst_step);
            for (int j = 0; j < W; ++j)
                dr[j] *= postscale;
        }
        return CV_HAL_ERROR_OK;
    }

    return CV_HAL_ERROR_NOT_IMPLEMENTED;
}

int armpl_hal_dctFree2D(cvhalDFT *context)
{
    if (!context) return CV_HAL_ERROR_OK;

    const ArmPLDFTMode mode = *reinterpret_cast<const ArmPLDFTMode*>(context);

    if (mode == ARMPL_DCT_2D)
    {
        ArmPLDCT2DContext *ctx = reinterpret_cast<ArmPLDCT2DContext*>(context);
        if (ctx->plan_fwd) fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->plan_inv) fftwf_destroy_plan(ctx->plan_inv);
        if (ctx->buf)      fftwf_free(ctx->buf);
        delete ctx;
    }
    else if (mode == ARMPL_DCT_ROW)
    {
        ArmPLDCTRowContext *ctx = reinterpret_cast<ArmPLDCTRowContext*>(context);
        if (ctx->plan_fwd) fftwf_destroy_plan(ctx->plan_fwd);
        if (ctx->fftw_buf) fftwf_free(ctx->fftw_buf);
        delete ctx;
    }

    return CV_HAL_ERROR_OK;
}

#endif // HAVE_ARMPL
