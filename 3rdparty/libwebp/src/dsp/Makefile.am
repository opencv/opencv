AM_CPPFLAGS += -I$(top_builddir) -I$(top_srcdir)
noinst_LTLIBRARIES =
noinst_LTLIBRARIES += libwebpdsp.la
noinst_LTLIBRARIES += libwebpdsp_sse2.la
noinst_LTLIBRARIES += libwebpdspdecode_sse2.la
noinst_LTLIBRARIES += libwebpdsp_sse41.la
noinst_LTLIBRARIES += libwebpdspdecode_sse41.la
noinst_LTLIBRARIES += libwebpdsp_neon.la
noinst_LTLIBRARIES += libwebpdspdecode_neon.la
noinst_LTLIBRARIES += libwebpdsp_msa.la
noinst_LTLIBRARIES += libwebpdspdecode_msa.la
noinst_LTLIBRARIES += libwebpdsp_mips32.la
noinst_LTLIBRARIES += libwebpdspdecode_mips32.la
noinst_LTLIBRARIES += libwebpdsp_mips_dsp_r2.la
noinst_LTLIBRARIES += libwebpdspdecode_mips_dsp_r2.la

if BUILD_LIBWEBPDECODER
  noinst_LTLIBRARIES += libwebpdspdecode.la
endif

common_HEADERS = ../webp/types.h
commondir = $(includedir)/webp

COMMON_SOURCES =
COMMON_SOURCES += alpha_processing.c
COMMON_SOURCES += cpu.c
COMMON_SOURCES += cpu.h
COMMON_SOURCES += dec.c
COMMON_SOURCES += dec_clip_tables.c
COMMON_SOURCES += dsp.h
COMMON_SOURCES += filters.c
COMMON_SOURCES += lossless.c
COMMON_SOURCES += lossless.h
COMMON_SOURCES += lossless_common.h
COMMON_SOURCES += rescaler.c
COMMON_SOURCES += upsampling.c
COMMON_SOURCES += yuv.c
COMMON_SOURCES += yuv.h

ENC_SOURCES =
ENC_SOURCES += cost.c
ENC_SOURCES += enc.c
ENC_SOURCES += lossless_enc.c
ENC_SOURCES += quant.h
ENC_SOURCES += ssim.c

libwebpdspdecode_sse41_la_SOURCES =
libwebpdspdecode_sse41_la_SOURCES += alpha_processing_sse41.c
libwebpdspdecode_sse41_la_SOURCES += dec_sse41.c
libwebpdspdecode_sse41_la_SOURCES += lossless_sse41.c
libwebpdspdecode_sse41_la_SOURCES += upsampling_sse41.c
libwebpdspdecode_sse41_la_SOURCES += yuv_sse41.c
libwebpdspdecode_sse41_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
libwebpdspdecode_sse41_la_CFLAGS = $(AM_CFLAGS) $(SSE41_FLAGS)

libwebpdspdecode_sse2_la_SOURCES =
libwebpdspdecode_sse2_la_SOURCES += alpha_processing_sse2.c
libwebpdspdecode_sse2_la_SOURCES += common_sse2.h
libwebpdspdecode_sse2_la_SOURCES += dec_sse2.c
libwebpdspdecode_sse2_la_SOURCES += filters_sse2.c
libwebpdspdecode_sse2_la_SOURCES += lossless_sse2.c
libwebpdspdecode_sse2_la_SOURCES += rescaler_sse2.c
libwebpdspdecode_sse2_la_SOURCES += upsampling_sse2.c
libwebpdspdecode_sse2_la_SOURCES += yuv_sse2.c
libwebpdspdecode_sse2_la_CPPFLAGS = $(libwebpdsp_sse2_la_CPPFLAGS)
libwebpdspdecode_sse2_la_CFLAGS = $(libwebpdsp_sse2_la_CFLAGS)

libwebpdspdecode_neon_la_SOURCES =
libwebpdspdecode_neon_la_SOURCES += alpha_processing_neon.c
libwebpdspdecode_neon_la_SOURCES += dec_neon.c
libwebpdspdecode_neon_la_SOURCES += filters_neon.c
libwebpdspdecode_neon_la_SOURCES += lossless_neon.c
libwebpdspdecode_neon_la_SOURCES += neon.h
libwebpdspdecode_neon_la_SOURCES += rescaler_neon.c
libwebpdspdecode_neon_la_SOURCES += upsampling_neon.c
libwebpdspdecode_neon_la_SOURCES += yuv_neon.c
libwebpdspdecode_neon_la_CPPFLAGS = $(libwebpdsp_neon_la_CPPFLAGS)
libwebpdspdecode_neon_la_CFLAGS = $(libwebpdsp_neon_la_CFLAGS)

libwebpdspdecode_msa_la_SOURCES =
libwebpdspdecode_msa_la_SOURCES += dec_msa.c
libwebpdspdecode_msa_la_SOURCES += filters_msa.c
libwebpdspdecode_msa_la_SOURCES += lossless_msa.c
libwebpdspdecode_msa_la_SOURCES += msa_macro.h
libwebpdspdecode_msa_la_SOURCES += rescaler_msa.c
libwebpdspdecode_msa_la_SOURCES += upsampling_msa.c
libwebpdspdecode_msa_la_CPPFLAGS = $(libwebpdsp_msa_la_CPPFLAGS)
libwebpdspdecode_msa_la_CFLAGS = $(libwebpdsp_msa_la_CFLAGS)

libwebpdspdecode_mips32_la_SOURCES =
libwebpdspdecode_mips32_la_SOURCES += dec_mips32.c
libwebpdspdecode_mips32_la_SOURCES += mips_macro.h
libwebpdspdecode_mips32_la_SOURCES += rescaler_mips32.c
libwebpdspdecode_mips32_la_SOURCES += yuv_mips32.c
libwebpdspdecode_mips32_la_CPPFLAGS = $(libwebpdsp_mips32_la_CPPFLAGS)
libwebpdspdecode_mips32_la_CFLAGS = $(libwebpdsp_mips32_la_CFLAGS)

libwebpdspdecode_mips_dsp_r2_la_SOURCES =
libwebpdspdecode_mips_dsp_r2_la_SOURCES += alpha_processing_mips_dsp_r2.c
libwebpdspdecode_mips_dsp_r2_la_SOURCES += dec_mips_dsp_r2.c
libwebpdspdecode_mips_dsp_r2_la_SOURCES += filters_mips_dsp_r2.c
libwebpdspdecode_mips_dsp_r2_la_SOURCES += lossless_mips_dsp_r2.c
libwebpdspdecode_mips_dsp_r2_la_SOURCES += mips_macro.h
libwebpdspdecode_mips_dsp_r2_la_SOURCES += rescaler_mips_dsp_r2.c
libwebpdspdecode_mips_dsp_r2_la_SOURCES += upsampling_mips_dsp_r2.c
libwebpdspdecode_mips_dsp_r2_la_SOURCES += yuv_mips_dsp_r2.c
libwebpdspdecode_mips_dsp_r2_la_CPPFLAGS = $(libwebpdsp_mips_dsp_r2_la_CPPFLAGS)
libwebpdspdecode_mips_dsp_r2_la_CFLAGS = $(libwebpdsp_mips_dsp_r2_la_CFLAGS)

libwebpdsp_sse2_la_SOURCES =
libwebpdsp_sse2_la_SOURCES += cost_sse2.c
libwebpdsp_sse2_la_SOURCES += enc_sse2.c
libwebpdsp_sse2_la_SOURCES += lossless_enc_sse2.c
libwebpdsp_sse2_la_SOURCES += ssim_sse2.c
libwebpdsp_sse2_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
libwebpdsp_sse2_la_CFLAGS = $(AM_CFLAGS) $(SSE2_FLAGS)
libwebpdsp_sse2_la_LIBADD = libwebpdspdecode_sse2.la

libwebpdsp_sse41_la_SOURCES =
libwebpdsp_sse41_la_SOURCES += enc_sse41.c
libwebpdsp_sse41_la_SOURCES += lossless_enc_sse41.c
libwebpdsp_sse41_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
libwebpdsp_sse41_la_CFLAGS = $(AM_CFLAGS) $(SSE41_FLAGS)
libwebpdsp_sse41_la_LIBADD = libwebpdspdecode_sse41.la

libwebpdsp_neon_la_SOURCES =
libwebpdsp_neon_la_SOURCES += cost_neon.c
libwebpdsp_neon_la_SOURCES += enc_neon.c
libwebpdsp_neon_la_SOURCES += lossless_enc_neon.c
libwebpdsp_neon_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
libwebpdsp_neon_la_CFLAGS = $(AM_CFLAGS) $(NEON_FLAGS)
libwebpdsp_neon_la_LIBADD = libwebpdspdecode_neon.la

libwebpdsp_msa_la_SOURCES =
libwebpdsp_msa_la_SOURCES += enc_msa.c
libwebpdsp_msa_la_SOURCES += lossless_enc_msa.c
libwebpdsp_msa_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
libwebpdsp_msa_la_CFLAGS = $(AM_CFLAGS)
libwebpdsp_msa_la_LIBADD = libwebpdspdecode_msa.la

libwebpdsp_mips32_la_SOURCES =
libwebpdsp_mips32_la_SOURCES += cost_mips32.c
libwebpdsp_mips32_la_SOURCES += enc_mips32.c
libwebpdsp_mips32_la_SOURCES += lossless_enc_mips32.c
libwebpdsp_mips32_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
libwebpdsp_mips32_la_CFLAGS = $(AM_CFLAGS)
libwebpdsp_mips32_la_LIBADD = libwebpdspdecode_mips32.la

libwebpdsp_mips_dsp_r2_la_SOURCES =
libwebpdsp_mips_dsp_r2_la_SOURCES += cost_mips_dsp_r2.c
libwebpdsp_mips_dsp_r2_la_SOURCES += enc_mips_dsp_r2.c
libwebpdsp_mips_dsp_r2_la_SOURCES += lossless_enc_mips_dsp_r2.c
libwebpdsp_mips_dsp_r2_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
libwebpdsp_mips_dsp_r2_la_CFLAGS = $(AM_CFLAGS)
libwebpdsp_mips_dsp_r2_la_LIBADD = libwebpdspdecode_mips_dsp_r2.la

libwebpdsp_la_SOURCES = $(COMMON_SOURCES) $(ENC_SOURCES)

noinst_HEADERS =
noinst_HEADERS += ../dec/vp8_dec.h
noinst_HEADERS += ../webp/decode.h

libwebpdsp_la_CPPFLAGS =
libwebpdsp_la_CPPFLAGS += $(AM_CPPFLAGS)
libwebpdsp_la_CPPFLAGS += $(USE_SWAP_16BIT_CSP)
libwebpdsp_la_LDFLAGS = -lm
libwebpdsp_la_LIBADD =
libwebpdsp_la_LIBADD += libwebpdsp_sse2.la
libwebpdsp_la_LIBADD += libwebpdsp_sse41.la
libwebpdsp_la_LIBADD += libwebpdsp_neon.la
libwebpdsp_la_LIBADD += libwebpdsp_msa.la
libwebpdsp_la_LIBADD += libwebpdsp_mips32.la
libwebpdsp_la_LIBADD += libwebpdsp_mips_dsp_r2.la

if BUILD_LIBWEBPDECODER
  libwebpdspdecode_la_SOURCES = $(COMMON_SOURCES)

  libwebpdspdecode_la_CPPFLAGS = $(libwebpdsp_la_CPPFLAGS)
  libwebpdspdecode_la_LDFLAGS = $(libwebpdsp_la_LDFLAGS)
  libwebpdspdecode_la_LIBADD =
  libwebpdspdecode_la_LIBADD += libwebpdspdecode_sse2.la
  libwebpdspdecode_la_LIBADD += libwebpdspdecode_sse41.la
  libwebpdspdecode_la_LIBADD += libwebpdspdecode_neon.la
  libwebpdspdecode_la_LIBADD += libwebpdspdecode_msa.la
  libwebpdspdecode_la_LIBADD += libwebpdspdecode_mips32.la
  libwebpdspdecode_la_LIBADD += libwebpdspdecode_mips_dsp_r2.la
endif
