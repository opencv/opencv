noinst_LTLIBRARIES = libwebputils.la

if BUILD_LIBWEBPDECODER
  noinst_LTLIBRARIES += libwebputilsdecode.la
endif

common_HEADERS = ../webp/types.h
commondir = $(includedir)/webp

noinst_HEADERS =
noinst_HEADERS += ../dsp/dsp.h
noinst_HEADERS += ../webp/decode.h
noinst_HEADERS += ../webp/encode.h
noinst_HEADERS += ../webp/format_constants.h

COMMON_SOURCES =
COMMON_SOURCES += bit_reader_utils.c
COMMON_SOURCES += bit_reader_utils.h
COMMON_SOURCES += bit_reader_inl_utils.h
COMMON_SOURCES += color_cache_utils.c
COMMON_SOURCES += color_cache_utils.h
COMMON_SOURCES += endian_inl_utils.h
COMMON_SOURCES += filters_utils.c
COMMON_SOURCES += filters_utils.h
COMMON_SOURCES += huffman_utils.c
COMMON_SOURCES += huffman_utils.h
COMMON_SOURCES += quant_levels_dec_utils.c
COMMON_SOURCES += quant_levels_dec_utils.h
COMMON_SOURCES += rescaler_utils.c
COMMON_SOURCES += rescaler_utils.h
COMMON_SOURCES += random_utils.c
COMMON_SOURCES += random_utils.h
COMMON_SOURCES += thread_utils.c
COMMON_SOURCES += thread_utils.h
COMMON_SOURCES += utils.c
COMMON_SOURCES += utils.h

ENC_SOURCES =
ENC_SOURCES += bit_writer_utils.c
ENC_SOURCES += bit_writer_utils.h
ENC_SOURCES += huffman_encode_utils.c
ENC_SOURCES += huffman_encode_utils.h
ENC_SOURCES += quant_levels_utils.c
ENC_SOURCES += quant_levels_utils.h

libwebputils_la_SOURCES = $(COMMON_SOURCES) $(ENC_SOURCES)

if BUILD_LIBWEBPDECODER
  libwebputilsdecode_la_SOURCES = $(COMMON_SOURCES)
endif
