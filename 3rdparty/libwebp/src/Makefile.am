# The mux and demux libraries depend on libwebp, thus the '.' to force
# the build order so it's available to them.
SUBDIRS = dec enc dsp utils .
if BUILD_MUX
  SUBDIRS += mux
endif
if BUILD_DEMUX
  SUBDIRS += demux
endif

lib_LTLIBRARIES = libwebp.la

if BUILD_LIBWEBPDECODER
  lib_LTLIBRARIES += libwebpdecoder.la
endif

common_HEADERS =
common_HEADERS += webp/decode.h
common_HEADERS += webp/types.h
commondir = $(includedir)/webp

libwebp_la_SOURCES =
libwebpinclude_HEADERS =
libwebpinclude_HEADERS += webp/encode.h

noinst_HEADERS =
noinst_HEADERS += webp/format_constants.h

libwebp_la_LIBADD =
libwebp_la_LIBADD += dec/libwebpdecode.la
libwebp_la_LIBADD += dsp/libwebpdsp.la
libwebp_la_LIBADD += enc/libwebpencode.la
libwebp_la_LIBADD += utils/libwebputils.la

# Use '-no-undefined' to declare that libwebp does not depend on any libraries
# other than the ones listed on the command line, i.e., after linking, it will
# not have unresolved symbols. Some platforms (Windows among them) require all
# symbols in shared libraries to be resolved at library creation.
libwebp_la_LDFLAGS = -no-undefined -version-info 8:8:1
libwebpincludedir = $(includedir)/webp
pkgconfig_DATA = libwebp.pc

if BUILD_LIBWEBPDECODER
  libwebpdecoder_la_SOURCES =

  libwebpdecoder_la_LIBADD =
  libwebpdecoder_la_LIBADD += dec/libwebpdecode.la
  libwebpdecoder_la_LIBADD += dsp/libwebpdspdecode.la
  libwebpdecoder_la_LIBADD += utils/libwebputilsdecode.la

  libwebpdecoder_la_LDFLAGS = -no-undefined -version-info 4:8:1
  pkgconfig_DATA += libwebpdecoder.pc
endif

${pkgconfig_DATA}: ${top_builddir}/config.status
